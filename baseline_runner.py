"""baseline_runner.py

This module runs the v1 baseline experiment for the memory_lab project.

The design stays intentionally plain:
- generate synthetic recall episodes
- apply a simple memory-selection policy
- retrieve by exact structured match
- summarize accuracy, misses, confusion, and storage size
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

from stream_generator import CivicRecord, RecallEpisode, generate_recall_episodes, load_record_pool

from data_types import (
    DualAxisFieldScore,
    EpisodeResult,
    LadderRow,
    RouteChooserRow,
    RouteComparisonResult,
    RouteExplanationRow,
    RouteStressRow,
    RoutingResult,
    SourceComparisonRow,
    StubSchema,
    WhisperFieldScore,
    WhisperStep,
)
from memory_policies import (
    SaveAllPolicy,
    RuleBasedSaliencePolicy,
    TieredMemoryPolicy,
    StubMemoryPolicy,
    STUB_SCHEMAS,
    estimate_stub_field_count,
)
from retrieval import (
    record_identity,
    query_identity,
    exact_structured_lookup,
    exact_structured_retrieval,
    exact_tiered_retrieval,
    stub_match_score,
    exact_stub_retrieval,
    full_records_for_policy,
)
from routing import (
    flat_lookup_with_candidates,
    identity_only_lookup_with_candidates,
    identity_plus_therapy_lookup_with_candidates,
    routed_lookup_with_candidates,
    two_stage_semantic_lookup_with_candidates,
    is_route_confusion,
    build_addresses_for_records,
    evaluate_routing_episode,
    evaluate_route_comparison_episode,
    summarize_routing_results,
    summarize_route_comparison_results,
)
from whisper import (
    WHISPER_SEMANTIC_FIELDS,
    WHISPER_PROVENANCE_FIELDS,
    WHISPER_RAW_CANDIDATE_FIELDS,
    record_field_value,
    unique_records_from_episodes,
    ambiguous_identity_neighborhoods,
    remaining_ambiguity_pairs,
    score_whisper_candidate_fields,
    infer_discriminator_ladder,
    stub_policy_name_for_single_field,
    build_dual_axis_field_scores,
    whisper_field_category,
)
from chooser import (
    explain_route_choice,
    build_route_chooser_row,
    build_route_stress_row,
    run_route_stress_experiment,
    run_route_chooser_experiment,
)


# This helper returns the stub schema metadata for a stub policy name.
# What this does:
# - Maps a printed policy name back to the schema that generated it.
# Why this exists:
# - The ladder report needs to know which discriminator fields and costs belong to each policy.
# What assumption it is making:
# - Stub policy names remain aligned with the schema naming convention.
def find_stub_schema_for_policy_name(policy_name: str) -> Optional[StubSchema]:
    for schema in STUB_SCHEMAS:
        if policy_name == f"StubMemoryPolicy[{schema.name}]":
            return schema
    return None


# This helper builds the single-field discriminator ranking rows.
# What this does:
# - Compares each single added field against the minimal-identity baseline.
# Why this exists:
# - The user asked for a formal ranking of cheapest useful discriminators.
# What assumption it is making:
# - Minimal identity is the correct baseline for judging added-field value.
def build_single_field_ladder_rows(policy_results: Dict[str, List[EpisodeResult]]) -> List[LadderRow]:
    baseline_policy_name = "StubMemoryPolicy[minimal_identity]"
    baseline_results = policy_results[baseline_policy_name]
    baseline_confusion_rate = summarize_results(baseline_results)["confusion_rate"]

    rows: List[LadderRow] = []
    for policy_name, results in policy_results.items():
        schema = find_stub_schema_for_policy_name(policy_name)
        if schema is None:
            continue
        if len(schema.discriminator_fields) != 1:
            continue

        summary = summarize_results(results)
        confusion_rate = summary["confusion_rate"]
        confusion_reduction = baseline_confusion_rate - confusion_rate
        added_field_cost = len(schema.discriminator_fields)
        confusion_reduction_per_cost = confusion_reduction / added_field_cost if added_field_cost else 0.0
        rows.append(
            LadderRow(
                policy_name=policy_name,
                discriminator_fields=schema.discriminator_fields,
                confusion_rate=confusion_rate,
                confusion_reduction=confusion_reduction,
                added_field_cost=added_field_cost,
                confusion_reduction_per_cost=confusion_reduction_per_cost,
            )
        )

    return sorted(
        rows,
        key=lambda row: (row.confusion_reduction_per_cost, row.confusion_reduction, -row.added_field_cost),
        reverse=True,
    )


# This helper builds pair-combination rows anchored on the top single field.
# What this does:
# - Evaluates two-field discriminator schemas that extend the best single added field.
# Why this exists:
# - The ladder should show the next fallback step after the top single field.
# What assumption it is making:
# - Pair schemas with the winning first field are the right next tradeoff to inspect.
def build_pair_ladder_rows(policy_results: Dict[str, List[EpisodeResult]], top_field: str) -> List[LadderRow]:
    baseline_policy_name = "StubMemoryPolicy[minimal_identity]"
    baseline_results = policy_results[baseline_policy_name]
    baseline_confusion_rate = summarize_results(baseline_results)["confusion_rate"]

    rows: List[LadderRow] = []
    for policy_name, results in policy_results.items():
        schema = find_stub_schema_for_policy_name(policy_name)
        if schema is None:
            continue
        if len(schema.discriminator_fields) != 2:
            continue
        if top_field not in schema.discriminator_fields:
            continue

        summary = summarize_results(results)
        confusion_rate = summary["confusion_rate"]
        confusion_reduction = baseline_confusion_rate - confusion_rate
        added_field_cost = len(schema.discriminator_fields)
        confusion_reduction_per_cost = confusion_reduction / added_field_cost if added_field_cost else 0.0
        rows.append(
            LadderRow(
                policy_name=policy_name,
                discriminator_fields=schema.discriminator_fields,
                confusion_rate=confusion_rate,
                confusion_reduction=confusion_reduction,
                added_field_cost=added_field_cost,
                confusion_reduction_per_cost=confusion_reduction_per_cost,
            )
        )

    return sorted(
        rows,
        key=lambda row: (row.confusion_reduction_per_cost, row.confusion_reduction, -row.added_field_cost),
        reverse=True,
    )


# This function evaluates one policy on one episode.
# What this does:
# - Applies memory selection, runs retrieval, and labels the outcome as correct, miss, or confusion.
# Why this exists:
# - A single explicit evaluation path keeps the baseline logic easy to understand and debug.
# What assumption it is making:
# - A wrong non-empty answer should count as confusion, while no answer should count as a miss.
def evaluate_episode(policy, episode: RecallEpisode) -> EpisodeResult:
    if isinstance(policy, TieredMemoryPolicy):
        memory_store = policy.build_memory_store(episode.stream_records)
        predicted_answer = exact_tiered_retrieval(memory_store=memory_store, query=episode.query)
        was_stub_hit = False
        matched_record_id = episode.target_record.record_id if predicted_answer == episode.query.answer else None
        full_records_stored = len(memory_store.durable_records)
        stubs_stored = 0
    elif isinstance(policy, StubMemoryPolicy):
        memory_store = policy.build_memory_store(episode.stream_records)
        predicted_answer, was_stub_hit, matched_record_id = exact_stub_retrieval(
            memory_store=memory_store,
            query=episode.query,
        )
        full_records_stored = len(memory_store.durable_records)
        stubs_stored = len(memory_store.stubs)
        stub_fields_retained = sum(estimate_stub_field_count(stub) for stub in memory_store.stubs)
    else:
        stored_records = policy.select_records(episode.stream_records)
        predicted_answer = exact_structured_retrieval(stored_records=stored_records, query=episode.query)
        was_stub_hit = False
        matched_record_id = episode.target_record.record_id if predicted_answer == episode.query.answer else None
        full_records_stored = len(stored_records)
        stubs_stored = 0
        stub_fields_retained = 0

    was_correct = predicted_answer == episode.query.answer
    was_full_hit = was_correct
    was_miss = predicted_answer is None
    was_confusion = False

    if predicted_answer is not None:
        if was_stub_hit:
            was_confusion = matched_record_id != episode.target_record.record_id
        else:
            was_confusion = predicted_answer != episode.query.answer

    if not isinstance(policy, StubMemoryPolicy):
        stub_fields_retained = 0

    return EpisodeResult(
        policy_name=policy.name,
        task_type=episode.task_type,
        episode_id=episode.episode_id,
        target_position=episode.target_position,
        expected_answer=episode.query.answer,
        predicted_answer=predicted_answer,
        was_correct=was_correct,
        was_full_hit=was_full_hit,
        was_stub_hit=was_stub_hit,
        was_miss=was_miss,
        was_confusion=was_confusion,
        full_records_stored=full_records_stored,
        stubs_stored=stubs_stored,
        stub_fields_retained=stub_fields_retained,
    )


# This function aggregates episode-level results into a small summary.
# What this does:
# - Calculates the simple metrics we want to inspect in the first research pass.
# Why this exists:
# - We want a readable output that tells us whether a policy is preserving useful facts
#   while keeping storage size under control.
# What assumption it is making:
# - Accuracy, misses, confusion count, and average storage size are enough for the first baseline.
def summarize_results(results: List[EpisodeResult]) -> Dict[str, float]:
    total = len(results)
    full_hits = sum(result.was_full_hit for result in results)
    stub_hits = sum(result.was_stub_hit for result in results)
    misses = sum(result.was_miss for result in results)
    confusions = sum(result.was_confusion for result in results)
    average_full_records = sum(result.full_records_stored for result in results) / total if total else 0.0
    average_stubs = sum(result.stubs_stored for result in results) / total if total else 0.0
    average_stub_fields_retained = sum(result.stub_fields_retained for result in results) / total if total else 0.0
    total_stubs = sum(result.stubs_stored for result in results)
    total_stub_fields = sum(result.stub_fields_retained for result in results)
    average_fields_per_stub = total_stub_fields / total_stubs if total_stubs else 0.0
    retrieval_success_rate = (full_hits + stub_hits) / total if total else 0.0
    retrieval_success_per_stub_field = (
        retrieval_success_rate / average_stub_fields_retained if average_stub_fields_retained else 0.0
    )

    return {
        "episodes": total,
        "accuracy": full_hits / total if total else 0.0,
        "full_hit_rate": full_hits / total if total else 0.0,
        "stub_hit_rate": stub_hits / total if total else 0.0,
        "miss_rate": misses / total if total else 0.0,
        "confusion_rate": confusions / total if total else 0.0,
        "average_full_records_stored": average_full_records,
        "average_stubs_stored": average_stubs,
        "average_stub_fields_retained": average_stub_fields_retained,
        "average_fields_per_stub": average_fields_per_stub,
        "retrieval_success_per_stub_field": retrieval_success_per_stub_field,
    }


# This helper counts how often each target position was used.
# What this does:
# - Produces a small early/middle/late breakdown for inspection.
# Why this exists:
# - The new late-relevance task varies target placement, so the printed output should show it.
# What assumption it is making:
# - A simple count by position is enough for v1; we do not need deeper order analysis yet.
def summarize_positions(results: List[EpisodeResult]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for result in results:
        counts[result.target_position] = counts.get(result.target_position, 0) + 1
    return counts


# This helper calculates metrics separately for each target position.
# What this does:
# - Produces a small per-position summary so we can inspect whether order affects results.
# Why this exists:
# - The new requirements ask for summary breakdowns by early, middle, and late placement.
# What assumption it is making:
# - Position-specific accuracy and miss rates are enough for the next round of interpretation.
def summarize_by_target_position(results: List[EpisodeResult]) -> Dict[str, Dict[str, float]]:
    grouped_results: Dict[str, List[EpisodeResult]] = {}

    for result in results:
        grouped_results.setdefault(result.target_position, []).append(result)

    summaries: Dict[str, Dict[str, float]] = {}
    for target_position in ["early", "middle", "late", "mixed"]:
        if target_position in grouped_results:
            summaries[target_position] = summarize_results(grouped_results[target_position])

    return summaries


# This helper prints a compact but readable summary for one policy and one task type.
# What this does:
# - Displays the main metrics with consistent formatting.
# Why this exists:
# - Early research work moves faster when terminal output is simple and easy to compare.
# What assumption it is making:
# - A short textual summary is enough for the first experimental loop.
def print_summary(task_type: str, policy_name: str, summary: Dict[str, float], position_counts: Dict[str, int]) -> None:
    print(f"\nTask: {task_type}")
    print(f"Policy: {policy_name}")
    print(f"  Episodes: {int(summary['episodes'])}")
    print(f"  Full-hit rate: {summary['full_hit_rate']:.2%}")
    print(f"  Stub-hit rate: {summary['stub_hit_rate']:.2%}")
    print(f"  Miss rate: {summary['miss_rate']:.2%}")
    print(f"  Confusion rate: {summary['confusion_rate']:.2%}")
    print(f"  Average full records stored: {summary['average_full_records_stored']:.2f}")
    print(f"  Average stubs stored: {summary['average_stubs_stored']:.2f}")
    print(f"  Average stub fields retained: {summary['average_stub_fields_retained']:.2f}")
    print(f"  Average fields per stub: {summary['average_fields_per_stub']:.2f}")
    print(f"  Retrieval success per stub field retained: {summary['retrieval_success_per_stub_field']:.4f}")
    print(f"  Target positions: {position_counts}")


# This helper prints a short per-position metric breakdown.
# What this does:
# - Displays accuracy and miss rate separately for early, middle, late, or mixed targets.
# Why this exists:
# - The overall summary can hide whether one placement regime is driving the effect.
# What assumption it is making:
# - A small text table is enough for v1 analysis.
def print_position_breakdown(position_summaries: Dict[str, Dict[str, float]]) -> None:
    print("  Position breakdown:")
    for target_position, summary in position_summaries.items():
        print(
            f"    {target_position}: "
            f"full_hit_rate={summary['full_hit_rate']:.2%}, "
            f"stub_hit_rate={summary['stub_hit_rate']:.2%}, "
            f"miss_rate={summary['miss_rate']:.2%}, "
            f"confusion_rate={summary['confusion_rate']:.2%}, "
            f"episodes={int(summary['episodes'])}"
        )


# This helper prints the small Meaning Address routing comparison.
# What this does:
# - Shows how flat lookup compares to address-guided lookup for candidate set size and confusion.
# Why this exists:
# - The user asked for a routing-layer experiment above the existing sandbox.
# What assumption it is making:
# - A short textual comparison is enough to interpret Meaning Address v0.
def print_routing_summary(summary: Dict[str, float]) -> None:
    print("  Meaning Address routing:")
    print(f"    Flat average candidates: {summary['average_flat_candidates']:.2f}")
    print(f"    Routed average candidates: {summary['average_routed_candidates']:.2f}")
    print(f"    Candidate reduction: {summary['candidate_reduction']:.2f}")
    print(f"    Flat confusion rate: {summary['flat_confusion_rate']:.2%}")
    print(f"    Routed confusion rate: {summary['routed_confusion_rate']:.2%}")


# This helper prints the focused two-stage routing comparison.
# What this does:
# - Displays candidate counts and confusion rates for flat, one-stage, and two-stage routes.
# Why this exists:
# - The new experiment is about whether semantic routing should use a coarse splitter first and a tie-breaker second.
# What assumption it is making:
# - A compact text summary is enough to compare these route shapes.
def print_route_comparison_summary(summary: Dict[str, float]) -> None:
    print("  Route comparison:")
    print(f"    Flat average candidates: {summary['average_flat_candidates']:.2f}")
    print(f"    Identity-only average candidates: {summary['average_identity_candidates']:.2f}")
    print(f"    Identity+therapy average candidates: {summary['average_identity_therapy_candidates']:.2f}")
    print(f"    Current Meaning Address average candidates: {summary['average_meaning_candidates']:.2f}")
    print(f"    Two-stage stage 1 average candidates: {summary['average_two_stage_stage_1_candidates']:.2f}")
    print(f"    Two-stage stage 2 average candidates: {summary['average_two_stage_stage_2_candidates']:.2f}")
    print(f"    Flat confusion rate: {summary['flat_confusion_rate']:.2%}")
    print(f"    Identity-only confusion rate: {summary['identity_confusion_rate']:.2%}")
    print(f"    Identity+therapy confusion rate: {summary['identity_therapy_confusion_rate']:.2%}")
    print(f"    Current Meaning Address confusion rate: {summary['meaning_confusion_rate']:.2%}")
    print(f"    Two-stage confusion rate: {summary['two_stage_confusion_rate']:.2%}")
    print(
        "    Two-stage conclusion: "
        f"beats_flat={summary['two_stage_beats_flat']}, "
        f"beats_identity_therapy={summary['two_stage_beats_identity_therapy']}, "
        f"beats_current_meaning={summary['two_stage_beats_meaning']}"
    )


# This helper prints the routing chooser rows across ambiguity levels and task types.
# What this does:
# - Displays the three route strategies and the chosen route for each regime.
# Why this exists:
# - Database Whisper should surface an actual routing policy, not just per-run metrics.
# What assumption it is making:
# - A flat text table is enough for this first chooser experiment.
def print_route_chooser_rows(rows: List[RouteChooserRow]) -> None:
    print("Database Whisper routing chooser")
    for row in rows:
        print(
            f"  task={row.task_type} | level={row.distractor_level} | "
            f"identity_candidates={row.identity_candidates:.2f} | "
            f"identity_confusion={row.identity_confusion_rate:.2%} | "
            f"identity+therapy_candidates={row.identity_therapy_candidates:.2f} | "
            f"identity+therapy_confusion={row.identity_therapy_confusion_rate:.2%} | "
            f"two_stage_stage1={row.two_stage_stage_1_candidates:.2f} | "
            f"two_stage_stage2={row.two_stage_stage_2_candidates:.2f} | "
            f"two_stage_confusion={row.two_stage_confusion_rate:.2%} | "
            f"coarse_paid={row.coarse_routing_paid_for_itself} | "
            f"choose={row.chosen_route}"
        )


# This helper prints one explanation line for each retrieval routed by the chooser.
# What this does:
# - Shows the identity neighborhood size, stage-by-stage candidate shrink, chosen route,
#   and the chooser's reason for each retrieval.
# Why this exists:
# - The user asked us to make the routing method legible before we expand it further.
# What assumption it is making:
# - A compact one-line explanation is enough per retrieval.
def print_route_explanations(rows: List[RouteExplanationRow]) -> None:
    print("Database Whisper route explainer")
    for row in rows:
        print(
            f"  episode={row.episode_id} | task={row.task_type} | level={row.distractor_level} | "
            f"identity_size={row.identity_neighborhood_size} | "
            f"after_evidence_type={row.evidence_type_candidate_count} | "
            f"after_therapy={row.therapy_candidate_count} | "
            f"choose={row.chosen_route} | "
            f"reason={row.reason}"
        )


# This helper prints a compact summary of the route explainer output.
# What this does:
# - Summarizes how often each route shape was chosen and how much each stage shrank the
#   candidate set on average.
# Why this exists:
# - The user asked for an end-of-run explainer summary, not only per-retrieval lines.
# What assumption it is making:
# - Route frequencies and average stage shrink are the right legibility metrics for this phase.
def print_route_explainer_summary(rows: List[RouteExplanationRow]) -> None:
    print("Database Whisper route explainer summary")
    total = len(rows)
    if not total:
        print("  no route explanations were recorded")
        return

    identity_only_count = sum(row.chosen_route == "identity_only" for row in rows)
    staged_count = sum(row.chosen_route == "identity_then_evidence_type_then_therapy" for row in rows)
    identity_therapy_count = sum(row.chosen_route == "identity_plus_therapy" for row in rows)
    average_stage_1_shrink = sum(
        row.identity_neighborhood_size - row.evidence_type_candidate_count for row in rows
    ) / total
    average_stage_2_shrink = sum(
        row.evidence_type_candidate_count - row.therapy_candidate_count for row in rows
    ) / total

    print(f"  identity-only chosen: {identity_only_count}/{total}")
    print(f"  identity+therapy chosen: {identity_therapy_count}/{total}")
    print(f"  staged routing chosen: {staged_count}/{total}")
    print(f"  average candidate shrink after evidence_type split: {average_stage_1_shrink:.2f}")
    print(f"  average candidate shrink after therapy split: {average_stage_2_shrink:.2f}")


# This helper prints the inferred routing policy in plain English.
# What this does:
# - Converts the chooser rows into a short human-readable route policy.
# Why this exists:
# - The user asked for the inferred routing policy, not just a numeric report.
# What assumption it is making:
# - Grouping by chosen route across low and high ambiguity regimes is enough for the first rule.
def print_route_chooser_policy(rows: List[RouteChooserRow]) -> None:
    print("Database Whisper inferred routing policy")
    low_rows = [row for row in rows if row.distractor_level in {"easy", "medium"}]
    high_rows = [row for row in rows if row.distractor_level in {"collision", "ambiguity"}]

    low_choices = {row.chosen_route for row in low_rows}
    high_choices = {row.chosen_route for row in high_rows}

    if low_choices == {"identity_only"}:
        print("  When ambiguity is low, prefer identity only.")
    elif low_choices == {"identity_plus_therapy"}:
        print("  When ambiguity is low, prefer identity plus therapy.")
    else:
        print("  When ambiguity is low, route choice is mixed in the current sandbox.")

    if high_choices == {"identity_then_evidence_type_then_therapy"}:
        print("  When ambiguity is high, prefer the staged route: identity, then evidence_type, then therapy.")
    elif "identity_then_evidence_type_then_therapy" in high_choices:
        print("  When ambiguity is high, the staged route is often worth using, but not universally.")
    else:
        print("  When ambiguity is high, the staged route does not consistently beat the simpler routes yet.")


# This helper prints the routing stress-test conditions.
# What this does:
# - Displays route performance, candidate counts, and route cost across the parameter sweep.
# Why this exists:
# - We want to inspect where the chooser policy holds under different ambiguity and scale settings.
# What assumption it is making:
# - A flat condition log is readable enough for the current research sandbox.
def print_route_stress_rows(rows: List[RouteStressRow]) -> None:
    print("Database Whisper routing stress test")
    for row in rows:
        print(
            f"  task={row.task_type} | level={row.distractor_level} | distractors={row.distractor_count} | pool={row.record_count} | "
            f"identity(cands={row.identity_candidates:.2f}, conf={row.identity_confusion_rate:.2%}, cost={row.identity_cost:.2f}) | "
            f"identity+therapy(cands={row.identity_therapy_candidates:.2f}, conf={row.identity_therapy_confusion_rate:.2%}, cost={row.identity_therapy_cost:.2f}) | "
            f"two_stage(s1={row.two_stage_stage_1_candidates:.2f}, s2={row.two_stage_stage_2_candidates:.2f}, conf={row.two_stage_confusion_rate:.2%}, cost={row.two_stage_cost:.2f}) | "
            f"stop_at_identity_therapy={row.stop_at_identity_therapy} | pay_for_two_stage={row.pay_for_two_stage}"
        )


# This helper prints the final stop rule in plain English.
# What this does:
# - Summarizes when staged routing is worth its extra routing cost and when it is not.
# Why this exists:
# - The user asked for a plain-English final chooser policy after stress testing.
# What assumption it is making:
# - Aggregating the winning conditions is enough to express a useful first stop rule.
def print_route_stress_policy(rows: List[RouteStressRow]) -> None:
    print("Database Whisper final routing policy")
    staged_rows = [row for row in rows if row.pay_for_two_stage]
    non_staged_rows = [row for row in rows if not row.pay_for_two_stage]

    if staged_rows:
        staged_levels = sorted({row.distractor_level for row in staged_rows})
        staged_tasks = sorted({row.task_type for row in staged_rows})
        print(
            "  Pay for full two-stage routing when ambiguity is high enough that identity+therapy still leaves "
            "meaningful confusion and stage 1 noticeably shrinks the candidate set."
        )
        print(f"  In this sweep, staged routing paid off mainly for levels={staged_levels} and tasks={staged_tasks}.")
    else:
        print("  In this sweep, full two-stage routing never paid for itself.")

    if non_staged_rows:
        non_staged_levels = sorted({row.distractor_level for row in non_staged_rows})
        print(
            "  Stop at identity+therapy when confusion is already low or when the extra stage adds cost "
            "without enough additional confusion reduction."
        )
        print(f"  In this sweep, stopping early was common for levels={non_staged_levels}.")


# This helper prints the Database Whisper candidate ranking for one ladder step.
# What this does:
# - Shows how each remaining field performed as the next discriminator choice.
# Why this exists:
# - The user wants field ranking to be inferred from the data rather than assumed.
# What assumption it is making:
# - A small per-step ranking is readable enough for terminal inspection.
def print_whisper_candidate_ranking(step_index: int, scores: List[WhisperFieldScore], label: str) -> None:
    print(f"  Database Whisper: {label} step {step_index} candidate ranking")
    for index, score in enumerate(scores, start=1):
        print(
            f"    {index}. {score.field_name}: "
            f"category={whisper_field_category(score.field_name)}, "
            f"ambiguity_pairs_before={score.ambiguity_pairs_before}, "
            f"ambiguity_pairs_after={score.ambiguity_pairs_after}, "
            f"ambiguity_reduction={score.ambiguity_reduction}, "
            f"reduction_rate={score.ambiguity_reduction_rate:.2%}, "
            f"field_cost={score.field_cost}, "
            f"reduction_per_cost={score.reduction_per_cost:.2%}"
        )


# This helper prints the inferred discriminator ladder in a short readable form.
# What this does:
# - Displays the best first, second, and third discriminator choices learned from the dataset.
# Why this exists:
# - Database Whisper should explain the discovered ladder, not just compute it silently.
# What assumption it is making:
# - A short step list is enough to communicate the learned fallback order.
def print_whisper_ladder(steps: List[WhisperStep], label: str) -> None:
    print(f"  Database Whisper: {label} inferred discriminator ladder")
    if not steps:
        print("    no useful discriminator field was discovered")
        return

    for step in steps:
        ordinal_name = {1: "first", 2: "second", 3: "third"}.get(step.step_index, f"step {step.step_index}")
        print(
            f"    best {ordinal_name} discriminator: {step.chosen_field} | "
            f"ambiguity_pairs_before={step.ambiguity_pairs_before}, "
            f"ambiguity_pairs_after={step.ambiguity_pairs_after}, "
            f"ambiguity_reduction={step.ambiguity_reduction}, "
            f"reduction_rate={step.ambiguity_reduction_rate:.2%}"
        )


# This helper prints the final Database Whisper recommendation in one short block.
# What this does:
# - Summarizes the learned field order as a compact ladder recommendation.
# Why this exists:
# - The user asked for a clear best-first discriminator ladder.
# What assumption it is making:
# - A short ordered list is enough to capture the current inferred policy.
def print_whisper_final_recommendation(steps: List[WhisperStep], label: str) -> None:
    print(f"  Database Whisper {label} recommendation:")
    if not steps:
        print("    no discriminator ladder recommendation")
        return

    print("    1. minimal_identity")
    chosen_fields: List[str] = []
    for index, step in enumerate(steps, start=2):
        chosen_fields.append(step.chosen_field)
        joined_fields = " + ".join(chosen_fields)
        print(f"    {index}. minimal_identity + {joined_fields}")


# This helper explains why provenance-like uniqueness can win the raw score.
# What this does:
# - Prints a short interpretation block that separates structural uniqueness from semantic routing value.
# Why this exists:
# - The user asked for a clear explanation of why `source` can look optimal while being the wrong route.
# What assumption it is making:
# - A short textual explanation next to the ranking is enough for this stage of the project.
def print_whisper_routing_note(raw_steps: List[WhisperStep], semantic_steps: List[WhisperStep]) -> None:
    print("  Database Whisper note:")
    if raw_steps and raw_steps[0].chosen_field == "source":
        print(
            "    source wins the raw ranking because record_id-like uniqueness can collapse "
            "every competing neighborhood immediately."
        )
        print(
            "    That is strong for provenance or reconstruction, but weak as a semantic route "
            "because it does not express conceptual separation that a future query can naturally provide."
        )
    if semantic_steps:
        print(
            "    The semantic ranking removes provenance shortcuts and asks which meaningful field "
            "best separates nearby records by content."
        )


# This helper prints the same semantic fields on both coarse and final-retrieval axes.
# What this does:
# - Shows which fields are best for neighborhood narrowing and which are best for task-time disambiguation.
# Why this exists:
# - The current project insight is that routing quality and final answer quality are not the same objective.
# What assumption it is making:
# - A small dual-axis text report is enough to expose the mismatch clearly.
def print_dual_axis_scores(scores: List[DualAxisFieldScore]) -> None:
    print("  Database Whisper: dual-axis field view")
    print("    Coarse routing ranking:")
    for index, score in enumerate(
        sorted(scores, key=lambda row: row.coarse_ambiguity_reduction_rate, reverse=True),
        start=1,
    ):
        print(
            f"      {index}. {score.field_name}: "
            f"coarse_reduction={score.coarse_ambiguity_reduction_rate:.2%}"
        )

    print("    Final retrieval ranking:")
    for index, score in enumerate(
        sorted(scores, key=lambda row: row.retrieval_confusion_reduction_rate, reverse=True),
        start=1,
    ):
        print(
            f"      {index}. {score.field_name}: "
            f"retrieval_confusion_rate={score.retrieval_confusion_rate:.2%}, "
            f"retrieval_confusion_reduction={score.retrieval_confusion_reduction:.2%}, "
            f"retrieval_reduction_rate={score.retrieval_confusion_reduction_rate:.2%}"
        )


# This helper prints the single-field discriminator ranking.
# What this does:
# - Shows which one-field additions buy the most ambiguity reduction per field cost.
# Why this exists:
# - The user wants a ranked ladder rather than a flat list of variants.
# What assumption it is making:
# - Sorting by confusion reduction per cost is the right first ranking rule.
def print_single_field_ladder(rows: List[LadderRow]) -> None:
    print("  Discriminator ladder: single fields")
    for index, row in enumerate(rows, start=1):
        field_name = row.discriminator_fields[0]
        print(
            f"    {index}. {field_name}: "
            f"confusion_rate={row.confusion_rate:.2%}, "
            f"confusion_reduction={row.confusion_reduction:.2%}, "
            f"added_field_cost={row.added_field_cost}, "
            f"reduction_per_cost={row.confusion_reduction_per_cost:.2%}"
        )


# This helper prints the pair-combination comparison for the top field.
# What this does:
# - Shows whether adding one more field on top of the winning field is worth it.
# Why this exists:
# - The user asked for a fallback ladder, not just a single winner.
# What assumption it is making:
# - Therapy-plus-one combinations are the right next rung after the best single field.
def print_pair_field_ladder(rows: List[LadderRow], top_field: str) -> None:
    print(f"  Discriminator ladder: {top_field} plus one")
    for index, row in enumerate(rows, start=1):
        paired_fields = "+".join(row.discriminator_fields)
        print(
            f"    {index}. {paired_fields}: "
            f"confusion_rate={row.confusion_rate:.2%}, "
            f"confusion_reduction={row.confusion_reduction:.2%}, "
            f"added_field_cost={row.added_field_cost}, "
            f"reduction_per_cost={row.confusion_reduction_per_cost:.2%}"
        )


# This helper prints the final ladder recommendation in one short block.
# What this does:
# - Summarizes the recommended fallback order for this domain.
# Why this exists:
# - The user asked for a final discriminator ladder recommendation.
# What assumption it is making:
# - The best single field should come first, and the best pair is the next fallback rung.
def print_final_ladder_recommendation(single_rows: List[LadderRow], pair_rows: List[LadderRow]) -> None:
    if not single_rows:
        return

    top_single = single_rows[0]
    print("  Final ladder recommendation:")
    print(f"    1. minimal_identity")
    print(f"    2. minimal_identity + {top_single.discriminator_fields[0]}")
    if pair_rows:
        best_pair = pair_rows[0]
        extra_fields = "+".join(best_pair.discriminator_fields)
        print(f"    3. minimal_identity + {extra_fields}")


# This helper prints a few example episodes to make the run interpretable.
# What this does:
# - Shows target facts, the query, and the model answer for the first few episodes.
# Why this exists:
# - Numbers alone are not enough during research; we also want quick qualitative inspection.
# What assumption it is making:
# - Looking at a few concrete examples helps catch obvious logic mistakes early.
def print_example_results(results: List[EpisodeResult], episodes: List[RecallEpisode], max_examples: int = 3) -> None:
    print("Example episodes:")
    for result, episode in list(zip(results, episodes))[:max_examples]:
        print(
            f"  {episode.episode_id} | position={episode.target_position} | "
            f"expected={result.expected_answer} | predicted={result.predicted_answer}"
        )
        print(f"    Query: {episode.query.prompt}")
        print(
            "    Target: "
            f"{episode.target_record.gene} {episode.target_record.variant} | "
            f"{episode.target_record.disease} | "
            f"drug={episode.target_record.drug} | "
            f"evidence_direction={episode.target_record.evidence_direction} | "
            f"evidence_level={episode.target_record.evidence_level}"
        )


# This helper decides which task types to run.
# What this does:
# - Expands a single CLI choice into one or both task types.
# Why this exists:
# - The runner should stay easy to use while still supporting per-task summaries.
# What assumption it is making:
# - v1 only needs two task types: direct recall and late-relevance recall.
def resolve_task_types(task_type: str) -> List[str]:
    if task_type == "all":
        return ["direct_recall", "late_relevance_recall"]
    return [task_type]


# This helper turns one source name into a short user-facing label.
# What this does:
# - Keeps synthetic-versus-real reporting readable without repeating small formatting rules.
# Why this exists:
# - The new validation pass needs to print both data paths clearly.
# What assumption it is making:
# - Two labels, synthetic and real, are enough for the current compare mode.
def format_data_source_label(data_source: str) -> str:
    if data_source == "real":
        return "real_civic_slice"
    if data_source == "real_clinvar":
        return "real_clinvar_slice"
    return "synthetic"


# This helper describes the identity fields in domain-facing language.
# What this does:
# - Translates the fixed internal identity slots into a domain-specific explanation.
# Why this exists:
# - The same internal method should still report legible field names when we move to a second domain.
# What assumption it is making:
# - A short human-readable identity label is enough to explain the domain mapping.
def primary_identity_label_for_data_source(data_source: str) -> str:
    if data_source == "real_clinvar":
        return "gene + variant + condition"
    return "gene + variant + disease"


# This helper maps internal field-slot names to domain-facing names.
# What this does:
# - Converts the current internal discriminator slots into language that fits the active domain.
# Why this exists:
# - We want cross-domain comparison to be interpretable even though the internal schema stays fixed.
# What assumption it is making:
# - A small alias table is enough for the current two-domain experiment.
def field_label_for_data_source(data_source: str, field_name: str) -> str:
    if data_source == "real_clinvar":
        aliases = {
            "therapy": "clinical_significance",
            "evidence_type": "review_pattern",
            "evidence_level": "review_status_tier",
            "direction": "assertion_direction",
        }
        return aliases.get(field_name, field_name)

    aliases = {
        "therapy": "therapy",
        "evidence_type": "evidence_type",
        "evidence_level": "evidence_level",
        "direction": "direction",
    }
    return aliases.get(field_name, field_name)


# This helper extracts the top coarse and final fields from the dual-axis report.
# What this does:
# - Converts the dual-axis field table into one compact comparison snapshot.
# Why this exists:
# - The final compare block should answer the transfer question directly instead of making
#   the user inspect the whole ranking manually.
# What assumption it is making:
# - The top-ranked field on each axis is enough for the current validation question.
def top_dual_axis_fields(scores: List[DualAxisFieldScore]) -> tuple[str, str]:
    if not scores:
        return ("none", "none")

    top_coarse = max(scores, key=lambda row: row.coarse_ambiguity_reduction_rate)
    top_final = max(scores, key=lambda row: row.retrieval_confusion_reduction_rate)
    return (top_coarse.field_name, top_final.field_name)


# This helper builds a small cross-source comparison row for one task type.
# What this does:
# - Summarizes the learned ladder and the ambiguity chooser into one comparable structure.
# Why this exists:
# - We want a concise synthetic-versus-real method check after the full reports print.
# What assumption it is making:
# - The ambiguity setting is the most informative regime for comparing learned routing behavior.
def build_source_comparison_row(
    data_source: str,
    task_type: str,
    semantic_steps: List[WhisperStep],
    dual_axis_scores: List[DualAxisFieldScore],
    chooser_rows: List[RouteChooserRow],
) -> SourceComparisonRow:
    coarse_splitter, final_tie_breaker = top_dual_axis_fields(dual_axis_scores)
    semantic_ladder = [step.chosen_field for step in semantic_steps]
    ambiguity_row = next(
        (
            row
            for row in chooser_rows
            if row.task_type == task_type and row.distractor_level == "ambiguity"
        ),
        None,
    )
    chooser_route_for_ambiguity = ambiguity_row.chosen_route if ambiguity_row is not None else "unknown"

    return SourceComparisonRow(
        data_source=data_source,
        task_type=task_type,
        coarse_splitter=coarse_splitter,
        final_tie_breaker=final_tie_breaker,
        semantic_ladder=semantic_ladder,
        chooser_route_for_ambiguity=chooser_route_for_ambiguity,
    )


# This helper updates a comparison row with the chooser outcome for ambiguity mode.
# What this does:
# - Fills in the route choice after the chooser experiment has run.
# Why this exists:
# - The ambiguity chooser is computed after the per-task policy loop, but the compare row
#   still needs to include that route decision.
# What assumption it is making:
# - Only the ambiguity-level chooser result matters for the current compare question.
def attach_chooser_route_to_comparison_row(
    row: SourceComparisonRow,
    chooser_rows: List[RouteChooserRow],
) -> SourceComparisonRow:
    ambiguity_row = next(
        (
            chooser_row
            for chooser_row in chooser_rows
            if chooser_row.task_type == row.task_type and chooser_row.distractor_level == "ambiguity"
        ),
        None,
    )
    chooser_route_for_ambiguity = ambiguity_row.chosen_route if ambiguity_row is not None else "unknown"
    return SourceComparisonRow(
        data_source=row.data_source,
        task_type=row.task_type,
        coarse_splitter=row.coarse_splitter,
        final_tie_breaker=row.final_tie_breaker,
        semantic_ladder=list(row.semantic_ladder),
        chooser_route_for_ambiguity=chooser_route_for_ambiguity,
    )


# This helper prints the final synthetic-versus-real comparison block.
# What this does:
# - Displays whether the learned ladder and route choice transfer from synthetic data to
#   the tiny real CIViC slice.
# Why this exists:
# - The user asked for a direct comparison, not just two independent runs.
# What assumption it is making:
# - A short per-task compare block is enough for this first real-data validation pass.
def print_data_source_comparison(
    synthetic_rows: List[SourceComparisonRow],
    real_rows: List[SourceComparisonRow],
) -> None:
    print("\nComparison: synthetic vs real CIViC slice")
    synthetic_by_task = {row.task_type: row for row in synthetic_rows}
    real_by_task = {row.task_type: row for row in real_rows}

    for task_type in sorted(set(synthetic_by_task) & set(real_by_task)):
        synthetic_row = synthetic_by_task[task_type]
        real_row = real_by_task[task_type]
        print(f"  task={task_type}")
        print(
            f"    synthetic: coarse_splitter={synthetic_row.coarse_splitter}, "
            f"final_tie_breaker={synthetic_row.final_tie_breaker}, "
            f"semantic_ladder={synthetic_row.semantic_ladder}, "
            f"ambiguity_route={synthetic_row.chooser_route_for_ambiguity}"
        )
        print(
            f"    real: coarse_splitter={real_row.coarse_splitter}, "
            f"final_tie_breaker={real_row.final_tie_breaker}, "
            f"semantic_ladder={real_row.semantic_ladder}, "
            f"ambiguity_route={real_row.chooser_route_for_ambiguity}"
        )

    synthetic_coarse_matches = all(row.coarse_splitter == "evidence_type" for row in synthetic_rows)
    synthetic_final_matches = all(row.final_tie_breaker == "therapy" for row in synthetic_rows)
    real_coarse_matches = all(row.coarse_splitter == "evidence_type" for row in real_rows)
    real_final_matches = all(row.final_tie_breaker == "therapy" for row in real_rows)

    print("  Transfer check:")
    print(f"    synthetic coarse splitter is evidence_type: {synthetic_coarse_matches}")
    print(f"    synthetic final tie-breaker is therapy: {synthetic_final_matches}")
    print(f"    real coarse splitter is evidence_type: {real_coarse_matches}")
    print(f"    real final tie-breaker is therapy: {real_final_matches}")


# This helper prints a cross-domain comparison for two real structured datasets.
# What this does:
# - Compares the discovered identity description, coarse splitter, final tie-breaker, and
#   chooser policy across the oncology and ClinVar-like domains.
# Why this exists:
# - The user asked whether the same discovery procedure transfers when the semantic ladder changes.
# What assumption it is making:
# - A short per-task comparison is enough for the first cross-domain validation pass.
def print_cross_domain_comparison(
    oncology_rows: List[SourceComparisonRow],
    second_domain_rows: List[SourceComparisonRow],
    second_domain_source: str,
) -> None:
    print("\nComparison: CIViC oncology vs second structured domain")
    oncology_by_task = {row.task_type: row for row in oncology_rows}
    second_by_task = {row.task_type: row for row in second_domain_rows}

    for task_type in sorted(set(oncology_by_task) & set(second_by_task)):
        oncology_row = oncology_by_task[task_type]
        second_row = second_by_task[task_type]
        print(f"  task={task_type}")
        print(
            f"    oncology: primary_identity={primary_identity_label_for_data_source('real')}, "
            f"coarse_splitter={field_label_for_data_source('real', oncology_row.coarse_splitter)}, "
            f"final_tie_breaker={field_label_for_data_source('real', oncology_row.final_tie_breaker)}, "
            f"chooser={oncology_row.chooser_route_for_ambiguity}"
        )
        print(
            f"    second_domain: primary_identity={primary_identity_label_for_data_source(second_domain_source)}, "
            f"coarse_splitter={field_label_for_data_source(second_domain_source, second_row.coarse_splitter)}, "
            f"final_tie_breaker={field_label_for_data_source(second_domain_source, second_row.final_tie_breaker)}, "
            f"chooser={second_row.chooser_route_for_ambiguity}"
        )

    same_coarse = all(
        oncology_by_task[task_type].coarse_splitter == second_by_task[task_type].coarse_splitter
        for task_type in set(oncology_by_task) & set(second_by_task)
    )
    same_final = all(
        oncology_by_task[task_type].final_tie_breaker == second_by_task[task_type].final_tie_breaker
        for task_type in set(oncology_by_task) & set(second_by_task)
    )
    same_route = all(
        oncology_by_task[task_type].chooser_route_for_ambiguity
        == second_by_task[task_type].chooser_route_for_ambiguity
        for task_type in set(oncology_by_task) & set(second_by_task)
    )

    print("  Cross-domain check:")
    print(f"    same coarse splitter slot as oncology: {same_coarse}")
    print(f"    same final tie-breaker slot as oncology: {same_final}")
    print(f"    same ambiguity chooser route as oncology: {same_route}")


# This helper runs the current baseline on one data source and returns comparison rows.
# What this does:
# - Executes the existing routing and memory experiments on either synthetic data or the
#   normalized real CIViC slice.
# Why this exists:
# - We want to validate the discovered method on real data without forking the codebase.
# What assumption it is making:
# - The same experiment logic is meaningful on both sources once the real sample is
#   normalized into the current internal schema.
def run_baseline_for_data_source(
    task_type: str,
    record_count: int,
    episodes: int,
    distractor_level: str,
    distractor_count: int,
    data_source: str,
    seed: int,
) -> List[SourceComparisonRow]:
    stub_policies = [StubMemoryPolicy(stub_schema=schema) for schema in STUB_SCHEMAS]
    policies = [SaveAllPolicy(), RuleBasedSaliencePolicy(), TieredMemoryPolicy(), *stub_policies]
    task_types = resolve_task_types(task_type)
    source_comparison_rows: List[SourceComparisonRow] = []
    base_records = load_record_pool(data_source=data_source, record_count=record_count, seed=seed)

    print("memory_lab baseline run")
    print(f"  data_source={format_data_source_label(data_source)}")
    print(f"  task_type={task_type}")
    print(f"  episodes={episodes}")
    print(f"  distractor_level={distractor_level}")
    print(f"  distractor_count={distractor_count}")
    print(f"  record_pool={len(base_records)}")
    print(f"  seed={seed}")

    for current_task_type in task_types:
        generated_episodes = generate_recall_episodes(
            task_type=current_task_type,
            record_count=record_count,
            episode_count=episodes,
            distractor_level=distractor_level,
            distractor_count=distractor_count,
            data_source=data_source,
            seed=seed,
        )

        print(f"\n{'=' * 60}")
        print(f"Running task type: {current_task_type}")
        print(f"{'=' * 60}")

        per_policy_results: Dict[str, List[EpisodeResult]] = {}
        for policy in policies:
            results = [evaluate_episode(policy=policy, episode=episode) for episode in generated_episodes]
            routing_results = [evaluate_routing_episode(policy=policy, episode=episode) for episode in generated_episodes]
            route_comparison_results = [
                evaluate_route_comparison_episode(policy=policy, episode=episode) for episode in generated_episodes
            ]
            per_policy_results[policy.name] = results
            summary = summarize_results(results)
            routing_summary = summarize_routing_results(routing_results)
            route_comparison_summary = summarize_route_comparison_results(route_comparison_results)
            position_counts = summarize_positions(results)
            position_summaries = summarize_by_target_position(results)
            print_summary(current_task_type, policy.name, summary, position_counts)
            print_routing_summary(routing_summary)
            print_route_comparison_summary(route_comparison_summary)
            print_position_breakdown(position_summaries)
            print_example_results(results, generated_episodes)
            print()

        # The ladder question is most meaningful in the ambiguity arena where confusion is real.
        if distractor_level == "ambiguity":
            whisper_records = unique_records_from_episodes(generated_episodes)
            whisper_neighborhoods = ambiguous_identity_neighborhoods(whisper_records)
            raw_whisper_steps, raw_whisper_rankings = infer_discriminator_ladder(
                neighborhoods=whisper_neighborhoods,
                candidate_fields=WHISPER_RAW_CANDIDATE_FIELDS,
                max_steps=3,
            )
            semantic_whisper_steps, semantic_whisper_rankings = infer_discriminator_ladder(
                neighborhoods=whisper_neighborhoods,
                candidate_fields=WHISPER_SEMANTIC_FIELDS,
                max_steps=3,
            )

            print(
                "  Database Whisper dataset summary: "
                f"records={len(whisper_records)}, "
                f"ambiguous_neighborhoods={len(whisper_neighborhoods)}, "
                f"baseline_ambiguity_pairs={remaining_ambiguity_pairs(whisper_neighborhoods, selected_fields=[])}"
            )
            for step_index, scores in enumerate(raw_whisper_rankings, start=1):
                print_whisper_candidate_ranking(step_index=step_index, scores=scores, label="raw")
            print_whisper_ladder(raw_whisper_steps, label="raw")
            print_whisper_final_recommendation(raw_whisper_steps, label="raw")
            for step_index, scores in enumerate(semantic_whisper_rankings, start=1):
                print_whisper_candidate_ranking(step_index=step_index, scores=scores, label="semantic-only")
            print_whisper_ladder(semantic_whisper_steps, label="semantic-only")
            print_whisper_final_recommendation(semantic_whisper_steps, label="semantic-only")
            if semantic_whisper_rankings:
                dual_axis_scores = build_dual_axis_field_scores(
                    policy_results=per_policy_results,
                    semantic_ranking=semantic_whisper_rankings[0],
                    summarize_results_fn=summarize_results,
                )
                print_dual_axis_scores(dual_axis_scores)
                source_comparison_rows.append(
                    build_source_comparison_row(
                        data_source=data_source,
                        task_type=current_task_type,
                        semantic_steps=semantic_whisper_steps,
                        dual_axis_scores=dual_axis_scores,
                        chooser_rows=[],
                    )
                )
            print_whisper_routing_note(raw_whisper_steps, semantic_whisper_steps)
            print()

    chooser_rows, explanation_rows = run_route_chooser_experiment(
        task_types=task_types,
        record_count=record_count,
        episodes=episodes,
        distractor_count=distractor_count,
        data_source=data_source,
        seed=seed,
    )
    print_route_chooser_rows(chooser_rows)
    print_route_chooser_policy(chooser_rows)
    print_route_explanations(explanation_rows)
    print_route_explainer_summary(explanation_rows)

    return [
        attach_chooser_route_to_comparison_row(row=row, chooser_rows=chooser_rows)
        for row in source_comparison_rows
    ]


# This is the main experiment entry point.
# What this does:
# - Runs the current sandbox on synthetic data, the small real CIViC slice, or both and
#   prints a transfer comparison when requested.
# Why this exists:
# - The new step is to validate the discovered routing method on a tiny real-data slice
#   without redesigning the system.
# What assumption it is making:
# - A side-by-side synthetic-versus-real run is enough for the first validation pass.
def run_baseline(
    task_type: str,
    record_count: int,
    episodes: int,
    distractor_level: str,
    distractor_count: int,
    data_source: str,
    seed: int,
) -> None:
    if data_source == "compare":
        synthetic_rows = run_baseline_for_data_source(
            task_type=task_type,
            record_count=record_count,
            episodes=episodes,
            distractor_level=distractor_level,
            distractor_count=distractor_count,
            data_source="synthetic",
            seed=seed,
        )
        print(f"\n{'#' * 60}")
        real_rows = run_baseline_for_data_source(
            task_type=task_type,
            record_count=record_count,
            episodes=episodes,
            distractor_level=distractor_level,
            distractor_count=distractor_count,
            data_source="real",
            seed=seed,
        )
        print_data_source_comparison(synthetic_rows=synthetic_rows, real_rows=real_rows)
        return

    if data_source == "compare_domains":
        oncology_rows = run_baseline_for_data_source(
            task_type=task_type,
            record_count=record_count,
            episodes=episodes,
            distractor_level=distractor_level,
            distractor_count=distractor_count,
            data_source="real",
            seed=seed,
        )
        print(f"\n{'#' * 60}")
        second_domain_rows = run_baseline_for_data_source(
            task_type=task_type,
            record_count=record_count,
            episodes=episodes,
            distractor_level=distractor_level,
            distractor_count=distractor_count,
            data_source="real_clinvar",
            seed=seed,
        )
        print_cross_domain_comparison(
            oncology_rows=oncology_rows,
            second_domain_rows=second_domain_rows,
            second_domain_source="real_clinvar",
        )
        return

    run_baseline_for_data_source(
        task_type=task_type,
        record_count=record_count,
        episodes=episodes,
        distractor_level=distractor_level,
        distractor_count=distractor_count,
        data_source=data_source,
        seed=seed,
    )


# This parser keeps the command-line surface area small and obvious.
# What this does:
# - Lets the user change task type, episode count, distractor difficulty, and seed.
# Why this exists:
# - Research iteration is faster when a few core stress settings are easy to vary.
# What assumption it is making:
# - These few arguments are enough for the first experimental loop.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the v1 memory_lab baseline experiment.")
    parser.add_argument(
        "--task-type",
        choices=["direct_recall", "late_relevance_recall", "all"],
        default="all",
        help="Which recall task type to run.",
    )
    parser.add_argument("--record-count", type=int, default=80, help="Number of synthetic records in the pool.")
    parser.add_argument("--episodes", type=int, default=12, help="Number of episodes to generate per task type.")
    parser.add_argument(
        "--distractor-level",
        choices=["easy", "medium", "hard", "collision", "ambiguity"],
        default="medium",
        help="How similar the distractors should be to the target fact.",
    )
    parser.add_argument("--distractor-count", type=int, default=5, help="Number of distractors per episode.")
    parser.add_argument(
        "--data-source",
        choices=["synthetic", "real", "compare", "real_clinvar", "compare_domains"],
        default="synthetic",
        help="Which data source to run: synthetic, real CIViC, real ClinVar-like, or comparison modes.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducible runs.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_baseline(
        task_type=args.task_type,
        record_count=args.record_count,
        episodes=args.episodes,
        distractor_level=args.distractor_level,
        distractor_count=args.distractor_count,
        data_source=args.data_source,
        seed=args.seed,
    )
