"""whisper.py

Database Whisper analysis: field scoring, discriminator ladder inference,
and dual-axis field comparison for the memory_lab baseline experiment.
"""

from __future__ import annotations

from typing import Dict, List

from stream_generator import CivicRecord, RecallEpisode
from data_types import (
    DualAxisFieldScore,
    EpisodeResult,
    WhisperFieldScore,
    WhisperStep,
)
from retrieval import record_identity


# These field lists separate semantic routing cues from provenance-style identifiers.
# What this does:
# - Splits the current candidate fields into "meaningful for routing" versus "useful for tracing."
# Why this exists:
# - A field can be excellent at making rows unique while still being the wrong choice for semantic routing.
# What assumption it is making:
# - Therapy, evidence_type, evidence_level, and direction are semantic, while source is provenance.
WHISPER_SEMANTIC_FIELDS = ["therapy", "evidence_type", "evidence_level", "direction"]
WHISPER_PROVENANCE_FIELDS = ["source"]
WHISPER_RAW_CANDIDATE_FIELDS = [*WHISPER_SEMANTIC_FIELDS, *WHISPER_PROVENANCE_FIELDS]


# This helper extracts one candidate discriminator value from a record.
# What this does:
# - Converts a record into the value used for one candidate ambiguity-breaking field.
# Why this exists:
# - Database Whisper should evaluate fields by reading the same structured records the lab already uses.
# What assumption it is making:
# - These simple field projections are enough to test the current ladder candidates.
def record_field_value(record: CivicRecord, field_name: str) -> str:
    if field_name == "therapy":
        return record.drug
    if field_name == "evidence_type":
        return f"{record.evidence_direction}:{record.evidence_level}"
    if field_name == "evidence_level":
        return record.evidence_level
    if field_name == "direction":
        return record.evidence_direction
    if field_name == "source":
        return record.record_id
    raise ValueError(f"Unsupported discriminator field: {field_name}")


# This helper extracts the unique records present across a generated episode set.
# What this does:
# - Collapses repeated episode-stream records into one dataset keyed by record id.
# Why this exists:
# - Database Whisper should learn from the actual ambiguity-heavy records used in the run.
# What assumption it is making:
# - Record ids are stable enough to use as the uniqueness key for one run.
def unique_records_from_episodes(episodes: List[RecallEpisode]) -> List[CivicRecord]:
    records_by_id: Dict[str, CivicRecord] = {}
    for episode in episodes:
        for record in episode.stream_records:
            records_by_id.setdefault(record.record_id, record)
    return list(records_by_id.values())


# This helper groups records into identity neighborhoods that currently compete.
# What this does:
# - Collects all records that share gene, variant, and disease into one neighborhood.
# Why this exists:
# - The discriminator ladder should be learned only where true identity ambiguity exists.
# What assumption it is making:
# - Identity neighborhoods with more than one record are the right ambiguity unit for v1.
def ambiguous_identity_neighborhoods(records: List[CivicRecord]) -> List[List[CivicRecord]]:
    neighborhoods_by_identity: Dict[tuple[str, str, str], List[CivicRecord]] = {}
    for record in records:
        neighborhoods_by_identity.setdefault(record_identity(record), []).append(record)

    return [
        neighborhood
        for neighborhood in neighborhoods_by_identity.values()
        if len(neighborhood) > 1
    ]


# This helper counts how many unresolved competing record pairs remain.
# What this does:
# - Measures residual ambiguity after partitioning each neighborhood by selected fields.
# Why this exists:
# - The ladder needs one simple score that says how much confusion is still structurally possible.
# What assumption it is making:
# - Counting unresolved record pairs is a good first ambiguity metric for this sandbox.
def remaining_ambiguity_pairs(
    neighborhoods: List[List[CivicRecord]],
    selected_fields: List[str],
) -> int:
    total_pairs = 0

    for neighborhood in neighborhoods:
        grouped_records: Dict[tuple[str, ...], List[CivicRecord]] = {}
        for record in neighborhood:
            if selected_fields:
                bucket_key = tuple(record_field_value(record, field_name) for field_name in selected_fields)
            else:
                bucket_key = ("identity_only",)
            grouped_records.setdefault(bucket_key, []).append(record)

        for bucket_records in grouped_records.values():
            bucket_size = len(bucket_records)
            if bucket_size > 1:
                total_pairs += (bucket_size * (bucket_size - 1)) // 2

    return total_pairs


# This helper scores all remaining candidate fields for one ladder step.
# What this does:
# - Measures how much each candidate field reduces residual ambiguity when added next.
# Why this exists:
# - Database Whisper should rank possible next discriminators from the data, not from prior beliefs.
# What assumption it is making:
# - One added field at a time is the right greedy search strategy for v0.
def score_whisper_candidate_fields(
    neighborhoods: List[List[CivicRecord]],
    already_selected_fields: List[str],
    candidate_fields: List[str],
) -> List[WhisperFieldScore]:
    ambiguity_pairs_before = remaining_ambiguity_pairs(
        neighborhoods=neighborhoods,
        selected_fields=already_selected_fields,
    )
    rows: List[WhisperFieldScore] = []

    for field_name in candidate_fields:
        candidate_fields_for_step = [*already_selected_fields, field_name]
        ambiguity_pairs_after = remaining_ambiguity_pairs(
            neighborhoods=neighborhoods,
            selected_fields=candidate_fields_for_step,
        )
        ambiguity_reduction = ambiguity_pairs_before - ambiguity_pairs_after
        ambiguity_reduction_rate = (
            ambiguity_reduction / ambiguity_pairs_before if ambiguity_pairs_before else 0.0
        )
        field_cost = 1
        reduction_per_cost = ambiguity_reduction_rate / field_cost if field_cost else 0.0
        rows.append(
            WhisperFieldScore(
                field_name=field_name,
                ambiguity_pairs_before=ambiguity_pairs_before,
                ambiguity_pairs_after=ambiguity_pairs_after,
                ambiguity_reduction=ambiguity_reduction,
                ambiguity_reduction_rate=ambiguity_reduction_rate,
                field_cost=field_cost,
                reduction_per_cost=reduction_per_cost,
            )
        )

    return sorted(
        rows,
        key=lambda row: (
            row.reduction_per_cost,
            row.ambiguity_reduction_rate,
            row.ambiguity_reduction,
            row.field_name != "source",
        ),
        reverse=True,
    )


# This helper learns a discriminator ladder directly from the ambiguity-heavy dataset.
# What this does:
# - Greedily chooses the best next field until ambiguity is gone or no further reduction appears.
# Why this exists:
# - Database Whisper should infer the cheapest useful field order instead of being told it.
# What assumption it is making:
# - A greedy ladder is good enough for a small readable v0 search.
def infer_discriminator_ladder(
    neighborhoods: List[List[CivicRecord]],
    candidate_fields: List[str],
    max_steps: int = 3,
) -> tuple[List[WhisperStep], List[List[WhisperFieldScore]]]:
    chosen_fields: List[str] = []
    remaining_fields = list(candidate_fields)
    steps: List[WhisperStep] = []
    candidate_rankings_by_step: List[List[WhisperFieldScore]] = []

    for step_index in range(1, max_steps + 1):
        if not remaining_fields:
            break

        ranked_candidates = score_whisper_candidate_fields(
            neighborhoods=neighborhoods,
            already_selected_fields=chosen_fields,
            candidate_fields=remaining_fields,
        )
        candidate_rankings_by_step.append(ranked_candidates)

        if not ranked_candidates:
            break

        best_candidate = ranked_candidates[0]
        if best_candidate.ambiguity_reduction <= 0:
            break

        steps.append(
            WhisperStep(
                step_index=step_index,
                chosen_field=best_candidate.field_name,
                ambiguity_pairs_before=best_candidate.ambiguity_pairs_before,
                ambiguity_pairs_after=best_candidate.ambiguity_pairs_after,
                ambiguity_reduction=best_candidate.ambiguity_reduction,
                ambiguity_reduction_rate=best_candidate.ambiguity_reduction_rate,
            )
        )
        chosen_fields.append(best_candidate.field_name)
        remaining_fields = [field_name for field_name in remaining_fields if field_name != best_candidate.field_name]

        if best_candidate.ambiguity_pairs_after == 0:
            break

    return steps, candidate_rankings_by_step


# This helper maps a semantic field name to the matching single-field stub policy.
# What this does:
# - Connects the data-driven field analysis to the existing retrieval experiments.
# Why this exists:
# - We want to score the same candidate fields on actual task confusion, not just on structural splitting.
# What assumption it is making:
# - The stub policy naming convention remains stable across these small experiments.
def stub_policy_name_for_single_field(field_name: str) -> str:
    return f"StubMemoryPolicy[minimal_identity_plus_{field_name}]"


# This helper compares candidate fields on both routing and retrieval axes.
# What this does:
# - Joins coarse ambiguity reduction from Database Whisper with real confusion reduction from task results.
# Why this exists:
# - The current mismatch between evidence_type and therapy should be measured explicitly.
# What assumption it is making:
# - Comparing against the minimal-identity stub baseline is the right retrieval-side reference point.
def build_dual_axis_field_scores(
    policy_results: Dict[str, List[EpisodeResult]],
    semantic_ranking: List[WhisperFieldScore],
    summarize_results_fn,
) -> List[DualAxisFieldScore]:
    baseline_policy_name = "StubMemoryPolicy[minimal_identity]"
    baseline_results = policy_results.get(baseline_policy_name, [])
    baseline_confusion_rate = summarize_results_fn(baseline_results)["confusion_rate"] if baseline_results else 0.0
    coarse_by_field = {row.field_name: row for row in semantic_ranking}
    scores: List[DualAxisFieldScore] = []

    for field_name in WHISPER_SEMANTIC_FIELDS:
        policy_name = stub_policy_name_for_single_field(field_name)
        results = policy_results.get(policy_name, [])
        summary = summarize_results_fn(results) if results else {"confusion_rate": 0.0}
        retrieval_confusion_rate = summary["confusion_rate"]
        retrieval_confusion_reduction = baseline_confusion_rate - retrieval_confusion_rate
        retrieval_confusion_reduction_rate = (
            retrieval_confusion_reduction / baseline_confusion_rate if baseline_confusion_rate else 0.0
        )
        coarse_score = coarse_by_field.get(field_name)
        coarse_ambiguity_reduction_rate = (
            coarse_score.ambiguity_reduction_rate if coarse_score is not None else 0.0
        )
        scores.append(
            DualAxisFieldScore(
                field_name=field_name,
                coarse_ambiguity_reduction_rate=coarse_ambiguity_reduction_rate,
                retrieval_confusion_rate=retrieval_confusion_rate,
                retrieval_confusion_reduction=retrieval_confusion_reduction,
                retrieval_confusion_reduction_rate=retrieval_confusion_reduction_rate,
            )
        )

    return scores


# This helper explains whether a field is semantic or provenance-oriented.
# What this does:
# - Returns a short human-readable label for one candidate field class.
# Why this exists:
# - The printed rankings should make the semantic-versus-provenance distinction explicit.
# What assumption it is making:
# - The current field split is stable enough to encode with a tiny helper.
def whisper_field_category(field_name: str) -> str:
    if field_name in WHISPER_SEMANTIC_FIELDS:
        return "semantic"
    if field_name in WHISPER_PROVENANCE_FIELDS:
        return "provenance"
    return "unknown"
