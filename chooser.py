"""chooser.py

Route chooser logic and stress-test experiment for the memory_lab baseline experiment.
"""

from __future__ import annotations

from typing import Dict, List

from stream_generator import generate_recall_episodes
from data_types import (
    RouteChooserRow,
    RouteExplanationRow,
    RouteStressRow,
)
from memory_policies import SaveAllPolicy
from routing import (
    evaluate_route_comparison_episode,
    summarize_route_comparison_results,
)


# This helper explains the current chooser rule in plain terms.
# What this does:
# - Returns the route that the chooser would pick plus a short reason for why it was picked.
# Why this exists:
# - We want the printed explainer to mirror the actual chooser logic exactly instead of
#   inventing a second interpretation layer.
# What assumption it is making:
# - A short three-case explanation is enough for the current route policy.
def explain_route_choice(summary: Dict[str, float]) -> tuple[str, bool, str]:
    identity_candidates = summary["average_identity_candidates"]
    identity_confusion_rate = summary["identity_confusion_rate"]
    identity_therapy_confusion_rate = summary["identity_therapy_confusion_rate"]
    two_stage_stage_1_candidates = summary["average_two_stage_stage_1_candidates"]
    two_stage_confusion_rate = summary["two_stage_confusion_rate"]

    coarse_routing_paid_for_itself = (
        two_stage_stage_1_candidates < identity_candidates
        and two_stage_confusion_rate <= identity_therapy_confusion_rate
    )

    if identity_candidates <= 2.0 and identity_confusion_rate <= 0.10:
        return (
            "identity_only",
            coarse_routing_paid_for_itself,
            "identity neighborhoods were already small and confusion stayed low, so extra routing was not worth paying for.",
        )

    if coarse_routing_paid_for_itself and two_stage_confusion_rate < identity_therapy_confusion_rate:
        return (
            "identity_then_evidence_type_then_therapy",
            coarse_routing_paid_for_itself,
            "evidence_type noticeably shrank the identity neighborhood and the staged route reduced confusion beyond identity plus therapy.",
        )

    return (
        "identity_plus_therapy",
        coarse_routing_paid_for_itself,
        "identity alone was not clean enough, but the extra evidence_type split did not buy enough additional value over therapy-first tie-breaking.",
    )


# This helper turns one route-comparison summary into a chooser row.
# What this does:
# - Applies a small readable rule that picks a route strategy from measured confusion and candidate counts.
# Why this exists:
# - Database Whisper should decide when staged routing is worth using, not just print raw metrics.
# What assumption it is making:
# - Low ambiguity should favor direct routing, while high ambiguity should favor staged routing.
def build_route_chooser_row(
    task_type: str,
    distractor_level: str,
    summary: Dict[str, float],
) -> RouteChooserRow:
    identity_candidates = summary["average_identity_candidates"]
    identity_confusion_rate = summary["identity_confusion_rate"]
    identity_therapy_candidates = summary["average_identity_therapy_candidates"]
    identity_therapy_confusion_rate = summary["identity_therapy_confusion_rate"]
    two_stage_stage_1_candidates = summary["average_two_stage_stage_1_candidates"]
    two_stage_stage_2_candidates = summary["average_two_stage_stage_2_candidates"]
    two_stage_confusion_rate = summary["two_stage_confusion_rate"]

    chosen_route, coarse_routing_paid_for_itself, _reason = explain_route_choice(summary)

    return RouteChooserRow(
        task_type=task_type,
        distractor_level=distractor_level,
        identity_candidates=identity_candidates,
        identity_confusion_rate=identity_confusion_rate,
        identity_therapy_candidates=identity_therapy_candidates,
        identity_therapy_confusion_rate=identity_therapy_confusion_rate,
        two_stage_stage_1_candidates=two_stage_stage_1_candidates,
        two_stage_stage_2_candidates=two_stage_stage_2_candidates,
        two_stage_confusion_rate=two_stage_confusion_rate,
        coarse_routing_paid_for_itself=coarse_routing_paid_for_itself,
        chosen_route=chosen_route,
    )


# This helper turns one route-comparison summary into a stress-test row.
# What this does:
# - Adds a small route-cost proxy and a simple stop rule for deciding whether staging is worth it.
# Why this exists:
# - The chooser should not only know which route wins, but whether the win justifies the extra routing work.
# What assumption it is making:
# - Summing average candidates touched across route stages is a good enough first cost proxy.
def build_route_stress_row(
    task_type: str,
    distractor_level: str,
    distractor_count: int,
    record_count: int,
    summary: Dict[str, float],
) -> RouteStressRow:
    identity_candidates = summary["average_identity_candidates"]
    identity_confusion_rate = summary["identity_confusion_rate"]
    identity_cost = identity_candidates

    identity_therapy_candidates = summary["average_identity_therapy_candidates"]
    identity_therapy_confusion_rate = summary["identity_therapy_confusion_rate"]
    identity_therapy_cost = identity_candidates + identity_therapy_candidates

    two_stage_stage_1_candidates = summary["average_two_stage_stage_1_candidates"]
    two_stage_stage_2_candidates = summary["average_two_stage_stage_2_candidates"]
    two_stage_confusion_rate = summary["two_stage_confusion_rate"]
    two_stage_cost = identity_candidates + two_stage_stage_1_candidates + two_stage_stage_2_candidates

    confusion_gain_from_two_stage = identity_therapy_confusion_rate - two_stage_confusion_rate
    added_cost_for_two_stage = two_stage_cost - identity_therapy_cost
    pay_for_two_stage = (
        confusion_gain_from_two_stage >= 0.05
        and two_stage_confusion_rate < identity_therapy_confusion_rate
        and added_cost_for_two_stage <= 3.0
    )
    stop_at_identity_therapy = not pay_for_two_stage

    return RouteStressRow(
        task_type=task_type,
        distractor_level=distractor_level,
        distractor_count=distractor_count,
        record_count=record_count,
        identity_candidates=identity_candidates,
        identity_confusion_rate=identity_confusion_rate,
        identity_cost=identity_cost,
        identity_therapy_candidates=identity_therapy_candidates,
        identity_therapy_confusion_rate=identity_therapy_confusion_rate,
        identity_therapy_cost=identity_therapy_cost,
        two_stage_stage_1_candidates=two_stage_stage_1_candidates,
        two_stage_stage_2_candidates=two_stage_stage_2_candidates,
        two_stage_confusion_rate=two_stage_confusion_rate,
        two_stage_cost=two_stage_cost,
        stop_at_identity_therapy=stop_at_identity_therapy,
        pay_for_two_stage=pay_for_two_stage,
    )


# This helper runs the route stress test across multiple ambiguity and scale conditions.
# What this does:
# - Sweeps task type, distractor level, distractor count, and record pool size on a full-memory substrate.
# Why this exists:
# - We want to validate the chooser policy before adding more ontology or data complexity.
# What assumption it is making:
# - SaveAll is the cleanest route-testing substrate because route errors are not mixed with omission.
def run_route_stress_experiment(seed: int, episodes: int) -> List[RouteStressRow]:
    stress_rows: List[RouteStressRow] = []
    chooser_policy = SaveAllPolicy()

    for task_type in ["direct_recall", "late_relevance_recall"]:
        for distractor_level in ["medium", "collision", "ambiguity"]:
            for distractor_count in [3, 5, 7]:
                for record_count in [40, 80, 120]:
                    generated_episodes = generate_recall_episodes(
                        task_type=task_type,
                        record_count=record_count,
                        episode_count=episodes,
                        distractor_level=distractor_level,
                        distractor_count=distractor_count,
                        data_source="synthetic",
                        seed=seed,
                    )
                    route_comparison_results = [
                        evaluate_route_comparison_episode(policy=chooser_policy, episode=episode)
                        for episode in generated_episodes
                    ]
                    summary = summarize_route_comparison_results(route_comparison_results)
                    stress_rows.append(
                        build_route_stress_row(
                            task_type=task_type,
                            distractor_level=distractor_level,
                            distractor_count=distractor_count,
                            record_count=record_count,
                            summary=summary,
                        )
                    )

    return stress_rows


# This helper runs the route chooser experiment across task types and ambiguity levels.
# What this does:
# - Evaluates the three route strategies on a fixed full-memory substrate so route quality is isolated.
# Why this exists:
# - The new question is when to use a staged route, not whether a storage policy kept the record.
# What assumption it is making:
# - SaveAll is the cleanest substrate for studying route choice because it does not add omission effects.
def run_route_chooser_experiment(
    task_types: List[str],
    record_count: int,
    episodes: int,
    distractor_count: int,
    data_source: str,
    seed: int,
) -> tuple[List[RouteChooserRow], List[RouteExplanationRow]]:
    chooser_rows: List[RouteChooserRow] = []
    explanation_rows: List[RouteExplanationRow] = []
    chooser_policy = SaveAllPolicy()

    for current_task_type in task_types:
        for distractor_level in ["easy", "medium", "collision", "ambiguity"]:
            generated_episodes = generate_recall_episodes(
                task_type=current_task_type,
                record_count=record_count,
                episode_count=episodes,
                distractor_level=distractor_level,
                distractor_count=distractor_count,
                data_source=data_source,
                seed=seed,
            )
            route_comparison_results = [
                evaluate_route_comparison_episode(policy=chooser_policy, episode=episode)
                for episode in generated_episodes
            ]
            summary = summarize_route_comparison_results(route_comparison_results)
            chooser_row = build_route_chooser_row(
                task_type=current_task_type,
                distractor_level=distractor_level,
                summary=summary,
            )
            chooser_rows.append(chooser_row)

            _chosen_route, _coarse_paid, reason = explain_route_choice(summary)
            for episode, route_result in zip(generated_episodes, route_comparison_results):
                explanation_rows.append(
                    RouteExplanationRow(
                        task_type=current_task_type,
                        distractor_level=distractor_level,
                        episode_id=episode.episode_id,
                        identity_neighborhood_size=route_result.identity_candidate_count,
                        evidence_type_candidate_count=route_result.two_stage_stage1_candidate_count,
                        therapy_candidate_count=route_result.two_stage_stage2_candidate_count,
                        chosen_route=chooser_row.chosen_route,
                        reason=reason,
                    )
                )

    return chooser_rows, explanation_rows
