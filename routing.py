"""routing.py

Routing-layer lookup helpers and evaluation for the memory_lab baseline experiment.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from meaning_address import (
    MeaningAddress,
    build_query_meaning_address,
    build_record_meaning_address,
    meaning_discriminator_matches,
    meaning_prefix_matches,
)
from stream_generator import CivicRecord, Query, RecallEpisode
from data_types import RouteComparisonResult, RoutingResult
from retrieval import full_records_for_policy, query_identity, record_identity


# This helper runs the existing flat lookup and exposes its candidate set.
# What this does:
# - Finds all same-identity records and returns the first-answer baseline.
# Why this exists:
# - We need a direct baseline to compare against address-guided routing.
# What assumption it is making:
# - The current flat retriever is equivalent to "first same-identity match wins."
def flat_lookup_with_candidates(records: List[CivicRecord], query: Query) -> tuple[Optional[str], List[CivicRecord]]:
    candidates = [record for record in records if record_identity(record) == query_identity(query)]
    if not candidates:
        return None, []
    return getattr(candidates[0], query.ask_field), candidates


# This helper runs the identity-only routing baseline explicitly.
# What this does:
# - Uses only the core structured identity neighborhood for routing.
# Why this exists:
# - We want a named one-stage baseline that is simpler than the current Meaning Address route.
# What assumption it is making:
# - Gene, variant, and disease define the first routing neighborhood.
def identity_only_lookup_with_candidates(records: List[CivicRecord], query: Query) -> tuple[Optional[str], List[CivicRecord]]:
    return flat_lookup_with_candidates(records=records, query=query)


# This helper runs a one-stage identity-plus-therapy route.
# What this does:
# - Narrows the identity neighborhood with therapy when the query carries that hint.
# Why this exists:
# - Therapy is the strongest current one-step retrieval discriminator in the sandbox.
# What assumption it is making:
# - Therapy should be used only as an added filter over the identity neighborhood, not as a replacement for it.
def identity_plus_therapy_lookup_with_candidates(
    records: List[CivicRecord],
    query: Query,
) -> tuple[Optional[str], List[CivicRecord]]:
    identity_candidates = [record for record in records if record_identity(record) == query_identity(query)]
    if not identity_candidates:
        return None, []

    therapy_candidates = [
        record for record in identity_candidates if record.drug == query.therapy_hint
    ]
    final_candidates = therapy_candidates if therapy_candidates else identity_candidates
    return getattr(final_candidates[0], query.ask_field), final_candidates


# This helper builds meaning addresses for the current full-record layer.
# What this does:
# - Converts each full record into the compact routing representation used in v0.
# Why this exists:
# - The routing experiment needs addresses side by side with the raw records.
# What assumption it is making:
# - One address per record is enough for this first prefix-routing test.
def build_addresses_for_records(records: List[CivicRecord], query: Query) -> List[MeaningAddress]:
    return [build_record_meaning_address(record, ask_field=query.ask_field) for record in records]


# This helper runs meaning-address-guided routing before final lookup.
# What this does:
# - Uses prefix routing and then discriminator routing to shrink the candidate set.
# Why this exists:
# - We want to know whether compact meaning addresses reduce search ambiguity.
# What assumption it is making:
# - Prefix plus discriminator is enough for the first address-guided routing experiment.
def routed_lookup_with_candidates(records: List[CivicRecord], query: Query) -> tuple[Optional[str], List[CivicRecord]]:
    query_address = build_query_meaning_address(query)
    record_addresses = build_addresses_for_records(records=records, query=query)
    address_by_source = {address.source_pointer: address for address in record_addresses}

    prefix_candidates = [
        record
        for record in records
        if meaning_prefix_matches(address_by_source[record.record_id], query_address)
    ]

    discriminator_candidates = [
        record
        for record in prefix_candidates
        if meaning_discriminator_matches(address_by_source[record.record_id], query_address)
    ]

    final_candidates = discriminator_candidates if discriminator_candidates else prefix_candidates
    if not final_candidates:
        return None, []
    return getattr(final_candidates[0], query.ask_field), final_candidates


# This helper runs the new two-stage semantic route.
# What this does:
# - First narrows by identity plus evidence_type, then uses therapy to break any remaining tie.
# Why this exists:
# - The dual-axis results suggest evidence_type is a strong coarse splitter while therapy is the final tie-breaker.
# What assumption it is making:
# - A coarse semantic filter followed by a targeted discriminator is the right routing shape for this domain.
def two_stage_semantic_lookup_with_candidates(
    records: List[CivicRecord],
    query: Query,
) -> tuple[Optional[str], List[CivicRecord], List[CivicRecord]]:
    identity_candidates = [record for record in records if record_identity(record) == query_identity(query)]
    if not identity_candidates:
        return None, [], []

    stage_1_candidates = [
        record
        for record in identity_candidates
        if f"{record.evidence_direction}:{record.evidence_level}" == query.evidence_type_hint
    ]
    if not stage_1_candidates:
        stage_1_candidates = identity_candidates

    stage_2_candidates = [record for record in stage_1_candidates if record.drug == query.therapy_hint]
    if not stage_2_candidates:
        stage_2_candidates = stage_1_candidates

    return getattr(stage_2_candidates[0], query.ask_field), stage_1_candidates, stage_2_candidates


# This helper turns a prediction into a route-side confusion flag.
# What this does:
# - Labels a route as confused when it returns a wrong non-empty answer.
# Why this exists:
# - Route comparison should focus on mistaken identity rather than misses.
# What assumption it is making:
# - A wrong answer is the right route-level failure signal for this ambiguity sandbox.
def is_route_confusion(predicted_answer: Optional[str], expected_answer: str) -> bool:
    return predicted_answer is not None and predicted_answer != expected_answer


# This helper evaluates the routing layer on one episode.
# What this does:
# - Compares flat lookup against meaning-address-guided lookup on the same retained records.
# Why this exists:
# - The new feature is a routing layer, so we need a direct side-by-side comparison.
# What assumption it is making:
# - Candidate-set reduction and confusion reduction are the main signals that matter here.
def evaluate_routing_episode(policy, episode: RecallEpisode) -> RoutingResult:
    records = full_records_for_policy(policy=policy, episode=episode)
    flat_answer, flat_candidates = flat_lookup_with_candidates(records=records, query=episode.query)
    routed_answer, routed_candidates = routed_lookup_with_candidates(records=records, query=episode.query)

    flat_confusion = flat_answer is not None and flat_answer != episode.query.answer
    routed_confusion = routed_answer is not None and routed_answer != episode.query.answer

    return RoutingResult(
        policy_name=policy.name,
        task_type=episode.task_type,
        flat_candidate_count=len(flat_candidates),
        routed_candidate_count=len(routed_candidates),
        flat_confusion=flat_confusion,
        routed_confusion=routed_confusion,
    )


# This helper evaluates several explicit routing styles on one episode.
# What this does:
# - Compares flat lookup, identity-only, identity-plus-therapy, current Meaning Address routing,
#   and the new two-stage semantic route on the same retained records.
# Why this exists:
# - We want to know whether coarse splitter first, final discriminator second beats simpler routes.
# What assumption it is making:
# - The current full-record layer is the right substrate for route-only comparisons.
def evaluate_route_comparison_episode(policy, episode: RecallEpisode) -> RouteComparisonResult:
    records = full_records_for_policy(policy=policy, episode=episode)
    flat_answer, flat_candidates = flat_lookup_with_candidates(records=records, query=episode.query)
    identity_answer, identity_candidates = identity_only_lookup_with_candidates(records=records, query=episode.query)
    identity_therapy_answer, identity_therapy_candidates = identity_plus_therapy_lookup_with_candidates(
        records=records,
        query=episode.query,
    )
    meaning_answer, meaning_candidates = routed_lookup_with_candidates(records=records, query=episode.query)
    two_stage_answer, stage_1_candidates, stage_2_candidates = two_stage_semantic_lookup_with_candidates(
        records=records,
        query=episode.query,
    )

    return RouteComparisonResult(
        policy_name=policy.name,
        task_type=episode.task_type,
        flat_candidate_count=len(flat_candidates),
        identity_candidate_count=len(identity_candidates),
        identity_therapy_candidate_count=len(identity_therapy_candidates),
        meaning_candidate_count=len(meaning_candidates),
        two_stage_stage1_candidate_count=len(stage_1_candidates),
        two_stage_stage2_candidate_count=len(stage_2_candidates),
        flat_confusion=is_route_confusion(flat_answer, episode.query.answer),
        identity_confusion=is_route_confusion(identity_answer, episode.query.answer),
        identity_therapy_confusion=is_route_confusion(identity_therapy_answer, episode.query.answer),
        meaning_confusion=is_route_confusion(meaning_answer, episode.query.answer),
        two_stage_confusion=is_route_confusion(two_stage_answer, episode.query.answer),
    )


# This helper summarizes routing-layer behavior across many episodes.
# What this does:
# - Aggregates candidate counts and confusion rates for flat versus routed lookup.
# Why this exists:
# - The routing layer needs its own small scorecard separate from storage-policy metrics.
# What assumption it is making:
# - Average candidate count and confusion rate are enough for Meaning Address v0.
def summarize_routing_results(results: List[RoutingResult]) -> Dict[str, float]:
    total = len(results)
    average_flat_candidates = sum(result.flat_candidate_count for result in results) / total if total else 0.0
    average_routed_candidates = sum(result.routed_candidate_count for result in results) / total if total else 0.0
    flat_confusion_rate = sum(result.flat_confusion for result in results) / total if total else 0.0
    routed_confusion_rate = sum(result.routed_confusion for result in results) / total if total else 0.0

    return {
        "average_flat_candidates": average_flat_candidates,
        "average_routed_candidates": average_routed_candidates,
        "flat_confusion_rate": flat_confusion_rate,
        "routed_confusion_rate": routed_confusion_rate,
        "candidate_reduction": average_flat_candidates - average_routed_candidates,
    }


# This helper summarizes the multi-route comparison across many episodes.
# What this does:
# - Aggregates candidate counts and confusion rates for all route variants we want to compare.
# Why this exists:
# - The two-stage routing experiment needs a fuller scorecard than the original flat-vs-routed summary.
# What assumption it is making:
# - Average candidate counts and confusion rates are enough to compare route quality in v1.
def summarize_route_comparison_results(results: List[RouteComparisonResult]) -> Dict[str, float]:
    total = len(results)
    if not total:
        return {}

    average_flat_candidates = sum(result.flat_candidate_count for result in results) / total
    average_identity_candidates = sum(result.identity_candidate_count for result in results) / total
    average_identity_therapy_candidates = sum(result.identity_therapy_candidate_count for result in results) / total
    average_meaning_candidates = sum(result.meaning_candidate_count for result in results) / total
    average_two_stage_stage_1_candidates = (
        sum(result.two_stage_stage1_candidate_count for result in results) / total
    )
    average_two_stage_stage_2_candidates = (
        sum(result.two_stage_stage2_candidate_count for result in results) / total
    )
    flat_confusion_rate = sum(result.flat_confusion for result in results) / total
    identity_confusion_rate = sum(result.identity_confusion for result in results) / total
    identity_therapy_confusion_rate = sum(result.identity_therapy_confusion for result in results) / total
    meaning_confusion_rate = sum(result.meaning_confusion for result in results) / total
    two_stage_confusion_rate = sum(result.two_stage_confusion for result in results) / total

    return {
        "average_flat_candidates": average_flat_candidates,
        "average_identity_candidates": average_identity_candidates,
        "average_identity_therapy_candidates": average_identity_therapy_candidates,
        "average_meaning_candidates": average_meaning_candidates,
        "average_two_stage_stage_1_candidates": average_two_stage_stage_1_candidates,
        "average_two_stage_stage_2_candidates": average_two_stage_stage_2_candidates,
        "flat_confusion_rate": flat_confusion_rate,
        "identity_confusion_rate": identity_confusion_rate,
        "identity_therapy_confusion_rate": identity_therapy_confusion_rate,
        "meaning_confusion_rate": meaning_confusion_rate,
        "two_stage_confusion_rate": two_stage_confusion_rate,
        "two_stage_beats_flat": two_stage_confusion_rate < flat_confusion_rate,
        "two_stage_beats_identity_therapy": two_stage_confusion_rate < identity_therapy_confusion_rate,
        "two_stage_beats_meaning": two_stage_confusion_rate < meaning_confusion_rate,
    }
