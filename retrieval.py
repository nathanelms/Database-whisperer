"""retrieval.py

Retrieval helpers for the memory_lab baseline experiment.
"""

from __future__ import annotations

from typing import List, Optional

from stream_generator import CivicRecord, Query, RecallEpisode
from data_types import MemoryStore, RecordStub, StubMemoryStore
from memory_policies import TieredMemoryPolicy, StubMemoryPolicy


# This helper gives a record a stable structured identity.
# What this does:
# - Returns the fields we currently treat as the exact lookup key.
# Why this exists:
# - Both memory and retrieval logic become more readable when identity handling is in one place.
# What assumption it is making:
# - Gene, variant, and disease together identify the fact we are trying to recall.
def record_identity(record: CivicRecord) -> tuple[str, str, str]:
    return (record.gene, record.variant, record.disease)


# This helper does the same identity construction for queries.
# What this does:
# - Converts a query into the same structured key format used for records.
# Why this exists:
# - Matching stays exact and easy to inspect when both sides use one helper format.
# What assumption it is making:
# - The query uses the same identity fields that defined the stored fact.
def query_identity(query: Query) -> tuple[str, str, str]:
    return (query.gene, query.variant, query.disease)


# This retrieval helper performs exact structured lookup against one memory list.
# What this does:
# - Finds the stored record whose identity exactly matches the query and returns the requested field.
# Why this exists:
# - Both one-tier and two-tier retrieval stay simple when exact matching lives in one helper.
# What assumption it is making:
# - If the right record is present in a searched tier, exact key-based lookup is enough.
def exact_structured_lookup(records: List[CivicRecord], query: Query) -> Optional[str]:
    query_key = query_identity(query)

    for record in records:
        if record_identity(record) == query_key:
            return getattr(record, query.ask_field)

    return None


# This retrieval function performs exact structured lookup for one-tier storage.
# What this does:
# - Searches a single stored-record list and returns the requested field on an exact match.
# Why this exists:
# - `SaveAll` and `RuleBasedSalience` still use a single-tier storage design in v1.
# What assumption it is making:
# - Single-list storage is enough for the simpler baseline policies.
def exact_structured_retrieval(stored_records: List[CivicRecord], query: Query) -> Optional[str]:
    return exact_structured_lookup(records=stored_records, query=query)


# This retrieval function performs exact structured lookup for the tiered policy.
# What this does:
# - Searches durable memory first and falls back to short-term memory if needed.
# Why this exists:
# - The user asked for a retrieval path that prefers promoted memory without losing access
#   to the full episode context.
# What assumption it is making:
# - Durable-first, short-term-second is a useful first approximation of tiered recall.
def exact_tiered_retrieval(memory_store: MemoryStore, query: Query) -> Optional[str]:
    durable_answer = exact_structured_lookup(records=memory_store.durable_records, query=query)
    if durable_answer is not None:
        return durable_answer

    return exact_structured_lookup(records=memory_store.short_term_records, query=query)


# This helper checks whether a stub matches the query identity.
# What this does:
# - Compares the query to the stub using the same identity fields used elsewhere.
# Why this exists:
# - Stub retrieval should remain as simple and legible as the full-record retrieval path.
# What assumption it is making:
# - Gene, variant, and disease are enough to match a stub back to a later query.
# This helper scores how well a stub matches a query.
# What this does:
# - Uses the fields retained in the stub to measure how specifically it matches the query hints.
# Why this exists:
# - Different stub schemas should live or die by the extra disambiguation fields they keep.
# What assumption it is making:
# - A higher count of matched retained fields is a useful first ranking rule.
def stub_match_score(stub: RecordStub, query: Query) -> int:
    if (stub.gene, stub.variant, stub.disease) != query_identity(query):
        return -1

    score = 3

    if stub.evidence_type is not None and stub.evidence_type == query.evidence_type_hint:
        score += 1
    if stub.evidence_level is not None and stub.evidence_level == query.evidence_level_hint:
        score += 1
    if stub.direction is not None and stub.direction == query.evidence_direction_hint:
        score += 1
    if stub.therapy is not None and stub.therapy == query.therapy_hint:
        score += 1
    if stub.compressed_claim_label is not None and stub.compressed_claim_label == query.claim_label_hint:
        score += 1

    return score


# This retrieval function performs full-then-stub search for the stub policy.
# What this does:
# - Searches durable full memory first and then searches stubs if full detail is missing.
# Why this exists:
# - The user asked for retrieval that first prefers retained detail and then falls back to
#   a stub that indicates the record was compressed rather than fully stored.
# What assumption it is making:
# - A stub hit is useful even though it cannot answer the query's requested field directly.
def exact_stub_retrieval(memory_store: StubMemoryStore, query: Query) -> tuple[Optional[str], bool, Optional[str]]:
    durable_matches = [
        record for record in memory_store.durable_records if record_identity(record) == query_identity(query)
    ]
    if len(durable_matches) == 1:
        durable_record = durable_matches[0]
        return getattr(durable_record, query.ask_field), False, durable_record.record_id

    # When the durable layer has multiple same-identity records, we treat that as an
    # ambiguity case and let the stub layer attempt disambiguation.
    # This keeps the retrieval order "full first, then stubs" while giving the retained
    # stub fields a real chance to prove their value.

    scored_stubs = [(stub_match_score(stub=stub, query=query), stub) for stub in memory_store.stubs]
    scored_stubs = [(score, stub) for score, stub in scored_stubs if score >= 0]
    if not scored_stubs:
        if durable_matches:
            fallback_record = durable_matches[0]
            return getattr(fallback_record, query.ask_field), False, fallback_record.record_id
        return None, False, None

    # We sort by score descending so richer retained fields can break identity ties.
    best_score, best_stub = sorted(scored_stubs, key=lambda item: item[0], reverse=True)[0]
    stub_message = (
        f"STUB_MATCH: matched {best_stub.record_id}; "
        f"score={best_score}; "
        f"full detail not retained; source_pointer={best_stub.source_pointer}"
    )
    return stub_message, True, best_stub.source_pointer


# This helper returns the full raw records available to a policy before any stub fallback.
# What this does:
# - Collects the full-record layer that the routing experiment should operate over.
# Why this exists:
# - Meaning addresses are a routing layer above current raw storage, not a new storage policy.
# What assumption it is making:
# - The raw full-record layer is the right substrate for the first routing experiment.
def full_records_for_policy(policy, episode: RecallEpisode) -> List[CivicRecord]:
    if isinstance(policy, TieredMemoryPolicy):
        memory_store = policy.build_memory_store(episode.stream_records)
        return list(memory_store.short_term_records)
    if isinstance(policy, StubMemoryPolicy):
        memory_store = policy.build_memory_store(episode.stream_records)
        return list(memory_store.durable_records)
    return policy.select_records(episode.stream_records)
