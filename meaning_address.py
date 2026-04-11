"""meaning_address.py

This module adds a tiny meaning-address routing layer on top of the existing
memory_lab sandbox.

The goal is not to replace storage. The goal is to assign small semantic
addresses that can route retrieval toward a smaller candidate neighborhood.
"""

from __future__ import annotations

from dataclasses import dataclass

from stream_generator import CivicRecord, Query, build_claim_label, build_evidence_type


# This dataclass represents a compact meaning address for one stored record.
# What this does:
# - Stores a small semantic routing handle that points back to the original record.
# Why this exists:
# - We want to test semantic routing over compact addresses without introducing a new
#   database or heavy indexing system.
# What assumption it is making:
# - A few structured semantic fields are enough to narrow a retrieval search space.
@dataclass(frozen=True)
class MeaningAddress:
    domain: str
    entity_primary: str
    context: str
    relation_type: str
    discriminator: str
    confidence: float
    source_pointer: str


# This helper names the relation type for a query or record-address experiment.
# What this does:
# - Maps the asked field into a simple memory-relation label.
# Why this exists:
# - The routing layer should distinguish drug facts from evidence-level facts.
# What assumption it is making:
# - These two relation types are enough for the current sandbox.
def relation_type_for_field(ask_field: str) -> str:
    if ask_field == "drug":
        return "drug_fact"
    return "evidence_level_fact"


# This helper creates the core biological identity string used by addresses.
# What this does:
# - Combines the main biological identity into one compact address field.
# Why this exists:
# - We want a stable primary entity key that is smaller than the whole record.
# What assumption it is making:
# - Gene plus variant is the right primary entity anchor for this first oncology address.
def build_entity_primary(gene: str, variant: str) -> str:
    return f"{gene}:{variant}"


# This helper chooses the current best discriminator field for a record address.
# What this does:
# - Picks the compact field that the routing layer should use to separate nearby records.
# Why this exists:
# - Meaning Address v1 should follow the strongest current sandbox finding.
# What assumption it is making:
# - Therapy is the cheapest single field that consistently reduces ambiguity in this arena.
def choose_record_discriminator(record: CivicRecord, ask_field: str) -> str:
    return record.drug


# This helper chooses the matching discriminator for a query address.
# What this does:
# - Builds the query-side discriminator that should line up with the record address.
# Why this exists:
# - Routing only helps if query and record addresses use the same semantic slot.
# What assumption it is making:
# - Therapy should be the default routing discriminator for Meaning Address v1.
def choose_query_discriminator(query: Query) -> str:
    return query.therapy_hint


# This helper builds a meaning address for one stored record.
# What this does:
# - Converts a full record into a compact address with a source pointer.
# Why this exists:
# - Stored records need a routeable representation above the raw storage layer.
# What assumption it is making:
# - A single address per record is enough for the first routing experiment.
def build_record_meaning_address(record: CivicRecord, ask_field: str) -> MeaningAddress:
    relation_type = relation_type_for_field(ask_field)
    return MeaningAddress(
        domain="oncology",
        entity_primary=build_entity_primary(record.gene, record.variant),
        context=record.disease,
        relation_type=relation_type,
        discriminator=choose_record_discriminator(record, ask_field=ask_field),
        confidence=0.9 if record.evidence_level in {"A", "B"} else 0.6,
        source_pointer=record.record_id,
    )


# This helper builds the query-side meaning address used for routing.
# What this does:
# - Converts a query into a compact routing target.
# Why this exists:
# - The routing layer needs a comparable address representation for lookup.
# What assumption it is making:
# - Query hints are enough to route into the right meaning neighborhood.
def build_query_meaning_address(query: Query) -> MeaningAddress:
    return MeaningAddress(
        domain="oncology",
        entity_primary=build_entity_primary(query.gene, query.variant),
        context=query.disease,
        relation_type=relation_type_for_field(query.ask_field),
        discriminator=choose_query_discriminator(query),
        confidence=0.8,
        source_pointer=query.query_id,
    )


# This helper tests whether a stored address shares the query's semantic prefix.
# What this does:
# - Checks the broad meaning neighborhood before applying a discriminator.
# Why this exists:
# - Prefix routing should narrow the candidate set before finer disambiguation.
# What assumption it is making:
# - Domain, entity, context, and relation type define a useful first routing prefix.
def meaning_prefix_matches(record_address: MeaningAddress, query_address: MeaningAddress) -> bool:
    return (
        record_address.domain == query_address.domain
        and record_address.entity_primary == query_address.entity_primary
        and record_address.context == query_address.context
        and record_address.relation_type == query_address.relation_type
    )


# This helper tests whether two addresses also agree on the discriminator.
# What this does:
# - Applies a finer match after the broad prefix route.
# Why this exists:
# - The experiment is about whether compact extra fields reduce ambiguity.
# What assumption it is making:
# - Exact discriminator agreement is a useful first routing rule.
def meaning_discriminator_matches(record_address: MeaningAddress, query_address: MeaningAddress) -> bool:
    return record_address.discriminator == query_address.discriminator
