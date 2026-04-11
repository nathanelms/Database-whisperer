"""memory_policies.py

Memory selection policies and stub-related helpers for the memory_lab baseline experiment.
"""

from __future__ import annotations

from typing import List

from stream_generator import CivicRecord
from data_types import (
    MemoryStore,
    RecordStub,
    StubMemoryStore,
    StubSchema,
)


# This policy stores every seen record.
# What this does:
# - Keeps the full stream untouched.
# Why this exists:
# - `SaveAll` is the simplest baseline and gives us a reference point for maximum recall.
# What assumption it is making:
# - Retaining everything should reduce omission errors, even if it may increase clutter.
class SaveAllPolicy:
    name = "SaveAll"

    def select_records(self, stream_records: List[CivicRecord]) -> List[CivicRecord]:
        return list(stream_records)


# This policy keeps only records that look important according to simple rules.
# What this does:
# - Selects records with stronger evidence or stronger directional support.
# Why this exists:
# - We need a first non-trivial policy that can trade storage size against possible fact loss.
# What assumption it is making:
# - In this synthetic domain, higher evidence levels and supportive findings are more likely
#   to be worth storing for future use.
class RuleBasedSaliencePolicy:
    name = "RuleBasedSalience"

    def select_records(self, stream_records: List[CivicRecord]) -> List[CivicRecord]:
        selected: List[CivicRecord] = []

        for record in stream_records:
            is_high_evidence = record.evidence_level in {"A", "B"}
            is_supportive = record.evidence_direction == "supports"

            if is_high_evidence or is_supportive:
                selected.append(record)

        return selected


# This policy keeps every record in short-term memory, then promotes only some.
# What this does:
# - Stores the full episode stream in short-term memory and a filtered subset in durable memory.
# Why this exists:
# - We want a simple middle ground between keeping everything forever and filtering too hard.
# What assumption it is making:
# - Episode-local availability can rescue late-hidden relevance, while durable promotion
#   should still remain selective.
class TieredMemoryPolicy:
    name = "TieredMemoryPolicy"

    def build_memory_store(self, stream_records: List[CivicRecord]) -> MemoryStore:
        short_term_records = list(stream_records)
        durable_records: List[CivicRecord] = []

        for record in stream_records:
            is_high_evidence = record.evidence_level in {"A", "B"}
            is_supportive = record.evidence_direction == "supports"
            shares_context = record.disease in {"Non-Small Cell Lung Cancer", "Breast Cancer", "Colorectal Cancer"}

            # The promotion rule is intentionally simple and readable.
            # We promote obviously strong records plus some common-disease context records.
            if is_high_evidence or is_supportive or shares_context:
                durable_records.append(record)

        return MemoryStore(
            short_term_records=short_term_records,
            durable_records=durable_records,
        )


# This policy keeps some full records and compresses the rest into stubs.
# What this does:
# - Retains salient records as full durable memories and converts lower-salience records
#   into lightweight indexed references.
# Why this exists:
# - We want to test whether a stub can preserve future retrievability without paying the
#   storage cost of keeping every full record.
# What assumption it is making:
# - A small identity-heavy stub is enough to preserve "something relevant existed here."
class StubMemoryPolicy:
    def __init__(self, stub_schema: StubSchema) -> None:
        self.stub_schema = stub_schema
        self.name = f"StubMemoryPolicy[{stub_schema.name}]"

    def build_memory_store(self, stream_records: List[CivicRecord]) -> StubMemoryStore:
        durable_records: List[CivicRecord] = []
        stubs: List[RecordStub] = []

        for record in stream_records:
            is_high_evidence = record.evidence_level in {"A", "B"}
            is_supportive = record.evidence_direction == "supports"

            if is_high_evidence or is_supportive:
                durable_records.append(record)
            else:
                stubs.append(self.build_stub(record))

        return StubMemoryStore(
            durable_records=durable_records,
            stubs=stubs,
        )

    # This helper compresses one full record into a stub.
    # What this does:
    # - Builds a small retrieval-oriented reference entry from a richer evidence record.
    # Why this exists:
    # - The policy should keep an address to forgotten facts instead of dropping them entirely.
    # What assumption it is making:
    # - Record identity plus a few hints are enough to support later stub matching.
    def build_stub(self, record: CivicRecord) -> RecordStub:
        evidence_type = None
        evidence_level = None
        direction = None
        therapy = None
        retrieval_hints = [f"askable_fields=drug,evidence_level"]

        if self.stub_schema.include_evidence_type:
            evidence_type = f"{record.evidence_direction}:{record.evidence_level}"
            retrieval_hints.append(f"evidence_type={evidence_type}")

        if self.stub_schema.include_evidence_level:
            evidence_level = record.evidence_level
            retrieval_hints.append(f"evidence_level={evidence_level}")

        if self.stub_schema.include_direction:
            direction = record.evidence_direction
            retrieval_hints.append(f"direction={direction}")

        if self.stub_schema.include_therapy:
            therapy = record.drug
            retrieval_hints.append(f"therapy={therapy}")

        if self.stub_schema.include_source_id_hint:
            retrieval_hints.append(f"source_id={record.record_id}")

        return RecordStub(
            record_id=record.record_id,
            gene=record.gene,
            variant=record.variant,
            disease=record.disease,
            evidence_type=evidence_type,
            evidence_level=evidence_level,
            direction=direction,
            therapy=therapy,
            compressed_claim_label=None,
            retrieval_hints=retrieval_hints,
            source_pointer=record.record_id,
            status="compressed_stub",
        )


# These are the stub designs we want to compare.
# What this does:
# - Defines several readable schema choices for compressed memory.
# Why this exists:
# - The new research question is about how much usefulness different stub field sets preserve.
# What assumption it is making:
# - These four designs are enough to start exploring the tradeoff without overengineering.
STUB_SCHEMAS = [
    StubSchema(
        name="minimal_identity",
        include_evidence_type=False,
        include_evidence_level=False,
        include_direction=False,
        include_therapy=False,
        include_source_id_hint=False,
        discriminator_fields=[],
    ),
    StubSchema(
        name="minimal_identity_plus_therapy",
        include_evidence_type=False,
        include_evidence_level=False,
        include_direction=False,
        include_therapy=True,
        include_source_id_hint=False,
        discriminator_fields=["therapy"],
    ),
    StubSchema(
        name="minimal_identity_plus_evidence_type",
        include_evidence_type=True,
        include_evidence_level=False,
        include_direction=False,
        include_therapy=False,
        include_source_id_hint=False,
        discriminator_fields=["evidence_type"],
    ),
    StubSchema(
        name="minimal_identity_plus_evidence_level",
        include_evidence_type=False,
        include_evidence_level=True,
        include_direction=False,
        include_therapy=False,
        include_source_id_hint=False,
        discriminator_fields=["evidence_level"],
    ),
    StubSchema(
        name="minimal_identity_plus_direction",
        include_evidence_type=False,
        include_evidence_level=False,
        include_direction=True,
        include_therapy=False,
        include_source_id_hint=False,
        discriminator_fields=["direction"],
    ),
    StubSchema(
        name="minimal_identity_plus_source",
        include_evidence_type=False,
        include_evidence_level=False,
        include_direction=False,
        include_therapy=False,
        include_source_id_hint=True,
        discriminator_fields=["source"],
    ),
    StubSchema(
        name="minimal_identity_plus_therapy_plus_evidence_type",
        include_evidence_type=True,
        include_evidence_level=False,
        include_direction=False,
        include_therapy=True,
        include_source_id_hint=False,
        discriminator_fields=["therapy", "evidence_type"],
    ),
    StubSchema(
        name="minimal_identity_plus_therapy_plus_evidence_level",
        include_evidence_type=False,
        include_evidence_level=True,
        include_direction=False,
        include_therapy=True,
        include_source_id_hint=False,
        discriminator_fields=["therapy", "evidence_level"],
    ),
    StubSchema(
        name="minimal_identity_plus_therapy_plus_direction",
        include_evidence_type=False,
        include_evidence_level=False,
        include_direction=True,
        include_therapy=True,
        include_source_id_hint=False,
        discriminator_fields=["therapy", "direction"],
    ),
    StubSchema(
        name="minimal_identity_plus_therapy_plus_source",
        include_evidence_type=False,
        include_evidence_level=False,
        include_direction=False,
        include_therapy=True,
        include_source_id_hint=True,
        discriminator_fields=["therapy", "source"],
    ),
]


# This helper estimates the size of one stub using a field-count proxy.
# What this does:
# - Counts how many retrieval-relevant fields are retained in the compressed stub.
# Why this exists:
# - The new research question is about usefulness per unit of retained stub memory.
# What assumption it is making:
# - Counting populated fields is a good enough first proxy for stub size.
def estimate_stub_field_count(stub: RecordStub) -> int:
    field_count = 0

    # These are the always-retained core fields that make the stub addressable.
    field_count += 1  # record_id
    field_count += 1  # gene
    field_count += 1  # variant
    field_count += 1  # disease
    field_count += 1  # source_pointer
    field_count += 1  # status

    if stub.evidence_type is not None:
        field_count += 1
    if stub.evidence_level is not None:
        field_count += 1
    if stub.direction is not None:
        field_count += 1
    if stub.therapy is not None:
        field_count += 1
    if stub.compressed_claim_label is not None:
        field_count += 1

    field_count += len(stub.retrieval_hints)
    return field_count


# This helper builds a stub from a record using the StubMemoryPolicy's build_stub method.
# What this does:
# - Provides a module-level convenience for building stubs when needed outside a policy instance.
# Why this exists:
# - Some callers may need stub construction without instantiating the full policy.
# What assumption it is making:
# - The StubMemoryPolicy.build_stub logic is the canonical stub builder.
def build_stub(policy: StubMemoryPolicy, record: CivicRecord) -> RecordStub:
    return policy.build_stub(record)
