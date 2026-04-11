"""data_types.py

All dataclasses used across the memory_lab baseline experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from stream_generator import CivicRecord


# This dataclass stores the result of one episode under one policy.
# What this does:
# - Keeps the episode-level evaluation transparent so printed summaries are easy to trust.
# Why this exists:
# - Research code becomes much easier to inspect when every episode result is explicit.
# What assumption it is making:
# - A small fixed set of fields is enough to understand baseline behavior.
@dataclass(frozen=True)
class EpisodeResult:
    policy_name: str
    task_type: str
    episode_id: str
    target_position: str
    expected_answer: str
    predicted_answer: Optional[str]
    was_correct: bool
    was_full_hit: bool
    was_stub_hit: bool
    was_miss: bool
    was_confusion: bool
    full_records_stored: int
    stubs_stored: int
    stub_fields_retained: int


# This dataclass stores one routing-layer comparison result.
# What this does:
# - Records how flat lookup and address-guided lookup behaved on the same episode.
# Why this exists:
# - We want to measure whether meaning addresses actually shrink the ambiguity set.
# What assumption it is making:
# - Candidate count and confusion are enough to show routing value in v0.
@dataclass(frozen=True)
class RoutingResult:
    policy_name: str
    task_type: str
    flat_candidate_count: int
    routed_candidate_count: int
    flat_confusion: bool
    routed_confusion: bool


# This dataclass stores the detailed route comparison for one episode.
# What this does:
# - Records candidate counts and confusion outcomes for several routing styles on the same episode.
# Why this exists:
# - We want to compare flat lookup, one-stage routing, and two-stage routing without changing storage policies.
# What assumption it is making:
# - Candidate counts and confusion rate are enough to judge routing quality in this sandbox.
@dataclass(frozen=True)
class RouteComparisonResult:
    policy_name: str
    task_type: str
    flat_candidate_count: int
    identity_candidate_count: int
    identity_therapy_candidate_count: int
    meaning_candidate_count: int
    two_stage_stage1_candidate_count: int
    two_stage_stage2_candidate_count: int
    flat_confusion: bool
    identity_confusion: bool
    identity_therapy_confusion: bool
    meaning_confusion: bool
    two_stage_confusion: bool


# This dataclass stores one discriminator-ladder comparison row.
# What this does:
# - Keeps the ranking math explicit for one stub schema.
# Why this exists:
# - We want a readable way to compare ambiguity reduction against added address cost.
# What assumption it is making:
# - Confusion reduction per added field is the right first efficiency metric here.
@dataclass(frozen=True)
class LadderRow:
    policy_name: str
    discriminator_fields: List[str]
    confusion_rate: float
    confusion_reduction: float
    added_field_cost: int
    confusion_reduction_per_cost: float


# This dataclass stores one candidate-field score for the data-driven ladder.
# What this does:
# - Keeps the ambiguity math for one possible discriminator readable and explicit.
# Why this exists:
# - Database Whisper should infer useful discriminator fields from the dataset itself.
# What assumption it is making:
# - Pair-count reduction is a reasonable first proxy for ambiguity reduction.
@dataclass(frozen=True)
class WhisperFieldScore:
    field_name: str
    ambiguity_pairs_before: int
    ambiguity_pairs_after: int
    ambiguity_reduction: int
    ambiguity_reduction_rate: float
    field_cost: int
    reduction_per_cost: float


# This dataclass stores one chosen rung in the inferred discriminator ladder.
# What this does:
# - Records the best field selected at one recursive step of the ladder build.
# Why this exists:
# - We want the final printed ladder to show what was chosen and how much ambiguity it removed.
# What assumption it is making:
# - A short step-by-step ladder is enough for the current sandbox.
@dataclass(frozen=True)
class WhisperStep:
    step_index: int
    chosen_field: str
    ambiguity_pairs_before: int
    ambiguity_pairs_after: int
    ambiguity_reduction: int
    ambiguity_reduction_rate: float


# This dataclass stores one field's score on both routing and retrieval axes.
# What this does:
# - Keeps the "coarse splitter" and "final tie-breaker" metrics side by side.
# Why this exists:
# - The current sandbox has revealed that dataset partition quality and query-time retrieval quality are different objectives.
# What assumption it is making:
# - Confusion reduction is the most important retrieval-side metric for this stage.
@dataclass(frozen=True)
class DualAxisFieldScore:
    field_name: str
    coarse_ambiguity_reduction_rate: float
    retrieval_confusion_rate: float
    retrieval_confusion_reduction: float
    retrieval_confusion_reduction_rate: float


# This dataclass stores one routing-strategy chooser summary row.
# What this does:
# - Keeps the per-task, per-difficulty route comparison compact and explicit.
# Why this exists:
# - Database Whisper now needs to choose a route shape, not just rank fields.
# What assumption it is making:
# - Candidate counts and confusion rates are enough to pick a simple route policy.
@dataclass(frozen=True)
class RouteChooserRow:
    task_type: str
    distractor_level: str
    identity_candidates: float
    identity_confusion_rate: float
    identity_therapy_candidates: float
    identity_therapy_confusion_rate: float
    two_stage_stage_1_candidates: float
    two_stage_stage_2_candidates: float
    two_stage_confusion_rate: float
    coarse_routing_paid_for_itself: bool
    chosen_route: str


# This dataclass stores one routing-policy stress-test condition.
# What this does:
# - Keeps one condition's route metrics together so the chooser can be validated across settings.
# Why this exists:
# - We want to know where the current routing policy holds and where it breaks.
# What assumption it is making:
# - A compact condition summary is enough for the first chooser stress test.
@dataclass(frozen=True)
class RouteStressRow:
    task_type: str
    distractor_level: str
    distractor_count: int
    record_count: int
    identity_candidates: float
    identity_confusion_rate: float
    identity_cost: float
    identity_therapy_candidates: float
    identity_therapy_confusion_rate: float
    identity_therapy_cost: float
    two_stage_stage_1_candidates: float
    two_stage_stage_2_candidates: float
    two_stage_confusion_rate: float
    two_stage_cost: float
    stop_at_identity_therapy: bool
    pay_for_two_stage: bool


# This dataclass stores one per-retrieval route explanation row.
# What this does:
# - Keeps the route chooser's intermediate candidate counts and final decision for one
#   retrieval episode.
# Why this exists:
# - The user asked us to make the current semantic routing method legible without changing it.
# What assumption it is making:
# - Identity size, stage candidate counts, chosen route, and a short textual reason are
#   enough to explain one route decision.
@dataclass(frozen=True)
class RouteExplanationRow:
    task_type: str
    distractor_level: str
    episode_id: str
    identity_neighborhood_size: int
    evidence_type_candidate_count: int
    therapy_candidate_count: int
    chosen_route: str
    reason: str


# This dataclass stores the smallest cross-source comparison snapshot we care about.
# What this does:
# - Captures the coarse splitter, final tie-breaker, semantic ladder, and chooser behavior
#   for one task type under one data source.
# Why this exists:
# - The new real-data validation step is about whether the discovered method transfers,
#   not about printing every metric twice by hand.
# What assumption it is making:
# - These few summary fields are enough to compare the synthetic and real runs at a
#   method-validation level.
@dataclass(frozen=True)
class SourceComparisonRow:
    data_source: str
    task_type: str
    coarse_splitter: str
    final_tie_breaker: str
    semantic_ladder: List[str]
    chooser_route_for_ambiguity: str


# This dataclass represents the two storage tiers used by the tiered policy.
# What this does:
# - Keeps short-term and durable memory separate but easy to inspect.
# Why this exists:
# - The new policy should preserve all records briefly while still applying a readable
#   promotion rule for durable storage.
# What assumption it is making:
# - In v1, two plain Python lists are enough to model a useful memory-tier idea.
@dataclass(frozen=True)
class MemoryStore:
    short_term_records: List[CivicRecord]
    durable_records: List[CivicRecord]


# This dataclass represents a lightweight indexed reference stub.
# What this does:
# - Stores just enough metadata to tell us that a relevant fact existed without keeping
#   the full evidence record around.
# Why this exists:
# - The new policy is testing the idea "forget the detail, keep the address."
# What assumption it is making:
# - A tiny identity-focused handle is useful even when the original record payload is gone.
@dataclass(frozen=True)
class RecordStub:
    record_id: str
    gene: str
    variant: str
    disease: str
    evidence_type: Optional[str]
    evidence_level: Optional[str]
    direction: Optional[str]
    therapy: Optional[str]
    compressed_claim_label: Optional[str]
    retrieval_hints: List[str]
    source_pointer: str
    status: str


# This dataclass describes one stub schema variant.
# What this does:
# - Names a stub design and says which optional fields it should keep.
# Why this exists:
# - We want to compare several compact stub designs without adding framework-like machinery.
# What assumption it is making:
# - A few booleans are enough to express the early stub-design search space.
@dataclass(frozen=True)
class StubSchema:
    name: str
    include_evidence_type: bool
    include_evidence_level: bool
    include_direction: bool
    include_therapy: bool
    include_source_id_hint: bool
    discriminator_fields: List[str]


# This dataclass stores the two outputs of the stub policy.
# What this does:
# - Separates retained full records from compressed stubs.
# Why this exists:
# - We want the storage design to stay explicit and easy to inspect.
# What assumption it is making:
# - In v1, one list for full records and one list for stubs is enough structure.
@dataclass(frozen=True)
class StubMemoryStore:
    durable_records: List[CivicRecord]
    stubs: List[RecordStub]
