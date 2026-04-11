"""stream_generator.py

This module creates a very small synthetic research environment inspired by CIViC-like
oncology evidence records.

The goal is not to simulate the full CIViC data model. The goal is to create a clean,
readable source of structured facts that we can use to test memory-selection policies.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

from real_civic_sample import load_real_civic_sample
from real_clinvar_sample import load_real_clinvar_sample


# This dataclass represents one synthetic oncology evidence record.
# What this does:
# - Stores the structured fact fields that later retrieval will use.
# Why this exists:
# - We want a simple stand-in for a CIViC-like evidence item without bringing in
#   the full complexity of the real CIViC schema yet.
# What assumption it is making:
# - A small set of fields such as gene, variant, disease, drug, and evidence level
#   is enough for the first memory-selection experiments.
@dataclass(frozen=True)
class CivicRecord:
    record_id: str
    gene: str
    variant: str
    disease: str
    drug: str
    evidence_direction: str
    evidence_level: str
    statement: str


# This dataclass represents the later question we ask about a stored fact.
# What this does:
# - Defines the structured lookup target and the answer field we want to recover.
# Why this exists:
# - The lab is about whether memory policies preserve later-useful facts, so we need
#   a clean query object that states exactly what later use looks like.
# What assumption it is making:
# - Direct fact recall can be represented as an exact request for one field from one
#   target record identity.
@dataclass(frozen=True)
class Query:
    query_id: str
    gene: str
    variant: str
    disease: str
    ask_field: str
    prompt: str
    answer: str
    evidence_direction_hint: str
    evidence_level_hint: str
    evidence_type_hint: str
    therapy_hint: str
    claim_label_hint: str


# This dataclass bundles the memory stream and the later query into one episode.
# What this does:
# - Captures one small experiment unit: a stream of records followed by one recall query.
# Why this exists:
# - Baseline runners become much easier to read when one object holds the stream,
#   the target fact, the question, and metadata about the task.
# What assumption it is making:
# - Both direct recall and late-relevance recall can share the same basic episode shape.
@dataclass(frozen=True)
class RecallEpisode:
    episode_id: str
    task_type: str
    distractor_level: str
    target_position: str
    stream_records: List[CivicRecord]
    target_record: CivicRecord
    query: Query


# Small synthetic ontology tables keep generation deterministic and readable.
GENES = ["BRAF", "EGFR", "ALK", "KRAS", "PIK3CA", "BRCA1", "BRCA2", "ERBB2", "MET", "RET", "ROS1", "KIT"]
VARIANTS = [
    "V600E",
    "L858R",
    "Exon 19 deletion",
    "G12C",
    "E545K",
    "Fusion",
    "Amplification",
    "T790M",
    "Exon 20 insertion",
    "NTRK-like fusion",
]
DISEASES = [
    "Melanoma",
    "Non-Small Cell Lung Cancer",
    "Colorectal Cancer",
    "Breast Cancer",
    "Ovarian Cancer",
    "Glioma",
    "Pancreatic Cancer",
    "Gastric Cancer",
]
DRUGS = [
    "Vemurafenib",
    "Osimertinib",
    "Alectinib",
    "Sotorasib",
    "Alpelisib",
    "Olaparib",
    "Trastuzumab",
    "Capmatinib",
    "Selpercatinib",
    "Crizotinib",
]
EVIDENCE_DIRECTIONS = ["supports", "does_not_support"]
EVIDENCE_LEVELS = ["A", "B", "C", "D"]
ASK_FIELDS = ["drug", "evidence_level"]
TARGET_POSITIONS = ["early", "middle", "late"]


# This helper makes the record statement human-readable.
# What this does:
# - Turns the structured fields into one short natural-language sentence.
# Why this exists:
# - Even though retrieval is structured, having a text statement makes the record feel
#   closer to a real evidence item and gives us room for future salience heuristics.
# What assumption it is making:
# - A single summary sentence is enough textual context for the first experiments.
def build_statement(record: CivicRecord) -> str:
    return (
        f"In {record.disease}, {record.gene} {record.variant} {record.evidence_direction} "
        f"response to {record.drug} with evidence level {record.evidence_level}."
    )


# This helper builds a compressed claim-style label from a record.
# What this does:
# - Turns a record into one short symbolic label that captures its directional claim pattern.
# Why this exists:
# - We want a tiny field that richer stubs can use for disambiguation later.
# What assumption it is making:
# - A compact claim label is enough to help separate nearby records in v1.
def build_claim_label(record: CivicRecord) -> str:
    return f"{record.gene}_{record.variant}_{record.evidence_direction}"


# This helper builds a short evidence-type label from a record.
# What this does:
# - Combines evidence direction and level into one retrieval hint.
# Why this exists:
# - Some stub schemas should be able to keep a small evidence-type handle for later disambiguation.
# What assumption it is making:
# - Direction plus level is a useful compact proxy for evidence type.
def build_evidence_type(record: CivicRecord) -> str:
    return f"{record.evidence_direction}:{record.evidence_level}"


# This function builds a pool of unique synthetic CIViC-like records.
# What this does:
# - Creates structured oncology evidence records with enough variety to support recall
#   and distractor experiments.
# Why this exists:
# - We need a repeatable synthetic dataset before we bring in real CIViC data later.
# What assumption it is making:
# - Randomly sampling from a small ontology gives us enough diversity for baseline tests
#   as long as we keep record identities unique.
def generate_synthetic_records(count: int = 80, seed: int = 7) -> List[CivicRecord]:
    rng = random.Random(seed)
    records: List[CivicRecord] = []
    seen_signatures = set()

    while len(records) < count:
        gene = rng.choice(GENES)
        variant = rng.choice(VARIANTS)
        disease = rng.choice(DISEASES)
        drug = rng.choice(DRUGS)
        evidence_direction = rng.choice(EVIDENCE_DIRECTIONS)
        evidence_level = rng.choice(EVIDENCE_LEVELS)

        # We allow multiple records to share the same identity neighborhood so the arena
        # can create adversarial near-collisions. We only block exact duplicate records.
        signature = (gene, variant, disease, drug, evidence_direction, evidence_level)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        record = CivicRecord(
            record_id=f"REC-{len(records) + 1:03d}",
            gene=gene,
            variant=variant,
            disease=disease,
            drug=drug,
            evidence_direction=evidence_direction,
            evidence_level=evidence_level,
            statement="",
        )
        record = CivicRecord(
            record_id=record.record_id,
            gene=record.gene,
            variant=record.variant,
            disease=record.disease,
            drug=record.drug,
            evidence_direction=record.evidence_direction,
            evidence_level=record.evidence_level,
            statement=build_statement(record),
        )
        records.append(record)

    return records


# This helper converts one real CIViC-like sample row into the current sandbox schema.
# What this does:
# - Maps the tiny real-data slice into the same `CivicRecord` structure used by the
#   synthetic experiments.
# Why this exists:
# - The user asked us to validate the current method on a small real CIViC slice without
#   redesigning the sandbox.
# What assumption it is making:
# - The current internal schema is still expressive enough for a small real-data check
#   even though it is simpler than the full CIViC data model.
def normalize_real_record_to_civic_record(real_record) -> CivicRecord:
    return CivicRecord(
        record_id=real_record.record_id,
        gene=real_record.gene,
        variant=real_record.variant,
        disease=real_record.disease,
        drug=real_record.therapy,
        evidence_direction=real_record.direction,
        evidence_level=real_record.evidence_level,
        statement=real_record.statement,
    )


# This helper converts one ClinVar-like sample row into the current sandbox schema.
# What this does:
# - Maps the second structured domain into the same `CivicRecord` structure used everywhere else.
# Why this exists:
# - We want to test whether the procedure transfers across domains without changing the core method.
# What assumption it is making:
# - ClinVar-like interpretation records can be represented with the current internal slots
#   if we treat clinical significance as the answer-bearing field and review tier as supporting context.
def normalize_real_clinvar_record_to_civic_record(real_record) -> CivicRecord:
    return CivicRecord(
        record_id=real_record.record_id,
        gene=real_record.gene,
        variant=real_record.variant,
        disease=real_record.condition,
        drug=real_record.clinical_significance,
        evidence_direction=real_record.assertion_direction,
        evidence_level=real_record.review_status_tier,
        statement=real_record.statement,
    )


# This helper loads the base record pool for one experiment source.
# What this does:
# - Returns either synthetic records or the tiny built-in real CIViC sample normalized
#   into the current internal schema.
# Why this exists:
# - We want to reuse the same episode generator and routing experiments across both data
#   sources with as little branching as possible.
# What assumption it is making:
# - The first real-data validation pass can use a fixed local sample rather than a full
#   external CIViC download.
def load_record_pool(data_source: str, record_count: int, seed: int) -> List[CivicRecord]:
    if data_source == "real":
        real_records = load_real_civic_sample()
        return [normalize_real_record_to_civic_record(real_record) for real_record in real_records]

    if data_source == "real_clinvar":
        real_records = load_real_clinvar_sample()
        return [
            normalize_real_clinvar_record_to_civic_record(real_record)
            for real_record in real_records
        ]

    return generate_synthetic_records(count=record_count, seed=seed)


# This helper ranks candidate distractors by how confusable they are with the target.
# What this does:
# - Scores records so the caller can pick easy, medium, or hard distractors.
# Why this exists:
# - The lab question depends on confusion pressure, so we need controllable distractor
#   difficulty rather than purely random background noise.
# What assumption it is making:
# - Shared structured fields like gene and disease are the main source of confusion in
#   early exact-retrieval experiments.
def distractor_match_score(target: CivicRecord, candidate: CivicRecord) -> int:
    score = 0
    if candidate.gene == target.gene:
        score += 2
    if candidate.disease == target.disease:
        score += 2
    if candidate.variant == target.variant:
        score += 1
    if candidate.drug == target.drug:
        score += 1
    return score


# This helper checks whether a candidate record forms a near-collision with the target.
# What this does:
# - Identifies records that are almost the same fact but differ on one key field.
# Why this exists:
# - The current arena mostly produces omission errors, so we need distractors that are
#   more likely to trigger wrong retrieval rather than simple misses.
# What assumption it is making:
# - Near-collisions built from shared gene/disease/variant context are a useful first
#   way to stress exact retrieval confusion.
def is_near_collision(target: CivicRecord, candidate: CivicRecord) -> bool:
    same_gene = candidate.gene == target.gene
    same_variant = candidate.variant == target.variant
    same_disease = candidate.disease == target.disease
    same_drug = candidate.drug == target.drug

    same_gene_same_disease_diff_variant = same_gene and same_disease and not same_variant
    same_gene_same_variant_diff_disease = same_gene and same_variant and not same_disease
    same_gene_same_disease_diff_drug = same_gene and same_disease and not same_drug

    return (
        same_gene_same_disease_diff_variant
        or same_gene_same_variant_diff_disease
        or same_gene_same_disease_diff_drug
    )


# This helper checks for a stronger collision that keeps the same lookup identity.
# What this does:
# - Identifies records that share gene, variant, and disease with the target but differ
#   on a field the query might later ask about.
# Why this exists:
# - Exact retrieval only becomes confused when multiple stored records fit the same lookup key.
# What assumption it is making:
# - Same-identity disagreements on drug or evidence level are the most direct way to trigger
#   retrieval confusion in the current sandbox.
def is_same_identity_different_answer(target: CivicRecord, candidate: CivicRecord) -> bool:
    same_identity = (
        candidate.gene == target.gene
        and candidate.variant == target.variant
        and candidate.disease == target.disease
    )
    different_drug = candidate.drug != target.drug
    different_evidence_level = candidate.evidence_level != target.evidence_level
    return same_identity and (different_drug or different_evidence_level)


# This helper checks whether a candidate threatens the specific asked field.
# What this does:
# - Prioritizes collisions that differ on the field the later query cares about.
# Why this exists:
# - We want adversarial distractors to attack the answer-bearing field, not just the neighborhood.
# What assumption it is making:
# - A collision is most dangerous when it preserves the lookup identity but changes the requested answer.
def is_answer_field_collision(target: CivicRecord, candidate: CivicRecord, ask_field: str) -> bool:
    same_identity = (
        candidate.gene == target.gene
        and candidate.variant == target.variant
        and candidate.disease == target.disease
    )
    if not same_identity:
        return False
    return getattr(candidate, ask_field) != getattr(target, ask_field)


# This helper creates one adversarial record that keeps the same lookup identity.
# What this does:
# - Copies gene, variant, and disease from the target while changing other fields.
# Why this exists:
# - We need multiple records that all look identical to the current exact retriever.
# What assumption it is making:
# - Same-identity disagreements are the cleanest way to expose mistaken identity.
def make_identity_adversarial_record(
    target: CivicRecord,
    clone_index: int,
    *,
    drug: str | None = None,
    evidence_direction: str | None = None,
    evidence_level: str | None = None,
) -> CivicRecord:
    cloned_record = CivicRecord(
        record_id=f"{target.record_id}-AMB-{clone_index}",
        gene=target.gene,
        variant=target.variant,
        disease=target.disease,
        drug=drug if drug is not None else target.drug,
        evidence_direction=evidence_direction if evidence_direction is not None else target.evidence_direction,
        evidence_level=evidence_level if evidence_level is not None else target.evidence_level,
        statement="",
    )
    return CivicRecord(
        record_id=cloned_record.record_id,
        gene=cloned_record.gene,
        variant=cloned_record.variant,
        disease=cloned_record.disease,
        drug=cloned_record.drug,
        evidence_direction=cloned_record.evidence_direction,
        evidence_level=cloned_record.evidence_level,
        statement=build_statement(cloned_record),
    )


# This helper creates a small set of true ambiguity distractors.
# What this does:
# - Builds several same-identity records that disagree with the target on answer-bearing
#   or claim-bearing fields.
# Why this exists:
# - The new research goal is to make minimal identity insufficient.
# What assumption it is making:
# - These controlled disagreements are enough to create real retrieval ambiguity.
def build_ambiguity_distractors(target: CivicRecord) -> List[CivicRecord]:
    alternative_drug = next(drug for drug in DRUGS if drug != target.drug)
    alternative_evidence_level = next(level for level in EVIDENCE_LEVELS if level != target.evidence_level)
    alternative_direction = next(direction for direction in EVIDENCE_DIRECTIONS if direction != target.evidence_direction)
    alternative_evidence_type_level = next(
        level for level in EVIDENCE_LEVELS if level != target.evidence_level and level != alternative_evidence_level
    )

    return [
        make_identity_adversarial_record(target, 1, drug=alternative_drug),
        make_identity_adversarial_record(target, 2, evidence_level=alternative_evidence_level),
        make_identity_adversarial_record(
            target,
            3,
            evidence_direction=alternative_direction,
            evidence_level=alternative_evidence_type_level,
        ),
        make_identity_adversarial_record(target, 4, evidence_direction=alternative_direction),
    ]


# This helper builds deliberately dangerous distractors when possible.
# What this does:
# - Prefers identity-level and neighborhood-level collisions, then fills remaining slots with
#   highly similar records.
# Why this exists:
# - We want a specific distractor setting that pushes the benchmark toward retrieval confusion.
# What assumption it is making:
# - Exact near-collision templates are more dangerous than generic \"hard\" distractors.
def select_collision_distractors(
    records: List[CivicRecord],
    target: CivicRecord,
    distractor_count: int,
    rng: random.Random,
    ask_field: str,
) -> List[CivicRecord]:
    candidates = [record for record in records if record.record_id != target.record_id]
    answer_field_collisions = [
        record for record in candidates if is_answer_field_collision(target, record, ask_field=ask_field)
    ]
    same_identity_other_answer = [record for record in candidates if is_same_identity_different_answer(target, record)]
    near_collision_candidates = [record for record in candidates if is_near_collision(target, record)]

    # We build the collision pool in priority order and deduplicate by record id.
    ordered_candidates: List[CivicRecord] = []
    seen_ids = set()
    for candidate_group in [
        answer_field_collisions,
        same_identity_other_answer,
        near_collision_candidates,
    ]:
        rng.shuffle(candidate_group)
        for candidate in candidate_group:
            if candidate.record_id not in seen_ids:
                ordered_candidates.append(candidate)
                seen_ids.add(candidate.record_id)

    other_candidates = [record for record in candidates if record.record_id not in seen_ids]

    chosen = ordered_candidates[:distractor_count]

    if len(chosen) < distractor_count:
        remaining_needed = distractor_count - len(chosen)
        ranked_others = sorted(
            other_candidates,
            key=lambda record: distractor_match_score(target, record),
            reverse=True,
        )
        chosen.extend(ranked_others[:remaining_needed])

    rng.shuffle(chosen)
    return chosen


# This helper creates a stronger ambiguity-focused distractor set.
# What this does:
# - Returns several same-identity adversarial clones and then fills any remaining slots
#   with collision-style distractors.
# Why this exists:
# - We want a mode where core identity is no longer enough to guarantee the right match.
# What assumption it is making:
# - Same-identity clones are the strongest stress test for mistaken identity in v1.
def select_ambiguity_distractors(
    records: List[CivicRecord],
    target: CivicRecord,
    distractor_count: int,
    rng: random.Random,
    ask_field: str,
) -> List[CivicRecord]:
    ambiguity_distractors = build_ambiguity_distractors(target)

    if len(ambiguity_distractors) >= distractor_count:
        chosen = ambiguity_distractors[:distractor_count]
        rng.shuffle(chosen)
        return chosen

    remaining_needed = distractor_count - len(ambiguity_distractors)
    additional_distractors = select_collision_distractors(
        records=records,
        target=target,
        distractor_count=remaining_needed,
        rng=rng,
        ask_field=ask_field,
    )
    chosen = [*ambiguity_distractors, *additional_distractors]
    rng.shuffle(chosen)
    return chosen[:distractor_count]


# This helper chooses distractors according to a requested difficulty level.
# What this does:
# - Returns records that are intentionally easy, medium, hard, or near-collision hard.
# Why this exists:
# - We want a direct way to stress test whether memory and retrieval preserve the right fact.
# What assumption it is making:
# - Higher structured overlap corresponds to higher memory and retrieval confusion risk.
def select_distractors(
    records: List[CivicRecord],
    target: CivicRecord,
    distractor_level: str,
    distractor_count: int,
    rng: random.Random,
    ask_field: str,
) -> List[CivicRecord]:
    if distractor_level == "ambiguity":
        return select_ambiguity_distractors(
            records=records,
            target=target,
            distractor_count=distractor_count,
            rng=rng,
            ask_field=ask_field,
        )

    if distractor_level == "collision":
        return select_collision_distractors(
            records=records,
            target=target,
            distractor_count=distractor_count,
            rng=rng,
            ask_field=ask_field,
        )

    candidates = [record for record in records if record.record_id != target.record_id]

    if distractor_level == "easy":
        ranked = sorted(candidates, key=lambda record: distractor_match_score(target, record))
    elif distractor_level == "hard":
        ranked = sorted(candidates, key=lambda record: distractor_match_score(target, record), reverse=True)
    else:
        ranked = sorted(candidates, key=lambda record: abs(distractor_match_score(target, record) - 2))

    chosen = ranked[: max(distractor_count * 2, distractor_count)]
    rng.shuffle(chosen)
    return chosen[:distractor_count]


# This helper creates the later recall query for one target record.
# What this does:
# - Builds a structured question that asks for one exact field from the target fact.
# Why this exists:
# - The runner needs a stable and inspectable way to test whether a memory policy kept
#   the useful fact available later.
# What assumption it is making:
# - Asking for either the drug or evidence level is enough to expose early retention and
#   confusion behavior.
def build_query(target: CivicRecord, episode_index: int, task_type: str, rng: random.Random) -> Query:
    ask_field = rng.choice(ASK_FIELDS)
    answer = getattr(target, ask_field)

    if task_type == "late_relevance_recall":
        prompt = (
            f"Later we realize {target.gene} {target.variant} in {target.disease} matters. "
            f"What is the recorded {ask_field}?"
        )
    else:
        prompt = f"For {target.gene} {target.variant} in {target.disease}, what is the recorded {ask_field}?"

    return Query(
        query_id=f"Q-{task_type[:2].upper()}-{episode_index:03d}",
        gene=target.gene,
        variant=target.variant,
        disease=target.disease,
        ask_field=ask_field,
        prompt=prompt,
        answer=answer,
        evidence_direction_hint=target.evidence_direction,
        evidence_level_hint=target.evidence_level,
        evidence_type_hint=build_evidence_type(target),
        therapy_hint=target.drug,
        claim_label_hint=build_claim_label(target),
    )


# This helper says whether a record would look salient to the current rule-based policy.
# What this does:
# - Mirrors the simple salience heuristic so the generator can build stress cases for it.
# Why this exists:
# - The new late-relevance task should sometimes target facts that do not look important
#   when they first appear in the stream.
# What assumption it is making:
# - In v1, salience is determined only by evidence level and evidence direction.
def looks_salient_under_rules(record: CivicRecord) -> bool:
    is_high_evidence = record.evidence_level in {"A", "B"}
    is_supportive = record.evidence_direction == "supports"
    return is_high_evidence or is_supportive


# This helper chooses a target record for the requested task type.
# What this does:
# - Picks a target fact, with special handling for late-relevance episodes.
# Why this exists:
# - We want the new task to expose the weakness of salience-only storage rules.
# What assumption it is making:
# - A late-relevance target is most informative when it often looks non-salient at intake time.
def choose_target_record(records: List[CivicRecord], task_type: str, rng: random.Random) -> CivicRecord:
    if task_type == "late_relevance_recall":
        non_salient_records = [record for record in records if not looks_salient_under_rules(record)]
        if non_salient_records:
            return rng.choice(non_salient_records)

    return rng.choice(records)


# This helper inserts the target into the stream at an approximate position.
# What this does:
# - Places the target early, middle, or late among the distractors.
# Why this exists:
# - Order matters in memory experiments, and the new task specifically asks us to vary it.
# What assumption it is making:
# - Coarse placement buckets are enough for the first pass; we do not need fine-grained timing yet.
def place_target_in_stream(
    target: CivicRecord,
    distractors: List[CivicRecord],
    target_position: str,
    rng: random.Random,
) -> List[CivicRecord]:
    stream_records = list(distractors)

    if target_position == "early":
        insert_index = 0
    elif target_position == "late":
        insert_index = len(stream_records)
    else:
        insert_index = len(stream_records) // 2

    # A tiny local shuffle keeps each bucket from looking perfectly rigid.
    # Early stays near the front, middle stays near the center, and late stays near the end.
    if target_position == "early" and len(stream_records) > 1:
        insert_index = rng.choice([0, 1])
    elif target_position == "middle" and len(stream_records) > 2:
        center = len(stream_records) // 2
        insert_index = max(0, min(len(stream_records), center + rng.choice([-1, 0, 1])))
    elif target_position == "late" and len(stream_records) > 1:
        insert_index = rng.choice([len(stream_records) - 1, len(stream_records)])

    stream_records.insert(insert_index, target)
    return stream_records


# This is the main episode generator used by the baseline runner.
# What this does:
# - Creates recall episodes for either direct recall or late-relevance recall.
# Why this exists:
# - The memory lab needs a tiny benchmark surface that can expose both omission and
#   delayed-importance failure modes.
# What assumption it is making:
# - These two task types are enough to deepen the v1 arena without adding complexity.
def generate_recall_episodes(
    task_type: str,
    record_count: int = 80,
    episode_count: int = 12,
    distractor_level: str = "medium",
    distractor_count: int = 5,
    data_source: str = "synthetic",
    seed: int = 7,
) -> List[RecallEpisode]:
    rng = random.Random(seed)
    records = load_record_pool(data_source=data_source, record_count=record_count, seed=seed)
    episodes: List[RecallEpisode] = []

    for episode_index in range(1, episode_count + 1):
        target = choose_target_record(records=records, task_type=task_type, rng=rng)
        query = build_query(target=target, episode_index=episode_index, task_type=task_type, rng=rng)
        distractors = select_distractors(
            records=records,
            target=target,
            distractor_level=distractor_level,
            distractor_count=distractor_count,
            rng=rng,
            ask_field=query.ask_field,
        )

        if task_type == "late_relevance_recall":
            target_position = rng.choice(TARGET_POSITIONS)
            stream_records = place_target_in_stream(
                target=target,
                distractors=distractors,
                target_position=target_position,
                rng=rng,
            )
        else:
            target_position = "mixed"
            stream_records = [target, *distractors]
            rng.shuffle(stream_records)

        episode = RecallEpisode(
            episode_id=f"EP-{task_type[:2].upper()}-{episode_index:03d}",
            task_type=task_type,
            distractor_level=distractor_level,
            target_position=target_position,
            stream_records=stream_records,
            target_record=target,
            query=query,
        )
        episodes.append(episode)

    return episodes
