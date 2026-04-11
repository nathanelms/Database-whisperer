"""semantic_router.py

A standalone semantic routing engine that assigns hierarchical meaning-addresses
to structured records and answers queries by staged routing instead of flat scan.

This is the product piece of the memory_lab project.

Core idea:
    Instead of scanning all records for every query, infer a discriminator ladder
    from the data, assign each record a hierarchical semantic address, and route
    queries through that address space — narrowing the candidate set at each stage.

The goal is speed: how many records did you NOT have to examine?

Usage:
    router = SemanticRouter()
    router.ingest(records, identity_fields=["gene", "variant", "disease"])
    result = router.query({"gene": "BRAF", "variant": "V600E", "disease": "Melanoma"}, ask_field="drug")

    result.answer           # the retrieved value
    result.records_examined  # how many records were touched
    result.total_records     # how many exist in the index
    result.route_used        # which routing strategy was chosen
    result.candidates_at_each_stage  # narrowing trace
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict


# ---------------------------------------------------------------------------
# Result object returned by every query
# ---------------------------------------------------------------------------

@dataclass
class RouteResult:
    """
    The answer to one routed query.

    What this does:
    - Bundles the retrieved answer with routing metadata so callers can measure
      both correctness and efficiency.

    Why this exists:
    - The whole point of semantic routing is speed. If we only return the answer
      we cannot prove the router is faster than a flat scan.

    What assumption it is making:
    - records_examined and candidates_at_each_stage are enough to demonstrate
      routing value in v1.
    """
    answer: Optional[str]
    records_examined: int
    total_records: int
    route_used: str
    candidates_at_each_stage: List[int]
    matched_record: Optional[Dict[str, Any]] = None
    confusion_candidates: int = 0


# ---------------------------------------------------------------------------
# Internal index structures
# ---------------------------------------------------------------------------

@dataclass
class LadderRung:
    """
    One level of the discovered discriminator ladder.

    What this does:
    - Names the field used at this routing stage and records how much ambiguity
      it removed when the ladder was inferred.

    Why this exists:
    - The router needs to know which fields to use at each narrowing stage.

    What assumption it is making:
    - Each rung is one field. Composite rungs are future work.
    """
    field_name: str
    ambiguity_reduction_rate: float


# ---------------------------------------------------------------------------
# The router
# ---------------------------------------------------------------------------

class SemanticRouter:
    """
    A routing engine that infers hierarchical semantic addresses from data
    and answers queries by staged narrowing instead of flat scan.

    What this does:
    - On ingest: discovers identity neighborhoods, infers a discriminator ladder,
      builds a hierarchical index.
    - On query: routes through the index stage by stage, touching as few records
      as possible.

    Why this exists:
    - This is the bridge from the memory_lab benchmark to a usable routing layer.

    What assumption it is making:
    - Records are dicts (or dict-like) with string field values.
    - Identity fields are provided by the caller.
    - Candidate discriminator fields are either provided or auto-detected.
    - Provenance fields (record_id-like) should be excluded from semantic routing.
    """

    def __init__(self) -> None:
        # All ingested records, kept for flat-scan baseline comparison.
        self._records: List[Dict[str, Any]] = []

        # The identity fields that define a record's primary neighborhood.
        self._identity_fields: List[str] = []

        # Fields explicitly excluded from semantic routing (e.g. record IDs).
        self._provenance_fields: List[str] = []

        # The discovered discriminator ladder.
        self._ladder: List[LadderRung] = []

        # Hierarchical index: identity_key -> stage_1_key -> stage_2_key -> [records]
        # Built during ingest after ladder inference.
        self._index: Dict[str, Any] = {}

        # Whether the router has been built.
        self._ready: bool = False

    # -------------------------------------------------------------------
    # Ingest
    # -------------------------------------------------------------------

    def ingest(
        self,
        records: List[Dict[str, Any]],
        identity_fields: List[str],
        candidate_fields: Optional[List[str]] = None,
        provenance_fields: Optional[List[str]] = None,
        max_ladder_depth: int = 3,
    ) -> None:
        """
        Ingest records, infer a discriminator ladder, and build the routing index.

        What this does:
        - Stores records, discovers ambiguous neighborhoods, ranks candidate
          discriminator fields by ambiguity reduction, and builds a staged index.

        Why this exists:
        - The router must learn the data's semantic structure before it can route.

        What assumption it is making:
        - All records share the same schema (same field names).
        - Identity fields are always present in every record.
        """
        self._records = list(records)
        self._identity_fields = list(identity_fields)
        self._provenance_fields = list(provenance_fields or [])

        # Auto-detect candidate fields if not provided:
        # all fields minus identity and provenance.
        if candidate_fields is None:
            if records:
                all_fields = set(records[0].keys())
                excluded = set(identity_fields) | set(self._provenance_fields)
                candidate_fields = sorted(all_fields - excluded)
            else:
                candidate_fields = []

        # Step 1: find ambiguous identity neighborhoods.
        neighborhoods = self._find_ambiguous_neighborhoods()

        # Step 2: infer discriminator ladder from those neighborhoods.
        self._ladder = self._infer_ladder(
            neighborhoods=neighborhoods,
            candidate_fields=candidate_fields,
            max_depth=max_ladder_depth,
        )

        # Step 3: build the hierarchical index.
        self._build_index()

        self._ready = True

    def _identity_key(self, record: Dict[str, Any]) -> tuple:
        """
        Extract the identity key from a record.

        What this does:
        - Creates a hashable tuple from the identity field values.

        Why this exists:
        - Neighborhoods, indexing, and query routing all need the same identity key.
        """
        return tuple(record.get(f, "") for f in self._identity_fields)

    def _find_ambiguous_neighborhoods(self) -> List[List[Dict[str, Any]]]:
        """
        Group records by identity key and return only neighborhoods with >1 record.

        What this does:
        - Identifies where retrieval ambiguity actually exists in the data.

        Why this exists:
        - The discriminator ladder should be inferred only from neighborhoods
          where disambiguation is actually needed.
        """
        neighborhoods: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        for record in self._records:
            neighborhoods[self._identity_key(record)].append(record)
        return [n for n in neighborhoods.values() if len(n) > 1]

    def _remaining_ambiguity_pairs(
        self,
        neighborhoods: List[List[Dict[str, Any]]],
        selected_fields: List[str],
    ) -> int:
        """
        Count unresolved competing record pairs after splitting by selected fields.

        What this does:
        - Measures how much ambiguity remains if we partition each neighborhood
          by the given field set.

        Why this exists:
        - This is the core scoring function for the greedy ladder search.
        """
        total = 0
        for neighborhood in neighborhoods:
            buckets: Dict[tuple, int] = defaultdict(int)
            for record in neighborhood:
                if selected_fields:
                    key = tuple(record.get(f, "") for f in selected_fields)
                else:
                    key = ("_identity_only_",)
                buckets[key] += 1
            for count in buckets.values():
                if count > 1:
                    total += count * (count - 1) // 2
        return total

    def _infer_ladder(
        self,
        neighborhoods: List[List[Dict[str, Any]]],
        candidate_fields: List[str],
        max_depth: int,
    ) -> List[LadderRung]:
        """
        Greedily discover the discriminator ladder.

        What this does:
        - At each step, picks the candidate field that reduces residual ambiguity
          the most, and adds it as the next rung.

        Why this exists:
        - This is the Database Whisper core: infer the cheapest useful field order
          from the data itself.

        What assumption it is making:
        - Greedy one-field-at-a-time selection is good enough for v1.
        """
        if not neighborhoods:
            return []

        chosen: List[str] = []
        remaining = list(candidate_fields)
        ladder: List[LadderRung] = []

        for _ in range(max_depth):
            if not remaining:
                break

            pairs_before = self._remaining_ambiguity_pairs(neighborhoods, chosen)
            if pairs_before == 0:
                break

            best_field = None
            best_reduction_rate = 0.0
            best_pairs_after = pairs_before

            for field_name in remaining:
                test_fields = chosen + [field_name]
                pairs_after = self._remaining_ambiguity_pairs(neighborhoods, test_fields)
                reduction = pairs_before - pairs_after
                rate = reduction / pairs_before if pairs_before else 0.0

                if rate > best_reduction_rate:
                    best_field = field_name
                    best_reduction_rate = rate
                    best_pairs_after = pairs_after

            if best_field is None or best_reduction_rate <= 0:
                break

            ladder.append(LadderRung(
                field_name=best_field,
                ambiguity_reduction_rate=best_reduction_rate,
            ))
            chosen.append(best_field)
            remaining = [f for f in remaining if f != best_field]

            if best_pairs_after == 0:
                break

        return ladder

    def _build_index(self) -> None:
        """
        Build a hierarchical index from identity key through ladder rungs.

        What this does:
        - Creates a nested dict structure:
          identity_key -> rung_1_value -> rung_2_value -> ... -> [records]

        Why this exists:
        - Staged routing needs a pre-built index to avoid scanning at query time.

        What assumption it is making:
        - The ladder is short (2-3 rungs max in practice), so nested dicts are fine.
        """
        self._index = {}

        for record in self._records:
            id_key = self._identity_key(record)

            # Navigate/create the nested index path.
            node = self._index
            node_key = id_key
            if node_key not in node:
                node[node_key] = {}
            node = node[node_key]

            for rung in self._ladder:
                rung_value = record.get(rung.field_name, "")
                if rung_value not in node:
                    node[rung_value] = {}
                node = node[rung_value]

            # Leaf level: list of records.
            if "_records" not in node:
                node["_records"] = []
            node["_records"].append(record)

    # -------------------------------------------------------------------
    # Query
    # -------------------------------------------------------------------

    def query(
        self,
        query_fields: Dict[str, Any],
        ask_field: str,
    ) -> RouteResult:
        """
        Answer a query by routing through the semantic index.

        What this does:
        - Extracts identity key from query, then walks the ladder index stage
          by stage, narrowing candidates at each step.

        Why this exists:
        - This is the core product function. It should be faster than flat scan
          because it touches fewer records.

        What assumption it is making:
        - The query carries hints for every ladder field (same keys as records).
        - If a ladder field is missing from the query, that stage is skipped
          and all branches at that level are searched.
        """
        if not self._ready:
            raise RuntimeError("Router has not been built. Call ingest() first.")

        total = len(self._records)
        stages: List[int] = []

        # Stage 0: identity lookup.
        id_key = tuple(query_fields.get(f, "") for f in self._identity_fields)
        if id_key not in self._index:
            return RouteResult(
                answer=None,
                records_examined=0,
                total_records=total,
                route_used="identity_miss",
                candidates_at_each_stage=[0],
            )

        node = self._index[id_key]
        # Count identity-level candidates.
        identity_count = self._count_records_in_subtree(node)
        stages.append(identity_count)

        # Walk the ladder.
        records_examined = 0
        for rung in self._ladder:
            rung_value = query_fields.get(rung.field_name, "")
            if rung_value in node:
                node = node[rung_value]
                stage_count = self._count_records_in_subtree(node)
                stages.append(stage_count)
            else:
                # Query does not have this hint — stay at current node.
                stages.append(stages[-1] if stages else 0)

        # Collect leaf records.
        leaf_records = self._collect_records_from_subtree(node)
        records_examined = len(leaf_records)

        if not leaf_records:
            return RouteResult(
                answer=None,
                records_examined=records_examined,
                total_records=total,
                route_used=self._describe_route(len(stages)),
                candidates_at_each_stage=stages,
            )

        # Pick the best match (first one — could be improved later).
        best = leaf_records[0]
        answer = best.get(ask_field)

        return RouteResult(
            answer=answer,
            records_examined=records_examined,
            total_records=total,
            route_used=self._describe_route(len(stages)),
            candidates_at_each_stage=stages,
            matched_record=best,
            confusion_candidates=len(leaf_records) - 1,
        )

    def flat_scan(
        self,
        query_fields: Dict[str, Any],
        ask_field: str,
    ) -> RouteResult:
        """
        Answer a query by flat scanning all records (baseline for speed comparison).

        What this does:
        - Checks every record against the identity key. Returns the first match.

        Why this exists:
        - We need a direct comparison to show that routing is faster.
        """
        total = len(self._records)
        id_key = tuple(query_fields.get(f, "") for f in self._identity_fields)
        examined = 0
        candidates = []

        for record in self._records:
            examined += 1
            if self._identity_key(record) == id_key:
                candidates.append(record)

        if not candidates:
            return RouteResult(
                answer=None,
                records_examined=examined,
                total_records=total,
                route_used="flat_scan",
                candidates_at_each_stage=[examined],
            )

        return RouteResult(
            answer=candidates[0].get(ask_field),
            records_examined=examined,
            total_records=total,
            route_used="flat_scan",
            candidates_at_each_stage=[len(candidates)],
            matched_record=candidates[0],
            confusion_candidates=len(candidates) - 1,
        )

    # -------------------------------------------------------------------
    # Inspection / explanation
    # -------------------------------------------------------------------

    def explain(self) -> Dict[str, Any]:
        """
        Return a human-readable summary of the router's learned structure.

        What this does:
        - Reports the discovered ladder, index depth, and record count.

        Why this exists:
        - The router should be legible, not a black box.
        """
        return {
            "total_records": len(self._records),
            "identity_fields": self._identity_fields,
            "provenance_fields": self._provenance_fields,
            "ladder": [
                {
                    "rung": i + 1,
                    "field": rung.field_name,
                    "ambiguity_reduction_rate": f"{rung.ambiguity_reduction_rate:.2%}",
                }
                for i, rung in enumerate(self._ladder)
            ],
            "identity_neighborhoods": len(self._index),
            "ambiguous_neighborhoods": len(self._find_ambiguous_neighborhoods()),
        }

    @property
    def ladder_fields(self) -> List[str]:
        """Return the ordered list of discriminator field names."""
        return [rung.field_name for rung in self._ladder]

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _count_records_in_subtree(self, node: dict) -> int:
        """Recursively count all records under a node."""
        if "_records" in node:
            return len(node["_records"])
        total = 0
        for key, child in node.items():
            if key != "_records" and isinstance(child, dict):
                total += self._count_records_in_subtree(child)
        return total

    def _collect_records_from_subtree(self, node: dict) -> List[Dict[str, Any]]:
        """Recursively collect all records under a node."""
        if "_records" in node:
            return list(node["_records"])
        results = []
        for key, child in node.items():
            if key != "_records" and isinstance(child, dict):
                results.extend(self._collect_records_from_subtree(child))
        return results

    def _describe_route(self, stages_used: int) -> str:
        """Build a human-readable route description."""
        parts = ["identity"]
        for i, rung in enumerate(self._ladder):
            if i + 1 < stages_used:
                parts.append(rung.field_name)
        return " -> ".join(parts)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build a small test dataset with ambiguity.
    test_records = [
        {"gene": "BRAF", "variant": "V600E", "disease": "Melanoma", "evidence_type": "supports:A", "drug": "Vemurafenib", "record_id": "R1"},
        {"gene": "BRAF", "variant": "V600E", "disease": "Melanoma", "evidence_type": "supports:A", "drug": "Dabrafenib", "record_id": "R2"},
        {"gene": "BRAF", "variant": "V600E", "disease": "Melanoma", "evidence_type": "does_not_support:C", "drug": "Sorafenib", "record_id": "R3"},
        {"gene": "BRAF", "variant": "V600E", "disease": "Melanoma", "evidence_type": "does_not_support:D", "drug": "Trametinib", "record_id": "R4"},
        {"gene": "EGFR", "variant": "L858R", "disease": "NSCLC", "evidence_type": "supports:A", "drug": "Erlotinib", "record_id": "R5"},
        {"gene": "EGFR", "variant": "L858R", "disease": "NSCLC", "evidence_type": "supports:B", "drug": "Osimertinib", "record_id": "R6"},
        {"gene": "EGFR", "variant": "L858R", "disease": "NSCLC", "evidence_type": "does_not_support:C", "drug": "Gefitinib", "record_id": "R7"},
        {"gene": "ALK", "variant": "Fusion", "disease": "NSCLC", "evidence_type": "supports:A", "drug": "Crizotinib", "record_id": "R8"},
    ]

    router = SemanticRouter()
    router.ingest(
        records=test_records,
        identity_fields=["gene", "variant", "disease"],
        provenance_fields=["record_id"],
    )

    print("=== Router Structure ===")
    info = router.explain()
    print(f"  Records: {info['total_records']}")
    print(f"  Identity fields: {info['identity_fields']}")
    print(f"  Identity neighborhoods: {info['identity_neighborhoods']}")
    print(f"  Ambiguous neighborhoods: {info['ambiguous_neighborhoods']}")
    print(f"  Discovered ladder:")
    for rung in info["ladder"]:
        print(f"    Rung {rung['rung']}: {rung['field']} (reduction={rung['ambiguity_reduction_rate']})")

    print()

    # Test queries.
    queries = [
        {"gene": "BRAF", "variant": "V600E", "disease": "Melanoma", "evidence_type": "supports:A", "drug": "Vemurafenib"},
        {"gene": "BRAF", "variant": "V600E", "disease": "Melanoma", "evidence_type": "does_not_support:C", "drug": "Sorafenib"},
        {"gene": "EGFR", "variant": "L858R", "disease": "NSCLC", "evidence_type": "supports:A", "drug": "Erlotinib"},
        {"gene": "ALK", "variant": "Fusion", "disease": "NSCLC", "evidence_type": "supports:A", "drug": "Crizotinib"},
    ]

    print("=== Query Results ===")
    for q in queries:
        # Routed query.
        routed = router.query(q, ask_field="drug")
        # Flat scan baseline.
        flat = router.flat_scan(q, ask_field="drug")

        print(f"\n  Query: {q['gene']} {q['variant']} in {q['disease']}")
        print(f"    Routed:    answer={routed.answer}, examined={routed.records_examined}/{routed.total_records}, "
              f"route={routed.route_used}, stages={routed.candidates_at_each_stage}, confusion={routed.confusion_candidates}")
        print(f"    Flat scan: answer={flat.answer}, examined={flat.records_examined}/{flat.total_records}, "
              f"confusion={flat.confusion_candidates}")

        speedup = flat.records_examined / max(routed.records_examined, 1)
        print(f"    Speedup: {speedup:.1f}x fewer records examined")

    print()

    # Summary.
    print("=== Speed Summary ===")
    total_routed = 0
    total_flat = 0
    for q in queries:
        routed = router.query(q, ask_field="drug")
        flat = router.flat_scan(q, ask_field="drug")
        total_routed += routed.records_examined
        total_flat += flat.records_examined

    print(f"  Total records examined (routed): {total_routed}")
    print(f"  Total records examined (flat):   {total_flat}")
    print(f"  Overall speedup: {total_flat / max(total_routed, 1):.1f}x")
