"""Core semantic routing engine.

Discovers discriminator ladders from structured data and routes queries
through hierarchical indexes for fast, exact retrieval.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from collections import defaultdict

from ._types import RouteResult, LadderRung


class Router:
    """
    Discovers a discriminator ladder from structured data and routes
    queries through it for fast retrieval.

    Usage:
        router = Router()
        router.ingest(records, identity_fields=["gene", "disease"])
        result = router.query({"gene": "BRAF", "disease": "Melanoma"}, ask_field="therapy")
    """

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []
        self._identity_fields: List[str] = []
        self._provenance_fields: List[str] = []
        self._ladder: List[LadderRung] = []
        self._index: Dict[str, Any] = {}
        self._ready: bool = False

    @property
    def ladder_fields(self) -> List[str]:
        return [rung.field_name for rung in self._ladder]

    @property
    def ladder(self) -> List[LadderRung]:
        return list(self._ladder)

    def ingest(
        self,
        records: List[Dict[str, Any]],
        identity_fields: List[str],
        candidate_fields: Optional[List[str]] = None,
        provenance_fields: Optional[List[str]] = None,
        max_ladder_depth: int = 4,
    ) -> None:
        """Ingest records, infer the discriminator ladder, build the routing index."""
        self._records = list(records)
        self._identity_fields = list(identity_fields)
        self._provenance_fields = list(provenance_fields or [])

        if candidate_fields is None:
            if records:
                all_fields = set(records[0].keys())
                excluded = set(identity_fields) | set(self._provenance_fields)
                candidate_fields = sorted(all_fields - excluded)
            else:
                candidate_fields = []

        neighborhoods = self._find_ambiguous_neighborhoods()
        self._ladder = self._infer_ladder(neighborhoods, candidate_fields, max_ladder_depth)
        self._build_index()
        self._ready = True

    def query(self, query_fields: Dict[str, Any], ask_field: str) -> RouteResult:
        """Route a query through the ladder index."""
        if not self._ready:
            raise RuntimeError("Router not built. Call ingest() first.")

        total = len(self._records)
        id_key = tuple(query_fields.get(f, "") for f in self._identity_fields)

        if id_key not in self._index:
            return RouteResult(answer=None, records_examined=0, total_records=total,
                               route_used="identity_miss", candidates_at_each_stage=[0])

        node = self._index[id_key]
        stages = [self._count_in_subtree(node)]

        for rung in self._ladder:
            rung_value = query_fields.get(rung.field_name, "")
            if rung_value in node:
                node = node[rung_value]
                stages.append(self._count_in_subtree(node))
            else:
                stages.append(stages[-1] if stages else 0)

        leaves = self._collect_from_subtree(node)
        if not leaves:
            return RouteResult(answer=None, records_examined=0, total_records=total,
                               route_used=self._describe_route(len(stages)),
                               candidates_at_each_stage=stages)

        best = leaves[0]
        return RouteResult(
            answer=best.get(ask_field),
            records_examined=len(leaves),
            total_records=total,
            route_used=self._describe_route(len(stages)),
            candidates_at_each_stage=stages,
            matched_record=best,
            confusion_candidates=len(leaves) - 1,
        )

    def flat_scan(self, query_fields: Dict[str, Any], ask_field: str) -> RouteResult:
        """Baseline flat scan for speed comparison."""
        total = len(self._records)
        id_key = tuple(query_fields.get(f, "") for f in self._identity_fields)
        examined = 0
        candidates = []
        for record in self._records:
            examined += 1
            if self._identity_key(record) == id_key:
                candidates.append(record)
        if not candidates:
            return RouteResult(answer=None, records_examined=examined, total_records=total,
                               route_used="flat_scan", candidates_at_each_stage=[examined])
        return RouteResult(
            answer=candidates[0].get(ask_field), records_examined=examined,
            total_records=total, route_used="flat_scan",
            candidates_at_each_stage=[len(candidates)],
            matched_record=candidates[0], confusion_candidates=len(candidates) - 1,
        )

    def explain(self) -> Dict[str, Any]:
        """Human-readable summary of the router's structure."""
        return {
            "total_records": len(self._records),
            "identity_fields": self._identity_fields,
            "provenance_fields": self._provenance_fields,
            "ladder": [
                {"rung": i + 1, "field": r.field_name,
                 "ambiguity_reduction_rate": f"{r.ambiguity_reduction_rate:.2%}"}
                for i, r in enumerate(self._ladder)
            ],
            "identity_neighborhoods": len(self._index),
            "ambiguous_neighborhoods": len(self._find_ambiguous_neighborhoods()),
        }

    # --- Internal ---

    def _identity_key(self, record: Dict[str, Any]) -> tuple:
        return tuple(record.get(f, "") for f in self._identity_fields)

    def _find_ambiguous_neighborhoods(self) -> List[List[Dict[str, Any]]]:
        neighborhoods: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        for record in self._records:
            neighborhoods[self._identity_key(record)].append(record)
        return [n for n in neighborhoods.values() if len(n) > 1]

    def _remaining_ambiguity_pairs(self, neighborhoods, selected_fields):
        total = 0
        for neighborhood in neighborhoods:
            buckets: Dict[tuple, int] = defaultdict(int)
            for record in neighborhood:
                key = tuple(record.get(f, "") for f in selected_fields) if selected_fields else ("_",)
                buckets[key] += 1
            for count in buckets.values():
                if count > 1:
                    total += count * (count - 1) // 2
        return total

    def _infer_ladder(self, neighborhoods, candidate_fields, max_depth):
        if not neighborhoods:
            return []
        chosen = []
        remaining = list(candidate_fields)
        ladder = []
        for _ in range(max_depth):
            if not remaining:
                break
            pairs_before = self._remaining_ambiguity_pairs(neighborhoods, chosen)
            if pairs_before == 0:
                break
            best_field, best_rate = None, 0.0
            for f in remaining:
                pairs_after = self._remaining_ambiguity_pairs(neighborhoods, chosen + [f])
                rate = (pairs_before - pairs_after) / pairs_before if pairs_before else 0.0
                if rate > best_rate:
                    best_field, best_rate = f, rate
            if best_field is None or best_rate <= 0:
                break
            ladder.append(LadderRung(field_name=best_field, ambiguity_reduction_rate=best_rate))
            chosen.append(best_field)
            remaining = [f for f in remaining if f != best_field]
            if self._remaining_ambiguity_pairs(neighborhoods, chosen) == 0:
                break
        return ladder

    def _build_index(self):
        self._index = {}
        for record in self._records:
            id_key = self._identity_key(record)
            if id_key not in self._index:
                self._index[id_key] = {}
            node = self._index[id_key]
            for rung in self._ladder:
                val = record.get(rung.field_name, "")
                if val not in node:
                    node[val] = {}
                node = node[val]
            if "_records" not in node:
                node["_records"] = []
            node["_records"].append(record)

    def _count_in_subtree(self, node):
        if "_records" in node:
            return len(node["_records"])
        return sum(self._count_in_subtree(c) for k, c in node.items()
                   if k != "_records" and isinstance(c, dict))

    def _collect_from_subtree(self, node):
        if "_records" in node:
            return list(node["_records"])
        results = []
        for k, c in node.items():
            if k != "_records" and isinstance(c, dict):
                results.extend(self._collect_from_subtree(c))
        return results

    def _describe_route(self, stages_used):
        parts = ["identity"]
        for i, rung in enumerate(self._ladder):
            if i + 1 < stages_used:
                parts.append(rung.field_name)
        return " -> ".join(parts)
