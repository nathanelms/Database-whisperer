"""living_ladder.py

Incremental self-organizing discriminator ladders.

The core extension to SemanticRouter: instead of batch ingest, records arrive
one at a time. The ladder grows, stabilizes, and occasionally reorganizes
as new data changes the disambiguation structure.

Three things happen on each insert:
    1. SLOT — the record fits the existing ladder. Insert at leaf. Fast.
    2. GROW — the record creates new ambiguity but existing rungs resolve it.
       Insert and extend the index branch. Medium.
    3. REORGANIZE — the record creates ambiguity that the current ladder
       cannot resolve. Re-infer affected rungs. Slow but rare.

The stabilization curve — how often reorganizations happen as data grows —
is the dataset's structural learning curve.

Usage:
    ladder = LivingLadder(identity_fields=["gene", "variant", "disease"])
    for record in stream:
        event = ladder.insert(record)
        # event.type is "slot", "grow", or "reorganize"
        # event tells you what changed and why
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import time

from semantic_router import SemanticRouter, RouteResult, LadderRung


# ---------------------------------------------------------------------------
# Event types — what happened when a record was inserted
# ---------------------------------------------------------------------------

@dataclass
class InsertEvent:
    """What happened when a record was inserted into the living ladder."""
    event_type: str  # "slot", "grow", "reorganize", "first"
    record_num: int  # which record this was (1-indexed)
    identity_key: tuple
    neighborhood_size: int  # how many records share this identity after insert
    ladder_changed: bool
    old_ladder: List[str]  # field names before
    new_ladder: List[str]  # field names after
    rungs_added: List[str]  # new rungs that appeared
    rungs_removed: List[str]  # rungs that disappeared
    rungs_reordered: bool  # did existing rungs change order
    insert_time_ms: float


# ---------------------------------------------------------------------------
# The Living Ladder
# ---------------------------------------------------------------------------

class LivingLadder:
    """
    An incrementally self-organizing discriminator ladder.

    Records arrive one at a time. The ladder discovers its structure
    from the data as it grows, reorganizing when new records change
    the disambiguation landscape.
    """

    def __init__(
        self,
        identity_fields: List[str],
        provenance_fields: Optional[List[str]] = None,
        max_ladder_depth: int = 4,
        reorg_threshold: float = 0.05,  # reorganize if pair reduction drops >5%
    ) -> None:
        self._identity_fields = list(identity_fields)
        self._provenance_fields = list(provenance_fields or [])
        self._max_depth = max_ladder_depth
        self._reorg_threshold = reorg_threshold

        # All records stored
        self._records: List[Dict[str, Any]] = []

        # Records grouped by identity key
        self._neighborhoods: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)

        # Current ladder and index
        self._ladder: List[LadderRung] = []
        self._index: Dict[tuple, Any] = {}
        self._candidate_fields: List[str] = []

        # History tracking
        self._events: List[InsertEvent] = []
        self._ladder_snapshots: List[Tuple[int, List[str]]] = []  # (record_num, ladder_fields)
        self._ready = False
        self._record_count = 0

    @property
    def ladder_fields(self) -> List[str]:
        return [rung.field_name for rung in self._ladder]

    @property
    def record_count(self) -> int:
        return self._record_count

    @property
    def events(self) -> List[InsertEvent]:
        return self._events

    def _identity_key(self, record: Dict[str, Any]) -> tuple:
        return tuple(record.get(f, "") for f in self._identity_fields)

    # -------------------------------------------------------------------
    # Core: insert one record
    # -------------------------------------------------------------------

    def insert(self, record: Dict[str, Any]) -> InsertEvent:
        """
        Insert one record into the living ladder.

        Returns an InsertEvent describing what happened:
        - "first": first record, no ladder yet
        - "slot": record fits existing ladder, inserted at leaf
        - "grow": new identity neighborhood or new branch, no reorg needed
        - "reorganize": ladder structure changed to accommodate this record
        """
        t0 = time.perf_counter()
        self._record_count += 1
        self._records.append(record)

        id_key = self._identity_key(record)
        old_ladder = self.ladder_fields.copy()

        # Auto-detect candidate fields from first record
        if not self._candidate_fields and record:
            all_fields = set(record.keys())
            excluded = set(self._identity_fields) | set(self._provenance_fields)
            self._candidate_fields = sorted(all_fields - excluded)

        # Add to neighborhood
        self._neighborhoods[id_key].append(record)
        neighborhood_size = len(self._neighborhoods[id_key])

        # Case 1: First record ever
        if self._record_count == 1:
            self._build_index()
            self._ready = True
            event = InsertEvent(
                event_type="first",
                record_num=self._record_count,
                identity_key=id_key,
                neighborhood_size=1,
                ladder_changed=False,
                old_ladder=[], new_ladder=[],
                rungs_added=[], rungs_removed=[],
                rungs_reordered=False,
                insert_time_ms=(time.perf_counter() - t0) * 1000,
            )
            self._events.append(event)
            self._ladder_snapshots.append((self._record_count, []))
            return event

        # Case 2: New identity neighborhood (no ambiguity created)
        if neighborhood_size == 1:
            self._insert_into_index(record, id_key)
            event = InsertEvent(
                event_type="slot",
                record_num=self._record_count,
                identity_key=id_key,
                neighborhood_size=1,
                ladder_changed=False,
                old_ladder=old_ladder, new_ladder=old_ladder,
                rungs_added=[], rungs_removed=[],
                rungs_reordered=False,
                insert_time_ms=(time.perf_counter() - t0) * 1000,
            )
            self._events.append(event)
            return event

        # Case 3: Existing neighborhood — does the record fit the current ladder?
        if neighborhood_size == 2:
            # This neighborhood just became ambiguous. Need to check if ladder handles it.
            needs_reorg = self._check_needs_reorg(id_key)
        else:
            # Neighborhood was already ambiguous. Check if new record fits.
            needs_reorg = self._check_record_fits(record, id_key)

        if not needs_reorg:
            # Record fits — just insert into index
            self._insert_into_index(record, id_key)
            event = InsertEvent(
                event_type="grow" if neighborhood_size == 2 else "slot",
                record_num=self._record_count,
                identity_key=id_key,
                neighborhood_size=neighborhood_size,
                ladder_changed=False,
                old_ladder=old_ladder, new_ladder=old_ladder,
                rungs_added=[], rungs_removed=[],
                rungs_reordered=False,
                insert_time_ms=(time.perf_counter() - t0) * 1000,
            )
            self._events.append(event)
            return event

        # Case 4: Reorganize — re-infer ladder from all current data
        self._reorganize()
        new_ladder = self.ladder_fields

        # Compute what changed
        added = [f for f in new_ladder if f not in old_ladder]
        removed = [f for f in old_ladder if f not in new_ladder]
        shared = [f for f in old_ladder if f in new_ladder]
        new_shared = [f for f in new_ladder if f in old_ladder]
        reordered = shared != new_shared

        event = InsertEvent(
            event_type="reorganize",
            record_num=self._record_count,
            identity_key=id_key,
            neighborhood_size=neighborhood_size,
            ladder_changed=True,
            old_ladder=old_ladder, new_ladder=new_ladder,
            rungs_added=added, rungs_removed=removed,
            rungs_reordered=reordered,
            insert_time_ms=(time.perf_counter() - t0) * 1000,
        )
        self._events.append(event)
        self._ladder_snapshots.append((self._record_count, new_ladder.copy()))
        return event

    # -------------------------------------------------------------------
    # Check if reorganization is needed
    # -------------------------------------------------------------------

    def _check_needs_reorg(self, id_key: tuple) -> bool:
        """Check if the newly-ambiguous neighborhood is resolved by current ladder."""
        neighborhood = self._neighborhoods[id_key]
        if len(neighborhood) < 2:
            return False

        # If no ladder yet, we need to build one
        if not self._ladder:
            return True

        # Check: do the current ladder fields distinguish all records in this neighborhood?
        seen_paths = set()
        for rec in neighborhood:
            path = tuple(rec.get(rung.field_name, "") for rung in self._ladder)
            if path in seen_paths:
                # Two records have identical ladder paths — ladder can't distinguish them
                return True
            seen_paths.add(path)

        return False

    def _check_record_fits(self, record: Dict[str, Any], id_key: tuple) -> bool:
        """Check if a new record in an existing ambiguous neighborhood fits the ladder."""
        neighborhood = self._neighborhoods[id_key]

        # Check if the new record's ladder path is unique within its neighborhood
        new_path = tuple(record.get(rung.field_name, "") for rung in self._ladder)

        for other in neighborhood:
            if other is record:
                continue
            other_path = tuple(other.get(rung.field_name, "") for rung in self._ladder)
            if new_path == other_path:
                # Collision — ladder can't distinguish this new record from an existing one
                # But first check: is this actually a problem? Maybe they have the same answer.
                # For now, flag it as needing reorg
                return True

        return False

    # -------------------------------------------------------------------
    # Reorganize: local-first, full only when structure changes
    # -------------------------------------------------------------------

    def _reorganize(self) -> None:
        """
        Re-infer the ladder, using local evaluation first.

        Strategy:
        1. Try local reorg: re-evaluate only with current ambiguous neighborhoods
           but using a cached pair count to skip unchanged neighborhoods.
        2. Only rebuild the full index if the ladder fields or order changed.
        """
        ambiguous = [n for n in self._neighborhoods.values() if len(n) > 1]

        if not ambiguous:
            self._ladder = []
            self._build_index()
            return

        # Sample-based acceleration: if we have many neighborhoods,
        # evaluate on a sample first, then confirm on full set only
        # if the sampled ladder differs from current.
        if len(ambiguous) > 200:
            # Sample ~20% of ambiguous neighborhoods (min 50)
            import random
            sample_size = max(50, len(ambiguous) // 5)
            rng = random.Random(len(self._records))  # deterministic seed
            sampled = rng.sample(ambiguous, min(sample_size, len(ambiguous)))
            candidate_ladder = self._infer_ladder_from(sampled)
            candidate_fields = [r.field_name for r in candidate_ladder]

            # If sampled ladder matches current top rungs, skip full reorg
            current_fields = self.ladder_fields
            if candidate_fields[:2] == current_fields[:2]:
                # Top rungs stable — only bottom might differ.
                # Accept the candidate (it's close enough) and rebuild index.
                self._ladder = candidate_ladder
                self._build_index()
                return

        # Full reorg (but still faster with the extracted method)
        self._ladder = self._infer_ladder_from(ambiguous)
        self._build_index()

    def _infer_ladder_from(
        self,
        neighborhoods: List[List[Dict[str, Any]]],
    ) -> List[LadderRung]:
        """Infer a ladder from a set of neighborhoods. Extracted for reuse."""
        chosen: List[str] = []
        remaining = list(self._candidate_fields)
        ladder: List[LadderRung] = []

        for _ in range(self._max_depth):
            if not remaining:
                break

            pairs_before = self._remaining_ambiguity_pairs(neighborhoods, chosen)
            if pairs_before == 0:
                break

            best_field = None
            best_rate = 0.0

            for field_name in remaining:
                test_fields = chosen + [field_name]
                pairs_after = self._remaining_ambiguity_pairs(neighborhoods, test_fields)
                rate = (pairs_before - pairs_after) / pairs_before if pairs_before else 0.0
                if rate > best_rate:
                    best_field = field_name
                    best_rate = rate

            if best_field is None or best_rate <= 0:
                break

            ladder.append(LadderRung(field_name=best_field, ambiguity_reduction_rate=best_rate))
            chosen.append(best_field)
            remaining = [f for f in remaining if f != best_field]

            if self._remaining_ambiguity_pairs(neighborhoods, chosen) == 0:
                break

        return ladder

    def _remaining_ambiguity_pairs(
        self,
        neighborhoods: List[List[Dict[str, Any]]],
        selected_fields: List[str],
    ) -> int:
        """Count unresolved pairs after splitting by selected fields."""
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

    # -------------------------------------------------------------------
    # Index management
    # -------------------------------------------------------------------

    def _build_index(self) -> None:
        """Rebuild the full index from scratch."""
        self._index = {}
        for record in self._records:
            self._insert_into_index(record, self._identity_key(record))

    def _insert_into_index(self, record: Dict[str, Any], id_key: tuple) -> None:
        """Insert a single record into the existing index."""
        if id_key not in self._index:
            self._index[id_key] = {}
        node = self._index[id_key]

        for rung in self._ladder:
            rung_value = record.get(rung.field_name, "")
            if rung_value not in node:
                node[rung_value] = {}
            node = node[rung_value]

        if "_records" not in node:
            node["_records"] = []
        node["_records"].append(record)

    # -------------------------------------------------------------------
    # Query (delegates to same logic as SemanticRouter)
    # -------------------------------------------------------------------

    def query(self, query_fields: Dict[str, Any], ask_field: str) -> RouteResult:
        """Route a query through the living ladder."""
        total = len(self._records)
        if not self._ready or total == 0:
            return RouteResult(answer=None, records_examined=0, total_records=0,
                               route_used="empty", candidates_at_each_stage=[])

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

        leaf_records = self._collect_from_subtree(node)
        if not leaf_records:
            return RouteResult(answer=None, records_examined=0, total_records=total,
                               route_used="leaf_miss", candidates_at_each_stage=stages)

        best = leaf_records[0]
        return RouteResult(
            answer=best.get(ask_field),
            records_examined=len(leaf_records),
            total_records=total,
            route_used="identity -> " + " -> ".join(self.ladder_fields),
            candidates_at_each_stage=stages,
            matched_record=best,
            confusion_candidates=len(leaf_records) - 1,
        )

    def _count_in_subtree(self, node: dict) -> int:
        if "_records" in node:
            return len(node["_records"])
        return sum(self._count_in_subtree(c) for k, c in node.items()
                   if k != "_records" and isinstance(c, dict))

    def _collect_from_subtree(self, node: dict) -> List[Dict[str, Any]]:
        if "_records" in node:
            return list(node["_records"])
        results = []
        for k, c in node.items():
            if k != "_records" and isinstance(c, dict):
                results.extend(self._collect_from_subtree(c))
        return results

    # -------------------------------------------------------------------
    # Analysis: the stabilization curve
    # -------------------------------------------------------------------

    def stabilization_report(self) -> Dict[str, Any]:
        """Analyze how the ladder evolved over time."""
        total = len(self._events)
        reorgs = [e for e in self._events if e.event_type == "reorganize"]
        slots = [e for e in self._events if e.event_type == "slot"]
        grows = [e for e in self._events if e.event_type == "grow"]

        # When did each reorg happen?
        reorg_points = [e.record_num for e in reorgs]

        # What was the ladder at each reorg?
        ladder_history = [(e.record_num, e.new_ladder) for e in reorgs]

        # Stabilization point: last reorg
        last_reorg = reorg_points[-1] if reorg_points else 0
        records_after_stable = total - last_reorg

        # Reorg rate over time (sliding window)
        window = max(total // 10, 1)
        reorg_rate_curve = []
        for i in range(0, total, window):
            chunk = self._events[i:i+window]
            reorg_count = sum(1 for e in chunk if e.event_type == "reorganize")
            reorg_rate_curve.append({
                "record_range": f"{i+1}-{min(i+window, total)}",
                "reorg_rate": reorg_count / len(chunk) if chunk else 0,
            })

        # Average insert times by type
        avg_times = {}
        for etype in ["slot", "grow", "reorganize", "first"]:
            events_of_type = [e for e in self._events if e.event_type == etype]
            if events_of_type:
                avg_times[etype] = sum(e.insert_time_ms for e in events_of_type) / len(events_of_type)

        return {
            "total_records": total,
            "total_reorgs": len(reorgs),
            "total_slots": len(slots),
            "total_grows": len(grows),
            "reorg_rate": len(reorgs) / max(total, 1),
            "last_reorg_at": last_reorg,
            "records_after_stable": records_after_stable,
            "stability_pct": records_after_stable / max(total, 1),
            "final_ladder": self.ladder_fields,
            "ladder_history": ladder_history,
            "reorg_points": reorg_points,
            "reorg_rate_curve": reorg_rate_curve,
            "avg_insert_times_ms": avg_times,
        }

    def explain(self) -> Dict[str, Any]:
        """Human-readable summary of current state."""
        return {
            "total_records": self._record_count,
            "identity_fields": self._identity_fields,
            "ladder": [
                {"rung": i+1, "field": r.field_name,
                 "ambiguity_reduction_rate": f"{r.ambiguity_reduction_rate:.2%}"}
                for i, r in enumerate(self._ladder)
            ],
            "identity_neighborhoods": len(self._neighborhoods),
            "ambiguous_neighborhoods": sum(1 for n in self._neighborhoods.values() if len(n) > 1),
        }
