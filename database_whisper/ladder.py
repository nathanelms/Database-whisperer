"""Incremental and sleep-based ladder architectures.

LiveRouter: records arrive one at a time, ladder self-organizes.
Memory: short-term buffer + sleep consolidation into long-term structured memory.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
from collections import defaultdict
import time

from ._types import RouteResult, LadderRung, InsertEvent, SleepEvent


class LiveRouter:
    """
    Incrementally self-organizing discriminator ladder.

    Records arrive one at a time. The ladder discovers its structure
    from the data as it grows.

    Usage:
        router = LiveRouter(identity_fields=["gene", "disease"])
        for record in stream:
            event = router.insert(record)
            if event.ladder_changed:
                print(f"Structure shifted: {event.new_ladder}")
    """

    def __init__(
        self,
        identity_fields: List[str],
        provenance_fields: Optional[List[str]] = None,
        max_ladder_depth: int = 4,
    ) -> None:
        self._identity_fields = list(identity_fields)
        self._provenance_fields = list(provenance_fields or [])
        self._max_depth = max_ladder_depth
        self._records: List[Dict[str, Any]] = []
        self._neighborhoods: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        self._ladder: List[LadderRung] = []
        self._index: Dict[tuple, Any] = {}
        self._candidate_fields: List[str] = []
        self._events: List[InsertEvent] = []
        self._ready = False
        self._record_count = 0

    @property
    def ladder_fields(self) -> List[str]:
        return [r.field_name for r in self._ladder]

    @property
    def record_count(self) -> int:
        return self._record_count

    @property
    def events(self) -> List[InsertEvent]:
        return self._events

    def insert(self, record: Dict[str, Any]) -> InsertEvent:
        """Insert one record. Returns an event describing what happened."""
        t0 = time.perf_counter()
        self._record_count += 1
        self._records.append(record)
        id_key = tuple(record.get(f, "") for f in self._identity_fields)
        old_ladder = self.ladder_fields.copy()

        if not self._candidate_fields and record:
            all_fields = set(record.keys())
            excluded = set(self._identity_fields) | set(self._provenance_fields)
            self._candidate_fields = sorted(all_fields - excluded)

        self._neighborhoods[id_key].append(record)
        ns = len(self._neighborhoods[id_key])

        if self._record_count == 1:
            self._build_index()
            self._ready = True
            event = InsertEvent("first", self._record_count, id_key, 1, False,
                                [], [], [], [], False, (time.perf_counter()-t0)*1000)
            self._events.append(event)
            return event

        if ns == 1:
            self._insert_into_index(record, id_key)
            event = InsertEvent("slot", self._record_count, id_key, 1, False,
                                old_ladder, old_ladder, [], [], False,
                                (time.perf_counter()-t0)*1000)
            self._events.append(event)
            return event

        needs_reorg = self._check_collision(record, id_key)
        if not needs_reorg:
            self._insert_into_index(record, id_key)
            etype = "grow" if ns == 2 else "slot"
            event = InsertEvent(etype, self._record_count, id_key, ns, False,
                                old_ladder, old_ladder, [], [], False,
                                (time.perf_counter()-t0)*1000)
            self._events.append(event)
            return event

        self._reorganize()
        new_ladder = self.ladder_fields
        added = [f for f in new_ladder if f not in old_ladder]
        removed = [f for f in old_ladder if f not in new_ladder]
        shared_old = [f for f in old_ladder if f in new_ladder]
        shared_new = [f for f in new_ladder if f in old_ladder]

        event = InsertEvent("reorganize", self._record_count, id_key, ns, True,
                            old_ladder, new_ladder, added, removed,
                            shared_old != shared_new,
                            (time.perf_counter()-t0)*1000)
        self._events.append(event)
        return event

    def query(self, query_fields: Dict[str, Any], ask_field: str) -> RouteResult:
        """Route a query through the live ladder."""
        total = len(self._records)
        if not self._ready or total == 0:
            return RouteResult(None, 0, 0, "empty", [])

        id_key = tuple(query_fields.get(f, "") for f in self._identity_fields)
        if id_key not in self._index:
            return RouteResult(None, 0, total, "identity_miss", [0])

        node = self._index[id_key]
        stages = [self._count_in_subtree(node)]
        for rung in self._ladder:
            val = query_fields.get(rung.field_name, "")
            if val in node:
                node = node[val]
                stages.append(self._count_in_subtree(node))
            else:
                stages.append(stages[-1])

        leaves = self._collect_from_subtree(node)
        if not leaves:
            return RouteResult(None, 0, total, "leaf_miss", stages)

        best = leaves[0]
        return RouteResult(best.get(ask_field), len(leaves), total,
                           "identity -> " + " -> ".join(self.ladder_fields),
                           stages, best, len(leaves)-1)

    def stabilization_report(self) -> Dict[str, Any]:
        """Analyze how the ladder evolved over time."""
        reorgs = [e for e in self._events if e.event_type == "reorganize"]
        total = len(self._events)
        last_reorg = reorgs[-1].record_num if reorgs else 0
        return {
            "total_records": total,
            "total_reorgs": len(reorgs),
            "reorg_rate": len(reorgs) / max(total, 1),
            "last_reorg_at": last_reorg,
            "stability_pct": (total - last_reorg) / max(total, 1),
            "final_ladder": self.ladder_fields,
        }

    # --- Internal ---

    def _check_collision(self, record, id_key):
        neighborhood = self._neighborhoods[id_key]
        if not self._ladder:
            return len(neighborhood) >= 2
        new_path = tuple(record.get(r.field_name, "") for r in self._ladder)
        for other in neighborhood:
            if other is record:
                continue
            if tuple(other.get(r.field_name, "") for r in self._ladder) == new_path:
                return True
        return False

    def _reorganize(self):
        ambiguous = [n for n in self._neighborhoods.values() if len(n) > 1]
        if not ambiguous:
            self._ladder = []
        else:
            self._ladder = self._infer_ladder(ambiguous)
        self._build_index()

    def _infer_ladder(self, neighborhoods):
        chosen, remaining, ladder = [], list(self._candidate_fields), []
        for _ in range(self._max_depth):
            if not remaining:
                break
            pairs_before = self._count_pairs(neighborhoods, chosen)
            if pairs_before == 0:
                break
            best_field, best_rate = None, 0.0
            for f in remaining:
                pairs_after = self._count_pairs(neighborhoods, chosen + [f])
                rate = (pairs_before - pairs_after) / pairs_before
                if rate > best_rate:
                    best_field, best_rate = f, rate
            if best_field is None or best_rate <= 0:
                break
            ladder.append(LadderRung(best_field, best_rate))
            chosen.append(best_field)
            remaining = [f for f in remaining if f != best_field]
            if self._count_pairs(neighborhoods, chosen) == 0:
                break
        return ladder

    def _count_pairs(self, neighborhoods, fields):
        total = 0
        for n in neighborhoods:
            buckets: Dict[tuple, int] = defaultdict(int)
            for rec in n:
                key = tuple(rec.get(f, "") for f in fields) if fields else ("_",)
                buckets[key] += 1
            for c in buckets.values():
                if c > 1:
                    total += c * (c - 1) // 2
        return total

    def _build_index(self):
        self._index = {}
        for rec in self._records:
            self._insert_into_index(rec, tuple(rec.get(f, "") for f in self._identity_fields))

    def _insert_into_index(self, record, id_key):
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


class Memory:
    """
    Memory with short-term buffer and sleep-based consolidation.

    Fast inserts (O(1) append to buffer). Periodic sleep consolidation
    into long-term structured memory. Adaptive sleep thresholds.

    Usage:
        mem = Memory(identity_fields=["gene", "disease"])
        for fact in facts:
            mem.insert(fact)
        result = mem.query({"gene": "BRAF"}, ask_field="therapy")
    """

    def __init__(
        self,
        identity_fields: List[str],
        provenance_fields: Optional[List[str]] = None,
        max_ladder_depth: int = 4,
        sleep_threshold: int = 100,
        core_rungs: int = 2,
    ) -> None:
        self._identity_fields = list(identity_fields)
        self._provenance_fields = list(provenance_fields or [])
        self._max_depth = max_ladder_depth
        self._sleep_threshold = sleep_threshold
        self._core_rungs = core_rungs

        self._lt_records: List[Dict[str, Any]] = []
        self._lt_neighborhoods: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        self._lt_ladder: List[LadderRung] = []
        self._lt_index: Dict[tuple, Any] = {}
        self._candidate_fields: List[str] = []
        self._buffer: List[Dict[str, Any]] = []
        self._sleep_events: List[SleepEvent] = []
        self._total_inserts = 0
        self._total_sleeps = 0

    @property
    def ladder_fields(self) -> List[str]:
        return [r.field_name for r in self._lt_ladder]

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def total_records(self) -> int:
        return len(self._lt_records) + len(self._buffer)

    @property
    def sleep_pressure(self) -> float:
        return len(self._buffer) / max(self._sleep_threshold, 1)

    def insert(self, record: Dict[str, Any]) -> str:
        """Insert into short-term buffer. Returns 'buffered' or 'buffered_and_slept'."""
        self._total_inserts += 1
        if not self._candidate_fields and record:
            excluded = set(self._identity_fields) | set(self._provenance_fields)
            self._candidate_fields = sorted(set(record.keys()) - excluded)

        self._buffer.append(record)
        if len(self._buffer) >= self._sleep_threshold:
            self.sleep()
            return "buffered_and_slept"
        return "buffered"

    def sleep(self) -> SleepEvent:
        """Consolidate buffer into long-term memory."""
        t0 = time.perf_counter()
        self._total_sleeps += 1
        old_ladder = self.ladder_fields.copy()
        buffer = list(self._buffer)
        self._buffer = []

        if not buffer:
            event = SleepEvent(self._total_sleeps, 0, 0, 0, False,
                               old_ladder, old_ladder, [], 0, 0, 0, [])
            self._sleep_events.append(event)
            return event

        slotted, collisions_list, new_nbrs = 0, [], 0

        for rec in buffer:
            self._lt_records.append(rec)
            id_key = tuple(rec.get(f, "") for f in self._identity_fields)
            self._lt_neighborhoods[id_key].append(rec)
            ns = len(self._lt_neighborhoods[id_key])

            if ns == 1:
                new_nbrs += 1
                slotted += 1
                continue

            if not self._lt_ladder:
                collisions_list.append(rec)
                continue

            new_path = tuple(rec.get(r.field_name, "") for r in self._lt_ladder)
            collision = any(
                tuple(o.get(r.field_name, "") for r in self._lt_ladder) == new_path
                for o in self._lt_neighborhoods[id_key] if o is not rec
            )
            if collision:
                collisions_list.append(rec)
            else:
                slotted += 1

        leaf_adjusted, core_shifted, surprises = 0, False, []

        if collisions_list or not self._lt_ladder:
            ambiguous = [n for n in self._lt_neighborhoods.values() if len(n) > 1]
            if ambiguous:
                new_ladder = self._infer_ladder(ambiguous)
                new_fields = [r.field_name for r in new_ladder]
                old_core = old_ladder[:self._core_rungs]
                new_core = new_fields[:self._core_rungs]
                if old_core != new_core:
                    core_shifted = True
                    surprises.append(f"CORE SHIFT: {old_core} -> {new_core}")
                self._lt_ladder = new_ladder
                leaf_adjusted = len(collisions_list)
            else:
                self._lt_ladder = []

        self._rebuild_index()
        new_ladder_fields = self.ladder_fields

        event = SleepEvent(
            self._total_sleeps, len(buffer), slotted, leaf_adjusted,
            core_shifted, old_ladder, new_ladder_fields, [],
            new_nbrs, len(collisions_list),
            (time.perf_counter()-t0)*1000, surprises,
        )
        self._sleep_events.append(event)
        return event

    def query(self, query_fields: Dict[str, Any], ask_field: str) -> RouteResult:
        """Query both short-term and long-term memory."""
        total = self.total_records
        id_key = tuple(query_fields.get(f, "") for f in self._identity_fields)

        # Long-term (routed)
        lt_answer, lt_examined, lt_record = None, 0, None
        if self._lt_records and self._lt_index and id_key in self._lt_index:
            node = self._lt_index[id_key]
            for rung in self._lt_ladder:
                val = query_fields.get(rung.field_name, "")
                if val in node:
                    node = node[val]
            leaves = self._collect_from_subtree(node)
            lt_examined = len(leaves)
            if leaves:
                lt_answer = leaves[0].get(ask_field)
                lt_record = leaves[0]

        # Short-term (flat scan)
        for rec in self._buffer:
            if tuple(rec.get(f, "") for f in self._identity_fields) == id_key:
                return RouteResult(rec.get(ask_field), lt_examined + len(self._buffer),
                                   total, "short_term", [], rec, 0)

        if lt_answer is not None:
            return RouteResult(lt_answer, lt_examined + len(self._buffer),
                               total, "long_term", [], lt_record, 0)

        return RouteResult(None, lt_examined + len(self._buffer), total, "miss", [])

    # --- Internal ---

    def _infer_ladder(self, neighborhoods):
        chosen, remaining, ladder = [], list(self._candidate_fields), []
        for _ in range(self._max_depth):
            if not remaining:
                break
            pairs_before = self._count_pairs(neighborhoods, chosen)
            if pairs_before == 0:
                break
            best_field, best_rate = None, 0.0
            for f in remaining:
                pairs_after = self._count_pairs(neighborhoods, chosen + [f])
                rate = (pairs_before - pairs_after) / pairs_before
                if rate > best_rate:
                    best_field, best_rate = f, rate
            if best_field is None or best_rate <= 0:
                break
            ladder.append(LadderRung(best_field, best_rate))
            chosen.append(best_field)
            remaining = [f for f in remaining if f != best_field]
            if self._count_pairs(neighborhoods, chosen) == 0:
                break
        return ladder

    def _count_pairs(self, neighborhoods, fields):
        total = 0
        for n in neighborhoods:
            buckets: Dict[tuple, int] = defaultdict(int)
            for rec in n:
                key = tuple(rec.get(f, "") for f in fields) if fields else ("_",)
                buckets[key] += 1
            for c in buckets.values():
                if c > 1:
                    total += c * (c - 1) // 2
        return total

    def _rebuild_index(self):
        self._lt_index = {}
        for rec in self._lt_records:
            id_key = tuple(rec.get(f, "") for f in self._identity_fields)
            if id_key not in self._lt_index:
                self._lt_index[id_key] = {}
            node = self._lt_index[id_key]
            for rung in self._lt_ladder:
                val = rec.get(rung.field_name, "")
                if val not in node:
                    node[val] = {}
                node = node[val]
            if "_records" not in node:
                node["_records"] = []
            node["_records"].append(rec)

    def _collect_from_subtree(self, node):
        if "_records" in node:
            return list(node["_records"])
        results = []
        for k, c in node.items():
            if k != "_records" and isinstance(c, dict):
                results.extend(self._collect_from_subtree(c))
        return results
