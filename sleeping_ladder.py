"""sleeping_ladder.py

Memory architecture with short-term buffer, sleep consolidation,
and long-term structured routing.

    AWAKE:  fast inserts into short-term buffer (O(1) append)
    SLEEP:  consolidate buffer into long-term ladder
            - most records slot in (no work)
            - collisions get leaf-level adjustment (cheap)
            - core shifts are rare structural surprises (expensive but meaningful)
    QUERY:  check short-term (flat scan, small) + long-term (routed, fast)

The sleep cycle IS the learning. Structural signals emerge during
consolidation, not during insert.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import time

from semantic_router import SemanticRouter, RouteResult, LadderRung


# ---------------------------------------------------------------------------
# Sleep event — what happened during consolidation
# ---------------------------------------------------------------------------

@dataclass
class SleepEvent:
    """What the memory learned during one sleep cycle."""
    cycle_num: int
    records_consolidated: int
    slotted: int           # fit existing structure, no work
    leaf_adjusted: int     # bottom rungs re-evaluated locally
    core_shifted: bool     # top rungs changed — genuine structural surprise
    old_ladder: List[str]
    new_ladder: List[str]
    rungs_changed: List[str]  # which rungs shifted
    new_neighborhoods: int    # entirely new identity neighborhoods
    collisions: int           # records that didn't fit
    sleep_time_ms: float
    structural_surprises: List[str]  # human-readable descriptions


@dataclass
class QueryResult:
    """Combined result from short-term + long-term memory."""
    answer: Optional[str]
    source: str             # "short_term", "long_term", "both", "miss"
    records_examined: int
    total_records: int
    route_used: str
    matched_record: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# The Sleeping Ladder
# ---------------------------------------------------------------------------

class SleepingLadder:
    """
    Memory with short-term buffer and sleep-based consolidation.

    Insert is O(1) — just append to the buffer.
    Sleep consolidates the buffer into the long-term ladder.
    Query checks both short-term and long-term.
    """

    def __init__(
        self,
        identity_fields: List[str],
        provenance_fields: Optional[List[str]] = None,
        max_ladder_depth: int = 4,
        sleep_threshold: int = 100,     # max buffer size before forced sleep
        core_rungs: int = 2,            # top N rungs considered "core" (stable)
        adaptive_sleep: bool = True,    # auto-adjust sleep timing
    ) -> None:
        self._identity_fields = list(identity_fields)
        self._provenance_fields = list(provenance_fields or [])
        self._max_depth = max_ladder_depth
        self._sleep_threshold = sleep_threshold
        self._max_threshold = sleep_threshold  # upper bound
        self._min_threshold = max(10, sleep_threshold // 10)  # lower bound
        self._core_rungs = core_rungs
        self._adaptive_sleep = adaptive_sleep

        # Long-term memory (structured)
        self._lt_records: List[Dict[str, Any]] = []
        self._lt_neighborhoods: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        self._lt_ladder: List[LadderRung] = []
        self._lt_index: Dict[tuple, Any] = {}
        self._candidate_fields: List[str] = []

        # Short-term buffer (unstructured)
        self._buffer: List[Dict[str, Any]] = []
        self._buffer_collisions: List[Dict[str, Any]] = []

        # Adaptive sleep state
        self._recent_collision_count: int = 0  # collisions in current buffer window
        self._recent_insert_count: int = 0     # inserts since last sleep
        self._last_sleep_time: float = time.perf_counter()

        # History
        self._sleep_events: List[SleepEvent] = []
        self._total_inserts: int = 0
        self._total_sleeps: int = 0
        self._awake = True

    @property
    def ladder_fields(self) -> List[str]:
        return [r.field_name for r in self._lt_ladder]

    @property
    def core_fields(self) -> List[str]:
        return self.ladder_fields[:self._core_rungs]

    @property
    def tail_fields(self) -> List[str]:
        return self.ladder_fields[self._core_rungs:]

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def total_records(self) -> int:
        return len(self._lt_records) + len(self._buffer)

    @property
    def sleep_pressure(self) -> float:
        """How much does the memory need to sleep? 0.0 = calm, 1.0 = must sleep."""
        if self._sleep_threshold == 0:
            return 0.0
        return len(self._buffer) / self._sleep_threshold

    def _identity_key(self, record: Dict[str, Any]) -> tuple:
        return tuple(record.get(f, "") for f in self._identity_fields)

    # -------------------------------------------------------------------
    # INSERT (awake) — O(1) append to buffer
    # -------------------------------------------------------------------

    def insert(self, record: Dict[str, Any]) -> str:
        """
        Insert a record into short-term memory.

        Returns:
            "buffered"           — stored, no sleep needed
            "buffered_and_slept" — stored, then auto-sleep triggered

        Sleep triggers (checked in order):
            1. Collision pressure — too many buffer records don't fit the ladder
            2. Buffer fullness   — buffer too large relative to long-term memory
            3. Max threshold     — hard cap on buffer size
        """
        self._total_inserts += 1
        self._recent_insert_count += 1

        # Auto-detect candidate fields from first record
        if not self._candidate_fields and record:
            all_fields = set(record.keys())
            excluded = set(self._identity_fields) | set(self._provenance_fields)
            self._candidate_fields = sorted(all_fields - excluded)

        # O(1) append
        self._buffer.append(record)

        # Cheap collision check against current ladder (O(ladder_depth))
        if self._lt_ladder:
            id_key = self._identity_key(record)
            if id_key in self._lt_neighborhoods and len(self._lt_neighborhoods[id_key]) > 0:
                # Check if this record's path collides with existing records
                new_path = tuple(record.get(r.field_name, "") for r in self._lt_ladder)
                for other in self._lt_neighborhoods[id_key]:
                    other_path = tuple(other.get(r.field_name, "") for r in self._lt_ladder)
                    if new_path == other_path:
                        self._recent_collision_count += 1
                        break

        # Decide whether to sleep
        should_sleep = self._should_sleep()
        if should_sleep:
            self.sleep()
            return "buffered_and_slept"

        return "buffered"

    def _should_sleep(self) -> bool:
        """
        Adaptive sleep decision based on three pressure signals.
        """
        buf_size = len(self._buffer)

        # Hard cap — always sleep
        if buf_size >= self._max_threshold:
            return True

        # Below minimum — never sleep (not enough data to consolidate)
        if buf_size < self._min_threshold:
            return False

        if not self._adaptive_sleep:
            return buf_size >= self._sleep_threshold

        # Signal 1: Collision pressure
        # If >30% of recent inserts are collisions, the current structure
        # doesn't fit what we're seeing. Sleep now.
        if self._recent_insert_count > 0:
            collision_rate = self._recent_collision_count / self._recent_insert_count
            if collision_rate > 0.30:
                return True

        # Signal 2: Buffer size relative to long-term
        # When buffer is >15% of long-term, queries are getting slow
        # because every query flat-scans the buffer too.
        lt_size = len(self._lt_records)
        if lt_size > 0 and buf_size > lt_size * 0.15:
            return True

        # Signal 3: Default threshold
        return buf_size >= self._sleep_threshold

    # -------------------------------------------------------------------
    # SLEEP — consolidate buffer into long-term memory
    # -------------------------------------------------------------------

    def sleep(self) -> SleepEvent:
        """
        Consolidate short-term buffer into long-term structured memory.

        Three tiers of work:
        1. SLOT — record fits existing ladder, append to leaf (free)
        2. LEAF ADJUST — collision at leaf, re-evaluate bottom rungs locally (cheap)
        3. CORE SHIFT — top rungs change (rare, expensive, meaningful)
        """
        t0 = time.perf_counter()
        self._total_sleeps += 1
        old_ladder = self.ladder_fields.copy()
        surprises = []

        buffer = list(self._buffer)
        self._buffer = []
        self._buffer_collisions = []

        # Reset adaptive tracking
        pre_collision_rate = (self._recent_collision_count / max(self._recent_insert_count, 1)
                              if self._recent_insert_count > 0 else 0)
        self._recent_collision_count = 0
        self._recent_insert_count = 0
        self._last_sleep_time = t0

        if not buffer:
            event = SleepEvent(
                cycle_num=self._total_sleeps,
                records_consolidated=0, slotted=0, leaf_adjusted=0,
                core_shifted=False, old_ladder=old_ladder, new_ladder=old_ladder,
                rungs_changed=[], new_neighborhoods=0, collisions=0,
                sleep_time_ms=(time.perf_counter() - t0) * 1000,
                structural_surprises=[],
            )
            self._sleep_events.append(event)
            return event

        slotted = 0
        collisions = []
        new_neighborhoods = 0

        # Phase 1: Try to slot each buffered record into existing long-term structure
        for rec in buffer:
            self._lt_records.append(rec)
            id_key = self._identity_key(rec)
            self._lt_neighborhoods[id_key].append(rec)

            neighborhood = self._lt_neighborhoods[id_key]

            if len(neighborhood) == 1:
                # New neighborhood — no collision possible
                new_neighborhoods += 1
                slotted += 1
                continue

            if not self._lt_ladder:
                # No ladder yet — everything is a collision
                collisions.append(rec)
                continue

            # Check if this record's ladder path is unique in its neighborhood
            new_path = tuple(rec.get(r.field_name, "") for r in self._lt_ladder)
            collision = False
            for other in neighborhood:
                if other is rec:
                    continue
                other_path = tuple(other.get(r.field_name, "") for r in self._lt_ladder)
                if new_path == other_path:
                    collision = True
                    break

            if collision:
                collisions.append(rec)
            else:
                slotted += 1

        # Phase 2: Handle collisions
        leaf_adjusted = 0
        core_shifted = False

        if collisions or not self._lt_ladder:
            # Identify which neighborhoods have collisions
            collision_neighborhoods: Set[tuple] = set()
            for rec in collisions:
                collision_neighborhoods.add(self._identity_key(rec))

            # Get ALL ambiguous neighborhoods for re-evaluation
            all_ambiguous = [n for n in self._lt_neighborhoods.values() if len(n) > 1]

            if not all_ambiguous:
                # No ambiguity — clear ladder
                self._lt_ladder = []
            else:
                # Try local fix first: re-evaluate only collision neighborhoods
                # to see if bottom rungs resolve them
                local_fix_worked = False

                if self._lt_ladder and len(collision_neighborhoods) <= len(all_ambiguous) * 0.3:
                    # Few collisions relative to total — try local adjustment
                    # Test if current top rungs still work globally
                    top_fields = self.ladder_fields[:self._core_rungs]

                    # Re-infer just the tail rungs
                    new_ladder = self._infer_ladder(all_ambiguous)
                    new_fields = [r.field_name for r in new_ladder]

                    if new_fields[:self._core_rungs] == top_fields:
                        # Core is stable — only tail changed
                        self._lt_ladder = new_ladder
                        leaf_adjusted = len(collisions)
                        local_fix_worked = True

                        if new_fields[self._core_rungs:] != old_ladder[self._core_rungs:]:
                            tail_changes = [f for f in new_fields[self._core_rungs:]
                                            if f not in old_ladder[self._core_rungs:]]
                            if tail_changes:
                                surprises.append(
                                    f"Tail adjusted: {tail_changes} entered bottom rungs"
                                )

                if not local_fix_worked:
                    # Full re-evaluation needed
                    new_ladder = self._infer_ladder(all_ambiguous)
                    new_fields = [r.field_name for r in new_ladder]

                    if new_fields[:self._core_rungs] != old_ladder[:self._core_rungs]:
                        core_shifted = True
                        old_core = old_ladder[:self._core_rungs]
                        new_core = new_fields[:self._core_rungs]
                        surprises.append(
                            f"CORE SHIFT: {old_core} -> {new_core}. "
                            f"Structural understanding changed."
                        )

                    self._lt_ladder = new_ladder
                    leaf_adjusted = len(collisions)

        # Phase 3: Rebuild index
        self._rebuild_index()

        new_ladder_fields = self.ladder_fields
        rungs_changed = []
        for i, (old, new) in enumerate(zip(
            old_ladder + [''] * 10,
            new_ladder_fields + [''] * 10
        )):
            if old != new and (old or new):
                rungs_changed.append(f"rung{i+1}: {old or '(none)'} -> {new or '(none)'}")

        event = SleepEvent(
            cycle_num=self._total_sleeps,
            records_consolidated=len(buffer),
            slotted=slotted,
            leaf_adjusted=leaf_adjusted,
            core_shifted=core_shifted,
            old_ladder=old_ladder,
            new_ladder=new_ladder_fields,
            rungs_changed=rungs_changed,
            new_neighborhoods=new_neighborhoods,
            collisions=len(collisions),
            sleep_time_ms=(time.perf_counter() - t0) * 1000,
            structural_surprises=surprises,
        )
        self._sleep_events.append(event)

        # Adaptive threshold adjustment
        if self._adaptive_sleep and len(buffer) > 0:
            collision_pct = len(collisions) / len(buffer)
            if core_shifted:
                # Core shifted — sleep sooner next time, structure is volatile
                self._sleep_threshold = max(self._min_threshold,
                                            self._sleep_threshold // 2)
            elif collision_pct < 0.05:
                # Almost no collisions — structure is stable, sleep less often
                self._sleep_threshold = min(self._max_threshold,
                                            int(self._sleep_threshold * 1.5))
            elif collision_pct > 0.30:
                # Many collisions — sleep sooner
                self._sleep_threshold = max(self._min_threshold,
                                            int(self._sleep_threshold * 0.7))
            # else: moderate collisions, keep current threshold

        return event

    # -------------------------------------------------------------------
    # QUERY — check both memories
    # -------------------------------------------------------------------

    def query(self, query_fields: Dict[str, Any], ask_field: str) -> QueryResult:
        """
        Query both short-term buffer and long-term ladder.
        Long-term is routed (fast). Short-term is flat-scanned (small).
        """
        total = self.total_records
        id_key = tuple(query_fields.get(f, "") for f in self._identity_fields)

        # Search long-term (routed)
        lt_answer = None
        lt_examined = 0
        lt_record = None
        route_desc = "empty"

        if self._lt_records and self._lt_index:
            if id_key in self._lt_index:
                node = self._lt_index[id_key]
                for rung in self._lt_ladder:
                    rung_val = query_fields.get(rung.field_name, "")
                    if rung_val in node:
                        node = node[rung_val]
                    # else stay at current node

                leaves = self._collect_leaves(node)
                lt_examined = len(leaves)
                if leaves:
                    lt_answer = leaves[0].get(ask_field)
                    lt_record = leaves[0]
                    route_desc = "identity -> " + " -> ".join(self.ladder_fields)

        # Search short-term (flat scan)
        st_answer = None
        st_examined = len(self._buffer)

        for rec in self._buffer:
            if self._identity_key(rec) == id_key:
                st_answer = rec.get(ask_field)
                lt_record = rec  # prefer short-term if found (more recent)
                break

        # Combine
        if st_answer is not None:
            return QueryResult(
                answer=st_answer, source="short_term",
                records_examined=lt_examined + st_examined,
                total_records=total, route_used=route_desc,
                matched_record=lt_record,
            )
        elif lt_answer is not None:
            return QueryResult(
                answer=lt_answer, source="long_term",
                records_examined=lt_examined + st_examined,
                total_records=total, route_used=route_desc,
                matched_record=lt_record,
            )
        else:
            return QueryResult(
                answer=None, source="miss",
                records_examined=lt_examined + st_examined,
                total_records=total, route_used="miss",
            )

    # -------------------------------------------------------------------
    # Internal: ladder inference
    # -------------------------------------------------------------------

    def _infer_ladder(
        self, neighborhoods: List[List[Dict[str, Any]]]
    ) -> List[LadderRung]:
        chosen: List[str] = []
        remaining = list(self._candidate_fields)
        ladder: List[LadderRung] = []

        for _ in range(self._max_depth):
            if not remaining:
                break
            pairs_before = self._count_pairs(neighborhoods, chosen)
            if pairs_before == 0:
                break

            best_field = None
            best_rate = 0.0
            for f in remaining:
                test = chosen + [f]
                pairs_after = self._count_pairs(neighborhoods, test)
                rate = (pairs_before - pairs_after) / pairs_before
                if rate > best_rate:
                    best_field = f
                    best_rate = rate

            if best_field is None or best_rate <= 0:
                break

            ladder.append(LadderRung(field_name=best_field, ambiguity_reduction_rate=best_rate))
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
            id_key = self._identity_key(rec)
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

    def _collect_leaves(self, node: dict) -> List[Dict[str, Any]]:
        if "_records" in node:
            return list(node["_records"])
        results = []
        for k, v in node.items():
            if k != "_records" and isinstance(v, dict):
                results.extend(self._collect_leaves(v))
        return results

    # -------------------------------------------------------------------
    # Reporting
    # -------------------------------------------------------------------

    def report(self) -> Dict[str, Any]:
        return {
            "total_inserts": self._total_inserts,
            "total_sleeps": self._total_sleeps,
            "long_term_records": len(self._lt_records),
            "buffer_size": len(self._buffer),
            "sleep_pressure": self.sleep_pressure,
            "ladder": self.ladder_fields,
            "core": self.core_fields,
            "tail": self.tail_fields,
            "neighborhoods": len(self._lt_neighborhoods),
            "ambiguous": sum(1 for n in self._lt_neighborhoods.values() if len(n) > 1),
            "sleep_history": [
                {
                    "cycle": e.cycle_num,
                    "consolidated": e.records_consolidated,
                    "slotted": e.slotted,
                    "collisions": e.collisions,
                    "leaf_adjusted": e.leaf_adjusted,
                    "core_shifted": e.core_shifted,
                    "time_ms": e.sleep_time_ms,
                    "surprises": e.structural_surprises,
                    "ladder": e.new_ladder,
                }
                for e in self._sleep_events
            ],
        }
