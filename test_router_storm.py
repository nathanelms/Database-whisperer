"""test_router_storm.py

Test the semantic router on NOAA Storm Events 2023 (~75k records).
Completely non-biomedical domain to prove the method is domain-agnostic.

Usage:
    python test_router_storm.py
"""

from __future__ import annotations

import csv
import random
import time
from typing import Any, Dict, List

from semantic_router import SemanticRouter


def load_storm_events(path: str = "storm_events_2023.csv") -> List[Dict[str, Any]]:
    """
    Load NOAA storm events into clean dicts.

    What this does:
    - Reads the CSV and normalizes the fields we care about.

    Why this exists:
    - The router needs dicts with consistent field names.

    What assumption it is making:
    - These fields capture enough of the storm event structure for routing.
    """
    records = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize damage into a category.
            damage_raw = (row.get("DAMAGE_PROPERTY") or "").strip()
            if not damage_raw or damage_raw == "0":
                damage_cat = "none"
            elif damage_raw.endswith("K"):
                damage_cat = "thousands"
            elif damage_raw.endswith("M"):
                damage_cat = "millions"
            elif damage_raw.endswith("B"):
                damage_cat = "billions"
            else:
                damage_cat = "other"

            record = {
                "state": (row.get("STATE") or "").strip(),
                "event_type": (row.get("EVENT_TYPE") or "").strip(),
                "month": (row.get("MONTH_NAME") or "").strip(),
                "county": (row.get("CZ_NAME") or "").strip(),
                "source": (row.get("SOURCE") or "").strip(),
                "damage_category": damage_cat,
                "magnitude": (row.get("MAGNITUDE") or "").strip(),
                "injuries": (row.get("INJURIES_DIRECT") or "0").strip(),
                "deaths": (row.get("DEATHS_DIRECT") or "0").strip(),
                "tor_scale": (row.get("TOR_F_SCALE") or "").strip(),
                "event_id": (row.get("EVENT_ID") or "").strip(),
            }
            if record["state"] and record["event_type"]:
                records.append(record)
    return records


def run_speed_test(
    router: SemanticRouter,
    queries: List[Dict[str, Any]],
    ask_field: str,
) -> Dict[str, Any]:
    """Run routed and flat-scan queries side by side."""
    total_routed = 0
    total_flat = 0
    routed_correct = 0
    flat_correct = 0
    routed_confused = 0
    flat_confused = 0
    n = len(queries)

    for q in queries:
        routed = router.query(q, ask_field=ask_field)
        flat = router.flat_scan(q, ask_field=ask_field)

        total_routed += routed.records_examined
        total_flat += flat.records_examined

        expected = q.get(ask_field, "")
        if routed.answer == expected:
            routed_correct += 1
        elif routed.answer is not None:
            routed_confused += 1

        if flat.answer == expected:
            flat_correct += 1
        elif flat.answer is not None:
            flat_confused += 1

    return {
        "queries": n,
        "total_records": router.explain()["total_records"],
        "routed_examined": total_routed,
        "flat_examined": total_flat,
        "speedup": total_flat / max(total_routed, 1),
        "avg_routed_examined": total_routed / max(n, 1),
        "avg_flat_examined": total_flat / max(n, 1),
        "routed_accuracy": routed_correct / max(n, 1),
        "flat_accuracy": flat_correct / max(n, 1),
        "routed_confusion_rate": routed_confused / max(n, 1),
        "flat_confusion_rate": flat_confused / max(n, 1),
    }


def main():
    print("Loading NOAA Storm Events 2023...")
    records = load_storm_events()
    print(f"  Loaded {len(records)} storm event records")

    # Unique states and event types for context.
    states = set(r["state"] for r in records)
    event_types = set(r["event_type"] for r in records)
    print(f"  Unique states: {len(states)}")
    print(f"  Unique event types: {len(event_types)}")

    # Build router: state + event_type as identity.
    print("\nBuilding semantic router...")
    t0 = time.time()

    router = SemanticRouter()
    router.ingest(
        records=records,
        identity_fields=["state", "event_type"],
        provenance_fields=["event_id"],
        max_ladder_depth=4,
    )

    build_time = time.time() - t0
    print(f"  Built in {build_time:.2f}s")

    info = router.explain()
    print(f"\n=== Router Structure ===")
    print(f"  Total records: {info['total_records']}")
    print(f"  Identity fields: {info['identity_fields']}")
    print(f"  Identity neighborhoods: {info['identity_neighborhoods']}")
    print(f"  Ambiguous neighborhoods: {info['ambiguous_neighborhoods']}")
    print(f"  Discovered ladder:")
    for rung in info["ladder"]:
        print(f"    Rung {rung['rung']}: {rung['field']} (reduction={rung['ambiguity_reduction_rate']})")

    # Sample queries from the dataset.
    rng = random.Random(42)
    sample_size = min(1000, len(records))
    query_records = rng.sample(records, sample_size)

    # Test 1: retrieve source.
    print(f"\n=== Speed Test: retrieve 'source' ({sample_size} queries) ===")
    r1 = run_speed_test(router, query_records, ask_field="source")
    print(f"  Routed: avg {r1['avg_routed_examined']:.1f} records examined")
    print(f"  Flat:   avg {r1['avg_flat_examined']:.1f} records examined")
    print(f"  Speedup: {r1['speedup']:.1f}x")
    print(f"  Routed accuracy: {r1['routed_accuracy']:.2%}")
    print(f"  Routed confusion: {r1['routed_confusion_rate']:.2%}")
    print(f"  Flat accuracy:   {r1['flat_accuracy']:.2%}")
    print(f"  Flat confusion:   {r1['flat_confusion_rate']:.2%}")

    # Test 2: retrieve damage_category.
    print(f"\n=== Speed Test: retrieve 'damage_category' ({sample_size} queries) ===")
    r2 = run_speed_test(router, query_records, ask_field="damage_category")
    print(f"  Routed: avg {r2['avg_routed_examined']:.1f} records examined")
    print(f"  Flat:   avg {r2['avg_flat_examined']:.1f} records examined")
    print(f"  Speedup: {r2['speedup']:.1f}x")
    print(f"  Routed accuracy: {r2['routed_accuracy']:.2%}")
    print(f"  Routed confusion: {r2['routed_confusion_rate']:.2%}")
    print(f"  Flat accuracy:   {r2['flat_accuracy']:.2%}")
    print(f"  Flat confusion:   {r2['flat_confusion_rate']:.2%}")

    # Test 3: retrieve injuries.
    print(f"\n=== Speed Test: retrieve 'injuries' ({sample_size} queries) ===")
    r3 = run_speed_test(router, query_records, ask_field="injuries")
    print(f"  Routed: avg {r3['avg_routed_examined']:.1f} records examined")
    print(f"  Flat:   avg {r3['avg_flat_examined']:.1f} records examined")
    print(f"  Speedup: {r3['speedup']:.1f}x")
    print(f"  Routed accuracy: {r3['routed_accuracy']:.2%}")
    print(f"  Routed confusion: {r3['routed_confusion_rate']:.2%}")
    print(f"  Flat accuracy:   {r3['flat_accuracy']:.2%}")
    print(f"  Flat confusion:   {r3['flat_confusion_rate']:.2%}")

    # Summary.
    avg_speedup = (r1["speedup"] + r2["speedup"] + r3["speedup"]) / 3
    avg_routed_acc = (r1["routed_accuracy"] + r2["routed_accuracy"] + r3["routed_accuracy"]) / 3
    avg_routed_conf = (r1["routed_confusion_rate"] + r2["routed_confusion_rate"] + r3["routed_confusion_rate"]) / 3

    print(f"\n=== Summary ===")
    print(f"  Dataset: NOAA Storm Events 2023")
    print(f"  Records: {len(records)}")
    print(f"  Router build time: {build_time:.2f}s")
    print(f"  Ladder discovered: {' -> '.join(['identity'] + router.ladder_fields)}")
    print(f"  Average speedup: {avg_speedup:.1f}x")
    print(f"  Average routed accuracy: {avg_routed_acc:.2%}")
    print(f"  Average routed confusion: {avg_routed_conf:.2%}")

    # Compare ladders.
    print(f"\n=== Cross-Domain Ladder Comparison ===")
    print(f"  CIViC (oncology):   identity -> rating -> therapies -> significance -> evidence_level")
    print(f"  Storm Events:       identity -> {' -> '.join(router.ladder_fields)}")
    print(f"  Same procedure, different ladder = domain-agnostic method")


if __name__ == "__main__":
    main()
