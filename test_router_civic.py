"""test_router_civic.py

Test the semantic router on the full CIViC nightly evidence export (~4800 records).

This is the real-scale test. The question is:
- Does the router still discover a meaningful discriminator ladder?
- Does routing give a real speedup over flat scan?
- Does routing maintain accuracy (low confusion)?

Usage:
    python test_router_civic.py
"""

from __future__ import annotations

import csv
import random
import time
from typing import Any, Dict, List

from semantic_router import SemanticRouter


def load_civic_evidence(path: str = "civic_evidence_full.tsv") -> List[Dict[str, Any]]:
    """
    Load the full CIViC nightly evidence export into a list of dicts.

    What this does:
    - Reads the TSV and normalizes the fields we care about into a clean dict.

    Why this exists:
    - The router works on dicts, so we need a simple loader.

    What assumption it is making:
    - The nightly export schema has the columns we expect.
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # Normalize into clean fields the router can use.
            record = {
                "molecular_profile": row.get("molecular_profile", "").strip(),
                "disease": row.get("disease", "").strip(),
                "therapies": row.get("therapies", "").strip(),
                "evidence_type": row.get("evidence_type", "").strip(),
                "evidence_direction": row.get("evidence_direction", "").strip(),
                "evidence_level": row.get("evidence_level", "").strip(),
                "significance": row.get("significance", "").strip(),
                "evidence_id": row.get("evidence_id", "").strip(),
                "rating": row.get("rating", "").strip(),
            }
            # Skip records with empty identity fields.
            if record["molecular_profile"] and record["disease"]:
                records.append(record)
    return records


def run_speed_test(
    router: SemanticRouter,
    queries: List[Dict[str, Any]],
    ask_field: str,
) -> Dict[str, Any]:
    """
    Run routed and flat-scan queries side by side and measure speed.

    What this does:
    - For each query, runs both the router and a flat scan, recording
      records examined and correctness.

    Why this exists:
    - The whole point is to prove routing is faster.
    """
    total_routed_examined = 0
    total_flat_examined = 0
    routed_correct = 0
    flat_correct = 0
    routed_confused = 0
    flat_confused = 0
    n = len(queries)

    for q in queries:
        routed = router.query(q, ask_field=ask_field)
        flat = router.flat_scan(q, ask_field=ask_field)

        total_routed_examined += routed.records_examined
        total_flat_examined += flat.records_examined

        # "Correct" = returned the same answer as the expected value in the query.
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
        "routed_examined": total_routed_examined,
        "flat_examined": total_flat_examined,
        "speedup": total_flat_examined / max(total_routed_examined, 1),
        "avg_routed_examined": total_routed_examined / max(n, 1),
        "avg_flat_examined": total_flat_examined / max(n, 1),
        "routed_accuracy": routed_correct / max(n, 1),
        "flat_accuracy": flat_correct / max(n, 1),
        "routed_confusion_rate": routed_confused / max(n, 1),
        "flat_confusion_rate": flat_confused / max(n, 1),
    }


def main():
    print("Loading CIViC evidence export...")
    records = load_civic_evidence()
    print(f"  Loaded {len(records)} evidence records")

    # Build router with molecular_profile + disease as identity.
    # Everything else is a candidate discriminator.
    print("\nBuilding semantic router...")
    t0 = time.time()

    router = SemanticRouter()
    router.ingest(
        records=records,
        identity_fields=["molecular_profile", "disease"],
        provenance_fields=["evidence_id"],
        max_ladder_depth=4,
    )

    build_time = time.time() - t0
    print(f"  Built in {build_time:.2f}s")

    # Print what the router discovered.
    info = router.explain()
    print(f"\n=== Router Structure ===")
    print(f"  Total records: {info['total_records']}")
    print(f"  Identity fields: {info['identity_fields']}")
    print(f"  Identity neighborhoods: {info['identity_neighborhoods']}")
    print(f"  Ambiguous neighborhoods: {info['ambiguous_neighborhoods']}")
    print(f"  Discovered ladder:")
    for rung in info["ladder"]:
        print(f"    Rung {rung['rung']}: {rung['field']} (reduction={rung['ambiguity_reduction_rate']})")

    # Build query set: sample records from the dataset itself.
    # Each sampled record becomes a query that should retrieve itself.
    rng = random.Random(42)
    sample_size = min(500, len(records))
    query_records = rng.sample(records, sample_size)

    # Test 1: Query for therapies.
    print(f"\n=== Speed Test: retrieve 'therapies' ({sample_size} queries) ===")
    therapy_results = run_speed_test(router, query_records, ask_field="therapies")
    print(f"  Routed: avg {therapy_results['avg_routed_examined']:.1f} records examined per query")
    print(f"  Flat:   avg {therapy_results['avg_flat_examined']:.1f} records examined per query")
    print(f"  Speedup: {therapy_results['speedup']:.1f}x")
    print(f"  Routed accuracy: {therapy_results['routed_accuracy']:.2%}")
    print(f"  Flat accuracy:   {therapy_results['flat_accuracy']:.2%}")
    print(f"  Routed confusion: {therapy_results['routed_confusion_rate']:.2%}")
    print(f"  Flat confusion:   {therapy_results['flat_confusion_rate']:.2%}")

    # Test 2: Query for significance.
    print(f"\n=== Speed Test: retrieve 'significance' ({sample_size} queries) ===")
    sig_results = run_speed_test(router, query_records, ask_field="significance")
    print(f"  Routed: avg {sig_results['avg_routed_examined']:.1f} records examined per query")
    print(f"  Flat:   avg {sig_results['avg_flat_examined']:.1f} records examined per query")
    print(f"  Speedup: {sig_results['speedup']:.1f}x")
    print(f"  Routed accuracy: {sig_results['routed_accuracy']:.2%}")
    print(f"  Flat accuracy:   {sig_results['flat_accuracy']:.2%}")
    print(f"  Routed confusion: {sig_results['routed_confusion_rate']:.2%}")
    print(f"  Flat confusion:   {sig_results['flat_confusion_rate']:.2%}")

    # Test 3: Query for evidence_level.
    print(f"\n=== Speed Test: retrieve 'evidence_level' ({sample_size} queries) ===")
    level_results = run_speed_test(router, query_records, ask_field="evidence_level")
    print(f"  Routed: avg {level_results['avg_routed_examined']:.1f} records examined per query")
    print(f"  Flat:   avg {level_results['avg_flat_examined']:.1f} records examined per query")
    print(f"  Speedup: {level_results['speedup']:.1f}x")
    print(f"  Routed accuracy: {level_results['routed_accuracy']:.2%}")
    print(f"  Flat accuracy:   {level_results['flat_accuracy']:.2%}")
    print(f"  Routed confusion: {level_results['routed_confusion_rate']:.2%}")
    print(f"  Flat confusion:   {level_results['flat_confusion_rate']:.2%}")

    # Summary.
    print(f"\n=== Summary ===")
    print(f"  Dataset size: {len(records)} records")
    print(f"  Router build time: {build_time:.2f}s")
    print(f"  Ladder discovered: {' -> '.join(['identity'] + router.ladder_fields)}")
    print(f"  Average speedup across tests: {(therapy_results['speedup'] + sig_results['speedup'] + level_results['speedup']) / 3:.1f}x")


if __name__ == "__main__":
    main()
