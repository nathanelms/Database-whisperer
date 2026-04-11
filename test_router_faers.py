"""test_router_faers.py

Test the semantic router on FDA FAERS adverse event data (~1.9M drug records).
The messiest real-world database we can find.

Usage:
    python test_router_faers.py
"""

from __future__ import annotations

import csv
import random
import time
from collections import defaultdict
from typing import Any, Dict, List

from semantic_router import SemanticRouter


def load_faers_joined(
    drug_path: str = "ASCII/DRUG24Q3.txt",
    reac_path: str = "ASCII/REAC24Q3.txt",
    demo_path: str = "ASCII/DEMO24Q3.txt",
    max_records: int = 200_000,
) -> List[Dict[str, Any]]:
    """
    Load FAERS drug + reaction + demographics into joined dicts.

    What this does:
    - Reads the three FAERS files, joins on primaryid, and produces one record
      per drug-reaction pair.

    Why this exists:
    - The router needs flat dicts. FAERS ships as separate relational files.

    What assumption it is making:
    - Drug-reaction pairs are the natural unit of retrieval for adverse events.
    - We cap at max_records to keep the demo runnable in seconds, not minutes.
    """
    # Load demographics keyed by primaryid.
    demo_by_id: Dict[str, Dict[str, str]] = {}
    with open(demo_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="$")
        for row in reader:
            pid = (row.get("primaryid") or "").strip()
            if pid:
                demo_by_id[pid] = {
                    "age": (row.get("age") or "").strip(),
                    "sex": (row.get("sex") or "").strip(),
                    "reporter_country": (row.get("reporter_country") or "").strip(),
                    "report_type": (row.get("rept_cod") or "").strip(),
                }

    # Load reactions keyed by primaryid (first reaction per case for simplicity).
    reac_by_id: Dict[str, str] = {}
    with open(reac_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="$")
        for row in reader:
            pid = (row.get("primaryid") or "").strip()
            pt = (row.get("pt") or "").strip()
            if pid and pt and pid not in reac_by_id:
                reac_by_id[pid] = pt

    # Load drugs and join.
    records = []
    with open(drug_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="$")
        for row in reader:
            if len(records) >= max_records:
                break

            pid = (row.get("primaryid") or "").strip()
            drugname = (row.get("drugname") or "").strip().upper()
            role = (row.get("role_cod") or "").strip()
            route = (row.get("route") or "").strip()
            dose_form = (row.get("dose_form") or "").strip()
            prod_ai = (row.get("prod_ai") or "").strip().upper()

            if not drugname or not pid:
                continue

            demo = demo_by_id.get(pid, {})
            reaction = reac_by_id.get(pid, "")

            record = {
                "drugname": drugname,
                "active_ingredient": prod_ai,
                "role": role,
                "route": route,
                "dose_form": dose_form,
                "reaction": reaction,
                "sex": demo.get("sex", ""),
                "reporter_country": demo.get("reporter_country", ""),
                "report_type": demo.get("report_type", ""),
                "primaryid": pid,
            }
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
    print("Loading FDA FAERS 2024-Q3 (drug + reaction + demographics)...")
    t_load = time.time()
    records = load_faers_joined(max_records=200_000)
    load_time = time.time() - t_load
    print(f"  Loaded {len(records)} drug-reaction records in {load_time:.1f}s")

    # Show the mess.
    unique_drugs = set(r["drugname"] for r in records)
    unique_ingredients = set(r["active_ingredient"] for r in records if r["active_ingredient"])
    unique_reactions = set(r["reaction"] for r in records if r["reaction"])
    print(f"  Unique drug names: {len(unique_drugs)}")
    print(f"  Unique active ingredients: {len(unique_ingredients)}")
    print(f"  Unique reactions: {len(unique_reactions)}")

    # Build router: drugname + reaction as identity.
    # This is intentionally using the messy raw drug name, not the cleaned active ingredient.
    print("\nBuilding semantic router on MESSY drug names...")
    t0 = time.time()

    router = SemanticRouter()
    router.ingest(
        records=records,
        identity_fields=["drugname", "reaction"],
        provenance_fields=["primaryid"],
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

    # Sample queries.
    rng = random.Random(42)
    sample_size = min(1000, len(records))
    query_records = rng.sample(records, sample_size)

    # Test 1: retrieve route (administration route).
    print(f"\n=== Speed Test: retrieve 'route' ({sample_size} queries) ===")
    r1 = run_speed_test(router, query_records, ask_field="route")
    print(f"  Routed: avg {r1['avg_routed_examined']:.1f} records examined")
    print(f"  Flat:   avg {r1['avg_flat_examined']:.1f} records examined")
    print(f"  Speedup: {r1['speedup']:.1f}x")
    print(f"  Routed accuracy: {r1['routed_accuracy']:.2%}")
    print(f"  Routed confusion: {r1['routed_confusion_rate']:.2%}")
    print(f"  Flat confusion:   {r1['flat_confusion_rate']:.2%}")

    # Test 2: retrieve sex.
    print(f"\n=== Speed Test: retrieve 'sex' ({sample_size} queries) ===")
    r2 = run_speed_test(router, query_records, ask_field="sex")
    print(f"  Routed: avg {r2['avg_routed_examined']:.1f} records examined")
    print(f"  Flat:   avg {r2['avg_flat_examined']:.1f} records examined")
    print(f"  Speedup: {r2['speedup']:.1f}x")
    print(f"  Routed accuracy: {r2['routed_accuracy']:.2%}")
    print(f"  Routed confusion: {r2['routed_confusion_rate']:.2%}")
    print(f"  Flat confusion:   {r2['flat_confusion_rate']:.2%}")

    # Test 3: retrieve reporter_country.
    print(f"\n=== Speed Test: retrieve 'reporter_country' ({sample_size} queries) ===")
    r3 = run_speed_test(router, query_records, ask_field="reporter_country")
    print(f"  Routed: avg {r3['avg_routed_examined']:.1f} records examined")
    print(f"  Flat:   avg {r3['avg_flat_examined']:.1f} records examined")
    print(f"  Speedup: {r3['speedup']:.1f}x")
    print(f"  Routed accuracy: {r3['routed_accuracy']:.2%}")
    print(f"  Routed confusion: {r3['routed_confusion_rate']:.2%}")
    print(f"  Flat confusion:   {r3['flat_confusion_rate']:.2%}")

    avg_speedup = (r1["speedup"] + r2["speedup"] + r3["speedup"]) / 3
    avg_acc = (r1["routed_accuracy"] + r2["routed_accuracy"] + r3["routed_accuracy"]) / 3
    avg_conf = (r1["routed_confusion_rate"] + r2["routed_confusion_rate"] + r3["routed_confusion_rate"]) / 3

    print(f"\n=== Summary ===")
    print(f"  Dataset: FDA FAERS 2024-Q3 (adverse event reports)")
    print(f"  Records: {len(records)}")
    print(f"  Unique drug names (messy): {len(unique_drugs)}")
    print(f"  Router build time: {build_time:.2f}s")
    print(f"  Ladder discovered: {' -> '.join(['identity'] + router.ladder_fields)}")
    print(f"  Average speedup: {avg_speedup:.1f}x")
    print(f"  Average routed accuracy: {avg_acc:.2%}")
    print(f"  Average routed confusion: {avg_conf:.2%}")

    print(f"\n=== Cross-Domain Ladder Comparison ===")
    print(f"  CIViC (oncology, 4.6k):     identity -> rating -> therapies -> significance -> evidence_level")
    print(f"  Storm Events (weather, 75k): identity -> county -> month -> source -> magnitude")
    print(f"  FAERS (adverse events, {len(records)//1000}k):  identity -> {' -> '.join(router.ladder_fields)}")
    print(f"  Three domains, three different ladders, same procedure.")


if __name__ == "__main__":
    main()
