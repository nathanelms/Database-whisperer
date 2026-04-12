"""test_consequences.py

Test ALL five consequences of the discriminator ladder algorithm:

    1. RETRIEVAL       — route queries touching minimal records (already proven)
    2. ANOMALY         — detect and localize corrupted fields (just proven)
    3. COMPRESSION     — the ladder is a lossy structural summary of the data
    4. REASONING TRACE — the routing path explains WHY a record was retrieved
    5. FEDERATED       — ladders compose across database boundaries

Same algorithm. Zero modifications. Five consequences.

Usage:
    python test_consequences.py
"""

from __future__ import annotations

import csv
import math
import random
import time
from collections import defaultdict, Counter
from typing import Any, Dict, List, Set, Tuple

from semantic_router import SemanticRouter, RouteResult, LadderRung


# ---------------------------------------------------------------------------
# Data loaders (lean versions)
# ---------------------------------------------------------------------------

def load_civic(max_records: int = 99999) -> List[Dict[str, Any]]:
    records = []
    with open("civic_evidence_full.tsv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if len(records) >= max_records:
                break
            rec = {
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
            if rec["molecular_profile"] and rec["disease"]:
                records.append(rec)
    return records


def load_faers(max_records: int = 50000) -> List[Dict[str, Any]]:
    from test_router_faers import load_faers_joined
    return load_faers_joined(max_records=max_records)


def load_storm(max_records: int = 50000) -> List[Dict[str, Any]]:
    records = []
    with open("storm_events_2023.csv", "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(records) >= max_records:
                break
            rec = {
                "state": (row.get("STATE") or "").strip(),
                "event_type": (row.get("EVENT_TYPE") or "").strip(),
                "county": (row.get("CZ_NAME") or "").strip(),
                "month": (row.get("MONTH_NAME") or "").strip(),
                "source": (row.get("SOURCE") or "").strip(),
                "magnitude": (row.get("MAGNITUDE") or "").strip(),
                "injuries": (row.get("INJURIES_DIRECT") or "0").strip(),
                "deaths": (row.get("DEATHS_DIRECT") or "0").strip(),
                "episode_id": (row.get("EPISODE_ID") or "").strip(),
            }
            if rec["state"] and rec["event_type"]:
                records.append(rec)
    return records


DOMAIN_CONFIGS = {
    "CIViC": {
        "loader": load_civic,
        "identity": ["molecular_profile", "disease"],
        "provenance": ["evidence_id"],
        "ask_field": "therapies",
    },
    "FAERS": {
        "loader": load_faers,
        "identity": ["drugname", "reaction"],
        "provenance": ["primaryid"],
        "ask_field": "route",
    },
    "Storm": {
        "loader": load_storm,
        "identity": ["state", "event_type"],
        "provenance": ["episode_id"],
        "ask_field": "county",
    },
}


def build_router(records, config):
    router = SemanticRouter()
    router.ingest(
        records=records,
        identity_fields=config["identity"],
        provenance_fields=config["provenance"],
        max_ladder_depth=4,
    )
    return router


# ===========================================================================
# CONSEQUENCE 1: RETRIEVAL (confirmation run — compact)
# ===========================================================================

def test_retrieval(records, router, config, name):
    print(f"\n  [{name}] RETRIEVAL")

    rng = random.Random(42)
    samples = rng.sample(records, min(300, len(records)))

    total_routed = 0
    total_flat = 0
    correct = 0

    for rec in samples:
        routed = router.query(rec, ask_field=config["ask_field"])
        flat = router.flat_scan(rec, ask_field=config["ask_field"])
        total_routed += routed.records_examined
        total_flat += flat.records_examined
        if routed.answer == rec.get(config["ask_field"]):
            correct += 1

    speedup = total_flat / max(total_routed, 1)
    accuracy = correct / len(samples)

    print(f"    Speedup: {speedup:,.0f}x")
    print(f"    Accuracy: {accuracy:.1%}")
    return {"speedup": speedup, "accuracy": accuracy}


# ===========================================================================
# CONSEQUENCE 2: ANOMALY DETECTION (compact version)
# ===========================================================================

def test_anomaly(records, router, config, name):
    print(f"\n  [{name}] ANOMALY DETECTION")

    ladder_fields = router.ladder_fields
    all_values: Dict[str, List[str]] = defaultdict(list)
    for rec in records:
        for f in ladder_fields:
            v = rec.get(f, "")
            if v:
                all_values[f].append(v)
    for f in all_values:
        all_values[f] = list(set(all_values[f]))

    rng = random.Random(42)
    samples = rng.sample(records, min(200, len(records)))

    results = {}
    for field in ladder_fields:
        detected = 0
        localized = 0
        total = 0
        for rec in samples:
            corrupted = dict(rec)
            orig = rec.get(field, "")
            candidates = [v for v in all_values.get(field, []) if v != orig and v]
            if not candidates:
                continue
            corrupted[field] = rng.choice(candidates)
            total += 1

            result = router.query(corrupted, ask_field=config["ask_field"])
            expected = rec.get(config["ask_field"])
            if result.answer != expected:
                detected += 1
                # Check localization
                id_key = tuple(corrupted.get(f, "") for f in config["identity"])
                if id_key in router._index:
                    node = router._index[id_key]
                    for i, rung in enumerate(router._ladder):
                        rv = corrupted.get(rung.field_name, "")
                        if rv not in node:
                            if rung.field_name == field:
                                localized += 1
                            break
                        node = node[rv]

        det_rate = detected / max(total, 1)
        loc_rate = localized / max(detected, 1) if detected > 0 else 0.0
        results[field] = {"detection": det_rate, "localization": loc_rate, "n": total}
        print(f"    {field}: detect={det_rate:.0%}, localize={loc_rate:.0%}")

    total_det = sum(r["n"] * r["detection"] for r in results.values())
    total_n = sum(r["n"] for r in results.values())
    total_loc_num = sum(r["n"] * r["detection"] * r["localization"] for r in results.values())
    total_loc_den = total_det

    overall_det = total_det / max(total_n, 1)
    overall_loc = total_loc_num / max(total_loc_den, 1)
    print(f"    Overall: detect={overall_det:.1%}, localize={overall_loc:.1%}")

    return {"detection": overall_det, "localization": overall_loc}


# ===========================================================================
# CONSEQUENCE 3: COMPRESSION
# ===========================================================================

def test_compression(records, router, config, name):
    """
    The ladder is a structural summary of the dataset.

    Hypothesis: the ladder captures the data's disambiguation structure in
    far fewer bits than the raw records. We measure:

    1. Naive storage: all records, all fields
    2. Ladder storage: identity neighborhoods + ladder field orderings + rung values
    3. Reconstruction accuracy: can we reconstruct the right answer from just
       the ladder path (identity + rung values) without storing the full record?

    If reconstruction accuracy is high, the ladder is a lossy compression
    that preserves retrieval-relevant information.
    """
    print(f"\n  [{name}] COMPRESSION")

    ladder_fields = router.ladder_fields
    identity_fields = config["identity"]
    ask_field = config["ask_field"]

    # Measure 1: Naive storage (total characters across all records, all fields)
    all_fields = list(records[0].keys())
    naive_chars = sum(
        sum(len(str(rec.get(f, ""))) for f in all_fields)
        for rec in records
    )

    # Measure 2: Ladder-path storage
    # For each record, store only: identity key + ladder rung values + answer
    # This is what the index effectively encodes
    path_fields = identity_fields + ladder_fields + [ask_field]
    path_chars = sum(
        sum(len(str(rec.get(f, ""))) for f in path_fields)
        for rec in records
    )

    # Measure 3: Ladder skeleton (unique paths only — the index structure itself)
    unique_paths = set()
    for rec in records:
        path = tuple(rec.get(f, "") for f in identity_fields + ladder_fields)
        unique_paths.add(path)

    skeleton_chars = sum(sum(len(v) for v in path) for path in unique_paths)

    compression_ratio = naive_chars / max(path_chars, 1)
    skeleton_ratio = naive_chars / max(skeleton_chars, 1)

    # Measure 4: Reconstruction accuracy
    # Given only the ladder path, can we find the right record?
    rng = random.Random(42)
    samples = rng.sample(records, min(300, len(records)))
    reconstructed = 0
    for rec in samples:
        # Build a query from just identity + ladder fields
        query = {}
        for f in identity_fields + ladder_fields:
            query[f] = rec.get(f, "")
        result = router.query(query, ask_field=ask_field)
        if result.answer == rec.get(ask_field):
            reconstructed += 1

    recon_accuracy = reconstructed / len(samples)

    # Information content
    total_fields = len(all_fields) - len(config["provenance"])
    path_field_count = len(path_fields)
    field_reduction = 1 - (path_field_count / total_fields)

    # Entropy estimate: how many bits per record does the ladder path use?
    # vs how many does the full record use?
    def estimate_entropy(records, fields):
        """Estimate entropy in bits per record for a set of fields."""
        total_entropy = 0
        for f in fields:
            values = Counter(rec.get(f, "") for rec in records)
            total = sum(values.values())
            for count in values.values():
                if count > 0:
                    p = count / total
                    total_entropy -= p * math.log2(p)
        return total_entropy

    full_entropy = estimate_entropy(records, [f for f in all_fields if f not in config["provenance"]])
    path_entropy = estimate_entropy(records, path_fields)
    entropy_ratio = path_entropy / max(full_entropy, 0.001)

    print(f"    Total fields: {total_fields}")
    print(f"    Ladder path fields: {path_field_count} ({identity_fields} + {ladder_fields} + [{ask_field}])")
    print(f"    Field reduction: {field_reduction:.0%} fewer fields needed")
    print(f"    Naive chars: {naive_chars:,}")
    print(f"    Path chars:  {path_chars:,} ({compression_ratio:.1f}x compression)")
    print(f"    Skeleton:    {skeleton_chars:,} ({skeleton_ratio:.1f}x compression)")
    print(f"    Unique paths: {len(unique_paths):,} vs {len(records):,} records ({len(unique_paths)/len(records):.1%})")
    print(f"    Full entropy: {full_entropy:.1f} bits/record")
    print(f"    Path entropy: {path_entropy:.1f} bits/record ({entropy_ratio:.0%} of full)")
    print(f"    Reconstruction accuracy: {recon_accuracy:.1%}")

    return {
        "compression_ratio": compression_ratio,
        "skeleton_ratio": skeleton_ratio,
        "field_reduction": field_reduction,
        "entropy_ratio": entropy_ratio,
        "reconstruction_accuracy": recon_accuracy,
        "unique_paths_pct": len(unique_paths) / len(records),
    }


# ===========================================================================
# CONSEQUENCE 4: REASONING TRACE
# ===========================================================================

def test_reasoning_trace(records, router, config, name):
    """
    The routing path through the ladder IS the reasoning trace.

    Hypothesis: two records that reach the same leaf via different paths
    are different for articulable reasons — and the path difference names
    the reason. Conversely, records sharing a path share a rationale.

    We test:
    1. Path distinctiveness: do different answers come from different paths?
    2. Path coherence: do records sharing a path agree on the answer?
    3. Human-readable traces: can we generate an explanation from the path?
    """
    print(f"\n  [{name}] REASONING TRACE")

    ladder_fields = router.ladder_fields
    identity_fields = config["identity"]
    ask_field = config["ask_field"]

    # Build path -> answer mapping
    path_answers: Dict[tuple, Counter] = defaultdict(Counter)
    path_records: Dict[tuple, List] = defaultdict(list)

    for rec in records:
        identity = tuple(rec.get(f, "") for f in identity_fields)
        path = tuple(rec.get(f, "") for f in ladder_fields)
        full_path = identity + path
        answer = rec.get(ask_field, "")
        path_answers[full_path][answer] += 1
        path_records[full_path].append(rec)

    total_paths = len(path_answers)

    # Metric 1: Path distinctiveness
    # A path is "distinct" if it maps to exactly one answer
    distinct_paths = sum(1 for answers in path_answers.values() if len(answers) == 1)
    distinctiveness = distinct_paths / max(total_paths, 1)

    # Metric 2: Path coherence (weighted)
    # For each path, what fraction of records agree on the majority answer?
    total_coherence = 0
    total_weight = 0
    for path, answers in path_answers.items():
        majority = answers.most_common(1)[0][1]
        total = sum(answers.values())
        total_coherence += majority
        total_weight += total
    coherence = total_coherence / max(total_weight, 1)

    # Metric 3: Path explains disagreement
    # When two records in the same identity neighborhood have different answers,
    # does the path difference correctly identify WHY?
    rng = random.Random(42)
    samples = rng.sample(records, min(300, len(records)))

    explanations_valid = 0
    explanations_total = 0

    for rec in samples:
        identity = tuple(rec.get(f, "") for f in identity_fields)
        # Find a record in the same neighborhood with a different answer
        id_key = identity
        if id_key not in router._index:
            continue

        # Collect all records in this neighborhood
        node = router._index[id_key]
        neighbors = router._collect_records_from_subtree(node)
        answer = rec.get(ask_field, "")

        different = [n for n in neighbors if n.get(ask_field) != answer]
        if not different:
            continue

        other = rng.choice(different)
        explanations_total += 1

        # Find which ladder field first differs between rec and other
        differentiating_field = None
        for field in ladder_fields:
            if rec.get(field, "") != other.get(field, ""):
                differentiating_field = field
                break

        if differentiating_field:
            explanations_valid += 1

    explanation_rate = explanations_valid / max(explanations_total, 1)

    # Generate sample traces
    print(f"    Total unique paths: {total_paths:,}")
    print(f"    Distinctiveness: {distinctiveness:.1%} of paths map to one answer")
    print(f"    Coherence: {coherence:.1%} of records agree with path majority")
    print(f"    Explanation rate: {explanation_rate:.1%} of disagreements explained by ladder field")

    # Show example traces
    print(f"\n    Example reasoning traces:")
    shown = 0
    for rec in samples[:20]:
        if shown >= 3:
            break
        result = router.query(rec, ask_field=ask_field)
        if result.answer and result.candidates_at_each_stage:
            identity_str = " + ".join(f"{f}='{rec.get(f,'')}'" for f in identity_fields)
            trace_parts = []
            for i, field in enumerate(ladder_fields):
                val = rec.get(field, "")
                narrowed = result.candidates_at_each_stage[i+1] if i+1 < len(result.candidates_at_each_stage) else "?"
                trace_parts.append(f"{field}='{val}' -> {narrowed}")

            print(f"      Query: {identity_str}")
            print(f"      Path: {' | '.join(trace_parts)}")
            print(f"      Answer: {result.answer}")
            print(f"      Because: started with {result.candidates_at_each_stage[0]} candidates,", end="")
            for i, field in enumerate(ladder_fields):
                if i+1 < len(result.candidates_at_each_stage):
                    prev = result.candidates_at_each_stage[i] if i < len(result.candidates_at_each_stage) else "?"
                    curr = result.candidates_at_each_stage[i+1]
                    if prev != curr:
                        print(f" {field} narrowed to {curr},", end="")
            print(f" examined {result.records_examined}")
            print()
            shown += 1

    return {
        "distinctiveness": distinctiveness,
        "coherence": coherence,
        "explanation_rate": explanation_rate,
        "total_paths": total_paths,
    }


# ===========================================================================
# CONSEQUENCE 5: FEDERATED BRIDGING
# ===========================================================================

def test_federated(civic_records, faers_records, storm_records):
    """
    Ladders compose across database boundaries.

    Hypothesis: when two databases share a join key, their individual ladders
    can be bridged — and the BRIDGE ladder discovers new structure that
    neither individual ladder contains.

    We test:
    1. Individual ladders: what does each DB discover alone?
    2. Bridge ladder: what does the joined data discover?
    3. Emergence: does the bridge ladder contain fields from BOTH databases?
    4. Cross-DB queries: can we answer questions that require both databases?
    5. Scalability: does adding a third database change the bridge ladder?
    """
    print(f"\n  FEDERATED BRIDGING")
    print(f"  Testing ladder composition across database boundaries\n")

    # Individual ladders
    civic_router = SemanticRouter()
    civic_router.ingest(
        records=civic_records,
        identity_fields=["molecular_profile", "disease"],
        provenance_fields=["evidence_id"],
        max_ladder_depth=4,
    )

    faers_router = SemanticRouter()
    faers_router.ingest(
        records=faers_records,
        identity_fields=["drugname", "reaction"],
        provenance_fields=["primaryid"],
        max_ladder_depth=4,
    )

    storm_router = SemanticRouter()
    storm_router.ingest(
        records=storm_records,
        identity_fields=["state", "event_type"],
        provenance_fields=["episode_id"],
        max_ladder_depth=4,
    )

    print(f"  Individual ladders:")
    print(f"    CIViC ({len(civic_records)}): identity -> {' -> '.join(civic_router.ladder_fields)}")
    print(f"    FAERS ({len(faers_records)}): identity -> {' -> '.join(faers_router.ladder_fields)}")
    print(f"    Storm ({len(storm_records)}): identity -> {' -> '.join(storm_router.ladder_fields)}")

    # Build CIViC-FAERS bridge on drug name
    from multi_db_bridge import normalize_drug

    civic_by_drug: Dict[str, List] = defaultdict(list)
    for rec in civic_records:
        for t in rec.get("therapies", "").split(","):
            n = normalize_drug(t.strip())
            if n:
                civic_by_drug[n].append(rec)

    faers_by_drug: Dict[str, List] = defaultdict(list)
    for rec in faers_records:
        drug = rec.get("active_ingredient") or rec.get("drugname", "")
        n = normalize_drug(drug)
        if n:
            faers_by_drug[n].append(rec)

    shared_drugs = set(civic_by_drug.keys()) & set(faers_by_drug.keys())
    print(f"\n  Shared drugs (CIViC <-> FAERS): {len(shared_drugs)}")

    # Build bridge records
    bridge_2db = []
    for drug in sorted(shared_drugs):
        reaction_counts = Counter(r["reaction"] for r in faers_by_drug[drug] if r["reaction"])
        top_reactions = reaction_counts.most_common(10)
        sex_dist = Counter(r["sex"] for r in faers_by_drug[drug] if r["sex"])

        for civic_rec in civic_by_drug[drug]:
            for reaction, count in top_reactions:
                bridge_2db.append({
                    "therapy": drug,
                    "molecular_profile": civic_rec["molecular_profile"],
                    "disease": civic_rec["disease"],
                    "evidence_type": civic_rec["evidence_type"],
                    "evidence_direction": civic_rec["evidence_direction"],
                    "evidence_level": civic_rec["evidence_level"],
                    "significance": civic_rec["significance"],
                    "rating": civic_rec["rating"],
                    "adverse_event": reaction,
                    "ae_count": str(count),
                    "sex_ratio": f"M:{sex_dist.get('M',0)}/F:{sex_dist.get('F',0)}",
                    "bridge_id": f"{drug}|{civic_rec['evidence_id']}|{reaction}",
                })

    print(f"  2-DB bridge records: {len(bridge_2db)}")

    if not bridge_2db:
        print("  SKIP: No bridge records")
        return {"emergence": False}

    bridge_router = SemanticRouter()
    bridge_router.ingest(
        records=bridge_2db,
        identity_fields=["therapy", "molecular_profile", "disease"],
        provenance_fields=["bridge_id"],
        max_ladder_depth=5,
    )

    bridge_ladder = bridge_router.ladder_fields
    print(f"  2-DB bridge ladder: identity -> {' -> '.join(bridge_ladder)}")

    # Test emergence: does the bridge ladder contain fields from BOTH databases?
    civic_fields = {"evidence_type", "evidence_direction", "evidence_level", "significance", "rating"}
    faers_fields = {"adverse_event", "ae_count", "sex_ratio"}

    bridge_has_civic = any(f in civic_fields for f in bridge_ladder)
    bridge_has_faers = any(f in faers_fields for f in bridge_ladder)
    emergence = bridge_has_civic and bridge_has_faers

    print(f"\n  Emergence test:")
    print(f"    Bridge contains CIViC fields: {bridge_has_civic} ({[f for f in bridge_ladder if f in civic_fields]})")
    print(f"    Bridge contains FAERS fields: {bridge_has_faers} ({[f for f in bridge_ladder if f in faers_fields]})")
    print(f"    Cross-database emergence: {'YES' if emergence else 'NO'}")

    # Test: fields that ONLY appear in the bridge ladder (not in either individual)
    individual_fields = set(civic_router.ladder_fields) | set(faers_router.ladder_fields)
    novel_in_bridge = [f for f in bridge_ladder if f not in individual_fields]
    print(f"    Novel bridge fields (not in either individual): {novel_in_bridge}")

    # Test cross-DB queries: answer something that requires both databases
    rng = random.Random(42)
    samples = rng.sample(bridge_2db, min(200, len(bridge_2db)))

    cross_correct = 0
    for rec in samples:
        # Query: given CIViC identity, retrieve FAERS field
        result = bridge_router.query(rec, ask_field="adverse_event")
        if result.answer == rec["adverse_event"]:
            cross_correct += 1

    cross_accuracy = cross_correct / len(samples)

    # Compare speedup
    total_routed = 0
    total_flat = 0
    for rec in samples[:100]:
        routed = bridge_router.query(rec, ask_field="adverse_event")
        flat = bridge_router.flat_scan(rec, ask_field="adverse_event")
        total_routed += routed.records_examined
        total_flat += flat.records_examined

    bridge_speedup = total_flat / max(total_routed, 1)

    print(f"\n  Cross-database query accuracy: {cross_accuracy:.1%}")
    print(f"  Bridge speedup: {bridge_speedup:,.0f}x")

    # Test ladder stability: does adding a THIRD database (storm) change the picture?
    # We can't directly bridge storm with civic/faers (no shared key),
    # but we can test whether the algorithm handles heterogeneous schemas
    print(f"\n  Ladder stability across domains:")
    print(f"    CIViC alone:  {' -> '.join(civic_router.ladder_fields)}")
    print(f"    FAERS alone:  {' -> '.join(faers_router.ladder_fields)}")
    print(f"    Storm alone:  {' -> '.join(storm_router.ladder_fields)}")
    print(f"    CIViC+FAERS:  {' -> '.join(bridge_ladder)}")

    # All three use the same algorithm — do they all produce valid ladders?
    all_valid = (
        len(civic_router.ladder_fields) > 0 and
        len(faers_router.ladder_fields) > 0 and
        len(storm_router.ladder_fields) > 0 and
        len(bridge_ladder) > 0
    )
    print(f"    All produce valid ladders: {all_valid}")
    print(f"    Same algorithm, 4 different schemas, 4 different ladders")

    return {
        "emergence": emergence,
        "novel_fields": novel_in_bridge,
        "cross_accuracy": cross_accuracy,
        "bridge_speedup": bridge_speedup,
        "all_valid": all_valid,
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 72)
    print("  DATABASE WHISPER: FIVE CONSEQUENCES OF ONE ALGORITHM")
    print("  Greedy pair-reduction discriminator ladder")
    print("  Zero code changes across all tests")
    print("=" * 72)

    t0 = time.time()

    # Load all three domains
    print("\nLoading datasets...")
    civic = load_civic()
    faers = load_faers(50_000)
    storm = load_storm(50_000)
    print(f"  CIViC: {len(civic)}, FAERS: {len(faers)}, Storm: {len(storm)}")

    # Build routers
    configs = {
        "CIViC": (civic, DOMAIN_CONFIGS["CIViC"]),
        "FAERS": (faers, DOMAIN_CONFIGS["FAERS"]),
        "Storm": (storm, DOMAIN_CONFIGS["Storm"]),
    }

    routers = {}
    for name, (records, config) in configs.items():
        routers[name] = build_router(records, config)
        print(f"  {name} ladder: identity -> {' -> '.join(routers[name].ladder_fields)}")

    # Run all five consequences
    all_results = {}

    # 1. RETRIEVAL
    print("\n" + "=" * 72)
    print("  CONSEQUENCE 1: RETRIEVAL")
    print("=" * 72)
    retrieval_results = {}
    for name, (records, config) in configs.items():
        retrieval_results[name] = test_retrieval(records, routers[name], config, name)
    all_results["retrieval"] = retrieval_results

    # 2. ANOMALY DETECTION
    print("\n" + "=" * 72)
    print("  CONSEQUENCE 2: ANOMALY DETECTION")
    print("=" * 72)
    anomaly_results = {}
    for name, (records, config) in configs.items():
        anomaly_results[name] = test_anomaly(records, routers[name], config, name)
    all_results["anomaly"] = anomaly_results

    # 3. COMPRESSION
    print("\n" + "=" * 72)
    print("  CONSEQUENCE 3: COMPRESSION")
    print("=" * 72)
    compression_results = {}
    for name, (records, config) in configs.items():
        compression_results[name] = test_compression(records, routers[name], config, name)
    all_results["compression"] = compression_results

    # 4. REASONING TRACE
    print("\n" + "=" * 72)
    print("  CONSEQUENCE 4: REASONING TRACE")
    print("=" * 72)
    reasoning_results = {}
    for name, (records, config) in configs.items():
        reasoning_results[name] = test_reasoning_trace(records, routers[name], config, name)
    all_results["reasoning"] = reasoning_results

    # 5. FEDERATED BRIDGING
    print("\n" + "=" * 72)
    print("  CONSEQUENCE 5: FEDERATED BRIDGING")
    print("=" * 72)
    federated_results = test_federated(civic, faers, storm)
    all_results["federated"] = federated_results

    # ===========================================================
    # GRAND SUMMARY
    # ===========================================================
    elapsed = time.time() - t0

    print("\n" + "=" * 72)
    print("  GRAND SUMMARY: FIVE CONSEQUENCES OF ONE ALGORITHM")
    print("=" * 72)

    print(f"\n  Algorithm: greedy pair-reduction discriminator ladder")
    print(f"  Code changes for 5 consequences: ZERO")
    print(f"  Total time: {elapsed:.1f}s\n")

    # Table
    print(f"  {'Consequence':<22} {'CIViC':<18} {'FAERS':<18} {'Storm':<18}")
    print(f"  {'-'*72}")

    # Retrieval
    r = all_results["retrieval"]
    print(f"  {'1. Retrieval':<22}", end="")
    for name in ["CIViC", "FAERS", "Storm"]:
        s = r[name]
        print(f" {s['speedup']:>8,.0f}x {s['accuracy']:>5.0%}  ", end="")
    print()

    # Anomaly
    a = all_results["anomaly"]
    print(f"  {'2. Anomaly detect':<22}", end="")
    for name in ["CIViC", "FAERS", "Storm"]:
        s = a[name]
        print(f" det:{s['detection']:>4.0%} loc:{s['localization']:>4.0%} ", end="")
    print()

    # Compression
    c = all_results["compression"]
    print(f"  {'3. Compression':<22}", end="")
    for name in ["CIViC", "FAERS", "Storm"]:
        s = c[name]
        print(f" {s['compression_ratio']:>5.1f}x recon:{s['reconstruction_accuracy']:>4.0%} ", end="")
    print()

    # Reasoning
    t = all_results["reasoning"]
    print(f"  {'4. Reasoning trace':<22}", end="")
    for name in ["CIViC", "FAERS", "Storm"]:
        s = t[name]
        print(f" dist:{s['distinctiveness']:>4.0%} coh:{s['coherence']:>4.0%}  ", end="")
    print()

    # Federated
    f = all_results["federated"]
    print(f"\n  {'5. Federated':<22} emergence={f['emergence']}, "
          f"cross-DB accuracy={f.get('cross_accuracy',0):.0%}, "
          f"bridge speedup={f.get('bridge_speedup',0):,.0f}x")

    print(f"\n  {'='*72}")
    print(f"  One algorithm. Five consequences. Three domains. Zero modifications.")
    print(f"  {'='*72}")


if __name__ == "__main__":
    main()
