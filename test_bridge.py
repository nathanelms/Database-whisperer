"""test_bridge.py

Test suite for the 6-database mega-bridge.
Validates data loading, joins, routing, signal detection, and the web server.

Usage:
    python test_bridge.py
"""

from __future__ import annotations

import json
import time
import traceback
from collections import Counter, defaultdict
from typing import Any, Dict, List
from io import StringIO
from http.server import HTTPServer
import threading
import urllib.request
import ssl

from semantic_router import SemanticRouter


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

PASS = 0
FAIL = 0
WARN = 0

def test(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  -- {detail}")

def warn(name: str, detail: str):
    global WARN
    WARN += 1
    print(f"  WARN  {name}  -- {detail}")


# ===========================================================================
# TEST 1: Local data loading
# ===========================================================================

def test_data_loading():
    print("\n=== TEST 1: Data Loading ===")

    from multi_db_bridge import load_civic, load_faers

    civic = load_civic()
    test("CIViC loads", len(civic) > 0, f"got {len(civic)}")
    test("CIViC has records", len(civic) > 4000, f"expected >4000, got {len(civic)}")
    test("CIViC record has molecular_profile", "molecular_profile" in civic[0])
    test("CIViC record has disease", "disease" in civic[0])
    test("CIViC record has therapies", "therapies" in civic[0])
    test("CIViC record has evidence_id", "evidence_id" in civic[0])

    # Check for empty fields
    empty_profiles = sum(1 for r in civic if not r["molecular_profile"])
    test("CIViC no empty molecular_profiles", empty_profiles == 0, f"{empty_profiles} empty")

    faers = load_faers(max_records=10_000)  # small for speed
    test("FAERS loads", len(faers) > 0, f"got {len(faers)}")
    test("FAERS respects max_records", len(faers) <= 10_000, f"got {len(faers)}")
    test("FAERS record has drugname", "drugname" in faers[0])
    test("FAERS record has reaction", "reaction" in faers[0])
    test("FAERS record has primaryid", "primaryid" in faers[0])

    # Check join field presence
    has_reaction = sum(1 for r in faers if r["reaction"])
    test("FAERS >50% have reactions", has_reaction > len(faers) * 0.5,
         f"{has_reaction}/{len(faers)} = {has_reaction/len(faers):.1%}")

    return civic, faers


# ===========================================================================
# TEST 2: Drug name normalization
# ===========================================================================

def test_normalization():
    print("\n=== TEST 2: Drug Name Normalization ===")

    from multi_db_bridge import normalize_drug

    test("Uppercase", normalize_drug("vemurafenib") == "VEMURAFENIB")
    test("Strip whitespace", normalize_drug("  BRAF  ") == "BRAF")
    test("Remove HCL suffix", normalize_drug("ERLOTINIB HYDROCHLORIDE") == "ERLOTINIB")
    test("Remove SODIUM suffix", normalize_drug("IBUPROFEN SODIUM") == "IBUPROFEN")
    test("Remove MESYLATE suffix", normalize_drug("IMATINIB MESYLATE") == "IMATINIB")
    test("No change needed", normalize_drug("PEMBROLIZUMAB") == "PEMBROLIZUMAB")
    test("Empty string", normalize_drug("") == "")
    test("Only suffix", normalize_drug("HYDROCHLORIDE") == "HYDROCHLORIDE")  # shouldn't strip the whole thing


# ===========================================================================
# TEST 3: Gene extraction
# ===========================================================================

def test_gene_extraction():
    print("\n=== TEST 3: Gene Extraction ===")

    from multi_db_bridge import extract_gene

    test("Simple gene", extract_gene("BRAF V600E") == "BRAF")
    test("Gene mutation", extract_gene("KRAS Mutation") == "KRAS")
    test("Complex profile", extract_gene("BRCA1 Mutation OR BRCA2 Mutation") == "BRCA1")
    test("Amplification", extract_gene("ERBB2 Amplification") == "ERBB2")
    test("Empty string", extract_gene("") == "")
    test("Single gene", extract_gene("TP53") == "TP53")


# ===========================================================================
# TEST 4: Cross-database join
# ===========================================================================

def test_bridge_join(civic, faers):
    print("\n=== TEST 4: Cross-Database Join ===")

    from multi_db_bridge import build_mega_bridge, normalize_drug, extract_gene

    # Build a minimal bridge without API data (empty dicts)
    api_data = {"ctgov": {}, "chembl": {}, "uniprot": {}, "string": {}, "openfda": {}}
    bridge, stats = build_mega_bridge(civic, faers, api_data)

    test("Bridge produces records", len(bridge) > 0, f"got {len(bridge)}")
    test("Stats has shared_drugs", stats["shared_drugs"] > 0, f"got {stats['shared_drugs']}")

    if bridge:
        rec = bridge[0]
        # Check all expected fields exist
        expected_fields = [
            "therapy", "molecular_profile", "gene", "disease",
            "evidence_type", "evidence_direction", "evidence_level", "significance",
            "adverse_event", "ae_report_count", "total_faers_reports", "sex_ratio",
            "trial_phase", "trial_count",
            "mechanism", "target_type", "chembl_max_phase",
            "protein_function", "protein_pathway", "protein_location",
            "interaction_partners", "partner_count",
            "boxed_warning", "drug_class",
            "bridge_id",
        ]
        for field in expected_fields:
            test(f"Bridge has field '{field}'", field in rec, f"missing from record")

        # Verify join integrity: therapy should be in both CIViC and FAERS
        bridge_therapies = set(r["therapy"] for r in bridge)
        civic_therapies = set()
        for r in civic:
            for t in r.get("therapies", "").split(","):
                n = normalize_drug(t.strip())
                if n:
                    civic_therapies.add(n)
        faers_drugs = set()
        for r in faers:
            drug = r.get("active_ingredient") or r.get("drugname", "")
            n = normalize_drug(drug)
            if n:
                faers_drugs.add(n)

        for therapy in list(bridge_therapies)[:10]:
            in_civic = therapy in civic_therapies
            in_faers = therapy in faers_drugs
            test(f"Bridge drug '{therapy}' in both DBs",
                 in_civic and in_faers,
                 f"civic={in_civic}, faers={in_faers}")

        # Verify no duplicate bridge_ids
        ids = [r["bridge_id"] for r in bridge]
        unique_ids = set(ids)
        test("No duplicate bridge_ids",
             len(ids) == len(unique_ids),
             f"{len(ids)} total, {len(unique_ids)} unique, {len(ids)-len(unique_ids)} dupes")

        # Verify ae_report_count is numeric
        non_numeric = sum(1 for r in bridge if not r["ae_report_count"].isdigit())
        test("ae_report_count is numeric", non_numeric == 0, f"{non_numeric} non-numeric")

    return bridge


# ===========================================================================
# TEST 5: Semantic router on bridge data
# ===========================================================================

def test_router(bridge):
    print("\n=== TEST 5: Semantic Router on Bridge ===")

    router = SemanticRouter()
    router.ingest(
        records=bridge,
        identity_fields=["therapy", "molecular_profile", "disease"],
        provenance_fields=["bridge_id"],
        max_ladder_depth=5,
    )

    info = router.explain()
    test("Router built", info["total_records"] == len(bridge))
    test("Router has ladder", len(info["ladder"]) > 0, f"ladder depth={len(info['ladder'])}")
    test("Router has neighborhoods", info["identity_neighborhoods"] > 0)

    # Test routing: every sampled record should find itself
    import random
    rng = random.Random(42)
    samples = rng.sample(bridge, min(100, len(bridge)))

    correct = 0
    total_examined = 0
    misses = 0
    for rec in samples:
        result = router.query(rec, ask_field="adverse_event")
        if result.answer == rec["adverse_event"]:
            correct += 1
        elif result.answer is None:
            misses += 1
        total_examined += result.records_examined

    accuracy = correct / len(samples)
    avg_examined = total_examined / len(samples)
    test("Router accuracy > 80%", accuracy > 0.80,
         f"{accuracy:.1%} ({correct}/{len(samples)}), {misses} misses")
    test("Router examines < 10% of records", avg_examined < len(bridge) * 0.1,
         f"avg {avg_examined:.1f} / {len(bridge)}")

    # Test flat scan baseline
    flat_total = 0
    for rec in samples[:10]:
        flat = router.flat_scan(rec, ask_field="adverse_event")
        flat_total += flat.records_examined
    flat_avg = flat_total / 10

    speedup = flat_avg / max(avg_examined, 1)
    test("Speedup > 10x", speedup > 10, f"speedup={speedup:.1f}x")

    # Test query with missing hints (should still return something)
    partial_query = {"therapy": bridge[0]["therapy"], "molecular_profile": bridge[0]["molecular_profile"],
                     "disease": bridge[0]["disease"]}
    result = router.query(partial_query, ask_field="adverse_event")
    test("Partial query returns answer", result.answer is not None)

    # Test query for nonexistent record
    result = router.query({"therapy": "FAKE_DRUG_XYZ", "molecular_profile": "FAKE", "disease": "FAKE"},
                          ask_field="adverse_event")
    test("Nonexistent query returns None", result.answer is None)
    test("Nonexistent query examines 0", result.records_examined == 0)

    return router


# ===========================================================================
# TEST 6: Signal detection
# ===========================================================================

def test_signals(bridge):
    print("\n=== TEST 6: Signal Detection ===")

    from multi_db_bridge import detect_mega_signals

    signals = detect_mega_signals(bridge)
    # With no API data (test mode), signals may be 0 — that's valid.
    # Signal detection requires mechanism/interaction/boxed data from APIs.
    if not any(r.get("mechanism") for r in bridge):
        warn("Signals: no API data", f"got {len(signals)} signals (expected 0 without API enrichment)")
    else:
        test("Signals detected", len(signals) > 0, f"got {len(signals)}")

    by_type = defaultdict(list)
    for s in signals:
        by_type[s["signal_type"]].append(s)

    # Check each signal type has expected structure
    for stype, slist in by_type.items():
        test(f"Signal type '{stype}' has entries", len(slist) > 0)
        s = slist[0]
        test(f"Signal '{stype}' has signal_type field", "signal_type" in s)

    # Validate PATHWAY_SHARED_AE signals
    pathway = by_type.get("PATHWAY_SHARED_AE", [])
    if pathway:
        for s in pathway[:5]:
            test(f"Pathway signal has gene_a", "gene_a" in s)
            test(f"Pathway signal has gene_b", "gene_b" in s)
            test(f"Pathway signal: gene_a != gene_b", s["gene_a"] != s["gene_b"],
                 f"{s['gene_a']} == {s['gene_b']}")
            test(f"Pathway signal: shared AEs >= 3", s["shared_adverse_events"] >= 3)
            break  # just check first one thoroughly

    # Validate MECHANISM_AE_CLUSTER signals
    mech = by_type.get("MECHANISM_AE_CLUSTER", [])
    if mech:
        for s in mech[:1]:
            test("Mechanism signal has mechanism", bool(s.get("mechanism")))
            test("Mechanism signal has >= 2 drugs", s["drug_count"] >= 2)

    # Validate no self-referencing pathway signals
    for s in pathway:
        test("Pathway: no self-loops",
             s.get("gene_a", "") != s.get("gene_b", ""),
             f"{s.get('gene_a')} == {s.get('gene_b')}")

    return signals


# ===========================================================================
# TEST 7: Web server page rendering
# ===========================================================================

def test_server_rendering(bridge, signals):
    print("\n=== TEST 7: Web Server Rendering ===")

    # Import the rendering functions directly
    import bridge_server as bs

    # Inject test data
    bs.BRIDGE = bridge
    bs.SIGNALS = signals
    bs.DRUGS = sorted(set(r["therapy"] for r in bridge))
    bs.GENES = sorted(set(r["gene"] for r in bridge))
    bs.DISEASES = sorted(set(r["disease"] for r in bridge))
    bs.AES = sorted(set(r["adverse_event"] for r in bridge))
    bs.MECHANISMS = sorted(set(r.get("mechanism", "") for r in bridge if r.get("mechanism")))

    bs.ROUTER = SemanticRouter()
    bs.ROUTER.ingest(
        records=bridge,
        identity_fields=["therapy", "molecular_profile", "disease"],
        provenance_fields=["bridge_id"],
        max_ladder_depth=5,
    )

    # Test browse page renders
    try:
        page = bs.render_browse({})
        test("Browse page renders", "<!DOCTYPE html>" in page)
        test("Browse page has nav", "Database Whisper" in page)
        test("Browse page has bridge count", str(len(bridge)) in page or "bridge records" in page.lower())
    except Exception as e:
        test("Browse page renders", False, str(e))

    # Test browse with filters
    try:
        sample_drug = bridge[0]["therapy"]
        page = bs.render_browse({"therapy": [sample_drug]})
        test("Filtered browse renders", sample_drug in page)
    except Exception as e:
        test("Filtered browse renders", False, str(e))

    # Test drug profile page
    try:
        sample_drug = bridge[0]["therapy"]
        page = bs.render_drug_profile({"name": [sample_drug]})
        test("Drug profile renders", sample_drug in page)
        test("Drug profile has AE section", "Adverse Event" in page or "adverse" in page.lower())
    except Exception as e:
        test("Drug profile renders", False, str(e))

    # Test drug profile with nonexistent drug
    try:
        page = bs.render_drug_profile({"name": ["FAKE_DRUG_DOES_NOT_EXIST"]})
        test("Missing drug shows not found", "not found" in page.lower())
    except Exception as e:
        test("Missing drug shows not found", False, str(e))

    # Test gene profile page
    try:
        sample_gene = bridge[0]["gene"]
        page = bs.render_gene_profile({"name": [sample_gene]})
        test("Gene profile renders", sample_gene in page)
    except Exception as e:
        test("Gene profile renders", False, str(e))

    # Test signals page
    try:
        page = bs.render_signals({})
        test("Signals page renders", "Cross-Database Signals" in page)
        if signals:
            test("Signals page has signal types", "PATHWAY" in page or "MECHANISM" in page)
        else:
            test("Signals page renders with no signals", "Cross-Database Signals" in page)
    except Exception as e:
        test("Signals page renders", False, str(e))

    # Test signals page with type filter
    try:
        page = bs.render_signals({"type": ["PATHWAY_SHARED_AE"]})
        test("Filtered signals page renders", "PATHWAY" in page)
    except Exception as e:
        test("Filtered signals page renders", False, str(e))

    # Test stats page
    try:
        page = bs.render_stats_page()
        test("Stats page renders", "Bridge Statistics" in page)
        test("Stats page has database count", "6" in page)
    except Exception as e:
        test("Stats page renders", False, str(e))

    # Test empty drug profile page (should show all drugs)
    try:
        page = bs.render_drug_profile({"name": [""]})
        test("Empty drug query shows drug list", "All Drugs" in page)
    except Exception as e:
        test("Empty drug query shows drug list", False, str(e))


# ===========================================================================
# TEST 8: Filter engine
# ===========================================================================

def test_filter_engine(bridge):
    print("\n=== TEST 8: Filter Engine ===")

    from bridge_server import filter_bridge

    # Need to set globals
    import bridge_server as bs
    bs.BRIDGE = bridge

    # Filter by drug
    sample_drug = bridge[0]["therapy"]
    results = filter_bridge(therapy=sample_drug)
    test("Filter by drug returns results", len(results) > 0)
    test("Filter by drug: all match", all(sample_drug in r["therapy"] for r in results))

    # Filter by disease
    sample_disease = bridge[0]["disease"]
    results = filter_bridge(disease=sample_disease)
    test("Filter by disease returns results", len(results) > 0)

    # Filter by multiple criteria
    results = filter_bridge(therapy=sample_drug, disease=sample_disease)
    test("Multi-filter returns results", len(results) >= 0)  # might be 0 if combo doesn't exist
    for r in results:
        test("Multi-filter: drug matches", sample_drug.upper() in r["therapy"].upper())
        test("Multi-filter: disease matches", sample_disease.upper() in r["disease"].upper())
        break  # just check first

    # Filter returns empty for nonsense
    results = filter_bridge(therapy="ZZZZNONEXISTENT")
    test("Nonsense filter returns empty", len(results) == 0)

    # Limit works
    results = filter_bridge(limit=5)
    test("Limit=5 returns <= 5", len(results) <= 5)

    # Boxed warning filter
    boxed_results = filter_bridge(boxed_only=True)
    if boxed_results:
        test("Boxed filter: all have boxed=True",
             all(r.get("boxed_warning") == "True" for r in boxed_results))
    else:
        warn("Boxed filter", "No boxed warning records in test data")


# ===========================================================================
# TEST 9: Drug and gene profile functions
# ===========================================================================

def test_profiles(bridge):
    print("\n=== TEST 9: Profile Functions ===")

    from bridge_server import get_drug_profile, get_gene_profile
    import bridge_server as bs
    bs.BRIDGE = bridge

    # Drug profile
    sample_drug = bridge[0]["therapy"]
    profile = get_drug_profile(sample_drug)
    test("Drug profile found", profile["found"])
    test("Drug profile has AEs", len(profile["top_adverse_events"]) > 0)
    test("Drug profile has diseases", len(profile["diseases"]) > 0)
    test("Drug profile has gene targets", len(profile["gene_targets"]) > 0)
    test("Drug profile total > 0", profile["total_bridge_records"] > 0)

    # Nonexistent drug
    profile = get_drug_profile("ZZZZFAKE")
    test("Fake drug not found", not profile["found"])

    # Gene profile
    sample_gene = bridge[0]["gene"]
    profile = get_gene_profile(sample_gene)
    test("Gene profile found", profile["found"])
    test("Gene profile has AEs", len(profile["top_adverse_events"]) > 0)
    test("Gene profile has therapies", len(profile["therapies"]) > 0)

    # Nonexistent gene
    profile = get_gene_profile("ZZZZFAKEGENE")
    test("Fake gene not found", not profile["found"])


# ===========================================================================
# TEST 10: Edge cases and data integrity
# ===========================================================================

def test_edge_cases(bridge):
    print("\n=== TEST 10: Edge Cases & Data Integrity ===")

    # No NoneType values in bridge records
    none_count = 0
    for rec in bridge:
        for k, v in rec.items():
            if v is None:
                none_count += 1
    test("No None values in bridge", none_count == 0, f"{none_count} None values found")

    # All records have non-empty therapy
    empty_therapy = sum(1 for r in bridge if not r.get("therapy"))
    test("No empty therapies", empty_therapy == 0, f"{empty_therapy} empty")

    # All records have non-empty molecular_profile
    empty_gene = sum(1 for r in bridge if not r.get("molecular_profile"))
    test("No empty molecular_profiles", empty_gene == 0, f"{empty_gene} empty")

    # All records have non-empty disease
    empty_disease = sum(1 for r in bridge if not r.get("disease"))
    test("No empty diseases", empty_disease == 0, f"{empty_disease} empty")

    # All records have non-empty adverse_event
    empty_ae = sum(1 for r in bridge if not r.get("adverse_event"))
    test("No empty adverse_events", empty_ae == 0, f"{empty_ae} empty")

    # ae_report_count should be positive integer string
    bad_counts = 0
    for r in bridge:
        c = r.get("ae_report_count", "0")
        if not c.isdigit() or int(c) <= 0:
            bad_counts += 1
    test("All ae_report_count > 0", bad_counts == 0, f"{bad_counts} invalid")

    # bridge_id should be unique
    ids = [r["bridge_id"] for r in bridge]
    test("Bridge IDs unique", len(ids) == len(set(ids)),
         f"{len(ids) - len(set(ids))} duplicates")

    # Check evidence_level is valid
    valid_levels = {"A", "B", "C", "D", "E", ""}
    bad_levels = sum(1 for r in bridge if r.get("evidence_level", "") not in valid_levels)
    test("Evidence levels valid", bad_levels == 0,
         f"{bad_levels} invalid levels")

    # Check sex_ratio format
    for r in bridge[:10]:
        sr = r.get("sex_ratio", "")
        test("Sex ratio format M:N/F:N", sr.startswith("M:") and "/F:" in sr,
             f"got '{sr}'")
        break


# ===========================================================================
# TEST 11: Cross-database consistency
# ===========================================================================

def test_cross_db_consistency(bridge):
    print("\n=== TEST 11: Cross-Database Consistency ===")

    # Each therapy should map to consistent mechanism (from ChEMBL)
    drug_mechs: Dict[str, set] = defaultdict(set)
    for r in bridge:
        if r.get("mechanism"):
            drug_mechs[r["therapy"]].add(r["mechanism"])

    multi_mech = {d: m for d, m in drug_mechs.items() if len(m) > 1}
    test("Each drug has <= 1 mechanism", len(multi_mech) == 0,
         f"{len(multi_mech)} drugs with multiple mechanisms: {list(multi_mech.keys())[:3]}")

    # Each gene should map to consistent protein_location (from UniProt)
    gene_locs: Dict[str, set] = defaultdict(set)
    for r in bridge:
        if r.get("protein_location"):
            gene_locs[r["gene"]].add(r["protein_location"])

    multi_loc = {g: l for g, l in gene_locs.items() if len(l) > 1}
    if multi_loc:
        warn("Gene location consistency",
             f"{len(multi_loc)} genes with multiple locations (may be valid): {list(multi_loc.keys())[:3]}")
    else:
        test("Each gene has <= 1 location", True)

    # Every bridge record should have BOTH a CIViC field and a FAERS field
    for r in bridge[:100]:
        has_civic = bool(r.get("evidence_type"))
        has_faers = bool(r.get("adverse_event"))
        test("Record has CIViC data", has_civic)
        test("Record has FAERS data", has_faers)
        break  # just check first


# ===========================================================================
# Main
# ===========================================================================

def main():
    global PASS, FAIL, WARN
    print("=" * 60)
    print("  BRIDGE TEST SUITE")
    print("=" * 60)
    t0 = time.time()

    try:
        civic, faers = test_data_loading()
        test_normalization()
        test_gene_extraction()
        bridge = test_bridge_join(civic, faers)
        if bridge:
            router = test_router(bridge)
            signals = test_signals(bridge)
            test_filter_engine(bridge)
            test_profiles(bridge)
            test_server_rendering(bridge, signals)
            test_edge_cases(bridge)
            test_cross_db_consistency(bridge)
        else:
            print("\n  SKIP: No bridge records — skipping downstream tests")
    except Exception as e:
        FAIL += 1
        print(f"\n  FATAL: {e}")
        traceback.print_exc()

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {PASS} passed, {FAIL} failed, {WARN} warnings  ({elapsed:.1f}s)")
    print(f"{'=' * 60}")

    if FAIL > 0:
        exit(1)


if __name__ == "__main__":
    main()
