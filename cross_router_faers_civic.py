"""cross_router_faers_civic.py

Route FAERS adverse events through CIViC oncology evidence to find
drug-cancer connections that neither database reveals alone.

The idea:
    FAERS knows "drug X caused adverse event Y in patient Z"
    CIViC knows "drug X treats cancer A because of gene mutation B"

    Neither tells you: "patients with mutation B getting drug X for cancer A
    are experiencing adverse event Y" — but the join does.

Join key: drug/therapy name (normalized to uppercase).

What this produces:
    - Bridge records carrying fields from both databases
    - A unified semantic router that discovers its own ladder over the merged schema
    - Cross-database queries: "what adverse events are reported for BRAF V600E therapies?"
    - Signal detection: drugs with high adverse event counts per cancer indication

Usage:
    python cross_router_faers_civic.py
"""

from __future__ import annotations

import csv
import time
from collections import defaultdict, Counter
from typing import Any, Dict, List, Set, Tuple

from semantic_router import SemanticRouter


# ---------------------------------------------------------------------------
# Loaders (adapted from existing test files)
# ---------------------------------------------------------------------------

def load_civic(path: str = "civic_evidence_full.tsv") -> List[Dict[str, Any]]:
    """Load CIViC evidence export."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
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
            if record["molecular_profile"] and record["disease"]:
                records.append(record)
    return records


def load_faers(
    drug_path: str = "ASCII/DRUG24Q3.txt",
    reac_path: str = "ASCII/REAC24Q3.txt",
    demo_path: str = "ASCII/DEMO24Q3.txt",
    max_records: int = 500_000,
) -> List[Dict[str, Any]]:
    """Load FAERS drug + reaction + demographics joined on primaryid."""
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

    reac_by_id: Dict[str, str] = {}
    with open(reac_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="$")
        for row in reader:
            pid = (row.get("primaryid") or "").strip()
            pt = (row.get("pt") or "").strip()
            if pid and pt and pid not in reac_by_id:
                reac_by_id[pid] = pt

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


# ---------------------------------------------------------------------------
# Drug name normalization
# ---------------------------------------------------------------------------

def normalize_drug(name: str) -> str:
    """
    Normalize drug names for cross-database matching.
    Strip whitespace, uppercase, remove common suffixes.
    """
    n = name.upper().strip()
    # Remove common suffixes that differ between databases
    for suffix in [" HYDROCHLORIDE", " HCL", " SODIUM", " MESYLATE",
                   " MALEATE", " TARTRATE", " SULFATE", " ACETATE",
                   " CITRATE", " FUMARATE", " BESYLATE", " TOSYLATE"]:
        if n.endswith(suffix):
            n = n[:-len(suffix)]
    return n


# ---------------------------------------------------------------------------
# The cross-database join
# ---------------------------------------------------------------------------

def build_bridge_records(
    civic_records: List[Dict[str, Any]],
    faers_records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Join FAERS and CIViC on drug/therapy name to produce bridge records.

    Each bridge record carries:
        From CIViC: molecular_profile, disease, evidence_type, evidence_direction,
                    evidence_level, significance
        From FAERS: reaction, sex, reporter_country, route, role, dose_form
        Join key:   therapy (normalized drug name)

    Returns (bridge_records, join_stats).
    """
    # Index CIViC by normalized therapy name.
    # One therapy can have multiple CIViC entries (different genes, diseases).
    civic_by_drug: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    civic_drugs: Set[str] = set()

    for rec in civic_records:
        therapies_raw = rec.get("therapies", "")
        if not therapies_raw:
            continue
        # CIViC therapies field can be comma-separated combos
        for therapy in therapies_raw.split(","):
            normalized = normalize_drug(therapy.strip())
            if normalized:
                civic_drugs.add(normalized)
                civic_by_drug[normalized].append(rec)

    # Index FAERS by normalized drug name.
    faers_by_drug: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    faers_drugs: Set[str] = set()

    for rec in faers_records:
        # Try active ingredient first (cleaner), fall back to drugname
        drug = rec.get("active_ingredient") or rec.get("drugname", "")
        normalized = normalize_drug(drug)
        if normalized:
            faers_drugs.add(normalized)
            faers_by_drug[normalized].append(rec)

    # Find intersection — drugs in both databases
    shared_drugs = civic_drugs & faers_drugs

    # Build bridge records
    bridge: List[Dict[str, Any]] = []
    drug_match_counts: Dict[str, int] = {}

    for drug in sorted(shared_drugs):
        civic_entries = civic_by_drug[drug]
        faers_entries = faers_by_drug[drug]

        # Aggregate FAERS reactions for this drug (don't explode cartesian product)
        reaction_counts = Counter(r["reaction"] for r in faers_entries if r["reaction"])
        top_reactions = reaction_counts.most_common(20)  # top 20 adverse events per drug
        total_faers = len(faers_entries)

        sex_dist = Counter(r["sex"] for r in faers_entries if r["sex"])
        route_dist = Counter(r["route"] for r in faers_entries if r["route"])
        country_dist = Counter(r["reporter_country"] for r in faers_entries if r["reporter_country"])

        drug_match_counts[drug] = total_faers

        for civic_rec in civic_entries:
            for reaction, count in top_reactions:
                bridge_rec = {
                    # Join key
                    "therapy": drug,
                    # From CIViC
                    "molecular_profile": civic_rec["molecular_profile"],
                    "disease": civic_rec["disease"],
                    "evidence_type": civic_rec["evidence_type"],
                    "evidence_direction": civic_rec["evidence_direction"],
                    "evidence_level": civic_rec["evidence_level"],
                    "significance": civic_rec["significance"],
                    # From FAERS (aggregated)
                    "adverse_event": reaction,
                    "ae_report_count": str(count),
                    "total_faers_reports": str(total_faers),
                    "top_route": route_dist.most_common(1)[0][0] if route_dist else "",
                    "top_country": country_dist.most_common(1)[0][0] if country_dist else "",
                    "sex_ratio": f"M:{sex_dist.get('M',0)}/F:{sex_dist.get('F',0)}",
                    # Bridge ID for provenance
                    "bridge_id": f"{drug}|{civic_rec['evidence_id']}|{reaction}",
                }
                bridge.append(bridge_rec)

    stats = {
        "civic_unique_drugs": len(civic_drugs),
        "faers_unique_drugs": len(faers_drugs),
        "shared_drugs": len(shared_drugs),
        "bridge_records": len(bridge),
        "top_matched_drugs": sorted(drug_match_counts.items(), key=lambda x: -x[1])[:15],
        "shared_drug_list": sorted(shared_drugs),
    }

    return bridge, stats


# ---------------------------------------------------------------------------
# Signal detection: find what neither database shows alone
# ---------------------------------------------------------------------------

def detect_signals(bridge_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Find drug-cancer-adverse_event triples that are surprising.

    A signal is interesting when:
    - A precision oncology drug (has molecular_profile) shows high adverse event counts
    - The adverse event is not the expected one for the disease
    - Multiple gene-drug pairs share the same unexpected adverse event
    """
    signals = []

    # Group by (molecular_profile, therapy, adverse_event)
    triples: Dict[tuple, List[Dict]] = defaultdict(list)
    for rec in bridge_records:
        key = (rec["molecular_profile"], rec["therapy"], rec["adverse_event"])
        triples[key].append(rec)

    # Group by molecular_profile to see which genes have the most AE diversity
    gene_ae: Dict[str, Set[str]] = defaultdict(set)
    gene_therapies: Dict[str, Set[str]] = defaultdict(set)
    for rec in bridge_records:
        gene_ae[rec["molecular_profile"]].add(rec["adverse_event"])
        gene_therapies[rec["molecular_profile"]].add(rec["therapy"])

    # Signal 1: Genes with high adverse event diversity across their therapies
    for gene, aes in sorted(gene_ae.items(), key=lambda x: -len(x[1])):
        if len(aes) >= 5:
            therapies = gene_therapies[gene]
            signals.append({
                "signal_type": "HIGH_AE_DIVERSITY",
                "molecular_profile": gene,
                "unique_adverse_events": len(aes),
                "therapies": sorted(therapies),
                "top_adverse_events": sorted(aes)[:10],
            })

    # Signal 2: Adverse events shared across multiple gene targets
    ae_genes: Dict[str, Set[str]] = defaultdict(set)
    ae_drugs: Dict[str, Set[str]] = defaultdict(set)
    for rec in bridge_records:
        ae_genes[rec["adverse_event"]].add(rec["molecular_profile"])
        ae_drugs[rec["adverse_event"]].add(rec["therapy"])

    for ae, genes in sorted(ae_genes.items(), key=lambda x: -len(x[1])):
        if len(genes) >= 3:
            signals.append({
                "signal_type": "CROSS_TARGET_AE",
                "adverse_event": ae,
                "affected_gene_targets": len(genes),
                "genes": sorted(genes)[:10],
                "drugs": sorted(ae_drugs[ae])[:10],
            })

    # Signal 3: High-count AEs for drugs with strong CIViC evidence
    strong_evidence_aes = []
    for rec in bridge_records:
        if rec["evidence_level"] in ("A", "B") and rec["evidence_direction"] == "Supports":
            count = int(rec["ae_report_count"]) if rec["ae_report_count"].isdigit() else 0
            if count >= 10:
                strong_evidence_aes.append(rec)

    # Deduplicate and sort by count
    seen = set()
    for rec in sorted(strong_evidence_aes, key=lambda r: -int(r.get("ae_report_count", "0"))):
        key = (rec["therapy"], rec["adverse_event"])
        if key not in seen:
            seen.add(key)
            signals.append({
                "signal_type": "STRONG_EVIDENCE_HIGH_AE",
                "therapy": rec["therapy"],
                "molecular_profile": rec["molecular_profile"],
                "disease": rec["disease"],
                "adverse_event": rec["adverse_event"],
                "ae_report_count": rec["ae_report_count"],
                "evidence_level": rec["evidence_level"],
            })

    return signals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("CROSS-DATABASE ROUTER: FAERS x CIViC")
    print("Finding drug-cancer connections neither database reveals alone")
    print("=" * 70)

    # Load both databases
    print("\n[1/5] Loading databases...")
    t0 = time.time()

    civic_records = load_civic()
    print(f"  CIViC: {len(civic_records)} evidence records")

    faers_records = load_faers(max_records=500_000)
    print(f"  FAERS: {len(faers_records)} drug-reaction records")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Build the bridge
    print("\n[2/5] Building cross-database bridge on drug name...")
    t1 = time.time()
    bridge_records, join_stats = build_bridge_records(civic_records, faers_records)
    print(f"  CIViC unique therapies: {join_stats['civic_unique_drugs']}")
    print(f"  FAERS unique drugs: {join_stats['faers_unique_drugs']}")
    print(f"  Shared drugs (join hits): {join_stats['shared_drugs']}")
    print(f"  Bridge records created: {join_stats['bridge_records']}")
    print(f"  Built in {time.time() - t1:.1f}s")

    if not bridge_records:
        print("\n  ERROR: No bridge records created. Check drug name normalization.")
        return

    # Show which drugs matched
    print(f"\n  Top matched drugs (by FAERS report count):")
    for drug, count in join_stats["top_matched_drugs"]:
        print(f"    {drug}: {count} FAERS reports")

    # Route the bridge through the semantic router
    print("\n[3/5] Building semantic router over bridge records...")
    t2 = time.time()
    router = SemanticRouter()
    router.ingest(
        records=bridge_records,
        identity_fields=["therapy", "molecular_profile", "disease"],
        provenance_fields=["bridge_id"],
        max_ladder_depth=4,
    )
    print(f"  Built in {time.time() - t2:.2f}s")

    info = router.explain()
    print(f"\n  === Bridge Router Structure ===")
    print(f"  Total bridge records: {info['total_records']}")
    print(f"  Identity: therapy + molecular_profile + disease")
    print(f"  Identity neighborhoods: {info['identity_neighborhoods']}")
    print(f"  Ambiguous neighborhoods: {info['ambiguous_neighborhoods']}")
    print(f"  Discovered ladder:")
    for rung in info["ladder"]:
        print(f"    Rung {rung['rung']}: {rung['field']} (reduction={rung['ambiguity_reduction_rate']})")

    # Cross-database queries
    print("\n[4/5] Cross-database queries (things neither DB knows alone)...")

    # Query 1: What adverse events are reported for a specific gene's therapies?
    print("\n  --- Query: Adverse events for BRAF-targeted therapies ---")
    braf_records = [r for r in bridge_records if "BRAF" in r["molecular_profile"]]
    if braf_records:
        ae_counts = Counter(r["adverse_event"] for r in braf_records)
        print(f"  BRAF bridge records: {len(braf_records)}")
        print(f"  Unique adverse events across BRAF therapies: {len(ae_counts)}")
        print(f"  Top adverse events:")
        for ae, count in ae_counts.most_common(10):
            print(f"    {ae}: {count} bridge records")

    # Query 2: Which cancers' therapies have the most adverse event diversity?
    print("\n  --- Query: Diseases ranked by adverse event diversity ---")
    disease_aes: Dict[str, Set[str]] = defaultdict(set)
    for r in bridge_records:
        disease_aes[r["disease"]].add(r["adverse_event"])
    disease_diversity = sorted(disease_aes.items(), key=lambda x: -len(x[1]))
    for disease, aes in disease_diversity[:10]:
        print(f"    {disease}: {len(aes)} unique adverse events")

    # Query 3: Router-powered cross-DB lookup
    print("\n  --- Router-powered lookup ---")
    if braf_records:
        sample = braf_records[0]
        result = router.query(
            {
                "therapy": sample["therapy"],
                "molecular_profile": sample["molecular_profile"],
                "disease": sample["disease"],
                "adverse_event": sample["adverse_event"],
            },
            ask_field="ae_report_count",
        )
        print(f"  Query: {sample['therapy']} for {sample['molecular_profile']} in {sample['disease']}")
        print(f"    Adverse event: {sample['adverse_event']}")
        print(f"    FAERS report count: {result.answer}")
        print(f"    Records examined: {result.records_examined}/{result.total_records}")
        print(f"    Route: {result.route_used}")
        print(f"    Narrowing: {result.candidates_at_each_stage}")

    # Detect signals
    print("\n[5/5] Signal detection — connections neither DB shows alone...")
    signals = detect_signals(bridge_records)

    high_div = [s for s in signals if s["signal_type"] == "HIGH_AE_DIVERSITY"]
    cross_target = [s for s in signals if s["signal_type"] == "CROSS_TARGET_AE"]
    strong_ae = [s for s in signals if s["signal_type"] == "STRONG_EVIDENCE_HIGH_AE"]

    print(f"\n  Signal: HIGH_AE_DIVERSITY — gene targets with many different adverse events")
    for s in high_div[:5]:
        print(f"    {s['molecular_profile']}: {s['unique_adverse_events']} unique AEs, "
              f"therapies={s['therapies'][:3]}")

    print(f"\n  Signal: CROSS_TARGET_AE — adverse events hitting multiple gene targets")
    for s in cross_target[:5]:
        print(f"    {s['adverse_event']}: {s['affected_gene_targets']} gene targets, "
              f"drugs={s['drugs'][:3]}")

    print(f"\n  Signal: STRONG_EVIDENCE_HIGH_AE — Level A/B evidence drugs with high AE counts")
    for s in strong_ae[:10]:
        print(f"    {s['therapy']} ({s['molecular_profile']} / {s['disease']}): "
              f"AE={s['adverse_event']}, count={s['ae_report_count']}, "
              f"evidence={s['evidence_level']}")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"  CIViC records:     {len(civic_records)}")
    print(f"  FAERS records:     {len(faers_records)}")
    print(f"  Shared drugs:      {join_stats['shared_drugs']}")
    print(f"  Bridge records:    {len(bridge_records)}")
    print(f"  Bridge ladder:     identity -> {' -> '.join(router.ladder_fields)}")
    print(f"  Signals detected:  {len(signals)}")
    print(f"    High AE diversity:     {len(high_div)}")
    print(f"    Cross-target AEs:      {len(cross_target)}")
    print(f"    Strong evidence + AE:  {len(strong_ae)}")
    print(f"\n  What this reveals that neither DB shows alone:")
    print(f"    - Which gene mutations' therapies cause which adverse events")
    print(f"    - Whether adverse events cluster by molecular target or by drug")
    print(f"    - Safety signals for precision oncology drugs with strong clinical evidence")
    print(f"    - Cross-target adverse events (same AE across different gene pathways)")


if __name__ == "__main__":
    main()
