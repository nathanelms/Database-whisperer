"""multi_db_bridge.py

Route FAERS adverse events through CIViC, ClinicalTrials.gov, ChEMBL,
UniProt, STRING, and OpenFDA to build maximum associations.

Six databases, one semantic router. Each adds a dimension the others lack:

    CIViC:              gene/variant -> drug -> disease (clinical evidence)
    FAERS:              drug -> adverse event -> patient demographics
    ClinicalTrials.gov: drug + disease -> trial phase, status, enrollment
    ChEMBL:             drug -> mechanism of action, target, bioactivity
    UniProt:            gene -> protein function, pathway, subcellular location
    STRING:             gene -> protein-protein interaction partners
    OpenFDA:            drug -> labeled indications, boxed warnings

Join surfaces:
    drug name:  FAERS <-> CIViC <-> ClinicalTrials <-> ChEMBL <-> OpenFDA
    gene name:  CIViC <-> UniProt <-> STRING <-> ChEMBL

Usage:
    python multi_db_bridge.py
"""

from __future__ import annotations

import csv
import json
import os
import time
import urllib.request
import urllib.parse
import urllib.error
import ssl
from collections import defaultdict, Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from semantic_router import SemanticRouter


# ---------------------------------------------------------------------------
# SSL context for API calls (Windows sometimes needs this)
# ---------------------------------------------------------------------------

SSL_CTX = ssl._create_unverified_context()


# ---------------------------------------------------------------------------
# Cache layer — avoid re-fetching on reruns
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".api_cache")


def _cache_path(api: str, key: str) -> str:
    safe_key = urllib.parse.quote(key, safe="")[:120]
    d = os.path.join(CACHE_DIR, api)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{safe_key}.json")


def _cache_get(api: str, key: str) -> Optional[Any]:
    p = _cache_path(api, key)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _cache_set(api: str, key: str, data: Any) -> None:
    p = _cache_path(api, key)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# Safe HTTP fetcher
# ---------------------------------------------------------------------------

def fetch_json(url: str, timeout: int = 15) -> Optional[Any]:
    """Fetch JSON from a URL, return None on failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MemoryLab/1.0"})
        with urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Local data loaders (reuse from cross_router_faers_civic.py)
# ---------------------------------------------------------------------------

def load_civic(path: str = "civic_evidence_full.tsv") -> List[Dict[str, Any]]:
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
    demo_by_id: Dict[str, Dict[str, str]] = {}
    with open(demo_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="$")
        for row in reader:
            pid = (row.get("primaryid") or "").strip()
            if pid:
                demo_by_id[pid] = {
                    "sex": (row.get("sex") or "").strip(),
                    "reporter_country": (row.get("reporter_country") or "").strip(),
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
            route = (row.get("route") or "").strip()
            prod_ai = (row.get("prod_ai") or "").strip().upper()
            if not drugname or not pid:
                continue
            demo = demo_by_id.get(pid, {})
            reaction = reac_by_id.get(pid, "")
            records.append({
                "drugname": drugname,
                "active_ingredient": prod_ai,
                "route": route,
                "reaction": reaction,
                "sex": demo.get("sex", ""),
                "reporter_country": demo.get("reporter_country", ""),
                "primaryid": pid,
            })
    return records


def normalize_drug(name: str) -> str:
    n = name.upper().strip()
    for suffix in [" HYDROCHLORIDE", " HCL", " SODIUM", " MESYLATE",
                   " MALEATE", " TARTRATE", " SULFATE", " ACETATE",
                   " CITRATE", " FUMARATE", " BESYLATE", " TOSYLATE"]:
        if n.endswith(suffix):
            n = n[:-len(suffix)]
    return n


def extract_gene(molecular_profile: str) -> str:
    """Extract the primary gene name from a CIViC molecular profile string."""
    mp = molecular_profile.strip()
    # Handle "GENE VARIANT" patterns
    parts = mp.split()
    if parts:
        gene = parts[0]
        # Clean common non-gene prefixes
        if gene in ("NOT", "AND", "OR"):
            return parts[1] if len(parts) > 1 else ""
        return gene
    return ""


# ---------------------------------------------------------------------------
# API pullers — one per database
# ---------------------------------------------------------------------------

def pull_clinicaltrials(drug: str) -> Dict[str, Any]:
    """Pull trial info from ClinicalTrials.gov v2 API."""
    cached = _cache_get("ctgov", drug)
    if cached is not None:
        return cached

    encoded = urllib.parse.quote(drug)
    url = (f"https://clinicaltrials.gov/api/v2/studies?"
           f"query.intr={encoded}&filter.overallStatus=RECRUITING,COMPLETED"
           f"&pageSize=5&fields=protocolSection")

    data = fetch_json(url)
    result = {"trials": 0, "phases": [], "statuses": [], "conditions": []}

    if data and "studies" in data:
        studies = data["studies"]
        result["trials"] = data.get("totalCount", len(studies))
        for study in studies[:5]:
            proto = study.get("protocolSection", {})
            design = proto.get("designModule", {})
            status_mod = proto.get("statusModule", {})
            cond_mod = proto.get("conditionsModule", {})

            phases = design.get("phases", [])
            if phases:
                result["phases"].extend(phases)
            status = status_mod.get("overallStatus", "")
            if status:
                result["statuses"].append(status)
            conditions = cond_mod.get("conditions", [])
            result["conditions"].extend(conditions[:3])

    # Deduplicate
    result["phases"] = list(set(result["phases"]))
    result["statuses"] = list(set(result["statuses"]))
    result["conditions"] = list(set(result["conditions"]))[:10]

    _cache_set("ctgov", drug, result)
    return result


def pull_chembl(drug: str) -> Dict[str, Any]:
    """Pull mechanism of action and target info from ChEMBL."""
    cached = _cache_get("chembl", drug)
    if cached is not None:
        return cached

    encoded = urllib.parse.quote(drug.lower())
    # Search for the molecule
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?q={encoded}&limit=1"
    data = fetch_json(url)

    result = {"mechanism": "", "target_type": "", "max_phase": "", "molecule_type": ""}

    if data and "molecules" in data and data["molecules"]:
        mol = data["molecules"][0]
        chembl_id = mol.get("molecule_chembl_id", "")
        result["max_phase"] = str(mol.get("max_phase", ""))
        result["molecule_type"] = mol.get("molecule_type", "")

        # Get mechanism of action
        if chembl_id:
            mech_url = f"https://www.ebi.ac.uk/chembl/api/data/mechanism.json?molecule_chembl_id={chembl_id}&limit=3"
            mech_data = fetch_json(mech_url)
            if mech_data and "mechanisms" in mech_data and mech_data["mechanisms"]:
                mechs = mech_data["mechanisms"]
                result["mechanism"] = mechs[0].get("mechanism_of_action", "")
                result["target_type"] = mechs[0].get("target_type", "")

    _cache_set("chembl", drug, result)
    return result


def pull_uniprot(gene: str) -> Dict[str, Any]:
    """Pull protein function and pathway from UniProt."""
    cached = _cache_get("uniprot", gene)
    if cached is not None:
        return cached

    encoded = urllib.parse.quote(gene)
    url = (f"https://rest.uniprot.org/uniprotkb/search?"
           f"query=gene_exact:{encoded}+AND+organism_id:9606"
           f"&format=json&size=1"
           f"&fields=protein_name,cc_function,cc_pathway,cc_subcellular_location")

    data = fetch_json(url)
    result = {"protein_name": "", "function": "", "pathway": "", "location": ""}

    if data and "results" in data and data["results"]:
        entry = data["results"][0]
        # Protein name
        pname = entry.get("proteinDescription", {})
        rec_name = pname.get("recommendedName", {})
        if rec_name:
            result["protein_name"] = rec_name.get("fullName", {}).get("value", "")

        # Function and pathway from comments
        for comment in entry.get("comments", []):
            ctype = comment.get("commentType", "")
            if ctype == "FUNCTION":
                texts = comment.get("texts", [])
                if texts:
                    func_text = texts[0].get("value", "")
                    # Truncate to keep record compact
                    result["function"] = func_text[:200]
            elif ctype == "PATHWAY":
                texts = comment.get("texts", [])
                if texts:
                    result["pathway"] = texts[0].get("value", "")
            elif ctype == "SUBCELLULAR LOCATION":
                locs = comment.get("subcellularLocations", [])
                if locs:
                    loc = locs[0].get("location", {})
                    result["location"] = loc.get("value", "")

    _cache_set("uniprot", gene, result)
    return result


def pull_string(gene: str) -> Dict[str, Any]:
    """Pull protein interaction partners from STRING."""
    cached = _cache_get("string", gene)
    if cached is not None:
        return cached

    encoded = urllib.parse.quote(gene)
    url = (f"https://string-db.org/api/json/interaction_partners?"
           f"identifiers={encoded}&species=9606&limit=10"
           f"&required_score=700")

    data = fetch_json(url)
    result = {"partners": [], "partner_count": 0}

    if data and isinstance(data, list):
        partners = []
        for interaction in data:
            partner = interaction.get("preferredName_B", "")
            score = interaction.get("score", 0)
            if partner and partner != gene:
                partners.append({"gene": partner, "score": score})
        result["partners"] = partners[:10]
        result["partner_count"] = len(partners)

    _cache_set("string", gene, result)
    return result


def pull_openfda_label(drug: str) -> Dict[str, Any]:
    """Pull drug label info (indications, warnings) from OpenFDA."""
    cached = _cache_get("openfda", drug)
    if cached is not None:
        return cached

    encoded = urllib.parse.quote(drug.lower())
    url = (f"https://api.fda.gov/drug/label.json?"
           f"search=openfda.generic_name:\"{encoded}\"&limit=1")

    data = fetch_json(url)
    result = {"indications": "", "boxed_warning": False, "drug_class": ""}

    if data and "results" in data and data["results"]:
        label = data["results"][0]
        # Indications (truncate)
        indications = label.get("indications_and_usage", [""])
        if indications:
            result["indications"] = indications[0][:200]
        # Boxed warning
        boxed = label.get("boxed_warning", [])
        result["boxed_warning"] = bool(boxed)
        # Drug class from openfda
        openfda = label.get("openfda", {})
        pharm_class = openfda.get("pharm_class_epc", [])
        if pharm_class:
            result["drug_class"] = pharm_class[0]

    _cache_set("openfda", drug, result)
    return result


# ---------------------------------------------------------------------------
# Batch API pulls with progress
# ---------------------------------------------------------------------------

def pull_all_apis(
    shared_drugs: List[str],
    unique_genes: List[str],
) -> Dict[str, Dict[str, Dict]]:
    """
    Pull from all 5 external APIs for our shared drugs and genes.
    Returns {api_name: {key: result_dict}}.
    """
    results: Dict[str, Dict[str, Dict]] = {
        "ctgov": {}, "chembl": {}, "uniprot": {}, "string": {}, "openfda": {},
    }

    total_calls = len(shared_drugs) * 3 + len(unique_genes) * 2
    done = 0

    def tick(label):
        nonlocal done
        done += 1
        if done % 20 == 0 or done == total_calls:
            print(f"    API progress: {done}/{total_calls} ({label})")

    # Drug-keyed APIs
    for drug in shared_drugs:
        results["ctgov"][drug] = pull_clinicaltrials(drug)
        tick("ClinicalTrials")
        time.sleep(0.15)

        results["chembl"][drug] = pull_chembl(drug)
        tick("ChEMBL")
        time.sleep(0.15)

        results["openfda"][drug] = pull_openfda_label(drug)
        tick("OpenFDA")
        time.sleep(0.15)

    # Gene-keyed APIs
    for gene in unique_genes:
        results["uniprot"][gene] = pull_uniprot(gene)
        tick("UniProt")
        time.sleep(0.15)

        results["string"][gene] = pull_string(gene)
        tick("STRING")
        time.sleep(0.3)  # STRING rate limit is stricter

    return results


# ---------------------------------------------------------------------------
# Build the mega-bridge
# ---------------------------------------------------------------------------

def build_mega_bridge(
    civic_records: List[Dict[str, Any]],
    faers_records: List[Dict[str, Any]],
    api_data: Dict[str, Dict[str, Dict]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Join all 6 databases into mega-bridge records.

    Schema per record:
        therapy, molecular_profile, disease          (join keys)
        evidence_type, evidence_direction,            (CIViC)
        evidence_level, significance
        adverse_event, ae_report_count, sex_ratio     (FAERS)
        trial_phase, trial_count                      (ClinicalTrials.gov)
        mechanism, target_type, max_phase             (ChEMBL)
        protein_function, pathway, location           (UniProt)
        interaction_partners, partner_count            (STRING)
        boxed_warning, drug_class                     (OpenFDA)
    """
    # Index CIViC by normalized therapy
    civic_by_drug: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    civic_drugs: Set[str] = set()
    for rec in civic_records:
        therapies_raw = rec.get("therapies", "")
        if not therapies_raw:
            continue
        for therapy in therapies_raw.split(","):
            normalized = normalize_drug(therapy.strip())
            if normalized:
                civic_drugs.add(normalized)
                civic_by_drug[normalized].append(rec)

    # Index FAERS by normalized drug
    faers_by_drug: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    faers_drugs: Set[str] = set()
    for rec in faers_records:
        drug = rec.get("active_ingredient") or rec.get("drugname", "")
        normalized = normalize_drug(drug)
        if normalized:
            faers_drugs.add(normalized)
            faers_by_drug[normalized].append(rec)

    shared_drugs = civic_drugs & faers_drugs

    # Build bridge
    bridge: List[Dict[str, Any]] = []
    db_hit_counts = {"ctgov": 0, "chembl": 0, "uniprot": 0, "string": 0, "openfda": 0}

    for drug in sorted(shared_drugs):
        civic_entries = civic_by_drug[drug]
        faers_entries = faers_by_drug[drug]

        # Aggregate FAERS
        reaction_counts = Counter(r["reaction"] for r in faers_entries if r["reaction"])
        top_reactions = reaction_counts.most_common(15)
        total_faers = len(faers_entries)
        sex_dist = Counter(r["sex"] for r in faers_entries if r["sex"])

        # API data for this drug
        ct = api_data["ctgov"].get(drug, {})
        ch = api_data["chembl"].get(drug, {})
        fda = api_data["openfda"].get(drug, {})

        if ct.get("trials", 0) > 0:
            db_hit_counts["ctgov"] += 1
        if ch.get("mechanism"):
            db_hit_counts["chembl"] += 1
        if fda.get("drug_class"):
            db_hit_counts["openfda"] += 1

        trial_phase = ", ".join(ct.get("phases", [])) or "unknown"
        trial_count = str(ct.get("trials", 0))

        for civic_rec in civic_entries:
            gene = extract_gene(civic_rec["molecular_profile"])

            # API data for this gene
            up = api_data["uniprot"].get(gene, {})
            st = api_data["string"].get(gene, {})

            if up.get("protein_name"):
                db_hit_counts["uniprot"] += 1
            if st.get("partner_count", 0) > 0:
                db_hit_counts["string"] += 1

            partner_genes = [p["gene"] for p in st.get("partners", [])[:5]]

            for reaction, count in top_reactions:
                bridge_rec = {
                    # Join keys
                    "therapy": drug,
                    "molecular_profile": civic_rec["molecular_profile"],
                    "gene": gene,
                    "disease": civic_rec["disease"],
                    # CIViC
                    "evidence_type": civic_rec["evidence_type"],
                    "evidence_direction": civic_rec["evidence_direction"],
                    "evidence_level": civic_rec["evidence_level"],
                    "significance": civic_rec["significance"],
                    # FAERS
                    "adverse_event": reaction,
                    "ae_report_count": str(count),
                    "total_faers_reports": str(total_faers),
                    "sex_ratio": f"M:{sex_dist.get('M',0)}/F:{sex_dist.get('F',0)}",
                    # ClinicalTrials.gov
                    "trial_phase": trial_phase,
                    "trial_count": trial_count,
                    # ChEMBL
                    "mechanism": ch.get("mechanism", ""),
                    "target_type": ch.get("target_type", ""),
                    "chembl_max_phase": ch.get("max_phase", ""),
                    # UniProt
                    "protein_function": (up.get("function", "") or "")[:100],
                    "protein_pathway": up.get("pathway", ""),
                    "protein_location": up.get("location", ""),
                    # STRING
                    "interaction_partners": ", ".join(partner_genes),
                    "partner_count": str(st.get("partner_count", 0)),
                    # OpenFDA
                    "boxed_warning": str(fda.get("boxed_warning", False)),
                    "drug_class": fda.get("drug_class", ""),
                    # Provenance
                    "bridge_id": f"{drug}|{civic_rec['evidence_id']}|{reaction}",
                }
                bridge.append(bridge_rec)

    stats = {
        "shared_drugs": len(shared_drugs),
        "bridge_records": len(bridge),
        "db_hits": db_hit_counts,
        "databases_used": 6,
    }
    return bridge, stats


# ---------------------------------------------------------------------------
# Signal detection across all 6 databases
# ---------------------------------------------------------------------------

def detect_mega_signals(bridge: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Detect cross-database signals that no single database reveals.
    """
    signals = []

    # Signal 1: Boxed warning drugs with strong CIViC evidence
    # "FDA flagged this drug AND it has Level A evidence for a specific mutation"
    for rec in bridge:
        if rec["boxed_warning"] == "True" and rec["evidence_level"] in ("A", "B"):
            count = int(rec["ae_report_count"]) if rec["ae_report_count"].isdigit() else 0
            if count >= 20:
                signals.append({
                    "signal_type": "BOXED_WARNING_STRONG_EVIDENCE",
                    "therapy": rec["therapy"],
                    "gene": rec["molecular_profile"],
                    "disease": rec["disease"],
                    "adverse_event": rec["adverse_event"],
                    "ae_count": count,
                    "evidence_level": rec["evidence_level"],
                    "drug_class": rec["drug_class"],
                    "mechanism": rec["mechanism"],
                })

    # Signal 2: Interaction partners sharing adverse events
    # "Genes that interact (STRING) have therapies causing the same AE"
    gene_ae_drugs: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    gene_partners: Dict[str, List[str]] = {}
    for rec in bridge:
        gene_ae_drugs[rec["gene"]][rec["adverse_event"]].add(rec["therapy"])
        if rec["interaction_partners"]:
            gene_partners[rec["gene"]] = [p.strip() for p in rec["interaction_partners"].split(",") if p.strip()]

    for gene, partners in gene_partners.items():
        if gene not in gene_ae_drugs:
            continue
        for partner in partners:
            if partner not in gene_ae_drugs:
                continue
            shared_aes = set(gene_ae_drugs[gene].keys()) & set(gene_ae_drugs[partner].keys())
            if len(shared_aes) >= 3:
                signals.append({
                    "signal_type": "PATHWAY_SHARED_AE",
                    "gene_a": gene,
                    "gene_b": partner,
                    "shared_adverse_events": len(shared_aes),
                    "top_shared_aes": sorted(shared_aes)[:5],
                    "relationship": "STRING interaction partners",
                })

    # Signal 3: Mechanism-of-action clusters with same AE
    # "Drugs with the same ChEMBL mechanism cause the same adverse event"
    mech_aes: Dict[str, Counter] = defaultdict(Counter)
    mech_drugs: Dict[str, Set[str]] = defaultdict(set)
    for rec in bridge:
        mech = rec["mechanism"]
        if mech:
            mech_aes[mech][rec["adverse_event"]] += 1
            mech_drugs[mech].add(rec["therapy"])

    for mech, ae_counter in mech_aes.items():
        drugs = mech_drugs[mech]
        if len(drugs) >= 2:
            top_ae = ae_counter.most_common(3)
            signals.append({
                "signal_type": "MECHANISM_AE_CLUSTER",
                "mechanism": mech,
                "drugs": sorted(drugs)[:5],
                "drug_count": len(drugs),
                "top_adverse_events": [(ae, c) for ae, c in top_ae],
            })

    # Signal 4: Trial phase vs AE severity
    # "Drug in Phase 3+ with high AE count — worth watching"
    seen = set()
    for rec in bridge:
        phases = rec["trial_phase"]
        count = int(rec["ae_report_count"]) if rec["ae_report_count"].isdigit() else 0
        if "PHASE3" in phases.upper().replace(" ", "") and count >= 30:
            key = (rec["therapy"], rec["adverse_event"])
            if key not in seen:
                seen.add(key)
                signals.append({
                    "signal_type": "PHASE3_HIGH_AE",
                    "therapy": rec["therapy"],
                    "disease": rec["disease"],
                    "trial_phase": phases,
                    "adverse_event": rec["adverse_event"],
                    "ae_count": count,
                    "gene": rec["molecular_profile"],
                })

    # Signal 5: Subcellular location patterns
    # "Do drugs targeting nuclear vs membrane proteins have different AE profiles?"
    location_aes: Dict[str, Counter] = defaultdict(Counter)
    for rec in bridge:
        loc = rec["protein_location"]
        if loc:
            location_aes[loc][rec["adverse_event"]] += 1

    for loc, ae_counter in location_aes.items():
        if sum(ae_counter.values()) >= 20:
            top = ae_counter.most_common(5)
            signals.append({
                "signal_type": "LOCATION_AE_PATTERN",
                "protein_location": loc,
                "total_bridge_records": sum(ae_counter.values()),
                "top_adverse_events": [(ae, c) for ae, c in top],
            })

    # Deduplicate pathway signals
    deduped = []
    pathway_seen = set()
    for s in signals:
        if s["signal_type"] == "PATHWAY_SHARED_AE":
            key = tuple(sorted([s["gene_a"], s["gene_b"]]))
            if key in pathway_seen:
                continue
            pathway_seen.add(key)
        deduped.append(s)

    return deduped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("  MEGA-BRIDGE: 6-Database Semantic Router")
    print("  FAERS x CIViC x ClinicalTrials x ChEMBL x UniProt x STRING + OpenFDA")
    print("=" * 72)

    # Step 1: Load local databases
    print("\n[1/6] Loading local databases...")
    t0 = time.time()
    civic_records = load_civic()
    faers_records = load_faers(max_records=500_000)
    print(f"  CIViC: {len(civic_records)} evidence records")
    print(f"  FAERS: {len(faers_records)} drug-reaction records")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Step 2: Find shared drugs and unique genes
    print("\n[2/6] Finding join surfaces...")
    civic_drugs: Set[str] = set()
    civic_by_drug: Dict[str, List] = defaultdict(list)
    unique_genes: Set[str] = set()

    for rec in civic_records:
        therapies_raw = rec.get("therapies", "")
        gene = extract_gene(rec["molecular_profile"])
        if gene:
            unique_genes.add(gene)
        if not therapies_raw:
            continue
        for therapy in therapies_raw.split(","):
            normalized = normalize_drug(therapy.strip())
            if normalized:
                civic_drugs.add(normalized)
                civic_by_drug[normalized].append(rec)

    faers_drugs: Set[str] = set()
    for rec in faers_records:
        drug = rec.get("active_ingredient") or rec.get("drugname", "")
        normalized = normalize_drug(drug)
        if normalized:
            faers_drugs.add(normalized)

    shared_drugs = sorted(civic_drugs & faers_drugs)
    genes_list = sorted(unique_genes)

    print(f"  CIViC therapies: {len(civic_drugs)}")
    print(f"  FAERS drugs: {len(faers_drugs)}")
    print(f"  Shared drugs: {len(shared_drugs)}")
    print(f"  Unique CIViC genes: {len(genes_list)}")
    print(f"  Total API calls needed: {len(shared_drugs) * 3 + len(genes_list) * 2}")

    # Step 3: Pull from external APIs
    print(f"\n[3/6] Pulling from 5 external APIs...")
    t_api = time.time()
    api_data = pull_all_apis(shared_drugs, genes_list)
    api_time = time.time() - t_api
    print(f"  API pulls complete in {api_time:.1f}s")

    # Count hits
    ct_hits = sum(1 for d in api_data["ctgov"].values() if d.get("trials", 0) > 0)
    ch_hits = sum(1 for d in api_data["chembl"].values() if d.get("mechanism"))
    up_hits = sum(1 for d in api_data["uniprot"].values() if d.get("protein_name"))
    st_hits = sum(1 for d in api_data["string"].values() if d.get("partner_count", 0) > 0)
    fda_hits = sum(1 for d in api_data["openfda"].values() if d.get("drug_class"))

    print(f"  ClinicalTrials.gov hits: {ct_hits}/{len(shared_drugs)} drugs")
    print(f"  ChEMBL mechanism hits:   {ch_hits}/{len(shared_drugs)} drugs")
    print(f"  UniProt protein hits:    {up_hits}/{len(genes_list)} genes")
    print(f"  STRING interaction hits: {st_hits}/{len(genes_list)} genes")
    print(f"  OpenFDA label hits:      {fda_hits}/{len(shared_drugs)} drugs")

    # Step 4: Build mega-bridge
    print(f"\n[4/6] Building mega-bridge records...")
    t_bridge = time.time()
    bridge, bridge_stats = build_mega_bridge(civic_records, faers_records, api_data)
    print(f"  Mega-bridge records: {len(bridge)}")
    print(f"  Built in {time.time() - t_bridge:.1f}s")

    if not bridge:
        print("  ERROR: No bridge records. Exiting.")
        return

    # Show field coverage
    filled_counts = defaultdict(int)
    for rec in bridge:
        for field, val in rec.items():
            if val and val not in ("", "0", "unknown", "False"):
                filled_counts[field] += 1

    print(f"\n  Field coverage across {len(bridge)} bridge records:")
    for field in ["mechanism", "trial_phase", "protein_function", "interaction_partners",
                  "drug_class", "boxed_warning", "protein_pathway", "protein_location"]:
        pct = filled_counts.get(field, 0) / len(bridge) * 100
        print(f"    {field}: {filled_counts.get(field, 0)} ({pct:.1f}%)")

    # Step 5: Route through DW
    print(f"\n[5/6] Building semantic router over mega-bridge...")
    t_route = time.time()
    router = SemanticRouter()
    router.ingest(
        records=bridge,
        identity_fields=["therapy", "molecular_profile", "disease"],
        provenance_fields=["bridge_id"],
        max_ladder_depth=5,
    )
    print(f"  Built in {time.time() - t_route:.2f}s")

    info = router.explain()
    print(f"\n  === Mega-Bridge Router Structure ===")
    print(f"  Total records: {info['total_records']}")
    print(f"  Identity: therapy + molecular_profile + disease")
    print(f"  Neighborhoods: {info['identity_neighborhoods']} total, {info['ambiguous_neighborhoods']} ambiguous")
    print(f"  Discovered ladder:")
    for rung in info["ladder"]:
        print(f"    Rung {rung['rung']}: {rung['field']} (reduction={rung['ambiguity_reduction_rate']})")
    print(f"  Full route: identity -> {' -> '.join(router.ladder_fields)}")

    # Demo queries
    print(f"\n  --- Sample cross-database queries ---")
    sample_queries = [r for r in bridge if r["mechanism"] and r["interaction_partners"]][:3]
    for sq in sample_queries:
        result = router.query(sq, ask_field="adverse_event")
        print(f"\n  Q: {sq['therapy']} / {sq['molecular_profile']} / {sq['disease']}")
        print(f"     Mechanism: {sq['mechanism']}")
        print(f"     Partners: {sq['interaction_partners']}")
        print(f"     -> AE: {result.answer}, examined {result.records_examined}/{result.total_records}")
        print(f"     Route: {result.route_used}")

    # Step 6: Signal detection
    print(f"\n[6/6] Cross-database signal detection...")
    signals = detect_mega_signals(bridge)

    by_type = defaultdict(list)
    for s in signals:
        by_type[s["signal_type"]].append(s)

    for stype, slist in by_type.items():
        print(f"\n  === {stype} ({len(slist)} signals) ===")

        if stype == "BOXED_WARNING_STRONG_EVIDENCE":
            seen = set()
            for s in slist[:8]:
                key = (s["therapy"], s["adverse_event"])
                if key in seen:
                    continue
                seen.add(key)
                print(f"    {s['therapy']} ({s['gene']}, Level {s['evidence_level']})")
                print(f"      AE: {s['adverse_event']} ({s['ae_count']} reports), "
                      f"class: {s['drug_class']}")

        elif stype == "PATHWAY_SHARED_AE":
            for s in slist[:8]:
                print(f"    {s['gene_a']} <-> {s['gene_b']}: {s['shared_adverse_events']} shared AEs")
                print(f"      Top: {', '.join(s['top_shared_aes'][:3])}")

        elif stype == "MECHANISM_AE_CLUSTER":
            for s in slist[:8]:
                top_ae_str = ", ".join(f"{ae}({c})" for ae, c in s["top_adverse_events"])
                print(f"    {s['mechanism']} ({s['drug_count']} drugs)")
                print(f"      Drugs: {', '.join(s['drugs'][:3])}")
                print(f"      Top AEs: {top_ae_str}")

        elif stype == "PHASE3_HIGH_AE":
            for s in slist[:8]:
                print(f"    {s['therapy']} (Phase 3): AE={s['adverse_event']} "
                      f"({s['ae_count']} reports)")
                print(f"      Gene: {s['gene']}, Disease: {s['disease']}")

        elif stype == "LOCATION_AE_PATTERN":
            for s in slist[:5]:
                top_ae_str = ", ".join(f"{ae}({c})" for ae, c in s["top_adverse_events"][:3])
                print(f"    {s['protein_location']} ({s['total_bridge_records']} records)")
                print(f"      Top AEs: {top_ae_str}")

    # Summary
    total_signals = len(signals)
    print(f"\n{'=' * 72}")
    print(f"  MEGA-BRIDGE SUMMARY")
    print(f"{'=' * 72}")
    print(f"  Databases joined:     6 (+ OpenFDA labels)")
    print(f"  Local:  CIViC ({len(civic_records)}) + FAERS ({len(faers_records)})")
    print(f"  Remote: ClinicalTrials ({ct_hits} hits) + ChEMBL ({ch_hits}) "
          f"+ UniProt ({up_hits}) + STRING ({st_hits}) + OpenFDA ({fda_hits})")
    print(f"  Shared drugs:         {len(shared_drugs)}")
    print(f"  Bridge records:       {len(bridge)}")
    print(f"  Bridge ladder:        identity -> {' -> '.join(router.ladder_fields)}")
    print(f"  Signals detected:     {total_signals}")
    for stype, slist in by_type.items():
        print(f"    {stype}: {len(slist)}")
    print(f"\n  Dimensions per record: {len(bridge[0])} fields from 6 databases")
    print(f"  API time:             {api_time:.1f}s")
    print(f"\n  What 6 databases reveal that none shows alone:")
    print(f"    - Boxed-warning drugs with strong mutation-level evidence + real AE counts")
    print(f"    - Protein interaction partners whose drugs cause the same adverse events")
    print(f"    - Mechanism-of-action clusters that predict adverse event profiles")
    print(f"    - Phase 3 trials with high real-world adverse event signals")
    print(f"    - Subcellular location patterns in adverse event distribution")


if __name__ == "__main__":
    main()
