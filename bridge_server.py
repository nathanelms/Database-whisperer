"""bridge_server.py

Web interface for the 6-database mega-bridge.
Opens in a browser. No command line needed.

Lets oncologists, pharmacovigilance teams, and researchers:
- Browse drugs, genes, diseases, adverse events
- Filter across all 6 databases at once
- See cross-database signals (things no single DB reveals)
- Ask natural language questions (optional, needs API key)

Usage:
    python bridge_server.py
    # Opens http://localhost:8050 in your browser

No dependencies beyond stdlib + the existing project files.
"""

from __future__ import annotations

import json
import os
import html
import time
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from collections import defaultdict, Counter
from typing import Any, Dict, List, Tuple

from multi_db_bridge import (
    load_civic,
    load_faers,
    normalize_drug,
    extract_gene,
    build_mega_bridge,
    pull_all_apis,
    detect_mega_signals,
)
from semantic_router import SemanticRouter


# ---------------------------------------------------------------------------
# Globals (loaded once at startup)
# ---------------------------------------------------------------------------

BRIDGE: List[Dict[str, Any]] = []
ROUTER: SemanticRouter = SemanticRouter()
SIGNALS: List[Dict[str, Any]] = []
STATS: Dict[str, Any] = {}

# Precomputed indexes for fast filtering
DRUGS: List[str] = []
GENES: List[str] = []
DISEASES: List[str] = []
AES: List[str] = []
MECHANISMS: List[str] = []


def startup():
    """Load everything and precompute indexes."""
    global BRIDGE, ROUTER, SIGNALS, STATS
    global DRUGS, GENES, DISEASES, AES, MECHANISMS

    print("Loading databases...")
    civic = load_civic()
    faers = load_faers(max_records=500_000)
    print(f"  CIViC: {len(civic)} | FAERS: {len(faers)}")

    # Find shared drugs and genes
    civic_drugs = set()
    unique_genes = set()
    for rec in civic:
        gene = extract_gene(rec["molecular_profile"])
        if gene:
            unique_genes.add(gene)
        for therapy in rec.get("therapies", "").split(","):
            n = normalize_drug(therapy.strip())
            if n:
                civic_drugs.add(n)

    faers_drugs = set()
    for rec in faers:
        drug = rec.get("active_ingredient") or rec.get("drugname", "")
        n = normalize_drug(drug)
        if n:
            faers_drugs.add(n)

    shared = sorted(civic_drugs & faers_drugs)
    genes_list = sorted(unique_genes)

    print("Loading cached API data...")
    api_data = pull_all_apis(shared, genes_list)

    print("Building mega-bridge...")
    BRIDGE, STATS = build_mega_bridge(civic, faers, api_data)
    print(f"  Bridge: {len(BRIDGE)} records")

    print("Building router...")
    ROUTER.ingest(
        records=BRIDGE,
        identity_fields=["therapy", "molecular_profile", "disease"],
        provenance_fields=["bridge_id"],
        max_ladder_depth=5,
    )
    info = ROUTER.explain()
    print(f"  Ladder: identity -> {' -> '.join(ROUTER.ladder_fields)}")

    print("Detecting signals...")
    SIGNALS = detect_mega_signals(BRIDGE)
    print(f"  Signals: {len(SIGNALS)}")

    # Precompute filter options
    drug_set = set()
    gene_set = set()
    disease_set = set()
    ae_set = set()
    mech_set = set()
    for rec in BRIDGE:
        drug_set.add(rec["therapy"])
        gene_set.add(rec["gene"])
        disease_set.add(rec["disease"])
        ae_set.add(rec["adverse_event"])
        if rec.get("mechanism"):
            mech_set.add(rec["mechanism"])

    DRUGS = sorted(drug_set)
    GENES = sorted(gene_set)
    DISEASES = sorted(disease_set)
    AES = sorted(ae_set)
    MECHANISMS = sorted(mech_set)

    print(f"\nReady: {len(DRUGS)} drugs, {len(GENES)} genes, {len(DISEASES)} diseases, {len(AES)} adverse events")


# ---------------------------------------------------------------------------
# Filter engine
# ---------------------------------------------------------------------------

def filter_bridge(
    therapy: str = "",
    gene: str = "",
    disease: str = "",
    adverse_event: str = "",
    mechanism: str = "",
    boxed_only: bool = False,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """Filter bridge records by any combination of fields."""
    results = []
    for rec in BRIDGE:
        if therapy and therapy.upper() not in rec["therapy"].upper():
            continue
        if gene and gene.upper() not in rec["molecular_profile"].upper():
            continue
        if disease and disease.upper() not in rec["disease"].upper():
            continue
        if adverse_event and adverse_event.upper() not in rec["adverse_event"].upper():
            continue
        if mechanism and mechanism.upper() not in rec.get("mechanism", "").upper():
            continue
        if boxed_only and rec.get("boxed_warning") != "True":
            continue
        results.append(rec)
        if len(results) >= limit:
            break
    return results


def get_drug_profile(drug: str) -> Dict[str, Any]:
    """Get a comprehensive profile for a single drug across all databases."""
    matches = [r for r in BRIDGE if drug.upper() in r["therapy"].upper()]
    if not matches:
        return {"drug": drug, "found": False}

    ae_counts = Counter(r["adverse_event"] for r in matches)
    diseases = Counter(r["disease"] for r in matches)
    genes = Counter(r["molecular_profile"] for r in matches)

    sample = matches[0]
    return {
        "drug": drug,
        "found": True,
        "total_bridge_records": len(matches),
        "mechanism": sample.get("mechanism", "unknown"),
        "drug_class": sample.get("drug_class", "unknown"),
        "boxed_warning": sample.get("boxed_warning", "False"),
        "trial_phase": sample.get("trial_phase", "unknown"),
        "top_adverse_events": ae_counts.most_common(15),
        "diseases": diseases.most_common(10),
        "gene_targets": genes.most_common(10),
        "interaction_partners": sample.get("interaction_partners", ""),
        "protein_location": sample.get("protein_location", ""),
        "sex_ratio": sample.get("sex_ratio", ""),
        "total_faers_reports": sample.get("total_faers_reports", "0"),
    }


def get_gene_profile(gene: str) -> Dict[str, Any]:
    """Get a comprehensive profile for a gene across all databases."""
    matches = [r for r in BRIDGE if gene.upper() in r["gene"].upper()
               or gene.upper() in r["molecular_profile"].upper()]
    if not matches:
        return {"gene": gene, "found": False}

    ae_counts = Counter(r["adverse_event"] for r in matches)
    drugs = Counter(r["therapy"] for r in matches)
    diseases = Counter(r["disease"] for r in matches)

    sample = matches[0]
    return {
        "gene": gene,
        "found": True,
        "total_bridge_records": len(matches),
        "protein_function": sample.get("protein_function", ""),
        "protein_pathway": sample.get("protein_pathway", ""),
        "protein_location": sample.get("protein_location", ""),
        "interaction_partners": sample.get("interaction_partners", ""),
        "partner_count": sample.get("partner_count", "0"),
        "top_adverse_events": ae_counts.most_common(15),
        "therapies": drugs.most_common(10),
        "diseases": diseases.most_common(10),
    }


# ---------------------------------------------------------------------------
# HTML templates
# ---------------------------------------------------------------------------

def esc(s: str) -> str:
    return html.escape(str(s))


def page_header(title: str, active: str = "") -> str:
    nav_items = [
        ("Browse", "/"),
        ("Drug Profile", "/drug"),
        ("Gene Profile", "/gene"),
        ("Signals", "/signals"),
        ("Stats", "/stats"),
    ]
    nav_html = ""
    for label, href in nav_items:
        cls = "active" if active == label else ""
        nav_html += f'<a href="{href}" class="{cls}">{label}</a> '

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>{esc(title)} — Database Whisper</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0a0a0f; color: #e0e0e0; }}
.nav {{ background: #12121a; padding: 12px 24px; border-bottom: 1px solid #2a2a3a; display: flex; align-items: center; gap: 20px; }}
.nav .logo {{ font-size: 18px; font-weight: 700; color: #7c8cf5; letter-spacing: -0.5px; }}
.nav a {{ color: #888; text-decoration: none; font-size: 14px; padding: 4px 12px; border-radius: 6px; }}
.nav a:hover {{ color: #fff; background: #1a1a2e; }}
.nav a.active {{ color: #7c8cf5; background: #1a1a2e; }}
.container {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
h1 {{ font-size: 24px; font-weight: 600; margin-bottom: 8px; color: #fff; }}
h2 {{ font-size: 18px; font-weight: 600; margin: 20px 0 12px; color: #c0c0d0; }}
.subtitle {{ color: #666; font-size: 14px; margin-bottom: 24px; }}
.filters {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 24px; }}
.filters input, .filters select {{ background: #16162a; border: 1px solid #2a2a3a; color: #e0e0e0; padding: 8px 14px; border-radius: 8px; font-size: 14px; min-width: 180px; }}
.filters input:focus, .filters select:focus {{ outline: none; border-color: #7c8cf5; }}
.filters button {{ background: #7c8cf5; color: #fff; border: none; padding: 8px 20px; border-radius: 8px; font-size: 14px; cursor: pointer; font-weight: 600; }}
.filters button:hover {{ background: #6b7be0; }}
.filters label {{ display: flex; align-items: center; gap: 6px; color: #888; font-size: 14px; }}
.filters label input[type=checkbox] {{ accent-color: #7c8cf5; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{ text-align: left; padding: 10px 12px; background: #12121a; color: #7c8cf5; font-weight: 600; border-bottom: 2px solid #2a2a3a; position: sticky; top: 0; }}
td {{ padding: 8px 12px; border-bottom: 1px solid #1a1a2a; }}
tr:hover td {{ background: #12121a; }}
.tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; margin: 1px 2px; }}
.tag-a {{ background: #1a3a1a; color: #4caf50; }}
.tag-b {{ background: #2a3a1a; color: #8bc34a; }}
.tag-c {{ background: #3a3a1a; color: #ffc107; }}
.tag-d {{ background: #3a2a1a; color: #ff9800; }}
.tag-boxed {{ background: #3a1a1a; color: #f44336; }}
.tag-mechanism {{ background: #1a1a3a; color: #64b5f6; }}
.card {{ background: #12121a; border: 1px solid #2a2a3a; border-radius: 12px; padding: 20px; margin-bottom: 16px; }}
.card-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px; }}
.stat-num {{ font-size: 32px; font-weight: 700; color: #7c8cf5; }}
.stat-label {{ font-size: 13px; color: #666; margin-top: 4px; }}
.bar {{ height: 8px; border-radius: 4px; background: #1a1a2a; margin-top: 6px; }}
.bar-fill {{ height: 100%; border-radius: 4px; background: #7c8cf5; }}
.badge {{ display: inline-block; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; }}
.badge-pathway {{ background: #1a2a3a; color: #4fc3f7; }}
.badge-mechanism {{ background: #2a1a3a; color: #ba68c8; }}
.badge-boxed {{ background: #3a1a1a; color: #ef5350; }}
.badge-location {{ background: #1a3a2a; color: #66bb6a; }}
.badge-phase3 {{ background: #3a3a1a; color: #ffca28; }}
a.drug-link, a.gene-link {{ color: #7c8cf5; text-decoration: none; }}
a.drug-link:hover, a.gene-link:hover {{ text-decoration: underline; }}
.profile-header {{ display: flex; gap: 20px; align-items: flex-start; margin-bottom: 24px; }}
.profile-title {{ font-size: 28px; font-weight: 700; color: #fff; }}
.profile-meta {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 8px; }}
.ae-bar {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; }}
.ae-bar .ae-name {{ min-width: 250px; font-size: 13px; }}
.ae-bar .ae-fill {{ height: 20px; border-radius: 4px; background: #7c8cf5; min-width: 2px; }}
.ae-bar .ae-count {{ font-size: 12px; color: #888; min-width: 40px; }}
.empty {{ text-align: center; padding: 60px; color: #444; font-size: 16px; }}
footer {{ text-align: center; padding: 40px; color: #333; font-size: 12px; }}
</style>
</head><body>
<div class="nav">
    <span class="logo">Database Whisper</span>
    {nav_html}
</div>
<div class="container">
"""


PAGE_FOOTER = """
</div>
<footer>Database Whisper — 6-database semantic routing engine</footer>
</body></html>"""


# ---------------------------------------------------------------------------
# Page renderers
# ---------------------------------------------------------------------------

def render_browse(params: Dict[str, List[str]]) -> str:
    therapy = (params.get("therapy", [""])[0]).strip()
    gene = (params.get("gene", [""])[0]).strip()
    disease = (params.get("disease", [""])[0]).strip()
    ae = (params.get("ae", [""])[0]).strip()
    mechanism = (params.get("mechanism", [""])[0]).strip()
    boxed = "boxed" in params

    results = filter_bridge(
        therapy=therapy, gene=gene, disease=disease,
        adverse_event=ae, mechanism=mechanism, boxed_only=boxed,
    )

    body = page_header("Browse", "Browse")
    body += f"<h1>Browse the Bridge</h1>"
    body += f'<p class="subtitle">{len(BRIDGE):,} records from 6 databases. Filter by any field.</p>'

    body += f"""<form class="filters" method="get" action="/">
    <input name="therapy" placeholder="Drug name" value="{esc(therapy)}">
    <input name="gene" placeholder="Gene / mutation" value="{esc(gene)}">
    <input name="disease" placeholder="Disease" value="{esc(disease)}">
    <input name="ae" placeholder="Adverse event" value="{esc(ae)}">
    <input name="mechanism" placeholder="Mechanism" value="{esc(mechanism)}">
    <label><input type="checkbox" name="boxed" {'checked' if boxed else ''}> Boxed warning only</label>
    <button type="submit">Filter</button>
</form>"""

    if not therapy and not gene and not disease and not ae and not mechanism and not boxed:
        # Landing page — show overview instead of full table
        body += render_landing()
    else:
        body += f"<p style='color:#888; margin-bottom:12px;'>{len(results)} results (max 200 shown)</p>"
        body += "<div style='overflow-x:auto;'><table>"
        body += "<tr><th>Drug</th><th>Gene</th><th>Disease</th><th>Adverse Event</th><th>AE Count</th><th>Evidence</th><th>Mechanism</th><th>Boxed</th><th>Partners</th></tr>"
        for rec in results:
            ev = rec["evidence_level"]
            ev_cls = f"tag-{ev.lower()}" if ev in ("A", "B", "C", "D") else ""
            boxed_tag = '<span class="tag tag-boxed">BOXED</span>' if rec.get("boxed_warning") == "True" else ""
            partners = rec.get("interaction_partners", "")[:40]
            body += f"""<tr>
            <td><a class="drug-link" href="/drug?name={esc(rec['therapy'])}">{esc(rec['therapy'])}</a></td>
            <td><a class="gene-link" href="/gene?name={esc(rec['gene'])}">{esc(rec['molecular_profile'][:40])}</a></td>
            <td>{esc(rec['disease'][:35])}</td>
            <td>{esc(rec['adverse_event'])}</td>
            <td>{esc(rec['ae_report_count'])}</td>
            <td><span class="tag {ev_cls}">{esc(ev)}</span> {esc(rec['evidence_direction'][:3])}</td>
            <td><span class="tag tag-mechanism">{esc(rec.get('mechanism', '')[:30])}</span></td>
            <td>{boxed_tag}</td>
            <td style="font-size:11px; color:#666;">{esc(partners)}</td>
            </tr>"""
        body += "</table></div>"

    body += PAGE_FOOTER
    return body


def render_landing() -> str:
    """Overview cards for the landing page."""
    info = ROUTER.explain()

    by_type = defaultdict(list)
    for s in SIGNALS:
        by_type[s["signal_type"]].append(s)

    body = '<div class="card-grid">'

    # Stats cards
    body += f"""<div class="card">
        <div class="stat-num">{len(BRIDGE):,}</div>
        <div class="stat-label">Bridge records from 6 databases</div>
    </div>"""
    body += f"""<div class="card">
        <div class="stat-num">{len(DRUGS)}</div>
        <div class="stat-label">Drugs (shared between FAERS + CIViC)</div>
    </div>"""
    body += f"""<div class="card">
        <div class="stat-num">{len(GENES)}</div>
        <div class="stat-label">Gene targets (from CIViC)</div>
    </div>"""
    body += f"""<div class="card">
        <div class="stat-num">{len(SIGNALS)}</div>
        <div class="stat-label">Cross-database signals detected</div>
    </div>"""

    body += '</div>'

    # Top drugs
    body += '<h2>Top Drugs by FAERS Reports</h2>'
    drug_counts = Counter()
    for rec in BRIDGE:
        drug_counts[rec["therapy"]] += int(rec.get("ae_report_count", "0") if rec.get("ae_report_count", "0").isdigit() else "0")
    body += '<div class="card"><table><tr><th>Drug</th><th>Total AE Reports</th><th>Mechanism</th></tr>'
    for drug, count in drug_counts.most_common(15):
        sample = next((r for r in BRIDGE if r["therapy"] == drug and r.get("mechanism")), None)
        mech = sample.get("mechanism", "") if sample else ""
        body += f'<tr><td><a class="drug-link" href="/drug?name={esc(drug)}">{esc(drug)}</a></td><td>{count:,}</td><td>{esc(mech[:50])}</td></tr>'
    body += '</table></div>'

    # Top signals
    body += '<h2>Notable Signals</h2>'
    body += '<div class="card-grid">'

    pathway_signals = by_type.get("PATHWAY_SHARED_AE", [])
    if pathway_signals:
        body += '<div class="card"><h2><span class="badge badge-pathway">PATHWAY</span> Shared Adverse Events</h2>'
        for s in pathway_signals[:5]:
            body += f'<p style="margin:6px 0;"><a class="gene-link" href="/gene?name={esc(s["gene_a"])}">{esc(s["gene_a"])}</a> ↔ <a class="gene-link" href="/gene?name={esc(s["gene_b"])}">{esc(s["gene_b"])}</a>: {s["shared_adverse_events"]} shared AEs</p>'
        body += '</div>'

    mech_signals = by_type.get("MECHANISM_AE_CLUSTER", [])
    if mech_signals:
        body += '<div class="card"><h2><span class="badge badge-mechanism">MECHANISM</span> AE Clusters</h2>'
        for s in mech_signals[:5]:
            drugs = ", ".join(s["drugs"][:3])
            body += f'<p style="margin:6px 0;">{esc(s["mechanism"][:40])} ({s["drug_count"]} drugs: {esc(drugs)})</p>'
        body += '</div>'

    body += '</div>'
    return body


def render_drug_profile(params: Dict[str, List[str]]) -> str:
    name = (params.get("name", [""])[0]).strip()
    body = page_header(f"Drug: {name}", "Drug Profile")

    if not name:
        body += "<h1>Drug Profile</h1>"
        body += f'<form class="filters" method="get" action="/drug"><input name="name" placeholder="Enter drug name (e.g. VEMURAFENIB)" style="min-width:300px;"><button type="submit">Look up</button></form>'
        body += '<h2>All Drugs in Bridge</h2><div class="card"><p>'
        for d in DRUGS:
            body += f'<a class="drug-link" href="/drug?name={esc(d)}" style="margin:4px; display:inline-block;">{esc(d)}</a> '
        body += '</p></div>'
        body += PAGE_FOOTER
        return body

    profile = get_drug_profile(name)

    if not profile["found"]:
        body += f'<h1>Drug not found: {esc(name)}</h1><p class="subtitle">Try a different name.</p>'
        body += PAGE_FOOTER
        return body

    # Header
    boxed_badge = ' <span class="badge badge-boxed">BOXED WARNING</span>' if profile["boxed_warning"] == "True" else ""
    body += f"""<div class="profile-header"><div>
        <div class="profile-title">{esc(profile['drug'])}{boxed_badge}</div>
        <div class="profile-meta">
            <span class="badge badge-mechanism">{esc(profile['mechanism'][:60])}</span>
            <span class="badge" style="background:#1a2a1a; color:#81c784;">{esc(profile['drug_class'][:40])}</span>
            <span class="badge" style="background:#1a1a2a; color:#90a4ae;">{profile['total_bridge_records']} bridge records</span>
            <span class="badge" style="background:#2a2a1a; color:#ffd54f;">{profile['total_faers_reports']} FAERS reports</span>
        </div></div></div>"""

    body += '<div class="card-grid">'

    # AE chart
    body += '<div class="card"><h2>Top Adverse Events (FAERS)</h2>'
    max_count = profile["top_adverse_events"][0][1] if profile["top_adverse_events"] else 1
    for ae_name, count in profile["top_adverse_events"]:
        pct = count / max_count * 100
        body += f"""<div class="ae-bar">
            <span class="ae-name">{esc(ae_name)}</span>
            <div class="ae-fill" style="width:{pct}%;"></div>
            <span class="ae-count">{count}</span>
        </div>"""
    body += '</div>'

    # Diseases
    body += '<div class="card"><h2>Cancer Indications (CIViC)</h2>'
    for disease, count in profile["diseases"]:
        body += f'<p style="margin:4px 0;">{esc(disease)} <span style="color:#888;">({count})</span></p>'
    body += '</div>'

    # Gene targets
    body += '<div class="card"><h2>Gene Targets (CIViC)</h2>'
    for gene, count in profile["gene_targets"]:
        body += f'<p style="margin:4px 0;"><a class="gene-link" href="/gene?name={esc(gene.split()[0])}">{esc(gene)}</a> <span style="color:#888;">({count})</span></p>'
    body += '</div>'

    # Protein info
    if profile.get("interaction_partners"):
        body += f'<div class="card"><h2>Protein Interactions (STRING)</h2><p>{esc(profile["interaction_partners"])}</p>'
        if profile.get("protein_location"):
            body += f'<p style="margin-top:8px; color:#888;">Location: {esc(profile["protein_location"])}</p>'
        body += '</div>'

    body += '</div>'
    body += PAGE_FOOTER
    return body


def render_gene_profile(params: Dict[str, List[str]]) -> str:
    name = (params.get("name", [""])[0]).strip()
    body = page_header(f"Gene: {name}", "Gene Profile")

    if not name:
        body += "<h1>Gene Profile</h1>"
        body += f'<form class="filters" method="get" action="/gene"><input name="name" placeholder="Enter gene name (e.g. BRAF)" style="min-width:300px;"><button type="submit">Look up</button></form>'
        body += '<h2>All Genes in Bridge</h2><div class="card"><p>'
        for g in GENES:
            body += f'<a class="gene-link" href="/gene?name={esc(g)}" style="margin:4px; display:inline-block;">{esc(g)}</a> '
        body += '</p></div>'
        body += PAGE_FOOTER
        return body

    profile = get_gene_profile(name)

    if not profile["found"]:
        body += f'<h1>Gene not found: {esc(name)}</h1>'
        body += PAGE_FOOTER
        return body

    body += f"""<div class="profile-header"><div>
        <div class="profile-title">{esc(profile['gene'])}</div>
        <div class="profile-meta">
            <span class="badge" style="background:#1a2a1a; color:#81c784;">{esc(profile.get('protein_location',''))}</span>
            <span class="badge" style="background:#1a1a2a; color:#90a4ae;">{profile['total_bridge_records']} bridge records</span>
            <span class="badge badge-pathway">{profile['partner_count']} interaction partners</span>
        </div></div></div>"""

    if profile.get("protein_function"):
        body += f'<div class="card"><h2>Protein Function (UniProt)</h2><p>{esc(profile["protein_function"])}</p></div>'

    if profile.get("protein_pathway"):
        body += f'<div class="card"><h2>Pathway</h2><p>{esc(profile["protein_pathway"])}</p></div>'

    if profile.get("interaction_partners"):
        body += f'<div class="card"><h2>Interaction Partners (STRING)</h2><p>'
        for partner in profile["interaction_partners"].split(", "):
            body += f'<a class="gene-link" href="/gene?name={esc(partner.strip())}" style="margin:4px; display:inline-block;">{esc(partner.strip())}</a> '
        body += '</p></div>'

    body += '<div class="card-grid">'

    # AEs
    body += '<div class="card"><h2>Top Adverse Events</h2>'
    max_count = profile["top_adverse_events"][0][1] if profile["top_adverse_events"] else 1
    for ae_name, count in profile["top_adverse_events"]:
        pct = count / max_count * 100
        body += f'<div class="ae-bar"><span class="ae-name">{esc(ae_name)}</span><div class="ae-fill" style="width:{pct}%;"></div><span class="ae-count">{count}</span></div>'
    body += '</div>'

    # Therapies
    body += '<div class="card"><h2>Therapies</h2>'
    for drug, count in profile["therapies"]:
        body += f'<p style="margin:4px 0;"><a class="drug-link" href="/drug?name={esc(drug)}">{esc(drug)}</a> <span style="color:#888;">({count})</span></p>'
    body += '</div>'

    # Diseases
    body += '<div class="card"><h2>Diseases</h2>'
    for disease, count in profile["diseases"]:
        body += f'<p style="margin:4px 0;">{esc(disease)} <span style="color:#888;">({count})</span></p>'
    body += '</div>'

    body += '</div>'
    body += PAGE_FOOTER
    return body


def render_signals(params: Dict[str, List[str]]) -> str:
    signal_type = (params.get("type", [""])[0]).strip()

    body = page_header("Signals", "Signals")
    body += "<h1>Cross-Database Signals</h1>"
    body += '<p class="subtitle">Connections that no single database reveals on its own.</p>'

    by_type = defaultdict(list)
    for s in SIGNALS:
        by_type[s["signal_type"]].append(s)

    # Type filter
    body += '<div class="filters">'
    body += '<a href="/signals"><button style="background:#333;">All</button></a>'
    for stype in sorted(by_type.keys()):
        active = 'background:#7c8cf5;' if stype == signal_type else 'background:#1a1a2a; color:#888;'
        body += f'<a href="/signals?type={esc(stype)}"><button style="{active}">{esc(stype)} ({len(by_type[stype])})</button></a>'
    body += '</div>'

    types_to_show = [signal_type] if signal_type else sorted(by_type.keys())

    for stype in types_to_show:
        slist = by_type.get(stype, [])
        badge_cls = {
            "PATHWAY_SHARED_AE": "badge-pathway",
            "MECHANISM_AE_CLUSTER": "badge-mechanism",
            "BOXED_WARNING_STRONG_EVIDENCE": "badge-boxed",
            "LOCATION_AE_PATTERN": "badge-location",
            "PHASE3_HIGH_AE": "badge-phase3",
        }.get(stype, "")

        body += f'<h2><span class="badge {badge_cls}">{esc(stype)}</span> — {len(slist)} signals</h2>'

        if stype == "PATHWAY_SHARED_AE":
            body += '<table><tr><th>Gene A</th><th>Gene B</th><th>Shared AEs</th><th>Top Shared AEs</th></tr>'
            for s in slist[:30]:
                aes = ", ".join(s.get("top_shared_aes", [])[:3])
                body += f'<tr><td><a class="gene-link" href="/gene?name={esc(s["gene_a"])}">{esc(s["gene_a"])}</a></td><td><a class="gene-link" href="/gene?name={esc(s["gene_b"])}">{esc(s["gene_b"])}</a></td><td>{s["shared_adverse_events"]}</td><td>{esc(aes)}</td></tr>'
            body += '</table>'

        elif stype == "MECHANISM_AE_CLUSTER":
            body += '<table><tr><th>Mechanism</th><th>Drugs</th><th>Top AEs</th></tr>'
            for s in slist[:30]:
                drugs = ", ".join(s.get("drugs", [])[:4])
                aes = ", ".join(f"{ae}({c})" for ae, c in s.get("top_adverse_events", [])[:3])
                body += f'<tr><td>{esc(s["mechanism"][:50])}</td><td>{esc(drugs)}</td><td>{esc(aes)}</td></tr>'
            body += '</table>'

        elif stype == "BOXED_WARNING_STRONG_EVIDENCE":
            body += '<table><tr><th>Drug</th><th>Gene</th><th>Disease</th><th>Adverse Event</th><th>AE Count</th><th>Evidence</th></tr>'
            seen = set()
            for s in slist[:50]:
                key = (s["therapy"], s["adverse_event"])
                if key in seen:
                    continue
                seen.add(key)
                body += f'<tr><td><a class="drug-link" href="/drug?name={esc(s["therapy"])}">{esc(s["therapy"])}</a></td><td>{esc(s["gene"][:30])}</td><td>{esc(s["disease"][:30])}</td><td>{esc(s["adverse_event"])}</td><td>{s["ae_count"]}</td><td>{esc(s["evidence_level"])}</td></tr>'
            body += '</table>'

        elif stype == "LOCATION_AE_PATTERN":
            body += '<table><tr><th>Protein Location</th><th>Bridge Records</th><th>Top AEs</th></tr>'
            for s in slist[:20]:
                aes = ", ".join(f"{ae}({c})" for ae, c in s.get("top_adverse_events", [])[:3])
                body += f'<tr><td>{esc(s["protein_location"])}</td><td>{s["total_bridge_records"]}</td><td>{esc(aes)}</td></tr>'
            body += '</table>'

        elif stype == "PHASE3_HIGH_AE":
            body += '<table><tr><th>Drug</th><th>Disease</th><th>Gene</th><th>Adverse Event</th><th>AE Count</th></tr>'
            for s in slist[:30]:
                body += f'<tr><td><a class="drug-link" href="/drug?name={esc(s["therapy"])}">{esc(s["therapy"])}</a></td><td>{esc(s["disease"][:30])}</td><td>{esc(s["gene"][:30])}</td><td>{esc(s["adverse_event"])}</td><td>{s["ae_count"]}</td></tr>'
            body += '</table>'

    body += PAGE_FOOTER
    return body


def render_stats_page() -> str:
    body = page_header("Stats", "Stats")
    body += "<h1>Bridge Statistics</h1>"

    info = ROUTER.explain()

    body += '<div class="card-grid">'
    body += f'<div class="card"><div class="stat-num">6</div><div class="stat-label">Databases joined</div><p style="margin-top:8px; font-size:13px; color:#888;">CIViC + FAERS + ChEMBL + UniProt + STRING + OpenFDA</p></div>'
    body += f'<div class="card"><div class="stat-num">25</div><div class="stat-label">Fields per record</div></div>'
    body += f'<div class="card"><div class="stat-num">{len(BRIDGE):,}</div><div class="stat-label">Bridge records</div></div>'
    body += f'<div class="card"><div class="stat-num">{len(SIGNALS)}</div><div class="stat-label">Signals detected</div></div>'
    body += '</div>'

    body += '<h2>Discovered Routing Ladder</h2><div class="card">'
    body += '<p style="font-size:16px; color:#7c8cf5;">identity'
    for rung in info["ladder"]:
        body += f' → {rung["field"]} <span style="color:#444;">({rung["ambiguity_reduction_rate"]})</span>'
    body += '</p></div>'

    body += '<h2>Database Coverage</h2><div class="card">'
    filled = defaultdict(int)
    for rec in BRIDGE:
        for f, v in rec.items():
            if v and v not in ("", "0", "unknown", "False"):
                filled[f] += 1
    fields_to_show = [
        ("therapy", "FAERS+CIViC"), ("molecular_profile", "CIViC"), ("disease", "CIViC"),
        ("adverse_event", "FAERS"), ("mechanism", "ChEMBL"), ("drug_class", "OpenFDA"),
        ("protein_function", "UniProt"), ("interaction_partners", "STRING"),
        ("boxed_warning", "OpenFDA"), ("protein_location", "UniProt"),
    ]
    for field, source in fields_to_show:
        count = filled.get(field, 0)
        pct = count / len(BRIDGE) * 100
        body += f"""<div style="margin:8px 0;">
            <span style="min-width:200px; display:inline-block;">{esc(field)} <span style="color:#444;">({source})</span></span>
            <span style="color:#888; min-width:80px; display:inline-block;">{pct:.0f}%</span>
            <div class="bar" style="display:inline-block; width:300px; vertical-align:middle;">
                <div class="bar-fill" style="width:{pct}%;"></div>
            </div>
        </div>"""
    body += '</div>'

    body += PAGE_FOOTER
    return body


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class BridgeHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path == "/" or parsed.path == "":
            content = render_browse(params)
        elif parsed.path == "/drug":
            content = render_drug_profile(params)
        elif parsed.path == "/gene":
            content = render_gene_profile(params)
        elif parsed.path == "/signals":
            content = render_signals(params)
        elif parsed.path == "/stats":
            content = render_stats_page()
        elif parsed.path == "/api/filter":
            results = filter_bridge(
                therapy=params.get("therapy", [""])[0],
                gene=params.get("gene", [""])[0],
                disease=params.get("disease", [""])[0],
                adverse_event=params.get("ae", [""])[0],
                limit=50,
            )
            content = json.dumps(results, indent=2)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(content.encode("utf-8"))
            return
        else:
            content = page_header("404") + "<h1>Page not found</h1>" + PAGE_FOOTER

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(content.encode("utf-8"))

    def log_message(self, format, *args):
        # Suppress default HTTP logging
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    port = 8050

    startup()

    server = HTTPServer(("", port), BridgeHandler)
    print(f"\nServer running at http://localhost:{port}")
    print(f"Opening in browser...")

    # Open browser after a short delay
    threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
