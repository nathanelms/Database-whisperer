"""whisper_mega.py

Ask natural language questions across 6 databases at once.

Loads the FAERS x CIViC mega-bridge (with ChEMBL, UniProt, STRING, OpenFDA
data cached from prior API pulls), routes through the semantic router,
and uses Claude to translate your question and synthesize the answer.

This is whisper_rag.py evolved: instead of one CSV, it talks to the
entire cross-database bridge.

Usage:
    python whisper_mega.py

Example questions:
    "What adverse events do BRAF V600E patients get from vemurafenib?"
    "Which drugs targeting EGFR have boxed warnings?"
    "What's the mechanism of action for drugs treating melanoma?"
    "Do KRAS and BRAF drugs share the same side effects?"
    "What protein pathways are linked to diarrhoea in cancer drugs?"

Requires:
    pip install anthropic
    ANTHROPIC_API_KEY environment variable set
    Prior run of multi_db_bridge.py (to cache API data)
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, List, Tuple

import anthropic

from semantic_router import SemanticRouter
from multi_db_bridge import (
    load_civic,
    load_faers,
    normalize_drug,
    extract_gene,
    build_mega_bridge,
    pull_all_apis,
    detect_mega_signals,
)
from collections import defaultdict, Counter


# ---------------------------------------------------------------------------
# Build the bridge (reuses cached API data from prior runs)
# ---------------------------------------------------------------------------

def load_bridge() -> Tuple[List[Dict[str, Any]], SemanticRouter, Dict[str, Any]]:
    """Load both databases, build bridge, build router. Returns (bridge, router, stats)."""
    print("Loading CIViC + FAERS...")
    civic = load_civic()
    faers = load_faers(max_records=500_000)
    print(f"  CIViC: {len(civic)} | FAERS: {len(faers)}")

    # Find shared drugs and genes for API pulls
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
    genes = sorted(unique_genes)

    print(f"Pulling API data (cached from prior run)...")
    api_data = pull_all_apis(shared, genes)

    print("Building mega-bridge...")
    bridge, stats = build_mega_bridge(civic, faers, api_data)
    print(f"  Bridge records: {len(bridge)}")

    print("Building semantic router...")
    router = SemanticRouter()
    router.ingest(
        records=bridge,
        identity_fields=["therapy", "molecular_profile", "disease"],
        provenance_fields=["bridge_id"],
        max_ladder_depth=5,
    )

    info = router.explain()
    print(f"  Ladder: identity -> {' -> '.join(router.ladder_fields)}")
    print(f"  Neighborhoods: {info['identity_neighborhoods']} ({info['ambiguous_neighborhoods']} ambiguous)")

    return bridge, router, stats


# ---------------------------------------------------------------------------
# Claude-powered question translation (adapted for mega-bridge schema)
# ---------------------------------------------------------------------------

BRIDGE_FIELDS = [
    "therapy", "molecular_profile", "gene", "disease",
    "evidence_type", "evidence_direction", "evidence_level", "significance",
    "adverse_event", "ae_report_count", "total_faers_reports", "sex_ratio",
    "trial_phase", "trial_count",
    "mechanism", "target_type", "chembl_max_phase",
    "protein_function", "protein_pathway", "protein_location",
    "interaction_partners", "partner_count",
    "boxed_warning", "drug_class",
]

SCHEMA_DESCRIPTION = """This database bridges 6 sources:
- CIViC: clinical evidence linking gene mutations to cancer therapies
- FAERS: FDA adverse event reports (real-world drug side effects)
- ChEMBL: drug mechanisms of action and molecular targets
- UniProt: protein functions, pathways, and subcellular locations
- STRING: protein-protein interaction networks
- OpenFDA: drug labels, boxed warnings, pharmacological classes

Each record connects a therapy (drug) to a molecular_profile (gene/variant),
disease, adverse_event, and enrichment from all 6 databases."""


def translate_question(
    client: anthropic.Anthropic,
    question: str,
    sample_records: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], str, str]:
    """
    Use Claude to translate a natural language question into:
    1. query_fields: dict of field values to route on
    2. ask_field: the field to retrieve
    3. mode: "route" for single-record lookup, "scan" for aggregate/count questions
    """
    samples_text = ""
    for i, rec in enumerate(sample_records[:3]):
        samples_text += f"\nSample {i+1}:\n"
        for k, v in rec.items():
            if k != "bridge_id" and v:
                samples_text += f"  {k}: {v}\n"

    prompt = f"""You translate natural language questions about a multi-database cancer/drug bridge.

{SCHEMA_DESCRIPTION}

Fields: {json.dumps(BRIDGE_FIELDS)}

{samples_text}

The user asks: "{question}"

Translate this into JSON with three keys:
1. "query_fields": dict mapping field names to values the user is asking about.
   Use UPPERCASE for therapy and gene names (e.g. "VEMURAFENIB", "BRAF V600E").
   Include as many fields as you can infer. Use values that would match the records.
2. "ask_field": the single field whose value answers the question.
3. "mode": "route" if the user wants a specific fact, "scan" if they want
   an aggregate (list, count, comparison, "which drugs...", "what are all...", etc.)

Return ONLY valid JSON. Example:
{{"query_fields": {{"molecular_profile": "BRAF V600E", "disease": "Melanoma"}}, "ask_field": "adverse_event", "mode": "scan"}}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
        else:
            raise ValueError(f"Could not parse: {text}")

    return (
        parsed.get("query_fields", {}),
        parsed.get("ask_field", ""),
        parsed.get("mode", "route"),
    )


# ---------------------------------------------------------------------------
# Answer synthesis — Claude reads the retrieved records and answers
# ---------------------------------------------------------------------------

def synthesize_answer(
    client: anthropic.Anthropic,
    question: str,
    records: List[Dict[str, Any]],
    ask_field: str,
    mode: str,
) -> str:
    """Have Claude synthesize a natural language answer from retrieved records."""
    if not records:
        return "No matching records found in the bridge."

    # Compact the records for the prompt
    compact = []
    for rec in records[:30]:  # cap at 30 to stay within context
        entry = {k: v for k, v in rec.items()
                 if k != "bridge_id" and v and v not in ("", "0", "unknown", "False")}
        compact.append(entry)

    prompt = f"""You are answering a question using data from a 6-database cancer/drug bridge
(CIViC + FAERS + ChEMBL + UniProt + STRING + OpenFDA).

Question: "{question}"

Retrieved {len(records)} records (showing up to 30):

{json.dumps(compact, indent=1)}

Answer the question directly and concisely using ONLY the data above.
Highlight connections that span multiple databases (e.g., a drug's mechanism
from ChEMBL combined with its adverse events from FAERS and its gene target
from CIViC). Be specific — cite numbers, drug names, gene names.
Keep it under 200 words."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# Scan mode — filter bridge records for aggregate questions
# ---------------------------------------------------------------------------

def scan_bridge(
    bridge: List[Dict[str, Any]],
    query_fields: Dict[str, Any],
    ask_field: str,
) -> List[Dict[str, Any]]:
    """Filter bridge records matching all query fields (partial/substring match)."""
    matches = []
    for rec in bridge:
        match = True
        for field, value in query_fields.items():
            rec_val = rec.get(field, "")
            # Substring match (case-insensitive)
            if value.upper() not in rec_val.upper():
                match = False
                break
        if match:
            matches.append(rec)
    return matches


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

def interactive_mode(
    client: anthropic.Anthropic,
    bridge: List[Dict[str, Any]],
    router: SemanticRouter,
    sample_records: List[Dict[str, Any]],
):
    info = router.explain()

    print(f"\n{'='*64}")
    print(f"  Database Whisper — 6-Database Natural Language Interface")
    print(f"{'='*64}")
    print(f"  Bridge: {info['total_records']} records from 6 databases")
    print(f"  Ladder: identity -> {' -> '.join(router.ladder_fields)}")
    print(f"  Fields per record: {len(BRIDGE_FIELDS)}")
    print(f"\n  Ask anything about drugs, genes, adverse events, mechanisms,")
    print(f"  protein interactions, clinical trials, or boxed warnings.")
    print(f"  Type 'signals' to see cross-database signals.")
    print(f"  Type 'quit' to exit.\n")

    while True:
        try:
            question = input("  Ask: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question or question.lower() in ("quit", "exit", "q"):
            break

        # Special command: show signals
        if question.lower() == "signals":
            print("\n  Detecting cross-database signals...")
            signals = detect_mega_signals(bridge)
            by_type = defaultdict(list)
            for s in signals:
                by_type[s["signal_type"]].append(s)
            for stype, slist in by_type.items():
                print(f"\n  {stype}: {len(slist)} signals")
                for s in slist[:3]:
                    print(f"    {json.dumps(s, indent=2)[:200]}")
            print()
            continue

        try:
            # Step 1: Translate question
            t0 = time.time()
            query_fields, ask_field, mode = translate_question(
                client, question, sample_records,
            )
            t_translate = time.time() - t0

            print(f"\n  Parsed: {json.dumps(query_fields)} -> {ask_field} [{mode}]")

            # Step 2: Route or scan
            t1 = time.time()
            if mode == "route":
                result = router.query(query_fields, ask_field=ask_field)
                retrieved = [result.matched_record] if result.matched_record else []
                examined = result.records_examined
                route_desc = result.route_used
                narrowing = result.candidates_at_each_stage
            else:
                retrieved = scan_bridge(bridge, query_fields, ask_field)
                examined = len(bridge)
                route_desc = "scan"
                narrowing = [len(bridge), len(retrieved)]

            t_route = time.time() - t1

            print(f"  Retrieved: {len(retrieved)} records (examined {examined}/{info['total_records']})")
            if narrowing:
                print(f"  Narrowing: {' -> '.join(str(n) for n in narrowing)}")
            print(f"  Route: {route_desc}")

            # Step 3: Synthesize answer
            t2 = time.time()
            answer = synthesize_answer(client, question, retrieved, ask_field, mode)
            t_synth = time.time() - t2

            print(f"\n  {answer}")
            print(f"\n  [translate: {t_translate:.1f}s | route: {t_route*1000:.1f}ms | synthesize: {t_synth:.1f}s]\n")

        except Exception as e:
            print(f"\n  Error: {e}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: Set ANTHROPIC_API_KEY environment variable")
        return

    client = anthropic.Anthropic(api_key=api_key)

    t0 = time.time()
    bridge, router, stats = load_bridge()
    load_time = time.time() - t0
    print(f"Ready in {load_time:.1f}s (API data cached from prior run)")

    # Pick diverse samples for Claude context
    rng = random.Random(42)
    # Get samples that have rich data across databases
    rich = [r for r in bridge if r.get("mechanism") and r.get("interaction_partners")]
    sample_records = rng.sample(rich, min(5, len(rich))) if rich else rng.sample(bridge, min(5, len(bridge)))

    interactive_mode(client, bridge, router, sample_records)
    print("Done.")


if __name__ == "__main__":
    main()
