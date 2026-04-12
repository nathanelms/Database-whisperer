"""whisper_rag.py

Database Whisper RAG Demo: Natural language queries routed through the
semantic router, answered by Claude.

This is the product demo. It shows:
1. Load any CSV
2. Router auto-discovers the semantic ladder
3. User asks a natural language question
4. Claude translates the question into structured query fields
5. Router narrows to a tiny candidate set instantly
6. Claude answers from just those candidates instead of scanning everything

Usage:
    python whisper_rag.py --csv civic_evidence_full.tsv --delimiter tab
    python whisper_rag.py --csv storm_events_2023.csv
    python whisper_rag.py --csv your_data.csv

Requires:
    pip install anthropic
    ANTHROPIC_API_KEY environment variable set
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from typing import Any, Dict, List, Optional

import anthropic

from semantic_router import SemanticRouter


def load_csv(
    path: str,
    delimiter: str = ",",
    max_records: int = 200_000,
) -> tuple[List[Dict[str, Any]], List[str]]:
    """
    Load a CSV/TSV into a list of dicts.

    What this does:
    - Reads any delimited file and returns clean records plus the field names.

    Why this exists:
    - The demo should work on any CSV the user points it at.
    """
    records = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        field_names = reader.fieldnames or []
        for row in reader:
            if len(records) >= max_records:
                break
            # Clean whitespace from all values.
            record = {k.strip(): (v or "").strip() for k, v in row.items() if k}
            records.append(record)
    return records, [f.strip() for f in field_names if f]


def auto_detect_identity_fields(
    records: List[Dict[str, Any]],
    field_names: List[str],
    max_candidates: int = 3,
) -> List[str]:
    """
    Heuristically pick identity fields by choosing the fields with highest
    cardinality that are not unique (not IDs).

    What this does:
    - Scores each field by how many unique values it has relative to total records.
    - Picks fields that have moderate cardinality (not too few, not unique per record).

    Why this exists:
    - The demo should work without the user manually specifying identity fields.

    What assumption it is making:
    - Good identity fields have many distinct values but are not unique per record.
    """
    total = len(records)
    if total == 0:
        return field_names[:2]

    scored = []
    for field in field_names:
        values = set(r.get(field, "") for r in records)
        unique_count = len(values)
        # Skip fields that are unique per record (likely IDs).
        if unique_count > total * 0.8:
            continue
        # Skip fields with very few unique values (likely flags/booleans).
        if unique_count < 3:
            continue
        # Prefer fields with moderate cardinality.
        score = unique_count
        scored.append((score, field))

    scored.sort(reverse=True)
    # Pick top candidates.
    chosen = [field for _, field in scored[:max_candidates]]

    if not chosen:
        return field_names[:2]
    return chosen


def auto_detect_provenance_fields(
    records: List[Dict[str, Any]],
    field_names: List[str],
) -> List[str]:
    """
    Detect fields that are likely record IDs or provenance markers.

    What this does:
    - Finds fields where nearly every value is unique.

    Why this exists:
    - These should be excluded from semantic routing.
    """
    total = len(records)
    if total == 0:
        return []

    provenance = []
    for field in field_names:
        values = set(r.get(field, "") for r in records)
        if len(values) > total * 0.8:
            provenance.append(field)
    return provenance


def build_query_prompt(
    question: str,
    field_names: List[str],
    identity_fields: List[str],
    ladder_fields: List[str],
    sample_records: List[Dict[str, Any]],
) -> str:
    """
    Build the prompt that asks Claude to translate a natural language question
    into structured query fields.

    What this does:
    - Gives Claude the schema, a few sample records, and asks it to produce
      a JSON dict of field values that would help route the query.

    Why this exists:
    - The router needs structured field values. The user types natural language.
      Claude bridges the gap.
    """
    samples_text = ""
    for i, record in enumerate(sample_records[:3]):
        samples_text += f"\nSample record {i+1}:\n"
        for k, v in record.items():
            samples_text += f"  {k}: {v}\n"

    return f"""You are a query translator for a structured database.

The database has these fields: {json.dumps(field_names)}

The primary identity fields are: {json.dumps(identity_fields)}

The discovered routing fields are: {json.dumps(ladder_fields)}

{samples_text}

The user asks: "{question}"

Your job: translate this natural language question into a JSON object with two keys:
1. "query_fields": a dict mapping field names to the values the user is asking about. Include as many relevant fields as you can infer from the question. Use exact values that would match the database records.
2. "ask_field": the single field name whose value the user wants to retrieve.

Return ONLY valid JSON, no explanation. Example format:
{{"query_fields": {{"field1": "value1", "field2": "value2"}}, "ask_field": "field3"}}"""


def translate_question(
    client: anthropic.Anthropic,
    question: str,
    field_names: List[str],
    identity_fields: List[str],
    ladder_fields: List[str],
    sample_records: List[Dict[str, Any]],
) -> tuple[Dict[str, Any], str]:
    """
    Use Claude to translate a natural language question into structured query fields.

    What this does:
    - Sends the question plus schema context to Claude and parses the JSON response.

    Why this exists:
    - This is the bridge between human language and the semantic router.
    """
    prompt = build_query_prompt(
        question=question,
        field_names=field_names,
        identity_fields=identity_fields,
        ladder_fields=ladder_fields,
        sample_records=sample_records,
    )

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = response.content[0].text.strip()

    # Parse JSON from response.
    try:
        parsed = json.loads(response_text)
        return parsed.get("query_fields", {}), parsed.get("ask_field", "")
    except json.JSONDecodeError:
        # Try to extract JSON from the response if it has extra text.
        import re
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            return parsed.get("query_fields", {}), parsed.get("ask_field", "")
        raise ValueError(f"Could not parse Claude response: {response_text}")


def format_answer(
    question: str,
    result,
    query_fields: Dict[str, Any],
    ask_field: str,
) -> str:
    """
    Format the routed answer for display.

    What this does:
    - Builds a human-readable response showing the answer and routing metadata.
    """
    lines = []
    lines.append(f"\nQuestion: {question}")
    lines.append(f"Translated to: {json.dumps(query_fields)} -> ask: {ask_field}")
    lines.append(f"Route used: {result.route_used}")
    lines.append(f"Records examined: {result.records_examined} / {result.total_records}")

    if result.answer is not None:
        lines.append(f"Answer: {result.answer}")
        if result.confusion_candidates > 0:
            lines.append(f"  (warning: {result.confusion_candidates} other candidates in same leaf)")
    else:
        lines.append("Answer: No match found")

    if result.candidates_at_each_stage:
        lines.append(f"Narrowing trace: {' -> '.join(str(c) for c in result.candidates_at_each_stage)}")

    speedup = result.total_records / max(result.records_examined, 1)
    lines.append(f"Speedup vs flat scan: {speedup:.0f}x")

    return "\n".join(lines)


def interactive_mode(
    client: anthropic.Anthropic,
    router: SemanticRouter,
    field_names: List[str],
    identity_fields: List[str],
    sample_records: List[Dict[str, Any]],
):
    """
    Run an interactive question-answer loop.

    What this does:
    - Lets the user type natural language questions and see routed answers.

    Why this exists:
    - This is the demo. Point at a CSV, ask questions, see the router work.
    """
    ladder_fields = router.ladder_fields
    info = router.explain()

    print(f"\n{'='*60}")
    print(f"Database Whisper RAG - Interactive Mode")
    print(f"{'='*60}")
    print(f"Records loaded: {info['total_records']}")
    print(f"Identity fields: {info['identity_fields']}")
    print(f"Discovered ladder: {' -> '.join(['identity'] + ladder_fields)}")
    print(f"Ambiguous neighborhoods: {info['ambiguous_neighborhoods']}")
    print(f"\nType a question about your data. Type 'quit' to exit.\n")

    while True:
        try:
            question = input("Ask: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question or question.lower() in ("quit", "exit", "q"):
            break

        try:
            # Step 1: Claude translates the question.
            t0 = time.time()
            query_fields, ask_field = translate_question(
                client=client,
                question=question,
                field_names=field_names,
                identity_fields=identity_fields,
                ladder_fields=ladder_fields,
                sample_records=sample_records,
            )
            translate_time = time.time() - t0

            # Step 2: Router answers the query.
            t1 = time.time()
            result = router.query(query_fields, ask_field=ask_field)
            route_time = time.time() - t1

            # Step 3: Display.
            print(format_answer(question, result, query_fields, ask_field))
            print(f"Translation time: {translate_time:.2f}s | Route time: {route_time*1000:.1f}ms")
            print()

        except Exception as e:
            print(f"Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Database Whisper RAG: natural language queries routed through semantic hierarchy"
    )
    parser.add_argument("--csv", required=True, help="Path to CSV/TSV file")
    parser.add_argument("--delimiter", default=",", help="Delimiter: ',' or 'tab'")
    parser.add_argument("--max-records", type=int, default=200_000, help="Max records to load")
    parser.add_argument("--identity-fields", nargs="*", help="Override identity field detection")
    parser.add_argument("--provenance-fields", nargs="*", help="Override provenance field detection")

    args = parser.parse_args()

    delimiter = "\t" if args.delimiter == "tab" else args.delimiter

    # Check API key.
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: Set ANTHROPIC_API_KEY environment variable")
        print("  export ANTHROPIC_API_KEY=your-key-here")
        return

    client = anthropic.Anthropic(api_key=api_key)

    # Load data.
    print(f"Loading {args.csv}...")
    records, field_names = load_csv(args.csv, delimiter=delimiter, max_records=args.max_records)
    print(f"  Loaded {len(records)} records with {len(field_names)} fields")
    print(f"  Fields: {field_names}")

    # Detect or use provided identity/provenance fields.
    if args.identity_fields:
        identity_fields = args.identity_fields
    else:
        identity_fields = auto_detect_identity_fields(records, field_names)
    print(f"  Identity fields: {identity_fields}")

    if args.provenance_fields:
        provenance_fields = args.provenance_fields
    else:
        provenance_fields = auto_detect_provenance_fields(records, field_names)
    print(f"  Provenance fields: {provenance_fields}")

    # Build router.
    print("\nBuilding semantic router...")
    t0 = time.time()
    router = SemanticRouter()
    router.ingest(
        records=records,
        identity_fields=identity_fields,
        provenance_fields=provenance_fields,
        max_ladder_depth=4,
    )
    build_time = time.time() - t0

    info = router.explain()
    print(f"  Built in {build_time:.2f}s")
    print(f"  Discovered ladder:")
    for rung in info["ladder"]:
        print(f"    {rung['rung']}: {rung['field']} (reduction={rung['ambiguity_reduction_rate']})")

    # Sample records for Claude context.
    import random
    rng = random.Random(42)
    sample_records = rng.sample(records, min(5, len(records)))

    # Run interactive mode.
    interactive_mode(
        client=client,
        router=router,
        field_names=field_names,
        identity_fields=identity_fields,
        sample_records=sample_records,
    )

    print("Done.")


if __name__ == "__main__":
    main()
