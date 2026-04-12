"""gap_finder.py

Find surprising gaps in scientific literature using structural routing.

A gap is surprising when:
    - The combination (subfield, method, data_type) is ABSENT in field X
    - But PRESENT in neighboring fields
    - AND the surrounding combinations in field X are DENSE

The surprise score = (neighbor_density * cross_field_presence) / distance_to_nearest_filled

Usage:
    python gap_finder.py
    # Pulls from OpenAlex, builds structural map, finds gaps, serves a page
"""

from __future__ import annotations

import json
import csv
import os
import time
import urllib.request
import urllib.parse
import ssl
import html
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from collections import defaultdict, Counter
from typing import Any, Dict, List, Optional, Tuple

SSL_CTX = ssl._create_unverified_context()


def fetch_json(url: str) -> Optional[Any]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "GapFinder/1.0 (datamunchies.com)"})
        with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  Fetch error: {e}")
        return None


# ---------------------------------------------------------------------------
# Pull papers from OpenAlex
# ---------------------------------------------------------------------------

def pull_papers(total: int = 10000, year_range: str = "2022-2025") -> List[Dict]:
    """Pull papers with topic metadata from OpenAlex."""
    cache_path = f"openalex_cache_{total}_{year_range}.json"
    if os.path.exists(cache_path):
        print(f"  Loading from cache: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    records = []
    cursor = "*"
    per_page = 200

    base_url = (
        f"https://api.openalex.org/works?"
        f"filter=cited_by_count:>5,publication_year:{year_range}"
        f"&select=id,title,publication_year,type,cited_by_count,"
        f"primary_topic,authorships,open_access,concepts"
        f"&per_page={per_page}"
    )

    while len(records) < total:
        url = f"{base_url}&cursor={cursor}"
        data = fetch_json(url)
        if not data or not data.get("results"):
            break

        for work in data["results"]:
            if len(records) >= total:
                break

            pt = work.get("primary_topic") or {}
            domain = (pt.get("domain") or {}).get("display_name", "")
            field = (pt.get("field") or {}).get("display_name", "")
            subfield = (pt.get("subfield") or {}).get("display_name", "")
            topic = pt.get("display_name", "")

            concepts = work.get("concepts") or []
            top_concepts = [c.get("display_name", "") for c in concepts[:5] if c.get("score", 0) > 0.3]

            authorships = work.get("authorships") or []
            num_authors = len(authorships)
            countries = set()
            for a in authorships:
                for inst in (a.get("institutions") or []):
                    cc = inst.get("country_code", "")
                    if cc:
                        countries.add(cc)

            oa = work.get("open_access") or {}
            work_type = work.get("type", "")
            cited = work.get("cited_by_count", 0)
            year = work.get("publication_year", "")

            records.append({
                "domain": domain,
                "field": field,
                "subfield": subfield,
                "topic": topic,
                "type": work_type,
                "year": str(year),
                "cited": cited,
                "num_authors": num_authors,
                "countries": "|".join(sorted(countries)),
                "num_countries": len(countries),
                "oa_status": oa.get("oa_status", ""),
                "concepts": "|".join(top_concepts[:3]),
                "title": (work.get("title") or "")[:150],
            })

        cursor = data.get("meta", {}).get("next_cursor", "")
        if not cursor:
            break
        print(f"  {len(records)}/{total}...")
        time.sleep(0.1)

    # Cache
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(records, f)
    print(f"  Cached {len(records)} papers")

    return records


# ---------------------------------------------------------------------------
# Gap detection with surprise scoring
# ---------------------------------------------------------------------------

def find_gaps(papers: List[Dict]) -> List[Dict]:
    """
    Find surprising gaps in the literature.

    Strategy:
    1. Build a 3D grid: field x subfield x type
    2. For each empty cell, score it by:
       - How many other fields have this (subfield, type) combo
       - How dense the surrounding cells are (same field, nearby subfields)
       - How many papers are in the closest filled cell
    3. Rank by surprise = cross_field_presence * neighbor_density
    """
    # Build the grid
    grid = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for p in papers:
        grid[p["field"]][p["subfield"]][p["type"]].append(p)

    # Get dimensions
    all_fields = sorted(set(p["field"] for p in papers if p["field"]))
    all_subfields = sorted(set(p["subfield"] for p in papers if p["subfield"]))
    core_types = ["article", "review", "preprint"]

    # For each field, find what subfields are active
    field_subfields = defaultdict(set)
    for p in papers:
        if p["field"] and p["subfield"]:
            field_subfields[p["field"]].add(p["subfield"])

    # Score gaps
    gaps = []
    for field in all_fields:
        active_subfields = field_subfields[field]
        if len(active_subfields) < 3:
            continue  # need enough subfields to detect gaps

        for subfield in all_subfields:
            for work_type in core_types:
                # Is this cell empty?
                if grid[field][subfield][work_type]:
                    continue  # not a gap

                # Score 1: Cross-field presence
                # How many OTHER fields have papers in this (subfield, type)?
                cross_count = 0
                cross_papers = []
                for other_field in all_fields:
                    if other_field != field:
                        cell = grid[other_field][subfield][work_type]
                        if cell:
                            cross_count += 1
                            cross_papers.extend(cell[:2])

                if cross_count < 2:
                    continue  # not enough cross-field evidence

                # Score 2: Neighbor density in same field
                # How many papers does this field have in the SAME subfield (any type)?
                same_subfield_count = sum(
                    len(grid[field][subfield][t]) for t in core_types
                )
                # How many papers in same field, same type, ANY subfield?
                same_type_count = sum(
                    len(grid[field][sf][work_type]) for sf in active_subfields
                )

                neighbor_density = same_subfield_count + same_type_count

                # Score 3: Is this subfield active in this field at all?
                subfield_active = subfield in active_subfields
                activity_bonus = 3.0 if subfield_active else 1.0

                # Surprise score
                surprise = cross_count * (neighbor_density + 1) * activity_bonus

                # Find closest filled neighbor for context
                nearby = []
                if subfield_active:
                    for t in core_types:
                        nearby.extend(p["title"] for p in grid[field][subfield][t][:2])

                gaps.append({
                    "field": field,
                    "subfield": subfield,
                    "missing_type": work_type,
                    "surprise_score": surprise,
                    "cross_field_count": cross_count,
                    "neighbor_density": neighbor_density,
                    "subfield_active_in_field": subfield_active,
                    "same_subfield_papers": same_subfield_count,
                    "same_type_papers": same_type_count,
                    "nearby_papers": nearby[:3],
                    "cross_field_papers": [p["title"] for p in cross_papers[:3]],
                })

    gaps.sort(key=lambda g: -g["surprise_score"])
    return gaps


# ---------------------------------------------------------------------------
# Web interface
# ---------------------------------------------------------------------------

def esc(s: str) -> str:
    return html.escape(str(s))


def render_page(papers: List[Dict], gaps: List[Dict], selected_field: str = "") -> str:
    fields = sorted(set(p["field"] for p in papers if p["field"]))
    field_counts = Counter(p["field"] for p in papers)

    body = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gap Finder - Data Munchies</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0a0a0f; color: #d0d0d0; line-height: 1.6; }
a { color: #7c8cf5; text-decoration: none; }
.header { text-align: center; padding: 40px 24px 20px; }
.header h1 { font-size: 32px; color: #fff; margin-bottom: 8px; }
.header p { color: #666; font-size: 15px; }
.container { max-width: 1100px; margin: 0 auto; padding: 20px 24px; }
.field-nav { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 24px; justify-content: center; }
.field-btn { background: #16162a; border: 1px solid #2a2a3a; color: #888; padding: 6px 14px;
             border-radius: 6px; font-size: 13px; cursor: pointer; text-decoration: none; }
.field-btn:hover { border-color: #7c8cf5; color: #fff; }
.field-btn.active { background: #7c8cf5; color: #fff; border-color: #7c8cf5; }
.gap-card { background: #12121a; border: 1px solid #2a2a3a; border-radius: 12px;
            padding: 20px; margin-bottom: 16px; }
.gap-card:hover { border-color: #7c8cf5; }
.gap-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px; }
.gap-title { font-size: 16px; font-weight: 600; color: #fff; }
.gap-score { font-size: 24px; font-weight: 700; color: #7c8cf5; }
.gap-meta { font-size: 13px; color: #888; margin-bottom: 12px; }
.gap-detail { font-size: 13px; color: #aaa; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px;
       font-weight: 600; margin: 2px; }
.tag-active { background: #1a3a1a; color: #4caf50; }
.tag-missing { background: #3a1a1a; color: #f44336; }
.tag-cross { background: #1a1a3a; color: #64b5f6; }
.nearby { background: #0d0d18; border-radius: 6px; padding: 10px; margin-top: 10px; }
.nearby-title { font-size: 11px; color: #555; text-transform: uppercase; margin-bottom: 6px; }
.nearby-paper { font-size: 12px; color: #888; margin: 3px 0; }
.stats { display: flex; gap: 24px; justify-content: center; margin-bottom: 24px; }
.stat { text-align: center; }
.stat-num { font-size: 28px; font-weight: 700; color: #7c8cf5; }
.stat-label { font-size: 12px; color: #555; }
footer { text-align: center; padding: 40px; color: #333; font-size: 12px; }
</style>
</head><body>
<div class="header">
    <h1>Gap Finder</h1>
    <p>Find where nobody has published yet. Powered by Database Whisper.</p>
</div>
<div class="container">
"""

    # Stats
    total_gaps = len(gaps)
    filtered_gaps = [g for g in gaps if g["field"] == selected_field] if selected_field else gaps
    high_surprise = len([g for g in filtered_gaps if g["surprise_score"] > 50])

    body += f"""<div class="stats">
    <div class="stat"><div class="stat-num">{len(papers):,}</div><div class="stat-label">Papers analyzed</div></div>
    <div class="stat"><div class="stat-num">{len(fields)}</div><div class="stat-label">Fields</div></div>
    <div class="stat"><div class="stat-num">{total_gaps}</div><div class="stat-label">Gaps found</div></div>
    <div class="stat"><div class="stat-num">{high_surprise}</div><div class="stat-label">High surprise</div></div>
</div>"""

    # Field navigation
    body += '<div class="field-nav">'
    all_cls = "active" if not selected_field else ""
    body += f'<a href="/" class="field-btn {all_cls}">All Fields</a>'
    for f in fields[:15]:
        cls = "active" if f == selected_field else ""
        count = field_counts[f]
        body += f'<a href="/?field={esc(f)}" class="field-btn {cls}">{esc(f)} ({count})</a>'
    body += '</div>'

    # Gaps
    show_gaps = filtered_gaps[:30]

    if not show_gaps:
        body += '<p style="text-align:center; color:#555; padding:40px;">No gaps found for this field. Try another one.</p>'
    else:
        for g in show_gaps:
            active_tag = '<span class="tag tag-active">subfield active</span>' if g["subfield_active_in_field"] else '<span class="tag tag-missing">subfield not yet in field</span>'

            body += f"""<div class="gap-card">
    <div class="gap-header">
        <div>
            <div class="gap-title">{esc(g['field'])}: no {esc(g['missing_type'])}s in {esc(g['subfield'])}</div>
            <div class="gap-meta">
                {active_tag}
                <span class="tag tag-cross">{g['cross_field_count']} other fields have this</span>
            </div>
        </div>
        <div class="gap-score">{g['surprise_score']:.0f}</div>
    </div>
    <div class="gap-detail">
        <strong>Why this matters:</strong>
        {g['cross_field_count']} other fields publish {esc(g['missing_type'])}s in {esc(g['subfield'])},
        but {esc(g['field'])} has none.
        {'This subfield IS active in ' + esc(g['field']) + ' (' + str(g['same_subfield_papers']) + ' papers in other types), making this gap more surprising.' if g['subfield_active_in_field'] else 'This subfield is not yet represented in ' + esc(g['field']) + '.'}
    </div>"""

            if g.get("nearby_papers"):
                body += '<div class="nearby"><div class="nearby-title">Nearby papers in this field + subfield</div>'
                for title in g["nearby_papers"][:3]:
                    body += f'<div class="nearby-paper">{esc(title)}</div>'
                body += '</div>'

            if g.get("cross_field_papers"):
                body += '<div class="nearby" style="margin-top:6px;"><div class="nearby-title">Papers from other fields in this gap</div>'
                for title in g["cross_field_papers"][:3]:
                    body += f'<div class="nearby-paper">{esc(title)}</div>'
                body += '</div>'

            body += '</div>'

    body += """</div>
<footer>Gap Finder by <a href="https://datamunchies.com">Data Munchies</a> | Powered by <a href="https://github.com/nathanelms/Database-whisperer">Database Whisper</a> + <a href="https://openalex.org">OpenAlex</a></footer>
</body></html>"""

    return body


class GapHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        field = params.get("field", [""])[0]

        content = render_page(PAPERS, GAPS, field)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(content.encode("utf-8"))

    def log_message(self, *args):
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PAPERS: List[Dict] = []
GAPS: List[Dict] = []


def main():
    global PAPERS, GAPS

    print("=" * 60)
    print("  GAP FINDER: Where nobody has published yet")
    print("=" * 60)

    print("\nPulling papers from OpenAlex...")
    PAPERS = pull_papers(total=10000, year_range="2022-2025")
    print(f"Total papers: {len(PAPERS)}")

    print("\nFinding gaps...")
    GAPS = find_gaps(PAPERS)
    print(f"Total gaps: {len(GAPS)}")
    print(f"High surprise (>50): {len([g for g in GAPS if g['surprise_score'] > 50])}")

    print(f"\nTop 10 gaps:")
    for g in GAPS[:10]:
        print(f"  [{g['surprise_score']:.0f}] {g['field']}: no {g['missing_type']}s in {g['subfield']}")
        print(f"       {g['cross_field_count']} other fields have this, "
              f"{'subfield active' if g['subfield_active_in_field'] else 'subfield absent'}")

    import threading
    import webbrowser

    port = 8060
    server = HTTPServer(("", port), GapHandler)
    print(f"\nGap Finder running at http://localhost:{port}")
    threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
