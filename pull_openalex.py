"""Pull structured paper metadata from OpenAlex.

OpenAlex has 250M+ papers with professional topic/concept tags.
No auth needed. Free API.

We pull papers with rich topic metadata and flatten into DW-friendly records:
    domain, field, subfield, topic, type, publication_year, cited_by_count,
    open_access, num_authors, num_concepts, top_concept
"""

import json
import csv
import time
import urllib.request
import urllib.parse
import ssl

SSL_CTX = ssl._create_unverified_context()

def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "DatabaseWhisper/1.0 (datamunchies.com)"})
    with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as resp:
        return json.loads(resp.read().decode("utf-8"))


def pull_papers(total=10000, per_page=200):
    """Pull papers from OpenAlex with topic metadata."""
    records = []
    cursor = "*"

    # Get papers from 2020-2025 with >10 citations and topic data
    base_url = (
        "https://api.openalex.org/works?"
        "filter=cited_by_count:>20,publication_year:2023"
        "&select=id,title,publication_year,type,cited_by_count,"
        "topics,concepts,primary_topic,authorships,open_access"
        f"&per_page={per_page}"
    )

    page = 0
    while len(records) < total:
        page += 1
        url = f"{base_url}&cursor={cursor}"
        print(f"  Page {page}: {len(records)}/{total} records...")

        try:
            data = fetch(url)
        except Exception as e:
            print(f"  Error: {e}")
            break

        results = data.get("results", [])
        if not results:
            break

        for work in results:
            if len(records) >= total:
                break

            # Primary topic hierarchy
            pt = work.get("primary_topic") or {}
            domain = (pt.get("domain") or {}).get("display_name", "")
            field = (pt.get("field") or {}).get("display_name", "")
            subfield = (pt.get("subfield") or {}).get("display_name", "")
            topic = pt.get("display_name", "")
            topic_score = pt.get("score", 0)

            # All topics (count)
            all_topics = work.get("topics") or []
            num_topics = len(all_topics)

            # Secondary domain (if different from primary)
            secondary_domain = ""
            for t in all_topics[1:3]:
                d = (t.get("domain") or {}).get("display_name", "")
                if d and d != domain:
                    secondary_domain = d
                    break

            # Concepts
            concepts = work.get("concepts") or []
            top_concept = concepts[0].get("display_name", "") if concepts else ""
            concept_level = str(concepts[0].get("level", "")) if concepts else ""
            num_concepts = len(concepts)

            # Authors
            authorships = work.get("authorships") or []
            num_authors = len(authorships)
            first_country = ""
            for auth in authorships[:1]:
                insts = auth.get("institutions") or []
                for inst in insts[:1]:
                    first_country = inst.get("country_code", "")

            # Open access
            oa = work.get("open_access") or {}
            oa_status = oa.get("oa_status", "")

            # Type
            work_type = work.get("type", "")
            cited = work.get("cited_by_count", 0)
            year = work.get("publication_year", "")
            title = (work.get("title") or "")[:120]

            # Citation tier
            if cited >= 1000:
                cite_tier = "very_high"
            elif cited >= 100:
                cite_tier = "high"
            elif cited >= 30:
                cite_tier = "medium"
            else:
                cite_tier = "low"

            records.append({
                "openalex_id": (work.get("id") or "").split("/")[-1],
                "title": title,
                "year": str(year),
                "type": work_type,
                "domain": domain,
                "field": field,
                "subfield": subfield,
                "topic": topic,
                "topic_score": f"{topic_score:.2f}" if topic_score else "",
                "secondary_domain": secondary_domain,
                "num_topics": str(num_topics),
                "top_concept": top_concept,
                "concept_level": concept_level,
                "num_concepts": str(num_concepts),
                "num_authors": str(num_authors),
                "first_country": first_country,
                "oa_status": oa_status,
                "cited_by_count": str(cited),
                "cite_tier": cite_tier,
            })

        # Get next cursor
        cursor = data.get("meta", {}).get("next_cursor", "")
        if not cursor:
            break

        time.sleep(0.1)  # polite rate limiting

    return records


def main():
    print("Pulling papers from OpenAlex...")
    records = pull_papers(total=10000)
    print(f"\nTotal papers: {len(records)}")

    # Save as CSV
    with open("openalex_papers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=records[0].keys())
        w.writeheader()
        w.writerows(records)
    print(f"Saved to openalex_papers.csv")

    # Show distributions
    from collections import Counter
    for field_name in ["domain", "field", "type", "cite_tier", "oa_status"]:
        dist = Counter(r[field_name] for r in records)
        print(f"\n{field_name}:")
        for val, count in dist.most_common(10):
            print(f"  {val}: {count}")

    # Run DW
    print("\n" + "=" * 60)
    print("  DW PROFILE: 10k papers from OpenAlex")
    print("=" * 60)

    import database_whisper as dw
    report = dw.profile("openalex_papers.csv")
    print()
    print(report)

    # Profile by domain — what distinguishes papers within Health Sciences vs Physical Sciences?
    print()
    print("=" * 60)
    print("  SEMANTIC ROUTING BY DOMAIN")
    print("=" * 60)

    report2 = dw.profile_records(
        records,
        source="openalex (by domain)",
        identity_fields=["domain"],
        provenance_fields=["openalex_id", "title", "cited_by_count", "topic_score"],
    )
    print()
    print(report2)

    # Profile by domain + field — deeper
    print()
    report3 = dw.profile_records(
        records,
        source="openalex (by domain+field)",
        identity_fields=["domain", "field"],
        provenance_fields=["openalex_id", "title", "cited_by_count", "topic_score"],
    )
    print()
    print(report3)

    # Gap detection
    print()
    print("=" * 60)
    print("  SEMANTIC GAP DETECTION ON 10k PAPERS")
    print("=" * 60)
    print()

    # Within each field, what subfield + type + oa_status combos are missing?
    from collections import defaultdict
    field_combos = defaultdict(lambda: defaultdict(int))
    for r in records:
        key = (r["subfield"], r["type"], r["oa_status"])
        field_combos[r["field"]][key] += 1

    major_fields = [f for f, _ in Counter(r["field"] for r in records).most_common(10)]
    all_subfields = set(r["subfield"] for r in records if r["subfield"])
    all_types = set(r["type"] for r in records if r["type"])
    all_oa = set(r["oa_status"] for r in records if r["oa_status"])

    gaps = []
    for field in major_fields:
        existing = set(field_combos[field].keys())
        for subfield in all_subfields:
            for work_type in all_types:
                for oa in all_oa:
                    combo = (subfield, work_type, oa)
                    if combo not in existing:
                        other_count = sum(1 for f in major_fields if f != field and combo in field_combos[f])
                        if other_count >= 3:
                            gaps.append({
                                "field": field,
                                "missing_subfield": subfield,
                                "missing_type": work_type,
                                "missing_oa": oa,
                                "present_in": other_count,
                            })

    gaps.sort(key=lambda g: -g["present_in"])
    print(f"Semantic gaps found: {len(gaps)}")
    print()
    for g in gaps[:15]:
        print(f"  {g['field']}")
        print(f"    Missing: subfield={g['missing_subfield']}, type={g['missing_type']}, oa={g['missing_oa']}")
        print(f"    Present in {g['present_in']} other fields")
        print()


if __name__ == "__main__":
    main()
