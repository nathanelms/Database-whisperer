# Next Steps — Where We Left Off

## The Big Idea
Apply DW to its own extraction pipeline. Instead of hand-coded keyword lists
for classifying concepts as TOOL/TARGET/FINDING, let DW discover what
contextual features distinguish them.

### How:
1. Take 1000+ (concept, abstract_snippet) pairs where we KNOW the role
   (from papers where it's obvious — "we apply X" = TOOL, "we study X" = TARGET)
2. Extract contextual features: surrounding verbs, position in abstract,
   concept level, co-occurring concepts, sentence structure
3. Feed these as structured records to DW
4. The ladder discovers which features best discriminate TOOL from TARGET from FINDING
5. Use the discovered ladder AS the extraction rules — replacing keyword lists

### Why this matters:
- DW discovers its own feature engineering
- The extraction improves as more data arrives (living ladder)
- Cross-domain: the same role patterns work in any field
- No LLM needed — pure structural extraction

## What's Built and Working
- `database-whisper` v0.1.0 on PyPI
- 7 file formats (CSV, TSV, JSON, SQLite, Excel, Parquet, SQL dumps)
- Gap detection on 20k papers with neighbor-based filtering
- Context-aware concept extraction from abstracts
- unmappeddata.com live
- Two papers published, DOI minted

## What Needs Work
1. **Extraction refinement** — DW on its own extraction (above)
2. **Formspree setup** — waitlist form on unmappeddata.com
3. **First post** — HN or r/GradSchool to test demand
4. **Tool-transfer scoring** — filter domain objects from methods using concept level + context
5. **Scale** — pull 100k papers for denser gap detection

## Product Direction
- **unmappeddata.com** = "Find tools that haven't traveled yet"
- Subscription for students: pick your field, see the gaps
- Powered by DW + OpenAlex + context extraction
- Free during beta, $9/month for students, $29/month for labs
