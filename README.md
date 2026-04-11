# Database Whisper

**A semantic routing engine that automatically discovers how your data wants to be searched.**

Database Whisper analyzes structured data, infers a hierarchical discriminator ladder, and routes queries through that ladder — touching a fraction of the records that flat search requires.

No manual indexing. No schema design. Point it at data, it learns the structure.

## Results

Tested on three completely different public datasets. Same procedure, different data, different ladders discovered automatically.

| Domain | Dataset | Records | Speedup vs flat scan | Routed accuracy | Ladder discovered |
|--------|---------|---------|---------------------|-----------------|-------------------|
| Oncology | CIViC evidence | 4,666 | **3,446x** | 100.00% | rating > therapies > significance > evidence_level |
| Weather | NOAA Storm Events 2023 | 75,593 | **29,808x** | 99.67% | county > month > source > magnitude |
| Pharma safety | FDA FAERS 2024-Q3 | 200,000 | **9,381x** | 99.63% | route > role > sex > dose_form |

Each ladder is different because each dataset has different ambiguity structure. The method found the right routing hierarchy for each domain without being told what to look for.

## How it works

**The core idea:** structured data has hidden routing structure. Some fields are good for narrowing neighborhoods (coarse splitters). Other fields are good for separating near-twin records (final tie-breakers). Database Whisper discovers which is which.

**Step 1: Ingest.** You provide records and tell it which fields define identity (the primary lookup key).

**Step 2: Discover.** The engine finds ambiguous neighborhoods — groups of records sharing the same identity key — and greedily ranks candidate fields by how much each one reduces ambiguity.

**Step 3: Index.** It builds a hierarchical index using the discovered ladder.

**Step 4: Route.** Queries walk the index stage by stage, narrowing candidates at each step instead of scanning everything.

```python
from semantic_router import SemanticRouter

router = SemanticRouter()
router.ingest(
    records=your_data,                           # list of dicts
    identity_fields=["name", "category"],        # primary lookup key
    provenance_fields=["id"],                    # exclude from routing
)

# Routed query -- touches only the records it needs to
result = router.query(
    {"name": "aspirin", "category": "pain", "form": "tablet", "dose": "500mg"},
    ask_field="manufacturer"
)

print(result.answer)              # the retrieved value
print(result.records_examined)    # how many records were touched
print(result.total_records)       # how many exist
print(result.route_used)          # which routing path was taken
```

## What the router discovers

```
=== Router Structure ===
  Total records: 200000
  Identity neighborhoods: 78625
  Ambiguous neighborhoods: 26955
  Discovered ladder:
    Rung 1: route (reduction=14.26%)
    Rung 2: role (reduction=10.54%)
    Rung 3: sex (reduction=6.78%)
    Rung 4: dose_form (reduction=4.34%)
```

The ladder tells you: "To disambiguate records in this dataset, first split by administration route, then by drug role, then by patient sex, then by dose form." Nobody told it that. It inferred it from the ambiguity structure.

## Key concepts

**Discriminator Ladder Learning:** The method of greedily discovering which fields best reduce retrieval ambiguity, ordered from coarsest splitter to finest tie-breaker.

**Semantic routing vs flat scan:** Flat scan checks every record. Semantic routing walks a pre-built hierarchy and only examines records at the leaf. The speedup comes from not looking at records that cannot possibly match.

**Domain-agnostic procedure:** The discovery procedure is the same for every dataset. Only the discovered ladder changes. Oncology data routes by therapy. Weather data routes by county. Adverse events route by administration route. The method finds what matters in each world.

## Project structure

```
semantic_router.py      # The standalone routing engine
test_router_civic.py    # CIViC oncology test (4.6k records)
test_router_storm.py    # NOAA Storm Events test (75k records)
test_router_faers.py    # FDA FAERS test (200k records)
stream_generator.py     # Synthetic data generator for the research sandbox
baseline_runner.py      # Original memory-lab benchmark runner
data_types.py           # Shared dataclasses
memory_policies.py      # Memory selection policies (SaveAll, Tiered, Stub)
retrieval.py            # Retrieval functions
routing.py              # Routing comparison functions
whisper.py              # Database Whisper field analysis
chooser.py              # Adaptive route chooser
meaning_address.py      # Meaning Address v0 prototype
research_log.md         # Running research notes
```

## Running the tests

```bash
# CIViC oncology (downloads data automatically)
python test_router_civic.py

# NOAA Storm Events (requires storm_events_2023.csv)
python test_router_storm.py

# FDA FAERS (requires FAERS extract in ASCII/ folder)
python test_router_faers.py

# Original memory-lab benchmark
python baseline_runner.py --episodes 100 --distractor-level ambiguity
```

## Requirements

Python 3.10+. No external dependencies. Standard library only.

## Origin

This project started as a memory architecture research lab — studying how systems should decide what to remember and how to retrieve it later. The discriminator ladder and semantic routing ideas emerged from testing memory policies on CIViC oncology data and asking: "what is the cheapest field that separates near-twin records?"

That question turned out to be more general than expected.

## License

MIT
