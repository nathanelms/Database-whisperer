# database-whisper

Auto-discovers structural patterns in datasets. Point it at a file, it tells you which fields matter for disambiguation, recommends indexes, and characterizes the structure.

Zero configuration. Zero dependencies (core). Works on CSV, TSV, JSON, SQLite, Excel, Parquet, and SQL dumps.

## Install

```bash
pip install database-whisper
```

Optional format support:
```bash
pip install openpyxl    # for Excel .xlsx
pip install pyarrow     # for Parquet
```

## Quick start

```python
import database_whisper as dw

report = dw.profile("your_data.csv")
print(report)
```

Output:
```
=== Structural Profile: your_data.csv ===
Records: 114,000 | Fields: 20

Structural Density: HIGH (112,871x speedup)
  This dataset has deep categorical structure.

Auto-detected Identity: track_id, track_name, duration_ms

Discriminator Ladder:
  1. track_genre          98.7% reduction  ####################  dominant

Recommended Indexes:
  CREATE INDEX idx_track_genre ON tracks (track_genre);
    -- Standalone index: 99% reduction alone.

Data Quality:
  Ambiguous neighborhoods: 16,641 / 89,741 (18.5%)
  Fully resolved by ladder: YES (100% accuracy)

Structural Fingerprint: SINGLE-AXIS
  One field dominates. Minimal disambiguation depth needed.
```

## What it does

Given any structured dataset, the algorithm:

1. **Auto-detects** which fields are identity (primary keys) and which are provenance (record IDs to exclude)
2. **Discovers** a discriminator ladder — the ordered sequence of fields that best resolves ambiguity among records sharing the same identity
3. **Measures** retrieval speedup vs flat scan and structural density
4. **Recommends** database indexes based on the discovered structure
5. **Classifies** the dataset by its structural fingerprint (SINGLE-AXIS, DEEP-PIPELINE, ALREADY-UNIQUE, LOW-STRUCTURE)

## Supported formats

| Format | Extension | Dependencies |
|--------|-----------|-------------|
| CSV / TSV | .csv, .tsv | none |
| JSON (array or nested) | .json | none |
| NDJSON | .ndjson, .jsonl | none |
| SQLite | .db, .sqlite | none |
| SQL dump | .sql | none |
| Excel | .xlsx | openpyxl |
| Parquet | .parquet | pyarrow |

## API

```python
import database_whisper as dw

# Profile a file (auto-detects format)
report = dw.profile("data.csv")
report = dw.profile("data.db")
report = dw.profile("data.xlsx")

# Profile in-memory records
report = dw.profile_records(records, field_names=["col1", "col2", ...])

# Batch router
router = dw.Router()
router.ingest(records, identity_fields=["gene", "disease"])
result = router.query({"gene": "BRAF", "disease": "Melanoma"}, ask_field="therapy")

# Streaming / incremental
live = dw.LiveRouter(identity_fields=["gene", "disease"])
for record in stream:
    event = live.insert(record)

# Memory with sleep consolidation
mem = dw.Memory(identity_fields=["gene", "disease"])
for fact in facts:
    mem.insert(fact)
```

## Tested domains

The algorithm has been validated on 9 datasets across different domains. Same code, different data, different structures discovered.

| Domain | Records | Speedup | Accuracy |
|--------|---------|---------|----------|
| Oncology (CIViC) | 4,825 | 4,761x | 100% |
| Pharma safety (FAERS) | 50,000 | 7,462x | 100% |
| Weather (NOAA Storm) | 50,000 | 50,000x | 100% |
| Astronomy (NASA Exoplanets) | 6,158 | 6,109x | 100% |
| Seismology (USGS Earthquakes) | 20,000 | 20,000x | 100% |
| Particle physics (CERN CMS) | 100,000 | 100,000x | 100% |
| Music (Spotify) | 114,000 | 112,871x | 100% |
| Astronomy (LSST PLAsTiCC) | 7,848 | 7,848x | 100% |
| Cosmology (LSST CosmoDC2) | 50,000 | 3x | 100% |

## Research

- [Paper I: Discriminator Ladder Learning](paper/paper.tex) — the algorithm and 3-domain validation
- [Paper II: Five Consequences of One Algorithm](paper/paper_v2.tex) — anomaly detection, reasoning traces, compression, federated bridging across 9 domains

## Requirements

Python 3.9+. Core package has zero external dependencies.

## License

MIT
