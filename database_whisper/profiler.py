"""Structural profiler for datasets.

Point it at any CSV/TSV/JSON. It tells you what you have.

    from database_whisper import profile
    report = profile("my_data.csv")
    print(report)
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional

from .router import Router
from .loader import load, auto_detect_identity, auto_detect_provenance
from ._types import StructuralProfile, LadderRung


def profile(
    path: str,
    delimiter: Optional[str] = None,
    identity_fields: Optional[List[str]] = None,
    provenance_fields: Optional[List[str]] = None,
    max_records: int = 500_000,
    max_ladder_depth: int = 5,
    sample_queries: int = 300,
) -> StructuralProfile:
    """
    Generate a complete structural profile of a dataset.

    Args:
        path: Path to CSV, TSV, or JSON file.
        delimiter: Field delimiter (auto-detected if None).
        identity_fields: Override auto-detection of identity fields.
        provenance_fields: Override auto-detection of provenance fields.
        max_records: Maximum records to load.
        max_ladder_depth: Maximum ladder depth to discover.
        sample_queries: Number of sample queries for accuracy/speedup measurement.

    Returns:
        StructuralProfile with complete structural analysis.
    """
    # Load data
    records, field_names = load(path, delimiter=delimiter, max_records=max_records)

    if not records:
        return StructuralProfile(
            source=path, total_records=0, total_fields=0, field_names=[],
            identity_fields=[], provenance_fields=[], candidate_fields=[],
            ladder=[], ladder_fields=[], speedup=1.0, accuracy=0.0,
            neighborhoods=0, ambiguous_neighborhoods=0,
            density="EMPTY", fingerprint="EMPTY",
            fingerprint_description="No records found.",
            index_recommendations=[], ambiguity_rate=0.0,
            anomaly_sensitive_fields=[], fully_resolved=True,
        )

    return profile_records(
        records=records,
        field_names=field_names,
        source=path,
        identity_fields=identity_fields,
        provenance_fields=provenance_fields,
        max_ladder_depth=max_ladder_depth,
        sample_queries=sample_queries,
    )


def profile_records(
    records: List[Dict[str, Any]],
    field_names: Optional[List[str]] = None,
    source: str = "<records>",
    identity_fields: Optional[List[str]] = None,
    provenance_fields: Optional[List[str]] = None,
    max_ladder_depth: int = 5,
    sample_queries: int = 300,
) -> StructuralProfile:
    """
    Profile a list of record dicts directly (no file loading).
    """
    if field_names is None:
        field_names = list(records[0].keys()) if records else []

    # Auto-detect if not provided
    if identity_fields is None:
        identity_fields = auto_detect_identity(records, field_names)
    if provenance_fields is None:
        provenance_fields = auto_detect_provenance(records, field_names)

    candidate_fields = [f for f in field_names
                        if f not in identity_fields and f not in provenance_fields]

    # Build router
    router = Router()
    router.ingest(
        records=records,
        identity_fields=identity_fields,
        provenance_fields=provenance_fields,
        max_ladder_depth=max_ladder_depth,
    )

    info = router.explain()
    ladder = router.ladder
    ladder_fields = router.ladder_fields

    # Measure speedup and accuracy
    rng = random.Random(42)
    samples = rng.sample(records, min(sample_queries, len(records)))
    # Pick a field to test retrieval on — prefer ladder fields, then candidates, then any
    available = ladder_fields or candidate_fields or [f for f in field_names if f not in identity_fields and f not in provenance_fields]
    ask_field = available[0] if available else field_names[0] if field_names else ""

    total_routed, total_flat, correct = 0, 0, 0
    for rec in samples:
        routed = router.query(rec, ask_field=ask_field)
        flat = router.flat_scan(rec, ask_field=ask_field)
        total_routed += routed.records_examined
        total_flat += flat.records_examined
        if routed.answer == rec.get(ask_field):
            correct += 1

    speedup = total_flat / max(total_routed, 1)
    accuracy = correct / len(samples)

    # Structural characterization
    neighborhoods = info["identity_neighborhoods"]
    ambiguous = info["ambiguous_neighborhoods"]
    ambiguity_rate = ambiguous / max(neighborhoods, 1)

    density = _classify_density(speedup, len(records))
    fingerprint, fingerprint_desc = _classify_fingerprint(ladder, speedup, len(records), ambiguous)

    # Index recommendations
    index_recs = _generate_index_recommendations(
        source, identity_fields, ladder, speedup
    )

    # Anomaly-sensitive fields (top rungs with high reduction rates)
    anomaly_fields = [r.field_name for r in ladder
                      if r.ambiguity_reduction_rate > 0.3]

    return StructuralProfile(
        source=source,
        total_records=len(records),
        total_fields=len(field_names),
        field_names=field_names,
        identity_fields=identity_fields,
        provenance_fields=provenance_fields,
        candidate_fields=candidate_fields,
        ladder=ladder,
        ladder_fields=ladder_fields,
        speedup=speedup,
        accuracy=accuracy,
        neighborhoods=neighborhoods,
        ambiguous_neighborhoods=ambiguous,
        density=density,
        fingerprint=fingerprint,
        fingerprint_description=fingerprint_desc,
        index_recommendations=index_recs,
        ambiguity_rate=ambiguity_rate,
        anomaly_sensitive_fields=anomaly_fields,
        fully_resolved=accuracy >= 0.99,
    )


def _classify_density(speedup: float, total: int) -> str:
    """Classify structural density from speedup ratio."""
    if speedup >= total * 0.9:
        return "UNIQUE"
    elif speedup >= 1000:
        return "HIGH"
    elif speedup >= 10:
        return "MEDIUM"
    else:
        return "LOW"


def _classify_fingerprint(
    ladder: List[LadderRung],
    speedup: float,
    total: int,
    ambiguous: int,
) -> tuple:
    """Classify the dataset's structural fingerprint."""
    if speedup >= total * 0.9:
        return ("ALREADY-UNIQUE",
                "Every record is already uniquely identified. "
                "No disambiguation hierarchy needed.")

    if not ladder:
        return ("LOW-STRUCTURE",
                "Minimal categorical structure detected. "
                "Data may be primarily continuous or highly uniform.")

    if len(ladder) == 1 and ladder[0].ambiguity_reduction_rate > 0.9:
        return ("SINGLE-AXIS",
                f"One field ({ladder[0].field_name}) dominates with "
                f"{ladder[0].ambiguity_reduction_rate:.0%} reduction. "
                f"Minimal disambiguation depth needed.")

    if len(ladder) >= 3:
        top_rate = ladder[0].ambiguity_reduction_rate
        if top_rate < 0.5:
            return ("DISTRIBUTED",
                    "No single field dominates. Multiple fields contribute "
                    "roughly equally to disambiguation. Deep pipeline needed.")
        else:
            return ("DEEP-PIPELINE",
                    f"Multi-stage disambiguation: {len(ladder)} rungs needed. "
                    f"Top field ({ladder[0].field_name}) handles {top_rate:.0%}, "
                    f"then progressively finer resolution.")

    # 1-2 rungs
    if len(ladder) == 1:
        return ("SINGLE-FIELD",
                f"One field ({ladder[0].field_name}) handles disambiguation "
                f"at {ladder[0].ambiguity_reduction_rate:.0%} reduction.")
    return ("TWO-STAGE",
            f"Two-level structure: {ladder[0].field_name} "
            f"({ladder[0].ambiguity_reduction_rate:.0%}) then "
            f"{ladder[1].field_name} ({ladder[1].ambiguity_reduction_rate:.0%}).")


def _generate_index_recommendations(
    source: str,
    identity_fields: List[str],
    ladder: List[LadderRung],
    speedup: float,
) -> List[str]:
    """Generate SQL CREATE INDEX recommendations from the ladder."""
    recs = []

    # Infer table name from filename
    table = source.split("/")[-1].split("\\")[-1].split(".")[0]

    if not ladder:
        recs.append(f"-- No additional indexes needed for '{table}'. Data is already unique.")
        return recs

    # Composite index on identity + top rungs
    all_fields = identity_fields + [r.field_name for r in ladder]
    idx_name = "idx_" + "_".join(f[:8] for f in all_fields[:4])
    fields_str = ", ".join(all_fields[:6])
    recs.append(f"CREATE INDEX {idx_name} ON {table} ({fields_str});")

    # Explain the recommendation
    top = ladder[0]
    recs.append(
        f"  -- {top.field_name} resolves {top.ambiguity_reduction_rate:.0%} "
        f"of ambiguity within identity neighborhoods."
    )

    if len(ladder) >= 2:
        second = ladder[1]
        recs.append(
            f"  -- {second.field_name} resolves {second.ambiguity_reduction_rate:.0%} "
            f"of remaining ambiguity."
        )

    # Single-field indexes for top rungs
    for rung in ladder[:2]:
        if rung.ambiguity_reduction_rate > 0.5:
            idx_name = f"idx_{rung.field_name[:16]}"
            recs.append(f"CREATE INDEX {idx_name} ON {table} ({rung.field_name});")
            recs.append(f"  -- Standalone index: {rung.ambiguity_reduction_rate:.0%} reduction alone.")

    return recs
