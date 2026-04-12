"""Core data types for database-whisper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LadderRung:
    """One level of the discovered discriminator ladder."""
    field_name: str
    ambiguity_reduction_rate: float


@dataclass
class RouteResult:
    """The answer to one routed query."""
    answer: Optional[str]
    records_examined: int
    total_records: int
    route_used: str
    candidates_at_each_stage: List[int]
    matched_record: Optional[Dict[str, Any]] = None
    confusion_candidates: int = 0


@dataclass
class InsertEvent:
    """What happened when a record was inserted into a live router."""
    event_type: str  # "slot", "grow", "reorganize", "first"
    record_num: int
    identity_key: tuple
    neighborhood_size: int
    ladder_changed: bool
    old_ladder: List[str]
    new_ladder: List[str]
    rungs_added: List[str]
    rungs_removed: List[str]
    rungs_reordered: bool
    insert_time_ms: float


@dataclass
class SleepEvent:
    """What the memory learned during one sleep consolidation cycle."""
    cycle_num: int
    records_consolidated: int
    slotted: int
    leaf_adjusted: int
    core_shifted: bool
    old_ladder: List[str]
    new_ladder: List[str]
    rungs_changed: List[str]
    new_neighborhoods: int
    collisions: int
    sleep_time_ms: float
    structural_surprises: List[str]


@dataclass
class StructuralProfile:
    """Complete structural profile of a dataset."""
    source: str
    total_records: int
    total_fields: int
    field_names: List[str]

    # Auto-detected schema
    identity_fields: List[str]
    provenance_fields: List[str]
    candidate_fields: List[str]

    # Ladder
    ladder: List[LadderRung]
    ladder_fields: List[str]

    # Routing metrics
    speedup: float
    accuracy: float
    neighborhoods: int
    ambiguous_neighborhoods: int

    # Structural characterization
    density: str  # "HIGH", "MEDIUM", "LOW", "UNIQUE"
    fingerprint: str  # "SINGLE-AXIS", "DEEP-PIPELINE", "ALREADY-UNIQUE", "LOW-STRUCTURE"
    fingerprint_description: str

    # Index recommendations
    index_recommendations: List[str]

    # Data quality
    ambiguity_rate: float
    anomaly_sensitive_fields: List[str]
    fully_resolved: bool

    def __str__(self) -> str:
        """Human-readable structural profile report."""
        lines = []
        lines.append(f"=== Structural Profile: {self.source} ===")
        lines.append(f"Records: {self.total_records:,} | Fields: {self.total_fields}")
        lines.append("")

        # Density
        lines.append(f"Structural Density: {self.density} ({self.speedup:,.0f}x speedup)")
        density_desc = {
            "HIGH": "This dataset has deep categorical structure.",
            "MEDIUM": "Moderate categorical structure with some ambiguity.",
            "LOW": "Minimal categorical structure. Mostly continuous or unique values.",
            "UNIQUE": "Every record is already unique. No disambiguation needed.",
        }
        lines.append(f"  {density_desc.get(self.density, '')}")
        lines.append("")

        # Schema
        lines.append(f"Auto-detected Identity: {', '.join(self.identity_fields)}")
        if self.provenance_fields:
            lines.append(f"Auto-detected Provenance: {', '.join(self.provenance_fields)}")
        lines.append("")

        # Ladder
        lines.append("Discriminator Ladder:")
        if self.ladder:
            max_rate = max(r.ambiguity_reduction_rate for r in self.ladder)
            for i, rung in enumerate(self.ladder):
                bar_len = int(rung.ambiguity_reduction_rate / max(max_rate, 0.01) * 20)
                bar = "#" * bar_len + "." * (20 - bar_len)
                label = "dominant" if rung.ambiguity_reduction_rate > 0.9 else ""
                lines.append(
                    f"  {i+1}. {rung.field_name:<25} "
                    f"{rung.ambiguity_reduction_rate:>6.1%} reduction  "
                    f"{bar}  {label}"
                )
        else:
            lines.append("  (no ladder needed - every record is unique)")
        lines.append("")

        # Index recommendations
        lines.append("Recommended Indexes:")
        if self.index_recommendations:
            for rec in self.index_recommendations:
                lines.append(f"  {rec}")
        else:
            lines.append("  No indexes needed - data is already unique.")
        lines.append("")

        # Data quality
        lines.append("Data Quality:")
        lines.append(f"  Ambiguous neighborhoods: {self.ambiguous_neighborhoods:,} / {self.neighborhoods:,} ({self.ambiguity_rate:.1%})")
        lines.append(f"  Fully resolved by ladder: {'YES' if self.fully_resolved else 'NO'} ({self.accuracy:.0%} accuracy)")
        if self.anomaly_sensitive_fields:
            lines.append(f"  Anomaly-sensitive fields: {', '.join(self.anomaly_sensitive_fields)}")
        lines.append("")

        # Fingerprint
        lines.append(f"Structural Fingerprint: {self.fingerprint}")
        lines.append(f"  {self.fingerprint_description}")

        return "\n".join(lines)
