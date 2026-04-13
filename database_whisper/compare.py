"""Structural Quality Index -- compare meaning spaces across datasets.

Measures whether a derivative dataset (synthetic, filtered, translated, summarized)
preserves the structural meaning space of a reference dataset.

Usage:
    import database_whisper as dw

    sqi = dw.compare(reference_records, test_records, text_field="text", concepts=[...])
    print(sqi)             # Overall SQI score
    print(sqi.per_concept) # Per-concept breakdown
    print(sqi.worst_offenders())  # Concepts with most collapse
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional

from .text import (
    auto_detect_concepts,
    extract_concept_instances,
    meaning_addresses,
)
from .router import Router


# All extractable text features.
_TEXT_FEATURES = [
    "verb_class",
    "syntactic_role",
    "paired_concept",
    "contrast",
    "equation",
    "modality",
    "voice",
    "negation",
    "clause_position",
    "transitivity",
]


def _shannon_entropy(counts: List[int]) -> float:
    """Shannon entropy (base-2) of a frequency distribution."""
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return entropy


def _discover_ladder_fields(instances: List[Dict[str, str]]) -> List[str]:
    """Discover the best ladder field order for a set of text instances.

    Uses the Router's own greedy algorithm: identity is "concept",
    candidates are the text features.
    """
    if not instances:
        return []
    candidate_fields = [
        f for f in _TEXT_FEATURES if any(f in inst for inst in instances)
    ]
    if not candidate_fields:
        return []
    router = Router()
    router.ingest(
        records=instances,
        identity_fields=["concept"],
        candidate_fields=candidate_fields,
        max_ladder_depth=min(5, len(candidate_fields)),
    )
    return router.ladder_fields


class ComparisonResult:
    """Stores the result of a structural comparison between two datasets."""

    def __init__(
        self,
        overall_sfi: float,
        overall_sci: float,
        overall_sqi: float,
        per_concept: Dict[str, Dict[str, Any]],
    ) -> None:
        self.overall_sfi = overall_sfi
        self.overall_sci = overall_sci
        self.overall_sqi = overall_sqi
        self.per_concept = per_concept

    def __str__(self) -> str:
        lines = []
        lines.append("=== Structural Quality Index ===")
        lines.append(f"  SFI (fidelity):    {self.overall_sfi:.4f}")
        lines.append(f"  SCI (conservatism):{self.overall_sci:.4f}")
        lines.append(f"  SQI (combined):    {self.overall_sqi:.4f}")
        lines.append("")
        lines.append(f"Per-concept breakdown ({len(self.per_concept)} concepts):")

        # Sort by SQI ascending so worst are first
        sorted_concepts = sorted(
            self.per_concept.items(), key=lambda kv: kv[1]["sqi"]
        )
        for concept, info in sorted_concepts:
            lines.append(
                f"  {concept:<25} SQI={info['sqi']:.3f}  "
                f"SFI={info['sfi']:.3f}  SCI={info['sci']:.3f}  "
                f"ref_addrs={info['ref_addrs']}  test_addrs={info['test_addrs']}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ComparisonResult(sqi={self.overall_sqi:.4f})"

    def __float__(self) -> float:
        return self.overall_sqi

    def worst_offenders(self, n: int = 5) -> List[Dict[str, Any]]:
        """Return n concepts with lowest SQI (most structural collapse)."""
        items = sorted(self.per_concept.items(), key=lambda kv: kv[1]["sqi"])
        return [
            {"concept": c, **info} for c, info in items[:n]
        ]

    def best_preserved(self, n: int = 5) -> List[Dict[str, Any]]:
        """Return n concepts with highest SQI (best preserved structure)."""
        items = sorted(
            self.per_concept.items(), key=lambda kv: kv[1]["sqi"], reverse=True
        )
        return [
            {"concept": c, **info} for c, info in items[:n]
        ]


def compare(
    ref_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    text_field: str,
    concepts: Optional[List[str]] = None,
    provenance_field: str = "id",
    metadata_fields: Optional[List[str]] = None,
    features: Optional[List[str]] = None,
) -> ComparisonResult:
    """Compare meaning spaces of a reference and test dataset.

    Runs DW on both datasets with the same concepts and features,
    then computes SFI, SCI, and SQI per concept and overall.

    Args:
        ref_records: the reference (ground truth) records.
        test_records: the derivative (test) records.
        text_field: name of the text field in both datasets.
        concepts: target concept words (auto-detected from ref if None).
        provenance_field: field used for instance provenance.
        metadata_fields: extra fields to copy to instances.
        features: ladder fields to use.  If None, discovered from ref.

    Returns:
        ComparisonResult with SFI, SCI, SQI scores.
    """
    # Auto-detect concepts from reference if needed
    if concepts is None:
        concepts = auto_detect_concepts(ref_records, text_field)
    if not concepts:
        return ComparisonResult(0.0, 0.0, 0.0, {})

    # Extract instances from both datasets
    ref_instances = extract_concept_instances(
        ref_records,
        text_field=text_field,
        concepts=concepts,
        metadata_fields=metadata_fields,
        provenance_field=provenance_field,
    )
    test_instances = extract_concept_instances(
        test_records,
        text_field=text_field,
        concepts=concepts,
        metadata_fields=metadata_fields,
        provenance_field=provenance_field,
    )

    # Discover ladder from reference (or use provided features)
    if features is not None:
        ladder_fields = list(features)
    else:
        ladder_fields = _discover_ladder_fields(ref_instances)

    if not ladder_fields:
        # No structure discoverable -- degenerate case
        return ComparisonResult(0.0, 0.0, 0.0, {})

    # Build meaning addresses
    ref_addrs = meaning_addresses(ref_instances, ladder_fields)
    test_addrs = meaning_addresses(test_instances, ladder_fields)

    # Compute per-concept metrics
    per_concept: Dict[str, Dict[str, Any]] = {}
    sfi_sum, sci_sum, count = 0.0, 0.0, 0

    for concept in [c.lower() for c in concepts]:
        r_map = ref_addrs.get(concept, {})
        t_map = test_addrs.get(concept, {})

        r_count_list = [len(v) for v in r_map.values()]
        t_count_list = [len(v) for v in t_map.values()]

        n_ref = len(r_map)
        n_test = len(t_map)
        r_entropy = _shannon_entropy(r_count_list)
        t_entropy = _shannon_entropy(t_count_list)

        # SFI: does the test preserve the reference's structure?
        #   address ratio * entropy ratio, both capped at 1.0
        if n_ref == 0:
            sfi = 1.0 if n_test == 0 else 0.0
        else:
            addr_ratio = min(n_test / n_ref, 1.0)
            ent_ratio = min(t_entropy / r_entropy, 1.0) if r_entropy > 0 else (1.0 if t_entropy == 0 else 0.0)
            sfi = addr_ratio * ent_ratio

        # SCI: does the test avoid hallucinating new structure?
        #   inverse direction
        if n_test == 0:
            sci = 1.0 if n_ref == 0 else 0.0
        else:
            addr_ratio_inv = min(n_ref / n_test, 1.0)
            ent_ratio_inv = min(r_entropy / t_entropy, 1.0) if t_entropy > 0 else (1.0 if r_entropy == 0 else 0.0)
            sci = addr_ratio_inv * ent_ratio_inv

        sqi = sfi * sci

        per_concept[concept] = {
            "sfi": round(sfi, 6),
            "sci": round(sci, 6),
            "sqi": round(sqi, 6),
            "ref_addrs": n_ref,
            "test_addrs": n_test,
            "ref_entropy": round(r_entropy, 4),
            "test_entropy": round(t_entropy, 4),
        }

        sfi_sum += sfi
        sci_sum += sci
        count += 1

    # Overall: macro-average across concepts
    if count == 0:
        return ComparisonResult(0.0, 0.0, 0.0, per_concept)

    overall_sfi = sfi_sum / count
    overall_sci = sci_sum / count
    overall_sqi = overall_sfi * overall_sci

    return ComparisonResult(
        overall_sfi=round(overall_sfi, 6),
        overall_sci=round(overall_sci, 6),
        overall_sqi=round(overall_sqi, 6),
        per_concept=per_concept,
    )


def structural_fidelity(
    ref_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    **kwargs: Any,
) -> float:
    """Convenience alias: compare two datasets and return just the SQI float."""
    result = compare(ref_records, test_records, **kwargs)
    return result.overall_sqi
