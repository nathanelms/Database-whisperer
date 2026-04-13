"""Meaning-addressed retrieval for RAG pipelines.

Instead of embedding similarity, routes queries by structural meaning address.

Usage:
    import database_whisper as dw

    # Build the index
    index = dw.MeaningIndex(records, text_field="text", concepts=["positive", "negative"])

    # Query by sense
    results = index.query("positive", sense_hint={"paired_concept": "blood", "verb_class": "being"})

    # Or query by example: give it a context, it finds the matching address
    results = index.query_by_context("The blood culture was positive for E. coli")

    # Or just get all addresses for a concept
    addresses = index.addresses("positive")
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from .text import (
    auto_detect_concepts,
    extract_concept_instances,
    meaning_addresses,
    _clean,
    _nearest_verb_class,
    _syntactic_role,
    _paired_concept,
    _detect_contrast,
    _detect_equation,
    _detect_modality,
    _detect_voice,
    _detect_negation,
    _detect_clause_position,
    _detect_transitivity,
)
from .router import Router
from ._types import LadderRung


# All extractable text features, in the order extract_concept_instances produces them.
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


class MeaningIndex:
    """Meaning-addressed retrieval index.

    Builds a DW discriminator ladder over concept-in-context instances,
    then provides address-based retrieval — no embeddings needed.
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        text_field: str,
        concepts: Optional[List[str]] = None,
        provenance_field: str = "id",
        metadata_fields: Optional[List[str]] = None,
    ) -> None:
        self._text_field = text_field
        self._provenance_field = provenance_field
        self._metadata_fields = metadata_fields or []

        # Auto-detect concepts if not supplied
        if concepts is None:
            concepts = auto_detect_concepts(records, text_field)
        self._concepts = [c.lower() for c in concepts]
        self._concept_set = set(self._concepts)

        # Extract structured instances from text
        self._instances = extract_concept_instances(
            records,
            text_field=text_field,
            concepts=concepts,
            metadata_fields=metadata_fields,
            provenance_field=provenance_field,
        )

        # Discover the ladder using DW's own Router on the instances.
        # Identity field is "concept"; candidate fields are the text features.
        self._router = Router()
        candidate_fields = [
            f for f in _TEXT_FEATURES if any(f in inst for inst in self._instances)
        ]
        if self._instances:
            self._router.ingest(
                records=self._instances,
                identity_fields=["concept"],
                candidate_fields=candidate_fields,
                max_ladder_depth=min(5, len(candidate_fields)),
            )
        self._ladder_fields = self._router.ladder_fields if self._instances else []
        self._ladder_rungs = self._router.ladder if self._instances else []

        # Build meaning addresses using discovered ladder
        self._addresses = meaning_addresses(self._instances, self._ladder_fields)

        # Inverted index: (concept, address_tuple) -> [instances]
        self._inverted: Dict[str, Dict[tuple, List[Dict]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for concept, addr_map in self._addresses.items():
            for addr_tuple, insts in addr_map.items():
                self._inverted[concept][addr_tuple] = insts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        concept: str,
        sense_hint: Optional[Dict[str, str]] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Query instances by concept and optional sense hint.

        Args:
            concept: the target concept word.
            sense_hint: dict of {feature_name: value} to filter addresses.
            top_k: return only the top_k most populated matching addresses.

        Returns:
            List of dicts, each with keys: address, instances, count.
        """
        concept = concept.lower()
        addr_map = self._inverted.get(concept, {})
        if not addr_map:
            return []

        # Build result list
        results = []
        for addr_tuple, insts in addr_map.items():
            addr_dict = dict(zip(self._ladder_fields, addr_tuple))

            # Filter by sense_hint
            if sense_hint:
                match = all(
                    addr_dict.get(feat) == val for feat, val in sense_hint.items()
                )
                if not match:
                    continue

            results.append({
                "address": addr_dict,
                "instances": insts,
                "count": len(insts),
            })

        # Sort by count descending
        results.sort(key=lambda r: r["count"], reverse=True)

        if top_k is not None:
            results = results[:top_k]

        return results

    def query_by_context(
        self,
        context_text: str,
        concept: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Find instances sharing the same meaning-address as the query context.

        Featurizes the query text using the same text featurizer, finds
        the address, and returns instances at that address (exact match)
        or nearest addresses (partial match on ladder fields).

        Args:
            context_text: a sentence or passage containing the concept in use.
            concept: the concept word to look up.  If None, auto-detect from
                     known concepts by scanning the text.

        Returns:
            List of dicts with address, instances, count.
        """
        # Auto-detect concept from context if not given
        if concept is None:
            text_lower = context_text.lower()
            for c in self._concepts:
                if c in text_lower:
                    concept = c
                    break
            if concept is None:
                return []
        concept = concept.lower()

        # Featurize the query context using the same extractors
        words = context_text.split()
        pos = -1
        for i, w in enumerate(words):
            if _clean(w) == concept:
                pos = i
                break
        if pos == -1:
            for i, w in enumerate(words):
                if concept in _clean(w):
                    pos = i
                    break
        if pos == -1:
            return []

        query_features = {
            "verb_class": _nearest_verb_class(words, pos),
            "syntactic_role": _syntactic_role(words, pos),
            "paired_concept": _paired_concept(words, pos, concept, self._concept_set),
            "contrast": _detect_contrast(context_text, concept, self._concept_set),
            "equation": _detect_equation(context_text, concept, self._concept_set),
            "modality": _detect_modality(words, pos),
            "voice": _detect_voice(context_text),
            "negation": _detect_negation(words, pos),
            "clause_position": _detect_clause_position(words, pos),
            "transitivity": _detect_transitivity(words, pos),
        }

        # Build the query address from ladder fields
        query_addr = tuple(query_features.get(f, "none") for f in self._ladder_fields)

        addr_map = self._inverted.get(concept, {})
        if not addr_map:
            return []

        # Exact match first
        if query_addr in addr_map:
            insts = addr_map[query_addr]
            return [{
                "address": dict(zip(self._ladder_fields, query_addr)),
                "instances": insts,
                "count": len(insts),
            }]

        # Partial match: score addresses by how many ladder fields match
        scored = []
        for addr_tuple, insts in addr_map.items():
            matches = sum(
                1 for q, a in zip(query_addr, addr_tuple) if q == a
            )
            if matches > 0:
                scored.append((matches, addr_tuple, insts))
        scored.sort(key=lambda x: (-x[0], -len(x[2])))

        results = []
        for _score, addr_tuple, insts in scored:
            results.append({
                "address": dict(zip(self._ladder_fields, addr_tuple)),
                "instances": insts,
                "count": len(insts),
            })
        return results

    def addresses(self, concept: str) -> List[Dict[str, Any]]:
        """Return all distinct addresses for a concept with instance counts.

        Sorted by count descending.
        """
        concept = concept.lower()
        addr_map = self._inverted.get(concept, {})
        results = []
        for addr_tuple, insts in addr_map.items():
            results.append({
                "address": dict(zip(self._ladder_fields, addr_tuple)),
                "count": len(insts),
            })
        results.sort(key=lambda r: r["count"], reverse=True)
        return results

    def ladder(self) -> List[LadderRung]:
        """Return the discovered ladder (field order and reduction rates)."""
        return list(self._ladder_rungs)

    def stats(self) -> Dict[str, Any]:
        """Summary statistics for the index."""
        total_instances = len(self._instances)
        total_addresses = sum(
            len(addrs) for addrs in self._inverted.values()
        )
        concepts_covered = len(self._inverted)
        avg_per_addr = (
            total_instances / total_addresses if total_addresses > 0 else 0.0
        )
        return {
            "total_instances": total_instances,
            "total_addresses": total_addresses,
            "concepts_covered": concepts_covered,
            "avg_instances_per_address": round(avg_per_addr, 2),
            "ladder_fields": list(self._ladder_fields),
            "ladder_depth": len(self._ladder_fields),
        }


def retrieve(
    records: List[Dict[str, Any]],
    text_field: str,
    concepts: List[str],
    query_concept: str,
    sense_hint: Optional[Dict[str, str]] = None,
    top_k: Optional[int] = None,
    provenance_field: str = "id",
    metadata_fields: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """One-liner convenience: build index, run query, return results.

    Args:
        records: list of dicts with at least text_field.
        text_field: name of the text field.
        concepts: target concept words.
        query_concept: the concept to query.
        sense_hint: optional dict of {feature: value} to narrow results.
        top_k: return only top_k most populated addresses.
        provenance_field: field used for instance provenance.
        metadata_fields: extra fields to copy to instances.

    Returns:
        List of dicts with address, instances, count.
    """
    index = MeaningIndex(
        records,
        text_field=text_field,
        concepts=concepts,
        provenance_field=provenance_field,
        metadata_fields=metadata_fields,
    )
    return index.query(query_concept, sense_hint=sense_hint, top_k=top_k)
