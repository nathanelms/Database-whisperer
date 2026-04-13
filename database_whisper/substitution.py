"""Meaning audit for Database Whisper.

Once concepts have addresses, they stop being "words" and become
addressed instances — specific meanings in specific contexts.
This module provides four meaning quality scores:

    1. CONFUSION  — does meaning drift within documents? (document level)
    2. COMPLETENESS — are all meanings represented?     (dataset level)
    3. PREDICTABILITY — does context disambiguate?       (sentence level)
    4. HAZARD — which words will cause hallucination?    (word level)

Before scoring, addresses are COLLAPSED: addresses with similar
substitution neighborhoods are merged into resolved meanings.
15,000 raw addresses might collapse to 150 resolved meanings.
The scores operate on resolved meanings, not syntactic noise.

Three-layer architecture (operators/meaning/expression):
    classify_features() classifies each address feature by role.
    collapse_addresses() respects layer rules when roles are provided.
    meaning_audit() can auto-classify with auto_classify=True.

Diagnostic tools (the toolbox):
    neighborhood_entropy()  — specialist vs generalist profile per concept
    mutual_exclusion()      — concept pairs that never share an address
    absence_patterns()      — expected concepts missing from specific docs
    bridging_score()        — concepts that span multiple meaning domains
    diagnose()              — reads audit scores, selects relevant tools

Usage:
    import database_whisper as dw

    instances = dw.extract_concept_instances(records, text_field="text", concepts=[...])
    profile = dw.profile_records(instances, identity_fields=["concept"], ...)
    addresses = dw.meaning_addresses(instances, ladder_fields=profile.ladder_fields)

    # Full meaning audit
    audit = dw.meaning_audit(addresses)
    print(audit)

    # Auto-diagnosis: audit → select tools → run them → report
    diagnosis = dw.diagnose(addresses)
    print(diagnosis)

    # Individual scores
    neighborhoods = dw.neighborhoods(addresses)
    coverage = dw.meaning_coverage(addresses)
    resolution = dw.meaning_resolution(addresses)
"""

import math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any

# Type aliases
Address = tuple
AddressMap = Dict[str, Dict[Address, List[Dict]]]

# Three-layer feature roles
OPERATOR = "operator"
MEANING = "meaning"
EXPRESSION = "expression"


# ───────────────────────────────────────────────────────────────────
# Core: Neighborhood discovery
# ───────────────────────────────────────────────────────────────────

class Neighbor:
    """A concept-at-address with a surprise score."""
    __slots__ = ("concept", "address", "count", "surprise")

    def __init__(self, concept: str, address: Address, count: int, surprise: float):
        self.concept = concept
        self.address = address
        self.count = count
        self.surprise = surprise

    def __repr__(self):
        addr_str = ".".join(str(a) for a in self.address)
        return f"{self.concept}@{addr_str}({self.count}, surprise={self.surprise:.2f})"


class Neighborhood:
    """The neighborhood of a concept at a specific address.

    Contains the nearby addressed instances ranked by surprise.
    High surprise = meaningful neighbor (like "negative" near "positive").
    Low surprise = expected co-resident (like "patient" near everything).
    """

    def __init__(self, concept: str, address: Address, instance_count: int,
                 neighbors: List[Neighbor]):
        self.concept = concept
        self.address = address
        self.instance_count = instance_count
        # Sort by surprise descending — most meaningful first
        self.neighbors = sorted(neighbors, key=lambda n: -n.surprise)

    def top(self, n: int = 5) -> List[Neighbor]:
        """Most surprising neighbors — the meaning signal."""
        return self.neighbors[:n]

    def substitutes(self, min_surprise: float = 0.0) -> List[Neighbor]:
        """Neighbors surprising enough to be genuine substitutes."""
        return [n for n in self.neighbors if n.surprise > min_surprise]

    def __repr__(self):
        addr_str = ".".join(str(a) for a in self.address)
        top3 = ", ".join(str(n) for n in self.top(3))
        return f"Neighborhood({self.concept}@{addr_str}, n={self.instance_count}, top=[{top3}])"


def neighborhoods(
    addresses: AddressMap,
    provenance_field: str = "_provenance",
    grain: str = "auto",
) -> Dict[str, Dict[Address, Neighborhood]]:
    """Discover substitution neighborhoods for all addressed instances.

    For each concept at each address, finds other addressed instances
    that are nearby — in the same documents, sentences, or windows.
    Ranks neighbors by surprise (PMI): how much more often they appear
    nearby than their base rate predicts.

    Once a concept has an address, it's no longer "the word positive" —
    it's a specific meaning-instance. Addressed instances don't swamp
    because high-frequency concepts are split across their addresses.

    Args:
        addresses: output of meaning_addresses()
            {concept: {address_tuple: [instances]}}
        provenance_field: field linking instances to source documents
        grain: co-occurrence granularity.
            "document" — two instances are neighbors if same document
            "context" — two instances are neighbors if overlapping context windows
            "auto" — use document if provenance available, else context

    Returns:
        {concept: {address: Neighborhood}}
    """
    # ── Step 1: Build the addressed instance inventory ──
    # Each instance becomes (concept, address) — no longer just a word

    # Index: document → list of (concept, address) pairs
    doc_to_addressed = defaultdict(list)

    # Count: how many total instances per (concept, address)
    addr_instance_counts = defaultdict(lambda: defaultdict(int))

    # Total instances for base rate
    total_instances = 0

    for concept, addr_dict in addresses.items():
        for addr, insts in addr_dict.items():
            addr_instance_counts[concept][addr] = len(insts)
            total_instances += len(insts)

            for inst in insts:
                doc_id = inst.get(provenance_field, inst.get("_context", "unknown"))
                doc_to_addressed[doc_id].append((concept, addr))

    # ── Step 2: Count co-occurrences between addressed instances ──
    # Two addressed instances are neighbors if they appear in the same doc
    # Key: (concept1, addr1) → Counter of (concept2, addr2) → count

    cooccurrence = defaultdict(Counter)

    for doc_id, addressed_list in doc_to_addressed.items():
        # All pairs of addressed instances in this document
        for i in range(len(addressed_list)):
            for j in range(len(addressed_list)):
                if i == j:
                    continue
                c1, a1 = addressed_list[i]
                c2, a2 = addressed_list[j]
                # Don't count self-concept co-occurrence at same address
                if c1 == c2 and a1 == a2:
                    continue
                cooccurrence[(c1, a1)][(c2, a2)] += 1

    # ── Step 3: Compute surprise (PMI) for each neighbor ──
    # PMI = log2(P(a,b) / (P(a) * P(b)))
    # P(a) = count of instances at this address / total
    # P(a,b) = co-occurrence count / total possible pairs
    # High PMI = surprisingly common neighbor = meaningful

    n_docs = len(doc_to_addressed) or 1

    # Base rate: how many documents contain each (concept, address)?
    addr_doc_count = defaultdict(int)
    for doc_id, addressed_list in doc_to_addressed.items():
        seen = set()
        for ca in addressed_list:
            if ca not in seen:
                addr_doc_count[ca] += 1
                seen.add(ca)

    result = {}
    for concept, addr_dict in addresses.items():
        result[concept] = {}

        for addr, insts in addr_dict.items():
            my_key = (concept, addr)
            my_doc_freq = addr_doc_count[my_key] / n_docs

            neighbors = []
            for (other_concept, other_addr), co_count in cooccurrence[my_key].items():
                other_doc_freq = addr_doc_count[(other_concept, other_addr)] / n_docs

                # Co-occurrence rate (fraction of my docs that also contain this neighbor)
                co_rate = co_count / max(addr_instance_counts[concept][addr], 1)

                # Expected rate if independent
                expected = other_doc_freq

                # Surprise: how much more often than expected?
                # Using log ratio, floored at 0 (negative = less than expected = not interesting)
                if expected > 0 and co_rate > 0:
                    surprise = math.log2(co_rate / expected)
                else:
                    surprise = 0.0

                neighbors.append(Neighbor(
                    concept=other_concept,
                    address=other_addr,
                    count=co_count,
                    surprise=surprise,
                ))

            result[concept][addr] = Neighborhood(
                concept=concept,
                address=addr,
                instance_count=len(insts),
                neighbors=neighbors,
            )

    return result


# ───────────────────────────────────────────────────────────────────
# Three-layer feature classification
# ───────────────────────────────────────────────────────────────────

# Known operator features — binary truth modifiers that co-occurrence
# can't detect (they don't change the topic, they flip the truth value).
# These are the "queens" — they cross everything.
KNOWN_OPERATORS = {"negation", "modality"}


class FeatureRoles:
    """Three-layer classification of address features.

    Every feature in a meaning address plays one of three roles:

        OPERATORS    negation, modality, tense, conditionality
                     Binary switches that flip truth values.
                     Rule: NEVER collapse across these.

        MEANING      paired_concept, equation, contrast
                     Define what you're talking about.
                     Rule: Collapse by neighborhood similarity.

        EXPRESSION   verb_class, clause_position, voice, transitivity
                     How you write it — stylistic variation.
                     Rule: Collapse freely (same meaning, different writing).

    The classification is data-driven: for each feature, we group
    addresses by that feature's value and compare substitution
    neighborhoods within-group vs across-group (Jaccard).
    If changing the feature changes neighborhoods → MEANING.
    If it doesn't → EXPRESSION.
    Operators are detected by name (known set) or by being binary
    features with low Jaccard ratio (they're invisible to co-occurrence).
    """

    def __init__(self, operators: List[str], meaning: List[str],
                 expression: List[str], ratios: Optional[Dict[str, float]] = None):
        self.operators = operators
        self.meaning = meaning
        self.expression = expression
        self.ratios = ratios or {}  # feature → within/across Jaccard ratio

    def role(self, feature: str) -> str:
        """Get the role of a feature."""
        if feature in self.operators:
            return OPERATOR
        if feature in self.meaning:
            return MEANING
        if feature in self.expression:
            return EXPRESSION
        return EXPRESSION  # default: treat unknown features as expression

    def __repr__(self):
        return (f"FeatureRoles(\n"
                f"  operators={self.operators},\n"
                f"  meaning={self.meaning},\n"
                f"  expression={self.expression}\n)")

    def __str__(self):
        lines = ["Feature Roles (Three-Layer Architecture):", ""]
        lines.append(f"  OPERATORS (never collapse):  {', '.join(self.operators) or '(none)'}")
        lines.append(f"  MEANING   (collapse by sim): {', '.join(self.meaning) or '(none)'}")
        lines.append(f"  EXPRESSION (collapse freely): {', '.join(self.expression) or '(none)'}")
        if self.ratios:
            lines.append("")
            lines.append(f"  {'Feature':<20s}  {'Ratio':>7s}  {'Role':>12s}")
            lines.append(f"  {'─' * 20}  {'─' * 7}  {'─' * 12}")
            for feat in sorted(self.ratios, key=lambda f: -self.ratios[f]):
                lines.append(f"  {feat:<20s}  {self.ratios[feat]:>7.2f}  {self.role(feat):>12s}")
        return "\n".join(lines)


def _jaccard_top_neighbors(nb1, nb2, top_n: int = 10) -> float:
    """Jaccard similarity of top neighbor concept sets between two neighborhoods."""
    if nb1 is None or nb2 is None:
        return 0.0
    top1 = set(n.concept for n in nb1.top(top_n) if n.surprise > 0)
    top2 = set(n.concept for n in nb2.top(top_n) if n.surprise > 0)
    if not top1 and not top2:
        return 1.0
    if not top1 or not top2:
        return 0.0
    return len(top1 & top2) / len(top1 | top2)


def classify_features(
    addresses: AddressMap,
    ladder_fields: List[str],
    nbhoods: Optional[Dict[str, Dict[Address, "Neighborhood"]]] = None,
    provenance_field: str = "_provenance",
    meaning_threshold: float = 1.2,
    operator_names: Optional[set] = None,
    top_n: int = 10,
) -> FeatureRoles:
    """Classify each ladder field as OPERATOR, MEANING, or EXPRESSION.

    For each feature, groups addresses by that feature's value and
    computes Jaccard similarity of substitution neighborhoods:
      - WITHIN groups (same feature value)
      - ACROSS groups (different feature values)

    If within >> across (ratio > meaning_threshold): changing this feature
    changes what concepts appear nearby → MEANING feature.

    If within ≈ across (ratio ≤ meaning_threshold): changing this feature
    doesn't change neighbors → EXPRESSION feature.

    OPERATORS are identified by name from a known set (negation, modality).
    They score low on the ratio test because co-occurrence can't see them —
    negated and affirmed instances live in the same documents near the same
    concepts. But they flip truth values, so collapsing across them is wrong.

    Args:
        addresses: output of meaning_addresses()
        ladder_fields: ordered list of feature names matching address positions
        nbhoods: pre-computed neighborhoods (optional)
        provenance_field: field linking instances to documents
        meaning_threshold: ratio above which a feature is classified as MEANING
        operator_names: set of feature names to force-classify as OPERATOR.
            Default: {"negation", "modality"}
        top_n: number of top neighbors for Jaccard computation

    Returns:
        FeatureRoles with classification and per-feature ratios
    """
    if operator_names is None:
        operator_names = KNOWN_OPERATORS

    if nbhoods is None:
        nbhoods = neighborhoods(addresses, provenance_field=provenance_field)

    ratios = {}

    for feat_idx, feature in enumerate(ladder_fields):
        within_jaccards = []
        across_jaccards = []

        for concept, addr_dict in addresses.items():
            if concept not in nbhoods:
                continue

            # Group addresses by this feature's value
            groups = defaultdict(list)
            for addr in addr_dict:
                if feat_idx < len(addr):
                    groups[addr[feat_idx]].append(addr)

            if len(groups) < 2:
                continue

            # Within-group: addresses with SAME feature value
            for feat_val, group_addrs in groups.items():
                if len(group_addrs) < 2:
                    continue
                for i in range(min(len(group_addrs), 10)):
                    for j in range(i + 1, min(len(group_addrs), 10)):
                        nb1 = nbhoods[concept].get(group_addrs[i])
                        nb2 = nbhoods[concept].get(group_addrs[j])
                        if nb1 and nb2:
                            within_jaccards.append(
                                _jaccard_top_neighbors(nb1, nb2, top_n))

            # Across-group: addresses with DIFFERENT feature values
            group_keys = list(groups.keys())
            for gi in range(min(len(group_keys), 5)):
                for gj in range(gi + 1, min(len(group_keys), 5)):
                    addrs_i = groups[group_keys[gi]][:5]
                    addrs_j = groups[group_keys[gj]][:5]
                    for a1 in addrs_i:
                        for a2 in addrs_j:
                            nb1 = nbhoods[concept].get(a1)
                            nb2 = nbhoods[concept].get(a2)
                            if nb1 and nb2:
                                across_jaccards.append(
                                    _jaccard_top_neighbors(nb1, nb2, top_n))

        avg_within = sum(within_jaccards) / len(within_jaccards) if within_jaccards else 0
        avg_across = sum(across_jaccards) / len(across_jaccards) if across_jaccards else 0
        ratios[feature] = avg_within / avg_across if avg_across > 0 else 1.0

    # Classify
    operators = []
    meaning = []
    expression = []

    for feature in ladder_fields:
        if feature in operator_names:
            # Known operators — force-classify regardless of ratio
            operators.append(feature)
        elif ratios.get(feature, 1.0) > meaning_threshold:
            meaning.append(feature)
        else:
            expression.append(feature)

    return FeatureRoles(
        operators=operators,
        meaning=meaning,
        expression=expression,
        ratios=ratios,
    )


# ───────────────────────────────────────────────────────────────────
# Meaning coverage map
# ───────────────────────────────────────────────────────────────────

class CoverageMap:
    """What meaning does a dataset contain?

    For each concept, lists all addresses and their instance counts.
    Computes coverage statistics: how many addresses, how concentrated,
    where the gaps are.
    """

    def __init__(self, addresses: AddressMap):
        self.addresses = addresses
        self._compute_stats()

    def _compute_stats(self):
        self.concept_stats = {}
        self.total_instances = 0
        self.total_addresses = 0

        for concept, addr_dict in self.addresses.items():
            counts = [len(insts) for insts in addr_dict.values()]
            total = sum(counts)
            n_addrs = len(counts)
            self.total_instances += total
            self.total_addresses += n_addrs

            # Entropy of address distribution (how spread out?)
            entropy = 0.0
            if total > 0:
                for c in counts:
                    if c > 0:
                        p = c / total
                        entropy -= p * math.log2(p)

            # Max entropy (uniform distribution)
            max_entropy = math.log2(n_addrs) if n_addrs > 1 else 0

            # Concentration: what fraction of instances are at top address?
            top_frac = max(counts) / total if total > 0 else 0

            # Effective addresses (2^entropy) — how many addresses
            # are "really" being used vs theoretical max
            effective = 2 ** entropy if entropy > 0 else 1

            self.concept_stats[concept] = {
                "n_addresses": n_addrs,
                "n_instances": total,
                "entropy": entropy,
                "max_entropy": max_entropy,
                "evenness": entropy / max_entropy if max_entropy > 0 else 1.0,
                "top_concentration": top_frac,
                "effective_addresses": effective,
            }

    def gaps(self, min_instances: int = 5) -> Dict[str, List[Address]]:
        """Addresses with fewer than min_instances — underrepresented meanings."""
        result = {}
        for concept, addr_dict in self.addresses.items():
            sparse = [addr for addr, insts in addr_dict.items()
                      if len(insts) < min_instances]
            if sparse:
                result[concept] = sparse
        return result

    def redundancies(self, top_frac: float = 0.5) -> Dict[str, Address]:
        """Concepts where > top_frac of instances are at one address — dominated meanings."""
        result = {}
        for concept, addr_dict in self.addresses.items():
            total = sum(len(insts) for insts in addr_dict.values())
            if total == 0:
                continue
            for addr, insts in addr_dict.items():
                if len(insts) / total > top_frac:
                    result[concept] = addr
                    break
        return result

    def __str__(self):
        lines = [
            f"MeaningCoverage: {self.total_instances} instances across "
            f"{self.total_addresses} addresses in {len(self.addresses)} concepts",
            "",
            f"  {'Concept':<15s}  {'Addrs':>6s}  {'Inst':>6s}  {'Entropy':>8s}  "
            f"{'Evenness':>9s}  {'TopConc':>8s}  {'Effective':>10s}",
        ]
        for concept in sorted(self.concept_stats.keys()):
            s = self.concept_stats[concept]
            lines.append(
                f"  {concept:<15s}  {s['n_addresses']:>6d}  {s['n_instances']:>6d}  "
                f"{s['entropy']:>8.2f}  {s['evenness']:>9.3f}  "
                f"{s['top_concentration']:>8.3f}  {s['effective_addresses']:>10.1f}"
            )
        return "\n".join(lines)


def meaning_coverage(addresses: AddressMap) -> CoverageMap:
    """Build a meaning coverage map for a dataset.

    Shows what meaning the dataset contains: how many addresses per concept,
    how evenly distributed (entropy), where the gaps and redundancies are.

    Args:
        addresses: output of meaning_addresses()

    Returns:
        CoverageMap with statistics and gap/redundancy detection
    """
    return CoverageMap(addresses)


# ───────────────────────────────────────────────────────────────────
# Meaning resolution score
# ───────────────────────────────────────────────────────────────────

def meaning_resolution(
    addresses: AddressMap,
    provenance_field: str = "_provenance",
) -> Dict[str, Any]:
    """How consistently do concepts resolve to specific addresses in this dataset?

    High resolution = concepts consistently resolve to 1-2 addresses per document.
    Low resolution = concepts scatter across many addresses within single documents.

    A document where "positive" means lab-result every time: high resolution.
    A document where "positive" shifts sense mid-text: low resolution.

    Args:
        addresses: output of meaning_addresses()
        provenance_field: field linking instances to source documents

    Returns:
        Dict with overall score and per-concept breakdown
    """
    # For each document, for each concept:
    # how many distinct addresses does the concept occupy?

    # Build: doc → concept → set of addresses
    doc_concept_addrs = defaultdict(lambda: defaultdict(set))

    for concept, addr_dict in addresses.items():
        for addr, insts in addr_dict.items():
            for inst in insts:
                doc_id = inst.get(provenance_field, "unknown")
                doc_concept_addrs[doc_id][concept].add(addr)

    # Per-concept resolution: average addresses-per-document
    concept_resolution = {}
    for concept in addresses:
        docs_with_concept = [
            addrs for doc_id, concepts in doc_concept_addrs.items()
            for c, addrs in concepts.items()
            if c == concept
        ]
        if not docs_with_concept:
            continue

        avg_addrs = sum(len(a) for a in docs_with_concept) / len(docs_with_concept)
        max_addrs = max(len(a) for a in docs_with_concept)

        # Resolution score: 1.0 = always resolves to exactly 1 address
        # Lower = more scattered. Score = 1 / avg_addresses_per_doc
        resolution = 1.0 / avg_addrs if avg_addrs > 0 else 0

        concept_resolution[concept] = {
            "avg_addresses_per_doc": avg_addrs,
            "max_addresses_in_any_doc": max_addrs,
            "resolution": resolution,
            "n_docs": len(docs_with_concept),
        }

    # Overall resolution: weighted average by instance count
    total_weight = 0
    weighted_sum = 0
    for concept, stats in concept_resolution.items():
        weight = stats["n_docs"]
        weighted_sum += stats["resolution"] * weight
        total_weight += weight

    overall = weighted_sum / total_weight if total_weight > 0 else 0

    return {
        "overall_resolution": overall,
        "per_concept": concept_resolution,
    }


# ───────────────────────────────────────────────────────────────────
# Address collapse — merge syntactic noise into resolved meanings
# ───────────────────────────────────────────────────────────────────

def collapse_addresses(
    addresses: AddressMap,
    nbhoods: Optional[Dict[str, Dict[Address, "Neighborhood"]]] = None,
    similarity_threshold: float = 0.5,
    provenance_field: str = "_provenance",
    feature_roles: Optional["FeatureRoles"] = None,
    ladder_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Collapse raw addresses into resolved meanings.

    Two addresses for the same concept are merged if their substitution
    neighborhoods are similar (high Jaccard overlap of top neighbors).
    This separates genuine meaning distinctions from syntactic noise.

    15,000 raw addresses → maybe 150 resolved meanings.

    Three-layer collapse (when feature_roles + ladder_fields provided):
        1. OPERATOR features: NEVER collapse across different values.
           Two addresses that differ on negation stay separate, period.
        2. EXPRESSION features: Collapse freely — strip these dimensions
           first, merging all instances that differ only on expression.
        3. MEANING features: Collapse by neighborhood similarity
           (same Jaccard greedy merge as before, but only within
           operator-protected groups).

    Without feature_roles: falls back to flat collapse (original behavior).

    Args:
        addresses: output of meaning_addresses()
        nbhoods: pre-computed neighborhoods (optional, will compute if None)
        similarity_threshold: Jaccard threshold for merging (0-1)
        provenance_field: field linking instances to documents
        feature_roles: three-layer classification (from classify_features)
        ladder_fields: ordered feature names matching address positions.
            Required when feature_roles is provided.

    Returns:
        Dict with:
            resolved: {concept: {meaning_id: [merged_addresses]}}
            mapping: {concept: {raw_address: meaning_id}}
            resolved_addresses: AddressMap with merged instances
            stats: summary statistics (includes layer_stats when layered)
    """
    if nbhoods is None:
        nbhoods = neighborhoods(addresses, provenance_field=provenance_field)

    # ── Layered collapse path ──
    if feature_roles is not None and ladder_fields is not None:
        return _layered_collapse(
            addresses, nbhoods, similarity_threshold,
            feature_roles, ladder_fields,
        )

    # ── Original flat collapse path (backward compatible) ──
    return _flat_collapse(addresses, nbhoods, similarity_threshold)


def _flat_collapse(
    addresses: AddressMap,
    nbhoods: Dict[str, Dict[Address, "Neighborhood"]],
    similarity_threshold: float,
) -> Dict[str, Any]:
    """Original flat collapse — all features treated equally."""

    resolved = {}
    mapping = {}
    resolved_addresses = {}

    total_raw = 0
    total_resolved = 0

    for concept, addr_dict in addresses.items():
        if concept not in nbhoods:
            continue

        # Get top-N neighbor signatures for each address
        addr_signatures = {}
        for addr in addr_dict:
            if addr in nbhoods[concept]:
                nbhood = nbhoods[concept][addr]
                top_n = nbhood.top(10)
                sig = frozenset(n.concept for n in top_n if n.surprise > 0)
                addr_signatures[addr] = sig
            else:
                addr_signatures[addr] = frozenset()

        # Greedy merge: group addresses with similar signatures
        sorted_addrs = sorted(addr_dict.keys(), key=lambda a: -len(addr_dict[a]))
        groups = []
        addr_to_group = {}

        for addr in sorted_addrs:
            sig = addr_signatures[addr]
            merged = False

            for g_idx, (g_sig, g_addrs) in enumerate(groups):
                if g_sig or sig:
                    intersection = len(sig & g_sig)
                    union = len(sig | g_sig)
                    jaccard = intersection / union if union > 0 else 0
                else:
                    jaccard = 1.0

                if jaccard >= similarity_threshold:
                    g_addrs.append(addr)
                    groups[g_idx] = (g_sig | sig, g_addrs)
                    addr_to_group[addr] = g_idx
                    merged = True
                    break

            if not merged:
                addr_to_group[addr] = len(groups)
                groups.append((sig, [addr]))

        # Build resolved output
        concept_resolved = {}
        concept_mapping = {}
        concept_resolved_addrs = {}

        for meaning_id, (sig, group_addrs) in enumerate(groups):
            concept_resolved[meaning_id] = group_addrs
            merged_instances = []
            for addr in group_addrs:
                merged_instances.extend(addr_dict[addr])
            rep_addr = max(group_addrs, key=lambda a: len(addr_dict[a]))
            concept_resolved_addrs[rep_addr] = merged_instances

            for addr in group_addrs:
                concept_mapping[addr] = meaning_id

        resolved[concept] = concept_resolved
        mapping[concept] = concept_mapping
        resolved_addresses[concept] = concept_resolved_addrs

        total_raw += len(addr_dict)
        total_resolved += len(groups)

    return {
        "resolved": resolved,
        "mapping": mapping,
        "resolved_addresses": resolved_addresses,
        "stats": {
            "raw_addresses": total_raw,
            "resolved_meanings": total_resolved,
            "collapse_ratio": total_resolved / total_raw if total_raw > 0 else 1.0,
        },
    }


def _layered_collapse(
    addresses: AddressMap,
    nbhoods: Dict[str, Dict[Address, "Neighborhood"]],
    similarity_threshold: float,
    feature_roles: "FeatureRoles",
    ladder_fields: List[str],
) -> Dict[str, Any]:
    """Three-layer collapse: operators protect, expression merges, meaning resolves.

    Phase 1 — EXPRESSION COLLAPSE:
        Strip expression dimensions. All addresses that differ only on
        expression features merge into the same bucket.

    Phase 2 — OPERATOR PROTECTION:
        Within each expression-collapsed bucket, addresses that differ
        on operator features are kept separate. Operators are walls.

    Phase 3 — MEANING COLLAPSE:
        Within each operator-protected partition, run the standard
        greedy Jaccard merge on substitution neighborhoods.

    The result: fewer false meanings from expression noise, zero lost
    meanings from operator collapse, and honest neighborhood-based
    resolution on what remains.
    """
    # Identify feature indices for each layer
    operator_idx = [i for i, f in enumerate(ladder_fields)
                    if feature_roles.role(f) == OPERATOR]
    meaning_idx = [i for i, f in enumerate(ladder_fields)
                   if feature_roles.role(f) == MEANING]
    expression_idx = [i for i, f in enumerate(ladder_fields)
                      if feature_roles.role(f) == EXPRESSION]

    resolved = {}
    mapping = {}
    resolved_addresses = {}

    total_raw = 0
    total_resolved = 0
    total_expression_collapsed = 0
    total_operator_partitions = 0

    for concept, addr_dict in addresses.items():
        if concept not in nbhoods:
            continue

        total_raw += len(addr_dict)

        # ── Phase 1: Expression collapse ──
        # Group addresses by their operator+meaning components only.
        # Addresses that differ only on expression features merge here.
        def _non_expression_key(addr):
            """Extract operator+meaning components of an address."""
            parts = []
            for i in range(len(addr)):
                if i < len(ladder_fields) and i not in expression_idx:
                    parts.append(addr[i])
                elif i >= len(ladder_fields):
                    # Extra dimensions beyond ladder — keep them (safe default)
                    parts.append(addr[i])
            return tuple(parts)

        expr_groups = defaultdict(list)  # non-expression key → [full addrs]
        for addr in addr_dict:
            key = _non_expression_key(addr)
            expr_groups[key].append(addr)

        total_expression_collapsed += len(addr_dict) - len(expr_groups)

        # ── Phase 2: Operator protection ──
        # Within expression-collapsed groups, partition by operator values.
        # Addresses with different operator values NEVER merge.
        def _operator_key(addr):
            """Extract operator components of an address."""
            return tuple(addr[i] for i in operator_idx if i < len(addr))

        # Build operator-protected partitions
        # Each partition = set of addresses that share ALL operator values
        # (and may have been expression-collapsed together)
        partitions = defaultdict(list)  # (concept, operator_key) → [full addrs]

        for non_expr_key, addrs_in_group in expr_groups.items():
            # All addresses in this group share operator+meaning values,
            # but we re-partition by operator key to be explicit
            for addr in addrs_in_group:
                op_key = _operator_key(addr)
                partitions[op_key].append(addr)

        total_operator_partitions += len(partitions)

        # ── Phase 3: Meaning collapse within each operator partition ──
        # Standard greedy Jaccard merge, but only within the partition.
        concept_resolved = {}
        concept_mapping = {}
        concept_resolved_addrs = {}
        meaning_id_counter = 0

        for op_key, partition_addrs in partitions.items():
            # Get neighborhood signatures for addresses in this partition
            addr_signatures = {}
            for addr in partition_addrs:
                if addr in nbhoods[concept]:
                    nbhood = nbhoods[concept][addr]
                    top_n = nbhood.top(10)
                    sig = frozenset(n.concept for n in top_n if n.surprise > 0)
                    addr_signatures[addr] = sig
                else:
                    addr_signatures[addr] = frozenset()

            # Greedy merge by neighborhood similarity
            sorted_addrs = sorted(partition_addrs,
                                  key=lambda a: -len(addr_dict[a]))
            groups = []

            for addr in sorted_addrs:
                sig = addr_signatures[addr]
                merged = False

                for g_idx, (g_sig, g_addrs) in enumerate(groups):
                    if g_sig or sig:
                        intersection = len(sig & g_sig)
                        union = len(sig | g_sig)
                        jaccard = intersection / union if union > 0 else 0
                    else:
                        jaccard = 1.0

                    if jaccard >= similarity_threshold:
                        g_addrs.append(addr)
                        groups[g_idx] = (g_sig | sig, g_addrs)
                        merged = True
                        break

                if not merged:
                    groups.append((sig, [addr]))

            # Assign meaning IDs for this partition
            for sig, group_addrs in groups:
                mid = meaning_id_counter
                meaning_id_counter += 1

                concept_resolved[mid] = group_addrs
                merged_instances = []
                for addr in group_addrs:
                    merged_instances.extend(addr_dict[addr])
                rep_addr = max(group_addrs, key=lambda a: len(addr_dict[a]))
                concept_resolved_addrs[rep_addr] = merged_instances

                for addr in group_addrs:
                    concept_mapping[addr] = mid

        resolved[concept] = concept_resolved
        mapping[concept] = concept_mapping
        resolved_addresses[concept] = concept_resolved_addrs
        total_resolved += meaning_id_counter

    return {
        "resolved": resolved,
        "mapping": mapping,
        "resolved_addresses": resolved_addresses,
        "stats": {
            "raw_addresses": total_raw,
            "resolved_meanings": total_resolved,
            "collapse_ratio": total_resolved / total_raw if total_raw > 0 else 1.0,
            "expression_collapsed": total_expression_collapsed,
            "operator_partitions": total_operator_partitions,
            "layers": {
                "operators": [ladder_fields[i] for i in operator_idx],
                "meaning": [ladder_fields[i] for i in meaning_idx],
                "expression": [ladder_fields[i] for i in expression_idx],
            },
        },
    }


# ───────────────────────────────────────────────────────────────────
# The Four Scores — full meaning audit
# ───────────────────────────────────────────────────────────────────

class MeaningAudit:
    """Full meaning quality audit for a dataset.

    Four scores at four grains:
        confusion     — does meaning drift?        (document level)
        completeness  — are all meanings present?   (dataset level)
        predictability — does context disambiguate? (sentence level)
        hazard        — which words cause trouble?  (word level)
    """

    def __init__(self, confusion, completeness, predictability, hazard,
                 collapse_stats, per_concept):
        self.confusion = confusion            # 0-1, lower = safer
        self.completeness = completeness      # 0-1, higher = richer
        self.predictability = predictability  # 0-1, higher = cleaner
        self.hazard = hazard                  # list of (concept, score) ranked
        self.collapse_stats = collapse_stats
        self.per_concept = per_concept

    def __str__(self):
        lines = [
            "=" * 70,
            "MEANING AUDIT",
            "=" * 70,
            "",
            f"  Addresses: {self.collapse_stats['raw_addresses']} raw → "
            f"{self.collapse_stats['resolved_meanings']} resolved "
            f"({self.collapse_stats['collapse_ratio']:.1%} of raw)",
        ]
        # Show layer info if three-layer collapse was used
        if "layers" in self.collapse_stats:
            layers = self.collapse_stats["layers"]
            lines.append(f"  Three-layer collapse:")
            lines.append(f"    Operators (protected):  {', '.join(layers['operators']) or '(none)'}")
            lines.append(f"    Meaning (by sim):       {', '.join(layers['meaning']) or '(none)'}")
            lines.append(f"    Expression (free merge): {', '.join(layers['expression']) or '(none)'}")
            lines.append(f"    Expression variants collapsed: "
                         f"{self.collapse_stats.get('expression_collapsed', 0)}")
        lines += [
            f"  CONFUSION:      {self.confusion:.3f}  (0=safe, 1=dangerous — meaning drift)",
            f"  COMPLETENESS:   {self.completeness:.3f}  (0=blind spots, 1=all meanings present)",
            f"  PREDICTABILITY: {self.predictability:.3f}  (0=noisy, 1=context always disambiguates)",
            "",
            "  HAZARD INDEX (most dangerous words):",
        ]
        for concept, score in self.hazard[:10]:
            detail = self.per_concept.get(concept, {})
            n_meanings = detail.get("resolved_meanings", "?")
            avg_drift = detail.get("avg_meanings_per_doc", "?")
            lines.append(
                f"    {concept:<15s}  hazard={score:.3f}  "
                f"({n_meanings} meanings, {avg_drift} avg/doc)"
            )

        lines.append("")
        lines.append("  PER-CONCEPT DETAIL:")
        lines.append(f"  {'Concept':<15s}  {'Meanings':>9s}  {'Confusion':>10s}  "
                     f"{'Complete':>9s}  {'Predict':>8s}  {'Hazard':>7s}")
        for concept in sorted(self.per_concept.keys()):
            p = self.per_concept[concept]
            lines.append(
                f"  {concept:<15s}  {p.get('resolved_meanings', 0):>9d}  "
                f"{p.get('confusion', 0):>10.3f}  "
                f"{p.get('completeness', 0):>9.3f}  "
                f"{p.get('predictability', 0):>8.3f}  "
                f"{p.get('hazard', 0):>7.3f}"
            )

        return "\n".join(lines)


def meaning_audit(
    addresses: AddressMap,
    provenance_field: str = "_provenance",
    similarity_threshold: float = 0.5,
    feature_roles: Optional["FeatureRoles"] = None,
    ladder_fields: Optional[List[str]] = None,
    auto_classify: bool = False,
) -> MeaningAudit:
    """Run a full meaning quality audit on a dataset.

    Collapses addresses into resolved meanings, then computes four scores:
        confusion, completeness, predictability, hazard.

    When feature_roles and ladder_fields are provided (or auto_classify=True),
    uses three-layer collapse: operators protect, expression merges freely,
    meaning resolves by neighborhood similarity.

    Args:
        addresses: output of meaning_addresses()
        provenance_field: field linking instances to documents
        similarity_threshold: for address collapse (0-1)
        feature_roles: three-layer classification (from classify_features).
            If None and auto_classify=True, will classify automatically.
        ladder_fields: ordered feature names matching address positions.
            Required when feature_roles is provided or auto_classify=True.
        auto_classify: if True and feature_roles is None, automatically
            classify features before collapse.

    Returns:
        MeaningAudit with all four scores and per-concept breakdown
    """
    # ── Step 1: Compute neighborhoods ──
    nbhoods = neighborhoods(addresses, provenance_field=provenance_field)

    # ── Step 1b: Auto-classify features if requested ──
    if auto_classify and feature_roles is None and ladder_fields is not None:
        feature_roles = classify_features(
            addresses, ladder_fields, nbhoods=nbhoods,
            provenance_field=provenance_field,
        )

    # ── Step 2: Collapse addresses ──
    collapsed = collapse_addresses(
        addresses, nbhoods=nbhoods,
        similarity_threshold=similarity_threshold,
        provenance_field=provenance_field,
        feature_roles=feature_roles,
        ladder_fields=ladder_fields,
    )
    resolved_addrs = collapsed["resolved_addresses"]
    concept_mapping = collapsed["mapping"]

    # ── Step 3: Compute per-concept scores ──
    per_concept = {}

    # Build doc → concept → set of RESOLVED meanings
    doc_concept_meanings = defaultdict(lambda: defaultdict(set))
    for concept, addr_dict in addresses.items():
        if concept not in concept_mapping:
            continue
        for addr, insts in addr_dict.items():
            meaning_id = concept_mapping[concept].get(addr, 0)
            for inst in insts:
                doc_id = inst.get(provenance_field, "unknown")
                doc_concept_meanings[doc_id][concept].add(meaning_id)

    for concept in addresses:
        if concept not in concept_mapping:
            continue

        n_resolved = len(collapsed["resolved"][concept])
        n_instances = sum(len(insts) for insts in addresses[concept].values())

        # CONFUSION: average resolved meanings per document (normalized)
        docs_with = []
        for doc_id, concepts in doc_concept_meanings.items():
            if concept in concepts:
                docs_with.append(len(concepts[concept]))

        avg_meanings_per_doc = sum(docs_with) / len(docs_with) if docs_with else 1
        max_meanings_per_doc = max(docs_with) if docs_with else 1

        # Confusion = 1 - (1/avg_meanings_per_doc)
        # 1 meaning/doc → confusion=0 (safe)
        # 5 meanings/doc → confusion=0.8 (dangerous)
        confusion = 1.0 - (1.0 / avg_meanings_per_doc) if avg_meanings_per_doc > 0 else 0

        # COMPLETENESS: evenness of distribution across resolved meanings
        meaning_counts = []
        for meaning_id, addrs in collapsed["resolved"][concept].items():
            count = sum(len(addresses[concept][a]) for a in addrs)
            meaning_counts.append(count)

        total = sum(meaning_counts) or 1
        entropy = 0
        for c in meaning_counts:
            if c > 0:
                p = c / total
                entropy -= p * math.log2(p)
        max_ent = math.log2(len(meaning_counts)) if len(meaning_counts) > 1 else 1
        completeness = entropy / max_ent if max_ent > 0 else 1.0

        # PREDICTABILITY: given the document, how well can you predict
        # which meaning is used? Measured as: fraction of documents where
        # the concept resolves to exactly 1 meaning.
        if docs_with:
            predictability = sum(1 for d in docs_with if d == 1) / len(docs_with)
        else:
            predictability = 1.0

        # HAZARD: combines confusion and unpredictability
        # High confusion + low predictability = hallucination risk
        hazard = confusion * (1.0 - predictability)
        # Scale by how common the word is (frequent dangerous words are worse)
        frequency_weight = math.log2(n_instances + 1) / 20  # normalize
        hazard_weighted = hazard * (1 + frequency_weight)

        per_concept[concept] = {
            "resolved_meanings": n_resolved,
            "n_instances": n_instances,
            "avg_meanings_per_doc": round(avg_meanings_per_doc, 2),
            "max_meanings_per_doc": max_meanings_per_doc,
            "confusion": round(confusion, 4),
            "completeness": round(completeness, 4),
            "predictability": round(predictability, 4),
            "hazard": round(hazard_weighted, 4),
        }

    # ── Step 4: Aggregate scores ──
    concepts_with_data = [c for c in per_concept if per_concept[c]["n_instances"] > 0]

    if concepts_with_data:
        # Weight by instance count
        total_inst = sum(per_concept[c]["n_instances"] for c in concepts_with_data)
        overall_confusion = sum(
            per_concept[c]["confusion"] * per_concept[c]["n_instances"]
            for c in concepts_with_data
        ) / total_inst
        overall_completeness = sum(
            per_concept[c]["completeness"] * per_concept[c]["n_instances"]
            for c in concepts_with_data
        ) / total_inst
        overall_predictability = sum(
            per_concept[c]["predictability"] * per_concept[c]["n_instances"]
            for c in concepts_with_data
        ) / total_inst
    else:
        overall_confusion = 0
        overall_completeness = 0
        overall_predictability = 1.0

    # Hazard ranking: sorted by hazard score descending
    hazard_ranking = sorted(
        [(c, per_concept[c]["hazard"]) for c in concepts_with_data],
        key=lambda x: -x[1],
    )

    return MeaningAudit(
        confusion=overall_confusion,
        completeness=overall_completeness,
        predictability=overall_predictability,
        hazard=hazard_ranking,
        collapse_stats=collapsed["stats"],
        per_concept=per_concept,
    )


# ───────────────────────────────────────────────────────────────────
# Diagnostic Tools — the toolbox
# ───────────────────────────────────────────────────────────────────
# Each tool answers one specific question about the data.
# diagnose() reads the audit scores and picks which tools to run.


def neighborhood_entropy(
    nbhoods: Dict[str, Dict[Address, "Neighborhood"]],
    top_n: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """How spread out are each concept's neighborhoods?

    High entropy = generalist word (neighbors come from many concepts).
    Low entropy = specialist word (neighbors cluster in a few concepts).

    A generalist with high hazard is dangerous because it bridges too
    many domains. A specialist with high hazard is misused — it's
    appearing in the wrong context.

    Args:
        nbhoods: output of neighborhoods()
        top_n: number of top neighbors to consider

    Returns:
        {concept: {entropy, effective_neighbors, n_neighborhoods,
                   specialist_score, top_neighbors}}
    """
    result = {}

    for concept, addr_nbhoods in nbhoods.items():
        # Collect all unique neighbor concepts across all addresses
        all_neighbor_counts = Counter()
        n_neighborhoods = 0

        for addr, nbhood in addr_nbhoods.items():
            n_neighborhoods += 1
            top = nbhood.top(top_n)
            for n in top:
                if n.surprise > 0:
                    all_neighbor_counts[n.concept] += n.count

        if not all_neighbor_counts:
            result[concept] = {
                "entropy": 0.0,
                "effective_neighbors": 1,
                "n_neighborhoods": n_neighborhoods,
                "specialist_score": 1.0,
                "top_neighbors": [],
            }
            continue

        # Entropy of neighbor distribution
        total = sum(all_neighbor_counts.values())
        entropy = 0.0
        for count in all_neighbor_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        effective = 2 ** entropy if entropy > 0 else 1
        n_unique = len(all_neighbor_counts)

        # Specialist score: 1.0 = perfectly specialized, 0.0 = maximally general
        max_entropy = math.log2(n_unique) if n_unique > 1 else 1
        specialist_score = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0

        result[concept] = {
            "entropy": round(entropy, 3),
            "effective_neighbors": round(effective, 1),
            "n_unique_neighbors": n_unique,
            "n_neighborhoods": n_neighborhoods,
            "specialist_score": round(specialist_score, 3),
            "top_neighbors": all_neighbor_counts.most_common(10),
        }

    return result


def mutual_exclusion(
    addresses: AddressMap,
    provenance_field: str = "_provenance",
    min_instances: int = 10,
) -> List[Dict[str, Any]]:
    """Find concept pairs that never (or rarely) share a document.

    Two concepts that never co-occur might be:
      - Genuinely exclusive (you don't discuss surgery and radiology together)
      - A domain boundary (legal vs medical concepts)
      - A data gap (should co-occur but doesn't)

    Args:
        addresses: output of meaning_addresses()
        provenance_field: field linking instances to documents
        min_instances: minimum instances for a concept to be considered

    Returns:
        List of {concept_a, concept_b, co_occurrence_rate, docs_a, docs_b,
                 docs_shared} sorted by co_occurrence_rate ascending
    """
    # Build concept → set of documents
    concept_docs = defaultdict(set)
    concept_instance_count = Counter()

    for concept, addr_dict in addresses.items():
        for addr, insts in addr_dict.items():
            concept_instance_count[concept] += len(insts)
            for inst in insts:
                doc_id = inst.get(provenance_field, inst.get("_context", "unknown"))
                concept_docs[concept].add(doc_id)

    # Filter to concepts with enough data
    concepts = [c for c in concept_docs if concept_instance_count[c] >= min_instances]

    # Pairwise co-occurrence
    exclusions = []
    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            c1, c2 = concepts[i], concepts[j]
            docs_a = concept_docs[c1]
            docs_b = concept_docs[c2]
            shared = docs_a & docs_b

            # Co-occurrence rate relative to smaller set
            smaller = min(len(docs_a), len(docs_b))
            co_rate = len(shared) / smaller if smaller > 0 else 0

            exclusions.append({
                "concept_a": c1,
                "concept_b": c2,
                "co_occurrence_rate": round(co_rate, 3),
                "docs_a": len(docs_a),
                "docs_b": len(docs_b),
                "docs_shared": len(shared),
            })

    return sorted(exclusions, key=lambda x: x["co_occurrence_rate"])


def absence_patterns(
    addresses: AddressMap,
    provenance_field: str = "_provenance",
    min_base_rate: float = 0.1,
) -> Dict[str, Any]:
    """Find expected concepts missing from specific documents.

    If a concept appears in 40% of documents but is absent from a
    specific document, that absence may be informative — the dog
    that didn't bark.

    Args:
        addresses: output of meaning_addresses()
        provenance_field: field linking instances to documents
        min_base_rate: minimum fraction of docs a concept must appear in
            to count as "expected" (default 10%)

    Returns:
        Dict with:
            concept_base_rates: {concept: fraction of docs containing it}
            anomalous_absences: list of {doc_id, missing_concepts,
                expected_count, actual_count, surprise}
            n_docs: total documents
    """
    # Build doc → set of concepts present
    doc_concepts = defaultdict(set)
    all_docs = set()

    for concept, addr_dict in addresses.items():
        for addr, insts in addr_dict.items():
            for inst in insts:
                doc_id = inst.get(provenance_field, inst.get("_context", "unknown"))
                doc_concepts[doc_id].add(concept)
                all_docs.add(doc_id)

    n_docs = len(all_docs)
    if n_docs == 0:
        return {"concept_base_rates": {}, "anomalous_absences": [], "n_docs": 0}

    # Base rates
    concept_doc_count = Counter()
    for doc_id, concepts in doc_concepts.items():
        for c in concepts:
            concept_doc_count[c] += 1

    base_rates = {c: count / n_docs for c, count in concept_doc_count.items()}

    # Expected concepts = those above min_base_rate
    expected = {c for c, rate in base_rates.items() if rate >= min_base_rate}

    # For each document, find expected-but-missing concepts
    anomalies = []
    for doc_id in all_docs:
        present = doc_concepts[doc_id]
        missing = expected - present

        if not missing:
            continue

        # Surprise = sum of base rates of missing concepts
        # High surprise = many common concepts are absent
        surprise = sum(base_rates[c] for c in missing)

        anomalies.append({
            "doc_id": doc_id,
            "missing_concepts": sorted(missing),
            "expected_count": len(expected),
            "actual_count": len(present & expected),
            "surprise": round(surprise, 3),
        })

    anomalies.sort(key=lambda x: -x["surprise"])

    return {
        "concept_base_rates": {c: round(r, 3) for c, r in
                               sorted(base_rates.items(), key=lambda x: -x[1])},
        "anomalous_absences": anomalies[:50],  # top 50 most surprising
        "n_docs": n_docs,
        "n_expected_concepts": len(expected),
    }


def bridging_score(
    nbhoods: Dict[str, Dict[Address, "Neighborhood"]],
    top_n: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """How many distinct meaning domains does each concept bridge?

    A concept with many neighborhoods that have LOW overlap between them
    is a bridge — it connects different meaning domains.
    A concept with many neighborhoods that are all similar is just
    polysemous within one domain.

    Dangerous bridges: high bridging + high hazard = the word is used
    in too many different contexts and nobody notices.

    Args:
        nbhoods: output of neighborhoods()
        top_n: number of top neighbors for comparison

    Returns:
        {concept: {n_neighborhoods, n_clusters, bridging_score,
                   avg_cross_similarity, cluster_sizes}}
    """
    result = {}

    for concept, addr_nbhoods in nbhoods.items():
        addrs = list(addr_nbhoods.keys())
        n_nbhoods = len(addrs)

        if n_nbhoods < 2:
            result[concept] = {
                "n_neighborhoods": n_nbhoods,
                "n_clusters": 1,
                "bridging_score": 0.0,
                "avg_cross_similarity": 1.0,
                "cluster_sizes": [n_nbhoods],
            }
            continue

        # Get signatures for each neighborhood
        signatures = {}
        for addr in addrs:
            nbhood = addr_nbhoods[addr]
            top = nbhood.top(top_n)
            sig = frozenset(n.concept for n in top if n.surprise > 0)
            signatures[addr] = sig

        # Greedy clustering: group neighborhoods with similar signatures
        # (same logic as collapse, but on neighborhoods of one concept)
        clusters = []  # list of [addrs]
        for addr in addrs:
            sig = signatures[addr]
            placed = False

            for cluster in clusters:
                # Compare against first member of cluster
                rep_sig = signatures[cluster[0]]
                if rep_sig or sig:
                    union = len(sig | rep_sig)
                    jaccard = len(sig & rep_sig) / union if union > 0 else 0
                else:
                    jaccard = 1.0

                if jaccard >= 0.3:  # loose threshold — same domain
                    cluster.append(addr)
                    placed = True
                    break

            if not placed:
                clusters.append([addr])

        # Average cross-cluster similarity
        cross_sims = []
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Compare representative signatures
                sig_i = signatures[clusters[i][0]]
                sig_j = signatures[clusters[j][0]]
                if sig_i or sig_j:
                    union = len(sig_i | sig_j)
                    jac = len(sig_i & sig_j) / union if union > 0 else 0
                else:
                    jac = 1.0
                cross_sims.append(jac)

        avg_cross = sum(cross_sims) / len(cross_sims) if cross_sims else 1.0

        # Bridging score: many clusters + low cross-similarity = high bridging
        # Range 0-1: 0 = single domain, 1 = many unrelated domains
        n_clusters = len(clusters)
        if n_clusters <= 1:
            bridge = 0.0
        else:
            cluster_factor = 1.0 - (1.0 / n_clusters)  # more clusters → higher
            dissimilarity = 1.0 - avg_cross  # less similar → higher
            bridge = cluster_factor * dissimilarity

        result[concept] = {
            "n_neighborhoods": n_nbhoods,
            "n_clusters": n_clusters,
            "bridging_score": round(bridge, 3),
            "avg_cross_similarity": round(avg_cross, 3),
            "cluster_sizes": sorted([len(c) for c in clusters], reverse=True),
        }

    return result


# ───────────────────────────────────────────────────────────────────
# Diagnosis — the workman's eye
# ───────────────────────────────────────────────────────────────────

class Diagnosis:
    """Result of diagnose(): audit scores + tool findings + recommendations.

    The audit is the eye. The tools are the hands. The diagnosis
    connects them: this score is bad → this tool explains why →
    here's what the data says.
    """

    def __init__(self, audit, tools_run, findings, recommendations):
        self.audit = audit                    # MeaningAudit
        self.tools_run = tools_run            # list of tool names that fired
        self.findings = findings              # {tool_name: result_dict}
        self.recommendations = recommendations  # list of plain-English strings

    def __str__(self):
        lines = [
            "=" * 70,
            "DIAGNOSIS",
            "=" * 70,
            "",
            f"  Audit scores:",
            f"    Confusion:      {self.audit.confusion:.3f}",
            f"    Completeness:   {self.audit.completeness:.3f}",
            f"    Predictability: {self.audit.predictability:.3f}",
            "",
            f"  Tools selected: {', '.join(self.tools_run) or '(none)'}",
            "",
        ]

        # ── Tool findings ──
        if "neighborhood_entropy" in self.findings:
            ent = self.findings["neighborhood_entropy"]
            lines.append("  NEIGHBORHOOD ENTROPY (specialist vs generalist):")
            # Show most generalist and most specialist
            by_spec = sorted(ent.items(), key=lambda x: x[1]["specialist_score"])
            generalists = [(c, s) for c, s in by_spec if s["specialist_score"] < 0.3]
            specialists = [(c, s) for c, s in by_spec if s["specialist_score"] > 0.7]
            if generalists:
                lines.append(f"    Generalists (low specialist score — spread thin):")
                for c, s in generalists[:5]:
                    lines.append(f"      {c:<15s}  specialist={s['specialist_score']:.2f}  "
                                 f"entropy={s['entropy']:.2f}  "
                                 f"effective_neighbors={s['effective_neighbors']:.0f}")
            if specialists:
                lines.append(f"    Specialists (high specialist score — focused):")
                for c, s in specialists[:5]:
                    lines.append(f"      {c:<15s}  specialist={s['specialist_score']:.2f}  "
                                 f"entropy={s['entropy']:.2f}  "
                                 f"effective_neighbors={s['effective_neighbors']:.0f}")
            lines.append("")

        if "mutual_exclusion" in self.findings:
            excl = self.findings["mutual_exclusion"]
            lines.append("  MUTUAL EXCLUSION (concept pairs that never co-occur):")
            never = [e for e in excl if e["co_occurrence_rate"] == 0]
            rare = [e for e in excl if 0 < e["co_occurrence_rate"] < 0.05]
            if never:
                lines.append(f"    Never co-occur ({len(never)} pairs):")
                for e in never[:5]:
                    lines.append(f"      {e['concept_a']} ↔ {e['concept_b']}  "
                                 f"(docs: {e['docs_a']} / {e['docs_b']})")
            if rare:
                lines.append(f"    Rarely co-occur ({len(rare)} pairs, <5%):")
                for e in rare[:5]:
                    lines.append(f"      {e['concept_a']} ↔ {e['concept_b']}  "
                                 f"rate={e['co_occurrence_rate']:.1%}")
            lines.append("")

        if "absence_patterns" in self.findings:
            absn = self.findings["absence_patterns"]
            lines.append("  ABSENCE PATTERNS (the dogs that didn't bark):")
            lines.append(f"    {absn['n_docs']} documents, "
                         f"{absn['n_expected_concepts']} expected concepts")
            anomalies = absn["anomalous_absences"]
            if anomalies:
                lines.append(f"    Top {min(5, len(anomalies))} most surprising absences:")
                for a in anomalies[:5]:
                    missing = ", ".join(a["missing_concepts"][:5])
                    if len(a["missing_concepts"]) > 5:
                        missing += f" (+{len(a['missing_concepts'])-5} more)"
                    lines.append(f"      {a['doc_id']}: missing [{missing}]  "
                                 f"surprise={a['surprise']:.2f}")
            lines.append("")

        if "bridging_score" in self.findings:
            bridge = self.findings["bridging_score"]
            lines.append("  BRIDGING SCORE (concepts spanning multiple domains):")
            by_bridge = sorted(bridge.items(), key=lambda x: -x[1]["bridging_score"])
            high_bridge = [(c, s) for c, s in by_bridge if s["bridging_score"] > 0.3]
            if high_bridge:
                lines.append(f"    High bridging (connects different meaning domains):")
                for c, s in high_bridge[:5]:
                    lines.append(f"      {c:<15s}  bridging={s['bridging_score']:.2f}  "
                                 f"clusters={s['n_clusters']}  "
                                 f"cross_sim={s['avg_cross_similarity']:.2f}")
            else:
                lines.append("    No high-bridging concepts found.")
            lines.append("")

        # ── Recommendations ──
        if self.recommendations:
            lines.append("  RECOMMENDATIONS:")
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"    {i}. {rec}")
        else:
            lines.append("  No specific recommendations — data looks clean.")

        return "\n".join(lines)


def diagnose(
    addresses: AddressMap,
    provenance_field: str = "_provenance",
    similarity_threshold: float = 0.5,
    confusion_threshold: float = 0.3,
    completeness_threshold: float = 0.5,
    predictability_threshold: float = 0.6,
    hazard_threshold: float = 0.1,
) -> Diagnosis:
    """The workman's eye: audit the data, then reach for the right tools.

    Runs meaning_audit() first, then selects diagnostic tools based
    on what the scores reveal:

        High confusion    → neighborhood_entropy + mutual_exclusion
                            (WHY are meanings drifting? generalists? conflation?)
        Low completeness  → absence_patterns
                            (WHAT meanings are missing? WHERE are the gaps?)
        Low predictability → bridging_score
                            (WHICH words span too many domains?)
        High hazard       → neighborhood_entropy + bridging_score
                            (dangerous words: generalist or bridge?)

    The diagnosis doesn't interpret. It measures, selects tools,
    runs them, and reports findings.

    Args:
        addresses: output of meaning_addresses()
        provenance_field: field linking instances to documents
        similarity_threshold: for address collapse
        confusion_threshold: above this → run confusion tools
        completeness_threshold: below this → run completeness tools
        predictability_threshold: below this → run predictability tools
        hazard_threshold: concepts above this → run hazard tools

    Returns:
        Diagnosis with audit, tool findings, and recommendations
    """
    # ── Step 1: The eye — run the audit ──
    nbhoods = neighborhoods(addresses, provenance_field=provenance_field)
    audit = meaning_audit(
        addresses, provenance_field=provenance_field,
        similarity_threshold=similarity_threshold,
    )

    tools_run = []
    findings = {}
    recommendations = []

    # ── Step 2: Read the scores — decide which tools to reach for ──
    need_entropy = False
    need_exclusion = False
    need_absence = False
    need_bridging = False

    # High confusion → why is meaning drifting?
    if audit.confusion > confusion_threshold:
        need_entropy = True
        need_exclusion = True
        recommendations.append(
            f"Confusion is {audit.confusion:.3f} (>{confusion_threshold}). "
            f"Concepts are drifting within documents."
        )

    # Low completeness → what's missing?
    if audit.completeness < completeness_threshold:
        need_absence = True
        recommendations.append(
            f"Completeness is {audit.completeness:.3f} (<{completeness_threshold}). "
            f"Some meanings are underrepresented."
        )

    # Low predictability → which words can't be disambiguated?
    if audit.predictability < predictability_threshold:
        need_bridging = True
        recommendations.append(
            f"Predictability is {audit.predictability:.3f} (<{predictability_threshold}). "
            f"Context doesn't disambiguate well."
        )

    # High hazard concepts → why are they dangerous?
    high_hazard = [c for c, h in audit.hazard if h > hazard_threshold]
    if high_hazard:
        need_entropy = True
        need_bridging = True
        recommendations.append(
            f"{len(high_hazard)} concepts above hazard {hazard_threshold}: "
            f"{', '.join(high_hazard[:5])}"
            f"{' (+ more)' if len(high_hazard) > 5 else ''}."
        )

    # If everything looks clean, still run entropy for the profile
    if not any([need_entropy, need_exclusion, need_absence, need_bridging]):
        need_entropy = True  # always useful as a concept profile

    # ── Step 3: Reach for the tools ──
    if need_entropy:
        tools_run.append("neighborhood_entropy")
        findings["neighborhood_entropy"] = neighborhood_entropy(nbhoods)

    if need_exclusion:
        tools_run.append("mutual_exclusion")
        findings["mutual_exclusion"] = mutual_exclusion(
            addresses, provenance_field=provenance_field)

    if need_absence:
        tools_run.append("absence_patterns")
        findings["absence_patterns"] = absence_patterns(
            addresses, provenance_field=provenance_field)

    if need_bridging:
        tools_run.append("bridging_score")
        findings["bridging_score"] = bridging_score(nbhoods)

    # ── Step 4: Cross-reference findings with audit ──
    # Add specific recommendations from tool results
    if "neighborhood_entropy" in findings:
        ent = findings["neighborhood_entropy"]
        for concept, h in audit.hazard[:5]:
            if h > hazard_threshold and concept in ent:
                spec = ent[concept]["specialist_score"]
                if spec < 0.3:
                    recommendations.append(
                        f"'{concept}' is hazardous AND a generalist "
                        f"(specialist={spec:.2f}). It's spread across too "
                        f"many contexts."
                    )
                elif spec > 0.7:
                    recommendations.append(
                        f"'{concept}' is hazardous but a specialist "
                        f"(specialist={spec:.2f}). It may be appearing "
                        f"in the wrong context."
                    )

    if "bridging_score" in findings:
        bridge = findings["bridging_score"]
        for concept, h in audit.hazard[:5]:
            if h > hazard_threshold and concept in bridge:
                b = bridge[concept]["bridging_score"]
                if b > 0.3:
                    recommendations.append(
                        f"'{concept}' is hazardous AND bridges "
                        f"{bridge[concept]['n_clusters']} meaning domains "
                        f"(bridging={b:.2f}). It means different things "
                        f"in different parts of the data."
                    )

    return Diagnosis(
        audit=audit,
        tools_run=tools_run,
        findings=findings,
        recommendations=recommendations,
    )
