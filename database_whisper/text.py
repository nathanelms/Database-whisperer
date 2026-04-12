"""Text featurizer for Database Whisper.

Turns unstructured text into structured records that DW can eat.
Domain-agnostic: works on any English text.

Usage:
    import database_whisper as dw

    # Featurize a text field in your records
    enriched = dw.featurize_text(
        records,
        text_field="abstract",
        target_words=["faith", "love", "sin"],  # optional focus
    )

    # Or featurize raw text into concept-in-context instances
    instances = dw.extract_concept_instances(
        records,
        text_field="text",
        concepts=["significant", "model", "robust"],
    )

    # Then profile as usual
    report = dw.profile_records(instances, identity_fields=["concept"])

The featurizer extracts relational features for each concept-in-context:
    - verb_class: what verb is nearest to the concept
    - syntactic_role: subject, object, prepositional, embedded
    - paired_concept: what other target concept co-occurs
    - contrast: is the concept contrasted with something (X but Y, X not Y)
    - equation: is the concept equated/defined (X is Y)
    - modality: declarative, negated, conditional, commanded, emphatic
    - speaker: who is speaking (if detectable)

These are universal English features. They work on the Bible,
the Constitution, clinical trials, and scientific abstracts.
DW discovers which ones carry signal for YOUR corpus.

Theoretical basis for each feature:
    - verb_class: frame semantics — the verb determines the situation schema
    - syntactic_role: dependency grammar — how concept relates to the verb frame
    - paired_concept: compositional semantics — argument type constrains sense
    - contrast: discourse pragmatics — contrastive structure separates uses
    - equation: predicate logic — copular definitions mark identity relations
    - modality: speech act theory — asserting vs commanding vs hedging
    - voice: information structure — passive/active shifts topic/focus
    - negation: discourse pragmatics — affirmed vs negated inverses meaning
    - clause_position: information structure — early = topic, late = focus
    - transitivity: predicate-argument structure — valency fingerprints sense
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Verb classification — universal English verb classes
# ---------------------------------------------------------------------------

VERB_CLASSES = {
    "being":      ["is", "are", "am", "was", "were", "be", "been", "being"],
    "having":     ["has", "have", "had", "hath", "having"],
    "doing":      ["do", "does", "did", "done", "doth", "doeth"],
    "giving":     ["give", "gives", "gave", "given", "giveth"],
    "making":     ["make", "makes", "made", "making", "maketh"],
    "causing":    ["cause", "causes", "caused", "causing"],
    "coming":     ["come", "comes", "came", "coming", "cometh"],
    "going":      ["go", "goes", "went", "going", "gone", "goeth"],
    "knowing":    ["know", "knows", "knew", "known", "knoweth"],
    "seeing":     ["see", "sees", "saw", "seen", "seeth"],
    "saying":     ["say", "says", "said", "saying", "saith", "speak",
                   "speaks", "spoke", "spoken", "spake", "tell", "tells", "told"],
    "bringing":   ["bring", "brings", "brought", "bringing", "bringeth"],
    "keeping":    ["keep", "keeps", "kept", "keeping", "keepeth"],
    "taking":     ["take", "takes", "took", "taken", "taking", "taketh"],
    "setting":    ["set", "sets", "put", "puts", "setteth"],
    "judging":    ["judge", "judges", "judged", "judging", "judgeth"],
    "requiring":  ["require", "requires", "required", "shall", "must",
                   "should", "ought", "need", "needs", "needed"],
    "providing":  ["provide", "provides", "provided", "grant", "grants",
                   "granted", "allow", "allows", "allowed", "permit"],
    "preventing": ["prevent", "prevents", "prevented", "prohibit",
                   "prohibits", "prohibited", "deny", "denies", "denied",
                   "restrict", "restricts", "restricted"],
    "changing":   ["change", "changes", "changed", "alter", "alters",
                   "altered", "amend", "amends", "amended", "modify"],
    "showing":    ["show", "shows", "showed", "shown", "demonstrate",
                   "demonstrates", "demonstrated", "reveal", "reveals"],
    "finding":    ["find", "finds", "found", "discover", "discovers",
                   "discovered", "observe", "observes", "observed"],
    "defining":   ["define", "defines", "defined", "mean", "means", "meant",
                   "constitute", "constitutes", "constituted", "include"],
    "applying":   ["apply", "applies", "applied", "use", "uses", "used",
                   "employ", "employs", "employed", "utilize"],
    "comparing":  ["compare", "compares", "compared", "differ", "differs",
                   "differed", "exceed", "exceeds", "exceeded", "outperform"],
}

# Flatten for lookup
_VERB_LOOKUP = {}
for _vclass, _forms in VERB_CLASSES.items():
    for _form in _forms:
        _VERB_LOOKUP[_form] = _vclass


def _clean(w: str) -> str:
    """Strip punctuation, lowercase."""
    return re.sub(r"[^a-z']", "", w.lower())


# ---------------------------------------------------------------------------
# Core feature extractors — domain-agnostic
# ---------------------------------------------------------------------------

def _nearest_verb_class(words: List[str], pos: int, window: int = 5) -> str:
    """Find the nearest verb class within a window around position."""
    best_dist = window + 1
    best_class = "none"

    start = max(0, pos - window)
    end = min(len(words), pos + window + 1)

    for i in range(start, end):
        if i == pos:
            continue
        clean = _clean(words[i])
        if clean in _VERB_LOOKUP:
            dist = abs(i - pos)
            if dist < best_dist:
                best_dist = dist
                best_class = _VERB_LOOKUP[clean]

    return best_class


def _syntactic_role(words: List[str], pos: int) -> str:
    """Classify the syntactic role of the word at position."""
    if pos == 0:
        return "subject"

    ARTICLES = {"the", "a", "an", "his", "her", "their", "our", "my",
                "thy", "its", "this", "that", "these", "those", "said", "such"}
    PREPS = {"of", "in", "by", "with", "through", "for", "from", "upon",
             "unto", "into", "without", "against", "toward", "towards",
             "under", "over", "between", "among", "about", "on", "to", "at"}

    if pos <= 2:
        prev = _clean(words[pos - 1])
        if prev in ARTICLES:
            return "subject"

    if pos >= 1:
        prev = _clean(words[pos - 1])
        if prev in PREPS:
            return "prepositional"

    # Check for verb before concept (concept is likely object)
    for i in range(max(0, pos - 3), pos):
        if _clean(words[i]) in _VERB_LOOKUP:
            return "object"

    return "embedded"


def _paired_concept(
    words: List[str],
    pos: int,
    concept: str,
    target_set: Set[str],
) -> str:
    """What other target concept co-occurs in this text?"""
    concept_lower = concept.lower()
    found = []
    for i, w in enumerate(words):
        if i == pos:
            continue
        clean = _clean(w)
        if clean in target_set and clean != concept_lower:
            found.append((abs(i - pos), clean))

    if not found:
        return "none"

    found.sort()
    return found[0][1]


def _detect_contrast(
    text: str,
    concept: str,
    target_set: Set[str],
) -> str:
    """Detect if the concept is contrasted with something.

    Universal contrast patterns:
        - "X; but Y" / "X, but Y" (most common in English)
        - "X against Y"
        - "not X but Y"
        - "X without Y"
        - "X, and not Y" / "X rather than Y"
        - "X however Y" / "X yet Y"
    """
    text_lower = text.lower()
    concept_lower = concept.lower()

    if concept_lower not in text_lower:
        return "none"

    # Pattern 1: "against"
    against_pats = [
        rf"(\w+)\s+\w*\s*against\s+(?:the\s+)?{concept_lower}",
        rf"{concept_lower}\s+\w*\s*against\s+(?:the\s+)?(\w+)",
    ]
    for pat in against_pats:
        m = re.search(pat, text_lower)
        if m:
            term = _clean(m.group(1))
            if term in target_set and term != concept_lower:
                return term

    # Pattern 2: clause-level contrast with ";/,/: but"
    for splitter in ["; but ", ": but ", ", but ", " but ", "; however ",
                     ", however ", " yet ", "; yet "]:
        if splitter in text_lower:
            parts = text_lower.split(splitter, 1)
            other = None
            if concept_lower in parts[0] and concept_lower not in parts[1]:
                other = parts[1]
            elif concept_lower in parts[1] and concept_lower not in parts[0]:
                other = parts[0]

            if other:
                for w in re.findall(r"[a-z]+", other):
                    if w in target_set and w != concept_lower:
                        return w

    # Pattern 3: "not X but Y" / "X, and not Y"
    not_pats = [
        rf"not\s+(?:by\s+|of\s+|through\s+)?{concept_lower}[\s,;]+but\s+(?:by\s+|of\s+)?(\w+)",
        rf"not\s+(?:by\s+|of\s+|through\s+)?(\w+)[\s,;]+but\s+(?:by\s+|of\s+)?{concept_lower}",
        rf"{concept_lower}[\s,]+and\s+not\s+(\w+)",
        rf"(\w+)[\s,]+and\s+not\s+{concept_lower}",
        rf"{concept_lower}[\s,]+rather\s+than\s+(\w+)",
        rf"(\w+)[\s,]+rather\s+than\s+{concept_lower}",
    ]
    for pat in not_pats:
        m = re.search(pat, text_lower)
        if m:
            term = _clean(m.group(1))
            if term in target_set and term != concept_lower:
                return term

    # Pattern 4: "X without Y"
    without_pats = [
        rf"{concept_lower}\s+without\s+(\w+)",
        rf"(\w+)\s+without\s+{concept_lower}",
    ]
    for pat in without_pats:
        m = re.search(pat, text_lower)
        if m:
            term = _clean(m.group(1))
            if term in target_set and term != concept_lower:
                return term

    return "none"


def _detect_equation(
    text: str,
    concept: str,
    target_set: Set[str],
) -> str:
    """Detect if the concept is equated/defined.

    Universal equation patterns:
        - "X is Y" / "X are Y" / "X was Y"
        - "the X of Z is Y"
        - "Y is X" (reverse)
        - "X means Y" / "X constitutes Y"
        - "X, defined as Y"
    """
    text_lower = text.lower()
    concept_lower = concept.lower()

    if concept_lower not in text_lower:
        return "none"

    SKIP = {"not", "a", "an", "the", "that", "this", "it", "he", "she", "we",
            "no", "in", "of", "to", "as", "at", "by", "or", "and", "but",
            "his", "her", "thy", "our", "my", "their", "its", "there",
            "also", "even", "yet", "so", "for", "with", "from", "upon",
            "which", "who", "whom", "whose", "what", "all", "shall",
            "unto", "thee", "thou", "hath", "have", "had", "been", "was",
            "when", "where", "how", "why", "if", "then", "than", "more",
            "any", "some", "every", "each", "only", "just", "still"}

    def _first_meaningful(words):
        for w in words:
            clean = _clean(w)
            if clean and clean not in SKIP and len(clean) > 2:
                return clean
        return None

    def _prefer_target(words):
        """Prefer target concepts, fall back to first meaningful."""
        for w in words:
            clean = _clean(w)
            if clean in target_set and clean != concept_lower:
                return clean
        return _first_meaningful(words)

    # Pattern 1: "concept is/are/be/means ... WORD"
    eq1 = rf"{concept_lower}\s+(?:is|are|be|was|were|means?|constitutes?|represents?|denotes?)\s+(.+?)(?:[.;,:]|$)"
    m = re.search(eq1, text_lower)
    if m:
        term = _prefer_target(m.group(1).split())
        if term and term != concept_lower:
            return term

    # Pattern 2: "the concept of X is Y"
    eq2 = rf"(?:the\s+)?{concept_lower}\s+of\s+.{{1,40}}?\s+(?:is|are|be|was|were|means?)\s+(.+?)(?:[.;,:]|$)"
    m = re.search(eq2, text_lower)
    if m:
        term = _prefer_target(m.group(1).split())
        if term and term != concept_lower:
            return term

    # Pattern 3: "WORD is concept" (reverse)
    eq3 = rf"(\w[\w\s]{{0,30}}?)\s+(?:is|are|be|was|were|means?)\s+(?:the\s+)?{concept_lower}(?:\s|[.;,:]|$)"
    m = re.search(eq3, text_lower)
    if m:
        before = m.group(1).split()
        for w in reversed(before):
            clean = _clean(w)
            if clean and clean not in SKIP and len(clean) > 2 and clean != concept_lower:
                if clean in target_set:
                    return clean
                return clean

    # Pattern 4: "concept, defined as / known as / referred to as"
    eq4 = rf"{concept_lower}[\s,]+(?:defined|known|referred to|described)\s+as\s+(.+?)(?:[.;,:]|$)"
    m = re.search(eq4, text_lower)
    if m:
        term = _prefer_target(m.group(1).split())
        if term and term != concept_lower:
            return term

    return "none"


def _detect_modality(words: List[str], pos: int) -> str:
    """Classify the modal context around a word."""
    window = words[max(0, pos - 4):pos + 5]
    window_str = " ".join(_clean(w) for w in window)

    if any(w in window_str for w in ["shall not", "should not", "must not",
                                      "not", "no", "neither", "nor", "never",
                                      "cannot", "can't"]):
        return "negated"
    if any(w in window_str for w in ["shall", "must", "should", "ought",
                                      "required", "commanded"]):
        return "commanded"
    if any(w in window_str for w in ["may", "might", "can", "could",
                                      "perhaps", "possibly", "probably"]):
        return "possible"
    if any(w in window_str for w in ["truly", "surely", "verily", "indeed",
                                      "certainly", "clearly", "obviously"]):
        return "emphatic"
    if any(w in window_str for w in ["if", "whether", "unless", "lest",
                                      "provided", "assuming", "when"]):
        return "conditional"

    return "declarative"


def _detect_voice(text: str) -> str:
    """Detect whether the text uses active or passive voice.

    Simple heuristic: look for "was/were/been + past participle" patterns.
    """
    passive_markers = re.findall(
        r"\b(?:is|are|was|were|be|been|being)\s+\w+(?:ed|en|t)\b",
        text.lower()
    )
    if len(passive_markers) >= 2:
        return "passive"
    elif passive_markers:
        return "mixed"
    return "active"


def _detect_negation(words: List[str], pos: int) -> str:
    """Detect if the concept is in a negated context.

    Discourse pragmatics: "God saves" vs "God does not save" — same verb,
    same role, opposite meaning. Negation is a binary sense-switch.
    """
    # Check a window before the concept for negation markers
    NEG_WORDS = {"not", "no", "never", "neither", "nor", "cannot", "nothing",
                 "none", "without", "lack", "absence"}
    NEG_CONTRACTIONS = {"n't", "nt"}

    start = max(0, pos - 4)
    for i in range(start, pos):
        w = _clean(words[i])
        if w in NEG_WORDS:
            return "negated"
        # Handle contractions: "don't", "isn't", "can't"
        raw = words[i].lower()
        if any(raw.endswith(c) for c in NEG_CONTRACTIONS):
            return "negated"

    # Also check one word after (for "saves not", archaic English)
    if pos + 1 < len(words) and _clean(words[pos + 1]) in NEG_WORDS:
        return "negated"

    return "affirmed"


def _detect_clause_position(words: List[str], pos: int) -> str:
    """Classify where in the clause this concept sits.

    Information structure (Prague School): early position = topic
    (what we're talking about), late position = focus (new info,
    the answer-bearing part). Focus is more discriminatory.

    Also detects cleft constructions ("it is X that...") which
    explicitly mark focus.
    """
    total = len(words)
    if total == 0:
        return "unknown"

    # Check for cleft: "it is/was CONCEPT that"
    if pos >= 2:
        w1 = _clean(words[pos - 2])
        w2 = _clean(words[pos - 1])
        if w1 == "it" and w2 in ("is", "was", "be"):
            return "cleft_focus"

    # Relative position in the word list
    rel = pos / total

    # Find clause boundaries (punctuation marks that split clauses)
    clause_start = 0
    clause_end = total
    CLAUSE_BREAKS = {",", ";", ":", ".", "!", "?", "--", "—"}
    for i in range(pos - 1, -1, -1):
        if words[i].rstrip() in CLAUSE_BREAKS or words[i].endswith((",", ";", ":")):
            clause_start = i + 1
            break
    for i in range(pos + 1, total):
        if words[i].rstrip() in CLAUSE_BREAKS or words[i].startswith((",", ";", ":")):
            clause_end = i
            break

    clause_len = clause_end - clause_start
    if clause_len <= 0:
        return "unknown"

    pos_in_clause = (pos - clause_start) / clause_len

    if pos_in_clause < 0.25:
        return "early"   # topic position
    elif pos_in_clause > 0.75:
        return "late"    # focus position
    return "medial"


def _detect_transitivity(words: List[str], pos: int) -> str:
    """Detect transitivity of the nearest verb.

    Predicate-argument structure: the "valency" of a verb is a
    structural fingerprint for meaning. "the engine fired" (intransitive)
    vs "the boss fired the clerk" (transitive) are different senses.
    """
    # Find nearest verb
    verb_pos = -1
    best_dist = 6
    for i in range(max(0, pos - 5), min(len(words), pos + 6)):
        if i == pos:
            continue
        if _clean(words[i]) in _VERB_LOOKUP:
            dist = abs(i - pos)
            if dist < best_dist:
                best_dist = dist
                verb_pos = i

    if verb_pos == -1:
        return "none"

    # Check if it's a copular verb (is/are/was/were/be/been)
    verb_clean = _clean(words[verb_pos])
    if verb_clean in ("is", "are", "am", "was", "were", "be", "been", "being"):
        return "copular"

    # Look after the verb for a direct object (noun/article)
    ARTICLES = {"the", "a", "an", "his", "her", "their", "our", "my",
                "thy", "its", "this", "that", "these", "those"}
    after_verb = verb_pos + 1
    if after_verb < len(words):
        next_w = _clean(words[after_verb])
        # If next word is a preposition, likely intransitive + PP
        PREPS = {"of", "in", "by", "with", "through", "for", "from", "upon",
                 "unto", "into", "to", "at", "on", "about"}
        if next_w in PREPS:
            return "intransitive"
        # If next word is article or noun-like, likely transitive
        if next_w in ARTICLES or (len(next_w) > 2 and next_w not in
                                   {"and", "but", "or", "not", "so", "yet"}):
            return "transitive"

    return "intransitive"


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def auto_detect_concepts(
    records: List[Dict],
    text_field: str,
    min_freq: int = 20,
    max_concepts: int = 50,
    min_word_length: int = 4,
) -> List[str]:
    """
    Automatically detect high-frequency content words as concepts.

    Filters out common English stopwords and very short words.
    Returns concepts sorted by frequency.
    """
    STOPWORDS = {
        "the", "and", "that", "this", "with", "for", "are", "but", "not",
        "you", "all", "can", "had", "her", "was", "one", "our", "out",
        "his", "has", "have", "from", "they", "been", "said", "each",
        "which", "their", "will", "other", "about", "many", "then",
        "them", "these", "some", "would", "make", "like", "into",
        "could", "time", "very", "when", "come", "made", "after",
        "also", "did", "back", "more", "before", "than", "most",
        "only", "over", "such", "just", "first", "may", "any",
        "new", "now", "way", "who", "does", "what", "where", "how",
        "much", "both", "between", "under", "being", "through",
        "well", "still", "should", "those", "shall", "upon", "unto",
        "were", "here", "there", "while", "whom", "whose", "your",
        "itself", "himself", "herself", "themselves", "myself",
        "because", "therefore", "however", "although", "though",
    }

    word_counts = Counter()
    for rec in records:
        text = rec.get(text_field, "")
        if not text:
            continue
        words = re.findall(r"[a-z]+", text.lower())
        for w in words:
            if len(w) >= min_word_length and w not in STOPWORDS:
                word_counts[w] += 1

    # Filter by frequency
    concepts = [
        word for word, count in word_counts.most_common(max_concepts * 3)
        if count >= min_freq
    ][:max_concepts]

    return concepts


def extract_concept_instances(
    records: List[Dict],
    text_field: str,
    concepts: Optional[List[str]] = None,
    metadata_fields: Optional[List[str]] = None,
    provenance_field: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Extract concept-in-context instances from text records.

    For each (concept, record) pair where the concept appears in the text,
    extract relational features describing how the concept is used.

    Args:
        records: list of dicts, each with at least text_field
        text_field: name of the field containing text
        concepts: target words to extract instances for.
                  If None, auto-detected from corpus.
        metadata_fields: additional fields to copy to each instance
                         (e.g., ["category", "year"]).
                         These become DW candidate features.
        provenance_field: field to use as instance provenance (e.g., "id")

    Returns:
        List of structured records, one per concept-in-context instance.
        Each record has: concept, verb_class, syntactic_role, paired_concept,
        contrast, equation, modality, voice, plus any metadata fields.
    """
    if concepts is None:
        concepts = auto_detect_concepts(records, text_field)
        if not concepts:
            return []

    target_set = set(c.lower() for c in concepts)

    instances = []

    for rec in records:
        text = rec.get(text_field, "")
        if not text:
            continue

        text_lower = text.lower()
        words = text.split()

        for concept in concepts:
            concept_lower = concept.lower()
            if concept_lower not in text_lower:
                continue

            # Find position
            pos = -1
            for i, w in enumerate(words):
                if _clean(w) == concept_lower:
                    pos = i
                    break
            if pos == -1:
                # Partial match
                for i, w in enumerate(words):
                    if concept_lower in _clean(w):
                        pos = i
                        break
            if pos == -1:
                continue

            instance = {
                "concept": concept_lower,
                "verb_class": _nearest_verb_class(words, pos),
                "syntactic_role": _syntactic_role(words, pos),
                "paired_concept": _paired_concept(words, pos, concept, target_set),
                "contrast": _detect_contrast(text, concept, target_set),
                "equation": _detect_equation(text, concept, target_set),
                "modality": _detect_modality(words, pos),
                "voice": _detect_voice(text),
                "negation": _detect_negation(words, pos),
                "clause_position": _detect_clause_position(words, pos),
                "transitivity": _detect_transitivity(words, pos),
            }

            # Copy metadata fields
            if metadata_fields:
                for field in metadata_fields:
                    if field in rec:
                        instance[field] = str(rec[field])

            # Provenance
            if provenance_field and provenance_field in rec:
                instance["_provenance"] = str(rec[provenance_field])

            # Text preview for human inspection
            start = max(0, pos - 8)
            end = min(len(words), pos + 8)
            instance["_context"] = " ".join(words[start:end])

            instances.append(instance)

    return instances


def meaning_addresses(
    instances: List[Dict[str, str]],
    ladder_fields: List[str],
) -> Dict[str, Dict[tuple, List[Dict]]]:
    """
    Group concept instances by their meaning-address.

    The address is defined by the values of the ladder fields
    for each instance. Instances at the same address are
    structurally indistinguishable — aliased.

    Args:
        instances: output of extract_concept_instances
        ladder_fields: field names from the DW ladder (in order)

    Returns:
        {concept: {address_tuple: [instances]}}
    """
    addresses = defaultdict(lambda: defaultdict(list))
    for inst in instances:
        concept = inst["concept"]
        addr = tuple(inst.get(f, "none") for f in ladder_fields)
        addresses[concept][addr].append(inst)
    return dict(addresses)


def resolution_report(
    addresses: Dict,
    ladder_fields: List[str],
    concepts: Optional[List[str]] = None,
    max_display: int = 5,
) -> str:
    """
    Generate a human-readable report of meaning-addresses and structural limits.

    Shows:
        - The coordinate system (ladder fields)
        - Top addresses per concept with example contexts
        - Most aliased addresses (where the text can't distinguish)
        - Theoretical vs actual resolution
    """
    lines = []

    lines.append("=" * 60)
    lines.append("  MEANING ADDRESS REPORT")
    lines.append(f"  Coordinate system: {' -> '.join(ladder_fields)}")
    lines.append("=" * 60)

    if concepts is None:
        concepts = sorted(addresses.keys(),
                          key=lambda c: -sum(len(v) for v in addresses[c].values()))[:8]

    for concept in concepts:
        if concept not in addresses:
            continue
        addrs = addresses[concept]
        total = sum(len(v) for v in addrs.values())
        lines.append(f"\n  {concept.upper()} -- {total} instances, "
                     f"{len(addrs)} distinct addresses")

        sorted_addrs = sorted(addrs.items(), key=lambda x: -len(x[1]))
        for addr, insts in sorted_addrs[:max_display]:
            addr_str = ".".join(f"{f}={v}" for f, v in zip(ladder_fields, addr))
            lines.append(f"\n    [{len(insts):>3}x] {addr_str}")
            for inst in insts[:2]:
                ctx = inst.get("_context", "")
                prov = inst.get("_provenance", "")
                if prov:
                    lines.append(f"          {prov}: {ctx}")
                else:
                    lines.append(f"          {ctx}")

    # Resolution limits
    lines.append(f"\n{'='*60}")
    lines.append("  STRUCTURAL RESOLUTION LIMITS")
    lines.append(f"{'='*60}\n")

    all_values = defaultdict(set)
    total_addresses = 0
    total_instances = 0
    for concept in addresses:
        total_addresses += len(addresses[concept])
        for addr, insts in addresses[concept].items():
            total_instances += len(insts)
            for field, val in zip(ladder_fields, addr):
                all_values[field].add(val)

    theoretical_max = 1
    for field in ladder_fields:
        n = len(all_values[field])
        lines.append(f"  {field}: {n} distinct values")
        theoretical_max *= n

    lines.append(f"\n  Theoretical max addresses: {theoretical_max}")
    lines.append(f"  Actually populated: {total_addresses}")
    lines.append(f"  Total instances: {total_instances}")
    lines.append(f"  Avg instances per address: "
                 f"{total_instances / max(total_addresses, 1):.1f}")

    # Estimate resolution at each ladder depth
    lines.append(f"\n  RESOLUTION BY LADDER DEPTH (rate-distortion curve):")
    for depth in range(1, len(ladder_fields) + 1):
        partial_fields = ladder_fields[:depth]
        partial_addrs = set()
        for concept in addresses:
            for addr in addresses[concept]:
                partial_addrs.add((concept,) + addr[:depth])
        lines.append(f"    depth {depth} ({partial_fields[-1]:>18}): "
                     f"{len(partial_addrs):>6} distinct addresses")

    lines.append(f"\n  Any meaning distinction finer than these "
                 f"{len(ladder_fields)} axes")
    lines.append(f"  is ALIASED -- structurally indistinguishable.")
    lines.append(f"\n  STOPPING GUIDANCE (MDL): If the last rung adds <5%")
    lines.append(f"  new addresses, it's fitting noise, not structure.")
    lines.append(f"  Look at the depth curve above -- the elbow is your")
    lines.append(f"  structural resolution limit.")

    return "\n".join(lines)
