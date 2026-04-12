"""context_extract.py

Context-aware concept extraction from abstracts.

Instead of trusting OpenAlex concept tags blindly, we:
1. Take their concepts as candidates
2. Scan the abstract for HOW each concept is used
3. Classify the role: TOOL, TARGET, FINDING, CONTEXT

The same word "GARCH" means different things:
    TOOL:    "we apply GARCH to..."
    TARGET:  "we study the properties of GARCH..."
    FINDING: "GARCH outperforms..."
    CONTEXT: "unlike previous GARCH studies..."

The role determines whether a missing concept is a tool-transfer opportunity.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict


# ---------------------------------------------------------------------------
# Context patterns — what words surround a concept tell you its role
# ---------------------------------------------------------------------------

TOOL_PATTERNS = [
    r"(?:we |this paper |this study )?(?:use[sd]?|appl(?:y|ied|ies)|employ[sed]*|implement[sed]*|adopt[sed]*|leverage[sd]*|utilize[sd]*)",
    r"(?:we )?(?:propose[sd]?|introduce[sd]?|develop[sed]*|present[sd]?|design[sed]*|construct[sed]*)",
    r"(?:using|via|through|with|based on|by means of)",
    r"(?:our |the proposed |the |a )?(?:method|approach|technique|algorithm|framework|model|tool|procedure)",
    r"(?:trained|fitted|estimated|calibrated|optimized) (?:using|with|via)",
]

TARGET_PATTERNS = [
    r"(?:we )?(?:study|studies|studied|analyze[sd]*|examine[sd]*|investigate[sd]*|explore[sd]*|assess[sed]*|evaluate[sd]*)",
    r"(?:we )?(?:compare[sd]?|test[sed]*|measure[sd]*|quantif(?:y|ied)|characterize[sd]*)",
    r"(?:the )?(?:effect|impact|influence|role|behavior|dynamics|properties|performance) of",
    r"(?:applied to|tested on|evaluated on|measured (?:in|on|for))",
    r"(?:in the context of|for the case of|in the (?:domain|field|area) of)",
]

FINDING_PATTERNS = [
    r"(?:outperform[sed]*|surpass[sed]*|exceed[sed]*|improve[sd]* (?:upon|over|on))",
    r"(?:we )?(?:find|found|show[sed]*|demonstrate[sd]*|reveal[sed]*|confirm[sed]*|establish[sed]*)",
    r"(?:result[sed]* (?:show|indicate|suggest|demonstrate|confirm))",
    r"(?:achieve[sd]?|attain[sed]*|obtain[sed]*|yield[sed]*) (?:a |an )?(?:significant|superior|better|higher|lower|improved)",
    r"(?:significant(?:ly)?|substantial(?:ly)?|strong(?:ly)?) (?:positive|negative|associated|correlated|related|impact)",
]

CONTEXT_PATTERNS = [
    r"(?:unlike|contrary to|in contrast (?:to|with)|compared (?:to|with)|as opposed to)",
    r"(?:previous(?:ly)?|prior|existing|traditional|conventional|classical|standard)",
    r"(?:literature|studies|research|work) (?:on|in|about|regarding)",
    r"(?:well.known|widely.used|commonly|typically|often|usually)",
    r"(?:motivated by|inspired by|building on|extending|following)",
]


def classify_concept_role(
    concept: str,
    abstract: str,
    window: int = 60,
) -> Tuple[str, float, str]:
    """
    Classify how a concept is used in an abstract.

    Returns (role, confidence, context_snippet)
    role: "tool", "target", "finding", "context", "mention"
    """
    abstract_lower = abstract.lower()
    concept_lower = concept.lower()

    # Find all occurrences of the concept in the abstract
    positions = []
    start = 0
    while True:
        pos = abstract_lower.find(concept_lower, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1

    if not positions:
        # Concept not found in abstract — it's an OpenAlex tag, not mentioned
        return ("tag_only", 0.0, "")

    # Check context around each occurrence
    role_scores = {"tool": 0.0, "target": 0.0, "finding": 0.0, "context": 0.0}
    best_snippet = ""

    for pos in positions:
        # Extract window around the concept
        win_start = max(0, pos - window)
        win_end = min(len(abstract), pos + len(concept) + window)
        snippet = abstract[win_start:win_end].lower()

        if not best_snippet:
            best_snippet = abstract[win_start:win_end]

        # Score each role
        for pattern in TOOL_PATTERNS:
            if re.search(pattern, snippet):
                role_scores["tool"] += 1.0
                if not best_snippet or "tool" not in best_snippet.lower():
                    best_snippet = abstract[win_start:win_end]

        for pattern in TARGET_PATTERNS:
            if re.search(pattern, snippet):
                role_scores["target"] += 1.0

        for pattern in FINDING_PATTERNS:
            if re.search(pattern, snippet):
                role_scores["finding"] += 1.0

        for pattern in CONTEXT_PATTERNS:
            if re.search(pattern, snippet):
                role_scores["context"] += 1.0

    # Pick highest-scoring role
    if max(role_scores.values()) == 0:
        return ("mention", 0.5, best_snippet)

    best_role = max(role_scores, key=role_scores.get)
    total = sum(role_scores.values())
    confidence = role_scores[best_role] / total if total > 0 else 0

    return (best_role, confidence, best_snippet)


def extract_roles(
    concepts: List[Dict],
    abstract: str,
    min_score: float = 0.2,
    max_concepts: int = 10,
) -> Dict[str, List[Dict]]:
    """
    Classify all concepts from a paper by their role in the abstract.

    Args:
        concepts: OpenAlex concept list [{display_name, level, score}, ...]
        abstract: Full abstract text
        min_score: Minimum OpenAlex concept score to consider
        max_concepts: Maximum concepts to process

    Returns:
        {role: [{name, level, oa_score, confidence, snippet}, ...]}
    """
    results = defaultdict(list)

    # Filter and sort concepts
    filtered = [c for c in concepts
                if c.get("score", 0) >= min_score and c.get("level", 0) >= 2]
    filtered.sort(key=lambda c: -c.get("score", 0))
    filtered = filtered[:max_concepts]

    for concept in filtered:
        name = concept.get("display_name", "")
        if not name:
            continue

        role, confidence, snippet = classify_concept_role(name, abstract)

        results[role].append({
            "name": name,
            "level": concept.get("level", 0),
            "oa_score": concept.get("score", 0),
            "confidence": confidence,
            "snippet": snippet[:100] if snippet else "",
        })

    return dict(results)


# ---------------------------------------------------------------------------
# Batch extraction for gap detection
# ---------------------------------------------------------------------------

def extract_paper_tools(
    concepts: List[Dict],
    abstract: str,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Extract tools, targets, and findings from a paper.

    Returns (tools, targets, findings) — lists of concept names.
    """
    roles = extract_roles(concepts, abstract)

    tools = [c["name"] for c in roles.get("tool", [])]
    targets = [c["name"] for c in roles.get("target", [])]
    findings = [c["name"] for c in roles.get("finding", [])]

    return tools, targets, findings


# ---------------------------------------------------------------------------
# Demo / test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test with sample abstracts
    test_cases = [
        {
            "title": "Volatility Forecasting with GARCH",
            "abstract": "We apply the GARCH(1,1) model to forecast equity market volatility. "
                        "Using daily returns from the S&P 500 index, we compare GARCH with "
                        "stochastic volatility models. Our results show that GARCH outperforms "
                        "traditional historical volatility estimates. We also examine the impact "
                        "of leverage effects on volatility dynamics.",
            "concepts": [
                {"display_name": "GARCH", "level": 3, "score": 0.9},
                {"display_name": "Stochastic volatility", "level": 3, "score": 0.7},
                {"display_name": "Volatility", "level": 2, "score": 0.8},
                {"display_name": "Leverage", "level": 2, "score": 0.5},
                {"display_name": "S&P 500", "level": 2, "score": 0.4},
            ],
        },
        {
            "title": "Machine Learning in Credit Risk",
            "abstract": "This study investigates the application of random forest and gradient "
                        "boosting methods to credit risk assessment. We use a dataset of 50,000 "
                        "loan applications and compare machine learning approaches with traditional "
                        "logistic regression. Our findings demonstrate that ensemble methods achieve "
                        "significantly higher accuracy in predicting default probability.",
            "concepts": [
                {"display_name": "Random forest", "level": 3, "score": 0.8},
                {"display_name": "Gradient boosting", "level": 3, "score": 0.7},
                {"display_name": "Logistic regression", "level": 3, "score": 0.6},
                {"display_name": "Credit risk", "level": 2, "score": 0.9},
                {"display_name": "Default probability", "level": 3, "score": 0.5},
            ],
        },
    ]

    for tc in test_cases:
        print(f"Paper: {tc['title']}")
        print(f"Abstract: {tc['abstract'][:100]}...")
        print()

        roles = extract_roles(tc["concepts"], tc["abstract"])
        for role, items in roles.items():
            print(f"  {role.upper()}:")
            for item in items:
                print(f"    {item['name']} (L{item['level']}, conf={item['confidence']:.1%})")
                if item["snippet"]:
                    print(f"      \"{item['snippet']}\"")
        print()
