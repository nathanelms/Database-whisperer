"""Extract semantic fields from arXiv abstracts using keyword matching.

Turns free-text abstracts into structured records with:
    method, problem_type, data_type, contribution

Then runs DW to see if it discovers meaningful routing structure.
"""

import json
import csv
from collections import Counter, defaultdict

# Load abstracts
with open("arxiv_abstracts.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

print(f"Extracting semantic fields from {len(papers)} abstracts...")

# ---------------------------------------------------------------
# Keyword dictionaries for semantic extraction
# ---------------------------------------------------------------

METHOD_KEYWORDS = {
    "deep_learning": ["deep learning", "neural network", "cnn", "rnn", "lstm", "transformer", "attention mechanism", "convolutional"],
    "reinforcement_learning": ["reinforcement learning", "reward", "policy gradient", "q-learning", "rl agent"],
    "nlp": ["natural language", "language model", "text classification", "sentiment", "named entity", "machine translation", "parsing"],
    "generative": ["generative", "gan", "vae", "diffusion", "autoregressive", "generation model"],
    "graph_methods": ["graph neural", "graph convolution", "knowledge graph", "graph embedding", "network analysis"],
    "bayesian": ["bayesian", "posterior", "prior distribution", "mcmc", "variational inference"],
    "optimization": ["optimization", "gradient descent", "convex", "linear programming", "combinatorial optimization"],
    "statistical": ["statistical", "regression", "hypothesis test", "estimator", "confidence interval"],
    "simulation": ["simulation", "monte carlo", "numerical", "finite element", "computational"],
    "survey": ["survey", "review", "overview", "taxonomy", "comprehensive study"],
    "theoretical": ["theorem", "proof", "lemma", "conjecture", "bound", "complexity analysis"],
    "empirical": ["experiment", "benchmark", "evaluation", "ablation", "baseline comparison"],
}

PROBLEM_KEYWORDS = {
    "classification": ["classification", "classifier", "categorization", "recognition"],
    "generation": ["generation", "synthesis", "produce", "creating new"],
    "detection": ["detection", "anomaly detection", "object detection", "intrusion"],
    "prediction": ["prediction", "forecasting", "predicting", "prognostic"],
    "clustering": ["clustering", "segmentation", "grouping", "partitioning"],
    "recommendation": ["recommendation", "collaborative filtering", "personalization"],
    "retrieval": ["retrieval", "search", "ranking", "information retrieval"],
    "optimization_prob": ["scheduling", "allocation", "assignment", "routing problem"],
    "representation": ["representation learning", "embedding", "feature learning", "dimensionality reduction"],
    "understanding": ["understanding", "comprehension", "interpretation", "reasoning"],
    "estimation": ["estimation", "measurement", "parameter estimation", "inference"],
    "modeling": ["modeling", "model", "framework", "architecture"],
}

DATA_KEYWORDS = {
    "images": ["image", "visual", "pixel", "photograph", "video"],
    "text": ["text", "document", "corpus", "sentence", "paragraph", "word"],
    "graphs": ["graph", "network", "node", "edge", "adjacency"],
    "tabular": ["table", "database", "structured data", "csv", "relational"],
    "time_series": ["time series", "temporal", "sequential", "longitudinal"],
    "genomic": ["genome", "gene", "dna", "rna", "protein", "sequence", "mutation"],
    "astronomical": ["galaxy", "stellar", "cosmological", "redshift", "photometric", "spectr"],
    "audio": ["audio", "speech", "sound", "acoustic", "voice"],
    "medical": ["clinical", "patient", "medical", "health", "disease", "diagnosis"],
    "social": ["social media", "user", "twitter", "community", "online"],
    "simulation_data": ["simulated", "synthetic", "generated data", "artificial"],
    "mathematical": ["mathematical", "algebraic", "geometric", "topological", "combinatorial"],
}

CONTRIBUTION_KEYWORDS = {
    "new_method": ["we propose", "we introduce", "we present", "novel method", "new approach", "new algorithm"],
    "improvement": ["improve", "enhance", "outperform", "state-of-the-art", "better than", "superior"],
    "analysis": ["we analyze", "we study", "we investigate", "we examine", "characterize"],
    "application": ["we apply", "application", "real-world", "practical", "deployed"],
    "framework": ["framework", "system", "pipeline", "platform", "toolkit"],
    "benchmark": ["benchmark", "dataset", "evaluation suite", "test bed"],
    "theoretical_contrib": ["we prove", "we show that", "theoretical contribution", "lower bound", "upper bound"],
}


def extract_field(abstract, keyword_map):
    """Extract the best-matching label from an abstract."""
    abstract_lower = abstract.lower()
    scores = {}
    for label, keywords in keyword_map.items():
        score = sum(1 for kw in keywords if kw in abstract_lower)
        if score > 0:
            scores[label] = score
    if scores:
        return max(scores, key=scores.get)
    return "other"


# Extract fields for each paper
semantic_records = []
for paper in papers:
    abstract = paper["abstract"]

    rec = {
        "arxiv_id": paper["arxiv_id"],
        "primary_category": paper["primary_category"],
        "year": paper["year"],
        "method": extract_field(abstract, METHOD_KEYWORDS),
        "problem_type": extract_field(abstract, PROBLEM_KEYWORDS),
        "data_type": extract_field(abstract, DATA_KEYWORDS),
        "contribution": extract_field(abstract, CONTRIBUTION_KEYWORDS),
        "title": paper["title"][:100],
    }
    semantic_records.append(rec)

# Save
with open("arxiv_semantic.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=semantic_records[0].keys())
    w.writeheader()
    w.writerows(semantic_records)

print(f"Extracted semantic fields for {len(semantic_records)} papers")
print()

# Show distributions
for field_name in ["method", "problem_type", "data_type", "contribution"]:
    dist = Counter(r[field_name] for r in semantic_records)
    print(f"{field_name}:")
    for val, count in dist.most_common():
        print(f"  {val}: {count}")
    print()

# ---------------------------------------------------------------
# Now run DW on the semantic fields
# ---------------------------------------------------------------
print("=" * 60)
print("  ROUTING MEANING: DW on semantic fields from abstracts")
print("=" * 60)

import database_whisper as dw

# Profile with category as identity — what distinguishes papers
# within the same field of science?
report = dw.profile_records(
    semantic_records,
    source="arxiv_semantic.csv",
    identity_fields=["primary_category"],
    provenance_fields=["arxiv_id", "title"],
)
print()
print(report)

# ---------------------------------------------------------------
# Gap detection: meaningful semantic gaps
# ---------------------------------------------------------------
print()
print("=" * 60)
print("  SEMANTIC GAP DETECTION")
print("  What method + data_type + problem combinations are missing?")
print("=" * 60)
print()

# Build the combination space
triples = defaultdict(lambda: defaultdict(set))
for rec in semantic_records:
    cat = rec["primary_category"]
    combo = (rec["method"], rec["data_type"], rec["problem_type"])
    triples[cat][combo].add(rec["arxiv_id"])

# What combos exist globally?
all_methods = sorted(set(r["method"] for r in semantic_records) - {"other"})
all_data = sorted(set(r["data_type"] for r in semantic_records) - {"other"})
all_problems = sorted(set(r["problem_type"] for r in semantic_records) - {"other"})

major_cats = [c for c, _ in Counter(r["primary_category"] for r in semantic_records).most_common(8)]

# Find gaps: combos that exist in 3+ other categories but not in this one
print("Cross-category semantic gaps:")
print("(combinations present in 3+ other fields but missing from this one)")
print()

gaps = []
for cat in major_cats:
    existing_combos = set(triples[cat].keys())
    for method in all_methods:
        for data_type in all_data:
            for problem in all_problems:
                combo = (method, data_type, problem)
                if combo in existing_combos:
                    continue
                # How many other major categories have this combo?
                other_count = sum(1 for c in major_cats if c != cat and combo in triples[c])
                if other_count >= 2:
                    gaps.append({
                        "category": cat,
                        "method": method,
                        "data_type": data_type,
                        "problem_type": problem,
                        "present_in": other_count,
                    })

# Sort by most surprising (present in most other categories)
gaps.sort(key=lambda g: -g["present_in"])

print(f"Total semantic gaps found: {len(gaps)}")
print()
for g in gaps[:25]:
    print(f"  {g['category']}")
    print(f"    Missing: {g['method']} + {g['data_type']} + {g['problem_type']}")
    print(f"    Present in {g['present_in']} other fields")
    print()

# ---------------------------------------------------------------
# The real test: does the LADDER discover meaningful structure?
# ---------------------------------------------------------------
print("=" * 60)
print("  THE REAL TEST: Does DW discover semantic structure?")
print("=" * 60)
print()

# Route with method as identity — what distinguishes papers
# that use the same method?
report2 = dw.profile_records(
    semantic_records,
    source="arxiv_semantic.csv (by method)",
    identity_fields=["method"],
    provenance_fields=["arxiv_id", "title"],
)
print(report2)

print()

# Route with problem_type as identity
report3 = dw.profile_records(
    semantic_records,
    source="arxiv_semantic.csv (by problem)",
    identity_fields=["problem_type"],
    provenance_fields=["arxiv_id", "title"],
)
print(report3)

print()

# The multi-axis test: identity = method + data_type
# What distinguishes papers with the same method applied to the same data type?
report4 = dw.profile_records(
    semantic_records,
    source="arxiv_semantic.csv (by method+data)",
    identity_fields=["method", "data_type"],
    provenance_fields=["arxiv_id", "title"],
)
print(report4)
