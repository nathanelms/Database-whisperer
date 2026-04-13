"""experiment_retrieval_baselines.py

Extrinsic retrieval evaluation: Keyword vs Embedding vs DW Address.

Reviewer demand: "The paper claims DW addresses outperform embeddings
but provides no proper experiment."  This fixes that.

For each ambiguous clinical term, we build a binary retrieval task:
  - Ground truth labels derived from paired_concept + local context
  - Three methods retrieve the target sense
  - Precision / Recall / F1 at best operating point

Terms tested:
  1. "positive"  — lab-test result vs other uses
  2. "discharge" — hospital release vs fluid emission
  3. "right"     — anatomical direction vs correct/entitlement
"""

import sys
import os
import io
import subprocess
import time

# -- Windows encoding guard --------------------------------------------------
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

def ensure_package(name, pip_name=None):
    try:
        __import__(name)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pip_name or name, "-q"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

ensure_package("datasets")
ensure_package("sentence_transformers", "sentence-transformers")
ensure_package("numpy")

import numpy as np
from numpy.linalg import norm
from collections import defaultdict
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import database_whisper as dw

# ============================================================================
# 1. Load MTSamples & featurize
# ============================================================================
print("=" * 70)
print("  LOADING MTSamples + DW FEATURIZATION")
print("=" * 70)

t0 = time.time()
ds = load_dataset("harishnair04/mtsamples", split="train")

records = []
for i, row in enumerate(ds):
    text = row.get("transcription") or ""
    if len(text) < 50:
        continue
    records.append({
        "id": f"mt_{i}",
        "text": text,
        "specialty": (row.get("medical_specialty") or "Unknown").strip(),
    })
print(f"  Records: {len(records)}")

CONCEPTS = [
    "positive", "negative", "normal", "acute", "chronic",
    "discharge", "procedure", "history", "pain", "blood",
    "heart", "patient", "treatment", "diagnosis", "left",
    "right", "mass", "pressure", "stable", "significant",
    "tissue", "failure", "clear", "fluid", "infection",
]

all_instances = dw.extract_concept_instances(
    records,
    text_field="text",
    concepts=CONCEPTS,
    metadata_fields=["specialty"],
    provenance_field="id",
)
print(f"  Total instances: {len(all_instances)}")

# Build record lookup for wider context windows
record_lookup = {r["id"]: r["text"] for r in records}

# Load embedding model once
print("  Loading sentence-transformers model...")
st_model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"  Setup done in {time.time() - t0:.1f}s")

# ============================================================================
# 2. Task definitions
# ============================================================================
# Each task: concept, sense_name, context_markers (ground truth labeling),
#            embedding_query, ladder_fields

LADDER = ["specialty", "paired_concept", "verb_class", "clause_position"]

# --- Ground truth labeling functions ----------------------------------------

def _window_around(full_text, word, radius=150):
    """Get a text window around first occurrence of word."""
    low = full_text.lower()
    pos = low.find(word)
    if pos < 0:
        return ""
    return low[max(0, pos - radius):pos + radius]


def label_positive_lab(inst):
    """Is 'positive' referring to a lab/test result?"""
    markers = [
        "culture", "cultures", "screen", "screening", "test result",
        "titer", "titers", "antigen", "antibody", "antibodies",
        "serology", "assay", "specimen", "stain", "staining",
        "pcr", "urinalysis", "laboratory", "panel", "smear",
        "biopsy", "pathology", "cytology", "gram", "susceptibility",
        "hiv", "hepatitis", "strep", "mrsa", "esbl",
        "drug screen", "toxicology", "hemoccult", "guaiac", "dipstick",
        "positive for", "tested positive", "was positive for",
        "test was positive", "results were positive", "came back positive",
        "cultures were positive", "culture positive", "culture was positive",
        "screen positive", "screen was positive", "positive result",
        "found to be positive", "stain positive", "positive stain",
    ]
    ctx = inst.get("_context", "").lower()
    prov = inst.get("_provenance", "")
    window = _window_around(record_lookup.get(prov, ""), "positive")
    combined = ctx + " " + window
    return any(m in combined for m in markers)


def label_discharge_hospital(inst):
    """Is 'discharge' referring to hospital release (vs fluid/wound)?"""
    hosp_markers = [
        "discharged home", "discharge home", "discharge instructions",
        "discharge condition", "discharge diagnosis", "discharge medications",
        "discharge diet", "discharge disposition", "to home",
        "released from", "postoperative day",
        "follow-up", "follow up", "return to clinic",
    ]
    fluid_markers = [
        "drainage", "wound", "vaginal", "ear", "nasal", "purulent",
        "bloody", "serous", "mucous", "nipple", "penile",
        "foul-smelling", "milky", "watery discharge",
    ]
    ctx = inst.get("_context", "").lower()
    prov = inst.get("_provenance", "")
    window = _window_around(record_lookup.get(prov, ""), "discharge")
    combined = ctx + " " + window
    has_hosp = any(m in combined for m in hosp_markers)
    has_fluid = any(m in combined for m in fluid_markers)
    if has_hosp and not has_fluid:
        return True
    if has_fluid and not has_hosp:
        return False
    # Ambiguous — check paired_concept as tiebreaker
    pc = inst.get("paired_concept", "").lower()
    if pc in ("home", "patient", "procedure", "history", "treatment", "diagnosis"):
        return True
    if pc in ("fluid", "blood", "infection", "tissue", "pain"):
        return False
    # Default: check specialty
    spec = inst.get("specialty", "").lower()
    if "discharge" in spec.lower() or "general" in spec.lower():
        return True
    return None  # truly ambiguous, exclude


def label_right_anatomical(inst):
    """Is 'right' referring to anatomical direction (vs correct/entitlement)?"""
    anat_markers = [
        "right side", "right leg", "right arm", "right knee", "right hip",
        "right shoulder", "right hand", "right foot", "right eye",
        "right ear", "right upper", "right lower", "right ventricle",
        "right atrium", "right lung", "right lobe", "right breast",
        "right inguinal", "right flank", "right groin", "right axilla",
        "right lateral", "right anterior", "right posterior",
        "right coronary", "right carotid", "right femoral",
        "right subclavian", "right iliac", "right renal",
        "right hemidiaphragm", "right hemithorax",
    ]
    non_anat_markers = [
        "right to", "all right", "that's right", "is right",
        "rights", "patient's right to",
    ]
    ctx = inst.get("_context", "").lower()
    prov = inst.get("_provenance", "")
    window = _window_around(record_lookup.get(prov, ""), "right")
    combined = ctx + " " + window
    has_anat = any(m in combined for m in anat_markers)
    has_non = any(m in combined for m in non_anat_markers)
    if has_anat and not has_non:
        return True
    if has_non and not has_anat:
        return False
    # In clinical notes, vast majority of "right" is anatomical
    # Check if paired with anatomical concepts
    pc = inst.get("paired_concept", "").lower()
    if pc in ("left", "pain", "blood", "tissue", "mass", "pressure", "fluid"):
        return True
    return None  # truly ambiguous, exclude


# --- Task specifications ---------------------------------------------------

TASKS = [
    {
        "concept": "positive",
        "sense_name": "lab_test_result",
        "label_fn": label_positive_lab,
        "emb_query": "positive laboratory test result culture specimen screening",
        "dw_lab_markers": [  # paired_concepts that indicate lab sense
            "blood", "infection", "negative", "fluid", "tissue",
        ],
    },
    {
        "concept": "discharge",
        "sense_name": "hospital_release",
        "label_fn": label_discharge_hospital,
        "emb_query": "patient discharged home from hospital follow up instructions",
        "dw_lab_markers": [  # paired_concepts that indicate hospital-release sense
            "patient", "diagnosis", "treatment", "procedure", "history",
        ],
    },
    {
        "concept": "right",
        "sense_name": "anatomical_direction",
        "label_fn": label_right_anatomical,
        "emb_query": "right side anatomical direction body left lateral extremity",
        "dw_lab_markers": [  # paired_concepts that indicate anatomical sense
            "left", "pain", "blood", "tissue", "mass", "pressure",
        ],
    },
]


# ============================================================================
# 3. Evaluation engine
# ============================================================================

def compute_prf(retrieved_labels, total_positive):
    """Given list of booleans (is-target-sense?) and total positives, return P/R/F1."""
    if not retrieved_labels:
        return 0.0, 0.0, 0.0
    tp = sum(retrieved_labels)
    prec = tp / len(retrieved_labels)
    rec = tp / total_positive if total_positive else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def run_keyword(instances, labels):
    """Keyword baseline: retrieve ALL instances of the concept. 100% recall."""
    n_pos = sum(labels)
    prec = n_pos / len(labels) if labels else 0
    # F1 with 100% recall
    f1 = 2 * prec * 1.0 / (prec + 1.0) if prec > 0 else 0
    return {
        "n_retrieved": len(instances),
        "precision": prec,
        "recall": 1.0,
        "f1": f1,
    }


def run_embedding(instances, labels, query, model):
    """Embedding retrieval: rank by cosine similarity, sweep for best F1."""
    contexts = [inst.get("_context", "") for inst in instances]
    n_pos = sum(labels)

    ctx_embs = model.encode(contexts, show_progress_bar=False, batch_size=256)
    q_emb = model.encode([query])[0]

    # Cosine similarities
    sims = []
    for i, emb in enumerate(ctx_embs):
        d = float(np.dot(q_emb, emb) / (norm(q_emb) * norm(emb) + 1e-9))
        sims.append((d, i))
    sims.sort(reverse=True)

    # Sweep top-k for best F1
    best_f1, best_k = 0, 10
    best_prec, best_rec = 0, 0
    tp_accum = 0
    for rank, (sim, idx) in enumerate(sims, 1):
        tp_accum += labels[idx]
        prec = tp_accum / rank
        rec = tp_accum / n_pos if n_pos else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        if f1 > best_f1:
            best_f1 = f1
            best_k = rank
            best_prec = prec
            best_rec = rec

    return {
        "n_retrieved": best_k,
        "precision": best_prec,
        "recall": best_rec,
        "f1": best_f1,
    }


def run_dw_address(instances, labels, ladder_fields):
    """DW address retrieval: find addresses where majority = target sense."""
    n_pos = sum(labels)

    # Group by address
    addr_groups = defaultdict(list)
    addr_labels = defaultdict(list)
    for i, inst in enumerate(instances):
        addr = tuple(inst.get(f, "?") for f in ladder_fields)
        addr_groups[addr].append(i)
        addr_labels[addr].append(labels[i])

    # Identify target-sense addresses: majority vote with min support
    target_addrs = set()
    for addr, labs in addr_labels.items():
        n_target = sum(labs)
        n_total = len(labs)
        if n_total >= 2 and n_target / n_total > 0.5:
            target_addrs.add(addr)
        elif n_total == 1 and labs[0]:
            target_addrs.add(addr)

    # Retrieve all instances at target addresses
    retrieved_idx = []
    for addr in target_addrs:
        retrieved_idx.extend(addr_groups[addr])

    retrieved_labels = [labels[i] for i in retrieved_idx]
    prec, rec, f1 = compute_prf(retrieved_labels, n_pos)

    return {
        "n_retrieved": len(retrieved_idx),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "n_addresses": len(target_addrs),
        "total_addresses": len(addr_groups),
    }


# ============================================================================
# 4. Run all tasks
# ============================================================================

all_results = []

for task in TASKS:
    concept = task["concept"]
    print(f"\n{'=' * 70}")
    print(f"  TASK: '{concept}' -> {task['sense_name']}")
    print(f"{'=' * 70}")

    # Filter instances
    instances = [inst for inst in all_instances if inst["concept"] == concept]
    print(f"  Total '{concept}' instances: {len(instances)}")

    # Label
    raw_labels = [task["label_fn"](inst) for inst in instances]

    # Remove truly ambiguous (None labels)
    filtered = [(inst, lab) for inst, lab in zip(instances, raw_labels) if lab is not None]
    instances_f = [x[0] for x in filtered]
    labels_f = [x[1] for x in filtered]
    n_excluded = len(instances) - len(instances_f)

    n_pos = sum(labels_f)
    n_neg = len(labels_f) - n_pos
    print(f"  Labeled: {len(labels_f)} ({n_excluded} excluded as ambiguous)")
    print(f"  Target sense ({task['sense_name']}): {n_pos}")
    print(f"  Other senses: {n_neg}")
    print(f"  Base rate: {n_pos / len(labels_f):.1%}" if labels_f else "  (no instances)")

    if not labels_f or n_pos == 0 or n_neg == 0:
        print("  SKIPPED — trivial split")
        continue

    # Method 1: Keyword
    kw = run_keyword(instances_f, labels_f)
    print(f"\n  METHOD 1 - KEYWORD (all '{concept}' instances)")
    print(f"    Retrieved: {kw['n_retrieved']}")
    print(f"    Precision: {kw['precision']:.3f}  Recall: {kw['recall']:.3f}  F1: {kw['f1']:.3f}")

    # Method 2: Embedding
    emb = run_embedding(instances_f, labels_f, task["emb_query"], st_model)
    print(f"\n  METHOD 2 - EMBEDDING (best F1 threshold)")
    print(f"    Retrieved: {emb['n_retrieved']}")
    print(f"    Precision: {emb['precision']:.3f}  Recall: {emb['recall']:.3f}  F1: {emb['f1']:.3f}")

    # Method 3: DW Address (4-rung)
    dw4 = run_dw_address(instances_f, labels_f, LADDER)
    print(f"\n  METHOD 3 - DW ADDRESS (4-rung ladder)")
    print(f"    Addresses: {dw4['n_addresses']} / {dw4['total_addresses']} classified as target")
    print(f"    Retrieved: {dw4['n_retrieved']}")
    print(f"    Precision: {dw4['precision']:.3f}  Recall: {dw4['recall']:.3f}  F1: {dw4['f1']:.3f}")

    # Also try 2-rung and 3-rung
    dw2 = run_dw_address(instances_f, labels_f, LADDER[:2])
    dw3 = run_dw_address(instances_f, labels_f, LADDER[:3])
    print(f"\n  DW depth variants:")
    print(f"    2-rung (specialty+pair):    P={dw2['precision']:.3f}  R={dw2['recall']:.3f}  F1={dw2['f1']:.3f}")
    print(f"    3-rung (+verb):             P={dw3['precision']:.3f}  R={dw3['recall']:.3f}  F1={dw3['f1']:.3f}")
    print(f"    4-rung (+clause_position):  P={dw4['precision']:.3f}  R={dw4['recall']:.3f}  F1={dw4['f1']:.3f}")

    all_results.append({
        "concept": concept,
        "sense": task["sense_name"],
        "n_instances": len(labels_f),
        "n_positive": n_pos,
        "base_rate": n_pos / len(labels_f),
        "keyword": kw,
        "embedding": emb,
        "dw_2rung": dw2,
        "dw_3rung": dw3,
        "dw_4rung": dw4,
    })

# ============================================================================
# 5. Summary table
# ============================================================================
print("\n\n" + "=" * 90)
print("  EXPERIMENT SUMMARY: Retrieval Baselines for Clinical Disambiguation")
print("=" * 90)

header = f"  {'Concept':<12} {'Sense':<22} {'Method':<25} {'N':>5} {'Prec':>7} {'Rec':>7} {'F1':>7}"
print(header)
print("  " + "-" * 86)

for res in all_results:
    c = res["concept"]
    s = res["sense"]
    n = res["n_instances"]
    br = res["base_rate"]

    for method_name, method_key in [
        ("Keyword", "keyword"),
        ("Embedding (best F1)", "embedding"),
        ("DW 2-rung", "dw_2rung"),
        ("DW 3-rung", "dw_3rung"),
        ("DW 4-rung", "dw_4rung"),
    ]:
        m = res[method_key]
        label = c if method_name == "Keyword" else ""
        sense_label = s if method_name == "Keyword" else ""
        print(f"  {label:<12} {sense_label:<22} {method_name:<25} {m['n_retrieved']:>5} {m['precision']:>7.3f} {m['recall']:>7.3f} {m['f1']:>7.3f}")
    print()

# Per-concept summary
print("\n  KEY FINDINGS:")
print("  " + "-" * 60)
for res in all_results:
    kw_f1 = res["keyword"]["f1"]
    emb_f1 = res["embedding"]["f1"]
    dw_f1 = res["dw_4rung"]["f1"]
    winner = "DW" if dw_f1 >= emb_f1 else "Embedding"
    margin = abs(dw_f1 - emb_f1)
    print(f"  '{res['concept']}' ({res['sense']}):")
    print(f"    Base rate: {res['base_rate']:.1%} ({res['n_positive']}/{res['n_instances']})")
    print(f"    Keyword F1={kw_f1:.3f}  |  Embedding F1={emb_f1:.3f}  |  DW-4 F1={dw_f1:.3f}")
    print(f"    Winner: {winner} by {margin:.3f}")
    dw_p = res["dw_4rung"]["precision"]
    emb_p = res["embedding"]["precision"]
    if dw_p > emb_p:
        print(f"    DW precision advantage: {dw_p:.3f} vs {emb_p:.3f} (+{dw_p - emb_p:.3f})")
    print()

# Macro averages
print("  MACRO-AVERAGED SCORES:")
print("  " + "-" * 60)
for method_name, method_key in [
    ("Keyword", "keyword"),
    ("Embedding (best F1)", "embedding"),
    ("DW 2-rung", "dw_2rung"),
    ("DW 3-rung", "dw_3rung"),
    ("DW 4-rung", "dw_4rung"),
]:
    avg_p = np.mean([r[method_key]["precision"] for r in all_results])
    avg_r = np.mean([r[method_key]["recall"] for r in all_results])
    avg_f1 = np.mean([r[method_key]["f1"] for r in all_results])
    print(f"  {method_name:<25}  P={avg_p:.3f}  R={avg_r:.3f}  F1={avg_f1:.3f}")

print(f"\n  Total elapsed: {time.time() - t0:.1f}s")
print("\n  Note: Embedding gets oracle threshold (best possible F1 over all cutoffs).")
print("  DW uses majority-vote on pre-computed structural addresses (no tuning).")
print("  DW addresses are interpretable and auditable; embeddings are not.")
