"""Bootstrap Variance for Cross-Domain Audit (Table 4.1 stability check)

Reviewer 2 asks: are the cross-domain numbers stable, or an artefact of
which 2000 records happened to land in the sample?

Method: 5 bootstrap resamples of 2000 records from MTSamples, full DW
pipeline on each.  Report mean +/- std for confusion, completeness,
predictability.
"""

import sys, os, io, subprocess, random, time

# ── Windows encoding fix ────────────────────────────────────────────
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def ensure_package(name, pip_name=None):
    try:
        __import__(name)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pip_name or name, "-q"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

ensure_package("datasets")

from datasets import load_dataset
import database_whisper as dw

# ── Config ──────────────────────────────────────────────────────────
N_RESAMPLES = 5
SAMPLE_SIZE = 2000
SEED_BASE   = 42

# ── Load MTSamples once ────────────────────────────────────────────
print("Loading MTSamples from HuggingFace...")
ds = load_dataset("harishnair04/mtsamples", split="train")

all_records = []
for i, row in enumerate(ds):
    text = row.get("transcription") or row.get("description") or ""
    if len(text) >= 50:
        all_records.append({"id": f"mt_{i}", "text": text})

print(f"  {len(all_records)} valid records (>= 50 chars) from {len(ds)} total")

# ── Run one full pipeline pass ──────────────────────────────────────
def run_pipeline(records, label=""):
    """Return (confusion, completeness, predictability) or None on failure."""
    t0 = time.time()
    print(f"\n  [{label}] Auto-detecting concepts on {len(records)} records...")
    concepts = dw.auto_detect_concepts(records, text_field="text", min_freq=20)
    if len(concepts) > 30:
        concepts = concepts[:30]
    print(f"    {len(concepts)} concepts (capped at 30)")

    print(f"  [{label}] Extracting instances...")
    instances = dw.extract_concept_instances(
        records, text_field="text", concepts=concepts, provenance_field="id"
    )
    print(f"    {len(instances)} instances")
    if len(instances) < 50:
        print(f"    SKIP — too few instances")
        return None

    print(f"  [{label}] Profiling...")
    profile = dw.profile_records(
        instances,
        identity_fields=["concept"],
        provenance_fields=["_provenance", "_context", "_occurrence"],
        max_ladder_depth=5,
    )
    ladder = profile.ladder_fields
    print(f"    Ladder: {ladder}")

    print(f"  [{label}] Computing meaning addresses...")
    addresses = dw.meaning_addresses(instances, ladder_fields=ladder)
    n_addrs = sum(len(v) for v in addresses.values())
    print(f"    {n_addrs} raw addresses")

    print(f"  [{label}] Running meaning audit...")
    audit = dw.meaning_audit(addresses)
    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s  —  "
          f"confusion={audit.confusion:.4f}  "
          f"completeness={audit.completeness:.4f}  "
          f"predictability={audit.predictability:.4f}")
    return (audit.confusion, audit.completeness, audit.predictability)


# ── Bootstrap loop ──────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"  BOOTSTRAP VARIANCE: {N_RESAMPLES} resamples of {SAMPLE_SIZE} records")
print(f"{'=' * 70}")

results = []  # list of (confusion, completeness, predictability)

for k in range(N_RESAMPLES):
    rng = random.Random(SEED_BASE + k)
    # sample WITH replacement
    sample = rng.choices(all_records, k=SAMPLE_SIZE)
    # re-id to avoid duplicate ids
    sample = [{"id": f"bs{k}_{i}", "text": r["text"]} for i, r in enumerate(sample)]

    scores = run_pipeline(sample, label=f"Resample {k+1}/{N_RESAMPLES}")
    if scores is not None:
        results.append(scores)
    else:
        print(f"  *** Resample {k+1} failed — skipping")


# ── Report ──────────────────────────────────────────────────────────
print(f"\n\n{'=' * 70}")
print(f"  BOOTSTRAP RESULTS  ({len(results)}/{N_RESAMPLES} successful resamples)")
print(f"{'=' * 70}")

if len(results) < 2:
    print("  Not enough successful resamples to compute variance.")
    sys.exit(1)

header = f"  {'Resample':<12s}  {'Confusion':>10s}  {'Completeness':>13s}  {'Predictability':>15s}"
print(header)
print(f"  {'─' * 12}  {'─' * 10}  {'─' * 13}  {'─' * 15}")

for i, (conf, comp, pred) in enumerate(results):
    print(f"  {'Sample ' + str(i+1):<12s}  {conf:>10.4f}  {comp:>13.4f}  {pred:>15.4f}")

# mean +/- std
import statistics
confs  = [r[0] for r in results]
comps  = [r[1] for r in results]
preds  = [r[2] for r in results]

mean_conf = statistics.mean(confs)
mean_comp = statistics.mean(comps)
mean_pred = statistics.mean(preds)
std_conf  = statistics.stdev(confs) if len(confs) > 1 else 0.0
std_comp  = statistics.stdev(comps) if len(comps) > 1 else 0.0
std_pred  = statistics.stdev(preds) if len(preds) > 1 else 0.0

print(f"  {'─' * 12}  {'─' * 10}  {'─' * 13}  {'─' * 15}")
print(f"  {'Mean':<12s}  {mean_conf:>10.4f}  {mean_comp:>13.4f}  {mean_pred:>15.4f}")
print(f"  {'Std':<12s}  {std_conf:>10.4f}  {std_comp:>13.4f}  {std_pred:>15.4f}")
print(f"  {'Mean +/- Std':<12s}  {mean_conf:.4f}+/-{std_conf:.4f}  "
      f"{mean_comp:.4f}+/-{std_comp:.4f}  {mean_pred:.4f}+/-{std_pred:.4f}")

# CV (coefficient of variation) — how noisy relative to the signal?
cv_conf = (std_conf / mean_conf * 100) if mean_conf > 0 else 0
cv_comp = (std_comp / mean_comp * 100) if mean_comp > 0 else 0
cv_pred = (std_pred / mean_pred * 100) if mean_pred > 0 else 0

print(f"\n  Coefficient of Variation:")
print(f"    Confusion:      {cv_conf:.1f}%")
print(f"    Completeness:   {cv_comp:.1f}%")
print(f"    Predictability: {cv_pred:.1f}%")

if max(cv_conf, cv_comp, cv_pred) < 10:
    print(f"\n  All CVs < 10% — scores are stable across bootstrap resamples.")
elif max(cv_conf, cv_comp, cv_pred) < 20:
    print(f"\n  CVs < 20% — moderate stability; scores vary somewhat with sampling.")
else:
    print(f"\n  Some CVs >= 20% — meaningful variance; report confidence intervals.")

print("\nDone.")
