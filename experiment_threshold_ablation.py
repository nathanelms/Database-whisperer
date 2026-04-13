"""Threshold Sensitivity Ablation — Jaccard collapse threshold.

Reviewer demand: show what happens to the four audit scores when
the Jaccard similarity threshold for address collapse varies from
0.3 to 0.7 (paper uses 0.5).

Dataset: MTSamples clinical text (same as main experiment).
"""

import sys, os, io, subprocess

# -- Windows encoding fix --------------------------------------------------
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

# ═══════════════════════════════════════════════════════════════════
# 1. Load MTSamples
# ═══════════════════════════════════════════════════════════════════

N = 2000
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

print("Loading MTSamples...")
ds = load_dataset("harishnair04/mtsamples", split="train")
records = [
    {"id": f"mt_{i}", "text": row.get("transcription") or row.get("description") or ""}
    for i, row in enumerate(ds)
    if len(row.get("transcription") or row.get("description") or "") > 50
][:N]
print(f"  {len(records)} records")

# ═══════════════════════════════════════════════════════════════════
# 2. Extract concepts + instances + addresses (shared across thresholds)
# ═══════════════════════════════════════════════════════════════════

print("Auto-detecting concepts...")
concepts = dw.auto_detect_concepts(records, text_field="text")
if len(concepts) > 30:
    concepts = concepts[:30]
print(f"  {len(concepts)} concepts: {concepts[:10]}...")

print("Extracting instances...")
instances = dw.extract_concept_instances(
    records, text_field="text", concepts=concepts, provenance_field="id"
)
print(f"  {len(instances)} instances")

print("Profiling...")
profile = dw.profile_records(
    instances,
    identity_fields=["concept"],
    provenance_fields=["_provenance", "_context", "_occurrence"],
)
ladder = profile.ladder_fields
print(f"  Ladder: {ladder}")

print("Computing addresses...")
addresses = dw.meaning_addresses(instances, ladder_fields=ladder)
n_raw = sum(len(v) for v in addresses.values())
print(f"  {n_raw} raw addresses")

# ═══════════════════════════════════════════════════════════════════
# 3. Run meaning_audit at each threshold
# ═══════════════════════════════════════════════════════════════════

results = []

for t in THRESHOLDS:
    print(f"\n  Running audit at threshold={t:.1f}...")
    audit = dw.meaning_audit(addresses, similarity_threshold=t)
    results.append({
        "threshold": t,
        "raw_addresses": audit.collapse_stats["raw_addresses"],
        "resolved_meanings": audit.collapse_stats["resolved_meanings"],
        "confusion": audit.confusion,
        "completeness": audit.completeness,
        "predictability": audit.predictability,
    })

# ═══════════════════════════════════════════════════════════════════
# 4. Print results table
# ═══════════════════════════════════════════════════════════════════

print("\n")
print("=" * 78)
print("THRESHOLD SENSITIVITY ABLATION — Jaccard collapse threshold")
print("=" * 78)
print(f"  Dataset:   MTSamples ({len(records)} records, {len(concepts)} concepts)")
print(f"  Instances: {len(instances)}")
print(f"  Thresholds tested: {THRESHOLDS}")
print()

hdr = (f"  {'Thresh':>6s}  {'Raw Addr':>9s}  {'Resolved':>9s}  "
       f"{'Confusion':>10s}  {'Complete':>9s}  {'Predict':>8s}")
print(hdr)
print(f"  {'------':>6s}  {'---------':>9s}  {'---------':>9s}  "
      f"{'----------':>10s}  {'---------':>9s}  {'--------':>8s}")

for r in results:
    print(f"  {r['threshold']:>6.1f}  {r['raw_addresses']:>9d}  {r['resolved_meanings']:>9d}  "
          f"{r['confusion']:>10.4f}  {r['completeness']:>9.4f}  {r['predictability']:>8.4f}")

# Stability summary
confusions = [r["confusion"] for r in results]
completes = [r["completeness"] for r in results]
predicts = [r["predictability"] for r in results]

print()
print(f"  Score ranges across thresholds:")
print(f"    Confusion:      {min(confusions):.4f} - {max(confusions):.4f}  "
      f"(range = {max(confusions)-min(confusions):.4f})")
print(f"    Completeness:   {min(completes):.4f} - {max(completes):.4f}  "
      f"(range = {max(completes)-min(completes):.4f})")
print(f"    Predictability: {min(predicts):.4f} - {max(predicts):.4f}  "
      f"(range = {max(predicts)-min(predicts):.4f})")

# Monotonicity check: resolved meanings should decrease as threshold drops
resolved = [r["resolved_meanings"] for r in results]
monotone_desc = all(resolved[i] >= resolved[i+1] for i in range(len(resolved)-1))
monotone_asc = all(resolved[i] <= resolved[i+1] for i in range(len(resolved)-1))

print()
if monotone_asc:
    print("  Resolved meanings increase monotonically with threshold (expected:")
    print("  stricter threshold = less merging = more distinct meanings).")
elif monotone_desc:
    print("  Resolved meanings decrease monotonically with threshold (looser")
    print("  threshold = more merging = fewer distinct meanings).")
else:
    print("  Resolved meanings are non-monotonic — worth investigating.")

print()
print("Done.")
