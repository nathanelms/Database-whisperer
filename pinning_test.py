"""pinning_test.py

Five variations of the options pinning hypothesis.
Uses SPY options OI + ES 1-minute data.

Does price gravitate toward the strike with highest open interest?
"""

import pandas as pd
import numpy as np
from scipy import stats

# Load ES 1-minute
print("Loading ES 1-minute data...")
es = pd.read_csv(
    "C:/Users/User/Dropbox/ExcelDataDump/@ES 1 Minute.txt",
    names=["date", "time", "open", "high", "low", "close", "volume", "p1", "p2"],
    skiprows=1,
)
es["date"] = pd.to_datetime(es["date"])

# RTH only
es_rth = es[(es["time"] >= "09:30:00") & (es["time"] <= "16:00:00")].copy()
print(f"ES RTH bars: {len(es_rth):,}")

daily_es = es_rth.groupby("date").agg(
    es_open=("open", "first"),
    es_close=("close", "last"),
    es_high=("high", "max"),
    es_low=("low", "min"),
).reset_index()
print(f"ES trading days: {len(daily_es)}")

# Load SPY options
print("Loading SPY options...")
opts = pd.read_parquet(
    "C:/Users/User/Dropbox/ExcelDataDump/DataWarehouse/options/spy_options.parquet"
)
opts["date"] = pd.to_datetime(opts["date"])
opts["expiration"] = pd.to_datetime(opts["expiration"])
opts["dte"] = (opts["expiration"] - opts["date"]).dt.days

# Near-expiry options
near = opts[opts["dte"] <= 3].copy()
print(f"Near-expiry options: {len(near):,}")

# Per-date: find strike with max OI
date_oi = (
    near.groupby("date")
    .apply(lambda g: pd.Series({
        "max_oi_strike": g.loc[g["open_interest"].idxmax(), "strike"],
        "max_oi_value": g["open_interest"].max(),
        "total_oi": g["open_interest"].sum(),
    }))
    .reset_index()
)

# Merge
merged = daily_es.merge(date_oi, on="date", how="inner").dropna()
merged["spy_open"] = merged["es_open"] / 10
merged["spy_close"] = merged["es_close"] / 10
merged["spy_high"] = merged["es_high"] / 10
merged["spy_low"] = merged["es_low"] / 10
print(f"Merged days: {len(merged)}")

print()
print("=" * 70)
print(f"  PINNING TEST: 5 VARIATIONS ({len(merged)} days)")
print("=" * 70)

# --- V1: Close closer to max OI strike than open? ---
merged["open_dist"] = (merged["spy_open"] - merged["max_oi_strike"]).abs()
merged["close_dist"] = (merged["spy_close"] - merged["max_oi_strike"]).abs()
merged["moved_toward"] = merged["close_dist"] < merged["open_dist"]
merged["dist_change"] = merged["open_dist"] - merged["close_dist"]

pct = merged["moved_toward"].mean()
t1, p1 = stats.ttest_1samp(merged["dist_change"], 0)

print()
print("V1: Does price close closer to max OI strike than it opened?")
print(f"  Moved TOWARD: {pct*100:.1f}% | AWAY: {(1-pct)*100:.1f}%")
print(f"  Avg distance change: ${merged['dist_change'].mean():.2f} (pos=toward)")
print(f"  T-test: t={t1:.3f} p={p1:.4f} {'SIG' if p1 < 0.05 else 'not sig'}")

# --- V2: Pinning effect by OI level ---
oi_med = merged["total_oi"].median()
hi = merged[merged["total_oi"] > oi_med]
lo = merged[merged["total_oi"] <= oi_med]

print()
print("V2: Does pinning strengthen with more OI?")
print(f"  High OI ({len(hi)} days): toward {hi['moved_toward'].mean()*100:.1f}%, avg ${hi['dist_change'].mean():.2f}")
print(f"  Low OI  ({len(lo)} days): toward {lo['moved_toward'].mean()*100:.1f}%, avg ${lo['dist_change'].mean():.2f}")

t2, p2 = stats.ttest_ind(hi["dist_change"], lo["dist_change"])
print(f"  T-test hi vs lo: t={t2:.3f} p={p2:.4f} {'SIG' if p2 < 0.05 else 'not sig'}")

# --- V3: Daily range compression ---
merged["range_pct"] = (merged["spy_high"] - merged["spy_low"]) / merged["spy_close"] * 100

print()
print("V3: Is daily range compressed when OI is high?")
terciles = pd.qcut(merged["total_oi"], 3, labels=["low", "med", "high"])
for label in ["low", "med", "high"]:
    s = merged[terciles == label]
    print(f"  {label} OI: range={s['range_pct'].mean():.3f}% (median={s['range_pct'].median():.3f}%) n={len(s)}")

t3, p3 = stats.ttest_ind(
    merged[terciles == "low"]["range_pct"],
    merged[terciles == "high"]["range_pct"],
)
print(f"  T-test low vs high: t={t3:.3f} p={p3:.4f} {'SIG' if p3 < 0.05 else 'not sig'}")

# --- V4: Time spent near max OI strike (intraday) ---
print()
print("V4: Time spent near max OI strike (intraday, 200 sample days)")

sample = merged.sample(min(200, len(merged)), random_state=42)
v4_results = []

for _, day in sample.iterrows():
    bars = es_rth[es_rth["date"] == day["date"]]
    if len(bars) == 0:
        continue
    spy_eq = bars["close"].values / 10
    target = day["max_oi_strike"]

    within_1 = np.mean(np.abs(spy_eq - target) < 1.0)
    within_2 = np.mean(np.abs(spy_eq - target) < 2.0)

    v4_results.append({
        "oi": day["total_oi"],
        "w1": within_1,
        "w2": within_2,
    })

v4 = pd.DataFrame(v4_results)
if len(v4) > 0:
    med = v4["oi"].median()
    h = v4[v4["oi"] > med]
    l = v4[v4["oi"] <= med]
    print(f"  High OI: {h['w2'].mean()*100:.1f}% of minutes within $2 of max OI strike")
    print(f"  Low OI:  {l['w2'].mean()*100:.1f}% of minutes within $2 of max OI strike")
    t4, p4 = stats.ttest_ind(h["w2"], l["w2"])
    print(f"  T-test: t={t4:.3f} p={p4:.4f} {'SIG' if p4 < 0.05 else 'not sig'}")

# --- V5: Final hour convergence ---
print()
print("V5: Does price converge toward max OI strike in final hour?")

sample2 = merged.sample(min(200, len(merged)), random_state=99)
v5_results = []

for _, day in sample2.iterrows():
    bars = es_rth[es_rth["date"] == day["date"]]
    if len(bars) < 60:
        continue
    spy_eq = bars["close"].values / 10
    target = day["max_oi_strike"]

    mid = len(spy_eq) // 2
    pm3 = int(len(spy_eq) * 0.85)

    v5_results.append({
        "oi": day["total_oi"],
        "dist_noon": abs(spy_eq[mid] - target),
        "dist_3pm": abs(spy_eq[pm3] - target),
        "dist_close": abs(spy_eq[-1] - target),
        "conv_noon": abs(spy_eq[-1] - target) < abs(spy_eq[mid] - target),
        "conv_3pm": abs(spy_eq[-1] - target) < abs(spy_eq[pm3] - target),
    })

v5 = pd.DataFrame(v5_results)
if len(v5) > 0:
    med = v5["oi"].median()
    h = v5[v5["oi"] > med]
    l = v5[v5["oi"] <= med]

    print(f"  Converged noon->close: High OI {h['conv_noon'].mean()*100:.1f}% | Low OI {l['conv_noon'].mean()*100:.1f}%")
    print(f"  Converged 3pm->close:  High OI {h['conv_3pm'].mean()*100:.1f}% | Low OI {l['conv_3pm'].mean()*100:.1f}%")

    t5, p5 = stats.ttest_ind(h["conv_noon"].astype(float), l["conv_noon"].astype(float))
    print(f"  T-test (noon conv): t={t5:.3f} p={p5:.4f} {'SIG' if p5 < 0.05 else 'not sig'}")

# --- SUMMARY ---
print()
print("=" * 70)
print("  SUMMARY: Which variations show pinning?")
print("=" * 70)
print()
print(f"  V1 Close-vs-open distance:    {'PINNING' if pct > 0.52 else 'NO PINNING':12s} (toward={pct*100:.1f}%, p={p1:.4f})")
print(f"  V2 OI-level effect:           {'PINNING' if p2 < 0.05 and hi['dist_change'].mean() > lo['dist_change'].mean() else 'NO PINNING':12s} (p={p2:.4f})")
print(f"  V3 Range compression:         {'PINNING' if p3 < 0.05 and merged[terciles=='high']['range_pct'].mean() < merged[terciles=='low']['range_pct'].mean() else 'NO PINNING':12s} (p={p3:.4f})")
if len(v4) > 0:
    print(f"  V4 Time near strike:          {'PINNING' if p4 < 0.05 and h['w2'].mean() > l['w2'].mean() else 'NO PINNING':12s} (p={p4:.4f})")
if len(v5) > 0:
    print(f"  V5 Final hour convergence:    {'PINNING' if h['conv_noon'].mean() > 0.55 else 'NO PINNING':12s} (hi={h['conv_noon'].mean()*100:.1f}% lo={l['conv_noon'].mean()*100:.1f}%)")
