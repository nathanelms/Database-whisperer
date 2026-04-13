"""
Generate Figure 2: Discriminator Ladder — sequential feature selection
showing collision reduction at each rung.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── data (from paper: MTSamples, 30 auto-detected concepts, 5-feature ladder) ──
rungs = [
    ("paired_concept",  84.1,  "meaning"),
    ("equation",        62.3,  "meaning"),
    ("negation",        38.7,  "operator"),
    ("verb_class",      18.2,  "expression"),
    ("clause_position",  8.1,  "expression"),
]
# remaining collisions as fraction
remaining = [1.0]
for _, pct, _ in rungs:
    remaining.append(remaining[-1] * (1 - pct/100))

# ── colours ──────────────────────────────────────────────────────────
LAYER_COLORS = {
    "meaning":    "#2980b9",
    "operator":   "#c0392b",
    "expression": "#7f8c8d",
}
C_TEXT      = "#2c3e50"
C_THRESH   = "#e67e22"

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5),
                                         gridspec_kw={"width_ratios": [1, 1.2]})
fig.patch.set_facecolor("white")

# ═══════════════════════════════════════════════════════════════════════
#  LEFT PANEL: The Ladder
# ═══════════════════════════════════════════════════════════════════════
ax = ax_left
y_pos = np.arange(len(rungs))[::-1]
bars = [r[1] for r in rungs]
colors = [LAYER_COLORS[r[2]] for r in rungs]
names = [f"  {i+1}. {r[0]}" for i, r in enumerate(rungs)]

ax.barh(y_pos, bars, height=0.55, color=colors, edgecolor="white", linewidth=1.5, zorder=3)

# 5% threshold line
ax.axvline(x=5, color=C_THRESH, linestyle="--", linewidth=1.5, zorder=2, alpha=0.8)
ax.text(7, -0.8, "5% stop threshold", fontsize=8, color=C_THRESH, fontweight="bold")

# percentage labels on bars
for yp, pct, col in zip(y_pos, bars, colors):
    ax.text(pct + 1.2, yp, f"{pct}%", fontsize=9.5, fontweight="bold",
            va="center", color=col)

ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=10, fontweight="bold", color=C_TEXT)
ax.set_xlabel("Collision reduction (%)", fontsize=10, color=C_TEXT)
ax.set_title("Discriminator Ladder", fontsize=13, fontweight="bold",
             color=C_TEXT, pad=12)
ax.set_xlim(0, 100)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(left=False)

legend_patches = [
    mpatches.Patch(color=LAYER_COLORS["meaning"],    label="Meaning layer"),
    mpatches.Patch(color=LAYER_COLORS["operator"],   label="Operator layer"),
    mpatches.Patch(color=LAYER_COLORS["expression"], label="Expression layer"),
]
ax.legend(handles=legend_patches, loc="lower right", fontsize=8.5,
          framealpha=0.9, edgecolor="#ddd")

# ═══════════════════════════════════════════════════════════════════════
#  RIGHT PANEL: Cumulative collapse curve
# ═══════════════════════════════════════════════════════════════════════
ax2 = ax_right
x_ticks = list(range(len(remaining)))
x_labels = ["None"] + [r[0].replace("_", "\n") for r in rungs]

ax2.fill_between(x_ticks, [r * 100 for r in remaining], alpha=0.12, color="#2980b9")
ax2.plot(x_ticks, [r * 100 for r in remaining], "o-", color="#2980b9",
         linewidth=2.5, markersize=9, zorder=3)

for i, (xt, rem) in enumerate(zip(x_ticks, remaining)):
    pct_str = f"{rem*100:.1f}%"
    ax2.annotate(pct_str, (xt, rem * 100), textcoords="offset points",
                 xytext=(0, 12), ha="center", fontsize=9,
                 fontweight="bold", color="#2980b9")

# final annotation
ax2.text(len(rungs), remaining[-1] * 100 + 8,
         "15,052 raw addr.\n-> 2,701 resolved",
         ha="center", fontsize=8.5, color=C_TEXT, style="italic",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecf0f1", edgecolor="#bdc3c7"))

ax2.set_xticks(x_ticks)
ax2.set_xticklabels(x_labels, fontsize=9, fontweight="bold", color=C_TEXT)
ax2.set_ylabel("Remaining collisions (%)", fontsize=10, color=C_TEXT)
ax2.set_title("Cumulative Collision Reduction", fontsize=13,
              fontweight="bold", color=C_TEXT, pad=12)
ax2.set_ylim(0, 115)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout(w_pad=3)
out = r"C:\Users\User\Dropbox\ExcelDataDump\memory_lab\paper\fig_ladder.pdf"
fig.savefig(out, bbox_inches="tight", dpi=300)
out_png = out.replace(".pdf", ".png")
fig.savefig(out_png, bbox_inches="tight", dpi=200)
print(f"Saved: {out}")
print(f"Saved: {out_png}")
