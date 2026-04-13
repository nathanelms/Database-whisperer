"""
Generate Figure 1: Semantic Routing Tree for "positive" in clinical text.
Shows how the discriminator ladder progressively routes instances to resolved meanings.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── colour palette ──────────────────────────────────────────────────
C_ROOT   = "#2c3e50"   # dark slate
C_MEAN   = "#2980b9"   # meaning feature (blue)
C_OP     = "#c0392b"   # operator feature (red)
C_EXPR   = "#7f8c8d"   # expression feature (grey)
C_LEAF   = "#27ae60"   # resolved meaning (green)
C_EDGE   = "#95a5a6"
C_NEIGH  = "#8e44ad"   # neighborhood text (purple)

fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor("white")
ax.set_xlim(-0.5, 13.5)
ax.set_ylim(-0.5, 7.5)
ax.axis("off")

# ── helper: draw a box with text ────────────────────────────────────
def box(x, y, text, color, fontsize=8.5, width=1.8, height=0.48, alpha=1.0,
        fontweight="normal", fontcolor="white", style="round,pad=0.1"):
    b = FancyBboxPatch((x - width/2, y - height/2), width, height,
                        boxstyle=style, facecolor=color, edgecolor="none",
                        alpha=alpha, zorder=3)
    ax.add_patch(b)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight=fontweight, color=fontcolor, zorder=4)
    return (x, y)

def edge(x1, y1, x2, y2, label="", color=C_EDGE, lw=1.2):
    ax.annotate("", xy=(x2, y2 + 0.26), xytext=(x1, y1 - 0.26),
                arrowprops=dict(arrowstyle="-", color=color, lw=lw))

def leaf_box(x, y, meaning, n, neighbors):
    """Green leaf with meaning label + neighbor list below."""
    box(x, y, f"{meaning}  (n={n})", C_LEAF, fontsize=8, width=2.4,
        height=0.44, fontweight="bold")
    ax.text(x, y - 0.40, neighbors, ha="center", va="top",
            fontsize=6.5, color=C_NEIGH, style="italic", zorder=4)

# ── Y levels ────────────────────────────────────────────────────────
Y0 = 6.8   # root
Y1 = 5.1   # paired_concept split
Y2 = 3.4   # negation split
Y3 = 1.7   # leaves

# ══════════════════════════════════════════════════════════════════════
#  ROOT
# ══════════════════════════════════════════════════════════════════════
box(6.5, Y0, '"positive"   (N = 1,847 instances)', C_ROOT,
    fontsize=11, width=4.2, height=0.55, fontweight="bold")

# ── Level 1: paired_concept (MEANING feature) ──────────────────────
ax.text(6.5, Y0 - 0.72, "paired_concept  (meaning layer,  eliminates 84% of collisions)",
        ha="center", va="center", fontsize=9, fontweight="bold", color=C_MEAN)

# branches
L1_x = [1.5, 4.5, 7.5, 11.0]
L1_labels = ["culture / test /\nscreen / specimen", "history / family",
             "outlook / response /\nattitude", "[other]"]

for xp, lab in zip(L1_x, L1_labels):
    box(xp, Y1, lab, C_MEAN, fontsize=7.5, width=2.3, height=0.55)
    edge(6.5, Y0, xp, Y1)

# ── Level 2: negation on [other] branch only ────────────────────────
ax.text(11.0, Y1 - 0.65, "negation  (operator layer)",
        ha="center", va="center", fontsize=8, fontweight="bold", color=C_OP)

box(9.5, Y2, "negated = yes", C_OP, fontsize=8, width=1.8, height=0.44)
box(12.5, Y2, "negated = no", "#5d6d7e", fontsize=8, width=1.8, height=0.44)
edge(11.0, Y1, 9.5, Y2)
edge(11.0, Y1, 12.5, Y2)

# ── Leaves ──────────────────────────────────────────────────────────
leaf_box(1.5, Y3, "lab-positive", 412,
         "neighbors: culture, blood, urine,\nspecimen, screen, test")

leaf_box(4.5, Y3, "family-hx positive", 298,
         "neighbors: family, mother, father,\ndiabetes, hypertension, cancer")

leaf_box(7.5, Y3, "sentiment-positive", 187,
         "neighbors: outlook, response,\nattitude, prognosis, recovery")

leaf_box(9.5, Y3, "not positive", 41,
         "neighbors: denied, negative,\nruled-out, absent")

leaf_box(12.5, Y3, "generic positive", 909,
         "neighbors: [diffuse]\nhazard = HIGH")

# edges from L1 to leaves (first 3 go straight down)
edge(1.5, Y1, 1.5, Y3)
edge(4.5, Y1, 4.5, Y3)
edge(7.5, Y1, 7.5, Y3)
# last two come from L2
edge(9.5, Y2, 9.5, Y3)
edge(12.5, Y2, 12.5, Y3)

# ── Legend ──────────────────────────────────────────────────────────
legend_y = 0.2
legend_items = [
    (C_MEAN,  "Meaning feature"),
    (C_OP,    "Operator (never collapse)"),
    (C_LEAF,  "Resolved meaning"),
    (C_NEIGH, "Substitution neighborhood"),
]
for i, (c, lab) in enumerate(legend_items):
    xp = 1.5 + i * 3.2
    ax.add_patch(FancyBboxPatch((xp - 0.18, legend_y - 0.14), 0.36, 0.28,
                 boxstyle="round,pad=0.05", facecolor=c, edgecolor="none", alpha=0.85))
    ax.text(xp + 0.35, legend_y, lab, fontsize=8, va="center", color="#333333")

plt.tight_layout()
out = r"C:\Users\User\Dropbox\ExcelDataDump\memory_lab\paper\fig_routing_tree.pdf"
fig.savefig(out, bbox_inches="tight", dpi=300)
out_png = out.replace(".pdf", ".png")
fig.savefig(out_png, bbox_inches="tight", dpi=200)
print(f"Saved: {out}")
print(f"Saved: {out_png}")
