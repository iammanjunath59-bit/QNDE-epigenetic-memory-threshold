"""
fig3_information_theory.py
===========================
Figure 3: Information-theoretic interpretation.
Panels A–B. Saves fig3_information_theory.pdf and .png at 600 DPI.

Run from CodeS1 directory:
    python fig3_information_theory.py
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

BASE   = os.path.dirname(os.path.abspath(__file__))
PARAMS = os.path.join(BASE, "params", "model_params.json")
FIGDIR = os.path.join(BASE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8,
    "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7,
    "ytick.major.width": 0.7, "xtick.major.size": 3,
    "ytick.major.size": 3, "figure.dpi": 150,
    "savefig.dpi": 600, "pdf.fonttype": 42,
})

with open(PARAMS) as f: P = json.load(f)

NC_H = P.get("NC_H_illustrative", 4.00)
NC_M = P.get("NC_M_illustrative", 7.29)

COL_H = "#C0392B"
COL_M = "#1A7A4A"

def pe(N, Nc):
    return 1.0 / (1.0 + np.exp((N - Nc) / Nc))

def Hb(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1-p) * np.log2(1-p))

def C(N, Nc):
    return 1.0 - Hb(pe(N, Nc))

def panel_label(ax, txt, x=-0.13, y=1.04):
    ax.text(x, y, txt, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left")

def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

Nvals = np.linspace(0.1, 14, 500)

fig = plt.figure(figsize=(14, 6))
gs  = GridSpec(1, 2, figure=fig,
               left=0.09, right=0.97,
               top=0.92, bottom=0.14,
               wspace=0.38)
ax_A = fig.add_subplot(gs[0])
ax_B = fig.add_subplot(gs[1])

# ══════════════════════════════════════════════════════════════
# PANEL A — Channel capacity C(N)
# ══════════════════════════════════════════════════════════════
ax = ax_A

C_H = C(Nvals, NC_H)
C_M = C(Nvals, NC_M)

# Shade zero-capacity region
ax.axhspan(-0.02, 0.005, color="#E8E8E8", zorder=0, lw=0)

ax.plot(Nvals, C_H, color=COL_H, lw=2.0, zorder=4)
ax.plot(Nvals, C_M, color=COL_M, lw=2.0, ls="--", zorder=4)

ax.axvline(NC_H, color=COL_H, lw=0.9, ls=":", alpha=0.7)
ax.axvline(NC_M, color=COL_M, lw=0.9, ls=":", alpha=0.7)
ax.axhline(0,    color="#aaa",  lw=0.6, ls="-")

# Gene annotations
genes = [
    ("POU5F1", 10, "human",  0.315, COL_H, "o", 1.8, -0.03, "right"),
    ("POU5F1", 10, "mouse",  0.024, COL_M, "D",  0.4, -0.03, "right"),
    ("NANOG",  11, "human",  0.395, COL_H, "o", 2.1,  0.02, "right"),
]
for gene, N_g, sp, C_val, col, mk, dx, dy, ha in genes:
    ax.scatter([N_g], [C_val], marker=mk, s=40, color=col,
               zorder=5, lw=0.5, edgecolors="#333")
    lbl = f"{gene}\n({sp[:1].upper()}): {C_val:.3f} bits"
    ax.annotate(lbl,
                xy=(N_g, C_val), xytext=(N_g+dx, C_val+dy),
                fontsize=5.8, color=col, ha=ha, va="center",
                arrowprops=dict(arrowstyle="-", color=col,
                                lw=0.5, shrinkA=4, shrinkB=2))

ax.set_xlabel("Circuit size N")
ax.set_ylabel("Channel capacity C(N) [bits/division]")
ax.set_xlim(0, 14)
ax.set_ylim(-0.02, 1.05)
ax.set_xticks(range(0, 15, 2))

# Axis labels for threshold lines
ax.text(NC_H-0.25, 0.82, f"$N_c^{{H}}$={NC_H:.0f}",
        ha="right", va="center", fontsize=7, color=COL_H)
ax.text(NC_M+0.25, 0.82, f"$N_c^{{M}}$={NC_M:.1f}",
        ha="left",  va="center", fontsize=7, color=COL_M)

# Text labels for curves (no legend box)
ax.text(12.5, C(12.5, NC_H)+0.03, "Human ESC", color=COL_H,
        fontsize=7.5, ha="right")
ax.text(12.5, C(12.5, NC_M)-0.05, "Mouse ESC", color=COL_M,
        fontsize=7.5, ha="right")
ax.text(0.5, 0.015,
        "C = 0 region (N ≤ $N_c$)",
        fontsize=6.5, color="#888", va="bottom")

despine(ax)
panel_label(ax, "A")

# ══════════════════════════════════════════════════════════════
# PANEL B — Memory error probability pe(N)
# ══════════════════════════════════════════════════════════════
ax = ax_B

pe_H = pe(Nvals, NC_H)
pe_M = pe(Nvals, NC_M)

# Shade regions
ax.fill_between(Nvals, 0.5, 1.0, where=Nvals <= NC_H,
                color="#D6EAF8", alpha=0.5, zorder=0)
ax.fill_between(Nvals, 0.0, 0.5, where=Nvals > NC_H,
                color="#FADBD8", alpha=0.4, zorder=0)

ax.plot(Nvals, pe_H, color=COL_H, lw=2.0, zorder=4)
ax.plot(Nvals, pe_M, color=COL_M, lw=2.0, ls="--", zorder=4)

ax.axhline(0.5, color="#aaa", lw=0.8, ls="--")
ax.axvline(NC_H, color=COL_H, lw=0.9, ls=":", alpha=0.7)
ax.axvline(NC_M, color=COL_M, lw=0.9, ls=":", alpha=0.7)

# Mark exact values at Nc
ax.scatter([NC_H, NC_M], [0.5, 0.5], s=35, color=[COL_H, COL_M],
           zorder=5, lw=0.5, edgecolors="#333")

ax.set_xlabel("Circuit size N")
ax.set_ylabel("Memory error probability $p_e$(N)")
ax.set_xlim(0, 14)
ax.set_ylim(-0.02, 1.05)
ax.set_xticks(range(0, 15, 2))

ax.text(NC_H-0.25, 0.90, f"$N_c^{{H}}$={NC_H:.0f}",
        ha="right", va="center", fontsize=7, color=COL_H)
ax.text(NC_M+0.25, 0.90, f"$N_c^{{M}}$={NC_M:.1f}",
        ha="left",  va="center", fontsize=7, color=COL_M)

# Region labels
ax.text(1.0, 0.95, "Stochastic switching\n$p_e > 0.5$",
        fontsize=6.5, color="#2471A3", va="top")
ax.text(10.5, 0.08, "Stable memory\n$p_e < 0.5$",
        fontsize=6.5, color="#C0392B", va="bottom", ha="center")
ax.text(0.5, 0.505, "$p_e = 0.5$", fontsize=6.5, color="#888", va="bottom")

# Curve labels
ax.text(2.5, pe(2.5, NC_H)+0.04, "Human ESC", color=COL_H,
        fontsize=7.5, ha="center")
ax.text(5.0, pe(5.0, NC_M)+0.04, "Mouse ESC", color=COL_M,
        fontsize=7.5, ha="center")

despine(ax)
panel_label(ax, "B")

for ext in ["pdf","png"]:
    path = os.path.join(FIGDIR, f"fig3_information_theory.{ext}")
    fig.savefig(path, dpi=600, bbox_inches="tight")
    print(f"Saved: {path}")
plt.close(fig)
print("Figure 3 done.")
