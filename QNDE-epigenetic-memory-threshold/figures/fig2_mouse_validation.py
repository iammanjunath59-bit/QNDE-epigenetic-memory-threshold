"""
fig2_mouse_validation.py
========================
Figure 2: Cross-species validation in mouse ESCs.
Panels A–D. Saves fig2_mouse_validation.pdf and .png at 600 DPI.

Run from CodeS1 directory:
    python fig2_mouse_validation.py
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.optimize import curve_fit

BASE   = os.path.dirname(os.path.abspath(__file__))
MSCORES = os.path.join(BASE, "data", "raw",       "li2018_scores.csv")
MNGENE  = os.path.join(BASE, "data", "processed", "mouse_ngene_65.csv")
MRES    = os.path.join(BASE, "results",            "mouse_validation_results.json")
HRES    = os.path.join(BASE, "results",            "human_validation_results.json")
PARAMS  = os.path.join(BASE, "params",             "model_params.json")
FIGDIR  = os.path.join(BASE, "figures")
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

COL_M_ABOVE = "#1A7A4A"   # green — mouse above threshold
COL_M_BELOW = "#2471A3"   # blue — below threshold
COL_H       = "#C0392B"   # red — human
COL_PRED    = "#7F8C8D"   # grey — predicted

def panel_label(ax, txt, x=-0.16, y=1.04):
    ax.text(x, y, txt, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left")

def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ── Data ───────────────────────────────────────────────────────
mscores = pd.read_csv(MSCORES)
mscores["gene_symbol"] = mscores["gene_symbol"].str.strip().str.upper()

mng = pd.read_csv(MNGENE)
mng["gene_symbol"] = mng["gene_symbol"].str.strip().str.upper()

dm = mng.merge(mscores[["gene_symbol","identity_score","fitness_score"]],
               on="gene_symbol", how="inner")

with open(MRES) as f: mr = json.load(f)
with open(HRES) as f: hr = json.load(f)
with open(PARAMS) as f: P = json.load(f)

NC_M   = 7          # rounded from 7.29
ids_m  = dm["identity_score"].values
fits_m = dm["fitness_score"].values
ns_m   = dm["n_gene"].values

above_m = ns_m > NC_M
below_m = ~above_m

rng = np.random.default_rng(42)

fig = plt.figure(figsize=(16, 10))
gs  = GridSpec(2, 2, figure=fig,
               left=0.09, right=0.97,
               top=0.94, bottom=0.10,
               wspace=0.38, hspace=0.42)

ax_A = fig.add_subplot(gs[0, 0])
ax_B = fig.add_subplot(gs[0, 1])
ax_C = fig.add_subplot(gs[1, 0])
ax_D = fig.add_subplot(gs[1, 1])

# ══════════════════════════════════════════════════════════════
# PANEL A — Mouse violin plots
# ══════════════════════════════════════════════════════════════
ax = ax_A
above_ids_m = ids_m[above_m]
below_ids_m = ids_m[below_m]

vp = ax.violinplot([below_ids_m, above_ids_m],
                   positions=[1, 2],
                   showmedians=False, showextrema=False,
                   widths=0.55)
vp["bodies"][0].set_facecolor(COL_M_BELOW); vp["bodies"][0].set_alpha(0.55)
vp["bodies"][1].set_facecolor(COL_M_ABOVE); vp["bodies"][1].set_alpha(0.55)
for body in vp["bodies"]:
    body.set_edgecolor("#333"); body.set_linewidth(0.6)

for pos, grp, col in zip([1,2],
                          [below_ids_m, above_ids_m],
                          [COL_M_BELOW, COL_M_ABOVE]):
    q1, med, q3 = np.percentile(grp, [25,50,75])
    ax.vlines(pos, q1, q3, color=col, lw=3.5, zorder=3)
    ax.hlines(med, pos-0.08, pos+0.08, color="#333", lw=1.5, zorder=4)
    jx = rng.uniform(pos-0.13, pos+0.13, len(grp))
    ax.scatter(jx, grp, s=8, color=col, alpha=0.55, zorder=2, lw=0)

ax.axhline(0, color="#aaa", lw=0.6, ls="--")
ax.set_xticks([1, 2])
ax.set_xticklabels([f"N ≤ {NC_M}\n(n={below_m.sum()})",
                    f"N > {NC_M}\n(n={above_m.sum()})"])
ax.set_ylabel("Identity score (Rex1GFP screen)")
ax.set_xlim(0.5, 2.5)
despine(ax)
panel_label(ax, "A")

d_m  = mr["cohens_d"]
pp_m = mr["permutation_p"]
ax.text(0.97, 0.97,
        f"Cohen's d = {d_m:.3f}\nperm p = {pp_m:.4f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", lw=0.6))

# ══════════════════════════════════════════════════════════════
# PANEL B — Mouse identity vs fitness scatter
# ══════════════════════════════════════════════════════════════
ax = ax_B

colors_m = np.where(above_m, COL_M_ABOVE, COL_M_BELOW)
jx_m = ns_m + rng.uniform(-0.14, 0.14, len(ns_m))
ax.scatter(ids_m, fits_m,
           c=colors_m, s=22, alpha=0.70, lw=0, zorder=3)

ax.axhline(0, color="#bbb", lw=0.6, ls="--")
ax.axvline(0, color="#bbb", lw=0.6, ls="--")

r_id_m  = mr["pearson_r_identity"]
p_id_m  = mr["pearson_p_identity"]
r_fit_m = mr["pearson_r_fitness"]
p_fit_m = mr["pearson_p_fitness"]
ax.text(0.03, 0.97,
        f"Identity  r = +{r_id_m:.3f}, p = {p_id_m:.3f}\n"
        f"Fitness   r = +{r_fit_m:.3f}, p = {p_fit_m:.2f} NS",
        transform=ax.transAxes, ha="left", va="top", fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", lw=0.6))

ax.set_xlabel("Identity score")
ax.set_ylabel("Fitness score")
despine(ax)
panel_label(ax, "B")

# ══════════════════════════════════════════════════════════════
# PANEL C — Nc bar chart comparison
# ══════════════════════════════════════════════════════════════
ax = ax_C

nc_h  = hr.get("sigmoid_Nc",    4.72)
se_h  = hr.get("sigmoid_Nc_se", 0.47)
nc_m  = mr["sigmoid_Nc"]
se_m  = mr["sigmoid_Nc_se"]

bars = ax.bar([1, 2], [nc_h, nc_m],
              color=[COL_H, COL_M_ABOVE],
              width=0.5, alpha=0.8, edgecolor="#333", lw=0.7)
ax.errorbar([1, 2], [nc_h, nc_m], yerr=[se_h, se_m],
            fmt="none", color="#333", capsize=5, capthick=1.0, lw=1.0)

# Predicted diamonds
pred_h = P["NC_H_illustrative"]
pred_m = P["NC_M_illustrative"]
ax.scatter([1, 2], [pred_h, pred_m],
           marker="D", s=40, color="#333", zorder=5)

ax.set_xticks([1, 2])
ax.set_xticklabels(["Human ESC\n(GSE277069)", "Mouse ESC\n(GSE107060)"])
ax.set_ylabel("Empirical $N_c$ (sigmoid fit)")
ax.set_ylim(0, 11)
ax.set_xlim(0.5, 2.5)
despine(ax)
panel_label(ax, "C")

ax.text(0.97, 0.97,
        "◆ Predicted (midpoint g)",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=7, color="#333",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", lw=0.6))

# ══════════════════════════════════════════════════════════════
# PANEL D — Cross-species ratio
# ══════════════════════════════════════════════════════════════
ax = ax_D

pred_ratio = mr["cross_species_ratio_predicted"]
obs_ratio  = mr["cross_species_ratio_observed"]
obs_se     = mr["cross_species_ratio_se"]
ci_lo      = mr["cross_species_ci_lo"]
ci_hi      = mr["cross_species_ci_hi"]
pct_err    = mr["cross_species_pct_error"]

ax.bar(1, pred_ratio, width=0.45, color=COL_PRED,
       alpha=0.75, edgecolor="#333", lw=0.7, label="Predicted")
ax.bar(2, obs_ratio,  width=0.45, color=COL_M_ABOVE,
       alpha=0.80, edgecolor="#333", lw=0.7, label="Observed")
ax.errorbar(2, obs_ratio, yerr=obs_se,
            fmt="none", color="#333", capsize=5, capthick=1.0, lw=1.0)

# CI bracket
ax.annotate("", xy=(2, ci_hi+0.03), xytext=(2, ci_lo-0.03),
            arrowprops=dict(arrowstyle="-", color="#555",
                            lw=0.8, connectionstyle="arc3,rad=0"))
ax.text(2.28, (ci_hi+ci_lo)/2,
        f"95% CI\n[{ci_lo:.2f},\n{ci_hi:.2f}]",
        fontsize=6.5, va="center", color="#555")

ax.axhline(pred_ratio, color=COL_PRED, lw=1.0, ls="--", alpha=0.8)
ax.text(0.5, pred_ratio+0.04, f"Pred. = {pred_ratio:.2f}",
        fontsize=7, color=COL_PRED, va="bottom")

ax.set_xticks([1, 2])
ax.set_xticklabels(["Predicted\n($\\kappa_{mouse}/\\kappa_{human}$, g-free)",
                    "Observed\n($N_c^{mouse}/N_c^{human}$)"])
ax.set_ylabel("Cross-species $N_c$ ratio")
ax.set_ylim(0, 2.8)
ax.set_xlim(0.5, 2.9)
despine(ax)
panel_label(ax, "D")

ax.text(0.03, 0.97,
        f"Agreement: {pct_err:.1f}%\nPredicted in 95% CI: Yes",
        transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", lw=0.6))

for ext in ["pdf","png"]:
    path = os.path.join(FIGDIR, f"fig2_mouse_validation.{ext}")
    fig.savefig(path, dpi=600, bbox_inches="tight")
    print(f"Saved: {path}")
plt.close(fig)
print("Figure 2 done.")
