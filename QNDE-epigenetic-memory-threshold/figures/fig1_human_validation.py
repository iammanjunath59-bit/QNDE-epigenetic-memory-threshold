"""
fig1_human_validation.py
========================
Figure 1: Derivation and human ESC validation.
Panels A–F. Saves fig1_human_validation.pdf and .png at 600 DPI.

Run from CodeS1 directory:
    python fig1_human_validation.py

Requires:
    data/raw/rosen2024_scores.csv
    data/processed/human_ngene_112.csv
    results/human_validation_results.json
    params/model_params.json
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.optimize import curve_fit

# ── Paths ──────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
SCORES  = os.path.join(BASE, "data", "raw",       "rosen2024_scores.csv")
NGENE   = os.path.join(BASE, "data", "processed", "human_ngene_112.csv")
RES     = os.path.join(BASE, "results",            "human_validation_results.json")
PARAMS  = os.path.join(BASE, "params",             "model_params.json")
FIGDIR  = os.path.join(BASE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "Arial",
    "font.size":         8,
    "axes.labelsize":    9,
    "axes.titlesize":    9,
    "xtick.labelsize":   7.5,
    "ytick.labelsize":   7.5,
    "axes.linewidth":    0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "figure.dpi":        150,
    "savefig.dpi":       600,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

COL_ABOVE = "#C0392B"   # red — above threshold
COL_BELOW = "#2471A3"   # blue — below threshold
COL_OUT   = "#E67E22"   # orange — outliers
COL_REG   = "#555555"   # regression line

# ── Data ───────────────────────────────────────────────────────
scores = pd.read_csv(SCORES)
scores["gene_symbol"] = scores["gene_symbol"].str.strip().str.upper()
scores = scores.rename(columns={
    "NE_pluripotency_score": "identity_score",
    "E8_self_renewal_score":  "fitness_score",
})

ng = pd.read_csv(NGENE)
ng["gene_symbol"] = ng["gene_symbol"].str.strip().str.upper()

df = ng.merge(scores[["gene_symbol","identity_score","fitness_score"]],
              on="gene_symbol", how="inner")

OUTLIERS = {"METTL3","METTL14","CARM1"}
df_out = df[df["gene_symbol"].isin(OUTLIERS)].copy()
df_main = df[~df["gene_symbol"].isin(OUTLIERS)].copy()

with open(RES)  as f: res    = json.load(f)
with open(PARAMS) as f: P    = json.load(f)

NC      = 4
NC_PRED = 4.00
NC_FIT  = res.get("sigmoid_Nc",    4.72)
NC_SE   = res.get("sigmoid_Nc_se", 0.47)

ids  = df_main["identity_score"].values
fits = df_main["fitness_score"].values
ns   = df_main["n_gene"].values

above_mask = ns > NC
below_mask = ~above_mask

rng  = np.random.default_rng(42)

# ── Helpers ────────────────────────────────────────────────────
def panel_label(ax, txt, x=-0.16, y=1.04):
    ax.text(x, y, txt, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left")

def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ── Figure layout ──────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
gs  = GridSpec(2, 3, figure=fig,
               left=0.07, right=0.97,
               top=0.94,  bottom=0.08,
               wspace=0.38, hspace=0.42)

ax_A = fig.add_subplot(gs[0, 0])
ax_B = fig.add_subplot(gs[0, 1])
ax_C = fig.add_subplot(gs[0, 2])
ax_D = fig.add_subplot(gs[1, 0])
ax_E = fig.add_subplot(gs[1, 1])
ax_F = fig.add_subplot(gs[1, 2])

# ══════════════════════════════════════════════════════════════
# PANEL A — Schematic
# ══════════════════════════════════════════════════════════════
ax = ax_A
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.axis("off")
panel_label(ax, "A", x=-0.04)

# Nucleosome oval
nuc = mpatches.Ellipse((5, 5), 2.8, 1.8, color="#F0F0F0",
                        ec="#888888", lw=1.2, zorder=2)
ax.add_patch(nuc)
ax.text(5, 5, "Nucleosome", ha="center", va="center",
        fontsize=7.5, fontstyle="italic", color="#555555", zorder=3)

# TF boxes — above threshold (red)
tf_pos_above = [(1.5, 8.5),(5, 9),(8.5, 8.5)]
tf_above_labels = ["TF₅","TF₆","TF₇"]
for (x, y), lbl in zip(tf_pos_above, tf_above_labels):
    box = mpatches.FancyBboxPatch((x-0.55, y-0.4), 1.1, 0.8,
        boxstyle="round,pad=0.05", color=COL_ABOVE,
        ec="#8B0000", lw=0.8, alpha=0.85, zorder=2)
    ax.add_patch(box)
    ax.text(x, y, lbl, ha="center", va="center",
            fontsize=7, color="white", fontweight="bold", zorder=3)
    ax.annotate("", xy=(5+(x-5)*0.43, 5+(y-5)*0.42),
                xytext=(x, y-0.4 if y > 5 else y+0.4),
                arrowprops=dict(arrowstyle="-", color="#C0392B",
                                lw=0.9, connectionstyle="arc3,rad=0.0"),
                zorder=1)

# TF boxes — below threshold (blue)
tf_pos_below = [(1.5, 1.5),(5, 1),(8.5, 1.5)]
tf_below_labels = ["TF₁","TF₂","TF₃"]
for (x, y), lbl in zip(tf_pos_below, tf_below_labels):
    box = mpatches.FancyBboxPatch((x-0.55, y-0.4), 1.1, 0.8,
        boxstyle="round,pad=0.05", color=COL_BELOW,
        ec="#1A5276", lw=0.8, alpha=0.85, zorder=2)
    ax.add_patch(box)
    ax.text(x, y, lbl, ha="center", va="center",
            fontsize=7, color="white", fontweight="bold", zorder=3)
    ax.annotate("", xy=(5+(x-5)*0.43, 5+(y-5)*0.42),
                xytext=(x, y+0.4 if y < 5 else y-0.4),
                arrowprops=dict(arrowstyle="-", color=COL_BELOW,
                                lw=0.9, connectionstyle="arc3,rad=0.0"),
                zorder=1)

# Coupling label
ax.annotate("", xy=(4.0, 6.8), xytext=(4.0, 8.1),
            arrowprops=dict(arrowstyle="<->", color="#555", lw=1.0))
ax.text(3.5, 7.45, "$J_{eff}$", ha="right", va="center",
        fontsize=9, color="#333")

# Equation boxes
ax.text(5, 3.65, "$J_{eff} = g^2/\\kappa$",
        ha="center", va="center", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="#EBF5FB", ec="#85C1E9", lw=0.8))
ax.text(5, 2.35, "$N_c = \\omega\\kappa/(2g^2)$",
        ha="center", va="center", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="#FDEBD0", ec="#F0B27A", lw=0.8))

# Labels
ax.text(0.5, 9.5, "N > $N_c$ (stable)", color=COL_ABOVE,
        fontsize=7.5, fontweight="bold")
ax.text(0.5, 0.3, "N ≤ $N_c$ (unstable)", color=COL_BELOW,
        fontsize=7.5, fontweight="bold")

# ══════════════════════════════════════════════════════════════
# PANEL B — Violin plots
# ══════════════════════════════════════════════════════════════
ax = ax_B
above_ids = ids[above_mask]
below_ids = ids[below_mask]

vp = ax.violinplot([below_ids, above_ids],
                   positions=[1, 2],
                   showmedians=False, showextrema=False,
                   widths=0.55)
vp["bodies"][0].set_facecolor(COL_BELOW); vp["bodies"][0].set_alpha(0.55)
vp["bodies"][1].set_facecolor(COL_ABOVE); vp["bodies"][1].set_alpha(0.55)
for body in vp["bodies"]:
    body.set_edgecolor("#333333"); body.set_linewidth(0.6)

for pos, grp, col in zip([1,2],[below_ids, above_ids],[COL_BELOW, COL_ABOVE]):
    q1, med, q3 = np.percentile(grp, [25,50,75])
    ax.vlines(pos, q1, q3, color=col, lw=3.5, zorder=3)
    ax.hlines(med, pos-0.08, pos+0.08, color="#333", lw=1.5, zorder=4)
    jx = rng.uniform(pos-0.13, pos+0.13, len(grp))
    ax.scatter(jx, grp, s=8, color=col, alpha=0.55, zorder=2, lw=0)

ax.axhline(0, color="#aaa", lw=0.6, ls="--")
ax.set_xticks([1, 2])
ax.set_xticklabels([f"N ≤ {NC}\n(n={below_mask.sum()})",
                    f"N > {NC}\n(n={above_mask.sum()})"])
ax.set_ylabel("Identity score (NE_pluripotency)")
ax.set_xlim(0.5, 2.5)
despine(ax)
panel_label(ax, "B")

d_val = res["cohens_d"]
pp    = res["permutation_p"]
ax.text(0.97, 0.97,
        f"Cohen's d = {d_val:.3f}\nperm p = {pp:.4f}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", lw=0.6))

# ══════════════════════════════════════════════════════════════
# PANEL C — Identity vs fitness scatter
# ══════════════════════════════════════════════════════════════
ax = ax_C
cmap = plt.cm.RdBu_r
norm = plt.Normalize(0, 11)

sc = ax.scatter(df_main["identity_score"], df_main["fitness_score"],
                c=df_main["n_gene"], cmap=cmap, norm=norm,
                s=22, alpha=0.75, lw=0, zorder=3)
ax.scatter(df_out["identity_score"], df_out["fitness_score"],
           marker="^", c=COL_OUT, s=30, alpha=0.8, lw=0.5,
           edgecolors="#8B4513", zorder=4)

ax.axhline(0, color="#bbb", lw=0.6, ls="--")
ax.axvline(0, color="#bbb", lw=0.6, ls="--")

# Annotate key genes
for _, row in df_main[df_main["gene_symbol"].isin(
        ["FASN","MRPL11","IDH1"])].iterrows():
    ax.annotate(row["gene_symbol"],
                (row["identity_score"], row["fitness_score"]),
                xytext=(4, 4), textcoords="offset points",
                fontsize=5.5, color="#333")

r_id  = res["pearson_r_identity"]
p_id  = res["pearson_p_identity"]
r_fit = res["pearson_r_fitness"]
p_fit = res["pearson_p_fitness"]
ax.text(0.03, 0.97,
        f"Identity  r = +{r_id:.3f}, p = {p_id:.1e}\n"
        f"Fitness   r = +{r_fit:.3f}, p = {p_fit:.2f} NS",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", lw=0.6))

ax.set_xlabel("Identity score")
ax.set_ylabel("Fitness score (E8_self-renewal)")
despine(ax)
panel_label(ax, "C")

cb = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
cb.set_label("$N_{gene}$", fontsize=8)
cb.ax.tick_params(labelsize=6.5)

# ══════════════════════════════════════════════════════════════
# PANEL D — Permutation distribution
# ══════════════════════════════════════════════════════════════
ax = ax_D

rng2 = np.random.default_rng(99)
def cohens_d(a, b):
    sp = np.sqrt((np.var(a, ddof=1)+np.var(b, ddof=1))/2)
    return (np.mean(a)-np.mean(b))/sp if sp>0 else 0.0

pds = []
for _ in range(10000):
    perm = rng2.permutation(ids)
    pds.append(cohens_d(perm[above_mask], perm[below_mask]))
pds = np.array(pds)

obs_d = res["cohens_d"]
ax.hist(pds, bins=60, color="#AABCCE", edgecolor="none", density=True)
ax.axvline(obs_d, color=COL_ABOVE, lw=1.8, zorder=4)

tail_x = np.linspace(obs_d, pds.max()+0.1, 200)
from scipy.stats import gaussian_kde
kde = gaussian_kde(pds)
ax.fill_between(tail_x, 0, kde(tail_x),
                color=COL_ABOVE, alpha=0.35, zorder=3)

ax.set_xlabel("Cohen's d (permuted)")
ax.set_ylabel("Density")
perm_p = res["permutation_p"]
perm_cnt = res["permutation_count"]
ax.text(0.97, 0.97,
        f"Observed d = {obs_d:.3f}\np = {perm_p:.4f}\n({perm_cnt}/10,000)",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", lw=0.6))
despine(ax)
panel_label(ax, "D")

# ══════════════════════════════════════════════════════════════
# PANEL E — Sigmoid fit
# ══════════════════════════════════════════════════════════════
ax = ax_E

id_min = ids.min(); id_max = ids.max()
ids_norm = (ids - id_min) / (id_max - id_min)

bins = []
for n_val in sorted(set(ns)):
    idx = ns == n_val
    if idx.sum() >= 2:
        v = ids_norm[idx]
        bins.append([n_val, v.mean(), v.std(ddof=1)/np.sqrt(idx.sum())])
bins = np.array(bins)

def sig4(N, Nc, w, sc, off):
    return off + sc / (1 + np.exp(-(N - Nc) / w))

try:
    popt, pcov = curve_fit(sig4, bins[:,0], bins[:,1],
        p0=[4.0, 2.0, 0.5, 0.1],
        bounds=([0.1,0.2,0.05,0],[15,8,2,0.95]), maxfev=10000)
    Nc_f, w_f, sc_f, off_f = popt
    Nc_se = np.sqrt(pcov[0,0])
    Nx = np.linspace(0, 12, 300)
    ax.plot(Nx, sig4(Nx, *popt), color=COL_ABOVE, lw=1.5, zorder=3)
    y_lo = sig4(Nx, Nc_f-1.96*Nc_se, *popt[1:])
    y_hi = sig4(Nx, Nc_f+1.96*Nc_se, *popt[1:])
    ax.fill_between(Nx, y_lo, y_hi, color=COL_ABOVE, alpha=0.18, zorder=2)
except Exception:
    Nc_f = NC_FIT; Nc_se = NC_SE

ax.errorbar(bins[:,0], bins[:,1], yerr=bins[:,2],
            fmt="o", color="#333", ms=4.5, lw=0.8,
            capsize=2.5, capthick=0.8, zorder=4)
ax.axvline(NC_PRED, color="#888", lw=0.9, ls="--", zorder=1)
ax.axvline(NC_FIT,  color=COL_ABOVE, lw=0.9, ls=":", zorder=1)

ax.set_xlabel("$N_{gene}$")
ax.set_ylabel("Normalised identity score")
ax.set_xlim(-0.5, 12.5)
ax.set_ylim(-0.05, 1.05)
ax.set_xticks(range(0,13,2))
despine(ax)
panel_label(ax, "E")

ax.text(0.97, 0.05,
        f"$N_c$(data) = {NC_FIT:.2f} ± {NC_SE:.2f}",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7.5, color=COL_ABOVE,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", lw=0.6))
ax.text(NC_PRED+0.15, 0.88, f"Pred. $N_c$={NC_PRED:.0f}",
        fontsize=6.5, color="#666", va="top")

# ══════════════════════════════════════════════════════════════
# PANEL F — Ngene vs identity scatter
# ══════════════════════════════════════════════════════════════
ax = ax_F

jx_main = df_main["n_gene"] + rng.uniform(-0.14, 0.14, len(df_main))
colors_f = np.where(df_main["n_gene"].values > NC, COL_ABOVE, COL_BELOW)
ax.scatter(jx_main, df_main["identity_score"],
           c=colors_f, s=18, alpha=0.65, lw=0, zorder=3)
ax.scatter(df_out["n_gene"] + rng.uniform(-0.14,0.14,len(df_out)),
           df_out["identity_score"],
           marker="^", c=COL_OUT, s=28, alpha=0.8, lw=0.4,
           edgecolors="#8B4513", zorder=4)

# Regression line
m, b, *_ = stats.linregress(ns, ids)
xr = np.array([ns.min()-0.3, ns.max()+0.3])
ax.plot(xr, m*xr+b, color=COL_REG, lw=1.0, ls="--", zorder=2)

ax.axvline(NC, color="#bbb", lw=0.7, ls=":", zorder=1)
ax.axhline(0, color="#bbb", lw=0.6, ls="--", zorder=1)

# Annotate MYC
myc = df_main[df_main["gene_symbol"]=="MYC"]
if len(myc):
    ax.annotate("MYC",
                (myc.iloc[0]["n_gene"]+0.1, myc.iloc[0]["identity_score"]),
                xytext=(6,0), textcoords="offset points",
                fontsize=5.5, color="#333")

ax.set_xlabel("$N_{gene}$")
ax.set_ylabel("Identity score")
ax.set_xticks(range(0,13,2))
ax.set_xlim(-0.5, 12.5)
despine(ax)
panel_label(ax, "F")

r_id  = res["pearson_r_identity"]
p_id  = res["pearson_p_identity"]
r_fit = res["pearson_r_fitness"]
ax.text(0.03, 0.97,
        f"Identity  r = +{r_id:.3f}\n"
        f"p = {p_id:.1e}\n"
        f"Fitness  r = +{r_fit:.3f} NS",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", lw=0.6))

# ── Save ───────────────────────────────────────────────────────
for ext in ["pdf", "png"]:
    path = os.path.join(FIGDIR, f"fig1_human_validation.{ext}")
    fig.savefig(path, dpi=600, bbox_inches="tight")
    print(f"Saved: {path}")
plt.close(fig)
print("Figure 1 done.")
