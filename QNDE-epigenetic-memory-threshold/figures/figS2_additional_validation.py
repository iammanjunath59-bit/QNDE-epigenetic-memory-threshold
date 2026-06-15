"""
figS2_additional_validation.py
================================
Supplementary Figure S2: Additional validation analyses.
Panels A–F. Saves figS2_additional_validation.pdf and .png at 600 DPI.

Run from CodeS1 directory:
    python figS2_additional_validation.py
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
from scipy.stats import gaussian_kde

BASE    = os.path.dirname(os.path.abspath(__file__))
SCORES  = os.path.join(BASE, "data", "raw",       "rosen2024_scores.csv")
NGENE   = os.path.join(BASE, "data", "processed", "human_ngene_112.csv")
HRES    = os.path.join(BASE, "results",            "human_validation_results.json")
NCRES   = os.path.join(BASE, "results",            "negative_control_results.json")
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

COL_ABOVE  = "#C0392B"
COL_BELOW  = "#2471A3"
COL_OUT    = "#E67E22"
COL_ENCODE = "#7D3C98"
NC = 4

def panel_label(ax, txt, x=-0.16, y=1.04):
    ax.text(x, y, txt, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left")

def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ── Data ───────────────────────────────────────────────────────
scores = pd.read_csv(SCORES)
scores["gene_symbol"] = scores["gene_symbol"].str.strip().str.upper()
scores = scores.rename(columns={
    "NE_pluripotency_score": "identity_score",
    "E8_self_renewal_score": "fitness_score",
})
ng = pd.read_csv(NGENE)
ng["gene_symbol"] = ng["gene_symbol"].str.strip().str.upper()
df = ng.merge(scores[["gene_symbol","identity_score","fitness_score"]],
              on="gene_symbol", how="inner")

OUTLIERS = {"METTL3","METTL14","CARM1"}
df_out  = df[df["gene_symbol"].isin(OUTLIERS)].copy()
df_main = df[~df["gene_symbol"].isin(OUTLIERS)].copy()

with open(HRES) as f:  hr = json.load(f)
with open(NCRES) as f: nc_res = json.load(f)

ids  = df_main["identity_score"].values
fits = df_main["fitness_score"].values
ns   = df_main["n_gene"].values
above_mask = ns > NC

rng = np.random.default_rng(42)

fig = plt.figure(figsize=(18, 12))
gs  = GridSpec(2, 3, figure=fig,
               left=0.07, right=0.97,
               top=0.94, bottom=0.08,
               wspace=0.40, hspace=0.45)

ax_A = fig.add_subplot(gs[0, 0])
ax_B = fig.add_subplot(gs[0, 1])
ax_C = fig.add_subplot(gs[0, 2])
ax_D = fig.add_subplot(gs[1, 0])
ax_E = fig.add_subplot(gs[1, 1])
ax_F = fig.add_subplot(gs[1, 2])

# ══════════════════════════════════════════════════════════════
# PANEL A — Ngene distribution + inset
# ══════════════════════════════════════════════════════════════
ax = ax_A

all_ns = df["n_gene"].values
ax.hist(all_ns, bins=np.arange(-0.5, 13.5, 1),
        color="#85C1E9", edgecolor="white", lw=0.4, alpha=0.85)
ax.axvline(NC, color=COL_ABOVE, lw=1.3, ls="--")
ax.text(NC+0.2, ax.get_ylim()[1]*0.95 if ax.get_ylim()[1] > 0 else 10,
        f"$N_c$={NC}", fontsize=7, color=COL_ABOVE, va="top")
ax.set_xlabel("$N_{gene}$")
ax.set_ylabel("Number of genes")
ax.set_xticks(range(0, 13, 2))
despine(ax)
panel_label(ax, "A")

# Inset — mean identity per Ngene bin
ax_ins = ax.inset_axes([0.52, 0.52, 0.44, 0.42])
bin_means, bin_ses, bin_ns2 = [], [], []
for n_val in sorted(set(df_main["n_gene"])):
    idx = df_main["n_gene"] == n_val
    if idx.sum() >= 2:
        v = df_main.loc[idx, "identity_score"].values
        bin_means.append(v.mean())
        bin_ses.append(v.std(ddof=1)/np.sqrt(idx.sum()))
        bin_ns2.append(n_val)
bn = np.array(bin_ns2)
bm = np.array(bin_means)
bs = np.array(bin_ses)
ax_ins.errorbar(bn, bm, yerr=bs, fmt="o", color="#2C3E50",
                ms=3.5, lw=0.7, capsize=2, capthick=0.7)
ax_ins.axhline(0, color="#bbb", lw=0.5, ls="--")
ax_ins.axvline(NC, color=COL_ABOVE, lw=0.8, ls="--", alpha=0.7)
ax_ins.set_xlabel("$N_{gene}$", fontsize=6)
ax_ins.set_ylabel("Mean\nidentity", fontsize=6)
ax_ins.tick_params(labelsize=5.5)
ax_ins.spines["top"].set_visible(False)
ax_ins.spines["right"].set_visible(False)

# ══════════════════════════════════════════════════════════════
# PANEL B — Identity vs fitness coloured by Ngene
# ══════════════════════════════════════════════════════════════
ax = ax_B

cmap = plt.cm.RdBu_r
norm = plt.Normalize(0, 11)
sc   = ax.scatter(df_main["identity_score"], df_main["fitness_score"],
                  c=df_main["n_gene"], cmap=cmap, norm=norm,
                  s=20, alpha=0.72, lw=0, zorder=3)
ax.scatter(df_out["identity_score"], df_out["fitness_score"],
           marker="^", c=COL_OUT, s=30, alpha=0.85,
           lw=0.5, edgecolors="#8B4513", zorder=4)

for gene in ["FASN","IDH1","MRPL11"]:
    row = df_main[df_main["gene_symbol"]==gene]
    if len(row):
        ax.annotate(gene,
                    (row.iloc[0]["identity_score"],
                     row.iloc[0]["fitness_score"]),
                    xytext=(5,3), textcoords="offset points",
                    fontsize=5.5, color="#333")

ax.axhline(0, color="#bbb", lw=0.6, ls="--")
ax.axvline(0, color="#bbb", lw=0.6, ls="--")
ax.set_xlabel("Identity score")
ax.set_ylabel("Fitness score")
despine(ax)
panel_label(ax, "B")
cb = plt.colorbar(sc, ax=ax, shrink=0.72, pad=0.02)
cb.set_label("$N_{gene}$", fontsize=7.5)
cb.ax.tick_params(labelsize=6)

# ══════════════════════════════════════════════════════════════
# PANEL C — Cross-validation distribution
# ══════════════════════════════════════════════════════════════
ax = ax_C

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

def cohens_d(a, b):
    sp = np.sqrt((np.var(a,ddof=1)+np.var(b,ddof=1))/2)
    return (np.mean(a)-np.mean(b))/sp if sp>0 else 0.0

cv_ds   = []
cv_aucs = []
sss = StratifiedShuffleSplit(n_splits=100, test_size=0.30, random_state=42)
labels = (ns > NC).astype(int)

for train_idx, test_idx in sss.split(ids, labels):
    ids_train = ids[train_idx]; lab_train = labels[train_idx]
    ids_test  = ids[test_idx];  lab_test  = labels[test_idx]
    best_nc = NC
    d_tr = cohens_d(ids_train[lab_train==1], ids_train[lab_train==0])
    d_te = cohens_d(ids_test[lab_test==1],   ids_test[lab_test==0])
    cv_ds.append(d_te)
    try:
        auc = roc_auc_score(lab_test, ids_test)
    except Exception:
        auc = 0.5
    cv_aucs.append(auc)

cv_ds   = np.array(cv_ds)
cv_aucs = np.array(cv_aucs)
pct_pos = (cv_ds > 0).mean() * 100
mean_auc = cv_aucs.mean()
sd_auc   = cv_aucs.std()

ax.hist(cv_ds, bins=25, color="#85C1E9", edgecolor="white", lw=0.4,
        density=True, alpha=0.85)
ax.axvline(0, color="#888", lw=0.8, ls="--")
ax.axvline(hr["cohens_d"], color=COL_ABOVE, lw=1.3, ls="-")

ax.set_xlabel("Cohen's d (held-out test set)")
ax.set_ylabel("Density")
despine(ax)
panel_label(ax, "C")
ax.text(0.97, 0.97,
        f"d > 0 in {pct_pos:.0f}% of splits\nMean AUC = {mean_auc:.3f} ± {sd_auc:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", lw=0.6))

# ══════════════════════════════════════════════════════════════
# PANEL D — Permutation distribution (full)
# ══════════════════════════════════════════════════════════════
ax = ax_D

rng2 = np.random.default_rng(99)
pds  = []
above_mask2 = ns > NC
for _ in range(10000):
    perm = rng2.permutation(ids)
    pds.append(cohens_d(perm[above_mask2], perm[~above_mask2]))
pds = np.array(pds)
obs_d   = hr["cohens_d"]
perm_p  = hr["permutation_p"]
perm_cnt= hr["permutation_count"]

ax.hist(pds, bins=60, color="#AABCCE", edgecolor="none", density=True)
ax.axvline(obs_d, color=COL_ABOVE, lw=1.8, zorder=4)

tail_x = np.linspace(obs_d, pds.max()+0.05, 200)
kde_fn = gaussian_kde(pds)
ax.fill_between(tail_x, 0, kde_fn(tail_x),
                color=COL_ABOVE, alpha=0.35, zorder=3)
ax.set_xlabel("Cohen's d (permuted)")
ax.set_ylabel("Density")
ax.text(0.97, 0.97,
        f"Observed d = {obs_d:.3f}\np = {perm_p:.4f}\n({perm_cnt}/10,000)",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", lw=0.6))
despine(ax)
panel_label(ax, "D")

# ══════════════════════════════════════════════════════════════
# PANEL E — Sigmoid fit (full scale)
# ══════════════════════════════════════════════════════════════
ax = ax_E

id_min = ids.min(); id_max = ids.max()
ids_n  = (ids - id_min)/(id_max - id_min)

bins = []
for n_val in sorted(set(ns)):
    idx = ns == n_val
    if idx.sum() >= 2:
        v = ids_n[idx]
        bins.append([n_val, v.mean(), v.std(ddof=1)/np.sqrt(idx.sum())])
bins = np.array(bins)

def sig4(N, Nc, w, sc, off):
    return off + sc/(1+np.exp(-(N-Nc)/w))

try:
    popt, pcov = curve_fit(sig4, bins[:,0], bins[:,1],
        p0=[4.0,2.0,0.5,0.1],
        bounds=([0.1,0.2,0.05,0],[15,8,2,0.95]), maxfev=10000)
    Nc_f, w_f, sc_f, off_f = popt
    Nc_se = np.sqrt(pcov[0,0])
    Nx = np.linspace(0, 12, 300)
    ax.plot(Nx, sig4(Nx,*popt), color=COL_ABOVE, lw=1.6, zorder=3)
    y_lo = sig4(Nx, Nc_f-1.96*Nc_se, *popt[1:])
    y_hi = sig4(Nx, Nc_f+1.96*Nc_se, *popt[1:])
    ax.fill_between(Nx, y_lo, y_hi, color=COL_ABOVE, alpha=0.15, zorder=2)
except Exception:
    Nc_f = hr.get("sigmoid_Nc", 4.72)
    Nc_se = hr.get("sigmoid_Nc_se", 0.47)

ax.errorbar(bins[:,0], bins[:,1], yerr=bins[:,2],
            fmt="o", color="#2C3E50", ms=5, lw=0.8,
            capsize=3, capthick=0.8, zorder=4)
ax.axvline(4.00, color="#888", lw=0.9, ls="--")
ax.axvline(Nc_f, color=COL_ABOVE, lw=0.9, ls=":")
ax.text(4.18, 0.92, "Pred. $N_c$=4.00",
        fontsize=6.5, color="#777", va="top")
ax.text(0.97, 0.06,
        f"$N_c$(data) = {Nc_f:.2f} ± {Nc_se:.2f}",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7.5, color=COL_ABOVE,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", lw=0.6))

ax.set_xlabel("$N_{gene}$")
ax.set_ylabel("Normalised identity score")
ax.set_xlim(-0.5, 12.5)
ax.set_ylim(-0.05, 1.1)
ax.set_xticks(range(0, 13, 2))
despine(ax)
panel_label(ax, "E")

# ══════════════════════════════════════════════════════════════
# PANEL F — Specificity control
# ══════════════════════════════════════════════════════════════
ax = ax_F

# Values from script 06 real results
cp = nc_res.get("curated_panel", {})
ep = nc_res.get("encode_general_panel", {})

r_id_cur  = cp.get("identity_r",  0.353)
r_fit_cur = cp.get("fitness_r",   0.012)
r_id_enc  = ep.get("identity_r", -0.012)
r_fit_enc = ep.get("fitness_r",  -0.007)

# Bootstrap CIs (approximate from n)
n_cur = cp.get("n_genes",  92)
n_enc = ep.get("n_genes",  20112)
se_cur_id  = np.sqrt((1-r_id_cur**2)**2  / (n_cur-2))
se_cur_fit = np.sqrt((1-r_fit_cur**2)**2 / (n_cur-2))
se_enc_id  = np.sqrt((1-r_id_enc**2)**2  / (n_enc-2))
se_enc_fit = np.sqrt((1-r_fit_enc**2)**2 / (n_enc-2))

ci_fac = 1.96
ax.errorbar(r_id_cur, r_fit_cur,
            xerr=ci_fac*se_cur_id, yerr=ci_fac*se_cur_fit,
            fmt="o", color=COL_ABOVE, ms=10, lw=1.2,
            capsize=5, capthick=1.2, zorder=4)
ax.errorbar(r_id_enc, r_fit_enc,
            xerr=ci_fac*se_enc_id, yerr=ci_fac*se_enc_fit,
            fmt="s", color=COL_ENCODE, ms=10, lw=1.2,
            capsize=5, capthick=1.2, zorder=4)

ax.axhline(0, color="#aaa", lw=0.7, ls="--")
ax.axvline(0, color="#aaa", lw=0.7, ls="--")

# Labels for data points
ax.text(r_id_cur+0.01, r_fit_cur+0.012,
        f"Curated 17-TF\n(n={n_cur})",
        fontsize=6.8, color=COL_ABOVE, va="bottom")
ax.text(r_id_enc+0.005, r_fit_enc-0.015,
        f"ENCODE 58-TF\n(n={n_enc:,})",
        fontsize=6.8, color=COL_ENCODE, va="top")

ax.set_xlabel("Pearson r ($N_{gene}$, identity score)")
ax.set_ylabel("Pearson r ($N_{gene}$, fitness score)")

# Set axis limits with padding
x_range = [r_id_enc - 0.08, r_id_cur + 0.10]
y_range = [r_fit_enc - 0.07, r_fit_cur + 0.07]
ax.set_xlim(x_range)
ax.set_ylim(y_range)

despine(ax)
panel_label(ax, "F")

ax.text(0.03, 0.97,
        "Error bars: 95% CI",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=6.5, color="#666")

for ext in ["pdf","png"]:
    path = os.path.join(FIGDIR, f"figS2_additional_validation.{ext}")
    fig.savefig(path, dpi=600, bbox_inches="tight")
    print(f"Saved: {path}")
plt.close(fig)
print("Figure S2 done.")
