"""
04_human_validation.py
=======================
Primary statistical analysis of the human ESC CRISPR screen
(Rosen et al. 2024; GEO: GSE278910) against the N_gene predictions.

All statistical tests and their order of application were
pre-specified before any gene-level score was examined.
The permutation test is the pre-specified primary significance test.

Input:
    data/raw/rosen2024_scores.csv          - CRISPR screen scores
    data/processed/human_ngene_112.csv     - N_gene assignments

Output:
    results/human_validation_results.json
    results/human_permutation_distribution.npy

Author: [Author Name]
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

BASE  = os.path.dirname(os.path.abspath(__file__))
PROC  = os.path.join(BASE, "data", "processed")
RAW   = os.path.join(BASE, "data", "raw")
RES   = os.path.join(BASE, "results")
os.makedirs(RES, exist_ok=True)

# Load model parameters
with open(os.path.join(BASE, "params", "model_params.json")) as f:
    PARAMS = json.load(f)
NC_H = PARAMS["NC_H_illustrative"]   # 4.00

# Pre-specified outlier criterion
OUTLIER_THRESHOLD_SD = PARAMS["outlier_SD_threshold"]   # 3.5


def cohens_d(a, b):
    """Pooled-SD Cohen's d."""
    sp = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return (np.mean(a) - np.mean(b)) / sp if sp > 0 else 0.0


def sigmoid4(N, Nc, width, scale, offset):
    return offset + scale / (1 + np.exp(-(N - Nc) / width))


def partial_correlation_is_tf(identity, n_gene, is_tf):
    """Partial r(N_gene, identity | Is-TF)."""
    reg_n = LinearRegression().fit(is_tf.reshape(-1,1), n_gene)
    reg_i = LinearRegression().fit(is_tf.reshape(-1,1), identity)
    resid_n = n_gene  - reg_n.predict(is_tf.reshape(-1,1))
    resid_i = identity - reg_i.predict(is_tf.reshape(-1,1))
    r, p = stats.pearsonr(resid_n, resid_i)
    return r, p


def run_human_validation():
    # ── Load data ──────────────────────────────────────────────
    print("Loading data...")
    scores = pd.read_csv(os.path.join(RAW, "rosen2024_scores.csv"))
    ngene  = pd.read_csv(os.path.join(PROC, "human_ngene_112.csv"))

    # Standardise column names
    scores = scores.rename(columns={
        "NE_pluripotency_score": "identity_score",
        "E8_self_renewal_score": "fitness_score",
    })
    scores["gene_symbol"] = scores["gene_symbol"].str.upper().str.strip()
    ngene["gene_symbol"]  = ngene["gene_symbol"].str.upper().str.strip()

    df = ngene.merge(scores[["gene_symbol","identity_score","fitness_score"]],
                     on="gene_symbol", how="inner")
    print(f"  Matched: {len(df)} genes")

    # ── Pre-specified outlier removal ──────────────────────────
    # Criterion: identity score > 3.5 SD below group mean for N_gene bin
    # This criterion was pre-specified before examining any gene scores.
    df["bin_mean"] = df.groupby("n_gene")["identity_score"].transform("mean")
    df["bin_std"]  = df.groupby("n_gene")["identity_score"].transform("std")
    df["outlier"]  = (
        (df["identity_score"] < df["bin_mean"] - OUTLIER_THRESHOLD_SD * df["bin_std"])
        | (df["outlier_flag"] == 1)
    )
    df_clean = df[~df["outlier"]].copy()
    outlier_genes = df[df["outlier"]]["gene_symbol"].tolist()
    print(f"  Outliers (pre-specified criterion {OUTLIER_THRESHOLD_SD} SD): "
          f"{outlier_genes}")
    print(f"  Analysis n = {len(df_clean)}")

    ids  = df_clean["identity_score"].values
    fits = df_clean["fitness_score"].values
    ns   = df_clean["n_gene"].values

    above = ns > NC_H
    below = ~above

    # ── 1. Threshold test: Cohen's d ──────────────────────────
    d = cohens_d(ids[above], ids[below])
    t_stat, p_t = stats.ttest_ind(ids[above], ids[below], equal_var=False)
    mw_stat, p_mw = stats.mannwhitneyu(ids[above], ids[below],
                                        alternative="greater")
    print(f"\nThreshold test (N > {NC_H} vs N <= {NC_H}):")
    print(f"  Cohen d = {d:.3f}  Welch p = {p_t:.2e}  MW p = {p_mw:.2e}")

    # ── 2. Permutation test (PRE-SPECIFIED PRIMARY TEST) ──────
    rng_perm = np.random.default_rng(PARAMS["permutation_seed"])
    n_perm   = 10_000
    perm_ds  = []
    for _ in range(n_perm):
        perm    = rng_perm.permutation(ids)
        perm_d  = cohens_d(perm[above], perm[below])
        perm_ds.append(perm_d)
    perm_ds    = np.array(perm_ds)
    perm_p     = (perm_ds >= d).mean()
    perm_count = int((perm_ds >= d).sum())
    print(f"  Permutation p = {perm_p:.4f} ({perm_count}/10,000)")
    np.save(os.path.join(RES, "human_permutation_distribution.npy"), perm_ds)

    # ── 3. Continuous correlation: Pearson and Spearman ───────
    r_pearson, p_pearson  = stats.pearsonr(ns, ids)
    r_spearman, p_spearman = stats.spearmanr(ns, ids)
    print(f"\nContinuous correlation:")
    print(f"  Pearson r  = {r_pearson:.3f}  p = {p_pearson:.2e}")
    print(f"  Spearman r = {r_spearman:.3f}  p = {p_spearman:.2e}")

    # ── 4. Identity-fitness dissociation ─────────────────────
    r_fit, p_fit = stats.pearsonr(ns, fits)
    print(f"\nFitness correlation:")
    print(f"  Pearson r = {r_fit:.3f}  p = {p_fit:.2e}")

    # ── 5. Cross-validation ───────────────────────────────────
    rng_cv  = np.random.default_rng(PARAMS["cv_seed"])
    cv      = StratifiedShuffleSplit(n_splits=100, test_size=0.30,
                                      random_state=42)
    # Stratify by above/below threshold
    labels  = (ns > NC_H).astype(int)
    cv_ds   = []
    cv_aucs = []
    for tr_idx, te_idx in cv.split(ids, labels):
        ids_te  = ids[te_idx]
        ns_te   = ns[te_idx]
        lab_te  = labels[te_idx]
        ab_te   = ns_te > NC_H
        be_te   = ~ab_te
        if ab_te.sum() > 1 and be_te.sum() > 1:
            cv_ds.append(cohens_d(ids_te[ab_te], ids_te[be_te]))
        if len(np.unique(lab_te)) == 2:
            cv_aucs.append(roc_auc_score(lab_te, ids_te))
    cv_d_pos_pct = (np.array(cv_ds) > 0).mean() * 100
    cv_auc_mean  = np.mean(cv_aucs)
    cv_auc_sd    = np.std(cv_aucs)
    print(f"\nCross-validation (100 x 70/30 splits):")
    print(f"  d>0 in {cv_d_pos_pct:.0f}% of splits")
    print(f"  Mean AUC = {cv_auc_mean:.3f} ± {cv_auc_sd:.3f}")

    # ── 6. Sigmoid fit ────────────────────────────────────────
    # Normalise identity scores to [0,1]
    id_min = ids.min(); id_max = ids.max()
    ids_norm = (ids - id_min) / (id_max - id_min)

    # Bin means for fitting (bins with >= 2 genes)
    bin_ns = []; bin_means = []; bin_ses = []
    for n_val in sorted(set(ns)):
        idx = ns == n_val
        if idx.sum() >= 2:
            v = ids_norm[idx]
            bin_ns.append(n_val)
            bin_means.append(v.mean())
            bin_ses.append(v.std(ddof=1) / np.sqrt(idx.sum()))

    try:
        popt, pcov = curve_fit(
            sigmoid4,
            bin_ns, bin_means,
            p0=[NC_H, 2.0, 0.5, 0.1],
            bounds=([0.1, 0.2, 0.05, 0.0], [15, 8, 2, 0.95]),
            maxfev=10_000,
        )
        Nc_fit   = popt[0]
        Nc_fit_se = np.sqrt(pcov[0, 0])
        print(f"\nSigmoid fit:")
        print(f"  N_c(data) = {Nc_fit:.2f} ± {Nc_fit_se:.2f}")
        print(f"  Predicted range: [{PARAMS['NC_H_predicted_range'][0]}, "
              f"{PARAMS['NC_H_predicted_range'][1]}]")
    except RuntimeError:
        Nc_fit = np.nan; Nc_fit_se = np.nan
        print("  WARNING: Sigmoid fit did not converge.")

    # ── 7. Partial correlation (controlling Is-TF) ────────────
    # Is-TF: 1 if gene encodes a transcription factor, 0 otherwise
    # Using AnimalTFDB 3.0 classification; encoded in the N_gene table
    # as column 'is_tf' if present, else skip.
    partial_r = partial_p = None
    if "is_tf" in df_clean.columns:
        is_tf_arr = df_clean["is_tf"].values.astype(float)
        partial_r, partial_p = partial_correlation_is_tf(ids, ns, is_tf_arr)
        print(f"\nPartial correlation (controlling Is-TF):")
        print(f"  r = {partial_r:.3f}  p = {partial_p:.2e}")

    # ── Save results ──────────────────────────────────────────
    results = {
        "n_genes":           int(len(df_clean)),
        "outlier_genes":     outlier_genes,
        "Nc_predicted":      float(NC_H),
        "cohens_d":          float(d),
        "welch_p":           float(p_t),
        "mannwhitney_p":     float(p_mw),
        "permutation_p":     float(perm_p),
        "permutation_count": perm_count,
        "pearson_r_identity": float(r_pearson),
        "pearson_p_identity": float(p_pearson),
        "spearman_r_identity": float(r_spearman),
        "spearman_p_identity": float(p_spearman),
        "pearson_r_fitness":  float(r_fit),
        "pearson_p_fitness":  float(p_fit),
        "cv_d_positive_pct":  float(cv_d_pos_pct),
        "cv_auc_mean":        float(cv_auc_mean),
        "cv_auc_sd":          float(cv_auc_sd),
        "sigmoid_Nc":         float(Nc_fit) if not np.isnan(Nc_fit) else None,
        "sigmoid_Nc_se":      float(Nc_fit_se) if not np.isnan(Nc_fit_se) else None,
        "partial_r_is_tf":    float(partial_r) if partial_r else None,
        "partial_p_is_tf":    float(partial_p) if partial_p else None,
    }

    out_path = os.path.join(RES, "human_validation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    return results


if __name__ == "__main__":
    run_human_validation()
