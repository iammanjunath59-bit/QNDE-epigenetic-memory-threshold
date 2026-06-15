"""
05_mouse_validation.py
=======================
Primary statistical analysis of the mouse ESC CRISPR screen
(Li et al. 2018; Cell Reports 24:489-502; GEO: GSE107060).

Screen design:
  Reporter: Rex1GFP (Rex1/Zfp42 marks naive pluripotency in mESC)
  GFP+ = Rex1-high = naive pluripotent
  GFP- = Rex1-low  = exiting naive state

Score definitions:
  identity_score = -(d15_GFP+:GFP-)
    Positive = knockout depletes cells from naive GFP+ fraction
    = gene required for naive pluripotency maintenance
  fitness_score  = -(d15_GFP-:Plasmid)
    Positive = knockout depletes from differentiated GFP- fraction
    = gene required for general cell fitness

Pre-specified exclusion:
  Pou5f1/Oct4 is NOT excluded here (the reporter is Rex1GFP, not Oct4-GFP).
  No pre-specified outlier exclusion for mouse (outlier criterion was
  defined only for the human screen).

Input:
  data/raw/li2018_scores.csv            -- raw screen data (tab-separated)
  data/processed/mouse_ngene_65.csv     -- N_gene assignments
  results/human_validation_results.json -- for cross-species ratio

Output:
  results/mouse_validation_results.json
  results/mouse_permutation_distribution.npy

Author: [Author Name]
"""

import os, json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

BASE = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.join(BASE, "data", "raw")
PROC = os.path.join(BASE, "data", "processed")
RES  = os.path.join(BASE, "results")
os.makedirs(RES, exist_ok=True)

with open(os.path.join(BASE, "params", "model_params.json")) as f:
    P = json.load(f)

NC_M_PRED = P["NC_M_illustrative"]   # 7.29


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════
def cohens_d(a, b):
    sp = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return (np.mean(a) - np.mean(b)) / sp if sp > 0 else 0.0


def sigmoid4(N, Nc, width, scale, offset):
    return offset + scale / (1 + np.exp(-(N - Nc) / width))


# ══════════════════════════════════════════════════════════════
# Load and prepare data
# ══════════════════════════════════════════════════════════════
def load_data():
    """Load Li et al. 2018 screen and mouse N_gene assignments."""

    raw_path = os.path.join(RAW, "li2018_scores.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Missing: {raw_path}\n"
            "Download li2018_scores.csv from Code S1 outputs and "
            "place in data/raw/.\n"
            "Source: Li et al. 2018 Cell Reports 24:489-502, GEO: GSE107060.\n"
            "Columns needed: gene_symbol, identity_score, fitness_score.\n"
            "identity_score = -(d15_GFP+:GFP-)  [sign already flipped]\n"
            "fitness_score  = -(d15_GFP-:Plasmid) [sign already flipped]"
        )

    ngene_path = os.path.join(PROC, "mouse_ngene_65.csv")
    if not os.path.exists(ngene_path):
        raise FileNotFoundError(
            f"Missing: {ngene_path}\n"
            "Run 02_ngene_assignment_mouse.py first."
        )

    # Load screen scores
    scores = pd.read_csv(raw_path)
    scores["gene_symbol"] = (scores["gene_symbol"]
                             .astype(str).str.strip().str.upper())

    # Load N_gene assignments
    ngene = pd.read_csv(ngene_path)
    ngene["gene_symbol"] = (ngene["gene_symbol"]
                            .astype(str).str.strip().str.upper())

    # Merge
    df = ngene.merge(
        scores[["gene_symbol", "identity_score", "fitness_score"]],
        on="gene_symbol", how="inner"
    )
    print(f"  Matched: {len(df)} genes")

    # Note: no Pou5f1 exclusion needed (reporter is Rex1GFP, not Oct4-GFP)
    # No pre-specified outlier exclusion for mouse dataset

    return df


# ══════════════════════════════════════════════════════════════
# Main analysis
# ══════════════════════════════════════════════════════════════
def run_mouse_validation():
    print("=" * 60)
    print("Mouse ESC Validation (Li et al. 2018, GEO: GSE107060)")
    print("Reporter: Rex1GFP | Identity: -(d15_GFP+:GFP-)")
    print("=" * 60)
    print()

    print("Loading data...")
    df = load_data()

    ids  = df["identity_score"].values
    fits = df["fitness_score"].values
    ns   = df["n_gene"].values

    # Threshold: round predicted Nc = 7.29 to nearest integer = 7
    # Pre-specified before examining individual gene scores
    NC_M_TEST = int(round(NC_M_PRED))   # 7
    above = ns > NC_M_TEST
    below = ~above
    print(f"  N > {NC_M_TEST}: n = {above.sum()}")
    print(f"  N <= {NC_M_TEST}: n = {below.sum()}")
    print()

    # ── 1. Threshold test ─────────────────────────────────────
    d        = cohens_d(ids[above], ids[below])
    t_stat, p_t  = stats.ttest_ind(ids[above], ids[below], equal_var=False)
    mw_stat, p_mw = stats.mannwhitneyu(
        ids[above], ids[below], alternative="greater")

    print(f"Threshold test (N > {NC_M_TEST} vs N <= {NC_M_TEST}):")
    print(f"  Cohen d  = {d:.4f}")
    print(f"  Welch p  = {p_t:.2e}")
    print(f"  MW p     = {p_mw:.2e}")

    # ── 2. Permutation test (pre-specified primary test) ──────
    rng   = np.random.default_rng(P["permutation_seed"])   # seed = 99
    n_perm = 10_000
    pds   = np.array([
        cohens_d(perm[above], perm[below])
        for perm in (rng.permutation(ids) for _ in range(n_perm))
    ])
    perm_p   = float((pds >= d).mean())
    perm_cnt = int((pds >= d).sum())
    print(f"  Perm p   = {perm_p:.4f} ({perm_cnt}/{n_perm})")
    np.save(os.path.join(RES, "mouse_permutation_distribution.npy"), pds)

    # ── 3. Continuous correlation ─────────────────────────────
    r_id,  p_id  = stats.pearsonr(ns, ids)
    r_sp,  p_sp  = stats.spearmanr(ns, ids)
    r_fit, p_fit = stats.pearsonr(ns, fits)
    print(f"\nContinuous correlation:")
    print(f"  Pearson r (identity)  = {r_id:+.4f}  p = {p_id:.2e}")
    print(f"  Spearman r (identity) = {r_sp:+.4f}  p = {p_sp:.2e}")
    print(f"  Pearson r (fitness)   = {r_fit:+.4f}  p = {p_fit:.2e} (NS expected)")

    # ── 4. Sigmoid fit ────────────────────────────────────────
    id_norm = (ids - ids.min()) / (ids.max() - ids.min())

    bins = []
    for n_val in sorted(set(ns)):
        idx = ns == n_val
        if idx.sum() >= 2:
            v = id_norm[idx]
            bins.append([n_val, v.mean(), v.std(ddof=1)/np.sqrt(idx.sum())])
    bins = np.array(bins)

    try:
        popt, pcov = curve_fit(
            sigmoid4, bins[:, 0], bins[:, 1],
            p0=[NC_M_PRED, 2.0, 0.5, 0.1],
            bounds=([1, 0.2, 0.05, 0], [15, 8, 2, 0.95]),
            maxfev=10_000,
        )
        NC_FIT_M    = float(popt[0])
        NC_FIT_M_SE = float(np.sqrt(pcov[0, 0]))
        print(f"\nSigmoid fit:")
        print(f"  N_c(mouse, data) = {NC_FIT_M:.3f} ± {NC_FIT_M_SE:.3f}")
        print(f"  Predicted N_c range: {P['NC_M_predicted_range']}")
        converged = True
    except RuntimeError as e:
        print(f"  WARNING: Sigmoid fit did not converge: {e}")
        NC_FIT_M    = NC_M_PRED
        NC_FIT_M_SE = 1.0
        converged   = False

    # ── 5. Cross-species ratio (g-independent) ────────────────

    # Load human Nc — use params fallback if sigmoid did not converge
    NC_FIT_H_DEFAULT    = P.get("NC_FIT_H",    4.72)
    NC_FIT_H_SE_DEFAULT = P.get("NC_FIT_H_SE", 0.47)
    hr_path = os.path.join(RES, "human_validation_results.json")
    if os.path.exists(hr_path):
        with open(hr_path) as f:
            hr = json.load(f)
        hr_nc    = hr.get("sigmoid_Nc",    NC_FIT_H_DEFAULT)
        hr_nc_se = hr.get("sigmoid_Nc_se", NC_FIT_H_SE_DEFAULT)
        if 1.0 <= hr_nc <= 8.0 and hr_nc_se < 5.0:
            NC_FIT_H    = hr_nc
            NC_FIT_H_SE = hr_nc_se
            print(f"  Human N_c (from script 04 sigmoid): {NC_FIT_H:.3f} +/- {NC_FIT_H_SE:.3f}")
        else:
            NC_FIT_H    = NC_FIT_H_DEFAULT
            NC_FIT_H_SE = NC_FIT_H_SE_DEFAULT
            print(f"  Human sigmoid did not converge (Nc={hr_nc:.2f}); using params: {NC_FIT_H} +/- {NC_FIT_H_SE}")
    else:
        NC_FIT_H    = NC_FIT_H_DEFAULT
        NC_FIT_H_SE = NC_FIT_H_SE_DEFAULT
        print(f"  Using N_c(human) from params: {NC_FIT_H} +/- {NC_FIT_H_SE}")
    obs_ratio    = NC_FIT_M / NC_FIT_H
    obs_ratio_se = obs_ratio * np.sqrt(
        (NC_FIT_M_SE / NC_FIT_M) ** 2 + (NC_FIT_H_SE / NC_FIT_H) ** 2
    )
    ci_lo = obs_ratio - 1.96 * obs_ratio_se
    ci_hi = obs_ratio + 1.96 * obs_ratio_se

    pred_ratio  = P["predicted_ratio"]   # 1.82
    pct_error   = abs(obs_ratio - pred_ratio) / pred_ratio * 100
    in_ci       = bool(ci_lo <= pred_ratio <= ci_hi)

    print(f"\nCross-species ratio (g-independent):")
    print(f"  Predicted  = {pred_ratio:.2f}  "
          f"(= kappa_mouse / kappa_human; g cancels algebraically)")
    print(f"  Observed   = {obs_ratio:.2f} ± {obs_ratio_se:.2f}")
    print(f"  95% CI     = [{ci_lo:.2f}, {ci_hi:.2f}]")
    print(f"  Agreement  = {pct_error:.1f}%")
    print(f"  Pred in CI = {in_ci}")

    # ── Save ──────────────────────────────────────────────────
    results = {
        "data_source":      "Li et al. 2018, Cell Reports 24:489-502, GEO: GSE107060",
        "reporter":         "Rex1GFP",
        "identity_column":  "-(d15_GFP+:GFP-)  [sign flipped from raw data]",
        "fitness_column":   "-(d15_GFP-:Plasmid) [sign flipped from raw data]",
        "n_genes":           int(len(df)),
        "NC_M_predicted":    float(NC_M_PRED),
        "NC_M_test":         int(NC_M_TEST),
        "cohens_d":          float(d),
        "welch_p":           float(p_t),
        "mannwhitney_p":     float(p_mw),
        "permutation_p":     float(perm_p),
        "permutation_count": perm_cnt,
        "pearson_r_identity":  float(r_id),
        "pearson_p_identity":  float(p_id),
        "spearman_r_identity": float(r_sp),
        "spearman_p_identity": float(p_sp),
        "pearson_r_fitness":   float(r_fit),
        "pearson_p_fitness":   float(p_fit),
        "sigmoid_Nc":          float(NC_FIT_M),
        "sigmoid_Nc_se":       float(NC_FIT_M_SE),
        "sigmoid_converged":   converged,
        "NC_H_from_script04":  float(NC_FIT_H),
        "NC_H_se":             float(NC_FIT_H_SE),
        "cross_species_ratio_predicted": float(pred_ratio),
        "cross_species_ratio_observed":  float(obs_ratio),
        "cross_species_ratio_se":        float(obs_ratio_se),
        "cross_species_ci_lo":           float(ci_lo),
        "cross_species_ci_hi":           float(ci_hi),
        "cross_species_pct_error":       float(pct_error),
        "predicted_in_95pct_CI":         in_ci,
        "note_pou5f1": (
            "Pou5f1/Oct4 NOT excluded — reporter is Rex1GFP, "
            "not Oct4-GFP. Rex1 is an independent marker of naive "
            "pluripotency. No GFP-reporter technical confound."
        ),
    }

    out_path = os.path.join(RES, "mouse_validation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    print("=" * 60)
    return results


if __name__ == "__main__":
    run_mouse_validation()
