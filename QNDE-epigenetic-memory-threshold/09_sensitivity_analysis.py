"""
09_sensitivity_analysis.py
===========================
Tests robustness of the primary results to:
  (A) Outlier inclusion / exclusion
  (B) N_c threshold choice (Nc = 3, 4, 5)
  (C) N_gene aggregation rule (max / mean / sum across elements)
  (D) g parameter value across the full biophysical range

For each condition, computes Cohen's d and permutation p-value
on the human 110-gene dataset.

Input:
    data/processed/human_ngene_112.csv
    data/raw/rosen2024_scores.csv

Output:
    results/sensitivity_results.json

Author: [Author Name]
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, "data", "processed")
RAW  = os.path.join(BASE, "data", "raw")
RES  = os.path.join(BASE, "results")
os.makedirs(RES, exist_ok=True)

with open(os.path.join(BASE, "params", "model_params.json")) as f:
    P = json.load(f)

OUTLIERS = {"METTL3", "METTL14", "CARM1"}


def cohens_d(a, b):
    sp = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return (np.mean(a) - np.mean(b)) / sp if sp > 0 else 0.0


def perm_test(ids, above, n_perm=10000, seed=99):
    rng = np.random.default_rng(seed)
    obs = cohens_d(ids[above], ids[~above])
    pds = [cohens_d(p[above], p[~above]) for p in
           [rng.permutation(ids) for _ in range(n_perm)]]
    return obs, float(np.mean(np.array(pds) >= obs))


def run_sensitivity():
    scores = pd.read_csv(os.path.join(RAW, "rosen2024_scores.csv"))
    scores = scores.rename(columns={
        "NE_pluripotency_score": "identity_score",
        "E8_self_renewal_score": "fitness_score",
    })
    ngene  = pd.read_csv(os.path.join(PROC, "human_ngene_112.csv"))
    df     = ngene.merge(
        scores[["gene_symbol","identity_score","fitness_score"]],
        on="gene_symbol", how="inner")

    results = {}

    # ── A: Outlier sensitivity ────────────────────────────────
    print("A: Outlier sensitivity...")
    for label, excl in [("with_outliers", set()), ("without_outliers", OUTLIERS)]:
        df_sub = df[~df["gene_symbol"].isin(excl)]
        ids    = df_sub["identity_score"].values
        ns     = df_sub["n_gene"].values
        above  = ns > P["NC_H_illustrative"]
        d, p   = perm_test(ids, above)
        results[f"outlier_{label}"] = {
            "n": int(len(df_sub)), "d": float(d), "perm_p": float(p)}
        print(f"  {label}: n={len(df_sub)}, d={d:.3f}, p={p:.4f}")

    # ── B: N_c threshold sensitivity ─────────────────────────
    print("\nB: N_c threshold sensitivity...")
    df_clean = df[~df["gene_symbol"].isin(OUTLIERS)]
    ids_c    = df_clean["identity_score"].values
    ns_c     = df_clean["n_gene"].values
    nc_results = {}
    for nc_test in range(2, 9):
        above = ns_c > nc_test
        if above.sum() < 3 or (~above).sum() < 3:
            continue
        d, p = perm_test(ids_c, above)
        nc_results[str(nc_test)] = {"d": float(d), "perm_p": float(p),
                                     "n_above": int(above.sum())}
        print(f"  Nc={nc_test}: d={d:.3f}, p={p:.4f}, n_above={above.sum()}")
    results["nc_threshold_sensitivity"] = nc_results

    # ── C: Aggregation rule sensitivity ──────────────────────
    # The manuscript uses max across regulatory elements.
    # Test mean and sum as alternatives.
    # NOTE: This requires multi-element N_gene table.
    # If only a single N_gene value is available, this test is skipped.
    print("\nC: Aggregation rule sensitivity...")
    if "n_gene_mean" in df.columns and "n_gene_sum" in df.columns:
        for agg, col in [("max","n_gene"), ("mean","n_gene_mean"),
                          ("sum","n_gene_sum")]:
            df_s = df[~df["gene_symbol"].isin(OUTLIERS)]
            ns_a = df_s[col].values
            ids_a = df_s["identity_score"].values
            above = ns_a > P["NC_H_illustrative"]
            d, p  = perm_test(ids_a, above)
            results[f"agg_{agg}"] = {"d": float(d), "perm_p": float(p)}
            print(f"  {agg}: d={d:.3f}, p={p:.4f}")
    else:
        print("  Skipped: multi-element N_gene table not available.")
        results["agg_sensitivity"] = "skipped: single N_gene value per gene"

    # ── D: g parameter range sensitivity ─────────────────────
    # The predicted N_c range depends on g. For each g value, compute
    # the predicted N_c and test at that threshold.
    print("\nD: g parameter sensitivity...")
    omega   = P["omega"]
    kappa   = P["kappa_human"]
    g_vals  = np.linspace(P["g_lo"], P["g_hi"], 20)
    g_sensitivity = []
    for g_val in g_vals:
        j_eff   = g_val**2 / kappa
        nc_pred = omega / (2 * j_eff)
        nc_test = int(round(nc_pred))
        nc_test = max(1, min(12, nc_test))

        df_s   = df[~df["gene_symbol"].isin(OUTLIERS)]
        ids_s  = df_s["identity_score"].values
        ns_s   = df_s["n_gene"].values
        above  = ns_s > nc_test
        if above.sum() < 3 or (~above).sum() < 3:
            continue
        d, p   = perm_test(ids_s, above, n_perm=2000)   # fewer for speed
        g_sensitivity.append({
            "g": float(g_val), "J_eff": float(j_eff),
            "Nc_predicted": float(nc_pred), "Nc_tested": nc_test,
            "d": float(d), "perm_p": float(p),
        })
        print(f"  g={g_val:.2e}: Nc_pred={nc_pred:.2f}, test at {nc_test}, "
              f"d={d:.3f}, p={p:.4f}")
    results["g_sensitivity"] = g_sensitivity

    out_path = os.path.join(RES, "sensitivity_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    return results


if __name__ == "__main__":
    run_sensitivity()
