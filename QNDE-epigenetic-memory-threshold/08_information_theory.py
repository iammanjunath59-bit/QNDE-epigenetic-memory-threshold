"""
08_information_theory.py
=========================
Computes the near-threshold approximation for the memory error
probability p_e(N) and the Shannon channel capacity C(N) for
both human and mouse ESCs.

Equations (main text):
    p_e(N) = 1 / (1 + exp[(N - N_c) / N_c])   [near-threshold approximation]
    C(N)   = max(0, 1 - H_b(p_e(N)))           [bits per cell division]
    H_b(p) = -p log2(p) - (1-p) log2(1-p)

IMPORTANT: The logistic form for p_e(N) is a near-threshold approximation
satisfying p_e(N_c) = 1/2. It is NOT uniquely derived from the
Fokker-Planck equation. See Supplementary Note 2.
The direct Fokker-Planck result gives exp(N - N_c) in the exponent;
the denominator N_c is chosen for transition width consistency.
C(N_c) = 0 exactly follows from p_e(N_c) = 1/2, independent of
which functional form is used.

Output:
    results/channel_capacity_curves.csv

Author: [Author Name]
"""

import os
import json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(BASE, "results")
os.makedirs(RES, exist_ok=True)

with open(os.path.join(BASE, "params", "model_params.json")) as f:
    P = json.load(f)


def pe(N, Nc):
    """Near-threshold approximation for memory error probability."""
    N  = np.asarray(N, dtype=float)
    return 1.0 / (1.0 + np.exp((N - Nc) / Nc))


def Hb(p):
    """Binary entropy function."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def channel_capacity(N, Nc):
    """Shannon channel capacity C(N) = max(0, 1 - H_b(p_e(N)))."""
    return np.maximum(0.0, 1.0 - Hb(pe(N, Nc)))


def compute_curves():
    N_arr = np.linspace(0.1, 14.0, 500)

    # Human ESC parameters
    NC_H   = P["NC_H_illustrative"]
    NC_M   = P["NC_M_illustrative"]
    NC_H_f = P["NC_FIT_H"]
    NC_M_f = P["NC_FIT_M"]

    rows = []
    for n in N_arr:
        rows.append({
            "N":                n,
            "pe_human_pred":    float(pe(n, NC_H)),
            "pe_mouse_pred":    float(pe(n, NC_M)),
            "pe_human_fit":     float(pe(n, NC_H_f)),
            "pe_mouse_fit":     float(pe(n, NC_M_f)),
            "C_human_pred":     float(channel_capacity(n, NC_H)),
            "C_mouse_pred":     float(channel_capacity(n, NC_M)),
            "C_human_fit":      float(channel_capacity(n, NC_H_f)),
            "C_mouse_fit":      float(channel_capacity(n, NC_M_f)),
        })

    df = pd.DataFrame(rows)
    out = os.path.join(RES, "channel_capacity_curves.csv")
    df.to_csv(out, index=False)
    print(f"Saved: {out}")

    # ── Key point checks ─────────────────────────────────────
    print(f"\nKey results:")
    print(f"  p_e(N_c_human = {NC_H}) = {pe(NC_H, NC_H):.6f}  (should be 0.5 exactly)")
    print(f"  C(N_c_human   = {NC_H}) = {channel_capacity(NC_H, NC_H):.6f}  (should be 0 exactly)")
    print(f"  C(POU5F1, N=10, human) = {channel_capacity(10, NC_H):.3f} bits/division")
    print(f"  C(NANOG,  N=11, human) = {channel_capacity(11, NC_H):.3f} bits/division")
    print(f"  C(POU5F1, N=10, mouse) = {channel_capacity(10, NC_M):.3f} bits/division  (8-fold reduction)")
    print(f"\nNOTE: These C(N) values are illustrative.")
    print(f"  The logistic form p_e(N) is a near-threshold approximation.")
    print(f"  See Supplementary Note 2 for derivation and scope of approximation.")

    return df


if __name__ == "__main__":
    compute_curves()
