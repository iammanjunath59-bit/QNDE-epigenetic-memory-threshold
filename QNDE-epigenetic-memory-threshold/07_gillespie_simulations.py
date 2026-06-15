"""
07_gillespie_simulations.py
============================
Exact Gillespie stochastic simulations of the cooperative binding
model (equation 2 of main text) for circuit sizes N = 1 to 12.

Model:
    Binding rate  = (N - n_bound) * (omega + J_eff * n_bound)
    Unbinding rate = n_bound * omega
where omega = 7.7e-4 s^-1, J_eff = g^2/kappa = 9.62e-5 s^-1.

Output:
    results/gillespie_fpt.csv            - mean FPT per N
    results/gillespie_on_fraction.csv    - ON-state fraction per N
    results/gillespie_trajectories.csv   - example trajectories (N=3,4,7)

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

OMEGA   = P["omega"]
KAPPA   = P["kappa_human"]
G       = P["g_midpoint"]
J_EFF   = G**2 / KAPPA
T_MAX   = P["t_max_h"]          # 64800 s = 18 h
SEED    = P["gillespie_seed"]

REPLICATES = {n: (50 if n in {3,4,5} else 30) for n in range(1, 13)}


def gillespie_single(N, omega, J_eff, t_max, rng):
    """
    One exact Gillespie simulation starting from all N TFs bound.
    Returns (fpt, on_fraction) where fpt is the first-passage time
    to the all-unbound state (n=0). If n=0 is not reached within
    t_max, fpt is returned as t_max (right-censored).
    """
    n     = N         # start all bound
    t     = 0.0
    t_on  = 0.0       # time spent with n > 0.8*N
    ON_THRESH = 0.8 * N

    while t < t_max:
        r_off = n * omega
        r_on  = (N - n) * (omega + J_eff * n)
        r_tot = r_off + r_on

        if r_tot == 0:
            break

        dt = rng.exponential(1.0 / r_tot)
        if t + dt > t_max:
            # Accumulate remaining time in on/off before capping
            remaining = t_max - t
            if n > ON_THRESH:
                t_on += remaining
            t = t_max
            break

        if n > ON_THRESH:
            t_on += dt
        t += dt

        # Choose reaction
        if rng.uniform() < r_off / r_tot:
            n -= 1
        else:
            n += 1

        # First passage to all-unbound
        if n == 0:
            return t, t_on / t if t > 0 else 0.0

    # Censored: never reached n=0
    return t_max, t_on / t_max if t_max > 0 else 0.0


def run_gillespie():
    print(f"Running Gillespie simulations (omega={OMEGA}, J_eff={J_EFF:.3e})")
    rng = np.random.default_rng(SEED)

    fpt_rows = []
    on_rows  = []
    traj_data = {}   # N -> list of (t_arr, m_arr) for example trajectories

    for N in range(1, 13):
        n_reps  = REPLICATES[N]
        fpts    = []
        on_fracs = []
        print(f"  N={N:2d}  ({n_reps} replicates)...")

        for rep in range(n_reps):
            fpt, on_f = gillespie_single(N, OMEGA, J_EFF, T_MAX, rng)
            fpts.append(fpt / 3600)      # convert s -> h
            on_fracs.append(on_f)

        fpts_arr = np.array(fpts)
        censored = fpts_arr >= T_MAX / 3600

        fpt_rows.append({
            "N":            N,
            "mean_fpt_h":   float(fpts_arr.mean()),
            "se_fpt_h":     float(fpts_arr.std(ddof=1) / np.sqrt(n_reps)),
            "censored":     bool(censored.all()),
            "n_censored":   int(censored.sum()),
            "n_replicates": n_reps,
        })
        on_rows.append({
            "N":           N,
            "on_fraction": float(np.mean(on_fracs)),
        })
        print(f"    mean FPT = {fpts_arr.mean():.2f} h  "
              f"(censored: {censored.sum()}/{n_reps})")

    df_fpt = pd.DataFrame(fpt_rows)
    df_on  = pd.DataFrame(on_rows)

    df_fpt.to_csv(os.path.join(RES, "gillespie_fpt.csv"), index=False)
    df_on.to_csv(os.path.join(RES, "gillespie_on_fraction.csv"), index=False)
    print(f"\nSaved gillespie_fpt.csv and gillespie_on_fraction.csv")

    # ── Example trajectories for Figure S1C-E ────────────────
    print("\nGenerating example trajectories for N = 3, 4, 7...")
    traj_rows = []
    for N, seed_offset in [(3, 100), (4, 200), (7, 300)]:
        rng_t  = np.random.default_rng(SEED + seed_offset)
        n_curr = N
        t_curr = 0.0
        ts     = [0.0]
        ms     = [1.0]

        while t_curr < T_MAX:
            r_off = n_curr * OMEGA
            r_on  = (N - n_curr) * (OMEGA + J_EFF * n_curr)
            r_tot = r_off + r_on
            if r_tot == 0:
                break
            dt    = rng_t.exponential(1.0 / r_tot)
            if t_curr + dt > T_MAX:
                break
            t_curr += dt
            if rng_t.uniform() < r_off / r_tot:
                n_curr = max(0, n_curr - 1)
            else:
                n_curr = min(N, n_curr + 1)
            ts.append(t_curr / 3600)
            ms.append(n_curr / N)
            if n_curr == 0 and len(ts) > 1:
                break   # stop at FPT for below-threshold circuits

        for t_val, m_val in zip(ts[::max(1, len(ts)//500)],
                                  ms[::max(1, len(ms)//500)]):
            traj_rows.append({"N": N, "t_h": t_val, "m": m_val})

    pd.DataFrame(traj_rows).to_csv(
        os.path.join(RES, "gillespie_trajectories.csv"), index=False)
    print("Saved gillespie_trajectories.csv")
    return df_fpt, df_on


if __name__ == "__main__":
    run_gillespie()
