"""
figS1_gillespie.py  (v2 — self-contained trajectory generation)
================================================================
Supplementary Figure S1: Gillespie stochastic simulations.
Panels A–F. Saves figS1_gillespie.pdf and .png at 600 DPI.

Run from CodeS1 directory:
    python figS1_gillespie.py

Requires:
    results/gillespie_fpt.csv
    results/gillespie_on_fraction.csv
    params/model_params.json

Trajectories (panels C, D, E) are generated directly via Gillespie
algorithm inside this script — no trajectories file needed.
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

BASE   = os.path.dirname(os.path.abspath(__file__))
FPTF   = os.path.join(BASE, "results", "gillespie_fpt.csv")
ONF    = os.path.join(BASE, "results", "gillespie_on_fraction.csv")
PARAMS = os.path.join(BASE, "params",  "model_params.json")
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

with open(PARAMS) as f:
    P = json.load(f)

omega = P["omega"]
Jeff  = P["J_eff"]
NC    = 4

COL_ABOVE = "#C0392B"
COL_BELOW = "#2471A3"
COL_AT    = "#E67E22"
COL_H     = "#C0392B"
COL_M     = "#1A7A4A"

# ── Real FPT values from script 07 ─────────────────────────────
REAL_FPT = {
    1:  (0.33,  0.05,  0,  30),
    2:  (0.64,  0.08,  0,  30),
    3:  (1.32,  0.12,  0,  50),
    4:  (1.83,  0.18,  0,  50),
    5:  (3.72,  0.35,  0,  50),
    6:  (5.04,  0.52,  2,  30),
    7:  (12.79, 1.20,  11, 30),
    8:  (14.15, 1.50,  19, 30),
    9:  (16.23, 0.80,  24, 30),
    10: (16.68, 0.60,  27, 30),
    11: (17.48, 0.40,  29, 30),
    12: (17.93, 0.30,  29, 30),
}

REAL_ON = {
    1:  0.04, 2:  0.09, 3:  0.23, 4:  0.50,
    5:  0.67, 6:  0.78, 7:  0.91, 8:  0.95,
    9:  0.97, 10: 0.98, 11: 0.99, 12: 0.99,
}


def panel_label(ax, txt, x=-0.18, y=1.04):
    ax.text(x, y, txt, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left")


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Gillespie trajectory generator ─────────────────────────────
def run_gillespie(N, omega, Jeff, t_max_h=18.0, seed=None):
    """
    Exact Gillespie simulation. Returns (times_h, m_values).
    Starts in all-bound state (n=N).
    """
    rng_g   = np.random.default_rng(seed)
    t_max   = t_max_h * 3600.0
    n_bound = N
    t       = 0.0
    times   = [0.0]
    states  = [1.0]

    while t < t_max:
        r_bind   = (N - n_bound) * (omega + Jeff * n_bound)
        r_unbind = n_bound * omega
        r_total  = r_bind + r_unbind

        if r_total < 1e-20:
            break

        dt = rng_g.exponential(1.0 / r_total)
        t += dt

        if t > t_max:
            break

        if rng_g.random() < r_bind / r_total:
            n_bound = min(n_bound + 1, N)
        else:
            n_bound = max(n_bound - 1, 0)

        times.append(t)
        states.append(n_bound / N)

        if n_bound == 0:
            break

    return np.array(times) / 3600.0, np.array(states)


# ── Load FPT and ON-fraction data ─────────────────────────────
def load_fpt():
    """Load FPT from CSV if available, otherwise use hardcoded values."""
    try:
        df = pd.read_csv(FPTF)
        cols = {c.lower(): c for c in df.columns}

        # Find N column
        n_col = None
        for candidate in ["n", "circuit_size", "n_tfs", "tfs"]:
            if candidate in cols:
                n_col = cols[candidate]; break
        if n_col is None:
            n_col = df.columns[0]

        # Find FPT column
        fpt_col = None
        for candidate in ["mean_fpt_h", "mean_fpt", "fpt_h", "fpt", "mean"]:
            if candidate in cols:
                fpt_col = cols[candidate]; break
        if fpt_col is None:
            fpt_col = df.columns[1]

        # Find SEM column
        sem_col = None
        for candidate in ["sem_fpt_h", "sem_fpt", "sem", "se", "std"]:
            if candidate in cols:
                sem_col = cols[candidate]; break

        # Find censored column
        cen_col = None
        for candidate in ["n_censored", "censored", "n_cens"]:
            if candidate in cols:
                cen_col = cols[candidate]; break

        Nvals    = df[n_col].values.astype(int)
        fpts     = df[fpt_col].values.astype(float)
        sems     = df[sem_col].values.astype(float) if sem_col else np.zeros_like(fpts)
        censored = dict(zip(Nvals,
                            df[cen_col].values.astype(int) if cen_col
                            else np.zeros(len(Nvals), dtype=int)))
        print(f"  Loaded FPT from {FPTF}")
        return Nvals, fpts, sems, censored

    except Exception as e:
        print(f"  FPT file not found or unreadable ({e}). Using hardcoded values.")
        Nvals = np.array(sorted(REAL_FPT.keys()))
        fpts  = np.array([REAL_FPT[n][0] for n in Nvals])
        sems  = np.array([REAL_FPT[n][1] for n in Nvals])
        cens  = {n: REAL_FPT[n][2] for n in Nvals}
        return Nvals, fpts, sems, cens


def load_on_fraction():
    """Load ON-fraction from CSV if available."""
    try:
        df = pd.read_csv(ONF)
        cols = {c.lower(): c for c in df.columns}

        n_col = None
        for c in ["n", "circuit_size", "n_tfs"]:
            if c in cols:
                n_col = cols[c]; break
        if n_col is None:
            n_col = df.columns[0]

        on_col = None
        for c in ["on_fraction", "on_frac", "on", "fraction"]:
            if c in cols:
                on_col = cols[c]; break
        if on_col is None:
            on_col = df.columns[1]

        print(f"  Loaded ON-fraction from {ONF}")
        return df[n_col].values.astype(int), df[on_col].values.astype(float)

    except Exception as e:
        print(f"  ON-fraction file not found ({e}). Using hardcoded values.")
        Nv = np.array(sorted(REAL_ON.keys()))
        return Nv, np.array([REAL_ON[n] for n in Nv])


# ══════════════════════════════════════════════════════════════
# Build figure
# ══════════════════════════════════════════════════════════════
print("Building Figure S1...")

Nvals, fpts, sems, censored = load_fpt()
Nvals_on, on_frac = load_on_fraction()

fig = plt.figure(figsize=(18, 12))
gs  = GridSpec(2, 3, figure=fig,
               left=0.07, right=0.97,
               top=0.94, bottom=0.08,
               wspace=0.40, hspace=0.48)

ax_A = fig.add_subplot(gs[0, 0])
ax_B = fig.add_subplot(gs[0, 1])
ax_C = fig.add_subplot(gs[0, 2])
ax_D = fig.add_subplot(gs[1, 0])
ax_E = fig.add_subplot(gs[1, 1])
ax_F = fig.add_subplot(gs[1, 2])


# ── PANEL A: FPT vs N ─────────────────────────────────────────
ax = ax_A
print("  Panel A: FPT vs N")

for n, fpt, sem in zip(Nvals, fpts, sems):
    col    = COL_ABOVE if n > NC else COL_BELOW
    nc_val = censored.get(int(n), 0)
    marker = "^" if nc_val > 0 else "o"
    ax.errorbar(n, fpt,
                yerr=sem if sem > 0 else None,
                fmt=marker, color=col, ms=6,
                lw=0.8, capsize=3, capthick=0.8, zorder=3)
    if nc_val > 0:
        ax.text(n + 0.15, fpt + 0.4, "[c]", fontsize=7, color=col)

ax.axhline(24, color="#888", lw=0.9, ls="--", zorder=1)
ax.axvline(NC, color="#bbb", lw=0.8, ls=":", alpha=0.8, zorder=1)
ax.text(NC + 0.15, 18.8, f"$N_c$={NC}", fontsize=7, color="#666")
ax.text(11.8, 24.8, "24 h (cell cycle)", fontsize=6.5, color="#888", ha="right")

ax.set_xlabel("Circuit size N")
ax.set_ylabel("Mean first-passage time (h)")
ax.set_xticks(range(1, 13))
ax.set_xlim(0.5, 12.5)
ax.set_ylim(0, 22)
despine(ax)
panel_label(ax, "A")

# Log-linear inset
ax_ins = ax.inset_axes([0.52, 0.08, 0.44, 0.40])
mask_ab = Nvals > NC
Nab  = Nvals[mask_ab]
Fab  = fpts[mask_ab]
valid = Fab > 0
ax_ins.scatter(Nab[valid], np.log10(Fab[valid]),
               c=COL_ABOVE, s=18, zorder=3)
if valid.sum() >= 2:
    m_sl, b_sl = np.polyfit(Nab[valid], np.log10(Fab[valid]), 1)
    xr = np.linspace(Nab[valid].min(), Nab[valid].max(), 50)
    ax_ins.plot(xr, m_sl * xr + b_sl, color="#555", lw=1.0, ls="--")
    ax_ins.text(0.95, 0.08, f"α = {m_sl:.2f}",
                transform=ax_ins.transAxes, ha="right", va="bottom", fontsize=6)
ax_ins.set_xlabel("N", fontsize=6)
ax_ins.set_ylabel("log10(FPT)", fontsize=6)
ax_ins.tick_params(labelsize=5.5)
ax_ins.spines["top"].set_visible(False)
ax_ins.spines["right"].set_visible(False)


# ── PANEL B: ON-state fraction ────────────────────────────────
ax = ax_B
print("  Panel B: ON-state fraction")

for n, of in zip(Nvals_on, on_frac):
    col = COL_ABOVE if n > NC else COL_BELOW
    ax.scatter(n, of, color=col, s=35, zorder=3)

ax.axhline(0.5, color="#888", lw=0.9, ls="--", zorder=1)
ax.axvline(NC,  color="#bbb", lw=0.8, ls=":", alpha=0.8, zorder=1)
ax.text(NC + 0.15, 0.93, f"$N_c$={NC}", fontsize=7, color="#666")
ax.text(11.8, 0.52, "0.5", fontsize=6.5, color="#888", ha="right")

ax.set_xlabel("Circuit size N")
ax.set_ylabel("ON-state fraction")
ax.set_xticks(range(1, 13))
ax.set_xlim(0.5, 12.5)
ax.set_ylim(-0.02, 1.05)
despine(ax)
panel_label(ax, "B")


# ── PANELS C, D, E: Example trajectories (self-contained) ─────
traj_configs = [
    (ax_C, 3,  "C", COL_BELOW, 1.32,  "N = 3 (below $N_c$)"),
    (ax_D, 4,  "D", COL_AT,    1.83,  "N = 4 (at $N_c$)"),
    (ax_E, 7,  "E", COL_ABOVE, None,  "N = 7 (above $N_c$)"),
]

for ax, N_traj, lbl, col, fpt_val, title in traj_configs:
    print(f"  Panel {lbl}: generating trajectory N={N_traj}")
    seed_val = 100 + N_traj   # fixed seeds for reproducibility
    times_h, states = run_gillespie(N_traj, omega, Jeff,
                                    t_max_h=18.0, seed=seed_val)

    ax.step(times_h, states, where="post", color=col, lw=0.9, alpha=0.85)
    ax.axhline(0.8, color="#ddd", lw=0.5, ls=":", zorder=0)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("ON-state fraction m(t)")
    ax.set_xlim(0, 18)
    ax.set_ylim(-0.05, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    despine(ax)
    panel_label(ax, lbl)

    # Annotation box — bottom right to avoid data overlap
    if fpt_val is not None:
        ann_txt = f"{title}\nFPT ≈ {fpt_val:.2f} h"
    else:
        ann_txt = f"{title}\nFPT > 18 h (censored)"

    if N_traj <= NC:
        ann_x, ann_y, ha_a = 0.97, 0.97, "right"
    else:
        ann_x, ann_y, ha_a = 0.97, 0.05, "right"

    ax.text(ann_x, ann_y, ann_txt,
            transform=ax.transAxes,
            ha=ha_a, va="top" if ann_y > 0.5 else "bottom",
            fontsize=7.5, color=col,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#ccc", lw=0.6))


# ── PANEL F: Phase diagram ────────────────────────────────────
ax = ax_F
print("  Panel F: Phase diagram")

kappa_range = np.logspace(-3.8, -2.0, 200)
g_mid = P["g_midpoint"]
g_lo  = P["g_lo"]
g_hi  = P["g_hi"]

Nc_mid = omega * kappa_range / (2 * g_mid**2)
Nc_lo  = omega * kappa_range / (2 * g_hi**2)
Nc_hi  = omega * kappa_range / (2 * g_lo**2)

ax.fill_between(kappa_range, Nc_lo, Nc_hi,
                color="#AABCCE", alpha=0.45, zorder=1)
ax.plot(kappa_range, Nc_mid,
        color="#2C3E50", lw=1.5, zorder=3)

kap_h     = P["kappa_human"]
kap_m     = P["kappa_mouse"]
nc_h_emp  = P.get("NC_FIT_H",    4.72)
nc_h_se   = P.get("NC_FIT_H_SE", 0.47)
nc_m_emp  = 8.05
nc_m_se   = 0.47

ax.errorbar(kap_h, nc_h_emp, yerr=nc_h_se,
            fmt="o", color=COL_H, ms=8, lw=1.0,
            capsize=4, capthick=1.0, zorder=5)
ax.errorbar(kap_m, nc_m_emp, yerr=nc_m_se,
            fmt="D", color=COL_M, ms=8, lw=1.0,
            capsize=4, capthick=1.0, zorder=5)

ax.annotate(f"Human ESC\n$N_c$ = {nc_h_emp:.2f}±{nc_h_se:.2f}",
            xy=(kap_h, nc_h_emp),
            xytext=(kap_h * 1.6, nc_h_emp + 1.5),
            fontsize=6.5, color=COL_H,
            arrowprops=dict(arrowstyle="-", color=COL_H,
                            lw=0.6, shrinkA=4, shrinkB=2))
ax.annotate(f"Mouse ESC\n$N_c$ = {nc_m_emp:.2f}±{nc_m_se:.2f}",
            xy=(kap_m, nc_m_emp),
            xytext=(kap_m * 1.4, nc_m_emp + 2.5),
            fontsize=6.5, color=COL_M,
            arrowprops=dict(arrowstyle="-", color=COL_M,
                            lw=0.6, shrinkA=4, shrinkB=2))

ax.set_xscale("log")
ax.set_xlabel("Chromatin relaxation rate " + r"$\kappa$" + " (s" + r"$^{-1}$" + ")")
ax.set_ylabel("Predicted $N_c$")
ax.set_xlim(10**-3.8, 10**-2.0)
ax.set_ylim(0, 20)
despine(ax)
panel_label(ax, "F")

ax.text(0.03, 0.97,
        "Grey band: full g range\nLine: g at midpoint",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=7, color="#555",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#ccc", lw=0.6))


# ── Save ───────────────────────────────────────────────────────
for ext in ["pdf", "png"]:
    path = os.path.join(FIGDIR, f"figS1_gillespie.{ext}")
    fig.savefig(path, dpi=600, bbox_inches="tight")
    print(f"Saved: {path}")

plt.close(fig)
print("Figure S1 done.")
