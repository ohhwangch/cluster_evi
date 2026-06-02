#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rejection_simulation.py
=======================
Empirical size / power of the two-sample group-EVI t-test (Section S.2.6). DGP: Same as in main_simulation.py, independent.

Output: {output_dir}/t_test_G{G}_size{groupsize}.csv  with columns
        Group, T, Effect Size, Power.   (No plotting — results only.)

Example
-------
python rejection_simulation.py --G 3 --groupsize 100 --nsim 1000 --T 1000 3000
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from common import fHill, datingtrimming, run_seeds, fDataGenerating_independent


# =============================================================================
# Test  (the DGP is common's independent Burr+t, shared with main_simulation)
# =============================================================================


def tTest(G, index_p1, index_p2, vgamma_g_p1, vgamma_g_p2, tk, sig_level):
    """Two-sample group-EVI test under the i.i.d. cross-sectional variance."""
    abs_test_stats = []
    for i in range(G):
        var_p1 = vgamma_g_p1[i] ** 2 / (tk * (index_p1[i + 1] - index_p1[i]))
        var_p2 = vgamma_g_p2[i] ** 2 / (tk * (index_p2[i + 1] - index_p2[i]))
        stat   = (vgamma_g_p1[i] - vgamma_g_p2[i]) / np.sqrt(var_p1 + var_p2)
        abs_test_stats.append(np.abs(stat))
    critical_value = norm.ppf(1 - sig_level / 2)
    result = np.array(abs_test_stats) > critical_value
    return abs_test_stats, result


def get_loc_and_estimates(k, tk, mSample, nObs, G):
    """Stage-1 ordering EVI -> structural-break grouping -> stage-2 group EVIs."""
    mSample = mSample.copy()
    mSample["gamma"] = mSample.iloc[:, 1:].apply(
        lambda row: fHill(row.values[~pd.isna(row.values)], k), axis=1)
    mSample = mSample.sort_values(by="gamma", ascending=True)
    vorderedgamma = mSample["gamma"].values
    _, loc, _ = datingtrimming(
        np.array(vorderedgamma).reshape(-1, 1),
        np.ones((nObs, 1)), 1, G - 1, 1, nObs, 0.15)
    vl    = loc[:, -1]
    index = np.array([0] + list(vl) + [len(mSample)]).astype(int)
    

    df_copy = mSample.copy()
    df_copy["gamma"] = df_copy.iloc[:, 1:-1].apply(
        lambda row: fHill(row.values[~pd.isna(row.values)], tk), axis=1)
    vgamma_ind = df_copy["gamma"].values
    vgamma_g = np.array([
        vgamma_ind[index[i]:index[i + 1]].mean() for i in range(len(index) - 1)
    ])
    return index, vgamma_g


# =============================================================================
# One (T, g, Δ) cell  +  the full sweep
# =============================================================================

def simulate_power(T, g, delta, G, groupsize, iNsim, rk, r_tk, sig_level):
    """Empirical rejection rate for shifting group g's EVI by `delta` in period 2."""
    test_results = []
    for i in range(iNsim):
        rng = np.random.default_rng(i)

        vG   = np.repeat(groupsize, G)
        vkb  = np.repeat(1, G)
        nObs = int(np.sum(vG))
        k    = int(rk * T)
        tk   = int(r_tk * T)

        vgamma_p1 = np.linspace(0.2, 1.5, G)
        vgamma_p2 = vgamma_p1.copy()
        vgamma_p2[g] += delta

        dfs_p1 = 1 / vgamma_p1
        dfs_p2 = 1 / vgamma_p2

        mSample1 = fDataGenerating_independent(dfs_p1, vG, vkb, T, 0.5, rng)
        mSample2 = fDataGenerating_independent(dfs_p2, vG, vkb, T, 0.5, rng)

        index_p1, vgamma_g_p1 = get_loc_and_estimates(k, tk, mSample1, nObs, G)
        index_p2, vgamma_g_p2 = get_loc_and_estimates(k, tk, mSample2, nObs, G)

        _, result = tTest(G, index_p1, index_p2,
                          vgamma_g_p1, vgamma_g_p2, tk, sig_level)
        test_results.append(result)

    powers_per_group = np.array(test_results).mean(axis=0)
    return {"Group": g + 1, "T": T, "Effect Size": delta,
            "Power": powers_per_group[g]}


def run_rejection_simulation(G=3, groupsize=100, iNsim=1000,
                             vT=(1000, 3000), rk=0.12, r_tk=0.03,
                             sig_level=0.05, effect_sizes=None,
                             output_dir=None, n_jobs=-1, verbose=1):
    """
    Sweep (T, group, Δγ), compute the per-group rejection rate, and save one CSV.

    Returns the results DataFrame (also written to
    {output_dir}/t_test_G{G}_size{groupsize}.csv).
    """
    if effect_sizes is None:
        effect_sizes = np.arange(-0.08, 0.0801, 0.005).round(5).tolist()
    if output_dir is None:
        output_dir = f"Rejection_G{G}_size{groupsize}"
    os.makedirs(output_dir, exist_ok=True)

    tasks = [(T, g, delta) for T in vT for g in range(G) for delta in effect_sizes]
    results = run_seeds(
        simulate_power,
        [(T, g, delta, G, groupsize, iNsim, rk, r_tk, sig_level)
         for (T, g, delta) in tasks],
        n_jobs=n_jobs,
    )
    df_power = pd.DataFrame(results)

    out_csv = os.path.join(output_dir, f"t_test_G{G}_size{groupsize}.csv")
    df_power.to_csv(out_csv, index=False)
    if verbose:
        print(f"Saved -> {out_csv}  ({len(df_power)} rows)")
    return df_power


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Empirical size/power of the two-sample group-EVI t-test "
                    "(independent Burr+t DGP).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--G",         type=int,   default=3,   help="number of groups")
    p.add_argument("--groupsize", type=int,   default=100, help="units per group")
    p.add_argument("--nsim",      type=int,   default=1000,
                   help="Monte Carlo replications per cell")
    p.add_argument("--T",         type=int,   nargs="+", default=[1000, 3000])
    p.add_argument("--rk",        type=float, default=0.12,
                   help="stage-1 sample fraction, k = rk*T")
    p.add_argument("--r_tk",      type=float, default=0.03,
                   help="stage-2 sample fraction, k-tilde = r_tk*T")
    p.add_argument("--sig",       type=float, default=0.05,
                   help="test significance level")
    p.add_argument("--effects",   type=float, nargs="+", default=None,
                   help="effect sizes Δγ (default: -0.08..0.08 step 0.01)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="default: Rejection_G{G}_size{groupsize}")
    p.add_argument("--n_jobs",    type=int,   default=-1,
                   help="-1 = all cores, 1 = sequential/deterministic")
    p.add_argument("--verbose",   type=int,   default=1)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    run_rejection_simulation(
        G            = args.G,
        groupsize    = args.groupsize,
        iNsim        = args.nsim,
        vT           = tuple(args.T),
        rk           = args.rk,
        r_tk         = args.r_tk,
        sig_level    = args.sig,
        effect_sizes = args.effects,
        output_dir   = args.output_dir,
        n_jobs       = args.n_jobs,
        verbose      = args.verbose,
    )
    total = time.time() - t0
    print(f"\nTotal wall time: {total / 3600:.2f} h ({total / 60:.1f} min)")


if __name__ == "__main__":
    main()
