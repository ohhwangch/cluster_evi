#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_simulation.py
==================
Monte-Carlo simulation for Clustering extreme value indices in large panels.

  * main text: the main simulation,
  * supplement:  the perturbed-DGP simulation (uniform individual noise), and the single-group (G = 1) case.

The behaviour is selected entirely through the command line:
  * `--dgp` picks one of the four DGPs
        independent | dependent | independent_noise | dependent_noise
  * `--G` sets the number of groups (accepts several, e.g. `--G 3 5`, which
    are swept); the true tail indices are
        vgamma_0 = linspace(--gamma_start, --gamma_end, G).
    For G = 1 this is just a single value (= --gamma_start), giving the
    one-group case.

Example runs
------------
# Main simulation
python main_simulation.py  --nsim 1000 --T 1000 --output_dir sim_results_main

# S.2.1 
python main_simulation.py  --nsim 1000 --T 3000 --output_dir sim_results_supp

# S.2.4 stress test
python main_simulation.py  --nsim 1000 --T 1000 --gamma_start 0.48 --gamma_end 0.64 --output_dir sim_results_small_gap

# S.2.5 perturbed setting & single group
python main_simulation.py --dgp independent_noise  dependent_noise  --nsim 1000 --output_dir sim_results_perturbed

python main_simulation.py \
    --dgp independent_noise dependent_noise \
    --G 1 \
    --gamma_single 0.2 0.5 0.7 1.0 \
    --output_dir sim_results_G1_noise

# Quick smoke-test
python main_simulation.py --n_jobs 1 --nsim 10 --T 1000 --groupsize 100 \
    --G 3 --dgp independent --k1_rate 0.12 --r_H 0.05

Outputs (written to --output_dir)
----------------------------------
elbow_accuracy.csv      elbow correct-selection rate
trueG_accuracy.csv      per-group classification accuracy, true-G breaks
Ghat_accuracy.csv       per-group classification accuracy, Ghat breaks
measure_sums.csv         raw per-unit error sums (s1, s2, s3) per group

measure_sums.csv stores cumulative sums over the M = Nsim seeds, so the plotting layer can derive any measure per unit i, e.g.
    Bias_i  = s1 / M,  MSE_i = s2 / M,  MAE_i = s3 / M,
    RMSE_i  = sqrt(s2 / M),  Std_i = sqrt(s2/M - (s1/M)^2).
"""

import argparse
import time
from itertools import product

import numpy as np
import pandas as pd

from common import (
    FIXED_THRED, M_MAX, ELBOW_THRESHOLDS, DEP_RHO, ALPHA,
    is_noise_dgp, is_dependent_dgp,
    fDataGenerating, fDataGenerating_disturbed,
    add_gamma_estimates,
    datingtrimming, ssrnul, elbow_ghat, extract_index,
    group_accuracy, run_seeds, save_dataframe,
    _cauchy_scale_matrix,
)

# =============================================================================
# One replication: DGP -> Hill -> sort -> datingtrimming (once)
# =============================================================================

def core_one_sim(seed, T, k1, vgamma_0, groupsize, G,
                 dgp_model="independent", dep_rho=DEP_RHO, alpha=0.5,
                 L_chol=None, disturb_a=0.05):
    """
    Generate one panel, Hill-estimate every unit, sort ascending by γ̂ and run
    the structural-break search once.

    For the '*_noise' DGPs each unit i in group g draws its own tail index
    γ̃_i = γ_g + ε_i with ε_i ~ Uniform(-disturb_a, disturb_a).  

    Returns (mSample, datevec, ssr_path, N), with mSample sorted ascending by γ̂
    and carrying an 'index' column of original (pre-sort) row positions.
    """
    rng  = np.random.default_rng(seed)
    vG   = np.repeat(groupsize, G)
    vkb  = np.repeat(1, G)

    if is_noise_dgp(dgp_model):
        if disturb_a < 0:
            raise ValueError(f"disturb_a must be non-negative, got {disturb_a}.")
        gamma_group  = np.repeat(vgamma_0, groupsize)
        eps          = rng.uniform(-disturb_a, disturb_a, len(gamma_group))
        gamma_tilde  = gamma_group + eps
        dfs_per_unit = 1.0 / gamma_tilde
        mSample      = fDataGenerating_disturbed(
            dfs_per_unit, vG, vkb, T, alpha, rng,
            dependent=is_dependent_dgp(dgp_model),
            dep_rho=dep_rho, L_chol=L_chol)
        mSample["G_0"]     = np.repeat(np.arange(1, G + 1), groupsize)
        mSample["gamma_0"] = gamma_tilde
    else:
        dfs     = 1.0 / vgamma_0
        mSample = fDataGenerating(dfs, vG, vkb, T, alpha, rng,
                                  dgp_model=dgp_model, dep_rho=dep_rho,
                                  L_chol=L_chol)
        mSample["G_0"]     = np.repeat(np.arange(1, G + 1), groupsize)
        mSample["gamma_0"] = np.repeat(vgamma_0, vG)

    mSample = add_gamma_estimates(mSample, k1, col_name="gamma")
    mSample = (mSample
               .sort_values(by="gamma", ascending=True)
               .reset_index(drop=False))   # 'index' = original row position
    N = len(mSample)

    y = mSample["gamma"].to_numpy().reshape(-1, 1)
    z = np.ones((N, 1))

    glb, datevec, _ = datingtrimming(y, z, 1, M_MAX + 1, 1, N, 0.15)
    ssr0     = ssrnul(y, z)
    ssr_path = [ssr0] + glb.flatten().tolist()

    return mSample, datevec, ssr_path, N


# =============================================================================
# Per-seed worker (module level so joblib can pickle it)
# =============================================================================

def one_seed_work(seed, T, k1, vk2, k_H, vgamma_0, groupsize, G,
                  thred_values, dgp_model, dep_rho, alpha, L_chol, disturb_a):
    """
    One replication -> elbow-selection and groupwise-accuracy quantities, plus
    the per-unit measure sums.
    
    Returns a dict with:
        correct_elbow   {thred: 0/1}        elbow == G ?
        acc_trueG_g  (G,)                per-group accuracy, true-G breaks
        acc_Ghat_g   (G,)                per-group accuracy, Ghat breaks
        ghat_eq_g    int                 Ghat == G ?
        measure      {k2: sub-dict}      per-unit running error sums
    """
    mSample, datevec, ssr_path, N = core_one_sim(
        seed, T, k1, vgamma_0, groupsize, G,
        dgp_model=dgp_model, dep_rho=dep_rho, alpha=alpha,
        L_chol=L_chol, disturb_a=disturb_a,
    )

    Ghat      = elbow_ghat(ssr_path, FIXED_THRED)
    ghat_eq_g = int(Ghat == G)
    idx_trueG = extract_index(datevec, G,    N)
    idx_Ghat  = extract_index(datevec, Ghat, N)

    # ── Elbow selection ───────────────────────────────────────────────────────
    correct_elbow = {thred: int(elbow_ghat(ssr_path, thred) == G)
                  for thred in thred_values}

    # ── Groupwise classification accuracy ─────────────────────────────────────
    pos = np.arange(N)

    g_hat_trueG = np.digitize(pos, idx_trueG, right=False)
    ms_tG       = mSample.assign(g_hat=g_hat_trueG).sort_values("index")
    acc_trueG_g = group_accuracy(ms_tG["g_hat"].to_numpy(),
                                 ms_tG["G_0"].to_numpy(), G)

    g_hat_Ghat = np.digitize(pos, idx_Ghat, right=False)
    ms_Gh      = mSample.assign(g_hat=g_hat_Ghat).sort_values("index")
    acc_Ghat_g = group_accuracy(ms_Gh["g_hat"].to_numpy(),
                                ms_Gh["G_0"].to_numpy(), G)

    # ── Individual Hill benchmark (per-unit errors feed the measure sums) ─────
    df_kH      = add_gamma_estimates(mSample, k_H, col_name="gamma_kH")
    df_kH_orig = df_kH.sort_values("index").reset_index(drop=True)
    err_kH     = (df_kH_orig["gamma_kH"].to_numpy()
                  - df_kH_orig["gamma_0"].to_numpy())
    mse_all    = err_kH ** 2
    mae_all    = np.abs(err_kH)

    # ── Group estimator: per-unit errors for trueG and Ghat partitions ────────
    measure = {}
    for k2 in vk2:
        k2i        = int(k2)
        df_k2      = add_gamma_estimates(mSample, k2i, col_name="gamma_k2")
        gk2_sorted = df_k2["gamma_k2"].to_numpy()

        grp_trueG  = np.array([gk2_sorted[idx_trueG[i]:idx_trueG[i + 1]].mean()
                               for i in range(G)])
        unit_trueG = np.repeat(grp_trueG, np.diff(idx_trueG))

        grp_Ghat   = np.array([gk2_sorted[idx_Ghat[i]:idx_Ghat[i + 1]].mean()
                               for i in range(Ghat)])
        unit_Ghat  = np.repeat(grp_Ghat, np.diff(idx_Ghat))

        df_k2               = df_k2.copy()
        df_k2["unit_trueG"] = unit_trueG
        df_k2["unit_Ghat"]  = unit_Ghat
        df_orig             = df_k2.sort_values("index").reset_index(drop=True)

        gamma_0 = df_orig["gamma_0"].to_numpy()
        err_tG  = df_orig["unit_trueG"].to_numpy() - gamma_0
        err_Gh  = df_orig["unit_Ghat"].to_numpy()  - gamma_0

        measure[k2i] = dict(
            # running per-unit error sums (accumulated across seeds)
            s1_g  = err_tG, s2_g  = err_tG ** 2, s3_g  = np.abs(err_tG),
            s1_gh = err_Gh, s2_gh = err_Gh ** 2, s3_gh = np.abs(err_Gh),
            s1_ind = err_kH, s2_ind = mse_all, s3_ind = mae_all,
        )

    return dict(
        correct_elbow  = correct_elbow,
        acc_trueG_g = acc_trueG_g,
        acc_Ghat_g  = acc_Ghat_g,
        ghat_eq_g   = ghat_eq_g,
        measure     = measure,
    )


# =============================================================================
# Master loop over scenarios
# =============================================================================

def run_all_simulations(
    vT           = (1000, 3000),
    vgroupsize   = (100, 300),
    G            = (3, 5),
    gamma_start  = 0.2,
    gamma_end    = 1.5,
    iNsim        = 1000,
    k1_rates     = (0.09, 0.12),
    r_kt         = (0.02, 0.03, 0.04),
    r_Hs         = (0.05, 0.07),
    thred_values = ELBOW_THRESHOLDS,
    dgp_models   = ("independent", "dependent"),
    dep_rho      = DEP_RHO,
    alpha        = 0.5,
    disturb_a    = 0.05,
    n_jobs       = -1,
    output_dir   = "results_main",
    verbose      = 1,
    save         = True,
):
    """
    Sweep over (G, dgp, groupsize, T, k1_rate, r_H); parallelise the seeds
    within each scenario. For each G the tail-index grid is
    vgamma_0 = linspace(gamma_start, gamma_end, G).
    """
    r_kt     = np.asarray(r_kt, dtype=float)
    r_Hs     = list(r_Hs)
    G_list   = [G] if np.isscalar(G) else list(G)

    records_elbow, records_groupwise_trueG, records_groupwise_Ghat = [], [], []
    records_measure = []

    scenarios = list(product(G_list, dgp_models, vgroupsize, vT, k1_rates, r_Hs))
    n_scen    = len(scenarios)

    for s_idx, (G, dgp_model, groupsize, T, k1_rate, r_H) in enumerate(scenarios, 1):

        vgamma_0    = np.linspace(gamma_start, gamma_end, G)
        t_scen_start = time.time()
        k1          = int(k1_rate * T)
        vk2         = (r_kt * T).astype(int)
        k_H         = int(r_H * T)
        N_units     = groupsize * G
        is_dep      = is_dependent_dgp(dgp_model)
        dep_rho_val = dep_rho if is_dep else float("nan")

        # Cholesky factor once per dependent scenario (skip O(N^3) per seed)
        if is_dep:
            L_chol = np.linalg.cholesky(_cauchy_scale_matrix(N_units, dep_rho))
        else:
            L_chol = None

        if verbose:
            print(f"[{s_idx:>3}/{n_scen}] dgp={dgp_model:18s} gs={groupsize:>3} "
                  f"G={G} T={T:>5} k1_rate={k1_rate} r_H={r_H}  N={N_units} ...",
                  flush=True)

        arg_tuples = [
            (seed, T, k1, vk2, k_H, vgamma_0, groupsize, G,
             thred_values, dgp_model, dep_rho, alpha, L_chol, disturb_a)
            for seed in range(1, iNsim + 1)
        ]
        seed_results = run_seeds(one_seed_work, arg_tuples, n_jobs=n_jobs)

        # ── Accumulators (summed over seeds, divided by iNsim later) ──────────
        correct_elbow    = {thred: 0 for thred in thred_values}
        acc_trueG_g   = np.zeros(G)
        acc_Ghat_g    = np.zeros(G)
        cnt_Ghat_eq_G = 0
        acc_trueG_measure  = {int(k2): dict(
            s1_g=np.zeros(N_units), s2_g=np.zeros(N_units), s3_g=np.zeros(N_units),
            s1_ind=np.zeros(N_units), s2_ind=np.zeros(N_units),
            s3_ind=np.zeros(N_units),
        ) for k2 in vk2}
        acc_Ghat_measure   = {int(k2): dict(
            s1_gh=np.zeros(N_units), s2_gh=np.zeros(N_units),
            s3_gh=np.zeros(N_units),
        ) for k2 in vk2}

        for res in seed_results:
            for thred in thred_values:
                correct_elbow[thred] += res["correct_elbow"][thred]
            acc_trueG_g   += res["acc_trueG_g"]
            acc_Ghat_g    += res["acc_Ghat_g"]
            cnt_Ghat_eq_G += res["ghat_eq_g"]
            for k2 in vk2:
                k2i = int(k2)
                for key in acc_trueG_measure[k2i]:
                    acc_trueG_measure[k2i][key] += res["measure"][k2i][key]
                for key in acc_Ghat_measure[k2i]:
                    acc_Ghat_measure[k2i][key] += res["measure"][k2i][key]

        # ── Elbow selection ───────────────────────────────────────────────────
        elbow_row = dict(dgp_model=dgp_model, dep_rho=dep_rho_val, T=T, G=G,
                      groupsize=groupsize, k1=k1, k1_rate=k1_rate, Nsim=iNsim)
        for thred in thred_values:
            elbow_row[f"correct_rate({round(thred, 5)})"] = correct_elbow[thred] / iNsim
        records_elbow.append(elbow_row)

        # ── Groupwise classification accuracy ──────────────────────────────────
        avg_acc_tG = acc_trueG_g / iNsim
        records_groupwise_trueG.append(dict(
            dgp_model=dgp_model, dep_rho=dep_rho_val, G=G, T=T,
            k1_rate=k1_rate, k1=k1, groupsize=groupsize, FIXED_THRED=FIXED_THRED,
            freq_Ghat_eq_G=cnt_Ghat_eq_G / iNsim,
            accuracy=float(avg_acc_tG.mean()), acc_g=avg_acc_tG,
            Nsim=iNsim, alpha=alpha,
        ))
        avg_acc_Gh = acc_Ghat_g / iNsim
        records_groupwise_Ghat.append(dict(
            dgp_model=dgp_model, dep_rho=dep_rho_val, G=G, T=T,
            k1_rate=k1_rate, k1=k1, groupsize=groupsize, FIXED_THRED=FIXED_THRED,
            freq_Ghat_eq_G=cnt_Ghat_eq_G / iNsim,
            accuracy=float(avg_acc_Gh.mean()), acc_g=avg_acc_Gh,
            Nsim=iNsim, alpha=alpha,
        ))

        # ── Measure: per-unit error sums ──
        def slice_by_group(arr):
            return [arr[j * groupsize:(j + 1) * groupsize] for j in range(G)]

        for k2_rate, k2 in zip(r_kt, vk2):
            k2i  = int(k2)

            base = dict(dgp_model=dgp_model, dep_rho=dep_rho_val,
                        r_kt=k2_rate, k2=k2i, T=T, G=G, group_size=groupsize,
                        Nsim=iNsim, k1=k1, k_H=k_H, k1_rate=k1_rate, r_H=r_H,
                        FIXED_THRED=FIXED_THRED,
                        freq_Ghat_eq_G=cnt_Ghat_eq_G / iNsim)

            records_measure.append({
                **base,
                "s1_g_byunit":   slice_by_group(acc_trueG_measure[k2i]["s1_g"]),
                "s2_g_byunit":   slice_by_group(acc_trueG_measure[k2i]["s2_g"]),
                "s3_g_byunit":   slice_by_group(acc_trueG_measure[k2i]["s3_g"]),
                "s1_gh_byunit":  slice_by_group(acc_Ghat_measure[k2i]["s1_gh"]),
                "s2_gh_byunit":  slice_by_group(acc_Ghat_measure[k2i]["s2_gh"]),
                "s3_gh_byunit":  slice_by_group(acc_Ghat_measure[k2i]["s3_gh"]),
                "s1_ind_byunit": slice_by_group(acc_trueG_measure[k2i]["s1_ind"]),
                "s2_ind_byunit": slice_by_group(acc_trueG_measure[k2i]["s2_ind"]),
                "s3_ind_byunit": slice_by_group(acc_trueG_measure[k2i]["s3_ind"]),
            })

        if verbose:
            print(f"         done in {time.time() - t_scen_start:.1f}s", flush=True)

    # ── Collect ─────────────────────────────────────────────────────────────────
    frames = {
        "elbow_accuracy.csv": pd.DataFrame(records_elbow),
        "trueG_accuracy.csv": pd.DataFrame(records_groupwise_trueG),
        "Ghat_accuracy.csv":  pd.DataFrame(records_groupwise_Ghat),
        "measure_sums.csv":          pd.DataFrame(records_measure),
    }

    if save:
        for fname, frame in frames.items():
            save_dataframe(frame, fname, output_dir)
        return output_dir
    return frames


# =============================================================================
# Gamma sweep: one combined CSV set tagged by gamma 
# =============================================================================

def run_gamma_sweep(gamma_values, G=1, output_dir="results_gamma_sweep",
                    save=True, verbose=1, **kwargs):
    """
    Run `run_all_simulations` once per γ in `gamma_values` and concatenate the
    results into one CSV set, with a `gamma_single` column identifying the γ of each row.

    For each γ the tail-index grid is the single value γ (gamma_start =
    gamma_end = γ), so this is meant for the G = 1 case. Extra keyword args are forwarded to
    `run_all_simulations` (dgp_models, vT, vgroupsize, iNsim, disturb_a, ...).
    """
    combined = {fname: [] for fname in (
        "elbow_accuracy.csv", "trueG_accuracy.csv",
        "Ghat_accuracy.csv",  "measure_sums.csv",
    )}

    for gi, gamma in enumerate(gamma_values, 1):
        if verbose:
            print(f"\n===== gamma {gi}/{len(gamma_values)}: "
                  f"gamma_single = {gamma} (G={G}) =====", flush=True)
        frames = run_all_simulations(
            G=G, gamma_start=gamma, gamma_end=gamma,
            output_dir=output_dir, verbose=verbose, save=False, **kwargs)
        for fname, frame in frames.items():
            frame = frame.copy()
            s_cols = [c for c in frame.columns if str(c).startswith("s1_")
                      or str(c).startswith("s2_") or str(c).startswith("s3_")]
            if s_cols:
                pos = list(frame.columns).index(s_cols[0])
                frame.insert(pos, "gamma_single", gamma)
            else:
                frame.insert(0, "gamma_single", gamma)
            combined[fname].append(frame)

    out = {fname: pd.concat(parts, ignore_index=True)
           for fname, parts in combined.items()}
    if save:
        for fname, frame in out.items():
            save_dataframe(frame, fname, output_dir)
        return output_dir
    return out


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=" Monte-Carlo simulation for Clustering extreme value indices in large panels. "
                    "(main, perturbed, and single-group in one).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n_jobs",     type=int,   default=-1,
                   help="-1 = all cores, 1 = sequential/deterministic")
    p.add_argument("--nsim",       type=int,   default=1000)
    p.add_argument("--output_dir", type=str,   default="results_main")
    p.add_argument("--verbose",    type=int,   default=1)

    # number of groups and the tail-index grid
    p.add_argument("--gamma_start", type=float, default=0.2,
                   help="first tail index (and the only one when G=1)")
    p.add_argument("--gamma_end",   type=float, default=1.5,
                   help="last tail index; vgamma_0 = linspace(start, end, G)")
    p.add_argument("--gamma_single", type=float, nargs="+", default=None,
                   help="sweep these single γ values (G=1 style) into ONE CSV set "
                        "with a gamma_single column; e.g. --gamma_single 0.2 0.5 0.7 1.0")

    # DGP
    p.add_argument("--dgp", type=str, nargs="+",
                   default=["independent", "dependent"],
                   choices=["independent", "dependent",
                            "independent_noise", "dependent_noise"])
    p.add_argument("--disturb_a", type=float, default=0.05,
                   help="half-width a of ε_i ~ Uniform(-a, a) for the *_noise DGPs")
    p.add_argument("--dep_rho",   type=float, default=DEP_RHO)
    p.add_argument("--alpha",     type=float, default=ALPHA)

    # scenario axes
    p.add_argument("--G",           type=int,   nargs="+", default=[3, 5],
                   help="number of groups (one or more); G=1 gives the "
                        "single-group case")
    p.add_argument("--T",         type=int,   nargs="+", default=[1000, 3000])
    p.add_argument("--groupsize", type=int,   nargs="+", default=[100, 300])
    p.add_argument("--k1_rate",   type=float, nargs="+", default=[0.09, 0.12])
    p.add_argument("--r_H",       type=float, nargs="+", default=[0.05, 0.07])
    p.add_argument("--r_kt",      type=float, nargs="+", default=[0.02, 0.03, 0.04])
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    common_kwargs = dict(
        vT          = tuple(args.T),
        vgroupsize  = tuple(args.groupsize),
        iNsim       = args.nsim,
        k1_rates    = tuple(args.k1_rate),
        r_kt        = tuple(args.r_kt),
        r_Hs        = tuple(args.r_H),
        dgp_models  = tuple(args.dgp),
        dep_rho     = args.dep_rho,
        alpha       = args.alpha,
        disturb_a   = args.disturb_a,
        n_jobs      = args.n_jobs,
    )

    if args.gamma_single is not None:
        # One-group construction, so it needs ONE G value.
        if len(args.G) != 1:
            raise SystemExit(
                "--gamma_single is the single-group case and takes exactly one "
                f"--G value (got {args.G}). Use e.g. --G 1.")
        run_gamma_sweep(
            gamma_values = args.gamma_single,
            G            = args.G[0],
            output_dir   = args.output_dir,
            verbose      = args.verbose,
            **common_kwargs,
        )
    else:
        run_all_simulations(
            G           = args.G,          
            gamma_start = args.gamma_start,
            gamma_end   = args.gamma_end,
            output_dir  = args.output_dir,
            verbose     = args.verbose,
            **common_kwargs,
        )

    total = time.time() - t0
    print(f"\nTotal wall time: {total / 3600:.2f} h ({total / 60:.1f} min)")


if __name__ == "__main__":
    main()
