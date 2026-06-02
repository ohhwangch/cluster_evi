#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_method.py
===============
Head-to-head comparison of two tail-grouping methods:

    iterative     =  Chen, L., M. Oesting, and C. Zhou (2025). Clustering tails in high dimension.  Iterative tail-clustering (Algorithms 1 & 2)
    segmentation  = Monte-Carlo simulation for Clustering extreme value indices in large panels. Segmentation (elbow)

It covers BOTH comparison directions through a single `--dgp_source` flag:

  --dgp_source iterative     Replicate Chen et al.'s setup: data come from their
                             DGP (Model A / B with spacing Delta) and BOTH methods
                             are run on it. 

  --dgp_source segmentation  Use our DGP (independent / dependent, the panel
                             design from main_simulation) and run BOTH methods. 

For each direction four estimators are scored against the truth:
    acc_iterative_known    : iterative, g known
    acc_iterative_unknown  : iterative, g unknown
    acc_seg_known          : segmentation, g known
    acc_seg_unknown        : segmentation, g unknown
plus per-true-group accuracy and the estimated number of groups (ghat_*).

Experiments (`--experiment`)
----------------------------
    vary_q       accuracy vs cluster/group size q            
    vary_delta   accuracy vs group spacing Delta              
    vary_rk      accuracy vs r_k 
Examples
--------
# Replicate Chen et al., vary q (Figure S.2.7)
python compare_method.py --dgp_source iterative --experiment vary_q 

# Replicate Chen et al., vary Delta (Figure S.2.8)
python compare_method.py --dgp_source iterative --experiment vary_delta

# Our DGP, vary q  (independent + dependent)
python compare_method.py --dgp_source segmentation --experiment vary_q \
    --n_list 1000 3000 --q_grid 100 300

The structural-break core (ssr / parti / datingtrimming / ssrnul / elbow_ghat)
lives in common.py; everything iterative-method-specific (Chen et al.'s DGP,
their algorithm, the tuning rules) lives here.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import multivariate_t
from scipy.optimize import linear_sum_assignment

from common import (
    M_MAX, DEP_RHO, ALPHA,
    fDataGenerating, is_dependent_dgp, time_series_columns,
    datingtrimming, ssrnul, elbow_ghat,
    run_seeds,
)


# =============================================================================
# Default parameter grids
# =============================================================================

SEED_BASE = 12345

# Iterative-source DGP defaults (Model A/B)
ITER_N            = 2000
ITER_G_LIST       = (3, 4, 5)
ITER_MODELS       = ("A", "B")
Q_GRID_ITER       = (10, 15, 20, 25, 30, 35, 40)
DELTA_GRID        = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
DELTA_FIX         = 0.5
Q_FIX             = 15
RK_GRID           = (0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12)

# Segmentation-source DGP defaults (our panel DGP)
SEG_N_LIST       = (1000, 3000)
SEG_G_LIST       = (3, 5)
SEG_MODELS       = ("independent", "dependent")
Q_GRID_SEG       = (100, 300)
SEG_GAMMA_START  = 0.48
SEG_GAMMA_END    = 0.64

# our method tuning
SEG_RK_DEFAULT    = 0.12
SEG_PARA_TRIMMING = 0.15
SEG_TAU           = 0.02


# =============================================================================
# Iterative-source data-generating processes (Chen et al. Model A/B)
# =============================================================================

def simulate_model_A(n, gamma, rng):
    """Independent |Student-t| margins with df_j = 1 / gamma_j."""
    p = gamma.size
    X = np.empty((n, p))
    for j in range(p):
        X[:, j] = np.abs(st.t.rvs(df=1.0 / gamma[j], size=n, random_state=rng))
    return X


def simulate_model_B(n, gamma, rng):
    """Multivariate Cauchy copula, then marginal |Student-t| transform."""
    p   = gamma.size
    idx = np.arange(p)
    Sigma   = 0.5 ** np.abs(idx[:, None] - idx[None, :])
    X_tilde = multivariate_t.rvs(loc=np.zeros(p), shape=Sigma, df=1, size=n,
                                 random_state=rng)
    U = st.t.cdf(X_tilde, df=1.0)
    X = np.empty((n, p))
    for j in range(p):
        X[:, j] = np.abs(st.t.ppf(U[:, j], df=1.0 / gamma[j]))
    return X


def make_gamma_iterative(g, q, Delta):
    """Iterative-source true EVI array + 1-based group labels (group 1 = heaviest)."""
    p            = g * q
    gamma_groups = np.array([(1.0 - Delta) ** (ell - 1) for ell in range(1, g + 1)])
    j            = np.arange(1, p + 1)
    c_true       = ((j - 1) // q) + 1
    gamma        = gamma_groups[c_true - 1]
    return gamma.astype(float), c_true.astype(int)


# =============================================================================
# Segmentation-source data-generating process, reusing common.fDataGenerating
# =============================================================================

def generate_segmentation(n, g, q, model, rng, gamma_start=SEG_GAMMA_START,
                  gamma_end=SEG_GAMMA_END, dep_rho=DEP_RHO, alpha=ALPHA,
                  L_chol=None):
    """
    Generate one panel from our DGP and return it in the (T, units) layout the
    the iterative algorithms (Chen et al.) expect.

    Returns (X, c_true, gamma) where
        X      : (n, p) array of |observations|, columns = units
        c_true : (p,) 1-based true group labels
        gamma  : (p,) true tail index per unit
    """
    vgamma_0 = np.linspace(gamma_start, gamma_end, g)
    dfs      = 1.0 / vgamma_0
    vG       = np.repeat(q, g).astype(int)
    vk       = np.repeat(1, g)

    mSample = fDataGenerating(dfs, vG, vk, n, alpha, rng,
                              dgp_model=model, dep_rho=dep_rho, L_chol=L_chol)
    data    = mSample[time_series_columns(mSample)].to_numpy(dtype=float)  # (p, n)
    X       = np.abs(data.T)                                               # (n, p)
    c_true  = np.repeat(np.arange(1, g + 1), q)
    gamma   = np.repeat(vgamma_0, q)
    return X, c_true, gamma


# =============================================================================
# Iterative tail-clustering algorithms (Operate on the (n, p) matrix)
# =============================================================================

def hill_estimates(X, k):
    """Column-wise Hill estimator on the top k order statistics (k terms)."""
    n, p      = X.shape
    Xs        = np.sort(X, axis=0)
    threshold = Xs[n - k - 1, :]
    top       = Xs[n - k:n, :]
    return np.mean(np.log(top) - np.log(threshold[None, :]), axis=0)


def rescale_Y(X, k_star):
    """Y_i^(j) = X_i^(j) / X_{n-k*:n}^(j) (column-wise)."""
    n, p  = X.shape
    Xs    = np.sort(X, axis=0)
    denom = Xs[n - k_star - 1, :]
    if np.any(denom <= 0):
        raise ValueError("k* normalisation requires X_{n-k*:n} > 0 in every margin.")
    return X / denom[None, :]


def default_tunings_iter(n, p):
    """Chen et al. (2025) Remark 2 recommendations: k, k*, beta."""
    k      = int(np.floor(3.0 * (np.log(p) ** 1.05)))
    k_star = int(np.floor(n ** 0.98))
    k_star = max(k + 1, min(k_star, n - 1))
    beta   = min(2.0 * (k / k_star) * p + 0.5, 0.9)
    return k, k_star, beta


def _threshold_u(Y_cands, k):
    """(k·m)-th upper order statistic of the pooled candidate columns."""
    pooled = Y_cands.ravel()
    N      = pooled.size
    r      = k * Y_cands.shape[1]
    r      = max(0, min(r, N - 1))
    return float(np.partition(pooled, -(r + 1))[-(r + 1)])


def iterative_clustering_known_g(Y, g, beta, k):
    """Iterative clustering, g known (Chen Algorithm 1). Y already rescaled."""
    n, p   = Y.shape
    k_beta = max(1, int(np.floor(beta * k)))
    Ys     = np.sort(Y, axis=0)
    q_beta = Ys[n - k_beta - 1, :]

    labels    = np.zeros(p, dtype=int)
    cand_mask = np.ones(p, dtype=bool)
    for ell in range(1, g):
        cand_idx = np.where(cand_mask)[0]
        if cand_idx.size == 0:
            break
        u   = _threshold_u(Y[:, cand_idx], k)
        grp = cand_idx[q_beta[cand_idx] >= u]
        if grp.size == 0:
            grp = cand_idx
        labels[grp]    = ell
        cand_mask[grp] = False
    labels[cand_mask] = g
    return labels


def iterative_clustering_unknown_g(Y, beta, k, max_groups=50):
    """Iterative clustering, g unknown (Chen Algorithm 2). Y already rescaled."""
    n, p   = Y.shape
    k_beta = max(1, int(np.floor(beta * k)))
    Ys     = np.sort(Y, axis=0)
    q_beta = Ys[n - k_beta - 1, :]

    labels    = np.zeros(p, dtype=int)
    cand_mask = np.ones(p, dtype=bool)
    ell = 1
    while cand_mask.any() and ell <= max_groups:
        cand_idx = np.where(cand_mask)[0]
        u        = _threshold_u(Y[:, cand_idx], k)
        grp      = cand_idx[q_beta[cand_idx] >= u]
        if grp.size == 0:
            grp = cand_idx
        labels[grp]    = ell
        cand_mask[grp] = False
        ell += 1
    return labels


# =============================================================================
# Segmentation: structural-break grouping (Wang et al.; uses common's core)
# =============================================================================

def segmentation_labels_known_g(gamma_hat, G, para_trimming=SEG_PARA_TRIMMING):
    """Known-G segmentation. Sort by descending gamma, find G-1 breaks, map back."""
    p = gamma_hat.size
    order         = np.argsort(-gamma_hat)
    vorderedgamma = gamma_hat[order]
    y = np.array(vorderedgamma, dtype=float).reshape(-1, 1)
    z = np.ones((p, 1))

    _, loc, _ = datingtrimming(y, z, 1, G - 1, 1, p, para_trimming)

    breaks        = np.asarray(loc[:, -1], dtype=int).ravel()
    edges         = np.concatenate(([0], breaks, [p]))
    lengths       = np.diff(edges)
    labels_sorted = np.repeat(np.arange(1, len(lengths) + 1), lengths)
    labels        = np.empty(p, dtype=int)
    labels[order] = labels_sorted
    return labels


def segmentation_labels_unknown_g(gamma_hat, tau=SEG_TAU,
                                  para_trimming=SEG_PARA_TRIMMING, m_max=M_MAX):
    """Unknown-G: pick Ghat with the elbow rule, then segment."""
    p     = gamma_hat.size
    order = np.argsort(-gamma_hat)
    y = np.array(gamma_hat[order], dtype=float).reshape(-1, 1)
    z = np.ones((p, 1))

    ssr0   = ssrnul(y, z)
    m_crit = min(m_max, p - 2)
    if m_crit < 1:
        return np.ones(p, dtype=int)

    n_breaks_to_compute = m_crit + 1
    glob, datevec, _    = datingtrimming(y, z, 1, n_breaks_to_compute, 1,
                                         p, para_trimming)
    ssr_path = np.r_[ssr0, glob[:n_breaks_to_compute, 0]]

    Ghat      = elbow_ghat(ssr_path, tau, m=m_crit)
    num_break = Ghat - 1
    if num_break == 0:
        return np.ones(p, dtype=int)

    breaks        = np.asarray(datevec[:num_break, num_break - 1], dtype=int).ravel()
    edges         = np.concatenate(([0], breaks, [p]))
    lengths       = np.diff(edges)
    labels_sorted = np.repeat(np.arange(1, len(lengths) + 1), lengths)
    labels        = np.empty(p, dtype=int)
    labels[order] = labels_sorted
    return labels


# =============================================================================
# Accuracy (overall + per true group)
# =============================================================================

def accuracy_all(c_true, c_pred):
    """Return (overall_accuracy, per_true_group_accuracy) via Hungarian matching."""
    c_true = np.asarray(c_true)
    c_pred = np.asarray(c_pred)
    tl     = np.unique(c_true)
    pl     = np.unique(c_pred)

    C = np.array([[np.sum((c_true == t) & (c_pred == pp)) for pp in pl]
                  for t in tl])
    ri, ci  = linear_sum_assignment(-C)
    overall = float(C[ri, ci].sum() / len(c_true))

    per_grp = np.full(len(tl), np.nan)
    for r, c in zip(ri, ci):
        n_in_group = (c_true == tl[r]).sum()
        per_grp[r] = C[r, c] / n_in_group if n_in_group > 0 else np.nan
    return overall, per_grp


# =============================================================================
# One replication (both methods) and batch runner
# =============================================================================

def _one_rep(seed_b, dgp_source, model, n, g, q, Delta,
             k_iter, k_star_iter, beta_iter, k_seg,
             seg_para_trimming, seg_tau,
             gamma_start, gamma_end, dep_rho, alpha, L_chol):
    """Generate one dataset (from the chosen source) and score all four methods."""
    rng = np.random.default_rng(seed_b)

    if dgp_source == "iterative":
        gamma, c_true = make_gamma_iterative(g, q, Delta)
        X = simulate_model_A(n, gamma, rng) if model == "A" \
            else simulate_model_B(n, gamma, rng)
    else:  # "segmentation"
        X, c_true, _ = generate_segmentation(n, g, q, model, rng,
                                     gamma_start=gamma_start, gamma_end=gamma_end,
                                     dep_rho=dep_rho, alpha=alpha, L_chol=L_chol)

    # ── iterative  ───────────────────────────────────────────────
    Y              = rescale_Y(X, k_star_iter)
    labels_iter_known      = iterative_clustering_known_g(Y, g, beta_iter, k_iter)
    acc_iterative_known, grp_iterative_known = accuracy_all(c_true, labels_iter_known)
    labels_iter_unknown      = iterative_clustering_unknown_g(Y, beta_iter, k_iter)
    acc_iterative_unknown, grp_iterative_unknown = accuracy_all(c_true, labels_iter_unknown)
    ghat_iterative_unknown        = int(np.unique(labels_iter_unknown).size)

    # ── segmentation ─────────────────────────────────────────────
    gamma_hat = hill_estimates(X, k_seg)
    try:
        labels_seg_known            = segmentation_labels_known_g(gamma_hat, g, seg_para_trimming)
        acc_seg_known, grp_seg_known = accuracy_all(c_true, labels_seg_known)
    except Exception:
        acc_seg_known, grp_seg_known = np.nan, np.full(g, np.nan)
    try:
        labels_seg_unknown            = segmentation_labels_unknown_g(
            gamma_hat, tau=seg_tau, para_trimming=seg_para_trimming)
        acc_seg_unknown, grp_seg_unknown = accuracy_all(c_true, labels_seg_unknown)
        ghat_seg_unknown        = int(np.unique(labels_seg_unknown).size)
    except Exception:
        acc_seg_unknown, grp_seg_unknown, ghat_seg_unknown = np.nan, np.full(g, np.nan), -1

    return dict(
        acc_iterative_known=acc_iterative_known, grp_iterative_known=grp_iterative_known, acc_iterative_unknown=acc_iterative_unknown, grp_iterative_unknown=grp_iterative_unknown,
        acc_seg_known=acc_seg_known, grp_seg_known=grp_seg_known, acc_seg_unknown=acc_seg_unknown, grp_seg_unknown=grp_seg_unknown,
        ghat_iterative_unknown=ghat_iterative_unknown, ghat_seg_unknown=ghat_seg_unknown,
    )


def run_batch(dgp_source, model, n, g, q, Delta,
              k_iter, k_star_iter, beta_iter, k_seg,
              seg_para_trimming, seg_tau,
              B, seed_base, gamma_start, gamma_end,
              dep_rho, alpha, n_jobs):
    """Run B replications for one experimental cell; return mean accuracies."""
    if dgp_source == "segmentation" and is_dependent_dgp(model):
        from common import _cauchy_scale_matrix
        L_chol = np.linalg.cholesky(_cauchy_scale_matrix(g * q, dep_rho))
    else:
        L_chol = None

    arg_tuples = [
        (seed_base + b, dgp_source, model, n, g, q, Delta,
         k_iter, k_star_iter, beta_iter, k_seg,
         seg_para_trimming, seg_tau,
         gamma_start, gamma_end, dep_rho, alpha, L_chol)
        for b in range(B)
    ]
    results = run_seeds(_one_rep, arg_tuples, n_jobs=n_jobs)

    scalar_keys = ["acc_iterative_known", "acc_iterative_unknown", "acc_seg_known", "acc_seg_unknown", "ghat_iterative_unknown", "ghat_seg_unknown"]
    agg = {k: float(np.nanmean([r[k] for r in results])) for k in scalar_keys}
    for k in ["grp_iterative_known", "grp_iterative_unknown", "grp_seg_known", "grp_seg_unknown"]:
        agg[k] = np.nanmean(np.array([r[k] for r in results]), axis=0)
    return agg


# =============================================================================
# Experiments
# =============================================================================

def _models_for(dgp_source, models):
    if models is not None:
        return tuple(models)
    return ITER_MODELS if dgp_source == "iterative" else SEG_MODELS


def _flatten_cell(dgp_source, model, n, g, q, Delta, rk, gamma_label, cell):
    row = dict(dgp_source=dgp_source, model=model, n=n, g=g, q=q,
               gamma=gamma_label, Delta=Delta, rk=rk,
               acc_iterative_known=cell["acc_iterative_known"], acc_iterative_unknown=cell["acc_iterative_unknown"],
               acc_seg_known=cell["acc_seg_known"], acc_seg_unknown=cell["acc_seg_unknown"],
               ghat_iterative_unknown=cell["ghat_iterative_unknown"], ghat_seg_unknown=cell["ghat_seg_unknown"])
    for gi in range(g):
        row[f"acc_iterative_known_g{gi + 1}"] = cell["grp_iterative_known"][gi]
        row[f"acc_iterative_unknown_g{gi + 1}"] = cell["grp_iterative_unknown"][gi]
        row[f"acc_seg_known_g{gi + 1}"] = cell["grp_seg_known"][gi]
        row[f"acc_seg_unknown_g{gi + 1}"] = cell["grp_seg_unknown"][gi]
    return row


def experiment(dgp_source, which, n_list, g_list, models, q_grid, q_fix,
               delta_grid, delta_fix, rk_grid, seg_rk,
               gamma_start, gamma_end, dep_rho, alpha,
               seg_para_trimming, seg_tau, B, seed_base, n_jobs):
    """Run the requested sweep and return a tidy DataFrame."""
    models = _models_for(dgp_source, models)
    rows   = []

    for n in n_list:
        for model in models:
            for g in g_list:
                # axis values for this sweep
                if which == "vary_q":
                    q_vals, delta_vals, rk_vals = q_grid, (delta_fix,), (seg_rk,)
                elif which == "vary_delta":
                    q_vals, delta_vals, rk_vals = (q_fix,), delta_grid, (seg_rk,)
                elif which == "vary_rk":
                    q_vals, delta_vals, rk_vals = (q_fix,), (delta_fix,), rk_grid
                else:
                    raise ValueError(f"Unknown experiment: {which!r}")

                for q in q_vals:
                    for Delta in delta_vals:
                        p = g * q
                        # true tail-index range (the `gamma` label column)
                        if dgp_source == "segmentation":
                            vgamma      = np.linspace(gamma_start, gamma_end, g)
                            gamma_label = f"[{vgamma[0]:.2f}, {vgamma[-1]:.2f}]"
                        else:
                            gg          = np.array([(1.0 - Delta) ** (e - 1)
                                                    for e in range(1, g + 1)])
                            gamma_label = f"[{gg.min():.3f}, {gg.max():.3f}]"
                        for rk in rk_vals:
                            if which == "vary_rk":
                                # both methods share k = floor(rk * n)
                                k_shared = max(1, int(np.floor(rk * n)))
                                _, k_star_iter, _ = default_tunings_iter(n, p)
                                k_star_iter = max(k_shared + 1, k_star_iter)
                                beta_iter   = min(2.0 * (k_shared / k_star_iter)
                                                  * p + 0.5, 0.9)
                                k_iter, k_seg = k_shared, k_shared
                            else:
                                k_iter, k_star_iter, beta_iter = \
                                    default_tunings_iter(n, p)
                                k_seg = max(1, int(np.floor(rk * n)))

                            cell = run_batch(
                                dgp_source, model, n, g, q, Delta,
                                k_iter, k_star_iter, beta_iter, k_seg,
                                seg_para_trimming, seg_tau,
                                B, seed_base, gamma_start, gamma_end,
                                dep_rho, alpha, n_jobs)
                            rows.append(_flatten_cell(dgp_source, model, n, g, q,
                                                      Delta, rk, gamma_label, cell))
                            gctl = (f"Δ={Delta}" if dgp_source == "iterative"
                                    else f"gamma={gamma_label}")
                            print(f"  [{which}] src={dgp_source} model={model} "
                                  f"n={n} g={g} q={q} {gctl} rk={rk}  "
                                  f"c1={cell['acc_iterative_known']:.3f} w1={cell['acc_seg_known']:.3f} "
                                  f"w2={cell['acc_seg_unknown']:.3f}", flush=True)
    return pd.DataFrame(rows)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare iterative tail-clustering (Chen et al.) with our segmentation (Wang et al.), "
                    "on either the iterative-source or segmentation-source DGP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dgp_source", type=str, default="iterative",
                   choices=["iterative", "segmentation"],
                   help="iterative = Chen Model A/B data; segmentation = our panel DGP")
    p.add_argument("--experiment", type=str, default="vary_q",
                   choices=["vary_q", "vary_delta", "vary_rk"])
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--nsim",   type=int, default=500, help="replications per cell")
    p.add_argument("--seed",   type=int, default=SEED_BASE)
    p.add_argument("--out_dir", type=str, default="results_compare_method")

    # axes (None -> source-appropriate defaults are filled in main)
    p.add_argument("--n_list", type=int,   nargs="+", default=None)
    p.add_argument("--g_list", type=int,   nargs="+", default=None)
    p.add_argument("--models", type=str,   nargs="+", default=None)
    p.add_argument("--q_grid", type=int,   nargs="+", default=None)
    p.add_argument("--q_fix",  type=int,   default=Q_FIX)
    p.add_argument("--rk_grid", type=float, nargs="+", default=list(RK_GRID))
    p.add_argument("--seg_rk", type=float, default=SEG_RK_DEFAULT)

    # ---- how the true tail indices gamma are distributed across groups ----------
    # iterative source: gamma_ell = (1 - Δ)^(ell-1), so Δ sets the group spacing.
    p.add_argument("--delta_grid", type=float, nargs="+", default=list(DELTA_GRID),
                   help="[iterative source] Δ values swept by --experiment vary_delta")
    p.add_argument("--delta_fix",  type=float, default=DELTA_FIX,
                   help="[iterative source] fixed Δ (group spacing) for vary_q / vary_rk")
    # segmentation source: gamma = linspace(gamma_start, gamma_end, g).
    p.add_argument("--gamma_start", type=float, default=SEG_GAMMA_START,
                   help="[segmentation source] first (smallest) group tail index")
    p.add_argument("--gamma_end",   type=float, default=SEG_GAMMA_END,
                   help="[segmentation source] last (largest) group tail index")

    p.add_argument("--dep_rho",     type=float, default=DEP_RHO)
    p.add_argument("--alpha",       type=float, default=ALPHA)

    # segmentation tuning
    p.add_argument("--seg_para_trimming", type=float, default=SEG_PARA_TRIMMING)
    p.add_argument("--seg_tau",           type=float, default=SEG_TAU)
    return p.parse_args()


def main():
    args = parse_args()

    if args.dgp_source == "segmentation" and args.experiment == "vary_delta":
        raise SystemExit(
            "vary_delta is only defined for --dgp_source iterative "
            "(our DGP has no Delta spacing; use --gamma_start/--gamma_end).")

    # Gentle warning if the gamma-control of the *other* source was supplied:
    # for the iterative source the spacing is Δ (--delta_fix/--delta_grid); for the segmentation source it is the
    # range (--gamma_start/--gamma_end). The off-source flag is simply ignored.
    passed = set(sys.argv[1:])
    if args.dgp_source == "iterative" and (
            "--gamma_start" in passed or "--gamma_end" in passed):
        print("[note] --gamma_start/--gamma_end are ignored for source 'iterative'; "
              "gamma spacing is set by --delta_fix / --delta_grid.", flush=True)
    if args.dgp_source == "segmentation" and (
            "--delta_fix" in passed or "--delta_grid" in passed):
        print("[note] --delta_fix/--delta_grid are ignored for source 'segmentation'; "
              "gamma is set by --gamma_start / --gamma_end.", flush=True)

    # Source-appropriate axis defaults
    if args.dgp_source == "iterative":
        n_list = args.n_list or [ITER_N]
        g_list = args.g_list or list(ITER_G_LIST)
        q_grid = args.q_grid or list(Q_GRID_ITER)
    else:
        n_list = args.n_list or list(SEG_N_LIST)
        g_list = args.g_list or list(SEG_G_LIST)
        q_grid = args.q_grid or list(Q_GRID_SEG)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"=== compare_method: source={args.dgp_source} "
          f"experiment={args.experiment} reps={args.nsim} ===", flush=True)

    df = experiment(
        dgp_source=args.dgp_source, which=args.experiment,
        n_list=n_list, g_list=g_list, models=args.models,
        q_grid=q_grid, q_fix=args.q_fix,
        delta_grid=tuple(args.delta_grid), delta_fix=args.delta_fix,
        rk_grid=tuple(args.rk_grid), seg_rk=args.seg_rk,
        gamma_start=args.gamma_start, gamma_end=args.gamma_end,
        dep_rho=args.dep_rho, alpha=args.alpha,
        seg_para_trimming=args.seg_para_trimming, seg_tau=args.seg_tau,
        B=args.nsim, seed_base=args.seed, n_jobs=args.n_jobs,
    )

    fname    = f"results_{args.dgp_source}_{args.experiment}.csv"
    csv_path = os.path.join(args.out_dir, fname)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved -> {csv_path}")


if __name__ == "__main__":
    main()
