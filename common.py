"""
common.py
=========
Shared building blocks for Clustering extreme value indices in large panels.

This module is imported by the experiment scripts
(`main_simulation.py`, `compare_method.py`, `rejection_simulation.py`,`empirics.py`).
It is *not* meant to be run on its own.

Contents
--------
1. Constants
2. Data-generating process (4 DGPs)
       independent      dependent
       independent_noise   dependent_noise   (uniform individual perturbation)
3. Structural-break machinery (ssr / parti / datingtrimming / ssrnul / elbow)
4. Misc helpers (Hill estimator, group accuracy, parallel runner, CSV serialisation)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import t as student_t

# =============================================================================
# 1. Constants
# =============================================================================

FIXED_THRED       = 0.02            # elbow threshold used to pick Ghat
M_MAX             = 7               # elbow rule searches m = 1, ..., M_MAX
ELBOW_THRESHOLDS = (0.015, 0.02, 0.025, 0.03, 0.035)   # thresholds for Elbow method 
DEP_RHO           = 0.5             # dependence parameter for the dependent DGPs
ALPHA             = 0.5             # fraction of Student-t units per group


def is_noise_dgp(dgp_model):
    """True for the two perturbed ('*_noise') DGP labels."""
    return dgp_model in {"independent_noise", "dependent_noise"}

def is_dependent_dgp(dgp_model):
    """True for the two cross-sectionally dependent DGP labels."""
    return dgp_model in {"dependent", "dependent_noise"}


# =============================================================================
# 2. Data-generating process
# =============================================================================

def fDGP_burr(N, T, c, k, rng):
    """Draw an (N, T) panel of Burr(c, k) observations."""
    u = rng.uniform(0, 1, (N, T))
    return ((1 - u) ** (-1 / k) - 1) ** (1 / c)


def fDGP_burr_inv(u, c, k=1):
    """Burr(c, k) inverse CDF evaluated at uniforms `u` (vectorised)."""
    u = np.clip(u, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return ((1 - u) ** (-1 / k) - 1) ** (1 / c)


def _cauchy_scale_matrix(N, dep_rho):
    """Cross-sectional dependence (Cauchy copula) scale matrix Sigma_{ij} = rho^|i-j|."""
    idx = np.arange(N)
    return dep_rho ** np.abs(idx[:, None] - idx[None, :])


def _multivariate_cauchy(T, rng, Sigma=None, L=None):
    """
    Draw T multivariate-Cauchy rows as Gaussian / sqrt(chi2_1).

    Provide exactly one of:
        Sigma : the scale matrix (its Cholesky factor is computed here), or
        L     : a precomputed lower-triangular Cholesky factor of Sigma.
    Passing `L` (precomputed once per scenario) avoids the O(N^3) decomposition on every call.
    """
    if (Sigma is None) == (L is None):
        raise ValueError("Pass exactly one of `Sigma` or `L`.")
    if L is None:
        L = np.linalg.cholesky(Sigma)
    N             = L.shape[0]
    gaussian_part = rng.standard_normal((T, N)) @ L.T
    chi2_part     = rng.chisquare(df=1, size=T)
    return gaussian_part / np.sqrt(chi2_part)[:, None]

def _build_marginal_specs(dfs, vG, vk, alpha):
    labels, specs = [], []
    for group_idx, (df, N, k) in enumerate(zip(dfs, vG, vk), start=1):
        df, N, k = float(df), int(N), float(k)
        num_t    = int(N * alpha)
        num_burr = N - num_t
        c        = df / k
        for _ in range(num_burr):
            specs.append({"distribution": "burr", "df": df, "c": c, "k": k,
                          "group": group_idx})
            labels.append(f"burr({1 / (c * k)},{-1 / k})")
        for _ in range(num_t):
            specs.append({"distribution": "t", "df": df, "c": c, "k": k,
                          "group": group_idx})
            labels.append(f"t({1 / df},{-2 / df})")
    return specs, labels


def fDataGenerating_independent(dfs, vG, vk, T, alpha, rng):
    all_samples = []
    for i in range(len(dfs)):
        df, k, N = dfs[i], vk[i], vG[i]
        num_t    = int(N * alpha)
        num_burr = N - num_t
        c        = df / k
        burr_s   = fDGP_burr(num_burr, T, c, k, rng)
        t_s      = rng.standard_t(df, size=(num_t, T))
        data     = np.vstack((burr_s, t_s))
        labels   = ([f"burr({1/(c*k)},{-1/k})"] * num_burr
                    + [f"t({1/df},{-2/df})"]  * num_t)
        gdf      = pd.DataFrame(data)
        gdf.insert(0, "Distribution", labels)
        all_samples.append(gdf)
    return pd.concat(all_samples, ignore_index=True)


def fDataGenerating_dependent(dfs, vG, vk, T, alpha, rng, dep_rho=DEP_RHO,
                              L_chol=None):
    """Dependent (Cauchy-copula) DGP. Pass `L_chol` to skip the O(N^3) Cholesky."""
    specs, labels = _build_marginal_specs(dfs, vG, vk, alpha)
    N_total       = len(specs)
    if L_chol is None:
        X = _multivariate_cauchy(T, rng, Sigma=_cauchy_scale_matrix(N_total, dep_rho))
    else:
        X = _multivariate_cauchy(T, rng, L=L_chol)
    U             = student_t.cdf(X, df=1)
    U             = np.clip(U, np.finfo(float).eps, 1 - np.finfo(float).eps)
    data          = np.empty((N_total, T), dtype=float)
    for idx, spec in enumerate(specs):
        if spec["distribution"] == "burr":
            data[idx, :] = fDGP_burr_inv(U[:, idx], spec["c"], spec["k"])
        else:
            data[idx, :] = student_t.ppf(U[:, idx], df=spec["df"])
    mSample = pd.DataFrame(data)
    mSample.insert(0, "Distribution", labels)
    return mSample


# ── individual (perturbed) tail index  γ̃_i = γ_g + ε_i,  ε_i ~ U(-a, a) ───────

def _build_marginal_specs_disturbed(dfs_per_unit, vG, vk, alpha):
    """Like `_build_marginal_specs` but with a per-unit df array."""
    labels, specs = [], []
    offset = 0
    for group_idx, (N, k) in enumerate(zip(vG, vk), start=1):
        N, k     = int(N), float(k)
        num_t    = int(N * alpha)
        num_burr = N - num_t
        grp_dfs  = dfs_per_unit[offset:offset + N]
        for i in range(num_burr):
            df_i = float(grp_dfs[i])
            specs.append({"distribution": "burr", "df": df_i, "c": df_i / k,
                          "k": k, "group": group_idx})
            labels.append(f"burr_noise_g{group_idx}")
        for i in range(num_t):
            df_i = float(grp_dfs[num_burr + i])
            specs.append({"distribution": "t", "df": df_i, "c": df_i / k,
                          "k": k, "group": group_idx})
            labels.append(f"t_noise_g{group_idx}")
        offset += N
    return specs, labels


def fDataGenerating_independent_disturbed(dfs_per_unit, vG, vk, T, alpha, rng):
    """Independent DGP with unit-specific tail indices (vectorised per group)."""
    all_samples = []
    offset = 0
    for group_idx, (N, k) in enumerate(zip(vG, vk), start=1):
        N, k     = int(N), float(k)
        num_t    = int(N * alpha)
        num_burr = N - num_t
        grp_dfs  = dfs_per_unit[offset:offset + N]

        data_parts = []
        if num_burr > 0:
            c_arr = (grp_dfs[:num_burr] / k).reshape(-1, 1)
            u_b   = rng.uniform(0.0, 1.0, (num_burr, T))
            data_parts.append(fDGP_burr_inv(u_b, c=c_arr, k=k))
        if num_t > 0:
            df_t_arr = grp_dfs[num_burr:].reshape(-1, 1)
            u_t      = rng.uniform(0.0, 1.0, (num_t, T))
            data_parts.append(student_t.ppf(u_t, df=df_t_arr))

        data   = np.vstack(data_parts) if data_parts else np.empty((0, T))
        labels = ([f"burr_noise_g{group_idx}"] * num_burr
                  + [f"t_noise_g{group_idx}"]  * num_t)
        gdf    = pd.DataFrame(data)
        gdf.insert(0, "Distribution", labels)
        all_samples.append(gdf)
        offset += N
    return pd.concat(all_samples, ignore_index=True)


def fDataGenerating_dependent_disturbed(dfs_per_unit, vG, vk, T, alpha, rng,
                                        dep_rho=DEP_RHO, L_chol=None):
    """Dependent (Cauchy-copula) DGP with unit-specific tail indices."""
    specs, labels = _build_marginal_specs_disturbed(dfs_per_unit, vG, vk, alpha)
    N_total       = len(specs)
    if L_chol is None:
        X = _multivariate_cauchy(T, rng, Sigma=_cauchy_scale_matrix(N_total, dep_rho))
    else:
        X = _multivariate_cauchy(T, rng, L=L_chol)
    U    = student_t.cdf(X, df=1)
    U    = np.clip(U, np.finfo(float).eps, 1 - np.finfo(float).eps)
    data = np.empty((N_total, T), dtype=float)
    for idx, spec in enumerate(specs):
        if spec["distribution"] == "burr":
            data[idx, :] = fDGP_burr_inv(U[:, idx], spec["c"], spec["k"])
        else:
            data[idx, :] = student_t.ppf(U[:, idx], df=spec["df"])
    mSample = pd.DataFrame(data)
    mSample.insert(0, "Distribution", labels)
    return mSample


# ── dispatchers ───────────────────────────────────────────────────────────────

def fDataGenerating(dfs, vG, vk, T, alpha, rng,
                    dgp_model="independent", dep_rho=DEP_RHO, L_chol=None):
    """Group-constant-γ dispatcher (the non-noise DGPs)."""
    if dgp_model == "independent":
        return fDataGenerating_independent(dfs, vG, vk, T, alpha, rng)
    if dgp_model == "dependent":
        return fDataGenerating_dependent(dfs, vG, vk, T, alpha, rng,
                                         dep_rho=dep_rho, L_chol=L_chol)
    raise ValueError(f"Unknown (non-noise) dgp_model: {dgp_model!r}")


def fDataGenerating_disturbed(dfs_per_unit, vG, vk, T, alpha, rng,
                              dependent=False, dep_rho=DEP_RHO, L_chol=None):
    """Individual-γ dispatcher (the '*_noise' DGPs)."""
    if dependent:
        return fDataGenerating_dependent_disturbed(
            dfs_per_unit, vG, vk, T, alpha, rng, dep_rho=dep_rho, L_chol=L_chol)
    return fDataGenerating_independent_disturbed(
        dfs_per_unit, vG, vk, T, alpha, rng)

# =============================================================================
# 3. Structural-break machinery (Bai-Perron style global SSR minimisation)
# =============================================================================

def ssr(start, y, z, h, last):
    """Recursive SSR for segments beginning at `start` (1-indexed)."""
    vecssr   = np.zeros((last, 1))
    z_seg    = z[start - 1: start - 1 + h, :]
    inv1     = np.linalg.inv(z_seg.T @ z_seg)
    delta1   = inv1 @ (z_seg.T @ y[start - 1: start - 1 + h, :])
    res      = y[start - 1: start - 1 + h, :] - z_seg @ delta1
    vecssr[start + h - 2, 0] = (res.T @ res)[0, 0]
    r = start + h
    while r <= last:
        v      = y[r - 1, 0] - (z[r - 1, :] @ delta1)[0]
        invz   = inv1 @ z[r - 1, :].reshape(-1, 1)
        f      = 1 + (z[r - 1, :].reshape(1, -1) @ invz)[0, 0]
        delta1 = delta1 + invz * v
        inv1   = inv1 - (invz @ invz.T) / f
        vecssr[r - 1, 0] = vecssr[r - 2, 0] + v * v / f
        r += 1
    return vecssr


def parti(start, b1, b2, last, bigvec, bigt):
    """Optimal one-break partition for a segment, breaks in [b1, b2]."""
    dvec_local = np.zeros((bigt, 1))
    ini        = (start - 1) * bigt - (start - 2) * (start - 1) // 2 + 1
    ini_idx    = int(ini - 1)
    j = b1
    while j <= b2:
        k = int(j * bigt - (j - 1) * j // 2 + last - j)
        dvec_local[j - 1, 0] = (bigvec[ini_idx + (j - start), 0]
                                 + bigvec[k - 1, 0])
        j += 1
    sub_dvec    = dvec_local[b1 - 1: b2, 0]
    ssrmin      = np.min(sub_dvec)
    minindcdvec = int(np.argmin(sub_dvec))
    dx          = (b1 - 1) + (minindcdvec + 1)
    return ssrmin, dx


def datingtrimming(y, z, h, m, q, bigt, para_trimming):
    """
    Globally SSR-minimising break dates for up to `m` breaks.

    Returns
    -------
    glb     : (m, 1) optimal SSR for 1, ..., m breaks.
    datevec : (m, m) break-date matrix (column j holds the j-break solution).
    bigvec  : flattened lower-triangular SSR store.

    `q` (number of regressors) is accepted for call-site compatibility and is
    not used internally; the regression dimension is taken from `z`.
    """
    datevec  = np.zeros((m, m))
    optdat   = np.zeros((bigt, m))
    optssr   = np.zeros((bigt, m))
    dvec     = np.zeros((bigt, 1))
    glb      = np.zeros((m, 1))
    bigvec   = np.zeros((bigt * (bigt + 1) // 2, 1))

    start_trimming = int(np.floor(para_trimming * bigt))
    end_trimming   = int(np.floor((1 - para_trimming) * bigt))

    for i in range(1, bigt - h + 2):
        vecssr    = ssr(i, y, z, h, bigt)
        start_idx = int((i - 1) * bigt + i - ((i - 1) * i) // 2) - 1
        end_idx   = int(i * bigt - ((i - 1) * i) // 2)
        bigvec[start_idx:end_idx, 0] = vecssr[i - 1:bigt, 0]

    if m == 1:
        ssrmin, datx      = parti(1, h, bigt - h, bigt, bigvec, bigt)
        datevec[0, 0]     = datx
        glb[0, 0]         = ssrmin
    else:
        for j1 in range(2 * h, bigt + 1):
            ssrmin, datx      = parti(1, h, j1 - h, j1, bigvec, bigt)
            optssr[j1 - 1, 0] = ssrmin
            optdat[j1 - 1, 0] = datx
        glb[0, 0]     = optssr[bigt - 1, 0]
        datevec[0, 0] = optdat[bigt - 1, 0]

        for ib in range(2, m + 1):
            if ib == m:
                jlast = bigt
                for jb in range(ib * h, jlast - h + 1):
                    dvec[jb - 1, 0] = (
                        optssr[jb - 1, ib - 2]
                        + bigvec[int((jb + 1) * bigt - jb * (jb + 1) // 2) - 1, 0]
                    )
                sub_range                 = dvec[ib * h - 1: jlast - h, 0]
                optssr[jlast - 1, ib - 1] = np.min(sub_range)
                minindcdvec               = int(np.argmin(sub_range))
                optdat[jlast - 1, ib - 1] = (ib * h - 1) + (minindcdvec + 1)
            else:
                for jlast in range((ib + 1) * h, bigt + 1):
                    for jb in range(ib * h, jlast - h + 1):
                        dvec[jb - 1, 0] = (
                            optssr[jb - 1, ib - 2]
                            + bigvec[int(jb * bigt - jb * (jb - 1) // 2
                                         + jlast - jb) - 1, 0]
                        )
                    sub_range                 = dvec[ib * h - 1: jlast - h, 0]
                    optssr[jlast - 1, ib - 1] = np.min(sub_range)
                    minindcdvec               = int(np.argmin(sub_range))
                    optdat[jlast - 1, ib - 1] = (ib * h - 1) + (minindcdvec + 1)

            datevec[ib - 1, ib - 1] = optdat[bigt - 1, ib - 1]
            for i_inner in range(1, ib):
                xx                      = ib - i_inner
                prev_break              = int(datevec[xx, ib - 1])
                datevec[xx - 1, ib - 1] = optdat[prev_break - 1, xx - 1]
            glb[ib - 1, 0] = optssr[bigt - 1, ib - 1]

    if datevec[0, 1] < start_trimming:
        datevec[0, 1] = start_trimming
    elif datevec[1, 1] > end_trimming:
        datevec[1, 1] = end_trimming

    return glb, datevec, bigvec


def ssrnul(y, zz):
    """SSR of the no-break (single-regime) model."""
    delta     = np.linalg.lstsq(zz, y, rcond=None)[0]
    residuals = y - zz @ delta
    return float((residuals.T @ residuals).item())


def elbow_ghat(ssr_path, thred, m=M_MAX):
    """
    Elbow criterion including the one-group (m = 0) case.

    `ssr_path` must be [SSR0, SSR(l_1), ..., SSR(l_{m+1})].
    Returns Ghat in {1, ..., m + 2}.

    Rule:
        Ghat = 1                         if (SSR0 - SSR(l_1)) / SSR0 <= thred,
        else first j with
            [SSR(l_{j+1}) - SSR(l_j)] / [SSR(l_j) - SSR0] <= thred  -> Ghat = j+1,
        else Ghat = m + 2.
    """
    ssr_path = np.asarray(ssr_path, dtype=float)

    if len(ssr_path) < m + 2:
        raise ValueError(
            f"ssr_path must contain SSR0 and SSR(l_1), ..., SSR(l_{m+1}); "
            f"expected at least {m + 2} values, got {len(ssr_path)}."
        )

    ssr0            = ssr_path[0]
    first_reduction = (ssr0 - ssr_path[1]) / ssr0
    if first_reduction <= thred:
        return 1

    numer = ssr_path[2:m + 2] - ssr_path[1:m + 1]
    denom = ssr_path[1:m + 1] - ssr0
    with np.errstate(divide="ignore", invalid="ignore"):
        std_ssr = numer / denom

    candidates = np.where(std_ssr <= thred)[0]
    if len(candidates):
        return int(candidates[0] + 2)   # candidates[0]=0 -> j=1 -> Ghat=2
    return int(m + 2)


def extract_index(datevec, G, N):
    """Group-boundary array of length G+1 from a `datingtrimming` datevec."""
    if G <= 1:
        return np.array([0, N], dtype=int)
    n_breaks    = G - 1
    breakpoints = datevec[:n_breaks, n_breaks - 1].astype(int)
    return np.array([0] + list(breakpoints) + [N], dtype=int)


# =============================================================================
# 4. Misc helpers
# =============================================================================

def fHill(sample, k):
    """Hill EVI estimate from a 1-D `sample` using the top `k` order stats."""
    sorted_sample = np.sort(sample)
    log_ratios    = np.log(sorted_sample[-k:]) - np.log(sorted_sample[-k - 1])
    return np.mean(log_ratios)


def time_series_columns(mSample):
    """The integer-named (time-series) columns of an mSample DataFrame."""
    return [c for c in mSample.columns if isinstance(c, (int, np.integer))]


def add_gamma_estimates(mSample, k, col_name="gamma"):
    """Append a column of row-wise Hill estimates (one per unit)."""
    out    = mSample.copy()
    values = out[time_series_columns(out)].to_numpy(dtype=float)
    out[col_name] = np.apply_along_axis(fHill, 1, values, int(k))
    return out
def group_accuracy(g_hat_orig, g0_orig, G):
    """Per-true-group classification accuracy: acc_g = mean(g_hat[G_0==g]==g)."""
    return np.array(
        [(g_hat_orig[g0_orig == g] == g).mean() for g in range(1, G + 1)],
        dtype=float,
    )


def run_seeds(func, arg_tuples, n_jobs=-1, backend="loky"):
    """
    Run `func(*args)` for every tuple in `arg_tuples`, in parallel.

    `func` must be a module-level (picklable) function. Set `n_jobs=1` for a
    sequential, deterministic run; `n_jobs=-1` uses all available cores.
    """
    return Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(func)(*args) for args in arg_tuples
    )


def to_serializable(value):
    """Convert numpy / list values to JSON strings so they survive a CSV round-trip."""
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    if isinstance(value, (list, tuple)):
        return json.dumps([v.tolist() if isinstance(v, np.ndarray) else v
                           for v in value])
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def save_dataframe(df, filename, output_dir):
    """Serialise array-valued columns and write `df` to `output_dir/filename` as CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path   = output_dir / filename
    csv_df     = df.copy()
    for col in csv_df.columns:
        csv_df[col] = csv_df[col].map(to_serializable)
    csv_df.to_csv(csv_path, index=False)
    return csv_path
