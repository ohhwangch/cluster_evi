#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py
===============
Families
--------
A. Dot plots (Bias^2 / SD / RMSE):
     plot_2x2(folder, kind, ...)   2x2 figure, reads {folder}/measure_sums.csv,
                                    saves to {folder}/figures/.  Four folders:
                                      main / supp   -> kind="normal"   (gamma 0.2-1.5)
                                      small_gap     -> kind="smallgap" (gamma 0.48-0.64)
                                      perturbed     -> kind="disturb"  (DGP *_noise)
     plot_G1(csv, ...)             1x2 figure for the single-group case,
                                    reads sim_results_G1_noise/measure_sums.csv.

B. Method comparison (read from results_compare_method):
     plot_method_accuracy(...)     accuracy-vs-x line panels (G unknown)
     plot_smallgap_pergroup(...)   per-group accuracy profiles
     plot_smallgap_heatmap(...)    overall-accuracy heatmaps
C. Empirical Rejection Rate (read from rejection_simulation.py output) 
    plot_rejection_rate()

The dot-plot panels for the 2x2 and the G1 figures share everything except
three cosmetic details (x tick-label format, x right-padding, title pad), which
are passed as arguments so each figure reproduces its original look exactly.
"""


import json
from pathlib import Path
from itertools import product
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


# =============================================================================
# Shared rc fragments
# =============================================================================

_SERIF = {
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
}

DOT_RC = {
    **_SERIF,
    "font.size":         10,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size":  3.5,
    "ytick.major.size":  3.5,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
}


# =============================================================================
# FAMILY A — dot plots (shared encoding, layout, drawing)
# =============================================================================

MEASURE_NAMES = [r"$\mathrm{Bias}^2$", "SD", "RMSE"]
EST_NAMES     = [
    r"$\hat{\gamma}_{g}(G)$",
    r"$\hat{\gamma}_{g}(\hat{G})$",
    r"$\hat{\gamma}^{\mathrm{B}}$",
]
MARKERS = ["o", "s", "^"]
#          Bias     SD       RMSE
_FC  = ["none",  "black", "none"]
_EC  = ["black", "black", "black"]
_HTC = [None,    None,    "////////"]
MS   = 6.5
MEW  = 0.9

EST_STEP  = 0.12
SUB_GAP   = 0.20
GROUP_GAP = 1.28


def _layout(n):
    """(offsets (3,3), centers (n,), total_w) — identical formula everywhere."""
    n_m, n_e = 3, 3
    sub_w    = (n_e - 1) * EST_STEP
    total_w  = (n_m - 1) * (sub_w + SUB_GAP) + sub_w
    start    = -total_w / 2
    offsets  = np.zeros((n_m, n_e))
    for m in range(n_m):
        sub_start = start + m * (sub_w + SUB_GAP)
        for e in range(n_e):
            offsets[m, e] = sub_start + e * EST_STEP
    centers = np.arange(n, dtype=float) * GROUP_GAP
    return offsets, centers, total_w


def _draw_panel(ax, values, gamma_vals, *, tick_fmt, right_pad,
                title=None, title_pad=6):
    """
    Draw one dot panel. `values` has shape (3 measures, 3 estimators, n).

    Per-figure differences passed in:
        tick_fmt  : gamma -> x-tick label
        right_pad : extra right x-limit padding (2x2: 0.15, G1: 0.02)
        title / title_pad : optional panel title
    """
    n = values.shape[2]
    offsets, centers, total_w = _layout(n)

    ax.yaxis.grid(True, linestyle=":", linewidth=0.6, color="0.75")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    for j in range(n):
        xc = centers[j]
        for m_idx in range(3):
            for e_idx in range(3):
                ax.scatter(
                    xc + offsets[m_idx, e_idx], values[m_idx, e_idx, j],
                    s=MS ** 2, marker=MARKERS[e_idx],
                    facecolors=_FC[m_idx], edgecolors=_EC[m_idx],
                    linewidths=MEW, hatch=_HTC[m_idx], zorder=4, clip_on=False,
                )

    for j in range(n - 1):
        sep = (centers[j] + centers[j + 1]) / 2
        ax.axvline(sep, color="0.80", linewidth=0.7, linestyle="-", zorder=1)

    ax.set_xticks(centers)
    ax.set_xticklabels([tick_fmt(g) for g in gamma_vals], fontsize=20)
    ax.set_xlim(centers[0] - total_w / 2 - 0.15,
                centers[-1] + total_w / 2 + right_pad)
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([0, 0.08, 0.16])
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda x, _: f"{x:.2f}" if x != 0 else "0"))
    ax.tick_params(axis="y", labelsize=20)
    if title is not None:
        ax.set_title(title, fontsize=20, pad=title_pad)


def _measures_from_sums(s1, s2, M):
    """
    (bias, sd, rmse) for one group from per-unit running sums s1, s2 over M reps.

    For unit i:
        b(i) = s1_i / M          (M^-1 sum_m (ghat - gamma))
        q(i) = s2_i / M          (M^-1 sum_m (ghat - gamma)^2)
    and per group (group size S_G = number of units in s1/s2):
        Bias = ( S_G^-1 sum_i b(i)^2 )^(1/2)
        SD   = ( S_G^-1 sum_i ( q(i) - b(i)^2 ) )^(1/2)    # group-level variance
        RMSE = ( S_G^-1 sum_i q(i) )^(1/2)
    """
    b = s1 / M
    q = s2 / M
    bias = np.sqrt(np.mean(b ** 2))
    sd   = np.sqrt(np.maximum(np.mean(q - b ** 2), 0))
    rmse = np.sqrt(np.mean(q))
    return bias, sd, rmse


# ── 2x2 figure ────────────────────────────────────────────────────────────────

def _load_2x2(csv_path, T, G, groupsize, k1_rate, dgp_model, r_kt):
    """One matching row of measure_sums.csv -> (3,3,G) Bias/SD/RMSE values."""
    df   = pd.read_csv(csv_path)
    mask = (
        (df["T"]          == T)         &
        (df["G"]          == G)         &
        (df["group_size"] == groupsize) &
        (df["k1_rate"]    == k1_rate)   &
        (df["dgp_model"]  == dgp_model)
    )
    df_s = df[mask].copy()
    if df_s.empty:
        raise ValueError(
            f"No rows for T={T}, G={G}, groupsize={groupsize}, "
            f"k1_rate={k1_rate}, dgp_model={dgp_model!r} in {csv_path}.\n"
            f"Available dgp_model: {sorted(df['dgp_model'].unique())}; "
            f"G: {sorted(df['G'].unique())}.")
    rows = df_s[np.isclose(df_s["r_kt"], r_kt)]
    if rows.empty:
        raise ValueError(f"No row for r_kt={r_kt}.")
    row = rows.iloc[0]
    M   = int(row["Nsim"])

    def group_measures(prefix):
        s1 = [np.array(a) for a in json.loads(row[f"s1_{prefix}_byunit"])]
        s2 = [np.array(a) for a in json.loads(row[f"s2_{prefix}_byunit"])]
        b = np.empty(G); s = np.empty(G); r = np.empty(G)
        for j in range(G):
            b[j], s[j], r[j] = _measures_from_sums(s1[j], s2[j], M)
        return b, s, r

    b_g,  s_g,  r_g  = group_measures("g")
    b_gh, s_gh, r_gh = group_measures("gh")
    b_i,  s_i,  r_i  = group_measures("ind")
    return np.array([
        [b_g,  b_gh,  b_i],    # Bias
        [s_g,  s_gh,  s_i],    # SD
        [r_g,  r_gh,  r_i],    # RMSE
    ])


def _draw_inset_legend(ax, fs=10):
    """3x3 legend matrix inset inside the top-left panel (2x2 figure)."""
    ins = ax.inset_axes([0.01, 0.52, 0.38, 0.46])
    ins.set_xlim(0, 1); ins.set_ylim(0, 1)
    ins.set_xticks([]); ins.set_yticks([])
    for spine in ins.spines.values():
        spine.set_linewidth(0.8); spine.set_edgecolor("black")
    ins.set_facecolor("white")
    col_x = [0.45, 0.7, 0.9]
    row_y = [0.7, 0.4, 0.1]
    hdr_y = 0.82
    for e, name in enumerate(EST_NAMES):
        ins.text(col_x[e], hdr_y, name, fontsize=fs, ha="center", va="baseline")
    for m_idx in range(3):
        ins.text(0.32, row_y[m_idx], MEASURE_NAMES[m_idx],
                 fontsize=fs, ha="right", va="center")
        for e_idx in range(3):
            ins.scatter(col_x[e_idx], row_y[m_idx], s=5.5**2,
                        marker=MARKERS[e_idx], facecolors=_FC[m_idx],
                        edgecolors=_EC[m_idx], linewidths=MEW,
                        hatch=_HTC[m_idx], zorder=4)


def _pct(x):
    return f"{int(round(x * 100)):02d}"


_KIND_INFIX = {"normal": "", "disturb": "_disturb", "smallgap": "_smallgap"}


def plot_2x2(
    folder,
    kind      = "normal",
    T         = 1000,
    groupsize = 300,
    k1_rate   = 0.12,
    r_kt      = 0.03,
    r_H       = 0.05,
):
    """
    2x2 dot figure (rows: independent/dependent; cols: G=3/G=5).

    Reads {folder}/measure_sums.csv and saves to
        {folder}/figures/fig2x2{infix}_T{T}_SG{groupsize}_k..._rkt..._rB....pdf
    infix: '' (normal) / '_disturb' / '_smallgap'.

    kind:
        "normal"   -> DGP independent/dependent,             gamma 0.2-1.5
        "disturb"  -> DGP independent_noise/dependent_noise, gamma 0.2-1.5
        "smallgap" -> DGP independent/dependent,             gamma 0.48-0.64
    """
    if kind == "disturb":
        dgp_indep, dgp_dep = "independent_noise", "dependent_noise"
    else:
        dgp_indep, dgp_dep = "independent", "dependent"
    gamma_lo, gamma_hi = (0.48, 0.64) if kind == "smallgap" else (0.2, 1.5)

    csv_path = f"{folder}/measure_sums.csv"
    CONFIGS  = [(dgp_indep, 3), (dgp_indep, 5), (dgp_dep, 3), (dgp_dep, 5)]
    TITLES   = {dgp_indep: "cross-sectionally independent",
                dgp_dep:   "cross-sectionally dependent"}

    with plt.rc_context(DOT_RC):
        fig, axes = plt.subplots(
            2, 2, figsize=(12.0, 6.0), dpi=300,
            gridspec_kw={"width_ratios": [3, 5], "wspace": 0.10, "hspace": 0.45})
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.94)

        for idx, (dgp_model, G) in enumerate(CONFIGS):
            row, col = divmod(idx, 2)
            ax = axes[row, col]
            vgamma_0 = np.linspace(gamma_lo, gamma_hi, G)
            values = _load_2x2(csv_path, T, G, groupsize, k1_rate, dgp_model, r_kt)
            _draw_panel(ax, values, vgamma_0,
                        tick_fmt=lambda v: f"${v:g}$", right_pad=0.15,
                        title=TITLES[dgp_model], title_pad=5)

        _draw_inset_legend(axes[0, 0], fs=10)

        name = (f"fig2x2{_KIND_INFIX[kind]}_T{T}_SG{groupsize}_"
                f"k{_pct(k1_rate)}_rkt{_pct(r_kt)}_rB{_pct(r_H)}.pdf")
        save_path = Path(folder) / "figures" / name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"Saved -> {save_path}")
        plt.show()
        return fig, axes


# ── G1 (single-group) figure ────────────────────────────────────────────────

def _load_G1(csv_path, T, groupsize, k1_rate, dgp_model, r_kt, r_H, gamma_list):
    """For G=1: one row per gamma_single -> (3,3,len(gamma_list)) values."""
    df = pd.read_csv(csv_path)
    if dgp_model not in set(df["dgp_model"].unique()):
        raise ValueError(
            f"dgp_model={dgp_model!r} not in {csv_path}. "
            f"Available: {sorted(df['dgp_model'].unique())}.")
    mask = (
        (df["T"]          == T)         &
        (df["G"]          == 1)         &
        (df["group_size"] == groupsize) &
        (df["k1_rate"]    == k1_rate)   &
        (df["dgp_model"]  == dgp_model) &
        (np.isclose(df["r_kt"], r_kt))  &
        (np.isclose(df["r_H"],  r_H))
    )
    df_s = df[mask].copy()
    if df_s.empty:
        raise ValueError(
            f"No rows for T={T}, G=1, groupsize={groupsize}, k1_rate={k1_rate}, "
            f"dgp_model={dgp_model!r}, r_kt={r_kt}, r_H={r_H} in {csv_path}.")

    values = np.full((3, 3, len(gamma_list)), np.nan)
    for gi, gval in enumerate(gamma_list):
        rows = df_s[np.isclose(df_s["gamma_single"], gval)]
        if rows.empty:
            raise ValueError(f"No row for gamma_single={gval}.")
        row = rows.iloc[0]
        M   = int(row["Nsim"])

        def arr(col):
            return np.array(json.loads(row[col])[0])   # G=1 -> one group

        for e_idx, pref in enumerate(["g", "gh", "ind"]):
            b, s, r = _measures_from_sums(arr(f"s1_{pref}_byunit"),
                                          arr(f"s2_{pref}_byunit"), M)
            values[0, e_idx, gi] = b
            values[1, e_idx, gi] = s
            values[2, e_idx, gi] = r
    return values


def _draw_right_legend(leg_ax, fs=13):
    """Framed legend in a dedicated third axis (G1 figure)."""
    leg_ax.set_xlim(0, 1); leg_ax.set_ylim(0, 1)
    leg_ax.set_xticks([]); leg_ax.set_yticks([])
    for spine in leg_ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(0.8); spine.set_edgecolor("black")
    col_x = [0.46, 0.70, 0.94]
    row_y = [0.72, 0.44, 0.16]
    hdr_y = 0.88
    for e, name in enumerate(EST_NAMES):
        leg_ax.text(col_x[e], hdr_y, name, fontsize=fs - 3, ha="center", va="baseline")
    for m_idx in range(3):
        leg_ax.text(0.06, row_y[m_idx], MEASURE_NAMES[m_idx],
                    fontsize=fs, ha="left", va="center")
        for e_idx in range(3):
            leg_ax.scatter(col_x[e_idx], row_y[m_idx], s=6**2,
                           marker=MARKERS[e_idx], facecolors=_FC[m_idx],
                           edgecolors=_EC[m_idx], linewidths=MEW,
                           hatch=_HTC[m_idx], zorder=4)


def plot_G1(
    csv_path        = "sim_results_G1_noise/measure_sums.csv",
    T               = 1000,
    groupsize       = 100,
    k1_rate         = 0.12,
    dgp_model_indep = "independent_noise",
    dgp_model_dep   = "dependent_noise",
    r_kt            = 0.03,
    r_H             = 0.05,
    gamma_list      = (0.2, 0.5, 0.7, 1.0),
    save_path       = "sim_results_G1_noise/one_evi.png",
):
    """1x2 dot figure for the single-group case + a framed shared legend."""
    gamma_list = list(gamma_list)
    with plt.rc_context(DOT_RC):
        val_indep = _load_G1(csv_path, T, groupsize, k1_rate,
                             dgp_model_indep, r_kt, r_H, gamma_list)
        val_dep   = _load_G1(csv_path, T, groupsize, k1_rate,
                             dgp_model_dep, r_kt, r_H, gamma_list)

        fig, axes = plt.subplots(
            1, 3, figsize=(14, 4), dpi=300,
            gridspec_kw={"width_ratios": [5, 5, 1.4], "wspace": 0.05})
        fig.subplots_adjust(left=0.06, right=0.98, bottom=0.20, top=0.90)
        ax_indep, ax_dep, leg_ax = axes

        for ax, vals, ttl in [
            (ax_indep, val_indep, "cross-sectionally independent"),
            (ax_dep,   val_dep,   "cross-sectionally dependent"),
        ]:
            _draw_panel(ax, vals, gamma_list,
                        tick_fmt=lambda v: f"$\\gamma = {v:g}$", right_pad=0.02,
                        title=ttl, title_pad=6)

        ax_dep.set_yticklabels([])
        ax_dep.tick_params(axis="y", length=0)
        _draw_right_legend(leg_ax, fs=13)

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300)
            print(f"Saved -> {save_path}")
        plt.show()
        return fig, axes


# =============================================================================
# FAMILY B — method comparison (reads ONLY from results_compare_method/)
# =============================================================================

COMPARE_DIR = "results_compare_method"

_LINE_COLS = ["acc_iterative_unknown", "acc_seg_unknown"]
_LINE_LABELS = {
    "acc_iterative_unknown": "G unknown (Iterative)",
    "acc_seg_unknown":       "G unknown (Segmentation)",
}
_LINE_COLORS = {
    "acc_iterative_unknown": "#d73027",
    "acc_seg_unknown":       "#7F77DD",
}
_LINE_LS = {"A": "-", "B": "--"}
_LINE_LEGEND_ORDER = [
    ("A", "acc_iterative_unknown"), ("B", "acc_iterative_unknown"),
    ("A", "acc_seg_unknown"),       ("B", "acc_seg_unknown"),
]

_LINE_RC = {
    **_SERIF,
    "axes.unicode_minus": False,
    "font.size": 24, "axes.labelsize": 25, "axes.titlesize": 26,
    "xtick.labelsize": 24, "ytick.labelsize": 24, "legend.fontsize": 21,
    "lines.linewidth": 3.0, "axes.linewidth": 1.2,
    "xtick.major.width": 1.1, "ytick.major.width": 1.1,
    "xtick.major.size": 5.5, "ytick.major.size": 5.5,
    "figure.dpi": 300, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}


def plot_method_accuracy(csv_file, x_col, x_label, output_prefix,
                         g_values=(3, 4, 5), ymin=0.5, figsize=(20, 6.2)):
    """Accuracy (G unknown) vs x_col, one panel per g; two methods x two models."""
    plt.style.use("default")
    with plt.rc_context(_LINE_RC):
        df = pd.read_csv(csv_file)
        plot_df = (
            df.melt(id_vars=["model", "g", x_col], value_vars=_LINE_COLS,
                    var_name="method", value_name="accuracy")
            .groupby(["model", "g", x_col, "method"], as_index=False)["accuracy"].mean()
        )
        fig, axes = plt.subplots(1, len(g_values), figsize=figsize, sharey=True)
        if len(g_values) == 1:
            axes = [axes]
        fig.patch.set_facecolor("white")

        for ax, g_val in zip(axes, g_values):
            sub_g = plot_df[plot_df["g"] == g_val]
            for method in _LINE_COLS:
                for model in ["A", "B"]:
                    tmp = sub_g[(sub_g["method"] == method) &
                                (sub_g["model"] == model)].sort_values(x_col)
                    ax.plot(tmp[x_col], tmp["accuracy"], color=_LINE_COLORS[method],
                            linestyle=_LINE_LS[model],
                            linewidth=3.2 if model == "A" else 3.0,
                            solid_capstyle="round", dash_capstyle="round", zorder=3)
            ax.set_title(rf"$G={g_val}$", fontsize=28, pad=14, fontweight="normal")
            ax.set_xlabel(x_label, fontsize=27, labelpad=8)
            xticks = sorted(sub_g[x_col].unique())
            ax.set_xticks(xticks)
            if x_col == "rk":
                ax.set_xticklabels(
                    ["0" if abs(x) < 1e-12 else f"{x:.2f}" for x in xticks],
                    fontsize=23, rotation=45, ha="right")
            else:
                ax.set_xticklabels([f"{x:g}" for x in xticks], fontsize=23)
            ax.tick_params(axis="both", which="major", labelsize=23, direction="out")
            ax.set_facecolor("white")
            ax.set_axisbelow(True)
            ax.grid(axis="y", linestyle=":", linewidth=1.0, alpha=0.45, color="0.55")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_linewidth(1.2); ax.spines[s].set_color("0.25")

        axes[0].set_ylabel("Accuracy", fontsize=28, labelpad=10)
        ymax = plot_df["accuracy"].max()
        ypad = 0.06 * (ymax - ymin)
        for ax in axes:
            ax.set_ylim(ymin - ypad, ymax + ypad)

        legend_handles = [
            Line2D([0], [0], color=_LINE_COLORS[m], linestyle=_LINE_LS[md],
                   linewidth=3.2, label=f"Model {md}: {_LINE_LABELS[m]}")
            for md, m in _LINE_LEGEND_ORDER
        ]
        fig.legend(handles=legend_handles, loc="lower center",
                   bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False,
                   handlelength=3.5, handletextpad=0.9, columnspacing=2.0, fontsize=22)
        fig.tight_layout(rect=[0.02, 0.20, 1.00, 1.00])

        png_name = f"{output_prefix}.png"
        fig.savefig(png_name, dpi=300, bbox_inches="tight", facecolor="white")
        plt.show()
        plt.close(fig)
        print(f"Saved -> {png_name}")


_SG_METHODS = {
    "seg_known":         ("Segmentation (G known)",   "#2166ac", "-",  2.6),
    "seg_unknown":       ("Segmentation (G unknown)", "#74add1", "--", 2.4),
    "iterative_known":   ("Iterative (G known)",      "#d73027", "-",  2.6),
    "iterative_unknown": ("Iterative (G unknown)",    "#f46d43", "--", 2.4),
}
_SG_N_ALPHA   = {1000: 0.45, 3000: 1.00}
_SG_N_LW_MULT = {1000: 0.95, 3000: 1.10}

_SG_RC = {
    **_SERIF,
    "axes.unicode_minus": False,
    "font.size": 30, "axes.titlesize": 30, "axes.labelsize": 30,
    "xtick.labelsize": 30, "ytick.labelsize": 30, "legend.fontsize": 30,
    "axes.linewidth": 1.1,
    "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "xtick.major.size": 5, "ytick.major.size": 5,
}


def plot_smallgap_pergroup(
    csv_file    = f"{COMPARE_DIR}/results_segmentation_vary_q.csv",
    gamma_start = 0.48,
    gamma_end   = 0.64,
    n_values    = (1000, 3000),
    save_path   = f"{COMPARE_DIR}/fig_pergroup.png",
):
    """Per-group accuracy profiles, 1x4 panels (G=3/5 x independent/dependent)."""
    gammas = {3: np.linspace(gamma_start, gamma_end, 3),
              5: np.linspace(gamma_start, gamma_end, 5)}
    with plt.rc_context(_SG_RC):
        df = pd.read_csv(csv_file)

        fig1, axes = plt.subplots(1, 4, figsize=(16, 7.0), sharey=True)
        fig1.patch.set_facecolor("white")
        fig1.subplots_adjust(left=0.065, right=0.99, top=0.84, bottom=0.34, wspace=0.02)
        panel_configs = [(3, "independent"), (3, "dependent"),
                         (5, "independent"), (5, "dependent")]

        for ax, (g, model) in zip(axes, panel_configs):
            gvals = gammas[g]
            x = np.arange(1, g + 1)
            sub = df[(df.g == g) & (df.model == model)]
            for n_val in n_values:
                sn = sub[sub.n == n_val].mean(numeric_only=True)
                for mkey, (mlabel, mcolor, mls, mlw) in _SG_METHODS.items():
                    yvals = [sn[f"acc_{mkey}_g{i}"] for i in range(1, g + 1)]
                    ax.plot(x, yvals, color=mcolor, linestyle=mls,
                            linewidth=mlw * _SG_N_LW_MULT[n_val],
                            alpha=_SG_N_ALPHA[n_val], zorder=3)
            xtick_labels = [
                r"$\gamma_{g_" + str(i) + r"}$" + "\n" + f"{gvals[i-1]:.2f}"
                for i in range(1, g + 1)]
            ax.set_xticks(x)
            ax.set_xticklabels(xtick_labels, fontsize=21)
            ax.set_ylim(-0.02, 1.05)
            ax.set_xlim(0.65, g + 0.35)
            ax.set_axisbelow(True)
            ax.grid(axis="y", linestyle=":", linewidth=1.0, alpha=0.45, color="0.55")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color("0.25"); ax.spines[s].set_linewidth(1.1)
            ax.tick_params(axis="both", which="major", direction="out", labelsize=21)
            model_title = "Independent" if model == "independent" else "Dependent"
            ax.set_title(rf"$G={g}$ {model_title}", fontsize=25, pad=12, fontweight="normal")

        axes[0].set_ylabel("Accuracy", fontsize=26)

        legend_elems = [
            Line2D([0], [0], color=mcolor, linestyle=mls, linewidth=3.0, label=mlabel)
            for _, (mlabel, mcolor, mls, _mlw) in _SG_METHODS.items()
        ]
        legend_elems += [
            Line2D([0], [0], color="0.35", linewidth=3.0, alpha=0.45, label=r"$T=1000$"),
            Line2D([0], [0], color="0.35", linewidth=3.0, alpha=1.00, label=r"$T=3000$"),
        ]
        fig1.legend(handles=legend_elems, loc="lower center", ncol=3, frameon=False,
                    bbox_to_anchor=(0.5, 0.045), columnspacing=1.8, handlelength=2.8,
                    handletextpad=0.8, fontsize=20)

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig1.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"Saved -> {save_path}")
        plt.show()
        return fig1, axes


def plot_smallgap_heatmap(
    csv_file  = f"{COMPARE_DIR}/results_segmentation_vary_q.csv",
    g_values  = (3, 5),
    save_path = f"{COMPARE_DIR}/fig_heatmap.png",
):
    """Overall-accuracy heatmaps, one column per G."""
    mkeys = ["seg_known", "seg_unknown", "iterative_known", "iterative_unknown"]
    col_labels = ["Segmentation\n G known", "Segmentation\n G unknown",
                  "Iterative\n G known", "Iterative\n G unknown"]
    with plt.rc_context({**_SERIF, "axes.unicode_minus": False}):
        df = pd.read_csv(csv_file)

        def build_panel_data(sub_df):
            sub_df = sub_df.copy()
            sub_df["_order"] = sub_df["model"].map({"independent": 0, "dependent": 1})
            sub_df = sub_df.sort_values(by=["n", "_order", "q"],
                                        ascending=[True, True, True]).reset_index(drop=True)
            labels, data = [], []
            for _, row in sub_df.iterrows():
                prefix = "Independent" if row.model == "independent" else "Dependent"
                labels.append(f"{prefix} ({int(row.q)}, {int(row.n)})")
                data.append([row[f"acc_{m}"] for m in mkeys])
            return labels, np.array(data)

        panels = [build_panel_data(df[df.g == g].reset_index(drop=True)) for g in g_values]
        n_rows = max(len(p[0]) for p in panels)

        fig2, axes2 = plt.subplots(1, 2, figsize=(11.5, 0.5 * n_rows))
        fig2.subplots_adjust(left=0.28, right=0.92, bottom=0.18, top=0.88, wspace=0.08)

        im = None
        for idx, (ax, g, (labels, arr)) in enumerate(zip(axes2, g_values, panels)):
            im = ax.imshow(arr, aspect="auto", cmap="Blues", vmin=0.35, vmax=1.0,
                           origin="upper")
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks(np.arange(-0.5, arr.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, arr.shape[0], 1), minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
            ax.tick_params(which="minor", bottom=False, left=False)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    v = arr[i, j]
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=14,
                            color="white" if v >= 0.58 else "#222222", fontweight="semibold")
            ax.set_xticks(range(arr.shape[1]))
            ax.set_xticklabels(col_labels, fontsize=10)
            ax.tick_params(axis="x", length=0, pad=2)
            ax.tick_params(axis="y", length=0, pad=6)
            ax.set_title(f"$G={g}$", fontsize=14, pad=6)
            if idx == 0:
                ax.set_yticks(range(len(labels)))
                ax.set_yticklabels(labels, fontsize=12)
                ax.text(-0.21, 1.01, r"Model $(S_G,\, T\,)$", transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=14)
            else:
                ax.set_yticks([])

        cbar = fig2.colorbar(im, ax=axes2, fraction=0.028, pad=0.04)
        cbar.set_label("Accuracy", fontsize=12)
        cbar.ax.tick_params(labelsize=12, length=2)
        cbar.outline.set_visible(False)

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig2.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"Saved -> {save_path}")
        plt.show()
        return fig2, axes2

# =============================================================================
# FAMILY C — empirical rejection rate (size/power) curves
# =============================================================================
# Reads the rejection_simulation.py output (columns Group, T, Effect Size, Power)
# and draws one panel per group: empirical rejection rate vs Δγ, one line per T.


_REJ_COLORS = ["#2166ac", "#1D9E75", "#f46d43"]
_REJ_BLUE = _REJ_COLORS[0]
_REJ_RED  = _REJ_COLORS[2]

_REJ_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.titlesize": 22,
    "axes.labelsize": 21,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 18,
    "axes.linewidth": 1.2,
}


def plot_rejection_rate(
    csv_path="t_test_G3_size100.csv",
    out_path="t_test_G3_size100_1x3.png",
    sig_level=0.05,
):
    """
    1×len(groups) panel of empirical rejection rate vs Δγ, one line per T.
    Reads a rejection_simulation.py CSV (Group, T, Effect Size, Power).
    """
    with plt.rc_context(_REJ_RC):
        df = pd.read_csv(csv_path)
        # Expected columns: Group, T, Effect Size, Power
        groups = sorted(df["Group"].unique())
        Ts = sorted(df["T"].unique())
        color_map = {Ts[0]: _REJ_RED, Ts[1]: _REJ_BLUE}

        fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9), sharey=True)

        for ax, group in zip(axes, groups):
            sub = df[df["Group"] == group]
            for T in Ts:
                d = sub[sub["T"] == T].sort_values("Effect Size")
                ax.plot(
                    d["Effect Size"], d["Power"],
                    color=color_map[T], marker="o", markersize=6.2,
                    linewidth=3.0, label=rf"$T = {T}$")
            ax.axhline(sig_level, color="0.35", linestyle="--", linewidth=2.0,
                       alpha=0.8, label="5% level")
            ax.set_xlabel(r"$\Delta \gamma$")
            ax.set_xlim(-0.083, 0.083)
            ax.set_ylim(0, 1.03)
            ax.xaxis.set_major_locator(MultipleLocator(0.04))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.grid(axis="y", color="0.82", linewidth=0.8, alpha=0.55)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("0.25")
            ax.spines["bottom"].set_color("0.25")
            ax.spines["left"].set_linewidth(1.2)
            ax.spines["bottom"].set_linewidth(1.2)
            ax.tick_params(axis="both", which="major", length=5, width=1.1,
                           color="0.25")

        legend_handles = [
            Line2D([0], [0], color=_REJ_BLUE, marker="o", markersize=7,
                   linewidth=3.2, label=rf"$T = {Ts[0]}$"),
            Line2D([0], [0], color=_REJ_RED, marker="o", markersize=7,
                   linewidth=3.2, label=rf"$T = {Ts[1]}$"),
            Line2D([0], [0], color="0.35", linestyle="--", linewidth=2.4,
                   label="5% level"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=3,
                   frameon=False, bbox_to_anchor=(0.5, -0.03),
                   handlelength=2.8, columnspacing=2.0)
        fig.subplots_adjust(left=0.075, right=0.995, top=0.86, bottom=0.27,
                            wspace=0.055)

        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.show()
        return fig, axes


# =============================================================================
# __main__  
# =============================================================================

if __name__ == "__main__":
    # ---- Family A: 2x2 dot figures, one per sim_results_* folder ------------
    for r_kt, r_H in product([0.02, 0.03, 0.04], [0.05, 0.07]):
        plot_2x2("sim_results_main",      kind="normal",   T=1000, groupsize=100, r_kt  = r_kt, r_H  = r_H)
    for T, groupsize in product([1000, 3000], [100, 300]):
        plot_2x2("sim_results_small_gap", kind="smallgap", T=T,  groupsize=groupsize, r_kt  = 0.03, r_H  = 0.05)
    plot_2x2("sim_results_perturbed", kind="disturb",  T=1000, groupsize=100, r_kt  = 0.03, r_H  = 0.05)

    # ---- Family A: single-group figure --------------------------------------
    plot_G1(csv_path="sim_results_G1_noise/measure_sums.csv",
            save_path="sim_results_G1_noise/one_evi.png")

    # ---- Family B: method comparison (results_compare_method/) --------------
    plot_method_accuracy(
        csv_file=f"{COMPARE_DIR}/results_iterative_vary_q.csv",
        x_col="q", x_label=r"$S_G$",
        output_prefix=f"{COMPARE_DIR}/accuracy_by_q_g_modelsAB_R1",
        g_values=(3, 4, 5), ymin=0.5, figsize=(20, 6.2))
    plot_method_accuracy(
        csv_file=f"{COMPARE_DIR}/results_iterative_vary_delta.csv",
        x_col="Delta", x_label=r"$\Delta$",
        output_prefix=f"{COMPARE_DIR}/accuracy_by_Delta_g_modelsAB_R2",
        g_values=(3, 4, 5), ymin=0.2, figsize=(20, 6.2))
    plot_smallgap_pergroup()
    plot_smallgap_heatmap()
    # ---- Family C: empirical rejection rate (rejection_simulation.py output) -
    plot_rejection_rate(
        csv_path="Rejection_G3_size100/t_test_G3_size100.csv",
        out_path="Rejection_G3_size100/t_test_G3_size100_1x3.png")
