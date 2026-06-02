#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
empirics.py
==========
Empirical application: grouped extreme-value-index (EVI) analysis of ECA&D
winter (DJF) rainfall, comparing period 1 (1950-1984) and period 2 (1985-2020).

Pipeline
--------
1. clean_raw_data()      (OPTIONAL) 
raw ECA&D text files -> application_data.nc + station_doc.pkl.  
                         Only needed once; skipped if the two files already exist.  
                         Requires the raw `ECA_blend_rr/`, `stations.txt`, `ECA_blend_source_rr.txt`.
2. load_clean_data()     read application_data.nc + station_doc.pkl.
3. analysis loop         decluster -> group via structural breaks -> two-sample
                         tests (iid + spatially-corrected) -> heatmaps + tables.

Shared structural-break / Hill machinery is imported from common.py:
    fHill, ssr, parti, datingtrimming, ssrnul
"""

import os
import datetime
import pickle

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import norm

import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter, FixedLocator

from common import fHill, datingtrimming, ssrnul

# Shared serif style (matches make_figures.py); applied to the heatmap figure.
_SERIF = {
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
}

# =============================================================================
# PART 1 — OPTIONAL DATA CLEANING  (raw ECA&D -> application_data.nc + .pkl)
# =============================================================================
# These run only if you have the raw inputs and the cleaned files are missing.  
# If application_data.nc and station_doc.pkl already exist, skip cleaning .

DEFAULT_RAW = dict(
    station_info = "stations.txt",
    folder_path  = "ECA_blend_rr",
    datapath     = "ECA_blend_source_rr.txt",
    target_start  = datetime.date(1950, 1, 1),
    target_end    = datetime.date(2020, 12, 31),
    period1_end   = datetime.date(1984, 12, 31),
    period2_start = datetime.date(1985, 1, 1),
    percent       = 0.12,     # k1 fraction for the moment-estimator screen
)


def _dms_to_decimal(dms_str):
    """'+50:51:00' -> 50.85 decimal degrees."""
    sign = -1 if "-" in dms_str else 1
    dms_str = dms_str.replace("+", "").replace("-", "")
    d, m, s = (float(x) for x in dms_str.split(":"))
    return sign * (d + m / 60 + s / 3600)


def _fMomentEstimator(sample, k):
    """Moment (Dekkers-Einmahl-de Haan) EVI estimate; used only for screening."""
    s = np.sort(sample)
    log_ratios = np.log(s[-k:]) - np.log(s[-k - 1])
    M1 = np.mean(log_ratios)
    M2 = np.mean(log_ratios ** 2)
    return M1 + 1 - 0.5 / (1 - M1 ** 2 / M2)


def _winter_only_calendar(start_date, end_date):
    """DatetimeIndex of Dec/Jan/Feb days within [start_date, end_date]."""
    full = pd.date_range(start=start_date, end=end_date, freq="D")
    return full[full.month.isin([12, 1, 2])]


def get_station_info(station_info, folder_path):
    """Parse stations.txt; return DataFrame [STAID, STANAME, CN, LAT, LON, HGHT]."""
    doc = pd.read_csv(os.path.join(folder_path, station_info), skiprows=range(0, 17))
    doc.columns = doc.columns.str.strip().str.replace(" ", "_")
    doc["LAT"] = doc["LAT"].astype(str).apply(_dms_to_decimal)
    doc["LON"] = doc["LON"].astype(str).apply(_dms_to_decimal)
    return doc


def get_valid_stations(station_doc, datapath, folder_path,
                       target_start, period2_start, period1_end, target_end):
    """
    Build DJF (winter-only) rainfall panels for both periods.

    Returns final_staids, winter_p1, winter_p2, winter_dates_p1, winter_dates_p2
    where the two winter_* frames have a 'station_id' column plus one column per
    DJF date.
    """
    winter_dates_p1 = _winter_only_calendar(target_start, period1_end)
    winter_dates_p2 = _winter_only_calendar(period2_start, target_end)

    # --- stations spanning the whole target window (from sources file) --------
    stations = {}
    with open(datapath, "r", encoding="utf-8") as f:
        data_started = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("STAID, SOUID"):
                data_started = True
                continue
            if not data_started:
                continue
            parts = [p.strip() for p in line.split(",")]
            try:
                staid = int(parts[0])
                sd = datetime.datetime.strptime(parts[8], "%Y%m%d").date()
                ed = datetime.datetime.strptime(parts[9], "%Y%m%d").date()
            except (ValueError, IndexError):
                continue
            if staid not in stations:
                stations[staid] = {"min_start": sd, "max_end": ed}
            else:
                stations[staid]["min_start"] = min(stations[staid]["min_start"], sd)
                stations[staid]["max_end"]   = max(stations[staid]["max_end"], ed)

    qualifying = {
        sid for sid, dr in stations.items()
        if dr["min_start"] <= target_start and dr["max_end"] >= target_end
    }

    data_p1, data_p2, final_staids = [], [], []
    for fname in os.listdir(folder_path):
        if not (fname.startswith("RR_STAID") and fname.endswith(".txt")):
            continue
        try:
            staid = int(fname.replace("RR_STAID", "").replace(".txt", ""))
        except ValueError:
            continue
        if staid not in qualifying:
            continue

        path = os.path.join(folder_path, fname)
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        header_idx = next((i for i, L in enumerate(lines)
                           if L.strip().startswith("STAID, SOUID")), None)
        if header_idx is None:
            continue

        df = pd.read_csv(path, skiprows=header_idx)
        df.columns = df.columns.str.strip()
        df["DATE"] = pd.to_datetime(df["DATE"].astype(str).str.strip(),
                                    format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["DATE"])
        df["Q_RR"] = pd.to_numeric(df["Q_RR"].astype(str).str.strip(), errors="coerce")
        df["RR"]   = pd.to_numeric(df["RR"].astype(str).str.strip(), errors="coerce")

        # valid (Q_RR == 0) winter measurements only
        df = df[(df["Q_RR"] == 0) & (df["DATE"].dt.month.isin([12, 1, 2]))]
        ser = df.set_index("DATE")["RR"]
        series_p1 = ser.reindex(winter_dates_p1)
        series_p2 = ser.reindex(winter_dates_p2)

        # geographic filter (Europe box) using the station catalogue
        loc = station_doc.loc[station_doc["STAID"] == staid, ["LAT", "LON"]]
        if loc.empty:
            continue
        lat, lon = loc.iloc[0]
        if not (35 <= lat <= 72 and -10 <= lon <= 21):
            continue

        # require enough valid winter days in both periods
        if series_p1.count() > 3000 and series_p2.count() > 3000:
            final_staids.append(staid)
            data_p1.append({"station_id": staid, **series_p1.to_dict()})
            data_p2.append({"station_id": staid, **series_p2.to_dict()})

    winter_p1 = pd.DataFrame(data_p1).reindex(columns=["station_id", *winter_dates_p1])
    winter_p2 = pd.DataFrame(data_p2).reindex(columns=["station_id", *winter_dates_p2])
    return final_staids, winter_p1, winter_p2, winter_dates_p1, winter_dates_p2


def get_intersect_mom(winter_p1, winter_p2, percent):
    """Keep stations with a positive moment-EVI in BOTH periods (common ids)."""
    def _screen(w):
        w = w.copy()
        data = w.drop(columns=["station_id"])          # data columns only
        vTi  = data.notna().sum(axis=1).values
        vk1  = (percent * vTi).astype(int)
        w["gamma_mom"] = [
            _fMomentEstimator(row.values[~pd.isna(row.values)], vk1[i])
            for i, (_, row) in enumerate(data.iterrows())
        ]
        return w[w["gamma_mom"] > 0].dropna(subset=["gamma_mom"])

    p1 = _screen(winter_p1)
    p2 = _screen(winter_p2)
    common = set(p1["station_id"]).intersection(p2["station_id"])
    p1 = p1[p1["station_id"].isin(common)].drop(columns=["gamma_mom"])
    p2 = p2[p2["station_id"].isin(common)].drop(columns=["gamma_mom"])
    return p1, p2


def clean_raw_data(out_nc="application_data.nc", out_pkl="station_doc.pkl",
                   raw=None, overwrite=False):
    """
    Build application_data.nc and station_doc.pkl from the raw ECA&D files.

    Streamlined vs. the original script: no redundant save/reload roundtrips,
    the temporary 'gamma_mom' screening column is dropped before saving, and
    the two cleaned panels are written straight to one NetCDF dataset.

    Skips work if both files already exist (unless overwrite=True).
    """
    raw = {**DEFAULT_RAW, **(raw or {})}

    if (not overwrite and os.path.exists(out_nc) and os.path.exists(out_pkl)):
        print(f"[clean_raw_data] {out_nc} and {out_pkl} already exist - skipping.")
        return

    if not os.path.isdir(raw["folder_path"]):
        raise FileNotFoundError(
            f"Raw data folder {raw['folder_path']!r} not found. "
            f"Cleaning needs the raw ECA&D files; if you already have "
            f"{out_nc} and {out_pkl}, you can skip clean_raw_data() entirely.")

    # station catalogue
    station_doc = get_station_info(raw["station_info"], raw["folder_path"])
    with open(out_pkl, "wb") as f:
        pickle.dump(station_doc, f)

    # winter panels for both periods
    _, winter_p1, winter_p2, _, _ = get_valid_stations(
        station_doc, raw["datapath"], raw["folder_path"],
        raw["target_start"], raw["period2_start"],
        raw["period1_end"], raw["target_end"])

    # screen to common positive-EVI stations
    p1, p2 = get_intersect_mom(winter_p1, winter_p2, raw["percent"])

    # write both panels to one NetCDF dataset (stations aligned across periods)
    p1_vals = p1.drop(columns=["station_id"])
    p2_vals = p2.drop(columns=["station_id"])
    ds = xr.Dataset(
        {
            "winter_p1_sub": (["station", "day_p1"], p1_vals.to_numpy()),
            "winter_p2_sub": (["station", "day_p2"], p2_vals.to_numpy()),
        },
        coords={
            "station": p1["station_id"].to_numpy(),
            "day_p1": pd.to_datetime(p1_vals.columns.values),
            "day_p2": pd.to_datetime(p2_vals.columns.values),
        },
    )
    ds.to_netcdf(out_nc)
    
    print(f"[clean_raw_data] wrote {out_nc} ({ds.sizes['station']} stations) and {out_pkl}.")
    print(ds.sizes["station"])   


def load_clean_data(nc_path="application_data.nc", pkl_path="station_doc.pkl"):
    """Read the cleaned NetCDF + station catalogue into tidy DataFrames."""
    ds = xr.open_dataset(nc_path)

    def _to_df(var):
        df = ds[var].to_pandas()
        df.index.name = "station_id"
        return df.reset_index()

    p1_df = _to_df("winter_p1_sub")
    p2_df = _to_df("winter_p2_sub")
    with open(pkl_path, "rb") as f:
        station_doc = pickle.load(f)
    return p1_df, p2_df, station_doc


# =============================================================================
# PART 2 — ANALYSIS HELPERS
# =============================================================================

def keep_every_second_day(df, start=0):
    """Decluster by setting every second date column to NaN."""
    out = df.copy()
    date_cols = [c for c in out.columns if c != "station_id"]
    cols_to_nan = date_cols[1 - start::2]
    out.loc[:, cols_to_nan] = np.nan
    out.columns = [
        c.strftime("%Y-%m-%d") if isinstance(c, pd.Timestamp) else c
        for c in out.columns
    ]
    return out


def get_EVI_k(df, rk, rkt):
    """
    Two-stage Hill EVIs per station.

    Stage 1 (fraction rk) gives the ordering EVI used for grouping;
    stage 2 (fraction rkt) gives the EVI used for the group estimates.
    Returns (vogamma_s1, vogamma_s2, df_ordered, vkt).

    The Hill estimator is computed on the rainfall columns ONLY; station_id and
    any temporary screening column are excluded so they never enter the tail.
    """
    df = df.reset_index(drop=True)
    data = df.drop(columns=["station_id", "gamma_mom", "gamma_H"], errors="ignore")
    vTi  = data.notna().sum(axis=1).values

    def hill_per_row(frac):
        vk = (frac * vTi).astype(int)
        return np.array([
            fHill(row.values[~pd.isna(row.values)], vk[i])
            for i, (_, row) in enumerate(data.iterrows())
        ])

    # stage 1: large k -> ordering EVI
    df = df.copy()
    df["gamma_H"] = hill_per_row(rk)
    order = df.dropna(subset=["gamma_H"]).sort_values(by="gamma_H").index
    df_ordered = df.loc[order].reset_index(drop=True)
    vogamma_s1 = df_ordered["gamma_H"].values

    # stage 2: small k-tilde -> group-estimate EVI (recompute on same rows/order)
    data_ord = df_ordered.drop(columns=["station_id", "gamma_mom", "gamma_H"],
                               errors="ignore")
    vTi_ord  = data_ord.notna().sum(axis=1).values
    vkt = (rkt * vTi_ord).astype(int)
    df_ordered["gamma_H"] = np.array([
        fHill(row.values[~pd.isna(row.values)], vkt[i])
        for i, (_, row) in enumerate(data_ord.iterrows())
    ])
    vogamma_s2 = df_ordered["gamma_H"].values
    return vogamma_s1, vogamma_s2, df_ordered, vkt


def get_Ghat(SSR, glob, ssrzero, threshold):
    """Elbow rule: smallest G whose standardised SSR drop is below `threshold`."""
    standard_ssr = np.diff(SSR) / (np.array(glob.flatten().tolist()) - ssrzero)
    return np.where(np.abs(standard_ssr) < threshold)[0][0] + 1


def weight_gamma_j(vgamma, vtilde_k, index):
    """Group EVI estimate per group (equal weighting within group)."""
    return [np.mean(vgamma[index[i]:index[i + 1]]) for i in range(len(index) - 1)]


def format_stat(stat, p):
    """Format a test statistic with significance stars."""
    if p < 0.01:
        stars = "***"
    elif p < 0.05:
        stars = "**"
    elif p < 0.1:
        stars = "*"
    else:
        stars = ""
    return f"{stat:.2f}{stars}"


def get_mR11(df, station_doc, h=None, id_col="station_id", qu=0.05):
    """
    Pairwise upper-tail dependence R_{il}(1,1) (eq. 4.2) with a distance kernel.

    Returns (R, dist): R is a dict {h_val: DataFrame} if `h` is a list, a single
    DataFrame if `h` is scalar/None; dist is the haversine distance matrix.
    """
    def haversine_matrix(lat_deg, lon_deg):
        R_earth = 6371.0
        lat = np.radians(lat_deg)
        lon = np.radians(lon_deg)
        dlat = lat[:, None] - lat[None, :]
        dlon = lon[:, None] - lon[None, :]
        a = np.clip(
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2,
            0.0, 1.0,
        )
        return 2.0 * R_earth * np.arcsin(np.sqrt(a))

    data = df.set_index(id_col)                 # (N_stations, T_time)

    # pseudo-observations U_{i,t} = rank / (T_i + 1), NaN-preserving
    T_i = data.notna().sum(axis=1)
    U = data.rank(axis=1, method="average", na_option="keep").div(T_i + 1, axis=0)

    E = (U > 1 - qu).to_numpy(dtype=np.float32)   # upper-tail exceedance indicator
    M = data.notna().to_numpy(dtype=np.float32)   # observed indicator

    C = E @ E.T                                   # co-exceedance counts
    N = M @ M.T                                   # co-observed counts
    with np.errstate(divide="ignore", invalid="ignore"):
        R_base = C / (qu * N)
    R_base = np.where(np.isfinite(R_base), R_base, np.nan)
    np.fill_diagonal(R_base, 1.0)

    coords = (station_doc[["STAID", "LAT", "LON"]]
              .rename(columns={"STAID": id_col})
              .set_index(id_col)
              .reindex(data.index))
    lat = pd.to_numeric(coords["LAT"], errors="coerce").to_numpy()
    lon = pd.to_numeric(coords["LON"], errors="coerce").to_numpy()
    dist_mat = haversine_matrix(lat, lon)
    np.fill_diagonal(dist_mat, 0.0)
    dist_df = pd.DataFrame(dist_mat, index=data.index, columns=data.index)

    def _apply_kernel(R, h_val):
        return pd.DataFrame(R * np.exp(-dist_mat / h_val),
                            index=data.index, columns=data.index)

    if h is None:
        return pd.DataFrame(R_base, index=data.index, columns=data.index), dist_df
    if np.isscalar(h):
        return _apply_kernel(R_base, h), dist_df
    return {h_val: _apply_kernel(R_base, h_val) for h_val in h}, dist_df


def tTest(G, index_p1, index_p2, vgamma_l_p1, vgamma_l_p2, vkt_p1, vkt_p2):
    """Two-sample group-EVI test under the iid (independent) variance."""
    test_stats = []
    for i in range(G):
        var_p1 = vgamma_l_p1[i] ** 2 * np.sum(1 / vkt_p1[index_p1[i]:index_p1[i + 1]]) \
            / (index_p1[i + 1] - index_p1[i]) ** 2
        var_p2 = vgamma_l_p2[i] ** 2 * np.sum(1 / vkt_p2[index_p2[i]:index_p2[i + 1]]) \
            / (index_p2[i + 1] - index_p2[i]) ** 2
        test_stats.append((vgamma_l_p1[i] - vgamma_l_p2[i]) / np.sqrt(var_p1 + var_p2))
    p_value = 2 * (1 - norm.cdf(np.abs(test_stats)))
    return test_stats, p_value


def tTest_withR(G, index_p1, index_p2, vgamma_l_p1, vgamma_l_p2,
                vkt_p1, vkt_p2, R_p1, R_p2):
    """Two-sample group-EVI test with the spatial-dependence-corrected variance."""
    test_stats = []
    R_p1 = np.nan_to_num(np.asarray(R_p1, dtype=float), nan=0.0)
    R_p2 = np.nan_to_num(np.asarray(R_p2, dtype=float), nan=0.0)
    for i in range(G):
        s1, e1 = index_p1[i], index_p1[i + 1]
        s2, e2 = index_p2[i], index_p2[i + 1]
        k1 = np.asarray(vkt_p1[s1:e1], dtype=float)
        k2 = np.asarray(vkt_p2[s2:e2], dtype=float)
        m1, m2 = len(k1), len(k2)

        R1 = R_p1[s1:e1, s1:e1]
        R2 = R_p2[s2:e2, s2:e2]
        fac1 = np.sqrt(np.outer(k1, k1))
        fac2 = np.sqrt(np.outer(k2, k2))
        u1 = np.triu_indices(m1, k=1)
        u2 = np.triu_indices(m2, k=1)
        cross1 = np.sum(fac1[u1] * R1[u1])
        cross2 = np.sum(fac2[u2] * R2[u2])

        r1 = np.sum(k1) / np.sqrt(np.sum(k1) + 2 * cross1)
        r2 = np.sum(k2) / np.sqrt(np.sum(k2) + 2 * cross2)
        var_p1 = vgamma_l_p1[i] ** 2 / (r1 ** 2)
        var_p2 = vgamma_l_p2[i] ** 2 / (r2 ** 2)
        test_stats.append((vgamma_l_p1[i] - vgamma_l_p2[i]) / np.sqrt(var_p1 + var_p2))

    test_stats = np.asarray(test_stats)
    p_value = 2 * (1 - norm.cdf(np.abs(test_stats)))
    return test_stats, p_value


def fGetResults(df, Rk, Rtk, tau):
    """Group one period: two-stage EVIs -> structural-break grouping -> Ghat."""
    vogamma_s1, vogamma_s2, winter_ordered, vkt = get_EVI_k(df, Rk, Rtk)
    y = vogamma_s1.reshape(-1, 1)
    z = np.ones((len(vogamma_s1), 1))
    glob, datevec, _ = datingtrimming(y, z, 3, 10, 1, len(vogamma_s1), Rk)
    ssrzero = ssrnul(y, z)
    SSR = [ssrzero] + glob.flatten().tolist()
    Ghat = get_Ghat(SSR, glob, ssrzero, tau)
    return winter_ordered, vkt, Ghat, datevec, vogamma_s2


# ----------------------------------------------------------------------------
# HEATMAP 
# ----------------------------------------------------------------------------
def fHeatMap(G, vgamma, datevec, df, station_doc, vtilde_k):
    """
    Map of per-group EVI estimates over the station network.

    G          : int, number of groups (>2)
    vgamma     : per-station EVI vector used for the group estimator
    datevec    : matrix of break locations (columns indexed by G - 2)
    df         : station frame ordered by the stage-1 EVI (provides station_id)
    station_doc: station catalogue providing LAT/LON
    vtilde_k   : per-station stage-2 sample fractions (group-estimate weights)
    """
    # break locations -> contiguous group index ranges
    vl    = [int(l) for l in datevec[:, G - 2] if l != 0]
    index = [0] + vl + [len(vgamma)]
    vgamma_l = [x.sum() for x in weight_gamma_j(vgamma, vtilde_k, index)]

    # gather (LAT, LON, group EVI) for every station, by group
    staid_to_loc = station_doc.set_index("STAID")[["LAT", "LON"]]
    records = []
    for g in range(len(index) - 1):
        for sid in df["station_id"].values[index[g]:index[g + 1]]:
            if sid in staid_to_loc.index:
                lat, lon = staid_to_loc.loc[sid, ["LAT", "LON"]]
                records.append({"Group": g, "StationID": sid,
                                "LAT": lat, "LON": lon, "gamma_g": vgamma_l[g]})
            else:
                print(f"Station '{sid}' not found")
    coords = pd.DataFrame(records)

    # map extent + aspect-matched figure size (same formula as before)
    min_lat, max_lat = coords["LAT"].min(), coords["LAT"].max()
    min_lon, max_lon = coords["LON"].min(), coords["LON"].max()
    extent = [min_lon - 1, max_lon + 1, min_lat - 1, max_lat + 1]
    aspect_ratio = (extent[1] - extent[0]) / (extent[3] - extent[2])
    base_height = 10
    figsize = (base_height * aspect_ratio, base_height)

    cmap = plt.get_cmap("turbo")
    norm = plt.Normalize(vmin=0.3, vmax=0.57)

    with plt.rc_context(_SERIF):
        plt.figure(figsize=figsize, dpi=300)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor="black")
        ax.add_feature(cfeature.BORDERS, linestyle="-", linewidth=0.8, edgecolor="black")

        plt.scatter(coords["LON"], coords["LAT"], c=coords["gamma_g"],
                    cmap=cmap, norm=norm, s=20, alpha=0.9,
                    transform=ccrs.PlateCarree())

        legend_elements = [
            plt.Line2D([0], [0], marker="o", color="w", label=f"{val:.3f}",
                       markerfacecolor=cmap(norm(val)), markersize=10)
            for val in sorted(coords["gamma_g"].unique())
        ]
        ax.legend(handles=legend_elements, fontsize=18, loc="upper left", frameon=True)
        plt.show()
# ----------------------------------------------------------------------------


def fGetAnalysis(df_full, df1, df2, Rk, Rtk, tau, station_doc):
    """
    Full empirical analysis for one (Rk, Rtk, tau):
      group on the full sample, estimate per-period group EVIs, draw the two
      heatmaps, and build the iid + spatially-corrected two-sample test table.
    """
    res = []
    winter_ordered, vkt, Ghat, datevec, vogamma_s2 = fGetResults(df_full, Rk, Rtk, tau)
    print("Done: Grouping using full sample")
    vl = [int(l) for l in datevec[:, Ghat - 2] if l != 0]
    index = [0] + vl + [len(vogamma_s2)]

    winter_p1_ordered, vkt_p1, _, _, vogamma_p1_s2 = fGetResults(df1, Rk, Rtk, tau)
    winter_p2_ordered, vkt_p2, _, _, vogamma_p2_s2 = fGetResults(df2, Rk, Rtk, tau)

    # spatial-dependence matrices at several kernel bandwidths
    h = [22.5, 45, 67.5,90]
    R_p1, _ = get_mR11(df1, station_doc=station_doc, h=h, id_col="station_id", qu=Rtk)
    R_p2, _ = get_mR11(df2, station_doc=station_doc, h=h, id_col="station_id", qu=Rtk)
    print("Done: Calculating R11")

    # ------------------------------------------------------------------
    # Align all period-specific objects to the FULL-SAMPLE station order.
    # Group membership is determined only by the full-sample ordering and
    # full-sample break positions; p1/p2 only provide the estimates/colors.
    # ------------------------------------------------------------------
    full_order = winter_ordered["station_id"].to_numpy()

    # Period-specific EVIs, keyed by station_id, then reordered to full sample.
    vog_p1_re = (
        pd.Series(vogamma_p1_s2, index=winter_p1_ordered["station_id"])
          .reindex(full_order)
    )
    vog_p2_re = (
        pd.Series(vogamma_p2_s2, index=winter_p2_ordered["station_id"])
          .reindex(full_order)
    )

    # Period-specific k-tilde values, reordered to the same full-sample order.
    vkt_p1_re = (
        pd.Series(vkt_p1, index=winter_p1_ordered["station_id"])
          .reindex(full_order)
          .to_numpy()
    )
    vkt_p2_re = (
        pd.Series(vkt_p2, index=winter_p2_ordered["station_id"])
          .reindex(full_order)
          .to_numpy()
    )

    # Period dataframes reordered to full-sample station order for heatmaps.
    winter_p1_for_map = (
        winter_p1_ordered
          .set_index("station_id")
          .reindex(full_order)
          .reset_index()
    )
    winter_p2_for_map = (
        winter_p2_ordered
          .set_index("station_id")
          .reindex(full_order)
          .reset_index()
    )

    # Spatial-dependence matrices must also be in full-sample station order.
    R_p1_re = {
        h_val: R_p1[h_val].reindex(index=full_order, columns=full_order)
        for h_val in R_p1.keys()
    }
    R_p2_re = {
        h_val: R_p2[h_val].reindex(index=full_order, columns=full_order)
        for h_val in R_p2.keys()
    }

    # Group averages now use full-sample groups and period-specific estimates.
    vgamma_l_p1 = weight_gamma_j(vog_p1_re.values, vkt_p1_re, index)
    vgamma_l_p2 = weight_gamma_j(vog_p2_re.values, vkt_p2_re, index)

    # Heatmaps: same group membership, different period-specific EVIs.
    fHeatMap(Ghat, vog_p1_re.values, datevec,
             winter_p1_for_map, station_doc, vkt_p1_re)
    fHeatMap(Ghat, vog_p2_re.values, datevec,
             winter_p2_for_map, station_doc, vkt_p2_re)

    # test table: iid row (h=0) + one row per spatial bandwidth
    L = 4500
    table_rows = []
    test_stats, p_value = tTest(
        Ghat, index, index, vgamma_l_p1, vgamma_l_p2, vkt_p1_re, vkt_p2_re
    )
    row = {"h": 0}
    for i, (s, p) in enumerate(zip(test_stats, p_value), start=1):
        row[f"group{i}"] = format_stat(s, p)
    table_rows.append(row)

    for h_val in R_p1.keys():
        test_stats_R, p_value_R = tTest_withR(
            Ghat, index, index, vgamma_l_p1, vgamma_l_p2,
            vkt_p1_re, vkt_p2_re, R_p1_re[h_val], R_p2_re[h_val]
        )
        row = {"h": h_val / L}
        for i, (s, p) in enumerate(zip(test_stats_R, p_value_R), start=1):
            row[f"group{i}"] = format_stat(s, p)
        table_rows.append(row)

    res.append(pd.DataFrame(table_rows).sort_values("h").reset_index(drop=True))
    return res


# =============================================================================
# PART 3 — MAIN: sensitivity loop over (tau, Rk, Rtk)
# =============================================================================

def run_sensitivity(declustered_p1_odd, declustered_p2_odd, data_full, station_doc,
                    tau_list=(0.015, 0.02, 0.05),
                    Rk_list=(0.12,),
                    Rtk_list=(0.03, 0.04)):
    """
    For each (tau, Rk, Rtk): decluster (keep every 2nd day), run fGetAnalysis,
    and save heatmaps, the per-run test table, and the group-membership data.

    Outputs land under empirics_results_{tau}/{figures,tables,group_data}/.
    """

    for tau in tau_list:
        root = f"empirics_results_{tau}"
        figs = os.path.join(root, "figures")
        tabs = os.path.join(root, "tables")
        grps = os.path.join(root, "group_data")
        for d in (figs, tabs, grps):
            os.makedirs(d, exist_ok=True)

        # intercept plt.show so each heatmap is saved to a queued path
        original_show = plt.show
        fig_queue = []

        def saving_show(*args, **kwargs):
            if fig_queue:
                save_path = fig_queue.pop(0)
                plt.gcf().savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"    [fig saved]  {save_path}")
            original_show(*args, **kwargs)

        plt.show = saving_show
        all_tables = []

        for Rk in Rk_list:
            for Rtk in Rtk_list:
                tag = f"Rk{Rk:.2f}_Rtk{Rtk:.2f}"
                print(f"\n{'='*62}\n  Rk={Rk}  |  Rtk={Rtk}  |  tau={tau}\n{'='*62}")

                # queue the two heatmap paths (p1 then p2, the fHeatMap order)
                fig_queue.append(os.path.join(figs, f"heatmap_p1_{tag}.png"))
                fig_queue.append(os.path.join(figs, f"heatmap_p2_{tag}.png"))

                res = fGetAnalysis(data_full, declustered_p1_odd, declustered_p2_odd,
                                   Rk, Rtk, tau, station_doc)

                # 1. results table (self-documenting with Rk/Rtk columns)
                tbl = res[0].copy()
                tbl.insert(0, "Rk", Rk)
                tbl.insert(1, "Rtk", Rtk)
                all_tables.append(tbl)
                per_run_csv = os.path.join(tabs, f"table_{tag}.csv")
                tbl.to_csv(per_run_csv, index=False)
                print(f"    [table saved] {per_run_csv}")

                # 2. group membership for heatmap reconstruction (deterministic re-run)
                w_full, vkt_full, Ghat, datevec, vog_full = fGetResults(data_full, Rk, Rtk, tau)
                w_p1, vkt_p1, _, _, vog_p1 = fGetResults(declustered_p1_odd, Rk, Rtk, tau)
                w_p2, vkt_p2, _, _, vog_p2 = fGetResults(declustered_p2_odd, Rk, Rtk, tau)

                vl = [int(l) for l in datevec[:, Ghat - 2] if l != 0]
                index = [0] + vl + [len(vog_full)]
                full_order = w_full["station_id"].to_numpy()

                vog_p1_re = (
                    pd.Series(vog_p1, index=w_p1["station_id"])
                      .reindex(full_order)
                )
                vog_p2_re = (
                    pd.Series(vog_p2, index=w_p2["station_id"])
                      .reindex(full_order)
                )

                vkt_p1_re = (
                    pd.Series(vkt_p1, index=w_p1["station_id"])
                      .reindex(full_order)
                      .to_numpy()
                )
                vkt_p2_re = (
                    pd.Series(vkt_p2, index=w_p2["station_id"])
                      .reindex(full_order)
                      .to_numpy()
                )

                gamma_l_p1 = weight_gamma_j(vog_p1_re.values, vkt_p1_re, index)
                gamma_l_p2 = weight_gamma_j(vog_p2_re.values, vkt_p2_re, index)

                station_ids = w_full["station_id"].values
                group_labels = np.empty(len(station_ids), dtype=int)
                for g in range(Ghat):
                    group_labels[index[g]:index[g + 1]] = g

                membership_df = pd.DataFrame({
                    "station_id": station_ids,
                    "group":      group_labels,
                    "gamma_g_p1": [gamma_l_p1[g] for g in group_labels],
                    "gamma_g_p2": [gamma_l_p2[g] for g in group_labels],
                })
                csv_path = os.path.join(grps, f"group_membership_{tag}.csv")
                membership_df.to_csv(csv_path, index=False)

                pkl_path = os.path.join(grps, f"group_data_{tag}.pkl")
                with open(pkl_path, "wb") as fh:
                    pickle.dump({
                        "Rk": Rk, "Rtk": Rtk, "tau": tau, "Ghat": Ghat,
                        "index": index, "datevec": datevec,
                        "membership": membership_df,
                        "gamma_l_p1": gamma_l_p1, "gamma_l_p2": gamma_l_p2,
                        "vkt_p1": vkt_p1_re, "vkt_p2": vkt_p2_re,
                        "station_order": station_ids,
                    }, fh)
                print(f"    [group CSV]   {csv_path}")
                print(f"    [group PKL]   {pkl_path}")

        plt.show = original_show

        combined = pd.concat(all_tables, ignore_index=True)
        combined_path = os.path.join(tabs, "all_results_combined.csv")
        combined.to_csv(combined_path, index=False)
        print(f"\n{'='*62}\n  Combined table ({len(all_tables)} runs) -> {combined_path}\n{'='*62}\n")
        print(combined.to_string(index=False))


# =============================================================================
# PART 4 — PRE-ANALYSIS DIAGNOSTIC FIGURES  
# =============================================================================
# Distribution of pairwise extremal dependence R11 after declustering

try:
    from IPython.display import display
except Exception:                       # plain `python` run -> fall back to print
    def display(*args, **kwargs):
        for a in args:
            print(a)

# Global colors
COLORS = ["#2166ac", "#1D9E75", "#f46d43", "#7F77DD", "#BA7517"]
BLUE = COLORS[0]
GREEN = COLORS[1]
RED = COLORS[2]

def set_paper_style():
    mpl.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,

        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",

        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",

        "font.size": 18,
        "axes.labelsize": 22,
        "axes.titlesize": 24,
        "legend.fontsize": 19,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,

        "axes.linewidth": 1.2,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.major.size": 5.0,
        "ytick.major.size": 5.0,
        "xtick.direction": "out",
        "ytick.direction": "out",

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def clean_tick_label(x, pos):
    """
    Format tick labels so that 0.0, 0.00, -0.00 become 0.
    Other labels are kept compact.
    """
    if np.isclose(x, 0.0, atol=1e-12):
        return "0"
    if np.isclose(x, round(x), atol=1e-12):
        return f"{int(round(x))}"
    return f"{x:.2f}".rstrip("0").rstrip(".")


def style_axis(ax):
    ax.set_facecolor("white")

    ax.grid(
        axis="y",
        color="0.82",
        linewidth=0.8,
        alpha=0.55,
    )

    ax.grid(axis="x", visible=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_color("0.25")
    ax.spines["bottom"].set_color("0.25")
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    ax.tick_params(
        axis="both",
        which="major",
        color="0.25",
        width=1.1,
        length=5.0,
    )

    ax.xaxis.set_major_formatter(FuncFormatter(clean_tick_label))
    ax.yaxis.set_major_formatter(FuncFormatter(clean_tick_label))


# ============================================================
# Pairwise R11 helper functions
# ============================================================



def _pairwise_R11_long(df, station_doc, period_label, qu):
    R11_df, dist_df = get_mR11(
        df,
        station_doc=station_doc,
        h=None,
        id_col="station_id",
        qu=qu,
    )

    R = R11_df.to_numpy(dtype=float)
    D = dist_df.to_numpy(dtype=float)

    i_upper, j_upper = np.triu_indices_from(R, k=1)

    out = pd.DataFrame({
        "period": period_label,
        "station_i": R11_df.index.to_numpy()[i_upper],
        "station_j": R11_df.columns.to_numpy()[j_upper],
        "R11": R[i_upper, j_upper],
        "distance_km": D[i_upper, j_upper],
    })

    out = (
        out.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["R11"])
    )

    return out, R11_df, dist_df


def _summary(series):
    x = pd.Series(series).dropna()

    return pd.Series({
        "n_pairs": int(x.size),
        "mean": x.mean(),
        "sd": x.std(ddof=1),
        "min": x.min(),
        "q10": x.quantile(0.10),
        "q25": x.quantile(0.25),
        "median": x.median(),
        "q75": x.quantile(0.75),
        "q90": x.quantile(0.90),
        "q95": x.quantile(0.95),
        "max": x.max(),
        "Pr(R11 = 0)": np.mean(np.isclose(x, 0.0)),
    })


def _fd_bin_edges(x, max_bins=30):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if x.size < 2 or np.nanmin(x) == np.nanmax(x):
        lo = np.nanmin(x) - 0.5
        hi = np.nanmax(x) + 0.5
        return np.linspace(lo, hi, 12)

    q25, q75 = np.nanpercentile(x, [25, 75])
    iqr = q75 - q25

    if iqr <= 0:
        n_bins = int(np.clip(np.sqrt(x.size), 10, max_bins))
    else:
        width = 2 * iqr * x.size ** (-1 / 3)
        n_bins = int(np.ceil((np.nanmax(x) - np.nanmin(x)) / width))
        n_bins = int(np.clip(n_bins, 10, max_bins))

    lo = 0.0 if np.nanmin(x) >= 0 else np.nanmin(x)
    hi = np.nanmax(x)

    if hi <= lo:
        hi = lo + 1.0

    return np.linspace(lo, hi, n_bins + 1)


# ============================================================
# R11 histogram and ECDF
# ============================================================
def plot_R11(declustered_p1_odd, declustered_p2_odd, station_doc):
    set_paper_style()
    Rtk = 0.03

    R11_p1_long, _, _ = _pairwise_R11_long(
        declustered_p1_odd,
        station_doc,
        "Period 1",
        qu=Rtk,
    )

    R11_p2_long, _, _ = _pairwise_R11_long(
        declustered_p2_odd,
        station_doc,
        "Period 2",
        qu=Rtk,
    )

    R11_pairwise_long_after_decluster = pd.concat(
        [R11_p1_long, R11_p2_long],
        ignore_index=True,
    )

    summary_R11_after_decluster = (
        R11_pairwise_long_after_decluster
        .groupby("period", sort=False)["R11"]
        .apply(_summary)
        .unstack()
    )

    summary_cols = [
        "n_pairs",
        "mean",
        "sd",
        "min",
        "q10",
        "q25",
        "median",
        "q75",
        "q90",
        "q95",
    ]

    summary_R11_after_decluster_pub = summary_R11_after_decluster[summary_cols]
    display(summary_R11_after_decluster_pub.round(4))

    plot_data = {
        "Period 1": R11_pairwise_long_after_decluster.loc[
            R11_pairwise_long_after_decluster["period"].eq("Period 1"),
            "R11",
        ].dropna().to_numpy(),

        "Period 2": R11_pairwise_long_after_decluster.loc[
            R11_pairwise_long_after_decluster["period"].eq("Period 2"),
            "R11",
        ].dropna().to_numpy(),
    }

    all_values = np.concatenate([
        v for v in plot_data.values()
        if len(v) > 0
    ])

    bins = _fd_bin_edges(all_values, max_bins=30)

    period_style = {
        "Period 1": {
            "color": BLUE,
            "alpha": 0.28,
            "linestyle": "-",
            "linewidth": 3.0,
        },
        "Period 2": {
            "color": RED,
            "alpha": 0.24,
            "linestyle": "--",
            "linewidth": 3.0,
        },
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 4.6),
        dpi=300,
    )

    fig.patch.set_facecolor("white")

    # --------------------------------------------------------
    # Panel 1: Histogram
    # --------------------------------------------------------
    ax = axes[0]

    for label in ["Period 1", "Period 2"]:
        x = plot_data[label]
        style = period_style[label]

        ax.hist(
            x,
            bins=bins,
            density=True,
            histtype="stepfilled",
            facecolor=style["color"],
            edgecolor=style["color"],
            alpha=style["alpha"],
            linewidth=1.2,
        )

        ax.hist(
            x,
            bins=bins,
            density=True,
            histtype="step",
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )

    ax.set_title("Histogram", pad=12)
    ax.set_xlabel(r"Pairwise $\widehat{R}_{i\ell}(1,1)$")
    ax.set_ylabel("Density")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    style_axis(ax)

    # --------------------------------------------------------
    # Panel 2: ECDF
    # --------------------------------------------------------
    ax = axes[1]

    for label in ["Period 1", "Period 2"]:
        x = plot_data[label]
        style = period_style[label]

        x_sorted = np.sort(x)
        ecdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)

        ax.step(
            x_sorted,
            ecdf,
            where="post",
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )

    ax.set_title("ECDF", pad=12)
    ax.set_xlabel(r"Pairwise $\widehat{R}_{i\ell}(1,1)$")
    ax.set_ylabel("Cumulative probability")

    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(FixedLocator([0.0, 0.5, 1.0]))
    ax.yaxis.labelpad = 4
    style_axis(ax)

    # --------------------------------------------------------
    # Shared legend below panels
    # --------------------------------------------------------
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=BLUE,
            linestyle="-",
            linewidth=3.4,
            label="Period 1",
        ),
        Line2D(
            [0],
            [0],
            color=RED,
            linestyle="--",
            linewidth=3.4,
            label="Period 2",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.035),
        handlelength=3.2,
        columnspacing=3.0,
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        top=0.83,
        bottom=0.28,
        wspace=0.2,
    )

    fig.savefig(
        "pairwise_R11_distribution_after_decluster.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.show()
    print("Saved: pairwise_R11_distribution_after_decluster.png")


# ============================================================
# Hill estimates
# ============================================================
def hill(x, k):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy()
    x = x[np.isfinite(x)]
    x = x[x > 0]

    if len(x) <= k or k < 1:
        return np.nan

    x = np.sort(x)

    return np.mean(
        np.log(x[-k:])
        - np.log(x[-k - 1])
    )

def plot_hill_histogram(data_full):
    set_paper_style()

    X = data_full.copy()
    X = X.drop(columns=["station_id"], errors="ignore")

    T = X.notna().sum(axis=1)

    k12 = np.floor(0.12 * T).astype(int)

    gamma12 = [
        hill(row, k)
        for row, k in zip(X.to_numpy(), k12)
    ]

    hill_df = pd.DataFrame({
        "hill_12pct": gamma12,
    })

    display(
        hill_df
        .describe(percentiles=(0.1, 0.25, 0.75, 0.9))
        .round(3)
    )

    plot_data = hill_df.dropna()
    x12 = plot_data["hill_12pct"].to_numpy()

    bins = np.histogram_bin_edges(x12, bins="fd")

    fig, ax = plt.subplots(
        figsize=(6.4, 4.5),
        dpi=300,
    )

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.hist(
        x12,
        bins=bins,
        density=True,
        histtype="stepfilled",
        facecolor=GREEN,
        edgecolor=GREEN,
        alpha=0.28,
        linewidth=1.2,
    )

    ax.hist(
        x12,
        bins=bins,
        density=True,
        histtype="step",
        color=GREEN,
        linewidth=3.0,
    )

    ax.set_xlabel(r"$\hat{\gamma}_i^H(k_i)$")
    ax.set_ylabel("Density")

    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    style_axis(ax)

    fig.subplots_adjust(
        left=0.18,
        right=0.98,
        top=0.96,
        bottom=0.20,
    )

    fig.savefig(
        "hill_histogram_data_full.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.show()

    print("Saved: hill_histogram_data_full.png")


if __name__ == "__main__":
    # 1. (optional) build the cleaned files from raw ECA&D data.
    #    Skipped  if application_data.nc + station_doc.pkl exist.
    clean_raw_data()
    # 2. load the cleaned data.
    p1_df, p2_df, station_doc = load_clean_data()

    # 3. pre-analysis diagnostic figures (R11 distribution + Hill histogram).
    declustered_p1_odd = keep_every_second_day(p1_df, start=0)
    declustered_p2_odd = keep_every_second_day(p2_df, start=0)
    data_full = pd.merge(declustered_p1_odd, declustered_p2_odd, on="station_id")
    plot_R11(declustered_p1_odd, declustered_p2_odd, station_doc)
    plot_hill_histogram(data_full)

    # 4. run the sensitivity analysis.
    run_sensitivity(declustered_p1_odd, declustered_p2_odd, data_full, station_doc,
                    tau_list =[0.02,0.05,0.015],
                    Rk_list=[0.12],
                    Rtk_list=[0.03, 0.04])
