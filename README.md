# Codes and datasets to replicate the results presented in the paper: Clustering extreme value indices in large panels
<!-- badges: start -->
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
<!-- badges: end -->

Authors: Chenhui Wang (c.h.wang@vu.nl, Vrije Universiteit Amsterdam), Juan-Juan Cai (j.cai@vu.nl, Vrije Universiteit Amsterdam & Tinbergen Institute), Yicong Lin (yc.lin@vu.nl, Vrije Universiteit Amsterdam & Tinbergen Institute) and Julia Schaumburg (j.schaumburg@vu.nl, Vrije Universiteit Amsterdam & Tinbergen Institute)

This repository contains the Python code and instructions needed to reproduce all
simulation and empirical results in the paper. The code is organised around a
shared library of core routines (`common.py`), three simulation scripts
(`main_simulation.py`, `compare_method.py`, `rejection_simulation.py`), a figure
script (`make_figures.py`), and the empirical application (`empirics.py`).


## Python packages required

The code was developed and tested with Python 3.11+. The following third-party
packages are required:
`numpy` , `pandas`, `scipy`, `joblib`, `matplotlib`, `xarray`, `netCDF4`, `cartopy`.

## Files contained
- **`common.py`** — core routines shared across all scripts: the data-generating
  processes, the estimator, the structural-break machinery, the parallel-seed driver.
- **`main_simulation.py`** — the main Monte Carlo study.
- **`compare_method.py`** , **`rejection_simulation.py`** — the Monte Carlo study in supplement.
- **`make_figures.py`** — produces figures.
- **`empirics.py`** — the empirical analysis of European winter daily precipitation. 

## Data

The empirical application uses **blended European daily precipitation (RR)** from
the European Climate Assessment & Dataset (ECA&D):

> ECA&D, *Daily data — predefined series*, blended series, element RR (daily
> precipitation amount), <https://www.ecad.eu/dailydata/predefinedseries.php>,
> accessed on January 27, 2025.
> 
**Build from the raw ECA&D files.** 
Download the blended RR series from the ECA&D link above and place the raw inputs 
in the repository root:
   - `ECA_blend_rr/` — the folder of per-station `RR_STAID*.txt` files,
   - `stations.txt` — the station catalogue,
   - `ECA_blend_source_rr.txt` — the source/coverage file.

Then run `python empirics.py`; `clean_raw_data()` will regenerate `application_data.nc` 
and `station_doc.pkl` before the analysis. 

## How to reproduce the results

Run `python <script>.py --help` to see more options.

### Main simulation (`main_simulation.py`)

```bash
# Main simulation
python main_simulation.py --nsim 1000 --T 1000 --output_dir sim_results_main

# S.2.1
python main_simulation.py --nsim 1000 --T 3000 --output_dir sim_results_supp

# S.2.4 stress test (small gaps between group EVIs)
python main_simulation.py --nsim 1000 --T 1000 \
    --gamma_start 0.48 --gamma_end 0.64 --output_dir sim_results_small_gap

# S.2.5 perturbed setting
python main_simulation.py --dgp independent_noise dependent_noise \
    --nsim 1000 --output_dir sim_results_perturbed

# S.2.5 single group
python main_simulation.py --dgp independent_noise dependent_noise \
    --G 1 --gamma_single 0.2 0.5 0.7 1.0 --output_dir sim_results_G1_noise
```

### Method comparison (`compare_method.py`)

```bash
# Replicate Chen et al., vary q (Figure S.2.7)
python compare_method.py --dgp_source iterative --experiment vary_q

# Replicate Chen et al., vary Delta (Figure S.2.8)
python compare_method.py --dgp_source iterative --experiment vary_delta

# DGP as in main simulation, vary q (independent + dependent)
python compare_method.py --dgp_source segmentation --experiment vary_q \
    --n_list 1000 3000 --q_grid 100 300
```

### Rejection-rate simulation (`rejection_simulation.py`)

```bash
python rejection_simulation.py --G 3 --groupsize 100 --nsim 1000 --T 1000 3000
```

### Figures (`make_figures.py`)

Once the CSV outputs above exist, render all figures with:

```bash
python make_figures.py
```

### Empirical application (`empirics.py`)

With the data in place (see the Data section), run:

```bash
python empirics.py
```



This version: June 2, 2026

Copyright: Chenhui Wang, Juan-Juan Cai, Yicong Lin, and Julia Schaumburg


For any questions or feedback, please feel free to contact: c.h.wang@vu.nl
