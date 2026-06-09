# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

WaSPS-DTW implements **Wasserstein + Soft-DTW** time series analysis: computing barycenters and classifying stochastic time series (sequences of distributions) using Soft-DTW with Euclidean or Wasserstein (exponential or Weibull) local costs. Primary applications: hydrological regime classification (NetCDF) and SAR amplitude classification (CPAZMaL/HDF5).

## Environment setup

```bash
# From repo root — creates venv/, installs dependencies, builds Cython extensions
bash setup_venv.sh
source venv/bin/activate
```

**Cython extensions must be compiled** before any `sdtw` imports will work. The `.so` files (`soft_dtw_fast`, `wasserstein_fast`) live in `src/sdtw/` and are rebuilt with:
```bash
cd src && python -c "
from setuptools import setup, Extension; from Cython.Build import cythonize; import numpy
exts=[Extension('sdtw.soft_dtw_fast',['sdtw/soft_dtw_fast.pyx'],include_dirs=[numpy.get_include()]),
      Extension('sdtw.wasserstein_fast',['sdtw/wasserstein_fast.pyx'],include_dirs=[numpy.get_include()])]
setup(ext_modules=cythonize(exts,language_level=3))
" build_ext --inplace
```

## Running tests

Tests must be run from `src/` (imports use `sys.path.insert` relative to file location):

```bash
cd src
python -m pytest sdtw/tests/ -v                        # all tests
python -m pytest sdtw/tests/test_soft_dtw.py -v        # Soft-DTW DP correctness
python -m pytest sdtw/tests/test_weibull.py -v         # Weibull W₂² + gradients + estimator
```

## Running experiments

All experiment scripts run from the **repo root**:

```bash
source venv/bin/activate

# Hydrological regime classification (exponential)
python src/experiments/sdtw_barycenter_classification.py --mode one-shot --plot-barycenters
python src/experiments/sdtw_barycenter_classification.py --mode kfold --n-splits 5
python src/experiments/seed_sweep_runner.py --n-seeds 50

# CPAZMaL SAR classification (Weibull) — new
python src/experiments/cpazmal_classification.py --mode kmedoid
python src/experiments/cpazmal_classification.py --mode shapelet --epochs 20
python src/experiments/cpazmal_classification.py --mode kmedoid --max-groups 6  # quick smoke-test

bash run_all_experiments.sh   # toggle booleans inside to select modes
```

CPAZMaL HDF5: `/home/mgallet/Documents/Codes/Python/1_DONE/CPAZMAL/DATASET/dataset_original/PAZTSX_CRYO_ML.hdf5`  
Results go to `results/` subdirectories. Figures (`.png`) are gitignored.

## Architecture

```
src/
├── sdtw/                      # Core library
│   ├── soft_dtw.py            # SoftDTW class + sdtw_divergence() helper
│   ├── distance.py            # SquaredEuclidean, WassersteinDistance (exp + Weibull)
│   ├── barycenter.py          # sdtw_barycenter() — L-BFGS-B via scipy
│   ├── classification_methods.py  # Wrapper functions; distance fns with divergence=True
│   ├── soft_dtw_fast.pyx      # Cython: DP forward/backward + Jacobian products (exp+Weibull)
│   └── wasserstein_fast.pyx   # Cython: MLE/log-cumulants + pairwise W₂² (exp+Weibull)
├── estimator/                 # Distribution parameter estimators
│   ├── mle.py                 # MLE class
│   └── log_cumulant.py        # LogCumulant (preferred — uses Cython internally)
├── optimizer/
│   ├── wasserstein_barycenter_sgd.py  # sgd_barycenter() — SGD with warmup/decay/clip
│   └── learning_shapelets.py  # LearningShapelets with SoftDTW-Wasserstein distance
├── dataloader/
│   ├── netcdf_loader.py          # Load river discharge NetCDF files
│   ├── classification_loader.py  # Load .npy datasets; estimate_parameters_for_samples()
│   ├── cpazmal_loader.py         # MLDatasetLoader, extract_time_series, estimate_weibull_params
│   ├── series_extraction.py      # Extract λ-series, sliding windows
│   └── preprocessing.py          # Train/test split, sliding windows
├── data_generator/            # Synthetic data (exponential/shifted series)
├── experiments/
│   ├── sdtw_barycenter_classification.py  # Regime experiment (4 modes)
│   ├── cpazmal_classification.py          # CPAZMaL+Weibull (kmedoid/shapelet)  ← new
│   ├── classification_evaluation.py       # evaluate_classification(), kfold runner
│   ├── classification_sensitivity.py      # Gamma/sample-size sweeps
│   ├── lstm_classifier.py                 # LSTM baseline
│   ├── ot_sta_classifier.py               # Regularized OT/STA baseline
│   ├── shapelets_classifier.py            # LearningShapelets wrapper
│   └── seed_sweep_runner.py               # Multi-seed subprocess sweep
├── plot/                      # Matplotlib helpers
└── utils/                     # timing, binning
```

**Data flow for CPAZMaL+Weibull classification:**
1. `extract_time_series(loader)` → `X_train[i]` shape `(T, W²)` — pixels at each timestep
2. `estimate_weibull_params(X_train[i])` → `(T, 2)` arrays `[k, λ_scale]`
3. `compute_barycenter_wasserstein_sgd(..., distribution='weibull')` → per-class `(T, 2)` barycenters
4. `compute_sdtw_distance_weibull(sample_params, barycenter_params, divergence=True)` → scalar

## Key conventions

**Array shapes:**
- Raw SAR/discharge time series: `(T, N_samples)` — T timesteps, each with N_samples values
- Exponential params: `(T, 1)` — rate β (not scale)
- Weibull params: `(T, 2)` — column 0=k (shape), column 1=λ_scale
- Barycenter init must match parameter shape

**Soft-DTW divergence (default = True everywhere):**
```python
D_γ(X,Y) = SDTW(X,Y) − ½(SDTW(X,X) + SDTW(Y,Y))
```
- `sdtw_divergence(D_xy, D_xx, D_yy, gamma)` in `soft_dtw.py`
- All `compute_sdtw_distance_*` functions accept `divergence=True` (default)
- Pass `divergence=False` to reproduce pre-divergence results
- Self-terms `SDTW(b,b)` are NOT constant across barycenters — always include them

**WassersteinDistance protocol:**
```python
wd = WassersteinDistance(X, Y, distribution='weibull', X_is_params=True, Y_is_params=True)
D  = wd.compute()              # [m, n] W₂² matrix
G  = wd.jacobian_product(E)    # [m, 2] for Weibull, [m, 1] for exponential
GY = wd.jacobian_product_Y(E)  # [n, 2] — uses symmetry trick (swapped args + E.T)
```

**Two barycenter optimization backends:**
- `sdtw_barycenter()` — L-BFGS-B; good for small T, exponential only
- `sgd_barycenter(distribution='weibull')` — SGD with warmup/decay; handles `(T, 2)` params

**`X_is_params` / `Y_is_params` flags:** Pass `True` when a series already holds distribution parameters to skip redundant re-estimation.

**Import pattern:** All modules use `sys.path.insert(0, str(parent_dir))` — there is no `pip install -e .`. Scripts must run from the correct working directory (repo root for experiments, `src/` for tests).

**Exponential parameterization:** Rate β (not scale), so `E[X] = 1/β`.  
**Weibull parameterization:** Scale λ (not rate), so `E[X^k] = λ^k`.  
Exponential ↔ Weibull(k=1, λ=1/β) — the two are *not* interchangeable distance objects.

**UCR data:** Located at `../dev/UCR_TS_Archive_2015/` (relative to repo root), files named `<Dataset>_TRAIN` / `<Dataset>_TEST` (no extension, space-separated, first column is label).
