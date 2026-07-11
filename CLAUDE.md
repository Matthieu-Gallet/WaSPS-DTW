# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

WaSPS-DTW implements **Wasserstein + Soft-DTW** time series analysis: computing barycenters and
classifying stochastic time series (sequences of distributions) using Soft-DTW with Euclidean or
Wasserstein (exponential or Weibull) local costs. Two real datasets are supported:
- **River discharge** (NetCDF → prebuilt `.npy`, exponential family)
- **CPAZMaL SAR** (HDF5, Weibull family)

This is the **JAX branch** (`feat/jax-refonte`). The old Cython/PyTorch pipeline lives in `.old/`.

## Environment setup

```bash
# From repo root — creates .venv/, installs dependencies (no Cython build step)
bash setup_venv.sh
source .venv/bin/activate
```

```bash
# Smoke test
python -c "import jax, ott, optax; print(jax.devices())"
```

## Running tests

```bash
source .venv/bin/activate
python -m pytest tests/ -q       # run from repo root (not src/)
```

## Running experiments

All scripts run from the **repo root**:

```bash
source .venv/bin/activate

# Gate-E smoke: all 4 methods × 2 modes in ~18s (T=4, n_steps=5)
python experiments/run_classification.py configs/classification/classification_smoke.yaml

# Synthetic exponential full run (4 methods × 2 modes × 4 seeds — slow: STA bary ~1h)
python experiments/run_classification.py configs/classification/classification.yaml

# River-discharge regime classification (exponential, wasps+eucl_params+eucl_raw)
# 5-fold StratifiedGroupKFold by default (fold=-1 loops all folds, reports mean±std)
python experiments/run_classification.py configs/classification/river.yaml         # gamma sweep (4 values) × 4 seeds × 5-fold, T=365, no STA
python experiments/run_classification.py configs/classification/river_agg.yaml     # weekly-aggregated (T=52), single split
python experiments/run_classification.py configs/classification/river_sta.yaml     # 4 methods w/ STA, T=25 truncated, single split
python experiments/run_classification.py configs/classification/river_smoke.yaml   # quick smoke (T=8, single fold)

# CPAZMaL SAR classification (Weibull) — classification task (one continuous series
# per spatial window, K-fold by geographic group), NOT the old temporal-prediction split.
# HAG excluded (45 samples/28 groups — too few for stratified K-fold).
python experiments/run_classification.py configs/classification/cpazmal.yaml       # gamma sweep (4 values) × 4 seeds × 5-fold, no STA
python experiments/run_classification.py configs/classification/cpazmal_smoke.yaml # quick multi-class smoke, 3-fold
python experiments/run_classification.py configs/classification/cpazmal_sta.yaml   # 4 methods w/ STA, single split
# → first run extracts from HDF5 (slow: ~24% of attempted groups survive full-year
#   validity filtering — see "CPAZMaL classification" below); caches to data/cpazmal/

# Fit + evaluate (F1, per-class F1, timing) + save barycenters (gamma sweep × 4 seeds × 5-fold)
python experiments/run_barycenters.py configs/classification/river_bary.yaml --n-jobs -1
python experiments/run_barycenters.py configs/classification/cpazmal_bary.yaml --n-jobs -1

# Sensitivity analysis (real data — river and CPAZMaL, NOT synthetic; see RUN.md for
# the full scenario list: run_optim_hyper.py for calibration grids (grid_knn/grid_bary),
# run_sensitivity.py for N-samples/N-train/decimation sweeps)
python experiments/run_optim_hyper.py --help
python experiments/run_sensitivity.py --help

# Final baseline comparison — KNN-only, both datasets in one config, divergence vs
# non-divergence (wasps/eucl_params/eucl_raw each get a _nodiv variant), 5 random seeds
python experiments/run_full_baseline.py --config configs/full_baseline.yaml --n-jobs -1
```

See `RUN.md` for the complete command reference (every config, expected runtime, and
parameters to double-check before launching).

Data paths:
- CPAZMaL HDF5: `/home/mgallet/Documents/Codes/Python/1_DONE/CPAZMAL/DATASET/dataset_original/PAZTSX_CRYO_ML.hdf5`
- River NetCDF (raw): `/home/mgallet/Documents/Dataset/RIVER_DISCHARGES/c7491e060d94c97212f0fe7ebcff57f0/data_version-5.nc`
- River npy (prebuilt): `data/river/` — **gitignored** (`data/**/*.npy`), despite being the
  primary dataset; regenerate with the external Explore2_HydroDataset project if missing.
  Currently 800 samples / 4 classes / 183 geographic groups (`groups_balanced.npy`).
- CPAZMaL npy cache: `data/cpazmal/` (gitignored, auto-populated on first run)

## Architecture

```
data/
├── river/          # Prebuilt: X/Y/metadata_{balanced,basic}.npy
└── cpazmal/        # Extraction cache (gitignored)

src/
├── distributions.py          # JAX distributions: exponential + Weibull; MLE + log-cumulant fitting
├── costs.py                  # SqEuclidean + WaSPS W₂² (closed-form, autodiff; log_correction, use_positivity_constraint)
├── softdtw.py                # SoftDTW forward + divergence; SoftDTW class (manual/auto × div/not-div)
├── barycenter.py             # Fréchet barycenter via optax (sgd default, adam optional); fit_barycenter(series, softdtw, …)
├── data/
│   ├── preprocess.py         # clean_series(), to_fixed_n() — canonical filters for estimation
│   ├── cpazmal_loader.py     # MLDatasetLoader + extract_time_series (HDF5, classification: one series/window + group)
│   └── river_loader.py       # load_river_classification() — npy + holdout or StratifiedGroupKFold
├── classification/
│   ├── nn.py                 # k-NN SoftDTW (vmap over train set)
│   └── barycenter_clf.py     # Nearest-barycenter (joblib-parallel per class)
└── baselines/
    └── sta_wrapper.py        # STA 1-NN + make_cost_fn() for STA barycenter

experiments/
├── data_utils.py             # Shared dataset loaders (synthetic/river/cpazmal), build_repr, subsample
├── method_defs.py            # _METHODS table (incl. *_nodiv variants), make_cost_fn, make_softdtw_bary
├── experiment_common.py      # Shared eval/logging/CSV infra — _eval_knn/_eval_bary, _load_and_cap,
│                              #   _iterations, env-var logging (EXPERIMENT_LOG_FILE/_DEBUG/_VERBOSE) —
│                              #   used by every script below except data_utils/method_defs themselves
├── run_classification.py     # multi-method × 2-mode × multi-seed/fold runner; gamma_values sweep
├── run_barycenters.py        # Fit + evaluate (F1/timing) + save per-class barycenters as .npy;
│                              #   backward compatible: no cross_validation/gamma_values → fit+save only
├── run_optim_hyper.py        # Hyperparameter grid search (grid_knn/grid_bary) — reads the same
│                              #   sensitivity_river.yaml/sensitivity_cpazmal.yaml as run_sensitivity.py
├── run_sensitivity.py        # Sensitivity sweeps only (n_samples/n_train/decimation) — see RUN.md
├── run_full_baseline.py      # Final KNN-only baseline, both datasets, divergence vs non-divergence,
│                              #   5 random seeds (no k-fold) — replaces the old final_comparison scenario
├── extract_bary_plots.py     # River PN/NG barycenter plots (β→λ inversion for wasps/eucl_params)
└── extract_latex_tables.py   # F1 mean±std → LaTeX NiceTabular tables (classif/bary gamma sweep + sensitivity)

configs/                      # See RUN.md for the full list with runtimes/parameters
├── sensitivity_river.yaml / sensitivity_cpazmal.yaml  # grid_knn/grid_bary (run_optim_hyper.py) +
│                                                       #   sweep_n_samples/n_train/decimation (run_sensitivity.py)
├── full_baseline.yaml         # run_full_baseline.py — both datasets in one file
└── classification/            # everything run via run_classification.py / run_barycenters.py
    ├── classification.yaml / classification_smoke.yaml   # Synthetic exponential
    ├── river.yaml / river_smoke.yaml / river_sta.yaml / river_agg.yaml / river_full.yaml / river_bary_debug.yaml
    ├── river_bary.yaml         # dedicated run_barycenters.py gamma-sweep config (separate from river.yaml)
    ├── cpazmal.yaml / cpazmal_smoke.yaml / cpazmal_sta.yaml / cpazmal_debug.yaml
    └── cpazmal_bary.yaml       # dedicated run_barycenters.py gamma-sweep config (separate from cpazmal.yaml)

analysis/
├── classification_notebook.ipynb    # Interactive: CPAZMaL classification
└── river_barycenter_agg4.ipynb      # Interactive: river discharge barycenters (weekly aggregation)

tests/                        # pytest suite

.old/                         # Archived legacy code (old Cython/PyTorch pipeline)
```

## Key conventions

**Array shapes:**
- Raw time series: `(T, N_samples)` float64 — T timesteps, N values each
- River samples: `(T, D·W·W)` after reshape from `(T, D, W, W)` raw npy
- Exponential params: `(T, 1)` — rate β (not scale, `E[X]=1/β`)
- Weibull params: `(T, 2)` — column 0=k (shape), column 1=λ_scale
- NaN preserved through load → filtered by `clean_series` inside `distributions.fit`

**Four base classification methods (defined in `experiments/method_defs.py:_METHODS`):**

| Key | Representation | Cost | Barycenter | Notes |
|-----|----------------|------|------------|-------|
| `wasps` | params `(T,1)` or `(T,2)` | WaSPS W₂² closed-form | `use_positivity_constraint=True`, manual grad | exponential or Weibull |
| `eucl_params` | params (MLE) | SqEuclidean | autodiff | no positivity constraint in barycenter |
| `eucl_raw` | raw samples `(T,N)` | SqEuclidean | autodiff | raw order preserved |
| `sta` | raw samples `(T,N)` | OT Sinkhorn (OTT) | autodiff through Sinkhorn | slow: O(T²·N·n_train) |

Both KNN (k=1) and Barycenter modes are supported for all 4 methods.

**Non-divergence variants** (`wasps_nodiv`, `eucl_params_nodiv`, `eucl_raw_nodiv` — added
2026-07-08, used only by `experiments/run_full_baseline.py`): same representation/cost as
the base method, but `is_divergence=False` everywhere (and, for `wasps_nodiv`,
`log_correction=False` too). `is_divergence` previously only affected barycenter
fitting/predict — `src/classification/nn.py`'s `knn_predict` gained an `is_divergence`
kwarg (default `False`, zero-regression for every other caller) so KNN itself can
distinguish divergence vs non-divergence; without that, `eucl_params_nodiv` would be
indistinguishable from `eucl_params` in KNN mode (SqEuclidean has no `log_correction`).

**`WaSPS` flags:**
- `log_correction=True`: applies `c = δ + log(2−exp(−δ))` to the raw W₂² cost; guarantees SDTW divergence ≥ 0. Auto-set by `SoftDTW` when `is_divergence=True`.
- `use_positivity_constraint=True`: applies `φ = softplus` to both args before W₂². The gradient `gradient_X` chains `σ(θ)`. Used in the barycenter (θ-space); KNN/predict use `False` (data already positive).

**`SoftDTW` class (`softdtw.py`):**
```python
SoftDTW(cost_fn, gamma, is_divergence=True, manual_grad=True)
```
- `value(X, Y)` → scalar: plain SDTW or `D_γ(X,Y) = SDTW(X,Y) − ½SDTW(X,X) − ½SDTW(Y,Y)`
- `value_and_grad(X, Y)` → `(value, ∂value/∂X)`: manual path uses `cost_fn.gradient_X`; autodiff path via `jax.value_and_grad`.
- Auto-couples `log_correction=True` on WaSPS when `is_divergence=True`.

**`fit_barycenter` signature (`barycenter.py`):**
```python
fit_barycenter(series, softdtw: SoftDTW, n_steps=200, lr=1e-2, init=None, verbose=False,
               dtype=jnp.float64, patience=15, min_rel_improve=1e-4, optimizer="sgd")
```
Positivity enforced via `cost_fn.use_positivity_constraint` — no `softplus` or `manual_grad` kwargs.
Default dtype is `float64`. **Required for WaSPS on wide-range data**: float32 ULP (~12 at loss=1e5)
exceeds the early stopping threshold (1e-4 × 1e5 = 10), causing false patience triggers. Observed
on river: seed=45 float32 f1=0.065 (35 steps), float64 f1=0.383 (100 steps). See also CLAUDE.md
WaSPS structural limitation section.

**Divergence self-term identity:**
`∂SDTW(X,X)/∂X = 2·gradient_X(E_xx,X,X)`, so `−½·∂f(x,x)/∂x = −gradient_X(E_xx,x,x)`.
Full divergence backward: `gX = gradient_X(E_xy,x,y) − gradient_X(E_xx,x,x)`.
`gradient_Y` is never needed and is not implemented.

**`max_train_samples` / `max_test_samples`:**  
Applied ONCE after loading (stratified subsample) so all methods compare on the same samples.
`-1` = no cap. Set `20/20` when `sta` is in the method list (Sinkhorn is the bottleneck).

**STA complexity warning:**  
STA KNN cost is O(T²·n_train·n_test) Sinkhorn calls — quadratic in timesteps.  
STA barycenter is even slower (O(T²) Sinkhorn calls per gradient step × n_steps × n_classes).  
Measured: CPAZMaL STA/bary (T=24, n_classes=10, 2 train/class) ≈ 6.6h total:
  t_train=7343s (122 min) + predict≈16126s (269 min). Two JAX anti-patterns:
  (1) Training: per-class closure captures data → 10 JIT recompiles (~815s each).
  (2) Predict: Python double loop calls cost_fn(z,b) per pair → ~200 JIT recompiles (~80s each).
  Fix: pass data as stacked JAX args (not closed-over); wrap predict value() in @jax.jit.
  Not yet implemented. This measurement predates the single-continuous-series CPAZMaL
  loader (T≈29 now, was T=24 for the train-only period) — order-of-magnitude still applies.
  **STA is excluded from `configs/classification/cpazmal.yaml`'s 5-fold K-fold** (5× this per-split cost is
  intractable); use `configs/classification/cpazmal_sta.yaml` (single split, KNN only, tight sample caps).
`configs/classification/river.yaml` excludes STA (T=365 daily — intractable).  
`configs/classification/river_sta.yaml` uses `max_time_steps: 25` (truncated) for STA **KNN** (~13 min/seed, 4 seeds ≈ 52 min).  
  Why T=25 not T=52: river Sinkhorn convergence is ~10× slower than CPAZMaL (river β ∈ [0.025, 44] →  
  large transport distances → many iterations at ε=0.05). Empirical: T=52 measured ~56 min/seed.  
  T=25 cuts T² from 2704 to 625, restoring the 13 min/seed estimate.  
  STA barycenter omitted (T²=625 Sinkhorn/step at T=25 is still slow; modes: [knn] only).  
`configs/classification/river_smoke.yaml` (T=8) also excludes STA.  
Gate-E smoke: `classification_smoke.yaml` (T=4, n_steps=5) runs STA in ~5s per seed.  
**`_sinkhorn_jit` must stay `@jax.jit`** in `sta_wrapper.py` — without it, each Sinkhorn call re-traces (64× slower; verified empirically: 4579s → 72s).

**Barycenter prediction uses plain SDTW (not divergence):**  
`predict()` in `barycenter_clf.py` uses `is_divergence=False`. The SoftDTW divergence requires
`T_test ≈ T_bary` to avoid bias — the self-term `½SDTW(b,b)` scales with T_bary, so when
T_test ≠ T_bary, it dominates the discriminative SDTW(z,b) term and all predictions collapse to
one class. For CPAZMaL (T_test=5, T_bary=24), using is_divergence=True caused f1=0.027 → fixed
to is_divergence=False → f1=0.150. Plain SDTW is also the semantically correct choice (nearest
centroid by distance, not divergence).

**WaSPS barycenter structural limitation (real data with wide β range):**  
For exponential distributions, W₂²(β₁,β₂) = 2(μ₁-μ₂)² where μ=1/β. The WaSPS Fréchet barycenter
per-timestep = arithmetic mean of μ_i = **harmonic mean** of β. For river β ∈ [0.025, 44], one
outlier sample with β=0.025 (μ=40) dominates, pulling the barycenter far from typical values.
Combined with the softplus bijector (Δβ ≈ β·lr → tiny step for small β) and only 5 train/class (20 cap).

**Float64 precision matters for WaSPS early stopping:** When the WaSPS loss is large (~1e5, driven
by the β=0.025 outlier), float32 ULP ≈ 12 at that magnitude — larger than the early stopping
threshold (1e-4 × 1e5 = 10). Float32 literally cannot represent sub-threshold loss improvements →
patience triggers falsely. Float64 ULP ≈ 2.2×10⁻¹¹ → correctly detects any improvement.
Observed on river seed=45: float32 stopped at step ~35 (f1=0.065), float64 ran to step ~100
(f1=0.383). Float64 is **essential** for WaSPS barycenters with wide-range data.

Despite float64, wasps/bary f1≈0.17 (float64, 4 seeds) is below eucl_params/bary f1≈0.25 — the
harmonic-mean structural limitation remains. **wasps/KNN is the primary river result** (f1≈0.47,
competitive with eucl_params/KNN ≈0.46). Increasing max_train_samples beyond 20 would give more
stable barycenters but was not tested.

**CPAZMaL classification (not temporal prediction):**  
`extract_time_series` yields **one continuous series per spatial window** over the full year
(`start_date`/`end_date`, T≈29) plus a geographic `group` index — a classification task,
K-fold by group via `StratifiedGroupKFold`, exactly like river. This replaced an earlier
train/predict temporal-forecasting split (train Jan–Oct, predict Nov–Dec); that framing and
its `X_train`/`X_predict` return keys no longer exist.
`exclude_classes=('STUDY', 'HAG')` by default — HAG has only 45 samples / 28 groups (live
HDF5: 9 classes total, `ACC,PLA,ROC,HAG,STUDY,ABL,ICA,LAC,FOR`).
Requiring validity across the *full* year (vs. checking train/predict periods separately) is
stricter: only ~24% of attempted groups survive extraction (58/244 at full scale), concentrated
in 7–9 surviving groups/class (down from 28–36 attempted). Total **sample** count barely
changes (7984 vs. 8009 pre-HAG-exclusion under the old scheme) since surviving groups tend to
be larger — but geographic diversity per class is reduced, and K-fold degenerates (whole class
missing from a fold) if too few groups are attempted at once. `max_groups_per_class` (not the
old flat `max_groups`, which only ever hit one class — group names cluster alphabetically by
class prefix) must be set high enough that enough groups per class survive; `configs/classification/cpazmal_smoke.yaml`
uses 12 (attempted) to reliably get ~3 surviving/class for its 3-fold smoke test.

**cpazmal cache:**  
`data/cpazmal/X_{all|mgpcN}.npy`, `y_*.npy`, `groups_*.npy`, `meta_*.json` (suffix derived
from `max_groups_per_class`). Delete to force re-extraction from HDF5 — required after the
loader's return-contract changed (old `X_train_*`/`X_predict_*` cache files are incompatible
and were removed).

**Optimizer and parameter clamping (`barycenter.py`, `costs.py`):**  
`fit_barycenter(..., optimizer="sgd")` — `sgd` (default, no momentum/clipping) or `adam`, via
`_get_optimizer(lr, optimizer)`. SGD is the deliberate default (empirically more stable here
than Adam for this problem), not a leftover — despite the module docstring's history of saying
"Adam," the code has used SGD since commit `7685594`.
`_inverse_softplus`'s clip floor (`costs.py`) and the barycenter's post-hoc output floor
(`barycenter.py`) were raised from `1e-8` to `1e-5`: at `β=1e-8` the WaSPS exponential gradient
(`∝1/β³`) is already ~1e24 — far past where a NaN would already have occurred, so the old floor
did not actually prevent the blowup it was meant to guard against. Both floors are still only
applied at the data-prep/output edges, not *inside* the optimization loop itself — see the MLE
NaN diagnostic in `RUN.md` for whether that residual gap still matters in practice.

**Import pattern:** All modules use `sys.path.insert(0, str(_SRC))` — no `pip install -e .`.
Experiments run from repo root, tests from repo root (not `src/`).

**Exponential parameterization:** Rate β (not scale), so `E[X] = 1/β`.  
**Weibull parameterization:** Scale λ (not rate), so `E[X^k] = λ^k`.
