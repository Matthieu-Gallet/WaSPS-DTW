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
python experiments/run_classification.py configs/classification_smoke.yaml

# Synthetic exponential full run (4 methods × 2 modes × 4 seeds — slow: STA bary ~1h)
python experiments/run_classification.py configs/classification.yaml

# River-discharge regime classification (exponential, wasps+eucl_params+eucl_raw)
python experiments/run_classification.py configs/river.yaml         # full (~25 min, T=365, no STA)
python experiments/run_classification.py configs/river_sta.yaml     # 4 methods w/ STA, T=52 truncated
python experiments/run_classification.py configs/river_smoke.yaml   # quick smoke (T=8, no STA)

# CPAZMaL SAR classification (Weibull, 4 methods × 2 modes — STA bary ~2h)
python experiments/run_classification.py configs/cpazmal.yaml
# → first run extracts from HDF5 (slow), caches to data/cpazmal/
# → set max_groups: 6 in cpazmal.yaml for a quick smoke test (3 methods, no STA)

# Fit and save barycenters
python experiments/run_barycenters.py configs/river.yaml

# Sensitivity sweeps (synthetic exponential only, KNN mode, 3 methods — no STA)
python experiments/run_sensitivity.py --output-dir results/jax_sensitivity
```

Data paths:
- CPAZMaL HDF5: `/home/mgallet/Documents/Codes/Python/1_DONE/CPAZMAL/DATASET/dataset_original/PAZTSX_CRYO_ML.hdf5`
- River NetCDF (raw): `/home/mgallet/Documents/Dataset/RIVER_DISCHARGES/c7491e060d94c97212f0fe7ebcff57f0/data_version-5.nc`
- River npy (prebuilt): `data/river/` (committed; rebuild with Explore2_HydroDataset project)
- CPAZMaL npy cache: `data/cpazmal/` (gitignored, auto-populated on first run)

## Architecture

```
data/
├── river/          # Prebuilt: X/Y/metadata_{balanced,basic}.npy
└── cpazmal/        # Extraction cache (gitignored)

src/
├── distributions.py          # JAX distributions: exponential + Weibull
├── estimation.py             # Log-cumulant / MLE fitting (outside JAX graph)
├── costs.py                  # SqEuclidean + WaSPS W₂² (closed-form, autodiff; log_correction, use_positivity_constraint)
├── softdtw.py                # SoftDTW forward + divergence; SoftDTW class (manual/auto × div/not-div)
├── barycenter.py             # Fréchet barycenter via optax (adam); fit_barycenter(series, softdtw, …)
├── data/
│   ├── preprocess.py         # clean_series() — canonical filter for estimation
│   ├── cpazmal_loader.py     # MLDatasetLoader + extract_time_series (HDF5)
│   └── river_loader.py       # load_river_classification() — npy + stratified split
├── classification/
│   ├── nn.py                 # k-NN SoftDTW (vmap over train set)
│   └── barycenter_clf.py     # Nearest-barycenter (joblib-parallel per class)
└── baselines/
    └── sta_wrapper.py        # STA 1-NN + make_cost_fn() for STA barycenter

experiments/
├── run_classification.py     # 4-method × 2-mode × multi-seed runner
├── run_barycenters.py        # Fit + save per-class barycenters as .npy
└── run_sensitivity.py        # γ / n_train sweeps (KNN, 3 methods — no STA)

configs/
├── classification.yaml       # Synthetic exponential (4 methods × 2 modes, 4 seeds)
├── classification_smoke.yaml # Gate-E smoke: T=4, n_steps=5 — all 4 methods × 2 modes in ~18s
├── river.yaml                # River discharge (wasps/eucl_params/eucl_raw)
├── river_smoke.yaml          # River smoke (T=8, 3 methods, no STA — too slow at T=8)
└── cpazmal.yaml              # CPAZMaL Weibull (wasps/eucl_params/eucl_raw)

analysis/
├── classification_notebook.ipynb  # Interactive: synthetic / CPAZMaL
└── river_notebook.ipynb           # Interactive: river discharge

tests/                        # pytest suite

.old/                         # Archived legacy code (old Cython/PyTorch pipeline)
```

## Key conventions

**Array shapes:**
- Raw time series: `(T, N_samples)` float64 — T timesteps, N values each
- River samples: `(T, D·W·W)` after reshape from `(T, D, W, W)` raw npy
- Exponential params: `(T, 1)` — rate β (not scale, `E[X]=1/β`)
- Weibull params: `(T, 2)` — column 0=k (shape), column 1=λ_scale
- NaN preserved through load → filtered by `clean_series` inside `estimation.fit`

**Four classification methods (defined in `experiments/run_classification.py:_METHODS`):**

| Key | Representation | Cost | Barycenter | Notes |
|-----|----------------|------|------------|-------|
| `wasps` | params `(T,1)` or `(T,2)` | WaSPS W₂² closed-form | `use_positivity_constraint=True`, manual grad | exponential or Weibull |
| `eucl_params` | params (MLE) | SqEuclidean | autodiff | no positivity constraint in barycenter |
| `eucl_raw` | raw samples `(T,N)` | SqEuclidean | autodiff | raw order preserved |
| `sta` | raw samples `(T,N)` | OT Sinkhorn (OTT) | autodiff through Sinkhorn | slow: O(T²·N·n_train) |

Both KNN (k=1) and Barycenter modes are supported for all 4 methods.

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
               dtype=jnp.float64, patience=15, min_rel_improve=1e-4)
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
Measured: CPAZMaL STA/bary (T=24, n_classes=10, 2 train/class) ≈ 6h. JAX recompiles per class
  (closure captures training data as embedded constants → no cache reuse). Fix: pass data as
  stacked JAX argument, not closed-over list. Not yet implemented.  
`configs/river.yaml` excludes STA (T=365 daily — intractable).  
`configs/river_sta.yaml` uses `max_time_steps: 52` (truncated) to make STA **KNN** tractable (~12 min/seed).  
  STA barycenter at T=52 is also intractable (T²=2704 Sinkhorn/step ≈ 12h → only modes: [knn] in river_sta.yaml).  
`configs/river_smoke.yaml` (T=8) also excludes STA.  
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

**cpazmal cache:**  
`data/cpazmal/X_train_{all|mgN}.npy` etc. Delete to force re-extraction from HDF5.

**Import pattern:** All modules use `sys.path.insert(0, str(_SRC))` — no `pip install -e .`.
Experiments run from repo root, tests from repo root (not `src/`).

**Exponential parameterization:** Rate β (not scale), so `E[X] = 1/β`.  
**Weibull parameterization:** Scale λ (not rate), so `E[X^k] = λ^k`.
