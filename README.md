# WaSPS-DTW

**Wasserstein + Soft-DTW** time series analysis in JAX: computing barycenters and
classifying *stochastic* time series — sequences where each timestep is a probability
distribution rather than a single scalar — using Soft-DTW with Euclidean or
Wasserstein (WaSPS) local costs, for exponential or Weibull distribution families.

Two real datasets are supported: river discharge (exponential family) and CPAZMaL SAR
backscatter (Weibull family). This is the JAX rewrite (branch `feat/jax-refonte`); the
previous Cython/PyTorch pipeline is archived under `.old/`.

## Installation

```bash
bash setup_venv.sh
source .venv/bin/activate

# Smoke test
python -c "import jax, ott, optax; print(jax.devices())"

# Run the test suite (from repo root, not src/)
python -m pytest tests/ -q
```

`setup_venv.sh` uses [`uv`](https://github.com/astral-sh/uv) to create `.venv/` and
install `requirements.txt` — no Cython/compiled-extension build step.

## Datasets

Neither dataset ships in the repo (`data/**/*.npy` is gitignored) — both are built
once and cached locally.

### CPAZMaL (SAR, Weibull family)

Retrieve from HuggingFace:
[`musmb/CPAZMaL`](https://huggingface.co/datasets/musmb/CPAZMaL).

```python
import sys; sys.path.insert(0, "src")   # repo's import convention — no pip install -e
from data.cpazmal_loader import download_cpazmal
hdf5_path = download_cpazmal(save_dir="path/to/local/dir")
```

Then point `dataset.hdf5_path` in `configs/config_baseline.yaml` /
`configs/config_decimation.yaml` at the resulting `.hdf5` file. The loader
(`MLDatasetLoader` in `src/data/cpazmal_loader.py`) is a verbatim port from
[`Matthieu-Gallet/CPAZMaL_dataset`](https://github.com/Matthieu-Gallet/CPAZMaL_dataset).
The HDF5 is read once and cached to `data/cpazmal/` (gitignored) on first run.

### River discharge (exponential family)

Built externally via
[`Matthieu-Gallet/Explore2_HydroDataset`](https://github.com/Matthieu-Gallet/Explore2_HydroDataset);
place the resulting `.npy` files under `data/river/`:

- `X_balanced.npy`, `Y_balanced.npy`, `metadata_balanced.npy` — required.
- `groups_balanced.npy` — optional, enables geography-aware K-fold
  (`StratifiedGroupKFold`) instead of a plain stratified split.

Current known scale: 800 samples / 4 hydrological-regime classes / 183 geographic
groups.

## Data description

| Dataset | Family | Shape | Classes | Notes |
|---|---|---|---|---|
| River discharge | Exponential | `(T=365, D·W·W)` per sample, daily | 4 (hydrological regimes) | Can be temporally aggregated (e.g. weekly, T=52) |
| CPAZMaL (SAR) | Weibull | `(T≈29, W²)` per spatial window, one year | Multiple land-cover classes | K-fold by geographic `group`, mirrors the river split |

Both are loaded as raw `(T, N_samples)` series; per-timestep distribution parameters
(β for exponential, `(k, λ)` for Weibull) are estimated on demand via
`src/distributions.py` (MLE or method-of-log-cumulants).

## Experiments

Four numbered experiments, each a shell script in `src/experiment/script/`, run from
the repo root after activating the venv:

```bash
bash src/experiment/script/exp1_knn_baseline.sh      # KNN baseline, both datasets, 7 methods
bash src/experiment/script/exp2_bary_baseline.sh      # barycenter baseline, same scope
bash src/experiment/script/exp3_decimation.sh         # CPAZMaL decimation sweep (barycenter mode)
bash src/experiment/script/exp4_river_bary_viz.sh     # standalone river barycenter figure
```

Methods compared: `wasps` (Wasserstein closed-form cost), `eucl_params` (Euclidean
cost on estimated distribution parameters), `eucl_raw` (Euclidean cost on raw
samples), and `sta` (Sinkhorn Transport Alignment baseline), each in both divergence
and non-divergence Soft-DTW variants. Configs:
`configs/config_baseline.yaml` (exp1/exp2) and `configs/config_decimation.yaml`
(exp3) — exp4 takes its parameters directly as script arguments.

See `RUN.md` for the full per-experiment command reference (expected runtimes, gamma
calibration flow, STA timing) and `CLAUDE.md` for architecture and implementation
conventions.

## Repository layout

```
src/
├── distributions.py, costs.py, softdtw.py, barycenter.py   # Core JAX library
├── data/            # Dataset loaders + preprocessing
├── classification/  # k-NN, nearest-barycenter, STA classifiers
├── plot/             # Plotting + exp4 driver
├── experiment/        # Experiment runners (exp1-3), gamma calibration, reporting
└── dev/               # Historical/exploratory scripts and notebooks

configs/    # config_baseline.yaml (exp1/exp2), config_decimation.yaml (exp3)
tests/      # pytest suite
.old/       # Archived legacy code (pre-JAX pipeline) and superseded docs/scripts
```
