#!/bin/bash
# Meta script: launches the full WaSPS-DTW experiment suite sequentially, with
# timestamped logging per step. Replaces the pre-feat/jax-refonte version of this
# script (stale: referenced removed src/experiments/*.py entry points and a
# nonexistent `venv/` — this repo uses `.venv/`, set up via setup_venv.sh).
#
# The 7 config files this suite drives:
#   configs/sensitivity_river.yaml / sensitivity_cpazmal.yaml   (grid search + sweeps)
#   configs/classification/river.yaml / cpazmal.yaml            (KNN+bary gamma sweep)
#   configs/classification/river_bary.yaml / cpazmal_bary.yaml  (barycenter fit+eval+save)
#   configs/full_baseline.yaml                                  (final KNN-only baseline)
#
# Order matters: run_optim_hyper.py's grid_knn/grid_bary must run before
# run_sensitivity.py's sweeps (which read the best_params.json grid search writes).
# Everything else is independent and could be reordered/parallelised across
# machines if needed — kept sequential here for simplicity and log clarity.

set -e
cd "$(dirname "$0")"
source .venv/bin/activate

_step() {
    local name="$1"
    shift
    echo "=== $name started $(date) ===" | tee -a run_all_experiments.log
    "$@"
    echo "=== $name done $(date) ===" | tee -a run_all_experiments.log
}

echo "=== run_all_experiments.sh started $(date) ===" | tee run_all_experiments.log

# --- 1. Hyperparameter grid search (calibration) ---------------------------
_step "river/grid_knn"    python experiments/run_optim_hyper.py --config configs/sensitivity_river.yaml   --scenario grid_knn  --n-jobs -1
_step "river/grid_bary"   python experiments/run_optim_hyper.py --config configs/sensitivity_river.yaml   --scenario grid_bary --n-jobs -1
_step "cpazmal/grid_knn"  python experiments/run_optim_hyper.py --config configs/sensitivity_cpazmal.yaml --scenario grid_knn  --n-jobs -1
_step "cpazmal/grid_bary" python experiments/run_optim_hyper.py --config configs/sensitivity_cpazmal.yaml --scenario grid_bary --n-jobs -1

# --- 2. Sensitivity sweeps (read best_params.json from step 1) -------------
_step "river/sweep_n_samples"    python experiments/run_sensitivity.py --config configs/sensitivity_river.yaml   --scenario sweep_n_samples   --n-jobs -1
_step "river/sweep_n_train"      python experiments/run_sensitivity.py --config configs/sensitivity_river.yaml   --scenario sweep_n_train     --n-jobs -1
_step "river/sweep_decimation"   python experiments/run_sensitivity.py --config configs/sensitivity_river.yaml   --scenario sweep_decimation  --n-jobs -1
_step "cpazmal/sweep_n_samples"  python experiments/run_sensitivity.py --config configs/sensitivity_cpazmal.yaml --scenario sweep_n_samples   --n-jobs -1
_step "cpazmal/sweep_n_train"    python experiments/run_sensitivity.py --config configs/sensitivity_cpazmal.yaml --scenario sweep_n_train     --n-jobs -1
_step "cpazmal/sweep_decimation" python experiments/run_sensitivity.py --config configs/sensitivity_cpazmal.yaml --scenario sweep_decimation  --n-jobs -1

# --- 3. Classification gamma sweep (KNN + barycenter, independent of 1-2) --
_step "river/classification"   python experiments/run_classification.py configs/classification/river.yaml   --n-jobs -1
_step "cpazmal/classification"  python experiments/run_classification.py configs/classification/cpazmal.yaml  --n-jobs -1

# --- 4. Barycenter fit + evaluate + export (independent of 1-3) ------------
_step "river/barycenters"   python experiments/run_barycenters.py configs/classification/river_bary.yaml   --n-jobs -1
_step "cpazmal/barycenters"  python experiments/run_barycenters.py configs/classification/cpazmal_bary.yaml  --n-jobs -1

# --- 5. Final baseline comparison (KNN-only, both datasets, div vs nodiv) --
_step "full_baseline" python experiments/run_full_baseline.py --config configs/full_baseline.yaml --n-jobs -1

echo "=== ALL DONE $(date) ===" | tee -a run_all_experiments.log
