#!/usr/bin/env bash
# Full sensitivity pipeline (both datasets): hyperparameter calibration (grid_knn,
# grid_bary) followed by the 3 sensitivity sweeps (n_samples, n_train, decimation),
# then LaTeX table + barycenter-plot extraction — all sequential, one OS process per
# scenario×dataset invocation (same "let the process exit reclaim memory" pattern as
# run_full_baseline_sweep.sh).
#
# --n-jobs now defaults to 4 in both run_optim_hyper.py and run_sensitivity.py (was -1)
# — grid_bary nests fit_barycenters' own n_jobs=4 per-class parallelism, so an unbounded
# outer -1 (os.cpu_count()=20 workers regardless of the ~16-20 actual grid points) times
# the inner 4 could reach ~80 concurrent JAX processes, the same uncapped-nesting
# pattern that OOM-crashed the machine during run_full_baseline.py's STA phase.
#
# Run AFTER run_full_baseline_sweep.sh completes — do not run both concurrently.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate

N_JOBS=4
LOG="results/sensitivity_pipeline.log"
mkdir -p results results/tables

run() {
  echo "----- $(date '+%H:%M:%S') $* starting -----" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  echo "----- $(date '+%H:%M:%S') $* done -----" | tee -a "$LOG"
}

echo "===== $(date '+%Y-%m-%d %H:%M:%S') sensitivity_pipeline starting =====" | tee -a "$LOG"

for cfg in configs/sensitivity_river.yaml configs/sensitivity_cpazmal.yaml; do
  run python experiments/run_optim_hyper.py --config "$cfg" --scenario grid_knn  --n-jobs "$N_JOBS" --debug --verbose
  run python experiments/run_optim_hyper.py --config "$cfg" --scenario grid_bary --n-jobs "$N_JOBS" --debug --verbose
  run python experiments/run_sensitivity.py --config "$cfg" --scenario sweep_n_samples  --n-jobs "$N_JOBS" --debug --verbose
  run python experiments/run_sensitivity.py --config "$cfg" --scenario sweep_n_train    --n-jobs "$N_JOBS" --debug --verbose
  run python experiments/run_sensitivity.py --config "$cfg" --scenario sweep_decimation --n-jobs "$N_JOBS" --debug --verbose
done

echo "===== $(date '+%H:%M:%S') sensitivity_pipeline sweeps done — extraction starting =====" | tee -a "$LOG"

# --- LaTeX tables: full_baseline (both datasets) ---
run python experiments/extract_latex_tables.py full_baseline \
    --detail-csv results/jax_full_baseline/full_baseline_river_detail.csv \
    --dataset-label "River" --label-prefix tab:baseline_river \
    --out results/tables/full_baseline_river.tex
run python experiments/extract_latex_tables.py full_baseline \
    --detail-csv results/jax_full_baseline/full_baseline_cpazmal_detail.csv \
    --dataset-label "CPAZMaL" --label-prefix tab:baseline_cpazmal \
    --out results/tables/full_baseline_cpazmal.tex

# --- LaTeX tables: sensitivity sweeps (both datasets x 3 scenarios x 2 modes) ---
declare -A SENS_DIRS=( [river]="results/jax_sensitivity" [cpazmal]="results/jax_sensitivity_cpazmal" )
for ds in river cpazmal; do
  dir="${SENS_DIRS[$ds]}"
  for scenario in n_samples n_train decimation; do
    for mode in knn barycenter; do
      run python experiments/extract_latex_tables.py sensitivity \
          --dir "$dir" --scenario "$scenario" --mode "$mode" \
          --out "results/tables/sensitivity_${ds}_${scenario}_${mode}.tex" \
          --label "tab:sens_${ds}_${scenario}_${mode}"
    done
  done
done

# --- Barycenter debug plots (river PN/NG classes) ---
run python experiments/extract_bary_plots.py \
    --bary-debug-dir results/jax_sensitivity/bary_debug \
    --output-dir results/jax_sensitivity/pn_ng_plots

echo "===== $(date '+%Y-%m-%d %H:%M:%S') sensitivity_pipeline + extraction done =====" | tee -a "$LOG"
