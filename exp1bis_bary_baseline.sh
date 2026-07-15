#!/usr/bin/env bash
# Experiment 1bis — barycenter baseline, ALL 7 methods (incl. STA — a real, full-scale
# participant here, not excluded), both datasets, per-method optimal gamma.
# Requires experiments/optimize_gamma.py to have already been run (writes
# results/gamma_search/{river,cpazmal}/best_params.json — see README/plan).
#
# STA barycenter fitting at full scale is expected to take a long time (hours-scale — see
# CLAUDE.md's documented STA-barycenter cost warning); this is accepted, not a bug.
#
# Ends with: per-dataset F1/time/RAM x method LaTeX table, a train/test series+labels +
# fitted-barycenters .npy dump per dataset, and the river 2-class (NG/PN) illustrative
# barycenter-vs-test-samples figure (wasps/eucl_params/eucl_raw + a narrow STA fit).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate

CONFIG="configs/exp1_baseline.yaml"
GAMMA_DIR="results/gamma_search"
OUT_DIR="results/jax_exp1bis_bary_baseline"
RESULTS_DIR="$OUT_DIR/results"
LOG="$OUT_DIR/exp1bis_bary_baseline.sh.log"
mkdir -p "$RESULTS_DIR"

echo "===== $(date '+%Y-%m-%d %H:%M:%S') exp1bis_bary_baseline starting =====" | tee -a "$LOG"

for DATASET in river cpazmal; do
  echo "----- $DATASET -----" | tee -a "$LOG"

  EXTRA_ARGS=()
  if [[ "$DATASET" == "river" ]]; then
    EXTRA_ARGS=(--bary-plot-dataset river --bary-plot-methods wasps,eucl_params,eucl_raw)
  fi

  python experiments/run_full_baseline.py --config "$CONFIG" --output-dir "$OUT_DIR" \
      --dataset "$DATASET" --modes barycenter --n-jobs 4 --sta-n-jobs 2 \
      --bary-methods-same-as-methods \
      --gamma-by-method-json "$GAMMA_DIR/$DATASET/best_params.json" --gamma-by-method-key bary \
      "${EXTRA_ARGS[@]}" --debug --verbose \
      2>&1 | tee -a "$LOG"

  python experiments/extract_latex_tables.py exp1 \
      --detail-csv "$OUT_DIR/full_baseline_${DATASET}_detail.csv" \
      --out "$RESULTS_DIR/exp1bis_${DATASET}.tex" --mode barycenter \
      --dataset-label "$DATASET" --label-prefix "tab:exp1bis_${DATASET}" \
      2>&1 | tee -a "$LOG"

  python experiments/dump_arrays.py --config "$CONFIG" --dataset "$DATASET" --seed 42 \
      --out-series "$OUT_DIR/${DATASET}_series.npy" \
      --out-barycenters "$OUT_DIR/${DATASET}_barycenters.npy" \
      --methods wasps,wasps_nodiv,eucl_params,eucl_params_nodiv,eucl_raw,eucl_raw_nodiv,sta \
      --gamma-by-method-json "$GAMMA_DIR/$DATASET/best_params.json" \
      2>&1 | tee -a "$LOG"
done

python experiments/plot_exp1_river_bary.py --config "$CONFIG" \
    --exp1-out-dir "$OUT_DIR" --output-dir "$RESULTS_DIR" \
    2>&1 | tee -a "$LOG"

echo "===== $(date '+%Y-%m-%d %H:%M:%S') exp1bis_bary_baseline done =====" | tee -a "$LOG"
