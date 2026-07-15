#!/usr/bin/env bash
# Experiment 1 — KNN baseline, 7 methods, both datasets, per-method optimal gamma.
# Requires experiments/optimize_gamma.py to have already been run (writes
# results/gamma_search/{river,cpazmal}/best_params.json — see README/plan).
#
# Ends with: per-dataset F1/time/RAM x method LaTeX table, and a train/test
# series+labels .npy dump per dataset.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate

CONFIG="configs/exp1_baseline.yaml"
GAMMA_DIR="results/gamma_search"
OUT_DIR="results/jax_exp1_knn_baseline"
RESULTS_DIR="$OUT_DIR/results"
LOG="$OUT_DIR/exp1_knn_baseline.sh.log"
mkdir -p "$RESULTS_DIR"

echo "===== $(date '+%Y-%m-%d %H:%M:%S') exp1_knn_baseline starting =====" | tee -a "$LOG"

for DATASET in river cpazmal; do
  echo "----- $DATASET -----" | tee -a "$LOG"

  python experiments/run_full_baseline.py --config "$CONFIG" --output-dir "$OUT_DIR" \
      --dataset "$DATASET" --modes knn --n-jobs 4 --sta-n-jobs 2 \
      --gamma-by-method-json "$GAMMA_DIR/$DATASET/best_params.json" --gamma-by-method-key knn \
      --debug --verbose \
      2>&1 | tee -a "$LOG"

  python experiments/extract_latex_tables.py exp1 \
      --detail-csv "$OUT_DIR/full_baseline_${DATASET}_detail.csv" \
      --out "$RESULTS_DIR/exp1_${DATASET}.tex" --mode knn \
      --dataset-label "$DATASET" --label-prefix "tab:exp1_${DATASET}" \
      2>&1 | tee -a "$LOG"

  python experiments/dump_arrays.py --config "$CONFIG" --dataset "$DATASET" --seed 42 \
      --out-series "$OUT_DIR/${DATASET}_series.npy" \
      2>&1 | tee -a "$LOG"
done

echo "===== $(date '+%Y-%m-%d %H:%M:%S') exp1_knn_baseline done =====" | tee -a "$LOG"
