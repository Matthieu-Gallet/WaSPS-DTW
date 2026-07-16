#!/usr/bin/env bash
# Baseline experiment — KNN mode, both datasets, per-method optimal gamma.
#
# Phase 1 (run): calibrate gamma if results/gamma_search/{dataset}/best_params.json is
# missing (shared with exp2_bary_baseline.sh — skipped here if that script already ran),
# then run all 6 non-STA methods at samples_per_step=480, then STA alone at 48.
#
# Phase 2 (extract): per-dataset + combined F1/Time/RAM x method LaTeX table, plus a
# train/test series+labels .npy dump per dataset.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source .venv/bin/activate

CONFIG="configs/config_baseline.yaml"
GAMMA_DIR="results/gamma_search"
OUT_DIR="results/jax_exp1_knn_baseline"
RESULTS_DIR="$OUT_DIR/results"
LOG="$OUT_DIR/exp1_knn_baseline.sh.log"
mkdir -p "$RESULTS_DIR"
cp "$CONFIG" "$OUT_DIR/"

echo "===== $(date '+%Y-%m-%d %H:%M:%S') exp1_knn_baseline starting =====" | tee -a "$LOG"

# ---- Phase 1: run ----
for DATASET in river cpazmal; do
  if [[ ! -f "$GAMMA_DIR/$DATASET/best_params.json" ]]; then
    echo "----- $DATASET: calibrating gamma -----" | tee -a "$LOG"
    python src/experiment/optimize_gamma.py --config "$CONFIG" --dataset "$DATASET" \
        --n-jobs 4 --sta-n-jobs 2 --out-dir "$GAMMA_DIR" --debug --verbose \
        2>&1 | tee -a "$LOG"
  fi
done

for DATASET in river cpazmal; do
  echo "----- $DATASET: non-STA methods (samples_per_step=480) -----" | tee -a "$LOG"
  python src/experiment/run_full_baseline.py --config "$CONFIG" --output-dir "$OUT_DIR" \
      --dataset "$DATASET" --modes knn --n-jobs 4 --sta-n-jobs 2 \
      --methods wasps,wasps_nodiv,eucl_params,eucl_params_nodiv,eucl_raw,eucl_raw_nodiv \
      --gamma-by-method-json "$GAMMA_DIR/$DATASET/best_params.json" --gamma-by-method-key knn \
      --debug --verbose \
      2>&1 | tee -a "$LOG"

  echo "----- $DATASET: STA (samples_per_step=48) -----" | tee -a "$LOG"
  python src/experiment/run_full_baseline.py --config "$CONFIG" --output-dir "$OUT_DIR" \
      --dataset "$DATASET" --modes knn --n-jobs 4 --sta-n-jobs 2 \
      --methods sta --samples-per-step 48 \
      --gamma-by-method-json "$GAMMA_DIR/$DATASET/best_params.json" --gamma-by-method-key knn \
      --debug --verbose \
      2>&1 | tee -a "$LOG"
done

# ---- Phase 2: extract ----
for DATASET in river cpazmal; do
  python src/experiment/reporting/extract_latex_tables.py exp1 \
      --detail-csv "$OUT_DIR/full_baseline_${DATASET}_detail.csv" \
      --out "$RESULTS_DIR/exp1_${DATASET}_knn.tex" --mode knn \
      --dataset-label "$DATASET" --label-prefix "tab:exp1_${DATASET}" \
      2>&1 | tee -a "$LOG"

  python src/experiment/reporting/dump_arrays.py --config "$CONFIG" --dataset "$DATASET" --seed 42 \
      --out-series "$OUT_DIR/${DATASET}_series.npy" \
      2>&1 | tee -a "$LOG"
done

python src/experiment/reporting/extract_latex_tables.py exp1_combined \
    --river-detail-csv "$OUT_DIR/full_baseline_river_detail.csv" \
    --cpazmal-detail-csv "$OUT_DIR/full_baseline_cpazmal_detail.csv" \
    --mode knn --out "$RESULTS_DIR/exp1_knn.tex" \
    2>&1 | tee -a "$LOG"

echo "===== $(date '+%Y-%m-%d %H:%M:%S') exp1_knn_baseline done =====" | tee -a "$LOG"
