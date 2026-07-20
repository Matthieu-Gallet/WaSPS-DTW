#!/usr/bin/env bash
# Baseline experiment — barycenter mode, both datasets, per-method optimal gamma.
# Both divergence methods (6 total: wasps/eucl/raw ± _nodiv) and STA.
#
# Phase 1 (run): calibrate gamma if results/gamma_search/{dataset}/best_params.json is
# missing (shared with exp1_knn_baseline.sh — skipped here if that script already ran),
# then run 6 non-STA methods at samples_per_step=480, then STA alone at 48.
#
# Phase 2 (extract): per-dataset + combined F1/Time/RAM x method LaTeX table. The
# train/test series+labels + fitted-barycenters .npy dump (all 7 methods) is OPTIONAL —
# it refits every barycenter from scratch (nothing from Phase 1 is kept in memory across
# these separate script invocations), so for STA this re-runs the same O(T^2)-per-step
# fit that Phase 1 already paid for, at reduced (bary_n_jobs=1) parallelism — river/STA
# alone took 3h50+ standalone. exp4_river_bary_viz.sh now covers the river barycenter
# visualization independently (its own gamma/sample settings), so this dump is normally
# unneeded — set DUMP_ARRAYS=true to re-enable it (e.g. for cpazmal-specific plots).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source .venv/bin/activate

CONFIG="configs/config_baseline.yaml"
GAMMA_DIR="results/gamma_search"
OUT_DIR="results/jax_exp2_bary_baseline"
RESULTS_DIR="$OUT_DIR/results"
LOG="$OUT_DIR/exp2_bary_baseline.sh.log"
DUMP_ARRAYS="${DUMP_ARRAYS:-false}"
mkdir -p "$RESULTS_DIR"
cp "$CONFIG" "$OUT_DIR/"

echo "===== $(date '+%Y-%m-%d %H:%M:%S') exp2_bary_baseline starting =====" | tee -a "$LOG"

# ---- Phase 1: run ----
for DATASET in river cpazmal; do
  if [[ ! -f "$GAMMA_DIR/$DATASET/best_params.json" ]]; then
    echo "----- $DATASET: calibrating gamma -----" | tee -a "$LOG"
    python src/experiment/optimize_gamma.py --config "$CONFIG" --dataset "$DATASET" \
        --n-jobs 4 --sta-n-jobs 2 --out-dir "$GAMMA_DIR" --debug --verbose \
        2>&1 | tee -a "$LOG"
  fi

  echo "----- $DATASET: barycenter, 6 non-STA methods (samples_per_step=480) -----" | tee -a "$LOG"
  python src/experiment/run_full_baseline.py --config "$CONFIG" --output-dir "$OUT_DIR" \
      --dataset "$DATASET" --modes barycenter --n-jobs 4 \
      --methods wasps,wasps_nodiv,eucl_params,eucl_params_nodiv,eucl_raw,eucl_raw_nodiv \
      --bary-methods-same-as-methods \
      --gamma-by-method-json "$GAMMA_DIR/$DATASET/best_params.json" --gamma-by-method-key bary \
      --debug --verbose \
      2>&1 | tee -a "$LOG"

  echo "----- $DATASET: barycenter, STA (samples_per_step=48) -----" | tee -a "$LOG"
  python src/experiment/run_full_baseline.py --config "$CONFIG" --output-dir "$OUT_DIR" \
      --dataset "$DATASET" --modes barycenter --n-jobs 4 --sta-n-jobs 2 \
      --methods sta --samples-per-step 48 \
      --bary-methods-same-as-methods \
      --gamma-by-method-json "$GAMMA_DIR/$DATASET/best_params.json" --gamma-by-method-key bary \
      --debug --verbose \
      2>&1 | tee -a "$LOG"
done

# ---- Phase 2: extract ----
for DATASET in river cpazmal; do
  python src/experiment/reporting/extract_latex_tables.py exp1 \
      --detail-csv "$OUT_DIR/full_baseline_${DATASET}_detail.csv" \
      --out "$RESULTS_DIR/exp2_${DATASET}_barycenter.tex" --mode barycenter \
      --dataset-label "$DATASET" --label-prefix "tab:exp2_${DATASET}" \
      2>&1 | tee -a "$LOG"

  if [[ "$DUMP_ARRAYS" == "true" ]]; then
    python src/experiment/reporting/dump_arrays.py --config "$CONFIG" --dataset "$DATASET" --seed 42 \
        --out-series "$OUT_DIR/${DATASET}_series.npy" \
        --out-barycenters "$OUT_DIR/${DATASET}_barycenters.npy" \
        --methods wasps,wasps_nodiv,eucl_params,eucl_params_nodiv,eucl_raw,eucl_raw_nodiv,sta \
        --gamma-by-method-json "$GAMMA_DIR/$DATASET/best_params.json" \
        2>&1 | tee -a "$LOG"
  fi
done

python src/experiment/reporting/extract_latex_tables.py exp1_combined \
    --river-detail-csv "$OUT_DIR/full_baseline_river_detail.csv" \
    --cpazmal-detail-csv "$OUT_DIR/full_baseline_cpazmal_detail.csv" \
    --mode barycenter --out "$RESULTS_DIR/exp2_barycenter.tex" \
    2>&1 | tee -a "$LOG"

echo "===== $(date '+%Y-%m-%d %H:%M:%S') exp2_bary_baseline done =====" | tee -a "$LOG"
