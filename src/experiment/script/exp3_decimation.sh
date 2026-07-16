#!/usr/bin/env bash
# Decimation sweep — cpazmal only, barycenter mode, divergence methods + STA, at the
# per-method optimal gamma from the baseline experiment's gamma search. Requires
# exp2_bary_baseline.sh (or at least optimize_gamma.py for cpazmal) to have already run
# (writes results/gamma_search/cpazmal/best_params.json).
#
# Phase 1 (run): run_decimation.py handles the non-STA (samples_per_step=480, from
# config_decimation.yaml) / STA (samples_per_step=48) split internally, one call each.
#
# Phase 2 (extract): decimation LaTeX table (F1 %, rows=method, columns=fraction).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source .venv/bin/activate

CONFIG="configs/config_decimation.yaml"
GAMMA_DIR="results/gamma_search"
OUT_DIR="results/jax_exp3_decimation"
RESULTS_DIR="$OUT_DIR/results"
LOG="$OUT_DIR/exp3_decimation.sh.log"
mkdir -p "$RESULTS_DIR"
cp "$CONFIG" "$OUT_DIR/"

echo "===== $(date '+%Y-%m-%d %H:%M:%S') exp3_decimation starting =====" | tee -a "$LOG"

# ---- Phase 1: run ----
python src/experiment/run_decimation.py --config "$CONFIG" \
    --gamma-search-dir "$GAMMA_DIR" --output-dir "$OUT_DIR" \
    --n-jobs 4 --sta-n-jobs 2 --sta-samples-per-step 48 --debug --verbose \
    2>&1 | tee -a "$LOG"

# ---- Phase 2: extract ----
python src/experiment/reporting/extract_latex_tables.py sensitivity \
    --dir "$OUT_DIR" --scenario decimation --mode barycenter \
    --out "$RESULTS_DIR/exp3_decimation.tex" \
    --caption "CPAZMaL decimation sweep, barycenter mode, per-method optimal gamma" \
    --label tab:exp3_decimation \
    2>&1 | tee -a "$LOG"

echo "===== $(date '+%Y-%m-%d %H:%M:%S') exp3_decimation done =====" | tee -a "$LOG"
