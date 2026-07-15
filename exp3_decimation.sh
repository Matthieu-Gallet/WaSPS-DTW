#!/usr/bin/env bash
# Experiment 3 — cpazmal decimation sweep, barycenter mode, divergence methods + STA, at the
# per-method optimal gamma found by Experiment 1bis (results/gamma_search/cpazmal/best_params.json).
# Requires exp1bis_bary_baseline.sh (or at least experiments/optimize_gamma.py for cpazmal) to
# have already been run.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate

CONFIG="configs/exp3_decimation.yaml"
GAMMA_DIR="results/gamma_search"
OUT_DIR="results/jax_exp3_decimation"
RESULTS_DIR="$OUT_DIR/results"
LOG="$OUT_DIR/exp3_decimation.sh.log"
mkdir -p "$RESULTS_DIR"

echo "===== $(date '+%Y-%m-%d %H:%M:%S') exp3_decimation starting =====" | tee -a "$LOG"

python experiments/run_exp3_decimation.py --config "$CONFIG" \
    --gamma-search-dir "$GAMMA_DIR" --n-jobs 4 --sta-n-jobs 2 --debug --verbose \
    2>&1 | tee -a "$LOG"

python experiments/extract_latex_tables.py sensitivity \
    --dir "$OUT_DIR" --scenario decimation --mode barycenter \
    --out "$RESULTS_DIR/exp3_decimation.tex" \
    --caption "CPAZMaL decimation sweep, barycenter mode, per-method optimal gamma" \
    --label tab:exp3_decimation \
    2>&1 | tee -a "$LOG"

echo "===== $(date '+%Y-%m-%d %H:%M:%S') exp3_decimation done =====" | tee -a "$LOG"
