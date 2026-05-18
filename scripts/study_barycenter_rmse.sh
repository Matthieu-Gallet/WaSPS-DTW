#!/bin/bash
# Sweep n_samples × estimator × gamma for barycenter RMSE analysis.
# One process per grid cell.
# Usage: bash scripts/study_barycenter_rmse.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

N_SAMPLES_LIST="5 25 100 1000 100000"
ESTIMATORS="mle log_cumulant"
GAMMAS="0.01 1.0 1000.0"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="$RESULTS_DIR/study_barycenter_rmse_$TIMESTAMP"
mkdir -p "$OUT"
cp "$0" "$OUT/"

for N in $N_SAMPLES_LIST; do
    for EST in $ESTIMATORS; do
        for GAMMA in $GAMMAS; do
            SUBDIR="$OUT/n${N}_est_${EST}_gamma_${GAMMA}"
            echo "=== n_samples=$N  estimator=$EST  gamma=$GAMMA ==="
            python experiments/barycenter_rmse.py \
                --n-samples  "$N" \
                --estimator  "$EST" \
                --gamma      "$GAMMA" \
                --output-dir "$SUBDIR"
        done
    done
done

echo "Study complete → $OUT"
