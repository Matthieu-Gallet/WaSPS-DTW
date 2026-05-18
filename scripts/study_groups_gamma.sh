#!/bin/bash
# Sweep gamma for the barycenter groups experiment.
# Usage: bash scripts/study_groups_gamma.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

GAMMAS="0.01 1.0 100.0"
GROUPS_DIR="${DATASET_DIR}/discharge_groups"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="$RESULTS_DIR/study_groups_gamma_$TIMESTAMP"
mkdir -p "$OUT"
cp "$0" "$OUT/"

for GAMMA in $GAMMAS; do
    SUBDIR="$OUT/gamma_${GAMMA}"
    echo "=== gamma=$GAMMA ==="
    python experiments/barycenter_groups.py \
        --groups-dir "$GROUPS_DIR" \
        --gamma      "$GAMMA" \
        --normalize \
        --output-dir "$SUBDIR"
done

echo "Study complete → $OUT"
