#!/bin/bash
# Sweep gamma for both barycenter scenarios (simple + complex).
# One process per (scenario, gamma) combination.
# Usage: bash scripts/study_barycenter_gamma.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

GAMMAS="0.01 1.0 1000.0"
SCENARIOS="simple complex"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="$RESULTS_DIR/study_barycenter_gamma_$TIMESTAMP"
mkdir -p "$OUT"
cp "$0" "$OUT/"

for SCENARIO in $SCENARIOS; do
    for GAMMA in $GAMMAS; do
        SUBDIR="$OUT/${SCENARIO}_gamma_${GAMMA}"
        echo "=== $SCENARIO  gamma=$GAMMA ==="
        python experiments/barycenter_compare.py \
            --scenario   "$SCENARIO" \
            --gamma      "$GAMMA" \
            --output-dir "$SUBDIR"
    done
done

echo "Study complete → $OUT"
