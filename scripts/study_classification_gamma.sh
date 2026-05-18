#!/bin/bash
# Sweep gamma × k-fold for classification sensitivity analysis.
# Fold indices are generated once and reused across all gamma values.
# Usage: bash scripts/study_classification_gamma.sh [n_splits=5]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

N_SPLITS="${1:-5}"
GAMMAS="0.001 0.01 0.1 1.0 10.0 100.0 1000.0"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="$RESULTS_DIR/study_classification_gamma_$TIMESTAMP"
mkdir -p "$OUT"
cp "$0" "$OUT/"

# Generate fold indices once
python - <<PYEOF
import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.dataloader import load_classification

X, Y, _ = load_classification("$DATASET_DIR")
skf = StratifiedKFold(n_splits=$N_SPLITS, shuffle=True, random_state=42)
out_dir = "$OUT"
for fold, (train_idx, test_idx) in enumerate(skf.split(X, Y)):
    np.save(out_dir + "/fold_%d_train.npy" % fold, train_idx)
    np.save(out_dir + "/fold_%d_test.npy"  % fold, test_idx)
PYEOF

for GAMMA in $GAMMAS; do
    for FOLD in $(seq 0 $((N_SPLITS - 1))); do
        SUBDIR="$OUT/gamma_${GAMMA}_fold_${FOLD}"
        echo "=== gamma=$GAMMA  fold=$FOLD ==="
        python experiments/classify.py \
            --data-dir        "$DATASET_DIR" \
            --train-indices   "$OUT/fold_${FOLD}_train.npy" \
            --test-indices    "$OUT/fold_${FOLD}_test.npy" \
            --gamma           "$GAMMA" \
            --normalize \
            --normalize-params log_zscore_linear \
            --output-dir      "$SUBDIR"
    done
done

echo "Study complete → $OUT"
