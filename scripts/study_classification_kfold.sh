#!/bin/bash
# k-fold cross-validation at a fixed gamma.
# Generates fold index arrays once, then calls classify.py once per fold.
# Usage: bash scripts/study_classification_kfold.sh [gamma=1.0] [n_splits=5]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

GAMMA="${1:-1.0}"
N_SPLITS="${2:-5}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="$RESULTS_DIR/study_classification_kfold_gamma${GAMMA}_$TIMESTAMP"
mkdir -p "$OUT"
cp "$0" "$OUT/"

# Generate stratified fold indices
python - <<PYEOF
import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.dataloader import load_classification

X, Y, _ = load_classification("$DATASET_DIR")
skf = StratifiedKFold(n_splits=$N_SPLITS, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(skf.split(X, Y)):
    np.save("$OUT/fold_${fold}_train.npy", train_idx)
    np.save("$OUT/fold_${fold}_test.npy",  test_idx)
    print(f"Fold {fold}: train={len(train_idx)}  test={len(test_idx)}")
PYEOF

for FOLD in $(seq 0 $((N_SPLITS - 1))); do
    SUBDIR="$OUT/fold_${FOLD}"
    echo "=== fold=$FOLD  gamma=$GAMMA ==="
    python experiments/classify.py \
        --data-dir        "$DATASET_DIR" \
        --train-indices   "$OUT/fold_${FOLD}_train.npy" \
        --test-indices    "$OUT/fold_${FOLD}_test.npy" \
        --gamma           "$GAMMA" \
        --normalize \
        --normalize-params log_zscore_linear \
        --output-dir      "$SUBDIR"
done

echo "Study complete → $OUT"
