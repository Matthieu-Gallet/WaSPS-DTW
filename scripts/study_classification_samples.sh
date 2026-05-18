#!/bin/bash
# Sweep training sample fraction × k-fold at a fixed gamma.
# Usage: bash scripts/study_classification_samples.sh [gamma=1.0] [n_splits=5]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

GAMMA="${1:-1.0}"
N_SPLITS="${2:-5}"
FRACTIONS="0.05 0.1 0.2 0.4 0.6 0.8 1.0"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="$RESULTS_DIR/study_classification_samples_gamma${GAMMA}_$TIMESTAMP"
mkdir -p "$OUT"
cp "$0" "$OUT/"

# Generate fold indices once, then subsample training sets per fraction
python - <<PYEOF
import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.dataloader import load_classification

X, Y, _ = load_classification("$DATASET_DIR")
skf = StratifiedKFold(n_splits=$N_SPLITS, shuffle=True, random_state=42)
out_dir = "$OUT"
for fold, (train_idx, test_idx) in enumerate(skf.split(X, Y)):
    np.save(out_dir + "/fold_%d_train_full.npy" % fold, train_idx)
    np.save(out_dir + "/fold_%d_test.npy" % fold,       test_idx)

    for frac_str in "$FRACTIONS".split():
        frac = float(frac_str)
        if frac >= 1.0:
            sub_idx = train_idx
        else:
            rng = np.random.RandomState(42)
            sub = []
            for label in np.unique(Y[train_idx]):
                cls = train_idx[Y[train_idx] == label]
                n = max(1, int(len(cls) * frac))
                rng.shuffle(cls)
                sub.extend(cls[:n])
            sub_idx = np.array(sub)
        np.save(out_dir + "/fold_%d_frac_%s_train.npy" % (fold, frac_str), sub_idx)
PYEOF

for FRAC in $FRACTIONS; do
    for FOLD in $(seq 0 $((N_SPLITS - 1))); do
        SUBDIR="$OUT/frac_${FRAC}_fold_${FOLD}"
        echo "=== frac=$FRAC  fold=$FOLD  gamma=$GAMMA ==="
        python experiments/classify.py \
            --data-dir        "$DATASET_DIR" \
            --train-indices   "$OUT/fold_${FOLD}_frac_${FRAC}_train.npy" \
            --test-indices    "$OUT/fold_${FOLD}_test.npy" \
            --gamma           "$GAMMA" \
            --normalize \
            --normalize-params log_zscore_linear \
            --output-dir      "$SUBDIR"
    done
done

echo "Study complete → $OUT"
