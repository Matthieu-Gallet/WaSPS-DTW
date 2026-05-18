#!/bin/bash
# Geodesic interpolation between every class pair, for all three methods.
# Reads barycenters from a completed classify.py results.zarr.
#
# Usage: bash scripts/study_interpolation.sh <kfold_subdir> [gamma=1.0] [k_steps=7]
#
# Example:
#   bash scripts/study_interpolation.sh \
#       results/study_classification_kfold_gamma1.0_20250101_120000/fold_0
#
# Reads:  <kfold_subdir>/results.zarr          (output of classify.py)
# Writes: $RESULTS_DIR/study_interpolation_TIMESTAMP/

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

KFOLD_DIR="${1:?Usage: $0 <kfold_subdir> [gamma=1.0] [k_steps=7]}"
GAMMA="${2:-1.0}"
K_STEPS="${3:-7}"

ZARR_PATH="$KFOLD_DIR/results.zarr"
if [ ! -d "$ZARR_PATH" ]; then
    echo "ERROR: $ZARR_PATH not found" >&2
    exit 1
fi

# Discover class labels from the zarr metadata
LABELS=$(python - <<PYEOF
import zarr, sys
store = zarr.open("$ZARR_PATH", mode='r')
# Labels are stored as barycenter_<label> datasets under any method group
import re
grp = store[list(store.group_keys())[0]]
labels = sorted(int(m.group(1))
                for k in grp.array_keys()
                if (m := re.match(r'barycenter_(\d+)', k)))
print(' '.join(map(str, labels)))
PYEOF
)

METHODS="euclidean_raw euclidean_params wasserstein_sgd"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="$RESULTS_DIR/study_interpolation_gamma${GAMMA}_$TIMESTAMP"
mkdir -p "$OUT"
cp "$0" "$OUT/"

for METHOD in $METHODS; do
    # All unordered class pairs
    labels_arr=($LABELS)
    n=${#labels_arr[@]}
    for (( i=0; i<n; i++ )); do
        for (( j=i+1; j<n; j++ )); do
            LA=${labels_arr[$i]}
            LB=${labels_arr[$j]}
            SUBDIR="$OUT/${METHOD}_${LA}_vs_${LB}"
            echo "=== method=$METHOD  $LA → $LB ==="
            python experiments/interpolate.py \
                --zarr-path   "$ZARR_PATH" \
                --label-a     "$LA" \
                --label-b     "$LB" \
                --method      "$METHOD" \
                --gamma       "$GAMMA" \
                --k-steps     "$K_STEPS" \
                --output-dir  "$SUBDIR"
        done
    done
done

echo "Study complete → $OUT"
