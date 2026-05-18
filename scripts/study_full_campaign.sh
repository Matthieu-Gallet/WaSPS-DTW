#!/bin/bash
# Full experimental campaign — runs all studies in sequence and generates the report.
#
# Usage: bash scripts/study_full_campaign.sh [gamma=1.0] [n_splits=5]
#
# Steps (in order):
#   1. Prepare Format A (balanced) and Format B datasets
#   2. k-fold classification at default gamma
#   3. Gamma sensitivity sweep (k-fold)
#   4. Training sample size sweep (k-fold)
#   5. Barycenter RMSE grid (synthetic)
#   6. Barycenter method comparison on groups
#   7. Geodesic interpolation between every class pair (from step 2)
#   8. Report generation

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

GAMMA="${1:-1.0}"
N_SPLITS="${2:-5}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CAMPAIGN_DIR="$RESULTS_DIR/campaign_$TIMESTAMP"
mkdir -p "$CAMPAIGN_DIR"
cp "$0" "$CAMPAIGN_DIR/"
echo "Campaign root: $CAMPAIGN_DIR"

# ---------------------------------------------------------------------------
# Step 1 — Dataset preparation
# ---------------------------------------------------------------------------
echo ""
echo "=== [1/8] Preparing datasets ==="
bash "$SCRIPT_DIR/prepare_discharge_classification.sh" balanced
bash "$SCRIPT_DIR/prepare_discharge_groups.sh"

# ---------------------------------------------------------------------------
# Step 2 — k-fold at default gamma  (needed for barycenters + interpolation)
# ---------------------------------------------------------------------------
echo ""
echo "=== [2/8] k-fold classification (gamma=$GAMMA, splits=$N_SPLITS) ==="
KFOLD_OUT="$CAMPAIGN_DIR/kfold_gamma${GAMMA}"
mkdir -p "$KFOLD_OUT"

python - <<PYEOF
import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.dataloader import load_classification

X, Y, _ = load_classification("$DATASET_DIR")
skf = StratifiedKFold(n_splits=$N_SPLITS, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(skf.split(X, Y)):
    np.save("$KFOLD_OUT/fold_${fold}_train.npy", train_idx)
    np.save("$KFOLD_OUT/fold_${fold}_test.npy",  test_idx)
    print(f"Fold {fold}: train={len(train_idx)}  test={len(test_idx)}")
PYEOF

for FOLD in $(seq 0 $((N_SPLITS - 1))); do
    python experiments/classify.py \
        --data-dir        "$DATASET_DIR" \
        --train-indices   "$KFOLD_OUT/fold_${FOLD}_train.npy" \
        --test-indices    "$KFOLD_OUT/fold_${FOLD}_test.npy" \
        --gamma           "$GAMMA" \
        --normalize \
        --normalize-params log_zscore_linear \
        --output-dir      "$KFOLD_OUT/fold_${FOLD}"
done

# ---------------------------------------------------------------------------
# Step 3 — Gamma sensitivity
# ---------------------------------------------------------------------------
echo ""
echo "=== [3/8] Gamma sensitivity sweep ==="
GAMMA_OUT="$CAMPAIGN_DIR/gamma_sweep"
mkdir -p "$GAMMA_OUT"
GAMMAS="0.001 0.01 0.1 1.0 10.0 100.0 1000.0"

python - <<PYEOF
import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.dataloader import load_classification

X, Y, _ = load_classification("$DATASET_DIR")
skf = StratifiedKFold(n_splits=$N_SPLITS, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(skf.split(X, Y)):
    np.save("$GAMMA_OUT/fold_${fold}_train.npy", train_idx)
    np.save("$GAMMA_OUT/fold_${fold}_test.npy",  test_idx)
PYEOF

for G in $GAMMAS; do
    for FOLD in $(seq 0 $((N_SPLITS - 1))); do
        python experiments/classify.py \
            --data-dir        "$DATASET_DIR" \
            --train-indices   "$GAMMA_OUT/fold_${FOLD}_train.npy" \
            --test-indices    "$GAMMA_OUT/fold_${FOLD}_test.npy" \
            --gamma           "$G" \
            --normalize \
            --normalize-params log_zscore_linear \
            --output-dir      "$GAMMA_OUT/gamma_${G}_fold_${FOLD}"
    done
done

# ---------------------------------------------------------------------------
# Step 4 — Sample size sensitivity
# ---------------------------------------------------------------------------
echo ""
echo "=== [4/8] Sample size sensitivity ==="
SAMPLES_OUT="$CAMPAIGN_DIR/samples_sweep"
mkdir -p "$SAMPLES_OUT"
FRACTIONS="0.05 0.1 0.2 0.4 0.6 0.8 1.0"

python - <<PYEOF
import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.dataloader import load_classification

X, Y, _ = load_classification("$DATASET_DIR")
skf = StratifiedKFold(n_splits=$N_SPLITS, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(skf.split(X, Y)):
    np.save("$SAMPLES_OUT/fold_${fold}_train_full.npy", train_idx)
    np.save("$SAMPLES_OUT/fold_${fold}_test.npy",       test_idx)
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
        np.save(f"$SAMPLES_OUT/fold_{fold}_frac_{frac_str}_train.npy", sub_idx)
PYEOF

for FRAC in $FRACTIONS; do
    for FOLD in $(seq 0 $((N_SPLITS - 1))); do
        python experiments/classify.py \
            --data-dir        "$DATASET_DIR" \
            --train-indices   "$SAMPLES_OUT/fold_${FOLD}_frac_${FRAC}_train.npy" \
            --test-indices    "$SAMPLES_OUT/fold_${FOLD}_test.npy" \
            --gamma           "$GAMMA" \
            --normalize \
            --normalize-params log_zscore_linear \
            --output-dir      "$SAMPLES_OUT/frac_${FRAC}_fold_${FOLD}"
    done
done

# ---------------------------------------------------------------------------
# Step 5 — Barycenter RMSE grid
# ---------------------------------------------------------------------------
echo ""
echo "=== [5/8] Barycenter RMSE grid ==="
RMSE_OUT="$CAMPAIGN_DIR/rmse_grid"
N_SAMPLES_LIST="50 100 500 1000 5000"
ESTIMATORS="mle log_cumulant"
RMSE_GAMMAS="0.1 1.0 10.0"

for N in $N_SAMPLES_LIST; do
    for EST in $ESTIMATORS; do
        for G in $RMSE_GAMMAS; do
            python experiments/barycenter_rmse.py \
                --n-samples  "$N" \
                --estimator  "$EST" \
                --gamma      "$G" \
                --output-dir "$RMSE_OUT/n${N}_est_${EST}_gamma_${G}"
        done
    done
done

# ---------------------------------------------------------------------------
# Step 6 — Groups barycenter
# ---------------------------------------------------------------------------
echo ""
echo "=== [6/8] Groups barycenter (gamma=$GAMMA) ==="
GROUPS_DIR="$DATASET_DIR/../discharge_groups"
python experiments/barycenter_groups.py \
    --groups-dir "$GROUPS_DIR" \
    --gamma      "$GAMMA" \
    --normalize \
    --output-dir "$CAMPAIGN_DIR/groups_gamma${GAMMA}"

# ---------------------------------------------------------------------------
# Step 7 — Geodesic interpolation (from fold 0 of kfold study)
# ---------------------------------------------------------------------------
echo ""
echo "=== [7/8] Geodesic interpolation ==="
bash "$SCRIPT_DIR/study_interpolation.sh" \
    "$KFOLD_OUT/fold_0" \
    "$GAMMA"

# ---------------------------------------------------------------------------
# Step 8 — Report
# ---------------------------------------------------------------------------
echo ""
echo "=== [8/8] Generating report ==="
REPORT_DIR="$CAMPAIGN_DIR/report"
INTERP_DIR=$(ls -dt "$RESULTS_DIR"/study_interpolation_* 2>/dev/null | head -1 || true)

python analysis/report.py \
    --results-dir   "$CAMPAIGN_DIR" \
    --data-dir      "$DATASET_DIR" \
    --output-dir    "$REPORT_DIR" \
    --kfold-study   "$KFOLD_OUT" \
    --gamma-study   "$GAMMA_OUT" \
    --samples-study "$SAMPLES_OUT" \
    --rmse-study    "$RMSE_OUT" \
    ${INTERP_DIR:+--interp-study "$INTERP_DIR"}

echo ""
echo "Campaign complete → $CAMPAIGN_DIR"
echo "Report          → $REPORT_DIR"
