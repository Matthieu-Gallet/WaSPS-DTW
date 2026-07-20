#!/usr/bin/env bash
# River barycenter illustration — river only, 2 classes (NG, PN), 4 methods (wasps,
# eucl_params, eucl_raw, sta) overlaid on one axis, one PDF per class. Standalone: does
# not depend on gamma calibration or any other experiment's output (gamma=10 is fixed
# directly, per spec — not calibrated).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source .venv/bin/activate

CONFIG="configs/config_baseline.yaml"
OUT_DIR="results/jax_exp4_river_bary_viz"
LOG="$OUT_DIR/exp4_river_bary_viz.sh.log"
mkdir -p "$OUT_DIR"

echo "===== $(date '+%Y-%m-%d %H:%M:%S') exp4_river_bary_viz starting =====" | tee -a "$LOG"

python src/plot/plot_river_bary_viz.py --config "$CONFIG" --output-dir "$OUT_DIR" \
    --classes NG,PN --gamma 10 \
    --n-samples-per-class 75 --sta-n-samples-per-class 25 \
    --samples-per-step 480 --sta-samples-per-step 48 \
    --n-steps-bary 150 --seed 45 \
    2>&1 | tee -a "$LOG"

echo "===== $(date '+%Y-%m-%d %H:%M:%S') exp4_river_bary_viz done =====" | tee -a "$LOG"
