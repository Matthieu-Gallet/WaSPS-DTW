#!/bin/bash
# Prepare Format B groups dataset from the river discharge NetCDF.
# Usage: bash scripts/prepare_discharge_groups.sh
#
# Reads: $DATA_NC_PATH  (from env.sh)
# Writes: $DATASET_DIR/discharge_groups/

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

OUT="$DATASET_DIR/discharge_groups"

python src/dataloader/discharge.py prepare-groups \
    --nc-path    "$DATA_NC_PATH" \
    --output-dir "$OUT" \
    --n-groups   10 \
    --n-series-per-group 8

echo "Groups written to: $OUT"
