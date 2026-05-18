#!/bin/bash
# Prepare Format A classification dataset from the river discharge NetCDF.
# Usage: bash scripts/prepare_discharge_classification.sh [basic|balanced]
#
# Reads: $DATA_NC_PATH, $DATA_CSV_PATH  (from env.sh)
# Writes: $DATASET_DIR/<mode>/

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

MODE="${1:-balanced}"
OUT="$DATASET_DIR/$MODE"

if [ "$MODE" = "balanced" ]; then
    python src/dataloader/discharge.py prepare-classification \
        --nc-path     "$DATA_NC_PATH" \
        --csv-path    "$DATA_CSV_PATH" \
        --output-dir  "$OUT" \
        --samples-per-class '{"NG": 300, "NP": 300, "PC": 300, "PM": 300, "PN": 300}'
else
    python src/dataloader/discharge.py prepare-classification \
        --nc-path    "$DATA_NC_PATH" \
        --csv-path   "$DATA_CSV_PATH" \
        --output-dir "$OUT"
fi

echo "Dataset written to: $OUT"
