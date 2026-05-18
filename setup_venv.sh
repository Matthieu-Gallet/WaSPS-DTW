#!/bin/bash
# Set up the virtual environment and build the sdtw Cython extensions.
#
# Usage: bash setup_venv.sh
#
# After running this script, create scripts/env.sh from scripts/env.sh.example
# and fill in the machine-specific paths before running any experiments.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_NAME="venv"

echo "=== Creating virtual environment '$VENV_NAME' ==="
python3 -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"

echo "=== Upgrading pip ==="
pip install --upgrade pip

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== Building sdtw Cython extensions ==="
cd "$PROJECT_ROOT/lib"
python setup.py build_ext --inplace
cd "$PROJECT_ROOT"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Copy scripts/env.sh.example to scripts/env.sh and fill in your paths."
echo "  2. Run experiments via: bash scripts/run_all_experiments.sh"
echo ""
echo "To import from the project in a Python session:"
echo "  source $PROJECT_ROOT/venv/bin/activate"
echo "  export PYTHONPATH=\"$PROJECT_ROOT:$PROJECT_ROOT/lib:\$PYTHONPATH\""
echo ""
echo "To run the sdtw unit tests:"
echo "  cd $PROJECT_ROOT/lib && python -m pytest sdtw/tests/ -v"