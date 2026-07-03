#!/bin/bash
# Creates .venv and installs all dependencies via uv.
# No Cython build — the JAX branch uses no compiled extensions.

set -e

VENV_DIR=".venv"

echo "=== Creating virtual environment ==="
uv venv "$VENV_DIR"

echo "=== Installing dependencies ==="
uv pip install --python "$VENV_DIR/bin/python" -r requirements.txt

echo ""
echo "=== Smoke test ==="
"$VENV_DIR/bin/python" -c "
import jax, ott, optax
jax.config.update('jax_enable_x64', True)
print('jax    ', jax.__version__)
print('ott    ', ott.__version__)
print('optax  ', optax.__version__)
print('devices', jax.devices())
print('OK')
"

echo ""
echo "=== Installing Jupyter kernel ==="
"$VENV_DIR/bin/python" -m ipykernel install --user --name wasps-dtw --display-name "WaSPS-DTW (.venv)" 2>/dev/null || echo "  (kernel already installed)"

echo ""
echo "=== Done ==="
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Run tests with: pytest tests/ -q"
echo "Run notebooks: jupyter notebook analysis/"
echo ""
echo "Note: JAX runs on CPU by default. The 'CUDA-enabled jaxlib not installed' warning is informational."
echo "      If you have an NVIDIA GPU, install cuda-enabled jaxlib separately (requires CUDA toolkit)."
