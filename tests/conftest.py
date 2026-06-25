"""pytest configuration: enable JAX x64 once for the entire test suite.

Individual tests that require float64 precision pass dtype=jnp.float64 explicitly
(the default computation dtype is float32).
"""

import sys
from pathlib import Path

import jax

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

jax.config.update("jax_enable_x64", True)  # must precede any jax.numpy usage
