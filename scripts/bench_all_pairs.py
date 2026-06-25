"""Micro-benchmark: vmap(vmap) full matrix vs triangular+symmetrisation for all_pairs.

Compares the cost of computing a symmetric (T, T) distance matrix (as in SDTW
self-terms D_xx / D_yy) using:
  - Full: jax.vmap(lambda xi: jax.vmap(lambda yj: cost(xi, yj))(x))(x)
  - Half: compute upper triangle, then symmetrise

Decision rule (per plan):
  - Judged at T ≤ 52 (real configs), not at large n.
  - If gain is significant at T ≤ 52 → integrate _all_pairs_sym into softdtw.py.
  - Otherwise → document and leave as-is.

Run from repo root:
    python scripts/bench_all_pairs.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
jax.config.update("jax_enable_x64", True)  # enable float64
from costs import WaSPS, SqEuclidean


# ---------------------------------------------------------------------------
# Implementations to compare
# ---------------------------------------------------------------------------

def all_pairs_full(cost_fn, x):
    """Standard vmap(vmap) — computes all T² pairs."""
    return jax.vmap(lambda xi: jax.vmap(lambda yj: cost_fn(xi, yj))(x))(x)


def all_pairs_sym(cost_fn, x):
    """Triangular + symmetrise — computes ~T²/2 pairs then mirrors."""
    n = x.shape[0]
    # Upper triangle (i ≤ j)
    rows, cols = jnp.triu_indices(n)
    xi = x[rows]   # (K, d)
    xj = x[cols]   # (K, d)
    vals = jax.vmap(cost_fn)(xi, xj)   # (K,)
    # Fill symmetric matrix
    D = jnp.zeros((n, n))
    D = D.at[rows, cols].set(vals)
    D = D.at[cols, rows].set(vals)
    return D


# ---------------------------------------------------------------------------
# Timer helper
# ---------------------------------------------------------------------------

def _median_time(fn, n_runs=10, n_warmup=3):
    for _ in range(n_warmup):
        fn().block_until_ready()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn().block_until_ready()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark_size(cost_fn, n, n_params, seed=0):
    rng = np.random.default_rng(seed)
    x = jnp.array(rng.uniform(0.5, 3.0, (n, n_params)), dtype=jnp.float64)

    full_jit = jax.jit(lambda: all_pairs_full(cost_fn, x))
    sym_jit  = jax.jit(lambda: all_pairs_sym(cost_fn, x))

    # Verify equivalence first
    D_full = full_jit()
    D_sym  = sym_jit()
    if not jnp.allclose(D_full, D_sym, atol=1e-6):
        print(f"  WARNING: full vs sym diverge at n={n}  max_diff={float(jnp.max(jnp.abs(D_full - D_sym))):.2e}")

    t_full = _median_time(full_jit)
    t_sym  = _median_time(sym_jit)
    return t_full, t_sym


def main():
    print("=" * 62)
    print("  Benchmark: all_pairs_full vs all_pairs_sym")
    print("  Cost: WaSPS exponential (n_params=1)")
    print("  Decision: judged at T ≤ 52 (real configs)")
    print("=" * 62)
    print(f"{'n':>6} | {'full (ms)':>10} | {'sym  (ms)':>10} | {'speedup':>8}")
    print("-" * 46)

    cost_fn = WaSPS('exponential', log_correction=True)
    sizes = [4, 8, 16, 32, 52, 100, 200, 500, 1000]

    significant_at_real = False
    for n in sizes:
        t_full, t_sym = benchmark_size(cost_fn, n, n_params=1)
        speedup = t_full / t_sym
        marker = " ← real configs" if n <= 52 else ""
        print(f"{n:>6} | {t_full*1000:>10.3f} | {t_sym*1000:>10.3f} | {speedup:>8.2f}x{marker}")
        if n <= 52 and speedup >= 1.3:
            significant_at_real = True

    print("-" * 46)
    print()
    print("CONCLUSION:")
    if significant_at_real:
        print("  Gain SIGNIFICANT at T ≤ 52 → integrate _all_pairs_sym into softdtw.py.")
    else:
        print("  No significant gain at T ≤ 52 (≥ 30% speedup threshold).")
        print("  Leave all_pairs unchanged.  Triangular indexing breaks XLA vectorisation.")
        print("  (Large-n gain at n=1000 is irrelevant — DP is the bottleneck, not cost matrix.)")


if __name__ == "__main__":
    main()
