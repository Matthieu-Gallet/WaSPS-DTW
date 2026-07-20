"""STA (Spatio-Temporal Alignment) baseline: SoftDTW over OT-Sinkhorn local costs.

Replaces the vendored hichamjanati PyTorch STA (RESSOURCES/spatio-temporal-alignements/)
using our JAX stack: OTT-JAX Sinkhorn for the per-timestep OT cost, and our SoftDTW
for the temporal alignment.  No torch/pyts dependency.

Input format: raw sample arrays of shape (T, N_samples) — T timesteps, each with
N_samples observed values.  Each row is cleaned with clean_series() before OT.

Reference: Janati et al., "Spatio-Temporal Alignments: Optimal Transport through
Space and Time", AISTATS 2020.
"""

from __future__ import annotations

import functools

import numpy as np
import jax
import jax.numpy as jnp

from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

from softdtw import sdtw_value


def _valid_samples(x: np.ndarray) -> np.ndarray:
    """Keep finite, positive samples.  Preserves absolute scale (unlike clean_series
    which shifts by min — that would erase the inter-class scale difference needed
    for OT to discriminate distributions at different locations).
    """
    x = np.asarray(x, dtype=np.float32).ravel()
    x = x[np.isfinite(x) & (x > 0.0)]
    return x if len(x) > 0 else np.array([1e-8])


# ---------------------------------------------------------------------------
# OT cost between two timestep sample sets
# ---------------------------------------------------------------------------

# JIT-compiled Sinkhorn call.  Shapes must be consistent within a call to
# knn_predict; epsilon is a traced value so different callers can use their own.
# Using reduced max_iterations (5) vs OTT default (2000) for CPU tractability
# (was 500 -> 10 on 2026-07-08; -> 5 on 2026-07-10 after run_full_baseline.py's
# cpazmal STA batch OOM'd at n_jobs=4 despite the pool-teardown-between-methods
# mitigation — cutting Sinkhorn iterations per call further reduces the compute
# (and apparent RSS growth) per worker over a batch's ~thousands of dispatches).
@jax.jit
def _sinkhorn_jit(x_pts, y_pts, epsilon):
    geom = pointcloud.PointCloud(x_pts, y_pts, epsilon=epsilon)
    return sinkhorn.Sinkhorn(max_iterations=5)(linear_problem.LinearProblem(geom)).reg_ot_cost


def _ot_cost(x_samples: np.ndarray, y_samples: np.ndarray, epsilon: float) -> float:
    """Regularized OT cost between two 1-D empirical distributions."""
    x_pts = jnp.array(x_samples.reshape(-1, 1), dtype=jnp.float32)
    y_pts = jnp.array(y_samples.reshape(-1, 1), dtype=jnp.float32)
    return float(_sinkhorn_jit(x_pts, y_pts, float(epsilon)))


def make_cost_fn(epsilon: float):
    """Return a JAX-compatible OT cost function for use with SoftDTW / fit_barycenter.

    Unlike _ot_cost (which returns a Python float), the returned callable is
    pure JAX — it is vmap- and grad-safe — enabling STA barycenter estimation
    via autodiff through Sinkhorn.

    Args:
        epsilon: Sinkhorn regularisation.

    Returns:
        cost_fn: (a, b) → scalar, where a, b are (N,) JAX float32 arrays.
    """
    def cost_fn(a: jax.Array, b: jax.Array) -> jax.Array:
        geom = pointcloud.PointCloud(a.reshape(-1, 1), b.reshape(-1, 1), epsilon=epsilon)
        return sinkhorn.Sinkhorn(max_iterations=5)(
            linear_problem.LinearProblem(geom)
        ).reg_ot_cost
    return cost_fn


# ---------------------------------------------------------------------------
# STA cost matrix and distance
# ---------------------------------------------------------------------------

def sta_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float = 0.05,
) -> np.ndarray:
    """Pairwise OT cost matrix between timesteps of two raw sample series.

    Args:
        x:       (T_x, N_x) raw sample array — rows = timesteps, cols = samples.
        y:       (T_y, N_y) raw sample array.
        epsilon: Sinkhorn regularisation.

    Returns:
        D: (T_x, T_y) cost matrix, D[i, j] = OT(x[i], y[j]).
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    T_x, T_y = x.shape[0], y.shape[0]
    D = np.empty((T_x, T_y), dtype=np.float32)
    for i in range(T_x):
        xi = _valid_samples(x[i])
        for j in range(T_y):
            yj = _valid_samples(y[j])
            D[i, j] = _ot_cost(xi, yj, epsilon)
    return D


@functools.partial(jax.jit, static_argnames=('epsilon',))
def _sta_cost_matrix_jit(x_arr: jax.Array, y_arr: jax.Array, epsilon: float) -> jax.Array:
    """(T_x, T_y) cost matrix via double vmap — 1 XLA kernel instead of T_x×T_y dispatches.

    x_arr: (T_x, N)  y_arr: (T_y, N)  — all values finite, positive (after to_fixed_n).
    """
    def row_costs(xi):
        return jax.vmap(
            lambda yj: sinkhorn.Sinkhorn(max_iterations=5)(
                linear_problem.LinearProblem(
                    pointcloud.PointCloud(xi.reshape(-1, 1), yj.reshape(-1, 1), epsilon=epsilon)
                )
            ).reg_ot_cost
        )(y_arr)
    return jax.vmap(row_costs)(x_arr)


def sta_cost_matrix_rect(
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float = 0.05,
) -> np.ndarray:
    """Vectorized cost matrix for rectangularised series (after to_fixed_n).

    Uses double jax.vmap for a single XLA kernel per test-train pair, replacing
    the T_x×T_y Python dispatch loop in sta_cost_matrix.  Requires x, y to be
    rectangular (no NaN) — falls back to sta_cost_matrix for ragged inputs.

    Args:
        x:       (T_x, N) float32 array — all values finite, positive.
        y:       (T_y, N) float32 array.
        epsilon: Sinkhorn regularisation.

    Returns:
        D: (T_x, T_y) float32 cost matrix.
    """
    x_jax = jnp.asarray(x, dtype=jnp.float32)
    y_jax = jnp.asarray(y, dtype=jnp.float32)
    return np.asarray(_sta_cost_matrix_jit(x_jax, y_jax, epsilon))


def sta_distance(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0,
    epsilon: float = 0.05,
) -> float:
    """STA distance: SoftDTW over Sinkhorn-OT local costs.

    Args:
        x:       (T, N_samples) raw series.
        y:       (T, N_samples) raw series.
        gamma:   SoftDTW smoothing.
        epsilon: Sinkhorn regularisation for the OT ground cost.

    Returns:
        Scalar STA distance.
    """
    D = sta_cost_matrix(x, y, epsilon)
    return float(sdtw_value(jnp.array(D), gamma))


# ---------------------------------------------------------------------------
# k-NN classifier
# ---------------------------------------------------------------------------

def knn_predict(
    train_series: list,
    train_labels: np.ndarray,
    test_series: list,
    gamma: float = 1.0,
    epsilon: float = 0.05,
    k: int = 1,
) -> np.ndarray:
    """k-NN classifier using STA distances.

    Args:
        train_series:  List of N_train (T, N_samples) raw arrays.
        train_labels:  Integer labels shape (N_train,).
        test_series:   List of N_test (T, N_samples) raw arrays.
        gamma:         SoftDTW smoothing.
        epsilon:       Sinkhorn regularisation.
        k:             Number of neighbours.

    Returns:
        predictions: (N_test,) integer array.
    """
    labels = np.asarray(train_labels)

    # Use fast double-vmap path when all series are rectangular (no NaN —
    # guaranteed after to_fixed_n; reverts to loop for ragged inputs).
    def _is_rect(s: np.ndarray) -> bool:
        a = np.asarray(s, dtype=np.float32)
        return bool(np.all(np.isfinite(a) & (a > 0)))

    preds = []
    for test in test_series:
        test_arr = np.asarray(test, dtype=np.float32)
        dists = []
        for tr in train_series:
            tr_arr = np.asarray(tr, dtype=np.float32)
            if _is_rect(test_arr) and _is_rect(tr_arr):
                D = sta_cost_matrix_rect(test_arr, tr_arr, epsilon)
            else:
                D = sta_cost_matrix(test_arr, tr_arr, epsilon)
            dists.append(float(sdtw_value(jnp.array(D), gamma)))
        dists = np.array(dists)
        nn_idx = np.argsort(dists)[:k]
        nn_labels = labels[nn_idx]
        counts = np.bincount(nn_labels, minlength=int(labels.max()) + 1)
        preds.append(int(np.argmax(counts)))
    return np.array(preds)
