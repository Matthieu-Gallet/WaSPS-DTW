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
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x) & (x > 0.0)]
    return x if len(x) > 0 else np.array([1e-8])


# ---------------------------------------------------------------------------
# OT cost between two timestep sample sets
# ---------------------------------------------------------------------------

# JIT-compiled Sinkhorn call.  Shapes must be consistent within a call to
# knn_predict; epsilon is a traced value so different callers can use their own.
# Using reduced max_iterations (500) vs OTT default (2000) for CPU tractability.
@jax.jit
def _sinkhorn_jit(x_pts, y_pts, epsilon):
    geom = pointcloud.PointCloud(x_pts, y_pts, epsilon=epsilon)
    return sinkhorn.Sinkhorn(max_iterations=500)(linear_problem.LinearProblem(geom)).reg_ot_cost


def _ot_cost(x_samples: np.ndarray, y_samples: np.ndarray, epsilon: float) -> float:
    """Regularized OT cost between two 1-D empirical distributions."""
    x_pts = jnp.array(x_samples.reshape(-1, 1), dtype=jnp.float64)
    y_pts = jnp.array(y_samples.reshape(-1, 1), dtype=jnp.float64)
    return float(_sinkhorn_jit(x_pts, y_pts, float(epsilon)))


def make_cost_fn(epsilon: float):
    """Return a JAX-compatible OT cost function for use with SoftDTW / fit_barycenter.

    Unlike _ot_cost (which returns a Python float), the returned callable is
    pure JAX — it is vmap- and grad-safe — enabling STA barycenter estimation
    via autodiff through Sinkhorn.

    Args:
        epsilon: Sinkhorn regularisation.

    Returns:
        cost_fn: (a, b) → scalar, where a, b are (N,) JAX float64 arrays.
    """
    def cost_fn(a: jax.Array, b: jax.Array) -> jax.Array:
        geom = pointcloud.PointCloud(a.reshape(-1, 1), b.reshape(-1, 1), epsilon=epsilon)
        return sinkhorn.Sinkhorn(max_iterations=500)(
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
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    T_x, T_y = x.shape[0], y.shape[0]
    D = np.empty((T_x, T_y), dtype=np.float64)
    for i in range(T_x):
        xi = _valid_samples(x[i])
        for j in range(T_y):
            yj = _valid_samples(y[j])
            D[i, j] = _ot_cost(xi, yj, epsilon)
    return D


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
    preds = []
    for test in test_series:
        dists = np.array([
            sta_distance(test, tr, gamma=gamma, epsilon=epsilon)
            for tr in train_series
        ])
        nn_idx = np.argsort(dists)[:k]
        nn_labels = labels[nn_idx]
        counts = np.bincount(nn_labels, minlength=int(labels.max()) + 1)
        preds.append(int(np.argmax(counts)))
    return np.array(preds)
