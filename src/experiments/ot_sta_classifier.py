"""
Regularized empirical OT + Soft-DTW helpers (STA-style) for raw time series.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from sdtw import SoftDTW


def _ground_metric(width: int, p: int = 2, normed: bool = True) -> np.ndarray:
    """Ground metric on a 1D grid, adapted from sta.utils.groundmetric."""
    x = np.arange(width, dtype=np.float64).reshape(-1, 1)
    m = np.abs(x - x.T) ** p
    if normed:
        med = float(np.median(m[m > 0])) if np.any(m > 0) else 1.0
        m = m / max(med, 1e-12)
    return m


def _to_hist(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    x = np.where(np.isfinite(x), x, 0.0)
    x = np.maximum(x, 0.0) + eps
    return x / max(x.sum(), eps)


def _compress_features(series: np.ndarray, n_bins: Optional[int]) -> np.ndarray:
    if n_bins is None or n_bins <= 0:
        return series
    t, f = series.shape
    if f <= n_bins:
        return series
    edges = np.linspace(0, f, n_bins + 1, dtype=int)
    out = np.empty((t, n_bins), dtype=np.float64)
    for b in range(n_bins):
        left = edges[b]
        right = max(edges[b + 1], left + 1)
        out[:, b] = series[:, left:right].sum(axis=1)
    return out


def _subsample_time(series: np.ndarray, stride: int) -> np.ndarray:
    s = max(int(stride), 1)
    return series[::s]


def _sinkhorn_distance(
    a: np.ndarray,
    b: np.ndarray,
    metric: np.ndarray,
    epsilon: float = 0.05,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> float:
    """Entropic OT cost between two discrete histograms."""
    reg = max(float(epsilon), 1e-8)
    k = np.exp(-metric / reg)
    k = np.maximum(k, 1e-300)

    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(max_iter):
        u_prev = u
        kv = k @ v + 1e-300
        u = a / kv
        ktu = k.T @ u + 1e-300
        v = b / ktu
        if np.linalg.norm(u - u_prev, ord=1) < tol:
            break

    transport = (u[:, None] * k) * v[None, :]
    return float(np.sum(transport * metric))


def _sinkhorn_barycenter(
    hists: np.ndarray,
    metric: np.ndarray,
    epsilon: float = 0.05,
    max_iter: int = 60,
    tol: float = 1e-6,
) -> np.ndarray:
    """Equal-weights entropic OT barycenter for histograms."""
    n_hists, n_bins = hists.shape
    reg = max(float(epsilon), 1e-8)
    k = np.exp(-metric / reg)
    k = np.maximum(k, 1e-300)

    q = np.mean(hists, axis=0)
    q = q / max(q.sum(), 1e-12)
    v = np.ones((n_hists, n_bins), dtype=np.float64)

    for _ in range(max_iter):
        q_prev = q
        kv = np.einsum("ij,nj->ni", k, v) + 1e-300
        u = hists / kv
        ktu = np.einsum("ij,ni->nj", k.T, u) + 1e-300
        q = np.exp(np.mean(np.log(ktu), axis=0))
        q = q / max(q.sum(), 1e-12)
        v = q[None, :] / ktu
        if np.linalg.norm(q - q_prev, ord=1) < tol:
            break
    return q


def compute_ot_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float = 0.05,
    max_iter: int = 100,
    tol: float = 1e-6,
    feature_bins: Optional[int] = 32,
    time_stride: int = 4,
    metric: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build STA local cost matrix with entropic OT between every pair of timestamps.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D arrays with shape (T, F)")
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have same feature dimension")

    x = _subsample_time(x, time_stride)
    y = _subsample_time(y, time_stride)
    x = _compress_features(x, feature_bins)
    y = _compress_features(y, feature_bins)

    m = _ground_metric(x.shape[1]) if metric is None else metric
    hx = np.stack([_to_hist(row) for row in x], axis=0)
    hy = np.stack([_to_hist(row) for row in y], axis=0)

    cost = np.empty((x.shape[0], y.shape[0]), dtype=np.float64)
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            cost[i, j] = _sinkhorn_distance(
                hx[i], hy[j], m, epsilon=epsilon, max_iter=max_iter, tol=tol
            )
    return cost


def compute_sdtw_distance_ot_regul(
    sample: np.ndarray,
    barycenter: np.ndarray,
    gamma: float = 1.0,
    ot_epsilon: float = 0.05,
    ot_max_iter: int = 100,
    ot_tol: float = 1e-6,
    ot_feature_bins: Optional[int] = 32,
    ot_time_stride: int = 4,
) -> float:
    """Soft-DTW distance using entropic OT local costs (STA-style)."""
    cost = compute_ot_cost_matrix(
        sample,
        barycenter,
        epsilon=ot_epsilon,
        max_iter=ot_max_iter,
        tol=ot_tol,
        feature_bins=ot_feature_bins,
        time_stride=ot_time_stride,
    )
    return float(SoftDTW(cost, gamma=gamma).compute())


def compute_barycenter_ot_regul_raw(
    samples: List[np.ndarray],
    ot_epsilon: float = 0.05,
    ot_barycenter_iters: int = 60,
    ot_tol: float = 1e-6,
    ot_feature_bins: Optional[int] = 32,
    ot_time_stride: int = 1,
) -> np.ndarray:
    """
    Compute an OT-regularized barycenter time series (one histogram per timestamp).
    """
    if len(samples) == 0:
        raise ValueError("samples must not be empty")
    t, f = samples[0].shape
    for s in samples:
        if s.shape != (t, f):
            raise ValueError("all samples must share the same shape (T, F)")

    proc_samples = [_compress_features(_subsample_time(s, ot_time_stride), ot_feature_bins) for s in samples]
    t_proc, f_proc = proc_samples[0].shape

    metric = _ground_metric(f_proc)
    bary_proc = np.zeros((t_proc, f_proc), dtype=np.float64)

    for ti in range(t_proc):
        hists = np.stack([_to_hist(s[ti]) for s in proc_samples], axis=0)
        q = _sinkhorn_barycenter(
            hists,
            metric,
            epsilon=ot_epsilon,
            max_iter=ot_barycenter_iters,
            tol=ot_tol,
        )
        mean_mass = float(np.mean([np.maximum(s[ti], 0.0).sum() for s in proc_samples]))
        bary_proc[ti] = q * max(mean_mass, 1e-8)

    # Re-expand time axis if subsampled.
    if t_proc != t:
        src_t = np.arange(t_proc, dtype=np.float64)
        dst_t = np.linspace(0.0, max(t_proc - 1, 0), num=t)
        bary_time = np.vstack([
            np.interp(dst_t, src_t, bary_proc[:, j]) for j in range(f_proc)
        ]).T
    else:
        bary_time = bary_proc

    # Re-expand feature axis if compressed.
    if f_proc != f:
        bary = np.zeros((t, f), dtype=np.float64)
        edges = np.linspace(0, f, f_proc + 1, dtype=int)
        for b in range(f_proc):
            left = edges[b]
            right = max(edges[b + 1], left + 1)
            width = float(right - left)
            bary[:, left:right] = bary_time[:, [b]] / width
    else:
        bary = bary_time
    return bary
