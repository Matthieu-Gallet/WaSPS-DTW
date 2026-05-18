"""
Pure preprocessing transforms for time series datasets.

All functions: arrays in, arrays/lists out. No I/O.
"""

from typing import List, Literal, Optional

import numpy as np

from src.estimator import LogCumulant, MLE


def preprocess_samples(X: np.ndarray,
                       max_time_steps: Optional[int] = None,
                       normalize: bool = False) -> List[np.ndarray]:
    """
    Convert a stacked array to a list of per-sample time series.

    Args:
        X:             float64 array (N, T, D) — already-flattened feature dimension.
        max_time_steps: If set, truncate each series to this length.
        normalize:     If True, z-score normalize each sample globally (mean=0, std=1).
                       Removes amplitude variation so DTW captures temporal pattern only.

    Returns:
        List of N arrays, each (T', D) where T' = min(T, max_time_steps).
    """
    N, T, D = X.shape
    T = min(T, max_time_steps) if max_time_steps else T
    samples = []
    for i in range(N):
        s = X[i, :T].astype(np.float64)
        s = np.nan_to_num(s, nan=0.0)
        if normalize:
            mu, sig = s.mean(), s.std()
            s = (s - mu) / sig if sig > 1e-10 else s - mu
        samples.append(s)
    return samples


def normalize_params(params_list: List[np.ndarray],
                     mode: Literal['zscore', 'mean_ratio', 'log_zscore', 'none'] = 'none',
                     clip_quantile: float = 0.0) -> List[np.ndarray]:
    """
    Normalize estimated parameter time series for scale-invariant comparison.

    Args:
        params_list:   List of (T, 1) parameter arrays.
        mode:          'zscore'     — subtract mean, divide by std (may yield negatives).
                       'mean_ratio' — divide by temporal mean (keeps positivity).
                       'log_zscore' — z-normalize log(λ) (scale-invariant, positive output
                                      via exp; robust to heavy tails).
                       'none'       — no normalization.
        clip_quantile: If > 0, clip each series to [q, 1-q] percentile before normalizing.
                       Use 0.01-0.05 to suppress extreme flood/drought outliers.

    Returns:
        List of normalized (T, 1) arrays.
    """
    result = []
    for p in params_list:
        p = p.copy()
        if clip_quantile > 0:
            lo, hi = np.percentile(p, clip_quantile * 100), np.percentile(p, (1 - clip_quantile) * 100)
            p = np.clip(p, lo, hi)
        if mode == 'zscore':
            mu, sig = p.mean(), p.std()
            p = (p - mu) / sig if sig > 1e-10 else p - mu
        elif mode == 'mean_ratio':
            mu = p.mean()
            p = p / mu if mu > 1e-10 else p
        elif mode == 'log_zscore':
            # z-normalize log(λ) then exponentiate — positive output, robust to heavy tails
            log_p = np.log(np.maximum(p, 1e-8))
            mu, sig = log_p.mean(), log_p.std()
            log_p = (log_p - mu) / sig if sig > 1e-10 else log_p - mu
            p = np.exp(log_p)
        elif mode == 'log_zscore_linear':
            # z-normalize log(λ) — returns z-scores directly (not exponentiated)
            # Suitable for Euclidean DTW: values in ~(-3, 3), no divergence issues
            log_p = np.log(np.maximum(p, 1e-8))
            mu, sig = log_p.mean(), log_p.std()
            p = (log_p - mu) / sig if sig > 1e-10 else log_p - mu
        elif mode == 'wasserstein_euclidean':
            # Wasserstein-equivalent Euclidean feature: μ = 1/λ, z-normalized.
            # W₂²(Exp(λ₁), Exp(λ₂)) = 2(1/λ₁ − 1/λ₂)² = 2‖μ₁ − μ₂‖², so
            # minimizing Soft-DTW squared Euclidean loss on μ is exactly equivalent
            # to minimizing the Soft-DTW Wasserstein loss on λ. Standard DBA applies
            # — no SGD, no custom Jacobian needed.
            mu_p = 1.0 / np.maximum(p, 1e-8)
            mean, sig = mu_p.mean(), mu_p.std()
            p = (mu_p - mean) / sig if sig > 1e-10 else mu_p - mean
        result.append(p)
    return result


def estimate_parameters(
        X_list: List[np.ndarray],
        distribution: str = 'exponential',
        estimator: Literal['log_cumulant', 'mle'] = 'log_cumulant',
) -> List[np.ndarray]:
    """
    Estimate per-timestep distribution parameters for each sample.

    For each sample (T, D), pools the D feature values at each timestep to
    estimate one parameter per timestep, returning shape (T, 1).

    When D = 1 the single value is treated as the parameter directly (no
    estimation), which is the correct interpretation for already-estimated
    or univariate series.

    Args:
        X_list:       List of (T, D) arrays.
        distribution: Distribution family — currently only 'exponential'.
        estimator:    'log_cumulant' (default) or 'mle'.

    Returns:
        List of (T, 1) parameter arrays.
    """
    est_cls = MLE if estimator == 'mle' else LogCumulant
    est_obj = est_cls(distribution=distribution)
    params_list = []
    for X in X_list:
        T, D = X.shape
        params = np.zeros((T, 1), dtype=np.float64)
        if D == 1:
            params[:, 0] = X[:, 0]
        else:
            for t in range(T):
                vals = X[t][~np.isnan(X[t]) & (X[t] > 0)]
                if len(vals) > 0:
                    est_obj.fit(vals)
                    params[t, 0] = est_obj.get_params()
                else:
                    params[t, 0] = np.nan
            # Fill residual NaN with column mean
            valid = ~np.isnan(params[:, 0])
            if valid.any() and (~valid).any():
                params[~valid, 0] = params[valid, 0].mean()
            elif not valid.any():
                params[:, 0] = 1.0
        params_list.append(params)
    return params_list
