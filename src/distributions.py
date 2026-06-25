"""Parametric distributions + estimation (merged from estimation.py).

Pure-JAX distributions for the pdf and two numpy/scipy estimators
(MLE and log-cumulants) for each family.

Parameterisations (matching CLAUDE.md conventions):
  Exponential: rate β  → E[X] = 1/β,  pdf = β·exp(−βx)
  Weibull:     (k, λ)  → CDF F(x) = 1 − exp(−(x/λ)^k)
               pdf = (k/λ)(x/λ)^(k−1)·exp(−(x/λ)^k)   (scale λ, NOT rate)

Estimation note:
  All estimation runs outside the JAX graph (numpy/scipy, no jit).
  fit_time_series expects clean input — call clean_time_series() from
  data.preprocess at the raw→estimation boundary (runners, notebooks).
  NaN policy: timesteps with < min_valid clean samples → NaN params
  (no silent imputation).

Float64 note:
  Weibull estimation uses digamma/gammaln — float64 required for precision.
  A UserWarning is raised when dtype != float64 for Weibull.
  JAX x64 must be enabled (`jax.config.update("jax_enable_x64", True)`) before float64 JAX ops (pdf).

All pdf functions are vmap-able.
"""

from __future__ import annotations

import warnings
from typing import Protocol

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)


import jax.numpy as jnp
from scipy.special import digamma
from scipy.stats import weibull_min, expon


# ψ(1) = −γ_EM  (digamma(1) ≈ −0.5772156649)
_DIGAMMA_1: float = digamma(1)
# π / sqrt(6)  — used in Weibull shape estimator
_PI_OVER_SQRT6: float = np.pi / np.sqrt(6.0)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class Distribution(Protocol):
    """Interface for a parametric 1-D distribution used in WaSPS."""

    n_params: int

    def pdf(self, x: jax.Array, params: jax.Array) -> jax.Array:
        """Probability density at x.  params shape (..., n_params)."""
        ...

    def estimate(self, sample: np.ndarray, method: str = 'mle'):
        """Fit parameters from a 1-D sample (already cleaned).

        Returns scalar (exponential) or (k, λ) tuple (Weibull).
        """
        ...

    def fit_time_series(
        self,
        series,
        method: str = 'mle',
        min_valid: int = 5,
        dtype=np.float64,
    ) -> np.ndarray:
        """Fit parameters for each timestep of a series.

        series: list of 1-D arrays or (T, N) 2-D array.
                Samples must already be cleaned (see clean_time_series).
        Returns (T, n_params) array of dtype.
        """
        ...


# ---------------------------------------------------------------------------
# Single-sample estimators (module-level for direct import / testing)
# ---------------------------------------------------------------------------

def fit_exponential_log_cumulant(x: np.ndarray) -> float:
    """Exponential rate β via log-cumulant: β = 1/exp(E[log X] − ψ(1)).

    Matches Cython estimate_exponential_fast.
    """
    x = np.asarray(x, dtype=np.float64)
    log_mean = np.mean(np.log(x))
    return 1.0 / np.exp(log_mean - _DIGAMMA_1)


def fit_exponential_mle(x: np.ndarray) -> float:
    """Exponential rate β via scipy MLE: β = 1/scale from expon.fit(floc=0)."""
    x = np.asarray(x, dtype=np.float64)
    _, scale = expon.fit(x, floc=0)
    return 1.0 / scale


def fit_weibull_log_cumulant(x: np.ndarray) -> tuple[float, float]:
    """Weibull (k, λ) via method of log-cumulants.

    k = π / (sqrt(6) · σ_log),  λ = exp(μ_log − ψ(1)/k).
    Matches Cython estimate_weibull_fast.
    """
    x = np.asarray(x, dtype=np.float64)
    logs = np.log(x)
    mu_log = logs.mean()
    sigma_log = np.sqrt(logs.var()) if logs.var() > 1e-12 else 1e-6
    k_hat = _PI_OVER_SQRT6 / sigma_log
    lam_hat = np.exp(mu_log - _DIGAMMA_1 / k_hat)
    return k_hat, lam_hat


def fit_weibull_mle(x: np.ndarray) -> tuple[float, float]:
    """Weibull (k, λ) via scipy MLE with floc=0.

    Returns (shape k, scale λ) — same parameterisation as W₂² in costs.py.
    Verified: scipy.stats.weibull_min(c=k, scale=λ, loc=0) matches our pdf.
    """
    x = np.asarray(x, dtype=np.float64)
    k_hat, _, lam_hat = weibull_min.fit(x, floc=0)
    return k_hat, lam_hat


# ---------------------------------------------------------------------------
# Distribution classes
# ---------------------------------------------------------------------------

class Exponential:
    """Exponential distribution parameterised by rate β > 0.

    E[X] = 1/β,  CDF F(x) = 1 − exp(−βx).
    Matches scipy.stats.expon(scale=1/β).pdf(x).
    params: shape (..., 1), column 0 = β.
    """

    n_params: int = 1

    def pdf(self, x: jax.Array, params: jax.Array) -> jax.Array:
        beta = params[..., 0]
        return beta * jnp.exp(-beta * x)

    def estimate(self, sample: np.ndarray, method: str = 'mle') -> float:
        """Fit rate β from a 1-D sample (already cleaned)."""
        if method == 'log_cumulant':
            return fit_exponential_log_cumulant(sample)
        return fit_exponential_mle(sample)

    def fit_time_series(
        self,
        series,
        method: str = 'mle',
        min_valid: int = 5,
        dtype=np.float64,
    ) -> np.ndarray:
        """Fit rate β for each timestep.  Returns (T, 1) array."""
        if method not in ('mle', 'log_cumulant'):
            raise ValueError(f"method must be 'mle' or 'log_cumulant', got '{method}'")
        rows = _to_rows(series)
        params = np.full((len(rows), 1), np.nan, dtype=dtype)
        for t, row in enumerate(rows):
            values = _finite(row)
            if len(values) < min_valid:
                continue
            params[t, 0] = self.estimate(values, method)
        return params.astype(dtype)


class Weibull:
    """Weibull distribution parameterised by shape k > 0 and scale λ > 0.

    CDF F(x) = 1 − exp(−(x/λ)^k).
    Matches scipy.stats.weibull_min(c=k, scale=λ, loc=0).pdf(x).
    params: shape (..., 2), column 0 = k, column 1 = λ.
    """

    n_params: int = 2

    def pdf(self, x: jax.Array, params: jax.Array) -> jax.Array:
        k = params[..., 0]
        lam = params[..., 1]
        z = x / lam
        return (k / lam) * z ** (k - 1.0) * jnp.exp(-(z ** k))

    def estimate(self, sample: np.ndarray, method: str = 'mle') -> tuple[float, float]:
        """Fit (k, λ) from a 1-D sample (already cleaned)."""
        if method == 'log_cumulant':
            return fit_weibull_log_cumulant(sample)
        return fit_weibull_mle(sample)

    def fit_time_series(
        self,
        series,
        method: str = 'mle',
        min_valid: int = 5,
        dtype=np.float64,
    ) -> np.ndarray:
        """Fit (k, λ) for each timestep.  Returns (T, 2) array."""
        if dtype != np.float64:
            warnings.warn(
                "Weibull estimation uses digamma/gammaln — "
                "float64 may provide better precision and higher numerical stability.",
                UserWarning,
                stacklevel=2,
            )
        if method not in ('mle', 'log_cumulant'):
            raise ValueError(f"method must be 'mle' or 'log_cumulant', got '{method}'")
        rows = _to_rows(series)
        params = np.full((len(rows), 2), np.nan, dtype=dtype)
        for t, row in enumerate(rows):
            values = _finite(row)
            if len(values) < min_valid:
                continue
            k, lam = self.estimate(values, method)
            params[t, 0] = k
            params[t, 1] = lam
        return params.astype(dtype)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _to_rows(series) -> list:
    """Convert (T, N) array or list to list of 1-D arrays."""
    if isinstance(series, np.ndarray) and series.ndim == 2:
        return [series[t] for t in range(series.shape[0])]
    return list(series)


def _finite(row: np.ndarray) -> np.ndarray:
    """Keep only finite values (NaN / ±inf removed).  Returns float64 1-D array."""
    v = np.asarray(row, dtype=np.float64).ravel()
    return v[np.isfinite(v)]


# ---------------------------------------------------------------------------
# Singletons (avoids re-instantiation)
# ---------------------------------------------------------------------------

exponential = Exponential()
weibull = Weibull()


def get(family: str) -> Distribution:
    """Return the distribution singleton for 'exponential' or 'weibull'."""
    family = family.lower()
    if family == 'exponential':
        return exponential
    if family == 'weibull':
        return weibull
    raise ValueError(f"Unknown distribution family: '{family}'. "
                     "Use 'exponential' or 'weibull'.")


# ---------------------------------------------------------------------------
# Thin wrapper (backward compat / callers that used estimation.fit)
# ---------------------------------------------------------------------------

def fit(
    series,
    family: str,
    method: str = 'mle',
    min_valid: int = 5,
    dtype=np.float64,
) -> np.ndarray:
    """Fit distribution parameters for each timestep.

    Thin wrapper around Distribution.fit_time_series.
    series must already be cleaned — call clean_time_series() from
    data.preprocess at the raw→estimation boundary.

    Returns: (T, n_params) array of dtype.
    """
    if family.lower() not in ('exponential', 'weibull'):
        raise ValueError(f"family must be 'exponential' or 'weibull', got '{family}'")
    return get(family).fit_time_series(series, method=method, min_valid=min_valid, dtype=dtype)
