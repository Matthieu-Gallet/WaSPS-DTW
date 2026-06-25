"""Canonical time-series preprocessing: clean_series(x), clean_time_series(series),
and to_fixed_n(series, n, rng) for rectangularising raw arrays."""

import numpy as np


def clean_series(x: np.ndarray) -> np.ndarray:
    """Return strictly positive values after removing non-finite/non-positive entries
    and subtracting (min + 1e-8).

    Steps: keep finite → keep positive → subtract (min + 1e-8) → keep positive.
    The subtraction ensures no value sits exactly at zero, satisfying floc=0 fits.

    Returns empty array (length 0) if no values survive.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    x = x[x > 0.0]
    if len(x) == 0:
        return x
    x = x - (x.min() + 1e-8)
    return x[x > 0.0]


def clean_time_series(series) -> list:
    """Clean each timestep of a time series (apply clean_series per row).

    Args:
        series: (T, N) array or list of length T, where each element contains
                N sample values at that timestep.

    Returns:
        List of length T; each element is a 1-D float64 array of clean
        (strictly positive, finite) values.  The output is ragged — rows may
        have different lengths after filtering.

    Call this at the raw→estimation boundary (before fit_time_series) for
    real-world data that may contain NaN, zeros, or non-positive values.
    Synthetic data from rng.exponential() / rng.weibull() does not need this.
    """
    if isinstance(series, np.ndarray) and series.ndim == 2:
        rows = [series[t] for t in range(series.shape[0])]
    else:
        rows = list(series)
    return [clean_series(row) for row in rows]


def to_fixed_n(series, n: int, rng: np.random.Generator) -> np.ndarray:
    """Rectangularise a (T, N_raw) array or list-of-arrays to shape (T, n).

    Per timestep: keep finite & strictly-positive values, then either subsample
    (without replacement, if valid ≥ n) or resample (with replacement, if valid < n).
    Timesteps with zero valid values are filled with 1e-8 (fallback).

    Does NOT call clean_series — this function must not apply the min-shift
    (``x - (min+1e-8)``) that clean_series performs for floc=0 MLE fits;
    that shift is reserved for the params estimation path.

    Args:
        series: (T, N_raw) float64 array or list of T 1-D arrays.
        n:      Target number of samples per timestep.
        rng:    NumPy random generator (caller controls the seed).

    Returns:
        (T, n) float64 array, NaN-free and strictly positive.
    """
    if isinstance(series, np.ndarray) and series.ndim == 2:
        rows = [series[t] for t in range(series.shape[0])]
    else:
        rows = list(series)

    out = np.empty((len(rows), n), dtype=np.float64)
    for t, row in enumerate(rows):
        row = np.asarray(row, dtype=np.float64).ravel()
        valid = row[np.isfinite(row) & (row > 0.0)]
        if len(valid) == 0:
            out[t] = 1e-8
        elif len(valid) >= n:
            idx = rng.choice(len(valid), size=n, replace=False)
            out[t] = valid[idx]
        else:
            idx = rng.choice(len(valid), size=n, replace=True)
            out[t] = valid[idx]
    return out
