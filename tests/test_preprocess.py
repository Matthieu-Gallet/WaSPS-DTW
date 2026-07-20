"""Tests for clean_series — Phase 2 validation."""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data.preprocess import clean_series, clean_time_series, to_fixed_n


def test_output_strictly_positive():
    x = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    out = clean_series(x)
    assert len(out) > 0
    assert np.all(out > 0)


def test_output_finite():
    x = np.array([1.0, 2.0, np.nan, np.inf, -np.inf, 3.0])
    out = clean_series(x)
    assert np.all(np.isfinite(out))


def test_removes_non_positive():
    x = np.array([-1.0, 0.0, 0.5, 1.0])
    out = clean_series(x)
    assert np.all(out > 0)
    # only 0.5 and 1.0 survive the first filter; after shift min+eps is dropped
    assert len(out) >= 1


def test_length_consistent():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = clean_series(x)
    # min (1.0) is subtracted, so the minimum value itself is removed
    assert len(out) == len(x) - 1


def test_empty_input_returns_empty():
    out = clean_series(np.array([]))
    assert len(out) == 0


def test_all_non_positive_returns_empty():
    out = clean_series(np.array([-1.0, 0.0, -5.0]))
    assert len(out) == 0


def test_single_positive_value_returns_empty():
    # single value → after shift it becomes -eps → removed
    out = clean_series(np.array([3.0]))
    assert len(out) == 0


def test_output_1d():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2D input
    out = clean_series(x)
    assert out.ndim == 1


def test_min_shift_is_applied():
    x = np.array([1.0, 2.0, 3.0])
    out = clean_series(x)
    # after shift: [1-1-eps, 2-1-eps, 3-1-eps] = [-eps, 1-eps, 2-eps]
    # keep positive: [1-eps, 2-eps]
    assert len(out) == 2
    assert np.allclose(out, np.array([1.0 - 1e-8, 2.0 - 1e-8]), atol=1e-12)


# ---------------------------------------------------------------------------
# clean_time_series
# ---------------------------------------------------------------------------

def test_clean_time_series_length_matches_T():
    series = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (T=2, N=3)
    out = clean_time_series(series)
    assert len(out) == 2


def test_clean_time_series_applies_clean_series_per_row():
    series = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 5.0]])
    out = clean_time_series(series)
    assert np.array_equal(out[0], clean_series(series[0]))
    assert np.array_equal(out[1], clean_series(series[1]))


def test_clean_time_series_output_is_ragged():
    # row 0: all survive clean_series's filters differently from row 1 (NaN present)
    series = [np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, np.nan, np.nan])]
    out = clean_time_series(series)
    assert len(out) == 2
    assert len(out[0]) != len(out[1])


def test_clean_time_series_accepts_list_input():
    series = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    out = clean_time_series(series)
    assert len(out) == 2


# ---------------------------------------------------------------------------
# to_fixed_n
# ---------------------------------------------------------------------------

def test_to_fixed_n_output_shape():
    rng = np.random.default_rng(0)
    series = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])  # (T=2, N=4)
    out = to_fixed_n(series, n=6, rng=rng)
    assert out.shape == (2, 6)


def test_to_fixed_n_subsamples_without_replacement_when_enough_valid():
    rng = np.random.default_rng(0)
    series = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])  # (T=1, N=5), all valid
    out = to_fixed_n(series, n=3, rng=rng)
    # subsampled without replacement -> 3 distinct values, all from the source
    assert len(np.unique(out[0])) == 3
    assert set(out[0]).issubset(set(series[0]))


def test_to_fixed_n_resamples_with_replacement_when_not_enough_valid():
    rng = np.random.default_rng(0)
    series = np.array([[1.0, 2.0]])  # only 2 valid values, n=5 requested
    out = to_fixed_n(series, n=5, rng=rng)
    assert out.shape == (1, 5)
    assert set(out[0]).issubset({1.0, 2.0})


def test_to_fixed_n_fills_zero_valid_timestep_with_floor():
    rng = np.random.default_rng(0)
    series = np.array([[-1.0, 0.0, np.nan]])  # no valid values at this timestep
    out = to_fixed_n(series, n=4, rng=rng)
    assert np.allclose(out[0], 1e-8)


def test_to_fixed_n_output_finite_and_positive():
    rng = np.random.default_rng(1)
    series = [np.array([1.0, 2.0, np.nan]), np.array([-1.0, 0.0, 3.0])]
    out = to_fixed_n(series, n=4, rng=rng)
    assert np.all(np.isfinite(out))
    assert np.all(out > 0)


def test_to_fixed_n_does_not_apply_min_shift():
    # unlike clean_series, to_fixed_n must preserve raw positive values as-is
    rng = np.random.default_rng(0)
    series = np.array([[5.0, 5.0, 5.0]])
    out = to_fixed_n(series, n=3, rng=rng)
    assert np.allclose(out[0], 5.0)
