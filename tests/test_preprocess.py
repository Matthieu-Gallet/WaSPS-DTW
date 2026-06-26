"""Tests for clean_series — Phase 2 validation."""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data.preprocess import clean_series


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
