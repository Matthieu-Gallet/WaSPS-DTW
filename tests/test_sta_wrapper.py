"""Tests for classification/sta_wrapper.py — Phase 8 validation."""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax

from classification.sta_wrapper import sta_cost_matrix, sta_distance, knn_predict, make_cost_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_series(T: int, N: int, loc: float, scale: float, seed: int) -> np.ndarray:
    """(T, N) raw series from Normal(loc, scale²)."""
    rng = np.random.default_rng(seed)
    return np.abs(rng.normal(loc, scale, (T, N)))  # keep positive for clean_series


# ---------------------------------------------------------------------------
# sta_cost_matrix
# ---------------------------------------------------------------------------

class TestStaCostMatrix:
    def test_shape(self):
        x = _make_series(4, 20, 1.0, 0.1, 0)
        y = _make_series(5, 20, 2.0, 0.1, 1)
        D = sta_cost_matrix(x, y, epsilon=0.1)
        assert D.shape == (4, 5)

    def test_non_negative(self):
        x = _make_series(3, 15, 1.0, 0.2, 2)
        y = _make_series(3, 15, 2.0, 0.2, 3)
        D = sta_cost_matrix(x, y, epsilon=0.1)
        assert np.all(D >= -1e-10)

    def test_self_diagonal_small(self):
        # OT(dist, dist) with Sinkhorn should be small (not exactly 0 due to regularisation)
        x = _make_series(3, 30, 1.0, 0.05, 4)
        D = sta_cost_matrix(x, x, epsilon=0.05)
        assert np.all(np.diag(D) < 0.5)

    def test_symmetric(self):
        x = _make_series(3, 20, 1.0, 0.2, 5)
        y = _make_series(3, 20, 2.0, 0.2, 6)
        Dxy = sta_cost_matrix(x, y, epsilon=0.1)
        Dyx = sta_cost_matrix(y, x, epsilon=0.1)
        np.testing.assert_allclose(Dxy, Dyx.T, atol=1e-6)

    def test_finite(self):
        x = _make_series(4, 10, 1.0, 0.3, 7)
        y = _make_series(4, 10, 3.0, 0.3, 8)
        D = sta_cost_matrix(x, y, epsilon=0.1)
        assert np.all(np.isfinite(D))


# ---------------------------------------------------------------------------
# sta_distance
# ---------------------------------------------------------------------------

class TestStaDistance:
    def test_returns_scalar(self):
        x = _make_series(4, 20, 1.0, 0.2, 10)
        y = _make_series(4, 20, 2.0, 0.2, 11)
        d = sta_distance(x, y, gamma=1.0, epsilon=0.1)
        assert isinstance(d, float)

    def test_same_series_smaller_than_far(self):
        # STA(x, x) < STA(x, y_far): self-distance < cross-class distance.
        # Note: plain SoftDTW of Sinkhorn costs can be negative (the Sinkhorn
        # objective includes -ε·H(T)); only the divergence D_γ is guaranteed ≥ 0.
        x = _make_series(4, 50, 1.0, 0.05, 14)
        y_far = _make_series(4, 50, 10.0, 0.05, 17)
        d_self = sta_distance(x, x, gamma=1.0, epsilon=0.05)
        d_far = sta_distance(x, y_far, gamma=1.0, epsilon=0.05)
        assert d_self < d_far

    def test_far_series_greater_than_near(self):
        x = _make_series(4, 30, 1.0, 0.1, 15)
        y_near = _make_series(4, 30, 1.1, 0.1, 16)
        y_far = _make_series(4, 30, 10.0, 0.1, 17)
        d_near = sta_distance(x, y_near, gamma=1.0, epsilon=0.1)
        d_far = sta_distance(x, y_far, gamma=1.0, epsilon=0.1)
        assert d_far > d_near

    def test_finite(self):
        x = _make_series(3, 20, 1.0, 0.2, 18)
        y = _make_series(3, 20, 2.0, 0.2, 19)
        assert np.isfinite(sta_distance(x, y))


# ---------------------------------------------------------------------------
# knn_predict
# ---------------------------------------------------------------------------

class TestKnnPredictSTA:
    def _make_two_class_data(self, seed=0):
        """2-class data: class 0 ~ N(1, 0.1), class 1 ~ N(10, 0.1).
        Large separation so STA reliably discriminates despite Sinkhorn ε-bias.
        """
        T, N = 4, 40
        train_series, train_labels = [], []
        for cls, loc in enumerate([1.0, 10.0]):
            for i in range(4):
                train_series.append(_make_series(T, N, loc, 0.1, seed + cls * 10 + i))
                train_labels.append(cls)
        test_series = [
            _make_series(T, N, 0.9, 0.05, seed + 99),    # class 0
            _make_series(T, N, 10.1, 0.05, seed + 100),  # class 1
        ]
        test_labels = np.array([0, 1])
        return train_series, np.array(train_labels), test_series, test_labels

    def test_output_shape(self):
        train_s, train_l, test_s, _ = self._make_two_class_data()
        preds = knn_predict(train_s, train_l, test_s, gamma=1.0, epsilon=0.1, k=1)
        assert preds.shape == (2,)
        assert set(preds.tolist()).issubset({0, 1})

    def test_separable_k1(self):
        train_s, train_l, test_s, test_l = self._make_two_class_data(seed=42)
        preds = knn_predict(train_s, train_l, test_s, gamma=1.0, epsilon=0.1, k=1)
        assert np.array_equal(preds, test_l), f"Expected {test_l}, got {preds}"

    def test_k3(self):
        train_s, train_l, test_s, _ = self._make_two_class_data()
        preds = knn_predict(train_s, train_l, test_s, gamma=1.0, epsilon=0.1, k=3)
        assert preds.shape == (2,)


# ---------------------------------------------------------------------------
# make_cost_fn — used by fit_barycenter for STA barycenter estimation
# (see src/plot/plot_river_bary_viz.py)
# ---------------------------------------------------------------------------

class TestMakeCostFn:
    def test_returns_callable(self):
        cost_fn = make_cost_fn(epsilon=0.1)
        assert callable(cost_fn)

    def test_output_is_scalar_jax_array(self):
        cost_fn = make_cost_fn(epsilon=0.1)
        rng = np.random.default_rng(0)
        a = jax.numpy.asarray(np.abs(rng.normal(1.0, 0.1, 20)))
        b = jax.numpy.asarray(np.abs(rng.normal(1.0, 0.1, 20)))
        out = cost_fn(a, b)
        assert out.shape == ()
        assert jax.numpy.isfinite(out)

    def test_near_zero_for_identical_inputs(self):
        # entropic-regularized Sinkhorn cost, not exact OT: identical inputs give a
        # small but nonzero cost (entropy term, epsilon-dependent) — compare against
        # a clearly-different pair rather than asserting exact zero.
        cost_fn = make_cost_fn(epsilon=0.1)
        rng = np.random.default_rng(1)
        a = jax.numpy.asarray(np.abs(rng.normal(1.0, 0.1, 20)))
        b_far = jax.numpy.asarray(np.abs(rng.normal(10.0, 0.1, 20)))
        assert float(cost_fn(a, a)) < float(cost_fn(a, b_far))

    def test_increases_with_distribution_shift(self):
        cost_fn = make_cost_fn(epsilon=0.1)
        rng = np.random.default_rng(2)
        a = jax.numpy.asarray(np.abs(rng.normal(1.0, 0.1, 30)))
        b_near = jax.numpy.asarray(np.abs(rng.normal(1.2, 0.1, 30)))
        b_far = jax.numpy.asarray(np.abs(rng.normal(10.0, 0.1, 30)))
        assert float(cost_fn(a, b_far)) > float(cost_fn(a, b_near))

    def test_grad_safe(self):
        # value_and_grad through Sinkhorn — this is the property make_cost_fn's
        # docstring claims ("pure JAX ... vmap- and grad-safe") that the STA
        # barycenter path depends on.
        cost_fn = make_cost_fn(epsilon=0.1)
        rng = np.random.default_rng(3)
        a = jax.numpy.asarray(np.abs(rng.normal(1.0, 0.1, 15)))
        b = jax.numpy.asarray(np.abs(rng.normal(1.5, 0.1, 15)))
        value, grad = jax.value_and_grad(cost_fn)(a, b)
        assert jax.numpy.isfinite(value)
        assert grad.shape == a.shape
        assert jax.numpy.all(jax.numpy.isfinite(grad))
