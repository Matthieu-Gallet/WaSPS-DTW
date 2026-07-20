"""Tests for classification/nn.py and barycenter_clf.py — Phase D validation."""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax

from costs import WaSPS, SqEuclidean
from softdtw import SoftDTW
from classification.nn import knn_predict
from classification.barycenter_clf import fit_barycenters, predict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_two_class_data(seed=0):
    """Synthetic 2-class data: class 0 has β≈1, class 1 has β≈3 (exponential params)."""
    rng = np.random.default_rng(seed)
    T = 6
    n_train_per_class = 5
    train_series, train_labels = [], []
    for cls, beta in enumerate([1.0, 3.0]):
        for _ in range(n_train_per_class):
            # (T, 1) exponential rate array, slightly perturbed
            s = rng.uniform(beta * 0.8, beta * 1.2, (T, 1)).astype(np.float64)
            train_series.append(s)
            train_labels.append(cls)
    test_series = [
        np.full((T, 1), 0.9, dtype=np.float64),   # should be class 0
        np.full((T, 1), 3.1, dtype=np.float64),   # should be class 1
    ]
    test_labels = np.array([0, 1])
    return train_series, np.array(train_labels), test_series, test_labels


def _make_softdtw_bary(gamma=1.0):
    """SoftDTW for barycenter (WaSPS with positivity constraint)."""
    cost_fn = WaSPS('exponential', use_positivity_constraint=True)
    return SoftDTW(cost_fn, gamma, is_divergence=True, manual_grad=True)


# ---------------------------------------------------------------------------
# k-NN
# ---------------------------------------------------------------------------

class TestKNN:
    def test_output_shape(self):
        train_s, train_l, test_s, _ = _make_two_class_data()
        preds = knn_predict(train_s, train_l, test_s, WaSPS('exponential'), gamma=1.0, k=1)
        assert preds.shape == (2,)

    def test_separable_k1(self):
        train_s, train_l, test_s, test_l = _make_two_class_data(seed=42)
        cost_fn = WaSPS('exponential')
        preds = knn_predict(train_s, train_l, test_s, cost_fn, gamma=1.0, k=1)
        assert np.array_equal(preds, test_l), f"Expected {test_l}, got {preds}"

    def test_k_larger_than_one(self):
        train_s, train_l, test_s, _ = _make_two_class_data()
        preds = knn_predict(train_s, train_l, test_s, WaSPS('exponential'), gamma=1.0, k=3)
        assert preds.shape == (2,)
        assert set(preds.tolist()).issubset({0, 1})

    def test_euclidean_knn(self):
        rng = np.random.default_rng(5)
        T = 4
        # Class 0: values near 0, class 1: values near 5
        train_s = [rng.uniform(-0.2, 0.2, (T, 2)).astype(np.float64) for _ in range(4)]
        train_s += [rng.uniform(4.8, 5.2, (T, 2)).astype(np.float64) for _ in range(4)]
        train_l = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        test_s = [np.zeros((T, 2)), np.full((T, 2), 5.0)]
        preds = knn_predict(train_s, train_l, test_s, SqEuclidean(), gamma=1.0, k=1)
        assert np.array_equal(preds, [0, 1])


# ---------------------------------------------------------------------------
# Barycenter classifier
# ---------------------------------------------------------------------------

class TestBarycentClf:
    def test_fit_barycenters_shape(self):
        train_s, train_l, _, _ = _make_two_class_data()
        barycenters = fit_barycenters(train_s, train_l, _make_softdtw_bary(),
                                      n_steps=50, lr=1e-2)
        assert set(barycenters.keys()) == {0, 1}
        assert barycenters[0].shape == (6, 1)
        assert barycenters[1].shape == (6, 1)
        # use_positivity_constraint ensures β > 0
        assert np.all(barycenters[0] > 0)
        assert np.all(barycenters[1] > 0)

    def test_barycenter_separable(self):
        train_s, train_l, test_s, test_l = _make_two_class_data(seed=1)
        barycenters = fit_barycenters(train_s, train_l, _make_softdtw_bary(),
                                      n_steps=200, lr=1e-2)
        # Barycenter of class 0 should have lower β than class 1
        b0 = barycenters[0].mean()
        b1 = barycenters[1].mean()
        assert b0 < b1, f"Class 0 barycenter β={b0:.3f} should be < class 1 β={b1:.3f}"

        # predict receives positive-param barycenters and positive-param test series
        cost_fn_eval = WaSPS('exponential')
        preds = predict(test_s, barycenters, cost_fn_eval, gamma=1.0)
        assert np.array_equal(preds, test_l), f"Expected {test_l}, got {preds}"

    def test_predict_output_shape(self):
        train_s, train_l, test_s, _ = _make_two_class_data()
        barycenters = fit_barycenters(train_s, train_l, _make_softdtw_bary(), n_steps=50)
        preds = predict(test_s, barycenters, WaSPS('exponential'), gamma=1.0)
        assert preds.shape == (2,)
        assert set(preds.tolist()).issubset({0, 1})

    def test_divergence_to_own_barycenter_lower(self):
        # Each class test point should have lower divergence to its own barycenter
        train_s, train_l, test_s, test_l = _make_two_class_data(seed=2)
        barycenters = fit_barycenters(train_s, train_l, _make_softdtw_bary(),
                                      n_steps=200, lr=1e-2)
        preds = predict(test_s, barycenters, WaSPS('exponential'), gamma=1.0)
        # At least 1 test point classified correctly (weak sanity check)
        assert np.sum(preds == test_l) >= 1

    def test_fit_barycenters_parallel_matches_sequential(self):
        # n_jobs=-1 dispatches one class per joblib/loky worker (separate process,
        # separate JAX state) — verify it produces the same result as n_jobs=1, not
        # just that it doesn't crash.
        train_s, train_l, _, _ = _make_two_class_data(seed=3)
        sequential = fit_barycenters(train_s, train_l, _make_softdtw_bary(),
                                     n_steps=30, lr=1e-2, n_jobs=1, verbose=False)
        parallel = fit_barycenters(train_s, train_l, _make_softdtw_bary(),
                                   n_steps=30, lr=1e-2, n_jobs=-1, verbose=False)
        assert set(parallel.keys()) == set(sequential.keys())
        for cls in sequential:
            np.testing.assert_allclose(parallel[cls], sequential[cls], rtol=1e-6)
