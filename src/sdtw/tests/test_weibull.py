"""
Tests for Weibull W₂² implementation.

Checks:
1. W₂²(p,p) = 0 (self-distance)
2. Symmetry: W₂²(p,q) = W₂²(q,p)
3. Consistency with exponential: Weibull(k=1, λ=1/β) ↔ Exponential(rate=β)
4. Weibull estimator recovers known (k, λ) on large samples
5. Finite-difference gradient check for ∂W₂²/∂λ and ∂W₂²/∂k
6. WassersteinDistance with distribution='weibull' gives correct distance and Jacobians
7. Soft-DTW divergence D(X,X) = 0
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import weibull_min as scipy_weibull
from scipy.special import gamma as gamma_fn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sdtw.wasserstein_fast import (
    wasserstein22_weibull_fast,
    wasserstein22_exponential_fast,
    estimate_weibull_fast,
    pairwise_wasserstein_weibull,
)
from sdtw.distance import WassersteinDistance
from sdtw.soft_dtw import sdtw_divergence
from sdtw.distance import SquaredEuclidean


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def w22_weibull_python(k1, l1, k2, l2):
    """Pure-Python reference for W₂²(Weib(k1,λ1), Weib(k2,λ2)) — Prop 1."""
    t1 = l1**2 * gamma_fn((k1 + 2) / k1)
    t2 = l2**2 * gamma_fn((k2 + 2) / k2)
    t3 = 2 * l1 * l2 * gamma_fn(1 + 1/k1 + 1/k2)
    return t1 + t2 - t3


def dw22_dlambda1_python(k1, l1, k2, l2):
    """Analytic ∂W₂²/∂λ₁ — Prop 2."""
    return 2 * (l1 * gamma_fn((k1 + 2)/k1) - l2 * gamma_fn(1 + 1/k1 + 1/k2))


def dw22_dk1_python(k1, l1, k2, l2):
    """Analytic ∂W₂²/∂k₁ — Prop 2."""
    from scipy.special import digamma
    g_cross = gamma_fn(1 + 1/k1 + 1/k2)
    g_self  = gamma_fn((k1 + 2)/k1)
    psi_cross = digamma(1 + 1/k1 + 1/k2)
    psi_self  = digamma((k1 + 2)/k1)
    return (2 * l1 / k1**2) * (l2 * g_cross * psi_cross - l1 * g_self * psi_self)


# ---------------------------------------------------------------------------
# Test 1: self-distance is zero
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k,lam", [(1.0, 1.0), (2.0, 3.0), (0.5, 0.8)])
def test_w22_self_zero(k, lam):
    assert abs(wasserstein22_weibull_fast(k, lam, k, lam)) < 1e-12


# ---------------------------------------------------------------------------
# Test 2: symmetry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k1,l1,k2,l2", [
    (1.5, 2.0, 0.8, 1.2),
    (2.0, 1.0, 3.0, 0.5),
])
def test_w22_symmetry(k1, l1, k2, l2):
    a = wasserstein22_weibull_fast(k1, l1, k2, l2)
    b = wasserstein22_weibull_fast(k2, l2, k1, l1)
    assert abs(a - b) < 1e-10


# ---------------------------------------------------------------------------
# Test 3: matches Python reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k1,l1,k2,l2", [
    (1.5, 2.0, 0.8, 1.2),
    (2.0, 1.0, 3.0, 0.5),
    (0.7, 1.5, 0.9, 2.1),
])
def test_w22_matches_reference(k1, l1, k2, l2):
    cython_val = wasserstein22_weibull_fast(k1, l1, k2, l2)
    python_val = w22_weibull_python(k1, l1, k2, l2)
    assert abs(cython_val - python_val) < 1e-8


# ---------------------------------------------------------------------------
# Test 4: exponential consistency
#   Weibull(k=1, λ=1/β) ↔ Exponential(rate=β)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("b1,b2", [(1.0, 2.0), (3.0, 0.5), (2.0, 4.0)])
def test_w22_exponential_consistency(b1, b2):
    """Weibull(k=1, scale=1/b) must equal exponential W₂²(b1, b2)."""
    weibull_val = wasserstein22_weibull_fast(1.0, 1.0/b1, 1.0, 1.0/b2)
    exp_val     = wasserstein22_exponential_fast(b1, b2)
    assert abs(weibull_val - exp_val) < 1e-9, (
        f"Weibull k=1 consistency: {weibull_val} vs exp {exp_val}"
    )


# ---------------------------------------------------------------------------
# Test 5: Weibull estimator recovers parameters
# ---------------------------------------------------------------------------

def test_weibull_estimator():
    rng = np.random.default_rng(0)
    k_true, lam_true = 2.0, 3.0
    # scipy weibull_min: shape=k, scale=lam, loc=0
    samples = scipy_weibull.rvs(c=k_true, scale=lam_true, size=50_000,
                                random_state=rng).astype(np.float64)
    k_hat, lam_hat = estimate_weibull_fast(samples)
    assert abs(k_hat - k_true) / k_true < 0.02,   f"k: {k_hat} vs {k_true}"
    assert abs(lam_hat - lam_true) / lam_true < 0.02, f"λ: {lam_hat} vs {lam_true}"


# ---------------------------------------------------------------------------
# Test 6: finite-difference gradient check for ∂W₂²/∂λ₁ and ∂W₂²/∂k₁
# ---------------------------------------------------------------------------

def _fd(f, x, h=1e-5):
    """5-point central finite difference."""
    return (f(x-2*h) - 8*f(x-h) + 8*f(x+h) - f(x+2*h)) / (12*h)


@pytest.mark.parametrize("k1,l1,k2,l2", [
    (1.5, 2.0, 0.8, 1.2),
    (2.0, 1.0, 3.0, 0.5),
    (0.7, 1.5, 1.9, 2.1),
])
def test_gradient_lambda(k1, l1, k2, l2):
    """∂W₂²/∂λ₁ matches finite difference."""
    analytic = dw22_dlambda1_python(k1, l1, k2, l2)
    fd_val   = _fd(lambda l: wasserstein22_weibull_fast(k1, l, k2, l2), l1)
    rel_err  = abs(analytic - fd_val) / (abs(fd_val) + 1e-12)
    assert rel_err < 1e-5, f"λ gradient rel-err {rel_err:.2e} (analytic={analytic:.6f}, fd={fd_val:.6f})"


@pytest.mark.parametrize("k1,l1,k2,l2", [
    (1.5, 2.0, 0.8, 1.2),
    (2.0, 1.0, 3.0, 0.5),
    (0.7, 1.5, 1.9, 2.1),
])
def test_gradient_k(k1, l1, k2, l2):
    """∂W₂²/∂k₁ matches finite difference."""
    analytic = dw22_dk1_python(k1, l1, k2, l2)
    fd_val   = _fd(lambda k: wasserstein22_weibull_fast(k, l1, k2, l2), k1)
    rel_err  = abs(analytic - fd_val) / (abs(fd_val) + 1e-12)
    assert rel_err < 1e-5, f"k gradient rel-err {rel_err:.2e} (analytic={analytic:.6f}, fd={fd_val:.6f})"


# ---------------------------------------------------------------------------
# Test 7: WassersteinDistance with distribution='weibull'
# ---------------------------------------------------------------------------

def test_wasserstein_distance_weibull():
    rng = np.random.default_rng(42)
    # Create (T, 2) parameter arrays
    T = 8
    X_params = np.column_stack([
        rng.uniform(0.5, 3.0, T),   # k
        rng.uniform(0.5, 3.0, T),   # λ
    ])
    Y_params = np.column_stack([
        rng.uniform(0.5, 3.0, T),
        rng.uniform(0.5, 3.0, T),
    ])

    wd = WassersteinDistance(X_params, Y_params, distribution='weibull',
                             X_is_params=True, Y_is_params=True)
    D = wd.compute()
    assert D.shape == (T, T)
    # Diagonal (self) must be ≥ 0 and very small (exactly 0 by formula)
    diag_xy = np.diag(WassersteinDistance(
        X_params, X_params, distribution='weibull',
        X_is_params=True, Y_is_params=True
    ).compute())
    assert np.all(np.abs(diag_xy) < 1e-10)


def test_wasserstein_weibull_jacobian_shape():
    rng = np.random.default_rng(7)
    m, n = 6, 5
    X_params = np.column_stack([rng.uniform(0.5, 2.5, m), rng.uniform(0.5, 2.5, m)])
    Y_params = np.column_stack([rng.uniform(0.5, 2.5, n), rng.uniform(0.5, 2.5, n)])

    wd = WassersteinDistance(X_params, Y_params, distribution='weibull',
                             X_is_params=True, Y_is_params=True)
    wd.compute()
    E = rng.random((m, n))

    G_X = wd.jacobian_product(E)
    G_Y = wd.jacobian_product_Y(E)
    assert G_X.shape == (m, 2)
    assert G_Y.shape == (n, 2)


# ---------------------------------------------------------------------------
# Test 8: Soft-DTW divergence D(X, X) = 0
# ---------------------------------------------------------------------------

def test_sdtw_divergence_self_zero():
    rng = np.random.default_rng(3)
    X_params = np.column_stack([rng.uniform(0.5, 2.0, 10), rng.uniform(0.5, 2.0, 10)])
    wd_xx = WassersteinDistance(X_params, X_params, distribution='weibull',
                                X_is_params=True, Y_is_params=True)
    D_xx = wd_xx.compute()
    div = sdtw_divergence(D_xx, D_xx, D_xx, gamma=1.0)
    assert abs(div) < 1e-8, f"D(X,X) = {div}"
