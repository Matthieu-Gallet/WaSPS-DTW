"""Tests for barycenter.py — SoftDTW-based Fréchet barycenter."""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp

from costs import WaSPS, SqEuclidean
from barycenter import fit_barycenter
from softdtw import sdtw_value, SoftDTW


# ---------------------------------------------------------------------------
# fit_barycenter — Euclidean (no positivity constraint)
# ---------------------------------------------------------------------------

class TestBarycentEuclidean:
    def _make_sdtw(self, gamma=1.0):
        return SoftDTW(SqEuclidean(), gamma, is_divergence=True, manual_grad=False)

    def test_single_series_small_gamma(self):
        rng = np.random.default_rng(1)
        s = rng.uniform(0, 1, (6, 2)).astype(np.float64)
        b = fit_barycenter([s], self._make_sdtw(gamma=1e-3), n_steps=500, lr=1e-2,
                           dtype=jnp.float64)
        assert b.shape == (6, 2)
        assert np.allclose(b, s, atol=0.05)

    def test_barycenter_of_constant_series(self):
        s = np.full((5, 1), 2.0, dtype=np.float64)
        b = fit_barycenter([s, s, s], self._make_sdtw(gamma=1.0), n_steps=300, lr=1e-2,
                           dtype=jnp.float64)
        assert np.allclose(b, s, atol=1e-3)

    def test_output_shape(self):
        rng = np.random.default_rng(3)
        series = [rng.uniform(0, 1, (8, 3)).astype(np.float64) for _ in range(5)]
        b = fit_barycenter(series, self._make_sdtw(gamma=0.5), n_steps=50,
                           dtype=jnp.float64)
        assert b.shape == (8, 3)
        assert np.all(np.isfinite(b))

    def test_loss_decreases(self):
        rng = np.random.default_rng(4)
        series = [rng.uniform(0, 2, (5, 2)).astype(np.float64) for _ in range(4)]
        init = np.mean(series, axis=0)
        cost_fn = SqEuclidean()
        gamma = 1.0

        def total_loss(b):
            b_jax = jnp.array(b, dtype=jnp.float64)
            total = 0.0
            for s in series:
                D = cost_fn.all_pairs(b_jax, jnp.array(s, dtype=jnp.float64))
                D_bb = cost_fn.all_pairs(b_jax, b_jax)
                D_ss = cost_fn.all_pairs(jnp.array(s, dtype=jnp.float64),
                                         jnp.array(s, dtype=jnp.float64))
                div = (float(sdtw_value(D, gamma))
                       - 0.5 * float(sdtw_value(D_bb, gamma))
                       - 0.5 * float(sdtw_value(D_ss, gamma)))
                total += div / len(series)
            return total

        loss_init = total_loss(init)
        sdtw = SoftDTW(cost_fn, gamma, is_divergence=True, manual_grad=False)
        b = fit_barycenter(series, sdtw, n_steps=200, lr=1e-2, init=init, dtype=jnp.float64)
        loss_final = total_loss(b)
        assert loss_final <= loss_init + 1e-3


# ---------------------------------------------------------------------------
# fit_barycenter — WaSPS / positivity constraint
# ---------------------------------------------------------------------------

class TestBarycentWaSPS:
    def test_positive_params_exponential(self):
        """use_positivity_constraint ensures β > 0 always."""
        rng = np.random.default_rng(5)
        series = [rng.uniform(0.5, 3.0, (6, 1)).astype(np.float64) for _ in range(4)]
        cost_fn = WaSPS('exponential', log_correction=True, use_positivity_constraint=True)
        sdtw = SoftDTW(cost_fn, gamma=1.0, is_divergence=True, manual_grad=True)
        b = fit_barycenter(series, sdtw, n_steps=100, dtype=jnp.float64)
        assert b.shape == (6, 1)
        assert np.all(b > 0)
        assert np.all(np.isfinite(b))

    def test_positive_params_weibull(self):
        rng = np.random.default_rng(6)
        series = [rng.uniform(0.5, 3.0, (5, 2)).astype(np.float64) for _ in range(3)]
        cost_fn = WaSPS('weibull', log_correction=True, use_positivity_constraint=True)
        sdtw = SoftDTW(cost_fn, gamma=1.0, is_divergence=True, manual_grad=True)
        b = fit_barycenter(series, sdtw, n_steps=50, dtype=jnp.float64)
        assert b.shape == (5, 2)
        assert np.all(b > 0)
        assert np.all(np.isfinite(b))

    def test_loss_decreases_weibull(self):
        rng = np.random.default_rng(7)
        series = [rng.uniform(0.8, 2.5, (4, 2)).astype(np.float64) for _ in range(3)]
        init = np.mean(series, axis=0)
        cost_fn = WaSPS('weibull', log_correction=True, use_positivity_constraint=True)
        gamma = 1.0

        def total_loss(b):
            b_jax = jnp.array(b, dtype=jnp.float64)
            total = 0.0
            for s in series:
                D = cost_fn.all_pairs(b_jax, jnp.array(s, dtype=jnp.float64))
                D_bb = cost_fn.all_pairs(b_jax, b_jax)
                D_ss = cost_fn.all_pairs(jnp.array(s, dtype=jnp.float64),
                                         jnp.array(s, dtype=jnp.float64))
                div = (float(sdtw_value(D, gamma))
                       - 0.5 * float(sdtw_value(D_bb, gamma))
                       - 0.5 * float(sdtw_value(D_ss, gamma)))
                total += div / len(series)
            return total

        loss_init = total_loss(init)
        sdtw = SoftDTW(cost_fn, gamma, is_divergence=True, manual_grad=True)
        b = fit_barycenter(series, sdtw, n_steps=200, lr=5e-3, init=init, dtype=jnp.float64)
        loss_final = total_loss(b)
        assert loss_final <= loss_init + 1e-2

    def test_single_series_exponential_convergence(self):
        """For small gamma, barycenter of one series should converge to it."""
        rng = np.random.default_rng(8)
        s = rng.uniform(0.8, 2.5, (5, 1)).astype(np.float64)
        cost_fn = WaSPS('exponential', log_correction=True, use_positivity_constraint=True)
        sdtw = SoftDTW(cost_fn, gamma=1e-3, is_divergence=True, manual_grad=True)
        b = fit_barycenter([s], sdtw, n_steps=500, lr=5e-3, dtype=jnp.float64)
        assert b.shape == (5, 1)
        assert np.allclose(b, s, atol=0.20)


# ---------------------------------------------------------------------------
# Spine: manual_grad == autodiff_grad (core correctness check)
# ---------------------------------------------------------------------------

class TestManualGradEqualsAutodiff:
    """manual_grad path must produce same result as full autodiff."""

    def _run_barycenter(self, series, cost_fn, manual, gamma=1.0):
        rng_init = np.mean([np.array(s) for s in series], axis=0)
        sdtw = SoftDTW(cost_fn, gamma, is_divergence=True, manual_grad=manual)
        # Fine-tolerance test: use float64 for precision (conftest enables x64)
        return fit_barycenter(series, sdtw, n_steps=30, lr=1e-2, init=rng_init,
                              dtype=jnp.float64)

    def test_exponential_manual_vs_autodiff(self):
        rng = np.random.default_rng(400)
        series = [rng.uniform(0.5, 2.0, (4, 1)).astype(np.float64) for _ in range(3)]
        cost_fn = WaSPS('exponential', log_correction=True, use_positivity_constraint=True)
        b_manual = self._run_barycenter(series, cost_fn, manual=True)
        b_auto = self._run_barycenter(series, cost_fn, manual=False)
        np.testing.assert_allclose(b_manual, b_auto, rtol=1e-4, atol=1e-5)

    def test_weibull_manual_vs_autodiff(self):
        rng = np.random.default_rng(401)
        series = [rng.uniform(0.8, 2.0, (4, 2)).astype(np.float64) for _ in range(3)]
        cost_fn = WaSPS('weibull', log_correction=True, use_positivity_constraint=True)
        b_manual = self._run_barycenter(series, cost_fn, manual=True)
        b_auto = self._run_barycenter(series, cost_fn, manual=False)
        np.testing.assert_allclose(b_manual, b_auto, rtol=1e-3, atol=1e-4)


# ---------------------------------------------------------------------------
# API contract: no legacy parameters
# ---------------------------------------------------------------------------

def test_no_weights_parameter():
    """fit_barycenter signature must NOT have a weights parameter."""
    import inspect
    sig = inspect.signature(fit_barycenter)
    assert 'weights' not in sig.parameters


def test_no_softplus_parameter():
    """fit_barycenter signature must NOT have a softplus parameter (moved to SoftDTW)."""
    import inspect
    sig = inspect.signature(fit_barycenter)
    assert 'softplus' not in sig.parameters


def test_dtype_param_exists():
    """fit_barycenter must expose dtype parameter (default float64)."""
    import inspect
    sig = inspect.signature(fit_barycenter)
    assert 'dtype' in sig.parameters
    assert sig.parameters['dtype'].default == jnp.float64


def test_no_manual_grad_parameter():
    """fit_barycenter signature must NOT have a manual_grad parameter (moved to SoftDTW)."""
    import inspect
    sig = inspect.signature(fit_barycenter)
    assert 'manual_grad' not in sig.parameters


# ---------------------------------------------------------------------------
# Softplus chain rule — gradient correctness (spine)
# ---------------------------------------------------------------------------

class TestSoftplusGradient:
    """use_positivity_constraint=True: θ unconstrained, cost applies softplus."""

    def test_softplus_is_positive(self):
        z = jnp.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        theta = jax.nn.softplus(z)
        assert jnp.all(theta > 0)

    def test_softplus_chain_rule(self):
        """Autodiff of f(cost_with_constraint(θ)) must match gradient_X(E, θ, y) output."""
        rng = np.random.default_rng(300)
        z = jnp.array(rng.uniform(-1.0, 2.0, (5, 1)), dtype=jnp.float64)
        cost_fn = WaSPS('exponential', log_correction=False, use_positivity_constraint=True)
        x = jnp.array(rng.uniform(-1.0, 2.0, (5, 1)), dtype=jnp.float64)

        def f_auto(z_):
            return jnp.sum(cost_fn.all_pairs(z_, x))

        E = jnp.ones((5, 5), dtype=jnp.float64)
        g_manual = cost_fn.gradient_X(E, z, x)
        g_auto = jax.grad(f_auto)(z)
        np.testing.assert_allclose(np.array(g_auto), np.array(g_manual),
                                   rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------------------
# New: SqEuclidean positivity constraint in barycenter
# ---------------------------------------------------------------------------

class TestBarycentSqEuclideanConstrained:
    """SqEuclidean(use_positivity_constraint=True) must give strictly positive barycenters."""

    def test_constrained_barycenter_positive(self):
        rng = np.random.default_rng(500)
        series = [rng.uniform(0.5, 3.0, (6, 2)).astype(np.float64) for _ in range(4)]
        cost_fn = SqEuclidean(use_positivity_constraint=True)
        sdtw = SoftDTW(cost_fn, gamma=1.0, is_divergence=True, manual_grad=False)
        b = fit_barycenter(series, sdtw, n_steps=100, dtype=jnp.float64)
        assert b.shape == (6, 2)
        assert np.all(b > 0), f"Expected all positive, got min={b.min():.6f}"
        assert np.all(np.isfinite(b))

    def test_unconstrained_barycenter_shape(self):
        """SqEuclidean() (unconstrained) barycenter keeps correct shape."""
        rng = np.random.default_rng(501)
        series = [rng.uniform(-1.0, 1.0, (5, 2)).astype(np.float64) for _ in range(3)]
        cost_fn = SqEuclidean()
        sdtw = SoftDTW(cost_fn, gamma=1.0, is_divergence=True, manual_grad=False)
        b = fit_barycenter(series, sdtw, n_steps=50, dtype=jnp.float64)
        assert b.shape == (5, 2)
        assert np.all(np.isfinite(b))
