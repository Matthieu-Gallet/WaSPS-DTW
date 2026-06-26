"""Tests for distributions.py (merged estimation + distributions)."""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp

from distributions import (
    exponential, weibull, get,
    fit_exponential_log_cumulant, fit_exponential_mle,
    fit_weibull_log_cumulant, fit_weibull_mle,
    fit,
)
from scipy.stats import expon, weibull_min


# ---------------------------------------------------------------------------
# PDF correctness vs scipy (spine tests)
# ---------------------------------------------------------------------------

class TestExponentialPDF:
    """Verify pdf matches scipy.stats.expon(scale=1/β) exactly."""

    def test_pdf_matches_scipy(self):
        beta = 2.5
        params = jnp.array([beta])
        xs = np.linspace(0.1, 3.0, 20)
        for x in xs:
            jax_val = float(exponential.pdf(jnp.array(x), params))
            scipy_val = expon.pdf(x, scale=1.0 / beta)
            assert abs(jax_val - scipy_val) < 1e-12, (
                f"pdf mismatch at x={x}: JAX={jax_val}, scipy={scipy_val}"
            )

    def test_pdf_positive(self):
        params = jnp.array([2.0])  # β=2
        assert exponential.pdf(jnp.array(1.0), params) > 0

    def test_pdf_integrates_to_one(self):
        """Rough numerical integration should be ≈1."""
        beta = 3.0
        params = jnp.array([beta])
        xs = jnp.linspace(1e-4, 5.0, 5000)
        dx = float(xs[1] - xs[0])
        integral = float(jnp.sum(jax.vmap(lambda x: exponential.pdf(x, params))(xs)) * dx)
        assert abs(integral - 1.0) < 0.01

    def test_n_params(self):
        assert exponential.n_params == 1


class TestWeibullPDF:
    """Verify pdf matches scipy.stats.weibull_min(c=k, scale=λ, loc=0) exactly."""

    def test_pdf_matches_scipy(self):
        k, lam = 1.8, 2.5
        params = jnp.array([k, lam])
        xs = np.linspace(0.1, 6.0, 20)
        for x in xs:
            jax_val = float(weibull.pdf(jnp.array(x), params))
            scipy_val = weibull_min.pdf(x, c=k, scale=lam, loc=0)
            assert abs(jax_val - scipy_val) < 1e-10, (
                f"pdf mismatch at x={x}: JAX={jax_val}, scipy={scipy_val}"
            )

    def test_pdf_positive(self):
        params = jnp.array([1.5, 2.0])  # k=1.5, λ=2
        assert weibull.pdf(jnp.array(1.0), params) > 0

    def test_weibull_k1_matches_exponential(self):
        """Weibull(k=1, λ=1/β) ≡ Exponential(β) — pdfs should match."""
        beta = 2.0
        exp_params = jnp.array([beta])
        weib_params = jnp.array([1.0, 1.0 / beta])
        xs = jnp.linspace(0.1, 3.0, 50)
        for x in xs:
            x_arr = jnp.array(x)
            p_exp = float(exponential.pdf(x_arr, exp_params))
            p_weib = float(weibull.pdf(x_arr, weib_params))
            assert abs(p_exp - p_weib) < 1e-10

    def test_n_params(self):
        assert weibull.n_params == 2

    def test_vmap_pdf(self):
        params = jnp.array([1.5, 2.0])
        xs = jnp.linspace(0.1, 5.0, 10)
        vals = jax.vmap(lambda x: weibull.pdf(x, params))(xs)
        assert vals.shape == (10,)
        assert jnp.all(vals > 0)


class TestGetHelper:
    def test_exponential(self):
        d = get('exponential')
        assert d.n_params == 1

    def test_weibull(self):
        d = get('weibull')
        assert d.n_params == 2

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get('gamma')


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

class TestExponentialEstimators:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.beta_true = 3.0
        self.samples = rng.exponential(scale=1.0 / self.beta_true, size=50_000)

    def test_log_cumulant_recovery(self):
        beta_hat = fit_exponential_log_cumulant(self.samples)
        assert abs(beta_hat - self.beta_true) / self.beta_true < 0.02

    def test_mle_recovery(self):
        beta_hat = fit_exponential_mle(self.samples)
        assert abs(beta_hat - self.beta_true) / self.beta_true < 0.02

    def test_mle_uses_scipy(self):
        """scipy expon.fit(floc=0) must give the same result as 1/mean."""
        x = self.samples
        from scipy.stats import expon as scipy_expon
        _, scale = scipy_expon.fit(x, floc=0)
        expected = 1.0 / scale
        got = fit_exponential_mle(x)
        assert abs(got - expected) < 1e-12


class TestWeibullEstimators:
    def setup_method(self):
        rng = np.random.default_rng(0)
        self.k_true, self.lam_true = 1.8, 2.5
        self.samples = rng.weibull(self.k_true, size=50_000) * self.lam_true

    def test_log_cumulant_recovery(self):
        k_hat, lam_hat = fit_weibull_log_cumulant(self.samples)
        assert abs(k_hat - self.k_true) / self.k_true < 0.02
        assert abs(lam_hat - self.lam_true) / self.lam_true < 0.02

    def test_mle_recovery(self):
        k_hat, lam_hat = fit_weibull_mle(self.samples)
        assert abs(k_hat - self.k_true) / self.k_true < 0.02
        assert abs(lam_hat - self.lam_true) / self.lam_true < 0.02

    def test_mle_params_match_cost_parameterisation(self):
        """Scipy weibull_min.fit(floc=0) returns (c=k, loc=0, scale=λ).
        The (k, λ) returned by fit_weibull_mle must match the parameterisation
        in costs.w2sq_weibull (k=shape, λ=scale)."""
        from scipy.stats import weibull_min as wm
        x = self.samples
        k_scipy, _, lam_scipy = wm.fit(x, floc=0)
        k_hat, lam_hat = fit_weibull_mle(x)
        assert abs(k_hat - k_scipy) < 1e-12
        assert abs(lam_hat - lam_scipy) < 1e-12

    def test_log_cumulant_matches_cython_formula(self):
        x = self.samples
        logs = np.log(x)
        mu_log = logs.mean()
        sigma_log = np.sqrt(logs.var())
        PI_OVER_SQRT6 = np.pi / np.sqrt(6.0)
        DIGAMMA_1 = -0.5772156649015329
        k_ref = PI_OVER_SQRT6 / sigma_log
        lam_ref = np.exp(mu_log - DIGAMMA_1 / k_ref)
        k_hat, lam_hat = fit_weibull_log_cumulant(x)
        assert abs(k_hat - k_ref) < 1e-12
        assert abs(lam_hat - lam_ref) < 1e-12


# ---------------------------------------------------------------------------
# Time-series fit()
# ---------------------------------------------------------------------------

class TestFitTimeSeries:
    def test_default_method_is_mle(self):
        """fit() default method must be 'mle' (not 'log_cumulant')."""
        import inspect
        sig = inspect.signature(fit)
        assert sig.parameters['method'].default == 'mle'

    def test_exponential_output_shape(self):
        rng = np.random.default_rng(1)
        series = [rng.exponential(scale=0.5, size=30) for _ in range(5)]
        params = fit(series, 'exponential')
        assert params.shape == (5, 1)
        assert np.all(np.isfinite(params))

    def test_weibull_output_shape(self):
        rng = np.random.default_rng(2)
        series = [rng.weibull(1.5, size=30) * 2.0 for _ in range(7)]
        params = fit(series, 'weibull')
        assert params.shape == (7, 2)
        assert np.all(np.isfinite(params))

    def test_nan_timestep_propagated(self):
        """Timestep with < min_valid clean samples → NaN, no silent imputation."""
        rng = np.random.default_rng(3)
        series = [rng.exponential(size=30) for _ in range(4)]
        series[1] = np.array([np.nan] * 3)  # only 3 values → < min_valid=5
        params = fit(series, 'exponential')
        # Valid timesteps → finite
        assert np.all(np.isfinite(params[0]))
        assert np.all(np.isfinite(params[2]))
        assert np.all(np.isfinite(params[3]))
        # Insufficient timestep → NaN
        assert np.all(np.isnan(params[1]))

    def test_dtype_float64_default(self):
        rng = np.random.default_rng(4)
        series = [rng.exponential(size=30) for _ in range(3)]
        params = fit(series, 'exponential')
        assert params.dtype == np.float64

    def test_dtype_option_exponential(self):
        rng = np.random.default_rng(5)
        series = [rng.exponential(size=30) for _ in range(3)]
        params = fit(series, 'exponential', dtype=np.float32)
        assert params.dtype == np.float32


    def test_unknown_family_raises(self):
        with pytest.raises(ValueError):
            fit([[1.0, 2.0]], 'gamma')


# ---------------------------------------------------------------------------
# estimate() method — must match standalone functions (merged API)
# ---------------------------------------------------------------------------

class TestEstimateMethod:
    """Distribution.estimate() must match standalone fit_* functions."""

    def test_exponential_mle_matches_standalone(self):
        rng = np.random.default_rng(100)
        x = rng.exponential(1 / 3.0, size=1000)
        b_standalone = fit_exponential_mle(x)
        b_method = exponential.estimate(x, method='mle')
        assert abs(b_standalone - b_method) < 1e-12

    def test_exponential_log_cumulant_matches_standalone(self):
        rng = np.random.default_rng(101)
        x = rng.exponential(1 / 2.5, size=1000)
        b_standalone = fit_exponential_log_cumulant(x)
        b_method = exponential.estimate(x, method='log_cumulant')
        assert abs(b_standalone - b_method) < 1e-12

    def test_weibull_mle_matches_standalone(self):
        rng = np.random.default_rng(102)
        x = rng.weibull(1.5, size=1000) * 2.0
        k1, l1 = fit_weibull_mle(x)
        k2, l2 = weibull.estimate(x, method='mle')
        assert abs(k1 - k2) < 1e-12
        assert abs(l1 - l2) < 1e-12

    def test_weibull_log_cumulant_matches_standalone(self):
        rng = np.random.default_rng(103)
        x = rng.weibull(1.8, size=1000) * 2.5
        k1, l1 = fit_weibull_log_cumulant(x)
        k2, l2 = weibull.estimate(x, method='log_cumulant')
        assert abs(k1 - k2) < 1e-12
        assert abs(l1 - l2) < 1e-12


# ---------------------------------------------------------------------------
# fit_time_series dtype — float32 opt-in, float64 default
# ---------------------------------------------------------------------------

class TestDtypeFitTimeSeries:
    def test_exponential_float64_default(self):
        rng = np.random.default_rng(200)
        series = [rng.exponential(size=30) for _ in range(3)]
        params = exponential.fit_time_series(series)
        assert params.dtype == np.float64

    def test_exponential_float32(self):
        rng = np.random.default_rng(201)
        series = [rng.exponential(size=30) for _ in range(3)]
        params = exponential.fit_time_series(series, dtype=np.float32)
        assert params.dtype == np.float32

    def test_weibull_float64_default(self):
        rng = np.random.default_rng(202)
        series = [rng.weibull(1.5, size=30) * 2.0 for _ in range(3)]
        params = weibull.fit_time_series(series)
        assert params.dtype == np.float64

    def test_weibull_warns_float32(self):
        import warnings
        rng = np.random.default_rng(203)
        series = [rng.weibull(1.5, size=30) * 2.0 for _ in range(3)]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            weibull.fit_time_series(series, dtype=np.float32)
        assert any(issubclass(warning.category, UserWarning) for warning in w)

    def test_fit_wrapper_consistent(self):
        """fit() thin wrapper must give same result as fit_time_series."""
        rng = np.random.default_rng(204)
        series = [rng.exponential(size=30) for _ in range(4)]
        p1 = fit(series, 'exponential')
        p2 = exponential.fit_time_series(series)
        np.testing.assert_array_equal(p1, p2)
