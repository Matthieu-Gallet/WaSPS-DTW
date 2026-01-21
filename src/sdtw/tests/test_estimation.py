"""
Tests for distribution parameter estimation functions.
"""
import numpy as np
import pytest
from scipy.stats import expon, rayleigh, norm, weibull_min

import sys
sys.path.insert(0, '/home/mgallet/Documents/Papiers/EN COURS/DTWOT_2026/soft-dtw')

from sdtw.estimations import (
    estimate_exponential,
    estimate_rayleigh,
    estimate_lognormal,
    estimate_weibull,
    estimate_gamma,
    estimate_fisher,
    estimate_gengamma,
    DistributionEstimator
)


class TestEstimationFunctions:
    """Test individual parameter estimation functions."""

    def test_exponential_estimation(self):
        """Test exponential parameter estimation."""
        # Generate data with known parameter
        lambda_true = 2.0
        np.random.seed(42)
        data = expon.rvs(scale=1/lambda_true, size=10000)

        # Estimate parameter
        lambda_est = estimate_exponential(data)

        # Check estimation accuracy (within 5%)
        assert abs(lambda_est - lambda_true) / lambda_true < 0.05

    def test_rayleigh_estimation(self):
        """Test Rayleigh parameter estimation."""
        # Generate data with known parameter
        sigma_true = 1.5
        np.random.seed(42)
        data = rayleigh.rvs(scale=sigma_true, size=10000)

        # Estimate parameter
        sigma_est = estimate_rayleigh(data)

        # Check estimation accuracy (within 5%)
        assert abs(sigma_est - sigma_true) / sigma_true < 0.05

    def test_lognormal_estimation(self):
        """Test log-normal parameter estimation."""
        # Generate data with known parameters
        mu_true, sigma_true = 0.5, 0.3
        np.random.seed(42)
        data = np.random.lognormal(mu_true, sigma_true, size=10000)

        # Estimate parameters
        mu_est, sigma_est = estimate_lognormal(data)

        # Check estimation accuracy
        assert abs(mu_est - mu_true) < 0.05
        assert abs(sigma_est - sigma_true) / sigma_true < 0.1

    def test_weibull_estimation(self):
        """Test Weibull parameter estimation."""
        # Generate data with known parameters
        k_true, lambda_true = 2.0, 1.5
        np.random.seed(42)
        data = weibull_min.rvs(k_true, scale=lambda_true, size=10000)

        # Estimate parameters
        k_est, lambda_est = estimate_weibull(data)

        # Check estimation accuracy
        assert abs(k_est - k_true) / k_true < 0.1
        assert abs(lambda_est - lambda_true) / lambda_true < 0.1

    def test_gamma_estimation(self):
        """Test Gamma parameter estimation."""
        # Generate data with known parameters
        alpha_true, beta_true = 2.0, 1.5
        np.random.seed(42)
        data = np.random.gamma(alpha_true, 1/beta_true, size=10000)

        # Estimate parameters
        alpha_est, beta_est = estimate_gamma(data)

        # Check estimation accuracy (more lenient for log-cumulants method)
        assert abs(alpha_est - alpha_true) / alpha_true < 0.2
        assert abs(beta_est - beta_true) / beta_true < 0.2

    def test_estimation_with_small_samples(self):
        """Test estimation with smaller sample sizes."""
        lambda_true = 3.0
        np.random.seed(42)
        data = expon.rvs(scale=1/lambda_true, size=1000)

        lambda_est = estimate_exponential(data)

        # Should still be reasonably accurate
        assert abs(lambda_est - lambda_true) / lambda_true < 0.2

    def test_estimation_with_edge_cases(self):
        """Test estimation with edge case data."""
        # Test with data close to zero
        lambda_true = 10.0
        np.random.seed(42)
        data = expon.rvs(scale=1/lambda_true, size=5000)

        lambda_est = estimate_exponential(data)
        assert lambda_est > 0  # Should be positive
        assert abs(lambda_est - lambda_true) / lambda_true < 0.1


class TestDistributionEstimatorClass:
    """Test the DistributionEstimator class."""

    def test_estimator_class_exponential(self):
        """Test DistributionEstimator class with exponential distribution."""
        # Generate exponential data
        lambda_true = 3.0
        np.random.seed(42)
        data = expon.rvs(scale=1/lambda_true, size=5000)

        # Use estimator class
        estimator = DistributionEstimator('exponential')
        lambda_est = estimator.fit(data)

        assert abs(lambda_est - lambda_true) / lambda_true < 0.1
        assert estimator.distribution == 'exponential'

    def test_estimator_class_rayleigh(self):
        """Test DistributionEstimator class with Rayleigh distribution."""
        # Generate Rayleigh data
        sigma_true = 2.0
        np.random.seed(42)
        data = rayleigh.rvs(scale=sigma_true, size=5000)

        # Use estimator class
        estimator = DistributionEstimator('rayleigh')
        sigma_est = estimator.fit(data)

        assert abs(sigma_est - sigma_true) / sigma_true < 0.1
        assert estimator.distribution == 'rayleigh'

    def test_estimator_class_lognormal(self):
        """Test DistributionEstimator class with log-normal distribution."""
        # Generate log-normal data
        mu_true, sigma_true = 0.2, 0.4
        np.random.seed(42)
        data = np.random.lognormal(mu_true, sigma_true, size=5000)

        # Use estimator class
        estimator = DistributionEstimator('lognormal')
        mu_est, sigma_est = estimator.fit(data)

        assert abs(mu_est - mu_true) < 0.1
        assert abs(sigma_est - sigma_true) / sigma_true < 0.15
        assert estimator.distribution == 'lognormal'

    def test_estimator_invalid_distribution(self):
        """Test error handling for unsupported distributions."""
        with pytest.raises(ValueError):
            estimator = DistributionEstimator('unsupported_dist')

    def test_estimator_call_method(self):
        """Test that __call__ method works like fit."""
        lambda_true = 4.0
        np.random.seed(42)
        data = expon.rvs(scale=1/lambda_true, size=3000)

        estimator = DistributionEstimator('exponential')

        # Test both methods
        params1 = estimator.fit(data)
        params2 = estimator(data)

        assert params1 == params2
        assert abs(params1 - lambda_true) / lambda_true < 0.15


class TestDataValidation:
    """Test data validation in estimation functions."""

    def test_data_with_zeros(self):
        """Test handling of data containing zeros."""
        data = np.array([0.0, 1.0, 2.0, 3.0])
        # Should add small epsilon and work
        lambda_est = estimate_exponential(data)
        assert lambda_est > 0

    def test_data_with_negatives(self):
        """Test handling of negative data."""
        data = np.array([-1.0, 1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="negative values"):
            estimate_exponential(data)

    def test_data_all_zeros(self):
        """Test handling of data with only zeros."""
        data = np.array([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="only zeros"):
            estimate_exponential(data)

    def test_data_with_nan(self):
        """Test handling of data containing NaN."""
        data = np.array([1.0, np.nan, 3.0])
        with pytest.raises(ValueError, match="NaN"):
            estimate_exponential(data)

    def test_data_with_inf(self):
        """Test handling of data containing infinity."""
        data = np.array([1.0, np.inf, 3.0])
        with pytest.raises(ValueError, match="Inf"):
            estimate_exponential(data)

    def test_data_wrong_dimension(self):
        """Test handling of multi-dimensional data."""
        data = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="1D"):
            estimate_exponential(data)


class TestEstimationAccuracy:
    """Test estimation accuracy across different parameter ranges."""

    @pytest.mark.parametrize("lambda_param", [0.1, 1.0, 5.0, 10.0, 50.0])
    def test_exponential_accuracy_range(self, lambda_param):
        """Test exponential estimation accuracy across parameter range."""
        np.random.seed(42)
        data = expon.rvs(scale=1/lambda_param, size=5000)

        lambda_est = estimate_exponential(data)

        relative_error = abs(lambda_est - lambda_param) / lambda_param
        assert relative_error < 0.1  # Within 10% for various ranges

    @pytest.mark.parametrize("sigma_param", [0.5, 1.0, 2.0, 5.0])
    def test_rayleigh_accuracy_range(self, sigma_param):
        """Test Rayleigh estimation accuracy across parameter range."""
        np.random.seed(42)
        data = rayleigh.rvs(scale=sigma_param, size=5000)

        sigma_est = estimate_rayleigh(data)

        relative_error = abs(sigma_est - sigma_param) / sigma_param
        assert relative_error < 0.1  # Within 10% for various ranges


class TestEstimationConsistency:
    """Test consistency of estimation results."""

    def test_exponential_consistency(self):
        """Test that estimation is consistent across multiple runs."""
        lambda_true = 2.5
        results = []

        for seed in [42, 123, 456]:
            np.random.seed(seed)
            data = expon.rvs(scale=1/lambda_true, size=3000)
            lambda_est = estimate_exponential(data)
            results.append(lambda_est)

        # All estimates should be reasonably close to true value
        for est in results:
            assert abs(est - lambda_true) / lambda_true < 0.15

    def test_rayleigh_consistency(self):
        """Test that Rayleigh estimation is consistent."""
        sigma_true = 1.8
        results = []

        for seed in [42, 123, 456]:
            np.random.seed(seed)
            data = rayleigh.rvs(scale=sigma_true, size=3000)
            sigma_est = estimate_rayleigh(data)
            results.append(sigma_est)

        # All estimates should be reasonably close to true value
        for est in results:
            assert abs(est - sigma_true) / sigma_true < 0.15