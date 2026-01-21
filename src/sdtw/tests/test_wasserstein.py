"""
Basic tests for Wasserstein distance functionality.
This file contains a subset of tests for quick validation.
For comprehensive testing, see test_estimation.py and test_wasserstein_distance.py
"""
import numpy as np
import pytest
from scipy.stats import expon, rayleigh

import sys
sys.path.insert(0, '/home/mgallet/Documents/Papiers/EN COURS/DTWOT_2026/soft-dtw')

from sdtw.distance import WassersteinDistance
from sdtw.estimations import estimate_exponential, DistributionEstimator
from sdtw.wasserstein_distances import wasserstein2_exponential


class TestBasicFunctionality:
    """Basic functionality tests."""

    def test_exponential_estimation_basic(self):
        """Basic test for exponential parameter estimation."""
        lambda_true = 2.0
        np.random.seed(42)
        data = expon.rvs(scale=1/lambda_true, size=5000)

        lambda_est = estimate_exponential(data)
        assert abs(lambda_est - lambda_true) / lambda_true < 0.1

    def test_wasserstein_distance_basic(self):
        """Basic test for WassersteinDistance class."""
        np.random.seed(42)
        lambda_X = [1.0, 3.0]
        lambda_Y = [2.0, 4.0]
        n_samples = 2000

        X = np.array([expon.rvs(scale=1/lam, size=n_samples) for lam in lambda_X])
        Y = np.array([expon.rvs(scale=1/lam, size=n_samples) for lam in lambda_Y])

        wass = WassersteinDistance(X, Y, distribution='exponential', order=2)
        D = wass.compute()

        assert D.shape == (2, 2)
        assert np.all(D >= 0)
        assert np.all(np.isfinite(D))

    def test_wasserstein_analytical_basic(self):
        """Basic test for analytical Wasserstein formula."""
        lambda1, lambda2 = 1.0, 2.0
        w2 = wasserstein2_exponential(lambda1, lambda2)
        expected = np.sqrt(2) * 0.5
        assert abs(w2 - expected) < 1e-10

    def test_distribution_estimator_basic(self):
        """Basic test for DistributionEstimator class."""
        lambda_true = 3.0
        np.random.seed(42)
        data = expon.rvs(scale=1/lambda_true, size=3000)

        estimator = DistributionEstimator('exponential')
        lambda_est = estimator.fit(data)

        assert abs(lambda_est - lambda_true) / lambda_true < 0.15
        assert estimator.distribution == 'exponential'
        
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
    
    def test_estimator_class(self):
        """Test DistributionEstimator class."""
        # Generate exponential data
        lambda_true = 3.0
        np.random.seed(42)
        data = expon.rvs(scale=1/lambda_true, size=5000)
        
        # Use estimator class
        estimator = DistributionEstimator('exponential')
        lambda_est = estimator.fit(data)
        
        assert abs(lambda_est - lambda_true) / lambda_true < 0.1
        assert estimator.distribution == 'exponential'


class TestWassersteinDistances:
    """Test analytical Wasserstein distance formulas."""
    
    def test_exponential_w1(self):
        """Test W1 distance for exponential distributions."""
        lambda1, lambda2 = 1.0, 2.0
        
        # Compute distance
        w1 = wasserstein1_exponential(lambda1, lambda2)
        
        # Expected value: |λ1 - λ2| / (λ1 * λ2) = |1 - 2| / (1 * 2) = 0.5
        expected = 0.5
        assert abs(w1 - expected) < 1e-10
    
    def test_exponential_w2(self):
        """Test W2 distance for exponential distributions."""
        lambda1, lambda2 = 1.0, 2.0
        
        # Compute distance
        w2 = wasserstein2_exponential(lambda1, lambda2)
        
        # Expected value: √2 * |λ1 - λ2| / (λ1 * λ2)
        expected = np.sqrt(2) * 0.5
        assert abs(w2 - expected) < 1e-10
    
    def test_rayleigh_w1(self):
        """Test W1 distance for Rayleigh distributions."""
        sigma1, sigma2 = 1.0, 2.0
        
        # Compute distance
        w1 = wasserstein1_rayleigh(sigma1, sigma2)
        
        # Expected value: √(π/2) * |σ1 - σ2|
        expected = np.sqrt(np.pi / 2) * 1.0
        assert abs(w1 - expected) < 1e-10
    
    def test_rayleigh_w2(self):
        """Test W2 distance for Rayleigh distributions."""
        sigma1, sigma2 = 1.0, 2.0
        
        # Compute distance
        w2 = wasserstein2_rayleigh(sigma1, sigma2)
        
        # Expected value: √2 * |σ1 - σ2|
        expected = np.sqrt(2) * 1.0
        assert abs(w2 - expected) < 1e-10
    
    def test_gaussian_w2(self):
        """Test W2 distance for Gaussian distributions."""
        mu1, sigma1 = 0.0, 1.0
        mu2, sigma2 = 1.0, 1.5
        
        # Compute distance
        w2 = wasserstein2_gaussian(mu1, sigma1, mu2, sigma2)
        
        # Expected value: √((μ1-μ2)² + (σ1²-σ2²))
        expected = np.sqrt((mu1 - mu2)**2 + (sigma1**2 - sigma2**2))
        assert abs(w2 - expected) < 1e-10
    
    def test_symmetry(self):
        """Test that Wasserstein distances are symmetric."""
        lambda1, lambda2 = 1.5, 3.0
        
        w1_12 = wasserstein1_exponential(lambda1, lambda2)
        w1_21 = wasserstein1_exponential(lambda2, lambda1)
        
        assert abs(w1_12 - w1_21) < 1e-10
    
    def test_identity(self):
        """Test that W(X, X) = 0."""
        lambda_val = 2.5
        
        w1 = wasserstein1_exponential(lambda_val, lambda_val)
        w2 = wasserstein2_exponential(lambda_val, lambda_val)
        
        assert abs(w1) < 1e-10
        assert abs(w2) < 1e-10


class TestWassersteinDistanceClass:
    """Test WassersteinDistance class."""
    
    def test_exponential_matrix_computation(self):
        """Test distance matrix computation for exponential distributions."""
        # Generate time series with known parameters
        np.random.seed(42)
        lambda_X = [1.0, 5.0, 10.0]
        lambda_Y = [2.0, 4.0, 8.0]
        n_samples = 1000
        
        # Generate samples
        X = np.array([expon.rvs(scale=1/lam, size=n_samples) for lam in lambda_X])
        Y = np.array([expon.rvs(scale=1/lam, size=n_samples) for lam in lambda_Y])
        
        # Compute distance matrix
        wass = WassersteinDistance(X, Y, distribution='exponential', order=2)
        D = wass.compute()
        
        # Check shape
        assert D.shape == (3, 3)
        
        # Check diagonal-like structure (similar lambdas should have small distance)
        assert D[0, 0] < D[0, 2]  # lambda_X[0]=1 closer to lambda_Y[0]=2 than to lambda_Y[2]=8
        assert D[1, 1] < D[1, 0]  # lambda_X[1]=5 closer to lambda_Y[1]=4 than to lambda_Y[0]=2
    
    def test_squared_option(self):
        """Test squared distance option."""
        np.random.seed(42)
        lambda_X = [1.0, 2.0]
        lambda_Y = [1.5, 2.5]
        n_samples = 1000
        
        X = np.array([expon.rvs(scale=1/lam, size=n_samples) for lam in lambda_X])
        Y = np.array([expon.rvs(scale=1/lam, size=n_samples) for lam in lambda_Y])
        
        # Compute with and without squaring
        wass_normal = WassersteinDistance(X, Y, distribution='exponential', squared=False)
        D_normal = wass_normal.compute()
        
        wass_squared = WassersteinDistance(X, Y, distribution='exponential', squared=True)
        D_squared = wass_squared.compute()
        
        # Check relationship
        np.testing.assert_array_almost_equal(D_squared, D_normal**2, decimal=5)
    
    def test_rayleigh_computation(self):
        """Test distance computation for Rayleigh distributions."""
        np.random.seed(42)
        sigma_X = [1.0, 2.0]
        sigma_Y = [1.5, 2.5]
        n_samples = 1000
        
        X = np.array([rayleigh.rvs(scale=sig, size=n_samples) for sig in sigma_X])
        Y = np.array([rayleigh.rvs(scale=sig, size=n_samples) for sig in sigma_Y])
        
        # Compute distance matrix
        wass = WassersteinDistance(X, Y, distribution='rayleigh', order=2)
        D = wass.compute()
        
        # Check shape
        assert D.shape == (2, 2)
        
        # Check positivity
        assert np.all(D >= 0)
    
    def test_parameter_estimation_storage(self):
        """Test that estimated parameters are stored correctly."""
        np.random.seed(42)
        lambda_X = [1.0, 5.0]
        n_samples = 5000
        
        X = np.array([expon.rvs(scale=1/lam, size=n_samples) for lam in lambda_X])
        Y = X.copy()
        
        wass = WassersteinDistance(X, Y, distribution='exponential')
        
        # Check that parameters are stored
        assert len(wass.X_params) == 2
        assert len(wass.Y_params) == 2
        
        # Check estimation accuracy
        for i, lam_true in enumerate(lambda_X):
            lam_est = wass.X_params[i]
            assert abs(lam_est - lam_true) / lam_true < 0.1
    
    def test_invalid_distribution(self):
        """Test error handling for unsupported distributions."""
        np.random.seed(42)
        X = np.random.rand(2, 100)
        Y = np.random.rand(2, 100)
        
        with pytest.raises(ValueError):
            WassersteinDistance(X, Y, distribution='unsupported_dist')
    
    def test_jacobian_not_implemented(self):
        """Test that jacobian_product raises NotImplementedError."""
        np.random.seed(42)
        X = expon.rvs(scale=1.0, size=(2, 100))
        Y = expon.rvs(scale=1.0, size=(2, 100))
        
        wass = WassersteinDistance(X, Y, distribution='exponential')
        E = np.ones((2, 2))
        
        with pytest.raises(NotImplementedError):
            wass.jacobian_product(E)


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_pipeline_exponential(self):
        """Test complete pipeline: generation -> estimation -> distance."""
        np.random.seed(42)
        
        # Define time-varying parameters
        lambda_series1 = [1.0, 5.0, 10.0, 5.0]
        lambda_series2 = [2.0, 4.0, 8.0, 6.0]
        n_samples = 2000
        
        # Generate samples
        X = np.array([expon.rvs(scale=1/lam, size=n_samples) for lam in lambda_series1])
        Y = np.array([expon.rvs(scale=1/lam, size=n_samples) for lam in lambda_series2])
        
        # Compute Wasserstein distance matrix
        wass = WassersteinDistance(X, Y, distribution='exponential', order=2)
        D = wass.compute()
        
        # Check that distance reflects parameter similarity
        # Times 1 and 1: lambda_X[1]=5 vs lambda_Y[1]=4 should be small
        # Times 0 and 2: lambda_X[0]=1 vs lambda_Y[2]=8 should be large
        assert D[1, 1] < D[0, 2]
        
        # Check matrix properties
        assert D.shape == (4, 4)
        assert np.all(D >= 0)
        assert np.all(np.isfinite(D))
    
    def test_comparison_with_euclidean(self):
        """Compare Wasserstein distances with Euclidean distances on parameters."""
        np.random.seed(42)
        
        lambda_X = [1.0, 10.0]
        lambda_Y = [2.0, 9.0]
        n_samples = 5000
        
        X = np.array([expon.rvs(scale=1/lam, size=n_samples) for lam in lambda_X])
        Y = np.array([expon.rvs(scale=1/lam, size=n_samples) for lam in lambda_Y])
        
        # Wasserstein distances
        wass = WassersteinDistance(X, Y, distribution='exponential', order=2)
        D_wass = wass.compute()
        
        # The structure should reflect that (1,2) and (10,9) are similar pairs
        # while (1,9) and (10,2) are dissimilar
        assert D_wass[0, 0] < D_wass[0, 1]
        assert D_wass[1, 1] < D_wass[1, 0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
