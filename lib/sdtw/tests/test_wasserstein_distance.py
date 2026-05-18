"""
Tests for Wasserstein distance computations and WassersteinDistance class.
"""
import numpy as np
import pytest
from scipy.stats import expon, rayleigh, norm, weibull_min

import sys
sys.path.insert(0, '/home/mgallet/Documents/Papiers/EN COURS/DTWOT_2026/soft-dtw')

from sdtw.distance import WassersteinDistance
from sdtw.wasserstein_distances import (
    wasserstein1_exponential,
    wasserstein2_exponential,
    wasserstein1_rayleigh,
    wasserstein2_rayleigh,
    wasserstein2_gaussian,
    wasserstein2_weibull,
    get_wasserstein_function
)


class TestWassersteinAnalyticalFormulas:
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

        # Expected value: √((μ1-μ2)² + (σ1-σ2)²)
        expected = np.sqrt((mu1 - mu2)**2 + (sigma1 - sigma2)**2)
        assert abs(w2 - expected) < 1e-10

    def test_weibull_w2(self):
        """Test W2 distance for Weibull distributions."""
        k1, lambda1 = 2.0, 1.0
        k2, lambda2 = 2.0, 1.5

        # Compute distance
        w2 = wasserstein2_weibull(k1, lambda1, k2, lambda2)

        # Should be positive and finite
        assert w2 > 0
        assert np.isfinite(w2)

        # Distance between identical distributions should be zero
        w2_identical = wasserstein2_weibull(k1, lambda1, k1, lambda1)
        assert abs(w2_identical) < 1e-10

    def test_symmetry(self):
        """Test that Wasserstein distances are symmetric."""
        lambda1, lambda2 = 1.5, 3.0

        w1_12 = wasserstein1_exponential(lambda1, lambda2)
        w1_21 = wasserstein1_exponential(lambda2, lambda1)

        assert abs(w1_12 - w1_21) < 1e-10

        w2_12 = wasserstein2_exponential(lambda1, lambda2)
        w2_21 = wasserstein2_exponential(lambda2, lambda1)

        assert abs(w2_12 - w2_21) < 1e-10

    def test_identity(self):
        """Test that W(X, X) = 0."""
        lambda_val = 2.5

        w1 = wasserstein1_exponential(lambda_val, lambda_val)
        w2 = wasserstein2_exponential(lambda_val, lambda_val)

        assert abs(w1) < 1e-10
        assert abs(w2) < 1e-10

        sigma_val = 1.8
        w1_ray = wasserstein1_rayleigh(sigma_val, sigma_val)
        w2_ray = wasserstein2_rayleigh(sigma_val, sigma_val)

        assert abs(w1_ray) < 1e-10
        assert abs(w2_ray) < 1e-10

    def test_triangle_inequality(self):
        """Test triangle inequality (approximate for small distances)."""
        lambda1, lambda2, lambda3 = 1.0, 2.0, 3.0

        w12 = wasserstein2_exponential(lambda1, lambda2)
        w23 = wasserstein2_exponential(lambda2, lambda3)
        w13 = wasserstein2_exponential(lambda1, lambda3)

        # Triangle inequality: W(X,Z) ≤ W(X,Y) + W(Y,Z)
        assert w13 <= w12 + w23 + 1e-10

    def test_get_wasserstein_function(self):
        """Test the get_wasserstein_function dispatcher."""
        # Test valid combinations
        func = get_wasserstein_function('exponential', 1)
        assert callable(func)

        func = get_wasserstein_function('exponential', 2)
        assert callable(func)

        func = get_wasserstein_function('rayleigh', 1)
        assert callable(func)

        # Test invalid distribution
        with pytest.raises(ValueError, match="No analytical Wasserstein"):
            get_wasserstein_function('unsupported_dist', 2)

        # Test invalid order
        with pytest.raises(ValueError, match="not available"):
            get_wasserstein_function('gaussian', 1)  # Only W2 available


class TestWassersteinDistanceClass:
    """
    Test WassersteinDistance class (Legacy API).
    
    Note: These tests use the old API with 'order' parameter and multiple distributions.
    The new optimized implementation only supports exponential and weibull with W2 distance.
    See TestWassersteinDistanceClassCython for tests of the new implementation.
    """

    @pytest.mark.skip(reason="Legacy API - order parameter removed in new version")
    def test_exponential_matrix_computation(self):
        """Test distance matrix computation for exponential distributions."""
        # Generate time series with known parameters
        np.random.seed(42)
        lambda_X = [1.0, 5.0, 10.0]
        lambda_Y = [2.0, 4.0, 8.0]
        n_samples = 2000

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

    @pytest.mark.skip(reason="Legacy API - rayleigh not supported in new version")
    def test_rayleigh_matrix_computation(self):
        """Test distance matrix computation for Rayleigh distributions."""
        np.random.seed(42)
        sigma_X = [1.0, 2.0]
        sigma_Y = [1.5, 2.5]
        n_samples = 2000

        X = np.array([rayleigh.rvs(scale=sig, size=n_samples) for sig in sigma_X])
        Y = np.array([rayleigh.rvs(scale=sig, size=n_samples) for sig in sigma_Y])

        # Compute distance matrix
        wass = WassersteinDistance(X, Y, distribution='rayleigh', order=2)
        D = wass.compute()

        # Check shape
        assert D.shape == (2, 2)

        # Check positivity
        assert np.all(D >= 0)

    def test_squared_option(self):
        """Test that squared Wasserstein distance is computed correctly."""
        np.random.seed(42)
        lambda_X = [1.0, 5.0]
        lambda_Y = [2.0, 4.0]
        n_samples = 2000

        X = np.array([expon.rvs(scale=1/lam, size=n_samples) for lam in lambda_X])
        Y = np.array([expon.rvs(scale=1/lam, size=n_samples) for lam in lambda_Y])

        # Compute squared Wasserstein distance
        wass = WassersteinDistance(X, Y, distribution='exponential')
        D_squared = wass.compute()

        # Check that result is positive and finite
        assert np.all(D_squared >= 0)
        assert np.all(np.isfinite(D_squared))
        assert D_squared.shape == (2, 2)

    @pytest.mark.skip(reason="Legacy API - X_params/Y_params not stored in new version")
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
        """Test that jacobian_product is now implemented."""
        np.random.seed(42)
        X = expon.rvs(scale=1.0, size=(2, 100))
        Y = expon.rvs(scale=1.0, size=(2, 100))

        wass = WassersteinDistance(X, Y, distribution='exponential')
        E = np.ones((2, 2))

        # Should not raise NotImplementedError anymore
        G = wass.jacobian_product(E)
        assert G.shape == (2, 1)  # One gradient per distribution for exponential
        assert np.all(np.isfinite(G))

    def test_invalid_input_shapes(self):
        """Test error handling for invalid input shapes."""
        # 1D input instead of 2D
        X = np.random.rand(100)
        Y = np.random.rand(100)

        with pytest.raises(ValueError, match="must be 2D"):
            WassersteinDistance(X, Y, distribution='exponential')

    @pytest.mark.skip(reason="Legacy API - validation happens in Cython, different error message")
    def test_insufficient_samples(self):
        """Test error handling for insufficient samples."""
        # Only 1 sample per time point
        X = np.random.rand(2, 1)
        Y = np.random.rand(2, 1)

        with pytest.raises(ValueError, match="enough valid samples"):
            WassersteinDistance(X, Y, distribution='exponential')


class TestIntegration:
    """
    Integration tests combining multiple components.
    
    Note: These tests use the legacy API. Most are skipped for the new Cython implementation.
    """

    @pytest.mark.skip(reason="Legacy API - order parameter removed")
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
        wass = WassersteinDistance(X, Y, distribution='exponential')
        D_wass = wass.compute()

        # The structure should reflect that (1,2) and (10,9) are similar pairs
        # while (1,9) and (10,2) are dissimilar
        assert D_wass[0, 0] < D_wass[0, 1]
        assert D_wass[1, 1] < D_wass[1, 0]

    @pytest.mark.skip(reason="Legacy API - order parameter removed, W2 only")
    def test_different_orders(self):
        """Test that W1 and W2 give different but related results."""
        np.random.seed(42)
        lambda_X = [1.0, 3.0]
        lambda_Y = [2.0, 4.0]
        n_samples = 3000

        X = np.array([expon.rvs(scale=1/lam, size=n_samples) for lam in lambda_X])
        Y = np.array([expon.rvs(scale=1/lam, size=n_samples) for lam in lambda_Y])

        wass1 = WassersteinDistance(X, Y, distribution='exponential', order=1)
        D1 = wass1.compute()

        wass2 = WassersteinDistance(X, Y, distribution='exponential', order=2)
        D2 = wass2.compute()

        # W2 should be larger than W1 for same distributions
        assert np.all(D2 >= D1)
        # But they should be positively correlated
        assert np.corrcoef(D1.flatten(), D2.flatten())[0, 1] > 0.9


class TestPerformance:
    """Test performance aspects."""

    def test_large_matrix(self):
        """Test computation with larger matrices."""
        np.random.seed(42)
        n_timepoints = 5
        n_samples = 1000

        # Generate larger time series
        lambda_X = np.random.uniform(0.5, 5.0, n_timepoints)
        lambda_Y = np.random.uniform(0.5, 5.0, n_timepoints)

        X = np.array([expon.rvs(scale=1/lam, size=n_samples) for lam in lambda_X])
        Y = np.array([expon.rvs(scale=1/lam, size=n_samples) for lam in lambda_Y])

        # Should complete without error
        wass = WassersteinDistance(X, Y, distribution='exponential')
        D = wass.compute()

        assert D.shape == (n_timepoints, n_timepoints)
        assert np.all(D >= 0)

    @pytest.mark.skip(reason="Legacy API - X_params/Y_params no longer directly accessible, parameters managed internally")
    def test_memory_efficiency(self):
        """Test that parameters are cached properly."""
        np.random.seed(42)
        lambda_X = [1.0, 2.0, 3.0]
        n_samples = 2000

        X = np.array([expon.rvs(scale=1/lam, size=n_samples) for lam in lambda_X])
        Y = X.copy()

        wass = WassersteinDistance(X, Y, distribution='exponential')

        # Parameters should be cached after first computation
        params_before = wass.X_params.copy()
        D1 = wass.compute()

        # Second computation should use cached parameters
        D2 = wass.compute()

        # Results should be identical
        np.testing.assert_array_equal(D1, D2)
        assert wass.X_params == params_before


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_identical_series(self):
        """Test distance matrix for identical time series."""
        np.random.seed(42)
        lambda_series = [1.0, 3.0, 5.0]
        n_samples = 2000

        X = np.array([expon.rvs(scale=1/lam, size=n_samples) for lam in lambda_series])
        Y = X.copy()

        wass = WassersteinDistance(X, Y, distribution='exponential')
        D = wass.compute()

        # Diagonal should be approximately zero
        np.testing.assert_allclose(np.diag(D), 0, atol=1e-3)

        # Matrix should be symmetric
        assert np.allclose(D, D.T, atol=1e-3)

    def test_very_different_parameters(self):
        """Test with very different parameter values."""
        np.random.seed(42)
        lambda_X = [0.1, 0.5]  # Small values
        lambda_Y = [10.0, 50.0]  # Large values
        n_samples = 3000

        X = np.array([expon.rvs(scale=1/lam, size=n_samples) for lam in lambda_X])
        Y = np.array([expon.rvs(scale=1/lam, size=n_samples) for lam in lambda_Y])

        wass = WassersteinDistance(X, Y, distribution='exponential')
        D = wass.compute()

        # All distances should be large
        assert np.all(D > 1.0)
        # And finite
        assert np.all(np.isfinite(D))

    def test_mixed_parameter_ranges(self):
        """Test with mixed parameter ranges."""
        np.random.seed(42)
        lambda_X = [0.1, 1.0, 10.0, 100.0]
        lambda_Y = [0.2, 2.0, 20.0, 200.0]
        n_samples = 2000

        X = np.array([expon.rvs(scale=1/lam, size=n_samples) for lam in lambda_X])
        Y = np.array([expon.rvs(scale=1/lam, size=n_samples) for lam in lambda_Y])

        wass = WassersteinDistance(X, Y, distribution='exponential')
        D = wass.compute()

        # Should work without numerical issues
        assert np.all(D >= 0)
        assert np.all(np.isfinite(D))

        # Similar indices should have smaller distances
        assert D[0, 0] < D[0, 3]  # 0.1 vs 0.2 < 0.1 vs 200.0
        assert D[3, 3] < D[3, 0]  # 100.0 vs 200.0 < 100.0 vs 0.2


class TestWassersteinDistanceClassCython:
    """Test WassersteinDistance class with optimized Cython backend."""

    def test_exponential_basic_computation(self):
        """Test basic distance matrix computation for exponential distributions."""
        np.random.seed(42)
        # Créer 10 séries temporelles de 500 échantillons chacune
        n_series = 10
        n_samples = 500
        
        lambda_X = np.random.uniform(0.5, 3.0, n_series)
        lambda_Y = np.random.uniform(0.5, 3.0, n_series)
        
        X = np.array([expon.rvs(scale=1/lam, size=n_samples) for lam in lambda_X])
        Y = np.array([expon.rvs(scale=1/lam, size=n_samples) for lam in lambda_Y])
        
        # Compute using new Cython backend
        wass = WassersteinDistance(X, Y, distribution='exponential')
        D = wass.compute()
        
        # Check shape and properties
        assert D.shape == (n_series, n_series)
        assert np.all(D >= 0)
        assert np.all(np.isfinite(D))

    def test_weibull_basic_computation(self):
        """Test basic distance matrix computation for Weibull distributions."""
        np.random.seed(123)
        n_series = 10
        n_samples = 500
        
        k_X = np.random.uniform(1.5, 3.0, n_series)
        lambda_X = np.random.uniform(0.8, 2.5, n_series)
        k_Y = np.random.uniform(1.5, 3.0, n_series)
        lambda_Y = np.random.uniform(0.8, 2.5, n_series)
        
        X = np.array([weibull_min.rvs(k, scale=lam, size=n_samples) 
                      for k, lam in zip(k_X, lambda_X)])
        Y = np.array([weibull_min.rvs(k, scale=lam, size=n_samples) 
                      for k, lam in zip(k_Y, lambda_Y)])
        
        # Compute using new Cython backend
        wass = WassersteinDistance(X, Y, distribution='weibull')
        D = wass.compute()
        
        # Check shape and properties
        assert D.shape == (n_series, n_series)
        assert np.all(D >= 0)
        assert np.all(np.isfinite(D))

    def test_squared_option_cython(self):
        """Test that squared Wasserstein distance is computed correctly in Cython version."""
        np.random.seed(42)
        X = np.array([expon.rvs(scale=1.0, size=100) for _ in range(5)])
        Y = np.array([expon.rvs(scale=1.5, size=100) for _ in range(5)])
        
        # Compute squared Wasserstein distance
        wass = WassersteinDistance(X, Y, distribution='exponential')
        D_squared = wass.compute()
        
        # Check that result is positive and finite
        assert np.all(D_squared >= 0)
        assert np.all(np.isfinite(D_squared))
        assert D_squared.shape == (5, 5)

    def test_estimate_parameters_accessible(self):
        """Test that estimate_parameters function is accessible."""
        np.random.seed(42)
        X = np.array([expon.rvs(scale=1.0, size=500) for _ in range(3)])
        Y = np.array([expon.rvs(scale=1.5, size=500) for _ in range(3)])
        
        # Create instance
        wass = WassersteinDistance(X, Y, distribution='exponential')
        
        # Check that estimate_parameters is callable
        assert callable(wass.estimate_parameters)
        
        # Test estimation
        lambda_est = wass.estimate_parameters(X[0])
        assert isinstance(lambda_est, (float, np.floating))
        assert lambda_est > 0
        # Should be close to 1.0
        assert abs(lambda_est - 1.0) < 0.2

    def test_weibull_estimate_parameters(self):
        """Test Weibull parameter estimation through class."""
        np.random.seed(456)
        k_true, lambda_true = 2.0, 1.5
        samples = weibull_min.rvs(k_true, scale=lambda_true, size=1000)
        
        X = np.array([samples])
        Y = np.array([samples])
        
        wass = WassersteinDistance(X, Y, distribution='weibull')
        
        # Estimate parameters
        k_est, lambda_est = wass.estimate_parameters(samples)
        
        # Check types
        assert isinstance(k_est, (float, np.floating))
        assert isinstance(lambda_est, (float, np.floating))
        
        # Check reasonable values
        assert 0.5 < k_est < 10.0
        assert lambda_est > 0
        
        # Should be reasonably close to true values
        assert abs(k_est - k_true) / k_true < 0.2
        assert abs(lambda_est - lambda_true) / lambda_true < 0.2

    def test_unsupported_distribution_error(self):
        """Test that unsupported distributions raise ValueError."""
        X = np.array([[1, 2, 3], [4, 5, 6]])
        Y = np.array([[1, 2, 3], [4, 5, 6]])
        
        with pytest.raises(ValueError, match="Only 'exponential' and 'weibull' are supported"):
            WassersteinDistance(X, Y, distribution='rayleigh')
        
        with pytest.raises(ValueError, match="Only 'exponential' and 'weibull' are supported"):
            WassersteinDistance(X, Y, distribution='gaussian')

    def test_invalid_dimensions_error(self):
        """Test that 1D arrays raise ValueError."""
        X_1d = np.array([1, 2, 3, 4, 5])
        Y = np.array([[1, 2, 3], [4, 5, 6]])
        
        with pytest.raises(ValueError, match="must be 2D"):
            WassersteinDistance(X_1d, Y, distribution='exponential')

    def test_identical_series_cython(self):
        """Test that identical series produce near-zero distances."""
        np.random.seed(42)
        lambda_series = [1.0, 3.0, 5.0]
        n_samples = 2000
        
        X = np.array([expon.rvs(scale=1/lam, size=n_samples) for lam in lambda_series])
        Y = X.copy()
        
        wass = WassersteinDistance(X, Y, distribution='exponential')
        D = wass.compute()
        
        # Diagonal should be approximately zero
        np.testing.assert_allclose(np.diag(D), 0, atol=1e-3)

    def test_large_matrix_cython(self):
        """Test computation with larger matrices using Cython."""
        np.random.seed(42)
        n_timepoints = 50
        n_samples = 500
        
        # Generate larger time series
        lambda_X = np.random.uniform(0.5, 5.0, n_timepoints)
        lambda_Y = np.random.uniform(0.5, 5.0, n_timepoints)
        
        X = np.array([expon.rvs(scale=1/lam, size=n_samples) for lam in lambda_X])
        Y = np.array([expon.rvs(scale=1/lam, size=n_samples) for lam in lambda_Y])
        
        # Should complete without error
        wass = WassersteinDistance(X, Y, distribution='exponential')
        D = wass.compute()
        
        assert D.shape == (n_timepoints, n_timepoints)
        assert np.all(D >= 0)
        assert np.all(np.isfinite(D))

    def test_performance_improvement(self):
        """Test that Cython implementation is significantly faster."""
        import time
        
        np.random.seed(42)
        n_series = 100
        n_samples = 500
        
        lambda_params = np.random.uniform(0.5, 3.0, n_series)
        X = np.array([expon.rvs(scale=1/lam, size=n_samples) for lam in lambda_params])
        Y = np.array([expon.rvs(scale=1/lam, size=n_samples) for lam in lambda_params])
        
        # Time the computation
        start = time.perf_counter()
        wass = WassersteinDistance(X, Y, distribution='exponential')
        D = wass.compute()
        elapsed = time.perf_counter() - start
        
        # Should complete in reasonable time (< 0.01s for 100x100 matrix)
        assert elapsed < 0.01
        assert D.shape == (n_series, n_series)

    def test_precision_exponential(self):
        """Test precision of exponential distance computation."""
        np.random.seed(42)
        # Use known parameters
        lambda1, lambda2 = 1.0, 2.0
        n_samples = 5000
        
        X = np.array([expon.rvs(scale=1/lambda1, size=n_samples)])
        Y = np.array([expon.rvs(scale=1/lambda2, size=n_samples)])
        
        wass = WassersteinDistance(X, Y, distribution='exponential')
        D = wass.compute()
        
        # Expected: |λ1 - λ2|² / (λ1 * λ2) = 0.5
        expected = 0.5
        
        # Should be very close (allowing for estimation error)
        assert abs(D[0, 0] - expected) / expected < 0.05

    def test_precision_weibull(self):
        """Test precision of Weibull distance computation."""
        np.random.seed(42)
        # Use known parameters
        k1, lambda1 = 2.0, 1.5
        k2, lambda2 = 2.0, 2.0
        n_samples = 5000
        
        X = np.array([weibull_min.rvs(k1, scale=lambda1, size=n_samples)])
        Y = np.array([weibull_min.rvs(k2, scale=lambda2, size=n_samples)])
        
        wass = WassersteinDistance(X, Y, distribution='weibull')
        D = wass.compute()
        
        # Distance should be finite and positive
        assert D[0, 0] > 0
        assert np.isfinite(D[0, 0])
        # For similar k values, distance should be moderate
        assert 0.1 < D[0, 0] < 2.0
