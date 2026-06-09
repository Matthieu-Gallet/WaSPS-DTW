# cython: boundscheck=False, wraparound=False, cdivision=True
"""
Cython-optimized functions for Wasserstein distance computations.
"""
import numpy as np
cimport numpy as np
from libc.math cimport log, exp, sqrt, fabs, pow, tgamma

# Constants (high-precision values)
cdef double DIGAMMA_1 = -0.57721566490153286060651209008240  # digamma(1) = -γ (Euler-Mascheroni)



cdef inline double compute_log_mean(double[:] data) nogil:
    """Compute mean of log(data)."""
    cdef int n = data.shape[0]
    cdef double sum_log = 0.0
    cdef int i
    
    for i in range(n):
        sum_log += log(data[i])
    
    return sum_log / n


cdef inline double compute_log_variance(double[:] data, double log_mean) nogil:
    """Compute variance of log(data)."""
    cdef int n = data.shape[0]
    cdef double sum_sq = 0.0
    cdef double log_val
    cdef int i
    
    for i in range(n):
        log_val = log(data[i]) - log_mean
        sum_sq += log_val * log_val
    
    return sum_sq / n


def estimate_exponential_fast(double[:] data):
    """
    Fast exponential parameter estimation.
    
    Parameters
    ----------
    data : array of double
        Input samples
        
    Returns
    -------
    lambda_param : double
        Estimated rate parameter
    """
    cdef double k1 = compute_log_mean(data)
    cdef double lambda_param = 1.0 / exp(k1 - DIGAMMA_1)
    return lambda_param



def wasserstein22_exponential_fast(double lambda1, double lambda2):
    """Fast W_2^2 distance for exponential distributions."""
    cdef double diff = lambda1 - lambda2
    return 2 * diff*diff / (lambda1 * lambda1 * lambda2 * lambda2)


def pairwise_wasserstein_exponential(double[:, :] X, double[:, :] Y, bint precompute_params=True):
    """
    Compute Wasserstein distance matrix for exponential distributions.

    Parameters
    ----------
    X : array, shape = [m, n_samples] or [m, 1]
        First time series (samples at each time point) or precomputed parameters
    Y : array, shape = [n, n_samples] or [n, 1]
        Second time series or precomputed parameters
    precompute_params : bool
        If True, X and Y contain precomputed parameters (shape [m, 1])
        If False, X and Y contain raw data (shape [m, n_samples])

    Returns
    -------
    D : array, shape = [m, n]
        Distance matrix
    """
    cdef int m = X.shape[0]
    cdef int n = Y.shape[0]
    cdef int i, j
    cdef double lambda_i, lambda_j

    # Allocate output with memory view for faster access
    cdef double[:, :] D = np.zeros((m, n), dtype=np.float64)

    # Handle parameter estimation based on precompute_params flag
    cdef double[:] X_params
    cdef double[:] Y_params

    if precompute_params:
        # X and Y already contain parameters
        X_params = np.asarray(X[:, 0], dtype=np.float64)
        Y_params = np.asarray(Y[:, 0], dtype=np.float64)
    else:
        # Need to estimate parameters from data
        X_params = np.zeros(m, dtype=np.float64)
        for i in range(m):
            X_params[i] = estimate_exponential_fast(X[i, :])

        Y_params = np.zeros(n, dtype=np.float64)
        for j in range(n):
            Y_params[j] = estimate_exponential_fast(Y[j, :])

    # Compute pairwise distances - inline computation for speed
    for i in range(m):
        lambda_i = X_params[i]
        for j in range(n):
            lambda_j = Y_params[j]
            D[i, j] = wasserstein22_exponential_fast(lambda_i, lambda_j)
    return np.asarray(D)


# =============================================================================
# Weibull distribution: W₂² forward + estimator + pairwise matrix
# =============================================================================

# π/sqrt(6) constant, used in log-cumulant shape estimator
cdef double _PI_OVER_SQRT6 = 1.2825498301618641  # π / sqrt(6)


def estimate_weibull_fast(double[:] data):
    """
    Estimate Weibull(k, λ) parameters by the method of log-cumulants.

    Uses the moments of log X:
      E[log X] = log(λ) + ψ(1)/k      (ψ(1) = DIGAMMA_1 ≈ -0.5772)
      Var[log X] = π²/(6k²)

    Returns
    -------
    k : double — shape parameter (> 0)
    lambda_scale : double — scale parameter (> 0)
    """
    cdef double log_mean = compute_log_mean(data)
    cdef double log_var  = compute_log_variance(data, log_mean)
    cdef double log_std  = sqrt(log_var) if log_var > 1e-12 else 1e-6
    cdef double k_hat    = _PI_OVER_SQRT6 / log_std          # k = π/(sqrt(6)·σ_log)
    cdef double lam_hat  = exp(log_mean - DIGAMMA_1 / k_hat) # λ = exp(μ_log - ψ(1)/k)
    return k_hat, lam_hat


def wasserstein22_weibull_fast(double k1, double l1, double k2, double l2):
    """
    Squared 2-Wasserstein distance between Weibull(k1,λ1) and Weibull(k2,λ2).

    W₂²(p,q) = λ₁²Γ((k₁+2)/k₁) + λ₂²Γ((k₂+2)/k₂) − 2λ₁λ₂Γ(1+1/k₁+1/k₂)

    Parameters use scale parameterisation (not rate).  W₂²(p,p) = 0 identically.
    """
    cdef double term1 = l1 * l1 * tgamma((k1 + 2.0) / k1)
    cdef double term2 = l2 * l2 * tgamma((k2 + 2.0) / k2)
    cdef double term3 = 2.0 * l1 * l2 * tgamma(1.0 + 1.0/k1 + 1.0/k2)
    return term1 + term2 - term3


def pairwise_wasserstein_weibull(double[:, :] X, double[:, :] Y, bint precompute_params=True):
    """
    Pairwise W₂² matrix between Weibull distributions.

    Parameters
    ----------
    X : array, shape = [m, 2] or [m, n_samples]
        Each row is (k, λ) if precompute_params=True, else raw samples.
    Y : array, shape = [n, 2] or [n, n_samples]
    precompute_params : bool
        True → rows are already (k, λ); False → estimate from raw samples.

    Returns
    -------
    D : array, shape = [m, n]
        Distance matrix of squared W₂ values.
    """
    cdef int m = X.shape[0]
    cdef int n = Y.shape[0]
    cdef int i, j
    cdef double k_i, l_i, k_j, l_j

    cdef double[:, :] D = np.zeros((m, n), dtype=np.float64)

    # Parameter arrays (k and λ separately for fast inner loop)
    X_k = np.zeros(m, dtype=np.float64)
    X_l = np.zeros(m, dtype=np.float64)
    Y_k = np.zeros(n, dtype=np.float64)
    Y_l = np.zeros(n, dtype=np.float64)

    if precompute_params:
        for i in range(m):
            X_k[i] = X[i, 0]
            X_l[i] = X[i, 1]
        for j in range(n):
            Y_k[j] = Y[j, 0]
            Y_l[j] = Y[j, 1]
    else:
        for i in range(m):
            params = estimate_weibull_fast(X[i, :])
            X_k[i] = params[0]
            X_l[i] = params[1]
        for j in range(n):
            params = estimate_weibull_fast(Y[j, :])
            Y_k[j] = params[0]
            Y_l[j] = params[1]

    cdef double[:] _X_k = X_k
    cdef double[:] _X_l = X_l
    cdef double[:] _Y_k = Y_k
    cdef double[:] _Y_l = Y_l

    for i in range(m):
        k_i = _X_k[i]
        l_i = _X_l[i]
        for j in range(n):
            k_j = _Y_k[j]
            l_j = _Y_l[j]
            D[i, j] = wasserstein22_weibull_fast(k_i, l_i, k_j, l_j)

    return np.asarray(D)
