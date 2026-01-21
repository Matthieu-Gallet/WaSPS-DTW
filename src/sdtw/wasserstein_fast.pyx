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
