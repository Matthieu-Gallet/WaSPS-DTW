# Author: Mathieu Blondel
# License: Simplified BSD

# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
np.import_array()

from scipy.special.cython_special cimport psi

from libc.float cimport DBL_MAX
from libc.math cimport exp, log, tgamma, lgamma, tan, sin, cos
from libc.string cimport memset


cdef inline double _softmin3(double a,
                             double b,
                             double c,
                             double gamma):
    a /= -gamma
    b /= -gamma
    c /= -gamma

    cdef double max_val = max(max(a, b), c)

    cdef double tmp = 0
    tmp += exp(a - max_val)
    tmp += exp(b - max_val)
    tmp += exp(c - max_val)

    return -gamma * (log(tmp) + max_val)


def _soft_dtw(np.ndarray[double, ndim=2] D,
              np.ndarray[double, ndim=2] R,
              double gamma):

    cdef int m = D.shape[0]
    cdef int n = D.shape[1]

    cdef int i, j

    # Initialization.
    memset(<void*>R.data, 0, (m+1) * (n+1) * sizeof(double))

    for i in range(m + 1):
        R[i, 0] = DBL_MAX

    for j in range(n + 1):
        R[0, j] = DBL_MAX

    R[0, 0] = 0

    # DP recursion.
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # D is indexed starting from 0.
            R[i, j] = D[i-1, j-1] + _softmin3(R[i-1, j],
                                              R[i-1, j-1],
                                              R[i, j-1],
                                              gamma)


def _soft_dtw_grad(np.ndarray[double, ndim=2] D,
                   np.ndarray[double, ndim=2] R,
                   np.ndarray[double, ndim=2] E,
                   double gamma):

    # We added an extra row and an extra column on the Python side.
    cdef int m = D.shape[0] - 1
    cdef int n = D.shape[1] - 1

    cdef int i, j
    cdef double a, b, c

    # Initialization.
    memset(<void*>E.data, 0, (m+2) * (n+2) * sizeof(double))

    for i in range(1, m+1):
        # For D, indices start from 0 throughout.
        D[i-1, n] = 0
        R[i, n+1] = -DBL_MAX

    for j in range(1, n+1):
        D[m, j-1] = 0
        R[m+1, j] = -DBL_MAX

    E[m+1, n+1] = 1
    R[m+1, n+1] = R[m, n]
    D[m, n] = 0

    # DP recursion.
    for j in reversed(range(1, n+1)):  # ranges from n to 1
        for i in reversed(range(1, m+1)):  # ranges from m to 1
            a = exp((R[i+1, j] - R[i, j] - D[i, j-1]) / gamma)
            b = exp((R[i, j+1] - R[i, j] - D[i-1, j]) / gamma)
            c = exp((R[i+1, j+1] - R[i, j] - D[i, j]) / gamma)
            E[i, j] = E[i+1, j] * a + E[i, j+1] * b + E[i+1,j+1] * c


def _jacobian_product_sq_euc(np.ndarray[double, ndim=2] X,
                             np.ndarray[double, ndim=2] Y,
                             np.ndarray[double, ndim=2] E,
                             np.ndarray[double, ndim=2] G):
    cdef int m = X.shape[0]
    cdef int n = Y.shape[0]
    cdef int d = X.shape[1]

    for i in range(m):
        for j in range(n):
            for k in range(d):
                G[i, k] += E[i,j] * 2 * (X[i, k] - Y[j, k])

# Add Wassertein distance support

def _jacobian_product_sq_was_exp(np.ndarray[double, ndim=2] X,
                                 np.ndarray[double, ndim=2] Y,
                                 np.ndarray[double, ndim=2] E,
                                 np.ndarray[double, ndim=2] G):
    """
    Jacobian product for squared Wasserstein-2 distance between exponential distributions.
    
    ∂W₂²(μ,ν)/∂λ₁ = 4(λ₁-λ₂)/(λ₁³λ₂)
    
    Parameters
    ----------
    X : array, shape = [m, 1]
        Lambda parameters for first distributions
    Y : array, shape = [n, 1] 
        Lambda parameters for second distributions
    E : array, shape = [m, n]
        Error matrix
    G : array, shape = [m, 1]
        Output gradient matrix (modified in-place)
    """
    cdef int m = X.shape[0]
    cdef int n = Y.shape[0]
    cdef double lambda1, lambda2, grad_val, denom
    cdef double eps = 1e-8  # Numerical stability
    
    for i in range(m):
        for j in range(n):
            lambda1 = max(X[i, 0], eps)
            lambda2 = max(Y[j, 0], eps)
            # ∂W₂²/∂λ₁ = 4(λ₁-λ₂)/(λ₁³λ₂)
            denom = lambda1 * lambda1 * lambda1 * lambda2
            grad_val = 4.0 * (lambda1 - lambda2) / (denom + eps)  # Signe corrigé
            # Clip gradient to prevent explosion
            if grad_val > 1e6:
                grad_val = 1e6
            elif grad_val < -1e6:
                grad_val = -1e6
            G[i, 0] += E[i, j] * grad_val


def _jacobian_product_sq_was_weibull(np.ndarray[double, ndim=2] X,
                                     np.ndarray[double, ndim=2] Y,
                                     np.ndarray[double, ndim=2] E,
                                     np.ndarray[double, ndim=2] G):
    """
    Jacobian product for squared Wasserstein-2 distance between Weibull distributions.
    
    ∂W₂²(μ,ν)/∂λ₁ = 2[λ₁Γ((k₁+2)/k₁) - λ₂Γ((k₁(1+k₂)+k₂)/(k₁k₂))]
    ∂W₂²(μ,ν)/∂k₁ = (2λ₁/k₁²)[λ₂Γ((k₁(1+k₂)+k₂)/(k₁k₂))ψ((k₁(1+k₂)+k₂)/(k₁k₂)) - λ₁Γ((k₁+2)/k₁)ψ((k₁+2)/k₁)]
    
    Parameters
    ----------
    X : array, shape = [m, 2]
        [k, lambda] parameters for first distributions
    Y : array, shape = [n, 2]
        [k, lambda] parameters for second distributions
    E : array, shape = [m, n]
        Error matrix
    G : array, shape = [m, 2]
        Output gradient matrix (modified in-place)
    """
    cdef int m = X.shape[0]
    cdef int n = Y.shape[0]
    cdef double k1, lambda1, k2, lambda2
    cdef double gamma_term1, gamma_term2, psi_term1, psi_term2
    
    for i in range(m):
        for j in range(n):
            k1 = X[i, 0]
            lambda1 = X[i, 1]
            k2 = Y[j, 0]
            lambda2 = Y[j, 1]
            
            # ∂W₂²/∂λ₁ = 2[λ₁Γ((k₁+2)/k₁) - λ₂Γ((k₁(1+k₂)+k₂)/(k₁k₂))]
            gamma_term1 = tgamma((k1 + 2.0) / k1)
            gamma_term2 = tgamma((k1 * (1.0 + k2) + k2) / (k1 * k2))
            G[i, 1] += E[i, j] * 2.0 * (lambda1 * gamma_term1 - lambda2 * gamma_term2)
            
            # ∂W₂²/∂k₁ = (2λ₁/k₁²)[λ₂Γ((k₁(1+k₂)+k₂)/(k₁k₂))ψ((k₁(1+k₂)+k₂)/(k₁k₂)) - λ₁Γ((k₁+2)/k₁)ψ((k₁+2)/k₁)]
            psi_term1 = psi((k1 * (1.0 + k2) + k2) / (k1 * k2))
            psi_term2 = psi((k1 + 2.0) / k1)
            G[i, 0] += E[i, j] * (2.0 * lambda1 / (k1 * k1)) * (lambda2 * gamma_term2 * psi_term1 - lambda1 * gamma_term1 * psi_term2)