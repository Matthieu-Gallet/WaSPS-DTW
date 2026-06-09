"""
Distance objects for Soft-DTW: Squared-Euclidean and Wasserstein (exponential + Weibull).

Each distance object exposes:
  - compute() → [m, n] distance matrix
  - jacobian_product(E) → [m, d] gradient w.r.t. X (first argument)
  - jacobian_product_Y(E) → [n, d] gradient w.r.t. Y (second argument)

Both are passed directly to SoftDTW, which calls .compute() if the object has that method.
"""
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances

from .wasserstein_fast import (
    estimate_exponential_fast,
    pairwise_wasserstein_exponential,
    estimate_weibull_fast,
    pairwise_wasserstein_weibull,
)
from .soft_dtw_fast import (
    _jacobian_product_sq_euc,
    _jacobian_product_sq_was_exp,
    _jacobian_product_sq_was_weibull,
)


class SquaredEuclidean:
    """Squared Euclidean distance for use as a Soft-DTW local cost."""

    def __init__(self, X, Y):
        """
        Parameters
        ----------
        X : array, shape = [m, d]
        Y : array, shape = [n, d]
        """
        self.X = X.astype(np.float64)
        self.Y = Y.astype(np.float64)

    def compute(self):
        """Return [m, n] squared-Euclidean distance matrix."""
        return euclidean_distances(self.X, self.Y, squared=True)

    def jacobian_product(self, E):
        """
        Jacobian–E product: G[i, k] = Σ_j E[i,j] · 2(X[i,k] - Y[j,k]).

        Returns
        -------
        G : array, shape = [m, d]
        """
        G = np.zeros_like(self.X)
        _jacobian_product_sq_euc(self.X, self.Y, E, G)
        return G


class WassersteinDistance:
    """
    Squared Wasserstein-2 (W₂²) distance for exponential or Weibull distributions.

    The local cost matrix is W₂² (not W₂).  This is consistent with the
    paper's forward recursion (eqn:wasps-recursion) and with the existing
    exponential implementation.

    Parameters
    ----------
    X : array, shape = [m, n_samples] or [m, d_params]
        First time series.  Each row holds raw samples OR precomputed parameters.
    Y : array, shape = [n, n_samples] or [n, d_params]
    distribution : {'exponential', 'weibull'}
        Distribution family.  Exponential uses a single rate parameter λ (d=1);
        Weibull uses (k, λ_scale) stored as column-0=k, column-1=λ (d=2).
    precompute_params : bool
        If True, estimate parameters once at construction time (faster for
        repeated calls such as barycenter optimisation).
    X_is_params : bool
        If True, X already holds distribution parameters — skip estimation.
    Y_is_params : bool
        If True, Y already holds distribution parameters — skip estimation.

    Notes
    -----
    - Exponential rate β ↔ Weibull(k=1, λ=1/β).  The two parameterisations
      are *not* interchangeable objects; pick one distribution and stay with it.
    - Weibull parameter array has shape (T, 2): [:, 0]=k, [:, 1]=λ_scale.
    """

    _SUPPORTED = {'exponential', 'weibull'}

    def __init__(self, X, Y, distribution='exponential', precompute_params=True,
                 X_is_params=False, Y_is_params=False):
        self.X = np.asarray(X, dtype=np.float64)
        self.Y = np.asarray(Y, dtype=np.float64)
        self.distribution = distribution.lower()
        self.precompute_params = precompute_params
        self.X_is_params = X_is_params
        self.Y_is_params = Y_is_params

        if self.X.ndim != 2 or self.Y.ndim != 2:
            raise ValueError("X and Y must be 2-D arrays.")
        if self.distribution not in self._SUPPORTED:
            raise ValueError(
                f"distribution must be one of {self._SUPPORTED}, got '{self.distribution}'."
            )

        if self.distribution == 'exponential':
            self._compute_matrix = pairwise_wasserstein_exponential
            self._jacobian_func   = _jacobian_product_sq_was_exp
            self._d_params        = 1
            if precompute_params:
                self.X_params_2d = (
                    self.X.copy() if X_is_params
                    else np.array([estimate_exponential_fast(self.X[i])
                                   for i in range(self.X.shape[0])]).reshape(-1, 1)
                )
                self.Y_params_2d = (
                    self.Y.copy() if Y_is_params
                    else np.array([estimate_exponential_fast(self.Y[j])
                                   for j in range(self.Y.shape[0])]).reshape(-1, 1)
                )

        elif self.distribution == 'weibull':
            self._compute_matrix = pairwise_wasserstein_weibull
            self._jacobian_func   = _jacobian_product_sq_was_weibull
            self._d_params        = 2  # (k, λ_scale)
            if precompute_params:
                self.X_params_2d = (
                    self.X.copy() if X_is_params
                    else np.array([estimate_weibull_fast(self.X[i])
                                   for i in range(self.X.shape[0])], dtype=np.float64)
                )
                self.Y_params_2d = (
                    self.Y.copy() if Y_is_params
                    else np.array([estimate_weibull_fast(self.Y[j])
                                   for j in range(self.Y.shape[0])], dtype=np.float64)
                )

        self.distance_matrix = None

    # ------------------------------------------------------------------
    def compute(self):
        """Compute and return the [m, n] W₂² distance matrix."""
        if self.precompute_params:
            self.distance_matrix = self._compute_matrix(
                self.X_params_2d, self.Y_params_2d, True
            )
        else:
            self.distance_matrix = self._compute_matrix(self.X, self.Y, False)
        return self.distance_matrix

    # ------------------------------------------------------------------
    def jacobian_product(self, E):
        """
        Jacobian product ∂L/∂params_X.

        Returns
        -------
        G : array, shape = [m, d_params]
            d_params = 1 for exponential (∂/∂λ_rate),
            d_params = 2 for Weibull      (∂/∂k, ∂/∂λ_scale).
        """
        E = np.asarray(E, dtype=np.float64)
        G = np.zeros((self.X.shape[0], self._d_params), dtype=np.float64)
        self._jacobian_func(self.X_params_2d, self.Y_params_2d, E, G)
        return G

    # ------------------------------------------------------------------
    def jacobian_product_Y(self, E):
        """
        Jacobian product ∂L/∂params_Y.

        For exponential an explicit formula is used; for Weibull we exploit
        the symmetry of W₂²(p,q): the gradient w.r.t. q is the same Prop-2
        formula with p↔q, implemented by calling the Cython function with
        swapped arguments and transposed E.

        Returns
        -------
        G_Y : array, shape = [n, d_params]
        """
        E = np.asarray(E, dtype=np.float64)

        if self.distribution == 'exponential':
            lx = self.X_params_2d.ravel()   # (m,)
            ly = self.Y_params_2d.ravel()   # (n,)
            # ∂W₂²/∂λ_y = 4*(1/λ_x - 1/λ_y) / λ_y²
            diff  = (1.0 / lx)[:, None] - (1.0 / ly)[None, :]  # (m, n)
            G_Y   = np.sum(E * 4.0 * diff / (ly[None, :] ** 2), axis=0)
            return G_Y.reshape(-1, 1)

        elif self.distribution == 'weibull':
            # ∂W₂²(p_i, q_j)/∂params_q is the same formula as ∂/∂params_p
            # with p ↔ q.  The Cython function computes G[i] += Σ_j E[i,j]·∂/∂p_i,
            # so calling it with (Y_params, X_params, E.T, G_Y) gives
            # G_Y[j] += Σ_i E[i,j]·∂/∂q_j  ✓
            G_Y = np.zeros((self.Y.shape[0], self._d_params), dtype=np.float64)
            self._jacobian_func(self.Y_params_2d, self.X_params_2d,
                                np.ascontiguousarray(E.T), G_Y)
            return G_Y

    def __repr__(self):
        return (f"WassersteinDistance(distribution='{self.distribution}', "
                f"squared=True)")
