"""
Barycenter and distance computation methods for classification tasks.

All distance functions accept ``divergence=True`` (default) which computes the
Blondel soft-DTW divergence D_γ(X,Y) = SDTW(X,Y) − ½(SDTW(X,X)+SDTW(Y,Y))
instead of the raw SDTW value.  Set ``divergence=False`` to reproduce the
original behaviour and compare against earlier results.

The divergence is applied only at the **classification** step (nearest-barycenter
assignment), not during barycenter optimisation.  Self-terms SDTW(b,b) are NOT
constant across barycenters, so they shift the argmin and must be included.
"""

import numpy as np
from typing import List
import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from sdtw.barycenter import sdtw_barycenter
from sdtw.soft_dtw import SoftDTW, sdtw_divergence
from sdtw.distance import SquaredEuclidean, WassersteinDistance
# sgd_barycenter imported lazily inside the function that uses it


# =============================================================================
# Barycenter computation
# =============================================================================

def compute_barycenter_euclidean_raw(samples: List[np.ndarray], gamma: float = 1.0,
                                      max_iter: int = 1000) -> np.ndarray:
    """Soft-DTW barycenter with squared-Euclidean local cost on raw data.

    Args:
        samples:  list of (T, D) arrays — raw observations per sample.
        gamma:    Soft-DTW regularisation parameter.
        max_iter: maximum L-BFGS-B iterations.

    Returns:
        Barycenter array, shape (T, D).
    """
    min_len = min(s.shape[0] for s in samples)
    samples_aligned = [s[:min_len].astype(np.float64) for s in samples]
    bary_init = np.mean(np.array(samples_aligned), axis=0)
    return sdtw_barycenter(samples_aligned, bary_init, gamma=gamma,
                           max_iter=max_iter, distance="euclidean")


def compute_barycenter_euclidean_params(params_list: List[np.ndarray], gamma: float = 1.0,
                                         max_iter: int = 1000) -> np.ndarray:
    """Soft-DTW barycenter with squared-Euclidean local cost on estimated parameters.

    Args:
        params_list: list of (T, d) parameter arrays (e.g. d=1 for exponential).
        gamma:       Soft-DTW regularisation parameter.
        max_iter:    maximum L-BFGS-B iterations.

    Returns:
        Barycenter parameter array, shape (T, d).
    """
    bary_init = np.mean(params_list, axis=0)
    return sdtw_barycenter(params_list, bary_init, gamma=gamma,
                           max_iter=max_iter, distance="euclidean")


def compute_barycenter_wasserstein_sgd(params_list: List[np.ndarray], gamma: float = 1.0,
                                        learning_rate: float = 0.10, num_epochs: int = 20,
                                        batch_size: int = 4, distribution: str = 'exponential',
                                        verbose: bool = False) -> np.ndarray:
    """Soft-DTW Wasserstein barycenter via SGD.

    Args:
        params_list:   list of (T, d) parameter arrays (d=1 exponential, d=2 Weibull).
        gamma:         Soft-DTW regularisation parameter.
        learning_rate: SGD learning rate.
        num_epochs:    number of SGD epochs.
        batch_size:    SGD mini-batch size.
        distribution:  ``'exponential'`` or ``'weibull'``.
        verbose:       print progress.

    Returns:
        Barycenter parameter array, shape (T, d).
    """
    from optimizer.wasserstein_barycenter_sgd import sgd_barycenter  # lazy import
    barycenter, _ = sgd_barycenter(
        params_list, gamma=gamma, barycenter_init_method='mean_lambda',
        learning_rate=learning_rate, num_epochs=num_epochs, batch_size=batch_size,
        lr_decay=0.5, grad_clip=10.0, distribution=distribution,
        verbose=verbose, use_softplus=True, X_is_params=True,
    )
    return barycenter


# =============================================================================
# Distance computation  (classification / nearest-barycenter assignment)
# =============================================================================

def _sdtw_or_divergence(dist_obj, gamma: float, divergence: bool) -> float:
    """Compute SDTW or the full divergence depending on the flag."""
    if not divergence:
        return SoftDTW(dist_obj, gamma=gamma).compute()

    # Cross term is already computed by the distance object;
    # we just need the matrix for the self-terms.
    D_xy = dist_obj.compute()

    # Self-term for X
    # Re-use the already-estimated params if available.
    if hasattr(dist_obj, 'X_params_2d'):
        D_xx = dist_obj._compute_matrix(
            dist_obj.X_params_2d, dist_obj.X_params_2d, True
        )
        D_yy = dist_obj._compute_matrix(
            dist_obj.Y_params_2d, dist_obj.Y_params_2d, True
        )
    else:
        D_xx = dist_obj.compute()  # fallback (SquaredEuclidean case)
        # For SquaredEuclidean, build self-distances properly.
        D_xx = SquaredEuclidean(dist_obj.X, dist_obj.X).compute()
        D_yy = SquaredEuclidean(dist_obj.Y, dist_obj.Y).compute()

    return sdtw_divergence(D_xy, D_xx, D_yy, gamma)


def compute_sdtw_distance_euclidean(sample: np.ndarray, barycenter: np.ndarray,
                                     gamma: float = 1.0, divergence: bool = True) -> float:
    """Soft-DTW distance (or divergence) with squared-Euclidean local cost.

    Args:
        sample:     (T, D) sample array.
        barycenter: (T, D) barycenter array.
        gamma:      Soft-DTW regularisation parameter.
        divergence: if True (default), compute the Blondel divergence
                    D_γ(X,Y) = SDTW(X,Y) − ½(SDTW(X,X)+SDTW(Y,Y)).

    Returns:
        Scalar distance value.
    """
    sample    = np.asarray(sample,    dtype=np.float64)
    barycenter = np.asarray(barycenter, dtype=np.float64)
    if not divergence:
        return SoftDTW(SquaredEuclidean(sample, barycenter), gamma=gamma).compute()

    D_xy = SquaredEuclidean(sample, barycenter).compute()
    D_xx = SquaredEuclidean(sample, sample).compute()
    D_yy = SquaredEuclidean(barycenter, barycenter).compute()
    return sdtw_divergence(D_xy, D_xx, D_yy, gamma)


def compute_sdtw_distance_wasserstein(params: np.ndarray, barycenter_params: np.ndarray,
                                       gamma: float = 1.0, divergence: bool = True) -> float:
    """Soft-DTW distance (or divergence) with Wasserstein local cost (exponential).

    Args:
        params:            (T, 1) sample parameter array (exponential rate λ).
        barycenter_params: (T, 1) barycenter parameter array.
        gamma:             Soft-DTW regularisation parameter.
        divergence:        if True, compute the Blondel divergence.

    Returns:
        Scalar distance value.
    """
    params            = np.asarray(params,            dtype=np.float64)
    barycenter_params = np.asarray(barycenter_params, dtype=np.float64)

    def _wass(A, B):
        return WassersteinDistance(A, B, distribution='exponential',
                                   precompute_params=True,
                                   X_is_params=True, Y_is_params=True)

    if not divergence:
        return SoftDTW(_wass(params, barycenter_params), gamma=gamma).compute()

    D_xy = _wass(params, barycenter_params).compute()
    D_xx = _wass(params, params).compute()
    D_yy = _wass(barycenter_params, barycenter_params).compute()
    return sdtw_divergence(D_xy, D_xx, D_yy, gamma)


def compute_sdtw_distance_weibull(params: np.ndarray, barycenter_params: np.ndarray,
                                   gamma: float = 1.0, divergence: bool = True) -> float:
    """Soft-DTW distance (or divergence) with Wasserstein local cost (Weibull).

    Args:
        params:            (T, 2) sample parameter array — column 0: k, column 1: λ_scale.
        barycenter_params: (T, 2) barycenter parameter array.
        gamma:             Soft-DTW regularisation parameter.
        divergence:        if True, compute the Blondel divergence.

    Returns:
        Scalar distance value.
    """
    params            = np.asarray(params,            dtype=np.float64)
    barycenter_params = np.asarray(barycenter_params, dtype=np.float64)

    def _wass(A, B):
        return WassersteinDistance(A, B, distribution='weibull',
                                   precompute_params=True,
                                   X_is_params=True, Y_is_params=True)

    if not divergence:
        return SoftDTW(_wass(params, barycenter_params), gamma=gamma).compute()

    D_xy = _wass(params, barycenter_params).compute()
    D_xx = _wass(params, params).compute()
    D_yy = _wass(barycenter_params, barycenter_params).compute()
    return sdtw_divergence(D_xy, D_xx, D_yy, gamma)
