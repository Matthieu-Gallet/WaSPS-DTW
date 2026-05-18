"""
Barycenter and distance computation for Soft-DTW classification.

Three barycenter methods:
  euclidean_raw    — Soft-DTW barycenter on raw time series (Euclidean distance)
  euclidean_params — Soft-DTW barycenter on estimated parameters (Euclidean distance)
  wasserstein_sgd  — Soft-DTW barycenter via SGD with Wasserstein distance

Two distance functions (for nearest-barycenter classification):
  sdtw_euclidean   — Soft-DTW with squared Euclidean cost
  sdtw_wasserstein — Soft-DTW with exponential Wasserstein cost

All functions: arrays in, arrays/scalars out. No I/O, no dataset coupling.
"""

import numpy as np
from typing import Callable, Dict, List, Optional

from sdtw import SoftDTW
from sdtw.barycenter import sdtw_barycenter
from sdtw.distance import SquaredEuclidean, WassersteinDistance
from src.optimizer import sgd_barycenter


# =============================================================================
# Barycenter computation
# =============================================================================

def compute_barycenter_euclidean_raw(samples: List[np.ndarray],
                                     gamma: float = 1.0,
                                     max_iter: int = 50) -> np.ndarray:
    """
    Soft-DTW barycenter with Euclidean distance on raw time series.

    Args:
        samples:  List of (T, D) arrays.
        gamma:    Soft-DTW regularisation parameter.
        max_iter: Maximum DBA iterations.

    Returns:
        Barycenter array (T, D).
    """
    min_len = min(s.shape[0] for s in samples)
    aligned = [s[:min_len].astype(np.float64) for s in samples]
    bary_init = np.mean(np.array(aligned), axis=0)
    return sdtw_barycenter(aligned, bary_init, gamma=gamma,
                           max_iter=max_iter, distance='euclidean')


def compute_barycenter_euclidean_params(params: List[np.ndarray],
                                        gamma: float = 1.0,
                                        max_iter: int = 100) -> np.ndarray:
    """
    Soft-DTW barycenter with Euclidean distance on estimated distribution parameters.

    Args:
        params:   List of (T, 1) parameter arrays.
        gamma:    Soft-DTW regularisation parameter.
        max_iter: Maximum DBA iterations.

    Returns:
        Barycenter parameter array (T, 1).
    """
    bary_init = np.mean(params, axis=0)
    return sdtw_barycenter(params, bary_init, gamma=gamma,
                           max_iter=max_iter, distance='euclidean')


def compute_barycenter_wasserstein_sgd(params: List[np.ndarray],
                                       gamma: float = 1.0,
                                       learning_rate: float = 0.075,
                                       num_epochs: int = 250,
                                       max_iter: Optional[int] = None,
                                       batch_size: int = 4,
                                       lr_decay: float = 0.995,
                                       grad_clip: float = 100.0,
                                       lambda_min: float = 1e-3,
                                       verbose: bool = False) -> np.ndarray:
    """
    Soft-DTW barycenter via SGD with Wasserstein distance on distribution parameters.

    Args:
        params:        List of (T, 1) parameter arrays.
        gamma:         Soft-DTW regularisation parameter.
        learning_rate: Initial SGD learning rate.
        num_epochs:    Number of SGD epochs.
        batch_size:    SGD mini-batch size.
        lr_decay:      Per-epoch learning rate decay factor.
        grad_clip:     Gradient clipping threshold.
        verbose:       Print SGD progress.

    Returns:
        Barycenter parameter array (T, 1).
    """
    epochs = max_iter if max_iter is not None else num_epochs
    bary, _ = sgd_barycenter(
        params, gamma=gamma, barycenter_init_method='mean_lambda',
        learning_rate=learning_rate, num_epochs=epochs,
        batch_size=batch_size, lr_decay=lr_decay, grad_clip=grad_clip,
        distribution='exponential', verbose=verbose, use_softplus=True,
        X_is_params=True, lambda_min=lambda_min,
    )
    return bary


# =============================================================================
# Distance computation
# =============================================================================

def compute_sdtw_distance_euclidean(sample: np.ndarray, barycenter: np.ndarray,
                                    gamma: float = 1.0) -> float:
    """Soft-DTW distance with squared Euclidean cost."""
    D = SquaredEuclidean(sample, barycenter)
    return SoftDTW(D, gamma=gamma).compute()


def compute_sdtw_distance_wasserstein(params: np.ndarray, barycenter_params: np.ndarray,
                                      gamma: float = 1.0) -> float:
    """Soft-DTW distance with exponential Wasserstein cost on parameter arrays."""
    D = WassersteinDistance(params, barycenter_params, distribution='exponential',
                            precompute_params=True, X_is_params=True, Y_is_params=True)
    return SoftDTW(D, gamma=gamma).compute()


# =============================================================================
# Classification
# =============================================================================

def classify_by_nearest_barycenter(samples: List[np.ndarray],
                                   barycenters: Dict[int, np.ndarray],
                                   distance_func: Callable,
                                   gamma: float = 1.0) -> np.ndarray:
    """
    Assign each sample to the class whose barycenter is nearest.

    Args:
        samples:       List of sample arrays.
        barycenters:   {class_label: barycenter_array}.
        distance_func: Callable(sample, barycenter, gamma) → float.
        gamma:         Soft-DTW regularisation parameter.

    Returns:
        Integer prediction array of length len(samples).
    """
    predictions = []
    for sample in samples:
        best_label = min(barycenters,
                         key=lambda lbl: distance_func(sample, barycenters[lbl], gamma))
        predictions.append(best_label)
    return np.array(predictions)
