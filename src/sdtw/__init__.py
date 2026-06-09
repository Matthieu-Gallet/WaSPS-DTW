"""
Soft-DTW module.

This module provides Soft-DTW implementation with:
- Euclidean and Wasserstein distance metrics
- Barycenter computation algorithms
- Classification helper methods
"""

from .soft_dtw import SoftDTW, sdtw_divergence
from .distance import SquaredEuclidean, WassersteinDistance
from .barycenter import sdtw_barycenter
from .classification_methods import (
    compute_barycenter_euclidean_raw,
    compute_barycenter_euclidean_params,
    compute_barycenter_wasserstein_sgd,
    compute_sdtw_distance_euclidean,
    compute_sdtw_distance_wasserstein,
    compute_sdtw_distance_weibull,
)

__all__ = [
    'SoftDTW',
    'sdtw_divergence',
    'SquaredEuclidean',
    'WassersteinDistance',
    'sdtw_barycenter',
    # Classification methods
    'compute_barycenter_euclidean_raw',
    'compute_barycenter_euclidean_params',
    'compute_barycenter_wasserstein_sgd',
    'compute_sdtw_distance_euclidean',
    'compute_sdtw_distance_wasserstein',
    'compute_sdtw_distance_weibull',
]
