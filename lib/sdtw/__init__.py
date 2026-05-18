"""
Soft-DTW — core algorithm library.

Provides Soft-DTW with Euclidean and Wasserstein distance metrics
and barycenter computation.
"""

from .soft_dtw import SoftDTW
from .distance import SquaredEuclidean, WassersteinDistance
from .barycenter import sdtw_barycenter

__all__ = [
    'SoftDTW',
    'SquaredEuclidean',
    'WassersteinDistance',
    'sdtw_barycenter',
]
