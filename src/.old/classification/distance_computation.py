"""
Distance computation functions for classification.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from sdtw import SoftDTW
from sdtw.distance import SquaredEuclidean, WassersteinDistance


def compute_sdtw_distance_euclidean(sample: np.ndarray, barycenter: np.ndarray,
                                     gamma: float = 1.0) -> float:
    """
    Compute Soft-DTW distance with Euclidean distance.
    
    Args:
        sample: Sample array
        barycenter: Barycenter array
        gamma: Soft-DTW regularization parameter
        
    Returns:
        Soft-DTW distance value
    """
    D = SquaredEuclidean(sample, barycenter)
    sdtw = SoftDTW(D, gamma=gamma)
    return sdtw.compute()


def compute_sdtw_distance_wasserstein(params: np.ndarray, barycenter_params: np.ndarray,
                                       gamma: float = 1.0) -> float:
    """
    Compute Soft-DTW distance with Wasserstein distance on parameters.
    
    Args:
        params: Sample parameter array with shape (T, 1)
        barycenter_params: Barycenter parameter array with shape (T, 1)
        gamma: Soft-DTW regularization parameter
        
    Returns:
        Soft-DTW distance value
    """
    D = WassersteinDistance(
        params, barycenter_params, distribution='exponential',
        precompute_params=True, X_is_params=True, Y_is_params=True
    )
    sdtw = SoftDTW(D, gamma=gamma)
    return sdtw.compute()
