"""
Barycenter computation methods for classification.
"""

import numpy as np
from typing import List
import sys
from pathlib import Path

# Add parent directory for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from sdtw.barycenter import sdtw_barycenter
from optimizer import sgd_barycenter


def compute_barycenter_euclidean_raw(samples: List[np.ndarray], gamma: float = 1.0,
                                      max_iter: int = 50) -> np.ndarray:
    """
    Compute Soft-DTW barycenter with Euclidean distance on raw data.
    
    Args:
        samples: List of arrays with shape (T, D*W*W)
        gamma: Soft-DTW regularization parameter
        max_iter: Maximum iterations for optimization
        
    Returns:
        Barycenter array with shape (T, D*W*W)
    """
    # Ensure all samples have the same shape
    min_len = min(s.shape[0] for s in samples)
    samples_aligned = [s[:min_len, :].astype(np.float64) for s in samples]
    
    # Initialize with mean
    bary_init = np.mean(np.array(samples_aligned), axis=0)
    
    # Compute barycenter
    barycenter = sdtw_barycenter(
        samples_aligned, bary_init, gamma=gamma,
        max_iter=max_iter, distance="euclidean"
    )
    
    return barycenter


def compute_barycenter_euclidean_params(params_list: List[np.ndarray], gamma: float = 1.0,
                                         max_iter: int = 100) -> np.ndarray:
    """
    Compute Soft-DTW barycenter with Euclidean distance on estimated parameters.
    
    Args:
        params_list: List of parameter arrays with shape (T, 1)
        gamma: Soft-DTW regularization parameter
        max_iter: Maximum iterations for optimization
        
    Returns:
        Barycenter parameters array with shape (T, 1)
    """
    # Initialize with mean
    bary_init = np.mean(params_list, axis=0)
    
    # Compute barycenter
    barycenter = sdtw_barycenter(
        params_list, bary_init, gamma=gamma,
        max_iter=max_iter, distance="euclidean"
    )
    
    return barycenter


def compute_barycenter_wasserstein_sgd(params_list: List[np.ndarray], gamma: float = 1.0,
                                        learning_rate: float = 0.075, num_epochs: int = 250,
                                        batch_size: int = 4, verbose: bool = False) -> np.ndarray:
    """
    Compute Soft-DTW barycenter with Wasserstein distance using SGD.
    
    Args:
        params_list: List of parameter arrays with shape (T, 1)
        gamma: Soft-DTW regularization parameter
        learning_rate: SGD learning rate
        num_epochs: Number of SGD epochs
        batch_size: SGD batch size
        verbose: Print debug information
        
    Returns:
        Barycenter parameters array with shape (T, 1)
    """
    # SGD barycenter expects parameters directly
    barycenter, _ = sgd_barycenter(
        params_list, gamma=gamma, barycenter_init_method='mean_lambda',
        learning_rate=learning_rate, num_epochs=num_epochs, batch_size=batch_size,
        lr_decay=0.995, grad_clip=100.0, distribution="exponential",
        verbose=verbose, use_softplus=True, X_is_params=True
    )
    
    return barycenter
