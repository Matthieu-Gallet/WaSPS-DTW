"""
Data loading and preprocessing functions for classification datasets.

This module provides functions for:
- Loading classification datasets (X, Y, metadata)
- Preprocessing raw data for Soft-DTW computation
- Estimating distribution parameters from raw data
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import sys

# Add parent directory for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from estimator import LogCumulant


def load_classification_dataset(data_dir: str, mode: str = "basic"
                                 ) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load classification dataset created by build_classification_dataset.py.
    
    Args:
        data_dir: Directory containing the dataset files
        mode: "basic" or "balanced"
        
    Returns:
        Tuple of (X, Y, metadata)
    """
    data_path = Path(data_dir)
    
    X = np.load(data_path / f"X_{mode}.npy")
    Y = np.load(data_path / f"Y_{mode}.npy")
    metadata = np.load(data_path / f"metadata_{mode}.npy", allow_pickle=True).item()
    
    return X, Y, metadata


def preprocess_samples(X: np.ndarray, max_time_steps: Optional[int] = None
                       ) -> List[np.ndarray]:
    """
    Preprocess samples for Soft-DTW computation.
    
    Input X has shape (N, T, D, W, W) where:
        - N: number of samples
        - T: number of time groups
        - D: temporal window size
        - W: spatial window size
    
    Output: List of arrays with shape (T, D*W*W) for each sample
    (each time step contains D*W*W values flattened)
    
    Args:
        X: Input array of shape (N, T, D, W, W)
        max_time_steps: Optional limit on number of time steps to use
        
    Returns:
        List of preprocessed samples
    """
    N, T, D, W1, W2 = X.shape
    
    if max_time_steps is not None:
        T = min(T, max_time_steps)
    
    samples = []
    for i in range(N):
        # Reshape to (T, D*W*W)
        sample = X[i, :T, :, :, :].reshape(T, -1).astype(np.float64)
        # Replace NaN with 0 (or could use mean/interpolation)
        sample = np.nan_to_num(sample, nan=0.0)
        samples.append(sample)
    
    return samples


def estimate_parameters_for_samples(X: np.ndarray,
                                    max_time_steps: Optional[int] = None,
                                    distribution: str = 'exponential',
                                    ) -> List[np.ndarray]:
    """
    Estimate distribution parameters for each sample.

    Expects X to already have NaN values where data was filtered out by
    build_classification_dataset (quantile threshold + shift applied at build time).

    Args:
        X:              Input array of shape (N, T, D, W, W).
        max_time_steps: Optional limit on number of time steps.
        distribution:   ``'exponential'`` (returns (T,1) λ-rate arrays) or
                        ``'weibull'``    (returns (T,2) [k, λ_scale] arrays).

    Returns:
        List of parameter arrays per sample.
        Shape (T, 1) for exponential, (T, 2) for Weibull.
    """
    N, T, D, W1, W2 = X.shape
    if max_time_steps is not None:
        T = min(T, max_time_steps)

    dist = distribution.lower()
    if dist == 'exponential':
        estimator = LogCumulant(distribution='exponential')
        d_params = 1
    elif dist == 'weibull':
        from sdtw.wasserstein_fast import estimate_weibull_fast
        d_params = 2
    else:
        raise ValueError(f"distribution must be 'exponential' or 'weibull', got '{distribution}'")

    params_list = []

    for i in range(N):
        params = np.zeros((T, d_params), dtype=np.float64)
        for t in range(T):
            values = X[i, t, :, :, :].flatten()
            values = values[np.isfinite(values) & (values > 0)]
            if len(values) >= 5:
                if dist == 'exponential':
                    estimator.fit(values)
                    params[t, 0] = estimator.get_params()
                else:
                    k_hat, lam_hat = estimate_weibull_fast(values.astype(np.float64))
                    params[t, 0] = k_hat
                    params[t, 1] = lam_hat
            else:
                params[t, :] = np.nan

        # Fill NaN time steps with column-wise mean, or a safe fallback.
        for col in range(d_params):
            valid_mask = np.isfinite(params[:, col])
            if valid_mask.sum() > 0 and (~valid_mask).sum() > 0:
                params[~valid_mask, col] = params[valid_mask, col].mean()
            elif (~valid_mask).sum() > 0:
                params[~valid_mask, col] = 1.0  # safe fallback

        params_list.append(params)

    return params_list

