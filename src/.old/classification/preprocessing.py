"""
Data preprocessing functions for classification.
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


def estimate_parameters_for_samples(X: np.ndarray, max_time_steps: Optional[int] = None
                                    ) -> List[np.ndarray]:
    """
    Estimate exponential distribution parameters for each sample.
    
    For each sample, estimates lambda parameter at each time step
    by pooling all spatial values.
    
    Args:
        X: Input array of shape (N, T, D, W, W)
        max_time_steps: Optional limit on number of time steps
        
    Returns:
        List of parameter arrays with shape (T, 1) for each sample
    """
    N, T, D, W1, W2 = X.shape
    
    if max_time_steps is not None:
        T = min(T, max_time_steps)
    
    estimator = LogCumulant(distribution='exponential')
    params_list = []
    
    for i in range(N):
        params = np.zeros((T, 1), dtype=np.float64)
        for t in range(T):
            # Pool all values at time step t
            values = X[i, t, :, :, :].flatten()
            values = values[~np.isnan(values) & (values > 0)]
            
            if len(values) > 0:
                estimator.fit(values)
                params[t, 0] = estimator.get_params()
            else:
                # Use a default value if no valid data
                params[t, 0] = 1.0
        
        # Replace any remaining NaN with mean of valid values
        valid_mask = ~np.isnan(params[:, 0])
        if valid_mask.sum() > 0 and (~valid_mask).sum() > 0:
            params[~valid_mask, 0] = params[valid_mask, 0].mean()
        elif (~valid_mask).sum() > 0:
            params[~valid_mask, 0] = 1.0
        
        params_list.append(params)
    
    return params_list
