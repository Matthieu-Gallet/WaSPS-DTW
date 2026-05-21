"""
Data loading and preprocessing functions for classification datasets.

This module provides functions for:
- Loading classification datasets (X, Y, metadata)
- Preprocessing raw data for Soft-DTW computation
- Estimating distribution parameters from raw data
"""

import numpy as np
import pandas as pd
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


def load_class_thresholds(
    ks_summary_path: str,
    ws: int,
    tw: int,
    idx_to_regime: Dict[int, str],
) -> Dict[int, float]:
    """
    Load per-class optimal thresholds from a ks_summary CSV for a given (ws, tw) config.

    Args:
        ks_summary_path: Path to the ks_summary.csv produced by dev.py
        ws: Spatial window size to look up
        tw: Temporal window size to look up
        idx_to_regime: Mapping class_idx → regime_code (from dataset metadata)

    Returns:
        Dict mapping class_idx → threshold value.
        Classes missing from the summary are assigned threshold 0.0.
    """
    df = pd.read_csv(ks_summary_path)
    subset = df[(df["ws"] == ws) & (df["tw"] == tw)].dropna(subset=["threshold"])
    regime_to_thr = dict(zip(subset["regime"], subset["threshold"]))
    return {
        idx: float(regime_to_thr.get(code, 0.0))
        for idx, code in idx_to_regime.items()
    }


def estimate_parameters_for_samples(X: np.ndarray, max_time_steps: Optional[int] = None,
                                    Y: Optional[np.ndarray] = None,
                                    class_thresholds: Optional[Dict[int, float]] = None,
                                    ) -> List[np.ndarray]:
    """
    Estimate exponential distribution parameters for each sample.

    For each sample, estimates lambda parameter at each time step
    by pooling all spatial values.

    When `Y` and `class_thresholds` are provided, a per-class threshold is applied:
    values below the threshold are discarded and the remaining values are shifted
    to zero (``values -= threshold``) before fitting.

    Args:
        X: Input array of shape (N, T, D, W, W)
        max_time_steps: Optional limit on number of time steps
        Y: Class labels, shape (N,). Required when class_thresholds is given.
        class_thresholds: Dict mapping class_idx → threshold value.

    Returns:
        List of parameter arrays with shape (T, 1) for each sample
    """
    N, T, D, W1, W2 = X.shape
    
    if max_time_steps is not None:
        T = min(T, max_time_steps)
    
    estimator = LogCumulant(distribution='exponential')
    params_list = []
    
    for i in range(N):
        thr = 0.0
        if class_thresholds is not None and Y is not None:
            thr = class_thresholds.get(int(Y[i]), 0.0)

        params = np.zeros((T, 1), dtype=np.float64)
        for t in range(T):
            # Pool all values at time step t
            values = X[i, t, :, :, :].flatten()
            values = values[np.isfinite(values) & (values > thr)]

            if len(values) >= 5:
                values = values - thr
                estimator.fit(values)
                params[t, 0] = estimator.get_params()
            else:
                # Use a default value if too few valid data points
                params[t, 0] = np.nan
        
        # Replace any remaining NaN with mean of valid values
        valid_mask = ~np.isnan(params[:, 0])
        if valid_mask.sum() > 0 and (~valid_mask).sum() > 0:
            params[~valid_mask, 0] = params[valid_mask, 0].mean()
        elif (~valid_mask).sum() > 0:
            params[~valid_mask, 0] = 1.0
        
        params_list.append(params)
    
    return params_list
