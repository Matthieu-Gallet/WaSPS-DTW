"""
Utility functions for optimal histogram binning.

This module provides functions to compute optimal bin configurations
for histogram visualization, ensuring numerical stability and meaningful
bin ranges.
"""

import numpy as np


def get_optimal_bins(data, min_bins=10, max_bins=100):
    """
    Compute optimal number of bins for a histogram based on data.
    
    Uses the Freedman-Diaconis rule for bin width, with fallback to
    Sturges' rule for small datasets.
    
    Parameters
    ----------
    data : array-like
        Input data to compute bins for.
    min_bins : int, optional
        Minimum number of bins (default: 10).
    max_bins : int, optional
        Maximum number of bins (default: 100).
    
    Returns
    -------
    int
        Optimal number of bins, clamped between min_bins and max_bins.
    
    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.exponential(2.0, 1000)
    >>> n_bins = get_optimal_bins(data)
    >>> print(f"Optimal bins: {n_bins}")
    """
    data = np.asarray(data).ravel()
    data = data[~np.isnan(data)]  # Remove NaN values
    
    n = len(data)
    if n < 2:
        return min_bins
    
    # Freedman-Diaconis rule
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    
    if iqr > 0:
        # Freedman-Diaconis bin width
        bin_width = 2 * iqr / (n ** (1/3))
        data_range = data.max() - data.min()
        
        if data_range > 0 and bin_width > 0:
            n_bins = int(np.ceil(data_range / bin_width))
        else:
            # Fallback to Sturges' rule
            n_bins = int(np.ceil(np.log2(n) + 1))
    else:
        # Fallback to Sturges' rule for data with no IQR spread
        n_bins = int(np.ceil(np.log2(n) + 1))
    
    return max(min_bins, min(n_bins, max_bins))


def compute_histogram_range(data, percentile_range=(1, 99)):
    """
    Compute a robust histogram range based on percentiles.
    
    Parameters
    ----------
    data : array-like
        Input data.
    percentile_range : tuple, optional
        Lower and upper percentiles for range (default: (1, 99)).
    
    Returns
    -------
    tuple
        (min_value, max_value) for histogram range.
    """
    data = np.asarray(data).ravel()
    data = data[~np.isnan(data)]
    
    if len(data) == 0:
        return (0, 1)
    
    low, high = np.percentile(data, percentile_range)
    return (low, high)
