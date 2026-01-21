"""
Shifted time series generation for barycenter experiments.
"""

import numpy as np
from scipy.stats import expon


def generate_shifted_series(n_series=4, n_samples=1000, random_seed=42):
    """
    Generate temporally shifted series with individual fluctuations.
    Based on lambda values from previous experiments, extended to 47 timesteps.
    
    This experiment highlights the difference between Wasserstein and Euclidean:
    - Wasserstein is weighted by the product of lambdas: W^2(lambda_1, lambda_2) proportional to (1/lambda_1 - 1/lambda_2)^2
    - For lambda << 1 or lambda >> 10, the behavior differs significantly from Euclidean

    Parameters
    ----------
    n_series : int
        Number of series to generate
    n_samples : int
        Number of samples per time step
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    dict : Dictionary containing:
        - lambda_series: List of true lambda values for each series
        - samples_series: List of generated samples for each series
        - lambda_est_series: List of estimated lambda values
        - base_pattern: The base pattern used
    """
    np.random.seed(random_seed)
    
    # Base pattern with extended plateaus in low and high regimes
    # Structure: low plateau -> transition -> high plateau (extended) -> transition -> low plateau (extended) -> 3 rising peaks
    base_pattern = np.array([
        # Extended low plateau (weak regime lambda < 1)
        0.2, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5,
        # Rising transition
        1.0, 3.0, 8.0, 20.0,
        # Extended high plateau (strong regime lambda > 50)
        60.0, 70.0, 80.0, 80.0, 90.0, 100.0, 100.0, 100.0,
        # Descending transition
        50.0, 20.0, 5.0, 2.0,
        # Extended low plateau (10-12 values)
        0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5, 0.3, 0.3,
        # 10 new samples with 3 peaks rising to 20
        0.3, 0.4, 5.0, 15.0, 20.0, 18.0, 12.0, 3.0, 0.5, 0.4
    ])
    
    n_timesteps = len(base_pattern)
    
    lambda_series = []
    samples_series = []
    
    for i in range(n_series):
        # Reduced temporal shift: minimal circular shift
        shift = i * 1  # Shift of 1 timestep only (instead of 2)
        shifted_pattern = np.roll(base_pattern, shift)
        
        # Add individual fluctuations for each series
        # Reduced fluctuations to keep plateaus visible
        fluctuation = np.random.RandomState(42 + i).randn(n_timesteps) * 0.08
        # Multiplicative fluctuation (rather than additive) to respect scale
        lambda_values = shifted_pattern * (1 + fluctuation)
        
        # Series 4: vertical shift of +2 on the entire series
        if i == 3:
            lambda_values = lambda_values + 2.0
        
        # Ensure lambda remains positive
        lambda_values = np.maximum(lambda_values, 0.05)
        
        lambda_series.append(lambda_values)
    
    # Add peaks AFTER the shift - same positions for all series
    # Peaks in the high plateau (around indices 18-19)
    for i in range(n_series):
        if i == 1:
            # Series 2: 2 weak peaks in the middle of the high plateau (fixed indices)
            lambda_series[i][18] = 1.9
            lambda_series[i][19] = 3.5
            lambda_series[i][20] = 40.5
        
        if i == 2:
            # Series 3: 1 weak peak at the high plateau (fixed index)
            lambda_series[i][16] = 4.0
        if i == 3:
            # Series 4: 1 weak peak at the high plateau (fixed index)
            lambda_series[i][21] = 0.125
    
    # Peaks after index 30: only 2 extreme values (min and max)
    for i in range(n_series):
        if i < 3:  # Series 1, 2, 3
            lambda_series[i][39] = 0.2   # Minimum value
            lambda_series[i][40] = 0.2   # Minimum value
            lambda_series[i][41] = 20.0  # Maximum peak
            lambda_series[i][42] = 20.0  # Maximum peak
            lambda_series[i][43] = 0.5   # Return low
        else:  # Series 4 with shift +2
            lambda_series[i][39] = 2.2   # Minimum value + shift
            lambda_series[i][40] = 2.2   # Minimum value + shift
            lambda_series[i][41] = 22.0  # Maximum peak + shift
            lambda_series[i][42] = 22.0  # Maximum peak + shift
            lambda_series[i][43] = 2.5   # Return low + shift
    
    # Regenerate samples with added peaks
    samples_series = []
    for lambda_vals in lambda_series:
        samples = np.array([expon.rvs(scale=1/lam, size=n_samples) for lam in lambda_vals])
        samples_series.append(samples)
    
    # Estimate parameters for each series
    lambda_est_series = []
    for samples in samples_series:
        lambda_est = np.array([1 / np.mean(samples[i]) for i in range(len(samples))])
        lambda_est_series.append(lambda_est)
    
    return {
        'lambda_series': lambda_series,
        'samples_series': samples_series,
        'lambda_est_series': lambda_est_series,
        'base_pattern': base_pattern,
        'n_timesteps': n_timesteps
    }
