"""
Exponential distribution sample generation.
"""

import numpy as np
from scipy.stats import expon


def generate_exponential_series(lambda_values, n_samples=10000):
    """
    Generate exponential distribution samples for given lambda values.

    Parameters
    ----------
    lambda_values : array-like
        List of lambda (rate) parameters for each time step
    n_samples : int
        Number of samples to generate per time step

    Returns
    -------
    samples : ndarray, shape (n_timesteps, n_samples)
        Generated samples for each time step
    lambda_estimated : ndarray, shape (n_timesteps,)
        Estimated lambda values from the generated samples
    """
    samples = np.array([expon.rvs(scale=1/lam, size=n_samples) for lam in lambda_values])
    lambda_estimated = np.array([1 / np.mean(samples[i]) for i in range(len(samples))])
    
    return samples, lambda_estimated


def generate_two_series_experiment(n_samples=10000):
    """
    Generate two series with different lengths for barycenter experiments.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate per time step

    Returns
    -------
    dict : Dictionary containing:
        - lambda_values1, lambda_values2: True lambda values
        - samples1, samples2: Generated samples
        - lambda_est1, lambda_est2: Estimated lambda values
    """
    lambda_values1 = [1, 10, 100, 100, 50, 10, 4, 4]
    lambda_values2 = [0.1, 0.1, 0.5, 0.5, 5, 20, 80, 5]
    
    samples1, lambda_est1 = generate_exponential_series(lambda_values1, n_samples)
    samples2, lambda_est2 = generate_exponential_series(lambda_values2, n_samples)
    
    return {
        'lambda_values1': lambda_values1,
        'lambda_values2': lambda_values2,
        'samples1': samples1,
        'samples2': samples2,
        'lambda_est1': lambda_est1,
        'lambda_est2': lambda_est2
    }
