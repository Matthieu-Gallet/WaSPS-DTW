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
