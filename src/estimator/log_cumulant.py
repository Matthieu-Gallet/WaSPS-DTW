"""
Log-cumulant based estimation for distributions.

Currently supports exponential distribution with Cython acceleration.
Automatically detects if input is a single sample set or a time series (2D).
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory for sdtw access
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Import Cython implementation (required)
from sdtw.wasserstein_fast import estimate_exponential_fast as _estimate_exponential_cython


class LogCumulant:
    """
    Log-cumulant based estimator for distributions.
    
    Currently supports exponential distribution with Cython acceleration.
    Automatically detects if input is a single sample set or a time series (2D).
    
    For exponential distribution:
    - Uses the relationship: lambda = 1 / exp(E[log(X)] + gamma)
    - where gamma is the Euler-Mascheroni constant
    """
    
    def __init__(self, distribution='exponential'):
        if distribution not in ['exponential']:
            raise ValueError(f"Unsupported distribution: {distribution}")
        self.distribution = distribution
        self.params = None
        self.n_samples = 0
        self.is_series = False
    
    def fit(self, data):
        """
        Estimate parameters from data using log-cumulants.
        
        Parameters
        ----------
        data : array-like
            Observed samples. Can be 1D (single time step) or 2D (time series)
            
        Returns
        -------
        self : LogCumulant
            The fitted estimator
        """
        data = np.asarray(data, dtype=np.float64)
        
        # Detect if it's a time series (2D) or single sample set (1D)
        if data.ndim == 2:
            self.is_series = True
            n_timesteps, n_samples_per_step = data.shape
            self.n_samples = n_samples_per_step
            
            # Estimate for each time step using Cython
            params = np.zeros(n_timesteps)
            for t in range(n_timesteps):
                timestep_data = data[t]
                timestep_data = timestep_data[~np.isnan(timestep_data) & (timestep_data > 0)]
                if len(timestep_data) > 0:
                    params[t] = _estimate_exponential_cython(timestep_data)
                else:
                    params[t] = np.nan
            
            self.params = params
            
        else:  # 1D array
            self.is_series = False
            data = data.flatten()
            data = data[~np.isnan(data) & (data > 0)]
            
            if len(data) == 0:
                raise ValueError("No valid data points for estimation")
            
            self.n_samples = len(data)
            self.params = _estimate_exponential_cython(data)
        
        return self
    
    def get_params(self):
        """
        Get the estimated parameters.
        
        Returns
        -------
        array or float : Estimated parameters
        """
        return self.params
    
    @staticmethod
    def estimate(data, distribution='exponential'):
        """
        Static method for quick estimation.
        
        Parameters
        ----------
        data : array-like
            Observed samples
        distribution : str
            Distribution type (currently only 'exponential')
            
        Returns
        -------
        array or float : Estimated parameters
        """
        estimator = LogCumulant(distribution)
        estimator.fit(data)
        return estimator.params
