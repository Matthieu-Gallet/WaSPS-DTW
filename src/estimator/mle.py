"""
Maximum Likelihood Estimation (MLE) for distributions.
"""

import numpy as np


class MLE:
    """
    Maximum Likelihood Estimator for distributions.
    
    Currently supports exponential distribution.
    Automatically detects if input is a single sample set or a time series (2D).
    
    For exponential distribution:
    - PDF: f(x; lambda) = lambda * exp(-lambda * x)
    - MLE: lambda_hat = 1 / mean(x)
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
        Estimate parameters from data.
        
        Parameters
        ----------
        data : array-like
            Observed samples. Can be 1D (single time step) or 2D (time series)
            
        Returns
        -------
        self : MLE
            The fitted estimator
        """
        data = np.asarray(data, dtype=np.float64)
        
        # Detect if it's a time series (2D) or single sample set (1D)
        if data.ndim == 2:
            self.is_series = True
            n_timesteps, n_samples_per_step = data.shape
            self.n_samples = n_samples_per_step
            
            # Estimate for each time step
            params = np.zeros(n_timesteps)
            for t in range(n_timesteps):
                timestep_data = data[t]
                timestep_data = timestep_data[~np.isnan(timestep_data) & (timestep_data > 0)]
                if len(timestep_data) > 0:
                    params[t] = 1.0 / np.mean(timestep_data)
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
            self.params = 1.0 / np.mean(data)
        
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
        estimator = MLE(distribution)
        estimator.fit(data)
        return estimator.params
