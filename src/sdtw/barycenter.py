# Author: Mathieu Blondel
# License: Simplified BSD

import numpy as np

from scipy.optimize import minimize
from scipy.stats import expon, weibull_min

from sdtw import SoftDTW
from sdtw.distance import SquaredEuclidean, WassersteinDistance


def sdtw_barycenter(X, barycenter_init, gamma=1.0, weights=None,
                    method="L-BFGS-B", bounds=None, tol=1e-3, max_iter=50,
                    distance="euclidean", distribution=None, return_estimate=False, precompute_params=True):
    """
    Compute barycenter (time series averaging) under the soft-DTW geometry.

    Parameters
    ----------
    X: list
        List of time series, numpy arrays of shape [len(X[i]), d].

    barycenter_init: array, shape = [length, d]
        Initialization.
        For euclidean distance: shape = [length, d]
        For wasserstein distance: shape = [length, n_params]
        where n_params=1 for exponential, n_params=2 for weibull.

    gamma: float
        Regularization parameter.
        Lower is less smoothed (closer to true DTW).

    weights: None or array
        Weights of each X[i]. Must be the same size as len(X).

    method: string
        Optimization method, passed to `scipy.optimize.minimize`.
        Default: L-BFGS.
    
    bounds: list of tuple, optional
        Bounds for variables (only for L-BFGS-B, TNC, SLSQP, Powell, and trust-constr).
        Not used for other methods.

    tol: float
        Tolerance of the method used.

    max_iter: int
        Maximum number of iterations.
        
    distance: str
        Distance type: 'euclidean' (default) or 'wasserstein'.
        
    distribution: str or None
        Distribution family for Wasserstein distance: 'exponential' or 'weibull'.
        Required when distance='wasserstein'.
        
    return_estimate: bool
        Only used when distance='wasserstein'.
        If True, returns parameter estimates [length, n_params].
        If False, generates samples from the estimated distributions
        with shape matching barycenter_init.

    Returns
    -------
    barycenter: array
        For euclidean: shape = [length, d]
        For wasserstein with return_estimate=True: shape = [length, n_params]
        For wasserstein with return_estimate=False: shape = [length, n_samples]
    """
    if weights is None:
        weights = np.ones(len(X))

    weights = np.array(weights)
    
    # Validate distance parameter
    distance = distance.lower()
    if distance not in ['euclidean', 'wasserstein']:
        raise ValueError(f"distance must be 'euclidean' or 'wasserstein', got '{distance}'")
    
    # Validate distribution parameter for Wasserstein
    if distance == 'wasserstein':
        if distribution is None:
            raise ValueError("distribution must be specified when distance='wasserstein'")
        distribution = distribution.lower()
        if distribution not in ['exponential', 'weibull']:
            raise ValueError(f"distribution must be 'exponential' or 'weibull', got '{distribution}'")
        
        # Pre-estimate parameters for each series in X (only once)
        # This avoids re-estimating at each optimization step
        X_params_list = []
        for series in X:
            if precompute_params and series.shape[1] > 1:
                # Series contains samples, estimate parameters
                from .wasserstein_fast import estimate_exponential_fast, estimate_weibull_fast
                if distribution == 'exponential':
                    params = np.array([estimate_exponential_fast(series[t]) for t in range(series.shape[0])]).reshape(-1, 1)
                else:  # weibull
                    params = np.array([estimate_weibull_fast(series[t]) for t in range(series.shape[0])])
                X_params_list.append(params)
            else:
                # Series already contains parameters
                X_params_list.append(series)

    def _func(Z):
        # Compute objective value and grad at Z.

        Z = Z.reshape(*barycenter_init.shape)

        m = Z.shape[0]
        G = np.zeros_like(Z)

        obj = 0

        for i in range(len(X)):
            if distance == 'euclidean':
                D = SquaredEuclidean(Z, X[i])
            else:  # wasserstein
                # Z is the barycenter (contains params), X_params_list[i] contains estimated params
                D = WassersteinDistance(Z, X_params_list[i], distribution=distribution, 
                                       precompute_params=True, X_is_params=True, Y_is_params=True)
            sdtw = SoftDTW(D, gamma=gamma)
            value = sdtw.compute()
            E = sdtw.grad()
            G_tmp = D.jacobian_product(E)
            G += weights[i] * G_tmp
            obj += weights[i] * value

        return obj, G.ravel()

    # The function works with vectors so we need to vectorize barycenter_init.
    res = minimize(_func, barycenter_init.ravel(), method=method, jac=True, bounds=bounds,
                   tol=tol, options=dict(maxiter=max_iter, disp=False))

    # Return result based on distance type
    if distance == 'euclidean':
        return res.x.reshape(*barycenter_init.shape)
    else:  # wasserstein
        if return_estimate:
            # Return parameter estimates
            return res.x.reshape(*barycenter_init.shape)
        else:
            # Generate samples from estimated distributions
            params = res.x.reshape(*barycenter_init.shape)
            n_timesteps = params.shape[0]
            n_samples = X[0].shape[1]  # Use same number of samples as input
            
            samples = np.zeros((n_timesteps, n_samples))
            
            if distribution == 'exponential':
                for i in range(n_timesteps):
                    lambda_param = params[i, 0]
                    samples[i, :] = expon.rvs(scale=1.0/lambda_param, size=n_samples)
            else:  # weibull
                for i in range(n_timesteps):
                    k_param = params[i, 0]
                    lambda_param = params[i, 1]
                    samples[i, :] = weibull_min.rvs(k_param, scale=lambda_param, size=n_samples)
            
            return samples
