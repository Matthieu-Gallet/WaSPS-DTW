"""
SGD-based barycenter computation for Soft-DTW with Wasserstein distance.
"""

import numpy as np
import time as time_module
import sys
from pathlib import Path

# Add parent directory for sdtw access
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from sdtw import SoftDTW
from sdtw.distance import WassersteinDistance

# Import estimators
import sys
sys.path.insert(0, str(parent_dir / 'src'))
from estimator import MLE, LogCumulant


def sgd_barycenter(X, gamma=1e-2, learning_rate=0.01,
                   num_epochs=100, batch_size=1, tol=1e-6, verbose=True,
                   lr_decay=0.95, grad_clip=10.0, distribution="exponential",
                   use_softplus=False, barycenter_init_method='mean_lambda', warmup_epochs=10, seed=123,
                   estimator_method='log_cumulant', X_is_params=False):
    """
    Simple Stochastic Gradient Descent for Soft-DTW barycenter computation.

    Parameters
    ----------
    X : list
        List of time series (numpy arrays containing samples or parameters)
    gamma : float
        Soft-DTW regularization parameter
    learning_rate : float
        Learning rate for SGD
    num_epochs : int
        Number of epochs (passes through all data)
    batch_size : int
        Mini-batch size (1 = pure SGD, len(X) = batch GD)
    tol : float
        Tolerance for early stopping
    verbose : bool
        Print progress
    lr_decay : float
        Learning rate decay factor per epoch
    grad_clip : float
        Maximum gradient norm for clipping
    distribution : str
        Distribution type ('exponential' supported)
    use_softplus : bool
        If True, use softplus parameterization: lambda = log(1 + exp(z))
        This provides unbounded optimization while constraining lambda > 0
    barycenter_init_method : str
        Initialization method: 'ones', 'random', 'mean_lambda'
    warmup_epochs : int
        Number of warmup epochs with increasing learning rate
    seed : int, optional
        Random seed for reproducibility. If None, no seed is set.
    estimator_method : str
        Parameter estimation method: 'log_cumulant' (default) or 'mle'
    X_is_params : bool
        If True, X already contains parameters (skip estimation). Default False.

    Returns
    -------
    barycenter : array
        Optimized barycenter parameters
    losses : list
        Loss history
    """
    # Convert to list if needed
    if not isinstance(X, list):
        X = [X]

    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    n_samples = len(X)

    # Pre-estimate parameters for each series (only once at the beginning)
    start_estimate = time_module.time()
    
    X_params_list = []
    
    if X_is_params:
        # X already contains parameters, no estimation needed
        for series in X:
            if series.ndim == 1:
                X_params_list.append(series.reshape(-1, 1))
            else:
                X_params_list.append(series)
    else:
        # Select estimator
        if estimator_method == 'mle':
            estimator_class = MLE
        elif estimator_method == 'log_cumulant':
            estimator_class = LogCumulant
        else:
            raise ValueError(f"Unknown estimator_method: {estimator_method}. Use 'mle' or 'log_cumulant'.")
        
        for series in X:
            if series.shape[1] > 1:
                # Series contains samples, estimate parameters per time step
                if distribution == 'exponential':
                    # Use selected estimator (automatically handles 2D data)
                    estimator = estimator_class(distribution=distribution)
                    estimator.fit(series)
                    params = estimator.get_params().reshape(-1, 1)
                else:  
                    raise NotImplementedError("Not implemented yet")
                X_params_list.append(params)
            else:
                # Series already contains parameters
                X_params_list.append(series)
    
    if verbose:
        print(f"  [DEBUG] Parameter estimation: {time_module.time() - start_estimate:.3f}s")
        print(f"  [DEBUG] Number of series: {len(X_params_list)}, First series shape: {X[0].shape}")
        print(f"  [DEBUG] Params shape: {X_params_list[0].shape}")

    # Initialize barycenter if not provided
    n_time_steps = X_params_list[0].shape[0]
    if barycenter_init_method == 'ones':
        barycenter_init = np.ones((n_time_steps, 1))
    elif barycenter_init_method == 'random':
        barycenter_init = np.random.uniform(0.1, 2.0, (n_time_steps, 1))
    elif barycenter_init_method == 'mean_lambda':
        # Compute mean of all lambda series, handling different lengths
        barycenter_init = np.mean(np.array([x.flatten() for x in X_params_list]), axis=0).reshape(-1, 1)

    # Initialize barycenter - choose parameterization
    Z_init = barycenter_init.copy().astype(np.float64)
    if verbose:
        print(f"  [DEBUG] Barycenter init shape: {Z_init.shape}, values (first 5): {Z_init.flatten()[:5]}")
    losses = []
    current_lr = learning_rate

    if use_softplus:
        # Softplus parameterization: optimize in z-space where lambda = softplus(z)
        # Convert initial lambda values to z-space for optimization
        transformed_Z = np.log(np.expm1(np.maximum(Z_init, 1e-8)))
    else:
        # Direct parameterization: optimize directly in lambda-space
        transformed_Z = Z_init.copy()

    # Warmup: start with smaller learning rate
    warmup_epochs = max(1, min(warmup_epochs, num_epochs // 10))
    if verbose:
        print(f"  [DEBUG] Starting optimization: {num_epochs} epochs, batch_size={batch_size} and warmup={warmup_epochs} epochs")
    epoch_times = []

    for epoch in range(num_epochs):
        epoch_start = time_module.time()
        # Apply learning rate schedule
        if epoch > warmup_epochs:
            current_lr = learning_rate * (lr_decay ** (epoch - warmup_epochs))
        else:
            # Warmup phase: gradually increase learning rate
            current_lr = learning_rate * (epoch + 1) / warmup_epochs
        epoch_loss = 0

        # Shuffle data for each epoch
        indices = np.random.permutation(n_samples)

        batch_count = 0
        batch_times = []
        G_TRACK = []  # Track gradient norms for display
        for start_idx in range(0, n_samples, batch_size):
            batch_start = time_module.time()
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch_X = [X_params_list[i] for i in batch_indices]

            # Convert from transformed space to parameter space for distance computation
            if use_softplus:
                # lambda = softplus(z) = log(1 + exp(z))
                exp_z = np.exp(transformed_Z)
                Z = np.where(transformed_Z > 50, transformed_Z, np.log1p(exp_z))
            else:
                # Direct parameterization
                Z = transformed_Z

            # Compute gradient for this batch
            G = np.zeros_like(Z)
            batch_loss = 0
            dist_time = 0
            grad_time = 0
            G_TRACK = []
            for x in batch_X:
                # Z is barycenter (params), x is pre-estimated params
                t0 = time_module.time()
                D = WassersteinDistance(Z, x, distribution=distribution,
                                       precompute_params=True, X_is_params=True, Y_is_params=True)
                sdtw = SoftDTW(D, gamma=gamma)
                value = sdtw.compute()
                dist_time += time_module.time() - t0

                t0 = time_module.time()
                E = sdtw.grad()
                G_tmp = D.jacobian_product(E)
                grad_time += time_module.time() - t0

                G += G_tmp
                batch_loss += value

            # Average gradient over batch
            G /= len(batch_X)
            batch_loss /= len(batch_X)

            # Check for NaN/Inf in gradient
            if np.any(np.isnan(G)) or np.any(np.isinf(G)):
                if verbose:
                    print(f"Warning: NaN/Inf gradient at epoch {epoch+1}, skipping batch")
                continue
            
            # Apply chain rule based on parameterization
            if use_softplus:
                # lambda = softplus(z) = log(1 + exp(z))
                # d(lambda)/dz = sigmoid(z) = exp(z) / (1 + exp(z))
                sigmoid_z = 1 / (1 + np.exp(-transformed_Z))  # More stable form
                G_transformed = G * sigmoid_z

                # Gradient clipping
                grad_norm = np.linalg.norm(G_transformed)
                G_TRACK.append(grad_norm)
                if grad_norm > grad_clip:
                    G_transformed = G_transformed * (grad_clip / grad_norm)

                # Update in transformed space
                transformed_Z -= current_lr * G_transformed
            else:
                # Direct update with clipping
                grad_norm = np.linalg.norm(G)
                G_TRACK.append(grad_norm)
                if grad_norm > grad_clip:
                    G = G * (grad_clip / grad_norm)
                transformed_Z -= current_lr * G
                # Ensure positivity
                transformed_Z = np.maximum(transformed_Z, 1e-8)

            epoch_loss += batch_loss
            batch_times.append(time_module.time() - batch_start)
            batch_count += 1
            if verbose:
                if epoch == 0 and batch_count == 1:
                    print(f"  [DEBUG] First batch - Distance: {dist_time:.3f}s, "
                          f"Gradient: {grad_time:.3f}s, Total: {batch_times[-1]:.3f}s")

        epoch_loss /= (n_samples // batch_size + 1)
        losses.append(epoch_loss)
        epoch_times.append(time_module.time() - epoch_start)
        if verbose:
            if epoch == 0:
                print(f"  [DEBUG] Epoch 1: {epoch_times[-1]:.3f}s, {batch_count} batches, "
                      f"Mean time/batch: {np.mean(batch_times):.3f}s")

        # Get current Z for display
        if verbose:
            if use_softplus:
                exp_z = np.exp(transformed_Z)
                Z_display = np.where(transformed_Z > 50, transformed_Z, np.log1p(exp_z))
            else:
                Z_display = transformed_Z

            if (epoch + 1) % 10 == 0:
                g_norm_display = np.mean(G_TRACK) if G_TRACK else 0.0
                print(f"Epoch {epoch+1:3d}: Loss = {epoch_loss:.6f}, "
                    f"LR = {current_lr:.6f}, ||Z|| = {np.linalg.norm(Z_display):.3f}, ||G|| = {g_norm_display:.3f}")  

        # Check for divergence
        if np.isnan(epoch_loss) or np.isinf(epoch_loss):
            if verbose:
                print(f"Diverged at epoch {epoch+1}")
            break

        # Early stopping check
        if epoch > 10 and len(losses) > 1:
            if abs(losses[-1] - losses[-2]) < tol:
                if verbose:
                    print(f"Converged after {epoch+1} epochs")
                break

    # Return final parameters in original space
    if verbose:
        print(f"  [DEBUG] Total epochs: {len(epoch_times)}, "
              f"Mean time/epoch: {np.mean(epoch_times):.3f}s, Total: {sum(epoch_times):.3f}s")
    
    if use_softplus:
        exp_z = np.exp(transformed_Z)
        Z_final = np.where(transformed_Z > 50, transformed_Z, np.log1p(exp_z))
    else:
        Z_final = transformed_Z

    return Z_final, losses
