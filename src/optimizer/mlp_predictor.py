"""
MLP predictor training with SGD for time series prediction using Soft-DTW.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory for sdtw access
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from sdtw import SoftDTW
from sdtw.distance import SquaredEuclidean, WassersteinDistance


class MLP:
    """
    Multi-Layer Perceptron with softplus activation.
    Architecture: input -> hidden -> output
    Activation: softplus(z) = log(1 + exp(z))
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, use_softplus_output=False, 
                 init_bias_value=None):
        """
        Parameters
        ----------
        input_dim : int
            Input dimension
        hidden_dim : int
            Hidden layer dimension
        output_dim : int
            Output dimension
        use_softplus_output : bool
            If True, apply softplus on output (to constrain positivity)
        init_bias_value : float or None
            Value to initialize the last layer bias
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_softplus_output = use_softplus_output
        
        # Xavier initialization for weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        
        # Smart initialization for b2 if a value is provided
        if init_bias_value is not None:
            # Add random noise between 0.95 and 1.05
            noise = np.random.uniform(0.95, 1.05, output_dim)
            self.b2 = np.ones(output_dim) * init_bias_value * noise
        else:
            self.b2 = np.zeros(output_dim)
        
        # Remember if the model uses Wasserstein (will be updated)
        self.is_wasserstein = False
        
    def softplus(self, z):
        """Softplus activation: log(1 + exp(z))"""
        # For numerical stability: if z > 20, softplus(z) ~ z
        return np.where(z > 20, z, np.log1p(np.exp(np.clip(z, -20, 20))))
    
    def softplus_derivative(self, z):
        """Derivative of softplus: sigmoid(z) = 1 / (1 + exp(-z))"""
        z_clip = np.clip(z, -20, 20)
        return np.where(z_clip >= 0,
                       1.0 / (1.0 + np.exp(-z_clip)),
                       np.exp(z_clip) / (1.0 + np.exp(z_clip)))
    
    def forward(self, x, save_cache=False):
        """
        Forward pass
        
        Parameters
        ----------
        x : array, shape (input_dim,)
            Input
        save_cache : bool
            If True, save intermediate values for backward
            
        Returns
        -------
        y : array, shape (output_dim,)
            Output
        """
        # Layer 1: z1 = W1 * x + b1
        self.z1 = np.dot(x, self.W1) + self.b1
        
        # Activation: h1 = softplus(z1)
        self.h1 = self.softplus(self.z1)
        
        # Layer 2: z2 = W2 * h1 + b2
        self.z2 = np.dot(self.h1, self.W2) + self.b2
        
        # Output with or without softplus
        if self.use_softplus_output:
            y = self.softplus(self.z2)
        else:
            y = self.z2
        
        if save_cache:
            self.x = x
            self.y = y
            
        return y
    
    def backward(self, grad_output):
        """
        Backward pass
        
        Parameters
        ----------
        grad_output : array, shape (output_dim,)
            Gradient of loss with respect to output
            
        Returns
        -------
        grads : dict
            Dictionary containing gradients for W1, b1, W2, b2
        """
        # Gradient with respect to z2
        if self.use_softplus_output:
            # dy/dz2 = softplus'(z2)
            dz2 = grad_output * self.softplus_derivative(self.z2)
        else:
            dz2 = grad_output
        
        # Gradients for W2 and b2
        dW2 = np.outer(self.h1, dz2)
        db2 = dz2
        
        # Gradient with respect to h1
        dh1 = np.dot(dz2, self.W2.T)
        
        # Gradient with respect to z1
        # dh1/dz1 = softplus'(z1)
        dz1 = dh1 * self.softplus_derivative(self.z1)
        
        # Gradients for W1 and b1
        dW1 = np.outer(self.x, dz1)
        db1 = dz1
        
        return {
            'W1': dW1,
            'b1': db1,
            'W2': dW2,
            'b2': db2
        }


def sgd_predictor(X_train, Y_train, hidden_dim=64, gamma=1.0, learning_rate=0.01,
                  num_epochs=100, batch_size=32, lr_decay=0.95, grad_clip=5.0,
                  use_softplus_output=False, loss_type='euclidean', verbose=True):
    """
    Train an MLP for time series prediction with Soft-DTW.
    
    Parameters
    ----------
    X_train : array, shape (n_samples, input_dim)
        Input data
    Y_train : array, shape (n_samples, output_dim)
        Targets (series to predict)
    hidden_dim : int
        Hidden layer dimension
    gamma : float
        Soft-DTW regularization parameter
    learning_rate : float
        Learning rate
    num_epochs : int
        Number of epochs
    batch_size : int
        Mini-batch size
    lr_decay : float
        Learning rate decay factor
    grad_clip : float
        Gradient clipping value
    use_softplus_output : bool
        If True, apply softplus on output (positivity constraint)
    loss_type : str
        'euclidean' for MSE, 'sdtw' for Soft-DTW, 'sdtw_wasserstein' for Soft-DTW Wasserstein
    verbose : bool
        Show progress
        
    Returns
    -------
    mlp : MLP
        Trained model
    losses : list
        Loss history
    """
    import time as time_module
    
    n_samples, input_dim = X_train.shape
    output_dim = Y_train.shape[1]
    
    # Compute mean of last 10 values of X_train for smart initialization
    n_last_values = min(10, input_dim)
    last_values = X_train[:, -n_last_values:]  # (n_samples, n_last_values)
    init_bias_value = np.mean(last_values)
    
    print(f"Smart initialization: mean of last {n_last_values} values = {init_bias_value:.4f}")
    
    # Initialize MLP with this value
    mlp = MLP(input_dim, hidden_dim, output_dim, use_softplus_output, 
              init_bias_value=init_bias_value)
    mlp.is_wasserstein = (loss_type == 'sdtw_wasserstein')
    
    losses = []
    current_lr = learning_rate
    
    # Warmup
    warmup_epochs = max(1, min(10, num_epochs // 10))
    
    print(f"Starting training: {num_epochs} epochs, batch_size={batch_size}")
    print(f"Architecture: {input_dim} -> {hidden_dim} -> {output_dim}")
    print(f"Loss type: {loss_type}, Softplus output: {use_softplus_output}")
    
    for epoch in range(num_epochs):
        epoch_start = time_module.time()
        
        # Learning rate schedule
        if epoch > warmup_epochs:
            current_lr = learning_rate * (lr_decay ** (epoch - warmup_epochs))
        else:
            current_lr = learning_rate * (epoch + 1) / warmup_epochs
        
        # Shuffle data
        indices = np.random.permutation(n_samples)
        epoch_loss = 0
        n_batches = 0
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            batch_size_actual = len(batch_indices)
            
            # Gradient accumulators
            grad_W1 = np.zeros_like(mlp.W1)
            grad_b1 = np.zeros_like(mlp.b1)
            grad_W2 = np.zeros_like(mlp.W2)
            grad_b2 = np.zeros_like(mlp.b2)
            batch_loss = 0
            
            # For each example in batch
            for idx in batch_indices:
                x = X_train[idx]
                y_true = Y_train[idx].reshape(-1, 1)
                
                # Forward pass
                y_pred = mlp.forward(x, save_cache=True)
                y_pred = y_pred.reshape(-1, 1)
                
                # Compute loss and gradient
                if loss_type == 'euclidean':
                    # MSE loss
                    diff = y_pred - y_true
                    loss = 0.5 * np.sum(diff ** 2)
                    grad_output = diff.flatten()
                    
                elif loss_type == 'sdtw':
                    # Soft-DTW with Euclidean distance
                    D = SquaredEuclidean(y_pred, y_true)
                    sdtw = SoftDTW(D, gamma=gamma)
                    loss = sdtw.compute()
                    
                    # Gradient via jacobian_product
                    E = sdtw.grad()
                    grad_output = D.jacobian_product(E).flatten()
                    
                elif loss_type == 'sdtw_wasserstein':
                    # Soft-DTW with Wasserstein distance
                    if use_softplus_output:
                        y_pred_values = np.maximum(y_pred, 1e-6)
                    else:
                        y_pred_clipped = np.clip(y_pred, -10, 10)
                        y_pred_values = mlp.softplus(y_pred_clipped)
                        y_pred_values = np.maximum(y_pred_values, 1e-6)
                    
                    y_true_safe = np.maximum(y_true, 1e-6)
                    
                    # Convert to lambda parameters = 1/mu for Wasserstein distance
                    y_pred_lambda = 1.0 / (y_pred_values + 1e-3)
                    y_true_lambda = 1.0 / (y_true_safe + 1e-3)
                    
                    # Clip to avoid extremes
                    y_pred_lambda = np.clip(y_pred_lambda, 1e-6, 1e6)
                    y_true_lambda = np.clip(y_true_lambda, 1e-6, 1e6)
                    
                    if np.any(np.isnan(y_pred_lambda)) or np.any(np.isinf(y_pred_lambda)):
                        continue
                    
                    try:
                        D = WassersteinDistance(y_pred_lambda, y_true_lambda,
                                              distribution='exponential',
                                              precompute_params=True,
                                              X_is_params=True,
                                              Y_is_params=True)
                        sdtw = SoftDTW(D, gamma=gamma)
                        loss = sdtw.compute()
                        
                        if np.isnan(loss) or np.isinf(loss):
                            continue
                        
                        # Gradient
                        E = sdtw.grad()
                        E_clip = np.clip(E, -10, 10)
                        grad_lambda = D.jacobian_product(E_clip).flatten()
                        
                        # Chain rule: d/d(values) = d/d(lambda) * d(lambda)/d(values)
                        # lambda = 1/(mu + eps), so d(lambda)/d(mu) = -1/(mu + eps)^2
                        grad_values = -grad_lambda / ((y_pred_values.flatten() + 1e-3) ** 2)
                        
                    except Exception as e:
                        if verbose:
                            print(f"Warning: Exception in Wasserstein computation: {e}")
                        continue
                    
                    # Normalize gradient
                    grad_norm = np.linalg.norm(grad_values)
                    if grad_norm > 1.0:
                        grad_values = grad_values / grad_norm
                    
                    # Chain rule if no softplus on output
                    if not use_softplus_output:
                        # d(values)/dz = softplus'(z) = sigmoid(z)
                        grad_output = grad_values * mlp.softplus_derivative(y_pred.flatten())
                    else:
                        grad_output = grad_values
                else:
                    raise ValueError(f"Unknown loss_type: {loss_type}")
                
                # Check for NaN
                if np.isnan(loss) or np.any(np.isnan(grad_output)):
                    if verbose:
                        print(f"Warning: NaN detected at epoch {epoch+1}, skipping sample")
                    continue
                
                # Backward pass
                grads = mlp.backward(grad_output)
                
                # Accumulate gradients
                grad_W1 += grads['W1']
                grad_b1 += grads['b1']
                grad_W2 += grads['W2']
                grad_b2 += grads['b2']
                
                batch_loss += loss
            
            # Average gradients over batch
            grad_W1 /= batch_size_actual
            grad_b1 /= batch_size_actual
            grad_W2 /= batch_size_actual
            grad_b2 /= batch_size_actual
            
            # Gradient clipping
            for grad in [grad_W1, grad_b1, grad_W2, grad_b2]:
                grad_norm = np.linalg.norm(grad)
                if grad_norm > grad_clip:
                    grad *= grad_clip / grad_norm
            
            # Weight update
            mlp.W1 -= current_lr * grad_W1
            mlp.b1 -= current_lr * grad_b1
            mlp.W2 -= current_lr * grad_W2
            mlp.b2 -= current_lr * grad_b2
            
            epoch_loss += batch_loss / batch_size_actual
            n_batches += 1
        
        epoch_loss /= n_batches
        losses.append(epoch_loss)
        
        epoch_time = time_module.time() - epoch_start
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: Loss = {epoch_loss:.6f}, "
                  f"LR = {current_lr:.6f}, Time = {epoch_time:.3f}s")
        
        # Check divergence
        if np.isnan(epoch_loss) or np.isinf(epoch_loss):
            if verbose:
                print(f"Training diverged at epoch {epoch+1}")
            break
    
    return mlp, losses


def predict(mlp, X_test):
    """
    Make predictions with trained MLP.
    
    Parameters
    ----------
    mlp : MLP
        Trained model
    X_test : array, shape (n_samples, input_dim)
        Test data
        
    Returns
    -------
    Y_pred : array, shape (n_samples, output_dim)
        Predictions (with softplus applied if model uses Wasserstein)
    """
    n_samples = X_test.shape[0]
    Y_pred = np.zeros((n_samples, mlp.output_dim))
    
    for i in range(n_samples):
        Y_pred[i] = mlp.forward(X_test[i])
    
    # If model uses Wasserstein and use_softplus_output is False,
    # apply softplus to get positive lambda values
    if hasattr(mlp, 'is_wasserstein') and mlp.is_wasserstein and not mlp.use_softplus_output:
        Y_pred = mlp.softplus(Y_pred)
    
    return Y_pred
