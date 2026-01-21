"""
Data preprocessing functions for train/test splitting and windowing.
"""

import numpy as np


def split_train_test(series_list, train_ratio=0.8, predict_length=90, random_seed=42):
    """
    Split data into train/test sets.

    Args:
        series_list: List of time series
        train_ratio: Training ratio
        predict_length: Number of values to predict
        random_seed: Random seed

    Returns:
        tuple: (X_train, Y_train, X_test, Y_test) in float32 for Chainer compatibility
               (loss computations use float64 internally for stability)
    """
    np.random.seed(random_seed)

    series_array = np.array(series_list)
    n_series = len(series_array)
    indices = np.random.permutation(n_series)
    series_array = series_array[indices]

    n_train = int(n_series * train_ratio)
    train_series = series_array[:n_train]
    test_series = series_array[n_train:]

    series_length = series_array.shape[1]
    if series_length < predict_length:
        print(f"Error: Series too short ({series_length}) to predict {predict_length} values")
        return None, None, None, None

    input_length = series_length - predict_length

    # Use float32 for Chainer (loss functions convert to float64 internally)
    X_train = np.float32(train_series[:, :input_length])
    Y_train = np.float32(train_series[:, input_length:])
    X_test = np.float32(test_series[:, :input_length])
    Y_test = np.float32(test_series[:, input_length:])

    print(f"\nData split:")
    print(f"  Train: {len(X_train)} series")
    print(f"  Test: {len(X_test)} series")
    print(f"  Input length: {input_length}")
    print(f"  Predict length: {predict_length}")
    print(f"  Y_train range: [{Y_train.min():.4f}, {Y_train.max():.4f}]")

    return X_train, Y_train, X_test, Y_test


def create_sliding_windows(X, Y, input_window_length, output_window_length):
    """
    Split time series into non-overlapping windows.
    
    Parameters
    ----------
    X : array, shape (n_samples, time_steps)
        Input data
    Y : array, shape (n_samples, time_steps)
        Output data
    input_window_length : int
        Input window length (e.g., 120)
    output_window_length : int
        Output window length (e.g., 30)
        
    Returns
    -------
    X_windowed : array, shape (n_windows, input_window_length)
    Y_windowed : array, shape (n_windows, output_window_length)
    """
    n_samples = X.shape[0]
    total_window = input_window_length + output_window_length
    
    # Concatenate X and Y for each sample
    XY_concat = np.concatenate([X, Y], axis=1)
    
    X_windows = []
    Y_windows = []
    
    for i in range(n_samples):
        series = XY_concat[i]  # Complete series for this sample
        
        # Split into non-overlapping windows
        for start_idx in range(0, len(series) - total_window + 1, total_window):
            end_idx = start_idx + total_window
            window = series[start_idx:end_idx]
            
            if len(window) == total_window:
                X_windows.append(window[:input_window_length])
                Y_windows.append(window[input_window_length:input_window_length + output_window_length])
    
    return np.array(X_windows), np.array(Y_windows)
