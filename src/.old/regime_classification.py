#!/usr/bin/env python3
"""
Hydrological Regime Classification Experiment using Soft-DTW Barycenters.

This script compares three methods for regime classification:
1. Soft-DTW Euclidean on raw data
2. Soft-DTW Euclidean on exponential parameter estimates
3. Soft-DTW Wasserstein on exponential parameter estimates

Workflow:
1. Load classification dataset created by build_classification_dataset.py
2. Split into train/test sets
3. Compute barycenters for each class on training data
4. Classify test samples by minimum distance to barycenters
5. Calculate F1 scores and other metrics

Author: Generated script
Date: 2026-01-22
"""

import numpy as np
import time
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# Add parent directory for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from sdtw.barycenter import sdtw_barycenter
from sdtw import SoftDTW
from sdtw.distance import SquaredEuclidean, WassersteinDistance
from optimizer import sgd_barycenter
from estimator import LogCumulant
from utils import print_timing


# ============================================================================
# Data Preprocessing Functions
# ============================================================================

def load_classification_dataset(data_dir: str, mode: str = "basic"
                                 ) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load classification dataset created by build_classification_dataset.py.
    
    Args:
        data_dir: Directory containing the dataset files
        mode: "basic" or "balanced"
        
    Returns:
        Tuple of (X, Y, metadata)
    """
    data_path = Path(data_dir)
    
    X = np.load(data_path / f"X_{mode}.npy")
    Y = np.load(data_path / f"Y_{mode}.npy")
    metadata = np.load(data_path / f"metadata_{mode}.npy", allow_pickle=True).item()
    
    return X, Y, metadata


def preprocess_samples(X: np.ndarray, max_time_steps: Optional[int] = None
                       ) -> List[np.ndarray]:
    """
    Preprocess samples for Soft-DTW computation.
    
    Input X has shape (N, T, D, W, W) where:
        - N: number of samples
        - T: number of time groups
        - D: temporal window size
        - W: spatial window size
    
    Output: List of arrays with shape (T, D*W*W) for each sample
    (each time step contains D*W*W values flattened)
    
    Args:
        X: Input array of shape (N, T, D, W, W)
        max_time_steps: Optional limit on number of time steps to use
        
    Returns:
        List of preprocessed samples
    """
    N, T, D, W1, W2 = X.shape
    
    if max_time_steps is not None:
        T = min(T, max_time_steps)
    
    samples = []
    for i in range(N):
        # Reshape to (T, D*W*W)
        sample = X[i, :T, :, :, :].reshape(T, -1).astype(np.float64)
        # Replace NaN with 0 (or could use mean/interpolation)
        sample = np.nan_to_num(sample, nan=0.0)
        samples.append(sample)
    
    return samples


def estimate_parameters_for_samples(X: np.ndarray, max_time_steps: Optional[int] = None
                                    ) -> List[np.ndarray]:
    """
    Estimate exponential distribution parameters for each sample.
    
    For each sample, estimates lambda parameter at each time step
    by pooling all spatial values.
    
    Args:
        X: Input array of shape (N, T, D, W, W)
        max_time_steps: Optional limit on number of time steps
        
    Returns:
        List of parameter arrays with shape (T, 1) for each sample
    """
    N, T, D, W1, W2 = X.shape
    
    if max_time_steps is not None:
        T = min(T, max_time_steps)
    
    estimator = LogCumulant(distribution='exponential')
    params_list = []
    
    for i in range(N):
        params = np.zeros((T, 1), dtype=np.float64)
        for t in range(T):
            # Pool all values at time step t
            values = X[i, t, :, :, :].flatten()
            values = values[~np.isnan(values) & (values > 0)]
            
            if len(values) > 0:
                estimator.fit(values)
                params[t, 0] = estimator.get_params()
            else:
                # Use a default value if no valid data
                params[t, 0] = 1.0
        
        # Replace any remaining NaN with mean of valid values
        valid_mask = ~np.isnan(params[:, 0])
        if valid_mask.sum() > 0 and (~valid_mask).sum() > 0:
            params[~valid_mask, 0] = params[valid_mask, 0].mean()
        elif (~valid_mask).sum() > 0:
            params[~valid_mask, 0] = 1.0
        
        params_list.append(params)
    
    return params_list


# ============================================================================
# Barycenter Computation Functions
# ============================================================================

def compute_barycenter_euclidean_raw(samples: List[np.ndarray], gamma: float = 1.0,
                                      max_iter: int = 50) -> np.ndarray:
    """
    Compute Soft-DTW barycenter with Euclidean distance on raw data.
    
    Args:
        samples: List of arrays with shape (T, D*W*W)
        gamma: Soft-DTW regularization parameter
        max_iter: Maximum iterations for optimization
        
    Returns:
        Barycenter array with shape (T, D*W*W)
    """
    # Ensure all samples have the same shape
    min_len = min(s.shape[0] for s in samples)
    samples_aligned = [s[:min_len, :].astype(np.float64) for s in samples]
    
    # Initialize with mean
    bary_init = np.mean(np.array(samples_aligned), axis=0)
    
    # Compute barycenter
    barycenter = sdtw_barycenter(
        samples_aligned, bary_init, gamma=gamma,
        max_iter=max_iter, distance="euclidean"
    )
    
    return barycenter


def compute_barycenter_euclidean_params(params_list: List[np.ndarray], gamma: float = 1.0,
                                         max_iter: int = 100) -> np.ndarray:
    """
    Compute Soft-DTW barycenter with Euclidean distance on estimated parameters.
    
    Args:
        params_list: List of parameter arrays with shape (T, 1)
        gamma: Soft-DTW regularization parameter
        max_iter: Maximum iterations for optimization
        
    Returns:
        Barycenter parameters array with shape (T, 1)
    """
    # Initialize with mean
    bary_init = np.mean(params_list, axis=0)
    
    # Compute barycenter
    barycenter = sdtw_barycenter(
        params_list, bary_init, gamma=gamma,
        max_iter=max_iter, distance="euclidean"
    )
    
    return barycenter


def compute_barycenter_wasserstein_sgd(params_list: List[np.ndarray], gamma: float = 1.0,
                                        learning_rate: float = 0.075, num_epochs: int = 250,
                                        batch_size: int = 4) -> np.ndarray:
    """
    Compute Soft-DTW barycenter with Wasserstein distance using SGD.
    
    Args:
        params_list: List of parameter arrays with shape (T, 1)
        gamma: Soft-DTW regularization parameter
        learning_rate: SGD learning rate
        num_epochs: Number of SGD epochs
        batch_size: SGD batch size
        
    Returns:
        Barycenter parameters array with shape (T, 1)
    """
    # SGD barycenter expects parameters directly
    barycenter, _ = sgd_barycenter(
        params_list, gamma=gamma, barycenter_init_method='mean_lambda',
        learning_rate=learning_rate, num_epochs=num_epochs, batch_size=batch_size,
        lr_decay=0.995, grad_clip=100.0, distribution="exponential",
        verbose=False, use_softplus=True, X_is_params=True
    )
    
    return barycenter


# ============================================================================
# Distance Computation Functions
# ============================================================================

def compute_sdtw_distance_euclidean(sample: np.ndarray, barycenter: np.ndarray,
                                     gamma: float = 1.0) -> float:
    """
    Compute Soft-DTW distance with Euclidean distance.
    
    Args:
        sample: Sample array
        barycenter: Barycenter array
        gamma: Soft-DTW regularization parameter
        
    Returns:
        Soft-DTW distance value
    """
    D = SquaredEuclidean(sample, barycenter)
    sdtw = SoftDTW(D, gamma=gamma)
    return sdtw.compute()


def compute_sdtw_distance_wasserstein(params: np.ndarray, barycenter_params: np.ndarray,
                                       gamma: float = 1.0) -> float:
    """
    Compute Soft-DTW distance with Wasserstein distance on parameters.
    
    Args:
        params: Sample parameter array with shape (T, 1)
        barycenter_params: Barycenter parameter array with shape (T, 1)
        gamma: Soft-DTW regularization parameter
        
    Returns:
        Soft-DTW distance value
    """
    D = WassersteinDistance(
        params, barycenter_params, distribution='exponential',
        precompute_params=True, X_is_params=True, Y_is_params=True
    )
    sdtw = SoftDTW(D, gamma=gamma)
    return sdtw.compute()


# ============================================================================
# Classification Functions
# ============================================================================

def classify_by_nearest_barycenter(samples: List, barycenters: Dict[int, np.ndarray],
                                    distance_func, gamma: float = 1.0) -> np.ndarray:
    """
    Classify samples by nearest barycenter.
    
    Args:
        samples: List of samples (raw data or parameters)
        barycenters: Dictionary mapping class labels to barycenters
        distance_func: Function to compute distance (takes sample, barycenter, gamma)
        gamma: Soft-DTW regularization parameter
        
    Returns:
        Array of predicted class labels
    """
    predictions = []
    
    for sample in tqdm(samples, desc="Classifying", leave=False):
        min_distance = np.inf
        predicted_class = None
        
        for class_label, barycenter in barycenters.items():
            distance = distance_func(sample, barycenter, gamma)
            
            if distance < min_distance:
                min_distance = distance
                predicted_class = class_label
        
        predictions.append(predicted_class)
    
    return np.array(predictions)


def run_classification_experiment(X_train_raw: List[np.ndarray], X_train_params: List[np.ndarray],
                                   X_test_raw: List[np.ndarray], X_test_params: List[np.ndarray],
                                   Y_train: np.ndarray, Y_test: np.ndarray,
                                   idx_to_regime: Dict[int, str], gamma: float = 1.0,
                                   sgd_epochs: int = 100, sgd_lr: float = 0.05,
                                   verbose: bool = True) -> Dict:
    """
    Run the complete classification experiment with all three methods.
    
    Args:
        X_train_raw: Training samples (raw data)
        X_train_params: Training samples (parameters)
        X_test_raw: Test samples (raw data)
        X_test_params: Test samples (parameters)
        Y_train: Training labels
        Y_test: Test labels
        idx_to_regime: Mapping from label index to regime code
        gamma: Soft-DTW regularization parameter
        sgd_epochs: Number of epochs for SGD barycenter
        sgd_lr: Learning rate for SGD barycenter
        verbose: Print progress
        
    Returns:
        Dictionary with results for all methods
    """
    unique_classes = np.unique(Y_train)
    results = {}
    
    # =========================================================================
    # Method 1: Soft-DTW Euclidean on Raw Data
    # =========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("Method 1: Soft-DTW Euclidean on Raw Data")
        print("=" * 60)
    
    start_time = time.time()
    
    # Compute barycenters for each class
    barycenters_raw = {}
    for class_label in unique_classes:
        class_samples = [X_train_raw[i] for i in range(len(X_train_raw)) if Y_train[i] == class_label]
        if verbose:
            print(f"  Computing barycenter for class {idx_to_regime[class_label]} ({len(class_samples)} samples)...")
        barycenters_raw[class_label] = compute_barycenter_euclidean_raw(
            class_samples, gamma=gamma, max_iter=30
        )
    
    barycenter_time_raw = time.time() - start_time
    if verbose:
        print(f"  Barycenter computation time: {barycenter_time_raw:.2f}s")
    
    # Classify test samples
    start_time = time.time()
    Y_pred_raw = classify_by_nearest_barycenter(
        X_test_raw, barycenters_raw, compute_sdtw_distance_euclidean, gamma
    )
    classify_time_raw = time.time() - start_time
    
    # Calculate metrics
    f1_raw = f1_score(Y_test, Y_pred_raw, average='weighted')
    f1_macro_raw = f1_score(Y_test, Y_pred_raw, average='macro')
    
    results['euclidean_raw'] = {
        'predictions': Y_pred_raw,
        'f1_weighted': f1_raw,
        'f1_macro': f1_macro_raw,
        'barycenter_time': barycenter_time_raw,
        'classify_time': classify_time_raw,
        'barycenters': barycenters_raw
    }
    
    if verbose:
        print(f"  Classification time: {classify_time_raw:.2f}s")
        print(f"  F1 Score (weighted): {f1_raw:.4f}")
        print(f"  F1 Score (macro): {f1_macro_raw:.4f}")
    
    # =========================================================================
    # Method 2: Soft-DTW Euclidean on Estimated Parameters
    # =========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("Method 2: Soft-DTW Euclidean on Estimated Parameters")
        print("=" * 60)
    
    start_time = time.time()
    
    # Compute barycenters for each class
    barycenters_params_euc = {}
    for class_label in unique_classes:
        class_params = [X_train_params[i] for i in range(len(X_train_params)) if Y_train[i] == class_label]
        if verbose:
            print(f"  Computing barycenter for class {idx_to_regime[class_label]} ({len(class_params)} samples)...")
        barycenters_params_euc[class_label] = compute_barycenter_euclidean_params(
            class_params, gamma=gamma, max_iter=100
        )
    
    barycenter_time_params = time.time() - start_time
    if verbose:
        print(f"  Barycenter computation time: {barycenter_time_params:.2f}s")
    
    # Classify test samples
    start_time = time.time()
    Y_pred_params_euc = classify_by_nearest_barycenter(
        X_test_params, barycenters_params_euc, compute_sdtw_distance_euclidean, gamma
    )
    classify_time_params = time.time() - start_time
    
    # Calculate metrics
    f1_params_euc = f1_score(Y_test, Y_pred_params_euc, average='weighted')
    f1_macro_params_euc = f1_score(Y_test, Y_pred_params_euc, average='macro')
    
    results['euclidean_params'] = {
        'predictions': Y_pred_params_euc,
        'f1_weighted': f1_params_euc,
        'f1_macro': f1_macro_params_euc,
        'barycenter_time': barycenter_time_params,
        'classify_time': classify_time_params,
        'barycenters': barycenters_params_euc
    }
    
    if verbose:
        print(f"  Classification time: {classify_time_params:.2f}s")
        print(f"  F1 Score (weighted): {f1_params_euc:.4f}")
        print(f"  F1 Score (macro): {f1_macro_params_euc:.4f}")
    
    # =========================================================================
    # Method 3: Soft-DTW Wasserstein on Estimated Parameters
    # =========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("Method 3: Soft-DTW Wasserstein on Estimated Parameters (SGD)")
        print("=" * 60)
    
    start_time = time.time()
    
    # Compute barycenters for each class
    barycenters_wass = {}
    for class_label in unique_classes:
        class_params = [X_train_params[i] for i in range(len(X_train_params)) if Y_train[i] == class_label]
        if verbose:
            print(f"  Computing barycenter for class {idx_to_regime[class_label]} ({len(class_params)} samples)...")
        barycenters_wass[class_label] = compute_barycenter_wasserstein_sgd(
            class_params, gamma=gamma, learning_rate=sgd_lr, num_epochs=sgd_epochs, batch_size=4
        )
    
    barycenter_time_wass = time.time() - start_time
    if verbose:
        print(f"  Barycenter computation time: {barycenter_time_wass:.2f}s")
    
    # Classify test samples
    start_time = time.time()
    Y_pred_wass = classify_by_nearest_barycenter(
        X_test_params, barycenters_wass, compute_sdtw_distance_wasserstein, gamma
    )
    classify_time_wass = time.time() - start_time
    
    # Calculate metrics
    f1_wass = f1_score(Y_test, Y_pred_wass, average='weighted')
    f1_macro_wass = f1_score(Y_test, Y_pred_wass, average='macro')
    
    results['wasserstein_params'] = {
        'predictions': Y_pred_wass,
        'f1_weighted': f1_wass,
        'f1_macro': f1_macro_wass,
        'barycenter_time': barycenter_time_wass,
        'classify_time': classify_time_wass,
        'barycenters': barycenters_wass
    }
    
    if verbose:
        print(f"  Classification time: {classify_time_wass:.2f}s")
        print(f"  F1 Score (weighted): {f1_wass:.4f}")
        print(f"  F1 Score (macro): {f1_macro_wass:.4f}")
    
    return results


def print_detailed_results(results: Dict, Y_test: np.ndarray, idx_to_regime: Dict[int, str]):
    """
    Print detailed classification results for all methods.
    
    Args:
        results: Dictionary with results from run_classification_experiment
        Y_test: True test labels
        idx_to_regime: Mapping from label index to regime code
    """
    target_names = [idx_to_regime[i] for i in sorted(idx_to_regime.keys())]
    
    print("\n" + "=" * 80)
    print("DETAILED CLASSIFICATION RESULTS")
    print("=" * 80)
    
    methods = [
        ('euclidean_raw', 'Soft-DTW Euclidean (Raw Data)'),
        ('euclidean_params', 'Soft-DTW Euclidean (Parameters)'),
        ('wasserstein_params', 'Soft-DTW Wasserstein (Parameters)')
    ]
    
    for method_key, method_name in methods:
        if method_key in results:
            print(f"\n{'-' * 60}")
            print(f"{method_name}")
            print(f"{'-' * 60}")
            
            Y_pred = results[method_key]['predictions']
            print("\nClassification Report:")
            print(classification_report(Y_test, Y_pred, target_names=target_names, zero_division=0))
            
            print("Confusion Matrix:")
            cm = confusion_matrix(Y_test, Y_pred)
            # Print with class labels
            print(f"{'':>10}", end="")
            for name in target_names:
                print(f"{name:>8}", end="")
            print()
            for i, row in enumerate(cm):
                print(f"{target_names[i]:>10}", end="")
                for val in row:
                    print(f"{val:>8}", end="")
                print()


def save_results(results: Dict, output_dir: str, metadata: Dict):
    """
    Save classification results to files.
    
    Args:
        results: Dictionary with classification results
        output_dir: Output directory
        metadata: Dataset metadata
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save results summary
    summary = {
        'methods': {},
        'metadata': {
            'regime_to_idx': metadata['regime_to_idx'],
            'idx_to_regime': metadata['idx_to_regime'],
            'window_size': metadata['window_size'],
            'time_window': metadata['time_window']
        }
    }
    
    for method_key, method_results in results.items():
        summary['methods'][method_key] = {
            'f1_weighted': float(method_results['f1_weighted']),
            'f1_macro': float(method_results['f1_macro']),
            'barycenter_time': float(method_results['barycenter_time']),
            'classify_time': float(method_results['classify_time']),
            'predictions': method_results['predictions'].tolist()
        }
    
    np.save(output_path / "classification_results.npy", summary, allow_pickle=True)
    
    # Save barycenters
    for method_key, method_results in results.items():
        barycenters = method_results['barycenters']
        np.save(output_path / f"barycenters_{method_key}.npy", barycenters, allow_pickle=True)
    
    print(f"\nResults saved to: {output_path}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hydrological Regime Classification using Soft-DTW Barycenters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with basic dataset
  python regime_classification.py --mode basic --test-size 0.2

  # Run with balanced dataset and custom parameters
  python regime_classification.py --mode balanced --gamma 1.0 --max-time-steps 100

  # Run with specific SGD parameters for Wasserstein
  python regime_classification.py --sgd-epochs 200 --sgd-lr 0.05
        """
    )
    
    parser.add_argument(
        "--data-dir", type=str,
        default=str(Path(__file__).parent.parent / "results" / "classification_dataset"),
        help="Directory containing the classification dataset"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(Path(__file__).parent.parent / "results" / "regime_classification"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--mode", type=str, choices=["basic", "balanced"], default="basic",
        help="Dataset mode to use"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data to use for testing"
    )
    parser.add_argument(
        "--gamma", type=float, default=1.0,
        help="Soft-DTW regularization parameter"
    )
    parser.add_argument(
        "--max-time-steps", type=int, default=100,
        help="Maximum number of time steps to use (reduces computation time)"
    )
    parser.add_argument(
        "--sgd-epochs", type=int, default=100,
        help="Number of epochs for SGD barycenter (Wasserstein method)"
    )
    parser.add_argument(
        "--sgd-lr", type=float, default=0.05,
        help="Learning rate for SGD barycenter (Wasserstein method)"
    )
    parser.add_argument(
        "--random-seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("HYDROLOGICAL REGIME CLASSIFICATION EXPERIMENT")
    print("=" * 80)
    
    total_start_time = time.time()
    
    # Load dataset
    print(f"\n[1/5] Loading dataset ({args.mode} mode)...")
    X, Y, metadata = load_classification_dataset(args.data_dir, args.mode)
    print(f"  Loaded {len(X)} samples")
    print(f"  Data shape: {X.shape}")
    print(f"  Classes: {metadata['idx_to_regime']}")
    
    # Display class distribution
    unique, counts = np.unique(Y, return_counts=True)
    print("  Class distribution:")
    for idx, count in zip(unique, counts):
        print(f"    {metadata['idx_to_regime'][idx]}: {count}")
    
    # Train/test split
    print(f"\n[2/5] Splitting into train/test (test_size={args.test_size})...")
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=args.test_size, random_state=args.random_seed, stratify=Y
    )
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Preprocess data
    print(f"\n[3/5] Preprocessing data (max_time_steps={args.max_time_steps})...")
    
    print("  Preprocessing raw training data...")
    X_train_raw = preprocess_samples(X_train, args.max_time_steps)
    print("  Preprocessing raw test data...")
    X_test_raw = preprocess_samples(X_test, args.max_time_steps)
    
    print("  Estimating parameters for training data...")
    X_train_params = estimate_parameters_for_samples(X_train, args.max_time_steps)
    print("  Estimating parameters for test data...")
    X_test_params = estimate_parameters_for_samples(X_test, args.max_time_steps)
    
    print(f"  Raw sample shape: {X_train_raw[0].shape}")
    print(f"  Params sample shape: {X_train_params[0].shape}")
    
    # Run classification experiment
    print(f"\n[4/5] Running classification experiment (gamma={args.gamma})...")
    results = run_classification_experiment(
        X_train_raw, X_train_params,
        X_test_raw, X_test_params,
        Y_train, Y_test,
        metadata['idx_to_regime'],
        gamma=args.gamma,
        sgd_epochs=args.sgd_epochs,
        sgd_lr=args.sgd_lr,
        verbose=True
    )
    
    # Print detailed results
    print_detailed_results(results, Y_test, metadata['idx_to_regime'])
    
    # Save results
    print(f"\n[5/5] Saving results...")
    save_results(results, args.output_dir, metadata)
    
    total_time = time.time() - total_start_time
    
    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Method':<45} {'F1 (weighted)':<15} {'F1 (macro)':<15}")
    print("-" * 75)
    print(f"{'Soft-DTW Euclidean (Raw Data)':<45} {results['euclidean_raw']['f1_weighted']:<15.4f} {results['euclidean_raw']['f1_macro']:<15.4f}")
    print(f"{'Soft-DTW Euclidean (Parameters)':<45} {results['euclidean_params']['f1_weighted']:<15.4f} {results['euclidean_params']['f1_macro']:<15.4f}")
    print(f"{'Soft-DTW Wasserstein (Parameters)':<45} {results['wasserstein_params']['f1_weighted']:<15.4f} {results['wasserstein_params']['f1_macro']:<15.4f}")
    print("-" * 75)
    print(f"\nTotal experiment time: {total_time:.2f}s")


if __name__ == "__main__":
    main()
