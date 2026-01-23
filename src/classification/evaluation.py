"""
Evaluation and classification functions.
"""

import numpy as np
import time
from typing import Dict, List, Callable, Tuple
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import pandas as pd
from pathlib import Path

from .barycenter_methods import (
    compute_barycenter_euclidean_raw,
    compute_barycenter_euclidean_params,
    compute_barycenter_wasserstein_sgd
)
from .distance_computation import (
    compute_sdtw_distance_euclidean,
    compute_sdtw_distance_wasserstein
)


def classify_by_nearest_barycenter(samples: List, barycenters: Dict[int, np.ndarray],
                                    distance_func: Callable, gamma: float = 1.0,
                                    show_progress: bool = True) -> np.ndarray:
    """
    Classify samples by nearest barycenter.
    
    Args:
        samples: List of samples (raw data or parameters)
        barycenters: Dictionary mapping class labels to barycenters
        distance_func: Function to compute distance (takes sample, barycenter, gamma)
        gamma: Soft-DTW regularization parameter
        show_progress: Show progress bar
        
    Returns:
        Array of predicted class labels
    """
    predictions = []
    
    iterator = tqdm(samples, desc="Classifying", leave=False) if show_progress else samples
    
    for sample in iterator:
        min_distance = np.inf
        predicted_class = None
        
        for class_label, barycenter in barycenters.items():
            distance = distance_func(sample, barycenter, gamma)
            
            if distance < min_distance:
                min_distance = distance
                predicted_class = class_label
        
        predictions.append(predicted_class)
    
    return np.array(predictions)


def evaluate_classification(X_train_raw: List[np.ndarray], X_train_params: List[np.ndarray],
                            X_test_raw: List[np.ndarray], X_test_params: List[np.ndarray],
                            Y_train: np.ndarray, Y_test: np.ndarray,
                            idx_to_regime: Dict[int, str], gamma: float = 1.0,
                            sgd_epochs: int = 100, sgd_lr: float = 0.05,
                            verbose: bool = True) -> Dict:
    """
    Run classification evaluation with all three methods.
    
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
        X_test_raw, barycenters_raw, compute_sdtw_distance_euclidean, gamma, show_progress=verbose
    )
    classify_time_raw = time.time() - start_time
    
    # Calculate metrics
    f1_raw = f1_score(Y_test, Y_pred_raw, average='weighted', zero_division=0)
    f1_macro_raw = f1_score(Y_test, Y_pred_raw, average='macro', zero_division=0)
    
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
        X_test_params, barycenters_params_euc, compute_sdtw_distance_euclidean, gamma, show_progress=verbose
    )
    classify_time_params = time.time() - start_time
    
    # Calculate metrics
    f1_params_euc = f1_score(Y_test, Y_pred_params_euc, average='weighted', zero_division=0)
    f1_macro_params_euc = f1_score(Y_test, Y_pred_params_euc, average='macro', zero_division=0)
    
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
            class_params, gamma=gamma, learning_rate=sgd_lr, num_epochs=sgd_epochs, 
            batch_size=4, verbose=False
        )
    
    barycenter_time_wass = time.time() - start_time
    if verbose:
        print(f"  Barycenter computation time: {barycenter_time_wass:.2f}s")
    
    # Classify test samples
    start_time = time.time()
    Y_pred_wass = classify_by_nearest_barycenter(
        X_test_params, barycenters_wass, compute_sdtw_distance_wasserstein, gamma, show_progress=verbose
    )
    classify_time_wass = time.time() - start_time
    
    # Calculate metrics
    f1_wass = f1_score(Y_test, Y_pred_wass, average='weighted', zero_division=0)
    f1_macro_wass = f1_score(Y_test, Y_pred_wass, average='macro', zero_division=0)
    
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
        results: Dictionary with results from evaluate_classification
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


def save_results_to_csv(results: Dict, output_dir: str, fold_id: str = ""):
    """
    Save classification results to CSV file.
    
    Args:
        results: Dictionary with classification results
        output_dir: Output directory
        fold_id: Identifier for the fold (for k-fold CV)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for CSV
    data = []
    for method_key, method_results in results.items():
        data.append({
            'method': method_key,
            'f1_weighted': method_results['f1_weighted'],
            'f1_macro': method_results['f1_macro'],
            'barycenter_time': method_results['barycenter_time'],
            'classify_time': method_results['classify_time']
        })
    
    df = pd.DataFrame(data)
    
    filename = f"classification_scores{fold_id}.csv"
    df.to_csv(output_path / filename, index=False)
    
    return output_path / filename
