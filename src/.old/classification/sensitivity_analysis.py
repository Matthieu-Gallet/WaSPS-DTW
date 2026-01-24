"""
Sensitivity analysis functions (gamma and sample size variations).
"""

import numpy as np
from typing import Dict, List
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .evaluation import evaluate_classification


def run_gamma_sensitivity_analysis(X_raw: List[np.ndarray], X_params: List[np.ndarray],
                                    Y: np.ndarray, idx_to_regime: Dict[int, str],
                                    gamma_values: List[float] = None,
                                    n_splits: int = 5, sgd_epochs: int = 100,
                                    sgd_lr: float = 0.05, output_dir: str = None,
                                    verbose: bool = True) -> Dict:
    """
    Analyze performance as a function of gamma parameter using k-fold CV.
    
    Args:
        X_raw: List of raw samples
        X_params: List of parameter samples
        Y: Labels
        idx_to_regime: Mapping from label index to regime code
        gamma_values: List of gamma values to test
        n_splits: Number of folds for cross-validation
        sgd_epochs: Number of epochs for SGD barycenter
        sgd_lr: Learning rate for SGD barycenter
        output_dir: Output directory for results
        verbose: Print progress
        
    Returns:
        Dictionary with results for each gamma value
    """
    if gamma_values is None:
        gamma_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    if verbose:
        print(f"\n{'=' * 80}")
        print("GAMMA SENSITIVITY ANALYSIS")
        print(f"{'=' * 80}")
        print(f"Testing gamma values: {gamma_values}")
        print(f"Using {n_splits}-fold cross-validation")
    
    all_results = {}
    
    for gamma in gamma_values:
        if verbose:
            print(f"\n{'-' * 80}")
            print(f"Testing gamma = {gamma}")
            print(f"{'-' * 80}")
        
        # Run k-fold CV for this gamma
        gamma_results = []
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_raw, Y)):
            if verbose:
                print(f"  Fold {fold_idx + 1}/{n_splits}...", end=" ")
            
            # Split data
            X_train_raw = [X_raw[i] for i in train_idx]
            X_test_raw = [X_raw[i] for i in test_idx]
            X_train_params = [X_params[i] for i in train_idx]
            X_test_params = [X_params[i] for i in test_idx]
            Y_train = Y[train_idx]
            Y_test = Y[test_idx]
            
            # Run classification
            fold_results = evaluate_classification(
                X_train_raw, X_train_params,
                X_test_raw, X_test_params,
                Y_train, Y_test,
                idx_to_regime, gamma=gamma,
                sgd_epochs=sgd_epochs, sgd_lr=sgd_lr,
                verbose=False
            )
            
            gamma_results.append(fold_results)
            
            if verbose:
                print("Done")
        
        # Aggregate results for this gamma
        methods = ['euclidean_raw', 'euclidean_params', 'wasserstein_params']
        aggregated = {}
        
        for method in methods:
            f1_weighted = [fold[method]['f1_weighted'] for fold in gamma_results]
            f1_macro = [fold[method]['f1_macro'] for fold in gamma_results]
            
            aggregated[method] = {
                'f1_weighted_mean': np.mean(f1_weighted),
                'f1_weighted_std': np.std(f1_weighted),
                'f1_macro_mean': np.mean(f1_macro),
                'f1_macro_std': np.std(f1_macro),
                'fold_scores': f1_weighted  # Store individual fold scores for boxplot
            }
        
        all_results[gamma] = aggregated
        
        if verbose:
            print(f"  Results for gamma = {gamma}:")
            for method in methods:
                print(f"    {method}: F1 = {aggregated[method]['f1_weighted_mean']:.4f} ± {aggregated[method]['f1_weighted_std']:.4f}")
    
    # Save results to CSV
    if output_dir:
        save_gamma_sensitivity_results(all_results, output_dir)
    
    return all_results


def run_sample_size_sensitivity_analysis(X_raw: List[np.ndarray], X_params: List[np.ndarray],
                                         Y: np.ndarray, idx_to_regime: Dict[int, str],
                                         sample_sizes: List[int] = None,
                                         n_splits: int = 5, gamma: float = 1.0,
                                         sgd_epochs: int = 100, sgd_lr: float = 0.05,
                                         output_dir: str = None, verbose: bool = True) -> Dict:
    """
    Analyze performance as a function of training sample size using k-fold CV.
    
    Args:
        X_raw: List of raw samples
        X_params: List of parameter samples
        Y: Labels
        idx_to_regime: Mapping from label index to regime code
        sample_sizes: List of training sample sizes to test (as fractions or counts)
        n_splits: Number of folds for cross-validation
        gamma: Soft-DTW regularization parameter
        sgd_epochs: Number of epochs for SGD barycenter
        sgd_lr: Learning rate for SGD barycenter
        output_dir: Output directory for results
        verbose: Print progress
        
    Returns:
        Dictionary with results for each sample size
    """
    if sample_sizes is None:
        # Use fractions of training data
        sample_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    if verbose:
        print(f"\n{'=' * 80}")
        print("SAMPLE SIZE SENSITIVITY ANALYSIS")
        print(f"{'=' * 80}")
        print(f"Testing sample sizes: {sample_sizes}")
        print(f"Using {n_splits}-fold cross-validation")
    
    all_results = {}
    
    for sample_size in sample_sizes:
        if verbose:
            print(f"\n{'-' * 80}")
            print(f"Testing sample size = {sample_size}")
            print(f"{'-' * 80}")
        
        # Run k-fold CV for this sample size
        size_results = []
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_raw, Y)):
            if verbose:
                print(f"  Fold {fold_idx + 1}/{n_splits}...", end=" ")
            
            # Subsample training data if needed
            if sample_size < 1.0:
                # Sample a fraction of training data (stratified)
                n_train_samples = int(len(train_idx) * sample_size)
                # Stratified sampling
                train_labels = Y[train_idx]
                unique_labels = np.unique(train_labels)
                sampled_train_idx = []
                
                for label in unique_labels:
                    label_idx = train_idx[train_labels == label]
                    n_samples_per_class = max(1, int(len(label_idx) * sample_size))
                    sampled_idx = np.random.choice(label_idx, size=n_samples_per_class, replace=False)
                    sampled_train_idx.extend(sampled_idx)
                
                train_idx = np.array(sampled_train_idx)
            
            # Split data
            X_train_raw = [X_raw[i] for i in train_idx]
            X_test_raw = [X_raw[i] for i in test_idx]
            X_train_params = [X_params[i] for i in train_idx]
            X_test_params = [X_params[i] for i in test_idx]
            Y_train = Y[train_idx]
            Y_test = Y[test_idx]
            
            # Run classification
            fold_results = evaluate_classification(
                X_train_raw, X_train_params,
                X_test_raw, X_test_params,
                Y_train, Y_test,
                idx_to_regime, gamma=gamma,
                sgd_epochs=sgd_epochs, sgd_lr=sgd_lr,
                verbose=False
            )
            
            size_results.append(fold_results)
            
            if verbose:
                print("Done")
        
        # Aggregate results for this sample size
        methods = ['euclidean_raw', 'euclidean_params', 'wasserstein_params']
        aggregated = {}
        
        for method in methods:
            f1_weighted = [fold[method]['f1_weighted'] for fold in size_results]
            f1_macro = [fold[method]['f1_macro'] for fold in size_results]
            
            aggregated[method] = {
                'f1_weighted_mean': np.mean(f1_weighted),
                'f1_weighted_std': np.std(f1_weighted),
                'f1_macro_mean': np.mean(f1_macro),
                'f1_macro_std': np.std(f1_macro),
                'fold_scores': f1_weighted  # Store individual fold scores for boxplot
            }
        
        all_results[sample_size] = aggregated
        
        if verbose:
            print(f"  Results for sample size = {sample_size}:")
            for method in methods:
                print(f"    {method}: F1 = {aggregated[method]['f1_weighted_mean']:.4f} ± {aggregated[method]['f1_weighted_std']:.4f}")
    
    # Save results to CSV
    if output_dir:
        save_sample_size_sensitivity_results(all_results, output_dir)
    
    return all_results


def save_gamma_sensitivity_results(results: Dict, output_dir: str):
    """
    Save gamma sensitivity analysis results to CSV.
    
    Args:
        results: Dictionary with results for each gamma value
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for CSV
    data = []
    for gamma, gamma_results in results.items():
        for method, method_results in gamma_results.items():
            data.append({
                'gamma': gamma,
                'method': method,
                'f1_weighted_mean': method_results['f1_weighted_mean'],
                'f1_weighted_std': method_results['f1_weighted_std'],
                'f1_macro_mean': method_results['f1_macro_mean'],
                'f1_macro_std': method_results['f1_macro_std']
            })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path / "gamma_sensitivity_results.csv", index=False)


def save_sample_size_sensitivity_results(results: Dict, output_dir: str):
    """
    Save sample size sensitivity analysis results to CSV.
    
    Args:
        results: Dictionary with results for each sample size
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for CSV
    data = []
    for sample_size, size_results in results.items():
        for method, method_results in size_results.items():
            data.append({
                'sample_size': sample_size,
                'method': method,
                'f1_weighted_mean': method_results['f1_weighted_mean'],
                'f1_weighted_std': method_results['f1_weighted_std'],
                'f1_macro_mean': method_results['f1_macro_mean'],
                'f1_macro_std': method_results['f1_macro_std']
            })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path / "sample_size_sensitivity_results.csv", index=False)
