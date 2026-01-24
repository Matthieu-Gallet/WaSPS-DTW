"""
K-fold cross-validation evaluation for classification.
"""

import numpy as np
import time
from typing import Dict, List, Tuple
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import pandas as pd

from .evaluation import evaluate_classification, save_results_to_csv


def run_kfold_classification(X_raw: List[np.ndarray], X_params: List[np.ndarray],
                             Y: np.ndarray, idx_to_regime: Dict[int, str],
                             n_splits: int = 5, gamma: float = 1.0,
                             sgd_epochs: int = 100, sgd_lr: float = 0.05,
                             output_dir: str = None, verbose: bool = True) -> Dict:
    """
    Run stratified k-fold cross-validation for classification.
    
    Args:
        X_raw: List of raw samples
        X_params: List of parameter samples
        Y: Labels
        idx_to_regime: Mapping from label index to regime code
        n_splits: Number of folds
        gamma: Soft-DTW regularization parameter
        sgd_epochs: Number of epochs for SGD barycenter
        sgd_lr: Learning rate for SGD barycenter
        output_dir: Output directory for results
        verbose: Print progress
        
    Returns:
        Dictionary with aggregated results across folds
    """
    if verbose:
        print(f"\n{'=' * 80}")
        print(f"STRATIFIED {n_splits}-FOLD CROSS-VALIDATION")
        print(f"{'=' * 80}")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    all_fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_raw, Y)):
        if verbose:
            print(f"\n{'-' * 80}")
            print(f"Fold {fold_idx + 1}/{n_splits}")
            print(f"{'-' * 80}")
            print(f"Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")
        
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
            verbose=verbose
        )
        
        # Save fold results
        if output_dir:
            save_results_to_csv(fold_results, output_dir, fold_id=f"_fold{fold_idx + 1}")
        
        all_fold_results.append(fold_results)
    
    # Aggregate results across folds
    aggregated_results = aggregate_fold_results(all_fold_results)
    
    if verbose:
        print_kfold_summary(aggregated_results, idx_to_regime)
    
    # Save aggregated results
    if output_dir:
        save_aggregated_results(aggregated_results, output_dir)
    
    return aggregated_results


def aggregate_fold_results(all_fold_results: List[Dict]) -> Dict:
    """
    Aggregate results across all folds.
    
    Args:
        all_fold_results: List of result dictionaries from each fold
        
    Returns:
        Dictionary with mean and std for each metric
    """
    methods = ['euclidean_raw', 'euclidean_params', 'wasserstein_params']
    aggregated = {}
    
    for method in methods:
        # Collect metrics across folds
        f1_weighted = [fold[method]['f1_weighted'] for fold in all_fold_results]
        f1_macro = [fold[method]['f1_macro'] for fold in all_fold_results]
        barycenter_time = [fold[method]['barycenter_time'] for fold in all_fold_results]
        classify_time = [fold[method]['classify_time'] for fold in all_fold_results]
        
        aggregated[method] = {
            'f1_weighted_mean': np.mean(f1_weighted),
            'f1_weighted_std': np.std(f1_weighted),
            'f1_macro_mean': np.mean(f1_macro),
            'f1_macro_std': np.std(f1_macro),
            'barycenter_time_mean': np.mean(barycenter_time),
            'barycenter_time_std': np.std(barycenter_time),
            'classify_time_mean': np.mean(classify_time),
            'classify_time_std': np.std(classify_time),
            'all_f1_weighted': f1_weighted,
            'all_f1_macro': f1_macro
        }
    
    return aggregated


def print_kfold_summary(aggregated_results: Dict, idx_to_regime: Dict[int, str]):
    """
    Print summary of k-fold cross-validation results.
    
    Args:
        aggregated_results: Aggregated results from all folds
        idx_to_regime: Mapping from label index to regime code
    """
    print("\n" + "=" * 80)
    print("K-FOLD CROSS-VALIDATION SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Method':<45} {'F1 (weighted)':<25} {'F1 (macro)':<25}")
    print("-" * 95)
    
    for method_key in ['euclidean_raw', 'euclidean_params', 'wasserstein_params']:
        if method_key in aggregated_results:
            results = aggregated_results[method_key]
            method_name = {
                'euclidean_raw': 'Soft-DTW Euclidean (Raw Data)',
                'euclidean_params': 'Soft-DTW Euclidean (Parameters)',
                'wasserstein_params': 'Soft-DTW Wasserstein (Parameters)'
            }[method_key]
            
            f1_w = f"{results['f1_weighted_mean']:.4f} ± {results['f1_weighted_std']:.4f}"
            f1_m = f"{results['f1_macro_mean']:.4f} ± {results['f1_macro_std']:.4f}"
            
            print(f"{method_name:<45} {f1_w:<25} {f1_m:<25}")
    
    print("-" * 95)


def save_aggregated_results(aggregated_results: Dict, output_dir: str):
    """
    Save aggregated k-fold results to CSV.
    
    Args:
        aggregated_results: Aggregated results from all folds
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for CSV
    data = []
    for method_key, method_results in aggregated_results.items():
        data.append({
            'method': method_key,
            'f1_weighted_mean': method_results['f1_weighted_mean'],
            'f1_weighted_std': method_results['f1_weighted_std'],
            'f1_macro_mean': method_results['f1_macro_mean'],
            'f1_macro_std': method_results['f1_macro_std'],
            'barycenter_time_mean': method_results['barycenter_time_mean'],
            'barycenter_time_std': method_results['barycenter_time_std'],
            'classify_time_mean': method_results['classify_time_mean'],
            'classify_time_std': method_results['classify_time_std']
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path / "kfold_aggregated_scores.csv", index=False)
