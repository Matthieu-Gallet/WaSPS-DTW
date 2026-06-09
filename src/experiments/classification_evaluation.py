"""
Classification evaluation functions for Soft-DTW barycenter-based classification.

This module provides functions for:
- Classification by nearest barycenter
- Full evaluation with multiple methods
- K-fold cross-validation
- Results printing and saving
"""

import numpy as np
import time
from typing import Dict, List, Callable
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from pathlib import Path
import sys

# Add project root directory for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.sdtw.classification_methods import (
    compute_barycenter_euclidean_raw,
    compute_barycenter_euclidean_params,
    compute_barycenter_wasserstein_sgd,
    compute_sdtw_distance_euclidean,
    compute_sdtw_distance_wasserstein,
)
from src.experiments.lstm_classifier import (
    train_lstm_classifier,
    compute_lstm_barycenters,
    classify_by_lstm_barycenters,
)
from src.experiments.ot_sta_classifier import (
    compute_barycenter_ot_regul_raw,
    compute_sdtw_distance_ot_regul,
)
from src.experiments.shapelets_classifier import (
    train_shapelets_classifier,
    predict_shapelets_classifier,
)


# =============================================================================
# Core classification functions
# =============================================================================

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
                            verbose: bool = True,
                            run_lstm: bool = False,
                            lstm_hidden_size: int = 64,
                            lstm_num_layers: int = 1,
                            lstm_dropout: float = 0.0,
                            lstm_epochs: int = 60,
                            lstm_batch_size: int = 32,
                            lstm_lr: float = 1e-3,
                            lstm_seed: int = 42,
                            run_ot_regul: bool = False,
                            ot_epsilon: float = 0.05,
                            ot_max_iter: int = 30,
                            ot_barycenter_iters: int = 60,
                            ot_tol: float = 1e-6,
                            ot_feature_bins: int = 32,
                            ot_time_stride: int = 4,
                            run_shapelets: bool = False,
                            shapelets_epochs: int = 20,
                            shapelets_batch_size: int = 32,
                            shapelets_lr: float = 1e-3,
                            shapelets_num_per_scale: int = 4,
                            shapelets_gamma: float = 1.0,
                            shapelets_wasserstein_epochs: int = 8,
                            shapelets_wasserstein_lr: float = 2e-4,
                            shapelets_wasserstein_num_per_scale: int = 2,
                            shapelets_verbose: int = 0,
                            shapelets_seed: int = 42) -> Dict:
    """
    Run classification evaluation with Soft-DTW methods and optional LSTM baseline.
    
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
        run_lstm: Whether to run LSTM baseline on raw data
        lstm_hidden_size: LSTM hidden size
        lstm_num_layers: Number of stacked LSTM layers
        lstm_dropout: Dropout between LSTM layers
        lstm_epochs: Number of LSTM training epochs
        lstm_batch_size: Batch size for LSTM training/inference
        lstm_lr: Learning rate for LSTM training
        lstm_seed: Random seed for LSTM reproducibility
        run_ot_regul: Whether to run STA-style regularized OT baseline on raw data
        ot_epsilon: Entropic OT regularization
        ot_max_iter: Max Sinkhorn iterations for distance computation
        ot_barycenter_iters: Max iterations for OT barycenter fixed-point updates
        ot_tol: Numerical tolerance for OT iterations
        ot_feature_bins: Number of feature bins used for fast OT local costs
        ot_time_stride: Temporal subsampling stride for OT local costs
        run_shapelets: Whether to run the 3 LearningShapelets baselines
        shapelets_epochs: Training epochs for LearningShapelets
        shapelets_batch_size: Batch size for LearningShapelets train/predict
        shapelets_lr: Learning rate for LearningShapelets optimizer
        shapelets_num_per_scale: Number of shapelets for each default time scale
        shapelets_gamma: SoftDTW gamma for shapelets_wasserstein_params
        shapelets_wasserstein_epochs: Dedicated epochs for shapelets_wasserstein_params
        shapelets_wasserstein_lr: Dedicated learning rate for shapelets_wasserstein_params
        shapelets_wasserstein_num_per_scale: Number of shapelets for method 8 (single short scale)
        shapelets_verbose: Verbosity level for LearningShapelets wrapper
        shapelets_seed: Random seed for LearningShapelets reproducibility
        
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
    # =========================================================================
    # Method 4: LSTM barycenters on Raw Data
    # =========================================================================
    if run_lstm:
        if verbose:
            print("\n" + "=" * 60)
            print("Method 4: LSTM Barycenter on Raw Data")
            print("=" * 60)

        start_time = time.time()
        model, state = train_lstm_classifier(
            X_train_raw,
            Y_train,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout,
            epochs=lstm_epochs,
            batch_size=lstm_batch_size,
            learning_rate=lstm_lr,
            seed=lstm_seed,
            verbose=verbose,
        )
        barycenters_lstm = compute_lstm_barycenters(
            model,
            state,
            X_train_raw,
            Y_train,
            batch_size=lstm_batch_size,
        )
        barycenter_time_lstm = time.time() - start_time
        if verbose:
            print(f"  Barycenter build time (train+proto): {barycenter_time_lstm:.2f}s")

        start_time = time.time()
        Y_pred_lstm = classify_by_lstm_barycenters(
            model,
            state,
            barycenters_lstm,
            X_test_raw,
            batch_size=lstm_batch_size,
        )
        classify_time_lstm = time.time() - start_time

        f1_lstm = f1_score(Y_test, Y_pred_lstm, average='weighted', zero_division=0)
        f1_macro_lstm = f1_score(Y_test, Y_pred_lstm, average='macro', zero_division=0)

        results['lstm_raw'] = {
            'predictions': Y_pred_lstm,
            'f1_weighted': f1_lstm,
            'f1_macro': f1_macro_lstm,
            'barycenter_time': barycenter_time_lstm,
            'classify_time': classify_time_lstm,
            'barycenters': barycenters_lstm,
        }

        if verbose:
            print(f"  Classification time: {classify_time_lstm:.2f}s")
            print(f"  F1 Score (weighted): {f1_lstm:.4f}")
            print(f"  F1 Score (macro): {f1_macro_lstm:.4f}")

    # =========================================================================
    # Method 5: STA-style regularized OT barycenter on Raw Data
    # =========================================================================
    if run_ot_regul:
        if verbose:
            print("\n" + "=" * 60)
            print("Method 5: Regularized OT (STA) Barycenter on Raw Data")
            print("=" * 60)

        start_time = time.time()
        barycenters_ot = {}
        for class_label in unique_classes:
            class_samples = [X_train_raw[i] for i in range(len(X_train_raw)) if Y_train[i] == class_label]
            if verbose:
                print(f"  Computing OT barycenter for class {idx_to_regime[class_label]} ({len(class_samples)} samples)...")
            barycenters_ot[class_label] = compute_barycenter_ot_regul_raw(
                class_samples,
                ot_epsilon=ot_epsilon,
                ot_barycenter_iters=ot_barycenter_iters,
                ot_tol=ot_tol,
                ot_feature_bins=ot_feature_bins,
                ot_time_stride=ot_time_stride,
            )
        barycenter_time_ot = time.time() - start_time
        if verbose:
            print(f"  Barycenter computation time: {barycenter_time_ot:.2f}s")

        start_time = time.time()
        Y_pred_ot = classify_by_nearest_barycenter(
            X_test_raw,
            barycenters_ot,
            lambda sample, barycenter, g: compute_sdtw_distance_ot_regul(
                sample,
                barycenter,
                gamma=g,
                ot_epsilon=ot_epsilon,
                ot_max_iter=ot_max_iter,
                ot_tol=ot_tol,
                ot_feature_bins=ot_feature_bins,
                ot_time_stride=ot_time_stride,
            ),
            gamma,
            show_progress=verbose,
        )
        classify_time_ot = time.time() - start_time

        f1_ot = f1_score(Y_test, Y_pred_ot, average='weighted', zero_division=0)
        f1_macro_ot = f1_score(Y_test, Y_pred_ot, average='macro', zero_division=0)

        results['ot_regul_raw'] = {
            'predictions': Y_pred_ot,
            'f1_weighted': f1_ot,
            'f1_macro': f1_macro_ot,
            'barycenter_time': barycenter_time_ot,
            'classify_time': classify_time_ot,
            'barycenters': barycenters_ot,
        }

        if verbose:
            print(f"  Classification time: {classify_time_ot:.2f}s")
            print(f"  F1 Score (weighted): {f1_ot:.4f}")
            print(f"  F1 Score (macro): {f1_macro_ot:.4f}")

    # =========================================================================
    # Methods 6-8: Learning Shapelets (true supervised learning)
    # =========================================================================
    if run_shapelets:
        ts_len_params = int(np.asarray(X_train_params[0]).shape[0])
        ws_shapelet_len = max(3, ts_len_params // 4)
        ws_shapelets_size_and_len = {
            ws_shapelet_len: shapelets_wasserstein_num_per_scale
        }

        shapelets_specs = [
            {
                "method_key": "shapelets_euclidean_raw",
                "dist_measure": "euclidean",
                "train_samples": X_train_raw,
                "test_samples": X_test_raw,
                "title": "Method 6: Learning Shapelets Euclidean (Raw Data)",
                "gamma": 1.0,
                "epochs": shapelets_epochs,
                "lr": shapelets_lr,
                "num_per_scale": shapelets_num_per_scale,
                "shapelets_size_and_len": None,
            },
            {
                "method_key": "shapelets_euclidean_params",
                "dist_measure": "euclidean",
                "train_samples": X_train_params,
                "test_samples": X_test_params,
                "title": "Method 7: Learning Shapelets Euclidean (Parameters)",
                "gamma": 1.0,
                "epochs": shapelets_epochs,
                "lr": shapelets_lr,
                "num_per_scale": shapelets_num_per_scale,
                "shapelets_size_and_len": None,
            },
            {
                "method_key": "shapelets_wasserstein_params",
                "dist_measure": "soft_dtw_wasserstein",
                "train_samples": X_train_params,
                "test_samples": X_test_params,
                "title": "Method 8: Learning Shapelets Soft-DTW Wasserstein (Parameters)",
                "gamma": shapelets_gamma,
                "epochs": shapelets_wasserstein_epochs,
                "lr": shapelets_wasserstein_lr,
                "num_per_scale": shapelets_wasserstein_num_per_scale,
                "shapelets_size_and_len": ws_shapelets_size_and_len,
            },
        ]

        for spec in shapelets_specs:
            method_key = spec["method_key"]
            dist_measure = spec["dist_measure"]
            train_samples = spec["train_samples"]
            test_samples = spec["test_samples"]
            title = spec["title"]
            method_gamma = spec["gamma"]
            if verbose:
                print("\n" + "=" * 60)
                print(title)
                print("=" * 60)

            clf, shape_state = train_shapelets_classifier(
                train_samples=train_samples,
                y_train=Y_train,
                dist_measure=dist_measure,
                epochs=spec["epochs"],
                batch_size=shapelets_batch_size,
                learning_rate=spec["lr"],
                shapelets_size_and_len=spec["shapelets_size_and_len"],
                shapelets_gamma=method_gamma,
                shapelets_num_per_scale=spec["num_per_scale"],
                seed=shapelets_seed,
                verbose=shapelets_verbose,
            )
            train_time = shape_state["train_time"]
            if verbose:
                print(f"  Training time: {train_time:.2f}s")

            start_time = time.time()
            y_pred = predict_shapelets_classifier(
                clf=clf,
                state=shape_state,
                test_samples=test_samples,
                batch_size=shapelets_batch_size,
            )
            classify_time = time.time() - start_time

            f1_w = f1_score(Y_test, y_pred, average='weighted', zero_division=0)
            f1_m = f1_score(Y_test, y_pred, average='macro', zero_division=0)

            results[method_key] = {
                'predictions': y_pred,
                'f1_weighted': f1_w,
                'f1_macro': f1_m,
                'barycenter_time': train_time,
                'classify_time': classify_time,
            }
            if verbose:
                print(f"  Classification time: {classify_time:.2f}s")
                print(f"  F1 Score (weighted): {f1_w:.4f}")
                print(f"  F1 Score (macro): {f1_m:.4f}")

    return results


# =============================================================================
# K-Fold cross-validation
# =============================================================================

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


# =============================================================================
# Results output functions
# =============================================================================

def print_detailed_results(results: Dict, Y_test: np.ndarray, idx_to_regime: Dict[int, str]):
    """
    Print detailed classification results for all methods.
    
    Args:
        results: Dictionary with results from evaluate_classification
        Y_test: True test labels
        idx_to_regime: Mapping from label index to regime code
    """
    # idx_Y_test = [idx_to_regime[i] for i in Y_test]
    target_names = [idx_to_regime[i] for i in np.unique(Y_test)]  # Only include classes present in test set
    # target_names = [idx_to_regime[i] for i in sorted(idx_to_regime.keys())]
    
    
    print("\n" + "=" * 80)
    print("DETAILED CLASSIFICATION RESULTS")
    print("=" * 80)
    
    methods = [
        ('euclidean_raw', 'Soft-DTW Euclidean (Raw Data)'),
        ('euclidean_params', 'Soft-DTW Euclidean (Parameters)'),
        ('wasserstein_params', 'Soft-DTW Wasserstein (Parameters)'),
        ('lstm_raw', 'LSTM Barycenter (Raw Data)'),
        ('ot_regul_raw', 'Regularized OT STA (Raw Data)'),
        ('shapelets_euclidean_raw', 'Learning Shapelets Euclidean (Raw Data)'),
        ('shapelets_euclidean_params', 'Learning Shapelets Euclidean (Parameters)'),
        ('shapelets_wasserstein_params', 'Learning Shapelets Soft-DTW Wasserstein (Parameters)')
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
