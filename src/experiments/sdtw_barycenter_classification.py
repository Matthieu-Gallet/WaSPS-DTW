#!/usr/bin/env python3
"""
Hydrological Regime Classification Experiment using Soft-DTW Barycenters.

This script provides multiple evaluation modes:
1. One-shot: Single train/test split
2. K-fold: Stratified k-fold cross-validation
3. Gamma sensitivity: Performance analysis across gamma values (with k-fold)
4. Sample size sensitivity: Performance analysis across training sample sizes (with k-fold)

Three classification methods are compared:
1. Soft-DTW Euclidean on raw data
2. Soft-DTW Euclidean on exponential parameter estimates
3. Soft-DTW Wasserstein on exponential parameter estimates

Author: Generated script
Date: 2026-01-22
"""

import argparse
import numpy as np
import pickle
import time
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add parent directory for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Data loading and preprocessing
from dataloader import (
    load_classification_dataset,
    preprocess_samples,
    estimate_parameters_for_samples,
)

# Evaluation functions
from experiments.classification_evaluation import (
    evaluate_classification,
    print_detailed_results,
    save_results_to_csv,
    run_kfold_classification
)

# Sensitivity analysis
from experiments.classification_sensitivity import (
    run_gamma_sensitivity_analysis,
    run_sample_size_sensitivity_analysis
)

# Visualization
from plot import (
    plot_confusion_matrices,
    plot_gamma_sensitivity,
    plot_sample_size_sensitivity,
    plot_class_pair_barycenters,
    plot_all_class_barycenters_grid
)


def main():
    parser = argparse.ArgumentParser(
        description="Hydrological Regime Classification using Soft-DTW Barycenters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Evaluation Modes:
  one-shot      : Single train/test split
  kfold         : Stratified k-fold cross-validation
  gamma-sens    : Gamma sensitivity analysis with k-fold CV
  sample-sens   : Sample size sensitivity analysis with k-fold CV

Examples:
  # One-shot evaluation
  python regime_classification_v2.py --mode one-shot --test-size 0.2

  # K-fold cross-validation
  python regime_classification_v2.py --mode kfold --n-splits 5

  # Gamma sensitivity analysis
  python regime_classification_v2.py --mode gamma-sens --gamma-values 0.1,0.5,1.0,2.0,5.0

  # Sample size sensitivity analysis
  python regime_classification_v2.py --mode sample-sens --sample-sizes 0.2,0.4,0.6,0.8,1.0
        """
    )
    
    # Data arguments
    parser.add_argument(
        "--data-dir", type=str,
        default=str(Path(__file__).parent.parent / "results" / "classification_dataset" ),
        help="Directory containing the classification dataset"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(Path(__file__).parent.parent / "results" / "regime_classification" / "balanced"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--dataset-mode", type=str, choices=["basic", "balanced"], default="balanced",
        help="Dataset mode to use"
    )
    
    # Evaluation mode
    parser.add_argument(
        "--mode", type=str, 
        choices=["one-shot", "kfold", "gamma-sens", "sample-sens"],
        default="one-shot",
        help="Evaluation mode"
    )
    
    # One-shot parameters
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data to use for testing (one-shot mode)"
    )
    
    # K-fold parameters
    parser.add_argument(
        "--n-splits", type=int, default=5,
        help="Number of folds for k-fold cross-validation"
    )
    
    # Gamma sensitivity parameters
    parser.add_argument(
        "--gamma-values", type=str, default="0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0",
        help="Comma-separated gamma values for sensitivity analysis"
    )
    
    # Sample size sensitivity parameters
    parser.add_argument(
        "--sample-sizes", type=str, default="0.05,0.1,0.2,0.4,0.6,0.8,1.0",
        help="Comma-separated sample sizes (fractions) for sensitivity analysis"
    )
    
    # General parameters
    parser.add_argument(
        "--gamma", type=float, default=10.0,
        help="Soft-DTW regularization parameter (for one-shot and kfold modes)"
    )
    parser.add_argument(
        "--max-time-steps", type=int, default=400,
        help="Maximum number of time steps to use"
    )
    parser.add_argument(
        "--sgd-epochs", type=int, default=75,
        help="Number of epochs for SGD barycenter (Wasserstein method)"
    )
    parser.add_argument(
        "--sgd-lr", type=float, default=0.1,
        help="Learning rate for SGD barycenter (Wasserstein method)"
    )
    parser.add_argument(
        "--disable-lstm", action="store_true",
        help="Disable LSTM baseline in one-shot mode"
    )
    parser.add_argument(
        "--lstm-hidden-size", type=int, default=32,
        help="LSTM hidden size (one-shot mode)"
    )
    parser.add_argument(
        "--lstm-num-layers", type=int, default=2,
        help="Number of LSTM layers (one-shot mode)"
    )
    parser.add_argument(
        "--lstm-dropout", type=float, default=0.0,
        help="LSTM dropout (used when num-layers > 1)"
    )
    parser.add_argument(
        "--lstm-epochs", type=int, default=60,
        help="Number of LSTM training epochs (one-shot mode)"
    )
    parser.add_argument(
        "--lstm-batch-size", type=int, default=32,
        help="LSTM batch size (one-shot mode)"
    )
    parser.add_argument(
        "--lstm-lr", type=float, default=1e-3,
        help="LSTM learning rate (one-shot mode)"
    )
    parser.add_argument(
        "--disable-ot-regul", action="store_true",
        help="Disable regularized OT (STA-style) baseline in one-shot mode"
    )
    parser.add_argument(
        "--ot-epsilon", type=float, default=0.05,
        help="Entropic regularization for OT baseline (one-shot mode)"
    )
    parser.add_argument(
        "--ot-max-iter", type=int, default=30,
        help="Max Sinkhorn iterations for OT distance (one-shot mode)"
    )
    parser.add_argument(
        "--ot-barycenter-iters", type=int, default=60,
        help="Max iterations for OT barycenter updates (one-shot mode)"
    )
    parser.add_argument(
        "--ot-tol", type=float, default=1e-6,
        help="Tolerance for OT numerical loops (one-shot mode)"
    )
    parser.add_argument(
        "--ot-feature-bins", type=int, default=16,
        help="Number of feature bins for fast OT local costs (one-shot mode)"
    )
    parser.add_argument(
        "--ot-time-stride", type=int, default=4,
        help="Temporal stride for OT local costs (one-shot mode)"
    )
    parser.add_argument(
        "--random-seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    
    # Visualization options
    parser.add_argument(
        "--plot-barycenters", default=True, 
        help="Generate barycenter plots with training samples"
    )
    parser.add_argument(
        "--n-samples-plot", type=int, default=200,
        help="Number of samples per class to plot with barycenters"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.random_seed)
    
    print("=" * 80)
    print("HYDROLOGICAL REGIME CLASSIFICATION EXPERIMENT")
    print("=" * 80)
    print(f"Evaluation mode: {args.mode.upper()}")
    
    total_start_time = time.time()
    
    # Load dataset
    print(f"\n[1/4] Loading dataset ({args.dataset_mode} mode)...")
    X, Y, metadata = load_classification_dataset(args.data_dir, args.dataset_mode)
    print(f"  Loaded {len(X)} samples")
    print(f"  Data shape: {X.shape}")
    print(f"  Classes: {metadata['idx_to_regime']}")
    
    # Display class distribution
    unique, counts = np.unique(Y, return_counts=True)
    print("  Class distribution:")
    for idx, count in zip(unique, counts):
        print(f"    {metadata['idx_to_regime'][idx]}: {count}")
    
    # Preprocess data
    print(f"\n[2/4] Preprocessing data (max_time_steps={args.max_time_steps})...")
    print("  Preprocessing raw data...")
    X_raw = preprocess_samples(X, args.max_time_steps)

    print("  Estimating parameters...")
    X_params = estimate_parameters_for_samples(X, args.max_time_steps)
    print(f"  Raw sample shape: {X_raw[0].shape}")
    print(f"  Params sample shape: {X_params[0].shape}")
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation based on mode
    print(f"\n[3/4] Running {args.mode} evaluation...")
    
    if args.mode == "one-shot":
        # =====================================================================
        # ONE-SHOT EVALUATION
        # =====================================================================
        print(f"Train/test split (test_size={args.test_size})...")
        X_train, X_test, Y_train, Y_test = train_test_split(
            np.arange(len(X)), Y, test_size=args.test_size, 
            random_state=args.random_seed, stratify=Y
        )
        
        X_train_raw = [X_raw[i] for i in X_train]
        X_test_raw = [X_raw[i] for i in X_test]
        X_train_params = [X_params[i] for i in X_train]
        X_test_params = [X_params[i] for i in X_test]
        Y_train = Y[X_train]
        Y_test = Y[X_test]
        
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        
        # Run classification
        results = evaluate_classification(
            X_train_raw, X_train_params,
            X_test_raw, X_test_params,
            Y_train, Y_test,
            metadata['idx_to_regime'],
            gamma=args.gamma,
            sgd_epochs=args.sgd_epochs,
            sgd_lr=args.sgd_lr,
            run_lstm=(not args.disable_lstm),
            lstm_hidden_size=args.lstm_hidden_size,
            lstm_num_layers=args.lstm_num_layers,
            lstm_dropout=args.lstm_dropout,
            lstm_epochs=args.lstm_epochs,
            lstm_batch_size=args.lstm_batch_size,
            lstm_lr=args.lstm_lr,
            lstm_seed=args.random_seed,
            run_ot_regul=(not args.disable_ot_regul),
            ot_epsilon=args.ot_epsilon,
            ot_max_iter=args.ot_max_iter,
            ot_barycenter_iters=args.ot_barycenter_iters,
            ot_tol=args.ot_tol,
            ot_feature_bins=args.ot_feature_bins,
            ot_time_stride=args.ot_time_stride,
            verbose=True
        )
        
        # Print detailed results
        print_detailed_results(results, Y_test, metadata['idx_to_regime'])
        
        # Save results
        print(f"\n[4/4] Saving results...")
        save_results_to_csv(results, output_dir)
        plot_confusion_matrices(results, Y_test, metadata['idx_to_regime'], output_dir)

        # Save full experiment data to pickle for replotting without re-running
        pkl_path = Path(output_dir) / "experiment_data.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump({
                'results': results,
                'X_train_raw': X_train_raw,
                'X_train_params': X_train_params,
                'Y_train': Y_train,
                'idx_to_regime': metadata['idx_to_regime'],
            }, f)
        print(f"  Experiment data saved to: {pkl_path}")
        
        # Plot barycenters if requested
        if args.plot_barycenters:
            print("  Generating barycenter plots...")
            
            # Create time coordinates for year 2019
            # Dataset has T time groups (days), we use max_time_steps of them
            n_timesteps = X_train_raw[0].shape[0] if X_train_raw[0].ndim == 2 else len(X_train_raw[0])
            
            # Create datetime array for the actual days used (starting from Jan 1, 2019)
            import pandas as pd
            start_date = pd.Timestamp('2019-01-01')
            time_coords = pd.date_range(start=start_date, periods=n_timesteps, freq='D')
            
            methods = [
                ('euclidean_raw', 'Soft-DTW Euclidean (Raw Data)', X_train_raw, True),
                ('euclidean_params', 'Soft-DTW Euclidean (Parameters)', X_train_params, False),
                ('wasserstein_params', 'Soft-DTW Wasserstein (Parameters)', X_train_params, False),
                ('lstm_raw', 'LSTM Barycenter (Raw Data)', X_train_raw, True),
                ('ot_regul_raw', 'Regularized OT STA (Raw Data)', X_train_raw, True),
            ]

            # Generate class pair plots (A4 format, one figure per pair, all methods overlaid)
            print("  Generating class pair barycenter plots...")
            barycenters_by_method = {
                method_key: results[method_key]['barycenters']
                for method_key, _, _, _ in methods
                if method_key in results
            }
            plot_class_pair_barycenters(
                barycenters_by_method=barycenters_by_method,
                X_train_raw=X_train_raw,
                X_train_params=X_train_params,
                Y_train=Y_train,
                idx_to_regime=metadata['idx_to_regime'],
                output_dir=output_dir,
                save_pdf=True,
                n_samples=args.n_samples_plot,
            )
            
            # Generate grid plots (all classes, one figure per parameter)
            print("  Generating grid barycenter plots...")
            for method_key, method_name, train_data, is_raw in methods:
                if method_key not in results or 'barycenters' not in results[method_key]:
                    continue
                plot_all_class_barycenters_grid(
                    barycenters=results[method_key]['barycenters'],
                    X_train=train_data,
                    Y_train=Y_train,
                    idx_to_regime=metadata['idx_to_regime'],
                    method_name=method_name,
                    output_dir=output_dir,
                    save_pdf=True,
                    n_samples=args.n_samples_plot,
                    is_raw=is_raw,
                )
        
        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"\n{'Method':<45} {'F1 (weighted)':<15} {'F1 (macro)':<15}")
        print("-" * 75)
        method_name_map = {
            'euclidean_raw': 'Soft-DTW Euclidean (Raw Data)',
            'euclidean_params': 'Soft-DTW Euclidean (Parameters)',
            'wasserstein_params': 'Soft-DTW Wasserstein (Parameters) SGD',
            'lstm_raw': 'LSTM Barycenter (Raw Data)',
            'ot_regul_raw': 'Regularized OT STA (Raw Data)',
        }
        for method_key in ['euclidean_raw', 'euclidean_params', 'wasserstein_params', 'lstm_raw', 'ot_regul_raw']:
            if method_key not in results:
                continue
            print(f"{method_name_map[method_key]:<45} {results[method_key]['f1_weighted']:<15.4f} {results[method_key]['f1_macro']:<15.4f}")
        print("-" * 75)
    
    elif args.mode == "kfold":
        # =====================================================================
        # K-FOLD CROSS-VALIDATION
        # =====================================================================
        results = run_kfold_classification(
            X_raw, X_params, Y, metadata['idx_to_regime'],
            n_splits=args.n_splits, gamma=args.gamma,
            sgd_epochs=args.sgd_epochs, sgd_lr=args.sgd_lr,
            output_dir=output_dir, verbose=True
        )
        
        print(f"\n[4/4] Results saved to: {output_dir}")
    
    elif args.mode == "gamma-sens":
        # =====================================================================
        # GAMMA SENSITIVITY ANALYSIS
        # =====================================================================
        gamma_values = [float(x) for x in args.gamma_values.split(',')]
        
        results = run_gamma_sensitivity_analysis(
            X_raw, X_params, Y, metadata['idx_to_regime'],
            gamma_values=gamma_values, n_splits=args.n_splits,
            sgd_epochs=args.sgd_epochs, sgd_lr=args.sgd_lr,
            output_dir=output_dir, verbose=True
        )
        
        # Generate plot
        print(f"\n[4/4] Generating plots...")
        plot_gamma_sensitivity(results, output_dir)
        print(f"  Results saved to: {output_dir}")
    
    elif args.mode == "sample-sens":
        # =====================================================================
        # SAMPLE SIZE SENSITIVITY ANALYSIS
        # =====================================================================
        sample_sizes = [float(x) for x in args.sample_sizes.split(',')]
        
        results = run_sample_size_sensitivity_analysis(
            X_raw, X_params, Y, metadata['idx_to_regime'],
            sample_sizes=sample_sizes, n_splits=args.n_splits,
            gamma=args.gamma, sgd_epochs=args.sgd_epochs,
            sgd_lr=args.sgd_lr, output_dir=output_dir, verbose=True
        )
        
        # Generate plot
        print(f"\n[4/4] Generating plots...")
        plot_sample_size_sensitivity(results, output_dir)
        print(f"  Results saved to: {output_dir}")
    
    total_time = time.time() - total_start_time
    print(f"\nTotal experiment time: {total_time:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
