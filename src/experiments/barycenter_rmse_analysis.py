#!/usr/bin/env python3
"""
Barycenter RMSE analysis experiment.

Tests the performance of Euclidean and Wasserstein barycenter methods
by varying:
- Number of samples: 5, 25, 100, 1000, 100000
- Estimator method: MLE, log-cumulant
- Gamma values: 0.01, 1.0, 1000.0 (for Wasserstein)
- 10 random seeds for statistical robustness

Compares estimated barycenters against baseline computed with true parameters.
"""

import numpy as np
import pandas as pd
from scipy.stats import expon
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent directories for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from sdtw.barycenter import sdtw_barycenter
from data_generator import generate_shifted_series
from optimizer import sgd_barycenter


def compute_rmse(bary1, bary2):
    """Compute RMSE between two barycenters."""
    return np.sqrt(np.mean((bary1 - bary2)**2))


def compute_euclidean_barycenter(samples_series, gamma=1.0, estimator_method='log_cumulant', X_is_params=False):
    """
    Compute Euclidean barycenter on estimated parameters.
    
    Parameters
    ----------
    samples_series : list of arrays
        List of sample series or parameter series
    gamma : float
        SoftDTW regularization parameter
    estimator_method : str
        'mle' or 'log_cumulant' (ignored if X_is_params=True)
    X_is_params : bool
        If True, samples_series contains parameters directly
        
    Returns
    -------
    array : Barycenter parameters (flattened)
    """
    from estimator import MLE, LogCumulant
    
    if X_is_params:
        # samples_series already contains parameters
        series_params = samples_series
    else:
        # Estimate parameters using selected method
        estimator_class = MLE if estimator_method == 'mle' else LogCumulant
        series_params = []
        for samples in samples_series:
            est = estimator_class(distribution='exponential')
            est.fit(samples)
            series_params.append(est.get_params())
    
    # Convert to proper format for sdtw_barycenter
    series_est = [params.reshape(-1, 1) if params.ndim == 1 else params 
                  for params in series_params]
    bary_init = np.mean([p.flatten() for p in series_params], axis=0)[:, np.newaxis]
    
    bary = sdtw_barycenter(series_est, bary_init, gamma=gamma, max_iter=100, distance="euclidean")
    return bary.flatten()


def compute_wasserstein_barycenter(samples_series, gamma=1.0, estimator_method='log_cumulant', 
                                   X_is_params=False, seed=42):
    """
    Compute Wasserstein barycenter using SGD.
    
    Parameters
    ----------
    samples_series : list of arrays
        List of sample series or parameter series
    gamma : float
        SoftDTW regularization parameter
    estimator_method : str
        'mle' or 'log_cumulant'
    X_is_params : bool
        If True, samples_series contains parameters directly
    seed : int
        Random seed
        
    Returns
    -------
    array : Barycenter parameters (flattened)
    """
    bary, _ = sgd_barycenter(
        samples_series, 
        gamma=gamma, 
        learning_rate=0.01,
        num_epochs=1000, 
        batch_size=8, 
        lr_decay=0.995, 
        grad_clip=25.0,
        use_softplus=True,
        distribution="exponential", 
        verbose=False, 
        barycenter_init_method='mean_lambda',
        warmup_epochs=20,
        seed=seed,
        estimator_method=estimator_method,
        X_is_params=X_is_params
    )
    return bary.flatten()


def run_experiment(n_samples_list, estimator_methods, gamma_values, n_seeds=10, 
                   base_seed=42, save_path='results_rmse_barycenter.csv'):
    """
    Run complete RMSE analysis experiment.
    
    Parameters
    ----------
    n_samples_list : list
        List of sample sizes to test
    estimator_methods : list
        List of estimator methods ('mle', 'log_cumulant')
    gamma_values : list
        List of gamma values to test
    n_seeds : int
        Number of random seeds
    base_seed : int
        Base random seed
    save_path : str
        Path to save results CSV
    """
    print("="*80)
    print("BARYCENTER RMSE ANALYSIS EXPERIMENT")
    print("="*80)
    
    # Generate base data structure (4 shifted series)
    data = generate_shifted_series(n_series=4, n_samples=1000, random_seed=base_seed)
    lambda_series = data['lambda_series']
    n_timesteps = data['n_timesteps']
    
    print(f"\nBase configuration:")
    print(f"  - Number of series: 4")
    print(f"  - Number of timesteps: {n_timesteps}")
    print(f"  - Sample sizes to test: {n_samples_list}")
    print(f"  - Estimators: {estimator_methods}")
    print(f"  - Gamma values: {gamma_values}")
    print(f"  - Number of seeds: {n_seeds}")
    
    # Step 1: Compute baselines with true parameters (gamma=1.0)
    print("\n" + "="*80)
    print("STEP 1: Computing baselines with true parameters (gamma=1.0)")
    print("="*80)
    
    # Convert lambda_series to proper format
    lambda_series_params = [lam.reshape(-1, 1) for lam in lambda_series]
    
    # Baseline Euclidean
    print("  Computing Euclidean baseline...")
    baseline_euclidean = compute_euclidean_barycenter(lambda_series, gamma=1.0, X_is_params=True)
    print(f"    Euclidean baseline shape: {baseline_euclidean.shape}")
    print(f"    Euclidean baseline mean: {np.mean(baseline_euclidean):.4f}")
    
    # Baseline Wasserstein
    print("  Computing Wasserstein baseline...")
    baseline_wasserstein = compute_wasserstein_barycenter(
        lambda_series_params, gamma=1.0, X_is_params=True, seed=base_seed
    )
    print(f"    Wasserstein baseline shape: {baseline_wasserstein.shape}")
    print(f"    Wasserstein baseline mean: {np.mean(baseline_wasserstein):.4f}")
    
    # Step 2: Run experiments
    print("\n" + "="*80)
    print("STEP 2: Running experiments with varying parameters")
    print("="*80)
    
    results = []
    
    # Total experiments: both barycenter types tested with all gamma values
    total_experiments = len(n_samples_list) * len(estimator_methods)  * len(gamma_values)
    
    with tqdm(total=total_experiments, desc="Experiments") as pbar:
        for n_samples in n_samples_list:
            for estimator in estimator_methods:
                for gamma in gamma_values:
                    
                    rmse_euclidean_seeds = []
                    rmse_wasserstein_seeds = []
                    
                    for seed_idx in range(n_seeds):
                        current_seed = base_seed + seed_idx
                        np.random.seed(current_seed)
                        
                        # Regenerate samples with current n_samples and seed
                        samples_series = []
                        for lambda_vals in lambda_series:
                            samples = np.array([expon.rvs(scale=1/lam, size=n_samples, random_state=current_seed) 
                                              for lam in lambda_vals])
                            samples_series.append(samples)
                        
                        # Compute Euclidean barycenter
                        bary_euclidean = compute_euclidean_barycenter(
                            samples_series, gamma=gamma, estimator_method=estimator, X_is_params=False
                        )
                        rmse_euc = compute_rmse(bary_euclidean, baseline_euclidean)
                        rmse_euclidean_seeds.append(rmse_euc)
                        
                        # Compute Wasserstein barycenter
                        bary_wasserstein = compute_wasserstein_barycenter(
                            samples_series, gamma=gamma, estimator_method=estimator, 
                            X_is_params=False, seed=current_seed
                        )
                        rmse_wass = compute_rmse(bary_wasserstein, baseline_wasserstein)
                        rmse_wasserstein_seeds.append(rmse_wass)
                    
                    # Store results for Euclidean
                    results.append({
                        'n_samples': n_samples,
                        'estimator': estimator,
                        'barycenter_type': 'euclidean',
                        'gamma': gamma,
                        'mean_rmse': np.mean(rmse_euclidean_seeds),
                        'std_rmse': np.std(rmse_euclidean_seeds)
                    })
                    # Store results for Wasserstein
                    results.append({
                        'n_samples': n_samples,
                        'estimator': estimator,
                        'barycenter_type': 'wasserstein',
                        'gamma': gamma,
                        'mean_rmse': np.mean(rmse_wasserstein_seeds),
                        'std_rmse': np.std(rmse_wasserstein_seeds)
                    })
                    pbar.update(1)
    
    # Step 3: Save results
    print("\n" + "="*80)
    print("STEP 3: Saving results")
    print("="*80)
    
    df = pd.DataFrame(results)
    df = df.sort_values(['barycenter_type', 'estimator', 'gamma', 'n_samples'])
    df.to_csv(save_path, index=False)
    
    print(f"\nResults saved to: {save_path}")
    print(f"Total rows: {len(df)}")
    print("\nSample of results:")
    print(df.head(10))
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print("\nBest configurations (lowest mean RMSE):")
    print(df.nsmallest(10, 'mean_rmse')[['n_samples', 'estimator', 'barycenter_type', 'gamma', 'mean_rmse', 'std_rmse']])
    
    print("\nWorst configurations (highest mean RMSE):")
    print(df.nlargest(10, 'mean_rmse')[['n_samples', 'estimator', 'barycenter_type', 'gamma', 'mean_rmse', 'std_rmse']])
    
    return df


def main():
    """Main function to run the experiment."""
    
    # Configuration
    n_samples_list = [5, 25, 100, 1000, 100000]
    estimator_methods = ['mle', 'log_cumulant']
    gamma_values = [0.01, 1.0, 1000.0]
    n_seeds = 10
    base_seed = 42
    
    # Create output directory
    output_dir = Path("results/barycenter_rmse_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "rmse_barycenter_analysis.csv"
    
    # Run experiment
    df = run_experiment(
        n_samples_list=n_samples_list,
        estimator_methods=estimator_methods,
        gamma_values=gamma_values,
        n_seeds=n_seeds,
        base_seed=base_seed,
        save_path=str(save_path)
    )
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()
