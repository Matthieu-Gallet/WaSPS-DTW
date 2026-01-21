#!/usr/bin/env python3
"""
Complex simulation experiment script for comparing barycenter methods.

Compares 3 methods on 4 temporally shifted series with individual fluctuations:
- SoftDTW Euclidean on raw data
- SoftDTW Euclidean on estimated parameters
- SoftDTW Wasserstein exponential (SGD with softplus)

This experiment highlights the difference between Wasserstein and Euclidean:
- Wasserstein is weighted by the product of lambdas: W^2(lambda_1, lambda_2) ∝ (1/lambda_1 - 1/lambda_2)^2
- For lambda << 1 or lambda >> 10, the behavior differs significantly from Euclidean

Generates 3 figures for gamma = 0.01, 1, 100
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add parent directories for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from sdtw.barycenter import sdtw_barycenter

from data_generator import generate_shifted_series
from optimizer import sgd_barycenter
from plot import plot_multiple_series


def experiment_shifted_series(gamma_value, folder_figures):
    """
    Experiment with 4 temporally shifted series with individual fluctuations.
    Based on lambda values from previous experiments, extended to 47 timesteps.

    This experiment highlights the difference between Wasserstein and Euclidean:
    - Wasserstein is weighted by the product of lambdas: W^2(lambda_1, lambda_2) ∝ (1/lambda_1 - 1/lambda_2)^2
    - For lambda << 1 or lambda >> 10, the behavior differs significantly from Euclidean

    Parameters
    ----------
    gamma_value : float
        Gamma regularization parameter value
    folder_figures : str
        Folder to save figures
    """
    print("="*80)
    print(f"SHIFTED SERIES EXPERIMENT (4 SERIES) - GAMMA = {gamma_value}")
    print("="*80)

    # Generate shifted series
    data = generate_shifted_series(n_series=4, n_samples=1000, random_seed=42)

    lambda_series = data['lambda_series']
    samples_series = data['samples_series']
    lambda_est_series = data['lambda_est_series']
    base_pattern = data['base_pattern']
    n_timesteps = data['n_timesteps']

    print(f"Number of series: {len(lambda_series)}")
    print(f"Length of each series: {n_timesteps} timesteps")
    print(f"Base pattern: {base_pattern}")
    print(f"Gamma: {gamma_value}")

    # --- Barycenter 1: Euclidean on raw data ---
    print("\n1. Computing Euclidean barycenter on raw data...")
    series_raw = samples_series
    # Initialization: mean of samples
    bary_init_raw = np.mean(samples_series, axis=0)

    bary_raw = sdtw_barycenter(series_raw, bary_init_raw, gamma=gamma_value, max_iter=100, distance="euclidean")
    lambda_bary_raw = np.array([1 / np.mean(bary_raw[i]) for i in range(len(bary_raw))])

    # --- Barycenter 2: Euclidean on estimated parameters ---
    print("2. Computing Euclidean barycenter on estimated parameters...")
    series_est = [lam[:, np.newaxis] for lam in lambda_est_series]
    bary_init_est = np.mean(lambda_est_series, axis=0)[:, np.newaxis]
    print(f"  [DEBUG] Barycenter init (est params): shape={bary_init_est.shape}")
    bary_est = sdtw_barycenter(series_est, bary_init_est, gamma=gamma_value, max_iter=100, distance="euclidean")
    lambda_bary_est = bary_est.flatten()

    # --- Barycenter 3: Wasserstein SGD (softplus) ---
    print("3. Computing Wasserstein barycenter (SGD softplus)...")
    series_wass = samples_series
    print(f"Initialization method: mean_lambda {series_wass[0].shape}")
    # Fix random seed for reproducibility
    bary_wass_sgd, losses = sgd_barycenter(
        series_wass, gamma=gamma_value, learning_rate=0.005,
        num_epochs=500, batch_size=8, lr_decay=0.995, grad_clip=100.0,use_softplus=True,
        distribution="exponential", verbose=False, barycenter_init_method='mean_lambda', warmup_epochs=20
    )
    lambda_bary_wass_sgd = bary_wass_sgd.flatten()

    # --- Visualization: 3 methods horizontally ---
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # Plot 1: Euclidean on raw data
    plot_multiple_series(axes[0], lambda_est_series, lambda_bary_raw,
                        f'Euclidean on raw data\n(gamma={gamma_value})', gamma_value)

    # Plot 2: Euclidean on estimated parameters
    plot_multiple_series(axes[1], lambda_est_series, lambda_bary_est,
                        f'Euclidean on estimated parameters\n(gamma={gamma_value})', gamma_value)

    # Plot 3: Wasserstein SGD
    plot_multiple_series(axes[2], lambda_est_series, lambda_bary_wass_sgd,
                        f'Wasserstein SGD (softplus)\n(gamma={gamma_value})', gamma_value)

    plt.tight_layout()
    filename = f'{folder_figures}/comparison_shifted_gamma_{gamma_value:.2f}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {filename}")
    plt.show()

    # --- Statistics ---
    print("\n" + "="*60)
    print("COMPARATIVE STATISTICS")
    print("="*60)
    for idx, lam_est in enumerate(lambda_est_series):
        print(f"Series {idx+1} (lambda):      {lam_est}")
    print(f"Bary Eucl. (raw):          {lambda_bary_raw}")
    print(f"Bary Eucl. (estimated):    {lambda_bary_est}")
    print(f"Bary Wass. SGD:            {lambda_bary_wass_sgd}")

    # Show SGD losses
    print(f"\nSGD final loss: {losses[-1]:.6f}")
    print(f"SGD converged after {len(losses)} epochs")

    # Analyze differences between methods
    print("\n" + "="*60)
    print("DIFFERENCE ANALYSIS")
    print("="*60)

    # Compute relative errors between Euclidean and Wasserstein
    diff_est_wass = np.abs(lambda_bary_est - lambda_bary_wass_sgd)
    rel_diff = diff_est_wass / (lambda_bary_est + 1e-8) * 100

    print(f"Absolute diff Eucl. vs Wass.: {diff_est_wass}")
    print(f"Relative diff (%):            {rel_diff}")
    print(f"Mean difference: {np.mean(diff_est_wass):.3f}")
    print(f"Max difference: {np.max(diff_est_wass):.3f} at index {np.argmax(diff_est_wass)}")

    # Identify zones where Wasserstein differs the most
    high_diff_indices = np.where(rel_diff > 20)[0]
    if len(high_diff_indices) > 0:
        print(f"\nTimesteps with difference > 20%: {high_diff_indices}")
        for idx in high_diff_indices:
            print(f"  t={idx}: lambda_eucl={lambda_bary_est[idx]:.3f}, lambda_wass={lambda_bary_wass_sgd[idx]:.3f}, "
                  f"diff={rel_diff[idx]:.1f}%")


def main():
    """Main function to run all experiments."""
    print("\n" + "="*80)
    print("COMPLEX SIMULATION - 4 SHIFTED SERIES COMPARISON")
    print("="*80)

    folder_figures = "results/simu_complex"
    os.makedirs(folder_figures, exist_ok=True)

    # Run experiments for different gamma values
    for gamma in [0.01, 1.0, 1000.0]:
        experiment_shifted_series(gamma, folder_figures)
        print("\n")

    print("="*80)
    print("COMPLEX SIMULATION COMPLETED - All figures have been generated")
    print("="*80)


if __name__ == "__main__":
    main()