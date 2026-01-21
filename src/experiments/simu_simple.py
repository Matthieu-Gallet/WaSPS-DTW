#!/usr/bin/env python3
"""
Simple simulation experiment script for comparing barycenter methods.

Compares 3 methods on 2 series of different lengths:
- SoftDTW Euclidean on raw data
- SoftDTW Euclidean on estimated parameters
- SoftDTW Wasserstein exponential (SGD with softplus)

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

from data_generator import generate_exponential_series
from optimizer import sgd_barycenter
from plot import plot_with_correspondences


def experiment_with_gamma(gamma_value, folder_figures):
    """
    Experiment with 2 series of different lengths.
    Compares the 3 barycenter methods for a given gamma value.

    Parameters
    ----------
    gamma_value : float
        Gamma regularization parameter value
    folder_figures : str
        Folder to save figures
    """
    print("="*80)
    print(f"EXPERIMENT WITH GAMMA = {gamma_value}")
    print("="*80)

    # Series of different lengths
    lambda_values1 = [1, 10, 100, 100, 50, 10, 4, 4]
    lambda_values2 = [0.1, 0.1, 0.5, 0.5, 5, 20, 80, 5]
    n_samples = 10000

    print(f"Lambda series 1 ({len(lambda_values1)} timesteps): {lambda_values1}")
    print(f"Lambda series 2 ({len(lambda_values2)} timesteps): {lambda_values2}")
    print(f"Gamma: {gamma_value}")

    # Generate samples
    samples1, lambda_est1 = generate_exponential_series(lambda_values1, n_samples)
    samples2, lambda_est2 = generate_exponential_series(lambda_values2, n_samples)

    print(f"Samples1 shape: {samples1.shape}")
    print(f"Samples2 shape: {samples2.shape}")

    # --- Barycenter 1: Euclidean on raw data ---
    print("\n1. Computing Euclidean barycenter on raw data...")
    series_raw = [samples1, samples2]
    # Initialization: use the longer series
    if len(samples1) < len(samples2):
        bary_init_raw = samples2.copy()
    else:
        bary_init_raw = samples1.copy()

    bary_raw = sdtw_barycenter(series_raw, bary_init_raw, gamma=gamma_value, max_iter=100, distance="euclidean")
    lambda_bary_raw = np.array([1 / np.mean(bary_raw[i]) for i in range(len(bary_raw))])

    # --- Barycenter 2: Euclidean on estimated parameters ---
    print("2. Computing Euclidean barycenter on estimated parameters...")
    series_est = [lambda_est1[:, np.newaxis], lambda_est2[:, np.newaxis]]
    # Init: use the longer series
    if len(lambda_est1) > len(lambda_est2):
        bary_init_est = lambda_est1[:, np.newaxis].copy()
    else:
        bary_init_est = lambda_est2[:, np.newaxis].copy()

    bary_est = sdtw_barycenter(series_est, bary_init_est, gamma=gamma_value, max_iter=100, distance="euclidean")
    lambda_bary_est = bary_est.flatten()

    # --- Barycenter 3: Wasserstein SGD (softplus) ---
    print("3. Computing Wasserstein barycenter (SGD softplus)...")
    series_wass = [samples1, samples2]

    # Fix random seed for reproducibility
    bary_wass_sgd, losses = sgd_barycenter(
        series_wass, gamma=gamma_value, learning_rate=0.01,
        num_epochs=500, batch_size=8, lr_decay=0.995, grad_clip=25.0, use_softplus=True,
        distribution="exponential", verbose=True, barycenter_init_method='mean_lambda', warmup_epochs=20
    )
    lambda_bary_wass_sgd = bary_wass_sgd.flatten()

    # --- Visualization: 3 methods horizontally ---
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # Plot 1: Euclidean on raw data
    plot_with_correspondences(axes[0], lambda_est1, lambda_est2, lambda_bary_raw,
                            f'Euclidean on raw data\n(gamma={gamma_value})', gamma_value)

    # Plot 2: Euclidean on estimated parameters
    plot_with_correspondences(axes[1], lambda_est1, lambda_est2, lambda_bary_est,
                            f'Euclidean on estimated parameters\n(gamma={gamma_value})', gamma_value)

    # Plot 3: Wasserstein SGD
    plot_with_correspondences(axes[2], lambda_est1, lambda_est2, lambda_bary_wass_sgd,
                            f'Wasserstein SGD (softplus)\n(gamma={gamma_value})', gamma_value)

    plt.tight_layout()
    filename = f'{folder_figures}/comparison_gamma_{gamma_value:.2f}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {filename}")
    plt.show()

    # --- Statistics ---
    print("\n" + "="*60)
    print("COMPARATIVE STATISTICS")
    print("="*60)
    print(f"Series 1 (lambda):         {lambda_est1}")
    print(f"Series 2 (lambda):         {lambda_est2}")
    print(f"Bary Eucl. (raw):          {lambda_bary_raw}")
    print(f"Bary Eucl. (estimated):    {lambda_bary_est}")
    print(f"Bary Wass. SGD:            {lambda_bary_wass_sgd}")

    # Show SGD losses
    print(f"\nSGD final loss: {losses[-1]:.6f}")
    print(f"SGD converged after {len(losses)} epochs")


def main():
    """Main function to run all experiments."""
    print("\n" + "="*80)
    print("SIMPLE SIMULATION - 2 SERIES COMPARISON")
    print("="*80)

    folder_figures = "results/simu_simple"
    os.makedirs(folder_figures, exist_ok=True)

    # Run experiments for different gamma values
    for gamma in [0.01, 1.0, 1000.0]:
        experiment_with_gamma(gamma, folder_figures)
        print("\n")

    print("="*80)
    print("SIMPLE SIMULATION COMPLETED - All figures have been generated")
    print("="*80)


if __name__ == "__main__":
    main()