#!/usr/bin/env python3
"""
Geographic analysis experiment with barycenters - 3 randomly selected zones.

This script:
1. Selects 3 randomly spaced geographic positions
2. For each position, extracts 4 windows of 4x4x4 (spatial:4x4, temporal:4) in a 20x20 pixel zone
3. Computes barycenters using 3 methods: Raw Euclidean, Parameter Euclidean, Wasserstein
4. Generates 3 figures with 3 subplots each (one per geographic zone)
"""

import numpy as np
import xarray as xr
import time
import sys
from pathlib import Path

# Add parent directories for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from sdtw.barycenter import sdtw_barycenter
from sdtw.wasserstein_fast import estimate_exponential_fast

from utils import print_timing, setup_paths
from optimizer import sgd_barycenter
from dataloader import extract_multiple_windows_around_position
from plot import create_geographic_barycenter_figures


def select_spatially_separated_positions(data, n_positions=3, min_distance=50, random_seed=42):
    """
    Select randomly spaced geographic positions.

    Parameters
    ----------
    data : xarray DataArray
        Input data
    n_positions : int
        Number of positions to select
    min_distance : int
        Minimum distance between positions (in pixels)
    random_seed : int
        Random seed

    Returns
    -------
    tuple : (lat_positions, lon_positions) - position indices
    """
    np.random.seed(random_seed)

    lat_size = data.sizes['latitude']
    lon_size = data.sizes['longitude']

    positions = []

    max_attempts = 100
    attempt = 0
    n_positions += 1
    while len(positions) < n_positions and attempt < max_attempts:
        # Generate a candidate position
        lat_idx = np.random.randint(30, lat_size - 30)  # Avoid edges
        lon_idx = np.random.randint(30, lon_size - 30)

        # Check distance with existing positions
        too_close = False
        for existing_lat, existing_lon in positions:
            distance = np.sqrt((lat_idx - existing_lat)**2 + (lon_idx - existing_lon)**2)
            if distance < min_distance:
                too_close = True
                break

        if not too_close:
            positions.append((lat_idx, lon_idx))

        attempt += 1

    if len(positions) < n_positions:
        print(f"Warning: Only {len(positions)} positions selected (requested {n_positions})")

    lat_positions, lon_positions = zip(*positions)
    lat_positions, lon_positions = np.array(lat_positions)[1:], np.array(lon_positions)[1:]
    return lat_positions, lon_positions


def compute_barycenters_for_geographic_zones(data, lat_positions, lon_positions, gamma=1.0):
    """
    Compute barycenters for each geographic zone using 3 methods.

    Parameters
    ----------
    data : xarray DataArray
        Input data
    lat_positions, lon_positions : arrays
        Zone center positions
    gamma : float
        Gamma parameter for SoftDTW

    Returns
    -------
    dict : Barycenter results by zone and method
    """
    results = {}

    for zone_idx, (lat_center, lon_center) in enumerate(zip(lat_positions, lon_positions)):
        print(f"\nProcessing zone {zone_idx + 1}/3 - Center: ({lat_center}, {lon_center})")

        # Extract 4 windows of 4x4x4 in the 20x20 zone
        series_list = extract_multiple_windows_around_position(
            data, lat_center, lon_center,
            window_size=20, n_windows=4, patch_size=4, time_window=4,
            random_seed=42 + zone_idx
        )

        if len(series_list) < 4:
            print(f"Warning: Only {len(series_list)} series extracted for zone {zone_idx + 1}")
            continue

        # Convert to numpy arrays and estimate parameters
        raw_series = series_list
        lambda_series = []

        # Estimate lambda parameters for methods that need them
        for raw_vals_list in raw_series:
            lambda_vals = []
            for raw_vals in raw_vals_list:
                if len(raw_vals) > 0:
                    lambda_val = estimate_exponential_fast(raw_vals)
                    lambda_vals.append(lambda_val)
                else:
                    lambda_vals.append(np.nan)
            lambda_series.append(np.array(lambda_vals))

        if len(raw_series) < 2:
            print(f"Warning: Not enough valid data for zone {zone_idx + 1}")
            continue

        # Prepare data for barycenters
        raw_series_2d = []
        for series in raw_series:
            # Check that all windows have data
            valid_windows = [win for win in series if len(win) > 0]
            if len(valid_windows) != len(series):
                continue
            
            # Ensure all windows have the same number of samples
            min_samples = min(len(win) for win in valid_windows)
            truncated_windows = [win[:min_samples] for win in valid_windows]
            
            # Concatenate all temporal windows into columns
            series_2d = np.column_stack(truncated_windows)
            raw_series_2d.append(series_2d)

        # Ensure all series have the same temporal length
        if len(raw_series_2d) > 0:
            min_time_steps = min(series.shape[1] for series in raw_series_2d)
            raw_series_2d_filter = [series[:, :min_time_steps] for series in raw_series_2d]
            
            # Ensure all series have the same number of samples
            global_min_samples = min(series.shape[0] for series in raw_series_2d_filter)
            raw_series_2d_filter = [series[:global_min_samples, :] for series in raw_series_2d_filter]

        # --- Barycenter 1: Euclidean on raw data ---
        print("  Computing Euclidean barycenter on raw data...")
        start_time = time.time()
        bary_init_raw = np.mean(np.array(raw_series_2d_filter), axis=0)
        bary_raw = sdtw_barycenter(raw_series_2d_filter, bary_init_raw, gamma=gamma,
                                 max_iter=50, distance="euclidean")
        lambda_bary_raw = [estimate_exponential_fast(bary_raw[:, i]) 
                         for i in range(bary_raw.shape[1]) 
                         if ((len(bary_raw[:, i]) > 0) and np.all(bary_raw[:, i] > 0))]
        lambda_bary_raw = np.array(lambda_bary_raw)
        print_timing(start_time, "Euclidean barycenter on raw data")

        # --- Barycenter 2: Euclidean on estimated parameters ---
        print("  Computing Euclidean barycenter on estimated parameters...")
        start_time = time.time()
        series_est = [lam[:, np.newaxis] for lam in lambda_series]
        bary_init_est = np.ones(len(lambda_series[0]))[:, np.newaxis]
        bary_est = sdtw_barycenter(series_est, bary_init_est, gamma=gamma,
                                 max_iter=500, distance="euclidean")
        lambda_bary_est = bary_est.flatten()
        print_timing(start_time, "Euclidean barycenter on estimated parameters")

        # --- Barycenter 3: Wasserstein SGD ---
        print("  Computing Wasserstein barycenter (SGD softplus)...")
        start_time = time.time()
        bary_wass_sgd, losses = sgd_barycenter(
            raw_series_2d, barycenter_init=None, init_method='mean_lambda', gamma=gamma,
            learning_rate=0.75, num_epochs=10, batch_size=4, lr_decay=0.95, grad_clip=10.0,
            distribution="exponential", verbose=False, use_softplus=True
        )
        lambda_bary_wass_sgd = bary_wass_sgd.flatten()
        print_timing(start_time, "Wasserstein SGD barycenter")

        results[f'zone_{zone_idx + 1}'] = {
            'lambda_series': lambda_series,
            'bary_raw': lambda_bary_raw,
            'bary_est': lambda_bary_est,
            'bary_wass': lambda_bary_wass_sgd,
            'center': (lat_center, lon_center)
        }

    return results


def main():
    """Main function."""
    print("=== GEOGRAPHIC ANALYSIS WITH BARYCENTERS ===")

    # Configuration
    paths = setup_paths()
    file_path = paths['data_file']
    figures_dir = paths['figures_dir']

    gamma_value = 1  # Gamma value for experiments

    start_total = time.time()

    try:
        # Open data
        start_open = time.time()
        ds = xr.open_dataset(file_path)
        data = ds['dis06']
        print_timing(start_open, "opening NetCDF file")

        # Select 4 spaced geographic positions
        print("\nSelecting geographic positions...")
        lat_positions, lon_positions = select_spatially_separated_positions(
            data, n_positions=3, min_distance=50, random_seed=42
        )

        print(f"Selected positions: {len(lat_positions)} zones")
        for i, (lat, lon) in enumerate(zip(lat_positions, lon_positions)):
            print(f"  Zone {i+1}: latitude index {lat}, longitude index {lon}")

        # Compute barycenters for each zone
        print(f"\nComputing barycenters (gamma={gamma_value})...")
        start_calc = time.time()
        results = compute_barycenters_for_geographic_zones(
            data, lat_positions, lon_positions, gamma=gamma_value
        )
        print_timing(start_calc, "barycenter computations")

        # Generate figures
        print("\nGenerating figures...")
        create_geographic_barycenter_figures(results, gamma_value, figures_dir)

        print_timing(start_total, "complete analysis")

        # Summary
        print("\n=== SUMMARY ===")
        print(f"Zones analyzed: {len(results)}")
        print(f"Gamma used: {gamma_value}")
        print(f"Figures generated: 3 (one per barycenter method)")
        print(f"Subplots per figure: 3 (one per geographic zone)")
        print(f"Output directory: {figures_dir}")

        ds.close()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
