"""
Time series extraction functions for lambda parameter estimation.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory for sdtw access
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from sdtw.wasserstein_fast import estimate_exponential_fast


def extract_lambda_series(data, n_patches=200, patch_size=4, time_window=4,
                          random_seed=42, slice_lat=None, slice_lon=None):
    """
    Extract time series of lambda parameters from NetCDF file.

    Args:
        data: xarray data (variable dis06)
        n_patches: Number of patches to extract
        patch_size: Spatial patch size (4x4)
        time_window: Temporal window size (4)
        random_seed: Random seed
        slice_lat: Latitude slice (optional)
        slice_lon: Longitude slice (optional)

    Returns:
        list: List of lambda parameter time series
    """
    np.random.seed(random_seed)

    print("Loading data into memory...")
    try:
        data_subset = data.isel(latitude=slice(0, slice_lat), longitude=slice(0, slice_lon))
        data_array = data_subset.values
        print(f"Data loaded: shape {data_array.shape}")
    except Exception as e:
        print(f"Error loading data into memory: {e}")
        raise

    lat_size = data_array.shape[1]
    lon_size = data_array.shape[2]
    time_size = data_array.shape[0]

    n_lat_patches = lat_size // patch_size
    n_lon_patches = lon_size // patch_size
    n_time_windows = time_size // time_window

    max_patches = n_lat_patches * n_lon_patches

    print(f"Spatial dimensions: {lat_size}x{lon_size}")
    print(f"Possible patches: {max_patches}")
    print(f"Temporal windows: {n_time_windows}")

    if n_patches > max_patches:
        n_patches = max_patches

    # Generate all possible positions
    all_positions = []
    for i in range(n_lat_patches):
        for j in range(n_lon_patches):
            all_positions.append((i * patch_size, j * patch_size))

    # Select randomly
    selected_indices = np.random.choice(len(all_positions), n_patches, replace=False)
    selected_positions = [all_positions[idx] for idx in selected_indices]

    lambda_series_list = []

    print(f"Extracting {n_patches} patches...")
    for patch_idx, (lat_start, lon_start) in enumerate(selected_positions):
        if (patch_idx + 1) % 100 == 0:
            print(f"  Patch {patch_idx + 1}/{n_patches}")

        spatial_patch = data_array[:, lat_start:lat_start + patch_size, lon_start:lon_start + patch_size]

        lambda_values = []

        for t_win in range(n_time_windows):
            t_start = t_win * time_window
            t_end = t_start + time_window

            window_data = spatial_patch[t_start:t_end, :, :].flatten()
            window_data = window_data[~np.isnan(window_data)]
            window_data = window_data[window_data > 0]  # Keep only positive values

            if len(window_data) > 10:
                if patch_idx < 3 and t_win < 2:  # Debug first few
                    print(f"    Patch {patch_idx}, window {t_win}: {len(window_data)} values, "
                          f"range [{window_data.min():.3f}, {window_data.max():.3f}]")
                try:
                    lam = estimate_exponential_fast(window_data.astype(np.float64))
                    if not np.isnan(lam) and not np.isinf(lam) and lam > 0:
                        lambda_values.append(float(lam))
                    else:
                        lambda_values.append(np.nan)
                        if patch_idx < 5:
                            print(f"    Invalid lambda: {lam} (nan={np.isnan(lam)}, "
                                  f"inf={np.isinf(lam)}, positive={lam > 0})")
                except Exception as e:
                    lambda_values.append(np.nan)
                    if patch_idx < 5:
                        print(f"    Exception in estimation: {e}")
            else:
                lambda_values.append(np.nan)

        lambda_array = np.array(lambda_values)
        if not np.any(np.isnan(lambda_array)) and len(lambda_array) == n_time_windows:
            lambda_series_list.append(lambda_array)

    print(f"Valid patches: {len(lambda_series_list)}")
    return lambda_series_list


def extract_multiple_windows_around_position(data, lat_center, lon_center, window_size=20,
                                             n_windows=4, patch_size=4, time_window=4,
                                             random_seed=42):
    """
    Extract multiple 4x4x4 windows in a 20x20 zone around a central position.

    Args:
        data: xarray data
        lat_center, lon_center: Central position (indices)
        window_size: Extraction zone size (20x20)
        n_windows: Number of windows to extract
        patch_size: Spatial patch size (4x4)
        time_window: Temporal window size (4)
        random_seed: Random seed

    Returns:
        list: List of raw time series extracted (one per window)
    """
    np.random.seed(random_seed)

    # Define extraction zone 20x20 around the center
    half_window = window_size // 2
    lat_min = max(0, lat_center - half_window)
    lat_max = min(data.sizes['latitude'], lat_center + half_window)
    lon_min = max(0, lon_center - half_window)
    lon_max = min(data.sizes['longitude'], lon_center + half_window)

    # Calculate possible positions for 4x4 patches
    max_lat_positions = lat_max - lat_min - patch_size + 1
    max_lon_positions = lon_max - lon_min - patch_size + 1

    if max_lat_positions <= 0 or max_lon_positions <= 0:
        print(f"Warning: Zone too small to extract 4x4 patches around ({lat_center}, {lon_center})")
        return []

    # Select n_windows random positions in this zone
    lat_offsets = np.random.randint(0, max_lat_positions, n_windows)
    lon_offsets = np.random.randint(0, max_lon_positions, n_windows)

    series_list = []

    for i in range(n_windows):
        lat_start = lat_min + lat_offsets[i]
        lon_start = lon_min + lon_offsets[i]

        # Extract complete time series for this spatial patch
        spatial_patch = data.isel(
            latitude=slice(lat_start, lat_start + patch_size),
            longitude=slice(lon_start, lon_start + patch_size)
        )

        # Extract raw data for each temporal window of 4 time steps
        n_time_windows = data.sizes['valid_time'] // time_window
        raw_series = []

        for t_win in range(n_time_windows):
            time_start = t_win * time_window
            time_end = (t_win + 1) * time_window

            # Extract temporal data
            temporal_data = spatial_patch.isel(valid_time=slice(time_start, time_end))
            values = temporal_data.values.flatten().astype(np.float64)

            # Filter NaN and negative values
            values = values[~np.isnan(values) & (values > 0)]
            if len(values) > 0:
                # Take a fixed number of samples (e.g., 100) or all if less
                n_samples = min(100, len(values))
                sampled_values = np.random.choice(values, n_samples, replace=False)
                raw_series.append(sampled_values)
            else:
                # If no valid data, create an empty array
                raw_series.append(np.array([]))

        series_list.append(raw_series)

    return series_list
