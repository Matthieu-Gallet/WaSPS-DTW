#!/usr/bin/env python3
"""
Build classification dataset for hydrological regime classification.

This script extracts spatial-temporal windows from NetCDF river discharge data
around Explore2 station locations and creates a classification dataset for
regime_code prediction.

The output format is:
- X: numpy array of shape (N, T, D, W, W) where:
    - N: number of samples
    - T: number of time steps (time_size // time_window)
    - D: temporal grouping (time_window)
    - W: spatial window size (window_size)
- Y: numpy array of shape (N,) containing regime_code labels

Author: Generated script
Date: 2026-01-22
"""

import argparse
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Coordinate Transformation Functions
# ============================================================================

def lambert93_to_wgs84(x_lambert: float, y_lambert: float) -> Tuple[float, float]:
    """
    Convert Lambert93 (EPSG:2154) coordinates to WGS84 (EPSG:4326).
    
    Args:
        x_lambert: X coordinate in Lambert93 (meters)
        y_lambert: Y coordinate in Lambert93 (meters)
        
    Returns:
        Tuple of (longitude, latitude) in WGS84 (degrees)
    """
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(x_lambert, y_lambert)
        return lon, lat
    except ImportError:
        # Fallback: approximate conversion using projection parameters
        # This is less accurate but works without pyproj
        import math
        
        # Lambert93 projection parameters
        n = 0.7256077650
        c = 11754255.426
        xs = 700000.0
        ys = 12655612.050
        e = 0.0818191910428
        
        # Inverse Lambert projection
        r = math.sqrt((x_lambert - xs) ** 2 + (ys - y_lambert) ** 2)
        gamma = math.atan((x_lambert - xs) / (ys - y_lambert))
        
        # Longitude
        lon_rad = gamma / n + 3.0 * math.pi / 180.0  # Reference longitude: 3°E
        
        # Latitude (iterative calculation)
        lat_iso = -math.log(r / c) / n
        lat_rad = 2 * math.atan(math.exp(lat_iso)) - math.pi / 2
        
        # Iterative refinement
        for _ in range(10):
            e_sin = e * math.sin(lat_rad)
            lat_rad_new = 2 * math.atan(
                ((1 + e_sin) / (1 - e_sin)) ** (e / 2) * math.exp(lat_iso)
            ) - math.pi / 2
            if abs(lat_rad_new - lat_rad) < 1e-11:
                break
            lat_rad = lat_rad_new
        
        # Convert to degrees
        lon = math.degrees(lon_rad)
        lat = math.degrees(lat_rad)
        
        return lon, lat


def find_nearest_pixel(lat: float, lon: float, 
                       lat_coords: np.ndarray, lon_coords: np.ndarray) -> Tuple[int, int]:
    """
    Find the nearest pixel indices for given lat/lon coordinates.
    
    Args:
        lat: Latitude in WGS84
        lon: Longitude in WGS84
        lat_coords: Array of latitude coordinates from the NetCDF file
        lon_coords: Array of longitude coordinates from the NetCDF file
        
    Returns:
        Tuple of (lat_idx, lon_idx) pixel indices
    """
    lat_idx = np.abs(lat_coords - lat).argmin()
    lon_idx = np.abs(lon_coords - lon).argmin()
    return int(lat_idx), int(lon_idx)


# ============================================================================
# Data Extraction Functions
# ============================================================================

def extract_window(data: np.ndarray, lat_idx: int, lon_idx: int,
                   window_size: int, time_window: int) -> Optional[np.ndarray]:
    """
    Extract a spatial-temporal window centered on a pixel.
    
    Args:
        data: Full data array of shape (time, lat, lon)
        lat_idx: Center latitude index
        lon_idx: Center longitude index
        window_size: Spatial window size (W)
        time_window: Temporal grouping size (D)
        
    Returns:
        Window array of shape (T, D, W, W) or None if extraction fails
    """
    time_size, lat_size, lon_size = data.shape
    half_w = window_size // 2
    
    # Calculate bounds
    lat_start = lat_idx - half_w
    lat_end = lat_idx + half_w + (window_size % 2)  # Handle odd window sizes
    lon_start = lon_idx - half_w
    lon_end = lon_idx + half_w + (window_size % 2)
    
    # Check bounds
    if lat_start < 0 or lat_end > lat_size or lon_start < 0 or lon_end > lon_size:
        return None
    
    # Extract spatial window for all time steps
    spatial_window = data[:, lat_start:lat_end, lon_start:lon_end]
    
    # Check for valid window size
    if spatial_window.shape[1] != window_size or spatial_window.shape[2] != window_size:
        return None
    
    # Reshape into temporal groups
    n_time_groups = time_size // time_window
    if n_time_groups == 0:
        return None
    
    # Trim to exact multiple of time_window
    trimmed_time = n_time_groups * time_window
    spatial_window = spatial_window[:trimmed_time, :, :]
    
    # Reshape to (T, D, W, W)
    window = spatial_window.reshape(n_time_groups, time_window, window_size, window_size)
    
    # Check for too many NaN values
    nan_ratio = np.isnan(window).sum() / window.size
    if nan_ratio > 0.5:  # More than 50% NaN
        return None
    
    return window


def extract_windows_in_neighborhood(data: np.ndarray, lat_idx: int, lon_idx: int,
                                    window_size: int, time_window: int,
                                    neighborhood_size: int, n_samples: int,
                                    random_seed: Optional[int] = None) -> List[np.ndarray]:
    """
    Extract multiple windows in a neighborhood around a reference point.
    
    Args:
        data: Full data array of shape (time, lat, lon)
        lat_idx: Center latitude index
        lon_idx: Center longitude index
        window_size: Spatial window size (W)
        time_window: Temporal grouping size (D)
        neighborhood_size: Size of the neighborhood in pixels
        n_samples: Number of samples to extract
        random_seed: Random seed for reproducibility
        
    Returns:
        List of window arrays of shape (T, D, W, W)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    half_n = neighborhood_size // 2
    half_w = window_size // 2
    time_size, lat_size, lon_size = data.shape
    
    # Calculate valid range for window centers
    lat_min = max(half_w, lat_idx - half_n)
    lat_max = min(lat_size - half_w - 1, lat_idx + half_n)
    lon_min = max(half_w, lon_idx - half_n)
    lon_max = min(lon_size - half_w - 1, lon_idx + half_n)
    
    if lat_min >= lat_max or lon_min >= lon_max:
        return []
    
    windows = []
    attempts = 0
    max_attempts = n_samples * 10  # Avoid infinite loops
    
    while len(windows) < n_samples and attempts < max_attempts:
        # Random position in neighborhood
        rand_lat = np.random.randint(lat_min, lat_max + 1)
        rand_lon = np.random.randint(lon_min, lon_max + 1)
        
        # Extract window
        window = extract_window(data, rand_lat, rand_lon, window_size, time_window)
        if window is not None:
            windows.append(window)
        
        attempts += 1
    
    return windows


# ============================================================================
# Dataset Building Functions
# ============================================================================

def load_stations(csv_path: str) -> pd.DataFrame:
    """
    Load station data from CSV file.
    
    Args:
        csv_path: Path to the CSV file with station data
        
    Returns:
        DataFrame with station information
    """
    df = pd.read_csv(csv_path)
    
    # Filter out stations with missing coordinates
    df = df.dropna(subset=['X_Lambert93', 'Y_Lambert93', 'regime_code'])
    
    # Convert coordinates to WGS84
    coords = df.apply(
        lambda row: lambert93_to_wgs84(row['X_Lambert93'], row['Y_Lambert93']),
        axis=1
    )
    df['longitude'] = [c[0] for c in coords]
    df['latitude'] = [c[1] for c in coords]
    
    return df


def build_basic_dataset(nc_path: str, csv_path: str, window_size: int = 5,
                        time_window: int = 4, output_dir: str = ".",
                        random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Build basic classification dataset (one sample per station).
    
    Output shape: (N_stations, T, D, W, W) for X and (N_stations,) for Y
    
    Args:
        nc_path: Path to the NetCDF file
        csv_path: Path to the stations CSV file
        window_size: Spatial window size (W)
        time_window: Temporal grouping size (D)
        output_dir: Output directory for numpy files
        random_seed: Random seed
        
    Returns:
        Tuple of (X, Y, metadata_dict)
    """
    print("=" * 60)
    print("Building Basic Classification Dataset")
    print("=" * 60)
    
    # Load station data
    print("\n[1/4] Loading station data...")
    stations_df = load_stations(csv_path)
    print(f"  Loaded {len(stations_df)} stations with valid coordinates")
    
    # Display regime distribution
    regime_counts = stations_df['regime_code'].value_counts()
    print("\n  Regime distribution:")
    for regime, count in regime_counts.items():
        print(f"    {regime}: {count}")
    
    # Create label encoder
    unique_regimes = sorted(stations_df['regime_code'].unique())
    regime_to_idx = {regime: idx for idx, regime in enumerate(unique_regimes)}
    idx_to_regime = {idx: regime for regime, idx in regime_to_idx.items()}
    print(f"\n  Label encoding: {regime_to_idx}")
    
    # Load NetCDF data
    print("\n[2/4] Loading NetCDF data...")
    ds = xr.open_dataset(nc_path, engine='netcdf4')
    data = ds['dis06'].values
    lat_coords = ds['latitude'].values
    lon_coords = ds['longitude'].values
    print(f"  Data shape: {data.shape} (time, lat, lon)")
    print(f"  Latitude range: [{lat_coords.min():.2f}, {lat_coords.max():.2f}]")
    print(f"  Longitude range: [{lon_coords.min():.2f}, {lon_coords.max():.2f}]")
    
    # Extract windows for each station
    print("\n[3/4] Extracting windows for each station...")
    X_list = []
    Y_list = []
    station_info = []
    
    for idx, row in tqdm(stations_df.iterrows(), total=len(stations_df), desc="  Processing"):
        lat = row['latitude']
        lon = row['longitude']
        regime = row['regime_code']
        
        # Find nearest pixel
        lat_idx, lon_idx = find_nearest_pixel(lat, lon, lat_coords, lon_coords)
        
        # Check if coordinates are within data bounds
        if lat_idx < 0 or lat_idx >= data.shape[1] or lon_idx < 0 or lon_idx >= data.shape[2]:
            continue
        
        # Extract window
        window = extract_window(data, lat_idx, lon_idx, window_size, time_window)
        
        if window is not None:
            X_list.append(window)
            Y_list.append(regime_to_idx[regime])
            station_info.append({
                'code_station': row['code_station'],
                'nom_station': row['nom_station'],
                'lat_idx': lat_idx,
                'lon_idx': lon_idx,
                'lat': lat,
                'lon': lon
            })
    
    ds.close()
    
    print(f"\n  Successfully extracted {len(X_list)} samples")
    
    # Convert to numpy arrays
    X = np.array(X_list)
    Y = np.array(Y_list)
    
    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")
    
    # Save data
    print("\n[4/4] Saving dataset...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path / "X_basic.npy", X)
    np.save(output_path / "Y_basic.npy", Y)
    
    # Save metadata
    metadata = {
        'window_size': window_size,
        'time_window': time_window,
        'regime_to_idx': regime_to_idx,
        'idx_to_regime': idx_to_regime,
        'n_samples': len(X),
        'station_info': station_info
    }
    np.save(output_path / "metadata_basic.npy", metadata, allow_pickle=True)
    
    print(f"  Saved to: {output_path}")
    print(f"    - X_basic.npy: {X.shape}")
    print(f"    - Y_basic.npy: {Y.shape}")
    print(f"    - metadata_basic.npy")
    
    return X, Y, metadata


def build_balanced_dataset(nc_path: str, csv_path: str, window_size: int = 5,
                           time_window: int = 4, neighborhood_size: int = 20,
                           samples_per_class: Dict[str, int] = None,
                           output_dir: str = ".", random_seed: int = 42
                           ) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Build balanced classification dataset with augmentation from neighborhoods.
    
    This function extracts additional windows from a neighborhood around each
    station to balance the dataset across classes.
    
    Args:
        nc_path: Path to the NetCDF file
        csv_path: Path to the stations CSV file
        window_size: Spatial window size (W)
        time_window: Temporal grouping size (D)
        neighborhood_size: Size of the neighborhood in pixels for augmentation
        samples_per_class: Dictionary mapping regime_code to desired number of samples
                          Example: {'PC': 300, 'PN': 300, 'PM': 300}
        output_dir: Output directory for numpy files
        random_seed: Random seed
        
    Returns:
        Tuple of (X, Y, metadata_dict)
    """
    print("=" * 60)
    print("Building Balanced Classification Dataset")
    print("=" * 60)
    
    np.random.seed(random_seed)
    
    # Load station data
    print("\n[1/5] Loading station data...")
    stations_df = load_stations(csv_path)
    print(f"  Loaded {len(stations_df)} stations with valid coordinates")
    
    # Display original regime distribution
    regime_counts = stations_df['regime_code'].value_counts()
    print("\n  Original regime distribution:")
    for regime, count in regime_counts.items():
        print(f"    {regime}: {count}")
    
    # Create label encoder
    unique_regimes = sorted(stations_df['regime_code'].unique())
    regime_to_idx = {regime: idx for idx, regime in enumerate(unique_regimes)}
    idx_to_regime = {idx: regime for regime, idx in regime_to_idx.items()}
    print(f"\n  Label encoding: {regime_to_idx}")
    
    # Set default samples_per_class if not provided
    if samples_per_class is None:
        max_count = regime_counts.max()
        samples_per_class = {regime: max_count for regime in unique_regimes}
    
    print(f"\n  Target samples per class: {samples_per_class}")
    
    # Load NetCDF data
    print("\n[2/5] Loading NetCDF data...")
    ds = xr.open_dataset(nc_path, engine='netcdf4')
    data = ds['dis06'].values
    lat_coords = ds['latitude'].values
    lon_coords = ds['longitude'].values
    print(f"  Data shape: {data.shape} (time, lat, lon)")
    
    # Group stations by regime
    print("\n[3/5] Grouping stations by regime...")
    stations_by_regime = {regime: [] for regime in unique_regimes}
    
    for idx, row in stations_df.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        regime = row['regime_code']
        
        lat_idx, lon_idx = find_nearest_pixel(lat, lon, lat_coords, lon_coords)
        
        if 0 <= lat_idx < data.shape[1] and 0 <= lon_idx < data.shape[2]:
            stations_by_regime[regime].append({
                'lat_idx': lat_idx,
                'lon_idx': lon_idx,
                'code_station': row['code_station'],
                'nom_station': row['nom_station'],
                'lat': lat,
                'lon': lon
            })
    
    for regime, stations in stations_by_regime.items():
        print(f"    {regime}: {len(stations)} valid stations")
    
    # Extract balanced samples
    print("\n[4/5] Extracting balanced samples...")
    X_list = []
    Y_list = []
    sample_info = []
    
    for regime in unique_regimes:
        regime_stations = stations_by_regime[regime]
        target_count = samples_per_class.get(regime, 100)
        
        if len(regime_stations) == 0:
            print(f"  Warning: No valid stations for regime {regime}")
            continue
        
        print(f"\n  Processing regime {regime} (target: {target_count} samples)...")
        
        # Calculate samples per station
        n_stations = len(regime_stations)
        base_samples_per_station = target_count // n_stations
        extra_samples = target_count % n_stations
        
        regime_samples = 0
        
        for i, station in enumerate(tqdm(regime_stations, desc=f"    {regime}")):
            # Determine number of samples for this station
            n_samples = base_samples_per_station + (1 if i < extra_samples else 0)
            
            if n_samples == 0:
                continue
            
            # First, extract the center window (original station location)
            center_window = extract_window(
                data, station['lat_idx'], station['lon_idx'],
                window_size, time_window
            )
            
            if center_window is not None:
                X_list.append(center_window)
                Y_list.append(regime_to_idx[regime])
                sample_info.append({
                    'code_station': station['code_station'],
                    'source': 'center',
                    'lat_idx': station['lat_idx'],
                    'lon_idx': station['lon_idx']
                })
                regime_samples += 1
                n_samples -= 1
            
            # Then extract additional samples from neighborhood
            if n_samples > 0:
                neighborhood_windows = extract_windows_in_neighborhood(
                    data, station['lat_idx'], station['lon_idx'],
                    window_size, time_window, neighborhood_size,
                    n_samples, random_seed=random_seed + i
                )
                
                for j, window in enumerate(neighborhood_windows):
                    X_list.append(window)
                    Y_list.append(regime_to_idx[regime])
                    sample_info.append({
                        'code_station': station['code_station'],
                        'source': f'neighborhood_{j}',
                        'lat_idx': station['lat_idx'],
                        'lon_idx': station['lon_idx']
                    })
                    regime_samples += 1
        
        print(f"    Extracted {regime_samples} samples for regime {regime}")
    
    ds.close()
    
    # Convert to numpy arrays
    X = np.array(X_list)
    Y = np.array(Y_list)
    
    print(f"\n  Total samples extracted: {len(X)}")
    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")
    
    # Final class distribution
    print("\n  Final class distribution:")
    unique, counts = np.unique(Y, return_counts=True)
    for idx, count in zip(unique, counts):
        print(f"    {idx_to_regime[idx]}: {count}")
    
    # Save data
    print("\n[5/5] Saving dataset...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path / "X_balanced.npy", X)
    np.save(output_path / "Y_balanced.npy", Y)
    
    # Save metadata
    metadata = {
        'window_size': window_size,
        'time_window': time_window,
        'neighborhood_size': neighborhood_size,
        'samples_per_class': samples_per_class,
        'regime_to_idx': regime_to_idx,
        'idx_to_regime': idx_to_regime,
        'n_samples': len(X),
        'sample_info': sample_info
    }
    np.save(output_path / "metadata_balanced.npy", metadata, allow_pickle=True)
    
    print(f"  Saved to: {output_path}")
    print(f"    - X_balanced.npy: {X.shape}")
    print(f"    - Y_balanced.npy: {Y.shape}")
    print(f"    - metadata_balanced.npy")
    
    return X, Y, metadata


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build classification dataset for hydrological regime classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic dataset (one sample per station)
  python build_classification_dataset.py --mode basic --window-size 5 --time-window 4

  # Balanced dataset with augmentation
  python build_classification_dataset.py --mode balanced --window-size 5 --time-window 4 \\
      --neighborhood-size 20 --samples-per-class '{"PC": 300, "PN": 300, "PM": 300}'

Output format:
  X: numpy array of shape (N, T, D, W, W)
     - N: number of samples
     - T: number of time steps (total_time / time_window)
     - D: temporal grouping (time_window)
     - W: spatial window size (window_size)
  Y: numpy array of shape (N,) containing regime_code labels (encoded as integers)
        """
    )
    
    parser.add_argument(
        "--nc-path", type=str,
        default="/home/mgallet/Documents/Dataset/RIVER_DISCHARGES/c7491e060d94c97212f0fe7ebcff57f0/data_version-5.nc",
        help="Path to the NetCDF file"
    )
    parser.add_argument(
        "--csv-path", type=str,
        default=str(Path(__file__).parent / "stations_regimes_explore2.csv"),
        help="Path to the stations CSV file"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(Path(__file__).parent.parent.parent / "results" / "classification_dataset"),
        help="Output directory for numpy files"
    )
    parser.add_argument(
        "--mode", type=str, choices=["basic", "balanced"], default="basic",
        help="Dataset building mode: 'basic' (one sample per station) or 'balanced' (with augmentation)"
    )
    parser.add_argument(
        "--window-size", type=int, default=5,
        help="Spatial window size (W) in pixels"
    )
    parser.add_argument(
        "--time-window", type=int, default=4,
        help="Temporal grouping size (D) in time steps"
    )
    parser.add_argument(
        "--neighborhood-size", type=int, default=20,
        help="Size of neighborhood for augmentation (in pixels, only for balanced mode)"
    )
    parser.add_argument(
        "--samples-per-class", type=str, default=None,
        help='JSON dict mapping regime codes to target sample counts, e.g., \'{"PC": 300, "PN": 300, "PM": 300}\''
    )
    parser.add_argument(
        "--random-seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Parse samples_per_class if provided
    samples_per_class = None
    if args.samples_per_class:
        import json
        samples_per_class = json.loads(args.samples_per_class)
    
    # Build dataset
    if args.mode == "basic":
        X, Y, metadata = build_basic_dataset(
            nc_path=args.nc_path,
            csv_path=args.csv_path,
            window_size=args.window_size,
            time_window=args.time_window,
            output_dir=args.output_dir,
            random_seed=args.random_seed
        )
    else:  # balanced
        X, Y, metadata = build_balanced_dataset(
            nc_path=args.nc_path,
            csv_path=args.csv_path,
            window_size=args.window_size,
            time_window=args.time_window,
            neighborhood_size=args.neighborhood_size,
            samples_per_class=samples_per_class,
            output_dir=args.output_dir,
            random_seed=args.random_seed
        )
    
    print("\n" + "=" * 60)
    print("Dataset building complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
