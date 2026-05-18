"""
Discharge dataset preparation — river discharge NetCDF + station CSV.

Produces two standardised formats:

  Format A (classification):
    <output_dir>/
      X.npy          float64  (N, T, D)
      Y.npy          int      (N,)
      metadata.npy   pickled dict: label_map, idx_to_label, source info

  Format B (groups, for barycenter comparison):
    <output_dir>/
      group_<i>/series.npy   float64  (M, T, D)
      metadata.npy           pickled dict: group_names, source info

Entry points (CLI):
  python src/dataloader/discharge.py prepare-classification --nc-path ... --csv-path ... --output-dir ...
  python src/dataloader/discharge.py prepare-groups         --nc-path ... --output-dir ...
"""

import argparse
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

warnings.filterwarnings('ignore')


# =============================================================================
# Coordinate helpers
# =============================================================================

def lambert93_to_wgs84(x: float, y: float) -> Tuple[float, float]:
    """Convert Lambert-93 (EPSG:2154) to WGS-84 (lon, lat)."""
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs('EPSG:2154', 'EPSG:4326', always_xy=True)
        return transformer.transform(x, y)
    except ImportError:
        n, c, xs, ys, e = 0.7256077650, 11754255.426, 700000.0, 12655612.050, 0.0818191910428
        r = math.sqrt((x - xs) ** 2 + (ys - y) ** 2)
        gamma = math.atan((x - xs) / (ys - y))
        lon_rad = gamma / n + 3.0 * math.pi / 180.0
        lat_iso = -math.log(r / c) / n
        lat_rad = 2 * math.atan(math.exp(lat_iso)) - math.pi / 2
        for _ in range(10):
            e_sin = e * math.sin(lat_rad)
            lat_new = 2 * math.atan(
                ((1 + e_sin) / (1 - e_sin)) ** (e / 2) * math.exp(lat_iso)
            ) - math.pi / 2
            if abs(lat_new - lat_rad) < 1e-11:
                break
            lat_rad = lat_new
        return math.degrees(lon_rad), math.degrees(lat_rad)


def nearest_pixel(lat: float, lon: float,
                  lat_coords: np.ndarray, lon_coords: np.ndarray) -> Tuple[int, int]:
    return int(np.abs(lat_coords - lat).argmin()), int(np.abs(lon_coords - lon).argmin())


# =============================================================================
# Window extraction (generic, parameterised on dim names)
# =============================================================================

def open_variable(nc_path: str, variable: str,
                  lat_dim: str = 'latitude', lon_dim: str = 'longitude',
                  time_dim: str = 'valid_time') -> xr.DataArray:
    """Open one variable from a NetCDF file."""
    ds = xr.open_dataset(nc_path, engine='netcdf4')
    return ds[variable]


def extract_window(data_array: np.ndarray, lat_idx: int, lon_idx: int,
                   window_size: int, time_window: int) -> Optional[np.ndarray]:
    """
    Extract a spatial-temporal window and return shape (T, D) where D = time_window * window_size^2.

    Returns None if the window falls outside bounds or contains > 50 % NaN.
    """
    n_time, n_lat, n_lon = data_array.shape
    hw = window_size // 2
    r0, r1 = lat_idx - hw, lat_idx + hw + (window_size % 2)
    c0, c1 = lon_idx - hw, lon_idx + hw + (window_size % 2)
    if r0 < 0 or r1 > n_lat or c0 < 0 or c1 > n_lon:
        return None
    patch = data_array[:, r0:r1, c0:c1]
    if patch.shape[1] != window_size or patch.shape[2] != window_size:
        return None
    n_t = (n_time // time_window) * time_window
    patch = patch[:n_t]
    T = n_t // time_window
    # shape (T, time_window, window_size, window_size) → (T, D)
    window = patch.reshape(T, time_window, window_size, window_size)
    if np.isnan(window).mean() > 0.5:
        return None
    return window.reshape(T, -1).astype(np.float64)


def extract_windows_in_neighborhood(data_array: np.ndarray,
                                    lat_idx: int, lon_idx: int,
                                    window_size: int, time_window: int,
                                    neighborhood_size: int, n_samples: int,
                                    rng: np.random.RandomState) -> List[np.ndarray]:
    """Random windows in a neighborhood around a center pixel."""
    hw = window_size // 2
    hn = neighborhood_size // 2
    _, n_lat, n_lon = data_array.shape
    lat_min = max(hw, lat_idx - hn)
    lat_max = min(n_lat - hw - 1, lat_idx + hn)
    lon_min = max(hw, lon_idx - hn)
    lon_max = min(n_lon - hw - 1, lon_idx + hn)
    if lat_min >= lat_max or lon_min >= lon_max:
        return []
    windows, attempts = [], 0
    while len(windows) < n_samples and attempts < n_samples * 10:
        r = rng.randint(lat_min, lat_max + 1)
        c = rng.randint(lon_min, lon_max + 1)
        w = extract_window(data_array, r, c, window_size, time_window)
        if w is not None:
            windows.append(w)
        attempts += 1
    return windows


# =============================================================================
# Station loading
# =============================================================================

def load_stations(csv_path: str,
                  x_col: str = 'X_Lambert93', y_col: str = 'Y_Lambert93',
                  label_col: str = 'regime_code',
                  crs_from: str = 'lambert93') -> pd.DataFrame:
    """
    Load station CSV, drop rows with missing coordinates or labels,
    convert to WGS-84 (lon, lat).

    Args:
        csv_path:  Path to the station CSV file.
        x_col:     Column name for the X coordinate.
        y_col:     Column name for the Y coordinate.
        label_col: Column name for the class label.
        crs_from:  Source CRS. Only 'lambert93' is currently supported;
                   for WGS-84 input simply name x_col='longitude', y_col='latitude'.
    """
    df = pd.read_csv(csv_path).dropna(subset=[x_col, y_col, label_col])
    if crs_from == 'lambert93':
        coords = df.apply(lambda r: lambert93_to_wgs84(r[x_col], r[y_col]), axis=1)
        df = df.copy()
        df['longitude'] = [c[0] for c in coords]
        df['latitude'] = [c[1] for c in coords]
    else:
        df = df.rename(columns={x_col: 'longitude', y_col: 'latitude'})
    return df


# =============================================================================
# Format A — classification dataset
# =============================================================================

def prepare_classification(nc_path: str, csv_path: str, output_dir: str,
                            variable: str = 'dis06',
                            lat_dim: str = 'latitude', lon_dim: str = 'longitude',
                            time_dim: str = 'valid_time',
                            x_col: str = 'X_Lambert93', y_col: str = 'Y_Lambert93',
                            label_col: str = 'regime_code',
                            crs_from: str = 'lambert93',
                            window_size: int = 5, time_window: int = 4,
                            neighborhood_size: int = 20,
                            samples_per_class: Optional[Dict[str, int]] = None,
                            random_seed: int = 42):
    """
    Produce Format A (X.npy, Y.npy, metadata.npy) from a NetCDF + station CSV pair.

    If samples_per_class is None, one sample per station (basic mode).
    If samples_per_class is provided, spatial neighbourhood augmentation is used
    to reach the target count for each class (balanced mode).

    Output X has shape (N, T, D) where D = time_window * window_size^2.
    """
    print('=== prepare_classification ===')
    np.random.seed(random_seed)

    stations = load_stations(csv_path, x_col=x_col, y_col=y_col,
                             label_col=label_col, crs_from=crs_from)
    print(f'  {len(stations)} stations')

    unique_labels = sorted(stations[label_col].unique())
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}

    da = open_variable(nc_path, variable, lat_dim, lon_dim, time_dim)
    data_array = da.values                         # (time, lat, lon)
    lat_coords = da[lat_dim].values
    lon_coords = da[lon_dim].values
    print(f'  data shape: {data_array.shape}')

    X_list, Y_list = [], []

    if samples_per_class is None:
        # Basic: one window per station
        for _, row in tqdm(stations.iterrows(), total=len(stations), desc='  stations'):
            li, ci = nearest_pixel(row['latitude'], row['longitude'], lat_coords, lon_coords)
            w = extract_window(data_array, li, ci, window_size, time_window)
            if w is not None:
                X_list.append(w)
                Y_list.append(label_to_idx[row[label_col]])
    else:
        # Balanced: augmentation from neighborhood
        stations_by_label = {lbl: [] for lbl in unique_labels}
        for _, row in stations.iterrows():
            li, ci = nearest_pixel(row['latitude'], row['longitude'], lat_coords, lon_coords)
            if 0 <= li < data_array.shape[1] and 0 <= ci < data_array.shape[2]:
                stations_by_label[row[label_col]].append((li, ci))

        for lbl in unique_labels:
            target = samples_per_class.get(lbl, 100)
            lbl_stations = stations_by_label[lbl]
            if not lbl_stations:
                continue
            base, extra = divmod(target, len(lbl_stations))
            n_collected = 0
            for i, (li, ci) in enumerate(tqdm(lbl_stations, desc=f'  {lbl}')):
                if n_collected >= target:
                    break
                n_here = base + (1 if i < extra else 0)
                if n_here == 0:
                    continue
                rng = np.random.RandomState(random_seed + i)
                center = extract_window(data_array, li, ci, window_size, time_window)
                if center is not None and n_collected < target:
                    X_list.append(center)
                    Y_list.append(label_to_idx[lbl])
                    n_collected += 1
                    n_here -= 1
                remaining = min(n_here, target - n_collected)
                for w in extract_windows_in_neighborhood(
                        data_array, li, ci, window_size, time_window,
                        neighborhood_size, remaining, rng):
                    if n_collected >= target:
                        break
                    X_list.append(w)
                    Y_list.append(label_to_idx[lbl])
                    n_collected += 1

    X = np.stack(X_list)            # (N, T, D)
    Y = np.array(Y_list, dtype=int)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'X.npy', X)
    np.save(out / 'Y.npy', Y)
    metadata = {
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'variable': variable, 'window_size': window_size,
        'time_window': time_window, 'n_samples': len(X),
        'source_nc': str(nc_path), 'source_csv': str(csv_path),
    }
    np.save(out / 'metadata.npy', metadata, allow_pickle=True)
    print(f'  saved {len(X)} samples → {out}  X shape: {X.shape}')


# =============================================================================
# Format B — groups dataset
# =============================================================================

def prepare_groups(nc_path: str, output_dir: str,
                   variable: str = 'dis06',
                   lat_dim: str = 'latitude', lon_dim: str = 'longitude',
                   time_dim: str = 'valid_time',
                   n_groups: int = 3, n_series_per_group: int = 4,
                   window_size: int = 4, time_window: int = 4,
                   neighborhood_size: int = 20, min_distance: int = 50,
                   random_seed: int = 42):
    """
    Produce Format B (group_<i>/series.npy, metadata.npy) by randomly sampling
    spatially separated groups from a NetCDF file.

    Each group contains n_series_per_group windows extracted from a spatial
    neighbourhood. Output series have shape (M, T, D) per group.
    """
    print('=== prepare_groups ===')
    np.random.seed(random_seed)

    da = open_variable(nc_path, variable, lat_dim, lon_dim, time_dim)
    data_array = da.values
    n_lat, n_lon = data_array.shape[1], data_array.shape[2]
    print(f'  data shape: {data_array.shape}')

    # Select spatially separated group centers
    positions = []
    attempts = 0
    while len(positions) < n_groups and attempts < 500:
        lat = np.random.randint(30, n_lat - 30)
        lon = np.random.randint(30, n_lon - 30)
        if all(np.sqrt((lat - p[0]) ** 2 + (lon - p[1]) ** 2) >= min_distance
               for p in positions):
            positions.append((lat, lon))
        attempts += 1
    print(f'  {len(positions)} group centers selected')

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    group_names = {}
    rng = np.random.RandomState(random_seed)

    for i, (lat, lon) in enumerate(positions):
        windows = extract_windows_in_neighborhood(
            data_array, lat, lon, window_size, time_window,
            neighborhood_size, n_series_per_group, rng
        )
        if len(windows) < 2:
            print(f'  group {i}: not enough valid windows, skipped')
            continue
        series = np.stack(windows)    # (M, T, D)
        group_dir = out / f'group_{i}'
        group_dir.mkdir(exist_ok=True)
        np.save(group_dir / 'series.npy', series)
        group_names[str(i)] = f'group_{i}_lat{lat}_lon{lon}'
        print(f'  group {i}: {series.shape}')

    metadata = {
        'group_names': group_names,
        'variable': variable, 'window_size': window_size,
        'time_window': time_window, 'n_groups': len(group_names),
        'source_nc': str(nc_path),
    }
    np.save(out / 'metadata.npy', metadata, allow_pickle=True)
    print(f'  saved {len(group_names)} groups → {out}')


# =============================================================================
# CLI
# =============================================================================

def _build_parser():
    parser = argparse.ArgumentParser(
        description='Prepare discharge datasets (Format A or B)'
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # prepare-classification
    p_cls = sub.add_parser('prepare-classification',
                            help='Produce Format A classification dataset')
    p_cls.add_argument('--nc-path', required=True)
    p_cls.add_argument('--csv-path', required=True)
    p_cls.add_argument('--output-dir', required=True)
    p_cls.add_argument('--variable', default='dis06')
    p_cls.add_argument('--lat-dim', default='latitude')
    p_cls.add_argument('--lon-dim', default='longitude')
    p_cls.add_argument('--time-dim', default='valid_time')
    p_cls.add_argument('--x-col', default='X_Lambert93')
    p_cls.add_argument('--y-col', default='Y_Lambert93')
    p_cls.add_argument('--label-col', default='regime_code')
    p_cls.add_argument('--crs-from', default='lambert93')
    p_cls.add_argument('--window-size', type=int, default=5)
    p_cls.add_argument('--time-window', type=int, default=4)
    p_cls.add_argument('--neighborhood-size', type=int, default=20)
    p_cls.add_argument('--samples-per-class', type=str, default=None,
                        help='JSON dict, e.g. \'{"NG": 300, "NP": 300}\'')
    p_cls.add_argument('--random-seed', type=int, default=42)

    # prepare-groups
    p_grp = sub.add_parser('prepare-groups',
                            help='Produce Format B groups dataset')
    p_grp.add_argument('--nc-path', required=True)
    p_grp.add_argument('--output-dir', required=True)
    p_grp.add_argument('--variable', default='dis06')
    p_grp.add_argument('--lat-dim', default='latitude')
    p_grp.add_argument('--lon-dim', default='longitude')
    p_grp.add_argument('--time-dim', default='valid_time')
    p_grp.add_argument('--n-groups', type=int, default=3)
    p_grp.add_argument('--n-series-per-group', type=int, default=4)
    p_grp.add_argument('--window-size', type=int, default=4)
    p_grp.add_argument('--time-window', type=int, default=4)
    p_grp.add_argument('--neighborhood-size', type=int, default=20)
    p_grp.add_argument('--min-distance', type=int, default=50)
    p_grp.add_argument('--random-seed', type=int, default=42)

    return parser


if __name__ == '__main__':
    import json
    args = _build_parser().parse_args()

    if args.command == 'prepare-classification':
        spc = json.loads(args.samples_per_class) if args.samples_per_class else None
        prepare_classification(
            nc_path=args.nc_path, csv_path=args.csv_path,
            output_dir=args.output_dir,
            variable=args.variable, lat_dim=args.lat_dim,
            lon_dim=args.lon_dim, time_dim=args.time_dim,
            x_col=args.x_col, y_col=args.y_col,
            label_col=args.label_col, crs_from=args.crs_from,
            window_size=args.window_size, time_window=args.time_window,
            neighborhood_size=args.neighborhood_size,
            samples_per_class=spc, random_seed=args.random_seed,
        )
    elif args.command == 'prepare-groups':
        prepare_groups(
            nc_path=args.nc_path, output_dir=args.output_dir,
            variable=args.variable, lat_dim=args.lat_dim,
            lon_dim=args.lon_dim, time_dim=args.time_dim,
            n_groups=args.n_groups,
            n_series_per_group=args.n_series_per_group,
            window_size=args.window_size, time_window=args.time_window,
            neighborhood_size=args.neighborhood_size,
            min_distance=args.min_distance, random_seed=args.random_seed,
        )
