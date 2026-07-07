#!/usr/bin/env python3
"""
CPAZMaL dataset loader for WaSPS-DTW.

This module provides:
- ``MLDatasetLoader``: HDF5 dataset loader (verbatim from the CPAZMaL project).
  Source: https://github.com/Matthieu-Gallet/CPAZMaL_dataset/blob/main/src/load_dataset.py
- ``download_cpazmal``: Download the dataset from HuggingFace.
- ``windows_to_time_series``: Reshape spatial windows (W, W, T) → (T, W²).
- ``extract_time_series``: Extract one windowed time series per spatial sample,
  with a geographic ``group`` index, for group-stratified K-fold classification
  (mirrors ``data.river_loader.load_river_classification``).

The output format of ``extract_time_series`` is compatible with the WaSPS-DTW
pipeline (``WassersteinDistance``, ``sdtw_barycenter``, ``sgd_barycenter``).

Dataset: https://huggingface.co/datasets/musmb/CPAZMaL
"""

import h5py
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from tqdm import tqdm
from joblib import Parallel, delayed


# =============================================================================
# MLDatasetLoader — verbatim from CPAZMaL_dataset/src/load_dataset.py
# =============================================================================

class MLDatasetLoader:
    """Class to efficiently load the optimized HDF5 dataset with window extraction."""

    def __init__(self, hdf5_path: str):
        """
        Args:
            hdf5_path: Path to the CPAZMaL HDF5 file.
        """
        self.hdf5_path = hdf5_path
        self.file = None
        self._load_metadata()

    def _load_metadata(self):
        """Load metadata in memory for fast access."""
        with h5py.File(self.hdf5_path, 'r') as f:
            meta = f['metadata']
            self.classes = json.loads(meta.attrs['classes'])
            self.n_groups = meta.attrs['n_total_groups']
            self.nodata = meta.attrs['nodata_value']

            self.class_index = {}
            for class_name in f['index/by_class'].keys():
                entries_json = f[f'index/by_class/{class_name}'].attrs['entries_json']
                self.class_index[class_name] = json.loads(entries_json)

            temp_ranges_json = f['index/temporal_ranges'].attrs['ranges_json']
            self.temporal_ranges = json.loads(temp_ranges_json)

    def __enter__(self):
        self.file = h5py.File(self.hdf5_path, 'r')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def get_group_info(self, group_name: str) -> Dict:
        """Return geographic and structural information for a group."""
        with h5py.File(self.hdf5_path, 'r') as f:
            if group_name not in f['data']:
                raise ValueError(f"Group {group_name} not found")
            group = f['data'][group_name]
            return {
                'class': group.attrs['class'],
                'latitude': group.attrs['latitude'],
                'longitude': group.attrs['longitude'],
                'elevation': group.attrs['elevation'],
                'orientation': group.attrs['orientation'],
                'slope': group.attrs['slope'],
                'orbits': list(group.keys()),
            }

    def extract_windows(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        window_size: int,
        stride: Optional[int] = None,
        max_mask_value: int = 3,
        max_mask_percentage: float = 100.0,
        min_valid_percentage: float = 50.0,
        skip_optim_offset: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """
        Extract non-overlapping windows from an image with quality filtering.

        Optionally optimises the starting offset (start_y, start_x) to maximise
        the number of valid windows (requires ``skip_optim_offset=False``).

        Args:
            image: ``(H, W)``, ``(H, W, C)``, ``(H, W, T)`` or ``(H, W, C, T)``
            mask: ``(H, W)`` or ``(H, W, T)``
            window_size: Square window side length.
            stride: Step between windows (default = window_size).
            max_mask_value: Maximum accepted mask value (0–3).
            max_mask_percentage: Max percentage of pixels with mask > max_mask_value.
            min_valid_percentage: Min percentage of valid (non-nodata) pixels.
            skip_optim_offset: If True, skip offset optimisation and use (0, 0).

        Returns:
            windows: Array of extracted windows or None.
            window_masks: Corresponding mask array or None.
            positions: List of ``(y, x)`` positions.
        """
        if stride is None:
            stride = window_size

        if image.ndim == 2:
            h, w = image.shape
            has_channels = False
            has_time = False
        elif image.ndim == 3:
            h, w, c = image.shape
            has_channels = True
            has_time = False
        elif image.ndim == 4:
            h, w, c, t = image.shape
            has_channels = True
            has_time = True

        mask_has_time = mask.ndim == 3

        if skip_optim_offset:
            best_start_y = 0
            best_start_x = 0
        else:
            def count_valid_windows(start_y, start_x):
                count = 0
                for y in range(start_y, h - window_size + 1, stride):
                    for x in range(start_x, w - window_size + 1, stride):
                        if mask_has_time:
                            window_mask = mask[y:y+window_size, x:x+window_size, :]
                            bad_pixels = np.any(window_mask > max_mask_value, axis=-1)
                        else:
                            window_mask = mask[y:y+window_size, x:x+window_size]
                            bad_pixels = window_mask > max_mask_value

                        bad_pct = (np.sum(bad_pixels) / (window_size * window_size)) * 100.0

                        if image.ndim == 2:
                            win = image[y:y+window_size, x:x+window_size]
                        elif image.ndim == 3:
                            win = image[y:y+window_size, x:x+window_size, :]
                        else:
                            win = image[y:y+window_size, x:x+window_size, :, :]

                        if has_time:
                            is_invalid = (win == self.nodata) | np.isnan(win)
                            valid_pixels = np.all(~is_invalid, axis=-1)
                            if has_channels:
                                valid_pixels = np.all(valid_pixels, axis=-1)
                        else:
                            is_invalid = (win == self.nodata) | np.isnan(win)
                            if has_channels:
                                valid_pixels = np.all(~is_invalid, axis=-1)
                            else:
                                valid_pixels = ~is_invalid
                        valid_pct = (np.sum(valid_pixels) / (window_size * window_size)) * 100.0

                        if bad_pct <= max_mask_percentage and valid_pct >= min_valid_percentage:
                            count += 1
                return count

            max_offset = min(stride, window_size)
            offsets_to_test = [
                (sy, sx)
                for sy in range(max_offset)
                for sx in range(max_offset)
                if sy + window_size <= h and sx + window_size <= w
            ]
            counts = Parallel(n_jobs=-1)(
                delayed(count_valid_windows)(sy, sx)
                for sy, sx in tqdm(offsets_to_test, desc="Optimising offset", leave=False)
            )
            if counts:
                best_idx = int(np.argmax(counts))
                best_start_y, best_start_x = offsets_to_test[best_idx]
            else:
                best_start_y = best_start_x = 0

        windows, window_masks, positions = [], [], []
        for y in range(best_start_y, h - window_size + 1, stride):
            for x in range(best_start_x, w - window_size + 1, stride):
                if image.ndim == 2:
                    window = image[y:y+window_size, x:x+window_size]
                elif image.ndim == 3:
                    window = image[y:y+window_size, x:x+window_size, :]
                else:
                    window = image[y:y+window_size, x:x+window_size, :, :]

                if mask_has_time:
                    window_mask = mask[y:y+window_size, x:x+window_size, :]
                    bad_pixels = np.any(window_mask > max_mask_value, axis=-1)
                else:
                    window_mask = mask[y:y+window_size, x:x+window_size]
                    bad_pixels = window_mask > max_mask_value

                bad_pct = (np.sum(bad_pixels) / (window_size * window_size)) * 100.0

                if has_time:
                    is_invalid = (window == self.nodata) | np.isnan(window)
                    valid_pixels = np.all(~is_invalid, axis=-1)
                    if has_channels:
                        valid_pixels = np.all(valid_pixels, axis=-1)
                else:
                    is_invalid = (window == self.nodata) | np.isnan(window)
                    if has_channels:
                        valid_pixels = np.all(~is_invalid, axis=-1)
                    else:
                        valid_pixels = ~is_invalid
                valid_pct = (np.sum(valid_pixels) / (window_size * window_size)) * 100.0

                if bad_pct <= max_mask_percentage and valid_pct >= min_valid_percentage:
                    windows.append(window.astype(np.float32))
                    window_masks.append(window_mask)
                    positions.append((y, x))

        if not windows:
            return None, None, []
        return np.array(windows), np.array(window_masks), positions

    def load_data(
        self,
        group_name: str,
        orbit: str = 'DSC',
        polarisation: Union[str, List[str]] = 'HH',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        normalize: bool = False,
        remove_nodata: bool = True,
        scale_type: str = 'intensity',
    ) -> Dict:
        """
        Load SAR imagery for a group over an optional date range.

        Args:
            group_name: Group identifier (e.g. ``'ABL001'``).
            orbit: ``'ASC'`` or ``'DSC'``.
            polarisation: ``'HH'``, ``'HV'``, or ``['HH', 'HV']`` for dual-pol.
            start_date: Inclusive start date ``'YYYYMMDD'``.
            end_date: Inclusive end date ``'YYYYMMDD'``.
            normalize: Normalise with pre-computed per-polarisation stats.
            remove_nodata: Replace nodata sentinel with NaN.
            scale_type: ``'intensity'`` (raw), ``'amplitude'`` (sqrt), ``'log10'``.

        Returns:
            Dict with keys: ``images``, ``masks``, ``timestamps``,
            ``angles_incidence``, ``metadata``, ``group``, ``orbit``.
        """
        with h5py.File(self.hdf5_path, 'r') as f:
            if isinstance(polarisation, list):
                data_list = []
                for pol in polarisation:
                    path = f'data/{group_name}/{orbit}/{pol}'
                    if path not in f:
                        raise ValueError(f"Path {path} not found in dataset")
                    data_list.append(f[path])

                timestamps_hh = data_list[0]['timestamps'][:]
                timestamps_hv = data_list[1]['timestamps'][:]
                common_ts = np.intersect1d(timestamps_hh, timestamps_hv)

                if len(common_ts) == 0:
                    raise ValueError(f"No common timestamps between HH and HV for {group_name}")

                if start_date or end_date:
                    mask_ts = np.ones(len(common_ts), dtype=bool)
                    if start_date:
                        mask_ts &= common_ts >= start_date.encode('utf-8')
                    if end_date:
                        mask_ts &= common_ts <= end_date.encode('utf-8')
                    common_ts = common_ts[mask_ts]

                if len(common_ts) == 0:
                    raise ValueError("No data in specified date range")

                min_h, min_w = None, None
                for pol, data_pol in zip(polarisation, data_list):
                    ts_pol = data_pol['timestamps'][:]
                    indices = [np.where(ts_pol == ts)[0][0] for ts in common_ts]
                    img_pol = data_pol['images'][:, :, indices]
                    h, w, _ = img_pol.shape
                    min_h = h if min_h is None else min(min_h, h)
                    min_w = w if min_w is None else min(min_w, w)

                images_list, masks_list = [], []
                for pol, data_pol in zip(polarisation, data_list):
                    ts_pol = data_pol['timestamps'][:]
                    indices = [np.where(ts_pol == ts)[0][0] for ts in common_ts]
                    images_list.append(data_pol['images'][:min_h, :min_w, indices])
                    masks_list.append(data_pol['masks'][:min_h, :min_w, indices])

                images = np.stack(images_list, axis=-1)
                masks = np.maximum(masks_list[0], masks_list[1])
                timestamps = common_ts
                angles = data_list[0]['angles_incidence'][:][
                    [np.where(data_list[0]['timestamps'][:] == ts)[0][0] for ts in common_ts]
                ]
                metadata = {'polarisation': polarisation, 'dual_pol': True}

            else:
                path = f'data/{group_name}/{orbit}/{polarisation}'
                if path not in f:
                    raise ValueError(f"Path {path} not found in dataset")

                pol_data = f[path]
                images = pol_data['images'][:]
                masks = pol_data['masks'][:]
                timestamps = pol_data['timestamps'][:]
                angles = pol_data['angles_incidence'][:]

                if start_date or end_date:
                    mask_ts = np.ones(len(timestamps), dtype=bool)
                    if start_date:
                        mask_ts &= timestamps >= start_date.encode('utf-8')
                    if end_date:
                        mask_ts &= timestamps <= end_date.encode('utf-8')
                    if not np.any(mask_ts):
                        raise ValueError("No data in specified date range")
                    images = images[:, :, mask_ts]
                    masks = masks[:, :, mask_ts]
                    timestamps = timestamps[mask_ts]
                    angles = angles[mask_ts]

                metadata = {
                    'mean': pol_data.attrs['stat_mean'],
                    'std': pol_data.attrs['stat_std'],
                    'min': pol_data.attrs['stat_min'],
                    'max': pol_data.attrs['stat_max'],
                    'n_samples': pol_data.attrs['n_timestamps'],
                    'polarisation': polarisation,
                    'dual_pol': False,
                }

            if remove_nodata:
                images = np.where(images == self.nodata, np.nan, images)

            if scale_type == 'amplitude':
                images = np.where(images >= 0, np.sqrt(images), np.nan).astype(np.float32)
            elif scale_type == 'log10':
                images = np.where(images > 0, np.log10(images), np.nan)

            if normalize and not isinstance(polarisation, list):
                std = metadata['std']
                if std > 0:
                    images = (images - metadata['mean']) / std

            return {
                'images': images,
                'masks': masks,
                'timestamps': [t.decode('utf-8') for t in timestamps],
                'angles_incidence': angles,
                'metadata': metadata,
                'group': group_name,
                'orbit': orbit,
            }

    def get_groups_by_class(self, class_name: str) -> List[str]:
        """Return the list of group names for a given class."""
        if class_name not in self.class_index:
            return []
        return [entry['group'] for entry in self.class_index[class_name]]

    def get_all_groups_with_classes(self) -> Dict[str, str]:
        """Return ``{group_name: class_name}`` for all groups."""
        group_to_class = {}
        for class_name in self.classes:
            for group in self.get_groups_by_class(class_name):
                group_to_class[group] = class_name
        return group_to_class

    def get_statistics_summary(self) -> Dict:
        """Return a high-level summary of group/class statistics."""
        stats = {
            'by_class': {},
            'global': {'n_groups': self.n_groups, 'n_classes': len(self.classes)},
        }
        for class_name in self.classes:
            groups = self.get_groups_by_class(class_name)
            stats['by_class'][class_name] = {'n_groups': len(groups), 'groups': groups}
        return stats


# =============================================================================
# Download helper
# =============================================================================

def download_cpazmal(save_dir: str, token: Optional[str] = None) -> str:
    """
    Download the CPAZMaL dataset from HuggingFace.

    Requires the ``huggingface_hub`` package (``pip install huggingface_hub``).

    Args:
        save_dir: Local directory where the dataset will be stored.
        token: Optional HuggingFace access token for private repositories.

    Returns:
        Path to the downloaded HDF5 file.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download CPAZMaL. "
            "Install it with: pip install huggingface_hub"
        )

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    local_dir = snapshot_download(
        repo_id="musmb/CPAZMaL",
        repo_type="dataset",
        local_dir=str(save_path),
        token=token,
    )

    hdf5_files = list(Path(local_dir).glob("*.hdf5"))
    if not hdf5_files:
        hdf5_files = list(Path(local_dir).glob("**/*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(
            f"No HDF5 file found after downloading to {local_dir}. "
            "Check the repository contents manually."
        )
    return str(hdf5_files[0])


# =============================================================================
# Reshape helper
# =============================================================================

def windows_to_time_series(window: np.ndarray) -> np.ndarray:
    """
    Reshape a spatial window to a time series compatible with WassersteinDistance.

    The CPAZMaL windows have shape ``(W, W, T)`` (spatial × temporal).
    WassersteinDistance expects ``(T, n_samples)`` where each row contains
    the W² pixel values observed at time step t (treated as i.i.d. samples
    from an exponential distribution).

    Args:
        window: Array of shape ``(W, W, T)``.

    Returns:
        Array of shape ``(T, W²)``.
    """
    W1, W2, T = window.shape
    # (W, W, T) → (T, W, W) → (T, W²)
    return window.transpose(2, 0, 1).reshape(T, W1 * W2).astype(np.float64)


# =============================================================================
# Dataset extraction — adapted from scenarios.py::scenario_2
# =============================================================================

def extract_time_series(
    loader: MLDatasetLoader,
    window_size: int = 12,
    max_mask_value: int = 1,
    max_mask_percentage: float = 10.0,
    min_valid_percentage: float = 50.0,
    orbit: str = 'DSC',
    polarization: str = 'HH',
    start_date: str = '20200101',
    end_date: str = '20201231',
    scale_type: str = 'amplitude',
    skip_optim_offset: bool = False,
    verbose: bool = True,
    exclude_classes: Tuple[str, ...] = ('STUDY', 'HAG'),
    max_groups_per_class: Optional[int] = None,
) -> Dict:
    """
    Extract windowed time series from the CPAZMaL dataset for classification.

    Each spatial window yields **one continuous time series** over
    ``[start_date, end_date]``, reshaped ``(W, W, T)`` → ``(T, W²)``, paired with
    a geographic ``group`` index. Use ``StratifiedGroupKFold`` on ``(y, groups)``
    for K-fold classification — mirrors ``river_loader.load_river_classification``.

    (Earlier versions of this function split the data into a *training* period
    and a *prediction* period for a temporal-forecasting task; that framing is
    gone — this is a classification loader now.)

    Args:
        loader: An ``MLDatasetLoader`` instance pointing to the HDF5 file.
        window_size: Side length of the square spatial window (pixels).
        max_mask_value: Maximum accepted mask value (0–3).
        max_mask_percentage: Max % of pixels with mask > max_mask_value.
        min_valid_percentage: Min % of valid (non-nodata) pixels.
        orbit: SAR orbit direction (``'ASC'`` or ``'DSC'``).
        polarization: SAR polarisation (``'HH'`` or ``'HV'``).
        start_date: Period start (``'YYYYMMDD'``).
        end_date: Period end (``'YYYYMMDD'``).
        scale_type: Radiometric scaling (``'intensity'``, ``'amplitude'``,
            ``'log10'``).
        skip_optim_offset: If True, skip the window-offset optimisation step
            (faster but may yield fewer windows).
        verbose: Print progress and summary statistics.
        exclude_classes: Class names dropped from the learning set. ``'STUDY'``
            is a non-semantic administrative class; ``'HAG'`` has too few
            samples (45 samples / 28 groups in the live dataset) to be useful
            for stratified K-fold.
        max_groups_per_class: If set, keep at most this many groups **per
            class** (not a flat global cap) — needed to build a multi-class
            smoke-test subset. Groups are sorted alphabetically overall, and a
            flat cap (the old ``max_groups``) only ever hits one class (e.g.
            all ``'ABL*'`` groups) since class names cluster alphabetically.
            ``None`` = all groups.

    Returns:
        Dict with:

        - ``X``: object array ``(N,)`` of ``(T, W²)`` float64 arrays.
        - ``y``: int array ``(N,)`` — class label per sample.
        - ``groups``: int array ``(N,)`` — group index per sample.
        - ``masks``: object array ``(N,)`` of ``(W, W, T)`` masks.
        - ``timestamps``: object array ``(N,)`` of timestamp lists.
        - ``class_names``: dict ``{int: str}`` mapping label → class name.
        - ``group_names``: dict ``{int: str}`` mapping group index → group name.
        - ``metadata``: dict with extraction parameters.
    """
    if verbose:
        print(f"\n{'='*70}")
        print("CPAZMaL: Time Series Extraction (for WaSPS-DTW classification)")
        print(f"{'='*70}")
        print(f"  Window:      {window_size}×{window_size}")
        print(f"  Orbit:       {orbit}  |  Polarisation: {polarization}")
        print(f"  Period:      {start_date} – {end_date}")
        print(f"  Scale type:  {scale_type}")
        print(f"  Excluded:    {list(exclude_classes)}")
        print(f"  Mask max:    ≤{max_mask_value} ({max_mask_percentage}%)\n")

    learning_classes = [c for c in loader.classes if c not in exclude_classes]
    class_to_int = {c: i for i, c in enumerate(learning_classes)}
    group_to_class = loader.get_all_groups_with_classes()

    if max_groups_per_class is not None:
        # Per-class cap so a small smoke-test subset still spans every class —
        # a flat global cap on the alphabetically-sorted list would not (class
        # name prefixes cluster alphabetically, e.g. all 'ABL*' groups first).
        unique_groups = sorted(
            g for cls in learning_classes
            for g in sorted(loader.get_groups_by_class(cls))[:max_groups_per_class]
        )
    else:
        all_groups = []
        for cls in learning_classes:
            all_groups.extend(loader.get_groups_by_class(cls))
        unique_groups = sorted(set(all_groups))
    group_to_int = {g: i for i, g in enumerate(unique_groups)}

    X_all, y_all, groups_all = [], [], []
    masks_all, timestamps_all = [], []

    pbar = tqdm(unique_groups, desc="Groups", unit="grp")
    for group_name in pbar:
        class_name = group_to_class.get(group_name)
        if class_name not in class_to_int:
            continue
        if verbose:
            pbar.set_postfix_str(f"{group_name} ({class_name})")

        try:
            data = loader.load_data(
                group_name=group_name,
                orbit=orbit,
                polarisation=polarization,
                start_date=start_date,
                end_date=end_date,
                normalize=False,
                remove_nodata=True,
                scale_type=scale_type,
            )
        except (ValueError, KeyError):
            continue

        img = data['images']    # (H, W, T)
        mask = data['masks']

        if img.shape[2] == 0:
            continue

        windows, wm, _positions = loader.extract_windows(
            image=img,
            mask=mask,
            window_size=window_size,
            stride=window_size,
            max_mask_value=max_mask_value,
            max_mask_percentage=max_mask_percentage,
            min_valid_percentage=min_valid_percentage,
            skip_optim_offset=skip_optim_offset,
        )

        if windows is None:
            continue

        ts = data['timestamps']
        cls_int = class_to_int[class_name]
        grp_int = group_to_int[group_name]

        for k in range(len(windows)):
            # Reshape (W, W, T) → (T, W²) for WassersteinDistance compatibility
            X_all.append(windows_to_time_series(windows[k]))
            y_all.append(cls_int)
            groups_all.append(grp_int)
            masks_all.append(wm[k])
            timestamps_all.append(ts)

    if not X_all:
        raise ValueError(
            "No windows were extracted. Check dataset path and parameters "
            "(orbit, polarization, date range, mask thresholds)."
        )

    X = np.empty(len(X_all), dtype=object)
    for i in range(len(X_all)):
        X[i] = X_all[i]

    masks = np.empty(len(masks_all), dtype=object)
    ts_arr = np.empty(len(timestamps_all), dtype=object)
    for i in range(len(masks_all)):
        masks[i] = masks_all[i]
        ts_arr[i] = timestamps_all[i]

    y = np.array(y_all, dtype=np.int32)
    groups = np.array(groups_all, dtype=np.int32)

    if verbose:
        print(f"\n{'='*70}")
        print(f"Extraction complete:")
        print(f"  Total samples (windows):  {len(X)}")
        print(f"  X[0].shape:               {X[0].shape}  (T, W²)")
        labels, counts = np.unique(y, return_counts=True)
        class_names_map = {v: k for k, v in class_to_int.items()}
        for lbl, cnt in zip(labels, counts):
            print(f"    Class {lbl:2d} ({class_names_map[lbl]:15s}): {cnt} samples")

    return {
        'X': X,
        'y': y,
        'groups': groups,
        'masks': masks,
        'timestamps': ts_arr,
        'class_names': {v: k for k, v in class_to_int.items()},
        'group_names': {v: k for k, v in group_to_int.items()},
        'metadata': {
            'window_size': window_size,
            'orbit': orbit,
            'polarization': polarization,
            'period': (start_date, end_date),
            'scale_type': scale_type,
            'exclude_classes': list(exclude_classes),
            'note': (
                'X[i] has shape (T, W²). Each row = W² pixel samples at time '
                'step t, usable directly as input to WassersteinDistance.'
            ),
        },
    }
