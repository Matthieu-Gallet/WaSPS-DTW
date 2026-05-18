#!/usr/bin/env python3
"""
Barycenter comparison on grouped real-world data — one gamma value.

Reads Format B: --groups-dir must contain group_<i>/ subdirectories each
holding series.npy (M, T, D), plus a metadata.npy file.

Three methods are computed per group and stored together:
  euclidean_raw    — Soft-DTW DBA on raw (T, D) series
  euclidean_params — Soft-DTW DBA on estimated (T, 1) parameters
  wasserstein_sgd  — Soft-DTW/Wasserstein SGD on estimated (T, 1) parameters

Dataset-agnostic: groups may come from geographic zones, UCR classes, or any
other source prepared in Format B (see src/dataloader/discharge.py and
scripts/prepare_discharge_groups.sh for an example).

Results written to --output-dir/results.zarr (one zarr sub-group per group × method).
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import zarr

from src.classification import (
    compute_barycenter_euclidean_params,
    compute_barycenter_euclidean_raw,
    compute_barycenter_wasserstein_sgd,
)
from src.dataloader import estimate_parameters, load_groups, normalize_params, preprocess_samples


def process_group(series_arr: np.ndarray, gamma: float,
                  max_iter: int, normalize: bool = False) -> dict:
    """
    Compute three barycenters for one group.

    Args:
        series_arr: (M, T, D) float64 array — M series in the group.
        gamma:      Soft-DTW regularisation parameter.
        max_iter:   Max DBA iterations for Euclidean methods.
        normalize:  If True, z-normalize raw series and mean-ratio-normalize params.

    Returns:
        Dict with keys 'euclidean_raw', 'euclidean_params', 'wasserstein_sgd',
        each mapping to a barycenter array, plus 'params' (raw estimates).
    """
    series_raw = [series_arr[i] for i in range(series_arr.shape[0])]  # List[(T, D)]
    params_raw = estimate_parameters(series_raw)                       # List[(T, 1)]

    if normalize:
        series = [((s - s.mean()) / s.std() if s.std() > 1e-10 else s - s.mean())
                  for s in series_raw]
        params_euc = normalize_params(params_raw, mode='zscore')
        params_wss = normalize_params(params_raw, mode='log_zscore')
    else:
        series = series_raw
        params_euc = params_raw
        params_wss = params_raw

    return {
        'params':           params_raw,
        'euclidean_raw':    compute_barycenter_euclidean_raw(
            series, gamma=gamma, max_iter=max_iter),
        'euclidean_params': compute_barycenter_euclidean_params(
            params_euc, gamma=gamma, max_iter=max_iter),
        'wasserstein_sgd':  compute_barycenter_wasserstein_sgd(
            params_wss, gamma=gamma),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compare 3 barycenter methods on grouped data — one gamma'
    )
    parser.add_argument('--groups-dir',  required=True,
                        help='Format B directory (group_<i>/series.npy + metadata.npy)')
    parser.add_argument('--gamma',       type=float, required=True)
    parser.add_argument('--output-dir',  required=True)
    parser.add_argument('--max-iter',    type=int, default=50)
    parser.add_argument('--normalize',   action='store_true',
                        help='Z-norm raw series; mean-ratio-norm lambda params')
    args = parser.parse_args()

    groups, metadata = load_groups(args.groups_dir)
    group_names = metadata.get('group_names', {})
    print(f'{len(groups)} groups  gamma={args.gamma}')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = zarr.open(str(out_dir / 'results.zarr'), mode='w')

    base_attrs = {
        'groups_dir': args.groups_dir,
        'gamma':      args.gamma,
        'max_iter':   args.max_iter,
        'normalize':  args.normalize,
        'timestamp':  datetime.now().isoformat(),
    }
    store.attrs.update(base_attrs)

    for group_key, series_arr in groups.items():
        print(f'  {group_key}: shape {series_arr.shape}')
        barycenters = process_group(series_arr, gamma=args.gamma,
                                    max_iter=args.max_iter, normalize=args.normalize)
        grp = store.require_group(group_key)
        grp.create_dataset('series', data=series_arr)
        # Store estimated parameters (T, 1) for each series
        params_arr = np.stack(barycenters.pop('params'), axis=0)  # (M, T, 1)
        grp.create_dataset('params', data=params_arr)
        for method_name, bary in barycenters.items():
            grp.create_dataset(f'barycenter_{method_name}', data=bary)
        grp.attrs.update({
            **base_attrs,
            'group_key':  group_key,
            'group_name': group_names.get(group_key.replace('group_', ''), group_key),
            'n_series':   series_arr.shape[0],
            'T':          series_arr.shape[1],
            'D':          series_arr.shape[2],
        })

    print(f'Results → {out_dir}')


if __name__ == '__main__':
    main()
