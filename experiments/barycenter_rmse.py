#!/usr/bin/env python3
"""
Barycenter RMSE — one grid cell (n_samples, estimator, gamma).

Baselines are computed from true parameters (high-sample limit).
Each method is run n_seeds times with fresh data; mean and std RMSE are stored.

Called once per cell by scripts/study_barycenter_rmse.sh, which sweeps the full
n_samples × estimator × gamma grid and writes one result per cell.

Results written to --output-dir/results.zarr.
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import zarr
from scipy.stats import expon

from src.classification import (
    compute_barycenter_euclidean_params,
    compute_barycenter_wasserstein_sgd,
)
from src.data_generator import generate_shifted_series
from src.dataloader import estimate_parameters


def _compute_rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def compute_baselines(lambda_series, base_seed):
    """Baselines from true parameters (x_is_params=True)."""
    params = [lam.reshape(-1, 1) for lam in lambda_series]
    baseline_euc  = compute_barycenter_euclidean_params(params, gamma=1.0).flatten()
    baseline_wass = compute_barycenter_wasserstein_sgd(params, gamma=1.0).flatten()
    return baseline_euc, baseline_wass


def main():
    parser = argparse.ArgumentParser(
        description='Barycenter RMSE for one (n_samples, estimator, gamma) cell'
    )
    parser.add_argument('--n-samples',        type=int,   required=True)
    parser.add_argument('--estimator',         choices=['mle', 'log_cumulant'],
                        required=True)
    parser.add_argument('--gamma',             type=float, required=True)
    parser.add_argument('--output-dir',        required=True)
    parser.add_argument('--n-seeds',           type=int,   default=10)
    parser.add_argument('--base-seed',         type=int,   default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = generate_shifted_series(n_series=4, n_samples=1000,
                                   random_seed=args.base_seed)
    lambda_series = data['lambda_series']

    baseline_euc, baseline_wass = compute_baselines(lambda_series, args.base_seed)

    rmse_euc_list, rmse_wass_list = [], []
    for seed_idx in range(args.n_seeds):
        seed = args.base_seed + seed_idx
        np.random.seed(seed)
        # samples_series: List of (T, n_samples) — treat each row as D features
        samples_series = [
            np.array([expon.rvs(scale=1.0 / lam, size=args.n_samples,
                                random_state=seed)
                      for lam in lam_vals])
            for lam_vals in lambda_series
        ]
        params = estimate_parameters(samples_series, estimator=args.estimator)

        bary_euc  = compute_barycenter_euclidean_params(
            params, gamma=args.gamma).flatten()
        bary_wass = compute_barycenter_wasserstein_sgd(
            params, gamma=args.gamma).flatten()

        rmse_euc_list.append(_compute_rmse(bary_euc,  baseline_euc))
        rmse_wass_list.append(_compute_rmse(bary_wass, baseline_wass))

    store = zarr.open(str(out_dir / 'results.zarr'), mode='w')
    store.create_dataset('rmse_euclidean_seeds',    data=np.array(rmse_euc_list))
    store.create_dataset('rmse_wasserstein_seeds',  data=np.array(rmse_wass_list))
    store.create_dataset('baseline_euclidean',      data=baseline_euc)
    store.create_dataset('baseline_wasserstein',    data=baseline_wass)
    store.attrs.update({
        'n_samples':          args.n_samples,
        'estimator':          args.estimator,
        'gamma':              args.gamma,
        'n_seeds':            args.n_seeds,
        'base_seed':          args.base_seed,
        'mean_rmse_euclidean':    float(np.mean(rmse_euc_list)),
        'std_rmse_euclidean':     float(np.std(rmse_euc_list)),
        'mean_rmse_wasserstein':  float(np.mean(rmse_wass_list)),
        'std_rmse_wasserstein':   float(np.std(rmse_wass_list)),
        'timestamp':          datetime.now().isoformat(),
    })
    print(f'n_samples={args.n_samples}  estimator={args.estimator}  gamma={args.gamma}')
    print(f'  RMSE euclidean:    {np.mean(rmse_euc_list):.4f} ± {np.std(rmse_euc_list):.4f}')
    print(f'  RMSE wasserstein:  {np.mean(rmse_wass_list):.4f} ± {np.std(rmse_wass_list):.4f}')
    print(f'Results → {out_dir}')


if __name__ == '__main__':
    main()
