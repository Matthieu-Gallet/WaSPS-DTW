#!/usr/bin/env python3
"""
Barycenter method comparison on synthetic data — one gamma value.

Two scenarios (--scenario):
  simple   2 independent exponential series with different lambda patterns
  complex  4 temporally shifted series highlighting Wasserstein vs Euclidean

Three methods are always compared and stored together:
  euclidean_raw    — Soft-DTW DBA on raw samples (T, n_samples)
  euclidean_params — Soft-DTW DBA on estimated parameters (T, 1)
  wasserstein_sgd  — Soft-DTW/Wasserstein SGD on estimated parameters (T, 1)

Results written to --output-dir/results.zarr + one PNG comparison figure.
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import zarr

from src.classification import (
    compute_barycenter_euclidean_params,
    compute_barycenter_euclidean_raw,
    compute_barycenter_wasserstein_sgd,
)
from src.data_generator import generate_exponential_series, generate_shifted_series
from src.dataloader import estimate_parameters


def _lambda_from_raw(series: np.ndarray) -> np.ndarray:
    """Estimated lambda series from raw samples array (T, n_samples)."""
    return np.array([1.0 / np.mean(series[t]) for t in range(series.shape[0])])


def load_simple_scenario(lambda1: List[float], lambda2: List[float],
                          n_samples: int):
    """Generate two independent exponential series."""
    s1, _ = generate_exponential_series(lambda1, n_samples)   # (T, n_samples)
    s2, _ = generate_exponential_series(lambda2, n_samples)
    series = [s1.astype(np.float64), s2.astype(np.float64)]
    params = estimate_parameters(series)
    lam_est = [_lambda_from_raw(s1), _lambda_from_raw(s2)]
    return series, params, lam_est


def load_complex_scenario(n_samples: int, random_seed: int):
    """Generate four temporally shifted series."""
    data = generate_shifted_series(n_series=4, n_samples=n_samples,
                                   random_seed=random_seed)
    series = [s.astype(np.float64) for s in data['samples_series']]
    params = estimate_parameters(series)
    lam_est = data['lambda_est_series']
    return series, params, lam_est


def main():
    parser = argparse.ArgumentParser(
        description='Compare 3 barycenter methods on synthetic data — one gamma'
    )
    parser.add_argument('--scenario',    choices=['simple', 'complex'], required=True)
    parser.add_argument('--gamma',       type=float, required=True)
    parser.add_argument('--output-dir',  required=True)
    parser.add_argument('--n-samples',   type=int, default=1000)
    parser.add_argument('--max-iter',    type=int, default=100)
    parser.add_argument('--lambda1',     type=str, default='1,10,100,100,50,10,4,4',
                        help='Lambda values for series 1 (simple scenario only)')
    parser.add_argument('--lambda2',     type=str, default='0.1,0.1,0.5,0.5,5,20,80,5',
                        help='Lambda values for series 2 (simple scenario only)')
    parser.add_argument('--random-seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.random_seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'scenario={args.scenario}  gamma={args.gamma}')

    if args.scenario == 'simple':
        lambda1 = [float(x) for x in args.lambda1.split(',')]
        lambda2 = [float(x) for x in args.lambda2.split(',')]
        series, params, lam_est = load_simple_scenario(lambda1, lambda2, args.n_samples)
    else:
        series, params, lam_est = load_complex_scenario(args.n_samples, args.random_seed)

    bary_raw  = compute_barycenter_euclidean_raw(series, gamma=args.gamma,
                                                  max_iter=args.max_iter)
    bary_epar = compute_barycenter_euclidean_params(params, gamma=args.gamma,
                                                     max_iter=args.max_iter)
    bary_wass = compute_barycenter_wasserstein_sgd(params, gamma=args.gamma)

    lam_bary_raw  = _lambda_from_raw(bary_raw) if bary_raw.ndim == 2 else bary_raw.flatten()
    lam_bary_epar = bary_epar.flatten()
    lam_bary_wass = bary_wass.flatten()

    # Persist results
    store = zarr.open(str(out_dir / 'results.zarr'), mode='w')
    for i, lam in enumerate(lam_est):
        store.create_dataset(f'lambda_est_{i}', data=np.asarray(lam))
    store.create_dataset('lambda_bary_euclidean_raw',    data=lam_bary_raw)
    store.create_dataset('lambda_bary_euclidean_params', data=lam_bary_epar)
    store.create_dataset('lambda_bary_wasserstein_sgd',  data=lam_bary_wass)
    store.attrs.update({
        'scenario': args.scenario, 'gamma': args.gamma,
        'n_samples': args.n_samples, 'max_iter': args.max_iter,
        'random_seed': args.random_seed, 'n_series': len(series),
        'timestamp': datetime.now().isoformat(),
    })

    # Comparison figure — simple log-scale overlay, one panel per method
    bary_labels = [
        (lam_bary_raw,  'Euclidean raw'),
        (lam_bary_epar, 'Euclidean params'),
        (lam_bary_wass, 'Wasserstein SGD'),
    ]
    colors = ['blue', 'red', 'orange', 'purple']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (lam_bary, title) in zip(axes, bary_labels):
        for i, lam in enumerate(lam_est):
            ax.semilogy(np.arange(len(lam)), lam,
                        color=colors[i % len(colors)], alpha=0.5,
                        linewidth=1.2, label=f'Series {i+1}')
        ax.semilogy(np.arange(len(lam_bary)), lam_bary,
                    color='green', linewidth=2.5, label='Barycenter')
        ax.set_title(f'{title}\n(γ={args.gamma})', fontsize=10)
        ax.set_xlabel('Time')
        ax.set_ylabel('λ')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(out_dir / f'comparison_gamma_{args.gamma:.3g}.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Results → {out_dir}')


if __name__ == '__main__':
    main()
