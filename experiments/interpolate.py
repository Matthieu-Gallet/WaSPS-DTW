#!/usr/bin/env python3
"""
Barycenter geodesic interpolation between two class barycenters.

Given two barycenter arrays A and B (stored in a classify.py results.zarr),
compute K evenly-spaced interpolants along the Soft-DTW geodesic:

  B(t) = argmin_z  (1-t) * SDTW(z, A)  +  t * SDTW(z, B),   t in [0, 1]

which is solved by running the barycenter algorithm on the weighted pair.

All three barycenter methods are supported:
  euclidean_raw    — interpolate in raw (T, D) space
  euclidean_params — interpolate in parameter (T, 1) space
  wasserstein_sgd  — interpolate in parameter space with Wasserstein distance

Results written to --output-dir/results.zarr.
  interpolants_<method>:  (K, T, D) or (K, T, 1) array of interpolants
  t_values:               (K,) array of t values in [0, 1]

Usage:
  python experiments/interpolate.py \
      --zarr-path  results/study_kfold/.../results.zarr \
      --label-a    0 \
      --label-b    1 \
      --method     euclidean_params \
      --gamma      1.0 \
      --k-steps    7 \
      --output-dir results/interp_0_vs_1/
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


_BARYCENTER_FN = {
    'euclidean_raw':    compute_barycenter_euclidean_raw,
    'euclidean_params': compute_barycenter_euclidean_params,
    'wasserstein_sgd':  compute_barycenter_wasserstein_sgd,
}


def _interpolate_pair(bary_a: np.ndarray, bary_b: np.ndarray,
                      t: float, method: str, gamma: float,
                      max_iter: int) -> np.ndarray:
    """
    Single interpolant at position t between bary_a and bary_b.

    Solved as the weighted barycenter of the two endpoints:
      B(t) = argmin_z  (1-t) * d(z, A)  +  t * d(z, B)

    This is equivalent to passing [A, A, ..., B, B, ...] with the appropriate
    number of copies.  For a differentiable alternative we use weight-scaled
    copies: repeat each endpoint to match the discrete weight ratio.
    """
    if t <= 0.0:
        return bary_a.copy()
    if t >= 1.0:
        return bary_b.copy()

    # Represent weights via integer repetitions (resolution = 1/20)
    n_total = 20
    n_b = max(1, round(t * n_total))
    n_a = n_total - n_b
    samples = [bary_a] * n_a + [bary_b] * n_b
    return _BARYCENTER_FN[method](samples, gamma=gamma, max_iter=max_iter)


def main():
    parser = argparse.ArgumentParser(
        description='Geodesic interpolation between two class barycenters'
    )
    parser.add_argument('--zarr-path',   required=True,
                        help='Path to a classify.py results.zarr')
    parser.add_argument('--label-a',     type=int, required=True,
                        help='Integer class label for endpoint A')
    parser.add_argument('--label-b',     type=int, required=True,
                        help='Integer class label for endpoint B')
    parser.add_argument('--method',
                        choices=['euclidean_raw', 'euclidean_params', 'wasserstein_sgd'],
                        default='euclidean_params')
    parser.add_argument('--gamma',       type=float, default=1.0)
    parser.add_argument('--k-steps',     type=int,   default=7,
                        help='Number of interpolants including endpoints')
    parser.add_argument('--max-iter',    type=int,   default=50)
    parser.add_argument('--output-dir',  required=True)
    args = parser.parse_args()

    src_store = zarr.open(str(args.zarr_path), mode='r')
    method_grp = src_store[args.method]
    bary_a = method_grp[f'barycenter_{args.label_a}'][:]
    bary_b = method_grp[f'barycenter_{args.label_b}'][:]

    t_values = np.linspace(0.0, 1.0, args.k_steps)
    print(f'Interpolating {args.label_a} → {args.label_b}  '
          f'method={args.method}  gamma={args.gamma}  steps={args.k_steps}')

    interpolants = []
    for i, t in enumerate(t_values):
        print(f'  step {i+1}/{args.k_steps}  t={t:.3f}')
        interp = _interpolate_pair(bary_a, bary_b, t, args.method,
                                   args.gamma, args.max_iter)
        interpolants.append(interp)

    interpolants_arr = np.stack(interpolants, axis=0)  # (K, T, D)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = zarr.open(str(out_dir / 'results.zarr'), mode='w')
    store.create_dataset(f'interpolants_{args.method}', data=interpolants_arr)
    store.create_dataset('t_values', data=t_values)
    store.attrs.update({
        'zarr_path':   str(args.zarr_path),
        'label_a':     args.label_a,
        'label_b':     args.label_b,
        'method':      args.method,
        'gamma':       args.gamma,
        'k_steps':     args.k_steps,
        'max_iter':    args.max_iter,
        'timestamp':   datetime.now().isoformat(),
    })
    print(f'Results → {out_dir}')


if __name__ == '__main__':
    main()
