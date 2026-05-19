#!/usr/bin/env python3
"""
Replot barycenter figures from a saved experiment_data.pkl without re-running
the expensive classification experiment.

Usage (from repo root):
    source venv/bin/activate
    python src/experiments/replot_barycenters.py \
        --pkl src/results/regime_classification/one-shot/experiment_data.pkl \
        --output-dir src/results/regime_classification/one-shot \
        --n-samples-plot 15
"""

import sys
import argparse
import pickle
from pathlib import Path

# Path setup — same convention as other experiment scripts
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plot.classification_plots import (
    plot_class_pair_barycenters,
    plot_all_class_barycenters_grid,
)


def main():
    parser = argparse.ArgumentParser(description="Replot barycenters from saved pkl")
    parser.add_argument(
        '--pkl',
        default='src/results/regime_classification/one-shot/experiment_data.pkl',
        help='Path to experiment_data.pkl',
    )
    parser.add_argument(
        '--output-dir',
        default='src/results/regime_classification/one-shot',
        help='Output directory (pair plots and grids written into subdirs)',
    )
    parser.add_argument('--n-samples-plot', type=int, default=15)
    parser.add_argument('--no-pdf', action='store_true', help='Skip PDF generation')
    args = parser.parse_args()

    pkl_path = Path(args.pkl)
    if not pkl_path.exists():
        print(f"ERROR: pkl not found at {pkl_path}")
        sys.exit(1)

    print(f"Loading experiment data from {pkl_path} ...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    results        = data['results']
    X_train_raw    = data['X_train_raw']
    X_train_params = data['X_train_params']
    Y_train        = data['Y_train']
    idx_to_regime  = data['idx_to_regime']

    output_dir = args.output_dir
    save_pdf   = not args.no_pdf

    # ------------------------------------------------------------------ #
    # Class-pair plots (all methods overlaid, one figure per pair)        #
    # ------------------------------------------------------------------ #
    print("Generating class-pair barycenter plots ...")
    barycenters_by_method = {
        mk: results[mk]['barycenters']
        for mk in ['euclidean_raw', 'euclidean_params', 'wasserstein_params']
        if mk in results
    }
    plot_class_pair_barycenters(
        barycenters_by_method=barycenters_by_method,
        X_train_raw=X_train_raw,
        X_train_params=X_train_params,
        Y_train=Y_train,
        idx_to_regime=idx_to_regime,
        output_dir=output_dir,
        save_pdf=save_pdf,
        n_samples=args.n_samples_plot,
    )
    print(f"  → saved to {output_dir}/class_pairs/")

    # ------------------------------------------------------------------ #
    # Grid plots (all classes per method, one figure per parameter)       #
    # ------------------------------------------------------------------ #
    print("Generating grid barycenter plots ...")

    methods = [
        ('euclidean_raw',      'Soft-DTW Euclidean (Raw Data)',        X_train_raw,    True),
        ('euclidean_params',   'Soft-DTW Euclidean (Parameters)',      X_train_params, False),
        ('wasserstein_params', 'Soft-DTW Wasserstein (Parameters)',    X_train_params, False),
    ]

    for method_key, method_name, train_data, is_raw in methods:
        if method_key not in results:
            print(f"  Skipping {method_key} (not in results)")
            continue
        print(f"  {method_name} ...")
        plot_all_class_barycenters_grid(
            barycenters=results[method_key]['barycenters'],
            X_train=train_data,
            Y_train=Y_train,
            idx_to_regime=idx_to_regime,
            method_name=method_name,
            output_dir=output_dir,
            save_pdf=save_pdf,
            n_samples=args.n_samples_plot,
            is_raw=is_raw,
        )
    print(f"  → saved to {output_dir}/barycenter_grids/")
    print("Done.")


if __name__ == '__main__':
    main()
