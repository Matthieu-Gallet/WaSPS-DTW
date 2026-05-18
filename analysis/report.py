#!/usr/bin/env python3
"""
Batch figure generator for a completed study campaign.

Discovers all results.zarr files under --results-dir, generates every standard
figure, and writes them to --output-dir.

Usage:
    python analysis/report.py \
        --results-dir  results/ \
        --output-dir   results/report_YYYYMMDD/ \
        [--kfold-study    results/study_classification_kfold_*/]  \
        [--gamma-study    results/study_classification_gamma_*/]  \
        [--samples-study  results/study_classification_samples_*/]\
        [--rmse-study     results/study_barycenter_rmse_*/]       \
        [--interp-study   results/study_interpolation_*/]

All --*-study arguments accept glob patterns or exact paths.
If not provided, the script attempts to find the most recent study of each type
under --results-dir by matching the directory name prefix.

Figures produced
----------------
  class_barycenters/       one PDF per (class, method)
  confusion_matrices/      one PDF per method (kfold mean)
  gamma_sensitivity.pdf    F1 vs gamma, all methods
  sample_sensitivity.pdf   F1 vs training fraction, all methods
  rmse_vs_samples.pdf      RMSE convergence plot
  interpolation/           one PDF per (method, class-pair)
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import analysis.plot as P
from analysis.load import (
    aggregate_folds,
    filter_results,
    load_classification,
    load_results,
    to_dataframe,
)

P.SAVE_FIGURES = True

METHODS = ['euclidean_raw', 'euclidean_params', 'wasserstein_sgd']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _most_recent(results_dir: Path, prefix: str) -> Path:
    """Return the most recently created directory matching prefix under results_dir."""
    candidates = sorted(results_dir.glob(f'{prefix}*/'), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    return candidates[-1]


def _resolve_study(arg: str, results_dir: Path, prefix: str) -> Path:
    """Resolve a study path from a CLI arg or by auto-discovery."""
    if arg:
        p = Path(arg)
        if not p.exists():
            # Try as glob
            hits = sorted(results_dir.glob(arg), key=lambda x: x.stat().st_mtime)
            if hits:
                return hits[-1]
        return p
    return _most_recent(results_dir, prefix)


# ---------------------------------------------------------------------------
# Per-study report sections
# ---------------------------------------------------------------------------

def report_kfold(kfold_dir: Path, data_dir: Path, output_dir: Path):
    """Class barycenters + confusion matrices from a kfold study."""
    results = load_results(str(kfold_dir))
    if not results:
        print(f'  [kfold] no results found in {kfold_dir}')
        return

    _, _, metadata = load_classification(str(data_dir))
    idx_to_label = metadata.get('idx_to_label', {})

    # Group results by method; collect predictions/Y_test from first fold
    by_method = {}
    for r in results:
        m = r.get('method') or r.get('_group')
        if m not in by_method:
            by_method[m] = r

    # Confusion matrices
    cm_dir = output_dir / 'confusion_matrices'
    results_by_method = {
        m: {'Y_test': r['Y_test'], 'predictions': r['predictions'],
            'f1_weighted': r.get('f1_weighted', 0.0)}
        for m, r in by_method.items()
        if 'Y_test' in r and 'predictions' in r
    }
    if results_by_method:
        P.plot_confusion_matrices_all_methods(results_by_method, idx_to_label,
                                              str(cm_dir))
        print(f'  confusion matrices → {cm_dir}')

    # Class barycenters — need training data
    bary_dir = output_dir / 'class_barycenters'
    from src.dataloader import (load_classification as src_load, preprocess_samples,
                                 estimate_parameters, normalize_params)
    X_arr, Y_arr, _ = src_load(str(data_dir))
    # Barycenters are stored in normalized space; match normalization per method
    X_raw_norm  = preprocess_samples(X_arr, normalize=True)     # z-normalized raw
    X_raw_pos   = preprocess_samples(X_arr, normalize=False)    # for param estimation
    X_params_r  = estimate_parameters(X_raw_pos)
    X_params_z  = normalize_params(X_params_r, mode='log_zscore_linear')  # for euclidean_params
    X_params_lg = normalize_params(X_params_r, mode='log_zscore')         # for wasserstein_sgd

    _input_for_method = {
        'euclidean_raw':    X_raw_norm,
        'euclidean_params': X_params_z,
        'wasserstein_sgd':  X_params_lg,
    }

    for method, r in by_method.items():
        barycenters = {int(k.replace('barycenter_', '')): r[k]
                       for k in r if k.startswith('barycenter_') and
                       isinstance(r[k], np.ndarray)}
        if not barycenters:
            continue
        X_for_method = _input_for_method.get(method, X_raw_norm)
        P.plot_class_barycenters(barycenters, X_for_method, Y_arr,
                                 idx_to_label, method_name=method,
                                 output_dir=str(bary_dir))
    print(f'  class barycenters → {bary_dir}')


def report_gamma(gamma_dir: Path, output_dir: Path):
    """Gamma sensitivity plot (all methods)."""
    results = load_results(str(gamma_dir))
    if not results:
        print(f'  [gamma] no results found in {gamma_dir}')
        return

    raw = aggregate_folds(results, group_by=['gamma', 'method'])
    # Reorganise: {method: {gamma: [f1_values]}}
    by_method: dict = {}
    for (gamma, method), metrics in raw.items():
        if method is None:
            continue
        by_method.setdefault(method, {})[gamma] = metrics.get('f1_weighted', [])

    fig = P.plot_gamma_sensitivity(by_method, output_dir=str(output_dir))
    plt.close(fig)
    print(f'  gamma sensitivity → {output_dir}/gamma_sensitivity.pdf')


def report_samples(samples_dir: Path, output_dir: Path):
    """Sample-size sensitivity plot (all methods)."""
    results = load_results(str(samples_dir))
    if not results:
        print(f'  [samples] no results found in {samples_dir}')
        return

    df = to_dataframe(results)
    # n_train as fraction proxy: use n_train directly
    raw = aggregate_folds(results, group_by=['n_train', 'method'])
    by_method: dict = {}
    for (n_train, method), metrics in raw.items():
        if method is None:
            continue
        by_method.setdefault(method, {})[n_train] = metrics.get('f1_weighted', [])

    fig = P.plot_sample_size_sensitivity(by_method, output_dir=str(output_dir))
    plt.close(fig)
    print(f'  sample sensitivity → {output_dir}/sample_size_sensitivity.pdf')


def report_rmse(rmse_dir: Path, output_dir: Path):
    """RMSE vs n_samples plot."""
    results = load_results(str(rmse_dir))
    if not results:
        print(f'  [rmse] no results found in {rmse_dir}')
        return

    aggregated = {}
    for r in results:
        key = (r.get('n_samples'), r.get('estimator'))
        if None in key:
            continue
        aggregated[key] = {
            'mean_rmse_euclidean':   r.get('mean_rmse_euclidean', 0.0),
            'std_rmse_euclidean':    r.get('std_rmse_euclidean',  0.0),
            'mean_rmse_wasserstein': r.get('mean_rmse_wasserstein', 0.0),
            'std_rmse_wasserstein':  r.get('std_rmse_wasserstein',  0.0),
        }

    fig = P.plot_rmse_vs_samples(aggregated, output_dir=str(output_dir))
    plt.close(fig)
    print(f'  RMSE plot → {output_dir}/rmse_vs_samples.pdf')


def report_interpolation(interp_dir: Path, metadata: dict, output_dir: Path):
    """Interpolation sequence plots."""
    results = load_results(str(interp_dir))
    if not results:
        print(f'  [interp] no results found in {interp_dir}')
        return

    idx_to_label = metadata.get('idx_to_label', {})
    interp_out = output_dir / 'interpolation'

    for r in results:
        method = r.get('method', 'unknown')
        la, lb = r.get('label_a'), r.get('label_b')
        t_values = r.get('t_values')
        interp_key = f'interpolants_{method}'
        if interp_key not in r or t_values is None:
            continue
        name_a = idx_to_label.get(la, str(la))
        name_b = idx_to_label.get(lb, str(lb))
        out_path = str(interp_out / f'interp_{method}_{la}_vs_{lb}.pdf')
        fig = P.plot_interpolation_sequence(
            r[interp_key], t_values, name_a, name_b, method, out_path
        )
        plt.close(fig)

    print(f'  interpolation figures → {interp_out}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate all report figures from a study campaign'
    )
    parser.add_argument('--results-dir',    required=True,
                        help='Root results directory')
    parser.add_argument('--data-dir',       required=True,
                        help='Format A dataset directory (for class names + training data)')
    parser.add_argument('--output-dir',     required=True,
                        help='Where to write figures')
    parser.add_argument('--kfold-study',    default='',
                        help='Path or glob for kfold study dir')
    parser.add_argument('--gamma-study',    default='',
                        help='Path or glob for gamma sweep study dir')
    parser.add_argument('--samples-study',  default='',
                        help='Path or glob for sample-size study dir')
    parser.add_argument('--rmse-study',     default='',
                        help='Path or glob for RMSE study dir')
    parser.add_argument('--interp-study',   default='',
                        help='Path or glob for interpolation study dir')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, metadata = load_classification(args.data_dir)

    studies = {
        'kfold':   ('study_classification_kfold',    args.kfold_study),
        'gamma':   ('study_classification_gamma',    args.gamma_study),
        'samples': ('study_classification_samples',  args.samples_study),
        'rmse':    ('study_barycenter_rmse',         args.rmse_study),
        'interp':  ('study_interpolation',           args.interp_study),
    }

    resolved = {}
    for key, (prefix, arg) in studies.items():
        p = _resolve_study(arg, results_dir, prefix)
        resolved[key] = p
        if p:
            print(f'[{key}] → {p}')
        else:
            print(f'[{key}] not found (skipping)')

    if resolved['kfold']:
        print('\n— Classification results —')
        report_kfold(resolved['kfold'], Path(args.data_dir), output_dir)

    if resolved['gamma']:
        print('\n— Gamma sensitivity —')
        report_gamma(resolved['gamma'], output_dir)

    if resolved['samples']:
        print('\n— Sample size sensitivity —')
        report_samples(resolved['samples'], output_dir)

    if resolved['rmse']:
        print('\n— RMSE analysis —')
        report_rmse(resolved['rmse'], output_dir)

    if resolved['interp']:
        print('\n— Interpolation —')
        report_interpolation(resolved['interp'], metadata, output_dir)

    print(f'\nReport complete → {output_dir}')


if __name__ == '__main__':
    main()
