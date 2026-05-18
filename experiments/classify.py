#!/usr/bin/env python3
"""
Hydrological regime classification — one train/test split, three barycenter methods.

Reads Format A: --data-dir must contain X.npy (N,T,D), Y.npy (N,), metadata.npy.
Train and test index files are plain .npy integer arrays; generate them once with
the fold-generation block in scripts/study_classification_kfold.sh.

Three methods are always run and stored together:
  euclidean_raw    — Soft-DTW barycenter on raw (T,D) series
  euclidean_params — Soft-DTW barycenter on estimated (T,1) parameters
  wasserstein_sgd  — Soft-DTW/Wasserstein SGD barycenter on (T,1) parameters

Results written to --output-dir/results.zarr (one zarr group per method).
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import zarr
from sklearn.metrics import classification_report, f1_score

from src.classification import (
    classify_by_nearest_barycenter,
    compute_barycenter_euclidean_params,
    compute_barycenter_euclidean_raw,
    compute_barycenter_wasserstein_sgd,
    compute_sdtw_distance_euclidean,
    compute_sdtw_distance_wasserstein,
)
from src.dataloader import estimate_parameters, load_classification, normalize_params, preprocess_samples


METHODS = {
    'euclidean_raw': {
        'barycenter': compute_barycenter_euclidean_raw,
        'distance':   compute_sdtw_distance_euclidean,
        'input_key':  'raw',
    },
    'euclidean_params': {
        'barycenter': compute_barycenter_euclidean_params,
        'distance':   compute_sdtw_distance_euclidean,
        'input_key':  'params',
    },
    'wasserstein_sgd': {
        'barycenter': compute_barycenter_wasserstein_sgd,
        'distance':   compute_sdtw_distance_wasserstein,
        'input_key':  'params_pos',  # must stay positive
    },
}


def run_one_split(X_raw, X_params, X_params_pos, Y, train_idx, test_idx,
                  idx_to_label, gamma, max_iter, verbose):
    """Run all 3 methods on one split. Returns dict of per-method results.

    X_params_pos must have positive values (for Wasserstein distance).
    """
    inputs = {'raw': X_raw, 'params': X_params, 'params_pos': X_params_pos}
    results = {}
    for method_name, cfg in METHODS.items():
        if verbose:
            print(f'\n  [{method_name}]')
        X_series = inputs[cfg['input_key']]

        t0 = time.time()
        barycenters = {}
        for label in np.unique(Y[train_idx]):
            class_samples = [X_series[i] for i in train_idx if Y[i] == label]
            if verbose:
                print(f'    class {idx_to_label[label]}: {len(class_samples)} samples')
            barycenters[label] = cfg['barycenter'](class_samples, gamma=gamma,
                                                    max_iter=max_iter)
        t_bary = time.time() - t0

        t0 = time.time()
        test_samples = [X_series[i] for i in test_idx]
        preds = classify_by_nearest_barycenter(
            test_samples, barycenters, cfg['distance'], gamma
        )
        t_cls = time.time() - t0

        Y_test = Y[test_idx]
        labels = sorted(np.unique(np.concatenate([Y_test, preds])))
        f1_w = float(f1_score(Y_test, preds, average='weighted', zero_division=0))
        f1_m = float(f1_score(Y_test, preds, average='macro',    zero_division=0))
        report = classification_report(Y_test, preds, labels=labels,
                                       output_dict=True, zero_division=0)
        per_class_f1        = {str(l): report[str(l)]['f1-score']  for l in labels}
        per_class_precision = {str(l): report[str(l)]['precision'] for l in labels}
        per_class_recall    = {str(l): report[str(l)]['recall']    for l in labels}
        if verbose:
            print(f'    F1 weighted={f1_w:.4f}  macro={f1_m:.4f}')

        results[method_name] = {
            'predictions':        preds,
            'Y_test':             Y_test,
            'f1_weighted':        f1_w,
            'f1_macro':           f1_m,
            'per_class_f1':       per_class_f1,
            'per_class_precision': per_class_precision,
            'per_class_recall':   per_class_recall,
            'barycenter_time':    t_bary,
            'classify_time':      t_cls,
            'barycenters':        barycenters,
        }
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Classify with 3 Soft-DTW barycenter methods — one split'
    )
    parser.add_argument('--data-dir',       required=True,
                        help='Format A directory (X.npy, Y.npy, metadata.npy)')
    parser.add_argument('--train-indices',  required=True,
                        help='.npy file of integer training indices')
    parser.add_argument('--test-indices',   required=True,
                        help='.npy file of integer test indices')
    parser.add_argument('--output-dir',     required=True)
    parser.add_argument('--gamma',          type=float, default=1.0)
    parser.add_argument('--max-iter',       type=int,   default=30)
    parser.add_argument('--max-time-steps',   type=int,   default=None)
    parser.add_argument('--random-seed',      type=int,   default=42)
    parser.add_argument('--normalize',        action='store_true',
                        help='Z-score normalize each raw sample (removes amplitude)')
    parser.add_argument('--normalize-params', default='none',
                        choices=['none', 'zscore', 'mean_ratio', 'log_zscore_linear',
                                 'wasserstein_euclidean'],
                        help='Normalize estimated lambda series (default: none)')
    parser.add_argument('--verbose',          action='store_true')
    args = parser.parse_args()

    np.random.seed(args.random_seed)

    X_raw_arr, Y, metadata = load_classification(args.data_dir)
    idx_to_label = {int(k): v for k, v in metadata['idx_to_label'].items()}
    train_idx = np.load(args.train_indices)
    test_idx  = np.load(args.test_indices)

    print(f'Dataset: {len(Y)} samples  train={len(train_idx)}  test={len(test_idx)}')
    print(f'gamma={args.gamma}  max_iter={args.max_iter}')

    # Raw series for euclidean_raw: normalize after clipping NaNs
    X_raw = preprocess_samples(X_raw_arr, max_time_steps=args.max_time_steps,
                               normalize=args.normalize)
    # Parameter estimation always on positive raw data (exponential dist requires x > 0)
    X_raw_for_params = preprocess_samples(X_raw_arr, max_time_steps=args.max_time_steps,
                                          normalize=False)
    X_params_raw = estimate_parameters(X_raw_for_params)
    # euclidean_params: normalize lambda series if requested (may yield negatives — ok for Euclidean)
    if args.normalize_params != 'none':
        X_params = normalize_params(X_params_raw, mode=args.normalize_params)
    else:
        X_params = X_params_raw
    # wasserstein_sgd: must stay positive — use log_zscore for scale-invariant, bounded values
    if args.normalize_params != 'none':
        X_params_pos = normalize_params(X_params_raw, mode='log_zscore')
    else:
        X_params_pos = X_params_raw

    results = run_one_split(
        X_raw, X_params, X_params_pos, Y, train_idx, test_idx,
        idx_to_label, args.gamma, args.max_iter, args.verbose
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = zarr.open(str(out_dir / 'results.zarr'), mode='w')

    base_attrs = {
        'data_dir':         args.data_dir,
        'gamma':            args.gamma,
        'max_iter':         args.max_iter,
        'max_time_steps':   args.max_time_steps,
        'random_seed':      args.random_seed,
        'normalize':        args.normalize,
        'normalize_params': args.normalize_params,
        'n_train':          int(len(train_idx)),
        'n_test':           int(len(test_idx)),
        'timestamp':        datetime.now().isoformat(),
    }

    for method_name, r in results.items():
        grp = store.require_group(method_name)
        grp.create_dataset('predictions', data=r['predictions'])
        grp.create_dataset('Y_test',      data=r['Y_test'])
        for lbl, bary in r['barycenters'].items():
            grp.create_dataset(f'barycenter_{lbl}', data=bary)
        grp.attrs.update({
            **base_attrs,
            'method':               method_name,
            'f1_weighted':          r['f1_weighted'],
            'f1_macro':             r['f1_macro'],
            'per_class_f1':         r['per_class_f1'],
            'per_class_precision':  r['per_class_precision'],
            'per_class_recall':     r['per_class_recall'],
            'barycenter_time':      r['barycenter_time'],
            'classify_time':        r['classify_time'],
        })

    store.attrs.update(base_attrs)
    print(f'\nResults → {out_dir}')
    for m, r in results.items():
        print(f'  {m:20s}  F1={r["f1_weighted"]:.4f}')


if __name__ == '__main__':
    main()
