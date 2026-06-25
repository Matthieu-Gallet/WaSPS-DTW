#!/usr/bin/env python3
"""
CPAZMaL sensitivity sub-experiments for WaSPS-DTW (Weibull).

Three sub-experiments vary one factor at a time on the CPAZMaL SAR dataset:

  ntrain  — fraction of training samples used (stratified subsample).
  W       — spatial window side length (re-extracts from HDF5 at each W).
  gamma   — Soft-DTW regularisation parameter γ.

All three record F1 (weighted + macro) for:
  euclidean_raw / euclidean_params / wasserstein_weibull

Usage
-----
  python src/experiments/cpazmal_sensitivity.py --sub-exp ntrain
  python src/experiments/cpazmal_sensitivity.py --sub-exp gamma
  python src/experiments/cpazmal_sensitivity.py --sub-exp W --W-values 4,8,12
  python src/experiments/cpazmal_sensitivity.py --sub-exp all

Load cached experiment_data.pkl to skip re-extraction for ntrain/gamma:
  python src/experiments/cpazmal_sensitivity.py --sub-exp gamma \\
      --data-pkl results/cpazmal_classification/compare/experiment_data.pkl
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parents[2]
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from dataloader import MLDatasetLoader, extract_time_series, estimate_weibull_params
from experiments.cpazmal_classification import (
    run_compare,
    _estimate_weibull_list,
    _clean_raw,
)
from plot.classification_plots import plot_cpazmal_sensitivity

_DEFAULT_HDF5 = str(
    Path("/home/mgallet/Documents/Codes/Python/1_DONE"
         "/CPAZMAL/DATASET/dataset_original/PAZTSX_CRYO_ML.hdf5")
)
_DEFAULT_PKL = str(
    ROOT / "results" / "cpazmal_classification" / "compare" / "experiment_data.pkl"
)


# =============================================================================
# Data loading helpers
# =============================================================================

def _load_pkl(pkl_path):
    import pickle
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    return (d['X_raw_train_clean'], d['X_params_train'],
            d['X_raw_test_clean'],  d['X_params_test'],
            d['y_train'], d['y_test'], d['idx_to_class'])


def _extract_dataset(hdf5_path, window_size, args):
    """Extract and split dataset for a given window size."""
    loader = MLDatasetLoader(str(hdf5_path))
    dataset = extract_time_series(
        loader=loader,
        window_size=window_size,
        max_mask_value=args.max_mask_value,
        max_mask_percentage=args.max_mask_pct,
        min_valid_percentage=args.min_valid_pct,
        orbit=args.orbit,
        polarization=args.polarization,
        train_start=args.train_start,
        train_end=args.train_end,
        predict_start=args.predict_start,
        predict_end=args.predict_end,
        scale_type=args.scale_type,
        skip_optim_offset=False,
        verbose=False,
    )
    X_raw        = dataset['X_train']
    y            = dataset['y']
    idx_to_class = dataset['class_names']
    X_params     = _estimate_weibull_list(X_raw)

    idx_all = np.arange(len(X_raw))
    tr, te  = train_test_split(idx_all, test_size=args.test_size,
                                stratify=y, random_state=args.seed)
    return (
        _clean_raw([X_raw[i] for i in tr]),
        [X_params[i] for i in tr],
        _clean_raw([X_raw[i] for i in te]),
        [X_params[i] for i in te],
        y[tr], y[te], idx_to_class,
        X_raw, X_params, y,
    )


def _ks_pass_rate(X_raw, X_params):
    """Mean Weibull KS pass rate across all (sample, timestep) pairs."""
    from scipy import stats as sp_stats
    passes, total = 0, 0
    for raw, params in zip(X_raw, X_params):
        for t in range(raw.shape[0]):
            pix = raw[t]
            valid = pix[np.isfinite(pix) & (pix > 0)]
            if len(valid) < 5:
                continue
            k, lam = float(params[t, 0]), float(params[t, 1])
            if k <= 0 or lam <= 0:
                continue
            _, p = sp_stats.kstest(valid, 'weibull_min', args=(k, 0.0, lam))
            passes += int(p > 0.05)
            total  += 1
    return passes / max(total, 1)


def _compare_to_rows(results, extra_cols):
    rows = []
    for method, r in results.items():
        row = {'method': method,
               'f1_weighted_mean': r['f1_weighted'],
               'f1_macro_mean':    r['f1_macro']}
        row.update(extra_cols)
        rows.append(row)
    return rows


# =============================================================================
# Sub-experiment 1: n_train sweep
# =============================================================================

def run_ntrain_sweep(X_raw_tr, X_params_tr, X_raw_te, X_params_te,
                     y_tr, y_te, idx_to_class,
                     fracs, gamma, sgd_epochs, sgd_lr, n_seeds=3):
    """
    Vary fraction of training data (stratified subsample) at each of n_seeds.
    Returns DataFrame with columns: frac, n_train, seed, method, f1_weighted_mean, f1_macro_mean.
    """
    rows = []
    for frac in fracs:
        for seed in range(n_seeds):
            if frac >= 1.0:
                idx_sub = np.arange(len(y_tr))
            else:
                try:
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - frac,
                                                 random_state=seed)
                    idx_sub, _ = next(sss.split(np.zeros(len(y_tr)), y_tr))
                except ValueError:
                    # Fallback for very small fractions with tiny classes
                    rng = np.random.default_rng(seed)
                    n = max(len(np.unique(y_tr)), int(len(y_tr) * frac))
                    idx_sub = rng.choice(len(y_tr), size=n, replace=False)

            X_raw_sub    = [X_raw_tr[i]    for i in idx_sub]
            X_params_sub = [X_params_tr[i] for i in idx_sub]
            y_sub        = y_tr[idx_sub]
            n_train      = len(y_sub)

            print(f"\n── ntrain: frac={frac:.2f}  n={n_train}  seed={seed} ──")
            results = run_compare(X_raw_sub, X_raw_te,
                                  X_params_sub, X_params_te,
                                  y_sub, y_te,
                                  gamma=gamma, sgd_epochs=sgd_epochs, sgd_lr=sgd_lr,
                                  idx_to_class=idx_to_class, verbose=False)

            rows += _compare_to_rows(results, {'frac': frac, 'n_train': n_train, 'seed': seed})

    df = pd.DataFrame(rows)
    # Average over seeds
    agg = (df.groupby(['frac', 'n_train', 'method'])
             .agg(f1_weighted_mean=('f1_weighted_mean', 'mean'),
                  f1_weighted_std= ('f1_weighted_mean', 'std'),
                  f1_macro_mean=   ('f1_macro_mean',    'mean'),
                  f1_macro_std=    ('f1_macro_mean',    'std'))
             .reset_index())
    return agg


# =============================================================================
# Sub-experiment 2: W (window size) sweep
# =============================================================================

def run_W_sweep(hdf5_path, W_values, args, gamma, sgd_epochs, sgd_lr):
    """
    Re-extract from HDF5 at each window side length W.
    Records mean Weibull KS pass rate at each W as a model-fit quality indicator.
    Returns DataFrame with columns: W, W_sq, ks_pass_rate, method, f1_weighted_mean, f1_macro_mean.
    """
    rows = []
    for W in W_values:
        print(f"\n{'─'*60}")
        print(f"W = {W}  (W² = {W*W} pixels/timestep)")
        print(f"{'─'*60}")

        (X_raw_tr, X_params_tr, X_raw_te, X_params_te,
         y_tr, y_te, idx_to_class,
         X_raw_full, X_params_full, y_full) = _extract_dataset(hdf5_path, W, args)

        print(f"  Extracted {len(y_tr)+len(y_te)} samples, "
              f"{len(np.unique(y_tr))} classes")

        # KS pass rate on a subsample (max 200 per class) for speed
        ks_rate = _ks_pass_rate(X_raw_full[:200], X_params_full[:200])
        print(f"  Mean KS pass rate (subsample): {ks_rate:.3f}")

        results = run_compare(X_raw_tr, X_raw_te,
                              X_params_tr, X_params_te,
                              y_tr, y_te,
                              gamma=gamma, sgd_epochs=sgd_epochs, sgd_lr=sgd_lr,
                              idx_to_class=idx_to_class, verbose=False)

        rows += _compare_to_rows(results, {'W': W, 'W_sq': W * W,
                                            'ks_pass_rate': ks_rate})

    df = pd.DataFrame(rows)
    df['f1_weighted_std'] = np.nan
    df['f1_macro_std']    = np.nan
    df.rename(columns={'f1_weighted_mean': 'f1_weighted_mean',
                        'f1_macro_mean':    'f1_macro_mean'}, inplace=True)
    return df


# =============================================================================
# Sub-experiment 3: gamma sweep
# =============================================================================

def run_gamma_sweep(X_raw_tr, X_params_tr, X_raw_te, X_params_te,
                    y_tr, y_te, idx_to_class,
                    gammas, sgd_epochs, sgd_lr):
    """
    Vary Soft-DTW γ across all 3 methods with a fixed train/test split.
    Returns DataFrame with columns: gamma, method, f1_weighted_mean, f1_macro_mean.
    """
    rows = []
    for gamma in gammas:
        print(f"\n── gamma = {gamma} ──")
        results = run_compare(X_raw_tr, X_raw_te,
                              X_params_tr, X_params_te,
                              y_tr, y_te,
                              gamma=gamma, sgd_epochs=sgd_epochs, sgd_lr=sgd_lr,
                              idx_to_class=idx_to_class, verbose=False)
        rows += _compare_to_rows(results, {'gamma': gamma})

    df = pd.DataFrame(rows)
    df['f1_weighted_std'] = np.nan
    df['f1_macro_std']    = np.nan
    return df


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CPAZMaL sensitivity sub-experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--sub-exp', choices=['ntrain', 'W', 'gamma', 'all'],
                        default='ntrain')
    parser.add_argument('--hdf5',       type=str, default=_DEFAULT_HDF5)
    parser.add_argument('--data-pkl',   type=str, default=_DEFAULT_PKL,
                        help="Cached experiment_data.pkl from compare mode "
                             "(skips HDF5 re-extraction for ntrain/gamma sweeps).")
    parser.add_argument('--output-dir', type=str,
                        default=str(ROOT / "results" / "cpazmal_sensitivity"))

    # Extraction parameters (used by W sweep and fresh extraction)
    parser.add_argument('--orbit',         type=str,   default='DSC')
    parser.add_argument('--polarization',  type=str,   default='HH')
    parser.add_argument('--train-start',   type=str,   default='20200101')
    parser.add_argument('--train-end',     type=str,   default='20201031')
    parser.add_argument('--predict-start', type=str,   default='20201101')
    parser.add_argument('--predict-end',   type=str,   default='20201231')
    parser.add_argument('--scale-type',    type=str,   default='amplitude')
    parser.add_argument('--max-mask-value',type=int,   default=1)
    parser.add_argument('--max-mask-pct',  type=float, default=10.0)
    parser.add_argument('--min-valid-pct', type=float, default=50.0)
    parser.add_argument('--test-size',     type=float, default=0.2)
    parser.add_argument('--seed',          type=int,   default=42)

    # Model parameters
    parser.add_argument('--gamma',       type=float, default=10.0)
    parser.add_argument('--sgd-epochs',  type=int,   default=30)
    parser.add_argument('--sgd-lr',      type=float, default=0.05)

    # Sweep grids
    parser.add_argument('--fracs',   type=str, default='0.05,0.1,0.2,0.4,0.6,0.8,1.0',
                        help="Training fractions for ntrain sweep")
    parser.add_argument('--W-values', type=str, default='4,6,8,12',
                        help="Window sizes for W sweep")
    parser.add_argument('--gammas',   type=str,
                        default='0.01,0.1,1.0,5.0,10.0,50.0,100.0',
                        help="γ values for gamma sweep")
    parser.add_argument('--n-seeds',  type=int, default=3,
                        help="Seeds for ntrain sweep averaging")
    args = parser.parse_args()

    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fracs    = [float(x) for x in args.fracs.split(',')]
    W_values = [int(x)   for x in args.W_values.split(',')]
    gammas   = [float(x) for x in args.gammas.split(',')]

    # ── Load or extract base dataset ─────────────────────────────────────────
    pkl_path = Path(args.data_pkl)
    if pkl_path.exists() and args.sub_exp in ('ntrain', 'gamma', 'all'):
        print(f"Loading cached data from {pkl_path}…")
        X_raw_tr, X_params_tr, X_raw_te, X_params_te, \
            y_tr, y_te, idx_to_class = _load_pkl(pkl_path)
        print(f"  Train: {len(y_tr)}  Test: {len(y_te)}")
    elif args.sub_exp in ('ntrain', 'gamma', 'all'):
        print("No cached data found — extracting from HDF5…")
        (X_raw_tr, X_params_tr, X_raw_te, X_params_te,
         y_tr, y_te, idx_to_class, _, _, _) = _extract_dataset(
             args.hdf5, 12, args)
        print(f"  Train: {len(y_tr)}  Test: {len(y_te)}")
    else:
        X_raw_tr = X_params_tr = X_raw_te = X_params_te = None
        y_tr = y_te = idx_to_class = None

    # ── Sub-experiments ───────────────────────────────────────────────────────
    do_ntrain = args.sub_exp in ('ntrain', 'all')
    do_W      = args.sub_exp in ('W', 'all')
    do_gamma  = args.sub_exp in ('gamma', 'all')

    if do_ntrain:
        print(f"\n{'='*70}")
        print("SUB-EXP 1: n_train sweep")
        print(f"{'='*70}")
        df_ntrain = run_ntrain_sweep(
            X_raw_tr, X_params_tr, X_raw_te, X_params_te,
            y_tr, y_te, idx_to_class,
            fracs=fracs, gamma=args.gamma,
            sgd_epochs=args.sgd_epochs, sgd_lr=args.sgd_lr,
            n_seeds=args.n_seeds,
        )
        csv = out_dir / 'sensitivity_ntrain.csv'
        df_ntrain.to_csv(csv, index=False)
        print(f"\nSaved: {csv}")
        plot_cpazmal_sensitivity(df_ntrain, x_col='frac',
                                  xlabel='Training fraction',
                                  output_dir=str(out_dir),
                                  filename='sensitivity_ntrain')

    if do_W:
        print(f"\n{'='*70}")
        print("SUB-EXP 2: Window size (W) sweep")
        print(f"{'='*70}")
        df_W = run_W_sweep(
            args.hdf5, W_values, args,
            gamma=args.gamma, sgd_epochs=args.sgd_epochs, sgd_lr=args.sgd_lr,
        )
        csv = out_dir / 'sensitivity_W.csv'
        df_W.to_csv(csv, index=False)
        print(f"\nSaved: {csv}")
        plot_cpazmal_sensitivity(df_W, x_col='W_sq',
                                  xlabel='Window area W² (pixels/timestep)',
                                  output_dir=str(out_dir),
                                  filename='sensitivity_W')

    if do_gamma:
        print(f"\n{'='*70}")
        print("SUB-EXP 3: gamma sweep")
        print(f"{'='*70}")
        df_gamma = run_gamma_sweep(
            X_raw_tr, X_params_tr, X_raw_te, X_params_te,
            y_tr, y_te, idx_to_class,
            gammas=gammas, sgd_epochs=args.sgd_epochs, sgd_lr=args.sgd_lr,
        )
        csv = out_dir / 'sensitivity_gamma.csv'
        df_gamma.to_csv(csv, index=False)
        print(f"\nSaved: {csv}")
        plot_cpazmal_sensitivity(df_gamma, x_col='gamma',
                                  xlabel='γ (Soft-DTW regularisation)',
                                  output_dir=str(out_dir),
                                  filename='sensitivity_gamma',
                                  x_log=True)

    print(f"\nAll results in: {out_dir}")


if __name__ == '__main__':
    main()
