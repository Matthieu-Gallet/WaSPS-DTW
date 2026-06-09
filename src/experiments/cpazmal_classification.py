#!/usr/bin/env python3
"""
CPAZMaL SAR time-series classification using WaSPS-DTW (Weibull).

Two classification modes:
  kmedoid  — per-class Weibull barycenter (SGD), then nearest-barycenter
             assignment using the Soft-DTW Weibull divergence.
  shapelet — Learning Shapelets with a Soft-DTW Wasserstein (Weibull) distance.

The SAR amplitude data follow a Weibull distribution, so Weibull parameters
(k, λ_scale) are estimated at each time step before computing distances.

Usage
-----
  # K-medoid mode, full dataset
  python src/experiments/cpazmal_classification.py --mode kmedoid

  # Shapelet mode, 3 epochs (quick smoke-test)
  python src/experiments/cpazmal_classification.py --mode shapelet --epochs 3

  # Subset for quick testing
  python src/experiments/cpazmal_classification.py --mode kmedoid --max-groups 6
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from dataloader import (
    MLDatasetLoader,
    extract_time_series,
    estimate_weibull_params,
)
from sdtw.classification_methods import (
    compute_barycenter_wasserstein_sgd,
    compute_sdtw_distance_weibull,
)

# default local HDF5 path (falls back to HF download via download_cpazmal)
_DEFAULT_HDF5 = str(
    Path("/home/mgallet/Documents/Codes/Python/1_DONE"
         "/CPAZMAL/DATASET/dataset_original/PAZTSX_CRYO_ML.hdf5")
)


# =============================================================================
# Parameter estimation helpers
# =============================================================================

def _estimate_weibull_list(X_series_list):
    """Estimate Weibull params for a list of (T, W²) arrays → list of (T, 2)."""
    return [estimate_weibull_params(s) for s in X_series_list]


# =============================================================================
# K-medoid classification
# =============================================================================

def run_kmedoid(X_params_train, X_params_test, y_train, y_test,
                gamma, sgd_epochs, sgd_lr, idx_to_class, verbose):
    """Classify by nearest Weibull-barycenter using the Soft-DTW divergence."""
    unique_classes = np.unique(y_train)

    print("\n── K-medoid / nearest Weibull barycenter ──")
    t0 = time.time()

    barycenters = {}
    for cls in unique_classes:
        cls_params = [X_params_train[i] for i in range(len(X_params_train))
                      if y_train[i] == cls]
        if verbose:
            print(f"  Barycenter class {idx_to_class[cls]} "
                  f"({len(cls_params)} samples)…")
        barycenters[cls] = compute_barycenter_wasserstein_sgd(
            cls_params, gamma=gamma, learning_rate=sgd_lr,
            num_epochs=sgd_epochs, batch_size=4,
            distribution='weibull', verbose=False,
        )
    bary_time = time.time() - t0
    print(f"  Barycenter time: {bary_time:.1f}s")

    # Classify
    t0 = time.time()
    y_pred = []
    for p in X_params_test:
        distances = {cls: compute_sdtw_distance_weibull(
            p, barycenters[cls], gamma=gamma, divergence=True)
            for cls in unique_classes}
        y_pred.append(min(distances, key=distances.get))
    classify_time = time.time() - t0

    y_pred = np.array(y_pred)
    f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_m = f1_score(y_test, y_pred, average='macro',    zero_division=0)
    print(f"  Classification time: {classify_time:.1f}s")
    print(f"  F1 weighted: {f1_w:.4f}  |  F1 macro: {f1_m:.4f}")
    if verbose:
        present = sorted(np.unique(np.concatenate([y_test, y_pred])))
        names = [idx_to_class[i] for i in present]
        print(classification_report(y_test, y_pred,
                                    labels=present, target_names=names,
                                    zero_division=0))
    return {
        'method': 'kmedoid_weibull',
        'predictions': y_pred,
        'barycenters': barycenters,
        'f1_weighted': f1_w,
        'f1_macro': f1_m,
        'barycenter_time': bary_time,
        'classify_time': classify_time,
    }


# =============================================================================
# Learning Shapelets classification
# =============================================================================

def run_shapelets(X_params_train, X_params_test, y_train, y_test,
                  gamma, epochs, batch_size, lr, num_shapelets_per_scale,
                  idx_to_class, verbose, seed):
    """Classify with Learning Shapelets + Soft-DTW Weibull distance."""
    from experiments.shapelets_classifier import (
        train_shapelets_classifier,
        predict_shapelets_classifier,
    )

    ts_len = X_params_train[0].shape[0]
    shapelet_len = max(3, ts_len // 4)

    print(f"\n── Learning Shapelets (Weibull, len={shapelet_len}) ──")
    clf, state = train_shapelets_classifier(
        train_samples=X_params_train,
        y_train=y_train,
        dist_measure='soft_dtw_wasserstein',
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        shapelets_size_and_len={shapelet_len: num_shapelets_per_scale},
        shapelets_gamma=gamma,
        shapelets_num_per_scale=num_shapelets_per_scale,
        seed=seed,
        verbose=int(verbose),
    )
    print(f"  Training time: {state['train_time']:.1f}s")

    t0 = time.time()
    y_pred = predict_shapelets_classifier(clf=clf, state=state,
                                          test_samples=X_params_test,
                                          batch_size=batch_size)
    classify_time = time.time() - t0

    f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_m = f1_score(y_test, y_pred, average='macro',    zero_division=0)
    print(f"  Classification time: {classify_time:.1f}s")
    print(f"  F1 weighted: {f1_w:.4f}  |  F1 macro: {f1_m:.4f}")

    return {
        'method': 'shapelets_weibull',
        'predictions': y_pred,
        'f1_weighted': f1_w,
        'f1_macro': f1_m,
        'barycenter_time': state['train_time'],
        'classify_time': classify_time,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CPAZMaL SAR classification — WaSPS-DTW (Weibull)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--hdf5', type=str, default=_DEFAULT_HDF5,
                        help="Path to PAZTSX_CRYO_ML.hdf5")
    parser.add_argument('--mode', choices=['kmedoid', 'shapelet', 'both'],
                        default='kmedoid')

    # Extraction parameters (scenario from dataset.md)
    parser.add_argument('--window-size', type=int, default=12)
    parser.add_argument('--orbit', type=str, default='DSC')
    parser.add_argument('--polarization', type=str, default='HH')
    parser.add_argument('--train-start', type=str, default='20200101')
    parser.add_argument('--train-end',   type=str, default='20201031')
    parser.add_argument('--predict-start', type=str, default='20201101')
    parser.add_argument('--predict-end',   type=str, default='20201231')
    parser.add_argument('--scale-type', type=str, default='amplitude')
    parser.add_argument('--max-mask-value', type=int, default=1)
    parser.add_argument('--max-mask-pct', type=float, default=10.0)
    parser.add_argument('--min-valid-pct', type=float, default=50.0)

    # Subset for quick testing
    parser.add_argument('--max-groups', type=int, default=None,
                        help="Limit to first N groups (smoke-test shortcut)")

    # Model parameters
    parser.add_argument('--gamma', type=float, default=10.0)
    parser.add_argument('--sgd-epochs', type=int, default=30)
    parser.add_argument('--sgd-lr', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=20,
                        help="Shapelet training epochs")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-shapelets', type=int, default=4)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str,
                        default=str(ROOT / "results" / "cpazmal_classification"))
    parser.add_argument('--verbose', action='store_true', default=True)
    args = parser.parse_args()

    np.random.seed(args.seed)
    out_dir = Path(args.output_dir) / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load HDF5 ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("CPAZMaL — Weibull WaSPS-DTW Classification")
    print(f"{'='*70}")

    hdf5 = Path(args.hdf5)
    if not hdf5.exists():
        print(f"HDF5 not found at {hdf5} — downloading from HuggingFace…")
        from dataloader import download_cpazmal
        hdf5 = Path(download_cpazmal(str(hdf5.parent)))

    loader = MLDatasetLoader(str(hdf5))

    # Optional group subsetting for quick tests
    if args.max_groups is not None:
        orig_class_index = loader.class_index
        trimmed = {}
        count = 0
        for cls, entries in orig_class_index.items():
            keep = entries[:max(1, args.max_groups // len(orig_class_index))]
            trimmed[cls] = keep
            count += len(keep)
            if count >= args.max_groups:
                break
        loader.class_index = trimmed
        print(f"  (Subset: {count} groups)")

    # ── 2. Extract time series ────────────────────────────────────────────────
    print("\n[1/4] Extracting time series…")
    dataset = extract_time_series(
        loader=loader,
        window_size=args.window_size,
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
        skip_optim_offset=True,
        verbose=args.verbose,
    )
    X_raw   = dataset['X_train']   # list of (T_train, W²) arrays
    y       = dataset['y']
    idx_to_class = dataset['class_names']

    print(f"  Loaded {len(X_raw)} samples, "
          f"{len(np.unique(y))} classes, "
          f"T_train={X_raw[0].shape[0]}, W²={X_raw[0].shape[1]}")

    # ── 3. Estimate Weibull parameters ────────────────────────────────────────
    print("\n[2/4] Estimating Weibull parameters…")
    t0 = time.time()
    X_params = _estimate_weibull_list(X_raw)
    print(f"  Done in {time.time()-t0:.1f}s  "
          f"— params shape: {X_params[0].shape}  (T, 2) = (k, λ_scale)")

    # ── 4. Train / test split ─────────────────────────────────────────────────
    idx_all = np.arange(len(X_raw))
    tr_idx, te_idx = train_test_split(
        idx_all, test_size=args.test_size,
        stratify=y, random_state=args.seed)

    X_params_train = [X_params[i] for i in tr_idx]
    X_params_test  = [X_params[i] for i in te_idx]
    y_train = y[tr_idx]
    y_test  = y[te_idx]
    print(f"  Train: {len(y_train)}  Test: {len(y_test)}")

    # ── 5. Classification ─────────────────────────────────────────────────────
    print("\n[3/4] Running classification…")
    results = {}

    if args.mode in ('kmedoid', 'both'):
        results['kmedoid'] = run_kmedoid(
            X_params_train, X_params_test, y_train, y_test,
            gamma=args.gamma, sgd_epochs=args.sgd_epochs, sgd_lr=args.sgd_lr,
            idx_to_class=idx_to_class, verbose=args.verbose,
        )

    if args.mode in ('shapelet', 'both'):
        results['shapelet'] = run_shapelets(
            X_params_train, X_params_test, y_train, y_test,
            gamma=args.gamma, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr,
            num_shapelets_per_scale=args.num_shapelets,
            idx_to_class=idx_to_class, verbose=args.verbose,
            seed=args.seed,
        )

    # ── 6. Save results ───────────────────────────────────────────────────────
    print(f"\n[4/4] Saving results to {out_dir}/")
    rows = []
    for key, res in results.items():
        rows.append({
            'method':        res['method'],
            'f1_weighted':   res['f1_weighted'],
            'f1_macro':      res['f1_macro'],
            'barycenter_time': res['barycenter_time'],
            'classify_time': res['classify_time'],
        })
    df = pd.DataFrame(rows)
    csv_path = out_dir / "classification_scores.csv"
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved: {csv_path}")


if __name__ == '__main__':
    main()
