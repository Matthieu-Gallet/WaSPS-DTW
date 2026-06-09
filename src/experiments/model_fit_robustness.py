#!/usr/bin/env python3
"""
Model-fit robustness analysis for river discharge classification.

Motivation (reviewer concern)
------------------------------
The KS analysis (``ks_results.csv``) shows that exponential model fit quality
depends strongly on the number of samples per time group (n = D×W²):
  ws=1, tw=28 → n=28  → ~75% KS pass rate  (good fit)
  ws=2, tw=28 → n=112 → similar or better
  ws=5, tw=4  → n=100 → <60% pass rate     (poor fit)

The default balanced dataset uses ws=2, tw=28 (n=112 pixels/group).
When we *subsample* the pixels used for parameter estimation, we simulate
what smaller windows (fewer samples/group) would look like — enabling a direct
robustness curve without rebuilding the dataset.

Experiment design
-----------------
For each effective pixel count n ∈ {5, 10, 20, 40, 80, 112}:
  1. Re-estimate exponential parameters using a random subsample of n pixels
     per time group (instead of all available pixels).
  2. Compute the mean KS pass rate across all (sample, time group) pairs
     to quantify the resulting model fit quality at that n.
  3. Run stratified k-fold classification with the 3 core methods.
  4. Record (n, mean_ks_pass, F1) for each method.

Plot: F1 vs n (x-axis) and mean KS pass rate on a secondary x-axis.
This shows how each method degrades as the statistical model becomes less
reliable — and whether WaSPS-DTW benefits more from better model fit.

Usage
-----
  python src/experiments/model_fit_robustness.py
  python src/experiments/model_fit_robustness.py --n-pixels 5,20,112 --n-splits 3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from dataloader import load_classification_dataset, preprocess_samples
from experiments.classification_evaluation import run_kfold_classification


# =============================================================================
# Parameter estimation with pixel subsampling
# =============================================================================

def estimate_params_subsampled(X: np.ndarray, n_pixels: int,
                                max_time_steps: int = None,
                                seed: int = 0) -> list:
    """
    Estimate exponential λ parameters using at most ``n_pixels`` per time group.

    Unlike ``estimate_parameters_for_samples``, this function randomly selects
    ``n_pixels`` finite positive values per group (without replacement when
    sufficient pixels exist) before fitting.  This simulates the effect of a
    smaller spatial window (lower D×W²) on estimation quality.

    Parameters
    ----------
    X : array, shape (N, T, D, W, W)
    n_pixels : int — effective pixel count per group.
    max_time_steps : int, optional — cap on T.
    seed : int — RNG seed for pixel subsampling.

    Returns
    -------
    X_params : list of (T, 1) float64 arrays, one per sample.
    ks_pass_rates : array (N,) — per-sample fraction of time groups where
        the KS test against Exp at α=0.05 passes (model fit quality).
    """
    from estimator import LogCumulant

    rng = np.random.default_rng(seed)
    estimator = LogCumulant(distribution='exponential')

    N, T_full, D, W1, W2 = X.shape
    T = min(T_full, max_time_steps) if max_time_steps else T_full

    X_params = []
    ks_pass_rates = np.zeros(N)

    for i in range(N):
        params = np.zeros((T, 1), dtype=np.float64)
        ks_passes = 0
        ks_total  = 0

        for t in range(T):
            values = X[i, t].flatten()
            valid  = values[np.isfinite(values) & (values > 0)]

            if len(valid) >= 5:
                # Subsample to at most n_pixels
                if len(valid) > n_pixels:
                    sub = rng.choice(valid, size=n_pixels, replace=False)
                else:
                    sub = valid
                estimator.fit(sub)
                params[t, 0] = estimator.get_params()

                # KS test against exponential (using the estimated scale)
                scale = 1.0 / max(params[t, 0], 1e-12)
                _, p  = stats.kstest(sub, 'expon', args=(0, scale))
                ks_passes += int(p > 0.05)
                ks_total  += 1
            else:
                params[t, 0] = np.nan

        # Fill NaN with column mean
        valid_mask = np.isfinite(params[:, 0])
        if valid_mask.sum() > 0 and (~valid_mask).sum() > 0:
            params[~valid_mask, 0] = params[valid_mask, 0].mean()
        elif (~valid_mask).sum() > 0:
            params[:, 0] = 1.0

        X_params.append(params)
        ks_pass_rates[i] = ks_passes / max(ks_total, 1)

    return X_params, ks_pass_rates


# =============================================================================
# Robustness sweep
# =============================================================================

def run_robustness_sweep(X, X_raw, Y, idx_to_regime,
                          n_pixels_values, n_splits, gamma,
                          sgd_epochs, sgd_lr, max_time_steps,
                          output_dir, verbose):
    """Run k-fold classification at each pixel-count level."""
    rows = []

    for n_pix in sorted(n_pixels_values):
        print(f"\n{'─'*60}")
        print(f"n_pixels = {n_pix}  (effective samples/group for estimation)")
        print(f"{'─'*60}")

        X_params, ks_rates = estimate_params_subsampled(
            X, n_pixels=n_pix, max_time_steps=max_time_steps, seed=42)

        mean_ks = float(np.mean(ks_rates))
        print(f"  Mean KS pass rate: {mean_ks:.3f}")

        # k-fold classification
        agg = run_kfold_classification(
            X_raw, X_params, Y, idx_to_regime,
            n_splits=n_splits, gamma=gamma,
            sgd_epochs=sgd_epochs, sgd_lr=sgd_lr,
            output_dir=None, verbose=verbose,
        )

        for method_key in ['euclidean_raw', 'euclidean_params', 'wasserstein_params']:
            r = agg[method_key]
            rows.append({
                'n_pixels':            n_pix,
                'mean_ks_pass':        mean_ks,
                'method':              method_key,
                'f1_weighted_mean':    r['f1_weighted_mean'],
                'f1_weighted_std':     r['f1_weighted_std'],
                'f1_macro_mean':       r['f1_macro_mean'],
                'f1_macro_std':        r['f1_macro_std'],
            })

    df = pd.DataFrame(rows)
    csv_path = Path(output_dir) / "robustness_scores.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")
    return df


# =============================================================================
# Plot
# =============================================================================

def plot_model_fit_robustness(df: pd.DataFrame, output_dir: str):
    """
    Plot F1 vs number of estimation pixels, with mean KS pass rate annotated.

    Two subplots: F1 weighted (left) and F1 macro (right).
    Reference vertical line at n=28 (ws=1, tw=28 — best-fit config in KS study).
    """
    import matplotlib.pyplot as plt

    method_styles = {
        'euclidean_raw':      {'color': '#1f77b4', 'ls': '--',  'label': 'SDTW Euclidean (Raw)'},
        'euclidean_params':   {'color': '#ff7f0e', 'ls': '-.',  'label': 'SDTW Euclidean (Params)'},
        'wasserstein_params': {'color': '#2ca02c', 'ls': '-',   'label': 'WaSPS-DTW (Wasserstein)'},
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    fig.suptitle(
        'Robustness to model fit quality — River discharge (exponential model)\n'
        'x-axis: effective pixel count per time group used for λ estimation',
        fontsize=10
    )

    for method, style in method_styles.items():
        sub = df[df['method'] == method].sort_values('n_pixels')
        if sub.empty:
            continue
        for ax, (metric, metric_std) in zip(
            axes,
            [('f1_weighted_mean', 'f1_weighted_std'),
             ('f1_macro_mean',    'f1_macro_std')]
        ):
            ax.plot(sub['n_pixels'], sub[metric],
                    color=style['color'], ls=style['ls'],
                    marker='o', ms=6, label=style['label'])
            ax.fill_between(
                sub['n_pixels'],
                sub[metric] - sub[metric_std],
                sub[metric] + sub[metric_std],
                color=style['color'], alpha=0.15)

            # Annotate KS pass rate
            for _, row in sub.iterrows():
                ax.annotate(
                    f"{row['mean_ks_pass']:.0%}",
                    xy=(row['n_pixels'], row[metric] + 0.005),
                    ha='center', fontsize=7, color='grey'
                )

    # Reference lines
    for ax, title in zip(axes, ['F1 (weighted)', 'F1 (macro)']):
        ax.axvline(28, color='grey', ls=':', lw=1, alpha=0.7,
                   label='ws=1, tw=28 (~75% KS pass)')
        ax.set_xlabel('Pixels per time group used for estimation (n)')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

    plt.tight_layout()
    out = Path(output_dir)
    fig.savefig(out / "robustness_f1.png", dpi=150, bbox_inches='tight')
    fig.savefig(out / "robustness_f1.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"Plots saved: {out / 'robustness_f1.png'}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Model-fit robustness: vary estimation pixel count, measure F1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--data-dir', type=str,
        default=str(SRC / "results" / "classification_dataset"),
    )
    parser.add_argument('--dataset-mode', choices=['basic', 'balanced'], default='balanced')
    parser.add_argument(
        '--output-dir', type=str,
        default=str(ROOT / "results" / "regime_classification" / "model_fit_robustness"),
    )
    parser.add_argument(
        '--n-pixels', type=str, default='5,10,20,40,80,112',
        help="Comma-separated pixel counts per time group for estimation"
    )
    parser.add_argument('--n-splits',       type=int,   default=3)
    parser.add_argument('--gamma',          type=float, default=10.0)
    parser.add_argument('--sgd-epochs',     type=int,   default=30)
    parser.add_argument('--sgd-lr',         type=float, default=0.1)
    parser.add_argument('--max-time-steps', type=int,   default=400)
    parser.add_argument('--verbose',        action='store_true', default=False)
    args = parser.parse_args()

    n_pixels_values = [int(x) for x in args.n_pixels.split(',')]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("MODEL FIT ROBUSTNESS — River discharge exponential model")
    print(f"{'='*70}")
    print(f"Dataset      : {args.dataset_mode}")
    print(f"n_pixels grid: {n_pixels_values}")
    print(f"K-fold splits: {args.n_splits}")

    print(f"\n[1/3] Loading dataset…")
    X, Y, metadata = load_classification_dataset(args.data_dir, args.dataset_mode)
    print(f"  {len(X)} samples, {len(np.unique(Y))} classes, shape {X.shape}")

    print(f"\n[2/3] Preparing raw (Euclidean) representation…")
    X_raw = preprocess_samples(X, args.max_time_steps)
    print(f"  X_raw[0].shape = {X_raw[0].shape}")

    print(f"\n[3/3] Running robustness sweep…")
    df = run_robustness_sweep(
        X, X_raw, Y, metadata['idx_to_regime'],
        n_pixels_values=n_pixels_values,
        n_splits=args.n_splits,
        gamma=args.gamma,
        sgd_epochs=args.sgd_epochs,
        sgd_lr=args.sgd_lr,
        max_time_steps=args.max_time_steps,
        output_dir=str(out_dir),
        verbose=args.verbose,
    )

    if not df.empty:
        plot_model_fit_robustness(df, str(out_dir))

        print(f"\n── Summary (F1 weighted mean) ──")
        pivot = df.pivot_table(
            index='n_pixels', columns='method', values='f1_weighted_mean')
        print(pivot.to_string())

    print(f"\nDone. Results in: {out_dir}")


if __name__ == '__main__':
    main()
