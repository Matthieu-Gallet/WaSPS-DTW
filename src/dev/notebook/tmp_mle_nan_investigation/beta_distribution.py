"""Step 1 — MLE vs log-cumulant beta-distribution comparison (no barycenter fitting).

Tests the hypothesis that MLE's outlier-sensitivity (beta = 1/mean(x), dragged toward 0
by a single large sample) produces a fatter near-zero tail than log-cumulant
(beta = 1/exp(mean(log x)), where log compresses outliers) — the mechanism the plan
suspects behind WaSPS's intermittent NaN barycenters (gradient ~ 1/beta^3).

Loads river data at river.yaml's real hyperparameters (samples_per_step=50,
max_train_samples=20 -> 5/class), across all 5 folds x 4 seeds, estimates both ways,
and reports tail statistics. No src/ changes; read-only diagnostic.

Run: python analysis/tmp_mle_nan_investigation/beta_distribution.py
Writes: beta_distribution_stats.json, beta_histograms.png (this directory).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
_SRC = _HERE.parent.parent / "src"
sys.path.insert(0, str(_SRC))

import distributions
from data.preprocess import clean_time_series
from data.river_loader import load_river_classification

DATA_DIR = str(_HERE.parent.parent / "data" / "river")
N_SPLITS = 5
SEEDS = [42, 43, 44, 45]
SAMPLES_PER_STEP = 50          # river.yaml
MAX_TRAIN_PER_CLASS = 5        # river.yaml max_train_samples=20 / 4 classes
THRESHOLDS = [1e-3, 1e-5, 1e-8]

dist = distributions.get("exponential")


def load_fold_capped(fold: int, seed: int):
    """One fold, capped to MAX_TRAIN_PER_CLASS per class (matches river.yaml's regime)."""
    data = load_river_classification(
        DATA_DIR, mode="balanced", n_splits=N_SPLITS, fold=fold, group_aware=True,
        samples_per_step=SAMPLES_PER_STEP, seed=seed, cv_seed=42,
    )
    X_train, y_train = data["X_train"], data["y_train"]
    classes = sorted(set(y_train.tolist()))
    rng = np.random.default_rng(seed)
    A, B = [], []
    for cls in classes:
        idx = [i for i, l in enumerate(y_train) if l == cls]
        rng.shuffle(idx)
        idx = idx[:MAX_TRAIN_PER_CLASS]
        A.extend([X_train[i] for i in idx])
        B.extend([cls] * len(idx))
    return A, np.array(B)


def estimate(X: list, method: str) -> list:
    # clean_time_series(s) — default threshold=-1.0, matches data_utils.py::build_repr exactly
    # (the notebook's explicit threshold=-1.0 is the same value, just spelled out).
    return [dist.fit_time_series(clean_time_series(s), dtype=np.float64, method=method) for s in X]


def main():
    beta_by_method = {"mle": [], "log_cumulant": []}
    nan_frac_by_trial = {"mle": [], "log_cumulant": []}

    for fold in range(N_SPLITS):
        for seed in SEEDS:
            X_train, y_train = load_fold_capped(fold, seed)
            for method in ("mle", "log_cumulant"):
                params = estimate(X_train, method)
                all_vals = np.concatenate([p[:, 0] for p in params])
                nan_frac_by_trial[method].append(float(np.isnan(all_vals).mean()))
                beta_by_method[method].append(all_vals[~np.isnan(all_vals)])
            print(f"[fold={fold} seed={seed}] done", flush=True)

    stats = {}
    for method in ("mle", "log_cumulant"):
        all_beta = np.concatenate(beta_by_method[method])
        s = {
            "n_trials": len(nan_frac_by_trial[method]),
            "n_values": int(len(all_beta)),
            "mean_nan_frac_per_trial": float(np.mean(nan_frac_by_trial[method])),
            "min": float(all_beta.min()),
            "p1": float(np.percentile(all_beta, 1)),
            "p5": float(np.percentile(all_beta, 5)),
            "median": float(np.median(all_beta)),
            "max": float(all_beta.max()),
            "below_threshold": {
                f"{t:.0e}": {
                    "count": int((all_beta < t).sum()),
                    "pct": float(100 * (all_beta < t).mean()),
                }
                for t in THRESHOLDS
            },
        }
        stats[method] = s
        print(f"\n=== {method} ===")
        print(f"  n_values={s['n_values']}  mean_nan_frac_per_trial={s['mean_nan_frac_per_trial']:.4f}")
        print(f"  min={s['min']:.4g}  p1={s['p1']:.4g}  p5={s['p5']:.4g}  "
              f"median={s['median']:.4g}  max={s['max']:.4g}")
        for t in THRESHOLDS:
            b = s["below_threshold"][f"{t:.0e}"]
            print(f"  n(beta < {t:.0e}) = {b['count']}  ({b['pct']:.4f}%)")

    out_path = _HERE / "beta_distribution_stats.json"
    out_path.write_text(json.dumps(stats, indent=2))
    print(f"\n[saved] {out_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, method in zip(axes, ("mle", "log_cumulant")):
            all_beta = np.concatenate(beta_by_method[method])
            ax.hist(np.log10(all_beta), bins=60, color="steelblue", alpha=0.85)
            ax.set_title(f"{method}  (n={len(all_beta)})")
            ax.set_xlabel("log10(beta)")
        axes[0].set_ylabel("count")
        plt.suptitle("beta distribution: MLE vs log-cumulant (river, 5-fold x 4 seeds, 5/class)")
        plt.tight_layout()
        fig_path = _HERE / "beta_histograms.png"
        fig.savefig(fig_path, dpi=120)
        print(f"[saved] {fig_path}")
    except ImportError:
        print("[skip] matplotlib not available — no histogram saved")


if __name__ == "__main__":
    main()
