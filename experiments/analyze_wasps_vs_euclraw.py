"""Investigation: why does eucl_raw outperform wasps by ~7% on cpazmal barycenter
classification (Experiment 1bis: eucl_raw=0.677±0.062 vs wasps=0.606±0.031)?

KNN mode, k=1, cpazmal only, per-method optimal gamma (grid search over the same
7-value grid used elsewhere), >=10 seeds per test. Reuses run_optim_hyper.py's
grid_knn unmodified — a pure gamma search per test config, same machinery as
experiments/optimize_gamma.py.

Five tests (4 requested + 1 proposed), all vs the same wasps/eucl_raw pair:
  baseline    — current pipeline as-is (window_size=9, samples_per_step=48,
                family=weibull, estimator=log_cumulant) — reference point.
  test1       — family=exponential on INTENSITY (not amplitude+weibull). SAR
                amplitude ~ Weibull(k~2), intensity=amplitude^2 ~ roughly
                Exponential/Gamma-like — a simpler 1-parameter fit.
  test2       — wasps only needs this: fix Weibull shape k=2 (Rayleigh-like,
                a common SAR amplitude assumption), estimate only scale λ via
                the new fit_weibull_fixed_k closed form (src/distributions.py).
                eucl_raw is included as a control — its raw representation is
                untouched by this change, so its F1 should match baseline.
  test3       — window_size=32 (1024 px available vs baseline's 81),
                samples_per_step=300 — far more samples per timestep for the
                log-cumulant fit (and for eucl_raw's raw representation size).
  test4       — same window/samples as test3, estimator=mle instead of
                log_cumulant (full MLE, asymptotically more efficient than the
                moment-based log-cumulant estimator, but only helps if sample
                size was the bottleneck rather than estimator choice itself).
  test5 (proposed) — baseline window/samples (9, 48) but estimator=mle instead
                of log_cumulant. Isolates "estimator choice" from "sample size"
                — if this alone closes much of the gap, bigger windows (test3/4,
                which require fresh HDF5 extraction) aren't actually necessary.

Output: results/wasps_vs_euclraw/summary.csv (test, method, gamma, f1_mean,
f1_std, n_seeds) + per-test grid_knn detail CSVs (kept, for the full 7-gamma
picture, not just the best point).

Usage:
    python experiments/analyze_wasps_vs_euclraw.py --n-jobs 2
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from experiment_common import _log
from run_optim_hyper import grid_knn

_HDF5_PATH = "/home/mgallet/Documents/Codes/Python/1_DONE/CPAZMAL/DATASET/dataset_original/PAZTSX_CRYO_ML.hdf5"

_BASE_DATASET = {
    "type": "cpazmal",
    "family": "weibull",
    "hdf5_path": _HDF5_PATH,
    "max_groups_per_class": None,
    "window_size": 9,
}
_BASE_CLF = {
    "k": 1,
    "sta_epsilon": 0.05,
    "samples_per_step": 48,
    "max_train_per_class": 15,
    "max_test_per_class": 25,
    "estimator": "log_cumulant",
}

GAMMA_GRID = [1.0e-4, 1.0e-2, 1.0e-1, 1.0, 1.0e+1, 1.0e+2, 1.0e+4]
N_SEEDS = 10
METHODS = ['wasps', 'eucl_raw']

TESTS = {
    "baseline": {},
    "test1_exponential_intensity": {
        "dataset": {"family": "exponential", "scale_type": "intensity"},
    },
    "test2_wasps_fixed_k2": {
        "classification": {"estimator": "fixed_k2"},
    },
    "test3_window32_logcumulant": {
        "dataset": {"window_size": 32},
        "classification": {"samples_per_step": 300},
    },
    "test4_window32_mle": {
        "dataset": {"window_size": 32},
        "classification": {"samples_per_step": 300, "estimator": "mle"},
    },
    "test5_baseline_mle": {
        "classification": {"estimator": "mle"},
    },
}


def run_for_test(test_name: str, overrides: dict, base_out_dir: Path,
                 n_jobs: int, seed: int = 42) -> dict:
    dataset = {**_BASE_DATASET, **overrides.get("dataset", {})}
    classification = {**_BASE_CLF, **overrides.get("classification", {})}
    cfg = {
        "dataset": dataset,
        "classification": classification,
        "cross_validation": {"n_splits": 1},
        "n_seeds": N_SEEDS,
    }
    out_dir = base_out_dir / test_name
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"[analyze] ===== {test_name} ===== dataset_overrides={overrides.get('dataset', {})} "
         f"clf_overrides={overrides.get('classification', {})}")
    return grid_knn(cfg, out_dir, METHODS, k_values=[1], gamma_values=GAMMA_GRID,
                    classification_overrides={}, seed=seed, n_jobs=n_jobs)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="wasps vs eucl_raw investigation (cpazmal, KNN k=1)")
    parser.add_argument("--n-jobs", type=int, default=2,
                        help="kept modest by default to avoid contending with other "
                             "concurrently-running experiments on this machine")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="results/wasps_vs_euclraw")
    parser.add_argument("--tests", default=None,
                        help="comma-separated subset of test names to run (default: all)")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["EXPERIMENT_LOG_FILE"] = str(out_dir / "analyze.log")
    if args.debug:
        os.environ["EXPERIMENT_DEBUG"] = "1"
    if args.verbose:
        os.environ["EXPERIMENT_VERBOSE"] = "1"

    tests = TESTS if not args.tests else {k: TESTS[k] for k in args.tests.split(",")}
    _log(f"===== analyze_wasps_vs_euclraw n_jobs={args.n_jobs} seed={args.seed} "
         f"tests={list(tests.keys())} n_seeds={N_SEEDS} gamma_grid={GAMMA_GRID} =====")

    summary_rows = []
    for test_name, overrides in tests.items():
        best = run_for_test(test_name, overrides, out_dir, args.n_jobs, args.seed)
        for method in METHODS:
            p = best[method]
            # grid_knn's own aggregate CSV has f1_std/n_folds_ok for the best row —
            # re-read it rather than recomputing, avoids a second evaluation pass.
            agg_path = out_dir / test_name / f"sensitivity_grid_knn_{method}.csv"
            f1_std, n_ok = float('nan'), N_SEEDS
            with open(agg_path, newline="") as f:
                for row in csv.DictReader(f):
                    if abs(float(row["gamma"]) - p["gamma"]) < 1e-12:
                        f1_std = float(row["f1_std"])
                        n_ok = int(row["n_folds_ok"])
                        break
            summary_rows.append({
                "test": test_name, "method": method, "gamma": p["gamma"],
                "f1_mean": p["f1"], "f1_std": f1_std, "n_seeds_ok": n_ok,
            })
            _log(f"[analyze] {test_name}/{method}: gamma={p['gamma']:.4g} "
                 f"f1={p['f1']:.3f}±{f1_std:.3f} (n={n_ok}/{N_SEEDS})")

    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["test", "method", "gamma", "f1_mean", "f1_std", "n_seeds_ok"])
        w.writeheader()
        w.writerows(summary_rows)
    _log(f"[analyze] wrote summary to {summary_path}")
    _log("===== analyze_wasps_vs_euclraw done =====")


if __name__ == "__main__":
    main()
