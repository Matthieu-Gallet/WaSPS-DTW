"""Hyperparameter grid search (calibration) for WaSPS-DTW on real data (river + CPAZMaL).

Scenarios (select with --scenario):
  grid_knn   — k×gamma grid per method, replicated over K-fold groups
  grid_bary  — lr×gamma grid per method, replicated over K-fold groups

Distinct from experiments/run_sensitivity.py's sensitivity SWEEPS (n_samples/n_train/
decimation): grid search here calibrates (k,gamma)/(lr,gamma) per method and writes
best_params.json, which run_sensitivity.py's sweeps then read as their fixed operating
point. Config-driven: configs/sensitivity_river.yaml / sensitivity_cpazmal.yaml — the
SAME files run_sensitivity.py reads; their `sensitivity.grid_knn`/`grid_bary` blocks
live there. This is a pure code split from the old combined run_sensitivity.py, not a
config split.

STA excluded (cost: O(T²) per pair, no calibrated grid point).

Parallelization: grid points run as independent joblib jobs, each evaluating all folds
sequentially inside the worker. grid_bary's per-iteration _eval_bary call nests
fit_barycenters' own per-class joblib parallelism (n_jobs=4 there, see
experiment_common._eval_bary) — with grid_bary's default 5 lr_values × 4 gamma_values
= 20 outer grid points, an unbounded outer --n-jobs (-1 → os.cpu_count() workers
regardless of task count, each importing JAX independently) combined with the inner
n_jobs=4 could reach up to 80 concurrent processes. --n-jobs now defaults to 4
(outer) — bounded 4×4=16 worst case — after a machine crash during
experiments/run_full_baseline.py's STA phase at n_jobs=5 (2026-07-09/10) showed
uncapped joblib nesting reliably OOMs this 31GB machine.

Output per scenario: an aggregate CSV (f1_mean/f1_std/n_folds_ok + train/test/total
time means) AND a `*_detail.csv` with one row per grid point × (fold, seed) —
individual F1, a per-class F1 breakdown (f1_class_<c> columns), and
train_time/test_time/total_time for that single run.

Usage:
    python experiments/run_optim_hyper.py --config configs/sensitivity_river.yaml --scenario grid_knn
    python experiments/run_optim_hyper.py --config configs/sensitivity_river.yaml --scenario grid_bary
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import yaml
from joblib import Parallel, delayed

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from experiment_common import (
    _log, _load_and_cap, _iterations, _eval_knn, _eval_bary,
    _write_csv, _write_detail_csv, _time_summary, _detail_row, _merge_best_params,
)

_ALL_METHODS_NO_STA = ['wasps', 'eucl_params', 'eucl_raw']


# ---------------------------------------------------------------------------
# grid_knn — k×gamma calibration grid (river), per method
# ---------------------------------------------------------------------------

def _grid_point_knn(base_cfg: dict, method: str, family: str, sta_epsilon: float,
                    k: int, gamma: float, iterations: list,
                    dataset_overrides: dict, classification_overrides: dict) -> list:
    """One (k, gamma) grid point for one method, evaluated per (fold, seed).
    Runs as a single joblib job — iterations are sequential inside here.

    Returns a list of per-iteration records: {"seed", "fold", "f1", "f1_per_class",
    "train_time", "test_time", "total_time"} — a degenerate (fold, seed) (see
    _eval_knn) contributes a NaN-filled record rather than being dropped, so
    n_folds_ok can still be derived downstream.
    """
    estimator = base_cfg["classification"].get("estimator", "mle")
    records = []
    for seed, fold in iterations:
        data = _load_and_cap(base_cfg, fold, seed, dataset_overrides, classification_overrides)
        res = _eval_knn(method, family, sta_epsilon,
                        data["X_train"], data["y_train"],
                        data["X_test"],  data["y_test"], gamma, k, estimator)
        records.append({"seed": seed, "fold": fold, **res})
    return records


def grid_knn(base_cfg: dict, out_dir: Path, methods: list, k_values: list, gamma_values: list,
            classification_overrides: dict, seed: int, n_jobs: int = -1) -> dict:
    """k×gamma grid per method (river), parallelized over grid points.

    Writes both an aggregate CSV (f1_mean/f1_std/n_folds_ok/train_test_total time
    means) and a `_detail.csv` (one row per (grid point × fold × seed) with the
    individual F1, per-class F1 breakdown, and timings).

    Returns {method: {"k": best_k, "gamma": best_gamma, "f1": best_f1}}.
    """
    family      = base_cfg["dataset"]["family"]
    sta_epsilon = base_cfg["classification"].get("sta_epsilon", 0.05)
    iterations  = _iterations(base_cfg, seed)
    best_params = {}

    for method in methods:
        grid = [(k, g) for k in k_values for g in gamma_values]
        results = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
            delayed(_grid_point_knn)(base_cfg, method, family, sta_epsilon, k, g,
                                     iterations, {}, classification_overrides)
            for k, g in grid
        )

        summary_rows, detail_rows = [], []
        for (k, g), records in zip(grid, results):
            f1s = np.array([r["f1"] for r in records])
            n_ok = int(np.sum(~np.isnan(f1s)))
            summary_rows.append({
                "k": k, "gamma": g,
                "f1_mean": float(np.nanmean(f1s)) if n_ok > 0 else float('nan'),
                "f1_std":  float(np.nanstd(f1s))  if n_ok > 0 else float('nan'),
                "n_folds_ok": n_ok,
                **_time_summary(records),
            })
            detail_rows.extend(_detail_row(r, k=k, gamma=g, seed=r["seed"], fold=r["fold"])
                               for r in records)

        _write_csv(out_dir / f"sensitivity_grid_knn_{method}.csv",
                  ["k", "gamma", "f1_mean", "f1_std", "n_folds_ok",
                   "train_time_mean", "test_time_mean", "total_time_mean"], summary_rows)
        _write_detail_csv(out_dir / f"sensitivity_grid_knn_{method}_detail.csv",
                          ["k", "gamma", "seed", "fold", "f1", "train_time", "test_time", "total_time"],
                          detail_rows)

        f1_means = [r["f1_mean"] for r in summary_rows]
        valid_idx = [i for i, f in enumerate(f1_means) if not np.isnan(f)]
        if not valid_idx:
            raise RuntimeError(f"grid_knn: all grid points NaN for method={method} "
                              "— check data availability / sample caps")
        best_idx = max(valid_idx, key=lambda i: f1_means[i])
        best_k, best_gamma = grid[best_idx]
        best_params[method] = {"k": best_k, "gamma": best_gamma, "f1": f1_means[best_idx]}
        _log(f"[grid_knn] {method}: best k={best_k} gamma={best_gamma:.4g} "
             f"f1={f1_means[best_idx]:.3f} ({summary_rows[best_idx]['n_folds_ok']}/{len(iterations)} runs)")
        _merge_best_params(out_dir, "knn", {method: best_params[method]})
    return best_params


# ---------------------------------------------------------------------------
# grid_bary — lr×gamma calibration grid (river), per method
# ---------------------------------------------------------------------------

def _grid_point_bary(base_cfg: dict, method: str, family: str, sta_epsilon: float,
                     lr: float, gamma: float, iterations: list, n_steps: int,
                     optimizer: str, dataset_overrides: dict, classification_overrides: dict) -> list:
    """Returns per-iteration records — nan-safe, see _grid_point_knn."""
    patience        = base_cfg["classification"].get("early_stop_patience", 15)
    min_rel_improve = base_cfg["classification"].get("early_stop_tol", 1e-4)
    estimator       = base_cfg["classification"].get("estimator", "mle")
    records = []
    for i, (seed, fold) in enumerate(iterations):
        data = _load_and_cap(base_cfg, fold, seed, dataset_overrides, classification_overrides)
        # tag only the first (fold,seed) per grid point — plotting/saving every
        # repetition across a 60+ point grid would be excessive (600+ files)
        tag = f"grid_bary_{method}_lr{lr:.4g}_g{gamma:.4g}" if i == 0 else None
        # bary_n_jobs=1: this function already runs inside an outer Parallel (across grid
        # points, see grid_bary) — the _eval_bary default of 4 would nest dangerously
        # (outer_n_jobs x 4 processes), the pattern documented as having OOM'd this
        # machine before (see run_full_baseline.py module docstring).
        res = _eval_bary(method, family, sta_epsilon,
                         data["X_train"], data["y_train"],
                         data["X_test"],  data["y_test"], gamma, lr, n_steps, optimizer,
                         patience, min_rel_improve, estimator, tag, bary_n_jobs=1)
        records.append({"seed": seed, "fold": fold, **res})
    return records


def grid_bary(base_cfg: dict, out_dir: Path, methods: list, lr_values: list, gamma_values: list,
             classification_overrides: dict, seed: int, n_steps: int,
             optimizer: str = "sgd", n_jobs: int = -1) -> dict:
    """lr×gamma grid per method (river), parallelized over grid points.

    Writes both an aggregate CSV and a `_detail.csv` — see grid_knn docstring.

    Returns {method: {"lr": best_lr, "gamma": best_gamma, "f1": best_f1}}.
    """
    family      = base_cfg["dataset"]["family"]
    sta_epsilon = base_cfg["classification"].get("sta_epsilon", 0.05)
    iterations  = _iterations(base_cfg, seed)
    best_params = {}

    for method in methods:
        grid = [(lr, g) for lr in lr_values for g in gamma_values]
        results = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
            delayed(_grid_point_bary)(base_cfg, method, family, sta_epsilon, lr, g,
                                      iterations, n_steps, optimizer, {}, classification_overrides)
            for lr, g in grid
        )

        summary_rows, detail_rows = [], []
        for (lr, g), records in zip(grid, results):
            f1s = np.array([r["f1"] for r in records])
            n_ok = int(np.sum(~np.isnan(f1s)))
            summary_rows.append({
                "lr": lr, "gamma": g,
                "f1_mean": float(np.nanmean(f1s)) if n_ok > 0 else float('nan'),
                "f1_std":  float(np.nanstd(f1s))  if n_ok > 0 else float('nan'),
                "n_folds_ok": n_ok,
                **_time_summary(records),
            })
            detail_rows.extend(_detail_row(r, lr=lr, gamma=g, seed=r["seed"], fold=r["fold"])
                               for r in records)

        _write_csv(out_dir / f"sensitivity_grid_bary_{method}.csv",
                  ["lr", "gamma", "f1_mean", "f1_std", "n_folds_ok",
                   "train_time_mean", "test_time_mean", "total_time_mean"], summary_rows)
        _write_detail_csv(out_dir / f"sensitivity_grid_bary_{method}_detail.csv",
                          ["lr", "gamma", "seed", "fold", "f1", "train_time", "test_time", "total_time"],
                          detail_rows)

        f1_means = [r["f1_mean"] for r in summary_rows]
        valid_idx = [i for i, f in enumerate(f1_means) if not np.isnan(f)]
        if not valid_idx:
            raise RuntimeError(f"grid_bary: all grid points NaN for method={method} "
                              "— check data availability / sample caps")
        best_idx = max(valid_idx, key=lambda i: f1_means[i])
        best_lr, best_gamma = grid[best_idx]
        best_params[method] = {"lr": best_lr, "gamma": best_gamma, "f1": f1_means[best_idx]}
        _log(f"[grid_bary] {method}: best lr={best_lr:.4g} gamma={best_gamma:.4g} "
             f"f1={f1_means[best_idx]:.3f} ({summary_rows[best_idx]['n_folds_ok']}/{len(iterations)} runs)")
        _merge_best_params(out_dir, "bary", {method: best_params[method]})
    return best_params


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="WaSPS-DTW hyperparameter grid search (real data)")
    parser.add_argument("--config", required=True, help="e.g. configs/sensitivity_river.yaml")
    parser.add_argument("--scenario", required=True, choices=["grid_knn", "grid_bary"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4,
                        help="capped at 4 by default (was -1) — see module docstring "
                             "'Parallelization' for the nested outer×inner risk this bounds")
    parser.add_argument("--debug", action="store_true",
                        help="log per-(fold,seed) resolved params (T, aggregate_days/"
                             "window_size, samples_per_step, n_train/n_test) to the log file")
    parser.add_argument("--verbose", action="store_true",
                        help="save barycenter arrays (.npz) + plots (.png) to "
                             "out_dir/bary_debug/ for the first (fold,seed) of every "
                             "grid point in grid_bary")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    sens = cfg.get("sensitivity", {})
    methods = sens.get("methods", _ALL_METHODS_NO_STA)
    out_dir = Path(cfg.get("output", {}).get("dir", "results/jax_sensitivity"))
    out_dir.mkdir(parents=True, exist_ok=True)
    n_steps = sens.get("n_steps_bary", 100)
    optimizer = cfg.get("classification", {}).get("optimizer", "sgd")

    os.environ["EXPERIMENT_LOG_FILE"] = str(out_dir / "sensitivity.log")
    if args.debug:
        os.environ["EXPERIMENT_DEBUG"] = "1"
    if args.verbose:
        os.environ["EXPERIMENT_VERBOSE"] = "1"
    _log(f"===== scenario={args.scenario} config={args.config} seed={args.seed} "
         f"n_jobs={args.n_jobs} debug={args.debug} verbose={args.verbose} =====")
    _log(f"methods={methods} n_steps_bary={n_steps} optimizer={optimizer} "
         f"estimator={cfg['classification'].get('estimator', 'mle')} "
         f"n_splits={cfg.get('cross_validation', {}).get('n_splits')} "
         f"n_seeds_per_fold={cfg.get('cross_validation', {}).get('n_seeds_per_fold', 1)} "
         f"dataset={cfg['dataset'].get('type')} "
         f"aggregate_days={cfg['dataset'].get('aggregate_days')} "
         f"window_size={cfg['dataset'].get('window_size')}")

    if args.scenario == "grid_knn":
        s = sens["grid_knn"]
        grid_knn(cfg, out_dir, methods, s["k_values"], s["gamma_values"],
                classification_overrides={k: v for k, v in s.items()
                                          if k not in ("k_values", "gamma_values")},
                seed=args.seed, n_jobs=args.n_jobs)

    elif args.scenario == "grid_bary":
        s = sens["grid_bary"]
        grid_bary(cfg, out_dir, methods, s["lr_values"], s["gamma_values"],
                 classification_overrides={k: v for k, v in s.items()
                                           if k not in ("lr_values", "gamma_values")},
                 seed=args.seed, n_steps=n_steps, optimizer=optimizer, n_jobs=args.n_jobs)

    _log(f"===== scenario={args.scenario} done =====")


if __name__ == "__main__":
    main()
