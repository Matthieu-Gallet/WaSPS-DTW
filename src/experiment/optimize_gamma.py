"""Per-method, per-dataset gamma search feeding the baseline experiment (KNN + barycenter,
exp1_knn_baseline.sh / exp2_bary_baseline.sh) — computes an automatic per-method optimum,
once up front, reused by both.

For each dataset (river, cpazmal) and mode (knn, barycenter): 3-seed holdout evaluation of
every method across configs/config_baseline.yaml's gamma_search.gamma_grid, picking the
best gamma per method by mean F1. STA gets its own pair of calls (knn + barycenter) at its
own samples_per_step (gamma_search.sta_samples_per_step, e.g. 48, vs the other methods'
480) — this matches the main run's own STA/non-STA split (see run_full_baseline.py's
--samples-per-step), not a separate reduced-scale calibration trick.

grid_knn/grid_bary (formerly in the now-deleted experiments/run_optim_hyper.py) are
self-contained here — only depend on utils/experiment_common.py.

Method list comes from configs/config_baseline.yaml's own top-level `methods:` key (7,
includes STA) — not duplicated here or hardcoded, and that file itself is never written to.

Usage:
    python src/experiment/optimize_gamma.py --config configs/config_baseline.yaml \\
        --dataset both --n-jobs 4 --sta-n-jobs 2
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import yaml
from joblib import Parallel, delayed

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "utils"))

from experiment_common import (
    _log, _load_and_cap, _iterations, _eval_knn, _eval_bary,
    _write_csv, _write_detail_csv, _time_summary, _detail_row, _merge_best_params,
)

sys.path.insert(0, str(_HERE))
from run_full_baseline import _build_dataset_cfg


# ---------------------------------------------------------------------------
# grid_knn — k×gamma calibration grid, per method
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
    """k×gamma grid per method, parallelized over grid points.

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
# grid_bary — lr×gamma calibration grid, per method
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
    """lr×gamma grid per method, parallelized over grid points.

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
# Per-dataset driver
# ---------------------------------------------------------------------------

def run_for_dataset(cfg: dict, name: str, out_dir: Path,
                    n_jobs: int, sta_n_jobs: int, seed: int = 42) -> None:
    gs_cfg = cfg["gamma_search"]
    grid = gs_cfg["gamma_grid"]
    n_seeds = gs_cfg["n_seeds"]
    sta_samples_per_step = gs_cfg["sta_samples_per_step"]
    methods = cfg["methods"]
    non_sta = [m for m in methods if m != "sta"]
    has_sta = "sta" in methods

    base = _build_dataset_cfg(cfg, name, cfg["estimator"], n_seeds, gamma=grid[0])
    k = base["classification"].get("k", 1)
    lr = base["classification"]["lr"]
    n_steps = base["classification"].get("n_steps_bary", 100)
    optimizer = base["classification"].get("optimizer", "sgd")

    dataset_out = out_dir / name
    dataset_out.mkdir(parents=True, exist_ok=True)

    _log(f"[optimize_gamma/{name}] knn: {len(non_sta)} methods @ samples_per_step="
         f"{base['classification'].get('samples_per_step')}, n_jobs={n_jobs}")
    grid_knn(base, dataset_out, non_sta, [k], grid, {}, seed, n_jobs=n_jobs)

    if has_sta:
        sta_overrides = {"samples_per_step": sta_samples_per_step}
        _log(f"[optimize_gamma/{name}] knn: sta @ {sta_overrides}, n_jobs={sta_n_jobs}")
        grid_knn(base, dataset_out, ["sta"], [k], grid, sta_overrides, seed, n_jobs=sta_n_jobs)
    else:
        _log(f"[optimize_gamma/{name}] knn: 'sta' not in methods, skipping")

    _log(f"[optimize_gamma/{name}] barycenter: {len(non_sta)} methods @ samples_per_step="
         f"{base['classification'].get('samples_per_step')}, n_jobs={n_jobs}")
    grid_bary(base, dataset_out, non_sta, [lr], grid, {}, seed, n_steps, optimizer, n_jobs=n_jobs)

    if has_sta:
        sta_overrides = {"samples_per_step": sta_samples_per_step}
        _log(f"[optimize_gamma/{name}] barycenter: sta @ {sta_overrides}, n_jobs={sta_n_jobs}")
        grid_bary(base, dataset_out, ["sta"], [lr], grid, sta_overrides, seed, n_steps, optimizer,
                 n_jobs=sta_n_jobs)
    else:
        _log(f"[optimize_gamma/{name}] barycenter: 'sta' not in methods, skipping")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Per-method gamma search (baseline experiment)")
    parser.add_argument("--config", default="configs/config_baseline.yaml")
    parser.add_argument("--dataset", choices=["river", "cpazmal", "both"], default="both")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--sta-n-jobs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=None,
                        help="override --config's own gamma_search output — defaults to "
                             "results/gamma_search")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.out_dir or "results/gamma_search")
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["EXPERIMENT_LOG_FILE"] = str(out_dir / "gamma_search.log")
    if args.debug:
        os.environ["EXPERIMENT_DEBUG"] = "1"
    if args.verbose:
        os.environ["EXPERIMENT_VERBOSE"] = "1"

    datasets = ("river", "cpazmal") if args.dataset == "both" else (args.dataset,)
    gs_cfg = cfg["gamma_search"]
    _log(f"===== optimize_gamma config={args.config} dataset={args.dataset} "
         f"n_jobs={args.n_jobs} sta_n_jobs={args.sta_n_jobs} seed={args.seed} =====")
    _log(f"gamma_grid={gs_cfg['gamma_grid']} n_seeds={gs_cfg['n_seeds']} "
         f"sta_samples_per_step={gs_cfg['sta_samples_per_step']} methods={cfg['methods']}")

    for name in datasets:
        run_for_dataset(cfg, name, out_dir, args.n_jobs, args.sta_n_jobs, args.seed)

    _log("===== optimize_gamma done =====")


if __name__ == "__main__":
    main()
