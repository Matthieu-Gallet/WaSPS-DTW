"""Sensitivity analysis on REAL data (river + CPAZMaL) — no synthetic sweeps.

Scenarios (select with --scenario):
  sweep_n_samples   — N samples/timestep sweep at calibrated params (river)
  sweep_n_train     — N samples/class sweep at calibrated params (river)
  sweep_decimation  — decimation (temporal misalignment) sweep at calibrated params (river)

Distinct from experiments/run_optim_hyper.py (grid_knn/grid_bary — hyperparameter
CALIBRATION, a different concern from these sensitivity SWEEPS): this file reads the
best_params.json that run_optim_hyper.py writes and holds (k,gamma)/(lr,gamma) fixed
at their calibrated values while sweeping a single axis (N samples/timestep, N
train/class, or decimation fraction). final_comparison (4 methods incl. STA, KNN-only,
fixed gamma/k, both datasets) has moved to experiments/run_full_baseline.py.

Config-driven: configs/sensitivity_river.yaml / configs/sensitivity_cpazmal.yaml select
methods/values per scenario — default methods are all 3 non-STA methods.

Repetition is via K-fold groups (StratifiedGroupKFold) combined with
`cross_validation.n_seeds_per_fold` (>1) — mean±std is taken across all (fold, seed)
combinations. See cv_seed/sample_seed decoupling in river_loader.py / data_utils.py:
the fold split itself never changes with seed; only to_fixed_n/subsample RNG does —
so `n_seeds_per_fold` is a genuine extra repetition axis, not a reshuffle of folds.

STA excluded from all 3 sweeps (cost: O(T²) per pair) — present only in
run_full_baseline.py. River loads at aggregate_days=7 (T=52, matching
analysis/river_barycenter_agg4.ipynb), set once in sensitivity_river.yaml's dataset
block — keeps every scenario's T consistent with the calibration grids.

Sample caps: `max_train_samples`/`max_test_samples` (total, matching river.yaml/
cpazmal.yaml convention) OR `max_train_per_class`/`max_test_per_class` (auto-multiplied
by the number of classes actually present in that fold). The loader itself
(load_dataset) does NOT apply either cap; _load_and_cap does, mirroring
run_classification.py's _run_one_seed.

Parallelization: sweep values run as independent joblib jobs, each evaluating all folds
sequentially inside the worker. Barycenter-mode sweep points nest fit_barycenters' own
per-class joblib parallelism (n_jobs=4 there, see experiment_common._eval_bary) — with
5-6 outer sweep values, an unbounded outer --n-jobs (-1 → os.cpu_count() workers
regardless of task count) combined with the inner n_jobs=4 risks the same
uncapped-nesting OOM that crashed the machine during run_full_baseline.py's STA phase
at n_jobs=5 (2026-07-09/10). --n-jobs now defaults to 4 (outer) — bounded 4×4=16
worst case.

Output per scenario: an aggregate CSV (f1_mean/f1_std/n_folds_ok + train/test/
total time means) AND a `*_detail.csv` with one row per sweep value × (fold, seed) —
individual F1, a per-class F1 breakdown (f1_class_<c> columns), and
train_time/test_time/total_time for that single run. train_time covers everything
that only touches the training data (representation build, plus fit_barycenters for
bary mode); test_time covers everything needing the test set (representation build +
the KNN/predict call). Timings are measured inside joblib workers running under
n_jobs>1 — not clean absolute numbers under contention, but comparable to each other
within the same run.

Usage:
    python experiments/run_sensitivity.py --config configs/sensitivity_river.yaml --scenario sweep_n_samples \\
        --best-params results/jax_sensitivity/best_params.json
    python experiments/run_sensitivity.py --config configs/sensitivity_river.yaml --scenario sweep_n_train
    python experiments/run_sensitivity.py --config configs/sensitivity_river.yaml --scenario sweep_decimation
"""

from __future__ import annotations

import copy
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
    _write_csv, _write_detail_csv, _time_summary, _detail_row, _load_best_params,
)

_ALL_METHODS_NO_STA = ['wasps', 'eucl_params', 'eucl_raw']


# ---------------------------------------------------------------------------
# Shared driver for the 3 per-fold sweeps (n_samples / n_train / decimation)
# ---------------------------------------------------------------------------

def _sweep_value_across_folds(base_cfg: dict, method: str, mode: str, iterations: list,
                              gamma: float, k: int, lr: float, n_steps: int, optimizer: str,
                              dataset_overrides: dict, classification_overrides: dict,
                              decimate_fraction: float = None, tag_prefix: str = None) -> list:
    """One sweep value (e.g. one N, one n_train cap, one decimation fraction) for
    one method/mode, evaluated per (fold, seed). Returns a list of per-iteration
    records: {"seed", "fold", "f1", "f1_per_class", "train_time", "test_time",
    "total_time"}."""
    family      = base_cfg["dataset"]["family"]
    sta_epsilon = base_cfg["classification"].get("sta_epsilon", 0.05)
    patience        = base_cfg["classification"].get("early_stop_patience", 15)
    min_rel_improve = base_cfg["classification"].get("early_stop_tol", 1e-4)
    estimator       = base_cfg["classification"].get("estimator", "mle")
    records = []
    for i, (seed, fold) in enumerate(iterations):
        data = _load_and_cap(base_cfg, fold, seed, dataset_overrides, classification_overrides)
        X_train, X_test = data["X_train"], data["X_test"]
        if decimate_fraction is not None:
            rng = np.random.default_rng(seed * 1000 + fold)
            X_train = decimate_series(X_train, decimate_fraction, rng)
            X_test  = decimate_series(X_test,  decimate_fraction, rng)
        if mode == 'knn':
            res = _eval_knn(method, family, sta_epsilon, X_train, data["y_train"],
                            X_test, data["y_test"], gamma, k, estimator)
        else:
            # tag only the first (fold,seed) per sweep value — see grid_bary's analogous choice
            tag = tag_prefix if (tag_prefix is not None and i == 0) else None
            res = _eval_bary(method, family, sta_epsilon, X_train, data["y_train"],
                             X_test, data["y_test"], gamma, lr, n_steps, optimizer,
                             patience, min_rel_improve, estimator, tag)
        records.append({"seed": seed, "fold": fold, **res})
    return records


def decimate_series(series: list, fraction: float, rng: np.random.Generator) -> list:
    """Remove `fraction` of timesteps independently per series.

    All output series have the same T' = max(2, T − floor(T*fraction)), but each
    series keeps a different random subset of timesteps → temporal distortion
    (misalignment) without shape inconsistency (jnp.stack still works).
    """
    T = np.asarray(series[0]).shape[0]
    n_keep = max(2, T - int(T * fraction))
    return [np.asarray(s)[np.sort(rng.choice(T, n_keep, replace=False))] for s in series]


def _run_sweep(base_cfg: dict, out_dir: Path, methods: list, best_params: dict, seed: int,
              n_steps: int, optimizer: str, value_name: str, values: list,
              make_overrides, out_prefix: str, n_jobs: int = -1,
              decimate: bool = False) -> None:
    """Generic sweep runner: for each method × mode (knn using best_params['knn'],
    bary using best_params['bary']) × value, evaluate across (fold, seed) and write
    a CSV with mean/std columns. `make_overrides(value)` returns the
    classification-key override dict for that sweep value (e.g. {"samples_per_step": value})."""
    iterations = _iterations(base_cfg, seed)
    for mode, params_key in (('knn', 'knn'), ('barycenter', 'bary')):
        if params_key not in best_params:
            _log(f"[{out_prefix}] skipping mode={mode}: no '{params_key}' in best_params")
            continue
        for method in methods:
            if method not in best_params[params_key]:
                continue
            p = best_params[params_key][method]
            gamma = p["gamma"]
            k  = p.get("k", 1)
            lr = p.get("lr", 1e-2)

            jobs = []
            for val in values:
                overrides = {} if decimate else make_overrides(val)
                jobs.append(delayed(_sweep_value_across_folds)(
                    base_cfg, method, mode, iterations, gamma, k, lr, n_steps, optimizer,
                    {}, overrides, decimate_fraction=(val if decimate else None),
                    tag_prefix=f"{out_prefix}_{method}_{value_name}{val}",
                ))
            results = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(jobs)

            summary_rows, detail_rows = [], []
            for val, records in zip(values, results):
                f1s = np.array([r["f1"] for r in records])
                n_ok = int(np.sum(~np.isnan(f1s)))
                summary_rows.append({
                    value_name: val,
                    "f1_mean": float(np.nanmean(f1s)) if n_ok > 0 else float('nan'),
                    "f1_std":  float(np.nanstd(f1s))  if n_ok > 0 else float('nan'),
                    "n_folds_ok": n_ok,
                    **_time_summary(records),
                })
                detail_rows.extend(_detail_row(r, **{value_name: val}, seed=r["seed"], fold=r["fold"])
                                   for r in records)

            csv_path = out_dir / f"sensitivity_{out_prefix}_{method}_{mode}.csv"
            _write_csv(csv_path, [value_name, "f1_mean", "f1_std", "n_folds_ok",
                                  "train_time_mean", "test_time_mean", "total_time_mean"], summary_rows)
            _write_detail_csv(out_dir / f"sensitivity_{out_prefix}_{method}_{mode}_detail.csv",
                              [value_name, "seed", "fold", "f1", "train_time", "test_time", "total_time"],
                              detail_rows)
            _log(f"[{out_prefix}] {method}/{mode}: " +
                 "  ".join(f"{value_name}={r[value_name]}:f1={r['f1_mean']:.3f}±{r['f1_std']:.3f}"
                          for r in summary_rows))


def sweep_n_samples(base_cfg: dict, out_dir: Path, methods: list, values: list,
                    best_params: dict, seed: int, n_steps: int, optimizer: str,
                    fixed_overrides: dict, n_jobs: int = -1) -> None:
    """N samples/timestep sweep at calibrated params, per method, per fold, KNN+bary."""
    _run_sweep(base_cfg, out_dir, methods, best_params, seed, n_steps, optimizer,
              "n_samples", values,
              make_overrides=lambda v: {**fixed_overrides, "samples_per_step": v},
              out_prefix="n_samples", n_jobs=n_jobs)


def sweep_n_train(base_cfg: dict, out_dir: Path, methods: list, values: list,
                  best_params: dict, seed: int, n_steps: int, optimizer: str,
                  fixed_overrides: dict, n_jobs: int = -1) -> None:
    """N samples/class sweep at calibrated params, per method, per fold, KNN+bary."""
    _run_sweep(base_cfg, out_dir, methods, best_params, seed, n_steps, optimizer,
              "n_train", values,
              make_overrides=lambda v: {**fixed_overrides, "max_train_per_class": v},
              out_prefix="n_train", n_jobs=n_jobs)


def sweep_decimation(base_cfg: dict, out_dir: Path, methods: list, fractions: list,
                     best_params: dict, seed: int, n_steps: int, optimizer: str,
                     fixed_overrides: dict, n_jobs: int = -1) -> None:
    """Decimation (temporal misalignment) sweep at calibrated params, per method,
    per fold, KNN+bary. `fixed_overrides` (samples_per_step, max_train_samples, …)
    are baked into base_cfg before calling this — decimation itself is applied
    post-load via decimate_series, not through classification_overrides."""
    cfg = copy.deepcopy(base_cfg)
    cfg["classification"].update(fixed_overrides)
    _run_sweep(cfg, out_dir, methods, best_params, seed, n_steps, optimizer,
              "fraction", fractions, make_overrides=None, out_prefix="decimation",
              n_jobs=n_jobs, decimate=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="WaSPS-DTW sensitivity sweeps (real data)")
    parser.add_argument("--config", required=True, help="e.g. configs/sensitivity_river.yaml")
    parser.add_argument("--scenario", required=True,
                        choices=["sweep_n_samples", "sweep_n_train", "sweep_decimation"])
    parser.add_argument("--best-params", default=None,
                        help="path to best_params.json from run_optim_hyper.py's "
                             "grid_knn/grid_bary (defaults to out_dir/best_params.json)")
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
                             "sweep value in bary mode")
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

    if args.best_params is None:
        args.best_params = str(out_dir / "best_params.json")
    best_params = _load_best_params(args.best_params)
    s = sens[args.scenario]
    if args.scenario == "sweep_n_samples":
        sweep_n_samples(cfg, out_dir, methods, s["values"], best_params, args.seed,
                        n_steps, optimizer,
                        fixed_overrides={k: v for k, v in s.items() if k != "values"},
                        n_jobs=args.n_jobs)
    elif args.scenario == "sweep_n_train":
        sweep_n_train(cfg, out_dir, methods, s["values"], best_params, args.seed,
                     n_steps, optimizer,
                     fixed_overrides={k: v for k, v in s.items() if k != "values"},
                     n_jobs=args.n_jobs)
    else:
        sweep_decimation(cfg, out_dir, methods, s["fractions"], best_params, args.seed,
                        n_steps, optimizer,
                        fixed_overrides={k: v for k, v in s.items() if k != "fractions"},
                        n_jobs=args.n_jobs)

    _log(f"===== scenario={args.scenario} done =====")


if __name__ == "__main__":
    main()
