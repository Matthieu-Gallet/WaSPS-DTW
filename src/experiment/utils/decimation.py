"""Decimation (temporal misalignment) sweep — library code only, imported by
run_decimation.py. No CLI here; this used to be run_sensitivity.py, which also had
sweep_n_samples/sweep_n_train scenarios and their own argparse entry point — both
dropped (only the legacy, now-deleted run_sensitivity_pipeline.sh drove them).

Holds a method's calibrated (k,gamma)/(lr,gamma) fixed (from a best_params.json
produced by optimize_gamma.py) and sweeps the decimation fraction, evaluating across
(fold, seed).

Parallelization: sweep values run as independent joblib jobs, each evaluating all folds
sequentially inside the worker. Barycenter-mode sweep points nest fit_barycenters' own
per-class joblib parallelism (n_jobs=4 there, see experiment_common._eval_bary) — with
5-6 outer sweep values, an unbounded outer --n-jobs (-1 → os.cpu_count() workers
regardless of task count) combined with the inner n_jobs=4 risks an uncapped-nesting OOM
(see run_full_baseline.py module docstring for the same pattern crashing this machine
before). --n-jobs defaults to 4 (outer) — bounded 4×4=16 worst case.

Output: an aggregate CSV (f1_mean/f1_std/n_folds_ok + train/test/total time means) AND
a `*_detail.csv` with one row per fraction × (fold, seed) — individual F1, a per-class
F1 breakdown (f1_class_<c> columns), and train_time/test_time/total_time.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from experiment_common import (
    _log, _load_and_cap, _iterations, _eval_knn, _eval_bary,
    _write_csv, _write_detail_csv, _time_summary, _detail_row,
)


def decimate_series(series: list, fraction: float, rng: np.random.Generator) -> list:
    """Remove `fraction` of timesteps independently per series.

    All output series have the same T' = max(2, T − floor(T*fraction)), but each
    series keeps a different random subset of timesteps → temporal distortion
    (misalignment) without shape inconsistency (jnp.stack still works).
    """
    T = np.asarray(series[0]).shape[0]
    n_keep = max(2, T - int(T * fraction))
    return [np.asarray(s)[np.sort(rng.choice(T, n_keep, replace=False))] for s in series]


def _sweep_value_across_folds(base_cfg: dict, method: str, mode: str, iterations: list,
                              gamma: float, k: int, lr: float, n_steps: int, optimizer: str,
                              dataset_overrides: dict, classification_overrides: dict,
                              decimate_fraction: float = None, tag_prefix: str = None) -> list:
    """One decimation fraction for one method/mode, evaluated per (fold, seed).
    Returns a list of per-iteration records: {"seed", "fold", "f1", "f1_per_class",
    "train_time", "test_time", "total_time"}."""
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
            rng = np.random.default_rng(seed * 1000 + (fold if fold is not None else 0))
            X_train = decimate_series(X_train, decimate_fraction, rng)
            X_test  = decimate_series(X_test,  decimate_fraction, rng)
        if mode == 'knn':
            res = _eval_knn(method, family, sta_epsilon, X_train, data["y_train"],
                            X_test, data["y_test"], gamma, k, estimator)
        else:
            # tag only the first (fold,seed) per sweep value — see grid_bary's analogous choice
            tag = tag_prefix if (tag_prefix is not None and i == 0) else None
            # bary_n_jobs=1: this function already runs inside an outer Parallel (across
            # sweep values, see _run_sweep) — the _eval_bary default of 4 would nest
            # dangerously (outer_n_jobs x 4 processes), the same pattern documented as
            # having OOM'd this machine before (see run_full_baseline.py module docstring).
            res = _eval_bary(method, family, sta_epsilon, X_train, data["y_train"],
                             X_test, data["y_test"], gamma, lr, n_steps, optimizer,
                             patience, min_rel_improve, estimator, tag, bary_n_jobs=1)
        records.append({"seed": seed, "fold": fold, **res})
    return records


def _run_sweep(base_cfg: dict, out_dir: Path, methods: list, best_params: dict, seed: int,
              n_steps: int, optimizer: str, value_name: str, values: list,
              make_overrides, out_prefix: str, n_jobs: int = -1,
              decimate: bool = False) -> None:
    """Generic sweep runner: for each method × mode (knn using best_params['knn'],
    bary using best_params['bary']) × value, evaluate across (fold, seed) and write
    a CSV with mean/std columns. `make_overrides(value)` returns the
    classification-key override dict for that sweep value — unused when `decimate=True`
    (decimation is applied post-load via decimate_series, not through overrides)."""
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


def sweep_decimation(base_cfg: dict, out_dir: Path, methods: list, fractions: list,
                     best_params: dict, seed: int, n_steps: int, optimizer: str,
                     fixed_overrides: dict, n_jobs: int = -1) -> None:
    """Decimation (temporal misalignment) sweep at calibrated params, per method,
    per fold, KNN+bary. `fixed_overrides` (samples_per_step, max_train_per_class, …)
    are baked into base_cfg before calling this — decimation itself is applied
    post-load via decimate_series, not through classification_overrides."""
    cfg = copy.deepcopy(base_cfg)
    cfg["classification"].update(fixed_overrides)
    _run_sweep(cfg, out_dir, methods, best_params, seed, n_steps, optimizer,
              "fraction", fractions, make_overrides=None, out_prefix="decimation",
              n_jobs=n_jobs, decimate=True)
