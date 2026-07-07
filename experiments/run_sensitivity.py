"""Sensitivity analysis on REAL data (river + CPAZMaL) — no synthetic sweeps.

Scenarios (select with --scenario):
  grid_knn          — k×gamma grid per method (river), replicated over K-fold groups
  grid_bary         — lr×gamma grid per method (river), replicated over K-fold groups
  sweep_n_samples   — N samples/timestep sweep at calibrated params (river)
  sweep_n_train     — N samples/class sweep at calibrated params (river)
  sweep_decimation  — decimation (temporal misalignment) sweep at calibrated params (river)
  final_comparison  — 4 methods incl. STA, KNN only, fixed gamma/k, river + CPAZMaL

Config-driven: configs/sensitivity_river.yaml (+ sensitivity_cpazmal.yaml for the
CPAZMaL half of final_comparison) select methods/values per scenario — default
methods are all 3 non-STA methods, except final_comparison which includes STA.

Repetition is via K-fold groups (StratifiedGroupKFold) combined with
`cross_validation.n_seeds_per_fold` (>1) — mean±std is taken across all (fold, seed)
combinations. See cv_seed/sample_seed decoupling in river_loader.py / data_utils.py:
the fold split itself never changes with seed; only to_fixed_n/subsample RNG does —
so `n_seeds_per_fold` is a genuine extra repetition axis, not a reshuffle of folds.

STA excluded from grid_knn/grid_bary/sweep_* (cost: O(T²) per pair) — present only
in final_comparison. River loads at aggregate_days=7 (T=52, matching
analysis/river_barycenter_agg4.ipynb) for ALL scenarios (not just final_comparison),
set once in sensitivity_river.yaml's dataset block — this keeps STA tractable in
final_comparison and keeps every other scenario's T consistent with it.

Sample caps: `max_train_samples`/`max_test_samples` (total, matching river.yaml/
cpazmal.yaml convention) OR `max_train_per_class`/`max_test_per_class` (auto-multiplied
by the number of classes actually present in that fold) — the latter is what
final_comparison uses for its "≤50/class" requirement. The loader itself
(load_dataset) does NOT apply either cap; _load_and_cap does, mirroring
run_classification.py's _run_one_seed.

Parallelization: grid points (grid_knn/grid_bary) run as independent joblib jobs,
each evaluating all folds sequentially inside the worker. Do NOT nest this with
fit_barycenters' own per-class joblib parallelism (n_jobs=1 there) — JAX + nested
joblib multiprocessing is a known hang/slowdown risk.

Usage:
    python experiments/run_sensitivity.py --config configs/sensitivity_river.yaml --scenario grid_knn
    python experiments/run_sensitivity.py --config configs/sensitivity_river.yaml --scenario grid_bary
    python experiments/run_sensitivity.py --config configs/sensitivity_river.yaml --scenario sweep_n_samples \\
        --best-params results/jax_sensitivity/best_params.json
    python experiments/run_sensitivity.py --config configs/sensitivity_river.yaml \\
        --cpazmal-config configs/sensitivity_cpazmal.yaml --scenario final_comparison
"""

from __future__ import annotations

import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from joblib import Parallel, delayed

_HERE = Path(__file__).parent
_SRC  = _HERE.parent / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_HERE))

from baselines.sta_wrapper import knn_predict as sta_knn
from classification.nn import knn_predict as sdtw_knn
from classification.barycenter_clf import fit_barycenters, predict

from data_utils import build_repr, load_dataset, subsample as _subsample
from method_defs import _METHODS, make_cost_fn as _make_cost_fn, make_softdtw_bary as _make_softdtw_bary

_ALL_METHODS_NO_STA = ['wasps', 'eucl_params', 'eucl_raw']
_ALL_METHODS        = ['wasps', 'eucl_params', 'eucl_raw', 'sta']


# ---------------------------------------------------------------------------
# Fold loading — one fold of real data, with per-scenario overrides + caps
# ---------------------------------------------------------------------------

def _load_and_cap(base_cfg: dict, fold: int, seed: int,
                  dataset_overrides: dict = None, classification_overrides: dict = None) -> dict:
    """Load one K-fold split with scenario-specific overrides, then apply the
    train/test sample cap — load_dataset itself does NOT do this (only
    run_classification.py's _run_one_seed does, via the same _subsample call).

    `max_train_per_class`/`max_test_per_class` (if present) take priority over
    `max_train_samples`/`max_test_samples` and are multiplied by the number of
    classes actually present in that split.
    """
    cfg = copy.deepcopy(base_cfg)
    cfg["dataset"].update(dataset_overrides or {})
    clf = {**cfg["classification"], **(classification_overrides or {})}
    cfg["classification"] = clf

    data = load_dataset(cfg, seed=seed, fold=fold)
    rng = np.random.default_rng(seed)

    max_train = clf.get("max_train_samples", -1)
    if "max_train_per_class" in clf:
        max_train = clf["max_train_per_class"] * len(np.unique(data["y_train"]))
    max_test = clf.get("max_test_samples", -1)
    if "max_test_per_class" in clf:
        max_test = clf["max_test_per_class"] * len(np.unique(data["y_test"]))

    X_train, y_train = _subsample(data["X_train"], data["y_train"], max_train, rng)
    X_test,  y_test  = _subsample(data["X_test"],  data["y_test"],  max_test,  rng)
    return {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}


def _n_splits(base_cfg: dict) -> int:
    return int(base_cfg.get("cross_validation", {}).get("n_splits", 5))


def _iterations(base_cfg: dict, base_seed: int) -> list:
    """(seed, fold) pairs — mirrors run_classification.py's n_seeds_per_fold
    convention. Fold assignment is seed-independent (cv_seed drives it); only the
    to_fixed_n/subsample RNG varies with seed, so this is a genuine repetition axis
    on top of K-fold, not a reshuffle of which groups are held out."""
    n_splits = _n_splits(base_cfg)
    n_seeds_per_fold = int(base_cfg.get("cross_validation", {}).get("n_seeds_per_fold", 1))
    return [(base_seed + s, f) for f in range(n_splits) for s in range(n_seeds_per_fold)]


# ---------------------------------------------------------------------------
# k clamp — required everywhere k meets a reduced training pool (calibration
# grid, sweep_n_train at small N, final_comparison's sample cap)
# ---------------------------------------------------------------------------

def _clamp_k(k: int, n_train: int) -> int:
    return max(1, min(k, n_train - 1))


# ---------------------------------------------------------------------------
# Per-method evaluation for one fold — mirrors run_classification.py's
# per-method dispatch (STA branch, build_repr, fit_barycenters/predict)
# ---------------------------------------------------------------------------

def _f1(preds, truth) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(truth, preds, average="weighted", zero_division=0))


def _eval_knn(method: str, family: str, sta_epsilon: float,
             train_raw: list, train_labels: np.ndarray,
             test_raw: list, test_labels: np.ndarray,
             gamma: float, k: int) -> float:
    repr_type = _METHODS[method]['repr']
    train_repr, train_repr_l = build_repr(train_raw, train_labels, repr_type, family)
    test_repr,  test_repr_l  = build_repr(test_raw,  test_labels,  repr_type, family)
    if len(train_repr) == 0 or len(test_repr) == 0:
        # Degenerate point (e.g. samples_per_step too small → all series NaN'd out
        # by build_repr for this fold/class). Expected at sweep extremes — record
        # NaN rather than crashing sdtw_knn's jnp.stack([]) or f1_score on empty input.
        return float('nan')
    k_eff = _clamp_k(k, len(train_repr))
    if method == 'sta':
        preds = sta_knn(train_raw, train_labels, test_raw, gamma=gamma, epsilon=sta_epsilon, k=k_eff)
        truth = test_labels
    else:
        cost_fn = _make_cost_fn(method, family, sta_epsilon)
        preds = sdtw_knn(train_repr, train_repr_l, test_repr, cost_fn=cost_fn, gamma=gamma, k=k_eff)
        truth = test_repr_l
    return _f1(preds, truth)


def _eval_bary(method: str, family: str, sta_epsilon: float,
              train_raw: list, train_labels: np.ndarray,
              test_raw: list, test_labels: np.ndarray,
              gamma: float, lr: float, n_steps: int, optimizer: str = "sgd",
              patience: int = 15, min_rel_improve: float = 1e-4) -> float:
    repr_type = _METHODS[method]['repr']
    train_repr, train_repr_l = build_repr(train_raw, train_labels, repr_type, family)
    test_repr,  test_repr_l  = build_repr(test_raw,  test_labels,  repr_type, family)
    if len(train_repr) == 0 or len(test_repr) == 0:
        # Same rationale as _eval_knn: total emptiness would crash fit_barycenters/
        # predict's jnp.stack([]) on the class or barycenter list — record NaN instead.
        return float('nan')
    softdtw_bary = _make_softdtw_bary(method, family, sta_epsilon, gamma)
    cost_fn      = _make_cost_fn(method, family, sta_epsilon)
    bary = fit_barycenters(train_repr, train_repr_l, softdtw_bary,
                           n_steps=n_steps, lr=lr, optimizer=optimizer,
                           patience=patience, min_rel_improve=min_rel_improve,
                           n_jobs=1, verbose=False)  # n_jobs=1: caller may already be in a joblib worker
    preds = predict(test_repr, bary, cost_fn, gamma)
    return _f1(preds, test_repr_l)


# ---------------------------------------------------------------------------
# CSV / JSON helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, fields: list, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _load_best_params(path: str) -> dict:
    """{"knn": {method: {"k":.., "gamma":..}}, "bary": {method: {"lr":.., "gamma":..}}}"""
    return json.loads(Path(path).read_text())


def _merge_best_params(out_dir: Path, key: str, values: dict):
    """Merge `values` under `key` into out_dir/best_params.json (create if absent)."""
    path = out_dir / "best_params.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[key] = values
    path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# grid_knn — k×gamma calibration grid (river), per method
# ---------------------------------------------------------------------------

def _grid_point_knn(base_cfg: dict, method: str, family: str, sta_epsilon: float,
                    k: int, gamma: float, iterations: list,
                    dataset_overrides: dict, classification_overrides: dict) -> tuple:
    """One (k, gamma) grid point for one method, averaged over all (fold, seed)
    combinations. Runs as a single joblib job — iterations are sequential inside here.

    Returns (f1_mean, n_ok) — nan-safe: a degenerate (fold, seed) (see _eval_knn)
    contributes NaN, excluded from the mean rather than poisoning it.
    """
    f1s = []
    for seed, fold in iterations:
        data = _load_and_cap(base_cfg, fold, seed, dataset_overrides, classification_overrides)
        f1s.append(_eval_knn(method, family, sta_epsilon,
                             data["X_train"], data["y_train"],
                             data["X_test"],  data["y_test"], gamma, k))
    f1s = np.array(f1s)
    n_ok = int(np.sum(~np.isnan(f1s)))
    f1_mean = float(np.nanmean(f1s)) if n_ok > 0 else float('nan')
    return f1_mean, n_ok


def grid_knn(base_cfg: dict, out_dir: Path, methods: list, k_values: list, gamma_values: list,
            classification_overrides: dict, seed: int, n_jobs: int = -1) -> dict:
    """k×gamma grid per method (river), parallelized over grid points.

    Returns {method: {"k": best_k, "gamma": best_gamma, "f1": best_f1}}.
    """
    family      = base_cfg["dataset"]["family"]
    sta_epsilon = base_cfg["classification"].get("sta_epsilon", 0.05)
    iterations  = _iterations(base_cfg, seed)
    best_params = {}

    for method in methods:
        grid = [(k, g) for k in k_values for g in gamma_values]
        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_grid_point_knn)(base_cfg, method, family, sta_epsilon, k, g,
                                     iterations, {}, classification_overrides)
            for k, g in grid
        )
        f1_means = [r[0] for r in results]
        n_oks    = [r[1] for r in results]
        rows = [{"k": k, "gamma": g, "f1_mean": f1, "n_folds_ok": n}
                for (k, g), f1, n in zip(grid, f1_means, n_oks)]
        _write_csv(out_dir / f"sensitivity_grid_knn_{method}.csv",
                  ["k", "gamma", "f1_mean", "n_folds_ok"], rows)

        valid_idx = [i for i, f in enumerate(f1_means) if not np.isnan(f)]
        if not valid_idx:
            raise RuntimeError(f"grid_knn: all grid points NaN for method={method} "
                              "— check data availability / sample caps")
        best_idx = max(valid_idx, key=lambda i: f1_means[i])
        best_k, best_gamma = grid[best_idx]
        best_params[method] = {"k": best_k, "gamma": best_gamma, "f1": f1_means[best_idx]}
        print(f"[grid_knn] {method}: best k={best_k} gamma={best_gamma:.4g} "
              f"f1={f1_means[best_idx]:.3f} ({n_oks[best_idx]}/{len(iterations)} runs)", flush=True)

    _merge_best_params(out_dir, "knn", best_params)
    return best_params


# ---------------------------------------------------------------------------
# grid_bary — lr×gamma calibration grid (river), per method
# ---------------------------------------------------------------------------

def _grid_point_bary(base_cfg: dict, method: str, family: str, sta_epsilon: float,
                     lr: float, gamma: float, iterations: list, n_steps: int,
                     optimizer: str, dataset_overrides: dict, classification_overrides: dict) -> tuple:
    """Returns (f1_mean, n_ok) — nan-safe, see _grid_point_knn."""
    patience        = base_cfg["classification"].get("early_stop_patience", 15)
    min_rel_improve = base_cfg["classification"].get("early_stop_tol", 1e-4)
    f1s = []
    for seed, fold in iterations:
        data = _load_and_cap(base_cfg, fold, seed, dataset_overrides, classification_overrides)
        f1s.append(_eval_bary(method, family, sta_epsilon,
                              data["X_train"], data["y_train"],
                              data["X_test"],  data["y_test"], gamma, lr, n_steps, optimizer,
                              patience, min_rel_improve))
    f1s = np.array(f1s)
    n_ok = int(np.sum(~np.isnan(f1s)))
    f1_mean = float(np.nanmean(f1s)) if n_ok > 0 else float('nan')
    return f1_mean, n_ok


def grid_bary(base_cfg: dict, out_dir: Path, methods: list, lr_values: list, gamma_values: list,
             classification_overrides: dict, seed: int, n_steps: int,
             optimizer: str = "sgd", n_jobs: int = -1) -> dict:
    """lr×gamma grid per method (river), parallelized over grid points.

    Returns {method: {"lr": best_lr, "gamma": best_gamma, "f1": best_f1}}.
    """
    family      = base_cfg["dataset"]["family"]
    sta_epsilon = base_cfg["classification"].get("sta_epsilon", 0.05)
    iterations  = _iterations(base_cfg, seed)
    best_params = {}

    for method in methods:
        grid = [(lr, g) for lr in lr_values for g in gamma_values]
        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_grid_point_bary)(base_cfg, method, family, sta_epsilon, lr, g,
                                      iterations, n_steps, optimizer, {}, classification_overrides)
            for lr, g in grid
        )
        f1_means = [r[0] for r in results]
        n_oks    = [r[1] for r in results]
        rows = [{"lr": lr, "gamma": g, "f1_mean": f1, "n_folds_ok": n}
                for (lr, g), f1, n in zip(grid, f1_means, n_oks)]
        _write_csv(out_dir / f"sensitivity_grid_bary_{method}.csv",
                  ["lr", "gamma", "f1_mean", "n_folds_ok"], rows)

        valid_idx = [i for i, f in enumerate(f1_means) if not np.isnan(f)]
        if not valid_idx:
            raise RuntimeError(f"grid_bary: all grid points NaN for method={method} "
                              "— check data availability / sample caps")
        best_idx = max(valid_idx, key=lambda i: f1_means[i])
        best_lr, best_gamma = grid[best_idx]
        best_params[method] = {"lr": best_lr, "gamma": best_gamma, "f1": f1_means[best_idx]}
        print(f"[grid_bary] {method}: best lr={best_lr:.4g} gamma={best_gamma:.4g} "
              f"f1={f1_means[best_idx]:.3f} ({n_oks[best_idx]}/{len(iterations)} runs)", flush=True)

    _merge_best_params(out_dir, "bary", best_params)
    return best_params


# ---------------------------------------------------------------------------
# Shared driver for the 3 per-fold sweeps (n_samples / n_train / decimation)
# ---------------------------------------------------------------------------

def _sweep_value_across_folds(base_cfg: dict, method: str, mode: str, iterations: list,
                              gamma: float, k: int, lr: float, n_steps: int, optimizer: str,
                              dataset_overrides: dict, classification_overrides: dict,
                              decimate_fraction: float = None) -> list:
    """One sweep value (e.g. one N, one n_train cap, one decimation fraction) for
    one method/mode, evaluated per (fold, seed). Returns list of per-run F1s."""
    family      = base_cfg["dataset"]["family"]
    sta_epsilon = base_cfg["classification"].get("sta_epsilon", 0.05)
    patience        = base_cfg["classification"].get("early_stop_patience", 15)
    min_rel_improve = base_cfg["classification"].get("early_stop_tol", 1e-4)
    f1s = []
    for seed, fold in iterations:
        data = _load_and_cap(base_cfg, fold, seed, dataset_overrides, classification_overrides)
        X_train, X_test = data["X_train"], data["X_test"]
        if decimate_fraction is not None:
            rng = np.random.default_rng(seed * 1000 + fold)
            X_train = decimate_series(X_train, decimate_fraction, rng)
            X_test  = decimate_series(X_test,  decimate_fraction, rng)
        if mode == 'knn':
            f1 = _eval_knn(method, family, sta_epsilon, X_train, data["y_train"],
                          X_test, data["y_test"], gamma, k)
        else:
            f1 = _eval_bary(method, family, sta_epsilon, X_train, data["y_train"],
                           X_test, data["y_test"], gamma, lr, n_steps, optimizer,
                           patience, min_rel_improve)
        f1s.append(f1)
    return f1s


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
            print(f"[{out_prefix}] skipping mode={mode}: no '{params_key}' in best_params", flush=True)
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
                ))
            results = Parallel(n_jobs=n_jobs, backend='loky')(jobs)

            rows = []
            for val, f1s in zip(values, results):
                f1s = np.array(f1s)
                n_ok = int(np.sum(~np.isnan(f1s)))
                f1_mean = float(np.nanmean(f1s)) if n_ok > 0 else float('nan')
                f1_std  = float(np.nanstd(f1s))  if n_ok > 0 else float('nan')
                rows.append({value_name: val, "f1_mean": f1_mean, "f1_std": f1_std, "n_folds_ok": n_ok})
            csv_path = out_dir / f"sensitivity_{out_prefix}_{method}_{mode}.csv"
            _write_csv(csv_path, [value_name, "f1_mean", "f1_std", "n_folds_ok"], rows)
            print(f"[{out_prefix}] {method}/{mode}: " +
                  "  ".join(f"{value_name}={r[value_name]}:f1={r['f1_mean']:.3f}±{r['f1_std']:.3f}"
                           for r in rows), flush=True)


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
# final_comparison — 4 methods incl. STA, KNN only, shared gamma/k, both datasets
# ---------------------------------------------------------------------------

def final_comparison(river_cfg: dict, cpazmal_cfg: dict, out_dir: Path,
                     methods: list, gamma: float, k: int, seed: int, n_jobs: int = -1) -> None:
    """4-method (incl. STA) KNN-only comparison, one shared (gamma, k) — NOT a
    per-method calibrated value, since STA has no calibrated pair (excluded from
    grid_knn/grid_bary). River uses aggregate_days=7 (T=52) specifically so STA
    stays tractable; CPAZMaL uses its natural full-year series (T≈29, no
    truncation needed)."""
    for name, cfg in (("river", river_cfg), ("cpazmal", cpazmal_cfg)):
        if cfg is None:
            print(f"[final_comparison] skipping {name}: no config provided", flush=True)
            continue
        fc = cfg["sensitivity"]["final_comparison"]
        family      = cfg["dataset"]["family"]
        sta_epsilon = cfg["classification"].get("sta_epsilon", 0.05)
        iterations  = _iterations(cfg, seed)
        dataset_overrides = {k2: v for k2, v in fc.items()
                             if k2 in ("aggregate_days", "max_time_steps")}
        classification_overrides = {k2: v for k2, v in fc.items()
                                    if k2 not in ("aggregate_days", "max_time_steps", "methods")}

        def _one_iter(seed_i, fold):
            data = _load_and_cap(cfg, fold, seed_i, dataset_overrides, classification_overrides)
            row = {}
            for method in methods:
                row[method] = _eval_knn(method, family, sta_epsilon,
                                        data["X_train"], data["y_train"],
                                        data["X_test"],  data["y_test"], gamma, k)
            return row

        fold_rows = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_one_iter)(seed_i, fold) for seed_i, fold in iterations
        )
        rows = []
        for m in methods:
            vals = np.array([r[m] for r in fold_rows])
            n_ok = int(np.sum(~np.isnan(vals)))
            f1_mean = float(np.nanmean(vals)) if n_ok > 0 else float('nan')
            f1_std  = float(np.nanstd(vals))  if n_ok > 0 else float('nan')
            rows.append({"method": m, "f1_mean": f1_mean, "f1_std": f1_std, "n_folds_ok": n_ok})
        _write_csv(out_dir / f"sensitivity_final_comparison_{name}.csv",
                  ["method", "f1_mean", "f1_std", "n_folds_ok"], rows)
        print(f"[final_comparison/{name}] " +
              "  ".join(f"{r['method']}={r['f1_mean']:.3f}±{r['f1_std']:.3f}" for r in rows),
              flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="WaSPS-DTW sensitivity analysis (real data)")
    parser.add_argument("--config", required=True, help="e.g. configs/sensitivity_river.yaml")
    parser.add_argument("--cpazmal-config", default=None, help="required for --scenario final_comparison")
    parser.add_argument("--scenario", required=True,
                        choices=["grid_knn", "grid_bary", "sweep_n_samples", "sweep_n_train",
                                 "sweep_decimation", "final_comparison"])
    parser.add_argument("--best-params", default=None,
                        help="path to best_params.json from grid_knn/grid_bary (required for sweep_*)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    sens = cfg.get("sensitivity", {})
    methods = sens.get("methods", _ALL_METHODS_NO_STA)
    out_dir = Path(cfg.get("output", {}).get("dir", "results/jax_sensitivity"))
    out_dir.mkdir(parents=True, exist_ok=True)
    n_steps = sens.get("n_steps_bary", 100)
    optimizer = cfg.get("classification", {}).get("optimizer", "sgd")

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

    elif args.scenario in ("sweep_n_samples", "sweep_n_train", "sweep_decimation"):
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

    elif args.scenario == "final_comparison":
        cpazmal_cfg = None
        if args.cpazmal_config:
            with open(args.cpazmal_config) as f:
                cpazmal_cfg = yaml.safe_load(f)
        fc = sens["final_comparison"]
        final_comparison(cfg, cpazmal_cfg, out_dir, fc.get("methods", _ALL_METHODS),
                         fc["gamma"], fc["k"], seed=args.seed, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
