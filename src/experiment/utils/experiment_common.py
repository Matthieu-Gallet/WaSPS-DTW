"""Shared infrastructure for src/experiment/*.py.

Env-var logging, per-fold data loading + sample capping, KNN/barycenter
evaluation primitives, CSV/detail-row helpers, and best_params.json I/O —
shared by optimize_gamma.py, run_full_baseline.py, run_decimation.py (and
utils/decimation.py) so they all share identical parameter handling, logging,
and metrics/timing recording.

Logging env vars (EXPERIMENT_LOG_FILE / EXPERIMENT_DEBUG / EXPERIMENT_VERBOSE)
are read at call time (not import time) so they're inherited by joblib 'loky'
worker subprocesses forked after a script's main() sets them, without
threading them through every function signature.
"""

from __future__ import annotations

import copy
import csv
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
_SRC  = _HERE.parent.parent          # src/experiment/utils -> src/experiment -> src
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_HERE))

from classification.sta_wrapper import knn_predict as sta_knn
from classification.nn import knn_predict as sdtw_knn
from classification.barycenter_clf import fit_barycenters, predict

from data_utils import build_repr, load_dataset, subsample as _subsample, subsample_per_class as _subsample_per_class
from method_defs import _METHODS, make_cost_fn as _make_cost_fn, make_softdtw_bary as _make_softdtw_bary


# ---------------------------------------------------------------------------
# Logging — a persistent log file (out_dir/<name>.log) in addition to stdout,
# so progress survives even if stdout is lost (e.g. the process is killed by
# the environment before Python's normal exit-time buffer flush). Log file
# path is passed via an env var (not a global) so it's inherited by joblib's
# 'loky' worker subprocesses without needing to thread it through every
# function signature — env vars set before Parallel() forks are inherited.
# EXPERIMENT_DEBUG=1 additionally enables per-(fold,seed) parameter dumps
# (T, aggregate_days/window_size, samples_per_step, resolved n_train/n_test)
# from _load_and_cap — off by default to avoid excessive log volume.
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_path = os.environ.get("EXPERIMENT_LOG_FILE")
    if log_path:
        try:
            with open(log_path, "a") as f:
                f.write(line + "\n")
        except OSError:
            pass


def _debug_log(msg: str) -> None:
    if os.environ.get("EXPERIMENT_DEBUG") == "1":
        _log(msg)


def _save_bary_debug(tag: str, bary: dict, train_repr: list, train_labels: np.ndarray,
                     family: str = None, method: str = None) -> None:
    """EXPERIMENT_VERBOSE=1 only: save the fitted per-class barycenters PLUS the
    training series they were fit from (raw .npz — `class_<c>` for barycenters,
    `series_<i>` + `train_labels` for the training set, plus `family`/`method`
    string scalars so downstream extraction scripts don't need to parse the
    filename tag) so the plot can be reconstructed later without re-running the
    fit, plus a quick semilogy plot (one panel per class, barycenter vs training
    series) to out_dir/bary_debug/. `out_dir` is derived from EXPERIMENT_LOG_FILE's
    parent (same env-var-based cross-process plumbing as _log). Called once per
    grid point/sweep value (first (fold,seed) only — not all repetitions, to
    keep output volume manageable across a large grid). Series are saved
    individually (not stacked) since 'raw' repr series can have a different N
    per series."""
    if os.environ.get("EXPERIMENT_VERBOSE") != "1":
        return
    log_path = os.environ.get("EXPERIMENT_LOG_FILE")
    if not log_path:
        return
    debug_dir = Path(log_path).parent / "bary_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    try:
        save_dict = {f"class_{c}": np.asarray(b) for c, b in bary.items()}
        save_dict["train_labels"] = np.asarray(train_labels)
        for i, s in enumerate(train_repr):
            save_dict[f"series_{i}"] = np.asarray(s)
        if family is not None:
            save_dict["family"] = np.array(family)
        if method is not None:
            save_dict["method"] = np.array(method)
        np.savez(debug_dir / f"{tag}.npz", **save_dict)
    except OSError as e:
        _log(f"[verbose] failed to save {tag}.npz: {e}")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        classes = sorted(bary.keys())
        fig, axes = plt.subplots(1, len(classes), figsize=(3.2 * len(classes), 3), squeeze=False)
        for ax, cls in zip(axes[0], classes):
            idx = [i for i, l in enumerate(train_labels) if l == cls]
            for i in idx:
                s = np.asarray(train_repr[i])
                y = s[:, 0] if s.ndim == 2 else np.nanmean(s, axis=1)
                ax.semilogy(np.clip(y, 1e-8, None), color="steelblue", alpha=0.15, linewidth=0.6)
            b = np.asarray(bary[cls])
            by = b[:, 0] if b.ndim == 2 else np.nanmean(b, axis=1)
            ax.semilogy(np.clip(by, 1e-8, None), color="crimson", linewidth=2)
            ax.set_title(f"class {cls}", fontsize=8)
        fig.suptitle(tag, fontsize=8)
        fig.tight_layout()
        fig.savefig(debug_dir / f"{tag}.png", dpi=80)
        plt.close(fig)
    except Exception as e:
        _log(f"[verbose] failed to plot {tag}: {e}")


# ---------------------------------------------------------------------------
# Fold loading — one fold of real data, with per-scenario overrides + caps
# ---------------------------------------------------------------------------

def _load_and_cap(base_cfg: dict, fold: int, seed: int,
                  dataset_overrides: dict = None, classification_overrides: dict = None) -> dict:
    """Load one K-fold split (fold=None → holdout, no CV) with scenario-specific
    overrides, then apply the train/test sample cap — load_dataset itself does
    NOT do this (only run_classification.py's _run_one_seed did originally, via
    the same _subsample call).

    `max_train_per_class`/`max_test_per_class` (if present) take priority over
    `max_train_samples`/`max_test_samples` and are a TRUE per-class quota (up to
    that many samples for EACH class independently, fewer if a class doesn't have
    that many — see data_utils.subsample_per_class) — NOT a flat total allocated
    proportionally to each class's frequency. (Before 2026-07-16 this multiplied
    max_train_per_class by the class count and did one flat, stratified-proportional
    subsample — silently imbalanced, since sklearn's stratify preserves input class
    proportions rather than equalizing them; e.g. cpazmal's largest class ended up
    with 2x its nominal cap while its smallest class got just 1 sample. See
    RESSOURCES-adjacent conversation history / results/jax_exp1bis_bary_cpazmal_w22
    for the diagnosis.) `max_train_samples`/`max_test_samples` (river's flat-total
    convention, no `_per_class` key) are unaffected — those still use the
    proportional `subsample()`.
    """
    cfg = copy.deepcopy(base_cfg)
    cfg["dataset"].update(dataset_overrides or {})
    clf = {**cfg["classification"], **(classification_overrides or {})}
    cfg["classification"] = clf

    data = load_dataset(cfg, seed=seed, fold=fold)
    rng = np.random.default_rng(seed)

    if "max_train_per_class" in clf:
        X_train, y_train = _subsample_per_class(data["X_train"], data["y_train"],
                                                clf["max_train_per_class"], rng)
        max_train = clf["max_train_per_class"]
    else:
        max_train = clf.get("max_train_samples", -1)
        X_train, y_train = _subsample(data["X_train"], data["y_train"], max_train, rng)
    if "max_test_per_class" in clf:
        X_test, y_test = _subsample_per_class(data["X_test"], data["y_test"],
                                              clf["max_test_per_class"], rng)
        max_test = clf["max_test_per_class"]
    else:
        max_test = clf.get("max_test_samples", -1)
        X_test, y_test = _subsample(data["X_test"], data["y_test"], max_test, rng)

    T = int(np.asarray(X_train[0]).shape[0]) if len(X_train) else -1
    _debug_log(
        f"[load] fold={fold} seed={seed} T={T} "
        f"aggregate_days={cfg['dataset'].get('aggregate_days')} "
        f"window_size={cfg['dataset'].get('window_size')} "
        f"samples_per_step={clf.get('samples_per_step')} "
        f"n_train={len(X_train)} (cap={max_train}) n_test={len(X_test)} (cap={max_test}) "
        f"classes_train={sorted(np.unique(y_train).tolist())}"
    )
    return {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}


def _n_splits(base_cfg: dict) -> int:
    return int(base_cfg.get("cross_validation", {}).get("n_splits", 5))


def _iterations(base_cfg: dict, base_seed: int) -> list:
    """(seed, fold) pairs — n_seeds_per_fold convention. Fold assignment is
    seed-independent (cv_seed drives it); only the to_fixed_n/subsample RNG
    varies with seed, so this is a genuine repetition axis on top of K-fold,
    not a reshuffle of which groups are held out.

    When n_splits <= 1 (no cross_validation block, or n_splits explicitly 1),
    returns (seed, None) pairs — the holdout path (fold=None), one entry per
    seed, no fold dimension at all."""
    n_splits = _n_splits(base_cfg)
    n_seeds_per_fold = int(base_cfg.get("cross_validation", {}).get("n_seeds_per_fold", 1))
    if n_splits <= 1:
        n_seeds = int(base_cfg.get("n_seeds", n_seeds_per_fold))
        return [(base_seed + s, None) for s in range(n_seeds)]
    return [(base_seed + s, f) for f in range(n_splits) for s in range(n_seeds_per_fold)]


# ---------------------------------------------------------------------------
# k clamp — required everywhere k meets a reduced training pool (calibration
# grid, sweep_n_train at small N, full-baseline's sample cap)
# ---------------------------------------------------------------------------

def _clamp_k(k: int, n_train: int) -> int:
    return max(1, min(k, n_train - 1))


# ---------------------------------------------------------------------------
# Per-method evaluation for one fold — mirrors run_classification.py's
# per-method dispatch (STA branch, build_repr, fit_barycenters/predict)
# ---------------------------------------------------------------------------

def _nan_eval_result() -> dict:
    """Degenerate point (e.g. samples_per_step too small → all series NaN'd out
    by build_repr for this fold/class). Expected at sweep extremes — record
    NaN rather than crashing sdtw_knn's jnp.stack([]) or f1_score on empty input."""
    return {"f1": float('nan'), "f1_per_class": {},
            "train_time": float('nan'), "test_time": float('nan'), "total_time": float('nan')}


def _eval_metrics(preds, truth) -> dict:
    """Weighted F1 (headline metric) plus a per-class F1 breakdown."""
    from sklearn.metrics import f1_score
    classes = sorted(set(np.asarray(truth).tolist()))
    f1_weighted = float(f1_score(truth, preds, average="weighted", zero_division=0))
    per_class = f1_score(truth, preds, average=None, labels=classes, zero_division=0)
    return {"f1": f1_weighted, "f1_per_class": {int(c): float(v) for c, v in zip(classes, per_class)}}


def _make_eval_result(preds, truth, t_train: float, t_test: float) -> dict:
    result = _eval_metrics(preds, truth)
    result["train_time"] = t_train
    result["test_time"] = t_test
    result["total_time"] = t_train + t_test
    return result


def _eval_knn(method: str, family: str, sta_epsilon: float,
             train_raw: list, train_labels: np.ndarray,
             test_raw: list, test_labels: np.ndarray,
             gamma: float, k: int, estimator: str = 'mle',
             is_divergence: bool = False) -> dict:
    """Returns {"f1", "f1_per_class", "train_time", "test_time", "total_time"}.
    train_time = building the train representation (the MLE/log_cumulant fit, for
    'params' methods); test_time = building the test representation + the KNN call
    itself (the only cost that depends on the test set).

    is_divergence: forwarded to sdtw_knn (src/classification/nn.py) — computes
    D_gamma(z,x) = SDTW(z,x) - 1/2 SDTW(z,z) - 1/2 SDTW(x,x) instead of plain
    SDTW(z,x) when True. No-op for method='sta' (STA has no divergence concept)."""
    repr_type = _METHODS[method]['repr']
    t0 = time.time()
    train_repr, train_repr_l = build_repr(train_raw, train_labels, repr_type, family, estimator)
    t_train = time.time() - t0
    if len(train_repr) == 0:
        return _nan_eval_result()

    t1 = time.time()
    test_repr, test_repr_l = build_repr(test_raw, test_labels, repr_type, family, estimator)
    if len(test_repr) == 0:
        return _nan_eval_result()
    k_eff = _clamp_k(k, len(train_repr))
    if method == 'sta':
        preds = sta_knn(train_raw, train_labels, test_raw, gamma=gamma, epsilon=sta_epsilon, k=k_eff)
        truth = test_labels
    else:
        cost_fn = _make_cost_fn(method, family, sta_epsilon)
        preds = sdtw_knn(train_repr, train_repr_l, test_repr, cost_fn=cost_fn, gamma=gamma, k=k_eff,
                         is_divergence=is_divergence)
        truth = test_repr_l
    t_test = time.time() - t1
    return _make_eval_result(preds, truth, t_train, t_test)


def _eval_bary(method: str, family: str, sta_epsilon: float,
              train_raw: list, train_labels: np.ndarray,
              test_raw: list, test_labels: np.ndarray,
              gamma: float, lr: float, n_steps: int, optimizer: str = "sgd",
              patience: int = 15, min_rel_improve: float = 1e-4,
              estimator: str = 'mle', tag: str = None, bary_n_jobs: int = 4,
              return_arrays: bool = False) -> dict:
    """Returns {"f1", "f1_per_class", "train_time", "test_time", "total_time"}.
    train_time = train repr build + fit_barycenters (the actual training cost);
    test_time = test repr build + predict (the only cost depending on the test set).

    bary_n_jobs: class-level parallelism inside fit_barycenters — defaults to 4
    (unchanged behavior for existing callers). Callers that already parallelize
    over an outer axis (e.g. run_full_baseline.py's per-seed batch) should pass a
    smaller value for expensive per-class costs (STA) to avoid nested-parallelism
    RSS blowup (outer_n_jobs x bary_n_jobs worker processes).

    return_arrays: when True, also returns "bary"/"train_repr"/"train_labels"/
    "test_repr"/"test_labels" — used to build a consolidated plotting dataset
    (e.g. run_full_baseline.py's bary_data.npy) without a second fit."""
    repr_type = _METHODS[method]['repr']
    t0 = time.time()
    train_repr, train_repr_l = build_repr(train_raw, train_labels, repr_type, family, estimator)
    if len(train_repr) == 0:
        return _nan_eval_result()
    softdtw_bary = _make_softdtw_bary(method, family, sta_epsilon, gamma)
    cost_fn      = _make_cost_fn(method, family, sta_epsilon)
    bary = fit_barycenters(train_repr, train_repr_l, softdtw_bary,
                           n_steps=n_steps, lr=lr, optimizer=optimizer,
                           patience=patience, min_rel_improve=min_rel_improve,
                           n_jobs=bary_n_jobs, verbose=False)
    t_train = time.time() - t0
    if tag is not None:
        _save_bary_debug(tag, bary, train_repr, train_repr_l, family=family, method=method)

    t1 = time.time()
    test_repr, test_repr_l = build_repr(test_raw, test_labels, repr_type, family, estimator)
    if len(test_repr) == 0:
        return _nan_eval_result()
    preds = predict(test_repr, bary, cost_fn, gamma)
    t_test = time.time() - t1
    result = _make_eval_result(preds, test_repr_l, t_train, t_test)
    if return_arrays:
        result["bary"] = bary
        result["train_repr"] = train_repr
        result["train_labels"] = train_repr_l
        result["test_repr"] = test_repr
        result["test_labels"] = test_repr_l
    return result


# ---------------------------------------------------------------------------
# CSV / JSON helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, fields: list, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _write_detail_csv(path: Path, base_fields: list, rows: list) -> None:
    """Per-(fold,seed) detail CSV. Rows carry variable f1_class_<c> columns since
    the class set differs by dataset (river=4, cpazmal up to 7) — fieldnames are
    base_fields plus the sorted union of extra keys actually present across rows."""
    extra = set()
    for r in rows:
        extra.update(k for k in r.keys() if k not in base_fields)
    _write_csv(path, base_fields + sorted(extra), rows)


def _time_summary(records: list) -> dict:
    """Mean of train_time/test_time/total_time across a list of per-iteration
    result dicts (NaN-safe, same convention as f1_mean)."""
    out = {}
    for key in ("train_time", "test_time", "total_time"):
        vals = np.array([r[key] for r in records], dtype=float)
        ok = ~np.isnan(vals)
        out[f"{key}_mean"] = float(np.nanmean(vals)) if ok.any() else float('nan')
    return out


def _detail_row(record: dict, **extra_keys) -> dict:
    """One detail-CSV row from an _eval_knn/_eval_bary result dict: extra_keys
    (e.g. k=.., gamma=.., seed=.., fold=..) first, then f1/timings, then the
    per-class F1 breakdown flattened as f1_class_<c> columns."""
    row = dict(extra_keys)
    row["f1"] = record["f1"]
    row["train_time"] = record["train_time"]
    row["test_time"] = record["test_time"]
    row["total_time"] = record["total_time"]
    row.update({f"f1_class_{c}": v for c, v in record["f1_per_class"].items()})
    return row


def _load_best_params(path: str) -> dict:
    """{"knn": {method: {"k":.., "gamma":..}}, "bary": {method: {"lr":.., "gamma":..}}}"""
    return json.loads(Path(path).read_text())


def _merge_best_params(out_dir: Path, key: str, values: dict):
    """Merge `values` into out_dir/best_params.json under `key`, updating only the
    methods present in `values` — existing methods already under `key` are kept.
    Called once per method (not once per whole grid_knn/grid_bary run) so a crash
    partway through the method loop doesn't lose already-completed methods' best
    params."""
    path = out_dir / "best_params.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data.setdefault(key, {}).update(values)
    path.write_text(json.dumps(data, indent=2))
