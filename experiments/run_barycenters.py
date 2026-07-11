"""Fit per-class barycenters, evaluate them (predict + F1), and save them as .npy files.

Usage:
    python experiments/run_barycenters.py configs/classification/river_bary.yaml --n-jobs -1
    python experiments/run_barycenters.py configs/classification/cpazmal_bary.yaml --n-jobs -1

Backward compatible: a config with no `cross_validation`/`n_seeds`/`gamma_values` block
(single implicit seed, single gamma) behaves exactly as before this file grew CV/seeds/
evaluation — one `load_dataset` call, fit + save, no metrics.

With `cross_validation`/`n_seeds`/`gamma_values` present, this becomes the barycenter-mode
analog of run_classification.py: the same (seed,fold) iteration convention, per-class F1 +
train/test timing via experiment_common, NaN-safe aggregation, detail CSV, env-var logging,
optional joblib parallelism — "same framework as sensitivity" applied to fit_barycenters +
predict, while STILL saving the fitted barycenter arrays as .npy (this script's original,
distinguishing role — needed by experiments/extract_bary_plots.py).

Saves to <output.dir>/barycenters/:
    <method>_class<label>.npy               — (T, n_params) array, single-gamma configs
    <method>_gamma<gamma>_class<label>.npy   — when gamma_values sweeps multiple values
    metadata.json                            — class names + shapes + config
    (saved once per (method,gamma), at the first (fold,seed) iteration only)

Outputs to <output.dir>/ (only when cross_validation/n_seeds/gamma_values are present):
    barycenter_scores.csv        — mean ± std over (fold,seed) [+ gamma column if swept]
    barycenter_scores_detail.csv — one row per (method[,gamma])×(fold,seed): F1,
                                    per-class F1, train/test timing
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import jax
import numpy as np
import yaml
from joblib import Parallel, delayed

_HERE = Path(__file__).parent
_SRC  = _HERE.parent / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_HERE))

jax.config.update("jax_enable_x64", True)  # must precede jax.numpy usage

from classification.barycenter_clf import fit_barycenters, predict

from data_utils import build_repr, load_dataset, subsample as _subsample
from method_defs import _METHODS, make_cost_fn as _make_cost_fn, make_softdtw_bary as _make_softdtw_bary
from experiment_common import (
    _log, _iterations, _nan_eval_result, _make_eval_result, _write_csv, _write_detail_csv,
    _save_bary_debug,
)


def _fit_eval_bary(method: str, family: str, sta_epsilon: float,
                   train_raw: list, train_labels: np.ndarray,
                   test_raw: list, test_labels: np.ndarray,
                   gamma: float, lr: float, n_steps: int, optimizer: str,
                   patience: int, min_rel_improve: float, estimator: str) -> tuple:
    """Like experiment_common._eval_bary, but also returns the fitted `bary` dict and
    the training representation it was fit from (needed here to save both to disk —
    _eval_bary discards them after computing metrics; the training repr is what
    extract_bary_plots.py needs alongside `bary` to plot sample traces).
    Returns (bary: dict, train_repr: list, train_repr_l: array, eval_result: dict)."""
    repr_type = _METHODS[method]['repr']
    t0 = time.time()
    train_repr, train_repr_l = build_repr(train_raw, train_labels, repr_type, family, estimator)
    if len(train_repr) == 0:
        return {}, [], np.array([]), _nan_eval_result()
    softdtw_bary = _make_softdtw_bary(method, family, sta_epsilon, gamma)
    cost_fn      = _make_cost_fn(method, family, sta_epsilon)
    bary = fit_barycenters(train_repr, train_repr_l, softdtw_bary,
                           n_steps=n_steps, lr=lr, optimizer=optimizer,
                           patience=patience, min_rel_improve=min_rel_improve,
                           n_jobs=4, verbose=False)  # class-level parallelism capped at 4
    t_train = time.time() - t0
    t1 = time.time()
    test_repr, test_repr_l = build_repr(test_raw, test_labels, repr_type, family, estimator)
    if len(test_repr) == 0:
        return bary, train_repr, train_repr_l, _nan_eval_result()
    preds = predict(test_repr, bary, cost_fn, gamma)
    t_test = time.time() - t1
    return bary, train_repr, train_repr_l, _make_eval_result(preds, test_repr_l, t_train, t_test)


def _save_barycenters(out_dir: Path, method: str, gamma: float, multi_gamma: bool,
                      bary: dict, metadata: dict) -> None:
    meta = metadata["methods"].setdefault(method, {})
    for cls, arr in bary.items():
        fname = out_dir / (f"{method}_gamma{gamma:.4g}_class{cls}.npy" if multi_gamma
                           else f"{method}_class{cls}.npy")
        np.save(fname, np.asarray(arr))
        meta[str(cls)] = {"file": fname.name, "shape": list(np.asarray(arr).shape), "gamma": gamma}
        _log(f"  saved {fname.name}  {np.asarray(arr).shape}")


def main(config_path: str, n_jobs: int = 1, verbose: bool = False, debug: bool = False,
        seed_override: int = None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    family      = cfg["dataset"]["family"]
    clf_cfg     = cfg["classification"]
    base_seed   = seed_override if seed_override is not None else cfg.get("seed", 42)
    out_dir     = Path(cfg["output"]["dir"]) / "barycenters"
    out_dir.mkdir(parents=True, exist_ok=True)
    sta_epsilon = clf_cfg.get("sta_epsilon", 0.05)
    estimator   = clf_cfg.get("estimator", "mle")
    optimizer   = clf_cfg.get("optimizer", "sgd")
    n_steps     = clf_cfg["n_steps"]
    lr          = clf_cfg["lr"]
    patience    = clf_cfg.get("early_stop_patience", 15)
    min_rel_improve = clf_cfg.get("early_stop_tol", 1e-4)
    gamma_values = clf_cfg.get("gamma_values") or [clf_cfg["gamma"]]
    multi_gamma  = len(gamma_values) > 1

    os.environ["EXPERIMENT_LOG_FILE"] = str(Path(cfg["output"]["dir"]) / "barycenters.log")
    if debug:
        os.environ["EXPERIMENT_DEBUG"] = "1"
    if verbose:
        os.environ["EXPERIMENT_VERBOSE"] = "1"

    methods_req = [m.lower() for m in cfg.get("methods", list(_METHODS.keys()))]
    methods     = [m for m in methods_req if m in _METHODS]
    unknown     = [m for m in methods_req if m not in _METHODS]
    if unknown:
        _log(f"[warn] unknown methods: {unknown} — skipping")

    has_cv = "cross_validation" in cfg or "n_seeds" in cfg or "gamma_values" in clf_cfg
    if not has_cv:
        # Original behaviour: single load_dataset call, fit + save only, no metrics.
        data = load_dataset(cfg, base_seed)
        train_raw, train_labels = data["X_train"], data["y_train"]
        metadata = {"config": config_path, "family": family, "methods": {}}
        for method in methods:
            repr_type = _METHODS[method]['repr']
            train_repr, train_repr_l = build_repr(train_raw, train_labels, repr_type, family, estimator)
            _log(f"[{method}] fitting barycenters …")
            softdtw_bary = _make_softdtw_bary(method, family, sta_epsilon, gamma_values[0])
            bary = fit_barycenters(train_repr, train_repr_l, softdtw_bary,
                                   n_steps=n_steps, lr=lr, patience=patience,
                                   min_rel_improve=min_rel_improve, optimizer=optimizer)
            _save_barycenters(out_dir, method, gamma_values[0], False, bary, metadata)
        meta_path = out_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))
        _log(f"[done] metadata saved to {meta_path}")
        return

    iterations = _iterations(cfg, base_seed)
    _log(f"[run] dataset={cfg['dataset'].get('type')}  family={family}  methods={methods}  "
         f"n_iter={len(iterations)}  n_jobs={n_jobs}  gamma_values={gamma_values}")

    metadata = {"config": config_path, "family": family, "methods": {}}
    all_results = []

    def _one_iter(seed, fold, save: bool):
        data = load_dataset(cfg, seed, fold=fold)
        rng = np.random.default_rng(seed)
        train_raw, train_labels = _subsample(data["X_train"], data["y_train"],
                                             clf_cfg.get("max_train_samples", -1), rng)
        test_raw, test_labels = _subsample(data["X_test"], data["y_test"],
                                           clf_cfg.get("max_test_samples", -1), rng)
        rows = []
        saved = {}
        for method in methods:
            for gamma in gamma_values:
                bary, train_repr, train_repr_l, res = _fit_eval_bary(
                    method, family, sta_epsilon,
                    train_raw, train_labels, test_raw, test_labels,
                    gamma, lr, n_steps, optimizer,
                    patience, min_rel_improve, estimator)
                rows.append({"method": method, "gamma": gamma, "seed": seed, "fold": fold, **res})
                if save and bary:
                    saved[(method, gamma)] = bary
                    # EXPERIMENT_VERBOSE=1 only: enriched .npz (bary + raw training
                    # series + family/method) for experiments/extract_bary_plots.py.
                    _save_bary_debug(f"barycenters_{method}_gamma{gamma:.4g}", bary,
                                     train_repr, train_repr_l, family=family, method=method)
                _log(f"  [{method} gamma={gamma:.4g}] f1={res['f1']:.3f} "
                     f"train_t={res['train_time']:.1f}s test_t={res['test_time']:.1f}s")
        return rows, saved

    if n_jobs == 1:
        for i, (seed, fold) in enumerate(iterations):
            rows, saved = _one_iter(seed, fold, save=(i == 0))
            all_results.extend(rows)
            for (method, gamma), bary in saved.items():
                _save_barycenters(out_dir, method, gamma, multi_gamma, bary, metadata)
    else:
        out = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
            delayed(_one_iter)(seed, fold, save=(i == 0)) for i, (seed, fold) in enumerate(iterations)
        )
        for rows, saved in out:
            all_results.extend(rows)
            for (method, gamma), bary in saved.items():
                _save_barycenters(out_dir, method, gamma, multi_gamma, bary, metadata)

    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    _log(f"[done] barycenters + metadata saved to {out_dir}")

    # ---- metrics/timing output (summary + detail CSVs), same shape as run_classification.py ----
    from collections import defaultdict
    buckets: dict = defaultdict(list)
    for r in all_results:
        buckets[(r["method"], r["gamma"])].append(r)

    summary_rows = []
    for (method, gamma), rows in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1])):
        f1s = np.array([r["f1"] for r in rows], dtype=float)
        ok = ~np.isnan(f1s)
        n_ok = int(ok.sum())
        train_times = np.array([r["train_time"] for r in rows], dtype=float)
        test_times  = np.array([r["test_time"]  for r in rows], dtype=float)
        summary_rows.append({
            "method": method, "gamma": gamma, "n_seeds": len(rows), "n_ok": n_ok,
            "f1_mean": float(np.nanmean(f1s)) if n_ok else float('nan'),
            "f1_std":  float(np.nanstd(f1s))  if n_ok else float('nan'),
            "train_time_mean": float(np.nanmean(train_times)) if n_ok else float('nan'),
            "test_time_mean":  float(np.nanmean(test_times))  if n_ok else float('nan'),
        })
    out_dir_top = Path(cfg["output"]["dir"])
    _write_csv(out_dir_top / "barycenter_scores.csv",
              ["method", "gamma", "n_seeds", "n_ok", "f1_mean", "f1_std",
               "train_time_mean", "test_time_mean"], summary_rows)

    detail_rows = []
    for r in all_results:
        row = {"method": r["method"], "gamma": r["gamma"], "seed": r["seed"], "fold": r["fold"],
              "f1": r["f1"], "train_time": r["train_time"], "test_time": r["test_time"],
              "total_time": r["total_time"]}
        row.update({f"f1_class_{c}": v for c, v in r.get("f1_per_class", {}).items()})
        detail_rows.append(row)
    _write_detail_csv(out_dir_top / "barycenter_scores_detail.csv",
                      ["method", "gamma", "seed", "fold", "f1", "train_time", "test_time", "total_time"],
                      detail_rows)

    for row in summary_rows:
        _log(f"  {row['method']:18s} gamma={row['gamma']:<8.4g}  "
             f"f1={row['f1_mean']:.3f}±{row['f1_std']:.3f}  ({row['n_ok']}/{row['n_seeds']} ok)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="WaSPS-DTW barycenter fit + evaluate + save")
    parser.add_argument("config", help="e.g. configs/classification/river_bary.yaml")
    parser.add_argument("--n-jobs", type=int, default=5,
                        help="joblib Parallel over (seed,fold), capped at 5 by default "
                             "(pass 1 for sequential); per-class barycenter fit is a separate, "
                             "always-on inner parallelism capped at 4 (_fit_eval_bary)")
    parser.add_argument("--seed", type=int, default=None, help="override config's base seed")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args.config, n_jobs=args.n_jobs, verbose=args.verbose, debug=args.debug,
        seed_override=args.seed)
