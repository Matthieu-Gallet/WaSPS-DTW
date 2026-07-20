"""Method-comparison baseline — both datasets, divergence vs non-divergence, KNN and/or
barycenter mode. Shared engine behind exp1_knn_baseline.sh and exp2_bary_baseline.sh
(each: per-method gamma from optimize_gamma.py's best_params.json, both datasets, single
mode).

Per-method gamma (2026-07-14): `--gamma-by-method-json <path> --gamma-by-method-key
{knn,bary}` loads `{method: {"gamma":..}}` from a best_params.json (the schema
optimize_gamma.py's grid_knn/grid_bary already write) and uses each method's own optimal
gamma instead of sweeping `gamma_values`. See `run_full_baseline()`'s `gamma_by_method`
param — when None (default), behavior is identical to the gamma-sweep path below.

Methods: wasps/eucl_params/eucl_raw + STA, plus a `_nodiv` variant of each of the 3
non-STA methods (is_divergence=False, and for wasps also log_correction=False) — see
utils/method_defs.py for exactly which methods get is_divergence=True vs False.

Modes: `--modes knn,barycenter` (default knn only, matching this script's original
scope). STA is excluded from barycenter mode by default (`--bary-methods` controls the
method list actually fit as barycenters) — CLAUDE.md's STA complexity warning + an
empirical timing gate on this machine (2 gradient steps on ONE cpazmal fold, 15
train/class × 7 classes, T=29, still running past 150s) confirm STA-barycenter is not
tractable at Experiment 1's scale (5 seeds × 2 datasets). STA-KNN remains tractable (the
Sinkhorn cost matrix is O(T²) but built once per pair, no gradient loop) and stays in
the default method list for knn mode.

Seed-level resume (2026-07-12): resumption is keyed on (method, gamma, mode, seed), not
(method, gamma) — bumping n_seeds and re-running only computes the missing seeds and
recomputes the mean/std over the full old+new seed set, so "keep 4 seeds, add 6 more"
does not silently no-op (the previous (method,gamma)-only key would skip everything
once any seed existed).

Gamma sweep: a dataset's `classification.gamma_values` (list) sweeps gamma instead of
using the single `classification.gamma` scalar — output rows gain a `gamma` column.
--dataset {river,cpazmal,both} restricts which dataset(s) to run (default both).
--gamma overrides gamma_values entirely with a single value.

Memory management (2026-07-09/10): STA's Sinkhorn calls accumulate RSS within a loky
worker over the course of one method's batch. Two mitigations:
  (1) Method barrier: methods run strictly sequentially, each in its own joblib/loky
      worker pool, torn down (get_reusable_executor().shutdown(kill_workers=True))
      after each method's batch.
  (2) A background psutil thread samples total RSS across the worker process tree once
      a second during each method's batch, reporting peak/n_jobs (average-per-job RSS)
      into the detail CSV's rss_mb column (one value per method/gamma/mode batch,
      replicated across that batch's seed rows — batch-granular, not a true per-seed
      figure; never compute a std from it).
--sta-n-jobs caps STA's own concurrency (both the outer per-seed pool AND, in
barycenter mode, the inner per-class fit_barycenters pool) separately from --n-jobs —
cpazmal's STA has ~5.75x more (test,train) pairs than river's at equal per-class sample
caps (7 vs 4 classes), OOM'd at n_jobs=4.

Output per dataset: full_baseline_{river,cpazmal}_detail.csv (one row per method x
gamma x mode x seed: F1, per-class F1 breakdown, timings, rss_mb) and
full_baseline_{river,cpazmal}.log — summary means are derived from the detail CSV at
table-extraction time (reporting/extract_latex_tables.py), not persisted separately.

Usage:
    python src/experiment/run_full_baseline.py --config configs/config_baseline.yaml \\
        --modes knn,barycenter --n-jobs 4 --sta-n-jobs 2 --methods sta --samples-per-step 48
"""

from __future__ import annotations

import csv
import gc
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import yaml
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "utils"))

from experiment_common import (
    _log, _iterations, _eval_knn, _eval_bary, _write_detail_csv, _detail_row,
)

# is_divergence=True for the base (divergence-on) methods; False for the _nodiv
# variants and unused (but harmless) for 'sta', which has no divergence concept.
_IS_DIVERGENCE = {
    'wasps': True,        'wasps_nodiv': False,
    'eucl_params': True,  'eucl_params_nodiv': False,
    'eucl_raw': True,     'eucl_raw_nodiv': False,
    'sta': True,
}

_DEFAULT_METHODS = ['wasps', 'wasps_nodiv', 'eucl_params', 'eucl_params_nodiv',
                    'eucl_raw', 'eucl_raw_nodiv', 'sta']
# STA excluded by default from barycenter fitting — see module docstring.
_DEFAULT_BARY_METHODS = ['wasps', 'wasps_nodiv', 'eucl_params', 'eucl_params_nodiv',
                         'eucl_raw', 'eucl_raw_nodiv']

_DETAIL_FIELDS = ["method", "gamma", "mode", "seed", "f1",
                  "train_time", "test_time", "total_time", "rss_mb"]

_NUMERIC_FIELDS = ("gamma", "seed", "f1", "train_time", "test_time", "total_time", "rss_mb")


def _gamma_key(g) -> str:
    return f"{float(g):.6g}"


def _read_existing_detail(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key in list(r.keys()):
            if r[key] in (None, ""):
                continue
            if key in _NUMERIC_FIELDS or key.startswith("f1_class_"):
                try:
                    r[key] = float(r[key])
                except ValueError:
                    pass
    return rows


def _build_dataset_cfg(base_cfg: dict, name: str, estimator: str, n_seeds: int,
                       gamma: float = None) -> dict:
    """One dataset's {dataset, classification} sub-block, forced to holdout mode
    (cross_validation.n_splits=1) — this script never k-folds."""
    ds_block = base_cfg[name]
    clf = {**ds_block["classification"], "estimator": estimator}
    if gamma is not None:
        clf["gamma"] = gamma
    return {
        "dataset": dict(ds_block["dataset"]),
        "classification": clf,
        "cross_validation": {"n_splits": 1},
        "n_seeds": n_seeds,
    }


# ---------------------------------------------------------------------------
# Resource tracking — separate loky worker processes' RSS isn't visible from
# the parent's own psutil.Process(), so a background thread polls the child
# process tree while a method's batch runs.
# ---------------------------------------------------------------------------

class _RssMonitor:
    def __init__(self, interval: float = 1.0):
        self._interval = interval
        self._stop = threading.Event()
        self._peak = 0.0
        self._thread = None

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        me = psutil.Process()
        while not self._stop.is_set():
            try:
                total = sum(c.memory_info().rss for c in me.children(recursive=True))
                self._peak = max(self._peak, total)
            except psutil.Error:
                pass
            self._stop.wait(self._interval)

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)

    @property
    def peak_mb(self) -> float:
        return self._peak / (1024 ** 2)


def _one_method_seed(cfg, name, gamma, method, mode, family, sta_epsilon, k,
                     estimator, bary_n_jobs, seed_i, fold, want_arrays):
    from experiment_common import _load_and_cap
    t0 = time.time()
    data = _load_and_cap(cfg, fold, seed_i)
    t_load = time.time() - t0
    _log(f"  [{name}] gamma={gamma} method={method} mode={mode} seed={seed_i} "
         f"data loaded in {t_load:.2f}s "
         f"(n_train={len(data['X_train'])} n_test={len(data['X_test'])})")
    t1 = time.time()
    if mode == 'knn':
        result = _eval_knn(method, family, sta_epsilon,
                           data["X_train"], data["y_train"],
                           data["X_test"], data["y_test"], gamma, k, estimator,
                           is_divergence=_IS_DIVERGENCE.get(method, False))
    else:
        clf = cfg["classification"]
        result = _eval_bary(method, family, sta_epsilon,
                            data["X_train"], data["y_train"],
                            data["X_test"], data["y_test"], gamma,
                            lr=clf.get("lr", 1e-2), n_steps=clf.get("n_steps_bary", 100),
                            optimizer=clf.get("optimizer", "sgd"),
                            patience=clf.get("early_stop_patience", 15),
                            min_rel_improve=clf.get("early_stop_tol", 1e-4),
                            estimator=estimator, bary_n_jobs=bary_n_jobs,
                            return_arrays=want_arrays)
    t_method = time.time() - t1
    _log(f"  [{name}] gamma={gamma} mode={mode} seed={seed_i} finished {method} in "
         f"{t_method:.1f}s (f1={result['f1']:.3f})")
    return {"seed": seed_i, "fold": fold, "result": result, "t_method": t_method}


def run_full_baseline(base_cfg: dict, out_dir: Path, methods: list, bary_methods: list,
                      modes: list, n_seeds: int, estimator: str, seed: int, n_jobs: int = 4,
                      datasets: tuple = ("river", "cpazmal"),
                      gamma_override: float = None, force: bool = False,
                      sta_n_jobs: int = None, bary_plot_dataset: str = None,
                      bary_plot_methods: list = None, gamma_by_method: dict = None) -> None:
    """gamma_by_method: optional {method: gamma} — when given, REPLACES the gamma_values
    sweep entirely: each method uses its own fixed gamma instead of the dataset's
    gamma_values/gamma_override/gamma scalar. `cfg["classification"]["gamma"]` (built once
    per dataset below) is never read downstream — the gamma actually used for evaluation is
    always the explicit loop variable — so this override is safe and doesn't need per-gamma
    cfg rebuilding. When None (default), behavior is byte-identical to the sweep path."""
    bary_plot_data: dict = {}

    for name in datasets:
        if name not in base_cfg:
            _log(f"[full_baseline] skipping {name}: no config block provided")
            continue
        ds_clf = base_cfg[name]["classification"]
        if gamma_override is not None:
            gamma_values = [gamma_override]
        elif "gamma_values" in ds_clf:
            gamma_values = ds_clf["gamma_values"]
        else:
            gamma_values = [ds_clf["gamma"]]

        detail_path = out_dir / f"full_baseline_{name}_detail.csv"
        detail_rows = _read_existing_detail(detail_path)
        existing_keys = {(r["method"], _gamma_key(r["gamma"]), r.get("mode", "knn"), r["seed"])
                         for r in detail_rows}

        want_plot = (name == bary_plot_dataset)
        plot_methods = set(bary_plot_methods or [])

        cfg = _build_dataset_cfg(base_cfg, name, estimator, n_seeds, gamma=gamma_values[0])
        family      = cfg["dataset"]["family"]
        sta_epsilon = cfg["classification"].get("sta_epsilon", 0.05)
        k           = cfg["classification"].get("k", 1)
        target_seeds = [s for s, _ in _iterations(cfg, seed)]

        for mode in modes:
            mode_methods = methods if mode == 'knn' else \
                [m for m in methods if m in bary_methods]
            for method in mode_methods:
                # _nodiv variants reuse base method's gamma (differ only in is_divergence, not cost)
                lookup_method = method.replace("_nodiv", "") if "_nodiv" in method else method
                gammas_to_try = [gamma_by_method[lookup_method]] if gamma_by_method else gamma_values
                for gamma in gammas_to_try:
                    gamma_key = _gamma_key(gamma)
                    seeds_needed = target_seeds if force else [
                        s for s in target_seeds
                        if (method, gamma_key, mode, s) not in existing_keys
                    ]
                    if not seeds_needed:
                        _log(f"[full_baseline/{name}] gamma={gamma} mode={mode} "
                             f"method={method} — all {len(target_seeds)} seeds already "
                             f"present, skip (use --force to recompute)")
                        continue
                    _log(f"[full_baseline/{name}] gamma={gamma} mode={mode} method={method} "
                         f"computing seeds={seeds_needed}")

                    is_sta = (method == 'sta')
                    job_n = sta_n_jobs if (is_sta and sta_n_jobs is not None) else n_jobs
                    bary_n_jobs = 1 if is_sta else 4
                    want_arrays = (mode == 'barycenter' and want_plot and method in plot_methods
                                  and seeds_needed[0] == target_seeds[0])

                    t_batch0 = time.time()
                    with _RssMonitor() as mon:
                        with Parallel(n_jobs=job_n, backend='loky', verbose=10) as parallel:
                            task_results = parallel(
                                delayed(_one_method_seed)(
                                    cfg, name, gamma, method, mode, family, sta_epsilon, k,
                                    estimator, bary_n_jobs, seed_i, None,
                                    want_arrays and (seed_i == seeds_needed[0]))
                                for seed_i in seeds_needed
                            )
                    t_batch = time.time() - t_batch0

                    get_reusable_executor().shutdown(kill_workers=True)
                    gc.collect()

                    rss_mb = round(mon.peak_mb / max(job_n, 1), 1)
                    for tr in task_results:
                        res = tr["result"]
                        if want_arrays and "bary" in res:
                            bary_plot_data.setdefault(method, {
                                "bary": res.pop("bary"),
                                "test_repr": np.asarray(res.pop("test_repr"), dtype=object),
                                "test_labels": res.pop("test_labels"),
                            })
                            res.pop("train_repr", None)
                            res.pop("train_labels", None)
                        row = _detail_row(res, method=method, gamma=gamma, mode=mode, seed=tr["seed"])
                        row["rss_mb"] = rss_mb
                        detail_rows.append(row)
                        existing_keys.add((method, gamma_key, mode, tr["seed"]))

                    _log(f"[full_baseline/{name}] gamma={gamma} mode={mode} method={method} "
                         f"batch done in {t_batch:.1f}s peak_rss={mon.peak_mb:.0f}MB "
                         f"(avg/job={rss_mb:.0f}MB)")

                    # Write after every method's batch so a crash/kill mid-sweep loses
                    # at most the in-progress method.
                    _write_detail_csv(detail_path, _DETAIL_FIELDS, detail_rows)

        f1_by_key = {}
        for r in detail_rows:
            f1_by_key.setdefault((r["method"], _gamma_key(r["gamma"]), r.get("mode", "knn")), []).append(r["f1"])
        _log(f"[full_baseline/{name}] " +
             "  ".join(f"{m}@g={g}/{mo}=" +
                       f"{np.nanmean(v):.3f}±{np.nanstd(v):.3f}(n={len(v)})"
                       for (m, g, mo), v in sorted(f1_by_key.items()))[:4000])

    if bary_plot_data:
        out_path = out_dir / f"{bary_plot_dataset}_bary_data.npy"
        np.save(out_path, bary_plot_data, allow_pickle=True)
        _log(f"[full_baseline] saved barycenter+sample plot data to {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="WaSPS-DTW baseline comparison (KNN and/or barycenter)")
    parser.add_argument("--config", default="configs/config_baseline.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--sta-n-jobs", type=int, default=None,
                        help="override n_jobs for the sta method's batch only (both the "
                             "outer per-seed pool in knn mode and the inner per-class "
                             "fit_barycenters pool — capped at 1 regardless — in barycenter mode)")
    parser.add_argument("--dataset", choices=["river", "cpazmal", "both"], default="both")
    parser.add_argument("--gamma", type=float, default=None,
                        help="override gamma_values with a single value")
    parser.add_argument("--modes", default="knn", help="comma-separated: knn,barycenter")
    parser.add_argument("--force", action="store_true",
                        help="recompute (method,gamma,mode,seed) rows already present")
    parser.add_argument("--bary-plot-dataset", default=None, choices=[None, "river", "cpazmal"],
                        help="save fitted barycenters + test samples for this dataset's "
                             "first seed to <dataset>_bary_data.npy (barycenter mode only)")
    parser.add_argument("--bary-plot-methods", default="wasps,eucl_params,eucl_raw",
                        help="comma-separated method list to include in the plot npy")
    parser.add_argument("--gamma-by-method-json", default=None,
                        help="path to a best_params.json (e.g. from optimize_gamma.py) — "
                             "when given, each method uses its own gamma from this file "
                             "instead of sweeping gamma_values")
    parser.add_argument("--gamma-by-method-key", default=None, choices=[None, "knn", "bary"],
                        help="which section of --gamma-by-method-json to read "
                             "({method: {'gamma':..}}) — required if --gamma-by-method-json is set")
    parser.add_argument("--bary-methods-same-as-methods", action="store_true",
                        help="use cfg['methods'] (not cfg['bary_methods']) as the barycenter "
                             "method list too — Experiment 1bis's all-7-incl.-STA design, "
                             "without touching the protected config's bary_methods key")
    parser.add_argument("--output-dir", default=None,
                        help="override --config's own output.dir — lets Experiment 1 (knn) "
                             "and Experiment 1bis (barycenter) write to separate dirs despite "
                             "sharing the same (protected) config file")
    parser.add_argument("--methods", default=None,
                        help="comma-separated override for cfg['methods'] — e.g. 'sta' alone, "
                             "so the STA sub-phase (its own samples_per_step, see "
                             "--samples-per-step) can be run as a separate invocation from the "
                             "non-STA methods without a second config file")
    parser.add_argument("--samples-per-step", type=int, default=None,
                        help="override classification.samples_per_step for every dataset "
                             "block in --config — lets the STA sub-phase run at a different "
                             "sample scale (e.g. 48) than the non-STA methods (e.g. 480) "
                             "against the same config file")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.samples_per_step is not None:
        for name in ("river", "cpazmal"):
            if name in cfg:
                cfg[name]["classification"]["samples_per_step"] = args.samples_per_step

    methods = [m.strip() for m in args.methods.split(",")] if args.methods else cfg.get("methods", _DEFAULT_METHODS)
    bary_methods = methods if args.bary_methods_same_as_methods else cfg.get("bary_methods", _DEFAULT_BARY_METHODS)

    gamma_by_method = None
    if args.gamma_by_method_json:
        import json
        assert args.gamma_by_method_key, "--gamma-by-method-key is required with --gamma-by-method-json"
        with open(args.gamma_by_method_json) as f:
            best_params = json.load(f)
        gamma_by_method = {m: p["gamma"] for m, p in best_params[args.gamma_by_method_key].items()}
    modes = [m.strip() for m in args.modes.split(",")]
    n_seeds = cfg.get("n_seeds", 4)
    estimator = cfg.get("estimator", "mle")
    out_dir = Path(args.output_dir or cfg.get("output", {}).get("dir", "results/jax_full_baseline"))
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["EXPERIMENT_LOG_FILE"] = str(out_dir / "full_baseline.log")
    if args.debug:
        os.environ["EXPERIMENT_DEBUG"] = "1"
    if args.verbose:
        os.environ["EXPERIMENT_VERBOSE"] = "1"
    datasets = ("river", "cpazmal") if args.dataset == "both" else (args.dataset,)
    _log(f"===== full_baseline config={args.config} seed={args.seed} n_jobs={args.n_jobs} "
         f"sta_n_jobs={args.sta_n_jobs} dataset={args.dataset} gamma={args.gamma} "
         f"modes={modes} force={args.force} debug={args.debug} verbose={args.verbose} =====")
    _log(f"methods={methods} bary_methods={bary_methods} n_seeds={n_seeds} estimator={estimator} "
         f"gamma_by_method={gamma_by_method}")

    run_full_baseline(cfg, out_dir, methods, bary_methods, modes, n_seeds, estimator,
                      seed=args.seed, n_jobs=args.n_jobs, datasets=datasets,
                      gamma_override=args.gamma, force=args.force, sta_n_jobs=args.sta_n_jobs,
                      bary_plot_dataset=args.bary_plot_dataset,
                      bary_plot_methods=[m.strip() for m in args.bary_plot_methods.split(",")],
                      gamma_by_method=gamma_by_method)
    _log("===== full_baseline done =====")


if __name__ == "__main__":
    main()
