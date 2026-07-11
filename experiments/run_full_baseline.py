"""Final method-comparison baseline — KNN-only, both datasets, divergence vs non-divergence.

Replaces experiments/run_sensitivity.py's old `final_comparison` scenario. Reads a
SINGLE config (configs/full_baseline.yaml) covering both river and cpazmal in their own
`river:`/`cpazmal:` sub-blocks — unlike the old final_comparison, which needed a second
`--cpazmal-config` file.

Methods: wasps/eucl_params/eucl_raw + STA (the original final_comparison scope), plus a
`_nodiv` variant of each of the 3 non-STA methods — is_divergence=False (and, for wasps,
log_correction=False) — to evaluate divergence vs non-divergence KNN performance. This
required adding an `is_divergence` flag to src/classification/nn.py's knn_predict
(2026-07-08): previously is_divergence only affected barycenter fitting/predict, so a
naive eucl_params_nodiv would have been indistinguishable from eucl_params in a KNN-only
comparison. See experiments/method_defs.py for exactly which methods get
is_divergence=True vs False.

Repetition: `n_seeds` (top-level int, default 4) independent random holdout splits — NO
k-fold (each dataset's cross_validation is forced to n_splits=1 internally, regardless
of what the dataset's own classification config would otherwise use elsewhere) — "shuffle
+ select" per the user's request, not StratifiedGroupKFold.

Gamma sweep: a dataset's `classification.gamma_values` (list) sweeps gamma instead of
using the single `classification.gamma` scalar — output rows/detail rows gain a `gamma`
column. Re-running is incremental: a (method, gamma) pair already present in the summary
CSV (from a prior run) is skipped unless --force, so e.g. adding two new gamma values for
a dataset that already has results for gamma=1.0 only computes the new ones and merges
them into the existing CSV rather than recomputing (and clobbering) everything.

--dataset {river,cpazmal,both} restricts which dataset(s) to run (default both).
--gamma overrides gamma_values entirely with a single value, for one-off single-gamma runs
(used by run_full_baseline_sweep.sh to run one (dataset, gamma) pair per OS process).

Memory management (2026-07-09): STA's Sinkhorn calls accumulate RSS within a loky worker
over the course of one method's batch (observed: ~5GB/worker growing over the ~60min it
takes to run 4-5 seeds of STA on river, T=52 — not caused by JIT recompiles, since river's
raw series are rectangular/uniform-shape after to_fixed_n; more likely XLA CPU allocator
buffer accumulation across many small Sinkhorn dispatches in the same process). Two
mitigations, both in this file:
  (1) Method barrier: methods now run strictly sequentially (all seeds of method M finish
      before method M+1 starts), each in its OWN joblib/loky worker pool — the pool is
      explicitly torn down (get_reusable_executor().shutdown(kill_workers=True)) after
      each method's batch so the next method's workers start with clean XLA allocators.
      This is required because joblib's loky backend otherwise reuses a persistent global
      executor across separate Parallel(...) calls, so jax.clear_caches() in the parent
      process would NOT touch the workers' accumulated memory.
  (2) A background psutil thread samples total RSS across the worker process tree once a
      second during each method's batch, reporting the peak (and peak/n_jobs as an
      average-per-job figure) to a dedicated `full_baseline_{name}_resource.csv`.
n_jobs/n_seeds default to 4 (was 5) — user-chosen tradeoff after a repeated OOM/crash at
n_jobs=5 (~26GB RSS for 5 STA workers, close to the 31GB machine's ceiling and still
growing over a single batch).

Output per dataset: full_baseline_{river,cpazmal}.csv (f1_mean/f1_std/n_ok + gamma +
train/test/total time means), full_baseline_{river,cpazmal}_detail.csv (one row per
method × gamma × seed — individual F1, per-class F1 breakdown, timings), and
full_baseline_{river,cpazmal}_resource.csv (one row per method × gamma — batch wall time,
average per-job time, peak/average-per-job RSS, data-load time/RSS) — all written
incrementally after each method's batch completes, so a crash mid-run only loses the
in-progress method, not the whole gamma/dataset.

Usage:
    python experiments/run_full_baseline.py --config configs/full_baseline.yaml --n-jobs 4
    python experiments/run_full_baseline.py --config configs/full_baseline.yaml \\
        --dataset cpazmal --n-jobs 4
    python experiments/run_full_baseline.py --config configs/full_baseline.yaml \\
        --dataset river --gamma 1e-2 --n-jobs 4

    # Preferred: sequential per-(dataset,gamma) OS processes, full memory reclaim between
    # them for free (process exit) — see run_full_baseline_sweep.sh
    ./run_full_baseline_sweep.sh
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
sys.path.insert(0, str(_HERE))

from experiment_common import (
    _log, _iterations, _eval_knn, _write_csv, _write_detail_csv, _time_summary, _detail_row,
)

# is_divergence=True for the base (divergence-on) methods; False for the _nodiv
# variants and unused (but harmless) for 'sta', which has no divergence concept —
# _eval_knn only forwards is_divergence to the non-STA (sdtw_knn) branch.
_IS_DIVERGENCE = {
    'wasps': True,        'wasps_nodiv': False,
    'eucl_params': True,  'eucl_params_nodiv': False,
    'eucl_raw': True,     'eucl_raw_nodiv': False,
    'sta': True,
}

_DEFAULT_METHODS = ['wasps', 'wasps_nodiv', 'eucl_params', 'eucl_params_nodiv',
                    'eucl_raw', 'eucl_raw_nodiv', 'sta']

_SUMMARY_FIELDS = ["method", "gamma", "f1_mean", "f1_std", "n_ok",
                   "train_time_mean", "test_time_mean", "total_time_mean"]
_DETAIL_FIELDS = ["method", "gamma", "seed", "f1", "train_time", "test_time", "total_time"]
_RESOURCE_FIELDS = ["method", "gamma", "n_jobs", "n_tasks", "batch_wall_time_s",
                    "avg_task_time_s", "peak_total_rss_mb", "avg_rss_per_job_mb",
                    "load_avg_time_s", "load_avg_rss_mb"]


def _gamma_key(g) -> str:
    return f"{float(g):.6g}"


_NUMERIC_FIELDS = ("gamma", "f1_mean", "f1_std", "n_ok", "seed", "f1",
                   "train_time_mean", "test_time_mean", "total_time_mean",
                   "train_time", "test_time", "total_time",
                   "n_jobs", "n_tasks", "batch_wall_time_s", "avg_task_time_s",
                   "peak_total_rss_mb", "avg_rss_per_job_mb",
                   "load_avg_time_s", "load_avg_rss_mb")


def _read_existing_csv(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r.setdefault("gamma", "")
        for key in list(r.keys()):
            if r[key] in (None, ""):
                if key == "gamma":
                    r[key] = 1.0
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
    (cross_validation.n_splits=1) regardless of any cross_validation block the
    original dataset config would otherwise carry — this script never k-folds."""
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


def run_full_baseline(base_cfg: dict, out_dir: Path, methods: list, n_seeds: int,
                      estimator: str, seed: int, n_jobs: int = 4,
                      datasets: tuple = ("river", "cpazmal"),
                      gamma_override: float = None, force: bool = False,
                      sta_n_jobs: int = None) -> None:
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

        summary_path  = out_dir / f"full_baseline_{name}.csv"
        detail_path   = out_dir / f"full_baseline_{name}_detail.csv"
        resource_path = out_dir / f"full_baseline_{name}_resource.csv"
        summary_by_key = {(r["method"], _gamma_key(r["gamma"])): r
                          for r in _read_existing_csv(summary_path)}
        detail_rows = _read_existing_csv(detail_path)
        resource_by_key = {(r["method"], _gamma_key(r["gamma"])): r
                           for r in _read_existing_csv(resource_path)}

        for gamma in gamma_values:
            gamma_key = _gamma_key(gamma)
            methods_needed = methods if force else [
                m for m in methods if (m, gamma_key) not in summary_by_key
            ]
            if not methods_needed:
                _log(f"[full_baseline/{name}] gamma={gamma} already computed for all "
                     f"methods — skip (use --force to recompute)")
                continue
            _log(f"[full_baseline/{name}] gamma={gamma} computing methods={methods_needed}")

            cfg = _build_dataset_cfg(base_cfg, name, estimator, n_seeds, gamma=gamma)
            family      = cfg["dataset"]["family"]
            sta_epsilon = cfg["classification"].get("sta_epsilon", 0.05)
            k           = cfg["classification"].get("k", 1)
            iterations  = _iterations(cfg, seed)  # holdout: [(seed_i, None), ...] — no fold axis

            # Methods run strictly sequentially (barrier), each with its own fresh
            # loky worker pool — required both to report clean per-method
            # time/RAM stats and to actually release worker RSS between methods
            # (see module docstring: "Memory management").
            for method in methods_needed:
                def _one_method_seed(seed_i, fold, _method=method, _cfg=cfg,
                                     _family=family, _sta_epsilon=sta_epsilon,
                                     _gamma=gamma, _k=k):
                    from experiment_common import _load_and_cap
                    t0 = time.time()
                    data = _load_and_cap(_cfg, fold, seed_i)
                    t_load = time.time() - t0
                    rss_load_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
                    _log(f"  [{name}] gamma={_gamma} method={_method} seed={seed_i} "
                         f"data loaded in {t_load:.2f}s rss={rss_load_mb:.0f}MB "
                         f"(n_train={len(data['X_train'])} n_test={len(data['X_test'])})")
                    t1 = time.time()
                    result = _eval_knn(_method, _family, _sta_epsilon,
                                       data["X_train"], data["y_train"],
                                       data["X_test"],  data["y_test"], _gamma, _k, estimator,
                                       is_divergence=_IS_DIVERGENCE.get(_method, False))
                    t_method = time.time() - t1
                    _log(f"  [{name}] gamma={_gamma} seed={seed_i} finished {_method} in "
                         f"{t_method:.1f}s (f1={result['f1']:.3f})")
                    return {"seed": seed_i, "fold": fold, "result": result,
                            "t_load": t_load, "rss_load_mb": rss_load_mb, "t_method": t_method}

                job_n = sta_n_jobs if (method == 'sta' and sta_n_jobs is not None) else n_jobs
                _log(f"[full_baseline/{name}] gamma={gamma} method={method} starting batch "
                     f"({len(iterations)} seeds, n_jobs={job_n})")
                t_batch0 = time.time()
                with _RssMonitor() as mon:
                    with Parallel(n_jobs=job_n, backend='loky', verbose=10) as parallel:
                        task_results = parallel(
                            delayed(_one_method_seed)(seed_i, fold) for seed_i, fold in iterations
                        )
                t_batch = time.time() - t_batch0

                # Tear down the loky worker pool so the next method starts with
                # fresh processes (clean XLA allocators) — Parallel(...) reuses a
                # persistent global executor across calls by default, so without
                # this explicit shutdown the accumulated RSS would carry over.
                get_reusable_executor().shutdown(kill_workers=True)
                gc.collect()

                records = [tr["result"] for tr in task_results]
                vals = np.array([r["f1"] for r in records])
                n_ok = int(np.sum(~np.isnan(vals)))
                summary_by_key[(method, gamma_key)] = {
                    "method": method,
                    "gamma": gamma,
                    "f1_mean": float(np.nanmean(vals)) if n_ok > 0 else float('nan'),
                    "f1_std":  float(np.nanstd(vals))  if n_ok > 0 else float('nan'),
                    "n_ok": n_ok,
                    **_time_summary(records),
                }
                detail_rows = [r for r in detail_rows
                              if not (r["method"] == method and _gamma_key(r["gamma"]) == gamma_key)]
                detail_rows.extend(_detail_row(r, method=method, gamma=gamma, seed=tr["seed"])
                                   for tr, r in zip(task_results, records))

                avg_task_time = float(np.mean([tr["t_method"] for tr in task_results]))
                avg_load_time = float(np.mean([tr["t_load"] for tr in task_results]))
                avg_load_rss  = float(np.mean([tr["rss_load_mb"] for tr in task_results]))
                resource_by_key[(method, gamma_key)] = {
                    "method": method, "gamma": gamma, "n_jobs": job_n, "n_tasks": len(iterations),
                    "batch_wall_time_s": round(t_batch, 2),
                    "avg_task_time_s": round(avg_task_time, 2),
                    "peak_total_rss_mb": round(mon.peak_mb, 1),
                    "avg_rss_per_job_mb": round(mon.peak_mb / max(job_n, 1), 1),
                    "load_avg_time_s": round(avg_load_time, 3),
                    "load_avg_rss_mb": round(avg_load_rss, 1),
                }
                _log(f"[full_baseline/{name}] gamma={gamma} method={method} batch done in "
                     f"{t_batch:.1f}s peak_rss={mon.peak_mb:.0f}MB "
                     f"(avg/job={mon.peak_mb / max(job_n, 1):.0f}MB)")

                # Write after every method's batch (not just every gamma) so a
                # crash/kill mid-sweep loses at most the in-progress method.
                summary_rows = sorted(summary_by_key.values(), key=lambda r: (r["gamma"], r["method"]))
                _write_csv(summary_path, _SUMMARY_FIELDS, summary_rows)
                _write_detail_csv(detail_path, _DETAIL_FIELDS, detail_rows)
                resource_rows = sorted(resource_by_key.values(), key=lambda r: (r["gamma"], r["method"]))
                _write_csv(resource_path, _RESOURCE_FIELDS, resource_rows)

        summary_rows = sorted(summary_by_key.values(), key=lambda r: (r["gamma"], r["method"]))
        _log(f"[full_baseline/{name}] " +
             "  ".join(f"{r['method']}@g={r['gamma']:g}={r['f1_mean']:.3f}±{r['f1_std']:.3f}"
                       for r in summary_rows))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="WaSPS-DTW final baseline comparison (KNN-only)")
    parser.add_argument("--config", default="configs/full_baseline.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4,
                        help="capped at 4 by default — 5 caused repeated OOM crashes during "
                             "STA (~5GB RSS/worker, growing over a single method's batch; see "
                             "module docstring 'Memory management')")
    parser.add_argument("--sta-n-jobs", type=int, default=None,
                        help="override n_jobs for the sta method's batch only (other "
                             "methods keep --n-jobs). cpazmal's STA has ~5.75x more "
                             "(test,train) pairs than river's at equal per-class sample "
                             "caps (7 vs 4 classes) — its per-worker RSS climbs faster, "
                             "OOM'd twice at n_jobs=4 (2026-07-10) even after reducing "
                             "Sinkhorn max_iterations and before that too")
    parser.add_argument("--dataset", choices=["river", "cpazmal", "both"], default="both")
    parser.add_argument("--gamma", type=float, default=None,
                        help="override gamma_values with a single value")
    parser.add_argument("--force", action="store_true",
                        help="recompute (method, gamma) pairs already present in the CSV")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    methods = cfg.get("methods", _DEFAULT_METHODS)
    n_seeds = cfg.get("n_seeds", 4)
    estimator = cfg.get("estimator", "mle")
    out_dir = Path(cfg.get("output", {}).get("dir", "results/jax_full_baseline"))
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["EXPERIMENT_LOG_FILE"] = str(out_dir / "full_baseline.log")
    if args.debug:
        os.environ["EXPERIMENT_DEBUG"] = "1"
    if args.verbose:
        os.environ["EXPERIMENT_VERBOSE"] = "1"
    datasets = ("river", "cpazmal") if args.dataset == "both" else (args.dataset,)
    _log(f"===== full_baseline config={args.config} seed={args.seed} n_jobs={args.n_jobs} "
         f"sta_n_jobs={args.sta_n_jobs} dataset={args.dataset} gamma={args.gamma} "
         f"force={args.force} debug={args.debug} verbose={args.verbose} =====")
    _log(f"methods={methods} n_seeds={n_seeds} estimator={estimator}")

    run_full_baseline(cfg, out_dir, methods, n_seeds, estimator, seed=args.seed, n_jobs=args.n_jobs,
                      datasets=datasets, gamma_override=args.gamma, force=args.force,
                      sta_n_jobs=args.sta_n_jobs)
    _log("===== full_baseline done =====")


if __name__ == "__main__":
    main()
