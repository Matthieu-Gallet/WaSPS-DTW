"""Per-method, per-dataset gamma search feeding Experiment 1 (KNN) and Experiment 1bis
(barycenter) — replaces the old fixed/swept-gamma baseline with an automatic per-method
optimum, computed once up front and reused by both experiments' main runs.

For each dataset (river, cpazmal) and mode (knn, barycenter): 3-seed holdout evaluation of
every method across configs/gamma_search.yaml's gamma_grid, picking the best gamma per method
by mean F1. STA gets its own pair of calls (knn + barycenter) at a much smaller sample scale
(configs/gamma_search.yaml's `sta_reduced` block) — this reduced scale is used ONLY here, never
by the main Experiment 1 / 1bis runs, which always evaluate every method (including STA) at
full scale.

Reuses experiments/run_optim_hyper.py's grid_knn/grid_bary unmodified: passing a single-element
k_values/lr_values list turns their (k,gamma)/(lr,gamma) grid search into a pure gamma search,
with zero new evaluation code. They already write best_params.json in the exact schema
run_sensitivity.py's sweep_decimation (and thus experiments/run_exp3_decimation.py) expects via
_load_best_params — so Experiment 3 reads this step's cpazmal output directly, unmodified.

Method list comes from configs/exp1_baseline.yaml's own top-level `methods:` key (7, includes
STA) — not duplicated here or hardcoded, and that file itself is never written to.

Usage:
    python experiments/optimize_gamma.py --config configs/exp1_baseline.yaml \\
        --gamma-search-config configs/gamma_search.yaml --dataset both --n-jobs 4 --sta-n-jobs 2
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from experiment_common import _log
from run_full_baseline import _build_dataset_cfg
from run_optim_hyper import grid_knn, grid_bary


def run_for_dataset(exp1_cfg: dict, gs_cfg: dict, name: str, out_dir: Path,
                    n_jobs: int, sta_n_jobs: int, seed: int = 42) -> None:
    grid = gs_cfg["gamma_grid"]
    n_seeds = gs_cfg["n_seeds"]
    sta_overrides = gs_cfg["sta_reduced"][name]
    methods = exp1_cfg["methods"]
    non_sta = [m for m in methods if m != "sta"]

    base = _build_dataset_cfg(exp1_cfg, name, exp1_cfg["estimator"], n_seeds, gamma=grid[0])
    k = base["classification"].get("k", 1)
    lr = base["classification"]["lr"]
    n_steps = base["classification"].get("n_steps_bary", 100)
    optimizer = base["classification"].get("optimizer", "sgd")

    dataset_out = out_dir / name
    dataset_out.mkdir(parents=True, exist_ok=True)

    _log(f"[optimize_gamma/{name}] knn: {len(non_sta)} methods @ full scale, n_jobs={n_jobs}")
    grid_knn(base, dataset_out, non_sta, [k], grid, {}, seed, n_jobs=n_jobs)

    _log(f"[optimize_gamma/{name}] knn: sta @ reduced scale {sta_overrides}, "
         f"n_jobs={sta_n_jobs}")
    grid_knn(base, dataset_out, ["sta"], [k], grid, sta_overrides, seed, n_jobs=sta_n_jobs)

    _log(f"[optimize_gamma/{name}] barycenter: {len(non_sta)} methods @ full scale, "
         f"n_jobs={n_jobs}")
    grid_bary(base, dataset_out, non_sta, [lr], grid, {}, seed, n_steps, optimizer, n_jobs=n_jobs)

    _log(f"[optimize_gamma/{name}] barycenter: sta @ reduced scale {sta_overrides}, "
         f"n_jobs={sta_n_jobs}")
    grid_bary(base, dataset_out, ["sta"], [lr], grid, sta_overrides, seed, n_steps, optimizer,
             n_jobs=sta_n_jobs)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Per-method gamma search (Experiment 1 / 1bis)")
    parser.add_argument("--config", default="configs/exp1_baseline.yaml")
    parser.add_argument("--gamma-search-config", default="configs/gamma_search.yaml")
    parser.add_argument("--dataset", choices=["river", "cpazmal", "both"], default="both")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--sta-n-jobs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        exp1_cfg = yaml.safe_load(f)
    with open(args.gamma_search_config) as f:
        gs_cfg = yaml.safe_load(f)

    out_dir = Path(gs_cfg.get("output", {}).get("dir", "results/gamma_search"))
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["EXPERIMENT_LOG_FILE"] = str(out_dir / "gamma_search.log")
    if args.debug:
        os.environ["EXPERIMENT_DEBUG"] = "1"
    if args.verbose:
        os.environ["EXPERIMENT_VERBOSE"] = "1"

    datasets = ("river", "cpazmal") if args.dataset == "both" else (args.dataset,)
    _log(f"===== optimize_gamma config={args.config} gamma_search_config={args.gamma_search_config} "
         f"dataset={args.dataset} n_jobs={args.n_jobs} sta_n_jobs={args.sta_n_jobs} seed={args.seed} =====")
    _log(f"gamma_grid={gs_cfg['gamma_grid']} n_seeds={gs_cfg['n_seeds']} methods={exp1_cfg['methods']}")

    for name in datasets:
        run_for_dataset(exp1_cfg, gs_cfg, name, out_dir, args.n_jobs, args.sta_n_jobs, args.seed)

    _log("===== optimize_gamma done =====")


if __name__ == "__main__":
    main()
