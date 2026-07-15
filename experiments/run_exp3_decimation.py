"""Experiment 3 — cpazmal decimation sweep (temporal-misalignment robustness), barycenter
mode only, divergence methods + STA (wasps, eucl_params, eucl_raw, sta — no _nodiv variants),
at each method's own optimal gamma from Experiment 1bis's gamma search.

Thin wrapper around run_sensitivity.py's sweep_decimation(), reusing it unmodified:
_run_sweep skips any mode whose key is absent from `best_params`, so passing a best_params
dict with ONLY a "bary" key (no "knn") makes it barycenter-only for free.

Per-method gamma comes directly from Experiment 1bis's cpazmal best_params.json['bary']
(results/gamma_search/cpazmal/ by default) — Experiment 1bis's own 7-method barycenter sweep
already includes STA, so no separate STA gamma search is needed here.

Settings (samples_per_step, max_train_per_class, max_test_per_class, single holdout fold,
5 seeds) come from configs/exp3_decimation.yaml, copied verbatim from Experiment 1bis's
cpazmal settings — only the decimation fraction varies on top. sweep_decimation runs twice:
div methods (wasps/eucl_params/eucl_raw) at --n-jobs, sta alone at --sta-n-jobs — disjoint
method lists write disjoint CSVs, no collision.

Usage:
    python experiments/run_exp3_decimation.py --config configs/exp3_decimation.yaml \\
        --gamma-search-dir results/gamma_search --n-jobs 4 --sta-n-jobs 2
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from experiment_common import _log, _load_best_params
from run_sensitivity import sweep_decimation

_DIV_METHODS = ['wasps', 'eucl_params', 'eucl_raw']


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Experiment 3 — cpazmal decimation, barycenter, div+sta")
    parser.add_argument("--config", default="configs/exp3_decimation.yaml")
    parser.add_argument("--gamma-search-dir", default="results/gamma_search",
                        help="directory containing <dataset>/best_params.json — reads "
                             "<gamma-search-dir>/cpazmal/best_params.json['bary']")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--sta-n-jobs", type=int, default=2)
    parser.add_argument("--output-dir", default=None,
                        help="override --config's own output.dir — for smoke-testing "
                             "against a scratch directory without touching real output")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    sens = cfg["sensitivity"]
    n_steps = sens.get("n_steps_bary", 75)
    optimizer = cfg["classification"].get("optimizer", "sgd")
    out_dir = Path(args.output_dir or cfg.get("output", {}).get("dir", "results/jax_exp3_decimation"))
    out_dir.mkdir(parents=True, exist_ok=True)

    calibrated = _load_best_params(f"{args.gamma_search_dir}/cpazmal/best_params.json")
    bary = calibrated["bary"]
    missing = [m for m in _DIV_METHODS + ['sta'] if m not in bary]
    if missing:
        raise RuntimeError(f"cpazmal best_params.json['bary'] missing methods {missing} — "
                           "run experiments/optimize_gamma.py (Experiment 1bis's gamma search) first")
    best_params = {"bary": {m: bary[m] for m in _DIV_METHODS + ['sta']}}

    os.environ["EXPERIMENT_LOG_FILE"] = str(out_dir / "exp3_decimation.log")
    if args.debug:
        os.environ["EXPERIMENT_DEBUG"] = "1"
    if args.verbose:
        os.environ["EXPERIMENT_VERBOSE"] = "1"
    gammas = {m: bary[m]["gamma"] for m in _DIV_METHODS + ['sta']}
    _log(f"===== exp3_decimation config={args.config} seed={args.seed} n_jobs={args.n_jobs} "
         f"sta_n_jobs={args.sta_n_jobs} gamma(bary)={gammas} =====")

    s = sens["sweep_decimation"]
    fractions = s["fractions"]
    fixed_overrides = {k: v for k, v in s.items() if k != "fractions"}

    sweep_decimation(cfg, out_dir, _DIV_METHODS, fractions, best_params, args.seed,
                     n_steps, optimizer, fixed_overrides=fixed_overrides, n_jobs=args.n_jobs)
    sweep_decimation(cfg, out_dir, ['sta'], fractions, best_params, args.seed,
                     n_steps, optimizer, fixed_overrides=fixed_overrides, n_jobs=args.sta_n_jobs)
    _log("===== exp3_decimation done =====")


if __name__ == "__main__":
    main()
