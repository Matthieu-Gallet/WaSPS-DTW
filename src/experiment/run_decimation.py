"""Decimation sweep (cpazmal, temporal-misalignment robustness), barycenter mode only,
divergence methods + STA (wasps, eucl_params, eucl_raw, sta — no _nodiv variants), at
each method's own optimal gamma from the baseline experiment's gamma search.

Thin wrapper around utils/decimation.py's sweep_decimation(), reusing it unmodified:
_run_sweep skips any mode whose key is absent from `best_params`, so passing a best_params
dict with ONLY a "bary" key (no "knn") makes it barycenter-only for free.

Per-method gamma comes directly from the baseline experiment's cpazmal
best_params.json['bary'] (results/gamma_search/cpazmal/ by default) — that gamma search
already includes STA, so no separate STA gamma search is needed here.

Settings (samples_per_step, max_train_per_class, max_test_per_class, single holdout fold,
5 seeds) come from configs/config_decimation.yaml, matching the baseline experiment's
cpazmal settings — only the decimation fraction varies on top. sweep_decimation runs
twice: div methods (wasps/eucl_params/eucl_raw) at samples_per_step=480 (config default)
and --n-jobs, sta alone at --sta-samples-per-step (default 48, matching what STA has
always used) and --sta-n-jobs — disjoint method lists write disjoint CSVs, no collision.

Usage:
    python src/experiment/run_decimation.py --config configs/config_decimation.yaml \\
        --gamma-search-dir results/gamma_search --n-jobs 4 --sta-n-jobs 2
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "utils"))

from experiment_common import _log, _load_best_params
from decimation import sweep_decimation

_DIV_METHODS = ['wasps', 'eucl_params', 'eucl_raw']


def main():
    import argparse
    parser = argparse.ArgumentParser(description="cpazmal decimation sweep, barycenter, div+sta")
    parser.add_argument("--config", default="configs/config_decimation.yaml")
    parser.add_argument("--gamma-search-dir", default="results/gamma_search",
                        help="directory containing <dataset>/best_params.json — reads "
                             "<gamma-search-dir>/cpazmal/best_params.json['bary']")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--sta-n-jobs", type=int, default=2)
    parser.add_argument("--sta-samples-per-step", type=int, default=48,
                        help="samples_per_step override for STA's sweep_decimation call "
                             "only — the non-STA call uses --config's own "
                             "sensitivity.sweep_decimation.samples_per_step (e.g. 480)")
    parser.add_argument("--output-dir", default=None,
                        help="override --config's own output.dir — for smoke-testing "
                             "against a scratch directory without touching real output")
    parser.add_argument("--methods", default=",".join(_DIV_METHODS),
                        help="comma-separated divergence method list (excl. sta)")
    parser.add_argument("--sta", dest="sta", action="store_true", default=True,
                        help="include STA as a second sweep_decimation call (default)")
    parser.add_argument("--no-sta", dest="sta", action="store_false",
                        help="skip STA entirely — no required-methods check, no second call")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    div_methods = [m.strip() for m in args.methods.split(",")]

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    sens = cfg["sensitivity"]
    n_steps = sens.get("n_steps_bary", 75)
    optimizer = cfg["classification"].get("optimizer", "sgd")
    out_dir = Path(args.output_dir or cfg.get("output", {}).get("dir", "results/jax_exp3_decimation"))
    out_dir.mkdir(parents=True, exist_ok=True)

    calibrated = _load_best_params(f"{args.gamma_search_dir}/cpazmal/best_params.json")
    bary = calibrated["bary"]
    needed = div_methods + (['sta'] if args.sta else [])
    missing = [m for m in needed if m not in bary]
    if missing:
        raise RuntimeError(f"cpazmal best_params.json['bary'] missing methods {missing} — "
                           "run src/experiment/optimize_gamma.py (baseline gamma search) first")
    best_params = {"bary": {m: bary[m] for m in needed}}

    os.environ["EXPERIMENT_LOG_FILE"] = str(out_dir / "decimation.log")
    if args.debug:
        os.environ["EXPERIMENT_DEBUG"] = "1"
    if args.verbose:
        os.environ["EXPERIMENT_VERBOSE"] = "1"
    gammas = {m: bary[m]["gamma"] for m in needed}
    _log(f"===== run_decimation config={args.config} seed={args.seed} n_jobs={args.n_jobs} "
         f"sta_n_jobs={args.sta_n_jobs} sta={args.sta} gamma(bary)={gammas} =====")

    s = sens["sweep_decimation"]
    fractions = s["fractions"]
    fixed_overrides = {k: v for k, v in s.items() if k != "fractions"}

    sweep_decimation(cfg, out_dir, div_methods, fractions, best_params, args.seed,
                     n_steps, optimizer, fixed_overrides=fixed_overrides, n_jobs=args.n_jobs)
    if args.sta:
        sta_overrides = {**fixed_overrides, "samples_per_step": args.sta_samples_per_step}
        sweep_decimation(cfg, out_dir, ['sta'], fractions, best_params, args.seed,
                         n_steps, optimizer, fixed_overrides=sta_overrides, n_jobs=args.sta_n_jobs)
    _log("===== run_decimation done =====")


if __name__ == "__main__":
    main()
