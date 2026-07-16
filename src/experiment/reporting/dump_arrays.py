"""Per-dataset .npy dump for the baseline experiment (KNN + barycenter modes): raw
train/test time series + labels, plus fitted per-class barycenters (barycenter mode
only).

Raw series are method-independent for a given (dataset, seed) — _load_and_cap's RNG
depends only on seed, not method — so this reads the data once via _load_and_cap
directly, entirely separate from run_full_baseline.py's own narrow
--bary-plot-dataset/--bary-plot-methods mechanism.

STA's methods list is a SEPARATE call with --samples-per-step 48 (--methods sta only)
— it can't share the non-STA call's samples_per_step=480, so callers wanting both must
invoke this script twice per dataset (see src/experiment/script/*.sh).

Usage:
    python src/experiment/reporting/dump_arrays.py --config configs/config_baseline.yaml \\
        --dataset river --seed 42 --out-series results/jax_exp1_baseline/river_series.npy

    python src/experiment/reporting/dump_arrays.py --config configs/config_baseline.yaml \\
        --dataset cpazmal --seed 42 \\
        --out-series results/jax_exp1_baseline/cpazmal_series.npy \\
        --out-barycenters results/jax_exp1_baseline/cpazmal_barycenters.npy \\
        --methods wasps,wasps_nodiv,eucl_params,eucl_params_nodiv,eucl_raw,eucl_raw_nodiv \\
        --gamma-by-method-json results/gamma_search/cpazmal/best_params.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "utils"))

from experiment_common import _load_and_cap, _eval_bary, _log


def dump_series(cfg_block: dict, seed: int, out_path: Path) -> None:
    """One holdout split's raw X_train/y_train/X_test/y_test — method-independent."""
    data = _load_and_cap(
        {"dataset": cfg_block["dataset"], "classification": cfg_block["classification"],
         "cross_validation": {"n_splits": 1}, "n_seeds": 1},
        fold=None, seed=seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, data, allow_pickle=True)
    _log(f"[dump_arrays] saved series+labels (n_train={len(data['X_train'])} "
         f"n_test={len(data['X_test'])}) to {out_path}")


def dump_barycenters(cfg_block: dict, methods: list, gamma_by_method: dict,
                     estimator: str, seed: int, out_path: Path) -> None:
    """Fitted per-class barycenters for each method, at that method's own optimal gamma."""
    data = _load_and_cap(
        {"dataset": cfg_block["dataset"], "classification": cfg_block["classification"],
         "cross_validation": {"n_splits": 1}, "n_seeds": 1},
        fold=None, seed=seed)
    clf = cfg_block["classification"]
    family      = cfg_block["dataset"]["family"]
    sta_epsilon = clf.get("sta_epsilon", 0.05)
    lr          = clf["lr"]
    n_steps     = clf.get("n_steps_bary", 100)
    optimizer   = clf.get("optimizer", "sgd")
    patience    = clf.get("early_stop_patience", 15)
    min_rel_improve = clf.get("early_stop_tol", 1e-4)

    out = {}
    for method in methods:
        bary_n_jobs = 1 if method == "sta" else 4
        res = _eval_bary(method, family, sta_epsilon,
                         data["X_train"], data["y_train"], data["X_test"], data["y_test"],
                         gamma_by_method[method], lr, n_steps, optimizer,
                         patience, min_rel_improve, estimator,
                         bary_n_jobs=bary_n_jobs, return_arrays=True)
        out[method] = {"bary": res["bary"], "gamma": gamma_by_method[method]}
        _log(f"[dump_arrays] barycenters fitted for {method} (gamma={gamma_by_method[method]:.4g})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, out, allow_pickle=True)
    _log(f"[dump_arrays] saved barycenters for {len(methods)} methods to {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Dump train/test series+labels (+barycenters) to .npy")
    parser.add_argument("--config", required=True, help="e.g. configs/config_baseline.yaml")
    parser.add_argument("--dataset", choices=["river", "cpazmal"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-series", required=True)
    parser.add_argument("--out-barycenters", default=None,
                        help="if given, also fit+save barycenters — requires --methods and "
                             "--gamma-by-method-json")
    parser.add_argument("--methods", default=None, help="comma-separated, required with --out-barycenters")
    parser.add_argument("--gamma-by-method-json", default=None,
                        help="best_params.json (e.g. from optimize_gamma.py) — reads its 'bary' section")
    parser.add_argument("--samples-per-step", type=int, default=None,
                        help="override classification.samples_per_step — e.g. 48 for a "
                             "STA-only call (--methods sta), separate from the non-STA "
                             "call's default (e.g. 480)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg_block = cfg[args.dataset]
    estimator = cfg.get("estimator", "mle")
    if args.samples_per_step is not None:
        cfg_block = {**cfg_block, "classification": {**cfg_block["classification"],
                                                      "samples_per_step": args.samples_per_step}}

    dump_series(cfg_block, args.seed, Path(args.out_series))

    if args.out_barycenters:
        assert args.methods and args.gamma_by_method_json, \
            "--out-barycenters requires --methods and --gamma-by-method-json"
        with open(args.gamma_by_method_json) as f:
            best_params = json.load(f)
        gamma_by_method = {m: p["gamma"] for m, p in best_params["bary"].items()}
        methods = [m.strip() for m in args.methods.split(",")]
        dump_barycenters(cfg_block, methods, gamma_by_method, estimator, args.seed,
                         Path(args.out_barycenters))


if __name__ == "__main__":
    main()
