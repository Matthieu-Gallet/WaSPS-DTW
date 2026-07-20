"""River barycenter illustration (Experiment 4) — river only, 2 classes (PN, NG),
4 methods (wasps, eucl_params, eucl_raw, sta) overlaid, one PDF per class.

wasps/eucl_params/eucl_raw are fit in their normal divergence mode (is_divergence=True,
method_defs.make_softdtw_bary's default for these keys); STA is fit WITHOUT divergence
(is_divergence=False) — STA has no _nodiv variant in method_defs.py, so this script
builds STA's SoftDTW instance directly rather than going through make_softdtw_bary.

wasps/eucl_params/eucl_raw load at samples_per_step=480 (75 samples/class); STA loads
separately at samples_per_step=48 (25 samples/class, matching what STA has always used
— see configs/config_baseline.yaml's gamma_search.sta_samples_per_step). Both loads use
the same river dataset settings (data_dir, mode, aggregate_days) from --config, so T is
identical across all 4 methods' barycenters despite the samples_per_step difference.

eucl_raw and sta are natively raw-sample barycenters (T, N) — this script fits an
exponential distribution per-timestep on top of each (distributions.fit_time_series)
before plotting, so all 4 methods end up in the same parameter (β) space for the
overlay plot (src/plot/classification_plots.py::plot_class_pair_barycenters).

Usage:
    python src/plot/plot_river_bary_viz.py --config configs/config_baseline.yaml \\
        --output-dir results/jax_exp4_river_bary_viz --gamma 10 --classes NG,PN
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "experiment" / "utils"))
sys.path.insert(0, str(_HERE.parent))

import distributions
from data.preprocess import clean_time_series
from data_utils import load_dataset, build_repr, subsample_per_class
from method_defs import make_softdtw_bary
from classification.barycenter_clf import fit_barycenters
from softdtw import SoftDTW
from baselines.sta_wrapper import make_cost_fn as sta_cost_fn

from classification_plots import plot_class_pair_barycenters

_FAMILY = "exponential"


def _build_cfg(base_cfg: dict, samples_per_step: int) -> dict:
    ds = dict(base_cfg["river"]["dataset"])
    clf = {**base_cfg["river"]["classification"], "samples_per_step": samples_per_step}
    return {"dataset": ds, "classification": clf, "cross_validation": {"n_splits": 1}}


def _filter_and_cap(X, y, class_names: dict, target_names: list, max_per_class: int, seed: int):
    name_to_idx = {v: k for k, v in class_names.items()}
    target_idx = [name_to_idx[n] for n in target_names]
    keep = np.isin(y, target_idx)
    X_f = [X[i] for i in np.flatnonzero(keep)]
    y_f = y[keep]
    rng = np.random.default_rng(seed)
    return subsample_per_class(X_f, y_f, max_per_class, rng)


def _fit_params_on_raw_barycenter(bary_raw: np.ndarray, estimator: str) -> np.ndarray:
    """Fit an exponential distribution per-timestep on a raw-space barycenter
    (T, N) -> (T, 1) β. Used for eucl_raw/sta, whose barycenters are natively raw."""
    dist = distributions.get(_FAMILY)
    return dist.fit_time_series(clean_time_series(np.asarray(bary_raw)), dtype=np.float64,
                                method=estimator)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="River barycenter illustration (2 classes, 4 methods)")
    parser.add_argument("--config", default="configs/config_baseline.yaml")
    parser.add_argument("--output-dir", default="results/jax_exp4_river_bary_viz")
    parser.add_argument("--classes", default="NG,PN")
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--n-samples-per-class", type=int, default=75)
    parser.add_argument("--sta-n-samples-per-class", type=int, default=25)
    parser.add_argument("--samples-per-step", type=int, default=480)
    parser.add_argument("--sta-samples-per-step", type=int, default=48)
    parser.add_argument("--n-steps-bary", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)
    estimator = base_cfg.get("estimator", "log_cumulant")
    river_clf = base_cfg["river"]["classification"]
    lr = river_clf["lr"]
    optimizer = river_clf.get("optimizer", "sgd")
    sta_epsilon = river_clf.get("sta_epsilon", 0.05)
    patience = river_clf.get("early_stop_patience", 20)
    min_rel_improve = river_clf.get("early_stop_tol", 1e-4)
    target_names = [c.strip() for c in args.classes.split(",")]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_dir / Path(args.config).name)

    # --- non-STA load (wasps, eucl_params, eucl_raw), samples_per_step=480 ---
    cfg_main = _build_cfg(base_cfg, args.samples_per_step)
    data_main = load_dataset(cfg_main, seed=args.seed, fold=None)
    class_names = data_main["class_names"]
    X_train, y_train = _filter_and_cap(data_main["X_train"], data_main["y_train"],
                                       class_names, target_names,
                                       args.n_samples_per_class, args.seed)

    X_params, y_params = build_repr(X_train, y_train, "params", _FAMILY, estimator)

    barycenters_by_method = {}

    for method in ("wasps", "eucl_params"):
        softdtw = make_softdtw_bary(method, _FAMILY, sta_epsilon, args.gamma)
        bary = fit_barycenters(X_params, y_params, softdtw, n_steps=args.n_steps_bary,
                               lr=lr, optimizer=optimizer, patience=patience,
                               min_rel_improve=min_rel_improve, verbose=False)
        barycenters_by_method[method] = bary
        print(f"[exp4] {method}: fit {len(bary)} class barycenters (param space)")

    softdtw_raw = make_softdtw_bary("eucl_raw", _FAMILY, sta_epsilon, args.gamma)
    bary_raw = fit_barycenters(X_train, y_train, softdtw_raw, n_steps=args.n_steps_bary,
                               lr=lr, optimizer=optimizer, patience=patience,
                               min_rel_improve=min_rel_improve, verbose=False)
    barycenters_by_method["eucl_raw"] = {
        cls: _fit_params_on_raw_barycenter(b, estimator) for cls, b in bary_raw.items()
    }
    print(f"[exp4] eucl_raw: fit {len(bary_raw)} class barycenters (raw -> params fit post-hoc)")

    # --- STA load, samples_per_step=48, no divergence ---
    cfg_sta = _build_cfg(base_cfg, args.sta_samples_per_step)
    data_sta = load_dataset(cfg_sta, seed=args.seed, fold=None)
    X_train_sta, y_train_sta = _filter_and_cap(data_sta["X_train"], data_sta["y_train"],
                                               data_sta["class_names"], target_names,
                                               args.sta_n_samples_per_class, args.seed)
    softdtw_sta = SoftDTW(sta_cost_fn(sta_epsilon), args.gamma, is_divergence=False, manual_grad=False)
    bary_sta = fit_barycenters(X_train_sta, y_train_sta, softdtw_sta, n_steps=args.n_steps_bary,
                               lr=lr, optimizer=optimizer, patience=patience,
                               min_rel_improve=min_rel_improve, verbose=False)
    barycenters_by_method["sta"] = {
        cls: _fit_params_on_raw_barycenter(b, estimator) for cls, b in bary_sta.items()
    }
    print(f"[exp4] sta: fit {len(bary_sta)} class barycenters (raw -> params fit post-hoc, no divergence)")
    # dump all time-series and barycenters to disk for later inspection
    np.savez(out_dir / "barycenters.npz", **barycenters_by_method)
    np.savez(out_dir / "X_params.npz", X_params=X_params, y_params=y_params)
    np.savez(out_dir / "X_train_sta.npz", X_train_sta=X_train_sta, y_train_sta=y_train_sta)

    plot_class_pair_barycenters(
        barycenters_by_method, X_params, y_params, class_names,
        output_dir=str(out_dir), save_pdf=True, n_samples=200, show_legend=False,
    )
    print(f"[exp4] wrote PDFs to {out_dir / 'class_pairs'}")


if __name__ == "__main__":
    main()
