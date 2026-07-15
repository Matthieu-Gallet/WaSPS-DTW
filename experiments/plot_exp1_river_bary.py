"""Experiment 1's final figure: river test samples + fitted barycenters for 2 classes
(NG idx 0, PN idx 5 — see data/river/metadata_balanced.npy's idx_to_regime) across the
3 divergence methods (wasps, eucl_params, eucl_raw) + STA.

wasps/eucl_params/eucl_raw barycenters + test samples come straight from
<exp1_out_dir>/river_bary_data.npy (written by run_full_baseline.py --bary-plot-dataset
river --bary-plot-methods wasps,eucl_params,eucl_raw at Experiment 1's first seed).

STA is excluded from Experiment 1's main barycenter sweep (not tractable at 5 seeds x
15 train/class x 2 datasets — see run_full_baseline.py module docstring), but a SINGLE
narrow STA barycenter fit (2 classes only, one seed) for this illustrative figure is
cheap enough to compute here directly.

One PDF per class (2 total), 4 subplots each (wasps / eucl_params / eucl_raw / sta),
reusing plot_barycenter_debug's β->λ inversion convention for wasps/eucl_params so all
4 panels show the same discharge-trend units as eucl_raw.

Usage:
    python experiments/plot_exp1_river_bary.py --config configs/exp1_baseline.yaml \\
        --exp1-out-dir results/jax_exp1_baseline \\
        --output-dir results/jax_exp1_baseline/results
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

_HERE = Path(__file__).parent
_SRC  = _HERE.parent / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_HERE))

from experiment_common import _log, _load_and_cap, _eval_bary

_TARGET_CLASSES = {0: 'NG', 5: 'PN'}
_METHOD_ORDER = ['wasps', 'eucl_params', 'eucl_raw', 'sta']
_BARY_COLORS = {'wasps': '#2ca02c', 'eucl_params': '#ff7f0e',
                'eucl_raw': '#1f77b4', 'sta': '#9467bd'}


def _fit_sta_bary(cfg: dict, seed: int, sta_n_steps: int) -> dict:
    """STA barycenter fit restricted to the 2 target classes only — cheap enough
    for a one-off illustrative fit (unlike the full 5-seed x 2-dataset sweep)."""
    data = _load_and_cap(cfg, fold=None, seed=seed)
    keep = np.isin(data["y_train"], list(_TARGET_CLASSES))
    X_train = [x for x, k in zip(data["X_train"], keep) if k]
    y_train = data["y_train"][keep]
    keep_test = np.isin(data["y_test"], list(_TARGET_CLASSES))
    X_test = [x for x, k in zip(data["X_test"], keep_test) if k]
    y_test = data["y_test"][keep_test]

    clf = cfg["classification"]
    _log(f"[exp1-plot] fitting narrow STA barycenter (2 classes, "
         f"n_train={len(X_train)}, n_steps={sta_n_steps}) …")
    result = _eval_bary('sta', cfg["dataset"]["family"], clf.get("sta_epsilon", 0.05),
                        X_train, y_train, X_test, y_test,
                        gamma=clf["gamma"], lr=clf.get("lr", 1e-2), n_steps=sta_n_steps,
                        optimizer=clf.get("optimizer", "sgd"),
                        patience=clf.get("early_stop_patience", 15),
                        min_rel_improve=clf.get("early_stop_tol", 1e-4),
                        estimator=cfg["classification"].get("estimator", "log_cumulant"),
                        bary_n_jobs=1, return_arrays=True)
    _log(f"[exp1-plot] STA barycenter fit done, f1={result['f1']:.3f}")
    return {"bary": result["bary"], "test_repr": np.asarray(result["test_repr"], dtype=object),
           "test_labels": result["test_labels"]}


def _plot_class(cls: int, cls_name: str, data_by_method: dict, family: str,
                output_path: Path, n_samples: int = 10) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [m for m in _METHOD_ORDER if m in data_by_method]
    fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 3.2), squeeze=False)
    fig.suptitle(f"River class {cls_name} — test samples + barycenter", fontsize=10, fontweight="bold")

    for ax, method in zip(axes[0], methods):
        d = data_by_method[method]
        bary = np.asarray(d["bary"][cls])
        test_repr = d["test_repr"]
        test_labels = np.asarray(d["test_labels"])
        idx = [i for i, l in enumerate(test_labels) if int(l) == cls]
        rng = np.random.RandomState(42 + cls)
        rng.shuffle(idx)
        idx = idx[:n_samples]

        is_raw = method in ('eucl_raw', 'sta')
        invert = (family == 'exponential' and method in ('wasps', 'eucl_params'))

        def _series_val(arr):
            arr = np.asarray(arr)
            if is_raw:
                v = arr if arr.ndim == 1 else np.nanmean(arr, axis=1)
            else:
                v = arr[:, 0]
            return 1.0 / np.maximum(v, 1e-10) if invert else v

        for i in idx:
            ax.semilogy(_series_val(test_repr[i]), color="black", alpha=0.25, linewidth=0.7)
        ax.semilogy(_series_val(bary), color=_BARY_COLORS.get(method, "crimson"), linewidth=2.2,
                   label="Barycenter", zorder=5)
        ax.set_title(method, fontsize=9)
        ax.set_xlabel("t")
    axes[0][0].set_ylabel("λ (discharge)" if family == 'exponential' else "value")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    _log(f"[exp1-plot] wrote {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Experiment 1 river 2-class barycenter+samples figure")
    parser.add_argument("--config", default="configs/exp1_baseline.yaml")
    parser.add_argument("--exp1-out-dir", default="results/jax_exp1_baseline")
    parser.add_argument("--output-dir", default=None,
                        help="defaults to <exp1-out-dir>/results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sta-n-steps", type=int, default=15,
                        help="reduced step count for the narrow illustrative STA fit "
                             "(2 classes only) — not Experiment 1's own n_steps_bary")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    river_cfg = cfg["river"]
    family = river_cfg["dataset"]["family"]
    exp1_out_dir = Path(args.exp1_out_dir)
    output_dir = Path(args.output_dir) if args.output_dir else exp1_out_dir / "results"

    npy_path = exp1_out_dir / "river_bary_data.npy"
    if not npy_path.exists():
        raise FileNotFoundError(
            f"{npy_path} not found — run run_full_baseline.py with "
            f"--bary-plot-dataset river --bary-plot-methods wasps,eucl_params,eucl_raw first")
    data_by_method = np.load(npy_path, allow_pickle=True).item()

    sta_cfg = {
        "dataset": dict(river_cfg["dataset"]),
        "classification": {**river_cfg["classification"], "estimator": cfg.get("estimator", "log_cumulant")},
        "cross_validation": {"n_splits": 1},
        "n_seeds": 1,
    }
    data_by_method["sta"] = _fit_sta_bary(sta_cfg, args.seed, args.sta_n_steps)

    for cls, cls_name in _TARGET_CLASSES.items():
        _plot_class(cls, cls_name, data_by_method, family,
                   output_dir / f"exp1_river_bary_{cls_name}.pdf")


if __name__ == "__main__":
    main()
