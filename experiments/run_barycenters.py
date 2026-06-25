"""Fit per-class barycenters and save them as .npy files for reuse.

Usage:
    python experiments/run_barycenters.py configs/classification.yaml
    python experiments/run_barycenters.py configs/cpazmal.yaml

Saves:
    <output.dir>/barycenters/<method>_class<label>.npy   — (T, n_params) array
    <output.dir>/barycenters/metadata.json               — class names + shapes + config
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import numpy as np
import yaml

_HERE = Path(__file__).parent
_SRC  = _HERE.parent / "src"
sys.path.insert(0, str(_SRC))

jax.config.update("jax_enable_x64", True)  # must precede jax.numpy usage
import distributions
from data.preprocess import clean_time_series, to_fixed_n
from baselines.sta_wrapper import make_cost_fn as sta_cost_fn
from classification.barycenter_clf import fit_barycenters
from costs import SqEuclidean, WaSPS
from softdtw import SoftDTW

# 4-method table — same keys as run_classification.py
_METHODS = {
    'wasps':       {'repr': 'params'},
    'eucl_params': {'repr': 'params'},
    'eucl_raw':    {'repr': 'raw'},
    'sta':         {'repr': 'raw'},
}


def _make_softdtw_bary(method: str, family: str, sta_epsilon: float, gamma: float) -> SoftDTW:
    """SoftDTW instance for barycenter fitting."""
    if method == 'wasps':
        cost_fn = WaSPS(family, use_positivity_constraint=True)
        return SoftDTW(cost_fn, gamma, is_divergence=True, manual_grad=True)
    if method in ('eucl_params', 'eucl_raw'):
        return SoftDTW(SqEuclidean(), gamma, is_divergence=True, manual_grad=False)
    if method == 'sta':
        return SoftDTW(sta_cost_fn(sta_epsilon), gamma, is_divergence=True, manual_grad=False)
    raise ValueError(f"unknown method '{method}'")


def _load_train(cfg: dict) -> tuple:
    """Return (train_raw, train_labels)."""
    ds_type = cfg["dataset"]["type"]
    seed    = cfg.get("seed", 42)
    rng     = np.random.default_rng(seed)

    if ds_type == "synthetic":
        rates   = cfg["dataset"]["rate_params"]
        n_train = cfg["dataset"]["n_train_per_class"]
        T       = cfg["dataset"]["T"]
        N       = cfg["dataset"]["N_samples"]
        train_raw, train_labels = [], []
        for cls_idx, rate in enumerate(rates):
            for _ in range(n_train):
                train_raw.append(rng.exponential(scale=1.0 / rate, size=(T, N)))
                train_labels.append(cls_idx)
        return train_raw, np.array(train_labels)

    if ds_type == "river":
        from data.river_loader import load_river_classification
        ds   = cfg["dataset"]
        clf_cfg = cfg["classification"]
        data = load_river_classification(
            data_dir=ds["data_dir"],
            mode=ds.get("mode", "balanced"),
            test_size=ds.get("test_size", 0.2),
            max_time_steps=ds.get("max_time_steps"),
            samples_per_step=clf_cfg.get("samples_per_step"),
            seed=seed,
        )
        return data["X_train"], np.asarray(data["y_train"])

    if ds_type == "cpazmal":
        from data.cpazmal_loader import MLDatasetLoader, extract_time_series
        import pathlib
        ds        = cfg["dataset"]
        clf_cfg   = cfg["classification"]
        cache_dir = pathlib.Path(ds.get("cache_dir", "data/cpazmal"))
        max_groups = ds.get("max_groups")
        suffix    = "all" if max_groups is None else f"mg{max_groups}"
        cx_train  = cache_dir / f"X_train_{suffix}.npy"
        cy        = cache_dir / f"y_{suffix}.npy"
        if cx_train.exists():
            X_train = list(np.load(cx_train, allow_pickle=True))
            labels  = np.load(cy)
        else:
            loader = MLDatasetLoader(ds["hdf5_path"])
            data   = extract_time_series(loader, max_groups=max_groups)
            X_train = list(data["X_train"])
            labels  = np.asarray(data["y"])
        n = clf_cfg.get("samples_per_step")
        if n is not None:
            rng = np.random.default_rng(seed)
            X_train = [to_fixed_n(s, n, rng) for s in X_train]
        return X_train, labels

    raise ValueError(f"dataset.type '{ds_type}' not supported")


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    family      = cfg["dataset"]["family"]
    clf_cfg     = cfg["classification"]
    out_dir     = Path(cfg["output"]["dir"]) / "barycenters"
    out_dir.mkdir(parents=True, exist_ok=True)
    sta_epsilon = clf_cfg.get("sta_epsilon", 0.05)

    train_raw, train_labels = _load_train(cfg)

    methods_req = [m.lower() for m in cfg.get("methods", list(_METHODS.keys()))]
    methods     = [m for m in methods_req if m in _METHODS]
    unknown     = [m for m in methods_req if m not in _METHODS]
    if unknown:
        print(f"[warn] unknown methods: {unknown} — skipping")

    params_cache: list | None = None  # lazy-fit once for all param-based methods
    metadata = {"config": config_path, "family": family, "methods": {}}

    for method in methods:
        cfg_m = _METHODS[method]

        if cfg_m['repr'] == 'params':
            if params_cache is None:
                print("[params] fitting distribution parameters …", flush=True)
                params_cache = [distributions.get(family).fit_time_series(clean_time_series(s), dtype=np.float64)
                                for s in train_raw]
            train_repr = params_cache
        else:
            train_repr = train_raw

        print(f"[{method}] fitting barycenters …", flush=True)
        softdtw_bary = _make_softdtw_bary(method, family, sta_epsilon, clf_cfg["gamma"])
        bary = fit_barycenters(
            train_repr, train_labels, softdtw_bary,
            n_steps         = clf_cfg["n_steps"],
            lr              = clf_cfg["lr"],
            patience        = clf_cfg.get("early_stop_patience", 15),
            min_rel_improve = clf_cfg.get("early_stop_tol", 1e-4),
        )
        meta = {}
        for cls, arr in bary.items():
            fname = out_dir / f"{method}_class{cls}.npy"
            np.save(fname, arr)
            meta[str(cls)] = {"file": fname.name, "shape": list(arr.shape)}
            print(f"  saved {fname.name}  {arr.shape}", flush=True)
        metadata["methods"][method] = meta

    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[done] metadata saved to {meta_path}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
