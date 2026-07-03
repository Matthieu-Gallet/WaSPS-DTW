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

from classification.barycenter_clf import fit_barycenters

from data_utils import build_repr, load_dataset
from method_defs import _METHODS, make_softdtw_bary as _make_softdtw_bary


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    family      = cfg["dataset"]["family"]
    clf_cfg     = cfg["classification"]
    seed        = cfg.get("seed", 42)
    out_dir     = Path(cfg["output"]["dir"]) / "barycenters"
    out_dir.mkdir(parents=True, exist_ok=True)
    sta_epsilon = clf_cfg.get("sta_epsilon", 0.05)

    data = load_dataset(cfg, seed)
    train_raw    = data["X_train"]
    train_labels = data["y_train"]

    methods_req = [m.lower() for m in cfg.get("methods", list(_METHODS.keys()))]
    methods     = [m for m in methods_req if m in _METHODS]
    unknown     = [m for m in methods_req if m not in _METHODS]
    if unknown:
        print(f"[warn] unknown methods: {unknown} — skipping")

    params_cache:     list | None = None
    params_labels:    np.ndarray | None = None

    metadata = {"config": config_path, "family": family, "methods": {}}

    for method in methods:
        repr_type = _METHODS[method]['repr']

        if repr_type == 'params':
            if params_cache is None:
                print("[params] fitting distribution parameters …", flush=True)
                params_cache, params_labels = build_repr(train_raw, train_labels, 'params', family)
            train_repr   = params_cache
            train_repr_labels = params_labels
        else:
            train_repr        = train_raw
            train_repr_labels = train_labels

        print(f"[{method}] fitting barycenters …", flush=True)
        softdtw_bary = _make_softdtw_bary(method, family, sta_epsilon, clf_cfg["gamma"])
        bary = fit_barycenters(
            train_repr, train_repr_labels, softdtw_bary,
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
