"""Classification experiment runner: 4-method × 2-mode (KNN + Barycenter) × multi-seed.

Usage:
    python experiments/run_classification.py configs/classification.yaml
    python experiments/run_classification.py configs/cpazmal.yaml
    python experiments/run_classification.py configs/river.yaml

Outputs to <output.dir>/:
  classification_scores.csv      — mean ± std over seeds (F1, accuracy, time)
  classification_full.json       — per-seed raw results + config
  confusion_matrix_knn.pdf       — confusion matrices (last seed, KNN mode)
  confusion_matrix_barycenter.pdf — confusion matrices (last seed, barycenter mode)

Four methods (defined once):

  wasps       — WaSPS W₂² on estimated params  (use_positivity_constraint, manual grad)
  eucl_params — SqEuclidean on estimated params (autodiff)
  eucl_raw    — SqEuclidean on raw samples      (autodiff)
  sta         — OT Sinkhorn on raw samples      (autodiff through Sinkhorn)

Two modes per method:
  knn         — k-NN (k=1) classify by nearest training series
  barycenter  — classify by nearest per-class Fréchet barycenter

Multi-seed: each seed produces an independent train/test split (or independent
synthetic draw).  Reported metrics = mean ± std across seeds.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

_HERE = Path(__file__).parent
_SRC  = _HERE.parent / "src"
sys.path.insert(0, str(_SRC))

jax.config.update("jax_enable_x64", True)  # must precede jax.numpy usage
import distributions
from data.preprocess import clean_time_series, to_fixed_n
from baselines.sta_wrapper import knn_predict as sta_knn, make_cost_fn as sta_cost_fn
from classification.barycenter_clf import fit_barycenters, predict
from classification.nn import knn_predict as sdtw_knn
from costs import SqEuclidean, WaSPS
from softdtw import SoftDTW


# ---------------------------------------------------------------------------
# Method table — one entry per method, referenced everywhere
# ---------------------------------------------------------------------------

# repr:        'params' → estimate distribution parameters first
#              'raw'    → use raw sample arrays directly
_METHODS = {
    'wasps':       {'repr': 'params'},
    'eucl_params': {'repr': 'params'},
    'eucl_raw':    {'repr': 'raw'},
    'sta':         {'repr': 'raw'},
}


def _make_cost_fn(method: str, family: str, sta_epsilon: float):
    """Cost function for KNN and predict (data in positive-param space, no θ conversion)."""
    if method == 'wasps':
        return WaSPS(family, log_correction=True)
    if method in ('eucl_params', 'eucl_raw'):
        return SqEuclidean()
    if method == 'sta':
        return sta_cost_fn(sta_epsilon)
    raise ValueError(f"unknown method '{method}'")


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


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _make_synthetic(cfg: dict, rng: np.random.Generator):
    family = cfg["dataset"]["family"]
    if family != "exponential":
        raise ValueError("synthetic dataset only supports family=exponential")
    rates   = cfg["dataset"]["rate_params"]
    n_train = cfg["dataset"]["n_train_per_class"]
    n_test  = cfg["dataset"]["n_test_per_class"]
    T, N    = cfg["dataset"]["T"], cfg["dataset"]["N_samples"]
    train_raw, test_raw, train_labels, test_labels = [], [], [], []
    for cls_idx, rate in enumerate(rates):
        for _ in range(n_train):
            train_raw.append(rng.exponential(1.0 / rate, (T, N)))
            train_labels.append(cls_idx)
        for _ in range(n_test):
            test_raw.append(rng.exponential(1.0 / rate, (T, N)))
            test_labels.append(cls_idx)
    return train_raw, test_raw, np.array(train_labels), np.array(test_labels)


def _load_river(cfg: dict, seed: int):
    from data.river_loader import load_river_classification
    ds   = cfg["dataset"]
    data = load_river_classification(
        data_dir=ds["data_dir"],
        mode=ds.get("mode", "balanced"),
        test_size=ds.get("test_size", 0.2),
        max_time_steps=ds.get("max_time_steps"),
        samples_per_step=cfg["classification"].get("samples_per_step"),
        seed=seed,
    )
    return (
        data["X_train"], data["X_test"],
        np.asarray(data["y_train"]), np.asarray(data["y_test"]),
    )


def _load_cpazmal(cfg: dict, seed: int = 42):
    from data.cpazmal_loader import MLDatasetLoader, extract_time_series
    ds        = cfg["dataset"]
    hdf5_path = ds["hdf5_path"]
    max_groups= ds.get("max_groups")
    cache_dir = Path(ds.get("cache_dir", "data/cpazmal"))
    suffix    = "all" if max_groups is None else f"mg{max_groups}"
    cx_train  = cache_dir / f"X_train_{suffix}.npy"
    cx_pred   = cache_dir / f"X_predict_{suffix}.npy"
    cy        = cache_dir / f"y_{suffix}.npy"
    if cx_train.exists():
        print(f"[cpazmal] loading from cache (max_groups={max_groups})")
        X_train = list(np.load(cx_train,  allow_pickle=True))
        X_test  = list(np.load(cx_pred,   allow_pickle=True))
        labels  = np.load(cy)
    else:
        print(f"[cpazmal] extracting from HDF5 (max_groups={max_groups}) …")
        loader = MLDatasetLoader(hdf5_path)
        data   = extract_time_series(loader, max_groups=max_groups)
        X_train = list(data["X_train"])
        X_test  = list(data["X_predict"])
        labels  = np.asarray(data["y"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cx_train, np.array(X_train, dtype=object), allow_pickle=True)
        np.save(cx_pred,  np.array(X_test,  dtype=object), allow_pickle=True)
        np.save(cy, labels)
        print(f"[cpazmal] cache saved to {cache_dir}")
    # Rectangularise raw arrays if samples_per_step is set
    n = cfg["classification"].get("samples_per_step")
    if n is not None:
        rng = np.random.default_rng(seed)
        X_train = [to_fixed_n(s, n, rng) for s in X_train]
        X_test  = [to_fixed_n(s, n, rng) for s in X_test]
    return X_train, X_test, labels, labels


# ---------------------------------------------------------------------------
# Subsample helper
# ---------------------------------------------------------------------------

def _subsample(X, y, max_n, rng):
    if max_n < 0 or max_n >= len(y):
        return X, y
    idx, _ = train_test_split(
        np.arange(len(y)), train_size=max_n,
        random_state=int(rng.integers(2**31)), stratify=y,
    )
    return [X[i] for i in np.sort(idx)], y[np.sort(idx)]


# ---------------------------------------------------------------------------
# Representation builder
# ---------------------------------------------------------------------------

def _build_repr(raw_series: list, repr_type: str, family: str) -> list:
    """params → estimate distribution parameters; raw → return as-is."""
    if repr_type == 'params':
        return [distributions.get(family).fit_time_series(clean_time_series(s), dtype=np.float64)
                for s in raw_series]
    return raw_series


# ---------------------------------------------------------------------------
# Metric helper
# ---------------------------------------------------------------------------

def _metrics(preds, truth):
    classes = sorted(set(truth.tolist()))
    return {
        "accuracy":         float(accuracy_score(truth, preds)),
        "f1_weighted":      float(f1_score(truth, preds, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(truth, preds, labels=classes).tolist(),
        "classes":          classes,
    }


# ---------------------------------------------------------------------------
# Single-seed runner
# ---------------------------------------------------------------------------

def _run_one_seed(cfg: dict, seed: int, methods: list) -> list:
    family    = cfg["dataset"]["family"]
    clf_cfg   = cfg["classification"]
    ds_type   = cfg["dataset"]["type"]
    rng       = np.random.default_rng(seed)

    if ds_type == "synthetic":
        train_raw, test_raw, train_labels, test_labels = _make_synthetic(cfg, rng)
    elif ds_type == "river":
        train_raw, test_raw, train_labels, test_labels = _load_river(cfg, seed)
    elif ds_type == "cpazmal":
        train_raw, test_raw, train_labels, test_labels = _load_cpazmal(cfg, seed=seed)
    else:
        raise ValueError(f"dataset.type '{ds_type}' not supported")

    train_raw, train_labels = _subsample(train_raw, train_labels,
                                          clf_cfg.get("max_train_samples", -1), rng)
    test_raw, test_labels   = _subsample(test_raw,  test_labels,
                                          clf_cfg.get("max_test_samples",  -1), rng)

    gamma       = clf_cfg["gamma"]
    k           = clf_cfg.get("k", 1)
    n_steps     = clf_cfg["n_steps"]
    lr          = clf_cfg["lr"]
    sta_epsilon = clf_cfg.get("sta_epsilon", 0.05)
    modes       = [m.lower() for m in cfg.get("modes", ["knn", "barycenter"])]

    results      = []
    _repr_cache: dict = {}

    def _get_repr(repr_type, split):
        if (repr_type, split) not in _repr_cache:
            raw = train_raw if split == 'train' else test_raw
            _repr_cache[(repr_type, split)] = _build_repr(raw, repr_type, family)
        return _repr_cache[(repr_type, split)]

    for method in methods:
        cfg_m     = _METHODS[method]
        repr_type = cfg_m['repr']
        cost_fn   = _make_cost_fn(method, family, sta_epsilon)
        train_repr = _get_repr(repr_type, 'train')
        test_repr  = _get_repr(repr_type, 'test')

        if 'knn' in modes:
            t0 = time.time()
            if method == 'sta':
                preds = sta_knn(train_raw, train_labels, test_raw,
                                gamma=gamma, epsilon=sta_epsilon, k=k)
            else:
                preds = sdtw_knn(train_repr, train_labels, test_repr,
                                 cost_fn=cost_fn, gamma=gamma, k=k)
            elapsed = time.time() - t0
            res = _metrics(preds, test_labels)
            res.update({"method": method, "mode": "knn",
                        "seed": seed, "time_s": elapsed})
            results.append(res)
            print(f"  [{method}/knn] acc={res['accuracy']:.3f}  "
                  f"f1={res['f1_weighted']:.3f}  t={elapsed:.1f}s", flush=True)

        if 'barycenter' in modes:
            t0 = time.time()
            softdtw_bary = _make_softdtw_bary(method, family, sta_epsilon, gamma)
            barycenters = fit_barycenters(
                train_repr, train_labels, softdtw_bary,
                n_steps=n_steps, lr=lr,
                patience=clf_cfg.get("early_stop_patience", 15),
                min_rel_improve=clf_cfg.get("early_stop_tol", 1e-4),
            )
            train_time = time.time() - t0
            t1 = time.time()
            preds = predict(test_repr, barycenters, cost_fn, gamma)
            infer_time = time.time() - t1
            res = _metrics(preds, test_labels)
            res.update({"method": method, "mode": "barycenter",
                        "seed": seed, "time_s": train_time + infer_time,
                        "train_time_s": train_time, "infer_time_s": infer_time})
            results.append(res)
            print(f"  [{method}/bary] acc={res['accuracy']:.3f}  "
                  f"f1={res['f1_weighted']:.3f}  t_train={train_time:.1f}s", flush=True)

    return results


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def _save_confusion_matrices(all_results: list, out_dir: Path) -> None:
    """Save one confusion-matrix PDF per mode, using the last seed's results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from collections import defaultdict

    by_mode: dict = defaultdict(list)
    for r in all_results:
        by_mode[r['mode']].append(r)

    for mode, rows in by_mode.items():
        last_seed = max(r['seed'] for r in rows)
        last_rows = sorted(
            [r for r in rows if r['seed'] == last_seed],
            key=lambda r: r['method'],
        )
        if not last_rows:
            continue
        classes = last_rows[0]['classes']
        n_cls   = len(classes)
        n_m     = len(last_rows)

        fig, axes = plt.subplots(1, n_m, figsize=(3.0 * n_m, 3.5), squeeze=False)
        axes_flat = axes[0]

        for ax, row in zip(axes_flat, last_rows):
            cm      = np.array(row['confusion_matrix'])
            cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
            ax.set_xticks(range(n_cls))
            ax.set_yticks(range(n_cls))
            ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=7)
            ax.set_yticklabels(classes, fontsize=7)
            ax.set_title(f"{row['method']}\nf1={row['f1_weighted']:.3f}", fontsize=8)
            ax.set_xlabel('Predicted', fontsize=7)
            for i in range(n_cls):
                for j in range(n_cls):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                            fontsize=7,
                            color='white' if cm_norm[i, j] > 0.5 else 'black')
        axes_flat[0].set_ylabel('True', fontsize=7)

        plt.tight_layout()
        fname = out_dir / f"confusion_matrix_{mode}.pdf"
        fig.savefig(fname, bbox_inches='tight')
        plt.close(fig)
        print(f"  figure: {fname}")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(all_results: list) -> list:
    from collections import defaultdict
    buckets: dict = defaultdict(list)
    for r in all_results:
        buckets[(r["method"], r["mode"])].append(r)
    summary = []
    for (method, mode), rows in sorted(buckets.items()):
        acc  = [r["accuracy"]    for r in rows]
        f1   = [r["f1_weighted"] for r in rows]
        time = [r["time_s"]      for r in rows]
        summary.append({
            "method": method, "mode": mode, "n_seeds": len(rows),
            "acc_mean":  float(np.mean(acc)),   "acc_std":  float(np.std(acc)),
            "f1_mean":   float(np.mean(f1)),    "f1_std":   float(np.std(f1)),
            "time_mean": float(np.mean(time)),  "time_std": float(np.std(time)),
        })
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    base_seed = cfg.get("seed", 42)
    n_seeds   = cfg.get("n_seeds", 1)
    seeds     = [base_seed + i for i in range(n_seeds)]
    family    = cfg["dataset"]["family"]

    methods_req = [m.lower() for m in cfg.get("methods", list(_METHODS.keys()))]
    methods = [m for m in methods_req if m in _METHODS]
    unknown = [m for m in methods_req if m not in _METHODS]
    if unknown:
        print(f"[warn] unknown methods: {unknown} — skipping")

    print(f"[run] dataset={cfg['dataset']['type']}  family={family}  "
          f"methods={methods}  seeds={seeds}", flush=True)

    all_results = []
    for seed in seeds:
        print(f"\n[seed={seed}]", flush=True)
        all_results.extend(_run_one_seed(cfg, seed, methods))

    summary = _aggregate(all_results)

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "classification_scores.csv"
    with open(csv_path, "w", newline="") as f:
        fields = ["method", "mode", "n_seeds",
                  "acc_mean", "acc_std", "f1_mean", "f1_std",
                  "time_mean", "time_std"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary)

    json_path = out_dir / "classification_full.json"
    with open(json_path, "w") as f:
        json.dump({"config": cfg, "summary": summary, "per_seed": all_results}, f, indent=2)

    _save_confusion_matrices(all_results, out_dir)

    print(f"\n[done] results saved to {out_dir}")
    for row in summary:
        print(f"  {row['method']:12s}/{row['mode']:10s}  "
              f"f1={row['f1_mean']:.3f}±{row['f1_std']:.3f}  "
              f"acc={row['acc_mean']:.3f}±{row['acc_std']:.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
