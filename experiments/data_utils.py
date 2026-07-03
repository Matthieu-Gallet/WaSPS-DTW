"""Shared dataset loading and preprocessing utilities for experiment scripts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

_HERE = Path(__file__).parent
_SRC  = _HERE.parent / "src"
sys.path.insert(0, str(_SRC))

import distributions
from data.preprocess import clean_time_series, to_fixed_n


# ---------------------------------------------------------------------------
# Representation builder
# ---------------------------------------------------------------------------

def build_repr(
    raw_series: list,
    labels: np.ndarray,
    repr_type: str,
    family: str,
) -> tuple[list, np.ndarray]:
    """Build method representation from raw series.

    For 'params': estimates distribution parameters via MLE. Series whose
    fit_time_series result contains any NaN (e.g. timesteps with no valid
    samples after clean_time_series) are excluded; their labels are dropped too.
    For 'raw': returns series as-is (NaN already handled by to_fixed_n in loader).

    Returns:
        (repr_list, filtered_labels) — filtered_labels aligns with repr_list.
        For 'raw', filtered_labels == labels (unchanged).
    """
    labels = np.asarray(labels)
    if repr_type == 'params':
        dist = distributions.get(family)
        all_params = [
            dist.fit_time_series(clean_time_series(s), dtype=np.float64)
            for s in raw_series
        ]
        valid = np.array([not np.isnan(p).any() for p in all_params])
        n_dropped = int((~valid).sum())
        if n_dropped > 0:
            print(f"[warn] build_repr: dropped {n_dropped}/{len(raw_series)} series "
                  f"with NaN params (timesteps where MLE failed)")
        return [p for p, v in zip(all_params, valid) if v], labels[valid]
    # raw path — NaN handled upstream by to_fixed_n; return as-is
    return raw_series, labels


# ---------------------------------------------------------------------------
# Raw downsampling helper (point 4)
# ---------------------------------------------------------------------------

def compute_raw_n_min(raw_series: list) -> int:
    """Minimum count of finite, positive values across all series × timesteps.

    Use this to choose the downsampling target N for the raw path when
    samples_per_step is not set explicitly.  Computed on a single split
    (train OR test) to avoid information leakage.
    """
    counts = [
        int(np.sum(np.isfinite(s[t]) & (np.asarray(s[t]) > 0)))
        for s in raw_series
        for t in range(np.asarray(s).shape[0])
    ]
    return max(1, min(counts))


# ---------------------------------------------------------------------------
# Subsample helper
# ---------------------------------------------------------------------------

def subsample(X, y, max_n: int, rng: np.random.Generator, extra=None):
    """Stratified subsample of (X, y) to at most max_n samples.

    Args:
        extra: optional 1-D numpy array of same length as y (e.g. group indices).

    Returns:
        (X_sub, y_sub) or (X_sub, y_sub, extra_sub) when extra is not None.
    """
    if max_n < 0 or max_n >= len(y):
        return (X, y, extra) if extra is not None else (X, y)
    idx, _ = train_test_split(
        np.arange(len(y)), train_size=max_n,
        random_state=int(rng.integers(2**31)), stratify=y,
    )
    sorted_idx = np.sort(idx)
    sub_extra = extra[sorted_idx] if extra is not None else None
    if extra is not None:
        return [X[i] for i in sorted_idx], y[sorted_idx], sub_extra
    return [X[i] for i in sorted_idx], y[sorted_idx]


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def metrics(preds: np.ndarray, truth: np.ndarray) -> dict:
    classes = sorted(set(truth.tolist()))
    return {
        "accuracy":         float(accuracy_score(truth, preds)),
        "f1_weighted":      float(f1_score(truth, preds, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(truth, preds, labels=classes).tolist(),
        "classes":          classes,
    }


# ---------------------------------------------------------------------------
# Dataset loader dispatcher
# ---------------------------------------------------------------------------

def load_dataset(cfg: dict, seed: int, fold: Optional[int] = None) -> dict:
    """Load and split dataset from config dict.

    Args:
        cfg:  Full experiment config.
        seed: Random seed for the split / subsampling.
        fold: Override the fold index from cfg (0-indexed).  None = use cfg.

    Returns:
        dict with keys:
          X_train, X_test   — lists of (T, N) arrays.
          y_train, y_test   — (N_train,) / (N_test,) int arrays.
          class_names       — {int: str}.
          metadata          — raw metadata dict.
          groups_train      — (N_train,) int group array or None.
          groups_test       — (N_test,)  int group array or None.
    """
    ds_type = cfg["dataset"]["type"]
    if ds_type == "synthetic":
        return _load_synthetic(cfg, seed)
    if ds_type == "river":
        return _load_river(cfg, seed, fold=fold)
    if ds_type == "cpazmal":
        return _load_cpazmal(cfg, seed, fold=fold)
    raise ValueError(f"dataset.type '{ds_type}' not supported")


# ---------------------------------------------------------------------------
# Private loaders
# ---------------------------------------------------------------------------

def _load_synthetic(cfg: dict, seed: int) -> dict:
    family = cfg["dataset"]["family"]
    if family != "exponential":
        raise ValueError("synthetic dataset only supports family=exponential")
    rates   = cfg["dataset"]["rate_params"]
    n_train = cfg["dataset"]["n_train_per_class"]
    n_test  = cfg["dataset"]["n_test_per_class"]
    T, N    = cfg["dataset"]["T"], cfg["dataset"]["N_samples"]
    rng = np.random.default_rng(seed)
    train_raw, test_raw, train_labels, test_labels = [], [], [], []
    for cls_idx, rate in enumerate(rates):
        for _ in range(n_train):
            train_raw.append(rng.exponential(1.0 / rate, (T, N)))
            train_labels.append(cls_idx)
        for _ in range(n_test):
            test_raw.append(rng.exponential(1.0 / rate, (T, N)))
            test_labels.append(cls_idx)
    return {
        "X_train":     train_raw,
        "X_test":      test_raw,
        "y_train":     np.array(train_labels),
        "y_test":      np.array(test_labels),
        "class_names": {i: str(r) for i, r in enumerate(rates)},
        "metadata":    {},
        "groups_train": None,
        "groups_test":  None,
    }


def _load_river(cfg: dict, seed: int, fold: Optional[int] = None) -> dict:
    from data.river_loader import load_river_classification
    ds      = cfg["dataset"]
    clf_cfg = cfg["classification"]
    cv      = cfg.get("cross_validation", {})

    n_splits    = int(cv.get("n_splits", 1))
    fold_idx    = fold if fold is not None else int(cv.get("fold", 0))
    group_aware = bool(cv.get("group_aware", False))

    data = load_river_classification(
        data_dir         = ds["data_dir"],
        mode             = ds.get("mode", "balanced"),
        n_splits         = n_splits,
        fold             = fold_idx,
        group_aware      = group_aware,
        test_size        = ds.get("test_size", 0.2),
        max_time_steps   = ds.get("max_time_steps"),
        samples_per_step = clf_cfg.get("samples_per_step"),
        aggregate_days   = ds.get("aggregate_days"),
        seed             = seed,
    )
    return {
        "X_train":     data["X_train"],
        "X_test":      data["X_test"],
        "y_train":     np.asarray(data["y_train"]),
        "y_test":      np.asarray(data["y_test"]),
        "class_names": data["class_names"],
        "metadata":    data["metadata"],
        "groups_train": data.get("groups_train"),
        "groups_test":  data.get("groups_test"),
    }


def _load_cpazmal(cfg: dict, seed: int = 42, fold: Optional[int] = None) -> dict:
    from data.cpazmal_loader import MLDatasetLoader, extract_time_series
    ds        = cfg["dataset"]
    hdf5_path = ds["hdf5_path"]
    max_groups = ds.get("max_groups")
    cache_dir  = Path(ds.get("cache_dir", "data/cpazmal"))
    suffix     = "all" if max_groups is None else f"mg{max_groups}"
    cx_train   = cache_dir / f"X_train_{suffix}.npy"
    cx_pred    = cache_dir / f"X_predict_{suffix}.npy"
    cy         = cache_dir / f"y_{suffix}.npy"
    cg         = cache_dir / f"groups_{suffix}.npy"
    clf_cfg    = cfg["classification"]

    groups = None
    if cx_train.exists():
        print(f"[cpazmal] loading from cache (max_groups={max_groups})")
        X_train = list(np.load(cx_train, allow_pickle=True))
        X_test  = list(np.load(cx_pred,  allow_pickle=True))
        labels  = np.load(cy)
        if cg.exists():
            groups = np.load(cg)
    else:
        print(f"[cpazmal] extracting from HDF5 (max_groups={max_groups}) …")
        loader = MLDatasetLoader(hdf5_path)
        data   = extract_time_series(loader, max_groups=max_groups)
        X_train = list(data["X_train"])
        X_test  = list(data["X_predict"])
        labels  = np.asarray(data["y"])
        groups  = np.asarray(data["groups"]) if "groups" in data else None
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cx_train, np.array(X_train, dtype=object), allow_pickle=True)
        np.save(cx_pred,  np.array(X_test,  dtype=object), allow_pickle=True)
        np.save(cy, labels)
        if groups is not None:
            np.save(cg, groups)
        print(f"[cpazmal] cache saved to {cache_dir}")

    n = clf_cfg.get("samples_per_step")
    if n is not None:
        rng = np.random.default_rng(seed)
        X_train = [to_fixed_n(s, n, rng) for s in X_train]
        X_test  = [to_fixed_n(s, n, rng) for s in X_test]

    cv       = cfg.get("cross_validation", {})
    n_splits = int(cv.get("n_splits", 1))

    if n_splits > 1 and groups is not None:
        # K-fold on CPAZMaL: concatenate both time periods, then GroupKFold by
        # geographic group so all windows from one site go to the same fold.
        fold_idx = fold if fold is not None else int(cv.get("fold", 0))
        from sklearn.model_selection import GroupKFold
        X_all  = X_train + X_test
        y_all  = np.concatenate([labels, labels])
        g_all  = np.concatenate([groups, groups])
        splits = list(GroupKFold(n_splits=n_splits).split(np.arange(len(y_all)), y_all, g_all))
        if fold_idx >= len(splits):
            raise ValueError(f"fold={fold_idx} out of range (n_splits={n_splits})")
        idx_tr, idx_te = splits[fold_idx]
        return {
            "X_train":     [X_all[i] for i in idx_tr],
            "X_test":      [X_all[i] for i in idx_te],
            "y_train":     y_all[idx_tr],
            "y_test":      y_all[idx_te],
            "class_names": {},
            "metadata":    {},
            "groups_train": g_all[idx_tr],
            "groups_test":  g_all[idx_te],
        }

    return {
        "X_train":     X_train,
        "X_test":      X_test,
        "y_train":     labels,
        "y_test":      labels,
        "class_names": {},
        "metadata":    {},
        "groups_train": groups,
        "groups_test":  groups,
    }
