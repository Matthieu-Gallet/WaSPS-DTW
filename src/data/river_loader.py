"""Loader for the prebuilt river-discharge classification dataset.

The dataset (.npy files) is built **once** from the raw NetCDF and stored in
``data/river/`` (legacy) or ``data/river_new/output_{2,3,4}/`` (with groups).
No estimation is performed here — callers call
``distributions.get(family).fit_time_series(clean_time_series(s))`` after loading.

Reshape convention: each raw sample is ``(T, D, W, W)``; we flatten the spatial
dimensions to ``(T, D·W·W)`` float64.  The raw arrays contain ~14.5 % NaN.
Passing ``samples_per_step`` rectangularises each sample to ``(T, samples_per_step)``
via ``to_fixed_n`` — required for eucl_raw and STA methods.

K-fold support:
  - ``n_splits=1`` (default): stratified holdout with ``test_size`` (legacy behaviour).
  - ``n_splits>1, group_aware=False``: ``StratifiedKFold`` — fold ``fold``.
  - ``n_splits>1, group_aware=True``: ``StratifiedGroupKFold`` using the
    ``groups_balanced.npy`` from ``data/river_new/output_N/``.

river_new suffix:
  ``output_2``, ``output_3``, ``output_4`` correspond to spatial window sizes 2, 3, 4.
  Each contains ``groups_balanced.npy`` in addition to the standard files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)

from data.preprocess import to_fixed_n


def _aggregate_days(arr_2d: np.ndarray, k: int) -> np.ndarray:
    """Pool k consecutive timesteps into one by concatenating sample columns.

    Args:
        arr_2d: (T, M) float64 array (may contain NaN).
        k:      Number of days to pool per output timestep.

    Returns:
        (T//k, k*M) array — trailing timesteps are discarded.
    """
    if k <= 1:
        return arr_2d
    T_new = arr_2d.shape[0] // k
    return arr_2d[:T_new * k].reshape(T_new, k, -1).reshape(T_new, -1)


def load_river_classification(
    data_dir: str,
    mode: str = "balanced",
    n_splits: int = 1,
    fold: int = 0,
    group_aware: bool = False,
    test_size: float = 0.2,
    max_time_steps: Optional[int] = None,
    samples_per_step: Optional[int] = None,
    aggregate_days: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """Load prebuilt river-discharge classification .npy and produce a train/test split.

    Works with both ``data/river/`` (legacy, no groups) and
    ``data/river_new/output_N/`` (with ``groups_balanced.npy``).

    Args:
        data_dir:          Directory containing ``X_{mode}.npy``, ``Y_{mode}.npy``,
                           ``metadata_{mode}.npy`` and optionally ``groups_{mode}.npy``.
        mode:              ``"balanced"`` (400 samples, T=365, 4 classes)
                           or ``"basic"`` (611 samples, T=52 weekly).
        n_splits:          Number of folds.  1 = stratified holdout (legacy).
        fold:              Fold index (0..n_splits-1).  Ignored when n_splits=1.
        group_aware:       If True and groups are available, use
                           ``StratifiedGroupKFold`` to keep groups disjoint across
                           folds.  Requires ``groups_{mode}.npy`` in data_dir.
        test_size:         Held-out fraction when n_splits=1.
        max_time_steps:    Truncate T to this value (smoke tests only).
        samples_per_step:  Target samples per timestep after rectangularisation via
                           ``to_fixed_n``.  Required for eucl_raw and STA.
                           ``None`` preserves NaN (params path only).
        aggregate_days:    Pool k consecutive timesteps into one before rectangularisation.
        seed:              Random seed for the stratified split / to_fixed_n.

    Returns:
        dict with keys:

        - ``X_train``     — list of (T, N) float64 arrays.
        - ``X_test``      — list of (T, N) float64 arrays.
        - ``y_train``     — (N_train,) int array.
        - ``y_test``      — (N_test,)  int array.
        - ``class_names`` — {int: str} from metadata.
        - ``metadata``    — full metadata dict.
        - ``groups_train``— (N_train,) int group array or None.
        - ``groups_test`` — (N_test,)  int group array or None.
    """
    d = Path(data_dir)
    X_path  = d / f"X_{mode}.npy"
    Y_path  = d / f"Y_{mode}.npy"
    md_path = d / f"metadata_{mode}.npy"
    g_path  = d / f"groups_{mode}.npy"

    if not X_path.exists():
        raise FileNotFoundError(
            f"{X_path} not found.\n"
            "Build the dataset with the Explore2_HydroDataset project:\n"
            f"  python src/build_dataset.py --mode {mode} --output-dir {data_dir}"
        )

    X        = np.load(X_path)                              # (N, T, D, W, W) or (N, T, M)
    Y        = np.load(Y_path)                              # (N,)
    metadata = np.load(md_path, allow_pickle=True).item()
    groups   = np.load(g_path) if g_path.exists() else None  # (N,) int or None

    # Time axis: handle both (N, T, D, W, W) and (N, T, M) shapes
    T_raw = X.shape[1]
    T = min(T_raw, max_time_steps) if max_time_steps is not None else T_raw

    rng = np.random.default_rng(seed)
    k   = aggregate_days if (aggregate_days and aggregate_days > 1) else 1

    # Flatten spatial dims → (T, M) per sample, then optional aggregation + rectangularisation
    if X.ndim > 3:
        flat = [X[i, :T].reshape(T, -1).astype(np.float64) for i in range(len(X))]
    else:
        flat = [X[i, :T].astype(np.float64) for i in range(len(X))]

    if samples_per_step is not None:
        samples = [to_fixed_n(_aggregate_days(s, k), samples_per_step, rng) for s in flat]
    else:
        samples = [_aggregate_days(s, k) for s in flat]

    # ---------------------------------------------------------------------------
    # Split
    # ---------------------------------------------------------------------------
    if n_splits == 1:
        idx_train, idx_test = train_test_split(
            np.arange(len(Y)), test_size=test_size, random_state=seed, stratify=Y,
        )
        idx_train = np.sort(idx_train)
        idx_test  = np.sort(idx_test)
    else:
        if fold >= n_splits:
            raise ValueError(f"fold={fold} out of range (n_splits={n_splits})")
        if group_aware and groups is not None:
            splitter = StratifiedGroupKFold(n_splits=n_splits)
            splits   = list(splitter.split(np.arange(len(Y)), Y, groups))
        else:
            if group_aware:
                print("[warn] group_aware=True but no groups file found — "
                      "falling back to StratifiedKFold")
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            splits   = list(splitter.split(np.arange(len(Y)), Y))
        idx_train, idx_test = splits[fold]
        idx_train = np.sort(idx_train)
        idx_test  = np.sort(idx_test)

    return {
        "X_train":     [samples[i] for i in idx_train],
        "X_test":      [samples[i] for i in idx_test],
        "y_train":     Y[idx_train],
        "y_test":      Y[idx_test],
        "class_names": metadata.get("idx_to_regime", {}),
        "metadata":    metadata,
        "groups_train": groups[idx_train] if groups is not None else None,
        "groups_test":  groups[idx_test]  if groups is not None else None,
    }
