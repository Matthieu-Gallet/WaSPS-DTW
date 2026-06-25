"""Loader for the prebuilt river-discharge classification dataset.

The dataset (X / Y / metadata .npy files) is built **once** from the raw
NetCDF and stored in ``data/river/``.  No estimation is performed here —
callers call ``distributions.get(family).fit_time_series(clean_time_series(s))``
after loading (see experiments/run_classification.py and analysis/river_notebook.ipynb).

Reshape convention: each raw sample is ``(T, D, W, W)``; we flatten the
spatial/temporal dimensions to ``(T, D·W·W)`` float64 **with NaN preserved**.
``clean_time_series`` (from data.preprocess) handles NaN / non-positive
filtering per timestep before estimation, exactly as for CPAZMaL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.model_selection import train_test_split


def load_river_classification(
    data_dir: str,
    mode: str = "balanced",
    test_size: float = 0.2,
    max_time_steps: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """Load prebuilt river-discharge classification .npy and split stratified.

    The ``data/river/`` .npy files must exist (build them with
    ``python src/build_dataset.py --mode balanced --output-dir data/river``
    in the Explore2_HydroDataset project).

    Unlike CPAZMaL, river has no separate *predict* period.  The stratified
    train/test split is performed here so that the harness receives genuinely
    disjoint sets.

    Args:
        data_dir:        Directory containing ``X_{mode}.npy``,
                         ``Y_{mode}.npy``, ``metadata_{mode}.npy``.
        mode:            ``"balanced"`` (400 samples, 4 populated classes) or
                         ``"basic"`` (611 samples, all stations).
        test_size:       Fraction of samples held out for testing.
        max_time_steps:  Truncate T to this value (smoke tests only).
        seed:            Random seed for the stratified split.

    Returns:
        dict with keys:

        - ``X_train``    — list of ``(T, D·W·W)`` float64 arrays, NaN preserved.
        - ``X_test``     — list of ``(T, D·W·W)`` float64 arrays, NaN preserved.
        - ``y_train``    — ``(N_train,)`` int array.
        - ``y_test``     — ``(N_test,)``  int array.
        - ``class_names``— ``{int: str}`` from ``metadata['idx_to_regime']``.
        - ``metadata``   — full metadata dict.
    """
    d = Path(data_dir)
    X_path  = d / f"X_{mode}.npy"
    Y_path  = d / f"Y_{mode}.npy"
    md_path = d / f"metadata_{mode}.npy"

    if not X_path.exists():
        raise FileNotFoundError(
            f"{X_path} not found.\n"
            "Build the dataset first with the Explore2_HydroDataset project:\n"
            "  python src/build_dataset.py"
            f" --mode {mode} --output-dir {data_dir}"
        )

    X        = np.load(X_path)                                # (N, T, D, W, W)
    Y        = np.load(Y_path)                                # (N,)
    metadata = np.load(md_path, allow_pickle=True).item()

    # Optional time truncation (smoke tests)
    T = X.shape[1]
    if max_time_steps is not None:
        T = min(T, max_time_steps)

    # Reshape (T, D, W, W) → (T, D·W·W) float64 per sample; preserve NaN
    samples = [X[i, :T].reshape(T, -1).astype(np.float64) for i in range(len(X))]

    # Stratified split — river has no separate predict period
    idx_train, idx_test = train_test_split(
        np.arange(len(Y)),
        test_size=test_size,
        random_state=seed,
        stratify=Y,
    )
    idx_train = np.sort(idx_train)
    idx_test  = np.sort(idx_test)

    return {
        "X_train":     [samples[i] for i in idx_train],
        "X_test":      [samples[i] for i in idx_test],
        "y_train":     Y[idx_train],
        "y_test":      Y[idx_test],
        "class_names": metadata.get("idx_to_regime", {}),
        "metadata":    metadata,
    }
