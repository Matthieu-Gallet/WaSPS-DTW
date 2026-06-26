"""Nearest-barycenter classifier: argmin SoftDTW divergence to per-class barycenters."""

from __future__ import annotations

import os
import numpy as np
import jax
import jax.numpy as jnp
from joblib import Parallel, delayed

from barycenter import fit_barycenter
from softdtw import SoftDTW


def _fit_one_class(cls, cls_series, softdtw, n_steps, lr, patience, min_rel_improve):
    """Fit a barycenter for one class (worker function for joblib)."""
    return cls, fit_barycenter(cls_series, softdtw, n_steps=n_steps, lr=lr,
                               patience=patience, min_rel_improve=min_rel_improve)


def fit_barycenters(
    train_series: list,
    train_labels: np.ndarray,
    softdtw: SoftDTW,
    n_steps: int = 200,
    lr: float = 1e-2,
    n_jobs: int = 1,
    patience: int = 15,
    min_rel_improve: float = 1e-4,
) -> dict:
    """Fit one SoftDTW barycenter per class.

    Args:
        softdtw:          SoftDTW instance (encodes cost, gamma, divergence, manual_grad).
        n_jobs:           Number of parallel workers for class-level parallelism.
                          Default 1 (sequential). Use -1 for all available cores.
                          Note: joblib loky spawns separate processes, each with own JAX state.
        patience:         Early-stop patience (steps without relative improvement).  0 = off.
        min_rel_improve:  Minimum relative improvement to count as progress.

    Returns:
        barycenters: dict mapping class label → (T, n_params) numpy array.
    """
    labels = np.asarray(train_labels)
    classes = sorted(set(labels.tolist()))

    class_series = {
        cls: [s for s, l in zip(train_series, labels) if l == cls]
        for cls in classes
    }

    if n_jobs == 1:
        # Sequential path — avoids joblib overhead for small n_classes
        return {
            cls: fit_barycenter(class_series[cls], softdtw, n_steps=n_steps, lr=lr,
                                patience=patience, min_rel_improve=min_rel_improve)
            for cls in classes
        }

    # Parallel path: each class in its own process (loky backend, JAX-safe)
    effective_jobs = min(len(classes), os.cpu_count() or 1) if n_jobs == -1 else n_jobs
    results = Parallel(n_jobs=effective_jobs, backend='loky')(
        delayed(_fit_one_class)(cls, class_series[cls], softdtw, n_steps, lr,
                                patience, min_rel_improve)
        for cls in classes
    )
    return dict(results)


def predict(
    test_series: list,
    barycenters: dict,
    cost_fn,
    gamma: float,
) -> np.ndarray:
    """Classify each test series by minimum SoftDTW divergence to class barycenters.

    Args:
        test_series:  List of (T, p) arrays.
        barycenters:  dict label → (T, p) array (from fit_barycenters).
        cost_fn:      Ground cost callable(a, b) → scalar (positive-param space).
        gamma:        SoftDTW regularisation.

    Returns:
        predictions: (N_test,) array of class labels.
    """
    classes = sorted(barycenters.keys())
    bary_jax = {cls: jnp.array(barycenters[cls], dtype=jnp.float64) for cls in classes}

    # Plain SDTW (not divergence): avoids self-term ½SDTW(b,b) dominating when
    # T_test ≠ T_bary, and matches KNN semantics (nearest centroid by DTW distance).
    sdtw_fn = SoftDTW(cost_fn, gamma, is_divergence=False, manual_grad=False)

    @jax.jit
    def divergence_to_bary(z: jax.Array, b: jax.Array) -> jax.Array:
        return sdtw_fn.value(z, b)

    preds = []
    for s in test_series:
        z = jnp.array(s, dtype=jnp.float64)
        dists = [float(divergence_to_bary(z, bary_jax[cls])) for cls in classes]
        preds.append(classes[int(np.argmin(dists))])
    return np.array(preds)
