"""k-NN classifier using SoftDTW distances, vectorised with vmap over training set."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from softdtw import SoftDTW


def knn_predict(
    train_series: list,
    train_labels: np.ndarray,
    test_series: list,
    cost_fn,
    gamma: float,
    k: int = 1,
    dtype = jnp.float32,
    is_divergence: bool = False,
) -> np.ndarray:
    """k-NN SoftDTW classifier.

    Args:
        train_series:  List of N_train (T, p) arrays (all same shape).
        train_labels:  Integer labels shape (N_train,).
        test_series:   List of N_test (T, p) arrays.
        cost_fn:       Ground cost callable(a, b) → scalar.  Must expose .all_pairs.
        gamma:         SoftDTW regularisation.
        k:             Number of neighbours.
        dtype:         Computation dtype.  float32 is sufficient for KNN (no self-term
                       precision requirement).  Use float64 for WaSPS wide-range data.
        is_divergence: Use D_gamma(z,x) = SDTW(z,x) - 1/2 SDTW(z,z) - 1/2 SDTW(x,x)
                       instead of plain SDTW(z,x) as the neighbour distance. Default
                       False reduces exactly to the prior plain-SDTW behaviour (see
                       SoftDTW.value(), src/softdtw.py) — zero regression for existing
                       callers. When cost_fn is WaSPS, True auto-forces
                       cost_fn.log_correction=True (same coupling as barycenter fitting).

    Returns:
        predictions: (N_test,) integer array.
    """
    labels = np.asarray(train_labels)
    train_jax = [jnp.array(s, dtype=dtype) for s in train_series]
    softdtw = SoftDTW(cost_fn, gamma, is_divergence=is_divergence, manual_grad=False)

    # Stack training series for vmap — requires homogeneous shape (T, p).
    shapes = [s.shape for s in train_jax]
    homogeneous = len(set(shapes)) == 1

    if homogeneous:
        train_stacked = jnp.stack(train_jax)   # (N_train, T, p)

        @jax.jit
        def dists_to_train(z: jax.Array) -> jax.Array:
            """vmap over training set: (N_train,) distances."""
            return jax.vmap(
                lambda x: softdtw.value(z, x)
            )(train_stacked)
    else:
        # Fallback: list comprehension (works for variable shapes)
        @jax.jit
        def dists_to_train(z: jax.Array) -> jax.Array:
            return jnp.stack([
                softdtw.value(z, x)
                for x in train_jax
            ])

    preds = []
    for s in test_series:
        z = jnp.array(s, dtype=dtype)
        dists = np.array(dists_to_train(z))
        nn_idx = np.argsort(dists)[:k]
        nn_labels = labels[nn_idx]
        counts = np.bincount(nn_labels, minlength=int(labels.max()) + 1)
        preds.append(int(np.argmax(counts)))

    return np.array(preds)
