"""SoftDTW barycenter estimation via optax gradient descent (default: sgd; adam available).

Objective: mean_i D_γ(z, x_i) where D_γ is the SoftDTW divergence.
  The −½SDTW(z,z) self-term lives inside SoftDTW.value_and_grad.

Positivity constraint (use_positivity_constraint in SoftDTW.cost_fn):
  The cost function owns the bijectors:
    to_unconstrained(p) = inverse_softplus(p)   (identity if not constrained)
    to_constrained(z)   = softplus(z)            (identity if not constrained)
  fit_barycenter calls these generically — the optimisation loop is identical
  regardless of whether the constraint is active.  All constraint math lives
  in costs.py, not here.
"""

from __future__ import annotations

import functools
import numpy as np
import optax
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from softdtw import SoftDTW


# ---------------------------------------------------------------------------
# Module-level JIT step — cached across fit_barycenter calls
# ---------------------------------------------------------------------------

def _step_body(z, opt_state, data_arr, sdtw, tx):
    """One optimizer step: map value_and_grad over N training series, update z.

    sdtw and tx are static (Python objects, not JAX arrays).  Keeping this
    function at module level means the same Python object is reused every call →
    jax.jit can find its compiled kernel in the XLA cache without retracing.
    """
    vals, grads = jax.lax.map(lambda xi: sdtw.value_and_grad(z, xi), data_arr)
    loss = jnp.mean(vals)
    grad_z = jnp.mean(grads, axis=0)
    updates, new_state = tx.update(grad_z, opt_state)
    new_z = optax.apply_updates(z, updates)
    return new_z, new_state, loss


# static_argnums=(3, 4): sdtw and tx are Python objects (non-JAX); JAX caches
# by their id().  Classes sharing the same sdtw and tx instances get cache hits.
_step_jit = jax.jit(_step_body, static_argnums=(3, 4))


@functools.lru_cache(maxsize=32)
def _get_optimizer(lr: float, optimizer: str = "sgd") -> optax.GradientTransformation:
    """Return a cached optax optimizer for (lr, optimizer).

    Same (lr, optimizer) → same Python object → same JIT cache entry (see _step_jit).
    """
    if optimizer == "sgd":
        return optax.sgd(lr)
    if optimizer == "adam":
        return optax.adam(lr)
    raise ValueError(f"unknown optimizer {optimizer!r} (expected 'sgd' or 'adam')")


# ---------------------------------------------------------------------------
# Barycenter
# ---------------------------------------------------------------------------

def fit_barycenter(
    series: list,
    softdtw: SoftDTW,
    n_steps: int = 200,
    lr: float = 1e-2,
    init: np.ndarray | None = None,
    verbose: bool = False,
    dtype = jnp.float64,
    patience: int = 15,
    min_rel_improve: float = 1e-4,
    optimizer: str = "sgd",
) -> np.ndarray:
    """Fit a SoftDTW Fréchet barycenter of a list of parameter series.

    Objective: mean_i SoftDTW.value(z, x_i)
      (= mean_i D_γ(z, x_i) when softdtw.is_divergence=True)

    Args:
        series:           List of (T, n_params) arrays (data; not differentiated).
        softdtw:          SoftDTW instance (encodes cost, gamma, divergence, manual_grad).
        n_steps:          Maximum update steps.
        lr:               Learning rate (optimizer selected via `optimizer`).
        init:             (T, n_params) initialisation; default = mean of series.
        verbose:          Print loss every 10% of n_steps.
        dtype:            JAX computation dtype.  Default float64.
        patience:         Stop early after this many steps without a relative improvement
                          of at least min_rel_improve.  0 = disabled.
        min_rel_improve:  Minimum relative loss improvement to count as progress.
        optimizer:        "sgd" (default) or "adam".

    Returns:
        barycenter: (T, n_params) numpy array in parameter (p) space.
    """
    N = len(series)
    if N == 0:
        raise ValueError("series must be non-empty")
    if verbose:
        print(f"Fitting barycenter of {N} series with {n_steps} steps, lr={lr}, patience={patience}, min_rel_improve={min_rel_improve}")
    # NaN-fill before converting to JAX: some timesteps may have no valid samples for MLE
    # (clean_time_series filters all raw values at that step). Replace NaN with per-series
    # column mean so those timesteps contribute a neutral value rather than crashing Adam.
    series_clean = []
    for s in series:
        s_np = np.asarray(s, dtype=dtype)
        if np.isnan(s_np).any():
            col_means = np.nanmean(s_np, axis=0, keepdims=True)
            s_np = np.where(np.isnan(s_np), col_means, s_np)
        series_clean.append(s_np)
    series_jax = [jnp.asarray(s, dtype=dtype) for s in series_clean]

    if init is None:
        init_arr = jnp.mean(jnp.stack(series_jax), axis=0)
    else:
        init_arr = jnp.asarray(init, dtype=dtype)

    # Round-trip: p-space → θ-space (identity for plain callables or unconstrained costs)
    to_unc = getattr(softdtw.cost_fn, 'to_unconstrained', lambda x: x)
    to_con = getattr(softdtw.cost_fn, 'to_constrained',   lambda x: x)
    z_init = to_unc(init_arr)
    # Stack all training series into (N, T, n_params) and apply to_unc in one batch.
    # to_unc is elementwise (inverse_softplus or identity) so applying to the stacked
    # array is equivalent to N separate calls — but avoids N eager JAX dispatches.
    data_z_stacked = to_unc(jnp.stack(series_jax))   # (N, T, n_params)

    # Use the module-level _step_jit (static on sdtw and tx) so that multiple calls
    # to fit_barycenter with the same (softdtw, lr, optimizer) share the XLA compiled
    # kernel. In fit_barycenters, all classes use the same softdtw/lr/optimizer → 1
    # compilation, N-1 cache hits.  _get_optimizer caches the optax transform so the
    # same Python object is returned for the same (lr, optimizer) → same id() → same
    # static arg → same JIT cache key.
    tx = _get_optimizer(lr, optimizer)
    opt_state = tx.init(z_init)
    z = z_init

    log_every = max(1, n_steps // 10)
    best_loss = None  # None until first step; avoids float("inf")/inf = nan
    no_improve = 0
    for step in range(n_steps):
        z, opt_state, loss = _step_jit(z, opt_state, data_z_stacked, softdtw, tx)
        loss_val = float(loss)
        if verbose and (step % log_every == 0 or step == n_steps - 1):
            print(f"  step {step+1:4d}/{n_steps}  loss={loss_val:.6f}", flush=True)
        if patience > 0:
            if best_loss is None or (best_loss - loss_val) / (abs(best_loss) + 1e-8) >= min_rel_improve:
                best_loss = loss_val
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    if verbose:
                        print(f"  early stop at step {step+1} (no improve for {patience} steps)")
                    break

    # θ-space → p-space (identity for plain callables or unconstrained costs)
    # Clip to strictly positive: distribution parameters (β, λ, k) must be > 0.
    # For wasps, to_con = softplus which is always > 0, so this is a no-op.
    # For eucl_params/eucl_raw (identity bijector), the unconstrained optimizer can
    # drift slightly negative; 1e-5 floor restores physical validity without bias.
    # (Raised from 1e-8: at that magnitude the WaSPS exponential gradient ∝1/β³ is
    # already ~1e24 — far past the point where a NaN would already have occurred.)
    return np.clip(np.asarray(to_con(z)), 1e-5, None)
