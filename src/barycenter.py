"""SoftDTW barycenter estimation via optax (adam) gradient descent.

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

import numpy as np
import optax
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from softdtw import SoftDTW


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
    dtype = jnp.float32,
    patience: int = 15,
    min_rel_improve: float = 1e-4,
) -> np.ndarray:
    """Fit a SoftDTW Fréchet barycenter of a list of parameter series.

    Objective: mean_i SoftDTW.value(z, x_i)
      (= mean_i D_γ(z, x_i) when softdtw.is_divergence=True)

    Args:
        series:           List of (T, n_params) arrays (data; not differentiated).
        softdtw:          SoftDTW instance (encodes cost, gamma, divergence, manual_grad).
        n_steps:          Maximum Adam update steps.
        lr:               Adam learning rate.
        init:             (T, n_params) initialisation; default = mean of series.
        verbose:          Print loss every 10% of n_steps.
        dtype:            JAX computation dtype.  Default float32.  Pass jnp.float64 for
                          higher precision (Weibull / fine-tolerance tests).
        patience:         Stop early after this many steps without a relative improvement
                          of at least min_rel_improve.  0 = disabled.
        min_rel_improve:  Minimum relative loss improvement to count as progress.

    Returns:
        barycenter: (T, n_params) numpy array in parameter (p) space.
    """
    N = len(series)
    if N == 0:
        raise ValueError("series must be non-empty")

    series_jax = [jnp.asarray(s, dtype=dtype) for s in series]

    if init is None:
        init_arr = jnp.mean(jnp.stack(series_jax), axis=0)
    else:
        init_arr = jnp.asarray(init, dtype=dtype)

    # Round-trip: p-space → θ-space (identity for plain callables or unconstrained costs)
    to_unc = getattr(softdtw.cost_fn, 'to_unconstrained', lambda x: x)
    to_con = getattr(softdtw.cost_fn, 'to_constrained',   lambda x: x)
    z_init = to_unc(init_arr)
    data_z = [to_unc(xi) for xi in series_jax]

    def step_fn(z, opt_state):
        vg_pairs = [softdtw.value_and_grad(z, xi) for xi in data_z]
        vals  = jnp.stack([v for v, _ in vg_pairs])
        grads = jnp.stack([g for _, g in vg_pairs])
        loss = jnp.mean(vals)
        grad_z = jnp.mean(grads, axis=0)
        updates, new_state = tx.update(grad_z, opt_state)
        new_z = optax.apply_updates(z, updates)
        return new_z, new_state, loss

    step_jit = jax.jit(step_fn)
    tx = optax.adam(lr)
    opt_state = tx.init(z_init)
    z = z_init

    log_every = max(1, n_steps // 10)
    best_loss = float("inf")
    no_improve = 0
    for step in range(n_steps):
        z, opt_state, loss = step_jit(z, opt_state)
        loss_val = float(loss)
        if verbose and (step % log_every == 0 or step == n_steps - 1):
            print(f"  step {step+1:4d}/{n_steps}  loss={loss_val:.6f}", flush=True)
        if patience > 0:
            rel_improve = (best_loss - loss_val) / (abs(best_loss) + 1e-8)
            if rel_improve >= min_rel_improve:
                best_loss = loss_val
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    if verbose:
                        print(f"  early stop at step {step+1} (no improve for {patience} steps)")
                    break

    # θ-space → p-space (identity for plain callables or unconstrained costs)
    return np.asarray(to_con(z))
