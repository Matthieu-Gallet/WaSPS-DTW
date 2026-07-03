"""Soft-DTW: forward DP, backward DP, and SoftDTW class.

Forward DP: nested lax.scan (outer=rows, inner=columns).
  R[i, j] = D[i-1,j-1] + softmin(R[i-1,j], R[i-1,j-1], R[i,j-1])
  Boundary: R[0,0]=0, R[i,0]=R[0,j]=+inf.

Backward DP (Cuturi-Blondel): reverse nested scan computes expected alignment E.
  E = ∂SDTW/∂D, used by SoftDTW.value_and_grad manual path.

Divergence:  D_γ(X,Y) = SDTW(X,Y) − ½SDTW(X,X) − ½SDTW(Y,Y).
  ∂D_γ/∂X = gradient_X(E_xy, X, Y) − gradient_X(E_xx, X, X)
  (the −½ and the ×2 from the symmetric self-term cancel).
  This requires cost_fn.gradient_X; see costs.py.
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Core primitive
# ---------------------------------------------------------------------------

def _softmin3(a: jax.Array, b: jax.Array, c: jax.Array, gamma: float) -> jax.Array:
    """Numerically stable softmin of three scalars: -γ·logsumexp(-{a,b,c}/γ)."""
    arr = jnp.stack([-a, -b, -c]) / gamma
    return -gamma * jax.nn.logsumexp(arr)


# ---------------------------------------------------------------------------
# Forward DP
# ---------------------------------------------------------------------------

def _sdtw_fwd(D: jax.Array, gamma: float) -> jax.Array:
    """Forward DP. Returns R of shape (m+1, n+1).

    R[0,:]=R[:,0]=+inf, R[0,0]=0. R[m,n] is the soft-DTW value.
    """
    # m, n = D.shape
    n = D.shape[1]
    dtype = D.dtype
    INF = jnp.asarray(jnp.inf, dtype=dtype)

    def col_step(r_left, args):
        d_ij, r_above, r_diag = args
        val = d_ij + _softmin3(r_above, r_diag, r_left, gamma)
        return val, val

    def row_step(r_prev, d_row):
        # r_prev: R[i-1, :] shape (n+1,)
        # xs for inner scan: (D[i-1,j-1], R[i-1,j], R[i-1,j-1]) for j=1..n
        _, r_1n = jax.lax.scan(col_step, INF, (d_row, r_prev[1:], r_prev[:-1]))
        r_row = jnp.concatenate([jnp.array([INF], dtype=dtype), r_1n])
        return r_row, r_row

    r_init = jnp.full(n + 1, INF).at[0].set(jnp.zeros((), dtype=dtype))
    _, R_rows = jax.lax.scan(row_step, r_init, D)
    # R_rows: (m, n+1) = rows R[1..m, :]
    return jnp.concatenate([r_init[None], R_rows], axis=0)  # (m+1, n+1)


def sdtw_value(D: jax.Array, gamma: float) -> jax.Array:
    """Soft-DTW value.  D: (m, n), gamma > 0.  Pure JAX autodiff path."""
    return _sdtw_fwd(D, gamma)[-1, -1]


# ---------------------------------------------------------------------------
# Backward DP (Cuturi-Blondel E matrix)
# ---------------------------------------------------------------------------

def _sdtw_backward(D: jax.Array, R_fwd: jax.Array, gamma: float) -> jax.Array:
    """Backward DP. Returns E of shape (m, n): gradient of SDTW w.r.t. D.

    D: (m, n), R_fwd: (m+1, n+1) from _sdtw_fwd.
    """
    m, n = D.shape
    dtype = D.dtype

    # Extend R to (m+2, n+2): borders at -inf, corner = R[m,n].
    R_ext = jnp.full((m + 2, n + 2), jnp.asarray(-jnp.inf, dtype=dtype))
    R_ext = R_ext.at[:m + 1, :n + 1].set(R_fwd)
    R_ext = R_ext.at[m + 1, n + 1].set(R_fwd[m, n])

    # Pad D to (m+1, n+1): last row and col = 0.
    D_pad = jnp.zeros((m + 1, n + 1), dtype=dtype)
    D_pad = D_pad.at[:m, :n].set(D)

    zero = jnp.zeros((), dtype=dtype)

    def col_step_bwd(e_right, args):
        # carry e_right = E[i, j+1]
        e_bel, e_bel1, r_ij, r_i1j, r_ij1, r_i1j1, d_a, d_b, d_c = args
        a = jnp.exp((r_i1j  - r_ij - d_a) / gamma)   # D_pad[i, j-1]
        b = jnp.exp((r_ij1  - r_ij - d_b) / gamma)   # D_pad[i-1, j]
        c = jnp.exp((r_i1j1 - r_ij - d_c) / gamma)   # D_pad[i, j]
        e_ij = e_bel * a + e_right * b + e_bel1 * c
        return e_ij, e_ij

    def row_step_bwd(e_row_below, i):
        # e_row_below = E[i+1, :] shape (n+2,); i = DTW row (m..1)
        j = jnp.arange(1, n + 1)  # DTW j = 1..n

        r_ij   = R_ext[i,     j]
        r_i1j  = R_ext[i + 1, j]
        r_ij1  = R_ext[i,     j + 1]
        r_i1j1 = R_ext[i + 1, j + 1]
        d_a    = D_pad[i,     j - 1]   # D_pad[i, j-1]
        d_b    = D_pad[i - 1, j]       # D_pad[i-1, j]
        d_c    = D_pad[i,     j]       # D_pad[i, j]
        e_bel  = e_row_below[j]        # E[i+1, j]
        e_bel1 = e_row_below[j + 1]    # E[i+1, j+1]

        # Scan j from n to 1 (reverse: process in reversed xs order).
        xs = (e_bel[::-1], e_bel1[::-1], r_ij[::-1], r_i1j[::-1],
              r_ij1[::-1], r_i1j1[::-1], d_a[::-1], d_b[::-1], d_c[::-1])
        _, e_rev = jax.lax.scan(col_step_bwd, zero, xs)
        # e_rev[0]=E[i,n], ..., e_rev[n-1]=E[i,1]
        e_1n = e_rev[::-1]  # E[i, 1..n]

        e_row = jnp.zeros(n + 2, dtype=dtype).at[1:n + 1].set(e_1n)
        return e_row, e_1n

    # Init: E[m+1, n+1] = 1, rest 0.
    e_init = jnp.zeros(n + 2, dtype=dtype).at[n + 1].set(jnp.ones((), dtype=dtype))
    i_range = jnp.arange(m, 0, -1)   # [m, m-1, ..., 1]
    _, E_rows = jax.lax.scan(row_step_bwd, e_init, i_range)
    # E_rows[k] = E[m-k, 1:n+1]; reverse rows → E[k] = E[k+1, 1:n+1]
    return E_rows[::-1]  # (m, n) = ∂SDTW/∂D


# ---------------------------------------------------------------------------
# Helper — cost matrix for any callable
# ---------------------------------------------------------------------------

def _all_pairs(cost_fn, x: jax.Array, y: jax.Array) -> jax.Array:
    """(m, n) cost matrix.  Works for OTT CostFn (uses .all_pairs) and plain callables.

    Note (symmetry): self-terms D_xx / D_yy are symmetric, so computing only the
    upper triangle might seem attractive.  Benchmark (scripts/bench_all_pairs.py,
    2026-06) shows the approach is *slower* at all real config sizes (T ≤ 52):
    triangular indexing via jnp.triu_indices breaks XLA vectorisation (≤ 0.72x vs
    full vmap).  Conclusion: no optimisation warranted; leave as-is.
    """
    if hasattr(cost_fn, 'all_pairs'):
        return cost_fn.all_pairs(x, y)
    return jax.vmap(lambda xi: jax.vmap(lambda yj: cost_fn(xi, yj))(y))(x)


# ---------------------------------------------------------------------------
# SoftDTW — unified forward + backward
# ---------------------------------------------------------------------------

class SoftDTW:
    """SoftDTW geometry parameterised by cost function and divergence mode.

    Instantiated once per method configuration; passed to fit_barycenter.

    Args:
        cost_fn:      Ground cost callable(a, b) → scalar.
                      For manual_grad=True, must expose cost_fn.gradient_X.
        gamma:        SoftDTW smoothing parameter > 0.
        is_divergence: Compute D_γ(X,Y) = SDTW(X,Y) − ½SDTW(X,X) − ½SDTW(Y,Y).
                      When True and cost_fn is WaSPS, forces cost_fn.log_correction=True.
        manual_grad:  Use closed-form backward (cost_fn.gradient_X) for value_and_grad.
                      Falls back to autodiff when False (required for STA / SqEuclidean).
    """

    def __init__(self, cost_fn, gamma: float, is_divergence: bool = True,
                 manual_grad: bool = True):
        from costs import WaSPS
        self.cost_fn = cost_fn
        self.gamma = gamma
        self.is_divergence = is_divergence
        self.manual_grad = manual_grad
        # Auto-couple log_correction: a WaSPS divergence must have log_correction on.
        if is_divergence and isinstance(cost_fn, WaSPS) and not cost_fn.log_correction:
            cost_fn.log_correction = True

    def value(self, X: jax.Array, Y: jax.Array) -> jax.Array:
        """Forward: SDTW value or divergence D_γ(X,Y).

        X: (T, n_params),  Y: (T, n_params).
        Returns scalar.
        """
        D_xy = _all_pairs(self.cost_fn, X, Y)
        v = sdtw_value(D_xy, self.gamma)
        if self.is_divergence:
            D_xx = _all_pairs(self.cost_fn, X, X)
            D_yy = _all_pairs(self.cost_fn, Y, Y)
            v = v - 0.5 * (sdtw_value(D_xx, self.gamma) + sdtw_value(D_yy, self.gamma))
        return v

    def value_and_grad(
        self, X: jax.Array, Y: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Forward + backward w.r.t. X (the barycenter; Y is data, not differentiated).

        Returns: (value, ∂value/∂X).

        Manual path (manual_grad=True, requires cost_fn.gradient_X):
          gX = gradient_X(E_xy, X, Y)
          if is_divergence: gX -= gradient_X(E_xx, X, X)
          (−½ × 2 = −1; the ×2 comes from X in both args of SDTW(X,X); E_xx is symmetric.)

        Autodiff path (manual_grad=False):
          jax.value_and_grad(lambda x: self.value(x, Y))(X)
        """
        if not self.manual_grad:
            return jax.value_and_grad(lambda x: self.value(x, Y))(X)

        gamma = self.gamma
        cost_fn = self.cost_fn

        # Cross term: SDTW(X, Y)
        D_xy = _all_pairs(cost_fn, X, Y)
        R_xy = _sdtw_fwd(D_xy, gamma)
        val = R_xy[-1, -1]
        E_xy = _sdtw_backward(D_xy, R_xy, gamma)
        gX = cost_fn.gradient_X(E_xy, X, Y)

        if self.is_divergence:
            # Self term: −½ SDTW(X, X); ∂/∂X = −gradient_X(E_xx, X, X)
            # because ∂SDTW(X,X)/∂X = 2·gradient_X(E_xx,X,X) for symmetric E_xx,
            # and −½ × 2 = −1.
            D_xx = _all_pairs(cost_fn, X, X)
            R_xx = _sdtw_fwd(D_xx, gamma)
            val_xx = R_xx[-1, -1]
            E_xx = _sdtw_backward(D_xx, R_xx, gamma)
            gX = gX - cost_fn.gradient_X(E_xx, X, X)

            D_yy = _all_pairs(cost_fn, Y, Y)
            val_yy = sdtw_value(D_yy, gamma)
            val = val - 0.5 * (val_xx + val_yy)

        return val, gX
