"""Ground costs for SoftDTW: SqEuclidean + WaSPS.

SqEuclidean extends OTT's SqEuclidean with an optional positivity constraint
(softplus) and log-correction.

WaSPS implements W₂² between parametric 1-D distributions as an OTT CostFn.
Only exponential and Weibull have closed-form W₂² (fully differentiable via JAX).

Reference formulas (RESSOURCES/sections/supplementary.tex):
  Exponential:  W₂²(β₁,β₂) = 2(β₁−β₂)²/(β₁²β₂²)
                ∂/∂β₁       = 4(β₁−β₂)/(β₁³β₂)
  Weibull:      W₂²(p,q)    = λ₁²Γ((k₁+2)/k₁) + λ₂²Γ((k₂+2)/k₂) − 2λ₁λ₂Γ(u)
                              where u = 1+1/k₁+1/k₂
                ∂/∂λ₁       = 2[λ₁Γ((k₁+2)/k₁) − λ₂Γ(u)]
                ∂/∂k₁       = (2λ₁λ₂/k₁²)Γ(u)ψ⁰(u) − (2λ₁²/k₁²)Γ((k₁+2)/k₁)ψ⁰((k₁+2)/k₁)
  Log-correction (log_correction=True): c = δ + log(2−exp(−δ)), ∂c/∂δ = 1 + exp(−δ)/(2−exp(−δ))
  Softplus (use_positivity_constraint=True): φ(θ) = log(1+exp(θ)), φ'(θ) = σ(θ).
    Full gradient: ∂c/∂θ_p = ∂W₂²/∂p · [1 + exp(−δ)/(2−exp(−δ))] · σ(θ_p)
    where p = φ(θ_p) and δ = W₂²(φ(θ_p), φ(θ_q)).

Bijector pattern (both WaSPS and SqEuclidean):
  to_unconstrained(p) = log(expm1(p))  if use_positivity_constraint, else identity
  to_constrained(z)   = softplus(z)     if use_positivity_constraint, else identity
  Used by fit_barycenter to round-trip params → θ-space → params.

Note: jax.scipy.special provides gammaln and digamma but not gamma() directly.
gammaln form is used throughout (log-stable, no overflow, and the only JAX option).
See CHANGES.md §Phase-A, §Phase-B.
"""

from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.tree_util as jtu
from jax.scipy.special import gammaln, digamma
from ott.geometry import costs


# ---------------------------------------------------------------------------
# Bijector helpers (shared by SqEuclidean and WaSPS)
# ---------------------------------------------------------------------------

def _inverse_softplus(x: jax.Array) -> jax.Array:
    """log(expm1(x)): numerically stable inverse of softplus for x > 0."""
    return jnp.log(jnp.expm1(jnp.clip(x, 1e-8, None)))


# ---------------------------------------------------------------------------
# SqEuclidean (OTT subclass)
# ---------------------------------------------------------------------------

@jtu.register_pytree_node_class
class SqEuclidean(costs.SqEuclidean):
    """Squared Euclidean distance with optional positivity constraint and log-correction.

    Extends OTT's SqEuclidean with the same constraint pattern as WaSPS:
      - use_positivity_constraint: apply softplus to both args before computing norm
      - log_correction: apply c = δ + log(2−exp(−δ)) to the squared distance

    Defaults: both False.  SoftDTW divergence with plain squared-Euclidean cost is
    already guaranteed ≥ 0 (Blondel et al. 2020, Proposition 4) — log_correction
    is not needed for correctness but is exposed for completeness.

    For use_positivity_constraint=True in a barycenter, pair with fit_barycenter
    via the bijectors:
        to_unconstrained(p) = inverse_softplus(p)
        to_constrained(z)   = softplus(z)
    which are identity when use_positivity_constraint=False.

    No manual gradient_X — SqEuclidean stays manual_grad=False (autodiff chains
    softplus and log-correction automatically).
    """

    def __init__(
        self,
        use_positivity_constraint: bool = False,
        log_correction: bool = False,
    ):
        self.use_positivity_constraint = use_positivity_constraint
        self.log_correction = log_correction

    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Squared Euclidean distance, with optional softplus and log-correction."""
        if self.use_positivity_constraint:
            x = jax.nn.softplus(x)
            y = jax.nn.softplus(y)
        d = super().__call__(x, y)
        if self.log_correction:
            d = d + jnp.log(2.0 - jnp.exp(-d))
        return d

    def all_pairs(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """(m, n) cost matrix.  x: (m, d), y: (n, d)."""
        return jax.vmap(lambda xi: jax.vmap(lambda yj: self(xi, yj))(y))(x)

    def to_unconstrained(self, p: jax.Array) -> jax.Array:
        """p-space → θ-space: inverse_softplus(p) if constrained, else identity."""
        if self.use_positivity_constraint:
            return _inverse_softplus(p)
        return p

    def to_constrained(self, z: jax.Array) -> jax.Array:
        """θ-space → p-space: softplus(z) if constrained, else identity."""
        if self.use_positivity_constraint:
            return jax.nn.softplus(z)
        return z

    # Pytree registration (required by OTT for jit-ability).
    # Registered separately from the parent SqEuclidean — same pattern as WaSPS.
    def tree_flatten(self):
        return (), (self.use_positivity_constraint, self.log_correction)

    @classmethod
    def tree_unflatten(cls, aux, _children):
        return cls(*aux)


# ---------------------------------------------------------------------------
# Pure-JAX closed-form W₂² (called by WaSPS and testable independently)
# ---------------------------------------------------------------------------

def w2sq_exponential(params_x: jax.Array, params_y: jax.Array) -> jax.Array:
    """W₂²(Exp(β₁), Exp(β₂)) = 2(β₁−β₂)² / (β₁²β₂²).

    params_x/y: shape (..., 1), column 0 = rate β.
    Note: supplementary.tex has a typo (missing square) — this formula is correct,
    verified by the gradient formula 4(β₁−β₂)/(β₁³β₂).
    """
    b1 = params_x[..., 0]
    b2 = params_y[..., 0]
    diff = b1 - b2
    return 2.0 * diff * diff / (b1 * b1 * b2 * b2)


def w2sq_weibull(params_x: jax.Array, params_y: jax.Array) -> jax.Array:
    """W₂²(Weibull(k₁,λ₁), Weibull(k₂,λ₂)) closed form.

    W₂² = λ₁²Γ((k₁+2)/k₁) + λ₂²Γ((k₂+2)/k₂) − 2λ₁λ₂Γ(1+1/k₁+1/k₂)

    params_x/y: shape (..., 2), col 0 = k (shape), col 1 = λ (scale).
    """
    k1, l1 = params_x[..., 0], params_x[..., 1]
    k2, l2 = params_y[..., 0], params_y[..., 1]
    term1 = l1 * l1 * jnp.exp(gammaln((k1 + 2.0) / k1))
    term2 = l2 * l2 * jnp.exp(gammaln((k2 + 2.0) / k2))
    term3 = 2.0 * l1 * l2 * jnp.exp(gammaln(1.0 + 1.0 / k1 + 1.0 / k2))
    return term1 + term2 - term3


# ---------------------------------------------------------------------------
# WaSPS CostFn
# ---------------------------------------------------------------------------

@jtu.register_pytree_node_class
class WaSPS(costs.CostFn):
    """Wasserstein parametric-series ground cost.

    Implements W₂² between parametric 1-D distributions as an OTT CostFn,
    making it pluggable as the ground cost of a SoftDTW geometry.

    Args:
        family:                    'exponential' or 'weibull'.
        log_correction:            Apply c = δ + log(2−exp(−δ)) to the raw W₂²
                                   (δ = W₂²) — required for SDTW divergence to be
                                   guaranteed non-negative. Set automatically by
                                   SoftDTW when is_divergence=True.
        use_positivity_constraint: Apply φ = softplus to both args before W₂²:
                                   cost(θ, θ') = W₂²(φ(θ), φ(θ')).
                                   gradient_X then returns ∂c/∂θ (not ∂c/∂p).
                                   Use in the barycenter (optimise unconstrained θ).
                                   KNN/predict: use False (params already positive).
    """

    def __init__(
        self,
        family: str,
        log_correction: bool = False,
        use_positivity_constraint: bool = False,
    ):
        self.family = family.lower()
        self.log_correction = log_correction
        self.use_positivity_constraint = use_positivity_constraint
        if self.family not in ('exponential', 'weibull'):
            raise NotImplementedError(
                f"WaSPS closed-form only for 'exponential' or 'weibull'; "
                f"got '{family}'."
            )

    def __call__(self, params_x: jax.Array, params_y: jax.Array) -> jax.Array:
        """Ground cost between two parameter vectors."""
        if self.use_positivity_constraint:
            params_x = jax.nn.softplus(params_x)
            params_y = jax.nn.softplus(params_y)
        if self.family == 'exponential':
            d = w2sq_exponential(params_x, params_y)
        else:
            d = w2sq_weibull(params_x, params_y)
        if self.log_correction:
            d = d + jnp.log(2.0 - jnp.exp(-d))
        return d

    def all_pairs(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """[m, n] matrix of cost values.  x: (m, n_params), y: (n, n_params)."""
        return jax.vmap(lambda xi: jax.vmap(lambda yj: self(xi, yj))(y))(x)

    # -------------------------------------------------------------------
    # Bijectors (p-space ↔ θ-space round-trip for the barycenter)
    # -------------------------------------------------------------------

    def to_unconstrained(self, p: jax.Array) -> jax.Array:
        """p-space → θ-space: inverse_softplus(p) if constrained, else identity."""
        if self.use_positivity_constraint:
            return _inverse_softplus(p)
        return p

    def to_constrained(self, z: jax.Array) -> jax.Array:
        """θ-space → p-space: softplus(z) if constrained, else identity."""
        if self.use_positivity_constraint:
            return jax.nn.softplus(z)
        return z

    # -------------------------------------------------------------------
    # Manual gradient — ∂c/∂θ_x (closed-form, w.r.t. the first argument only)
    # Used by SoftDTW.value_and_grad when manual_grad=True.
    #
    # gradient_Y is not implemented: the divergence backward differentiates
    # only w.r.t. x (the barycenter). The self-term identity gives:
    #   ∂SDTW(x,x)/∂x = 2·gradient_X(E_xx, x, x)   (E_xx and c symmetric)
    # so  −½·∂f(x,x)/∂x = −gradient_X(E_xx, x, x).
    # Full divergence backward: gradient_X(E_xy, x, y) − gradient_X(E_xx, x, x).
    # -------------------------------------------------------------------

    def gradient_X(self, E: jax.Array, X: jax.Array, Y: jax.Array) -> jax.Array:
        """∂(Σ_ij E_ij · c(X_i, Y_j))/∂X — manual gradient w.r.t. the first arg.

        E: (m, n) alignment matrix (∂SDTW/∂D from _sdtw_backward).
        X: (m, n_params),  Y: (n, n_params).
        Returns: (m, n_params).

        Full chain (when log_correction and/or use_positivity_constraint enabled):
          ∂c/∂θ_x = ∂W₂²/∂p · [1 + e^{−δ}/(2−e^{−δ})] · σ(θ_x)
        where p = φ(θ_x), δ = W₂²(φ(θ_x), φ(θ_y)).
        """
        if self.family == 'exponential':
            return self._grad_exp_X(E, X, Y)
        elif self.family == 'weibull':
            return self._grad_weibull_X(E, X, Y)
        else:
            raise NotImplementedError(
                f"WaSPS closed-form gradient only for 'exponential' or 'weibull'; "
                f"got '{self.family}'."
            )

    # ---- Exponential helper ----

    def _grad_exp_X(self, E, X, Y):
        if self.use_positivity_constraint:
            px = jax.nn.softplus(X)      # (m, 1)
            py = jax.nn.softplus(Y)      # (n, 1)
            sigma_x = jax.nn.sigmoid(X)  # (m, 1) — φ'(θ_x)
        else:
            px, py, sigma_x = X, Y, jnp.ones_like(X)

        bx = px[:, 0]   # (m,)
        by = py[:, 0]   # (n,)
        diff = bx[:, None] - by[None, :]                           # (m,n)
        # ∂W₂²/∂p[i] = 4(p[i]−q[j]) / (p[i]³ q[j])
        grad_D = 4.0 * diff / (bx[:, None] ** 3 * by[None, :])    # (m,n)
        if self.log_correction:
            d_raw = 2.0 * diff ** 2 / (bx[:, None] ** 2 * by[None, :] ** 2)
            grad_D = grad_D * _derivative_log_factor(d_raw)
        result = jnp.sum(E * grad_D, axis=1, keepdims=True)        # (m,1)
        return result * sigma_x                                     # (m,1)

    # ---- Weibull helper ----

    def _grad_weibull_X(self, E, X, Y):
        if self.use_positivity_constraint:
            pX = jax.nn.softplus(X)      # (m, 2)
            pY = jax.nn.softplus(Y)      # (n, 2)
            sigma_x = jax.nn.sigmoid(X)  # (m, 2) — φ'(θ_x)
        else:
            pX, pY, sigma_x = X, Y, jnp.ones_like(X)

        k1 = pX[:, 0];  l1 = pX[:, 1]   # (m,)
        k2 = pY[:, 0];  l2 = pY[:, 1]   # (n,)

        u = 1.0 + 1.0 / k1[:, None] + 1.0 / k2[None, :]   # (m,n)
        v = (k1 + 2.0) / k1                                 # (m,)

        Gv = jnp.exp(gammaln(v))   # (m,)
        Gu = jnp.exp(gammaln(u))   # (m,n)

        # ∂W²/∂λ₁[i] = 2[λ₁[i]Γ(v_i) − λ₂[j]Γ(u_ij)]
        d_dl = 2.0 * (l1[:, None] * Gv[:, None] - l2[None, :] * Gu)  # (m,n)

        # ∂W²/∂k₁[i] = (2λ₁[i]λ₂[j]/k₁[i]²)Γ(u_ij)ψ⁰(u_ij)
        #             − (2λ₁[i]²/k₁[i]²)Γ(v_i)ψ⁰(v_i)
        psi_u = digamma(u)         # (m,n)
        psi_v = digamma(v)         # (m,)
        d_dk = ((2.0 * l1[:, None] * l2[None, :] / k1[:, None] ** 2) * Gu * psi_u
                - (2.0 * l1[:, None] ** 2 / k1[:, None] ** 2) * Gv[:, None] * psi_v[:, None])  # (m,n)

        if self.log_correction:
            G2_x = l2 ** 2 * jnp.exp(gammaln((k2 + 2.0) / k2))    # (n,)
            d_raw = (l1[:, None] ** 2 * Gv[:, None]
                     + G2_x[None, :]
                     - 2.0 * l1[:, None] * l2[None, :] * Gu)        # (m,n)
            cf = _derivative_log_factor(d_raw)
            d_dl = d_dl * cf
            d_dk = d_dk * cf

        grad_k = jnp.sum(E * d_dk, axis=1)    # (m,)
        grad_l = jnp.sum(E * d_dl, axis=1)    # (m,)
        result = jnp.stack([grad_k, grad_l], axis=1)   # (m,2)
        return result * sigma_x                          # (m,2)

    # Pytree registration (required by OTT for jit-ability)
    def tree_flatten(self):
        return (), (self.family, self.log_correction, self.use_positivity_constraint)

    @classmethod
    def tree_unflatten(cls, aux, _children):
        return cls(*aux)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _derivative_log_factor(d: jax.Array) -> jax.Array:
    """∂c/∂δ for c = δ + log(2−exp(−δ)):  1 + exp(−δ)/(2−exp(−δ))."""
    e = jnp.exp(-d)
    return 1.0 + e / (2.0 - e)
