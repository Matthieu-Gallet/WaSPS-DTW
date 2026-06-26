"""Tests for costs.py — closed-form W₂² + WaSPS CostFn."""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp

from costs import SqEuclidean, WaSPS, w2sq_exponential, w2sq_weibull
from ott.geometry import costs as ott_costs


# ---------------------------------------------------------------------------
# SqEuclidean subclass
# ---------------------------------------------------------------------------

class TestSqEuclidean:
    """SqEuclidean subclass: basic distance, constraint, log-correction, bijectors."""

    def test_is_subclass_of_ott(self):
        assert issubclass(SqEuclidean, ott_costs.SqEuclidean)

    def test_default_no_constraint(self):
        cost = SqEuclidean()
        assert cost.use_positivity_constraint is False
        assert cost.log_correction is False

    def test_basic_distance(self):
        """‖x − y‖² for plain vectors."""
        cost = SqEuclidean()
        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])
        expected = float(jnp.sum((x - y) ** 2))
        assert abs(float(cost(x, y)) - expected) < 1e-10

    def test_self_distance_zero(self):
        cost = SqEuclidean()
        x = jnp.array([1.5, 2.0])
        assert float(cost(x, x)) < 1e-12

    def test_all_pairs_shape(self):
        cost = SqEuclidean()
        X = jnp.ones((3, 2))
        Y = jnp.ones((5, 2))
        D = cost.all_pairs(X, Y)
        assert D.shape == (3, 5)

    def test_constrained_non_negative(self):
        """With use_positivity_constraint, cost(θ, θ') ≥ 0 for any real θ."""
        cost = SqEuclidean(use_positivity_constraint=True)
        x = jnp.array([-1.0, 0.5])
        y = jnp.array([0.3, -0.5])
        assert float(cost(x, y)) >= 0

    def test_constrained_self_zero(self):
        cost = SqEuclidean(use_positivity_constraint=True)
        x = jnp.array([0.5, -1.0])
        assert float(cost(x, x)) < 1e-12

    def test_constrained_matches_manual(self):
        """cost(θ, θ') must equal ‖softplus(θ) − softplus(θ')‖²."""
        import jax.numpy as jnp2
        cost = SqEuclidean(use_positivity_constraint=True)
        x = jnp.array([-0.5, 1.0])
        y = jnp.array([0.3, -0.5])
        px, py = jax.nn.softplus(x), jax.nn.softplus(y)
        expected = float(jnp.sum((px - py) ** 2))
        assert abs(float(cost(x, y)) - expected) < 1e-10

    def test_log_correction_exceeds_plain(self):
        """log_correction=True: c = δ + log(2−exp(−δ)) > δ for δ > 0."""
        cost_plain = SqEuclidean()
        cost_lc = SqEuclidean(log_correction=True)
        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])
        d_plain = float(cost_plain(x, y))
        d_lc = float(cost_lc(x, y))
        assert d_lc > d_plain

    def test_pytree_round_trip(self):
        cost = SqEuclidean(use_positivity_constraint=True, log_correction=False)
        leaves, treedef = jax.tree_util.tree_flatten(cost)
        cost2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert cost2.use_positivity_constraint is True
        assert cost2.log_correction is False

    def test_jit_compatible(self):
        cost = SqEuclidean(use_positivity_constraint=True)
        x = jnp.array([0.5, 1.0])
        y = jnp.array([1.0, 0.5])
        result = jax.jit(cost)(x, y)
        assert jnp.isfinite(result)

    def test_autodiff_through_softplus(self):
        """Gradient must propagate through softplus when constrained."""
        cost = SqEuclidean(use_positivity_constraint=True)
        x = jnp.array([0.5])
        y = jnp.array([1.0])
        grad = jax.grad(lambda xi: cost(xi, y))(x)
        assert jnp.isfinite(grad).all()


# ---------------------------------------------------------------------------
# Bijectors: to_constrained ∘ to_unconstrained ≈ identity
# ---------------------------------------------------------------------------

class TestBijectors:
    """to_constrained(to_unconstrained(p)) ≈ p ; identity when unconstrained."""

    def test_wasps_round_trip_exponential(self):
        cost = WaSPS('exponential', use_positivity_constraint=True)
        p = jnp.array([0.5, 1.0, 2.0])
        z = cost.to_unconstrained(p)
        p_back = cost.to_constrained(z)
        np.testing.assert_allclose(np.array(p_back), np.array(p), rtol=1e-6)

    def test_wasps_identity_no_constraint(self):
        cost = WaSPS('exponential', use_positivity_constraint=False)
        p = jnp.array([0.5, -1.0, 2.0])
        np.testing.assert_allclose(np.array(cost.to_unconstrained(p)), np.array(p), atol=1e-12)
        np.testing.assert_allclose(np.array(cost.to_constrained(p)), np.array(p), atol=1e-12)

    def test_sqeuclidean_round_trip_constrained(self):
        cost = SqEuclidean(use_positivity_constraint=True)
        p = jnp.array([0.5, 1.0, 2.0])
        z = cost.to_unconstrained(p)
        p_back = cost.to_constrained(z)
        np.testing.assert_allclose(np.array(p_back), np.array(p), rtol=1e-6)

    def test_sqeuclidean_identity_no_constraint(self):
        cost = SqEuclidean(use_positivity_constraint=False)
        p = jnp.array([0.5, -1.0, 2.0])
        np.testing.assert_allclose(np.array(cost.to_unconstrained(p)), np.array(p), atol=1e-12)
        np.testing.assert_allclose(np.array(cost.to_constrained(p)), np.array(p), atol=1e-12)

    def test_to_constrained_is_positive(self):
        """softplus output is strictly positive."""
        cost = WaSPS('exponential', use_positivity_constraint=True)
        z = jnp.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        p = cost.to_constrained(z)
        assert jnp.all(p > 0)


# ---------------------------------------------------------------------------
# Closed-form W₂² — exponential
# ---------------------------------------------------------------------------

class TestW2sqExponential:
    def test_self_distance_zero(self):
        p = jnp.array([2.0])
        assert float(w2sq_exponential(p, p)) < 1e-12

    def test_symmetry(self):
        p = jnp.array([1.5])
        q = jnp.array([3.0])
        assert abs(float(w2sq_exponential(p, q)) - float(w2sq_exponential(q, p))) < 1e-12

    def test_non_negative(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            p = jnp.array(rng.uniform(0.5, 5.0, 1))
            q = jnp.array(rng.uniform(0.5, 5.0, 1))
            assert float(w2sq_exponential(p, q)) >= 0.0

    def test_matches_manual_formula(self):
        b1, b2 = 2.0, 3.0
        expected = 2.0 * (b1 - b2) ** 2 / (b1 ** 2 * b2 ** 2)
        got = float(w2sq_exponential(jnp.array([b1]), jnp.array([b2])))
        assert abs(got - expected) < 1e-14

    # ----- Spine: closed-form gradient vs finite-difference -----
    def test_gradient_formula_vs_finite_diff(self):
        """∂W₂²/∂β₁ = 4(β₁−β₂)/(β₁³β₂)  (supplementary.tex Corollary)."""
        b1, b2 = 2.0, 3.0
        eps = 1e-6
        fd = (float(w2sq_exponential(jnp.array([b1 + eps]), jnp.array([b2])))
              - float(w2sq_exponential(jnp.array([b1 - eps]), jnp.array([b2])))) / (2 * eps)
        formula = 4.0 * (b1 - b2) / (b1 ** 3 * b2)
        assert abs(fd - formula) / (abs(formula) + 1e-12) < 1e-5, (
            f"FD={fd:.8f}, formula={formula:.8f}"
        )

    def test_autodiff_matches_gradient_formula(self):
        """JAX autodiff of w2sq_exponential must match the analytic formula."""
        b1, b2 = 2.0, 3.0
        p = jnp.array([b1])
        q = jnp.array([b2])
        analytic = float(jax.grad(lambda x: w2sq_exponential(x, q))(p)[0])
        formula = 4.0 * (b1 - b2) / (b1 ** 3 * b2)
        assert abs(analytic - formula) / (abs(formula) + 1e-12) < 1e-8

    def test_differentiable(self):
        p = jnp.array([2.0])
        q = jnp.array([3.0])
        grad = jax.grad(lambda x: w2sq_exponential(x, q))(p)
        assert jnp.isfinite(grad).all()


# ---------------------------------------------------------------------------
# Closed-form W₂² — Weibull
# ---------------------------------------------------------------------------

class TestW2sqWeibull:
    def test_self_distance_zero(self):
        p = jnp.array([1.5, 2.0])
        assert float(w2sq_weibull(p, p)) < 1e-10

    def test_symmetry(self):
        p = jnp.array([1.5, 2.0])
        q = jnp.array([2.0, 1.0])
        assert abs(float(w2sq_weibull(p, q)) - float(w2sq_weibull(q, p))) < 1e-10

    def test_non_negative(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            p = jnp.array([rng.uniform(0.5, 3.0), rng.uniform(0.5, 5.0)])
            q = jnp.array([rng.uniform(0.5, 3.0), rng.uniform(0.5, 5.0)])
            assert float(w2sq_weibull(p, q)) >= -1e-12

    def test_matches_scipy_gamma(self):
        """gammaln form must match direct scipy gamma computation."""
        from scipy.special import gamma
        k1, l1, k2, l2 = 1.5, 2.0, 2.0, 3.0
        ref = (
            l1**2 * gamma((k1 + 2) / k1)
            + l2**2 * gamma((k2 + 2) / k2)
            - 2 * l1 * l2 * gamma(1 + 1/k1 + 1/k2)
        )
        got = float(w2sq_weibull(jnp.array([k1, l1]), jnp.array([k2, l2])))
        assert abs(got - ref) / (abs(ref) + 1e-12) < 1e-8

    def test_exponential_consistency(self):
        """Weibull(k=1, λ=1/β) ≡ Exponential(β) — W₂² must match."""
        beta, beta2 = 2.0, 3.0
        d_exp = float(w2sq_exponential(jnp.array([beta]), jnp.array([beta2])))
        d_weib = float(w2sq_weibull(jnp.array([1.0, 1.0/beta]), jnp.array([1.0, 1.0/beta2])))
        assert abs(d_exp - d_weib) / (abs(d_exp) + 1e-12) < 1e-8

    # ----- Spine: closed-form gradients vs finite-difference -----
    def test_gradient_wrt_lambda_fd(self):
        """∂W₂²/∂λ₁ = 2[λ₁Γ((k₁+2)/k₁) − λ₂Γ(1+1/k₁+1/k₂)]  (supplementary.tex)."""
        k1, l1, k2, l2 = 1.5, 2.0, 2.0, 1.0
        eps = 1e-5
        p = jnp.array([k1, l1])
        q = jnp.array([k2, l2])
        analytic = float(jax.grad(lambda x: w2sq_weibull(x, q))(p)[1])
        fd = (float(w2sq_weibull(jnp.array([k1, l1 + eps]), q))
              - float(w2sq_weibull(jnp.array([k1, l1 - eps]), q))) / (2 * eps)
        assert abs(analytic - fd) / (abs(fd) + 1e-12) < 1e-5

    def test_gradient_wrt_k_fd(self):
        """∂W₂²/∂k₁ — supplementary.tex digamma formula."""
        k1, l1, k2, l2 = 1.5, 2.0, 2.0, 1.0
        eps = 1e-5
        p = jnp.array([k1, l1])
        q = jnp.array([k2, l2])
        analytic = float(jax.grad(lambda x: w2sq_weibull(x, q))(p)[0])
        fd = (float(w2sq_weibull(jnp.array([k1 + eps, l1]), q))
              - float(w2sq_weibull(jnp.array([k1 - eps, l1]), q))) / (2 * eps)
        assert abs(analytic - fd) / (abs(fd) + 1e-12) < 1e-5

    def test_gradient_lambda_matches_formula(self):
        """Verify ∂W₂²/∂λ₁ matches the explicit formula from supplementary.tex."""
        from jax.scipy.special import gammaln
        k1, l1, k2, l2 = 1.5, 2.0, 2.0, 1.0
        p = jnp.array([k1, l1])
        q = jnp.array([k2, l2])
        analytic = float(jax.grad(lambda x: w2sq_weibull(x, q))(p)[1])
        # ∂/∂λ₁ = 2[λ₁Γ((k₁+2)/k₁) − λ₂Γ(1+1/k₁+1/k₂)]
        import jax.numpy as jnp2
        formula = float(2.0 * (l1 * jnp2.exp(gammaln((k1+2)/k1))
                               - l2 * jnp2.exp(gammaln(1.0 + 1.0/k1 + 1.0/k2))))
        assert abs(analytic - formula) / (abs(formula) + 1e-12) < 1e-8


# ---------------------------------------------------------------------------
# WaSPS CostFn — exponential
# ---------------------------------------------------------------------------

class TestWaSPSExponential:
    def setup_method(self):
        self.cost = WaSPS('exponential')
        self.p = jnp.array([2.0])
        self.q = jnp.array([3.0])

    def test_self_distance_zero(self):
        assert float(self.cost(self.p, self.p)) < 1e-12

    def test_symmetry(self):
        d_pq = float(self.cost(self.p, self.q))
        d_qp = float(self.cost(self.q, self.p))
        assert abs(d_pq - d_qp) < 1e-12

    def test_non_negative(self):
        assert float(self.cost(self.p, self.q)) >= 0.0

    def test_all_pairs_shape(self):
        X = jnp.array([[1.0], [2.0], [3.0]])
        Y = jnp.array([[1.5], [2.5]])
        D = self.cost.all_pairs(X, Y)
        assert D.shape == (3, 2)
        assert jnp.all(D >= -1e-12)

    def test_all_pairs_diagonal_zero(self):
        X = jnp.array([[1.0], [2.0], [3.0]])
        D = self.cost.all_pairs(X, X)
        assert jnp.all(jnp.diag(D) < 1e-12)

    def test_jit_compatible(self):
        fn = jax.jit(self.cost)
        result = fn(self.p, self.q)
        assert jnp.isfinite(result)

    def test_grad_through_cost(self):
        grad = jax.grad(lambda x: self.cost(x, self.q))(self.p)
        assert jnp.isfinite(grad).all()


class TestWaSPSWeibull:
    def setup_method(self):
        self.cost = WaSPS('weibull')
        self.p = jnp.array([1.5, 2.0])
        self.q = jnp.array([2.0, 1.0])

    def test_self_distance_zero(self):
        assert float(self.cost(self.p, self.p)) < 1e-10

    def test_symmetry(self):
        d_pq = float(self.cost(self.p, self.q))
        d_qp = float(self.cost(self.q, self.p))
        assert abs(d_pq - d_qp) < 1e-10

    def test_all_pairs_shape(self):
        X = jnp.array([[1.0, 1.0], [1.5, 2.0]])
        Y = jnp.array([[2.0, 1.5], [1.2, 0.8], [1.8, 2.5]])
        D = self.cost.all_pairs(X, Y)
        assert D.shape == (2, 3)

    def test_jit_compatible(self):
        fn = jax.jit(self.cost)
        result = fn(self.p, self.q)
        assert jnp.isfinite(result)

    def test_grad_through_cost(self):
        grad = jax.grad(lambda x: self.cost(x, self.q))(self.p)
        assert jnp.isfinite(grad).all()
        assert grad.shape == (2,)

    def test_unknown_family_raises(self):
        with pytest.raises(NotImplementedError):
            WaSPS('gamma')


# ---------------------------------------------------------------------------
# Log-correction for valid SDTW divergence (log_correction=True)
# ---------------------------------------------------------------------------

class TestWaSPSLogCorrection:
    def test_self_distance_still_zero_exponential(self):
        # c(x,x) = 0 + log(2−exp(0)) = log(1) = 0
        cost = WaSPS('exponential', log_correction=True)
        p = jnp.array([2.0])
        assert float(cost(p, p)) < 1e-12

    def test_self_distance_still_zero_weibull(self):
        cost = WaSPS('weibull', log_correction=True)
        p = jnp.array([1.5, 2.0])
        assert float(cost(p, p)) < 1e-10

    def test_formula_matches_manual(self):
        p = jnp.array([2.0])
        q = jnp.array([3.0])
        delta = float(w2sq_exponential(p, q))
        expected = delta + np.log(2.0 - np.exp(-delta))
        got = float(WaSPS('exponential', log_correction=True)(p, q))
        assert abs(got - expected) < 1e-12

    def test_modified_cost_exceeds_plain(self):
        p = jnp.array([1.5, 2.0])
        q = jnp.array([2.0, 1.0])
        plain = float(WaSPS('weibull')(p, q))
        modified = float(WaSPS('weibull', log_correction=True)(p, q))
        assert modified > plain

    def test_log_correction_gradient(self):
        """Gradient of log-correction term: ∂c/∂δ = 1 + exp(−δ)/(2−exp(−δ)).
        Spine test: modified gradient must differ from plain gradient."""
        cost_plain = WaSPS('exponential', log_correction=False)
        cost_div = WaSPS('exponential', log_correction=True)
        p = jnp.array([2.0])
        q = jnp.array([3.0])
        grad_plain = jax.grad(lambda x: cost_plain(x, q))(p)
        grad_div = jax.grad(lambda x: cost_div(x, q))(p)
        assert jnp.isfinite(grad_div).all()
        # Modified gradient = plain × (1 + exp(−δ)/(2−exp(−δ))) > plain gradient
        delta = float(w2sq_exponential(p, q))
        factor = 1.0 + np.exp(-delta) / (2.0 - np.exp(-delta))
        np.testing.assert_allclose(
            float(grad_div[0]), float(grad_plain[0]) * factor, rtol=1e-8
        )

    def test_pytree_round_trip(self):
        cost = WaSPS('weibull', log_correction=True, use_positivity_constraint=True)
        leaves, treedef = jax.tree_util.tree_flatten(cost)
        cost2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert cost2.log_correction is True
        assert cost2.use_positivity_constraint is True
        assert cost2.family == 'weibull'

    def test_jit_compatible(self):
        cost = WaSPS('weibull', log_correction=True)
        p = jnp.array([1.5, 2.0])
        q = jnp.array([2.0, 1.0])
        result = jax.jit(cost)(p, q)
        assert jnp.isfinite(result)


# ---------------------------------------------------------------------------
# Spine: gradient_X — manual gradient vs autodiff (w.r.t. first argument only)
# ---------------------------------------------------------------------------

class TestGradientExponential:
    """gradient_X(E, X, Y) must match autodiff of Σ E⊙D w.r.t. X."""

    def setup_method(self):
        rng = np.random.default_rng(200)
        self.cost = WaSPS('exponential')
        self.E = jnp.array(rng.uniform(0, 1, (4, 5)))
        self.X = jnp.array(rng.uniform(0.5, 3.0, (4, 1)))
        self.Y = jnp.array(rng.uniform(0.5, 3.0, (5, 1)))

    def _autodiff_grad_X(self, cost, E, X, Y):
        def f(X_): return jnp.sum(E * cost.all_pairs(X_, Y))
        return jax.grad(f)(X)

    def test_gradient_X_matches_autodiff(self):
        manual = self.cost.gradient_X(self.E, self.X, self.Y)
        auto = self._autodiff_grad_X(self.cost, self.E, self.X, self.Y)
        np.testing.assert_allclose(np.array(manual), np.array(auto), rtol=1e-5, atol=1e-8)

    def test_gradient_X_shape(self):
        g = self.cost.gradient_X(self.E, self.X, self.Y)
        assert g.shape == (4, 1)

    def test_gradient_X_log_correction_matches_autodiff(self):
        cost = WaSPS('exponential', log_correction=True)
        manual = cost.gradient_X(self.E, self.X, self.Y)
        auto = self._autodiff_grad_X(cost, self.E, self.X, self.Y)
        np.testing.assert_allclose(np.array(manual), np.array(auto), rtol=1e-5, atol=1e-8)


class TestGradientWeibull:
    def setup_method(self):
        rng = np.random.default_rng(210)
        self.cost = WaSPS('weibull')
        self.E = jnp.array(rng.uniform(0, 1, (3, 4)))
        self.X = jnp.array(np.column_stack([rng.uniform(0.8, 2.5, 3), rng.uniform(0.5, 3.0, 3)]))
        self.Y = jnp.array(np.column_stack([rng.uniform(0.8, 2.5, 4), rng.uniform(0.5, 3.0, 4)]))

    def _autodiff_grad_X(self, cost, E, X, Y):
        def f(X_): return jnp.sum(E * cost.all_pairs(X_, Y))
        return jax.grad(f)(X)

    def test_gradient_X_matches_autodiff(self):
        manual = self.cost.gradient_X(self.E, self.X, self.Y)
        auto = self._autodiff_grad_X(self.cost, self.E, self.X, self.Y)
        np.testing.assert_allclose(np.array(manual), np.array(auto), rtol=1e-5, atol=1e-8)

    def test_gradient_X_log_correction_matches_autodiff(self):
        cost = WaSPS('weibull', log_correction=True)
        manual = cost.gradient_X(self.E, self.X, self.Y)
        auto = self._autodiff_grad_X(cost, self.E, self.X, self.Y)
        np.testing.assert_allclose(np.array(manual), np.array(auto), rtol=1e-5, atol=1e-8)

    def test_gradient_X_shape(self):
        g = self.cost.gradient_X(self.E, self.X, self.Y)
        assert g.shape == (3, 2)


# ---------------------------------------------------------------------------
# Spine: use_positivity_constraint — gradient_X with softplus chain rule
# ---------------------------------------------------------------------------

class TestGradientXPositivityConstraint:
    """gradient_X with use_positivity_constraint must match autodiff of constrained cost."""

    def _autodiff_grad_X(self, cost, E, X, Y):
        def f(X_): return jnp.sum(E * cost.all_pairs(X_, Y))
        return jax.grad(f)(X)

    def test_exponential_no_log_correction(self):
        rng = np.random.default_rng(300)
        cost = WaSPS('exponential', log_correction=False, use_positivity_constraint=True)
        E = jnp.array(rng.uniform(0, 1, (4, 5)))
        X = jnp.array(rng.uniform(-1.0, 1.0, (4, 1)))   # unconstrained θ
        Y = jnp.array(rng.uniform(-1.0, 1.0, (5, 1)))
        manual = cost.gradient_X(E, X, Y)
        auto = self._autodiff_grad_X(cost, E, X, Y)
        np.testing.assert_allclose(np.array(manual), np.array(auto), rtol=1e-5, atol=1e-8)

    def test_exponential_with_log_correction(self):
        rng = np.random.default_rng(301)
        cost = WaSPS('exponential', log_correction=True, use_positivity_constraint=True)
        E = jnp.array(rng.uniform(0, 1, (4, 5)))
        X = jnp.array(rng.uniform(-1.0, 1.0, (4, 1)))
        Y = jnp.array(rng.uniform(-1.0, 1.0, (5, 1)))
        manual = cost.gradient_X(E, X, Y)
        auto = self._autodiff_grad_X(cost, E, X, Y)
        np.testing.assert_allclose(np.array(manual), np.array(auto), rtol=1e-5, atol=1e-8)

    def test_weibull_no_log_correction(self):
        rng = np.random.default_rng(302)
        cost = WaSPS('weibull', log_correction=False, use_positivity_constraint=True)
        E = jnp.array(rng.uniform(0, 1, (3, 4)))
        X = jnp.array(np.column_stack([rng.uniform(-1.0, 1.0, 3), rng.uniform(-1.0, 1.0, 3)]))
        Y = jnp.array(np.column_stack([rng.uniform(-1.0, 1.0, 4), rng.uniform(-1.0, 1.0, 4)]))
        manual = cost.gradient_X(E, X, Y)
        auto = self._autodiff_grad_X(cost, E, X, Y)
        np.testing.assert_allclose(np.array(manual), np.array(auto), rtol=1e-5, atol=1e-8)

    def test_weibull_with_log_correction(self):
        rng = np.random.default_rng(303)
        cost = WaSPS('weibull', log_correction=True, use_positivity_constraint=True)
        E = jnp.array(rng.uniform(0, 1, (3, 4)))
        X = jnp.array(np.column_stack([rng.uniform(-1.0, 1.0, 3), rng.uniform(-1.0, 1.0, 3)]))
        Y = jnp.array(np.column_stack([rng.uniform(-1.0, 1.0, 4), rng.uniform(-1.0, 1.0, 4)]))
        manual = cost.gradient_X(E, X, Y)
        auto = self._autodiff_grad_X(cost, E, X, Y)
        np.testing.assert_allclose(np.array(manual), np.array(auto), rtol=1e-5, atol=1e-8)

    def test_self_term_symmetric_identity(self):
        """Self-term identity: for symmetric E, autodiff(Σ E·all_pairs(x,x)) = 2·gradient_X(E,x,x).
        This underpins −½·∂SDTW(x,x)/∂x = −gradient_X(E_xx,x,x) in the divergence backward.
        The SDTW E_xx matrix is always symmetric (symmetric D → symmetric E)."""
        rng = np.random.default_rng(304)
        cost = WaSPS('exponential', log_correction=True, use_positivity_constraint=True)
        X = jnp.array(rng.uniform(-1.0, 1.0, (4, 1)))
        E_raw = jnp.array(rng.uniform(0, 1, (4, 4)))
        E_sym = (E_raw + E_raw.T) / 2.0   # symmetric E (as SDTW backward yields for symmetric D)
        manual_2x = 2.0 * cost.gradient_X(E_sym, X, X)
        auto = jax.grad(lambda x: jnp.sum(E_sym * cost.all_pairs(x, x)))(X)
        np.testing.assert_allclose(np.array(manual_2x), np.array(auto), rtol=1e-5, atol=1e-8)
