"""Tests for softdtw.py — forward DP, backward DP, SoftDTW class."""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp

from softdtw import sdtw_value, _sdtw_fwd, _sdtw_backward, SoftDTW
from costs import WaSPS, SqEuclidean


# ---------------------------------------------------------------------------
# Forward DP correctness
# ---------------------------------------------------------------------------

class TestForwardDP:
    def test_identity_path(self):
        # D = I₃: off-diagonal zeros are cheaper. Optimal path: (1,1)→(1,2)→(1,3)→(2,3)→(3,3),
        # hitting D[0,0]+D[0,1]+D[0,2]+D[1,2]+D[2,2] = 1+0+0+0+1 = 2.
        D = jnp.eye(3, dtype=jnp.float64)
        val = sdtw_value(D, gamma=1e-4)
        assert abs(float(val) - 2.0) < 1e-3, f"Expected ≈2, got {float(val)}"

    def test_constant_cost_matrix(self):
        # All costs = c: SDTW(X,Y) should equal m+n-1 * c in hard DTW limit.
        # With small gamma, softmin → min.
        c = 2.0
        m, n = 4, 4
        D = jnp.full((m, n), c)
        val = float(sdtw_value(D, gamma=1e-6))
        # Hard DTW path on constant matrix has length m+n-1=7, cost=7c (for m=n=4 one step is shared)
        # Actually optimal DTW alignment for m=n has path length n = 4 (diagonal), cost = 4c
        expected = m * c  # diagonal path for square matrix
        assert abs(val - expected) / (expected + 1e-12) < 1e-4

    def test_matches_brute_force_small(self):
        """Compare JAX SDTW against numpy brute-force DP for a small example."""
        rng = np.random.default_rng(42)
        D_np = rng.uniform(0.5, 2.0, (4, 5))
        gamma = 0.5

        # Brute-force numpy
        m, n = D_np.shape
        R = np.full((m + 1, n + 1), np.inf)
        R[0, 0] = 0.0
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                vals = np.array([R[i-1, j], R[i-1, j-1], R[i, j-1]])
                v = vals / (-gamma)
                mv = v.max()
                sm = -gamma * (np.log(np.sum(np.exp(v - mv))) + mv)
                R[i, j] = D_np[i-1, j-1] + sm
        ref = R[m, n]

        got = float(sdtw_value(jnp.array(D_np), gamma))
        assert abs(got - ref) / (abs(ref) + 1e-12) < 1e-10

    def test_non_negative(self):
        rng = np.random.default_rng(1)
        for _ in range(10):
            D = jnp.array(rng.uniform(0, 5.0, (5, 6)))
            assert float(sdtw_value(D, gamma=1.0)) >= 0.0

    def test_shape_1x1(self):
        D = jnp.array([[3.14]])
        assert abs(float(sdtw_value(D, 1.0)) - 3.14) < 1e-10

    def test_shape_1xn(self):
        # m=1, n=3: only path is (1,1)→(1,2)→(1,3), cost = 1+2+3 = 6.
        D = jnp.array([[1.0, 2.0, 3.0]])
        val = float(sdtw_value(D, gamma=1e-6))
        assert abs(val - 6.0) < 1e-4

    def test_shape_mx1(self):
        # m=3, n=1: only path is (1,1)→(2,1)→(3,1), cost = 1+2+3 = 6.
        D = jnp.array([[1.0], [2.0], [3.0]])
        val = float(sdtw_value(D, gamma=1e-6))
        assert abs(val - 6.0) < 1e-4


# ---------------------------------------------------------------------------
# Gradients (sdtw_value autodiff path — oracle for SoftDTW tests)
# ---------------------------------------------------------------------------

class TestGradients:
    def _finite_diff(self, D, gamma, eps=1e-5):
        m, n = D.shape
        G = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                D_plus = D.at[i, j].add(eps)
                D_minus = D.at[i, j].add(-eps)
                G[i, j] = (float(sdtw_value(D_plus, gamma)) -
                            float(sdtw_value(D_minus, gamma))) / (2 * eps)
        return G

    def test_autodiff_matches_finite_diff(self):
        rng = np.random.default_rng(7)
        D = jnp.array(rng.uniform(0.5, 2.0, (5, 5)))
        gamma = 1.0
        G_auto = np.array(jax.grad(sdtw_value)(D, gamma))
        G_fd = self._finite_diff(D, gamma)
        np.testing.assert_allclose(G_auto, G_fd, rtol=1e-4, atol=1e-6)

    def test_grad_shape(self):
        D = jnp.ones((3, 4))
        G = jax.grad(sdtw_value)(D, 1.0)
        assert G.shape == (3, 4)

    def test_grad_non_negative(self):
        # Gradient w.r.t. D should be in [0, 1] (it's a probability distribution E)
        rng = np.random.default_rng(3)
        D = jnp.array(rng.uniform(0, 3.0, (5, 5)))
        G = np.array(jax.grad(sdtw_value)(D, 1.0))
        assert np.all(G >= -1e-10)

    def test_grad_sums_to_path_length(self):
        # E is a probability distribution over alignments: sum = expected path length.
        # For small gamma, path concentrates on the diagonal, length ≈ n = 4.
        rng = np.random.default_rng(4)
        D = jnp.array(rng.uniform(0.5, 2.0, (4, 4)))
        G = np.array(jax.grad(sdtw_value)(D, 0.1))
        # Path length for 4x4 ≥ 4 (diagonal) and ≤ 7 (all boundary).
        assert 4.0 - 0.1 <= G.sum() <= 7.0 + 0.1

    def test_hard_backward_standalone(self):
        rng = np.random.default_rng(10)
        D = jnp.array(rng.uniform(0.5, 2.0, (4, 4)))
        gamma = 0.5
        E = _sdtw_backward(D, _sdtw_fwd(D, gamma), gamma)
        G_fd = self._finite_diff(D, gamma)
        np.testing.assert_allclose(np.array(E), G_fd, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# SoftDTW class — forward (value)
# ---------------------------------------------------------------------------

class TestSoftDTWValue:
    def test_value_matches_sdtw_value_eucl(self):
        """SoftDTW.value without divergence == sdtw_value(all_pairs(X,Y))."""
        rng = np.random.default_rng(30)
        X = jnp.array(rng.uniform(0, 1, (4, 2)))
        Y = jnp.array(rng.uniform(0, 1, (5, 2)))
        gamma = 1.0
        cost_fn = SqEuclidean()
        sdtw = SoftDTW(cost_fn, gamma, is_divergence=False, manual_grad=False)
        D_xy = cost_fn.all_pairs(X, Y)
        expected = sdtw_value(D_xy, gamma)
        got = sdtw.value(X, Y)
        np.testing.assert_allclose(float(got), float(expected), rtol=1e-12)

    def test_divergence_self_zero(self):
        """D_γ(X,X) = 0 by construction."""
        rng = np.random.default_rng(31)
        X = jnp.array(rng.uniform(0.5, 3.0, (4, 1)))
        cost_fn = WaSPS('exponential', log_correction=True)
        sdtw = SoftDTW(cost_fn, gamma=1.0, is_divergence=True, manual_grad=False)
        assert abs(float(sdtw.value(X, X))) < 1e-10

    def test_log_correction_auto_coupled(self):
        """SoftDTW(is_divergence=True) forces log_correction=True on WaSPS."""
        cost_fn = WaSPS('exponential', log_correction=False)
        SoftDTW(cost_fn, gamma=1.0, is_divergence=True)
        assert cost_fn.log_correction is True

    def test_divergence_formula(self):
        rng = np.random.default_rng(32)
        X = jnp.array(rng.uniform(0.5, 2.0, (3, 1)))
        Y = jnp.array(rng.uniform(0.5, 2.0, (4, 1)))
        cost_fn = WaSPS('exponential', log_correction=True)
        gamma = 0.8
        sdtw = SoftDTW(cost_fn, gamma, is_divergence=True, manual_grad=False)
        D_xy = cost_fn.all_pairs(X, Y)
        D_xx = cost_fn.all_pairs(X, X)
        D_yy = cost_fn.all_pairs(Y, Y)
        expected = (sdtw_value(D_xy, gamma)
                    - 0.5 * sdtw_value(D_xx, gamma)
                    - 0.5 * sdtw_value(D_yy, gamma))
        got = sdtw.value(X, Y)
        np.testing.assert_allclose(float(got), float(expected), rtol=1e-12)


# ---------------------------------------------------------------------------
# SoftDTW class — value_and_grad: manual == autodiff (spine test)
# ---------------------------------------------------------------------------

class TestSoftDTWValueAndGrad:
    """Spine: manual_grad path must match jax.value_and_grad(value) for all configs."""

    def _run(self, cost_fn, X, Y, is_divergence, gamma=1.0):
        """Return (val_manual, gX_manual, val_auto, gX_auto)."""
        sdtw_manual = SoftDTW(cost_fn, gamma, is_divergence=is_divergence, manual_grad=True)
        sdtw_auto = SoftDTW(cost_fn, gamma, is_divergence=is_divergence, manual_grad=False)
        val_m, gX_m = sdtw_manual.value_and_grad(X, Y)
        val_a, gX_a = sdtw_auto.value_and_grad(X, Y)
        return val_m, gX_m, val_a, gX_a

    def test_exponential_no_divergence(self):
        rng = np.random.default_rng(40)
        X = jnp.array(rng.uniform(0.5, 3.0, (4, 1)))
        Y = jnp.array(rng.uniform(0.5, 3.0, (5, 1)))
        cost_fn = WaSPS('exponential')
        val_m, gX_m, val_a, gX_a = self._run(cost_fn, X, Y, is_divergence=False)
        np.testing.assert_allclose(float(val_m), float(val_a), rtol=1e-10)
        np.testing.assert_allclose(np.array(gX_m), np.array(gX_a), rtol=1e-5, atol=1e-8)

    def test_exponential_divergence(self):
        rng = np.random.default_rng(41)
        X = jnp.array(rng.uniform(0.5, 3.0, (4, 1)))
        Y = jnp.array(rng.uniform(0.5, 3.0, (4, 1)))
        cost_fn = WaSPS('exponential', log_correction=True)
        val_m, gX_m, val_a, gX_a = self._run(cost_fn, X, Y, is_divergence=True)
        np.testing.assert_allclose(float(val_m), float(val_a), rtol=1e-10)
        np.testing.assert_allclose(np.array(gX_m), np.array(gX_a), rtol=1e-5, atol=1e-8)

    def test_weibull_divergence(self):
        rng = np.random.default_rng(42)
        X = jnp.array(np.column_stack([rng.uniform(0.8, 2.5, 4), rng.uniform(0.5, 3.0, 4)]))
        Y = jnp.array(np.column_stack([rng.uniform(0.8, 2.5, 4), rng.uniform(0.5, 3.0, 4)]))
        cost_fn = WaSPS('weibull', log_correction=True)
        val_m, gX_m, val_a, gX_a = self._run(cost_fn, X, Y, is_divergence=True)
        np.testing.assert_allclose(float(val_m), float(val_a), rtol=1e-10)
        np.testing.assert_allclose(np.array(gX_m), np.array(gX_a), rtol=1e-4, atol=1e-6)

    def test_weibull_with_positivity_constraint(self):
        """use_positivity_constraint=True: manual must still match autodiff."""
        rng = np.random.default_rng(43)
        # θ in unconstrained space
        X = jnp.array(np.column_stack([rng.uniform(-1.0, 1.0, 4), rng.uniform(-1.0, 1.0, 4)]))
        Y = jnp.array(np.column_stack([rng.uniform(-1.0, 1.0, 4), rng.uniform(-1.0, 1.0, 4)]))
        cost_fn = WaSPS('weibull', log_correction=True, use_positivity_constraint=True)
        val_m, gX_m, val_a, gX_a = self._run(cost_fn, X, Y, is_divergence=True)
        np.testing.assert_allclose(float(val_m), float(val_a), rtol=1e-10)
        np.testing.assert_allclose(np.array(gX_m), np.array(gX_a), rtol=1e-4, atol=1e-6)

    def test_euclidean_autodiff_only(self):
        """SqEuclidean has no gradient_X — manual_grad=True should raise or work via autodiff."""
        rng = np.random.default_rng(44)
        X = jnp.array(rng.uniform(0, 1, (4, 2)))
        Y = jnp.array(rng.uniform(0, 1, (4, 2)))
        cost_fn = SqEuclidean()
        # SqEuclidean has no gradient_X: always use manual_grad=False
        sdtw = SoftDTW(cost_fn, gamma=1.0, is_divergence=True, manual_grad=False)
        val, gX = sdtw.value_and_grad(X, Y)
        assert jnp.isfinite(val)
        assert jnp.all(jnp.isfinite(gX))


# ---------------------------------------------------------------------------
# SoftDTW divergence gradient at self — spine
# ---------------------------------------------------------------------------

class TestDivergenceGradientAtSelf:
    def test_gradient_at_self_is_zero_wasps_exp(self):
        """∂D_γ(X,X)/∂X = 0 — manual path, exponential."""
        rng = np.random.default_rng(50)
        X = jnp.array(rng.uniform(0.5, 3.0, (4, 1)))
        cost_fn = WaSPS('exponential', log_correction=True)
        sdtw = SoftDTW(cost_fn, gamma=1.0, is_divergence=True, manual_grad=True)
        _, gX = sdtw.value_and_grad(X, X)
        np.testing.assert_allclose(np.array(gX), 0.0, atol=1e-8)

    def test_gradient_at_self_is_zero_wasps_weibull(self):
        """∂D_γ(X,X)/∂X = 0 — manual path, Weibull."""
        rng = np.random.default_rng(51)
        X = jnp.array(np.column_stack([rng.uniform(0.8, 2.5, 4), rng.uniform(0.5, 3.0, 4)]))
        cost_fn = WaSPS('weibull', log_correction=True)
        sdtw = SoftDTW(cost_fn, gamma=1.0, is_divergence=True, manual_grad=True)
        _, gX = sdtw.value_and_grad(X, X)
        np.testing.assert_allclose(np.array(gX), 0.0, atol=1e-7)

    def test_divergence_gradient_at_self_is_zero_autodiff(self):
        """∂D_γ(X,X)/∂X = 0 — autodiff spine test (rows AND columns of E).

        Verifies that when D_xy = D_xx = D_yy (self-divergence), autodiff
        correctly sums contributions from all three terms.
        """
        rng = np.random.default_rng(99)
        X = jnp.array(rng.uniform(0.5, 3.0, (4, 1)))
        cost_fn = WaSPS('exponential', log_correction=True)
        sdtw = SoftDTW(cost_fn, gamma=1.0, is_divergence=True, manual_grad=False)
        grad = jax.grad(lambda x: sdtw.value(x, x))(X)
        np.testing.assert_allclose(np.array(grad), 0.0, atol=1e-8)


# ---------------------------------------------------------------------------
# JIT + vmap
# ---------------------------------------------------------------------------

class TestJitVmap:
    def test_jit_sdtw_value(self):
        D = jnp.ones((4, 4))
        fn = jax.jit(lambda d: sdtw_value(d, 1.0))
        assert jnp.isfinite(fn(D))

    def test_jit_grad_sdtw_value(self):
        D = jnp.ones((4, 4))
        fn = jax.jit(jax.grad(lambda d: sdtw_value(d, 1.0)))
        G = fn(D)
        assert G.shape == (4, 4)
        assert jnp.all(jnp.isfinite(G))

    def test_vmap_batch(self):
        rng = np.random.default_rng(20)
        batch = jnp.array(rng.uniform(0.5, 2.0, (8, 4, 5)))
        vals = jax.vmap(lambda d: sdtw_value(d, 1.0))(batch)
        assert vals.shape == (8,)
        assert jnp.all(jnp.isfinite(vals))

    def test_vmap_grad(self):
        rng = np.random.default_rng(21)
        batch = jnp.array(rng.uniform(0.5, 2.0, (4, 3, 3)))
        grads = jax.vmap(jax.grad(lambda d: sdtw_value(d, 0.5)))(batch)
        assert grads.shape == (4, 3, 3)
        assert jnp.all(jnp.isfinite(grads))

    def test_softdtw_value_jit(self):
        rng = np.random.default_rng(22)
        X = jnp.array(rng.uniform(0.5, 2.0, (4, 1)))
        Y = jnp.array(rng.uniform(0.5, 2.0, (5, 1)))
        cost_fn = WaSPS('exponential', log_correction=True)
        sdtw = SoftDTW(cost_fn, gamma=1.0, is_divergence=True, manual_grad=True)
        fn = jax.jit(sdtw.value)
        assert jnp.isfinite(fn(X, Y))
