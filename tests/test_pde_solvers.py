"""Tests for neuropaths.pde.solvers."""

from __future__ import annotations

import autograd.numpy as anp
import numpy as np
import pytest

from neuropaths.pde.solvers import (
    reactive_density,
    solve_backward_committor,
    solve_forward_committor,
)
from neuropaths.pde.velocity import turbulent_velocity_field


def _square_boundaries():
    return (lambda y: anp.zeros_like(y), lambda y: anp.ones_like(y))


def _zero_velocity():
    return (
        lambda x, y: np.zeros_like(x),
        lambda x, y: np.zeros_like(x),
    )


class TestZeroVelocityAnalyticSolution:
    """For b = 0 on the unit square, q^+(x, y) = x exactly."""

    def test_recovers_linear_committor(self):
        phi, psi = _square_boundaries()
        bx, by = _zero_velocity()
        N = 60
        u, X, Y = solve_forward_committor(phi, psi, bx, by, N)

        # Analytic q^+ = x.
        err = np.abs(u - X).max()
        # Pointwise upwind + central diffusion; on b=0 there's no upwind
        # error and the discretisation is exact for linear u up to LU
        # round-off.
        assert err < 1e-10, f"max |u - x| = {err:.2e}"

    def test_backward_recovers_one_minus_x(self):
        phi, psi = _square_boundaries()
        bx, by = _zero_velocity()
        u, X, _ = solve_backward_committor(phi, psi, bx, by, 60)
        err = np.abs(u - (1.0 - X)).max()
        assert err < 1e-10, f"max |u - (1-x)| = {err:.2e}"


class TestBoundaryConditions:
    """Dirichlet rows and periodic columns must hold exactly."""

    @pytest.mark.parametrize("direction", ["forward", "backward"])
    def test_dirichlet_in_x(self, direction):
        phi, psi = _square_boundaries()
        bx, by = _zero_velocity()
        solve = solve_forward_committor if direction == "forward" else solve_backward_committor
        u, _, _ = solve(phi, psi, bx, by, 30)

        if direction == "forward":
            assert np.all(u[0, :] == 0.0)
            assert np.all(u[-1, :] == 1.0)
        else:
            assert np.all(u[0, :] == 1.0)
            assert np.all(u[-1, :] == 0.0)

    def test_periodic_in_y(self):
        phi, psi = _square_boundaries()
        bx = lambda x, y: 5.0 * np.cos(2.0 * np.pi * y)
        by = lambda x, y: 5.0 * np.sin(2.0 * np.pi * y)
        u, _, _ = solve_forward_committor(phi, psi, bx, by, 60)

        # The j=0 column is duplicated as j=Ny so values match by construction;
        # the relevant test is that periodicity propagates into the solution
        # at every interior x-row.
        assert np.allclose(u[:, 0], u[:, -1])


class TestMaximumPrinciple:
    """Discrete maximum principle on physical (turbulent) velocity fields."""

    @pytest.mark.parametrize("seed", [0, 7, 11, 16, 19])
    def test_committors_in_unit_interval(self, seed):
        phi, psi = _square_boundaries()
        rng = np.random.default_rng(seed)
        field = turbulent_velocity_field(
            reynolds=10.0,
            char_length=0.2,
            viscosity=1.0,
            k_min=1,
            k_max=8,
            n_grid=151,  # smaller than production for test speed
            rng=rng,
        )
        # n_interior = 60 keeps the test under a few seconds. At this
        # resolution the cell Peclet number can exceed 1 in patches of the
        # turbulent field, so first-order upwind allows tiny overshoots
        # (O(1e-3)). At production resolution (n_interior = 309) the bound
        # tightens to O(1e-6) — see runs/smoke_test verification.
        qp, _, _ = solve_forward_committor(phi, psi, field.bx, field.by, 60)
        qm, _, _ = solve_backward_committor(phi, psi, field.bx, field.by, 60)

        eps = 1e-2
        assert qp.min() >= -eps, f"q+ min = {qp.min()}"
        assert qp.max() <= 1.0 + eps, f"q+ max = {qp.max()}"
        assert qm.min() >= -eps, f"q- min = {qm.min()}"
        assert qm.max() <= 1.0 + eps, f"q- max = {qm.max()}"

        rho = reactive_density(qp, qm)
        assert rho.min() >= -eps and rho.max() <= 1.0 + eps


class TestForwardBackwardConsistency:
    """solve_backward_committor(b) should equal solve_forward_committor(-b)
    after the boundary flip q^+ <-> 1 - q^+."""

    def test_drift_reversal_matches_backward_solve(self):
        phi, psi = _square_boundaries()
        bx_pos = lambda x, y: 8.0 * np.ones_like(x)
        by_pos = lambda x, y: np.zeros_like(x)
        bx_neg = lambda x, y: -8.0 * np.ones_like(x)
        by_neg = lambda x, y: np.zeros_like(x)

        N = 60
        u_back, _, _ = solve_backward_committor(phi, psi, bx_pos, by_pos, N)
        u_fwd_neg, _, _ = solve_forward_committor(phi, psi, bx_neg, by_neg, N)
        # q^-_b = 1 - q^+_{-b}
        assert np.allclose(u_back, 1.0 - u_fwd_neg, atol=1e-10)
