"""Tests for neuropaths.pde.boundaries."""

from __future__ import annotations

import numpy as np

from neuropaths.pde.boundaries import (
    generate_boundary_pair,
    identity_boundaries,
    random_trig_boundaries,
)


class TestIdentityBoundaries:
    def test_phi_zero_psi_one(self):
        phi, psi = identity_boundaries()
        y = np.linspace(0.0, 1.0, 32)
        assert np.allclose(phi(y), 0.0)
        assert np.allclose(psi(y), 1.0)


class TestRandomTrigBoundaries:
    def test_periodic_endpoints(self):
        # sin(k pi y) is 0 at y=0 and y=1, so phi/psi must hit their
        # designed endpoints exactly: phi(0)=phi(1)=0 and psi(0)=psi(1)=1.
        rng = np.random.default_rng(0)
        b_phi, b_psi = random_trig_boundaries(n_max=8, eps=0.1, rng=rng)
        # Boundary callables are vectorised; pass a small array so the
        # output shape is well-defined.
        ends = np.array([0.0, 1.0])
        assert np.allclose(b_phi.phi(ends), 0.0, atol=1e-12)
        assert np.allclose(b_psi.psi(ends), 1.0, atol=1e-12)

    def test_min_gap_respected(self):
        rng = np.random.default_rng(1)
        b_phi, b_psi = random_trig_boundaries(n_max=8, eps=0.1, rng=rng)
        y = np.linspace(0.0, 1.0, 1024)
        gap = np.asarray(b_psi.psi(y)) - np.asarray(b_phi.phi(y))
        assert gap.min() >= 0.1 - 1e-9

    def test_meshgrid_inputs(self):
        # Regression: the FD solver passes a 2D Y meshgrid of shape
        # (Nx+2, Ny) to phi/psi. The trig-polynomial closure must
        # broadcast its mode axis against any y shape.
        rng = np.random.default_rng(2)
        phi, psi = generate_boundary_pair(rng, kind="curved", n_max=6, eps=0.1)
        y_1d = np.linspace(0.0, 1.0, 50)
        Y_2d = np.tile(y_1d, (40, 1))
        phi_1d = np.asarray(phi(y_1d))
        phi_2d = np.asarray(phi(Y_2d))
        assert phi_1d.shape == (50,)
        assert phi_2d.shape == (40, 50)
        for row in phi_2d:
            assert np.allclose(row, phi_1d)


class TestGenerateBoundaryPair:
    def test_kind_square_returns_identity(self):
        rng = np.random.default_rng(0)
        phi, psi = generate_boundary_pair(rng, kind="square")
        y = np.linspace(0.0, 1.0, 16)
        assert np.allclose(phi(y), 0.0)
        assert np.allclose(psi(y), 1.0)

    def test_unknown_kind_raises(self):
        rng = np.random.default_rng(0)
        try:
            generate_boundary_pair(rng, kind="oblong")
        except ValueError:
            return
        raise AssertionError("expected ValueError for unknown kind")
