"""Tests for neuropaths.pde.transforms.

TODO:
    * forward_map(phi, psi) and inverse_map(phi, psi) compose to the
      identity on [0,1]^2 for a few randomly drawn (phi, psi).
    * f_x = 1, f_xx = 0 identically (dissertation's key algebraic
      simplification from Theorem 4.1).
    * coord_transform_derivatives returns f_y matching a central-
      difference estimate on a dense y-grid (autograd vs FD sanity).
"""
