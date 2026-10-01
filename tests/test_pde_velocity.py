"""Tests for neuropaths.pde.velocity.

TODO:
    * Generated field is divergence-free on the FFT grid
      (|div u| < tol after the Helmholtz projection).
    * Field is y-periodic: ux[:, 0] ~ ux[:, -1] and similarly for uy.
    * RMS speed matches target U = nu * Re / L within a small tolerance.
    * Spectral slope over the inertial range is close to -5/3
      (log-log linear fit on azimuthally-averaged E(k)).
    * Seeding via rng reproduces identical fields.
"""
