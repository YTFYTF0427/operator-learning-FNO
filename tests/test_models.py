"""Tests for neuropaths.models.fno2d.

TODO:
    * Shape correctness: FNO2D(cfg)(x) takes [B, C_in, G, G] to
      [B, C_out, G, G].
    * Zero-shot super-resolution: a model trained at G=32 can be
      evaluated at G=64 without a shape error (dissertation's headline
      claim in Section 5.1). Use a freshly-initialised model and just
      check the forward pass shapes.
    * Gradient flow: a single backward pass through FNO2D fills grads
      on every parameter, including the complex Fourier weights.
    * FourierLayer2D: with a unit-kernel in Fourier space (identity in
      spectral domain) it reduces to the identity in physical space up
      to the mode truncation.
"""
