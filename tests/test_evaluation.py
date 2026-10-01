"""Tests for neuropaths.evaluation.

TODO:
    * relative_l2_error returns 0 when pred == target.
    * relative_l2_error matches the loss used in training on identical
      inputs (guards against the legacy drift between MSE-at-train-time
      and relative-L2-at-eval-time).
    * plot_prediction_comparison writes a PNG of non-zero size.
"""
