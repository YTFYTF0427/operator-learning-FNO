"""Tests for neuropaths.training.

TODO:
    * RelativeL2Loss returns 0 when y_hat == y.
    * RelativeL2Loss is non-negative and scale-invariant in y.
    * train(...) reduces training loss on a toy deterministic dataset
      over >= 5 epochs (sanity check, not a convergence test).
    * Checkpoint written to cfg.train.checkpoint_path is loadable and
      matches the in-memory state_dict.
"""
