"""Tests for neuropaths.config (dataclass <-> YAML round-trip).

TODO:
    * ExperimentConfig() defaults match the dissertation's square_32
      baseline (coarse_grid=32, modes=12, width=128, depth=4, lr=1e-3,
      epochs=100, batch_size=32).
    * load_config(dump_config(cfg)) round-trips to an equivalent cfg.
    * Unknown YAML keys raise a clear error (don't silently drop).
    * Configs under `configs/` all parse cleanly.
"""
