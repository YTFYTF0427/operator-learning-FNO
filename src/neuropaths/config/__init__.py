"""Dataclass-based experiment configs.

Every experiment is described by a single ``ExperimentConfig`` instance,
serialisable to/from YAML. CLI entry points load a config from disk, so
nothing in the codebase holds a hardcoded path or hyperparameter.

The split mirrors the codebase structure:

    ExperimentConfig
     ├─ DataConfig       -- grid size, number of solutions, output CSV path
     ├─ PDEConfig        -- Reynolds number, domain parameters, FD grid
     ├─ ModelConfig      -- FNO hyperparameters (modes, width, depth, ...)
     ├─ TrainConfig      -- epochs, lr, batch size, loss, scheduler, device
     └─ EvalConfig       -- test CSV, plots dir, checkpoint path
"""

from neuropaths.config.schema import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
    PDEConfig,
    TrainConfig,
)
from neuropaths.config.loader import load_config, dump_config

__all__ = [
    "DataConfig",
    "EvalConfig",
    "ExperimentConfig",
    "ModelConfig",
    "PDEConfig",
    "TrainConfig",
    "load_config",
    "dump_config",
]
