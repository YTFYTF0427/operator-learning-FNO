"""YAML <-> dataclass (de)serialisation for ExperimentConfig.

We avoid a pydantic / hydra dependency for now; a hand-rolled loader is
fine for the flat dataclass structure here. If the config tree grows,
swap in pydantic v2 BaseModels -- the surrounding API won't change.

TODO: implement string interpolation for ``${name}`` tokens in paths
(e.g. ``output_dir: "runs/${name}"``), since the dissertation's
resolution-sweep experiments want run-scoped output dirs.
"""

from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from neuropaths.config.schema import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
    PDEConfig,
    TrainConfig,
)

_SECTION_TYPES = {
    "pde": PDEConfig,
    "data": DataConfig,
    "model": ModelConfig,
    "train": TrainConfig,
    "eval": EvalConfig,
}


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Minimal dataclass-from-dict that ignores unknown keys loudly."""
    if not is_dataclass(cls):
        return data
    known = {f.name for f in fields(cls)}
    unknown = set(data.keys()) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**{k: v for k, v in data.items() if k in known})


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML file into an ExperimentConfig."""
    path = Path(path)
    with path.open("r") as fh:
        raw = yaml.safe_load(fh) or {}

    top_level = {k: v for k, v in raw.items() if k not in _SECTION_TYPES}
    sections = {k: _from_dict(_SECTION_TYPES[k], raw.get(k, {})) for k in _SECTION_TYPES}

    return ExperimentConfig(**top_level, **sections)


def dump_config(cfg: ExperimentConfig, path: str | Path) -> None:
    """Write an ExperimentConfig as YAML (for run-log provenance)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(asdict(cfg), fh, sort_keys=False)
