"""PyTorch Dataset for the committor training CSVs.

Each CSV row is a single coarse-grid sample (x, y, b1, b2, rho[, finv])
tagged by solution_id. The Dataset groups by solution_id once in
__init__ and yields, per sample, a dict matching what
``neuralop.Trainer`` expects (batch keys "x" and "y"):

    {
        "x": (C, G, G) float32 tensor,
              C = 2 (b1, b2)        for square domains
              C = 3 (b1, b2, finv)  for curved domains
        "y": (1, G, G) float32 tensor,  # rho_react on the coarse grid
    }

The (x, y) coordinate columns are *not* loaded as channels: neuralop's
FNO appends a grid positional embedding internally (and re-instantiates
it at the evaluation resolution, which is what makes zero-shot
super-resolution well-defined). They are still used to deterministically
order rows within a solution group.

Input normalization
-------------------
The velocity components have RMS ~ 50 and peaks up to ~120 while the
target rho_react sits in [0, 1]. Without normalisation the FNO has to
absorb a factor-of-100 dynamic range purely through its weights, which
slows convergence. ``CommittorDataset`` standardises each input channel
(mean 0, std 1) using statistics computed once over the loaded data.
At inference / on validation splits, pass the train dataset's
``stats`` to apply the same transform consistently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class ChannelStats:
    """Per-channel mean / std for input standardisation."""

    mean: np.ndarray  # shape (C,)
    std: np.ndarray  # shape (C,)


class CommittorDataset(Dataset):
    """Reads a committor CSV once, caches the per-solution tensors."""

    def __init__(
        self,
        csv_path: str | Path,
        *,
        grid_size: int,
        include_finv: bool = False,
        stats: ChannelStats | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.grid_size = int(grid_size)
        self.include_finv = bool(include_finv)

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        required = {"solution_id", "x", "y", "b1", "b2", "rho"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV {self.csv_path} missing columns: {sorted(missing)}")
        if self.include_finv and "finv" not in df.columns:
            raise ValueError(
                f"include_finv=True but CSV {self.csv_path} has no 'finv' column."
            )

        expected = self.grid_size * self.grid_size
        groups = list(df.groupby("solution_id", sort=True))

        raw_inputs: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for sid, group in groups:
            if len(group) != expected:
                raise ValueError(
                    f"solution_id={sid} has {len(group)} rows, expected {expected} "
                    f"for grid_size={self.grid_size}. CSV {self.csv_path} may be "
                    f"from a different coarse_grid."
                )

            # Match the generator's meshgrid(indexing='ij'): rows ordered
            # by (x ascending, y ascending). The (x, y) columns are used
            # only for ordering; they are not loaded as input channels.
            group = group.sort_values(["x", "y"], kind="mergesort").reset_index(drop=True)
            G = self.grid_size

            def reshape(col: str) -> np.ndarray:
                return group[col].to_numpy(dtype=np.float32).reshape(G, G)

            channels = [reshape("b1"), reshape("b2")]
            if self.include_finv:
                channels.append(reshape("finv"))
            raw_inputs.append(np.stack(channels, axis=0))  # (C, G, G)
            targets.append(reshape("rho")[None, :, :])  # (1, G, G)

        # (N, C, G, G) for stats; back to per-sample tensors after normalisation.
        all_inputs = np.stack(raw_inputs, axis=0)
        if stats is None:
            mean = all_inputs.mean(axis=(0, 2, 3))  # (C,)
            std = all_inputs.std(axis=(0, 2, 3))  # (C,)
            # Guard against degenerate channels (constant data); leave them alone.
            std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
            stats = ChannelStats(mean=mean.astype(np.float32), std=std)
        self.stats = stats

        # m = stats.mean.reshape(1, -1, 1, 1)
        # s = stats.std.reshape(1, -1, 1, 1)
        # all_inputs = (all_inputs - m) / s

        self._inputs = [torch.from_numpy(x) for x in all_inputs]
        self._targets = [torch.from_numpy(y) for y in targets]

    def __len__(self) -> int:
        return len(self._inputs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"x": self._inputs[idx], "y": self._targets[idx]}


def make_dataloader(
    dataset: CommittorDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
