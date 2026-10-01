"""Tests for neuropaths.data."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neuropaths.config.schema import DataConfig, PDEConfig
from neuropaths.data.generator import generate_dataset


@pytest.fixture
def small_pde_cfg() -> PDEConfig:
    """A coarse PDE setup so each solve takes <1s."""
    return PDEConfig(fine_grid=63, coarse_grid=16, domain="square")


@pytest.fixture
def small_data_cfg(tmp_path: Path) -> DataConfig:
    return DataConfig(
        num_train_solutions=4,
        num_test_solutions=2,
        train_csv=str(tmp_path / "train.csv"),
        test_csv=str(tmp_path / "test.csv"),
        rho_min_max=0.0,  # disable rejection by default
        oversample_factor=1.0,
    )


class TestRejectionSampling:
    """generate_dataset must respect DataConfig.rho_min_max."""

    def test_disabled_rejection_keeps_all(self, small_pde_cfg, small_data_cfg):
        # rho_min_max = 0.0 disables rejection; expect num_target rows.
        path = generate_dataset(
            small_pde_cfg, small_data_cfg, split="train", num_workers=1
        )
        df = pd.read_csv(path)
        sids = sorted(df["solution_id"].unique().tolist())
        assert sids == [float(i) for i in range(small_data_cfg.num_train_solutions)]

    def test_enabled_rejection_filters_and_renumbers(
        self, small_pde_cfg, small_data_cfg
    ):
        # Aggressive threshold: many turbulent draws have rho_max well below 0.1.
        cfg = replace(small_data_cfg, rho_min_max=0.1, oversample_factor=4.0)
        path = generate_dataset(small_pde_cfg, cfg, split="train", num_workers=1)
        df = pd.read_csv(path)

        sids = sorted(df["solution_id"].unique().tolist())
        assert sids == [float(i) for i in range(cfg.num_train_solutions)]

        per_sol_rho_max = df.groupby("solution_id")["rho"].max()
        assert (per_sol_rho_max >= cfg.rho_min_max).all(), (
            f"rejected sample slipped through: {per_sol_rho_max.tolist()}"
        )

    def test_insufficient_oversample_raises(self, small_pde_cfg, small_data_cfg):
        # Threshold of 0.99 will reject essentially every draw; oversample_factor=1
        # gives no slack so generation must fail.
        cfg = replace(small_data_cfg, rho_min_max=0.99, oversample_factor=1.0)
        with pytest.raises(RuntimeError, match="passed rejection"):
            generate_dataset(small_pde_cfg, cfg, split="train", num_workers=1)

    def test_l2_norm_filter_complements_peak(self, small_pde_cfg, small_data_cfg):
        # Filter on the L2 norm: catches samples whose peak passes
        # rho_min_max but whose overall mass is too small for relative-L2
        # to be well-conditioned.
        cfg = replace(
            small_data_cfg,
            rho_min_max=0.0,
            rho_min_l2=2.0,
            oversample_factor=4.0,
        )
        path = generate_dataset(small_pde_cfg, cfg, split="train", num_workers=1)
        df = pd.read_csv(path)
        per_sol_l2 = df.groupby("solution_id")["rho"].apply(
            lambda r: float(np.linalg.norm(r.to_numpy()))
        )
        assert (per_sol_l2 >= cfg.rho_min_l2).all(), per_sol_l2.tolist()
