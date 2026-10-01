"""`neuropaths-evaluate --config path/to.yaml` -- test-set evaluation + plots."""

from __future__ import annotations

import argparse

from pathlib import Path

from neuralop.models import FNO

from neuropaths.config import load_config
from neuropaths.data import CommittorDataset, make_dataloader
from neuropaths.utils import describe_device, get_device


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained FNO2D.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    device = get_device(cfg.train.device)
    print(f"Using device: {describe_device(device)}")

    dataset = CommittorDataset(
        cfg.data.test_csv,
        grid_size=cfg.pde.coarse_grid,
        include_finv=cfg.data.include_finv_column,
    )
    loader = make_dataloader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    ckpt = Path(cfg.eval.checkpoint_path)
    model = FNO.from_checkpoint(ckpt.parent, ckpt.stem, map_location=device).to(device)
    model.eval()

    # TODO: compute metrics via neuropaths.evaluation.metrics and plot
    # via neuropaths.evaluation.plots; write a JSON summary + PNGs under
    # cfg.eval.output_dir.
    _ = loader
    raise NotImplementedError(
        "TODO: port from Neural_operator/FNO_1_Test.py evaluate_model + "
        "plotting, writing outputs under cfg.eval.output_dir."
    )


if __name__ == "__main__":
    raise SystemExit(main())
