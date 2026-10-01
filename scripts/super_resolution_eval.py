"""Zero-shot super-resolution evaluation.

The Fourier Neural Operator's discretisation invariance means a model
trained at one grid size can be evaluated at any other resolution
without retraining (subject to ``n_modes <= grid//2``). This script
reproduces the dissertation's Table 1: train at one ``s``, evaluate at
multiple ``s'``.

Usage:

    uv run python scripts/super_resolution_eval.py --config configs/square_32.yaml
    uv run python scripts/super_resolution_eval.py --config configs/square_32.yaml \\
        --regenerate --num-workers 8

For each resolution in ``cfg.eval.test_resolutions``, the script:

    1. Generates (or reuses) ``runs/<name>/eval/super_res_test_<G>.csv``
       using the same seed sequence at all resolutions, so the underlying
       FD PDEs are identical across grids -- only the coarse subsampling
       differs.
    2. Loads the train-time per-channel normalisation stats (saved as
       ``<stem>_stats.npz`` next to the checkpoint) and applies them to
       the test inputs.
    3. Runs ``FNO.from_checkpoint`` and computes four metrics:
       per_sample_mean_rel_l2, global_rel_l2, MSE, MAE.

Rejection sampling is *disabled* for the super-res test sets so all
three resolutions see the same PDEs. The seed band is offset from the
training seed so train/test/super-res draws are disjoint.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from neuralop.models import FNO
from torch.utils.data import DataLoader

from neuropaths.config.loader import load_config
from neuropaths.data import ChannelStats, CommittorDataset, generate_dataset


@torch.no_grad()
def evaluate_at(
    model: torch.nn.Module,
    dataset: CommittorDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Run inference and accumulate the four summary metrics."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    sq_diff = sq_y = abs_diff = 0.0
    n_cells = n_samples = 0
    sum_per_sample = 0.0

    model.eval()
    for sample in loader:
        x = sample["x"].to(device)
        y = sample["y"].to(device)
        out = model(x)
        diff = (out - y).flatten(1)
        y_flat = y.flatten(1)
        sq_diff += (diff * diff).sum().item()
        sq_y += (y_flat * y_flat).sum().item()
        abs_diff += diff.abs().sum().item()
        n_cells += diff.numel()
        sum_per_sample += (diff.norm(dim=1) / (y_flat.norm(dim=1) + 1e-8)).sum().item()
        n_samples += y.shape[0]

    return {
        "per_sample_mean_rel_l2": sum_per_sample / max(n_samples, 1),
        "global_rel_l2": (sq_diff / sq_y) ** 0.5,
        "mse": sq_diff / max(n_cells, 1),
        "mae": abs_diff / max(n_cells, 1),
        "n": n_samples,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Zero-shot super-resolution evaluation across grid sizes."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Override checkpoint path prefix (default: cfg.eval.checkpoint_path).",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate the super-res CSVs even if they already exist.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Workers for parallel PDE solving during generation.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "mps", "cpu"),
        help="Device for inference.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    # Device for inference (generation is CPU-bound regardless).
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    ckpt = Path(args.checkpoint or cfg.eval.checkpoint_path)
    out_dir = Path(cfg.eval.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load model + train-time normalisation stats.
    print(f"Loading checkpoint from prefix {ckpt} on {device}...")
    model = FNO.from_checkpoint(ckpt.parent, ckpt.stem, map_location=device).to(device)
    stats_path = ckpt.parent / f"{ckpt.stem}_stats.npz"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"normalisation stats not found at {stats_path}; was the checkpoint "
            f"trained with the current trainer? (it writes <stem>_stats.npz)"
        )
    npz = np.load(stats_path)
    train_stats = ChannelStats(mean=npz["mean"], std=npz["std"])

    resolutions = list(cfg.eval.test_resolutions)
    print(f"Evaluating at resolutions {resolutions}...")

    # 2. Generate (or reuse) test data at each resolution.
    rows: list[dict[str, float]] = []
    for G in resolutions:
        csv_path = out_dir / f"super_res_test_{G}.csv"
        if args.regenerate or not csv_path.exists():
            print(f"\n[grid {G}] generating {csv_path}...")
            pde_cfg_g = replace(cfg.pde, coarse_grid=G)
            data_cfg_g = replace(
                cfg.data,
                test_csv=str(csv_path),
                # Disable rejection so every seed in the spawn produces a
                # row at every resolution -- otherwise the three test sets
                # drift apart when small-rho draws pass at one grid but
                # not another.
                rho_min_max=0.0,
                rho_min_l2=0.0,
                oversample_factor=1.0,
                # Offset the seed band: training uses seed+0, the existing
                # test set uses seed+1, super-res uses seed+100 so its
                # draws are disjoint from both.
                seed=cfg.data.seed + 100,
            )
            generate_dataset(
                pde_cfg_g,
                data_cfg_g,
                split="test",
                output_path=csv_path,
                num_workers=args.num_workers,
            )
        else:
            print(f"\n[grid {G}] reusing {csv_path}")

        # Load with the train-time normalisation. Per-channel mean/std are
        # scalars so they apply at any grid size.
        ds = CommittorDataset(
            csv_path,
            grid_size=G,
            include_finv=cfg.data.include_finv_column,
            stats=train_stats,
        )
        metrics = evaluate_at(
            model, ds, batch_size=cfg.data.batch_size, device=device
        )
        metrics["resolution"] = G
        rows.append(metrics)

    # 3. Print summary table.
    print("\n" + "=" * 90)
    print(
        f"{'grid':>6} | {'N':>5} | {'global rel L2':>15} | "
        f"{'per-sample mean':>17} | {'MSE':>10} | {'MAE':>10}"
    )
    print("-" * 90)
    for r in rows:
        print(
            f"{int(r['resolution']):>6} | {int(r['n']):>5} | "
            f"{r['global_rel_l2']:>15.4f} | {r['per_sample_mean_rel_l2']:>17.4f} | "
            f"{r['mse']:>10.4f} | {r['mae']:>10.4f}"
        )
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
