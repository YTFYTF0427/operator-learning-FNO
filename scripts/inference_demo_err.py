"""Generate one held-out PDE solution and compare against a checkpoint.

Usage:

    uv run python scripts/inference_demo.py --config configs/square_32.yaml
    uv run python scripts/inference_demo.py --config configs/square_32.yaml \
        --seed 999 --output runs/square_32/inference_demo.png

Pipeline:

    1. Re-derive the same FD pipeline the generator uses (turbulent
       velocity field -> q^+ / q^- -> rho_react -> coarse subsample).
       A user-supplied seed (default 999) keeps this draw distinct from
       any training/test sample.
    2. Load the trained FNO via ``neuralop.models.FNO.from_checkpoint``
       (so the model rebuilds itself from the saved init kwargs and
       doesn't need ModelConfig).
    3. Run a forward pass on the coarse-grid input.
    4. Render a 3-panel figure: |b| (input), ground-truth rho, predicted
       rho. Ground truth and prediction share a colour scale so their
       difference is visually honest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from neuralop.models import FNO

from neuropaths.config.loader import load_config
from neuropaths.data.generator import _generate_one_solution, _SolutionJob
from neuropaths.pde.boundaries import generate_boundary_pair


def _input_target_from_rows(
    rows: list[list[float]], coarse_grid: int, include_finv: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Reshape a generator's row list into (C, G, G) input + (G, G) rho."""
    arr = np.asarray(rows, dtype=np.float32)
    # Columns: solution_id, x, y, b1, b2, rho, [finv]
    # The generator emits rows in (i, j) lexicographic order on the
    # transformed coarse grid, matching meshgrid(indexing='ij').
    G = coarse_grid
    arr = arr.reshape(G, G, -1)
    b1 = arr[..., 3]
    b2 = arr[..., 4]
    rho = arr[..., 5]
    channels = [b1, b2]
    if include_finv:
        channels.append(arr[..., 6])
    inp = np.stack(channels, axis=0)  # (C, G, G)
    return inp, rho


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run a held-out PDE through a trained FNO and plot.")
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=1003,
                   help="Independent seed for this demo PDE (default 1003).")
    p.add_argument("--checkpoint", default=None,
                   help="Override checkpoint path prefix (default: cfg.eval.checkpoint_path).")
    p.add_argument("--output", type=Path, default=None,
                   help="Output PNG path (default: <output_dir>/inference_demo.png).")
    args = p.parse_args(argv)

    cfg = load_config(args.config)

    # Reuse the generator's worker so the data pipeline is identical to
    # what the model saw during training.
    seed_seq = np.random.SeedSequence(args.seed)
    job = _SolutionJob(
        sid=0, pde_cfg=cfg.pde, data_cfg=cfg.data, seed_seq=seed_seq
    )
    print(f"Solving demo PDE (seed={args.seed}) at fine_grid={cfg.pde.fine_grid}...")
    _, rows, rho_max, rho_l2 = _generate_one_solution(job)
    print(f"  rho.max() = {rho_max:.4f},  ||rho||_2 = {rho_l2:.4f}")

    # Recreate the same (phi, psi) the worker used so we can plot the
    # physical boundary curves and warp the transformed grid back to
    # original coordinates. ``np.random.default_rng(seed_seq)`` is
    # deterministic in `seed_seq` so this reproduces the worker's draw
    # exactly. For square domains phi=0, psi=1 and the warped grid is
    # the unit square; the same rendering code handles both.
    phi, psi = generate_boundary_pair(
        np.random.default_rng(seed_seq),
        kind=cfg.pde.domain,
        n_max=cfg.pde.boundary_n_max,
        eps=cfg.pde.boundary_eps,
    )

    inp_np, target_np = _input_target_from_rows(
        rows, cfg.pde.coarse_grid, cfg.data.include_finv_column
    )

    # Load the trained model. neuralop's BaseModel.from_checkpoint takes
    # (folder, name) and rebuilds the architecture from the metadata file.
    ckpt_path = Path(args.checkpoint or cfg.eval.checkpoint_path)
    print(f"Loading checkpoint from prefix {ckpt_path}...")
    model = FNO.from_checkpoint(ckpt_path.parent, ckpt_path.stem, map_location="cpu")
    model.eval()

    # Apply the same per-channel normalisation that was used at training
    # time. CLI train writes <stem>_stats.npz next to the checkpoint.
    stats_path = ckpt_path.parent / f"{ckpt_path.stem}_stats.npz"
    if stats_path.exists():
        stats = np.load(stats_path)
        mean = stats["mean"].reshape(-1, 1, 1)
        std = stats["std"].reshape(-1, 1, 1)
        # inp_np = (inp_np - mean) / std
        print(f"Applied normalisation from {stats_path}.")
    else:
        print(
            f"WARNING: no normalisation stats at {stats_path}; running unnormalised. "
            f"This is only correct if the checkpoint was trained without normalisation."
        )

    with torch.no_grad():
        inp_t = torch.from_numpy(inp_np.astype(np.float32)).unsqueeze(0)  # (1, C, G, G)
        pred_t = model(inp_t)
    pred_np = pred_t.squeeze().cpu().numpy()

    rel_l2 = np.linalg.norm(pred_np - target_np) / max(
        np.linalg.norm(target_np), 1e-12
    )
    abs_err = np.abs(pred_np - target_np).mean()
    print(f"Demo metrics: rel L2 = {rel_l2:.4f}  mean |err| = {abs_err:.4f}")

    # Render a three-panel comparison in *physical* (x', y) coordinates.
    # The dataset stores values on a uniform (x_hat, y) grid (transformed
    # coords); we warp to physical x' = phi(y) + x_hat * (psi(y) - phi(y))
    # and use pcolormesh to render on the resulting non-rectilinear grid.
    # For square domains (phi=0, psi=1) the warp is the identity.
    G = cfg.pde.coarse_grid
    x_hat_axis = np.linspace(0.0, 1.0, G)
    y_axis = np.linspace(0.0, 1.0, G)
    X_hat, Y = np.meshgrid(x_hat_axis, y_axis, indexing="ij")
    phi_Y = np.asarray(phi(Y))
    psi_Y = np.asarray(psi(Y))
    X_phys = phi_Y + X_hat * (psi_Y - phi_Y)

    bmag = np.sqrt(inp_np[0] ** 2 + inp_np[1] ** 2)
    vmin = float(min(target_np.min(), pred_np.min()))
    vmax = float(max(target_np.max(), pred_np.max()))

    # Boundary curves for overlay.
    y_curve = np.linspace(0.0, 1.0, 200)
    phi_curve = np.asarray(phi(y_curve))
    psi_curve = np.asarray(psi(y_curve))
    is_curved = cfg.pde.domain == "curved"

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True)

    im1 = axes[0].pcolormesh(X_phys, Y, target_np, cmap="coolwarm",
                             vmin=vmin, vmax=vmax, shading="auto")
    axes[0].set_title(r"$\rho_{\mathrm{react}}$ (FD ground truth)")
    fig.colorbar(im1, ax=axes[0], shrink=0.85)

    im2 = axes[1].pcolormesh(X_phys, Y, pred_np, cmap="coolwarm",
                             vmin=vmin, vmax=vmax, shading="auto")
    axes[1].set_title(rf"$\hat\rho$  (rel L2 = {rel_l2:.3f})")

    im3 = axes[2].pcolormesh(X_phys, Y, target_np-pred_np, cmap="coolwarm",
                             shading="auto")
    axes[2].set_title(rf"difference")
    fig.colorbar(im3, ax=axes[2], shrink=0.85)

    # For curved domains, overlay the A (left) and B (right) boundaries
    # so the irregular shape is visible.
    if is_curved:
        for ax in axes:
            ax.plot(phi_curve, y_curve, "k-", linewidth=1.5, label="A: $\\varphi$")
            ax.plot(psi_curve, y_curve, "k-", linewidth=1.5, label="B: $\\psi$")
            ax.set_xlim(min(phi_curve.min(), 0.0) - 0.02,
                        max(psi_curve.max(), 1.0) + 0.02)

    for ax in axes:
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_aspect("equal")

    out = args.output or (Path(cfg.output_dir) / "inference_demo.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 1, figsize=(5.0, 4.5), constrained_layout=True)
    im0 = axes.pcolormesh(X_phys, Y, bmag, cmap="viridis", shading="auto")
    axes.set_title(r"$|b|$ (input magnitude)")
    fig.colorbar(im0, ax=axes, shrink=1.0)
    out = args.output or (Path(cfg.output_dir) / "inference_v.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)

    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
