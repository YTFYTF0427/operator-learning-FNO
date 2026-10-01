"""Visualise sample random trig-polynomial boundary draws.

Sanity check / debugging tool. Draws ``--n`` random (phi, psi) pairs
via ``generate_boundary_pair(kind="curved")`` and lays them out as a
grid of small panels showing the resulting domain shape Omega = {(x,y)
: phi(y) <= x <= psi(y), 0 <= y <= 1}.

Usage:

    uv run python scripts/plot_boundaries.py --n 12 --output figs/boundaries.png

    # Override the trig-polynomial settings:
    uv run python scripts/plot_boundaries.py --n-max 4 --eps 0.05 --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neuropaths.pde.boundaries import generate_boundary_pair


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Plot random trig-polynomial domain boundaries.")
    p.add_argument("--n", type=int, default=12,
                   help="Number of random boundary pairs to draw.")
    p.add_argument("--n-max", type=int, default=8,
                   help="Trig-polynomial mode cap (Algorithm 1's N_max).")
    p.add_argument("--eps", type=float, default=0.1,
                   help="Minimum domain gap inf_y |psi - phi|.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for reproducibility.")
    p.add_argument("--output", type=Path, default=Path("boundaries.png"),
                   help="Output PNG path.")
    args = p.parse_args(argv)

    seed_seq = np.random.SeedSequence(args.seed)
    child_seeds = seed_seq.spawn(args.n)

    cols = min(4, args.n)
    rows = (args.n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.6 * rows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    y = np.linspace(0.0, 1.0, 200)
    for i in range(args.n):
        rng = np.random.default_rng(child_seeds[i])
        phi, psi = generate_boundary_pair(
            rng, kind="curved", n_max=args.n_max, eps=args.eps,
        )
        phi_y = np.asarray(phi(y))
        psi_y = np.asarray(psi(y))

        ax = axes[i]
        ax.fill_betweenx(y, phi_y, psi_y, alpha=0.25, color="C0")
        ax.plot(phi_y, y, "k-", linewidth=1.5)
        ax.plot(psi_y, y, "k-", linewidth=1.5)
        # Mark x=0, x=1 reference lines so the deformation is visible.
        ax.axvline(0.0, color="grey", linewidth=0.5, alpha=0.4)
        ax.axvline(1.0, color="grey", linewidth=0.5, alpha=0.4)
        ax.set_xlim(min(phi_y.min(), 0.0) - 0.05, max(psi_y.max(), 1.0) + 0.05)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal")
        ax.set_title(f"seed {args.seed}+{i}", fontsize=9)

    # Hide any unused panels in the last row.
    for j in range(args.n, len(axes)):
        axes[j].axis("off")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}  ({args.n} boundary pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
