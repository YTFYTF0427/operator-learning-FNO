"""Plot the training / validation loss curves from a SLURM log.

Usage:

    uv run python scripts/plot_training_curve.py slurm/logs/neuropaths-train-12345.out
    uv run python scripts/plot_training_curve.py slurm/logs/run.out -o curve.png

Parses the per-epoch lines that ``neuralop.Trainer(verbose=True)`` prints:

    [0] time=5.07, avg_loss=2564.8236, train_err=4274.7059
    Eval: val_l2=6.9625

and writes a two-line plot (train vs val) on a log y-scale next to the
log file (or to ``-o`` if given).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt

# Lines the trainer emits for the train and val metrics each epoch.
_TRAIN_RE = re.compile(r"^\[(\d+)\][^,]*,[^,]*,\s*train_err=([0-9.eE+\-]+)")
_VAL_RE = re.compile(r"^Eval:\s*val_l2=([0-9.eE+\-]+)")


def parse_log(text: str) -> tuple[list[int], list[float], list[float]]:
    """Return (epochs, train_err, val_l2) parsed from `text`."""
    epochs: list[int] = []
    train_err: list[float] = []
    val_l2: list[float] = []
    pending_epoch: int | None = None
    for line in text.splitlines():
        if (m := _TRAIN_RE.match(line)) is not None:
            pending_epoch = int(m.group(1))
            epochs.append(pending_epoch)
            train_err.append(float(m.group(2)))
        elif (m := _VAL_RE.match(line)) is not None and pending_epoch is not None:
            val_l2.append(float(m.group(1)))
    if not epochs:
        raise ValueError(
            "no epoch lines (`[N] ... train_err=...`) found. Was the trainer "
            "run with verbose=True?"
        )
    return epochs, train_err, val_l2


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Plot training/val loss from a SLURM log.")
    p.add_argument("log", type=Path, help="Path to the SLURM .out log.")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output PNG path (default: <log>.png).")
    args = p.parse_args(argv)

    text = args.log.read_text()
    epochs, train_err, val_l2 = parse_log(text)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(epochs, train_err, marker="o", ms=3, label="train (rel L2)")
    if val_l2:
        # val may be reported less frequently than train; align by epoch index.
        val_epochs = epochs[: len(val_l2)]
        ax.plot(val_epochs, val_l2, marker="s", ms=3, label="val (rel L2)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("relative L2 loss")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_title(args.log.name)

    out = args.output or args.log.with_suffix(".png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}  ({len(epochs)} train epochs, {len(val_l2)} val points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
