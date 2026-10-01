"""`neuropaths-generate --config path/to.yaml` -- dataset generation."""

from __future__ import annotations

import argparse

from neuropaths.config import load_config
from neuropaths.data import generate_dataset
from neuropaths.utils import seed_everything


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate committor training data.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML.")
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
        help="Which split to generate; picks data.train_csv or data.test_csv.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Parallel PDE solves via multiprocessing.Pool. "
            "1 = serial (default); -1 = all cores (os.cpu_count()). "
            "Output CSV is byte-identical across worker counts for a given seed."
        ),
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    seed_everything(cfg.data.seed)

    output = cfg.data.train_csv if args.split == "train" else cfg.data.test_csv
    path = generate_dataset(
        cfg.pde,
        cfg.data,
        split=args.split,
        output_path=output,
        num_workers=args.num_workers,
    )
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
