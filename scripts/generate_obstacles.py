"""Generate an obstacle layout JSON without running the PF demo."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from measurement.obstacles import generate_obstacle_grid


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate an obstacle layout JSON file with the standard connected "
            "exploration backbone."
        )
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the obstacle JSON (relative to repo root if not absolute).",
    )
    parser.add_argument(
        "--size-x",
        type=float,
        default=10.0,
        help="Environment size in x (meters).",
    )
    parser.add_argument(
        "--size-y",
        type=float,
        default=20.0,
        help="Environment size in y (meters).",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=1.0,
        help="Grid cell size (meters).",
    )
    parser.add_argument(
        "--blocked-fraction",
        type=float,
        default=0.4,
        help="Fraction of blocked grid cells (0.0 to 1.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for obstacle generation.",
    )
    return parser


def main() -> None:
    """Generate and save an obstacle layout file."""
    parser = build_parser()
    args = parser.parse_args()
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = (ROOT / out_path).resolve()
    rng = np.random.default_rng(args.seed)
    grid = generate_obstacle_grid(
        size_x=args.size_x,
        size_y=args.size_y,
        cell_size=args.cell_size,
        blocked_fraction=args.blocked_fraction,
        rng=rng,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    print(f"Saved obstacle layout to {out_path}")


if __name__ == "__main__":
    main()
