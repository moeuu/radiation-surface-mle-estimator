"""CLI for generating RA-L ablation experiment configurations."""

from __future__ import annotations

import argparse
from pathlib import Path

from baselines.ral_ablation.config_factory import (
    DEFAULT_BASE_CONFIG,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE_INTENSITY_RANGE_CPS_1M,
    build_ablation_plan,
    write_ablation_plan,
)


def main() -> None:
    """Generate RA-L ablation config/source files and command manifests."""
    parser = argparse.ArgumentParser(description="Generate RA-L ablation trials.")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=DEFAULT_BASE_CONFIG,
        help="Base Geant4/PF runtime config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated configs, sources, and manifests.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[2026050901],
        help="Obstacle/source seed list.",
    )
    parser.add_argument(
        "--intensity-cps-1m",
        type=float,
        default=None,
        help="Fixed detector cps@1m assigned to generated sources.",
    )
    parser.add_argument(
        "--intensity-min-cps-1m",
        type=float,
        default=float(DEFAULT_SOURCE_INTENSITY_RANGE_CPS_1M[0]),
        help="Minimum detector cps@1m for uniformly sampled generated sources.",
    )
    parser.add_argument(
        "--intensity-max-cps-1m",
        type=float,
        default=float(DEFAULT_SOURCE_INTENSITY_RANGE_CPS_1M[1]),
        help="Maximum detector cps@1m for uniformly sampled generated sources.",
    )
    args = parser.parse_args()
    intensity_spec: float | tuple[float, float]
    if args.intensity_cps_1m is not None:
        intensity_spec = float(args.intensity_cps_1m)
    else:
        intensity_spec = (
            float(args.intensity_min_cps_1m),
            float(args.intensity_max_cps_1m),
        )
    entries = build_ablation_plan(
        base_config_path=args.base_config,
        output_dir=args.output_dir,
        seeds=tuple(int(seed) for seed in args.seeds),
        intensity_cps_1m=intensity_spec,
    )
    manifest_path, script_path = write_ablation_plan(entries, output_dir=args.output_dir)
    print(f"Wrote {len(entries)} ablation trials.")
    print(f"Manifest: {manifest_path}")
    print(f"Run script: {script_path}")


if __name__ == "__main__":
    main()
