"""Command-line entry point for the Kemp comparison baseline."""

from __future__ import annotations

import argparse

from baselines.kemp.runner import KempRunConfig, run_kemp_full_simulation
from runtime_defaults import (
    DEFAULT_MAX_SOURCES_PER_ISOTOPE,
    DEFAULT_MEASUREMENT_TIME_S,
)


def _parse_grid_spacing(value: str) -> tuple[float, float, float]:
    """Parse a comma-separated 3D grid spacing value."""
    parts = [float(part.strip()) for part in str(value).split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("grid spacing must be 'x,y,z'.")
    return (parts[0], parts[1], parts[2])


def _parse_z_levels(value: str) -> tuple[float, ...]:
    """Parse comma-separated source z levels."""
    parts = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("at least one z level is required.")
    return tuple(parts)


def build_parser() -> argparse.ArgumentParser:
    """Build the Kemp baseline CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run the Kemp et al. parallel log-domain DDPF baseline."
    )
    parser.add_argument("--sim-backend", default="geant4", choices=("geant4", "analytic"))
    parser.add_argument(
        "--sim-config",
        dest="sim_config_path",
        default="configs/geant4/variance_reduction_external_no_isaac_32threads.json",
    )
    parser.add_argument("--source-config", dest="source_config_path", default=None)
    parser.add_argument("--obstacle-config", dest="obstacle_config_path", default=None)
    parser.add_argument("--output-dir", default="results/baselines/kemp/latest")
    parser.add_argument("--max-poses", type=int, default=10)
    parser.add_argument("--dwell-time-s", type=float, default=DEFAULT_MEASUREMENT_TIME_S)
    parser.add_argument("--measurement-spacing-m", type=float, default=4.0)
    parser.add_argument("--num-particles", type=int, default=2000)
    parser.add_argument("--max-sources", type=int, default=DEFAULT_MAX_SOURCES_PER_ISOTOPE)
    parser.add_argument("--grid-spacing-m", type=_parse_grid_spacing, default=(0.5, 0.5, 0.5))
    parser.add_argument(
        "--grid-z-levels-m",
        type=_parse_z_levels,
        default=(0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5),
    )
    parser.add_argument("--eval-match-radius-m", type=float, default=1.0)
    parser.add_argument("--rng-seed", type=int, default=20260502)
    parser.add_argument("--shield-fe-index", type=int, default=0)
    parser.add_argument("--shield-pb-index", type=int, default=0)
    return parser


def main() -> None:
    """Parse CLI arguments and execute the Kemp baseline run."""
    args = build_parser().parse_args()
    config = KempRunConfig(
        sim_backend=args.sim_backend,
        sim_config_path=args.sim_config_path,
        source_config_path=args.source_config_path,
        obstacle_config_path=args.obstacle_config_path,
        output_dir=args.output_dir,
        max_poses=args.max_poses,
        dwell_time_s=args.dwell_time_s,
        measurement_spacing_m=args.measurement_spacing_m,
        num_particles=args.num_particles,
        max_sources=args.max_sources,
        grid_spacing_m=args.grid_spacing_m,
        grid_z_levels_m=args.grid_z_levels_m,
        eval_match_radius_m=args.eval_match_radius_m,
        rng_seed=args.rng_seed,
        shield_fe_index=args.shield_fe_index,
        shield_pb_index=args.shield_pb_index,
    )
    run_kemp_full_simulation(config)


if __name__ == "__main__":
    main()
