"""CLI entry point for the baseline PF demo."""

from __future__ import annotations

import argparse
from pathlib import Path

from baselines.legacy_no_shield.realtime_demo import (
    DEFAULT_ENVIRONMENT_MODE,
    DEFAULT_OBSTACLE_CONFIG,
    DEFAULT_SOURCE_CONFIG,
    load_sources_from_json,
    run_baseline_pf,
)


def main() -> None:
    """Parse CLI arguments and run the baseline PF demo."""
    parser = argparse.ArgumentParser(
        description="Baseline particle filter demo (no shielding)."
    )
    parser.add_argument(
        "--total-time-s",
        "--measurement-time",
        dest="total_time_s",
        type=float,
        default=300.0,
        help="Total measurement time (seconds). Each measurement uses 30 s.",
    )
    parser.add_argument(
        "--no-live",
        action="store_true",
        help=(
            "Disable interactive updating (still saves results/baselines/legacy_no_shield/* images)."
        ),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help=(
            "Run without an interactive window (forces Agg backend and "
            "disables live updates)."
        ),
    )
    parser.add_argument(
        "--source-config",
        type=str,
        default=DEFAULT_SOURCE_CONFIG.as_posix(),
        help="Path to a JSON file that defines the point sources.",
    )
    parser.add_argument(
        "--obstacle-config",
        type=str,
        default=DEFAULT_OBSTACLE_CONFIG.as_posix(),
        help="Path to a JSON file that defines blocked grid cells.",
    )
    parser.add_argument(
        "--environment-mode",
        type=str,
        default=DEFAULT_ENVIRONMENT_MODE,
        choices=("fixed", "random"),
        help=(
            "Environment generation mode: fixed loads the obstacle JSON, "
            "random creates a fresh obstacle layout at startup "
            f"(default: {DEFAULT_ENVIRONMENT_MODE})."
        ),
    )
    parser.add_argument(
        "--obstacle-seed",
        type=int,
        default=None,
        help="RNG seed used when creating a fixed missing layout or a random startup layout.",
    )
    parser.add_argument(
        "--no-obstacles",
        action="store_true",
        help="Disable obstacles during pose selection and visualization.",
    )
    parser.add_argument(
        "--detect-threshold-abs",
        type=float,
        default=30.0,
        help="Absolute detection threshold for peak-matched activity (counts).",
    )
    parser.add_argument(
        "--detect-threshold-rel",
        type=float,
        default=0.2,
        help="Relative detection threshold as a fraction of max peak-matched activity.",
    )
    parser.add_argument(
        "--eval-match-radius",
        type=float,
        default=0.5,
        help="Match radius (m) for evaluation metrics.",
    )
    parser.add_argument(
        "--converge",
        action="store_true",
        help="Enable per-isotope convergence gating (default: disabled).",
    )
    parser.add_argument(
        "--blender-executable",
        type=str,
        default=None,
        help="Blender executable path used by --environment-mode random.",
    )
    parser.add_argument(
        "--blender-output",
        type=str,
        default=None,
        help="Optional USD output path for the Blender-generated random environment.",
    )
    parser.add_argument(
        "--blender-timeout-s",
        type=float,
        default=120.0,
        help="Timeout for Blender random environment generation.",
    )
    args = parser.parse_args()

    sources = None
    if args.source_config:
        source_path = Path(args.source_config)
        if not source_path.is_absolute():
            repo_root = DEFAULT_SOURCE_CONFIG.parent.parent
            source_path = repo_root / source_path
        if source_path.exists():
            try:
                sources = load_sources_from_json(source_path)
                print(f"Loaded {len(sources)} sources from {source_path}")
            except (OSError, ValueError) as exc:
                print(f"Failed to load sources from {source_path}: {exc}")
        else:
            print(
                f"Source config not found: {source_path}. "
                "Using built-in demo sources."
            )

    run_baseline_pf(
        live=not (args.no_live or args.headless),
        total_time_s=args.total_time_s,
        sources=sources,
        environment_mode=args.environment_mode,
        obstacle_layout_path=None if args.no_obstacles else args.obstacle_config,
        obstacle_seed=args.obstacle_seed,
        detect_threshold_abs=args.detect_threshold_abs,
        detect_threshold_rel=args.detect_threshold_rel,
        eval_match_radius_m=args.eval_match_radius,
        converge=args.converge,
        blender_executable=args.blender_executable,
        blender_output_path=args.blender_output,
        blender_timeout_s=args.blender_timeout_s,
    )


if __name__ == "__main__":
    main()
