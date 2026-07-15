"""Create a deterministic sampled-observation log for the surface MLE demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from three_d_estimation.demo import (  # noqa: E402
    build_analytic_mle_demo,
    create_analytic_mle_demo_log,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the deterministic demo-log command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory that will contain the versioned measurement log.",
    )
    parser.add_argument(
        "--mle-config-output",
        type=Path,
        default=None,
        help="Optional path for the demo's recommended count-MLE JSON config.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable creation summary.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate sampled observations, persist the log, and print its summary."""
    args = build_argument_parser().parse_args(
        None if argv is None else list(argv)
    )
    scenario = build_analytic_mle_demo()
    run_dir = create_analytic_mle_demo_log(
        args.output_dir,
        scenario=scenario,
    )
    config_path = args.mle_config_output
    if config_path is not None:
        scenario.mle_config.save(config_path)

    source_patch = scenario.patches.patches[scenario.source_patch_index]
    summary: dict[str, object] = {
        "run_dir": str(run_dir),
        "record_count": len(scenario.records),
        "sim_backend": scenario.context.sim_backend,
        "spectrum_count_method": scenario.context.spectrum_count_method,
        "observation_generation": "sampled_analytic_runtime_spectrum",
        "source_isotope": scenario.context.isotopes[0],
        "source_patch_id": int(scenario.source_patch_id),
        "source_position_xyz": [
            float(value) for value in source_patch.centroid_xyz
        ],
        "source_strength_cps_1m": float(
            scenario.source_strength_cps_1m
        ),
        "route_response_rank": int(scenario.route_response_rank),
        "mle_config": None if config_path is None else str(config_path),
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"run_dir: {summary['run_dir']}")
        print(f"records: {summary['record_count']}")
        print(f"source_patch_id: {summary['source_patch_id']}")
        print(f"source_strength_cps_1m: {summary['source_strength_cps_1m']}")
        if config_path is not None:
            print(f"mle_config: {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
