"""Command-line interface for standalone surface MLE replay and reporting."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

from runtime.measurement_log import load_measurement_log

from .config import MLEConfig
from .replay import run_replay
from .reporting import load_mle_estimate, save_mle_estimate


ROOT = Path(__file__).resolve().parents[2]


def _add_fit_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by count and spectral replay commands."""
    parser.add_argument("--run-dir", type=Path, required=True, help="Versioned measurement-log directory.")
    parser.add_argument("--mle-config", type=Path, default=None, help="MLE JSON configuration file.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Result directory (default: RUN_DIR/mle_count or mle_spectral).")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing result directory.")
    device = parser.add_mutually_exclusive_group()
    device.add_argument("--gpu", action="store_true", help="Use the local CUDA kernel when available.")
    device.add_argument("--cpu", action="store_true", help="Force CPU response construction.")
    parser.add_argument("--no-debias", action="store_true", help="Disable the support-selected unregularized refit.")
    parser.add_argument("--json", action="store_true", help="Print the fit summary as JSON.")


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the top-level replay/report command parser."""
    parser = argparse.ArgumentParser(
        prog="estimate-radiation-mle",
        description="Standalone rotating-shield surface maximum-likelihood estimation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    replay_parser = subparsers.add_parser(
        "replay",
        help="Fit count-domain MLE from response_poisson isotope counts.",
    )
    _add_fit_arguments(replay_parser)
    spectral_parser = subparsers.add_parser(
        "fit-spectrum",
        help="Fit line-resolved MLE directly from raw spectra.",
    )
    _add_fit_arguments(spectral_parser)
    report_parser = subparsers.add_parser(
        "report",
        help="Read a saved MLE estimate and print its summary.",
    )
    report_parser.add_argument("--estimate", type=Path, required=True, help="Result directory or mle_estimate.npz.")
    report_parser.add_argument("--json", action="store_true", help="Print JSON output.")
    return parser


def _config_for_command(args: argparse.Namespace, mode: str) -> MLEConfig:
    """Load or derive a configuration and apply explicit CLI overrides."""
    if args.mle_config is None:
        log = load_measurement_log(args.run_dir)
        config = MLEConfig(mode=mode, isotope_names=log.context.isotopes)
    else:
        config = MLEConfig.load(args.mle_config)
        config = replace(config, mode=mode)
    if args.gpu:
        config = replace(config, use_gpu=True)
    elif args.cpu:
        config = replace(config, use_gpu=False)
    if args.no_debias:
        config = replace(config, debias_refit=False)
    return config


def _estimate_summary(estimate: object, output_dir: Path | None = None) -> dict[str, object]:
    """Return a compact JSON-safe estimate summary."""
    diagnostics = dict(getattr(estimate, "diagnostics"))
    clusters = diagnostics.get("hotspot_clusters", [])
    return {
        "mode": diagnostics.get("mode"),
        "isotopes": list(getattr(estimate, "isotope_names")),
        "patch_count": len(getattr(estimate, "patches")),
        "objective": float(getattr(estimate, "objective_value")),
        "poisson_deviance": float(getattr(estimate, "poisson_deviance")),
        "iterations": int(getattr(estimate, "iterations")),
        "converged": bool(getattr(estimate, "converged")),
        "cluster_count": len(clusters) if isinstance(clusters, list) else 0,
        "output_dir": None if output_dir is None else str(output_dir),
    }


def _run_fit(args: argparse.Namespace, mode: str) -> int:
    """Run one replay command, save all outputs, and print a summary."""
    config = _config_for_command(args, mode)
    replay_result = run_replay(args.run_dir, config=config)
    estimate = replay_result.estimate
    output_dir = args.output_dir or args.run_dir / f"mle_{mode}"
    save_mle_estimate(
        output_dir,
        estimate,
        config=config,
        overwrite=bool(args.overwrite),
    )
    summary = _estimate_summary(estimate, output_dir)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"mode: {summary['mode']}")
        print(f"patches: {summary['patch_count']}")
        print(f"objective: {summary['objective']:.8g}")
        print(f"poisson_deviance: {summary['poisson_deviance']:.8g}")
        print(f"iterations: {summary['iterations']}")
        print(f"converged: {summary['converged']}")
        print(f"output_dir: {summary['output_dir']}")
    return 0


def _run_report(args: argparse.Namespace) -> int:
    """Load a saved estimate and print its summary without refitting."""
    estimate = load_mle_estimate(args.estimate)
    estimate_path = Path(args.estimate)
    output_dir = estimate_path if estimate_path.is_dir() else estimate_path.parent
    summary = _estimate_summary(estimate, output_dir)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        for key, value in summary.items():
            print(f"{key}: {value}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and execute the requested standalone operation."""
    parser = build_argument_parser()
    args = parser.parse_args(None if argv is None else list(argv))
    if args.command == "replay":
        return _run_fit(args, "count")
    if args.command == "fit-spectrum":
        return _run_fit(args, "spectral")
    if args.command == "report":
        return _run_report(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
