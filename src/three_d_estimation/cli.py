"""Command-line interface for standalone surface MLE replay and reporting."""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Sequence

from runtime.measurement_log import load_measurement_log

from .conformance import compute_forward_conformance, save_forward_conformance
from .config import MLEConfig
from .future_scoring import (
    save_future_candidate_scores,
    score_future_count_candidates,
)
from .replay import run_replay
from .reporting import load_mle_estimate, save_mle_estimate


ROOT = Path(__file__).resolve().parents[2]


def _add_fit_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by count and spectral replay commands."""
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Versioned measurement-log directory.",
    )
    parser.add_argument(
        "--mle-config", type=Path, default=None, help="MLE JSON configuration file."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Result directory (default: RUN_DIR/mle_count or mle_spectral).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing result directory."
    )
    device = parser.add_mutually_exclusive_group()
    device.add_argument(
        "--gpu", action="store_true", help="Use the local CUDA kernel when available."
    )
    device.add_argument(
        "--cpu", action="store_true", help="Force CPU response construction."
    )
    parser.add_argument(
        "--no-debias",
        action="store_true",
        help="Disable the support-selected unregularized refit.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the fit summary as JSON."
    )
    parser.add_argument(
        "--initial-estimate",
        type=Path,
        default=None,
        help="Prior station-complete MLE report used only as a warm initialization.",
    )


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
    report_parser.add_argument(
        "--estimate",
        type=Path,
        required=True,
        help="Result directory or mle_estimate.npz.",
    )
    report_parser.add_argument("--json", action="store_true", help="Print JSON output.")
    conformance_parser = subparsers.add_parser(
        "forward-conformance",
        help="Generate canonical unit-strength forward-response cases.",
    )
    conformance_parser.add_argument(
        "--axes",
        type=Path,
        default=ROOT / "fixtures" / "forward_response_conformance.json",
        help="Provider-neutral forward-response axes JSON.",
    )
    conformance_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination NPZ containing case_ids and unit_response.",
    )
    conformance_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing conformance NPZ.",
    )
    score_parser = subparsers.add_parser(
        "score-future",
        help="Score frozen count-MLE candidates on post-cutoff observations only.",
    )
    score_parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Current station-complete MeasurementLog prefix.",
    )
    score_parser.add_argument(
        "--mle-config",
        type=Path,
        required=True,
        help="Exact count-MLE JSON configuration used by the snapshot report.",
    )
    score_parser.add_argument(
        "--snapshot-estimate",
        type=Path,
        required=True,
        help="Earlier count-MLE report directory or mle_estimate.npz.",
    )
    score_parser.add_argument(
        "--snapshot",
        type=Path,
        required=True,
        help="Earlier MLESnapshot v2 JSON artifact.",
    )
    score_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination future-only score JSON.",
    )
    score_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing score JSON artifact.",
    )
    score_parser.add_argument(
        "--json", action="store_true", help="Print the complete score as JSON."
    )
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


def _estimate_summary(
    estimate: object, output_dir: Path | None = None
) -> dict[str, object]:
    """Return a compact JSON-safe estimate summary."""
    diagnostics = dict(getattr(estimate, "diagnostics"))
    clusters = diagnostics.get("hotspot_clusters", [])
    provenance = diagnostics.get("provenance", {})
    return {
        "schema_version": 1,
        "estimator_family": diagnostics.get("estimator_family"),
        "estimator_variant": diagnostics.get("estimator_variant"),
        "candidate_domain": diagnostics.get("candidate_domain"),
        "uses_pf_state": diagnostics.get("uses_pf_state"),
        "uses_pf_candidates": diagnostics.get("uses_pf_candidates"),
        "provenance": provenance if isinstance(provenance, dict) else {},
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
    config_source_sha256 = (
        None
        if args.mle_config is None
        else sha256(args.mle_config.read_bytes()).hexdigest()
    )
    replay_result = run_replay(
        args.run_dir,
        config=config,
        config_source_sha256=config_source_sha256,
        initial_estimate_path=args.initial_estimate,
    )
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


def _run_forward_conformance(args: argparse.Namespace) -> int:
    """Generate all canonical local forward responses and report their count."""
    result = compute_forward_conformance(args.axes)
    output = save_forward_conformance(
        args.output,
        result,
        overwrite=bool(args.overwrite),
    )
    print(f"cases: {result.case_ids.size}")
    print(f"output: {output}")
    return 0


def _run_score_future(args: argparse.Namespace) -> int:
    """Run frozen future-only candidate verification and save its artifact."""
    payload = score_future_count_candidates(
        args.run_dir,
        config=args.mle_config,
        snapshot_estimate=args.snapshot_estimate,
        snapshot=args.snapshot,
    )
    output = save_future_candidate_scores(
        args.output,
        payload,
        overwrite=bool(args.overwrite),
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"snapshot_id: {payload['snapshot_id']}")
        print(f"future_steps: {len(payload['future_step_ids'])}")
        print(f"candidates: {len(payload['candidates'])}")
        print(f"output: {output}")
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
    if args.command == "forward-conformance":
        return _run_forward_conformance(args)
    if args.command == "score-future":
        return _run_score_future(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
