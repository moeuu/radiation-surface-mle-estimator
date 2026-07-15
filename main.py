"""CLI entry point for the real-time PF demo."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure src/ is on sys.path for direct script execution.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realtime_demo import (
    DEFAULT_OBSTACLE_CONFIG,
    DEFAULT_SOURCE_CONFIG,
    load_sources_from_json,
    run_live_pf,
)
from piplup_notify import PIPLUP_DEFAULT_BASE_URL, PiplupNotificationConfig
from runtime_defaults import (
    DEFAULT_ENVIRONMENT_MODE,
    DEFAULT_MAX_SOURCES_PER_ISOTOPE,
    DEFAULT_MEASUREMENT_TIME_S,
    DEFAULT_RANDOM_SOURCE_COUNT,
    DEFAULT_RANDOM_SOURCE_INTENSITY_CPS_1M,
    DEFAULT_ROBOT_SPEED_M_S,
    DEFAULT_ROTATION_OVERHEAD_S,
)

STANDARD_GEANT4_FULL_CONFIG = (
    ROOT
    / "configs"
    / "geant4"
    / "variance_reduction_external_no_isaac_32threads.json"
)
STANDARD_GEANT4_GUI_CONFIG = (
    ROOT
    / "configs"
    / "geant4"
    / "variance_reduction_external_gui_32threads.json"
)

RUN_MODE_ALIASES = {
    "gui": "geant4-isaacsim-gui",
    "cui": "geant4-cui",
    "full-simulation": "geant4-cui",
    "standard-geant4-full": "geant4-cui",
}
RUN_MODE_DEFAULTS = {
    "python-gui": {
        "sim_backend": "isaacsim",
        "sim_config": (
            ROOT / "configs" / "isaacsim" / "demo_room_gui.json"
        ).as_posix(),
        "matplotlib_live": False,
    },
    "geant4-isaacsim-gui": {
        "sim_backend": "geant4",
        "sim_config": STANDARD_GEANT4_GUI_CONFIG.as_posix(),
        "matplotlib_live": False,
    },
    "python-cui": {
        "sim_backend": "analytic",
        "sim_config": (
            ROOT / "configs" / "python" / "high_fidelity_no_isaac.json"
        ).as_posix(),
        "matplotlib_live": False,
    },
    "geant4-cui": {
        "sim_backend": "geant4",
        "sim_config": STANDARD_GEANT4_FULL_CONFIG.as_posix(),
        "matplotlib_live": False,
    },
}
RUN_MODE_CHOICES = tuple(RUN_MODE_DEFAULTS.keys()) + tuple(RUN_MODE_ALIASES.keys())
RANDOM_SOURCE_CONFIG_TOKENS = {"random", "surface-random", "surface_random"}


def _normalize_run_mode(mode: str | None) -> str:
    """Return the canonical execution mode name."""
    raw_mode = "geant4-cui" if mode is None else mode.strip().lower()
    return RUN_MODE_ALIASES.get(raw_mode, raw_mode)


def _default_run_mode_for_backend(backend: str | None) -> str:
    """Return the default run mode for an explicitly requested backend."""
    if backend is None:
        return "geant4-cui"
    backend_name = backend.strip().lower()
    if backend_name == "analytic":
        return "python-cui"
    if backend_name == "isaacsim":
        return "python-gui"
    return "geant4-cui"


def _resolve_run_settings(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> tuple[str, str, str | None, bool]:
    """Resolve high-level run mode into backend, config, and live-plot settings."""
    selected_mode = args.run_mode
    if selected_mode is None:
        selected_mode = _default_run_mode_for_backend(args.sim_backend)
    run_mode = _normalize_run_mode(selected_mode)
    if run_mode not in RUN_MODE_DEFAULTS:
        parser.error(f"Unknown run mode: {args.run_mode}")
    mode_is_sim_gui = run_mode.endswith("-gui")
    if mode_is_sim_gui and args.headless:
        parser.error(f"--mode {run_mode} cannot be combined with --headless.")
    defaults = RUN_MODE_DEFAULTS[run_mode]
    sim_backend = args.sim_backend or str(defaults["sim_backend"])
    sim_config = args.sim_config
    if sim_config is None:
        default_config = defaults["sim_config"]
        sim_config = None if default_config is None else str(default_config)
    matplotlib_live = bool(defaults["matplotlib_live"])
    if args.matplotlib_live:
        matplotlib_live = True
    if args.no_live or args.headless:
        matplotlib_live = False
    return run_mode, sim_backend, sim_config, matplotlib_live


def _source_config_was_explicit(argv: list[str] | None = None) -> bool:
    """Return whether the CLI explicitly set --source-config."""
    raw_args = sys.argv[1:] if argv is None else argv
    return any(arg == "--source-config" or arg.startswith("--source-config=") for arg in raw_args)


def main() -> None:
    """Parse CLI arguments and run the real-time PF demo."""
    parser = argparse.ArgumentParser(
        description="Real-time rotating-shield PF visualization demo."
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--mode",
        dest="run_mode",
        type=str,
        default=None,
        choices=RUN_MODE_CHOICES,
        help=(
            "Execution mode: python-gui, geant4-isaacsim-gui, "
            "python-cui, or geant4-cui. Default: geant4-cui."
        ),
    )
    mode_group.add_argument(
        "--gui",
        dest="run_mode",
        action="store_const",
        const="geant4-isaacsim-gui",
        help="Alias for --mode geant4-isaacsim-gui.",
    )
    mode_group.add_argument(
        "--cui",
        dest="run_mode",
        action="store_const",
        const="geant4-cui",
        help=(
            "Alias for the standard no-GUI Geant4 full simulation "
            "(--mode geant4-cui). Use --python-cui for the analytic Python CUI."
        ),
    )
    mode_group.add_argument(
        "--full-simulation",
        "--standard-geant4-full",
        dest="run_mode",
        action="store_const",
        const="geant4-cui",
        help=(
            "Run the standard full Geant4/PF simulation: geant4-cui with "
            "configs/geant4/variance_reduction_external_no_isaac_32threads.json."
        ),
    )
    mode_group.add_argument(
        "--python-gui",
        dest="run_mode",
        action="store_const",
        const="python-gui",
        help="Alias for --mode python-gui.",
    )
    mode_group.add_argument(
        "--geant4-isaacsim-gui",
        dest="run_mode",
        action="store_const",
        const="geant4-isaacsim-gui",
        help="Alias for --mode geant4-isaacsim-gui.",
    )
    mode_group.add_argument(
        "--python-cui",
        dest="run_mode",
        action="store_const",
        const="python-cui",
        help="Alias for --mode python-cui.",
    )
    mode_group.add_argument(
        "--geant4-cui",
        dest="run_mode",
        action="store_const",
        const="geant4-cui",
        help="Alias for --mode geant4-cui.",
    )
    parser.add_argument(
        "--max-steps",
        "--steps",
        dest="max_steps",
        type=int,
        default=None,
        help="Maximum number of measurement steps (default: run until convergence).",
    )
    parser.add_argument(
        "--max-poses",
        type=int,
        default=None,
        help="Maximum number of measurement poses (default: runtime config or no cap).",
    )
    parser.add_argument(
        "--pose-candidates",
        type=int,
        default=64,
        help="Number of candidate poses to generate per step (default: 64).",
    )
    parser.add_argument(
        "--pose-min-dist",
        type=float,
        default=3.0,
        help="Minimum distance (m) from visited poses for candidates (default: 3.0).",
    )
    parser.add_argument(
        "--no-live",
        action="store_true",
        help=(
            "Disable the Matplotlib live plot (still saves results/result_pf.png and "
            "results/result_spectrum.png by default)."
        ),
    )
    parser.add_argument(
        "--matplotlib-live",
        action="store_true",
        help="Open the Matplotlib live plot in addition to the selected simulator mode.",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default=None,
        help="Optional tag appended to result output filenames (ex: ex5 -> result_pf_ex5.png).",
    )
    parser.add_argument(
        "--measurement-log-dir",
        type=str,
        default=None,
        help=(
            "Write a standalone replay log to this new directory. Each finalized "
            "observation is durably staged before the PF update."
        ),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Force a non-GUI simulator mode and disable the Matplotlib live plot.",
    )
    parser.add_argument(
        "--source-config",
        type=str,
        default=DEFAULT_SOURCE_CONFIG.as_posix(),
        help=(
            "Path to a JSON file that defines the point sources, or 'random' "
            "for surface-constrained random sources."
        ),
    )
    parser.add_argument(
        "--source-seed",
        type=int,
        default=None,
        help="RNG seed for surface-constrained random source generation.",
    )
    parser.add_argument(
        "--random-source-count",
        type=int,
        default=DEFAULT_RANDOM_SOURCE_COUNT,
        help="Number of surface-constrained random sources to generate.",
    )
    parser.add_argument(
        "--random-source-isotopes",
        type=str,
        default=None,
        help=(
            "Comma-separated isotope names for surface-constrained random "
            "source generation. Defaults to all isotopes in the spectrum library."
        ),
    )
    parser.add_argument(
        "--random-source-intensity-cps-1m",
        type=float,
        default=DEFAULT_RANDOM_SOURCE_INTENSITY_CPS_1M,
        help="Detector cps@1m assigned to each generated random source.",
    )
    parser.add_argument(
        "--random-source-intensity-min-cps-1m",
        type=float,
        default=None,
        help="Minimum detector cps@1m for uniformly sampled random sources.",
    )
    parser.add_argument(
        "--random-source-intensity-max-cps-1m",
        type=float,
        default=None,
        help="Maximum detector cps@1m for uniformly sampled random sources.",
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
        "--ig-threshold-mode",
        type=str,
        default="relative_pose",
        choices=("absolute", "relative_max", "relative_pose"),
        help="IG threshold mode: absolute or relative to max IG.",
    )
    parser.add_argument(
        "--ig-threshold-rel",
        type=float,
        default=0.02,
        help="Relative IG threshold fraction for dynamic modes.",
    )
    parser.add_argument(
        "--ig-threshold-min",
        type=float,
        default=None,
        help="Minimum IG threshold floor (defaults to config value).",
    )
    parser.add_argument(
        "--detect-threshold-abs",
        type=float,
        default=30.0,
        help="Absolute detection threshold for peak-matched activity (counts).",
    )
    parser.add_argument(
        "--detect-threshold",
        dest="detect_threshold_abs",
        type=float,
        default=argparse.SUPPRESS,
        help="Alias for --detect-threshold-abs.",
    )
    parser.add_argument(
        "--detect-threshold-rel",
        type=float,
        default=0.2,
        help="Relative detection threshold as a fraction of max peak-matched activity.",
    )
    parser.add_argument(
        "--detect-consecutive",
        type=int,
        default=20,
        help="Consecutive detections required to enable an isotope.",
    )
    parser.add_argument(
        "--detect-min-steps",
        type=int,
        default=None,
        help="Minimum steps before locking detected isotopes (defaults to detect_consecutive).",
    )
    parser.add_argument(
        "--eval-match-radius",
        type=float,
        default=0.5,
        help="Match radius (m) for evaluation metrics.",
    )
    parser.add_argument(
        "--birth",
        action="store_true",
        help="Enable birth/death/split/merge moves (default: disabled).",
    )
    parser.add_argument(
        "--num-particles",
        type=int,
        default=2000,
        help="Particle count per isotope filter (default: 2000).",
    )
    parser.add_argument(
        "--merge-prob",
        type=float,
        default=None,
        help="Merge proposal probability when birth/death is enabled (default: 0.05).",
    )
    parser.add_argument(
        "--merge-distance-max",
        type=float,
        default=None,
        help="Max distance (m) to merge nearby sources (default: 0.5).",
    )
    parser.add_argument(
        "--merge-delta-ll-threshold",
        type=float,
        default=None,
        help="Log-likelihood threshold for merge acceptance (default: 0.0).",
    )
    parser.add_argument(
        "--cluster-eps-m",
        type=float,
        default=None,
        help="Clustering epsilon (m) for output estimates (default: 0.8).",
    )
    parser.add_argument(
        "--birth-detector-min-sep-m",
        type=float,
        default=None,
        help="Minimum distance (m) from measured detector poses for birth candidates.",
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        default=DEFAULT_MAX_SOURCES_PER_ISOTOPE,
        help=(
            "Maximum number of sources per isotope. Defaults to "
            f"{DEFAULT_MAX_SOURCES_PER_ISOTOPE}; use an explicit value only "
            "for ablation/debug cases."
        ),
    )
    parser.add_argument(
        "--temper-max-resamples",
        type=int,
        default=2,
        help="Max resamples per observation during tempering (default: 2).",
    )
    parser.add_argument(
        "--no-roughen-on-temper-resample",
        action="store_true",
        help="Disable roughening on resamples triggered inside tempering.",
    )
    parser.add_argument(
        "--roughening-k",
        type=float,
        default=None,
        help="Override roughening coefficient k (optional).",
    )
    parser.add_argument(
        "--min-sigma-pos",
        type=float,
        default=None,
        help="Override minimum roughening sigma (optional).",
    )
    parser.add_argument(
        "--max-sigma-pos",
        type=float,
        default=None,
        help="Override maximum roughening sigma (optional).",
    )
    parser.add_argument(
        "--converge",
        action="store_true",
        help="Enable per-isotope convergence gating (default: disabled).",
    )
    parser.add_argument(
        "--sim-backend",
        type=str,
        default=None,
        choices=("analytic", "isaacsim", "geant4"),
        help="Override the simulation backend selected by --mode.",
    )
    parser.add_argument(
        "--sim-config",
        type=str,
        default=None,
        help="Optional JSON config path for the selected simulation backend.",
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
    parser.add_argument(
        "--passage-width-m",
        type=float,
        default=1.0,
        help="Minimum robot corridor width reserved in random environments.",
    )
    parser.add_argument(
        "--robot-radius-m",
        type=float,
        default=0.35,
        help="Robot footprint radius used for 2D traversability maps.",
    )
    parser.add_argument(
        "--robot-speed",
        type=float,
        default=DEFAULT_ROBOT_SPEED_M_S,
        help="Nominal robot travel speed in m/s used for mission-time accounting.",
    )
    parser.add_argument(
        "--rotation-overhead-s",
        type=float,
        default=DEFAULT_ROTATION_OVERHEAD_S,
        help="Fixed shield actuation overhead per measurement in seconds.",
    )
    parser.add_argument(
        "--measurement-time-s",
        type=float,
        default=DEFAULT_MEASUREMENT_TIME_S,
        help=(
            "Fixed dwell time, or adaptive dwell cap, per measurement in seconds. "
            "Use <=0 with --adaptive-dwell for no dwell cap."
        ),
    )
    parser.add_argument(
        "--adaptive-dwell",
        action="store_true",
        help="Acquire spectra in chunks and stop when isotope counts are reliable.",
    )
    parser.add_argument(
        "--adaptive-dwell-chunk-s",
        type=float,
        default=2.0,
        help="Geant4 dwell duration for each adaptive chunk in seconds.",
    )
    parser.add_argument(
        "--adaptive-min-dwell-s",
        type=float,
        default=2.0,
        help="Minimum accumulated live time before adaptive early stopping.",
    )
    parser.add_argument(
        "--adaptive-ready-min-counts",
        type=float,
        default=100.0,
        help="Minimum extracted count per detected isotope before stopping.",
    )
    parser.add_argument(
        "--adaptive-ready-min-isotopes",
        type=int,
        default=1,
        help="Required detected isotope count for adaptive dwell readiness.",
    )
    parser.add_argument(
        "--adaptive-ready-min-snr",
        type=float,
        default=0.0,
        help=(
            "Optional minimum isotope-count SNR for adaptive dwell readiness; "
            "0 uses the count threshold only."
        ),
    )
    parser.add_argument(
        "--no-adaptive-strength-prior",
        dest="adaptive_strength_prior",
        action="store_false",
        default=True,
        help="Disable count-conditioned PF strength prior adaptation.",
    )
    parser.add_argument(
        "--adaptive-strength-prior-steps",
        type=int,
        default=3,
        help="Number of initial measurements used to adapt PF strength particles.",
    )
    parser.add_argument(
        "--adaptive-strength-prior-min-counts",
        type=float,
        default=3.0,
        help="Poisson count floor used by strength adaptation for weak observations.",
    )
    parser.add_argument(
        "--adaptive-strength-prior-log-sigma",
        type=float,
        default=0.7,
        help="Log-space proposal spread around count-matched PF strengths.",
    )
    parser.add_argument(
        "--pose-min-observation-counts",
        type=float,
        default=None,
        help=(
            "Minimum posterior-predicted counts per isotope used as a soft "
            "constraint in next-pose selection. Defaults to the strength-prior "
            "count floor; use 0 to disable."
        ),
    )
    parser.add_argument(
        "--pose-min-observation-penalty-scale",
        type=float,
        default=1.0,
        help="Relative weight of the all-isotope pose observability constraint.",
    )
    parser.add_argument(
        "--pose-min-observation-aggregate",
        choices=("max", "mean"),
        default="max",
        help="Aggregate shield-orientation predicted counts for pose observability.",
    )
    parser.add_argument(
        "--path-planner",
        choices=("one_step", "dss_pp"),
        default=None,
        help="Next-pose planner: original one-step selector or DSS-PP.",
    )
    parser.add_argument(
        "--dss-horizon",
        type=int,
        default=None,
        help="DSS-PP receding-horizon length.",
    )
    parser.add_argument(
        "--dss-beam-width",
        type=int,
        default=None,
        help="DSS-PP beam width.",
    )
    parser.add_argument(
        "--dss-program-length",
        type=int,
        default=None,
        help="Number of shield postures in each DSS-PP measurement program.",
    )
    parser.add_argument(
        "--dss-signature-weight",
        type=float,
        default=None,
        help="DSS-PP shield-signature separation weight.",
    )
    parser.add_argument(
        "--dss-differential-weight",
        type=float,
        default=None,
        help="DSS-PP differential-observability penalty weight.",
    )
    parser.add_argument(
        "--dss-rotation-weight",
        type=float,
        default=None,
        help="DSS-PP shield-transition penalty weight.",
    )
    parser.add_argument(
        "--rotations-per-pose",
        "--orientation-k",
        dest="rotations_per_pose",
        type=int,
        default=None,
        help="Number of shield orientation pairs measured at each robot pose.",
    )
    parser.add_argument(
        "--min-rotations-per-pose",
        type=int,
        default=None,
        help=(
            "Minimum shield orientation measurements before IG early stopping. "
            "Defaults to --rotations-per-pose when that option is set."
        ),
    )
    parser.add_argument(
        "--init-grid-spacing-m",
        type=float,
        default=None,
        help=(
            "Initial PF grid spacing in meters. Use <=0 to disable grid "
            "initialization and use --num-particles random particles."
        ),
    )
    parser.add_argument(
        "--planning-eig-samples",
        type=int,
        default=None,
        help="Monte Carlo samples used for planning EIG estimates.",
    )
    parser.add_argument(
        "--planning-rollout-particles",
        type=int,
        default=None,
        help="Particle cap used for planning rollouts.",
    )
    parser.add_argument(
        "--notify",
        "--notify-piplup",
        dest="notify_piplup",
        action="store_true",
        default=None,
        help=(
            "Send start/final/failure notifications to piplup-notify. "
            "Requires PIPLUP_NOTIFY_TOKEN or --notify-token."
        ),
    )
    parser.add_argument(
        "--no-notify",
        dest="notify_piplup",
        action="store_false",
        help="Disable piplup notifications even if PIPLUP_NOTIFY_ENABLED is set.",
    )
    parser.add_argument(
        "--notify-url",
        type=str,
        default=None,
        help=f"piplup-notify base URL (default: {PIPLUP_DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--notify-token",
        type=str,
        default=None,
        help="Bearer token for piplup /api/events. Prefer PIPLUP_NOTIFY_TOKEN.",
    )
    parser.add_argument(
        "--notify-account",
        type=str,
        default=None,
        help="Optional account label stored with piplup events.",
    )
    parser.add_argument(
        "--notify-run-id",
        type=str,
        default=None,
        help="Optional stable run id for piplup event dedupe.",
    )
    parser.add_argument(
        "--notify-timeout-s",
        type=float,
        default=None,
        help="HTTP timeout for piplup notification requests.",
    )
    parser.add_argument(
        "--notify-spectrum",
        action="store_true",
        help="Send per-measurement spectrum payloads to piplup/Railway.",
    )
    parser.add_argument(
        "--notify-spectrum-every",
        type=int,
        default=1,
        help="Send one spectrum event every N measurements when --notify-spectrum is set.",
    )
    parser.add_argument(
        "--notify-spectrum-max-bins",
        type=int,
        default=800,
        help="Maximum spectrum bins included in each piplup spectrum event.",
    )
    args = parser.parse_args()
    run_mode, sim_backend, sim_config_path, matplotlib_live = _resolve_run_settings(
        args,
        parser,
    )
    if args.max_sources is None:
        args.max_sources = DEFAULT_MAX_SOURCES_PER_ISOTOPE
    pf_overrides: dict[str, object] = {
        "max_sources": args.max_sources,
        "max_resamples_per_observation": args.temper_max_resamples,
    }
    if args.merge_prob is not None:
        pf_overrides["merge_prob"] = float(args.merge_prob)
    if args.merge_distance_max is not None:
        pf_overrides["merge_distance_max"] = float(args.merge_distance_max)
    if args.merge_delta_ll_threshold is not None:
        pf_overrides["merge_delta_ll_threshold"] = float(args.merge_delta_ll_threshold)
    if args.cluster_eps_m is not None:
        pf_overrides["cluster_eps_m"] = float(args.cluster_eps_m)
    if args.birth_detector_min_sep_m is not None:
        pf_overrides["birth_detector_min_sep_m"] = float(args.birth_detector_min_sep_m)
    if args.no_roughen_on_temper_resample:
        pf_overrides["disable_regularize_on_temper_resample"] = True
    if args.roughening_k is not None:
        pf_overrides["roughening_k"] = float(args.roughening_k)
    if args.min_sigma_pos is not None:
        pf_overrides["min_sigma_pos"] = float(args.min_sigma_pos)
    if args.max_sigma_pos is not None:
        pf_overrides["max_sigma_pos"] = float(args.max_sigma_pos)
    if args.rotations_per_pose is not None:
        pf_overrides["orientation_k"] = max(1, int(args.rotations_per_pose))
        pf_overrides["min_rotations_per_pose"] = max(1, int(args.rotations_per_pose))
    if args.min_rotations_per_pose is not None:
        pf_overrides["min_rotations_per_pose"] = max(
            0,
            int(args.min_rotations_per_pose),
        )
    if args.init_grid_spacing_m is not None:
        pf_overrides["init_grid_spacing_m"] = (
            None
            if float(args.init_grid_spacing_m) <= 0.0
            else float(args.init_grid_spacing_m)
        )
    if args.planning_eig_samples is not None:
        pf_overrides["planning_eig_samples"] = max(1, int(args.planning_eig_samples))
    if args.planning_rollout_particles is not None:
        pf_overrides["planning_rollout_particles"] = max(
            1,
            int(args.planning_rollout_particles),
        )
    sources = None
    source_layout_path_for_log: str | None = None
    source_generation_mode = "demo"
    source_config_token = str(args.source_config).strip().lower() if args.source_config else ""
    source_config_explicit = _source_config_was_explicit()
    if source_config_token in RANDOM_SOURCE_CONFIG_TOKENS or (
        args.environment_mode == "random" and not source_config_explicit
    ):
        source_generation_mode = "surface_random"
        print("Using surface-constrained random source generation.")
    elif args.source_config:
        source_path = Path(args.source_config)
        if not source_path.is_absolute():
            source_path = (ROOT / source_path).resolve()
        if source_path.exists():
            try:
                sources = load_sources_from_json(source_path)
                source_layout_path_for_log = source_path.as_posix()
                print(f"Loaded {len(sources)} sources from {source_path}")
            except (OSError, ValueError) as exc:
                print(f"Failed to load sources from {source_path}: {exc}")
        else:
            print(f"Source config not found: {source_path}. Using built-in demo sources.")
    print(
        "Execution mode: "
        f"{run_mode} (backend={sim_backend}, "
        f"sim_config={sim_config_path or 'none'}, "
        f"matplotlib_live={matplotlib_live})"
    )
    notify_enabled = args.notify_piplup
    if notify_enabled is None and args.notify_spectrum:
        notify_enabled = True
    notification_config = PiplupNotificationConfig.from_env(
        enabled=notify_enabled,
        base_url=args.notify_url,
        token=args.notify_token,
        account=args.notify_account,
        run_id=args.notify_run_id,
        timeout_s=args.notify_timeout_s,
    )
    run_live_pf(
        live=matplotlib_live,
        max_steps=args.max_steps,
        max_poses=args.max_poses,
        sources=sources,
        detect_threshold_abs=args.detect_threshold_abs,
        detect_threshold_rel=args.detect_threshold_rel,
        detect_consecutive=args.detect_consecutive,
        detect_min_steps=args.detect_min_steps,
        ig_threshold_mode=args.ig_threshold_mode,
        ig_threshold_rel=args.ig_threshold_rel,
        ig_threshold_min=args.ig_threshold_min,
        environment_mode=args.environment_mode,
        obstacle_layout_path=None if args.no_obstacles else args.obstacle_config,
        obstacle_seed=args.obstacle_seed,
        eval_match_radius_m=args.eval_match_radius,
        birth_enabled=args.birth,
        num_particles=args.num_particles,
        pf_config_overrides=pf_overrides,
        output_tag=args.output_tag,
        measurement_log_dir=args.measurement_log_dir,
        source_layout_path=source_layout_path_for_log,
        pose_candidates=args.pose_candidates,
        pose_min_dist=args.pose_min_dist,
        converge=args.converge,
        sim_backend=sim_backend,
        sim_config_path=sim_config_path,
        blender_executable=args.blender_executable,
        blender_output_path=args.blender_output,
        blender_timeout_s=args.blender_timeout_s,
        passage_width_m=args.passage_width_m,
        robot_radius_m=args.robot_radius_m,
        nominal_motion_speed_m_s=args.robot_speed,
        rotation_overhead_s=args.rotation_overhead_s,
        measurement_time_s=args.measurement_time_s,
        adaptive_dwell=args.adaptive_dwell,
        adaptive_dwell_chunk_s=args.adaptive_dwell_chunk_s,
        adaptive_min_dwell_s=args.adaptive_min_dwell_s,
        adaptive_ready_min_counts=args.adaptive_ready_min_counts,
        adaptive_ready_min_isotopes=args.adaptive_ready_min_isotopes,
        adaptive_ready_min_snr=args.adaptive_ready_min_snr,
        adaptive_strength_prior=args.adaptive_strength_prior,
        adaptive_strength_prior_steps=args.adaptive_strength_prior_steps,
        adaptive_strength_prior_min_counts=args.adaptive_strength_prior_min_counts,
        adaptive_strength_prior_log_sigma=args.adaptive_strength_prior_log_sigma,
        pose_min_observation_counts=args.pose_min_observation_counts,
        pose_min_observation_penalty_scale=args.pose_min_observation_penalty_scale,
        pose_min_observation_aggregate=args.pose_min_observation_aggregate,
        path_planner=args.path_planner,
        dss_horizon=args.dss_horizon,
        dss_beam_width=args.dss_beam_width,
        dss_program_length=args.dss_program_length,
        dss_signature_weight=args.dss_signature_weight,
        dss_differential_weight=args.dss_differential_weight,
        dss_rotation_weight=args.dss_rotation_weight,
        source_generation_mode=source_generation_mode,
        random_source_seed=args.source_seed,
        random_source_count=args.random_source_count,
        random_source_isotopes=args.random_source_isotopes,
        random_source_intensity_cps_1m=args.random_source_intensity_cps_1m,
        random_source_intensity_min_cps_1m=args.random_source_intensity_min_cps_1m,
        random_source_intensity_max_cps_1m=args.random_source_intensity_max_cps_1m,
        notification_config=notification_config,
        notify_spectrum=args.notify_spectrum,
        notify_spectrum_every=args.notify_spectrum_every,
        notify_spectrum_max_bins=args.notify_spectrum_max_bins,
    )


if __name__ == "__main__":
    main()
