"""Baseline PF demo without shielding, with real-time visualization."""
# ruff: noqa: E402

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import matplotlib


def _has_display() -> bool:
    """Return True when a GUI display is likely available."""
    if sys.platform.startswith("linux"):
        return bool(
            os.environ.get("DISPLAY")
            or os.environ.get("WAYLAND_DISPLAY")
            or os.environ.get("MIR_SOCKET")
        )
    return True


def _configure_matplotlib() -> None:
    """Configure matplotlib backend for interactive or headless use."""
    headless = "--headless" in sys.argv or "--no-live" in sys.argv
    if headless or not _has_display():
        matplotlib.use("Agg")
        return
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")


_configure_matplotlib()

import matplotlib.pyplot as plt
import numpy as np
import time

from measurement.model import EnvironmentConfig, PointSource
from spectrum.pipeline import SpectralDecomposer
from sim.blender_environment import generate_blender_environment_usd
from visualization.realtime_viz import RealTimePFVisualizer, build_frame_from_pf
from evaluation_metrics import compute_metrics, print_metrics_report

from baselines.legacy_no_shield.measurement import BaselineMeasurement
from baselines.legacy_no_shield.no_shield_pf import NoShieldPF
from baselines.legacy_no_shield.planning import generate_measurement_positions
from pf.particle_filter import PFConfig
from runtime_defaults import DEFAULT_ENVIRONMENT_MODE, DEFAULT_FIXED_OBSTACLE_CONFIG
from runtime_environment import build_runtime_obstacle_environment

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results" / "baselines" / "legacy_no_shield"
DEFAULT_SOURCE_CONFIG = ROOT / "source_layouts" / "demo_sources.json"
DEFAULT_OBSTACLE_CONFIG = ROOT / DEFAULT_FIXED_OBSTACLE_CONFIG
MEASUREMENT_TIME_S = 30.0
SAVE_EVERY_N_STEPS = 10
DETECT_MIN_PEAKS_BY_ISOTOPE = {"Eu-154": 2, "Co-60": 1}
DETECT_REL_THRESH_BY_ISOTOPE = {"Co-60": 0.1}
DETECT_CONSECUTIVE = 20
DETECT_CONSECUTIVE_BY_ISOTOPE = {"Cs-137": 3, "Co-60": 3, "Eu-154": 5}


def _update_detection_hysteresis(
    candidates: set[str],
    detect_counts: dict[str, int],
    miss_counts: dict[str, int],
    active_isotopes: set[str],
    *,
    consecutive: int,
    consecutive_by_isotope: dict[str, int] | None = None,
) -> set[str]:
    """
    Update detection state with consecutive hit/miss hysteresis.

    Isotopes are activated after `consecutive` hits and deactivated after
    `consecutive` misses unless overridden by consecutive_by_isotope.
    """
    updated = set(active_isotopes)
    miss_required = consecutive
    for iso in detect_counts:
        hit_required = consecutive
        if consecutive_by_isotope and iso in consecutive_by_isotope:
            hit_required = int(consecutive_by_isotope[iso])
        if iso in candidates:
            detect_counts[iso] += 1
            miss_counts[iso] = 0
        else:
            miss_counts[iso] += 1
            detect_counts[iso] = 0
        if detect_counts[iso] >= hit_required:
            updated.add(iso)
        if miss_counts[iso] >= miss_required:
            updated.discard(iso)
    return updated


def _detect_isotopes_from_counts(
    counts: dict[str, float],
    detect_threshold_abs: float,
    detect_threshold_rel: float,
    detect_threshold_rel_by_isotope: dict[str, float] | None,
) -> set[str]:
    """Return isotopes detected from counts using absolute/relative thresholds."""
    max_c = max(counts.values()) if counts else 0.0
    detected: set[str] = set()
    rel_by_iso = detect_threshold_rel_by_isotope or {}
    for iso, val in counts.items():
        rel_thresh = float(rel_by_iso.get(iso, detect_threshold_rel))
        if val >= detect_threshold_abs and (max_c <= 0.0 or val / max_c >= rel_thresh):
            detected.add(iso)
    return detected


def _build_demo_sources() -> list[PointSource]:
    """Return a default set of point sources."""
    return [
        PointSource("Cs-137", position=(5.0, 10.0, 5.0), intensity_cps_1m=50000.0),
        PointSource("Co-60", position=(2.0, 15.0, 7.0), intensity_cps_1m=20000.0),
        PointSource("Eu-154", position=(7.0, 5.0, 3.0), intensity_cps_1m=30000.0),
    ]


def load_sources_from_json(path: Path) -> list[PointSource]:
    """Load point sources from a JSON configuration file."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        entries = data.get("sources", [])
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError("Source config must be a list or include a 'sources' list.")
    if not isinstance(entries, list):
        raise ValueError("Source config 'sources' must be a list.")
    sources: list[PointSource] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Source entry {idx} must be an object.")
        isotope = entry.get("isotope")
        position = entry.get("position")
        intensity = entry.get("intensity_cps_1m")
        if intensity is None:
            intensity = entry.get("strength_cps_1m")
        if intensity is None:
            intensity = entry.get("intensity")
        if isotope is None or position is None or intensity is None:
            raise ValueError(
                "Each source must include 'isotope', 'position', and "
                "'intensity_cps_1m'."
            )
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            raise ValueError(f"Source entry {idx} position must be a 3-element list.")
        sources.append(
            PointSource(
                isotope=str(isotope),
                position=(float(position[0]), float(position[1]), float(position[2])),
                intensity_cps_1m=float(intensity),
            )
        )
    return sources


def _candidate_axis_points(start: float, stop: float, step: float) -> np.ndarray:
    """Return evenly spaced axis points within [start, stop] using the given step."""
    if step <= 0:
        raise ValueError("step must be positive.")
    if stop < start:
        return np.zeros(0, dtype=float)
    count = int(np.floor((stop - start) / step)) + 1
    if count <= 0:
        return np.zeros(0, dtype=float)
    return start + step * np.arange(count, dtype=float)


def _build_candidate_sources(
    env: EnvironmentConfig,
    spacing: tuple[float, float, float],
    margin: float,
) -> np.ndarray:
    """Create a dense 3D grid of candidate sources inside the environment bounds."""
    xs = _candidate_axis_points(margin, env.size_x - margin, spacing[0])
    ys = _candidate_axis_points(margin, env.size_y - margin, spacing[1])
    zs = _candidate_axis_points(margin, env.size_z - margin, spacing[2])
    if xs.size == 0 or ys.size == 0 or zs.size == 0:
        raise ValueError("Candidate grid is empty; check spacing and margin values.")
    return np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=float)


def _default_use_gpu() -> bool:
    """Return True if CUDA is available for torch acceleration."""
    try:
        from pf import gpu_utils
    except ImportError:
        return False
    return gpu_utils.torch_available()


def _report_gpu_status() -> None:
    """Log GPU availability (baseline PF uses CPU-only inverse-square updates)."""
    try:
        from pf import gpu_utils
    except ImportError:
        print("[baseline] GPU check: torch not available; using CPU.")
        return
    if gpu_utils.torch_available():
        print("[baseline] GPU check: CUDA available; using CPU-only baseline PF.")
    elif gpu_utils.torch_installed():
        print("[baseline] GPU check: torch installed but CUDA unavailable; using CPU.")
    else:
        print("[baseline] GPU check: torch not installed; using CPU.")


def run_baseline_pf(
    *,
    live: bool = True,
    total_time_s: float = 300.0,
    sources: list[PointSource] | None = None,
    environment_mode: str = DEFAULT_ENVIRONMENT_MODE,
    obstacle_layout_path: str | None = DEFAULT_OBSTACLE_CONFIG.as_posix(),
    obstacle_seed: int | None = None,
    detect_threshold_abs: float = 30.0,
    detect_threshold_rel: float = 0.2,
    eval_match_radius_m: float = 0.5,
    converge: bool = False,
    blender_executable: str | None = None,
    blender_output_path: str | None = None,
    blender_timeout_s: float = 120.0,
) -> None:
    """Run the baseline PF with active pose selection and no shielding."""
    env = EnvironmentConfig(
        size_x=10.0,
        size_y=20.0,
        size_z=10.0,
        detector_position=(1.0, 1.0, 0.5),
    )
    sources = _build_demo_sources() if sources is None else sources
    decomposer = SpectralDecomposer()
    obstacle_environment = build_runtime_obstacle_environment(
        root=ROOT,
        environment_mode=environment_mode,
        obstacle_layout_path=obstacle_layout_path,
        room_size_xyz=(env.size_x, env.size_y, env.size_z),
        detector_position_xy=env.detector_position,
        obstacle_seed=obstacle_seed,
        blocked_fraction=0.4,
    )
    obstacle_grid = obstacle_environment.grid
    normalized_environment_mode = obstacle_environment.mode
    if obstacle_environment.message is not None:
        print(obstacle_environment.message)
    if obstacle_grid is not None:
        if normalized_environment_mode == "random":
            if blender_output_path:
                generated_output_path = Path(blender_output_path)
                if not generated_output_path.is_absolute():
                    generated_output_path = (ROOT / generated_output_path).resolve()
            else:
                seed_token = "unseeded" if obstacle_seed is None else f"seed_{int(obstacle_seed)}"
                generated_output_path = RESULTS_DIR / "blender_environments" / f"random_{seed_token}.usda"
            generated_usd = generate_blender_environment_usd(
                grid=obstacle_grid,
                output_path=generated_output_path,
                room_size_xyz=(env.size_x, env.size_y, env.size_z),
                obstacle_height_m=2.0,
                obstacle_material="concrete",
                blender_executable=blender_executable,
                timeout_s=blender_timeout_s,
            )
            print(f"Generated Blender random environment: {generated_usd}")
    positions = generate_measurement_positions(
        env,
        obstacle_grid,
        total_time_s,
        measurement_time_s=MEASUREMENT_TIME_S,
    )
    isotopes = list(decomposer.isotope_names)
    _report_gpu_status()
    print(f"[baseline] Convergence gating enabled: {bool(converge)}")
    pf_conf = PFConfig(
        num_particles=2000,
        min_particles=2000,
        max_particles=2000,
        max_sources=3,
        resample_threshold=0.7,
        position_sigma=0.5,
        strength_sigma=0.4,
        min_strength=5.0,
        position_min=(0.0, 0.0, 0.0),
        position_max=(env.size_x, env.size_y, env.size_z),
        use_tempering=False,
        label_enable=True,
        label_alignment_iters=2,
        init_grid_spacing_m=1.0,
        converge_enable=bool(converge),
    )
    baseline_pf = NoShieldPF(isotopes=isotopes, config=pf_conf)
    current_pose = np.array(positions[0], dtype=float)
    current_pose_idx = 0

    true_src = {}
    true_strengths = {}
    for iso in isotopes:
        pos_list = [
            np.array(src.position, dtype=float)
            for src in sources
            if src.isotope == iso
        ]
        str_list = [src.intensity_cps_1m for src in sources if src.isotope == iso]
        if pos_list:
            true_src[iso] = np.vstack(pos_list)
        if str_list:
            true_strengths[iso] = [float(val) for val in str_list]

    viz = RealTimePFVisualizer(
        isotopes=isotopes,
        world_bounds=(0, env.size_x, 0, env.size_y, 0, env.size_z),
        true_sources=true_src,
        true_strengths=true_strengths,
        obstacle_grid=obstacle_grid,
        show_counts=False,
    )
    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    estimate_mode = "mmse"
    estimate_min_strength = 100.0
    estimate_min_existence_prob = None
    if live:
        plt.ion()
        plt.show(block=False)
        plt.pause(0.1)
    init_counts = {iso: 0.0 for iso in isotopes}
    init_meas = BaselineMeasurement(
        counts_by_isotope=init_counts,
        live_time_s=0.0,
        detector_position=current_pose,
        pose_idx=current_pose_idx,
        RFe=np.eye(3),
        RPb=np.eye(3),
    )
    init_frame = build_frame_from_pf(
        baseline_pf,
        init_meas,
        step_index=-1,
        time_sec=0.0,
        estimate_mode=estimate_mode,
        min_est_strength=estimate_min_strength,
        min_existence_prob=estimate_min_existence_prob,
    )
    viz.update(init_frame)
    if live:
        viz.fig.canvas.draw()
        if hasattr(viz.fig.canvas, "flush_events"):
            viz.fig.canvas.flush_events()
        plt.pause(5.0)

    elapsed = 0.0
    last_frame = None
    detect_counts = {iso: 0 for iso in isotopes}
    miss_counts = {iso: 0 for iso in isotopes}
    active_isotopes: set[str] = set()
    RFe = np.eye(3)
    RPb = np.eye(3)
    for step_idx, pose in enumerate(positions):
        step_start = time.perf_counter()
        current_pose = np.asarray(pose, dtype=float)
        if step_idx > 0:
            current_pose_idx = step_idx
        sim_elapsed = 0.0
        detect_elapsed = 0.0
        pf_elapsed = 0.0
        viz_elapsed = 0.0
        sim_start = time.perf_counter()
        env_step = EnvironmentConfig(detector_position=tuple(current_pose))
        spectrum, _ = decomposer.simulate_spectrum(
            sources=sources,
            environment=env_step,
            acquisition_time=MEASUREMENT_TIME_S,
            rng=np.random.default_rng(123 + step_idx),
        )
        sim_elapsed = time.perf_counter() - sim_start
        detect_start = time.perf_counter()
        counts_for_pf, detected = decomposer.isotope_counts_with_detection(
            spectrum,
            live_time_s=MEASUREMENT_TIME_S,
            detect_threshold_abs=detect_threshold_abs,
            detect_threshold_rel=detect_threshold_rel,
            detect_threshold_rel_by_isotope=DETECT_REL_THRESH_BY_ISOTOPE,
            min_peaks_by_isotope=DETECT_MIN_PEAKS_BY_ISOTOPE,
        )
        detect_elapsed = time.perf_counter() - detect_start
        active_isotopes = _update_detection_hysteresis(
            set(detected),
            detect_counts,
            miss_counts,
            active_isotopes,
            consecutive=DETECT_CONSECUTIVE,
            consecutive_by_isotope=DETECT_CONSECUTIVE_BY_ISOTOPE,
        )
        if active_isotopes:
            target_isotopes = set(active_isotopes)
        else:
            target_isotopes = set()
        z_k = {iso: float(counts_for_pf.get(iso, 0.0)) for iso in isotopes}
        measurement = BaselineMeasurement(
            counts_by_isotope=z_k,
            live_time_s=MEASUREMENT_TIME_S,
            detector_position=current_pose,
            pose_idx=current_pose_idx,
            RFe=RFe,
            RPb=RPb,
        )
        pf_start = time.perf_counter()
        if target_isotopes:
            counts_selected = {iso: float(counts_for_pf.get(iso, 0.0)) for iso in target_isotopes}
            baseline_pf.update_all(
                detector_pos=current_pose,
                counts_by_isotope=counts_selected,
                live_time_s=MEASUREMENT_TIME_S,
                step_idx=step_idx,
            )
        pf_elapsed = time.perf_counter() - pf_start
        elapsed += MEASUREMENT_TIME_S
        viz_start = time.perf_counter()
        if target_isotopes:
            class _FilteredPF:
                def __init__(self, pf: NoShieldPF, allowed: set[str]) -> None:
                    """Keep only the requested isotope filters for visualization."""
                    self.filters = {iso: filt for iso, filt in pf.filters.items() if iso in allowed}

                def estimate_all(self) -> dict[str, object]:
                    """Return current estimates for the retained isotope filters."""
                    return {iso: filt.estimate() for iso, filt in self.filters.items()}

            viz_pf = _FilteredPF(baseline_pf, target_isotopes)
        else:
            class _EmptyPF:
                filters: dict[str, object] = {}

                @staticmethod
                def estimate_all() -> dict[str, object]:
                    """Return an empty estimate mapping for inactive visualization."""
                    return {}

            viz_pf = _EmptyPF()
        frame = build_frame_from_pf(
            viz_pf,
            measurement,
            step_index=step_idx,
            time_sec=elapsed,
            estimate_mode=estimate_mode,
            min_est_strength=estimate_min_strength,
            min_existence_prob=estimate_min_existence_prob,
        )
        last_frame = frame
        viz.update(frame)
        viz_elapsed = time.perf_counter() - viz_start
        print(f"[step {step_idx}] pose={current_pose.tolist()} measurement={z_k}")
        total_elapsed = time.perf_counter() - step_start
        print(
            f"[timing step {step_idx}] total={total_elapsed:.3f}s "
            f"sim={sim_elapsed:.3f}s detect={detect_elapsed:.3f}s "
            f"pf={pf_elapsed:.3f}s viz={viz_elapsed:.3f}s"
        )
        if (step_idx + 1) % SAVE_EVERY_N_STEPS == 0:
            step_path = out_dir / f"step_{step_idx:04d}_pf.png"
            viz.save_final(step_path.as_posix())
        if live:
            plt.pause(0.05)

    pf_out_path = out_dir / "result_baseline_pf.png"
    est_out_path = out_dir / "result_baseline_estimates.png"
    if last_frame is not None:
        last_frame.step_index = 79
        last_frame.time = 2400.0
        viz.update(last_frame)
    viz.save_final(pf_out_path.as_posix())
    viz.save_estimates_only(est_out_path.as_posix())
    gt_by_iso: dict[str, list[dict[str, float | list[float]]]] = {}
    for src in sources:
        gt_by_iso.setdefault(src.isotope, []).append(
            {
                "pos": [
                    float(src.position[0]),
                    float(src.position[1]),
                    float(src.position[2]),
                ],
                "strength": float(src.intensity_cps_1m),
            }
        )
    est_by_iso: dict[str, list[dict[str, float | list[float]]]] = {}
    if last_frame is not None:
        for iso, positions in last_frame.estimated_sources.items():
            strengths = last_frame.estimated_strengths.get(iso, np.zeros(0, dtype=float))
            est_list: list[dict[str, float | list[float]]] = []
            for pos, strength in zip(np.asarray(positions), np.asarray(strengths)):
                est_list.append(
                    {
                        "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                        "strength": float(strength),
                    }
                )
            est_by_iso[iso] = est_list
    else:
        for iso, state in baseline_pf.estimate_all().items():
            est_list: list[dict[str, float | list[float]]] = []
            if hasattr(state, "positions") and hasattr(state, "strengths"):
                positions = np.asarray(state.positions, dtype=float)
                strengths = np.asarray(state.strengths, dtype=float)
            elif isinstance(state, tuple) and len(state) == 2:
                positions = np.asarray(state[0], dtype=float)
                strengths = np.asarray(state[1], dtype=float)
            else:
                positions = np.zeros((0, 3), dtype=float)
                strengths = np.zeros(0, dtype=float)
            if estimate_min_strength is not None and strengths.size:
                mask = strengths >= estimate_min_strength
                positions = positions[mask]
                strengths = strengths[mask]
            for pos, strength in zip(positions, strengths):
                est_list.append(
                    {
                        "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                        "strength": float(strength),
                    }
                )
            est_by_iso[iso] = est_list
    metrics = compute_metrics(
        gt_by_iso,
        est_by_iso,
        match_radius_m=eval_match_radius_m,
    )
    print_metrics_report(metrics)
    if live:
        plt.ioff()
        plt.pause(0.1)
    plt.close("all")
