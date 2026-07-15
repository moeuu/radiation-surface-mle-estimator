"""Geant4-backed experiment runner for the Kemp comparison baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from baselines.kemp.filter import KempFilterConfig
from baselines.kemp.kernels import DiscreteAttenuationKernel, KempKernelConfig
from baselines.kemp.parallel import KempParallelConfig, KempParallelLogDDPF
from evaluation_metrics import compute_metrics
from measurement.model import EnvironmentConfig, PointSource
from measurement.obstacles import ObstacleGrid
from measurement.observation_model import build_runtime_observation_model
from realtime_demo import (
    _analysis_spectrum_array,
    _analysis_spectrum_variance,
    _evaluate_spectrum_counts,
    _has_environment_obstacles,
    _spectrum_config_from_runtime_config,
    load_sources_from_json,
)
from sim import SimulationCommand, create_simulation_runtime, load_runtime_config
from spectrum.pipeline import SpectralDecomposer
from runtime_defaults import (
    DEFAULT_MAX_SOURCES_PER_ISOTOPE,
    DEFAULT_MEASUREMENT_TIME_S,
    DEFAULT_RANDOM_SOURCE_INTENSITY_CPS_1M,
)


@dataclass(frozen=True)
class KempRunConfig:
    """Configure one Kemp baseline full-simulation run."""

    sim_backend: str = "geant4"
    sim_config_path: str = "configs/geant4/variance_reduction_external_no_isaac_32threads.json"
    source_config_path: str | None = None
    obstacle_config_path: str | None = None
    output_dir: str = "results/baselines/kemp/latest"
    max_poses: int = 10
    dwell_time_s: float = DEFAULT_MEASUREMENT_TIME_S
    measurement_spacing_m: float = 4.0
    shield_fe_index: int = 0
    shield_pb_index: int = 0
    num_particles: int = 2000
    max_sources: int = DEFAULT_MAX_SOURCES_PER_ISOTOPE
    grid_spacing_m: tuple[float, float, float] = (0.5, 0.5, 0.5)
    grid_z_levels_m: tuple[float, ...] = (
        0.5,
        1.5,
        2.5,
        3.5,
        4.5,
        5.5,
        6.5,
        7.5,
        8.5,
        9.5,
    )
    eval_match_radius_m: float = 1.0
    rng_seed: int = 20260502
    source_strength_log_mean: float = float(
        np.log(DEFAULT_RANDOM_SOURCE_INTENSITY_CPS_1M)
    )
    source_strength_log_sigma: float = 1.5
    detector_height_m: float = 0.5
    room_size_xyz: tuple[float, float, float] = (10.0, 20.0, 10.0)
    background_cps: float = 0.0
    spectrum_count_method: str = "response_poisson"
    detect_threshold_abs: float = 0.0
    detect_threshold_rel: float = 0.0
    detect_threshold_rel_by_isotope: dict[str, float] = field(default_factory=dict)
    min_peaks_by_isotope: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class KempRunResult:
    """Summarize one completed Kemp baseline run."""

    output_dir: Path
    estimates: dict[str, list[dict[str, object]]]
    metrics: dict[str, Any]
    observation_log: list[dict[str, object]]
    measurement_path: list[list[float]]


def _repo_root() -> Path:
    """Return the repository root path."""
    return Path(__file__).resolve().parents[3]


def _resolve_path(path_value: str | None) -> Path | None:
    """Resolve a path relative to the repository root."""
    if path_value in (None, ""):
        return None
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (_repo_root() / path).resolve()


def _load_sources(path_value: str | None) -> list[PointSource]:
    """Load configured sources or use the standard demo layout."""
    path = _resolve_path(path_value)
    if path is None:
        return [
            PointSource("Cs-137", (5.0, 10.0, 1.0), 30000.0),
            PointSource("Co-60", (2.0, 15.0, 1.0), 30000.0),
            PointSource("Eu-154", (7.0, 5.0, 1.0), 30000.0),
        ]
    return load_sources_from_json(path)


def _load_obstacle_grid(path_value: str | None) -> ObstacleGrid | None:
    """Load an obstacle grid if a path is configured."""
    path = _resolve_path(path_value)
    if path is None:
        return None
    return ObstacleGrid.load(path)


def _free_measurement_candidates(
    env: EnvironmentConfig,
    spacing_m: float,
    obstacle_grid: ObstacleGrid | None,
) -> NDArray[np.float64]:
    """Return free 2D coverage stations at the detector height."""
    spacing = max(float(spacing_m), 0.5)
    xs = np.arange(1.0, float(env.size_x), spacing)
    ys = np.arange(1.0, float(env.size_y), spacing)
    points = []
    for y in ys:
        row_xs = xs if int(round(y / spacing)) % 2 == 0 else xs[::-1]
        for x in row_xs:
            point = (float(x), float(y), float(env.detector_position[2]))
            if obstacle_grid is None or obstacle_grid.is_free(point):
                points.append(point)
    if not points:
        raise ValueError("No free Kemp measurement candidates are available.")
    return np.asarray(points, dtype=float)


def _nearest_neighbor_path(
    candidates: NDArray[np.float64],
    start: Sequence[float],
    max_poses: int,
) -> NDArray[np.float64]:
    """Order measurement candidates by nearest-neighbor coverage travel."""
    remaining = list(range(candidates.shape[0]))
    current = np.asarray(start, dtype=float)
    path = []
    for _ in range(min(int(max_poses), len(remaining))):
        distances = [float(np.linalg.norm(candidates[idx, :2] - current[:2])) for idx in remaining]
        best_local = int(np.argmin(distances))
        best_idx = remaining.pop(best_local)
        point = candidates[best_idx]
        path.append(point)
        current = point
    return np.asarray(path, dtype=float)


def build_measurement_path(
    env: EnvironmentConfig,
    *,
    spacing_m: float,
    obstacle_grid: ObstacleGrid | None,
    max_poses: int,
) -> NDArray[np.float64]:
    """Build the fixed discrete measurement path used by the Kemp baseline."""
    candidates = _free_measurement_candidates(env, spacing_m, obstacle_grid)
    return _nearest_neighbor_path(candidates, env.detector(), max_poses)


def _scene_reset_payload(
    *,
    env: EnvironmentConfig,
    sources: Sequence[PointSource],
    obstacle_grid: ObstacleGrid | None,
    runtime_config: dict[str, Any],
) -> dict[str, Any]:
    """Return the simulator reset payload for a Kemp run."""
    has_environment_obstacles = _has_environment_obstacles(obstacle_grid)
    return {
        "usd_path": None if has_environment_obstacles else "",
        "room_size_xyz": [float(env.size_x), float(env.size_y), float(env.size_z)],
        "source_count": len(sources),
        "sources": [
            {
                "isotope": source.isotope,
                "position": [float(value) for value in source.position],
                "intensity_cps_1m": float(source.intensity_cps_1m),
            }
            for source in sources
        ],
        "obstacle_origin_xy": (
            [0.0, 0.0] if obstacle_grid is None else list(obstacle_grid.origin)
        ),
        "obstacle_cell_size_m": (
            1.0 if obstacle_grid is None else float(obstacle_grid.cell_size)
        ),
        "obstacle_material": str(runtime_config.get("obstacle_material", "concrete")),
        "obstacle_grid_shape": (
            [0, 0] if obstacle_grid is None else list(obstacle_grid.grid_shape)
        ),
        "obstacle_cells": (
            [] if obstacle_grid is None else list(obstacle_grid.blocked_cells)
        ),
        "author_obstacle_prims": obstacle_grid is not None,
        "author_room_boundary_prims": bool(
            runtime_config.get("author_room_boundary_prims", True)
        ),
        "use_config_usd_fallback": has_environment_obstacles,
    }


def _true_sources_by_isotope(
    sources: Sequence[PointSource],
) -> dict[str, list[dict[str, object]]]:
    """Return ground-truth sources grouped for metric computation."""
    out: dict[str, list[dict[str, object]]] = {}
    for source in sources:
        out.setdefault(source.isotope, []).append(
            {
                "position": [float(value) for value in source.position],
                "strength": float(source.intensity_cps_1m),
            }
        )
    return out


def _save_run_plot(
    *,
    output_path: Path,
    env: EnvironmentConfig,
    sources: Sequence[PointSource],
    measurement_path: NDArray[np.float64],
    estimates: dict[str, list[dict[str, object]]],
) -> None:
    """Save a 2D source-estimate and path diagnostic plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 9))
    ax.set_xlim(0.0, float(env.size_x))
    ax.set_ylim(0.0, float(env.size_y))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    if measurement_path.size:
        ax.plot(
            measurement_path[:, 0],
            measurement_path[:, 1],
            "-o",
            color="tab:cyan",
            label="Kemp path",
        )
    for source in sources:
        ax.scatter(
            source.position[0],
            source.position[1],
            marker="*",
            s=130,
            label=f"true {source.isotope}",
        )
    for isotope, entries in estimates.items():
        for entry in entries:
            pos = np.asarray(entry["position"], dtype=float)
            ax.scatter(pos[0], pos[1], marker="x", s=80, label=f"Kemp {isotope}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Kemp baseline fixed-path estimates")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_kemp_full_simulation(config: KempRunConfig) -> KempRunResult:
    """Run the Kemp baseline against Geant4 or analytic full observations."""
    root = _repo_root()
    output_dir = _resolve_path(config.output_dir)
    if output_dir is None:
        output_dir = root / "results" / "baselines" / "kemp" / "latest"
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_config_path = _resolve_path(config.sim_config_path)
    runtime_config = load_runtime_config(runtime_config_path)
    sources = _load_sources(config.source_config_path)
    obstacle_grid = _load_obstacle_grid(config.obstacle_config_path)
    env = EnvironmentConfig(
        size_x=float(config.room_size_xyz[0]),
        size_y=float(config.room_size_xyz[1]),
        size_z=float(config.room_size_xyz[2]),
        detector_position=(1.0, 1.0, float(config.detector_height_m)),
    )
    spectrum_config = _spectrum_config_from_runtime_config(runtime_config)
    decomposer = SpectralDecomposer(spectrum_config=spectrum_config)
    isotopes = list(decomposer.isotope_names)
    observation_model = build_runtime_observation_model(
        runtime_config,
        isotopes=isotopes,
    )
    kernel = DiscreteAttenuationKernel.from_environment(
        env=env,
        isotopes=isotopes,
        mu_by_isotope=observation_model.mu_by_isotope,
        shield_params=observation_model.shield_params,
        obstacle_grid=obstacle_grid,
        obstacle_mu_by_isotope=observation_model.obstacle_mu_by_isotope,
        line_mu_by_isotope=observation_model.line_mu_by_isotope,
        config=KempKernelConfig(
            grid_spacing_m=tuple(config.grid_spacing_m),
            z_levels_m=tuple(config.grid_z_levels_m),
            obstacle_height_m=float(observation_model.obstacle_height_m),
            obstacle_buildup_coeff=(
                float(observation_model.obstacle_buildup_coeff)
                if obstacle_grid is not None
                else 0.0
            ),
            detector_radius_m=float(observation_model.detector_geometry.count_radius_m),
            detector_aperture_radius_m=float(
                observation_model.detector_geometry.aperture_radius_m
            ),
            detector_aperture_samples=int(
                observation_model.detector_geometry.aperture_samples
            ),
            source_extent_radius_m=float(observation_model.source_extent_radius_m),
            source_extent_samples=int(observation_model.source_extent_samples),
            transport_response_model=observation_model.transport_response_model,
            use_gpu=True,
            gpu_dtype="float64",
        ),
    )
    filter_config = KempFilterConfig(
        num_particles=int(config.num_particles),
        max_sources=int(config.max_sources),
        background_cps=float(config.background_cps),
        init_strength_log_mean=float(config.source_strength_log_mean),
        init_strength_log_sigma=float(config.source_strength_log_sigma),
        rng_seed=int(config.rng_seed),
    )
    estimator = KempParallelLogDDPF(
        isotopes=isotopes,
        kernel=kernel,
        config=KempParallelConfig(filter_config=filter_config),
    )
    runtime = create_simulation_runtime(
        config.sim_backend,
        sources=sources,
        decomposer=decomposer,
        mu_by_isotope=observation_model.mu_by_isotope,
        shield_params=observation_model.shield_params,
        runtime_config=runtime_config,
        runtime_config_path=runtime_config_path,
    )
    measurement_path = build_measurement_path(
        env,
        spacing_m=float(config.measurement_spacing_m),
        obstacle_grid=obstacle_grid,
        max_poses=int(config.max_poses),
    )
    observation_log: list[dict[str, object]] = []
    start_wall = time.perf_counter()
    try:
        runtime.reset(
            _scene_reset_payload(
                env=env,
                sources=sources,
                obstacle_grid=obstacle_grid,
                runtime_config=runtime_config,
            )
        )
        for step_idx, pose in enumerate(measurement_path):
            observation = runtime.step(
                SimulationCommand(
                    step_id=step_idx,
                    target_pose_xyz=tuple(float(value) for value in pose),
                    target_base_yaw_rad=0.0,
                    fe_orientation_index=int(config.shield_fe_index),
                    pb_orientation_index=int(config.shield_pb_index),
                    dwell_time_s=float(config.dwell_time_s),
                )
            )
            spectrum = _analysis_spectrum_array(observation, decomposer)
            variance = _analysis_spectrum_variance(observation, decomposer)
            counts, count_variances, detected = _evaluate_spectrum_counts(
                decomposer,
                spectrum,
                live_time_s=float(config.dwell_time_s),
                spectrum_count_method=str(config.spectrum_count_method),
                detect_threshold_abs=float(config.detect_threshold_abs),
                detect_threshold_rel=float(config.detect_threshold_rel),
                detect_threshold_rel_by_isotope=dict(config.detect_threshold_rel_by_isotope),
                min_peaks_by_isotope=dict(config.min_peaks_by_isotope),
                spectrum_variance=variance,
                transport_metadata=observation.metadata,
            )
            diagnostics = estimator.update(
                detector_pos=observation.detector_pose_xyz,
                live_time_s=float(config.dwell_time_s),
                counts_by_isotope=counts,
                variances_by_isotope=count_variances,
                fe_index=int(config.shield_fe_index),
                pb_index=int(config.shield_pb_index),
            )
            observation_log.append(
                {
                    "step": int(step_idx),
                    "pose": [float(value) for value in observation.detector_pose_xyz],
                    "counts": {key: float(value) for key, value in counts.items()},
                    "variances": {key: float(value) for key, value in count_variances.items()},
                    "detected": sorted(detected),
                    "diagnostics": diagnostics,
                    "geant4_metadata": dict(observation.metadata),
                }
            )
            print(
                f"[Kemp step {step_idx}] pose={np.asarray(pose).round(3).tolist()} "
                f"counts={{{', '.join(f'{k}:{v:.1f}' for k, v in sorted(counts.items()))}}}"
            )
    finally:
        runtime.close()
    raw_estimates = estimator.raw_estimates_for_metrics()
    estimates = estimator.estimates_for_metrics()
    metrics = compute_metrics(
        _true_sources_by_isotope(sources),
        estimates,
        match_radius_m=float(config.eval_match_radius_m),
    )
    result = KempRunResult(
        output_dir=output_dir,
        estimates=estimates,
        metrics=metrics,
        observation_log=observation_log,
        measurement_path=measurement_path.tolist(),
    )
    payload = {
        "config": config.__dict__,
        "wall_time_s": float(time.perf_counter() - start_wall),
        "sources": [
            {
                "isotope": source.isotope,
                "position": list(source.position),
                "intensity_cps_1m": float(source.intensity_cps_1m),
            }
            for source in sources
        ],
        "raw_estimates": raw_estimates,
        "estimates": estimates,
        "metrics": metrics,
        "observations": observation_log,
        "measurement_path": result.measurement_path,
        "method": "Kemp et al. parallel log-domain DDPF comparison baseline",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _save_run_plot(
        output_path=output_dir / "kemp_baseline_path_estimates.png",
        env=env,
        sources=sources,
        measurement_path=measurement_path,
        estimates=estimates,
    )
    print(f"Kemp baseline summary written to {output_dir / 'summary.json'}")
    return result
