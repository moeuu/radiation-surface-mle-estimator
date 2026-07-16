"""Deterministic sampled-observation demo for standalone surface MLE.

The demo deliberately follows the production acquisition boundary: a local
analytic simulator samples a spectrum, the runtime spectrum-count extractor
produces the count-domain observation, and only then is a finalized
``MeasurementRecord`` created.  The count response matrix is used solely to
verify that the fixed route identifies every base surface patch; it is never
substituted for a simulator observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from measurement.continuous_kernels import ContinuousKernel
from measurement.model import EnvironmentConfig, PointSource
from measurement.observation_model import (
    build_runtime_observation_model,
    continuous_kernel_from_observation_model,
)
from measurement.obstacles import ObstacleGrid
from runtime.measurement_log import save_measurement_log
from runtime.records import MeasurementRecord, RunContext
from sim.protocol import SimulationCommand
from sim.runtime import AnalyticSimulationRuntime
from sim.shield_geometry import resolve_shield_thickness_config
from spectrum.pipeline import SpectralDecomposer
from spectrum.runtime_config import spectrum_config_from_runtime_config
from spectrum.runtime_counts import RuntimeCountExtractor, RuntimeCountResult

from .config import MLEConfig
from .provenance import repository_commit
from .response_builder import build_count_response
from .surface_patches import build_surface_patches
from .types import SurfacePatch, SurfacePatchSet


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_ISOTOPE = "Cs-137"
_ROOM_SIZE_XYZ = (2.0, 2.0, 2.0)
_PATCH_SPACING_M = (1.0, 1.0, 1.0)
_SOURCE_PATCH_OBJECT_ID = "room:wall:x_min"
_SOURCE_PATCH_CENTROID_XYZ = (0.0, 0.5, 0.5)
_SOURCE_STRENGTH_CPS_1M = 150.0
_LIVE_TIME_S = 5.0
_TRAVEL_TIME_S = 1.0
_SHIELD_ACTUATION_TIME_S = 0.25
_RNG_SEED = 827

_FIXED_ROUTE_XYZ: tuple[tuple[float, float, float], ...] = (
    (0.3, 0.3, 0.4),
    (0.7, 0.3, 0.4),
    (1.2, 0.3, 0.4),
    (1.7, 0.3, 0.4),
    (1.7, 0.7, 0.4),
    (1.7, 1.2, 0.4),
    (1.7, 1.7, 0.4),
    (1.2, 1.7, 0.4),
    (0.7, 1.7, 0.4),
    (0.3, 1.7, 0.4),
    (0.3, 1.2, 0.4),
    (0.3, 0.7, 0.4),
    (0.3, 0.7, 1.6),
    (0.3, 1.2, 1.6),
    (0.3, 1.7, 1.6),
    (0.7, 1.7, 1.6),
    (1.2, 1.7, 1.6),
    (1.7, 1.7, 1.6),
    (1.7, 1.2, 1.6),
    (1.7, 0.7, 1.6),
    (1.7, 0.3, 1.6),
    (1.2, 0.3, 1.6),
    (0.7, 0.3, 1.6),
    (0.3, 0.3, 1.6),
    (0.6, 0.6, 1.0),
    (1.0, 0.6, 1.0),
    (1.4, 0.6, 1.0),
    (1.4, 1.0, 1.0),
    (1.4, 1.4, 1.0),
    (1.0, 1.4, 1.0),
    (0.6, 1.4, 1.0),
    (0.6, 1.0, 1.0),
)

_SHIELD_PAIR_CYCLE: tuple[tuple[int, int], ...] = (
    (0, 7),
    (1, 6),
    (2, 5),
    (3, 4),
    (4, 3),
    (5, 2),
    (6, 1),
    (7, 0),
    (0, 3),
    (1, 4),
    (2, 7),
    (3, 0),
    (4, 1),
    (5, 6),
    (6, 2),
    (7, 5),
)
_FIXED_SHIELD_PAIRS = _SHIELD_PAIR_CYCLE * 2


@dataclass(frozen=True)
class _RouteObservationPlan:
    """Structural observation input used for route-response validation."""

    detector_positions_xyz: NDArray[np.float64]
    fe_indices: NDArray[np.int64]
    pb_indices: NDArray[np.int64]
    live_times_s: NDArray[np.float64]


@dataclass(frozen=True)
class AnalyticMLEDemoScenario:
    """All deterministic inputs and sampled records for one analytic demo."""

    context: RunContext
    records: tuple[MeasurementRecord, ...]
    mle_config: MLEConfig
    environment: EnvironmentConfig
    obstacle_grid: ObstacleGrid
    kernel: ContinuousKernel
    patches: SurfacePatchSet
    source_patch_index: int
    source_patch_id: int
    source_strength_cps_1m: float
    route_response_rank: int
    route_response_condition_number: float


def analytic_mle_demo_config() -> MLEConfig:
    """Return the CPU count-domain configuration tuned for the sampled demo."""
    return MLEConfig(
        mode="count",
        isotope_names=(_ISOTOPE,),
        patch_spacing_m=_PATCH_SPACING_M,
        quadrature_order=1,
        l1_weight=2.0,
        tv_weight=0.0,
        isotope_group_weight=0.0,
        fit_background_nuisance=False,
        fit_scatter_nuisance=False,
        max_iterations=6000,
        tolerance=0.02,
        objective_tolerance=1.0e-7,
        check_interval=20,
        support_threshold_fraction=1.0e-3,
        debias_refit=True,
        coarse_to_fine_levels=0,
        cluster_threshold_fraction=0.1,
        cluster_min_strength_cps_1m=0.0,
        held_out_fraction=0.0,
        use_gpu=False,
        random_seed=0,
    )


def _runtime_config() -> dict[str, object]:
    """Return the complete local physics and spectrum-processing payload."""
    shield = resolve_shield_thickness_config({})
    return {
        "source_rate_model": "detector_cps_1m",
        "candidate_isotopes": [_ISOTOPE],
        "sim_backend": "analytic",
        "spectrum_count_method": "response_poisson",
        "rng_seed": _RNG_SEED,
        "measurement_time_s": _LIVE_TIME_S,
        "adaptive_dwell": False,
        "scatter_gain": 0.0,
        "dead_time_s": 0.0,
        "dead_time_tau_s": 0.0,
        "background_cps": 3.0,
        "response_poisson_line_resolved_fit": True,
        "response_poisson_photopeak_fusion": False,
        "normalize_line_intensities": False,
        "pf_line_resolved_shield_attenuation": False,
        "shield_thickness_scale": float(shield.thickness_scale),
        "shield_transmission_target": shield.transmission_target,
        "fe_shield_thickness_cm": float(shield.thickness_fe_cm),
        "pb_shield_thickness_cm": float(shield.thickness_pb_cm),
        "detector_model": {
            "crystal_radius_m": 0.038,
            "housing_thickness_m": 0.0015,
        },
        "pf_detector_count_radius_m": 0.0,
        "pf_detector_aperture_radius_m": 0.0,
        "pf_detector_aperture_samples": 1,
        "pf_detector_aperture_sampling": "solid_angle_cone",
        "obstacle_height_m": 2.0,
        "obstacle_material": "concrete",
        "pf_obstacle_mu_by_isotope": {_ISOTOPE: 0.0},
        "pf_buildup": {
            "fe_coeff": 0.0,
            "pb_coeff": 0.0,
            "obstacle_coeff": 0.0,
        },
        "pf_obstacle_source_extent_radius_m": 0.0,
        "pf_obstacle_source_extent_samples": 1,
        "environment": {
            "size_x": _ROOM_SIZE_XYZ[0],
            "size_y": _ROOM_SIZE_XYZ[1],
            "size_z": _ROOM_SIZE_XYZ[2],
        },
    }


def _environment_and_obstacles() -> tuple[EnvironmentConfig, ObstacleGrid]:
    """Return the embedded room and its intentionally empty obstacle grid."""
    environment = EnvironmentConfig(
        size_x=_ROOM_SIZE_XYZ[0],
        size_y=_ROOM_SIZE_XYZ[1],
        size_z=_ROOM_SIZE_XYZ[2],
        detector_position=_FIXED_ROUTE_XYZ[0],
    )
    obstacle_grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(2, 2),
        blocked_cells=(),
    )
    return environment, obstacle_grid


def _find_source_patch(patches: SurfacePatchSet) -> tuple[int, SurfacePatch]:
    """Return the unique base patch selected as the demo source support."""
    matches = [
        (index, patch)
        for index, patch in enumerate(patches.patches)
        if patch.object_id == _SOURCE_PATCH_OBJECT_ID
        and np.allclose(
            patch.centroid_xyz,
            _SOURCE_PATCH_CENTROID_XYZ,
            rtol=0.0,
            atol=1.0e-12,
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "The analytic MLE demo requires exactly one matching source patch."
        )
    return matches[0]


def _route_plan() -> _RouteObservationPlan:
    """Return immutable arrays describing the fixed positions and shield pairs."""
    return _RouteObservationPlan(
        detector_positions_xyz=np.asarray(_FIXED_ROUTE_XYZ, dtype=np.float64),
        fe_indices=np.asarray(
            [pair[0] for pair in _FIXED_SHIELD_PAIRS],
            dtype=np.int64,
        ),
        pb_indices=np.asarray(
            [pair[1] for pair in _FIXED_SHIELD_PAIRS],
            dtype=np.int64,
        ),
        live_times_s=np.full(
            len(_FIXED_ROUTE_XYZ),
            _LIVE_TIME_S,
            dtype=np.float64,
        ),
    )


def _route_response_diagnostics(
    plan: _RouteObservationPlan,
    patches: SurfacePatchSet,
    kernel: ContinuousKernel,
) -> tuple[int, float]:
    """Verify route identifiability without using responses as observations."""
    response = build_count_response(plan, patches, (_ISOTOPE,), kernel)[:, :, 0]
    rank = int(np.linalg.matrix_rank(response))
    condition_number = float(np.linalg.cond(response))
    if rank != patches.patch_count:
        raise RuntimeError(
            "The analytic MLE demo route is not full column rank for its patches."
        )
    if not np.isfinite(condition_number):
        raise RuntimeError("The analytic MLE demo route has a singular response.")
    return rank, condition_number


def _simulator_reset_payload(
    source: PointSource,
    obstacle_grid: ObstacleGrid,
    runtime_config: dict[str, object],
) -> dict[str, object]:
    """Return a fully local scene payload for the analytic simulator."""
    return {
        "sources": [
            {
                "isotope": source.isotope,
                "position": [float(value) for value in source.position],
                "intensity_cps_1m": float(source.intensity_cps_1m),
            }
        ],
        "obstacle_origin_xy": [float(value) for value in obstacle_grid.origin],
        "obstacle_cell_size_m": float(obstacle_grid.cell_size),
        "obstacle_grid_shape": [int(value) for value in obstacle_grid.grid_shape],
        "obstacle_cells": [list(cell) for cell in obstacle_grid.blocked_cells],
        "obstacle_material": str(runtime_config["obstacle_material"]),
        "detector_model": dict(runtime_config["detector_model"]),
    }


def _selected_covariance(
    result: RuntimeCountResult,
    isotopes: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Return a complete covariance restricted to the persisted channels."""
    names = tuple(str(name) for name in isotopes)
    source_covariance = result.covariance or {}
    covariance: dict[str, dict[str, float]] = {}
    for row_name in names:
        source_row = source_covariance.get(row_name, {})
        row: dict[str, float] = {}
        for column_name in names:
            if column_name in source_row:
                value = float(source_row[column_name])
            elif row_name == column_name:
                value = float(result.variances.get(row_name, 1.0))
            else:
                value = 0.0
            row[column_name] = value
        covariance[row_name] = row
    return covariance


def _sample_measurement_records(
    *,
    source: PointSource,
    plan: _RouteObservationPlan,
    obstacle_grid: ObstacleGrid,
    runtime_config: dict[str, object],
    kernel: ContinuousKernel,
) -> tuple[MeasurementRecord, ...]:
    """Sample spectra and finalize runtime-extracted count observations."""
    spectrum_config = spectrum_config_from_runtime_config(runtime_config)
    decomposer = SpectralDecomposer(spectrum_config, use_gpu=False)
    extractor = RuntimeCountExtractor(
        decomposer,
        count_method="response_poisson",
    )
    runtime = AnalyticSimulationRuntime(
        sources=[source],
        decomposer=decomposer,
        mu_by_isotope=dict(kernel.mu_by_isotope or {}),
        shield_params=kernel.shield_params,
        rng_seed=_RNG_SEED,
        obstacle_height_m=float(runtime_config["obstacle_height_m"]),
        obstacle_material=str(runtime_config["obstacle_material"]),
        scatter_gain=float(runtime_config["scatter_gain"]),
        dead_time_s=float(runtime_config["dead_time_s"]),
        detector_model=dict(runtime_config["detector_model"]),
    )
    records: list[MeasurementRecord] = []
    try:
        runtime.reset(
            _simulator_reset_payload(source, obstacle_grid, runtime_config)
        )
        for step_id, detector_position in enumerate(plan.detector_positions_xyz):
            travel_time_s = 0.0 if step_id == 0 else _TRAVEL_TIME_S
            observation = runtime.step(
                SimulationCommand(
                    step_id=step_id,
                    target_pose_xyz=tuple(float(value) for value in detector_position),
                    target_base_yaw_rad=0.0,
                    fe_orientation_index=int(plan.fe_indices[step_id]),
                    pb_orientation_index=int(plan.pb_indices[step_id]),
                    dwell_time_s=float(plan.live_times_s[step_id]),
                    travel_time_s=travel_time_s,
                    shield_actuation_time_s=_SHIELD_ACTUATION_TIME_S,
                )
            )
            spectrum = np.asarray(observation.spectrum_counts, dtype=np.float64)
            count_result = extractor.extract(
                spectrum,
                live_time_s=float(plan.live_times_s[step_id]),
                detect_threshold_abs=0.1,
                detect_threshold_rel=0.2,
                detect_threshold_rel_by_isotope={},
                min_peaks_by_isotope={_ISOTOPE: 1},
                spectrum_variance=None,
                transport_metadata=observation.metadata,
            )
            isotope_count = float(count_result.counts.get(_ISOTOPE, 0.0))
            if not np.isfinite(isotope_count) or isotope_count < 0.0:
                raise RuntimeError(
                    "Runtime spectrum processing returned an invalid isotope count."
                )
            records.append(
                MeasurementRecord.from_simulation_observation(
                    observation,
                    station_id=step_id,
                    live_time_s=float(plan.live_times_s[step_id]),
                    travel_time_s=travel_time_s,
                    shield_actuation_time_s=_SHIELD_ACTUATION_TIME_S,
                    spectrum_variance=None,
                    counts_by_isotope={_ISOTOPE: isotope_count},
                    count_covariance_by_isotope=_selected_covariance(
                        count_result,
                        (_ISOTOPE,),
                    ),
                    metadata={
                        "backend": str(observation.metadata.get("backend", "analytic")),
                        "transport_backend": str(
                            observation.metadata.get("transport_backend", "python")
                        ),
                        "count_method": "response_poisson",
                        "detected_isotopes": sorted(
                            set(count_result.detected) & {_ISOTOPE}
                        ),
                        "observation_generation": (
                            "sampled_analytic_runtime_spectrum"
                        ),
                        "spectrum_variance_available": False,
                    },
                    include_observation_metadata=False,
                )
            )
    finally:
        runtime.close()
    return tuple(records)


def _run_context(
    *,
    runtime_config: dict[str, object],
    environment: EnvironmentConfig,
    obstacle_grid: ObstacleGrid,
    mle_config: MLEConfig,
    response_rank: int,
    response_condition_number: float,
) -> RunContext:
    """Build the self-contained context persisted beside demo observations."""
    environment_payload: dict[str, object] = {
        "size_x": float(environment.size_x),
        "size_y": float(environment.size_y),
        "size_z": float(environment.size_z),
        "room_size_xyz": [float(value) for value in _ROOM_SIZE_XYZ],
        "detector_position": [float(value) for value in environment.detector()],
        "environment_mode": "analytic_mle_demo",
        "obstacle_grid": obstacle_grid.to_dict(),
    }
    return RunContext(
        repository_commit=repository_commit(),
        runtime_config=runtime_config,
        environment=environment_payload,
        sim_backend="analytic",
        spectrum_count_method="response_poisson",
        isotopes=(_ISOTOPE,),
        obstacle_layout_path=None,
        source_rate_model="detector_cps_1m",
        run_id="analytic-surface-mle-demo",
        metadata={
            "demo_name": "analytic_surface_mle",
            "demo_schema_version": 1,
            "runtime_repository": "3D_estimation",
            "observation_generation": (
                "AnalyticSimulationRuntime -> RuntimeCountExtractor"
            ),
            "fixed_route_xyz": [list(position) for position in _FIXED_ROUTE_XYZ],
            "fixed_shield_pairs": [
                [int(fe_index), int(pb_index)]
                for fe_index, pb_index in _FIXED_SHIELD_PAIRS
            ],
            "route_response_rank": int(response_rank),
            "route_response_condition_number": float(
                response_condition_number
            ),
            "recommended_mle_config": mle_config.to_dict(),
        },
    )


def build_analytic_mle_demo() -> AnalyticMLEDemoScenario:
    """Build one deterministic sampled-observation analytic MLE scenario."""
    runtime_config = _runtime_config()
    environment, obstacle_grid = _environment_and_obstacles()
    mle_config = analytic_mle_demo_config()
    patches = build_surface_patches(
        environment,
        obstacle_grid,
        mle_config.patch_spacing_m,
        obstacle_height_m=mle_config.obstacle_height_m,
        quadrature_points_per_patch=mle_config.quadrature_order,
    )
    source_patch_index, source_patch = _find_source_patch(patches)
    observation_model = build_runtime_observation_model(
        runtime_config,
        isotopes=(_ISOTOPE,),
    )
    kernel = continuous_kernel_from_observation_model(
        observation_model,
        obstacle_grid=obstacle_grid,
        use_gpu=False,
    )
    plan = _route_plan()
    response_rank, response_condition_number = _route_response_diagnostics(
        plan,
        patches,
        kernel,
    )
    source = PointSource(
        isotope=_ISOTOPE,
        position=tuple(float(value) for value in source_patch.centroid_xyz),
        intensity_cps_1m=_SOURCE_STRENGTH_CPS_1M,
    )
    records = _sample_measurement_records(
        source=source,
        plan=plan,
        obstacle_grid=obstacle_grid,
        runtime_config=runtime_config,
        kernel=kernel,
    )
    context = _run_context(
        runtime_config=runtime_config,
        environment=environment,
        obstacle_grid=obstacle_grid,
        mle_config=mle_config,
        response_rank=response_rank,
        response_condition_number=response_condition_number,
    )
    return AnalyticMLEDemoScenario(
        context=context,
        records=records,
        mle_config=mle_config,
        environment=environment,
        obstacle_grid=obstacle_grid,
        kernel=kernel,
        patches=patches,
        source_patch_index=source_patch_index,
        source_patch_id=source_patch.patch_id,
        source_strength_cps_1m=_SOURCE_STRENGTH_CPS_1M,
        route_response_rank=response_rank,
        route_response_condition_number=response_condition_number,
    )


def create_analytic_mle_demo_log(
    run_dir: str | Path,
    *,
    scenario: AnalyticMLEDemoScenario | None = None,
) -> Path:
    """Create a deterministic versioned log from sampled runtime observations."""
    demo = build_analytic_mle_demo() if scenario is None else scenario
    if not isinstance(demo, AnalyticMLEDemoScenario):
        raise TypeError("scenario must be an AnalyticMLEDemoScenario or None.")
    return save_measurement_log(run_dir, demo.context, demo.records)


__all__ = [
    "AnalyticMLEDemoScenario",
    "analytic_mle_demo_config",
    "build_analytic_mle_demo",
    "create_analytic_mle_demo_log",
]
