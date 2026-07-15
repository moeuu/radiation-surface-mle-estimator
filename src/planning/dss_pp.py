"""Differential Shield-Signature Path Planning.

DSS-PP plans over a joint robot-pose and shield-program action. The module
uses the same PF expected-count kernel as the online estimator; it does not
generate spectra or replace Geant4 transport.
"""

from __future__ import annotations

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from measurement.continuous_kernels import ContinuousKernel
from measurement.detector_geometry import DEFAULT_PF_DETECTOR_APERTURE_SAMPLES
from pf.estimator import RotatingShieldPFEstimator
from pf.likelihood import expected_counts_per_source
from planning.pose_selection import (
    _auto_scale_observation_penalty,
    _minimum_observation_feasible_mask,
    estimate_lambda_cost,
    minimum_observation_shortfall,
)
from planning.traversability import shortest_grid_path_length
from runtime_defaults import (
    DEFAULT_MEASUREMENT_TIME_S,
    DEFAULT_ROBOT_SPEED_M_S,
    DEFAULT_ROTATION_OVERHEAD_S,
)


_DSS_PP_POSE_EVAL_CONTEXT: dict[str, object] | None = None
_DSS_PP_PATH_LENGTH_CACHE: dict[tuple[int, tuple[int, int], tuple[int, int]], float] = {}
_DSS_PP_PATH_LENGTH_CACHE_MAX = 20000


@dataclass(frozen=True)
class ShieldProgram:
    """Represent a short sequence of Fe/Pb shield orientation pairs."""

    name: str
    pair_ids: tuple[int, ...]
    kind: str


@dataclass(frozen=True)
class SignatureMode:
    """Represent one posterior source mode used for shield signatures."""

    isotope: str
    position_xyz: NDArray[np.float64]
    strength_cps_1m: float
    weight: float
    spread_m: float


@dataclass(frozen=True)
class DSSPPConfig:
    """Configuration for Differential Shield-Signature Path Planning."""

    horizon: int = 2
    beam_width: int = 8
    max_programs: int = 40
    program_length: int = 2
    mode_cluster_radius_m: float = 1.5
    max_modes_per_isotope: int = 4
    planning_particles: int | None = None
    planning_method: str | None = None
    live_time_s: float = DEFAULT_MEASUREMENT_TIME_S
    lambda_eig: float = 1.0
    lambda_signature: float = 1.0
    lambda_distance: float | None = None
    lambda_time: float = 0.0
    lambda_rotation: float = 0.15
    lambda_dose: float = 0.0
    lambda_coverage: float = 0.0
    lambda_bearing_diversity: float = 0.0
    lambda_frontier: float = 0.0
    lambda_turn_smoothness: float = 0.0
    lambda_temporal_separation: float = 0.0
    lambda_count_utility: float = 0.0
    lambda_local_orbit: float = 0.0
    lambda_station_condition: float = 0.0
    lambda_correlation_reduction: float = 0.0
    lambda_cardinality_discrimination: float = 0.0
    lambda_isotope_balance: float = 0.0
    lambda_environment_signature: float = 0.0
    lambda_occlusion_boundary: float = 0.0
    lambda_elevation_signature: float = 0.0
    lambda_elevation_condition: float = 0.0
    lambda_vertical_environment_signature: float = 0.0
    residual_signature_weight: float = 1.0
    eta_observation: float = 1.0
    eta_differential: float = 1.0
    eta_count_balance: float = 0.5
    eta_revisit: float = 0.0
    min_observation_counts: float = 0.0
    enforce_min_observation: bool = True
    signature_std_min_counts: float = 1.0
    count_variance_floor: float = 1.0
    coverage_radius_m: float = 3.0
    coverage_grid_max_cells: int = 5000
    coverage_floor_quantile: float = 0.0
    coverage_floor_weight: float = 0.0
    min_station_separation_m: float = 0.0
    detector_aperture_samples: int = DEFAULT_PF_DETECTOR_APERTURE_SAMPLES
    robot_speed_m_s: float = DEFAULT_ROBOT_SPEED_M_S
    rotation_overhead_s: float = DEFAULT_ROTATION_OVERHEAD_S
    augment_candidates: bool = True
    max_augmented_candidates: int = 256
    ring_radii_m: tuple[float, ...] = (2.0, 3.5, 5.0)
    ring_angles: int = 12
    count_utility_saturation_counts: float = 250.0
    local_orbit_sigma_m: float = 0.75
    station_condition_ridge: float = 1.0e-3
    station_condition_min_singular_weight: float = 0.0
    station_condition_inverse_condition_weight: float = 0.0
    station_condition_coherence_weight: float = 0.0
    environment_contrast_threshold: float = 0.25
    environment_signature_score_clip: float = 3.0
    occlusion_boundary_step_m: float = 0.5
    elevation_pair_z_scale_m: float = 2.0
    elevation_pair_xy_scale_m: float = 4.0
    elevation_angle_threshold_deg: float = 15.0
    rng_seed: int | None = 0
    eig_candidate_limit: int | None = 8
    temporal_cover_weight: float = 1.0
    temporal_logdet_weight: float = 0.25
    temporal_decorrelation_weight: float = 0.5
    temporal_pair_contrast_threshold: float = 0.25
    temporal_logdet_ridge: float = 1.0e-3
    temporal_cover_programs: int = 1
    temporal_cover_beam_width: int = 4
    program_eval_workers: int | None = None
    candidate_preselect_enable: bool = True
    candidate_preselect_min: int = 32
    candidate_preselect_multiplier: int = 8
    remaining_budget_guidance: bool = False
    remaining_station_estimate: int | None = None
    remaining_budget_urgency_stations: int = 4
    remaining_route_weight: float = 0.0
    remaining_route_distance_weight: float = 0.5
    remaining_route_revisit_weight: float = 1.0
    remaining_route_turn_weight: float = 0.75
    remaining_route_backtrack_weight: float = 1.0
    remaining_route_coverage_weight: float = 0.5
    remaining_route_frontier_weight: float = 0.5
    cardinality_evidence_gap_target: float = 10.0
    cardinality_bic_parameter_count_per_source: int = 4
    same_isotope_direct_separation_guard: bool = True
    same_isotope_direct_separation_epsilon: float = 1.0e-9
    include_runtime_rescue_modes: bool = True
    runtime_rescue_mode_weight: float = 0.5
    include_global_surface_rescue_modes: bool = True
    global_surface_rescue_mode_weight: float = 0.75
    recovery_isotopes: tuple[str, ...] = ()
    recovery_isotope_mode_weight_multiplier: float = 2.0
    weak_mode_weight_floor: float = 0.0
    dominant_mode_weight_cap: float = 1.0
    high_surface_pair_boost: float = 1.0
    high_surface_cross_stratum_boost: float = 1.0
    high_surface_z_fraction: float = 0.75
    high_surface_pair_distance_m: float = 0.0
    forced_program_pair_ids: tuple[int, ...] | None = None
    diagnostic_ranked_node_limit: int = 64
    explicit_mode_switch: bool = False
    planner_mode: str = "balanced"


@dataclass(frozen=True)
class DSSPPNode:
    """Store one candidate station and shield program evaluation."""

    pose_index: int
    pose_xyz: NDArray[np.float64]
    program: ShieldProgram
    score: float
    static_score: float
    distance_weight: float
    observation_penalty_weight: float
    information_gain: float
    signature_score: float
    temporal_separation_score: float
    observation_penalty: float
    count_balance_penalty: float
    differential_penalty: float
    dose_score: float
    count_utility: float
    coverage_gain: float
    revisit_penalty: float
    bearing_diversity_gain: float
    frontier_gain: float
    turn_penalty: float
    local_orbit_gain: float
    station_condition_gain: float
    correlation_reduction_gain: float
    isotope_balance_gain: float
    environment_signature_score: float
    occlusion_boundary_gain: float
    elevation_signature_score: float
    elevation_condition_gain: float
    vertical_environment_signature_score: float
    remaining_route_pressure: float = 0.0
    remaining_route_penalty: float = 0.0
    remaining_route_gain: float = 0.0
    cardinality_gap_gain: float = 0.0


@dataclass(frozen=True)
class DSSPPResult:
    """Return the selected receding-horizon DSS-PP action."""

    next_pose: NDArray[np.float64]
    next_pose_index: int
    shield_program: ShieldProgram
    score: float
    sequence: tuple[DSSPPNode, ...]
    diagnostics: dict[str, Any]


def _node_diagnostic_payload(node: DSSPPNode, rank: int) -> dict[str, object]:
    """Return a JSON-serializable diagnostic payload for one DSS-PP node."""
    return {
        "rank": int(rank),
        "pose_index": int(node.pose_index),
        "pose_xyz": [float(value) for value in np.asarray(node.pose_xyz, dtype=float)],
        "program_name": str(node.program.name),
        "program_kind": str(node.program.kind),
        "pair_ids": [int(value) for value in node.program.pair_ids],
        "score": float(node.score),
        "static_score": float(node.static_score),
        "distance_weight": float(node.distance_weight),
        "observation_penalty_weight": float(node.observation_penalty_weight),
        "information_gain": float(node.information_gain),
        "signature_score": float(node.signature_score),
        "temporal_separation_score": float(node.temporal_separation_score),
        "elevation_signature_score": float(node.elevation_signature_score),
        "observation_penalty": float(node.observation_penalty),
        "count_balance_penalty": float(node.count_balance_penalty),
        "differential_penalty": float(node.differential_penalty),
        "dose_score": float(node.dose_score),
        "count_utility": float(node.count_utility),
        "coverage_gain": float(node.coverage_gain),
        "revisit_penalty": float(node.revisit_penalty),
        "bearing_diversity_gain": float(node.bearing_diversity_gain),
        "frontier_gain": float(node.frontier_gain),
        "turn_penalty": float(node.turn_penalty),
        "local_orbit_gain": float(node.local_orbit_gain),
        "station_condition_gain": float(node.station_condition_gain),
        "correlation_reduction_gain": float(node.correlation_reduction_gain),
        "cardinality_gap_gain": float(node.cardinality_gap_gain),
        "isotope_balance_gain": float(node.isotope_balance_gain),
        "environment_signature_score": float(node.environment_signature_score),
        "occlusion_boundary_gain": float(node.occlusion_boundary_gain),
        "elevation_condition_gain": float(node.elevation_condition_gain),
        "vertical_environment_signature_score": float(
            node.vertical_environment_signature_score
        ),
        "remaining_route_pressure": float(node.remaining_route_pressure),
        "remaining_route_penalty": float(node.remaining_route_penalty),
        "remaining_route_gain": float(node.remaining_route_gain),
    }


def _mode_diagnostic_payload(mode: SignatureMode, index: int) -> dict[str, object]:
    """Return a compact diagnostic payload for one posterior source mode."""
    return {
        "index": int(index),
        "pos": [float(value) for value in np.asarray(mode.position_xyz, dtype=float)],
        "q": float(mode.strength_cps_1m),
        "weight": float(mode.weight),
        "spread_m": float(mode.spread_m),
    }


def _component_leader_payloads(nodes: Sequence[DSSPPNode]) -> dict[str, dict[str, object]]:
    """Return best-node diagnostics for individual DSS-PP score components."""
    node_list = list(nodes)
    if not node_list:
        return {}
    selectors: dict[str, Any] = {
        "score": lambda node: float(node.score),
        "information_gain": lambda node: float(node.information_gain),
        "signature": lambda node: float(node.signature_score),
        "temporal_separation": lambda node: float(node.temporal_separation_score),
        "elevation_separation": lambda node: float(node.elevation_signature_score),
        "coverage": lambda node: float(node.coverage_gain),
        "count_utility": lambda node: float(node.count_utility),
        "correlation_reduction": lambda node: float(node.correlation_reduction_gain),
        "cardinality_gap": lambda node: float(node.cardinality_gap_gain),
        "isotope_balance": lambda node: float(node.isotope_balance_gain),
        "obstacle_signature": lambda node: float(node.environment_signature_score),
        "vertical_obstacle_signature": lambda node: float(
            node.vertical_environment_signature_score
        ),
        "remaining_route": lambda node: float(node.remaining_route_gain)
        - float(node.remaining_route_penalty),
    }
    leaders: dict[str, dict[str, object]] = {}
    for name, selector in selectors.items():
        finite_nodes = [
            node
            for node in node_list
            if np.isfinite(float(selector(node)))
        ]
        if not finite_nodes:
            continue
        leader = max(finite_nodes, key=lambda node: float(selector(node)))
        payload = _node_diagnostic_payload(leader, 1)
        payload["component_value"] = float(selector(leader))
        leaders[name] = payload
    return leaders


def _normalise_weights(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return normalized nonnegative weights with a uniform fallback."""
    arr = np.asarray(weights, dtype=float).ravel()
    if arr.size == 0:
        return arr
    arr = np.maximum(arr, 0.0)
    total = float(np.sum(arr))
    if total <= 0.0:
        return np.ones(arr.size, dtype=float) / float(arr.size)
    return arr / total


def _rebalance_signature_mode_weights(
    modes: Sequence[SignatureMode],
    *,
    weak_floor: float,
    dominant_cap: float,
) -> list[SignatureMode]:
    """Return modes with planner-only floor/cap weights for weak-mode visibility."""
    mode_list = list(modes)
    count = len(mode_list)
    if count <= 1:
        return mode_list
    weights = _normalise_weights(
        np.asarray([max(float(mode.weight), 0.0) for mode in mode_list], dtype=float)
    )
    floor = max(0.0, min(float(weak_floor), 1.0 / float(count)))
    if floor > 0.0:
        weights = np.maximum(weights, floor)
        weights = _normalise_weights(weights)
    cap = float(dominant_cap)
    if cap > 0.0:
        cap = min(1.0, max(cap, 1.0 / float(count)))
        for _ in range(count):
            over = weights > cap
            if not np.any(over):
                break
            excess = float(np.sum(weights[over] - cap))
            weights[over] = cap
            under = ~over
            if not np.any(under) or excess <= 0.0:
                break
            under_total = float(np.sum(weights[under]))
            if under_total <= 0.0:
                weights[under] += excess / float(np.count_nonzero(under))
            else:
                weights[under] += excess * weights[under] / under_total
        weights = _normalise_weights(weights)
    return [
        replace(mode, weight=float(weight))
        for mode, weight in zip(mode_list, weights)
    ]


def _estimator_room_z(estimator: RotatingShieldPFEstimator) -> float:
    """Return the source-prior room height used by planner high-surface logic."""
    hi = getattr(getattr(estimator, "pf_config", None), "position_max", (0.0, 0.0, 0.0))
    try:
        return max(float(np.asarray(hi, dtype=float).reshape(3)[2]), 0.0)
    except (TypeError, ValueError):
        return 0.0


def _high_surface_pair_priority_weights(
    modes: Sequence[SignatureMode],
    *,
    config: DSSPPConfig,
    room_z_m: float,
) -> NDArray[np.float64]:
    """
    Return pair weights that prioritize ceiling and high-wall mode separation.

    The vector follows ``np.triu_indices(len(modes), k=1)`` order.  It only
    changes how existing batched separation scores are weighted; it does not
    alter the response model or introduce a transport approximation.
    """
    mode_count = len(modes)
    if mode_count < 2:
        return np.ones(0, dtype=float)
    pair_i, pair_j = np.triu_indices(mode_count, k=1)
    boost = max(float(config.high_surface_pair_boost), 1.0)
    cross_stratum_boost = max(
        float(config.high_surface_cross_stratum_boost),
        1.0,
    )
    if boost <= 1.0 and cross_stratum_boost <= 1.0:
        return np.ones(pair_i.size, dtype=float)
    positions = np.vstack(
        [np.asarray(mode.position_xyz, dtype=float).reshape(3) for mode in modes]
    )
    room_z = max(float(room_z_m), float(np.max(positions[:, 2])), 1.0e-9)
    threshold = float(np.clip(config.high_surface_z_fraction, 0.0, 1.0)) * room_z
    high = positions[:, 2] >= threshold
    if not np.any(high):
        return np.ones(pair_i.size, dtype=float)
    distances = np.linalg.norm(positions[pair_i] - positions[pair_j], axis=1)
    max_distance = max(float(config.high_surface_pair_distance_m), 0.0)
    distance_ok = (
        np.ones(pair_i.size, dtype=bool)
        if max_distance <= 0.0
        else distances <= max_distance
    )
    either_high = high[pair_i] | high[pair_j]
    both_high = high[pair_i] & high[pair_j]
    priorities = np.ones(pair_i.size, dtype=float)
    if boost > 1.0:
        priorities[either_high & distance_ok] = np.sqrt(boost)
        priorities[both_high & distance_ok] = boost
    if cross_stratum_boost > 1.0:
        ceiling_threshold = max(0.0, 0.95 * room_z)
        ceiling_like = positions[:, 2] >= ceiling_threshold
        cross_stratum = both_high & (ceiling_like[pair_i] != ceiling_like[pair_j])
        priorities[cross_stratum & distance_ok] = np.maximum(
            priorities[cross_stratum & distance_ok],
            boost * cross_stratum_boost,
        )
    return priorities


def _pair_id(fe_index: int, pb_index: int, num_orients: int) -> int:
    """Return the flattened orientation-pair id."""
    return int(fe_index) * int(num_orients) + int(pb_index)


def _pair_indices(pair_id: int, num_orients: int) -> tuple[int, int]:
    """Return Fe and Pb indices from a flattened pair id."""
    return int(pair_id) // int(num_orients), int(pair_id) % int(num_orients)


def _opposite_indices(normals: NDArray[np.float64]) -> list[int]:
    """Return the approximately opposite octant index for each normal."""
    normal_arr = np.asarray(normals, dtype=float)
    opposite: list[int] = []
    for normal in normal_arr:
        dots = normal_arr @ normal
        opposite.append(int(np.argmin(dots)))
    return opposite


def _cycle_program_pairs(pair_ids: Sequence[int], length: int) -> tuple[int, ...]:
    """Return a non-empty pair-id sequence repeated to the requested length."""
    base = tuple(int(pair_id) for pair_id in pair_ids)
    target = max(1, int(length))
    if not base:
        return tuple()
    repeats = int(np.ceil(target / float(len(base))))
    return (base * repeats)[:target]


def build_shield_program_library(
    normals: NDArray[np.float64],
    *,
    program_length: int = 2,
    max_programs: int = 40,
) -> list[ShieldProgram]:
    """Build bearing, material, occlusion, and vertical shield programs."""
    normal_arr = np.asarray(normals, dtype=float)
    if normal_arr.ndim != 2 or normal_arr.shape[1] != 3:
        raise ValueError("normals must be shaped (N, 3).")
    num_orients = int(normal_arr.shape[0])
    if num_orients <= 0:
        return []
    length = max(1, int(program_length))
    opposite = _opposite_indices(normal_arr)
    programs: list[ShieldProgram] = []
    for idx in range(num_orients):
        opp = opposite[idx]
        blocked = _pair_id(idx, idx, num_orients)
        unblocked = _pair_id(opp, opp, num_orients)
        fe_only = _pair_id(idx, opp, num_orients)
        pb_only = _pair_id(opp, idx, num_orients)
        programs.append(
            ShieldProgram(
                name=f"bearing_split_{idx}",
                pair_ids=_cycle_program_pairs((blocked, unblocked), length),
                kind="bearing_split",
            )
        )
        programs.append(
            ShieldProgram(
                name=f"material_split_{idx}",
                pair_ids=_cycle_program_pairs((fe_only, pb_only), length),
                kind="material_split",
            )
        )
        programs.append(
            ShieldProgram(
                name=f"occlusion_test_{idx}",
                pair_ids=_cycle_program_pairs((unblocked, blocked, fe_only), length),
                kind="occlusion_test",
            )
        )
    up_idx = int(np.argmax(normal_arr[:, 2]))
    down_idx = int(np.argmin(normal_arr[:, 2]))
    programs.append(
        ShieldProgram(
            name="vertical_split_up_down",
            pair_ids=_cycle_program_pairs(
                (
                    _pair_id(up_idx, up_idx, num_orients),
                    _pair_id(down_idx, down_idx, num_orients),
                ),
                length,
            ),
            kind="vertical_split",
        )
    )
    programs.append(
        ShieldProgram(
            name="vertical_material_split_up_down",
            pair_ids=_cycle_program_pairs(
                (
                    _pair_id(up_idx, down_idx, num_orients),
                    _pair_id(down_idx, up_idx, num_orients),
                ),
                length,
            ),
            kind="vertical_material_split",
        )
    )
    for idx in range(num_orients):
        opp = opposite[idx]
        programs.append(
            ShieldProgram(
                name=f"elevation_bearing_split_{idx}",
                pair_ids=_cycle_program_pairs(
                    (
                        _pair_id(idx, up_idx, num_orients),
                        _pair_id(opp, down_idx, num_orients),
                        _pair_id(up_idx, idx, num_orients),
                        _pair_id(down_idx, opp, num_orients),
                    ),
                    length,
                ),
                kind="elevation_bearing_split",
            )
        )
    deduped: dict[tuple[int, ...], ShieldProgram] = {}
    for program in programs:
        if program.pair_ids:
            deduped.setdefault(program.pair_ids, program)
    return list(deduped.values())[: max(1, int(max_programs))]


def _continuous_kernel_for_estimator(
    estimator: RotatingShieldPFEstimator,
    *,
    detector_aperture_samples: int | None = None,
) -> ContinuousKernel:
    """Build a ContinuousKernel matching the estimator."""
    return estimator.continuous_kernel(
        detector_aperture_samples=detector_aperture_samples,
    )


def _response_scales_for_all_pairs(
    estimator: RotatingShieldPFEstimator,
    isotope: str,
    num_orients: int,
) -> NDArray[np.float64]:
    """Return source-response scales for every Fe/Pb shield pair."""
    pair_ids = np.arange(max(int(num_orients), 1) ** 2, dtype=int)
    fe_indices = pair_ids // max(int(num_orients), 1)
    pb_indices = pair_ids % max(int(num_orients), 1)
    return estimator.response_scales_for_measurements(
        isotope,
        fe_indices.astype(np.int64, copy=False),
        pb_indices.astype(np.int64, copy=False),
    )


def _cluster_source_samples(
    isotope: str,
    positions: list[NDArray[np.float64]],
    strengths: list[float],
    weights: list[float],
    *,
    radius_m: float,
    max_modes: int,
) -> list[SignatureMode]:
    """Cluster weighted posterior source samples into signature modes."""
    if not positions:
        return []
    pos_arr = np.asarray(positions, dtype=float)
    str_arr = np.asarray(strengths, dtype=float)
    w_arr = _normalise_weights(np.asarray(weights, dtype=float))
    order = np.argsort(w_arr)[::-1]
    clusters: list[list[int]] = []
    centers: list[NDArray[np.float64]] = []
    for idx in order:
        pos = pos_arr[int(idx)]
        assigned = False
        for cluster_idx, center in enumerate(centers):
            if float(np.linalg.norm(pos - center)) <= float(radius_m):
                clusters[cluster_idx].append(int(idx))
                cluster_weights = w_arr[clusters[cluster_idx]]
                centers[cluster_idx] = np.average(
                    pos_arr[clusters[cluster_idx]],
                    axis=0,
                    weights=cluster_weights,
                )
                assigned = True
                break
        if not assigned:
            clusters.append([int(idx)])
            centers.append(pos.copy())
    modes: list[SignatureMode] = []
    for cluster in clusters:
        cluster_weights = w_arr[cluster]
        cluster_weight_sum = float(np.sum(cluster_weights))
        if cluster_weight_sum <= 0.0:
            continue
        center = np.average(pos_arr[cluster], axis=0, weights=cluster_weights)
        strength = float(np.average(str_arr[cluster], weights=cluster_weights))
        spread = float(
            np.sqrt(
                np.average(
                    np.sum((pos_arr[cluster] - center[None, :]) ** 2, axis=1),
                    weights=cluster_weights,
                )
            )
        )
        modes.append(
            SignatureMode(
                isotope=isotope,
                position_xyz=center.astype(float),
                strength_cps_1m=max(strength, 0.0),
                weight=cluster_weight_sum,
                spread_m=spread,
            )
        )
    modes.sort(key=lambda mode: mode.weight, reverse=True)
    return modes[: max(1, int(max_modes))]


def _rescue_modes_from_payload(
    isotope: str,
    entry: tuple[NDArray[np.float64], NDArray[np.float64], float] | None,
    *,
    mass: float,
    eps: float,
) -> list[SignatureMode]:
    """Return planner modes represented by a runtime/global rescue payload."""
    rescue_mass = max(float(mass), 0.0)
    if entry is None or rescue_mass <= 0.0:
        return []
    rescue_pos, rescue_q, _pf_mass = entry
    rescue_pos_arr = np.asarray(rescue_pos, dtype=float).reshape(-1, 3)
    rescue_q_arr = np.maximum(
        np.asarray(rescue_q, dtype=float).reshape(-1),
        0.0,
    )
    valid = (
        np.isfinite(rescue_pos_arr).all(axis=1)
        & np.isfinite(rescue_q_arr)
        & (rescue_q_arr > 0.0)
    )
    rescue_pos_arr = rescue_pos_arr[valid]
    rescue_q_arr = rescue_q_arr[valid]
    if rescue_pos_arr.shape[0] != rescue_q_arr.size or rescue_q_arr.size == 0:
        return []
    total_rescue_q = float(np.sum(rescue_q_arr))
    if total_rescue_q <= eps:
        rescue_rel = np.full(
            rescue_q_arr.size,
            1.0 / float(rescue_q_arr.size),
            dtype=float,
        )
    else:
        rescue_rel = rescue_q_arr / total_rescue_q
    return [
        SignatureMode(
            isotope=isotope,
            position_xyz=np.asarray(pos, dtype=float).reshape(3),
            strength_cps_1m=float(strength),
            weight=float(rescue_mass) * float(rel_weight),
            spread_m=0.0,
        )
        for pos, strength, rel_weight in zip(
            rescue_pos_arr,
            rescue_q_arr,
            rescue_rel,
        )
    ]


def _preserve_external_rescue_modes(
    modes: Sequence[SignatureMode],
    rescue_modes: Sequence[SignatureMode],
    *,
    radius_m: float,
    max_modes: int,
) -> list[SignatureMode]:
    """
    Keep distinct runtime rescue hypotheses visible to DSS-PP planning.

    The rescue list is already produced by all-history report/residual scoring.
    When PF modes fill the planner mode budget, this replaces the lowest-weight
    duplicate-free planner mode with a distinct rescue hypothesis instead of
    silently dropping every posterior-external hypothesis.
    """
    limit = max(1, int(max_modes))
    result = list(modes[:limit])
    if not rescue_modes:
        return result
    radius = max(float(radius_m), 0.0)
    rescue_order = sorted(
        rescue_modes,
        key=lambda mode: max(float(mode.weight), 0.0),
        reverse=True,
    )
    for rescue in rescue_order:
        rescue_pos = np.asarray(rescue.position_xyz, dtype=float).reshape(3)
        if result:
            distances = np.asarray(
                [
                    float(
                        np.linalg.norm(
                            np.asarray(mode.position_xyz, dtype=float).reshape(3)
                            - rescue_pos
                        )
                    )
                    for mode in result
                ],
                dtype=float,
            )
            if np.any(distances <= radius):
                continue
        if len(result) < limit:
            result.append(rescue)
            continue
        weights = np.asarray(
            [max(float(mode.weight), 0.0) for mode in result],
            dtype=float,
        )
        weakest_idx = int(np.argmin(weights))
        result[weakest_idx] = rescue
    result.sort(key=lambda mode: max(float(mode.weight), 0.0), reverse=True)
    return result[:limit]


def extract_signature_modes(
    estimator: RotatingShieldPFEstimator,
    *,
    max_particles: int | None = None,
    method: str | None = None,
    mode_cluster_radius_m: float = 1.5,
    max_modes_per_isotope: int = 4,
    tentative_weight_multiplier: float = 1.0,
    include_runtime_rescue_modes: bool = True,
    runtime_rescue_mode_weight: float = 0.5,
    include_global_surface_rescue_modes: bool = True,
    global_surface_rescue_mode_weight: float = 0.75,
    weak_mode_weight_floor: float = 0.0,
    dominant_mode_weight_cap: float = 1.0,
) -> dict[str, list[SignatureMode]]:
    """Extract isotope-wise posterior and runtime-rescue modes for planning."""
    particles = estimator.planning_particles(
        max_particles=max_particles,
        method=method,
    )
    modes_by_isotope: dict[str, list[SignatureMode]] = {}
    eps = 1e-12
    rescue_payload: dict[
        str,
        tuple[NDArray[np.float64], NDArray[np.float64], float],
    ] = {}
    global_rescue_payload: dict[
        str,
        tuple[NDArray[np.float64], NDArray[np.float64], float],
    ] = {}
    if bool(include_runtime_rescue_modes):
        rescue_getter = getattr(estimator, "runtime_report_rescue_modes", None)
        if callable(rescue_getter):
            try:
                rescue_payload = dict(rescue_getter())
            except (RuntimeError, ValueError, TypeError):
                rescue_payload = {}
    if bool(include_global_surface_rescue_modes):
        global_rescue_getter = getattr(
            estimator,
            "planning_surface_rescue_modes",
            None,
        )
        if callable(global_rescue_getter):
            try:
                global_rescue_payload = dict(global_rescue_getter())
            except (RuntimeError, ValueError, TypeError):
                global_rescue_payload = {}
    exclude_quarantined = bool(
        getattr(
            getattr(estimator, "pf_config", None),
            "pseudo_source_quarantine_excludes_runtime",
            False,
        )
    )
    for isotope in estimator.isotopes:
        positions: list[NDArray[np.float64]] = []
        strengths: list[float] = []
        sample_weights: list[float] = []
        external_rescue_modes: list[SignatureMode] = []
        if isotope in particles:
            states, weights = particles[isotope]
            norm_weights = _normalise_weights(np.asarray(weights, dtype=float))
            for state, particle_weight in zip(states, norm_weights):
                num_sources = int(state.num_sources)
                if num_sources <= 0:
                    continue
                tentative_raw = getattr(state, "tentative_sources", None)
                tentative = (
                    np.zeros(num_sources, dtype=bool)
                    if tentative_raw is None
                    else np.asarray(tentative_raw, dtype=bool)
                )
                if tentative.size != num_sources:
                    padded = np.zeros(num_sources, dtype=bool)
                    padded[: min(tentative.size, num_sources)] = tentative[:num_sources]
                    tentative = padded
                failed_raw = getattr(state, "verification_fail_streaks", None)
                failed = (
                    np.zeros(num_sources, dtype=int)
                    if failed_raw is None
                    else np.asarray(failed_raw, dtype=int)
                )
                if failed.size != num_sources:
                    padded = np.zeros(num_sources, dtype=int)
                    padded[: min(failed.size, num_sources)] = failed[:num_sources]
                    failed = padded
                quarantine_mask = tentative & (failed > 0)
                state_strengths = np.maximum(
                    np.asarray(state.strengths[:num_sources], dtype=float),
                    0.0,
                )
                total_strength = float(np.sum(state_strengths))
                if total_strength <= eps:
                    rel_strengths = (
                        np.ones(num_sources, dtype=float) / float(num_sources)
                    )
                else:
                    rel_strengths = state_strengths / total_strength
                for source_idx, (pos, strength, rel_strength) in enumerate(
                    zip(
                        state.positions[:num_sources],
                        state_strengths,
                        rel_strengths,
                    )
                ):
                    if exclude_quarantined and bool(quarantine_mask[source_idx]):
                        continue
                    positions.append(np.asarray(pos, dtype=float))
                    strengths.append(float(strength))
                    multiplier = (
                        max(float(tentative_weight_multiplier), 1.0)
                        if bool(tentative[source_idx])
                        else 1.0
                    )
                    sample_weights.append(
                        float(particle_weight) * float(rel_strength) * multiplier
                    )
        rescue_modes = _rescue_modes_from_payload(
            isotope,
            rescue_payload.get(isotope),
            mass=float(runtime_rescue_mode_weight),
            eps=eps,
        )
        global_rescue_modes = _rescue_modes_from_payload(
            isotope,
            global_rescue_payload.get(isotope),
            mass=float(global_surface_rescue_mode_weight),
            eps=eps,
        )
        external_rescue_modes.extend(rescue_modes)
        external_rescue_modes.extend(global_rescue_modes)
        for rescue_mode in external_rescue_modes:
            positions.append(np.asarray(rescue_mode.position_xyz, dtype=float))
            strengths.append(float(rescue_mode.strength_cps_1m))
            sample_weights.append(float(rescue_mode.weight))
        total_sample_weight = float(np.sum(np.maximum(sample_weights, 0.0)))
        if total_sample_weight > eps:
            external_rescue_modes = [
                replace(mode, weight=float(mode.weight) / total_sample_weight)
                for mode in external_rescue_modes
            ]
        modes = _cluster_source_samples(
            isotope,
            positions,
            strengths,
            sample_weights,
            radius_m=mode_cluster_radius_m,
            max_modes=max_modes_per_isotope,
        )
        modes = _preserve_external_rescue_modes(
            modes,
            external_rescue_modes,
            radius_m=mode_cluster_radius_m,
            max_modes=max_modes_per_isotope,
        )
        modes_by_isotope[isotope] = _rebalance_signature_mode_weights(
            modes,
            weak_floor=weak_mode_weight_floor,
            dominant_cap=dominant_mode_weight_cap,
        )
    return modes_by_isotope


def _is_free(map_api: object | None, point: NDArray[np.float64]) -> bool:
    """Return True when point is in free space according to map_api."""
    if map_api is None:
        return True
    if callable(map_api):
        return bool(map_api(point))
    for attr in ("is_free", "is_free_space", "is_free_cell"):
        fn = getattr(map_api, attr, None)
        if callable(fn):
            try:
                return bool(fn(point))
            except TypeError:
                continue
    return True


def _cell_center(map_api: object, cell: tuple[int, int], z_value: float) -> NDArray[np.float64]:
    """Return the world-space center of a map cell."""
    fn = getattr(map_api, "cell_center", None)
    if callable(fn):
        x_val, y_val = fn(cell)
    else:
        origin = getattr(map_api, "origin", (0.0, 0.0))
        cell_size = float(getattr(map_api, "cell_size", 1.0))
        x_val = float(origin[0]) + (float(cell[0]) + 0.5) * cell_size
        y_val = float(origin[1]) + (float(cell[1]) + 0.5) * cell_size
    return np.array([float(x_val), float(y_val), float(z_value)], dtype=float)


def _bounds_filter(
    points: list[NDArray[np.float64]],
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]] | None,
    map_api: object | None,
) -> list[NDArray[np.float64]]:
    """Filter points by bounds and traversability."""
    filtered: list[NDArray[np.float64]] = []
    if bounds_xyz is None:
        lo = None
        hi = None
    else:
        lo = np.asarray(bounds_xyz[0], dtype=float)
        hi = np.asarray(bounds_xyz[1], dtype=float)
    for point in points:
        pt = np.asarray(point, dtype=float)
        if pt.shape != (3,):
            continue
        if lo is not None and hi is not None:
            if bool(np.any(pt < lo) or np.any(pt > hi)):
                continue
        if _is_free(map_api, pt):
            filtered.append(pt)
    return filtered


def _dedupe_points(
    points: Sequence[NDArray[np.float64]],
    *,
    decimals: int = 3,
) -> NDArray[np.float64]:
    """Return unique points while preserving first occurrence order."""
    seen: set[tuple[float, float, float]] = set()
    unique: list[NDArray[np.float64]] = []
    for point in points:
        pt = np.asarray(point, dtype=float)
        key = tuple(float(v) for v in np.round(pt, int(decimals)))
        if key in seen:
            continue
        seen.add(key)
        unique.append(pt)
    if not unique:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(unique).astype(float)


def _bearing_angle_xy(source: NDArray[np.float64], pose: NDArray[np.float64]) -> float:
    """Return the planar bearing angle from source to pose."""
    delta = np.asarray(pose[:2], dtype=float) - np.asarray(source[:2], dtype=float)
    return float(np.arctan2(delta[1], delta[0]))


def _angle_distance_rad(left: float, right: float) -> float:
    """Return wrapped absolute angular distance in radians."""
    return float(abs(np.arctan2(np.sin(left - right), np.cos(left - right))))


def augment_candidate_stations(
    candidate_poses_xyz: NDArray[np.float64],
    *,
    modes_by_isotope: dict[str, list[SignatureMode]],
    current_pose_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
    map_api: object | None,
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]] | None,
    config: DSSPPConfig,
) -> NDArray[np.float64]:
    """Add posterior-ring, occlusion-boundary, and cross-bearing candidates."""
    base = np.asarray(candidate_poses_xyz, dtype=float)
    if base.ndim != 2 or base.shape[1] != 3:
        raise ValueError("candidate_poses_xyz must be shape (N, 3).")
    z_value = float(current_pose_xyz[2])
    generated: list[NDArray[np.float64]] = [row.copy() for row in base]
    all_modes = [
        mode
        for modes in modes_by_isotope.values()
        for mode in modes
        if mode.weight > 0.0
    ]
    all_modes.sort(key=lambda mode: mode.weight, reverse=True)
    top_modes = all_modes[: max(1, int(config.max_modes_per_isotope) * 2)]
    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        num=max(4, int(config.ring_angles)),
        endpoint=False,
    )
    for mode in top_modes:
        for radius in config.ring_radii_m:
            for angle in angles:
                point = np.array(
                    [
                        mode.position_xyz[0] + float(radius) * np.cos(angle),
                        mode.position_xyz[1] + float(radius) * np.sin(angle),
                        z_value,
                    ],
                    dtype=float,
                )
                generated.append(point)
    cells = getattr(map_api, "traversable_cells", None)
    if cells is None and hasattr(map_api, "blocked_cells"):
        blocked = set(getattr(map_api, "blocked_cells"))
        grid_shape = getattr(map_api, "grid_shape", (0, 0))
        free_neighbors: set[tuple[int, int]] = set()
        for ix, iy in blocked:
            for nb in ((ix - 1, iy), (ix + 1, iy), (ix, iy - 1), (ix, iy + 1)):
                if 0 <= nb[0] < grid_shape[0] and 0 <= nb[1] < grid_shape[1]:
                    if nb not in blocked:
                        free_neighbors.add(nb)
        cells = tuple(sorted(free_neighbors))
    if cells is not None:
        boundary_points = [_cell_center(map_api, tuple(cell), z_value) for cell in cells]
        if top_modes:
            ref = top_modes[0].position_xyz
            boundary_points.sort(key=lambda pt: float(np.linalg.norm(pt - ref)))
        generated.extend(boundary_points[: max(0, int(config.max_augmented_candidates) // 2)])
    coverage_points = _free_cell_centers(
        map_api,
        z_value=z_value,
        max_cells=max(0, int(config.max_augmented_candidates)),
        bounds_xyz=bounds_xyz,
    )
    if coverage_points.size:
        if visited_poses_xyz is not None:
            visited = np.asarray(visited_poses_xyz, dtype=float)
            if visited.ndim == 1 and visited.size == 3:
                visited = visited.reshape(1, 3)
            if visited.ndim == 2 and visited.shape[1] == 3 and visited.size:
                distances = np.linalg.norm(
                    coverage_points[:, None, :2] - visited[None, :, :2],
                    axis=2,
                )
                order = np.argsort(np.min(distances, axis=1))[::-1]
                coverage_points = coverage_points[order]
        generated.extend(
            [
                point.copy()
                for point in coverage_points[: max(0, int(config.max_augmented_candidates) // 2)]
            ]
        )
    if visited_poses_xyz is not None and top_modes:
        visited = np.asarray(visited_poses_xyz, dtype=float)
        if visited.ndim == 2 and visited.shape[1] == 3:
            for mode in top_modes:
                prior_angles = [
                    _bearing_angle_xy(mode.position_xyz, pose) for pose in visited
                ]
                for base_angle in prior_angles:
                    for offset in (0.5 * np.pi, -0.5 * np.pi, np.pi):
                        angle = base_angle + offset
                        for radius in config.ring_radii_m:
                            generated.append(
                                np.array(
                                    [
                                        mode.position_xyz[0]
                                        + float(radius) * np.cos(angle),
                                        mode.position_xyz[1]
                                        + float(radius) * np.sin(angle),
                                        z_value,
                                    ],
                                    dtype=float,
                                )
                            )
    filtered = _bounds_filter(generated, bounds_xyz, map_api)
    deduped = _dedupe_points(filtered)
    limit = max(base.shape[0], int(config.max_augmented_candidates))
    return deduped[:limit]


def _expected_signature(
    *,
    kernel: ContinuousKernel,
    estimator: RotatingShieldPFEstimator,
    mode: SignatureMode,
    pose_xyz: NDArray[np.float64],
    program: ShieldProgram,
    num_orients: int,
    live_time_s: float,
) -> NDArray[np.float64]:
    """Return a source-mode shield-signature count vector."""
    values: list[float] = []
    for pair_id in program.pair_ids:
        fe_index, pb_index = _pair_indices(pair_id, num_orients)
        source_scale = estimator.response_scale_for_isotope(
            mode.isotope,
            fe_index=fe_index,
            pb_index=pb_index,
        )
        kernel_value = kernel.kernel_value_pair(
            isotope=mode.isotope,
            detector_pos=pose_xyz,
            source_pos=mode.position_xyz,
            fe_index=fe_index,
            pb_index=pb_index,
        )
        count = (
            float(live_time_s)
            * float(mode.strength_cps_1m)
            * float(source_scale)
            * float(kernel_value)
        )
        values.append(max(count, 0.0))
    return np.asarray(values, dtype=float)


def _build_pair_signature_cache(
    *,
    kernel: ContinuousKernel,
    estimator: RotatingShieldPFEstimator,
    modes_by_isotope: dict[str, list[SignatureMode]],
    pose_xyz: NDArray[np.float64],
    config: DSSPPConfig,
) -> dict[str, tuple[NDArray[np.float64], list[float]]]:
    """Precompute single-posture signatures for every Fe/Pb orientation pair."""
    num_orients = int(estimator.num_orientations)
    num_pairs = num_orients * num_orients
    cache: dict[str, tuple[NDArray[np.float64], list[float]]] = {}
    for isotope in estimator.isotopes:
        modes = modes_by_isotope.get(isotope, [])
        if not modes:
            cache[isotope] = (np.zeros((num_pairs, 0), dtype=float), [])
            continue
        mode_positions = np.vstack(
            [np.asarray(mode.position_xyz, dtype=float).reshape(3) for mode in modes]
        )
        mode_strengths = np.asarray(
            [float(mode.strength_cps_1m) for mode in modes],
            dtype=float,
        )
        source_scales = _response_scales_for_all_pairs(
            estimator,
            isotope,
            num_orients,
        )
        try:
            kernel_values = kernel.kernel_values_all_pairs(
                isotope=isotope,
                detector_pos=pose_xyz,
                sources=mode_positions,
            )
        except RuntimeError:
            kernel_values = np.asarray(
                [
                    [
                        kernel.kernel_value_pair(
                            isotope=isotope,
                            detector_pos=pose_xyz,
                            source_pos=mode.position_xyz,
                            fe_index=fe_index,
                            pb_index=pb_index,
                        )
                        for mode in modes
                    ]
                    for pair_id in range(num_pairs)
                    for fe_index, pb_index in (
                        _pair_indices(pair_id, num_orients),
                    )
                ],
                dtype=float,
            )
        if kernel_values.shape != (num_pairs, len(modes)):
            matrix = np.zeros((num_pairs, len(modes)), dtype=float)
            for mode_idx, mode in enumerate(modes):
                for pair_id in range(num_pairs):
                    fe_index, pb_index = _pair_indices(pair_id, num_orients)
                    matrix[pair_id, mode_idx] = kernel.kernel_value_pair(
                        isotope=isotope,
                        detector_pos=pose_xyz,
                        source_pos=mode.position_xyz,
                        fe_index=fe_index,
                        pb_index=pb_index,
                    )
            kernel_values = matrix
        matrix = np.maximum(
            float(config.live_time_s)
            * source_scales[:, None]
            * kernel_values
            * mode_strengths[None, :],
            0.0,
        )
        cache[isotope] = (matrix, [float(mode.weight) for mode in modes])
    return cache


def _build_pair_signature_caches_for_poses(
    *,
    kernel: ContinuousKernel,
    estimator: RotatingShieldPFEstimator,
    modes_by_isotope: dict[str, list[SignatureMode]],
    poses_xyz: NDArray[np.float64],
    config: DSSPPConfig,
) -> list[dict[str, tuple[NDArray[np.float64], list[float]]]]:
    """Precompute all-pair shield signatures for every candidate pose."""
    pose_arr = np.asarray(poses_xyz, dtype=float)
    if pose_arr.ndim != 2 or pose_arr.shape[1] != 3:
        raise ValueError("poses_xyz must be shaped (P, 3).")
    num_orients = int(estimator.num_orientations)
    num_pairs = num_orients * num_orients
    caches: list[dict[str, tuple[NDArray[np.float64], list[float]]]] = [
        {} for _ in range(pose_arr.shape[0])
    ]
    for isotope in estimator.isotopes:
        modes = modes_by_isotope.get(isotope, [])
        if not modes:
            for cache in caches:
                cache[isotope] = (np.zeros((num_pairs, 0), dtype=float), [])
            continue
        mode_positions = np.vstack(
            [np.asarray(mode.position_xyz, dtype=float).reshape(3) for mode in modes]
        )
        mode_strengths = np.asarray(
            [float(mode.strength_cps_1m) for mode in modes],
            dtype=float,
        )
        source_scales = _response_scales_for_all_pairs(
            estimator,
            isotope,
            num_orients,
        )
        try:
            kernel_values = kernel.kernel_values_all_pairs_for_detectors(
                isotope=isotope,
                detector_positions=pose_arr,
                sources=mode_positions,
            )
        except RuntimeError:
            kernel_values = np.stack(
                [
                    kernel.kernel_values_all_pairs(
                        isotope=isotope,
                        detector_pos=pose,
                        sources=mode_positions,
                    )
                    for pose in pose_arr
                ],
                axis=0,
            )
        expected_shape = (pose_arr.shape[0], num_pairs, len(modes))
        if kernel_values.shape != expected_shape:
            fallback = np.zeros(expected_shape, dtype=float)
            for pose_idx, pose in enumerate(pose_arr):
                for mode_idx, mode in enumerate(modes):
                    for pair_id in range(num_pairs):
                        fe_index, pb_index = _pair_indices(pair_id, num_orients)
                        fallback[pose_idx, pair_id, mode_idx] = (
                            kernel.kernel_value_pair(
                                isotope=isotope,
                                detector_pos=pose,
                                source_pos=mode.position_xyz,
                                fe_index=fe_index,
                                pb_index=pb_index,
                            )
                        )
            kernel_values = fallback
        matrices = np.maximum(
            float(config.live_time_s)
            * source_scales[None, :, None]
            * kernel_values
            * mode_strengths[None, None, :],
            0.0,
        )
        weights = [float(mode.weight) for mode in modes]
        for pose_idx, cache in enumerate(caches):
            cache[isotope] = (matrices[pose_idx], weights)
    return caches


def _score_program_from_pair_cache(
    *,
    estimator: RotatingShieldPFEstimator,
    pair_cache: dict[str, tuple[NDArray[np.float64], list[float]]],
    modes_by_isotope: dict[str, list[SignatureMode]] | None = None,
    program: ShieldProgram,
    config: DSSPPConfig,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """Score a shield program from cached single-posture signatures."""
    isotope_weights = estimator.pf_config.alpha_weights or {
        isotope: 1.0 for isotope in estimator.isotopes
    }
    alpha_sum = sum(float(v) for v in isotope_weights.values()) or 1.0
    pair_ids = np.asarray(program.pair_ids, dtype=int)
    signature_total = 0.0
    temporal_total = 0.0
    elevation_total = 0.0
    cardinality_gap_total = 0.0
    differential_terms: list[float] = []
    observation_counts: dict[str, float] = {}
    balance_counts: dict[str, float] = {}
    dose_score = 0.0
    room_z_m = _estimator_room_z(estimator)
    for isotope in estimator.isotopes:
        matrix, weights = pair_cache.get(
            isotope,
            (np.zeros((0, 0), dtype=float), []),
        )
        signatures: list[NDArray[np.float64]] = []
        if matrix.size and pair_ids.size:
            clipped_pair_ids = np.clip(pair_ids, 0, matrix.shape[0] - 1)
            signatures = [
                np.asarray(matrix[clipped_pair_ids, mode_idx], dtype=float)
                for mode_idx in range(matrix.shape[1])
            ]
        if signatures:
            isotope_weight = float(isotope_weights.get(isotope, 1.0)) / alpha_sum
            modes_for_isotope = (modes_by_isotope or {}).get(isotope, [])
            pair_priority = _high_surface_pair_priority_weights(
                modes_for_isotope,
                config=config,
                room_z_m=room_z_m,
            )
            signature_total += (
                isotope_weight
                * _signature_separation_score(
                    signatures,
                    variance_floor=config.count_variance_floor,
                )
            )
            temporal_total += isotope_weight * _temporal_separation_score_from_signatures(
                signatures,
                weights,
                config=config,
                pair_priority_weights=pair_priority,
            )
            elevation_total += isotope_weight * _elevation_signature_score_from_signatures(
                signatures,
                weights,
                modes_for_isotope,
                config=config,
                room_z_m=room_z_m,
            )
            program_raw = np.stack(signatures, axis=1).reshape(
                1,
                len(signatures[0]),
                len(signatures),
            )
            cardinality_gap_total += isotope_weight * float(
                _expected_bic_gap_against_source_removal_batch(
                    program_raw,
                    parameter_count_per_source=(
                        config.cardinality_bic_parameter_count_per_source
                    ),
                )[0]
            )
            mean_signature = _weighted_mean_signature(signatures, weights)
            observation_counts[isotope] = (
                float(np.max(mean_signature)) if mean_signature.size else 0.0
            )
            balance_counts[isotope] = (
                float(np.sum(mean_signature)) if mean_signature.size else 0.0
            )
            dose_score += float(np.sum(mean_signature))
            signature_std = float(np.std(mean_signature)) if mean_signature.size else 0.0
        else:
            observation_counts[isotope] = 0.0
            balance_counts[isotope] = 0.0
            signature_std = 0.0
        min_std = max(float(config.signature_std_min_counts), 0.0)
        if min_std > 0.0:
            shortfall = max(0.0, 1.0 - signature_std / min_std)
            differential_terms.append(shortfall * shortfall)
    observation_penalty = minimum_observation_shortfall(
        observation_counts,
        min_counts=float(config.min_observation_counts),
    )
    differential_penalty = (
        float(np.mean(differential_terms)) if differential_terms else 0.0
    )
    count_balance_penalty = _count_balance_penalty(balance_counts)
    count_utility = _saturated_count_utility(
        balance_counts,
        saturation_counts=float(config.count_utility_saturation_counts),
    )
    return (
        signature_total,
        temporal_total,
        elevation_total,
        observation_penalty,
        count_balance_penalty,
        differential_penalty,
        dose_score,
        count_utility,
        cardinality_gap_total,
    )


def _program_pair_id_matrix(
    programs: Sequence[ShieldProgram],
) -> NDArray[np.int64]:
    """Return a padded pair-id matrix for a set of shield programs."""
    if not programs:
        return np.zeros((0, 0), dtype=np.int64)
    max_length = max(len(program.pair_ids) for program in programs)
    if max_length <= 0:
        return np.zeros((len(programs), 0), dtype=np.int64)
    matrix = np.zeros((len(programs), max_length), dtype=np.int64)
    for row_idx, program in enumerate(programs):
        pair_ids = np.asarray(program.pair_ids, dtype=np.int64)
        if pair_ids.size:
            matrix[row_idx, : pair_ids.size] = pair_ids
    return matrix


def _expected_bic_gap_against_source_removal_batch(
    signatures_plm: NDArray[np.float64],
    *,
    parameter_count_per_source: int = 4,
) -> NDArray[np.float64]:
    """
    Return expected BIC gap between a multi-source model and K-1 alternatives.

    ``signatures_plm`` is shaped ``(program, shield_view, source_mode)`` and
    contains expected counts from each source mode. The full model explains the
    expected mean exactly. Each K-1 alternative removes one source column and
    refits nonnegative strengths for the remaining columns in a batched weighted
    least-squares approximation to the local Poisson likelihood.
    """
    raw = np.maximum(np.asarray(signatures_plm, dtype=float), 0.0)
    if raw.ndim != 3:
        raise ValueError("signatures_plm must be shaped (program, view, mode).")
    program_count, view_count, mode_count = raw.shape
    if program_count == 0:
        return np.zeros(0, dtype=float)
    if view_count < 2 or mode_count < 2:
        return np.zeros(program_count, dtype=float)
    expected = np.sum(raw, axis=2)
    sqrt_weight = 1.0 / np.sqrt(np.maximum(expected, 1.0))
    best_deviance = np.full(program_count, np.inf, dtype=float)
    ridge = 1.0e-9
    # The loop is bounded by DSSPPConfig.max_modes_per_isotope, while every
    # candidate program and shield view is evaluated by batched NumPy algebra.
    for removed_idx in range(mode_count):
        keep = np.ones(mode_count, dtype=bool)
        keep[removed_idx] = False
        design = raw[:, :, keep]
        weighted_design = design * sqrt_weight[:, :, None]
        weighted_expected = expected * sqrt_weight
        normal = np.einsum("pvm,pvn->pmn", weighted_design, weighted_design)
        normal += ridge * np.eye(mode_count - 1, dtype=float).reshape(
            1,
            mode_count - 1,
            mode_count - 1,
        )
        rhs = np.einsum("pvm,pv->pm", weighted_design, weighted_expected)
        coeff = np.einsum("pmn,pn->pm", np.linalg.pinv(normal), rhs)
        coeff = np.maximum(np.where(np.isfinite(coeff), coeff, 0.0), 0.0)
        fitted = np.sum(design * coeff[:, None, :], axis=2)
        best_deviance = np.minimum(
            best_deviance,
            _poisson_deviance_matrix(expected, fitted),
        )
    penalty = max(int(parameter_count_per_source), 1) * float(
        np.log(max(view_count, 2))
    )
    gap = best_deviance - penalty
    return np.maximum(np.where(np.isfinite(gap), gap, 0.0), 0.0)


def _poisson_deviance_matrix(
    expected: NDArray[np.float64],
    fitted: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return row-wise Poisson deviance for expected and fitted means."""
    y = np.maximum(np.asarray(expected, dtype=float), 0.0)
    mu = np.maximum(np.asarray(fitted, dtype=float), 1.0e-12)
    if y.shape != mu.shape or y.ndim != 2:
        raise ValueError("expected and fitted must be matching 2-D arrays.")
    log_term = np.zeros_like(y, dtype=float)
    positive = y > 0.0
    log_term[positive] = y[positive] * np.log(y[positive] / mu[positive])
    return 2.0 * np.sum(np.maximum(log_term - (y - mu), 0.0), axis=1)


def _batched_signature_separation_scores(
    raw_counts: NDArray[np.float64],
    *,
    variance_floor: float,
) -> NDArray[np.float64]:
    """Return scalar-equivalent signature scores for many programs."""
    raw = np.maximum(np.asarray(raw_counts, dtype=float), 0.0)
    if raw.ndim != 3 or raw.shape[0] == 0 or raw.shape[2] < 2:
        return np.zeros(raw.shape[0] if raw.ndim >= 1 else 0, dtype=float)
    pair_i, pair_j = np.triu_indices(raw.shape[2], k=1)
    if pair_i.size == 0:
        return np.zeros(raw.shape[0], dtype=float)
    left = raw[:, :, pair_i]
    right = raw[:, :, pair_j]
    variance = np.maximum(left + right, max(float(variance_floor), 1.0e-12))
    distances = np.sum((left - right) * (left - right) / variance, axis=1)
    scores = np.min(distances, axis=1)
    return np.maximum(scores, 0.0)


def _batched_temporal_separation_scores(
    raw_counts: NDArray[np.float64],
    mode_weights: NDArray[np.float64],
    *,
    config: DSSPPConfig,
    pair_priority_weights: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Return scalar-equivalent temporal separation scores for many programs."""
    raw = np.maximum(np.asarray(raw_counts, dtype=float), 0.0)
    if raw.ndim != 3 or raw.shape[0] == 0:
        return np.zeros(raw.shape[0] if raw.ndim >= 1 else 0, dtype=float)
    program_count, program_length, mode_count = raw.shape
    if program_length == 0 or mode_count < 2:
        return np.zeros(program_count, dtype=float)
    totals = np.sum(raw, axis=1)
    active = totals > 1.0e-12
    active_counts = np.sum(active, axis=1)
    if not np.any(active_counts >= 2):
        return np.zeros(program_count, dtype=float)

    weights = np.asarray(mode_weights, dtype=float).ravel()
    if weights.size != mode_count:
        weights = np.ones(mode_count, dtype=float) / float(mode_count)
    weights = np.maximum(weights, 0.0)
    row_weights = weights[None, :] * active
    row_weight_sums = np.sum(row_weights, axis=1, keepdims=True)
    row_weights = np.divide(
        row_weights,
        row_weight_sums,
        out=np.zeros_like(row_weights),
        where=row_weight_sums > 0.0,
    )

    normalized = np.divide(
        raw,
        totals[:, None, :],
        out=np.zeros_like(raw),
        where=totals[:, None, :] > 0.0,
    )
    pair_i, pair_j = np.triu_indices(mode_count, k=1)
    if pair_i.size == 0:
        return np.zeros(program_count, dtype=float)
    pair_priority = (
        np.ones(pair_i.size, dtype=float)
        if pair_priority_weights is None
        else np.asarray(pair_priority_weights, dtype=float).reshape(-1)
    )
    if pair_priority.size != pair_i.size:
        pair_priority = np.ones(pair_i.size, dtype=float)
    pair_priority = np.maximum(pair_priority, 0.0)

    eps = 1.0e-12
    threshold = max(float(config.temporal_pair_contrast_threshold), eps)
    log_matrix = np.log(normalized + eps)
    pair_weights = np.sqrt(
        np.maximum(row_weights[:, pair_i] * row_weights[:, pair_j], 0.0),
    ) * pair_priority.reshape(1, -1)
    contrasts = np.max(
        np.abs(log_matrix[:, :, pair_i] - log_matrix[:, :, pair_j]),
        axis=1,
    )
    useful_contrast = np.minimum(1.0, contrasts / threshold)
    covered = np.sum(pair_weights * useful_contrast, axis=1)
    total_pair_weight = np.sum(pair_weights, axis=1)
    cover = np.divide(
        covered,
        total_pair_weight,
        out=np.zeros(program_count, dtype=float),
        where=total_pair_weight > 0.0,
    )
    cover = np.clip(cover, 0.0, 1.0)

    floor = max(float(config.count_variance_floor), 1.0e-12)
    row_variance = np.maximum(np.sum(normalized, axis=2), floor)
    whitened = normalized / np.sqrt(row_variance[:, :, None])
    weighted = whitened * np.sqrt(np.maximum(row_weights, 0.0))[:, None, :]
    gram = (
        np.einsum("plm,pln->pmn", weighted, weighted)
        + max(float(config.temporal_logdet_ridge), eps)
        * np.eye(mode_count, dtype=float)[None, :, :]
    )
    signs, logdets = np.linalg.slogdet(gram)
    baselines = float(mode_count) * np.log(
        max(float(config.temporal_logdet_ridge), eps),
    )
    logdet_scores = np.where(
        signs > 0,
        np.maximum(logdets - baselines, 0.0),
        0.0,
    )

    left = normalized[:, :, pair_i]
    right = normalized[:, :, pair_j]
    variance = np.maximum(left + right, floor)
    weighted_left = left / np.sqrt(variance)
    weighted_right = right / np.sqrt(variance)
    numerators = np.sum(weighted_left * weighted_right, axis=1)
    denominators = (
        np.linalg.norm(weighted_left, axis=1)
        * np.linalg.norm(weighted_right, axis=1)
    )
    correlations = np.divide(
        numerators,
        denominators,
        out=np.zeros_like(numerators),
        where=denominators > 0.0,
    )
    correlations = np.where(denominators > 0.0, correlations, -np.inf)
    max_corr = np.max(correlations, axis=1)
    max_corr = np.where(np.isfinite(max_corr), max_corr, 0.0)
    max_corr = np.clip(max_corr, 0.0, 1.0)
    decorrelation = 1.0 - max_corr
    scores = (
        float(config.temporal_cover_weight) * cover
        + float(config.temporal_logdet_weight)
        * decorrelation
        * np.log1p(logdet_scores)
        + float(config.temporal_decorrelation_weight) * decorrelation
    )
    scores = np.where(active_counts >= 2, scores, 0.0)
    return np.maximum(scores, 0.0)


def _batched_count_balance_penalties(
    counts_by_program_isotope: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return count-balance penalties for many program rows."""
    values = np.maximum(np.asarray(counts_by_program_isotope, dtype=float), 0.0)
    if values.ndim != 2 or values.shape[1] <= 1:
        return np.zeros(values.shape[0] if values.ndim >= 1 else 0, dtype=float)
    totals = np.sum(values, axis=1, keepdims=True)
    probabilities = np.divide(
        values,
        totals,
        out=np.zeros_like(values),
        where=totals > 0.0,
    )
    positive = probabilities > 0.0
    entropy_terms = np.zeros_like(probabilities)
    entropy_terms[positive] = probabilities[positive] * np.log(
        probabilities[positive],
    )
    entropy = -np.sum(entropy_terms, axis=1)
    normalized_entropy = entropy / max(float(np.log(values.shape[1])), 1.0e-12)
    penalties = np.clip(1.0 - normalized_entropy, 0.0, 1.0)
    penalties = np.where(totals.ravel() > 0.0, penalties, 1.0)
    return penalties


def _score_programs_from_pair_cache(
    *,
    estimator: RotatingShieldPFEstimator,
    pair_cache: dict[str, tuple[NDArray[np.float64], list[float]]],
    modes_by_isotope: dict[str, list[SignatureMode]] | None = None,
    programs: Sequence[ShieldProgram],
    config: DSSPPConfig,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Score many shield programs from cached signatures without changing math."""
    if not programs:
        empty = np.zeros(0, dtype=float)
        return empty, empty, empty, empty, empty, empty, empty, empty, empty
    pair_matrix = _program_pair_id_matrix(programs)
    program_count = len(programs)
    signature_total = np.zeros(program_count, dtype=float)
    temporal_total = np.zeros(program_count, dtype=float)
    elevation_total = np.zeros(program_count, dtype=float)
    cardinality_gap_total = np.zeros(program_count, dtype=float)
    differential_terms: list[NDArray[np.float64]] = []
    observation_counts: list[NDArray[np.float64]] = []
    balance_counts: list[NDArray[np.float64]] = []
    dose_score = np.zeros(program_count, dtype=float)
    isotope_weights = estimator.pf_config.alpha_weights or {
        isotope: 1.0 for isotope in estimator.isotopes
    }
    alpha_sum = sum(float(value) for value in isotope_weights.values()) or 1.0
    room_z_m = _estimator_room_z(estimator)
    for isotope in estimator.isotopes:
        matrix, weights_raw = pair_cache.get(
            isotope,
            (np.zeros((0, 0), dtype=float), []),
        )
        if matrix.size and pair_matrix.size:
            clipped_ids = np.clip(pair_matrix, 0, matrix.shape[0] - 1)
            raw = np.maximum(matrix[clipped_ids, :], 0.0)
        else:
            raw = np.zeros((program_count, pair_matrix.shape[1], 0), dtype=float)
        if raw.shape[2] > 0:
            weights = _normalise_weights(np.asarray(weights_raw, dtype=float))
            if weights.size != raw.shape[2]:
                weights = np.ones(raw.shape[2], dtype=float) / float(raw.shape[2])
            isotope_weight = float(isotope_weights.get(isotope, 1.0)) / alpha_sum
            modes_for_isotope = (modes_by_isotope or {}).get(isotope, [])
            pair_priority = _high_surface_pair_priority_weights(
                modes_for_isotope,
                config=config,
                room_z_m=room_z_m,
            )
            signature_total += isotope_weight * _batched_signature_separation_scores(
                raw,
                variance_floor=float(config.count_variance_floor),
            )
            temporal_total += isotope_weight * _batched_temporal_separation_scores(
                raw,
                weights,
                config=config,
                pair_priority_weights=pair_priority,
            )
            elevation_total += isotope_weight * _batched_elevation_signature_scores(
                raw,
                weights,
                modes_for_isotope,
                config=config,
                room_z_m=room_z_m,
            )
            cardinality_gap_total += (
                isotope_weight
                * _expected_bic_gap_against_source_removal_batch(
                    raw,
                    parameter_count_per_source=(
                        config.cardinality_bic_parameter_count_per_source
                    ),
                )
            )
            mean_signature = np.sum(raw * weights[None, None, :], axis=2)
            observation = (
                np.max(mean_signature, axis=1)
                if mean_signature.shape[1] > 0
                else np.zeros(program_count, dtype=float)
            )
            balance = np.sum(mean_signature, axis=1)
            dose_score += balance
            signature_std = (
                np.std(mean_signature, axis=1)
                if mean_signature.shape[1] > 0
                else np.zeros(program_count, dtype=float)
            )
        else:
            observation = np.zeros(program_count, dtype=float)
            balance = np.zeros(program_count, dtype=float)
            signature_std = np.zeros(program_count, dtype=float)
        observation_counts.append(observation)
        balance_counts.append(balance)
        min_std = max(float(config.signature_std_min_counts), 0.0)
        if min_std > 0.0:
            shortfall = np.maximum(0.0, 1.0 - signature_std / min_std)
            differential_terms.append(shortfall * shortfall)
    observation_matrix = (
        np.stack(observation_counts, axis=1)
        if observation_counts
        else np.zeros((program_count, 0), dtype=float)
    )
    balance_matrix = (
        np.stack(balance_counts, axis=1)
        if balance_counts
        else np.zeros((program_count, 0), dtype=float)
    )
    min_counts = float(config.min_observation_counts)
    if min_counts <= 0.0:
        observation_penalty = np.zeros(program_count, dtype=float)
    elif observation_matrix.shape[1] == 0:
        observation_penalty = np.ones(program_count, dtype=float)
    else:
        shortfalls = np.maximum(0.0, 1.0 - observation_matrix / min_counts)
        observation_penalty = np.mean(shortfalls * shortfalls, axis=1)
    differential_penalty = (
        np.mean(np.stack(differential_terms, axis=1), axis=1)
        if differential_terms
        else np.zeros(program_count, dtype=float)
    )
    count_balance_penalty = _batched_count_balance_penalties(balance_matrix)
    saturation = max(float(config.count_utility_saturation_counts), 1.0e-12)
    if balance_matrix.shape[1] == 0:
        count_utility = np.zeros(program_count, dtype=float)
    else:
        utilities = 1.0 - np.exp(-np.maximum(balance_matrix, 0.0) / saturation)
        count_utility = np.mean(np.clip(utilities, 0.0, 1.0), axis=1)
    return (
        signature_total,
        temporal_total,
        elevation_total,
        observation_penalty,
        count_balance_penalty,
        differential_penalty,
        dose_score,
        count_utility,
        cardinality_gap_total,
    )


def _temporal_scores_programs_from_pair_cache(
    *,
    estimator: RotatingShieldPFEstimator,
    pair_cache: dict[str, tuple[NDArray[np.float64], list[float]]],
    modes_by_isotope: dict[str, list[SignatureMode]] | None = None,
    programs: Sequence[ShieldProgram],
    config: DSSPPConfig,
) -> NDArray[np.float64]:
    """Return temporal separation scores for many programs from cached pairs."""
    if not programs:
        return np.zeros(0, dtype=float)
    pair_matrix = _program_pair_id_matrix(programs)
    if pair_matrix.size == 0:
        return np.zeros(len(programs), dtype=float)
    isotope_weights = estimator.pf_config.alpha_weights or {
        isotope: 1.0 for isotope in estimator.isotopes
    }
    alpha_sum = sum(float(value) for value in isotope_weights.values()) or 1.0
    room_z_m = _estimator_room_z(estimator)
    temporal_total = np.zeros(len(programs), dtype=float)
    for isotope in estimator.isotopes:
        matrix, weights_raw = pair_cache.get(
            isotope,
            (np.zeros((0, 0), dtype=float), []),
        )
        if not matrix.size or matrix.shape[1] < 2:
            continue
        clipped_ids = np.clip(pair_matrix, 0, matrix.shape[0] - 1)
        raw = np.maximum(matrix[clipped_ids, :], 0.0)
        isotope_weight = float(isotope_weights.get(isotope, 1.0)) / alpha_sum
        pair_priority = _high_surface_pair_priority_weights(
            (modes_by_isotope or {}).get(isotope, []),
            config=config,
            room_z_m=room_z_m,
        )
        temporal_total += isotope_weight * _batched_temporal_separation_scores(
            raw,
            np.asarray(weights_raw, dtype=float),
            config=config,
            pair_priority_weights=pair_priority,
        )
    return temporal_total


def _temporal_score_program_from_pair_cache(
    *,
    estimator: RotatingShieldPFEstimator,
    pair_cache: dict[str, tuple[NDArray[np.float64], list[float]]],
    modes_by_isotope: dict[str, list[SignatureMode]] | None = None,
    program: ShieldProgram,
    config: DSSPPConfig,
) -> float:
    """Return only the temporal separation term from cached pair signatures."""
    isotope_weights = estimator.pf_config.alpha_weights or {
        isotope: 1.0 for isotope in estimator.isotopes
    }
    alpha_sum = sum(float(value) for value in isotope_weights.values()) or 1.0
    pair_ids = np.asarray(program.pair_ids, dtype=int)
    if pair_ids.size == 0:
        return 0.0
    room_z_m = _estimator_room_z(estimator)
    temporal_total = 0.0
    for isotope in estimator.isotopes:
        matrix, weights = pair_cache.get(
            isotope,
            (np.zeros((0, 0), dtype=float), []),
        )
        if not matrix.size or matrix.shape[1] < 2:
            continue
        clipped_pair_ids = np.clip(pair_ids, 0, matrix.shape[0] - 1)
        signatures = [
            np.asarray(matrix[clipped_pair_ids, mode_idx], dtype=float)
            for mode_idx in range(matrix.shape[1])
        ]
        isotope_weight = float(isotope_weights.get(isotope, 1.0)) / alpha_sum
        pair_priority = _high_surface_pair_priority_weights(
            (modes_by_isotope or {}).get(isotope, []),
            config=config,
            room_z_m=room_z_m,
        )
        temporal_total += isotope_weight * _temporal_separation_score_from_signatures(
            signatures,
            weights,
            config=config,
            pair_priority_weights=pair_priority,
        )
    return float(temporal_total)


def _weighted_mean_signature(
    signatures: list[NDArray[np.float64]],
    weights: list[float],
) -> NDArray[np.float64]:
    """Return the weighted mean signature vector."""
    if not signatures:
        return np.zeros(0, dtype=float)
    sig_arr = np.vstack(signatures)
    weight_arr = _normalise_weights(np.asarray(weights, dtype=float))
    return np.sum(weight_arr[:, None] * sig_arr, axis=0)


def _signature_separation_score(
    signatures: list[NDArray[np.float64]],
    *,
    variance_floor: float,
) -> float:
    """Return the worst-pair Mahalanobis shield-signature separation."""
    if len(signatures) < 2:
        return 0.0
    floor = max(float(variance_floor), 1e-12)
    best_worst = np.inf
    for idx in range(len(signatures)):
        for jdx in range(idx + 1, len(signatures)):
            left = signatures[idx]
            right = signatures[jdx]
            diff = left - right
            variance = np.maximum(left + right, floor)
            distance = float(np.sum(diff * diff / variance))
            best_worst = min(best_worst, distance)
    if not np.isfinite(best_worst):
        return 0.0
    return max(float(best_worst), 0.0)


def _pairwise_contrast_cover_score(
    response_matrix: NDArray[np.float64],
    mode_weights: NDArray[np.float64],
    *,
    contrast_threshold: float,
) -> float:
    """Return the weighted fraction of mode pairs separated by any posture."""
    matrix = np.maximum(np.asarray(response_matrix, dtype=float), 0.0)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        return 0.0
    weights = _normalise_weights(np.asarray(mode_weights, dtype=float))
    if weights.size != matrix.shape[1]:
        weights = np.ones(matrix.shape[1], dtype=float) / float(matrix.shape[1])
    eps = 1.0e-12
    threshold = max(float(contrast_threshold), eps)
    log_matrix = np.log(matrix + eps)
    pair_i, pair_j = np.triu_indices(matrix.shape[1], k=1)
    if pair_i.size == 0:
        return 0.0
    pair_weights = np.sqrt(
        np.maximum(weights[pair_i] * weights[pair_j], 0.0),
    )
    positive = pair_weights > 0.0
    if not np.any(positive):
        return 0.0
    contrasts = np.max(
        np.abs(log_matrix[:, pair_i[positive]] - log_matrix[:, pair_j[positive]]),
        axis=0,
    )
    useful_contrast = np.minimum(1.0, contrasts / threshold)
    covered = float(np.sum(pair_weights[positive] * useful_contrast))
    total_weight = float(np.sum(pair_weights[positive]))
    if total_weight <= 0.0:
        return 0.0
    return float(np.clip(covered / total_weight, 0.0, 1.0))


def _elevation_pair_indices_and_weights(
    modes: Sequence[SignatureMode],
    mode_weights: NDArray[np.float64],
    *,
    config: DSSPPConfig,
    room_z_m: float = 0.0,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Return mode-pair weights emphasizing vertical ambiguity."""
    mode_count = len(modes)
    if mode_count < 2:
        empty_idx = np.zeros(0, dtype=np.int64)
        return empty_idx, empty_idx, np.zeros(0, dtype=float)
    weights = _normalise_weights(np.asarray(mode_weights, dtype=float))
    if weights.size != mode_count:
        weights = np.ones(mode_count, dtype=float) / float(mode_count)
    positions = np.vstack(
        [np.asarray(mode.position_xyz, dtype=float).reshape(3) for mode in modes]
    )
    left, right = np.triu_indices(mode_count, k=1)
    z_delta = np.abs(positions[left, 2] - positions[right, 2])
    xy_delta = np.linalg.norm(positions[left, :2] - positions[right, :2], axis=1)
    z_scale = max(float(config.elevation_pair_z_scale_m), 1.0e-9)
    xy_scale = max(float(config.elevation_pair_xy_scale_m), 1.0e-9)
    z_factor = z_delta / (z_delta + z_scale)
    xy_factor = xy_scale / (xy_delta + xy_scale)
    posterior_factor = np.sqrt(np.maximum(weights[left] * weights[right], 0.0))
    pair_priority = _high_surface_pair_priority_weights(
        modes,
        config=config,
        room_z_m=room_z_m,
    )
    if pair_priority.size != left.size:
        pair_priority = np.ones(left.size, dtype=float)
    pair_weights = posterior_factor * z_factor * xy_factor * pair_priority
    valid = pair_weights > 0.0
    return (
        left[valid].astype(np.int64, copy=False),
        right[valid].astype(np.int64, copy=False),
        pair_weights[valid].astype(float, copy=False),
    )


def _batched_elevation_signature_scores(
    response_by_program: NDArray[np.float64],
    mode_weights: NDArray[np.float64],
    modes: Sequence[SignatureMode],
    *,
    config: DSSPPConfig,
    room_z_m: float = 0.0,
) -> NDArray[np.float64]:
    """Return height-weighted temporal shield separability for many programs."""
    responses = np.maximum(np.asarray(response_by_program, dtype=float), 0.0)
    if responses.ndim != 3 or responses.shape[2] < 2:
        return np.zeros(responses.shape[0] if responses.ndim >= 1 else 0, dtype=float)
    if len(modes) != responses.shape[2]:
        return np.zeros(responses.shape[0], dtype=float)
    left, right, pair_weights = _elevation_pair_indices_and_weights(
        modes,
        np.asarray(mode_weights, dtype=float),
        config=config,
        room_z_m=room_z_m,
    )
    if left.size == 0:
        return np.zeros(responses.shape[0], dtype=float)
    log_responses = np.log1p(responses)
    contrast = np.linalg.norm(
        log_responses[:, :, left] - log_responses[:, :, right],
        axis=1,
    ) / np.sqrt(max(responses.shape[1], 1))
    threshold = max(float(config.temporal_pair_contrast_threshold), 1.0e-12)
    pair_scores = np.minimum(contrast / threshold, 1.0)
    return np.sum(pair_scores * pair_weights.reshape(1, -1), axis=1) / max(
        float(np.sum(pair_weights)),
        1.0e-12,
    )


def _elevation_signature_score_from_signatures(
    signatures: Sequence[NDArray[np.float64]],
    mode_weights: Sequence[float],
    modes: Sequence[SignatureMode],
    *,
    config: DSSPPConfig,
    room_z_m: float = 0.0,
) -> float:
    """Return height-weighted temporal shield separability for one program."""
    if len(signatures) < 2 or len(signatures) != len(modes):
        return 0.0
    response_matrix = np.stack(
        [np.asarray(signature, dtype=float) for signature in signatures]
    ).T
    response = response_matrix.reshape(
        1,
        response_matrix.shape[0],
        response_matrix.shape[1],
    )
    scores = _batched_elevation_signature_scores(
        response,
        np.asarray(mode_weights, dtype=float),
        modes,
        config=config,
        room_z_m=room_z_m,
    )
    return float(scores[0]) if scores.size else 0.0


def _pairwise_cover_objective(
    signature_scores: NDArray[np.float64] | None,
    temporal_scores: NDArray[np.float64],
    elevation_scores: NDArray[np.float64],
    *,
    config: DSSPPConfig,
) -> NDArray[np.float64]:
    """Return the cover-search objective for pairwise source ambiguity."""
    signature = (
        np.zeros_like(np.asarray(temporal_scores, dtype=float))
        if signature_scores is None
        else np.maximum(np.asarray(signature_scores, dtype=float), 0.0)
    )
    temporal = np.maximum(np.asarray(temporal_scores, dtype=float), 0.0)
    elevation = np.maximum(np.asarray(elevation_scores, dtype=float), 0.0)
    if signature.shape != temporal.shape:
        signature = np.zeros_like(temporal)
    if elevation.shape != temporal.shape:
        elevation = np.zeros_like(temporal)
    signature_weight = max(float(config.lambda_signature), 0.0)
    temporal_weight = max(float(config.lambda_temporal_separation), 0.0)
    elevation_weight = max(float(config.lambda_elevation_signature), 0.0)
    if signature_weight <= 0.0 and temporal_weight <= 0.0 and elevation_weight <= 0.0:
        temporal_weight = 1.0
    return (
        signature_weight * signature
        + temporal_weight * temporal
        + elevation_weight * elevation
    )


def _response_logdet_score(
    response_matrix: NDArray[np.float64],
    mode_weights: NDArray[np.float64],
    *,
    ridge: float,
    variance_floor: float,
) -> float:
    """Return a D-optimality score for temporal shield response columns."""
    matrix = np.maximum(np.asarray(response_matrix, dtype=float), 0.0)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        return 0.0
    weights = _normalise_weights(np.asarray(mode_weights, dtype=float))
    if weights.size != matrix.shape[1]:
        weights = np.ones(matrix.shape[1], dtype=float) / float(matrix.shape[1])
    row_variance = np.maximum(np.sum(matrix, axis=1), max(float(variance_floor), 1.0e-12))
    whitened = matrix / np.sqrt(row_variance[:, None])
    weighted = whitened * np.sqrt(np.maximum(weights, 0.0))[None, :]
    ridge_val = max(float(ridge), 1.0e-12)
    gram = weighted.T @ weighted + ridge_val * np.eye(matrix.shape[1], dtype=float)
    sign, logdet = np.linalg.slogdet(gram)
    if sign <= 0 or not np.isfinite(logdet):
        return 0.0
    baseline = float(matrix.shape[1]) * float(np.log(ridge_val))
    return max(float(logdet - baseline), 0.0)


def _maximum_response_correlation(
    response_matrix: NDArray[np.float64],
    *,
    variance_floor: float,
) -> float:
    """Return the largest Poisson-whitened cosine between mode responses."""
    matrix = np.maximum(np.asarray(response_matrix, dtype=float), 0.0)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        return 0.0
    floor = max(float(variance_floor), 1.0e-12)
    pair_i, pair_j = np.triu_indices(matrix.shape[1], k=1)
    if pair_i.size == 0:
        return 0.0
    left = matrix[:, pair_i]
    right = matrix[:, pair_j]
    variance = np.maximum(left + right, floor)
    weighted_left = left / np.sqrt(variance)
    weighted_right = right / np.sqrt(variance)
    numerators = np.sum(weighted_left * weighted_right, axis=0)
    denominators = (
        np.linalg.norm(weighted_left, axis=0)
        * np.linalg.norm(weighted_right, axis=0)
    )
    valid = denominators > 0.0
    if not np.any(valid):
        return 0.0
    max_corr = float(np.max(numerators[valid] / denominators[valid]))
    return float(np.clip(max_corr, 0.0, 1.0))


def _pairwise_distance_correlation_payload(
    response_matrix: NDArray[np.float64],
    variance_by_row: NDArray[np.float64] | None,
    *,
    variance_floor: float,
    threshold: float,
) -> dict[str, object]:
    """Return vectorized source-pair separation diagnostics for one response matrix."""
    matrix = np.maximum(np.asarray(response_matrix, dtype=float), 0.0)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        return {
            "left_indices": [],
            "right_indices": [],
            "distances": [],
            "correlations": [],
            "min_separation": 0.0,
            "max_correlation": 0.0,
            "unresolved_pairs": 0,
        }
    left_idx, right_idx = np.triu_indices(matrix.shape[1], k=1)
    left = matrix[:, left_idx]
    right = matrix[:, right_idx]
    if variance_by_row is None:
        variance = np.maximum(left + right, float(variance_floor))
    else:
        row_variance = np.maximum(
            np.asarray(variance_by_row, dtype=float).reshape(-1),
            float(variance_floor),
        )
        if row_variance.size != matrix.shape[0]:
            row_variance = np.full(matrix.shape[0], float(variance_floor), dtype=float)
        variance = row_variance[:, None]
    variance = np.maximum(variance, 1.0e-12)
    diff = left - right
    distances = np.sum((diff * diff) / variance, axis=0)
    weighted_left = left / np.sqrt(variance)
    weighted_right = right / np.sqrt(variance)
    numerators = np.sum(weighted_left * weighted_right, axis=0)
    denominators = (
        np.linalg.norm(weighted_left, axis=0) * np.linalg.norm(weighted_right, axis=0)
    )
    correlations = np.divide(
        numerators,
        denominators,
        out=np.zeros_like(numerators, dtype=float),
        where=denominators > 0.0,
    )
    correlations = np.clip(correlations, 0.0, 1.0)
    return {
        "left_indices": [int(value) for value in left_idx],
        "right_indices": [int(value) for value in right_idx],
        "distances": [float(value) for value in distances],
        "correlations": [float(value) for value in correlations],
        "min_separation": float(np.min(distances)) if distances.size else 0.0,
        "max_correlation": float(np.max(correlations)) if correlations.size else 0.0,
        "unresolved_pairs": int(np.count_nonzero(distances < float(threshold))),
    }


def _pairwise_stat_value(
    payload: dict[str, object],
    key: str,
    pair_index: int,
    default: float = 0.0,
) -> float:
    """Return one pair statistic from a pairwise diagnostic payload."""
    values = payload.get(key, [])
    if not isinstance(values, Sequence) or pair_index >= len(values):
        return float(default)
    return float(values[pair_index])


def _mode_pair_geometry_payload(
    left_mode: SignatureMode,
    right_mode: SignatureMode,
    pose_xyz: NDArray[np.float64],
) -> dict[str, float]:
    """Return bearing and elevation separation for a source-mode pair."""
    pose = np.asarray(pose_xyz, dtype=float).reshape(3)
    left_delta = pose - np.asarray(left_mode.position_xyz, dtype=float).reshape(3)
    right_delta = pose - np.asarray(right_mode.position_xyz, dtype=float).reshape(3)
    left_bearing = float(np.arctan2(left_delta[1], left_delta[0]))
    right_bearing = float(np.arctan2(right_delta[1], right_delta[0]))
    bearing_delta = _angle_distance_rad(left_bearing, right_bearing)
    left_horizontal = max(float(np.linalg.norm(left_delta[:2])), 1.0e-12)
    right_horizontal = max(float(np.linalg.norm(right_delta[:2])), 1.0e-12)
    left_elevation = float(np.arctan2(left_delta[2], left_horizontal))
    right_elevation = float(np.arctan2(right_delta[2], right_horizontal))
    elevation_delta = abs(left_elevation - right_elevation)
    return {
        "bearing_delta_deg": float(np.rad2deg(bearing_delta)),
        "elevation_delta_deg": float(np.rad2deg(elevation_delta)),
    }


def _program_pairwise_ambiguity_diagnostics(
    *,
    estimator: RotatingShieldPFEstimator,
    kernel: ContinuousKernel,
    modes_by_isotope: dict[str, list[SignatureMode]],
    pose_xyz: NDArray[np.float64],
    program: ShieldProgram,
    config: DSSPPConfig,
    max_pairs: int = 3,
) -> dict[str, dict[str, object]]:
    """Return selected-program diagnostics for the hardest same-isotope mode pairs."""
    if not program.pair_ids:
        return {}
    pose = np.asarray(pose_xyz, dtype=float).reshape(3)
    pair_cache = _build_pair_signature_cache(
        kernel=kernel,
        estimator=estimator,
        modes_by_isotope=modes_by_isotope,
        pose_xyz=pose,
        config=config,
    )
    threshold = max(float(config.count_variance_floor), 1.0e-12)
    separation_threshold = max(float(config.temporal_pair_contrast_threshold), 1.0)
    payload: dict[str, dict[str, object]] = {}
    for isotope, modes in modes_by_isotope.items():
        active_modes = [mode for mode in modes if float(mode.weight) > 0.0]
        if len(active_modes) < 2:
            continue
        matrix, _weights = pair_cache.get(
            isotope,
            (np.zeros((0, 0), dtype=float), []),
        )
        if matrix.ndim != 2 or matrix.shape[1] != len(active_modes):
            continue
        pair_ids = np.clip(
            np.asarray(program.pair_ids, dtype=int),
            0,
            max(matrix.shape[0] - 1, 0),
        )
        program_response = np.maximum(matrix[pair_ids, :], 0.0)
        program_variance = np.maximum(
            np.sum(program_response, axis=1),
            threshold,
        )
        before_response = np.zeros((0, len(active_modes)), dtype=float)
        before_variance = np.zeros(0, dtype=float)
        data = estimator._measurement_data_for_iso(isotope, window=None)
        if data is not None and data.z_k.size:
            before_response = expected_counts_per_source(
                kernel=kernel,
                isotope=isotope,
                detector_positions=data.detector_positions,
                sources=np.vstack([mode.position_xyz for mode in active_modes]),
                strengths=np.asarray(
                    [max(float(mode.strength_cps_1m), 0.0) for mode in active_modes],
                    dtype=float,
                ),
                live_times=data.live_times,
                fe_indices=data.fe_indices,
                pb_indices=data.pb_indices,
                source_scale=estimator.response_scales_for_measurements(
                    isotope,
                    data.fe_indices,
                    data.pb_indices,
                ),
            )
            before_response = np.maximum(before_response, 0.0)
            before_variance = np.maximum(data.observation_variances, threshold)
        combined_response = np.vstack([before_response, program_response])
        combined_variance = np.concatenate([before_variance, program_variance])
        before_stats = _pairwise_distance_correlation_payload(
            before_response,
            before_variance,
            variance_floor=threshold,
            threshold=separation_threshold,
        )
        program_stats = _pairwise_distance_correlation_payload(
            program_response,
            program_variance,
            variance_floor=threshold,
            threshold=separation_threshold,
        )
        combined_stats = _pairwise_distance_correlation_payload(
            combined_response,
            combined_variance,
            variance_floor=threshold,
            threshold=separation_threshold,
        )
        distances = np.asarray(combined_stats.get("distances", []), dtype=float)
        correlations = np.asarray(combined_stats.get("correlations", []), dtype=float)
        left_indices = list(combined_stats.get("left_indices", []))
        right_indices = list(combined_stats.get("right_indices", []))
        if distances.size == 0:
            continue
        order = np.lexsort((-correlations, distances))
        pair_details: list[dict[str, object]] = []
        for rank, pair_pos in enumerate(order[: max(0, int(max_pairs))], start=1):
            left_idx = int(left_indices[int(pair_pos)])
            right_idx = int(right_indices[int(pair_pos)])
            left_mode = active_modes[left_idx]
            right_mode = active_modes[right_idx]
            detail = {
                "rank": int(rank),
                "left_mode": _mode_diagnostic_payload(left_mode, left_idx),
                "right_mode": _mode_diagnostic_payload(right_mode, right_idx),
                "before_separation": _pairwise_stat_value(
                    before_stats,
                    "distances",
                    int(pair_pos),
                ),
                "program_separation": _pairwise_stat_value(
                    program_stats,
                    "distances",
                    int(pair_pos),
                ),
                "combined_separation": _pairwise_stat_value(
                    combined_stats,
                    "distances",
                    int(pair_pos),
                ),
                "combined_correlation": _pairwise_stat_value(
                    combined_stats,
                    "correlations",
                    int(pair_pos),
                ),
                "program_left_response": [
                    float(value) for value in program_response[:, left_idx]
                ],
                "program_right_response": [
                    float(value) for value in program_response[:, right_idx]
                ],
            }
            detail.update(_mode_pair_geometry_payload(left_mode, right_mode, pose))
            pair_details.append(detail)
        payload[str(isotope)] = {
            "mode_count": int(len(active_modes)),
            "program_pair_ids": [int(value) for value in program.pair_ids],
            "before_measurements": int(before_response.shape[0]),
            "program_measurements": int(program_response.shape[0]),
            "before_min_separation": float(before_stats["min_separation"]),
            "before_max_correlation": float(before_stats["max_correlation"]),
            "before_unresolved_pairs": int(before_stats["unresolved_pairs"]),
            "program_min_separation": float(program_stats["min_separation"]),
            "program_max_correlation": float(program_stats["max_correlation"]),
            "program_unresolved_pairs": int(program_stats["unresolved_pairs"]),
            "combined_min_separation": float(combined_stats["min_separation"]),
            "combined_max_correlation": float(combined_stats["max_correlation"]),
            "combined_unresolved_pairs": int(combined_stats["unresolved_pairs"]),
            "bottleneck_pairs": pair_details,
        }
    return payload


def _temporal_separation_score_from_signatures(
    signatures: list[NDArray[np.float64]],
    weights: list[float],
    *,
    config: DSSPPConfig,
    pair_priority_weights: NDArray[np.float64] | None = None,
) -> float:
    """Score a shield program by same-isotope temporal-code separability."""
    if len(signatures) < 2:
        return 0.0
    raw_matrix = np.column_stack(
        [np.maximum(np.asarray(sig, dtype=float).ravel(), 0.0) for sig in signatures]
    )
    if raw_matrix.ndim != 2 or raw_matrix.shape[0] == 0 or raw_matrix.shape[1] < 2:
        return 0.0
    scores = _batched_temporal_separation_scores(
        raw_matrix.reshape(1, raw_matrix.shape[0], raw_matrix.shape[1]),
        np.asarray(weights, dtype=float),
        config=config,
        pair_priority_weights=pair_priority_weights,
    )
    return float(scores[0]) if scores.size else 0.0


def _count_balance_penalty(counts: dict[str, float]) -> float:
    """Return an isotope-agnostic penalty for single-isotope dominated programs."""
    values = np.asarray(
        [max(float(value), 0.0) for value in counts.values()],
        dtype=float,
    )
    if values.size <= 1:
        return 0.0
    total = float(np.sum(values))
    if total <= 0.0:
        return 1.0
    probabilities = values / total
    positive = probabilities > 0.0
    entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
    normalized_entropy = entropy / max(float(np.log(values.size)), 1e-12)
    return float(np.clip(1.0 - normalized_entropy, 0.0, 1.0))


def _saturated_count_utility(
    counts: dict[str, float],
    *,
    saturation_counts: float,
) -> float:
    """Return a bounded utility for usable counts without rewarding raw proximity."""
    values = np.asarray(
        [max(float(value), 0.0) for value in counts.values()],
        dtype=float,
    )
    if values.size == 0:
        return 0.0
    saturation = max(float(saturation_counts), 1.0e-12)
    utilities = 1.0 - np.exp(-values / saturation)
    return float(np.mean(np.clip(utilities, 0.0, 1.0)))


def _mode_visibility_proxy(
    mode: SignatureMode,
    pose_xyz: NDArray[np.float64],
    *,
    live_time_s: float,
    saturation_counts: float,
) -> float:
    """Return a bounded inverse-square visibility proxy for planning heuristics."""
    pose = np.asarray(pose_xyz, dtype=float).reshape(3)
    distance = max(float(np.linalg.norm(pose - mode.position_xyz)), 0.25)
    expected = float(live_time_s) * max(float(mode.strength_cps_1m), 0.0) / (
        distance * distance
    )
    saturation = max(float(saturation_counts), 1.0e-12)
    return float(np.clip(1.0 - np.exp(-expected / saturation), 0.0, 1.0))


def _local_orbit_gain(
    candidate_pose_xyz: NDArray[np.float64],
    modes_by_isotope: dict[str, list[SignatureMode]],
    *,
    config: DSSPPConfig,
) -> float:
    """Return a gain for near-source annular viewpoints rather than source chasing."""
    radii = tuple(
        float(radius)
        for radius in config.ring_radii_m
        if float(radius) > 0.0
    )
    if not radii:
        return 0.0
    candidate = np.asarray(candidate_pose_xyz, dtype=float).reshape(3)
    sigma = max(float(config.local_orbit_sigma_m), 1.0e-6)
    gains: list[float] = []
    weights: list[float] = []
    for modes in modes_by_isotope.values():
        for mode in modes:
            if float(mode.weight) <= 0.0:
                continue
            distance = float(np.linalg.norm(candidate[:2] - mode.position_xyz[:2]))
            radial_error = min(abs(distance - radius) for radius in radii)
            radial_gain = float(np.exp(-0.5 * (radial_error / sigma) ** 2))
            visibility = _mode_visibility_proxy(
                mode,
                candidate,
                live_time_s=float(config.live_time_s),
                saturation_counts=float(config.count_utility_saturation_counts),
            )
            gains.append(radial_gain * visibility)
            weights.append(float(mode.weight))
    if not gains:
        return 0.0
    weight_arr = _normalise_weights(np.asarray(weights, dtype=float))
    return float(np.sum(weight_arr * np.asarray(gains, dtype=float)))


def _local_orbit_gains_batch(
    candidate_poses_xyz: NDArray[np.float64],
    modes_by_isotope: dict[str, list[SignatureMode]],
    *,
    config: DSSPPConfig,
) -> NDArray[np.float64]:
    """Return local-orbit gains for many candidate stations."""
    candidates = np.asarray(candidate_poses_xyz, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidate_poses_xyz must be shaped (N, 3).")
    radii = np.asarray(
        [float(radius) for radius in config.ring_radii_m if float(radius) > 0.0],
        dtype=float,
    )
    if radii.size == 0 or candidates.shape[0] == 0:
        return np.zeros(candidates.shape[0], dtype=float)
    modes = [
        mode
        for mode_list in modes_by_isotope.values()
        for mode in mode_list
        if float(mode.weight) > 0.0
    ]
    if not modes:
        return np.zeros(candidates.shape[0], dtype=float)
    mode_positions = np.vstack(
        [np.asarray(mode.position_xyz, dtype=float) for mode in modes]
    )
    mode_strengths = np.asarray(
        [max(float(mode.strength_cps_1m), 0.0) for mode in modes],
        dtype=float,
    )
    mode_weights = _normalise_weights(
        np.asarray([float(mode.weight) for mode in modes], dtype=float)
    )
    xy_distances = np.linalg.norm(
        candidates[:, None, :2] - mode_positions[None, :, :2],
        axis=2,
    )
    radial_error = np.min(np.abs(xy_distances[:, :, None] - radii[None, None, :]), axis=2)
    sigma = max(float(config.local_orbit_sigma_m), 1.0e-6)
    radial_gain = np.exp(-0.5 * (radial_error / sigma) ** 2)
    distances_3d = np.maximum(
        np.linalg.norm(candidates[:, None, :] - mode_positions[None, :, :], axis=2),
        0.25,
    )
    expected = (
        float(config.live_time_s)
        * mode_strengths.reshape(1, -1)
        / (distances_3d * distances_3d)
    )
    saturation = max(float(config.count_utility_saturation_counts), 1.0e-12)
    visibility = np.clip(1.0 - np.exp(-expected / saturation), 0.0, 1.0)
    return np.sum(radial_gain * visibility * mode_weights.reshape(1, -1), axis=1)


def _station_response_matrix(
    poses_xyz: NDArray[np.float64],
    modes: Sequence[SignatureMode],
    *,
    live_time_s: float,
    kernel: ContinuousKernel | None = None,
    estimator: RotatingShieldPFEstimator | None = None,
    isotope: str | None = None,
) -> NDArray[np.float64]:
    """Return a station-response design matrix for planning."""
    pose_arr = np.asarray(poses_xyz, dtype=float)
    if pose_arr.ndim == 1 and pose_arr.size == 3:
        pose_arr = pose_arr.reshape(1, 3)
    if pose_arr.ndim != 2 or pose_arr.shape[1] != 3 or not modes:
        return np.zeros((0, 0), dtype=float)
    if kernel is not None and estimator is not None and isotope is not None:
        return _station_response_matrix_from_kernel(
            pose_arr,
            modes,
            live_time_s=live_time_s,
            kernel=kernel,
            estimator=estimator,
            isotope=isotope,
        )
    matrix = np.zeros((pose_arr.shape[0], len(modes)), dtype=float)
    for mode_idx, mode in enumerate(modes):
        deltas = pose_arr - mode.position_xyz[None, :]
        distances = np.maximum(np.linalg.norm(deltas, axis=1), 0.25)
        matrix[:, mode_idx] = (
            float(live_time_s)
            * max(float(mode.strength_cps_1m), 0.0)
            / (distances * distances)
        )
    return matrix


def _station_response_matrix_from_kernel(
    pose_arr: NDArray[np.float64],
    modes: Sequence[SignatureMode],
    *,
    live_time_s: float,
    kernel: ContinuousKernel,
    estimator: RotatingShieldPFEstimator,
    isotope: str,
) -> NDArray[np.float64]:
    """Return pair-averaged station responses from the shared PF kernel."""
    if pose_arr.shape[0] == 0:
        return np.zeros((0, len(modes)), dtype=float)
    mode_positions = np.vstack(
        [np.asarray(mode.position_xyz, dtype=float).reshape(3) for mode in modes]
    )
    mode_strengths = np.asarray(
        [max(float(mode.strength_cps_1m), 0.0) for mode in modes],
        dtype=float,
    )
    num_orients = int(len(getattr(kernel, "orientations", estimator.normals)))
    num_pairs = max(num_orients, 1) * max(num_orients, 1)
    source_scales = _response_scales_for_all_pairs(estimator, isotope, num_orients)
    if source_scales.size != num_pairs:
        source_scales = np.ones(num_pairs, dtype=float)
    try:
        kernel_values = kernel.kernel_values_all_pairs_for_detectors(
            isotope=isotope,
            detector_positions=pose_arr,
            sources=mode_positions,
        )
    except RuntimeError:
        kernel_values = np.stack(
            [
                kernel.kernel_values_all_pairs(
                    isotope=isotope,
                    detector_pos=pose,
                    sources=mode_positions,
                )
                for pose in pose_arr
            ],
            axis=0,
        )
    expected_shape = (pose_arr.shape[0], num_pairs, len(modes))
    if kernel_values.shape != expected_shape:
        raise ValueError(
            "shared station response kernel returned shape "
            f"{kernel_values.shape}, expected {expected_shape}."
        )
    pair_counts = (
        np.maximum(kernel_values, 0.0)
        * np.maximum(source_scales, 0.0).reshape(1, num_pairs, 1)
    )
    return np.maximum(
        float(live_time_s)
        * np.mean(pair_counts, axis=1)
        * mode_strengths.reshape(1, -1),
        0.0,
    )


def _station_condition_logdet(
    matrix: NDArray[np.float64],
    *,
    ridge: float,
) -> float:
    """Return a column-normalized D-optimal station-design score."""
    design = np.maximum(np.asarray(matrix, dtype=float), 0.0)
    if design.ndim != 2 or design.shape[0] == 0 or design.shape[1] < 2:
        return 0.0
    column_norms = np.linalg.norm(design, axis=0)
    active = column_norms > 1.0e-12
    if int(np.count_nonzero(active)) < 2:
        return 0.0
    normalized = design[:, active] / column_norms[active][None, :]
    ridge_val = max(float(ridge), 1.0e-12)
    gram = normalized.T @ normalized + ridge_val * np.eye(
        int(np.count_nonzero(active)),
        dtype=float,
    )
    sign, logdet = np.linalg.slogdet(gram)
    if sign <= 0 or not np.isfinite(logdet):
        return 0.0
    baseline = float(np.count_nonzero(active)) * float(np.log(ridge_val))
    return max(float(logdet - baseline), 0.0)


def _station_design_quality(
    matrix: NDArray[np.float64],
    *,
    config: DSSPPConfig,
) -> float:
    """Return the configured conditioning quality of a response design."""
    design = np.maximum(np.asarray(matrix, dtype=float), 0.0)
    if design.ndim != 2 or design.shape[0] == 0 or design.shape[1] < 2:
        return 0.0
    column_norms = np.linalg.norm(design, axis=0)
    active = column_norms > 1.0e-12
    if int(np.count_nonzero(active)) < 2:
        return 0.0
    normalized = design[:, active] / column_norms[active][None, :]
    gram = normalized.T @ normalized
    eigvals = np.linalg.eigvalsh(0.5 * (gram + gram.T))
    eigvals = np.maximum(np.asarray(eigvals, dtype=float), 0.0)
    if eigvals.size == 0:
        return 0.0
    max_eig = max(float(np.max(eigvals)), 1.0e-24)
    min_eig = max(float(np.min(eigvals)), 0.0)
    min_singular = float(np.sqrt(min_eig))
    inverse_condition = 0.0
    if min_eig > 1.0e-24:
        inverse_condition = float(np.sqrt(min_eig / max_eig))
    corr = _max_column_correlation_from_normalized_gram(gram)
    coherence_quality = 1.0 - float(np.clip(corr, 0.0, 1.0))
    return float(
        _station_condition_logdet(
            design,
            ridge=float(config.station_condition_ridge),
        )
        + float(config.station_condition_min_singular_weight) * min_singular
        + float(config.station_condition_inverse_condition_weight)
        * inverse_condition
        + float(config.station_condition_coherence_weight) * coherence_quality
    )


def _max_column_correlation_from_normalized_gram(
    gram: NDArray[np.float64],
) -> float:
    """Return max off-diagonal coherence from a normalized Gram matrix."""
    arr = np.asarray(gram, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[0] != arr.shape[1]:
        return 1.0
    corr = np.clip(np.abs(arr.copy()), 0.0, 1.0)
    np.fill_diagonal(corr, 0.0)
    return float(np.max(corr))


def _station_condition_logdet_batch(
    *,
    before_matrix: NDArray[np.float64],
    candidate_rows: NDArray[np.float64],
    ridge: float,
) -> NDArray[np.float64]:
    """Return station-condition logdet scores after adding each candidate row."""
    rows = np.maximum(np.asarray(candidate_rows, dtype=float), 0.0)
    if rows.ndim != 2:
        raise ValueError("candidate_rows must be shaped (N, M).")
    if rows.shape[1] < 2 or rows.shape[0] == 0:
        return np.zeros(rows.shape[0], dtype=float)
    before = np.maximum(np.asarray(before_matrix, dtype=float), 0.0)
    if before.ndim != 2 or before.shape[1] != rows.shape[1]:
        before = np.zeros((0, rows.shape[1]), dtype=float)
    before_cross = before.T @ before if before.size else np.zeros((rows.shape[1], rows.shape[1]), dtype=float)
    before_norm_sq = np.sum(before * before, axis=0) if before.size else np.zeros(rows.shape[1], dtype=float)
    norm_sq = before_norm_sq.reshape(1, -1) + rows * rows
    active = norm_sq > 1.0e-24
    active_counts = np.count_nonzero(active, axis=1)
    denom = np.sqrt(norm_sq[:, :, None] * norm_sq[:, None, :])
    cross = before_cross.reshape(1, rows.shape[1], rows.shape[1]) + np.einsum(
        "ni,nj->nij",
        rows,
        rows,
    )
    normalized = np.divide(
        cross,
        denom,
        out=np.zeros_like(cross),
        where=denom > 0.0,
    )
    ridge_val = max(float(ridge), 1.0e-12)
    gram = normalized + ridge_val * np.eye(rows.shape[1], dtype=float).reshape(
        1,
        rows.shape[1],
        rows.shape[1],
    )
    signs, logdets = np.linalg.slogdet(gram)
    baselines = active_counts.astype(float) * float(np.log(ridge_val))
    scores = np.where(
        (active_counts >= 2) & (signs > 0.0) & np.isfinite(logdets),
        logdets - baselines,
        0.0,
    )
    return np.maximum(scores, 0.0)


def _station_design_quality_batch(
    *,
    before_matrix: NDArray[np.float64],
    candidate_rows: NDArray[np.float64],
    config: DSSPPConfig,
) -> NDArray[np.float64]:
    """Return configured conditioning quality after adding candidate rows."""
    rows = np.maximum(np.asarray(candidate_rows, dtype=float), 0.0)
    if rows.ndim != 2:
        raise ValueError("candidate_rows must be shaped (N, M).")
    if rows.shape[0] == 0:
        return np.zeros(0, dtype=float)
    if rows.shape[1] < 2:
        return np.zeros(rows.shape[0], dtype=float)
    logdet = _station_condition_logdet_batch(
        before_matrix=before_matrix,
        candidate_rows=rows,
        ridge=float(config.station_condition_ridge),
    )
    if (
        float(config.station_condition_min_singular_weight) <= 0.0
        and float(config.station_condition_inverse_condition_weight) <= 0.0
        and float(config.station_condition_coherence_weight) <= 0.0
    ):
        return logdet
    before = np.maximum(np.asarray(before_matrix, dtype=float), 0.0)
    if before.ndim != 2 or before.shape[1] != rows.shape[1]:
        before = np.zeros((0, rows.shape[1]), dtype=float)
    before_cross = (
        before.T @ before
        if before.size
        else np.zeros((rows.shape[1], rows.shape[1]), dtype=float)
    )
    before_norm_sq = (
        np.sum(before * before, axis=0)
        if before.size
        else np.zeros(rows.shape[1], dtype=float)
    )
    cross = before_cross.reshape(1, rows.shape[1], rows.shape[1]) + np.einsum(
        "ni,nj->nij",
        rows,
        rows,
    )
    norm_sq = before_norm_sq.reshape(1, -1) + rows * rows
    denom = np.sqrt(norm_sq[:, :, None] * norm_sq[:, None, :])
    normalized = np.divide(
        cross,
        denom,
        out=np.zeros_like(cross, dtype=float),
        where=denom > 0.0,
    )
    eigvals = np.linalg.eigvalsh(0.5 * (normalized + np.swapaxes(normalized, 1, 2)))
    eigvals = np.maximum(np.asarray(eigvals, dtype=float), 0.0)
    max_eig = np.maximum(np.max(eigvals, axis=1), 1.0e-24)
    min_eig = np.maximum(np.min(eigvals, axis=1), 0.0)
    min_singular = np.sqrt(min_eig)
    inverse_condition = np.divide(
        np.sqrt(min_eig),
        np.sqrt(max_eig),
        out=np.zeros_like(min_eig, dtype=float),
        where=min_eig > 1.0e-24,
    )
    diag = np.arange(rows.shape[1])
    coherence_matrix = np.clip(np.abs(normalized), 0.0, 1.0)
    coherence_matrix[:, diag, diag] = 0.0
    coherence_quality = 1.0 - np.max(coherence_matrix, axis=(1, 2))
    active_counts = np.count_nonzero(norm_sq > 1.0e-24, axis=1)
    quality = (
        logdet
        + float(config.station_condition_min_singular_weight) * min_singular
        + float(config.station_condition_inverse_condition_weight)
        * inverse_condition
        + float(config.station_condition_coherence_weight) * coherence_quality
    )
    quality[active_counts < 2] = 0.0
    return np.maximum(np.where(np.isfinite(quality), quality, 0.0), 0.0)


def _station_condition_gain(
    candidate_pose_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
    modes_by_isotope: dict[str, list[SignatureMode]],
    *,
    config: DSSPPConfig,
    kernel: ContinuousKernel | None = None,
    estimator: RotatingShieldPFEstimator | None = None,
) -> float:
    """Return the added response-matrix conditioning from one candidate station."""
    candidate = np.asarray(candidate_pose_xyz, dtype=float).reshape(1, 3)
    visited = np.zeros((0, 3), dtype=float)
    if visited_poses_xyz is not None:
        visited = np.asarray(visited_poses_xyz, dtype=float)
        if visited.ndim == 1 and visited.size == 3:
            visited = visited.reshape(1, 3)
        if visited.ndim != 2 or visited.shape[1] != 3:
            visited = np.zeros((0, 3), dtype=float)
    gains: list[float] = []
    weights: list[float] = []
    for isotope, modes in modes_by_isotope.items():
        active = [mode for mode in modes if float(mode.weight) > 0.0]
        if len(active) < 2:
            continue
        before = _station_response_matrix(
            visited,
            active,
            live_time_s=float(config.live_time_s),
            kernel=kernel,
            estimator=estimator,
            isotope=isotope,
        )
        after = _station_response_matrix(
            np.vstack([visited, candidate]) if visited.size else candidate,
            active,
            live_time_s=float(config.live_time_s),
            kernel=kernel,
            estimator=estimator,
            isotope=isotope,
        )
        before_score = _station_design_quality(before, config=config)
        after_score = _station_design_quality(after, config=config)
        gains.append(max(after_score - before_score, 0.0))
        weights.append(float(sum(mode.weight for mode in active)))
    if not gains:
        return 0.0
    weight_arr = _normalise_weights(np.asarray(weights, dtype=float))
    return float(np.sum(weight_arr * np.asarray(gains, dtype=float)))


def _station_condition_gains_batch(
    candidate_poses_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
    modes_by_isotope: dict[str, list[SignatureMode]],
    *,
    config: DSSPPConfig,
    kernel: ContinuousKernel | None = None,
    estimator: RotatingShieldPFEstimator | None = None,
) -> NDArray[np.float64]:
    """Return response-matrix conditioning gains for many candidate stations."""
    candidates = np.asarray(candidate_poses_xyz, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidate_poses_xyz must be shaped (N, 3).")
    if candidates.shape[0] == 0:
        return np.zeros(0, dtype=float)
    visited = _pose_matrix_or_empty(visited_poses_xyz)
    gain_rows: list[NDArray[np.float64]] = []
    weight_values: list[float] = []
    for isotope, modes in modes_by_isotope.items():
        active = [mode for mode in modes if float(mode.weight) > 0.0]
        if len(active) < 2:
            continue
        before = _station_response_matrix(
            visited,
            active,
            live_time_s=float(config.live_time_s),
            kernel=kernel,
            estimator=estimator,
            isotope=isotope,
        )
        candidate_rows = _station_response_matrix(
            candidates,
            active,
            live_time_s=float(config.live_time_s),
            kernel=kernel,
            estimator=estimator,
            isotope=isotope,
        )
        before_score = _station_design_quality(before, config=config)
        after_scores = _station_design_quality_batch(
            before_matrix=before,
            candidate_rows=candidate_rows,
            config=config,
        )
        gain_rows.append(np.maximum(after_scores - before_score, 0.0))
        weight_values.append(float(sum(mode.weight for mode in active)))
    if not gain_rows:
        return np.zeros(candidates.shape[0], dtype=float)
    weights = _normalise_weights(np.asarray(weight_values, dtype=float))
    stacked = np.vstack(gain_rows)
    return np.sum(stacked * weights.reshape(-1, 1), axis=0)


def _max_column_correlation_from_design(
    matrix: NDArray[np.float64],
) -> float:
    """Return the maximum normalized column correlation of a response design."""
    design = np.maximum(np.asarray(matrix, dtype=float), 0.0)
    if design.ndim != 2 or design.shape[0] == 0 or design.shape[1] < 2:
        return 1.0
    norm_sq = np.sum(design * design, axis=0)
    active = norm_sq > 1.0e-24
    if int(np.count_nonzero(active)) < 2:
        return 1.0
    active_design = design[:, active]
    active_norm_sq = norm_sq[active]
    cross = active_design.T @ active_design
    denom = np.sqrt(active_norm_sq[:, None] * active_norm_sq[None, :])
    corr = np.divide(
        cross,
        denom,
        out=np.zeros_like(cross, dtype=float),
        where=denom > 0.0,
    )
    np.fill_diagonal(corr, 0.0)
    return float(np.clip(np.max(corr), 0.0, 1.0))


def _max_column_correlation_after_candidate_batch(
    *,
    before_matrix: NDArray[np.float64],
    candidate_rows: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return max column correlation after appending each candidate row."""
    rows = np.maximum(np.asarray(candidate_rows, dtype=float), 0.0)
    if rows.ndim != 2:
        raise ValueError("candidate_rows must be shaped (N, M).")
    if rows.shape[0] == 0:
        return np.zeros(0, dtype=float)
    if rows.shape[1] < 2:
        return np.ones(rows.shape[0], dtype=float)
    before = np.maximum(np.asarray(before_matrix, dtype=float), 0.0)
    if before.ndim != 2 or before.shape[1] != rows.shape[1]:
        before = np.zeros((0, rows.shape[1]), dtype=float)
    before_cross = (
        before.T @ before
        if before.size
        else np.zeros((rows.shape[1], rows.shape[1]), dtype=float)
    )
    before_norm_sq = (
        np.sum(before * before, axis=0)
        if before.size
        else np.zeros(rows.shape[1], dtype=float)
    )
    cross = before_cross.reshape(1, rows.shape[1], rows.shape[1]) + np.einsum(
        "ni,nj->nij",
        rows,
        rows,
    )
    norm_sq = before_norm_sq.reshape(1, -1) + rows * rows
    denom = np.sqrt(norm_sq[:, :, None] * norm_sq[:, None, :])
    corr = np.divide(
        cross,
        denom,
        out=np.zeros_like(cross, dtype=float),
        where=denom > 0.0,
    )
    diagonal = np.arange(rows.shape[1])
    corr[:, diagonal, diagonal] = 0.0
    active_counts = np.count_nonzero(norm_sq > 1.0e-24, axis=1)
    max_corr = np.max(np.clip(corr, 0.0, 1.0), axis=(1, 2))
    max_corr[active_counts < 2] = 1.0
    return max_corr


def _station_correlation_reduction_gains_batch(
    candidate_poses_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
    modes_by_isotope: dict[str, list[SignatureMode]],
    *,
    config: DSSPPConfig,
    kernel: ContinuousKernel | None = None,
    estimator: RotatingShieldPFEstimator | None = None,
) -> NDArray[np.float64]:
    """Return candidate gains for reducing same-isotope response correlation."""
    candidates = np.asarray(candidate_poses_xyz, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidate_poses_xyz must be shaped (N, 3).")
    if candidates.shape[0] == 0:
        return np.zeros(0, dtype=float)
    visited = _pose_matrix_or_empty(visited_poses_xyz)
    gain_rows: list[NDArray[np.float64]] = []
    weight_values: list[float] = []
    for isotope, modes in modes_by_isotope.items():
        active = [mode for mode in modes if float(mode.weight) > 0.0]
        if len(active) < 2:
            continue
        before = _station_response_matrix(
            visited,
            active,
            live_time_s=float(config.live_time_s),
            kernel=kernel,
            estimator=estimator,
            isotope=isotope,
        )
        candidate_rows = _station_response_matrix(
            candidates,
            active,
            live_time_s=float(config.live_time_s),
            kernel=kernel,
            estimator=estimator,
            isotope=isotope,
        )
        before_corr = _max_column_correlation_from_design(before)
        after_corr = _max_column_correlation_after_candidate_batch(
            before_matrix=before,
            candidate_rows=candidate_rows,
        )
        gain_rows.append(np.maximum(before_corr - after_corr, 0.0))
        weight_values.append(float(sum(mode.weight for mode in active)))
    if not gain_rows:
        return np.zeros(candidates.shape[0], dtype=float)
    weights = _normalise_weights(np.asarray(weight_values, dtype=float))
    stacked = np.vstack(gain_rows)
    return np.sum(stacked * weights.reshape(-1, 1), axis=0)


def _isotope_balance_gains_batch(
    candidate_poses_xyz: NDArray[np.float64],
    modes_by_isotope: dict[str, list[SignatureMode]],
    *,
    config: DSSPPConfig,
    kernel: ContinuousKernel | None = None,
    estimator: RotatingShieldPFEstimator | None = None,
) -> NDArray[np.float64]:
    """Return gains that favor stations observing every modeled isotope."""
    candidates = np.asarray(candidate_poses_xyz, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidate_poses_xyz must be shaped (N, 3).")
    if candidates.shape[0] == 0:
        return np.zeros(0, dtype=float)
    utility_rows: list[NDArray[np.float64]] = []
    saturation = max(float(config.count_utility_saturation_counts), 1.0e-12)
    for isotope, modes in modes_by_isotope.items():
        active = [mode for mode in modes if float(mode.weight) > 0.0]
        if not active:
            continue
        response = _station_response_matrix(
            candidates,
            active,
            live_time_s=float(config.live_time_s),
            kernel=kernel,
            estimator=estimator,
            isotope=isotope,
        )
        mode_weights = _normalise_weights(
            np.asarray([float(mode.weight) for mode in active], dtype=float)
        )
        expected = np.sum(response * mode_weights.reshape(1, -1), axis=1)
        utility_rows.append(1.0 - np.exp(-np.maximum(expected, 0.0) / saturation))
    if not utility_rows:
        return np.zeros(candidates.shape[0], dtype=float)
    utilities = np.clip(np.vstack(utility_rows), 0.0, 1.0)
    weakest = np.min(utilities, axis=0)
    mean_utility = np.mean(utilities, axis=0)
    return np.clip(0.75 * weakest + 0.25 * mean_utility, 0.0, 1.0)


def _elevation_condition_gains_batch(
    candidate_poses_xyz: NDArray[np.float64],
    modes_by_isotope: dict[str, list[SignatureMode]],
    *,
    config: DSSPPConfig,
) -> NDArray[np.float64]:
    """Return candidate gains for separating posterior modes by elevation angle."""
    candidates = np.asarray(candidate_poses_xyz, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidate_poses_xyz must be shaped (N, 3).")
    gains = np.zeros(candidates.shape[0], dtype=float)
    if candidates.shape[0] == 0:
        return gains
    threshold = np.deg2rad(max(float(config.elevation_angle_threshold_deg), 1.0e-6))
    isotope_weight_values: list[float] = []
    isotope_gain_rows: list[NDArray[np.float64]] = []
    for modes in modes_by_isotope.values():
        active = [mode for mode in modes if float(mode.weight) > 0.0]
        if len(active) < 2:
            continue
        weights = _normalise_weights(
            np.asarray([float(mode.weight) for mode in active], dtype=float)
        )
        left, right, pair_weights = _elevation_pair_indices_and_weights(
            active,
            weights,
            config=config,
        )
        if left.size == 0:
            continue
        positions = np.vstack(
            [np.asarray(mode.position_xyz, dtype=float).reshape(3) for mode in active]
        )
        vectors = positions[None, :, :] - candidates[:, None, :]
        horizontal = np.linalg.norm(vectors[:, :, :2], axis=2)
        elevation = np.arctan2(vectors[:, :, 2], np.maximum(horizontal, 1.0e-9))
        pair_contrast = np.abs(elevation[:, left] - elevation[:, right])
        pair_scores = np.minimum(pair_contrast / threshold, 1.0)
        row = np.sum(pair_scores * pair_weights.reshape(1, -1), axis=1) / max(
            float(np.sum(pair_weights)),
            1.0e-12,
        )
        isotope_gain_rows.append(row)
        isotope_weight_values.append(float(sum(mode.weight for mode in active)))
    if not isotope_gain_rows:
        return gains
    isotope_weights = _normalise_weights(np.asarray(isotope_weight_values, dtype=float))
    stacked = np.vstack(isotope_gain_rows)
    return np.sum(stacked * isotope_weights.reshape(-1, 1), axis=0)


def _environment_log_attenuation_values(
    *,
    kernel: ContinuousKernel,
    isotope: str,
    modes: Sequence[SignatureMode],
    pose_xyz: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return obstacle-only log-transmission values for posterior modes."""
    pose = np.asarray(pose_xyz, dtype=float).reshape(3)
    values = [
        kernel.obstacle_log_attenuation_pair(
            isotope=isotope,
            source_pos=mode.position_xyz,
            detector_pos=pose,
        )
        for mode in modes
    ]
    return np.asarray(values, dtype=float)


def _environment_signature_score(
    *,
    kernel: ContinuousKernel,
    estimator: RotatingShieldPFEstimator,
    modes_by_isotope: dict[str, list[SignatureMode]],
    pose_xyz: NDArray[np.float64],
    config: DSSPPConfig,
) -> float:
    """Return obstacle-only separation between currently confusable modes."""
    if kernel.obstacle_boxes_m().size == 0:
        return 0.0
    isotope_weights = estimator.pf_config.alpha_weights or {
        isotope: 1.0 for isotope in estimator.isotopes
    }
    alpha_sum = sum(float(value) for value in isotope_weights.values()) or 1.0
    threshold = max(float(config.environment_contrast_threshold), 1.0e-12)
    total = 0.0
    for isotope in estimator.isotopes:
        modes = [
            mode
            for mode in modes_by_isotope.get(isotope, [])
            if float(mode.weight) > 0.0
        ]
        if len(modes) < 2:
            continue
        values = _environment_log_attenuation_values(
            kernel=kernel,
            isotope=isotope,
            modes=modes,
            pose_xyz=pose_xyz,
        )
        weights = _normalise_weights(
            np.asarray([mode.weight for mode in modes], dtype=float)
        )
        pair_score = 0.0
        pair_weight_sum = 0.0
        for left_idx in range(values.size):
            for right_idx in range(left_idx + 1, values.size):
                pair_weight = float(np.sqrt(max(weights[left_idx] * weights[right_idx], 0.0)))
                if pair_weight <= 0.0:
                    continue
                contrast = abs(float(values[left_idx] - values[right_idx]))
                pair_score += pair_weight * min(contrast / threshold, 1.0)
                pair_weight_sum += pair_weight
        if pair_weight_sum <= 0.0:
            continue
        isotope_weight = float(isotope_weights.get(isotope, 1.0)) / alpha_sum
        total += isotope_weight * float(pair_score / pair_weight_sum)
    return max(float(total), 0.0)


def _environment_signature_scores_batch(
    *,
    kernel: ContinuousKernel,
    estimator: RotatingShieldPFEstimator,
    modes_by_isotope: dict[str, list[SignatureMode]],
    poses_xyz: NDArray[np.float64],
    config: DSSPPConfig,
) -> NDArray[np.float64]:
    """Return obstacle signature separation for many candidate detector poses."""
    poses = np.asarray(poses_xyz, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 3:
        raise ValueError("poses_xyz must be shaped (N, 3).")
    scores = np.zeros(poses.shape[0], dtype=float)
    boxes = kernel.obstacle_boxes_m()
    if boxes.size == 0 or poses.shape[0] == 0:
        return scores
    isotope_weights = estimator.pf_config.alpha_weights or {
        isotope: 1.0 for isotope in estimator.isotopes
    }
    alpha_sum = sum(float(value) for value in isotope_weights.values()) or 1.0
    threshold = max(float(config.environment_contrast_threshold), 1.0e-12)
    for isotope in estimator.isotopes:
        modes = [
            mode
            for mode in modes_by_isotope.get(isotope, [])
            if float(mode.weight) > 0.0
        ]
        if len(modes) < 2:
            continue
        sources = np.vstack([np.asarray(mode.position_xyz, dtype=float) for mode in modes])
        values = kernel.obstacle_log_attenuation_matrix(
            isotope=isotope,
            sources_xyz=sources,
            detector_poses_xyz=poses,
        )
        weights = _normalise_weights(
            np.asarray([mode.weight for mode in modes], dtype=float)
        )
        left, right = np.triu_indices(values.shape[1], k=1)
        if left.size == 0:
            continue
        pair_weights = np.sqrt(np.maximum(weights[left] * weights[right], 0.0))
        valid = pair_weights > 0.0
        if not np.any(valid):
            continue
        left = left[valid]
        right = right[valid]
        pair_weights = pair_weights[valid]
        contrast = np.abs(values[:, left] - values[:, right])
        pair_scores = np.sum(
            np.minimum(contrast / threshold, 1.0) * pair_weights.reshape(1, -1),
            axis=1,
        ) / max(float(np.sum(pair_weights)), 1.0e-12)
        isotope_weight = float(isotope_weights.get(isotope, 1.0)) / alpha_sum
        scores += isotope_weight * pair_scores
    return np.maximum(scores, 0.0)


def _vertical_environment_signature_scores_batch(
    *,
    kernel: ContinuousKernel,
    estimator: RotatingShieldPFEstimator,
    modes_by_isotope: dict[str, list[SignatureMode]],
    poses_xyz: NDArray[np.float64],
    config: DSSPPConfig,
) -> NDArray[np.float64]:
    """Return obstacle attenuation contrast for vertically ambiguous modes."""
    poses = np.asarray(poses_xyz, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 3:
        raise ValueError("poses_xyz must be shaped (N, 3).")
    scores = np.zeros(poses.shape[0], dtype=float)
    boxes = kernel.obstacle_boxes_m()
    if boxes.size == 0 or poses.shape[0] == 0:
        return scores
    isotope_weights = estimator.pf_config.alpha_weights or {
        isotope: 1.0 for isotope in estimator.isotopes
    }
    alpha_sum = sum(float(value) for value in isotope_weights.values()) or 1.0
    threshold = max(float(config.environment_contrast_threshold), 1.0e-12)
    for isotope in estimator.isotopes:
        modes = [
            mode
            for mode in modes_by_isotope.get(isotope, [])
            if float(mode.weight) > 0.0
        ]
        if len(modes) < 2:
            continue
        weights = _normalise_weights(
            np.asarray([float(mode.weight) for mode in modes], dtype=float)
        )
        left, right, pair_weights = _elevation_pair_indices_and_weights(
            modes,
            weights,
            config=config,
        )
        if left.size == 0:
            continue
        sources = np.vstack([np.asarray(mode.position_xyz, dtype=float) for mode in modes])
        values = kernel.obstacle_log_attenuation_matrix(
            isotope=isotope,
            sources_xyz=sources,
            detector_poses_xyz=poses,
        )
        contrast = np.abs(values[:, left] - values[:, right])
        pair_scores = np.minimum(contrast / threshold, 1.0)
        isotope_weight = float(isotope_weights.get(isotope, 1.0)) / alpha_sum
        scores += isotope_weight * np.sum(
            pair_scores * pair_weights.reshape(1, -1),
            axis=1,
        ) / max(float(np.sum(pair_weights)), 1.0e-12)
    return np.maximum(scores, 0.0)


def _occlusion_boundary_gain(
    *,
    kernel: ContinuousKernel,
    estimator: RotatingShieldPFEstimator,
    modes_by_isotope: dict[str, list[SignatureMode]],
    pose_xyz: NDArray[np.float64],
    config: DSSPPConfig,
) -> float:
    """Return a gain for poses near obstacle-shadow transition boundaries."""
    if kernel.obstacle_boxes_m().size == 0:
        return 0.0
    step = max(float(config.occlusion_boundary_step_m), 0.0)
    if step <= 0.0:
        return 0.0
    pose = np.asarray(pose_xyz, dtype=float).reshape(3)
    offsets = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [step, 0.0, 0.0],
            [-step, 0.0, 0.0],
            [0.0, step, 0.0],
            [0.0, -step, 0.0],
        ],
        dtype=float,
    )
    isotope_weights = estimator.pf_config.alpha_weights or {
        isotope: 1.0 for isotope in estimator.isotopes
    }
    alpha_sum = sum(float(value) for value in isotope_weights.values()) or 1.0
    threshold = max(float(config.environment_contrast_threshold), 1.0e-12)
    total = 0.0
    for isotope in estimator.isotopes:
        modes = [
            mode
            for mode in modes_by_isotope.get(isotope, [])
            if float(mode.weight) > 0.0
        ]
        if not modes:
            continue
        mode_gains: list[float] = []
        mode_weights: list[float] = []
        for mode in modes:
            values = [
                kernel.obstacle_log_attenuation_pair(
                    isotope=isotope,
                    source_pos=mode.position_xyz,
                    detector_pos=pose + offset,
                )
                for offset in offsets
            ]
            spread = max(values) - min(values)
            mode_gains.append(min(float(spread) / threshold, 1.0))
            mode_weights.append(float(mode.weight))
        if not mode_gains:
            continue
        weights = _normalise_weights(np.asarray(mode_weights, dtype=float))
        isotope_weight = float(isotope_weights.get(isotope, 1.0)) / alpha_sum
        total += isotope_weight * float(np.sum(weights * np.asarray(mode_gains, dtype=float)))
    return max(float(total), 0.0)


def _occlusion_boundary_gains_batch(
    *,
    kernel: ContinuousKernel,
    estimator: RotatingShieldPFEstimator,
    modes_by_isotope: dict[str, list[SignatureMode]],
    poses_xyz: NDArray[np.float64],
    config: DSSPPConfig,
) -> NDArray[np.float64]:
    """Return obstacle-boundary gains for many candidate detector poses."""
    poses = np.asarray(poses_xyz, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 3:
        raise ValueError("poses_xyz must be shaped (N, 3).")
    gains = np.zeros(poses.shape[0], dtype=float)
    boxes = kernel.obstacle_boxes_m()
    if boxes.size == 0 or poses.shape[0] == 0:
        return gains
    step = max(float(config.occlusion_boundary_step_m), 0.0)
    if step <= 0.0:
        return gains
    offsets = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [step, 0.0, 0.0],
            [-step, 0.0, 0.0],
            [0.0, step, 0.0],
            [0.0, -step, 0.0],
        ],
        dtype=float,
    )
    offset_poses = (poses[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    isotope_weights = estimator.pf_config.alpha_weights or {
        isotope: 1.0 for isotope in estimator.isotopes
    }
    alpha_sum = sum(float(value) for value in isotope_weights.values()) or 1.0
    threshold = max(float(config.environment_contrast_threshold), 1.0e-12)
    for isotope in estimator.isotopes:
        modes = [
            mode
            for mode in modes_by_isotope.get(isotope, [])
            if float(mode.weight) > 0.0
        ]
        if not modes:
            continue
        sources = np.vstack([np.asarray(mode.position_xyz, dtype=float) for mode in modes])
        values = kernel.obstacle_log_attenuation_matrix(
            isotope=isotope,
            sources_xyz=sources,
            detector_poses_xyz=offset_poses,
        )
        values = values.reshape(poses.shape[0], offsets.shape[0], len(modes))
        spread = np.max(values, axis=1) - np.min(values, axis=1)
        mode_gains = np.minimum(spread / threshold, 1.0)
        weights = _normalise_weights(
            np.asarray([mode.weight for mode in modes], dtype=float)
        )
        isotope_weight = float(isotope_weights.get(isotope, 1.0)) / alpha_sum
        gains += isotope_weight * np.sum(mode_gains * weights.reshape(1, -1), axis=1)
    return np.maximum(gains, 0.0)


def _score_program(
    *,
    estimator: RotatingShieldPFEstimator,
    kernel: ContinuousKernel,
    modes_by_isotope: dict[str, list[SignatureMode]],
    pose_xyz: NDArray[np.float64],
    program: ShieldProgram,
    config: DSSPPConfig,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """Return all scalar shield-program score components."""
    num_orients = int(estimator.num_orientations)
    isotope_weights = estimator.pf_config.alpha_weights or {
        isotope: 1.0 for isotope in estimator.isotopes
    }
    alpha_sum = sum(float(v) for v in isotope_weights.values()) or 1.0
    signature_total = 0.0
    temporal_total = 0.0
    elevation_total = 0.0
    cardinality_gap_total = 0.0
    differential_terms: list[float] = []
    observation_counts: dict[str, float] = {}
    balance_counts: dict[str, float] = {}
    dose_score = 0.0
    room_z_m = _estimator_room_z(estimator)
    for isotope in estimator.isotopes:
        modes = modes_by_isotope.get(isotope, [])
        signatures: list[NDArray[np.float64]] = []
        weights: list[float] = []
        for mode in modes:
            signature = _expected_signature(
                kernel=kernel,
                estimator=estimator,
                mode=mode,
                pose_xyz=pose_xyz,
                program=program,
                num_orients=num_orients,
                live_time_s=config.live_time_s,
            )
            signatures.append(signature)
            weights.append(float(mode.weight))
        if signatures:
            isotope_weight = float(isotope_weights.get(isotope, 1.0)) / alpha_sum
            pair_priority = _high_surface_pair_priority_weights(
                modes,
                config=config,
                room_z_m=room_z_m,
            )
            signature_total += (
                isotope_weight
                * _signature_separation_score(
                    signatures,
                    variance_floor=config.count_variance_floor,
                )
            )
            temporal_total += isotope_weight * _temporal_separation_score_from_signatures(
                signatures,
                weights,
                config=config,
                pair_priority_weights=pair_priority,
            )
            elevation_total += isotope_weight * _elevation_signature_score_from_signatures(
                signatures,
                weights,
                modes,
                config=config,
                room_z_m=room_z_m,
            )
            program_raw = np.stack(signatures, axis=1).reshape(
                1,
                len(signatures[0]),
                len(signatures),
            )
            cardinality_gap_total += isotope_weight * float(
                _expected_bic_gap_against_source_removal_batch(
                    program_raw,
                    parameter_count_per_source=(
                        config.cardinality_bic_parameter_count_per_source
                    ),
                )[0]
            )
            mean_signature = _weighted_mean_signature(signatures, weights)
            observation_counts[isotope] = (
                float(np.max(mean_signature)) if mean_signature.size else 0.0
            )
            balance_counts[isotope] = (
                float(np.sum(mean_signature)) if mean_signature.size else 0.0
            )
            dose_score += float(np.sum(mean_signature))
            signature_std = float(np.std(mean_signature)) if mean_signature.size else 0.0
        else:
            observation_counts[isotope] = 0.0
            balance_counts[isotope] = 0.0
            signature_std = 0.0
        min_std = max(float(config.signature_std_min_counts), 0.0)
        if min_std > 0.0:
            shortfall = max(0.0, 1.0 - signature_std / min_std)
            differential_terms.append(shortfall * shortfall)
    observation_penalty = minimum_observation_shortfall(
        observation_counts,
        min_counts=float(config.min_observation_counts),
    )
    differential_penalty = (
        float(np.mean(differential_terms)) if differential_terms else 0.0
    )
    count_balance_penalty = _count_balance_penalty(balance_counts)
    count_utility = _saturated_count_utility(
        balance_counts,
        saturation_counts=float(config.count_utility_saturation_counts),
    )
    return (
        signature_total,
        temporal_total,
        elevation_total,
        observation_penalty,
        count_balance_penalty,
        differential_penalty,
        dose_score,
        count_utility,
        cardinality_gap_total,
    )


def _pairwise_contrast_cover_programs(
    *,
    estimator: RotatingShieldPFEstimator,
    kernel: ContinuousKernel,
    modes_by_isotope: dict[str, list[SignatureMode]],
    pose_xyz: NDArray[np.float64],
    config: DSSPPConfig,
    pair_cache: dict[str, tuple[NDArray[np.float64], list[float]]] | None = None,
) -> list[ShieldProgram]:
    """Build pose-specific pairwise-cover programs from all shield pairs."""
    if (
        float(config.lambda_signature) <= 0.0
        and float(config.lambda_temporal_separation) <= 0.0
        and float(config.lambda_elevation_signature) <= 0.0
    ):
        return []
    max_programs = max(0, int(config.temporal_cover_programs))
    if max_programs <= 0:
        return []
    has_multi_mode = any(len(modes) >= 2 for modes in modes_by_isotope.values())
    if not has_multi_mode:
        return []
    num_orients = int(estimator.num_orientations)
    program_length = max(1, int(config.program_length))
    all_pairs = tuple(range(num_orients * num_orients))
    beam_width = max(1, int(config.temporal_cover_beam_width))
    if pair_cache is None:
        pair_cache = _build_pair_signature_cache(
            kernel=kernel,
            estimator=estimator,
            modes_by_isotope=modes_by_isotope,
            pose_xyz=pose_xyz,
            config=config,
        )
    beam: list[tuple[tuple[int, ...], float]] = [(tuple(), 0.0)]
    for _ in range(program_length):
        candidate_sequences: list[tuple[int, ...]] = []
        for prefix, _score in beam:
            used = set(prefix)
            candidate_sequences.extend(
                tuple(prefix + (int(pair_id),))
                for pair_id in all_pairs
                if int(pair_id) not in used
            )
        if not candidate_sequences:
            break
        if pair_cache is not None:
            candidate_programs = [
                ShieldProgram(
                    name="pairwise_contrast_probe",
                    pair_ids=sequence,
                    kind="pairwise_contrast_cover",
                )
                for sequence in candidate_sequences
            ]
            signature_scores, temporal_scores, elevation_scores, _, _, _, _, _, _ = (
                _score_programs_from_pair_cache(
                    estimator=estimator,
                    pair_cache=pair_cache,
                    modes_by_isotope=modes_by_isotope,
                    programs=candidate_programs,
                    config=config,
                )
            )
            objective_scores = _pairwise_cover_objective(
                signature_scores,
                temporal_scores,
                elevation_scores,
                config=config,
            )
        else:
            objective_values: list[float] = []
            for sequence in candidate_sequences:
                program = ShieldProgram(
                    name="pairwise_contrast_probe",
                    pair_ids=sequence,
                    kind="pairwise_contrast_cover",
                )
                score_row = _score_program(
                    estimator=estimator,
                    kernel=kernel,
                    modes_by_isotope=modes_by_isotope,
                    pose_xyz=pose_xyz,
                    program=program,
                    config=config,
                )
                objective_score = float(
                    _pairwise_cover_objective(
                        np.asarray([score_row[0]], dtype=float),
                        np.asarray([score_row[1]], dtype=float),
                        np.asarray([score_row[2]], dtype=float),
                        config=config,
                    )[0]
                )
                objective_values.append(objective_score)
            objective_scores = np.asarray(objective_values, dtype=float)
        finite = np.isfinite(objective_scores) & (objective_scores > 0.0)
        if not np.any(finite):
            break
        valid_indices = np.flatnonzero(finite)
        order = valid_indices[np.argsort(objective_scores[valid_indices])[::-1]]
        next_beam: list[tuple[tuple[int, ...], float]] = []
        seen: set[tuple[int, ...]] = set()
        for idx in order:
            sequence = candidate_sequences[int(idx)]
            if sequence in seen:
                continue
            seen.add(sequence)
            next_beam.append((sequence, float(objective_scores[int(idx)])))
            if len(next_beam) >= beam_width:
                break
        beam = next_beam
        if not beam:
            break
    ranked = [
        (sequence, score)
        for sequence, score in beam
        if sequence and np.isfinite(score) and float(score) > 0.0
    ]
    ranked.sort(key=lambda item: float(item[1]), reverse=True)
    return [
        ShieldProgram(
            name=f"pairwise_contrast_cover_{rank + 1}_{len(sequence)}",
            pair_ids=tuple(sequence),
            kind="pairwise_contrast_cover",
        )
        for rank, (sequence, _score) in enumerate(ranked[:max_programs])
    ]


def _greedy_pairwise_contrast_program(
    *,
    estimator: RotatingShieldPFEstimator,
    kernel: ContinuousKernel,
    modes_by_isotope: dict[str, list[SignatureMode]],
    pose_xyz: NDArray[np.float64],
    config: DSSPPConfig,
    pair_cache: dict[str, tuple[NDArray[np.float64], list[float]]] | None = None,
) -> ShieldProgram | None:
    """Build the best pose-specific pairwise-cover shield program."""
    programs = _pairwise_contrast_cover_programs(
        estimator=estimator,
        kernel=kernel,
        modes_by_isotope=modes_by_isotope,
        pose_xyz=pose_xyz,
        config=config,
        pair_cache=pair_cache,
    )
    if not programs:
        return None
    return programs[0]


def _batch_program_score_rows(
    *,
    estimator: RotatingShieldPFEstimator,
    pair_cache: dict[str, tuple[NDArray[np.float64], list[float]]],
    modes_by_isotope: dict[str, list[SignatureMode]] | None = None,
    programs: Sequence[ShieldProgram],
    config: DSSPPConfig,
) -> list[tuple[float, float, float, float, float, float, float, float, float]]:
    """Return program score tuples in the same order as the input programs."""
    (
        signature_scores,
        temporal_scores,
        elevation_scores,
        observation_penalties,
        count_balance_penalties,
        differential_penalties,
        dose_scores,
        count_utilities,
        cardinality_gap_gains,
    ) = _score_programs_from_pair_cache(
        estimator=estimator,
        pair_cache=pair_cache,
        modes_by_isotope=modes_by_isotope,
        programs=programs,
        config=config,
    )
    return [
        (
            float(signature_scores[index]),
            float(temporal_scores[index]),
            float(elevation_scores[index]),
            float(observation_penalties[index]),
            float(count_balance_penalties[index]),
            float(differential_penalties[index]),
            float(dose_scores[index]),
            float(count_utilities[index]),
            float(cardinality_gap_gains[index]),
        )
        for index in range(len(programs))
    ]


def _programs_for_pose(
    *,
    estimator: RotatingShieldPFEstimator,
    kernel: ContinuousKernel,
    modes_by_isotope: dict[str, list[SignatureMode]],
    pose_xyz: NDArray[np.float64],
    base_programs: Sequence[ShieldProgram],
    config: DSSPPConfig,
    include_pose_specific_cover: bool = True,
    pair_cache: dict[str, tuple[NDArray[np.float64], list[float]]] | None = None,
) -> list[ShieldProgram]:
    """Return base programs plus pose-specific temporal-code programs."""
    programs = list(base_programs)
    if not include_pose_specific_cover:
        return programs
    cover_programs = _pairwise_contrast_cover_programs(
        estimator=estimator,
        kernel=kernel,
        modes_by_isotope=modes_by_isotope,
        pose_xyz=pose_xyz,
        config=config,
        pair_cache=pair_cache,
    )
    if not cover_programs:
        return programs
    seen = {tuple(program.pair_ids) for program in programs}
    for cover_program in cover_programs:
        key = tuple(cover_program.pair_ids)
        if key not in seen:
            seen.add(key)
            programs.append(cover_program)
    return programs


def _cardinality_evidence_gap_pressure(
    estimator: RotatingShieldPFEstimator,
    config: DSSPPConfig,
) -> float:
    """Return planning pressure from unresolved sparse cardinality evidence gaps."""
    target = max(float(config.cardinality_evidence_gap_target), 1.0e-12)
    diagnostics: dict[str, object] = {}
    getter = getattr(estimator, "sparse_poisson_evidence_diagnostics", None)
    if callable(getter):
        try:
            diagnostics = dict(getter())
        except (RuntimeError, ValueError, TypeError):
            diagnostics = {}
    if not diagnostics:
        report_getter = getattr(estimator, "report_model_order_diagnostics", None)
        if callable(report_getter):
            try:
                report_payload = dict(report_getter())
            except (RuntimeError, ValueError, TypeError):
                report_payload = {}
            for isotope, stats in report_payload.items():
                if not isinstance(stats, dict):
                    continue
                sparse_stats = stats.get("sparse_poisson_evidence")
                diagnostics[str(isotope)] = (
                    sparse_stats if isinstance(sparse_stats, dict) else stats
                )
    pressure = 0.0
    for stats in diagnostics.values():
        if not isinstance(stats, dict) or not bool(stats.get("available", True)):
            continue
        selected_count = int(stats.get("selected_count", 0))
        gaps: list[float] = []
        for key in ("criterion_margin_to_runner_up", "bic_margin_to_runner_up"):
            if key in stats:
                gaps.append(float(stats.get(key, float("inf"))))
        if selected_count > 0:
            gaps.append(float(stats.get("criterion_margin_to_simpler", float("inf"))))
            gaps.append(float(stats.get("bic_gap_to_previous_count", float("inf"))))
        gaps.append(float(stats.get("bic_gap_to_next_count", float("inf"))))
        finite = [value for value in gaps if np.isfinite(value)]
        ready = bool(stats.get("model_order_ready", False))
        if not finite:
            pressure += 0.0 if ready else 1.0
            continue
        weakest_gap = min(float(value) for value in finite)
        gap_pressure = max(target - weakest_gap, 0.0) / target
        if not ready and gap_pressure <= 0.0:
            gap_pressure = 1.0
        pressure += gap_pressure
    return float(max(pressure, 0.0))


def _static_station_program_score(
    *,
    signature_score: float,
    temporal_separation_score: float,
    elevation_signature_score: float,
    count_utility: float,
    count_balance_penalty: float,
    differential_penalty: float,
    dose_score: float,
    coverage_norm: float,
    revisit_penalty: float,
    bearing_gain: float,
    frontier_gain: float,
    turn_penalty: float,
    local_orbit_gain: float,
    station_condition_gain: float,
    correlation_reduction_gain: float,
    cardinality_gap_gain: float,
    isotope_balance_gain: float,
    environment_signature_score: float,
    occlusion_boundary_gain: float,
    elevation_condition_gain: float,
    vertical_environment_signature_score: float,
    cardinality_evidence_pressure: float,
    remaining_route_pressure: float,
    remaining_route_penalty: float,
    remaining_route_gain: float,
    coverage_floor: float,
    config: DSSPPConfig,
) -> float:
    """Return the non-transition score for one station-program pair."""
    return float(
        float(config.lambda_signature) * float(np.log1p(max(signature_score, 0.0)))
        + float(config.lambda_temporal_separation)
        * float(np.log1p(max(temporal_separation_score, 0.0)))
        + float(config.lambda_elevation_signature)
        * float(np.log1p(max(elevation_signature_score, 0.0)))
        + float(config.lambda_coverage) * float(coverage_norm)
        + float(config.lambda_bearing_diversity) * float(bearing_gain)
        + float(config.lambda_frontier) * float(frontier_gain)
        + float(config.lambda_count_utility) * float(count_utility)
        + float(config.lambda_local_orbit) * float(local_orbit_gain)
        + float(config.lambda_station_condition)
        * float(np.log1p(max(station_condition_gain, 0.0)))
        + float(config.lambda_correlation_reduction)
        * float(np.log1p(max(correlation_reduction_gain, 0.0)))
        + float(config.lambda_cardinality_discrimination)
        * float(max(cardinality_evidence_pressure, 0.0))
        * (
            float(np.log1p(max(station_condition_gain, 0.0)))
            + float(np.log1p(max(correlation_reduction_gain, 0.0)))
            + float(np.log1p(max(elevation_condition_gain, 0.0)))
            + float(
                np.log1p(
                    max(cardinality_gap_gain, 0.0)
                    / max(float(config.cardinality_evidence_gap_target), 1.0e-12)
                )
            )
        )
        + float(config.lambda_isotope_balance) * float(isotope_balance_gain)
        + float(config.lambda_elevation_condition)
        * float(np.log1p(max(elevation_condition_gain, 0.0)))
        + float(config.lambda_environment_signature)
        * _normalized_environment_signature_score(
            environment_signature_score,
            config=config,
        )
        + float(config.lambda_vertical_environment_signature)
        * _normalized_environment_signature_score(
            vertical_environment_signature_score,
            config=config,
        )
        + float(config.lambda_occlusion_boundary)
        * float(np.log1p(max(occlusion_boundary_gain, 0.0)))
        + float(config.remaining_route_weight)
        * float(remaining_route_pressure)
        * (float(remaining_route_gain) - float(remaining_route_penalty))
        - float(config.lambda_dose) * float(dose_score)
        - float(config.eta_count_balance) * float(count_balance_penalty)
        - float(config.eta_differential) * float(differential_penalty)
        - float(config.eta_revisit) * float(revisit_penalty)
        - float(config.lambda_turn_smoothness) * float(turn_penalty)
        - float(config.coverage_floor_weight)
        * max(0.0, float(coverage_floor) - float(coverage_norm)) ** 2
    )


def _normalized_environment_signature_score(
    score: float | NDArray[np.float64],
    *,
    config: DSSPPConfig,
) -> float | NDArray[np.float64]:
    """
    Return a clipped 0..1 obstacle-signature score for weak planner biasing.

    The raw obstacle signature can be very large near occlusion boundaries.  The
    planner should treat it as a bounded tie-breaker, not as a replacement for
    count observability, shield-signature separation, or traversability costs.
    """
    raw = np.log1p(np.maximum(np.asarray(score, dtype=float), 0.0))
    clip = max(float(config.environment_signature_score_clip), 1.0e-12)
    denom = max(float(np.log1p(clip)), 1.0e-12)
    normalized = np.clip(raw / denom, 0.0, 1.0)
    if np.ndim(score) == 0:
        return float(normalized)
    return normalized


def _evaluate_pose_index_from_context(
    pose_index_value: int,
) -> tuple[int, float, list[tuple[Any, ...]], list[float], list[float]]:
    """Evaluate all shield programs for one candidate station from worker context."""
    context = _DSS_PP_POSE_EVAL_CONTEXT
    if context is None:
        raise RuntimeError("DSS-PP pose evaluation context is not initialized.")
    pose_index = int(pose_index_value)
    candidate_poses = np.asarray(context["candidate_poses"], dtype=float)
    path_lengths = np.asarray(context["path_lengths"], dtype=float)
    pair_caches_by_pose = cast(
        list[dict[str, tuple[NDArray[np.float64], list[float]]]],
        context["pair_caches_by_pose"],
    )
    programs = cast(Sequence[ShieldProgram], context["programs"])
    estimator = cast(RotatingShieldPFEstimator, context["estimator"])
    kernel = cast(ContinuousKernel, context["kernel"])
    modes_by_isotope = cast(
        dict[str, list[SignatureMode]],
        context["modes_by_isotope"],
    )
    config = cast(DSSPPConfig, context["config"])
    coverage_norm = np.asarray(context["coverage_norm"], dtype=float)
    coverage_raw = np.asarray(context["coverage_raw"], dtype=float)
    revisit_penalties = np.asarray(context["revisit_penalties"], dtype=float)
    bearing_gains = np.asarray(context["bearing_gains"], dtype=float)
    frontier_gains = np.asarray(context["frontier_gains"], dtype=float)
    turn_penalties = np.asarray(context["turn_penalties"], dtype=float)
    local_orbit_gains = np.asarray(context["local_orbit_gains"], dtype=float)
    station_condition_gains = np.asarray(
        context["station_condition_gains"],
        dtype=float,
    )
    correlation_reduction_gains = np.asarray(
        context["correlation_reduction_gains"],
        dtype=float,
    )
    isotope_balance_gains = np.asarray(
        context["isotope_balance_gains"],
        dtype=float,
    )
    elevation_condition_gains = np.asarray(
        context["elevation_condition_gains"],
        dtype=float,
    )
    environment_signature_scores = np.asarray(
        context["environment_signature_scores"],
        dtype=float,
    )
    vertical_environment_signature_scores = np.asarray(
        context["vertical_environment_signature_scores"],
        dtype=float,
    )
    occlusion_boundary_gains = np.asarray(
        context["occlusion_boundary_gains"],
        dtype=float,
    )
    cardinality_evidence_pressure = float(context["cardinality_evidence_pressure"])
    remaining_route_pressure = float(context["remaining_route_pressure"])
    remaining_route_penalties = np.asarray(
        context["remaining_route_penalties"],
        dtype=float,
    )
    remaining_route_gains = np.asarray(
        context["remaining_route_gains"],
        dtype=float,
    )
    coverage_floor = float(context["coverage_floor"])

    local_pending: list[tuple[Any, ...]] = []
    local_scores: list[float] = []
    local_observation_penalties: list[float] = []
    local_cheap_score = -np.inf
    pose = candidate_poses[pose_index]
    if not np.isfinite(path_lengths[pose_index]):
        return (
            pose_index,
            local_cheap_score,
            local_pending,
            local_scores,
            local_observation_penalties,
        )
    pair_cache = pair_caches_by_pose[pose_index]
    pose_programs = _programs_for_pose(
        estimator=estimator,
        kernel=kernel,
        modes_by_isotope=modes_by_isotope,
        pose_xyz=pose,
        base_programs=programs,
        config=config,
        include_pose_specific_cover=config.forced_program_pair_ids is None,
        pair_cache=pair_cache,
    )
    program_score_rows = _batch_program_score_rows(
        estimator=estimator,
        pair_cache=pair_cache,
        modes_by_isotope=modes_by_isotope,
        programs=pose_programs,
        config=config,
    )
    for program, score_row in zip(pose_programs, program_score_rows):
        (
            signature_score,
            temporal_separation_score,
            elevation_signature_score,
            observation_penalty,
            count_balance_penalty,
            differential_penalty,
            dose_score,
            count_utility,
            cardinality_gap_gain,
        ) = score_row
        static_score = _static_station_program_score(
            signature_score=signature_score,
            temporal_separation_score=temporal_separation_score,
            elevation_signature_score=elevation_signature_score,
            count_utility=count_utility,
            count_balance_penalty=count_balance_penalty,
            differential_penalty=differential_penalty,
            dose_score=dose_score,
            coverage_norm=float(coverage_norm[pose_index]),
            revisit_penalty=float(revisit_penalties[pose_index]),
            bearing_gain=float(bearing_gains[pose_index]),
            frontier_gain=float(frontier_gains[pose_index]),
            turn_penalty=float(turn_penalties[pose_index]),
            local_orbit_gain=float(local_orbit_gains[pose_index]),
            station_condition_gain=float(station_condition_gains[pose_index]),
            correlation_reduction_gain=float(correlation_reduction_gains[pose_index]),
            cardinality_gap_gain=float(cardinality_gap_gain),
            isotope_balance_gain=float(isotope_balance_gains[pose_index]),
            elevation_condition_gain=float(elevation_condition_gains[pose_index]),
            environment_signature_score=float(
                environment_signature_scores[pose_index]
            ),
            vertical_environment_signature_score=float(
                vertical_environment_signature_scores[pose_index]
            ),
            occlusion_boundary_gain=float(occlusion_boundary_gains[pose_index]),
            cardinality_evidence_pressure=cardinality_evidence_pressure,
            remaining_route_pressure=remaining_route_pressure,
            remaining_route_penalty=float(remaining_route_penalties[pose_index]),
            remaining_route_gain=float(remaining_route_gains[pose_index]),
            coverage_floor=coverage_floor,
            config=config,
        )
        local_cheap_score = max(float(local_cheap_score), float(static_score))
        local_scores.append(static_score)
        local_observation_penalties.append(observation_penalty)
        local_pending.append(
            (
                pose_index,
                pose.copy(),
                program,
                0.0,
                signature_score,
                temporal_separation_score,
                elevation_signature_score,
                observation_penalty,
                count_balance_penalty,
                differential_penalty,
                dose_score,
                count_utility,
                float(coverage_raw[pose_index]),
                float(revisit_penalties[pose_index]),
                float(bearing_gains[pose_index]),
                float(frontier_gains[pose_index]),
                float(turn_penalties[pose_index]),
                float(local_orbit_gains[pose_index]),
                float(station_condition_gains[pose_index]),
                float(correlation_reduction_gains[pose_index]),
                float(cardinality_gap_gain),
                float(isotope_balance_gains[pose_index]),
                float(elevation_condition_gains[pose_index]),
                float(environment_signature_scores[pose_index]),
                float(vertical_environment_signature_scores[pose_index]),
                float(occlusion_boundary_gains[pose_index]),
                remaining_route_pressure,
                float(remaining_route_penalties[pose_index]),
                float(remaining_route_gains[pose_index]),
            )
        )
    return (
        pose_index,
        local_cheap_score,
        local_pending,
        local_scores,
        local_observation_penalties,
    )


def _evaluate_pose_indices_parallel(
    eval_indices: Sequence[int],
    *,
    context: dict[str, object],
    worker_count: int,
) -> list[tuple[int, float, list[tuple[Any, ...]], list[float], list[float]]]:
    """Evaluate pose-program scores using process workers when available."""
    global _DSS_PP_POSE_EVAL_CONTEXT
    indices = [int(index) for index in eval_indices]
    previous_context = _DSS_PP_POSE_EVAL_CONTEXT
    _DSS_PP_POSE_EVAL_CONTEXT = context
    try:
        if int(worker_count) <= 1 or len(indices) <= 1:
            return [_evaluate_pose_index_from_context(index) for index in indices]
        max_workers = min(len(indices), max(1, int(worker_count)))
        if os.name == "posix":
            try:
                fork_context = mp.get_context("fork")
                with ProcessPoolExecutor(
                    max_workers=max_workers,
                    mp_context=fork_context,
                ) as executor:
                    return list(
                        executor.map(
                            _evaluate_pose_index_from_context,
                            indices,
                            chunksize=1,
                        )
                    )
            except Exception:
                pass
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(_evaluate_pose_index_from_context, indices))
    finally:
        _DSS_PP_POSE_EVAL_CONTEXT = previous_context


def _node_path_length(
    map_api: object | None,
    start_xyz: NDArray[np.float64],
    goal_xyz: NDArray[np.float64],
) -> float:
    """Return grid path length when possible, otherwise Euclidean distance."""
    start = np.asarray(start_xyz, dtype=float)
    goal = np.asarray(goal_xyz, dtype=float)
    if map_api is None:
        return float(np.linalg.norm(goal - start))
    cell_index = getattr(map_api, "cell_index", None)
    if not callable(cell_index):
        return float(np.linalg.norm(goal - start))
    start_cell = cell_index(start)
    goal_cell = cell_index(goal)
    if start_cell is None or goal_cell is None:
        return float(np.linalg.norm(goal - start))
    cache_key = (id(map_api), tuple(start_cell), tuple(goal_cell))
    cached = _DSS_PP_PATH_LENGTH_CACHE.get(cache_key)
    if cached is not None:
        return float(cached)
    length = shortest_grid_path_length(map_api, start, goal, allow_diagonal=True)
    if len(_DSS_PP_PATH_LENGTH_CACHE) >= _DSS_PP_PATH_LENGTH_CACHE_MAX:
        _DSS_PP_PATH_LENGTH_CACHE.clear()
    _DSS_PP_PATH_LENGTH_CACHE[cache_key] = float(length)
    _DSS_PP_PATH_LENGTH_CACHE[(id(map_api), tuple(goal_cell), tuple(start_cell))] = float(
        length
    )
    return float(length)


def _filter_path_reachable_stations(
    candidate_poses_xyz: NDArray[np.float64],
    *,
    current_pose_xyz: NDArray[np.float64],
    map_api: object | None,
) -> tuple[NDArray[np.float64], int]:
    """Remove station candidates that have no traversable path from the robot."""
    candidates = np.asarray(candidate_poses_xyz, dtype=float)
    if candidates.size == 0:
        return np.zeros((0, 3), dtype=float), 0
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidate_poses_xyz must be shape (N, 3).")
    reachable = np.asarray(
        [
            np.isfinite(_node_path_length(map_api, current_pose_xyz, candidate))
            for candidate in candidates
        ],
        dtype=bool,
    )
    removed = int(np.count_nonzero(~reachable))
    if not np.any(reachable):
        return np.zeros((0, 3), dtype=float), removed
    return candidates[reachable], removed


def _free_cell_centers(
    map_api: object | None,
    *,
    z_value: float,
    max_cells: int,
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
) -> NDArray[np.float64]:
    """Return free-cell center positions for coverage scoring."""
    if map_api is None:
        return _bounds_cell_centers(
            bounds_xyz,
            z_value=z_value,
            max_cells=max_cells,
        )
    grid_shape = getattr(map_api, "grid_shape", None)
    if grid_shape is None:
        return _bounds_cell_centers(
            bounds_xyz,
            z_value=z_value,
            max_cells=max_cells,
        )
    traversable_cells = getattr(map_api, "traversable_cells", None)
    if traversable_cells is not None:
        cells = [tuple(cell) for cell in traversable_cells]
    else:
        is_free_cell = getattr(map_api, "is_free_cell", None)
        if not callable(is_free_cell):
            is_free_cell = getattr(map_api, "is_cell_free", None)
        if not callable(is_free_cell):
            return np.zeros((0, 3), dtype=float)
        cells = [
            (ix, iy)
            for ix in range(int(grid_shape[0]))
            for iy in range(int(grid_shape[1]))
            if bool(is_free_cell((ix, iy)))
        ]
    if not cells:
        return np.zeros((0, 3), dtype=float)
    max_count = max(0, int(max_cells))
    if max_count > 0 and len(cells) > max_count:
        indices = np.linspace(0, len(cells) - 1, max_count, dtype=int)
        cells = [cells[int(idx)] for idx in indices]
    centers = [_cell_center(map_api, tuple(cell), z_value) for cell in cells]
    return np.vstack(centers).astype(float)


def _bounds_cell_centers(
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]] | None,
    *,
    z_value: float,
    max_cells: int,
) -> NDArray[np.float64]:
    """Return rectangular free-space samples when no traversability map exists."""
    if bounds_xyz is None:
        return np.zeros((0, 3), dtype=float)
    lo = np.asarray(bounds_xyz[0], dtype=float)
    hi = np.asarray(bounds_xyz[1], dtype=float)
    if lo.shape != (3,) or hi.shape != (3,):
        return np.zeros((0, 3), dtype=float)
    span = np.maximum(hi[:2] - lo[:2], 0.0)
    if float(span[0]) <= 0.0 or float(span[1]) <= 0.0:
        return np.zeros((0, 3), dtype=float)
    target = max(4, int(max_cells))
    aspect = float(span[0]) / max(float(span[1]), 1e-12)
    nx = max(2, int(np.sqrt(float(target) * aspect)))
    ny = max(2, int(np.ceil(float(target) / max(nx, 1))))
    if nx * ny > target:
        scale = np.sqrt(float(target) / float(nx * ny))
        nx = max(2, int(np.floor(nx * scale)))
        ny = max(2, int(np.floor(ny * scale)))
    xs = np.linspace(float(lo[0]), float(hi[0]), num=nx)
    ys = np.linspace(float(lo[1]), float(hi[1]), num=ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    zz = np.full(xx.size, float(z_value), dtype=float)
    return np.column_stack([xx.ravel(), yy.ravel(), zz])


def _coverage_gain_fraction(
    *,
    cell_centers_xyz: NDArray[np.float64],
    candidate_pose_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
    radius_m: float,
) -> float:
    """Return newly covered free-space fraction for a candidate station."""
    centers = np.asarray(cell_centers_xyz, dtype=float)
    if centers.size == 0:
        return 0.0
    radius = max(float(radius_m), 0.0)
    if radius <= 0.0:
        return 0.0
    candidate = np.asarray(candidate_pose_xyz, dtype=float).reshape(3)
    candidate_dist = np.linalg.norm(centers[:, :2] - candidate[None, :2], axis=1)
    candidate_covered = candidate_dist <= radius
    if not np.any(candidate_covered):
        return 0.0
    visited_covered = np.zeros(centers.shape[0], dtype=bool)
    if visited_poses_xyz is not None:
        visited = np.asarray(visited_poses_xyz, dtype=float)
        if visited.ndim == 1 and visited.size == 3:
            visited = visited.reshape(1, 3)
        if visited.ndim == 2 and visited.shape[1] == 3 and visited.size:
            visited_dist = np.linalg.norm(
                centers[:, None, :2] - visited[None, :, :2],
                axis=2,
            )
            visited_covered = np.min(visited_dist, axis=1) <= radius
    newly_covered = candidate_covered & ~visited_covered
    return float(np.count_nonzero(newly_covered)) / float(centers.shape[0])


def _pose_matrix_or_empty(poses_xyz: NDArray[np.float64] | None) -> NDArray[np.float64]:
    """Return poses as an N x 3 array or an empty array if invalid."""
    if poses_xyz is None:
        return np.zeros((0, 3), dtype=float)
    poses = np.asarray(poses_xyz, dtype=float)
    if poses.ndim == 1 and poses.size == 3:
        poses = poses.reshape(1, 3)
    if poses.ndim != 2 or poses.shape[1] != 3 or poses.size == 0:
        return np.zeros((0, 3), dtype=float)
    return poses


def _coverage_gain_fractions_batch(
    *,
    cell_centers_xyz: NDArray[np.float64],
    candidate_poses_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
    radius_m: float,
) -> NDArray[np.float64]:
    """Return newly covered free-space fractions for many candidate stations."""
    centers = np.asarray(cell_centers_xyz, dtype=float)
    candidates = np.asarray(candidate_poses_xyz, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidate_poses_xyz must be shaped (N, 3).")
    if centers.size == 0 or candidates.shape[0] == 0:
        return np.zeros(candidates.shape[0], dtype=float)
    radius = max(float(radius_m), 0.0)
    if radius <= 0.0:
        return np.zeros(candidates.shape[0], dtype=float)
    visited = _pose_matrix_or_empty(visited_poses_xyz)
    visited_covered = np.zeros(centers.shape[0], dtype=bool)
    if visited.size:
        visited_dist = np.linalg.norm(
            centers[:, None, :2] - visited[None, :, :2],
            axis=2,
        )
        visited_covered = np.min(visited_dist, axis=1) <= radius
    candidate_dist = np.linalg.norm(
        candidates[:, None, :2] - centers[None, :, :2],
        axis=2,
    )
    newly_covered = (candidate_dist <= radius) & ~visited_covered.reshape(1, -1)
    return np.count_nonzero(newly_covered, axis=1).astype(float) / float(
        centers.shape[0]
    )


def _station_revisit_penalty(
    candidate_pose_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
    *,
    min_separation_m: float,
) -> float:
    """Return a normalized penalty for selecting a near-visited station."""
    min_sep = max(float(min_separation_m), 0.0)
    if min_sep <= 0.0 or visited_poses_xyz is None:
        return 0.0
    visited = np.asarray(visited_poses_xyz, dtype=float)
    if visited.ndim == 1 and visited.size == 3:
        visited = visited.reshape(1, 3)
    if visited.ndim != 2 or visited.shape[1] != 3 or visited.size == 0:
        return 0.0
    candidate = np.asarray(candidate_pose_xyz, dtype=float).reshape(3)
    min_dist = float(np.min(np.linalg.norm(visited[:, :2] - candidate[None, :2], axis=1)))
    if min_dist >= min_sep:
        return 0.0
    shortfall = 1.0 - min_dist / max(min_sep, 1e-12)
    return float(shortfall * shortfall)


def _station_revisit_penalties_batch(
    candidate_poses_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
    *,
    min_separation_m: float,
) -> NDArray[np.float64]:
    """Return revisit penalties for many candidate stations."""
    candidates = np.asarray(candidate_poses_xyz, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidate_poses_xyz must be shaped (N, 3).")
    penalties = np.zeros(candidates.shape[0], dtype=float)
    min_sep = max(float(min_separation_m), 0.0)
    visited = _pose_matrix_or_empty(visited_poses_xyz)
    if min_sep <= 0.0 or visited.size == 0 or candidates.shape[0] == 0:
        return penalties
    distances = np.linalg.norm(
        candidates[:, None, :2] - visited[None, :, :2],
        axis=2,
    )
    min_dist = np.min(distances, axis=1)
    shortfall = 1.0 - min_dist / max(min_sep, 1.0e-12)
    active = min_dist < min_sep
    penalties[active] = shortfall[active] * shortfall[active]
    return penalties


def _bearing_diversity_gain(
    candidate_pose_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
    modes_by_isotope: dict[str, list[SignatureMode]],
) -> float:
    """
    Return an isotope-agnostic gain for new bearings of multi-mode posteriors.

    The term activates only for isotopes with multiple posterior modes. It
    rewards stations that separate those modes angularly and provide bearings
    different from already visited stations, which is the generic observability
    need behind same-isotope source separation.
    """
    candidate = np.asarray(candidate_pose_xyz, dtype=float).reshape(3)
    visited = None
    if visited_poses_xyz is not None:
        visited = np.asarray(visited_poses_xyz, dtype=float)
        if visited.ndim == 1 and visited.size == 3:
            visited = visited.reshape(1, 3)
        if visited.ndim != 2 or visited.shape[1] != 3 or visited.size == 0:
            visited = None
    gains: list[float] = []
    weights: list[float] = []
    for modes in modes_by_isotope.values():
        active = [mode for mode in modes if mode.weight > 0.0]
        if len(active) < 2:
            continue
        candidate_angles = [
            _bearing_angle_xy(mode.position_xyz, candidate)
            for mode in active
        ]
        pair_separations: list[float] = []
        for idx, left in enumerate(candidate_angles):
            for right in candidate_angles[idx + 1 :]:
                pair_separations.append(_angle_distance_rad(left, right) / np.pi)
        pair_gain = min(pair_separations) if pair_separations else 0.0
        novelty_gain = 0.0
        if visited is not None:
            novelty_terms: list[float] = []
            for mode, cand_angle in zip(active, candidate_angles):
                prior_angles = [
                    _bearing_angle_xy(mode.position_xyz, pose)
                    for pose in visited
                ]
                if prior_angles:
                    novelty_terms.append(
                        min(
                            _angle_distance_rad(cand_angle, prior_angle)
                            for prior_angle in prior_angles
                        )
                        / np.pi
                    )
            novelty_gain = float(np.mean(novelty_terms)) if novelty_terms else 0.0
        gains.append(0.5 * float(pair_gain) + 0.5 * float(novelty_gain))
        weights.append(float(sum(mode.weight for mode in active)))
    if not gains:
        return 0.0
    weight_arr = _normalise_weights(np.asarray(weights, dtype=float))
    return float(np.sum(weight_arr * np.asarray(gains, dtype=float)))


def _bearing_diversity_gains_batch(
    candidate_poses_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
    modes_by_isotope: dict[str, list[SignatureMode]],
) -> NDArray[np.float64]:
    """Return bearing-diversity gains for many candidate stations."""
    candidates = np.asarray(candidate_poses_xyz, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidate_poses_xyz must be shaped (N, 3).")
    total_gains: list[NDArray[np.float64]] = []
    total_weights: list[float] = []
    visited = _pose_matrix_or_empty(visited_poses_xyz)
    for modes in modes_by_isotope.values():
        active = [mode for mode in modes if float(mode.weight) > 0.0]
        if len(active) < 2:
            continue
        positions = np.vstack([np.asarray(mode.position_xyz, dtype=float) for mode in active])
        deltas = candidates[:, None, :2] - positions[None, :, :2]
        candidate_angles = np.arctan2(deltas[:, :, 1], deltas[:, :, 0])
        left, right = np.triu_indices(len(active), k=1)
        pair_distances = np.abs(
            np.arctan2(
                np.sin(candidate_angles[:, left] - candidate_angles[:, right]),
                np.cos(candidate_angles[:, left] - candidate_angles[:, right]),
            )
        ) / np.pi
        pair_gain = (
            np.min(pair_distances, axis=1)
            if pair_distances.size
            else np.zeros(candidates.shape[0], dtype=float)
        )
        novelty_gain = np.zeros(candidates.shape[0], dtype=float)
        if visited.size:
            prior_deltas = visited[:, None, :2] - positions[None, :, :2]
            prior_angles = np.arctan2(prior_deltas[:, :, 1], prior_deltas[:, :, 0])
            novelty_terms = []
            for mode_idx in range(len(active)):
                distances = np.abs(
                    np.arctan2(
                        np.sin(candidate_angles[:, mode_idx, None] - prior_angles[None, :, mode_idx]),
                        np.cos(candidate_angles[:, mode_idx, None] - prior_angles[None, :, mode_idx]),
                    )
                ) / np.pi
                novelty_terms.append(np.min(distances, axis=1))
            if novelty_terms:
                novelty_gain = np.mean(np.vstack(novelty_terms), axis=0)
        total_gains.append(0.5 * pair_gain + 0.5 * novelty_gain)
        total_weights.append(float(sum(mode.weight for mode in active)))
    if not total_gains:
        return np.zeros(candidates.shape[0], dtype=float)
    weights = _normalise_weights(np.asarray(total_weights, dtype=float))
    stacked = np.vstack(total_gains)
    return np.sum(stacked * weights.reshape(-1, 1), axis=0)


def _frontier_band_gain(
    candidate_pose_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
    *,
    target_radius_m: float,
) -> float:
    """Return a gain for expanding from the current explored frontier."""
    target = max(float(target_radius_m), 1.0e-12)
    if visited_poses_xyz is None:
        return 0.0
    visited = np.asarray(visited_poses_xyz, dtype=float)
    if visited.ndim == 1 and visited.size == 3:
        visited = visited.reshape(1, 3)
    if visited.ndim != 2 or visited.shape[1] != 3 or visited.size == 0:
        return 0.0
    candidate = np.asarray(candidate_pose_xyz, dtype=float).reshape(3)
    nearest = float(np.min(np.linalg.norm(visited[:, :2] - candidate[None, :2], axis=1)))
    return float(np.exp(-((nearest - target) / target) ** 2))


def _frontier_band_gains_batch(
    candidate_poses_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
    *,
    target_radius_m: float,
) -> NDArray[np.float64]:
    """Return frontier-band gains for many candidate stations."""
    candidates = np.asarray(candidate_poses_xyz, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidate_poses_xyz must be shaped (N, 3).")
    target = max(float(target_radius_m), 1.0e-12)
    visited = _pose_matrix_or_empty(visited_poses_xyz)
    if visited.size == 0 or candidates.shape[0] == 0:
        return np.zeros(candidates.shape[0], dtype=float)
    distances = np.linalg.norm(
        candidates[:, None, :2] - visited[None, :, :2],
        axis=2,
    )
    nearest = np.min(distances, axis=1)
    return np.exp(-((nearest - target) / target) ** 2)


def _route_turn_penalty(
    candidate_pose_xyz: NDArray[np.float64],
    current_pose_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
) -> float:
    """Return a normalized penalty for sharp reversals from the previous leg."""
    if visited_poses_xyz is None:
        return 0.0
    visited = np.asarray(visited_poses_xyz, dtype=float)
    if visited.ndim == 1 and visited.size == 3:
        visited = visited.reshape(1, 3)
    if visited.ndim != 2 or visited.shape[1] != 3 or visited.shape[0] < 1:
        return 0.0
    current = np.asarray(current_pose_xyz, dtype=float).reshape(3)
    if (
        visited.shape[0] >= 2
        and float(np.linalg.norm(visited[-1, :2] - current[:2])) < 1.0e-6
    ):
        previous = visited[-2]
    else:
        previous = visited[-1]
    prev_vec = current[:2] - previous[:2]
    next_vec = np.asarray(candidate_pose_xyz, dtype=float).reshape(3)[:2] - current[:2]
    prev_norm = float(np.linalg.norm(prev_vec))
    next_norm = float(np.linalg.norm(next_vec))
    if prev_norm <= 1.0e-9 or next_norm <= 1.0e-9:
        return 0.0
    dot = float(np.clip(np.dot(prev_vec, next_vec) / (prev_norm * next_norm), -1.0, 1.0))
    return float(0.5 * (1.0 - dot))


def _route_turn_penalties_batch(
    candidate_poses_xyz: NDArray[np.float64],
    current_pose_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Return route-turn penalties for many candidate stations."""
    candidates = np.asarray(candidate_poses_xyz, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidate_poses_xyz must be shaped (N, 3).")
    penalties = np.zeros(candidates.shape[0], dtype=float)
    visited = _pose_matrix_or_empty(visited_poses_xyz)
    if visited.shape[0] < 1 or candidates.shape[0] == 0:
        return penalties
    current = np.asarray(current_pose_xyz, dtype=float).reshape(3)
    if (
        visited.shape[0] >= 2
        and float(np.linalg.norm(visited[-1, :2] - current[:2])) < 1.0e-6
    ):
        previous = visited[-2]
    else:
        previous = visited[-1]
    prev_vec = current[:2] - previous[:2]
    prev_norm = float(np.linalg.norm(prev_vec))
    next_vecs = candidates[:, :2] - current[None, :2]
    next_norms = np.linalg.norm(next_vecs, axis=1)
    active = (prev_norm > 1.0e-9) & (next_norms > 1.0e-9)
    if not np.any(active):
        return penalties
    dots = np.sum(next_vecs[active] * prev_vec.reshape(1, 2), axis=1) / (
        prev_norm * next_norms[active]
    )
    penalties[active] = 0.5 * (1.0 - np.clip(dots, -1.0, 1.0))
    return penalties


def _remaining_budget_pressure(config: DSSPPConfig) -> float:
    """Return a 0..1 route-efficiency pressure from remaining station budget."""
    if not bool(config.remaining_budget_guidance):
        return 0.0
    if float(config.remaining_route_weight) <= 0.0:
        return 0.0
    if config.remaining_station_estimate is None:
        return 0.0
    remaining = max(0, int(config.remaining_station_estimate))
    if remaining <= 0:
        return 0.0
    urgency = max(1, int(config.remaining_budget_urgency_stations))
    pressure = (float(urgency) + 1.0 - float(remaining)) / float(urgency)
    return float(np.clip(pressure, 0.0, 1.0))


def _route_regression_penalties_batch(
    candidate_poses_xyz: NDArray[np.float64],
    current_pose_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
    *,
    radius_m: float,
) -> NDArray[np.float64]:
    """Return penalties for moving back near older visited stations."""
    candidates = np.asarray(candidate_poses_xyz, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidate_poses_xyz must be shaped (N, 3).")
    penalties = np.zeros(candidates.shape[0], dtype=float)
    radius = max(float(radius_m), 1.0e-12)
    visited = _pose_matrix_or_empty(visited_poses_xyz)
    if visited.shape[0] <= 1 or candidates.shape[0] == 0:
        return penalties
    current = np.asarray(current_pose_xyz, dtype=float).reshape(3)
    current_dist = np.linalg.norm(visited[:, :2] - current[None, :2], axis=1)
    older = visited[current_dist > max(0.25 * radius, 1.0e-6)]
    if older.size == 0:
        return penalties
    distances = np.linalg.norm(
        candidates[:, None, :2] - older[None, :, :2],
        axis=2,
    )
    nearest = np.min(distances, axis=1)
    active = nearest < radius
    shortfall = 1.0 - nearest[active] / radius
    penalties[active] = shortfall * shortfall
    return penalties


def _remaining_route_terms_batch(
    *,
    candidate_poses_xyz: NDArray[np.float64],
    current_pose_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
    path_lengths: NDArray[np.float64],
    coverage_norm: NDArray[np.float64],
    revisit_penalties: NDArray[np.float64],
    frontier_gains: NDArray[np.float64],
    turn_penalties: NDArray[np.float64],
    config: DSSPPConfig,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Return remaining-budget route pressure, penalties, and gains in batch."""
    candidates = np.asarray(candidate_poses_xyz, dtype=float)
    pressure = _remaining_budget_pressure(config)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidate_poses_xyz must be shaped (N, 3).")
    zeros = np.zeros(candidates.shape[0], dtype=float)
    if pressure <= 0.0 or candidates.shape[0] == 0:
        return pressure, zeros, zeros
    nominal_step = max(
        float(config.min_station_separation_m),
        float(config.coverage_radius_m),
        1.0,
    )
    remaining = max(1, int(config.remaining_station_estimate or 1))
    path_arr = np.asarray(path_lengths, dtype=float)
    distance_penalty = np.divide(
        path_arr,
        nominal_step * float(remaining),
        out=np.zeros_like(path_arr, dtype=float),
        where=np.isfinite(path_arr),
    )
    distance_penalty = np.clip(distance_penalty, 0.0, 2.0)
    coverage_arr = np.clip(np.asarray(coverage_norm, dtype=float), 0.0, 1.0)
    backtrack_penalty = _route_regression_penalties_batch(
        candidates,
        current_pose_xyz,
        visited_poses_xyz,
        radius_m=2.0 * nominal_step,
    )
    penalty = (
        float(config.remaining_route_distance_weight) * distance_penalty
        + float(config.remaining_route_revisit_weight)
        * np.asarray(revisit_penalties, dtype=float)
        + float(config.remaining_route_turn_weight)
        * np.asarray(turn_penalties, dtype=float)
        + float(config.remaining_route_backtrack_weight) * backtrack_penalty
        + float(config.remaining_route_coverage_weight) * (1.0 - coverage_arr)
    )
    gain = (
        float(config.remaining_route_coverage_weight) * coverage_arr
        + float(config.remaining_route_frontier_weight)
        * np.asarray(frontier_gains, dtype=float)
    )
    penalty[~np.isfinite(path_arr)] = np.inf
    return pressure, np.maximum(penalty, 0.0), np.maximum(gain, 0.0)


def _program_evaluation_pose_indices(
    *,
    path_lengths: NDArray[np.float64],
    coverage_norm: NDArray[np.float64],
    revisit_penalties: NDArray[np.float64],
    bearing_gains: NDArray[np.float64],
    frontier_gains: NDArray[np.float64],
    turn_penalties: NDArray[np.float64],
    local_orbit_gains: NDArray[np.float64],
    station_condition_gains: NDArray[np.float64],
    correlation_reduction_gains: NDArray[np.float64],
    isotope_balance_gains: NDArray[np.float64],
    environment_signature_scores: NDArray[np.float64],
    occlusion_boundary_gains: NDArray[np.float64],
    elevation_condition_gains: NDArray[np.float64],
    vertical_environment_signature_scores: NDArray[np.float64],
    cardinality_evidence_pressure: float,
    remaining_route_pressure: float,
    remaining_route_penalties: NDArray[np.float64],
    remaining_route_gains: NDArray[np.float64],
    lambda_distance: float,
    config: DSSPPConfig,
) -> NDArray[np.int64]:
    """Return candidate indices that merit full shield-program evaluation."""
    count = int(path_lengths.size)
    if count == 0:
        return np.zeros(0, dtype=np.int64)
    eig_limit = int(config.eig_candidate_limit or 0)
    multiplier = max(1, int(config.candidate_preselect_multiplier))
    target = max(
        max(1, int(config.candidate_preselect_min)),
        int(config.beam_width) * multiplier,
        eig_limit * multiplier if eig_limit > 0 else 0,
        int(config.max_programs),
    )
    if count <= target:
        return np.arange(count, dtype=np.int64)
    finite = np.isfinite(path_lengths)
    if not np.any(finite):
        return np.zeros(0, dtype=np.int64)
    finite_lengths = path_lengths[finite]
    length_scale = max(float(np.max(finite_lengths)), 1.0e-12)
    cheap_score = (
        float(config.lambda_coverage) * np.asarray(coverage_norm, dtype=float)
        + float(config.lambda_bearing_diversity) * np.asarray(bearing_gains, dtype=float)
        + float(config.lambda_frontier) * np.asarray(frontier_gains, dtype=float)
        + float(config.lambda_local_orbit) * np.asarray(local_orbit_gains, dtype=float)
        + float(config.lambda_station_condition)
        * np.log1p(np.maximum(np.asarray(station_condition_gains, dtype=float), 0.0))
        + float(config.lambda_correlation_reduction)
        * np.log1p(
            np.maximum(np.asarray(correlation_reduction_gains, dtype=float), 0.0)
        )
        + float(config.lambda_cardinality_discrimination)
        * float(max(cardinality_evidence_pressure, 0.0))
        * (
            np.log1p(np.maximum(np.asarray(station_condition_gains, dtype=float), 0.0))
            + np.log1p(
                np.maximum(
                    np.asarray(correlation_reduction_gains, dtype=float),
                    0.0,
                )
            )
            + np.log1p(
                np.maximum(np.asarray(elevation_condition_gains, dtype=float), 0.0)
            )
        )
        + float(config.lambda_isotope_balance)
        * np.asarray(isotope_balance_gains, dtype=float)
        + float(config.lambda_elevation_condition)
        * np.log1p(np.maximum(np.asarray(elevation_condition_gains, dtype=float), 0.0))
        + float(config.lambda_environment_signature)
        * _normalized_environment_signature_score(
            environment_signature_scores,
            config=config,
        )
        + float(config.lambda_vertical_environment_signature)
        * _normalized_environment_signature_score(
            vertical_environment_signature_scores,
            config=config,
        )
        + float(config.lambda_occlusion_boundary)
        * np.log1p(np.maximum(np.asarray(occlusion_boundary_gains, dtype=float), 0.0))
        + float(config.remaining_route_weight)
        * float(remaining_route_pressure)
        * (
            np.asarray(remaining_route_gains, dtype=float)
            - np.asarray(remaining_route_penalties, dtype=float)
        )
        - float(config.eta_revisit) * np.asarray(revisit_penalties, dtype=float)
        - float(config.lambda_turn_smoothness) * np.asarray(turn_penalties, dtype=float)
        - float(lambda_distance) * np.asarray(path_lengths, dtype=float) / length_scale
    )
    cheap_score[~finite] = -np.inf
    selected_count = min(count, target)
    order = np.argsort(cheap_score)[::-1]
    selected = order[:selected_count]
    return np.asarray(selected[np.isfinite(cheap_score[selected])], dtype=np.int64)


def _filter_station_separation(
    candidate_poses_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
    *,
    min_separation_m: float,
) -> tuple[NDArray[np.float64], int]:
    """Remove near-revisited stations when at least one unvisited option exists."""
    min_sep = max(float(min_separation_m), 0.0)
    candidates = np.asarray(candidate_poses_xyz, dtype=float)
    if candidates.size == 0 or min_sep <= 0.0 or visited_poses_xyz is None:
        return candidates, 0
    visited = np.asarray(visited_poses_xyz, dtype=float)
    if visited.ndim == 1 and visited.size == 3:
        visited = visited.reshape(1, 3)
    if visited.ndim != 2 or visited.shape[1] != 3 or visited.size == 0:
        return candidates, 0
    distances = np.linalg.norm(
        candidates[:, None, :2] - visited[None, :, :2],
        axis=2,
    )
    keep = np.min(distances, axis=1) >= min_sep
    if not np.any(keep):
        return candidates, 0
    removed = int(np.count_nonzero(~keep))
    return candidates[keep], removed


def _shield_transition_cost(
    normals: NDArray[np.float64],
    from_pair_id: int | None,
    program: ShieldProgram,
) -> float:
    """Return angular shield-transition cost for a program."""
    if not program.pair_ids:
        return 0.0
    normal_arr = np.asarray(normals, dtype=float)
    num_orients = int(normal_arr.shape[0])
    sequence: list[int] = []
    if from_pair_id is not None and int(from_pair_id) >= 0:
        sequence.append(int(from_pair_id))
    sequence.extend(int(pair_id) for pair_id in program.pair_ids)
    if len(sequence) < 2:
        return 0.0
    cost = 0.0
    for prev_id, next_id in zip(sequence[:-1], sequence[1:]):
        prev_fe, prev_pb = _pair_indices(prev_id, num_orients)
        next_fe, next_pb = _pair_indices(next_id, num_orients)
        for prev_idx, next_idx in ((prev_fe, next_fe), (prev_pb, next_pb)):
            dot = float(
                np.clip(
                    np.dot(normal_arr[prev_idx], normal_arr[next_idx]),
                    -1.0,
                    1.0,
                )
            )
            cost += float(np.arccos(dot))
    return cost


def _compose_transition_score(
    *,
    node: DSSPPNode,
    previous_pose_xyz: NDArray[np.float64],
    previous_pair_id: int | None,
    estimator: RotatingShieldPFEstimator,
    map_api: object | None,
    config: DSSPPConfig,
) -> tuple[float, float]:
    """Return node score and path length for a specific predecessor."""
    path_length = _node_path_length(map_api, previous_pose_xyz, node.pose_xyz)
    if not np.isfinite(path_length):
        return -float("inf"), float("inf")
    travel_time = path_length / max(float(config.robot_speed_m_s), 1e-9)
    time_cost = (
        travel_time
        + len(node.program.pair_ids)
        * (float(config.rotation_overhead_s) + float(config.live_time_s))
    )
    rotation_cost = _shield_transition_cost(
        estimator.normals,
        previous_pair_id,
        node.program,
    )
    score = (
        float(node.static_score)
        - float(node.distance_weight) * float(path_length)
        - float(config.lambda_time) * float(time_cost)
        - float(config.lambda_rotation) * float(rotation_cost)
        - float(node.observation_penalty_weight) * float(node.observation_penalty)
    )
    return float(score), float(path_length)


def _candidate_information_gain(
    estimator: RotatingShieldPFEstimator,
    pose_xyz: NDArray[np.float64],
    *,
    config: DSSPPConfig,
    rng_seed: int | None,
) -> float:
    """Return candidate-level predicted log-uncertainty reduction."""
    if float(config.lambda_eig) <= 0.0:
        return 0.0
    try:
        cached_uncertainty = getattr(
            estimator,
            "_dss_pp_current_uncertainty_cache",
            None,
        )
        if cached_uncertainty is None:
            current_uncertainty = max(float(estimator.global_uncertainty()), 0.0)
        else:
            current_uncertainty = max(float(cached_uncertainty), 0.0)
        after_uncertainty = float(
            estimator.expected_uncertainty_after_rotation(
                pose_xyz=pose_xyz,
                live_time_per_rot_s=float(config.live_time_s),
                tau_ig=float(estimator.pf_config.ig_threshold),
                tmax_s=float(config.live_time_s)
                * max(1, int(config.program_length)),
                n_rollouts=0,
                orient_selection="IG",
                rng_seed=rng_seed,
            )
        )
    except RuntimeError:
        return 0.0
    current_information = float(np.log1p(current_uncertainty))
    after_information = float(np.log1p(max(after_uncertainty, 0.0)))
    return max(current_information - after_information, 0.0)


def _build_nodes(
    *,
    estimator: RotatingShieldPFEstimator,
    candidate_poses_xyz: NDArray[np.float64],
    programs: Sequence[ShieldProgram],
    modes_by_isotope: dict[str, list[SignatureMode]],
    current_pose_xyz: NDArray[np.float64],
    current_pair_id: int | None,
    visited_poses_xyz: NDArray[np.float64] | None,
    map_api: object | None,
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]] | None,
    config: DSSPPConfig,
) -> list[DSSPPNode]:
    """Evaluate all station-program nodes for the first horizon layer."""
    kernel = _continuous_kernel_for_estimator(
        estimator,
        detector_aperture_samples=max(1, int(config.detector_aperture_samples)),
    )
    candidate_poses = np.asarray(candidate_poses_xyz, dtype=float)
    if candidate_poses.ndim != 2 or candidate_poses.shape[1] != 3:
        raise ValueError("candidate_poses_xyz must be shape (N, 3).")
    info_gains = np.zeros(candidate_poses.shape[0], dtype=float)
    path_lengths = np.asarray(
        [
            _node_path_length(map_api, current_pose_xyz, pose)
            for pose in candidate_poses
        ],
        dtype=float,
    )
    free_cell_centers = _free_cell_centers(
        map_api,
        z_value=float(current_pose_xyz[2]),
        max_cells=int(config.coverage_grid_max_cells),
        bounds_xyz=bounds_xyz,
    )
    coverage_raw = _coverage_gain_fractions_batch(
        cell_centers_xyz=free_cell_centers,
        candidate_poses_xyz=candidate_poses,
        visited_poses_xyz=visited_poses_xyz,
        radius_m=float(config.coverage_radius_m),
    )
    coverage_norm = coverage_raw.copy()
    max_coverage = float(np.max(coverage_norm)) if coverage_norm.size else 0.0
    if max_coverage > 0.0:
        coverage_norm = coverage_norm / max_coverage
    coverage_floor = 0.0
    coverage_floor_quantile = float(config.coverage_floor_quantile)
    if (
        coverage_norm.size
        and float(config.coverage_floor_weight) > 0.0
        and coverage_floor_quantile > 0.0
    ):
        positive_coverage = coverage_norm[coverage_norm > 0.0]
        if positive_coverage.size:
            coverage_floor = float(
                np.quantile(
                    positive_coverage,
                    np.clip(coverage_floor_quantile, 0.0, 1.0),
                )
            )
    revisit_penalties = _station_revisit_penalties_batch(
        candidate_poses,
        visited_poses_xyz,
        min_separation_m=float(config.min_station_separation_m),
    )
    bearing_gains = _bearing_diversity_gains_batch(
        candidate_poses,
        visited_poses_xyz,
        modes_by_isotope,
    )
    frontier_target = max(
        float(config.min_station_separation_m),
        float(config.coverage_radius_m),
    )
    frontier_gains = _frontier_band_gains_batch(
        candidate_poses,
        visited_poses_xyz,
        target_radius_m=frontier_target,
    )
    turn_penalties = _route_turn_penalties_batch(
        candidate_poses,
        current_pose_xyz,
        visited_poses_xyz,
    )
    local_orbit_gains = _local_orbit_gains_batch(
        candidate_poses,
        modes_by_isotope,
        config=config,
    )
    station_condition_gains = _station_condition_gains_batch(
        candidate_poses,
        visited_poses_xyz,
        modes_by_isotope,
        config=config,
        kernel=kernel,
        estimator=estimator,
    )
    correlation_reduction_gains = _station_correlation_reduction_gains_batch(
        candidate_poses,
        visited_poses_xyz,
        modes_by_isotope,
        config=config,
        kernel=kernel,
        estimator=estimator,
    )
    isotope_balance_gains = _isotope_balance_gains_batch(
        candidate_poses,
        modes_by_isotope,
        config=config,
        kernel=kernel,
        estimator=estimator,
    )
    elevation_condition_gains = _elevation_condition_gains_batch(
        candidate_poses,
        modes_by_isotope,
        config=config,
    )
    environment_signature_scores = _environment_signature_scores_batch(
        kernel=kernel,
        estimator=estimator,
        modes_by_isotope=modes_by_isotope,
        poses_xyz=candidate_poses,
        config=config,
    )
    vertical_environment_signature_scores = (
        _vertical_environment_signature_scores_batch(
            kernel=kernel,
            estimator=estimator,
            modes_by_isotope=modes_by_isotope,
            poses_xyz=candidate_poses,
            config=config,
        )
    )
    occlusion_boundary_gains = _occlusion_boundary_gains_batch(
        kernel=kernel,
        estimator=estimator,
        modes_by_isotope=modes_by_isotope,
        poses_xyz=candidate_poses,
        config=config,
    )
    cardinality_evidence_pressure = _cardinality_evidence_gap_pressure(
        estimator,
        config,
    )
    (
        remaining_route_pressure,
        remaining_route_penalties,
        remaining_route_gains,
    ) = _remaining_route_terms_batch(
        candidate_poses_xyz=candidate_poses,
        current_pose_xyz=current_pose_xyz,
        visited_poses_xyz=visited_poses_xyz,
        path_lengths=path_lengths,
        coverage_norm=coverage_norm,
        revisit_penalties=revisit_penalties,
        frontier_gains=frontier_gains,
        turn_penalties=turn_penalties,
        config=config,
    )
    finite_path = np.isfinite(path_lengths)
    if config.lambda_distance is None:
        lambda_distance = estimate_lambda_cost(
            -info_gains,
            path_lengths[finite_path] if np.any(finite_path) else path_lengths,
            method="range",
        )
    else:
        lambda_distance = float(config.lambda_distance)
    if bool(config.candidate_preselect_enable):
        selected_indices = _program_evaluation_pose_indices(
            path_lengths=path_lengths,
            coverage_norm=coverage_norm,
            revisit_penalties=revisit_penalties,
            bearing_gains=bearing_gains,
            frontier_gains=frontier_gains,
            turn_penalties=turn_penalties,
            local_orbit_gains=local_orbit_gains,
            station_condition_gains=station_condition_gains,
            correlation_reduction_gains=correlation_reduction_gains,
            isotope_balance_gains=isotope_balance_gains,
            environment_signature_scores=environment_signature_scores,
            occlusion_boundary_gains=occlusion_boundary_gains,
            elevation_condition_gains=elevation_condition_gains,
            vertical_environment_signature_scores=vertical_environment_signature_scores,
            cardinality_evidence_pressure=cardinality_evidence_pressure,
            remaining_route_pressure=remaining_route_pressure,
            remaining_route_penalties=remaining_route_penalties,
            remaining_route_gains=remaining_route_gains,
            lambda_distance=lambda_distance,
            config=config,
        )
        if selected_indices.size:
            candidate_poses = candidate_poses[selected_indices]
            info_gains = info_gains[selected_indices]
            path_lengths = path_lengths[selected_indices]
            coverage_raw = coverage_raw[selected_indices]
            coverage_norm = coverage_norm[selected_indices]
            revisit_penalties = revisit_penalties[selected_indices]
            bearing_gains = bearing_gains[selected_indices]
            frontier_gains = frontier_gains[selected_indices]
            turn_penalties = turn_penalties[selected_indices]
            local_orbit_gains = local_orbit_gains[selected_indices]
            station_condition_gains = station_condition_gains[selected_indices]
            correlation_reduction_gains = correlation_reduction_gains[selected_indices]
            isotope_balance_gains = isotope_balance_gains[selected_indices]
            elevation_condition_gains = elevation_condition_gains[selected_indices]
            environment_signature_scores = environment_signature_scores[selected_indices]
            vertical_environment_signature_scores = (
                vertical_environment_signature_scores[selected_indices]
            )
            occlusion_boundary_gains = occlusion_boundary_gains[selected_indices]
            remaining_route_penalties = remaining_route_penalties[selected_indices]
            remaining_route_gains = remaining_route_gains[selected_indices]
    evaluation_pose_indices = np.arange(candidate_poses.shape[0], dtype=np.int64)
    raw_nodes: list[DSSPPNode] = []
    static_scores: list[float] = []
    observation_penalties: list[float] = []
    cheap_pose_scores = np.full(candidate_poses.shape[0], -np.inf, dtype=float)
    pending: list[tuple[Any, ...]] = []
    pair_caches_by_pose = _build_pair_signature_caches_for_poses(
        kernel=kernel,
        estimator=estimator,
        modes_by_isotope=modes_by_isotope,
        poses_xyz=candidate_poses,
        config=config,
    )
    eval_indices = [int(idx) for idx in evaluation_pose_indices]
    worker_cfg = config.program_eval_workers
    if worker_cfg is None:
        worker_count = min(len(eval_indices), max(1, os.cpu_count() or 1))
    else:
        worker_count = min(len(eval_indices), max(1, int(worker_cfg)))
    pose_eval_context: dict[str, object] = {
        "candidate_poses": candidate_poses,
        "path_lengths": path_lengths,
        "pair_caches_by_pose": pair_caches_by_pose,
        "programs": programs,
        "estimator": estimator,
        "kernel": kernel,
        "modes_by_isotope": modes_by_isotope,
        "config": config,
        "coverage_norm": coverage_norm,
        "coverage_raw": coverage_raw,
        "revisit_penalties": revisit_penalties,
        "bearing_gains": bearing_gains,
        "frontier_gains": frontier_gains,
        "turn_penalties": turn_penalties,
        "local_orbit_gains": local_orbit_gains,
        "station_condition_gains": station_condition_gains,
        "correlation_reduction_gains": correlation_reduction_gains,
        "isotope_balance_gains": isotope_balance_gains,
        "elevation_condition_gains": elevation_condition_gains,
        "environment_signature_scores": environment_signature_scores,
        "vertical_environment_signature_scores": vertical_environment_signature_scores,
        "occlusion_boundary_gains": occlusion_boundary_gains,
        "cardinality_evidence_pressure": float(cardinality_evidence_pressure),
        "remaining_route_pressure": float(remaining_route_pressure),
        "remaining_route_penalties": remaining_route_penalties,
        "remaining_route_gains": remaining_route_gains,
        "coverage_floor": float(coverage_floor),
    }
    pose_results = _evaluate_pose_indices_parallel(
        eval_indices,
        context=pose_eval_context,
        worker_count=worker_count,
    )
    for (
        pose_index,
        local_cheap_score,
        local_pending,
        local_scores,
        local_observation_penalties,
    ) in pose_results:
        if local_pending:
            cheap_pose_scores[int(pose_index)] = max(
                float(cheap_pose_scores[int(pose_index)]),
                float(local_cheap_score),
            )
            pending.extend(local_pending)
            static_scores.extend(local_scores)
            observation_penalties.extend(local_observation_penalties)
    if not pending:
        return []
    if float(config.lambda_eig) > 0.0:
        valid_pose_indices = np.flatnonzero(
            np.isfinite(cheap_pose_scores) & np.isfinite(path_lengths)
        )
        eig_limit = config.eig_candidate_limit
        if eig_limit is None or int(eig_limit) <= 0:
            selected_pose_indices = valid_pose_indices
        else:
            limit = min(int(eig_limit), int(valid_pose_indices.size))
            order = np.argsort(cheap_pose_scores[valid_pose_indices])[::-1]
            selected_pose_indices = valid_pose_indices[order[:limit]]

        def _candidate_ig_for_index(pose_index_value: int) -> tuple[int, float]:
            """Return the candidate EIG for one station index."""
            pose_index = int(pose_index_value)
            value = _candidate_information_gain(
                estimator,
                candidate_poses[int(pose_index)],
                config=config,
                rng_seed=None
                if config.rng_seed is None
                else int(config.rng_seed) + int(pose_index),
            )
            return pose_index, float(value)

        eig_indices = [int(index) for index in selected_pose_indices]
        eig_workers = min(
            len(eig_indices),
            max(1, int(worker_count)),
        )
        if eig_indices:
            try:
                cached_current_uncertainty = max(
                    float(estimator.global_uncertainty()),
                    0.0,
                )
            except RuntimeError:
                cached_current_uncertainty = None
            if cached_current_uncertainty is not None:
                setattr(
                    estimator,
                    "_dss_pp_current_uncertainty_cache",
                    cached_current_uncertainty,
                )
        if getattr(estimator, "_can_use_gpu", lambda: False)():
            # Candidate EIG calls use the same CUDA expected-count kernel.
            # Parallel CPU workers would oversubscribe one GPU and can exhaust
            # memory in obstacle-rich scenes without changing the math.
            eig_workers = 1
        try:
            if eig_workers <= 1 or len(eig_indices) <= 1:
                eig_results = [_candidate_ig_for_index(index) for index in eig_indices]
            else:
                with ThreadPoolExecutor(max_workers=eig_workers) as executor:
                    eig_results = list(
                        executor.map(_candidate_ig_for_index, eig_indices)
                    )
        finally:
            if hasattr(estimator, "_dss_pp_current_uncertainty_cache"):
                delattr(estimator, "_dss_pp_current_uncertainty_cache")
        for pose_index, value in eig_results:
            info_gains[int(pose_index)] = float(value)
        static_scores = [
            float(score) + float(config.lambda_eig) * float(info_gains[item[0]])
            for score, item in zip(static_scores, pending)
        ]
    obs_arr = np.asarray(observation_penalties, dtype=float)
    feasible_mask = _minimum_observation_feasible_mask(
        obs_arr,
        float(config.min_observation_counts),
    )
    if bool(config.enforce_min_observation) and np.any(feasible_mask):
        keep_indices = [
            idx for idx, feasible in enumerate(feasible_mask) if bool(feasible)
        ]
        pending = [pending[idx] for idx in keep_indices]
        static_scores = [static_scores[idx] for idx in keep_indices]
        observation_penalties = [
            observation_penalties[idx] for idx in keep_indices
        ]
        obs_arr = np.asarray(observation_penalties, dtype=float)
    obs_weight = _auto_scale_observation_penalty(
        np.asarray(static_scores, dtype=float),
        obs_arr,
        scale=float(config.eta_observation),
    )
    for item, base_score, observation_penalty in zip(
        pending,
        static_scores,
        observation_penalties,
    ):
        (
            pose_index,
            pose,
            program,
            _info_gain,
            signature_score,
            temporal_separation_score,
            elevation_signature_score,
            _obs_penalty,
            count_balance_penalty,
            differential_penalty,
            dose_score,
            count_utility,
            coverage_gain,
            revisit_penalty,
            bearing_gain,
            frontier_gain,
            turn_penalty,
            local_orbit_gain,
            station_condition_gain,
            correlation_reduction_gain,
            cardinality_gap_gain,
            isotope_balance_gain,
            elevation_condition_gain,
            environment_signature_score,
            vertical_environment_signature_score,
            occlusion_boundary_gain,
            remaining_route_pressure,
            remaining_route_penalty,
            remaining_route_gain,
        ) = item
        info_gain = float(info_gains[pose_index])
        placeholder_node = DSSPPNode(
            pose_index=int(pose_index),
            pose_xyz=pose,
            program=program,
            score=0.0,
            static_score=float(base_score),
            distance_weight=float(lambda_distance),
            observation_penalty_weight=float(obs_weight),
            information_gain=float(info_gain),
            signature_score=float(signature_score),
            temporal_separation_score=float(temporal_separation_score),
            elevation_signature_score=float(elevation_signature_score),
            observation_penalty=float(observation_penalty),
            count_balance_penalty=float(count_balance_penalty),
            differential_penalty=float(differential_penalty),
            dose_score=float(dose_score),
            count_utility=float(count_utility),
            coverage_gain=float(coverage_gain),
            revisit_penalty=float(revisit_penalty),
            bearing_diversity_gain=float(bearing_gain),
            frontier_gain=float(frontier_gain),
            turn_penalty=float(turn_penalty),
            local_orbit_gain=float(local_orbit_gain),
            station_condition_gain=float(station_condition_gain),
            correlation_reduction_gain=float(correlation_reduction_gain),
            cardinality_gap_gain=float(cardinality_gap_gain),
            isotope_balance_gain=float(isotope_balance_gain),
            elevation_condition_gain=float(elevation_condition_gain),
            environment_signature_score=float(environment_signature_score),
            vertical_environment_signature_score=float(
                vertical_environment_signature_score
            ),
            occlusion_boundary_gain=float(occlusion_boundary_gain),
            remaining_route_pressure=float(remaining_route_pressure),
            remaining_route_penalty=float(remaining_route_penalty),
            remaining_route_gain=float(remaining_route_gain),
        )
        score, _ = _compose_transition_score(
            node=placeholder_node,
            previous_pose_xyz=current_pose_xyz,
            previous_pair_id=current_pair_id,
            estimator=estimator,
            map_api=map_api,
            config=config,
        )
        raw_nodes.append(
            DSSPPNode(
                pose_index=int(pose_index),
                pose_xyz=pose,
                program=program,
                score=score,
                static_score=float(base_score),
                distance_weight=float(lambda_distance),
                observation_penalty_weight=float(obs_weight),
                information_gain=float(info_gain),
                signature_score=float(signature_score),
                temporal_separation_score=float(temporal_separation_score),
                elevation_signature_score=float(elevation_signature_score),
                observation_penalty=float(observation_penalty),
                count_balance_penalty=float(count_balance_penalty),
                differential_penalty=float(differential_penalty),
                dose_score=float(dose_score),
                count_utility=float(count_utility),
                coverage_gain=float(coverage_gain),
                revisit_penalty=float(revisit_penalty),
                bearing_diversity_gain=float(bearing_gain),
                frontier_gain=float(frontier_gain),
                turn_penalty=float(turn_penalty),
                local_orbit_gain=float(local_orbit_gain),
                station_condition_gain=float(station_condition_gain),
                correlation_reduction_gain=float(correlation_reduction_gain),
                cardinality_gap_gain=float(cardinality_gap_gain),
                isotope_balance_gain=float(isotope_balance_gain),
                elevation_condition_gain=float(elevation_condition_gain),
                environment_signature_score=float(environment_signature_score),
                vertical_environment_signature_score=float(
                    vertical_environment_signature_score
                ),
                occlusion_boundary_gain=float(occlusion_boundary_gain),
                remaining_route_pressure=float(remaining_route_pressure),
                remaining_route_penalty=float(remaining_route_penalty),
                remaining_route_gain=float(remaining_route_gain),
            )
        )
    raw_nodes.sort(key=lambda node: node.score, reverse=True)
    return raw_nodes


def _filter_nodes_for_multimode_separation(
    nodes: Sequence[DSSPPNode],
    modes_by_isotope: dict[str, list[SignatureMode]],
    *,
    config: DSSPPConfig | None = None,
    unresolved_evidence: bool = False,
    epsilon: float = 1.0e-12,
) -> list[DSSPPNode]:
    """
    Prefer station-program nodes that can separate multiple same-isotope modes.

    If at least one isotope has multiple posterior modes or unresolved residual
    evidence, a zero-signature station is not evidence that those hypotheses are
    false.  It is simply an uninformative observation for source-cardinality
    decisions.  The first preference is a direct shield/temporal/elevation
    signature because that is the evidence used to split same-isotope sources.
    Obstacle and station condition scores are retained as a fallback when no
    direct separating node exists.
    """
    node_list = list(nodes)
    has_multi_mode = any(len(mode_list) >= 2 for mode_list in modes_by_isotope.values())
    if not has_multi_mode and not bool(unresolved_evidence):
        return node_list
    cfg = config or DSSPPConfig()
    if not bool(cfg.same_isotope_direct_separation_guard):
        return node_list
    threshold = max(
        float(epsilon),
        float(cfg.same_isotope_direct_separation_epsilon),
        0.0,
    )
    direct_positive: list[DSSPPNode] = []
    fallback_positive: list[DSSPPNode] = []
    for node in node_list:
        direct_separable = (
            float(node.signature_score) > threshold
            or float(node.temporal_separation_score) > threshold
            or float(node.elevation_signature_score) > threshold
        )
        if direct_separable:
            direct_positive.append(node)
            continue
        fallback_separable = (
            float(node.station_condition_gain) > threshold
            or float(node.elevation_condition_gain) > threshold
            or float(node.environment_signature_score) > threshold
            or float(node.vertical_environment_signature_score) > threshold
        )
        if fallback_separable:
            fallback_positive.append(node)
    if direct_positive:
        return direct_positive
    return fallback_positive if fallback_positive else node_list


def _has_unresolved_planning_evidence(
    estimator: RotatingShieldPFEstimator,
) -> bool:
    """Return True when PF diagnostics still request discriminative views."""
    for name in ("unresolved_structural_evidence", "unresolved_isotope_evidence"):
        getter = getattr(estimator, name, None)
        if not callable(getter):
            continue
        try:
            payload = getter()
        except (RuntimeError, ValueError, TypeError):
            continue
        if isinstance(payload, dict) and bool(payload):
            return True
    return False


def _apply_recovery_isotope_mode_weights(
    modes_by_isotope: dict[str, list[SignatureMode]],
    config: DSSPPConfig,
) -> dict[str, list[SignatureMode]]:
    """Boost modes for isotopes currently blocking remaining-measurement progress."""
    recovery = {str(value) for value in tuple(config.recovery_isotopes)}
    multiplier = max(1.0, float(config.recovery_isotope_mode_weight_multiplier))
    if not recovery or multiplier <= 1.0:
        return modes_by_isotope
    boosted: dict[str, list[SignatureMode]] = {}
    for isotope, modes in modes_by_isotope.items():
        if str(isotope) not in recovery:
            boosted[isotope] = list(modes)
            continue
        boosted[isotope] = [
            replace(mode, weight=float(mode.weight) * multiplier)
            for mode in modes
        ]
    return boosted


def _beam_search_sequence(
    nodes: Sequence[DSSPPNode],
    *,
    current_pose_xyz: NDArray[np.float64],
    current_pair_id: int | None,
    estimator: RotatingShieldPFEstimator,
    map_api: object | None,
    config: DSSPPConfig,
) -> tuple[DSSPPNode, ...]:
    """Return a receding-horizon node sequence using beam search."""
    if not nodes:
        return tuple()
    horizon = max(1, int(config.horizon))
    beam_width = max(1, int(config.beam_width))
    beam: list[tuple[float, tuple[DSSPPNode, ...], NDArray[np.float64], int | None]] = [
        (0.0, tuple(), np.asarray(current_pose_xyz, dtype=float), current_pair_id)
    ]
    expansion_nodes = list(nodes[: max(beam_width * 4, beam_width)])
    for _depth in range(horizon):
        next_beam: list[
            tuple[float, tuple[DSSPPNode, ...], NDArray[np.float64], int | None]
        ] = []
        for cumulative, sequence, prev_pose, prev_pair in beam:
            for node in expansion_nodes:
                transition_score, path_len = _compose_transition_score(
                    node=node,
                    previous_pose_xyz=prev_pose,
                    previous_pair_id=prev_pair,
                    estimator=estimator,
                    map_api=map_api,
                    config=config,
                )
                if not np.isfinite(path_len):
                    continue
                score = cumulative + transition_score
                next_pair = node.program.pair_ids[-1] if node.program.pair_ids else prev_pair
                next_beam.append(
                    (
                        score,
                        sequence + (node,),
                        node.pose_xyz,
                        int(next_pair) if next_pair is not None else None,
                    )
                )
        if not next_beam:
            break
        next_beam.sort(key=lambda item: item[0], reverse=True)
        beam = next_beam[:beam_width]
    if not beam:
        return tuple(nodes[:1])
    return beam[0][1]


def select_dss_pp_next_station(
    estimator: RotatingShieldPFEstimator,
    candidate_poses_xyz: NDArray[np.float64],
    current_pose_xyz: NDArray[np.float64],
    *,
    current_pair_id: int | None = None,
    visited_poses_xyz: NDArray[np.float64] | None = None,
    map_api: object | None = None,
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    config: DSSPPConfig | None = None,
) -> DSSPPResult:
    """Select the next station and shield program using DSS-PP."""
    cfg = config or DSSPPConfig()
    current_pose = np.asarray(current_pose_xyz, dtype=float)
    if current_pose.shape != (3,):
        raise ValueError("current_pose_xyz must be shape (3,).")
    modes = extract_signature_modes(
        estimator,
        max_particles=cfg.planning_particles,
        method=cfg.planning_method,
        mode_cluster_radius_m=float(cfg.mode_cluster_radius_m),
        max_modes_per_isotope=int(cfg.max_modes_per_isotope),
        tentative_weight_multiplier=1.0 + max(float(cfg.residual_signature_weight), 0.0),
        include_runtime_rescue_modes=bool(cfg.include_runtime_rescue_modes),
        runtime_rescue_mode_weight=float(cfg.runtime_rescue_mode_weight),
        include_global_surface_rescue_modes=bool(
            cfg.include_global_surface_rescue_modes
        ),
        global_surface_rescue_mode_weight=float(
            cfg.global_surface_rescue_mode_weight
        ),
        weak_mode_weight_floor=float(cfg.weak_mode_weight_floor),
        dominant_mode_weight_cap=float(cfg.dominant_mode_weight_cap),
    )
    modes = _apply_recovery_isotope_mode_weights(modes, cfg)
    candidates = np.asarray(candidate_poses_xyz, dtype=float)
    base_candidates = candidates.copy()
    if cfg.augment_candidates:
        candidates = augment_candidate_stations(
            candidates,
            modes_by_isotope=modes,
            current_pose_xyz=current_pose,
            visited_poses_xyz=visited_poses_xyz,
            map_api=map_api,
            bounds_xyz=bounds_xyz,
            config=cfg,
        )
    candidates, separation_filtered = _filter_station_separation(
        candidates,
        visited_poses_xyz,
        min_separation_m=float(cfg.min_station_separation_m),
    )
    candidates, path_filtered = _filter_path_reachable_stations(
        candidates,
        current_pose_xyz=current_pose,
        map_api=map_api,
    )
    fallback_used = False
    if candidates.size == 0 and base_candidates.size != 0:
        candidates, base_path_filtered = _filter_path_reachable_stations(
            base_candidates,
            current_pose_xyz=current_pose,
            map_api=map_api,
        )
        path_filtered += int(base_path_filtered)
        fallback_used = candidates.size != 0
    if cfg.forced_program_pair_ids is None:
        programs = build_shield_program_library(
            estimator.normals,
            program_length=int(cfg.program_length),
            max_programs=int(cfg.max_programs),
        )
    else:
        programs = [
            ShieldProgram(
                name="forced_baseline_shield_program",
                pair_ids=tuple(
                    int(pair_id) for pair_id in cfg.forced_program_pair_ids
                ),
                kind="forced_baseline",
            )
        ]
    nodes = _build_nodes(
        estimator=estimator,
        candidate_poses_xyz=candidates,
        programs=programs,
        modes_by_isotope=modes,
        current_pose_xyz=current_pose,
        current_pair_id=current_pair_id,
        visited_poses_xyz=visited_poses_xyz,
        map_api=map_api,
        bounds_xyz=bounds_xyz,
        config=cfg,
    )
    if not nodes:
        raise ValueError("DSS-PP could not evaluate any station-program node.")
    original_node_count = int(len(nodes))
    unresolved_planning_evidence = _has_unresolved_planning_evidence(estimator)
    nodes = _filter_nodes_for_multimode_separation(
        nodes,
        modes,
        config=cfg,
        unresolved_evidence=unresolved_planning_evidence,
    )
    sequence = _beam_search_sequence(
        nodes,
        current_pose_xyz=current_pose,
        current_pair_id=current_pair_id,
        estimator=estimator,
        map_api=map_api,
        config=cfg,
    )
    if not sequence:
        sequence = (nodes[0],)
    first = sequence[0]
    best_score = float(sum(node.score for node in sequence))
    diagnostic_kernel = _continuous_kernel_for_estimator(
        estimator,
        detector_aperture_samples=max(1, int(cfg.detector_aperture_samples)),
    )
    selected_pairwise_ambiguity = _program_pairwise_ambiguity_diagnostics(
        estimator=estimator,
        kernel=diagnostic_kernel,
        modes_by_isotope=modes,
        pose_xyz=first.pose_xyz,
        program=first.program,
        config=cfg,
    )
    mode_count = sum(len(mode_list) for mode_list in modes.values())
    runtime_rescue_mode_counts: dict[str, int] = {}
    if bool(cfg.include_runtime_rescue_modes):
        rescue_getter = getattr(estimator, "runtime_report_rescue_modes", None)
        if callable(rescue_getter):
            try:
                for isotope, (positions, _strengths, _weight) in dict(
                    rescue_getter()
                ).items():
                    runtime_rescue_mode_counts[str(isotope)] = int(
                        np.asarray(positions, dtype=float).reshape(-1, 3).shape[0]
                    )
            except (RuntimeError, ValueError, TypeError):
                runtime_rescue_mode_counts = {}
    raw_global_counts = getattr(
        estimator,
        "_last_planning_surface_rescue_mode_counts",
        {},
    )
    global_surface_rescue_mode_counts = (
        {str(key): int(value) for key, value in raw_global_counts.items()}
        if isinstance(raw_global_counts, dict)
        else {}
    )
    configured_workers = cfg.program_eval_workers
    program_eval_workers = (
        min(int(candidates.shape[0]), max(1, os.cpu_count() or 1))
        if configured_workers is None
        else min(int(candidates.shape[0]), max(1, int(configured_workers)))
    )
    ranked_limit = int(cfg.diagnostic_ranked_node_limit)
    ranked_nodes = (
        sorted(nodes, key=lambda node: float(node.score), reverse=True)[:ranked_limit]
        if ranked_limit > 0
        else []
    )
    diagnostics: dict[str, Any] = {
        "candidate_count": int(candidates.shape[0]),
        "separation_filtered_candidates": int(separation_filtered),
        "path_filtered_candidates": int(path_filtered),
        "path_fallback_used": int(fallback_used),
        "program_count": int(len(programs)),
        "evaluated_candidate_count": int(
            len({int(node.pose_index) for node in nodes})
        ),
        "node_count": int(len(nodes)),
        "separation_guard_filtered_nodes": int(original_node_count - len(nodes)),
        "mode_count": int(mode_count),
        "runtime_rescue_mode_counts": runtime_rescue_mode_counts,
        "runtime_rescue_mode_weight": float(cfg.runtime_rescue_mode_weight),
        "global_surface_rescue_mode_counts": global_surface_rescue_mode_counts,
        "global_surface_rescue_mode_weight": float(
            cfg.global_surface_rescue_mode_weight
        ),
        "recovery_isotopes": [str(value) for value in cfg.recovery_isotopes],
        "recovery_isotope_mode_weight_multiplier": float(
            cfg.recovery_isotope_mode_weight_multiplier
        ),
        "explicit_mode_switch": bool(cfg.explicit_mode_switch),
        "planner_mode": str(cfg.planner_mode),
        "station_condition_min_singular_weight": float(
            cfg.station_condition_min_singular_weight
        ),
        "station_condition_inverse_condition_weight": float(
            cfg.station_condition_inverse_condition_weight
        ),
        "station_condition_coherence_weight": float(
            cfg.station_condition_coherence_weight
        ),
        "lambda_cardinality_discrimination": float(
            cfg.lambda_cardinality_discrimination
        ),
        "cardinality_evidence_gap_target": float(
            cfg.cardinality_evidence_gap_target
        ),
        "cardinality_bic_parameter_count_per_source": int(
            cfg.cardinality_bic_parameter_count_per_source
        ),
        "cardinality_evidence_pressure": float(
            _cardinality_evidence_gap_pressure(estimator, cfg)
        ),
        "unresolved_planning_evidence": bool(unresolved_planning_evidence),
        "program_eval_workers": int(program_eval_workers),
        "horizon": int(max(1, cfg.horizon)),
        "beam_width": int(max(1, cfg.beam_width)),
        "first_program_kind": first.program.kind,
        "first_information_gain": float(first.information_gain),
        "first_signature_score": float(first.signature_score),
        "first_temporal_separation_score": float(first.temporal_separation_score),
        "first_elevation_signature_score": float(first.elevation_signature_score),
        "first_observation_penalty": float(first.observation_penalty),
        "first_count_balance_penalty": float(first.count_balance_penalty),
        "first_differential_penalty": float(first.differential_penalty),
        "first_dose_score": float(first.dose_score),
        "first_count_utility": float(first.count_utility),
        "first_coverage_gain": float(first.coverage_gain),
        "first_revisit_penalty": float(first.revisit_penalty),
        "first_bearing_diversity_gain": float(first.bearing_diversity_gain),
        "first_frontier_gain": float(first.frontier_gain),
        "first_turn_penalty": float(first.turn_penalty),
        "first_local_orbit_gain": float(first.local_orbit_gain),
        "first_station_condition_gain": float(first.station_condition_gain),
        "first_correlation_reduction_gain": float(first.correlation_reduction_gain),
        "first_cardinality_gap_gain": float(first.cardinality_gap_gain),
        "first_isotope_balance_gain": float(first.isotope_balance_gain),
        "first_elevation_condition_gain": float(first.elevation_condition_gain),
        "first_environment_signature_score": float(
            first.environment_signature_score
        ),
        "first_environment_signature_norm": float(
            _normalized_environment_signature_score(
                first.environment_signature_score,
                config=cfg,
            )
        ),
        "first_vertical_environment_signature_score": float(
            first.vertical_environment_signature_score
        ),
        "first_vertical_environment_signature_norm": float(
            _normalized_environment_signature_score(
                first.vertical_environment_signature_score,
                config=cfg,
            )
        ),
        "first_occlusion_boundary_gain": float(first.occlusion_boundary_gain),
        "remaining_budget_guidance": int(bool(cfg.remaining_budget_guidance)),
        "remaining_station_estimate": (
            -1
            if cfg.remaining_station_estimate is None
            else int(cfg.remaining_station_estimate)
        ),
        "remaining_route_pressure": float(first.remaining_route_pressure),
        "first_remaining_route_penalty": float(first.remaining_route_penalty),
        "first_remaining_route_gain": float(first.remaining_route_gain),
        "diagnostic_ranked_node_limit": int(ranked_limit),
        "selected_pairwise_ambiguity": selected_pairwise_ambiguity,
        "component_leaders": _component_leader_payloads(nodes),
        "ranked_nodes": [
            _node_diagnostic_payload(node, rank)
            for rank, node in enumerate(ranked_nodes, start=1)
        ],
    }
    return DSSPPResult(
        next_pose=first.pose_xyz.copy(),
        next_pose_index=int(first.pose_index),
        shield_program=first.program,
        score=best_score,
        sequence=tuple(sequence),
        diagnostics=diagnostics,
    )
