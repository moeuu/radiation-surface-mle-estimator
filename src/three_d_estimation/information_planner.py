"""Laplace/Fisher optimal experimental design for online surface MLE."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from measurement.continuous_kernels import ContinuousKernel
from runtime.discrepancy_calibration import load_discrepancy_calibration

from .config import MLEConfig
from .spectral_response_builder import build_spectral_response
from .types import MLEEstimate, ObservationBatch, SurfacePatch


PLANNING_METHOD = "laplace_poisson_fisher_d_s_station_block_optimal_v2"


@dataclass(frozen=True, slots=True)
class MLEPlanningConfig:
    """Configure local-Fisher measurement-pose and shield-program selection."""

    live_time_s: float = 10.0
    shield_program_length: int = 2
    max_active_source_parameters: int = 48
    max_total_source_parameters: int = 96
    active_strength_fraction: float = 1.0e-3
    source_strength_scale_floor_cps_1m: float = 1.0
    nuisance_scale_floor: float = 1.0
    laplace_prior_precision: float = 1.0
    minimum_expected_bin_count: float = 1.0e-3
    motion_cost_weight: float = 0.0
    rotation_cost_weight: float = 0.0
    candidate_pose_chunk_size: int = 8
    ranked_action_limit: int = 32
    shield_program_beam_width: int = 64
    future_station_rate_prior_precision: float = 1.0
    floor_ceiling_separation_weight: float = 1.0
    support_hypothesis_separation_weight: float = 0.5
    z_fisher_weight: float = 0.5
    response_correlation_reduction_weight: float = 0.25
    elevation_diversity_weight: float = 0.25
    geometry_exploration_weight: float = 0.5
    geometry_bootstrap_measurements: int = 6
    local_refinement_top_k: int = 8

    def __post_init__(self) -> None:
        """Validate all values that affect the planning objective."""
        integer_fields = {
            "shield_program_length": self.shield_program_length,
            "max_active_source_parameters": self.max_active_source_parameters,
            "max_total_source_parameters": self.max_total_source_parameters,
            "candidate_pose_chunk_size": self.candidate_pose_chunk_size,
            "ranked_action_limit": self.ranked_action_limit,
            "shield_program_beam_width": self.shield_program_beam_width,
            "geometry_bootstrap_measurements": self.geometry_bootstrap_measurements,
            "local_refinement_top_k": self.local_refinement_top_k,
        }
        for name, value in integer_fields.items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value,
                (int, np.integer),
            ):
                raise TypeError(f"{name} must be an integer.")
            if int(value) < 1:
                raise ValueError(f"{name} must be positive.")
        if int(self.max_active_source_parameters) > int(
            self.max_total_source_parameters
        ):
            raise ValueError(
                "max_active_source_parameters cannot exceed "
                "max_total_source_parameters."
            )
        positive_fields = {
            "live_time_s": self.live_time_s,
            "source_strength_scale_floor_cps_1m": (
                self.source_strength_scale_floor_cps_1m
            ),
            "nuisance_scale_floor": self.nuisance_scale_floor,
            "laplace_prior_precision": self.laplace_prior_precision,
            "minimum_expected_bin_count": self.minimum_expected_bin_count,
            "future_station_rate_prior_precision": (
                self.future_station_rate_prior_precision
            ),
        }
        for name, value in positive_fields.items():
            parsed = float(value)
            if not np.isfinite(parsed) or parsed <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        nonnegative_fields = {
            "active_strength_fraction": self.active_strength_fraction,
            "motion_cost_weight": self.motion_cost_weight,
            "rotation_cost_weight": self.rotation_cost_weight,
            "floor_ceiling_separation_weight": (self.floor_ceiling_separation_weight),
            "support_hypothesis_separation_weight": (
                self.support_hypothesis_separation_weight
            ),
            "z_fisher_weight": self.z_fisher_weight,
            "response_correlation_reduction_weight": (
                self.response_correlation_reduction_weight
            ),
            "elevation_diversity_weight": self.elevation_diversity_weight,
            "geometry_exploration_weight": self.geometry_exploration_weight,
        }
        for name, value in nonnegative_fields.items():
            parsed = float(value)
            if not np.isfinite(parsed) or parsed < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if float(self.active_strength_fraction) > 1.0:
            raise ValueError("active_strength_fraction must not exceed one.")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe planner configuration."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "MLEPlanningConfig":
        """Construct planner settings from one JSON object."""
        if not isinstance(payload, Mapping):
            raise TypeError("MLE planning configuration must be a mapping.")
        return cls(**dict(payload))

    @classmethod
    def load(cls, path: str | Path) -> "MLEPlanningConfig":
        """Load one strict JSON planner configuration file."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("MLE planning configuration root must be an object.")
        return cls.from_dict(payload)


@dataclass(frozen=True, slots=True)
class MLEPlanningAction:
    """Describe one candidate pose and its jointly optimized shield program."""

    candidate_index: int
    detector_pose_xyz: tuple[float, float, float]
    shield_pair_ids: tuple[int, ...]
    fe_orientation_indices: tuple[int, ...]
    pb_orientation_indices: tuple[int, ...]
    information_gain_nats: float
    travel_cost: float
    rotation_radians: float
    score: float
    live_time_s_by_view: tuple[float, ...]
    expected_total_counts_by_view: tuple[float, ...]
    floor_ceiling_separation: float = 0.0
    support_hypothesis_separation: float = 0.0
    z_fisher_information: float = 0.0
    response_correlation_reduction: float = 0.0
    elevation_diversity: float = 0.0
    geometry_exploration: float = 0.0

    def __post_init__(self) -> None:
        """Validate that the recommendation is a complete executable program."""
        count = len(self.shield_pair_ids)
        if count == 0 or any(
            len(values) != count
            for values in (
                self.fe_orientation_indices,
                self.pb_orientation_indices,
                self.live_time_s_by_view,
                self.expected_total_counts_by_view,
            )
        ):
            raise ValueError("Planning action view fields must be nonempty and align.")
        if int(self.candidate_index) < 0:
            raise ValueError("candidate_index must be nonnegative.")
        pose = np.asarray(self.detector_pose_xyz, dtype=np.float64)
        if pose.shape != (3,) or np.any(~np.isfinite(pose)):
            raise ValueError("detector_pose_xyz must contain three finite values.")
        if any(int(value) < 0 for value in self.shield_pair_ids):
            raise ValueError("shield_pair_ids must be nonnegative.")
        if any(int(value) < 0 for value in self.fe_orientation_indices) or any(
            int(value) < 0 for value in self.pb_orientation_indices
        ):
            raise ValueError("Shield orientation indices must be nonnegative.")
        finite_values = (
            self.information_gain_nats,
            self.travel_cost,
            self.rotation_radians,
            self.score,
            *self.live_time_s_by_view,
            *self.expected_total_counts_by_view,
            self.floor_ceiling_separation,
            self.support_hypothesis_separation,
            self.z_fisher_information,
            self.response_correlation_reduction,
            self.elevation_diversity,
            self.geometry_exploration,
        )
        if any(not np.isfinite(float(value)) for value in finite_values):
            raise ValueError("Planning action numerical values must be finite.")
        if float(self.information_gain_nats) < -1.0e-10:
            raise ValueError("information_gain_nats must be nonnegative.")
        if float(self.travel_cost) < 0.0 or float(self.rotation_radians) < 0.0:
            raise ValueError("Travel and rotation costs must be nonnegative.")
        if any(float(value) <= 0.0 for value in self.live_time_s_by_view):
            raise ValueError("Every planned live time must be positive.")
        if any(float(value) < 0.0 for value in self.expected_total_counts_by_view):
            raise ValueError("Expected counts must be nonnegative.")

    def to_dict(self) -> dict[str, object]:
        """Return the runtime-neutral action recommendation as JSON data."""
        program = [
            {
                "sequence_index": index,
                "shield_pair_id": int(pair_id),
                "fe_orientation_index": int(self.fe_orientation_indices[index]),
                "pb_orientation_index": int(self.pb_orientation_indices[index]),
                "live_time_s": float(self.live_time_s_by_view[index]),
                "station_complete": index == len(self.shield_pair_ids) - 1,
            }
            for index, pair_id in enumerate(self.shield_pair_ids)
        ]
        return {
            "action_schema_version": 1,
            "candidate_index": int(self.candidate_index),
            "detector_pose_xyz": [float(value) for value in self.detector_pose_xyz],
            "shield_pair_ids": [int(value) for value in self.shield_pair_ids],
            "fe_orientation_indices": [
                int(value) for value in self.fe_orientation_indices
            ],
            "pb_orientation_indices": [
                int(value) for value in self.pb_orientation_indices
            ],
            "information_gain_nats": float(self.information_gain_nats),
            "travel_cost": float(self.travel_cost),
            "rotation_radians": float(self.rotation_radians),
            "score": float(self.score),
            "live_time_s_by_view": [float(value) for value in self.live_time_s_by_view],
            "expected_total_counts_by_view": [
                float(value) for value in self.expected_total_counts_by_view
            ],
            "floor_ceiling_separation": float(self.floor_ceiling_separation),
            "support_hypothesis_separation": float(self.support_hypothesis_separation),
            "z_fisher_information": float(self.z_fisher_information),
            "response_correlation_reduction": float(
                self.response_correlation_reduction
            ),
            "elevation_diversity": float(self.elevation_diversity),
            "geometry_exploration": float(self.geometry_exploration),
            "measurement_program": program,
        }


@dataclass(frozen=True, slots=True)
class MLEPlanningResult:
    """Return the selected action and bounded deterministic candidate ranking."""

    selected_action: MLEPlanningAction
    ranked_actions: tuple[MLEPlanningAction, ...]
    diagnostics: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """Return a strict JSON planning artifact."""
        return {
            "schema_version": 1,
            "planning_method": PLANNING_METHOD,
            "selected_action": self.selected_action.to_dict(),
            "ranked_actions": [action.to_dict() for action in self.ranked_actions],
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True, slots=True)
class _PlanningGeometry:
    """Expose only geometry fields consumed by the spectral response builder."""

    detector_positions_xyz: NDArray[np.float64]
    fe_indices: NDArray[np.int64]
    pb_indices: NDArray[np.int64]
    live_times_s: NDArray[np.float64]
    energy_bin_edges_keV: NDArray[np.float64]
    station_ids: NDArray[np.int64] | None = None


@dataclass(frozen=True, slots=True)
class _PatchView:
    """Expose estimated patches with aggregate area access."""

    patches: tuple[SurfacePatch, ...]

    @property
    def areas_m2(self) -> NDArray[np.float64]:
        """Return one physical area per estimated patch."""
        return np.asarray(
            [patch.area_m2 for patch in self.patches],
            dtype=np.float64,
        )


def _validated_candidate_poses(
    candidate_poses_xyz: object,
) -> NDArray[np.float64]:
    """Return a nonempty finite C x 3 candidate-pose array."""
    poses = np.asarray(candidate_poses_xyz, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1:] != (3,) or poses.shape[0] == 0:
        raise ValueError("candidate_poses_xyz must have nonempty shape (C, 3).")
    if np.any(~np.isfinite(poses)):
        raise ValueError("candidate_poses_xyz must contain only finite values.")
    return np.ascontiguousarray(poses)


def _validated_travel_costs(
    travel_costs: object | None,
    candidate_count: int,
) -> NDArray[np.float64]:
    """Return one finite nonnegative externally supplied cost per pose."""
    if travel_costs is None:
        return np.zeros(candidate_count, dtype=np.float64)
    costs = np.asarray(travel_costs, dtype=np.float64)
    if costs.shape != (candidate_count,):
        raise ValueError(f"travel_costs must have shape ({candidate_count},).")
    if np.any(~np.isfinite(costs)) or np.any(costs < 0.0):
        raise ValueError("travel_costs must contain finite nonnegative values.")
    return np.ascontiguousarray(costs)


def _validated_pair_ids(
    allowed_pair_ids: Sequence[int] | None,
    orientation_count: int,
) -> NDArray[np.int64]:
    """Return unique valid pair IDs under the shared runtime pair convention."""
    pair_count = int(orientation_count) ** 2
    if allowed_pair_ids is None:
        return np.arange(pair_count, dtype=np.int64)
    raw = np.asarray(tuple(allowed_pair_ids))
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("allowed_pair_ids must contain at least one pair ID.")
    if not np.issubdtype(raw.dtype, np.integer) or np.issubdtype(
        raw.dtype,
        np.bool_,
    ):
        raise TypeError("allowed_pair_ids must contain only integers.")
    pairs = np.asarray(raw, dtype=np.int64)
    if np.any(pairs < 0) or np.any(pairs >= pair_count):
        raise ValueError(f"allowed_pair_ids must lie in [0, {pair_count - 1}].")
    if np.unique(pairs).size != pairs.size:
        raise ValueError("allowed_pair_ids must not contain duplicates.")
    return np.ascontiguousarray(pairs)


def _source_basis(
    estimate: MLEEstimate,
    config: MLEPlanningConfig,
) -> tuple[NDArray[np.float64], tuple[dict[str, object], ...]]:
    """Build active patch modes plus residual exploration modes.

    The basis acts on patch-integrated strengths. Strong fitted patch/isotope
    entries receive individual dimensions. Every remaining entry is retained
    in an object-, surface-, or isotope-level aggregate, so a zero MLE region
    never disappears from the planning hypothesis space.
    """
    patch_count = len(estimate.patches)
    isotope_count = len(estimate.isotope_names)
    maximum_parameters = int(config.max_total_source_parameters)
    if maximum_parameters < isotope_count:
        raise ValueError(
            "max_total_source_parameters must be at least the isotope count."
        )
    strengths = np.asarray(
        estimate.patch_strength_by_isotope,
        dtype=np.float64,
    ).T
    candidates: list[tuple[float, int, int]] = []
    for isotope_index in range(isotope_count):
        isotope_values = strengths[:, isotope_index]
        maximum = float(np.max(isotope_values))
        threshold = maximum * float(config.active_strength_fraction)
        for patch_index in np.flatnonzero(
            (isotope_values > 0.0) & (isotope_values >= threshold)
        ):
            candidates.append(
                (
                    float(isotope_values[int(patch_index)]),
                    isotope_index,
                    int(patch_index),
                )
            )
    candidates.sort(
        key=lambda item: (
            -item[0],
            item[1],
            estimate.patches[item[2]].patch_id,
        )
    )
    active_limit = min(
        int(config.max_active_source_parameters),
        maximum_parameters - isotope_count,
    )
    active = candidates[:active_limit]
    active_indices = {(item[2], item[1]) for item in active}

    def grouped(mode: str) -> list[tuple[tuple[object, ...], list[tuple[int, int]]]]:
        """Return deterministic groups of all non-active source coordinates."""
        rows: dict[tuple[object, ...], list[tuple[int, int]]] = {}
        for isotope_index, isotope in enumerate(estimate.isotope_names):
            for patch_index, patch in enumerate(estimate.patches):
                coordinate = (patch_index, isotope_index)
                if coordinate in active_indices:
                    continue
                if mode == "object":
                    key = (isotope, patch.object_id)
                elif mode == "surface":
                    key = (isotope, patch.surface_kind)
                else:
                    key = (isotope,)
                rows.setdefault(key, []).append(coordinate)
        return sorted(rows.items(), key=lambda item: tuple(map(str, item[0])))

    residual_groups = grouped("object")
    grouping = "object_id"
    if len(active) + len(residual_groups) > maximum_parameters:
        residual_groups = grouped("surface")
        grouping = "surface_kind"
    if len(active) + len(residual_groups) > maximum_parameters:
        residual_groups = grouped("isotope")
        grouping = "isotope"
    if len(active) + len(residual_groups) > maximum_parameters:
        raise RuntimeError("Source planning basis could not satisfy its size cap.")

    parameter_count = len(active) + len(residual_groups)
    basis = np.zeros(
        (patch_count, isotope_count, parameter_count),
        dtype=np.float64,
    )
    labels: list[dict[str, object]] = []
    floor = float(config.source_strength_scale_floor_cps_1m)
    for column, (strength, isotope_index, patch_index) in enumerate(active):
        scale = max(float(strength), floor)
        basis[patch_index, isotope_index, column] = scale
        patch = estimate.patches[patch_index]
        labels.append(
            {
                "kind": "active_patch",
                "isotope": estimate.isotope_names[isotope_index],
                "patch_ids": [int(patch.patch_id)],
                "scale_cps_1m": scale,
            }
        )
    for offset, (group_key, coordinates) in enumerate(residual_groups):
        column = len(active) + offset
        areas = np.asarray(
            [estimate.patches[index].area_m2 for index, _ in coordinates],
            dtype=np.float64,
        )
        weights = areas / float(np.sum(areas))
        for weight, (patch_index, isotope_index) in zip(
            weights,
            coordinates,
            strict=True,
        ):
            basis[patch_index, isotope_index, column] = floor * float(weight)
        isotope_index = coordinates[0][1]
        labels.append(
            {
                "kind": "residual_exploration",
                "grouping": grouping,
                "group_key": [str(value) for value in group_key],
                "isotope": estimate.isotope_names[isotope_index],
                "patch_ids": [
                    int(estimate.patches[index].patch_id) for index, _ in coordinates
                ],
                "scale_cps_1m": floor,
            }
        )
    if parameter_count == 0 or np.any(np.sum(basis, axis=(0, 1)) <= 0.0):
        raise RuntimeError("Every source planning basis column must be nonzero.")
    return basis, tuple(labels)


def _nuisance_coefficients(
    estimate: MLEEstimate,
    nuisance_names: Sequence[str],
) -> NDArray[np.float64]:
    """Restore fitted nuisance values in the response builder's exact order."""
    fitted_names = tuple(
        str(value) for value in estimate.diagnostics.get("nuisance_names", [])
    )
    fitted_values = np.concatenate(
        (
            np.asarray(estimate.background_parameters, dtype=float),
            np.asarray(estimate.nuisance_parameters, dtype=float),
        )
    )
    if len(fitted_names) == fitted_values.size and fitted_names:
        by_name = dict(zip(fitted_names, fitted_values, strict=True))
        missing = [name for name in nuisance_names if name not in by_name]
        if missing:
            raise ValueError(
                f"Estimate does not contain planner nuisance coefficients {missing}."
            )
        return np.asarray([by_name[name] for name in nuisance_names], dtype=np.float64)
    background = iter(np.asarray(estimate.background_parameters, dtype=float))
    other = iter(np.asarray(estimate.nuisance_parameters, dtype=float))
    values: list[float] = []
    for name in nuisance_names:
        selected = background if name.startswith("background") else other
        try:
            values.append(float(next(selected)))
        except StopIteration as exc:
            raise ValueError(
                "Estimate nuisance parameters do not match the response basis."
            ) from exc
    try:
        next(background)
    except StopIteration:
        pass
    else:
        raise ValueError("Estimate contains unused background parameters.")
    try:
        next(other)
    except StopIteration:
        pass
    else:
        raise ValueError("Estimate contains unused non-background nuisance parameters.")
    return np.asarray(values, dtype=np.float64)


def _spectral_design(
    observations: object,
    estimate: MLEEstimate,
    kernel: ContinuousKernel,
    mle_config: MLEConfig,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    tuple[str, ...],
]:
    """Build integrated-strength and nuisance responses with shared physics."""
    calibration = (
        None
        if mle_config.discrepancy_calibration_path is None
        else load_discrepancy_calibration(mle_config.discrepancy_calibration_path)
    )
    details = build_spectral_response(
        observations,
        _PatchView(estimate.patches),
        estimate.isotope_names,
        kernel,
        chunk_size=int(mle_config.response_chunk_size),
        continuum_to_peak=float(mle_config.continuum_to_peak),
        backscatter_fraction=float(mle_config.backscatter_fraction),
        require_line_resolved=True,
        include_background_nuisance=bool(mle_config.fit_background_nuisance),
        include_scatter_nuisance=bool(mle_config.fit_scatter_nuisance),
        discrepancy_calibration=calibration,
        include_shield_leakage_nuisance=bool(mle_config.fit_shield_leakage_nuisance),
        # A future station coefficient has no fitted value.  Planner Fisher
        # information therefore marginalizes only calibrated run-global bases.
        include_station_rate_nuisance=False,
        include_low_rank_residual_nuisance=bool(
            mle_config.fit_low_rank_residual_nuisance
        ),
        include_gain_resolution_drift=bool(mle_config.fit_gain_resolution_drift),
    )
    return (
        details.response_per_integrated_strength,
        details.nuisance_response,
        details.nuisance_names,
    )


def _fisher_information(
    source_response: NDArray[np.float64],
    nuisance_response: NDArray[np.float64],
    source_basis: NDArray[np.float64],
    source_strengths: NDArray[np.float64],
    nuisance_coefficients: NDArray[np.float64],
    nuisance_scales: NDArray[np.float64],
    *,
    minimum_expected_count: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return Fisher terms including a shared future station-rate nuisance."""
    response = np.asarray(source_response, dtype=np.float64)
    nuisance = np.asarray(nuisance_response, dtype=np.float64)
    if response.ndim != 4:
        raise ValueError("Spectral source_response must have shape (A, B, G, I).")
    action_count, bin_count, patch_count, isotope_count = response.shape
    if source_basis.shape[:2] != (patch_count, isotope_count):
        raise ValueError("source_basis does not match response patch/isotope axes.")
    if source_strengths.shape != (patch_count, isotope_count):
        raise ValueError("source_strengths do not match the response axes.")
    if nuisance.shape[:2] != (action_count, bin_count):
        raise ValueError("nuisance_response does not match action/bin axes.")
    if nuisance.shape[2] != nuisance_coefficients.size or (
        nuisance_coefficients.shape != nuisance_scales.shape
    ):
        raise ValueError("Nuisance response, coefficients, and scales must align.")
    source_jacobian = np.einsum(
        "abgi,gik->abk",
        response,
        source_basis,
        optimize=True,
    )
    nuisance_jacobian = nuisance * nuisance_scales[None, None, :]
    jacobian = np.concatenate((source_jacobian, nuisance_jacobian), axis=2)
    expected = np.einsum(
        "abgi,gi->ab",
        response,
        source_strengths,
        optimize=True,
    )
    if nuisance_coefficients.size:
        expected = expected + np.einsum(
            "abn,n->ab",
            nuisance,
            nuisance_coefficients,
            optimize=True,
        )
    expected = np.maximum(expected, float(minimum_expected_count))
    information = np.einsum(
        "abp,abq,ab->apq",
        jacobian,
        jacobian,
        1.0 / expected,
        optimize=True,
    )
    information = 0.5 * (information + np.swapaxes(information, 1, 2))
    station_cross = np.sum(jacobian, axis=1)
    station_information = np.sum(expected, axis=1)
    return information, station_information, station_cross, station_information


def _symmetric_spectral_separation(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return bounded symmetric chi-square separation along the bin axis."""
    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    first_normalized = first_values / np.maximum(
        np.sum(first_values, axis=-1, keepdims=True),
        1.0e-30,
    )
    second_normalized = second_values / np.maximum(
        np.sum(second_values, axis=-1, keepdims=True),
        1.0e-30,
    )
    denominator = first_normalized + second_normalized
    return 0.5 * np.sum(
        np.divide(
            (first_normalized - second_normalized) ** 2,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0.0,
        ),
        axis=-1,
    )


def _ambiguity_metrics(
    response: NDArray[np.float64],
    information: NDArray[np.float64],
    poses: NDArray[np.float64],
    estimate: MLEEstimate,
    historical: ObservationBatch,
    source_basis: NDArray[np.float64],
    alternatives: Sequence[MLEEstimate],
) -> dict[str, NDArray[np.float64]]:
    """Return pose/pair metrics for vertical and support-hypothesis ambiguity."""
    action_count, _bin_count, patch_count, isotope_count = response.shape
    if patch_count != len(estimate.patches):
        raise ValueError("Planner response and estimate patches do not align.")
    floor = np.asarray(
        [patch.surface_kind == "floor" for patch in estimate.patches],
        dtype=bool,
    )
    ceiling = np.asarray(
        [patch.surface_kind == "ceiling" for patch in estimate.patches],
        dtype=bool,
    )

    def surface_spectrum(mask: NDArray[np.bool_]) -> NDArray[np.float64]:
        """Return equal-total-strength spectra for one competing surface."""
        weights = np.zeros((patch_count, isotope_count), dtype=np.float64)
        if np.any(mask):
            weights[mask] = 1.0 / (float(np.count_nonzero(mask)) * isotope_count)
        return np.einsum("abgi,gi->ab", response, weights, optimize=True)

    floor_spectrum = surface_spectrum(floor)
    ceiling_spectrum = surface_spectrum(ceiling)
    floor_ceiling = _symmetric_spectral_separation(
        floor_spectrum,
        ceiling_spectrum,
    )
    floor_centered = floor_spectrum - np.mean(floor_spectrum, axis=1, keepdims=True)
    ceiling_centered = ceiling_spectrum - np.mean(
        ceiling_spectrum,
        axis=1,
        keepdims=True,
    )
    denominator = np.linalg.norm(floor_centered, axis=1) * np.linalg.norm(
        ceiling_centered,
        axis=1,
    )
    correlation = np.divide(
        np.sum(floor_centered * ceiling_centered, axis=1),
        denominator,
        out=np.ones(action_count, dtype=np.float64),
        where=denominator > 0.0,
    )
    correlation_reduction = 1.0 - np.clip(np.abs(correlation), 0.0, 1.0)

    patch_z = np.asarray([patch.centroid_xyz[2] for patch in estimate.patches])
    basis_mass = np.sum(np.abs(source_basis), axis=(0, 1))
    basis_z = np.einsum(
        "g,gik->k",
        patch_z,
        np.abs(source_basis),
        optimize=True,
    ) / np.maximum(basis_mass, 1.0e-30)
    z_span = max(float(np.ptp(patch_z)), 1.0e-12)
    z_scale = (basis_z - float(np.mean(basis_z))) / z_span
    source_count = source_basis.shape[2]
    z_fisher = np.einsum(
        "ak,ak->a",
        np.diagonal(information[:, :source_count, :source_count], axis1=1, axis2=2),
        np.broadcast_to(z_scale * z_scale, (action_count, source_count)),
        optimize=True,
    )
    z_fisher = np.log1p(np.maximum(z_fisher, 0.0))

    support_separation = np.zeros(action_count, dtype=np.float64)
    base_strength = np.asarray(estimate.patch_strength_by_isotope, dtype=float).T
    base_prediction = np.einsum(
        "abgi,gi->ab",
        response,
        base_strength,
        optimize=True,
    )
    valid_alternatives = [
        alternative
        for alternative in alternatives
        if tuple(patch.patch_id for patch in alternative.patches)
        == tuple(patch.patch_id for patch in estimate.patches)
        and tuple(alternative.isotope_names) == tuple(estimate.isotope_names)
    ]
    for alternative in valid_alternatives:
        prediction = np.einsum(
            "abgi,gi->ab",
            response,
            np.asarray(alternative.patch_strength_by_isotope, dtype=float).T,
            optimize=True,
        )
        support_separation = np.maximum(
            support_separation,
            _symmetric_spectral_separation(base_prediction, prediction),
        )

    strengths = np.sum(base_strength, axis=1)
    source_centroid = (
        np.average(
            np.vstack([patch.centroid_xyz for patch in estimate.patches]),
            axis=0,
            weights=np.maximum(strengths, 0.0),
        )
        if np.any(strengths > 0.0)
        else np.mean(
            np.vstack([patch.centroid_xyz for patch in estimate.patches]),
            axis=0,
        )
    )

    def elevation(candidate: NDArray[np.float64]) -> float:
        """Return source-centroid elevation from one detector pose."""
        delta = source_centroid - candidate
        return float(np.arctan2(delta[2], max(np.linalg.norm(delta[:2]), 1.0e-12)))

    historical_elevations = np.asarray(
        [elevation(pose) for pose in historical.detector_positions_xyz],
        dtype=np.float64,
    )
    pose_count = poses.shape[0]
    elevation_diversity = np.zeros(pose_count, dtype=np.float64)
    geometry_exploration = np.zeros(pose_count, dtype=np.float64)
    scale = max(
        float(
            np.linalg.norm(
                np.ptp(np.vstack((poses, historical.detector_positions_xyz)), axis=0)
            )
        ),
        1.0e-12,
    )
    for pose_index, pose in enumerate(poses):
        angle = elevation(pose)
        elevation_diversity[pose_index] = min(
            1.0,
            float(np.min(np.abs(angle - historical_elevations))) / (0.5 * np.pi),
        )
        geometry_exploration[pose_index] = min(
            1.0,
            float(
                np.min(
                    np.linalg.norm(
                        historical.detector_positions_xyz - pose[None, :],
                        axis=1,
                    )
                )
            )
            / scale,
        )
    return {
        "floor_ceiling": floor_ceiling,
        "support": support_separation,
        "z_fisher": z_fisher,
        "correlation": correlation_reduction,
        "elevation": np.repeat(elevation_diversity, action_count // pose_count),
        "geometry": np.repeat(geometry_exploration, action_count // pose_count),
    }


def _source_log_precision(
    precision: NDArray[np.float64],
    nuisance_count: int,
) -> float:
    """Return log determinant of source precision after nuisance marginalization."""
    matrix = np.asarray(precision, dtype=np.float64)
    sign, full_logdet = np.linalg.slogdet(matrix)
    if sign <= 0.0 or not np.isfinite(full_logdet):
        raise np.linalg.LinAlgError("Planning precision must be positive definite.")
    if nuisance_count == 0:
        return float(full_logdet)
    nuisance = matrix[-nuisance_count:, -nuisance_count:]
    nuisance_sign, nuisance_logdet = np.linalg.slogdet(nuisance)
    if nuisance_sign <= 0.0 or not np.isfinite(nuisance_logdet):
        raise np.linalg.LinAlgError(
            "Planning nuisance precision must be positive definite."
        )
    return float(full_logdet - nuisance_logdet)


def _pair_rotation_radians(
    first_pair_id: int | None,
    second_pair_id: int,
    orientations: NDArray[np.float64],
) -> float:
    """Return summed Fe/Pb angular motion between two runtime pair IDs."""
    if first_pair_id is None:
        return 0.0
    count = int(orientations.shape[0])
    first_fe, first_pb = divmod(int(first_pair_id), count)
    second_fe, second_pb = divmod(int(second_pair_id), count)

    def angle(first: int, second: int) -> float:
        """Return the stable angle between two orientation normals."""
        dot = float(np.dot(orientations[first], orientations[second]))
        return float(np.arccos(np.clip(dot, -1.0, 1.0)))

    return angle(first_fe, second_fe) + angle(first_pb, second_pb)


def _select_pose_program(
    candidate_index: int,
    pose_xyz: NDArray[np.float64],
    pair_ids: NDArray[np.int64],
    pair_information: NDArray[np.float64],
    expected_counts: NDArray[np.float64],
    base_precision: NDArray[np.float64],
    nuisance_count: int,
    orientations: NDArray[np.float64],
    config: MLEPlanningConfig,
    *,
    travel_cost: float,
    current_pair_id: int | None,
    station_rate_cross_information: NDArray[np.float64] | None = None,
    station_rate_information: NDArray[np.float64] | None = None,
    pair_utility_bonus: NDArray[np.float64] | None = None,
) -> MLEPlanningAction:
    """Jointly optimize a complete station shield program with beam search."""
    if int(config.shield_program_length) > pair_ids.size:
        raise ValueError("shield_program_length cannot exceed the allowed pair count.")
    precision = np.asarray(base_precision, dtype=np.float64)
    information = np.asarray(pair_information, dtype=np.float64)
    if (station_rate_cross_information is None) != (station_rate_information is None):
        raise ValueError("Both future station-rate Fisher terms must be supplied.")
    effective_nuisance_count = int(nuisance_count)
    if station_rate_cross_information is not None:
        cross = np.asarray(station_rate_cross_information, dtype=np.float64)
        station = np.asarray(station_rate_information, dtype=np.float64)
        parameter_count = int(precision.shape[0])
        if cross.shape != (pair_ids.size, parameter_count):
            raise ValueError("station_rate_cross_information has invalid shape.")
        if station.shape != (pair_ids.size,):
            raise ValueError("station_rate_information has invalid shape.")
        extended_precision = np.zeros(
            (parameter_count + 1, parameter_count + 1),
            dtype=np.float64,
        )
        extended_precision[:-1, :-1] = precision
        extended_precision[-1, -1] = float(config.future_station_rate_prior_precision)
        extended_information = np.zeros(
            (pair_ids.size, parameter_count + 1, parameter_count + 1),
            dtype=np.float64,
        )
        extended_information[:, :-1, :-1] = information
        extended_information[:, :-1, -1] = cross
        extended_information[:, -1, :-1] = cross
        extended_information[:, -1, -1] = station
        precision = extended_precision
        information = extended_information
        effective_nuisance_count += 1
    bonuses = (
        np.zeros(pair_ids.size, dtype=np.float64)
        if pair_utility_bonus is None
        else np.asarray(pair_utility_bonus, dtype=np.float64)
    )
    if bonuses.shape != (pair_ids.size,) or np.any(~np.isfinite(bonuses)):
        raise ValueError("pair_utility_bonus must be one finite value per pair.")
    base_value = _source_log_precision(precision, effective_nuisance_count)
    states: list[
        tuple[
            tuple[int, ...],
            tuple[int, ...],
            NDArray[np.float64],
            float,
            float,
            int | None,
        ]
    ] = [((), (), precision.copy(), base_value, 0.0, current_pair_id)]
    for _ in range(int(config.shield_program_length)):
        expansions: list[
            tuple[
                tuple[float, float, tuple[int, ...]],
                tuple[
                    tuple[int, ...],
                    tuple[int, ...],
                    NDArray[np.float64],
                    float,
                    float,
                    int,
                ],
            ]
        ] = []
        for (
            selected_indices,
            selected_pairs,
            state_precision,
            _,
            rotation,
            previous,
        ) in states:
            selected_set = set(selected_indices)
            for pair_index, raw_pair_id in enumerate(pair_ids):
                if pair_index in selected_set:
                    continue
                pair_id = int(raw_pair_id)
                next_indices = (*selected_indices, pair_index)
                next_pairs = (*selected_pairs, pair_id)
                next_precision = state_precision + information[pair_index]
                next_value = _source_log_precision(
                    next_precision,
                    effective_nuisance_count,
                )
                next_rotation = rotation + _pair_rotation_radians(
                    previous,
                    pair_id,
                    orientations,
                )
                information_gain = 0.5 * (next_value - base_value)
                utility_bonus = float(np.mean(bonuses[list(next_indices)]))
                partial_score = (
                    information_gain
                    + utility_bonus
                    - float(config.rotation_cost_weight) * next_rotation
                )
                expansions.append(
                    (
                        (-partial_score, -information_gain, next_pairs),
                        (
                            next_indices,
                            next_pairs,
                            next_precision,
                            next_value,
                            next_rotation,
                            pair_id,
                        ),
                    )
                )
        if not expansions:
            raise RuntimeError("No unselected shield pair remains.")
        expansions.sort(key=lambda item: item[0])
        states = [
            state for _, state in expansions[: int(config.shield_program_beam_width)]
        ]
    selected_indices, selected_pair_ids, _, current_value, total_rotation, _ = states[0]
    information_gain = 0.5 * (current_value - base_value)
    utility_bonus = float(np.mean(bonuses[list(selected_indices)]))
    score = (
        information_gain
        + utility_bonus
        - float(config.motion_cost_weight) * float(travel_cost)
        - float(config.rotation_cost_weight) * total_rotation
    )
    orientation_count = int(orientations.shape[0])
    return MLEPlanningAction(
        candidate_index=int(candidate_index),
        detector_pose_xyz=tuple(float(value) for value in pose_xyz),
        shield_pair_ids=tuple(selected_pair_ids),
        fe_orientation_indices=tuple(
            pair_id // orientation_count for pair_id in selected_pair_ids
        ),
        pb_orientation_indices=tuple(
            pair_id % orientation_count for pair_id in selected_pair_ids
        ),
        information_gain_nats=float(information_gain),
        travel_cost=float(travel_cost),
        rotation_radians=float(total_rotation),
        score=float(score),
        live_time_s_by_view=tuple(float(config.live_time_s) for _ in selected_indices),
        expected_total_counts_by_view=tuple(
            float(expected_counts[index]) for index in selected_indices
        ),
    )


def select_fisher_action(
    candidate_poses_xyz: object,
    pair_ids: Sequence[int],
    candidate_information: object,
    expected_total_counts: object,
    base_precision: object,
    orientations: object,
    *,
    nuisance_count: int,
    config: MLEPlanningConfig | None = None,
    travel_costs: object | None = None,
    current_pair_id: int | None = None,
    station_rate_cross_information: object | None = None,
    station_rate_information: object | None = None,
    pair_utility_bonus: object | None = None,
) -> tuple[MLEPlanningAction, tuple[MLEPlanningAction, ...]]:
    """Select a joint pose/program from precomputed expected Fisher matrices."""
    resolved = MLEPlanningConfig() if config is None else config
    poses = _validated_candidate_poses(candidate_poses_xyz)
    information = np.asarray(candidate_information, dtype=np.float64)
    totals = np.asarray(expected_total_counts, dtype=np.float64)
    precision = np.asarray(base_precision, dtype=np.float64)
    orientation_array = np.asarray(orientations, dtype=np.float64)
    candidate_count = int(poses.shape[0])
    parameter_count = int(precision.shape[0]) if precision.ndim == 2 else 0
    if orientation_array.ndim != 2 or orientation_array.shape[1:] != (3,):
        raise ValueError("orientations must have shape (R, 3).")
    pair_array = _validated_pair_ids(
        tuple(pair_ids),
        int(orientation_array.shape[0]),
    )
    pair_count = int(pair_array.size)
    if information.shape != (
        candidate_count,
        pair_count,
        parameter_count,
        parameter_count,
    ):
        raise ValueError("candidate_information has incompatible dimensions.")
    if totals.shape != (candidate_count, pair_count):
        raise ValueError("expected_total_counts must align with candidates and pairs.")
    if precision.shape != (parameter_count, parameter_count):
        raise ValueError("base_precision must be a square matrix.")
    if isinstance(nuisance_count, bool) or not isinstance(
        nuisance_count,
        (int, np.integer),
    ):
        raise TypeError("nuisance_count must be an integer.")
    if not 0 <= int(nuisance_count) < parameter_count:
        raise ValueError("nuisance_count must lie in [0, parameter_count).")
    if np.any(~np.isfinite(information)) or np.any(~np.isfinite(totals)):
        raise ValueError("Candidate information and expected counts must be finite.")
    if np.any(totals < 0.0):
        raise ValueError("expected_total_counts must be nonnegative.")
    if (station_rate_cross_information is None) != (station_rate_information is None):
        raise ValueError("Both future station-rate Fisher terms must be supplied.")
    station_cross = None
    station_information = None
    if station_rate_cross_information is not None:
        station_cross = np.asarray(
            station_rate_cross_information,
            dtype=np.float64,
        )
        station_information = np.asarray(station_rate_information, dtype=np.float64)
        if station_cross.shape != (candidate_count, pair_count, parameter_count):
            raise ValueError("station_rate_cross_information has invalid dimensions.")
        if station_information.shape != (candidate_count, pair_count):
            raise ValueError("station_rate_information has invalid dimensions.")
        if np.any(~np.isfinite(station_cross)) or np.any(
            ~np.isfinite(station_information)
        ):
            raise ValueError("Future station-rate Fisher terms must be finite.")
        if np.any(station_information < 0.0):
            raise ValueError("station_rate_information must be nonnegative.")
    bonuses = (
        np.zeros((candidate_count, pair_count), dtype=np.float64)
        if pair_utility_bonus is None
        else np.asarray(pair_utility_bonus, dtype=np.float64)
    )
    if bonuses.shape != (candidate_count, pair_count) or np.any(~np.isfinite(bonuses)):
        raise ValueError("pair_utility_bonus must align with candidates and pairs.")
    costs = _validated_travel_costs(travel_costs, candidate_count)
    actions = [
        _select_pose_program(
            index,
            poses[index],
            pair_array,
            information[index],
            totals[index],
            precision,
            nuisance_count,
            orientation_array,
            resolved,
            travel_cost=float(costs[index]),
            current_pair_id=current_pair_id,
            station_rate_cross_information=(
                None if station_cross is None else station_cross[index]
            ),
            station_rate_information=(
                None if station_information is None else station_information[index]
            ),
            pair_utility_bonus=bonuses[index],
        )
        for index in range(candidate_count)
    ]
    ranked = tuple(
        sorted(
            actions,
            key=lambda action: (
                -action.score,
                -action.information_gain_nats,
                action.candidate_index,
                action.shield_pair_ids,
            ),
        )
    )
    return ranked[0], ranked[: int(resolved.ranked_action_limit)]


def plan_next_measurement(
    estimate: MLEEstimate,
    historical_observations: ObservationBatch,
    kernel: ContinuousKernel,
    mle_config: MLEConfig,
    candidate_poses_xyz: object,
    *,
    planning_config: MLEPlanningConfig | None = None,
    allowed_pair_ids: Sequence[int] | None = None,
    travel_costs: object | None = None,
    current_pair_id: int | None = None,
    alternative_estimates: Sequence[MLEEstimate] = (),
) -> MLEPlanningResult:
    """Plan a joint next station and Fe/Pb program from one fitted MLE.

    Candidate generation, obstacle traversability, and exact travel costs stay
    with the shared runtime. This function ranks only the truth-free candidates
    supplied by that controller.
    """
    if not isinstance(estimate, MLEEstimate):
        raise TypeError("estimate must be an MLEEstimate.")
    if not isinstance(historical_observations, ObservationBatch):
        raise TypeError("historical_observations must be an ObservationBatch.")
    if not isinstance(kernel, ContinuousKernel):
        raise TypeError("kernel must be the shared runtime ContinuousKernel.")
    if not isinstance(mle_config, MLEConfig):
        raise TypeError("mle_config must be an MLEConfig.")
    if not isinstance(alternative_estimates, Sequence):
        raise TypeError("alternative_estimates must be a sequence.")
    if any(
        not isinstance(alternative, MLEEstimate)
        for alternative in alternative_estimates
    ):
        raise TypeError("Every alternative estimate must be an MLEEstimate.")
    if mle_config.mode != "spectral":
        raise ValueError("Online MLE planning requires spectral mode.")
    if tuple(estimate.isotope_names) != tuple(mle_config.isotope_names) or (
        tuple(historical_observations.isotope_names) != tuple(estimate.isotope_names)
    ):
        raise ValueError("Planner isotope order must match estimate and history.")
    resolved = MLEPlanningConfig() if planning_config is None else planning_config
    poses = _validated_candidate_poses(candidate_poses_xyz)
    costs = _validated_travel_costs(travel_costs, int(poses.shape[0]))
    orientations = np.asarray(kernel.orientations, dtype=np.float64)
    if orientations.ndim != 2 or orientations.shape[1:] != (3,):
        raise ValueError("Shared kernel orientations must have shape (R, 3).")
    pair_ids = _validated_pair_ids(allowed_pair_ids, int(orientations.shape[0]))
    if int(resolved.shield_program_length) > pair_ids.size:
        raise ValueError("shield_program_length cannot exceed the allowed pair count.")
    pair_limit = int(orientations.shape[0]) ** 2
    if current_pair_id is not None and (
        isinstance(current_pair_id, bool)
        or not isinstance(current_pair_id, (int, np.integer))
        or not 0 <= int(current_pair_id) < pair_limit
    ):
        raise ValueError(f"current_pair_id must lie in [0, {pair_limit - 1}].")

    source_basis, basis_labels = _source_basis(estimate, resolved)
    source_strengths = np.asarray(
        estimate.patch_strength_by_isotope,
        dtype=np.float64,
    ).T
    historical_source, historical_nuisance, nuisance_names = _spectral_design(
        historical_observations,
        estimate,
        kernel,
        mle_config,
    )
    nuisance_coefficients = _nuisance_coefficients(estimate, nuisance_names)
    nuisance_scales = np.maximum(
        nuisance_coefficients,
        float(resolved.nuisance_scale_floor),
    )
    historical_information, _, _, _ = _fisher_information(
        historical_source,
        historical_nuisance,
        source_basis,
        source_strengths,
        nuisance_coefficients,
        nuisance_scales,
        minimum_expected_count=float(resolved.minimum_expected_bin_count),
    )
    parameter_count = int(source_basis.shape[2] + nuisance_coefficients.size)
    base_precision = float(resolved.laplace_prior_precision) * np.eye(
        parameter_count, dtype=np.float64
    ) + np.sum(historical_information, axis=0)
    nuisance_count = int(nuisance_coefficients.size)
    actions: list[MLEPlanningAction] = []
    pose_chunk = int(resolved.candidate_pose_chunk_size)
    orientation_count = int(orientations.shape[0])
    pair_fe = pair_ids // orientation_count
    pair_pb = pair_ids % orientation_count
    for start in range(0, int(poses.shape[0]), pose_chunk):
        stop = min(start + pose_chunk, int(poses.shape[0]))
        local_poses = poses[start:stop]
        local_count = int(local_poses.shape[0])
        expanded = np.repeat(local_poses, pair_ids.size, axis=0)
        geometry = _PlanningGeometry(
            detector_positions_xyz=expanded,
            fe_indices=np.tile(pair_fe, local_count),
            pb_indices=np.tile(pair_pb, local_count),
            live_times_s=np.full(
                expanded.shape[0],
                float(resolved.live_time_s),
                dtype=np.float64,
            ),
            energy_bin_edges_keV=historical_observations.energy_bin_edges_keV,
        )
        candidate_source, candidate_nuisance, candidate_names = _spectral_design(
            geometry,
            estimate,
            kernel,
            mle_config,
        )
        if tuple(candidate_names) != tuple(nuisance_names):
            raise RuntimeError("Historical and candidate nuisance bases differ.")
        (
            information,
            expected_counts,
            station_rate_cross,
            station_rate_information,
        ) = _fisher_information(
            candidate_source,
            candidate_nuisance,
            source_basis,
            source_strengths,
            nuisance_coefficients,
            nuisance_scales,
            minimum_expected_count=float(resolved.minimum_expected_bin_count),
        )
        ambiguity = _ambiguity_metrics(
            candidate_source,
            information,
            local_poses,
            estimate,
            historical_observations,
            source_basis,
            alternative_estimates,
        )
        information = information.reshape(
            local_count,
            pair_ids.size,
            parameter_count,
            parameter_count,
        )
        expected_counts = expected_counts.reshape(local_count, pair_ids.size)
        station_rate_cross = station_rate_cross.reshape(
            local_count,
            pair_ids.size,
            parameter_count,
        )
        station_rate_information = station_rate_information.reshape(
            local_count,
            pair_ids.size,
        )
        ambiguity_by_pose = {
            name: values.reshape(local_count, pair_ids.size)
            for name, values in ambiguity.items()
        }
        for local_index in range(local_count):
            global_index = start + local_index
            bootstrap_multiplier = (
                1.0
                if historical_observations.measurement_count
                < int(resolved.geometry_bootstrap_measurements)
                else 0.25
            )
            pair_utility_bonus = (
                float(resolved.floor_ceiling_separation_weight)
                * ambiguity_by_pose["floor_ceiling"][local_index]
                + float(resolved.support_hypothesis_separation_weight)
                * ambiguity_by_pose["support"][local_index]
                + float(resolved.z_fisher_weight)
                * ambiguity_by_pose["z_fisher"][local_index]
                + float(resolved.response_correlation_reduction_weight)
                * ambiguity_by_pose["correlation"][local_index]
                + float(resolved.elevation_diversity_weight)
                * ambiguity_by_pose["elevation"][local_index]
                + bootstrap_multiplier
                * float(resolved.geometry_exploration_weight)
                * ambiguity_by_pose["geometry"][local_index]
            )
            action = _select_pose_program(
                global_index,
                poses[global_index],
                pair_ids,
                information[local_index],
                expected_counts[local_index],
                base_precision,
                nuisance_count,
                orientations,
                resolved,
                travel_cost=float(costs[global_index]),
                current_pair_id=current_pair_id,
                station_rate_cross_information=(
                    station_rate_cross[local_index]
                    if mle_config.fit_station_rate_nuisance
                    else None
                ),
                station_rate_information=(
                    station_rate_information[local_index]
                    if mle_config.fit_station_rate_nuisance
                    else None
                ),
                pair_utility_bonus=pair_utility_bonus,
            )
            selected_indices = np.asarray(
                [
                    int(np.flatnonzero(pair_ids == pair_id)[0])
                    for pair_id in action.shield_pair_ids
                ],
                dtype=np.int64,
            )

            def selected_mean(name: str) -> float:
                """Return the selected program mean of one ambiguity metric."""
                return float(
                    np.mean(ambiguity_by_pose[name][local_index, selected_indices])
                )

            floor_ceiling = selected_mean("floor_ceiling")
            support = selected_mean("support")
            z_fisher = selected_mean("z_fisher")
            correlation = selected_mean("correlation")
            elevation = selected_mean("elevation")
            geometry = selected_mean("geometry")
            actions.append(
                replace(
                    action,
                    floor_ceiling_separation=floor_ceiling,
                    support_hypothesis_separation=support,
                    z_fisher_information=z_fisher,
                    response_correlation_reduction=correlation,
                    elevation_diversity=elevation,
                    geometry_exploration=geometry,
                )
            )
    ranked = tuple(
        sorted(
            actions,
            key=lambda action: (
                -action.score,
                -action.information_gain_nats,
                action.candidate_index,
                action.shield_pair_ids,
            ),
        )
    )
    selected = ranked[0]
    diagnostics: dict[str, object] = {
        "criterion": "D_s-optimal expected Poisson Fisher information",
        "laplace_approximation": True,
        "nuisance_marginalization": "Schur determinant",
        "shield_program_selection": "joint_station_block_beam_search",
        "future_station_rate_marginalized": bool(mle_config.fit_station_rate_nuisance),
        "candidate_generation_owner": "shared_runtime_controller",
        "candidate_count": int(poses.shape[0]),
        "allowed_pair_count": int(pair_ids.size),
        "source_parameter_count": int(source_basis.shape[2]),
        "nuisance_parameter_count": nuisance_count,
        "source_basis": list(basis_labels),
        "historical_measurement_count": historical_observations.measurement_count,
        "historical_step_ids": historical_observations.step_ids.astype(int).tolist(),
        "alternative_support_count": len(alternative_estimates),
        "ambiguity_aware": True,
        "geometry_bootstrap_active": (
            historical_observations.measurement_count
            < int(resolved.geometry_bootstrap_measurements)
        ),
        "config": resolved.to_dict(),
    }
    return MLEPlanningResult(
        selected_action=selected,
        ranked_actions=ranked[: int(resolved.ranked_action_limit)],
        diagnostics=diagnostics,
    )


def save_mle_planning_result(
    result: MLEPlanningResult,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically save one deterministic runtime-neutral planning artifact."""
    if not isinstance(result, MLEPlanningResult):
        raise TypeError("result must be an MLEPlanningResult.")
    target = Path(path).resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(f"MLE planning output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            result.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"MLE planning staging file exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


__all__ = [
    "MLEPlanningAction",
    "MLEPlanningConfig",
    "MLEPlanningResult",
    "PLANNING_METHOD",
    "plan_next_measurement",
    "save_mle_planning_result",
    "select_fisher_action",
]
