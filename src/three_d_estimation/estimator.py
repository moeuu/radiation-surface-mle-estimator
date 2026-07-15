"""All-history count and line-resolved spectral surface MLE estimators."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from measurement.continuous_kernels import ContinuousKernel
from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid

from .config import MLEConfig
from .observation_batch import subset_observation_batch
from .postprocess import (
    cluster_surface_hotspots,
    poisson_deviance,
    response_identifiability_diagnostics,
)
from .response_builder import build_count_responses
from .solver import SurfaceMapConfig, SurfaceMapResult, fit_surface_map_poisson
from .spectral_response_builder import SpectralResponseResult, build_spectral_response
from .surface_patches import build_surface_patches, refine_surface_patches
from .types import MLEEstimate, ObservationBatch, SurfacePatch, SurfacePatchSet


@dataclass(frozen=True)
class _FitState:
    """Store one solver result and the response used to obtain it."""

    patches: SurfacePatchSet
    response: NDArray[np.float64]
    nuisance_response: NDArray[np.float64]
    nuisance_names: tuple[str, ...]
    result: SurfaceMapResult
    fit_indices: NDArray[np.int64]
    held_out_indices: NDArray[np.int64]
    spectral_details: SpectralResponseResult | None


def _surface_map_config(config: MLEConfig, *, regularized: bool = True) -> SurfaceMapConfig:
    """Translate public MLE settings into the numerical solver contract."""
    return SurfaceMapConfig(
        l1_weight=float(config.l1_weight) if regularized else 0.0,
        tv_weight=float(config.tv_weight) if regularized else 0.0,
        isotope_group_weight=(
            float(config.isotope_group_weight) if regularized else 0.0
        ),
        nuisance_l1_weight=float(config.nuisance_l1_weight),
        nuisance_l2_weight=float(config.nuisance_l2_weight),
        max_iterations=int(config.max_iterations),
        tolerance=float(config.tolerance),
        objective_tolerance=float(config.objective_tolerance),
        check_interval=int(config.check_interval),
        step_safety=float(config.step_safety),
        over_relaxation=float(config.over_relaxation),
        min_mean=float(config.min_mean),
    )


def _split_fit_indices(
    measurement_count: int,
    fraction: float,
    seed: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return deterministic train and held-out measurement row indices."""
    all_indices = np.arange(int(measurement_count), dtype=np.int64)
    held_out_count = int(np.floor(float(fraction) * int(measurement_count)))
    if held_out_count <= 0:
        return all_indices, np.zeros(0, dtype=np.int64)
    held_out_count = min(held_out_count, int(measurement_count) - 1)
    rng = np.random.default_rng(int(seed))
    held_out = np.sort(rng.choice(all_indices, size=held_out_count, replace=False))
    fit = np.setdiff1d(all_indices, held_out, assume_unique=True)
    return fit.astype(np.int64), held_out.astype(np.int64)


def _count_solver_response(
    count_response: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Embed isotope-resolved count columns in a diagonal channel tensor."""
    isotope_count = int(count_response.shape[-1])
    identity = np.eye(isotope_count, dtype=float)
    return count_response[:, None, :, :] * identity[None, :, None, :]


def _count_nuisance_response(
    live_times_s: NDArray[np.float64],
    isotope_count: int,
    enabled: bool,
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """Return one isotope-specific background-rate basis per count channel."""
    measurement_count = int(np.asarray(live_times_s).size)
    if not enabled:
        return np.zeros((measurement_count, isotope_count, 0), dtype=float), ()
    identity = np.eye(isotope_count, dtype=float)
    basis = np.asarray(live_times_s, dtype=float)[:, None, None] * identity[None, :, :]
    names = tuple(f"background_rate_cps:{index}" for index in range(isotope_count))
    return basis, names


def _fit_problem(
    batch: ObservationBatch,
    patches: SurfacePatchSet,
    kernel: ContinuousKernel,
    config: MLEConfig,
    fit_indices: NDArray[np.int64],
    held_out_indices: NDArray[np.int64],
    *,
    initial_densities: NDArray[np.float64] | None = None,
    initial_nuisance: NDArray[np.float64] | None = None,
) -> _FitState:
    """Build the configured forward model and solve one patch resolution."""
    if config.mode == "count":
        if batch.isotope_counts is None:
            raise ValueError("Count-domain MLE requires response_poisson isotope counts.")
        count_details = build_count_responses(
            batch,
            patches,
            config.isotope_names,
            kernel,
            kernel_chunk_size=int(config.response_chunk_size),
        )
        response = _count_solver_response(
            count_details.response_by_integrated_strength
        )
        nuisance_response, nuisance_names = _count_nuisance_response(
            batch.live_times_s,
            batch.isotope_count,
            bool(config.fit_background_nuisance),
        )
        observed = batch.isotope_counts
        spectral_details = None
    else:
        spectral_details = build_spectral_response(
            batch,
            patches,
            config.isotope_names,
            kernel,
            chunk_size=int(config.response_chunk_size),
            continuum_to_peak=float(config.continuum_to_peak),
            backscatter_fraction=float(config.backscatter_fraction),
            require_line_resolved=True,
            include_background_nuisance=bool(config.fit_background_nuisance),
            include_scatter_nuisance=bool(config.fit_scatter_nuisance),
        )
        response = spectral_details.response_per_integrated_strength
        nuisance_response = spectral_details.nuisance_response
        nuisance_names = spectral_details.nuisance_names
        observed = batch.spectrum_counts

    fitted = fit_surface_map_poisson(
        observed[fit_indices],
        response[fit_indices],
        patches.areas_m2,
        adjacency_edges=patches.adjacency_index_edges,
        adjacency_weights=patches.shared_edge_lengths_m,
        background=0.0,
        nuisance_response=nuisance_response[fit_indices],
        initial_densities_cps_1m_m2=initial_densities,
        initial_nuisance_coefficients=initial_nuisance,
        config=_surface_map_config(config),
    )
    return _FitState(
        patches=patches,
        response=response,
        nuisance_response=nuisance_response,
        nuisance_names=nuisance_names,
        result=fitted,
        fit_indices=fit_indices,
        held_out_indices=held_out_indices,
        spectral_details=spectral_details,
    )


def _warm_start_refined(
    old_patches: SurfacePatchSet,
    new_patches: SurfacePatchSet,
    old_densities: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Transfer density, not total strength, to unchanged and child patches."""
    old_by_id = {
        patch.patch_id: np.asarray(old_densities[index], dtype=float)
        for index, patch in enumerate(old_patches.patches)
    }
    result = np.zeros((new_patches.patch_count, old_densities.shape[1]), dtype=float)
    for index, patch in enumerate(new_patches.patches):
        source_id = patch.patch_id if patch.patch_id in old_by_id else patch.parent_patch_id
        if source_id is not None and source_id in old_by_id:
            result[index] = old_by_id[source_id]
    return result


def _refinement_patch_ids(
    state: _FitState,
    fraction: float,
) -> tuple[int, ...]:
    """Select strongest active patches for the next coarse-to-fine level."""
    integrated = np.sum(state.result.integrated_strengths_cps_1m, axis=1)
    positive = np.flatnonzero(integrated > 0.0)
    if positive.size == 0 or float(fraction) <= 0.0:
        return ()
    selected_count = max(1, int(np.ceil(float(fraction) * positive.size)))
    order = positive[np.argsort(integrated[positive])[::-1][:selected_count]]
    return tuple(state.patches.patches[int(index)].patch_id for index in order)


def _debias_state(
    state: _FitState,
    observed: NDArray[np.float64],
    config: MLEConfig,
) -> _FitState:
    """Refit selected support without L1, TV, or group shrinkage bias."""
    densities = state.result.densities_cps_1m_m2
    maxima = np.max(densities, axis=0, keepdims=True)
    support = densities >= maxima * float(config.support_threshold_fraction)
    support &= maxima > 0.0
    if not np.any(support):
        return state
    if state.response.ndim == 4:
        masked_response = state.response * support[None, None, :, :]
    else:  # pragma: no cover - all production response tensors are rank four
        masked_response = state.response * support[None, :, :]
    result = fit_surface_map_poisson(
        observed[state.fit_indices],
        masked_response[state.fit_indices],
        state.patches.areas_m2,
        adjacency_edges=state.patches.adjacency_index_edges,
        adjacency_weights=state.patches.shared_edge_lengths_m,
        background=0.0,
        nuisance_response=state.nuisance_response[state.fit_indices],
        initial_densities_cps_1m_m2=np.where(support, densities, 0.0),
        initial_nuisance_coefficients=state.result.nuisance_coefficients,
        config=_surface_map_config(config, regularized=False),
    )
    return replace(state, response=masked_response, result=result)


def _full_prediction(state: _FitState) -> NDArray[np.float64]:
    """Evaluate the fitted density and nuisance model for every measurement."""
    strengths = state.result.integrated_strengths_cps_1m
    expected = np.einsum("...gi,gi->...", state.response, strengths, optimize=True)
    if state.result.nuisance_coefficients.size:
        expected = expected + np.einsum(
            "...n,n->...",
            state.nuisance_response,
            state.result.nuisance_coefficients,
            optimize=True,
        )
    return np.maximum(expected, 0.0)


def _json_floats(values: NDArray[np.float64]) -> list[float]:
    """Return a flat JSON-safe float list."""
    return [float(value) for value in np.asarray(values, dtype=float).reshape(-1)]


def _initial_density_for_patches(
    initial_estimate: MLEEstimate | None,
    patches: SurfacePatchSet,
    isotope_names: Sequence[str],
) -> NDArray[np.float64] | None:
    """Map a prior estimate onto a newly rebuilt deterministic base patch set."""
    if initial_estimate is None:
        return None
    if tuple(initial_estimate.isotope_names) != tuple(isotope_names):
        raise ValueError("Warm-start isotope order must match MLEConfig.isotope_names.")
    old_density = np.asarray(initial_estimate.density_by_isotope, dtype=float).T
    old_by_id = {
        patch.patch_id: (patch, old_density[index])
        for index, patch in enumerate(initial_estimate.patches)
    }

    def is_descendant(candidate: SurfacePatch, ancestor: SurfacePatch) -> bool:
        """Return whether one active leaf belongs below a rebuilt base patch.

        Refined estimates retain only active leaves, so the immediate parent of
        a level-two-or-deeper leaf is no longer present.  Stable lineage handles
        direct children; exact rectangular containment recovers the remainder
        while checking the same face identity, orientation, and refinement
        direction.
        """
        if candidate.parent_patch_id == ancestor.patch_id:
            return True
        if (
            candidate.parent_patch_id is None
            or candidate.refinement_level <= ancestor.refinement_level
            or candidate.surface_kind != ancestor.surface_kind
            or candidate.object_id != ancestor.object_id
            or not np.allclose(
                candidate.normal_xyz,
                ancestor.normal_xyz,
                rtol=1.0e-9,
                atol=1.0e-9,
            )
        ):
            return False

        origin = ancestor.vertices_xyz[0]
        u_vector = ancestor.vertices_xyz[1] - origin
        v_vector = ancestor.vertices_xyz[3] - origin
        relative = candidate.vertices_xyz - origin
        scale = max(
            float(np.linalg.norm(u_vector)),
            float(np.linalg.norm(v_vector)),
            1.0,
        )
        plane_tolerance = 1.0e-9 * scale
        if np.any(np.abs(relative @ ancestor.normal_xyz) > plane_tolerance):
            return False
        coordinate_tolerance = 1.0e-9
        u_coordinate = relative @ u_vector / float(np.dot(u_vector, u_vector))
        v_coordinate = relative @ v_vector / float(np.dot(v_vector, v_vector))
        return bool(
            np.all(u_coordinate >= -coordinate_tolerance)
            and np.all(u_coordinate <= 1.0 + coordinate_tolerance)
            and np.all(v_coordinate >= -coordinate_tolerance)
            and np.all(v_coordinate <= 1.0 + coordinate_tolerance)
        )

    result = np.zeros((patches.patch_count, len(isotope_names)), dtype=float)
    for index, patch in enumerate(patches.patches):
        previous = old_by_id.get(patch.patch_id)
        if previous is not None:
            previous_patch, density = previous
            if not np.isclose(previous_patch.area_m2, patch.area_m2) or not np.allclose(
                previous_patch.vertices_xyz,
                patch.vertices_xyz,
            ):
                raise ValueError(
                    f"Warm-start patch {patch.patch_id} geometry does not match."
                )
            result[index] = density
            continue
        descendants = [
            (candidate, old_density[candidate_index])
            for candidate_index, candidate in enumerate(initial_estimate.patches)
            if is_descendant(candidate, patch)
        ]
        if descendants:
            integrated = np.sum(
                [descendant.area_m2 * density for descendant, density in descendants],
                axis=0,
            )
            result[index] = integrated / patch.area_m2
    return result


def _initial_nuisance_from_estimate(
    initial_estimate: MLEEstimate | None,
) -> NDArray[np.float64] | None:
    """Recombine persisted background and other nuisance coefficients."""
    if initial_estimate is None:
        return None
    return np.concatenate(
        [
            np.asarray(initial_estimate.background_parameters, dtype=float),
            np.asarray(initial_estimate.nuisance_parameters, dtype=float),
        ]
    )


class SurfaceMLEEstimator:
    """Fit count-domain or line-resolved spectral surface intensity maps."""

    def __init__(self, config: MLEConfig) -> None:
        """Store immutable estimator configuration."""
        if not isinstance(config, MLEConfig):
            raise TypeError("config must be an MLEConfig.")
        self.config = config

    def fit(
        self,
        batch: ObservationBatch,
        environment: EnvironmentConfig,
        kernel: ContinuousKernel,
        *,
        obstacle_grid: ObstacleGrid | None = None,
        initial_estimate: MLEEstimate | None = None,
    ) -> MLEEstimate:
        """Fit all history, optionally warm-starting from a prior surface map."""
        if tuple(batch.isotope_names) != tuple(self.config.isotope_names):
            raise ValueError("Observation isotope order must match MLEConfig.isotope_names.")
        kernel.use_gpu = bool(self.config.use_gpu)
        kernel.gpu_device = str(self.config.gpu_device)
        kernel.gpu_dtype = str(self.config.gpu_dtype)
        patches = build_surface_patches(
            environment,
            obstacle_grid,
            self.config.patch_spacing_m,
            obstacle_height_m=float(self.config.obstacle_height_m),
            quadrature_points_per_patch=int(self.config.quadrature_order),
        )
        fit_indices, held_out_indices = _split_fit_indices(
            batch.measurement_count,
            self.config.held_out_fraction,
            self.config.random_seed,
        )
        initial_densities = _initial_density_for_patches(
            initial_estimate,
            patches,
            self.config.isotope_names,
        )
        initial_nuisance = _initial_nuisance_from_estimate(initial_estimate)
        state = _fit_problem(
            batch,
            patches,
            kernel,
            self.config,
            fit_indices,
            held_out_indices,
            initial_densities=initial_densities,
            initial_nuisance=initial_nuisance,
        )
        for _level in range(int(self.config.coarse_to_fine_levels)):
            selected = _refinement_patch_ids(state, self.config.refinement_fraction)
            if not selected:
                break
            refined = refine_surface_patches(
                state.patches,
                selected,
                quadrature_points_per_patch=int(self.config.quadrature_order),
            )
            warm = _warm_start_refined(
                state.patches,
                refined,
                state.result.densities_cps_1m_m2,
            )
            state = _fit_problem(
                batch,
                refined,
                kernel,
                self.config,
                fit_indices,
                held_out_indices,
                initial_densities=warm,
                initial_nuisance=state.result.nuisance_coefficients,
            )

        observed = (
            batch.isotope_counts if self.config.mode == "count" else batch.spectrum_counts
        )
        if observed is None:  # guarded by _fit_problem; keeps static typing explicit
            raise ValueError("Configured observation domain is unavailable.")
        if self.config.debias_refit:
            state = _debias_state(state, observed, self.config)
        predicted = _full_prediction(state)
        held_out_deviance = (
            poisson_deviance(
                observed[held_out_indices],
                predicted[held_out_indices],
                min_mean=float(self.config.min_mean),
            )
            if held_out_indices.size
            else None
        )
        clusters = cluster_surface_hotspots(
            state.patches,
            state.result.densities_cps_1m_m2,
            self.config.isotope_names,
            threshold_fraction=float(self.config.cluster_threshold_fraction),
            min_strength_cps_1m=float(self.config.cluster_min_strength_cps_1m),
        )
        identifiability = response_identifiability_diagnostics(
            state.response[state.fit_indices],
            correlation_threshold=float(self.config.response_correlation_threshold),
        )
        residual = observed - predicted
        diagnostics: dict[str, object] = {
            "mode": self.config.mode,
            "warm_started": initial_estimate is not None,
            "density_unit": "detector_cps_1m_per_m2",
            "patch_strength_unit": "detector_cps_1m",
            "patch_count": state.patches.patch_count,
            "response_shape": [int(value) for value in state.response.shape],
            "fit_measurement_indices": state.fit_indices.astype(int).tolist(),
            "held_out_measurement_indices": state.held_out_indices.astype(int).tolist(),
            "held_out_poisson_deviance": held_out_deviance,
            "residual_l2": float(np.linalg.norm(residual)),
            "residual_by_observation": np.asarray(residual, dtype=float).tolist(),
            "objective_history": _json_floats(
                np.asarray(state.result.objective_history, dtype=float)
            ),
            "poisson_nll": float(state.result.poisson_nll),
            "l1_penalty": float(state.result.l1_penalty),
            "tv_penalty": float(state.result.tv_penalty),
            "group_penalty": float(state.result.group_penalty),
            "nuisance_penalty": float(state.result.nuisance_penalty),
            "relative_change": float(state.result.relative_change),
            "relative_objective_change": float(
                state.result.relative_objective_change
            ),
            "kkt_residual": float(state.result.kkt_residual),
            "nuisance_names": list(state.nuisance_names),
            "isotope_covariance_preserved": batch.isotope_covariances is not None,
            "hotspot_clusters": [cluster.to_dict() for cluster in clusters],
            "identifiability": identifiability,
        }
        if state.spectral_details is not None:
            diagnostics["line_energies_keV_by_isotope"] = {
                key: list(values)
                for key, values in state.spectral_details.line_energies_keV_by_isotope.items()
            }
            diagnostics["line_weights_by_isotope"] = {
                key: list(values)
                for key, values in state.spectral_details.line_weights_by_isotope.items()
            }

        nuisance = state.result.nuisance_coefficients
        background_mask = np.asarray(
            [name.startswith("background") for name in state.nuisance_names],
            dtype=bool,
        )
        background_parameters = nuisance[background_mask]
        other_nuisance = nuisance[~background_mask]
        if self.config.mode == "count":
            predicted_spectra = None
            predicted_isotope_counts = predicted
        else:
            predicted_spectra = predicted
            predicted_isotope_counts = np.einsum(
                "mbgi,gi->mi",
                state.response,
                state.result.integrated_strengths_cps_1m,
                optimize=True,
            )
        return MLEEstimate(
            isotope_names=self.config.isotope_names,
            patches=state.patches.patches,
            density_by_isotope=state.result.densities_cps_1m_m2.T,
            patch_strength_by_isotope=state.result.integrated_strengths_cps_1m.T,
            predicted_spectra=predicted_spectra,
            predicted_isotope_counts=predicted_isotope_counts,
            background_parameters=background_parameters,
            nuisance_parameters=other_nuisance,
            objective_value=float(state.result.objective),
            poisson_deviance=float(state.result.deviance),
            iterations=int(state.result.iterations),
            converged=bool(state.result.converged),
            diagnostics=diagnostics,
        )


def fit_surface_mle(
    batch: ObservationBatch,
    environment: EnvironmentConfig,
    kernel: ContinuousKernel,
    config: MLEConfig,
    *,
    obstacle_grid: ObstacleGrid | None = None,
    initial_estimate: MLEEstimate | None = None,
) -> MLEEstimate:
    """Convenience wrapper for one all-history surface MLE fit."""
    return SurfaceMLEEstimator(config).fit(
        batch,
        environment,
        kernel,
        obstacle_grid=obstacle_grid,
        initial_estimate=initial_estimate,
    )


__all__ = ["SurfaceMLEEstimator", "fit_surface_mle"]
