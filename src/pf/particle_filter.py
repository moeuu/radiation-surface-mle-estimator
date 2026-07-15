"""Coordinate the per-isotope particle filter main loop (predict, update, resample)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List, Mapping, Sequence, Tuple
from collections import deque
import os
import time

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import chi2

from measurement.model import EnvironmentConfig
from measurement.kernels import KernelPrecomputer, ShieldParams
from measurement.continuous_kernels import ContinuousKernel
from measurement.obstacles import ObstacleGrid
from measurement.source_surfaces import (
    build_surface_candidate_sources,
    project_positions_to_allowed_surfaces,
    source_surface_kinds,
)
from pf.defaults import DEFAULT_MAX_SOURCES_PER_ISOTOPE
from pf.diagnostics import build_source_event_record, reset_step_diagnostics
from pf.likelihood import (
    count_log_likelihood,
    count_likelihood_variance,
    count_likelihood_variance_torch,
    delta_log_likelihood_remove,
    delta_log_likelihood_update,
    expected_counts_per_source,
    normalize_count_likelihood_model,
)
from pf.state import IsotopeState
from pf.resampling import systematic_resample, systematic_resample_count

if TYPE_CHECKING:
    import torch


def _pf_debug_timing_enabled() -> bool:
    """Return True when verbose PF phase timing should be printed."""
    return os.environ.get("PF_DEBUG_TIMING", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


@dataclass
class PFConfig:
    """Particle filter configuration (Sec. 3.4)."""

    num_particles: int = 200
    min_particles: int | None = None
    max_particles: int | None = None
    max_sources: int | None = DEFAULT_MAX_SOURCES_PER_ISOTOPE
    resample_threshold: float = 0.5  # relative to N
    position_sigma: float = 0.1
    strength_sigma: float = 0.1
    background_sigma: float = 0.1
    background_level: float | dict[str, float] = 0.0
    measurement_scale_by_isotope: dict[str, float] | None = None
    measurement_scale_by_isotope_and_pair: dict[str, dict[int, float]] | None = None
    count_likelihood_model: str = "poisson"
    transport_model_rel_sigma: float | dict[str, float] = 0.0
    transport_model_abs_sigma: float | dict[str, float] = 0.0
    spectrum_count_rel_sigma: float | dict[str, float] = 0.0
    spectrum_count_abs_sigma: float | dict[str, float] = 0.0
    low_count_abs_sigma: float | dict[str, float] = 0.0
    low_count_transition_counts: float | dict[str, float] = 0.0
    count_likelihood_df: float = 5.0
    shield_contrast_likelihood_enable: bool = False
    shield_contrast_likelihood_weight: float = 1.0
    shield_contrast_log_sigma_floor: float = 0.5
    shield_contrast_log_sigma_ceiling: float = 2.0
    shield_contrast_min_count: float = 25.0
    shield_contrast_min_views: int = 2
    shield_contrast_likelihood_df: float = 5.0
    shield_view_ratio_likelihood_enable: bool = False
    shield_view_ratio_likelihood_weight: float = 1.0
    shield_view_ratio_likelihood_concentration: float = 128.0
    shield_view_ratio_likelihood_min_total_count: float = 25.0
    shield_view_ratio_likelihood_min_views: int = 2
    station_view_covariance_enable: bool = False
    station_view_correlated_spectrum_fraction: float = 0.0
    spectrum_likelihood_bin_chunk: int = 512
    min_strength: float = 0.01
    p_birth: float = 0.05
    p_kill: float = 0.1
    death_low_q_streak: int = 10
    death_delta_ll_threshold: float = 0.0
    support_ema_alpha: float = 0.3
    support_window: int = 1
    birth_window: int = 10
    birth_softmax_temp: float = 1.0
    birth_min_score: float = 1e-12
    birth_enable: bool = True
    birth_topk_particles: int = 10
    birth_use_weighted_topk: bool = True
    birth_min_sep_m: float = 0.8
    birth_detector_min_sep_m: float = 1.0
    source_detector_exclusion_m: float = 0.0
    birth_candidate_jitter_sigma: float = 0.5
    birth_num_local_jitter: int = 8
    birth_alpha: float = 0.2
    birth_q_max: float = 3e5
    birth_q_min: float = 1e2
    birth_max_per_update: int | None = None
    birth_delta_ll_threshold: float = 0.0
    birth_complexity_penalty: float = 0.0
    birth_bic_penalty_params: int = 4
    structural_update_min_counts: float = 0.0
    structural_update_min_snr: float = 0.0
    structural_update_count_min_snr: float = 0.0
    structural_update_max_rel_sigma: float = 0.0
    birth_min_distinct_poses: int = 1
    birth_residual_clip_quantile: float = 0.95
    birth_residual_gate_p_value: float = 0.05
    birth_residual_min_support: int = 2
    birth_residual_support_sigma: float = 1.0
    birth_min_distinct_stations: int = 1
    birth_candidate_support_fraction: float = 0.05
    birth_refit_residual_gate: bool = True
    birth_refit_residual_min_fraction: float = 0.5
    birth_use_shield_coded_residual: bool = True
    birth_existing_response_corr_max: float = 1.0
    birth_response_condition_max: float = 0.0
    birth_count_distance_prior_weight: float = 0.5
    birth_count_distance_strength_weight: float = 0.25
    birth_count_distance_log_clip: float = 3.0
    birth_count_distance_strength_sigma: float = 2.0
    birth_residual_always_try: bool = True
    birth_residual_expand_structural_particles: bool = True
    birth_residual_expanded_structural_topk_particles: int | None = 256
    birth_residual_acceptance_complexity_scale: float = 0.0
    birth_residual_force_proposal_on_gate: bool = True
    birth_residual_forced_min_delta_ll: float = -50.0
    birth_residual_force_relax_candidate_masks: bool = True
    birth_residual_suppress_death: bool = True
    birth_matching_pursuit_max_new_sources: int = 3
    birth_matching_pursuit_topk_candidates: int = 16
    birth_orthogonalize_residual_candidates: bool = False
    birth_orthogonal_candidate_corr_max: float = 0.98
    birth_jitter_topk_candidates: int | None = 512
    birth_global_rescue_enable: bool = False
    birth_global_rescue_max_candidates: int = 8
    birth_global_rescue_min_residual_fraction: float = 0.005
    birth_global_rescue_dedup_radius_m: float = 0.5
    birth_global_rescue_force_proposal_on_gate: bool = False
    birth_global_rescue_forced_min_delta_ll: float = 0.0
    birth_global_rescue_min_support: int | None = None
    birth_global_rescue_min_distinct_poses: int | None = None
    birth_global_rescue_min_distinct_stations: int | None = None
    high_strength_split_enable: bool = True
    high_strength_split_q_multiple: float = 2.0
    high_strength_split_offset_m: float = 1.5
    high_strength_split_candidate_count: int = 12
    runtime_report_rescue_enable: bool = False
    runtime_report_rescue_particle_fraction: float = 0.15
    runtime_report_rescue_min_particles_per_source: int = 4
    runtime_report_rescue_weight: float = 0.10
    runtime_report_rescue_jitter_sigma_m: float = 0.10
    residual_decomposition_enable: bool = True
    peak_suppression_enable: bool = True
    peak_suppression_min_source_fraction: float = 0.25
    peak_suppression_factor: float = 1.0
    residual_decomposition_max_layers: int = 4
    pseudo_source_verification_enable: bool = True
    pseudo_source_min_delta_ll: float = 0.0
    pseudo_source_min_distinct_views: int = 2
    pseudo_source_fail_grace_stations: int = 2
    pseudo_source_corr_max: float = 0.995
    pseudo_source_temporal_sep_min: float = 0.0
    pseudo_source_quarantine_on_suppress: bool = True
    pseudo_source_quarantine_excludes_runtime: bool = False
    report_exclude_unverified_sources: bool = False
    source_prune_min_distinct_stations: int = 2
    source_prune_min_distinct_views: int = 2
    source_prune_fail_grace_stations: int = 2
    source_prune_delta_ll_threshold: float = 0.0
    source_prune_refit_after_remove: bool = True
    source_prune_bic_penalty_params: int = 4
    refit_after_moves: bool = True
    refit_iters: int = 3
    refit_eps: float = 1e-12
    weak_source_prune_min_expected_count: float = 0.0
    weak_source_prune_min_fraction: float = 0.0
    weak_source_prune_min_age: int = 0
    weak_source_prune_require_observable: bool = True
    weak_source_prune_min_observable_measurements: int = 1
    weak_source_prune_observable_count: float = 0.0
    weak_source_prune_observable_fraction: float = 0.0
    weak_source_prune_visibility_reference_strength: float = 0.0
    conditional_strength_refit: bool = True
    conditional_strength_refit_window: int = 10
    conditional_strength_refit_iters: int = 3
    conditional_strength_refit_reweight: bool = False
    conditional_strength_refit_cardinality_neutral_reweight: bool = True
    conditional_strength_refit_reweight_clip: float = 50.0
    conditional_strength_refit_min_count: float = 5.0
    conditional_strength_refit_min_snr: float = 1.0
    conditional_strength_refit_prior_weight: float = 0.0
    conditional_strength_refit_prior_rel_sigma: float = 2.0
    source_strength_prior_mean: float = 0.0
    source_strength_prior_weight: float = 0.0
    source_strength_prior_rel_sigma: float = 1.0
    source_strength_absorption_penalty_weight: float = 0.0
    source_strength_absorption_q_multiple: float = 4.0
    source_strength_observation_overshoot_penalty_weight: float = 0.0
    source_strength_observation_overshoot_sigma: float = 5.0
    source_strength_observation_overshoot_quantile: float = 0.05
    source_strength_observation_overshoot_min_visible_fraction: float = 0.05
    source_strength_observation_overshoot_min_visible_measurements: int = 3
    birth_stage_single_station_as_quarantine: bool = True
    min_age_to_split: int = 5
    use_clustered_output: bool = True
    cluster_eps_m: float = 0.8
    cluster_min_samples: int = 20
    cluster_report_max_points: int = 6000
    cluster_exact_max_points: int = 5000
    split_prob: float = 0.05
    split_strength_min: float = 0.1
    split_position_sigma: float = 0.25
    split_strength_min_frac: float = 0.3
    split_strength_max_frac: float = 0.7
    split_delta_ll_threshold: float = 0.0
    split_complexity_penalty: float = 0.0
    split_residual_guided: bool = True
    split_residual_always_try: bool = False
    split_residual_candidate_count: int = 8
    merge_prob: float = 0.0
    merge_distance_max: float = 0.5
    merge_delta_ll_threshold: float = 0.0
    merge_response_corr_min: float = 0.995
    merge_search_topk_pairs: int = 8
    structural_proposal_topk_particles: int | None = None
    structural_trial_workers: int = 1
    structural_trial_parallel_min_trials: int = 8
    ess_low: float = 0.5
    ess_high: float = 0.9
    target_ess_ratio: float = 0.5
    max_temper_steps: int = 16
    min_delta_beta: float = 1e-3
    use_tempering: bool = True
    max_resamples_per_observation: int = 2
    temper_resample_cooldown_steps: int = 2
    temper_resample_force_ratio: float = 0.1
    disable_regularize_on_temper_resample: bool = False
    deferred_resample_roughening_scale: float = 0.15
    cardinality_preserving_resample: bool = True
    cardinality_preserving_min_stations: int = 0
    cardinality_preserving_require_confirmed_structure: bool = False
    mode_preserving_resample: bool = False
    mode_preserving_max_modes: int = 4
    mode_preserving_particles_per_mode: int = 2
    mode_preserving_radius_m: float = 1.5
    mode_preserving_min_weight_fraction: float = 1e-4
    mode_preserving_surface_strata: bool = True
    mode_preserving_height_bin_m: float = 2.0
    mode_preserving_high_surface_extra_particles: int = 0
    mode_preserving_high_surface_z_fraction: float = 0.75
    mode_preserving_support_score_weight: float = 0.0
    mode_preserving_tentative_boost: float = 1.0
    mode_preserving_residual_boost: float = 1.0
    mode_preserving_cardinality_strata: bool = True
    mode_preserving_min_particles_per_cardinality: int = 2
    mode_preserving_report_cardinality_strata: bool = True
    mode_preserving_report_cardinality_extra_particles: int = 0
    mode_preserving_dynamic_cardinality_allocation: bool = False
    mode_preserving_dynamic_cardinality_extra_particles: int = 0
    mode_preserving_dynamic_cardinality_min_mass: float = 0.02
    mode_preserving_dynamic_cardinality_entropy_min: float = 0.5
    mode_preserving_dynamic_spatial_allocation: bool = False
    mode_preserving_dynamic_spatial_extra_particles: int = 0
    mode_preserving_dynamic_spatial_min_score_fraction: float = 0.005
    adapt_cooldown_steps: int = 0
    # Continuous PF priors (Sec. 3.3.2)
    position_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    position_max: Tuple[float, float, float] = (10.0, 10.0, 10.0)
    source_position_prior: str = "volume"
    roughening_k: float = 0.5
    surface_rejuvenation_enable: bool = True
    min_sigma_pos: float = 0.05
    max_sigma_pos: float = 1.5
    roughening_decay: float = 0.5
    roughening_min_mult: float = 0.25
    init_num_sources: Tuple[int, int] = (0, 3)  # inclusive range
    # Strength prior (cps@1m scale). Defaults cover ~1e3–1e5 cps via log-normal.
    init_strength_log_mean: float = 9.0  # exp(9) ~ 8e3
    init_strength_log_sigma: float = 1.0
    init_grid_spacing_m: float | None = None
    init_grid_repeats: int = 1
    strength_log_sigma: float = 0.3
    use_gpu: bool = True
    gpu_device: str = "cuda"
    gpu_dtype: str = "float32"
    label_alignment_iters: int = 2
    label_pos_weight: float = 1.0
    label_strength_weight: float = 0.2
    label_missing_cost: float = 1e3
    label_pos_scale: float | None = None
    label_strength_scale: float | None = None
    label_enable: bool = True
    converge_enable: bool = False
    converge_window: int = 8
    converge_map_move_eps_m: float = 0.4
    converge_ess_ratio_high: float = 0.2
    converge_ll_improve_eps: float = 1e5
    converge_min_steps: int = 30
    converge_require_all: bool = True
    converge_cardinality_var_max: float = 0.05
    converge_require_no_tentative: bool = True
    converge_freeze_updates: bool = False
    converge_min_stations: int = 0
    converge_cluster_spread_max_m: float = 0.0
    converge_cluster_min_support_fraction: float = 0.0


@dataclass
class PFConvergenceMonitor:
    """Track per-isotope convergence statistics over a sliding window."""

    window: int
    min_steps: int
    map_move_eps_m: float
    ess_ratio_high: float
    ll_improve_eps: float
    require_all: bool

    def __post_init__(self) -> None:
        """Initialize bounded convergence-history buffers."""
        self.positions: deque[NDArray[np.float64] | None] = deque(maxlen=self.window)
        self.ess_ratios: deque[float] = deque(maxlen=self.window)
        self.ll_values: deque[float] = deque(maxlen=self.window)

    def update_stats(
        self,
        step_idx: int,
        pos: NDArray[np.float64] | None,
        ess_ratio: float,
        ll_value: float,
    ) -> None:
        """Append the latest statistics to the window."""
        if step_idx < 0:
            return
        self.positions.append(pos.copy() if pos is not None else None)
        self.ess_ratios.append(float(ess_ratio))
        self.ll_values.append(float(ll_value))

    def is_converged(self, step_idx: int) -> bool:
        """Return True if all convergence criteria are satisfied."""
        if step_idx < self.min_steps:
            return False
        if len(self.positions) < self.window:
            return False
        if any(pos is None for pos in self.positions):
            return False
        pos_list = [pos for pos in self.positions if pos is not None]
        reference_shape = pos_list[0].shape
        if any(pos.shape != reference_shape for pos in pos_list):
            return False
        max_move = 0.0
        for prev, curr in zip(pos_list[:-1], pos_list[1:]):
            max_move = max(max_move, float(np.linalg.norm(curr - prev)))
        move_ok = max_move <= float(self.map_move_eps_m)
        ess_ok = min(self.ess_ratios) >= float(self.ess_ratio_high)
        ll_span = max(self.ll_values) - min(self.ll_values)
        ll_ok = ll_span <= float(self.ll_improve_eps)
        if self.require_all:
            return move_ok and ess_ok and ll_ok
        return sum([move_ok, ess_ok, ll_ok]) >= 2


@dataclass
class IsotopeParticle:
    """Continuous-state particle (Sec. 3.3.2)."""

    state: IsotopeState
    log_weight: float


@dataclass(frozen=True)
class MeasurementData:
    """Bundle measurement arrays for birth/death and split/merge proposals."""

    z_k: NDArray[np.float64]
    observation_variances: NDArray[np.float64]
    detector_positions: NDArray[np.float64]
    fe_indices: NDArray[np.int64]
    pb_indices: NDArray[np.int64]
    live_times: NDArray[np.float64]


@dataclass(frozen=True)
class BirthResidualLayer:
    """Store one residual layer used for residual-driven source birth."""

    name: str
    residual: NDArray[np.float64]


class IsotopeParticleFilter:
    """Per-isotope particle filter (continuous state is the primary mode)."""

    def __init__(
        self,
        isotope: str,
        kernel: KernelPrecomputer | None,
        config: PFConfig | None = None,
        obstacle_grid: ObstacleGrid | None = None,
        obstacle_height_m: float = 2.0,
        obstacle_mu_by_isotope: dict[str, float] | None = None,
        obstacle_buildup_coeff: float = 0.0,
        detector_radius_m: float = 0.0,
        detector_aperture_radius_m: float | None = None,
        detector_aperture_samples: int = 1,
        detector_aperture_sampling: str = "solid_angle_cone",
        source_extent_radius_m: float = 0.0,
        source_extent_samples: int = 1,
        line_mu_by_isotope: dict[str, object] | None = None,
        transport_response_model: dict[str, object] | None = None,
    ) -> None:
        """Initialize particle state, priors, and continuous measurement kernels."""
        self.isotope = isotope
        self.kernel = kernel
        self.config = config or PFConfig()
        self.N = self.config.num_particles
        self.obstacle_grid = obstacle_grid
        self.obstacle_height_m = float(obstacle_height_m)
        self.obstacle_mu_by_isotope = obstacle_mu_by_isotope
        self.obstacle_buildup_coeff = max(float(obstacle_buildup_coeff), 0.0)
        self.detector_radius_m = max(float(detector_radius_m), 0.0)
        if detector_aperture_radius_m is None:
            detector_aperture_radius_m = self.detector_radius_m
        self.detector_aperture_radius_m = max(float(detector_aperture_radius_m), 0.0)
        self.detector_aperture_samples = max(int(detector_aperture_samples), 1)
        self.detector_aperture_sampling = str(detector_aperture_sampling)
        self.source_extent_radius_m = max(float(source_extent_radius_m), 0.0)
        self.source_extent_samples = max(int(source_extent_samples), 1)
        self.line_mu_by_isotope = line_mu_by_isotope
        self.transport_response_model = transport_response_model
        self._surface_candidate_cache: dict[float, NDArray[np.float64]] = {}
        mu_by_isotope = getattr(kernel, "mu_by_isotope", None) if kernel is not None else None
        shield_params = getattr(kernel, "shield_params", ShieldParams()) if kernel is not None else ShieldParams()
        self.continuous_kernel = self._build_continuous_kernel(
            mu_by_isotope=mu_by_isotope,
            shield_params=shield_params,
        )
        self.continuous_particles: List[IsotopeParticle] = []
        self._label_reference: IsotopeState | None = None
        self.last_ess: float | None = None
        self.last_ess_pre: float | None = None
        self.last_ess_post: float | None = None
        self.last_resample_ess = False
        self.last_resample_count = 0
        self.last_birth_count = 0
        self.last_kill_count = 0
        self.last_n_after_adapt: int | None = None
        self.last_temper_steps: list[dict[str, float]] = []
        self.last_temper_resample_count = 0
        self.last_mode_preserved_count = 0
        self.last_mode_preserving_strata_summary: dict[str, float] = {}
        self.last_mode_preserving_selected_strata: list[dict[str, object]] = []
        self.last_mode_preserving_cardinality_summary: dict[str, float] = {}
        self.last_mode_preserving_selected_cardinalities: list[dict[str, object]] = []
        self.last_mode_preserving_dynamic_spatial_summary: list[dict[str, object]] = []
        self._external_protected_cardinalities: set[int] = set()
        self.last_birth_residual_chi2 = 0.0
        self.last_birth_residual_p_value = 1.0
        self.last_birth_residual_support = 0
        self.last_birth_residual_distinct_poses = 0
        self.last_birth_residual_distinct_stations = 0
        self.last_birth_residual_gate_passed = False
        self.last_birth_residual_refit_fraction = 1.0
        self.last_birth_residual_refit_gate_passed = True
        self.last_birth_residual_layer = "none"
        self.last_birth_residual_layer_count = 0
        self.last_birth_forced_attempts = 0
        self.last_birth_forced_accepts = 0
        self.last_birth_forced_mask_relaxations = 0
        self.last_birth_forced_no_candidate = 0
        self.last_birth_forced_rejected = 0
        self.last_birth_forced_best_delta = -np.inf
        self.last_birth_global_rescue_candidates = 0
        self.last_birth_global_rescue_attempts = 0
        self.last_birth_global_rescue_accepts = 0
        self.last_birth_global_rescue_rejected = 0
        self.last_birth_global_rescue_best_delta = -np.inf
        self.last_runtime_report_rescue_candidates = 0
        self.last_runtime_report_rescue_sources = 0
        self.last_runtime_report_rescue_injected = 0
        self.last_runtime_report_rescue_weight = 0.0
        self.last_weak_source_prune_occlusion_protected = 0
        self.last_birth_structural_eligible = 0
        self.last_pseudo_source_verified = 0
        self.last_pseudo_source_failed = 0
        self.last_pseudo_source_pruned = 0
        self.last_pseudo_source_quarantined = 0
        self.last_pseudo_source_quarantine_active = 0
        self.last_pseudo_source_fail_reasons: dict[str, int] = {}
        self.last_source_event_diagnostics: list[dict[str, object]] = []
        self.last_structural_timing_s: dict[str, float] = {}
        self._deferred_resampled_any = False
        self._deferred_ess_min: float | None = None
        self._deferred_convergence_args: tuple[
            int | None,
            NDArray[np.float64],
            int,
            int,
            float,
            float,
        ] | None = None
        self._adapt_cooldown_remaining = 0
        self._resample_count_in_observation = 0
        self._observed_station_labels: set[tuple[float, float]] = set()
        self.is_converged = False
        self.frozen_estimate: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
        self.updates_skipped = 0
        self._converge_monitor = (
            PFConvergenceMonitor(
                window=int(self.config.converge_window),
                min_steps=int(self.config.converge_min_steps),
                map_move_eps_m=float(self.config.converge_map_move_eps_m),
                ess_ratio_high=float(self.config.converge_ess_ratio_high),
                ll_improve_eps=float(self.config.converge_ll_improve_eps),
                require_all=bool(self.config.converge_require_all),
            )
            if self.config.converge_enable
            else None
        )
        self._init_continuous_particles()

    def set_external_protected_cardinalities(
        self,
        cardinalities: Sequence[int] | set[int] | None,
    ) -> None:
        """Set externally supported source counts to preserve during resampling."""
        if cardinalities is None:
            self._external_protected_cardinalities = set()
            return
        protected: set[int] = set()
        for value in cardinalities:
            try:
                count = int(value)
            except (TypeError, ValueError):
                continue
            if count >= 0:
                protected.add(count)
        self._external_protected_cardinalities = protected

    def _build_continuous_kernel(
        self,
        mu_by_isotope: dict[str, object] | None,
        shield_params: ShieldParams,
    ) -> ContinuousKernel:
        """Build the continuous kernel with the filter's environment attenuation settings."""
        kernel_kwargs: dict[str, object] = {}
        orientations = getattr(self.kernel, "orientations", None)
        if orientations is not None and len(orientations) > 1:
            kernel_kwargs["orientations"] = orientations
        return ContinuousKernel(
            mu_by_isotope=mu_by_isotope,
            shield_params=shield_params,
            use_gpu=bool(self.config.use_gpu),
            gpu_device=str(self.config.gpu_device),
            gpu_dtype=str(self.config.gpu_dtype),
            obstacle_grid=self.obstacle_grid,
            obstacle_height_m=self.obstacle_height_m,
            obstacle_mu_by_isotope=self.obstacle_mu_by_isotope,
            obstacle_buildup_coeff=self.obstacle_buildup_coeff,
            detector_radius_m=self.detector_radius_m,
            detector_aperture_radius_m=self.detector_aperture_radius_m,
            detector_aperture_samples=self.detector_aperture_samples,
            detector_aperture_sampling=self.detector_aperture_sampling,
            source_extent_radius_m=self.source_extent_radius_m,
            source_extent_samples=self.source_extent_samples,
            line_mu_by_isotope=self.line_mu_by_isotope,
            transport_response_model=self.transport_response_model,
            **kernel_kwargs,
        )

    def _measurement_source_scale(
        self,
        *,
        fe_index: int | None = None,
        pb_index: int | None = None,
        shield_pair_id: int | None = None,
    ) -> float:
        """Return the isotope-specific source response scale for PF likelihoods."""
        pair_id = self._measurement_shield_pair_id(
            fe_index=fe_index,
            pb_index=pb_index,
            shield_pair_id=shield_pair_id,
        )
        pair_scales = self.config.measurement_scale_by_isotope_and_pair
        if pair_id is not None and isinstance(pair_scales, Mapping):
            isotope_pair_scales = pair_scales.get(str(self.isotope), {})
            if isinstance(isotope_pair_scales, Mapping):
                value = isotope_pair_scales.get(int(pair_id))
                if value is None:
                    value = isotope_pair_scales.get(str(int(pair_id)))  # type: ignore[arg-type]
                if value is not None:
                    return max(float(value), 0.0)
        scales = self.config.measurement_scale_by_isotope
        if not isinstance(scales, dict):
            return 1.0
        return max(float(scales.get(self.isotope, 1.0)), 0.0)

    def _measurement_source_scale_vector(
        self,
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Return one source response scale for each measurement pair."""
        fe_arr = np.asarray(fe_indices, dtype=int).reshape(-1)
        pb_arr = np.asarray(pb_indices, dtype=int).reshape(-1)
        if fe_arr.size != pb_arr.size:
            raise ValueError("fe_indices and pb_indices must have matching length.")
        return np.asarray(
            [
                self._measurement_source_scale(
                    fe_index=int(fe),
                    pb_index=int(pb),
                )
                for fe, pb in zip(fe_arr, pb_arr)
            ],
            dtype=float,
        )

    def _measurement_shield_pair_id(
        self,
        *,
        fe_index: int | None = None,
        pb_index: int | None = None,
        shield_pair_id: int | None = None,
    ) -> int | None:
        """Return the configured shield-pair id when orientation indices exist."""
        if shield_pair_id is not None:
            return int(shield_pair_id)
        if fe_index is None or pb_index is None:
            return None
        orientations = getattr(self.kernel, "orientations", None)
        num_orientations = len(orientations) if orientations is not None else 8
        return int(fe_index) * max(int(num_orientations), 1) + int(pb_index)

    @staticmethod
    def _measurement_vector(
        values: float | NDArray[np.float64],
        count: int,
        name: str,
        *,
        min_value: float | None = None,
        allow_scalar: bool = True,
    ) -> NDArray[np.float64]:
        """Return a validated one-value-per-measurement vector."""
        expected = max(int(count), 0)
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == 0:
            if expected == 0:
                return np.zeros(0, dtype=float)
            raise ValueError(f"{name} must contain one value per measurement.")
        if arr.size == 1 and expected != 1 and allow_scalar:
            arr = np.full(expected, float(arr[0]), dtype=float)
        elif arr.size != expected:
            scalar_text = "scalar or " if allow_scalar else ""
            raise ValueError(
                f"{name} must be {scalar_text}one value per measurement."
            )
        if min_value is not None:
            arr = np.maximum(arr, float(min_value))
        return np.asarray(arr, dtype=float)

    def _reset_structural_residual_gate(self) -> None:
        """Reset birth residual diagnostics when structural updates are skipped."""
        self.last_birth_residual_chi2 = 0.0
        self.last_birth_residual_p_value = 1.0
        self.last_birth_residual_support = 0
        self.last_birth_residual_distinct_poses = 0
        self.last_birth_residual_distinct_stations = 0
        self.last_birth_residual_gate_passed = False

    def _measurement_rows(self, data: MeasurementData, mask: NDArray[np.bool_]) -> MeasurementData:
        """Return a measurement bundle restricted to the selected row mask."""
        row_mask = np.asarray(mask, dtype=bool).reshape(-1)
        return MeasurementData(
            z_k=np.asarray(data.z_k, dtype=float)[row_mask],
            observation_variances=np.asarray(data.observation_variances, dtype=float)[
                row_mask
            ],
            detector_positions=np.asarray(data.detector_positions, dtype=float)[row_mask],
            fe_indices=np.asarray(data.fe_indices, dtype=int)[row_mask],
            pb_indices=np.asarray(data.pb_indices, dtype=int)[row_mask],
            live_times=np.asarray(data.live_times, dtype=float)[row_mask],
        )

    def _structural_evidence_data(
        self,
        data: MeasurementData | None,
    ) -> MeasurementData | None:
        """
        Return rows reliable enough for birth, split, merge, and prune moves.

        Low-count observations still enter the Bayesian weight update.  They are
        removed only from structure-changing proposals because a tiny residual
        can otherwise create or delete same-isotope sources without enough
        statistical support.
        """
        if data is None or data.z_k.size == 0:
            return None
        min_count = max(float(self.config.structural_update_min_counts), 0.0)
        min_snr = max(float(self.config.structural_update_min_snr), 0.0)
        count_min_snr = max(
            float(getattr(self.config, "structural_update_count_min_snr", 0.0)),
            0.0,
        )
        max_rel_sigma = max(
            float(getattr(self.config, "structural_update_max_rel_sigma", 0.0)),
            0.0,
        )
        if min_count <= 0.0 and min_snr <= 0.0:
            return data
        counts = np.asarray(data.z_k, dtype=float).reshape(-1)
        variances = self._measurement_vector(
            data.observation_variances,
            counts.size,
            "observation_variances",
            min_value=1.0e-12,
        )
        count_ok = (
            counts >= min_count
            if min_count > 0.0
            else np.zeros_like(counts, dtype=bool)
        )
        sigma = np.sqrt(variances)
        snr = np.maximum(counts, 0.0) / sigma
        if min_count > 0.0 and count_min_snr > 0.0:
            count_ok &= snr >= count_min_snr
        if min_count > 0.0 and max_rel_sigma > 0.0:
            rel_sigma = sigma / np.maximum(np.maximum(counts, 0.0), 1.0)
            count_ok &= rel_sigma <= max_rel_sigma
        snr_ok = (
            snr >= min_snr
            if min_snr > 0.0
            else np.zeros_like(counts, dtype=bool)
        )
        finite = np.isfinite(counts) & np.isfinite(snr)
        keep = finite & (count_ok | snr_ok)
        if not np.any(keep):
            return None
        return self._measurement_rows(data, keep)

    def _isotope_float_config(self, value: float | dict[str, float], default: float = 0.0) -> float:
        """Resolve a scalar or isotope-indexed float config value."""
        if isinstance(value, dict):
            return max(float(value.get(self.isotope, default)), 0.0)
        return max(float(value), 0.0)

    def _count_likelihood_kwargs(self) -> dict[str, float | str]:
        """Return likelihood keyword arguments for this isotope filter."""
        return {
            "model": str(self.config.count_likelihood_model),
            "transport_model_rel_sigma": self._isotope_float_config(
                self.config.transport_model_rel_sigma,
            ),
            "transport_model_abs_sigma": self._isotope_float_config(
                self.config.transport_model_abs_sigma,
            ),
            "spectrum_count_rel_sigma": self._isotope_float_config(
                self.config.spectrum_count_rel_sigma,
            ),
            "spectrum_count_abs_sigma": self._isotope_float_config(
                self.config.spectrum_count_abs_sigma,
            ),
            "low_count_abs_sigma": self._isotope_float_config(
                self.config.low_count_abs_sigma,
            ),
            "low_count_transition_counts": self._isotope_float_config(
                self.config.low_count_transition_counts,
            ),
            "student_t_df": max(float(self.config.count_likelihood_df), 1.0),
        }

    def _count_log_likelihood_np(
        self,
        z_k: NDArray[np.float64],
        lambda_k: NDArray[np.float64],
        observation_count_variance: float | NDArray[np.float64] = 0.0,
    ) -> float:
        """Evaluate this filter's configured count log-likelihood in NumPy."""
        return count_log_likelihood(
            z_k,
            lambda_k,
            observation_count_variance=observation_count_variance,
            **self._count_likelihood_kwargs(),
        )

    def _count_log_likelihood_matrix_np(
        self,
        z_k: NDArray[np.float64],
        lambda_kp: NDArray[np.float64],
        observation_count_variance: float | NDArray[np.float64] = 0.0,
    ) -> NDArray[np.float64]:
        """Evaluate per-particle count log-likelihoods for a KxP lambda matrix."""
        z_arr = np.asarray(z_k, dtype=float).reshape(-1)
        lam = np.maximum(np.asarray(lambda_kp, dtype=float), 1.0e-12)
        if lam.ndim == 1:
            lam = lam[:, None]
        if lam.shape[0] != z_arr.size:
            raise ValueError("lambda_kp must have one row per measurement.")
        obs_var = np.asarray(observation_count_variance, dtype=float).reshape(-1)
        if obs_var.size == 0:
            obs_var = np.zeros(z_arr.size, dtype=float)
        elif obs_var.size == 1:
            obs_var = np.full(z_arr.size, float(obs_var[0]), dtype=float)
        elif obs_var.size != z_arr.size:
            raise ValueError(
                "observation_count_variance must be scalar or have one value per measurement."
            )
        kwargs = self._count_likelihood_kwargs()
        model = normalize_count_likelihood_model(str(kwargs["model"]))
        z_col = z_arr[:, None]
        if model == "poisson":
            return np.sum(z_col * np.log(lam) - lam, axis=0)
        variance = count_likelihood_variance(
            z_col,
            lam,
            transport_model_rel_sigma=float(kwargs["transport_model_rel_sigma"]),
            transport_model_abs_sigma=float(kwargs["transport_model_abs_sigma"]),
            spectrum_count_rel_sigma=float(kwargs["spectrum_count_rel_sigma"]),
            spectrum_count_abs_sigma=float(kwargs["spectrum_count_abs_sigma"]),
            low_count_abs_sigma=float(kwargs["low_count_abs_sigma"]),
            low_count_transition_counts=float(kwargs["low_count_transition_counts"]),
            observation_count_variance=obs_var[:, None],
        )
        residual = z_col - lam
        if model == "gaussian":
            terms = -0.5 * ((residual**2) / variance + np.log(variance))
            return np.sum(terms, axis=0)
        df = max(float(kwargs["student_t_df"]), 1.0 + 1.0e-12)
        terms = -0.5 * (df + 1.0) * np.log1p((residual**2) / (df * variance))
        terms -= 0.5 * np.log(variance)
        return np.sum(terms, axis=0)

    def _signal_bearing_refit_data(
        self,
        data: MeasurementData,
    ) -> MeasurementData | None:
        """
        Return measurements suitable for conditional strength refitting.

        Censored low-signal observations are still valid PF likelihood updates,
        but using only those upper bounds in a non-negative strength projection
        collapses source rates to the numerical floor.  The deterministic
        strength refit therefore uses only recent measurements with either a
        minimum positive count or a minimum count SNR.
        """
        z_arr = np.asarray(data.z_k, dtype=float).reshape(-1)
        if z_arr.size == 0:
            return None
        variances = self._measurement_vector(
            data.observation_variances,
            z_arr.size,
            "observation_variances",
            min_value=1.0,
        )
        min_count = max(float(self.config.conditional_strength_refit_min_count), 0.0)
        min_snr = max(float(self.config.conditional_strength_refit_min_snr), 0.0)
        if min_count <= 0.0 and min_snr <= 0.0:
            return data
        snr = np.divide(
            np.maximum(z_arr, 0.0),
            np.sqrt(variances),
            out=np.zeros_like(z_arr, dtype=float),
            where=variances > 0.0,
        )
        finite = np.isfinite(z_arr) & np.isfinite(variances)
        mask = finite & ((z_arr >= min_count) | (snr >= min_snr))
        if not np.any(mask):
            return None
        return MeasurementData(
            z_k=data.z_k[mask],
            observation_variances=data.observation_variances[mask],
            detector_positions=data.detector_positions[mask],
            fe_indices=data.fe_indices[mask],
            pb_indices=data.pb_indices[mask],
            live_times=data.live_times[mask],
        )

    def _strength_refit_prior_precision(
        self,
        strengths: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return local quadratic prior precision for strength MAP refits."""
        weight = max(float(self.config.conditional_strength_refit_prior_weight), 0.0)
        if weight <= 0.0:
            return np.zeros_like(np.asarray(strengths, dtype=float), dtype=float)
        rel_sigma = max(
            float(self.config.conditional_strength_refit_prior_rel_sigma),
            1.0e-6,
        )
        floor = max(float(self.config.min_strength), 1.0)
        strengths_arr = np.asarray(strengths, dtype=float)
        scale = np.maximum(np.abs(strengths_arr), floor)
        sigma = rel_sigma * scale
        precision = weight / np.maximum(sigma * sigma, 1.0e-12)
        inactive = np.abs(strengths_arr) <= max(float(self.config.min_strength), 0.0) * (
            1.0 + 1.0e-6
        )
        return np.where(inactive, 0.0, precision)

    def _absolute_strength_prior_terms(
        self,
        shape: tuple[int, ...],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return absolute source-strength prior precision and mean arrays."""
        mean = max(float(self.config.source_strength_prior_mean), 0.0)
        weight = max(float(self.config.source_strength_prior_weight), 0.0)
        if mean <= 0.0 or weight <= 0.0:
            zeros = np.zeros(shape, dtype=float)
            return zeros, zeros
        rel_sigma = max(float(self.config.source_strength_prior_rel_sigma), 1.0e-6)
        sigma = max(rel_sigma * mean, 1.0e-12)
        precision = weight / (sigma * sigma)
        precision_arr = np.full(shape, precision, dtype=float)
        mean_arr = np.full(shape, mean, dtype=float)
        return precision_arr, mean_arr

    def _absolute_strength_prior_log_ratio(
        self,
        prior_strengths: NDArray[np.float64],
        posterior_strengths: NDArray[np.float64],
    ) -> float:
        """Return absolute strength-prior log-density change for one particle."""
        prior = np.asarray(prior_strengths, dtype=float)
        posterior = np.asarray(posterior_strengths, dtype=float)
        if prior.shape != posterior.shape or prior.size == 0:
            return 0.0
        precision, mean = self._absolute_strength_prior_terms(prior.shape)
        if not np.any(precision > 0.0):
            return 0.0
        before = np.sum(precision * (prior - mean) ** 2)
        after = np.sum(precision * (posterior - mean) ** 2)
        return float(-0.5 * (after - before))

    def _absolute_strength_prior_log_ratio_batched(
        self,
        prior_strengths: NDArray[np.float64],
        posterior_strengths: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return absolute strength-prior log-density changes for particles."""
        prior = np.asarray(prior_strengths, dtype=float)
        posterior = np.asarray(posterior_strengths, dtype=float)
        if prior.shape != posterior.shape or prior.size == 0:
            return np.zeros(prior.shape[0] if prior.ndim else 0, dtype=float)
        precision, mean = self._absolute_strength_prior_terms(prior.shape)
        if not np.any(precision > 0.0):
            return np.zeros(prior.shape[0], dtype=float)
        before = np.sum(precision * (prior - mean) ** 2, axis=1)
        after = np.sum(precision * (posterior - mean) ** 2, axis=1)
        return -0.5 * (after - before)

    def _strength_refit_prior_log_ratio(
        self,
        prior_mean: NDArray[np.float64],
        posterior_strengths: NDArray[np.float64],
    ) -> float:
        """
        Return the local strength-prior log-density ratio after a MAP refit.

        The deterministic strength refit is a proposal that moves particle
        strengths while fixed source positions are kept.  When particle weights
        are corrected by a profile likelihood, the same local Gaussian strength
        prior used by the MAP solve must also be included; otherwise particles
        with poor geometry can survive by making an unpenalized jump to an
        extreme source rate.
        """
        prior = np.asarray(prior_mean, dtype=float)
        posterior = np.asarray(posterior_strengths, dtype=float)
        if prior.size == 0 or posterior.size == 0:
            return 0.0
        if prior.shape != posterior.shape:
            raise ValueError("posterior_strengths must match prior_mean.")
        precision = self._strength_refit_prior_precision(prior)
        if precision.shape != prior.shape:
            raise ValueError("strength prior precision must match prior_mean.")
        delta = posterior - prior
        local_ratio = float(-0.5 * np.sum(precision * delta * delta))
        return local_ratio + self._absolute_strength_prior_log_ratio(prior, posterior)

    def _strength_refit_prior_log_ratio_batched(
        self,
        prior_mean: NDArray[np.float64],
        posterior_strengths: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return per-particle local strength-prior log-density ratios."""
        prior = np.asarray(prior_mean, dtype=float)
        posterior = np.asarray(posterior_strengths, dtype=float)
        if prior.size == 0 or posterior.size == 0:
            return np.zeros(prior.shape[0] if prior.ndim else 0, dtype=float)
        if prior.shape != posterior.shape:
            raise ValueError("posterior_strengths must match prior_mean.")
        precision = self._strength_refit_prior_precision(prior)
        if precision.shape != prior.shape:
            raise ValueError("strength prior precision must match prior_mean.")
        delta = posterior - prior
        local_ratio = -0.5 * np.sum(precision * delta * delta, axis=1)
        return local_ratio + self._absolute_strength_prior_log_ratio_batched(
            prior,
            posterior,
        )

    def _delta_log_likelihood_remove(
        self,
        z_k: NDArray[np.float64],
        lambda_total: NDArray[np.float64],
        lambda_m: NDArray[np.float64],
        observation_count_variance: float | NDArray[np.float64] = 0.0,
    ) -> NDArray[np.float64]:
        """Return per-source support using the configured count likelihood."""
        return delta_log_likelihood_remove(
            z_k,
            lambda_total,
            lambda_m,
            observation_count_variance=observation_count_variance,
            **self._count_likelihood_kwargs(),
        )

    def _delta_log_likelihood_update(
        self,
        z_k: NDArray[np.float64],
        lambda_old: NDArray[np.float64],
        lambda_new: NDArray[np.float64],
        observation_count_variance: float | NDArray[np.float64] = 0.0,
    ) -> float:
        """Return proposal support using the configured count likelihood."""
        return delta_log_likelihood_update(
            z_k,
            lambda_old,
            lambda_new,
            observation_count_variance=observation_count_variance,
            **self._count_likelihood_kwargs(),
        )

    def set_kernel(self, kernel: KernelPrecomputer) -> None:
        """Attach a kernel and refresh the continuous-kernel configuration."""
        self.kernel = kernel
        self.continuous_kernel = self._build_continuous_kernel(
            mu_by_isotope=getattr(kernel, "mu_by_isotope", None),
            shield_params=getattr(kernel, "shield_params", ShieldParams()),
        )

    def _source_prior_is_surface(self) -> bool:
        """Return True when source positions should be constrained to surfaces."""
        raw = getattr(self.config, "source_position_prior", "volume")
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in {
            "surface",
            "surfaces",
            "surface_constrained",
            "surface-constrained",
        }

    def _source_prior_environment(self) -> EnvironmentConfig:
        """Return the room geometry used by the source-position prior."""
        hi = np.array(self.config.position_max, dtype=float)
        if hi.shape != (3,):
            raise ValueError("position_max must be a 3-element vector.")
        if np.any(hi <= 0.0):
            raise ValueError("position_max must define positive room dimensions.")
        return EnvironmentConfig(
            size_x=float(hi[0]),
            size_y=float(hi[1]),
            size_z=float(hi[2]),
        )

    def _surface_grid_positions(self, spacing: float) -> NDArray[np.float64]:
        """Return cached source-position candidates on allowed surfaces."""
        key = float(spacing)
        cached = self._surface_candidate_cache.get(key)
        if cached is not None:
            return cached
        candidates = build_surface_candidate_sources(
            self._source_prior_environment(),
            self.obstacle_grid,
            (key, key, key),
            position_min=self.config.position_min,
            position_max=self.config.position_max,
            obstacle_height_m=self.obstacle_height_m,
        )
        self._surface_candidate_cache[key] = candidates
        return candidates

    def _project_positions_to_source_prior(
        self,
        positions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Clip or project source positions according to the configured prior."""
        arr = np.asarray(positions, dtype=float)
        lo = np.array(self.config.position_min, dtype=float)
        hi = np.array(self.config.position_max, dtype=float)
        clipped = np.clip(arr, lo, hi)
        if not self._source_prior_is_surface() or clipped.size == 0:
            return clipped
        projected = project_positions_to_allowed_surfaces(
            clipped,
            self._source_prior_environment(),
            self.obstacle_grid,
            obstacle_height_m=self.obstacle_height_m,
        )
        return np.clip(projected, lo, hi)

    def _sample_prior_positions(self, count: int) -> NDArray[np.float64]:
        """Sample initial source positions from the configured position prior."""
        source_count = max(0, int(count))
        if source_count <= 0:
            return np.zeros((0, 3), dtype=float)
        if self._source_prior_is_surface():
            spacing = self.config.init_grid_spacing_m
            spacing_f = 1.0 if spacing is None else max(float(spacing), 1.0e-6)
            candidates = self._surface_grid_positions(spacing_f)
            replace = candidates.shape[0] < source_count
            indices = np.random.choice(
                candidates.shape[0],
                size=source_count,
                replace=replace,
            )
            return candidates[indices].copy()
        lo = np.array(self.config.position_min, dtype=float)
        hi = np.array(self.config.position_max, dtype=float)
        return lo + np.random.rand(source_count, 3) * (hi - lo)

    def _initial_grid_positions(self) -> NDArray[np.float64]:
        """Return initial grid-center positions when grid init is enabled."""
        spacing = self.config.init_grid_spacing_m
        if spacing is None:
            return np.zeros((0, 3), dtype=float)
        spacing = float(spacing)
        if spacing <= 0.0:
            return np.zeros((0, 3), dtype=float)
        if self._source_prior_is_surface():
            return self._surface_grid_positions(spacing)
        lo = np.array(self.config.position_min, dtype=float)
        hi = np.array(self.config.position_max, dtype=float)
        starts = lo + spacing * 0.5
        xs = np.arange(starts[0], hi[0], spacing)
        ys = np.arange(starts[1], hi[1], spacing)
        zs = np.arange(starts[2], hi[2], spacing)
        if xs.size == 0 or ys.size == 0 or zs.size == 0:
            return np.zeros((0, 3), dtype=float)
        grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
        return grid.reshape(-1, 3)

    def _initial_source_count_for_particle(self, particle_index: int) -> int:
        """Return an initial source count that respects the configured prior range."""
        min_r, max_r = self.config.init_num_sources
        min_r = max(0, int(min_r))
        max_r = max(min_r, int(max_r))
        if self.config.max_sources is not None:
            max_r = min(max_r, max(0, int(self.config.max_sources)))
            min_r = min(min_r, max_r)
        if max_r <= min_r:
            return min_r
        span = max_r - min_r + 1
        return min_r + (int(particle_index) % span)

    def _initial_grid_state_positions(
        self,
        anchor_position: NDArray[np.float64],
        source_count: int,
        grid_positions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return initial source positions for one grid-anchored particle."""
        count = max(0, int(source_count))
        if count <= 0:
            return np.zeros((0, 3), dtype=float)
        anchor = np.asarray(anchor_position, dtype=float).reshape(1, 3)
        if count == 1:
            return anchor.copy()
        grid = np.asarray(grid_positions, dtype=float)
        if grid.ndim != 2 or grid.shape[1] != 3 or grid.shape[0] == 0:
            extra = self._sample_prior_positions(count - 1)
            return np.vstack([anchor, extra])
        replace = grid.shape[0] < count - 1
        extra_idx = np.random.choice(grid.shape[0], size=count - 1, replace=replace)
        return np.vstack([anchor, grid[extra_idx]])

    def _init_continuous_particles(self) -> None:
        """Sample continuous positions/strengths/background from broad priors (Sec. 3.3.2)."""
        self.continuous_particles = []
        grid_positions = self._initial_grid_positions()
        if grid_positions.size:
            repeat_count = max(1, int(self.config.init_grid_repeats))
            target_n = int(grid_positions.shape[0]) * repeat_count
            self.N = target_n
            self.config.num_particles = target_n
            self.config.min_particles = target_n
            self.config.max_particles = target_n
            repeated_positions = np.repeat(grid_positions, repeat_count, axis=0)
            for particle_idx, pos in enumerate(repeated_positions):
                r_h = self._initial_source_count_for_particle(particle_idx)
                if r_h > 0:
                    positions = self._initial_grid_state_positions(
                        anchor_position=pos,
                        source_count=r_h,
                        grid_positions=grid_positions,
                    )
                    strengths = np.random.lognormal(
                        mean=self.config.init_strength_log_mean,
                        sigma=self.config.init_strength_log_sigma,
                        size=r_h,
                    )
                    ages = np.zeros(r_h, dtype=int)
                    low_q_streaks = np.zeros(r_h, dtype=int)
                    support_scores = np.zeros(r_h, dtype=float)
                    tentative_sources = np.zeros(r_h, dtype=bool)
                    verification_fail_streaks = np.zeros(r_h, dtype=int)
                else:
                    positions = np.zeros((0, 3), dtype=float)
                    strengths = np.zeros(0, dtype=float)
                    ages = np.zeros(0, dtype=int)
                    low_q_streaks = np.zeros(0, dtype=int)
                    support_scores = np.zeros(0, dtype=float)
                    tentative_sources = np.zeros(0, dtype=bool)
                    verification_fail_streaks = np.zeros(0, dtype=int)
                b_h = self._background_level()
                st = IsotopeState(
                    num_sources=r_h,
                    positions=positions,
                    strengths=strengths,
                    background=b_h,
                    ages=ages,
                    low_q_streaks=low_q_streaks,
                    support_scores=support_scores,
                    tentative_sources=tentative_sources,
                    verification_fail_streaks=verification_fail_streaks,
                )
                self.continuous_particles.append(
                    IsotopeParticle(state=st, log_weight=float(np.log(1.0 / self.N)))
                )
            return
        min_r, max_r = self.config.init_num_sources
        for _ in range(self.N):
            r_h = int(np.random.randint(min_r, max_r + 1))
            if self.config.max_sources is not None and self.config.max_sources > 0:
                r_h = min(r_h, self.config.max_sources)
            if r_h > 0:
                positions = self._sample_prior_positions(r_h)
                strengths = np.random.lognormal(
                    mean=self.config.init_strength_log_mean, sigma=self.config.init_strength_log_sigma, size=r_h
                )
                ages = np.zeros(r_h, dtype=int)
                low_q_streaks = np.zeros(r_h, dtype=int)
                support_scores = np.zeros(r_h, dtype=float)
                tentative_sources = np.zeros(r_h, dtype=bool)
                verification_fail_streaks = np.zeros(r_h, dtype=int)
            else:
                positions = np.zeros((0, 3), dtype=float)
                strengths = np.zeros(0, dtype=float)
                ages = np.zeros(0, dtype=int)
                low_q_streaks = np.zeros(0, dtype=int)
                support_scores = np.zeros(0, dtype=float)
                tentative_sources = np.zeros(0, dtype=bool)
                verification_fail_streaks = np.zeros(0, dtype=int)
            b_h = self._background_level()
            st = IsotopeState(
                num_sources=r_h,
                positions=positions,
                strengths=strengths,
                background=b_h,
                ages=ages,
                low_q_streaks=low_q_streaks,
                support_scores=support_scores,
                tentative_sources=tentative_sources,
                verification_fail_streaks=verification_fail_streaks,
            )
            self.continuous_particles.append(IsotopeParticle(state=st, log_weight=float(np.log(1.0 / self.N))))

    def reset_step_stats(self) -> None:
        """Reset per-step diagnostic counters."""
        reset_step_diagnostics(self)

    def _advance_adapt_cooldown(self) -> None:
        """Decrement the adapt cooldown counter after each update."""
        if self._adapt_cooldown_remaining > 0:
            self._adapt_cooldown_remaining -= 1

    def _trigger_adapt_cooldown(self) -> None:
        """Start the adapt cooldown after a resampling event."""
        steps = max(0, int(self.config.adapt_cooldown_steps))
        if steps > 0:
            self._adapt_cooldown_remaining = max(self._adapt_cooldown_remaining, steps + 1)

    def _record_source_event(
        self,
        event: str,
        st: IsotopeState,
        source_idx: int,
        *,
        reason: str,
        extra: dict[str, object] | None = None,
    ) -> None:
        """Record a source-slot birth, death, quarantine, or verification event."""
        self._ensure_source_metadata(st)
        record = build_source_event_record(
            event=event,
            isotope=self.isotope,
            state=st,
            source_idx=int(source_idx),
            reason=reason,
            extra=extra,
        )
        if record is None:
            return
        self.last_source_event_diagnostics.append(record)

    def _gpu_enabled(self) -> bool:
        """Return True if GPU computation is enabled and available."""
        from pf import gpu_utils

        if not self.config.use_gpu:
            raise RuntimeError("GPU-only mode: enable use_gpu in PFConfig.")
        if not gpu_utils.torch_device_available(self.config.gpu_device):
            raise RuntimeError("GPU-only mode requires torch on the requested device.")
        return True

    def _can_use_gpu(self) -> bool:
        """Return whether this filter can use the configured torch device."""
        from pf import gpu_utils

        return bool(
            self.config.use_gpu
            and gpu_utils.torch_device_available(self.config.gpu_device)
        )

    def _ll_proxy_pair(
        self,
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        z_obs: float,
    ) -> float:
        """Return a Poisson log-likelihood proxy for convergence checks."""
        if not self.continuous_particles:
            return 0.0
        state = self.best_particle().state
        lam_rate = float(state.background)
        source_scale = self._measurement_source_scale(
            fe_index=fe_index,
            pb_index=pb_index,
        )
        if state.num_sources > 0:
            for pos, strength in zip(state.positions[:state.num_sources], state.strengths[:state.num_sources]):
                kernel_val = self.continuous_kernel.kernel_value_pair(
                    isotope=self.isotope,
                    detector_pos=detector_pos,
                    source_pos=pos,
                    fe_index=fe_index,
                    pb_index=pb_index,
                )
                lam_rate += source_scale * float(kernel_val) * float(strength)
        lam = float(live_time_s) * lam_rate
        return self._count_log_likelihood_np(
            np.array([float(z_obs)], dtype=float),
            np.array([lam], dtype=float),
        )

    def _mmse_primary_position(self) -> NDArray[np.float64] | None:
        """Return the MMSE position for the first source slot, if available."""
        if not self.continuous_particles:
            return None
        weights = np.asarray(self.continuous_weights, dtype=float)
        if weights.size == 0:
            return None
        pos_stack: list[NDArray[np.float64]] = []
        w_stack: list[float] = []
        for weight, particle in zip(weights, self.continuous_particles):
            state = particle.state
            if state.num_sources > 0:
                pos_stack.append(state.positions[0])
                w_stack.append(float(weight))
        if not w_stack:
            return None
        w = np.asarray(w_stack, dtype=float)
        w_sum = float(np.sum(w))
        if w_sum <= 0.0:
            w = np.full_like(w, 1.0 / max(len(w), 1))
        else:
            w = w / w_sum
        pos_arr = np.vstack(pos_stack)
        return np.sum(w[:, None] * pos_arr, axis=0)

    def _cardinality_variance(self) -> float:
        """Return posterior variance of the active source count."""
        if not self.continuous_particles:
            return 0.0
        weights = np.asarray(self.continuous_weights, dtype=float)
        if weights.size != len(self.continuous_particles):
            weights = np.ones(len(self.continuous_particles), dtype=float)
        weights = weights / max(float(np.sum(weights)), 1.0e-12)
        counts = np.asarray(
            [particle.state.num_sources for particle in self.continuous_particles],
            dtype=float,
        )
        mean = float(np.sum(weights * counts))
        return float(np.sum(weights * (counts - mean) ** 2))

    def _has_unverified_sources(self) -> bool:
        """Return True if posterior-supported source hypotheses are tentative."""
        if not self.continuous_particles:
            return False
        weights = np.asarray(self.continuous_weights, dtype=float)
        if weights.size != len(self.continuous_particles):
            weights = np.ones(len(self.continuous_particles), dtype=float)
        weights = weights / max(float(np.sum(weights)), 1.0e-12)
        support_mass = 0.0
        for weight, particle in zip(weights, self.continuous_particles):
            st = particle.state
            self._ensure_source_metadata(st)
            if st.num_sources <= 0:
                continue
            tentative = np.asarray(st.tentative_sources[: st.num_sources], dtype=bool)
            failed = np.asarray(st.verification_fail_streaks[: st.num_sources], dtype=int)
            if np.any(tentative | (failed > 0)):
                support_mass += float(weight)
        return support_mass > 1.0e-3

    def _convergence_state_vector(self) -> NDArray[np.float64] | None:
        """Return a cardinality-aware vector for convergence monitoring."""
        if not self.continuous_particles:
            return None
        if bool(self.config.birth_enable and self.config.use_clustered_output):
            positions, strengths = self.estimate_clustered()
        else:
            positions, strengths = self.estimate()
        if positions.size == 0:
            return None
        order = np.lexsort((positions[:, 2], positions[:, 1], positions[:, 0]))
        pos_sorted = np.asarray(positions[order], dtype=float)
        str_sorted = np.asarray(strengths, dtype=float).reshape(-1)[order]
        strength_scale = max(float(np.max(np.abs(str_sorted))), 1.0)
        return np.concatenate([pos_sorted.reshape(-1), str_sorted / strength_scale])

    def _cluster_convergence_supported(self) -> bool:
        """
        Return True when each reported cluster is locally supported and compact.

        Isotope-level convergence can hide one stable strong source and one
        drifting weak cluster.  This guard keeps updates active until each
        output cluster has enough posterior mass nearby and, when configured, a
        bounded spatial spread.
        """
        max_spread = max(float(self.config.converge_cluster_spread_max_m), 0.0)
        min_support = max(
            float(self.config.converge_cluster_min_support_fraction),
            0.0,
        )
        if max_spread <= 0.0 and min_support <= 0.0:
            return True
        positions, _strengths = (
            self.estimate_clustered()
            if bool(self.config.birth_enable and self.config.use_clustered_output)
            else self.estimate()
        )
        if positions.size == 0:
            return False
        cluster_positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        cluster_count = int(cluster_positions.shape[0])
        support = np.zeros(cluster_count, dtype=float)
        spread_sum = np.zeros(cluster_count, dtype=float)
        weights = np.asarray(self.continuous_weights, dtype=float)
        if weights.size != len(self.continuous_particles):
            weights = np.ones(len(self.continuous_particles), dtype=float)
        total_weight = max(float(np.sum(weights)), 1.0e-12)
        weights = weights / total_weight
        support_radius = max(
            2.0 * max(float(self.config.cluster_eps_m), 1.0e-6),
            max_spread if max_spread > 0.0 else 0.0,
        )
        for weight, particle in zip(weights, self.continuous_particles):
            st = self.state_without_quarantined_sources(particle.state)
            if st.num_sources <= 0:
                continue
            source_positions = np.asarray(st.positions[: st.num_sources], dtype=float)
            distances = np.linalg.norm(
                source_positions[:, None, :] - cluster_positions[None, :, :],
                axis=2,
            )
            nearest = np.min(distances, axis=0)
            supported = nearest <= support_radius
            support[supported] += float(weight)
            spread_sum[supported] += float(weight) * nearest[supported] ** 2
        if min_support > 0.0 and np.any(support < min_support):
            return False
        if max_spread > 0.0:
            if np.any(support <= 0.0):
                return False
            rms = np.sqrt(spread_sum / np.maximum(support, 1.0e-12))
            if np.any(rms > max_spread):
                return False
        return True

    def _distinct_observed_station_count(self) -> int:
        """Return the number of distinct detector stations seen by this filter."""
        return int(len(self._observed_station_labels))

    def _record_observed_station(
        self,
        detector_pos: NDArray[np.float64] | None,
    ) -> None:
        """Record a detector station using rounded XY coordinates."""
        if detector_pos is None:
            return
        pos = np.asarray(detector_pos, dtype=float).reshape(-1)
        if pos.size < 2 or not np.all(np.isfinite(pos[:2])):
            return
        self._observed_station_labels.add(
            (round(float(pos[0]), 3), round(float(pos[1]), 3))
        )

    def _confirmed_source_structure(self) -> bool:
        """Return True when source structure is stable enough to preserve."""
        if self._cardinality_variance() > float(self.config.converge_cardinality_var_max):
            return False
        if bool(self.config.converge_require_no_tentative) and self._has_unverified_sources():
            return False
        if not self._cluster_convergence_supported():
            return False
        return True

    def _convergence_can_freeze(self) -> bool:
        """Return True when no unresolved source structure should keep updating."""
        if not self.config.converge_enable:
            return False
        min_stations = max(0, int(getattr(self.config, "converge_min_stations", 0)))
        if self._distinct_observed_station_count() < min_stations:
            return False
        if not self._confirmed_source_structure():
            return False
        return True

    def _should_skip_converged_update(self) -> bool:
        """Return True when a converged filter can safely ignore more updates."""
        if not (
            self.config.converge_enable
            and self.config.converge_freeze_updates
            and self.is_converged
        ):
            return False
        if self._convergence_can_freeze():
            return True
        self.is_converged = False
        self.frozen_estimate = None
        return False

    def _maybe_update_convergence(
        self,
        step_idx: int | None,
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        z_obs: float,
    ) -> None:
        """Update convergence monitor and freeze if criteria are met."""
        if not self.config.converge_enable or self._converge_monitor is None:
            return
        if step_idx is None:
            return
        if not self.continuous_particles:
            return
        summary_vec = self._convergence_state_vector()
        ess_pre = self.last_ess_pre
        if ess_pre is None:
            w = self.continuous_weights
            ess_pre = float(1.0 / max(np.sum(w**2), 1e-12)) if w.size else 0.0
        ess_ratio = float(ess_pre) / max(len(self.continuous_particles), 1)
        ll_value = self._ll_proxy_pair(
            detector_pos=detector_pos,
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=live_time_s,
            z_obs=z_obs,
        )
        self._converge_monitor.update_stats(step_idx, summary_vec, ess_ratio, ll_value)
        if self._converge_monitor.is_converged(step_idx) and self._convergence_can_freeze():
            self.is_converged = True
            self.frozen_estimate = (
                self.estimate_clustered()
                if bool(self.config.birth_enable and self.config.use_clustered_output)
                else self.estimate()
            )

    def _continuous_expected_counts_torch(
        self, pose_idx: int, orient_idx: int, live_time_s: float
    ) -> "torch.Tensor":
        """Compute Λ_{k,h}^{(n)} using torch for a single orientation index."""
        if self.kernel is None:
            from pf import gpu_utils

            device = gpu_utils.resolve_device(self.config.gpu_device)
            dtype = gpu_utils.resolve_dtype(self.config.gpu_dtype)
            import torch

            return torch.zeros(0, device=device, dtype=dtype)
        orient_vec = self.kernel.orientations[orient_idx]
        octant_idx = self.continuous_kernel.orient_index_from_vector(orient_vec)
        return self._continuous_expected_counts_pair_torch(
            pose_idx=pose_idx,
            fe_index=octant_idx,
            pb_index=octant_idx,
            live_time_s=live_time_s,
        )

    def _continuous_expected_counts_pair_torch(
        self, pose_idx: int, fe_index: int, pb_index: int, live_time_s: float
    ) -> "torch.Tensor":
        """Compute Λ_{k,h}^{(n)} using torch for Fe/Pb orientation indices."""
        from pf import gpu_utils
        import torch

        device = gpu_utils.resolve_device(self.config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.config.gpu_dtype)
        if not self.continuous_particles or self.kernel is None:
            return torch.zeros(0, device=device, dtype=dtype)
        states = [p.state for p in self.continuous_particles]
        positions, strengths, backgrounds, mask = gpu_utils.pack_states(states, device=device, dtype=dtype)
        detector_pos = np.asarray(self.kernel.poses[pose_idx], dtype=float)
        (
            packed_positions,
            packed_strengths,
            packed_mask,
            inverse,
        ) = self._compress_identical_packed_sources_torch(
            positions,
            strengths,
            mask,
        )
        if inverse is None:
            return self.continuous_kernel.expected_counts_pair_for_packed_states_torch(
                isotope=self.isotope,
                detector_pos=detector_pos,
                positions=positions,
                strengths=strengths,
                backgrounds=backgrounds,
                mask=mask,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
                source_scale=self._measurement_source_scale(
                    fe_index=fe_index,
                    pb_index=pb_index,
                ),
                device=device,
                dtype=dtype,
            )
        zero_backgrounds = torch.zeros(
            int(packed_positions.shape[0]),
            device=device,
            dtype=dtype,
        )
        source_counts = self.continuous_kernel.expected_counts_pair_for_packed_states_torch(
            isotope=self.isotope,
            detector_pos=detector_pos,
            positions=packed_positions,
            strengths=packed_strengths,
            backgrounds=zero_backgrounds,
            mask=packed_mask,
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=live_time_s,
            source_scale=self._measurement_source_scale(
                fe_index=fe_index,
                pb_index=pb_index,
            ),
            device=device,
            dtype=dtype,
        )
        return source_counts.index_select(0, inverse) + float(live_time_s) * backgrounds

    def _continuous_expected_counts_pair_sequence_torch(
        self,
        pose_idx: int,
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
        live_times_s: NDArray[np.float64],
    ) -> "torch.Tensor":
        """Compute expected counts for a same-station shield sequence in one GPU batch."""
        from pf import gpu_utils
        import torch

        device = gpu_utils.resolve_device(self.config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.config.gpu_dtype)
        fe_arr = np.asarray(fe_indices, dtype=int).reshape(-1)
        pb_arr = np.asarray(pb_indices, dtype=int).reshape(-1)
        live_arr = np.asarray(live_times_s, dtype=float).reshape(-1)
        if not (fe_arr.size == pb_arr.size == live_arr.size):
            raise ValueError(
                "Fe, Pb, and live-time arrays must have matching lengths."
            )
        if fe_arr.size == 0 or not self.continuous_particles or self.kernel is None:
            return torch.zeros((0, 0), device=device, dtype=dtype)
        states = [p.state for p in self.continuous_particles]
        positions, strengths, backgrounds, mask = gpu_utils.pack_states(
            states,
            device=device,
            dtype=dtype,
        )
        detector_pos = np.asarray(self.kernel.poses[pose_idx], dtype=float)
        (
            packed_positions,
            packed_strengths,
            packed_mask,
            inverse,
        ) = self._compress_identical_packed_sources_torch(
            positions,
            strengths,
            mask,
        )
        if inverse is None:
            unit_live_counts = (
                self.continuous_kernel.expected_counts_selected_pairs_for_packed_states_torch(
                    isotope=self.isotope,
                    detector_pos=detector_pos,
                    positions=positions,
                    strengths=strengths,
                    backgrounds=backgrounds,
                    mask=mask,
                    fe_indices=fe_arr,
                    pb_indices=pb_arr,
                    live_time_s=1.0,
                    source_scale=self._measurement_source_scale_vector(fe_arr, pb_arr),
                    device=device,
                    dtype=dtype,
                )
            )
        else:
            zero_backgrounds = torch.zeros(
                int(packed_positions.shape[0]),
                device=device,
                dtype=dtype,
            )
            source_unit_counts = (
                self.continuous_kernel.expected_counts_selected_pairs_for_packed_states_torch(
                    isotope=self.isotope,
                    detector_pos=detector_pos,
                    positions=packed_positions,
                    strengths=packed_strengths,
                    backgrounds=zero_backgrounds,
                    mask=packed_mask,
                    fe_indices=fe_arr,
                    pb_indices=pb_arr,
                    live_time_s=1.0,
                    source_scale=self._measurement_source_scale_vector(fe_arr, pb_arr),
                    device=device,
                    dtype=dtype,
                )
            )
            unit_live_counts = source_unit_counts.index_select(1, inverse)
            unit_live_counts = unit_live_counts + backgrounds.unsqueeze(0)
        live_t = torch.as_tensor(live_arr, device=device, dtype=dtype).view(-1, 1)
        return unit_live_counts * live_t

    def _compress_identical_packed_sources_torch(
        self,
        positions: "torch.Tensor",
        strengths: "torch.Tensor",
        mask: "torch.Tensor",
    ) -> tuple[
        "torch.Tensor",
        "torch.Tensor",
        "torch.Tensor",
        "torch.Tensor | None",
    ]:
        """Return one source row plus inverse when all packed source states match."""
        import torch

        particle_count = int(positions.shape[0])
        if particle_count <= 1:
            return positions, strengths, mask, None
        if positions.ndim != 3 or strengths.ndim != 2 or mask.ndim != 2:
            return positions, strengths, mask, None
        flat = torch.cat(
            (
                positions.reshape(particle_count, -1),
                strengths.reshape(particle_count, -1),
                mask.reshape(particle_count, -1),
            ),
            dim=1,
        )
        if flat.numel() == 0:
            return positions, strengths, mask, None
        identical = bool(torch.all(flat == flat[:1]).detach().cpu().item())
        if not identical:
            return positions, strengths, mask, None
        inverse = torch.zeros(particle_count, device=positions.device, dtype=torch.long)
        return positions[:1], strengths[:1], mask[:1], inverse

    def _continuous_expected_counts_pair_sequence_torch_uncompressed(
        self,
        pose_idx: int,
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
        live_times_s: NDArray[np.float64],
    ) -> "torch.Tensor":
        """Compute same-station expected counts without duplicate-state compression."""
        from pf import gpu_utils
        import torch

        device = gpu_utils.resolve_device(self.config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.config.gpu_dtype)
        fe_arr = np.asarray(fe_indices, dtype=int).reshape(-1)
        pb_arr = np.asarray(pb_indices, dtype=int).reshape(-1)
        live_arr = np.asarray(live_times_s, dtype=float).reshape(-1)
        if not (fe_arr.size == pb_arr.size == live_arr.size):
            raise ValueError(
                "Fe, Pb, and live-time arrays must have matching lengths."
            )
        if fe_arr.size == 0 or not self.continuous_particles or self.kernel is None:
            return torch.zeros((0, 0), device=device, dtype=dtype)
        states = [p.state for p in self.continuous_particles]
        positions, strengths, backgrounds, mask = gpu_utils.pack_states(
            states,
            device=device,
            dtype=dtype,
        )
        detector_pos = np.asarray(self.kernel.poses[pose_idx], dtype=float)
        unit_live_counts = (
            self.continuous_kernel.expected_counts_selected_pairs_for_packed_states_torch(
                isotope=self.isotope,
                detector_pos=detector_pos,
                positions=positions,
                strengths=strengths,
                backgrounds=backgrounds,
                mask=mask,
                fe_indices=fe_arr,
                pb_indices=pb_arr,
                live_time_s=1.0,
                source_scale=self._measurement_source_scale_vector(fe_arr, pb_arr),
                device=device,
                dtype=dtype,
            )
        )
        live_t = torch.as_tensor(live_arr, device=device, dtype=dtype).view(-1, 1)
        return unit_live_counts * live_t

    def _current_log_weights_torch(self, device: "torch.device") -> "torch.Tensor":
        """Return log-weights as a float64 torch tensor on the requested device."""
        import torch

        return torch.as_tensor(
            [p.log_weight for p in self.continuous_particles],
            device=device,
            dtype=torch.float64,
        )

    def _log_likelihood_increment_gpu(
        self,
        lam_t: "torch.Tensor",
        z_obs: float,
        observation_count_variance: float = 0.0,
    ) -> "torch.Tensor":
        """Return the per-particle count log-likelihood increment in float64."""
        import torch

        lam_t = lam_t.to(dtype=torch.float64)
        lam_t = torch.clamp(lam_t, min=1e-12)
        z = torch.as_tensor(z_obs, device=lam_t.device, dtype=torch.float64)
        model = normalize_count_likelihood_model(str(self.config.count_likelihood_model))
        if model in {"poisson", ""}:
            return z * torch.log(lam_t) - lam_t

        transport_rel = self._isotope_float_config(self.config.transport_model_rel_sigma)
        transport_abs = self._isotope_float_config(self.config.transport_model_abs_sigma)
        spectrum_rel = self._isotope_float_config(self.config.spectrum_count_rel_sigma)
        spectrum_abs = self._isotope_float_config(self.config.spectrum_count_abs_sigma)
        low_count_abs = self._isotope_float_config(self.config.low_count_abs_sigma)
        low_count_transition = self._isotope_float_config(self.config.low_count_transition_counts)
        obs_var = max(float(observation_count_variance), 0.0)
        variance = count_likelihood_variance_torch(
            z,
            lam_t,
            transport_model_rel_sigma=float(transport_rel),
            transport_model_abs_sigma=float(transport_abs),
            spectrum_count_rel_sigma=float(spectrum_rel),
            spectrum_count_abs_sigma=float(spectrum_abs),
            low_count_abs_sigma=float(low_count_abs),
            low_count_transition_counts=float(low_count_transition),
            observation_count_variance=obs_var,
        )
        residual = z - lam_t
        if model == "gaussian":
            return -0.5 * ((residual**2) / variance + torch.log(variance))

        df = max(float(self.config.count_likelihood_df), 1.0 + 1e-12)
        return -0.5 * (df + 1.0) * torch.log1p((residual**2) / (df * variance)) - 0.5 * torch.log(variance)

    def _log_likelihood_sequence_gpu(
        self,
        lam_kn: "torch.Tensor",
        z_obs: NDArray[np.float64],
        observation_count_variances: NDArray[np.float64],
        observation_count_covariance: NDArray[np.float64] | None = None,
    ) -> "torch.Tensor":
        """Return summed per-particle log-likelihoods for a measurement sequence."""
        import torch

        lam = lam_kn.to(dtype=torch.float64)
        if lam.ndim != 2:
            raise ValueError("Sequence expected counts must have shape K x N.")
        z_arr = np.asarray(z_obs, dtype=float).reshape(-1)
        var_arr = np.asarray(observation_count_variances, dtype=float).reshape(-1)
        if z_arr.size != int(lam.shape[0]) or var_arr.size != int(lam.shape[0]):
            raise ValueError("Observation arrays must match the sequence length.")
        if lam.numel() == 0:
            return torch.zeros(0, device=lam.device, dtype=torch.float64)
        lam = torch.clamp(lam, min=1e-12)
        z = torch.as_tensor(
            z_arr,
            device=lam.device,
            dtype=torch.float64,
        ).view(-1, 1)
        model = normalize_count_likelihood_model(
            str(self.config.count_likelihood_model)
        )
        if model in {"poisson", ""}:
            count_ll = torch.sum(z * torch.log(lam) - lam, dim=0)
            return count_ll + self._shield_shape_sequence_log_likelihood_gpu(
                lam,
                z_arr,
                var_arr,
            )
        if self._sequence_covariance_enabled(
            z_arr.size,
            observation_count_covariance,
        ):
            return self._log_likelihood_sequence_covariance_gpu(
                lam,
                z_arr,
                var_arr,
                observation_count_covariance=observation_count_covariance,
            )

        transport_rel = self._isotope_float_config(
            self.config.transport_model_rel_sigma
        )
        transport_abs = self._isotope_float_config(
            self.config.transport_model_abs_sigma
        )
        spectrum_rel = self._isotope_float_config(
            self.config.spectrum_count_rel_sigma
        )
        spectrum_abs = self._isotope_float_config(
            self.config.spectrum_count_abs_sigma
        )
        low_count_abs = self._isotope_float_config(self.config.low_count_abs_sigma)
        low_count_transition = self._isotope_float_config(
            self.config.low_count_transition_counts
        )
        obs_var = torch.clamp(
            torch.as_tensor(
                var_arr,
                device=lam.device,
                dtype=torch.float64,
            ).view(-1, 1),
            min=0.0,
        )
        variance = count_likelihood_variance_torch(
            z,
            lam,
            transport_model_rel_sigma=float(transport_rel),
            transport_model_abs_sigma=float(transport_abs),
            spectrum_count_rel_sigma=float(spectrum_rel),
            spectrum_count_abs_sigma=float(spectrum_abs),
            low_count_abs_sigma=float(low_count_abs),
            low_count_transition_counts=float(low_count_transition),
            observation_count_variance=obs_var,
        )
        residual = z - lam
        if model == "gaussian":
            ll = -0.5 * ((residual**2) / variance + torch.log(variance))
            count_ll = torch.sum(ll, dim=0)
            return count_ll + self._shield_shape_sequence_log_likelihood_gpu(
                lam,
                z_arr,
                var_arr,
            )

        df = max(float(self.config.count_likelihood_df), 1.0 + 1e-12)
        ll = (
            -0.5
            * (df + 1.0)
            * torch.log1p((residual**2) / (df * variance))
            - 0.5 * torch.log(variance)
        )
        count_ll = torch.sum(ll, dim=0)
        return count_ll + self._shield_shape_sequence_log_likelihood_gpu(
            lam,
            z_arr,
            var_arr,
        )

    def _sequence_covariance_enabled(
        self,
        sequence_length: int,
        observation_count_covariance: NDArray[np.float64] | None,
    ) -> bool:
        """Return whether same-station view covariance should be evaluated."""
        if int(sequence_length) < 2:
            return False
        if observation_count_covariance is not None:
            return True
        return bool(self.config.station_view_covariance_enable) and (
            float(self.config.station_view_correlated_spectrum_fraction) > 0.0
        )

    def _spectral_bin_sequence_log_likelihood_gpu(
        self,
        expected_spectrum_kbn: "torch.Tensor",
        observed_spectrum_kb: NDArray[np.float64],
        observation_spectrum_variance_kb: NDArray[np.float64] | None = None,
    ) -> "torch.Tensor":
        """Return direct spectrum-bin log likelihoods for each particle."""
        import torch

        expected = expected_spectrum_kbn.to(dtype=torch.float64)
        if expected.ndim == 2:
            expected = expected.unsqueeze(0)
        if expected.ndim != 3:
            raise ValueError("expected_spectrum_kbn must have shape K x B x N.")
        observed_arr = np.asarray(observed_spectrum_kb, dtype=float)
        if observed_arr.ndim == 1:
            observed_arr = observed_arr.reshape(1, -1)
        if observed_arr.shape != tuple(expected.shape[:2]):
            raise ValueError("observed_spectrum_kb must match K x B expected bins.")
        expected = torch.clamp(expected, min=1.0e-12)
        observed = torch.as_tensor(
            np.maximum(np.where(np.isfinite(observed_arr), observed_arr, 0.0), 0.0),
            device=expected.device,
            dtype=torch.float64,
        ).unsqueeze(-1)
        model = normalize_count_likelihood_model(str(self.config.count_likelihood_model))
        if model in {"poisson", ""} and observation_spectrum_variance_kb is None:
            return torch.sum(observed * torch.log(expected) - expected, dim=(0, 1))

        if observation_spectrum_variance_kb is None:
            obs_var = torch.zeros_like(observed)
        else:
            obs_var_arr = np.asarray(observation_spectrum_variance_kb, dtype=float)
            if obs_var_arr.ndim == 1:
                obs_var_arr = obs_var_arr.reshape(1, -1)
            if obs_var_arr.shape != tuple(expected.shape[:2]):
                raise ValueError("observation_spectrum_variance_kb must match K x B.")
            obs_var = torch.as_tensor(
                np.maximum(np.where(np.isfinite(obs_var_arr), obs_var_arr, 0.0), 0.0),
                device=expected.device,
                dtype=torch.float64,
            ).unsqueeze(-1)
        spectrum_rel = self._isotope_float_config(self.config.spectrum_count_rel_sigma)
        spectrum_abs = self._isotope_float_config(self.config.spectrum_count_abs_sigma)
        variance = torch.clamp(
            expected
            + obs_var
            + (float(spectrum_rel) * expected) ** 2
            + float(spectrum_abs) ** 2,
            min=1.0e-12,
        )
        residual = observed - expected
        if model == "gaussian":
            ll = -0.5 * ((residual**2) / variance + torch.log(variance))
        else:
            df = max(float(self.config.count_likelihood_df), 1.0 + 1.0e-12)
            ll = (
                -0.5
                * (df + 1.0)
                * torch.log1p((residual**2) / (df * variance))
                - 0.5 * torch.log(variance)
        )
        return torch.sum(ll, dim=(0, 1))

    @staticmethod
    def _spectrum_update_arrays(
        observed_spectrum: NDArray[np.float64] | None,
        response_template: NDArray[np.float64] | None,
        background_spectrum: NDArray[np.float64] | None,
        spectrum_variance: NDArray[np.float64] | None,
        *,
        sequence_length: int,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64] | None,
    ] | None:
        """Return validated spectrum-bin arrays for a PF likelihood update."""
        if observed_spectrum is None or response_template is None:
            return None
        rows = max(1, int(sequence_length))
        observed = np.asarray(observed_spectrum, dtype=float)
        if observed.ndim == 1:
            observed = observed.reshape(1, -1)
        template = np.asarray(response_template, dtype=float)
        if template.ndim == 1:
            template = np.broadcast_to(template.reshape(1, -1), observed.shape)
        background = (
            np.zeros_like(observed, dtype=float)
            if background_spectrum is None
            else np.asarray(background_spectrum, dtype=float)
        )
        if background.ndim == 1:
            background = np.broadcast_to(background.reshape(1, -1), observed.shape)
        variance = None
        if spectrum_variance is not None:
            variance = np.asarray(spectrum_variance, dtype=float)
            if variance.ndim == 1:
                variance = np.broadcast_to(variance.reshape(1, -1), observed.shape)
        if observed.shape[0] != rows:
            raise ValueError("observed_spectrum row count must match sequence length.")
        if template.shape != observed.shape:
            raise ValueError("response_template must match observed spectrum bins.")
        if background.shape != observed.shape:
            raise ValueError("background_spectrum must match observed spectrum bins.")
        if variance is not None and variance.shape != observed.shape:
            raise ValueError("spectrum_variance must match observed spectrum bins.")
        observed = np.maximum(np.where(np.isfinite(observed), observed, 0.0), 0.0)
        template = np.maximum(np.where(np.isfinite(template), template, 0.0), 0.0)
        background = np.maximum(
            np.where(np.isfinite(background), background, 0.0),
            0.0,
        )
        if variance is not None:
            variance = np.maximum(np.where(np.isfinite(variance), variance, 0.0), 0.0)
        return observed, template, background, variance

    def _spectral_bin_sequence_log_likelihood_from_lambda_gpu(
        self,
        lam_kn: "torch.Tensor",
        observed_spectrum_kb: NDArray[np.float64],
        response_template_kb: NDArray[np.float64],
        background_spectrum_kb: NDArray[np.float64],
        observation_spectrum_variance_kb: NDArray[np.float64] | None = None,
    ) -> "torch.Tensor":
        """Return direct spectrum-bin log likelihoods without a full KxBxN tensor."""
        import torch

        lam = lam_kn.to(dtype=torch.float64)
        if lam.ndim == 1:
            lam = lam.view(1, -1)
        if lam.ndim != 2:
            raise ValueError("lam_kn must have shape K x N.")
        observed_arr = np.asarray(observed_spectrum_kb, dtype=float)
        if observed_arr.ndim == 1:
            observed_arr = observed_arr.reshape(1, -1)
        if observed_arr.ndim != 2 or observed_arr.shape[0] != int(lam.shape[0]):
            raise ValueError("observed_spectrum_kb rows must match lambda rows.")
        template_arr = np.asarray(response_template_kb, dtype=float)
        if template_arr.ndim == 1:
            template_arr = np.broadcast_to(
                template_arr.reshape(1, -1),
                observed_arr.shape,
            )
        background_arr = np.asarray(background_spectrum_kb, dtype=float)
        if background_arr.ndim == 1:
            background_arr = np.broadcast_to(
                background_arr.reshape(1, -1),
                observed_arr.shape,
            )
        if template_arr.shape != observed_arr.shape:
            raise ValueError("response_template_kb must match observed spectrum bins.")
        if background_arr.shape != observed_arr.shape:
            raise ValueError("background_spectrum_kb must match observed spectrum bins.")
        variance_arr = None
        if observation_spectrum_variance_kb is not None:
            variance_arr = np.asarray(observation_spectrum_variance_kb, dtype=float)
            if variance_arr.ndim == 1:
                variance_arr = np.broadcast_to(
                    variance_arr.reshape(1, -1),
                    observed_arr.shape,
                )
            if variance_arr.shape != observed_arr.shape:
                raise ValueError("observation_spectrum_variance_kb must match bins.")
        observed_all = torch.as_tensor(
            np.maximum(np.where(np.isfinite(observed_arr), observed_arr, 0.0), 0.0),
            device=lam.device,
            dtype=torch.float64,
        )
        template_all = torch.as_tensor(
            np.maximum(np.where(np.isfinite(template_arr), template_arr, 0.0), 0.0),
            device=lam.device,
            dtype=torch.float64,
        )
        background_all = torch.as_tensor(
            np.maximum(
                np.where(np.isfinite(background_arr), background_arr, 0.0),
                0.0,
            ),
            device=lam.device,
            dtype=torch.float64,
        )
        variance_all = None
        if variance_arr is not None:
            variance_all = torch.as_tensor(
                np.maximum(np.where(np.isfinite(variance_arr), variance_arr, 0.0), 0.0),
                device=lam.device,
                dtype=torch.float64,
            )
        model = normalize_count_likelihood_model(str(self.config.count_likelihood_model))
        rel_sigma = self._isotope_float_config(self.config.spectrum_count_rel_sigma)
        abs_sigma = self._isotope_float_config(self.config.spectrum_count_abs_sigma)
        chunk = max(1, int(getattr(self.config, "spectrum_likelihood_bin_chunk", 512)))
        bin_count = int(observed_all.shape[1])
        total_ll = torch.zeros(int(lam.shape[1]), device=lam.device, dtype=torch.float64)
        for start in range(0, bin_count, chunk):
            stop = min(start + chunk, bin_count)
            expected = torch.clamp(
                lam[:, None, :] * template_all[:, start:stop, None]
                + background_all[:, start:stop, None],
                min=1.0e-12,
            )
            observed = observed_all[:, start:stop, None]
            if model in {"poisson", ""} and variance_all is None:
                total_ll = total_ll + torch.sum(
                    observed * torch.log(expected) - expected,
                    dim=(0, 1),
                )
                continue
            if variance_all is None:
                obs_var = torch.zeros_like(observed)
            else:
                obs_var = variance_all[:, start:stop, None]
            variance = torch.clamp(
                expected
                + obs_var
                + (float(rel_sigma) * expected) ** 2
                + float(abs_sigma) ** 2,
                min=1.0e-12,
            )
            residual = observed - expected
            if model == "gaussian":
                ll = -0.5 * ((residual**2) / variance + torch.log(variance))
            else:
                df = max(float(self.config.count_likelihood_df), 1.0 + 1.0e-12)
                ll = (
                    -0.5
                    * (df + 1.0)
                    * torch.log1p((residual**2) / (df * variance))
                    - 0.5 * torch.log(variance)
                )
            total_ll = total_ll + torch.sum(ll, dim=(0, 1))
        return total_ll

    def _expected_spectrum_sequence_torch(
        self,
        lam_kn: "torch.Tensor",
        response_template_kb: NDArray[np.float64],
        background_spectrum_kb: NDArray[np.float64],
    ) -> "torch.Tensor":
        """Return KxBxN expected spectra from particle count rates."""
        import torch

        lam = lam_kn.to(dtype=torch.float64)
        if lam.ndim == 1:
            lam = lam.view(1, -1)
        template = torch.as_tensor(
            response_template_kb,
            device=lam.device,
            dtype=torch.float64,
        )
        background = torch.as_tensor(
            background_spectrum_kb,
            device=lam.device,
            dtype=torch.float64,
        )
        if template.ndim == 1:
            template = template.view(1, -1)
        if background.ndim == 1:
            background = background.view(1, -1)
        if template.shape[0] != lam.shape[0] or background.shape[0] != lam.shape[0]:
            raise ValueError("spectrum template rows must match lambda rows.")
        expected = lam[:, None, :] * template[:, :, None] + background[:, :, None]
        return torch.clamp(expected, min=1.0e-12)

    def _log_likelihood_sequence_covariance_gpu(
        self,
        lam_kn: "torch.Tensor",
        z_obs: NDArray[np.float64],
        observation_count_variances: NDArray[np.float64],
        *,
        observation_count_covariance: NDArray[np.float64] | None,
    ) -> "torch.Tensor":
        """Return a batched multivariate count likelihood for a shield sequence."""
        import torch

        lam = torch.clamp(lam_kn.to(dtype=torch.float64), min=1.0e-12)
        z_arr = np.asarray(z_obs, dtype=float).reshape(-1)
        var_arr = np.asarray(observation_count_variances, dtype=float).reshape(-1)
        if z_arr.size != int(lam.shape[0]) or var_arr.size != int(lam.shape[0]):
            raise ValueError("Observation arrays must match the sequence length.")
        z = torch.as_tensor(
            z_arr,
            device=lam.device,
            dtype=torch.float64,
        ).view(-1, 1)
        variance = self._sequence_diagonal_variance_torch(lam, z, var_arr)
        covariance_nkk = torch.diag_embed(variance.T)
        covariance_nkk = covariance_nkk + self._sequence_observation_offdiag_torch(
            observation_count_covariance,
            sequence_length=z_arr.size,
            device=lam.device,
        )
        covariance_nkk = covariance_nkk + self._sequence_common_mode_offdiag_torch(
            lam,
        )
        covariance_nkk = self._regularize_sequence_covariance_torch(
            covariance_nkk,
            variance.T,
        )
        residual = (z - lam).T.unsqueeze(-1)
        chol = torch.linalg.cholesky(covariance_nkk)
        solved = torch.cholesky_solve(residual, chol)
        quad = torch.sum(residual * solved, dim=(1, 2))
        logdet = 2.0 * torch.sum(
            torch.log(torch.diagonal(chol, dim1=1, dim2=2)),
            dim=1,
        )
        model = normalize_count_likelihood_model(
            str(self.config.count_likelihood_model)
        )
        if model == "gaussian":
            count_ll = -0.5 * (quad + logdet)
        else:
            df = max(float(self.config.count_likelihood_df), 1.0 + 1.0e-12)
            dim = float(z_arr.size)
            count_ll = -0.5 * (df + dim) * torch.log1p(quad / df) - 0.5 * logdet
        return count_ll + self._shield_shape_sequence_log_likelihood_gpu(
            lam,
            z_arr,
            var_arr,
        )

    def _sequence_diagonal_variance_torch(
        self,
        lam_kn: "torch.Tensor",
        z_kn: "torch.Tensor",
        observation_count_variances: NDArray[np.float64],
    ) -> "torch.Tensor":
        """Return KxN diagonal variances for same-station sequence likelihood."""
        import torch

        transport_rel = self._isotope_float_config(
            self.config.transport_model_rel_sigma
        )
        transport_abs = self._isotope_float_config(
            self.config.transport_model_abs_sigma
        )
        spectrum_rel = self._isotope_float_config(
            self.config.spectrum_count_rel_sigma
        )
        spectrum_abs = self._isotope_float_config(
            self.config.spectrum_count_abs_sigma
        )
        low_count_abs = self._isotope_float_config(self.config.low_count_abs_sigma)
        low_count_transition = self._isotope_float_config(
            self.config.low_count_transition_counts
        )
        obs_var = torch.clamp(
            torch.as_tensor(
                observation_count_variances,
                device=lam_kn.device,
                dtype=torch.float64,
            ).view(-1, 1),
            min=0.0,
        )
        return count_likelihood_variance_torch(
            z_kn,
            lam_kn,
            transport_model_rel_sigma=float(transport_rel),
            transport_model_abs_sigma=float(transport_abs),
            spectrum_count_rel_sigma=float(spectrum_rel),
            spectrum_count_abs_sigma=float(spectrum_abs),
            low_count_abs_sigma=float(low_count_abs),
            low_count_transition_counts=float(low_count_transition),
            observation_count_variance=obs_var,
        )

    def _sequence_observation_offdiag_torch(
        self,
        observation_count_covariance: NDArray[np.float64] | None,
        *,
        sequence_length: int,
        device: "torch.device",
    ) -> "torch.Tensor":
        """Return fixed off-diagonal observation covariance for one sequence."""
        import torch

        size = int(sequence_length)
        if observation_count_covariance is None:
            return torch.zeros((1, size, size), device=device, dtype=torch.float64)
        covariance = np.asarray(observation_count_covariance, dtype=float)
        if covariance.shape != (size, size):
            raise ValueError(
                "observation_count_covariance must be shaped sequence x sequence."
            )
        covariance = 0.5 * (covariance + covariance.T)
        np.fill_diagonal(covariance, 0.0)
        return torch.as_tensor(
            covariance,
            device=device,
            dtype=torch.float64,
        ).view(1, size, size)

    def _sequence_common_mode_offdiag_torch(
        self,
        lam_kn: "torch.Tensor",
    ) -> "torch.Tensor":
        """Return particle-wise off-diagonal covariance from common spectrum error."""
        import torch

        fraction = max(
            0.0,
            float(self.config.station_view_correlated_spectrum_fraction),
        )
        if not bool(self.config.station_view_covariance_enable) or fraction <= 0.0:
            size = int(lam_kn.shape[0])
            return torch.zeros(
                (int(lam_kn.shape[1]), size, size),
                device=lam_kn.device,
                dtype=torch.float64,
            )
        spectrum_rel = fraction * self._isotope_float_config(
            self.config.spectrum_count_rel_sigma
        )
        spectrum_abs = fraction * self._isotope_float_config(
            self.config.spectrum_count_abs_sigma
        )
        if spectrum_rel <= 0.0 and spectrum_abs <= 0.0:
            size = int(lam_kn.shape[0])
            return torch.zeros(
                (int(lam_kn.shape[1]), size, size),
                device=lam_kn.device,
                dtype=torch.float64,
            )
        lam_nk = lam_kn.T
        common = (float(spectrum_rel) ** 2) * (
            lam_nk[:, :, None] * lam_nk[:, None, :]
        )
        if spectrum_abs > 0.0:
            common = common + float(spectrum_abs) ** 2
        diag = torch.arange(int(lam_kn.shape[0]), device=lam_kn.device)
        common[:, diag, diag] = 0.0
        return common

    def _regularize_sequence_covariance_torch(
        self,
        covariance_nkk: "torch.Tensor",
        diagonal_nk: "torch.Tensor",
    ) -> "torch.Tensor":
        """Return a positive-definite covariance batch for Cholesky solves."""
        import torch

        size = int(covariance_nkk.shape[1])
        eye = torch.eye(size, device=covariance_nkk.device, dtype=torch.float64)
        diag_scale = torch.clamp(torch.mean(diagonal_nk, dim=1), min=1.0)
        jitter = 1.0e-9 * diag_scale
        regularized = covariance_nkk
        for _ in range(6):
            candidate = regularized + jitter.view(-1, 1, 1) * eye.view(1, size, size)
            _chol, info = torch.linalg.cholesky_ex(candidate)
            if bool(torch.all(info == 0)):
                return candidate
            jitter = jitter * 10.0
            regularized = covariance_nkk
        fallback = torch.diag_embed(torch.clamp(diagonal_nk, min=1.0e-12))
        return fallback + jitter.view(-1, 1, 1) * eye.view(1, size, size)

    def _shield_shape_sequence_log_likelihood_gpu(
        self,
        lam_kn: "torch.Tensor",
        z_obs: NDArray[np.float64],
        observation_count_variances: NDArray[np.float64],
    ) -> "torch.Tensor":
        """Return same-station shield signature likelihood terms."""
        return self._shield_contrast_sequence_log_likelihood_gpu(
            lam_kn,
            z_obs,
            observation_count_variances,
        ) + self._shield_view_ratio_sequence_log_likelihood_gpu(lam_kn, z_obs)

    def _shield_contrast_sequence_log_likelihood_gpu(
        self,
        lam_kn: "torch.Tensor",
        z_obs: NDArray[np.float64],
        observation_count_variances: NDArray[np.float64],
    ) -> "torch.Tensor":
        """Return a robust same-station shield-signature likelihood term."""
        import torch

        lam = lam_kn.to(dtype=torch.float64)
        if lam.ndim != 2:
            raise ValueError("Sequence expected counts must have shape K x N.")
        particle_count = int(lam.shape[1]) if lam.ndim == 2 else 0
        zeros = torch.zeros(particle_count, device=lam.device, dtype=torch.float64)
        if not bool(getattr(self.config, "shield_contrast_likelihood_enable", False)):
            return zeros
        weight = max(
            float(getattr(self.config, "shield_contrast_likelihood_weight", 1.0)),
            0.0,
        )
        min_views = max(
            int(getattr(self.config, "shield_contrast_min_views", 2)),
            2,
        )
        if weight <= 0.0 or int(lam.shape[0]) < min_views or particle_count == 0:
            return zeros

        z_arr = np.asarray(z_obs, dtype=float).reshape(-1)
        var_arr = np.asarray(observation_count_variances, dtype=float).reshape(-1)
        if z_arr.size != int(lam.shape[0]) or var_arr.size != int(lam.shape[0]):
            raise ValueError("Observation arrays must match the sequence length.")
        min_count = max(
            float(getattr(self.config, "shield_contrast_min_count", 25.0)),
            1.0e-6,
        )
        sigma_floor = max(
            float(getattr(self.config, "shield_contrast_log_sigma_floor", 0.5)),
            1.0e-6,
        )
        sigma_ceiling = max(
            float(getattr(self.config, "shield_contrast_log_sigma_ceiling", 2.0)),
            sigma_floor,
        )
        df = max(
            float(getattr(self.config, "shield_contrast_likelihood_df", 5.0)),
            1.0 + 1.0e-12,
        )

        z = torch.as_tensor(
            z_arr,
            device=lam.device,
            dtype=torch.float64,
        ).view(-1, 1)
        obs_var = torch.clamp(
            torch.as_tensor(
                var_arr,
                device=lam.device,
                dtype=torch.float64,
            ).view(-1, 1),
            min=0.0,
        )
        z_safe = torch.clamp(z, min=min_count)
        lam_safe = torch.clamp(lam, min=min_count)
        log_z = torch.log(z_safe)
        log_lam = torch.log(lam_safe)
        log_var = obs_var / torch.clamp(z_safe**2, min=1.0e-12)
        log_var = torch.clamp(
            log_var + sigma_floor**2,
            min=sigma_floor**2,
            max=sigma_ceiling**2,
        )
        view_weight = torch.reciprocal(log_var)
        weight_sum = torch.clamp(
            torch.sum(view_weight, dim=0, keepdim=True),
            min=1e-12,
        )
        obs_center = log_z - (
            torch.sum(view_weight * log_z, dim=0, keepdim=True) / weight_sum
        )
        pred_center = log_lam - torch.sum(
            view_weight * log_lam,
            dim=0,
            keepdim=True,
        ) / weight_sum
        residual = obs_center - pred_center
        terms = -0.5 * (df + 1.0) * torch.log1p(
            (residual**2) / (df * log_var)
        )
        terms -= 0.5 * torch.log(log_var)
        return float(weight) * torch.sum(terms, dim=0)

    def _shield_view_ratio_sequence_log_likelihood_gpu(
        self,
        lam_kn: "torch.Tensor",
        z_obs: NDArray[np.float64],
    ) -> "torch.Tensor":
        """Return a Dirichlet-multinomial shield-view ratio likelihood term."""
        import torch

        lam = lam_kn.to(dtype=torch.float64)
        if lam.ndim != 2:
            raise ValueError("Sequence expected counts must have shape K x N.")
        particle_count = int(lam.shape[1]) if lam.ndim == 2 else 0
        zeros = torch.zeros(particle_count, device=lam.device, dtype=torch.float64)
        if not bool(
            getattr(self.config, "shield_view_ratio_likelihood_enable", False)
        ):
            return zeros
        weight = max(
            float(getattr(self.config, "shield_view_ratio_likelihood_weight", 1.0)),
            0.0,
        )
        concentration = max(
            float(
                getattr(
                    self.config,
                    "shield_view_ratio_likelihood_concentration",
                    128.0,
                )
            ),
            1.0e-6,
        )
        min_total = max(
            float(
                getattr(
                    self.config,
                    "shield_view_ratio_likelihood_min_total_count",
                    25.0,
                )
            ),
            0.0,
        )
        min_views = max(
            int(getattr(self.config, "shield_view_ratio_likelihood_min_views", 2)),
            2,
        )
        if (
            weight <= 0.0
            or int(lam.shape[0]) < min_views
            or particle_count == 0
        ):
            return zeros
        z_arr = np.asarray(z_obs, dtype=float).reshape(-1)
        if z_arr.size != int(lam.shape[0]):
            raise ValueError("Observation arrays must match the sequence length.")
        z_arr = np.maximum(np.where(np.isfinite(z_arr), z_arr, 0.0), 0.0)
        total = float(np.sum(z_arr))
        if total < min_total or not np.isfinite(total):
            return zeros
        z = torch.as_tensor(
            z_arr,
            device=lam.device,
            dtype=torch.float64,
        ).view(-1, 1)
        lam_safe = torch.clamp(lam, min=1.0e-12)
        lam_total = torch.clamp(torch.sum(lam_safe, dim=0, keepdim=True), min=1.0e-12)
        probabilities = lam_safe / lam_total
        alpha = torch.clamp(concentration * probabilities, min=1.0e-12)
        alpha0 = torch.clamp(torch.sum(alpha, dim=0), min=1.0e-12)
        total_t = torch.as_tensor(total, device=lam.device, dtype=torch.float64)
        ll = torch.lgamma(alpha0) - torch.lgamma(alpha0 + total_t)
        ll = ll + torch.sum(torch.lgamma(alpha + z) - torch.lgamma(alpha), dim=0)
        return float(weight) * ll

    def _normalized_log_weights_torch(self, logw: "torch.Tensor") -> "torch.Tensor":
        """Normalize log-weights using logsumexp in float64."""
        import torch

        return logw - torch.logsumexp(logw, dim=0)

    def _ess_from_logw_torch(self, logw: "torch.Tensor") -> float:
        """Return the effective sample size from normalized log-weights."""
        import torch

        w = torch.exp(logw)
        ess = 1.0 / torch.sum(w**2)
        return float(ess.detach().cpu().item())

    def _assign_logw_from_torch(self, logw: "torch.Tensor") -> None:
        """Copy log-weights from torch back into particle objects."""
        logw_cpu = logw.detach().cpu().numpy()
        for p, lw in zip(self.continuous_particles, logw_cpu):
            p.log_weight = float(lw)

    def _update_continuous_weights_gpu(
        self,
        lam_t: "torch.Tensor",
        z_obs: float,
        *,
        observation_count_variance: float = 0.0,
        delta_beta: float = 1.0,
        logw_prev: "torch.Tensor | None" = None,
        ll_t: "torch.Tensor | None" = None,
        return_logw: bool = False,
    ) -> "torch.Tensor | None":
        """
        Update continuous log-weights using tempered Poisson increments.

        When return_logw is True, returns the normalized log-weights after the update.
        """
        if lam_t.numel() == 0:
            return
        logw_prev = logw_prev if logw_prev is not None else self._current_log_weights_torch(lam_t.device)
        ll_t = ll_t if ll_t is not None else self._log_likelihood_increment_gpu(
            lam_t,
            z_obs,
            observation_count_variance=observation_count_variance,
        )
        logw = self._normalized_log_weights_torch(logw_prev + float(delta_beta) * ll_t)
        self._assign_logw_from_torch(logw)
        if return_logw:
            return logw
        return None

    def _select_delta_beta(
        self,
        logw_prev: "torch.Tensor",
        ll_t: "torch.Tensor",
        remaining: float,
        target_ess: float,
    ) -> tuple[float, "torch.Tensor", float]:
        """
        Return the largest delta_beta that keeps ESS above the target.

        Returns (delta_beta, logw_new, ess).
        """
        remaining = float(remaining)
        min_delta = max(float(self.config.min_delta_beta), 0.0)
        if remaining <= min_delta:
            logw_new = self._normalized_log_weights_torch(logw_prev + remaining * ll_t)
            ess = self._ess_from_logw_torch(logw_new)
            return remaining, logw_new, ess

        logw_full = self._normalized_log_weights_torch(logw_prev + remaining * ll_t)
        ess_full = self._ess_from_logw_torch(logw_full)
        if ess_full >= target_ess:
            return remaining, logw_full, ess_full

        logw_low = self._normalized_log_weights_torch(logw_prev + min_delta * ll_t)
        ess_low = self._ess_from_logw_torch(logw_low)
        if ess_low < target_ess:
            return min_delta, logw_low, ess_low

        low = min_delta
        high = remaining
        logw_best = logw_low
        ess_best = ess_low
        for _ in range(24):
            mid = 0.5 * (low + high)
            logw_mid = self._normalized_log_weights_torch(logw_prev + mid * ll_t)
            ess_mid = self._ess_from_logw_torch(logw_mid)
            if ess_mid >= target_ess:
                low = mid
                logw_best = logw_mid
                ess_best = ess_mid
            else:
                high = mid
        return low, logw_best, ess_best

    def _tempered_update(
        self,
        lam_fn: Callable[[], "torch.Tensor"],
        z_obs: float,
        observation_count_variance: float = 0.0,
        disable_regularize_on_resample: bool | None = None,
        roughening_scale_on_resample: float = 1.0,
    ) -> tuple[float, bool]:
        """
        Apply ESS-targeted tempering for a single Poisson update.

        The update increments beta from 0 to 1 using delta_beta steps that
        maintain ESS above the configured target ratio when possible.

        Returns (ess_pre, resampled_any) for downstream adaptation logic.
        """
        def _ll_fn() -> "torch.Tensor":
            """Return per-particle log-likelihood increments for one count."""
            import torch

            lam_t_inner = lam_fn()
            if lam_t_inner.numel() == 0:
                return lam_t_inner.to(dtype=torch.float64)
            return self._log_likelihood_increment_gpu(
                lam_t_inner,
                z_obs,
                observation_count_variance=observation_count_variance,
            )

        return self._tempered_update_likelihood(
            ll_fn=_ll_fn,
            disable_regularize_on_resample=disable_regularize_on_resample,
            roughening_scale_on_resample=roughening_scale_on_resample,
        )

    def _tempered_update_likelihood(
        self,
        ll_fn: Callable[[], "torch.Tensor"],
        *,
        disable_regularize_on_resample: bool | None = None,
        roughening_scale_on_resample: float = 1.0,
    ) -> tuple[float, bool]:
        """
        Apply ESS-targeted tempering to a precomputed likelihood increment.

        ``ll_fn`` is re-evaluated after a tempering resample, which keeps joint
        multi-orientation updates consistent with the newly roughened particles.
        """
        beta_total = 0.0
        steps: list[dict[str, float]] = []
        resamples = 0
        resampled_any = False
        ess_min: float | None = None
        target_ess = float(self.config.target_ess_ratio) * max(self.N, 1)
        resample_threshold = float(self.config.resample_threshold) * max(self.N, 1)
        max_resamples = max(0, int(self.config.max_resamples_per_observation))
        cooldown_steps = max(0, int(self.config.temper_resample_cooldown_steps))
        force_resample_ess = float(self.config.temper_resample_force_ratio) * max(self.N, 1)
        disable_regularize = bool(self.config.disable_regularize_on_temper_resample)
        if disable_regularize_on_resample is not None:
            disable_regularize = disable_regularize or bool(
                disable_regularize_on_resample
            )
        ll_t = ll_fn()
        if ll_t.numel() == 0:
            self.last_temper_steps = []
            self.last_temper_resample_count = 0
            return 0.0, False
        logw = self._current_log_weights_torch(ll_t.device)

        cooldown_remaining = 0
        while beta_total < 1.0 - 1e-12:
            remaining = 1.0 - beta_total
            delta_beta, logw_new, ess = self._select_delta_beta(
                logw_prev=logw,
                ll_t=ll_t,
                remaining=remaining,
                target_ess=target_ess,
            )
            logw = logw_new
            self._assign_logw_from_torch(logw)
            beta_total += delta_beta
            ess_min = ess if ess_min is None else min(ess_min, ess)
            steps.append(
                {
                    "beta_total": float(beta_total),
                    "delta_beta": float(delta_beta),
                    "ess": float(ess),
                }
            )
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
            do_resample = (
                ess < resample_threshold
                and resamples < max_resamples
                and (cooldown_remaining == 0 or ess < force_resample_ess)
            )
            can_jump_without_intermediate_state_change = (
                not do_resample
                and cooldown_remaining == 0
                and delta_beta < remaining - 1.0e-12
            )
            if can_jump_without_intermediate_state_change:
                remaining_after_delta = 1.0 - beta_total
                logw_full = self._normalized_log_weights_torch(
                    logw + remaining_after_delta * ll_t
                )
                ess_full = self._ess_from_logw_torch(logw_full)
                logw = logw_full
                self._assign_logw_from_torch(logw)
                beta_total = 1.0
                ess_min = ess_full if ess_min is None else min(ess_min, ess_full)
                steps.append(
                    {
                        "beta_total": float(beta_total),
                        "delta_beta": float(remaining_after_delta),
                        "ess": float(ess_full),
                    }
                )
                if ess_full < resample_threshold and resamples < max_resamples:
                    self._maybe_resample_continuous(
                        disable_regularize=disable_regularize,
                        roughening_scale=roughening_scale_on_resample,
                    )
                    if self.last_resample_ess:
                        resampled_any = True
                        resamples += 1
                break
            if do_resample:
                self._maybe_resample_continuous(
                    disable_regularize=disable_regularize,
                    roughening_scale=roughening_scale_on_resample,
                )
                if self.last_resample_ess:
                    resampled_any = True
                    resamples += 1
                    cooldown_remaining = max(cooldown_remaining, cooldown_steps)
                    ll_t = ll_fn()
                    if ll_t.numel() == 0:
                        break
                    logw = self._current_log_weights_torch(ll_t.device)
            if resamples >= max_resamples and beta_total < 1.0 - 1e-12:
                remaining = 1.0 - beta_total
                logw = self._normalized_log_weights_torch(logw + remaining * ll_t)
                self._assign_logw_from_torch(logw)
                beta_total = 1.0
                ess = self._ess_from_logw_torch(logw)
                ess_min = ess if ess_min is None else min(ess_min, ess)
                steps.append(
                    {
                        "beta_total": float(beta_total),
                        "delta_beta": float(remaining),
                        "ess": float(ess),
                    }
                )
                break
        self.last_temper_steps = steps
        self.last_temper_resample_count = resamples
        if ess_min is None:
            ess_min = 0.0
        return float(ess_min), resampled_any

    def _continuous_expected_counts_gpu(
        self, pose_idx: int, orient_idx: int, live_time_s: float
    ) -> NDArray[np.float64]:
        """Compute Λ_{k,h}^{(n)} using torch for a single orientation index."""
        lam_t = self._continuous_expected_counts_torch(
            pose_idx=pose_idx, orient_idx=orient_idx, live_time_s=live_time_s
        )
        return lam_t.detach().cpu().numpy()

    def _continuous_expected_counts_pair_gpu(
        self, pose_idx: int, fe_index: int, pb_index: int, live_time_s: float
    ) -> NDArray[np.float64]:
        """Compute Λ_{k,h}^{(n)} using torch for Fe/Pb orientation indices."""
        lam_t = self._continuous_expected_counts_pair_torch(
            pose_idx=pose_idx, fe_index=fe_index, pb_index=pb_index, live_time_s=live_time_s
        )
        return lam_t.detach().cpu().numpy()

    def _continuous_expected_counts_cpu(
        self,
        pose_idx: int,
        orient_idx: int,
        live_time_s: float,
    ) -> NDArray[np.float64]:
        """Compute single-orientation expected counts on CPU."""
        if self.kernel is None:
            raise RuntimeError("Continuous PF update requires an attached kernel.")
        detector_pos = np.asarray(self.kernel.poses[int(pose_idx)], dtype=float)
        lam = np.zeros(len(self.continuous_particles), dtype=float)
        source_scale = self._measurement_source_scale(
            fe_index=orient_idx,
            pb_index=orient_idx,
        )
        for particle_idx, particle in enumerate(self.continuous_particles):
            state = particle.state
            rate = float(state.background)
            for pos, strength in zip(
                state.positions[: state.num_sources],
                state.strengths[: state.num_sources],
            ):
                kernel_val = self.continuous_kernel.kernel_value(
                    isotope=self.isotope,
                    detector_pos=detector_pos,
                    source_pos=pos,
                    orient_idx=int(orient_idx),
                )
                rate += source_scale * float(kernel_val) * float(strength)
            lam[particle_idx] = float(live_time_s) * rate
        return lam

    def _continuous_expected_counts_pair_cpu(
        self,
        pose_idx: int,
        fe_index: int,
        pb_index: int,
        live_time_s: float,
    ) -> NDArray[np.float64]:
        """Compute Fe/Pb pair expected counts on CPU."""
        if self.kernel is None:
            raise RuntimeError("Continuous PF update requires an attached kernel.")
        detector_pos = np.asarray(self.kernel.poses[int(pose_idx)], dtype=float)
        return self._continuous_expected_counts_pair_at_pose_cpu(
            detector_pos=detector_pos,
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=live_time_s,
        )

    def _continuous_expected_counts_pair_at_pose_cpu(
        self,
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        live_time_s: float,
    ) -> NDArray[np.float64]:
        """Compute Fe/Pb pair expected counts at an explicit pose on CPU."""
        detector_arr = np.asarray(detector_pos, dtype=float)
        lam = np.zeros(len(self.continuous_particles), dtype=float)
        source_scale = self._measurement_source_scale(
            fe_index=fe_index,
            pb_index=pb_index,
        )
        for particle_idx, particle in enumerate(self.continuous_particles):
            state = particle.state
            rate = float(state.background)
            for pos, strength in zip(
                state.positions[: state.num_sources],
                state.strengths[: state.num_sources],
            ):
                kernel_val = self.continuous_kernel.kernel_value_pair(
                    isotope=self.isotope,
                    detector_pos=detector_arr,
                    source_pos=pos,
                    fe_index=int(fe_index),
                    pb_index=int(pb_index),
                )
                rate += source_scale * float(kernel_val) * float(strength)
            lam[particle_idx] = float(live_time_s) * rate
        return lam

    def _continuous_expected_counts(self, pose_idx: int, orient_idx: int, live_time_s: float) -> NDArray[np.float64]:
        """Compute Λ_{k,h}^{(n)} for each continuous particle using ContinuousKernel."""
        if not self._can_use_gpu():
            return self._continuous_expected_counts_cpu(
                pose_idx=pose_idx,
                orient_idx=orient_idx,
                live_time_s=live_time_s,
            )
        return self._continuous_expected_counts_gpu(
            pose_idx=pose_idx, orient_idx=orient_idx, live_time_s=live_time_s
        )

    def _continuous_expected_counts_pair(
        self, pose_idx: int, fe_index: int, pb_index: int, live_time_s: float
    ) -> NDArray[np.float64]:
        """Compute Λ_{k,h}^{(n)} using Fe/Pb octant indices (Eq. 3.41)."""
        if not self._can_use_gpu():
            return self._continuous_expected_counts_pair_cpu(
                pose_idx=pose_idx,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
            )
        return self._continuous_expected_counts_pair_gpu(
            pose_idx=pose_idx, fe_index=fe_index, pb_index=pb_index, live_time_s=live_time_s
        )

    def _continuous_expected_counts_pair_at_pose_torch(
        self,
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        live_time_s: float,
    ) -> "torch.Tensor":
        """Compute Λ_{k,h}^{(n)} using torch for explicit detector position."""
        from pf import gpu_utils
        import torch

        device = gpu_utils.resolve_device(self.config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.config.gpu_dtype)
        if not self.continuous_particles:
            return torch.zeros(0, device=device, dtype=dtype)
        positions, strengths, backgrounds, mask = gpu_utils.pack_states(
            [p.state for p in self.continuous_particles],
            device=device,
            dtype=dtype,
        )
        det_pos = np.asarray(detector_pos, dtype=float)
        return self.continuous_kernel.expected_counts_pair_for_packed_states_torch(
            isotope=self.isotope,
            detector_pos=det_pos,
            positions=positions,
            strengths=strengths,
            backgrounds=backgrounds,
            mask=mask,
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=live_time_s,
            source_scale=self._measurement_source_scale(
                fe_index=fe_index,
                pb_index=pb_index,
            ),
            device=device,
            dtype=dtype,
        )

    def _continuous_expected_counts_pair_at_pose(
        self,
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        live_time_s: float,
    ) -> NDArray[np.float64]:
        """Compute Λ_{k,h}^{(n)} for explicit detector position."""
        if not self._can_use_gpu():
            return self._continuous_expected_counts_pair_at_pose_cpu(
                detector_pos=detector_pos,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
            )
        lam_t = self._continuous_expected_counts_pair_at_pose_torch(
            detector_pos=detector_pos,
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=live_time_s,
        )
        return lam_t.detach().cpu().numpy()

    def update_continuous_pair(
        self,
        z_obs: float,
        pose_idx: int,
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        observation_count_variance: float = 0.0,
        step_idx: int | None = None,
        defer_resample: bool = False,
        spectrum_counts: NDArray[np.float64] | None = None,
        spectrum_response_template: NDArray[np.float64] | None = None,
        spectrum_background: NDArray[np.float64] | None = None,
        spectrum_variance: NDArray[np.float64] | None = None,
    ) -> None:
        """
        Count-likelihood weight update using Fe/Pb orientation indices.

        z_obs must come from spectrum unfolding; expected Λ_{k,h} is computed via expected_counts_pair.
        When ``defer_resample`` is True, structural updates are deferred to the
        caller's end-of-station finalization, but ESS/tempered resampling is
        still allowed for this posture to avoid burst-level weight collapse.
        """
        if self._should_skip_converged_update():
            self.updates_skipped += 1
            return
        self.reset_step_stats()
        self._gpu_enabled()
        detector_pos = np.asarray(self.kernel.poses[pose_idx], dtype=float) if self.kernel else None
        self._record_observed_station(detector_pos)

        def _lam_fn() -> "torch.Tensor":
            """Return expected counts for the current particle set."""
            return self._continuous_expected_counts_pair_torch(
                pose_idx=pose_idx,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
            )

        spectrum_arrays = self._spectrum_update_arrays(
            spectrum_counts,
            spectrum_response_template,
            spectrum_background,
            spectrum_variance,
            sequence_length=1,
        )

        def _spectral_ll_fn() -> "torch.Tensor":
            """Return direct spectrum-bin likelihood increments for one view."""
            if spectrum_arrays is None:
                raise RuntimeError("spectrum_arrays must be available.")
            observed, template, background, variance = spectrum_arrays
            lam_t_inner = _lam_fn()
            return self._spectral_bin_sequence_log_likelihood_from_lambda_gpu(
                lam_t_inner,
                observed,
                template,
                background,
                variance,
            )

        roughening_scale = 1.0
        disable_regularize = False
        if defer_resample:
            roughening_scale = max(
                0.0,
                float(self.config.deferred_resample_roughening_scale),
            )
            disable_regularize = roughening_scale <= 0.0
        if self.config.use_tempering and spectrum_arrays is not None:
            ess_pre, resampled_any = self._tempered_update_likelihood(
                ll_fn=_spectral_ll_fn,
                disable_regularize_on_resample=disable_regularize,
                roughening_scale_on_resample=roughening_scale,
            )
        elif self.config.use_tempering:
            debug_timing = _pf_debug_timing_enabled()
            debug_start = time.perf_counter()
            if debug_timing:
                print(
                    f"[pf_internal] isotope={self.isotope} step={step_idx} "
                    f"phase=tempered_start defer={defer_resample} "
                    f"fe={fe_index} pb={pb_index} z={float(z_obs):.6g}",
                    flush=True,
                )
            ess_pre, resampled_any = self._tempered_update(
                lam_fn=_lam_fn,
                z_obs=z_obs,
                observation_count_variance=observation_count_variance,
                disable_regularize_on_resample=disable_regularize,
                roughening_scale_on_resample=roughening_scale,
            )
            if debug_timing:
                print(
                    f"[pf_internal] isotope={self.isotope} step={step_idx} "
                    f"phase=tempered_done elapsed={time.perf_counter() - debug_start:.3f}s "
                    f"resampled={resampled_any} ess={float(ess_pre):.3f}",
                    flush=True,
                )
        elif spectrum_arrays is not None:
            ll_t = _spectral_ll_fn()
            if ll_t.numel() == 0:
                ess_pre = 0.0
            else:
                logw_prev = self._current_log_weights_torch(ll_t.device)
                logw = self._normalized_log_weights_torch(logw_prev + ll_t)
                self._assign_logw_from_torch(logw)
                ess_pre = self._ess_from_logw_torch(logw)
            self._maybe_resample_continuous(
                disable_regularize=disable_regularize,
                roughening_scale=roughening_scale,
            )
            resampled_any = bool(self.last_resample_ess)
        else:
            lam_t = _lam_fn()
            logw = self._update_continuous_weights_gpu(
                lam_t,
                z_obs,
                observation_count_variance=observation_count_variance,
                return_logw=True,
            )
            if logw is None:
                ess_pre = 0.0
            else:
                ess_pre = self._ess_from_logw_torch(logw)
            self._maybe_resample_continuous(
                disable_regularize=disable_regularize,
                roughening_scale=roughening_scale,
            )
            resampled_any = bool(self.last_resample_ess)
            if logw is None and self.last_ess_pre is not None:
                ess_pre = float(self.last_ess_pre)
        if resampled_any:
            self._trigger_adapt_cooldown()
        if defer_resample:
            self._deferred_resampled_any = self._deferred_resampled_any or bool(
                resampled_any
            )
            if np.isfinite(float(ess_pre)):
                if self._deferred_ess_min is None:
                    self._deferred_ess_min = float(ess_pre)
                else:
                    self._deferred_ess_min = min(
                        float(self._deferred_ess_min),
                        float(ess_pre),
                    )
            if resampled_any:
                self.align_continuous_labels()
        else:
            self.adapt_num_particles(ess_pre=ess_pre, resampled=resampled_any)
            self.align_continuous_labels()
            self._advance_adapt_cooldown()
        if detector_pos is not None:
            if defer_resample:
                self._deferred_convergence_args = (
                    step_idx,
                    np.asarray(detector_pos, dtype=float).copy(),
                    int(fe_index),
                    int(pb_index),
                    float(live_time_s),
                    float(z_obs),
                )
                return
            self._maybe_update_convergence(
                step_idx=step_idx,
                detector_pos=detector_pos,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
                z_obs=z_obs,
            )

    def finalize_deferred_update(self) -> None:
        """Finalize a station whose structural updates were delayed."""
        if self._should_skip_converged_update():
            return
        if self._deferred_ess_min is not None:
            ess_pre = float(self._deferred_ess_min)
        elif self.last_ess_pre is not None:
            ess_pre = float(self.last_ess_pre)
        else:
            weights = np.asarray(self.continuous_weights, dtype=float)
            ess_pre = (
                float(1.0 / max(np.sum(weights**2), 1.0e-12))
                if weights.size
                else 0.0
            )
        resampled_any = bool(self._deferred_resampled_any)
        self.adapt_num_particles(ess_pre=ess_pre, resampled=resampled_any)
        self.align_continuous_labels()
        self._advance_adapt_cooldown()
        convergence_args = self._deferred_convergence_args
        self._deferred_convergence_args = None
        if convergence_args is not None:
            step_idx, detector_pos, fe_index, pb_index, live_time_s, z_obs = (
                convergence_args
            )
            self._maybe_update_convergence(
                step_idx=step_idx,
                detector_pos=detector_pos,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
                z_obs=z_obs,
            )
        self._deferred_resampled_any = False
        self._deferred_ess_min = None

    def update_continuous_pair_sequence(
        self,
        z_obs: NDArray[np.float64],
        pose_idx: int,
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
        live_times_s: NDArray[np.float64],
        observation_count_variances: NDArray[np.float64] | None = None,
        observation_count_covariance: NDArray[np.float64] | None = None,
        step_idx: int | None = None,
        spectrum_counts: NDArray[np.float64] | None = None,
        spectrum_response_template: NDArray[np.float64] | None = None,
        spectrum_background: NDArray[np.float64] | None = None,
        spectrum_variance: NDArray[np.float64] | None = None,
    ) -> None:
        """
        Jointly update weights using a same-pose shield-orientation sequence.

        The measurement model evaluates the shield program as one station-level
        observation. When covariance is supplied or configured, same-station
        shield-view correlations are handled by a batched multivariate
        likelihood; otherwise the update reduces to the product likelihood over
        views. Updating views jointly avoids resampling, roughening, or
        birth/death moves between postures from the same physical station.
        """
        if self._should_skip_converged_update():
            self.updates_skipped += 1
            return
        self.reset_step_stats()
        self._gpu_enabled()
        detector_pos = np.asarray(self.kernel.poses[pose_idx], dtype=float) if self.kernel else None
        self._record_observed_station(detector_pos)
        z_arr = np.asarray(z_obs, dtype=float).ravel()
        fe_arr = np.asarray(fe_indices, dtype=int).ravel()
        pb_arr = np.asarray(pb_indices, dtype=int).ravel()
        live_arr = np.asarray(live_times_s, dtype=float).ravel()
        if observation_count_variances is None:
            var_arr = np.zeros_like(z_arr, dtype=float)
        else:
            var_arr = np.asarray(observation_count_variances, dtype=float).ravel()
        covariance = None
        if observation_count_covariance is not None:
            covariance = np.asarray(observation_count_covariance, dtype=float)
            if covariance.shape != (z_arr.size, z_arr.size):
                raise ValueError(
                    "observation_count_covariance must be shaped K x K."
                )
        if not (
            z_arr.size == fe_arr.size == pb_arr.size == live_arr.size == var_arr.size
        ):
            raise ValueError("Joint PF update arrays must have matching lengths.")
        if z_arr.size == 0:
            return
        spectrum_arrays = self._spectrum_update_arrays(
            spectrum_counts,
            spectrum_response_template,
            spectrum_background,
            spectrum_variance,
            sequence_length=z_arr.size,
        )

        def _ll_fn() -> "torch.Tensor":
            """Return summed per-particle log-likelihood for the shield sequence."""
            lam_kn = self._continuous_expected_counts_pair_sequence_torch(
                pose_idx=pose_idx,
                fe_indices=fe_arr,
                pb_indices=pb_arr,
                live_times_s=live_arr,
            )
            if spectrum_arrays is not None:
                observed, template, background_spectrum, variance = spectrum_arrays
                spectrum_ll = self._spectral_bin_sequence_log_likelihood_from_lambda_gpu(
                    lam_kn,
                    observed,
                    template,
                    background_spectrum,
                    variance,
                )
                return spectrum_ll + self._shield_shape_sequence_log_likelihood_gpu(
                    lam_kn,
                    z_arr,
                    var_arr,
                )
            return self._log_likelihood_sequence_gpu(
                lam_kn,
                z_arr,
                var_arr,
                observation_count_covariance=covariance,
            )

        if self.config.use_tempering:
            ess_pre, resampled_any = self._tempered_update_likelihood(ll_fn=_ll_fn)
        else:
            ll_t = _ll_fn()
            if ll_t.numel() == 0:
                ess_pre = 0.0
                resampled_any = False
            else:
                logw_prev = self._current_log_weights_torch(ll_t.device)
                logw = self._normalized_log_weights_torch(logw_prev + ll_t)
                self._assign_logw_from_torch(logw)
                ess_pre = self._ess_from_logw_torch(logw)
                self._maybe_resample_continuous()
                resampled_any = bool(self.last_resample_ess)
        if resampled_any:
            self._trigger_adapt_cooldown()
        self.adapt_num_particles(ess_pre=ess_pre, resampled=resampled_any)
        self.align_continuous_labels()
        self._advance_adapt_cooldown()
        if detector_pos is not None:
            self._maybe_update_convergence(
                step_idx=step_idx,
                detector_pos=detector_pos,
                fe_index=int(fe_arr[-1]),
                pb_index=int(pb_arr[-1]),
                live_time_s=float(np.sum(live_arr)),
                z_obs=float(np.sum(z_arr)),
            )

    def update_continuous_pair_at_pose(
        self,
        z_obs: float,
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        observation_count_variance: float = 0.0,
        step_idx: int | None = None,
    ) -> None:
        """
        Count-likelihood weight update using explicit detector position.

        This avoids reliance on pose indices for planning-time evaluations.
        """
        if self._should_skip_converged_update():
            self.updates_skipped += 1
            return
        self.reset_step_stats()
        self._gpu_enabled()

        def _lam_fn() -> "torch.Tensor":
            """Return expected counts for the current particle set."""
            return self._continuous_expected_counts_pair_at_pose_torch(
                detector_pos=detector_pos,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
            )

        if self.config.use_tempering:
            ess_pre, resampled_any = self._tempered_update(
                lam_fn=_lam_fn,
                z_obs=z_obs,
                observation_count_variance=observation_count_variance,
            )
        else:
            lam_t = _lam_fn()
            logw = self._update_continuous_weights_gpu(
                lam_t,
                z_obs,
                observation_count_variance=observation_count_variance,
                return_logw=True,
            )
            if logw is None:
                ess_pre = 0.0
            else:
                ess_pre = self._ess_from_logw_torch(logw)
            self._maybe_resample_continuous()
            resampled_any = bool(self.last_resample_ess)
            if logw is None and self.last_ess_pre is not None:
                ess_pre = float(self.last_ess_pre)
        if resampled_any:
            self._trigger_adapt_cooldown()
        self.adapt_num_particles(ess_pre=ess_pre, resampled=resampled_any)
        self.align_continuous_labels()
        self._advance_adapt_cooldown()
        self._maybe_update_convergence(
            step_idx=step_idx,
            detector_pos=np.asarray(detector_pos, dtype=float),
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=live_time_s,
            z_obs=z_obs,
        )

    @property
    def continuous_weights(self) -> NDArray[np.float64]:
        """Return normalized weights for continuous particles."""
        logw = np.asarray([p.log_weight for p in self.continuous_particles], dtype=np.float64)
        if logw.size == 0:
            return np.zeros(0, dtype=float)
        logw = logw - np.max(logw)
        w = np.exp(logw)
        s = np.sum(w)
        if s <= 0:
            return np.ones(len(self.continuous_particles), dtype=float) / len(self.continuous_particles)
        return w / s

    def _normalize_continuous_log_weights(self) -> None:
        """Normalize continuous-particle log weights in place."""
        if not self.continuous_particles:
            return
        logw = np.asarray(
            [p.log_weight for p in self.continuous_particles],
            dtype=np.float64,
        )
        norm = logsumexp(logw)
        if not np.isfinite(norm):
            uniform = -np.log(max(len(self.continuous_particles), 1))
            for particle in self.continuous_particles:
                particle.log_weight = float(uniform)
            return
        for particle, value in zip(self.continuous_particles, logw - norm):
            particle.log_weight = float(value)

    def inject_runtime_report_rescue_particles(
        self,
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
        *,
        particle_fraction: float,
        min_particles_per_source: int,
        total_weight: float,
        jitter_sigma_m: float,
        combine_sources: bool = True,
    ) -> int:
        """Inject bounded report-rescue particles into the continuous PF population."""
        if not self.continuous_particles:
            return 0
        pos_arr = np.asarray(positions, dtype=float).reshape(-1, 3)
        q_arr = np.asarray(strengths, dtype=float).reshape(-1)
        if pos_arr.size == 0 or q_arr.size == 0 or pos_arr.shape[0] != q_arr.size:
            return 0
        finite = np.isfinite(pos_arr).all(axis=1) & np.isfinite(q_arr) & (q_arr > 0.0)
        if not np.any(finite):
            return 0
        pos_arr = pos_arr[finite]
        q_arr = q_arr[finite]
        if combine_sources and self.config.max_sources is not None:
            max_sources = max(0, int(self.config.max_sources))
            if max_sources <= 0:
                return 0
            if pos_arr.shape[0] > max_sources:
                order = np.argsort(q_arr)[::-1][:max_sources]
                pos_arr = pos_arr[order]
                q_arr = q_arr[order]
        source_count = int(pos_arr.shape[0])
        if source_count <= 0:
            return 0
        q_min = max(float(self.config.min_strength), 0.0)
        q_max = float(self.config.birth_q_max)
        if q_max > 0.0:
            q_arr = np.clip(q_arr, q_min, q_max)
        else:
            q_arr = np.maximum(q_arr, q_min)
        full_rescue_mass = max(
            float(getattr(self.config, "runtime_report_rescue_weight", total_weight)),
            0.0,
        )
        quarantine_streak = 1 if total_weight + 1.0e-12 < full_rescue_mass else 0
        q_arr = np.maximum(q_arr, q_min)
        pos_arr = self._project_positions_to_source_prior(pos_arr)
        particle_count = len(self.continuous_particles)
        fraction = float(np.clip(float(particle_fraction), 0.0, 1.0))
        requested = int(np.ceil(fraction * float(particle_count)))
        requested = max(
            requested,
            max(1, int(min_particles_per_source)) * source_count,
        )
        inject_count = min(particle_count, max(1, requested))
        rescue_mass = float(np.clip(float(total_weight), 0.0, 0.5))
        if inject_count <= 0 or rescue_mass <= 0.0:
            return 0
        weights = np.asarray(self.continuous_weights, dtype=float)
        if weights.size != particle_count or float(np.sum(weights)) <= 0.0:
            weights = np.full(particle_count, 1.0 / float(particle_count), dtype=float)
        replace_indices = np.argsort(weights)[:inject_count]
        replace_mask = np.zeros(particle_count, dtype=bool)
        replace_mask[replace_indices] = True
        keep_weight_sum = float(np.sum(weights[~replace_mask]))
        keep_mass = max(1.0 - rescue_mass, 0.0)
        if keep_weight_sum <= 0.0 and keep_mass > 0.0:
            keep_weights = np.full(
                int(np.count_nonzero(~replace_mask)),
                keep_mass / max(int(np.count_nonzero(~replace_mask)), 1),
                dtype=float,
            )
            keep_iter = iter(keep_weights.tolist())
            for idx, particle in enumerate(self.continuous_particles):
                if not replace_mask[idx]:
                    particle.log_weight = float(np.log(max(next(keep_iter), 1.0e-300)))
        elif keep_mass > 0.0:
            for idx, particle in enumerate(self.continuous_particles):
                if not replace_mask[idx]:
                    scaled = keep_mass * float(weights[idx]) / keep_weight_sum
                    particle.log_weight = float(np.log(max(scaled, 1.0e-300)))
        rescue_log_weight = float(
            np.log(max(rescue_mass / max(inject_count, 1), 1.0e-300))
        )
        try:
            background = float(self.best_particle().state.background)
        except ValueError:
            background = self._background_level()
        jitter_sigma = max(0.0, float(jitter_sigma_m))
        strength_log_sigma = min(max(float(self.config.strength_log_sigma), 0.0), 0.25)
        try:
            base_state = self.best_particle().state.copy()
        except ValueError:
            base_state = IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3), dtype=float),
                strengths=np.zeros(0, dtype=float),
                background=background,
            )
        for local_idx, particle_idx in enumerate(replace_indices):
            if combine_sources:
                if jitter_sigma > 0.0 and local_idx > 0:
                    jitter = np.random.normal(scale=jitter_sigma, size=pos_arr.shape)
                    pos_new = self._project_positions_to_source_prior(pos_arr + jitter)
                else:
                    pos_new = pos_arr.copy()
                if strength_log_sigma > 0.0 and local_idx > 0:
                    q_new = q_arr * np.exp(
                        np.random.normal(scale=strength_log_sigma, size=q_arr.shape)
                    )
                    if q_max > 0.0:
                        q_new = np.clip(q_new, q_min, q_max)
                    else:
                        q_new = np.maximum(q_new, q_min)
                    q_new = np.maximum(q_new, q_min)
                else:
                    q_new = q_arr.copy()
                state = IsotopeState(
                    num_sources=source_count,
                    positions=pos_new.astype(float, copy=True),
                    strengths=q_new.astype(float, copy=True),
                    background=background,
                    ages=np.zeros(source_count, dtype=np.int64),
                    low_q_streaks=np.zeros(source_count, dtype=np.int64),
                    support_scores=np.ones(source_count, dtype=float),
                    tentative_sources=np.ones(source_count, dtype=bool),
                    verification_fail_streaks=np.full(
                        source_count,
                        quarantine_streak,
                        dtype=np.int64,
                    ),
                )
            else:
                candidate_idx = int(local_idx % source_count)
                candidate_pos = pos_arr[candidate_idx : candidate_idx + 1].copy()
                candidate_q = q_arr[candidate_idx : candidate_idx + 1].copy()
                if jitter_sigma > 0.0 and local_idx > 0:
                    candidate_pos = self._project_positions_to_source_prior(
                        candidate_pos
                        + np.random.normal(scale=jitter_sigma, size=candidate_pos.shape)
                    )
                if strength_log_sigma > 0.0 and local_idx > 0:
                    candidate_q = candidate_q * np.exp(
                        np.random.normal(scale=strength_log_sigma, size=candidate_q.shape)
                    )
                    if q_max > 0.0:
                        candidate_q = np.clip(candidate_q, q_min, q_max)
                    else:
                        candidate_q = np.maximum(candidate_q, q_min)
                state = self._state_with_single_quarantined_candidate(
                    base_state,
                    candidate_pos.reshape(3),
                    float(candidate_q[0]),
                    quarantine_streak=quarantine_streak,
                    background=background,
                )
            self.continuous_particles[int(particle_idx)] = IsotopeParticle(
                state=state,
                log_weight=rescue_log_weight,
            )
        self._normalize_continuous_log_weights()
        self.align_continuous_labels()
        self.last_runtime_report_rescue_candidates = source_count
        self.last_runtime_report_rescue_sources = source_count
        self.last_runtime_report_rescue_injected = int(inject_count)
        self.last_runtime_report_rescue_weight = float(rescue_mass)
        return int(inject_count)

    def _state_with_single_quarantined_candidate(
        self,
        base_state: IsotopeState,
        candidate_pos: NDArray[np.float64],
        candidate_strength: float,
        *,
        quarantine_streak: int,
        background: float,
    ) -> IsotopeState:
        """Return a copy of ``base_state`` with one tentative candidate added."""
        state = base_state.copy()
        count = max(int(state.num_sources), 0)
        positions = np.asarray(state.positions, dtype=float).reshape(-1, 3)[:count]
        strengths = np.asarray(state.strengths, dtype=float).reshape(-1)[:count]
        ages = self._metadata_vector(state.ages, count, 1, np.int64)
        low_q = self._metadata_vector(state.low_q_streaks, count, 0, np.int64)
        support = self._metadata_vector(state.support_scores, count, 1.0, float)
        tentative = self._metadata_vector(state.tentative_sources, count, False, bool)
        fail = self._metadata_vector(
            state.verification_fail_streaks,
            count,
            0,
            np.int64,
        )
        candidate = np.asarray(candidate_pos, dtype=float).reshape(1, 3)
        candidate = self._project_positions_to_source_prior(candidate).reshape(3)
        candidate_q = max(float(candidate_strength), max(float(self.config.min_strength), 0.0))
        if float(self.config.birth_q_max) > 0.0:
            candidate_q = min(candidate_q, float(self.config.birth_q_max))
        dedup_radius = max(float(getattr(self.config, "birth_global_rescue_dedup_radius_m", 0.0)), 0.0)
        if count > 0:
            distances = np.linalg.norm(positions - candidate[None, :], axis=1)
            nearest = int(np.argmin(distances))
            if dedup_radius > 0.0 and float(distances[nearest]) <= dedup_radius:
                positions = positions.copy()
                strengths = strengths.copy()
                strengths[nearest] = max(float(strengths[nearest]), candidate_q)
                tentative = tentative.copy()
                fail = fail.copy()
                support = support.copy()
                tentative[nearest] = True
                fail[nearest] = max(int(fail[nearest]), int(quarantine_streak))
                support[nearest] = max(float(support[nearest]), 0.25)
                return IsotopeState(
                    num_sources=count,
                    positions=positions,
                    strengths=strengths,
                    background=float(background),
                    ages=ages,
                    low_q_streaks=low_q,
                    support_scores=support,
                    tentative_sources=tentative,
                    verification_fail_streaks=fail,
                )
        max_sources = None
        if self.config.max_sources is not None:
            max_sources = max(0, int(self.config.max_sources))
        if max_sources is not None and max_sources <= 0:
            return IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3), dtype=float),
                strengths=np.zeros(0, dtype=float),
                background=float(background),
            )
        if max_sources is None or count < max_sources:
            positions = np.vstack([positions, candidate.reshape(1, 3)])
            strengths = np.concatenate([strengths, np.array([candidate_q], dtype=float)])
            ages = np.concatenate([ages, np.array([0], dtype=np.int64)])
            low_q = np.concatenate([low_q, np.array([0], dtype=np.int64)])
            support = np.concatenate([support, np.array([0.25], dtype=float)])
            tentative = np.concatenate([tentative, np.array([True], dtype=bool)])
            fail = np.concatenate(
                [fail, np.array([int(quarantine_streak)], dtype=np.int64)]
            )
        elif count > 0:
            replace = int(np.argmin(strengths))
            positions = positions.copy()
            strengths = strengths.copy()
            ages = ages.copy()
            low_q = low_q.copy()
            support = support.copy()
            tentative = tentative.copy()
            fail = fail.copy()
            positions[replace] = candidate
            strengths[replace] = candidate_q
            ages[replace] = 0
            low_q[replace] = 0
            support[replace] = 0.25
            tentative[replace] = True
            fail[replace] = int(quarantine_streak)
        count_new = int(strengths.size)
        return IsotopeState(
            num_sources=count_new,
            positions=positions.astype(float, copy=True).reshape(count_new, 3),
            strengths=strengths.astype(float, copy=True).reshape(count_new),
            background=float(background),
            ages=ages.astype(np.int64, copy=True).reshape(count_new),
            low_q_streaks=low_q.astype(np.int64, copy=True).reshape(count_new),
            support_scores=support.astype(float, copy=True).reshape(count_new),
            tentative_sources=tentative.astype(bool, copy=True).reshape(count_new),
            verification_fail_streaks=fail.astype(np.int64, copy=True).reshape(count_new),
        )

    @staticmethod
    def _metadata_vector(
        values: NDArray[np.float64] | NDArray[np.int64] | NDArray[np.bool_] | None,
        count: int,
        default: float | int | bool,
        dtype: object,
    ) -> NDArray:
        """Return a metadata vector with the requested length and dtype."""
        if values is None:
            return np.full(count, default, dtype=dtype)
        arr = np.asarray(values, dtype=dtype).reshape(-1)
        if arr.size < count:
            pad = np.full(count - arr.size, default, dtype=dtype)
            arr = np.concatenate([arr, pad])
        return arr[:count].astype(dtype, copy=True)

    def _runtime_strength_absorption_soft_cap(self) -> float:
        """Return the runtime source-strength soft cap in cps, or zero if disabled."""
        mean = max(float(getattr(self.config, "source_strength_prior_mean", 0.0)), 0.0)
        if mean <= 0.0:
            return 0.0
        multiple = max(
            1.0,
            float(
                getattr(
                    self.config,
                    "source_strength_absorption_q_multiple",
                    4.0,
                )
            ),
        )
        return mean * multiple

    def _source_mode_preserving_indices(
        self,
        weights: NDArray[np.float64],
    ) -> NDArray[np.int64]:
        """Return particle indices that preserve distinct posterior source modes."""
        self.last_mode_preserving_dynamic_spatial_summary = []
        if not bool(self.config.mode_preserving_resample):
            self.last_mode_preserving_strata_summary = {}
            self.last_mode_preserving_selected_strata = []
            self.last_mode_preserving_cardinality_summary = {}
            self.last_mode_preserving_selected_cardinalities = []
            return np.zeros(0, dtype=np.int64)
        max_modes = max(0, int(self.config.mode_preserving_max_modes))
        per_mode = max(0, int(self.config.mode_preserving_particles_per_mode))
        radius = max(float(self.config.mode_preserving_radius_m), 1e-9)
        if max_modes <= 0 or per_mode <= 0:
            self.last_mode_preserving_strata_summary = {}
            self.last_mode_preserving_selected_strata = []
            self.last_mode_preserving_cardinality_summary = {}
            self.last_mode_preserving_selected_cardinalities = []
            return np.zeros(0, dtype=np.int64)
        positions: list[NDArray[np.float64]] = []
        scores: list[float] = []
        particle_indices: list[int] = []
        for particle_idx, particle in enumerate(self.continuous_particles):
            st = particle.state
            if st.num_sources <= 0 or st.positions.size == 0:
                continue
            count = min(int(st.num_sources), st.positions.shape[0], st.strengths.size)
            particle_weight = float(weights[particle_idx])
            for source_idx in range(count):
                strength = max(float(st.strengths[source_idx]), 0.0)
                if strength <= float(self.config.min_strength):
                    continue
                support = 0.0
                if st.support_scores is not None and source_idx < st.support_scores.size:
                    support = max(float(st.support_scores[source_idx]), 0.0)
                support_boost = 1.0 + max(
                    0.0,
                    float(
                        getattr(
                            self.config,
                            "mode_preserving_support_score_weight",
                            0.0,
                        )
                    ),
                ) * np.log1p(support)
                tentative_boost = 1.0
                if (
                    st.tentative_sources is not None
                    and source_idx < st.tentative_sources.size
                    and bool(st.tentative_sources[source_idx])
                ):
                    tentative_boost = max(
                        1.0,
                        float(
                            getattr(
                                self.config,
                                "mode_preserving_tentative_boost",
                                1.0,
                            )
                        ),
                    )
                residual_boost = (
                    max(
                        1.0,
                        float(
                            getattr(
                                self.config,
                                "mode_preserving_residual_boost",
                                1.0,
                            )
                        ),
                    )
                    if bool(getattr(self, "last_birth_residual_gate_passed", False))
                    else 1.0
                )
                positions.append(np.asarray(st.positions[source_idx], dtype=float))
                scores.append(
                    particle_weight
                    * strength
                    * support_boost
                    * tentative_boost
                    * residual_boost
                )
                particle_indices.append(particle_idx)
        if not scores:
            self.last_mode_preserving_strata_summary = {}
            self.last_mode_preserving_selected_strata = []
            self.last_mode_preserving_cardinality_summary = {}
            self.last_mode_preserving_selected_cardinalities = []
            return np.zeros(0, dtype=np.int64)

        pos_arr = np.vstack(positions)
        score_arr = np.asarray(scores, dtype=float)
        particle_arr = np.asarray(particle_indices, dtype=np.int64)
        strata_enabled = bool(
            getattr(self.config, "mode_preserving_surface_strata", True)
        ) and self._source_prior_is_surface()
        height_bin_m = max(
            0.0,
            float(getattr(self.config, "mode_preserving_height_bin_m", 0.0)),
        )
        labels: list[tuple[str, int]] = [("all", 0)] * int(pos_arr.shape[0])
        if strata_enabled:
            kinds = source_surface_kinds(
                pos_arr,
                self._source_prior_environment(),
                self.obstacle_grid,
                obstacle_height_m=self.obstacle_height_m,
                tolerance_m=max(1.0e-6, 0.25 * radius),
            )
            if height_bin_m > 0.0:
                height_bins = np.floor(pos_arr[:, 2] / height_bin_m).astype(np.int64)
            else:
                height_bins = np.zeros(pos_arr.shape[0], dtype=np.int64)
            labels = [
                (
                    str(kind) if kind is not None else "off_surface",
                    int(height_bin),
                )
                for kind, height_bin in zip(kinds, height_bins)
            ]
        high_surface_extra = max(
            0,
            int(getattr(self.config, "mode_preserving_high_surface_extra_particles", 0)),
        )
        room_z = max(float(self._source_prior_environment().size_z), 1.0e-9)
        high_z_threshold = (
            float(
                np.clip(
                    float(
                        getattr(
                            self.config,
                            "mode_preserving_high_surface_z_fraction",
                            0.75,
                        )
                    ),
                    0.0,
                    1.0,
                )
            )
            * room_z
        )
        total_score = max(float(np.sum(score_arr)), 1e-300)
        strata_scores: dict[str, float] = {}
        for label, score in zip(labels, score_arr):
            key = f"{label[0]}:zbin{int(label[1])}"
            strata_scores[key] = strata_scores.get(key, 0.0) + float(score)
        self.last_mode_preserving_strata_summary = {
            key: float(value) / total_score
            for key, value in sorted(strata_scores.items())
        }
        min_score = (
            max(float(self.config.mode_preserving_min_weight_fraction), 0.0)
            * total_score
        )
        order = np.argsort(score_arr)[::-1]
        centers = np.empty((len(order), 3), dtype=float)
        cluster_scores = np.empty(len(order), dtype=float)
        cluster_labels: list[tuple[str, int]] = []
        cluster_members: list[list[int]] = []
        cluster_count = 0
        for entry_idx in order:
            pos = pos_arr[entry_idx]
            entry_score = float(score_arr[entry_idx])
            entry_label = labels[int(entry_idx)]
            if cluster_count > 0:
                distances = np.linalg.norm(centers[:cluster_count] - pos, axis=1)
                label_matches = np.asarray(
                    [
                        cluster_labels[int(cluster_idx)] == entry_label
                        for cluster_idx in range(cluster_count)
                    ],
                    dtype=bool,
                )
                matches = np.flatnonzero((distances <= radius) & label_matches)
            else:
                matches = np.zeros(0, dtype=np.int64)
            if matches.size:
                cluster_idx = int(matches[0])
                old_weight = float(cluster_scores[cluster_idx])
                weight = old_weight + entry_score
                centers[cluster_idx] = (
                    centers[cluster_idx] * old_weight + pos * entry_score
                ) / max(weight, 1.0e-300)
                cluster_scores[cluster_idx] = weight
                cluster_members[cluster_idx].append(int(entry_idx))
                continue
            centers[cluster_count] = pos
            cluster_scores[cluster_count] = entry_score
            cluster_labels.append(entry_label)
            cluster_members.append([int(entry_idx)])
            cluster_count += 1
        sorted_clusters = np.argsort(cluster_scores[:cluster_count])[::-1]
        selected_clusters: list[int] = []
        if strata_enabled:
            seen_labels: set[tuple[str, int]] = set()
            for cluster_idx in sorted_clusters:
                label = cluster_labels[int(cluster_idx)]
                if label in seen_labels:
                    continue
                selected_clusters.append(int(cluster_idx))
                seen_labels.add(label)
                if len(selected_clusters) >= max_modes:
                    break
        selected_set = set(selected_clusters)
        for cluster_idx in sorted_clusters:
            if len(selected_clusters) >= max_modes:
                break
            cluster_int = int(cluster_idx)
            if cluster_int in selected_set:
                continue
            selected_clusters.append(cluster_int)
            selected_set.add(cluster_int)

        dynamic_spatial_enabled = bool(
            getattr(
                self.config,
                "mode_preserving_dynamic_spatial_allocation",
                False,
            )
        )
        dynamic_spatial_extra = max(
            0,
            int(
                getattr(
                    self.config,
                    "mode_preserving_dynamic_spatial_extra_particles",
                    0,
                )
            ),
        )
        dynamic_spatial_min_fraction = max(
            0.0,
            float(
                getattr(
                    self.config,
                    "mode_preserving_dynamic_spatial_min_score_fraction",
                    0.005,
                )
            ),
        )
        protected: list[int] = []
        protected_lookup: set[int] = set()
        selected_details: list[dict[str, object]] = []
        dynamic_spatial_details: list[dict[str, object]] = []
        for cluster_idx in selected_clusters:
            cluster_int = int(cluster_idx)
            label = cluster_labels[cluster_int]
            members = cluster_members[cluster_int]
            ranked_members = sorted(
                members,
                key=lambda idx: float(score_arr[idx]),
                reverse=True,
            )
            protected_for_cluster: list[int] = []
            cluster_score = float(cluster_scores[cluster_int])
            accepted = cluster_score >= min_score
            if not accepted:
                selected_details.append(
                    {
                        "surface": str(label[0]),
                        "height_bin": int(label[1]),
                        "center": [
                            float(value) for value in centers[cluster_int].astype(float)
                        ],
                        "high_surface": bool(
                            str(label[0]) in {"ceiling", "high_wall", "obstacle_top"}
                            or float(centers[cluster_int][2]) >= high_z_threshold
                        ),
                        "score": cluster_score,
                        "score_fraction": cluster_score / total_score,
                        "member_count": int(len(members)),
                        "protected_particles": [],
                        "protected_count": 0,
                        "accepted": False,
                    }
                )
                continue
            high_surface = bool(
                str(label[0]) in {"ceiling", "high_wall", "obstacle_top"}
                or float(centers[cluster_int][2]) >= high_z_threshold
            )
            target_per_mode = per_mode + (high_surface_extra if high_surface else 0)
            added = 0
            for member_idx in ranked_members:
                particle_idx = int(particle_arr[member_idx])
                if particle_idx in protected_lookup:
                    continue
                protected.append(particle_idx)
                protected_lookup.add(particle_idx)
                protected_for_cluster.append(particle_idx)
                added += 1
                if added >= target_per_mode:
                    break
            dynamic_spatial_added: dict[int, int] = {}
            dynamic_spatial_active = bool(
                dynamic_spatial_enabled
                and dynamic_spatial_extra > 0
                and (cluster_score / total_score) >= dynamic_spatial_min_fraction
            )
            if dynamic_spatial_active:
                source_counts = sorted(
                    {
                        int(
                            self.continuous_particles[int(particle_arr[member_idx])]
                            .state
                            .num_sources
                        )
                        for member_idx in ranked_members
                    }
                )
                for source_count in source_counts:
                    added_for_count = 0
                    for member_idx in ranked_members:
                        particle_idx = int(particle_arr[member_idx])
                        particle_count = int(
                            self.continuous_particles[particle_idx].state.num_sources
                        )
                        if particle_count != int(source_count):
                            continue
                        if particle_idx in protected_lookup:
                            continue
                        protected.append(particle_idx)
                        protected_lookup.add(particle_idx)
                        protected_for_cluster.append(particle_idx)
                        added_for_count += 1
                        if added_for_count >= dynamic_spatial_extra:
                            break
                    if added_for_count > 0:
                        dynamic_spatial_added[int(source_count)] = int(added_for_count)
                if dynamic_spatial_added:
                    dynamic_spatial_details.append(
                        {
                            "surface": str(label[0]),
                            "height_bin": int(label[1]),
                            "center": [
                                float(value)
                                for value in centers[cluster_int].astype(float)
                            ],
                            "score_fraction": cluster_score / total_score,
                            "extra_per_cardinality": int(dynamic_spatial_extra),
                            "protected_by_cardinality": {
                                str(key): int(value)
                                for key, value in sorted(
                                    dynamic_spatial_added.items()
                                )
                            },
                        }
                    )
            selected_details.append(
                {
                    "surface": str(label[0]),
                    "height_bin": int(label[1]),
                    "center": [float(value) for value in centers[cluster_int].astype(float)],
                    "high_surface": bool(high_surface),
                    "score": cluster_score,
                    "score_fraction": cluster_score / total_score,
                    "member_count": int(len(members)),
                    "protected_particles": protected_for_cluster,
                    "protected_count": int(len(protected_for_cluster)),
                    "accepted": True,
                    "dynamic_spatial_protected": bool(dynamic_spatial_active),
                    "dynamic_spatial_counts": {
                        str(key): int(value)
                        for key, value in sorted(dynamic_spatial_added.items())
                    },
                }
            )
        self.last_mode_preserving_dynamic_spatial_summary = dynamic_spatial_details
        cardinality_details: list[dict[str, object]] = []
        self.last_mode_preserving_cardinality_summary = {}
        self.last_mode_preserving_selected_cardinalities = []
        if bool(getattr(self.config, "mode_preserving_cardinality_strata", True)):
            min_per_cardinality = max(
                0,
                int(
                    getattr(
                        self.config,
                        "mode_preserving_min_particles_per_cardinality",
                        0,
                    )
                ),
            )
            external_cardinalities = set()
            if bool(
                getattr(
                    self.config,
                    "mode_preserving_report_cardinality_strata",
                    True,
                )
            ):
                external_cardinalities = set(
                    int(value)
                    for value in getattr(
                        self,
                        "_external_protected_cardinalities",
                        set(),
                    )
                    if int(value) >= 0
                )
            extra_per_external = max(
                0,
                int(
                    getattr(
                        self.config,
                        "mode_preserving_report_cardinality_extra_particles",
                        0,
                    )
                ),
            )
            dynamic_cardinality = bool(
                getattr(
                    self.config,
                    "mode_preserving_dynamic_cardinality_allocation",
                    False,
                )
            )
            dynamic_extra = max(
                0,
                int(
                    getattr(
                        self.config,
                        "mode_preserving_dynamic_cardinality_extra_particles",
                        0,
                    )
                ),
            )
            dynamic_min_mass = max(
                0.0,
                float(
                    getattr(
                        self.config,
                        "mode_preserving_dynamic_cardinality_min_mass",
                        0.02,
                    )
                ),
            )
            dynamic_entropy_min = max(
                0.0,
                float(
                    getattr(
                        self.config,
                        "mode_preserving_dynamic_cardinality_entropy_min",
                        0.5,
                    )
                ),
            )
            if min_per_cardinality > 0:
                particle_counts = np.asarray(
                    [
                        max(0, int(particle.state.num_sources))
                        for particle in self.continuous_particles
                    ],
                    dtype=np.int64,
                )
                particle_weights = np.asarray(weights, dtype=float).reshape(-1)
                total_mass = max(float(np.sum(particle_weights)), 1.0e-300)
                protected_lookup = set(int(value) for value in protected)
                source_counts = sorted(set(int(value) for value in particle_counts))
                mass_by_count: dict[int, float] = {}
                for source_count in source_counts:
                    member_indices = np.flatnonzero(particle_counts == source_count)
                    mass_by_count[int(source_count)] = (
                        float(np.sum(particle_weights[member_indices])) / total_mass
                    )
                mass_arr = np.asarray(list(mass_by_count.values()), dtype=float)
                positive_mass = mass_arr[mass_arr > 0.0]
                cardinality_entropy = 0.0
                if positive_mass.size > 1:
                    cardinality_entropy = float(
                        -np.sum(positive_mass * np.log(positive_mass))
                    )
                dynamic_active = bool(
                    dynamic_cardinality
                    and dynamic_extra > 0
                    and cardinality_entropy >= dynamic_entropy_min
                )
                for source_count in source_counts:
                    member_indices = np.flatnonzero(particle_counts == source_count)
                    if member_indices.size == 0:
                        continue
                    mass_fraction = mass_by_count[int(source_count)]
                    self.last_mode_preserving_cardinality_summary[
                        str(int(source_count))
                    ] = mass_fraction
                    ranked = member_indices[
                        np.argsort(particle_weights[member_indices])[::-1]
                    ]
                    protected_for_count: list[int] = []
                    externally_protected = int(source_count) in external_cardinalities
                    target_for_count = min_per_cardinality + (
                        extra_per_external if externally_protected else 0
                    )
                    dynamically_protected = bool(
                        dynamic_active and mass_fraction >= dynamic_min_mass
                    )
                    if dynamically_protected:
                        target_for_count += dynamic_extra
                    for particle_idx in ranked:
                        idx_int = int(particle_idx)
                        if idx_int in protected_lookup:
                            continue
                        protected.append(idx_int)
                        protected_lookup.add(idx_int)
                        protected_for_count.append(idx_int)
                        if len(protected_for_count) >= target_for_count:
                            break
                    cardinality_details.append(
                        {
                            "num_sources": int(source_count),
                            "mass_fraction": mass_fraction,
                            "member_count": int(member_indices.size),
                            "externally_protected": bool(externally_protected),
                            "dynamically_protected": bool(dynamically_protected),
                            "cardinality_entropy": cardinality_entropy,
                            "target_protected_count": int(target_for_count),
                            "protected_particles": protected_for_count,
                            "protected_count": int(len(protected_for_count)),
                        }
                    )
        self.last_mode_preserving_selected_cardinalities = cardinality_details
        self.last_mode_preserving_selected_strata = selected_details
        return np.asarray(protected, dtype=np.int64)

    def _inject_mode_preserving_indices(
        self,
        indices: NDArray[np.int64],
        protected: NDArray[np.int64],
    ) -> NDArray[np.int64]:
        """Inject protected particle indices into a resampling draw."""
        idx = np.asarray(indices, dtype=np.int64).copy()
        protected_arr = np.asarray(protected, dtype=np.int64)
        if idx.size == 0 or protected_arr.size == 0:
            return idx
        valid = protected_arr[
            (protected_arr >= 0) & (protected_arr < len(self.continuous_particles))
        ]
        if valid.size == 0:
            return idx
        unique_protected = []
        for value in valid.tolist():
            if value not in unique_protected:
                unique_protected.append(int(value))
        counts = np.bincount(idx, minlength=len(self.continuous_particles))
        missing = [value for value in unique_protected if counts[value] == 0]
        if not missing:
            return idx
        replace_slots: list[int] = []
        for slot, value in enumerate(idx):
            if counts[value] > 1:
                replace_slots.append(slot)
                counts[value] -= 1
                if len(replace_slots) >= len(missing):
                    break
        if len(replace_slots) < len(missing):
            for slot, value in enumerate(idx):
                if slot in replace_slots or value in unique_protected:
                    continue
                replace_slots.append(slot)
                if len(replace_slots) >= len(missing):
                    break
        for slot, value in zip(replace_slots, missing):
            idx[slot] = int(value)
        self.last_mode_preserved_count += int(len(replace_slots))
        return idx

    def _cardinality_preserving_resample_draw(
        self,
        weights: NDArray[np.float64],
        protected_indices: NDArray[np.int64] | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]] | None:
        """Return resampling indices and log-weights preserving source-count mass.

        Spatial source-mode protection is applied inside each cardinality group
        so that preserving K-mass does not accidentally discard distinct
        same-isotope modes during a shield burst.
        """
        if not bool(self.config.cardinality_preserving_resample):
            return None
        min_stations = max(
            0,
            int(getattr(self.config, "cardinality_preserving_min_stations", 0)),
        )
        if self._distinct_observed_station_count() < min_stations:
            return None
        if bool(
            getattr(
                self.config,
                "cardinality_preserving_require_confirmed_structure",
                False,
            )
        ) and not self._confirmed_source_structure():
            return None
        n_particles = len(self.continuous_particles)
        if n_particles <= 0 or weights.size != n_particles:
            return None
        labels = np.asarray(
            [
                max(0, int(particle.state.num_sources))
                for particle in self.continuous_particles
            ],
            dtype=np.int64,
        )
        unique_labels = np.unique(labels)
        if unique_labels.size <= 1:
            return None
        w = np.asarray(weights, dtype=np.float64)
        w = np.clip(w, 0.0, np.inf)
        total = float(np.sum(w))
        if not np.isfinite(total) or total <= 0.0:
            return None
        w = w / total
        masses = np.array([float(np.sum(w[labels == label])) for label in unique_labels])
        active = masses > 0.0
        if not np.any(active):
            return None
        unique_labels = unique_labels[active]
        masses = masses[active]
        desired = masses * float(n_particles)
        counts = np.floor(desired).astype(np.int64)
        counts = np.maximum(counts, 1)
        while int(np.sum(counts)) > n_particles:
            removable = np.where(counts > 1)[0]
            if removable.size == 0:
                break
            idx = int(removable[np.argmin(desired[removable] - np.floor(desired[removable]))])
            counts[idx] -= 1
        while int(np.sum(counts)) < n_particles:
            idx = int(np.argmax(desired - counts))
            counts[idx] += 1
        drawn: list[int] = []
        log_weights_after: list[float] = []
        protected_arr = (
            np.asarray(protected_indices, dtype=np.int64)
            if protected_indices is not None
            else np.zeros(0, dtype=np.int64)
        )
        for label, mass, count in zip(unique_labels, masses, counts):
            group_idx = np.flatnonzero(labels == label)
            if group_idx.size == 0 or count <= 0:
                continue
            local_w = w[group_idx]
            local_draw = systematic_resample_count(local_w, count=int(count))
            selected = group_idx[local_draw]
            if protected_arr.size:
                valid_protected = protected_arr[
                    (protected_arr >= 0) & (protected_arr < n_particles)
                ]
                group_protected = valid_protected[labels[valid_protected] == label]
                selected = self._inject_mode_preserving_indices(
                    selected,
                    group_protected,
                )
            drawn.extend(int(value) for value in selected.tolist())
            per_particle_weight = float(mass) / max(int(count), 1)
            log_weights_after.extend(
                [float(np.log(max(per_particle_weight, 1.0e-300)))] * int(count)
            )
        if len(drawn) != n_particles or len(log_weights_after) != n_particles:
            return None
        return np.asarray(drawn, dtype=np.int64), np.asarray(log_weights_after, dtype=float)

    def _maybe_resample_continuous(
        self,
        *,
        disable_regularize: bool = False,
        roughening_scale: float = 1.0,
    ) -> None:
        """ESS check and systematic resampling for continuous particles (Sec. 3.3.4, Eq. 3.29)."""
        w = np.asarray(self.continuous_weights, dtype=np.float64)
        self.last_mode_preserved_count = 0
        self.last_mode_preserving_strata_summary = {}
        self.last_mode_preserving_selected_strata = []
        self.last_mode_preserving_cardinality_summary = {}
        self.last_mode_preserving_selected_cardinalities = []
        if w.size == 0:
            self.last_ess = 0.0
            self.last_ess_pre = 0.0
            self.last_ess_post = 0.0
            self.last_resample_ess = False
            return
        ess = 1.0 / max(np.sum(w**2), 1e-12)
        self.last_ess = float(ess)
        self.last_ess_pre = float(ess)
        self.last_ess_post = None
        self.last_resample_ess = False
        if ess < self.config.resample_threshold * self.N:
            debug_timing = _pf_debug_timing_enabled()
            debug_start = time.perf_counter()
            if debug_timing:
                print(
                    f"[pf_internal] isotope={self.isotope} "
                    f"phase=resample_start ess={float(ess):.3f} "
                    f"n={len(self.continuous_particles)}",
                    flush=True,
                )
            self.last_resample_ess = True
            self.last_resample_count += 1
            logw = np.log(np.clip(w, 1e-300, 1.0))
            protected_idx = self._source_mode_preserving_indices(w)
            if debug_timing:
                print(
                    f"[pf_internal] isotope={self.isotope} "
                    f"phase=mode_protection_done elapsed={time.perf_counter() - debug_start:.3f}s "
                    f"protected={int(protected_idx.size)}",
                    flush=True,
                )
            cardinality_draw = self._cardinality_preserving_resample_draw(
                w,
                protected_indices=protected_idx,
            )
            if debug_timing:
                print(
                    f"[pf_internal] isotope={self.isotope} "
                    f"phase=cardinality_draw_done elapsed={time.perf_counter() - debug_start:.3f}s "
                    f"used={cardinality_draw is not None}",
                    flush=True,
                )
            if cardinality_draw is None:
                idx = systematic_resample(logw)
                idx = self._inject_mode_preserving_indices(idx, protected_idx)
                log_weights_after = np.full(
                    idx.size,
                    float(-np.log(max(idx.size, 1))),
                    dtype=float,
                )
            else:
                idx, log_weights_after = cardinality_draw
            self.continuous_particles = [self.continuous_particles[i].state.copy() for i in idx]
            self.continuous_particles = [
                IsotopeParticle(state=st, log_weight=float(log_weight))
                for st, log_weight in zip(self.continuous_particles, log_weights_after)
            ]
            post_w = np.asarray(self.continuous_weights, dtype=np.float64)
            self.last_ess_post = float(1.0 / max(np.sum(post_w**2), 1.0e-12))
            roughening_scale = max(0.0, float(roughening_scale))
            surface_rejuvenation = bool(
                getattr(self.config, "surface_rejuvenation_enable", True)
            )
            if (
                surface_rejuvenation
                and not disable_regularize
                and roughening_scale > 0.0
            ):
                mult = self._roughening_multiplier()
                sigma_pos = (
                    self._roughening_sigma_pos(len(self.continuous_particles))
                    * mult
                    * roughening_scale
                )
                self.regularize_continuous(
                    sigma_pos=sigma_pos,
                    strength_log_sigma=(
                        self.config.strength_log_sigma * mult * roughening_scale
                    ),
                    p_birth=self.config.p_birth,
                    p_kill=self.config.p_kill,
                    intensity_threshold=self.config.min_strength,
                )
            self._resample_count_in_observation += 1
            if debug_timing:
                print(
                    f"[pf_internal] isotope={self.isotope} "
                    f"phase=resample_done elapsed={time.perf_counter() - debug_start:.3f}s",
                    flush=True,
                )

    def _maybe_resample_after_structural_update(self) -> bool:
        """Resample after delayed structural moves if their weight ratios collapse ESS."""
        roughening_scale = max(
            0.0,
            float(self.config.deferred_resample_roughening_scale),
        )
        disable_regularize = roughening_scale <= 0.0
        self._maybe_resample_continuous(
            disable_regularize=disable_regularize,
            roughening_scale=roughening_scale,
        )
        resampled = bool(self.last_resample_ess)
        if resampled:
            self._trigger_adapt_cooldown()
        return resampled

    def _cardinality_neutral_refit_corrections(
        self,
        corrections: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Preserve model-order mass while reweighting particles inside each cardinality."""
        corr = np.asarray(corrections, dtype=float).copy()
        if corr.size != len(self.continuous_particles) or corr.size == 0:
            return corr
        logw = np.asarray(
            [particle.log_weight for particle in self.continuous_particles],
            dtype=float,
        )
        source_counts = np.asarray(
            [int(max(0, particle.state.num_sources)) for particle in self.continuous_particles],
            dtype=int,
        )
        for source_count in np.unique(source_counts):
            mask = source_counts == int(source_count)
            if not np.any(mask):
                continue
            old_mass = float(logsumexp(logw[mask]))
            new_mass = float(logsumexp(logw[mask] + corr[mask]))
            if np.isfinite(old_mass) and np.isfinite(new_mass):
                corr[mask] += old_mass - new_mass
        return corr

    def _label_scales(
        self,
        ref_positions: NDArray[np.float64],
        ref_strengths: NDArray[np.float64],
    ) -> tuple[float, float]:
        """Return (pos_scale, strength_scale) for label alignment costs."""
        if self.config.label_pos_scale is not None:
            pos_scale = float(self.config.label_pos_scale)
        else:
            span = np.array(self.config.position_max, dtype=float) - np.array(self.config.position_min, dtype=float)
            pos_scale = float(np.linalg.norm(span))
        if pos_scale <= 0.0:
            pos_scale = 1.0
        if self.config.label_strength_scale is not None:
            strength_scale = float(self.config.label_strength_scale)
        else:
            positive = ref_strengths[ref_strengths > 0]
            strength_scale = float(np.median(positive)) if positive.size else 1.0
        if strength_scale <= 0.0:
            strength_scale = 1.0
        return pos_scale, strength_scale

    def _label_cost_matrix(
        self,
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
        ref_positions: NDArray[np.float64],
        ref_strengths: NDArray[np.float64],
        pos_scale: float,
        strength_scale: float,
    ) -> NDArray[np.float64]:
        """Compute the label-alignment cost matrix between particle and reference sources."""
        if positions.size == 0 or ref_positions.size == 0:
            return np.zeros((positions.shape[0], ref_positions.shape[0]), dtype=float)
        if positions.shape[0] * ref_positions.shape[0] <= 64:
            pos_diff = positions[:, None, :] - ref_positions[None, :, :]
            pos_cost = np.linalg.norm(pos_diff, axis=-1) / float(pos_scale)
            str_cost = (
                np.abs(strengths[:, None] - ref_strengths[None, :])
                / float(strength_scale)
            )
            return np.asarray(
                self.config.label_pos_weight * pos_cost
                + self.config.label_strength_weight * str_cost,
                dtype=float,
            )
        self._gpu_enabled()
        import torch
        from pf import gpu_utils

        device = gpu_utils.resolve_device(self.config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.config.gpu_dtype)
        pos_t = torch.as_tensor(positions, device=device, dtype=dtype)
        ref_pos_t = torch.as_tensor(ref_positions, device=device, dtype=dtype)
        str_t = torch.as_tensor(strengths, device=device, dtype=dtype)
        ref_str_t = torch.as_tensor(ref_strengths, device=device, dtype=dtype)
        diff = pos_t[:, None, :] - ref_pos_t[None, :, :]
        pos_cost = torch.linalg.norm(diff, dim=-1) / float(pos_scale)
        str_cost = torch.abs(str_t[:, None] - ref_str_t[None, :]) / float(strength_scale)
        cost = self.config.label_pos_weight * pos_cost + self.config.label_strength_weight * str_cost
        return cost.detach().cpu().numpy()

    def _align_particle_to_reference(
        self,
        particle: IsotopeParticle,
        ref_positions: NDArray[np.float64],
        ref_strengths: NDArray[np.float64],
        pos_scale: float,
        strength_scale: float,
    ) -> None:
        """Reorder a particle's sources to best match the reference ordering."""
        from scipy.optimize import linear_sum_assignment

        st = particle.state
        if st.num_sources == 0 or ref_positions.size == 0:
            return
        self._ensure_source_metadata(st)
        cost = self._label_cost_matrix(
            positions=st.positions,
            strengths=st.strengths,
            ref_positions=ref_positions,
            ref_strengths=ref_strengths,
            pos_scale=pos_scale,
            strength_scale=strength_scale,
        )
        n_rows, n_cols = cost.shape
        size = max(n_rows, n_cols)
        padded = np.full((size, size), float(self.config.label_missing_cost), dtype=float)
        padded[:n_rows, :n_cols] = cost
        row_ind, col_ind = linear_sum_assignment(padded)
        assigned = {c: r for r, c in zip(row_ind, col_ind) if r < n_rows and c < n_cols}
        ordered_pos: list[NDArray[np.float64]] = []
        ordered_str: list[float] = []
        ordered_rows: list[int] = []
        used_rows: set[int] = set()
        for ref_idx in range(n_cols):
            row = assigned.get(ref_idx)
            if row is None:
                continue
            ordered_pos.append(st.positions[row])
            ordered_str.append(float(st.strengths[row]))
            ordered_rows.append(row)
            used_rows.add(row)
        for row in range(n_rows):
            if row in used_rows:
                continue
            ordered_pos.append(st.positions[row])
            ordered_str.append(float(st.strengths[row]))
            ordered_rows.append(row)
        if ordered_pos:
            st.positions = np.vstack(ordered_pos)
            st.strengths = np.array(ordered_str, dtype=float)
            st.ages = st.ages[ordered_rows]
            st.low_q_streaks = st.low_q_streaks[ordered_rows]
            st.support_scores = st.support_scores[ordered_rows]
            st.tentative_sources = st.tentative_sources[ordered_rows]
            st.verification_fail_streaks = st.verification_fail_streaks[ordered_rows]
            st.num_sources = st.positions.shape[0]

    def _reference_from_particles(self, ref_count: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute a weighted reference ordering from the aligned particle set."""
        if ref_count <= 0:
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        w = self.continuous_weights
        positions = np.zeros((ref_count, 3), dtype=float)
        strengths = np.zeros(ref_count, dtype=float)
        for j in range(ref_count):
            pos_list = []
            str_list = []
            w_list = []
            for wi, p in zip(w, self.continuous_particles):
                if p.state.num_sources > j:
                    pos_list.append(p.state.positions[j])
                    str_list.append(p.state.strengths[j])
                    w_list.append(wi)
            if not w_list:
                continue
            wj = np.array(w_list, dtype=float)
            wj = wj / max(np.sum(wj), 1e-12)
            pos_arr = np.vstack(pos_list)
            str_arr = np.array(str_list, dtype=float)
            positions[j] = np.sum(wj[:, None] * pos_arr, axis=0)
            strengths[j] = float(np.sum(wj * str_arr))
        return positions, strengths

    def align_continuous_labels(self) -> None:
        """
        Align per-particle source ordering to mitigate label switching.

        Uses Hungarian assignment against a reference ordering built from the
        highest-weight particle, then refines the reference iteratively.
        """
        if not self.config.label_enable or not self.continuous_particles:
            return
        ref_state = self._label_reference or self.best_particle().state
        if ref_state.num_sources == 0:
            return
        if ref_state.num_sources == 1:
            self._label_reference = ref_state.copy()
            return
        ref_positions = ref_state.positions.copy()
        ref_strengths = ref_state.strengths.copy()
        pos_scale, strength_scale = self._label_scales(ref_positions, ref_strengths)
        for _ in range(max(1, int(self.config.label_alignment_iters))):
            for particle in self.continuous_particles:
                self._align_particle_to_reference(
                    particle=particle,
                    ref_positions=ref_positions,
                    ref_strengths=ref_strengths,
                    pos_scale=pos_scale,
                    strength_scale=strength_scale,
                )
            ref_positions, ref_strengths = self._reference_from_particles(ref_positions.shape[0])
        self._label_reference = IsotopeState(
            num_sources=ref_positions.shape[0],
            positions=ref_positions,
            strengths=ref_strengths,
            background=0.0,
        )

    def adapt_num_particles(self, *, ess_pre: float | None = None, resampled: bool = False) -> None:
        """
        Optional: adapt N based on variance/entropy of weights (Chapter 3.3.4).

        Uses ess_pre when provided to avoid the resampling inflation of ESS.
        Resampling or cooldown windows only allow growth.
        """
        if not self.continuous_particles:
            self.last_n_after_adapt = 0
            return
        min_particles = (
            max(1, int(self.config.min_particles))
            if self.config.min_particles is not None
            else max(1, int(self.config.num_particles))
        )
        max_particles = (
            max(1, int(self.config.max_particles))
            if self.config.max_particles is not None
            else max(1, int(self.config.num_particles))
        )
        if ess_pre is None:
            w = self.continuous_weights
            if w.size == 0:
                ess_pre = 0.0
            else:
                ess_pre = float(1.0 / max(np.sum(w**2), 1e-12))
        ess_ratio = float(ess_pre) / max(len(self.continuous_particles), 1)
        allow_shrink = not resampled and self._adapt_cooldown_remaining <= 0
        if ess_ratio < self.config.ess_low and len(self.continuous_particles) < max_particles:
            grown = max(len(self.continuous_particles) + 1, int(len(self.continuous_particles) * 1.25))
            target = min(max_particles, grown)
            self._resample_continuous_to(target, jitter=True)
        elif allow_shrink and ess_ratio > self.config.ess_high and len(self.continuous_particles) > min_particles:
            target = max(min_particles, int(len(self.continuous_particles) * 0.8))
            self._resample_continuous_to(target, jitter=False)
        self.last_n_after_adapt = int(len(self.continuous_particles))

    def _resample_continuous_to(self, target_n: int, jitter: bool = False) -> None:
        """Resample the continuous particles to a new population size."""
        target_n = max(1, int(target_n))
        self.last_resample_count += 1
        w = self.continuous_weights
        protected_idx = self._source_mode_preserving_indices(w)
        idx = np.random.choice(len(self.continuous_particles), size=target_n, p=w)
        idx = self._inject_mode_preserving_indices(idx, protected_idx)
        states = [self.continuous_particles[i].state.copy() for i in idx]
        self.continuous_particles = [
            IsotopeParticle(state=st, log_weight=float(-np.log(target_n))) for st in states
        ]
        self.N = target_n
        self.config.num_particles = target_n
        if jitter:
            mult = self._roughening_multiplier()
            sigma_pos = self._roughening_sigma_pos(len(self.continuous_particles)) * mult
            self.regularize_continuous(
                sigma_pos=sigma_pos,
                strength_log_sigma=self.config.strength_log_sigma * mult,
                p_birth=self.config.p_birth,
                p_kill=self.config.p_kill,
                intensity_threshold=self.config.min_strength,
            )
        self._resample_count_in_observation += 1

    def best_particle(self) -> IsotopeParticle:
        """Return the particle with maximum log_weight."""
        return max(self.continuous_particles, key=lambda p: p.log_weight)

    def _resize_metadata_array(
        self,
        arr: NDArray[np.float64] | NDArray[np.int64] | None,
        size: int,
        fill_value: float,
        dtype: type,
    ) -> NDArray:
        """Resize or initialize a metadata array to a target length."""
        if arr is None:
            return np.full(size, fill_value, dtype=dtype)
        arr = np.asarray(arr)
        if arr.size == size:
            return arr.astype(dtype, copy=False)
        if arr.size < size:
            pad = np.full(size - arr.size, fill_value, dtype=dtype)
            return np.concatenate([arr.astype(dtype, copy=False), pad])
        return arr[:size].astype(dtype, copy=False)

    def _ensure_source_metadata(self, st: IsotopeState) -> None:
        """Ensure per-source metadata arrays exist and match num_sources."""
        r = int(st.num_sources)
        st.ages = self._resize_metadata_array(st.ages, r, 0, int)
        st.low_q_streaks = self._resize_metadata_array(st.low_q_streaks, r, 0, int)
        st.support_scores = self._resize_metadata_array(st.support_scores, r, 0.0, float)
        st.tentative_sources = self._resize_metadata_array(
            st.tentative_sources,
            r,
            False,
            bool,
        )
        st.verification_fail_streaks = self._resize_metadata_array(
            st.verification_fail_streaks,
            r,
            0,
            int,
        )

    def _pseudo_source_fail_grace(self) -> int:
        """Return failed-verification count needed before quarantine or prune."""
        return max(
            0,
            int(self.config.pseudo_source_fail_grace_stations),
            int(self.config.source_prune_fail_grace_stations),
        )

    def _quarantined_source_mask(self, st: IsotopeState) -> NDArray[np.bool_]:
        """Return tentative sources currently quarantined by verification failures."""
        if st.num_sources <= 0:
            return np.zeros(0, dtype=bool)
        self._ensure_source_metadata(st)
        count = int(st.num_sources)
        tentative = np.asarray(st.tentative_sources[:count], dtype=bool)
        failed = np.asarray(st.verification_fail_streaks[:count], dtype=int)
        if failed.size != count:
            failed_padded = np.zeros(count, dtype=int)
            failed_padded[: min(failed.size, count)] = failed[:count]
            failed = failed_padded
        grace = self._pseudo_source_fail_grace()
        return tentative & (failed > 0) & (failed >= grace)

    def _needs_refit_prune_allowed(
        self,
        st: IsotopeState,
        *,
        next_delta_ll: NDArray[np.float64] | None = None,
        suppress_death: bool = False,
    ) -> bool:
        """
        Return whether expensive refit-after-remove prune masks can affect a state.

        This is a pure prefilter for implementation efficiency.  It only skips the
        refit-after-remove matrix when no pseudo-source prune or death/prune branch
        could consume the mask during the current structural update.
        """
        if st.num_sources <= 1:
            return False
        self._ensure_source_metadata(st)
        count = int(st.num_sources)
        if (
            bool(self.config.pseudo_source_verification_enable)
            and not suppress_death
            and count > 1
        ):
            tentative = np.asarray(st.tentative_sources[:count], dtype=bool)
            if np.any(tentative):
                fail_streaks = np.asarray(
                    st.verification_fail_streaks[:count],
                    dtype=int,
                )
                grace = self._pseudo_source_fail_grace()
                if np.any(tentative & ((fail_streaks + 1) >= grace)):
                    return True
        if suppress_death:
            return False
        projected_low_q = np.asarray(st.low_q_streaks[:count], dtype=int).copy()
        below_strength = np.asarray(st.strengths[:count], dtype=float) < float(
            self.config.min_strength,
        )
        projected_low_q[below_strength] += 1
        projected_low_q[~below_strength] = 0
        eligible = projected_low_q >= int(self.config.death_low_q_streak)
        if not np.any(eligible):
            return False
        q_min = float(self.config.min_strength)
        if q_min <= 0.0:
            q_min = float(self.config.birth_q_min)
        if np.any(eligible & (np.asarray(st.strengths[:count], dtype=float) < q_min)):
            return True
        if next_delta_ll is None or next_delta_ll.shape != (count,):
            support_scores = np.asarray(st.support_scores[:count], dtype=float)
        else:
            alpha = float(self.config.support_ema_alpha)
            support_scores = (1.0 - alpha) * np.asarray(
                st.support_scores[:count],
                dtype=float,
            ) + alpha * np.asarray(next_delta_ll, dtype=float)
        return bool(
            np.any(
                eligible
                & (support_scores < float(self.config.death_delta_ll_threshold)),
            ),
        )

    def _active_source_mask(
        self,
        st: IsotopeState,
        *,
        include_quarantined: bool = False,
    ) -> NDArray[np.bool_]:
        """Return source mask for reporting, planning, and residual proposals."""
        if st.num_sources <= 0:
            return np.zeros(0, dtype=bool)
        self._ensure_source_metadata(st)
        mask = np.ones(int(st.num_sources), dtype=bool)
        if (
            not include_quarantined
            and bool(self.config.pseudo_source_quarantine_excludes_runtime)
        ):
            mask &= ~self._quarantined_source_mask(st)
        return mask

    def _report_source_mask(self, st: IsotopeState) -> NDArray[np.bool_]:
        """Return the per-source mask used by report-only estimates."""
        if st.num_sources <= 0:
            return np.zeros(0, dtype=bool)
        self._ensure_source_metadata(st)
        mask = self._active_source_mask(st, include_quarantined=False)
        if bool(self.config.report_exclude_unverified_sources):
            count = int(st.num_sources)
            tentative = np.asarray(st.tentative_sources[:count], dtype=bool)
            failed = np.asarray(st.verification_fail_streaks[:count], dtype=int)
            if tentative.size != count:
                tentative_padded = np.zeros(count, dtype=bool)
                tentative_padded[: min(tentative.size, count)] = tentative[:count]
                tentative = tentative_padded
            if failed.size != count:
                failed_padded = np.zeros(count, dtype=int)
                failed_padded[: min(failed.size, count)] = failed[:count]
                failed = failed_padded
            mask &= ~(tentative | (failed > 0))
        return mask

    def state_without_quarantined_sources(self, st: IsotopeState) -> IsotopeState:
        """Return a copy of a state with runtime-excluded tentative sources removed."""
        out = st.copy()
        if out.num_sources <= 0:
            return out
        self._ensure_source_metadata(out)
        keep = self._active_source_mask(out, include_quarantined=False)
        self._apply_source_keep_mask(out, keep)
        return out

    def state_without_report_excluded_sources(self, st: IsotopeState) -> IsotopeState:
        """Return a copy of a state with report-excluded sources removed."""
        out = st.copy()
        if out.num_sources <= 0:
            return out
        self._ensure_source_metadata(out)
        keep = self._report_source_mask(out)
        self._apply_source_keep_mask(out, keep)
        return out

    def _apply_source_keep_mask(
        self,
        st: IsotopeState,
        keep: NDArray[np.bool_],
    ) -> int:
        """Apply a per-source keep mask and return the removed source count."""
        if st.num_sources <= 0:
            return 0
        self._ensure_source_metadata(st)
        keep_arr = np.asarray(keep, dtype=bool).ravel()[: int(st.num_sources)]
        if keep_arr.size != int(st.num_sources):
            raise ValueError("keep mask must match source count.")
        removed = int(np.count_nonzero(~keep_arr))
        if removed <= 0:
            return 0
        st.positions = st.positions[keep_arr]
        st.strengths = st.strengths[keep_arr]
        st.ages = st.ages[keep_arr]
        st.low_q_streaks = st.low_q_streaks[keep_arr]
        st.support_scores = st.support_scores[keep_arr]
        st.tentative_sources = st.tentative_sources[keep_arr]
        st.verification_fail_streaks = st.verification_fail_streaks[keep_arr]
        st.num_sources = int(st.positions.shape[0])
        return removed

    def apply_report_model_order_cluster_prune(
        self,
        report_positions: NDArray[np.float64],
        selected_mask: NDArray[np.bool_],
        *,
        radius_m: float,
    ) -> int:
        """
        Remove PF source slots that only support report clusters rejected by BIC.

        Distance classification is evaluated as one padded NumPy batch over all
        particles and source slots.  The final per-particle loop only applies the
        already-computed keep masks to ragged particle states.
        """
        if not self.continuous_particles:
            return 0
        pos_arr = np.asarray(report_positions, dtype=float).reshape(-1, 3)
        selected = np.asarray(selected_mask, dtype=bool).reshape(-1)
        if pos_arr.shape[0] == 0:
            return 0
        if selected.size != pos_arr.shape[0]:
            raise ValueError("selected_mask must match report_positions.")
        dropped = pos_arr[~selected]
        kept = pos_arr[selected]
        if dropped.shape[0] == 0 or kept.shape[0] == 0:
            return 0
        radius = float(radius_m)
        if radius <= 0.0:
            radius = max(float(self.config.cluster_eps_m), 1.0e-6)
        states = [particle.state for particle in self.continuous_particles]
        max_sources = max((max(0, int(st.num_sources)) for st in states), default=0)
        if max_sources <= 0:
            return 0
        particle_count = len(states)
        positions = np.zeros((particle_count, max_sources, 3), dtype=float)
        active = np.zeros((particle_count, max_sources), dtype=bool)
        for particle_idx, state in enumerate(states):
            count = max(0, int(state.num_sources))
            if count <= 0:
                continue
            count = min(count, max_sources)
            positions[particle_idx, :count] = np.asarray(
                state.positions[:count],
                dtype=float,
            )
            active[particle_idx, :count] = True
        if not np.any(active):
            return 0
        dropped_delta = positions[:, :, None, :] - dropped[None, None, :, :]
        kept_delta = positions[:, :, None, :] - kept[None, None, :, :]
        dropped_dist = np.min(np.linalg.norm(dropped_delta, axis=-1), axis=-1)
        kept_dist = np.min(np.linalg.norm(kept_delta, axis=-1), axis=-1)
        prune_slots = active & (dropped_dist <= radius) & (kept_dist > radius)
        if not np.any(prune_slots):
            return 0
        removed_total = 0
        for particle_idx in np.flatnonzero(np.any(prune_slots, axis=1)):
            state = states[int(particle_idx)]
            count = max(0, int(state.num_sources))
            if count <= 0:
                continue
            keep_mask = ~prune_slots[int(particle_idx), :count]
            removed_total += self._apply_source_keep_mask(state, keep_mask)
        return int(removed_total)

    def _lambda_components(
        self,
        st: IsotopeState,
        data: MeasurementData,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (lambda_m, lambda_total) for a state across measurements."""
        if data.z_k.size == 0:
            return np.zeros((0, st.num_sources), dtype=float), np.zeros(0, dtype=float)
        lambda_m = expected_counts_per_source(
            kernel=self.continuous_kernel,
            isotope=self.isotope,
            detector_positions=data.detector_positions,
            sources=st.positions,
            strengths=st.strengths,
            live_times=data.live_times,
            fe_indices=data.fe_indices,
            pb_indices=data.pb_indices,
            source_scale=self._measurement_source_scale_vector(
                data.fe_indices,
                data.pb_indices,
            ),
        )
        background_counts = float(st.background) * data.live_times
        lambda_total = background_counts + np.sum(lambda_m, axis=1)
        return lambda_m, lambda_total

    def _particle_indices_by_source_count(
        self,
        particle_indices: list[int] | None = None,
    ) -> tuple[dict[int, list[int]], list[int]]:
        """Group valid particle indices by active source count for batched kernels."""
        if particle_indices is None:
            candidate_indices = range(len(self.continuous_particles))
        else:
            candidate_indices = [int(idx) for idx in particle_indices]
        grouped: dict[int, list[int]] = {}
        fallback_indices: list[int] = []
        for idx in candidate_indices:
            if idx < 0 or idx >= len(self.continuous_particles):
                continue
            st = self.continuous_particles[idx].state
            self._ensure_source_metadata(st)
            source_count = max(0, int(st.num_sources))
            if source_count > 0 and (
                st.positions.ndim != 2
                or st.positions.shape[0] < source_count
                or st.strengths.size < source_count
            ):
                fallback_indices.append(idx)
                continue
            grouped.setdefault(source_count, []).append(idx)
        return grouped, fallback_indices

    def _lambda_components_for_particle_group(
        self,
        data: MeasurementData,
        particle_indices: list[int],
        source_count: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return batched per-source and total counts for equal-cardinality particles."""
        num_meas = int(data.z_k.size)
        particle_count = int(len(particle_indices))
        count = max(0, int(source_count))
        if num_meas == 0 or particle_count == 0:
            return (
                np.zeros((num_meas, particle_count, count), dtype=float),
                np.zeros((num_meas, particle_count), dtype=float),
            )
        backgrounds = np.asarray(
            [
                float(self.continuous_particles[idx].state.background)
                for idx in particle_indices
            ],
            dtype=float,
        )
        background_counts = data.live_times[:, None] * backgrounds[None, :]
        if count <= 0:
            return np.zeros((num_meas, particle_count, 0), dtype=float), background_counts
        sources = np.vstack(
            [
                np.asarray(
                    self.continuous_particles[idx].state.positions[:count],
                    dtype=float,
                )
                for idx in particle_indices
            ]
        )
        strengths = np.concatenate(
            [
                np.asarray(
                    self.continuous_particles[idx].state.strengths[:count],
                    dtype=float,
                )
                for idx in particle_indices
            ]
        )
        source_rows = sources.reshape(particle_count, count, 3)
        strength_rows = strengths.reshape(particle_count, count)
        if (
            particle_count > 1
            and np.array_equal(source_rows, source_rows[:1])
            and np.array_equal(strength_rows, strength_rows[:1])
        ):
            lambda_one = expected_counts_per_source(
                kernel=self.continuous_kernel,
                isotope=self.isotope,
                detector_positions=data.detector_positions,
                sources=source_rows[0],
                strengths=strength_rows[0],
                live_times=data.live_times,
                fe_indices=data.fe_indices,
                pb_indices=data.pb_indices,
                source_scale=self._measurement_source_scale_vector(
                    data.fe_indices,
                    data.pb_indices,
                ),
            )
            lambda_m = np.broadcast_to(
                np.asarray(lambda_one, dtype=float)[:, None, :],
                (num_meas, particle_count, count),
            ).copy()
            lambda_total = background_counts + np.sum(lambda_m, axis=2)
            return lambda_m, lambda_total
        lambda_flat = expected_counts_per_source(
            kernel=self.continuous_kernel,
            isotope=self.isotope,
            detector_positions=data.detector_positions,
            sources=sources,
            strengths=strengths,
            live_times=data.live_times,
            fe_indices=data.fe_indices,
            pb_indices=data.pb_indices,
            source_scale=self._measurement_source_scale_vector(
                data.fe_indices,
                data.pb_indices,
            ),
        )
        lambda_m = np.asarray(lambda_flat, dtype=float).reshape(
            num_meas,
            particle_count,
            count,
        )
        lambda_total = background_counts + np.sum(lambda_m, axis=2)
        return lambda_m, lambda_total

    def _unit_kernel_tensor_for_particle_group(
        self,
        data: MeasurementData,
        particle_indices: list[int],
        source_count: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Return unit-strength kernel tensor, background counts, and strengths.

        The tensor has shape K x P x S and is reused by batched refit and
        refit-after-remove tests so that structural updates do not recompute the
        same geometry response once per particle.
        """
        num_meas = int(data.z_k.size)
        particle_count = int(len(particle_indices))
        count = max(0, int(source_count))
        backgrounds = np.asarray(
            [
                float(self.continuous_particles[idx].state.background)
                for idx in particle_indices
            ],
            dtype=float,
        )
        background_counts = data.live_times[:, None] * backgrounds[None, :]
        if count > 0 and particle_count > 0:
            strengths = np.vstack(
                [
                    np.asarray(
                        self.continuous_particles[idx].state.strengths[:count],
                        dtype=float,
                    )
                    for idx in particle_indices
                ]
            )
        else:
            strengths = np.zeros((particle_count, count), dtype=float)
        if num_meas == 0 or particle_count == 0 or count <= 0:
            return (
                np.zeros((num_meas, particle_count, count), dtype=float),
                background_counts,
                strengths,
            )
        sources = np.vstack(
            [
                np.asarray(
                    self.continuous_particles[idx].state.positions[:count],
                    dtype=float,
                )
                for idx in particle_indices
            ]
        )
        unit_strengths = np.ones(sources.shape[0], dtype=float)
        k_flat = expected_counts_per_source(
            kernel=self.continuous_kernel,
            isotope=self.isotope,
            detector_positions=data.detector_positions,
            sources=sources,
            strengths=unit_strengths,
            live_times=data.live_times,
            fe_indices=data.fe_indices,
            pb_indices=data.pb_indices,
            source_scale=self._measurement_source_scale_vector(
                data.fe_indices,
                data.pb_indices,
            ),
        )
        k_tensor = np.asarray(k_flat, dtype=float).reshape(
            num_meas,
            particle_count,
            count,
        )
        return k_tensor, background_counts, strengths

    def _delta_log_likelihood_remove_group(
        self,
        data: MeasurementData,
        lambda_total: NDArray[np.float64],
        lambda_components: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return per-particle, per-source removal support for a batched group."""
        total = np.asarray(lambda_total, dtype=float)
        components = np.asarray(lambda_components, dtype=float)
        if components.ndim != 3 or components.shape[:2] != total.shape:
            return np.zeros((total.shape[1] if total.ndim == 2 else 0, 0), dtype=float)
        source_count = int(components.shape[2])
        if source_count <= 0:
            return np.zeros((total.shape[1], 0), dtype=float)
        model = str(self._count_likelihood_kwargs()["model"]).strip().lower()
        if model in {"poisson", ""}:
            total_safe = np.maximum(total, 1.0e-12)
            ratio = np.clip(
                components / total_safe[:, :, None],
                0.0,
                1.0 - 1.0e-12,
            )
            terms = (
                np.asarray(data.z_k, dtype=float)[:, None, None] * (-np.log1p(-ratio))
                - components
            )
            return np.sum(terms, axis=0)
        base_ll = self._count_log_likelihood_matrix_np(
            data.z_k,
            total,
            observation_count_variance=data.observation_variances,
        )
        deltas = np.zeros((total.shape[1], source_count), dtype=float)
        for source_idx in range(source_count):
            reduced = np.maximum(total - components[:, :, source_idx], 1.0e-12)
            reduced_ll = self._count_log_likelihood_matrix_np(
                data.z_k,
                reduced,
                observation_count_variance=data.observation_variances,
            )
            deltas[:, source_idx] = base_ll - reduced_ll
        return deltas

    def _compute_birth_proposal(
        self,
        data: MeasurementData | None,
        candidate_positions: NDArray[np.float64] | None,
    ) -> (
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            float,
            NDArray[np.float64],
            NDArray[np.float64],
        ]
        | None
    ):
        """
        Build residual-driven birth proposal and cached candidate responses.
        """
        if data is None or candidate_positions is None or candidate_positions.size == 0:
            return None
        if data.z_k.size == 0:
            return None
        if not self.continuous_particles:
            return None
        weights = np.asarray(self.continuous_weights, dtype=float)
        if weights.size != len(self.continuous_particles):
            return None
        topk = max(1, int(self.config.birth_topk_particles))
        order = np.argsort(weights)[::-1][:topk]
        sel_weights = weights[order]
        if np.sum(sel_weights) <= 0.0:
            sel_weights = np.ones_like(sel_weights, dtype=float)
        sel_weights = sel_weights / np.sum(sel_weights)
        layers = self._compute_birth_residual_layers(
            data=data,
            particle_indices=order,
            particle_weights=sel_weights,
        )
        self.last_birth_residual_layer_count = len(layers)
        if not layers:
            return None
        raw_layer = next((layer for layer in layers if layer.name == "raw"), None)
        raw_gate_passed_cache: bool | None = None
        raw_refit_ok: bool | None = None
        if raw_layer is not None:
            raw_sum = float(np.sum(np.maximum(raw_layer.residual, 0.0)))
            raw_gate_passed = False
            if raw_sum > 0.0:
                raw_gate_passed = self._birth_residual_gate_allows(
                    raw_layer.residual,
                    data.observation_variances,
                    data.detector_positions,
                    data.fe_indices,
                    data.pb_indices,
                )
            raw_gate_passed_cache = bool(raw_gate_passed)
            if raw_gate_passed:
                raw_refit_ok = self._birth_residual_survives_strength_refit(
                    data=data,
                    particle_indices=order,
                    particle_weights=sel_weights,
                    residual_sum_before=raw_sum,
                )
        selected_layer: BirthResidualLayer | None = None
        selected_sum = 0.0
        for layer in sorted(
            layers,
            key=lambda item: float(np.sum(np.maximum(item.residual, 0.0))),
            reverse=True,
        ):
            residual_sum = float(np.sum(np.maximum(layer.residual, 0.0)))
            if residual_sum <= 0.0:
                continue
            if layer.name == "raw" and raw_gate_passed_cache is not None:
                gate_passed = raw_gate_passed_cache
            else:
                gate_passed = self._birth_residual_gate_allows(
                    layer.residual,
                    data.observation_variances,
                    data.detector_positions,
                    data.fe_indices,
                    data.pb_indices,
                )
            if not gate_passed:
                continue
            refit_ok = True
            if layer.name == "raw":
                if raw_refit_ok is None:
                    refit_ok = self._birth_residual_survives_strength_refit(
                        data=data,
                        particle_indices=order,
                        particle_weights=sel_weights,
                        residual_sum_before=residual_sum,
                    )
                else:
                    refit_ok = raw_refit_ok
            else:
                refit_ok = True
            if not refit_ok:
                continue
            selected_layer = layer
            selected_sum = residual_sum
            break
        if selected_layer is None:
            return None
        residual_mix = np.maximum(selected_layer.residual, 0.0)
        residual_sum = float(selected_sum)
        self.last_birth_residual_layer = str(selected_layer.name)

        base_candidates = self._exclude_birth_candidates_near_detectors(
            candidate_positions.copy(),
            data,
        )
        if base_candidates.size == 0:
            return None
        existing_response_counts = self._birth_existing_unit_response_counts(
            data,
            particle_indices=order,
        )

        unit_strengths = np.ones(base_candidates.shape[0], dtype=float)
        base_candidate_counts = expected_counts_per_source(
            kernel=self.continuous_kernel,
            isotope=self.isotope,
            detector_positions=data.detector_positions,
            sources=base_candidates,
            strengths=unit_strengths,
            live_times=data.live_times,
            fe_indices=data.fe_indices,
            pb_indices=data.pb_indices,
            source_scale=self._measurement_source_scale_vector(
                data.fe_indices,
                data.pb_indices,
            ),
        )
        base_support_mask = self._birth_candidate_support_mask(
            candidate_counts=base_candidate_counts,
            residual_mix=residual_mix,
            observation_variances=data.observation_variances,
            detector_positions=data.detector_positions,
            fe_indices=data.fe_indices,
            pb_indices=data.pb_indices,
        )
        if not np.any(base_support_mask):
            return None
        base_scores, _ = self._birth_residual_candidate_scores(
            candidate_counts=base_candidate_counts,
            residual_mix=residual_mix,
            observation_variances=data.observation_variances,
        )
        base_supported_indices = np.flatnonzero(base_support_mask)
        num_jitter = max(0, int(self.config.birth_num_local_jitter))
        candidate_counts = base_candidate_counts[:, base_supported_indices]
        candidates = base_candidates[base_supported_indices]
        if num_jitter > 0 and base_supported_indices.size > 0:
            jitter_limit = self.config.birth_jitter_topk_candidates
            if jitter_limit is None:
                jitter_indices = base_supported_indices
            else:
                top_count = min(max(1, int(jitter_limit)), base_supported_indices.size)
                ranked = base_supported_indices[
                    np.argsort(base_scores[base_supported_indices])[::-1][:top_count]
                ]
                jitter_indices = ranked
            jitter_sigma = float(self.config.birth_candidate_jitter_sigma)
            jitter = np.random.normal(
                loc=0.0,
                scale=jitter_sigma,
                size=(jitter_indices.size, num_jitter, 3),
            )
            jittered = base_candidates[jitter_indices, None, :] + jitter
            jittered = self._project_positions_to_source_prior(
                jittered.reshape(-1, 3)
            )
            jittered = self._exclude_birth_candidates_near_detectors(jittered, data)
            if jittered.size:
                jitter_counts = expected_counts_per_source(
                    kernel=self.continuous_kernel,
                    isotope=self.isotope,
                    detector_positions=data.detector_positions,
                    sources=jittered,
                    strengths=np.ones(jittered.shape[0], dtype=float),
                    live_times=data.live_times,
                    fe_indices=data.fe_indices,
                    pb_indices=data.pb_indices,
                    source_scale=self._measurement_source_scale_vector(
                        data.fe_indices,
                        data.pb_indices,
                    ),
                )
                candidate_counts = np.hstack([candidate_counts, jitter_counts])
                candidates = np.vstack([candidates, jittered])
                final_support_mask = self._birth_candidate_support_mask(
                    candidate_counts=candidate_counts,
                    residual_mix=residual_mix,
                    observation_variances=data.observation_variances,
                    detector_positions=data.detector_positions,
                    fe_indices=data.fe_indices,
                    pb_indices=data.pb_indices,
                )
                if not np.any(final_support_mask):
                    return None
                candidate_counts = candidate_counts[:, final_support_mask]
                candidates = candidates[final_support_mask]
        response_keep = self._birth_existing_response_correlation_mask(
            candidate_counts=candidate_counts,
            existing_response_counts=existing_response_counts,
            observation_variances=data.observation_variances,
        )
        response_keep &= self._birth_response_condition_mask(
            candidate_counts=candidate_counts,
            existing_response_counts=existing_response_counts,
            observation_variances=data.observation_variances,
        )
        if not np.any(response_keep):
            return None
        candidate_counts = candidate_counts[:, response_keep]
        candidates = candidates[response_keep]
        scores, q_hat = self._birth_residual_candidate_scores(
            candidate_counts=candidate_counts,
            residual_mix=residual_mix,
            observation_variances=data.observation_variances,
        )
        finite = np.isfinite(scores) & np.isfinite(q_hat) & (scores > 0.0) & (q_hat > 0.0)
        if not np.any(finite):
            return None
        candidate_counts = candidate_counts[:, finite]
        candidates = candidates[finite]
        scores = scores[finite]
        q_hat = q_hat[finite]
        if np.max(scores) <= 0.0:
            return None
        order = np.argsort(scores)[::-1]
        scores = scores[order]
        q_hat = q_hat[order]
        kernel_sums = max(residual_sum, 1.0e-12) / np.maximum(q_hat, 1.0e-12)
        candidates = candidates[order]
        candidate_counts = candidate_counts[:, order]
        scores = np.maximum(scores, float(self.config.birth_min_score))
        temp = max(float(self.config.birth_softmax_temp), 1e-6)
        scaled = scores / temp
        scaled = scaled - np.max(scaled)
        probs = np.exp(scaled)
        probs = probs / max(float(np.sum(probs)), 1e-12)
        if selected_layer.name != "raw":
            self.last_birth_residual_refit_fraction = 1.0
            self.last_birth_residual_refit_gate_passed = True
        return probs, kernel_sums, residual_sum, candidates, candidate_counts

    def _compute_birth_residual_layers(
        self,
        *,
        data: MeasurementData,
        particle_indices: NDArray[np.int64],
        particle_weights: NDArray[np.float64],
    ) -> list[BirthResidualLayer]:
        """
        Return residual layers using batched per-cardinality expected counts.

        This is mathematically equivalent to the scalar residual-layer oracle,
        but it evaluates all selected particles with the same active source
        count in one expected-count kernel call.
        """
        if data.z_k.size == 0:
            return []
        indices = np.asarray(particle_indices, dtype=int).ravel()
        weights = np.asarray(particle_weights, dtype=float).ravel()
        if weights.size != indices.size:
            weights = np.ones(indices.size, dtype=float)
        valid = (indices >= 0) & (indices < len(self.continuous_particles))
        indices = indices[valid]
        weights = weights[valid]
        if indices.size == 0:
            return []
        if float(np.sum(weights)) <= 0.0:
            weights = np.ones_like(weights, dtype=float)
        weights = weights / max(float(np.sum(weights)), 1.0e-12)

        records: list[
            tuple[
                int,
                float,
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.int64],
                NDArray[np.float64],
            ]
        ] = []
        groups: dict[int, list[int]] = {}
        for local_idx, particle_idx in enumerate(indices):
            st = self.continuous_particles[int(particle_idx)].state
            self._ensure_source_metadata(st)
            active_mask = self._active_source_mask(st, include_quarantined=True)
            if st.num_sources > 0 and np.any(active_mask):
                active_positions = np.asarray(
                    st.positions[: st.num_sources][active_mask],
                    dtype=float,
                )
                active_strengths = np.asarray(
                    st.strengths[: st.num_sources][active_mask],
                    dtype=float,
                )
                active_indices = np.flatnonzero(active_mask).astype(np.int64)
            else:
                active_positions = np.zeros((0, 3), dtype=float)
                active_strengths = np.zeros(0, dtype=float)
                active_indices = np.zeros(0, dtype=np.int64)
            background_counts = float(st.background) * data.live_times
            record_idx = len(records)
            records.append(
                (
                    int(particle_idx),
                    float(weights[local_idx]),
                    active_positions,
                    active_strengths,
                    active_indices,
                    np.asarray(background_counts, dtype=float),
                )
            )
            groups.setdefault(int(active_positions.shape[0]), []).append(record_idx)

        lambda_by_record: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
        for active_count, record_indices in groups.items():
            if active_count <= 0:
                for record_idx in record_indices:
                    background_counts = records[record_idx][5]
                    lambda_by_record[record_idx] = (
                        np.zeros((data.z_k.size, 0), dtype=float),
                        background_counts,
                    )
                continue
            stacked_positions = np.vstack(
                [records[record_idx][2] for record_idx in record_indices]
            )
            stacked_strengths = np.concatenate(
                [records[record_idx][3] for record_idx in record_indices]
            )
            lambda_flat = expected_counts_per_source(
                kernel=self.continuous_kernel,
                isotope=self.isotope,
                detector_positions=data.detector_positions,
                sources=stacked_positions,
                strengths=stacked_strengths,
                live_times=data.live_times,
                fe_indices=data.fe_indices,
                pb_indices=data.pb_indices,
                source_scale=self._measurement_source_scale_vector(
                    data.fe_indices,
                    data.pb_indices,
                ),
            )
            lambda_group = np.asarray(lambda_flat, dtype=float).reshape(
                int(data.z_k.size),
                int(len(record_indices)),
                int(active_count),
            )
            for local_group_idx, record_idx in enumerate(record_indices):
                background_counts = records[record_idx][5]
                lambda_m = lambda_group[:, local_group_idx, :]
                lambda_total = background_counts + np.sum(lambda_m, axis=1)
                lambda_by_record[record_idx] = (lambda_m, lambda_total)

        layer_parts: dict[str, list[NDArray[np.float64]]] = {"raw": []}
        weighted_lambda_total = np.zeros(data.z_k.size, dtype=float)
        cluster_records: list[
            tuple[NDArray[np.float64], NDArray[np.float64], float]
        ] = []
        max_layers = max(1, int(self.config.residual_decomposition_max_layers))
        min_fraction = max(float(self.config.peak_suppression_min_source_fraction), 0.0)
        suppress_factor = float(
            np.clip(float(self.config.peak_suppression_factor), 0.0, 1.0)
        )
        allow_suppression = (
            bool(self.config.residual_decomposition_enable)
            and bool(self.config.peak_suppression_enable)
            and max_layers > 1
        )
        for record_idx, record in enumerate(records):
            particle_idx, weight, _positions, _strengths, active_indices, _bg = record
            lambda_m, lambda_total = lambda_by_record[record_idx]
            weighted_lambda_total += weight * np.asarray(lambda_total, dtype=float)
            raw = self._clip_birth_residual(np.maximum(data.z_k - lambda_total, 0.0))
            layer_parts["raw"].append(raw * weight)
            if not allow_suppression or lambda_m.shape[1] == 0:
                continue
            st = self.continuous_particles[int(particle_idx)].state
            source_totals = np.sum(np.maximum(lambda_m, 0.0), axis=0)
            total_source = max(float(np.sum(source_totals)), 1.0e-12)
            strong_order = np.argsort(source_totals)[::-1]
            added = 0
            for source_idx in strong_order:
                if added >= max_layers - 1:
                    break
                if float(source_totals[int(source_idx)]) < min_fraction * total_source:
                    continue
                suppressed_total = (
                    lambda_total - suppress_factor * lambda_m[:, int(source_idx)]
                )
                residual = self._clip_birth_residual(
                    np.maximum(data.z_k - suppressed_total, 0.0)
                )
                layer_name = f"strong_suppressed_{added}"
                layer_parts.setdefault(layer_name, []).append(residual * weight)
                state_source_idx = int(active_indices[int(source_idx)])
                cluster_records.append(
                    (
                        np.asarray(st.positions[state_source_idx], dtype=float).copy(),
                        weight * np.asarray(lambda_m[:, int(source_idx)], dtype=float),
                        weight * float(source_totals[int(source_idx)]),
                    )
                )
                added += 1
        return self._finalize_birth_residual_layers(
            data=data,
            layer_parts=layer_parts,
            weighted_lambda_total=weighted_lambda_total,
            cluster_records=cluster_records,
        )

    def _compute_birth_residual_layers_scalar(
        self,
        *,
        data: MeasurementData,
        particle_indices: NDArray[np.int64],
        particle_weights: NDArray[np.float64],
    ) -> list[BirthResidualLayer]:
        """
        Return raw, source-suppressed, and cluster-suppressed residual layers.

        The raw layer is the usual positive residual after all existing sources.
        Peak-suppressed layers add back a strong source or source cluster before
        candidate ranking.  These layers are used only to propose new source
        hypotheses; the accepted move is still judged by the original full
        observation likelihood.
        """
        if data.z_k.size == 0:
            return []
        weights = np.asarray(particle_weights, dtype=float).ravel()
        particle_indices = np.asarray(particle_indices, dtype=int).ravel()
        if weights.size != particle_indices.size:
            weights = np.ones(particle_indices.size, dtype=float)
        if weights.size == 0:
            return []
        if float(np.sum(weights)) <= 0.0:
            weights = np.ones_like(weights, dtype=float)
        weights = weights / max(float(np.sum(weights)), 1.0e-12)
        layer_parts: dict[str, list[NDArray[np.float64]]] = {"raw": []}
        weighted_lambda_total = np.zeros(data.z_k.size, dtype=float)
        cluster_records: list[
            tuple[NDArray[np.float64], NDArray[np.float64], float]
        ] = []
        max_layers = max(1, int(self.config.residual_decomposition_max_layers))
        min_fraction = max(float(self.config.peak_suppression_min_source_fraction), 0.0)
        suppress_factor = float(np.clip(float(self.config.peak_suppression_factor), 0.0, 1.0))
        allow_suppression = (
            bool(self.config.residual_decomposition_enable)
            and bool(self.config.peak_suppression_enable)
            and max_layers > 1
        )
        for local_idx, particle_idx in enumerate(particle_indices):
            if particle_idx < 0 or particle_idx >= len(self.continuous_particles):
                continue
            weight = float(weights[local_idx])
            st = self.continuous_particles[int(particle_idx)].state
            background_counts = float(st.background) * data.live_times
            active_mask = self._active_source_mask(st, include_quarantined=False)
            if st.num_sources > 0 and np.any(active_mask):
                lambda_m = expected_counts_per_source(
                    kernel=self.continuous_kernel,
                    isotope=self.isotope,
                    detector_positions=data.detector_positions,
                    sources=st.positions[: st.num_sources][active_mask],
                    strengths=st.strengths[: st.num_sources][active_mask],
                    live_times=data.live_times,
                    fe_indices=data.fe_indices,
                    pb_indices=data.pb_indices,
                    source_scale=self._measurement_source_scale_vector(
                        data.fe_indices,
                        data.pb_indices,
                    ),
                )
                lambda_total = background_counts + np.sum(lambda_m, axis=1)
            else:
                lambda_m = np.zeros((data.z_k.size, 0), dtype=float)
                lambda_total = background_counts
            weighted_lambda_total += weight * np.asarray(lambda_total, dtype=float)
            raw = self._clip_birth_residual(np.maximum(data.z_k - lambda_total, 0.0))
            layer_parts["raw"].append(raw * weight)
            if not allow_suppression or lambda_m.shape[1] == 0:
                continue
            source_totals = np.sum(np.maximum(lambda_m, 0.0), axis=0)
            total_source = max(float(np.sum(source_totals)), 1.0e-12)
            strong_order = np.argsort(source_totals)[::-1]
            active_indices = np.flatnonzero(active_mask)
            added = 0
            for source_idx in strong_order:
                if added >= max_layers - 1:
                    break
                if float(source_totals[int(source_idx)]) < min_fraction * total_source:
                    continue
                suppressed_total = lambda_total - suppress_factor * lambda_m[:, int(source_idx)]
                residual = self._clip_birth_residual(
                    np.maximum(data.z_k - suppressed_total, 0.0)
                )
                layer_name = f"strong_suppressed_{added}"
                layer_parts.setdefault(layer_name, []).append(residual * weight)
                state_source_idx = int(active_indices[int(source_idx)])
                cluster_records.append(
                    (
                        np.asarray(st.positions[state_source_idx], dtype=float).copy(),
                        weight * np.asarray(lambda_m[:, int(source_idx)], dtype=float),
                        weight * float(source_totals[int(source_idx)]),
                    )
                )
                added += 1
        return self._finalize_birth_residual_layers(
            data=data,
            layer_parts=layer_parts,
            weighted_lambda_total=weighted_lambda_total,
            cluster_records=cluster_records,
        )

    def _finalize_birth_residual_layers(
        self,
        *,
        data: MeasurementData,
        layer_parts: dict[str, list[NDArray[np.float64]]],
        weighted_lambda_total: NDArray[np.float64],
        cluster_records: list[
            tuple[NDArray[np.float64], NDArray[np.float64], float]
        ],
    ) -> list[BirthResidualLayer]:
        """Finalize weighted residual layer parts into ordered birth layers."""
        max_layers = max(1, int(self.config.residual_decomposition_max_layers))
        suppress_factor = float(
            np.clip(float(self.config.peak_suppression_factor), 0.0, 1.0)
        )
        allow_suppression = (
            bool(self.config.residual_decomposition_enable)
            and bool(self.config.peak_suppression_enable)
            and max_layers > 1
        )
        if allow_suppression and cluster_records and max_layers > 2:
            cluster_radius = max(
                float(self.config.cluster_eps_m),
                0.5 * float(self.config.birth_min_sep_m),
                1.0e-6,
            )
            cluster_components: list[NDArray[np.float64]] = []
            cluster_scores: list[float] = []
            used = np.zeros(len(cluster_records), dtype=bool)
            positions = np.vstack([record[0] for record in cluster_records])
            record_scores = np.asarray(
                [record[2] for record in cluster_records],
                dtype=float,
            )
            for seed_idx in np.argsort(record_scores)[::-1]:
                if used[int(seed_idx)]:
                    continue
                dists = np.linalg.norm(
                    positions - positions[int(seed_idx)][None, :],
                    axis=1,
                )
                members = np.flatnonzero((dists <= cluster_radius) & (~used))
                if members.size == 0:
                    continue
                used[members] = True
                component = np.sum(
                    np.vstack([cluster_records[int(idx)][1] for idx in members]),
                    axis=0,
                )
                score = float(np.sum([cluster_records[int(idx)][2] for idx in members]))
                if score <= 0.0 or float(np.sum(component)) <= 0.0:
                    continue
                cluster_components.append(component)
                cluster_scores.append(score)
            for layer_idx, cluster_idx in enumerate(np.argsort(cluster_scores)[::-1]):
                if layer_idx >= max_layers - 1:
                    break
                suppressed_total = (
                    weighted_lambda_total
                    - suppress_factor * cluster_components[int(cluster_idx)]
                )
                residual = self._clip_birth_residual(
                    np.maximum(data.z_k - suppressed_total, 0.0)
                )
                layer_name = f"leave_one_cluster_out_{layer_idx}"
                layer_parts.setdefault(layer_name, []).append(residual)
        layers: list[BirthResidualLayer] = []
        for name, parts in layer_parts.items():
            if not parts:
                continue
            stack = np.vstack(parts)
            residual = (
                np.sum(stack, axis=0)
                if bool(self.config.birth_use_weighted_topk)
                else np.mean(stack, axis=0)
            )
            if float(np.sum(np.maximum(residual, 0.0))) > 0.0:
                layers.append(BirthResidualLayer(name=name, residual=residual))
        raw_layers = [layer for layer in layers if layer.name == "raw"]
        aux_layers = [layer for layer in layers if layer.name != "raw"]
        aux_layers.sort(
            key=lambda layer: float(np.sum(np.maximum(layer.residual, 0.0))),
            reverse=True,
        )
        return raw_layers + aux_layers[: max(0, max_layers - len(raw_layers))]

    def _clip_birth_residual(
        self,
        residual: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Clip residual outliers before residual-driven structural proposals."""
        clipped = np.asarray(residual, dtype=float).copy()
        clip_q = float(self.config.birth_residual_clip_quantile)
        if 0.0 < clip_q < 1.0 and clipped.size:
            clip_val = float(np.quantile(clipped, clip_q))
            clipped = np.minimum(clipped, clip_val)
        return clipped

    def _birth_residual_candidate_scores(
        self,
        *,
        candidate_counts: NDArray[np.float64],
        residual_mix: NDArray[np.float64],
        observation_variances: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return shield-coded residual matching scores and fitted strengths."""
        counts = np.maximum(np.asarray(candidate_counts, dtype=float), 0.0)
        if counts.ndim != 2 or counts.size == 0:
            return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
        residual = self._measurement_vector(
            residual_mix,
            counts.shape[0],
            "residual_mix",
            min_value=0.0,
            allow_scalar=False,
        )
        if not bool(self.config.birth_use_shield_coded_residual):
            sums = np.maximum(np.sum(counts, axis=0), 1.0e-12)
            residual_sum = max(float(np.sum(residual)), 0.0)
            scores = np.asarray(residual @ counts, dtype=float)
            q_hat = residual_sum / sums
            scores *= self._birth_count_distance_prior(
                candidate_counts=counts,
                residual_mix=residual,
                q_hat=q_hat,
            )
            return scores, q_hat
        variances = self._measurement_vector(
            observation_variances,
            counts.shape[0],
            "observation_variances",
            min_value=1.0e-12,
        )
        weights = 1.0 / variances
        numerator = np.sum(weights[:, None] * residual[:, None] * counts, axis=0)
        denominator = np.sum(weights[:, None] * counts * counts, axis=0)
        q_hat = np.maximum(numerator / np.maximum(denominator, 1.0e-12), 0.0)
        scores = numerator * q_hat
        scores *= self._birth_count_distance_prior(
            candidate_counts=counts,
            residual_mix=residual,
            q_hat=q_hat,
        )
        return np.asarray(scores, dtype=float), np.asarray(q_hat, dtype=float)

    def _birth_count_distance_prior(
        self,
        *,
        candidate_counts: NDArray[np.float64],
        residual_mix: NDArray[np.float64],
        q_hat: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return a soft proposal prior favoring high unit-response candidates."""
        response_weight = max(
            0.0,
            float(getattr(self.config, "birth_count_distance_prior_weight", 0.0)),
        )
        strength_weight = max(
            0.0,
            float(getattr(self.config, "birth_count_distance_strength_weight", 0.0)),
        )
        counts = np.maximum(np.asarray(candidate_counts, dtype=float), 0.0)
        if counts.ndim != 2 or counts.size == 0:
            return np.zeros(0, dtype=float)
        num_candidates = counts.shape[1]
        if response_weight <= 0.0 and strength_weight <= 0.0:
            return np.ones(num_candidates, dtype=float)
        residual = self._measurement_vector(
            residual_mix,
            counts.shape[0],
            "residual_mix",
            min_value=0.0,
            allow_scalar=False,
        )
        residual_sum = float(np.sum(residual))
        if residual_sum <= 0.0:
            return np.ones(num_candidates, dtype=float)

        eps = 1.0e-12
        prior = np.ones(num_candidates, dtype=float)
        residual_weights = residual / max(residual_sum, eps)
        unit_response = np.sum(residual_weights[:, None] * counts, axis=0)
        finite_response = np.isfinite(unit_response) & (unit_response > eps)
        log_clip = max(
            0.0,
            float(getattr(self.config, "birth_count_distance_log_clip", 3.0)),
        )

        if response_weight > 0.0 and np.any(finite_response):
            response_ref = float(np.median(unit_response[finite_response]))
            if response_ref > eps:
                log_response = np.log(
                    np.maximum(unit_response, eps) / response_ref
                )
                prior *= np.exp(
                    response_weight * np.clip(log_response, -log_clip, log_clip)
                )

        q = np.maximum(np.asarray(q_hat, dtype=float).ravel(), 0.0)
        if q.size != num_candidates:
            raise ValueError("q_hat must have one value per candidate.")
        finite_q = np.isfinite(q) & (q > eps)
        if strength_weight > 0.0 and np.any(finite_q):
            q_ref = float(np.median(q[finite_q]))
            sigma = max(
                float(
                    getattr(
                        self.config,
                        "birth_count_distance_strength_sigma",
                        2.0,
                    )
                ),
                eps,
            )
            if q_ref > eps:
                log_q = np.log(np.maximum(q, eps) / q_ref)
                high_strength = np.maximum(log_q, 0.0)
                high_strength = np.clip(high_strength, 0.0, log_clip)
                prior *= np.exp(
                    -0.5 * strength_weight * (high_strength / sigma) ** 2
                )

        finite_prior = np.isfinite(prior) & (prior > 0.0)
        if np.any(finite_prior):
            norm = float(np.median(prior[finite_prior]))
            if norm > eps:
                prior /= norm
        return np.clip(np.where(np.isfinite(prior), prior, 0.0), eps, 1.0e6)

    def _birth_existing_unit_response_counts(
        self,
        data: MeasurementData,
        *,
        particle_indices: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Return batched unit response columns for existing top-particle sources."""
        return self._birth_existing_unit_response_counts_batched(
            data,
            particle_indices=particle_indices,
        )

    def _birth_existing_unit_response_counts_scalar(
        self,
        data: MeasurementData,
        *,
        particle_indices: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Scalar oracle for existing top-particle unit response columns."""
        columns: list[NDArray[np.float64]] = []
        for particle_idx in np.asarray(particle_indices, dtype=int).ravel():
            if particle_idx < 0 or particle_idx >= len(self.continuous_particles):
                continue
            st = self.continuous_particles[int(particle_idx)].state
            if st.num_sources <= 0:
                continue
            active_mask = self._active_source_mask(st, include_quarantined=False)
            if not np.any(active_mask):
                continue
            positions = st.positions[: st.num_sources][active_mask]
            counts = expected_counts_per_source(
                kernel=self.continuous_kernel,
                isotope=self.isotope,
                detector_positions=data.detector_positions,
                sources=positions,
                strengths=np.ones(positions.shape[0], dtype=float),
                live_times=data.live_times,
                fe_indices=data.fe_indices,
                pb_indices=data.pb_indices,
                source_scale=self._measurement_source_scale_vector(
                    data.fe_indices,
                    data.pb_indices,
                ),
            )
            for col_idx in range(counts.shape[1]):
                columns.append(np.asarray(counts[:, col_idx], dtype=float))
        if not columns:
            return np.zeros((data.z_k.size, 0), dtype=float)
        return np.column_stack(columns)

    def _birth_existing_unit_response_counts_batched(
        self,
        data: MeasurementData,
        *,
        particle_indices: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Return existing-source unit responses using grouped batched kernels."""
        records: list[tuple[int, NDArray[np.float64]]] = []
        groups: dict[int, list[int]] = {}
        for particle_idx in np.asarray(particle_indices, dtype=int).ravel():
            if particle_idx < 0 or particle_idx >= len(self.continuous_particles):
                continue
            st = self.continuous_particles[int(particle_idx)].state
            if st.num_sources <= 0:
                continue
            active_mask = self._active_source_mask(st, include_quarantined=False)
            if not np.any(active_mask):
                continue
            positions = np.asarray(
                st.positions[: st.num_sources][active_mask],
                dtype=float,
            )
            active_count = int(positions.shape[0])
            if active_count <= 0:
                continue
            record_idx = len(records)
            records.append((active_count, positions))
            groups.setdefault(active_count, []).append(record_idx)
        if not records:
            return np.zeros((data.z_k.size, 0), dtype=float)

        counts_by_record: dict[int, NDArray[np.float64]] = {}
        for active_count, record_indices in groups.items():
            stacked_positions = np.vstack(
                [records[record_idx][1] for record_idx in record_indices]
            )
            flat_counts = expected_counts_per_source(
                kernel=self.continuous_kernel,
                isotope=self.isotope,
                detector_positions=data.detector_positions,
                sources=stacked_positions,
                strengths=np.ones(stacked_positions.shape[0], dtype=float),
                live_times=data.live_times,
                fe_indices=data.fe_indices,
                pb_indices=data.pb_indices,
                source_scale=self._measurement_source_scale_vector(
                    data.fe_indices,
                    data.pb_indices,
                ),
            )
            group_counts = np.asarray(flat_counts, dtype=float).reshape(
                int(data.z_k.size),
                int(len(record_indices)),
                int(active_count),
            )
            for local_idx, record_idx in enumerate(record_indices):
                counts_by_record[record_idx] = group_counts[:, local_idx, :]

        columns: list[NDArray[np.float64]] = []
        for record_idx in range(len(records)):
            counts = counts_by_record.get(record_idx)
            if counts is None:
                continue
            for col_idx in range(counts.shape[1]):
                columns.append(np.asarray(counts[:, col_idx], dtype=float))
        if not columns:
            return np.zeros((data.z_k.size, 0), dtype=float)
        return np.column_stack(columns)

    def _birth_existing_response_correlation_mask(
        self,
        *,
        candidate_counts: NDArray[np.float64],
        existing_response_counts: NDArray[np.float64],
        observation_variances: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """Reject birth candidates collinear with already represented responses."""
        candidates = np.maximum(np.asarray(candidate_counts, dtype=float), 0.0)
        if candidates.ndim != 2 or candidates.size == 0:
            return np.zeros(0, dtype=bool)
        max_corr = float(self.config.birth_existing_response_corr_max)
        if max_corr >= 1.0:
            return np.ones(candidates.shape[1], dtype=bool)
        existing = np.maximum(np.asarray(existing_response_counts, dtype=float), 0.0)
        if existing.ndim != 2 or existing.size == 0 or existing.shape[1] == 0:
            return np.ones(candidates.shape[1], dtype=bool)
        variances = self._measurement_vector(
            observation_variances,
            candidates.shape[0],
            "observation_variances",
            min_value=1.0e-12,
        )
        if existing.shape[0] != candidates.shape[0]:
            raise ValueError(
                "existing_response_counts must have one row per measurement."
            )
        scale = 1.0 / np.sqrt(variances)
        cand_w = candidates * scale[:, None]
        exist_w = existing * scale[:, None]
        cand_norm = np.maximum(np.linalg.norm(cand_w, axis=0), 1.0e-12)
        exist_norm = np.maximum(np.linalg.norm(exist_w, axis=0), 1.0e-12)
        corr = np.abs(exist_w.T @ cand_w) / (exist_norm[:, None] * cand_norm[None, :])
        worst = np.max(corr, axis=0) if corr.size else np.zeros(candidates.shape[1])
        return np.asarray(worst < max(float(max_corr), 0.0), dtype=bool)

    def _weighted_response_condition_number(
        self,
        response_counts: NDArray[np.float64],
        observation_variances: NDArray[np.float64],
    ) -> float:
        """Return the weighted condition number of response columns."""
        columns = np.maximum(np.asarray(response_counts, dtype=float), 0.0)
        if columns.ndim != 2 or columns.shape[1] <= 1:
            return 1.0
        variances = self._measurement_vector(
            observation_variances,
            columns.shape[0],
            "observation_variances",
            min_value=1.0e-12,
        )
        weighted = columns * (1.0 / np.sqrt(variances))[:, None]
        norms = np.linalg.norm(weighted, axis=0)
        active = norms > 1.0e-12
        if np.count_nonzero(active) <= 1:
            return 1.0
        normalized = weighted[:, active] / norms[active][None, :]
        singular_values = np.linalg.svd(normalized, compute_uv=False)
        singular_values = singular_values[singular_values > 1.0e-12]
        if singular_values.size <= 1:
            return 1.0
        return float(np.max(singular_values) / np.min(singular_values))

    def _birth_response_condition_mask(
        self,
        *,
        candidate_counts: NDArray[np.float64],
        existing_response_counts: NDArray[np.float64],
        observation_variances: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """
        Reject candidates explainable by existing same-isotope response columns.

        This is a model-identifiability test, not a scenario-specific rule: if a
        candidate column is almost explainable by existing source columns over
        the recent distance, shield, and obstacle measurement sequence, adding
        it cannot be reliably distinguished from strength refitting.

        The test is incremental. Existing source columns may already be
        ill-conditioned during early same-isotope separation, and that should
        not by itself block a new source. A candidate is rejected only when its
        weighted response has too little component outside the span of the
        existing active responses.
        """
        candidates = np.maximum(np.asarray(candidate_counts, dtype=float), 0.0)
        if candidates.ndim != 2 or candidates.size == 0:
            return np.zeros(0, dtype=bool)
        max_condition = float(self.config.birth_response_condition_max)
        if max_condition <= 0.0 or not np.isfinite(max_condition):
            return np.ones(candidates.shape[1], dtype=bool)
        existing = np.maximum(np.asarray(existing_response_counts, dtype=float), 0.0)
        if existing.ndim != 2 or existing.size == 0 or existing.shape[1] == 0:
            return np.ones(candidates.shape[1], dtype=bool)
        if existing.shape[0] != candidates.shape[0]:
            raise ValueError(
                "existing_response_counts must have one row per measurement."
            )
        variances = self._measurement_vector(
            observation_variances,
            candidates.shape[0],
            "observation_variances",
            min_value=1.0e-12,
        )
        scale = 1.0 / np.sqrt(variances)
        exist_w = existing * scale[:, None]
        cand_w = candidates * scale[:, None]
        exist_norm = np.linalg.norm(exist_w, axis=0)
        active_existing = exist_norm > 1.0e-12
        if not np.any(active_existing):
            return np.ones(candidates.shape[1], dtype=bool)
        exist_unit = exist_w[:, active_existing] / exist_norm[active_existing][None, :]
        cand_norm = np.linalg.norm(cand_w, axis=0)
        active_candidate = cand_norm > 1.0e-12
        keep = np.zeros(candidates.shape[1], dtype=bool)
        if not np.any(active_candidate):
            return keep
        cand_unit = cand_w[:, active_candidate] / cand_norm[active_candidate][None, :]
        try:
            u, svals, _ = np.linalg.svd(exist_unit, full_matrices=False)
        except np.linalg.LinAlgError:
            return keep
        if svals.size == 0:
            return np.ones(candidates.shape[1], dtype=bool)
        rank_tol = max(exist_unit.shape) * np.finfo(float).eps * max(float(svals[0]), 1.0)
        rank = int(np.count_nonzero(svals > rank_tol))
        if rank <= 0:
            return np.ones(candidates.shape[1], dtype=bool)
        basis = u[:, :rank]
        projection = basis @ (basis.T @ cand_unit)
        residual_norm = np.linalg.norm(cand_unit - projection, axis=0)
        # For two normalized columns, cond <= k corresponds to
        # ||orthogonal residual|| >= 2k / (k^2 + 1).  This maps the configured
        # condition limit to an incremental independence threshold.
        min_residual = 2.0 * max_condition / (max_condition * max_condition + 1.0)
        keep[np.flatnonzero(active_candidate)] = residual_norm >= min_residual
        return keep

    def _birth_residual_survives_strength_refit(
        self,
        *,
        data: MeasurementData,
        particle_indices: NDArray[np.int64],
        particle_weights: NDArray[np.float64],
        residual_sum_before: float,
    ) -> bool:
        """
        Return True when residual evidence remains after refitting existing sources.

        A positive residual should create a new source only when it cannot be
        explained by re-estimating the strengths of sources already present in
        high-posterior particles. This makes the birth move a residual-support
        test rather than a response to a stale strength estimate.
        """
        return self._birth_residual_survives_strength_refit_batched(
            data=data,
            particle_indices=particle_indices,
            particle_weights=particle_weights,
            residual_sum_before=residual_sum_before,
        )

    def _birth_residual_survives_strength_refit_scalar(
        self,
        *,
        data: MeasurementData,
        particle_indices: NDArray[np.int64],
        particle_weights: NDArray[np.float64],
        residual_sum_before: float,
    ) -> bool:
        """Scalar reference for residual survival after strength refitting."""
        self.last_birth_residual_refit_fraction = 1.0
        self.last_birth_residual_refit_gate_passed = True
        if not bool(self.config.birth_refit_residual_gate):
            return True
        before = max(float(residual_sum_before), 1.0e-12)
        if data.z_k.size == 0:
            return False
        residuals: list[NDArray[np.float64]] = []
        weights = np.asarray(particle_weights, dtype=float).ravel()
        if weights.size == 0 or float(np.sum(weights)) <= 0.0:
            weights = np.ones(len(particle_indices), dtype=float)
        weights = weights / max(float(np.sum(weights)), 1.0e-12)
        for local_idx, particle_idx in enumerate(np.asarray(particle_indices, dtype=int).ravel()):
            st = self.continuous_particles[int(particle_idx)].state.copy()
            self._ensure_source_metadata(st)
            self._refit_strengths_for_particle(
                st,
                data,
                iters=max(1, int(self.config.refit_iters)),
                eps=float(self.config.refit_eps),
            )
            _, lambda_total = self._lambda_components(st, data)
            residual = np.maximum(data.z_k - lambda_total, 0.0)
            clip_q = float(self.config.birth_residual_clip_quantile)
            if 0.0 < clip_q < 1.0 and residual.size:
                clip_val = float(np.quantile(residual, clip_q))
                residual = np.minimum(residual, clip_val)
            residuals.append(residual * float(weights[local_idx]))
        if not residuals:
            self.last_birth_residual_refit_fraction = 0.0
            self.last_birth_residual_refit_gate_passed = False
            return False
        residual_after = np.sum(np.vstack(residuals), axis=0)
        after_sum = float(np.sum(residual_after))
        fraction = after_sum / before
        self.last_birth_residual_refit_fraction = float(fraction)
        min_fraction = max(float(self.config.birth_refit_residual_min_fraction), 0.0)
        passed = fraction >= min_fraction and self._birth_residual_gate_allows(
            residual_after,
            data.observation_variances,
            data.detector_positions,
            data.fe_indices,
            data.pb_indices,
        )
        self.last_birth_residual_refit_gate_passed = bool(passed)
        return bool(passed)

    def _birth_residual_survives_strength_refit_batched(
        self,
        *,
        data: MeasurementData,
        particle_indices: NDArray[np.int64],
        particle_weights: NDArray[np.float64],
        residual_sum_before: float,
    ) -> bool:
        """Batched residual survival test after fixed-position strength refits."""
        self.last_birth_residual_refit_fraction = 1.0
        self.last_birth_residual_refit_gate_passed = True
        if not bool(self.config.birth_refit_residual_gate):
            return True
        before = max(float(residual_sum_before), 1.0e-12)
        if data.z_k.size == 0:
            return False
        indices = np.asarray(particle_indices, dtype=int).ravel()
        weights = np.asarray(particle_weights, dtype=float).ravel()
        if indices.size == 0:
            self.last_birth_residual_refit_fraction = 0.0
            self.last_birth_residual_refit_gate_passed = False
            return False
        if weights.size != indices.size:
            weights = np.ones(indices.size, dtype=float)
        valid = (indices >= 0) & (indices < len(self.continuous_particles))
        indices = indices[valid]
        weights = weights[valid]
        if indices.size == 0:
            self.last_birth_residual_refit_fraction = 0.0
            self.last_birth_residual_refit_gate_passed = False
            return False
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            weights = np.ones(indices.size, dtype=float)
            weight_sum = float(indices.size)
        weights = weights / max(weight_sum, 1.0e-12)
        residual_after = np.zeros(data.z_k.size, dtype=float)
        weights_by_index: dict[int, list[float]] = {}
        for particle_idx, weight in zip(indices, weights):
            weights_by_index.setdefault(int(particle_idx), []).append(float(weight))
        grouped, fallback_indices = self._particle_indices_by_source_count(
            indices.astype(int).tolist(),
        )
        q_min = max(float(self.config.min_strength), 0.0)
        q_max = float(self.config.birth_q_max)
        if q_max < q_min:
            q_min, q_max = q_max, q_min
        for source_count, group_indices in grouped.items():
            if not group_indices:
                continue
            local_weights = np.asarray(
                [weights_by_index[int(idx)].pop(0) for idx in group_indices],
                dtype=float,
            )
            background_counts = data.live_times[:, None] * np.asarray(
                [
                    float(self.continuous_particles[idx].state.background)
                    for idx in group_indices
                ],
                dtype=float,
            )[None, :]
            if source_count <= 0:
                lambda_total = background_counts
            else:
                count = max(1, int(source_count))
                sources = np.vstack(
                    [
                        self.continuous_particles[idx].state.positions[:count]
                        for idx in group_indices
                    ]
                )
                k_flat = expected_counts_per_source(
                    kernel=self.continuous_kernel,
                    isotope=self.isotope,
                    detector_positions=data.detector_positions,
                    sources=sources,
                    strengths=np.ones(sources.shape[0], dtype=float),
                    live_times=data.live_times,
                    fe_indices=data.fe_indices,
                    pb_indices=data.pb_indices,
                    source_scale=self._measurement_source_scale_vector(
                        data.fe_indices,
                        data.pb_indices,
                    ),
                )
                k_tensor = np.asarray(k_flat, dtype=float).reshape(
                    int(data.z_k.size),
                    int(len(group_indices)),
                    count,
                )
                prior_mean = np.vstack(
                    [
                        np.asarray(
                            self.continuous_particles[idx].state.strengths[:count],
                            dtype=float,
                        )
                        for idx in group_indices
                    ]
                )
                _, lambda_total = self._solve_strengths_for_kernel_tensor_batched(
                    data,
                    k_tensor=k_tensor,
                    background_counts=background_counts,
                    prior_mean=prior_mean,
                    iters=max(1, int(self.config.refit_iters)),
                    eps=float(self.config.refit_eps),
                    q_min=q_min,
                    q_max=q_max,
                )
            residual = np.maximum(data.z_k[:, None] - lambda_total, 0.0)
            clip_q = float(self.config.birth_residual_clip_quantile)
            if 0.0 < clip_q < 1.0 and residual.size:
                clip_val = np.quantile(residual, clip_q, axis=0)
                residual = np.minimum(residual, clip_val[None, :])
            residual_after += residual @ local_weights
        for particle_idx in fallback_indices:
            st = self.continuous_particles[int(particle_idx)].state.copy()
            self._ensure_source_metadata(st)
            self._refit_strengths_for_particle(
                st,
                data,
                iters=max(1, int(self.config.refit_iters)),
                eps=float(self.config.refit_eps),
            )
            _, lambda_total = self._lambda_components(st, data)
            residual = np.maximum(data.z_k - lambda_total, 0.0)
            clip_q = float(self.config.birth_residual_clip_quantile)
            if 0.0 < clip_q < 1.0 and residual.size:
                clip_val = float(np.quantile(residual, clip_q))
                residual = np.minimum(residual, clip_val)
            local_weight = (
                weights_by_index.get(int(particle_idx), [0.0]).pop(0)
                if weights_by_index.get(int(particle_idx))
                else 0.0
            )
            residual_after += residual * local_weight
        after_sum = float(np.sum(residual_after))
        fraction = after_sum / before
        self.last_birth_residual_refit_fraction = float(fraction)
        min_fraction = max(float(self.config.birth_refit_residual_min_fraction), 0.0)
        passed = fraction >= min_fraction and self._birth_residual_gate_allows(
            residual_after,
            data.observation_variances,
            data.detector_positions,
            data.fe_indices,
            data.pb_indices,
        )
        self.last_birth_residual_refit_gate_passed = bool(passed)
        return bool(passed)

    def _birth_residual_gate_allows(
        self,
        residual_mix: NDArray[np.float64],
        observation_variances: NDArray[np.float64],
        detector_positions: NDArray[np.float64] | None = None,
        fe_indices: NDArray[np.int64] | None = None,
        pb_indices: NDArray[np.int64] | None = None,
    ) -> bool:
        """
        Return True when positive residuals statistically justify a birth move.

        The gate tests whether the posterior residual that cannot be explained by
        the current particles is large relative to the spectrum/PF observation
        variance. It prevents a single low-confidence spectrum decomposition
        outlier from creating persistent false-positive sources.
        """
        residual = np.maximum(np.asarray(residual_mix, dtype=float).reshape(-1), 0.0)
        if residual.size == 0:
            return False
        variances = self._measurement_vector(
            observation_variances,
            residual.size,
            "observation_variances",
            min_value=1.0e-12,
        )
        sigma = np.sqrt(variances)
        z_score = residual / np.maximum(sigma, 1.0e-12)
        min_sigma = max(float(self.config.birth_residual_support_sigma), 0.0)
        support_mask = z_score >= min_sigma
        support_count = int(np.count_nonzero(support_mask))
        distinct_supported = self._distinct_supported_view_count(
            detector_positions,
            fe_indices,
            pb_indices,
            support_mask,
        )
        distinct_stations = self._distinct_supported_station_count(
            detector_positions,
            support_mask,
        )
        chi2_stat = float(np.sum((residual[support_mask] ** 2) / variances[support_mask]))
        dof = max(support_count, 1)
        p_value = float(chi2.sf(chi2_stat, dof)) if support_count > 0 else 1.0
        self.last_birth_residual_chi2 = chi2_stat
        self.last_birth_residual_p_value = p_value
        self.last_birth_residual_support = support_count
        self.last_birth_residual_distinct_poses = distinct_supported
        self.last_birth_residual_distinct_stations = distinct_stations
        min_support = max(1, int(self.config.birth_residual_min_support))
        min_distinct = max(1, int(self.config.birth_min_distinct_poses))
        min_stations = max(1, int(self.config.birth_min_distinct_stations))
        p_threshold = float(self.config.birth_residual_gate_p_value)
        if p_threshold <= 0.0:
            passed = (
                support_count >= min_support
                and distinct_supported >= min_distinct
                and distinct_stations >= min_stations
                and chi2_stat > 0.0
            )
        else:
            p_threshold = min(max(p_threshold, 0.0), 1.0)
            passed = (
                support_count >= min_support
                and distinct_supported >= min_distinct
                and distinct_stations >= min_stations
                and p_value <= p_threshold
            )
        self.last_birth_residual_gate_passed = bool(passed)
        return bool(passed)

    def _distinct_supported_view_count(
        self,
        detector_positions: NDArray[np.float64] | None,
        fe_indices: NDArray[np.int64] | None,
        pb_indices: NDArray[np.int64] | None,
        support_mask: NDArray[np.bool_],
    ) -> int:
        """Return the number of distinct pose/shield views with residual support."""
        if detector_positions is None:
            return int(np.count_nonzero(support_mask))
        positions = np.asarray(detector_positions, dtype=float)
        mask = np.asarray(support_mask, dtype=bool).ravel()
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] != mask.size:
            return int(np.count_nonzero(mask))
        if not np.any(mask):
            return 0
        rounded = np.round(positions[mask], decimals=3)
        if fe_indices is None or pb_indices is None:
            return int(np.unique(rounded, axis=0).shape[0])
        fe = np.asarray(fe_indices, dtype=int).reshape(-1)
        pb = np.asarray(pb_indices, dtype=int).reshape(-1)
        if fe.size != mask.size or pb.size != mask.size:
            return int(np.unique(rounded, axis=0).shape[0])
        views = np.column_stack([rounded, fe[mask], pb[mask]])
        return int(np.unique(views, axis=0).shape[0])

    def _distinct_supported_station_count(
        self,
        detector_positions: NDArray[np.float64] | None,
        support_mask: NDArray[np.bool_],
    ) -> int:
        """Return the number of distinct robot stations with residual support."""
        if detector_positions is None:
            return int(np.count_nonzero(support_mask))
        positions = np.asarray(detector_positions, dtype=float)
        mask = np.asarray(support_mask, dtype=bool).ravel()
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] != mask.size:
            return int(np.count_nonzero(mask))
        if not np.any(mask):
            return 0
        rounded_xy = np.round(positions[mask, :2], decimals=3)
        return int(np.unique(rounded_xy, axis=0).shape[0])

    def _source_prune_support_ready(self, data: MeasurementData) -> bool:
        """Return True when a measurement block can justify source removal."""
        if data.z_k.size == 0:
            return False
        full_support = np.ones(data.z_k.size, dtype=bool)
        distinct_views = self._distinct_supported_view_count(
            data.detector_positions,
            data.fe_indices,
            data.pb_indices,
            full_support,
        )
        distinct_stations = self._distinct_supported_station_count(
            data.detector_positions,
            full_support,
        )
        return (
            distinct_views >= max(1, int(self.config.source_prune_min_distinct_views))
            and distinct_stations
            >= max(1, int(self.config.source_prune_min_distinct_stations))
        )

    def _source_prune_delta_threshold(self) -> float:
        """Return the leave-one-out ΔLL threshold used for source removal."""
        return float(self.config.source_prune_delta_ll_threshold)

    def _bic_model_penalty(self, measurement_count: int, parameter_count: int) -> float:
        """Return half-BIC penalty gain for removing model parameters."""
        params = max(0, int(parameter_count))
        if params <= 0:
            return 0.0
        count = max(2, int(measurement_count))
        return 0.5 * float(params) * float(np.log(count))

    def _remove_source_trial(self, st: IsotopeState, source_idx: int) -> IsotopeState:
        """Return a copy of a state with one source removed."""
        trial = st.copy()
        self._ensure_source_metadata(trial)
        if trial.num_sources <= 0:
            return trial
        keep = np.ones(int(trial.num_sources), dtype=bool)
        keep[int(source_idx)] = False
        self._apply_source_keep_mask(trial, keep)
        return trial

    def _source_prune_refit_after_remove_mask(
        self,
        st: IsotopeState,
        data: MeasurementData,
    ) -> NDArray[np.bool_]:
        """Return sources removable after refitting the remaining strengths."""
        if st.num_sources <= 1 or data.z_k.size == 0:
            return np.zeros(max(0, int(st.num_sources)), dtype=bool)
        full = st.copy()
        self._ensure_source_metadata(full)
        self._refit_strengths_for_particle(
            full,
            data,
            iters=max(1, int(self.config.refit_iters)),
            eps=float(self.config.refit_eps),
        )
        ll_full = self._trial_log_likelihood(full, data)
        if not np.isfinite(ll_full):
            return np.zeros(int(st.num_sources), dtype=bool)
        penalty_gain = self._bic_model_penalty(
            int(data.z_k.size),
            int(self.config.source_prune_bic_penalty_params),
        )
        allowed_loss = penalty_gain + float(self.config.source_prune_delta_ll_threshold)
        removable = np.zeros(int(st.num_sources), dtype=bool)
        for source_idx in range(int(st.num_sources)):
            reduced = self._remove_source_trial(full, source_idx)
            if reduced.num_sources <= 0:
                continue
            self._refit_strengths_for_particle(
                reduced,
                data,
                iters=max(1, int(self.config.refit_iters)),
                eps=float(self.config.refit_eps),
            )
            ll_without = self._trial_log_likelihood(reduced, data)
            if not np.isfinite(ll_without):
                continue
            loss = float(ll_full - ll_without)
            removable[source_idx] = loss <= allowed_loss
        return removable

    def _solve_strengths_for_kernel_tensor_batched(
        self,
        data: MeasurementData,
        *,
        k_tensor: NDArray[np.float64],
        background_counts: NDArray[np.float64],
        prior_mean: NDArray[np.float64],
        iters: int,
        eps: float,
        q_min: float,
        q_max: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Return batched MAP strengths and total counts for fixed source geometry.

        The input tensor has shape K x P x S, where K is the number of
        measurements, P is the number of particles, and S is the source count.
        """
        k_arr = np.asarray(k_tensor, dtype=float)
        if k_arr.ndim != 3:
            raise ValueError("k_tensor must have shape K x P x S.")
        num_meas, particle_count, source_count = k_arr.shape
        bg = np.asarray(background_counts, dtype=float)
        if bg.shape != (num_meas, particle_count):
            raise ValueError("background_counts must have shape K x P.")
        if source_count <= 0 or particle_count <= 0:
            return (
                np.zeros((particle_count, source_count), dtype=float),
                bg.copy(),
            )
        prior = np.asarray(prior_mean, dtype=float)
        if prior.shape != (particle_count, source_count):
            raise ValueError("prior_mean must have shape P x S.")
        z_arr = np.asarray(data.z_k, dtype=float).reshape(-1)
        if z_arr.size != num_meas:
            raise ValueError("data.z_k must have one value per measurement.")
        obs_variances = self._measurement_vector(
            data.observation_variances,
            num_meas,
            "observation_variances",
            min_value=1.0,
        )
        obs_weights = 1.0 / obs_variances
        strengths = np.clip(np.maximum(prior.copy(), 0.0), q_min, q_max)
        local_precision = self._strength_refit_prior_precision(prior)
        abs_precision, abs_mean = self._absolute_strength_prior_terms(prior.shape)
        prior_precision = local_precision + abs_precision
        prior_target = np.divide(
            local_precision * prior + abs_precision * abs_mean,
            np.maximum(prior_precision, 1.0e-12),
            out=prior.copy(),
            where=prior_precision > 0.0,
        )
        if prior_precision.shape != prior.shape:
            raise ValueError("strength prior precision must match prior_mean.")

        gram = np.einsum("kps,kpt,k->pst", k_arr, k_arr, obs_weights)
        rhs = np.einsum(
            "kps,kp,k->ps",
            k_arr,
            z_arr[:, None] - bg,
            obs_weights,
        )
        prior_diag = np.zeros_like(gram, dtype=float)
        diag_idx = np.arange(source_count)
        prior_diag[:, diag_idx, diag_idx] = prior_precision
        rhs = rhs + prior_precision * prior_target
        eye = np.eye(source_count, dtype=float)[None, :, :] * float(eps)
        try:
            direct = np.linalg.solve(gram + prior_diag + eye, rhs[:, :, None])
            strengths = np.clip(np.maximum(direct[:, :, 0], 0.0), q_min, q_max)
            strengths = self._apply_runtime_strength_absorption_guard(strengths)
            strengths = self._apply_runtime_observation_overshoot_guard(
                strengths,
                k_arr,
                data,
                bg,
                eps=eps,
            )
        except np.linalg.LinAlgError:
            pass

        lambda_total = bg + np.einsum("kps,ps->kp", k_arr, strengths)
        weight_col = obs_weights[:, None]
        z_col = z_arr[:, None]
        for _ in range(max(1, int(iters))):
            for source_idx in range(source_count):
                k_col = k_arr[:, :, source_idx]
                prior_col = prior_precision[:, source_idx]
                denom = (
                    np.sum(weight_col * k_col * k_col, axis=0)
                    + prior_col
                    + float(eps)
                )
                old_strength = strengths[:, source_idx].copy()
                residual = z_col - (lambda_total - old_strength[None, :] * k_col)
                numer = (
                    np.sum(weight_col * residual * k_col, axis=0)
                    + prior_col * prior_target[:, source_idx]
                )
                q_new = np.divide(
                    numer,
                    denom,
                    out=np.zeros_like(numer, dtype=float),
                    where=denom > 0.0,
                )
                q_new = np.clip(np.maximum(q_new, 0.0), q_min, q_max)
                lambda_total += (q_new - old_strength)[None, :] * k_col
                strengths[:, source_idx] = q_new
            guarded = self._apply_runtime_strength_absorption_guard(strengths)
            guarded = self._apply_runtime_observation_overshoot_guard(
                guarded,
                k_arr,
                data,
                bg,
                eps=eps,
            )
            if not np.allclose(guarded, strengths, rtol=0.0, atol=0.0):
                strengths = guarded
                lambda_total = bg + np.einsum("kps,ps->kp", k_arr, strengths)
        return strengths, lambda_total

    def _apply_runtime_observation_overshoot_guard(
        self,
        strengths: NDArray[np.float64],
        k_tensor: NDArray[np.float64],
        data: MeasurementData,
        background_counts: NDArray[np.float64],
        *,
        eps: float,
    ) -> NDArray[np.float64]:
        """Shrink source strengths only when their own response exceeds observations."""
        q_arr = np.asarray(strengths, dtype=float)
        weight = max(
            0.0,
            float(
                getattr(
                    self.config,
                    "source_strength_observation_overshoot_penalty_weight",
                    0.0,
                )
            ),
        )
        if weight <= 0.0 or q_arr.size == 0:
            return q_arr
        k_arr = np.maximum(np.asarray(k_tensor, dtype=float), 0.0)
        if k_arr.ndim != 3 or k_arr.shape[1:] != q_arr.shape:
            return q_arr
        bg = np.asarray(background_counts, dtype=float)
        if bg.shape != k_arr.shape[:2]:
            return q_arr
        z_arr = np.asarray(data.z_k, dtype=float).reshape(-1)
        if z_arr.size != k_arr.shape[0]:
            return q_arr
        variances = self._measurement_vector(
            data.observation_variances,
            z_arr.size,
            "observation_variances",
            min_value=1.0,
        )
        sigma = max(
            0.0,
            float(
                getattr(
                    self.config,
                    "source_strength_observation_overshoot_sigma",
                    5.0,
                )
            ),
        )
        allowed = (
            np.maximum(z_arr[:, None] - bg, 0.0)
            + sigma * np.sqrt(variances)[:, None]
        )
        max_response = np.max(k_arr, axis=0)
        visible_fraction = max(
            0.0,
            float(
                getattr(
                    self.config,
                    "source_strength_observation_overshoot_min_visible_fraction",
                    0.05,
                )
            ),
        )
        visible_threshold = np.maximum(float(eps), visible_fraction * max_response)
        visible = k_arr > visible_threshold[None, :, :]
        visible_count = np.sum(visible, axis=0)
        min_visible = max(
            1,
            int(
                getattr(
                    self.config,
                    "source_strength_observation_overshoot_min_visible_measurements",
                    3,
                )
            ),
        )
        quantile = float(
            np.clip(
                float(
                    getattr(
                        self.config,
                        "source_strength_observation_overshoot_quantile",
                        0.05,
                    )
                ),
                0.0,
                1.0,
            )
        )
        ratios = np.divide(
            allowed[:, :, None],
            np.maximum(k_arr, float(eps)),
            out=np.full_like(k_arr, np.inf, dtype=float),
            where=visible,
        )
        ratios = np.where(visible, ratios, np.inf)
        sorted_ratios = np.sort(ratios, axis=0)
        count_for_index = np.maximum(visible_count, 1)
        positions = quantile * np.maximum(count_for_index - 1, 0)
        lower_indices = np.floor(positions).astype(int)
        upper_indices = np.ceil(positions).astype(int)
        lower = np.take_along_axis(
            sorted_ratios,
            lower_indices[None, :, :],
            axis=0,
        )[0]
        upper = np.take_along_axis(
            sorted_ratios,
            upper_indices[None, :, :],
            axis=0,
        )[0]
        deltas = np.subtract(
            upper,
            lower,
            out=np.zeros_like(lower),
            where=np.isfinite(lower) & np.isfinite(upper),
        )
        bounds = lower + deltas * (positions - lower_indices)
        bounds = np.where(visible_count > 0, bounds, np.inf)
        bounds = np.where(
            (visible_count >= min_visible) & np.isfinite(bounds) & (bounds >= 0.0),
            bounds,
            np.inf,
        )
        high = q_arr > bounds
        if not np.any(high):
            return q_arr
        guarded = q_arr.copy()
        guarded[high] = bounds[high] + (guarded[high] - bounds[high]) / (
            1.0 + weight
        )
        return np.where(np.isfinite(guarded), np.maximum(guarded, 0.0), q_arr)

    def _apply_runtime_strength_absorption_guard(
        self,
        strengths: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Apply a one-sided runtime guard against one-source count absorption."""
        q_arr = np.asarray(strengths, dtype=float)
        weight = max(
            0.0,
            float(
                getattr(
                    self.config,
                    "source_strength_absorption_penalty_weight",
                    0.0,
                )
            ),
        )
        mean = max(float(getattr(self.config, "source_strength_prior_mean", 0.0)), 0.0)
        if weight <= 0.0 or mean <= 0.0 or q_arr.size == 0:
            return q_arr
        soft_cap = self._runtime_strength_absorption_soft_cap()
        if soft_cap <= 0.0:
            return q_arr
        guarded = q_arr.copy()
        high = guarded > soft_cap
        if np.any(high):
            shrink = 1.0 / (1.0 + weight)
            guarded[high] = soft_cap + (guarded[high] - soft_cap) * shrink
        return guarded

    def _source_prune_refit_after_remove_mask_batched(
        self,
        data: MeasurementData,
        *,
        k_tensor: NDArray[np.float64],
        background_counts: NDArray[np.float64],
        full_strengths: NDArray[np.float64],
        full_lambda_total: NDArray[np.float64],
        iters: int,
        eps: float,
        q_min: float,
        q_max: float,
    ) -> NDArray[np.bool_]:
        """Return batched refit-after-remove source-removal masks."""
        return self._source_prune_refit_after_remove_mask_vectorized(
            data,
            k_tensor=k_tensor,
            background_counts=background_counts,
            full_strengths=full_strengths,
            full_lambda_total=full_lambda_total,
            iters=iters,
            eps=eps,
            q_min=q_min,
            q_max=q_max,
        )

    def _source_prune_refit_after_remove_mask_loop(
        self,
        data: MeasurementData,
        *,
        k_tensor: NDArray[np.float64],
        background_counts: NDArray[np.float64],
        full_strengths: NDArray[np.float64],
        full_lambda_total: NDArray[np.float64],
        iters: int,
        eps: float,
        q_min: float,
        q_max: float,
    ) -> NDArray[np.bool_]:
        """Return the scalar-source-loop oracle for refit-after-remove pruning."""
        k_arr = np.asarray(k_tensor, dtype=float)
        if k_arr.ndim != 3:
            raise ValueError("k_tensor must have shape K x P x S.")
        _, particle_count, source_count = k_arr.shape
        removable = np.zeros((particle_count, source_count), dtype=bool)
        if source_count <= 1 or data.z_k.size == 0:
            return removable
        if not self._source_prune_support_ready(data):
            return removable
        ll_full = self._count_log_likelihood_matrix_np(
            data.z_k,
            full_lambda_total,
            observation_count_variance=data.observation_variances,
        )
        if ll_full.size != particle_count:
            raise ValueError("full_lambda_total must have one column per particle.")
        penalty_gain = self._bic_model_penalty(
            int(data.z_k.size),
            int(self.config.source_prune_bic_penalty_params),
        )
        allowed_loss = penalty_gain + float(self.config.source_prune_delta_ll_threshold)
        for source_idx in range(source_count):
            keep = np.ones(source_count, dtype=bool)
            keep[source_idx] = False
            if int(np.count_nonzero(keep)) <= 0:
                continue
            _, lambda_without = self._solve_strengths_for_kernel_tensor_batched(
                data,
                k_tensor=k_arr[:, :, keep],
                background_counts=background_counts,
                prior_mean=np.asarray(full_strengths, dtype=float)[:, keep],
                iters=iters,
                eps=eps,
                q_min=q_min,
                q_max=q_max,
            )
            ll_without = self._count_log_likelihood_matrix_np(
                data.z_k,
                lambda_without,
                observation_count_variance=data.observation_variances,
            )
            if ll_without.size != particle_count:
                raise ValueError("lambda_without must have one column per particle.")
            loss = ll_full - ll_without
            removable[:, source_idx] = np.isfinite(loss) & (loss <= allowed_loss)
        return removable

    def _source_prune_refit_after_remove_mask_vectorized(
        self,
        data: MeasurementData,
        *,
        k_tensor: NDArray[np.float64],
        background_counts: NDArray[np.float64],
        full_strengths: NDArray[np.float64],
        full_lambda_total: NDArray[np.float64],
        iters: int,
        eps: float,
        q_min: float,
        q_max: float,
    ) -> NDArray[np.bool_]:
        """Return refit-after-remove masks with source-removal trials flattened."""
        k_arr = np.asarray(k_tensor, dtype=float)
        if k_arr.ndim != 3:
            raise ValueError("k_tensor must have shape K x P x S.")
        num_meas, particle_count, source_count = k_arr.shape
        removable = np.zeros((particle_count, source_count), dtype=bool)
        if source_count <= 1 or data.z_k.size == 0:
            return removable
        if not self._source_prune_support_ready(data):
            return removable
        ll_full = self._count_log_likelihood_matrix_np(
            data.z_k,
            full_lambda_total,
            observation_count_variance=data.observation_variances,
        )
        if ll_full.size != particle_count:
            raise ValueError("full_lambda_total must have one column per particle.")
        penalty_gain = self._bic_model_penalty(
            int(data.z_k.size),
            int(self.config.source_prune_bic_penalty_params),
        )
        allowed_loss = penalty_gain + float(self.config.source_prune_delta_ll_threshold)
        trial_count = int(particle_count * source_count)
        reduced_count = int(source_count - 1)
        k_removed = np.zeros((num_meas, trial_count, reduced_count), dtype=float)
        prior_removed = np.zeros((trial_count, reduced_count), dtype=float)
        full_strengths_arr = np.asarray(full_strengths, dtype=float)
        if full_strengths_arr.shape != (particle_count, source_count):
            raise ValueError("full_strengths must have shape P x S.")
        for removed_idx in range(source_count):
            keep = np.ones(source_count, dtype=bool)
            keep[removed_idx] = False
            start = removed_idx * particle_count
            stop = start + particle_count
            k_removed[:, start:stop, :] = k_arr[:, :, keep]
            prior_removed[start:stop, :] = full_strengths_arr[:, keep]
        bg = np.asarray(background_counts, dtype=float)
        if bg.shape != (num_meas, particle_count):
            raise ValueError("background_counts must have shape K x P.")
        background_removed = np.tile(bg, (1, source_count))
        _, lambda_without_all = self._solve_strengths_for_kernel_tensor_batched(
            data,
            k_tensor=k_removed,
            background_counts=background_removed,
            prior_mean=prior_removed,
            iters=iters,
            eps=eps,
            q_min=q_min,
            q_max=q_max,
        )
        ll_without_all = self._count_log_likelihood_matrix_np(
            data.z_k,
            lambda_without_all,
            observation_count_variance=data.observation_variances,
        )
        if ll_without_all.size != trial_count:
            raise ValueError("lambda_without_all must have one column per trial.")
        ll_without = ll_without_all.reshape(source_count, particle_count).T
        loss = ll_full[:, None] - ll_without
        removable = np.isfinite(loss) & (loss <= allowed_loss)
        return removable

    def _source_prune_allowed_mask(
        self,
        st: IsotopeState,
        data: MeasurementData,
        *,
        lambda_m: NDArray[np.float64] | None = None,
        lambda_total: NDArray[np.float64] | None = None,
        delta_ll: NDArray[np.float64] | None = None,
    ) -> NDArray[np.bool_]:
        """
        Return sources whose removal is supported across multiple stations.

        A source is removable only when leaving it out has low likelihood loss
        in enough distinct robot stations. This prevents a single shield view or
        one low-count station from deleting a weak but physically plausible
        source hypothesis.
        """
        if st.num_sources <= 0 or data.z_k.size == 0:
            return np.zeros(max(0, int(st.num_sources)), dtype=bool)
        self._ensure_source_metadata(st)
        if not self._source_prune_support_ready(data):
            return np.zeros(int(st.num_sources), dtype=bool)
        if lambda_m is None or lambda_total is None:
            lambda_m, lambda_total = self._lambda_components(st, data)
        if lambda_m.shape != (int(data.z_k.size), int(st.num_sources)):
            return np.zeros(int(st.num_sources), dtype=bool)
        separation_allowed = self._tentative_response_separation_prune_mask(
            st,
            lambda_m,
        )
        if bool(self.config.source_prune_refit_after_remove):
            return (
                self._source_prune_refit_after_remove_mask(st, data)
                & separation_allowed
            )
        threshold = self._source_prune_delta_threshold()
        if delta_ll is None or delta_ll.shape != (int(st.num_sources),):
            delta_ll = self._delta_log_likelihood_remove(
                data.z_k,
                lambda_total,
                lambda_m,
                observation_count_variance=data.observation_variances,
            )
        station_labels = self._support_station_labels(
            data.detector_positions,
            int(data.z_k.size),
        )
        fail_counts = np.zeros(int(st.num_sources), dtype=int)
        for label in np.unique(station_labels):
            rows = station_labels == int(label)
            if not np.any(rows):
                continue
            station_delta = self._delta_log_likelihood_remove(
                data.z_k[rows],
                lambda_total[rows],
                lambda_m[rows, :],
                observation_count_variance=data.observation_variances[rows],
            )
            fail_counts += (station_delta < threshold).astype(int)
        min_stations = max(1, int(self.config.source_prune_min_distinct_stations))
        global_failed = np.asarray(delta_ll, dtype=float) < threshold
        return (fail_counts >= min_stations) & global_failed & separation_allowed

    def _tentative_response_separation_prune_mask(
        self,
        st: IsotopeState,
        lambda_m: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """
        Return sources that may be removed without confusing collinear modes.

        A tentative source whose response is nearly collinear with another source
        is not proven false; the current measurement block simply lacks enough
        differential information.  Such sources remain eligible for future
        measurements and are not deleted by prune/death until a more
        discriminative block is available.
        """
        count = max(0, int(st.num_sources))
        allowed = np.ones(count, dtype=bool)
        if count <= 1:
            return allowed
        corr_max = float(np.clip(float(self.config.pseudo_source_corr_max), 0.0, 1.0))
        if corr_max >= 1.0:
            return allowed
        self._ensure_source_metadata(st)
        tentative = np.asarray(st.tentative_sources[:count], dtype=bool)
        if not np.any(tentative):
            return allowed
        responses = np.maximum(np.asarray(lambda_m, dtype=float), 0.0)
        if responses.ndim != 2 or responses.shape[1] != count:
            return allowed
        norms = np.linalg.norm(responses, axis=0)
        for source_idx in range(count):
            if not bool(tentative[source_idx]) or norms[source_idx] <= 0.0:
                continue
            for other_idx in range(count):
                if source_idx == other_idx or norms[other_idx] <= 0.0:
                    continue
                denom = max(float(norms[source_idx] * norms[other_idx]), 1.0e-12)
                corr = float(
                    np.dot(responses[:, source_idx], responses[:, other_idx])
                    / denom
                )
                if corr >= corr_max:
                    allowed[source_idx] = False
                    break
        return allowed

    def _birth_candidate_support_mask(
        self,
        *,
        candidate_counts: NDArray[np.float64],
        residual_mix: NDArray[np.float64],
        observation_variances: NDArray[np.float64],
        detector_positions: NDArray[np.float64],
        fe_indices: NDArray[np.int64] | None = None,
        pb_indices: NDArray[np.int64] | None = None,
        min_support: int | None = None,
        min_distinct_poses: int | None = None,
        min_distinct_stations: int | None = None,
    ) -> NDArray[np.bool_]:
        """
        Return candidates whose residual explanation is coherent across views.

        A birth location is accepted only when its expected count pattern overlaps
        statistically significant positive residuals in enough measurements and
        enough distinct pose/shield views. This uses the rotating shield as an
        independent measurement primitive instead of requiring robot motion
        before a new-source hypothesis can be born.
        """
        counts = np.asarray(candidate_counts, dtype=float)
        if counts.ndim != 2 or counts.size == 0:
            return np.zeros(0, dtype=bool)
        residual = self._measurement_vector(
            residual_mix,
            counts.shape[0],
            "residual_mix",
            min_value=0.0,
            allow_scalar=False,
        )
        variances = self._measurement_vector(
            observation_variances,
            counts.shape[0],
            "observation_variances",
            min_value=1.0e-12,
        )
        sigma = np.sqrt(variances)
        z_score = residual / np.maximum(sigma, 1.0e-12)
        residual_support = z_score >= max(float(self.config.birth_residual_support_sigma), 0.0)
        if not np.any(residual_support):
            return np.zeros(counts.shape[1], dtype=bool)
        overlap = np.maximum(counts, 0.0) * residual[:, None]
        max_overlap = np.max(overlap, axis=0)
        fraction = float(self.config.birth_candidate_support_fraction)
        fraction = float(np.clip(fraction, 0.0, 1.0))
        threshold = max_overlap[None, :] * fraction
        support = (overlap >= threshold) & (max_overlap[None, :] > 0.0)
        support &= residual_support[:, None]
        support_counts = np.sum(support, axis=0)
        support_floor = (
            self.config.birth_residual_min_support
            if min_support is None
            else min_support
        )
        distinct_floor = (
            self.config.birth_min_distinct_poses
            if min_distinct_poses is None
            else min_distinct_poses
        )
        station_floor = (
            self.config.birth_min_distinct_stations
            if min_distinct_stations is None
            else min_distinct_stations
        )
        support_floor = max(1, int(support_floor))
        distinct_floor = max(1, int(distinct_floor))
        station_floor = max(1, int(station_floor))
        keep = support_counts >= support_floor
        if distinct_floor <= 1:
            view_keep = np.ones(counts.shape[1], dtype=bool)
        else:
            view_labels = self._support_view_labels(
                detector_positions,
                fe_indices,
                pb_indices,
                support.shape[0],
            )
            distinct_counts = self._distinct_label_counts_for_support_matrix(
                support,
                view_labels,
            )
            view_keep = distinct_counts >= distinct_floor
        if station_floor <= 1:
            station_keep = np.ones(counts.shape[1], dtype=bool)
        else:
            station_labels = self._support_station_labels(
                detector_positions,
                support.shape[0],
            )
            station_counts = self._distinct_label_counts_for_support_matrix(
                support,
                station_labels,
            )
            station_keep = station_counts >= station_floor
        keep &= view_keep
        keep &= station_keep
        return keep.astype(bool)

    @staticmethod
    def _distinct_label_counts_for_support_matrix(
        support: NDArray[np.bool_],
        labels: NDArray[np.int64],
    ) -> NDArray[np.int64]:
        """Count distinct measurement labels supporting each candidate column."""
        support_arr = np.asarray(support, dtype=bool)
        if support_arr.ndim != 2 or support_arr.size == 0:
            return np.zeros(support_arr.shape[1] if support_arr.ndim == 2 else 0, dtype=int)
        label_arr = np.asarray(labels, dtype=int).reshape(-1)
        if label_arr.size != support_arr.shape[0]:
            return np.sum(support_arr, axis=0).astype(int)
        counts = np.zeros(support_arr.shape[1], dtype=int)
        for label in np.unique(label_arr):
            rows = label_arr == int(label)
            counts += np.any(support_arr[rows, :], axis=0).astype(int)
        return counts

    @staticmethod
    def _support_view_labels(
        detector_positions: NDArray[np.float64] | None,
        fe_indices: NDArray[np.int64] | None,
        pb_indices: NDArray[np.int64] | None,
        measurement_count: int,
    ) -> NDArray[np.int64]:
        """Return compact labels for distinct detector pose and shield views."""
        count = max(0, int(measurement_count))
        if detector_positions is None:
            return np.arange(count, dtype=int)
        positions = np.asarray(detector_positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] != count:
            return np.arange(count, dtype=int)
        rounded = np.round(positions, decimals=3)
        if fe_indices is None or pb_indices is None:
            _, labels = np.unique(rounded, axis=0, return_inverse=True)
            return labels.astype(int, copy=False)
        fe = np.asarray(fe_indices, dtype=int).reshape(-1)
        pb = np.asarray(pb_indices, dtype=int).reshape(-1)
        if fe.size != count or pb.size != count:
            _, labels = np.unique(rounded, axis=0, return_inverse=True)
            return labels.astype(int, copy=False)
        views = np.column_stack([rounded, fe, pb])
        _, labels = np.unique(views, axis=0, return_inverse=True)
        return labels.astype(int, copy=False)

    @staticmethod
    def _support_station_labels(
        detector_positions: NDArray[np.float64] | None,
        measurement_count: int,
    ) -> NDArray[np.int64]:
        """Return compact labels for distinct detector station positions."""
        count = max(0, int(measurement_count))
        if detector_positions is None:
            return np.arange(count, dtype=int)
        positions = np.asarray(detector_positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] != count:
            return np.arange(count, dtype=int)
        rounded_xy = np.round(positions[:, :2], decimals=3)
        _, labels = np.unique(rounded_xy, axis=0, return_inverse=True)
        return labels.astype(int, copy=False)

    def _exclude_birth_candidates_near_detectors(
        self,
        candidates: NDArray[np.float64],
        data: MeasurementData,
    ) -> NDArray[np.float64]:
        """Remove birth candidates that are too close to measured detector poses."""
        min_sep = float(self.config.birth_detector_min_sep_m)
        if min_sep <= 0.0 or candidates.size == 0 or data.detector_positions.size == 0:
            return candidates
        diff = candidates[:, None, :] - data.detector_positions[None, :, :]
        distances = np.linalg.norm(diff, axis=2)
        keep = np.all(distances >= min_sep, axis=1)
        return candidates[keep]

    def _roughening_sigma_pos(self, num_particles: int) -> NDArray[np.float64]:
        """
        Compute per-axis roughening sigma based on the current particle count.

        Uses sigma = k * range * N^(-1/d) with clamping.
        """
        count = max(1, int(num_particles))
        lo = np.array(self.config.position_min, dtype=float)
        hi = np.array(self.config.position_max, dtype=float)
        span = np.maximum(hi - lo, 0.0)
        scale = float(self.config.roughening_k) * (count ** (-1.0 / 3.0))
        sigma = scale * span
        min_sigma = float(self.config.min_sigma_pos)
        max_sigma = float(self.config.max_sigma_pos)
        if max_sigma < min_sigma:
            min_sigma, max_sigma = max_sigma, min_sigma
        return np.clip(sigma, min_sigma, max_sigma)

    def _roughening_multiplier(self) -> float:
        """Return the roughening multiplier based on resamples in this observation."""
        decay = float(self.config.roughening_decay)
        min_mult = float(self.config.roughening_min_mult)
        if decay <= 0.0:
            decay = 1.0
        if min_mult < 0.0:
            min_mult = 0.0
        count = max(0, int(self._resample_count_in_observation))
        mult = decay**count
        return max(min_mult, mult)

    def regularize_continuous(
        self,
        sigma_pos: float | NDArray[np.float64] = 0.05,
        strength_log_sigma: float | None = None,
        p_birth: float = 0.05,
        p_kill: float = 0.1,
        intensity_threshold: float = 0.05,
    ) -> None:
        """
        Apply position roughening and log-space strength jitter (Sec. 3.3.4).

        Birth/death moves are handled in apply_birth_death().
        """
        sigma_pos_arr = np.asarray(sigma_pos, dtype=float)
        if sigma_pos_arr.size not in (1, 3):
            raise ValueError("sigma_pos must be a scalar or a 3-element vector.")
        log_sigma = (
            float(self.config.strength_log_sigma)
            if strength_log_sigma is None
            else float(strength_log_sigma)
        )
        log_sigma = max(log_sigma, 0.0)
        for p in self.continuous_particles:
            st = p.state
            self._ensure_source_metadata(st)
            st.background = self._background_level()
            if st.positions.size:
                st.positions = st.positions + np.random.normal(
                    scale=sigma_pos_arr,
                    size=st.positions.shape,
                )
                st.positions = self._project_positions_to_source_prior(st.positions)
                if log_sigma > 0.0:
                    logq = np.log(st.strengths + 1e-12)
                    logq = logq + np.random.normal(scale=log_sigma, size=st.strengths.shape)
                    st.strengths = np.exp(logq)
                st.strengths = np.maximum(st.strengths, 0.0)
                st.num_sources = st.positions.shape[0]

    def _refit_strengths_for_particle(
        self,
        st: IsotopeState,
        data: MeasurementData,
        *,
        iters: int,
        eps: float,
    ) -> float:
        """
        Refit per-source strengths using coordinate ascent with fixed positions.

        This stabilizes source intensities after birth/kill/split/merge moves.
        """
        if st.num_sources <= 0 or data.z_k.size == 0:
            return 0.0
        num_sources = int(st.num_sources)
        k_mat = expected_counts_per_source(
            kernel=self.continuous_kernel,
            isotope=self.isotope,
            detector_positions=data.detector_positions,
            sources=st.positions[:num_sources],
            strengths=np.ones(num_sources, dtype=float),
            live_times=data.live_times,
            fe_indices=data.fe_indices,
            pb_indices=data.pb_indices,
            source_scale=self._measurement_source_scale_vector(
                data.fe_indices,
                data.pb_indices,
            ),
        )
        k_mat = np.asarray(k_mat, dtype=float)
        if k_mat.shape != (int(data.z_k.size), num_sources):
            raise ValueError("unit response matrix must have shape K x S.")
        q_min = max(float(self.config.min_strength), 0.0)
        q_max = float(self.config.birth_q_max)
        if q_max < q_min:
            q_min, q_max = q_max, q_min
        strengths = np.asarray(st.strengths[:num_sources], dtype=float)
        background_counts = float(st.background) * data.live_times
        lambda_before = background_counts + k_mat @ strengths
        ll_before = self._count_log_likelihood_np(
            data.z_k,
            lambda_before,
            observation_count_variance=data.observation_variances,
        )
        obs_weights = 1.0 / np.maximum(
            np.asarray(data.observation_variances, dtype=float),
            1.0,
        )
        prior_mean = strengths.copy()
        local_precision = self._strength_refit_prior_precision(prior_mean)
        abs_precision, abs_mean = self._absolute_strength_prior_terms(
            prior_mean.shape
        )
        prior_precision = local_precision + abs_precision
        prior_target = np.divide(
            local_precision * prior_mean + abs_precision * abs_mean,
            np.maximum(prior_precision, 1.0e-12),
            out=prior_mean.copy(),
            where=prior_precision > 0.0,
        )
        gram = (k_mat.T * obs_weights[None, :]) @ k_mat
        rhs = (k_mat.T * obs_weights[None, :]) @ (data.z_k - background_counts)
        try:
            gram = gram + np.diag(prior_precision)
            rhs = rhs + prior_precision * prior_target
            direct = np.linalg.solve(
                gram + np.eye(num_sources, dtype=float) * float(eps),
                rhs,
            )
            strengths = np.clip(np.maximum(direct, 0.0), q_min, q_max)
        except np.linalg.LinAlgError:
            pass
        lambda_total = background_counts + k_mat @ strengths
        for _ in range(max(1, int(iters))):
            for j in range(num_sources):
                k_col = k_mat[:, j]
                denom = float(
                    np.sum(obs_weights * k_col * k_col)
                    + prior_precision[j]
                    + float(eps)
                )
                if denom <= 0.0:
                    strengths[j] = 0.0
                    continue
                residual = data.z_k - (lambda_total - strengths[j] * k_col)
                numer = float(
                    np.sum(obs_weights * residual * k_col)
                    + prior_precision[j] * prior_target[j]
                )
                q_new = max(0.0, numer / denom)
                q_new = float(np.clip(q_new, q_min, q_max))
                if q_new != strengths[j]:
                    lambda_total = lambda_total - strengths[j] * k_col + q_new * k_col
                    strengths[j] = q_new
        st.strengths = strengths
        st.num_sources = st.positions.shape[0]
        ll_after = self._count_log_likelihood_np(
            data.z_k,
            lambda_total,
            observation_count_variance=data.observation_variances,
        )
        prior_ratio = self._strength_refit_prior_log_ratio(prior_mean, strengths)
        return float(ll_after - ll_before + prior_ratio)

    def refit_strengths_for_particles(
        self,
        data: MeasurementData | None,
        *,
        iters: int | None = None,
        eps: float | None = None,
        suppress_prune_after_refit: bool = False,
    ) -> None:
        """
        Refit all particle strengths conditioned on their sampled positions.

        This is a Rao-Blackwellized-style deterministic update: the PF keeps the
        nonlinear source positions in particles, while source rates are projected
        to the non-negative weighted least-squares optimum for recent
        spectrum-derived counts.

        When ``suppress_prune_after_refit`` is true, the deterministic strength
        projection is still performed, but floor-strength source deletion is
        deferred to the explicit structural model-selection step.
        """
        if data is None or data.z_k.size == 0:
            return
        if not bool(self.config.conditional_strength_refit):
            return
        data = self._signal_bearing_refit_data(data)
        if data is None or data.z_k.size == 0:
            return
        max_iters = (
            int(self.config.conditional_strength_refit_iters)
            if iters is None
            else int(iters)
        )
        eps_value = float(self.config.refit_eps if eps is None else eps)
        reweight = bool(self.config.conditional_strength_refit_reweight)
        corrections = np.zeros(len(self.continuous_particles), dtype=float)
        grouped: dict[int, list[int]] = {}
        fallback_indices: list[int] = []
        for idx, particle in enumerate(self.continuous_particles):
            st = particle.state
            self._ensure_source_metadata(st)
            source_count = int(st.num_sources)
            if source_count <= 0:
                continue
            if st.positions.shape[0] < source_count or st.strengths.size < source_count:
                fallback_indices.append(idx)
                continue
            grouped.setdefault(source_count, []).append(idx)
        for source_count, particle_indices in grouped.items():
            group_corrections = self._refit_fixed_source_count_particles_batched(
                data,
                particle_indices=particle_indices,
                source_count=source_count,
                iters=max_iters,
                eps=eps_value,
                compute_loglike_correction=reweight,
                suppress_prune_after_refit=suppress_prune_after_refit,
            )
            if reweight and group_corrections.size == len(particle_indices):
                corrections[np.asarray(particle_indices, dtype=int)] = group_corrections
        for idx in fallback_indices:
            st = self.continuous_particles[idx].state
            correction = self._refit_strengths_for_particle(
                st,
                data,
                iters=max_iters,
                eps=eps_value,
            )
            if reweight:
                corrections[int(idx)] = float(correction)
            self._prune_floor_sources_after_refit(
                st,
                data,
                suppress_prune=suppress_prune_after_refit,
            )
        if reweight and corrections.size:
            clip = max(float(self.config.conditional_strength_refit_reweight_clip), 0.0)
            if clip > 0.0:
                corrections = np.clip(corrections, -clip, clip)
            if bool(self.config.conditional_strength_refit_cardinality_neutral_reweight):
                corrections = self._cardinality_neutral_refit_corrections(corrections)
            for particle, delta in zip(self.continuous_particles, corrections):
                particle.log_weight += float(delta)
            self._normalize_continuous_log_weights()
            self._maybe_resample_after_structural_update()
        self.align_continuous_labels()

    def _refit_particle_indices_batched(
        self,
        data: MeasurementData,
        particle_indices: list[int],
        *,
        iters: int,
        eps: float,
        suppress_prune_after_refit: bool = False,
    ) -> None:
        """Refit selected particles by grouped cardinality after structural moves."""
        if not particle_indices or data.z_k.size == 0:
            return
        grouped, fallback_indices = self._particle_indices_by_source_count(particle_indices)
        for source_count, group_indices in grouped.items():
            if source_count <= 0 or not group_indices:
                continue
            self._refit_fixed_source_count_particles_batched(
                data,
                particle_indices=group_indices,
                source_count=source_count,
                iters=iters,
                eps=eps,
                compute_loglike_correction=False,
                suppress_prune_after_refit=suppress_prune_after_refit,
            )
        for idx in fallback_indices:
            st = self.continuous_particles[idx].state
            self._refit_strengths_for_particle(
                st,
                data,
                iters=iters,
                eps=eps,
            )
            self._prune_floor_sources_after_refit(
                st,
                data,
                suppress_prune=suppress_prune_after_refit,
            )

    def _refit_single_source_particles_batched(
        self,
        data: MeasurementData,
        *,
        particle_indices: list[int],
        eps: float,
    ) -> None:
        """Refit one-source particle strengths with a batched kernel evaluation."""
        self._refit_fixed_source_count_particles_batched(
            data,
            particle_indices=particle_indices,
            source_count=1,
            iters=1,
            eps=eps,
        )

    def _refit_fixed_source_count_particles_batched(
        self,
        data: MeasurementData,
        *,
        particle_indices: list[int],
        source_count: int,
        iters: int,
        eps: float,
        compute_loglike_correction: bool = False,
        suppress_prune_after_refit: bool = False,
    ) -> NDArray[np.float64]:
        """Refit equal-cardinality particles using one batched kernel evaluation."""
        if not particle_indices:
            return np.zeros(0, dtype=float)
        count = max(1, int(source_count))
        sources = np.vstack(
            [
                self.continuous_particles[idx].state.positions[:count]
                for idx in particle_indices
            ]
        )
        unit_strengths = np.ones(sources.shape[0], dtype=float)
        k_flat = expected_counts_per_source(
            kernel=self.continuous_kernel,
            isotope=self.isotope,
            detector_positions=data.detector_positions,
            sources=sources,
            strengths=unit_strengths,
            live_times=data.live_times,
            fe_indices=data.fe_indices,
            pb_indices=data.pb_indices,
            source_scale=self._measurement_source_scale_vector(
                data.fe_indices,
                data.pb_indices,
            ),
        )
        num_meas = int(data.z_k.size)
        particle_count = int(len(particle_indices))
        k_tensor = k_flat.reshape(num_meas, particle_count, count)
        q_min = max(float(self.config.min_strength), 0.0)
        q_max = float(self.config.birth_q_max)
        if q_max < q_min:
            q_min, q_max = q_max, q_min
        live_arr = np.asarray(data.live_times, dtype=float)
        backgrounds = np.asarray(
            [
                float(self.continuous_particles[idx].state.background)
                for idx in particle_indices
            ],
            dtype=float,
        )
        strengths = np.vstack(
            [
                np.asarray(
                    self.continuous_particles[idx].state.strengths[:count],
                    dtype=float,
                )
                for idx in particle_indices
            ]
        )
        background_counts = live_arr[:, None] * backgrounds[None, :]
        strengths_before = strengths.copy()
        prior_mean = strengths_before.copy()
        lambda_before = background_counts + np.einsum(
            "kps,ps->kp",
            k_tensor,
            strengths_before,
        )
        strengths, lambda_total = self._solve_strengths_for_kernel_tensor_batched(
            data,
            k_tensor=k_tensor,
            background_counts=background_counts,
            prior_mean=prior_mean,
            iters=max(1, int(iters)),
            eps=float(eps),
            q_min=q_min,
            q_max=q_max,
        )
        expected_source_counts = np.sum(k_tensor * strengths[None, :, :], axis=0)
        lambda_after_for_prune = lambda_total
        lambda_m_after = k_tensor * strengths[None, :, :]
        observable_counts_matrix = self._weak_source_observable_counts_from_unit_response(
            k_tensor,
            signal_counts=np.maximum(
                data.z_k[:, None] - background_counts,
                0.0,
            ),
            observation_variances=data.observation_variances,
        )
        drop_allowed_matrix: NDArray[np.bool_] | None = None
        if (
            not bool(suppress_prune_after_refit)
            and bool(self.config.source_prune_refit_after_remove)
        ):
            drop_allowed_matrix = self._source_prune_refit_after_remove_mask_batched(
                data,
                k_tensor=k_tensor,
                background_counts=background_counts,
                full_strengths=strengths,
                full_lambda_total=lambda_after_for_prune,
                iters=max(1, int(iters)),
                eps=float(eps),
                q_min=q_min,
                q_max=q_max,
            )
        for row_idx, particle_idx in enumerate(particle_indices):
            st = self.continuous_particles[particle_idx].state
            st.strengths[:count] = strengths[row_idx]
            st.num_sources = st.positions.shape[0]
            if bool(suppress_prune_after_refit):
                continue
            if drop_allowed_matrix is not None:
                drop_allowed = drop_allowed_matrix[row_idx]
            else:
                drop_allowed = self._source_prune_allowed_mask(
                    st,
                    data,
                    lambda_m=lambda_m_after[:, row_idx, :],
                    lambda_total=lambda_after_for_prune[:, row_idx],
                )
            self._prune_floor_sources_by_expected_counts(
                st,
                expected_source_counts[row_idx],
                drop_allowed_mask=drop_allowed,
                observable_view_counts=observable_counts_matrix[row_idx],
            )
        if not compute_loglike_correction:
            return np.zeros(len(particle_indices), dtype=float)
        lambda_after = background_counts + np.einsum("kps,ps->kp", k_tensor, strengths)
        ll_before = self._count_log_likelihood_matrix_np(
            data.z_k,
            lambda_before,
            observation_count_variance=data.observation_variances,
        )
        ll_after = self._count_log_likelihood_matrix_np(
            data.z_k,
            lambda_after,
            observation_count_variance=data.observation_variances,
        )
        prior_ratio = self._strength_refit_prior_log_ratio_batched(
            prior_mean,
            strengths,
        )
        return np.asarray(ll_after - ll_before + prior_ratio, dtype=float)

    def _weak_source_visibility_reference_strength(self) -> float:
        """Return the configured absolute visibility reference, or zero if disabled."""
        configured = max(
            0.0,
            float(self.config.weak_source_prune_visibility_reference_strength),
        )
        if configured > 0.0:
            return configured
        return 0.0

    def _data_driven_visibility_strengths_from_unit_response(
        self,
        unit_counts: NDArray[np.float64],
        *,
        signal_counts: NDArray[np.float64] | None,
        observation_variances: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        """Estimate per-candidate visibility strengths from observed signal counts."""
        unit_arr = np.maximum(np.asarray(unit_counts, dtype=float), 0.0)
        if unit_arr.ndim < 2 or signal_counts is None:
            return np.zeros(unit_arr.shape[1:], dtype=float)
        prefix_shape = unit_arr.shape[:-1]
        signal_arr = np.maximum(np.asarray(signal_counts, dtype=float), 0.0)
        if signal_arr.ndim == 1:
            if signal_arr.size != unit_arr.shape[0]:
                return np.zeros(unit_arr.shape[1:], dtype=float)
            signal_arr = signal_arr.reshape(
                (unit_arr.shape[0],) + (1,) * (unit_arr.ndim - 2)
            )
        elif signal_arr.shape != prefix_shape:
            return np.zeros(unit_arr.shape[1:], dtype=float)
        if observation_variances is None:
            weights = np.ones_like(signal_arr, dtype=float)
        else:
            var_arr = np.maximum(np.asarray(observation_variances, dtype=float), 1.0)
            if var_arr.ndim == 1:
                if var_arr.size != unit_arr.shape[0]:
                    weights = np.ones_like(signal_arr, dtype=float)
                else:
                    weights = 1.0 / var_arr.reshape(
                        (unit_arr.shape[0],) + (1,) * (unit_arr.ndim - 2)
                    )
            elif var_arr.shape == signal_arr.shape:
                weights = 1.0 / var_arr
            else:
                weights = np.ones_like(signal_arr, dtype=float)
        weighted_signal = weights[..., None] * signal_arr[..., None]
        numerator = np.sum(weighted_signal * unit_arr, axis=0)
        denominator = np.sum(weights[..., None] * unit_arr * unit_arr, axis=0)
        return np.divide(
            numerator,
            np.maximum(denominator, 1.0e-12),
            out=np.zeros_like(numerator, dtype=float),
            where=denominator > 1.0e-12,
        )

    def _weak_source_reference_counts_from_unit_response(
        self,
        unit_counts: NDArray[np.float64],
        *,
        signal_counts: NDArray[np.float64] | None = None,
        observation_variances: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Return per-measurement visibility counts without assuming a fixed source rate."""
        unit_arr = np.maximum(np.asarray(unit_counts, dtype=float), 0.0)
        if unit_arr.ndim < 2:
            return np.zeros_like(unit_arr, dtype=float)
        ref_strength = self._weak_source_visibility_reference_strength()
        if ref_strength > 0.0:
            return unit_arr * ref_strength
        q_hat = self._data_driven_visibility_strengths_from_unit_response(
            unit_arr,
            signal_counts=signal_counts,
            observation_variances=observation_variances,
        )
        if np.any(q_hat > 0.0):
            return unit_arr * np.expand_dims(q_hat, axis=0)
        prior_mean = max(float(self.config.source_strength_prior_mean), 0.0)
        if prior_mean > 0.0:
            return unit_arr * prior_mean
        birth_min = max(float(self.config.birth_q_min), 0.0)
        if birth_min > 0.0:
            return unit_arr * birth_min
        min_strength = max(float(self.config.min_strength), 1.0)
        return unit_arr * min_strength

    def _weak_source_observable_counts_from_unit_response(
        self,
        unit_counts: NDArray[np.float64],
        *,
        signal_counts: NDArray[np.float64] | None = None,
        observation_variances: NDArray[np.float64] | None = None,
    ) -> NDArray[np.int64]:
        """Count measurements where a source would have been observable."""
        unit_arr = np.maximum(np.asarray(unit_counts, dtype=float), 0.0)
        if unit_arr.ndim < 2:
            return np.zeros(0, dtype=np.int64)
        reference_counts = self._weak_source_reference_counts_from_unit_response(
            unit_arr,
            signal_counts=signal_counts,
            observation_variances=observation_variances,
        )
        min_count = max(
            float(self.config.weak_source_prune_observable_count),
            float(self.config.weak_source_prune_min_expected_count),
            0.0,
        )
        min_fraction = max(
            float(self.config.weak_source_prune_observable_fraction),
            float(self.config.weak_source_prune_min_fraction),
            0.0,
        )
        visible = np.zeros(reference_counts.shape, dtype=bool)
        if min_count > 0.0:
            visible |= reference_counts >= min_count
        if min_fraction > 0.0:
            total = np.sum(reference_counts, axis=-1, keepdims=True)
            fraction = np.divide(
                reference_counts,
                np.maximum(total, 1.0e-12),
                out=np.zeros_like(reference_counts, dtype=float),
                where=total > 0.0,
            )
            visible |= fraction >= min_fraction
        if min_count <= 0.0 and min_fraction <= 0.0:
            visible = reference_counts > 0.0
        return np.count_nonzero(visible, axis=0).astype(np.int64, copy=False)

    def _weak_source_observable_counts_from_components(
        self,
        lambda_m: NDArray[np.float64],
        strengths: NDArray[np.float64],
        *,
        signal_counts: NDArray[np.float64] | None = None,
        observation_variances: NDArray[np.float64] | None = None,
    ) -> NDArray[np.int64]:
        """Count observable measurements from per-source count components."""
        lambda_arr = np.maximum(np.asarray(lambda_m, dtype=float), 0.0)
        strength_arr = np.asarray(strengths, dtype=float).reshape(-1)
        if lambda_arr.ndim != 2 or lambda_arr.shape[1] != strength_arr.size:
            return np.zeros(0, dtype=np.int64)
        unit_counts = np.divide(
            lambda_arr,
            np.maximum(strength_arr[None, :], 1.0e-12),
            out=np.zeros_like(lambda_arr, dtype=float),
            where=strength_arr[None, :] > 0.0,
        )
        return self._weak_source_observable_counts_from_unit_response(
            unit_counts,
            signal_counts=signal_counts,
            observation_variances=observation_variances,
        )

    def _prune_floor_sources_after_refit(
        self,
        st: IsotopeState,
        data: MeasurementData,
        *,
        suppress_prune: bool = False,
        record_kill_count: bool = True,
    ) -> None:
        """Prune min-clamped sources using expected counts from fresh data."""
        if suppress_prune:
            return
        if st.num_sources <= 1 or data.z_k.size == 0:
            return
        self._ensure_source_metadata(st)
        lambda_m, lambda_total = self._lambda_components(st, data)
        if lambda_m.size == 0:
            return
        expected_counts = np.sum(np.maximum(lambda_m, 0.0), axis=0)
        drop_allowed = self._source_prune_allowed_mask(
            st,
            data,
            lambda_m=lambda_m,
            lambda_total=lambda_total,
        )
        observable_counts = self._weak_source_observable_counts_from_components(
            lambda_m,
            st.strengths[: st.num_sources],
            signal_counts=np.maximum(
                data.z_k - float(st.background) * data.live_times,
                0.0,
            ),
            observation_variances=data.observation_variances,
        )
        self._prune_floor_sources_by_expected_counts(
            st,
            expected_counts,
            drop_allowed_mask=drop_allowed,
            observable_view_counts=observable_counts,
            record_kill_count=record_kill_count,
        )

    def _prune_floor_sources_by_expected_counts(
        self,
        st: IsotopeState,
        expected_counts: NDArray[np.float64],
        drop_allowed_mask: NDArray[np.bool_] | None = None,
        observable_view_counts: NDArray[np.int64] | None = None,
        *,
        record_kill_count: bool = True,
    ) -> None:
        """Prune min-clamped sources with negligible model support."""
        if st.num_sources <= 1:
            return
        if (
            float(self.config.weak_source_prune_min_expected_count) <= 0.0
            and float(self.config.weak_source_prune_min_fraction) <= 0.0
        ):
            return
        self._ensure_source_metadata(st)
        expected_counts = np.asarray(expected_counts, dtype=float).ravel()[
            : st.num_sources
        ]
        if expected_counts.size != st.num_sources:
            return
        total = float(np.sum(expected_counts))
        fraction = expected_counts / max(total, 1.0e-12)
        min_expected = max(float(self.config.weak_source_prune_min_expected_count), 0.0)
        min_fraction = max(float(self.config.weak_source_prune_min_fraction), 0.0)
        strength_floor = max(float(self.config.min_strength), 0.0)
        at_floor = (
            st.strengths[: st.num_sources] <= strength_floor * (1.0 + 1.0e-6)
        )
        weak_count = (
            expected_counts < min_expected
            if min_expected > 0.0
            else np.zeros(st.num_sources, dtype=bool)
        )
        weak_fraction = (
            fraction < min_fraction
            if min_fraction > 0.0
            else np.zeros(st.num_sources, dtype=bool)
        )
        drop = at_floor & (weak_count | weak_fraction)
        if drop_allowed_mask is not None:
            allowed = np.asarray(drop_allowed_mask, dtype=bool).ravel()[
                : st.num_sources
            ]
            if allowed.size != st.num_sources:
                raise ValueError("drop_allowed_mask must match source count.")
            drop &= allowed
        if bool(self.config.weak_source_prune_require_observable):
            if observable_view_counts is None:
                observable_counts = np.full(st.num_sources, np.iinfo(np.int64).max)
            else:
                observable_counts = np.asarray(
                    observable_view_counts,
                    dtype=np.int64,
                ).ravel()[: st.num_sources]
                if observable_counts.size != st.num_sources:
                    raise ValueError("observable_view_counts must match source count.")
            min_observable = max(
                1,
                int(self.config.weak_source_prune_min_observable_measurements),
            )
            not_observable = observable_counts < min_observable
            protected = drop & not_observable
            if np.any(protected):
                self.last_weak_source_prune_occlusion_protected += int(
                    np.count_nonzero(protected)
                )
                drop &= ~not_observable
        if bool(self.config.pseudo_source_verification_enable):
            grace = max(0, int(self.config.pseudo_source_fail_grace_stations))
            protected = (
                np.asarray(st.tentative_sources[: st.num_sources], dtype=bool)
                & (np.asarray(st.ages[: st.num_sources], dtype=int) < grace)
            )
            drop &= ~protected
        min_age = max(0, int(self.config.weak_source_prune_min_age))
        if min_age > 0:
            ages = np.asarray(st.ages[: st.num_sources], dtype=int)
            if ages.size != st.num_sources:
                raise ValueError("source ages must match source count.")
            drop &= ages >= min_age
        if not np.any(drop):
            return
        if np.count_nonzero(~drop) == 0:
            keep_idx = int(np.argmax(expected_counts))
            drop[keep_idx] = False
        if not np.any(drop):
            return
        keep = ~drop
        if record_kill_count:
            self.last_kill_count += int(np.count_nonzero(drop))
        for idx in np.flatnonzero(drop):
            observable_count = None
            if observable_view_counts is not None:
                observable_arr = np.asarray(observable_view_counts, dtype=np.int64)
                if int(idx) < observable_arr.size:
                    observable_count = int(observable_arr[int(idx)])
            self._record_source_event(
                "source_removed",
                st,
                int(idx),
                reason="weak_floor_prune",
                extra={
                    "expected_count": float(expected_counts[int(idx)]),
                    "expected_fraction": float(fraction[int(idx)]),
                    "min_expected_count": float(min_expected),
                    "min_expected_fraction": float(min_fraction),
                    "observable_measurements": observable_count,
                    "record_kill_count": bool(record_kill_count),
                },
            )
        st.positions = st.positions[keep]
        st.strengths = st.strengths[keep]
        st.ages = st.ages[keep]
        st.low_q_streaks = st.low_q_streaks[keep]
        st.support_scores = st.support_scores[keep]
        st.tentative_sources = st.tentative_sources[keep]
        st.verification_fail_streaks = st.verification_fail_streaks[keep]
        st.num_sources = st.positions.shape[0]

    def _replace_particle_state_from_trial(
        self,
        target: IsotopeState,
        trial: IsotopeState,
    ) -> None:
        """Replace a particle state from an accepted structural proposal."""
        self._ensure_source_metadata(trial)
        target.positions = np.asarray(trial.positions, dtype=float).copy()
        target.strengths = np.asarray(trial.strengths, dtype=float).copy()
        target.background = float(trial.background)
        target.ages = np.asarray(trial.ages, dtype=int).copy()
        target.low_q_streaks = np.asarray(trial.low_q_streaks, dtype=int).copy()
        target.support_scores = np.asarray(trial.support_scores, dtype=float).copy()
        target.tentative_sources = np.asarray(
            trial.tentative_sources,
            dtype=bool,
        ).copy()
        target.verification_fail_streaks = np.asarray(
            trial.verification_fail_streaks,
            dtype=int,
        ).copy()
        target.num_sources = int(target.positions.shape[0])

    def sync_particles_to_evidence_sources(
        self,
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
        data: MeasurementData | None = None,
    ) -> int:
        """
        Project all particles onto authoritative sparse-evidence source slots.

        The sparse evidence layer is the model-order authority.  Once it has a
        decisive cardinality, PF particles keep their role as continuous
        trackers by adopting the evidence-selected source slots instead of
        running birth/death/split/merge proposals that can reintroduce rejected
        cardinalities.
        """
        pos_arr = np.asarray(positions, dtype=float).reshape(-1, 3)
        q_arr = np.maximum(
            np.asarray(strengths, dtype=float).reshape(-1),
            0.0,
        )
        if pos_arr.shape[0] != q_arr.size:
            raise ValueError("evidence positions and strengths must have matching rows.")
        target_count = int(pos_arr.shape[0])
        changed = 0
        for particle in self.continuous_particles:
            st = particle.state
            current_count = int(st.num_sources)
            current_pos = np.asarray(st.positions[:current_count], dtype=float).reshape(
                -1,
                3,
            )
            current_q = np.asarray(st.strengths[:current_count], dtype=float).reshape(-1)
            same_state = (
                current_count == target_count
                and current_pos.shape == pos_arr.shape
                and current_q.shape == q_arr.shape
                and np.allclose(current_pos, pos_arr, rtol=0.0, atol=1.0e-12)
                and np.allclose(current_q, q_arr, rtol=1.0e-12, atol=1.0e-12)
            )
            if not same_state:
                changed += 1
            st.positions = pos_arr.copy()
            st.strengths = q_arr.copy()
            st.num_sources = target_count
            st.covariances = None
            st.ages = np.zeros(target_count, dtype=int)
            st.low_q_streaks = np.zeros(target_count, dtype=int)
            st.support_scores = np.zeros(target_count, dtype=float)
            st.tentative_sources = np.zeros(target_count, dtype=bool)
            st.verification_fail_streaks = np.zeros(target_count, dtype=int)
        if data is not None and data.z_k.size > 0:
            self.refresh_weights_from_measurements(data)
        self.align_continuous_labels()
        self.last_structural_timing_s = {
            "total": 0.0,
            "sparse_evidence_particle_sync": 1.0,
            "sparse_evidence_synced_particles": float(changed),
            "sparse_evidence_selected_count": float(target_count),
        }
        return int(changed)

    def _trial_log_likelihood(
        self,
        st: IsotopeState,
        data: MeasurementData,
    ) -> float:
        """Return the configured count log-likelihood for one fixed state."""
        _, lambda_total = self._lambda_components(st, data)
        return self._count_log_likelihood_np(
            data.z_k,
            lambda_total,
            observation_count_variance=data.observation_variances,
        )

    def _trial_log_likelihood_from_lambda(
        self,
        data: MeasurementData,
        lambda_total: NDArray[np.float64],
    ) -> float:
        """Return the configured count log-likelihood for precomputed counts."""
        return self._count_log_likelihood_np(
            data.z_k,
            np.asarray(lambda_total, dtype=float),
            observation_count_variance=data.observation_variances,
        )

    def _unit_response_counts_for_state(
        self,
        st: IsotopeState,
        data: MeasurementData,
    ) -> NDArray[np.float64]:
        """Return unit-strength response columns for every source in a state."""
        if st.num_sources <= 0 or data.z_k.size == 0:
            return np.zeros((int(data.z_k.size), 0), dtype=float)
        self._ensure_source_metadata(st)
        positions = np.asarray(st.positions[: st.num_sources], dtype=float)
        if positions.ndim != 2 or positions.shape[0] == 0:
            return np.zeros((int(data.z_k.size), 0), dtype=float)
        counts = expected_counts_per_source(
            kernel=self.continuous_kernel,
            isotope=self.isotope,
            detector_positions=data.detector_positions,
            sources=positions,
            strengths=np.ones(positions.shape[0], dtype=float),
            live_times=data.live_times,
            fe_indices=data.fe_indices,
            pb_indices=data.pb_indices,
            source_scale=self._measurement_source_scale_vector(
                data.fe_indices,
                data.pb_indices,
            ),
        )
        return np.asarray(counts, dtype=float)

    def _solve_trial_strengths_from_unit_counts(
        self,
        data: MeasurementData,
        unit_counts: NDArray[np.float64],
        prior_strengths: NDArray[np.float64],
        background: float,
        *,
        iters: int,
        eps: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """
        Refit a trial source set from precomputed unit response columns.

        This is mathematically the same fixed-position strength refit used by
        `_refit_strengths_for_particle`; it only avoids recomputing detector,
        shield, and obstacle response columns for every structural trial.
        """
        counts = np.asarray(unit_counts, dtype=float)
        if counts.ndim != 2:
            if counts.size == 0:
                counts = np.zeros((int(data.z_k.size), 0), dtype=float)
            else:
                raise ValueError("unit_counts must have shape K x S.")
        if counts.shape[0] != int(data.z_k.size):
            raise ValueError("unit_counts must have one row per measurement.")
        source_count = int(counts.shape[1])
        prior = np.asarray(prior_strengths, dtype=float).reshape(-1)
        if prior.size != source_count:
            raise ValueError("prior_strengths must have one value per source.")
        q_min = max(float(self.config.min_strength), 0.0)
        q_max = float(self.config.birth_q_max)
        if q_max < q_min:
            q_min, q_max = q_max, q_min
        background_counts = np.asarray(data.live_times, dtype=float)[:, None] * float(
            background
        )
        strengths, lambda_total = self._solve_strengths_for_kernel_tensor_batched(
            data,
            k_tensor=counts[:, None, :],
            background_counts=background_counts,
            prior_mean=prior[None, :],
            iters=max(1, int(iters)),
            eps=float(eps),
            q_min=q_min,
            q_max=q_max,
        )
        trial_strengths = np.asarray(strengths[0], dtype=float)
        trial_lambda = np.asarray(lambda_total[:, 0], dtype=float)
        ll_after = self._trial_log_likelihood_from_lambda(data, trial_lambda)
        return trial_strengths, trial_lambda, float(ll_after)

    def _orthogonalized_residual_candidate_indices(
        self,
        ranked_candidate_indices: NDArray[np.int64],
        *,
        candidate_counts: NDArray[np.float64],
        existing_response_counts: NDArray[np.float64],
        observation_variances: NDArray[np.float64] | None,
        max_corr: float,
    ) -> NDArray[np.int64]:
        """
        Return residual-birth candidates after response-column orthogonalization.

        Matching-pursuit birth evaluates only a tiny pre-ranked candidate set,
        so the greedy Gram-Schmidt-style loop here is bounded by the configured
        top-k candidate count. The heavy response evaluation and likelihood
        refit remain batched; this helper only prevents multiple nearly
        collinear response columns from entering the same birth proposal set.
        """
        ranked = np.asarray(ranked_candidate_indices, dtype=int).ravel()
        if ranked.size <= 1:
            return ranked.astype(np.int64, copy=False)
        counts = np.asarray(candidate_counts, dtype=float)
        if counts.ndim != 2:
            return ranked.astype(np.int64, copy=False)
        valid_ranked = ranked[(ranked >= 0) & (ranked < counts.shape[1])]
        if valid_ranked.size <= 1:
            return valid_ranked.astype(np.int64, copy=False)
        corr_limit = float(np.clip(float(max_corr), 0.0, 1.0))
        if corr_limit >= 1.0:
            return valid_ranked.astype(np.int64, copy=False)
        if observation_variances is None:
            scale = np.ones(counts.shape[0], dtype=float)
        else:
            variances = np.asarray(observation_variances, dtype=float).reshape(-1)
            if variances.size != counts.shape[0]:
                scale = np.ones(counts.shape[0], dtype=float)
            else:
                scale = 1.0 / np.sqrt(np.maximum(variances, 1.0e-12))

        def _normalized_columns(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
            """Return variance-whitened unit-norm response columns."""
            arr = np.asarray(matrix, dtype=float)
            if arr.ndim != 2 or arr.shape[0] != counts.shape[0] or arr.shape[1] == 0:
                return np.zeros((counts.shape[0], 0), dtype=float)
            whitened = arr * scale[:, None]
            norms = np.linalg.norm(whitened, axis=0)
            keep = norms > 1.0e-12
            if not np.any(keep):
                return np.zeros((counts.shape[0], 0), dtype=float)
            return whitened[:, keep] / norms[keep][None, :]

        basis = _normalized_columns(existing_response_counts)
        candidate_basis = _normalized_columns(counts[:, valid_ranked])
        if candidate_basis.shape[1] != valid_ranked.size:
            return valid_ranked.astype(np.int64, copy=False)
        selected: list[int] = []
        selected_columns: list[NDArray[np.float64]] = []
        for local_idx, candidate_idx in enumerate(valid_ranked.tolist()):
            column = candidate_basis[:, int(local_idx)]
            if basis.size:
                corr_existing = float(np.max(np.abs(basis.T @ column)))
                if corr_existing > corr_limit:
                    continue
            if selected_columns:
                selected_matrix = np.column_stack(selected_columns)
                corr_selected = float(np.max(np.abs(selected_matrix.T @ column)))
                if corr_selected > corr_limit:
                    continue
            selected.append(int(candidate_idx))
            selected_columns.append(column)
        if not selected:
            return valid_ranked[:1].astype(np.int64, copy=False)
        return np.asarray(selected, dtype=np.int64)

    def _best_cached_matching_pursuit_birth_trial_batched(
        self,
        st: IsotopeState,
        data: MeasurementData,
        *,
        candidates: NDArray[np.float64],
        ranked_candidate_indices: NDArray[np.int64],
        q_hat: NDArray[np.float64],
        unit_counts_existing: NDArray[np.float64],
        unit_counts_all: NDArray[np.float64],
        source_strengths: NDArray[np.float64],
        base_ll: float,
    ) -> tuple[IsotopeState | None, float]:
        """Return the best cached matching-pursuit birth trial by batched refit."""
        ranked = np.asarray(ranked_candidate_indices, dtype=int).ravel()
        if ranked.size == 0:
            return None, -np.inf
        candidate_positions = np.asarray(candidates, dtype=float)
        candidate_counts = np.asarray(unit_counts_all, dtype=float)
        existing_counts = np.asarray(unit_counts_existing, dtype=float)
        if candidate_positions.ndim != 2 or candidate_positions.shape[1] != 3:
            return None, -np.inf
        if candidate_counts.shape != (int(data.z_k.size), candidate_positions.shape[0]):
            return None, -np.inf
        if existing_counts.ndim != 2 or existing_counts.shape[0] != int(data.z_k.size):
            return None, -np.inf
        ranked = ranked[(ranked >= 0) & (ranked < candidate_positions.shape[0])]
        if ranked.size == 0:
            return None, -np.inf
        q_min = float(self.config.birth_q_min)
        q_max = float(self.config.birth_q_max)
        if q_max < q_min:
            q_min, q_max = q_max, q_min
        trial_q = np.clip(np.asarray(q_hat, dtype=float).reshape(-1)[ranked], q_min, q_max)
        existing_count = int(existing_counts.shape[1])
        trial_count = existing_count + 1
        trial_num = int(ranked.size)
        k_tensor = np.zeros((int(data.z_k.size), trial_num, trial_count), dtype=float)
        if existing_count > 0:
            k_tensor[:, :, :existing_count] = existing_counts[:, None, :]
        k_tensor[:, :, -1] = candidate_counts[:, ranked]
        background_counts = (
            np.asarray(data.live_times, dtype=float)[:, None] * float(st.background)
        )
        background_counts = np.repeat(background_counts, trial_num, axis=1)
        prior = np.zeros((trial_num, trial_count), dtype=float)
        if existing_count > 0:
            source_prior = np.asarray(source_strengths, dtype=float).reshape(-1)
            if source_prior.size != existing_count:
                raise ValueError("source_strengths must match existing source count.")
            prior[:, :existing_count] = source_prior[None, :]
        prior[:, -1] = trial_q
        strengths, lambda_total = self._solve_strengths_for_kernel_tensor_batched(
            data,
            k_tensor=k_tensor,
            background_counts=background_counts,
            prior_mean=prior,
            iters=max(1, int(self.config.refit_iters)),
            eps=float(self.config.refit_eps),
            q_min=max(float(self.config.min_strength), 0.0),
            q_max=q_max,
        )
        ll_after = self._count_log_likelihood_matrix_np(
            data.z_k,
            lambda_total,
            observation_count_variance=data.observation_variances,
        )
        deltas = np.asarray(ll_after, dtype=float) - float(base_ll)
        finite = np.isfinite(deltas)
        if not np.any(finite):
            return None, -np.inf
        best_local = int(np.flatnonzero(finite)[np.argmax(deltas[finite])])
        best_delta = float(deltas[best_local])
        best_candidate_idx = int(ranked[best_local])
        pos_new = self._project_positions_to_source_prior(
            candidate_positions[best_candidate_idx].reshape(1, 3)
        )[0]
        trial = st.copy()
        self._ensure_source_metadata(trial)
        trial.positions = np.vstack([trial.positions[: trial.num_sources], pos_new])
        trial.strengths = np.asarray(strengths[best_local], dtype=float)
        trial.ages = np.append(trial.ages[: trial.num_sources], 0)
        trial.low_q_streaks = np.append(trial.low_q_streaks[: trial.num_sources], 0)
        trial.support_scores = np.append(trial.support_scores[: trial.num_sources], 0.0)
        trial.tentative_sources = np.append(
            trial.tentative_sources[: trial.num_sources],
            True,
        )
        trial.verification_fail_streaks = np.append(
            trial.verification_fail_streaks[: trial.num_sources],
            0,
        )
        trial.num_sources = int(trial.positions.shape[0])
        return trial, best_delta

    def _structural_acceptance_threshold(
        self,
        *,
        base_threshold: float,
        complexity_penalty: float,
    ) -> float:
        """Return the likelihood-gain threshold for one structural parameter jump."""
        return float(base_threshold) + max(float(complexity_penalty), 0.0)

    def _birth_complexity_penalty(
        self,
        *,
        residual_gate_forced: bool,
        measurement_count: int = 0,
    ) -> float:
        """
        Return the birth complexity penalty after residual-gate correction.

        The station-level residual gate is already a statistical model-order
        test. When that gate forces a birth proposal, the local candidate test
        should not charge the full complexity penalty a second time; it still
        requires the configured base likelihood gain and non-negative local
        improvement.  The BIC model-order term is kept outside that scaling so
        residual-forced births still pay for adding source parameters.
        """
        penalty = max(float(self.config.birth_complexity_penalty), 0.0)
        if residual_gate_forced:
            scale = float(self.config.birth_residual_acceptance_complexity_scale)
            scale = min(max(scale, 0.0), 1.0)
            penalty *= scale
        return penalty + self._bic_model_penalty(
            int(measurement_count),
            int(self.config.birth_bic_penalty_params),
        )

    def _candidate_initial_strengths(
        self,
        *,
        candidate_count: int,
        candidate_kernel_sums: NDArray[np.float64] | None,
        residual_sum: float,
    ) -> NDArray[np.float64]:
        """Return residual-scaled initial strengths for structural candidates."""
        count = max(0, int(candidate_count))
        if count <= 0:
            return np.zeros(0, dtype=float)
        q_min = float(self.config.birth_q_min)
        q_max = float(self.config.birth_q_max)
        if q_max < q_min:
            q_min, q_max = q_max, q_min
        kernel_sums = (
            np.ones(count, dtype=float)
            if candidate_kernel_sums is None
            else np.asarray(candidate_kernel_sums, dtype=float).reshape(-1)[:count]
        )
        if kernel_sums.size != count:
            raise ValueError("candidate_kernel_sums must match candidate_count.")
        denom = np.maximum(kernel_sums, 1.0e-12)
        q = float(self.config.birth_alpha) * max(float(residual_sum), 0.0) / denom
        q = np.clip(q, q_min, q_max)
        return np.where(np.isfinite(q), q, q_min)

    def _best_residual_guided_split_trial(
        self,
        st: IsotopeState,
        data: MeasurementData,
        candidate_positions: NDArray[np.float64] | None,
        candidate_strengths: NDArray[np.float64] | None,
        *,
        suppress_prune_after_refit: bool = False,
        candidate_unit_counts: NDArray[np.float64] | None = None,
    ) -> tuple[IsotopeState | None, float]:
        """
        Return the best residual-guided split trial and its likelihood gain.

        The proposal adds a residual-supported candidate as a split-off
        component from an existing high-strength source, then refits all
        strengths jointly with fixed positions. This is a general RJ-style
        structural move: it is accepted only when the observed count vector is
        better explained by two same-isotope components than by the original
        component.
        """
        if not bool(self.config.split_residual_guided):
            return None, -np.inf
        if data.z_k.size == 0 or st.num_sources <= 0:
            return None, -np.inf
        if (
            self.config.max_sources is not None
            and st.num_sources >= self.config.max_sources
        ):
            return None, -np.inf
        candidates = (
            np.asarray(candidate_positions, dtype=float)
            if candidate_positions is not None
            else np.zeros((0, 3))
        )
        if candidates.ndim != 2 or candidates.shape[1] != 3 or candidates.shape[0] == 0:
            return None, -np.inf
        self._ensure_source_metadata(st)
        eligible = np.flatnonzero(
            (st.strengths[: st.num_sources] >= float(self.config.split_strength_min))
            & (st.ages[: st.num_sources] > int(self.config.min_age_to_split))
        )
        if eligible.size == 0:
            return None, -np.inf
        use_cached_trial_counts = (
            suppress_prune_after_refit and candidate_unit_counts is not None
        )
        existing_unit_counts: NDArray[np.float64] | None = None
        candidate_counts_arr: NDArray[np.float64] | None = None
        if use_cached_trial_counts:
            existing_unit_counts = self._unit_response_counts_for_state(st, data)
            candidate_counts_arr = np.asarray(candidate_unit_counts, dtype=float)
            expected_shape = (int(data.z_k.size), int(candidates.shape[0]))
            if candidate_counts_arr.shape != expected_shape:
                raise ValueError("candidate_unit_counts must have shape K x C.")
            base_lambda = (
                np.asarray(data.live_times, dtype=float) * float(st.background)
                + existing_unit_counts
                @ np.asarray(st.strengths[: st.num_sources], dtype=float)
            )
            base_ll = self._trial_log_likelihood_from_lambda(data, base_lambda)
        else:
            base_ll = self._trial_log_likelihood(st, data)
        if not np.isfinite(base_ll):
            return None, -np.inf
        max_candidates = max(1, int(self.config.split_residual_candidate_count))
        candidate_count = min(max_candidates, candidates.shape[0])
        cand_strengths = self._candidate_initial_strengths(
            candidate_count=candidates.shape[0],
            candidate_kernel_sums=None,
            residual_sum=float(np.sum(np.maximum(data.z_k, 0.0))),
        )
        if candidate_strengths is not None:
            candidate_strengths_arr = np.asarray(candidate_strengths, dtype=float).reshape(-1)
            if candidate_strengths_arr.size:
                copy_count = min(cand_strengths.size, candidate_strengths_arr.size)
                cand_strengths[:copy_count] = candidate_strengths_arr[:copy_count]
        ranked_sources = eligible[np.argsort(st.strengths[eligible])[::-1]]
        best_trial: IsotopeState | None = None
        best_delta = -np.inf
        min_sep = max(float(self.config.birth_min_sep_m), 0.0)
        split_pairs = [
            (int(source_idx), int(cand_idx))
            for cand_idx in range(candidate_count)
            for source_idx in ranked_sources
        ][:max_candidates]
        if (
            use_cached_trial_counts
            and existing_unit_counts is not None
            and candidate_counts_arr is not None
        ):
            return self._best_cached_residual_guided_split_trial_batched(
                st,
                data,
                candidates=candidates,
                candidate_strengths=cand_strengths,
                split_pairs=split_pairs,
                existing_unit_counts=existing_unit_counts,
                candidate_unit_counts=candidate_counts_arr,
                base_ll=base_ll,
                min_sep=min_sep,
            )
        for source_idx, cand_idx in split_pairs:
            pos_new = self._project_positions_to_source_prior(
                candidates[cand_idx].reshape(1, 3)
            )[0]
            if existing_unit_counts is not None and candidate_counts_arr is not None:
                keep_condition = self._birth_response_condition_mask(
                    candidate_counts=candidate_counts_arr[:, [int(cand_idx)]],
                    existing_response_counts=existing_unit_counts,
                    observation_variances=data.observation_variances,
                )
                if not bool(keep_condition[0]):
                    continue
            if st.num_sources > 0:
                dists = np.linalg.norm(
                    st.positions[: st.num_sources] - pos_new[None, :],
                    axis=1,
                )
                dists[int(source_idx)] = np.inf
                if np.any(dists < min_sep):
                    continue
                if np.linalg.norm(st.positions[int(source_idx)] - pos_new) < 0.5 * min_sep:
                    continue
            q_new = max(float(cand_strengths[cand_idx]), float(self.config.min_strength))
            keep_strength = max(
                float(st.strengths[int(source_idx)]) - q_new,
                float(self.config.min_strength),
            )
            if (
                use_cached_trial_counts
                and existing_unit_counts is not None
                and candidate_counts_arr is not None
            ):
                trial_unit_counts = np.column_stack(
                    [
                        existing_unit_counts,
                        candidate_counts_arr[:, int(cand_idx)],
                    ]
                )
                trial_prior = np.concatenate(
                    [
                        np.asarray(st.strengths[: st.num_sources], dtype=float),
                        [q_new],
                    ]
                )
                trial_prior[int(source_idx)] = keep_strength
                trial_strengths, _, ll_after = (
                    self._solve_trial_strengths_from_unit_counts(
                        data,
                        trial_unit_counts,
                        trial_prior,
                        float(st.background),
                        iters=max(1, int(self.config.refit_iters)),
                        eps=float(self.config.refit_eps),
                    )
                )
                trial = st.copy()
                self._ensure_source_metadata(trial)
                trial.positions = np.vstack(
                    [trial.positions[: trial.num_sources], pos_new]
                )
                trial.strengths = trial_strengths
                trial.ages = np.append(trial.ages[: trial.num_sources], 0)
                trial.low_q_streaks = np.append(
                    trial.low_q_streaks[: trial.num_sources],
                    0,
                )
                trial.support_scores = np.append(
                    trial.support_scores[: trial.num_sources],
                    0.0,
                )
                trial.tentative_sources = np.append(
                    trial.tentative_sources[: trial.num_sources],
                    True,
                )
                trial.verification_fail_streaks = np.append(
                    trial.verification_fail_streaks[: trial.num_sources],
                    0,
                )
                trial.num_sources = int(trial.positions.shape[0])
                delta_ll = float(ll_after - base_ll)
                if delta_ll > best_delta:
                    best_delta = delta_ll
                    best_trial = trial
                continue
            trial = st.copy()
            self._ensure_source_metadata(trial)
            trial.strengths[int(source_idx)] = keep_strength
            trial.positions = np.vstack([trial.positions[: trial.num_sources], pos_new])
            trial.strengths = np.append(trial.strengths[: trial.num_sources], q_new)
            trial.ages = np.append(trial.ages[: trial.num_sources], 0)
            trial.low_q_streaks = np.append(trial.low_q_streaks[: trial.num_sources], 0)
            trial.support_scores = np.append(trial.support_scores[: trial.num_sources], 0.0)
            trial.tentative_sources = np.append(
                trial.tentative_sources[: trial.num_sources],
                True,
            )
            trial.verification_fail_streaks = np.append(
                trial.verification_fail_streaks[: trial.num_sources],
                0,
            )
            trial.num_sources = int(trial.positions.shape[0])
            self._refit_strengths_for_particle(
                trial,
                data,
                iters=max(1, int(self.config.refit_iters)),
                eps=float(self.config.refit_eps),
            )
            self._prune_floor_sources_after_refit(
                trial,
                data,
                suppress_prune=suppress_prune_after_refit,
                record_kill_count=False,
            )
            if trial.num_sources <= st.num_sources:
                continue
            ll_after = self._trial_log_likelihood(trial, data)
            delta_ll = float(ll_after - base_ll)
            if delta_ll > best_delta:
                best_delta = delta_ll
                best_trial = trial
        return best_trial, best_delta

    def _best_cached_residual_guided_split_trial_batched(
        self,
        st: IsotopeState,
        data: MeasurementData,
        *,
        candidates: NDArray[np.float64],
        candidate_strengths: NDArray[np.float64],
        split_pairs: list[tuple[int, int]],
        existing_unit_counts: NDArray[np.float64],
        candidate_unit_counts: NDArray[np.float64],
        base_ll: float,
        min_sep: float,
        allow_parallel: bool = True,
    ) -> tuple[IsotopeState | None, float]:
        """Return the best cached residual split trial by batched refit."""
        if not split_pairs or data.z_k.size == 0:
            return None, -np.inf
        worker_count = (
            self._structural_trial_worker_count(len(split_pairs))
            if allow_parallel
            else 1
        )
        if worker_count > 1:
            chunks = self._chunk_sequence(split_pairs, worker_count)
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                results = list(
                    executor.map(
                        lambda chunk: self._best_cached_residual_guided_split_trial_batched(
                            st,
                            data,
                            candidates=candidates,
                            candidate_strengths=candidate_strengths,
                            split_pairs=chunk,
                            existing_unit_counts=existing_unit_counts,
                            candidate_unit_counts=candidate_unit_counts,
                            base_ll=base_ll,
                            min_sep=min_sep,
                            allow_parallel=False,
                        ),
                        chunks,
                    )
                )
            best_trial: IsotopeState | None = None
            best_delta = -np.inf
            for trial, delta in results:
                if delta > best_delta:
                    best_delta = float(delta)
                    best_trial = trial
            return best_trial, best_delta
        self._ensure_source_metadata(st)
        candidates_arr = np.asarray(candidates, dtype=float)
        cand_strengths = np.asarray(candidate_strengths, dtype=float).reshape(-1)
        existing_counts = np.asarray(existing_unit_counts, dtype=float)
        candidate_counts = np.asarray(candidate_unit_counts, dtype=float)
        if existing_counts.shape != (int(data.z_k.size), int(st.num_sources)):
            return None, -np.inf
        if candidate_counts.shape[0] != int(data.z_k.size):
            return None, -np.inf
        condition_mask = self._birth_response_condition_mask(
            candidate_counts=candidate_counts,
            existing_response_counts=existing_counts,
            observation_variances=data.observation_variances,
        )
        valid_pairs: list[tuple[int, int, NDArray[np.float64], float, float]] = []
        for source_idx, cand_idx in split_pairs:
            cand_i = int(cand_idx)
            src_i = int(source_idx)
            if cand_i < 0 or cand_i >= candidates_arr.shape[0]:
                continue
            if src_i < 0 or src_i >= int(st.num_sources):
                continue
            if condition_mask.size and not bool(condition_mask[cand_i]):
                continue
            pos_new = self._project_positions_to_source_prior(
                candidates_arr[cand_i].reshape(1, 3)
            )[0]
            if st.num_sources > 0:
                dists = np.linalg.norm(
                    st.positions[: st.num_sources] - pos_new[None, :],
                    axis=1,
                )
                dists[src_i] = np.inf
                if np.any(dists < min_sep):
                    continue
                if np.linalg.norm(st.positions[src_i] - pos_new) < 0.5 * min_sep:
                    continue
            q_new = max(float(cand_strengths[cand_i]), float(self.config.min_strength))
            keep_strength = max(
                float(st.strengths[src_i]) - q_new,
                float(self.config.min_strength),
            )
            valid_pairs.append((src_i, cand_i, pos_new, q_new, keep_strength))
        if not valid_pairs:
            return None, -np.inf
        trial_count = int(len(valid_pairs))
        source_count = int(st.num_sources) + 1
        k_tensor = np.zeros((int(data.z_k.size), trial_count, source_count), dtype=float)
        k_tensor[:, :, : int(st.num_sources)] = existing_counts[:, None, :]
        prior = np.zeros((trial_count, source_count), dtype=float)
        base_strengths = np.asarray(st.strengths[: st.num_sources], dtype=float)
        prior[:, : int(st.num_sources)] = base_strengths[None, :]
        for trial_idx, (source_idx, cand_idx, _pos_new, q_new, keep_strength) in enumerate(
            valid_pairs
        ):
            k_tensor[:, trial_idx, -1] = candidate_counts[:, cand_idx]
            prior[trial_idx, int(source_idx)] = keep_strength
            prior[trial_idx, -1] = q_new
        q_min = max(float(self.config.min_strength), 0.0)
        q_max = float(self.config.birth_q_max)
        if q_max < q_min:
            q_min, q_max = q_max, q_min
        background_counts = (
            np.asarray(data.live_times, dtype=float)[:, None] * float(st.background)
        )
        background_counts = np.repeat(background_counts, trial_count, axis=1)
        strengths, lambda_total = self._solve_strengths_for_kernel_tensor_batched(
            data,
            k_tensor=k_tensor,
            background_counts=background_counts,
            prior_mean=prior,
            iters=max(1, int(self.config.refit_iters)),
            eps=float(self.config.refit_eps),
            q_min=q_min,
            q_max=q_max,
        )
        ll_after = self._count_log_likelihood_matrix_np(
            data.z_k,
            lambda_total,
            observation_count_variance=data.observation_variances,
        )
        deltas = np.asarray(ll_after, dtype=float) - float(base_ll)
        finite = np.isfinite(deltas)
        if not np.any(finite):
            return None, -np.inf
        best_local = int(np.flatnonzero(finite)[np.argmax(deltas[finite])])
        best_delta = float(deltas[best_local])
        _source_idx, _cand_idx, pos_new, _q_new, _keep_strength = valid_pairs[best_local]
        trial = st.copy()
        self._ensure_source_metadata(trial)
        trial.positions = np.vstack([trial.positions[: trial.num_sources], pos_new])
        trial.strengths = np.asarray(strengths[best_local], dtype=float)
        trial.ages = np.append(trial.ages[: trial.num_sources], 0)
        trial.low_q_streaks = np.append(trial.low_q_streaks[: trial.num_sources], 0)
        trial.support_scores = np.append(trial.support_scores[: trial.num_sources], 0.0)
        trial.tentative_sources = np.append(
            trial.tentative_sources[: trial.num_sources],
            True,
        )
        trial.verification_fail_streaks = np.append(
            trial.verification_fail_streaks[: trial.num_sources],
            0,
        )
        trial.num_sources = int(trial.positions.shape[0])
        return trial, best_delta

    def _apply_matching_pursuit_births_to_state(
        self,
        st: IsotopeState,
        data: MeasurementData,
        candidate_positions: NDArray[np.float64],
        *,
        max_new_sources: int,
        residual_gate_forced: bool = False,
        candidate_unit_counts: NDArray[np.float64] | None = None,
        global_rescue: bool = False,
    ) -> int:
        """
        Add multiple residual-supported sources by matching pursuit.

        Each iteration recomputes the positive residual for the current state,
        ranks candidate positions by shield-coded response matching, tentatively
        adds one source, refits all strengths with fixed positions, and accepts
        only if the configured ΔLL threshold plus complexity penalty is met.
        """
        max_new = max(0, int(max_new_sources))
        if max_new <= 0 or data.z_k.size == 0:
            return 0
        candidates = np.asarray(candidate_positions, dtype=float)
        if candidates.ndim != 2 or candidates.shape[1] != 3 or candidates.shape[0] == 0:
            return 0
        self._ensure_source_metadata(st)
        accepted = 0
        topk = max(1, int(self.config.birth_matching_pursuit_topk_candidates))
        q_min = float(self.config.birth_q_min)
        q_max = float(self.config.birth_q_max)
        if q_max < q_min:
            q_min, q_max = q_max, q_min
        threshold = self._structural_acceptance_threshold(
            base_threshold=float(self.config.birth_delta_ll_threshold),
            complexity_penalty=self._birth_complexity_penalty(
                residual_gate_forced=residual_gate_forced,
                measurement_count=int(data.z_k.size),
            ),
        )
        if candidate_unit_counts is None:
            unit_counts_all = expected_counts_per_source(
                kernel=self.continuous_kernel,
                isotope=self.isotope,
                detector_positions=data.detector_positions,
                sources=candidates,
                strengths=np.ones(candidates.shape[0], dtype=float),
                live_times=data.live_times,
                fe_indices=data.fe_indices,
                pb_indices=data.pb_indices,
                source_scale=self._measurement_source_scale_vector(
                    data.fe_indices,
                    data.pb_indices,
                ),
            )
        else:
            unit_counts_all = np.asarray(candidate_unit_counts, dtype=float)
            expected_shape = (int(data.z_k.size), int(candidates.shape[0]))
            if unit_counts_all.shape != expected_shape:
                raise ValueError("candidate_unit_counts must have shape K x C.")
        for _ in range(max_new):
            if self.config.max_sources is not None and st.num_sources >= self.config.max_sources:
                break
            if global_rescue:
                self.last_birth_global_rescue_attempts += 1
            if residual_gate_forced and bool(self.config.birth_residual_force_proposal_on_gate):
                self.last_birth_forced_attempts += 1
            unit_counts_existing = self._unit_response_counts_for_state(st, data)
            source_strengths = np.asarray(
                st.strengths[: st.num_sources],
                dtype=float,
            )
            lambda_total = (
                np.asarray(data.live_times, dtype=float) * float(st.background)
                + unit_counts_existing @ source_strengths
            )
            residual = np.maximum(np.asarray(data.z_k, dtype=float) - lambda_total, 0.0)
            if float(np.sum(residual)) <= 0.0 and global_rescue:
                background_counts = (
                    np.asarray(data.live_times, dtype=float) * float(st.background)
                )
                residual = np.maximum(
                    np.asarray(data.z_k, dtype=float) - background_counts,
                    0.0,
                )
            if float(np.sum(residual)) <= 0.0:
                break
            support_min_support = (
                self.config.birth_global_rescue_min_support
                if (
                    bool(global_rescue)
                    and self.config.birth_global_rescue_min_support is not None
                )
                else None
            )
            support_min_distinct_poses = (
                self.config.birth_global_rescue_min_distinct_poses
                if (
                    bool(global_rescue)
                    and self.config.birth_global_rescue_min_distinct_poses is not None
                )
                else None
            )
            support_min_distinct_stations = (
                self.config.birth_global_rescue_min_distinct_stations
                if (
                    bool(global_rescue)
                    and (
                        self.config.birth_global_rescue_min_distinct_stations
                        is not None
                    )
                )
                else None
            )
            support_mask = self._birth_candidate_support_mask(
                candidate_counts=unit_counts_all,
                residual_mix=residual,
                observation_variances=data.observation_variances,
                detector_positions=data.detector_positions,
                fe_indices=data.fe_indices,
                pb_indices=data.pb_indices,
                min_support=support_min_support,
                min_distinct_poses=support_min_distinct_poses,
                min_distinct_stations=support_min_distinct_stations,
            )
            active_mask = self._active_source_mask(st, include_quarantined=False)
            if active_mask.size == st.num_sources and unit_counts_existing.shape[1] == st.num_sources:
                existing_counts = unit_counts_existing[:, active_mask]
            else:
                existing_counts = self._birth_existing_unit_response_counts_for_state(
                    st,
                    data,
                )
            corr_mask = self._birth_existing_response_correlation_mask(
                candidate_counts=unit_counts_all,
                existing_response_counts=existing_counts,
                observation_variances=data.observation_variances,
            )
            condition_mask = self._birth_response_condition_mask(
                candidate_counts=unit_counts_all,
                existing_response_counts=existing_counts,
                observation_variances=data.observation_variances,
            )
            distance_mask = np.ones(candidates.shape[0], dtype=bool)
            if st.num_sources > 0:
                distances = np.linalg.norm(
                    candidates[:, None, :] - st.positions[None, : st.num_sources, :],
                    axis=2,
                )
                distance_mask = (
                    np.min(distances, axis=1) >= float(self.config.birth_min_sep_m)
                )
            keep = support_mask & corr_mask & condition_mask
            keep &= distance_mask
            if (
                not np.any(keep)
                and (residual_gate_forced or global_rescue)
                and bool(self.config.birth_residual_force_proposal_on_gate)
                and bool(self.config.birth_residual_force_relax_candidate_masks)
            ):
                # The station-level residual gate already established that the
                # current model order leaves a structured positive residual.
                # Candidate masks are proposal heuristics, not likelihood
                # terms; if they remove every candidate, relax them in a fixed
                # order and still let the joint Poisson refit decide acceptance.
                relaxed_masks = (
                    support_mask & corr_mask,
                    support_mask & condition_mask,
                    support_mask,
                    corr_mask & condition_mask,
                    condition_mask,
                    np.ones_like(distance_mask, dtype=bool),
                )
                for relaxed in relaxed_masks:
                    keep = np.asarray(relaxed, dtype=bool) & distance_mask
                    if np.any(keep):
                        if residual_gate_forced:
                            self.last_birth_forced_mask_relaxations += 1
                        break
            if not np.any(keep):
                if residual_gate_forced and bool(
                    self.config.birth_residual_force_proposal_on_gate
                ):
                    self.last_birth_forced_no_candidate += 1
                if global_rescue:
                    self.last_birth_global_rescue_rejected += 1
                break
            scores, q_hat = self._birth_residual_candidate_scores(
                candidate_counts=unit_counts_all,
                residual_mix=residual,
                observation_variances=data.observation_variances,
            )
            valid = keep & np.isfinite(scores) & np.isfinite(q_hat) & (scores > 0.0) & (q_hat > 0.0)
            if not np.any(valid):
                if global_rescue:
                    self.last_birth_global_rescue_rejected += 1
                break
            ranked = np.flatnonzero(valid)
            ranked = ranked[np.argsort(scores[ranked])[::-1][:topk]]
            if bool(self.config.birth_orthogonalize_residual_candidates):
                ranked = self._orthogonalized_residual_candidate_indices(
                    ranked.astype(np.int64, copy=False),
                    candidate_counts=unit_counts_all,
                    existing_response_counts=existing_counts,
                    observation_variances=data.observation_variances,
                    max_corr=float(self.config.birth_orthogonal_candidate_corr_max),
                )
            base_ll = self._trial_log_likelihood_from_lambda(data, lambda_total)
            best_trial: IsotopeState | None = None
            best_delta = -np.inf
            suppress_prune = (
                (residual_gate_forced or global_rescue)
                and bool(self.config.birth_residual_suppress_death)
            )
            if suppress_prune:
                best_trial, best_delta = (
                    self._best_cached_matching_pursuit_birth_trial_batched(
                        st,
                        data,
                        candidates=candidates,
                        ranked_candidate_indices=ranked.astype(int, copy=False),
                        q_hat=q_hat,
                        unit_counts_existing=unit_counts_existing,
                        unit_counts_all=unit_counts_all,
                        source_strengths=source_strengths,
                        base_ll=base_ll,
                    )
                )
            else:
                for cand_idx in ranked:
                    pos_new = self._project_positions_to_source_prior(
                        candidates[int(cand_idx)].reshape(1, 3)
                    )[0]
                    q_new = float(np.clip(q_hat[int(cand_idx)], q_min, q_max))
                    trial = st.copy()
                    self._ensure_source_metadata(trial)
                    trial.positions = np.vstack(
                        [trial.positions[: trial.num_sources], pos_new]
                    )
                    trial.strengths = np.append(
                        trial.strengths[: trial.num_sources],
                        q_new,
                    )
                    trial.ages = np.append(trial.ages[: trial.num_sources], 0)
                    trial.low_q_streaks = np.append(
                        trial.low_q_streaks[: trial.num_sources],
                        0,
                    )
                    trial.support_scores = np.append(
                        trial.support_scores[: trial.num_sources],
                        0.0,
                    )
                    trial.tentative_sources = np.append(
                        trial.tentative_sources[: trial.num_sources],
                        True,
                    )
                    trial.verification_fail_streaks = np.append(
                        trial.verification_fail_streaks[: trial.num_sources],
                        0,
                    )
                    trial.num_sources = int(trial.positions.shape[0])
                    self._refit_strengths_for_particle(
                        trial,
                        data,
                        iters=max(1, int(self.config.refit_iters)),
                        eps=float(self.config.refit_eps),
                    )
                    self._prune_floor_sources_after_refit(
                        trial,
                        data,
                        suppress_prune=(
                            residual_gate_forced
                            and bool(self.config.birth_residual_suppress_death)
                        ),
                        record_kill_count=False,
                    )
                    if trial.num_sources <= st.num_sources:
                        continue
                    delta_ll = float(self._trial_log_likelihood(trial, data) - base_ll)
                    if delta_ll > best_delta:
                        best_delta = delta_ll
                        best_trial = trial
            forced_proposal = (
                residual_gate_forced
                and bool(self.config.birth_residual_force_proposal_on_gate)
                and np.isfinite(best_delta)
                and best_delta >= float(self.config.birth_residual_forced_min_delta_ll)
            )
            if global_rescue and np.isfinite(best_delta):
                self.last_birth_global_rescue_best_delta = max(
                    float(self.last_birth_global_rescue_best_delta),
                    float(best_delta),
                )
            global_forced_proposal = (
                global_rescue
                and bool(self.config.birth_global_rescue_force_proposal_on_gate)
                and np.isfinite(best_delta)
                and best_delta
                >= float(self.config.birth_global_rescue_forced_min_delta_ll)
            )
            forced_proposal = bool(forced_proposal or global_forced_proposal)
            if residual_gate_forced and bool(self.config.birth_residual_force_proposal_on_gate):
                if np.isfinite(best_delta):
                    self.last_birth_forced_best_delta = max(
                        float(self.last_birth_forced_best_delta),
                        float(best_delta),
                    )
                if best_trial is None or not np.isfinite(best_delta):
                    self.last_birth_forced_no_candidate += 1
                elif best_delta < threshold and not forced_proposal:
                    self.last_birth_forced_rejected += 1
            if (
                global_rescue
                and best_trial is not None
                and np.isfinite(best_delta)
                and best_delta < threshold
                and not forced_proposal
            ):
                self.last_birth_global_rescue_rejected += 1
            if (
                best_trial is None
                or not np.isfinite(best_delta)
                or (best_delta < threshold and not forced_proposal)
            ):
                if global_rescue and (
                    best_trial is None or not np.isfinite(best_delta)
                ):
                    self.last_birth_global_rescue_rejected += 1
                break
            old_count = int(st.num_sources)
            self._stage_new_birth_metadata(
                best_trial,
                data,
                old_count=old_count,
                delta_ll=best_delta,
            )
            for idx in range(old_count, int(best_trial.num_sources)):
                self._record_source_event(
                    "source_birth_accepted",
                    best_trial,
                    int(idx),
                    reason=(
                        "global_mle_rescue_birth"
                        if global_rescue
                        else "matching_pursuit_birth"
                    ),
                    extra={
                        "delta_ll": float(best_delta),
                        "forced_proposal": bool(forced_proposal),
                    },
                )
            self._replace_particle_state_from_trial(st, best_trial)
            accepted += 1
            if forced_proposal and residual_gate_forced:
                self.last_birth_forced_accepts += 1
            if global_rescue:
                self.last_birth_global_rescue_accepts += 1
        return accepted

    def _stage_new_birth_metadata(
        self,
        trial: IsotopeState,
        data: MeasurementData,
        *,
        old_count: int,
        delta_ll: float,
    ) -> None:
        """Mark single-station births as tentative and boost their support score."""
        self._ensure_source_metadata(trial)
        new_start = max(0, int(old_count))
        if int(trial.num_sources) <= new_start:
            return
        support_value = max(float(delta_ll), 0.0)
        if trial.support_scores.size < int(trial.num_sources):
            trial.support_scores = self._resize_metadata_array(
                trial.support_scores,
                int(trial.num_sources),
                0.0,
                float,
            )
        trial.support_scores[new_start : int(trial.num_sources)] = np.maximum(
            trial.support_scores[new_start : int(trial.num_sources)],
            support_value,
        )
        if not bool(
            getattr(self.config, "birth_stage_single_station_as_quarantine", True)
        ):
            return
        support = np.ones(int(data.z_k.size), dtype=bool)
        station_count = self._distinct_supported_station_count(
            data.detector_positions,
            support,
        )
        if int(station_count) > 1:
            return
        if trial.verification_fail_streaks.size < int(trial.num_sources):
            trial.verification_fail_streaks = self._resize_metadata_array(
                trial.verification_fail_streaks,
                int(trial.num_sources),
                0,
                int,
            )
        trial.tentative_sources[new_start : int(trial.num_sources)] = True
        trial.verification_fail_streaks[
            new_start : int(trial.num_sources)
        ] = np.maximum(
            trial.verification_fail_streaks[new_start : int(trial.num_sources)],
            1,
        )

    def _birth_existing_unit_response_counts_for_state(
        self,
        st: IsotopeState,
        data: MeasurementData,
    ) -> NDArray[np.float64]:
        """Return unit-strength response columns for one particle state."""
        if st.num_sources <= 0:
            return np.zeros((data.z_k.size, 0), dtype=float)
        active_mask = self._active_source_mask(st, include_quarantined=True)
        if not np.any(active_mask):
            return np.zeros((data.z_k.size, 0), dtype=float)
        positions = st.positions[: st.num_sources][active_mask]
        counts = expected_counts_per_source(
            kernel=self.continuous_kernel,
            isotope=self.isotope,
            detector_positions=data.detector_positions,
            sources=positions,
            strengths=np.ones(positions.shape[0], dtype=float),
            live_times=data.live_times,
            fe_indices=data.fe_indices,
            pb_indices=data.pb_indices,
            source_scale=self._measurement_source_scale_vector(
                data.fe_indices,
                data.pb_indices,
            ),
        )
        return np.asarray(counts, dtype=float)

    def _verify_pseudo_sources_for_state(
        self,
        st: IsotopeState,
        data: MeasurementData,
        *,
        suppress_prune: bool = False,
        cached_lambda_m: NDArray[np.float64] | None = None,
        cached_lambda_total: NDArray[np.float64] | None = None,
        cached_delta_ll: NDArray[np.float64] | None = None,
        cached_prune_allowed: NDArray[np.bool_] | None = None,
    ) -> bool:
        """
        Verify tentative birth sources with leave-one-out likelihood support.

        Tentative sources are kept only when the original observation block
        supports them across enough independent shield views.  This implements
        a Bai-style pseudo-source verification gate without altering transport
        or spectrum-derived counts.
        """
        if not bool(self.config.pseudo_source_verification_enable):
            return False
        if st.num_sources <= 0 or data.z_k.size == 0:
            return False
        self._ensure_source_metadata(st)
        tentative = np.asarray(st.tentative_sources[: st.num_sources], dtype=bool)
        if not np.any(tentative):
            return False
        if (
            cached_lambda_m is not None
            and cached_lambda_total is not None
            and cached_delta_ll is not None
            and cached_lambda_m.shape == (int(data.z_k.size), int(st.num_sources))
            and cached_lambda_total.shape == (int(data.z_k.size),)
            and cached_delta_ll.shape == (int(st.num_sources),)
        ):
            lambda_m = cached_lambda_m
            lambda_total = cached_lambda_total
            delta_ll = cached_delta_ll
        else:
            lambda_m, lambda_total = self._lambda_components(st, data)
            delta_ll = self._delta_log_likelihood_remove(
                data.z_k,
                lambda_total,
                lambda_m,
                observation_count_variance=data.observation_variances,
            )
        if lambda_m.shape[1] != st.num_sources:
            return False
        variances = self._measurement_vector(
            data.observation_variances,
            data.z_k.size,
            "observation_variances",
            min_value=1.0e-12,
        )
        sigma = np.sqrt(variances)
        min_delta = float(self.config.pseudo_source_min_delta_ll)
        min_views = max(1, int(self.config.pseudo_source_min_distinct_views))
        grace = self._pseudo_source_fail_grace()
        corr_max = float(np.clip(float(self.config.pseudo_source_corr_max), 0.0, 1.0))
        temporal_sep_min = max(
            0.0,
            float(getattr(self.config, "pseudo_source_temporal_sep_min", 0.0)),
        )
        keep = np.ones(st.num_sources, dtype=bool)
        changed = False
        prune_allowed: NDArray[np.bool_] | None = None
        quarantined_before = self._quarantined_source_mask(st)
        for source_idx in range(st.num_sources):
            if not bool(tentative[source_idx]):
                continue
            component = np.maximum(lambda_m[:, source_idx], 0.0)
            support_mask = (
                component / np.maximum(sigma, 1.0e-12)
                >= max(float(self.config.birth_residual_support_sigma), 0.0)
            )
            distinct_views = self._distinct_supported_view_count(
                data.detector_positions,
                data.fe_indices,
                data.pb_indices,
                support_mask,
            )
            response_supported = (
                float(delta_ll[source_idx]) >= min_delta
                and int(distinct_views) >= min_views
            )
            corr_failed = False
            if st.num_sources > 1 and corr_max < 1.0:
                stronger = [
                    idx
                    for idx in range(st.num_sources)
                    if idx != source_idx
                    and float(st.strengths[idx]) >= float(st.strengths[source_idx])
                ]
                if stronger:
                    correlations = [
                        self._response_correlation(component, lambda_m[:, idx])
                        for idx in stronger
                    ]
                    corr_failed = max(correlations, default=0.0) >= corr_max
                    if corr_failed and temporal_sep_min > 0.0:
                        separations = [
                            self._temporal_response_separation(
                                component,
                                lambda_m[:, idx],
                                sigma,
                            )
                            for idx in stronger
                        ]
                        if max(separations, default=0.0) >= temporal_sep_min:
                            corr_failed = False
            if response_supported and not corr_failed:
                self._record_source_event(
                    "pseudo_source_verified",
                    st,
                    source_idx,
                    reason="delta_ll_and_distinct_views_supported",
                    extra={
                        "delta_ll": float(delta_ll[source_idx]),
                        "distinct_views": int(distinct_views),
                        "min_delta_ll": float(min_delta),
                        "min_distinct_views": int(min_views),
                    },
                )
                st.tentative_sources[source_idx] = False
                st.verification_fail_streaks[source_idx] = 0
                self.last_pseudo_source_verified += 1
                changed = True
                continue
            was_quarantined = bool(quarantined_before[source_idx])
            self.last_pseudo_source_failed += 1
            fail_reasons: list[str] = []
            if float(delta_ll[source_idx]) < min_delta:
                fail_reasons.append("insufficient_delta_ll")
            if int(distinct_views) < min_views:
                fail_reasons.append("insufficient_distinct_views")
            if corr_failed:
                fail_reasons.append("high_response_corr")
                if temporal_sep_min > 0.0:
                    fail_reasons.append("insufficient_temporal_separation")
            if float(np.sum(component)) <= 0.0:
                fail_reasons.append("low_expected_contribution")
            if int(st.verification_fail_streaks[source_idx]) < grace:
                fail_reasons.append("too_young_to_prune")
            if not fail_reasons:
                fail_reasons.append("unsupported")
            for reason in fail_reasons:
                self.last_pseudo_source_fail_reasons[reason] = (
                    int(self.last_pseudo_source_fail_reasons.get(reason, 0)) + 1
                )
            observation_limited = int(distinct_views) < min_views or bool(corr_failed)
            if observation_limited:
                self.last_pseudo_source_fail_reasons[
                    "needs_discriminative_views"
                ] = (
                    int(
                        self.last_pseudo_source_fail_reasons.get(
                            "needs_discriminative_views",
                            0,
                        )
                    )
                    + 1
                )
                continue
            st.verification_fail_streaks[source_idx] += 1
            if int(st.verification_fail_streaks[source_idx]) < grace:
                self.last_pseudo_source_fail_reasons["too_young_to_prune"] = (
                    int(
                        self.last_pseudo_source_fail_reasons.get(
                            "too_young_to_prune",
                            0,
                        )
                    )
                    + 1
                )
            if (
                st.num_sources > 1
                and int(st.verification_fail_streaks[source_idx]) >= grace
            ):
                quarantine_enabled = bool(self.config.pseudo_source_quarantine_on_suppress)
                if (
                    not suppress_prune
                    and (
                        was_quarantined
                        or not quarantine_enabled
                    )
                ):
                    if prune_allowed is None:
                        if (
                            cached_prune_allowed is not None
                            and cached_prune_allowed.shape == (int(st.num_sources),)
                        ):
                            prune_allowed = np.asarray(cached_prune_allowed, dtype=bool)
                        else:
                            prune_allowed = self._source_prune_allowed_mask(
                                st,
                                data,
                                lambda_m=lambda_m,
                                lambda_total=lambda_total,
                                delta_ll=delta_ll,
                            )
                prune_now = (
                    not suppress_prune
                    and prune_allowed is not None
                    and bool(prune_allowed[source_idx])
                    and (was_quarantined or not quarantine_enabled)
                )
                if quarantine_enabled and not was_quarantined:
                    self._record_source_event(
                        "pseudo_source_quarantined",
                        st,
                        source_idx,
                        reason=";".join(fail_reasons),
                        extra={
                            "delta_ll": float(delta_ll[source_idx]),
                            "distinct_views": int(distinct_views),
                            "min_delta_ll": float(min_delta),
                            "min_distinct_views": int(min_views),
                            "suppress_prune": bool(suppress_prune),
                        },
                    )
                    self.last_pseudo_source_quarantined += 1
                    changed = True
                elif prune_now:
                    keep[source_idx] = False
        quarantine_mask_after = self._quarantined_source_mask(st)
        self.last_pseudo_source_quarantine_active += int(
            np.count_nonzero(quarantine_mask_after)
        )
        if np.all(keep):
            return changed
        if np.count_nonzero(keep) == 0:
            strongest = int(np.argmax(st.strengths[: st.num_sources]))
            keep[strongest] = True
        pruned = int(np.count_nonzero(~keep))
        if pruned <= 0:
            return changed
        for idx in np.flatnonzero(~keep):
            self._record_source_event(
                "source_removed",
                st,
                int(idx),
                reason="pseudo_source_pruned",
                extra={"suppress_prune": bool(suppress_prune)},
            )
        st.positions = st.positions[keep]
        st.strengths = st.strengths[keep]
        st.ages = st.ages[keep]
        st.low_q_streaks = st.low_q_streaks[keep]
        st.support_scores = st.support_scores[keep]
        st.tentative_sources = st.tentative_sources[keep]
        st.verification_fail_streaks = st.verification_fail_streaks[keep]
        st.num_sources = st.positions.shape[0]
        self.last_pseudo_source_pruned += pruned
        self.last_kill_count += pruned
        return True

    @staticmethod
    def _response_correlation(
        first: NDArray[np.float64],
        second: NDArray[np.float64],
    ) -> float:
        """Return non-negative response-pattern correlation for two sources."""
        a = np.asarray(first, dtype=float).reshape(-1)
        b = np.asarray(second, dtype=float).reshape(-1)
        if a.size == 0 or b.size == 0 or a.size != b.size:
            return 0.0
        a_norm = float(np.linalg.norm(a))
        b_norm = float(np.linalg.norm(b))
        if a_norm <= 0.0 or b_norm <= 0.0:
            return 0.0
        return float(np.dot(a, b) / max(a_norm * b_norm, 1.0e-12))

    @staticmethod
    def _temporal_response_separation(
        first: NDArray[np.float64],
        second: NDArray[np.float64],
        sigma: NDArray[np.float64],
    ) -> float:
        """Return whitened temporal-code separation between two source responses."""
        a = np.asarray(first, dtype=float).reshape(-1)
        b = np.asarray(second, dtype=float).reshape(-1)
        s = np.asarray(sigma, dtype=float).reshape(-1)
        if a.size == 0 or a.size != b.size or a.size != s.size:
            return 0.0
        denom = np.maximum(s, 1.0e-12)
        diff = (a - b) / denom
        value = float(np.sum(diff * diff))
        return value if np.isfinite(value) else 0.0

    def _structural_trial_worker_count(self, trial_count: int) -> int:
        """Return worker count for deterministic structural trial chunks."""
        count = max(0, int(trial_count))
        if count <= 1:
            return 1
        min_trials = max(1, int(self.config.structural_trial_parallel_min_trials))
        if count < min_trials:
            return 1
        workers = max(1, int(self.config.structural_trial_workers))
        return min(count, workers)

    @staticmethod
    def _chunk_sequence(
        values: list[Any],
        worker_count: int,
    ) -> list[list[Any]]:
        """Split values into non-empty ordered chunks for deterministic workers."""
        workers = max(1, int(worker_count))
        if workers <= 1 or len(values) <= 1:
            return [values]
        chunks: list[list[Any]] = []
        for index_array in np.array_split(np.arange(len(values)), workers):
            if index_array.size == 0:
                continue
            chunks.append([values[int(idx)] for idx in index_array])
        return chunks

    def _best_merge_trial(
        self,
        st: IsotopeState,
        data: MeasurementData,
    ) -> tuple[IsotopeState | None, float]:
        """
        Return the best likelihood-tested merge trial and its likelihood gain.

        Candidate pairs are selected either by spatial proximity or by nearly
        collinear response signatures over the actual measurement block. The
        merged state is accepted only if the joint refit does not reduce the
        configured count likelihood beyond ``merge_delta_ll_threshold``.
        """
        if data.z_k.size == 0 or st.num_sources < 2:
            return None, -np.inf
        self._ensure_source_metadata(st)
        lambda_m, _ = self._lambda_components(st, data)
        if lambda_m.shape[1] < 2:
            return None, -np.inf
        base_ll = self._trial_log_likelihood(st, data)
        if not np.isfinite(base_ll):
            return None, -np.inf
        pair_scores = self._merge_candidate_pair_scores(st, lambda_m)
        if not pair_scores:
            return None, -np.inf
        pair_scores.sort(reverse=True)
        max_pairs = max(1, int(self.config.merge_search_topk_pairs))
        return self._best_merge_trial_batched(
            st,
            data,
            pair_scores=pair_scores[:max_pairs],
            base_ll=base_ll,
        )

    def _best_merge_trial_scalar(
        self,
        st: IsotopeState,
        data: MeasurementData,
    ) -> tuple[IsotopeState | None, float]:
        """Return the best merge trial using the scalar reference path."""
        if data.z_k.size == 0 or st.num_sources < 2:
            return None, -np.inf
        self._ensure_source_metadata(st)
        lambda_m, _ = self._lambda_components(st, data)
        if lambda_m.shape[1] < 2:
            return None, -np.inf
        base_ll = self._trial_log_likelihood(st, data)
        if not np.isfinite(base_ll):
            return None, -np.inf
        pair_scores = self._merge_candidate_pair_scores(st, lambda_m)
        if not pair_scores:
            return None, -np.inf
        pair_scores.sort(reverse=True)
        max_pairs = max(1, int(self.config.merge_search_topk_pairs))
        best_trial: IsotopeState | None = None
        best_delta = -np.inf
        for _, i, j in pair_scores[:max_pairs]:
            trial = self._make_merge_trial_state(st, int(i), int(j))
            self._refit_strengths_for_particle(
                trial,
                data,
                iters=max(1, int(self.config.refit_iters)),
                eps=float(self.config.refit_eps),
            )
            self._prune_floor_sources_after_refit(
                trial,
                data,
                record_kill_count=False,
            )
            ll_after = self._trial_log_likelihood(trial, data)
            delta_ll = float(ll_after - base_ll)
            if delta_ll > best_delta:
                best_delta = delta_ll
                best_trial = trial
        return best_trial, best_delta

    def _merge_candidate_pair_scores(
        self,
        st: IsotopeState,
        lambda_m: NDArray[np.float64],
    ) -> list[tuple[float, int, int]]:
        """Return sorted-eligible merge pair scores before likelihood testing."""
        corr_min = float(self.config.merge_response_corr_min)
        distance_max = max(float(self.config.merge_distance_max), 0.0)
        pair_scores: list[tuple[float, int, int]] = []
        for i in range(st.num_sources):
            for j in range(i + 1, st.num_sources):
                distance = float(np.linalg.norm(st.positions[i] - st.positions[j]))
                corr = self._response_correlation(lambda_m[:, i], lambda_m[:, j])
                close_enough = distance_max > 0.0 and distance <= distance_max
                response_redundant = corr_min > 0.0 and corr >= corr_min
                if not close_enough and not response_redundant:
                    continue
                score = corr - distance / max(distance_max, 1.0)
                pair_scores.append((float(score), i, j))
        return pair_scores

    def _make_merge_trial_state(
        self,
        st: IsotopeState,
        first_idx: int,
        second_idx: int,
    ) -> IsotopeState:
        """Return a merge trial state for one pair without refitting strengths."""
        self._ensure_source_metadata(st)
        i = int(first_idx)
        j = int(second_idx)
        q1 = float(st.strengths[i])
        q2 = float(st.strengths[j])
        if q1 + q2 > 0.0:
            merged_pos = (q1 * st.positions[i] + q2 * st.positions[j]) / (q1 + q2)
        else:
            merged_pos = 0.5 * (st.positions[i] + st.positions[j])
        keep = np.ones(st.num_sources, dtype=bool)
        keep[[i, j]] = False
        return IsotopeState(
            num_sources=int(np.count_nonzero(keep) + 1),
            positions=np.vstack([st.positions[keep], merged_pos]),
            strengths=np.append(st.strengths[keep], q1 + q2),
            background=float(st.background),
            ages=np.append(st.ages[keep], max(int(st.ages[i]), int(st.ages[j]))),
            low_q_streaks=np.append(
                st.low_q_streaks[keep],
                min(int(st.low_q_streaks[i]), int(st.low_q_streaks[j])),
            ),
            support_scores=np.append(
                st.support_scores[keep],
                max(float(st.support_scores[i]), float(st.support_scores[j])),
            ),
            tentative_sources=np.append(
                st.tentative_sources[keep],
                bool(st.tentative_sources[i] or st.tentative_sources[j]),
            ),
            verification_fail_streaks=np.append(
                st.verification_fail_streaks[keep],
                min(
                    int(st.verification_fail_streaks[i]),
                    int(st.verification_fail_streaks[j]),
                ),
            ),
        )

    def _best_merge_trial_batched(
        self,
        st: IsotopeState,
        data: MeasurementData,
        *,
        pair_scores: list[tuple[float, int, int]],
        base_ll: float,
        allow_parallel: bool = True,
    ) -> tuple[IsotopeState | None, float]:
        """Return the best merge trial after batched strength refits."""
        if not pair_scores or data.z_k.size == 0:
            return None, -np.inf
        worker_count = (
            self._structural_trial_worker_count(len(pair_scores))
            if allow_parallel
            else 1
        )
        if worker_count > 1:
            chunks = self._chunk_sequence(pair_scores, worker_count)
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                results = list(
                    executor.map(
                        lambda chunk: self._best_merge_trial_batched(
                            st,
                            data,
                            pair_scores=chunk,
                            base_ll=base_ll,
                            allow_parallel=False,
                        ),
                        chunks,
                    )
                )
            best_trial: IsotopeState | None = None
            best_delta = -np.inf
            for trial, delta in results:
                if delta > best_delta:
                    best_delta = float(delta)
                    best_trial = trial
            return best_trial, best_delta
        trials = [
            self._make_merge_trial_state(st, int(i), int(j))
            for _, i, j in pair_scores
        ]
        if not trials:
            return None, -np.inf
        source_count = int(trials[0].num_sources)
        if source_count <= 0 or any(int(trial.num_sources) != source_count for trial in trials):
            return self._best_merge_trial_scalar(st, data)
        original_unit_counts = self._unit_response_counts_for_state(st, data)
        if original_unit_counts.shape != (int(data.z_k.size), int(st.num_sources)):
            return self._best_merge_trial_scalar(st, data)
        merged_sources = np.vstack([trial.positions[source_count - 1] for trial in trials])
        merged_unit_counts = expected_counts_per_source(
            kernel=self.continuous_kernel,
            isotope=self.isotope,
            detector_positions=data.detector_positions,
            sources=merged_sources,
            strengths=np.ones(merged_sources.shape[0], dtype=float),
            live_times=data.live_times,
            fe_indices=data.fe_indices,
            pb_indices=data.pb_indices,
            source_scale=self._measurement_source_scale_vector(
                data.fe_indices,
                data.pb_indices,
            ),
        )
        trial_count = int(len(trials))
        merged_unit_counts = np.asarray(merged_unit_counts, dtype=float)
        if merged_unit_counts.shape != (int(data.z_k.size), trial_count):
            raise ValueError("merged_unit_counts must have shape K x trial_count.")
        k_tensor = np.zeros((int(data.z_k.size), trial_count, source_count), dtype=float)
        for trial_idx, (_score, i, j) in enumerate(pair_scores):
            keep = np.ones(int(st.num_sources), dtype=bool)
            keep[[int(i), int(j)]] = False
            kept_indices = np.flatnonzero(keep)
            kept_count = int(kept_indices.size)
            if kept_count:
                k_tensor[:, trial_idx, :kept_count] = original_unit_counts[:, kept_indices]
            k_tensor[:, trial_idx, -1] = merged_unit_counts[:, trial_idx]
        q_min = max(float(self.config.min_strength), 0.0)
        q_max = float(self.config.birth_q_max)
        if q_max < q_min:
            q_min, q_max = q_max, q_min
        live_arr = np.asarray(data.live_times, dtype=float)
        backgrounds = np.asarray([float(trial.background) for trial in trials], dtype=float)
        background_counts = live_arr[:, None] * backgrounds[None, :]
        prior = np.vstack(
            [np.asarray(trial.strengths[:source_count], dtype=float) for trial in trials]
        )
        strengths, lambda_total = self._solve_strengths_for_kernel_tensor_batched(
            data,
            k_tensor=k_tensor,
            background_counts=background_counts,
            prior_mean=prior,
            iters=max(1, int(self.config.refit_iters)),
            eps=float(self.config.refit_eps),
            q_min=q_min,
            q_max=q_max,
        )
        lambda_m_after = k_tensor * strengths[None, :, :]
        expected_source_counts = np.sum(np.maximum(lambda_m_after, 0.0), axis=0)
        if bool(self.config.source_prune_refit_after_remove):
            drop_allowed_matrix: NDArray[np.bool_] | None = (
                self._source_prune_refit_after_remove_mask_batched(
                    data,
                    k_tensor=k_tensor,
                    background_counts=background_counts,
                    full_strengths=strengths,
                    full_lambda_total=lambda_total,
                    iters=max(1, int(self.config.refit_iters)),
                    eps=float(self.config.refit_eps),
                    q_min=q_min,
                    q_max=q_max,
                )
            )
        else:
            drop_allowed_matrix = None
        ll_cached = self._count_log_likelihood_matrix_np(
            data.z_k,
            lambda_total,
            observation_count_variance=data.observation_variances,
        )
        best_trial: IsotopeState | None = None
        best_delta = -np.inf
        for trial_idx, trial in enumerate(trials):
            trial.strengths[:source_count] = strengths[trial_idx]
            if drop_allowed_matrix is None:
                drop_allowed = self._source_prune_allowed_mask(
                    trial,
                    data,
                    lambda_m=lambda_m_after[:, trial_idx, :],
                    lambda_total=lambda_total[:, trial_idx],
                )
            else:
                drop_allowed = drop_allowed_matrix[trial_idx]
            before_count = int(trial.num_sources)
            self._prune_floor_sources_by_expected_counts(
                trial,
                expected_source_counts[trial_idx],
                drop_allowed_mask=drop_allowed,
                record_kill_count=False,
            )
            if int(trial.num_sources) == before_count:
                ll_after = float(ll_cached[trial_idx])
            else:
                ll_after = self._trial_log_likelihood(trial, data)
            delta_ll = float(ll_after - base_ll)
            if delta_ll > best_delta:
                best_delta = delta_ll
                best_trial = trial
        return best_trial, best_delta

    def _source_detector_exclusion_mask(
        self,
        st: IsotopeState,
        data: MeasurementData | None,
    ) -> NDArray[np.bool_]:
        """Return source mask enforcing that sources cannot occupy detector poses."""
        if st.num_sources <= 0:
            return np.ones(0, dtype=bool)
        min_sep = max(float(self.config.source_detector_exclusion_m), 0.0)
        if min_sep <= 0.0 or data is None or data.detector_positions.size == 0:
            return np.ones(st.num_sources, dtype=bool)
        det = np.asarray(data.detector_positions, dtype=float)
        if det.ndim != 2 or det.shape[1] != 3:
            return np.ones(st.num_sources, dtype=bool)
        dist = np.linalg.norm(st.positions[:, None, :] - det[None, :, :], axis=2)
        return np.min(dist, axis=1) >= min_sep

    def _select_structural_proposal_indices(
        self,
        limit: int | None,
        *,
        require_birth_capacity: bool = False,
    ) -> set[int] | None:
        """
        Return posterior-diverse particle indices for expensive structural moves.

        Structural birth/split/merge proposals are exact likelihood-tested moves,
        but evaluating them for every particle can dominate runtime without
        changing transport fidelity.  This selector keeps the highest posterior
        particles while reserving quota for each active source cardinality so
        low-probability multi-source hypotheses are not discarded solely by a
        global weight sort.
        """
        total = len(self.continuous_particles)
        if total <= 0:
            return set()
        if limit is None:
            return None
        max_count = int(limit)
        if max_count <= 0:
            return set()
        if require_birth_capacity and self.config.max_sources is not None:
            eligible_indices = [
                int(idx)
                for idx, particle in enumerate(self.continuous_particles)
                if int(particle.state.num_sources) < int(self.config.max_sources)
            ]
        else:
            eligible_indices = list(range(total))
        if not eligible_indices:
            return set()
        if max_count >= len(eligible_indices):
            return None if len(eligible_indices) == total else set(eligible_indices)
        weights = np.asarray(self.continuous_weights, dtype=float)
        if weights.size != total:
            weights = np.ones(total, dtype=float) / float(total)
        finite = np.isfinite(weights)
        if not np.any(finite):
            weights = np.ones(total, dtype=float) / float(total)
        else:
            weights = np.where(finite, weights, 0.0)
        global_quota = max(1, max_count // 2)
        eligible_set = set(eligible_indices)
        order = np.asarray(
            [idx for idx in np.argsort(weights)[::-1] if int(idx) in eligible_set],
            dtype=int,
        )
        if order.size == 0:
            return set()
        selected: set[int] = set(int(idx) for idx in order[:global_quota])
        grouped: dict[int, list[int]] = {}
        for idx in eligible_indices:
            particle = self.continuous_particles[int(idx)]
            grouped.setdefault(int(particle.state.num_sources), []).append(idx)
        if grouped and len(selected) < max_count:
            group_quota = max(1, (max_count - len(selected)) // max(len(grouped), 1))
            for indices in grouped.values():
                ranked = sorted(indices, key=lambda item: float(weights[item]), reverse=True)
                for idx in ranked[:group_quota]:
                    selected.add(int(idx))
                    if len(selected) >= max_count:
                        break
                if len(selected) >= max_count:
                    break
        if len(selected) < max_count:
            for idx in order:
                selected.add(int(idx))
                if len(selected) >= max_count:
                    break
        return selected

    def refresh_weights_from_measurements(
        self,
        data: MeasurementData | None,
        *,
        lambda_total_by_index: dict[int, NDArray[np.float64]] | None = None,
        reference_log_likelihood_by_index: dict[int, float] | None = None,
        moved_indices: set[int] | None = None,
    ) -> None:
        """
        Recompute particle weights from a measurement block after structural moves.

        Birth, death, split, and merge moves change the state dimension.  When
        those moves are proposed after a station-level resampling step, the
        modified particles must be reweighted by the same station likelihood;
        otherwise a proposal can affect the reported posterior without being
        judged by the observation that triggered it.

        ``lambda_total_by_index`` may contain exact expected-count vectors for
        particles whose states were not modified during the structural update.
        Reusing those vectors only avoids duplicate kernel evaluations; it does
        not alter the likelihood or PF update rule.

        When ``reference_log_likelihood_by_index`` and ``moved_indices`` are
        given, only moved particles are corrected by the likelihood ratio
        ``new_window_ll - old_window_ll``.  This preserves all previous
        posterior evidence already accumulated before the structural move.
        """
        if data is None or data.z_k.size == 0 or not self.continuous_particles:
            return
        if reference_log_likelihood_by_index is not None and moved_indices is not None:
            self._refresh_moved_particle_weights_from_measurements(
                data,
                reference_log_likelihood_by_index=reference_log_likelihood_by_index,
                moved_indices=moved_indices,
            )
            return
        log_likelihoods = np.full(len(self.continuous_particles), -np.inf, dtype=float)
        cached_lambda = lambda_total_by_index or {}
        expected_shape = (int(data.z_k.size),)
        grouped, fallback_indices = self._particle_indices_by_source_count()
        for source_count, particle_indices in grouped.items():
            missing_indices: list[int] = []
            cached_indices: list[int] = []
            cached_values: list[NDArray[np.float64]] = []
            for particle_idx in particle_indices:
                cached = cached_lambda.get(int(particle_idx))
                if cached is not None and np.asarray(cached).shape == expected_shape:
                    cached_indices.append(int(particle_idx))
                    cached_values.append(np.asarray(cached, dtype=float))
                else:
                    missing_indices.append(int(particle_idx))
            if cached_indices:
                cached_matrix = np.column_stack(cached_values)
                cached_ll = self._count_log_likelihood_matrix_np(
                    data.z_k,
                    cached_matrix,
                    observation_count_variance=data.observation_variances,
                )
                log_likelihoods[np.asarray(cached_indices, dtype=int)] = cached_ll
            if not missing_indices:
                continue
            _, lambda_total = self._lambda_components_for_particle_group(
                data,
                missing_indices,
                source_count,
            )
            group_ll = self._count_log_likelihood_matrix_np(
                data.z_k,
                lambda_total,
                observation_count_variance=data.observation_variances,
            )
            log_likelihoods[np.asarray(missing_indices, dtype=int)] = group_ll
        for idx in fallback_indices:
            cached = cached_lambda.get(int(idx))
            if cached is not None and np.asarray(cached).shape == expected_shape:
                lambda_total = np.asarray(cached, dtype=float)
            else:
                st = self.continuous_particles[idx].state
                _, lambda_total = self._lambda_components(st, data)
            log_likelihoods[idx] = self._count_log_likelihood_np(
                data.z_k,
                lambda_total,
                observation_count_variance=data.observation_variances,
            )
        norm = logsumexp(log_likelihoods)
        if not np.isfinite(norm):
            uniform = -np.log(max(len(self.continuous_particles), 1))
            for particle in self.continuous_particles:
                particle.log_weight = float(uniform)
            return
        for particle, value in zip(self.continuous_particles, log_likelihoods - norm):
            particle.log_weight = float(value)

    def _refresh_moved_particle_weights_from_measurements(
        self,
        data: MeasurementData,
        *,
        reference_log_likelihood_by_index: dict[int, float],
        moved_indices: set[int],
    ) -> None:
        """Apply station-window likelihood-ratio corrections to moved particles."""
        if data.z_k.size == 0 or not moved_indices:
            return
        valid_indices = [
            int(idx)
            for idx in sorted(moved_indices)
            if 0 <= int(idx) < len(self.continuous_particles)
        ]
        if not valid_indices:
            return
        new_ll = self._window_log_likelihoods_for_indices(data, valid_indices)
        for particle_idx, ll_new in zip(valid_indices, new_ll):
            ll_old = float(reference_log_likelihood_by_index.get(int(particle_idx), np.nan))
            if not np.isfinite(ll_old) or not np.isfinite(ll_new):
                continue
            particle = self.continuous_particles[int(particle_idx)]
            particle.log_weight = float(particle.log_weight + float(ll_new) - ll_old)
        self._normalize_continuous_log_weights()

    def _window_log_likelihoods_for_indices(
        self,
        data: MeasurementData,
        indices: list[int],
    ) -> NDArray[np.float64]:
        """Return measurement-window log likelihoods for selected particles."""
        out = np.full(len(indices), -np.inf, dtype=float)
        if not indices:
            return out
        grouped: dict[int, list[tuple[int, int]]] = {}
        fallback: list[tuple[int, int]] = []
        for out_idx, particle_idx in enumerate(indices):
            st = self.continuous_particles[int(particle_idx)].state
            source_count = int(st.num_sources)
            if source_count > 0:
                grouped.setdefault(source_count, []).append((out_idx, int(particle_idx)))
            else:
                fallback.append((out_idx, int(particle_idx)))
        for source_count, pairs in grouped.items():
            particle_indices = [particle_idx for _, particle_idx in pairs]
            _, lambda_total = self._lambda_components_for_particle_group(
                data,
                particle_indices,
                source_count,
            )
            group_ll = self._count_log_likelihood_matrix_np(
                data.z_k,
                lambda_total,
                observation_count_variance=data.observation_variances,
            )
            for local_idx, (out_idx, _) in enumerate(pairs):
                out[int(out_idx)] = float(group_ll[int(local_idx)])
        for out_idx, particle_idx in fallback:
            st = self.continuous_particles[int(particle_idx)].state
            _, lambda_total = self._lambda_components(st, data)
            out[int(out_idx)] = self._count_log_likelihood_np(
                data.z_k,
                lambda_total,
                observation_count_variance=data.observation_variances,
            )
        return out

    @staticmethod
    def _weighted_quantile(
        values: NDArray[np.float64],
        weights: NDArray[np.float64],
        quantile: float,
    ) -> float:
        """Return a robust weighted quantile for finite one-dimensional samples."""
        vals = np.asarray(values, dtype=float).reshape(-1)
        w = np.asarray(weights, dtype=float).reshape(-1)
        if vals.size == 0:
            return 0.0
        if w.size != vals.size:
            w = np.ones(vals.size, dtype=float)
        finite = np.isfinite(vals) & np.isfinite(w) & (w > 0.0)
        if not np.any(finite):
            return float(np.median(vals[np.isfinite(vals)])) if np.any(np.isfinite(vals)) else 0.0
        vals = vals[finite]
        w = w[finite]
        order = np.argsort(vals)
        vals = vals[order]
        w = w[order]
        cumulative = np.cumsum(w)
        total = float(cumulative[-1])
        if total <= 0.0:
            return float(np.median(vals))
        target = float(np.clip(quantile, 0.0, 1.0)) * total
        idx = int(np.searchsorted(cumulative, target, side="left"))
        idx = min(max(idx, 0), vals.size - 1)
        return float(vals[idx])

    def estimate_clustered(
        self,
        max_k: int | None = None,
        *,
        include_report_excluded: bool = False,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Estimate source positions/strengths by robust posterior clustering.

        Source-existence ordering uses posterior mass, while the reported
        position and strength use weighted medians.  This avoids high-intensity
        posterior tails dominating the displayed estimate without changing the
        PF update itself.
        """
        if not self.continuous_particles:
            return np.zeros((0, 3)), np.zeros(0)
        positions: list[NDArray[np.float64]] = []
        weights: list[float] = []
        strengths: list[float] = []
        cont_weights = np.asarray(self.continuous_weights, dtype=float)
        for p, w in zip(self.continuous_particles, cont_weights):
            st = p.state
            if st.num_sources <= 0:
                continue
            active_mask = (
                self._active_source_mask(st, include_quarantined=True)
                if bool(include_report_excluded)
                else self._report_source_mask(st)
            )
            for pos, q in zip(
                st.positions[: st.num_sources][active_mask],
                st.strengths[: st.num_sources][active_mask],
            ):
                positions.append(np.asarray(pos, dtype=float))
                weights.append(float(w))
                strengths.append(float(q))
        if not positions:
            return np.zeros((0, 3)), np.zeros(0)
        pos_arr = np.vstack(positions)
        w_arr = np.asarray(weights, dtype=float)
        q_arr = np.asarray(strengths, dtype=float)
        pos_arr, w_arr, q_arr = self._downsample_report_points(
            pos_arr,
            w_arr,
            q_arr,
            max_points=int(self.config.cluster_report_max_points),
        )
        eps = float(self.config.cluster_eps_m)
        if eps <= 0.0:
            eps = 1e-6
        min_samples = max(1, int(self.config.cluster_min_samples))
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            return self.estimate()
        tree = cKDTree(pos_arr)
        clusters = self._connected_position_clusters(
            tree,
            point_count=int(pos_arr.shape[0]),
            eps=eps,
            min_samples=min_samples,
            exact_max_points=int(self.config.cluster_exact_max_points),
        )
        if not clusters:
            return np.zeros((0, 3)), np.zeros(0)
        cluster_pos: list[NDArray[np.float64]] = []
        cluster_q: list[float] = []
        cluster_mass: list[float] = []
        strength_floor = max(float(self.config.min_strength), 0.0) * (1.0 + 1.0e-6)
        for members in clusters:
            member_strengths = q_arr[members]
            active = member_strengths > strength_floor
            members_for_summary = members[active] if np.any(active) else members
            w = w_arr[members_for_summary]
            if np.sum(w) <= 0.0:
                w = np.ones_like(w, dtype=float)
            w = w / np.sum(w)
            member_pos = pos_arr[members_for_summary]
            member_q = q_arr[members_for_summary]
            pos_robust = np.array(
                [
                    self._weighted_quantile(member_pos[:, dim], w, 0.5)
                    for dim in range(member_pos.shape[1])
                ],
                dtype=float,
            )
            q_robust = self._weighted_quantile(member_q, w, 0.5)
            cluster_pos.append(pos_robust)
            cluster_q.append(q_robust)
            cluster_mass.append(float(np.sum(w_arr[members_for_summary])))
        order = np.argsort(cluster_mass)[::-1]
        if max_k is None:
            max_k = self.config.max_sources
        if max_k is not None:
            order = order[: max(0, int(max_k))]
        pos_out = np.vstack([cluster_pos[i] for i in order]) if order.size else np.zeros((0, 3))
        q_out = np.array([cluster_q[i] for i in order], dtype=float) if order.size else np.zeros(0, dtype=float)
        return pos_out, q_out

    @staticmethod
    def _downsample_report_points(
        positions: NDArray[np.float64],
        weights: NDArray[np.float64],
        strengths: NDArray[np.float64],
        *,
        max_points: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return a deterministic bounded point set for report-only clustering."""
        n_points = int(positions.shape[0])
        limit = int(max_points)
        if limit <= 0 or n_points <= limit:
            return positions, weights, strengths
        finite_weights = np.asarray(weights, dtype=float)
        finite_weights = np.where(np.isfinite(finite_weights), finite_weights, 0.0)
        if finite_weights.size != n_points or np.allclose(finite_weights, finite_weights[0]):
            idx = np.linspace(0, n_points - 1, num=limit, dtype=np.int64)
        else:
            top_count = max(1, limit // 2)
            top_idx = np.argsort(finite_weights)[::-1][:top_count]
            uniform_count = max(0, limit - top_idx.size)
            uniform_idx = (
                np.linspace(0, n_points - 1, num=uniform_count, dtype=np.int64)
                if uniform_count > 0
                else np.zeros(0, dtype=np.int64)
            )
            idx = np.unique(np.concatenate([top_idx, uniform_idx]))
            if idx.size < limit:
                missing = limit - idx.size
                fill = np.setdiff1d(
                    np.linspace(0, n_points - 1, num=min(n_points, 2 * missing), dtype=np.int64),
                    idx,
                    assume_unique=False,
                )[:missing]
                idx = np.concatenate([idx, fill])
        idx = np.asarray(idx[:limit], dtype=np.int64)
        return positions[idx], weights[idx], strengths[idx]

    @staticmethod
    def _connected_position_clusters(
        tree: Any,
        *,
        point_count: int,
        eps: float,
        min_samples: int,
        exact_max_points: int = 5000,
    ) -> list[NDArray[np.int64]]:
        """Return epsilon-neighborhood connected components for source positions."""
        count = max(0, int(point_count))
        if count <= 0:
            return []
        try:
            data = np.asarray(tree.data, dtype=float)
        except AttributeError:
            data = np.zeros((count, 0), dtype=float)
        if data.shape[0] != count:
            data = np.zeros((count, 0), dtype=float)
        if int(exact_max_points) > 0 and count > int(exact_max_points):
            return IsotopeParticleFilter._grid_position_clusters(
                data,
                eps=eps,
                min_samples=min_samples,
            )
        visited = np.zeros(count, dtype=bool)
        clusters: list[NDArray[np.int64]] = []
        sample_floor = max(1, int(min_samples))
        radius = float(eps)
        for seed in range(count):
            if visited[seed]:
                continue
            visited[seed] = True
            members: list[int] = [int(seed)]
            queue: list[int] = [int(seed)]
            while queue:
                idx = queue.pop()
                if data.size:
                    neighbors = tree.query_ball_point(data[idx], r=radius)
                else:
                    neighbors = [idx]
                if not neighbors:
                    continue
                new_neighbors: list[int] = []
                for neighbor in neighbors:
                    n_idx = int(neighbor)
                    if n_idx < 0 or n_idx >= count or visited[n_idx]:
                        continue
                    visited[n_idx] = True
                    members.append(n_idx)
                    new_neighbors.append(n_idx)
                if len(members) >= count:
                    queue.clear()
                    break
                queue.extend(new_neighbors)
            if len(members) >= sample_floor:
                clusters.append(np.asarray(members, dtype=np.int64))
        return clusters

    @staticmethod
    def _grid_position_clusters(
        data: NDArray[np.float64],
        *,
        eps: float,
        min_samples: int,
    ) -> list[NDArray[np.int64]]:
        """Return scalable report clusters by connected occupied spatial cells."""
        n_points = int(data.shape[0])
        if n_points <= 0:
            return []
        if data.ndim != 2 or data.shape[1] == 0:
            members = np.arange(n_points, dtype=np.int64)
            return [members] if members.size >= max(1, int(min_samples)) else []
        cell_size = max(float(eps), 1.0e-6)
        cells = np.floor(data / cell_size).astype(np.int64, copy=False)
        cell_to_points: dict[tuple[int, ...], list[int]] = {}
        for idx, cell in enumerate(cells):
            key = tuple(int(v) for v in cell)
            cell_to_points.setdefault(key, []).append(int(idx))
        parent: dict[tuple[int, ...], tuple[int, ...]] = {
            key: key for key in cell_to_points
        }

        def find(key: tuple[int, ...]) -> tuple[int, ...]:
            """Find the representative occupied cell."""
            root = key
            while parent[root] != root:
                root = parent[root]
            while parent[key] != key:
                nxt = parent[key]
                parent[key] = root
                key = nxt
            return root

        def union(a: tuple[int, ...], b: tuple[int, ...]) -> None:
            """Union two occupied cells."""
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[rb] = ra

        offsets = np.array(np.meshgrid(*([[-1, 0, 1]] * data.shape[1]), indexing="ij"))
        offsets = offsets.reshape(data.shape[1], -1).T
        occupied = set(cell_to_points)
        for key in list(cell_to_points):
            key_arr = np.asarray(key, dtype=np.int64)
            for offset in offsets:
                if not np.any(offset):
                    continue
                neighbor = tuple(int(v) for v in (key_arr + offset))
                if neighbor in occupied:
                    union(key, neighbor)
        grouped: dict[tuple[int, ...], list[int]] = {}
        for key, members in cell_to_points.items():
            grouped.setdefault(find(key), []).extend(members)
        sample_floor = max(1, int(min_samples))
        return [
            np.asarray(sorted(members), dtype=np.int64)
            for members in grouped.values()
            if len(members) >= sample_floor
        ]

    def apply_birth_death(
        self,
        support_data: MeasurementData | None,
        birth_data: MeasurementData | None,
        candidate_positions: NDArray[np.float64] | None = None,
        global_birth_candidates: NDArray[np.float64] | None = None,
        global_birth_candidate_counts: NDArray[np.float64] | None = None,
        allow_structural_birth_proposals: bool = True,
    ) -> None:
        """
        Apply hysteretic death, residual-driven birth, and split/merge proposals.
        """
        if not self.continuous_particles:
            return
        if not bool(self.config.birth_enable):
            return
        timing: dict[str, float] = {
            "total": 0.0,
            "cache": 0.0,
            "birth": 0.0,
            "prune": 0.0,
            "pseudo": 0.0,
            "split": 0.0,
            "merge": 0.0,
            "refit": 0.0,
            "refresh_weights": 0.0,
            "label": 0.0,
            "report_cluster": 0.0,
        }
        structural_start = time.perf_counter()
        support_data = self._structural_evidence_data(support_data)
        birth_data = self._structural_evidence_data(birth_data)
        structural_data = birth_data if birth_data is not None else support_data
        if structural_data is None or structural_data.z_k.size == 0:
            self._reset_structural_residual_gate()
            return
        proposal_enabled = bool(allow_structural_birth_proposals)
        min_distinct = max(1, int(self.config.birth_min_distinct_poses))
        min_stations = max(1, int(self.config.birth_min_distinct_stations))
        global_candidates = np.zeros((0, 3), dtype=float)
        if proposal_enabled and global_birth_candidates is not None:
            global_candidates = np.asarray(global_birth_candidates, dtype=float).reshape(
                -1,
                3,
            )
        has_global_rescue_candidates = (
            bool(self.config.birth_global_rescue_enable)
            and global_candidates.shape[0] > 0
        )
        structural_min_distinct = min_distinct
        structural_min_stations = min_stations
        if has_global_rescue_candidates:
            if self.config.birth_global_rescue_min_distinct_poses is not None:
                structural_min_distinct = min(
                    structural_min_distinct,
                    max(
                        1,
                        int(self.config.birth_global_rescue_min_distinct_poses),
                    ),
                )
            if self.config.birth_global_rescue_min_distinct_stations is not None:
                structural_min_stations = min(
                    structural_min_stations,
                    max(
                        1,
                        int(self.config.birth_global_rescue_min_distinct_stations),
                    ),
                )
        regular_structural_support_ready = True
        structural_distinct_count: int | None = None
        structural_station_count: int | None = None
        if structural_min_distinct > 1 or structural_min_stations > 1:
            full_support = np.ones(structural_data.z_k.size, dtype=bool)
            distinct_count = self._distinct_supported_view_count(
                structural_data.detector_positions,
                structural_data.fe_indices,
                structural_data.pb_indices,
                full_support,
            )
            station_count = self._distinct_supported_station_count(
                structural_data.detector_positions,
                full_support,
            )
            structural_distinct_count = int(distinct_count)
            structural_station_count = int(station_count)
            self.last_birth_residual_distinct_poses = int(distinct_count)
            self.last_birth_residual_distinct_stations = int(station_count)
            regular_structural_support_ready = (
                distinct_count >= min_distinct and station_count >= min_stations
            )
            if (
                distinct_count < structural_min_distinct
                or station_count < structural_min_stations
            ):
                self._reset_structural_residual_gate()
                self.last_birth_residual_distinct_poses = int(distinct_count)
                self.last_birth_residual_distinct_stations = int(station_count)
                return
        if self.config.max_sources is None:
            birth_capacity_available = True
        else:
            max_sources = int(self.config.max_sources)
            birth_capacity_available = any(
                int(particle.state.num_sources) < max_sources
                for particle in self.continuous_particles
            )
        birth_proposal = None
        if proposal_enabled and birth_capacity_available:
            if bool(regular_structural_support_ready):
                birth_start = time.perf_counter()
                birth_proposal = self._compute_birth_proposal(
                    birth_data,
                    candidate_positions,
                )
                timing["birth"] += time.perf_counter() - birth_start
            else:
                self._reset_structural_residual_gate()
                if structural_distinct_count is not None:
                    self.last_birth_residual_distinct_poses = int(
                        structural_distinct_count
                    )
                if structural_station_count is not None:
                    self.last_birth_residual_distinct_stations = int(
                        structural_station_count
                    )
        if birth_proposal is not None:
            if len(birth_proposal) == 4:
                birth_probs, birth_kernel_sums, residual_sum, birth_candidates = (
                    birth_proposal
                )
                birth_candidate_counts = None
            else:
                (
                    birth_probs,
                    birth_kernel_sums,
                    residual_sum,
                    birth_candidates,
                    birth_candidate_counts,
                ) = birth_proposal
        else:
            birth_probs = None
            birth_kernel_sums = None
            residual_sum = 0.0
            birth_candidates = None
            birth_candidate_counts = None
        residual_birth_gate_active = (
            birth_proposal is not None
            and bool(self.last_birth_residual_gate_passed)
            and bool(self.last_birth_residual_refit_gate_passed)
            and bool(self.config.birth_residual_always_try)
            and residual_sum > 0.0
        )
        split_candidate_strengths = (
            self._candidate_initial_strengths(
                candidate_count=birth_candidates.shape[0],
                candidate_kernel_sums=birth_kernel_sums,
                residual_sum=residual_sum,
            )
            if birth_candidates is not None and birth_kernel_sums is not None
            else None
        )
        proposal_data = None
        if birth_data is not None and birth_data.z_k.size:
            proposal_data = birth_data
        elif support_data is not None and support_data.z_k.size:
            proposal_data = support_data
        refit_data = proposal_data
        if proposal_data is not None and global_candidates.size:
            global_candidates = self._exclude_birth_candidates_near_detectors(
                global_candidates,
                proposal_data,
            )
        global_birth_rescue_active = (
            proposal_enabled
            and
            bool(self.config.birth_global_rescue_enable)
            and proposal_data is not None
            and proposal_data.z_k.size > 0
            and global_candidates.shape[0] > 0
        )
        self.last_birth_global_rescue_candidates = int(global_candidates.shape[0])
        global_candidate_counts_for_trial: NDArray[np.float64] | None = None
        if global_birth_rescue_active and proposal_data is not None:
            if global_birth_candidate_counts is not None:
                counts_arr = np.asarray(global_birth_candidate_counts, dtype=float)
                expected_shape = (
                    int(proposal_data.z_k.size),
                    int(global_candidates.shape[0]),
                )
                if counts_arr.shape == expected_shape:
                    global_candidate_counts_for_trial = counts_arr
            if global_candidate_counts_for_trial is None:
                global_candidate_counts_for_trial = expected_counts_per_source(
                    kernel=self.continuous_kernel,
                    isotope=self.isotope,
                    detector_positions=proposal_data.detector_positions,
                    sources=global_candidates,
                    strengths=np.ones(global_candidates.shape[0], dtype=float),
                    live_times=proposal_data.live_times,
                    fe_indices=proposal_data.fe_indices,
                    pb_indices=proposal_data.pb_indices,
                    source_scale=self._measurement_source_scale_vector(
                        proposal_data.fe_indices,
                        proposal_data.pb_indices,
                    ),
                )
        split_candidates_for_trial = birth_candidates
        split_candidate_counts_for_trial = birth_candidate_counts
        split_candidate_strengths_for_trial = split_candidate_strengths
        if (
            global_birth_rescue_active
            and (
                split_candidates_for_trial is None
                or np.asarray(split_candidates_for_trial).size == 0
            )
            and proposal_data is not None
        ):
            split_candidates_for_trial = global_candidates
            split_candidate_counts_for_trial = global_candidate_counts_for_trial
            split_candidate_strengths_for_trial = self._candidate_initial_strengths(
                candidate_count=global_candidates.shape[0],
                candidate_kernel_sums=np.sum(
                    np.maximum(
                        np.asarray(split_candidate_counts_for_trial, dtype=float),
                        0.0,
                    ),
                    axis=0,
                ),
                residual_sum=max(
                    float(residual_sum),
                    float(np.sum(np.maximum(proposal_data.z_k, 0.0))),
                ),
            )
        suppress_death = (
            (residual_birth_gate_active or global_birth_rescue_active)
            and bool(self.config.birth_residual_suppress_death)
        )
        max_births = self.config.birth_max_per_update
        births_remaining = (
            None if max_births is None else max(0, int(max_births))
        )
        any_moved = False
        moved_indices: set[int] = set()
        moved_refit_indices: list[int] = []
        refresh_reference_ll: dict[int, float] = {}
        has_support_data = support_data is not None and support_data.z_k.size > 0
        support_cache: dict[
            int,
            tuple[
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.bool_],
            ],
        ] = {}
        structural_proposal_indices: set[int] | None = None
        topk_structural = self.config.structural_proposal_topk_particles
        if topk_structural is not None:
            structural_proposal_indices = self._select_structural_proposal_indices(
                int(topk_structural),
            )
        if (
            (residual_birth_gate_active or global_birth_rescue_active)
            and bool(self.config.birth_residual_expand_structural_particles)
        ):
            structural_proposal_indices = self._select_structural_proposal_indices(
                self.config.birth_residual_expanded_structural_topk_particles,
                require_birth_capacity=True,
            )
            if structural_proposal_indices is not None:
                self.last_birth_structural_eligible = len(structural_proposal_indices)
        if has_support_data:
            cache_start = time.perf_counter()
            grouped, fallback_indices = self._particle_indices_by_source_count()
            for source_count, particle_indices in grouped.items():
                if source_count <= 0 or not particle_indices:
                    continue
                lambda_m_group, lambda_total_group = self._lambda_components_for_particle_group(
                    support_data,
                    particle_indices,
                    source_count,
                )
                delta_ll_group = self._delta_log_likelihood_remove_group(
                    support_data,
                    lambda_total_group,
                    lambda_m_group,
                )
                prune_allowed_group = np.zeros(
                    (len(particle_indices), source_count),
                    dtype=bool,
                )
                if bool(self.config.source_prune_refit_after_remove) and source_count > 1:
                    needs_prune_rows = np.asarray(
                        [
                            self._needs_refit_prune_allowed(
                                self.continuous_particles[int(particle_idx)].state,
                                next_delta_ll=delta_ll_group[row_idx],
                                suppress_death=suppress_death,
                            )
                            for row_idx, particle_idx in enumerate(particle_indices)
                        ],
                        dtype=bool,
                    )
                    if np.any(needs_prune_rows):
                        subset_indices = [
                            int(particle_indices[row_idx])
                            for row_idx, needed in enumerate(needs_prune_rows)
                            if bool(needed)
                        ]
                        k_tensor, background_counts, strengths = (
                            self._unit_kernel_tensor_for_particle_group(
                                support_data,
                                subset_indices,
                                source_count,
                            )
                        )
                        q_min = max(float(self.config.min_strength), 0.0)
                        q_max = float(self.config.birth_q_max)
                        if q_max < q_min:
                            q_min, q_max = q_max, q_min
                        full_strengths, full_lambda_total = (
                            self._solve_strengths_for_kernel_tensor_batched(
                                support_data,
                                k_tensor=k_tensor,
                                background_counts=background_counts,
                                prior_mean=strengths,
                                iters=max(1, int(self.config.refit_iters)),
                                eps=float(self.config.refit_eps),
                                q_min=q_min,
                                q_max=q_max,
                            )
                        )
                        prune_allowed_subset = (
                            self._source_prune_refit_after_remove_mask_batched(
                                support_data,
                                k_tensor=k_tensor,
                                background_counts=background_counts,
                                full_strengths=full_strengths,
                                full_lambda_total=full_lambda_total,
                                iters=max(1, int(self.config.refit_iters)),
                                eps=float(self.config.refit_eps),
                                q_min=q_min,
                                q_max=q_max,
                            )
                        )
                        prune_allowed_group[needs_prune_rows] = prune_allowed_subset
                for row_idx, particle_idx in enumerate(particle_indices):
                    prune_allowed = prune_allowed_group[row_idx]
                    if (
                        not bool(self.config.source_prune_refit_after_remove)
                        and prune_allowed.size == source_count
                    ):
                        prune_allowed = self._source_prune_allowed_mask(
                            self.continuous_particles[particle_idx].state,
                            support_data,
                            lambda_m=lambda_m_group[:, row_idx, :],
                            lambda_total=lambda_total_group[:, row_idx],
                            delta_ll=delta_ll_group[row_idx],
                        )
                    support_cache[int(particle_idx)] = (
                        lambda_m_group[:, row_idx, :],
                        lambda_total_group[:, row_idx],
                        delta_ll_group[row_idx],
                        prune_allowed,
                    )
            for particle_idx in fallback_indices:
                st = self.continuous_particles[particle_idx].state
                if st.num_sources <= 0:
                    continue
                lambda_m, lambda_total = self._lambda_components(st, support_data)
                delta_ll = self._delta_log_likelihood_remove(
                    support_data.z_k,
                    lambda_total,
                    lambda_m,
                    observation_count_variance=support_data.observation_variances,
                )
                if (
                    bool(self.config.source_prune_refit_after_remove)
                    and not self._needs_refit_prune_allowed(
                        st,
                        next_delta_ll=delta_ll,
                        suppress_death=suppress_death,
                    )
                ):
                    prune_allowed = np.zeros(int(st.num_sources), dtype=bool)
                else:
                    prune_allowed = self._source_prune_allowed_mask(
                        st,
                        support_data,
                        lambda_m=lambda_m,
                        lambda_total=lambda_total,
                        delta_ll=delta_ll,
                    )
                support_cache[int(particle_idx)] = (
                    lambda_m,
                    lambda_total,
                    delta_ll,
                    prune_allowed,
                )
            timing["cache"] += time.perf_counter() - cache_start

        for particle_idx, particle in enumerate(self.continuous_particles):
            st = particle.state
            self._ensure_source_metadata(st)
            allow_structural_proposal = (
                structural_proposal_indices is None
                or int(particle_idx) in structural_proposal_indices
            )
            has_support = has_support_data
            moved = False
            global_birth_moved = False
            if st.num_sources > 0:
                st.ages = st.ages + 1
                below = st.strengths < float(self.config.min_strength)
                st.low_q_streaks[below] += 1
                st.low_q_streaks[~below] = 0
            lambda_m = None
            lambda_total = None
            cached_prune_allowed = None
            if has_support and st.num_sources > 0:
                cached_support = support_cache.get(int(particle_idx))
                if (
                    cached_support is not None
                    and cached_support[2].size == st.num_sources
                    and cached_support[3].size == st.num_sources
                ):
                    (
                        lambda_m,
                        lambda_total,
                        delta_ll,
                        cached_prune_allowed,
                    ) = cached_support
                else:
                    lambda_m, lambda_total = self._lambda_components(st, support_data)
                    delta_ll = self._delta_log_likelihood_remove(
                        support_data.z_k,
                        lambda_total,
                        lambda_m,
                        observation_count_variance=support_data.observation_variances,
                    )
                alpha = float(self.config.support_ema_alpha)
                st.support_scores = (1.0 - alpha) * st.support_scores + alpha * delta_ll
            if refit_data is not None and refit_data.z_k.size > 0:
                if (
                    refit_data is support_data
                    and lambda_total is not None
                    and np.asarray(lambda_total).shape == (int(refit_data.z_k.size),)
                ):
                    refresh_reference_ll[int(particle_idx)] = self._count_log_likelihood_np(
                        refit_data.z_k,
                        np.asarray(lambda_total, dtype=float),
                        observation_count_variance=refit_data.observation_variances,
                    )
                else:
                    refresh_reference_ll[int(particle_idx)] = self._trial_log_likelihood(
                        st,
                        refit_data,
                    )
            if has_support and st.num_sources > 0:
                pseudo_start = time.perf_counter()
                pseudo_moved = self._verify_pseudo_sources_for_state(
                    st,
                    support_data,
                    suppress_prune=suppress_death,
                    cached_lambda_m=lambda_m,
                    cached_lambda_total=lambda_total,
                    cached_delta_ll=delta_ll,
                    cached_prune_allowed=cached_prune_allowed,
                )
                timing["pseudo"] += time.perf_counter() - pseudo_start
                moved = moved or pseudo_moved
                if lambda_m is not None and lambda_m.shape[1] != st.num_sources:
                    lambda_m = None
                    lambda_total = None
                    delta_ll = None
                    cached_prune_allowed = None
            if st.num_sources > 0 and has_support:
                prune_start = time.perf_counter()
                kill_mask = np.ones(st.num_sources, dtype=bool)
                if not suppress_death:
                    exclusion_mask = self._source_detector_exclusion_mask(
                        st,
                        structural_data,
                    )
                    kill_mask[~exclusion_mask] = False
                    if (
                        cached_prune_allowed is not None
                        and cached_prune_allowed.shape == (int(st.num_sources),)
                    ):
                        prune_allowed = np.asarray(cached_prune_allowed, dtype=bool)
                    else:
                        prune_allowed = self._source_prune_allowed_mask(
                            st,
                            support_data,
                            lambda_m=lambda_m,
                            lambda_total=lambda_total,
                            delta_ll=delta_ll,
                        )
                    q_min = float(self.config.min_strength)
                    if q_min <= 0.0:
                        q_min = float(self.config.birth_q_min)
                    deterministic = (
                        st.low_q_streaks >= int(self.config.death_low_q_streak)
                    ) & (st.strengths < q_min) & prune_allowed
                    kill_mask[deterministic] = False
                    kill_candidates = (
                        st.low_q_streaks >= int(self.config.death_low_q_streak)
                    ) & (
                        st.support_scores
                        < float(self.config.death_delta_ll_threshold)
                    ) & prune_allowed
                    for idx, do_kill in enumerate(kill_candidates):
                        if (
                            kill_mask[idx]
                            and do_kill
                            and np.random.rand() < float(self.config.p_kill)
                        ):
                            kill_mask[idx] = False
                    if not np.all(kill_mask):
                        self.last_kill_count += int(np.sum(~kill_mask))
                        for idx in np.flatnonzero(~kill_mask):
                            reason = (
                                "death_low_q_deterministic"
                                if bool(deterministic[int(idx)])
                                else "death_low_support_stochastic"
                            )
                            self._record_source_event(
                                "source_removed",
                                st,
                                int(idx),
                                reason=reason,
                                extra={
                                    "support_score": float(st.support_scores[int(idx)]),
                                    "low_q_streak": int(st.low_q_streaks[int(idx)]),
                                    "death_delta_ll_threshold": float(
                                        self.config.death_delta_ll_threshold
                                    ),
                                    "death_low_q_streak": int(
                                        self.config.death_low_q_streak
                                    ),
                                },
                            )
                        st.positions = st.positions[kill_mask]
                        st.strengths = st.strengths[kill_mask]
                        st.ages = st.ages[kill_mask]
                        st.low_q_streaks = st.low_q_streaks[kill_mask]
                        st.support_scores = st.support_scores[kill_mask]
                        st.tentative_sources = st.tentative_sources[kill_mask]
                        st.verification_fail_streaks = (
                            st.verification_fail_streaks[kill_mask]
                        )
                        st.num_sources = st.positions.shape[0]
                        moved = True
                if self.config.max_sources is not None and st.num_sources > self.config.max_sources:
                    over = int(st.num_sources - self.config.max_sources)
                    if over > 0:
                        drop = np.argsort(st.support_scores)[:over]
                        keep = np.ones(st.num_sources, dtype=bool)
                        keep[drop] = False
                        for idx in np.asarray(drop, dtype=int):
                            self._record_source_event(
                                "source_removed",
                                st,
                                int(idx),
                                reason="max_sources_support_drop",
                                extra={
                                    "max_sources": int(self.config.max_sources),
                                    "support_score": float(st.support_scores[int(idx)]),
                                },
                            )
                        st.positions = st.positions[keep]
                        st.strengths = st.strengths[keep]
                        st.ages = st.ages[keep]
                        st.low_q_streaks = st.low_q_streaks[keep]
                        st.support_scores = st.support_scores[keep]
                        st.tentative_sources = st.tentative_sources[keep]
                        st.verification_fail_streaks = st.verification_fail_streaks[keep]
                        st.num_sources = st.positions.shape[0]
                        moved = True
                timing["prune"] += time.perf_counter() - prune_start

            can_try_split = (
                proposal_enabled
                and
                allow_structural_proposal
                and st.num_sources > 0
                and proposal_data is not None
                and proposal_data.z_k.size
            )
            if can_try_split:
                split_start = time.perf_counter()
                split_moved = False
                if self.config.max_sources is None or st.num_sources < self.config.max_sources:
                    try_residual_split = bool(self.config.split_residual_always_try) or (
                        np.random.rand() < float(self.config.split_prob)
                    )
                    if try_residual_split:
                        split_trial, split_delta = self._best_residual_guided_split_trial(
                            st,
                            proposal_data,
                            split_candidates_for_trial,
                            split_candidate_strengths_for_trial,
                            suppress_prune_after_refit=suppress_death,
                            candidate_unit_counts=split_candidate_counts_for_trial,
                        )
                        if (
                            split_trial is not None
                            and split_delta
                            >= self._structural_acceptance_threshold(
                                base_threshold=float(
                                    self.config.split_delta_ll_threshold
                                ),
                                complexity_penalty=float(
                                    self.config.split_complexity_penalty
                                ),
                            )
                        ):
                            old_count = int(st.num_sources)
                            for idx in range(old_count, int(split_trial.num_sources)):
                                self._record_source_event(
                                    "source_birth_accepted",
                                    split_trial,
                                    int(idx),
                                    reason="residual_guided_split",
                                    extra={"delta_ll": float(split_delta)},
                                )
                            self._replace_particle_state_from_trial(st, split_trial)
                            split_moved = True
                            moved = True
                    if not split_moved and np.random.rand() < float(self.config.split_prob):
                        candidates = np.where(
                            st.strengths >= float(self.config.split_strength_min)
                        )[0]
                        if candidates.size > 0:
                            idx = int(np.random.choice(candidates))
                            if st.ages[idx] > int(self.config.min_age_to_split):
                                split_lambda_m, split_lambda_total = self._lambda_components(
                                    st,
                                    proposal_data,
                                )
                                delta = np.random.normal(
                                    scale=float(self.config.split_position_sigma),
                                    size=3,
                                )
                                split_positions = self._project_positions_to_source_prior(
                                    np.vstack(
                                        [
                                            st.positions[idx] + delta,
                                            st.positions[idx] - delta,
                                        ]
                                    )
                                )
                                s1 = split_positions[0]
                                s2 = split_positions[1]
                                if np.linalg.norm(s1 - s2) >= 0.5 * float(
                                    self.config.birth_min_sep_m
                                ):
                                    u_min = float(self.config.split_strength_min_frac)
                                    u_max = float(self.config.split_strength_max_frac)
                                    u_low, u_high = (
                                        (u_min, u_max)
                                        if u_min <= u_max
                                        else (u_max, u_min)
                                    )
                                    u = np.random.uniform(u_low, u_high)
                                    q1 = float(st.strengths[idx]) * float(u)
                                    q2 = float(st.strengths[idx]) * float(1.0 - u)
                                    lam_new = expected_counts_per_source(
                                        kernel=self.continuous_kernel,
                                        isotope=self.isotope,
                                        detector_positions=proposal_data.detector_positions,
                                        sources=np.vstack([s1, s2]),
                                        strengths=np.array([q1, q2], dtype=float),
                                        live_times=proposal_data.live_times,
                                        fe_indices=proposal_data.fe_indices,
                                        pb_indices=proposal_data.pb_indices,
                                        source_scale=self._measurement_source_scale_vector(
                                            proposal_data.fe_indices,
                                            proposal_data.pb_indices,
                                        ),
                                    )
                                    lambda_new = (
                                        split_lambda_total
                                        - split_lambda_m[:, idx]
                                        + np.sum(lam_new, axis=1)
                                    )
                                    delta_ll = self._delta_log_likelihood_update(
                                        proposal_data.z_k,
                                        split_lambda_total,
                                        lambda_new,
                                        observation_count_variance=(
                                            proposal_data.observation_variances
                                        ),
                                    )
                                    split_threshold = self._structural_acceptance_threshold(
                                        base_threshold=float(
                                            self.config.split_delta_ll_threshold
                                        ),
                                        complexity_penalty=float(
                                            self.config.split_complexity_penalty
                                        ),
                                    )
                                    if delta_ll >= split_threshold and np.log(
                                        np.random.rand()
                                    ) < delta_ll:
                                        self._record_source_event(
                                            "source_removed",
                                            st,
                                            int(idx),
                                            reason="random_split_replaced_parent",
                                            extra={"delta_ll": float(delta_ll)},
                                        )
                                        st.positions = np.vstack(
                                            [
                                                st.positions[:idx],
                                                st.positions[idx + 1 :],
                                                s1,
                                                s2,
                                            ]
                                        )
                                        st.strengths = np.concatenate(
                                            [
                                                st.strengths[:idx],
                                                st.strengths[idx + 1 :],
                                                [q1, q2],
                                            ]
                                        )
                                        st.ages = np.concatenate(
                                            [st.ages[:idx], st.ages[idx + 1 :], [0, 0]]
                                        )
                                        st.low_q_streaks = np.concatenate(
                                            [
                                                st.low_q_streaks[:idx],
                                                st.low_q_streaks[idx + 1 :],
                                                [0, 0],
                                            ]
                                        )
                                        st.support_scores = np.concatenate(
                                            [
                                                st.support_scores[:idx],
                                                st.support_scores[idx + 1 :],
                                                [0.0, 0.0],
                                            ]
                                        )
                                        st.tentative_sources = np.concatenate(
                                            [
                                                st.tentative_sources[:idx],
                                                st.tentative_sources[idx + 1 :],
                                                [True, True],
                                            ]
                                        )
                                        st.verification_fail_streaks = np.concatenate(
                                            [
                                                st.verification_fail_streaks[:idx],
                                                st.verification_fail_streaks[idx + 1 :],
                                                [0, 0],
                                            ]
                                        )
                                        st.num_sources = st.positions.shape[0]
                                        self._record_source_event(
                                            "source_birth_accepted",
                                            st,
                                            int(st.num_sources - 2),
                                            reason="random_split_child",
                                            extra={"delta_ll": float(delta_ll)},
                                        )
                                        self._record_source_event(
                                            "source_birth_accepted",
                                            st,
                                            int(st.num_sources - 1),
                                            reason="random_split_child",
                                            extra={"delta_ll": float(delta_ll)},
                                        )
                                        moved = True
                timing["split"] += time.perf_counter() - split_start

            if (
                proposal_enabled
                and
                allow_structural_proposal
                and not suppress_death
                and st.num_sources >= 2
                and proposal_data is not None
                and proposal_data.z_k.size
                and np.random.rand() < float(self.config.merge_prob)
            ):
                merge_start = time.perf_counter()
                merge_trial, merge_delta = self._best_merge_trial(st, proposal_data)
                timing["merge"] += time.perf_counter() - merge_start
                if (
                    merge_trial is not None
                    and merge_delta >= float(self.config.merge_delta_ll_threshold)
                ):
                    for idx in range(int(st.num_sources)):
                        self._record_source_event(
                            "source_merge_accepted",
                            st,
                            int(idx),
                            reason="merge_replaced_particle_state",
                            extra={
                                "delta_ll": float(merge_delta),
                                "merged_source_count": int(merge_trial.num_sources),
                            },
                        )
                    self._replace_particle_state_from_trial(st, merge_trial)
                    moved = True

            if (
                proposal_enabled
                and
                allow_structural_proposal
                and global_birth_rescue_active
                and proposal_data is not None
                and proposal_data.z_k.size
                and (births_remaining is None or births_remaining > 0)
                and (
                    self.config.max_sources is None
                    or st.num_sources < self.config.max_sources
                )
            ):
                rescue_start = time.perf_counter()
                mp_limit = max(1, int(self.config.birth_matching_pursuit_max_new_sources))
                if births_remaining is None:
                    max_new = mp_limit
                else:
                    max_new = min(mp_limit, max(0, int(births_remaining)))
                accepted_births = self._apply_matching_pursuit_births_to_state(
                    st,
                    proposal_data,
                    global_candidates,
                    max_new_sources=max_new,
                    residual_gate_forced=False,
                    candidate_unit_counts=global_candidate_counts_for_trial,
                    global_rescue=True,
                )
                timing["birth"] += time.perf_counter() - rescue_start
                if accepted_births > 0:
                    self.last_birth_count += int(accepted_births)
                    if births_remaining is not None:
                        births_remaining -= int(accepted_births)
                    global_birth_moved = True
                    moved = True

            if (
                proposal_enabled
                and
                allow_structural_proposal
                and not global_birth_moved
                and birth_probs is not None
                and birth_kernel_sums is not None
                and birth_candidates is not None
                and residual_sum > 0.0
                and (births_remaining is None or births_remaining > 0)
                and (
                    (
                        bool(self.config.birth_residual_always_try)
                        and float(self.config.p_birth) > 0.0
                    )
                    or np.random.rand() < float(self.config.p_birth)
                )
            ):
                if (
                    self.config.max_sources is not None
                    and st.num_sources >= self.config.max_sources
                ):
                    continue
                birth_moved = False
                mp_limit = max(1, int(self.config.birth_matching_pursuit_max_new_sources))
                if mp_limit > 1 and proposal_data is not None:
                    if births_remaining is None:
                        max_new = mp_limit
                    else:
                        max_new = min(mp_limit, max(0, int(births_remaining)))
                    birth_mp_start = time.perf_counter()
                    accepted_births = self._apply_matching_pursuit_births_to_state(
                        st,
                        proposal_data,
                        birth_candidates,
                        max_new_sources=max_new,
                        residual_gate_forced=residual_birth_gate_active,
                        candidate_unit_counts=birth_candidate_counts,
                    )
                    timing["birth"] += time.perf_counter() - birth_mp_start
                    if accepted_births > 0:
                        self.last_birth_count += int(accepted_births)
                        if births_remaining is not None:
                            births_remaining -= int(accepted_births)
                        birth_moved = True
                        moved = True
                if birth_moved:
                    pass
                else:
                    idx = int(np.random.choice(len(birth_probs), p=birth_probs))
                    denom = float(birth_kernel_sums[idx])
                    if denom <= 0.0:
                        continue
                    q_new = float(self.config.birth_alpha) * residual_sum / max(denom, 1e-12)
                    if q_new <= 0.0:
                        continue
                    q_min = float(self.config.birth_q_min)
                    q_max = float(self.config.birth_q_max)
                    if q_max < q_min:
                        q_min, q_max = q_max, q_min
                    q_new = float(np.clip(q_new, q_min, q_max))
                    pos_new = birth_candidates[idx]
                    if st.num_sources > 0:
                        dist = np.linalg.norm(st.positions - pos_new[None, :], axis=1)
                        if np.any(dist < float(self.config.birth_min_sep_m)):
                            continue
                    trial = st.copy()
                    self._ensure_source_metadata(trial)
                    trial.positions = np.vstack(
                        [trial.positions[: trial.num_sources], pos_new]
                    )
                    trial.strengths = np.append(trial.strengths[: trial.num_sources], q_new)
                    trial.ages = np.append(trial.ages[: trial.num_sources], 0)
                    trial.low_q_streaks = np.append(
                        trial.low_q_streaks[: trial.num_sources],
                        0,
                    )
                    trial.support_scores = np.append(
                        trial.support_scores[: trial.num_sources],
                        0.0,
                    )
                    trial.tentative_sources = np.append(
                        trial.tentative_sources[: trial.num_sources],
                        True,
                    )
                    trial.verification_fail_streaks = np.append(
                        trial.verification_fail_streaks[: trial.num_sources],
                        0,
                    )
                    trial.num_sources = int(trial.positions.shape[0])
                    birth_ll_start = time.perf_counter()
                    base_ll = self._trial_log_likelihood(st, proposal_data)
                    timing["birth"] += time.perf_counter() - birth_ll_start
                    refit_start = time.perf_counter()
                    self._refit_strengths_for_particle(
                        trial,
                        proposal_data,
                        iters=max(1, int(self.config.refit_iters)),
                        eps=float(self.config.refit_eps),
                    )
                    timing["refit"] += time.perf_counter() - refit_start
                    prune_start = time.perf_counter()
                    self._prune_floor_sources_after_refit(
                        trial,
                        proposal_data,
                        suppress_prune=suppress_death,
                        record_kill_count=False,
                    )
                    timing["prune"] += time.perf_counter() - prune_start
                    if trial.num_sources <= st.num_sources:
                        continue
                    birth_ll_start = time.perf_counter()
                    delta_ll = float(
                        self._trial_log_likelihood(trial, proposal_data) - base_ll
                    )
                    timing["birth"] += time.perf_counter() - birth_ll_start
                    birth_threshold = self._structural_acceptance_threshold(
                        base_threshold=float(self.config.birth_delta_ll_threshold),
                        complexity_penalty=self._birth_complexity_penalty(
                            residual_gate_forced=residual_birth_gate_active,
                            measurement_count=int(proposal_data.z_k.size),
                        ),
                    )
                    forced_birth_proposal = (
                        residual_birth_gate_active
                        and bool(self.config.birth_residual_force_proposal_on_gate)
                        and np.isfinite(delta_ll)
                        and delta_ll
                        >= float(self.config.birth_residual_forced_min_delta_ll)
                    )
                    if (
                        not np.isfinite(delta_ll)
                        or (delta_ll < birth_threshold and not forced_birth_proposal)
                    ):
                        continue
                    self._stage_new_birth_metadata(
                        trial,
                        proposal_data,
                        old_count=int(st.num_sources),
                        delta_ll=delta_ll,
                    )
                    self._record_source_event(
                        "source_birth_accepted",
                        trial,
                        int(trial.num_sources - 1),
                        reason="single_residual_birth",
                        extra={
                            "delta_ll": float(delta_ll),
                            "forced_proposal": bool(forced_birth_proposal),
                        },
                    )
                    self._replace_particle_state_from_trial(st, trial)
                    self.last_birth_count += 1
                    if births_remaining is not None:
                        births_remaining -= 1
                    moved = True

            if moved and refit_data is not None and bool(self.config.refit_after_moves):
                moved_refit_indices.append(int(particle_idx))
            if moved:
                moved_indices.add(int(particle_idx))
            any_moved = any_moved or moved

        if moved_refit_indices and refit_data is not None and bool(self.config.refit_after_moves):
            refit_start = time.perf_counter()
            self._refit_particle_indices_batched(
                refit_data,
                moved_refit_indices,
                iters=int(self.config.refit_iters),
                eps=float(self.config.refit_eps),
                suppress_prune_after_refit=(
                    residual_birth_gate_active
                    and bool(self.config.birth_residual_suppress_death)
                ),
            )
            timing["refit"] += time.perf_counter() - refit_start
        if any_moved and refit_data is not None:
            refresh_start = time.perf_counter()
            refresh_lambda_cache = None
            if refit_data is support_data and support_cache:
                refresh_lambda_cache = {
                    int(particle_idx): np.asarray(cached[1], dtype=float)
                    for particle_idx, cached in support_cache.items()
                    if int(particle_idx) not in moved_indices
                    and np.asarray(cached[1]).shape == (int(refit_data.z_k.size),)
                }
            self.refresh_weights_from_measurements(
                refit_data,
                lambda_total_by_index=refresh_lambda_cache,
                reference_log_likelihood_by_index=refresh_reference_ll,
                moved_indices=moved_indices,
            )
            timing["refresh_weights"] += time.perf_counter() - refresh_start
            self._maybe_resample_after_structural_update()
        label_start = time.perf_counter()
        self.align_continuous_labels()
        timing["label"] += time.perf_counter() - label_start
        timing["total"] = time.perf_counter() - structural_start
        self.last_structural_timing_s = {
            key: float(value)
            for key, value in timing.items()
            if float(value) > 0.0 or key == "total"
        }
        if not proposal_enabled:
            self.last_structural_timing_s["birth_proposals_gated"] = 1.0

    def _background_level(self) -> float:
        """Resolve per-isotope background level."""
        level = self.config.background_level
        if isinstance(level, dict):
            return float(level.get(self.isotope, 0.0))
        return float(level)

    def estimate(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Continuous MMSE estimate over positions/strengths using continuous_particles.
        """
        if (
            self.config.converge_enable
            and self.is_converged
            and self.frozen_estimate is not None
            and self._convergence_can_freeze()
        ):
            return self.frozen_estimate
        if not self.continuous_particles:
            return np.zeros((0, 3)), np.zeros(0)
        if not self._can_use_gpu():
            states = [
                self.state_without_report_excluded_sources(p.state)
                for p in self.continuous_particles
            ]
            weights = np.asarray(self.continuous_weights, dtype=float)
            weight_sum = float(np.sum(weights))
            if weight_sum <= 0.0:
                weights = np.ones(len(states), dtype=float) / max(len(states), 1)
            else:
                weights = weights / weight_sum
            max_sources = max((state.num_sources for state in states), default=0)
            positions = np.zeros((max_sources, 3), dtype=float)
            strengths = np.zeros(max_sources, dtype=float)
            for source_idx in range(max_sources):
                source_weights = []
                source_positions = []
                source_strengths = []
                for weight, state in zip(weights, states):
                    if state.num_sources > source_idx:
                        source_weights.append(float(weight))
                        source_positions.append(state.positions[source_idx])
                        source_strengths.append(float(state.strengths[source_idx]))
                if not source_weights:
                    continue
                w_arr = np.asarray(source_weights, dtype=float)
                w_arr = w_arr / max(float(np.sum(w_arr)), 1e-12)
                pos_arr = np.vstack(source_positions)
                q_arr = np.asarray(source_strengths, dtype=float)
                positions[source_idx] = np.sum(w_arr[:, None] * pos_arr, axis=0)
                strengths[source_idx] = float(np.sum(w_arr * q_arr))
            active = strengths > 0.0
            return positions[active], strengths[active]
        from pf import gpu_utils
        import torch

        device = gpu_utils.resolve_device(self.config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.config.gpu_dtype)
        states = [
            self.state_without_report_excluded_sources(p.state)
            for p in self.continuous_particles
        ]
        positions_t, strengths_t, _, mask_t = gpu_utils.pack_states(states, device=device, dtype=dtype)
        weights = torch.as_tensor(self.continuous_weights, device=device, dtype=dtype)
        weight_sum = torch.sum(weights)
        if float(weight_sum) <= 0.0:
            weights = torch.full_like(weights, 1.0 / max(weights.numel(), 1))
        else:
            weights = weights / weight_sum
        w_mask = weights[:, None] * mask_t
        w_sum = torch.sum(w_mask, dim=0)
        w_sum_safe = torch.where(w_sum > 0, w_sum, torch.ones_like(w_sum))
        pos_mean = torch.sum(w_mask[:, :, None] * positions_t, dim=0) / w_sum_safe[:, None]
        str_mean = torch.sum(w_mask * strengths_t, dim=0) / w_sum_safe
        positions = pos_mean.detach().cpu().numpy()
        strengths = str_mean.detach().cpu().numpy()
        # Trim zero-strength slots.
        mask = strengths > 0
        positions = positions[mask]
        strengths = strengths[mask]
        return positions, strengths
