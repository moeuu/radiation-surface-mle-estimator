"""High-level estimator coordinating parallel PFs and shield rotation (Chapter 3)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import itertools
import re
import time
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple
import copy
import os

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import chi2

from measurement.kernels import KernelPrecomputer, ShieldParams
from measurement.model import EnvironmentConfig
from measurement.shielding import octant_index_from_rotation
from measurement.continuous_kernels import ContinuousKernel
from measurement.obstacles import ObstacleGrid
from measurement.source_surfaces import source_surface_kinds
from pf.defaults import DEFAULT_MAX_SOURCES_PER_ISOTOPE
from pf.likelihood import expected_counts_per_source
from pf.particle_filter import IsotopeParticleFilter, MeasurementData, PFConfig
from pf.reporting import dedupe_report_candidates, measurement_vector
from pf.resampling import systematic_resample
from pf.sparse_evidence import (
    SparsePoissonEvidenceConfig,
    fit_joint_sparse_poisson_evidence,
    fit_sparse_poisson_evidence,
    fit_sparse_poisson_spectral_evidence,
    joint_sparse_poisson_evidence_to_diagnostics,
    refine_sparse_poisson_evidence_offgrid,
    sparse_poisson_ambiguity_diagnostics,
    sparse_poisson_evidence_to_diagnostics,
)
from pf.state import IsotopeState


def _weighted_quantile(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    quantile: float,
) -> float:
    """Return a weighted quantile for non-negative planning statistics."""
    values = np.asarray(values, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    if values.size == 0:
        return 0.0
    if weights.size != values.size:
        raise ValueError("weights must have the same size as values.")
    finite = np.isfinite(values) & np.isfinite(weights) & (weights >= 0.0)
    if not np.any(finite):
        return 0.0
    values = values[finite]
    weights = weights[finite]
    total = float(np.sum(weights))
    if total <= 0.0:
        return float(np.quantile(values, np.clip(float(quantile), 0.0, 1.0)))
    order = np.argsort(values)
    values = values[order]
    weights = weights[order] / total
    cdf = np.cumsum(weights)
    idx = int(np.searchsorted(cdf, np.clip(float(quantile), 0.0, 1.0), side="left"))
    idx = min(max(idx, 0), values.size - 1)
    return float(values[idx])


@dataclass
class RotatingShieldPFConfig:
    """
    Configuration parameters for the rotating-shield PF (Sec. 3.4–3.5).

    Users can tune convergence thresholds and planning settings:
        - max_sources: cap on sources per isotope
        - ig_threshold: max IG below which rotation stops (Eq. 3.49)
        - max_dwell_time_s: per-pose dwell cap
        - credible_volume_threshold: max ellipsoid volume for positional credible regions
        - lambda_cost: motion-cost weight in Eq. 3.51
        - position_sigma: Gaussian jitter for positions (meters)
        - alpha_weights: isotope weights for IG criteria
        - death_low_q_streak: steps below min_strength before death is allowed
        - death_delta_ll_threshold: ΔLL threshold required to kill weak sources
        - support_ema_alpha: EMA weight for per-source ΔLL support
        - support_window: measurement window for per-source support scoring
        - birth_window: measurement window for residual-driven birth proposals
        - birth_softmax_temp: temperature for residual proposal sampling
        - birth_min_score: score floor for residual proposal sampling
        - birth_enable: enable birth/death/split/merge moves
        - birth_topk_particles: number of top-weight particles for residual mix
        - birth_use_weighted_topk: weight residual mix by particle weights
        - birth_min_sep_m: minimum separation between sources during birth
        - birth_detector_min_sep_m: minimum separation from measured detector poses
        - source_detector_exclusion_m: hard exclusion around measured detector poses
        - birth_candidate_jitter_sigma: position jitter (m) for birth candidates
        - birth_num_local_jitter: local jitter samples per candidate
        - birth_alpha: damping factor for new source strength
        - birth_q_max: clamp max for new source strength
        - birth_q_min: clamp min for new source strength
        - birth_max_per_update: cap accepted birth proposals per structural update
        - birth_delta_ll_threshold: likelihood-gain floor for accepting birth
        - birth_complexity_penalty: extra model-complexity penalty for birth
        - birth_bic_penalty_params: source parameter count for BIC birth penalty
        - structural_update_count_min_snr: SNR floor for count-threshold structural rows
        - structural_update_max_rel_sigma: relative-sigma cap for count-threshold structural rows
        - birth_residual_clip_quantile: clip residuals at this quantile
        - birth_residual_gate_p_value: chi-square p-value for residual birth evidence
        - birth_residual_min_support: minimum independent residual-supported measurements
        - birth_residual_support_sigma: per-measurement residual z-score support floor
        - birth_min_distinct_stations: minimum robot stations with residual birth evidence
        - birth_candidate_support_fraction: per-candidate residual overlap floor
        - birth_refit_residual_gate: require residuals to survive fixed-position strength refit
        - birth_refit_residual_min_fraction: residual fraction retained after refit for birth
        - birth_use_shield_coded_residual: rank birth candidates by full shield-coded residual response
        - birth_existing_response_corr_max: reject birth candidates collinear with existing responses
        - birth_count_distance_prior_weight: soft proposal weight for high unit-response candidates
        - birth_count_distance_strength_weight: soft penalty for candidates needing high fitted strength
        - birth_count_distance_log_clip: robust log-ratio clip for count-distance proposal terms
        - birth_count_distance_strength_sigma: log-strength scale for the high-strength penalty
        - birth_residual_always_try: try residual birth when the statistical gate passes
        - birth_residual_expand_structural_particles: expand residual birth proposals beyond normal top-k
        - birth_residual_expanded_structural_topk_particles: cap residual-gated structural proposals
        - birth_residual_acceptance_complexity_scale: scale complexity penalty after residual gate
        - birth_residual_suppress_death: delay death while positive residual birth evidence remains
        - birth_matching_pursuit_max_new_sources: max sequential residual births per particle
        - birth_matching_pursuit_topk_candidates: candidates evaluated per matching-pursuit birth
        - birth_jitter_topk_candidates: base residual-supported candidates jittered for birth
        - birth_global_rescue_enable: add station-level MLE-style surface candidates to runtime birth
        - birth_global_rescue_max_candidates: max global surface candidates used by runtime birth
        - birth_global_rescue_min_residual_fraction: residual fraction needed before runtime global rescue
        - birth_global_rescue_dedup_radius_m: distance used to merge runtime global birth candidates
        - birth_global_rescue_force_proposal_on_gate: allow global rescue births to bypass the LL/BIC threshold
        - birth_global_rescue_forced_min_delta_ll: minimum ΔLL for forced runtime global rescue births
        - birth_global_rescue_min_support: support count override for global rescue births
        - birth_global_rescue_min_distinct_poses: distinct-view override for global rescue births
        - birth_global_rescue_min_distinct_stations: station-count override for global rescue births
        - runtime_report_rescue_enable: inject report-level rescue modes after each station
        - runtime_report_rescue_particle_fraction: PF particle fraction reserved for runtime rescue
        - runtime_report_rescue_min_particles_per_source: minimum injected particles per rescued source
        - runtime_report_rescue_weight: posterior mass assigned to injected rescue particles
        - runtime_report_rescue_jitter_sigma_m: surface-projected rescue-particle position jitter
        - runtime_report_rescue_quarantine_enable: inject unstable rescue modes with reduced mass
        - runtime_report_rescue_quarantine_weight: posterior mass for quarantined rescue modes
        - runtime_report_rescue_candidate_weight: posterior mass for BIC-selected report modes that are not yet fully verified
        - runtime_report_rescue_memory_enable: retain recent report-rescue modes across stations
        - runtime_report_rescue_memory_decay: station-to-station rescue-memory score decay
        - runtime_report_rescue_memory_max_sources: max remembered rescue modes per isotope
        - report_mle_rescue_surface_quota_enable: reserve rescue slots across known surface strata
        - report_mle_rescue_surface_quota_min_score_fraction: score floor for stratum quota candidates
        - report_mle_rescue_surface_quota_per_stratum: reserved rescue slots per surface stratum
        - residual_decomposition_enable: enable raw/peak-suppressed residual layers
        - peak_suppression_enable: use strong-source leave-one-out residual layers
        - peak_suppression_min_source_fraction: source fraction defining suppressible peaks
        - peak_suppression_factor: fraction of a strong source contribution added back
        - residual_decomposition_max_layers: max residual layers used for birth proposals
        - pseudo_source_verification_enable: verify tentative birth sources before pruning
        - pseudo_source_min_delta_ll: leave-one-out ΔLL floor for confirming sources
        - pseudo_source_min_distinct_views: distinct shield views needed to confirm sources
        - pseudo_source_fail_grace_stations: failed verifications before pruning tentative sources
        - pseudo_source_corr_max: response-correlation ceiling against stronger sources
        - pseudo_source_temporal_sep_min: whitened temporal-code separation that can confirm high-correlation sources
        - report_exclude_unverified_sources: hide tentative birth sources from report estimates
        - source_prune_refit_after_remove: refit remaining strengths before pruning
        - source_prune_bic_penalty_params: source parameter count for BIC prune gain
        - refit_after_moves: refit strengths after birth/kill/split/merge
        - refit_iters: iterations for strength refit
        - refit_eps: epsilon for refit stability
        - weak_source_prune_min_expected_count: prune floor-strength sources below this support
        - weak_source_prune_min_fraction: prune floor-strength sources below this source fraction
        - weak_source_prune_min_age: source age before weak-source pruning is allowed
        - weak_source_prune_require_observable: only prune after enough observable measurements
        - weak_source_prune_min_observable_measurements: visible-view count needed before weak pruning
        - weak_source_prune_observable_count: per-measurement visible-count threshold
        - weak_source_prune_observable_fraction: per-measurement visible-fraction threshold
        - weak_source_prune_visibility_reference_strength: optional fixed cps@1m for visibility checks
        - conditional_strength_refit: refit strengths at station finalization
        - conditional_strength_refit_window: recent measurements used for strength refit
        - conditional_strength_refit_iters: iterations for conditional strength refit
        - conditional_strength_refit_reweight: reweight particles by profile-likelihood gain
        - conditional_strength_refit_cardinality_neutral_reweight: keep model-order mass during refit reweighting
        - conditional_strength_refit_reweight_clip: robust clip for profile-likelihood correction
        - conditional_strength_refit_min_count: minimum positive count for strength refit
        - conditional_strength_refit_min_snr: minimum count SNR for strength refit
        - conditional_strength_refit_prior_weight: MAP strength-prior weight
        - conditional_strength_refit_prior_rel_sigma: relative strength-prior sigma
        - source_strength_prior_mean: absolute source-rate prior mean shared by runtime/report refits
        - source_strength_prior_weight: absolute source-rate prior weight to suppress one-source absorption
        - source_strength_prior_rel_sigma: relative sigma for the absolute source-rate prior
        - source_strength_absorption_penalty_weight: legacy prior-based runtime high-strength penalty
        - source_strength_absorption_q_multiple: legacy prior-based runtime high-strength scale
        - source_strength_observation_overshoot_penalty_weight: data-driven runtime overshoot penalty
        - source_strength_observation_overshoot_sigma: count-sigma margin for runtime overshoot tests
        - source_strength_observation_overshoot_quantile: visible-view quantile for runtime overshoot bounds
        - source_strength_observation_overshoot_min_visible_fraction: response visibility threshold fraction
        - source_strength_observation_overshoot_min_visible_measurements: visible views needed for runtime overshoot
        - report_strength_refit: refit reported strengths conditioned on reported positions
        - report_strength_refit_iters: multiplicative Poisson regression iterations
        - report_strength_refit_eps: numerical floor for reported-strength regression
        - report_strength_refit_use_all_measurements: include shielded and low-count postures in report refit
        - report_strength_refit_preserve_cardinality: keep posterior clusters during report refit
        - report_strength_refit_prior_weight: MAP strength-prior weight for report refit
        - report_strength_refit_prior_rel_sigma: relative strength-prior sigma for report refit
        - report_strength_absorption_penalty_weight: legacy prior-based report high-strength penalty
        - report_strength_absorption_q_multiple: legacy prior-based report high-strength scale
        - report_strength_observation_overshoot_penalty_weight: data-driven report overshoot penalty
        - report_strength_observation_overshoot_sigma: count-sigma margin for report overshoot tests
        - report_strength_observation_overshoot_quantile: visible-view quantile for report overshoot bounds
        - report_strength_observation_overshoot_min_visible_fraction: response visibility threshold fraction
        - report_strength_observation_overshoot_min_visible_measurements: visible views needed for report overshoot
        - report_surface_local_refine: bounded local surface refinement before report BIC
        - report_surface_local_refine_radius_m: local refinement search radius
        - report_surface_local_refine_grid_steps: half-grid steps for local refinement
        - report_surface_local_refine_max_candidates_per_source: stencil cap per source
        - report_surface_local_refine_max_sources: source-slot cap for local refinement
        - report_surface_local_refine_min_ll_gain: minimum accepted local-refine LL gain
        - report_mle_rescue_enable: add MLE-style global/residual surface candidates before report BIC
        - report_mle_rescue_max_candidates: total report candidates kept after rescue
        - report_mle_rescue_max_posterior_candidates: unverified posterior clusters added as rescue candidates
        - report_mle_rescue_max_residual_candidates: global/residual-ranked surface candidates added as rescue candidates
        - report_mle_rescue_dedup_radius_m: distance used to merge duplicate rescue candidates
        - report_mle_rescue_min_residual_fraction: residual fraction needed before surface-grid rescue
        - report_mle_rescue_visibility_weight: blend weight for visible-window rescue scoring
        - report_mle_rescue_min_visible_measurements: visible measurements needed for rescue candidates
        - report_mle_rescue_visible_count: per-measurement count threshold for visible rescue support
        - report_mle_rescue_visibility_reference_strength: optional fixed cps@1m for rescue visibility checks
        - report_cluster_model_selection: remove redundant reported clusters by refit-after-remove
        - report_cluster_bic_penalty_params: source parameter count for reported cluster BIC
        - report_cluster_delta_ll_threshold: tolerated report-model likelihood loss
        - report_model_order_require_posterior_match: require PF K posterior to match report K for stopping
        - report_model_order_prune_particles: apply report-level BIC rejections to PF source slots
        - report_model_order_particle_prune_radius_m: source-slot radius for BIC-driven PF pruning
        - report_model_order_zero_source_min_bic_margin: BIC margin required before a count-supported isotope is treated as absent
        - report_model_order_workers: workers for report-level subset BIC scoring
        - report_model_order_parallel_min_subsets: subset count needed before parallel scoring
        - sparse_poisson_evidence_enable: compute all-history dictionary Poisson evidence
        - sparse_poisson_evidence_authoritative: use sparse evidence as report model-order authority
        - sparse_poisson_evidence_candidate_limit: max dictionary columns scored after the batched prefilter
        - sparse_poisson_evidence_holdout_stride: deterministic held-out view stride for predictive deviance
        - sparse_poisson_evidence_min_bic_margin: minimum BIC gap required before evidence is ready
        - sparse_poisson_evidence_min_distinct_stations: station support required before evidence is authoritative
        - sparse_poisson_offgrid_refine_enable: profile selected sparse sources in continuous coordinates
        - sparse_poisson_offgrid_refine_radius_m: local bounds around each coarse dictionary source
        - runtime_report_rescue_verification_queue_only: queue report rescue candidates instead of injecting them
        - min_age_to_split: minimum age before split proposals
        - use_clustered_output: use clustered estimate when birth is enabled
        - cluster_eps_m: clustering radius in meters
        - cluster_min_samples: minimum samples per cluster
        - split_prob: probability of split proposals per particle
        - split_strength_min: minimum strength for split candidates
        - split_position_sigma: position jitter for split proposals
        - split_strength_min_frac: min split fraction for q1/q2
        - split_strength_max_frac: max split fraction for q1/q2
        - split_delta_ll_threshold: ΔLL threshold for split acceptance
        - split_complexity_penalty: extra model-complexity penalty for split
        - split_residual_guided: use posterior residual candidates for split moves
        - split_residual_always_try: always test residual split on selected particles
        - split_residual_candidate_count: residual candidates evaluated per split
        - merge_prob: probability of merge proposals per particle
        - merge_distance_max: max distance for merge candidates
        - merge_delta_ll_threshold: ΔLL threshold for merge acceptance
        - merge_response_corr_min: response-correlation floor for merge candidates
        - merge_search_topk_pairs: max response-redundant pairs tested per merge move
        - structural_proposal_topk_particles: posterior-support cap for split/merge proposals
        - structural_trial_workers: worker count for deterministic split/merge trial chunks
        - structural_trial_parallel_min_trials: minimum trial count before worker chunks
        - source_position_prior: "volume" or "surface" PF source-position support
        - init_num_sources: inclusive range for initial source count per particle
        - init_grid_spacing_m: grid spacing for deterministic particle initialization
        - init_grid_repeats: repeated strength samples per deterministic grid point
        - roughening_k: roughening coefficient for post-resample position jitter
        - min_sigma_pos: minimum roughening sigma (meters)
        - max_sigma_pos: maximum roughening sigma (meters)
        - roughening_decay: multiplier decay per resample within an observation
        - roughening_min_mult: minimum multiplier for roughening decay
        - init_strength_log_mean: log-normal median for fallback strength initialization
        - init_strength_log_sigma: log-normal spread for fallback strength initialization
        - strength_log_sigma: log-space jitter for strengths
        - adaptive_strength_prior: rescale early strength particles from observed counts
        - adaptive_strength_prior_steps: number of first measurements allowed to rescale strengths
        - adaptive_strength_prior_min_counts: Poisson upper-count floor for zero/weak observations
        - adaptive_strength_prior_log_sigma: log-normal proposal spread around count-matched strength
        - adaptive_strength_prior_max_upscale: per-update upper strength multiplier
        - pose_min_observation_quantile: posterior quantile used for observability guarantees
        - orientation_k: maximum number of orientations to execute per pose
        - min_rotations_per_pose: minimum orientations before IG early stopping
        - orientation_selection_mode: "eig"
        - planning_particles: particle count used for orientation scoring (None = all)
        - planning_method: how to select planning particles (top_weight/resample)
        - use_gpu: enable torch acceleration for continuous kernel evaluation
        - gpu_device: torch device string (e.g., "cuda" or "cpu")
        - gpu_dtype: torch dtype string ("float32" or "float64")
        - target_ess_ratio: target ESS/N for tempered updates
        - max_temper_steps: max sub-steps for tempered updates
        - min_delta_beta: minimum delta_beta for tempering
        - use_tempering: enable ESS-targeted likelihood tempering
        - max_resamples_per_observation: cap resamples per observation update
        - temper_resample_cooldown_steps: substeps to skip resampling after resample
        - temper_resample_force_ratio: ESS/N ratio forcing resample despite cooldown
        - disable_regularize_on_temper_resample: skip roughening on temper resamples
        - deferred_resample_roughening_scale: roughening scale during station-burst resampling
        - cardinality_preserving_resample: preserve posterior source-count mass during resampling
        - mode_preserving_resample: keep distinct source modes during resampling
        - mode_preserving_max_modes: max spatial source modes protected per resample
        - mode_preserving_particles_per_mode: particles retained per protected mode
        - mode_preserving_radius_m: spatial clustering radius for protected source modes
        - mode_preserving_min_weight_fraction: minimum mode support fraction to protect
        - mode_preserving_cardinality_strata: keep source-count hypotheses during mode protection
        - mode_preserving_min_particles_per_cardinality: particles protected per source count
        - adapt_cooldown_steps: block particle-count shrink steps after resampling
        - eig_num_samples: Monte-Carlo samples for EIG (Eq. 3.44)
        - planning_eig_samples: Monte-Carlo samples for EIG inside planning rollouts
        - planning_rollout_particles: particle cap for IG evaluation in rollouts
        - planning_rollout_method: selection method for rollout particles
        - preselect_*: optional surrogate stage settings for candidate reduction
        - use_fast_gpu_rollout: enable approximate fast GPU rollouts for uncertainty prediction
        - ig_workers: number of parallel workers for IG grid evaluation (0 = auto)
        - use_tempering: enable ESS-targeted tempered updates in the PF
        - measurement_scale_by_isotope: isotope-wise source response scales
        - measurement_scale_by_isotope_and_pair: shield-pair response scales
        - count_likelihood_model: "poisson", "gaussian", or "student_t"
        - transport_model_rel_sigma: relative model mismatch from scatter/build-up omissions
        - transport_model_abs_sigma: absolute transport-model mismatch floor in counts
        - spectrum_count_rel_sigma: relative spectrum-decomposition count uncertainty
        - spectrum_count_abs_sigma: additive spectrum-decomposition count uncertainty
        - low_count_abs_sigma: extra low-count uncertainty floor in counts
        - low_count_transition_counts: count scale where the low-count floor decays
        - count_likelihood_df: Student-t degrees of freedom for robust count likelihood
        - history_estimate_interval: exact report-history stride; 0 disables history
        - candidate_response_cache_max_entries: LRU entries for deterministic candidate responses
        - parallel_isotope_updates: run independent isotope structural updates in parallel
        - parallel_isotope_workers: worker count for parallel isotope structural updates
        - label_enable: enable label alignment for continuous particles
        - label_alignment_iters: iterations for label alignment refinement
        - label_pos_weight: position cost weight for label alignment
        - label_strength_weight: strength cost weight for label alignment
        - label_missing_cost: missing-source cost for label alignment
        - label_pos_scale: optional position scale for label alignment
        - label_strength_scale: optional strength scale for label alignment
        - converge_enable: enable per-isotope convergence gating
        - converge_window: window length for convergence checks
        - converge_map_move_eps_m: MMSE position stability threshold (meters)
        - converge_ess_ratio_high: ESS/N threshold for convergence
        - converge_ll_improve_eps: LL improvement tolerance
        - converge_min_steps: minimum steps before convergence
        - converge_require_all: if True, all criteria must hold; else any two
    """

    num_particles: int = 200
    min_particles: int | None = None
    max_particles: int | None = None
    ess_low: float = 0.5
    ess_high: float = 0.9
    max_sources: int | None = DEFAULT_MAX_SOURCES_PER_ISOTOPE
    resample_threshold: float = 0.5
    position_sigma: float = 0.1
    strength_sigma: float = 0.1
    background_sigma: float = 0.1
    background_level: float | dict[str, float] = 0.0
    measurement_scale_by_isotope: Dict[str, float] | None = None
    measurement_scale_by_isotope_and_pair: Dict[str, Dict[int, float]] | None = None
    count_likelihood_model: str = "poisson"
    transport_model_rel_sigma: float | Dict[str, float] = 0.0
    transport_model_abs_sigma: float | Dict[str, float] = 0.0
    spectrum_count_rel_sigma: float | Dict[str, float] = 0.0
    spectrum_count_abs_sigma: float | Dict[str, float] = 0.0
    low_count_abs_sigma: float | Dict[str, float] = 0.0
    low_count_transition_counts: float | Dict[str, float] = 0.0
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
    birth_q_max: float = 5e6
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
    birth_global_rescue_candidate_memory_enable: bool = False
    birth_global_rescue_candidate_memory_decay: float = 0.85
    birth_global_rescue_candidate_memory_max_candidates: int = 0
    birth_global_rescue_candidate_memory_min_retained: int = 0
    high_strength_split_enable: bool = True
    high_strength_split_q_multiple: float = 2.0
    high_strength_split_offset_m: float = 1.5
    high_strength_split_candidate_count: int = 12
    report_best_so_far_enable: bool = True
    report_best_so_far_min_score_improvement: float = 1.0e-9
    report_best_so_far_final_min_measurement_fraction: float = 0.8
    report_best_so_far_final_require_resolved: bool = True
    runtime_report_rescue_enable: bool = False
    runtime_report_rescue_particle_fraction: float = 0.15
    runtime_report_rescue_min_particles_per_source: int = 4
    runtime_report_rescue_weight: float = 0.10
    runtime_report_rescue_jitter_sigma_m: float = 0.10
    runtime_report_rescue_quarantine_enable: bool = True
    runtime_report_rescue_quarantine_weight: float = 0.02
    runtime_report_rescue_candidate_weight: float = 0.06
    runtime_report_rescue_memory_enable: bool = True
    runtime_report_rescue_memory_decay: float = 0.90
    runtime_report_rescue_memory_max_sources: int = 0
    all_history_dictionary_proposal_enable: bool = False
    all_history_dictionary_proposal_weight: float = 0.04
    all_history_dictionary_proposal_max_candidates: int = 0
    candidate_verification_queue_enable: bool = False
    candidate_verification_queue_weight: float = 0.05
    candidate_verification_queue_decay: float = 0.85
    candidate_verification_queue_max_sources: int = 0
    report_mle_rescue_surface_quota_enable: bool = True
    report_mle_rescue_surface_quota_min_score_fraction: float = 0.0
    report_mle_rescue_surface_quota_per_stratum: int = 1
    report_mle_rescue_spatial_quota_enable: bool = True
    report_mle_rescue_spatial_quota_tile_m: float = 2.5
    report_mle_rescue_spatial_quota_per_tile: int = 1
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
    report_strength_refit: bool = False
    report_strength_refit_iters: int = 64
    report_strength_refit_eps: float = 1.0e-9
    report_strength_refit_use_all_measurements: bool = True
    report_strength_refit_preserve_cardinality: bool = False
    report_strength_refit_prior_weight: float = 0.0
    report_strength_refit_prior_rel_sigma: float = 2.0
    report_strength_absorption_penalty_weight: float = 0.0
    report_strength_absorption_q_multiple: float = 4.0
    report_strength_observation_overshoot_penalty_weight: float = 0.0
    report_strength_observation_overshoot_sigma: float = 5.0
    report_strength_observation_overshoot_quantile: float = 0.05
    report_strength_observation_overshoot_min_visible_fraction: float = 0.05
    report_strength_observation_overshoot_min_visible_measurements: int = 3
    report_surface_local_refine: bool = False
    report_surface_local_refine_radius_m: float = 0.5
    report_surface_local_refine_grid_steps: int = 1
    report_surface_local_refine_max_candidates_per_source: int = 27
    report_surface_local_refine_max_sources: int = 0
    report_surface_local_refine_min_ll_gain: float = 0.0
    report_mle_rescue_enable: bool = False
    report_mle_rescue_max_candidates: int = 12
    report_mle_rescue_max_posterior_candidates: int = 8
    report_mle_rescue_max_residual_candidates: int = 8
    report_mle_rescue_dedup_radius_m: float = 0.5
    report_mle_rescue_min_residual_fraction: float = 0.01
    report_mle_rescue_visibility_weight: float = 0.0
    report_mle_rescue_min_visible_measurements: int = 1
    report_mle_rescue_visible_count: float = 0.0
    report_mle_rescue_visibility_reference_strength: float = 0.0
    report_cluster_model_selection: bool = True
    report_cluster_bic_penalty_params: int = 4
    report_cluster_delta_ll_threshold: float = 0.0
    report_cluster_model_selection_max_candidates: int = 12
    report_model_order_require_posterior_match: bool = True
    report_model_order_prune_particles: bool = False
    report_model_order_particle_prune_radius_m: float = 0.0
    report_model_order_min_bic_margin: float = 0.0
    report_model_order_zero_source_min_bic_margin: float = 10.0
    report_model_order_condition_max: float = 0.0
    report_model_order_corr_penalty_weight: float = 0.0
    report_model_order_corr_penalty_threshold: float = 0.98
    report_model_order_corr_penalty_power: float = 1.0
    report_model_order_subset_corr_prune_threshold: float = 0.0
    report_model_order_underfit_gate: bool = True
    report_model_order_underfit_min_residual_fraction: float = -1.0
    report_model_order_underfit_min_positive_chi2: float = 0.0
    report_model_order_workers: int = 1
    report_model_order_parallel_min_subsets: int = 128
    sparse_poisson_evidence_enable: bool = False
    sparse_poisson_evidence_authoritative: bool = False
    sparse_poisson_evidence_candidate_limit: int = 2048
    sparse_poisson_evidence_refit_iters: int = 64
    sparse_poisson_evidence_holdout_stride: int = 4
    sparse_poisson_evidence_parameter_count_per_source: int = 4
    sparse_poisson_evidence_min_bic_margin: float = 2.0
    sparse_poisson_evidence_min_distinct_stations: int = 1
    sparse_poisson_evidence_corr_prune_threshold: float = 0.995
    sparse_poisson_evidence_max_response_correlation: float = 0.98
    sparse_poisson_evidence_condition_max: float = 100.0
    sparse_poisson_spectral_evidence_enable: bool = True
    sparse_poisson_spectral_evidence_primary: bool = True
    sparse_poisson_spectral_nuisance_enable: bool = True
    sparse_poisson_joint_evidence_enable: bool = True
    sparse_poisson_offgrid_refine_enable: bool = True
    sparse_poisson_offgrid_refine_radius_m: float = 0.75
    sparse_poisson_offgrid_refine_max_iter: int = 64
    sparse_poisson_offgrid_refine_min_ll_gain: float = 0.0
    sparse_poisson_ambiguity_report_enable: bool = True
    sparse_poisson_ambiguity_corr_threshold: float = 0.98
    sparse_poisson_ambiguity_bic_gap_threshold: float = 2.0
    sparse_poisson_ambiguity_condition_max: float = 100.0
    runtime_report_rescue_verification_queue_only: bool = False
    report_pre_finalize_guard: bool = True
    history_estimate_interval: int = 1
    candidate_response_cache_max_entries: int = 24
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
    short_time_s: float = 0.5  # Recommended short-time measurement (Sec. 3.4.3).
    ig_threshold: float = 1e-3  # ΔIG stopping threshold (Sec. 3.4.4).
    max_dwell_time_s: float = 5.0  # Max dwell time per pose.
    lambda_cost: float = 1.0  # Motion-cost weight (Eq. 3.51).
    alpha_weights: Dict[str, float] | None = None  # EIG isotope weights alpha_h.
    credible_volume_threshold: float = 1e-3  # Max 95% credible volume for convergence.
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
    mode_preserving_resample: bool = True
    mode_preserving_max_modes: int = 6
    mode_preserving_particles_per_mode: int = 3
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
    position_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    position_max: Tuple[float, float, float] = (10.0, 10.0, 10.0)
    source_position_prior: str = "volume"
    init_num_sources: Tuple[int, int] = (0, 3)
    init_grid_spacing_m: float | None = None
    init_grid_repeats: int = 1
    roughening_k: float = 0.5
    surface_rejuvenation_enable: bool = True
    min_sigma_pos: float = 0.05
    max_sigma_pos: float = 1.5
    roughening_decay: float = 0.5
    roughening_min_mult: float = 0.25
    init_strength_log_mean: float = 9.0
    init_strength_log_sigma: float = 1.0
    strength_log_sigma: float = 0.3
    adaptive_strength_prior: bool = False
    adaptive_strength_prior_steps: int = 3
    adaptive_strength_prior_min_counts: float = 3.0
    adaptive_strength_prior_log_sigma: float = 0.7
    adaptive_strength_prior_max_upscale: float = 10.0
    observation_covariance_projection_enable: bool = True
    observation_covariance_projection_weight: float = 1.0
    observation_covariance_projection_max_corr: float = 0.999
    pose_min_observation_counts: float = 0.0
    pose_min_observation_penalty_scale: float = 1.0
    pose_min_observation_aggregate: str = "max"
    pose_min_observation_max_particles: int | None = None
    pose_min_observation_quantile: float = 0.25
    orientation_k: int = 8
    min_rotations_per_pose: int = 0
    orientation_selection_mode: str = "eig"
    planning_particles: int | None = None
    planning_method: str = "top_weight"
    use_gpu: bool = True
    gpu_device: str = "cuda"
    gpu_dtype: str = "float32"
    eig_num_samples: int = 50
    planning_eig_samples: int | None = None
    planning_rollout_particles: int | None = None
    planning_rollout_method: str | None = None
    preselect_orientations: bool = False
    preselect_metric: str = "var_log_lambda"
    preselect_delta: float = 0.05
    preselect_k_min: int = 8
    preselect_k_max: int = 16
    use_fast_gpu_rollout: bool = False
    ig_workers: int = 0
    parallel_isotope_updates: bool = True
    parallel_isotope_workers: int | None = None
    label_enable: bool = True
    label_alignment_iters: int = 2
    label_pos_weight: float = 1.0
    label_strength_weight: float = 0.2
    label_missing_cost: float = 1e3
    label_pos_scale: float | None = None
    label_strength_scale: float | None = None
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

    def __post_init__(self) -> None:
        """Validate and normalize estimator configuration values."""
        if self.min_particles is None:
            self.min_particles = max(1, int(self.num_particles * 0.5))
        if self.max_particles is None:
            self.max_particles = max(self.num_particles, int(self.num_particles * 2.0))
        self.ess_low = float(self.ess_low)
        self.ess_high = float(self.ess_high)
        if not 0.0 < self.ess_low < self.ess_high < 1.0:
            raise ValueError(
                "ess_low and ess_high must satisfy 0 < ess_low < ess_high < 1."
            )
        self.init_grid_repeats = max(1, int(self.init_grid_repeats))
        self.ig_workers = int(self.ig_workers)
        if self.ig_workers < 0:
            raise ValueError("ig_workers must be >= 0.")
        self.adaptive_strength_prior_steps = int(self.adaptive_strength_prior_steps)
        if self.adaptive_strength_prior_steps < 0:
            raise ValueError("adaptive_strength_prior_steps must be >= 0.")
        self.adaptive_strength_prior_min_counts = float(
            self.adaptive_strength_prior_min_counts
        )
        if self.adaptive_strength_prior_min_counts < 0.0:
            raise ValueError("adaptive_strength_prior_min_counts must be >= 0.")
        self.adaptive_strength_prior_log_sigma = float(
            self.adaptive_strength_prior_log_sigma
        )
        if self.adaptive_strength_prior_log_sigma < 0.0:
            raise ValueError("adaptive_strength_prior_log_sigma must be >= 0.")
        self.adaptive_strength_prior_max_upscale = float(
            self.adaptive_strength_prior_max_upscale
        )
        if self.adaptive_strength_prior_max_upscale < 1.0:
            raise ValueError("adaptive_strength_prior_max_upscale must be >= 1.")
        self.observation_covariance_projection_enable = bool(
            self.observation_covariance_projection_enable
        )
        self.observation_covariance_projection_weight = max(
            0.0,
            float(self.observation_covariance_projection_weight),
        )
        self.observation_covariance_projection_max_corr = float(
            np.clip(
                float(self.observation_covariance_projection_max_corr),
                0.0,
                1.0,
            )
        )
        self.pose_min_observation_counts = float(self.pose_min_observation_counts)
        if self.pose_min_observation_counts < 0.0:
            raise ValueError("pose_min_observation_counts must be >= 0.")
        self.pose_min_observation_penalty_scale = float(
            self.pose_min_observation_penalty_scale
        )
        if self.pose_min_observation_penalty_scale < 0.0:
            raise ValueError("pose_min_observation_penalty_scale must be >= 0.")
        self.pose_min_observation_aggregate = (
            str(self.pose_min_observation_aggregate).strip().lower()
        )
        if self.pose_min_observation_aggregate not in {"max", "mean"}:
            raise ValueError("pose_min_observation_aggregate must be max or mean.")
        if self.pose_min_observation_max_particles is not None:
            self.pose_min_observation_max_particles = int(
                self.pose_min_observation_max_particles
            )
            if self.pose_min_observation_max_particles < 0:
                raise ValueError("pose_min_observation_max_particles must be >= 0.")
        self.pose_min_observation_quantile = float(self.pose_min_observation_quantile)
        if not 0.0 <= self.pose_min_observation_quantile <= 1.0:
            raise ValueError("pose_min_observation_quantile must be in [0, 1].")
        normalized_likelihood = str(self.count_likelihood_model).strip().lower()
        if normalized_likelihood in {"normal"}:
            normalized_likelihood = "gaussian"
        if normalized_likelihood in {"robust", "robust_gaussian", "t"}:
            normalized_likelihood = "student_t"
        if normalized_likelihood not in {"poisson", "gaussian", "student_t"}:
            raise ValueError(
                "count_likelihood_model must be poisson, gaussian, or student_t."
            )
        self.count_likelihood_model = normalized_likelihood
        self.count_likelihood_df = max(float(self.count_likelihood_df), 1.0)
        self.shield_view_ratio_likelihood_enable = bool(
            self.shield_view_ratio_likelihood_enable
        )
        self.shield_view_ratio_likelihood_weight = max(
            0.0,
            float(self.shield_view_ratio_likelihood_weight),
        )
        self.shield_view_ratio_likelihood_concentration = max(
            1.0e-6,
            float(self.shield_view_ratio_likelihood_concentration),
        )
        self.shield_view_ratio_likelihood_min_total_count = max(
            0.0,
            float(self.shield_view_ratio_likelihood_min_total_count),
        )
        self.shield_view_ratio_likelihood_min_views = max(
            2,
            int(self.shield_view_ratio_likelihood_min_views),
        )
        self.station_view_covariance_enable = bool(
            self.station_view_covariance_enable
        )
        self.station_view_correlated_spectrum_fraction = max(
            0.0,
            float(self.station_view_correlated_spectrum_fraction),
        )
        self.spectrum_likelihood_bin_chunk = max(
            1,
            int(self.spectrum_likelihood_bin_chunk),
        )
        self.birth_residual_gate_p_value = float(self.birth_residual_gate_p_value)
        if self.birth_residual_gate_p_value < 0.0:
            raise ValueError("birth_residual_gate_p_value must be >= 0.")
        self.birth_residual_gate_p_value = min(self.birth_residual_gate_p_value, 1.0)
        self.birth_residual_min_support = max(1, int(self.birth_residual_min_support))
        self.birth_residual_support_sigma = max(
            0.0,
            float(self.birth_residual_support_sigma),
        )
        self.birth_candidate_support_fraction = float(
            np.clip(float(self.birth_candidate_support_fraction), 0.0, 1.0)
        )
        self.source_detector_exclusion_m = max(
            0.0,
            float(self.source_detector_exclusion_m),
        )
        self.birth_refit_residual_min_fraction = max(
            0.0,
            float(self.birth_refit_residual_min_fraction),
        )
        self.birth_existing_response_corr_max = float(
            np.clip(float(self.birth_existing_response_corr_max), 0.0, 1.0)
        )
        self.birth_response_condition_max = max(
            0.0,
            float(self.birth_response_condition_max),
        )
        self.structural_update_min_snr = max(
            0.0,
            float(self.structural_update_min_snr),
        )
        self.structural_update_count_min_snr = max(
            0.0,
            float(self.structural_update_count_min_snr),
        )
        self.structural_update_max_rel_sigma = max(
            0.0,
            float(self.structural_update_max_rel_sigma),
        )
        self.converge_cluster_spread_max_m = max(
            0.0,
            float(self.converge_cluster_spread_max_m),
        )
        self.converge_cluster_min_support_fraction = float(
            np.clip(float(self.converge_cluster_min_support_fraction), 0.0, 1.0)
        )
        self.birth_count_distance_prior_weight = max(
            0.0,
            float(self.birth_count_distance_prior_weight),
        )
        self.birth_count_distance_strength_weight = max(
            0.0,
            float(self.birth_count_distance_strength_weight),
        )
        self.birth_count_distance_log_clip = max(
            0.0,
            float(self.birth_count_distance_log_clip),
        )
        self.birth_count_distance_strength_sigma = max(
            1.0e-12,
            float(self.birth_count_distance_strength_sigma),
        )
        self.birth_matching_pursuit_max_new_sources = max(
            1,
            int(self.birth_matching_pursuit_max_new_sources),
        )
        self.birth_matching_pursuit_topk_candidates = max(
            1,
            int(self.birth_matching_pursuit_topk_candidates),
        )
        self.birth_orthogonalize_residual_candidates = bool(
            self.birth_orthogonalize_residual_candidates
        )
        self.birth_orthogonal_candidate_corr_max = float(
            np.clip(float(self.birth_orthogonal_candidate_corr_max), 0.0, 1.0)
        )
        if self.birth_residual_expanded_structural_topk_particles is not None:
            expanded_topk = int(self.birth_residual_expanded_structural_topk_particles)
            self.birth_residual_expanded_structural_topk_particles = (
                None if expanded_topk <= 0 else expanded_topk
            )
        self.mode_preserving_max_modes = max(
            0,
            int(self.mode_preserving_max_modes),
        )
        self.cardinality_preserving_min_stations = max(
            0,
            int(self.cardinality_preserving_min_stations),
        )
        self.cardinality_preserving_require_confirmed_structure = bool(
            self.cardinality_preserving_require_confirmed_structure
        )
        self.deferred_resample_roughening_scale = max(
            0.0,
            float(self.deferred_resample_roughening_scale),
        )
        self.mode_preserving_particles_per_mode = max(
            0,
            int(self.mode_preserving_particles_per_mode),
        )
        self.mode_preserving_radius_m = max(
            1.0e-6,
            float(self.mode_preserving_radius_m),
        )
        self.mode_preserving_min_weight_fraction = max(
            0.0,
            float(self.mode_preserving_min_weight_fraction),
        )
        self.mode_preserving_surface_strata = bool(self.mode_preserving_surface_strata)
        self.mode_preserving_height_bin_m = max(
            0.0,
            float(self.mode_preserving_height_bin_m),
        )
        self.mode_preserving_high_surface_extra_particles = max(
            0,
            int(self.mode_preserving_high_surface_extra_particles),
        )
        self.mode_preserving_high_surface_z_fraction = float(
            np.clip(float(self.mode_preserving_high_surface_z_fraction), 0.0, 1.0)
        )
        self.mode_preserving_support_score_weight = max(
            0.0,
            float(self.mode_preserving_support_score_weight),
        )
        self.mode_preserving_tentative_boost = max(
            1.0,
            float(self.mode_preserving_tentative_boost),
        )
        self.mode_preserving_residual_boost = max(
            1.0,
            float(self.mode_preserving_residual_boost),
        )
        self.mode_preserving_cardinality_strata = bool(
            self.mode_preserving_cardinality_strata
        )
        self.mode_preserving_min_particles_per_cardinality = max(
            0,
            int(self.mode_preserving_min_particles_per_cardinality),
        )
        self.mode_preserving_report_cardinality_strata = bool(
            self.mode_preserving_report_cardinality_strata
        )
        self.mode_preserving_report_cardinality_extra_particles = max(
            0,
            int(self.mode_preserving_report_cardinality_extra_particles),
        )
        self.mode_preserving_dynamic_cardinality_allocation = bool(
            self.mode_preserving_dynamic_cardinality_allocation
        )
        self.mode_preserving_dynamic_cardinality_extra_particles = max(
            0,
            int(self.mode_preserving_dynamic_cardinality_extra_particles),
        )
        self.mode_preserving_dynamic_cardinality_min_mass = max(
            0.0,
            float(self.mode_preserving_dynamic_cardinality_min_mass),
        )
        self.mode_preserving_dynamic_cardinality_entropy_min = max(
            0.0,
            float(self.mode_preserving_dynamic_cardinality_entropy_min),
        )
        self.mode_preserving_dynamic_spatial_allocation = bool(
            self.mode_preserving_dynamic_spatial_allocation
        )
        self.mode_preserving_dynamic_spatial_extra_particles = max(
            0,
            int(self.mode_preserving_dynamic_spatial_extra_particles),
        )
        self.mode_preserving_dynamic_spatial_min_score_fraction = max(
            0.0,
            float(self.mode_preserving_dynamic_spatial_min_score_fraction),
        )
        if isinstance(self.source_position_prior, bool):
            prior = "surface" if self.source_position_prior else "volume"
        else:
            prior = str(self.source_position_prior).strip().lower()
        if prior in {"surface_constrained", "surface-constrained", "surfaces"}:
            prior = "surface"
        if prior not in {"volume", "surface"}:
            raise ValueError("source_position_prior must be 'volume' or 'surface'.")
        self.source_position_prior = prior
        self.surface_rejuvenation_enable = bool(self.surface_rejuvenation_enable)
        if (
            bool(self.birth_enable)
            and self.max_sources is not None
            and int(self.max_sources) > 1
        ):
            self.use_clustered_output = True
        if self.birth_jitter_topk_candidates is not None:
            self.birth_jitter_topk_candidates = max(
                1,
                int(self.birth_jitter_topk_candidates),
            )
        self.birth_global_rescue_enable = bool(self.birth_global_rescue_enable)
        self.birth_global_rescue_max_candidates = max(
            0,
            int(self.birth_global_rescue_max_candidates),
        )
        self.birth_global_rescue_min_residual_fraction = max(
            0.0,
            float(self.birth_global_rescue_min_residual_fraction),
        )
        self.birth_global_rescue_dedup_radius_m = max(
            0.0,
            float(self.birth_global_rescue_dedup_radius_m),
        )
        self.birth_global_rescue_force_proposal_on_gate = bool(
            self.birth_global_rescue_force_proposal_on_gate
        )
        self.birth_global_rescue_forced_min_delta_ll = float(
            self.birth_global_rescue_forced_min_delta_ll
        )
        if self.birth_global_rescue_min_support is not None:
            self.birth_global_rescue_min_support = max(
                1,
                int(self.birth_global_rescue_min_support),
            )
        if self.birth_global_rescue_min_distinct_poses is not None:
            self.birth_global_rescue_min_distinct_poses = max(
                1,
                int(self.birth_global_rescue_min_distinct_poses),
            )
        if self.birth_global_rescue_min_distinct_stations is not None:
            self.birth_global_rescue_min_distinct_stations = max(
                1,
                int(self.birth_global_rescue_min_distinct_stations),
            )
        self.birth_global_rescue_candidate_memory_enable = bool(
            self.birth_global_rescue_candidate_memory_enable
        )
        self.birth_global_rescue_candidate_memory_decay = float(
            np.clip(
                float(self.birth_global_rescue_candidate_memory_decay),
                0.0,
                1.0,
            )
        )
        self.birth_global_rescue_candidate_memory_max_candidates = max(
            0,
            int(self.birth_global_rescue_candidate_memory_max_candidates),
        )
        self.birth_global_rescue_candidate_memory_min_retained = max(
            0,
            int(self.birth_global_rescue_candidate_memory_min_retained),
        )
        self.high_strength_split_enable = bool(self.high_strength_split_enable)
        self.high_strength_split_q_multiple = max(
            1.0,
            float(self.high_strength_split_q_multiple),
        )
        self.high_strength_split_offset_m = max(
            1.0e-6,
            float(self.high_strength_split_offset_m),
        )
        self.high_strength_split_candidate_count = max(
            1,
            int(self.high_strength_split_candidate_count),
        )
        self.report_best_so_far_enable = bool(self.report_best_so_far_enable)
        self.report_best_so_far_min_score_improvement = max(
            0.0,
            float(self.report_best_so_far_min_score_improvement),
        )
        self.report_best_so_far_final_min_measurement_fraction = float(
            np.clip(
                float(self.report_best_so_far_final_min_measurement_fraction),
                0.0,
                1.0,
            )
        )
        self.report_best_so_far_final_require_resolved = bool(
            self.report_best_so_far_final_require_resolved
        )
        self.runtime_report_rescue_enable = bool(self.runtime_report_rescue_enable)
        self.runtime_report_rescue_particle_fraction = float(
            np.clip(float(self.runtime_report_rescue_particle_fraction), 0.0, 1.0)
        )
        self.runtime_report_rescue_min_particles_per_source = max(
            1,
            int(self.runtime_report_rescue_min_particles_per_source),
        )
        self.runtime_report_rescue_weight = float(
            np.clip(float(self.runtime_report_rescue_weight), 0.0, 0.5)
        )
        self.runtime_report_rescue_jitter_sigma_m = max(
            0.0,
            float(self.runtime_report_rescue_jitter_sigma_m),
        )
        self.runtime_report_rescue_quarantine_enable = bool(
            self.runtime_report_rescue_quarantine_enable
        )
        self.runtime_report_rescue_quarantine_weight = float(
            np.clip(float(self.runtime_report_rescue_quarantine_weight), 0.0, 0.5)
        )
        self.runtime_report_rescue_candidate_weight = float(
            np.clip(float(self.runtime_report_rescue_candidate_weight), 0.0, 0.5)
        )
        self.runtime_report_rescue_memory_enable = bool(
            self.runtime_report_rescue_memory_enable
        )
        self.runtime_report_rescue_memory_decay = float(
            np.clip(float(self.runtime_report_rescue_memory_decay), 0.0, 1.0)
        )
        self.runtime_report_rescue_memory_max_sources = max(
            0,
            int(self.runtime_report_rescue_memory_max_sources),
        )
        self.all_history_dictionary_proposal_enable = bool(
            self.all_history_dictionary_proposal_enable
        )
        self.all_history_dictionary_proposal_weight = float(
            np.clip(float(self.all_history_dictionary_proposal_weight), 0.0, 0.5)
        )
        self.all_history_dictionary_proposal_max_candidates = max(
            0,
            int(self.all_history_dictionary_proposal_max_candidates),
        )
        self.candidate_verification_queue_enable = bool(
            self.candidate_verification_queue_enable
        )
        self.candidate_verification_queue_weight = float(
            np.clip(float(self.candidate_verification_queue_weight), 0.0, 0.5)
        )
        self.candidate_verification_queue_decay = float(
            np.clip(float(self.candidate_verification_queue_decay), 0.0, 1.0)
        )
        self.candidate_verification_queue_max_sources = max(
            0,
            int(self.candidate_verification_queue_max_sources),
        )
        self.report_mle_rescue_surface_quota_enable = bool(
            self.report_mle_rescue_surface_quota_enable
        )
        self.report_mle_rescue_surface_quota_min_score_fraction = max(
            0.0,
            float(self.report_mle_rescue_surface_quota_min_score_fraction),
        )
        self.report_mle_rescue_surface_quota_per_stratum = max(
            1,
            int(self.report_mle_rescue_surface_quota_per_stratum),
        )
        self.report_mle_rescue_spatial_quota_enable = bool(
            self.report_mle_rescue_spatial_quota_enable
        )
        self.report_mle_rescue_spatial_quota_tile_m = max(
            1.0e-6,
            float(self.report_mle_rescue_spatial_quota_tile_m),
        )
        self.report_mle_rescue_spatial_quota_per_tile = max(
            1,
            int(self.report_mle_rescue_spatial_quota_per_tile),
        )
        self.residual_decomposition_enable = bool(self.residual_decomposition_enable)
        self.peak_suppression_enable = bool(self.peak_suppression_enable)
        self.peak_suppression_min_source_fraction = float(
            np.clip(float(self.peak_suppression_min_source_fraction), 0.0, 1.0)
        )
        self.peak_suppression_factor = float(
            np.clip(float(self.peak_suppression_factor), 0.0, 1.0)
        )
        self.residual_decomposition_max_layers = max(
            1,
            int(self.residual_decomposition_max_layers),
        )
        self.pseudo_source_verification_enable = bool(
            self.pseudo_source_verification_enable
        )
        self.pseudo_source_min_delta_ll = float(self.pseudo_source_min_delta_ll)
        self.pseudo_source_min_distinct_views = max(
            1,
            int(self.pseudo_source_min_distinct_views),
        )
        self.pseudo_source_fail_grace_stations = max(
            0,
            int(self.pseudo_source_fail_grace_stations),
        )
        self.pseudo_source_corr_max = float(
            np.clip(float(self.pseudo_source_corr_max), 0.0, 1.0)
        )
        self.pseudo_source_temporal_sep_min = max(
            0.0,
            float(self.pseudo_source_temporal_sep_min),
        )
        self.pseudo_source_quarantine_on_suppress = bool(
            self.pseudo_source_quarantine_on_suppress
        )
        self.pseudo_source_quarantine_excludes_runtime = bool(
            self.pseudo_source_quarantine_excludes_runtime
        )
        self.report_exclude_unverified_sources = bool(
            self.report_exclude_unverified_sources
        )
        self.source_prune_min_distinct_stations = max(
            1,
            int(self.source_prune_min_distinct_stations),
        )
        self.source_prune_min_distinct_views = max(
            1,
            int(self.source_prune_min_distinct_views),
        )
        self.source_prune_fail_grace_stations = max(
            1,
            int(self.source_prune_fail_grace_stations),
        )
        self.source_prune_delta_ll_threshold = float(
            self.source_prune_delta_ll_threshold
        )
        self.source_prune_refit_after_remove = bool(
            self.source_prune_refit_after_remove
        )
        self.source_prune_bic_penalty_params = max(
            0,
            int(self.source_prune_bic_penalty_params),
        )
        self.weak_source_prune_min_expected_count = max(
            0.0,
            float(self.weak_source_prune_min_expected_count),
        )
        self.weak_source_prune_min_fraction = max(
            0.0,
            float(self.weak_source_prune_min_fraction),
        )
        self.weak_source_prune_min_age = max(0, int(self.weak_source_prune_min_age))
        self.weak_source_prune_require_observable = bool(
            self.weak_source_prune_require_observable
        )
        self.weak_source_prune_min_observable_measurements = max(
            1,
            int(self.weak_source_prune_min_observable_measurements),
        )
        self.weak_source_prune_observable_count = max(
            0.0,
            float(self.weak_source_prune_observable_count),
        )
        self.weak_source_prune_observable_fraction = max(
            0.0,
            float(self.weak_source_prune_observable_fraction),
        )
        self.weak_source_prune_visibility_reference_strength = max(
            0.0,
            float(self.weak_source_prune_visibility_reference_strength),
        )
        self.birth_residual_always_try = bool(self.birth_residual_always_try)
        self.birth_residual_expand_structural_particles = bool(
            self.birth_residual_expand_structural_particles
        )
        self.birth_residual_acceptance_complexity_scale = float(
            np.clip(
                float(self.birth_residual_acceptance_complexity_scale),
                0.0,
                1.0,
            )
        )
        self.birth_residual_suppress_death = bool(self.birth_residual_suppress_death)
        if self.birth_max_per_update is not None:
            self.birth_max_per_update = max(0, int(self.birth_max_per_update))
        self.birth_delta_ll_threshold = float(self.birth_delta_ll_threshold)
        self.birth_complexity_penalty = max(0.0, float(self.birth_complexity_penalty))
        self.birth_bic_penalty_params = max(0, int(self.birth_bic_penalty_params))
        self.conditional_strength_refit_window = max(
            0,
            int(self.conditional_strength_refit_window),
        )
        self.conditional_strength_refit_iters = max(
            1,
            int(self.conditional_strength_refit_iters),
        )
        self.conditional_strength_refit_reweight_clip = max(
            0.0,
            float(self.conditional_strength_refit_reweight_clip),
        )
        self.conditional_strength_refit_min_count = max(
            0.0,
            float(self.conditional_strength_refit_min_count),
        )
        self.conditional_strength_refit_min_snr = max(
            0.0,
            float(self.conditional_strength_refit_min_snr),
        )
        self.conditional_strength_refit_prior_weight = max(
            0.0,
            float(self.conditional_strength_refit_prior_weight),
        )
        self.conditional_strength_refit_prior_rel_sigma = max(
            1.0e-6,
            float(self.conditional_strength_refit_prior_rel_sigma),
        )
        self.source_strength_prior_mean = max(
            0.0,
            float(self.source_strength_prior_mean),
        )
        self.source_strength_prior_weight = max(
            0.0,
            float(self.source_strength_prior_weight),
        )
        self.source_strength_prior_rel_sigma = max(
            1.0e-6,
            float(self.source_strength_prior_rel_sigma),
        )
        self.source_strength_absorption_penalty_weight = max(
            0.0,
            float(self.source_strength_absorption_penalty_weight),
        )
        self.source_strength_absorption_q_multiple = max(
            1.0,
            float(self.source_strength_absorption_q_multiple),
        )
        self.source_strength_observation_overshoot_penalty_weight = max(
            0.0,
            float(self.source_strength_observation_overshoot_penalty_weight),
        )
        self.source_strength_observation_overshoot_sigma = max(
            0.0,
            float(self.source_strength_observation_overshoot_sigma),
        )
        self.source_strength_observation_overshoot_quantile = float(
            np.clip(
                float(self.source_strength_observation_overshoot_quantile),
                0.0,
                1.0,
            )
        )
        self.source_strength_observation_overshoot_min_visible_fraction = max(
            0.0,
            float(self.source_strength_observation_overshoot_min_visible_fraction),
        )
        self.source_strength_observation_overshoot_min_visible_measurements = max(
            1,
            int(self.source_strength_observation_overshoot_min_visible_measurements),
        )
        self.birth_stage_single_station_as_quarantine = bool(
            self.birth_stage_single_station_as_quarantine
        )
        self.report_strength_refit_iters = max(1, int(self.report_strength_refit_iters))
        self.report_strength_refit_eps = max(
            1.0e-15,
            float(self.report_strength_refit_eps),
        )
        self.report_strength_refit_use_all_measurements = bool(
            self.report_strength_refit_use_all_measurements
        )
        self.report_strength_refit_preserve_cardinality = bool(
            self.report_strength_refit_preserve_cardinality
        )
        self.report_strength_refit_prior_weight = max(
            0.0,
            float(self.report_strength_refit_prior_weight),
        )
        self.report_strength_refit_prior_rel_sigma = max(
            1.0e-6,
            float(self.report_strength_refit_prior_rel_sigma),
        )
        self.report_strength_absorption_penalty_weight = max(
            0.0,
            float(self.report_strength_absorption_penalty_weight),
        )
        self.report_strength_absorption_q_multiple = max(
            1.0,
            float(self.report_strength_absorption_q_multiple),
        )
        self.report_strength_observation_overshoot_penalty_weight = max(
            0.0,
            float(self.report_strength_observation_overshoot_penalty_weight),
        )
        self.report_strength_observation_overshoot_sigma = max(
            0.0,
            float(self.report_strength_observation_overshoot_sigma),
        )
        self.report_strength_observation_overshoot_quantile = float(
            np.clip(
                float(self.report_strength_observation_overshoot_quantile),
                0.0,
                1.0,
            )
        )
        self.report_strength_observation_overshoot_min_visible_fraction = max(
            0.0,
            float(self.report_strength_observation_overshoot_min_visible_fraction),
        )
        self.report_strength_observation_overshoot_min_visible_measurements = max(
            1,
            int(self.report_strength_observation_overshoot_min_visible_measurements),
        )
        self.report_mle_rescue_enable = bool(self.report_mle_rescue_enable)
        self.report_mle_rescue_max_candidates = max(
            1,
            int(self.report_mle_rescue_max_candidates),
        )
        self.report_mle_rescue_max_posterior_candidates = max(
            0,
            int(self.report_mle_rescue_max_posterior_candidates),
        )
        self.report_mle_rescue_max_residual_candidates = max(
            0,
            int(self.report_mle_rescue_max_residual_candidates),
        )
        self.report_surface_local_refine = bool(self.report_surface_local_refine)
        self.report_surface_local_refine_radius_m = max(
            0.0,
            float(self.report_surface_local_refine_radius_m),
        )
        self.report_surface_local_refine_grid_steps = max(
            0,
            int(self.report_surface_local_refine_grid_steps),
        )
        self.report_surface_local_refine_max_candidates_per_source = max(
            1,
            int(self.report_surface_local_refine_max_candidates_per_source),
        )
        self.report_surface_local_refine_max_sources = max(
            0,
            int(self.report_surface_local_refine_max_sources),
        )
        self.report_surface_local_refine_min_ll_gain = max(
            0.0,
            float(self.report_surface_local_refine_min_ll_gain),
        )
        self.report_mle_rescue_dedup_radius_m = max(
            0.0,
            float(self.report_mle_rescue_dedup_radius_m),
        )
        self.report_mle_rescue_min_residual_fraction = max(
            0.0,
            float(self.report_mle_rescue_min_residual_fraction),
        )
        self.report_mle_rescue_visibility_weight = float(
            np.clip(float(self.report_mle_rescue_visibility_weight), 0.0, 1.0)
        )
        self.report_mle_rescue_min_visible_measurements = max(
            1,
            int(self.report_mle_rescue_min_visible_measurements),
        )
        self.report_mle_rescue_visible_count = max(
            0.0,
            float(self.report_mle_rescue_visible_count),
        )
        self.report_mle_rescue_visibility_reference_strength = max(
            0.0,
            float(self.report_mle_rescue_visibility_reference_strength),
        )
        self.report_cluster_model_selection = bool(self.report_cluster_model_selection)
        self.report_cluster_bic_penalty_params = max(
            0,
            int(self.report_cluster_bic_penalty_params),
        )
        self.report_cluster_delta_ll_threshold = float(
            self.report_cluster_delta_ll_threshold
        )
        self.report_cluster_model_selection_max_candidates = max(
            1,
            int(self.report_cluster_model_selection_max_candidates),
        )
        self.report_model_order_require_posterior_match = bool(
            self.report_model_order_require_posterior_match
        )
        self.report_model_order_prune_particles = bool(
            self.report_model_order_prune_particles
        )
        self.report_model_order_particle_prune_radius_m = max(
            0.0,
            float(self.report_model_order_particle_prune_radius_m),
        )
        self.report_model_order_min_bic_margin = max(
            0.0,
            float(self.report_model_order_min_bic_margin),
        )
        self.report_model_order_zero_source_min_bic_margin = max(
            0.0,
            float(self.report_model_order_zero_source_min_bic_margin),
        )
        self.report_model_order_condition_max = max(
            0.0,
            float(self.report_model_order_condition_max),
        )
        self.report_model_order_corr_penalty_weight = max(
            0.0,
            float(self.report_model_order_corr_penalty_weight),
        )
        self.report_model_order_corr_penalty_threshold = float(
            np.clip(float(self.report_model_order_corr_penalty_threshold), 0.0, 1.0)
        )
        self.report_model_order_corr_penalty_power = max(
            1.0e-6,
            float(self.report_model_order_corr_penalty_power),
        )
        self.report_model_order_subset_corr_prune_threshold = float(
            np.clip(
                float(self.report_model_order_subset_corr_prune_threshold),
                0.0,
                1.0,
            )
        )
        self.report_model_order_underfit_gate = bool(
            self.report_model_order_underfit_gate
        )
        self.report_model_order_underfit_min_residual_fraction = float(
            self.report_model_order_underfit_min_residual_fraction
        )
        self.report_model_order_underfit_min_positive_chi2 = max(
            0.0,
            float(self.report_model_order_underfit_min_positive_chi2),
        )
        self.report_model_order_workers = max(
            1,
            int(self.report_model_order_workers),
        )
        self.report_model_order_parallel_min_subsets = max(
            1,
            int(self.report_model_order_parallel_min_subsets),
        )
        self.sparse_poisson_evidence_enable = bool(
            self.sparse_poisson_evidence_enable
        )
        self.sparse_poisson_evidence_authoritative = bool(
            self.sparse_poisson_evidence_authoritative
        )
        self.sparse_poisson_evidence_candidate_limit = max(
            0,
            int(self.sparse_poisson_evidence_candidate_limit),
        )
        self.sparse_poisson_evidence_refit_iters = max(
            1,
            int(self.sparse_poisson_evidence_refit_iters),
        )
        self.sparse_poisson_evidence_holdout_stride = max(
            0,
            int(self.sparse_poisson_evidence_holdout_stride),
        )
        self.sparse_poisson_evidence_parameter_count_per_source = max(
            0,
            int(self.sparse_poisson_evidence_parameter_count_per_source),
        )
        self.sparse_poisson_evidence_min_bic_margin = max(
            0.0,
            float(self.sparse_poisson_evidence_min_bic_margin),
        )
        self.sparse_poisson_evidence_min_distinct_stations = max(
            1,
            int(self.sparse_poisson_evidence_min_distinct_stations),
        )
        self.sparse_poisson_evidence_corr_prune_threshold = float(
            np.clip(float(self.sparse_poisson_evidence_corr_prune_threshold), 0.0, 1.0)
        )
        self.sparse_poisson_evidence_max_response_correlation = max(
            0.0,
            float(self.sparse_poisson_evidence_max_response_correlation),
        )
        self.sparse_poisson_evidence_condition_max = max(
            0.0,
            float(self.sparse_poisson_evidence_condition_max),
        )
        self.sparse_poisson_spectral_evidence_enable = bool(
            self.sparse_poisson_spectral_evidence_enable
        )
        self.sparse_poisson_spectral_evidence_primary = bool(
            self.sparse_poisson_spectral_evidence_primary
        )
        self.sparse_poisson_spectral_nuisance_enable = bool(
            self.sparse_poisson_spectral_nuisance_enable
        )
        self.sparse_poisson_joint_evidence_enable = bool(
            self.sparse_poisson_joint_evidence_enable
        )
        self.sparse_poisson_offgrid_refine_enable = bool(
            self.sparse_poisson_offgrid_refine_enable
        )
        self.sparse_poisson_offgrid_refine_radius_m = max(
            0.0,
            float(self.sparse_poisson_offgrid_refine_radius_m),
        )
        self.sparse_poisson_offgrid_refine_max_iter = max(
            1,
            int(self.sparse_poisson_offgrid_refine_max_iter),
        )
        self.sparse_poisson_offgrid_refine_min_ll_gain = max(
            0.0,
            float(self.sparse_poisson_offgrid_refine_min_ll_gain),
        )
        self.sparse_poisson_ambiguity_report_enable = bool(
            self.sparse_poisson_ambiguity_report_enable
        )
        self.sparse_poisson_ambiguity_corr_threshold = float(
            np.clip(float(self.sparse_poisson_ambiguity_corr_threshold), 0.0, 1.0)
        )
        self.sparse_poisson_ambiguity_bic_gap_threshold = max(
            0.0,
            float(self.sparse_poisson_ambiguity_bic_gap_threshold),
        )
        self.sparse_poisson_ambiguity_condition_max = max(
            0.0,
            float(self.sparse_poisson_ambiguity_condition_max),
        )
        self.runtime_report_rescue_verification_queue_only = bool(
            self.runtime_report_rescue_verification_queue_only
        )
        if self.runtime_report_rescue_verification_queue_only:
            self.candidate_verification_queue_enable = True
        self.report_pre_finalize_guard = bool(self.report_pre_finalize_guard)
        self.history_estimate_interval = max(0, int(self.history_estimate_interval))
        self.candidate_response_cache_max_entries = max(
            0,
            int(self.candidate_response_cache_max_entries),
        )
        self.converge_cardinality_var_max = max(
            0.0,
            float(self.converge_cardinality_var_max),
        )
        self.converge_require_no_tentative = bool(self.converge_require_no_tentative)
        self.converge_freeze_updates = bool(self.converge_freeze_updates)
        self.converge_min_stations = max(0, int(self.converge_min_stations))
        self.split_residual_guided = bool(self.split_residual_guided)
        self.split_complexity_penalty = max(0.0, float(self.split_complexity_penalty))
        self.split_residual_always_try = bool(self.split_residual_always_try)
        self.split_residual_candidate_count = max(
            1,
            int(self.split_residual_candidate_count),
        )
        self.merge_response_corr_min = float(
            np.clip(float(self.merge_response_corr_min), 0.0, 1.0)
        )
        self.merge_search_topk_pairs = max(1, int(self.merge_search_topk_pairs))
        self.structural_trial_workers = max(1, int(self.structural_trial_workers))
        self.structural_trial_parallel_min_trials = max(
            1,
            int(self.structural_trial_parallel_min_trials),
        )
        self.parallel_isotope_updates = bool(self.parallel_isotope_updates)
        if self.parallel_isotope_workers is not None:
            self.parallel_isotope_workers = max(1, int(self.parallel_isotope_workers))


@dataclass(frozen=True)
class MeasurementRecord:
    """Store a single isotope-wise measurement and metadata."""

    z_k: Dict[str, float]
    pose_idx: int
    orient_idx: int
    live_time_s: float
    fe_index: int | None = None
    pb_index: int | None = None
    z_variance_k: Dict[str, float] | None = None
    z_covariance_k: Dict[str, Dict[str, float]] | None = None
    ig_value: float | None = None
    spectrum_counts: tuple[float, ...] | None = None
    spectrum_variance: tuple[float, ...] | None = None
    spectrum_background: tuple[float, ...] | None = None
    spectrum_response_templates_by_isotope: Dict[str, tuple[float, ...]] | None = None


@dataclass(frozen=True)
class ReportSnapshot:
    """Store a station-level report model for best-so-far final reporting."""

    label: str
    measurement_count: int
    estimates: Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]
    diagnostics: Dict[str, Dict[str, Any]]
    unresolved: Dict[str, Any]
    score_tuple: Tuple[float, ...]
    components: Dict[str, float]


class RotatingShieldPFEstimator:
    """
    Online source estimator using parallel PFs with shield rotation (Sec. 3.4–3.6).

    - Maintains one PF per isotope.
    - Updates each PF with pose/orientation and Poisson weight updates.
    """

    def __init__(
        self,
        isotopes: Sequence[str],
        candidate_sources: NDArray[np.float64],
        shield_normals: NDArray[np.float64] | None,
        mu_by_isotope: Dict[str, object] | None,
        pf_config: RotatingShieldPFConfig | None = None,
        shield_params: ShieldParams | None = None,
        obstacle_grid: ObstacleGrid | None = None,
        obstacle_height_m: float = 2.0,
        obstacle_mu_by_isotope: Dict[str, float] | None = None,
        obstacle_buildup_coeff: float = 0.0,
        detector_radius_m: float = 0.0,
        detector_aperture_radius_m: float | None = None,
        detector_aperture_samples: int = 1,
        detector_aperture_sampling: str = "solid_angle_cone",
        source_extent_radius_m: float = 0.0,
        source_extent_samples: int = 1,
        line_mu_by_isotope: Dict[str, object] | None = None,
        transport_response_model: Dict[str, object] | None = None,
    ) -> None:
        """Initialize per-isotope filters and shared measurement-model state."""
        self.all_isotopes = list(isotopes)
        self.isotopes = list(isotopes)
        self.pf_config = pf_config or RotatingShieldPFConfig()
        self.shield_params = shield_params or ShieldParams()
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
        # Measurement poses are appended incrementally.
        self.poses: List[NDArray[np.float64]] = []
        if shield_normals is None:
            from measurement.shielding import generate_octant_orientations

            self.normals = generate_octant_orientations()
        else:
            self.normals = shield_normals
        self.mu_by_isotope = self._resolve_mu_by_isotope(mu_by_isotope)
        self.kernel_cache: KernelPrecomputer | None = None
        self.filters: Dict[str, IsotopeParticleFilter] = {}
        self.candidate_sources = candidate_sources
        self.history_estimates: List[
            Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]
        ] = []
        self.history_scores: List[float] = []
        self.measurements: List[MeasurementRecord] = []
        self.last_strength_prior_diagnostics: Dict[str, Dict[str, float]] = {}
        self._defer_resample_birth = False
        self._deferred_measurement_count = 0
        self._previous_deferred_measurement_count = 0
        self._pre_finalize_guard_estimates: Dict[
            str,
            Tuple[NDArray[np.float64], NDArray[np.float64]],
        ] = {}
        self._last_report_model_order_diagnostics: Dict[str, Dict[str, Any]] = {}
        self._last_sparse_poisson_evidence_diagnostics: Dict[
            str,
            Dict[str, Any],
        ] = {}
        self._last_joint_sparse_poisson_evidence_diagnostics: Dict[str, Any] = {}
        self._runtime_report_rescue_modes: Dict[
            str,
            Tuple[NDArray[np.float64], NDArray[np.float64], float],
        ] = {}
        self._last_planning_surface_rescue_mode_counts: Dict[str, int] = {}
        self._runtime_report_rescue_memory: Dict[
            str,
            Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        ] = {}
        self._candidate_verification_queue: Dict[
            str,
            Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        ] = {}
        self._global_birth_rescue_candidate_memory: Dict[
            str,
            Tuple[NDArray[np.float64], NDArray[np.float64]],
        ] = {}
        self._runtime_global_birth_quarantine_stats: Dict[
            str, Dict[str, float | int]
        ] = {}
        self._report_snapshots: list[ReportSnapshot] = []
        self._best_report_snapshot: ReportSnapshot | None = None
        self._last_final_report_selection: dict[str, Any] = {}
        self.last_pair_sequence_update_workers = 1
        self.last_pair_sequence_update_wall_s = 0.0
        self.last_pair_sequence_stage_wall_s: Dict[str, float] = {}
        self.last_structural_update_workers = 1
        self.last_structural_update_wall_s = 0.0
        self.last_sparse_poisson_refresh_wall_s = 0.0
        self.last_sparse_poisson_refresh_stage_wall_s: Dict[str, float] = {}
        self._report_cache_revision = 0
        self._report_estimate_cache: dict[
            tuple[int, bool],
            Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]],
        ] = {}
        self._candidate_response_cache: dict[
            tuple[Any, ...],
            NDArray[np.float64],
        ] = {}
        self._candidate_response_cache_order: list[tuple[Any, ...]] = []
        self._candidate_response_prefix_cache: dict[
            tuple[Any, ...],
            dict[str, NDArray[np.float64]],
        ] = {}
        self._candidate_response_prefix_cache_order: list[tuple[Any, ...]] = []

    @staticmethod
    def _copy_estimate_map(
        estimates: Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]],
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Return a deep array copy of a per-isotope estimate mapping."""
        return {
            isotope: (
                np.asarray(positions, dtype=float).copy(),
                np.asarray(strengths, dtype=float).copy(),
            )
            for isotope, (positions, strengths) in estimates.items()
        }

    def _invalidate_report_cache(self) -> None:
        """Invalidate cached report estimates after any PF/report state change."""
        self._report_cache_revision += 1
        self._report_estimate_cache.clear()

    def _record_history_estimate(self, measurement_count: int) -> None:
        """Record an exact report estimate when the configured history stride allows it."""
        interval = max(
            0,
            int(getattr(self.pf_config, "history_estimate_interval", 1)),
        )
        if interval <= 0:
            return
        count = max(0, int(measurement_count))
        if count <= 0 or count % interval != 0:
            return
        self.history_estimates.append(self.estimates())

    def _report_snapshot_components(
        self,
        estimates: Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]],
        diagnostics: Dict[str, Dict[str, Any]],
        unresolved: Dict[str, Any],
    ) -> Dict[str, float]:
        """Return lexicographic best-report diagnostics for one snapshot."""
        unready_count = 0
        underfit_count = 0
        source_excess_count = 0
        model_count_mismatch = 0
        residual_fraction_sum = 0.0
        negative_residual_fraction_sum = 0.0
        overprediction_fraction_sum = 0.0
        positive_chi2_sum = 0.0
        max_response_corr = 0.0
        log_condition_sum = 0.0
        total_sources = 0
        report_source_cap = self._report_max_sources_per_isotope()
        for isotope in self.all_isotopes:
            stats = diagnostics.get(str(isotope), {})
            if isinstance(stats, dict) and not bool(
                stats.get("model_order_ready", False)
            ):
                unready_count += 1
            residual_fraction = 0.0
            if isinstance(stats, dict):
                residual_fraction = max(
                    0.0,
                    float(stats.get("selected_positive_residual_fraction", 0.0)),
                )
                negative_residual_fraction_sum += max(
                    0.0,
                    float(stats.get("selected_negative_residual_fraction", 0.0)),
                )
                overprediction_fraction_sum += max(
                    0.0,
                    float(stats.get("selected_overprediction_fraction", 0.0)),
                )
                positive_chi2_sum += max(
                    0.0,
                    float(stats.get("selected_positive_residual_chi2", 0.0)),
                )
                corr = float(stats.get("selected_max_response_correlation", 0.0))
                if np.isfinite(corr):
                    max_response_corr = max(max_response_corr, max(0.0, corr))
                condition = float(stats.get("condition_number", 0.0))
                if np.isfinite(condition) and condition > 0.0:
                    log_condition_sum += float(np.log1p(condition))
                min_fraction = max(
                    float(self.pf_config.report_mle_rescue_min_residual_fraction),
                    0.0,
                )
                if residual_fraction >= min_fraction and min_fraction > 0.0:
                    underfit_count += 1
            residual_fraction_sum += residual_fraction
            pos_arr = np.asarray(
                estimates.get(
                    str(isotope),
                    (np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)),
                )[0],
                dtype=float,
            ).reshape(-1, 3)
            estimate_count = int(pos_arr.shape[0])
            total_sources += estimate_count
            source_excess_count += max(0, estimate_count - report_source_cap)
            if isinstance(stats, dict):
                selected_count = int(stats.get("selected_count", estimate_count))
                model_count_mismatch += abs(estimate_count - selected_count)
        unresolved_count = int(len(unresolved))
        score_tuple = (
            float(unready_count),
            float(unresolved_count),
            float(source_excess_count),
            float(model_count_mismatch),
            float(underfit_count),
            float(overprediction_fraction_sum),
            float(negative_residual_fraction_sum),
            float(residual_fraction_sum),
            float(positive_chi2_sum),
            float(max_response_corr),
            float(log_condition_sum),
            float(total_sources),
        )
        return {
            "unready_count": float(unready_count),
            "unresolved_count": float(unresolved_count),
            "source_excess_count": float(source_excess_count),
            "model_count_mismatch": float(model_count_mismatch),
            "underfit_count": float(underfit_count),
            "residual_fraction_sum": float(residual_fraction_sum),
            "negative_residual_fraction_sum": float(negative_residual_fraction_sum),
            "overprediction_fraction_sum": float(overprediction_fraction_sum),
            "positive_chi2_sum": float(positive_chi2_sum),
            "max_response_correlation": float(max_response_corr),
            "log_condition_sum": float(log_condition_sum),
            "total_sources": float(total_sources),
            **{f"score_tuple_{idx}": value for idx, value in enumerate(score_tuple)},
        }

    def _report_snapshot_score_tuple(
        self,
        components: Dict[str, float],
    ) -> Tuple[float, ...]:
        """Return the lexicographic report quality tuple for snapshot ranking."""
        return tuple(float(components[f"score_tuple_{idx}"]) for idx in range(12))

    def _report_max_sources_per_isotope(self) -> int:
        """Return the report-side source-count cap for one isotope."""
        configured = self.pf_config.max_sources
        if configured is None:
            return int(DEFAULT_MAX_SOURCES_PER_ISOTOPE)
        return max(0, int(configured))

    def _report_snapshot_limit_for_isotope(
        self,
        isotope: str,
        diagnostics: Dict[str, Dict[str, Any]],
    ) -> int:
        """Return a conservative report snapshot source limit for one isotope."""
        cap = self._report_max_sources_per_isotope()
        stats = diagnostics.get(str(isotope), {})
        if isinstance(stats, dict):
            selected_count = int(stats.get("selected_count", 0))
            if selected_count > 0:
                return min(cap, selected_count)
        return cap

    def _report_snapshot_is_better(
        self,
        candidate: ReportSnapshot,
        reference: ReportSnapshot | None,
    ) -> bool:
        """Return True when a candidate report is better than the reference."""
        if reference is None:
            return True
        eps = float(self.pf_config.report_best_so_far_min_score_improvement)
        for cand_value, ref_value in zip(
            candidate.score_tuple,
            reference.score_tuple,
        ):
            cand = float(cand_value)
            ref = float(ref_value)
            if cand < ref - eps:
                return True
            if cand > ref + eps:
                return False
        return int(candidate.measurement_count) > int(reference.measurement_count)

    def _report_snapshot_is_recent_enough(
        self,
        candidate: ReportSnapshot,
        reference_measurement_count: int,
    ) -> tuple[bool, str]:
        """Return whether a best-so-far snapshot is recent enough to reuse."""
        current_count = max(0, int(reference_measurement_count))
        if current_count <= 0:
            return True, "no_reference_measurements"
        candidate_count = max(0, int(candidate.measurement_count))
        min_fraction = float(
            self.pf_config.report_best_so_far_final_min_measurement_fraction
        )
        if min_fraction <= 0.0:
            return True, "measurement_fraction_disabled"
        fraction = float(candidate_count) / float(current_count)
        if fraction + 1.0e-12 < min_fraction:
            return (
                False,
                (
                    "insufficient_measurement_fraction:"
                    f"{fraction:.6g}<{min_fraction:.6g}"
                ),
            )
        return True, "recent_enough"

    def _report_snapshot_is_resolved_enough(
        self,
        candidate: ReportSnapshot,
    ) -> tuple[bool, str]:
        """Return whether a best-so-far snapshot is structurally resolved."""
        if not bool(self.pf_config.report_best_so_far_final_require_resolved):
            return True, "resolved_requirement_disabled"
        component_keys = (
            "unready_count",
            "unresolved_count",
            "source_excess_count",
            "model_count_mismatch",
            "underfit_count",
        )
        for key in component_keys:
            value = float(candidate.components.get(key, 0.0))
            if value > 1.0e-12:
                return False, f"{key}_nonzero:{value:.6g}"
        return True, "resolved_enough"

    def _report_snapshot_final_eligible(
        self,
        candidate: ReportSnapshot,
        current: ReportSnapshot,
    ) -> tuple[bool, str]:
        """Return whether a snapshot may replace the final current report."""
        recent, recent_reason = self._report_snapshot_is_recent_enough(
            candidate,
            int(current.measurement_count),
        )
        if not recent:
            return False, recent_reason
        resolved, resolved_reason = self._report_snapshot_is_resolved_enough(candidate)
        if not resolved:
            return False, resolved_reason
        return True, "eligible"

    def _best_report_snapshot_for_runtime(
        self,
    ) -> tuple[ReportSnapshot | None, str]:
        """Return a best-so-far snapshot that is safe to reuse online."""
        snapshot = self._best_report_snapshot
        if snapshot is None:
            return None, "missing"
        recent, reason = self._report_snapshot_is_recent_enough(
            snapshot,
            int(len(self.measurements)),
        )
        if not recent:
            return None, reason
        return snapshot, "eligible"

    def _lightweight_report_snapshot_estimates(
        self,
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Return station report candidates without running final-report rescue."""
        estimates: Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
        diagnostics = self._last_report_model_order_diagnostics
        dedup_radius = max(float(self.pf_config.report_mle_rescue_dedup_radius_m), 0.0)
        for isotope, filt in self.filters.items():
            rescue = self._runtime_report_rescue_modes.get(str(isotope))
            if rescue is not None:
                positions, strengths, _weight = rescue
                limit = self._report_snapshot_limit_for_isotope(
                    str(isotope),
                    diagnostics,
                )
                positions_arr = np.asarray(positions, dtype=float).reshape(-1, 3)
                strengths_arr = np.maximum(
                    np.asarray(strengths, dtype=float).reshape(-1),
                    0.0,
                )
                if limit <= 0:
                    positions_arr = np.zeros((0, 3), dtype=float)
                    strengths_arr = np.zeros(0, dtype=float)
                elif positions_arr.shape[0] > limit:
                    order = np.argsort(strengths_arr)[::-1]
                    positions_arr = positions_arr[order]
                    strengths_arr = strengths_arr[order]
                    positions_arr, strengths_arr = dedupe_report_candidates(
                        positions_arr,
                        strengths_arr,
                        radius_m=dedup_radius,
                        max_candidates=limit,
                    )
                estimates[str(isotope)] = (
                    positions_arr.copy(),
                    strengths_arr.copy(),
                )
                continue
            try:
                positions, strengths = filt.estimate()
            except RuntimeError:
                positions = np.zeros((0, 3), dtype=float)
                strengths = np.zeros(0, dtype=float)
            estimates[str(isotope)] = (
                np.asarray(positions, dtype=float).reshape(-1, 3).copy(),
                np.maximum(np.asarray(strengths, dtype=float).reshape(-1), 0.0).copy(),
            )
        return estimates

    def record_report_snapshot(
        self,
        label: str | None = None,
        *,
        estimates: Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]
        | None = None,
        allow_heavy_estimate: bool = True,
    ) -> ReportSnapshot | None:
        """Record the current BIC/report model for best-so-far diagnostics."""
        if not bool(self.pf_config.report_best_so_far_enable):
            return None
        if estimates is None:
            if allow_heavy_estimate:
                estimates = self.estimates()
            else:
                estimates = self._lightweight_report_snapshot_estimates()
        diagnostics = copy.deepcopy(self._last_report_model_order_diagnostics)
        try:
            unresolved = copy.deepcopy(self.unresolved_structural_evidence())
        except (RuntimeError, ValueError, AttributeError):
            unresolved = {}
        components = self._report_snapshot_components(
            estimates,
            diagnostics,
            unresolved,
        )
        snapshot = ReportSnapshot(
            label=str(label or f"measurement_{len(self.measurements)}"),
            measurement_count=int(len(self.measurements)),
            estimates=self._copy_estimate_map(estimates),
            diagnostics=diagnostics,
            unresolved=unresolved,
            score_tuple=self._report_snapshot_score_tuple(components),
            components=components,
        )
        self._report_snapshots.append(snapshot)
        if self._report_snapshot_is_better(snapshot, self._best_report_snapshot):
            self._best_report_snapshot = snapshot
        return snapshot

    def report_snapshot_progress(self) -> Dict[str, float | bool | int]:
        """Return whether the most recent report snapshot improved."""
        if len(self._report_snapshots) < 2:
            return {
                "available": False,
                "has_progress": False,
                "residual_improved": False,
                "model_order_improved": False,
                "score_improved": False,
            }
        previous = self._report_snapshots[-2]
        current = self._report_snapshots[-1]
        residual_prev = float(previous.components.get("residual_fraction_sum", 0.0))
        residual_cur = float(current.components.get("residual_fraction_sum", 0.0))
        residual_improved = residual_cur < residual_prev - 1.0e-9
        model_prev = (
            float(previous.components.get("unready_count", 0.0)),
            float(previous.components.get("unresolved_count", 0.0)),
            float(previous.components.get("underfit_count", 0.0)),
        )
        model_cur = (
            float(current.components.get("unready_count", 0.0)),
            float(current.components.get("unresolved_count", 0.0)),
            float(current.components.get("underfit_count", 0.0)),
        )
        model_order_improved = model_cur < model_prev
        score_improved = self._report_snapshot_is_better(current, previous)
        return {
            "available": True,
            "has_progress": bool(
                residual_improved or model_order_improved or score_improved
            ),
            "residual_improved": bool(residual_improved),
            "model_order_improved": bool(model_order_improved),
            "score_improved": bool(score_improved),
            "previous_measurement_count": int(previous.measurement_count),
            "current_measurement_count": int(current.measurement_count),
            "previous_residual_fraction_sum": float(residual_prev),
            "current_residual_fraction_sum": float(residual_cur),
            "previous_unready_count": int(model_prev[0]),
            "current_unready_count": int(model_cur[0]),
            "previous_unresolved_count": int(model_prev[1]),
            "current_unresolved_count": int(model_cur[1]),
        }

    def _snapshot_summary(self, snapshot: ReportSnapshot | None) -> dict[str, Any]:
        """Return a JSON-safe summary for a report snapshot."""
        if snapshot is None:
            return {}
        selected_counts = {
            str(isotope): int(
                np.asarray(
                    snapshot.estimates.get(
                        str(isotope),
                        (
                            np.zeros((0, 3), dtype=float),
                            np.zeros(0, dtype=float),
                        ),
                    )[0],
                    dtype=float,
                )
                .reshape(-1, 3)
                .shape[0]
            )
            for isotope in self.all_isotopes
        }
        return {
            "label": str(snapshot.label),
            "measurement_count": int(snapshot.measurement_count),
            "selected_counts": selected_counts,
            "score_tuple": [float(value) for value in snapshot.score_tuple],
            "components": {
                str(key): float(value) for key, value in snapshot.components.items()
            },
            "unresolved_isotopes": sorted(str(key) for key in snapshot.unresolved),
        }

    def final_report_estimate(
        self,
        *,
        use_best_so_far: bool = True,
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Return the current or best-so-far report estimate for final output."""
        current = self.record_report_snapshot(label="final")
        if current is None:
            estimates = self.estimates()
            self._last_final_report_selection = {
                "selected": "current",
                "reason": "best_so_far_disabled",
            }
            return self._copy_estimate_map(estimates)
        best = self._best_report_snapshot
        if (
            bool(use_best_so_far)
            and best is not None
            and self._report_snapshot_is_better(best, current)
        ):
            eligible, reason = self._report_snapshot_final_eligible(best, current)
            if eligible:
                self._last_final_report_selection = {
                    "selected": "best_so_far",
                    "reason": reason,
                    "current": self._snapshot_summary(current),
                    "best": self._snapshot_summary(best),
                }
                return self._copy_estimate_map(best.estimates)
            self._last_final_report_selection = {
                "selected": "current",
                "reason": "best_so_far_ineligible",
                "best_so_far_rejected_reason": reason,
                "current": self._snapshot_summary(current),
                "best": self._snapshot_summary(best),
            }
            return self._copy_estimate_map(current.estimates)
        self._last_final_report_selection = {
            "selected": "current",
            "reason": "current_not_worse",
            "current": self._snapshot_summary(current),
            "best": self._snapshot_summary(best),
        }
        return self._copy_estimate_map(current.estimates)

    def best_report_summary(self) -> dict[str, Any]:
        """Return best-so-far report diagnostics for run summaries."""
        return {
            "snapshot_count": int(len(self._report_snapshots)),
            "best": self._snapshot_summary(self._best_report_snapshot),
            "last": self._snapshot_summary(
                self._report_snapshots[-1] if self._report_snapshots else None
            ),
            "final_selection": copy.deepcopy(self._last_final_report_selection),
        }

    def _best_report_modes_for_isotope(
        self,
        isotope: str,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return best-so-far report modes for one isotope."""
        snapshot, _reason = self._best_report_snapshot_for_runtime()
        if snapshot is None:
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        positions, strengths = snapshot.estimates.get(
            str(isotope),
            (np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)),
        )
        return (
            np.asarray(positions, dtype=float).reshape(-1, 3).copy(),
            np.maximum(np.asarray(strengths, dtype=float).reshape(-1), 0.0).copy(),
        )

    def _candidate_response_source_key(
        self,
        sources: NDArray[np.float64],
    ) -> tuple[str, int] | None:
        """Return a stable cache key for the full shared candidate-source grid."""
        source_arr = np.asarray(sources, dtype=float).reshape(-1, 3)
        candidate_arr = np.asarray(self.candidate_sources, dtype=float).reshape(
            -1,
            3,
        )
        if (
            source_arr.shape == candidate_arr.shape
            and source_arr.size > 0
            and np.shares_memory(source_arr, candidate_arr)
        ):
            return ("candidate_sources", int(source_arr.shape[0]))
        return None

    @staticmethod
    def _measurement_geometry_digest(data: MeasurementData) -> bytes:
        """Return a compact digest for measurement geometry arrays used by responses."""
        digest = hashlib.blake2b(digest_size=16)
        for array in (
            np.asarray(data.detector_positions, dtype=np.float64),
            np.asarray(data.live_times, dtype=np.float64),
            np.asarray(data.fe_indices, dtype=np.int64),
            np.asarray(data.pb_indices, dtype=np.int64),
        ):
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.shape).encode("ascii"))
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(contiguous.tobytes())
        return digest.digest()

    @staticmethod
    def _response_scale_vector(
        scale: NDArray[np.float64] | float,
        measurement_count: int,
    ) -> NDArray[np.float64]:
        """Return one response-scale value per measurement row."""
        count = max(0, int(measurement_count))
        scale_arr = np.asarray(scale, dtype=float).reshape(-1)
        if count == 0:
            return np.zeros(0, dtype=float)
        if scale_arr.size == 0:
            return np.ones(count, dtype=float)
        if scale_arr.size == 1 and count != 1:
            return np.full(count, float(scale_arr[0]), dtype=float)
        if scale_arr.size != count:
            raise ValueError(
                "source_scale must be scalar or one value per measurement."
            )
        return scale_arr.astype(float, copy=False)

    def _store_candidate_response_cache(
        self,
        cache_key: tuple[Any, ...],
        counts: NDArray[np.float64],
    ) -> None:
        """Store an exact deterministic candidate response with LRU eviction."""
        self._candidate_response_cache[cache_key] = np.asarray(
            counts,
            dtype=float,
        ).copy()
        self._candidate_response_cache_order.append(cache_key)
        max_entries = max(
            0,
            int(self.pf_config.candidate_response_cache_max_entries),
        )
        while len(self._candidate_response_cache_order) > max_entries:
            old_key = self._candidate_response_cache_order.pop(0)
            if old_key not in self._candidate_response_cache_order:
                self._candidate_response_cache.pop(old_key, None)

    @staticmethod
    def _candidate_response_prefix_matches(
        payload: Mapping[str, NDArray[np.float64]],
        data: MeasurementData,
        scale: NDArray[np.float64],
        row_count: int,
    ) -> bool:
        """Return True when cached response rows match the requested prefix."""
        rows = max(0, int(row_count))
        if rows < 0:
            return False
        try:
            return bool(
                np.allclose(
                    np.asarray(payload["detector_positions"], dtype=float)[:rows],
                    np.asarray(data.detector_positions, dtype=float)[:rows],
                    rtol=0.0,
                    atol=1.0e-12,
                )
                and np.allclose(
                    np.asarray(payload["live_times"], dtype=float)[:rows],
                    np.asarray(data.live_times, dtype=float)[:rows],
                    rtol=0.0,
                    atol=1.0e-12,
                )
                and np.array_equal(
                    np.asarray(payload["fe_indices"], dtype=int)[:rows],
                    np.asarray(data.fe_indices, dtype=int)[:rows],
                )
                and np.array_equal(
                    np.asarray(payload["pb_indices"], dtype=int)[:rows],
                    np.asarray(data.pb_indices, dtype=int)[:rows],
                )
                and np.allclose(
                    np.asarray(payload["source_scale"], dtype=float)[:rows],
                    np.asarray(scale, dtype=float)[:rows],
                    rtol=0.0,
                    atol=1.0e-12,
                )
            )
        except (KeyError, ValueError, TypeError):
            return False

    def _store_candidate_response_prefix_cache(
        self,
        prefix_key: tuple[Any, ...],
        data: MeasurementData,
        scale: NDArray[np.float64],
        counts: NDArray[np.float64],
    ) -> None:
        """Store a full-history response prefix for incremental extension."""
        self._candidate_response_prefix_cache[prefix_key] = {
            "detector_positions": np.asarray(
                data.detector_positions, dtype=float
            ).copy(),
            "live_times": np.asarray(data.live_times, dtype=float).copy(),
            "fe_indices": np.asarray(data.fe_indices, dtype=int).copy(),
            "pb_indices": np.asarray(data.pb_indices, dtype=int).copy(),
            "source_scale": np.asarray(scale, dtype=float).copy(),
            "counts": np.asarray(counts, dtype=float).copy(),
        }
        self._candidate_response_prefix_cache_order.append(prefix_key)
        max_entries = max(
            0,
            int(self.pf_config.candidate_response_cache_max_entries),
        )
        while len(self._candidate_response_prefix_cache_order) > max_entries:
            old_key = self._candidate_response_prefix_cache_order.pop(0)
            if old_key not in self._candidate_response_prefix_cache_order:
                self._candidate_response_prefix_cache.pop(old_key, None)

    def _cached_expected_counts_per_source(
        self,
        *,
        filt: IsotopeParticleFilter,
        isotope: str,
        data: MeasurementData,
        sources: NDArray[np.float64],
        strengths: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return expected counts, reusing deterministic full-grid responses."""
        source_arr = np.asarray(sources, dtype=float).reshape(-1, 3)
        strength_arr = np.asarray(strengths, dtype=float).reshape(-1)
        source_key = self._candidate_response_source_key(source_arr)
        scale = self.response_scales_for_measurements(
            isotope,
            data.fe_indices,
            data.pb_indices,
        )
        measurement_count = int(np.asarray(data.live_times, dtype=float).size)
        scale_arr = self._response_scale_vector(scale, measurement_count)
        cache_enabled = (
            source_key is not None
            and strength_arr.size == source_arr.shape[0]
            and np.allclose(strength_arr, 1.0)
            and int(self.pf_config.candidate_response_cache_max_entries) > 0
        )
        cache_key: tuple[Any, ...] | None = None
        prefix_key: tuple[Any, ...] | None = None
        if cache_enabled:
            cache_key = (
                str(isotope),
                int(id(filt.continuous_kernel)),
                source_key,
                self._measurement_geometry_digest(data),
                tuple(scale_arr.round(12).tolist()),
            )
            cached = self._candidate_response_cache.get(cache_key)
            if cached is not None:
                return cached.copy()
            prefix_key = (str(isotope), int(id(filt.continuous_kernel)), source_key)
            prefix_payload = self._candidate_response_prefix_cache.get(prefix_key)
            if isinstance(prefix_payload, dict):
                cached_counts = np.asarray(
                    prefix_payload.get("counts", np.zeros((0, 0))),
                    dtype=float,
                )
                cached_rows = int(cached_counts.shape[0])
                if (
                    cached_rows >= measurement_count
                    and self._candidate_response_prefix_matches(
                        prefix_payload,
                        data,
                        scale_arr,
                        measurement_count,
                    )
                ):
                    counts_arr = cached_counts[:measurement_count].copy()
                    self._store_candidate_response_cache(cache_key, counts_arr)
                    return counts_arr
                if (
                    cached_rows < measurement_count
                    and self._candidate_response_prefix_matches(
                        prefix_payload,
                        data,
                        scale_arr,
                        cached_rows,
                    )
                ):
                    suffix_counts = expected_counts_per_source(
                        kernel=filt.continuous_kernel,
                        isotope=isotope,
                        detector_positions=np.asarray(
                            data.detector_positions,
                            dtype=float,
                        )[cached_rows:],
                        sources=source_arr,
                        strengths=strength_arr,
                        live_times=np.asarray(data.live_times, dtype=float)[
                            cached_rows:
                        ],
                        fe_indices=np.asarray(data.fe_indices, dtype=int)[cached_rows:],
                        pb_indices=np.asarray(data.pb_indices, dtype=int)[cached_rows:],
                        source_scale=scale_arr[cached_rows:],
                    )
                    counts_arr = np.vstack(
                        [
                            cached_counts,
                            np.asarray(suffix_counts, dtype=float),
                        ]
                    )
                    self._store_candidate_response_prefix_cache(
                        prefix_key,
                        data,
                        scale_arr,
                        counts_arr,
                    )
                    self._store_candidate_response_cache(cache_key, counts_arr)
                    return counts_arr.copy()
        counts = expected_counts_per_source(
            kernel=filt.continuous_kernel,
            isotope=isotope,
            detector_positions=data.detector_positions,
            sources=source_arr,
            strengths=strength_arr,
            live_times=data.live_times,
            fe_indices=data.fe_indices,
            pb_indices=data.pb_indices,
            source_scale=scale_arr,
        )
        counts_arr = np.asarray(counts, dtype=float)
        if cache_enabled and cache_key is not None:
            self._store_candidate_response_cache(cache_key, counts_arr)
            if prefix_key is not None:
                self._store_candidate_response_prefix_cache(
                    prefix_key,
                    data,
                    scale_arr,
                    counts_arr,
                )
        return counts_arr

    def _cached_candidate_grid_counts(
        self,
        *,
        filt: IsotopeParticleFilter,
        isotope: str,
        data: MeasurementData,
    ) -> NDArray[np.float64]:
        """Return unit-strength responses for the full source-candidate grid."""
        pool = np.asarray(self.candidate_sources, dtype=float).reshape(-1, 3)
        if pool.size == 0:
            return np.zeros((int(data.z_k.size), 0), dtype=float)
        counts = self._cached_expected_counts_per_source(
            filt=filt,
            isotope=isotope,
            data=data,
            sources=pool,
            strengths=np.ones(pool.shape[0], dtype=float),
        )
        return np.asarray(counts, dtype=float)

    def _resolve_mu_by_isotope(
        self, mu_by_isotope: Dict[str, object] | None
    ) -> Dict[str, object]:
        """
        Ensure per-isotope attenuation coefficients are available for all isotopes.

        When missing, attempt to populate values from the HVL/TVL table; otherwise raise.
        """
        from measurement.shielding import HVL_TVL_TABLE_MM, mu_by_isotope_from_tvl_mm

        def _norm_key(name: str) -> str:
            """Return a normalized isotope key for attenuation lookup."""
            return re.sub(r"[^A-Za-z0-9]", "", name).upper()

        canonical_by_norm = {
            "CS137": "Cs-137",
            "CO60": "Co-60",
            "EU154": "Eu-154",
        }

        resolved: Dict[str, object] = {}
        if mu_by_isotope is not None:
            resolved.update(mu_by_isotope)
        normalized: Dict[str, object] = {}
        for key, value in resolved.items():
            normalized[_norm_key(key)] = value
        isotope_names = (
            self.all_isotopes if hasattr(self, "all_isotopes") else self.isotopes
        )
        if isotope_names:
            still_missing: List[str] = []
            for iso in isotope_names:
                if iso in resolved:
                    continue
                norm = _norm_key(iso)
                if norm in normalized:
                    resolved[iso] = normalized[norm]
                    continue
                canonical = canonical_by_norm.get(norm)
                if canonical is not None:
                    table_vals = mu_by_isotope_from_tvl_mm(
                        HVL_TVL_TABLE_MM, isotopes=[canonical]
                    )
                    if canonical in table_vals:
                        resolved[iso] = table_vals[canonical]
                        normalized[norm] = table_vals[canonical]
                        if canonical not in resolved:
                            resolved[canonical] = table_vals[canonical]
                        continue
                still_missing.append(iso)
            if still_missing:
                missing_list = ", ".join(still_missing)
                raise ValueError(
                    "mu_by_isotope is missing entries for isotopes: "
                    f"{missing_list}. Ensure isotope names match the HVL/TVL table keys."
                )
        return resolved

    def _ensure_kernel_cache(self) -> None:
        """Build the discrete kernel cache when it is first needed."""
        if self.kernel_cache is not None:
            return
        if len(self.poses) == 0:
            raise ValueError("No poses added; cannot build kernel cache.")
        poses_arr = np.stack(self.poses, axis=0)
        self.kernel_cache = KernelPrecomputer(
            candidate_sources=self.candidate_sources,
            poses=poses_arr,
            orientations=self.normals,
            shield_params=self.shield_params,
            mu_by_isotope=self.mu_by_isotope,
            use_gpu=self.pf_config.use_gpu,
            gpu_device=self.pf_config.gpu_device,
            gpu_dtype=self.pf_config.gpu_dtype,
        )
        pf_conf = self._build_pf_config()
        if self.filters:
            for iso in self.isotopes:
                if iso in self.filters:
                    self.filters[iso].set_kernel(self.kernel_cache)
                else:
                    self.filters[iso] = self._build_filter(iso, pf_conf)
        else:
            for iso in self.isotopes:
                self.filters[iso] = self._build_filter(iso, pf_conf)

    def _build_filter(self, isotope: str, pf_conf: PFConfig) -> IsotopeParticleFilter:
        """Build an isotope filter with shared PF observation-model settings."""
        return IsotopeParticleFilter(
            isotope,
            kernel=self.kernel_cache,
            config=pf_conf,
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
        )

    def _build_pf_config(self) -> PFConfig:
        """Build a per-isotope PFConfig from the estimator configuration."""
        return PFConfig(
            num_particles=self.pf_config.num_particles,
            min_particles=self.pf_config.min_particles,
            max_particles=self.pf_config.max_particles,
            ess_low=self.pf_config.ess_low,
            ess_high=self.pf_config.ess_high,
            max_sources=self.pf_config.max_sources,
            resample_threshold=self.pf_config.resample_threshold,
            position_sigma=self.pf_config.position_sigma,
            strength_sigma=self.pf_config.strength_sigma,
            background_sigma=self.pf_config.background_sigma,
            background_level=self.pf_config.background_level,
            measurement_scale_by_isotope=self.pf_config.measurement_scale_by_isotope,
            measurement_scale_by_isotope_and_pair=(
                self.pf_config.measurement_scale_by_isotope_and_pair
            ),
            count_likelihood_model=self.pf_config.count_likelihood_model,
            transport_model_rel_sigma=self.pf_config.transport_model_rel_sigma,
            transport_model_abs_sigma=self.pf_config.transport_model_abs_sigma,
            spectrum_count_rel_sigma=self.pf_config.spectrum_count_rel_sigma,
            spectrum_count_abs_sigma=self.pf_config.spectrum_count_abs_sigma,
            low_count_abs_sigma=self.pf_config.low_count_abs_sigma,
            low_count_transition_counts=self.pf_config.low_count_transition_counts,
            count_likelihood_df=self.pf_config.count_likelihood_df,
            shield_contrast_likelihood_enable=(
                self.pf_config.shield_contrast_likelihood_enable
            ),
            shield_contrast_likelihood_weight=(
                self.pf_config.shield_contrast_likelihood_weight
            ),
            shield_contrast_log_sigma_floor=(
                self.pf_config.shield_contrast_log_sigma_floor
            ),
            shield_contrast_log_sigma_ceiling=(
                self.pf_config.shield_contrast_log_sigma_ceiling
            ),
            shield_contrast_min_count=self.pf_config.shield_contrast_min_count,
            shield_contrast_min_views=self.pf_config.shield_contrast_min_views,
            shield_contrast_likelihood_df=(
                self.pf_config.shield_contrast_likelihood_df
            ),
            shield_view_ratio_likelihood_enable=(
                self.pf_config.shield_view_ratio_likelihood_enable
            ),
            shield_view_ratio_likelihood_weight=(
                self.pf_config.shield_view_ratio_likelihood_weight
            ),
            shield_view_ratio_likelihood_concentration=(
                self.pf_config.shield_view_ratio_likelihood_concentration
            ),
            shield_view_ratio_likelihood_min_total_count=(
                self.pf_config.shield_view_ratio_likelihood_min_total_count
            ),
            shield_view_ratio_likelihood_min_views=(
                self.pf_config.shield_view_ratio_likelihood_min_views
            ),
            station_view_covariance_enable=(
                self.pf_config.station_view_covariance_enable
            ),
            station_view_correlated_spectrum_fraction=(
                self.pf_config.station_view_correlated_spectrum_fraction
            ),
            spectrum_likelihood_bin_chunk=(
                self.pf_config.spectrum_likelihood_bin_chunk
            ),
            min_strength=self.pf_config.min_strength,
            p_birth=self.pf_config.p_birth,
            p_kill=self.pf_config.p_kill,
            death_low_q_streak=self.pf_config.death_low_q_streak,
            death_delta_ll_threshold=self.pf_config.death_delta_ll_threshold,
            support_ema_alpha=self.pf_config.support_ema_alpha,
            support_window=self.pf_config.support_window,
            birth_window=self.pf_config.birth_window,
            birth_softmax_temp=self.pf_config.birth_softmax_temp,
            birth_min_score=self.pf_config.birth_min_score,
            birth_enable=self.pf_config.birth_enable,
            birth_topk_particles=self.pf_config.birth_topk_particles,
            birth_use_weighted_topk=self.pf_config.birth_use_weighted_topk,
            birth_min_sep_m=self.pf_config.birth_min_sep_m,
            birth_detector_min_sep_m=self.pf_config.birth_detector_min_sep_m,
            source_detector_exclusion_m=self.pf_config.source_detector_exclusion_m,
            birth_candidate_jitter_sigma=self.pf_config.birth_candidate_jitter_sigma,
            birth_num_local_jitter=self.pf_config.birth_num_local_jitter,
            birth_alpha=self.pf_config.birth_alpha,
            birth_q_max=self.pf_config.birth_q_max,
            birth_q_min=self.pf_config.birth_q_min,
            birth_max_per_update=self.pf_config.birth_max_per_update,
            birth_delta_ll_threshold=self.pf_config.birth_delta_ll_threshold,
            birth_complexity_penalty=self.pf_config.birth_complexity_penalty,
            birth_bic_penalty_params=self.pf_config.birth_bic_penalty_params,
            structural_update_min_counts=self.pf_config.structural_update_min_counts,
            structural_update_min_snr=self.pf_config.structural_update_min_snr,
            structural_update_count_min_snr=(
                self.pf_config.structural_update_count_min_snr
            ),
            structural_update_max_rel_sigma=(
                self.pf_config.structural_update_max_rel_sigma
            ),
            birth_min_distinct_poses=self.pf_config.birth_min_distinct_poses,
            birth_residual_clip_quantile=self.pf_config.birth_residual_clip_quantile,
            birth_residual_gate_p_value=self.pf_config.birth_residual_gate_p_value,
            birth_residual_min_support=self.pf_config.birth_residual_min_support,
            birth_residual_support_sigma=self.pf_config.birth_residual_support_sigma,
            birth_min_distinct_stations=self.pf_config.birth_min_distinct_stations,
            birth_candidate_support_fraction=self.pf_config.birth_candidate_support_fraction,
            birth_refit_residual_gate=self.pf_config.birth_refit_residual_gate,
            birth_refit_residual_min_fraction=self.pf_config.birth_refit_residual_min_fraction,
            birth_use_shield_coded_residual=self.pf_config.birth_use_shield_coded_residual,
            birth_existing_response_corr_max=self.pf_config.birth_existing_response_corr_max,
            birth_response_condition_max=self.pf_config.birth_response_condition_max,
            birth_count_distance_prior_weight=(
                self.pf_config.birth_count_distance_prior_weight
            ),
            birth_count_distance_strength_weight=(
                self.pf_config.birth_count_distance_strength_weight
            ),
            birth_count_distance_log_clip=self.pf_config.birth_count_distance_log_clip,
            birth_count_distance_strength_sigma=(
                self.pf_config.birth_count_distance_strength_sigma
            ),
            birth_residual_always_try=self.pf_config.birth_residual_always_try,
            birth_residual_expand_structural_particles=(
                self.pf_config.birth_residual_expand_structural_particles
            ),
            birth_residual_expanded_structural_topk_particles=(
                self.pf_config.birth_residual_expanded_structural_topk_particles
            ),
            birth_residual_acceptance_complexity_scale=(
                self.pf_config.birth_residual_acceptance_complexity_scale
            ),
            birth_residual_force_proposal_on_gate=(
                self.pf_config.birth_residual_force_proposal_on_gate
            ),
            birth_residual_forced_min_delta_ll=(
                self.pf_config.birth_residual_forced_min_delta_ll
            ),
            birth_residual_force_relax_candidate_masks=(
                self.pf_config.birth_residual_force_relax_candidate_masks
            ),
            birth_residual_suppress_death=self.pf_config.birth_residual_suppress_death,
            birth_matching_pursuit_max_new_sources=(
                self.pf_config.birth_matching_pursuit_max_new_sources
            ),
            birth_matching_pursuit_topk_candidates=(
                self.pf_config.birth_matching_pursuit_topk_candidates
            ),
            birth_orthogonalize_residual_candidates=(
                self.pf_config.birth_orthogonalize_residual_candidates
            ),
            birth_orthogonal_candidate_corr_max=(
                self.pf_config.birth_orthogonal_candidate_corr_max
            ),
            birth_jitter_topk_candidates=self.pf_config.birth_jitter_topk_candidates,
            birth_global_rescue_enable=self.pf_config.birth_global_rescue_enable,
            birth_global_rescue_max_candidates=(
                self.pf_config.birth_global_rescue_max_candidates
            ),
            birth_global_rescue_min_residual_fraction=(
                self.pf_config.birth_global_rescue_min_residual_fraction
            ),
            birth_global_rescue_dedup_radius_m=(
                self.pf_config.birth_global_rescue_dedup_radius_m
            ),
            birth_global_rescue_force_proposal_on_gate=(
                self.pf_config.birth_global_rescue_force_proposal_on_gate
            ),
            birth_global_rescue_forced_min_delta_ll=(
                self.pf_config.birth_global_rescue_forced_min_delta_ll
            ),
            birth_global_rescue_min_support=(
                self.pf_config.birth_global_rescue_min_support
            ),
            birth_global_rescue_min_distinct_poses=(
                self.pf_config.birth_global_rescue_min_distinct_poses
            ),
            birth_global_rescue_min_distinct_stations=(
                self.pf_config.birth_global_rescue_min_distinct_stations
            ),
            high_strength_split_enable=self.pf_config.high_strength_split_enable,
            high_strength_split_q_multiple=(
                self.pf_config.high_strength_split_q_multiple
            ),
            high_strength_split_offset_m=(self.pf_config.high_strength_split_offset_m),
            high_strength_split_candidate_count=(
                self.pf_config.high_strength_split_candidate_count
            ),
            runtime_report_rescue_enable=self.pf_config.runtime_report_rescue_enable,
            runtime_report_rescue_particle_fraction=(
                self.pf_config.runtime_report_rescue_particle_fraction
            ),
            runtime_report_rescue_min_particles_per_source=(
                self.pf_config.runtime_report_rescue_min_particles_per_source
            ),
            runtime_report_rescue_weight=self.pf_config.runtime_report_rescue_weight,
            runtime_report_rescue_jitter_sigma_m=(
                self.pf_config.runtime_report_rescue_jitter_sigma_m
            ),
            residual_decomposition_enable=(
                self.pf_config.residual_decomposition_enable
            ),
            peak_suppression_enable=self.pf_config.peak_suppression_enable,
            peak_suppression_min_source_fraction=(
                self.pf_config.peak_suppression_min_source_fraction
            ),
            peak_suppression_factor=self.pf_config.peak_suppression_factor,
            residual_decomposition_max_layers=(
                self.pf_config.residual_decomposition_max_layers
            ),
            pseudo_source_verification_enable=(
                self.pf_config.pseudo_source_verification_enable
            ),
            pseudo_source_min_delta_ll=self.pf_config.pseudo_source_min_delta_ll,
            pseudo_source_min_distinct_views=(
                self.pf_config.pseudo_source_min_distinct_views
            ),
            pseudo_source_fail_grace_stations=(
                self.pf_config.pseudo_source_fail_grace_stations
            ),
            pseudo_source_corr_max=self.pf_config.pseudo_source_corr_max,
            pseudo_source_temporal_sep_min=(
                self.pf_config.pseudo_source_temporal_sep_min
            ),
            pseudo_source_quarantine_on_suppress=(
                self.pf_config.pseudo_source_quarantine_on_suppress
            ),
            pseudo_source_quarantine_excludes_runtime=(
                self.pf_config.pseudo_source_quarantine_excludes_runtime
            ),
            report_exclude_unverified_sources=(
                self.pf_config.report_exclude_unverified_sources
            ),
            source_prune_min_distinct_stations=(
                self.pf_config.source_prune_min_distinct_stations
            ),
            source_prune_min_distinct_views=(
                self.pf_config.source_prune_min_distinct_views
            ),
            source_prune_fail_grace_stations=(
                self.pf_config.source_prune_fail_grace_stations
            ),
            source_prune_delta_ll_threshold=(
                self.pf_config.source_prune_delta_ll_threshold
            ),
            source_prune_refit_after_remove=(
                self.pf_config.source_prune_refit_after_remove
            ),
            source_prune_bic_penalty_params=(
                self.pf_config.source_prune_bic_penalty_params
            ),
            refit_after_moves=self.pf_config.refit_after_moves,
            refit_iters=self.pf_config.refit_iters,
            refit_eps=self.pf_config.refit_eps,
            weak_source_prune_min_expected_count=self.pf_config.weak_source_prune_min_expected_count,
            weak_source_prune_min_fraction=self.pf_config.weak_source_prune_min_fraction,
            weak_source_prune_min_age=self.pf_config.weak_source_prune_min_age,
            weak_source_prune_require_observable=(
                self.pf_config.weak_source_prune_require_observable
            ),
            weak_source_prune_min_observable_measurements=(
                self.pf_config.weak_source_prune_min_observable_measurements
            ),
            weak_source_prune_observable_count=(
                self.pf_config.weak_source_prune_observable_count
            ),
            weak_source_prune_observable_fraction=(
                self.pf_config.weak_source_prune_observable_fraction
            ),
            weak_source_prune_visibility_reference_strength=(
                self.pf_config.weak_source_prune_visibility_reference_strength
            ),
            conditional_strength_refit=self.pf_config.conditional_strength_refit,
            conditional_strength_refit_window=self.pf_config.conditional_strength_refit_window,
            conditional_strength_refit_iters=self.pf_config.conditional_strength_refit_iters,
            conditional_strength_refit_reweight=self.pf_config.conditional_strength_refit_reweight,
            conditional_strength_refit_cardinality_neutral_reweight=(
                self.pf_config.conditional_strength_refit_cardinality_neutral_reweight
            ),
            conditional_strength_refit_reweight_clip=self.pf_config.conditional_strength_refit_reweight_clip,
            conditional_strength_refit_min_count=self.pf_config.conditional_strength_refit_min_count,
            conditional_strength_refit_min_snr=self.pf_config.conditional_strength_refit_min_snr,
            conditional_strength_refit_prior_weight=self.pf_config.conditional_strength_refit_prior_weight,
            conditional_strength_refit_prior_rel_sigma=self.pf_config.conditional_strength_refit_prior_rel_sigma,
            source_strength_prior_mean=self.pf_config.source_strength_prior_mean,
            source_strength_prior_weight=self.pf_config.source_strength_prior_weight,
            source_strength_prior_rel_sigma=(
                self.pf_config.source_strength_prior_rel_sigma
            ),
            source_strength_absorption_penalty_weight=(
                self.pf_config.source_strength_absorption_penalty_weight
            ),
            source_strength_absorption_q_multiple=(
                self.pf_config.source_strength_absorption_q_multiple
            ),
            source_strength_observation_overshoot_penalty_weight=(
                self.pf_config.source_strength_observation_overshoot_penalty_weight
            ),
            source_strength_observation_overshoot_sigma=(
                self.pf_config.source_strength_observation_overshoot_sigma
            ),
            source_strength_observation_overshoot_quantile=(
                self.pf_config.source_strength_observation_overshoot_quantile
            ),
            source_strength_observation_overshoot_min_visible_fraction=(
                self.pf_config.source_strength_observation_overshoot_min_visible_fraction
            ),
            source_strength_observation_overshoot_min_visible_measurements=(
                self.pf_config.source_strength_observation_overshoot_min_visible_measurements
            ),
            birth_stage_single_station_as_quarantine=(
                self.pf_config.birth_stage_single_station_as_quarantine
            ),
            min_age_to_split=self.pf_config.min_age_to_split,
            use_clustered_output=self.pf_config.use_clustered_output,
            cluster_eps_m=self.pf_config.cluster_eps_m,
            cluster_min_samples=self.pf_config.cluster_min_samples,
            cluster_report_max_points=self.pf_config.cluster_report_max_points,
            cluster_exact_max_points=self.pf_config.cluster_exact_max_points,
            split_prob=self.pf_config.split_prob,
            split_strength_min=self.pf_config.split_strength_min,
            split_position_sigma=self.pf_config.split_position_sigma,
            split_strength_min_frac=self.pf_config.split_strength_min_frac,
            split_strength_max_frac=self.pf_config.split_strength_max_frac,
            split_delta_ll_threshold=self.pf_config.split_delta_ll_threshold,
            split_complexity_penalty=self.pf_config.split_complexity_penalty,
            split_residual_guided=self.pf_config.split_residual_guided,
            split_residual_always_try=self.pf_config.split_residual_always_try,
            split_residual_candidate_count=self.pf_config.split_residual_candidate_count,
            merge_prob=self.pf_config.merge_prob,
            merge_distance_max=self.pf_config.merge_distance_max,
            merge_delta_ll_threshold=self.pf_config.merge_delta_ll_threshold,
            merge_response_corr_min=self.pf_config.merge_response_corr_min,
            merge_search_topk_pairs=self.pf_config.merge_search_topk_pairs,
            structural_proposal_topk_particles=(
                self.pf_config.structural_proposal_topk_particles
            ),
            structural_trial_workers=self.pf_config.structural_trial_workers,
            structural_trial_parallel_min_trials=(
                self.pf_config.structural_trial_parallel_min_trials
            ),
            target_ess_ratio=self.pf_config.target_ess_ratio,
            max_temper_steps=self.pf_config.max_temper_steps,
            min_delta_beta=self.pf_config.min_delta_beta,
            max_resamples_per_observation=self.pf_config.max_resamples_per_observation,
            temper_resample_cooldown_steps=self.pf_config.temper_resample_cooldown_steps,
            temper_resample_force_ratio=self.pf_config.temper_resample_force_ratio,
            disable_regularize_on_temper_resample=self.pf_config.disable_regularize_on_temper_resample,
            deferred_resample_roughening_scale=(
                self.pf_config.deferred_resample_roughening_scale
            ),
            cardinality_preserving_resample=self.pf_config.cardinality_preserving_resample,
            cardinality_preserving_min_stations=(
                self.pf_config.cardinality_preserving_min_stations
            ),
            cardinality_preserving_require_confirmed_structure=(
                self.pf_config.cardinality_preserving_require_confirmed_structure
            ),
            mode_preserving_resample=self.pf_config.mode_preserving_resample,
            mode_preserving_max_modes=self.pf_config.mode_preserving_max_modes,
            mode_preserving_particles_per_mode=(
                self.pf_config.mode_preserving_particles_per_mode
            ),
            mode_preserving_radius_m=self.pf_config.mode_preserving_radius_m,
            mode_preserving_min_weight_fraction=(
                self.pf_config.mode_preserving_min_weight_fraction
            ),
            mode_preserving_surface_strata=(
                self.pf_config.mode_preserving_surface_strata
            ),
            mode_preserving_height_bin_m=self.pf_config.mode_preserving_height_bin_m,
            mode_preserving_high_surface_extra_particles=(
                self.pf_config.mode_preserving_high_surface_extra_particles
            ),
            mode_preserving_high_surface_z_fraction=(
                self.pf_config.mode_preserving_high_surface_z_fraction
            ),
            mode_preserving_support_score_weight=(
                self.pf_config.mode_preserving_support_score_weight
            ),
            mode_preserving_tentative_boost=(
                self.pf_config.mode_preserving_tentative_boost
            ),
            mode_preserving_residual_boost=(
                self.pf_config.mode_preserving_residual_boost
            ),
            mode_preserving_cardinality_strata=(
                self.pf_config.mode_preserving_cardinality_strata
            ),
            mode_preserving_min_particles_per_cardinality=(
                self.pf_config.mode_preserving_min_particles_per_cardinality
            ),
            mode_preserving_report_cardinality_strata=(
                self.pf_config.mode_preserving_report_cardinality_strata
            ),
            mode_preserving_report_cardinality_extra_particles=(
                self.pf_config.mode_preserving_report_cardinality_extra_particles
            ),
            mode_preserving_dynamic_cardinality_allocation=(
                self.pf_config.mode_preserving_dynamic_cardinality_allocation
            ),
            mode_preserving_dynamic_cardinality_extra_particles=(
                self.pf_config.mode_preserving_dynamic_cardinality_extra_particles
            ),
            mode_preserving_dynamic_cardinality_min_mass=(
                self.pf_config.mode_preserving_dynamic_cardinality_min_mass
            ),
            mode_preserving_dynamic_cardinality_entropy_min=(
                self.pf_config.mode_preserving_dynamic_cardinality_entropy_min
            ),
            mode_preserving_dynamic_spatial_allocation=(
                self.pf_config.mode_preserving_dynamic_spatial_allocation
            ),
            mode_preserving_dynamic_spatial_extra_particles=(
                self.pf_config.mode_preserving_dynamic_spatial_extra_particles
            ),
            mode_preserving_dynamic_spatial_min_score_fraction=(
                self.pf_config.mode_preserving_dynamic_spatial_min_score_fraction
            ),
            adapt_cooldown_steps=self.pf_config.adapt_cooldown_steps,
            position_min=self.pf_config.position_min,
            position_max=self.pf_config.position_max,
            source_position_prior=self.pf_config.source_position_prior,
            init_num_sources=self.pf_config.init_num_sources,
            init_grid_spacing_m=self.pf_config.init_grid_spacing_m,
            init_grid_repeats=self.pf_config.init_grid_repeats,
            roughening_k=self.pf_config.roughening_k,
            surface_rejuvenation_enable=self.pf_config.surface_rejuvenation_enable,
            min_sigma_pos=self.pf_config.min_sigma_pos,
            max_sigma_pos=self.pf_config.max_sigma_pos,
            roughening_decay=self.pf_config.roughening_decay,
            roughening_min_mult=self.pf_config.roughening_min_mult,
            init_strength_log_mean=self.pf_config.init_strength_log_mean,
            init_strength_log_sigma=self.pf_config.init_strength_log_sigma,
            strength_log_sigma=self.pf_config.strength_log_sigma,
            use_gpu=self.pf_config.use_gpu,
            gpu_device=self.pf_config.gpu_device,
            gpu_dtype=self.pf_config.gpu_dtype,
            use_tempering=self.pf_config.use_tempering,
            label_enable=self.pf_config.label_enable,
            label_alignment_iters=self.pf_config.label_alignment_iters,
            label_pos_weight=self.pf_config.label_pos_weight,
            label_strength_weight=self.pf_config.label_strength_weight,
            label_missing_cost=self.pf_config.label_missing_cost,
            label_pos_scale=self.pf_config.label_pos_scale,
            label_strength_scale=self.pf_config.label_strength_scale,
            converge_enable=self.pf_config.converge_enable,
            converge_window=self.pf_config.converge_window,
            converge_map_move_eps_m=self.pf_config.converge_map_move_eps_m,
            converge_ess_ratio_high=self.pf_config.converge_ess_ratio_high,
            converge_ll_improve_eps=self.pf_config.converge_ll_improve_eps,
            converge_min_steps=self.pf_config.converge_min_steps,
            converge_require_all=self.pf_config.converge_require_all,
            converge_cardinality_var_max=self.pf_config.converge_cardinality_var_max,
            converge_require_no_tentative=self.pf_config.converge_require_no_tentative,
            converge_freeze_updates=self.pf_config.converge_freeze_updates,
            converge_min_stations=self.pf_config.converge_min_stations,
            converge_cluster_spread_max_m=(
                self.pf_config.converge_cluster_spread_max_m
            ),
            converge_cluster_min_support_fraction=(
                self.pf_config.converge_cluster_min_support_fraction
            ),
        )

    def _gpu_enabled(self) -> bool:
        """Return True if GPU computation is enabled and available."""
        from pf import gpu_utils

        if not self.pf_config.use_gpu:
            raise RuntimeError(
                "GPU-only mode: enable use_gpu in RotatingShieldPFConfig."
            )
        if not gpu_utils.torch_device_available(self.pf_config.gpu_device):
            raise RuntimeError("GPU-only mode requires torch on the requested device.")
        return True

    def _can_use_gpu(self) -> bool:
        """Return whether torch-backed estimator math is available."""
        from pf import gpu_utils

        return bool(
            self.pf_config.use_gpu
            and gpu_utils.torch_device_available(self.pf_config.gpu_device)
        )

    def response_scale_for_isotope(
        self,
        isotope: str,
        *,
        fe_index: int | None = None,
        pb_index: int | None = None,
        shield_pair_id: int | None = None,
    ) -> float:
        """Return the configured source response scale for one isotope."""
        pair_id = self._shield_pair_id(
            fe_index=fe_index,
            pb_index=pb_index,
            shield_pair_id=shield_pair_id,
        )
        pair_scales = self.pf_config.measurement_scale_by_isotope_and_pair
        if pair_id is not None and isinstance(pair_scales, Mapping):
            iso_pair_scales = pair_scales.get(str(isotope), {})
            if isinstance(iso_pair_scales, Mapping):
                value = iso_pair_scales.get(int(pair_id))
                if value is None:
                    value = iso_pair_scales.get(str(int(pair_id)))  # type: ignore[arg-type]
                if value is not None:
                    return max(float(value), 0.0)
        scales = self.pf_config.measurement_scale_by_isotope
        if not isinstance(scales, dict):
            return 1.0
        return max(float(scales.get(isotope, 1.0)), 0.0)

    def response_scales_for_measurements(
        self,
        isotope: str,
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Return one source response scale per Fe/Pb measurement pair."""
        fe_arr = np.asarray(fe_indices, dtype=int).reshape(-1)
        pb_arr = np.asarray(pb_indices, dtype=int).reshape(-1)
        if fe_arr.size != pb_arr.size:
            raise ValueError("fe_indices and pb_indices must have matching length.")
        return np.asarray(
            [
                self.response_scale_for_isotope(
                    isotope,
                    fe_index=int(fe),
                    pb_index=int(pb),
                )
                for fe, pb in zip(fe_arr, pb_arr)
            ],
            dtype=float,
        )

    def _shield_pair_id(
        self,
        *,
        fe_index: int | None = None,
        pb_index: int | None = None,
        shield_pair_id: int | None = None,
    ) -> int | None:
        """Return the canonical shield-pair id when a pair is available."""
        if shield_pair_id is not None:
            return int(shield_pair_id)
        if fe_index is None or pb_index is None:
            return None
        return int(fe_index) * int(self.num_orientations) + int(pb_index)

    def _project_observation_covariance_to_variance(
        self,
        z_k: Mapping[str, float],
        z_variance_k: Mapping[str, float] | None,
        z_covariance_k: Mapping[str, Mapping[str, float]] | None,
    ) -> tuple[Dict[str, float] | None, Dict[str, Dict[str, float]] | None]:
        """
        Return diagonal PF variances that conservatively cover isotope covariance.

        The per-isotope filters are conditionally independent, while the
        response-Poisson spectrum regression reports a same-spectrum covariance
        across isotope count channels.  This projection keeps the observed count
        means unchanged and uses a Gershgorin-style diagonal envelope,
        ``var_i + sum_j |cov_ij|``, so ignoring off-diagonal terms cannot make
        the independent isotope filters more confident than the structured
        covariance supports.
        """
        if z_variance_k is None and z_covariance_k is None:
            return None, None
        isotopes = [str(isotope) for isotope in z_k]
        if not isotopes:
            return {}, None
        base_variances = np.asarray(
            [
                max(
                    float(
                        z_variance_k.get(isotope, max(float(z_k.get(isotope, 0.0)), 1.0))
                        if z_variance_k is not None
                        else max(float(z_k.get(isotope, 0.0)), 1.0)
                    ),
                    1.0,
                )
                for isotope in isotopes
            ],
            dtype=float,
        )
        if (
            z_covariance_k is None
            or not bool(self.pf_config.observation_covariance_projection_enable)
        ):
            return (
                {
                    isotope: float(variance)
                    for isotope, variance in zip(isotopes, base_variances)
                },
                self._sanitize_observation_covariance(
                    isotopes,
                    base_variances,
                    z_covariance_k,
                ),
            )
        covariance = self._observation_covariance_matrix(
            isotopes,
            base_variances,
            z_covariance_k,
        )
        if covariance is None:
            return (
                {
                    isotope: float(variance)
                    for isotope, variance in zip(isotopes, base_variances)
                },
                None,
            )
        offdiag_abs = np.sum(np.abs(covariance), axis=1) - np.abs(
            np.diag(covariance)
        )
        projected = np.maximum(
            base_variances,
            np.diag(covariance)
            + float(self.pf_config.observation_covariance_projection_weight)
            * offdiag_abs,
        )
        return (
            {
                isotope: float(variance)
                for isotope, variance in zip(isotopes, projected)
            },
            {
                row_iso: {
                    col_iso: float(covariance[row_idx, col_idx])
                    for col_idx, col_iso in enumerate(isotopes)
                }
                for row_idx, row_iso in enumerate(isotopes)
            },
        )

    def _observation_covariance_matrix(
        self,
        isotopes: Sequence[str],
        base_variances: NDArray[np.float64],
        z_covariance_k: Mapping[str, Mapping[str, float]],
    ) -> NDArray[np.float64] | None:
        """Build a symmetric isotope covariance matrix for one spectrum."""
        covariance = np.diag(np.maximum(np.asarray(base_variances, dtype=float), 1.0))
        index_by_isotope = {str(isotope): idx for idx, isotope in enumerate(isotopes)}
        for row_iso, row_payload in z_covariance_k.items():
            if str(row_iso) not in index_by_isotope or not isinstance(
                row_payload,
                Mapping,
            ):
                continue
            row_idx = index_by_isotope[str(row_iso)]
            for col_iso, raw_value in row_payload.items():
                col_key = str(col_iso)
                if col_key not in index_by_isotope:
                    continue
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(value):
                    continue
                col_idx = index_by_isotope[col_key]
                covariance[row_idx, col_idx] = value
        covariance = 0.5 * (covariance + covariance.T)
        np.fill_diagonal(covariance, np.maximum(np.diag(covariance), base_variances))
        diag = np.maximum(np.diag(covariance), 1.0)
        corr_limit = float(self.pf_config.observation_covariance_projection_max_corr)
        if corr_limit < 1.0:
            scale = np.sqrt(diag[:, None] * diag[None, :])
            corr = np.divide(
                covariance,
                scale,
                out=np.zeros_like(covariance, dtype=float),
                where=scale > 0.0,
            )
            corr = np.clip(corr, -corr_limit, corr_limit)
            covariance = corr * scale
            np.fill_diagonal(covariance, diag)
        return covariance

    def _sanitize_observation_covariance(
        self,
        isotopes: Sequence[str],
        base_variances: NDArray[np.float64],
        z_covariance_k: Mapping[str, Mapping[str, float]] | None,
    ) -> Dict[str, Dict[str, float]] | None:
        """Return a JSON-safe covariance payload when one was supplied."""
        if z_covariance_k is None:
            return None
        covariance = self._observation_covariance_matrix(
            isotopes,
            base_variances,
            z_covariance_k,
        )
        if covariance is None:
            return None
        return {
            row_iso: {
                col_iso: float(covariance[row_idx, col_idx])
                for col_idx, col_iso in enumerate(isotopes)
            }
            for row_idx, row_iso in enumerate(isotopes)
        }

    @staticmethod
    def _sanitize_spectrum_vector(
        values: object,
        *,
        name: str,
        expected_size: int | None = None,
    ) -> tuple[float, ...] | None:
        """Return a finite non-negative spectrum vector payload."""
        if values is None:
            return None
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == 0:
            return None
        if expected_size is not None and arr.size != int(expected_size):
            raise ValueError(f"{name} must have {int(expected_size)} bins.")
        arr = np.maximum(np.where(np.isfinite(arr), arr, 0.0), 0.0)
        return tuple(float(value) for value in arr)

    @staticmethod
    def _looks_like_spectrum_payload(payload: object) -> bool:
        """Return True when a mapping carries direct spectrum-bin fields."""
        if not isinstance(payload, Mapping):
            return False
        keys = {str(key) for key in payload.keys()}
        return bool(
            {
                "spectrum_counts",
                "spectrum_variance",
                "spectrum_background",
                "spectrum_response_templates_by_isotope",
            }
            & keys
        )

    @staticmethod
    def _sanitize_spectrum_payload(
        payload: Mapping[str, object] | None,
    ) -> dict[str, object] | None:
        """Return a normalized spectrum-bin payload for measurement history."""
        if payload is None:
            return None
        spectrum_counts = RotatingShieldPFEstimator._sanitize_spectrum_vector(
            payload.get("spectrum_counts"),
            name="spectrum_counts",
        )
        if spectrum_counts is None:
            return None
        bin_count = len(spectrum_counts)
        spectrum_variance = RotatingShieldPFEstimator._sanitize_spectrum_vector(
            payload.get("spectrum_variance"),
            name="spectrum_variance",
            expected_size=bin_count,
        )
        spectrum_background = RotatingShieldPFEstimator._sanitize_spectrum_vector(
            payload.get("spectrum_background"),
            name="spectrum_background",
            expected_size=bin_count,
        )
        template_payload = payload.get("spectrum_response_templates_by_isotope", {})
        templates: dict[str, tuple[float, ...]] = {}
        if isinstance(template_payload, Mapping):
            for isotope, values in template_payload.items():
                template = RotatingShieldPFEstimator._sanitize_spectrum_vector(
                    values,
                    name=f"spectrum_response_templates_by_isotope[{isotope}]",
                    expected_size=bin_count,
                )
                if template is not None:
                    templates[str(isotope)] = template
        return {
            "spectrum_counts": spectrum_counts,
            "spectrum_variance": spectrum_variance,
            "spectrum_background": spectrum_background,
            "spectrum_response_templates_by_isotope": templates,
        }

    @staticmethod
    def _pf_spectrum_update_payload_for_isotope(
        isotope: str,
        z_k: Mapping[str, float],
        spectrum_payload: Mapping[str, object] | None,
    ) -> dict[str, NDArray[np.float64]] | None:
        """Return target-isotope spectrum arrays for a PF weight update."""
        if spectrum_payload is None:
            return None
        counts_raw = spectrum_payload.get("spectrum_counts")
        templates_raw = spectrum_payload.get("spectrum_response_templates_by_isotope")
        if counts_raw is None or not isinstance(templates_raw, Mapping):
            return None
        isotope_key = str(isotope)
        if isotope_key not in templates_raw:
            return None
        counts = np.asarray(counts_raw, dtype=float).reshape(-1)
        target_template = np.asarray(templates_raw[isotope_key], dtype=float).reshape(-1)
        if counts.size == 0 or target_template.size != counts.size:
            return None
        background_raw = spectrum_payload.get("spectrum_background")
        if background_raw is None:
            background = np.zeros_like(counts, dtype=float)
        else:
            background = np.asarray(background_raw, dtype=float).reshape(-1)
            if background.size != counts.size:
                background = np.zeros_like(counts, dtype=float)
        for other_isotope, other_template_raw in templates_raw.items():
            other_key = str(other_isotope)
            if other_key == isotope_key:
                continue
            other_template = np.asarray(other_template_raw, dtype=float).reshape(-1)
            if other_template.size != counts.size:
                continue
            other_counts = max(float(z_k.get(other_key, 0.0)), 0.0)
            if other_counts > 0.0:
                background = background + other_counts * np.maximum(
                    np.where(np.isfinite(other_template), other_template, 0.0),
                    0.0,
                )
        variance = None
        variance_raw = spectrum_payload.get("spectrum_variance")
        if variance_raw is not None:
            variance_candidate = np.asarray(variance_raw, dtype=float).reshape(-1)
            if variance_candidate.size == counts.size:
                variance = np.maximum(
                    np.where(np.isfinite(variance_candidate), variance_candidate, 0.0),
                    0.0,
                )
        payload = {
            "spectrum_counts": np.maximum(
                np.where(np.isfinite(counts), counts, 0.0),
                0.0,
            ),
            "spectrum_response_template": np.maximum(
                np.where(np.isfinite(target_template), target_template, 0.0),
                0.0,
            ),
            "spectrum_background": np.maximum(
                np.where(np.isfinite(background), background, 0.0),
                0.0,
            ),
        }
        if variance is not None:
            payload["spectrum_variance"] = variance
        return payload

    @staticmethod
    def _stack_pf_spectrum_sequence_payloads(
        payloads: Sequence[dict[str, NDArray[np.float64]] | None],
    ) -> dict[str, NDArray[np.float64]] | None:
        """Stack per-view PF spectrum payloads into KxB arrays."""
        if not payloads or any(payload is None for payload in payloads):
            return None
        concrete = [payload for payload in payloads if payload is not None]
        if not concrete:
            return None
        bin_count = int(concrete[0]["spectrum_counts"].reshape(-1).size)
        if bin_count <= 0:
            return None
        required_keys = (
            "spectrum_counts",
            "spectrum_response_template",
            "spectrum_background",
        )
        stacked: dict[str, NDArray[np.float64]] = {}
        for key in required_keys:
            rows = [np.asarray(payload[key], dtype=float).reshape(-1) for payload in concrete]
            if any(row.size != bin_count for row in rows):
                return None
            stacked[key] = np.vstack(rows)
        if all("spectrum_variance" in payload for payload in concrete):
            variance_rows = [
                np.asarray(payload["spectrum_variance"], dtype=float).reshape(-1)
                for payload in concrete
            ]
            if all(row.size == bin_count for row in variance_rows):
                stacked["spectrum_variance"] = np.vstack(variance_rows)
        return stacked

    @staticmethod
    def _normalize_pair_sequence_record(
        record: Sequence[object],
    ) -> tuple[
        Dict[str, float],
        int,
        int,
        float,
        Dict[str, float] | None,
        Dict[str, Dict[str, float]] | None,
        dict[str, object] | None,
    ]:
        """Return a canonical same-pose shield-program observation record."""
        spectrum_payload = None
        if len(record) == 5:
            z_k, fe_index, pb_index, live_time_s, z_variance_k = record
            z_covariance_k = None
        elif len(record) == 6:
            z_k, fe_index, pb_index, live_time_s, z_variance_k, sixth = record
            if RotatingShieldPFEstimator._looks_like_spectrum_payload(sixth):
                z_covariance_k = None
                spectrum_payload = sixth
            else:
                z_covariance_k = sixth
        elif len(record) == 7:
            (
                z_k,
                fe_index,
                pb_index,
                live_time_s,
                z_variance_k,
                z_covariance_k,
                spectrum_payload,
            ) = record
        else:
            raise ValueError(
                "Pair sequence records must have 5 fields "
                "(z, fe, pb, live, variance), 6 fields with covariance or "
                "spectrum payload, or 7 fields with both."
            )
        return (
            {str(isotope): float(value) for isotope, value in dict(z_k).items()},
            int(fe_index),
            int(pb_index),
            float(live_time_s),
            None
            if z_variance_k is None
            else {str(isotope): float(value) for isotope, value in dict(z_variance_k).items()},
            None
            if z_covariance_k is None
            else {
                str(row_iso): {
                    str(col_iso): float(value)
                    for col_iso, value in dict(row_payload).items()
                }
                for row_iso, row_payload in dict(z_covariance_k).items()
            },
            RotatingShieldPFEstimator._sanitize_spectrum_payload(
                spectrum_payload if isinstance(spectrum_payload, Mapping) else None
            ),
        )

    @staticmethod
    def _view_covariance_for_isotope(
        isotope: str,
        *,
        sequence_length: int,
        z_view_covariance_by_isotope: Mapping[str, NDArray[np.float64]] | None,
    ) -> NDArray[np.float64] | None:
        """Return a same-station shield-view covariance matrix for one isotope."""
        if z_view_covariance_by_isotope is None:
            return None
        payload = z_view_covariance_by_isotope.get(str(isotope))
        if payload is None:
            return None
        covariance = np.asarray(payload, dtype=float)
        expected_shape = (int(sequence_length), int(sequence_length))
        if covariance.shape != expected_shape:
            raise ValueError(
                "z_view_covariance_by_isotope entries must be shaped K x K."
            )
        return 0.5 * (covariance + covariance.T)

    def adapt_strength_prior_to_observation(
        self,
        z_k: Dict[str, float],
        pose_idx: int,
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        z_variance_k: Dict[str, float] | None = None,
        completed_measurement_count: int | None = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Rescale early source-strength particles using the current count observation.

        The adaptation uses only spectrum-derived counts and the same forward
        model used by the PF. For each particle, source positions and relative
        source-strength proportions are kept fixed, while the total strength is
        set to the value implied by z ~= T * sum_j q_j K_j. This creates a
        count-conditioned proposal over strength without inserting any ground
        truth source information or a scenario-specific cps value.
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        if pose_idx < 0 or pose_idx >= len(self.poses):
            raise IndexError("pose_idx out of range")
        return self._adapt_strength_prior_at_detector(
            z_k=z_k,
            detector_pos=np.asarray(self.poses[pose_idx], dtype=float),
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=live_time_s,
            z_variance_k=z_variance_k,
            completed_measurement_count=completed_measurement_count,
        )

    def _adapt_strength_prior_at_detector(
        self,
        z_k: Dict[str, float],
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        z_variance_k: Dict[str, float] | None = None,
        completed_measurement_count: int | None = None,
    ) -> Dict[str, Dict[str, float]]:
        """Apply the count-conditioned strength proposal at an explicit detector position."""
        self.last_strength_prior_diagnostics = {}
        if not bool(self.pf_config.adaptive_strength_prior):
            return {}
        max_steps = int(self.pf_config.adaptive_strength_prior_steps)
        measurement_count = (
            len(self.measurements)
            if completed_measurement_count is None
            else max(0, int(completed_measurement_count))
        )
        if max_steps <= 0 or measurement_count >= max_steps:
            return {}
        live_time = float(live_time_s)
        if live_time <= 0.0:
            return {}
        detector = np.asarray(detector_pos, dtype=float)
        min_counts = float(self.pf_config.adaptive_strength_prior_min_counts)
        log_sigma = float(self.pf_config.adaptive_strength_prior_log_sigma)
        max_upscale = float(self.pf_config.adaptive_strength_prior_max_upscale)
        min_strength = max(float(self.pf_config.min_strength), 0.0)
        eps = 1e-12
        diagnostics: Dict[str, Dict[str, float]] = {}
        for iso, filt in self.filters.items():
            if iso not in z_k:
                continue
            particles = list(filt.continuous_particles)
            if not particles:
                continue
            observed_counts = max(float(z_k.get(iso, 0.0)), 0.0)
            target_counts = max(observed_counts, min_counts)
            floor_only_target = observed_counts < min_counts
            obs_variance = (
                0.0
                if z_variance_k is None
                else max(float(z_variance_k.get(iso, target_counts)), 0.0)
            )
            relative_count_variance = obs_variance / max(target_counts**2, eps)
            effective_log_sigma = float(
                np.sqrt(log_sigma**2 + np.log1p(relative_count_variance))
            )
            states = [particle.state for particle in particles]
            total_strengths = np.zeros(len(states), dtype=float)
            source_counts = np.zeros(len(states), dtype=float)
            eligible = np.zeros(len(states), dtype=bool)
            for index, state in enumerate(states):
                num_sources = int(state.num_sources)
                if num_sources <= 0 or state.strengths.size < num_sources:
                    continue
                strengths = np.maximum(
                    np.asarray(state.strengths[:num_sources], dtype=float),
                    0.0,
                )
                total_strength = float(np.sum(strengths))
                total_strengths[index] = total_strength
                eligible[index] = total_strength > eps
            if np.any(eligible):
                expected_counts = self.expected_counts_pair_for_states_at_detector(
                    isotope=iso,
                    detector_pos=detector,
                    fe_index=fe_index,
                    pb_index=pb_index,
                    live_time_s=live_time,
                    states=states,
                )
                backgrounds = np.asarray(
                    [float(state.background) for state in states],
                    dtype=float,
                )
                source_counts = np.maximum(
                    np.asarray(expected_counts, dtype=float) - live_time * backgrounds,
                    0.0,
                )
            before_totals: list[float] = []
            after_totals: list[float] = []
            for index, particle in enumerate(particles):
                state = particle.state
                num_sources = int(state.num_sources)
                if num_sources <= 0 or state.strengths.size < num_sources:
                    continue
                strengths = np.maximum(
                    np.asarray(state.strengths[:num_sources], dtype=float),
                    0.0,
                )
                total_strength = float(total_strengths[index])
                if total_strength <= eps:
                    continue
                proportions = strengths / total_strength
                unit_counts = float(source_counts[index]) / total_strength
                if not np.isfinite(unit_counts) or unit_counts <= eps:
                    continue
                proposed_total = target_counts / unit_counts
                if effective_log_sigma > 0.0:
                    proposed_total *= float(
                        np.random.lognormal(mean=0.0, sigma=effective_log_sigma)
                    )
                if total_strength > eps:
                    if floor_only_target:
                        proposed_total = min(proposed_total, total_strength)
                    else:
                        proposed_total = min(
                            proposed_total,
                            total_strength * max_upscale,
                        )
                proposed_total = max(float(proposed_total), min_strength * num_sources)
                if not np.isfinite(proposed_total):
                    continue
                before_totals.append(total_strength)
                after_totals.append(proposed_total)
                state.strengths[:num_sources] = proportions * proposed_total
            if after_totals:
                diagnostics[iso] = {
                    "observed_counts": float(observed_counts),
                    "target_counts": float(target_counts),
                    "observation_count_variance": float(obs_variance),
                    "effective_log_sigma": float(effective_log_sigma),
                    "floor_only_target": float(floor_only_target),
                    "max_upscale": float(max_upscale),
                    "before_median_strength": float(np.median(before_totals)),
                    "after_median_strength": float(np.median(after_totals)),
                    "particles_changed": float(len(after_totals)),
                }
        self.last_strength_prior_diagnostics = diagnostics
        return diagnostics

    def continuous_kernel(
        self,
        *,
        detector_aperture_samples: int | None = None,
        use_gpu: bool | None = None,
    ) -> ContinuousKernel:
        """Build the shared ContinuousKernel for PF, planning, and diagnostics."""
        active_aperture_samples = (
            int(self.detector_aperture_samples)
            if detector_aperture_samples is None
            else int(detector_aperture_samples)
        )
        active_use_gpu = (
            bool(self.pf_config.use_gpu) if use_gpu is None else bool(use_gpu)
        )
        return ContinuousKernel(
            mu_by_isotope=self.mu_by_isotope,
            shield_params=self.shield_params,
            orientations=self.normals,
            use_gpu=active_use_gpu,
            gpu_device=str(self.pf_config.gpu_device),
            gpu_dtype=str(self.pf_config.gpu_dtype),
            obstacle_grid=self.obstacle_grid,
            obstacle_height_m=self.obstacle_height_m,
            obstacle_mu_by_isotope=self.obstacle_mu_by_isotope,
            obstacle_buildup_coeff=self.obstacle_buildup_coeff,
            detector_radius_m=self.detector_radius_m,
            detector_aperture_radius_m=self.detector_aperture_radius_m,
            detector_aperture_samples=max(1, active_aperture_samples),
            detector_aperture_sampling=self.detector_aperture_sampling,
            source_extent_radius_m=self.source_extent_radius_m,
            source_extent_samples=self.source_extent_samples,
            line_mu_by_isotope=self.line_mu_by_isotope,
            transport_response_model=self.transport_response_model,
        )

    def _continuous_kernel(self) -> ContinuousKernel:
        """Build a ContinuousKernel matching the estimator observation model."""
        return self.continuous_kernel()

    def expected_counts_pair_for_states(
        self,
        isotope: str,
        pose_idx: int,
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        states: Sequence[IsotopeState],
    ) -> NDArray[np.float64]:
        """
        Compute Λ_{k,h}^{(n)} for an isotope over a list of states at a pose.

        Uses torch acceleration when enabled; otherwise falls back to CPU kernels.
        """
        if pose_idx < 0 or pose_idx >= len(self.poses):
            raise IndexError("pose_idx out of range")
        detector_pos = np.asarray(self.poses[pose_idx], dtype=float)
        return self.expected_counts_pair_for_states_at_detector(
            isotope=isotope,
            detector_pos=detector_pos,
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=live_time_s,
            states=states,
        )

    def expected_counts_pair_for_states_at_detector(
        self,
        isotope: str,
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        states: Sequence[IsotopeState],
    ) -> NDArray[np.float64]:
        """
        Compute Λ for a state subset at an arbitrary detector position.

        This helper keeps candidate-pose and shield-selection scoring on the
        same GPU-accelerated transport approximation as normal PF updates,
        even when a planning particle subset is used.
        """
        if not states:
            return np.zeros(0, dtype=float)
        kernel = self._continuous_kernel()
        detector_pos = np.asarray(detector_pos, dtype=float)
        use_gpu = False
        if self.pf_config.use_gpu and int(self.num_orientations) == 8:
            try:
                use_gpu = bool(self._gpu_enabled())
            except RuntimeError:
                use_gpu = False
        if not use_gpu:
            values = np.zeros(len(states), dtype=float)
            source_scale = self.response_scale_for_isotope(
                isotope,
                fe_index=fe_index,
                pb_index=pb_index,
            )
            for idx, state in enumerate(states):
                rate = float(state.background)
                for pos, strength in zip(
                    state.positions[: state.num_sources],
                    state.strengths[: state.num_sources],
                ):
                    rate += (
                        source_scale
                        * float(strength)
                        * kernel.kernel_value_pair(
                            isotope=isotope,
                            detector_pos=detector_pos,
                            source_pos=pos,
                            fe_index=fe_index,
                            pb_index=pb_index,
                        )
                    )
                values[idx] = float(live_time_s) * rate
            return values
        from pf import gpu_utils

        device = gpu_utils.resolve_device(self.pf_config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.pf_config.gpu_dtype)
        positions, strengths, backgrounds, mask = gpu_utils.pack_states(
            states,
            device=device,
            dtype=dtype,
        )
        lam_t = kernel.expected_counts_pair_for_packed_states_torch(
            isotope=isotope,
            detector_pos=detector_pos,
            positions=positions,
            strengths=strengths,
            backgrounds=backgrounds,
            mask=mask,
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=live_time_s,
            source_scale=self.response_scale_for_isotope(
                isotope,
                fe_index=fe_index,
                pb_index=pb_index,
            ),
            device=device,
            dtype=dtype,
        )
        return lam_t.detach().cpu().numpy()

    def expected_counts_all_pairs_for_states_at_detector(
        self,
        isotope: str,
        detector_pos: NDArray[np.float64],
        live_time_s: float,
        states: Sequence[IsotopeState],
    ) -> NDArray[np.float64]:
        """
        Compute expected counts for all Fe/Pb orientation pairs for state subsets.

        The returned array is shaped ``(num_pairs, num_states)`` and uses the same
        continuous kernel, spherical-octant shield geometry, detector aperture,
        response scale, and obstacle attenuation as the per-pair helper.
        """
        num_pairs = int(self.num_orientations) * int(self.num_orientations)
        if not states:
            return np.zeros((num_pairs, 0), dtype=float)
        kernel = self._continuous_kernel()
        detector_pos = np.asarray(detector_pos, dtype=float)
        use_gpu = False
        if self.pf_config.use_gpu and int(self.num_orientations) == 8:
            try:
                use_gpu = bool(self._gpu_enabled())
            except RuntimeError:
                use_gpu = False
        if not use_gpu:
            rows: list[NDArray[np.float64]] = []
            for fe_index in range(int(self.num_orientations)):
                for pb_index in range(int(self.num_orientations)):
                    rows.append(
                        self.expected_counts_pair_for_states_at_detector(
                            isotope=isotope,
                            detector_pos=detector_pos,
                            fe_index=fe_index,
                            pb_index=pb_index,
                            live_time_s=live_time_s,
                            states=states,
                        )
                    )
            return np.vstack(rows).astype(float, copy=False)

        from pf import gpu_utils

        device = gpu_utils.resolve_device(self.pf_config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.pf_config.gpu_dtype)
        positions, strengths, backgrounds, mask = gpu_utils.pack_states(
            states,
            device=device,
            dtype=dtype,
        )
        num_orients = int(self.num_orientations)
        fe_indices = np.repeat(np.arange(num_orients), num_orients)
        pb_indices = np.tile(np.arange(num_orients), num_orients)
        lam_t = kernel.expected_counts_all_pairs_for_packed_states_torch(
            isotope=isotope,
            detector_pos=detector_pos,
            positions=positions,
            strengths=strengths,
            backgrounds=backgrounds,
            mask=mask,
            live_time_s=live_time_s,
            source_scale=self.response_scales_for_measurements(
                isotope,
                fe_indices,
                pb_indices,
            ),
            device=device,
            dtype=dtype,
        )
        return lam_t.detach().cpu().numpy().astype(float, copy=False)

    def shield_selection_batch_grids(
        self,
        pose_idx: int,
        *,
        live_time_s: float,
        max_particles: int | None = None,
        particles_by_isotope: Dict[str, Tuple[List[IsotopeState], NDArray[np.float64]]]
        | None = None,
        alpha_by_isotope: Dict[str, float] | None = None,
        variance_floor: float = 1.0,
        include_count_quantiles: bool = True,
    ) -> tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]:
        """
        Return all-pair shield-signature and observability count grids.

        This batches the same per-pair expected-count calculation used by
        ``orientation_signature_separation_score`` and
        ``expected_observation_counts_by_isotope_at_pair``. It is a planning
        acceleration only; it does not change the observation model used for PF
        updates.
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        if pose_idx < 0 or pose_idx >= len(self.poses):
            raise IndexError("pose_idx out of range")
        detector_pos = np.asarray(self.poses[int(pose_idx)], dtype=float)
        num_orients = int(self.num_orientations)
        num_pairs = num_orients * num_orients
        eps = 1e-12
        alphas = alpha_by_isotope or {iso: 1.0 for iso in self.filters}
        alpha_sum = sum(float(v) for v in alphas.values()) or 1.0
        floor = max(float(variance_floor), eps)
        signature_flat = np.zeros(num_pairs, dtype=float)
        count_quantiles: Dict[str, NDArray[np.float64]] = {}
        if particles_by_isotope is None:
            particles_by_isotope = self.planning_particles(
                max_particles=max_particles,
                method=self.pf_config.planning_method,
            )

        for iso, filt in self.filters.items():
            if getattr(filt, "is_converged", False) and getattr(
                filt.config, "converge_enable", False
            ):
                continue
            if iso in particles_by_isotope:
                states, weights = particles_by_isotope[iso]
            else:
                if not filt.continuous_particles:
                    continue
                states = [p.state for p in filt.continuous_particles]
                weights = filt.continuous_weights
            if not states:
                continue
            weights_arr = np.asarray(weights, dtype=float)
            weight_sum = float(np.sum(weights_arr))
            if weight_sum <= eps:
                weights_arr = np.ones(len(states), dtype=float) / max(len(states), 1)
            else:
                weights_arr = weights_arr / weight_sum
            lambdas = self.expected_counts_all_pairs_for_states_at_detector(
                isotope=iso,
                detector_pos=detector_pos,
                live_time_s=float(live_time_s),
                states=states,
            )
            if lambdas.size == 0:
                continue
            means = lambdas @ weights_arr
            centered = lambdas - means[:, None]
            variances = (centered * centered) @ weights_arr
            signature_flat += (
                float(alphas.get(iso, 1.0))
                / alpha_sum
                * np.maximum(variances, 0.0)
                / np.maximum(means, floor)
            )
            if include_count_quantiles:
                quantile = float(self.pf_config.pose_min_observation_quantile)
                count_quantiles[iso] = np.asarray(
                    [
                        _weighted_quantile(lambdas[pair_idx], weights_arr, quantile)
                        for pair_idx in range(num_pairs)
                    ],
                    dtype=float,
                ).reshape(num_orients, num_orients)
        return (
            np.maximum(signature_flat, 0.0).reshape(num_orients, num_orients),
            count_quantiles,
        )

    def expected_observation_counts_by_isotope_at_pose(
        self,
        pose_xyz: NDArray[np.float64],
        *,
        live_time_s: float,
        fe_pb_pairs: Sequence[tuple[int, int]] | None = None,
        aggregate: str = "max",
        max_particles: int | None = None,
    ) -> Dict[str, float]:
        """
        Return posterior-mean expected counts for each isotope at a candidate pose.

        The value for one isotope is computed from the same inverse-square,
        spherical shield, and obstacle attenuation model used by PF updates.
        Across shield pairs, ``aggregate="max"`` returns the best achievable
        expected count at that pose, while ``aggregate="mean"`` returns the
        orientation-average expected count. Each pair uses a weighted posterior
        quantile rather than the posterior mean, so a few high-strength outlier
        particles cannot make the pose look observable for every isotope.
        """
        detector = np.asarray(pose_xyz, dtype=float)
        if detector.shape != (3,):
            raise ValueError("pose_xyz must be shape (3,).")
        live_time = float(live_time_s)
        if live_time <= 0.0:
            return {iso: 0.0 for iso in self.isotopes}
        aggregate = str(aggregate).strip().lower()
        if aggregate not in {"max", "mean"}:
            raise ValueError("aggregate must be max or mean.")
        num_orients = max(1, int(self.num_orientations))
        if fe_pb_pairs is None:
            pairs = [
                (fe_index, pb_index)
                for fe_index in range(num_orients)
                for pb_index in range(num_orients)
            ]
        else:
            pairs = [(int(fe), int(pb)) for fe, pb in fe_pb_pairs]
        if not pairs:
            return {iso: 0.0 for iso in self.isotopes}
        particles = self.planning_particles(max_particles=max_particles)
        counts_by_isotope: Dict[str, float] = {}
        eps = 1e-12
        for iso in self.isotopes:
            filt = self.filters.get(iso)
            use_gpu_quantile = False
            if (
                max_particles is None
                and filt is not None
                and filt.continuous_particles
                and self.pf_config.use_gpu
            ):
                try:
                    use_gpu_quantile = bool(self._gpu_enabled())
                except RuntimeError:
                    use_gpu_quantile = False
            if use_gpu_quantile:
                weights_arr = np.asarray(filt.continuous_weights, dtype=float)
                weight_sum = float(np.sum(weights_arr))
                if weight_sum <= eps:
                    weights_arr = np.ones(len(weights_arr), dtype=float) / max(
                        len(weights_arr),
                        1,
                    )
                else:
                    weights_arr = weights_arr / weight_sum
                pair_means = []
                for fe_index, pb_index in pairs:
                    lambdas = filt._continuous_expected_counts_pair_at_pose(
                        detector_pos=detector,
                        fe_index=fe_index,
                        pb_index=pb_index,
                        live_time_s=live_time,
                    )
                    pair_means.append(
                        _weighted_quantile(
                            lambdas,
                            weights_arr,
                            self.pf_config.pose_min_observation_quantile,
                        )
                    )
                if aggregate == "mean":
                    counts_by_isotope[iso] = float(np.mean(pair_means))
                else:
                    counts_by_isotope[iso] = float(np.max(pair_means))
                continue
            if iso not in particles:
                counts_by_isotope[iso] = 0.0
                continue
            states, weights = particles[iso]
            if not states:
                counts_by_isotope[iso] = 0.0
                continue
            weights_arr = np.asarray(weights, dtype=float)
            weight_sum = float(np.sum(weights_arr))
            if weight_sum <= eps:
                weights_arr = np.ones(len(states), dtype=float) / max(len(states), 1)
            else:
                weights_arr = weights_arr / weight_sum
            pair_means: list[float] = []
            for fe_index, pb_index in pairs:
                lambdas = self.expected_counts_pair_for_states_at_detector(
                    isotope=iso,
                    detector_pos=detector,
                    fe_index=fe_index,
                    pb_index=pb_index,
                    live_time_s=live_time,
                    states=states,
                )
                pair_means.append(
                    _weighted_quantile(
                        lambdas,
                        weights_arr,
                        self.pf_config.pose_min_observation_quantile,
                    )
                )
            if aggregate == "mean":
                counts_by_isotope[iso] = float(np.mean(pair_means))
            else:
                counts_by_isotope[iso] = float(np.max(pair_means))
        return counts_by_isotope

    def expected_observation_counts_by_isotope_at_pair(
        self,
        pose_idx: int,
        fe_index: int,
        pb_index: int,
        *,
        live_time_s: float,
        max_particles: int | None = None,
    ) -> Dict[str, float]:
        """Return posterior-quantile expected counts for one Fe/Pb pair."""
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        pose = np.asarray(self.poses[int(pose_idx)], dtype=float)
        return self.expected_observation_counts_by_isotope_at_pose(
            pose,
            live_time_s=float(live_time_s),
            fe_pb_pairs=[(int(fe_index), int(pb_index))],
            aggregate="max",
            max_particles=max_particles,
        )

    def orientation_signature_separation_score(
        self,
        pose_idx: int,
        fe_index: int,
        pb_index: int,
        *,
        live_time_s: float,
        particles_by_isotope: Dict[str, Tuple[List[IsotopeState], NDArray[np.float64]]]
        | None = None,
        alpha_by_isotope: Dict[str, float] | None = None,
        variance_floor: float = 1.0,
    ) -> float:
        """
        Return a shield-signature separation score for one orientation pair.

        The score is a weighted posterior variance of predicted counts,
        normalized by the mean count scale. It favors shield postures whose
        response differs across currently plausible source hypotheses.
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        eps = 1e-12
        alphas = alpha_by_isotope or {iso: 1.0 for iso in self.filters}
        alpha_sum = sum(float(v) for v in alphas.values()) or 1.0
        score = 0.0
        floor = max(float(variance_floor), eps)
        for iso, filt in self.filters.items():
            if getattr(filt, "is_converged", False) and getattr(
                filt.config, "converge_enable", False
            ):
                continue
            if particles_by_isotope is not None and iso in particles_by_isotope:
                states, weights = particles_by_isotope[iso]
            else:
                states = [p.state for p in filt.continuous_particles]
                weights = filt.continuous_weights
            if not states:
                continue
            weights_arr = np.asarray(weights, dtype=float)
            weights_arr = weights_arr / max(float(np.sum(weights_arr)), eps)
            lambdas = self.expected_counts_pair_for_states(
                isotope=iso,
                pose_idx=int(pose_idx),
                fe_index=int(fe_index),
                pb_index=int(pb_index),
                live_time_s=float(live_time_s),
                states=states,
            )
            if lambdas.size == 0:
                continue
            mean = float(np.sum(weights_arr * lambdas))
            var = float(np.sum(weights_arr * (lambdas - mean) ** 2))
            score += (
                float(alphas.get(iso, 1.0))
                / alpha_sum
                * max(var, 0.0)
                / max(mean, floor)
            )
        return float(max(score, 0.0))

    def planning_particles(
        self,
        max_particles: int | None = None,
        method: str | None = None,
        rng: np.random.Generator | None = None,
    ) -> Dict[str, Tuple[List[IsotopeState], NDArray[np.float64]]]:
        """
        Select per-isotope particle subsets for orientation evaluation.

        Args:
            max_particles: cap on particles per isotope; None uses config default.
            method: "top_weight" or "resample"; None uses config default.
            rng: optional RNG for resampling.
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        if max_particles is None:
            max_particles = self.pf_config.planning_particles
        method = method or self.pf_config.planning_method
        rng = rng or np.random.default_rng()
        subsets: Dict[str, Tuple[List[IsotopeState], NDArray[np.float64]]] = {}
        for iso, filt in self.filters.items():
            if getattr(filt, "is_converged", False) and getattr(
                filt.config, "converge_enable", False
            ):
                continue
            if not filt.continuous_particles:
                continue
            weights = filt.continuous_weights
            total = float(np.sum(weights))
            if total <= 0.0:
                continue
            weights = weights / total
            n_particles = len(weights)
            if (
                max_particles is None
                or max_particles <= 0
                or max_particles >= n_particles
            ):
                states = [p.state.copy() for p in filt.continuous_particles]
                subsets[iso] = (states, weights)
                continue
            if method == "top_weight":
                idx = np.argsort(weights)[::-1][:max_particles]
                sel_weights = weights[idx]
                sel_weights = sel_weights / max(np.sum(sel_weights), 1e-12)
            elif method == "resample":
                idx = rng.choice(n_particles, size=max_particles, p=weights)
                sel_weights = np.ones(max_particles, dtype=float) / max_particles
            else:
                raise ValueError(
                    f"Unknown planning particle selection method: {method}"
                )
            states = [filt.continuous_particles[i].state.copy() for i in idx]
            subsets[iso] = (states, sel_weights)
        return subsets

    def weight_entropy_ratio(
        self,
        particles_by_isotope: Dict[str, Tuple[List[IsotopeState], NDArray[np.float64]]]
        | None = None,
    ) -> float:
        """
        Return the mean normalized weight entropy across isotopes.

        The entropy ratio is H(w)/log(N) in [0, 1]. Lower values indicate a more
        concentrated posterior (less multi-modality).
        """
        entropies: List[float] = []
        eps = 1e-12
        for iso, filt in self.filters.items():
            if particles_by_isotope is not None and iso in particles_by_isotope:
                _, weights = particles_by_isotope[iso]
            else:
                if not filt.continuous_particles:
                    continue
                weights = filt.continuous_weights
            weights = np.asarray(weights, dtype=float)
            if weights.size == 0:
                continue
            weights = weights / max(float(np.sum(weights)), eps)
            if weights.size == 1:
                entropies.append(0.0)
                continue
            entropy = float(-np.sum(weights * np.log(weights + eps)))
            entropies.append(entropy / max(np.log(weights.size), eps))
        if not entropies:
            return 0.0
        return float(np.mean(entropies))

    def add_measurement_pose(
        self, pose: NDArray[np.float64], reset_filters: bool = True
    ) -> None:
        """Register a new measurement pose and invalidate the kernel cache."""
        self.poses.append(np.asarray(pose, dtype=float))
        # Rebuild lazily on the next access.
        self.kernel_cache = None
        if reset_filters:
            self.filters = {}
        self._invalidate_report_cache()

    def restrict_isotopes(
        self,
        active_isotopes: Sequence[str],
        *,
        allow_empty: bool = False,
    ) -> None:
        """
        Restrict estimator state to the specified isotopes.

        This drops filters and cached estimates for isotopes that are not in
        active_isotopes while preserving the original isotope ordering. When
        allow_empty is true, no isotope PFs remain active until add_isotopes()
        is called by the spectrum-detection gate.
        """
        active_set = set(active_isotopes)
        if not active_set and not allow_empty:
            raise ValueError("active_isotopes must contain at least one isotope.")
        self.isotopes = [iso for iso in self.all_isotopes if iso in active_set]
        if self.filters:
            self.filters = {
                iso: filt for iso, filt in self.filters.items() if iso in active_set
            }
        if self.history_estimates:
            self.history_estimates = [
                {iso: val for iso, val in est.items() if iso in active_set}
                for est in self.history_estimates
            ]
        self._invalidate_report_cache()

    def add_isotopes(self, new_isotopes: Sequence[str]) -> None:
        """
        Add isotopes to the estimator and initialize their PF filters.

        This is useful when new isotopes are detected after an initial restriction.
        """
        requested = set(new_isotopes)
        active_set = set(self.isotopes) | requested
        to_add = [
            iso
            for iso in self.all_isotopes
            if iso in requested and iso not in self.isotopes
        ]
        if not to_add:
            return
        self.isotopes = [iso for iso in self.all_isotopes if iso in active_set]
        if self.kernel_cache is None and self.poses:
            self._ensure_kernel_cache()
        if self.kernel_cache is None:
            return
        pf_conf = self._build_pf_config()
        for iso in to_add:
            if iso not in self.filters:
                self.filters[iso] = self._build_filter(iso, pf_conf)
        self._invalidate_report_cache()

    def update(
        self,
        z_k: Dict[str, float],
        pose_idx: int,
        orient_idx: int,
        live_time_s: float,
    ) -> None:
        """
        Update per-isotope PFs using isotope-wise counts z_k.

        z_k must come from the spectrum unfolding pipeline (Sec. 2.5.7); this method
        never fabricates observations from geometric kernels or ground truth.
        """
        raise RuntimeError(
            "Single-orientation updates are disabled. Use update_pair or short_time_update "
            "with Fe/Pb indices to preserve the 64-orientation shield model."
        )

    def predict(self) -> None:
        """Run the prediction step for all PFs."""
        for f in self.filters.values():
            f.predict()
        self._invalidate_report_cache()

    def short_time_update(
        self,
        z_k: Dict[str, float],
        pose_idx: int,
        RFe: NDArray[np.float64],
        RPb: NDArray[np.float64],
        live_time_s: float | None = None,
    ) -> None:
        """
        Apply a short-time measurement update (Sec. 3.4.3).

        - Use shield orientations (RFe, RPb) and isotope-wise counts z_k.
        - T_k defaults to pf_config.short_time_s unless specified.
        - z_k must come from the spectrum pipeline (Sec. 2.5.7), not from geometry.
        """
        duration = (
            live_time_s if live_time_s is not None else self.pf_config.short_time_s
        )
        fe_index = octant_index_from_rotation(RFe)
        pb_index = octant_index_from_rotation(RPb)
        self.update_pair(
            z_k=z_k,
            pose_idx=pose_idx,
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=duration,
        )

    def update_pair(
        self,
        z_k: Dict[str, float],
        pose_idx: int,
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        z_variance_k: Dict[str, float] | None = None,
        z_covariance_k: Dict[str, Dict[str, float]] | None = None,
        spectrum_payload: Mapping[str, object] | None = None,
    ) -> None:
        """
        Update PFs using Fe/Pb orientation indices (RFe, RPb) and isotope-wise counts z_k.

        This feeds the continuous 3D PF path (Sec. 3.3.3) with Λ computed via expected_counts_pair.
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        effective_variance_k, sanitized_covariance_k = (
            self._project_observation_covariance_to_variance(
                z_k,
                z_variance_k,
                z_covariance_k,
            )
        )
        sanitized_spectrum_payload = self._sanitize_spectrum_payload(spectrum_payload)
        if self._defer_resample_birth:
            self.last_strength_prior_diagnostics = {}
        else:
            self.adapt_strength_prior_to_observation(
                z_k=z_k,
                pose_idx=pose_idx,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
                z_variance_k=effective_variance_k,
            )
        for iso, val in z_k.items():
            if iso not in self.filters:
                continue
            pf_spectrum_payload = self._pf_spectrum_update_payload_for_isotope(
                iso,
                z_k,
                sanitized_spectrum_payload,
            )
            # Use continuous PF update that relies on spectrum-unfolded counts.
            self.filters[iso].update_continuous_pair(
                z_obs=val,
                pose_idx=pose_idx,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
                observation_count_variance=(
                    0.0
                    if effective_variance_k is None
                    else float(effective_variance_k.get(iso, 0.0))
                ),
                step_idx=len(self.measurements),
                defer_resample=bool(self._defer_resample_birth),
                **({} if pf_spectrum_payload is None else pf_spectrum_payload),
            )
        self._invalidate_report_cache()
        self.measurements.append(
            MeasurementRecord(
                z_k={iso: float(v) for iso, v in z_k.items()},
                pose_idx=pose_idx,
                orient_idx=fe_index,
                live_time_s=live_time_s,
                fe_index=fe_index,
                pb_index=pb_index,
                z_variance_k=None
                if effective_variance_k is None
                else {iso: float(v) for iso, v in effective_variance_k.items()},
                z_covariance_k=sanitized_covariance_k,
                ig_value=None,
                spectrum_counts=(
                    None
                    if sanitized_spectrum_payload is None
                    else sanitized_spectrum_payload.get("spectrum_counts")
                ),
                spectrum_variance=(
                    None
                    if sanitized_spectrum_payload is None
                    else sanitized_spectrum_payload.get("spectrum_variance")
                ),
                spectrum_background=(
                    None
                    if sanitized_spectrum_payload is None
                    else sanitized_spectrum_payload.get("spectrum_background")
                ),
                spectrum_response_templates_by_isotope=(
                    None
                    if sanitized_spectrum_payload is None
                    else sanitized_spectrum_payload.get(
                        "spectrum_response_templates_by_isotope"
                    )
                ),
            )
        )
        if self._defer_resample_birth:
            self._deferred_measurement_count += 1
        else:
            self.refresh_sparse_poisson_evidence()
            self._apply_birth_death()
        self._invalidate_report_cache()
        if not self._defer_resample_birth:
            self._record_history_estimate(len(self.measurements))
            self.record_report_snapshot(
                label=f"measurement_{len(self.measurements)}",
                allow_heavy_estimate=False,
            )

    def begin_deferred_pose_update(self) -> None:
        """Start a station-level update that delays only structural moves."""
        self._defer_resample_birth = True
        self._deferred_measurement_count = 0

    def finalize_deferred_pose_update(self) -> int:
        """
        Finish a station-level delayed update and return finalized measurements.

        During a delayed update, each shield posture updates particle weights
        immediately and may resample on ESS. This method then performs
        station-level adaptation, label alignment, and residual-gated
        birth/death once.
        """
        count = int(self._deferred_measurement_count)
        self._defer_resample_birth = False
        self._deferred_measurement_count = 0
        if count <= 0:
            return 0
        pre_finalize_estimates = self.estimates(use_pre_finalize_guard=False)
        for filt in self.filters.values():
            filt.finalize_deferred_update()
        birth_context_count = count + max(
            0,
            int(self._previous_deferred_measurement_count),
        )
        self.refresh_sparse_poisson_evidence()
        self._apply_birth_death(birth_window_override=birth_context_count)
        self._invalidate_report_cache()
        post_finalize_estimates = self.estimates(use_pre_finalize_guard=False)
        self._update_pre_finalize_guard(
            pre_finalize_estimates,
            post_finalize_estimates,
        )
        self._invalidate_report_cache()
        self._previous_deferred_measurement_count = count
        self._record_history_estimate(len(self.measurements))
        self.record_report_snapshot(
            label=f"measurement_{len(self.measurements)}",
            allow_heavy_estimate=False,
        )
        return count

    def update_pair_sequence(
        self,
        records: Sequence[Sequence[object]],
        *,
        pose_idx: int,
        z_view_covariance_by_isotope: Mapping[str, NDArray[np.float64]] | None = None,
    ) -> None:
        """
        Jointly update PFs from a same-pose shield-orientation sequence.

        Each record is ``(z_k, fe_index, pb_index, live_time_s, z_variance_k)``.
        A sixth ``z_covariance_k`` field may be supplied for same-spectrum
        isotope covariance.
        A seventh spectrum payload field may be supplied for direct spectrum-bin
        sparse evidence.
        ``z_view_covariance_by_isotope`` may also supply KxK same-station
        shield-view covariance for each isotope. The joint update uses one
        station-level likelihood over all postures and only applies birth/death
        after the full shield program is observed.
        """
        if not records:
            return
        sequence_start = time.perf_counter()
        stage_wall: Dict[str, float] = {}
        stage_start = sequence_start
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        normalized_records = []
        for record in records:
            (
                z_k,
                fe_index,
                pb_index,
                live_time_s,
                z_variance_k,
                z_covariance_k,
                spectrum_payload,
            ) = self._normalize_pair_sequence_record(record)
            effective_variance_k, sanitized_covariance_k = (
                self._project_observation_covariance_to_variance(
                    z_k,
                    z_variance_k,
                    z_covariance_k,
                )
            )
            normalized_records.append(
                (
                    z_k,
                    fe_index,
                    pb_index,
                    live_time_s,
                    effective_variance_k,
                    sanitized_covariance_k,
                    spectrum_payload,
                )
            )
        stage_wall["normalize_records"] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        base_measurement_count = len(self.measurements)
        max_strength_prior_steps = int(self.pf_config.adaptive_strength_prior_steps)
        self.last_strength_prior_diagnostics = {}
        for offset, (
            z_k,
            fe_index,
            pb_index,
            live_time_s,
            z_variance_k,
            _z_covariance_k,
            _spectrum_payload,
        ) in enumerate(normalized_records):
            completed_measurement_count = base_measurement_count + int(offset)
            if completed_measurement_count >= max_strength_prior_steps:
                break
            self.adapt_strength_prior_to_observation(
                z_k=z_k,
                pose_idx=pose_idx,
                fe_index=int(fe_index),
                pb_index=int(pb_index),
                live_time_s=float(live_time_s),
                z_variance_k=z_variance_k,
                completed_measurement_count=completed_measurement_count,
            )
        stage_wall["adaptive_strength_prior"] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        step_idx = len(self.measurements)
        tasks: list[
            tuple[
                str,
                IsotopeParticleFilter,
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.float64] | None,
                int,
                int,
                NDArray[np.float64] | None,
                NDArray[np.float64] | None,
                NDArray[np.float64] | None,
                NDArray[np.float64] | None,
            ]
        ] = []
        for iso, filt in self.filters.items():
            z_arr = np.asarray(
                [
                    float(z_k.get(iso, 0.0))
                    for z_k, _, _, _, _, _, _ in normalized_records
                ],
                dtype=float,
            )
            var_arr = np.asarray(
                [
                    0.0 if z_variance_k is None else float(z_variance_k.get(iso, 0.0))
                    for _, _, _, _, z_variance_k, _, _ in normalized_records
                ],
                dtype=float,
            )
            fe_arr = np.asarray(
                [int(fe_index) for _, fe_index, _, _, _, _, _ in normalized_records],
                dtype=int,
            )
            pb_arr = np.asarray(
                [int(pb_index) for _, _, pb_index, _, _, _, _ in normalized_records],
                dtype=int,
            )
            live_arr = np.asarray(
                [
                    float(live_time_s)
                    for _, _, _, live_time_s, _, _, _ in normalized_records
                ],
                dtype=float,
            )
            view_covariance = self._view_covariance_for_isotope(
                iso,
                sequence_length=z_arr.size,
                z_view_covariance_by_isotope=z_view_covariance_by_isotope,
            )
            sequence_spectrum_payload = self._stack_pf_spectrum_sequence_payloads(
                [
                    self._pf_spectrum_update_payload_for_isotope(
                        iso,
                        z_k,
                        spectrum_payload,
                    )
                    for (
                        z_k,
                        _fe_index,
                        _pb_index,
                        _live_time_s,
                        _z_variance_k,
                        _z_covariance_k,
                        spectrum_payload,
                    ) in normalized_records
                ]
            )
            tasks.append(
                (
                    iso,
                    filt,
                    z_arr,
                    fe_arr,
                    pb_arr,
                    live_arr,
                    var_arr,
                    view_covariance,
                    int(pose_idx),
                    int(step_idx),
                    None
                    if sequence_spectrum_payload is None
                    else sequence_spectrum_payload["spectrum_counts"],
                    None
                    if sequence_spectrum_payload is None
                    else sequence_spectrum_payload["spectrum_response_template"],
                    None
                    if sequence_spectrum_payload is None
                    else sequence_spectrum_payload["spectrum_background"],
                    None
                    if sequence_spectrum_payload is None
                    else sequence_spectrum_payload.get("spectrum_variance"),
                )
            )
        stage_wall["build_isotope_tasks"] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        worker_count = self._structural_update_worker_count(len(tasks))
        self.last_pair_sequence_update_workers = int(worker_count)
        if worker_count <= 1:
            for task in tasks:
                self._run_isotope_pair_sequence_update(task)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                list(executor.map(self._run_isotope_pair_sequence_update, tasks))
        self.last_pair_sequence_update_wall_s = time.perf_counter() - stage_start
        stage_wall["isotope_sequence_update"] = self.last_pair_sequence_update_wall_s
        stage_start = time.perf_counter()
        self._invalidate_report_cache()
        for (
            z_k,
            fe_index,
            pb_index,
            live_time_s,
            z_variance_k,
            z_covariance_k,
            spectrum_payload,
        ) in normalized_records:
            self.measurements.append(
                MeasurementRecord(
                    z_k={iso: float(v) for iso, v in z_k.items()},
                    pose_idx=pose_idx,
                    orient_idx=int(fe_index),
                    live_time_s=float(live_time_s),
                    fe_index=int(fe_index),
                    pb_index=int(pb_index),
                    z_variance_k=None
                    if z_variance_k is None
                    else {iso: float(v) for iso, v in z_variance_k.items()},
                    z_covariance_k=z_covariance_k,
                    ig_value=None,
                    spectrum_counts=(
                        None
                        if spectrum_payload is None
                        else spectrum_payload.get("spectrum_counts")
                    ),
                    spectrum_variance=(
                        None
                        if spectrum_payload is None
                        else spectrum_payload.get("spectrum_variance")
                    ),
                    spectrum_background=(
                        None
                        if spectrum_payload is None
                        else spectrum_payload.get("spectrum_background")
                    ),
                    spectrum_response_templates_by_isotope=(
                        None
                        if spectrum_payload is None
                        else spectrum_payload.get(
                            "spectrum_response_templates_by_isotope"
                        )
                    ),
                )
            )
        stage_wall["append_measurements"] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        self.refresh_sparse_poisson_evidence()
        stage_wall["sparse_poisson_refresh"] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        self._apply_birth_death()
        stage_wall["birth_death"] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        self._invalidate_report_cache()
        self._record_history_estimate(len(self.measurements))
        stage_wall["history_estimate"] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        self.record_report_snapshot(
            label=f"measurement_{len(self.measurements)}",
            allow_heavy_estimate=False,
        )
        stage_wall["report_snapshot"] = time.perf_counter() - stage_start
        stage_wall["total"] = time.perf_counter() - sequence_start
        self.last_pair_sequence_stage_wall_s = {
            key: float(value) for key, value in stage_wall.items()
        }

    @staticmethod
    def _run_isotope_pair_sequence_update(
        task: tuple[
            str,
            IsotopeParticleFilter,
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64] | None,
            int,
            int,
            NDArray[np.float64] | None,
            NDArray[np.float64] | None,
            NDArray[np.float64] | None,
            NDArray[np.float64] | None,
        ],
    ) -> None:
        """Run one isotope's same-station shield-program likelihood update."""
        (
            _isotope,
            filt,
            z_arr,
            fe_arr,
            pb_arr,
            live_arr,
            var_arr,
            view_covariance,
            pose_idx,
            step_idx,
            spectrum_counts,
            spectrum_response_template,
            spectrum_background,
            spectrum_variance,
        ) = task
        filt.update_continuous_pair_sequence(
            z_obs=z_arr,
            pose_idx=pose_idx,
            fe_indices=fe_arr,
            pb_indices=pb_arr,
            live_times_s=live_arr,
            observation_count_variances=var_arr,
            observation_count_covariance=view_covariance,
            step_idx=step_idx,
            spectrum_counts=spectrum_counts,
            spectrum_response_template=spectrum_response_template,
            spectrum_background=spectrum_background,
            spectrum_variance=spectrum_variance,
        )

    def update_pair_at_pose(
        self,
        z_k: Dict[str, float],
        detector_pos: NDArray[np.float64],
        pose_idx: int,
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        z_variance_k: Dict[str, float] | None = None,
        z_covariance_k: Dict[str, Dict[str, float]] | None = None,
    ) -> None:
        """
        Update PFs using explicit detector position without rebuilding the kernel cache.

        This avoids kernel-cache growth with many poses by using per-pose updates.
        """
        if pose_idx < 0 or pose_idx >= len(self.poses):
            raise IndexError("pose_idx out of range")
        detector_pos = np.asarray(detector_pos, dtype=float)
        if not self.filters:
            pf_conf = self._build_pf_config()
            for iso in self.isotopes:
                self.filters[iso] = IsotopeParticleFilter(
                    iso,
                    kernel=None,
                    config=pf_conf,
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
                )
        effective_variance_k, sanitized_covariance_k = (
            self._project_observation_covariance_to_variance(
                z_k,
                z_variance_k,
                z_covariance_k,
            )
        )
        self._adapt_strength_prior_at_detector(
            z_k=z_k,
            detector_pos=detector_pos,
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=live_time_s,
            z_variance_k=effective_variance_k,
        )
        for iso, val in z_k.items():
            if iso not in self.filters:
                continue
            self.filters[iso].update_continuous_pair_at_pose(
                z_obs=val,
                detector_pos=detector_pos,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
                observation_count_variance=(
                    0.0
                    if effective_variance_k is None
                    else float(effective_variance_k.get(iso, 0.0))
                ),
                step_idx=len(self.measurements),
            )
        self._invalidate_report_cache()
        self.measurements.append(
            MeasurementRecord(
                z_k={iso: float(v) for iso, v in z_k.items()},
                pose_idx=pose_idx,
                orient_idx=fe_index,
                live_time_s=live_time_s,
                fe_index=fe_index,
                pb_index=pb_index,
                z_variance_k=None
                if effective_variance_k is None
                else {iso: float(v) for iso, v in effective_variance_k.items()},
                z_covariance_k=sanitized_covariance_k,
                ig_value=None,
            )
        )
        self.refresh_sparse_poisson_evidence()
        self._apply_birth_death()
        self._invalidate_report_cache()
        self._record_history_estimate(len(self.measurements))

    def _measurement_data_for_iso(
        self,
        isotope: str,
        window: int | None,
        records: Sequence[MeasurementRecord] | None = None,
    ) -> MeasurementData | None:
        """Build measurement arrays for a single isotope with an optional window."""
        if records is None and not self.measurements:
            return None
        if records is not None:
            selected_records = list(records)
        elif window is None or window <= 0:
            selected_records = self.measurements
        else:
            selected_records = self.measurements[-int(window) :]
        if not selected_records:
            return None
        z_list = []
        poses = []
        fe_indices = []
        pb_indices = []
        live_times = []
        variance_list = []
        for rec in selected_records:
            z_list.append(float(rec.z_k.get(isotope, 0.0)))
            if rec.z_variance_k is None:
                variance_list.append(max(float(rec.z_k.get(isotope, 0.0)), 1.0))
            else:
                variance_list.append(
                    max(float(rec.z_variance_k.get(isotope, 1.0)), 1.0)
                )
            poses.append(self.poses[rec.pose_idx])
            live_times.append(float(rec.live_time_s))
            if rec.fe_index is not None and rec.pb_index is not None:
                fe_indices.append(int(rec.fe_index))
                pb_indices.append(int(rec.pb_index))
            else:
                fe_indices.append(int(rec.orient_idx))
                pb_indices.append(int(rec.orient_idx))
        return MeasurementData(
            z_k=np.asarray(z_list, dtype=float),
            observation_variances=np.asarray(variance_list, dtype=float),
            detector_positions=np.asarray(poses, dtype=float),
            fe_indices=np.asarray(fe_indices, dtype=int),
            pb_indices=np.asarray(pb_indices, dtype=int),
            live_times=np.asarray(live_times, dtype=float),
        )

    def _background_counts_for_report_refit(
        self,
        isotope: str,
        live_times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return background counts used by reported-strength refitting."""
        background_rate = 0.0
        filt = self.filters.get(isotope)
        if filt is not None and filt.continuous_particles:
            background_rate = float(filt.best_particle().state.background)
        elif filt is not None:
            level = filt.config.background_level
            if isinstance(level, dict):
                background_rate = float(level.get(isotope, 0.0))
            else:
                background_rate = float(level)
        return np.maximum(background_rate, 0.0) * np.asarray(live_times, dtype=float)

    def _source_prior_environment(self) -> EnvironmentConfig:
        """Return the room geometry used by surface-candidate diagnostics."""
        hi = np.asarray(self.pf_config.position_max, dtype=float).reshape(3)
        return EnvironmentConfig(
            size_x=float(hi[0]),
            size_y=float(hi[1]),
            size_z=float(hi[2]),
        )

    @staticmethod
    def _response_design_observability_stats(
        design: NDArray[np.float64],
        *,
        eps: float,
    ) -> dict[str, float | int]:
        """Return condition and correlation statistics for a response design."""
        design_arr = np.maximum(np.asarray(design, dtype=float), 0.0)
        if design_arr.ndim != 2 or design_arr.shape[0] == 0:
            return {
                "candidate_count": int(
                    design_arr.shape[1] if design_arr.ndim == 2 else 0
                ),
                "active_candidate_count": 0,
                "weak_column_count": 0,
                "condition_number": 1.0,
                "max_abs_correlation": 0.0,
                "ambiguous_pair_count_corr_ge_0p99": 0,
                "ambiguous_pair_count_corr_ge_0p995": 0,
            }
        column_norm = np.linalg.norm(design_arr, axis=0)
        valid = column_norm > max(float(eps), 1.0e-12)
        weak_count = int(np.count_nonzero(~valid))
        if np.count_nonzero(valid) <= 1:
            return {
                "candidate_count": int(design_arr.shape[1]),
                "active_candidate_count": int(np.count_nonzero(valid)),
                "weak_column_count": weak_count,
                "condition_number": 1.0,
                "max_abs_correlation": 0.0,
                "ambiguous_pair_count_corr_ge_0p99": 0,
                "ambiguous_pair_count_corr_ge_0p995": 0,
            }
        normalized = design_arr[:, valid] / np.maximum(column_norm[valid], eps)
        try:
            singular_values = np.linalg.svd(normalized, compute_uv=False)
            positive = singular_values[singular_values > max(float(eps), 1.0e-12)]
            condition = (
                float(np.max(positive) / max(float(np.min(positive)), eps))
                if positive.size
                else float("inf")
            )
        except np.linalg.LinAlgError:
            condition = float("inf")
        corr = np.abs(normalized.T @ normalized)
        upper = np.triu_indices(corr.shape[0], k=1)
        upper_values = corr[upper] if upper[0].size else np.zeros(0, dtype=float)
        max_corr = float(np.max(upper_values)) if upper_values.size else 0.0
        return {
            "candidate_count": int(design_arr.shape[1]),
            "active_candidate_count": int(np.count_nonzero(valid)),
            "weak_column_count": weak_count,
            "condition_number": condition,
            "max_abs_correlation": max_corr,
            "ambiguous_pair_count_corr_ge_0p99": int(
                np.count_nonzero(upper_values >= 0.99)
            ),
            "ambiguous_pair_count_corr_ge_0p995": int(
                np.count_nonzero(upper_values >= 0.995)
            ),
        }

    def surface_candidate_observability_diagnostics(
        self,
        *,
        window: int | None = None,
        max_candidates: int = 256,
    ) -> dict[str, dict[str, Any]]:
        """Return truth-independent observability diagnostics over surface candidates."""
        diagnostics: dict[str, dict[str, Any]] = {}
        if self.candidate_sources.size == 0:
            return diagnostics
        self._ensure_kernel_cache()
        pool_all = np.asarray(self.candidate_sources, dtype=float).reshape(-1, 3)
        sample_count = max(1, min(int(max_candidates), int(pool_all.shape[0])))
        if pool_all.shape[0] > sample_count:
            sample_indices = np.linspace(
                0,
                pool_all.shape[0] - 1,
                sample_count,
                dtype=np.int64,
            )
            pool = pool_all[sample_indices]
        else:
            pool = pool_all
        env = self._source_prior_environment()
        surface_kinds = source_surface_kinds(
            pool,
            env,
            self.obstacle_grid,
            obstacle_height_m=self.obstacle_height_m,
        )
        surface_counts = {
            str(kind): int(np.count_nonzero(surface_kinds == kind))
            for kind in ("floor", "ceiling", "wall", "obstacle_side", "obstacle_top")
        }
        surface_counts["off_surface"] = int(
            np.count_nonzero(np.equal(surface_kinds, None))
        )
        eps = max(float(self.pf_config.refit_eps), 1.0e-12)
        for isotope, filt in self.filters.items():
            data = self._measurement_data_for_iso(isotope, window)
            if data is None or data.z_k.size == 0:
                diagnostics[isotope] = {
                    "candidate_count": int(pool_all.shape[0]),
                    "sampled_candidate_count": int(pool.shape[0]),
                    "measurement_count": 0,
                    "surface_counts": surface_counts,
                }
                continue
            candidate_counts = self._cached_expected_counts_per_source(
                filt=filt,
                isotope=isotope,
                data=data,
                sources=pool,
                strengths=np.ones(pool.shape[0], dtype=float),
            )
            variances = measurement_vector(
                data.observation_variances,
                data.z_k.size,
                "observation_variances",
                min_value=1.0,
            )
            whitened = np.asarray(candidate_counts, dtype=float) / np.sqrt(
                variances[:, None]
            )
            stats = self._response_design_observability_stats(whitened, eps=eps)
            stats.update(
                {
                    "candidate_count": int(pool_all.shape[0]),
                    "sampled_candidate_count": int(pool.shape[0]),
                    "measurement_count": int(data.z_k.size),
                    "surface_counts": surface_counts,
                    "window": None if window is None else int(window),
                }
            )
            diagnostics[isotope] = stats
        return diagnostics

    def _report_absolute_strength_prior_terms(
        self,
        shape: tuple[int, ...],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return absolute report-strength prior precision and mean arrays."""
        mean = max(float(self.pf_config.source_strength_prior_mean), 0.0)
        weight = max(float(self.pf_config.source_strength_prior_weight), 0.0)
        if mean <= 0.0 or weight <= 0.0:
            zeros = np.zeros(shape, dtype=float)
            return zeros, zeros
        rel_sigma = max(float(self.pf_config.source_strength_prior_rel_sigma), 1.0e-6)
        sigma = max(rel_sigma * mean, 1.0e-12)
        precision = weight / (sigma * sigma)
        return (
            np.full(shape, precision, dtype=float),
            np.full(shape, mean, dtype=float),
        )

    def _apply_report_strength_absorption_guard(
        self,
        strengths: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Apply a one-sided high-strength guard to report refit strengths."""
        q_arr = np.maximum(np.asarray(strengths, dtype=float), 0.0)
        weight = max(
            0.0,
            float(self.pf_config.report_strength_absorption_penalty_weight),
        )
        mean = max(float(self.pf_config.source_strength_prior_mean), 0.0)
        if weight <= 0.0 or mean <= 0.0 or q_arr.size == 0:
            return q_arr
        multiple = max(
            1.0,
            float(self.pf_config.report_strength_absorption_q_multiple),
        )
        soft_cap = mean * multiple
        over = q_arr > soft_cap
        if not np.any(over):
            return q_arr
        guarded = q_arr.copy()
        guarded[over] = (guarded[over] + weight * soft_cap) / (1.0 + weight)
        return guarded

    def _report_observation_strength_bounds(
        self,
        *,
        design: NDArray[np.float64],
        z_obs: NDArray[np.float64],
        background: NDArray[np.float64],
        observation_variances: NDArray[np.float64],
        eps: float,
    ) -> NDArray[np.float64]:
        """Return data-driven per-source strength bounds from visible measurements."""
        design_arr = np.maximum(np.asarray(design, dtype=float), 0.0)
        z_arr = np.asarray(z_obs, dtype=float).reshape(-1)
        bg_arr = np.asarray(background, dtype=float).reshape(-1)
        var_arr = np.asarray(observation_variances, dtype=float).reshape(-1)
        if (
            design_arr.ndim != 2
            or z_arr.size != design_arr.shape[0]
            or bg_arr.size != z_arr.size
            or var_arr.size != z_arr.size
        ):
            return np.full(design_arr.shape[-1], np.inf, dtype=float)
        sigma = max(
            0.0,
            float(self.pf_config.report_strength_observation_overshoot_sigma),
        )
        allowed = np.maximum(z_arr - bg_arr, 0.0) + sigma * np.sqrt(
            np.maximum(var_arr, 1.0)
        )
        max_response = np.max(design_arr, axis=0)
        visible_fraction = max(
            0.0,
            float(
                self.pf_config.report_strength_observation_overshoot_min_visible_fraction
            ),
        )
        visible_threshold = np.maximum(float(eps), visible_fraction * max_response)
        visible = design_arr > visible_threshold[None, :]
        min_visible = max(
            1,
            int(
                self.pf_config.report_strength_observation_overshoot_min_visible_measurements
            ),
        )
        quantile = float(
            np.clip(
                float(self.pf_config.report_strength_observation_overshoot_quantile),
                0.0,
                1.0,
            )
        )
        ratios = np.divide(
            allowed[:, None],
            np.maximum(design_arr, float(eps)),
            out=np.full_like(design_arr, np.inf, dtype=float),
            where=visible,
        )
        visible_count = np.sum(visible, axis=0)
        ratios = np.where(visible, ratios, np.inf)
        sorted_ratios = np.sort(ratios, axis=0)
        count_for_index = np.maximum(visible_count, 1)
        positions = quantile * np.maximum(count_for_index - 1, 0)
        lower_indices = np.floor(positions).astype(int)
        upper_indices = np.ceil(positions).astype(int)
        lower = np.take_along_axis(
            sorted_ratios,
            lower_indices[None, :],
            axis=0,
        )[0]
        upper = np.take_along_axis(
            sorted_ratios,
            upper_indices[None, :],
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
        return np.where(
            (visible_count >= min_visible) & np.isfinite(bounds) & (bounds >= 0.0),
            bounds,
            np.inf,
        )

    def _apply_report_observation_overshoot_guard(
        self,
        strengths: NDArray[np.float64],
        *,
        design: NDArray[np.float64],
        z_obs: NDArray[np.float64],
        background: NDArray[np.float64],
        observation_variances: NDArray[np.float64],
        eps: float,
    ) -> NDArray[np.float64]:
        """Shrink report strengths only when their response over-explains data."""
        q_arr = np.asarray(strengths, dtype=float)
        weight = max(
            0.0,
            float(self.pf_config.report_strength_observation_overshoot_penalty_weight),
        )
        if weight <= 0.0 or q_arr.size == 0:
            return q_arr
        bounds = self._report_observation_strength_bounds(
            design=design,
            z_obs=z_obs,
            background=background,
            observation_variances=observation_variances,
            eps=eps,
        )
        if bounds.shape != q_arr.shape:
            return q_arr
        high = q_arr > bounds
        if not np.any(high):
            return q_arr
        guarded = q_arr.copy()
        guarded[high] = bounds[high] + (guarded[high] - bounds[high]) / (1.0 + weight)
        return np.where(np.isfinite(guarded), np.maximum(guarded, 0.0), q_arr)

    def _apply_report_observation_overshoot_guard_batch(
        self,
        strengths: NDArray[np.float64],
        *,
        design_batch: NDArray[np.float64],
        z_obs: NDArray[np.float64],
        background: NDArray[np.float64],
        observation_variances: NDArray[np.float64],
        eps: float,
    ) -> NDArray[np.float64]:
        """Shrink batched report strengths that visibly over-explain observations."""
        q_arr = np.asarray(strengths, dtype=float)
        weight = max(
            0.0,
            float(self.pf_config.report_strength_observation_overshoot_penalty_weight),
        )
        if weight <= 0.0 or q_arr.size == 0:
            return q_arr
        design_arr = np.maximum(np.asarray(design_batch, dtype=float), 0.0)
        if design_arr.ndim != 3 or design_arr.shape[0] != q_arr.shape[0]:
            return q_arr
        z_arr = np.asarray(z_obs, dtype=float).reshape(-1)
        bg_arr = np.asarray(background, dtype=float).reshape(-1)
        var_arr = np.asarray(observation_variances, dtype=float).reshape(-1)
        if (
            z_arr.size != design_arr.shape[1]
            or bg_arr.size != z_arr.size
            or var_arr.size != z_arr.size
            or design_arr.shape[2] != q_arr.shape[1]
        ):
            return q_arr
        sigma = max(
            0.0,
            float(self.pf_config.report_strength_observation_overshoot_sigma),
        )
        allowed = np.maximum(z_arr - bg_arr, 0.0) + sigma * np.sqrt(
            np.maximum(var_arr, 1.0)
        )
        max_response = np.max(design_arr, axis=1)
        visible_fraction = max(
            0.0,
            float(
                self.pf_config.report_strength_observation_overshoot_min_visible_fraction
            ),
        )
        visible_threshold = np.maximum(float(eps), visible_fraction * max_response)
        visible = design_arr > visible_threshold[:, None, :]
        ratios = np.divide(
            allowed[None, :, None],
            np.maximum(design_arr, float(eps)),
            out=np.full_like(design_arr, np.inf, dtype=float),
            where=visible,
        )
        quantile = float(
            np.clip(
                float(self.pf_config.report_strength_observation_overshoot_quantile),
                0.0,
                1.0,
            )
        )
        visible_count = np.sum(visible, axis=1)
        ratios = np.where(visible, ratios, np.inf)
        sorted_ratios = np.sort(ratios, axis=1)
        count_for_index = np.maximum(visible_count, 1)
        positions = quantile * np.maximum(count_for_index - 1, 0)
        lower_indices = np.floor(positions).astype(int)
        upper_indices = np.ceil(positions).astype(int)
        lower = np.take_along_axis(
            sorted_ratios,
            lower_indices[:, None, :],
            axis=1,
        )[:, 0, :]
        upper = np.take_along_axis(
            sorted_ratios,
            upper_indices[:, None, :],
            axis=1,
        )[:, 0, :]
        deltas = np.subtract(
            upper,
            lower,
            out=np.zeros_like(lower),
            where=np.isfinite(lower) & np.isfinite(upper),
        )
        bounds = lower + deltas * (positions - lower_indices)
        bounds = np.where(visible_count > 0, bounds, np.inf)
        min_visible = max(
            1,
            int(
                self.pf_config.report_strength_observation_overshoot_min_visible_measurements
            ),
        )
        bounds = np.where(
            (visible_count >= min_visible) & np.isfinite(bounds) & (bounds >= 0.0),
            bounds,
            np.inf,
        )
        high = q_arr > bounds
        if not np.any(high):
            return q_arr
        guarded = q_arr.copy()
        guarded[high] = bounds[high] + (guarded[high] - bounds[high]) / (1.0 + weight)
        return np.where(np.isfinite(guarded), np.maximum(guarded, 0.0), q_arr)

    def _solve_report_strengths(
        self,
        *,
        design: NDArray[np.float64],
        z_obs: NDArray[np.float64],
        background: NDArray[np.float64],
        observation_variances: NDArray[np.float64],
        initial_strengths: NDArray[np.float64],
        eps: float,
        q_max: float,
    ) -> NDArray[np.float64]:
        """Return non-negative strengths for fixed reported cluster positions."""
        design_arr = np.maximum(np.asarray(design, dtype=float), 0.0)
        z_arr = np.maximum(np.asarray(z_obs, dtype=float).reshape(-1), 0.0)
        if design_arr.ndim != 2 or design_arr.shape[0] != z_arr.size:
            return np.asarray(initial_strengths, dtype=float).reshape(-1)
        bg_arr = measurement_vector(
            background,
            z_arr.size,
            "background",
            min_value=0.0,
        )
        source_count = int(design_arr.shape[1])
        if source_count <= 0:
            return np.zeros(0, dtype=float)
        q = np.maximum(np.asarray(initial_strengths, dtype=float).reshape(-1), eps)
        if q.size == 0:
            q = np.full(source_count, eps, dtype=float)
        elif q.size != source_count:
            raise ValueError("initial_strengths must have one value per source.")
        if not np.any(np.isfinite(q)):
            q = np.full(source_count, eps, dtype=float)
        prior_q = np.maximum(np.where(np.isfinite(q), q, eps), eps)
        if q_max > 0.0:
            prior_q = np.minimum(prior_q, q_max)
        prior_weight = max(
            0.0,
            float(self.pf_config.report_strength_refit_prior_weight),
        )
        prior_rel_sigma = max(
            1.0e-6,
            float(self.pf_config.report_strength_refit_prior_rel_sigma),
        )
        abs_precision, abs_mean = self._report_absolute_strength_prior_terms(
            prior_q.shape
        )
        local_precision = np.zeros(source_count, dtype=float)
        if prior_weight > 0.0:
            prior_sigma = np.maximum(prior_rel_sigma * prior_q, eps)
            local_precision = prior_weight / np.maximum(prior_sigma**2, eps)
        prior_precision = local_precision + abs_precision
        prior_target = np.divide(
            local_precision * prior_q + abs_precision * abs_mean,
            np.maximum(prior_precision, eps),
            out=prior_q.copy(),
            where=prior_precision > 0.0,
        )
        column_sum = np.sum(design_arr, axis=0)
        observable = column_sum > eps
        signal_total = max(float(np.sum(z_arr - bg_arr)), 0.0)
        weak_or_invalid = ~np.isfinite(q) | (q <= eps)
        if np.any(weak_or_invalid) and signal_total > 0.0:
            denom = max(float(np.sum(column_sum[observable])), eps)
            q[weak_or_invalid & observable] = signal_total / denom
        q[~observable] = 0.0
        obs_variances = measurement_vector(
            observation_variances,
            z_arr.size,
            "observation_variances",
            min_value=1.0,
        )
        obs_weights = 1.0 / obs_variances
        gram = (design_arr.T * obs_weights[None, :]) @ design_arr
        rhs = (design_arr.T * obs_weights[None, :]) @ (z_arr - bg_arr)
        if np.any(prior_precision > 0.0):
            gram = gram + np.diag(prior_precision)
            rhs = rhs + prior_precision * prior_target
        try:
            direct = np.linalg.solve(
                gram + np.eye(source_count, dtype=float) * eps,
                rhs,
            )
            direct = np.where(np.isfinite(direct), direct, 0.0)
            if np.any(direct > 0.0):
                q = np.maximum(direct, 0.0)
                q = self._apply_report_strength_absorption_guard(q)
                q = self._apply_report_observation_overshoot_guard(
                    q,
                    design=design_arr,
                    z_obs=z_arr,
                    background=bg_arr,
                    observation_variances=obs_variances,
                    eps=eps,
                )
                q[~observable] = 0.0
        except np.linalg.LinAlgError:
            pass
        for _ in range(int(self.pf_config.report_strength_refit_iters)):
            lam = np.maximum(bg_arr + design_arr @ q, eps)
            ratio = np.divide(z_arr, lam, out=np.zeros_like(z_arr), where=lam > 0.0)
            numerator = design_arr.T @ ratio
            denominator = np.maximum(column_sum, eps)
            if np.any(prior_precision > 0.0):
                numerator = numerator + prior_precision * prior_target
                denominator = denominator + prior_precision * np.maximum(q, eps)
            q = q * np.clip(numerator / denominator, 0.0, np.inf)
            q = self._apply_report_strength_absorption_guard(q)
            q = self._apply_report_observation_overshoot_guard(
                q,
                design=design_arr,
                z_obs=z_arr,
                background=bg_arr,
                observation_variances=obs_variances,
                eps=eps,
            )
            q[~observable] = 0.0
            if q_max > 0.0:
                q = np.minimum(q, q_max)
            q = np.where(np.isfinite(q), q, 0.0)
        return np.maximum(q, 0.0)

    def _solve_report_strengths_batch(
        self,
        *,
        design_batch: NDArray[np.float64],
        z_obs: NDArray[np.float64],
        background: NDArray[np.float64],
        observation_variances: NDArray[np.float64],
        initial_strengths: NDArray[np.float64],
        eps: float,
        q_max: float,
    ) -> NDArray[np.float64]:
        """
        Return non-negative fixed-position strengths for a batch of subsets.

        This is a batched NumPy implementation of ``_solve_report_strengths``.
        It evaluates the same weighted least-squares initialization and the same
        multiplicative Poisson-regression iterations, but across many source
        subsets at once.
        """
        design_arr = np.maximum(np.asarray(design_batch, dtype=float), 0.0)
        z_arr = np.maximum(np.asarray(z_obs, dtype=float).reshape(-1), 0.0)
        if design_arr.ndim != 3 or design_arr.shape[1] != z_arr.size:
            return np.asarray(initial_strengths, dtype=float)
        bg_arr = measurement_vector(
            background,
            z_arr.size,
            "background",
            min_value=0.0,
        )
        batch_count, _, source_count = design_arr.shape
        if source_count <= 0:
            return np.zeros((batch_count, 0), dtype=float)
        q = np.maximum(np.asarray(initial_strengths, dtype=float), eps)
        if q.shape == (source_count,):
            q = np.broadcast_to(q[None, :], (batch_count, source_count)).copy()
        elif q.shape != (batch_count, source_count):
            raise ValueError("initial_strengths must have shape B x S.")
        q = np.where(np.isfinite(q), q, eps)
        prior_q = np.maximum(np.where(np.isfinite(q), q, eps), eps)
        if q_max > 0.0:
            prior_q = np.minimum(prior_q, q_max)
        prior_weight = max(
            0.0,
            float(self.pf_config.report_strength_refit_prior_weight),
        )
        prior_rel_sigma = max(
            1.0e-6,
            float(self.pf_config.report_strength_refit_prior_rel_sigma),
        )
        local_precision = np.zeros((batch_count, source_count), dtype=float)
        if prior_weight > 0.0:
            prior_sigma = np.maximum(prior_rel_sigma * prior_q, eps)
            local_precision = prior_weight / np.maximum(prior_sigma**2, eps)
        abs_precision, abs_mean = self._report_absolute_strength_prior_terms(
            prior_q.shape
        )
        prior_precision = local_precision + abs_precision
        prior_target = np.divide(
            local_precision * prior_q + abs_precision * abs_mean,
            np.maximum(prior_precision, eps),
            out=prior_q.copy(),
            where=prior_precision > 0.0,
        )
        column_sum = np.sum(design_arr, axis=1)
        observable = column_sum > eps
        signal_total = max(float(np.sum(z_arr - bg_arr)), 0.0)
        weak_or_invalid = ~np.isfinite(q) | (q <= eps)
        if signal_total > 0.0:
            denom = np.maximum(
                np.sum(np.where(observable, column_sum, 0.0), axis=1),
                eps,
            )
            q = np.where(
                weak_or_invalid & observable,
                signal_total / denom[:, None],
                q,
            )
        q = np.where(observable, q, 0.0)
        obs_variances = measurement_vector(
            observation_variances,
            z_arr.size,
            "observation_variances",
            min_value=1.0,
        )
        obs_weights = 1.0 / obs_variances
        weighted_design = design_arr * obs_weights[None, :, None]
        gram = np.einsum("bmi,bmj->bij", weighted_design, design_arr)
        rhs = np.einsum("bmk,m->bk", design_arr, obs_weights * (z_arr - bg_arr))
        if np.any(prior_precision > 0.0):
            eye = np.eye(source_count, dtype=float)[None, :, :]
            gram = gram + eye * prior_precision[:, None, :]
            rhs = rhs + prior_precision * prior_target
        try:
            eye = np.eye(source_count, dtype=float)[None, :, :]
            direct = np.linalg.solve(gram + eye * eps, rhs[:, :, None])[:, :, 0]
            direct = np.where(np.isfinite(direct), direct, 0.0)
            positive_rows = np.any(direct > 0.0, axis=1)
            q[positive_rows] = np.maximum(direct[positive_rows], 0.0)
            q = self._apply_report_strength_absorption_guard(q)
            q = self._apply_report_observation_overshoot_guard_batch(
                q,
                design_batch=design_arr,
                z_obs=z_arr,
                background=bg_arr,
                observation_variances=obs_variances,
                eps=eps,
            )
            q = np.where(observable, q, 0.0)
        except np.linalg.LinAlgError:
            pass
        for _ in range(int(self.pf_config.report_strength_refit_iters)):
            lam = np.maximum(
                bg_arr[None, :] + np.einsum("bmk,bk->bm", design_arr, q),
                eps,
            )
            ratio = np.divide(
                z_arr[None, :],
                lam,
                out=np.zeros_like(lam),
                where=lam > 0.0,
            )
            numerator = np.einsum("bmk,bm->bk", design_arr, ratio)
            denominator = np.maximum(column_sum, eps)
            if np.any(prior_precision > 0.0):
                numerator = numerator + prior_precision * prior_target
                denominator = denominator + prior_precision * np.maximum(q, eps)
            q = q * np.clip(numerator / denominator, 0.0, np.inf)
            q = self._apply_report_strength_absorption_guard(q)
            q = self._apply_report_observation_overshoot_guard_batch(
                q,
                design_batch=design_arr,
                z_obs=z_arr,
                background=bg_arr,
                observation_variances=obs_variances,
                eps=eps,
            )
            q = np.where(observable, q, 0.0)
            if q_max > 0.0:
                q = np.minimum(q, q_max)
            q = np.where(np.isfinite(q), q, 0.0)
        return np.maximum(q, 0.0)

    def _surface_rescue_strata(
        self,
        positions: NDArray[np.float64],
    ) -> NDArray[np.object_]:
        """Return coarse source-surface strata used for rescue diversity quotas."""
        pos_arr = np.asarray(positions, dtype=float).reshape(-1, 3)
        if pos_arr.size == 0:
            return np.zeros(0, dtype=object)
        env = self._source_prior_environment()
        kinds = source_surface_kinds(
            pos_arr,
            env,
            self.obstacle_grid,
            obstacle_height_m=self.obstacle_height_m,
            tolerance_m=1.0e-5,
        )
        strata = np.asarray(kinds, dtype=object).copy()
        room_z = max(float(env.size_z), 1.0e-9)
        high_wall = (np.asarray(kinds, dtype=object) == "wall") & (
            pos_arr[:, 2] >= 0.5 * room_z
        )
        strata[high_wall] = "high_wall"
        strata[np.equal(strata, None)] = "off_surface"
        return strata

    def _surface_spatial_rescue_keys(
        self,
        positions: NDArray[np.float64],
        strata: NDArray[np.object_],
    ) -> NDArray[np.object_]:
        """Return surface tile keys for spatial rescue diversity quotas."""
        pos_arr = np.asarray(positions, dtype=float).reshape(-1, 3)
        strata_arr = np.asarray(strata, dtype=object).reshape(-1)
        if pos_arr.shape[0] != strata_arr.size or pos_arr.size == 0:
            return np.full(pos_arr.shape[0], None, dtype=object)
        if not bool(self.pf_config.report_mle_rescue_spatial_quota_enable):
            return np.full(pos_arr.shape[0], None, dtype=object)
        tile_m = max(float(self.pf_config.report_mle_rescue_spatial_quota_tile_m), 1e-6)
        tile_xy = np.floor(pos_arr[:, :2] / tile_m).astype(np.int64)
        tile_z = np.floor(pos_arr[:, 2] / tile_m).astype(np.int64)
        keys = np.full(pos_arr.shape[0], None, dtype=object)
        for idx, stratum in enumerate(strata_arr):
            key = str(stratum)
            if key in {"off_surface", "None", ""}:
                continue
            # Wall and obstacle-side candidates can differ mainly in height.
            # Floor/ceiling/obstacle-top candidates are still separated in xy.
            if key in {"wall", "high_wall", "obstacle_side"}:
                keys[idx] = (
                    f"{key}:x{int(tile_xy[idx, 0])}:"
                    f"y{int(tile_xy[idx, 1])}:z{int(tile_z[idx])}"
                )
            else:
                keys[idx] = f"{key}:x{int(tile_xy[idx, 0])}:y{int(tile_xy[idx, 1])}"
        return keys

    def _surface_stratified_rescue_indices(
        self,
        pool: NDArray[np.float64],
        scores: NDArray[np.float64],
        valid: NDArray[np.bool_],
        *,
        max_candidates: int,
        strata: NDArray[np.object_] | None = None,
        spatial_keys: NDArray[np.object_] | None = None,
    ) -> NDArray[np.int64]:
        """Return top candidate indices while reserving slots for surface strata."""
        limit = max(0, int(max_candidates))
        if limit <= 0:
            return np.zeros(0, dtype=np.int64)
        score_arr = np.asarray(scores, dtype=float).reshape(-1)
        valid_arr = np.asarray(valid, dtype=bool).reshape(-1)
        if score_arr.size != valid_arr.size or score_arr.size == 0:
            return np.zeros(0, dtype=np.int64)
        valid_indices = np.flatnonzero(valid_arr & np.isfinite(score_arr))
        if valid_indices.size == 0:
            return np.zeros(0, dtype=np.int64)
        order = valid_indices[np.argsort(score_arr[valid_indices])[::-1]]
        if not bool(self.pf_config.report_mle_rescue_surface_quota_enable):
            return order[:limit].astype(np.int64, copy=False)
        selected: list[int] = []
        best_score = max(float(score_arr[order[0]]), 1.0e-300)
        min_fraction = max(
            0.0,
            float(self.pf_config.report_mle_rescue_surface_quota_min_score_fraction),
        )
        per_stratum = max(
            1,
            int(
                getattr(
                    self.pf_config, "report_mle_rescue_surface_quota_per_stratum", 1
                )
            ),
        )
        pool_arr = np.asarray(pool, dtype=float).reshape(-1, 3)
        strata_arr = (
            self._surface_rescue_strata(pool_arr)
            if strata is None
            else np.asarray(strata, dtype=object).reshape(-1)
        )
        if strata_arr.size != score_arr.size:
            return order[:limit].astype(np.int64, copy=False)

        def _append(index: int) -> bool:
            """Append an index if it is still eligible."""
            if len(selected) >= limit or int(index) in selected:
                return False
            if score_arr[int(index)] < best_score * min_fraction:
                return False
            selected.append(int(index))
            return True

        _append(int(order[0]))
        for stratum in (
            "ceiling",
            "high_wall",
            "obstacle_top",
            "obstacle_side",
            "wall",
            "floor",
            "off_surface",
        ):
            if len(selected) >= limit:
                break
            matches = order[np.asarray(strata_arr[order], dtype=object) == stratum]
            if matches.size:
                for match in matches[:per_stratum]:
                    _append(int(match))
                    if len(selected) >= limit:
                        break
        spatial_arr = (
            self._surface_spatial_rescue_keys(pool_arr, strata_arr)
            if spatial_keys is None
            else np.asarray(spatial_keys, dtype=object).reshape(-1)
        )
        if bool(self.pf_config.report_mle_rescue_spatial_quota_enable):
            per_tile = max(
                1,
                int(self.pf_config.report_mle_rescue_spatial_quota_per_tile),
            )
            key_counts: dict[str, int] = {}
            for selected_idx in selected:
                key = spatial_arr[int(selected_idx)]
                if key is None:
                    continue
                key_text = str(key)
                key_counts[key_text] = key_counts.get(key_text, 0) + 1
            for index in order:
                if len(selected) >= limit:
                    break
                key = spatial_arr[int(index)]
                if key is None:
                    continue
                key_text = str(key)
                if key_counts.get(key_text, 0) >= per_tile:
                    continue
                if _append(int(index)):
                    key_counts[key_text] = key_counts.get(key_text, 0) + 1
        for index in order:
            if len(selected) >= limit:
                break
            _append(int(index))
        return np.asarray(selected, dtype=np.int64)

    def _rescue_visibility_reference_strength(self) -> float:
        """Return the configured absolute rescue visibility reference, if any."""
        configured = max(
            0.0,
            float(self.pf_config.report_mle_rescue_visibility_reference_strength),
        )
        if configured > 0.0:
            return configured
        weak_reference = max(
            0.0,
            float(self.pf_config.weak_source_prune_visibility_reference_strength),
        )
        if weak_reference > 0.0:
            return weak_reference
        prior_mean = max(float(self.pf_config.source_strength_prior_mean), 0.0)
        if prior_mean > 0.0:
            return prior_mean
        return 0.0

    def _visibility_adjusted_rescue_scores(
        self,
        *,
        candidate_counts: NDArray[np.float64],
        residual: NDArray[np.float64],
        weights: NDArray[np.float64],
        base_scores: NDArray[np.float64],
        eps: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_], dict[str, int | float]]:
        """Blend all-history and visible-window scores for rescue candidates."""
        counts = np.maximum(np.asarray(candidate_counts, dtype=float), 0.0)
        score_arr = np.maximum(np.asarray(base_scores, dtype=float).reshape(-1), 0.0)
        if counts.ndim != 2 or counts.shape[1] != score_arr.size:
            return score_arr, np.ones(score_arr.size, dtype=bool), {}
        residual_arr = np.maximum(np.asarray(residual, dtype=float).reshape(-1), 0.0)
        weight_arr = np.maximum(np.asarray(weights, dtype=float).reshape(-1), 0.0)
        if residual_arr.size != counts.shape[0] or weight_arr.size != counts.shape[0]:
            return score_arr, np.ones(score_arr.size, dtype=bool), {}
        visible_count_floor = max(
            float(self.pf_config.report_mle_rescue_visible_count),
            float(self.pf_config.weak_source_prune_observable_count),
            float(self.pf_config.weak_source_prune_min_expected_count),
            0.0,
        )
        all_weights = weight_arr[:, None]
        all_numerator = np.sum(all_weights * residual_arr[:, None] * counts, axis=0)
        all_denominator = np.sum(all_weights * counts * counts, axis=0)
        all_q_hat = np.divide(
            all_numerator,
            np.maximum(all_denominator, eps),
            out=np.zeros_like(all_numerator, dtype=float),
            where=all_denominator > eps,
        )
        ref_strength = self._rescue_visibility_reference_strength()
        if ref_strength > 0.0:
            reference_counts = counts * ref_strength
            reference_mode = "configured"
        else:
            reference_counts = counts * np.maximum(all_q_hat[None, :], 0.0)
            reference_mode = "data_driven_qhat"
        visible = (
            reference_counts >= visible_count_floor
            if visible_count_floor > 0.0
            else reference_counts > 0.0
        )
        visible_counts = np.count_nonzero(visible, axis=0)
        min_visible = max(
            1,
            int(self.pf_config.report_mle_rescue_min_visible_measurements),
        )
        valid_visible = visible_counts >= min_visible
        blend = float(self.pf_config.report_mle_rescue_visibility_weight)
        if blend <= 0.0:
            return (
                score_arr,
                valid_visible,
                {
                    "rescue_visibility_min_visible": int(min_visible),
                    "rescue_visibility_valid_candidates": int(
                        np.count_nonzero(valid_visible)
                    ),
                    "rescue_visibility_reference_mode": reference_mode,
                    "rescue_visibility_reference_strength": float(ref_strength),
                    "rescue_visibility_qhat_median": float(
                        np.median(all_q_hat[all_q_hat > 0.0])
                        if np.any(all_q_hat > 0.0)
                        else 0.0
                    ),
                },
            )
        visible_weights = weight_arr[:, None] * visible
        numerator = np.sum(
            visible_weights * residual_arr[:, None] * counts,
            axis=0,
        )
        denominator = np.sum(visible_weights * counts * counts, axis=0)
        q_hat = np.divide(
            numerator,
            np.maximum(denominator, eps),
            out=np.zeros_like(numerator, dtype=float),
            where=denominator > eps,
        )
        visible_scores = np.maximum(numerator * np.maximum(q_hat, 0.0), 0.0)
        adjusted = (1.0 - blend) * score_arr + blend * visible_scores
        return (
            adjusted,
            valid_visible,
            {
                "rescue_visibility_weight": float(blend),
                "rescue_visibility_min_visible": int(min_visible),
                "rescue_visibility_valid_candidates": int(
                    np.count_nonzero(valid_visible)
                ),
                "rescue_visibility_reference_mode": reference_mode,
                "rescue_visibility_reference_strength": float(ref_strength),
                "rescue_visibility_qhat_median": float(
                    np.median(all_q_hat[all_q_hat > 0.0])
                    if np.any(all_q_hat > 0.0)
                    else 0.0
                ),
            },
        )

    def _next_surface_stratified_rescue_index(
        self,
        pool: NDArray[np.float64],
        scores: NDArray[np.float64],
        valid: NDArray[np.bool_],
        selected_indices: Sequence[int],
        strata: NDArray[np.object_] | None = None,
        spatial_keys: NDArray[np.object_] | None = None,
    ) -> int | None:
        """Return the next greedy rescue index with surface-stratum diversity."""
        score_arr = np.asarray(scores, dtype=float).reshape(-1)
        valid_arr = np.asarray(valid, dtype=bool).reshape(-1)
        valid_indices = np.flatnonzero(valid_arr & np.isfinite(score_arr))
        if valid_indices.size == 0:
            return None
        order = valid_indices[np.argsort(score_arr[valid_indices])[::-1]]
        if not bool(self.pf_config.report_mle_rescue_surface_quota_enable):
            return int(order[0])
        pool_arr = np.asarray(pool, dtype=float).reshape(-1, 3)
        strata_arr = (
            self._surface_rescue_strata(pool_arr)
            if strata is None
            else np.asarray(strata, dtype=object).reshape(-1)
        )
        if strata_arr.size != score_arr.size:
            return int(order[0])
        selected = np.asarray(list(selected_indices), dtype=int)
        selected_strata_counts: dict[str, int] = {}
        if selected.size:
            for value in strata_arr[selected]:
                key = str(value)
                selected_strata_counts[key] = selected_strata_counts.get(key, 0) + 1
        best_score = max(float(score_arr[order[0]]), 1.0e-300)
        min_fraction = max(
            0.0,
            float(self.pf_config.report_mle_rescue_surface_quota_min_score_fraction),
        )
        per_stratum = max(
            1,
            int(
                getattr(
                    self.pf_config, "report_mle_rescue_surface_quota_per_stratum", 1
                )
            ),
        )
        for stratum in (
            "ceiling",
            "high_wall",
            "obstacle_top",
            "obstacle_side",
            "wall",
            "floor",
            "off_surface",
        ):
            if selected_strata_counts.get(stratum, 0) >= per_stratum:
                continue
            matches = order[np.asarray(strata_arr[order], dtype=object) == stratum]
            if matches.size and score_arr[int(matches[0])] >= best_score * min_fraction:
                return int(matches[0])
        spatial_arr = (
            self._surface_spatial_rescue_keys(pool_arr, strata_arr)
            if spatial_keys is None
            else np.asarray(spatial_keys, dtype=object).reshape(-1)
        )
        if bool(self.pf_config.report_mle_rescue_spatial_quota_enable):
            per_tile = max(
                1,
                int(self.pf_config.report_mle_rescue_spatial_quota_per_tile),
            )
            selected_key_counts: dict[str, int] = {}
            if selected.size:
                for key in spatial_arr[selected]:
                    if key is None:
                        continue
                    key_text = str(key)
                    selected_key_counts[key_text] = (
                        selected_key_counts.get(key_text, 0) + 1
                    )
            for index in order:
                key = spatial_arr[int(index)]
                if key is None:
                    continue
                key_text = str(key)
                if selected_key_counts.get(key_text, 0) >= per_tile:
                    continue
                if score_arr[int(index)] >= best_score * min_fraction:
                    return int(index)
        return int(order[0])

    def _surface_stratum_count_payload(
        self,
        positions: NDArray[np.float64],
    ) -> dict[str, int]:
        """Return JSON-safe counts for rescue-selected surface strata."""
        strata = self._surface_rescue_strata(positions)
        payload: dict[str, int] = {}
        for value in strata:
            key = str(value)
            payload[key] = int(payload.get(key, 0) + 1)
        return payload

    def _rank_residual_surface_candidates(
        self,
        isotope: str,
        filt: IsotopeParticleFilter,
        data: MeasurementData,
        *,
        residual: NDArray[np.float64],
        existing_positions: NDArray[np.float64],
        background: NDArray[np.float64],
        eps: float,
        max_candidates: int | None = None,
        min_residual_fraction: float | None = None,
        dedup_radius_m: float | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
        """Return residual-ranked surface candidates for MLE-style report rescue."""
        max_candidates = max(
            0,
            int(
                self.pf_config.report_mle_rescue_max_residual_candidates
                if max_candidates is None
                else max_candidates
            ),
        )
        if max_candidates <= 0 or self.candidate_sources.size == 0:
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float), {}
        residual_arr = np.maximum(np.asarray(residual, dtype=float).reshape(-1), 0.0)
        if residual_arr.size != data.z_k.size:
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float), {}
        residual_sum = float(np.sum(residual_arr))
        reference_sum = max(
            float(np.sum(np.maximum(np.asarray(data.z_k, dtype=float), 0.0))),
            float(np.sum(np.maximum(np.asarray(background, dtype=float), 0.0))),
            eps,
        )
        min_fraction = max(
            0.0,
            float(
                self.pf_config.report_mle_rescue_min_residual_fraction
                if min_residual_fraction is None
                else min_residual_fraction
            ),
        )
        residual_fraction = residual_sum / reference_sum
        if residual_fraction < min_fraction:
            return (
                np.zeros((0, 3), dtype=float),
                np.zeros(0, dtype=float),
                {
                    "residual_sum": residual_sum,
                    "residual_reference_sum": reference_sum,
                    "residual_fraction": residual_fraction,
                    "residual_candidates_skipped": True,
                },
            )
        full_pool = np.asarray(self.candidate_sources, dtype=float).reshape(-1, 3)
        available = np.ones(full_pool.shape[0], dtype=bool)
        if existing_positions.size:
            existing = np.asarray(existing_positions, dtype=float).reshape(-1, 3)
            distances = np.linalg.norm(
                full_pool[:, None, :] - existing[None, :, :],
                axis=2,
            )
            dedup_radius = max(
                float(
                    self.pf_config.report_mle_rescue_dedup_radius_m
                    if dedup_radius_m is None
                    else dedup_radius_m
                ),
                0.0,
            )
            if dedup_radius > 0.0:
                available &= np.min(distances, axis=1) > dedup_radius
        if not np.any(available):
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float), {}
        pool = full_pool[available]
        candidate_counts_full = self._cached_candidate_grid_counts(
            filt=filt,
            isotope=isotope,
            data=data,
        )
        if (
            candidate_counts_full.ndim != 2
            or candidate_counts_full.shape[1] != full_pool.shape[0]
        ):
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float), {}
        candidate_counts = candidate_counts_full[:, available]
        candidate_counts = np.maximum(np.asarray(candidate_counts, dtype=float), 0.0)
        variances = measurement_vector(
            data.observation_variances,
            data.z_k.size,
            "observation_variances",
            min_value=1.0,
        )
        weights = 1.0 / variances
        weighted_residual = weights * residual_arr
        numerator = weighted_residual @ candidate_counts
        denominator = weights @ (candidate_counts * candidate_counts)
        q_hat = np.divide(
            numerator,
            np.maximum(denominator, eps),
            out=np.zeros_like(numerator, dtype=float),
            where=denominator > eps,
        )
        q_hat = np.maximum(np.where(np.isfinite(q_hat), q_hat, 0.0), 0.0)
        scores = numerator * q_hat
        scores, visibility_valid, visibility_stats = (
            self._visibility_adjusted_rescue_scores(
                candidate_counts=candidate_counts,
                residual=residual_arr,
                weights=weights,
                base_scores=scores,
                eps=eps,
            )
        )
        valid = np.isfinite(scores) & (scores > 0.0) & (q_hat > 0.0) & visibility_valid
        if not np.any(valid):
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float), {}
        selected = self._surface_stratified_rescue_indices(
            pool,
            scores,
            valid,
            max_candidates=max_candidates,
            strata=self._surface_rescue_strata(pool),
        )
        stats = {
            "residual_sum": residual_sum,
            "residual_reference_sum": reference_sum,
            "residual_fraction": residual_fraction,
            "residual_candidate_pool": int(pool.shape[0]),
            "residual_candidate_count": int(selected.size),
            "residual_candidate_best_score": (
                float(scores[selected[0]]) if selected.size else 0.0
            ),
            "residual_candidate_surface_counts": (
                self._surface_stratum_count_payload(pool[selected])
                if selected.size
                else {}
            ),
        }
        stats.update(visibility_stats)
        return pool[selected], q_hat[selected], stats

    def _rank_global_surface_candidates(
        self,
        isotope: str,
        filt: IsotopeParticleFilter,
        data: MeasurementData,
        *,
        existing_positions: NDArray[np.float64],
        background: NDArray[np.float64],
        eps: float,
        q_max: float,
        max_candidates: int | None = None,
        min_residual_fraction: float | None = None,
        dedup_radius_m: float | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
        """Return greedy surface candidates from all observations for report rescue."""
        max_candidates_value = max(
            0,
            int(
                self.pf_config.report_mle_rescue_max_residual_candidates
                if max_candidates is None
                else max_candidates
            ),
        )
        if max_candidates_value <= 0 or self.candidate_sources.size == 0:
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float), {}
        z_obs = np.maximum(np.asarray(data.z_k, dtype=float).reshape(-1), 0.0)
        background_arr = measurement_vector(
            background,
            z_obs.size,
            "background",
            min_value=0.0,
        )
        initial_residual = np.maximum(z_obs - background_arr, 0.0)
        reference_sum = max(float(np.sum(z_obs)), float(np.sum(background_arr)), eps)
        min_fraction = max(
            0.0,
            float(
                self.pf_config.report_mle_rescue_min_residual_fraction
                if min_residual_fraction is None
                else min_residual_fraction
            ),
        )
        initial_fraction = float(np.sum(initial_residual)) / reference_sum
        if initial_fraction < min_fraction:
            return (
                np.zeros((0, 3), dtype=float),
                np.zeros(0, dtype=float),
                {
                    "global_rescue_initial_residual_fraction": initial_fraction,
                    "global_rescue_candidates_skipped": True,
                },
            )
        pool = np.asarray(self.candidate_sources, dtype=float).reshape(-1, 3)
        dedup_radius = max(
            float(
                self.pf_config.report_mle_rescue_dedup_radius_m
                if dedup_radius_m is None
                else dedup_radius_m
            ),
            0.0,
        )
        unavailable = np.zeros(pool.shape[0], dtype=bool)
        if existing_positions.size and dedup_radius > 0.0:
            existing = np.asarray(existing_positions, dtype=float).reshape(-1, 3)
            distances = np.linalg.norm(pool[:, None, :] - existing[None, :, :], axis=2)
            unavailable |= np.min(distances, axis=1) <= dedup_radius
        candidate_counts = self._cached_candidate_grid_counts(
            filt=filt,
            isotope=isotope,
            data=data,
        )
        candidate_counts = np.maximum(np.asarray(candidate_counts, dtype=float), 0.0)
        if candidate_counts.ndim != 2 or candidate_counts.shape[1] != pool.shape[0]:
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float), {}
        variances = measurement_vector(
            data.observation_variances,
            z_obs.size,
            "observation_variances",
            min_value=1.0,
        )
        weights = 1.0 / variances
        denominator = weights @ (candidate_counts * candidate_counts)
        existing_count = 0
        existing_residual_fraction = initial_fraction
        if existing_positions.size:
            existing_arr = np.asarray(existing_positions, dtype=float).reshape(-1, 3)
            existing_count = int(existing_arr.shape[0])
            existing_design = self._cached_expected_counts_per_source(
                filt=filt,
                isotope=isotope,
                data=data,
                sources=existing_arr,
                strengths=np.ones(existing_arr.shape[0], dtype=float),
            )
            existing_design = np.maximum(
                np.asarray(existing_design, dtype=float),
                0.0,
            )
            if (
                existing_design.ndim == 2
                and existing_design.shape[0] == z_obs.size
                and existing_design.shape[1] == existing_arr.shape[0]
            ):
                existing_q = self._solve_report_strengths(
                    design=existing_design,
                    z_obs=z_obs,
                    background=background_arr,
                    observation_variances=variances,
                    initial_strengths=np.full(
                        existing_arr.shape[0],
                        max(float(self.pf_config.birth_q_min), 1.0),
                        dtype=float,
                    ),
                    eps=eps,
                    q_max=q_max,
                )
                initial_residual = np.maximum(
                    z_obs - (background_arr + existing_design @ existing_q),
                    0.0,
                )
                existing_residual_fraction = (
                    float(np.sum(initial_residual)) / reference_sum
                )
        strata = self._surface_rescue_strata(pool)
        spatial_keys = self._surface_spatial_rescue_keys(pool, strata)
        selected_indices: list[int] = []
        selected_q: list[float] = []
        selected_design = np.zeros((z_obs.size, 0), dtype=float)
        residual = initial_residual
        best_score = 0.0
        final_fraction = initial_fraction
        visibility_stats: dict[str, int | float] = {}
        for _ in range(max_candidates_value):
            residual_sum = float(np.sum(residual))
            final_fraction = residual_sum / reference_sum
            if final_fraction < min_fraction:
                break
            numerator = (weights * residual) @ candidate_counts
            q_hat = np.divide(
                numerator,
                np.maximum(denominator, eps),
                out=np.zeros_like(numerator, dtype=float),
                where=denominator > eps,
            )
            q_hat = np.maximum(np.where(np.isfinite(q_hat), q_hat, 0.0), 0.0)
            if q_max > 0.0:
                q_hat = np.minimum(q_hat, q_max)
            scores = numerator * q_hat
            scores, visibility_valid, visibility_stats = (
                self._visibility_adjusted_rescue_scores(
                    candidate_counts=candidate_counts,
                    residual=residual,
                    weights=weights,
                    base_scores=scores,
                    eps=eps,
                )
            )
            valid = (
                np.isfinite(scores)
                & (scores > 0.0)
                & (q_hat > eps)
                & (~unavailable)
                & visibility_valid
            )
            if not np.any(valid):
                break
            next_idx = self._next_surface_stratified_rescue_index(
                pool,
                scores,
                valid,
                selected_indices,
                strata=strata,
                spatial_keys=spatial_keys,
            )
            if next_idx is None:
                break
            best_idx = int(next_idx)
            best_score = max(best_score, float(scores[best_idx]))
            selected_indices.append(best_idx)
            selected_q.append(float(q_hat[best_idx]))
            selected_design = candidate_counts[:, selected_indices]
            selected_q_arr = self._solve_report_strengths(
                design=selected_design,
                z_obs=z_obs,
                background=background_arr,
                observation_variances=variances,
                initial_strengths=np.asarray(selected_q, dtype=float),
                eps=eps,
                q_max=q_max,
            )
            selected_q = [float(value) for value in selected_q_arr]
            residual = np.maximum(
                z_obs - (background_arr + selected_design @ selected_q_arr),
                0.0,
            )
            if dedup_radius > 0.0:
                distances = np.linalg.norm(
                    pool - pool[best_idx][None, :],
                    axis=1,
                )
                unavailable |= distances <= dedup_radius
            else:
                unavailable[best_idx] = True
        if not selected_indices:
            return (
                np.zeros((0, 3), dtype=float),
                np.zeros(0, dtype=float),
                {
                    "global_rescue_initial_residual_fraction": initial_fraction,
                    "global_rescue_candidate_pool": int(pool.shape[0]),
                    "global_rescue_candidate_count": 0,
                },
            )
        final_fraction = float(np.sum(residual)) / reference_sum
        selected = np.asarray(selected_indices, dtype=int)
        final_q = np.asarray(selected_q, dtype=float)
        stats = {
            "global_rescue_initial_residual_fraction": initial_fraction,
            "global_rescue_existing_source_count": int(existing_count),
            "global_rescue_existing_residual_fraction": float(
                existing_residual_fraction
            ),
            "global_rescue_final_residual_fraction": float(final_fraction),
            "global_rescue_candidate_pool": int(pool.shape[0]),
            "global_rescue_candidate_count": int(selected.size),
            "global_rescue_best_score": float(best_score),
            "global_rescue_surface_counts": self._surface_stratum_count_payload(
                pool[selected]
            ),
        }
        stats.update(visibility_stats)
        return pool[selected], final_q, stats

    def _augment_report_candidates_with_mle_rescue(
        self,
        isotope: str,
        filt: IsotopeParticleFilter,
        data: MeasurementData,
        *,
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
        background: NDArray[np.float64],
        eps: float,
        q_max: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
        """Add MLE-style global and residual surface candidates for report BIC."""
        base_pos = np.asarray(positions, dtype=float).reshape(-1, 3)
        base_q = np.asarray(strengths, dtype=float).reshape(-1)
        max_total = max(
            1,
            min(
                int(self.pf_config.report_mle_rescue_max_candidates),
                int(self.pf_config.report_cluster_model_selection_max_candidates),
            ),
        )
        dedup_radius = max(float(self.pf_config.report_mle_rescue_dedup_radius_m), 0.0)
        all_pos = [base_pos] if base_pos.size else []
        all_q = [np.maximum(base_q, eps)] if base_q.size else []
        global_pos, global_q, global_stats = self._rank_global_surface_candidates(
            isotope,
            filt,
            data,
            existing_positions=base_pos,
            background=background,
            eps=eps,
            q_max=q_max,
        )
        if global_pos.size:
            all_pos.append(global_pos)
            all_q.append(np.maximum(global_q, eps))
        posterior_count = 0
        max_posterior = max(
            0,
            int(self.pf_config.report_mle_rescue_max_posterior_candidates),
        )
        if max_posterior > 0 and hasattr(filt, "estimate_clustered"):
            try:
                post_pos, post_q = filt.estimate_clustered(
                    max_k=max_posterior,
                    include_report_excluded=True,
                )
            except RuntimeError:
                post_pos = np.zeros((0, 3), dtype=float)
                post_q = np.zeros(0, dtype=float)
            if post_pos.size and post_q.size:
                posterior_count = int(post_pos.shape[0])
                all_pos.append(np.asarray(post_pos, dtype=float).reshape(-1, 3))
                all_q.append(
                    np.maximum(np.asarray(post_q, dtype=float).reshape(-1), eps)
                )
        if all_pos:
            seed_pos = np.vstack(all_pos)
            seed_q = np.concatenate(all_q)
        else:
            seed_pos = np.zeros((0, 3), dtype=float)
            seed_q = np.zeros(0, dtype=float)
        seed_pos, seed_q = dedupe_report_candidates(
            seed_pos,
            seed_q,
            radius_m=dedup_radius,
            max_candidates=max_total,
        )
        residual_stats: dict[str, Any] = {}
        residual_pos = np.zeros((0, 3), dtype=float)
        residual_q = np.zeros(0, dtype=float)
        if seed_pos.size and seed_pos.shape[0] < max_total:
            seed_design = self._cached_expected_counts_per_source(
                filt=filt,
                isotope=isotope,
                data=data,
                sources=seed_pos,
                strengths=np.ones(seed_pos.shape[0], dtype=float),
            )
            seed_q_fit = self._solve_report_strengths(
                design=seed_design,
                z_obs=data.z_k,
                background=background,
                observation_variances=data.observation_variances,
                initial_strengths=seed_q,
                eps=eps,
                q_max=q_max,
            )
            residual = np.maximum(
                np.asarray(data.z_k, dtype=float).reshape(-1)
                - (
                    np.asarray(background, dtype=float).reshape(-1)
                    + np.asarray(seed_design, dtype=float) @ seed_q_fit
                ),
                0.0,
            )
            residual_pos, residual_q, residual_stats = (
                self._rank_residual_surface_candidates(
                    isotope,
                    filt,
                    data,
                    residual=residual,
                    existing_positions=seed_pos,
                    background=background,
                    eps=eps,
                )
            )
        elif seed_pos.shape[0] == 0:
            residual = np.maximum(
                np.asarray(data.z_k, dtype=float).reshape(-1)
                - np.asarray(background, dtype=float).reshape(-1),
                0.0,
            )
            residual_pos, residual_q, residual_stats = (
                self._rank_residual_surface_candidates(
                    isotope,
                    filt,
                    data,
                    residual=residual,
                    existing_positions=seed_pos,
                    background=background,
                    eps=eps,
                )
            )
        if residual_pos.size:
            merged_pos = (
                np.vstack([seed_pos, residual_pos]) if seed_pos.size else residual_pos
            )
            merged_q = (
                np.concatenate([seed_q, residual_q]) if seed_q.size else residual_q
            )
        else:
            merged_pos = seed_pos
            merged_q = seed_q
        final_pos, final_q = dedupe_report_candidates(
            merged_pos,
            merged_q,
            radius_m=dedup_radius,
            max_candidates=max_total,
        )
        stats = {
            "mle_rescue_enabled": True,
            "mle_rescue_base_candidates": int(base_pos.shape[0]),
            "mle_rescue_global_candidates": int(global_pos.shape[0]),
            "mle_rescue_posterior_candidates": int(posterior_count),
            "mle_rescue_residual_candidates": int(residual_pos.shape[0]),
            "mle_rescue_candidate_limit": int(max_total),
            "mle_rescue_candidate_count": int(final_pos.shape[0]),
        }
        stats.update(global_stats)
        stats.update(residual_stats)
        return final_pos, final_q, stats

    def _report_cluster_log_likelihood(
        self,
        filt: IsotopeParticleFilter,
        data: MeasurementData,
        *,
        design: NDArray[np.float64],
        strengths: NDArray[np.float64],
        background: NDArray[np.float64],
    ) -> float:
        """Return the report-level count log-likelihood for fixed clusters."""
        lam = np.asarray(background, dtype=float).reshape(-1) + np.asarray(
            design,
            dtype=float,
        ) @ np.asarray(strengths, dtype=float).reshape(-1)
        return float(
            filt._count_log_likelihood_np(
                data.z_k,
                lam,
                observation_count_variance=data.observation_variances,
            )
        )

    @staticmethod
    def _report_design_condition_number(
        design: NDArray[np.float64],
        *,
        eps: float,
    ) -> float:
        """Return a scale-normalized condition number for reported sources."""
        design_arr = np.maximum(np.asarray(design, dtype=float), 0.0)
        if design_arr.ndim != 2 or design_arr.shape[1] <= 1:
            return 1.0
        column_norm = np.linalg.norm(design_arr, axis=0)
        valid = column_norm > max(float(eps), 1.0e-12)
        if np.count_nonzero(valid) <= 1:
            return float("inf")
        normalized = design_arr[:, valid] / np.maximum(column_norm[valid], eps)
        try:
            singular_values = np.linalg.svd(normalized, compute_uv=False)
        except np.linalg.LinAlgError:
            return float("inf")
        positive = singular_values[singular_values > max(float(eps), 1.0e-12)]
        if positive.size <= 0:
            return float("inf")
        return float(np.max(positive) / max(float(np.min(positive)), eps))

    @staticmethod
    def _report_design_condition_numbers_batch(
        design_batch: NDArray[np.float64],
        *,
        eps: float,
    ) -> NDArray[np.float64]:
        """Return condition numbers for a stack of response designs."""
        designs = np.maximum(np.asarray(design_batch, dtype=float), 0.0)
        if designs.ndim != 3:
            return np.zeros(0, dtype=float)
        batch_count = int(designs.shape[0])
        source_count = int(designs.shape[2])
        if source_count <= 1:
            return np.ones(batch_count, dtype=float)
        floor = max(float(eps), 1.0e-12)
        column_norm = np.linalg.norm(designs, axis=1)
        valid = column_norm > floor
        valid_count = np.count_nonzero(valid, axis=1)
        normalized = np.divide(
            designs,
            np.maximum(column_norm[:, None, :], floor),
            out=np.zeros_like(designs, dtype=float),
            where=valid[:, None, :],
        )
        try:
            singular_values = np.linalg.svd(normalized, compute_uv=False)
        except np.linalg.LinAlgError:
            return np.full(batch_count, float("inf"), dtype=float)
        positive = singular_values > floor
        max_positive = np.max(
            np.where(positive, singular_values, 0.0),
            axis=1,
        )
        min_positive = np.min(
            np.where(positive, singular_values, float("inf")),
            axis=1,
        )
        condition = np.divide(
            max_positive,
            np.maximum(min_positive, float(eps)),
            out=np.full(batch_count, float("inf"), dtype=float),
            where=np.isfinite(min_positive) & (min_positive > 0.0),
        )
        condition[valid_count <= 1] = float("inf")
        return condition

    @staticmethod
    def _report_design_max_abs_correlation(
        design: NDArray[np.float64],
        *,
        eps: float,
    ) -> float:
        """Return the maximum normalized response-column correlation."""
        design_arr = np.maximum(np.asarray(design, dtype=float), 0.0)
        if design_arr.ndim != 2 or design_arr.shape[1] <= 1:
            return 0.0
        column_norm = np.linalg.norm(design_arr, axis=0)
        valid = column_norm > max(float(eps), 1.0e-12)
        if np.count_nonzero(valid) <= 1:
            return 0.0
        normalized = design_arr[:, valid] / np.maximum(column_norm[valid], eps)
        corr = np.abs(normalized.T @ normalized)
        upper = np.triu_indices(corr.shape[0], k=1)
        if upper[0].size == 0:
            return 0.0
        return float(np.max(corr[upper]))

    @staticmethod
    def _report_design_correlation_penalty(
        design: NDArray[np.float64],
        *,
        threshold: float,
        weight: float,
        power: float,
        eps: float,
    ) -> float:
        """Return a model-order penalty for unresolved collinear responses."""
        if float(weight) <= 0.0:
            return 0.0
        design_arr = np.maximum(np.asarray(design, dtype=float), 0.0)
        if design_arr.ndim != 2 or design_arr.shape[1] <= 1:
            return 0.0
        column_norm = np.linalg.norm(design_arr, axis=0)
        valid = column_norm > max(float(eps), 1.0e-12)
        if np.count_nonzero(valid) <= 1:
            return 0.0
        normalized = np.zeros_like(design_arr, dtype=float)
        normalized[:, valid] = design_arr[:, valid] / np.maximum(
            column_norm[valid],
            eps,
        )
        corr = np.abs(normalized.T @ normalized)
        upper = np.triu_indices(corr.shape[0], k=1)
        if upper[0].size == 0:
            return 0.0
        pair_valid = valid[upper[0]] & valid[upper[1]]
        excess = np.maximum(
            corr[upper] - float(np.clip(threshold, 0.0, 1.0)),
            0.0,
        )
        excess = excess[pair_valid]
        if excess.size == 0:
            return 0.0
        return float(
            max(float(weight), 0.0) * np.sum(excess ** max(float(power), 1.0e-6))
        )

    @staticmethod
    def _report_design_correlation_penalties_batch(
        design_batch: NDArray[np.float64],
        *,
        threshold: float,
        weight: float,
        power: float,
        eps: float,
    ) -> NDArray[np.float64]:
        """Return correlation penalties for a batch of report designs."""
        designs = np.maximum(np.asarray(design_batch, dtype=float), 0.0)
        if float(weight) <= 0.0 or designs.ndim != 3 or designs.shape[2] <= 1:
            return np.zeros(designs.shape[0] if designs.ndim == 3 else 0, dtype=float)
        norms = np.linalg.norm(designs, axis=1)
        valid = norms > max(float(eps), 1.0e-12)
        normalized = np.divide(
            designs,
            np.maximum(norms[:, None, :], max(float(eps), 1.0e-12)),
            out=np.zeros_like(designs, dtype=float),
            where=valid[:, None, :],
        )
        corr = np.abs(np.einsum("bmk,bml->bkl", normalized, normalized))
        upper = np.triu_indices(corr.shape[1], k=1)
        if upper[0].size == 0:
            return np.zeros(designs.shape[0], dtype=float)
        pair_valid = valid[:, upper[0]] & valid[:, upper[1]]
        excess = np.maximum(
            corr[:, upper[0], upper[1]] - float(np.clip(threshold, 0.0, 1.0)),
            0.0,
        )
        excess = np.where(pair_valid, excess, 0.0)
        return max(float(weight), 0.0) * np.sum(
            excess ** max(float(power), 1.0e-6),
            axis=1,
        )

    def _spectrum_history_arrays(
        self,
        required_isotopes: Sequence[str],
    ) -> dict[str, object] | None:
        """Return aligned spectrum-bin histories and templates for evidence."""
        required = tuple(str(isotope) for isotope in required_isotopes)
        selected_records: list[MeasurementRecord] = []
        bin_count: int | None = None
        for record in self.measurements:
            if record.spectrum_counts is None:
                continue
            templates = record.spectrum_response_templates_by_isotope or {}
            if any(isotope not in templates for isotope in required):
                continue
            count_bins = len(record.spectrum_counts)
            if bin_count is None:
                bin_count = count_bins
            if count_bins != bin_count:
                continue
            if record.spectrum_background is not None:
                if len(record.spectrum_background) != bin_count:
                    continue
            if not all(len(templates[isotope]) == bin_count for isotope in required):
                continue
            selected_records.append(record)
        if not selected_records or bin_count is None:
            return None
        counts = np.asarray(
            [record.spectrum_counts for record in selected_records],
            dtype=float,
        )
        backgrounds = np.asarray(
            [
                record.spectrum_background
                if record.spectrum_background is not None
                else tuple(0.0 for _ in range(bin_count))
                for record in selected_records
            ],
            dtype=float,
        )
        template_keys = set(
            selected_records[0].spectrum_response_templates_by_isotope or {}
        )
        for record in selected_records[1:]:
            template_keys &= set(record.spectrum_response_templates_by_isotope or {})
        templates_by_isotope: dict[str, NDArray[np.float64]] = {}
        for isotope in sorted(template_keys | set(required)):
            rows = []
            valid = True
            for record in selected_records:
                templates = record.spectrum_response_templates_by_isotope or {}
                if isotope not in templates or len(templates[isotope]) != bin_count:
                    valid = False
                    break
                rows.append(templates[isotope])
            if valid:
                templates_by_isotope[str(isotope)] = np.asarray(rows, dtype=float)
        if any(isotope not in templates_by_isotope for isotope in required):
            return None
        return {
            "records": selected_records,
            "spectrum_counts": np.maximum(
                np.where(np.isfinite(counts), counts, 0.0),
                0.0,
            ),
            "background_spectrum": np.maximum(
                np.where(np.isfinite(backgrounds), backgrounds, 0.0),
                0.0,
            ),
            "templates_by_isotope": templates_by_isotope,
        }

    def _spectral_nuisance_matrix(
        self,
        history: Mapping[str, object],
        *,
        target_isotope: str | None,
    ) -> NDArray[np.float64] | None:
        """Return low-dimensional spectrum nuisance columns for evidence."""
        if not bool(self.pf_config.sparse_poisson_spectral_nuisance_enable):
            return None
        counts = np.asarray(history.get("spectrum_counts"), dtype=float)
        if counts.ndim != 2 or counts.size == 0:
            return None
        records = list(history.get("records", []))
        live_times = np.asarray(
            [float(getattr(record, "live_time_s", 1.0)) for record in records],
            dtype=float,
        )
        if live_times.size != counts.shape[0]:
            live_times = np.ones(counts.shape[0], dtype=float)
        columns: list[NDArray[np.float64]] = []
        background = np.asarray(history.get("background_spectrum"), dtype=float)
        if background.shape == counts.shape and float(np.sum(background)) > 0.0:
            columns.append(background.reshape(-1))
        else:
            flat = np.broadcast_to(
                live_times[:, None] / max(int(counts.shape[1]), 1),
                counts.shape,
            )
            columns.append(flat.reshape(-1))
        templates = history.get("templates_by_isotope", {})
        if target_isotope is not None and isinstance(templates, Mapping):
            target = str(target_isotope)
            for isotope, template_values in sorted(templates.items()):
                if str(isotope) == target:
                    continue
                template = np.asarray(template_values, dtype=float)
                if template.shape != counts.shape:
                    continue
                column = live_times[:, None] * np.maximum(
                    np.where(np.isfinite(template), template, 0.0),
                    0.0,
                )
                if float(np.sum(column)) > 0.0:
                    columns.append(column.reshape(-1))
        active_columns = [
            np.maximum(np.where(np.isfinite(column), column, 0.0), 0.0)
            for column in columns
            if column.size == counts.size and float(np.sum(column)) > 0.0
        ]
        if not active_columns:
            return None
        return np.column_stack(active_columns)

    def _spectral_response_tensor_for_isotopes(
        self,
        history: Mapping[str, object],
        isotope_names: Sequence[str],
    ) -> tuple[NDArray[np.float64], dict[str, MeasurementData]]:
        """Return MxBxCxI spectrum response tensor and isotope data arrays."""
        counts = np.asarray(history.get("spectrum_counts"), dtype=float)
        if counts.ndim != 2:
            raise ValueError("spectrum_counts history must be measurements x bins.")
        records = list(history.get("records", []))
        templates = history.get("templates_by_isotope", {})
        if not isinstance(templates, Mapping):
            raise ValueError("spectrum template history is missing.")
        candidate_pool = np.asarray(self.candidate_sources, dtype=float).reshape(-1, 3)
        isotope_tuple = tuple(str(isotope) for isotope in isotope_names)
        tensor = np.zeros(
            (
                int(counts.shape[0]),
                int(counts.shape[1]),
                int(candidate_pool.shape[0]),
                len(isotope_tuple),
            ),
            dtype=float,
        )
        data_by_isotope: dict[str, MeasurementData] = {}
        for isotope_index, isotope in enumerate(isotope_tuple):
            filt = self.filters.get(isotope)
            if filt is None:
                continue
            data = self._measurement_data_for_iso(
                isotope,
                None,
                records=records,
            )
            if data is None:
                continue
            template = np.asarray(templates[str(isotope)], dtype=float)
            if template.shape != counts.shape:
                continue
            design = self._cached_candidate_grid_counts(
                filt=filt,
                isotope=isotope,
                data=data,
            )
            tensor[:, :, :, isotope_index] = (
                np.maximum(np.asarray(design, dtype=float), 0.0)[:, None, :]
                * np.maximum(np.where(np.isfinite(template), template, 0.0), 0.0)[
                    :, :, None
                ]
            )
            data_by_isotope[isotope] = data
        return tensor, data_by_isotope

    def _sparse_offgrid_bounds(
        self,
        positions: NDArray[np.float64],
    ) -> tuple[tuple[float, float], ...]:
        """Return local continuous bounds around selected sparse candidates."""
        pos = np.asarray(positions, dtype=float).reshape(-1, 3)
        radius = max(float(self.pf_config.sparse_poisson_offgrid_refine_radius_m), 0.0)
        lo_global = np.asarray(self.pf_config.position_min, dtype=float).reshape(3)
        hi_global = np.asarray(self.pf_config.position_max, dtype=float).reshape(3)
        lower = np.maximum(pos - radius, lo_global[None, :])
        upper = np.minimum(pos + radius, hi_global[None, :])
        bounds = []
        for low, high in zip(lower.reshape(-1), upper.reshape(-1)):
            low_f = float(low)
            high_f = float(high)
            if high_f < low_f:
                low_f, high_f = high_f, low_f
            bounds.append((low_f, high_f))
        return tuple(bounds)

    def _sparse_offgrid_refinement_payload(
        self,
        isotope: str,
        filt: IsotopeParticleFilter,
        payload: Mapping[str, Any],
        *,
        counts: NDArray[np.float64],
        background: NDArray[np.float64],
        nuisance_response_matrix: NDArray[np.float64] | None,
        response_at_positions: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    ) -> dict[str, Any] | None:
        """Return off-grid sparse evidence refinement diagnostics."""
        if not bool(self.pf_config.sparse_poisson_offgrid_refine_enable):
            return None
        selected_positions = np.asarray(
            payload.get("selected_positions", []),
            dtype=float,
        ).reshape(-1, 3)
        if selected_positions.size == 0:
            return None
        if max(float(self.pf_config.sparse_poisson_offgrid_refine_radius_m), 0.0) <= 0.0:
            return None
        refined = refine_sparse_poisson_evidence_offgrid(
            np.maximum(np.asarray(counts, dtype=float).reshape(-1), 0.0),
            selected_positions,
            response_at_positions,
            background=np.maximum(np.asarray(background, dtype=float).reshape(-1), 0.0),
            nuisance_response_matrix=nuisance_response_matrix,
            bounds=self._sparse_offgrid_bounds(selected_positions),
            config=self._sparse_poisson_evidence_config(
                nuisance_parameter_count=(
                    0
                    if nuisance_response_matrix is None
                    else int(np.asarray(nuisance_response_matrix).shape[1])
                )
            ),
            max_iter=int(self.pf_config.sparse_poisson_offgrid_refine_max_iter),
        )
        raw_positions = np.asarray(refined.positions, dtype=float).reshape(-1, 3)
        projected_positions = (
            filt._project_positions_to_source_prior(raw_positions)
            if raw_positions.size
            else np.zeros((0, 3), dtype=float)
        )
        accepted = bool(
            refined.available
            and refined.success
            and refined.improvement_log_likelihood
            >= float(self.pf_config.sparse_poisson_offgrid_refine_min_ll_gain)
        )
        return {
            "available": bool(refined.available),
            "accepted": bool(accepted),
            "reason": str(refined.reason),
            "success": bool(refined.success),
            "iterations": int(refined.iterations),
            "log_likelihood": float(refined.log_likelihood),
            "bic": float(refined.bic),
            "improvement_log_likelihood": float(refined.improvement_log_likelihood),
            "positions": [
                [float(value) for value in row]
                for row in np.asarray(projected_positions, dtype=float).reshape(-1, 3)
            ],
            "raw_positions": [
                [float(value) for value in row]
                for row in np.asarray(raw_positions, dtype=float).reshape(-1, 3)
            ],
            "strengths": [float(value) for value in refined.strengths],
            "nuisance_strengths": [
                float(value) for value in refined.nuisance_strengths
            ],
        }

    def _count_sparse_offgrid_refinement_payload(
        self,
        isotope: str,
        filt: IsotopeParticleFilter,
        data: MeasurementData,
        payload: Mapping[str, Any],
        *,
        background: NDArray[np.float64],
    ) -> dict[str, Any] | None:
        """Return count-level off-grid sparse evidence refinement diagnostics."""
        def _response_at_positions(positions: NDArray[np.float64]) -> NDArray[np.float64]:
            """Evaluate all selected source positions as one batched response matrix."""
            pos = filt._project_positions_to_source_prior(
                np.asarray(positions, dtype=float).reshape(-1, 3)
            )
            if pos.size == 0:
                return np.zeros((int(data.z_k.size), 0), dtype=float)
            return np.maximum(
                np.asarray(
                    self._cached_expected_counts_per_source(
                        filt=filt,
                        isotope=isotope,
                        data=data,
                        sources=pos,
                        strengths=np.ones(pos.shape[0], dtype=float),
                    ),
                    dtype=float,
                ),
                0.0,
            )

        return self._sparse_offgrid_refinement_payload(
            isotope,
            filt,
            payload,
            counts=np.asarray(data.z_k, dtype=float),
            background=np.asarray(background, dtype=float),
            nuisance_response_matrix=None,
            response_at_positions=_response_at_positions,
        )

    def _spectral_sparse_offgrid_refinement_payload(
        self,
        isotope: str,
        filt: IsotopeParticleFilter,
        payload: Mapping[str, Any],
        *,
        history: Mapping[str, object] | None = None,
        data: MeasurementData | None = None,
        nuisance: NDArray[np.float64] | None = None,
    ) -> dict[str, Any] | None:
        """Return spectrum-bin off-grid sparse evidence refinement diagnostics."""
        if history is None:
            history = self._spectrum_history_arrays([str(isotope)])
        if history is None:
            return None
        counts = np.asarray(history["spectrum_counts"], dtype=float)
        records = list(history.get("records", []))
        if data is None:
            data = self._measurement_data_for_iso(str(isotope), None, records=records)
        if data is None:
            return None
        templates = history.get("templates_by_isotope", {})
        if not isinstance(templates, Mapping) or str(isotope) not in templates:
            return None
        template = np.asarray(templates[str(isotope)], dtype=float)
        if template.shape != counts.shape:
            return None
        if nuisance is None:
            nuisance = self._spectral_nuisance_matrix(
                history,
                target_isotope=str(isotope),
            )

        def _response_at_positions(positions: NDArray[np.float64]) -> NDArray[np.float64]:
            """Evaluate selected positions as a flattened spectrum-bin response."""
            pos = filt._project_positions_to_source_prior(
                np.asarray(positions, dtype=float).reshape(-1, 3)
            )
            if pos.size == 0:
                return np.zeros((int(counts.size), 0), dtype=float)
            design = np.maximum(
                np.asarray(
                    self._cached_expected_counts_per_source(
                        filt=filt,
                        isotope=str(isotope),
                        data=data,
                        sources=pos,
                        strengths=np.ones(pos.shape[0], dtype=float),
                    ),
                    dtype=float,
                ),
                0.0,
            )
            tensor = (
                design[:, None, :]
                * np.maximum(np.where(np.isfinite(template), template, 0.0), 0.0)[
                    :, :, None
                ]
            )
            return tensor.reshape(counts.size, pos.shape[0])

        return self._sparse_offgrid_refinement_payload(
            isotope,
            filt,
            payload,
            counts=counts.reshape(-1),
            background=np.asarray(history["background_spectrum"], dtype=float).reshape(
                -1
            ),
            nuisance_response_matrix=nuisance,
            response_at_positions=_response_at_positions,
        )

    def _joint_spectral_sparse_offgrid_refinement_payload(
        self,
        isotope_names: Sequence[str],
        payload: Mapping[str, Any],
        history: Mapping[str, object],
        data_by_isotope: Mapping[str, MeasurementData],
    ) -> dict[str, Any] | None:
        """Return joint multi-isotope off-grid refinement diagnostics."""
        if not bool(self.pf_config.sparse_poisson_offgrid_refine_enable):
            return None
        if max(float(self.pf_config.sparse_poisson_offgrid_refine_radius_m), 0.0) <= 0.0:
            return None
        counts = np.asarray(history.get("spectrum_counts"), dtype=float)
        if counts.ndim != 2 or counts.size == 0:
            return None
        templates = history.get("templates_by_isotope", {})
        if not isinstance(templates, Mapping):
            return None
        selected_positions_by_iso = payload.get("selected_positions_by_isotope", {})
        if not isinstance(selected_positions_by_iso, Mapping):
            return None
        labels: list[str] = []
        initial_rows: list[list[float]] = []
        for isotope in isotope_names:
            positions = np.asarray(
                selected_positions_by_iso.get(str(isotope), []),
                dtype=float,
            ).reshape(-1, 3)
            for row in positions:
                labels.append(str(isotope))
                initial_rows.append([float(value) for value in row])
        if not initial_rows:
            return None
        initial_positions = np.asarray(initial_rows, dtype=float).reshape(-1, 3)
        label_arr = np.asarray(labels, dtype=object)
        nuisance = self._spectral_nuisance_matrix(history, target_isotope=None)

        def _response_at_positions(positions: NDArray[np.float64]) -> NDArray[np.float64]:
            """
            Evaluate joint source slots as one flattened spectrum response matrix.

            The loop is over the small configured isotope set, while candidate
            positions and spectrum bins stay batched inside each isotope group.
            """
            pos_all = np.asarray(positions, dtype=float).reshape(-1, 3)
            if pos_all.shape[0] != label_arr.size:
                raise ValueError("joint off-grid positions must match source slots.")
            response = np.zeros((int(counts.size), pos_all.shape[0]), dtype=float)
            for isotope in sorted(set(labels)):
                mask = label_arr == str(isotope)
                if not np.any(mask):
                    continue
                filt = self.filters.get(str(isotope))
                data = data_by_isotope.get(str(isotope))
                template = np.asarray(templates.get(str(isotope)), dtype=float)
                if filt is None or data is None or template.shape != counts.shape:
                    continue
                pos_group = filt._project_positions_to_source_prior(pos_all[mask])
                design = np.maximum(
                    np.asarray(
                        self._cached_expected_counts_per_source(
                            filt=filt,
                            isotope=str(isotope),
                            data=data,
                            sources=pos_group,
                            strengths=np.ones(pos_group.shape[0], dtype=float),
                        ),
                        dtype=float,
                    ),
                    0.0,
                )
                tensor = (
                    design[:, None, :]
                    * np.maximum(
                        np.where(np.isfinite(template), template, 0.0),
                        0.0,
                    )[:, :, None]
                )
                response[:, mask] = tensor.reshape(counts.size, pos_group.shape[0])
            return response

        refined = refine_sparse_poisson_evidence_offgrid(
            counts.reshape(-1),
            initial_positions,
            _response_at_positions,
            background=np.asarray(history["background_spectrum"], dtype=float).reshape(
                -1
            ),
            nuisance_response_matrix=nuisance,
            bounds=self._sparse_offgrid_bounds(initial_positions),
            config=self._sparse_poisson_evidence_config(
                nuisance_parameter_count=0 if nuisance is None else int(nuisance.shape[1])
            ),
            max_iter=int(self.pf_config.sparse_poisson_offgrid_refine_max_iter),
        )
        raw_positions = np.asarray(refined.positions, dtype=float).reshape(-1, 3)
        projected_positions = np.zeros_like(raw_positions, dtype=float)
        for isotope in sorted(set(labels)):
            mask = label_arr == str(isotope)
            filt = self.filters.get(str(isotope))
            if filt is None or not np.any(mask):
                projected_positions[mask] = raw_positions[mask]
                continue
            projected_positions[mask] = filt._project_positions_to_source_prior(
                raw_positions[mask]
            )
        accepted = bool(
            refined.available
            and refined.success
            and refined.improvement_log_likelihood
            >= float(self.pf_config.sparse_poisson_offgrid_refine_min_ll_gain)
        )
        positions_by_iso: dict[str, list[list[float]]] = {
            str(isotope): [] for isotope in isotope_names
        }
        strengths_by_iso: dict[str, list[float]] = {
            str(isotope): [] for isotope in isotope_names
        }
        for label, position, strength in zip(
            labels,
            projected_positions,
            np.asarray(refined.strengths, dtype=float).reshape(-1),
        ):
            positions_by_iso.setdefault(str(label), []).append(
                [float(value) for value in np.asarray(position, dtype=float).reshape(3)]
            )
            strengths_by_iso.setdefault(str(label), []).append(float(strength))
        return {
            "available": bool(refined.available),
            "accepted": bool(accepted),
            "reason": str(refined.reason),
            "success": bool(refined.success),
            "iterations": int(refined.iterations),
            "log_likelihood": float(refined.log_likelihood),
            "bic": float(refined.bic),
            "improvement_log_likelihood": float(refined.improvement_log_likelihood),
            "source_isotopes": [str(label) for label in labels],
            "positions": [
                [float(value) for value in row]
                for row in np.asarray(projected_positions, dtype=float).reshape(-1, 3)
            ],
            "raw_positions": [
                [float(value) for value in row]
                for row in np.asarray(raw_positions, dtype=float).reshape(-1, 3)
            ],
            "strengths": [float(value) for value in refined.strengths],
            "nuisance_strengths": [
                float(value) for value in refined.nuisance_strengths
            ],
            "positions_by_isotope": positions_by_iso,
            "strengths_by_isotope": strengths_by_iso,
        }

    @staticmethod
    def _apply_offgrid_refinement_to_payload(
        payload: dict[str, Any],
        refinement: Mapping[str, Any] | None,
    ) -> None:
        """Attach accepted off-grid refinement to a sparse evidence payload."""
        if refinement is None:
            return
        payload["offgrid_refinement"] = dict(refinement)
        if not bool(refinement.get("accepted", False)):
            return
        positions = refinement.get("positions", [])
        strengths = refinement.get("strengths", [])
        payload["selected_positions"] = [
            [float(value) for value in row]
            for row in np.asarray(positions, dtype=float).reshape(-1, 3)
        ]
        payload["selected_strengths"] = [
            float(value) for value in np.asarray(strengths, dtype=float).reshape(-1)
        ]
        payload["offgrid_refined"] = True

    def _spectral_sparse_poisson_evidence_payload(
        self,
        isotope: str,
    ) -> dict[str, Any] | None:
        """Return direct spectrum-bin sparse evidence diagnostics for one isotope."""
        spectral_start = time.perf_counter()
        stage_wall: Dict[str, float] = {}
        if not bool(self.pf_config.sparse_poisson_spectral_evidence_enable):
            return None
        if self.candidate_sources.size == 0 or str(isotope) not in self.filters:
            return None
        filt = self.filters[str(isotope)]
        stage_start = time.perf_counter()
        history = self._spectrum_history_arrays([str(isotope)])
        stage_wall["history"] = time.perf_counter() - stage_start
        if history is None:
            return None
        counts = np.asarray(history["spectrum_counts"], dtype=float)
        stage_start = time.perf_counter()
        tensor, _data_by_isotope = self._spectral_response_tensor_for_isotopes(
            history,
            [str(isotope)],
        )
        stage_wall["response_tensor"] = time.perf_counter() - stage_start
        if tensor.shape[2] == 0:
            return None
        stage_start = time.perf_counter()
        nuisance = self._spectral_nuisance_matrix(
            history,
            target_isotope=str(isotope),
        )
        stage_wall["nuisance"] = time.perf_counter() - stage_start
        nuisance_count = 0 if nuisance is None else int(nuisance.shape[1])
        stage_start = time.perf_counter()
        evidence = fit_sparse_poisson_spectral_evidence(
            counts,
            tensor,
            background_spectrum=np.asarray(history["background_spectrum"], dtype=float),
            isotope_names=[str(isotope)],
            nuisance_response_tensor=nuisance,
            config=self._sparse_poisson_evidence_config(
                nuisance_parameter_count=nuisance_count
            ),
        )
        stage_wall["spectral_fit"] = time.perf_counter() - stage_start
        payload = sparse_poisson_evidence_to_diagnostics(evidence)
        candidate_pool = np.asarray(self.candidate_sources, dtype=float).reshape(-1, 3)
        selected_indices = np.asarray(payload.get("selected_indices", []), dtype=int)
        selected_positions = (
            candidate_pool[selected_indices]
            if selected_indices.size
            else np.zeros((0, 3), dtype=float)
        )
        payload["selected_positions"] = [
            [float(value) for value in row]
            for row in np.asarray(selected_positions, dtype=float).reshape(-1, 3)
        ]
        payload["spectrum_measurement_count"] = int(counts.shape[0])
        payload["spectrum_bin_count"] = int(counts.shape[1])
        payload["nuisance_parameter_count"] = int(nuisance_count)
        if bool(self.pf_config.sparse_poisson_ambiguity_report_enable):
            response_flat = tensor.reshape(counts.shape[0] * counts.shape[1], -1)
            payload["ambiguity_clusters"] = [
                dict(cluster)
                for cluster in sparse_poisson_ambiguity_diagnostics(
                    evidence,
                    np.maximum(np.asarray(response_flat, dtype=float), 0.0),
                    candidate_positions=candidate_pool,
                    correlation_threshold=float(
                        self.pf_config.sparse_poisson_ambiguity_corr_threshold
                    ),
                    bic_gap_threshold=float(
                        self.pf_config.sparse_poisson_ambiguity_bic_gap_threshold
                    ),
                    condition_number_threshold=float(
                        self.pf_config.sparse_poisson_ambiguity_condition_max
                    ),
                )
            ]
        stage_start = time.perf_counter()
        refinement = self._spectral_sparse_offgrid_refinement_payload(
            str(isotope),
            filt,
            payload,
            history=history,
            data=_data_by_isotope.get(str(isotope)),
            nuisance=nuisance,
        )
        self._apply_offgrid_refinement_to_payload(payload, refinement)
        stage_wall["offgrid_refine"] = time.perf_counter() - stage_start
        stage_wall["total"] = time.perf_counter() - spectral_start
        payload["spectral_evidence_stage_wall_s"] = {
            key: float(value) for key, value in stage_wall.items()
        }
        return payload

    def _refresh_joint_sparse_poisson_evidence(
        self,
        isotope_names: Sequence[str],
    ) -> dict[str, Any] | None:
        """Refresh joint multi-isotope spectrum-bin sparse evidence diagnostics."""
        joint_start = time.perf_counter()
        stage_wall: Dict[str, float] = {}
        self._last_joint_sparse_poisson_evidence_diagnostics = {}
        if not (
            bool(self.pf_config.sparse_poisson_joint_evidence_enable)
            and bool(self.pf_config.sparse_poisson_spectral_evidence_enable)
        ):
            return None
        isotopes = tuple(
            str(isotope) for isotope in isotope_names if isotope in self.filters
        )
        if len(isotopes) < 2 or self.candidate_sources.size == 0:
            return None
        stage_start = time.perf_counter()
        history = self._spectrum_history_arrays(isotopes)
        stage_wall["history"] = time.perf_counter() - stage_start
        if history is None:
            return None
        counts = np.asarray(history["spectrum_counts"], dtype=float)
        stage_start = time.perf_counter()
        tensor, data_by_isotope = self._spectral_response_tensor_for_isotopes(
            history,
            isotopes,
        )
        stage_wall["response_tensor"] = time.perf_counter() - stage_start
        if any(isotope not in data_by_isotope for isotope in isotopes):
            return None
        response_by_isotope = {
            isotope: tensor[:, :, :, isotope_index].reshape(
                counts.shape[0] * counts.shape[1],
                -1,
            )
            for isotope_index, isotope in enumerate(isotopes)
        }
        stage_start = time.perf_counter()
        nuisance = self._spectral_nuisance_matrix(
            history,
            target_isotope=None,
        )
        stage_wall["nuisance"] = time.perf_counter() - stage_start
        nuisance_count = 0 if nuisance is None else int(nuisance.shape[1])
        stage_start = time.perf_counter()
        evidence = fit_joint_sparse_poisson_evidence(
            counts.reshape(-1),
            response_by_isotope,
            max_sources_by_isotope={
                isotope: self._report_max_sources_per_isotope()
                for isotope in isotopes
            },
            background=np.asarray(history["background_spectrum"], dtype=float).reshape(
                -1
            ),
            nuisance_response_matrix=nuisance,
            config=self._sparse_poisson_evidence_config(
                nuisance_parameter_count=nuisance_count
            ),
        )
        stage_wall["joint_fit"] = time.perf_counter() - stage_start
        payload = joint_sparse_poisson_evidence_to_diagnostics(evidence)
        payload["spectrum_measurement_count"] = int(counts.shape[0])
        payload["spectrum_bin_count"] = int(counts.shape[1])
        payload["nuisance_parameter_count"] = int(nuisance_count)
        candidate_pool = np.asarray(self.candidate_sources, dtype=float).reshape(-1, 3)
        selected_positions_by_isotope: dict[str, list[list[float]]] = {}
        for isotope in isotopes:
            selected = np.asarray(
                payload.get("selected_indices_by_isotope", {}).get(isotope, []),
                dtype=int,
            )
            positions = (
                candidate_pool[selected]
                if selected.size
                else np.zeros((0, 3), dtype=float)
            )
            selected_positions_by_isotope[isotope] = [
                [float(value) for value in row]
                for row in np.asarray(positions, dtype=float).reshape(-1, 3)
            ]
        payload["selected_positions_by_isotope"] = selected_positions_by_isotope
        stage_start = time.perf_counter()
        joint_refinement = self._joint_spectral_sparse_offgrid_refinement_payload(
            isotopes,
            payload,
            history,
            data_by_isotope,
        )
        stage_wall["offgrid_refine"] = time.perf_counter() - stage_start
        if joint_refinement is not None:
            payload["offgrid_refinement"] = copy.deepcopy(joint_refinement)
            if bool(joint_refinement.get("accepted", False)):
                refined_positions = joint_refinement.get("positions_by_isotope", {})
                refined_strengths = joint_refinement.get("strengths_by_isotope", {})
                if isinstance(refined_positions, Mapping):
                    selected_positions_by_isotope = {
                        str(isotope): [
                            [float(value) for value in row]
                            for row in np.asarray(
                                refined_positions.get(str(isotope), []),
                                dtype=float,
                            ).reshape(-1, 3)
                        ]
                        for isotope in isotopes
                    }
                    payload["selected_positions_by_isotope"] = (
                        selected_positions_by_isotope
                    )
                if isinstance(refined_strengths, Mapping):
                    payload["selected_strengths_by_isotope"] = {
                        str(isotope): [
                            float(value)
                            for value in np.asarray(
                                refined_strengths.get(str(isotope), []),
                                dtype=float,
                            ).reshape(-1)
                        ]
                        for isotope in isotopes
                    }
                payload["offgrid_refined"] = True
        distinct_station_count = 0
        if data_by_isotope:
            distinct_station_count = max(
                self._measurement_distinct_station_count(data)
                for data in data_by_isotope.values()
            )
        min_distinct_stations = max(
            1,
            int(self.pf_config.sparse_poisson_evidence_min_distinct_stations),
        )
        payload["distinct_station_count"] = int(distinct_station_count)
        payload["min_distinct_stations_for_ready"] = int(min_distinct_stations)
        min_margin = max(
            float(self.pf_config.sparse_poisson_evidence_min_bic_margin),
            0.0,
        )
        joint_margin = float(payload.get("bic_margin_to_runner_up", float("-inf")))
        payload["model_order_ready"] = bool(
            bool(payload.get("available", False))
            and np.isfinite(joint_margin)
            and joint_margin >= min_margin
            and int(distinct_station_count) >= int(min_distinct_stations)
        )
        stage_wall["total"] = time.perf_counter() - joint_start
        payload["joint_evidence_stage_wall_s"] = {
            key: float(value) for key, value in stage_wall.items()
        }
        self._last_joint_sparse_poisson_evidence_diagnostics = copy.deepcopy(payload)
        for isotope in isotopes:
            current = self._last_sparse_poisson_evidence_diagnostics.get(isotope)
            if not isinstance(current, dict):
                continue
            current["joint_multi_isotope_evidence"] = copy.deepcopy(payload)
            if (
                bool(payload.get("available", False))
                and bool(self.pf_config.sparse_poisson_evidence_authoritative)
            ):
                original = copy.deepcopy(current)
                current["single_isotope_sparse_poisson_evidence"] = original
                current["method"] = "joint_multi_isotope_sparse_poisson_projection"
                current["selected_count"] = int(
                    payload.get("selected_counts_by_isotope", {}).get(isotope, 0)
                )
                current["selected_indices"] = [
                    int(idx)
                    for idx in payload.get("selected_indices_by_isotope", {}).get(
                        isotope,
                        [],
                    )
                ]
                current["selected_strengths"] = [
                    float(value)
                    for value in payload.get("selected_strengths_by_isotope", {}).get(
                        isotope,
                        [],
                    )
                ]
                current["selected_positions"] = selected_positions_by_isotope.get(
                    isotope,
                    [],
                )
                current["model_order_ready"] = bool(payload["model_order_ready"])
                pruned = self._prune_candidate_verification_queue_with_sparse_evidence(
                    isotope,
                    current,
                )
                if pruned > 0:
                    current["candidate_verification_queue_pruned"] = int(pruned)
                self._last_report_model_order_diagnostics[isotope] = copy.deepcopy(
                    current
                )
        return copy.deepcopy(payload)

    def refresh_sparse_poisson_evidence(self) -> Dict[str, Dict[str, Any]]:
        """Refresh all-history sparse Poisson evidence for every isotope."""
        refresh_start = time.perf_counter()
        stage_wall: Dict[str, float] = {}
        if not bool(self.pf_config.sparse_poisson_evidence_enable):
            self._last_sparse_poisson_evidence_diagnostics.clear()
            self._last_joint_sparse_poisson_evidence_diagnostics = {}
            self.last_sparse_poisson_refresh_wall_s = time.perf_counter() - refresh_start
            self.last_sparse_poisson_refresh_stage_wall_s = {
                "disabled": float(self.last_sparse_poisson_refresh_wall_s),
            }
            return {}
        if not self.filters:
            self._last_sparse_poisson_evidence_diagnostics.clear()
            self._last_joint_sparse_poisson_evidence_diagnostics = {}
            self.last_sparse_poisson_refresh_wall_s = time.perf_counter() - refresh_start
            self.last_sparse_poisson_refresh_stage_wall_s = {
                "no_filters": float(self.last_sparse_poisson_refresh_wall_s),
            }
            return {}

        refreshed: Dict[str, Dict[str, Any]] = {}
        isotope_order = list(self.filters.keys())
        for isotope, filt in self.filters.items():
            isotope_start = time.perf_counter()
            data = self._measurement_data_for_iso(isotope, None)
            if data is None:
                stage_wall[f"single_isotope:{isotope}"] = (
                    time.perf_counter() - isotope_start
                )
                continue
            background = self._background_counts_for_report_refit(
                isotope,
                data.live_times,
            )
            payload = self._update_sparse_poisson_evidence_for_isotope(
                isotope,
                filt,
                data,
                background=background,
            )
            if payload is not None:
                isotope_wall = float(time.perf_counter() - isotope_start)
                payload["sparse_refresh_wall_s"] = isotope_wall
                current = self._last_sparse_poisson_evidence_diagnostics.get(
                    str(isotope)
                )
                if isinstance(current, dict):
                    current["sparse_refresh_wall_s"] = isotope_wall
                refreshed[str(isotope)] = copy.deepcopy(payload)
            stage_wall[f"single_isotope:{isotope}"] = (
                time.perf_counter() - isotope_start
            )
        joint_start = time.perf_counter()
        joint_payload = self._refresh_joint_sparse_poisson_evidence(isotope_order)
        stage_wall["joint_multi_isotope"] = time.perf_counter() - joint_start
        if joint_payload is not None:
            joint_payload["sparse_refresh_wall_s"] = float(
                stage_wall["joint_multi_isotope"]
            )
            if isinstance(self._last_joint_sparse_poisson_evidence_diagnostics, dict):
                self._last_joint_sparse_poisson_evidence_diagnostics[
                    "sparse_refresh_wall_s"
                ] = float(stage_wall["joint_multi_isotope"])
            refreshed["joint_multi_isotope"] = copy.deepcopy(joint_payload)
            for isotope in isotope_order:
                current = self._last_sparse_poisson_evidence_diagnostics.get(
                    str(isotope)
                )
                if isinstance(current, dict):
                    refreshed[str(isotope)] = copy.deepcopy(current)
        self.last_sparse_poisson_refresh_wall_s = time.perf_counter() - refresh_start
        stage_wall["total"] = float(self.last_sparse_poisson_refresh_wall_s)
        self.last_sparse_poisson_refresh_stage_wall_s = {
            key: float(value) for key, value in stage_wall.items()
        }
        return refreshed

    def _sparse_poisson_evidence_config(
        self,
        *,
        nuisance_parameter_count: int = 0,
    ) -> SparsePoissonEvidenceConfig:
        """Return the all-history sparse Poisson evidence configuration."""
        return SparsePoissonEvidenceConfig(
            max_sources=self._report_max_sources_per_isotope(),
            candidate_limit=int(self.pf_config.sparse_poisson_evidence_candidate_limit),
            parameter_count_per_source=int(
                self.pf_config.sparse_poisson_evidence_parameter_count_per_source
            ),
            refit_iters=int(self.pf_config.sparse_poisson_evidence_refit_iters),
            holdout_stride=int(self.pf_config.sparse_poisson_evidence_holdout_stride),
            correlation_prune_threshold=float(
                self.pf_config.sparse_poisson_evidence_corr_prune_threshold
            ),
            eps=max(float(self.pf_config.report_strength_refit_eps), 1.0e-12),
            q_max=float(self.pf_config.birth_q_max),
            nuisance_parameter_count=max(0, int(nuisance_parameter_count)),
        )

    @staticmethod
    def _measurement_distinct_station_count(data: MeasurementData) -> int:
        """Return the number of distinct robot stations in measurement data."""
        measurement_count = int(np.asarray(data.z_k, dtype=float).reshape(-1).size)
        labels = IsotopeParticleFilter._support_station_labels(
            np.asarray(data.detector_positions, dtype=float),
            measurement_count,
        )
        if labels.size == 0:
            return 0
        return int(np.unique(labels).size)

    def _sparse_poisson_count_support(
        self,
        data: MeasurementData,
        background: NDArray[np.float64],
    ) -> dict[str, float | bool]:
        """Return observed-signal support diagnostics for sparse evidence."""
        z_arr = np.maximum(np.asarray(data.z_k, dtype=float).reshape(-1), 0.0)
        bg_arr = measurement_vector(
            background,
            z_arr.size,
            "background",
            min_value=0.0,
        )
        variances = measurement_vector(
            data.observation_variances,
            z_arr.size,
            "observation_variances",
            min_value=1.0,
        )
        signal = np.maximum(z_arr - bg_arr, 0.0)
        signal_total = float(np.sum(signal))
        signal_max = float(np.max(signal)) if signal.size else 0.0
        distinct_station_count = self._measurement_distinct_station_count(data)
        min_distinct_stations = max(
            1,
            int(self.pf_config.sparse_poisson_evidence_min_distinct_stations),
        )
        signal_snr = float(
            signal_total / np.sqrt(max(float(np.sum(variances)), 1.0e-12))
        )
        count_supported = bool(
            signal_total >= max(float(self.pf_config.structural_update_min_counts), 0.0)
            or signal_max
            >= max(float(self.pf_config.conditional_strength_refit_min_count), 0.0)
            or signal_snr >= max(float(self.pf_config.structural_update_min_snr), 0.0)
        )
        return {
            "observed_signal_total_counts": signal_total,
            "observed_signal_max_count": signal_max,
            "observed_signal_snr": signal_snr,
            "distinct_station_count": int(distinct_station_count),
            "min_distinct_stations_for_ready": int(min_distinct_stations),
            "count_supported": count_supported,
        }

    def _sparse_poisson_evidence_ready(
        self,
        payload: Mapping[str, Any],
        *,
        count_supported_zero: bool,
    ) -> bool:
        """Return whether sparse evidence has a decisive enough model-order gap."""
        if not bool(payload.get("available", False)):
            return False
        distinct_station_count = int(payload.get("distinct_station_count", 0))
        min_distinct_stations = max(
            1,
            int(self.pf_config.sparse_poisson_evidence_min_distinct_stations),
        )
        if distinct_station_count < min_distinct_stations:
            return False
        selected_count = int(payload.get("selected_count", 0))
        min_margin = max(
            float(self.pf_config.sparse_poisson_evidence_min_bic_margin),
            0.0,
        )
        runner_up_margin = float(
            payload.get("criterion_margin_to_runner_up", float("-inf"))
        )
        if min_margin > 0.0 and np.isfinite(runner_up_margin):
            if runner_up_margin < min_margin:
                return False
        if selected_count > 0:
            simpler_margin = float(
                payload.get("criterion_margin_to_simpler", float("-inf"))
            )
            if min_margin > 0.0 and np.isfinite(simpler_margin):
                if simpler_margin < min_margin:
                    return False
        if selected_count == 0 and bool(count_supported_zero):
            zero_margin = max(
                float(self.pf_config.report_model_order_zero_source_min_bic_margin),
                min_margin,
            )
            if not np.isfinite(runner_up_margin) or runner_up_margin < zero_margin:
                return False
        if selected_count > 1:
            condition_limit = float(self.pf_config.sparse_poisson_evidence_condition_max)
            condition = float(payload.get("condition_number", 0.0))
            if condition_limit > 0.0 and np.isfinite(condition):
                if condition > condition_limit:
                    return False
            corr_limit = float(
                self.pf_config.sparse_poisson_evidence_max_response_correlation
            )
            response_corr = float(
                payload.get("selected_max_response_correlation", 0.0)
            )
            if corr_limit > 0.0 and np.isfinite(response_corr):
                if response_corr > corr_limit:
                    return False
        return True

    def _prune_candidate_verification_queue_with_sparse_evidence(
        self,
        isotope: str,
        payload: Mapping[str, Any],
    ) -> int:
        """Drop queued rescue candidates contradicted by decisive sparse evidence."""
        if not (
            bool(self.pf_config.sparse_poisson_evidence_authoritative)
            or bool(self.pf_config.runtime_report_rescue_verification_queue_only)
        ):
            return 0
        if not bool(self.pf_config.candidate_verification_queue_enable):
            return 0
        if not bool(payload.get("model_order_ready", False)):
            return 0
        memory = self._candidate_verification_queue.get(str(isotope))
        if memory is None:
            return 0
        mem_pos, mem_q, mem_score = memory
        pos_arr = np.asarray(mem_pos, dtype=float).reshape(-1, 3)
        q_arr = np.asarray(mem_q, dtype=float).reshape(-1)
        score_arr = np.asarray(mem_score, dtype=float).reshape(-1)
        if pos_arr.size == 0:
            self._candidate_verification_queue.pop(str(isotope), None)
            return 0
        selected_positions = np.asarray(
            payload.get("selected_positions", []),
            dtype=float,
        ).reshape(-1, 3)
        if selected_positions.size == 0:
            removed = int(pos_arr.shape[0])
            self._candidate_verification_queue.pop(str(isotope), None)
            return removed
        radius = max(float(self.pf_config.report_mle_rescue_dedup_radius_m), 0.0)
        if radius <= 0.0:
            radius = max(float(self.pf_config.cluster_eps_m), 1.0e-6)
        distances = np.linalg.norm(
            pos_arr[:, None, :] - selected_positions[None, :, :],
            axis=2,
        )
        keep = np.min(distances, axis=1) <= radius
        removed = int(np.count_nonzero(~keep))
        if removed <= 0:
            return 0
        if not np.any(keep):
            self._candidate_verification_queue.pop(str(isotope), None)
            return removed
        self._candidate_verification_queue[str(isotope)] = (
            pos_arr[keep].copy(),
            q_arr[keep].copy(),
            score_arr[keep].copy(),
        )
        return removed

    def _merge_sparse_poisson_diagnostics(
        self,
        isotope: str,
        payload: dict[str, Any],
    ) -> None:
        """Store sparse evidence diagnostics and optionally make them authoritative."""
        self._last_sparse_poisson_evidence_diagnostics[str(isotope)] = copy.deepcopy(
            payload
        )
        existing = self._last_report_model_order_diagnostics.get(str(isotope), {})
        if not bool(self.pf_config.sparse_poisson_evidence_authoritative):
            if isinstance(existing, dict):
                existing["sparse_poisson_evidence"] = copy.deepcopy(payload)
                self._last_report_model_order_diagnostics[str(isotope)] = existing
            return
        original = copy.deepcopy(existing) if isinstance(existing, dict) else {}
        merged = copy.deepcopy(payload)
        merged["method"] = str(merged.get("method", "all_history_sparse_poisson"))
        merged["evidence_authoritative"] = True
        merged["report_cluster_model_order"] = original
        self._last_report_model_order_diagnostics[str(isotope)] = merged

    def _update_sparse_poisson_evidence_for_isotope(
        self,
        isotope: str,
        filt: IsotopeParticleFilter,
        data: MeasurementData,
        *,
        background: NDArray[np.float64],
    ) -> dict[str, Any] | None:
        """Update all-history sparse Poisson evidence diagnostics for one isotope."""
        evidence_start = time.perf_counter()
        stage_wall: Dict[str, float] = {}
        if not bool(self.pf_config.sparse_poisson_evidence_enable):
            self._last_sparse_poisson_evidence_diagnostics.pop(str(isotope), None)
            return None
        if data.z_k.size == 0 or self.candidate_sources.size == 0:
            payload = {
                "available": False,
                "reason": "no_measurements_or_candidates",
                "method": "all_history_sparse_poisson",
                "selected_count": 0,
                "candidate_count": int(self.candidate_sources.reshape(-1, 3).shape[0])
                if self.candidate_sources.size
                else 0,
                "model_order_ready": False,
            }
            self._merge_sparse_poisson_diagnostics(isotope, payload)
            return payload
        stage_start = time.perf_counter()
        design = self._cached_candidate_grid_counts(
            filt=filt,
            isotope=isotope,
            data=data,
        )
        stage_wall["response_matrix"] = time.perf_counter() - stage_start
        if design.ndim != 2 or design.shape[0] != data.z_k.size:
            payload = {
                "available": False,
                "reason": "invalid_response_matrix",
                "method": "all_history_sparse_poisson",
                "selected_count": 0,
                "candidate_count": int(self.candidate_sources.reshape(-1, 3).shape[0]),
                "model_order_ready": False,
            }
            self._merge_sparse_poisson_diagnostics(isotope, payload)
            return payload
        stage_start = time.perf_counter()
        evidence = fit_sparse_poisson_evidence(
            np.maximum(np.asarray(data.z_k, dtype=float).reshape(-1), 0.0),
            np.maximum(np.asarray(design, dtype=float), 0.0),
            background=background,
            config=self._sparse_poisson_evidence_config(),
        )
        stage_wall["count_sparse_fit"] = time.perf_counter() - stage_start
        payload = sparse_poisson_evidence_to_diagnostics(evidence)
        candidate_pool = np.asarray(self.candidate_sources, dtype=float).reshape(-1, 3)
        selected_indices = np.asarray(payload.get("selected_indices", []), dtype=int)
        selected_positions = (
            candidate_pool[selected_indices]
            if selected_indices.size
            else np.zeros((0, 3), dtype=float)
        )
        payload["selected_positions"] = [
            [float(value) for value in row]
            for row in np.asarray(selected_positions, dtype=float).reshape(-1, 3)
        ]
        if bool(self.pf_config.sparse_poisson_ambiguity_report_enable):
            payload["ambiguity_clusters"] = [
                dict(cluster)
                for cluster in sparse_poisson_ambiguity_diagnostics(
                    evidence,
                    np.maximum(np.asarray(design, dtype=float), 0.0),
                    candidate_positions=candidate_pool,
                    correlation_threshold=float(
                        self.pf_config.sparse_poisson_ambiguity_corr_threshold
                    ),
                    bic_gap_threshold=float(
                        self.pf_config.sparse_poisson_ambiguity_bic_gap_threshold
                    ),
                    condition_number_threshold=float(
                        self.pf_config.sparse_poisson_ambiguity_condition_max
                    ),
                )
            ]
        stage_start = time.perf_counter()
        count_refinement = self._count_sparse_offgrid_refinement_payload(
            isotope,
            filt,
            data,
            payload,
            background=np.asarray(background, dtype=float),
        )
        self._apply_offgrid_refinement_to_payload(payload, count_refinement)
        stage_wall["count_offgrid_refine"] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        spectral_payload = self._spectral_sparse_poisson_evidence_payload(isotope)
        stage_wall["spectral_sparse_fit"] = time.perf_counter() - stage_start
        if spectral_payload is not None:
            count_payload = copy.deepcopy(payload)
            if (
                bool(self.pf_config.sparse_poisson_spectral_evidence_primary)
                and bool(spectral_payload.get("available", False))
            ):
                spectral_payload["count_sparse_poisson_evidence"] = count_payload
                payload = spectral_payload
            else:
                payload["spectral_bin_evidence"] = copy.deepcopy(spectral_payload)
        count_support = self._sparse_poisson_count_support(data, background)
        payload.update(count_support)
        count_supported_zero = bool(
            int(payload.get("selected_count", 0)) == 0
            and bool(count_support.get("count_supported", False))
        )
        payload["count_supported_zero_source"] = bool(count_supported_zero)
        payload["zero_source_ready_margin"] = float(
            max(
                float(self.pf_config.report_model_order_zero_source_min_bic_margin),
                float(self.pf_config.sparse_poisson_evidence_min_bic_margin),
            )
        )
        payload["model_order_ready"] = bool(
            self._sparse_poisson_evidence_ready(
                payload,
                count_supported_zero=count_supported_zero,
            )
        )
        payload["evidence_authoritative"] = bool(
            self.pf_config.sparse_poisson_evidence_authoritative
        )
        pruned = self._prune_candidate_verification_queue_with_sparse_evidence(
            isotope,
            payload,
        )
        if pruned > 0:
            payload["candidate_verification_queue_pruned"] = int(pruned)
        stage_wall["total"] = time.perf_counter() - evidence_start
        payload["sparse_evidence_stage_wall_s"] = {
            key: float(value) for key, value in stage_wall.items()
        }
        self._merge_sparse_poisson_diagnostics(isotope, payload)
        return payload

    def _refit_sparse_evidence_strengths(
        self,
        isotope: str,
        filt: IsotopeParticleFilter | None,
        data: MeasurementData | None,
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
        *,
        background: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Profile evidence-selected source strengths with the count response model."""
        pos_arr = np.asarray(positions, dtype=float).reshape(-1, 3)
        source_count = int(pos_arr.shape[0])
        if source_count <= 0:
            return np.zeros(0, dtype=float)
        q_arr = np.maximum(np.asarray(strengths, dtype=float).reshape(-1), 0.0)
        if q_arr.size != source_count:
            q_arr = np.zeros(source_count, dtype=float)
        if (
            filt is None
            or data is None
            or data.z_k.size == 0
            or not hasattr(filt, "continuous_kernel")
        ):
            return q_arr
        z_obs = np.maximum(np.asarray(data.z_k, dtype=float).reshape(-1), 0.0)
        if background is None:
            background_arr = self._background_counts_for_report_refit(
                isotope,
                data.live_times,
            )
        else:
            background_arr = np.asarray(background, dtype=float)
        background_arr = measurement_vector(
            background_arr,
            z_obs.size,
            "background",
            min_value=0.0,
        )
        unit_strengths = np.ones(source_count, dtype=float)
        design = self._cached_expected_counts_per_source(
            filt=filt,
            isotope=isotope,
            data=data,
            sources=pos_arr,
            strengths=unit_strengths,
        )
        design = np.maximum(np.asarray(design, dtype=float), 0.0)
        if design.ndim != 2 or design.shape != (z_obs.size, source_count):
            return q_arr
        eps = max(float(self.pf_config.report_strength_refit_eps), 1.0e-12)
        q_max = float(self.pf_config.birth_q_max)
        support_floor = max(float(self.pf_config.min_strength), 0.0) * (1.0 + 1.0e-6)
        finite_positive = np.isfinite(q_arr) & (q_arr > support_floor)
        if np.any(finite_positive):
            seed = q_arr.copy()
            fallback = float(np.median(seed[finite_positive]))
            seed[~np.isfinite(seed) | (seed <= support_floor)] = max(
                fallback,
                support_floor,
            )
        else:
            signal = np.maximum(z_obs - background_arr, 0.0)
            column_sums = np.maximum(np.sum(design, axis=0), eps)
            signal_total = float(np.sum(signal))
            if signal_total > 0.0:
                seed = np.full(
                    source_count,
                    signal_total / max(float(np.sum(column_sums)), eps),
                    dtype=float,
                )
            else:
                seed = np.zeros(source_count, dtype=float)
            seed = np.maximum(seed, support_floor)
        if np.isfinite(q_max) and q_max > 0.0:
            seed = np.minimum(seed, q_max)
        q_refit = self._solve_report_strengths(
            design=design,
            z_obs=z_obs,
            background=background_arr,
            observation_variances=data.observation_variances,
            initial_strengths=seed,
            eps=eps,
            q_max=q_max,
        )
        q_refit = np.maximum(np.asarray(q_refit, dtype=float).reshape(-1), 0.0)
        if q_refit.size != source_count:
            return seed
        q_refit[~np.isfinite(q_refit)] = seed[~np.isfinite(q_refit)]
        return np.maximum(q_refit, support_floor)

    def _sparse_poisson_report_override(
        self,
        isotope: str,
        *,
        filt: IsotopeParticleFilter | None = None,
        data: MeasurementData | None = None,
        background: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        """Return evidence-selected report positions when sparse evidence is authoritative."""
        if not bool(self.pf_config.sparse_poisson_evidence_authoritative):
            return None
        payload = self._last_sparse_poisson_evidence_diagnostics.get(str(isotope))
        if not isinstance(payload, dict) or not bool(payload.get("available", False)):
            return None
        if not bool(payload.get("model_order_ready", False)):
            return None
        try:
            selected_count = int(payload.get("selected_count", 0))
        except (TypeError, ValueError):
            return None
        positions = np.asarray(payload.get("selected_positions", []), dtype=float).reshape(
            -1,
            3,
        )
        strengths = np.maximum(
            np.asarray(payload.get("selected_strengths", []), dtype=float).reshape(-1),
            0.0,
        )
        if selected_count == 0:
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        if positions.shape[0] != selected_count:
            return None
        strengths = self._refit_sparse_evidence_strengths(
            isotope,
            filt,
            data,
            positions,
            strengths,
            background=background,
        )
        if strengths.size != selected_count:
            return None
        keep = np.isfinite(positions).all(axis=1) & np.isfinite(strengths)
        if not bool(np.all(keep)):
            return None
        return positions.copy(), strengths.copy()

    def _apply_sparse_poisson_particle_prune(
        self,
        isotope: str,
        filt: IsotopeParticleFilter,
        data: MeasurementData,
        *,
        report_positions: NDArray[np.float64],
        evidence_positions: NDArray[np.float64],
    ) -> int:
        """Prune PF slots near report modes rejected by authoritative sparse evidence."""
        if not bool(self.pf_config.report_model_order_prune_particles):
            return 0
        evidence_pos = np.asarray(evidence_positions, dtype=float).reshape(-1, 3)
        report_pos = np.asarray(report_positions, dtype=float).reshape(-1, 3)
        if evidence_pos.size == 0 or report_pos.size == 0:
            return 0
        combined = np.vstack([evidence_pos, report_pos])
        selected_mask = np.zeros(combined.shape[0], dtype=bool)
        selected_mask[: evidence_pos.shape[0]] = True
        radius = float(self.pf_config.report_model_order_particle_prune_radius_m)
        if radius <= 0.0:
            radius = max(float(self.pf_config.cluster_eps_m), 1.0e-6)
        removed = filt.apply_report_model_order_cluster_prune(
            combined,
            selected_mask,
            radius_m=radius,
        )
        if removed <= 0:
            return 0
        filt.refresh_weights_from_measurements(data)
        diagnostics = self._last_report_model_order_diagnostics.get(str(isotope))
        if isinstance(diagnostics, dict):
            diagnostics["particle_pruned_source_slots"] = int(removed)
            diagnostics["particle_prune_radius_m"] = float(radius)
            diagnostics["particle_prune_authority"] = "sparse_poisson_evidence"
        self._invalidate_report_cache()
        return int(removed)

    def _select_report_clusters_by_model_order(
        self,
        isotope: str,
        filt: IsotopeParticleFilter,
        data: MeasurementData,
        *,
        design: NDArray[np.float64],
        strengths: NDArray[np.float64],
        background: NDArray[np.float64],
        eps: float,
        q_max: float,
    ) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
        """
        Select reported clusters by exhaustive BIC model-order comparison.

        The comparison is over all subsets of the currently reported candidate
        clusters up to a configurable cap.  It is not tied to any particular
        source count: K=0,1,...,N are all evaluated, and the selected K is the
        one with the highest refit Poisson log-likelihood minus the standard
        half-BIC parameter penalty.
        """
        model_order_start = time.perf_counter()
        design_arr = np.asarray(design, dtype=float)
        q_initial = np.asarray(strengths, dtype=float).reshape(-1)
        source_count = int(q_initial.size)
        q_full = np.zeros(source_count, dtype=float)
        keep = np.zeros(source_count, dtype=bool)
        if source_count <= 0:
            self._last_report_model_order_diagnostics[isotope] = {
                "candidate_count": 0,
                "selected_count": 0,
                "method": "empty",
            }
            return keep, q_full
        if not bool(self.pf_config.report_cluster_model_selection):
            self._last_report_model_order_diagnostics[isotope] = {
                "candidate_count": source_count,
                "selected_count": source_count,
                "method": "disabled",
            }
            return np.ones(source_count, dtype=bool), q_initial
        max_exhaustive = max(
            1,
            int(self.pf_config.report_cluster_model_selection_max_candidates),
        )
        max_model_sources = min(source_count, self._report_max_sources_per_isotope())
        if max_model_sources <= 0:
            self._last_report_model_order_diagnostics[isotope] = {
                "candidate_count": source_count,
                "selected_count": 0,
                "method": "source_cap_zero",
                "max_model_sources": int(max_model_sources),
            }
            return keep, q_full
        if source_count > max_exhaustive:
            keep_greedy, q_greedy = self._select_report_clusters_by_refit_after_remove(
                filt,
                data,
                design=design_arr,
                strengths=q_initial,
                background=background,
                eps=eps,
                q_max=q_max,
            )
            selected_count = int(np.count_nonzero(keep_greedy))
            if selected_count > max_model_sources:
                selected = np.flatnonzero(keep_greedy)
                q_selected = np.asarray(q_greedy, dtype=float).reshape(-1)[selected]
                order = np.argsort(np.maximum(q_selected, 0.0))[::-1]
                keep_limited = np.zeros_like(keep_greedy, dtype=bool)
                keep_limited[selected[order[:max_model_sources]]] = True
                q_limited = np.zeros_like(q_greedy, dtype=float)
                q_limited[keep_limited] = q_greedy[keep_limited]
                keep_greedy = keep_limited
                q_greedy = q_limited
                selected_count = int(np.count_nonzero(keep_greedy))
            self._last_report_model_order_diagnostics[isotope] = {
                "candidate_count": source_count,
                "selected_count": selected_count,
                "method": "greedy_refit_after_remove",
                "max_exhaustive_candidates": max_exhaustive,
                "max_model_sources": int(max_model_sources),
            }
            return keep_greedy, q_greedy

        measurement_count = int(data.z_k.size)
        penalty_params = int(self.pf_config.report_cluster_bic_penalty_params)
        background_ll = self._report_cluster_log_likelihood(
            filt,
            data,
            design=np.zeros((measurement_count, 0), dtype=float),
            strengths=np.zeros(0, dtype=float),
            background=background,
        )
        z_arr = np.maximum(np.asarray(data.z_k, dtype=float).reshape(-1), 0.0)
        bg_arr = measurement_vector(
            background,
            z_arr.size,
            "background",
            min_value=0.0,
        )
        variances = measurement_vector(
            data.observation_variances,
            z_arr.size,
            "observation_variances",
            min_value=1.0,
        )
        signal = np.maximum(z_arr - bg_arr, 0.0)
        signal_total = float(np.sum(signal))
        signal_max = float(np.max(signal)) if signal.size else 0.0
        signal_snr = float(
            signal_total / np.sqrt(max(float(np.sum(variances)), 1.0e-12))
        )
        corr_penalty_weight = max(
            0.0,
            float(self.pf_config.report_model_order_corr_penalty_weight),
        )
        corr_penalty_threshold = float(
            np.clip(
                float(self.pf_config.report_model_order_corr_penalty_threshold),
                0.0,
                1.0,
            )
        )
        corr_penalty_power = max(
            1.0e-6,
            float(self.pf_config.report_model_order_corr_penalty_power),
        )
        subset_corr_prune_threshold = float(
            np.clip(
                float(self.pf_config.report_model_order_subset_corr_prune_threshold),
                0.0,
                1.0,
            )
        )
        subset_corr_matrix: NDArray[np.float64] | None = None
        if 0.0 < subset_corr_prune_threshold < 1.0 and source_count > 1:
            column_norm = np.linalg.norm(np.maximum(design_arr, 0.0), axis=0)
            valid = column_norm > max(float(eps), 1.0e-12)
            normalized = np.divide(
                np.maximum(design_arr, 0.0),
                np.maximum(column_norm[None, :], max(float(eps), 1.0e-12)),
                out=np.zeros_like(design_arr, dtype=float),
                where=valid[None, :],
            )
            subset_corr_matrix = np.abs(normalized.T @ normalized)

        def _subset_pruned_by_response_correlation(indices: Sequence[int]) -> bool:
            """Return True when a subset contains indistinguishable candidates."""
            if subset_corr_matrix is None or len(indices) <= 1:
                return False
            subset = np.asarray(indices, dtype=int)
            corr_sub = subset_corr_matrix[np.ix_(subset, subset)]
            upper = np.triu_indices(corr_sub.shape[0], k=1)
            if upper[0].size == 0:
                return False
            return bool(np.any(corr_sub[upper] >= float(subset_corr_prune_threshold)))

        def _residual_stats_for_subset(
            indices: Sequence[int],
            strengths_for_subset: NDArray[np.float64],
        ) -> dict[str, float]:
            """Return full-history count residual diagnostics for one subset."""
            subset_indices = tuple(int(idx) for idx in indices)
            if subset_indices:
                subset_design = design_arr[
                    :,
                    np.asarray(subset_indices, dtype=int),
                ]
                subset_strengths = np.asarray(
                    strengths_for_subset,
                    dtype=float,
                ).reshape(-1)
                prediction = bg_arr + subset_design @ subset_strengths
            else:
                prediction = bg_arr
            positive = np.maximum(z_arr - prediction, 0.0)
            negative = np.maximum(prediction - z_arr, 0.0)
            reference_signal = max(float(np.sum(z_arr)), float(np.sum(bg_arr)), eps)
            observed_total = max(float(np.sum(z_arr)), eps)
            prediction_total = float(np.sum(prediction))
            return {
                "positive_residual_fraction": (
                    float(np.sum(positive)) / reference_signal
                ),
                "negative_residual_fraction": (
                    float(np.sum(negative)) / reference_signal
                ),
                "overprediction_fraction": (
                    max(0.0, prediction_total - observed_total) / observed_total
                ),
                "positive_residual_chi2": float(
                    np.sum((positive * positive) / variances)
                ),
            }

        background_residual_stats = _residual_stats_for_subset(
            (),
            np.zeros(0, dtype=float),
        )
        best: dict[str, Any] = {
            "criterion": background_ll,
            "ll": background_ll,
            "indices": tuple(),
            "strengths": np.zeros(0, dtype=float),
            "condition_number": 1.0,
            "correlation_penalty": 0.0,
            **background_residual_stats,
        }
        best_by_k: dict[int, dict[str, Any]] = {
            0: {
                "criterion": float(background_ll),
                "ll": float(background_ll),
                "condition_number": 1.0,
                "correlation_penalty": 0.0,
                "indices": [],
                "strengths": np.zeros(0, dtype=float),
                **background_residual_stats,
            }
        }
        subset_tasks: list[tuple[int, int, tuple[int, ...], float]] = []
        ordinal = 0
        subset_corr_pruned = 0
        for k in range(1, max_model_sources + 1):
            penalty = filt._bic_model_penalty(
                measurement_count,
                penalty_params * k,
            )
            for indices in itertools.combinations(range(source_count), k):
                if _subset_pruned_by_response_correlation(indices):
                    subset_corr_pruned += 1
                    continue
                subset_tasks.append(
                    (ordinal, k, tuple(int(i) for i in indices), penalty)
                )
                ordinal += 1

        def _score_subset(
            task: tuple[int, int, tuple[int, ...], float],
        ) -> dict[str, Any]:
            """Score one fixed source subset without mutating estimator state."""
            task_ordinal, k, indices, penalty = task
            subset = np.asarray(indices, dtype=int)
            subset_design = design_arr[:, subset]
            subset_initial = q_initial[subset]
            subset_q = self._solve_report_strengths(
                design=subset_design,
                z_obs=data.z_k,
                background=background,
                observation_variances=data.observation_variances,
                initial_strengths=subset_initial,
                eps=eps,
                q_max=q_max,
            )
            ll_value = self._report_cluster_log_likelihood(
                filt,
                data,
                design=subset_design,
                strengths=subset_q,
                background=background,
            )
            corr_penalty = self._report_design_correlation_penalty(
                subset_design,
                threshold=corr_penalty_threshold,
                weight=corr_penalty_weight,
                power=corr_penalty_power,
                eps=eps,
            )
            criterion = float(ll_value - penalty - corr_penalty)
            condition_number = self._report_design_condition_number(
                subset_design,
                eps=eps,
            )
            return {
                "ordinal": int(task_ordinal),
                "k": int(k),
                "criterion": float(criterion),
                "ll": float(ll_value),
                "condition_number": float(condition_number),
                "correlation_penalty": float(corr_penalty),
                "indices": tuple(int(i) for i in indices),
                "strengths": np.asarray(subset_q, dtype=float),
            }

        def _score_subset_batch(
            tasks: list[tuple[int, int, tuple[int, ...], float]],
        ) -> list[dict[str, Any]]:
            """Score same-cardinality subsets using batched NumPy operations."""
            results: list[dict[str, Any]] = []
            tasks_by_k: dict[int, list[tuple[int, int, tuple[int, ...], float]]] = {}
            for task in tasks:
                tasks_by_k.setdefault(int(task[1]), []).append(task)
            for k, grouped_tasks in sorted(tasks_by_k.items()):
                indices = np.asarray(
                    [task[2] for task in grouped_tasks],
                    dtype=int,
                )
                if indices.size == 0:
                    continue
                subset_designs = np.asarray(
                    design_arr[:, indices],
                    dtype=float,
                ).transpose(1, 0, 2)
                subset_initial = q_initial[indices]
                subset_q = self._solve_report_strengths_batch(
                    design_batch=subset_designs,
                    z_obs=data.z_k,
                    background=background,
                    observation_variances=data.observation_variances,
                    initial_strengths=subset_initial,
                    eps=eps,
                    q_max=q_max,
                )
                lam_bm = np.maximum(
                    np.asarray(background, dtype=float)[None, :]
                    + np.einsum("bmk,bk->bm", subset_designs, subset_q),
                    eps,
                )
                ll_values = filt._count_log_likelihood_matrix_np(
                    data.z_k,
                    lam_bm.T,
                    observation_count_variance=data.observation_variances,
                )
                condition_numbers = self._report_design_condition_numbers_batch(
                    subset_designs,
                    eps=eps,
                )
                corr_penalties = self._report_design_correlation_penalties_batch(
                    subset_designs,
                    threshold=corr_penalty_threshold,
                    weight=corr_penalty_weight,
                    power=corr_penalty_power,
                    eps=eps,
                )
                for local_idx, task in enumerate(grouped_tasks):
                    task_ordinal, _, task_indices, penalty = task
                    ll_value = float(ll_values[local_idx])
                    corr_penalty = float(corr_penalties[local_idx])
                    criterion = float(ll_value - penalty - corr_penalty)
                    results.append(
                        {
                            "ordinal": int(task_ordinal),
                            "k": int(k),
                            "criterion": float(criterion),
                            "ll": float(ll_value),
                            "condition_number": float(condition_numbers[local_idx]),
                            "correlation_penalty": float(corr_penalty),
                            "indices": tuple(int(i) for i in task_indices),
                            "strengths": np.asarray(
                                subset_q[local_idx],
                                dtype=float,
                            ),
                        }
                    )
            return results

        workers = max(1, int(self.pf_config.report_model_order_workers))
        parallel_min = max(
            1,
            int(self.pf_config.report_model_order_parallel_min_subsets),
        )
        use_batched_subset_scoring = workers > 1 and len(subset_tasks) >= parallel_min
        if workers > 1 and len(subset_tasks) >= parallel_min:
            subset_results = _score_subset_batch(subset_tasks)
        else:
            subset_results = [_score_subset(task) for task in subset_tasks]

        for result in sorted(subset_results, key=lambda item: int(item["ordinal"])):
            k = int(result["k"])
            indices = tuple(int(i) for i in result["indices"])
            criterion = float(result["criterion"])
            ll_value = float(result["ll"])
            condition_number = float(result["condition_number"])
            corr_penalty = float(result.get("correlation_penalty", 0.0))
            strengths_for_subset = np.asarray(result["strengths"], dtype=float)
            residual_stats = _residual_stats_for_subset(
                indices,
                strengths_for_subset,
            )
            existing = best_by_k.get(k)
            if existing is None or criterion > float(existing["criterion"]):
                best_by_k[k] = {
                    "criterion": float(criterion),
                    "ll": float(ll_value),
                    "condition_number": float(condition_number),
                    "correlation_penalty": float(corr_penalty),
                    "indices": [int(i) for i in indices],
                    "strengths": strengths_for_subset.copy(),
                    **residual_stats,
                }
            if criterion > float(best["criterion"]):
                best = {
                    "criterion": criterion,
                    "ll": float(ll_value),
                    "indices": tuple(int(i) for i in indices),
                    "strengths": strengths_for_subset.copy(),
                    "condition_number": float(condition_number),
                    "correlation_penalty": float(corr_penalty),
                    **residual_stats,
                }

        underfit_min_fraction = float(
            self.pf_config.report_model_order_underfit_min_residual_fraction
        )
        if underfit_min_fraction < 0.0:
            underfit_min_fraction = float(
                self.pf_config.report_mle_rescue_min_residual_fraction
            )
        underfit_min_fraction = max(0.0, underfit_min_fraction)
        underfit_min_chi2 = max(
            0.0,
            float(self.pf_config.report_model_order_underfit_min_positive_chi2),
        )

        def _model_underfits(result: Mapping[str, Any]) -> bool:
            """Return whether a candidate model leaves a count-supported residual."""
            if not bool(self.pf_config.report_model_order_underfit_gate):
                return False
            if underfit_min_fraction <= 0.0:
                return False
            residual_fraction = float(result.get("positive_residual_fraction", 0.0))
            positive_chi2 = float(result.get("positive_residual_chi2", 0.0))
            if residual_fraction < underfit_min_fraction:
                return False
            if underfit_min_chi2 > 0.0 and positive_chi2 < underfit_min_chi2:
                return False
            return True

        underfit_gate_overrode_selection = False
        selected_model_underfit = _model_underfits(best)
        if selected_model_underfit:
            replacement_candidates = [
                stats for stats in best_by_k.values() if not _model_underfits(stats)
            ]
            if replacement_candidates:
                replacement = max(
                    replacement_candidates,
                    key=lambda stats: float(stats["criterion"]),
                )
                best = {
                    "criterion": float(replacement["criterion"]),
                    "ll": float(replacement["ll"]),
                    "indices": tuple(int(i) for i in replacement["indices"]),
                    "strengths": np.asarray(
                        replacement["strengths"],
                        dtype=float,
                    ).copy(),
                    "condition_number": float(replacement["condition_number"]),
                    "correlation_penalty": float(
                        replacement.get("correlation_penalty", 0.0)
                    ),
                    "positive_residual_fraction": float(
                        replacement.get("positive_residual_fraction", 0.0)
                    ),
                    "negative_residual_fraction": float(
                        replacement.get("negative_residual_fraction", 0.0)
                    ),
                    "overprediction_fraction": float(
                        replacement.get("overprediction_fraction", 0.0)
                    ),
                    "positive_residual_chi2": float(
                        replacement.get("positive_residual_chi2", 0.0)
                    ),
                }
                underfit_gate_overrode_selection = True
                selected_model_underfit = False
        selected_indices = tuple(int(i) for i in best["indices"])
        if selected_indices:
            keep[np.asarray(selected_indices, dtype=int)] = True
            q_full[np.asarray(selected_indices, dtype=int)] = np.asarray(
                best["strengths"],
                dtype=float,
            )
            selected_design = design_arr[:, np.asarray(selected_indices, dtype=int)]
            selected_max_corr = self._report_design_max_abs_correlation(
                selected_design,
                eps=eps,
            )
        else:
            selected_max_corr = 0.0
        selected_count = int(np.count_nonzero(keep))
        simpler_criteria = [
            float(stats["criterion"])
            for k, stats in best_by_k.items()
            if k < selected_count
        ]
        simpler_margin = (
            float(best["criterion"]) - max(simpler_criteria)
            if simpler_criteria
            else float("inf")
        )
        runner_up_criteria = [
            float(stats["criterion"])
            for stats in best_by_k.values()
            if list(stats["indices"]) != [int(i) for i in selected_indices]
        ]
        runner_up_margin = (
            float(best["criterion"]) - max(runner_up_criteria)
            if runner_up_criteria
            else float("inf")
        )
        selected_residual_fraction = float(best.get("positive_residual_fraction", 0.0))
        selected_negative_residual_fraction = float(
            best.get("negative_residual_fraction", 0.0)
        )
        selected_overprediction_fraction = float(
            best.get("overprediction_fraction", 0.0)
        )
        selected_residual_chi2 = float(best.get("positive_residual_chi2", 0.0))
        selected_correlation_penalty = float(best.get("correlation_penalty", 0.0))
        zero_source_ready_margin = max(
            float(self.pf_config.report_model_order_min_bic_margin),
            float(self.pf_config.report_model_order_zero_source_min_bic_margin),
        )
        count_supported_zero = selected_count == 0 and (
            signal_total >= max(float(self.pf_config.structural_update_min_counts), 0.0)
            or signal_max
            >= max(float(self.pf_config.conditional_strength_refit_min_count), 0.0)
            or signal_snr >= max(float(self.pf_config.structural_update_min_snr), 0.0)
        )
        model_order_ready = True
        min_margin = float(self.pf_config.report_model_order_min_bic_margin)
        if min_margin > 0.0:
            if np.isfinite(runner_up_margin) and runner_up_margin < min_margin:
                model_order_ready = False
        if count_supported_zero and (
            not np.isfinite(runner_up_margin)
            or runner_up_margin < zero_source_ready_margin
        ):
            model_order_ready = False
        if selected_count > 1:
            if np.isfinite(simpler_margin) and simpler_margin < min_margin:
                model_order_ready = False
            max_condition = float(self.pf_config.report_model_order_condition_max)
            if max_condition > 0.0 and float(best["condition_number"]) > max_condition:
                model_order_ready = False
            if selected_correlation_penalty > 0.0:
                model_order_ready = False
        if selected_model_underfit:
            model_order_ready = False
        model_order_wall_s = time.perf_counter() - model_order_start
        self._last_report_model_order_diagnostics[isotope] = {
            "candidate_count": source_count,
            "selected_count": selected_count,
            "selected_indices": [int(i) for i in selected_indices],
            "selected_strengths": [
                float(value)
                for value in np.asarray(q_full[keep], dtype=float).reshape(-1)
            ],
            "method": "exhaustive_bic",
            "max_model_sources": int(max_model_sources),
            "workers": int(workers if len(subset_tasks) >= parallel_min else 1),
            "evaluation_mode": (
                "batched_numpy" if use_batched_subset_scoring else "serial"
            ),
            "evaluated_subsets": int(len(subset_tasks) + 1),
            "subset_correlation_pruned_subsets": int(subset_corr_pruned),
            "subset_correlation_prune_threshold": float(subset_corr_prune_threshold),
            "background_ll": float(background_ll),
            "best_ll": float(best["ll"]),
            "best_criterion": float(best["criterion"]),
            "condition_number": float(best["condition_number"]),
            "selected_max_response_correlation": float(selected_max_corr),
            "selected_correlation_penalty": float(selected_correlation_penalty),
            "correlation_penalty_weight": float(corr_penalty_weight),
            "correlation_penalty_threshold": float(corr_penalty_threshold),
            "correlation_penalty_power": float(corr_penalty_power),
            "criterion_margin_to_simpler": float(simpler_margin),
            "criterion_margin_to_runner_up": float(runner_up_margin),
            "evaluation_wall_s": float(model_order_wall_s),
            "observed_signal_total_counts": float(signal_total),
            "observed_signal_max_count": float(signal_max),
            "observed_signal_snr": float(signal_snr),
            "selected_positive_residual_fraction": float(selected_residual_fraction),
            "selected_negative_residual_fraction": float(
                selected_negative_residual_fraction
            ),
            "selected_overprediction_fraction": float(selected_overprediction_fraction),
            "selected_positive_residual_chi2": float(selected_residual_chi2),
            "underfit_gate_enabled": bool(
                self.pf_config.report_model_order_underfit_gate
            ),
            "underfit_gate_min_residual_fraction": float(underfit_min_fraction),
            "underfit_gate_min_positive_chi2": float(underfit_min_chi2),
            "underfit_gate_overrode_selection": bool(underfit_gate_overrode_selection),
            "underfit_gate_selected_model_underfit": bool(selected_model_underfit),
            "zero_source_ready_margin": float(zero_source_ready_margin),
            "count_supported_zero_source": bool(count_supported_zero),
            "model_order_ready": bool(model_order_ready),
            "best_by_k": {
                str(k): {
                    "criterion": float(stats["criterion"]),
                    "ll": float(stats["ll"]),
                    "condition_number": float(stats["condition_number"]),
                    "correlation_penalty": float(stats.get("correlation_penalty", 0.0)),
                    "indices": [int(i) for i in stats["indices"]],
                    "positive_residual_fraction": float(
                        stats.get("positive_residual_fraction", 0.0)
                    ),
                    "negative_residual_fraction": float(
                        stats.get("negative_residual_fraction", 0.0)
                    ),
                    "overprediction_fraction": float(
                        stats.get("overprediction_fraction", 0.0)
                    ),
                    "positive_residual_chi2": float(
                        stats.get("positive_residual_chi2", 0.0)
                    ),
                    "underfit": bool(_model_underfits(stats)),
                }
                for k, stats in sorted(best_by_k.items())
            },
        }
        return keep, q_full

    def _select_report_clusters_by_refit_after_remove(
        self,
        filt: IsotopeParticleFilter,
        data: MeasurementData,
        *,
        design: NDArray[np.float64],
        strengths: NDArray[np.float64],
        background: NDArray[np.float64],
        eps: float,
        q_max: float,
    ) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
        """Drop redundant reported clusters using refit-after-remove BIC tests."""
        design_arr = np.asarray(design, dtype=float)
        q = np.asarray(strengths, dtype=float).reshape(-1)
        source_count = int(q.size)
        keep = np.ones(source_count, dtype=bool)
        if (
            source_count <= 1
            or not bool(self.pf_config.report_cluster_model_selection)
            or data.z_k.size == 0
        ):
            return keep, q
        penalty_gain = filt._bic_model_penalty(
            int(data.z_k.size),
            int(self.pf_config.report_cluster_bic_penalty_params),
        )
        allowed_loss = penalty_gain + float(
            self.pf_config.report_cluster_delta_ll_threshold
        )
        while int(np.count_nonzero(keep)) > 1:
            active_idx = np.flatnonzero(keep)
            active_design = design_arr[:, active_idx]
            active_q = q[active_idx]
            ll_full = self._report_cluster_log_likelihood(
                filt,
                data,
                design=active_design,
                strengths=active_q,
                background=background,
            )
            best_global_idx: int | None = None
            best_loss = np.inf
            best_trial_q: NDArray[np.float64] | None = None
            for local_idx, global_idx in enumerate(active_idx):
                trial_local_keep = np.ones(active_idx.size, dtype=bool)
                trial_local_keep[local_idx] = False
                trial_design = active_design[:, trial_local_keep]
                trial_initial = active_q[trial_local_keep]
                trial_q = self._solve_report_strengths(
                    design=trial_design,
                    z_obs=data.z_k,
                    background=background,
                    observation_variances=data.observation_variances,
                    initial_strengths=trial_initial,
                    eps=eps,
                    q_max=q_max,
                )
                ll_without = self._report_cluster_log_likelihood(
                    filt,
                    data,
                    design=trial_design,
                    strengths=trial_q,
                    background=background,
                )
                loss = float(ll_full - ll_without)
                if np.isfinite(loss) and loss <= allowed_loss and loss < best_loss:
                    best_loss = loss
                    best_global_idx = int(global_idx)
                    best_trial_q = trial_q
            if best_global_idx is None or best_trial_q is None:
                break
            keep[best_global_idx] = False
            q[np.flatnonzero(keep)] = best_trial_q
        return keep, q

    def _apply_report_model_order_particle_prune(
        self,
        isotope: str,
        filt: IsotopeParticleFilter,
        data: MeasurementData,
        positions: NDArray[np.float64],
        selected_by_model: NDArray[np.bool_],
    ) -> int:
        """Apply report-level BIC cluster rejections to PF source slots."""
        if not bool(self.pf_config.report_model_order_prune_particles):
            return 0
        selected = np.asarray(selected_by_model, dtype=bool).reshape(-1)
        if selected.size == 0 or np.all(selected) or not np.any(selected):
            return 0
        radius = float(self.pf_config.report_model_order_particle_prune_radius_m)
        if radius <= 0.0:
            radius = max(float(self.pf_config.cluster_eps_m), 1.0e-6)
        removed = filt.apply_report_model_order_cluster_prune(
            np.asarray(positions, dtype=float),
            selected,
            radius_m=radius,
        )
        if removed <= 0:
            return 0
        filt.refresh_weights_from_measurements(data)
        diagnostics = self._last_report_model_order_diagnostics.get(isotope)
        if isinstance(diagnostics, dict):
            diagnostics["particle_pruned_source_slots"] = int(removed)
            diagnostics["particle_prune_radius_m"] = float(radius)
        self._invalidate_report_cache()
        return int(removed)

    def _report_surface_local_candidates(
        self,
        filt: IsotopeParticleFilter,
        position_xyz: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Return a bounded local surface stencil around one report candidate.

        The runtime count response is still evaluated by the normal PF kernel.
        The only approximation here is the finite local candidate stencil, which
        is capped by configuration so it cannot become a global surface search.
        """
        center = np.asarray(position_xyz, dtype=float).reshape(1, 3)
        radius = max(float(self.pf_config.report_surface_local_refine_radius_m), 0.0)
        steps = max(0, int(self.pf_config.report_surface_local_refine_grid_steps))
        max_points = max(
            1,
            int(self.pf_config.report_surface_local_refine_max_candidates_per_source),
        )
        if radius <= 0.0 or steps <= 0:
            projected = filt._project_positions_to_source_prior(center)
            return np.asarray(projected, dtype=float).reshape(-1, 3)
        axis = np.linspace(-radius, radius, 2 * steps + 1, dtype=float)
        mesh = np.meshgrid(axis, axis, axis, indexing="ij")
        offsets = np.stack([item.reshape(-1) for item in mesh], axis=1)
        offset_norm = np.linalg.norm(offsets, axis=1)
        order = np.lexsort((offsets[:, 2], offsets[:, 1], offsets[:, 0], offset_norm))
        offsets = offsets[order[:max_points]]
        projected = filt._project_positions_to_source_prior(center + offsets)
        projected = np.asarray(projected, dtype=float).reshape(-1, 3)
        finite = np.all(np.isfinite(projected), axis=1)
        projected = projected[finite]
        if projected.size == 0:
            return np.zeros((0, 3), dtype=float)
        _, unique_indices = np.unique(
            np.round(projected, 9),
            axis=0,
            return_index=True,
        )
        return projected[np.sort(unique_indices)]

    def _refine_report_surface_positions(
        self,
        isotope: str,
        filt: IsotopeParticleFilter,
        data: MeasurementData,
        *,
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
        design: NDArray[np.float64],
        background: NDArray[np.float64],
        eps: float,
        q_max: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
        """
        Refine report positions with a bounded, batched local surface search.

        The coordinate sweep over source slots is intentionally small and
        configuration-capped; each source's local stencil is scored in one
        vectorized likelihood batch.  This avoids the combinatorial explosion of
        a joint continuous all-surface search while preserving the PF response
        model and observation likelihood.
        """
        pos_arr = np.asarray(positions, dtype=float).reshape(-1, 3).copy()
        q_arr = np.asarray(strengths, dtype=float).reshape(-1).copy()
        design_arr = np.maximum(np.asarray(design, dtype=float), 0.0).copy()
        if (
            not bool(self.pf_config.report_surface_local_refine)
            or pos_arr.size == 0
            or q_arr.size != pos_arr.shape[0]
            or design_arr.ndim != 2
            or design_arr.shape[1] != pos_arr.shape[0]
        ):
            return pos_arr, q_arr, {"surface_local_refine_enabled": False}
        source_count = int(pos_arr.shape[0])
        max_sources = int(self.pf_config.report_surface_local_refine_max_sources)
        if max_sources > 0 and source_count > max_sources:
            source_order = np.argsort(np.maximum(q_arr, 0.0))[::-1][:max_sources]
        else:
            source_order = np.arange(source_count, dtype=int)
        if source_order.size == 0:
            return pos_arr, q_arr, {"surface_local_refine_enabled": True}
        current_q = self._solve_report_strengths(
            design=design_arr,
            z_obs=data.z_k,
            background=background,
            observation_variances=data.observation_variances,
            initial_strengths=q_arr,
            eps=eps,
            q_max=q_max,
        )
        current_ll = self._report_cluster_log_likelihood(
            filt,
            data,
            design=design_arr,
            strengths=current_q,
            background=background,
        )
        obs_variances = measurement_vector(
            data.observation_variances,
            int(data.z_k.size),
            "observation_variances",
            min_value=1.0,
        )
        weights = 1.0 / obs_variances
        accepted = 0
        tested = 0
        total_gain = 0.0
        min_gain = max(
            0.0,
            float(self.pf_config.report_surface_local_refine_min_ll_gain),
        )
        for source_idx_raw in source_order:
            source_idx = int(source_idx_raw)
            local_positions = self._report_surface_local_candidates(
                filt,
                pos_arr[source_idx],
            )
            if local_positions.shape[0] <= 1:
                continue
            tested += int(local_positions.shape[0])
            local_design = self._cached_expected_counts_per_source(
                filt=filt,
                isotope=isotope,
                data=data,
                sources=local_positions,
                strengths=np.ones(local_positions.shape[0], dtype=float),
            )
            local_design = np.maximum(np.asarray(local_design, dtype=float), 0.0)
            if (
                local_design.ndim != 2
                or local_design.shape[1] != local_positions.shape[0]
            ):
                continue
            other_mask = np.ones(source_count, dtype=bool)
            other_mask[source_idx] = False
            base = np.asarray(background, dtype=float).reshape(-1).copy()
            if np.any(other_mask):
                base = base + design_arr[:, other_mask] @ current_q[other_mask]
            residual = np.asarray(data.z_k, dtype=float).reshape(-1) - base
            numerator = (weights * residual) @ local_design
            denominator = weights @ (local_design * local_design)
            local_q = np.divide(
                numerator,
                np.maximum(denominator, eps),
                out=np.zeros(local_positions.shape[0], dtype=float),
                where=denominator > eps,
            )
            local_q = np.maximum(np.where(np.isfinite(local_q), local_q, 0.0), 0.0)
            if q_max > 0.0:
                local_q = np.minimum(local_q, q_max)
            lam_matrix = np.maximum(
                base[:, None] + local_design * local_q[None, :],
                eps,
            )
            ll_values = filt._count_log_likelihood_matrix_np(
                data.z_k,
                lam_matrix,
                observation_count_variance=data.observation_variances,
            )
            if ll_values.size == 0:
                continue
            best_local = int(np.argmax(ll_values))
            best_ll = float(ll_values[best_local])
            if not np.isfinite(best_ll) or best_ll <= current_ll + min_gain:
                continue
            trial_positions = pos_arr.copy()
            trial_positions[source_idx] = local_positions[best_local]
            trial_design = design_arr.copy()
            trial_design[:, source_idx] = local_design[:, best_local]
            trial_q = self._solve_report_strengths(
                design=trial_design,
                z_obs=data.z_k,
                background=background,
                observation_variances=data.observation_variances,
                initial_strengths=current_q,
                eps=eps,
                q_max=q_max,
            )
            trial_ll = self._report_cluster_log_likelihood(
                filt,
                data,
                design=trial_design,
                strengths=trial_q,
                background=background,
            )
            if not np.isfinite(trial_ll) or trial_ll <= current_ll + min_gain:
                continue
            total_gain += float(trial_ll - current_ll)
            pos_arr = trial_positions
            design_arr = trial_design
            current_q = trial_q
            current_ll = float(trial_ll)
            accepted += 1
        stats = {
            "surface_local_refine_enabled": True,
            "surface_local_refine_sources_considered": int(source_order.size),
            "surface_local_refine_candidates_tested": int(tested),
            "surface_local_refine_accept_count": int(accepted),
            "surface_local_refine_ll_gain": float(total_gain),
        }
        return pos_arr, current_q, stats

    def _refit_reported_strengths(
        self,
        isotope: str,
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Refit reported source strengths with non-negative Poisson regression.

        PF particles estimate source existence and position.  Conditioned on the
        reported positions, the intensity parameters are linear in the expected
        counts, so a multiplicative Poisson regression update gives a
        Rao-Blackwellized strength estimate without changing the transport model
        or inventing source-specific thresholds.
        """
        pos_arr = np.asarray(positions, dtype=float)
        str_arr = np.asarray(strengths, dtype=float).reshape(-1)
        if not bool(self.pf_config.report_strength_refit):
            return pos_arr, str_arr
        rescue_enabled = bool(self.pf_config.report_mle_rescue_enable)
        if (pos_arr.size == 0 or str_arr.size == 0) and not rescue_enabled:
            return pos_arr, str_arr
        if pos_arr.size == 0:
            pos_arr = np.zeros((0, 3), dtype=float)
            str_arr = np.zeros(0, dtype=float)
        if pos_arr.shape[0] != str_arr.size:
            return pos_arr, str_arr
        data = self._measurement_data_for_iso(isotope, None)
        if data is None or data.z_k.size == 0:
            return pos_arr, str_arr
        filt = self.filters.get(isotope)
        if filt is None:
            return pos_arr, str_arr
        if not bool(self.pf_config.report_strength_refit_use_all_measurements):
            refit_data = filt._signal_bearing_refit_data(data)
            if refit_data is None or refit_data.z_k.size == 0:
                return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
            data = refit_data
        z_obs = np.maximum(np.asarray(data.z_k, dtype=float).reshape(-1), 0.0)
        background = self._background_counts_for_report_refit(isotope, data.live_times)
        background = measurement_vector(
            background,
            z_obs.size,
            "background",
            min_value=0.0,
        )
        eps = float(self.pf_config.report_strength_refit_eps)
        q_max = float(getattr(self.pf_config, "birth_q_max", 0.0))
        rescue_stats: dict[str, Any] = {}
        if rescue_enabled:
            pos_arr, str_arr, rescue_stats = (
                self._augment_report_candidates_with_mle_rescue(
                    isotope,
                    filt,
                    data,
                    positions=pos_arr,
                    strengths=str_arr,
                    background=background,
                    eps=eps,
                    q_max=q_max,
                )
            )
            if pos_arr.size == 0 or str_arr.size == 0:
                return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        unit_strengths = np.ones(str_arr.size, dtype=float)
        design = self._cached_expected_counts_per_source(
            filt=filt,
            isotope=isotope,
            data=data,
            sources=pos_arr,
            strengths=unit_strengths,
        )
        design = np.maximum(np.asarray(design, dtype=float), 0.0)
        if design.ndim != 2 or design.shape[1] != str_arr.size:
            return pos_arr, str_arr
        column_sum = np.sum(design, axis=0)
        observable = column_sum > eps
        if not np.any(observable):
            return pos_arr, np.zeros_like(str_arr, dtype=float)
        q = self._solve_report_strengths(
            design=design,
            z_obs=z_obs,
            background=background,
            observation_variances=data.observation_variances,
            initial_strengths=str_arr,
            eps=eps,
            q_max=q_max,
        )
        if bool(self.pf_config.report_surface_local_refine):
            pos_arr, q, refine_stats = self._refine_report_surface_positions(
                isotope,
                filt,
                data,
                positions=pos_arr,
                strengths=q,
                design=design,
                background=background,
                eps=eps,
                q_max=q_max,
            )
            rescue_stats.update(refine_stats)
            str_arr = q.copy()
            unit_strengths = np.ones(str_arr.size, dtype=float)
            design = self._cached_expected_counts_per_source(
                filt=filt,
                isotope=isotope,
                data=data,
                sources=pos_arr,
                strengths=unit_strengths,
            )
            design = np.maximum(np.asarray(design, dtype=float), 0.0)
            if design.ndim != 2 or design.shape[1] != str_arr.size:
                return pos_arr, str_arr
            column_sum = np.sum(design, axis=0)
            observable = column_sum > eps
            if not np.any(observable):
                return pos_arr, np.zeros_like(str_arr, dtype=float)
            q = self._solve_report_strengths(
                design=design,
                z_obs=z_obs,
                background=background,
                observation_variances=data.observation_variances,
                initial_strengths=str_arr,
                eps=eps,
                q_max=q_max,
            )
        q_regression = q.copy()
        selected_by_model, q = self._select_report_clusters_by_model_order(
            isotope,
            filt,
            data,
            design=design,
            strengths=q,
            background=background,
            eps=eps,
            q_max=q_max,
        )
        diagnostics = self._last_report_model_order_diagnostics.get(isotope)
        if rescue_stats and isinstance(diagnostics, dict):
            diagnostics.update(rescue_stats)
        self._update_sparse_poisson_evidence_for_isotope(
            isotope,
            filt,
            data,
            background=background,
        )
        sparse_override = self._sparse_poisson_report_override(
            isotope,
            filt=filt,
            data=data,
            background=background,
        )
        if sparse_override is not None:
            sparse_pos, sparse_q = sparse_override
            pruned_slots = self._apply_sparse_poisson_particle_prune(
                isotope,
                filt,
                data,
                report_positions=pos_arr,
                evidence_positions=sparse_pos,
            )
            diagnostics = self._last_report_model_order_diagnostics.get(isotope)
            if isinstance(diagnostics, dict):
                diagnostics.setdefault(
                    "particle_pruned_source_slots",
                    int(pruned_slots),
                )
            if sparse_pos.size == 0 or sparse_q.size == 0:
                return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
            return sparse_pos, sparse_q
        support_floor = max(float(self.pf_config.min_strength), 0.0) * (1.0 + 1.0e-6)
        preserve_cardinality = bool(
            self.pf_config.report_strength_refit_preserve_cardinality
        )
        if preserve_cardinality:
            original_supported = np.asarray(str_arr, dtype=float) > support_floor
            keep = observable & original_supported
            if int(np.count_nonzero(keep)) > 1:
                active = np.flatnonzero(keep)
                duplicate_tol = max(
                    1.0e-6,
                    1.0e-3 * float(self.pf_config.cluster_eps_m),
                )
                assigned: set[int] = set()
                for idx in active:
                    idx_int = int(idx)
                    if idx_int in assigned:
                        continue
                    distances = np.linalg.norm(
                        pos_arr[active] - pos_arr[idx_int], axis=1
                    )
                    group = active[distances <= duplicate_tol]
                    assigned.update(int(item) for item in group)
                    if group.size <= 1:
                        continue
                    selected_group = group[selected_by_model[group]]
                    survivor = int(
                        selected_group[0] if selected_group.size > 0 else group[0]
                    )
                    for duplicate_idx in group:
                        if int(duplicate_idx) != survivor:
                            keep[int(duplicate_idx)] = False
            q_report = q.copy()
            collapsed = keep & (q_report <= support_floor)
            q_report[collapsed] = q_regression[collapsed]
            still_collapsed = keep & (q_report <= support_floor)
            q_report[still_collapsed] = np.asarray(str_arr, dtype=float)[
                still_collapsed
            ]
            diagnostics = self._last_report_model_order_diagnostics.get(isotope)
            if isinstance(diagnostics, dict):
                raw_model_count = int(np.count_nonzero(selected_by_model))
                reported_count = int(np.count_nonzero(keep))
                raw_model_indices = [
                    int(idx) for idx in np.flatnonzero(selected_by_model)
                ]
                reported_indices = [int(idx) for idx in np.flatnonzero(keep)]
                model_order_overridden = bool(np.any(keep & ~selected_by_model))
                diagnostics["preserve_cardinality"] = True
                diagnostics["model_selected_count"] = raw_model_count
                diagnostics["model_selected_indices"] = raw_model_indices
                diagnostics["selected_count"] = reported_count
                diagnostics["selected_indices"] = reported_indices
                diagnostics["reported_count_after_preserve"] = reported_count
                diagnostics["model_order_overridden"] = model_order_overridden
                if model_order_overridden:
                    diagnostics["model_order_ready"] = False
        else:
            keep = observable & selected_by_model & (q > support_floor)
            q_report = q
            pruned_slots = self._apply_report_model_order_particle_prune(
                isotope,
                filt,
                data,
                pos_arr,
                selected_by_model,
            )
            diagnostics = self._last_report_model_order_diagnostics.get(isotope)
            if isinstance(diagnostics, dict):
                diagnostics.setdefault(
                    "particle_pruned_source_slots", int(pruned_slots)
                )
        if not np.any(keep):
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        return pos_arr[keep], q_report[keep]

    def runtime_report_rescue_modes(
        self,
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64], float]]:
        """Return the latest station-level report-rescue modes for planning."""
        payload = {
            isotope: (positions.copy(), strengths.copy(), float(weight))
            for isotope, (positions, strengths, weight) in (
                self._runtime_report_rescue_modes
            ).items()
        }
        queue_payload = self._candidate_verification_queue_modes()
        for isotope, (queue_pos, queue_q, queue_weight) in queue_payload.items():
            existing = payload.get(str(isotope))
            if existing is None:
                payload[str(isotope)] = (
                    queue_pos.copy(),
                    queue_q.copy(),
                    float(queue_weight),
                )
                continue
            pos_arr, q_arr, weight = existing
            merged_pos = np.vstack(
                [
                    np.asarray(pos_arr, dtype=float).reshape(-1, 3),
                    np.asarray(queue_pos, dtype=float).reshape(-1, 3),
                ]
            )
            merged_q = np.concatenate(
                [
                    np.maximum(np.asarray(q_arr, dtype=float).reshape(-1), 0.0),
                    np.maximum(np.asarray(queue_q, dtype=float).reshape(-1), 0.0),
                ]
            )
            merged_pos, merged_q = dedupe_report_candidates(
                merged_pos,
                merged_q,
                radius_m=max(
                    float(self.pf_config.report_mle_rescue_dedup_radius_m),
                    0.0,
                ),
                max_candidates=max(
                    int(np.asarray(pos_arr, dtype=float).reshape(-1, 3).shape[0]),
                    int(np.asarray(queue_pos, dtype=float).reshape(-1, 3).shape[0]),
                    self._candidate_verification_queue_limit(),
                    1,
                ),
            )
            payload[str(isotope)] = (
                merged_pos.copy(),
                merged_q.copy(),
                float(max(float(weight), float(queue_weight))),
            )
        if not bool(self.pf_config.report_best_so_far_enable):
            return payload
        quarantine_weight = float(
            self.pf_config.runtime_report_rescue_quarantine_weight
        )
        candidate_weight = float(self.pf_config.runtime_report_rescue_candidate_weight)
        best_weight = max(quarantine_weight, candidate_weight)
        max_candidates = max(
            1,
            min(
                int(self.pf_config.report_mle_rescue_max_candidates),
                self._report_max_sources_per_isotope(),
            ),
        )
        radius = max(float(self.pf_config.report_mle_rescue_dedup_radius_m), 0.0)
        for isotope in self.all_isotopes:
            best_pos, best_q = self._best_report_modes_for_isotope(str(isotope))
            if best_pos.size == 0 or best_q.size == 0:
                continue
            existing = payload.get(str(isotope))
            if existing is None:
                payload[str(isotope)] = (
                    best_pos.copy(),
                    best_q.copy(),
                    float(best_weight),
                )
                continue
            pos_arr, q_arr, weight = existing
            merged_pos = np.vstack(
                [
                    np.asarray(pos_arr, dtype=float).reshape(-1, 3),
                    best_pos,
                ]
            )
            merged_q = np.concatenate(
                [
                    np.maximum(np.asarray(q_arr, dtype=float).reshape(-1), 0.0),
                    best_q,
                ]
            )
            deduped_pos, deduped_q = dedupe_report_candidates(
                merged_pos,
                merged_q,
                radius_m=radius,
                max_candidates=max_candidates,
            )
            payload[str(isotope)] = (
                deduped_pos.copy(),
                deduped_q.copy(),
                float(max(float(weight), best_weight)),
            )
        return payload

    def planning_surface_rescue_modes(
        self,
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64], float]]:
        """Return residual-ranked global surface candidates for DSS-PP planning."""
        self._last_planning_surface_rescue_mode_counts = {}
        if not (
            bool(self.pf_config.birth_global_rescue_enable)
            or bool(self.pf_config.report_mle_rescue_enable)
            or bool(self.pf_config.all_history_dictionary_proposal_enable)
        ):
            return {}
        payload: Dict[
            str,
            Tuple[NDArray[np.float64], NDArray[np.float64], float],
        ] = {}
        for isotope, filt in self.filters.items():
            data = self._measurement_data_for_iso(isotope, window=None)
            if data is None or data.z_k.size == 0:
                continue
            positions = self._runtime_global_birth_rescue_candidates(
                isotope,
                filt,
                data,
            )
            if positions.size == 0 and (
                bool(self.pf_config.report_mle_rescue_enable)
                or bool(self.pf_config.all_history_dictionary_proposal_enable)
            ):
                try:
                    existing_positions, _existing_strengths = filt.estimate_clustered(
                        max_k=max(1, self._report_max_sources_per_isotope()),
                        include_report_excluded=True,
                    )
                except (RuntimeError, ValueError, AttributeError):
                    if filt.continuous_particles:
                        best = filt.best_particle().state
                        existing_positions = np.asarray(
                            best.positions[: best.num_sources],
                            dtype=float,
                        )
                    else:
                        existing_positions = np.zeros((0, 3), dtype=float)
                background = self._background_counts_for_report_refit(
                    isotope,
                    data.live_times,
                )
                positions, _q_init, _stats = self._rank_global_surface_candidates(
                    isotope,
                    filt,
                    data,
                    existing_positions=np.asarray(
                        existing_positions,
                        dtype=float,
                    ).reshape(-1, 3),
                    background=background,
                    eps=max(float(self.pf_config.refit_eps), 1.0e-12),
                    q_max=float(self.pf_config.birth_q_max),
                    max_candidates=max(
                        1,
                        int(self.pf_config.birth_global_rescue_max_candidates),
                    ),
                    min_residual_fraction=(
                        self.pf_config.birth_global_rescue_min_residual_fraction
                    ),
                    dedup_radius_m=self.pf_config.birth_global_rescue_dedup_radius_m,
                )
            pos_arr = np.asarray(positions, dtype=float).reshape(-1, 3)
            if pos_arr.size == 0:
                continue
            background = self._background_counts_for_report_refit(
                isotope,
                data.live_times,
            )
            design = self._cached_expected_counts_per_source(
                filt=filt,
                isotope=isotope,
                data=data,
                sources=pos_arr,
                strengths=np.ones(pos_arr.shape[0], dtype=float),
            )
            q_arr = self._solve_report_strengths(
                design=design,
                z_obs=data.z_k,
                background=background,
                observation_variances=data.observation_variances,
                initial_strengths=np.full(
                    pos_arr.shape[0],
                    max(float(self.pf_config.birth_q_min), 1.0),
                    dtype=float,
                ),
                eps=max(float(self.pf_config.refit_eps), 1.0e-12),
                q_max=float(self.pf_config.birth_q_max),
            )
            q_arr = np.maximum(np.asarray(q_arr, dtype=float).reshape(-1), 0.0)
            valid = (
                np.isfinite(pos_arr).all(axis=1)
                & np.isfinite(q_arr)
                & (q_arr > max(float(self.pf_config.min_strength), 0.0))
            )
            if not np.any(valid):
                continue
            self._last_planning_surface_rescue_mode_counts[str(isotope)] = int(
                np.count_nonzero(valid)
            )
            payload[str(isotope)] = (
                pos_arr[valid].copy(),
                q_arr[valid].copy(),
                float(self.pf_config.runtime_report_rescue_quarantine_weight),
            )
        return payload

    def _all_history_dictionary_candidate_limit(self) -> int:
        """Return the all-history dictionary proposal source limit."""
        configured = max(
            0,
            int(self.pf_config.all_history_dictionary_proposal_max_candidates),
        )
        if configured > 0:
            return configured
        return max(
            1,
            min(
                int(self.pf_config.report_mle_rescue_max_candidates),
                self._report_max_sources_per_isotope(),
            ),
        )

    def _all_history_dictionary_candidates(
        self,
        isotope: str,
        filt: IsotopeParticleFilter,
        *,
        existing_positions: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return surface dictionary candidates scored against all measurements."""
        if (
            not bool(self.pf_config.all_history_dictionary_proposal_enable)
            or not self.measurements
        ):
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        data = self._measurement_data_for_iso(isotope, window=None)
        if data is None or data.z_k.size == 0 or self.candidate_sources.size == 0:
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        background = self._background_counts_for_report_refit(isotope, data.live_times)
        pos_arr, q_arr, _stats = self._rank_global_surface_candidates(
            isotope,
            filt,
            data,
            existing_positions=np.asarray(existing_positions, dtype=float).reshape(
                -1, 3
            ),
            background=background,
            eps=max(float(self.pf_config.report_strength_refit_eps), 1.0e-12),
            q_max=float(getattr(self.pf_config, "birth_q_max", 0.0)),
            max_candidates=self._all_history_dictionary_candidate_limit(),
            min_residual_fraction=0.0,
            dedup_radius_m=self.pf_config.birth_global_rescue_dedup_radius_m,
        )
        if pos_arr.size == 0 or q_arr.size == 0:
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        return dedupe_report_candidates(
            np.asarray(pos_arr, dtype=float).reshape(-1, 3),
            np.asarray(q_arr, dtype=float).reshape(-1),
            radius_m=max(float(self.pf_config.birth_global_rescue_dedup_radius_m), 0.0),
            max_candidates=self._all_history_dictionary_candidate_limit(),
        )

    def _runtime_report_rescue_estimate(
        self,
        isotope: str,
        filt: IsotopeParticleFilter,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return a report-rescue model using all measurements collected so far."""
        if (
            not bool(self.pf_config.runtime_report_rescue_enable)
            or not bool(self.pf_config.report_strength_refit)
            or not bool(self.pf_config.report_mle_rescue_enable)
            or not self.measurements
        ):
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        max_k = max(
            1,
            min(
                int(self.pf_config.report_mle_rescue_max_candidates),
                self._report_max_sources_per_isotope(),
            ),
        )
        try:
            base_pos, base_q = filt.estimate_clustered(
                max_k=max_k,
                include_report_excluded=True,
            )
        except (RuntimeError, ValueError, AttributeError):
            if not filt.continuous_particles:
                base_pos = np.zeros((0, 3), dtype=float)
                base_q = np.zeros(0, dtype=float)
            else:
                best = filt.best_particle().state
                count = max(0, int(best.num_sources))
                base_pos = np.asarray(best.positions[:count], dtype=float).reshape(
                    -1, 3
                )
                base_q = np.asarray(best.strengths[:count], dtype=float).reshape(-1)
        prune_enabled = bool(self.pf_config.report_model_order_prune_particles)
        self.pf_config.report_model_order_prune_particles = False
        try:
            pos_arr, q_arr = self._refit_reported_strengths(
                isotope,
                np.asarray(base_pos, dtype=float).reshape(-1, 3),
                np.asarray(base_q, dtype=float).reshape(-1),
            )
        finally:
            self.pf_config.report_model_order_prune_particles = prune_enabled
        if pos_arr.size == 0 or q_arr.size == 0:
            if bool(self.pf_config.all_history_dictionary_proposal_enable):
                dict_pos, dict_q = self._all_history_dictionary_candidates(
                    isotope,
                    filt,
                    existing_positions=np.asarray(base_pos, dtype=float).reshape(-1, 3),
                )
                if dict_pos.size and dict_q.size:
                    return dict_pos, dict_q
            return self._runtime_unresolved_surface_rescue_estimate(isotope, filt)
        if bool(self.pf_config.all_history_dictionary_proposal_enable):
            dict_pos, dict_q = self._all_history_dictionary_candidates(
                isotope,
                filt,
                existing_positions=np.asarray(pos_arr, dtype=float).reshape(-1, 3),
            )
            if dict_pos.size and dict_q.size:
                pos_arr, q_arr = dedupe_report_candidates(
                    np.vstack(
                        [
                            np.asarray(pos_arr, dtype=float).reshape(-1, 3),
                            np.asarray(dict_pos, dtype=float).reshape(-1, 3),
                        ]
                    ),
                    np.concatenate(
                        [
                            np.maximum(
                                np.asarray(q_arr, dtype=float).reshape(-1),
                                0.0,
                            ),
                            np.maximum(
                                np.asarray(dict_q, dtype=float).reshape(-1),
                                0.0,
                            ),
                        ]
                    ),
                    radius_m=max(
                        float(self.pf_config.birth_global_rescue_dedup_radius_m),
                        0.0,
                    ),
                    max_candidates=max(
                        self._all_history_dictionary_candidate_limit(),
                        max_k,
                    ),
                )
        return (
            np.asarray(pos_arr, dtype=float).reshape(-1, 3),
            np.asarray(q_arr, dtype=float).reshape(-1),
        )

    def _runtime_unresolved_surface_rescue_estimate(
        self,
        isotope: str,
        filt: IsotopeParticleFilter,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return quarantined surface-grid rescue modes for count-supported isotopes."""
        unresolved = self.unresolved_structural_evidence().get(str(isotope), {})
        rescue_reasons = {
            "isotope_absence",
            "report_underfit",
            "birth_residual",
            "runtime_rescue_unverified",
        }
        if not any(reason in unresolved for reason in rescue_reasons):
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        data = self._measurement_data_for_iso(isotope, window=None)
        if data is None or data.z_k.size == 0 or self.candidate_sources.size == 0:
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        try:
            existing_positions, _existing_strengths = filt.estimate_clustered(
                max_k=max(1, self._report_max_sources_per_isotope()),
                include_report_excluded=True,
            )
        except (RuntimeError, ValueError, AttributeError):
            if filt.continuous_particles:
                state = filt.best_particle().state
                existing_positions = np.asarray(
                    state.positions[: state.num_sources],
                    dtype=float,
                )
            else:
                existing_positions = np.zeros((0, 3), dtype=float)
        existing_positions = np.asarray(existing_positions, dtype=float).reshape(-1, 3)
        background = self._background_counts_for_report_refit(isotope, data.live_times)
        eps = max(float(self.pf_config.report_strength_refit_eps), 1.0e-12)
        q_max = float(getattr(self.pf_config, "birth_q_max", 0.0))
        max_candidates = max(
            1,
            min(
                int(self.pf_config.report_mle_rescue_max_candidates),
                self._report_max_sources_per_isotope(),
            ),
        )
        global_pos, global_q, _global_stats = self._rank_global_surface_candidates(
            isotope,
            filt,
            data,
            existing_positions=existing_positions,
            background=background,
            eps=eps,
            q_max=q_max,
            max_candidates=max_candidates,
            min_residual_fraction=0.0,
            dedup_radius_m=self.pf_config.birth_global_rescue_dedup_radius_m,
        )
        if global_pos.size and global_q.size:
            return dedupe_report_candidates(
                np.asarray(global_pos, dtype=float).reshape(-1, 3),
                np.asarray(global_q, dtype=float).reshape(-1),
                radius_m=self.pf_config.birth_global_rescue_dedup_radius_m,
                max_candidates=max_candidates,
            )
        residual = np.maximum(
            np.asarray(data.z_k, dtype=float).reshape(-1)
            - np.asarray(background, dtype=float).reshape(-1),
            0.0,
        )
        residual_pos, residual_q, _residual_stats = (
            self._rank_residual_surface_candidates(
                isotope,
                filt,
                data,
                residual=residual,
                existing_positions=existing_positions,
                background=background,
                eps=eps,
                max_candidates=max_candidates,
                min_residual_fraction=0.0,
                dedup_radius_m=self.pf_config.birth_global_rescue_dedup_radius_m,
            )
        )
        if residual_pos.size == 0 or residual_q.size == 0:
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        return dedupe_report_candidates(
            np.asarray(residual_pos, dtype=float).reshape(-1, 3),
            np.asarray(residual_q, dtype=float).reshape(-1),
            radius_m=self.pf_config.birth_global_rescue_dedup_radius_m,
            max_candidates=max_candidates,
        )

    def _runtime_report_rescue_memory_limit(self) -> int:
        """Return the per-isotope rescue-memory source limit."""
        configured = max(
            0, int(self.pf_config.runtime_report_rescue_memory_max_sources)
        )
        if configured > 0:
            return configured
        return max(
            1,
            min(
                int(self.pf_config.report_mle_rescue_max_candidates),
                self._report_max_sources_per_isotope(),
            ),
        )

    def _candidate_verification_queue_limit(self) -> int:
        """Return the per-isotope candidate-verification queue limit."""
        configured = max(
            0,
            int(self.pf_config.candidate_verification_queue_max_sources),
        )
        if configured > 0:
            return configured
        return max(
            1,
            min(
                int(self.pf_config.report_mle_rescue_max_candidates),
                self._report_max_sources_per_isotope(),
            ),
        )

    def _candidate_verification_queue_should_track(
        self,
        isotope: str,
        injection_weight: float,
    ) -> bool:
        """Return True when current candidates should stay in the verification queue."""
        if not bool(self.pf_config.candidate_verification_queue_enable):
            return False
        unresolved = self.unresolved_structural_evidence().get(str(isotope), {})
        if unresolved:
            return True
        full_weight = float(self.pf_config.runtime_report_rescue_weight)
        return float(injection_weight) < full_weight - 1.0e-12

    def _merge_candidate_verification_queue(
        self,
        isotope: str,
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Merge new candidates into the decayed verification queue."""
        pos_arr = np.asarray(positions, dtype=float).reshape(-1, 3)
        q_arr = np.maximum(np.asarray(strengths, dtype=float).reshape(-1), 0.0)
        if pos_arr.shape[0] != q_arr.size:
            pos_arr = np.zeros((0, 3), dtype=float)
            q_arr = np.zeros(0, dtype=float)
        if not bool(self.pf_config.candidate_verification_queue_enable):
            self._candidate_verification_queue.pop(isotope, None)
            return pos_arr, q_arr

        memory = self._candidate_verification_queue.get(isotope)
        decay = float(self.pf_config.candidate_verification_queue_decay)
        if memory is None:
            mem_pos = np.zeros((0, 3), dtype=float)
            mem_q = np.zeros(0, dtype=float)
            mem_score = np.zeros(0, dtype=float)
        else:
            mem_pos, mem_q, mem_score = memory
            mem_pos = np.asarray(mem_pos, dtype=float).reshape(-1, 3)
            mem_q = np.maximum(np.asarray(mem_q, dtype=float).reshape(-1), 0.0)
            mem_score = (
                np.maximum(np.asarray(mem_score, dtype=float).reshape(-1), 0.0) * decay
            )
        valid_mem = (
            np.isfinite(mem_pos).all(axis=1)
            & np.isfinite(mem_q)
            & np.isfinite(mem_score)
            & (mem_q > max(float(self.pf_config.min_strength), 0.0))
            & (mem_score > 0.0)
        )
        mem_pos = mem_pos[valid_mem]
        mem_q = mem_q[valid_mem]
        mem_score = mem_score[valid_mem]
        valid_current = (
            np.isfinite(pos_arr).all(axis=1)
            & np.isfinite(q_arr)
            & (q_arr > max(float(self.pf_config.min_strength), 0.0))
        )
        pos_arr = pos_arr[valid_current]
        q_arr = q_arr[valid_current]
        current_score = q_arr.copy()
        if mem_pos.size and pos_arr.size:
            merged_pos = np.vstack([pos_arr, mem_pos])
            merged_q = np.concatenate([q_arr, mem_q])
            merged_score = np.concatenate([current_score, mem_score])
        elif pos_arr.size:
            merged_pos = pos_arr
            merged_q = q_arr
            merged_score = current_score
        elif mem_pos.size:
            merged_pos = mem_pos
            merged_q = mem_q
            merged_score = mem_score
        else:
            self._candidate_verification_queue.pop(isotope, None)
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)

        order = np.argsort(merged_score)[::-1]
        radius = max(float(self.pf_config.report_mle_rescue_dedup_radius_m), 0.0)
        limit = self._candidate_verification_queue_limit()
        kept_pos: list[NDArray[np.float64]] = []
        kept_q: list[float] = []
        kept_score: list[float] = []
        for idx in order:
            if len(kept_pos) >= limit:
                break
            pos = merged_pos[int(idx)]
            if radius > 0.0 and kept_pos:
                distances = np.linalg.norm(np.vstack(kept_pos) - pos[None, :], axis=1)
                if np.any(distances <= radius):
                    continue
            kept_pos.append(np.asarray(pos, dtype=float))
            kept_q.append(float(merged_q[int(idx)]))
            kept_score.append(float(merged_score[int(idx)]))
        if not kept_pos:
            self._candidate_verification_queue.pop(isotope, None)
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        final_pos = np.vstack(kept_pos)
        final_q = np.asarray(kept_q, dtype=float)
        final_score = np.asarray(kept_score, dtype=float)
        self._candidate_verification_queue[isotope] = (
            final_pos.copy(),
            final_q.copy(),
            final_score.copy(),
        )
        return final_pos, final_q

    def _candidate_verification_queue_modes(
        self,
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64], float]]:
        """Return queued candidates as low-mass planning modes."""
        if not bool(self.pf_config.candidate_verification_queue_enable):
            return {}
        payload: Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64], float]] = {}
        for isotope, memory in self._candidate_verification_queue.items():
            pos_arr, q_arr, score_arr = memory
            pos_arr = np.asarray(pos_arr, dtype=float).reshape(-1, 3)
            q_arr = np.maximum(np.asarray(q_arr, dtype=float).reshape(-1), 0.0)
            score_arr = np.maximum(np.asarray(score_arr, dtype=float).reshape(-1), 0.0)
            valid = (
                np.isfinite(pos_arr).all(axis=1)
                & np.isfinite(q_arr)
                & np.isfinite(score_arr)
                & (q_arr > max(float(self.pf_config.min_strength), 0.0))
                & (score_arr > 0.0)
            )
            if not np.any(valid):
                continue
            payload[str(isotope)] = (
                pos_arr[valid].copy(),
                q_arr[valid].copy(),
                float(self.pf_config.candidate_verification_queue_weight),
            )
        return payload

    def _merge_runtime_report_rescue_memory(
        self,
        isotope: str,
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Merge current station rescue modes with decayed rescue memory."""
        pos_arr = np.asarray(positions, dtype=float).reshape(-1, 3)
        q_arr = np.maximum(np.asarray(strengths, dtype=float).reshape(-1), 0.0)
        if pos_arr.shape[0] != q_arr.size:
            pos_arr = np.zeros((0, 3), dtype=float)
            q_arr = np.zeros(0, dtype=float)
        if not bool(self.pf_config.runtime_report_rescue_memory_enable):
            if pos_arr.size == 0:
                self._runtime_report_rescue_memory.pop(isotope, None)
            return pos_arr, q_arr

        memory = self._runtime_report_rescue_memory.get(isotope)
        decay = float(self.pf_config.runtime_report_rescue_memory_decay)
        if memory is None:
            mem_pos = np.zeros((0, 3), dtype=float)
            mem_q = np.zeros(0, dtype=float)
            mem_score = np.zeros(0, dtype=float)
        else:
            mem_pos, mem_q, mem_score = memory
            mem_pos = np.asarray(mem_pos, dtype=float).reshape(-1, 3)
            mem_q = np.maximum(np.asarray(mem_q, dtype=float).reshape(-1), 0.0)
            mem_score = (
                np.maximum(np.asarray(mem_score, dtype=float).reshape(-1), 0.0) * decay
            )
            valid_mem = (
                np.isfinite(mem_pos).all(axis=1)
                & np.isfinite(mem_q)
                & np.isfinite(mem_score)
                & (mem_q > 0.0)
                & (mem_score > max(float(self.pf_config.min_strength), 0.0))
            )
            mem_pos = mem_pos[valid_mem]
            mem_q = mem_q[valid_mem]
            mem_score = mem_score[valid_mem]

        valid_current = (
            np.isfinite(pos_arr).all(axis=1)
            & np.isfinite(q_arr)
            & (q_arr > max(float(self.pf_config.min_strength), 0.0))
        )
        pos_arr = pos_arr[valid_current]
        q_arr = q_arr[valid_current]
        current_score = q_arr.copy()
        if mem_pos.size and pos_arr.size:
            merged_pos = np.vstack([pos_arr, mem_pos])
            merged_q = np.concatenate([q_arr, mem_q])
            merged_score = np.concatenate([current_score, mem_score])
        elif pos_arr.size:
            merged_pos = pos_arr
            merged_q = q_arr
            merged_score = current_score
        elif mem_pos.size:
            merged_pos = mem_pos
            merged_q = mem_q
            merged_score = mem_score
        else:
            self._runtime_report_rescue_memory.pop(isotope, None)
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)

        order = np.argsort(merged_score)[::-1]
        radius = max(float(self.pf_config.report_mle_rescue_dedup_radius_m), 0.0)
        limit = self._runtime_report_rescue_memory_limit()
        kept_pos: list[NDArray[np.float64]] = []
        kept_q: list[float] = []
        kept_score: list[float] = []
        for idx in order:
            if len(kept_pos) >= limit:
                break
            pos = merged_pos[int(idx)]
            if radius > 0.0 and kept_pos:
                distances = np.linalg.norm(np.vstack(kept_pos) - pos[None, :], axis=1)
                if np.any(distances <= radius):
                    continue
            kept_pos.append(np.asarray(pos, dtype=float))
            kept_q.append(float(merged_q[int(idx)]))
            kept_score.append(float(merged_score[int(idx)]))
        if not kept_pos:
            self._runtime_report_rescue_memory.pop(isotope, None)
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        final_pos = np.vstack(kept_pos)
        final_q = np.asarray(kept_q, dtype=float)
        final_score = np.asarray(kept_score, dtype=float)
        self._runtime_report_rescue_memory[isotope] = (
            final_pos.copy(),
            final_q.copy(),
            final_score.copy(),
        )
        return final_pos, final_q

    def _runtime_report_rescue_injection_weight(
        self,
        isotope: str,
        source_count: int,
    ) -> float:
        """Return the PF mass assigned to a runtime rescue hypothesis."""
        full_weight = float(self.pf_config.runtime_report_rescue_weight)
        candidate_weight = float(self.pf_config.runtime_report_rescue_candidate_weight)
        if not bool(self.pf_config.runtime_report_rescue_quarantine_enable):
            return full_weight
        quarantine_weight = float(
            self.pf_config.runtime_report_rescue_quarantine_weight
        )
        diagnostics = self._last_report_model_order_diagnostics.get(str(isotope), {})
        selected_count = int(diagnostics.get("selected_count", source_count))
        candidate_count = int(diagnostics.get("candidate_count", source_count))
        ready = bool(diagnostics.get("model_order_ready", False))
        method = str(diagnostics.get("method", ""))
        min_margin = max(float(self.pf_config.report_model_order_min_bic_margin), 0.0)
        runner_up_margin = float(
            diagnostics.get("criterion_margin_to_runner_up", np.inf)
        )
        simpler_margin = float(diagnostics.get("criterion_margin_to_simpler", np.inf))
        margin_ok = (
            min_margin <= 0.0
            or not np.isfinite(runner_up_margin)
            or runner_up_margin >= min_margin
        )
        if selected_count > 1 and np.isfinite(simpler_margin):
            margin_ok = bool(margin_ok and simpler_margin >= min_margin)
        unresolved = self.unresolved_structural_evidence().get(str(isotope), {})
        if (
            ready
            and not unresolved
            and selected_count == int(source_count)
            and candidate_count >= selected_count
        ):
            return full_weight
        if (
            selected_count > 0
            and selected_count == int(source_count)
            and candidate_count >= selected_count
            and method not in {"", "empty", "source_cap_zero"}
            and margin_ok
        ):
            return float(max(quarantine_weight, min(candidate_weight, full_weight)))
        return quarantine_weight

    def _runtime_report_rescue_injection_fraction(
        self, injection_weight: float
    ) -> float:
        """Return the PF particle fraction used for report-rescue injection."""
        base_fraction = float(self.pf_config.runtime_report_rescue_particle_fraction)
        full_weight = max(float(self.pf_config.runtime_report_rescue_weight), 1.0e-12)
        weight = max(float(injection_weight), 0.0)
        if weight + 1.0e-12 >= full_weight:
            return base_fraction
        scaled = base_fraction * min(max(weight / full_weight, 0.0), 1.0)
        return float(min(base_fraction, max(scaled, 0.0)))

    def _sync_report_cardinality_protection(
        self,
        isotope: str,
        filt: IsotopeParticleFilter,
    ) -> None:
        """Expose report/BIC-supported source counts to PF resampling."""
        setter = getattr(filt, "set_external_protected_cardinalities", None)
        if not callable(setter):
            return
        if not bool(self.pf_config.mode_preserving_report_cardinality_strata):
            setter(set())
            return
        diagnostics = self._last_report_model_order_diagnostics.get(str(isotope), {})
        if not isinstance(diagnostics, dict):
            setter(set())
            return
        method = str(diagnostics.get("method", ""))
        if method in {"", "empty", "source_cap_zero"}:
            setter(set())
            return
        candidate_count = int(diagnostics.get("candidate_count", 0))
        selected_count = int(diagnostics.get("selected_count", 0))
        selected_counts: set[int] = set()
        if candidate_count > 0 and selected_count >= 0:
            selected_counts.add(selected_count)
        model_selected = diagnostics.get("model_selected_count")
        if model_selected is not None:
            try:
                model_count = int(model_selected)
            except (TypeError, ValueError):
                model_count = -1
            if model_count >= 0:
                selected_counts.add(model_count)
        setter(selected_counts)

    def _inject_runtime_report_rescue(
        self,
        isotope: str,
        filt: IsotopeParticleFilter,
    ) -> int:
        """Inject station-level report-rescue modes into PF particles."""
        positions, strengths = self._runtime_report_rescue_estimate(isotope, filt)
        self._sync_report_cardinality_protection(isotope, filt)
        best_positions, best_strengths = self._best_report_modes_for_isotope(isotope)
        if best_positions.size and best_strengths.size:
            if positions.size and strengths.size:
                merged_positions = np.vstack(
                    [
                        np.asarray(positions, dtype=float).reshape(-1, 3),
                        best_positions,
                    ]
                )
                merged_strengths = np.concatenate(
                    [
                        np.maximum(
                            np.asarray(strengths, dtype=float).reshape(-1),
                            0.0,
                        ),
                        best_strengths,
                    ]
                )
            else:
                merged_positions = best_positions
                merged_strengths = best_strengths
            positions, strengths = dedupe_report_candidates(
                merged_positions,
                merged_strengths,
                radius_m=max(
                    float(self.pf_config.report_mle_rescue_dedup_radius_m),
                    0.0,
                ),
                max_candidates=max(
                    1,
                    min(
                        int(self.pf_config.report_mle_rescue_max_candidates),
                        self._report_max_sources_per_isotope(),
                    ),
                ),
            )
        positions, strengths = self._merge_runtime_report_rescue_memory(
            isotope,
            positions,
            strengths,
        )
        if positions.size == 0 or strengths.size == 0:
            self._runtime_report_rescue_modes.pop(isotope, None)
            return 0
        injection_weight = self._runtime_report_rescue_injection_weight(
            isotope,
            int(np.asarray(positions, dtype=float).reshape(-1, 3).shape[0]),
        )
        if bool(self.pf_config.runtime_report_rescue_verification_queue_only):
            self._merge_candidate_verification_queue(isotope, positions, strengths)
            self._runtime_report_rescue_modes[isotope] = (
                positions.copy(),
                strengths.copy(),
                0.0,
            )
            return 0
        if self._candidate_verification_queue_should_track(isotope, injection_weight):
            self._merge_candidate_verification_queue(isotope, positions, strengths)
        elif bool(self.pf_config.candidate_verification_queue_enable):
            self._candidate_verification_queue.pop(isotope, None)
        self._runtime_report_rescue_modes[isotope] = (
            positions.copy(),
            strengths.copy(),
            float(max(injection_weight, 0.0)),
        )
        if injection_weight <= 0.0:
            return 0
        particle_fraction = self._runtime_report_rescue_injection_fraction(
            injection_weight
        )
        full_weight = float(self.pf_config.runtime_report_rescue_weight)
        combine_sources = injection_weight + 1.0e-12 >= full_weight
        injected = filt.inject_runtime_report_rescue_particles(
            positions,
            strengths,
            particle_fraction=particle_fraction,
            min_particles_per_source=(
                self.pf_config.runtime_report_rescue_min_particles_per_source
            ),
            total_weight=injection_weight,
            jitter_sigma_m=self.pf_config.runtime_report_rescue_jitter_sigma_m,
            combine_sources=combine_sources,
        )
        if injected > 0:
            self._runtime_report_rescue_modes[isotope] = (
                positions.copy(),
                strengths.copy(),
                float(injection_weight),
            )
            self._invalidate_report_cache()
        else:
            self._runtime_report_rescue_modes[isotope] = (
                positions.copy(),
                strengths.copy(),
                float(max(injection_weight, 0.0)),
            )
        return int(injected)

    def _global_birth_candidate_memory_limit(self) -> int:
        """Return the residual surface-candidate memory limit."""
        configured = max(
            0,
            int(self.pf_config.birth_global_rescue_candidate_memory_max_candidates),
        )
        if configured > 0:
            return configured
        return max(1, int(self.pf_config.birth_global_rescue_max_candidates))

    def _global_birth_candidate_memory_min_retained(self) -> int:
        """Return the number of residual candidates reserved from memory."""
        limit = self._global_birth_candidate_memory_limit()
        configured = max(
            0,
            int(self.pf_config.birth_global_rescue_candidate_memory_min_retained),
        )
        if configured > 0:
            return min(configured, limit)
        return min(max(1, self._report_max_sources_per_isotope()), limit)

    def _merge_global_birth_candidate_memory(
        self,
        isotope: str,
        positions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Merge ranked global residual candidates with decayed station memory."""
        pos_arr = np.asarray(positions, dtype=float).reshape(-1, 3)
        if not bool(self.pf_config.birth_global_rescue_candidate_memory_enable):
            if pos_arr.size == 0:
                self._global_birth_rescue_candidate_memory.pop(isotope, None)
            return pos_arr
        memory = self._global_birth_rescue_candidate_memory.get(str(isotope))
        if memory is None:
            mem_pos = np.zeros((0, 3), dtype=float)
            mem_score = np.zeros(0, dtype=float)
        else:
            mem_pos, mem_score = memory
            mem_pos = np.asarray(mem_pos, dtype=float).reshape(-1, 3)
            mem_score = np.maximum(
                np.asarray(mem_score, dtype=float).reshape(-1), 0.0
            ) * float(self.pf_config.birth_global_rescue_candidate_memory_decay)
            valid = (
                np.isfinite(mem_pos).all(axis=1)
                & np.isfinite(mem_score)
                & (mem_score > 1.0e-12)
            )
            mem_pos = mem_pos[valid]
            mem_score = mem_score[valid]
        valid_current = np.isfinite(pos_arr).all(axis=1)
        pos_arr = pos_arr[valid_current]
        if pos_arr.size:
            rank = np.arange(pos_arr.shape[0], dtype=float)
            current_score = 1.0 / (1.0 + rank)
            merged_pos = np.vstack([pos_arr, mem_pos]) if mem_pos.size else pos_arr
            merged_score = (
                np.concatenate([current_score, mem_score])
                if mem_score.size
                else current_score
            )
        elif mem_pos.size:
            merged_pos = mem_pos
            merged_score = mem_score
        else:
            self._global_birth_rescue_candidate_memory.pop(str(isotope), None)
            return np.zeros((0, 3), dtype=float)

        order = np.argsort(merged_score)[::-1]
        radius = max(float(self.pf_config.birth_global_rescue_dedup_radius_m), 0.0)
        limit = self._global_birth_candidate_memory_limit()
        kept_pos: list[NDArray[np.float64]] = []
        kept_score: list[float] = []

        memory_reserve = self._global_birth_candidate_memory_min_retained()
        if mem_pos.size and memory_reserve > 0:
            mem_order = np.argsort(mem_score)[::-1]
            for idx in mem_order:
                if len(kept_pos) >= min(memory_reserve, limit):
                    break
                pos = np.asarray(mem_pos[int(idx)], dtype=float)
                if radius > 0.0 and kept_pos:
                    distances = np.linalg.norm(
                        np.vstack(kept_pos) - pos[None, :],
                        axis=1,
                    )
                    if np.any(distances <= radius):
                        continue
                kept_pos.append(pos.copy())
                kept_score.append(float(mem_score[int(idx)]))

        for idx in order:
            if len(kept_pos) >= limit:
                break
            pos = np.asarray(merged_pos[int(idx)], dtype=float)
            if radius > 0.0 and kept_pos:
                distances = np.linalg.norm(np.vstack(kept_pos) - pos[None, :], axis=1)
                if np.any(distances <= radius):
                    continue
            kept_pos.append(pos.copy())
            kept_score.append(float(merged_score[int(idx)]))
        if not kept_pos:
            self._global_birth_rescue_candidate_memory.pop(str(isotope), None)
            return np.zeros((0, 3), dtype=float)
        final_pos = np.vstack(kept_pos)
        final_score = np.asarray(kept_score, dtype=float)
        self._global_birth_rescue_candidate_memory[str(isotope)] = (
            final_pos.copy(),
            final_score.copy(),
        )
        return final_pos

    def _global_birth_candidate_counts_for_update(
        self,
        isotope: str,
        filt: IsotopeParticleFilter,
        data: MeasurementData | None,
        candidates: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
        """Return detector-filtered global birth candidates and response counts."""
        candidate_arr = np.asarray(candidates, dtype=float).reshape(-1, 3)
        if data is None or data.z_k.size == 0 or candidate_arr.size == 0:
            return candidate_arr, None
        filtered = filt._exclude_birth_candidates_near_detectors(
            candidate_arr,
            data,
        )
        if filtered.size == 0:
            return np.zeros((0, 3), dtype=float), None
        counts = self._cached_expected_counts_per_source(
            filt=filt,
            isotope=isotope,
            data=data,
            sources=filtered,
            strengths=np.ones(filtered.shape[0], dtype=float),
        )
        return filtered, np.asarray(counts, dtype=float)

    def _authoritative_sparse_evidence_cardinality(
        self,
        isotope: str,
    ) -> tuple[bool, int]:
        """Return whether authoritative sparse evidence has a ready cardinality."""
        if not bool(self.pf_config.sparse_poisson_evidence_authoritative):
            return False, -1
        payload = self._last_sparse_poisson_evidence_diagnostics.get(str(isotope))
        if not isinstance(payload, dict):
            return False, -1
        if not bool(payload.get("available", False)):
            return False, -1
        if not bool(payload.get("model_order_ready", False)):
            return False, -1
        try:
            selected_count = int(payload.get("selected_count", 0))
        except (TypeError, ValueError):
            return False, -1
        return True, max(0, selected_count)

    def _sync_sparse_evidence_cardinality_protection(
        self,
        isotope: str,
        filt: IsotopeParticleFilter,
    ) -> tuple[bool, int]:
        """Expose authoritative sparse-evidence cardinality to PF resampling."""
        ready, selected_count = self._authoritative_sparse_evidence_cardinality(
            isotope
        )
        setter = getattr(filt, "set_external_protected_cardinalities", None)
        if callable(setter):
            setter({selected_count} if ready else set())
        return ready, selected_count

    def _authoritative_sparse_evidence_sources(
        self,
        isotope: str,
        *,
        filt: IsotopeParticleFilter | None = None,
        data: MeasurementData | None = None,
    ) -> tuple[bool, NDArray[np.float64], NDArray[np.float64]]:
        """Return evidence-selected source positions and strengths for PF sync."""
        ready, selected_count = self._authoritative_sparse_evidence_cardinality(
            isotope
        )
        if not ready:
            return False, np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        payload = self._last_sparse_poisson_evidence_diagnostics.get(str(isotope))
        if not isinstance(payload, Mapping):
            return False, np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        positions = np.asarray(
            payload.get("selected_positions", []),
            dtype=float,
        ).reshape(-1, 3)
        strengths = np.maximum(
            np.asarray(payload.get("selected_strengths", []), dtype=float).reshape(-1),
            0.0,
        )
        if selected_count == 0:
            return True, np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        if positions.shape[0] != selected_count or strengths.size != selected_count:
            if positions.shape[0] != selected_count:
                return False, np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
            strengths = np.zeros(selected_count, dtype=float)
        strengths = self._refit_sparse_evidence_strengths(
            isotope,
            filt,
            data,
            positions,
            strengths,
        )
        if strengths.size != selected_count:
            return False, np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        finite = np.isfinite(positions).all(axis=1) & np.isfinite(strengths)
        if not bool(np.all(finite)):
            return False, np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        return True, positions.copy(), strengths.copy()

    def _run_isotope_structural_update(
        self,
        task: tuple[
            str,
            IsotopeParticleFilter,
            MeasurementData | None,
            MeasurementData | None,
            MeasurementData | None,
        ],
    ) -> None:
        """Run one isotope's deferred strength refit and birth/death update."""
        isotope, filt, refit_data, support_data, birth_data = task
        evidence_ready, evidence_count = (
            self._sync_sparse_evidence_cardinality_protection(isotope, filt)
        )
        if evidence_ready:
            sync_data = refit_data if refit_data is not None else support_data
            if sync_data is None:
                sync_data = birth_data
            sync_ready, evidence_positions, evidence_strengths = (
                self._authoritative_sparse_evidence_sources(
                    isotope,
                    filt=filt,
                    data=sync_data,
                )
            )
            sync_method = getattr(filt, "sync_particles_to_evidence_sources", None)
            if sync_ready and callable(sync_method):
                sync_method(
                    evidence_positions,
                    evidence_strengths,
                    data=sync_data,
                )
                timing = getattr(filt, "last_structural_timing_s", {})
                if isinstance(timing, dict):
                    timing["sparse_evidence_cardinality_ready"] = 1.0
                    timing["sparse_evidence_selected_count"] = float(evidence_count)
                    timing["birth_proposals_gated"] = 1.0
                    timing["structural_moves_skipped_by_sparse_evidence"] = 1.0
                return
        if bool(self.pf_config.conditional_strength_refit):
            filt.refit_strengths_for_particles(
                refit_data,
                iters=self.pf_config.conditional_strength_refit_iters,
                eps=self.pf_config.refit_eps,
                suppress_prune_after_refit=bool(
                    self.pf_config.birth_residual_suppress_death
                ),
            )
        proposal_data = birth_data if birth_data is not None else support_data
        if evidence_ready:
            global_birth_candidates = np.zeros((0, 3), dtype=float)
        else:
            global_birth_candidates = self._runtime_global_birth_rescue_candidates(
                isotope,
                filt,
                proposal_data,
            )
        global_birth_candidate_counts = None
        if global_birth_candidates.size:
            (
                global_birth_candidates,
                global_birth_candidate_counts,
            ) = self._global_birth_candidate_counts_for_update(
                isotope,
                filt,
                proposal_data,
                global_birth_candidates,
            )
        filt.apply_birth_death(
            support_data=support_data,
            birth_data=birth_data,
            candidate_positions=self.candidate_sources,
            global_birth_candidates=global_birth_candidates,
            global_birth_candidate_counts=global_birth_candidate_counts,
            allow_structural_birth_proposals=not evidence_ready,
        )
        if evidence_ready:
            timing = getattr(filt, "last_structural_timing_s", {})
            if isinstance(timing, dict):
                timing["sparse_evidence_cardinality_ready"] = 1.0
                timing["sparse_evidence_selected_count"] = float(evidence_count)
        rescue_accepts = int(getattr(filt, "last_birth_global_rescue_accepts", 0))
        rescue_rejected = int(getattr(filt, "last_birth_global_rescue_rejected", 0))
        residual_support = int(getattr(filt, "last_birth_residual_support", 0))
        residual_gate_passed = bool(
            getattr(filt, "last_birth_residual_gate_passed", False)
        )
        should_quarantine_residual_candidates = bool(
            np.asarray(global_birth_candidates, dtype=float).size
            and (
                rescue_accepts <= 0
                or rescue_rejected > 0
                or (residual_support > 0 and not residual_gate_passed)
            )
        )
        if should_quarantine_residual_candidates:
            self._inject_runtime_global_birth_quarantine(
                isotope,
                filt,
                proposal_data,
                global_birth_candidates,
                global_birth_candidate_counts,
            )
        self._inject_runtime_report_rescue(isotope, filt)

    def _estimate_global_birth_quarantine_strengths(
        self,
        isotope: str,
        filt: IsotopeParticleFilter,
        data: MeasurementData | None,
        candidates: NDArray[np.float64],
        candidate_counts: NDArray[np.float64] | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Estimate strengths for quarantined global residual candidates."""
        candidate_arr = np.asarray(candidates, dtype=float).reshape(-1, 3)
        if data is None or data.z_k.size == 0 or candidate_arr.size == 0:
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        counts = candidate_counts
        if counts is None:
            candidate_arr, counts = self._global_birth_candidate_counts_for_update(
                isotope,
                filt,
                data,
                candidate_arr,
            )
        if counts is None:
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        candidate_arr = np.asarray(candidate_arr, dtype=float).reshape(-1, 3)
        design = np.asarray(counts, dtype=float)
        if (
            candidate_arr.size == 0
            or design.ndim != 2
            or design.shape[1] != candidate_arr.shape[0]
        ):
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        background = self._background_counts_for_report_refit(isotope, data.live_times)
        strengths = self._solve_report_strengths(
            design=design,
            z_obs=data.z_k,
            background=background,
            observation_variances=data.observation_variances,
            initial_strengths=np.full(
                candidate_arr.shape[0],
                max(float(self.pf_config.birth_q_min), 1.0),
                dtype=float,
            ),
            eps=max(float(self.pf_config.refit_eps), 1.0e-12),
            q_max=float(self.pf_config.birth_q_max),
        )
        strengths = np.maximum(np.asarray(strengths, dtype=float).reshape(-1), 0.0)
        valid = (
            np.isfinite(candidate_arr).all(axis=1)
            & np.isfinite(strengths)
            & (strengths > max(float(self.pf_config.min_strength), 0.0))
        )
        if not np.any(valid):
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        positions, q_arr = dedupe_report_candidates(
            candidate_arr[valid],
            strengths[valid],
            radius_m=max(float(self.pf_config.birth_global_rescue_dedup_radius_m), 0.0),
            max_candidates=max(
                1, int(self.pf_config.birth_global_rescue_max_candidates)
            ),
        )
        return (
            np.asarray(positions, dtype=float).reshape(-1, 3),
            np.asarray(q_arr, dtype=float).reshape(-1),
        )

    def _inject_runtime_global_birth_quarantine(
        self,
        isotope: str,
        filt: IsotopeParticleFilter,
        data: MeasurementData | None,
        candidates: NDArray[np.float64],
        candidate_counts: NDArray[np.float64] | None,
    ) -> int:
        """Inject unaccepted global residual candidates as low-weight hypotheses."""
        if not hasattr(self, "_runtime_global_birth_quarantine_stats"):
            self._runtime_global_birth_quarantine_stats = {}
        self._runtime_global_birth_quarantine_stats[str(isotope)] = {
            "candidates": int(
                np.asarray(candidates, dtype=float).reshape(-1, 3).shape[0]
            )
            if np.asarray(candidates).size
            else 0,
            "sources": 0,
            "injected": 0,
            "weight": 0.0,
        }
        if not (
            bool(self.pf_config.birth_global_rescue_enable)
            and bool(self.pf_config.runtime_report_rescue_quarantine_enable)
        ):
            return 0
        positions, strengths = self._estimate_global_birth_quarantine_strengths(
            isotope,
            filt,
            data,
            candidates,
            candidate_counts,
        )
        if positions.size == 0 or strengths.size == 0:
            return 0
        quarantine_weight = min(
            float(self.pf_config.runtime_report_rescue_quarantine_weight),
            float(self.pf_config.runtime_report_rescue_weight),
        )
        if quarantine_weight <= 0.0:
            return 0
        particle_fraction = self._runtime_report_rescue_injection_fraction(
            quarantine_weight
        )
        injected = filt.inject_runtime_report_rescue_particles(
            positions,
            strengths,
            particle_fraction=particle_fraction,
            min_particles_per_source=(
                self.pf_config.runtime_report_rescue_min_particles_per_source
            ),
            total_weight=quarantine_weight,
            jitter_sigma_m=self.pf_config.runtime_report_rescue_jitter_sigma_m,
            combine_sources=False,
        )
        self._runtime_global_birth_quarantine_stats[str(isotope)] = {
            "candidates": int(
                np.asarray(candidates, dtype=float).reshape(-1, 3).shape[0]
            ),
            "sources": int(positions.shape[0]),
            "injected": int(injected),
            "weight": float(quarantine_weight if injected > 0 else 0.0),
        }
        if injected > 0:
            self._invalidate_report_cache()
        return int(injected)

    def _runtime_global_birth_rescue_candidates(
        self,
        isotope: str,
        filt: IsotopeParticleFilter,
        data: MeasurementData | None,
    ) -> NDArray[np.float64]:
        """Return surface-grid candidates for station-level global birth rescue."""
        if (
            not bool(self.pf_config.birth_global_rescue_enable)
            or data is None
            or data.z_k.size == 0
        ):
            return np.zeros((0, 3), dtype=float)
        max_candidates = max(0, int(self.pf_config.birth_global_rescue_max_candidates))
        if max_candidates <= 0:
            return np.zeros((0, 3), dtype=float)
        try:
            existing_positions, existing_strengths = filt.estimate_clustered(
                max_k=max(max_candidates, self._report_max_sources_per_isotope()),
                include_report_excluded=True,
            )
        except (RuntimeError, ValueError, AttributeError):
            if filt.continuous_particles:
                best_state = filt.best_particle().state
                existing_positions = np.asarray(
                    best_state.positions[: best_state.num_sources],
                    dtype=float,
                )
                existing_strengths = np.asarray(
                    best_state.strengths[: best_state.num_sources],
                    dtype=float,
                )
            else:
                existing_positions = np.zeros((0, 3), dtype=float)
                existing_strengths = np.zeros(0, dtype=float)
        background = self._background_counts_for_report_refit(isotope, data.live_times)
        existing_positions = np.asarray(existing_positions, dtype=float).reshape(-1, 3)
        existing_strengths = np.maximum(
            np.asarray(existing_strengths, dtype=float).reshape(-1),
            0.0,
        )
        positions = np.zeros((0, 3), dtype=float)
        if self.candidate_sources.size:
            positions, _strengths, _stats = self._rank_global_surface_candidates(
                isotope,
                filt,
                data,
                existing_positions=existing_positions,
                background=background,
                eps=max(float(self.pf_config.refit_eps), 1.0e-12),
                q_max=float(self.pf_config.birth_q_max),
                max_candidates=max_candidates,
                min_residual_fraction=(
                    self.pf_config.birth_global_rescue_min_residual_fraction
                ),
                dedup_radius_m=self.pf_config.birth_global_rescue_dedup_radius_m,
            )
        residual_positions = np.zeros((0, 3), dtype=float)
        if existing_positions.size and self.candidate_sources.size:
            existing_design = self._cached_expected_counts_per_source(
                filt=filt,
                isotope=isotope,
                data=data,
                sources=existing_positions,
                strengths=np.ones(existing_positions.shape[0], dtype=float),
            )
            existing_q = self._solve_report_strengths(
                design=existing_design,
                z_obs=data.z_k,
                background=background,
                observation_variances=data.observation_variances,
                initial_strengths=np.full(
                    existing_positions.shape[0],
                    max(float(self.pf_config.birth_q_min), 1.0),
                    dtype=float,
                ),
                eps=max(float(self.pf_config.refit_eps), 1.0e-12),
                q_max=float(self.pf_config.birth_q_max),
            )
            residual = np.maximum(
                np.asarray(data.z_k, dtype=float).reshape(-1)
                - (
                    np.asarray(background, dtype=float).reshape(-1)
                    + np.asarray(existing_design, dtype=float) @ existing_q
                ),
                0.0,
            )
            residual_positions, _residual_strengths, _residual_stats = (
                self._rank_residual_surface_candidates(
                    isotope,
                    filt,
                    data,
                    residual=residual,
                    existing_positions=existing_positions,
                    background=background,
                    eps=max(float(self.pf_config.refit_eps), 1.0e-12),
                    max_candidates=max_candidates,
                    min_residual_fraction=(
                        self.pf_config.birth_global_rescue_min_residual_fraction
                    ),
                    dedup_radius_m=self.pf_config.birth_global_rescue_dedup_radius_m,
                )
            )
        merged = (
            np.vstack([positions, residual_positions])
            if positions.size and residual_positions.size
            else positions
            if positions.size
            else residual_positions
        )
        split_positions = self._runtime_high_strength_split_candidates(
            filt,
            existing_positions,
            existing_strengths,
            data=data,
        )
        if split_positions.size:
            merged = (
                np.vstack([merged, split_positions])
                if np.asarray(merged).size
                else split_positions
            )
        if merged.size == 0:
            return self._merge_global_birth_candidate_memory(
                isotope,
                np.zeros((0, 3), dtype=float),
            )
        final_positions, _final_strengths = dedupe_report_candidates(
            np.asarray(merged, dtype=float).reshape(-1, 3),
            np.ones(np.asarray(merged).reshape(-1, 3).shape[0], dtype=float),
            radius_m=self.pf_config.birth_global_rescue_dedup_radius_m,
            max_candidates=max_candidates,
        )
        return self._merge_global_birth_candidate_memory(
            isotope,
            np.asarray(final_positions, dtype=float).reshape(-1, 3),
        )

    def _runtime_high_strength_split_candidates(
        self,
        filt: IsotopeParticleFilter,
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
        *,
        data: MeasurementData | None = None,
    ) -> NDArray[np.float64]:
        """Return surface-projected split candidates around over-strong modes."""
        if not bool(self.pf_config.high_strength_split_enable):
            return np.zeros((0, 3), dtype=float)
        pos_arr = np.asarray(positions, dtype=float).reshape(-1, 3)
        q_arr = np.maximum(np.asarray(strengths, dtype=float).reshape(-1), 0.0)
        if pos_arr.shape[0] == 0 or pos_arr.shape[0] != q_arr.size:
            return np.zeros((0, 3), dtype=float)
        prior_mean = max(float(self.pf_config.source_strength_prior_mean), 0.0)
        if prior_mean > 0.0:
            reference = max(
                prior_mean,
                float(self.pf_config.split_strength_min),
                1.0,
            )
            threshold = reference * max(
                float(self.pf_config.high_strength_split_q_multiple),
                1.0,
            )
            high = q_arr >= threshold
        else:
            high = self._observation_limited_high_strength_mask(
                filt,
                data,
                pos_arr,
                q_arr,
            )
        if not np.any(high):
            return np.zeros((0, 3), dtype=float)
        count = max(1, int(self.pf_config.high_strength_split_candidate_count))
        offset = max(float(self.pf_config.high_strength_split_offset_m), 1.0e-6)
        angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        xy_offsets = np.column_stack(
            [
                np.cos(angles) * offset,
                np.sin(angles) * offset,
                np.zeros(count, dtype=float),
            ]
        )
        axial_offsets = np.asarray(
            [
                [0.0, 0.0, offset],
                [0.0, 0.0, -offset],
            ],
            dtype=float,
        )
        offsets = np.vstack([xy_offsets, axial_offsets])
        candidates = (pos_arr[high, None, :] + offsets[None, :, :]).reshape(-1, 3)
        projected = filt._project_positions_to_source_prior(candidates)
        parent_positions = pos_arr[high]
        parent_repeated = np.repeat(parent_positions, offsets.shape[0], axis=0)
        distances = np.linalg.norm(projected - parent_repeated, axis=1)
        keep = np.isfinite(projected).all(axis=1) & (
            distances >= 0.5 * max(float(self.pf_config.birth_min_sep_m), 0.0)
        )
        if not np.any(keep):
            return np.zeros((0, 3), dtype=float)
        kept = projected[keep]
        deduped, _ = dedupe_report_candidates(
            kept,
            np.ones(kept.shape[0], dtype=float),
            radius_m=max(float(self.pf_config.birth_global_rescue_dedup_radius_m), 0.0),
            max_candidates=max(
                1, int(self.pf_config.birth_global_rescue_max_candidates)
            ),
        )
        return np.asarray(deduped, dtype=float).reshape(-1, 3)

    def _observation_limited_high_strength_mask(
        self,
        filt: IsotopeParticleFilter,
        data: MeasurementData | None,
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """Return sources whose individual response is too large for observations."""
        pos_arr = np.asarray(positions, dtype=float).reshape(-1, 3)
        q_arr = np.maximum(np.asarray(strengths, dtype=float).reshape(-1), 0.0)
        if data is None or data.z_k.size == 0 or pos_arr.shape[0] != q_arr.size:
            return np.zeros(q_arr.size, dtype=bool)
        if q_arr.size == 0:
            return np.zeros(0, dtype=bool)
        counts = self._cached_expected_counts_per_source(
            filt=filt,
            isotope=filt.isotope,
            data=data,
            sources=pos_arr,
            strengths=np.ones(pos_arr.shape[0], dtype=float),
        )
        counts = np.maximum(np.asarray(counts, dtype=float), 0.0)
        if counts.ndim != 2 or counts.shape != (data.z_k.size, q_arr.size):
            return np.zeros(q_arr.size, dtype=bool)
        try:
            background_rate = float(filt.best_particle().state.background)
        except (RuntimeError, ValueError, AttributeError):
            background_rate = 0.0
        background = np.asarray(data.live_times, dtype=float).reshape(-1, 1) * max(
            background_rate, 0.0
        )
        guarded = filt._apply_runtime_observation_overshoot_guard(
            q_arr[None, :],
            counts[:, None, :],
            data,
            background,
            eps=max(float(self.pf_config.refit_eps), 1.0e-12),
        )
        if guarded.shape != (1, q_arr.size):
            return np.zeros(q_arr.size, dtype=bool)
        shrink = np.asarray(guarded[0], dtype=float) < q_arr * (1.0 - 1.0e-9)
        return np.asarray(shrink, dtype=bool)

    def _structural_update_worker_count(self, task_count: int) -> int:
        """Return the worker count for independent per-isotope structural updates."""
        if task_count <= 1 or not bool(self.pf_config.parallel_isotope_updates):
            return 1
        configured = self.pf_config.parallel_isotope_workers
        if configured is None:
            configured = os.cpu_count() or 1
        return max(1, min(int(configured), int(task_count)))

    def _apply_birth_death(self, birth_window_override: int | None = None) -> None:
        """Apply per-isotope birth/death updates using recent measurements."""
        structural_start = time.perf_counter()
        tasks: list[
            tuple[
                str,
                IsotopeParticleFilter,
                MeasurementData | None,
                MeasurementData | None,
                MeasurementData | None,
            ]
        ] = []
        birth_window = (
            self.pf_config.birth_window
            if birth_window_override is None
            else max(1, int(birth_window_override))
        )
        for iso, filt in self.filters.items():
            if getattr(filt, "is_converged", False) and getattr(
                filt.config, "converge_enable", False
            ):
                continue
            refit_data = (
                self._measurement_data_for_iso(
                    iso,
                    self.pf_config.conditional_strength_refit_window,
                )
                if bool(self.pf_config.conditional_strength_refit)
                else None
            )
            support_data = self._measurement_data_for_iso(
                iso, self.pf_config.support_window
            )
            birth_data = self._measurement_data_for_iso(iso, birth_window)
            tasks.append((iso, filt, refit_data, support_data, birth_data))
        worker_count = self._structural_update_worker_count(len(tasks))
        self.last_structural_update_workers = int(worker_count)
        if worker_count <= 1:
            for task in tasks:
                self._run_isotope_structural_update(task)
            self.last_structural_update_wall_s = time.perf_counter() - structural_start
            return
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            list(executor.map(self._run_isotope_structural_update, tasks))
        self.last_structural_update_wall_s = time.perf_counter() - structural_start

    def _update_pre_finalize_guard(
        self,
        pre_finalize: Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]],
        post_finalize: Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]],
    ) -> None:
        """
        Preserve reporting candidates if station finalization collapses them.

        The guard is non-destructive: it only affects reported estimates.  PF
        particles remain governed by their likelihoods, but the final report can
        compare pre/post finalize posterior modes and avoid losing a
        multi-source hypothesis solely due to a late resample/refit/prune step.
        """
        if not bool(self.pf_config.report_pre_finalize_guard):
            self._pre_finalize_guard_estimates.clear()
            return
        max_sources = self.pf_config.max_sources
        for isotope in self.isotopes:
            pre_pos, pre_q = pre_finalize.get(
                isotope,
                (np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)),
            )
            post_pos, _ = post_finalize.get(
                isotope,
                (np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)),
            )
            pre_count = int(np.asarray(pre_pos).shape[0])
            post_count = int(np.asarray(post_pos).shape[0])
            if max_sources is not None and pre_count > int(max_sources):
                self._pre_finalize_guard_estimates.pop(isotope, None)
                continue
            if pre_count > post_count and pre_count > 0:
                self._pre_finalize_guard_estimates[isotope] = (
                    np.asarray(pre_pos, dtype=float).copy(),
                    np.asarray(pre_q, dtype=float).copy(),
                )
            elif post_count >= pre_count:
                self._pre_finalize_guard_estimates.pop(isotope, None)

    def estimates(
        self,
        *,
        use_pre_finalize_guard: bool = True,
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Return per-isotope position/strength estimates for reporting."""
        cache_revision = int(self._report_cache_revision)
        cache_key = (cache_revision, bool(use_pre_finalize_guard))
        cached = self._report_estimate_cache.get(cache_key)
        if cached is not None:
            return self._copy_estimate_map(cached)
        estimates: Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
        for isotope, filt in self.filters.items():
            use_clustered = bool(
                filt.config.birth_enable and filt.config.use_clustered_output
            )
            if use_clustered and hasattr(filt, "estimate_clustered"):
                try:
                    clustered = filt.estimate_clustered()
                    if clustered[0].shape[0] > 0:
                        estimates[isotope] = self._guarded_report_estimate(
                            isotope,
                            *self._refit_reported_strengths(
                                isotope,
                                clustered[0],
                                clustered[1],
                            ),
                            use_pre_finalize_guard=use_pre_finalize_guard,
                        )
                        continue
                except RuntimeError:
                    estimates[isotope] = (
                        np.zeros((0, 3), dtype=float),
                        np.zeros(0, dtype=float),
                    )
                    continue
            raw_positions, raw_strengths = filt.estimate()
            estimates[isotope] = self._guarded_report_estimate(
                isotope,
                *self._refit_reported_strengths(isotope, raw_positions, raw_strengths),
                use_pre_finalize_guard=use_pre_finalize_guard,
            )
        if int(self._report_cache_revision) == cache_revision:
            self._report_estimate_cache[cache_key] = self._copy_estimate_map(estimates)
        return self._copy_estimate_map(estimates)

    def _guarded_report_estimate(
        self,
        isotope: str,
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
        *,
        use_pre_finalize_guard: bool,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return protected pre-finalize estimates when post-finalize collapsed."""
        pos_arr = np.asarray(positions, dtype=float)
        q_arr = np.asarray(strengths, dtype=float)
        if not use_pre_finalize_guard:
            return pos_arr, q_arr
        guarded = self._pre_finalize_guard_estimates.get(isotope)
        if guarded is None:
            return pos_arr, q_arr
        guard_pos, guard_q = guarded
        if guard_pos.shape[0] <= pos_arr.shape[0]:
            return pos_arr, q_arr
        refit_pos, refit_q = self._refit_reported_strengths(
            isotope,
            guard_pos,
            guard_q,
        )
        if refit_pos.shape[0] >= guard_pos.shape[0]:
            return refit_pos, refit_q
        return guard_pos.copy(), guard_q.copy()

    def estimate_all(
        self,
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Alias for estimates() to align with visualization helpers."""
        return self.estimates()

    def report_model_order_diagnostics(self) -> Dict[str, Dict[str, Any]]:
        """Return the latest report-level model-order diagnostics."""
        return copy.deepcopy(self._last_report_model_order_diagnostics)

    def sparse_poisson_evidence_diagnostics(self) -> Dict[str, Dict[str, Any]]:
        """Return the latest all-history sparse Poisson evidence diagnostics."""
        diagnostics = copy.deepcopy(self._last_sparse_poisson_evidence_diagnostics)
        if self._last_joint_sparse_poisson_evidence_diagnostics:
            diagnostics["joint_multi_isotope"] = copy.deepcopy(
                self._last_joint_sparse_poisson_evidence_diagnostics
            )
        return diagnostics

    def report_model_order_ready(self) -> bool:
        """Return True when report-level multi-source model orders are stable."""
        diagnostics = self.report_model_order_diagnostics()
        if not diagnostics:
            return False
        for stats in diagnostics.values():
            candidate_count = int(stats.get("candidate_count", 0))
            selected_count = int(stats.get("selected_count", 0))
            if max(candidate_count, selected_count) <= 1:
                continue
            if not bool(stats.get("model_order_ready", False)):
                return False
        return not bool(self.unresolved_structural_evidence())

    def unresolved_isotope_evidence(
        self,
        *,
        window: int | None = None,
        min_total_counts: float = 25.0,
        min_max_count: float = 5.0,
        min_snr: float = 2.0,
    ) -> dict[str, dict[str, Any]]:
        """Return isotopes whose observations remain unsupported by zero-source PF MAPs."""
        evidence: dict[str, dict[str, Any]] = {}
        total_floor = max(float(min_total_counts), 0.0)
        max_floor = max(float(min_max_count), 0.0)
        snr_floor = max(float(min_snr), 0.0)
        for isotope, filt in self.filters.items():
            data = self._measurement_data_for_iso(isotope, window)
            if data is None or data.z_k.size == 0:
                continue
            counts = np.maximum(np.asarray(data.z_k, dtype=float).reshape(-1), 0.0)
            variances = np.maximum(
                np.asarray(data.observation_variances, dtype=float).reshape(-1),
                1.0,
            )
            total_counts = float(np.sum(counts))
            max_count = float(np.max(counts)) if counts.size else 0.0
            snr = float(total_counts / np.sqrt(max(float(np.sum(variances)), 1.0e-12)))
            if not filt.continuous_particles:
                map_count = 0
                source_probability = 0.0
                map_confidence = 1.0
            else:
                weights = np.asarray(filt.continuous_weights, dtype=float).reshape(-1)
                weights_sum = float(np.sum(weights))
                if weights.size == 0 or weights_sum <= 0.0:
                    weights = np.full(
                        len(filt.continuous_particles),
                        1.0 / max(len(filt.continuous_particles), 1),
                        dtype=float,
                    )
                else:
                    weights = weights / weights_sum
                source_counts = np.asarray(
                    [
                        int(particle.state.num_sources)
                        for particle in filt.continuous_particles
                    ],
                    dtype=int,
                )
                unique, inverse = np.unique(source_counts, return_inverse=True)
                probs = np.zeros(unique.size, dtype=float)
                np.add.at(probs, inverse, weights)
                best = int(np.argmax(probs)) if probs.size else 0
                map_count = int(unique[best]) if unique.size else 0
                map_confidence = float(probs[best]) if probs.size else 1.0
                source_probability = float(np.sum(weights[source_counts > 0]))
            total_ratio = (
                total_counts / total_floor
                if total_floor > 0.0
                else float(total_counts > 0.0)
            )
            max_ratio = (
                max_count / max_floor if max_floor > 0.0 else float(max_count > 0.0)
            )
            snr_ratio = snr / snr_floor if snr_floor > 0.0 else float(snr > 0.0)
            count_floor_met = total_counts >= total_floor or max_count >= max_floor
            snr_floor_met = snr >= snr_floor
            if snr_floor <= 0.0:
                observed = count_floor_met
            elif total_floor <= 0.0 and max_floor <= 0.0:
                observed = snr_floor_met
            else:
                observed = count_floor_met and snr_floor_met
            if map_count <= 0 and observed:
                evidence[str(isotope)] = {
                    "reason": "observed_counts_without_map_source",
                    "total_counts": total_counts,
                    "max_count": max_count,
                    "count_snr": snr,
                    "map_source_count": int(map_count),
                    "map_cardinality_confidence": map_confidence,
                    "source_probability": source_probability,
                    "budget": float(max(total_ratio, max_ratio, snr_ratio, 1.0) - 1.0),
                    "min_total_counts": total_floor,
                    "min_max_count": max_floor,
                    "min_snr": snr_floor,
                }
        return evidence

    def unresolved_structural_evidence(self) -> dict[str, dict[str, Any]]:
        """
        Return isotope-wise pseudo-source or birth-residual evidence that still needs views.

        Report-level BIC can only compare candidates that survived into the current
        report set.  If residual birth or pseudo-source verification is still asking
        for discriminative views, a one-candidate BIC result must not be treated as
        a settled model order.
        """
        unresolved: dict[str, dict[str, Any]] = {}
        discriminative_reasons = {
            "needs_discriminative_views",
            "insufficient_distinct_views",
            "high_response_corr",
            "too_young_to_prune",
        }
        support_floor = max(1, int(self.pf_config.birth_residual_min_support))
        for isotope, filt in self.filters.items():
            payload: dict[str, Any] = {}
            report_stats = self._last_report_model_order_diagnostics.get(
                str(isotope),
                {},
            )
            if isinstance(report_stats, dict):
                selected_count = int(report_stats.get("selected_count", 0))
                count_supported_zero = bool(
                    report_stats.get("count_supported_zero_source", False)
                )
                ready = bool(report_stats.get("model_order_ready", False))
                if selected_count == 0 and count_supported_zero and not ready:
                    payload["report_underfit"] = {
                        "reason": "count_supported_zero_source",
                        "observed_signal_total_counts": float(
                            report_stats.get("observed_signal_total_counts", 0.0)
                        ),
                        "observed_signal_max_count": float(
                            report_stats.get("observed_signal_max_count", 0.0)
                        ),
                        "observed_signal_snr": float(
                            report_stats.get("observed_signal_snr", 0.0)
                        ),
                        "criterion_margin_to_runner_up": float(
                            report_stats.get(
                                "criterion_margin_to_runner_up",
                                float("inf"),
                            )
                        ),
                        "zero_source_ready_margin": float(
                            report_stats.get("zero_source_ready_margin", 0.0)
                        ),
                    }
                residual_fraction = float(
                    report_stats.get("selected_positive_residual_fraction", 0.0)
                )
                min_fraction = max(
                    float(self.pf_config.report_mle_rescue_min_residual_fraction),
                    0.0,
                )
                if selected_count > 0 and residual_fraction >= min_fraction:
                    payload["report_underfit"] = {
                        "reason": "selected_model_positive_residual",
                        "selected_count": int(selected_count),
                        "model_order_ready": bool(ready),
                        "selected_positive_residual_fraction": float(residual_fraction),
                        "selected_positive_residual_chi2": float(
                            report_stats.get("selected_positive_residual_chi2", 0.0)
                        ),
                    }
            reasons = getattr(filt, "last_pseudo_source_fail_reasons", {})
            reason_payload = (
                {str(reason): int(count) for reason, count in reasons.items()}
                if isinstance(reasons, dict)
                else {}
            )
            unresolved_pseudo = {
                reason: count
                for reason, count in reason_payload.items()
                if reason in discriminative_reasons and int(count) > 0
            }
            if unresolved_pseudo:
                payload["pseudo_source_fail_reasons"] = unresolved_pseudo
            birth_gate_passed = bool(
                getattr(filt, "last_birth_residual_gate_passed", False)
            )
            birth_support = int(getattr(filt, "last_birth_residual_support", 0))
            if birth_gate_passed and birth_support >= support_floor:
                payload["birth_residual"] = {
                    "gate_passed": True,
                    "support": int(birth_support),
                    "support_floor": int(support_floor),
                    "chi2": float(getattr(filt, "last_birth_residual_chi2", 0.0)),
                    "p_value": float(getattr(filt, "last_birth_residual_p_value", 1.0)),
                }
            rescue_entry = self._runtime_report_rescue_modes.get(str(isotope))
            if rescue_entry is not None:
                rescue_pos, _rescue_q, rescue_weight = rescue_entry
                rescue_count = int(
                    np.asarray(rescue_pos, dtype=float).reshape(-1, 3).shape[0]
                )
                diagnostics = self._last_report_model_order_diagnostics.get(
                    str(isotope),
                    {},
                )
                selected_count = int(diagnostics.get("selected_count", 0))
                report_ready = bool(diagnostics.get("model_order_ready", False))
                quarantine_weight = float(
                    self.pf_config.runtime_report_rescue_quarantine_weight
                )
                if rescue_count > max(selected_count, 0) or (
                    rescue_count > 0
                    and not report_ready
                    and float(rescue_weight) <= quarantine_weight + 1.0e-12
                ):
                    payload["runtime_rescue_unverified"] = {
                        "rescue_source_count": int(rescue_count),
                        "report_selected_count": int(selected_count),
                        "rescue_weight": float(rescue_weight),
                        "quarantine_weight": float(quarantine_weight),
                    }
            if payload:
                unresolved[str(isotope)] = payload
        absent_evidence = self.unresolved_isotope_evidence(
            min_total_counts=max(
                float(self.pf_config.structural_update_min_counts), 25.0
            ),
            min_max_count=max(
                float(self.pf_config.conditional_strength_refit_min_count), 5.0
            ),
            min_snr=max(float(self.pf_config.structural_update_min_snr), 2.0),
        )
        for isotope, payload in absent_evidence.items():
            unresolved.setdefault(str(isotope), {})["isotope_absence"] = payload
        return unresolved

    def step_diagnostics(
        self,
        top_k: int = 3,
        *,
        include_estimates: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return per-isotope diagnostics for the current PF state.

        The diagnostics include ESS, resample/birth/kill counts, and the source
        count distribution.  When include_estimates is false, the routine avoids
        report-only clustered MMSE recomputation so per-measurement health logs
        cannot stall the runtime path.
        """
        diagnostics: Dict[str, Dict[str, Any]] = {}
        eps = 1e-12
        k = max(0, int(top_k))
        for iso, filt in self.filters.items():
            if not filt.continuous_particles:
                diagnostics[iso] = {
                    "ess_pre": 0.0,
                    "resampled": False,
                    "ess_post": None,
                    "n_after_adapt": 0,
                    "resample_count": int(getattr(filt, "last_resample_count", 0)),
                    "mode_preserved_count": int(
                        getattr(filt, "last_mode_preserved_count", 0)
                    ),
                    "mode_preserving_strata_summary": dict(
                        getattr(filt, "last_mode_preserving_strata_summary", {})
                    ),
                    "mode_preserving_selected_strata": list(
                        getattr(filt, "last_mode_preserving_selected_strata", [])
                    ),
                    "mode_preserving_cardinality_summary": dict(
                        getattr(filt, "last_mode_preserving_cardinality_summary", {})
                    ),
                    "mode_preserving_selected_cardinalities": list(
                        getattr(
                            filt,
                            "last_mode_preserving_selected_cardinalities",
                            [],
                        )
                    ),
                    "mode_preserving_dynamic_spatial_summary": list(
                        getattr(
                            filt,
                            "last_mode_preserving_dynamic_spatial_summary",
                            [],
                        )
                    ),
                    "birth_count": int(getattr(filt, "last_birth_count", 0)),
                    "kill_count": int(getattr(filt, "last_kill_count", 0)),
                    "weak_source_prune_occlusion_protected": int(
                        getattr(
                            filt,
                            "last_weak_source_prune_occlusion_protected",
                            0,
                        )
                    ),
                    "birth_residual_chi2": float(
                        getattr(filt, "last_birth_residual_chi2", 0.0)
                    ),
                    "birth_residual_p_value": float(
                        getattr(filt, "last_birth_residual_p_value", 1.0)
                    ),
                    "birth_residual_support": int(
                        getattr(filt, "last_birth_residual_support", 0)
                    ),
                    "birth_residual_distinct_poses": int(
                        getattr(filt, "last_birth_residual_distinct_poses", 0)
                    ),
                    "birth_residual_distinct_stations": int(
                        getattr(filt, "last_birth_residual_distinct_stations", 0)
                    ),
                    "birth_residual_gate_passed": bool(
                        getattr(filt, "last_birth_residual_gate_passed", False)
                    ),
                    "birth_residual_refit_fraction": float(
                        getattr(filt, "last_birth_residual_refit_fraction", 1.0)
                    ),
                    "birth_residual_refit_gate_passed": bool(
                        getattr(filt, "last_birth_residual_refit_gate_passed", True)
                    ),
                    "birth_residual_layer": str(
                        getattr(filt, "last_birth_residual_layer", "none")
                    ),
                    "birth_residual_layer_count": int(
                        getattr(filt, "last_birth_residual_layer_count", 0)
                    ),
                    "birth_forced_attempts": int(
                        getattr(filt, "last_birth_forced_attempts", 0)
                    ),
                    "birth_forced_accepts": int(
                        getattr(filt, "last_birth_forced_accepts", 0)
                    ),
                    "birth_forced_mask_relaxations": int(
                        getattr(filt, "last_birth_forced_mask_relaxations", 0)
                    ),
                    "birth_forced_no_candidate": int(
                        getattr(filt, "last_birth_forced_no_candidate", 0)
                    ),
                    "birth_forced_rejected": int(
                        getattr(filt, "last_birth_forced_rejected", 0)
                    ),
                    "birth_forced_best_delta": float(
                        getattr(filt, "last_birth_forced_best_delta", -np.inf)
                    ),
                    "birth_global_rescue_candidates": int(
                        getattr(filt, "last_birth_global_rescue_candidates", 0)
                    ),
                    "birth_global_rescue_attempts": int(
                        getattr(filt, "last_birth_global_rescue_attempts", 0)
                    ),
                    "birth_global_rescue_accepts": int(
                        getattr(filt, "last_birth_global_rescue_accepts", 0)
                    ),
                    "birth_global_rescue_rejected": int(
                        getattr(filt, "last_birth_global_rescue_rejected", 0)
                    ),
                    "birth_global_rescue_best_delta": float(
                        getattr(filt, "last_birth_global_rescue_best_delta", -np.inf)
                    ),
                    "runtime_global_birth_quarantine_candidates": int(
                        self._runtime_global_birth_quarantine_stats.get(iso, {}).get(
                            "candidates",
                            0,
                        )
                    ),
                    "runtime_global_birth_quarantine_sources": int(
                        self._runtime_global_birth_quarantine_stats.get(iso, {}).get(
                            "sources",
                            0,
                        )
                    ),
                    "runtime_global_birth_quarantine_injected": int(
                        self._runtime_global_birth_quarantine_stats.get(iso, {}).get(
                            "injected",
                            0,
                        )
                    ),
                    "runtime_global_birth_quarantine_weight": float(
                        self._runtime_global_birth_quarantine_stats.get(iso, {}).get(
                            "weight",
                            0.0,
                        )
                    ),
                    "runtime_report_rescue_candidates": int(
                        getattr(filt, "last_runtime_report_rescue_candidates", 0)
                    ),
                    "runtime_report_rescue_sources": int(
                        getattr(filt, "last_runtime_report_rescue_sources", 0)
                    ),
                    "runtime_report_rescue_injected": int(
                        getattr(filt, "last_runtime_report_rescue_injected", 0)
                    ),
                    "runtime_report_rescue_weight": float(
                        getattr(filt, "last_runtime_report_rescue_weight", 0.0)
                    ),
                    "birth_structural_eligible": int(
                        getattr(filt, "last_birth_structural_eligible", 0)
                    ),
                    "pseudo_source_verified": int(
                        getattr(filt, "last_pseudo_source_verified", 0)
                    ),
                    "pseudo_source_failed": int(
                        getattr(filt, "last_pseudo_source_failed", 0)
                    ),
                    "pseudo_source_pruned": int(
                        getattr(filt, "last_pseudo_source_pruned", 0)
                    ),
                    "pseudo_source_quarantined": int(
                        getattr(filt, "last_pseudo_source_quarantined", 0)
                    ),
                    "pseudo_source_quarantine_active": int(
                        getattr(filt, "last_pseudo_source_quarantine_active", 0)
                    ),
                    "pseudo_source_fail_reasons": dict(
                        getattr(filt, "last_pseudo_source_fail_reasons", {})
                    ),
                    "structural_timing_s": dict(
                        getattr(filt, "last_structural_timing_s", {})
                    ),
                    "temper_steps": [],
                    "temper_resamples": 0,
                    "r_mean": 0.0,
                    "r_var": 0.0,
                    "r_weighted_mean": 0.0,
                    "r_weighted_var": 0.0,
                    "r_probability_by_count": {},
                    "r_particle_count_by_count": {},
                    "map": (np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)),
                    "mmse": (np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)),
                    "top_k": [],
                    "converged": bool(getattr(filt, "is_converged", False)),
                    "updates_skipped": int(getattr(filt, "updates_skipped", 0)),
                }
                continue
            weights = np.asarray(filt.continuous_weights, dtype=float)
            total = float(np.sum(weights))
            if total > 0.0:
                weights = weights / total
            elif weights.size:
                weights = np.full(weights.size, 1.0 / float(weights.size), dtype=float)
            r_vals = np.array(
                [p.state.num_sources for p in filt.continuous_particles], dtype=float
            )
            if weights.size != r_vals.size and r_vals.size:
                weights = np.full(r_vals.size, 1.0 / float(r_vals.size), dtype=float)
            r_mean = float(np.mean(r_vals)) if r_vals.size else 0.0
            r_var = float(np.var(r_vals)) if r_vals.size else 0.0
            r_int = r_vals.astype(int, copy=False)
            r_weighted_mean = float(np.sum(weights * r_vals)) if r_vals.size else 0.0
            r_weighted_var = (
                float(np.sum(weights * (r_vals - r_weighted_mean) ** 2))
                if r_vals.size
                else 0.0
            )
            r_probability_by_count = {
                str(int(value)): float(np.sum(weights[r_int == int(value)]))
                for value in sorted(set(int(v) for v in r_int.tolist()))
            }
            r_particle_count_by_count = {
                str(int(value)): int(np.count_nonzero(r_int == int(value)))
                for value in sorted(set(int(v) for v in r_int.tolist()))
            }
            ess_pre = getattr(filt, "last_ess_pre", None)
            if ess_pre is None and weights.size:
                ess_pre = float(1.0 / max(np.sum(weights**2), eps))
            if ess_pre is None:
                ess_pre = 0.0
            resampled = bool(getattr(filt, "last_resample_ess", False))
            ess_post = getattr(filt, "last_ess_post", None)
            n_after_adapt = getattr(filt, "last_n_after_adapt", None)
            if n_after_adapt is None:
                n_after_adapt = int(len(filt.continuous_particles))
            best_state = filt.state_without_quarantined_sources(
                filt.best_particle().state
            )
            best_source_count = max(0, int(best_state.num_sources))
            map_positions = best_state.positions[:best_source_count].copy()
            map_strengths = best_state.strengths[:best_source_count].copy()
            if include_estimates:
                try:
                    if bool(
                        filt.config.birth_enable and filt.config.use_clustered_output
                    ) and hasattr(filt, "estimate_clustered"):
                        mmse_positions, mmse_strengths = filt.estimate_clustered()
                    else:
                        mmse_positions, mmse_strengths = filt.estimate()
                except RuntimeError:
                    mmse_positions = np.zeros((0, 3), dtype=float)
                    mmse_strengths = np.zeros(0, dtype=float)
            else:
                mmse_positions = np.zeros((0, 3), dtype=float)
                mmse_strengths = np.zeros(0, dtype=float)
            top_entries: List[Dict[str, Any]] = []
            if k > 0 and weights.size:
                order = np.argsort(weights)[::-1][:k]
                for idx in order:
                    state = filt.state_without_quarantined_sources(
                        filt.continuous_particles[int(idx)].state
                    )
                    source_count = max(0, int(state.num_sources))
                    top_entries.append(
                        {
                            "weight": float(weights[idx]),
                            "num_sources": source_count,
                            "positions": state.positions[:source_count].copy(),
                            "strengths": state.strengths[:source_count].copy(),
                        }
                    )
            diagnostics[iso] = {
                "ess_pre": float(ess_pre),
                "resampled": resampled,
                "ess_post": ess_post,
                "n_after_adapt": int(n_after_adapt),
                "resample_count": int(getattr(filt, "last_resample_count", 0)),
                "mode_preserved_count": int(
                    getattr(filt, "last_mode_preserved_count", 0)
                ),
                "mode_preserving_strata_summary": dict(
                    getattr(filt, "last_mode_preserving_strata_summary", {})
                ),
                "mode_preserving_selected_strata": list(
                    getattr(filt, "last_mode_preserving_selected_strata", [])
                ),
                "mode_preserving_cardinality_summary": dict(
                    getattr(filt, "last_mode_preserving_cardinality_summary", {})
                ),
                "mode_preserving_selected_cardinalities": list(
                    getattr(
                        filt,
                        "last_mode_preserving_selected_cardinalities",
                        [],
                    )
                ),
                "mode_preserving_dynamic_spatial_summary": list(
                    getattr(
                        filt,
                        "last_mode_preserving_dynamic_spatial_summary",
                        [],
                    )
                ),
                "birth_count": int(getattr(filt, "last_birth_count", 0)),
                "kill_count": int(getattr(filt, "last_kill_count", 0)),
                "weak_source_prune_occlusion_protected": int(
                    getattr(
                        filt,
                        "last_weak_source_prune_occlusion_protected",
                        0,
                    )
                ),
                "birth_residual_chi2": float(
                    getattr(filt, "last_birth_residual_chi2", 0.0)
                ),
                "birth_residual_p_value": float(
                    getattr(filt, "last_birth_residual_p_value", 1.0)
                ),
                "birth_residual_support": int(
                    getattr(filt, "last_birth_residual_support", 0)
                ),
                "birth_residual_distinct_poses": int(
                    getattr(filt, "last_birth_residual_distinct_poses", 0)
                ),
                "birth_residual_distinct_stations": int(
                    getattr(filt, "last_birth_residual_distinct_stations", 0)
                ),
                "birth_residual_gate_passed": bool(
                    getattr(filt, "last_birth_residual_gate_passed", False)
                ),
                "birth_residual_refit_fraction": float(
                    getattr(filt, "last_birth_residual_refit_fraction", 1.0)
                ),
                "birth_residual_refit_gate_passed": bool(
                    getattr(filt, "last_birth_residual_refit_gate_passed", True)
                ),
                "birth_residual_layer": str(
                    getattr(filt, "last_birth_residual_layer", "none")
                ),
                "birth_residual_layer_count": int(
                    getattr(filt, "last_birth_residual_layer_count", 0)
                ),
                "birth_forced_attempts": int(
                    getattr(filt, "last_birth_forced_attempts", 0)
                ),
                "birth_forced_accepts": int(
                    getattr(filt, "last_birth_forced_accepts", 0)
                ),
                "birth_forced_mask_relaxations": int(
                    getattr(filt, "last_birth_forced_mask_relaxations", 0)
                ),
                "birth_forced_no_candidate": int(
                    getattr(filt, "last_birth_forced_no_candidate", 0)
                ),
                "birth_forced_rejected": int(
                    getattr(filt, "last_birth_forced_rejected", 0)
                ),
                "birth_forced_best_delta": float(
                    getattr(filt, "last_birth_forced_best_delta", -np.inf)
                ),
                "birth_global_rescue_candidates": int(
                    getattr(filt, "last_birth_global_rescue_candidates", 0)
                ),
                "birth_global_rescue_attempts": int(
                    getattr(filt, "last_birth_global_rescue_attempts", 0)
                ),
                "birth_global_rescue_accepts": int(
                    getattr(filt, "last_birth_global_rescue_accepts", 0)
                ),
                "birth_global_rescue_rejected": int(
                    getattr(filt, "last_birth_global_rescue_rejected", 0)
                ),
                "birth_global_rescue_best_delta": float(
                    getattr(filt, "last_birth_global_rescue_best_delta", -np.inf)
                ),
                "runtime_global_birth_quarantine_candidates": int(
                    self._runtime_global_birth_quarantine_stats.get(iso, {}).get(
                        "candidates",
                        0,
                    )
                ),
                "runtime_global_birth_quarantine_sources": int(
                    self._runtime_global_birth_quarantine_stats.get(iso, {}).get(
                        "sources",
                        0,
                    )
                ),
                "runtime_global_birth_quarantine_injected": int(
                    self._runtime_global_birth_quarantine_stats.get(iso, {}).get(
                        "injected",
                        0,
                    )
                ),
                "runtime_global_birth_quarantine_weight": float(
                    self._runtime_global_birth_quarantine_stats.get(iso, {}).get(
                        "weight",
                        0.0,
                    )
                ),
                "runtime_report_rescue_candidates": int(
                    getattr(filt, "last_runtime_report_rescue_candidates", 0)
                ),
                "runtime_report_rescue_sources": int(
                    getattr(filt, "last_runtime_report_rescue_sources", 0)
                ),
                "runtime_report_rescue_injected": int(
                    getattr(filt, "last_runtime_report_rescue_injected", 0)
                ),
                "runtime_report_rescue_weight": float(
                    getattr(filt, "last_runtime_report_rescue_weight", 0.0)
                ),
                "candidate_verification_queue_sources": int(
                    np.asarray(
                        self._candidate_verification_queue.get(
                            str(iso),
                            (
                                np.zeros((0, 3), dtype=float),
                                np.zeros(0, dtype=float),
                                np.zeros(0, dtype=float),
                            ),
                        )[0],
                        dtype=float,
                    )
                    .reshape(-1, 3)
                    .shape[0]
                ),
                "birth_structural_eligible": int(
                    getattr(filt, "last_birth_structural_eligible", 0)
                ),
                "pseudo_source_verified": int(
                    getattr(filt, "last_pseudo_source_verified", 0)
                ),
                "pseudo_source_failed": int(
                    getattr(filt, "last_pseudo_source_failed", 0)
                ),
                "pseudo_source_pruned": int(
                    getattr(filt, "last_pseudo_source_pruned", 0)
                ),
                "pseudo_source_quarantined": int(
                    getattr(filt, "last_pseudo_source_quarantined", 0)
                ),
                "pseudo_source_quarantine_active": int(
                    getattr(filt, "last_pseudo_source_quarantine_active", 0)
                ),
                "pseudo_source_fail_reasons": dict(
                    getattr(filt, "last_pseudo_source_fail_reasons", {})
                ),
                "structural_timing_s": dict(
                    getattr(filt, "last_structural_timing_s", {})
                ),
                "temper_steps": list(getattr(filt, "last_temper_steps", [])),
                "temper_resamples": int(getattr(filt, "last_temper_resample_count", 0)),
                "r_mean": r_mean,
                "r_var": r_var,
                "r_weighted_mean": r_weighted_mean,
                "r_weighted_var": r_weighted_var,
                "r_probability_by_count": r_probability_by_count,
                "r_particle_count_by_count": r_particle_count_by_count,
                "map": (map_positions, map_strengths),
                "mmse": (mmse_positions, mmse_strengths),
                "top_k": top_entries,
                "converged": bool(getattr(filt, "is_converged", False)),
                "updates_skipped": int(getattr(filt, "updates_skipped", 0)),
            }
        return diagnostics

    def isotope_log_likelihood_gain(
        self, window: int | None = None
    ) -> Dict[str, float]:
        """
        Return per-isotope log-likelihood gain vs background-only (evidence mixing).
        """
        if not self.measurements:
            return {iso: 0.0 for iso in self.filters}
        estimates = self.pruned_estimates(method="deltall")
        gains: Dict[str, float] = {}
        for iso, filt in self.filters.items():
            data = self._measurement_data_for_iso(iso, window)
            if data is None or data.z_k.size == 0:
                gains[iso] = 0.0
                continue
            positions, strengths = estimates.get(iso, (np.zeros((0, 3)), np.zeros(0)))
            if filt.continuous_particles:
                background_rate = float(filt.best_particle().state.background)
            else:
                background_rate = 0.0
            background_counts = background_rate * data.live_times
            if positions.size == 0:
                gains[iso] = 0.0
                continue
            lambda_m = self._cached_expected_counts_per_source(
                filt=filt,
                isotope=iso,
                data=data,
                sources=positions,
                strengths=strengths,
            )
            lambda_total = background_counts + np.sum(lambda_m, axis=1)
            ll = filt._count_log_likelihood_np(
                data.z_k,
                lambda_total,
                observation_count_variance=data.observation_variances,
            )
            ll_bg = filt._count_log_likelihood_np(
                data.z_k,
                background_counts,
                observation_count_variance=data.observation_variances,
            )
            gains[iso] = float(ll - ll_bg)
        return gains

    def isotopes_by_evidence(
        self, min_delta_ll: float = 0.0, window: int | None = None
    ) -> List[str]:
        """
        Return isotopes whose LL gain exceeds min_delta_ll for the given window.
        """
        gains = self.isotope_log_likelihood_gain(window=window)
        return [iso for iso, gain in gains.items() if gain >= float(min_delta_ll)]

    @property
    def num_orientations(self) -> int:
        """Return the number of shield orientation normals."""
        return self.normals.shape[0]

    def orientation_information_gain(
        self, pose_idx: int, orient_idx: int, live_time_s: float = 1.0
    ) -> float:
        """
        Information gain surrogate using Eq. (3.40)–(3.42) style variance ratio.

        IG_k(phi) ~= 0.5 * log(1 + Var[Lambda_k(phi)] / E[Lambda_k(phi)]) aggregated over isotopes.
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        ig_total = 0.0
        eps = 1e-9
        for iso, filt in self.filters.items():
            if getattr(filt, "is_converged", False) and getattr(
                filt.config, "converge_enable", False
            ):
                continue
            use_continuous = bool(filt.continuous_particles)
            if use_continuous:
                lam = filt._continuous_expected_counts(
                    pose_idx=pose_idx, orient_idx=orient_idx, live_time_s=live_time_s
                )
                w = filt.continuous_weights
            else:
                lam = np.zeros(0, dtype=float)
                w = np.zeros(0, dtype=float)
            mean = float(np.sum(w * lam))
            var = float(np.sum(w * (lam - mean) ** 2))
            ig_total += 0.5 * float(np.log1p(var / max(mean, eps)))
        return ig_total

    def max_orientation_information_gain(
        self, pose_idx: int, live_time_s: float = 1.0
    ) -> float:
        """Return max_phi IG_k(phi) at pose k (Eq. 3.45 surrogate)."""
        scores = [
            self.orientation_information_gain(
                pose_idx=pose_idx, orient_idx=oidx, live_time_s=live_time_s
            )
            for oidx in range(self.num_orientations)
        ]
        return float(np.max(scores)) if scores else 0.0

    def orientation_expected_information_gain(
        self,
        pose_idx: int,
        RFe: NDArray[np.float64],
        RPb: NDArray[np.float64],
        live_time_s: float = 1.0,
        num_samples: int | None = None,
        alpha_by_isotope: Dict[str, float] | None = None,
        particles_by_isotope: Dict[str, Tuple[List[IsotopeState], NDArray[np.float64]]]
        | None = None,
        rng: np.random.Generator | None = None,
        detector_pos: NDArray[np.float64] | None = None,
    ) -> float:
        """
        Monte-Carlo approximation of EIG (Eq. 3.44) for a Fe/Pb orientation pair.

        - Uses continuous particles and ContinuousKernel expected counts (Eq. 3.41).
        - For each isotope h: IG_h = H(w_h) - E_z[H(w'_h(z; RFe, RPb))].
        - Global IG = Σ_h α_h IG_h, with α_h uniform if not provided.
        - If detector_pos is provided, pose_idx is ignored.
        """
        if detector_pos is None:
            if self.kernel_cache is None:
                self._ensure_kernel_cache()
            detector_pos = self.kernel_cache.poses[pose_idx]
        detector_pos = np.asarray(detector_pos, dtype=float)
        rng = rng or np.random.default_rng()
        num_samples = (
            self.pf_config.eig_num_samples if num_samples is None else num_samples
        )
        eps = 1e-12
        fe_idx = octant_index_from_rotation(RFe)
        pb_idx = octant_index_from_rotation(RPb)
        if not self._can_use_gpu():
            return self._orientation_expected_information_gain_cpu(
                pose_idx=pose_idx,
                detector_pos=detector_pos,
                fe_idx=fe_idx,
                pb_idx=pb_idx,
                live_time_s=live_time_s,
                num_samples=int(num_samples),
                alpha_by_isotope=alpha_by_isotope,
                particles_by_isotope=particles_by_isotope,
                rng=rng,
                eps=eps,
            )
        kernel = self._continuous_kernel()
        alphas = alpha_by_isotope or {iso: 1.0 for iso in self.filters}
        # normalize alphas
        alpha_sum = sum(alphas.values()) or 1.0
        alphas = {k: v / alpha_sum for k, v in alphas.items()}
        self._gpu_enabled()
        from pf import gpu_utils as gpu_mod
        import torch as torch_mod

        gpu_utils = gpu_mod
        device = gpu_utils.resolve_device(self.pf_config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.pf_config.gpu_dtype)
        torch = torch_mod

        def _compute_lam_torch(
            states: Sequence[IsotopeState], isotope: str
        ) -> "torch.Tensor":
            """Compute expected counts for a state subset on the torch backend."""
            if not states:
                return torch.zeros(0, device=device, dtype=dtype)
            positions, strengths, backgrounds, mask = gpu_utils.pack_states(
                states, device=device, dtype=dtype
            )
            return kernel.expected_counts_pair_for_packed_states_torch(
                isotope=isotope,
                detector_pos=detector_pos,
                positions=positions,
                strengths=strengths,
                backgrounds=backgrounds,
                mask=mask,
                fe_index=fe_idx,
                pb_index=pb_idx,
                live_time_s=live_time_s,
                source_scale=self.response_scale_for_isotope(
                    isotope,
                    fe_index=fe_idx,
                    pb_index=pb_idx,
                ),
                device=device,
                dtype=dtype,
            )

        total_ig = 0.0
        for iso, filt in self.filters.items():
            if getattr(filt, "is_converged", False) and getattr(
                filt.config, "converge_enable", False
            ):
                continue
            if particles_by_isotope is not None and iso in particles_by_isotope:
                states, weights = particles_by_isotope[iso]
            else:
                if not filt.continuous_particles:
                    continue
                states = [p.state for p in filt.continuous_particles]
                weights = filt.continuous_weights
            if not states:
                continue
            weights = np.asarray(weights, dtype=float)
            weights = weights / max(np.sum(weights), eps)
            lam_t = _compute_lam_torch(states, iso)
            weights_t = torch.as_tensor(weights, device=device, dtype=dtype)
            weight_sum = torch.sum(weights_t)
            if float(weight_sum) <= 0.0:
                weights_t = torch.full_like(weights_t, 1.0 / max(weights_t.numel(), 1))
            else:
                weights_t = weights_t / weight_sum
            H_prior = -torch.sum(weights_t * torch.log(weights_t + eps))
            if num_samples <= 0:
                H_post_mean = torch.zeros((), device=device, dtype=dtype)
            else:
                idx = torch.multinomial(weights_t, num_samples, replacement=True)
                z = torch.poisson(lam_t[idx])
                logw = (
                    torch.log(weights_t + eps)
                    + z.unsqueeze(1) * torch.log(lam_t + eps)
                    - lam_t
                )
                logw = logw - torch.logsumexp(logw, dim=1, keepdim=True)
                w_post = torch.exp(logw)
                H_post = -torch.sum(w_post * torch.log(w_post + eps), dim=1)
                H_post_mean = torch.mean(H_post)
            ig_h = float((H_prior - H_post_mean).item())
            total_ig += alphas.get(iso, 0.0) * ig_h
        return float(total_ig)

    def orientation_expected_information_gain_grid(
        self,
        pose_idx: int,
        *,
        live_time_s: float = 1.0,
        num_samples: int | None = None,
        alpha_by_isotope: Dict[str, float] | None = None,
        particles_by_isotope: Dict[str, Tuple[List[IsotopeState], NDArray[np.float64]]]
        | None = None,
    ) -> NDArray[np.float64]:
        """
        Compute MC-EIG for all Fe/Pb orientation pairs using shared lambdas.

        This evaluates the same likelihood-entropy estimator as
        ``orientation_expected_information_gain`` but avoids recomputing the
        continuous expected-count kernel separately for every orientation pair.
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        num_orients = int(self.num_orientations)
        num_pairs = num_orients * num_orients
        if not self._can_use_gpu():
            from measurement.shielding import generate_octant_rotation_matrices

            rot_mats = generate_octant_rotation_matrices()
            scores = np.zeros((num_orients, num_orients), dtype=float)
            for fe_idx, RFe in enumerate(rot_mats[:num_orients]):
                for pb_idx, RPb in enumerate(rot_mats[:num_orients]):
                    scores[fe_idx, pb_idx] = self.orientation_expected_information_gain(
                        pose_idx=pose_idx,
                        RFe=RFe,
                        RPb=RPb,
                        live_time_s=live_time_s,
                        num_samples=num_samples,
                        alpha_by_isotope=alpha_by_isotope,
                        particles_by_isotope=particles_by_isotope,
                    )
            return scores

        detector_pos = np.asarray(self.kernel_cache.poses[int(pose_idx)], dtype=float)
        num_samples = (
            self.pf_config.eig_num_samples if num_samples is None else num_samples
        )
        eps = 1e-12
        alphas = alpha_by_isotope or {iso: 1.0 for iso in self.filters}
        alpha_sum = sum(float(v) for v in alphas.values()) or 1.0
        alphas = {key: float(value) / alpha_sum for key, value in alphas.items()}
        self._gpu_enabled()
        from pf import gpu_utils
        import torch

        device = gpu_utils.resolve_device(self.pf_config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.pf_config.gpu_dtype)
        iso_data: Dict[str, tuple["torch.Tensor", "torch.Tensor"]] = {}
        for iso, filt in self.filters.items():
            if getattr(filt, "is_converged", False) and getattr(
                filt.config, "converge_enable", False
            ):
                continue
            if particles_by_isotope is not None and iso in particles_by_isotope:
                states, weights = particles_by_isotope[iso]
            else:
                if not filt.continuous_particles:
                    continue
                states = [p.state for p in filt.continuous_particles]
                weights = filt.continuous_weights
            if not states:
                continue
            weights_arr = np.asarray(weights, dtype=float)
            weight_sum = float(np.sum(weights_arr))
            if weight_sum <= eps:
                weights_arr = np.ones(len(states), dtype=float) / max(len(states), 1)
            else:
                weights_arr = weights_arr / weight_sum
            lambdas_np = self.expected_counts_all_pairs_for_states_at_detector(
                isotope=iso,
                detector_pos=detector_pos,
                live_time_s=float(live_time_s),
                states=states,
            )
            lam_all = torch.as_tensor(lambdas_np, device=device, dtype=dtype)
            weights_t = torch.as_tensor(weights_arr, device=device, dtype=dtype)
            weight_sum_t = torch.sum(weights_t)
            if float(weight_sum_t.detach().cpu().item()) <= 0.0:
                weights_t = torch.full_like(weights_t, 1.0 / max(weights_t.numel(), 1))
            else:
                weights_t = weights_t / weight_sum_t
            iso_data[iso] = (lam_all, weights_t)
        if not iso_data:
            return np.zeros((num_orients, num_orients), dtype=float)

        scores = np.zeros(num_pairs, dtype=float)
        for pair_idx in range(num_pairs):
            total_ig = 0.0
            for iso, (lam_all, weights_t) in iso_data.items():
                lam_t = lam_all[pair_idx]
                log_weights = torch.log(weights_t + eps)
                h_prior = -torch.sum(weights_t * log_weights)
                if int(num_samples) <= 0:
                    h_post_mean = torch.zeros((), device=device, dtype=dtype)
                else:
                    idx = torch.multinomial(
                        weights_t, int(num_samples), replacement=True
                    )
                    z = torch.poisson(lam_t[idx])
                    logw = (
                        log_weights.view(1, -1)
                        + z.unsqueeze(1) * torch.log(lam_t + eps).view(1, -1)
                        - lam_t.view(1, -1)
                    )
                    logw = logw - torch.logsumexp(logw, dim=1, keepdim=True)
                    w_post = torch.exp(logw)
                    h_post = -torch.sum(w_post * torch.log(w_post + eps), dim=1)
                    h_post_mean = torch.mean(h_post)
                total_ig += float(alphas.get(iso, 0.0)) * float(
                    (h_prior - h_post_mean).item()
                )
            scores[pair_idx] = total_ig
        return scores.reshape(num_orients, num_orients)

    def _orientation_expected_information_gain_cpu(
        self,
        *,
        pose_idx: int,
        detector_pos: NDArray[np.float64],
        fe_idx: int,
        pb_idx: int,
        live_time_s: float,
        num_samples: int,
        alpha_by_isotope: Dict[str, float] | None,
        particles_by_isotope: Dict[str, Tuple[List[IsotopeState], NDArray[np.float64]]]
        | None,
        rng: np.random.Generator,
        eps: float,
    ) -> float:
        """Compute orientation EIG on CPU using the same expected-count kernel."""
        alphas = alpha_by_isotope or {iso: 1.0 for iso in self.filters}
        alpha_sum = sum(float(value) for value in alphas.values()) or 1.0
        alphas = {key: float(value) / alpha_sum for key, value in alphas.items()}
        total_ig = 0.0
        for iso, filt in self.filters.items():
            if getattr(filt, "is_converged", False) and getattr(
                filt.config, "converge_enable", False
            ):
                continue
            if particles_by_isotope is not None and iso in particles_by_isotope:
                states, weights = particles_by_isotope[iso]
            else:
                if not filt.continuous_particles:
                    continue
                states = [p.state for p in filt.continuous_particles]
                weights = filt.continuous_weights
            if not states:
                continue
            weights_arr = np.asarray(weights, dtype=float)
            weights_arr = weights_arr / max(float(np.sum(weights_arr)), eps)
            lam = self.expected_counts_pair_for_states_at_detector(
                isotope=iso,
                detector_pos=detector_pos,
                fe_index=int(fe_idx),
                pb_index=int(pb_idx),
                live_time_s=float(live_time_s),
                states=states,
            )
            lam = np.maximum(np.asarray(lam, dtype=float).reshape(-1), eps)
            h_prior = -float(np.sum(weights_arr * np.log(weights_arr + eps)))
            if num_samples <= 0:
                h_post_mean = 0.0
            else:
                sample_indices = rng.choice(
                    weights_arr.size,
                    size=int(num_samples),
                    replace=True,
                    p=weights_arr,
                )
                z_samples = rng.poisson(lam[sample_indices])
                logw = (
                    np.log(weights_arr + eps)[None, :]
                    + z_samples[:, None] * np.log(lam + eps)[None, :]
                    - lam[None, :]
                )
                logw -= logw.max(axis=1, keepdims=True)
                w_post = np.exp(logw)
                w_post /= np.maximum(np.sum(w_post, axis=1, keepdims=True), eps)
                h_post = -np.sum(w_post * np.log(w_post + eps), axis=1)
                h_post_mean = float(np.mean(h_post))
            total_ig += alphas.get(iso, 0.0) * (h_prior - h_post_mean)
        return float(total_ig)

    def _strength_matrix(self, filt: IsotopeParticleFilter) -> NDArray[np.float64]:
        """
        Build a (N, max_r) matrix of source strengths for variance computation (Eq. 3.38 surrogate).
        """
        max_r = max((p.state.num_sources for p in filt.continuous_particles), default=0)
        mat = np.zeros((len(filt.continuous_particles), max_r), dtype=float)
        for i, p in enumerate(filt.continuous_particles):
            r = p.state.num_sources
            if r > 0:
                mat[i, :r] = p.state.strengths
        return mat

    def expected_uncertainty_after_pose(
        self,
        pose_idx: int,
        fe_index: int | None = None,
        pb_index: int | None = None,
        orient_idx: int = 0,
        live_time_s: float = 1.0,
        num_samples: int = 20,
        rng: np.random.Generator | None = None,
    ) -> float:
        """
        Monte-Carlo estimate of E[U | q_cand] where U = Σ_h Σ_m Var(q_{h,m}) (Eq. 3.38 surrogate).

        Draw hypothetical Poisson observations at pose q_cand and average posterior variance of strengths.
        Uses either Fe/Pb indices (if provided) or orient_idx into the kernel orientations.
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        rng = rng or np.random.default_rng()
        eps = 1e-12
        total_U = 0.0
        for iso, filt in self.filters.items():
            if not filt.continuous_particles:
                continue
            weights = filt.continuous_weights
            if fe_index is not None and pb_index is not None:
                lam = filt._continuous_expected_counts_pair(
                    pose_idx=pose_idx,
                    fe_index=fe_index,
                    pb_index=pb_index,
                    live_time_s=live_time_s,
                )
            else:
                lam = filt._continuous_expected_counts(
                    pose_idx=pose_idx, orient_idx=orient_idx, live_time_s=live_time_s
                )
            strengths_mat = self._strength_matrix(filt)
            U_accum = 0.0
            for _ in range(num_samples):
                n = int(rng.choice(len(lam), p=weights))
                z = rng.poisson(lam[n])
                logw = np.log(weights + eps) + z * np.log(lam + eps) - lam
                logw -= logsumexp(logw)
                w_post = np.exp(logw)
                if strengths_mat.size == 0:
                    continue
                mean = np.sum(w_post[:, None] * strengths_mat, axis=0)
                var = np.sum(w_post[:, None] * (strengths_mat - mean) ** 2, axis=0)
                U_accum += float(np.sum(var))
            total_U += U_accum / max(num_samples, 1)
        return float(total_U)

    def expected_uncertainty_after_pose_xyz(
        self,
        pose_xyz: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        live_time_s: float = 1.0,
        num_samples: int = 20,
        rng: np.random.Generator | None = None,
    ) -> float:
        """
        Monte-Carlo estimate of E[U | pose_xyz] for an explicit detector position.

        Uses Fe/Pb indices to compute expected counts without relying on pose indices.
        """
        detector_pos = np.asarray(pose_xyz, dtype=float)
        if detector_pos.shape != (3,):
            raise ValueError("pose_xyz must be shape (3,).")
        rng = rng or np.random.default_rng()
        num_samples = max(int(num_samples), 1)
        eps = 1e-12
        total_U = 0.0
        for iso, filt in self.filters.items():
            if not filt.continuous_particles:
                continue
            weights = np.asarray(filt.continuous_weights, dtype=float)
            if weights.size == 0:
                continue
            weights = weights / max(np.sum(weights), eps)
            lam = filt._continuous_expected_counts_pair_at_pose(
                detector_pos=detector_pos,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
            )
            if lam.size == 0:
                continue
            strengths_mat = self._strength_matrix(filt)
            U_accum = 0.0
            for _ in range(num_samples):
                n = int(rng.choice(len(lam), p=weights))
                z = rng.poisson(lam[n])
                logw = np.log(weights + eps) + z * np.log(lam + eps) - lam
                logw -= logsumexp(logw)
                w_post = np.exp(logw)
                if strengths_mat.size == 0:
                    continue
                mean = np.sum(w_post[:, None] * strengths_mat, axis=0)
                var = np.sum(w_post[:, None] * (strengths_mat - mean) ** 2, axis=0)
                U_accum += float(np.sum(var))
            total_U += U_accum / max(num_samples, 1)
        return float(total_U)

    def expected_uncertainty_after_rotation(
        self,
        pose_xyz: NDArray[np.float64],
        live_time_per_rot_s: float,
        tau_ig: float,
        tmax_s: float,
        n_rollouts: int = 64,
        orient_selection: str = "IG",
        return_debug: bool = False,
        rng_seed: int | None = None,
    ) -> float | Tuple[float, Dict[str, Any]]:
        """
        Estimate E[U_after-rotation | pose_xyz] by Monte Carlo rollouts.

        This method has no side effects on the estimator state. Rotation policy:
        - choose the next orientation by maximizing IG
        - stop if max IG < tau_ig
        - stop if accumulated live time reaches tmax_s

        rng_seed can be set to make rollouts deterministic for debugging.
        """
        detector_pos = np.asarray(pose_xyz, dtype=float)
        if detector_pos.shape != (3,):
            raise ValueError("pose_xyz must be shape (3,).")
        if orient_selection.lower() != "ig":
            raise ValueError("Only orient_selection='IG' is supported.")
        n_rollouts = int(n_rollouts)
        use_mean_measurement = n_rollouts <= 0
        rollouts = max(1, n_rollouts)
        if rng_seed is None:
            rng = np.random.default_rng(np.random.randint(0, 2**32 - 1))
        else:
            rng = np.random.default_rng(int(rng_seed))
        from measurement.shielding import generate_octant_rotation_matrices

        RFe_candidates = generate_octant_rotation_matrices()
        RPb_candidates = generate_octant_rotation_matrices()
        num_fe = len(RFe_candidates)
        num_pb = len(RPb_candidates)
        alphas = self.pf_config.alpha_weights
        eig_samples = (
            self.pf_config.planning_eig_samples
            if self.pf_config.planning_eig_samples is not None
            else self.pf_config.eig_num_samples
        )
        rollout_particles = self.pf_config.planning_rollout_particles
        if rollout_particles is None:
            rollout_particles = self.pf_config.planning_particles
        rollout_method = (
            self.pf_config.planning_rollout_method or self.pf_config.planning_method
        )

        fast_result = self._expected_uncertainty_after_rotation_fast(
            detector_pos=detector_pos,
            live_time_per_rot_s=live_time_per_rot_s,
            tau_ig=tau_ig,
            tmax_s=tmax_s,
            rollouts=rollouts,
            eig_samples=eig_samples,
            alpha_by_isotope=alphas,
            rollout_particles=rollout_particles,
            rollout_method=rollout_method,
            use_mean_measurement=use_mean_measurement,
            rng=rng,
            return_debug=return_debug,
        )
        if fast_result is not None:
            return fast_result

        def _select_best_orientation(
            estimator: "RotatingShieldPFEstimator", rng_local: np.random.Generator
        ) -> Tuple[int, int, float]:
            """Return the (fe_idx, pb_idx) pair with the maximum EIG at the given pose."""
            best_ig = -np.inf
            best_fe = 0
            best_pb = 0
            particles_by_iso = None
            if rollout_particles is not None and rollout_particles > 0:
                particles_by_iso = estimator.planning_particles(
                    max_particles=int(rollout_particles),
                    method=rollout_method,
                    rng=rng_local,
                )
            for fe_idx in range(num_fe):
                for pb_idx in range(num_pb):
                    ig_val = estimator.orientation_expected_information_gain(
                        pose_idx=0,
                        RFe=RFe_candidates[fe_idx],
                        RPb=RPb_candidates[pb_idx],
                        live_time_s=live_time_per_rot_s,
                        num_samples=eig_samples,
                        alpha_by_isotope=alphas,
                        particles_by_isotope=particles_by_iso,
                        rng=rng_local,
                        detector_pos=detector_pos,
                    )
                    if ig_val > best_ig:
                        best_ig = ig_val
                        best_fe = fe_idx
                        best_pb = pb_idx
            return best_fe, best_pb, float(best_ig)

        def _simulate_measurement(
            estimator: "RotatingShieldPFEstimator",
            fe_idx: int,
            pb_idx: int,
            rng_local: np.random.Generator,
        ) -> Dict[str, float]:
            """Simulate isotope-wise Poisson observations at the candidate pose."""
            z_k: Dict[str, float] = {}
            for iso, filt in estimator.filters.items():
                if not filt.continuous_particles:
                    z_k[iso] = 0.0
                    continue
                lam = filt._continuous_expected_counts_pair_at_pose(
                    detector_pos=detector_pos,
                    fe_index=fe_idx,
                    pb_index=pb_idx,
                    live_time_s=live_time_per_rot_s,
                )
                if lam.size == 0:
                    z_k[iso] = 0.0
                    continue
                weights = filt.continuous_weights
                if use_mean_measurement:
                    z_k[iso] = float(np.sum(weights * lam))
                else:
                    idx = int(rng_local.choice(len(lam), p=weights))
                    z_k[iso] = float(rng_local.poisson(lam[idx]))
            return z_k

        def _run_once(
            estimator: "RotatingShieldPFEstimator", rng_local: np.random.Generator
        ) -> Tuple[float, Dict[str, Any]]:
            """Run a single rotation rollout and return uncertainty plus debug metadata."""
            elapsed = 0.0
            rotations = 0
            iterations: List[Dict[str, Any]] = []
            while elapsed < tmax_s:
                fe_idx, pb_idx, ig_val = _select_best_orientation(estimator, rng_local)
                iterations.append(
                    {
                        "fe_idx": fe_idx,
                        "pb_idx": pb_idx,
                        "ig": ig_val,
                        "elapsed": elapsed,
                    }
                )
                if ig_val < tau_ig:
                    break
                z_k = _simulate_measurement(estimator, fe_idx, pb_idx, rng_local)
                for iso, val in z_k.items():
                    if iso not in estimator.filters:
                        continue
                    estimator.filters[iso].update_continuous_pair_at_pose(
                        z_obs=val,
                        detector_pos=detector_pos,
                        fe_index=fe_idx,
                        pb_index=pb_idx,
                        live_time_s=live_time_per_rot_s,
                    )
                elapsed += live_time_per_rot_s
                rotations += 1
            return estimator.global_uncertainty(), {
                "iterations": iterations,
                "elapsed": elapsed,
                "num_rotations": rotations,
            }

        u_vals: List[float] = []
        debug_rollouts: List[Dict[str, Any]] = []
        for _ in range(rollouts):
            estimator_copy = copy.deepcopy(self)
            u_val, debug = _run_once(estimator_copy, rng)
            u_vals.append(u_val)
            debug_rollouts.append(debug)
        mean_u = float(np.mean(u_vals)) if u_vals else 0.0
        if return_debug:
            debug_payload = {"rollouts": debug_rollouts, "u_vals": u_vals}
            return mean_u, debug_payload
        return mean_u

    def _expected_uncertainty_after_rotation_fast(
        self,
        detector_pos: NDArray[np.float64],
        live_time_per_rot_s: float,
        tau_ig: float,
        tmax_s: float,
        rollouts: int,
        eig_samples: int,
        alpha_by_isotope: Dict[str, float] | None,
        rollout_particles: int | None,
        rollout_method: str | None,
        use_mean_measurement: bool,
        rng: np.random.Generator,
        return_debug: bool,
    ) -> float | Tuple[float, Dict[str, Any]] | None:
        """
        Fast GPU rollout evaluation using precomputed lambdas and index-based updates.

        Returns None when the fast path cannot be used.
        """
        if not self.pf_config.use_fast_gpu_rollout:
            return None
        if not self.pf_config.use_gpu:
            return None
        self._gpu_enabled()
        from pf import gpu_utils
        import torch
        from measurement.shielding import generate_octant_rotation_matrices

        RFe_candidates = generate_octant_rotation_matrices()
        RPb_candidates = generate_octant_rotation_matrices()
        num_fe = len(RFe_candidates)
        num_pb = len(RPb_candidates)
        fe_indices = np.repeat(np.arange(num_fe), num_pb)
        pb_indices = np.tile(np.arange(num_pb), num_fe)
        eps = 1e-12
        alphas = alpha_by_isotope or {iso: 1.0 for iso in self.filters}
        alpha_sum = sum(alphas.values()) or 1.0
        alphas = {k: v / alpha_sum for k, v in alphas.items()}
        device = gpu_utils.resolve_device(self.pf_config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.pf_config.gpu_dtype)
        planning_subset = self.planning_particles(
            max_particles=rollout_particles,
            method=rollout_method,
            rng=rng,
        )

        iso_data: Dict[str, Dict[str, Any]] = {}
        for iso, filt in self.filters.items():
            if not filt.continuous_particles:
                continue
            if iso in planning_subset and planning_subset[iso][0]:
                states, weights = planning_subset[iso]
            else:
                states = [p.state for p in filt.continuous_particles]
                weights = np.asarray(filt.continuous_weights, dtype=float)
            weights = np.asarray(weights, dtype=float)
            if weights.size == 0 or not states:
                continue
            weights = weights / max(np.sum(weights), eps)
            positions, strengths, backgrounds, mask = gpu_utils.pack_states(
                states, device=device, dtype=dtype
            )
            lam_all = filt.continuous_kernel.expected_counts_all_pairs_for_packed_states_torch(
                isotope=iso,
                detector_pos=detector_pos,
                positions=positions,
                strengths=strengths,
                backgrounds=backgrounds,
                mask=mask,
                live_time_s=live_time_per_rot_s,
                source_scale=self.response_scales_for_measurements(
                    iso,
                    fe_indices,
                    pb_indices,
                ),
                device=device,
                dtype=dtype,
            )
            iso_data[iso] = {
                "lam": lam_all,
                "strengths": strengths,
                "weights": weights,
                "num_particles": weights.size,
                "resample_threshold": filt.config.resample_threshold,
            }
        if not iso_data:
            return 0.0 if not return_debug else (0.0, {"rollouts": [], "u_vals": []})

        def _select_subset(
            weights: NDArray[np.float64],
            indices: NDArray[np.int64],
            max_particles: int | None,
            method: str | None,
            rng_local: np.random.Generator,
        ) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
            """Return subset indices and normalized weights for EIG evaluation."""
            if (
                max_particles is None
                or max_particles <= 0
                or max_particles >= len(weights)
            ):
                return indices, weights
            method = method or "top_weight"
            if method == "top_weight":
                sel = np.argsort(weights)[::-1][:max_particles]
                sel_weights = weights[sel]
                sel_weights = sel_weights / max(np.sum(sel_weights), eps)
                return indices[sel], sel_weights
            if method == "resample":
                sel = rng_local.choice(len(weights), size=max_particles, p=weights)
                sel_weights = np.ones(max_particles, dtype=float) / max(
                    max_particles, 1
                )
                return indices[sel], sel_weights
            raise ValueError(f"Unknown planning particle selection method: {method}")

        def _ig_scores_from_lam(
            lam_all: "torch.Tensor",
            subset_indices: NDArray[np.int64],
            subset_weights: NDArray[np.float64],
            num_samples: int,
        ) -> "torch.Tensor":
            """Compute IG scores for all orientations from precomputed lambdas."""
            if num_samples <= 0:
                weights_t = torch.as_tensor(
                    subset_weights, device=lam_all.device, dtype=lam_all.dtype
                )
                weights_t = weights_t / torch.sum(weights_t)
                h_prior = -torch.sum(weights_t * torch.log(weights_t + eps))
                return torch.full(
                    (lam_all.shape[0],),
                    h_prior,
                    device=lam_all.device,
                    dtype=lam_all.dtype,
                )
            idx_t = torch.as_tensor(
                subset_indices, device=lam_all.device, dtype=torch.long
            )
            lam_sel = torch.index_select(lam_all, 1, idx_t)
            weights_t = torch.as_tensor(
                subset_weights, device=lam_all.device, dtype=lam_all.dtype
            )
            weights_t = weights_t / torch.sum(weights_t)
            log_weights = torch.log(weights_t + eps)
            h_prior = -torch.sum(weights_t * log_weights)
            weights_row = weights_t.expand(lam_sel.shape[0], -1)
            idx_samples = torch.multinomial(weights_row, num_samples, replacement=True)
            lam_samples = torch.gather(lam_sel, 1, idx_samples)
            z = torch.poisson(lam_samples)
            log_lam = torch.log(lam_sel + eps)
            logw = (
                log_weights.view(1, 1, -1)
                + z.unsqueeze(2) * log_lam.unsqueeze(1)
                - lam_sel.unsqueeze(1)
            )
            logw = logw - torch.logsumexp(logw, dim=2, keepdim=True)
            w_post = torch.exp(logw)
            h_post = -torch.sum(w_post * torch.log(w_post + eps), dim=2)
            h_post_mean = torch.mean(h_post, dim=1)
            return h_prior - h_post_mean

        def _update_weights(
            lam_curr: NDArray[np.float64],
            weights: NDArray[np.float64],
            z_obs: float,
        ) -> NDArray[np.float64]:
            """Update weights using Poisson log-likelihood and normalize."""
            logw = np.log(weights + eps) + z_obs * np.log(lam_curr + eps) - lam_curr
            logw -= np.max(logw)
            w = np.exp(logw)
            total = np.sum(w)
            if total <= 0.0:
                return np.ones_like(weights) / max(len(weights), 1)
            return w / total

        u_vals: List[float] = []
        debug_rollouts: List[Dict[str, Any]] = []
        for _ in range(int(rollouts)):
            weights_by_iso: Dict[str, NDArray[np.float64]] = {}
            indices_by_iso: Dict[str, NDArray[np.int64]] = {}
            for iso, data in iso_data.items():
                n_particles = int(data["num_particles"])
                weights_by_iso[iso] = data["weights"].copy()
                indices_by_iso[iso] = np.arange(n_particles, dtype=int)
            elapsed = 0.0
            iterations: List[Dict[str, Any]] = []
            while elapsed < tmax_s:
                total_ig: "torch.Tensor" | None = None
                for iso, data in iso_data.items():
                    weights = weights_by_iso[iso]
                    indices = indices_by_iso[iso]
                    if weights.size == 0:
                        continue
                    subset_idx, subset_w = _select_subset(
                        weights=weights,
                        indices=indices,
                        max_particles=rollout_particles,
                        method=rollout_method,
                        rng_local=rng,
                    )
                    if subset_w.size == 0:
                        continue
                    ig_scores = _ig_scores_from_lam(
                        lam_all=data["lam"],
                        subset_indices=subset_idx,
                        subset_weights=subset_w,
                        num_samples=int(eig_samples),
                    )
                    weight = float(alphas.get(iso, 0.0))
                    ig_scores = ig_scores * weight
                    if total_ig is None:
                        total_ig = ig_scores
                    else:
                        total_ig = total_ig + ig_scores
                if total_ig is None:
                    break
                best_orient = int(torch.argmax(total_ig).item())
                best_ig = float(total_ig[best_orient].detach().cpu().item())
                iterations.append(
                    {
                        "fe_idx": int(fe_indices[best_orient]),
                        "pb_idx": int(pb_indices[best_orient]),
                        "ig": best_ig,
                        "elapsed": elapsed,
                    }
                )
                if best_ig < tau_ig:
                    break
                for iso, data in iso_data.items():
                    weights = weights_by_iso[iso]
                    indices = indices_by_iso[iso]
                    if weights.size == 0:
                        continue
                    idx_t = torch.as_tensor(indices, device=device, dtype=torch.long)
                    lam_curr_t = torch.index_select(data["lam"][best_orient], 0, idx_t)
                    lam_curr = lam_curr_t.detach().cpu().numpy()
                    if lam_curr.size == 0:
                        continue
                    if use_mean_measurement:
                        z_obs = float(np.sum(weights * lam_curr))
                    else:
                        idx = int(rng.choice(len(lam_curr), p=weights))
                        z_obs = float(rng.poisson(lam_curr[idx]))
                    weights = _update_weights(lam_curr, weights, z_obs)
                    ess = 1.0 / max(np.sum(weights**2), eps)
                    if ess < float(data["resample_threshold"]) * len(weights):
                        resampled = systematic_resample(np.log(weights + eps))
                        indices = indices[resampled]
                        weights = np.ones_like(weights) / max(len(weights), 1)
                    weights_by_iso[iso] = weights
                    indices_by_iso[iso] = indices
                elapsed += live_time_per_rot_s
            total_u = 0.0
            for iso, data in iso_data.items():
                weights = weights_by_iso[iso]
                indices = indices_by_iso[iso]
                if weights.size == 0:
                    continue
                idx_t = torch.as_tensor(indices, device=device, dtype=torch.long)
                strengths_t = torch.index_select(data["strengths"], 0, idx_t)
                weights_t = torch.as_tensor(weights, device=device, dtype=dtype)
                weights_t = weights_t / torch.sum(weights_t)
                mean = torch.sum(weights_t[:, None] * strengths_t, dim=0)
                var = torch.sum(weights_t[:, None] * (strengths_t - mean) ** 2, dim=0)
                total_u += float(torch.sum(var).detach().cpu().item())
            u_vals.append(total_u)
            debug_rollouts.append(
                {
                    "iterations": iterations,
                    "elapsed": elapsed,
                    "num_rotations": len(iterations),
                }
            )
        mean_u = float(np.mean(u_vals)) if u_vals else 0.0
        if return_debug:
            debug_payload = {"rollouts": debug_rollouts, "u_vals": u_vals}
            return mean_u, debug_payload
        return mean_u

    def expected_uncertainty_after_rotation_at_pose(
        self,
        detector_pos: NDArray[np.float64],
        *,
        tau_ig: float,
        t_max_s: float,
        t_short_s: float,
        num_rollouts: int = 0,
        use_mean_measurement: bool = True,
        rng_seed: int | None = 0,
        return_debug: bool = False,
    ) -> float | Tuple[float, Dict[str, Any]]:
        """
        Backward-compatible wrapper for expected_uncertainty_after_rotation.
        """
        n_rollouts = int(num_rollouts)
        if n_rollouts <= 0 and not use_mean_measurement:
            n_rollouts = 1
        if rng_seed is not None:
            np.random.seed(rng_seed)
        return self.expected_uncertainty_after_rotation(
            pose_xyz=detector_pos,
            live_time_per_rot_s=t_short_s,
            tau_ig=tau_ig,
            tmax_s=t_max_s,
            n_rollouts=n_rollouts,
            orient_selection="IG",
            return_debug=return_debug,
            rng_seed=rng_seed,
        )

    def estimate_change_norm(self) -> float:
        """
        Return ||Δs|| + ||Δq|| between the last two estimates (Sec. 3.6 convergence check).
        """
        if len(self.history_estimates) < 2:
            return float("inf")
        prev = self.history_estimates[-2]
        curr = self.history_estimates[-1]
        diff = 0.0
        for iso in self.isotopes:
            prev_pos, prev_str = prev.get(iso, (None, None))
            curr_pos, curr_str = curr.get(iso, (None, None))
            if prev_pos is None or curr_pos is None:
                continue
            m = min(len(prev_pos), len(curr_pos))
            if m > 0:
                diff += float(np.linalg.norm(prev_pos[:m] - curr_pos[:m]))
                diff += float(np.linalg.norm(prev_str[:m] - curr_str[:m]))
        return diff

    def global_uncertainty(self) -> float:
        """
        Return global uncertainty U = Σ_h Σ_j Var(q_{h,j}) (Sec. 3.6).
        """
        total = 0.0
        for iso, filt in self.filters.items():
            if not filt.continuous_particles:
                continue
            self._gpu_enabled()
            from pf import gpu_utils
            import torch

            device = gpu_utils.resolve_device(self.pf_config.gpu_device)
            dtype = gpu_utils.resolve_dtype(self.pf_config.gpu_dtype)
            states = [p.state for p in filt.continuous_particles]
            _, strengths_t, _, _ = gpu_utils.pack_states(
                states, device=device, dtype=dtype
            )
            weights = torch.as_tensor(
                filt.continuous_weights, device=device, dtype=dtype
            )
            weight_sum = torch.sum(weights)
            if float(weight_sum) <= 0.0:
                weights = torch.full_like(weights, 1.0 / max(weights.numel(), 1))
            else:
                weights = weights / weight_sum
            mean = torch.sum(weights[:, None] * strengths_t, dim=0)
            var = torch.sum(weights[:, None] * (strengths_t - mean) ** 2, dim=0)
            total += float(torch.sum(var).detach().cpu().item())
        return total

    def credible_region_volumes(
        self, confidence: float = 0.95
    ) -> Dict[str, List[float]]:
        """
        Compute 3D positional credible region volumes for each isotope/source (Sec. 3.5).

        For each source index m (up to max_r across particles), compute weighted mean/cov
        of positions and return ellipsoid volume using chi-square threshold. Used by
        should_stop_shield_rotation/should_stop_exploration to enforce small positional
        uncertainty before declaring convergence.
        """
        volumes: Dict[str, List[float]] = {}
        chi2_thresh = float(chi2.ppf(confidence, df=3))
        for iso, filt in self.filters.items():
            vols: List[float] = []
            if not filt.continuous_particles:
                volumes[iso] = vols
                continue
            w = filt.continuous_weights
            max_r = max(
                (p.state.num_sources for p in filt.continuous_particles), default=0
            )
            for j in range(max_r):
                positions = []
                weights = []
                for wi, p in zip(w, filt.continuous_particles):
                    if p.state.num_sources > j:
                        positions.append(p.state.positions[j])
                        weights.append(wi)
                if not positions:
                    continue
                pos_arr = np.vstack(positions)
                weights_arr = np.asarray(weights)
                weights_arr = weights_arr / max(np.sum(weights_arr), 1e-12)
                mean = np.sum(weights_arr[:, None] * pos_arr, axis=0)
                centered = pos_arr - mean
                cov = centered.T @ (centered * weights_arr[:, None])
                # Ellipsoid volume = 4/3 π sqrt(det(cov * chi2_thresh))
                det_val = np.linalg.det(cov * chi2_thresh)
                if det_val < 0:
                    vol = 0.0
                else:
                    vol = float((4.0 / 3.0) * np.pi * np.sqrt(det_val + 1e-12))
                vols.append(vol)
            volumes[iso] = vols
        return volumes

    def should_stop_shield_rotation(
        self,
        pose_idx: int,
        ig_threshold: float = 1e-3,
        change_tol: float = 1e-2,
        uncertainty_tol: float = 1e-3,
        live_time_s: float = 1.0,
    ) -> bool:
        """
        Stop shield rotation when convergence criteria are met (Sec. 3.5–3.6).

        - max IG_k(φ) below threshold
        - estimate change ||Δs|| + ||Δq|| < change_tol
        - global uncertainty U below threshold
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        if len(self.history_estimates) < 2:
            return False
        ig_scores = []
        for oidx in range(self.num_orientations):
            ig_scores.append(
                self.orientation_information_gain(
                    pose_idx=pose_idx, orient_idx=oidx, live_time_s=live_time_s
                )
            )
        max_ig = max(ig_scores) if ig_scores else 0.0
        dwell_time = sum(
            rec.live_time_s for rec in self.measurements if rec.pose_idx == pose_idx
        )
        # Credible region volumes check (Sec. 3.5)
        volumes = self.credible_region_volumes()
        max_volume = 0.0
        for vols in volumes.values():
            if vols:
                max_volume = max(max_volume, max(vols))
        return (
            (max_ig < ig_threshold)
            and (self.estimate_change_norm() < change_tol)
            and (self.global_uncertainty() < uncertainty_tol)
            and (max_volume < self.pf_config.credible_volume_threshold)
            or (dwell_time >= self.pf_config.max_dwell_time_s)
        )

    def should_stop_exploration(
        self,
        ig_threshold: float = 5e-4,
        change_tol: float = 5e-3,
        uncertainty_tol: float = 5e-4,
        live_time_s: float = 1.0,
    ) -> bool:
        """
        Stop the overall exploration (Sec. 3.6) based on IG and uncertainty convergence.

        - Max IG at the last pose is small
        - Estimate change is small
        - Global uncertainty U is small
        """
        if not self.poses:
            return False
        last_pose_idx = len(self.poses) - 1
        return self.should_stop_shield_rotation(
            pose_idx=last_pose_idx,
            ig_threshold=ig_threshold,
            change_tol=change_tol,
            uncertainty_tol=uncertainty_tol,
            live_time_s=live_time_s,
        )

    def prune_spurious_sources(
        self,
        method: str = "legacy",
        params: Dict[str, float] | None = None,
        tau_mix: float = 0.9,
        epsilon: float = 1e-6,
        min_support: int = 1,
        min_obs_count: float = 0.0,
        min_strength_abs: float | None = None,
        min_strength_ratio: float | None = None,
    ) -> Dict[str, NDArray[np.bool_]]:
        """
        Compute spurious-source keep masks using delta-LL, best-case residual gating, or legacy dominance.

        For each isotope h and each candidate source, the pruning method is applied using
        per-measurement expected counts from the continuous kernel. Optionally drop sources
        below max(min_strength_abs, min_strength_ratio * max_strength).

        Method params (passed via params):
        - deltaLL: deltaLL_min, penalty_d (BIC-style), epsilon
        - bestcase: alpha, lambda_min, lrt_threshold, epsilon
        - legacy: tau_mix, epsilon
        """
        from pf.mixing import prune_spurious_sources_continuous

        keep_masks = prune_spurious_sources_continuous(
            self,
            method=method,
            params=params,
            tau_mix=tau_mix,
            epsilon=epsilon,
            min_support=min_support,
            min_obs_count=min_obs_count,
            min_strength_abs=min_strength_abs,
            min_strength_ratio=min_strength_ratio,
        )
        return keep_masks

    def pruned_estimates(
        self,
        method: str = "legacy",
        params: Dict[str, float] | None = None,
        tau_mix: float = 0.9,
        epsilon: float = 1e-6,
        min_support: int = 1,
        min_obs_count: float = 0.0,
        min_strength_abs: float | None = None,
        min_strength_ratio: float | None = None,
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """
        Return non-destructively pruned estimates derived from MMSE outputs.

        This uses prune_spurious_sources_continuous() in estimate space and does not
        mutate particle states.
        """
        from pf.mixing import prune_spurious_sources_continuous

        est = self.estimates()
        keep_masks = prune_spurious_sources_continuous(
            self,
            method=method,
            params=params,
            tau_mix=tau_mix,
            epsilon=epsilon,
            min_support=min_support,
            min_obs_count=min_obs_count,
            min_strength_abs=min_strength_abs,
            min_strength_ratio=min_strength_ratio,
            estimate_snapshot=est,
        )
        pruned: Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
        for iso, (pos, strg) in est.items():
            keep = keep_masks.get(iso)
            if keep is None or keep.size == 0:
                pruned[iso] = (pos, strg)
            else:
                keep_arr = np.asarray(keep, dtype=bool).reshape(-1)
                if keep_arr.size != pos.shape[0]:
                    raise ValueError(
                        "report model-order keep mask must match estimate count."
                    )
                diagnostics = self._last_report_model_order_diagnostics.get(iso)
                if isinstance(diagnostics, dict) and bool(
                    diagnostics.get("preserve_cardinality", False)
                ):
                    target_count = min(
                        int(diagnostics.get("selected_count", 0)),
                        int(pos.shape[0]),
                    )
                    if target_count > int(np.count_nonzero(keep_arr)):
                        selected_indices = diagnostics.get("selected_indices", [])
                        selected_iter = (
                            selected_indices
                            if isinstance(selected_indices, list)
                            else []
                        )
                        for idx in selected_iter:
                            idx_int = int(idx)
                            if 0 <= idx_int < keep_arr.size:
                                keep_arr[idx_int] = True
                            if int(np.count_nonzero(keep_arr)) >= target_count:
                                break
                    if target_count > int(np.count_nonzero(keep_arr)):
                        dropped = np.flatnonzero(~keep_arr)
                        strengths = np.asarray(strg, dtype=float).reshape(-1)
                        if dropped.size and strengths.size == keep_arr.size:
                            order = dropped[np.argsort(strengths[dropped])[::-1]]
                        else:
                            order = dropped
                        for idx_int in order:
                            keep_arr[int(idx_int)] = True
                            if int(np.count_nonzero(keep_arr)) >= target_count:
                                break
                pruned[iso] = (pos[keep_arr], strg[keep_arr])
        return pruned
