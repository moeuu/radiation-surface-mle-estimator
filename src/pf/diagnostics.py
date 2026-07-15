"""Diagnostic state helpers for isotope particle filters."""

from __future__ import annotations

from typing import Any

import numpy as np

from pf.state import IsotopeState


def reset_step_diagnostics(target: Any) -> None:
    """Reset per-step diagnostic counters on an isotope filter-like object."""
    target.last_ess = None
    target.last_ess_pre = None
    target.last_ess_post = None
    target.last_resample_ess = False
    target.last_resample_count = 0
    target.last_birth_count = 0
    target.last_kill_count = 0
    target.last_n_after_adapt = None
    target.last_temper_steps = []
    target.last_temper_resample_count = 0
    target.last_mode_preserved_count = 0
    target.last_mode_preserving_strata_summary = {}
    target.last_mode_preserving_selected_strata = []
    target.last_mode_preserving_cardinality_summary = {}
    target.last_mode_preserving_selected_cardinalities = []
    target.last_birth_residual_chi2 = 0.0
    target.last_birth_residual_p_value = 1.0
    target.last_birth_residual_support = 0
    target.last_birth_residual_distinct_poses = 0
    target.last_birth_residual_distinct_stations = 0
    target.last_birth_residual_gate_passed = False
    target.last_birth_residual_refit_fraction = 1.0
    target.last_birth_residual_refit_gate_passed = True
    target.last_birth_residual_layer = "none"
    target.last_birth_residual_layer_count = 0
    target.last_birth_forced_attempts = 0
    target.last_birth_forced_accepts = 0
    target.last_birth_forced_mask_relaxations = 0
    target.last_birth_forced_no_candidate = 0
    target.last_birth_forced_rejected = 0
    target.last_birth_forced_best_delta = -np.inf
    target.last_birth_global_rescue_candidates = 0
    target.last_birth_global_rescue_attempts = 0
    target.last_birth_global_rescue_accepts = 0
    target.last_birth_global_rescue_rejected = 0
    target.last_birth_global_rescue_best_delta = -np.inf
    target.last_runtime_report_rescue_candidates = 0
    target.last_runtime_report_rescue_sources = 0
    target.last_runtime_report_rescue_injected = 0
    target.last_runtime_report_rescue_weight = 0.0
    target.last_weak_source_prune_occlusion_protected = 0
    target.last_birth_structural_eligible = 0
    target.last_pseudo_source_verified = 0
    target.last_pseudo_source_failed = 0
    target.last_pseudo_source_pruned = 0
    target.last_pseudo_source_quarantined = 0
    target.last_pseudo_source_quarantine_active = 0
    target.last_pseudo_source_fail_reasons = {}
    target.last_source_event_diagnostics = []
    target.last_structural_timing_s = {}
    target._resample_count_in_observation = 0


def build_source_event_record(
    *,
    event: str,
    isotope: str,
    state: IsotopeState,
    source_idx: int,
    reason: str,
    extra: dict[str, object] | None = None,
) -> dict[str, object] | None:
    """Return a source-slot diagnostic record or None for an invalid source."""
    idx = int(source_idx)
    if idx < 0 or idx >= int(state.num_sources):
        return None
    record: dict[str, object] = {
        "event": str(event),
        "isotope": str(isotope),
        "reason": str(reason),
        "source_index": idx,
        "position": [float(value) for value in state.positions[idx]],
        "strength": float(state.strengths[idx]),
        "age": int(state.ages[idx]) if state.ages is not None else None,
        "low_q_streak": int(state.low_q_streaks[idx])
        if state.low_q_streaks is not None
        else None,
        "support_score": float(state.support_scores[idx])
        if state.support_scores is not None
        else None,
        "tentative": bool(state.tentative_sources[idx])
        if state.tentative_sources is not None
        else None,
        "verification_fail_streak": int(state.verification_fail_streaks[idx])
        if state.verification_fail_streaks is not None
        else None,
    }
    if extra:
        record.update(extra)
    return record
