"""Mission-level stopping and adaptive shield-program helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from planning.dss_pp import DSSPPConfig


def remaining_measurement_ready_for_stop(
    estimate: Mapping[str, Any] | None,
) -> bool:
    """Return True when the remaining-measurement estimate has no unresolved budget."""
    if not estimate:
        return True
    unresolved_raw = estimate.get("unresolved_factors", [])
    if isinstance(unresolved_raw, str):
        unresolved_factors = [unresolved_raw] if unresolved_raw else []
    else:
        unresolved_factors = [str(value) for value in unresolved_raw or []]
    try:
        remaining = int(estimate.get("estimated_remaining_stations", 0))
    except (TypeError, ValueError):
        remaining = 0
    try:
        budget = float(estimate.get("current_budget", 0.0))
    except (TypeError, ValueError):
        budget = 0.0
    return not (unresolved_factors and remaining > 0 and budget > 0.0)


def remaining_measurement_payload(
    estimate: Mapping[str, Any] | object | None,
) -> Mapping[str, Any]:
    """Return a mapping view of a remaining-measurement estimate."""
    if estimate is None:
        return {}
    if isinstance(estimate, Mapping):
        return estimate
    if hasattr(estimate, "to_dict"):
        try:
            payload = estimate.to_dict()
        except (TypeError, ValueError):
            return {}
        if isinstance(payload, Mapping):
            return payload
    return {}


def remaining_measurement_component(
    estimate: Mapping[str, Any] | object | None,
    key: str,
) -> float:
    """Return one remaining-measurement component with a zero fallback."""
    payload = remaining_measurement_payload(estimate)
    components = payload.get("components", {})
    if not isinstance(components, Mapping):
        return 0.0
    try:
        return float(components.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def report_model_order_ready_for_stop(
    estimator: object,
    *,
    refresh_estimates: bool = True,
) -> bool:
    """Return whether report-level model-order diagnostics are stable enough to stop."""
    if bool(refresh_estimates) and hasattr(estimator, "estimates"):
        try:
            estimator.estimates()
        except RuntimeError:
            return False
    if hasattr(estimator, "report_model_order_ready"):
        return bool(estimator.report_model_order_ready())
    diagnostics_getter = getattr(estimator, "report_model_order_diagnostics", None)
    if not callable(diagnostics_getter):
        return False
    try:
        diagnostics = diagnostics_getter()
    except (RuntimeError, ValueError, TypeError):
        return False
    if not isinstance(diagnostics, Mapping) or not diagnostics:
        return False
    for stats in diagnostics.values():
        if not isinstance(stats, Mapping):
            return False
        try:
            candidate_count = int(stats.get("candidate_count", 0))
            selected_count = int(stats.get("selected_count", 0))
        except (TypeError, ValueError):
            return False
        if max(candidate_count, selected_count) <= 1:
            continue
        if not bool(stats.get("model_order_ready", False)):
            return False
    return True


def sparse_cardinality_evidence_gap_unresolved(
    diagnostics: Mapping[str, Any] | object | None,
    *,
    gap_target: float,
) -> bool:
    """Return True when sparse Poisson evidence still has a weak cardinality gap."""
    if not isinstance(diagnostics, Mapping) or not diagnostics:
        return False
    target = max(float(gap_target), 0.0)
    gap_keys = (
        "criterion_margin_to_runner_up",
        "criterion_margin_to_simpler",
        "bic_gap_to_next_count",
        "bic_gap_to_previous_count",
    )
    for stats_raw in diagnostics.values():
        if not isinstance(stats_raw, Mapping):
            continue
        stats = dict(stats_raw)
        sparse_stats = stats.get("sparse_poisson_evidence")
        if isinstance(sparse_stats, Mapping):
            stats = dict(sparse_stats)
        if not bool(stats.get("available", False)):
            continue
        if not bool(stats.get("model_order_ready", False)):
            return True
        if target <= 0.0:
            continue
        finite_gaps: list[float] = []
        for key in gap_keys:
            try:
                value = float(stats.get(key, float("inf")))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                finite_gaps.append(value)
        if finite_gaps and min(finite_gaps) < target:
            return True
    return False


def has_birth_residual_evidence(
    estimator: object,
    *,
    min_support: int,
) -> bool:
    """Return True when any isotope still has residual evidence for a new source."""
    support_floor = max(1, int(min_support))
    filters = getattr(estimator, "filters", {})
    for filt in filters.values():
        gate_passed = bool(getattr(filt, "last_birth_residual_gate_passed", False))
        support = int(getattr(filt, "last_birth_residual_support", 0))
        if gate_passed and support >= support_floor:
            return True
    return False


def report_model_order_simple_ready_for_stop(
    estimator: object,
    *,
    remaining_measurement_estimate: Mapping[str, Any] | object | None,
    max_sources_per_isotope: int = 1,
    min_bic_margin: float = 10.0,
    max_condition_number: float = 100.0,
    max_response_correlation: float = 0.98,
    residual_budget_threshold: float = 1.0e-9,
    ambiguity_budget_threshold: float = 1.0e-9,
    allow_high_surface_ambiguity: bool = False,
    require_no_birth_residual: bool = True,
    birth_residual_min_support: int = 1,
    refresh_estimates: bool = True,
) -> bool:
    """Return True when report-level BIC makes the mission structurally simple."""
    residual_budget = remaining_measurement_component(
        remaining_measurement_estimate,
        "residual",
    )
    isotope_absence_budget = remaining_measurement_component(
        remaining_measurement_estimate,
        "isotope_absence",
    )
    report_residual_budget = remaining_measurement_component(
        remaining_measurement_estimate,
        "report_residual",
    )
    if (
        residual_budget > float(residual_budget_threshold)
        or isotope_absence_budget > float(residual_budget_threshold)
        or report_residual_budget > float(residual_budget_threshold)
    ):
        return False
    same_isotope_budget = remaining_measurement_component(
        remaining_measurement_estimate,
        "same_isotope_separation",
    )
    report_corr_budget = remaining_measurement_component(
        remaining_measurement_estimate,
        "report_response_correlation",
    )
    strength_absorption_budget = remaining_measurement_component(
        remaining_measurement_estimate,
        "strength_absorption",
    )
    high_surface_budget = remaining_measurement_component(
        remaining_measurement_estimate,
        "high_surface_ambiguity",
    )
    if (
        same_isotope_budget > float(ambiguity_budget_threshold)
        or report_corr_budget > float(ambiguity_budget_threshold)
        or strength_absorption_budget > float(ambiguity_budget_threshold)
    ):
        return False
    if (
        high_surface_budget > float(ambiguity_budget_threshold)
        and not bool(allow_high_surface_ambiguity)
    ):
        return False
    if bool(require_no_birth_residual) and has_birth_residual_evidence(
        estimator,
        min_support=max(1, int(birth_residual_min_support)),
    ):
        return False
    if bool(refresh_estimates) and hasattr(estimator, "estimates"):
        try:
            estimator.estimates()
        except RuntimeError:
            return False
    if not hasattr(estimator, "report_model_order_diagnostics"):
        return False
    diagnostics = estimator.report_model_order_diagnostics()
    if not diagnostics:
        return False
    max_sources = max(0, int(max_sources_per_isotope))
    margin_floor = max(0.0, float(min_bic_margin))
    condition_limit = max(0.0, float(max_condition_number))
    corr_limit = max(0.0, float(max_response_correlation))
    for _isotope, stats in diagnostics.items():
        if not isinstance(stats, Mapping):
            return False
        try:
            selected_count = int(stats.get("selected_count", 0))
        except (TypeError, ValueError):
            return False
        if selected_count > max_sources:
            return False
        try:
            margin_default = float("-inf") if selected_count > 0 else float("inf")
            margin = float(stats.get("criterion_margin_to_simpler", margin_default))
        except (TypeError, ValueError):
            margin = float("-inf")
        if selected_count > 0 and (
            not np.isfinite(margin) or margin < margin_floor
        ):
            return False
        try:
            condition = float(stats.get("condition_number", 0.0))
        except (TypeError, ValueError):
            condition = float("inf")
        if (
            selected_count > 1
            and condition_limit > 0.0
            and np.isfinite(condition)
            and condition > condition_limit
        ):
            return False
        try:
            response_corr = float(
                stats.get("selected_max_response_correlation", 0.0)
            )
        except (TypeError, ValueError):
            response_corr = float("inf")
        if (
            selected_count > 1
            and np.isfinite(response_corr)
            and response_corr > corr_limit
        ):
            return False
    return True


def adapt_dss_program_length_for_budget(
    config: DSSPPConfig,
    *,
    enabled: bool,
    simple_program_length: int,
    residual_program_length: int,
    residual_burst_active: bool,
    report_simple_ready: bool,
    remaining_measurement_estimate: Mapping[str, Any] | object | None,
    residual_budget_threshold: float,
    ambiguity_budget_threshold: float,
    allow_high_surface_simple: bool = False,
    residual_extension_requires_cardinality_evidence: bool = False,
    cardinality_evidence_unresolved: bool = False,
) -> tuple[DSSPPConfig, str]:
    """Return a DSS-PP config with a measurement-budget-aware program length."""
    if not bool(enabled):
        return config, "disabled"
    if config.forced_program_pair_ids is not None:
        return config, "forced_program"
    base_length = max(1, int(config.program_length))
    simple_length = max(1, min(base_length, int(simple_program_length)))
    residual_length = max(base_length, int(residual_program_length))
    residual_budget = remaining_measurement_component(
        remaining_measurement_estimate,
        "residual",
    )
    isotope_absence_budget = remaining_measurement_component(
        remaining_measurement_estimate,
        "isotope_absence",
    )
    report_residual_budget = remaining_measurement_component(
        remaining_measurement_estimate,
        "report_residual",
    )
    if (
        residual_budget > float(residual_budget_threshold)
        or isotope_absence_budget > float(residual_budget_threshold)
        or report_residual_budget > float(residual_budget_threshold)
    ):
        if bool(residual_extension_requires_cardinality_evidence) and not bool(
            cardinality_evidence_unresolved
        ):
            return config, "residual_without_cardinality_gap"
        if residual_length != base_length:
            return replace(config, program_length=residual_length), "residual"
        return config, "residual"
    if bool(residual_burst_active) and not bool(report_simple_ready):
        if bool(residual_extension_requires_cardinality_evidence) and not bool(
            cardinality_evidence_unresolved
        ):
            return config, "burst_without_cardinality_gap"
        if residual_length != base_length:
            return replace(config, program_length=residual_length), "residual"
        return config, "residual"
    same_isotope_budget = remaining_measurement_component(
        remaining_measurement_estimate,
        "same_isotope_separation",
    )
    report_corr_budget = remaining_measurement_component(
        remaining_measurement_estimate,
        "report_response_correlation",
    )
    strength_absorption_budget = remaining_measurement_component(
        remaining_measurement_estimate,
        "strength_absorption",
    )
    high_surface_budget = remaining_measurement_component(
        remaining_measurement_estimate,
        "high_surface_ambiguity",
    )
    if (
        same_isotope_budget > float(ambiguity_budget_threshold)
        or report_corr_budget > float(ambiguity_budget_threshold)
        or strength_absorption_budget > float(ambiguity_budget_threshold)
    ):
        return config, "ambiguity"
    if (
        high_surface_budget > float(ambiguity_budget_threshold)
        and not (bool(report_simple_ready) and bool(allow_high_surface_simple))
    ):
        return config, "ambiguity"
    if bool(report_simple_ready) and simple_length < base_length:
        return replace(config, program_length=simple_length), "simple_report"
    remaining_payload = remaining_measurement_payload(remaining_measurement_estimate)
    if (
        remaining_payload
        and remaining_measurement_ready_for_stop(remaining_payload)
        and simple_length < base_length
    ):
        return replace(config, program_length=simple_length), "remaining_ready"
    return config, "full"


def resolve_mission_max_poses(
    cli_max_poses: int | None,
    runtime_config: Mapping[str, Any],
) -> int | None:
    """Resolve the mission pose cap while preserving explicit CLI overrides."""
    if cli_max_poses is not None:
        return max(1, int(cli_max_poses)) if int(cli_max_poses) > 0 else None
    mission_stop_max_poses_value = runtime_config.get(
        "mission_stop_max_poses",
        runtime_config.get("mission_stop_min_poses", None),
    )
    if mission_stop_max_poses_value is None:
        return None
    return max(1, int(mission_stop_max_poses_value))
