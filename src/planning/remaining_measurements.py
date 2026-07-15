"""Estimate remaining station windows from PF ambiguity and DSS-PP gain."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from pf.estimator import RotatingShieldPFEstimator
from pf.likelihood import expected_counts_per_source
from planning.dss_pp import DSSPPNode, extract_signature_modes


@dataclass(frozen=True)
class RemainingMeasurementConfig:
    """Configuration for online remaining-measurement estimation."""

    enabled: bool = True
    mode_cluster_radius_m: float = 1.5
    max_modes_per_isotope: int = 4
    max_particles: int | None = None
    planning_method: str | None = None
    target_position_spread_m: float = 1.0
    target_strength_cv: float = 0.5
    target_cardinality_confidence: float = 0.9
    pairwise_separation_threshold: float = 9.0
    residual_chi2_threshold: float = 9.0
    count_variance_floor: float = 1.0
    stop_budget: float = 0.0
    eta_default: float = 0.7
    eta_min: float = 0.3
    eta_max: float = 1.0
    gain_epsilon: float = 1.0e-6
    max_reported_stations: int = 99
    uncertainty_weight: float = 1.0
    cardinality_weight: float = 1.0
    separation_weight: float = 1.5
    verification_weight: float = 1.0
    residual_weight: float = 1.0
    report_response_correlation_weight: float = 1.0
    report_residual_weight: float = 1.0
    strength_absorption_weight: float = 0.5
    report_response_correlation_threshold: float = 0.9
    report_positive_residual_fraction_threshold: float = 0.02
    report_strength_concentration_threshold: float = 0.75
    high_surface_ambiguity_weight: float = 1.0
    high_surface_z_fraction: float = 0.75
    high_surface_pairwise_separation_threshold: float = 9.0
    high_surface_absorption_q_multiple: float = 2.0
    dss_information_gain_weight: float = 1.0
    dss_count_utility_weight: float = 0.25
    range_scale: float = 1.35
    unresolved_absent_min_total_counts: float = 25.0
    unresolved_absent_min_max_counts: float = 5.0
    unresolved_absent_min_snr: float = 2.0
    unresolved_absent_budget_weight: float = 1.0
    residual_surface_gain_candidate_limit: int = 2048


@dataclass(frozen=True)
class RemainingMeasurementEstimate:
    """Summarize the predicted remaining station and spectrum budget."""

    current_station_count: int
    estimated_remaining_stations: int
    estimated_remaining_station_low: int
    estimated_remaining_station_high: int
    estimated_remaining_spectra_low: int
    estimated_remaining_spectra_high: int
    program_length: int
    current_budget: float
    stop_budget: float
    predicted_gain: float
    empirical_eta: float
    bottleneck: str
    unresolved_factors: tuple[str, ...]
    components: dict[str, float] = field(default_factory=dict)
    gains: dict[str, float] = field(default_factory=dict)
    isotope_details: dict[str, dict[str, float | int]] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable estimate payload."""
        payload = _json_safe(asdict(self))
        payload["unresolved_factors"] = list(self.unresolved_factors)
        return payload


@dataclass(frozen=True)
class _PairwiseSignatureStats:
    """Store pairwise response-separation statistics."""

    deficit: float
    min_separation: float
    unresolved_pairs: int
    weighted_increment: float


def _json_safe(value: Any) -> Any:
    """Return a JSON-safe copy with nonfinite floats converted to null."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _timer_add(timings: dict[str, float] | None, key: str, elapsed_s: float) -> None:
    """Accumulate a nonnegative timing value when timing collection is enabled."""
    if timings is None:
        return
    timings[key] = float(timings.get(key, 0.0)) + max(float(elapsed_s), 0.0)


def _remaining_state_cache_key(
    estimator: RotatingShieldPFEstimator,
    config: RemainingMeasurementConfig,
) -> tuple[Any, ...]:
    """Return a conservative key for one-shot remaining-state reuse."""
    config_items = tuple(
        sorted((str(key), repr(value)) for key, value in asdict(config).items())
    )
    return (
        int(len(getattr(estimator, "measurements", []))),
        int(getattr(estimator, "_report_cache_revision", 0)),
        config_items,
    )


def _cached_remaining_state_payload(
    estimator: RotatingShieldPFEstimator,
    cache_key: tuple[Any, ...],
) -> dict[str, Any] | None:
    """Return cached remaining-state payload for the immediately following call."""
    payload = getattr(estimator, "_remaining_measurement_state_cache", None)
    if not isinstance(payload, dict):
        return None
    if payload.get("key") != cache_key:
        return None
    return payload


def _store_remaining_state_payload(
    estimator: RotatingShieldPFEstimator,
    cache_key: tuple[Any, ...],
    *,
    components: dict[str, float],
    isotope_details: dict[str, dict[str, float | int]],
    mode_arrays: dict[
        str,
        list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]],
    ],
    residual_surface_gain: float,
) -> None:
    """Store state diagnostics for the next post-DSS remaining-budget call."""
    setattr(
        estimator,
        "_remaining_measurement_state_cache",
        {
            "key": cache_key,
            "components": components,
            "isotope_details": isotope_details,
            "mode_arrays": mode_arrays,
            "residual_surface_gain": float(residual_surface_gain),
        },
    )


def _clear_remaining_state_payload(estimator: RotatingShieldPFEstimator) -> None:
    """Drop one-shot remaining-measurement state cache."""
    if hasattr(estimator, "_remaining_measurement_state_cache"):
        setattr(estimator, "_remaining_measurement_state_cache", None)


def _normalise_weights(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return normalized nonnegative weights with a uniform fallback."""
    arr = np.maximum(np.asarray(weights, dtype=float).reshape(-1), 0.0)
    if arr.size == 0:
        return arr
    total = float(np.sum(arr))
    if total <= 0.0:
        return np.full(arr.size, 1.0 / float(arr.size), dtype=float)
    return arr / total


def _pair_indices(pair_id: int, num_orients: int) -> tuple[int, int]:
    """Return Fe and Pb orientation indices from a flattened pair id."""
    n = max(1, int(num_orients))
    return int(pair_id) // n, int(pair_id) % n


def _pairwise_signature_stats_batched(
    response_by_measurement_mode: NDArray[np.float64],
    variance_by_measurement: NDArray[np.float64],
    mode_weights: NDArray[np.float64],
    *,
    threshold: float,
) -> _PairwiseSignatureStats:
    """Return batched same-isotope pairwise signature deficits."""
    response = np.asarray(response_by_measurement_mode, dtype=float)
    if response.ndim != 2 or response.shape[1] <= 1:
        return _PairwiseSignatureStats(0.0, float("inf"), 0, 0.0)
    variance = np.maximum(
        np.asarray(variance_by_measurement, dtype=float).reshape(-1),
        1.0e-12,
    )
    if variance.size == 1 and response.shape[0] != 1:
        variance = np.full(response.shape[0], float(variance[0]), dtype=float)
    if variance.size != response.shape[0]:
        raise ValueError("variance_by_measurement must match response rows.")
    weights = _normalise_weights(np.asarray(mode_weights, dtype=float))
    if weights.size != response.shape[1]:
        weights = np.ones(response.shape[1], dtype=float) / float(response.shape[1])
    diff = response[:, :, None] - response[:, None, :]
    d_matrix = np.sum((diff * diff) / variance[:, None, None], axis=0)
    upper = np.triu_indices(response.shape[1], k=1)
    distances = np.asarray(d_matrix[upper], dtype=float)
    if distances.size == 0:
        return _PairwiseSignatureStats(0.0, float("inf"), 0, 0.0)
    pair_weights = weights[upper[0]] * weights[upper[1]]
    pair_weight_sum = float(np.sum(pair_weights))
    if pair_weight_sum <= 0.0:
        pair_weights = np.full(distances.size, 1.0 / float(distances.size))
    else:
        pair_weights = pair_weights / pair_weight_sum
    deficit = np.maximum(float(threshold) - distances, 0.0)
    unresolved = distances < float(threshold)
    return _PairwiseSignatureStats(
        deficit=float(np.sum(pair_weights * deficit)),
        min_separation=float(np.min(distances)),
        unresolved_pairs=int(np.count_nonzero(unresolved)),
        weighted_increment=float(np.sum(pair_weights * distances)),
    )


def _weighted_cardinality_stats(
    source_counts: NDArray[np.int64],
    weights: NDArray[np.float64],
) -> tuple[float, float, int, float]:
    """Return entropy, MAP confidence, MAP count, and weighted variance."""
    counts = np.asarray(source_counts, dtype=int).reshape(-1)
    norm_weights = _normalise_weights(np.asarray(weights, dtype=float))
    if counts.size == 0 or norm_weights.size != counts.size:
        return 0.0, 1.0, 0, 0.0
    unique, inverse = np.unique(counts, return_inverse=True)
    probs = np.zeros(unique.size, dtype=float)
    np.add.at(probs, inverse, norm_weights)
    probs = _normalise_weights(probs)
    entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1.0e-12))))
    best_idx = int(np.argmax(probs))
    mean = float(np.sum(norm_weights * counts))
    variance = float(np.sum(norm_weights * (counts - mean) ** 2))
    return entropy, float(probs[best_idx]), int(unique[best_idx]), variance


def _weighted_strength_cv(
    strengths_by_particle: list[float],
    weights: NDArray[np.float64],
) -> float:
    """Return weighted coefficient of variation for total isotope strength."""
    values = np.asarray(strengths_by_particle, dtype=float).reshape(-1)
    norm_weights = _normalise_weights(np.asarray(weights, dtype=float))
    if values.size == 0 or norm_weights.size != values.size:
        return 0.0
    mean = float(np.sum(norm_weights * values))
    if mean <= 1.0e-12:
        return 0.0
    variance = float(np.sum(norm_weights * (values - mean) ** 2))
    return float(np.sqrt(max(variance, 0.0)) / mean)


def _mode_response_matrix(
    estimator: RotatingShieldPFEstimator,
    isotope: str,
    detector_positions: NDArray[np.float64],
    fe_indices: NDArray[np.int64],
    pb_indices: NDArray[np.int64],
    live_times: NDArray[np.float64],
    mode_positions: NDArray[np.float64],
    mode_strengths: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return expected counts for source modes over measurement rows."""
    if mode_positions.size == 0:
        return np.zeros((np.asarray(detector_positions).shape[0], 0), dtype=float)
    filt = estimator.filters[isotope]
    return np.maximum(
        expected_counts_per_source(
            kernel=filt.continuous_kernel,
            isotope=isotope,
            detector_positions=np.asarray(detector_positions, dtype=float),
            sources=np.asarray(mode_positions, dtype=float),
            strengths=np.asarray(mode_strengths, dtype=float),
            live_times=np.asarray(live_times, dtype=float),
            fe_indices=np.asarray(fe_indices, dtype=int),
            pb_indices=np.asarray(pb_indices, dtype=int),
            source_scale=estimator.response_scales_for_measurements(
                isotope,
                fe_indices,
                pb_indices,
            ),
        ),
        0.0,
    )


def _high_surface_mode_mask(
    estimator: RotatingShieldPFEstimator,
    positions: NDArray[np.float64],
    *,
    config: RemainingMeasurementConfig,
) -> NDArray[np.bool_]:
    """Return modes high enough to require explicit vertical disambiguation."""
    pos_arr = np.asarray(positions, dtype=float).reshape(-1, 3)
    if pos_arr.size == 0:
        return np.zeros(0, dtype=bool)
    hi = getattr(getattr(estimator, "pf_config", None), "position_max", (0.0, 0.0, 0.0))
    try:
        room_z = max(float(np.asarray(hi, dtype=float).reshape(3)[2]), 0.0)
    except (TypeError, ValueError):
        room_z = 0.0
    room_z = max(room_z, float(np.max(pos_arr[:, 2])), 1.0e-9)
    threshold = float(np.clip(config.high_surface_z_fraction, 0.0, 1.0)) * room_z
    return pos_arr[:, 2] >= threshold


def _state_budget_components(
    estimator: RotatingShieldPFEstimator,
    config: RemainingMeasurementConfig,
    timings: dict[str, float] | None = None,
) -> tuple[
    dict[str, float],
    dict[str, dict[str, float | int]],
    dict[
        str,
        list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]],
    ],
]:
    """Return current ambiguity components and cached mode arrays."""
    start = time.perf_counter()
    modes_start = time.perf_counter()
    modes_by_iso = extract_signature_modes(
        estimator,
        max_particles=config.max_particles,
        method=config.planning_method,
        mode_cluster_radius_m=float(config.mode_cluster_radius_m),
        max_modes_per_isotope=int(config.max_modes_per_isotope),
        tentative_weight_multiplier=1.5,
    )
    _timer_add(timings, "state_signature_modes_s", time.perf_counter() - modes_start)
    target_spread = max(float(config.target_position_spread_m), 1.0e-12)
    target_cv = max(float(config.target_strength_cv), 1.0e-12)
    target_cardinality = float(np.clip(config.target_cardinality_confidence, 0.0, 1.0))
    threshold = max(float(config.pairwise_separation_threshold), 0.0)
    residual_threshold = max(float(config.residual_chi2_threshold), 1.0e-12)
    variance_floor = max(float(config.count_variance_floor), 1.0e-12)
    components = {
        "uncertainty": 0.0,
        "cardinality": 0.0,
        "same_isotope_separation": 0.0,
        "pseudo_source_verification": 0.0,
        "residual": 0.0,
        "report_response_correlation": 0.0,
        "report_residual": 0.0,
        "strength_absorption": 0.0,
        "isotope_absence": 0.0,
        "high_surface_ambiguity": 0.0,
    }
    unresolved_absent = {}
    evidence_getter = getattr(estimator, "unresolved_isotope_evidence", None)
    if callable(evidence_getter):
        evidence_start = time.perf_counter()
        unresolved_absent = evidence_getter(
            min_total_counts=float(config.unresolved_absent_min_total_counts),
            min_max_count=float(config.unresolved_absent_min_max_counts),
            min_snr=float(config.unresolved_absent_min_snr),
        )
        _timer_add(
            timings, "state_absent_evidence_s", time.perf_counter() - evidence_start
        )
    report_start = time.perf_counter()
    try:
        report_diagnostics = estimator.report_model_order_diagnostics()
    except (RuntimeError, ValueError, TypeError):
        report_diagnostics = {}
    _timer_add(
        timings, "state_report_diagnostics_s", time.perf_counter() - report_start
    )
    isotope_details: dict[str, dict[str, float | int]] = {}
    mode_arrays: dict[
        str,
        list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]],
    ] = {}
    isotope_loop_start = time.perf_counter()
    for isotope, filt in estimator.filters.items():
        particles = filt.continuous_particles
        if not particles:
            isotope_details[isotope] = {}
            continue
        weights = _normalise_weights(np.asarray(filt.continuous_weights, dtype=float))
        source_counts = np.asarray(
            [int(particle.state.num_sources) for particle in particles],
            dtype=int,
        )
        entropy, confidence, map_count, cardinality_var = _weighted_cardinality_stats(
            source_counts,
            weights,
        )
        data = estimator._measurement_data_for_iso(isotope, window=None)
        count_supported, total_counts, max_count, signal_snr = (
            _measurement_data_count_evidence(data, config)
        )
        report_stats = (
            report_diagnostics.get(str(isotope), {})
            if isinstance(report_diagnostics, dict)
            else {}
        )
        report_selected_count = (
            int(report_stats.get("selected_count", 0))
            if isinstance(report_stats, dict)
            else 0
        )
        report_candidate_count = (
            int(report_stats.get("candidate_count", 0))
            if isinstance(report_stats, dict)
            else 0
        )
        report_count_supported_zero = (
            bool(report_stats.get("count_supported_zero_source", False))
            if isinstance(report_stats, dict)
            else False
        )
        report_model_order_unready = (
            not bool(report_stats.get("model_order_ready", False))
            and (
                report_candidate_count > 0
                or report_selected_count > 0
                or report_count_supported_zero
            )
            if isinstance(report_stats, dict)
            else False
        )
        report_max_response_corr = (
            float(report_stats.get("selected_max_response_correlation", 0.0))
            if isinstance(report_stats, dict)
            else 0.0
        )
        report_positive_residual_fraction = (
            float(report_stats.get("selected_positive_residual_fraction", 0.0))
            if isinstance(report_stats, dict)
            else 0.0
        )
        report_positive_residual_chi2 = (
            float(report_stats.get("selected_positive_residual_chi2", 0.0))
            if isinstance(report_stats, dict)
            else 0.0
        )
        report_strengths = (
            np.maximum(
                np.asarray(
                    report_stats.get("selected_strengths", []), dtype=float
                ).reshape(-1),
                0.0,
            )
            if isinstance(report_stats, dict)
            else np.zeros(0, dtype=float)
        )
        active_evidence = bool(
            count_supported
            or report_selected_count > 0
            or report_count_supported_zero
            or report_model_order_unready
            or str(isotope) in unresolved_absent
        )
        strength_totals = [
            float(
                np.sum(
                    np.maximum(
                        particle.state.strengths[: particle.state.num_sources],
                        0.0,
                    )
                )
            )
            for particle in particles
        ]
        strength_cv = _weighted_strength_cv(strength_totals, weights)
        modes = modes_by_iso.get(isotope, [])
        if modes:
            spread_budget = float(
                np.sum(
                    [
                        max(float(mode.spread_m) / target_spread - 1.0, 0.0)
                        * max(float(mode.weight), 0.0)
                        for mode in modes
                    ]
                )
            )
            mode_positions = np.vstack([mode.position_xyz for mode in modes])
            mode_strengths = np.asarray(
                [max(float(mode.strength_cps_1m), 0.0) for mode in modes],
                dtype=float,
            )
            mode_weights = _normalise_weights(
                np.asarray([mode.weight for mode in modes], dtype=float)
            )
            mode_arrays[isotope] = [(mode_positions, mode_strengths, mode_weights)]
        else:
            spread_budget = 0.0
            mode_arrays[isotope] = []
        strength_budget = max(strength_cv / target_cv - 1.0, 0.0)
        cardinality_budget = entropy + max(target_cardinality - confidence, 0.0)
        tentative_expected = _weighted_tentative_source_count(particles, weights)
        verification_views = int(getattr(filt, "last_birth_residual_distinct_poses", 0))
        required_views = max(1, int(filt.config.pseudo_source_min_distinct_views))
        verification_budget = (
            tentative_expected
            * max(
                required_views - min(verification_views, required_views),
                0,
            )
            / float(required_views)
        )
        verification_budget += float(
            max(int(getattr(filt, "last_pseudo_source_quarantine_active", 0)), 0)
        )
        separation_budget = 0.0
        report_corr_budget = 0.0
        report_residual_budget = 0.0
        strength_absorption_budget = 0.0
        report_strength_concentration = 0.0
        min_separation = float("inf")
        unresolved_pairs = 0
        high_surface_budget = 0.0
        high_surface_mode_count = 0
        high_surface_min_separation = float("inf")
        high_surface_unresolved_pairs = 0
        residual_budget = 0.0
        absent_payload = unresolved_absent.get(str(isotope), {})
        absence_budget = float(absent_payload.get("budget", 0.0))
        if not active_evidence:
            mode_arrays[isotope] = []
            isotope_details[isotope] = {
                "mode_count": int(len(modes)),
                "map_source_count": int(map_count),
                "active_evidence": 0,
                "observed_signal_total_counts": float(total_counts),
                "observed_signal_max_count": float(max_count),
                "observed_signal_snr": float(signal_snr),
                "cardinality_confidence": float(confidence),
                "cardinality_entropy": float(entropy),
                "cardinality_variance": float(cardinality_var),
                "strength_cv": float(strength_cv),
                "tentative_source_expectation": float(tentative_expected),
                "verification_views": int(verification_views),
                "required_verification_views": int(required_views),
                "min_pairwise_separation": 0.0,
                "unresolved_pair_count": 0,
                "high_surface_mode_count": 0,
                "high_surface_min_pairwise_separation": 0.0,
                "high_surface_unresolved_pair_count": 0,
                "high_surface_ambiguity_budget": 0.0,
                "residual_chi2": 0.0,
                "report_max_response_correlation": float(report_max_response_corr),
                "report_response_correlation_budget": 0.0,
                "report_positive_residual_fraction": float(
                    report_positive_residual_fraction
                ),
                "report_positive_residual_chi2": float(report_positive_residual_chi2),
                "report_residual_budget": 0.0,
                "report_strength_concentration": 0.0,
                "strength_absorption_budget": 0.0,
                "unresolved_absent_budget": 0.0,
                "unresolved_absent_total_counts": float(
                    absent_payload.get("total_counts", 0.0)
                ),
                "unresolved_absent_count_snr": float(
                    absent_payload.get("count_snr", 0.0)
                ),
            }
            continue
        if data is not None and data.z_k.size and mode_arrays[isotope]:
            mode_positions, mode_strengths, mode_weights = mode_arrays[isotope][0]
            response = _mode_response_matrix(
                estimator,
                isotope,
                data.detector_positions,
                data.fe_indices,
                data.pb_indices,
                data.live_times,
                mode_positions,
                mode_strengths,
            )
            variance = np.maximum(data.observation_variances, variance_floor)
            stats = _pairwise_signature_stats_batched(
                response,
                variance,
                mode_weights,
                threshold=threshold,
            )
            separation_budget = stats.deficit
            min_separation = stats.min_separation
            unresolved_pairs = stats.unresolved_pairs
            high_mask = _high_surface_mode_mask(
                estimator,
                mode_positions,
                config=config,
            )
            high_surface_mode_count = int(np.count_nonzero(high_mask))
            if high_surface_mode_count >= 2:
                high_stats = _pairwise_signature_stats_batched(
                    response[:, high_mask],
                    variance,
                    mode_weights[high_mask],
                    threshold=max(
                        float(config.high_surface_pairwise_separation_threshold),
                        0.0,
                    ),
                )
                high_surface_budget += float(high_stats.deficit)
                high_surface_min_separation = high_stats.min_separation
                high_surface_unresolved_pairs = high_stats.unresolved_pairs
            prior_mean = max(
                float(
                    getattr(
                        getattr(estimator, "pf_config", None),
                        "source_strength_prior_mean",
                        0.0,
                    )
                ),
                0.0,
            )
            if prior_mean > 0.0 and np.any(high_mask):
                high_strengths = mode_strengths[high_mask]
                absorption_threshold = prior_mean * max(
                    float(config.high_surface_absorption_q_multiple),
                    1.0,
                )
                high_surface_budget += float(
                    np.sum(
                        np.maximum(
                            high_strengths / max(absorption_threshold, 1.0e-12) - 1.0,
                            0.0,
                        )
                    )
                )
            background_rate = (
                float(filt.best_particle().state.background)
                if filt.continuous_particles
                else 0.0
            )
            prediction = background_rate * data.live_times + np.sum(response, axis=1)
            residual = np.maximum(np.asarray(data.z_k, dtype=float) - prediction, 0.0)
            residual_chi2 = float(np.sum((residual * residual) / variance))
            residual_budget = max(residual_chi2 / residual_threshold - 1.0, 0.0)
        else:
            residual_chi2 = 0.0
        if report_selected_count > 1:
            corr_threshold = float(
                np.clip(config.report_response_correlation_threshold, 0.0, 1.0)
            )
            if corr_threshold < 1.0:
                report_corr_budget = max(
                    (report_max_response_corr - corr_threshold)
                    / max(1.0 - corr_threshold, 1.0e-12),
                    0.0,
                )
        residual_fraction_threshold = max(
            float(config.report_positive_residual_fraction_threshold),
            0.0,
        )
        if residual_fraction_threshold > 0.0:
            report_residual_budget = max(
                report_positive_residual_fraction / residual_fraction_threshold - 1.0,
                0.0,
            )
        if report_strengths.size:
            total_report_strength = float(np.sum(report_strengths))
            if total_report_strength > 1.0e-12:
                report_strength_concentration = float(
                    np.max(report_strengths) / total_report_strength
                )
        concentration_threshold = float(
            np.clip(config.report_strength_concentration_threshold, 1.0e-12, 1.0)
        )
        if (
            report_selected_count > 1
            and report_strength_concentration > concentration_threshold
            and (report_residual_budget > 0.0 or report_corr_budget > 0.0)
        ):
            strength_absorption_budget = max(
                report_strength_concentration / concentration_threshold - 1.0,
                0.0,
            )
        components["uncertainty"] += spread_budget + strength_budget
        components["cardinality"] += cardinality_budget
        components["same_isotope_separation"] += separation_budget
        components["pseudo_source_verification"] += verification_budget
        components["residual"] += residual_budget
        components["report_response_correlation"] += report_corr_budget
        components["report_residual"] += report_residual_budget
        components["strength_absorption"] += strength_absorption_budget
        components["isotope_absence"] += float(
            config.unresolved_absent_budget_weight
        ) * max(absence_budget, 0.0)
        components["high_surface_ambiguity"] += high_surface_budget
        isotope_details[isotope] = {
            "mode_count": int(len(modes)),
            "map_source_count": int(map_count),
            "active_evidence": int(active_evidence),
            "observed_signal_total_counts": float(total_counts),
            "observed_signal_max_count": float(max_count),
            "observed_signal_snr": float(signal_snr),
            "cardinality_confidence": float(confidence),
            "cardinality_entropy": float(entropy),
            "cardinality_variance": float(cardinality_var),
            "strength_cv": float(strength_cv),
            "tentative_source_expectation": float(tentative_expected),
            "verification_views": int(verification_views),
            "required_verification_views": int(required_views),
            "min_pairwise_separation": (
                0.0 if not np.isfinite(min_separation) else float(min_separation)
            ),
            "unresolved_pair_count": int(unresolved_pairs),
            "high_surface_mode_count": int(high_surface_mode_count),
            "high_surface_min_pairwise_separation": (
                0.0
                if not np.isfinite(high_surface_min_separation)
                else float(high_surface_min_separation)
            ),
            "high_surface_unresolved_pair_count": int(high_surface_unresolved_pairs),
            "high_surface_ambiguity_budget": float(high_surface_budget),
            "residual_chi2": float(residual_chi2),
            "report_max_response_correlation": float(report_max_response_corr),
            "report_response_correlation_budget": float(report_corr_budget),
            "report_positive_residual_fraction": float(
                report_positive_residual_fraction
            ),
            "report_positive_residual_chi2": float(report_positive_residual_chi2),
            "report_residual_budget": float(report_residual_budget),
            "report_strength_concentration": float(report_strength_concentration),
            "strength_absorption_budget": float(strength_absorption_budget),
            "unresolved_absent_budget": float(max(absence_budget, 0.0)),
            "unresolved_absent_total_counts": float(
                absent_payload.get("total_counts", 0.0)
            ),
            "unresolved_absent_count_snr": float(absent_payload.get("count_snr", 0.0)),
        }
    _timer_add(
        timings, "state_isotope_loop_s", time.perf_counter() - isotope_loop_start
    )
    _timer_add(timings, "state_total_s", time.perf_counter() - start)
    return components, isotope_details, mode_arrays


def _measurement_data_count_evidence(
    data: object | None,
    config: RemainingMeasurementConfig,
) -> tuple[bool, float, float, float]:
    """Return whether measurement data gives count-supported isotope evidence."""
    if data is None:
        return False, 0.0, 0.0, 0.0
    counts_raw = getattr(data, "z_k", None)
    if counts_raw is None:
        return False, 0.0, 0.0, 0.0
    counts = np.maximum(np.asarray(counts_raw, dtype=float).reshape(-1), 0.0)
    if counts.size == 0:
        return False, 0.0, 0.0, 0.0
    variances = np.maximum(
        np.asarray(
            getattr(data, "observation_variances", np.ones_like(counts)), dtype=float
        ).reshape(-1),
        1.0,
    )
    total_counts = float(np.sum(counts))
    max_count = float(np.max(counts))
    signal_snr = float(total_counts / np.sqrt(max(float(np.sum(variances)), 1.0e-12)))
    total_floor = max(float(config.unresolved_absent_min_total_counts), 0.0)
    max_floor = max(float(config.unresolved_absent_min_max_counts), 0.0)
    snr_floor = max(float(config.unresolved_absent_min_snr), 0.0)
    count_floor_met = total_counts >= total_floor or max_count >= max_floor
    if snr_floor <= 0.0:
        supported = count_floor_met
    elif total_floor <= 0.0 and max_floor <= 0.0:
        supported = signal_snr >= snr_floor
    else:
        supported = count_floor_met and signal_snr >= snr_floor
    return bool(supported), total_counts, max_count, signal_snr


def _weighted_tentative_source_count(
    particles: Sequence[object],
    weights: NDArray[np.float64],
) -> float:
    """Return the weighted number of tentative or failed source slots."""
    total = 0.0
    for particle, weight in zip(particles, weights):
        state = particle.state
        count = max(0, int(state.num_sources))
        if count <= 0:
            continue
        tentative_raw = getattr(state, "tentative_sources", None)
        tentative = (
            np.zeros(count, dtype=bool)
            if tentative_raw is None
            else np.asarray(tentative_raw, dtype=bool)[:count]
        )
        failed_raw = getattr(state, "verification_fail_streaks", None)
        failed = (
            np.zeros(count, dtype=int)
            if failed_raw is None
            else np.asarray(failed_raw, dtype=int)[:count]
        )
        if tentative.size != count:
            padded = np.zeros(count, dtype=bool)
            padded[: min(tentative.size, count)] = tentative[:count]
            tentative = padded
        if failed.size != count:
            padded = np.zeros(count, dtype=int)
            padded[: min(failed.size, count)] = failed[:count]
            failed = padded
        total += float(weight) * float(np.count_nonzero(tentative | (failed > 0)))
    return float(total)


def _prediction_gain_components(
    estimator: RotatingShieldPFEstimator,
    next_pose_xyz: NDArray[np.float64] | None,
    shield_program_pair_ids: Sequence[int] | None,
    live_time_s: float,
    mode_arrays: dict[
        str,
        list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]],
    ],
    config: RemainingMeasurementConfig,
    dss_node: DSSPPNode | None,
    dss_diagnostics: dict[str, float | int | str] | None,
    *,
    residual_surface_gain: float | None = None,
    timings: dict[str, float] | None = None,
) -> tuple[dict[str, float], int]:
    """Return predicted one-station ambiguity reduction components."""
    start = time.perf_counter()
    program = tuple(int(pair_id) for pair_id in (shield_program_pair_ids or ()))
    if not program:
        program = (0,)
    program_length = max(1, len(program))
    gains = {
        "uncertainty": 0.0,
        "same_isotope_separation": 0.0,
        "pseudo_source_verification": 0.0,
        "residual": 0.0,
        "residual_surface": 0.0,
        "dss_information": 0.0,
        "high_surface_ambiguity": 0.0,
    }
    if next_pose_xyz is not None:
        pose = np.asarray(next_pose_xyz, dtype=float).reshape(3)
        fe = []
        pb = []
        for pair_id in program:
            fe_idx, pb_idx = _pair_indices(pair_id, estimator.num_orientations)
            fe.append(fe_idx)
            pb.append(pb_idx)
        detector_positions = np.repeat(pose[None, :], program_length, axis=0)
        live_times = np.full(program_length, max(float(live_time_s), 0.0), dtype=float)
        fe_indices = np.asarray(fe, dtype=int)
        pb_indices = np.asarray(pb, dtype=int)
        threshold = max(float(config.pairwise_separation_threshold), 0.0)
        floor = max(float(config.count_variance_floor), 1.0e-12)
        for isotope, arrays in mode_arrays.items():
            if not arrays:
                continue
            mode_positions, mode_strengths, mode_weights = arrays[0]
            response = _mode_response_matrix(
                estimator,
                isotope,
                detector_positions,
                fe_indices,
                pb_indices,
                live_times,
                mode_positions,
                mode_strengths,
            )
            row_variance = np.maximum(np.mean(response, axis=1), floor)
            stats = _pairwise_signature_stats_batched(
                response,
                row_variance,
                mode_weights,
                threshold=threshold,
            )
            gains["same_isotope_separation"] += stats.weighted_increment
            high_mask = _high_surface_mode_mask(
                estimator,
                mode_positions,
                config=config,
            )
            if int(np.count_nonzero(high_mask)) >= 2:
                high_stats = _pairwise_signature_stats_batched(
                    response[:, high_mask],
                    row_variance,
                    mode_weights[high_mask],
                    threshold=max(
                        float(config.high_surface_pairwise_separation_threshold),
                        0.0,
                    ),
                )
                gains["high_surface_ambiguity"] += high_stats.weighted_increment
    if dss_node is not None:
        gains["uncertainty"] += max(float(dss_node.information_gain), 0.0)
        gains["dss_information"] += max(float(dss_node.information_gain), 0.0)
        gains["same_isotope_separation"] += max(
            float(dss_node.signature_score)
            + float(dss_node.temporal_separation_score)
            + float(dss_node.elevation_signature_score),
            0.0,
        )
        gains["high_surface_ambiguity"] += max(
            float(dss_node.temporal_separation_score)
            + float(dss_node.elevation_signature_score)
            + float(dss_node.correlation_reduction_gain),
            0.0,
        )
        gains["residual"] += max(float(dss_node.count_utility), 0.0)
    if dss_diagnostics:
        for key in ("best_information_gain", "information_gain", "eig"):
            if key in dss_diagnostics:
                gains["dss_information"] += max(float(dss_diagnostics[key]), 0.0)
                break
    if residual_surface_gain is None:
        residual_surface_gain = _residual_surface_gain_estimate(
            estimator,
            mode_arrays,
            config,
            timings=timings,
        )
    else:
        _timer_add(timings, "residual_surface_cache_hit", 1.0)
    gains["residual_surface"] = residual_surface_gain
    gains["residual"] += residual_surface_gain
    gains["pseudo_source_verification"] += float(program_length)
    _timer_add(timings, "prediction_total_s", time.perf_counter() - start)
    return gains, program_length


def _residual_surface_gain_estimate(
    estimator: RotatingShieldPFEstimator,
    mode_arrays: dict[
        str,
        list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]],
    ],
    config: RemainingMeasurementConfig,
    timings: dict[str, float] | None = None,
) -> float:
    """
    Return a batched estimate of residual reduction available from one surface source.

    The residual budget can be very large after PF posterior collapse.  DSS count
    utility is a station-local score and is often too small to scale that budget,
    so this diagnostic computes the best single fixed-surface source that could
    reduce the current positive residual over the accumulated measurements.  It
    uses the same PF response model and a batched weighted least-squares strength
    fit over known source-surface candidates; no transport shortcut is introduced.
    """
    start = time.perf_counter()
    pool_all = np.asarray(
        getattr(estimator, "candidate_sources", np.zeros((0, 3))),
        dtype=float,
    )
    if pool_all.size == 0:
        return 0.0
    pool_all = pool_all.reshape(-1, 3)
    limit = int(config.residual_surface_gain_candidate_limit)
    if limit > 0 and pool_all.shape[0] > limit:
        sample_indices = np.linspace(
            0,
            pool_all.shape[0] - 1,
            max(1, limit),
            dtype=np.int64,
        )
        pool = pool_all[sample_indices]
    else:
        sample_indices = np.arange(pool_all.shape[0], dtype=np.int64)
        pool = pool_all
    if pool.size == 0:
        return 0.0
    threshold = max(float(config.residual_chi2_threshold), 1.0e-12)
    variance_floor = max(float(config.count_variance_floor), 1.0e-12)
    eps = max(float(config.gain_epsilon), 1.0e-12)
    total_gain = 0.0
    for isotope, filt in estimator.filters.items():
        iso_start = time.perf_counter()
        data = estimator._measurement_data_for_iso(isotope, window=None)
        if data is None or data.z_k.size == 0:
            continue
        variances = np.maximum(
            np.asarray(data.observation_variances, dtype=float).reshape(-1),
            variance_floor,
        )
        if variances.size != data.z_k.size:
            continue
        background_rate = (
            float(filt.best_particle().state.background)
            if getattr(filt, "continuous_particles", None)
            else 0.0
        )
        prediction = background_rate * np.asarray(data.live_times, dtype=float)
        arrays = mode_arrays.get(isotope, [])
        if arrays:
            mode_positions, mode_strengths, _mode_weights = arrays[0]
            response = _mode_response_matrix(
                estimator,
                isotope,
                data.detector_positions,
                data.fe_indices,
                data.pb_indices,
                data.live_times,
                mode_positions,
                mode_strengths,
            )
            prediction = prediction + np.sum(response, axis=1)
        residual = np.maximum(
            np.asarray(data.z_k, dtype=float).reshape(-1) - prediction,
            0.0,
        )
        current_chi2 = float(np.sum((residual * residual) / variances))
        if current_chi2 <= threshold:
            continue
        cached_grid_getter = getattr(estimator, "_cached_candidate_grid_counts", None)
        if callable(cached_grid_getter):
            candidate_counts = cached_grid_getter(
                filt=filt,
                isotope=isotope,
                data=data,
            )[:, sample_indices]
            _timer_add(timings, "residual_surface_candidate_cache_path", 1.0)
        else:
            candidate_counts = expected_counts_per_source(
                kernel=filt.continuous_kernel,
                isotope=isotope,
                detector_positions=data.detector_positions,
                sources=pool,
                strengths=np.ones(pool.shape[0], dtype=float),
                live_times=data.live_times,
                fe_indices=data.fe_indices,
                pb_indices=data.pb_indices,
                source_scale=estimator.response_scales_for_measurements(
                    isotope,
                    data.fe_indices,
                    data.pb_indices,
                ),
            )
        counts = np.maximum(np.asarray(candidate_counts, dtype=float), 0.0)
        if counts.ndim != 2 or counts.shape != (data.z_k.size, pool.shape[0]):
            continue
        weights = 1.0 / np.maximum(variances, eps)
        numerator = np.sum(weights[:, None] * residual[:, None] * counts, axis=0)
        denominator = np.sum(weights[:, None] * counts * counts, axis=0)
        q_hat = np.divide(
            numerator,
            np.maximum(denominator, eps),
            out=np.zeros_like(numerator),
            where=denominator > eps,
        )
        valid = np.isfinite(q_hat) & (q_hat > 0.0)
        if not np.any(valid):
            continue
        trial_residual = np.maximum(
            residual[:, None] - counts[:, valid] * q_hat[valid][None, :],
            0.0,
        )
        trial_chi2 = np.sum(weights[:, None] * trial_residual * trial_residual, axis=0)
        reduction = np.maximum(current_chi2 - trial_chi2, 0.0)
        if reduction.size:
            total_gain += float(np.max(reduction)) / threshold
        _timer_add(
            timings,
            f"residual_surface_{isotope}_s",
            time.perf_counter() - iso_start,
        )
    _timer_add(timings, "residual_surface_total_s", time.perf_counter() - start)
    return float(max(total_gain, 0.0))


def _empirical_eta(
    estimator: RotatingShieldPFEstimator,
    current_budget: float,
    predicted_gain: float,
    config: RemainingMeasurementConfig,
    *,
    update_history: bool = True,
) -> float:
    """Update and return the empirical predicted-vs-realized gain correction."""
    history = getattr(estimator, "_remaining_measurement_budget_history", [])
    if not isinstance(history, list):
        history = []
    ratios = getattr(estimator, "_remaining_measurement_eta_ratios", [])
    if not isinstance(ratios, list):
        ratios = []
    if update_history and history:
        previous = history[-1]
        prev_budget = float(previous.get("budget", current_budget))
        prev_gain = max(float(previous.get("predicted_gain", 0.0)), 1.0e-12)
        realized = max(prev_budget - float(current_budget), 0.0)
        ratios.append(realized / prev_gain)
        ratios = ratios[-8:]
    if update_history:
        history.append(
            {
                "budget": float(current_budget),
                "predicted_gain": float(predicted_gain),
            }
        )
        setattr(estimator, "_remaining_measurement_budget_history", history[-8:])
        setattr(estimator, "_remaining_measurement_eta_ratios", ratios)
    if ratios:
        eta = float(np.median(np.asarray(ratios, dtype=float)))
    else:
        eta = float(config.eta_default)
    return float(np.clip(eta, float(config.eta_min), float(config.eta_max)))


def estimate_remaining_measurement_budget(
    estimator: RotatingShieldPFEstimator,
    *,
    next_pose_xyz: NDArray[np.float64] | None = None,
    shield_program_pair_ids: Sequence[int] | None = None,
    live_time_s: float = 1.0,
    dss_node: DSSPPNode | None = None,
    dss_diagnostics: dict[str, float | int | str] | None = None,
    config: RemainingMeasurementConfig | None = None,
    current_station_count: int | None = None,
    update_history: bool = True,
) -> RemainingMeasurementEstimate:
    """Estimate remaining station windows from current PF ambiguity."""
    total_start = time.perf_counter()
    cfg = config or RemainingMeasurementConfig()
    timings: dict[str, float] = {}
    cache_key = _remaining_state_cache_key(estimator, cfg)
    cached_state = _cached_remaining_state_payload(estimator, cache_key)
    cached_residual_surface_gain: float | None = None
    if cached_state is not None:
        timings["state_cache_hit"] = 1.0
        components = dict(cached_state.get("components", {}))
        isotope_details = dict(cached_state.get("isotope_details", {}))
        mode_arrays = dict(cached_state.get("mode_arrays", {}))
        cached_residual_surface_gain = float(
            cached_state.get("residual_surface_gain", 0.0)
        )
    else:
        timings["state_cache_hit"] = 0.0
        components, isotope_details, mode_arrays = _state_budget_components(
            estimator,
            cfg,
            timings=timings,
        )
        cached_residual_surface_gain = _residual_surface_gain_estimate(
            estimator,
            mode_arrays,
            cfg,
            timings=timings,
        )
        if not update_history:
            _store_remaining_state_payload(
                estimator,
                cache_key,
                components=components,
                isotope_details=isotope_details,
                mode_arrays=mode_arrays,
                residual_surface_gain=cached_residual_surface_gain,
            )
    gains, program_length = _prediction_gain_components(
        estimator,
        next_pose_xyz,
        shield_program_pair_ids,
        live_time_s,
        mode_arrays,
        cfg,
        dss_node,
        dss_diagnostics,
        residual_surface_gain=cached_residual_surface_gain,
        timings=timings,
    )
    weighted_budget = (
        float(cfg.uncertainty_weight) * components["uncertainty"]
        + float(cfg.cardinality_weight) * components["cardinality"]
        + float(cfg.separation_weight) * components["same_isotope_separation"]
        + float(cfg.verification_weight) * components["pseudo_source_verification"]
        + float(cfg.residual_weight) * components["residual"]
        + float(cfg.report_response_correlation_weight)
        * components["report_response_correlation"]
        + float(cfg.report_residual_weight) * components["report_residual"]
        + float(cfg.strength_absorption_weight) * components["strength_absorption"]
        + float(cfg.high_surface_ambiguity_weight)
        * components["high_surface_ambiguity"]
        + components["isotope_absence"]
    )
    weighted_gain = (
        float(cfg.uncertainty_weight) * gains["uncertainty"]
        + float(cfg.separation_weight) * gains["same_isotope_separation"]
        + float(cfg.verification_weight) * gains["pseudo_source_verification"]
        + float(cfg.residual_weight) * gains["residual"]
        + float(cfg.high_surface_ambiguity_weight) * gains["high_surface_ambiguity"]
        + float(cfg.dss_information_gain_weight) * gains["dss_information"]
        + float(cfg.dss_count_utility_weight) * gains["residual"]
    )
    eta = _empirical_eta(
        estimator,
        weighted_budget,
        weighted_gain,
        cfg,
        update_history=update_history,
    )
    if update_history:
        _clear_remaining_state_payload(estimator)
    remaining_budget = max(weighted_budget - float(cfg.stop_budget), 0.0)
    denom = max(eta * weighted_gain, float(cfg.gain_epsilon))
    estimate = int(np.ceil(remaining_budget / denom)) if remaining_budget > 0.0 else 0
    estimate = min(max(estimate, 0), max(0, int(cfg.max_reported_stations)))
    if estimate > 0:
        low = max(1, int(np.floor(float(estimate) / max(float(cfg.range_scale), 1.0))))
    else:
        low = 0
    high = min(
        max(0, int(cfg.max_reported_stations)),
        max(
            estimate,
            int(np.ceil(float(estimate) * max(float(cfg.range_scale), 1.0))),
        ),
    )
    unresolved = tuple(
        key for key, value in sorted(components.items()) if float(value) > 1.0e-9
    )
    bottleneck = (
        "none"
        if not unresolved
        else max(components, key=lambda key: float(components[key]))
    )
    station_count = (
        int(current_station_count)
        if current_station_count is not None
        else int(len({record.pose_idx for record in estimator.measurements}))
    )
    _timer_add(timings, "total_s", time.perf_counter() - total_start)
    return RemainingMeasurementEstimate(
        current_station_count=station_count,
        estimated_remaining_stations=estimate,
        estimated_remaining_station_low=low,
        estimated_remaining_station_high=high,
        estimated_remaining_spectra_low=low * program_length,
        estimated_remaining_spectra_high=high * program_length,
        program_length=program_length,
        current_budget=float(weighted_budget),
        stop_budget=float(cfg.stop_budget),
        predicted_gain=float(weighted_gain),
        empirical_eta=float(eta),
        bottleneck=str(bottleneck),
        unresolved_factors=unresolved,
        components={key: float(value) for key, value in components.items()},
        gains={key: float(value) for key, value in gains.items()},
        isotope_details=isotope_details,
        timings={key: float(value) for key, value in timings.items()},
    )


def format_remaining_measurement_estimate(
    estimate: RemainingMeasurementEstimate,
) -> str:
    """Return a compact log line for a remaining-measurement estimate."""
    factors = ",".join(estimate.unresolved_factors) or "none"
    return (
        "Remaining measurement estimate: "
        f"stations={estimate.estimated_remaining_station_low}-"
        f"{estimate.estimated_remaining_station_high} "
        f"spectra={estimate.estimated_remaining_spectra_low}-"
        f"{estimate.estimated_remaining_spectra_high} "
        f"bottleneck={estimate.bottleneck} "
        f"budget={estimate.current_budget:.3g} "
        f"gain={estimate.predicted_gain:.3g} "
        f"eta={estimate.empirical_eta:.2f} "
        f"unresolved={factors} "
        f"timing_total={estimate.timings.get('total_s', 0.0):.3f}s "
        f"state_cache_hit={int(estimate.timings.get('state_cache_hit', 0.0) > 0.5)}"
    )
