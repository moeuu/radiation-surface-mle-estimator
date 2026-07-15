"""Calibrate response-Poisson counts onto transport-truth count semantics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


DEFAULT_FEATURE_NAMES = (
    "log_count",
    "log_total_spectrum",
    "log_raw_to_count_ratio",
    "log_photopeak_to_count_ratio",
    "log_reduced_chi2",
)


def load_response_truth_calibration_payload(path: str | Path) -> dict[str, object]:
    """Load a response-truth calibration JSON payload."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("response truth calibration JSON root must be an object.")
    if isinstance(payload.get("response_poisson_truth_calibration"), Mapping):
        payload = payload["response_poisson_truth_calibration"]
    return dict(payload)


def response_truth_calibration_enabled(payload: Mapping[str, object] | None) -> bool:
    """Return whether a response-truth calibration payload is active."""
    if not isinstance(payload, Mapping):
        return False
    return bool(payload.get("enabled", True))


def response_truth_feature_vector(
    *,
    count: float,
    diagnostics: Mapping[str, object] | None,
    isotope: str,
    feature_names: Sequence[str] = DEFAULT_FEATURE_NAMES,
    spectrum_total: float | None = None,
    raw_count: float | None = None,
    photopeak_count: float | None = None,
) -> list[float]:
    """Return the runtime-observable feature vector for one isotope count."""
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    raw = _finite_positive(
        raw_count,
        _mapping_isotope_float(diagnostics.get("coefficients", {}), isotope),
        count,
    )
    photopeak = _finite_positive(
        photopeak_count,
        _mapping_isotope_float(diagnostics.get("photopeak_counts", {}), isotope),
        count,
    )
    reduced_chi2 = _finite_positive(
        _mapping_float(diagnostics, "reduced_chi2"),
        1.0,
    )
    total = _finite_positive(spectrum_total, count, 1.0)
    reference_count = _finite_positive(count, 1.0)

    values: list[float] = []
    for name in feature_names:
        feature = str(name)
        if feature == "log_count":
            values.append(float(np.log(reference_count)))
        elif feature == "log_total_spectrum":
            values.append(float(np.log(total)))
        elif feature == "log_raw_to_count_ratio":
            values.append(float(np.log(raw / reference_count)))
        elif feature == "log_photopeak_to_count_ratio":
            values.append(float(np.log(photopeak / reference_count)))
        elif feature == "log_reduced_chi2":
            values.append(float(np.log(max(reduced_chi2, 1.0))))
        else:
            raise ValueError(f"unsupported response-truth feature: {feature}")
    return values


def fit_local_knn_response_truth_calibration(
    records: Iterable[Mapping[str, object]],
    *,
    feature_names: Sequence[str] = DEFAULT_FEATURE_NAMES,
    neighbor_count: int = 8,
    min_group_points: int | None = None,
    distance_epsilon: float = 0.05,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """
    Fit a local detector-response calibration from validation records.

    The fitted model stores feature vectors and truth/response scale ratios by
    isotope and shield-pair group. Runtime evaluation uses only the fitted
    table plus features available from the current spectrum decomposition; it
    does not key on validation case names or transport truth values.
    """
    feature_names = tuple(str(name) for name in feature_names)
    neighbors = max(int(neighbor_count), 1)
    minimum = max(int(min_group_points or neighbors), 1)
    grouped: dict[str, list[dict[str, object]]] = {}
    fallback_records: dict[str, list[dict[str, object]]] = {}
    for record in records:
        isotope = str(record.get("isotope", ""))
        if not isotope:
            continue
        count = _mapping_float(record, "response_count")
        truth = _mapping_float(record, "truth_count")
        if count is None or truth is None or count <= 0.0 or truth <= 0.0:
            continue
        pair_id = _optional_int(record.get("shield_pair_id"))
        fallback_records.setdefault(isotope, []).append(dict(record))
        if pair_id is None:
            continue
        grouped.setdefault(_group_key(isotope, pair_id), []).append(dict(record))

    groups: dict[str, object] = {}
    skipped_groups: dict[str, float] = {}
    for key, values in sorted(grouped.items()):
        if len(values) < minimum:
            skipped_groups[key] = float(len(values))
            continue
        matrix, log_scales = _record_feature_matrix(
            values,
            feature_names=feature_names,
        )
        if matrix.size == 0:
            continue
        feature_mean = np.mean(matrix, axis=0)
        feature_std = np.std(matrix, axis=0)
        feature_std = np.where(feature_std < 1.0e-9, 1.0, feature_std)
        normalized = (matrix - feature_mean) / feature_std
        groups[key] = {
            "n": float(len(values)),
            "feature_mean": _json_float_list(feature_mean),
            "feature_std": _json_float_list(feature_std),
            "features": [
                _json_float_list(row)
                for row in np.asarray(normalized, dtype=float)
            ],
            "log_scales": _json_float_list(log_scales),
        }

    fallback_scale_by_isotope = {
        isotope: _fit_relative_through_origin_scale(values)
        for isotope, values in sorted(fallback_records.items())
    }
    payload_metadata = dict(metadata or {})
    payload_metadata.update(
        {
            "num_fit_records": float(
                sum(len(values) for values in fallback_records.values())
            ),
            "num_groups": float(len(groups)),
            "num_skipped_groups": float(len(skipped_groups)),
        }
    )
    return {
        "enabled": True,
        "measurement_space": "transport_truth_counts",
        "model": (
            "transport_truth_counts = local_knn_scale(feature_vector) "
            "* response_poisson_counts"
        ),
        "feature_names": list(feature_names),
        "group_keys": ["isotope", "shield_pair_id"],
        "neighbor_count": float(neighbors),
        "min_group_points": float(minimum),
        "distance_epsilon": float(max(distance_epsilon, 1.0e-12)),
        "min_scale": 0.05,
        "max_scale": 20.0,
        "fallback_scale_by_isotope": fallback_scale_by_isotope,
        "groups": groups,
        "skipped_group_fit_points": skipped_groups,
        "metadata": payload_metadata,
    }


def response_truth_calibration_scale(
    payload: Mapping[str, object] | None,
    *,
    isotope: str,
    shield_pair_id: int | None,
    count: float,
    diagnostics: Mapping[str, object] | None = None,
    spectrum_total: float | None = None,
    raw_count: float | None = None,
    photopeak_count: float | None = None,
) -> float:
    """Return the fitted response-truth scale for one isotope count."""
    if not response_truth_calibration_enabled(payload):
        return 1.0
    payload = dict(payload or {})
    groups = payload.get("groups", {})
    feature_names = tuple(
        str(name)
        for name in payload.get("feature_names", DEFAULT_FEATURE_NAMES)
        if str(name)
    )
    if not isinstance(groups, Mapping) or not feature_names:
        return _fallback_scale(payload, isotope)
    group = None
    if shield_pair_id is not None:
        group = groups.get(_group_key(isotope, int(shield_pair_id)))
    if not isinstance(group, Mapping):
        return _fallback_scale(payload, isotope)
    feature = np.asarray(
        response_truth_feature_vector(
            count=count,
            diagnostics=diagnostics,
            isotope=isotope,
            feature_names=feature_names,
            spectrum_total=spectrum_total,
            raw_count=raw_count,
            photopeak_count=photopeak_count,
        ),
        dtype=float,
    )
    scale = _local_knn_scale(payload, group, feature)
    return _bounded_scale(payload, scale)


def apply_response_truth_calibration(
    counts: Mapping[str, float],
    variances: Mapping[str, float],
    payload: Mapping[str, object] | None,
    *,
    shield_pair_id: int | None,
    diagnostics: Mapping[str, object] | None = None,
    spectrum_total: float | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, object]]:
    """Apply response-truth calibration to counts and propagate variances."""
    calibrated_counts = {str(key): float(value) for key, value in counts.items()}
    calibrated_variances = {
        str(key): float(value) for key, value in variances.items()
    }
    debug: dict[str, object] = {
        "enabled": response_truth_calibration_enabled(payload),
        "shield_pair_id": None if shield_pair_id is None else int(shield_pair_id),
        "scales": {},
        "original_counts": dict(calibrated_counts),
        "calibrated_counts": {},
    }
    if not response_truth_calibration_enabled(payload):
        debug["calibrated_counts"] = dict(calibrated_counts)
        return calibrated_counts, calibrated_variances, debug

    for isotope, count in list(calibrated_counts.items()):
        scale = response_truth_calibration_scale(
            payload,
            isotope=isotope,
            shield_pair_id=shield_pair_id,
            count=float(count),
            diagnostics=diagnostics,
            spectrum_total=spectrum_total,
        )
        calibrated_counts[isotope] = float(max(count * scale, 0.0))
        calibrated_variances[isotope] = float(
            max(calibrated_variances.get(isotope, 1.0) * scale * scale, 1.0)
        )
        debug["scales"][isotope] = float(scale)
    debug["calibrated_counts"] = dict(calibrated_counts)
    return calibrated_counts, calibrated_variances, debug


def _record_feature_matrix(
    records: Sequence[Mapping[str, object]],
    *,
    feature_names: Sequence[str],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return feature and log-scale arrays from calibration records."""
    features: list[list[float]] = []
    log_scales: list[float] = []
    for record in records:
        isotope = str(record.get("isotope", ""))
        count = _mapping_float(record, "response_count")
        truth = _mapping_float(record, "truth_count")
        if (
            not isotope
            or count is None
            or truth is None
            or count <= 0.0
            or truth <= 0.0
        ):
            continue
        feature = response_truth_feature_vector(
            count=count,
            diagnostics={
                "coefficients": {isotope: record.get("raw_count", count)},
                "photopeak_counts": {
                    isotope: record.get("photopeak_count", count)
                },
                "reduced_chi2": record.get("reduced_chi2", 1.0),
            },
            isotope=isotope,
            feature_names=feature_names,
            spectrum_total=_mapping_float(record, "spectrum_total"),
        )
        features.append(feature)
        log_scales.append(float(np.log(max(truth / count, 1.0e-12))))
    return np.asarray(features, dtype=float), np.asarray(log_scales, dtype=float)


def _local_knn_scale(
    payload: Mapping[str, object],
    group: Mapping[str, object],
    feature: NDArray[np.float64],
) -> float:
    """Return a local KNN scale for a normalized calibration group."""
    features = np.asarray(group.get("features", ()), dtype=float)
    log_scales = np.asarray(group.get("log_scales", ()), dtype=float)
    if features.ndim != 2 or log_scales.ndim != 1 or features.shape[0] == 0:
        return 1.0
    feature_mean = np.asarray(group.get("feature_mean", ()), dtype=float)
    feature_std = np.asarray(group.get("feature_std", ()), dtype=float)
    if feature_mean.shape != feature.shape or feature_std.shape != feature.shape:
        return 1.0
    feature_std = np.where(feature_std < 1.0e-9, 1.0, feature_std)
    normalized = (feature - feature_mean) / feature_std
    distances = np.sqrt(np.sum((features - normalized.reshape(1, -1)) ** 2, axis=1))
    if distances.size == 0:
        return 1.0
    neighbor_count = max(int(float(payload.get("neighbor_count", 8.0))), 1)
    neighbor_count = min(neighbor_count, distances.size)
    neighbor_idx = np.argsort(distances)[:neighbor_count]
    epsilon = max(float(payload.get("distance_epsilon", 0.05)), 1.0e-12)
    weights = 1.0 / (distances[neighbor_idx] + epsilon)
    scale = float(
        np.exp(
            np.sum(weights * log_scales[neighbor_idx])
            / max(float(np.sum(weights)), 1.0e-12)
        )
    )
    return scale


def _fit_relative_through_origin_scale(
    records: Sequence[Mapping[str, object]],
) -> float:
    """Fit a relative-error weighted through-origin fallback scale."""
    x_values: list[float] = []
    y_values: list[float] = []
    for record in records:
        count = _mapping_float(record, "response_count")
        truth = _mapping_float(record, "truth_count")
        if count is None or truth is None or count <= 0.0 or truth <= 0.0:
            continue
        x_values.append(float(count))
        y_values.append(float(truth))
    if not x_values:
        return 1.0
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    weights = 1.0 / np.maximum(y, 1.0) ** 2
    numerator = float(np.sum(weights * x * y))
    denominator = float(np.sum(weights * x * x))
    if denominator <= 0.0:
        return 1.0
    return float(max(numerator / denominator, 0.0))


def _fallback_scale(payload: Mapping[str, object], isotope: str) -> float:
    """Return isotope fallback scale from a payload."""
    fallback = payload.get("fallback_scale_by_isotope", {})
    if not isinstance(fallback, Mapping):
        return 1.0
    value = _mapping_float(fallback, isotope)
    return _bounded_scale(payload, 1.0 if value is None else value)


def _bounded_scale(payload: Mapping[str, object], scale: float) -> float:
    """Clamp a fitted scale to configured positive bounds."""
    if not np.isfinite(scale):
        return 1.0
    min_scale = max(float(payload.get("min_scale", 0.0)), 0.0)
    max_scale = max(float(payload.get("max_scale", float("inf"))), min_scale)
    return float(np.clip(max(float(scale), 0.0), min_scale, max_scale))


def _group_key(isotope: str, shield_pair_id: int) -> str:
    """Return a stable JSON group key for isotope and shield pair."""
    return f"{str(isotope)}|{int(shield_pair_id)}"


def _optional_int(value: object) -> int | None:
    """Return an integer from a finite payload value when present."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return int(result)


def _mapping_float(mapping: Mapping[str, object], key: str) -> float | None:
    """Return a finite mapping value as float when available."""
    value = mapping.get(key)
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _mapping_isotope_float(value: object, isotope: str) -> float | None:
    """Return a finite isotope-keyed mapping value as float when available."""
    if not isinstance(value, Mapping):
        return None
    result = value.get(isotope)
    if result is None:
        return None
    try:
        parsed = float(result)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _finite_positive(*values: object) -> float:
    """Return the first finite positive value from a list of candidates."""
    for value in values:
        if value is None:
            continue
        try:
            result = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(result) and result > 0.0:
            return result
    return 1.0


def _json_float_list(values: NDArray[np.float64]) -> list[float]:
    """Return a JSON-safe float list."""
    return [float(value) for value in np.asarray(values, dtype=float).reshape(-1)]
