"""Calibrate spectrum-derived net counts onto the PF measurement scale."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class NetResponseCalibration:
    """Store isotope-wise response factors from ideal counts to spectrum net counts."""

    scale_by_isotope: dict[str, float]
    scale_by_isotope_and_pair: dict[str, dict[int, float]] = field(default_factory=dict)
    fit_statistics: dict[str, dict[str, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def response_scale(
        self,
        isotope: str,
        default: float = 1.0,
        *,
        shield_pair_id: int | None = None,
    ) -> float:
        """Return the response scale for one isotope with a unity fallback."""
        if shield_pair_id is not None:
            pair_scales = self.scale_by_isotope_and_pair.get(str(isotope), {})
            pair_value = pair_scales.get(int(shield_pair_id))
            if pair_value is not None:
                return max(float(pair_value), 0.0)
        value = self.scale_by_isotope.get(isotope, default)
        return max(float(value), 0.0)

    def apply_expected_counts(
        self,
        counts: Mapping[str, float],
        *,
        shield_pair_id: int | None = None,
    ) -> dict[str, float]:
        """Map ideal inverse-square/shield counts into calibrated net-count space."""
        return {
            str(isotope): float(value)
            * self.response_scale(str(isotope), shield_pair_id=shield_pair_id)
            for isotope, value in counts.items()
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable calibration payload."""
        return {
            "measurement_space": "spectrum_net_counts",
            "model": "net_counts = response_scale[isotope] * ideal_inverse_square_shield_counts",
            "scale_by_isotope": {
                str(isotope): float(scale) for isotope, scale in self.scale_by_isotope.items()
            },
            "scale_by_isotope_and_pair": {
                str(isotope): {
                    str(pair_id): float(scale)
                    for pair_id, scale in sorted(pair_scales.items())
                }
                for isotope, pair_scales in sorted(
                    self.scale_by_isotope_and_pair.items()
                )
            },
            "fit_statistics": self.fit_statistics,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NetResponseCalibration":
        """Build a calibration object from a JSON-like mapping."""
        scales_payload = payload.get("scale_by_isotope", payload.get("isotope_scales", {}))
        if not isinstance(scales_payload, Mapping):
            raise ValueError("scale_by_isotope must be a JSON object.")
        pair_scales_payload = payload.get("scale_by_isotope_and_pair", {})
        if not isinstance(pair_scales_payload, Mapping):
            raise ValueError("scale_by_isotope_and_pair must be a JSON object.")
        stats_payload = payload.get("fit_statistics", {})
        metadata_payload = payload.get("metadata", {})
        if not isinstance(stats_payload, Mapping):
            raise ValueError("fit_statistics must be a JSON object.")
        if not isinstance(metadata_payload, Mapping):
            raise ValueError("metadata must be a JSON object.")
        return cls(
            scale_by_isotope={str(key): float(value) for key, value in scales_payload.items()},
            scale_by_isotope_and_pair={
                str(isotope): {
                    int(pair_id): float(scale)
                    for pair_id, scale in dict(pair_scales).items()
                }
                for isotope, pair_scales in pair_scales_payload.items()
                if isinstance(pair_scales, Mapping)
            },
            fit_statistics={
                str(key): {str(k): float(v) for k, v in dict(value).items()}
                for key, value in stats_payload.items()
                if isinstance(value, Mapping)
            },
            metadata=dict(metadata_payload),
        )

    @classmethod
    def load(cls, path: str | Path) -> "NetResponseCalibration":
        """Read a calibration JSON file."""
        payload = json.loads(Path(path).read_text())
        if not isinstance(payload, Mapping):
            raise ValueError("Calibration JSON root must be an object.")
        return cls.from_dict(payload)

    def save(self, path: str | Path) -> None:
        """Write the calibration JSON file with deterministic formatting."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


def fit_net_response_calibration(
    records: Iterable[Mapping[str, Any]],
    *,
    isotopes: Iterable[str] | None = None,
    min_theory_counts: float = 1.0,
    min_pair_fit_points: int = 1,
    pair_shrinkage_count: float = 0.0,
    metadata: Mapping[str, Any] | None = None,
) -> NetResponseCalibration:
    """
    Fit isotope-wise net response factors by weighted through-origin regression.

    Each record must contain ``isotope``, ``theory_counts``, and ``net_counts``.
    A record may optionally provide ``weight``. Without an explicit weight, the
    fit uses inverse Poisson variance, ``1 / max(net_counts, 1)``.

    Optional shield-pair scales form a hierarchical calibration: isotope-wise
    scales are always fitted first, and pair-conditioned scales are fitted only
    when a pair has enough support. ``pair_shrinkage_count`` adds pseudo-counts
    at the isotope-wise scale so poorly supported pair estimates shrink toward
    the physically shared detector-response normalization instead of overfitting
    a few noisy spectra.
    """
    grouped: dict[str, list[tuple[float, float, float]]] = {}
    grouped_by_pair: dict[str, dict[int, list[tuple[float, float, float]]]] = {}
    requested = {str(iso) for iso in isotopes} if isotopes is not None else None
    for record in records:
        isotope = str(record["isotope"])
        if requested is not None and isotope not in requested:
            continue
        theory = float(record["theory_counts"])
        net = float(record["net_counts"])
        if theory < float(min_theory_counts):
            continue
        weight = float(record.get("weight", 1.0 / max(net, 1.0)))
        if weight <= 0.0:
            continue
        grouped.setdefault(isotope, []).append((theory, net, weight))
        pair_id = _optional_int(record.get("shield_pair_id"))
        if pair_id is not None:
            grouped_by_pair.setdefault(isotope, {}).setdefault(pair_id, []).append(
                (theory, net, weight)
            )

    scale_by_isotope: dict[str, float] = {}
    scale_by_isotope_and_pair: dict[str, dict[int, float]] = {}
    fit_statistics: dict[str, dict[str, float]] = {}
    for isotope, values in grouped.items():
        fit = _fit_through_origin_scale(values)
        if fit is None:
            continue
        scale, standard_error, relative_error, theory_sum, net_sum = fit
        scale_by_isotope[isotope] = max(scale, 0.0)
        fit_statistics[isotope] = {
            "num_fit_points": float(len(values)),
            "scale": max(scale, 0.0),
            "standard_error": standard_error,
            "relative_standard_error": standard_error / max(abs(scale), 1e-12),
            "mean_relative_residual": float(np.mean(relative_error)) if relative_error.size else 0.0,
            "max_abs_relative_residual": float(np.max(np.abs(relative_error))) if relative_error.size else 0.0,
            "theory_counts_sum": float(theory_sum),
            "net_counts_sum": float(net_sum),
        }
        pair_stats: dict[str, float] = {}
        min_pair_points = max(int(min_pair_fit_points), 1)
        shrink_count = max(float(pair_shrinkage_count), 0.0)
        for pair_id, pair_values in sorted(grouped_by_pair.get(isotope, {}).items()):
            if len(pair_values) < min_pair_points:
                pair_stats[f"pair_{int(pair_id)}_num_fit_points"] = float(
                    len(pair_values)
                )
                pair_stats[f"pair_{int(pair_id)}_skipped_min_points"] = 1.0
                continue
            pair_fit = _fit_through_origin_scale(pair_values)
            if pair_fit is None:
                continue
            pair_scale, _, pair_relative_error, _, _ = pair_fit
            raw_pair_scale = max(float(pair_scale), 0.0)
            if shrink_count > 0.0 and raw_pair_scale > 0.0 and scale > 0.0:
                data_weight = float(len(pair_values))
                shrink_alpha = data_weight / (data_weight + shrink_count)
                pair_scale = float(
                    np.exp(
                        shrink_alpha * np.log(raw_pair_scale)
                        + (1.0 - shrink_alpha) * np.log(max(scale, 1.0e-12))
                    )
                )
            scale_by_isotope_and_pair.setdefault(isotope, {})[int(pair_id)] = max(
                float(pair_scale),
                0.0,
            )
            pair_stats[f"pair_{int(pair_id)}_scale"] = max(float(pair_scale), 0.0)
            pair_stats[f"pair_{int(pair_id)}_raw_scale"] = raw_pair_scale
            pair_stats[f"pair_{int(pair_id)}_shrinkage_count"] = shrink_count
            pair_stats[f"pair_{int(pair_id)}_num_fit_points"] = float(
                len(pair_values)
            )
            pair_stats[f"pair_{int(pair_id)}_max_abs_relative_residual"] = (
                float(np.max(np.abs(pair_relative_error)))
                if pair_relative_error.size
                else 0.0
            )
        if pair_stats:
            fit_statistics[isotope].update(pair_stats)

    fit_metadata = dict(metadata or {})
    fit_metadata.update(
        {
            "min_pair_fit_points": float(max(int(min_pair_fit_points), 1)),
            "pair_shrinkage_count": float(max(float(pair_shrinkage_count), 0.0)),
        }
    )
    return NetResponseCalibration(
        scale_by_isotope=scale_by_isotope,
        scale_by_isotope_and_pair=scale_by_isotope_and_pair,
        fit_statistics=fit_statistics,
        metadata=fit_metadata,
    )


def _optional_int(value: Any) -> int | None:
    """Return an integer when the payload contains a finite number."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return int(result)


def _fit_through_origin_scale(
    values: Iterable[tuple[float, float, float]],
) -> tuple[float, float, NDArray[np.float64], float, float] | None:
    """Fit one weighted through-origin scale and return residual diagnostics."""
    entries = list(values)
    if not entries:
        return None
    x = np.asarray([entry[0] for entry in entries], dtype=float)
    y = np.asarray([entry[1] for entry in entries], dtype=float)
    w = np.asarray([entry[2] for entry in entries], dtype=float)
    denom = float(np.sum(w * x * x))
    if denom <= 0.0:
        return None
    scale = float(np.sum(w * x * y) / denom)
    prediction = scale * x
    residual = y - prediction
    dof = max(int(x.size) - 1, 1)
    weighted_sse = float(np.sum(w * residual * residual))
    sigma2 = weighted_sse / float(dof)
    standard_error = float(np.sqrt(max(sigma2, 0.0) / denom))
    relative_error = np.divide(
        residual,
        np.maximum(np.abs(y), 1.0),
        out=np.zeros_like(residual),
        where=np.maximum(np.abs(y), 1.0) > 0.0,
    )
    return scale, standard_error, relative_error, float(np.sum(x)), float(np.sum(y))
