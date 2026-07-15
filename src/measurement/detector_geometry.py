"""Shared detector geometry helpers for count and shield models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

DEFAULT_CRYSTAL_RADIUS_M = 0.038
DEFAULT_HOUSING_THICKNESS_M = 0.0015
DEFAULT_PF_DETECTOR_APERTURE_SAMPLES = 121


@dataclass(frozen=True)
class DetectorObservationGeometry:
    """Describe detector geometry used by PF and DSS-PP observation kernels."""

    count_radius_m: float
    aperture_radius_m: float
    aperture_samples: int
    aperture_sampling: str = "solid_angle_cone"


def normalize_detector_aperture_sampling(value: object | None) -> str:
    """Return a canonical detector-aperture sampling mode."""
    text = str(value or "solid_angle_cone").strip().lower().replace("-", "_")
    aliases = {
        "cone": "solid_angle_cone",
        "detector_cone": "solid_angle_cone",
        "solid_angle": "solid_angle_cone",
        "solid_angle_cone": "solid_angle_cone",
        "geometric_cone": "solid_angle_cone",
        "disk": "disk",
        "area_disk": "disk",
        "detector_disk": "disk",
    }
    if text not in aliases:
        raise ValueError(f"Unsupported detector aperture sampling mode: {value!r}")
    return aliases[text]


def detector_active_radius_m(
    detector_model: Mapping[str, Any] | None,
    *,
    default_radius_m: float = 0.0,
) -> float:
    """Return the active crystal radius used by detector-cps count geometry."""
    payload = detector_model if isinstance(detector_model, Mapping) else {}
    return max(float(payload.get("crystal_radius_m", default_radius_m)), 0.0)


def detector_outer_radius_cm(
    detector_model: Mapping[str, Any] | None,
    *,
    default_crystal_radius_m: float = DEFAULT_CRYSTAL_RADIUS_M,
    default_housing_thickness_m: float = DEFAULT_HOUSING_THICKNESS_M,
) -> float:
    """Return the crystal-plus-housing radius used by shield contact geometry."""
    payload = detector_model if isinstance(detector_model, Mapping) else {}
    crystal_radius_m = max(
        float(payload.get("crystal_radius_m", default_crystal_radius_m)),
        0.0,
    )
    housing_thickness_m = max(
        float(payload.get("housing_thickness_m", default_housing_thickness_m)),
        0.0,
    )
    return 100.0 * (crystal_radius_m + housing_thickness_m)


def detector_outer_radius_m(
    detector_model: Mapping[str, Any] | None,
    *,
    default_crystal_radius_m: float = DEFAULT_CRYSTAL_RADIUS_M,
    default_housing_thickness_m: float = DEFAULT_HOUSING_THICKNESS_M,
) -> float:
    """Return the crystal-plus-housing radius in meters."""
    return detector_outer_radius_cm(
        detector_model,
        default_crystal_radius_m=default_crystal_radius_m,
        default_housing_thickness_m=default_housing_thickness_m,
    ) / 100.0


def detector_observation_geometry_from_runtime_config(
    runtime_config: Mapping[str, Any] | None,
    *,
    default_aperture_samples: int = DEFAULT_PF_DETECTOR_APERTURE_SAMPLES,
) -> DetectorObservationGeometry:
    """
    Resolve detector geometry shared by PF likelihoods and DSS-PP scoring.

    ``count_radius_m`` follows the active crystal radius used by the Geant4
    detector-cps@1m source-rate normalization.  ``aperture_radius_m`` follows
    the source-to-detector target radius used for ray-level shield/obstacle
    sampling; by default this includes the housing because Geant4 directs
    detector-cone primaries to the detector outer radius.
    """
    payload = runtime_config if isinstance(runtime_config, Mapping) else {}
    detector_model = payload.get("detector_model", {})
    if not isinstance(detector_model, Mapping):
        detector_model = {}
    count_radius = detector_active_radius_m(detector_model)
    if "pf_detector_count_radius_m" in payload:
        count_radius = max(float(payload["pf_detector_count_radius_m"]), 0.0)
    aperture_radius = detector_outer_radius_m(detector_model)
    if "pf_detector_aperture_radius_m" in payload:
        aperture_radius = max(float(payload["pf_detector_aperture_radius_m"]), 0.0)
    samples = int(payload.get("pf_detector_aperture_samples", default_aperture_samples))
    sampling = normalize_detector_aperture_sampling(
        payload.get("pf_detector_aperture_sampling", "solid_angle_cone")
    )
    return DetectorObservationGeometry(
        count_radius_m=max(float(count_radius), 0.0),
        aperture_radius_m=max(float(aperture_radius), 0.0),
        aperture_samples=max(int(samples), 1),
        aperture_sampling=sampling,
    )
