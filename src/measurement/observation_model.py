"""Shared runtime observation-model construction for PF and DSS-PP."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from measurement.continuous_kernels import ContinuousKernel
from measurement.detector_geometry import (
    DetectorObservationGeometry,
    detector_observation_geometry_from_runtime_config,
    detector_outer_radius_cm,
)
from measurement.kernels import ShieldParams
from measurement.obstacle_assets import material_mu_cm_inv
from measurement.obstacles import ObstacleGrid
from measurement.shielding import (
    HVL_TVL_TABLE_MM,
    line_resolved_shield_mu_by_isotope,
    mu_by_isotope_from_tvl_mm,
)
from sim.shield_geometry import nested_shield_inner_radii_cm
from sim.shield_geometry import resolve_shield_thickness_config


@dataclass(frozen=True)
class RuntimeObservationModel:
    """Collect the shared PF/DSS-PP observation-kernel parameters."""

    detector_geometry: DetectorObservationGeometry
    shield_params: ShieldParams
    mu_by_isotope: dict[str, object]
    line_mu_by_isotope: dict[str, tuple[dict[str, float], ...]] | None
    transport_response_model: dict[str, object] | None
    obstacle_mu_by_isotope: dict[str, float] | None
    obstacle_height_m: float
    obstacle_buildup_coeff: float
    source_extent_radius_m: float
    source_extent_samples: int


def _buildup_runtime_config(runtime_config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the nested PF buildup configuration payload."""
    payload = runtime_config.get("pf_buildup", {})
    if isinstance(payload, Mapping):
        return payload
    return {}


def _transport_response_model_from_runtime_config(
    payload: Mapping[str, Any],
) -> dict[str, object] | None:
    """Return an inline or file-backed PF transport-response model."""
    inline_model = payload.get("pf_transport_response_model")
    if isinstance(inline_model, dict):
        return dict(inline_model)
    model_path = payload.get("pf_transport_response_model_path")
    if model_path is None:
        return None
    path = _resolve_runtime_model_path(model_path)
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if isinstance(loaded, dict) and isinstance(
        loaded.get("pf_transport_response_model"),
        dict,
    ):
        loaded = loaded["pf_transport_response_model"]
    if not isinstance(loaded, dict):
        raise ValueError(
            "pf_transport_response_model_path must point to a JSON object or "
            "an object containing 'pf_transport_response_model'."
        )
    return dict(loaded)


def _resolve_runtime_model_path(path_value: object) -> Path:
    """Resolve a runtime model path from cwd or repository root."""
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    repo_path = Path(__file__).resolve().parents[2] / path
    if repo_path.exists():
        return repo_path
    return cwd_path


def _explicit_obstacle_mu_by_isotope(
    payload: Mapping[str, Any],
) -> dict[str, float] | None:
    """Return an explicitly configured isotope obstacle-mu override."""
    for key in ("pf_obstacle_mu_by_isotope", "obstacle_mu_by_isotope"):
        raw = payload.get(key)
        if raw is None:
            continue
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"{key} must map isotope names to attenuation coefficients."
            )
        parsed = {
            str(isotope): max(float(value), 0.0)
            for isotope, value in raw.items()
        }
        return parsed
    return None


def _obstacle_mu_by_isotope_from_runtime_config(
    payload: Mapping[str, Any],
    *,
    isotopes: Sequence[str],
) -> dict[str, float] | None:
    """Resolve obstacle attenuation coefficients for the runtime material."""
    explicit = _explicit_obstacle_mu_by_isotope(payload)
    if explicit is not None:
        return explicit
    material = str(
        payload.get(
            "pf_obstacle_material",
            payload.get("obstacle_material", "concrete"),
        )
    )
    return {
        str(isotope): max(float(material_mu_cm_inv(material, str(isotope))), 0.0)
        for isotope in isotopes
    }


def build_runtime_observation_model(
    runtime_config: Mapping[str, Any] | None,
    *,
    isotopes: Sequence[str],
) -> RuntimeObservationModel:
    """Build the shared observation model used by PF, planning, and validation."""
    payload: Mapping[str, Any] = runtime_config if isinstance(runtime_config, Mapping) else {}
    detector_model = payload.get("detector_model", {})
    if not isinstance(detector_model, Mapping):
        detector_model = {}
    detector_geometry = detector_observation_geometry_from_runtime_config(payload)
    detector_outer_radius_cm_value = detector_outer_radius_cm(detector_model)
    shield_thickness = resolve_shield_thickness_config(dict(payload))
    inner_radius_fe_cm, inner_radius_pb_cm = nested_shield_inner_radii_cm(
        thickness_fe_cm=float(shield_thickness.thickness_fe_cm),
        detector_outer_radius_cm=detector_outer_radius_cm_value,
    )
    buildup = _buildup_runtime_config(payload)
    shield_params = ShieldParams(
        thickness_fe_cm=float(shield_thickness.thickness_fe_cm),
        thickness_pb_cm=float(shield_thickness.thickness_pb_cm),
        inner_radius_fe_cm=inner_radius_fe_cm,
        inner_radius_pb_cm=inner_radius_pb_cm,
        buildup_fe_coeff=float(
            buildup.get("fe_coeff", payload.get("pf_buildup_fe_coeff", 0.0))
        ),
        buildup_pb_coeff=float(
            buildup.get("pb_coeff", payload.get("pf_buildup_pb_coeff", 0.0))
        ),
    )
    mu_by_isotope = mu_by_isotope_from_tvl_mm(HVL_TVL_TABLE_MM, isotopes=isotopes)
    if not mu_by_isotope:
        mu_by_isotope = {
            str(isotope): {"fe": shield_params.mu_fe, "pb": shield_params.mu_pb}
            for isotope in isotopes
        }
    line_mu_by_isotope = None
    if bool(payload.get("pf_line_resolved_shield_attenuation", True)):
        line_mu_by_isotope = line_resolved_shield_mu_by_isotope(
            isotopes=isotopes,
            normalize_line_intensities=(
                str(payload.get("source_rate_model", "")).strip().lower()
                == "detector_cps_1m"
            ),
        )
    transport_response_model = _transport_response_model_from_runtime_config(payload)
    obstacle_mu_by_isotope = _obstacle_mu_by_isotope_from_runtime_config(
        payload,
        isotopes=isotopes,
    )
    return RuntimeObservationModel(
        detector_geometry=detector_geometry,
        shield_params=shield_params,
        mu_by_isotope=mu_by_isotope,
        line_mu_by_isotope=line_mu_by_isotope,
        transport_response_model=transport_response_model,
        obstacle_mu_by_isotope=obstacle_mu_by_isotope,
        obstacle_height_m=float(payload.get("obstacle_height_m", 2.0)),
        obstacle_buildup_coeff=float(
            buildup.get(
                "obstacle_coeff",
                payload.get("pf_obstacle_buildup_coeff", 0.0),
            )
        ),
        source_extent_radius_m=max(
            float(
                payload.get(
                    "pf_obstacle_source_extent_radius_m",
                    payload.get("pf_source_extent_radius_m", 0.0),
                )
            ),
            0.0,
        ),
        source_extent_samples=max(
            int(
                payload.get(
                    "pf_obstacle_source_extent_samples",
                    payload.get("pf_source_extent_samples", 1),
                )
            ),
            1,
        ),
    )


def continuous_kernel_from_observation_model(
    model: RuntimeObservationModel,
    *,
    obstacle_grid: ObstacleGrid | None,
    use_gpu: bool,
) -> ContinuousKernel:
    """Build a ContinuousKernel from the shared runtime observation model."""
    return ContinuousKernel(
        mu_by_isotope=model.mu_by_isotope,
        shield_params=model.shield_params,
        obstacle_grid=obstacle_grid,
        obstacle_height_m=model.obstacle_height_m,
        obstacle_mu_by_isotope=model.obstacle_mu_by_isotope,
        obstacle_buildup_coeff=(
            model.obstacle_buildup_coeff if obstacle_grid is not None else 0.0
        ),
        detector_radius_m=model.detector_geometry.count_radius_m,
        detector_aperture_radius_m=model.detector_geometry.aperture_radius_m,
        detector_aperture_samples=model.detector_geometry.aperture_samples,
        detector_aperture_sampling=model.detector_geometry.aperture_sampling,
        source_extent_radius_m=model.source_extent_radius_m,
        source_extent_samples=model.source_extent_samples,
        line_mu_by_isotope=model.line_mu_by_isotope,
        transport_response_model=model.transport_response_model,
        use_gpu=bool(use_gpu),
    )
