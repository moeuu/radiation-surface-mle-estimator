"""Continuous 3D kernel evaluations for the Chapter 3.3 measurement model.

Implements geometric and shielded kernels for arbitrary source coordinates,
consistent with Sec. 3.2–3.3 of the thesis (inverse-square law plus attenuation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict

import numpy as np
from numpy.typing import NDArray

from measurement.detector_geometry import normalize_detector_aperture_sampling
from measurement.kernels import ShieldParams
from measurement.obstacles import ObstacleGrid
from measurement.shielding import (
    CONCRETE_MU_CM_INV,
    LOCAL_POSITIVE_OCTANT_CENTER,
    OctantShield,
    SHIELD_GEOMETRY_SPHERICAL_OCTANT,
    generate_octant_orientations,
    octant_index_from_normal,
    path_length_cm,
    rotated_positive_octant_blocks_direction,
    resolve_mu_values,
    rotation_matrix_between_vectors,
    spherical_shell_path_length_cm,
    spherical_shell_path_length_cm_torch,
)

try:  # optional dependency
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    _TORCH_AVAILABLE = False


def _finite_sphere_geometric_term_torch(
    distance: "torch.Tensor",
    *,
    detector_radius_m: float,
    tol: "torch.Tensor",
) -> "torch.Tensor":
    """Return finite-sphere detector-cps@1m scaling for torch distances."""
    if torch is None:
        raise RuntimeError("torch is not available")
    radius = max(float(detector_radius_m), 0.0)
    if radius <= 0.0:
        dist = torch.clamp(distance, min=tol)
        return 1.0 / (dist**2)
    radius_t = torch.as_tensor(radius, device=distance.device, dtype=distance.dtype)
    d_eff = torch.maximum(torch.clamp(distance, min=tol), radius_t)
    ratio = torch.clamp(radius_t / torch.clamp(d_eff, min=tol), max=1.0)
    fraction = 0.5 * (1.0 - torch.sqrt(torch.clamp(1.0 - ratio * ratio, min=0.0)))
    reference = max(1.0, radius)
    ref_ratio = min(radius / reference, 1.0)
    ref_fraction = max(0.5 * (1.0 - float(np.sqrt(max(1.0 - ref_ratio * ref_ratio, 0.0)))), 1.0e-12)
    return fraction / ref_fraction


def _source_scale_rows_torch(
    source_scale: float | NDArray[np.float64] | "torch.Tensor",
    num_rows: int,
    *,
    device: "torch.device",
    dtype: "torch.dtype",
) -> "torch.Tensor":
    """Return a nonnegative row-wise source scale tensor for pair batches."""
    if torch is None:
        raise RuntimeError("torch is not available")
    scale = torch.as_tensor(source_scale, device=device, dtype=dtype)
    if scale.numel() == 1:
        return torch.clamp(scale.reshape(1, 1), min=0.0)
    scale = scale.reshape(-1)
    if int(scale.numel()) != int(num_rows):
        raise ValueError("source_scale must be scalar or contain one value per pair.")
    return torch.clamp(scale, min=0.0).view(int(num_rows), 1)


def geometric_term(detector: NDArray[np.float64], source: NDArray[np.float64]) -> float:
    """Inverse-square geometric term 1/d^2 for detector cps@1m scaling."""
    d = float(np.linalg.norm(detector - source))
    if d == 0.0:
        d = 1e-6
    return float(1.0 / (d**2))


def finite_sphere_geometric_term(
    detector: NDArray[np.float64],
    source: NDArray[np.float64],
    detector_radius_m: float,
) -> float:
    """
    Return detector-cps@1m scaling for a finite spherical detector.

    For a configured spherical detector, ``intensity_cps_1m`` is defined as the
    expected detector count rate at 1 m.  The near-field scaling should
    therefore use the sphere solid angle relative to the 1 m solid angle, not a
    point-detector singularity.  When no detector radius is configured this
    falls back to the inverse-square term.
    """
    radius = max(float(detector_radius_m), 0.0)
    if radius <= 0.0:
        return geometric_term(detector, source)
    d = float(np.linalg.norm(np.asarray(detector, dtype=float) - np.asarray(source, dtype=float)))
    d_eff = max(d, radius)
    reference = max(1.0, radius)

    def _sphere_fraction(distance: float) -> float:
        """Return the external point-source solid-angle fraction of a sphere."""
        if distance <= radius:
            return 0.5
        ratio = min(radius / max(distance, 1.0e-12), 1.0)
        return 0.5 * (1.0 - float(np.sqrt(max(1.0 - ratio * ratio, 0.0))))

    ref_fraction = max(_sphere_fraction(reference), 1.0e-12)
    return float(_sphere_fraction(d_eff) / ref_fraction)


def _normalize_isotope_key(isotope: str) -> str:
    """Return a normalized isotope key for table lookups."""
    return re.sub(r"[^A-Za-z0-9]", "", str(isotope)).upper()


_TRANSPORT_RESPONSE_COEFFICIENT_ALIASES: dict[str, tuple[str, ...]] = {
    "shield": ("shield", "shield_tau"),
    "obstacle": ("obstacle", "obstacle_tau"),
    "shield_squared": ("shield_squared", "shield_tau_squared"),
    "obstacle_squared": ("obstacle_squared", "obstacle_tau_squared"),
    "shield_obstacle": ("shield_obstacle", "shield_tau_obstacle_tau"),
    "fe": ("fe", "fe_tau", "tau_fe"),
    "pb": ("pb", "pb_tau", "tau_pb"),
    "fe_squared": ("fe_squared", "fe_tau_squared", "tau_fe_squared"),
    "pb_squared": ("pb_squared", "pb_tau_squared", "tau_pb_squared"),
    "fe_pb": ("fe_pb", "fe_tau_pb_tau", "tau_fe_tau_pb"),
    "fe_obstacle": (
        "fe_obstacle",
        "fe_tau_obstacle_tau",
        "tau_fe_tau_obstacle",
    ),
    "pb_obstacle": (
        "pb_obstacle",
        "pb_tau_obstacle_tau",
        "tau_pb_tau_obstacle",
    ),
    "distance": (
        "distance",
        "distance_m",
        "source_distance",
        "source_detector_distance_m",
    ),
    "distance_shield": (
        "distance_shield",
        "distance_shield_tau",
        "shield_distance",
        "source_distance_shield_tau",
    ),
    "distance_fe": (
        "distance_fe",
        "distance_fe_tau",
        "fe_distance",
        "source_distance_fe_tau",
    ),
    "distance_pb": (
        "distance_pb",
        "distance_pb_tau",
        "pb_distance",
        "source_distance_pb_tau",
    ),
    "distance_obstacle": (
        "distance_obstacle",
        "distance_obstacle_tau",
        "obstacle_distance",
        "source_distance_obstacle_tau",
    ),
}


def transport_response_payload_for_isotope(
    model: dict[str, object] | None,
    isotope: str,
) -> dict[str, object]:
    """Return the configured transport-response payload for one isotope."""
    if not isinstance(model, dict) or bool(model.get("enabled", True)) is False:
        return {}
    by_isotope = model.get("by_isotope", {})
    if not isinstance(by_isotope, dict):
        return {}
    payload = by_isotope.get(str(isotope))
    if payload is None:
        normalized = {
            _normalize_isotope_key(key): value
            for key, value in by_isotope.items()
        }
        payload = normalized.get(_normalize_isotope_key(isotope))
    return dict(payload) if isinstance(payload, dict) else {}


def transport_response_coefficients_from_payload(
    payload: dict[str, object],
    *,
    default_min_log: float = -50.0,
    default_max_log: float = 50.0,
) -> tuple[dict[str, float], float, float]:
    """Return optical-depth response coefficients and log-scale bounds."""
    if not isinstance(payload, dict) or not payload:
        return {}, float(default_min_log), float(default_max_log)
    coeffs = payload.get("tau_coefficients", {})
    if not isinstance(coeffs, dict):
        coeffs = {}
    parsed: dict[str, float] = {}
    for canonical, keys in _TRANSPORT_RESPONSE_COEFFICIENT_ALIASES.items():
        for key in keys:
            if key in coeffs:
                parsed[canonical] = float(coeffs[key])
                break
    min_log = float(payload.get("min_log_scale", default_min_log))
    max_log = float(payload.get("max_log_scale", default_max_log))
    if max_log < min_log:
        min_log, max_log = max_log, min_log
    return parsed, min_log, max_log


def transport_response_feature_caps_from_payload(
    payload: dict[str, object],
) -> tuple[
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    """Return optional optical-depth feature caps from one payload."""
    caps = payload.get("tau_feature_caps", payload.get("feature_caps", {}))
    if not isinstance(caps, dict):
        return None, None, None, None, None, None, None, None

    def _cap_value(*names: str) -> float | None:
        """Return the first finite nonnegative cap from candidate keys."""
        for name in names:
            if name not in caps:
                continue
            value = float(caps[name])
            if np.isfinite(value) and value >= 0.0:
                return value
        return None

    return (
        _cap_value("shield", "shield_tau", "tau_shield"),
        _cap_value("obstacle", "obstacle_tau", "tau_obstacle"),
        _cap_value("fe", "fe_tau", "tau_fe"),
        _cap_value("pb", "pb_tau", "tau_pb"),
        _cap_value(
            "distance_shield",
            "distance_shield_tau",
            "source_distance_shield_tau",
        ),
        _cap_value("distance_fe", "distance_fe_tau", "source_distance_fe_tau"),
        _cap_value("distance_pb", "distance_pb_tau", "source_distance_pb_tau"),
        _cap_value(
            "distance_obstacle",
            "distance_obstacle_tau",
            "source_distance_obstacle_tau",
        ),
    )


def capped_transport_response_feature(
    value: float,
    cap: float | None,
) -> float:
    """Return a nonnegative optical-depth feature with an optional cap."""
    feature = max(float(value), 0.0)
    if cap is None:
        return feature
    return min(feature, float(cap))


def transport_response_pair_log_scale_from_payload(
    payload: dict[str, object],
    pair_id: int,
) -> float:
    """Return the log-scale transport-response term for one shield pair."""
    if not isinstance(payload, dict) or not payload:
        return 0.0
    scale = float(payload.get("scale", 1.0))
    pair_scales = payload.get("scale_by_pair", {})
    if isinstance(pair_scales, dict):
        value = pair_scales.get(str(int(pair_id)), pair_scales.get(int(pair_id)))
        if value is not None:
            scale = float(value)
    return float(np.log(max(scale, 1.0e-12)))


def transport_response_factor_from_payload(
    payload: dict[str, object],
    *,
    pair_id: int,
    shield_tau_feature: float,
    obstacle_tau_feature: float,
    fe_tau_feature: float | None = None,
    pb_tau_feature: float | None = None,
    distance_feature: float | None = None,
    distance_shield_feature: float | None = None,
    default_min_log: float = -50.0,
    default_max_log: float = 50.0,
) -> float:
    """Return source-local transport-response factor from one payload."""
    if not isinstance(payload, dict) or not payload:
        return 1.0
    coeffs, min_log, max_log = transport_response_coefficients_from_payload(
        payload,
        default_min_log=default_min_log,
        default_max_log=default_max_log,
    )
    (
        shield_cap,
        obstacle_cap,
        fe_cap,
        pb_cap,
        distance_shield_cap,
        distance_fe_cap,
        distance_pb_cap,
        distance_obstacle_cap,
    ) = (
        transport_response_feature_caps_from_payload(payload)
    )
    shield_tau_raw = max(float(shield_tau_feature), 0.0)
    obstacle_tau_raw = max(float(obstacle_tau_feature), 0.0)
    fe_tau_raw = max(float(fe_tau_feature or 0.0), 0.0)
    pb_tau_raw = max(float(pb_tau_feature or 0.0), 0.0)
    shield_tau = capped_transport_response_feature(shield_tau_raw, shield_cap)
    obstacle_tau = capped_transport_response_feature(
        obstacle_tau_raw,
        obstacle_cap,
    )
    fe_tau = capped_transport_response_feature(fe_tau_raw, fe_cap)
    pb_tau = capped_transport_response_feature(pb_tau_raw, pb_cap)
    distance = max(float(distance_feature or 0.0), 0.0)
    distance_shield = (
        max(float(distance_shield_feature), 0.0)
        if distance_shield_feature is not None
        else distance * shield_tau_raw
    )
    distance_shield = capped_transport_response_feature(
        distance_shield,
        distance_shield_cap,
    )
    distance_fe = capped_transport_response_feature(
        distance * fe_tau_raw,
        distance_fe_cap,
    )
    distance_pb = capped_transport_response_feature(
        distance * pb_tau_raw,
        distance_pb_cap,
    )
    distance_obstacle = capped_transport_response_feature(
        distance * obstacle_tau_raw,
        distance_obstacle_cap,
    )
    log_scale = transport_response_pair_log_scale_from_payload(payload, pair_id)
    log_scale += float(coeffs.get("shield", 0.0)) * shield_tau
    log_scale += float(coeffs.get("obstacle", 0.0)) * obstacle_tau
    log_scale += float(coeffs.get("shield_squared", 0.0)) * shield_tau * shield_tau
    log_scale += (
        float(coeffs.get("obstacle_squared", 0.0))
        * obstacle_tau
        * obstacle_tau
    )
    log_scale += float(coeffs.get("shield_obstacle", 0.0)) * shield_tau * obstacle_tau
    log_scale += float(coeffs.get("fe", 0.0)) * fe_tau
    log_scale += float(coeffs.get("pb", 0.0)) * pb_tau
    log_scale += float(coeffs.get("fe_squared", 0.0)) * fe_tau * fe_tau
    log_scale += float(coeffs.get("pb_squared", 0.0)) * pb_tau * pb_tau
    log_scale += float(coeffs.get("fe_pb", 0.0)) * fe_tau * pb_tau
    log_scale += float(coeffs.get("fe_obstacle", 0.0)) * fe_tau * obstacle_tau
    log_scale += float(coeffs.get("pb_obstacle", 0.0)) * pb_tau * obstacle_tau
    log_scale += float(coeffs.get("distance", 0.0)) * distance
    log_scale += float(coeffs.get("distance_shield", 0.0)) * distance_shield
    log_scale += float(coeffs.get("distance_fe", 0.0)) * distance_fe
    log_scale += float(coeffs.get("distance_pb", 0.0)) * distance_pb
    log_scale += (
        float(coeffs.get("distance_obstacle", 0.0)) * distance_obstacle
    )
    log_scale = min(max(log_scale, min_log), max_log)
    return float(np.exp(log_scale))


def resolve_obstacle_mu_cm_inv(
    isotope: str,
    mu_by_isotope: Dict[str, float] | None = None,
) -> float:
    """Resolve concrete obstacle attenuation coefficient in 1/cm for an isotope."""
    table = mu_by_isotope if mu_by_isotope is not None else CONCRETE_MU_CM_INV
    if isotope in table:
        return float(table[isotope])
    normalized = {_normalize_isotope_key(key): float(value) for key, value in table.items()}
    norm_key = _normalize_isotope_key(isotope)
    if norm_key in normalized:
        return normalized[norm_key]
    raise ValueError(
        "Concrete obstacle attenuation is enabled but no coefficient is defined "
        f"for isotope {isotope!r}."
    )


def segment_box_intersection_length_m(
    source_pos: NDArray[np.float64],
    detector_pos: NDArray[np.float64],
    box_m: NDArray[np.float64],
    tol: float = 1e-12,
) -> float:
    """Return the line-segment path length inside one axis-aligned box in meters."""
    source = np.asarray(source_pos, dtype=float)
    detector = np.asarray(detector_pos, dtype=float)
    box = np.asarray(box_m, dtype=float)
    if source.shape != (3,) or detector.shape != (3,) or box.shape != (6,):
        raise ValueError("source_pos, detector_pos, and box_m must have shapes (3,), (3,), and (6,).")
    direction = detector - source
    segment_length = float(np.linalg.norm(direction))
    if segment_length <= tol:
        return 0.0
    lower = box[:3]
    upper = box[3:]
    t_enter = 0.0
    t_exit = 1.0
    for axis in range(3):
        value = source[axis]
        delta = direction[axis]
        lo = lower[axis]
        hi = upper[axis]
        if abs(delta) <= tol:
            if value < lo or value > hi:
                return 0.0
            continue
        t0 = (lo - value) / delta
        t1 = (hi - value) / delta
        if t0 > t1:
            t0, t1 = t1, t0
        t_enter = max(t_enter, float(t0))
        t_exit = min(t_exit, float(t1))
        if t_exit <= t_enter:
            return 0.0
    return max(0.0, t_exit - t_enter) * segment_length


def obstacle_path_length_cm(
    source_pos: NDArray[np.float64],
    detector_pos: NDArray[np.float64],
    obstacle_boxes_m: NDArray[np.float64],
) -> float:
    """Return total source-detector path length inside obstacle boxes in centimeters."""
    return float(
        np.sum(
            obstacle_path_lengths_by_box_cm(
                source_pos=source_pos,
                detector_pos=detector_pos,
                obstacle_boxes_m=obstacle_boxes_m,
            )
        )
    )


def obstacle_path_lengths_by_box_cm(
    source_pos: NDArray[np.float64],
    detector_pos: NDArray[np.float64],
    obstacle_boxes_m: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return per-box source-detector path lengths in centimeters."""
    boxes = np.asarray(obstacle_boxes_m, dtype=float)
    if boxes.size == 0:
        return np.zeros(0, dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] != 6:
        raise ValueError("obstacle_boxes_m must be shaped (N, 6).")
    return np.asarray(
        [
            100.0 * segment_box_intersection_length_m(source_pos, detector_pos, box)
            for box in boxes
        ],
        dtype=float,
    )


def obstacle_optical_depth(
    source_pos: NDArray[np.float64],
    detector_pos: NDArray[np.float64],
    obstacle_boxes_m: NDArray[np.float64],
    obstacle_mu_cm_inv_by_box: NDArray[np.float64],
) -> float:
    """Return summed material optical depth through known obstacle components."""
    boxes = np.asarray(obstacle_boxes_m, dtype=float)
    mu_values = np.asarray(obstacle_mu_cm_inv_by_box, dtype=float)
    if boxes.size == 0:
        return 0.0
    if boxes.ndim != 2 or boxes.shape[1] != 6:
        raise ValueError("obstacle_boxes_m must be shaped (N, 6).")
    if mu_values.shape != (boxes.shape[0],):
        raise ValueError("obstacle_mu_cm_inv_by_box must match obstacle box count.")
    path_cm_by_box = obstacle_path_lengths_by_box_cm(
        source_pos=source_pos,
        detector_pos=detector_pos,
        obstacle_boxes_m=boxes,
    )
    return float(np.sum(mu_values * path_cm_by_box))


def obstacle_log_attenuation_matrix(
    sources_xyz: NDArray[np.float64],
    detector_poses_xyz: NDArray[np.float64],
    obstacle_boxes_m: NDArray[np.float64],
    obstacle_mu_cm_inv_by_box: NDArray[np.float64],
    *,
    element_budget: int = 4_000_000,
    tol: float = 1.0e-12,
) -> NDArray[np.float64]:
    """Return log obstacle transmission for all detector-source pairs."""
    sources = np.asarray(sources_xyz, dtype=float)
    detectors = np.asarray(detector_poses_xyz, dtype=float)
    boxes = np.asarray(obstacle_boxes_m, dtype=float)
    mu_values = np.asarray(obstacle_mu_cm_inv_by_box, dtype=float)
    if sources.ndim != 2 or sources.shape[1] != 3:
        raise ValueError("sources_xyz must be shaped (N, 3).")
    if detectors.ndim != 2 or detectors.shape[1] != 3:
        raise ValueError("detector_poses_xyz must be shaped (M, 3).")
    if boxes.size == 0:
        return np.zeros((detectors.shape[0], sources.shape[0]), dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] != 6:
        raise ValueError("obstacle_boxes_m must be shaped (B, 6).")
    if mu_values.shape != (boxes.shape[0],):
        raise ValueError("obstacle_mu_cm_inv_by_box must match obstacle box count.")

    pose_count = int(detectors.shape[0])
    source_count = int(sources.shape[0])
    box_count = int(boxes.shape[0])
    if pose_count == 0 or source_count == 0:
        return np.zeros((pose_count, source_count), dtype=float)

    budget = max(int(element_budget), 1)
    chunk = max(1, min(pose_count, budget // max(1, source_count * box_count)))
    out = np.zeros((pose_count, source_count), dtype=float)
    lower = boxes[:, :3]
    upper = boxes[:, 3:]
    mu = mu_values.reshape(1, 1, box_count)
    tol_value = float(tol)
    src = sources.reshape(1, source_count, 1, 3)
    for start in range(0, pose_count, chunk):
        stop = min(start + chunk, pose_count)
        det = detectors[start:stop].reshape(stop - start, 1, 1, 3)
        direction = det - src
        distance = np.linalg.norm(direction[:, :, 0, :], axis=2)
        t_min_axes: list[NDArray[np.float64]] = []
        t_max_axes: list[NDArray[np.float64]] = []
        for axis in range(3):
            value = src[..., axis]
            step = direction[..., axis]
            lo = lower[:, axis].reshape(1, 1, box_count)
            hi = upper[:, axis].reshape(1, 1, box_count)
            parallel = np.abs(step) <= tol_value
            inside = (value >= lo) & (value <= hi)
            safe_step = np.where(parallel, 1.0, step)
            t0 = (lo - value) / safe_step
            t1 = (hi - value) / safe_step
            axis_min = np.minimum(t0, t1)
            axis_max = np.maximum(t0, t1)
            axis_min = np.where(parallel & inside, -np.inf, axis_min)
            axis_max = np.where(parallel & inside, np.inf, axis_max)
            axis_min = np.where(parallel & ~inside, np.inf, axis_min)
            axis_max = np.where(parallel & ~inside, -np.inf, axis_max)
            t_min_axes.append(axis_min)
            t_max_axes.append(axis_max)
        t_enter = np.maximum(np.stack(t_min_axes, axis=-1).max(axis=-1), 0.0)
        t_exit = np.minimum(np.stack(t_max_axes, axis=-1).min(axis=-1), 1.0)
        valid = t_exit > t_enter
        span = np.zeros_like(t_exit)
        np.subtract(t_exit, t_enter, out=span, where=valid)
        span = np.where(np.isfinite(span), span, 0.0)
        length_cm = span * distance[:, :, None] * 100.0
        tau = np.sum(length_cm * mu, axis=2)
        out[start:stop] = -tau
    return out


def segment_sphere_intersection_length_m(
    source_pos: NDArray[np.float64],
    target_pos: NDArray[np.float64],
    center_pos: NDArray[np.float64],
    radius_m: float,
    tol: float = 1e-12,
) -> float:
    """Return segment length inside a sphere in meters."""
    source = np.asarray(source_pos, dtype=float)
    target = np.asarray(target_pos, dtype=float)
    center = np.asarray(center_pos, dtype=float)
    radius = max(float(radius_m), 0.0)
    direction = target - source
    segment_length = float(np.linalg.norm(direction))
    if radius <= 0.0 or segment_length <= tol:
        return 0.0
    rel_source = source - center
    a = float(np.dot(direction, direction))
    b = 2.0 * float(np.dot(rel_source, direction))
    c = float(np.dot(rel_source, rel_source)) - radius * radius
    rel_target = target - center
    source_inside = c <= 0.0
    target_inside = float(np.dot(rel_target, rel_target)) <= radius * radius
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return segment_length if source_inside and target_inside else 0.0
    sqrt_disc = float(np.sqrt(max(discriminant, 0.0)))
    t0 = (-b - sqrt_disc) / (2.0 * a)
    t1 = (-b + sqrt_disc) / (2.0 * a)
    enter = max(0.0, min(t0, t1))
    exit_ = min(1.0, max(t0, t1))
    if exit_ <= enter:
        return segment_length if source_inside and target_inside else 0.0
    return max(0.0, exit_ - enter) * segment_length


def segment_spherical_shell_path_length_cm(
    source_pos: NDArray[np.float64],
    target_pos: NDArray[np.float64],
    center_pos: NDArray[np.float64],
    inner_radius_cm: float,
    outer_radius_cm: float,
    blocked: bool,
) -> float:
    """Return segment length through a spherical shell centered at center_pos."""
    if not blocked:
        return 0.0
    inner_m = max(0.0, float(inner_radius_cm) / 100.0)
    outer_m = max(inner_m, float(outer_radius_cm) / 100.0)
    if outer_m <= inner_m:
        return 0.0
    outer_length = segment_sphere_intersection_length_m(
        source_pos,
        target_pos,
        center_pos,
        outer_m,
    )
    inner_length = segment_sphere_intersection_length_m(
        source_pos,
        target_pos,
        center_pos,
        inner_m,
    )
    return float(100.0 * max(outer_length - inner_length, 0.0))


def segment_rotated_octant_shell_path_length_cm(
    source_pos: NDArray[np.float64],
    target_pos: NDArray[np.float64],
    center_pos: NDArray[np.float64],
    shield_normal: NDArray[np.float64],
    inner_radius_cm: float,
    outer_radius_cm: float,
    tol: float = 1.0e-12,
) -> float:
    """Return exact segment length inside a rotated local +X/+Y/+Z octant shell."""
    source = np.asarray(source_pos, dtype=float)
    target = np.asarray(target_pos, dtype=float)
    center = np.asarray(center_pos, dtype=float)
    rotation = rotation_matrix_between_vectors(
        LOCAL_POSITIVE_OCTANT_CENTER,
        np.asarray(shield_normal, dtype=float),
    )
    source_local = rotation.T @ (source - center)
    target_local = rotation.T @ (target - center)
    delta = target_local - source_local
    segment_length = float(np.linalg.norm(delta))
    if segment_length <= tol:
        return 0.0
    inner_m = max(0.0, float(inner_radius_cm) / 100.0)
    outer_m = max(inner_m, float(outer_radius_cm) / 100.0)
    if outer_m <= inner_m:
        return 0.0
    breakpoints: list[float] = [0.0, 1.0]
    a = float(np.dot(delta, delta))
    b = 2.0 * float(np.dot(source_local, delta))
    for radius in (inner_m, outer_m):
        c = float(np.dot(source_local, source_local)) - radius * radius
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            continue
        root = float(np.sqrt(max(discriminant, 0.0)))
        breakpoints.extend(
            value
            for value in (
                (-b - root) / (2.0 * a),
                (-b + root) / (2.0 * a),
            )
            if -tol <= value <= 1.0 + tol
        )
    for axis in range(3):
        if abs(float(delta[axis])) <= tol:
            continue
        value = -float(source_local[axis]) / float(delta[axis])
        if -tol <= value <= 1.0 + tol:
            breakpoints.append(value)
    clipped = sorted({float(np.clip(value, 0.0, 1.0)) for value in breakpoints})
    length = 0.0
    for left, right in zip(clipped[:-1], clipped[1:]):
        if right - left <= tol:
            continue
        mid = 0.5 * (left + right)
        point = source_local + mid * delta
        radius_sq = float(np.dot(point, point))
        inside_shell = (inner_m * inner_m - tol) <= radius_sq <= (outer_m * outer_m + tol)
        inside_octant = bool(np.all(point >= -tol))
        if inside_shell and inside_octant:
            length += (right - left) * segment_length
    return float(100.0 * length)


def segment_rotated_octant_shell_path_length_cm_torch(
    source_pos: "torch.Tensor",
    target_pos: "torch.Tensor",
    center_pos: "torch.Tensor",
    shield_normal: NDArray[np.float64] | None,
    inner_radius_cm: float,
    outer_radius_cm: float,
    tol: float = 1.0e-9,
    rotation: "torch.Tensor" | None = None,
) -> "torch.Tensor":
    """Return exact segment length inside a rotated octant shell for torch tensors."""
    if torch is None:
        raise RuntimeError("torch is not available")
    if rotation is None:
        if shield_normal is None:
            raise ValueError("shield_normal is required when rotation is not provided.")
        rotation_np = rotation_matrix_between_vectors(
            LOCAL_POSITIVE_OCTANT_CENTER,
            np.asarray(shield_normal, dtype=float),
        )
        rotation = torch.as_tensor(
            rotation_np,
            device=source_pos.device,
            dtype=source_pos.dtype,
        )
    else:
        rotation = rotation.to(device=source_pos.device, dtype=source_pos.dtype)
    source_local = (source_pos - center_pos) @ rotation
    target_local = (target_pos - center_pos) @ rotation
    delta = target_local - source_local
    segment_length = torch.linalg.norm(delta, dim=-1)
    inner_m = max(0.0, float(inner_radius_cm) / 100.0)
    outer_m = max(inner_m, float(outer_radius_cm) / 100.0)
    if outer_m <= inner_m:
        return torch.zeros_like(segment_length)
    a = torch.sum(delta * delta, dim=-1)
    b = 2.0 * torch.sum(source_local * delta, dim=-1)
    breakpoints = [
        torch.zeros_like(segment_length),
        torch.ones_like(segment_length),
    ]
    for radius in (inner_m, outer_m):
        c = torch.sum(source_local * source_local, dim=-1) - float(radius * radius)
        discriminant = b * b - 4.0 * a * c
        root = torch.sqrt(torch.clamp(discriminant, min=0.0))
        denom = torch.clamp(2.0 * a, min=float(tol))
        valid = discriminant >= 0.0
        for sign in (-1.0, 1.0):
            value = (-b + sign * root) / denom
            value = torch.where(valid, value, torch.zeros_like(value))
            breakpoints.append(torch.clamp(value, 0.0, 1.0))
    for axis in range(3):
        axis_delta = delta[..., axis]
        valid = torch.abs(axis_delta) > float(tol)
        value = -source_local[..., axis] / torch.where(
            valid,
            axis_delta,
            torch.ones_like(axis_delta),
        )
        value = torch.where(valid, value, torch.zeros_like(value))
        breakpoints.append(torch.clamp(value, 0.0, 1.0))
    ordered = torch.sort(torch.stack(breakpoints, dim=-1), dim=-1).values
    left = ordered[..., :-1]
    right = ordered[..., 1:]
    width = torch.clamp(right - left, min=0.0)
    mid = 0.5 * (left + right)
    point = source_local.unsqueeze(-2) + mid.unsqueeze(-1) * delta.unsqueeze(-2)
    radius_sq = torch.sum(point * point, dim=-1)
    inside_shell = (radius_sq >= inner_m * inner_m - float(tol)) & (
        radius_sq <= outer_m * outer_m + float(tol)
    )
    inside_octant = torch.all(point >= -float(tol), dim=-1)
    length_m = torch.sum(
        torch.where(inside_shell & inside_octant, width, torch.zeros_like(width)),
        dim=-1,
    ) * segment_length
    return torch.where(segment_length > float(tol), 100.0 * length_m, torch.zeros_like(length_m))


def obstacle_path_lengths_cm_torch(
    positions: "torch.Tensor",
    detector_pos: "torch.Tensor",
    obstacle_boxes_m: "torch.Tensor",
    tol: float = 1e-9,
) -> "torch.Tensor":
    """Return batched obstacle path lengths through axis-aligned boxes in centimeters."""
    lengths_by_box = obstacle_path_lengths_by_box_cm_torch(
        positions=positions,
        detector_pos=detector_pos,
        obstacle_boxes_m=obstacle_boxes_m,
        tol=tol,
    )
    return torch.sum(lengths_by_box, dim=-1)


def obstacle_path_lengths_by_box_cm_torch(
    positions: "torch.Tensor",
    detector_pos: "torch.Tensor",
    obstacle_boxes_m: "torch.Tensor",
    tol: float = 1e-9,
) -> "torch.Tensor":
    """Return batched per-box obstacle path lengths in centimeters."""
    if torch is None:
        raise RuntimeError("torch is not available")
    if obstacle_boxes_m.numel() == 0:
        return torch.zeros(
            (*positions.shape[:-1], 0),
            device=positions.device,
            dtype=positions.dtype,
        )
    if obstacle_boxes_m.ndim != 2 or obstacle_boxes_m.shape[1] != 6:
        raise ValueError("obstacle_boxes_m must be shaped (N, 6).")
    detector = detector_pos.to(device=positions.device, dtype=positions.dtype)
    detector = detector.view(*([1] * (positions.ndim - 1)), 3)
    direction = detector - positions
    distance = torch.linalg.norm(direction, dim=-1)
    p0 = positions.unsqueeze(-2)
    delta = direction.unsqueeze(-2)
    lower = obstacle_boxes_m[:, :3].to(device=positions.device, dtype=positions.dtype)
    upper = obstacle_boxes_m[:, 3:].to(device=positions.device, dtype=positions.dtype)
    tol_t = torch.as_tensor(tol, device=positions.device, dtype=positions.dtype)
    t_min_axes = []
    t_max_axes = []
    for axis in range(3):
        value = p0[..., axis]
        step = delta[..., axis]
        lo = lower[:, axis]
        hi = upper[:, axis]
        parallel = torch.abs(step) <= tol_t
        inside = (value >= lo) & (value <= hi)
        safe_step = torch.where(parallel, torch.ones_like(step), step)
        t0 = (lo - value) / safe_step
        t1 = (hi - value) / safe_step
        axis_min = torch.minimum(t0, t1)
        axis_max = torch.maximum(t0, t1)
        neg_inf = torch.full_like(axis_min, -float("inf"))
        pos_inf = torch.full_like(axis_max, float("inf"))
        axis_min = torch.where(parallel & inside, neg_inf, axis_min)
        axis_max = torch.where(parallel & inside, pos_inf, axis_max)
        axis_min = torch.where(parallel & ~inside, pos_inf, axis_min)
        axis_max = torch.where(parallel & ~inside, neg_inf, axis_max)
        t_min_axes.append(axis_min)
        t_max_axes.append(axis_max)
    t_enter = torch.maximum(torch.stack(t_min_axes, dim=-1).amax(dim=-1), torch.zeros_like(distance).unsqueeze(-1))
    t_exit = torch.minimum(torch.stack(t_max_axes, dim=-1).amin(dim=-1), torch.ones_like(distance).unsqueeze(-1))
    length_m = torch.where(t_exit > t_enter, (t_exit - t_enter) * distance.unsqueeze(-1), torch.zeros_like(t_exit))
    return 100.0 * length_m


def obstacle_path_lengths_between_points_cm_torch(
    source_pos: "torch.Tensor",
    target_pos: "torch.Tensor",
    obstacle_boxes_m: "torch.Tensor",
    tol: float = 1e-9,
) -> "torch.Tensor":
    """Return path lengths through axis-aligned boxes for source-target segments."""
    lengths_by_box = obstacle_path_lengths_between_points_by_box_cm_torch(
        source_pos=source_pos,
        target_pos=target_pos,
        obstacle_boxes_m=obstacle_boxes_m,
        tol=tol,
    )
    return torch.sum(lengths_by_box, dim=-1)


def obstacle_path_lengths_between_points_by_box_cm_torch(
    source_pos: "torch.Tensor",
    target_pos: "torch.Tensor",
    obstacle_boxes_m: "torch.Tensor",
    tol: float = 1e-9,
) -> "torch.Tensor":
    """Return per-box path lengths through axis-aligned boxes for segments."""
    if torch is None:
        raise RuntimeError("torch is not available")
    if obstacle_boxes_m.numel() == 0:
        return torch.zeros(
            (*source_pos.shape[:-1], 0),
            device=source_pos.device,
            dtype=source_pos.dtype,
        )
    if obstacle_boxes_m.ndim != 2 or obstacle_boxes_m.shape[1] != 6:
        raise ValueError("obstacle_boxes_m must be shaped (N, 6).")
    p0 = source_pos.unsqueeze(-2)
    delta = (target_pos - source_pos).unsqueeze(-2)
    distance = torch.linalg.norm(target_pos - source_pos, dim=-1)
    lower = obstacle_boxes_m[:, :3].to(device=source_pos.device, dtype=source_pos.dtype)
    upper = obstacle_boxes_m[:, 3:].to(device=source_pos.device, dtype=source_pos.dtype)
    tol_t = torch.as_tensor(tol, device=source_pos.device, dtype=source_pos.dtype)
    t_min_axes = []
    t_max_axes = []
    for axis in range(3):
        value = p0[..., axis]
        step = delta[..., axis]
        lo = lower[:, axis]
        hi = upper[:, axis]
        parallel = torch.abs(step) <= tol_t
        inside = (value >= lo) & (value <= hi)
        safe_step = torch.where(parallel, torch.ones_like(step), step)
        t0 = (lo - value) / safe_step
        t1 = (hi - value) / safe_step
        axis_min = torch.minimum(t0, t1)
        axis_max = torch.maximum(t0, t1)
        neg_inf = torch.full_like(axis_min, -float("inf"))
        pos_inf = torch.full_like(axis_max, float("inf"))
        axis_min = torch.where(parallel & inside, neg_inf, axis_min)
        axis_max = torch.where(parallel & inside, pos_inf, axis_max)
        axis_min = torch.where(parallel & ~inside, pos_inf, axis_min)
        axis_max = torch.where(parallel & ~inside, neg_inf, axis_max)
        t_min_axes.append(axis_min)
        t_max_axes.append(axis_max)
    t_enter = torch.maximum(
        torch.stack(t_min_axes, dim=-1).amax(dim=-1),
        torch.zeros_like(distance).unsqueeze(-1),
    )
    t_exit = torch.minimum(
        torch.stack(t_max_axes, dim=-1).amin(dim=-1),
        torch.ones_like(distance).unsqueeze(-1),
    )
    length_m = torch.where(
        t_exit > t_enter,
        (t_exit - t_enter) * distance.unsqueeze(-1),
        torch.zeros_like(t_exit),
    )
    return 100.0 * length_m


def _torch_available() -> bool:
    """Return True if torch is available and CUDA is usable."""
    return bool(_TORCH_AVAILABLE and torch is not None and torch.cuda.is_available())


def _torch_installed() -> bool:
    """Return True if torch is available (CUDA not required)."""
    return bool(_TORCH_AVAILABLE and torch is not None)


def _torch_device_available(device: str | None = None) -> bool:
    """Return True when torch can run on the requested device."""
    if not _torch_installed():
        return False
    device_name = "cuda" if device is None else str(device)
    if device_name.startswith("cuda"):
        return bool(torch is not None and torch.cuda.is_available())
    return True


def _resolve_device(device: str | None) -> "torch.device":
    """Resolve a torch device string with CUDA fallback."""
    if torch is None:
        raise RuntimeError("torch is not available")
    if device is None:
        device = "cuda"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available.")
    return torch.device(device)


def _resolve_dtype(dtype: str) -> "torch.dtype":
    """Map a dtype string to a torch dtype."""
    if torch is None:
        raise RuntimeError("torch is not available")
    if dtype == "float32":
        return torch.float32
    if dtype == "float64":
        return torch.float64
    raise ValueError(f"Unsupported torch dtype: {dtype}")


@dataclass
class ContinuousKernel:
    """
    Continuous-coordinate kernel for Poisson expected counts (Sec. 3.3).

    Shield attenuation is applied using an octant-based model with exponential
    attenuation exp(-mu * L) for Fe/Pb shells.
    """

    mu_by_isotope: Dict[str, object] | None = None
    shield_params: ShieldParams = field(default_factory=ShieldParams)
    octant_shield: OctantShield = OctantShield()
    orientations: NDArray[np.float64] = field(default_factory=generate_octant_orientations)
    use_gpu: bool = True
    gpu_device: str = "cuda"
    gpu_dtype: str = "float32"
    obstacle_grid: ObstacleGrid | None = None
    obstacle_height_m: float = 2.0
    obstacle_mu_by_isotope: Dict[str, float] | None = None
    obstacle_buildup_coeff: float = 0.0
    detector_radius_m: float = 0.0
    detector_aperture_radius_m: float | None = None
    detector_aperture_samples: int = 1
    detector_aperture_sampling: str = "solid_angle_cone"
    source_extent_radius_m: float = 0.0
    source_extent_samples: int = 1
    line_mu_by_isotope: Dict[str, object] | None = None
    transport_response_model: Dict[str, object] | None = None
    _obstacle_boxes_cache: NDArray[np.float64] | None = field(default=None, init=False, repr=False)
    _torch_octant_rotation_cache: dict[tuple[str, str, tuple[float, float, float]], object] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate obstacle attenuation settings."""
        self.obstacle_height_m = float(self.obstacle_height_m)
        if self.obstacle_height_m < 0.0:
            raise ValueError("obstacle_height_m must be non-negative.")
        self.obstacle_buildup_coeff = max(float(self.obstacle_buildup_coeff), 0.0)
        self.shield_params = ShieldParams(
            mu_pb=float(self.shield_params.mu_pb),
            mu_fe=float(self.shield_params.mu_fe),
            thickness_pb_cm=float(self.shield_params.thickness_pb_cm),
            thickness_fe_cm=float(self.shield_params.thickness_fe_cm),
            inner_radius_fe_cm=float(self.shield_params.inner_radius_fe_cm),
            inner_radius_pb_cm=float(self.shield_params.inner_radius_pb_cm),
            buildup_fe_coeff=max(float(self.shield_params.buildup_fe_coeff), 0.0),
            buildup_pb_coeff=max(float(self.shield_params.buildup_pb_coeff), 0.0),
            shield_geometry_model=self.shield_params.shield_geometry_model,
            use_angle_attenuation=bool(self.shield_params.use_angle_attenuation),
        )
        self.detector_radius_m = max(float(self.detector_radius_m), 0.0)
        if self.detector_aperture_radius_m is None:
            self.detector_aperture_radius_m = self.detector_radius_m
        self.detector_aperture_radius_m = max(
            float(self.detector_aperture_radius_m),
            0.0,
        )
        self.detector_aperture_samples = max(int(self.detector_aperture_samples), 1)
        self.detector_aperture_sampling = normalize_detector_aperture_sampling(
            self.detector_aperture_sampling
        )
        self.source_extent_radius_m = max(float(self.source_extent_radius_m), 0.0)
        self.source_extent_samples = max(int(self.source_extent_samples), 1)

    def _rotated_octant_rotation_torch(
        self,
        shield_normal: NDArray[np.float64],
        *,
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> "torch.Tensor":
        """Return a cached torch rotation for a shield octant normal."""
        if torch is None:
            raise RuntimeError("torch is not available")
        normal = np.asarray(shield_normal, dtype=float).reshape(3)
        key = (
            str(device),
            str(dtype),
            tuple(float(np.round(value, 12)) for value in normal),
        )
        cached = self._torch_octant_rotation_cache.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        rotation_np = rotation_matrix_between_vectors(
            LOCAL_POSITIVE_OCTANT_CENTER,
            normal,
        )
        rotation = torch.as_tensor(rotation_np, device=device, dtype=dtype)
        self._torch_octant_rotation_cache[key] = rotation
        return rotation

    def _adaptive_torch_chunk_size(
        self,
        requested: int,
        *,
        all_orientation_pairs: bool = False,
    ) -> int:
        """
        Return a GPU chunk size that preserves math while bounding tensors.

        Obstacle attenuation with finite detector aperture expands each source
        into ``source_count * aperture_samples * obstacle_box_count`` segment-box
        intersections. Random Manchester-style scenes can contain hundreds of
        transport components, so the old fixed 8192-source chunk could allocate
        tens of GB. This only changes batching, not the kernel being evaluated.
        """
        chunk = max(1, int(requested))
        aperture_sample_count = (
            max(int(self.detector_aperture_samples), 1)
            if float(self.detector_aperture_radius_m or 0.0) > 0.0
            else 1
        )
        source_sample_count = (
            max(int(self.source_extent_samples), 1)
            if float(self.source_extent_radius_m or 0.0) > 0.0
            else 1
        )
        sample_count = aperture_sample_count * source_sample_count
        obstacle_count = int(self.obstacle_boxes_m().shape[0])
        num_pairs = int(len(self.orientations)) ** 2
        line_count = self._max_line_count()
        denom = max(1, sample_count * line_count)
        if obstacle_count > 0:
            denom = max(denom, sample_count * obstacle_count * line_count)
        if all_orientation_pairs:
            denom = max(denom, sample_count * num_pairs * line_count)
        element_budget = (
            4_000_000
            if str(self.gpu_dtype).lower() == "float64"
            else 8_000_000
        )
        safe_chunk = max(1, int(element_budget // denom))
        return max(1, min(chunk, safe_chunk))

    def _mu_values(self, isotope: str) -> tuple[float, float]:
        """Return (mu_fe, mu_pb) for the given isotope with fallbacks."""
        return resolve_mu_values(
            self.mu_by_isotope,
            isotope,
            default_fe=self.shield_params.mu_fe,
            default_pb=self.shield_params.mu_pb,
        )

    def _line_mu_values(self, isotope: str) -> tuple[tuple[float, float, float], ...]:
        """
        Return line-resolved ``(weight, mu_fe, mu_pb)`` entries for an isotope.

        If no line-resolved table is configured, callers fall back to the
        existing isotope-effective coefficients.
        """
        table = self.line_mu_by_isotope
        if not isinstance(table, dict):
            return ()
        entry = table.get(isotope)
        if entry is None:
            normalized = {_normalize_isotope_key(key): value for key, value in table.items()}
            entry = normalized.get(_normalize_isotope_key(isotope))
        if entry is None:
            return ()
        rows: list[tuple[float, float, float]] = []
        for item in entry if isinstance(entry, (list, tuple)) else ():
            if isinstance(item, dict):
                weight = float(item.get("weight", 0.0))
                mu_fe = float(item.get("fe", item.get("mu_fe", self.shield_params.mu_fe)))
                mu_pb = float(item.get("pb", item.get("mu_pb", self.shield_params.mu_pb)))
            elif isinstance(item, (list, tuple, np.ndarray)) and len(item) >= 3:
                weight = float(item[0])
                mu_fe = float(item[1])
                mu_pb = float(item[2])
            else:
                continue
            if weight > 0.0 and np.isfinite(weight) and np.isfinite(mu_fe) and np.isfinite(mu_pb):
                rows.append((weight, mu_fe, mu_pb))
        total_weight = sum(weight for weight, _, _ in rows)
        if total_weight <= 0.0:
            return ()
        return tuple((weight / total_weight, mu_fe, mu_pb) for weight, mu_fe, mu_pb in rows)

    def _max_line_count(self) -> int:
        """Return the maximum configured line count used by attenuation batching."""
        table = self.line_mu_by_isotope
        if not isinstance(table, dict):
            return 1
        count = 1
        for isotope in table:
            count = max(count, len(self._line_mu_values(str(isotope))))
        return max(1, int(count))

    def _transport_response_payload(self, isotope: str) -> dict[str, object]:
        """Return the configured transport-response payload for one isotope."""
        return transport_response_payload_for_isotope(
            self.transport_response_model,
            isotope,
        )

    def _transport_response_pair_log_scale(
        self,
        isotope: str,
        fe_index: int,
        pb_index: int,
    ) -> float:
        """Return the isotope/pair log-scale transport response term."""
        payload = self._transport_response_payload(isotope)
        pair_id = int(fe_index) * int(len(self.orientations)) + int(pb_index)
        return transport_response_pair_log_scale_from_payload(payload, pair_id)

    def _transport_response_coefficients(
        self,
        isotope: str,
    ) -> tuple[dict[str, float], float, float]:
        """Return optical-depth response coefficients and log-scale bounds."""
        payload = self._transport_response_payload(isotope)
        return transport_response_coefficients_from_payload(payload)

    def _transport_response_feature_caps(
        self,
        isotope: str,
    ) -> tuple[
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
    ]:
        """Return optional optical-depth feature caps for one isotope."""
        payload = self._transport_response_payload(isotope)
        return transport_response_feature_caps_from_payload(payload)

    @staticmethod
    def _capped_transport_feature(value: float, cap: float | None) -> float:
        """Return a nonnegative optical-depth feature with an optional cap."""
        return capped_transport_response_feature(value, cap)

    def _transport_response_factor(
        self,
        isotope: str,
        fe_index: int,
        pb_index: int,
        shield_tau_feature: float,
        obstacle_tau_feature: float,
        fe_tau_feature: float | None = None,
        pb_tau_feature: float | None = None,
        distance_feature: float | None = None,
        distance_shield_feature: float | None = None,
    ) -> float:
        """Return source-local transport response factor for CPU kernel math."""
        payload = self._transport_response_payload(isotope)
        pair_id = int(fe_index) * int(len(self.orientations)) + int(pb_index)
        return transport_response_factor_from_payload(
            payload,
            pair_id=pair_id,
            shield_tau_feature=shield_tau_feature,
            obstacle_tau_feature=obstacle_tau_feature,
            fe_tau_feature=fe_tau_feature,
            pb_tau_feature=pb_tau_feature,
            distance_feature=distance_feature,
            distance_shield_feature=distance_shield_feature,
        )

    def _response_adjusted_attenuation(
        self,
        isotope: str,
        attenuation: float,
        response_factor: float,
    ) -> float:
        """
        Return attenuation after applying an optional transport-response model.

        Without a calibrated transport-response payload, attenuation remains a
        transmission probability and is capped at unity.  A configured
        transport-response model is fitted to ``transport truth / PF target``
        and may represent detected interacted-primary contributions, so it must
        be allowed to exceed the unattenuated transmission cap to match the
        validation residual model that exported it.
        """
        base_attenuation = float(min(1.0, max(0.0, float(attenuation))))
        if self._transport_response_payload(isotope):
            return max(0.0, base_attenuation * float(response_factor))
        return base_attenuation

    def _response_adjusted_attenuation_torch(
        self,
        isotope: str,
        attenuation: "torch.Tensor",
        response_factor: "torch.Tensor | float" = 1.0,
    ) -> "torch.Tensor":
        """Return torch attenuation with the same response-model cap semantics."""
        if torch is None:
            raise RuntimeError("torch is not available")
        base_attenuation = torch.clamp(attenuation, min=0.0, max=1.0)
        if self._transport_response_payload(isotope):
            return torch.clamp(base_attenuation * response_factor, min=0.0)
        return base_attenuation

    def _transport_response_factor_torch(
        self,
        isotope: str,
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
        shield_tau_feature: "torch.Tensor",
        obstacle_tau_feature: "torch.Tensor",
        fe_tau_feature: "torch.Tensor | None" = None,
        pb_tau_feature: "torch.Tensor | None" = None,
        distance_feature: "torch.Tensor | None" = None,
        distance_shield_feature: "torch.Tensor | None" = None,
        *,
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> "torch.Tensor":
        """Return source-local transport response factors for torch kernel math."""
        if torch is None:
            raise RuntimeError("torch is not available")
        payload = self._transport_response_payload(isotope)
        if not payload:
            return torch.ones_like(shield_tau_feature, device=device, dtype=dtype)
        fe_arr = np.asarray(fe_indices, dtype=int).reshape(-1)
        pb_arr = np.asarray(pb_indices, dtype=int).reshape(-1)
        pair_logs = [
            self._transport_response_pair_log_scale(isotope, int(fe), int(pb))
            for fe, pb in zip(fe_arr, pb_arr)
        ]
        pair_log_t = torch.as_tensor(pair_logs, device=device, dtype=dtype)
        while pair_log_t.ndim < shield_tau_feature.ndim:
            pair_log_t = pair_log_t.unsqueeze(-1)
        coeffs, min_log, max_log = self._transport_response_coefficients(isotope)
        (
            shield_cap,
            obstacle_cap,
            fe_cap,
            pb_cap,
            distance_shield_cap,
            distance_fe_cap,
            distance_pb_cap,
            distance_obstacle_cap,
        ) = (
            self._transport_response_feature_caps(isotope)
        )
        shield_tau_raw = torch.clamp(shield_tau_feature, min=0.0)
        shield_tau = shield_tau_raw
        if shield_cap is not None:
            shield_tau = torch.clamp(shield_tau, max=float(shield_cap))
        obstacle_tau_raw = torch.clamp(obstacle_tau_feature, min=0.0)
        obstacle_tau = obstacle_tau_raw
        if obstacle_cap is not None:
            obstacle_tau = torch.clamp(obstacle_tau, max=float(obstacle_cap))
        fe_tau_raw = (
            torch.zeros_like(shield_tau)
            if fe_tau_feature is None
            else torch.clamp(fe_tau_feature, min=0.0)
        )
        fe_tau = fe_tau_raw
        if fe_cap is not None:
            fe_tau = torch.clamp(fe_tau, max=float(fe_cap))
        pb_tau_raw = (
            torch.zeros_like(shield_tau)
            if pb_tau_feature is None
            else torch.clamp(pb_tau_feature, min=0.0)
        )
        pb_tau = pb_tau_raw
        if pb_cap is not None:
            pb_tau = torch.clamp(pb_tau, max=float(pb_cap))
        distance = (
            torch.zeros_like(shield_tau)
            if distance_feature is None
            else torch.clamp(distance_feature, min=0.0)
        )
        while distance.ndim < shield_tau.ndim:
            distance = distance.unsqueeze(-1)
        distance_shield = (
            distance * shield_tau_raw
            if distance_shield_feature is None
            else torch.clamp(distance_shield_feature, min=0.0)
        )
        while distance_shield.ndim < shield_tau.ndim:
            distance_shield = distance_shield.unsqueeze(-1)
        if distance_shield_cap is not None:
            distance_shield = torch.clamp(
                distance_shield,
                max=float(distance_shield_cap),
            )
        distance_fe = distance * fe_tau_raw
        if distance_fe_cap is not None:
            distance_fe = torch.clamp(distance_fe, max=float(distance_fe_cap))
        distance_pb = distance * pb_tau_raw
        if distance_pb_cap is not None:
            distance_pb = torch.clamp(distance_pb, max=float(distance_pb_cap))
        distance_obstacle = distance * obstacle_tau_raw
        if distance_obstacle_cap is not None:
            distance_obstacle = torch.clamp(
                distance_obstacle,
                max=float(distance_obstacle_cap),
            )
        log_scale = pair_log_t
        log_scale = log_scale + float(coeffs.get("shield", 0.0)) * shield_tau
        log_scale = log_scale + float(coeffs.get("obstacle", 0.0)) * obstacle_tau
        log_scale = log_scale + float(coeffs.get("shield_squared", 0.0)) * shield_tau * shield_tau
        log_scale = log_scale + float(coeffs.get("obstacle_squared", 0.0)) * obstacle_tau * obstacle_tau
        log_scale = log_scale + float(coeffs.get("shield_obstacle", 0.0)) * shield_tau * obstacle_tau
        log_scale = log_scale + float(coeffs.get("fe", 0.0)) * fe_tau
        log_scale = log_scale + float(coeffs.get("pb", 0.0)) * pb_tau
        log_scale = log_scale + float(coeffs.get("fe_squared", 0.0)) * fe_tau * fe_tau
        log_scale = log_scale + float(coeffs.get("pb_squared", 0.0)) * pb_tau * pb_tau
        log_scale = log_scale + float(coeffs.get("fe_pb", 0.0)) * fe_tau * pb_tau
        log_scale = log_scale + float(coeffs.get("fe_obstacle", 0.0)) * fe_tau * obstacle_tau
        log_scale = log_scale + float(coeffs.get("pb_obstacle", 0.0)) * pb_tau * obstacle_tau
        log_scale = log_scale + float(coeffs.get("distance", 0.0)) * distance
        log_scale = (
            log_scale
            + float(coeffs.get("distance_shield", 0.0)) * distance_shield
        )
        log_scale = log_scale + float(coeffs.get("distance_fe", 0.0)) * distance_fe
        log_scale = log_scale + float(coeffs.get("distance_pb", 0.0)) * distance_pb
        log_scale = (
            log_scale
            + float(coeffs.get("distance_obstacle", 0.0)) * distance_obstacle
        )
        log_scale = torch.clamp(log_scale, min=float(min_log), max=float(max_log))
        return torch.exp(log_scale)

    def obstacle_boxes_m(self) -> NDArray[np.float64]:
        """Return cached obstacle boxes in meters as (x0, y0, z0, x1, y1, z1)."""
        if self.obstacle_grid is None:
            return np.zeros((0, 6), dtype=float)
        if self._obstacle_boxes_cache is None:
            if getattr(self.obstacle_grid, "has_transport_model", False):
                boxes = self.obstacle_grid.transport_boxes()
            else:
                boxes = self.obstacle_grid.blocked_boxes(
                    z_min=0.0,
                    z_max=float(self.obstacle_height_m),
                )
            if boxes:
                self._obstacle_boxes_cache = np.asarray(boxes, dtype=float)
            else:
                self._obstacle_boxes_cache = np.zeros((0, 6), dtype=float)
        return self._obstacle_boxes_cache.copy()

    def obstacle_mu_cm_inv(self, isotope: str) -> float:
        """Return concrete obstacle attenuation coefficient in 1/cm for an isotope."""
        if self.obstacle_grid is None:
            return 0.0
        return resolve_obstacle_mu_cm_inv(isotope, self.obstacle_mu_by_isotope)

    def obstacle_mu_values_cm_inv(self, isotope: str) -> NDArray[np.float64]:
        """Return per-obstacle-box attenuation coefficients in 1/cm."""
        boxes = self.obstacle_boxes_m()
        if boxes.size == 0:
            return np.zeros(0, dtype=float)
        if self.obstacle_grid is not None:
            values = self.obstacle_grid.transport_mu_values(isotope)
            if values is not None:
                return np.asarray(values, dtype=float)
        return np.full(boxes.shape[0], self.obstacle_mu_cm_inv(isotope), dtype=float)

    def obstacle_line_mu_values_cm_inv(self, isotope: str) -> NDArray[np.float64]:
        """Return per-line, per-box obstacle attenuation coefficients."""
        boxes = self.obstacle_boxes_m()
        if boxes.size == 0 or self.obstacle_grid is None:
            return np.zeros((0, 0), dtype=float)
        values = self.obstacle_grid.transport_line_mu_values(isotope)
        if values is None:
            return np.zeros((0, boxes.shape[0]), dtype=float)
        array = np.asarray(values, dtype=float)
        if array.ndim != 2 or array.shape[1] != boxes.shape[0]:
            return np.zeros((0, boxes.shape[0]), dtype=float)
        return array

    def obstacle_path_length_cm(
        self,
        source_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
    ) -> float:
        """Return total source-detector path length inside configured obstacles in centimeters."""
        return obstacle_path_length_cm(
            source_pos=source_pos,
            detector_pos=detector_pos,
            obstacle_boxes_m=self.obstacle_boxes_m(),
        )

    def obstacle_path_lengths_by_box_cm(
        self,
        source_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return per-obstacle-box path lengths in centimeters."""
        return obstacle_path_lengths_by_box_cm(
            source_pos=source_pos,
            detector_pos=detector_pos,
            obstacle_boxes_m=self.obstacle_boxes_m(),
        )

    def obstacle_optical_depth_pair(
        self,
        isotope: str,
        source_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
    ) -> float:
        """Return obstacle-only optical depth for one source-detector ray."""
        if self.obstacle_grid is None:
            return 0.0
        return obstacle_optical_depth(
            source_pos=source_pos,
            detector_pos=detector_pos,
            obstacle_boxes_m=self.obstacle_boxes_m(),
            obstacle_mu_cm_inv_by_box=self.obstacle_mu_values_cm_inv(isotope),
        )

    def obstacle_log_attenuation_pair(
        self,
        isotope: str,
        source_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
    ) -> float:
        """Return log obstacle transmission, log(A_env), for one ray."""
        return -float(
            self.obstacle_optical_depth_pair(
                isotope=isotope,
                source_pos=source_pos,
                detector_pos=detector_pos,
            )
        )

    def obstacle_log_attenuation_matrix(
        self,
        isotope: str,
        sources_xyz: NDArray[np.float64],
        detector_poses_xyz: NDArray[np.float64],
        *,
        element_budget: int = 4_000_000,
    ) -> NDArray[np.float64]:
        """Return obstacle-only log transmission for detector/source batches."""
        return obstacle_log_attenuation_matrix(
            sources_xyz=sources_xyz,
            detector_poses_xyz=detector_poses_xyz,
            obstacle_boxes_m=self.obstacle_boxes_m(),
            obstacle_mu_cm_inv_by_box=self.obstacle_mu_values_cm_inv(isotope),
            element_budget=element_budget,
        )

    def obstacle_attenuation_factor_pair(
        self,
        isotope: str,
        source_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
    ) -> float:
        """Return obstacle-only Beer-Lambert attenuation for one ray."""
        tau = self.obstacle_optical_depth_pair(
            isotope=isotope,
            source_pos=source_pos,
            detector_pos=detector_pos,
        )
        if tau <= 0.0:
            return 1.0
        return float(np.exp(-tau))

    def obstacle_area_averaged_attenuation_pair(
        self,
        isotope: str,
        source_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
    ) -> float:
        """Return obstacle attenuation averaged over source and aperture rays."""
        if self.obstacle_grid is None:
            return 1.0
        sampled_sources, targets = self._ray_sample_points(source_pos, detector_pos)
        transmissions = [
            float(
                np.exp(
                    -self.obstacle_optical_depth_pair(
                        isotope=isotope,
                        source_pos=sampled_source,
                        detector_pos=target,
                    )
                )
            )
            for sampled_source, target in zip(sampled_sources, targets)
        ]
        if not transmissions:
            return 1.0
        return float(np.mean(np.asarray(transmissions, dtype=float)))

    def obstacle_area_averaged_optical_depth_pair(
        self,
        isotope: str,
        source_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
    ) -> float:
        """Return the equivalent tau from area-averaged obstacle transmission."""
        attenuation = self.obstacle_area_averaged_attenuation_pair(
            isotope=isotope,
            source_pos=source_pos,
            detector_pos=detector_pos,
        )
        return float(-np.log(max(float(attenuation), 1.0e-300)))

    def _obstacle_attenuation_factor(
        self,
        isotope: str,
        source_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
    ) -> float:
        """Return Beer-Lambert attenuation through known obstacle components."""
        return self.obstacle_attenuation_factor_pair(
            isotope=isotope,
            source_pos=source_pos,
            detector_pos=detector_pos,
        )

    def obstacle_gpu_kwargs(self, isotope: str) -> dict[str, object]:
        """Return optional GPU kwargs for obstacle attenuation."""
        boxes = self.obstacle_boxes_m()
        line_entries = self._line_mu_values(isotope)
        kwargs: dict[str, object] = {}
        if line_entries:
            kwargs.update(
                {
                    "line_weights": np.asarray(
                        [entry[0] for entry in line_entries],
                        dtype=float,
                    ),
                    "line_mu_fe": np.asarray(
                        [entry[1] for entry in line_entries],
                        dtype=float,
                    ),
                    "line_mu_pb": np.asarray(
                        [entry[2] for entry in line_entries],
                        dtype=float,
                    ),
                }
            )
            obstacle_line_mu = self.obstacle_line_mu_values_cm_inv(isotope)
            if obstacle_line_mu.shape == (len(line_entries), boxes.shape[0]):
                kwargs["obstacle_line_mu_cm_inv_by_box"] = obstacle_line_mu
        if boxes.size == 0:
            return kwargs
        kwargs.update(
            {
                "obstacle_boxes_m": boxes,
                "obstacle_mu_cm_inv": 0.0,
                "obstacle_mu_cm_inv_by_box": self.obstacle_mu_values_cm_inv(isotope),
                "obstacle_buildup_coeff": self.obstacle_buildup_coeff,
            }
        )
        return kwargs

    def _line_obstacle_tau_values(
        self,
        isotope: str,
        path_by_box_cm: NDArray[np.float64],
        *,
        line_count: int,
    ) -> tuple[float, ...] | None:
        """Return line-resolved obstacle optical depths for one ray."""
        mu_values = self.obstacle_line_mu_values_cm_inv(isotope)
        if mu_values.shape != (int(line_count), path_by_box_cm.shape[0]):
            return None
        return tuple(float(np.sum(row * path_by_box_cm)) for row in mu_values)

    def _buildup_factor(
        self,
        tau_fe: float,
        tau_pb: float,
        tau_obstacle: float,
    ) -> float:
        """Return a bounded broad-beam build-up factor from optical depths."""
        factor = 1.0
        factor += self.shield_params.buildup_fe_coeff * (1.0 - float(np.exp(-max(tau_fe, 0.0))))
        factor += self.shield_params.buildup_pb_coeff * (1.0 - float(np.exp(-max(tau_pb, 0.0))))
        factor += self.obstacle_buildup_coeff * (1.0 - float(np.exp(-max(tau_obstacle, 0.0))))
        return max(1.0, float(factor))

    def _buildup_factor_torch(
        self,
        tau_fe: "torch.Tensor",
        tau_pb: "torch.Tensor",
        tau_obstacle: "torch.Tensor",
    ) -> "torch.Tensor":
        """Return a broad-beam build-up factor for torch optical depths."""
        if torch is None:
            raise RuntimeError("torch is not available")
        factor = torch.ones_like(tau_fe)
        factor = factor + self.shield_params.buildup_fe_coeff * (1.0 - torch.exp(-torch.clamp(tau_fe, min=0.0)))
        factor = factor + self.shield_params.buildup_pb_coeff * (1.0 - torch.exp(-torch.clamp(tau_pb, min=0.0)))
        factor = factor + self.obstacle_buildup_coeff * (
            1.0 - torch.exp(-torch.clamp(tau_obstacle, min=0.0))
        )
        return torch.clamp(factor, min=1.0)

    def _obstacle_tau_from_lengths_torch(
        self,
        isotope: str,
        path_by_box_cm: "torch.Tensor",
        *,
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> "torch.Tensor":
        """Return material optical depth from per-box torch path lengths."""
        if torch is None:
            raise RuntimeError("torch is not available")
        if path_by_box_cm.shape[-1] == 0:
            return torch.zeros(path_by_box_cm.shape[:-1], device=device, dtype=dtype)
        mu_values = torch.as_tensor(
            self.obstacle_mu_values_cm_inv(isotope),
            device=device,
            dtype=dtype,
        )
        return torch.sum(path_by_box_cm * mu_values, dim=-1)

    def _obstacle_line_tau_from_lengths_torch(
        self,
        isotope: str,
        path_by_box_cm: "torch.Tensor",
        *,
        line_count: int,
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> "torch.Tensor | None":
        """Return line-resolved obstacle optical depths from path lengths."""
        if torch is None:
            raise RuntimeError("torch is not available")
        if path_by_box_cm.shape[-1] == 0:
            return torch.zeros(
                (*path_by_box_cm.shape[:-1], int(line_count)),
                device=device,
                dtype=dtype,
            )
        mu_values = self.obstacle_line_mu_values_cm_inv(isotope)
        if mu_values.shape != (int(line_count), int(path_by_box_cm.shape[-1])):
            return None
        mu_t = torch.as_tensor(mu_values, device=device, dtype=dtype)
        return torch.sum(path_by_box_cm.unsqueeze(-2) * mu_t, dim=-1)

    def _gpu_enabled(self) -> bool:
        """Return True if GPU computation is enabled and available."""
        if not self.use_gpu:
            raise RuntimeError("GPU-only mode: enable use_gpu for ContinuousKernel.")
        if not _torch_device_available(self.gpu_device):
            raise RuntimeError("GPU-only mode requires torch on the requested device.")
        return True

    def _blocked_mask_torch(
        self,
        dir_unit: "torch.Tensor",
        octant_index: int,
        tol: float,
    ) -> "torch.Tensor":
        """Return a boolean mask for rays blocked by the selected octant (torch)."""
        (theta_low, theta_high), (phi_low, phi_high) = self.octant_shield.theta_phi_ranges[octant_index]
        theta = torch.acos(torch.clamp(dir_unit[:, 2], -1.0, 1.0))
        phi = torch.remainder(torch.atan2(dir_unit[:, 1], dir_unit[:, 0]), 2.0 * np.pi)
        tol_t = torch.as_tensor(tol, device=dir_unit.device, dtype=dir_unit.dtype)
        return (
            (theta + tol_t >= theta_low)
            & (theta - tol_t < theta_high)
            & (phi + tol_t >= phi_low)
            & (phi - tol_t < phi_high)
        )

    def _rotated_octant_blocked_mask_torch(
        self,
        detector_to_source_unit: "torch.Tensor",
        octant_index: int,
        tol: float,
    ) -> "torch.Tensor":
        """Return a mask for the rotated local +X/+Y/+Z shield octant."""
        if torch is None:
            raise RuntimeError("torch is not available")
        physical_normal = -np.asarray(self.orientations[octant_index], dtype=float)
        rotation_np = rotation_matrix_between_vectors(
            LOCAL_POSITIVE_OCTANT_CENTER,
            physical_normal,
        )
        rotation = torch.as_tensor(
            rotation_np,
            device=detector_to_source_unit.device,
            dtype=detector_to_source_unit.dtype,
        )
        local_direction = detector_to_source_unit @ rotation
        return torch.all(local_direction >= -float(tol), dim=-1)

    def _shield_path_length_cm(
        self,
        direction_m: NDArray[np.float64],
        normal: NDArray[np.float64],
        thickness_cm: float,
        inner_radius_cm: float,
        blocked: bool,
    ) -> float:
        """Return Pb/Fe path length through the configured shield geometry."""
        if (
            self.shield_params.shield_geometry_model == SHIELD_GEOMETRY_SPHERICAL_OCTANT
            and not self.shield_params.use_angle_attenuation
        ):
            return spherical_shell_path_length_cm(
                direction_m=direction_m,
                inner_radius_cm=inner_radius_cm,
                outer_radius_cm=inner_radius_cm + thickness_cm,
                blocked=blocked,
            )
        return path_length_cm(
            direction_m,
            normal,
            thickness_cm,
            blocked=blocked,
            use_angle_attenuation=self.shield_params.use_angle_attenuation,
        )

    def _shield_segment_path_length_cm(
        self,
        source_pos: NDArray[np.float64],
        target_pos: NDArray[np.float64],
        detector_center: NDArray[np.float64],
        incoming_normal: NDArray[np.float64],
        physical_normal: NDArray[np.float64],
        thickness_cm: float,
        inner_radius_cm: float,
        blocked: bool,
    ) -> float:
        """Return shield path length for a source-to-detector-aperture segment."""
        if (
            self.shield_params.shield_geometry_model == SHIELD_GEOMETRY_SPHERICAL_OCTANT
            and not self.shield_params.use_angle_attenuation
        ):
            return segment_rotated_octant_shell_path_length_cm(
                source_pos=source_pos,
                target_pos=target_pos,
                center_pos=detector_center,
                shield_normal=physical_normal,
                inner_radius_cm=inner_radius_cm,
                outer_radius_cm=inner_radius_cm + thickness_cm,
            )
        return self._shield_path_length_cm(
            direction_m=np.asarray(target_pos, dtype=float) - np.asarray(source_pos, dtype=float),
            normal=incoming_normal,
            thickness_cm=thickness_cm,
            inner_radius_cm=inner_radius_cm,
            blocked=blocked,
        )

    def _detector_aperture_targets(
        self,
        source_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return deterministic source-to-detector aperture target points."""
        detector = np.asarray(detector_pos, dtype=float)
        source = np.asarray(source_pos, dtype=float)
        aperture_radius = float(self.detector_aperture_radius_m or 0.0)
        if aperture_radius <= 0.0 or self.detector_aperture_samples <= 1:
            return detector.reshape(1, 3)
        axis = detector - source
        distance = float(np.linalg.norm(axis))
        if distance <= 1.0e-12:
            return detector.reshape(1, 3)
        axis /= distance
        helper = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(axis, helper))) > 0.9:
            helper = np.array([0.0, 1.0, 0.0], dtype=float)
        basis_u = np.cross(axis, helper)
        basis_u_norm = float(np.linalg.norm(basis_u))
        if basis_u_norm <= 1.0e-12:
            return detector.reshape(1, 3)
        basis_u /= basis_u_norm
        basis_v = np.cross(axis, basis_u)
        if self.detector_aperture_sampling == "solid_angle_cone":
            return self._detector_aperture_targets_cone(
                source=source,
                detector=detector,
                axis=axis,
                basis_u=basis_u,
                basis_v=basis_v,
                distance=distance,
                aperture_radius=aperture_radius,
            )
        return self._detector_aperture_targets_disk(
            detector=detector,
            basis_u=basis_u,
            basis_v=basis_v,
            distance=distance,
            aperture_radius=aperture_radius,
        )

    @staticmethod
    def _ray_perpendicular_basis(
        source_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        """Return a stable basis perpendicular to the source-detector ray."""
        source = np.asarray(source_pos, dtype=float)
        detector = np.asarray(detector_pos, dtype=float)
        axis = detector - source
        distance = float(np.linalg.norm(axis))
        if distance <= 1.0e-12:
            return None
        axis /= distance
        helper = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(axis, helper))) > 0.9:
            helper = np.array([0.0, 1.0, 0.0], dtype=float)
        basis_u = np.cross(axis, helper)
        basis_u_norm = float(np.linalg.norm(basis_u))
        if basis_u_norm <= 1.0e-12:
            return None
        basis_u /= basis_u_norm
        basis_v = np.cross(axis, basis_u)
        return basis_u, basis_v

    def _source_extent_points(
        self,
        source_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return deterministic source-extent sample points for ray averaging."""
        source = np.asarray(source_pos, dtype=float)
        radius = float(self.source_extent_radius_m or 0.0)
        count = int(self.source_extent_samples)
        if radius <= 0.0 or count <= 1:
            return source.reshape(1, 3)
        basis = self._ray_perpendicular_basis(source, detector_pos)
        if basis is None:
            return source.reshape(1, 3)
        basis_u, basis_v = basis
        points = np.empty((count, 3), dtype=float)
        points[0] = source
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for index in range(1, count):
            fraction = (float(index) - 0.5) / float(count - 1)
            sample_radius = radius * float(np.sqrt(np.clip(fraction, 0.0, 1.0)))
            angle = golden_angle * float(index)
            offset = sample_radius * (
                float(np.cos(angle)) * basis_u + float(np.sin(angle)) * basis_v
            )
            points[index] = source + offset
        return points

    def _ray_sample_points(
        self,
        source_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return flattened source and detector target pairs for area averaging."""
        source_points = self._source_extent_points(source_pos, detector_pos)
        sources: list[NDArray[np.float64]] = []
        targets: list[NDArray[np.float64]] = []
        for source_point in source_points:
            target_points = self._detector_aperture_targets(source_point, detector_pos)
            for target_point in target_points:
                sources.append(np.asarray(source_point, dtype=float))
                targets.append(np.asarray(target_point, dtype=float))
        if not sources:
            return (
                np.asarray(source_pos, dtype=float).reshape(1, 3),
                np.asarray(detector_pos, dtype=float).reshape(1, 3),
            )
        return np.vstack(sources), np.vstack(targets)

    def _detector_aperture_targets_disk(
        self,
        *,
        detector: NDArray[np.float64],
        basis_u: NDArray[np.float64],
        basis_v: NDArray[np.float64],
        distance: float,
        aperture_radius: float,
    ) -> NDArray[np.float64]:
        """Return legacy deterministic points on the detector aperture disk."""
        radius = min(aperture_radius, 0.95 * distance)
        count = int(self.detector_aperture_samples)
        targets = np.empty((count, 3), dtype=float)
        targets[0] = detector
        if count == 1:
            return targets
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for index in range(1, count):
            fraction = (float(index) - 0.5) / float(count - 1)
            sample_radius = radius * float(np.sqrt(np.clip(fraction, 0.0, 1.0)))
            angle = golden_angle * float(index)
            offset = sample_radius * (
                float(np.cos(angle)) * basis_u + float(np.sin(angle)) * basis_v
            )
            targets[index] = detector + offset
        return targets

    def _detector_aperture_targets_cone(
        self,
        *,
        source: NDArray[np.float64],
        detector: NDArray[np.float64],
        axis: NDArray[np.float64],
        basis_u: NDArray[np.float64],
        basis_v: NDArray[np.float64],
        distance: float,
        aperture_radius: float,
    ) -> NDArray[np.float64]:
        """Return Geant4 detector-cone targets on the aperture sphere."""
        if distance <= aperture_radius:
            return detector.reshape(1, 3)
        count = int(self.detector_aperture_samples)
        targets = np.empty((count, 3), dtype=float)
        sin_theta_max = min(max(aperture_radius / max(distance, 1.0e-12), 0.0), 1.0)
        cos_theta_max = float(np.sqrt(max(1.0 - sin_theta_max * sin_theta_max, 0.0)))
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        radius_sq = aperture_radius * aperture_radius
        for index in range(count):
            fraction = (float(index) + 0.5) / float(count)
            cos_theta = 1.0 - fraction * (1.0 - cos_theta_max)
            sin_theta = float(np.sqrt(max(1.0 - cos_theta * cos_theta, 0.0)))
            angle = golden_angle * float(index)
            direction = (
                cos_theta * axis
                + sin_theta
                * (float(np.cos(angle)) * basis_u + float(np.sin(angle)) * basis_v)
            )
            radial_sq = (distance * sin_theta) ** 2
            chord = float(np.sqrt(max(radius_sq - radial_sq, 0.0)))
            path_length = distance * cos_theta - chord
            targets[index] = source + path_length * direction
        return targets

    def _attenuation_factor_for_target(
        self,
        isotope: str,
        source_pos: NDArray[np.float64],
        target_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
    ) -> float:
        """Return attenuation for one source-to-aperture ray."""
        mu_fe, mu_pb = self._mu_values(isotope=isotope)
        normal_fe = self.orientations[fe_index]
        normal_pb = self.orientations[pb_index]
        detector_to_source = np.asarray(source_pos, dtype=float) - np.asarray(target_pos, dtype=float)
        distance_feature = float(
            np.linalg.norm(
                np.asarray(source_pos, dtype=float) - np.asarray(detector_pos, dtype=float)
            )
        )
        blocked_fe = rotated_positive_octant_blocks_direction(
            detector_to_source,
            -normal_fe,
        )
        blocked_pb = rotated_positive_octant_blocks_direction(
            detector_to_source,
            -normal_pb,
        )
        l_fe = self._shield_segment_path_length_cm(
            source_pos=source_pos,
            target_pos=target_pos,
            detector_center=detector_pos,
            incoming_normal=normal_fe,
            physical_normal=-normal_fe,
            thickness_cm=self.shield_params.thickness_fe_cm,
            inner_radius_cm=self.shield_params.inner_radius_fe_cm,
            blocked=blocked_fe,
        )
        l_pb = self._shield_segment_path_length_cm(
            source_pos=source_pos,
            target_pos=target_pos,
            detector_center=detector_pos,
            incoming_normal=normal_pb,
            physical_normal=-normal_pb,
            thickness_cm=self.shield_params.thickness_pb_cm,
            inner_radius_cm=self.shield_params.inner_radius_pb_cm,
            blocked=blocked_pb,
        )
        obstacle_path_by_box = np.zeros(0, dtype=float)
        tau_obstacle = 0.0
        if self.obstacle_grid is None:
            tau_obstacle = 0.0
        else:
            obstacle_path_by_box = obstacle_path_lengths_by_box_cm(
                source_pos=source_pos,
                detector_pos=target_pos,
                obstacle_boxes_m=self.obstacle_boxes_m(),
            )
            tau_obstacle = float(
                np.sum(self.obstacle_mu_values_cm_inv(isotope) * obstacle_path_by_box)
            )
        line_entries = self._line_mu_values(isotope)
        if line_entries:
            line_obstacle_tau = self._line_obstacle_tau_values(
                isotope,
                obstacle_path_by_box,
                line_count=len(line_entries),
            )
            attenuation = 0.0
            shield_tau_feature = 0.0
            fe_tau_feature = 0.0
            pb_tau_feature = 0.0
            obstacle_tau_feature = 0.0
            for line_index, (weight, line_mu_fe, line_mu_pb) in enumerate(line_entries):
                tau_fe = float(line_mu_fe * l_fe)
                tau_pb = float(line_mu_pb * l_pb)
                tau_obs_line = (
                    tau_obstacle
                    if line_obstacle_tau is None
                    else float(line_obstacle_tau[line_index])
                )
                total_tau = tau_fe + tau_pb + tau_obs_line
                buildup = self._buildup_factor(tau_fe, tau_pb, tau_obs_line)
                attenuation += float(weight) * float(np.exp(-total_tau)) * buildup
                shield_tau_feature += float(weight) * (tau_fe + tau_pb)
                fe_tau_feature += float(weight) * tau_fe
                pb_tau_feature += float(weight) * tau_pb
                obstacle_tau_feature += float(weight) * tau_obs_line
            response_factor = self._transport_response_factor(
                isotope,
                fe_index,
                pb_index,
                shield_tau_feature,
                obstacle_tau_feature,
                fe_tau_feature,
                pb_tau_feature,
                distance_feature,
                distance_feature * shield_tau_feature,
            )
            return self._response_adjusted_attenuation(
                isotope,
                attenuation,
                response_factor,
            )
        tau_fe = float(mu_fe * l_fe)
        tau_pb = float(mu_pb * l_pb)
        total_tau = tau_fe + tau_pb + tau_obstacle
        buildup = self._buildup_factor(tau_fe, tau_pb, tau_obstacle)
        response_factor = self._transport_response_factor(
            isotope,
            fe_index,
            pb_index,
            tau_fe + tau_pb,
            tau_obstacle,
            tau_fe,
            tau_pb,
            distance_feature,
            distance_feature * (tau_fe + tau_pb),
        )
        return self._response_adjusted_attenuation(
            isotope,
            float(np.exp(-total_tau)) * buildup,
            response_factor,
        )

    def _shield_path_lengths_torch(
        self,
        direction: "torch.Tensor",
        blocked_fe: "torch.Tensor",
        blocked_pb: "torch.Tensor",
        cos_fe: "torch.Tensor",
        cos_pb: "torch.Tensor",
        tol_t: "torch.Tensor",
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Return Fe/Pb path lengths through the configured shield geometry."""
        if (
            self.shield_params.shield_geometry_model == SHIELD_GEOMETRY_SPHERICAL_OCTANT
            and not self.shield_params.use_angle_attenuation
        ):
            l_fe = spherical_shell_path_length_cm_torch(
                direction,
                self.shield_params.inner_radius_fe_cm,
                self.shield_params.inner_radius_fe_cm + self.shield_params.thickness_fe_cm,
                blocked_fe,
            )
            l_pb = spherical_shell_path_length_cm_torch(
                direction,
                self.shield_params.inner_radius_pb_cm,
                self.shield_params.inner_radius_pb_cm + self.shield_params.thickness_pb_cm,
                blocked_pb,
            )
            return l_fe, l_pb
        if self.shield_params.use_angle_attenuation:
            l_fe = torch.where(
                blocked_fe & (cos_fe > tol_t),
                torch.as_tensor(self.shield_params.thickness_fe_cm, device=device, dtype=dtype) / cos_fe,
                torch.zeros_like(cos_fe),
            )
            l_pb = torch.where(
                blocked_pb & (cos_pb > tol_t),
                torch.as_tensor(self.shield_params.thickness_pb_cm, device=device, dtype=dtype) / cos_pb,
                torch.zeros_like(cos_pb),
            )
            return l_fe, l_pb
        l_fe = torch.where(
            blocked_fe,
            torch.as_tensor(self.shield_params.thickness_fe_cm, device=device, dtype=dtype),
            torch.zeros_like(cos_fe),
        )
        l_pb = torch.where(
            blocked_pb,
            torch.as_tensor(self.shield_params.thickness_pb_cm, device=device, dtype=dtype),
            torch.zeros_like(cos_pb),
        )
        return l_fe, l_pb

    def _detector_aperture_targets_torch(
        self,
        sources: "torch.Tensor",
        detector: "torch.Tensor",
        dist: "torch.Tensor",
        tol: float,
    ) -> tuple["torch.Tensor", int]:
        """Return deterministic source-to-detector aperture targets for torch."""
        if torch is None:
            raise RuntimeError("torch is not available")
        aperture_radius = float(self.detector_aperture_radius_m or 0.0)
        if aperture_radius <= 0.0 or self.detector_aperture_samples <= 1:
            return detector.expand_as(sources).unsqueeze(-2), 1
        sample_count = max(int(self.detector_aperture_samples), 1)
        detector_expanded = detector.expand_as(sources)
        axis = (detector_expanded - sources) / dist.unsqueeze(-1)
        helper_z = torch.zeros_like(axis)
        helper_z[..., 2] = 1.0
        helper_y = torch.zeros_like(axis)
        helper_y[..., 1] = 1.0
        helper = torch.where(torch.abs(axis[..., 2:3]) > 0.9, helper_y, helper_z)
        basis_u = torch.linalg.cross(axis, helper, dim=-1)
        basis_u = basis_u / torch.clamp(torch.linalg.norm(basis_u, dim=-1, keepdim=True), min=tol)
        basis_v = torch.linalg.cross(axis, basis_u, dim=-1)
        if self.detector_aperture_sampling == "solid_angle_cone":
            return (
                self._detector_aperture_targets_cone_torch(
                    sources=sources,
                    detector=detector_expanded,
                    axis=axis,
                    basis_u=basis_u,
                    basis_v=basis_v,
                    dist=dist,
                    aperture_radius=aperture_radius,
                    sample_count=sample_count,
                    tol=tol,
                ),
                sample_count,
            )
        return (
            self._detector_aperture_targets_disk_torch(
                detector=detector_expanded,
                basis_u=basis_u,
                basis_v=basis_v,
                dist=dist,
                aperture_radius=aperture_radius,
                sample_count=sample_count,
            ),
            sample_count,
        )

    def _source_extent_points_torch(
        self,
        sources: "torch.Tensor",
        detector: "torch.Tensor",
        dist: "torch.Tensor",
        tol: float,
    ) -> tuple["torch.Tensor", int]:
        """Return deterministic source-extent sample points for torch kernels."""
        if torch is None:
            raise RuntimeError("torch is not available")
        radius = float(self.source_extent_radius_m or 0.0)
        if radius <= 0.0 or self.source_extent_samples <= 1:
            return sources.unsqueeze(-2), 1
        sample_count = max(int(self.source_extent_samples), 1)
        detector_expanded = (
            detector.expand_as(sources)
            if int(detector.shape[0]) == 1
            else detector.reshape_as(sources)
        )
        axis = (detector_expanded - sources) / dist.unsqueeze(-1)
        helper_z = torch.zeros_like(axis)
        helper_z[..., 2] = 1.0
        helper_y = torch.zeros_like(axis)
        helper_y[..., 1] = 1.0
        helper = torch.where(torch.abs(axis[..., 2:3]) > 0.9, helper_y, helper_z)
        basis_u = torch.linalg.cross(axis, helper, dim=-1)
        basis_u = basis_u / torch.clamp(
            torch.linalg.norm(basis_u, dim=-1, keepdim=True),
            min=tol,
        )
        basis_v = torch.linalg.cross(axis, basis_u, dim=-1)
        indices = torch.arange(sample_count, device=sources.device, dtype=sources.dtype)
        fractions = torch.zeros_like(indices)
        if sample_count > 1:
            fractions[1:] = (indices[1:] - 0.5) / float(sample_count - 1)
        radii = float(radius) * torch.sqrt(torch.clamp(fractions, min=0.0, max=1.0))
        angles = indices * float(np.pi * (3.0 - np.sqrt(5.0)))
        offsets = (
            radii.view(1, sample_count, 1)
            * (
                torch.cos(angles).view(1, sample_count, 1)
                * basis_u.unsqueeze(-2)
                + torch.sin(angles).view(1, sample_count, 1)
                * basis_v.unsqueeze(-2)
            )
        )
        return sources.unsqueeze(-2) + offsets, sample_count

    def _ray_sample_points_torch(
        self,
        sources: "torch.Tensor",
        detector: "torch.Tensor",
        dist: "torch.Tensor",
        tol: float,
    ) -> tuple["torch.Tensor", "torch.Tensor", int]:
        """Return flattened source and detector target pairs for torch kernels."""
        if torch is None:
            raise RuntimeError("torch is not available")
        source_points, source_sample_count = self._source_extent_points_torch(
            sources=sources,
            detector=detector,
            dist=dist,
            tol=tol,
        )
        detector_expanded = (
            detector.expand_as(sources)
            if int(detector.shape[0]) == 1
            else detector.reshape_as(sources)
        )
        flat_sources = source_points.reshape(-1, 3)
        flat_detectors = (
            detector_expanded.unsqueeze(-2)
            .expand(-1, source_sample_count, -1)
            .reshape(-1, 3)
        )
        flat_dist = torch.linalg.norm(flat_detectors - flat_sources, dim=1)
        tol_t = torch.as_tensor(tol, device=sources.device, dtype=sources.dtype)
        flat_dist = torch.where(flat_dist <= tol_t, tol_t, flat_dist)
        flat_targets, aperture_sample_count = self._detector_aperture_targets_torch(
            sources=flat_sources,
            detector=flat_detectors,
            dist=flat_dist,
            tol=tol,
        )
        total_sample_count = int(source_sample_count) * int(aperture_sample_count)
        sampled_sources = flat_sources.unsqueeze(-2).expand_as(flat_targets)
        return (
            sampled_sources.reshape(int(sources.shape[0]), total_sample_count, 3),
            flat_targets.reshape(int(sources.shape[0]), total_sample_count, 3),
            total_sample_count,
        )

    def _detector_aperture_targets_disk_torch(
        self,
        *,
        detector: "torch.Tensor",
        basis_u: "torch.Tensor",
        basis_v: "torch.Tensor",
        dist: "torch.Tensor",
        aperture_radius: float,
        sample_count: int,
    ) -> "torch.Tensor":
        """Return legacy deterministic disk aperture targets for torch."""
        indices = torch.arange(
            sample_count,
            device=detector.device,
            dtype=detector.dtype,
        )
        fractions = torch.clamp(
            (indices - 0.5) / float(sample_count - 1),
            min=0.0,
            max=1.0,
        )
        radii = torch.sqrt(fractions)
        radii[0] = 0.0
        max_radius = torch.minimum(
            torch.as_tensor(aperture_radius, device=detector.device, dtype=detector.dtype),
            0.95 * dist,
        )
        angles = indices * torch.as_tensor(
            np.pi * (3.0 - np.sqrt(5.0)),
            device=detector.device,
            dtype=detector.dtype,
        )
        offsets = (
            max_radius.unsqueeze(-1).unsqueeze(-1)
            * radii.view(1, sample_count, 1)
            * (
                torch.cos(angles).view(1, sample_count, 1) * basis_u.unsqueeze(-2)
                + torch.sin(angles).view(1, sample_count, 1) * basis_v.unsqueeze(-2)
            )
        )
        return detector.unsqueeze(-2) + offsets

    def _detector_aperture_targets_cone_torch(
        self,
        *,
        sources: "torch.Tensor",
        detector: "torch.Tensor",
        axis: "torch.Tensor",
        basis_u: "torch.Tensor",
        basis_v: "torch.Tensor",
        dist: "torch.Tensor",
        aperture_radius: float,
        sample_count: int,
        tol: float,
    ) -> "torch.Tensor":
        """Return Geant4 detector-cone targets on the aperture sphere for torch."""
        radius_t = torch.as_tensor(
            aperture_radius,
            device=sources.device,
            dtype=sources.dtype,
        )
        indices = torch.arange(sample_count, device=sources.device, dtype=sources.dtype)
        fractions = (indices + 0.5) / float(sample_count)
        dist_safe = torch.clamp(dist, min=tol)
        sin_theta_max = torch.clamp(radius_t / dist_safe, min=0.0, max=1.0)
        cos_theta_max = torch.sqrt(
            torch.clamp(1.0 - sin_theta_max * sin_theta_max, min=0.0)
        )
        cos_theta = 1.0 - fractions.view(1, sample_count) * (
            1.0 - cos_theta_max.unsqueeze(-1)
        )
        sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta * cos_theta, min=0.0))
        angles = indices * torch.as_tensor(
            np.pi * (3.0 - np.sqrt(5.0)),
            device=sources.device,
            dtype=sources.dtype,
        )
        direction = (
            cos_theta.unsqueeze(-1) * axis.unsqueeze(-2)
            + sin_theta.unsqueeze(-1)
            * (
                torch.cos(angles).view(1, sample_count, 1) * basis_u.unsqueeze(-2)
                + torch.sin(angles).view(1, sample_count, 1) * basis_v.unsqueeze(-2)
            )
        )
        radial_sq = (dist.unsqueeze(-1) * sin_theta) ** 2
        chord = torch.sqrt(torch.clamp(radius_t * radius_t - radial_sq, min=0.0))
        path_length = dist.unsqueeze(-1) * cos_theta - chord
        targets = sources.unsqueeze(-2) + path_length.unsqueeze(-1) * direction
        inside = dist <= radius_t
        if bool(torch.any(inside)):
            fallback = detector.unsqueeze(-2).expand_as(targets)
            targets = torch.where(inside.view(-1, 1, 1), fallback, targets)
        return targets

    def _expected_rate_pair_torch(
        self,
        isotope: str,
        detector_pos: NDArray[np.float64],
        sources: NDArray[np.float64],
        strengths: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        background: float,
        tol: float = 1e-6,
    ) -> float:
        """Compute expected rate for a Fe/Pb orientation pair using torch."""
        if torch is None:
            raise RuntimeError("torch is not available")
        device = _resolve_device(self.gpu_device)
        dtype = _resolve_dtype(self.gpu_dtype)
        sources_t = torch.as_tensor(sources, device=device, dtype=dtype)
        if sources_t.numel() == 0:
            return float(background)
        strengths_t = torch.as_tensor(strengths, device=device, dtype=dtype)
        detector_t = torch.as_tensor(detector_pos, device=device, dtype=dtype).view(1, 3)
        direction = detector_t - sources_t
        dist = torch.linalg.norm(direction, dim=1)
        tol_t = torch.as_tensor(tol, device=device, dtype=dtype)
        dist = torch.where(dist <= tol_t, tol_t, dist)
        geom = _finite_sphere_geometric_term_torch(
            dist,
            detector_radius_m=self.detector_radius_m,
            tol=tol_t,
        )
        sampled_sources, targets, sample_count = self._ray_sample_points_torch(
            sources=sources_t,
            detector=detector_t,
            dist=dist,
            tol=tol,
        )
        sampled_direction = targets - sampled_sources
        sampled_dist = torch.linalg.norm(sampled_direction, dim=-1)
        sampled_dist = torch.where(sampled_dist <= tol_t, tol_t, sampled_dist)
        dir_unit = sampled_direction / sampled_dist.unsqueeze(-1)

        detector_to_source_unit = -dir_unit
        blocked_fe = self._rotated_octant_blocked_mask_torch(detector_to_source_unit, fe_index, tol)
        blocked_pb = self._rotated_octant_blocked_mask_torch(detector_to_source_unit, pb_index, tol)
        normal_fe = torch.as_tensor(self.orientations[fe_index], device=device, dtype=dtype)
        normal_pb = torch.as_tensor(self.orientations[pb_index], device=device, dtype=dtype)
        cos_fe = torch.clamp(torch.sum(dir_unit * normal_fe, dim=-1), 0.0, 1.0)
        cos_pb = torch.clamp(torch.sum(dir_unit * normal_pb, dim=-1), 0.0, 1.0)
        if (
            self.shield_params.shield_geometry_model == SHIELD_GEOMETRY_SPHERICAL_OCTANT
            and not self.shield_params.use_angle_attenuation
        ):
            center = detector_t.expand_as(sources_t).unsqueeze(-2)
            fe_normal = -np.asarray(self.orientations[fe_index], dtype=float)
            pb_normal = -np.asarray(self.orientations[pb_index], dtype=float)
            fe_rotation = self._rotated_octant_rotation_torch(
                fe_normal,
                device=device,
                dtype=dtype,
            )
            pb_rotation = self._rotated_octant_rotation_torch(
                pb_normal,
                device=device,
                dtype=dtype,
            )
            L_fe = segment_rotated_octant_shell_path_length_cm_torch(
                source_pos=sampled_sources,
                target_pos=targets,
                center_pos=center,
                shield_normal=fe_normal,
                inner_radius_cm=self.shield_params.inner_radius_fe_cm,
                outer_radius_cm=self.shield_params.inner_radius_fe_cm + self.shield_params.thickness_fe_cm,
                tol=tol,
                rotation=fe_rotation,
            )
            L_pb = segment_rotated_octant_shell_path_length_cm_torch(
                source_pos=sampled_sources,
                target_pos=targets,
                center_pos=center,
                shield_normal=pb_normal,
                inner_radius_cm=self.shield_params.inner_radius_pb_cm,
                outer_radius_cm=self.shield_params.inner_radius_pb_cm + self.shield_params.thickness_pb_cm,
                tol=tol,
                rotation=pb_rotation,
            )
        else:
            L_fe, L_pb = self._shield_path_lengths_torch(
                direction=sampled_direction,
                blocked_fe=blocked_fe,
                blocked_pb=blocked_pb,
                cos_fe=cos_fe,
                cos_pb=cos_pb,
                tol_t=tol_t,
                device=device,
                dtype=dtype,
            )
        mu_fe, mu_pb = self._mu_values(isotope=isotope)
        tau_fe = float(mu_fe) * L_fe
        tau_pb = float(mu_pb) * L_pb
        tau_obstacle = torch.zeros_like(tau_fe)
        obstacle_path_cm = None
        boxes_np = self.obstacle_boxes_m()
        if boxes_np.size:
            boxes_t = torch.as_tensor(boxes_np, device=device, dtype=dtype)
            if sample_count > 1:
                obstacle_path_cm = obstacle_path_lengths_between_points_by_box_cm_torch(
                    source_pos=sampled_sources,
                    target_pos=targets,
                    obstacle_boxes_m=boxes_t,
                    tol=tol,
                )
                tau_obstacle = self._obstacle_tau_from_lengths_torch(
                    isotope,
                    obstacle_path_cm,
                    device=device,
                    dtype=dtype,
                )
            else:
                obstacle_path_cm = obstacle_path_lengths_by_box_cm_torch(
                    positions=sources_t,
                    detector_pos=detector_t.reshape(3),
                    obstacle_boxes_m=boxes_t,
                )
                tau_obstacle = self._obstacle_tau_from_lengths_torch(
                    isotope,
                    obstacle_path_cm,
                    device=device,
                    dtype=dtype,
                ).unsqueeze(-1)
        line_entries = self._line_mu_values(isotope)
        if line_entries:
            weights_t = torch.as_tensor(
                [entry[0] for entry in line_entries],
                device=device,
                dtype=dtype,
            )
            mu_fe_t = torch.as_tensor(
                [entry[1] for entry in line_entries],
                device=device,
                dtype=dtype,
            )
            mu_pb_t = torch.as_tensor(
                [entry[2] for entry in line_entries],
                device=device,
                dtype=dtype,
            )
            line_tau_fe = L_fe.unsqueeze(-1) * mu_fe_t.view(1, 1, -1)
            line_tau_pb = L_pb.unsqueeze(-1) * mu_pb_t.view(1, 1, -1)
            line_tau_obstacle = tau_obstacle.unsqueeze(-1)
            if obstacle_path_cm is not None:
                candidate_tau = self._obstacle_line_tau_from_lengths_torch(
                    isotope,
                    obstacle_path_cm,
                    line_count=len(line_entries),
                    device=device,
                    dtype=dtype,
                )
                if candidate_tau is not None:
                    line_tau_obstacle = (
                        candidate_tau
                        if sample_count > 1
                        else candidate_tau.unsqueeze(-2)
                    )
            line_total_tau = line_tau_fe + line_tau_pb + line_tau_obstacle
            line_buildup = self._buildup_factor_torch(
                line_tau_fe,
                line_tau_pb,
                line_tau_obstacle,
            )
            shield_tau_feature = torch.sum(
                (line_tau_fe + line_tau_pb) * weights_t.view(1, 1, -1),
                dim=-1,
            )
            fe_tau_feature = torch.sum(
                line_tau_fe * weights_t.view(1, 1, -1),
                dim=-1,
            )
            pb_tau_feature = torch.sum(
                line_tau_pb * weights_t.view(1, 1, -1),
                dim=-1,
            )
            obstacle_tau_feature = torch.sum(
                line_tau_obstacle * weights_t.view(1, 1, -1),
                dim=-1,
            )
            response_factor = self._transport_response_factor_torch(
                isotope,
                np.asarray([int(fe_index)], dtype=np.int64),
                np.asarray([int(pb_index)], dtype=np.int64),
                shield_tau_feature,
                obstacle_tau_feature,
                fe_tau_feature,
                pb_tau_feature,
                distance_feature=dist.unsqueeze(-1),
                distance_shield_feature=dist.unsqueeze(-1)
                * shield_tau_feature,
                device=device,
                dtype=dtype,
            )
            base_att = torch.sum(
                torch.exp(-line_total_tau) * line_buildup * weights_t.view(1, 1, -1),
                dim=-1,
            )
            att = self._response_adjusted_attenuation_torch(
                isotope,
                base_att,
                response_factor,
            )
        else:
            total_tau = tau_fe + tau_pb + tau_obstacle
            buildup = self._buildup_factor_torch(tau_fe, tau_pb, tau_obstacle)
            response_factor = self._transport_response_factor_torch(
                isotope,
                np.asarray([int(fe_index)], dtype=np.int64),
                np.asarray([int(pb_index)], dtype=np.int64),
                tau_fe + tau_pb,
                tau_obstacle,
                tau_fe,
                tau_pb,
                distance_feature=dist.unsqueeze(-1),
                distance_shield_feature=dist.unsqueeze(-1)
                * (tau_fe + tau_pb),
                device=device,
                dtype=dtype,
            )
            att = self._response_adjusted_attenuation_torch(
                isotope,
                torch.exp(-total_tau) * buildup,
                response_factor,
            )
        att = torch.mean(att, dim=-1)
        rate = torch.sum(geom * att * strengths_t) + float(background)
        return float(rate.detach().cpu().item())

    def _kernel_values_selected_pairs_torch_tensor(
        self,
        isotope: str,
        detector_pos: NDArray[np.float64],
        sources_t: "torch.Tensor",
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
        tol: float = 1e-6,
    ) -> "torch.Tensor":
        """Return selected Fe/Pb-pair kernels for flat torch source rows."""
        if torch is None:
            raise RuntimeError("torch is not available")
        if sources_t.ndim != 2 or int(sources_t.shape[1]) != 3:
            raise ValueError("sources_t must be shaped (N, 3).")
        device = sources_t.device
        dtype = sources_t.dtype
        num_orients = int(len(self.orientations))
        fe_arr = np.asarray(fe_indices, dtype=int).reshape(-1) % num_orients
        pb_arr = np.asarray(pb_indices, dtype=int).reshape(-1) % num_orients
        if fe_arr.size != pb_arr.size:
            raise ValueError("Fe and Pb index arrays must have matching lengths.")
        pair_count = int(fe_arr.size)
        if pair_count == 0:
            return torch.zeros((0, int(sources_t.shape[0])), device=device, dtype=dtype)
        if sources_t.numel() == 0:
            return torch.zeros((pair_count, 0), device=device, dtype=dtype)

        detector_t = torch.as_tensor(detector_pos, device=device, dtype=dtype).view(1, 3)
        direction = detector_t - sources_t
        dist = torch.linalg.norm(direction, dim=1)
        tol_t = torch.as_tensor(tol, device=device, dtype=dtype)
        dist = torch.where(dist <= tol_t, tol_t, dist)
        geom = _finite_sphere_geometric_term_torch(
            dist,
            detector_radius_m=self.detector_radius_m,
            tol=tol_t,
        )
        sampled_sources, targets, sample_count = self._ray_sample_points_torch(
            sources=sources_t,
            detector=detector_t,
            dist=dist,
            tol=tol,
        )
        sampled_direction = targets - sampled_sources
        sampled_dist = torch.linalg.norm(sampled_direction, dim=-1)
        sampled_dist = torch.where(sampled_dist <= tol_t, tol_t, sampled_dist)
        dir_unit = sampled_direction / sampled_dist.unsqueeze(-1)
        detector_to_source_unit = -dir_unit

        unique_orients = np.unique(np.concatenate([fe_arr, pb_arr]))
        path_lengths: dict[int, tuple["torch.Tensor", "torch.Tensor"]] = {}
        for orient_idx in unique_orients:
            orient_int = int(orient_idx)
            if (
                self.shield_params.shield_geometry_model
                == SHIELD_GEOMETRY_SPHERICAL_OCTANT
                and not self.shield_params.use_angle_attenuation
            ):
                center = detector_t.expand_as(sources_t).unsqueeze(-2)
                shield_normal = -np.asarray(
                    self.orientations[orient_int],
                    dtype=float,
                )
                rotation = self._rotated_octant_rotation_torch(
                    shield_normal,
                    device=device,
                    dtype=dtype,
                )
                l_fe = segment_rotated_octant_shell_path_length_cm_torch(
                    source_pos=sampled_sources,
                    target_pos=targets,
                    center_pos=center,
                    shield_normal=shield_normal,
                    inner_radius_cm=self.shield_params.inner_radius_fe_cm,
                    outer_radius_cm=(
                        self.shield_params.inner_radius_fe_cm
                        + self.shield_params.thickness_fe_cm
                    ),
                    tol=tol,
                    rotation=rotation,
                )
                l_pb = segment_rotated_octant_shell_path_length_cm_torch(
                    source_pos=sampled_sources,
                    target_pos=targets,
                    center_pos=center,
                    shield_normal=shield_normal,
                    inner_radius_cm=self.shield_params.inner_radius_pb_cm,
                    outer_radius_cm=(
                        self.shield_params.inner_radius_pb_cm
                        + self.shield_params.thickness_pb_cm
                    ),
                    tol=tol,
                    rotation=rotation,
                )
            else:
                blocked = self._rotated_octant_blocked_mask_torch(
                    detector_to_source_unit,
                    orient_int,
                    tol,
                )
                normal = torch.as_tensor(
                    self.orientations[orient_int],
                    device=device,
                    dtype=dtype,
                )
                cos_theta = torch.clamp(
                    torch.sum(dir_unit * normal, dim=-1),
                    0.0,
                    1.0,
                )
                l_fe, l_pb = self._shield_path_lengths_torch(
                    direction=sampled_direction,
                    blocked_fe=blocked,
                    blocked_pb=blocked,
                    cos_fe=cos_theta,
                    cos_pb=cos_theta,
                    tol_t=tol_t,
                    device=device,
                    dtype=dtype,
                )
            path_lengths[orient_int] = (l_fe, l_pb)

        l_fe_pairs = torch.stack([path_lengths[int(idx)][0] for idx in fe_arr], dim=0)
        l_pb_pairs = torch.stack([path_lengths[int(idx)][1] for idx in pb_arr], dim=0)
        tau_fe = torch.zeros_like(l_fe_pairs)
        tau_pb = torch.zeros_like(l_pb_pairs)
        mu_fe, mu_pb = self._mu_values(isotope=isotope)
        tau_fe = tau_fe + float(mu_fe) * l_fe_pairs
        tau_pb = tau_pb + float(mu_pb) * l_pb_pairs

        obstacle_path_cm = None
        tau_obstacle_base = torch.zeros(
            sampled_sources.shape[:-1],
            device=device,
            dtype=dtype,
        )
        boxes_np = self.obstacle_boxes_m()
        if boxes_np.size:
            boxes_t = torch.as_tensor(boxes_np, device=device, dtype=dtype)
            obstacle_path_cm = obstacle_path_lengths_between_points_by_box_cm_torch(
                source_pos=sampled_sources,
                target_pos=targets,
                obstacle_boxes_m=boxes_t,
                tol=tol,
            )
            tau_obstacle_base = self._obstacle_tau_from_lengths_torch(
                isotope,
                obstacle_path_cm,
                device=device,
                dtype=dtype,
            )
        tau_obstacle = tau_obstacle_base.unsqueeze(0)

        line_entries = self._line_mu_values(isotope)
        if line_entries:
            weights_t = torch.as_tensor(
                [entry[0] for entry in line_entries],
                device=device,
                dtype=dtype,
            )
            mu_fe_t = torch.as_tensor(
                [entry[1] for entry in line_entries],
                device=device,
                dtype=dtype,
            )
            mu_pb_t = torch.as_tensor(
                [entry[2] for entry in line_entries],
                device=device,
                dtype=dtype,
            )
            line_tau_fe = l_fe_pairs.unsqueeze(-1) * mu_fe_t.view(1, 1, 1, -1)
            line_tau_pb = l_pb_pairs.unsqueeze(-1) * mu_pb_t.view(1, 1, 1, -1)
            line_tau_obstacle = tau_obstacle.unsqueeze(-1)
            if obstacle_path_cm is not None:
                candidate_tau = self._obstacle_line_tau_from_lengths_torch(
                    isotope,
                    obstacle_path_cm,
                    line_count=len(line_entries),
                    device=device,
                    dtype=dtype,
                )
                if candidate_tau is not None:
                    line_tau_obstacle = candidate_tau.unsqueeze(0)
            line_buildup = self._buildup_factor_torch(
                line_tau_fe,
                line_tau_pb,
                line_tau_obstacle,
            )
            shield_tau_feature = torch.sum(
                (line_tau_fe + line_tau_pb) * weights_t.view(1, 1, 1, -1),
                dim=-1,
            )
            fe_tau_feature = torch.sum(
                line_tau_fe * weights_t.view(1, 1, 1, -1),
                dim=-1,
            )
            pb_tau_feature = torch.sum(
                line_tau_pb * weights_t.view(1, 1, 1, -1),
                dim=-1,
            )
            obstacle_tau_feature = torch.sum(
                line_tau_obstacle * weights_t.view(1, 1, 1, -1),
                dim=-1,
            )
            response_factor = self._transport_response_factor_torch(
                isotope,
                fe_arr,
                pb_arr,
                shield_tau_feature,
                obstacle_tau_feature,
                fe_tau_feature,
                pb_tau_feature,
                distance_feature=dist.view(1, -1, 1),
                distance_shield_feature=dist.view(1, -1, 1)
                * shield_tau_feature,
                device=device,
                dtype=dtype,
            )
            base_att = torch.sum(
                torch.exp(-(line_tau_fe + line_tau_pb + line_tau_obstacle))
                * line_buildup
                * weights_t.view(1, 1, 1, -1),
                dim=-1,
            )
            att = self._response_adjusted_attenuation_torch(
                isotope,
                base_att,
                response_factor,
            )
        else:
            total_tau = tau_fe + tau_pb + tau_obstacle
            buildup = self._buildup_factor_torch(tau_fe, tau_pb, tau_obstacle)
            response_factor = self._transport_response_factor_torch(
                isotope,
                fe_arr,
                pb_arr,
                tau_fe + tau_pb,
                tau_obstacle,
                tau_fe,
                tau_pb,
                distance_feature=dist.view(1, -1, 1),
                distance_shield_feature=dist.view(1, -1, 1)
                * (tau_fe + tau_pb),
                device=device,
                dtype=dtype,
            )
            att = self._response_adjusted_attenuation_torch(
                isotope,
                torch.exp(-total_tau) * buildup,
                response_factor,
            )

        att = torch.mean(att, dim=-1)
        return geom.unsqueeze(0) * att

    def _expected_counts_selected_pairs_for_packed_states_torch(
        self,
        *,
        isotope: str,
        detector_pos: NDArray[np.float64],
        positions: "torch.Tensor",
        strengths: "torch.Tensor",
        backgrounds: "torch.Tensor",
        mask: "torch.Tensor",
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
        live_time_s: float,
        source_scale: float | NDArray[np.float64] | "torch.Tensor",
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> "torch.Tensor":
        """Compute selected pair counts from the ContinuousKernel torch kernel."""
        if torch is None:
            raise RuntimeError("torch is not available")
        fe_arr = np.asarray(fe_indices, dtype=int).reshape(-1)
        pb_arr = np.asarray(pb_indices, dtype=int).reshape(-1)
        if fe_arr.size != pb_arr.size:
            raise ValueError("Fe and Pb index arrays must have matching lengths.")
        pair_count = int(fe_arr.size)
        particle_count = int(positions.shape[0])
        slot_count = int(positions.shape[1]) if positions.ndim >= 2 else 0
        if pair_count == 0:
            return torch.zeros((0, particle_count), device=device, dtype=dtype)
        if particle_count == 0 or slot_count == 0:
            source_scale_t = _source_scale_rows_torch(
                source_scale,
                pair_count,
                device=device,
                dtype=dtype,
            )
            rate = source_scale_t * torch.zeros(
                (pair_count, particle_count),
                device=device,
                dtype=dtype,
            )
            return float(live_time_s) * (rate + backgrounds.unsqueeze(0))

        all_pairs = pair_count >= int(len(self.orientations)) ** 2
        source_chunk = self._adaptive_torch_chunk_size(
            8192,
            all_orientation_pairs=all_pairs,
        )
        particle_chunk = max(1, int(source_chunk) // max(slot_count, 1))
        source_terms: list["torch.Tensor"] = []
        strengths_masked = strengths * mask
        with torch.no_grad():
            for start in range(0, particle_count, particle_chunk):
                stop = min(start + particle_chunk, particle_count)
                flat_sources = positions[start:stop].reshape(-1, 3)
                kernel_values = self._kernel_values_selected_pairs_torch_tensor(
                    isotope=isotope,
                    detector_pos=detector_pos,
                    sources_t=flat_sources,
                    fe_indices=fe_arr,
                    pb_indices=pb_arr,
                )
                kernel_values = kernel_values.reshape(
                    pair_count,
                    stop - start,
                    slot_count,
                )
                weighted = kernel_values * strengths_masked[start:stop].unsqueeze(0)
                source_terms.append(torch.sum(weighted, dim=-1))
            source_term = torch.cat(source_terms, dim=1)
            source_scale_t = _source_scale_rows_torch(
                source_scale,
                pair_count,
                device=device,
                dtype=dtype,
            )
            rate = source_scale_t * source_term + backgrounds.unsqueeze(0)
        return float(live_time_s) * rate

    def _kernel_values_pair_torch_chunk(
        self,
        isotope: str,
        detector_pos: NDArray[np.float64],
        sources: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        tol: float = 1e-6,
    ) -> NDArray[np.float64]:
        """Return per-source kernel values for one GPU chunk."""
        if torch is None:
            raise RuntimeError("torch is not available")
        device = _resolve_device(self.gpu_device)
        dtype = _resolve_dtype(self.gpu_dtype)
        sources_t = torch.as_tensor(sources, device=device, dtype=dtype)
        values = self._kernel_values_selected_pairs_torch_tensor(
            isotope=isotope,
            detector_pos=detector_pos,
            sources_t=sources_t,
            fe_indices=np.asarray([fe_index], dtype=np.int64),
            pb_indices=np.asarray([pb_index], dtype=np.int64),
            tol=tol,
        )
        return values[0].detach().cpu().numpy().astype(float, copy=False)

    def _kernel_values_all_pairs_torch_chunk(
        self,
        isotope: str,
        detector_pos: NDArray[np.float64],
        sources: NDArray[np.float64],
        tol: float = 1e-6,
    ) -> NDArray[np.float64]:
        """Return per-source kernel values for every Fe/Pb pair on the GPU."""
        if torch is None:
            raise RuntimeError("torch is not available")
        device = _resolve_device(self.gpu_device)
        dtype = _resolve_dtype(self.gpu_dtype)
        sources_t = torch.as_tensor(sources, device=device, dtype=dtype)
        num_orients = int(len(self.orientations))
        num_pairs = num_orients * num_orients
        if sources_t.numel() == 0:
            return np.zeros((num_pairs, 0), dtype=float)
        pair_ids = np.arange(num_pairs, dtype=np.int64)
        values = self._kernel_values_selected_pairs_torch_tensor(
            isotope=isotope,
            detector_pos=detector_pos,
            sources_t=sources_t,
            fe_indices=pair_ids // num_orients,
            pb_indices=pair_ids % num_orients,
            tol=tol,
        )
        return values.detach().cpu().numpy().astype(float, copy=False)

    def _kernel_values_all_pairs_for_detector_source_torch_chunk(
        self,
        isotope: str,
        detector_positions: NDArray[np.float64],
        sources: NDArray[np.float64],
        tol: float = 1e-6,
    ) -> NDArray[np.float64]:
        """Return all Fe/Pb pair kernels for matched detector-source rows."""
        if torch is None:
            raise RuntimeError("torch is not available")
        device = _resolve_device(self.gpu_device)
        dtype = _resolve_dtype(self.gpu_dtype)
        detectors_t = torch.as_tensor(detector_positions, device=device, dtype=dtype)
        sources_t = torch.as_tensor(sources, device=device, dtype=dtype)
        if detectors_t.ndim != 2 or detectors_t.shape[1] != 3:
            raise ValueError("detector_positions must be shaped (N, 3).")
        if sources_t.ndim != 2 or sources_t.shape[1] != 3:
            raise ValueError("sources must be shaped (N, 3).")
        if detectors_t.shape[0] != sources_t.shape[0]:
            raise ValueError("detector_positions and sources must have the same row count.")
        num_orients = int(len(self.orientations))
        num_pairs = num_orients * num_orients
        if sources_t.numel() == 0:
            return np.zeros((0, num_pairs), dtype=float)
        with torch.no_grad():
            direction = detectors_t - sources_t
            dist = torch.linalg.norm(direction, dim=1)
            tol_t = torch.as_tensor(tol, device=device, dtype=dtype)
            dist = torch.where(dist <= tol_t, tol_t, dist)
            geom = _finite_sphere_geometric_term_torch(
                dist,
                detector_radius_m=self.detector_radius_m,
                tol=tol_t,
            )
            sampled_sources, targets, sample_count = self._ray_sample_points_torch(
                sources=sources_t,
                detector=detectors_t,
                dist=dist,
                tol=tol,
            )
            sampled_direction = targets - sampled_sources
            sampled_dist = torch.linalg.norm(sampled_direction, dim=-1)
            sampled_dist = torch.where(sampled_dist <= tol_t, tol_t, sampled_dist)
            dir_unit = sampled_direction / sampled_dist.unsqueeze(-1)
            detector_to_source_unit = -dir_unit

            l_fe_by_orient: list[torch.Tensor] = []
            l_pb_by_orient: list[torch.Tensor] = []
            for orient_idx in range(num_orients):
                if (
                    self.shield_params.shield_geometry_model
                    == SHIELD_GEOMETRY_SPHERICAL_OCTANT
                    and not self.shield_params.use_angle_attenuation
                ):
                    shield_normal = -np.asarray(
                        self.orientations[orient_idx],
                        dtype=float,
                    )
                    rotation = self._rotated_octant_rotation_torch(
                        shield_normal,
                        device=device,
                        dtype=dtype,
                    )
                    center = detectors_t.unsqueeze(-2)
                    l_fe = segment_rotated_octant_shell_path_length_cm_torch(
                        source_pos=sampled_sources,
                        target_pos=targets,
                        center_pos=center,
                        shield_normal=shield_normal,
                        inner_radius_cm=self.shield_params.inner_radius_fe_cm,
                        outer_radius_cm=(
                            self.shield_params.inner_radius_fe_cm
                            + self.shield_params.thickness_fe_cm
                        ),
                        tol=tol,
                        rotation=rotation,
                    )
                    l_pb = segment_rotated_octant_shell_path_length_cm_torch(
                        source_pos=sampled_sources,
                        target_pos=targets,
                        center_pos=center,
                        shield_normal=shield_normal,
                        inner_radius_cm=self.shield_params.inner_radius_pb_cm,
                        outer_radius_cm=(
                            self.shield_params.inner_radius_pb_cm
                            + self.shield_params.thickness_pb_cm
                        ),
                        tol=tol,
                        rotation=rotation,
                    )
                else:
                    blocked = self._rotated_octant_blocked_mask_torch(
                        detector_to_source_unit,
                        orient_idx,
                        tol,
                    )
                    normal = torch.as_tensor(
                        self.orientations[orient_idx],
                        device=device,
                        dtype=dtype,
                    )
                    cos_theta = torch.clamp(
                        torch.sum(dir_unit * normal, dim=-1),
                        0.0,
                        1.0,
                    )
                    l_fe, l_pb = self._shield_path_lengths_torch(
                        direction=sampled_direction,
                        blocked_fe=blocked,
                        blocked_pb=blocked,
                        cos_fe=cos_theta,
                        cos_pb=cos_theta,
                        tol_t=tol_t,
                        device=device,
                        dtype=dtype,
                    )
                l_fe_by_orient.append(l_fe)
                l_pb_by_orient.append(l_pb)

            l_fe_stack = torch.stack(l_fe_by_orient, dim=0)
            l_pb_stack = torch.stack(l_pb_by_orient, dim=0)
            pair_ids = torch.arange(num_pairs, device=device, dtype=torch.long)
            fe_indices = torch.div(pair_ids, num_orients, rounding_mode="floor")
            pb_indices = torch.remainder(pair_ids, num_orients)
            pair_ids_np = np.arange(num_pairs, dtype=np.int64)
            fe_indices_np = pair_ids_np // num_orients
            pb_indices_np = pair_ids_np % num_orients
            l_fe_pairs = l_fe_stack.index_select(0, fe_indices)
            l_pb_pairs = l_pb_stack.index_select(0, pb_indices)

            mu_fe, mu_pb = self._mu_values(isotope=isotope)
            tau_fe = float(mu_fe) * l_fe_pairs
            tau_pb = float(mu_pb) * l_pb_pairs
            tau_obstacle = torch.zeros_like(tau_fe)
            obstacle_path_cm = None
            boxes_np = self.obstacle_boxes_m()
            if boxes_np.size:
                boxes_t = torch.as_tensor(boxes_np, device=device, dtype=dtype)
                obstacle_path_cm = obstacle_path_lengths_between_points_by_box_cm_torch(
                    source_pos=sampled_sources,
                    target_pos=targets,
                    obstacle_boxes_m=boxes_t,
                    tol=tol,
                )
                tau_obstacle = self._obstacle_tau_from_lengths_torch(
                    isotope,
                    obstacle_path_cm,
                    device=device,
                    dtype=dtype,
                ).unsqueeze(0)
            line_entries = self._line_mu_values(isotope)
            if line_entries:
                weights_t = torch.as_tensor(
                    [entry[0] for entry in line_entries],
                    device=device,
                    dtype=dtype,
                )
                mu_fe_t = torch.as_tensor(
                    [entry[1] for entry in line_entries],
                    device=device,
                    dtype=dtype,
                )
                mu_pb_t = torch.as_tensor(
                    [entry[2] for entry in line_entries],
                    device=device,
                    dtype=dtype,
                )
                line_tau_fe = l_fe_pairs.unsqueeze(-1) * mu_fe_t.view(1, 1, 1, -1)
                line_tau_pb = l_pb_pairs.unsqueeze(-1) * mu_pb_t.view(1, 1, 1, -1)
                line_tau_obstacle = tau_obstacle.unsqueeze(-1)
                if obstacle_path_cm is not None:
                    candidate_tau = self._obstacle_line_tau_from_lengths_torch(
                        isotope,
                        obstacle_path_cm,
                        line_count=len(line_entries),
                        device=device,
                        dtype=dtype,
                    )
                    if candidate_tau is not None:
                        line_tau_obstacle = candidate_tau.unsqueeze(0)
                line_total_tau = line_tau_fe + line_tau_pb + line_tau_obstacle
                line_buildup = self._buildup_factor_torch(
                    line_tau_fe,
                    line_tau_pb,
                    line_tau_obstacle,
                )
                shield_tau_feature = torch.sum(
                    (line_tau_fe + line_tau_pb) * weights_t.view(1, 1, 1, -1),
                    dim=-1,
                )
                fe_tau_feature = torch.sum(
                    line_tau_fe * weights_t.view(1, 1, 1, -1),
                    dim=-1,
                )
                pb_tau_feature = torch.sum(
                    line_tau_pb * weights_t.view(1, 1, 1, -1),
                    dim=-1,
                )
                obstacle_tau_feature = torch.sum(
                    line_tau_obstacle * weights_t.view(1, 1, 1, -1),
                    dim=-1,
                )
                response_factor = self._transport_response_factor_torch(
                    isotope,
                    fe_indices_np,
                    pb_indices_np,
                    shield_tau_feature,
                    obstacle_tau_feature,
                    fe_tau_feature,
                    pb_tau_feature,
                    distance_feature=dist.view(1, -1, 1),
                    distance_shield_feature=dist.view(1, -1, 1)
                    * shield_tau_feature,
                    device=device,
                    dtype=dtype,
                )
                base_att = torch.sum(
                    torch.exp(-line_total_tau)
                    * line_buildup
                    * weights_t.view(1, 1, 1, -1),
                    dim=-1,
                )
                att = self._response_adjusted_attenuation_torch(
                    isotope,
                    base_att,
                    response_factor,
                )
            else:
                total_tau = tau_fe + tau_pb + tau_obstacle
                buildup = self._buildup_factor_torch(tau_fe, tau_pb, tau_obstacle)
                response_factor = self._transport_response_factor_torch(
                    isotope,
                    fe_indices_np,
                    pb_indices_np,
                    tau_fe + tau_pb,
                    tau_obstacle,
                    tau_fe,
                    tau_pb,
                    distance_feature=dist.view(1, -1, 1),
                    distance_shield_feature=dist.view(1, -1, 1)
                    * (tau_fe + tau_pb),
                    device=device,
                    dtype=dtype,
                )
                att = self._response_adjusted_attenuation_torch(
                    isotope,
                    torch.exp(-total_tau) * buildup,
                    response_factor,
                )
            att = torch.mean(att, dim=-1)
            values = geom.unsqueeze(0) * att
        return values.transpose(0, 1).detach().cpu().numpy().astype(float, copy=False)

    def _kernel_values_selected_pairs_for_detector_source_torch_chunk(
        self,
        isotope: str,
        detector_positions: NDArray[np.float64],
        sources: NDArray[np.float64],
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
        tol: float = 1e-6,
    ) -> NDArray[np.float64]:
        """Return selected Fe/Pb-pair kernels for matched detector-source rows."""
        if torch is None:
            raise RuntimeError("torch is not available")
        device = _resolve_device(self.gpu_device)
        dtype = _resolve_dtype(self.gpu_dtype)
        detectors_t = torch.as_tensor(detector_positions, device=device, dtype=dtype)
        sources_t = torch.as_tensor(sources, device=device, dtype=dtype)
        if detectors_t.ndim != 2 or detectors_t.shape[1] != 3:
            raise ValueError("detector_positions must be shaped (N, 3).")
        if sources_t.ndim != 2 or sources_t.shape[1] != 3:
            raise ValueError("sources must be shaped (N, 3).")
        if detectors_t.shape[0] != sources_t.shape[0]:
            raise ValueError("detector_positions and sources must have the same row count.")
        row_count = int(sources_t.shape[0])
        if row_count == 0:
            return np.zeros(0, dtype=float)
        num_orients = int(len(self.orientations))
        fe_arr = np.asarray(fe_indices, dtype=int).reshape(-1) % num_orients
        pb_arr = np.asarray(pb_indices, dtype=int).reshape(-1) % num_orients
        if fe_arr.size != row_count or pb_arr.size != row_count:
            raise ValueError("Fe/Pb index arrays must match the detector-source row count.")
        unique_orients = np.unique(np.concatenate([fe_arr, pb_arr]))
        orient_to_row = {int(orient): idx for idx, orient in enumerate(unique_orients)}
        fe_select = torch.as_tensor(
            [orient_to_row[int(orient)] for orient in fe_arr],
            device=device,
            dtype=torch.long,
        )
        pb_select = torch.as_tensor(
            [orient_to_row[int(orient)] for orient in pb_arr],
            device=device,
            dtype=torch.long,
        )
        row_ids = torch.arange(row_count, device=device, dtype=torch.long)
        with torch.no_grad():
            direction = detectors_t - sources_t
            dist = torch.linalg.norm(direction, dim=1)
            tol_t = torch.as_tensor(tol, device=device, dtype=dtype)
            dist = torch.where(dist <= tol_t, tol_t, dist)
            geom = _finite_sphere_geometric_term_torch(
                dist,
                detector_radius_m=self.detector_radius_m,
                tol=tol_t,
            )
            sampled_sources, targets, sample_count = self._ray_sample_points_torch(
                sources=sources_t,
                detector=detectors_t,
                dist=dist,
                tol=tol,
            )
            sampled_direction = targets - sampled_sources
            sampled_dist = torch.linalg.norm(sampled_direction, dim=-1)
            sampled_dist = torch.where(sampled_dist <= tol_t, tol_t, sampled_dist)
            dir_unit = sampled_direction / sampled_dist.unsqueeze(-1)
            detector_to_source_unit = -dir_unit

            l_fe_by_orient: list[torch.Tensor] = []
            l_pb_by_orient: list[torch.Tensor] = []
            for orient_idx in unique_orients:
                orient_int = int(orient_idx)
                if (
                    self.shield_params.shield_geometry_model
                    == SHIELD_GEOMETRY_SPHERICAL_OCTANT
                    and not self.shield_params.use_angle_attenuation
                ):
                    shield_normal = -np.asarray(
                        self.orientations[orient_int],
                        dtype=float,
                    )
                    rotation = self._rotated_octant_rotation_torch(
                        shield_normal,
                        device=device,
                        dtype=dtype,
                    )
                    center = detectors_t.unsqueeze(-2)
                    l_fe = segment_rotated_octant_shell_path_length_cm_torch(
                        source_pos=sampled_sources,
                        target_pos=targets,
                        center_pos=center,
                        shield_normal=shield_normal,
                        inner_radius_cm=self.shield_params.inner_radius_fe_cm,
                        outer_radius_cm=(
                            self.shield_params.inner_radius_fe_cm
                            + self.shield_params.thickness_fe_cm
                        ),
                        tol=tol,
                        rotation=rotation,
                    )
                    l_pb = segment_rotated_octant_shell_path_length_cm_torch(
                        source_pos=sampled_sources,
                        target_pos=targets,
                        center_pos=center,
                        shield_normal=shield_normal,
                        inner_radius_cm=self.shield_params.inner_radius_pb_cm,
                        outer_radius_cm=(
                            self.shield_params.inner_radius_pb_cm
                            + self.shield_params.thickness_pb_cm
                        ),
                        tol=tol,
                        rotation=rotation,
                    )
                else:
                    blocked = self._rotated_octant_blocked_mask_torch(
                        detector_to_source_unit,
                        orient_int,
                        tol,
                    )
                    normal = torch.as_tensor(
                        self.orientations[orient_int],
                        device=device,
                        dtype=dtype,
                    )
                    cos_theta = torch.clamp(
                        torch.sum(dir_unit * normal, dim=-1),
                        0.0,
                        1.0,
                    )
                    l_fe, l_pb = self._shield_path_lengths_torch(
                        direction=sampled_direction,
                        blocked_fe=blocked,
                        blocked_pb=blocked,
                        cos_fe=cos_theta,
                        cos_pb=cos_theta,
                        tol_t=tol_t,
                        device=device,
                        dtype=dtype,
                    )
                l_fe_by_orient.append(l_fe)
                l_pb_by_orient.append(l_pb)

            l_fe_stack = torch.stack(l_fe_by_orient, dim=0)
            l_pb_stack = torch.stack(l_pb_by_orient, dim=0)
            l_fe_pairs = l_fe_stack[fe_select, row_ids, :]
            l_pb_pairs = l_pb_stack[pb_select, row_ids, :]

            mu_fe, mu_pb = self._mu_values(isotope=isotope)
            tau_fe = float(mu_fe) * l_fe_pairs
            tau_pb = float(mu_pb) * l_pb_pairs
            tau_obstacle = torch.zeros_like(tau_fe)
            obstacle_path_cm = None
            boxes_np = self.obstacle_boxes_m()
            if boxes_np.size:
                boxes_t = torch.as_tensor(boxes_np, device=device, dtype=dtype)
                obstacle_path_cm = obstacle_path_lengths_between_points_by_box_cm_torch(
                    source_pos=sampled_sources,
                    target_pos=targets,
                    obstacle_boxes_m=boxes_t,
                    tol=tol,
                )
                tau_obstacle = self._obstacle_tau_from_lengths_torch(
                    isotope,
                    obstacle_path_cm,
                    device=device,
                    dtype=dtype,
                )
            line_entries = self._line_mu_values(isotope)
            if line_entries:
                weights_t = torch.as_tensor(
                    [entry[0] for entry in line_entries],
                    device=device,
                    dtype=dtype,
                )
                mu_fe_t = torch.as_tensor(
                    [entry[1] for entry in line_entries],
                    device=device,
                    dtype=dtype,
                )
                mu_pb_t = torch.as_tensor(
                    [entry[2] for entry in line_entries],
                    device=device,
                    dtype=dtype,
                )
                line_tau_fe = l_fe_pairs.unsqueeze(-1) * mu_fe_t.view(1, 1, -1)
                line_tau_pb = l_pb_pairs.unsqueeze(-1) * mu_pb_t.view(1, 1, -1)
                line_tau_obstacle = tau_obstacle.unsqueeze(-1)
                if obstacle_path_cm is not None:
                    candidate_tau = self._obstacle_line_tau_from_lengths_torch(
                        isotope,
                        obstacle_path_cm,
                        line_count=len(line_entries),
                        device=device,
                        dtype=dtype,
                    )
                    if candidate_tau is not None:
                        line_tau_obstacle = candidate_tau
                line_total_tau = line_tau_fe + line_tau_pb + line_tau_obstacle
                line_buildup = self._buildup_factor_torch(
                    line_tau_fe,
                    line_tau_pb,
                    line_tau_obstacle,
                )
                shield_tau_feature = torch.sum(
                    (line_tau_fe + line_tau_pb) * weights_t.view(1, 1, -1),
                    dim=-1,
                )
                fe_tau_feature = torch.sum(
                    line_tau_fe * weights_t.view(1, 1, -1),
                    dim=-1,
                )
                pb_tau_feature = torch.sum(
                    line_tau_pb * weights_t.view(1, 1, -1),
                    dim=-1,
                )
                obstacle_tau_feature = torch.sum(
                    line_tau_obstacle * weights_t.view(1, 1, -1),
                    dim=-1,
                )
                response_factor = self._transport_response_factor_torch(
                    isotope,
                    fe_arr,
                    pb_arr,
                    shield_tau_feature,
                    obstacle_tau_feature,
                    fe_tau_feature,
                    pb_tau_feature,
                    distance_feature=dist.view(-1, 1),
                    distance_shield_feature=dist.view(-1, 1) * shield_tau_feature,
                    device=device,
                    dtype=dtype,
                )
                base_att = torch.sum(
                    torch.exp(-line_total_tau)
                    * line_buildup
                    * weights_t.view(1, 1, -1),
                    dim=-1,
                )
                att = self._response_adjusted_attenuation_torch(
                    isotope,
                    base_att,
                    response_factor,
                )
            else:
                total_tau = tau_fe + tau_pb + tau_obstacle
                buildup = self._buildup_factor_torch(tau_fe, tau_pb, tau_obstacle)
                response_factor = self._transport_response_factor_torch(
                    isotope,
                    fe_arr,
                    pb_arr,
                    tau_fe + tau_pb,
                    tau_obstacle,
                    tau_fe,
                    tau_pb,
                    distance_feature=dist.view(-1, 1),
                    distance_shield_feature=dist.view(-1, 1) * (tau_fe + tau_pb),
                    device=device,
                    dtype=dtype,
                )
                att = self._response_adjusted_attenuation_torch(
                    isotope,
                    torch.exp(-total_tau) * buildup,
                    response_factor,
                )
            att = torch.mean(att, dim=-1)
            values = geom * att
        return values.detach().cpu().numpy().astype(float, copy=False)

    def kernel_values_all_pairs(
        self,
        isotope: str,
        detector_pos: NDArray[np.float64],
        sources: NDArray[np.float64],
        chunk_size: int = 8192,
    ) -> NDArray[np.float64]:
        """Evaluate K values for every Fe/Pb orientation pair and source."""
        sources_arr = np.asarray(sources, dtype=float)
        num_orients = int(len(self.orientations))
        num_pairs = num_orients * num_orients
        if sources_arr.size == 0:
            return np.zeros((num_pairs, 0), dtype=float)
        if sources_arr.ndim != 2 or sources_arr.shape[1] != 3:
            raise ValueError("sources must be shaped (N, 3).")
        if not self.use_gpu:
            rows = [
                self.kernel_values_pair(
                    isotope=isotope,
                    detector_pos=detector_pos,
                    sources=sources_arr,
                    fe_index=fe_index,
                    pb_index=pb_index,
                    chunk_size=chunk_size,
                )
                for fe_index in range(num_orients)
                for pb_index in range(num_orients)
            ]
            return np.vstack(rows).astype(float, copy=False)
        self._gpu_enabled()
        chunk = self._adaptive_torch_chunk_size(chunk_size, all_orientation_pairs=True)
        parts: list[NDArray[np.float64]] = []
        for start in range(0, sources_arr.shape[0], chunk):
            stop = min(start + chunk, sources_arr.shape[0])
            parts.append(
                self._kernel_values_all_pairs_torch_chunk(
                    isotope=isotope,
                    detector_pos=detector_pos,
                    sources=sources_arr[start:stop],
                )
            )
        if not parts:
            return np.zeros((num_pairs, 0), dtype=float)
        return np.concatenate(parts, axis=1)

    def kernel_values_all_pairs_for_detectors(
        self,
        isotope: str,
        detector_positions: NDArray[np.float64],
        sources: NDArray[np.float64],
        chunk_size: int = 262144,
    ) -> NDArray[np.float64]:
        """Evaluate all Fe/Pb pair kernels for many detectors and sources."""
        detectors_arr = np.asarray(detector_positions, dtype=float)
        sources_arr = np.asarray(sources, dtype=float)
        num_orients = int(len(self.orientations))
        num_pairs = num_orients * num_orients
        if detectors_arr.size == 0 or sources_arr.size == 0:
            if detectors_arr.size == 0:
                pose_count = 0
            else:
                pose_count = int(detectors_arr.reshape(-1, 3).shape[0])
            return np.zeros((pose_count, num_pairs, 0), dtype=float)
        if detectors_arr.ndim != 2 or detectors_arr.shape[1] != 3:
            raise ValueError("detector_positions must be shaped (P, 3).")
        if sources_arr.ndim != 2 or sources_arr.shape[1] != 3:
            raise ValueError("sources must be shaped (S, 3).")
        pose_count = int(detectors_arr.shape[0])
        source_count = int(sources_arr.shape[0])
        if not self.use_gpu:
            return np.stack(
                [
                    self.kernel_values_all_pairs(
                        isotope=isotope,
                        detector_pos=detector,
                        sources=sources_arr,
                        chunk_size=chunk_size,
                    )
                    for detector in detectors_arr
                ],
                axis=0,
            ).astype(float, copy=False)
        self._gpu_enabled()
        detectors_flat = np.repeat(detectors_arr, source_count, axis=0)
        sources_flat = np.tile(sources_arr, (pose_count, 1))
        chunk = self._adaptive_torch_chunk_size(chunk_size, all_orientation_pairs=True)
        parts: list[NDArray[np.float64]] = []
        for start in range(0, sources_flat.shape[0], chunk):
            stop = min(start + chunk, sources_flat.shape[0])
            parts.append(
                self._kernel_values_all_pairs_for_detector_source_torch_chunk(
                    isotope=isotope,
                    detector_positions=detectors_flat[start:stop],
                    sources=sources_flat[start:stop],
                )
            )
        if not parts:
            return np.zeros((pose_count, num_pairs, source_count), dtype=float)
        flat_values = np.concatenate(parts, axis=0)
        return flat_values.reshape(pose_count, source_count, num_pairs).transpose(
            0,
            2,
            1,
        )

    def kernel_values_selected_pairs_for_detectors(
        self,
        isotope: str,
        detector_positions: NDArray[np.float64],
        sources: NDArray[np.float64],
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
        chunk_size: int = 262144,
    ) -> NDArray[np.float64]:
        """Evaluate one selected Fe/Pb pair per detector for many sources."""
        detectors_arr = np.asarray(detector_positions, dtype=float)
        sources_arr = np.asarray(sources, dtype=float)
        if detectors_arr.ndim != 2 or detectors_arr.shape[1] != 3:
            raise ValueError("detector_positions must be shaped (P, 3).")
        if sources_arr.ndim != 2 or sources_arr.shape[1] != 3:
            raise ValueError("sources must be shaped (S, 3).")
        pose_count = int(detectors_arr.shape[0])
        source_count = int(sources_arr.shape[0])
        fe_arr = np.asarray(fe_indices, dtype=int).reshape(-1)
        pb_arr = np.asarray(pb_indices, dtype=int).reshape(-1)
        if fe_arr.size != pose_count or pb_arr.size != pose_count:
            raise ValueError("Fe/Pb index arrays must match detector count.")
        if pose_count == 0 or source_count == 0:
            return np.zeros((pose_count, source_count), dtype=float)
        if not self.use_gpu:
            return np.vstack(
                [
                    self.kernel_values_pair(
                        isotope=isotope,
                        detector_pos=detector,
                        sources=sources_arr,
                        fe_index=int(fe_idx),
                        pb_index=int(pb_idx),
                    )
                    for detector, fe_idx, pb_idx in zip(detectors_arr, fe_arr, pb_arr)
                ]
            ).astype(float, copy=False)
        self._gpu_enabled()
        detectors_flat = np.repeat(detectors_arr, source_count, axis=0)
        sources_flat = np.tile(sources_arr, (pose_count, 1))
        fe_flat = np.repeat(fe_arr, source_count)
        pb_flat = np.repeat(pb_arr, source_count)
        chunk = self._adaptive_torch_chunk_size(
            chunk_size,
            all_orientation_pairs=False,
        )
        parts: list[NDArray[np.float64]] = []
        for start in range(0, sources_flat.shape[0], chunk):
            stop = min(start + chunk, sources_flat.shape[0])
            parts.append(
                self._kernel_values_selected_pairs_for_detector_source_torch_chunk(
                    isotope=isotope,
                    detector_positions=detectors_flat[start:stop],
                    sources=sources_flat[start:stop],
                    fe_indices=fe_flat[start:stop],
                    pb_indices=pb_flat[start:stop],
                )
            )
        if not parts:
            return np.zeros((pose_count, source_count), dtype=float)
        return np.concatenate(parts).reshape(pose_count, source_count)

    def kernel_values_pair(
        self,
        isotope: str,
        detector_pos: NDArray[np.float64],
        sources: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        chunk_size: int = 8192,
    ) -> NDArray[np.float64]:
        """Evaluate K values for many sources at one detector pose."""
        sources_arr = np.asarray(sources, dtype=float)
        if sources_arr.size == 0:
            return np.zeros(0, dtype=float)
        if sources_arr.ndim != 2 or sources_arr.shape[1] != 3:
            raise ValueError("sources must be shaped (N, 3).")
        if not self.use_gpu:
            return np.asarray(
                [
                    self.kernel_value_pair(
                        isotope=isotope,
                        detector_pos=detector_pos,
                        source_pos=source,
                        fe_index=fe_index,
                        pb_index=pb_index,
                    )
                    for source in sources_arr
                ],
                dtype=float,
            )
        self._gpu_enabled()
        chunk = self._adaptive_torch_chunk_size(chunk_size, all_orientation_pairs=False)
        parts: list[NDArray[np.float64]] = []
        for start in range(0, sources_arr.shape[0], chunk):
            stop = min(start + chunk, sources_arr.shape[0])
            parts.append(
                self._kernel_values_pair_torch_chunk(
                    isotope=isotope,
                    detector_pos=detector_pos,
                    sources=sources_arr[start:stop],
                    fe_index=fe_index,
                    pb_index=pb_index,
                )
            )
        return np.concatenate(parts) if parts else np.zeros(0, dtype=float)

    def attenuation_factor(
        self,
        isotope: str,
        source_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
        orient_idx: int,
    ) -> float:
        """
        Return attenuation factor A^{sh} (Sec. 3.2) for a single orientation.

        This treats Fe and Pb shells as sharing the same orientation index.
        """
        return self.attenuation_factor_pair(
            isotope=isotope,
            source_pos=source_pos,
            detector_pos=detector_pos,
            fe_index=orient_idx,
            pb_index=orient_idx,
        )

    def attenuation_factor_pair(
        self,
        isotope: str,
        source_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
    ) -> float:
        """Return combined Fe/Pb attenuation factor A^{sh} (Sec. 3.2)."""
        sampled_sources, targets = self._ray_sample_points(source_pos, detector_pos)
        values = [
            self._attenuation_factor_for_target(
                isotope=isotope,
                source_pos=sampled_source,
                target_pos=target,
                detector_pos=detector_pos,
                fe_index=fe_index,
                pb_index=pb_index,
            )
            for sampled_source, target in zip(sampled_sources, targets)
        ]
        return float(np.mean(values)) if values else 1.0

    def transport_response_terms_pair(
        self,
        isotope: str,
        detector_pos: NDArray[np.float64],
        source_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
    ) -> list[dict[str, float]]:
        """
        Return per-aperture transport-response basis terms for calibration.

        Runtime expected counts apply the optional transport-response factor
        per detector-aperture ray before averaging.  Calibration and validation
        must use the same feature granularity; otherwise a fitted response model
        is trained on center-ray features but evaluated on aperture-ray
        features.  This diagnostic helper mirrors the CPU attenuation kernel.
        Its ``kernel`` values sum to ``kernel_value_pair`` for this kernel, and
        its ``base_kernel`` values sum to the capped base kernel before applying
        any configured transport-response factor.
        """
        geom = finite_sphere_geometric_term(
            detector_pos,
            source_pos,
            self.detector_radius_m,
        )
        sampled_sources, targets = self._ray_sample_points(source_pos, detector_pos)
        if len(targets) == 0:
            return []
        sample_scale = float(geom) / float(len(targets))
        return [
            self._transport_response_term_for_target(
                isotope=isotope,
                source_pos=sampled_source,
                target_pos=target,
                detector_pos=detector_pos,
                fe_index=fe_index,
                pb_index=pb_index,
                sample_scale=sample_scale,
            )
            for sampled_source, target in zip(sampled_sources, targets)
        ]

    def _transport_response_term_for_target(
        self,
        isotope: str,
        source_pos: NDArray[np.float64],
        target_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        sample_scale: float,
    ) -> dict[str, float]:
        """Return one aperture-ray term matching the CPU attenuation kernel."""
        mu_fe, mu_pb = self._mu_values(isotope=isotope)
        normal_fe = self.orientations[fe_index]
        normal_pb = self.orientations[pb_index]
        detector_to_source = (
            np.asarray(source_pos, dtype=float) - np.asarray(target_pos, dtype=float)
        )
        distance_feature = float(
            np.linalg.norm(
                np.asarray(source_pos, dtype=float)
                - np.asarray(detector_pos, dtype=float)
            )
        )
        blocked_fe = rotated_positive_octant_blocks_direction(
            detector_to_source,
            -normal_fe,
        )
        blocked_pb = rotated_positive_octant_blocks_direction(
            detector_to_source,
            -normal_pb,
        )
        l_fe = self._shield_segment_path_length_cm(
            source_pos=source_pos,
            target_pos=target_pos,
            detector_center=detector_pos,
            incoming_normal=normal_fe,
            physical_normal=-normal_fe,
            thickness_cm=self.shield_params.thickness_fe_cm,
            inner_radius_cm=self.shield_params.inner_radius_fe_cm,
            blocked=blocked_fe,
        )
        l_pb = self._shield_segment_path_length_cm(
            source_pos=source_pos,
            target_pos=target_pos,
            detector_center=detector_pos,
            incoming_normal=normal_pb,
            physical_normal=-normal_pb,
            thickness_cm=self.shield_params.thickness_pb_cm,
            inner_radius_cm=self.shield_params.inner_radius_pb_cm,
            blocked=blocked_pb,
        )
        obstacle_path_by_box = np.zeros(0, dtype=float)
        tau_obstacle = 0.0
        if self.obstacle_grid is not None:
            obstacle_path_by_box = obstacle_path_lengths_by_box_cm(
                source_pos=source_pos,
                detector_pos=target_pos,
                obstacle_boxes_m=self.obstacle_boxes_m(),
            )
            tau_obstacle = float(
                np.sum(self.obstacle_mu_values_cm_inv(isotope) * obstacle_path_by_box)
            )
        line_entries = self._line_mu_values(isotope)
        if line_entries:
            line_obstacle_tau = self._line_obstacle_tau_values(
                isotope,
                obstacle_path_by_box,
                line_count=len(line_entries),
            )
            attenuation = 0.0
            shield_tau_feature = 0.0
            fe_tau_feature = 0.0
            pb_tau_feature = 0.0
            obstacle_tau_feature = 0.0
            for line_index, (weight, line_mu_fe, line_mu_pb) in enumerate(line_entries):
                tau_fe = float(line_mu_fe * l_fe)
                tau_pb = float(line_mu_pb * l_pb)
                tau_obs_line = (
                    tau_obstacle
                    if line_obstacle_tau is None
                    else float(line_obstacle_tau[line_index])
                )
                buildup = self._buildup_factor(tau_fe, tau_pb, tau_obs_line)
                attenuation += (
                    float(weight)
                    * float(np.exp(-(tau_fe + tau_pb + tau_obs_line)))
                    * buildup
                )
                shield_tau_feature += float(weight) * (tau_fe + tau_pb)
                fe_tau_feature += float(weight) * tau_fe
                pb_tau_feature += float(weight) * tau_pb
                obstacle_tau_feature += float(weight) * tau_obs_line
        else:
            tau_fe = float(mu_fe * l_fe)
            tau_pb = float(mu_pb * l_pb)
            buildup = self._buildup_factor(tau_fe, tau_pb, tau_obstacle)
            attenuation = float(np.exp(-(tau_fe + tau_pb + tau_obstacle))) * buildup
            shield_tau_feature = tau_fe + tau_pb
            fe_tau_feature = tau_fe
            pb_tau_feature = tau_pb
            obstacle_tau_feature = tau_obstacle
        (
            shield_cap,
            obstacle_cap,
            fe_cap,
            pb_cap,
            distance_shield_cap,
            distance_fe_cap,
            distance_pb_cap,
            distance_obstacle_cap,
        ) = self._transport_response_feature_caps(isotope)
        distance_shield_feature = (
            max(distance_feature, 0.0) * max(shield_tau_feature, 0.0)
        )
        distance_fe_feature = max(distance_feature, 0.0) * max(fe_tau_feature, 0.0)
        distance_pb_feature = max(distance_feature, 0.0) * max(pb_tau_feature, 0.0)
        distance_obstacle_feature = (
            max(distance_feature, 0.0) * max(obstacle_tau_feature, 0.0)
        )
        shield_tau_capped = self._capped_transport_feature(
            shield_tau_feature,
            shield_cap,
        )
        fe_tau_capped = self._capped_transport_feature(fe_tau_feature, fe_cap)
        pb_tau_capped = self._capped_transport_feature(pb_tau_feature, pb_cap)
        obstacle_tau_capped = self._capped_transport_feature(
            obstacle_tau_feature,
            obstacle_cap,
        )
        distance_shield_capped = self._capped_transport_feature(
            distance_shield_feature,
            distance_shield_cap,
        )
        distance_fe_capped = self._capped_transport_feature(
            distance_fe_feature,
            distance_fe_cap,
        )
        distance_pb_capped = self._capped_transport_feature(
            distance_pb_feature,
            distance_pb_cap,
        )
        distance_obstacle_capped = self._capped_transport_feature(
            distance_obstacle_feature,
            distance_obstacle_cap,
        )
        response_factor = self._transport_response_factor(
            isotope,
            fe_index,
            pb_index,
            shield_tau_feature,
            obstacle_tau_feature,
            fe_tau_feature,
            pb_tau_feature,
            distance_feature,
            distance_shield_feature,
        )
        base_attenuation = self._response_adjusted_attenuation(
            isotope,
            attenuation,
            1.0,
        )
        adjusted_attenuation = self._response_adjusted_attenuation(
            isotope,
            attenuation,
            response_factor,
        )
        uncapped_kernel = float(sample_scale) * float(attenuation)
        base_kernel = float(sample_scale) * float(base_attenuation)
        kernel = float(sample_scale) * adjusted_attenuation
        return {
            "kernel": kernel,
            "base_kernel": base_kernel,
            "uncapped_kernel": uncapped_kernel,
            "response_factor": float(response_factor),
            "shield_tau_feature": float(max(shield_tau_feature, 0.0)),
            "fe_tau_feature": float(max(fe_tau_feature, 0.0)),
            "pb_tau_feature": float(max(pb_tau_feature, 0.0)),
            "obstacle_tau_feature": float(max(obstacle_tau_feature, 0.0)),
            "distance_feature": float(max(distance_feature, 0.0)),
            "distance_shield_feature": float(distance_shield_feature),
            "distance_fe_feature": float(distance_fe_feature),
            "distance_pb_feature": float(distance_pb_feature),
            "distance_obstacle_feature": float(distance_obstacle_feature),
            "shield_tau_feature_capped": float(shield_tau_capped),
            "fe_tau_feature_capped": float(fe_tau_capped),
            "pb_tau_feature_capped": float(pb_tau_capped),
            "obstacle_tau_feature_capped": float(obstacle_tau_capped),
            "distance_shield_feature_capped": float(distance_shield_capped),
            "distance_fe_feature_capped": float(distance_fe_capped),
            "distance_pb_feature_capped": float(distance_pb_capped),
            "distance_obstacle_feature_capped": float(distance_obstacle_capped),
            "sample_scale": float(sample_scale),
        }

    def kernel_value(
        self,
        isotope: str,
        detector_pos: NDArray[np.float64],
        source_pos: NDArray[np.float64],
        orient_idx: int,
    ) -> float:
        """
        Evaluate K_{k,j,h} = G_{k,j} * A^{sh}_{k,j,h} (Eq. 3.11).
        """
        geom = finite_sphere_geometric_term(
            detector_pos,
            source_pos,
            self.detector_radius_m,
        )
        att = self.attenuation_factor(isotope, source_pos, detector_pos, orient_idx)
        return geom * att

    def kernel_value_pair(
        self,
        isotope: str,
        detector_pos: NDArray[np.float64],
        source_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
    ) -> float:
        """Evaluate K_{k,j,h}(R_Fe, R_Pb) for a Fe/Pb orientation pair."""
        geom = finite_sphere_geometric_term(
            detector_pos,
            source_pos,
            self.detector_radius_m,
        )
        att = self.attenuation_factor_pair(isotope, source_pos, detector_pos, fe_index, pb_index)
        return geom * att

    def expected_rate(
        self,
        isotope: str,
        detector_pos: NDArray[np.float64],
        sources: NDArray[np.float64],
        strengths: NDArray[np.float64],
        orient_idx: int,
        background: float = 0.0,
    ) -> float:
        """
        Compute λ_{k,h} = b_h + Σ_j K_{k,j,h} q_{h,j} (Eq. 3.12).
        """
        return self.expected_rate_pair(
            isotope=isotope,
            detector_pos=detector_pos,
            sources=sources,
            strengths=strengths,
            fe_index=orient_idx,
            pb_index=orient_idx,
            background=background,
        )

    def expected_rate_pair(
        self,
        isotope: str,
        detector_pos: NDArray[np.float64],
        sources: NDArray[np.float64],
        strengths: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        background: float = 0.0,
    ) -> float:
        """
        Compute λ_{k,h} for a Fe/Pb orientation pair (Eq. 3.41 with separate R_Fe, R_Pb).
        """
        if not self.use_gpu:
            rate = float(background)
            sources_arr = np.asarray(sources, dtype=float)
            strengths_arr = np.asarray(strengths, dtype=float)
            if sources_arr.size == 0:
                return rate
            for source_pos, strength in zip(sources_arr, strengths_arr):
                rate += float(strength) * self.kernel_value_pair(
                    isotope=isotope,
                    detector_pos=detector_pos,
                    source_pos=source_pos,
                    fe_index=fe_index,
                    pb_index=pb_index,
                )
            return float(rate)
        self._gpu_enabled()
        return self._expected_rate_pair_torch(
            isotope=isotope,
            detector_pos=detector_pos,
            sources=sources,
            strengths=strengths,
            fe_index=fe_index,
            pb_index=pb_index,
            background=background,
        )

    def expected_counts(
        self,
        isotope: str,
        detector_pos: NDArray[np.float64],
        sources: NDArray[np.float64],
        strengths: NDArray[np.float64],
        orient_idx: int,
        live_time_s: float = 1.0,
        background: float = 0.0,
    ) -> float:
        """
        Compute Λ_{k,h} = T_k λ_{k,h} (Eq. 3.13).
        """
        rate = self.expected_rate(isotope, detector_pos, sources, strengths, orient_idx, background=background)
        return float(live_time_s * rate)

    def expected_counts_pair(
        self,
        isotope: str,
        detector_pos: NDArray[np.float64],
        sources: NDArray[np.float64],
        strengths: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        live_time_s: float = 1.0,
        background: float = 0.0,
    ) -> float:
        """
        Compute Λ_{k,h}(R_Fe, R_Pb) per Eq. (3.41) using octant indices for Fe/Pb.
        """
        rate = self.expected_rate_pair(
            isotope=isotope,
            detector_pos=detector_pos,
            sources=sources,
            strengths=strengths,
            fe_index=fe_index,
            pb_index=pb_index,
            background=background,
        )
        return float(live_time_s * rate)

    def expected_counts_pair_for_packed_states_torch(
        self,
        *,
        isotope: str,
        detector_pos: NDArray[np.float64],
        positions: "torch.Tensor",
        strengths: "torch.Tensor",
        backgrounds: "torch.Tensor",
        mask: "torch.Tensor",
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        source_scale: float | NDArray[np.float64] | "torch.Tensor" = 1.0,
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> "torch.Tensor":
        """Compute packed-state pair counts through ContinuousKernel GPU math."""
        counts = self._expected_counts_selected_pairs_for_packed_states_torch(
            isotope=isotope,
            detector_pos=np.asarray(detector_pos, dtype=float),
            positions=positions,
            strengths=strengths,
            backgrounds=backgrounds,
            mask=mask,
            fe_indices=np.asarray([int(fe_index)], dtype=np.int64),
            pb_indices=np.asarray([int(pb_index)], dtype=np.int64),
            live_time_s=float(live_time_s),
            source_scale=source_scale,
            device=device,
            dtype=dtype,
        )
        return counts[0]

    def expected_counts_all_pairs_for_packed_states_torch(
        self,
        *,
        isotope: str,
        detector_pos: NDArray[np.float64],
        positions: "torch.Tensor",
        strengths: "torch.Tensor",
        backgrounds: "torch.Tensor",
        mask: "torch.Tensor",
        live_time_s: float,
        source_scale: float | NDArray[np.float64] | "torch.Tensor" = 1.0,
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> "torch.Tensor":
        """Compute all Fe/Pb-pair counts through ContinuousKernel GPU math."""
        num_orients = int(len(self.orientations))
        pair_ids = np.arange(num_orients * num_orients, dtype=np.int64)
        fe_indices = pair_ids // num_orients
        pb_indices = pair_ids % num_orients
        return self._expected_counts_selected_pairs_for_packed_states_torch(
            isotope=isotope,
            detector_pos=np.asarray(detector_pos, dtype=float),
            positions=positions,
            strengths=strengths,
            backgrounds=backgrounds,
            mask=mask,
            fe_indices=fe_indices,
            pb_indices=pb_indices,
            live_time_s=float(live_time_s),
            source_scale=source_scale,
            device=device,
            dtype=dtype,
        )

    def expected_counts_selected_pairs_for_packed_states_torch(
        self,
        *,
        isotope: str,
        detector_pos: NDArray[np.float64],
        positions: "torch.Tensor",
        strengths: "torch.Tensor",
        backgrounds: "torch.Tensor",
        mask: "torch.Tensor",
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
        live_time_s: float,
        source_scale: float | NDArray[np.float64] | "torch.Tensor" = 1.0,
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> "torch.Tensor":
        """Compute selected Fe/Pb-pair counts through ContinuousKernel GPU math."""
        return self._expected_counts_selected_pairs_for_packed_states_torch(
            isotope=isotope,
            detector_pos=np.asarray(detector_pos, dtype=float),
            positions=positions,
            strengths=strengths,
            backgrounds=backgrounds,
            mask=mask,
            fe_indices=np.asarray(fe_indices, dtype=int),
            pb_indices=np.asarray(pb_indices, dtype=int),
            live_time_s=float(live_time_s),
            source_scale=source_scale,
            device=device,
            dtype=dtype,
        )

    def orient_index_from_vector(self, orientation: NDArray[np.float64]) -> int:
        """Map an orientation vector to the closest octant index."""
        return octant_index_from_normal(orientation)


def expected_counts_single_isotope(
    detector_position: NDArray[np.float64],
    RFe: NDArray[np.float64],
    RPb: NDArray[np.float64],
    sources: NDArray[np.float64],
    strengths: NDArray[np.float64],
    background: float,
    duration: float,
    isotope_id: str | None = None,
    kernel: ContinuousKernel | None = None,
    mu_by_isotope: Dict[str, object] | None = None,
    shield_params: ShieldParams | None = None,
    use_gpu: bool | None = None,
    gpu_device: str = "cuda",
    gpu_dtype: str = "float32",
) -> float:
    """
    Continuous expected counts Λ_{k,h} for a single isotope and time step (Sec. 3.2–3.3).

    RFe / RPb are interpreted as orientation matrices; the third column is used as the
    shield normal. If a 3-vector is passed, it is used directly.
    mu_by_isotope and shield_params are used only when a kernel is not provided.
    use_gpu controls optional CUDA acceleration for batch kernel evaluation.
    """
    if kernel is None:
        k = ContinuousKernel(
            mu_by_isotope=mu_by_isotope,
            shield_params=shield_params or ShieldParams(),
            use_gpu=bool(use_gpu) if use_gpu is not None else False,
            gpu_device=gpu_device,
            gpu_dtype=gpu_dtype,
        )
    else:
        k = kernel

    def _normal_from_R(R: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return a shield normal from either a vector or rotation matrix."""
        if R.ndim == 1:
            return np.asarray(R, dtype=float)
        if R.shape == (3, 3):
            return np.asarray(R[:, 2], dtype=float)
        raise ValueError("RFe/RPb must be shape (3,) or (3,3)")

    n_fe = _normal_from_R(RFe)
    n_pb = _normal_from_R(RPb)
    idx_fe = k.orient_index_from_vector(n_fe)
    idx_pb = k.orient_index_from_vector(n_pb)

    return k.expected_counts_pair(
        isotope=isotope_id or "generic",
        detector_pos=detector_position,
        sources=sources,
        strengths=strengths,
        fe_index=idx_fe,
        pb_index=idx_pb,
        live_time_s=duration,
        background=background,
    )
