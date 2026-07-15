"""Geometry helpers for stage-backed radiation attenuation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OrientedBox:
    """Represent an oriented box in world coordinates."""

    center_xyz: tuple[float, float, float]
    size_xyz: tuple[float, float, float]
    rotation_matrix: np.ndarray


@dataclass(frozen=True)
class Sphere:
    """Represent a sphere in world coordinates."""

    center_xyz: tuple[float, float, float]
    radius_m: float


@dataclass(frozen=True)
class TriangleMesh:
    """Represent a triangle mesh in world coordinates."""

    triangles_xyz: tuple[
        tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
        ...,
    ]


def quaternion_wxyz_to_matrix(
    quaternion_wxyz: tuple[float, float, float, float],
) -> np.ndarray:
    """Convert a quaternion into a 3x3 rotation matrix."""
    w, x, y, z = (float(v) for v in quaternion_wxyz)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        return np.eye(3, dtype=float)
    w /= norm
    x /= norm
    y /= norm
    z /= norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def segment_path_length_through_box(
    start_xyz: tuple[float, float, float] | np.ndarray,
    end_xyz: tuple[float, float, float] | np.ndarray,
    box: OrientedBox,
    *,
    epsilon: float = 1e-9,
) -> float:
    """Return the segment path length inside an oriented box."""
    start = np.asarray(start_xyz, dtype=float)
    end = np.asarray(end_xyz, dtype=float)
    center = np.asarray(box.center_xyz, dtype=float)
    rotation = np.asarray(box.rotation_matrix, dtype=float)
    half_extents = 0.5 * np.asarray(box.size_xyz, dtype=float)

    local_start = rotation.T @ (start - center)
    local_end = rotation.T @ (end - center)
    direction = local_end - local_start

    t_min = 0.0
    t_max = 1.0
    for axis in range(3):
        origin = float(local_start[axis])
        delta = float(direction[axis])
        min_bound = -float(half_extents[axis])
        max_bound = float(half_extents[axis])
        if abs(delta) <= epsilon:
            if origin < min_bound or origin > max_bound:
                return 0.0
            continue
        inv_delta = 1.0 / delta
        t0 = (min_bound - origin) * inv_delta
        t1 = (max_bound - origin) * inv_delta
        if t0 > t1:
            t0, t1 = t1, t0
        t_min = max(t_min, t0)
        t_max = min(t_max, t1)
        if t_max <= t_min:
            return 0.0

    segment_length = float(np.linalg.norm(end - start))
    return max(0.0, t_max - t_min) * segment_length


def segment_path_length_through_sphere(
    start_xyz: tuple[float, float, float] | np.ndarray,
    end_xyz: tuple[float, float, float] | np.ndarray,
    sphere: Sphere,
    *,
    epsilon: float = 1e-9,
) -> float:
    """Return the segment path length inside a sphere."""
    start = np.asarray(start_xyz, dtype=float)
    end = np.asarray(end_xyz, dtype=float)
    center = np.asarray(sphere.center_xyz, dtype=float)
    direction = end - start
    segment_length = float(np.linalg.norm(direction))
    if segment_length <= epsilon:
        return 0.0
    direction_unit = direction / segment_length
    offset = start - center
    a = 1.0
    b = 2.0 * float(np.dot(direction_unit, offset))
    c = float(np.dot(offset, offset) - sphere.radius_m**2)
    discriminant = b * b - 4.0 * a * c
    if discriminant <= 0.0:
        return 0.0
    sqrt_disc = float(np.sqrt(discriminant))
    t0 = (-b - sqrt_disc) / (2.0 * a)
    t1 = (-b + sqrt_disc) / (2.0 * a)
    enter = max(0.0, min(t0, t1))
    exit = min(segment_length, max(t0, t1))
    if exit <= enter:
        return 0.0
    return exit - enter


def segment_path_length_through_mesh(
    start_xyz: tuple[float, float, float] | np.ndarray,
    end_xyz: tuple[float, float, float] | np.ndarray,
    mesh: TriangleMesh,
    *,
    epsilon: float = 1e-9,
) -> float:
    """Return the segment path length through a closed triangle mesh."""
    start = np.asarray(start_xyz, dtype=float)
    end = np.asarray(end_xyz, dtype=float)
    direction = end - start
    segment_length = float(np.linalg.norm(direction))
    if segment_length <= epsilon:
        return 0.0

    t_values: list[float] = []
    for triangle in mesh.triangles_xyz:
        t_hit = _segment_triangle_intersection_t(start, end, triangle, epsilon=epsilon)
        if t_hit is not None:
            t_values.append(float(t_hit))

    if not t_values:
        return 0.0
    t_values.sort()
    unique_t_values: list[float] = []
    for value in t_values:
        if not unique_t_values or abs(value - unique_t_values[-1]) > 1e-6:
            unique_t_values.append(value)
    if len(unique_t_values) < 2:
        return 0.0

    total_fraction = 0.0
    for index in range(0, len(unique_t_values) - 1, 2):
        enter = unique_t_values[index]
        exit = unique_t_values[index + 1]
        if exit > enter:
            total_fraction += exit - enter
    return total_fraction * segment_length


def _segment_triangle_intersection_t(
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    triangle_xyz: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    *,
    epsilon: float = 1e-9,
) -> float | None:
    """Return the normalized segment parameter when a triangle intersection exists."""
    vertex0 = np.asarray(triangle_xyz[0], dtype=float)
    vertex1 = np.asarray(triangle_xyz[1], dtype=float)
    vertex2 = np.asarray(triangle_xyz[2], dtype=float)
    direction = end_xyz - start_xyz
    edge1 = vertex1 - vertex0
    edge2 = vertex2 - vertex0
    pvec = np.cross(direction, edge2)
    det = float(np.dot(edge1, pvec))
    if abs(det) <= epsilon:
        return None
    inv_det = 1.0 / det
    tvec = start_xyz - vertex0
    u = float(np.dot(tvec, pvec) * inv_det)
    if u < -epsilon or u > 1.0 + epsilon:
        return None
    qvec = np.cross(tvec, edge1)
    v = float(np.dot(direction, qvec) * inv_det)
    if v < -epsilon or (u + v) > 1.0 + epsilon:
        return None
    t_hit = float(np.dot(edge2, qvec) * inv_det)
    if t_hit < -epsilon or t_hit > 1.0 + epsilon:
        return None
    return float(np.clip(t_hit, 0.0, 1.0))
