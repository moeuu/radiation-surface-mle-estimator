"""Exact rectangular room and obstacle patches for surface MLE.

Patch construction is vectorized within each physical face.  The resulting
graph connects every pair of patches that shares a non-zero physical edge,
including perpendicular room faces and coarse/fine neighbors after selective
refinement.  Graph weights are shared-edge lengths in metres.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid

from .types import SurfaceKind, SurfacePatch, SurfacePatchSet


_GEOMETRY_TOLERANCE_M = 1.0e-10


def _normalize_spacing(spacing: float | Sequence[float]) -> tuple[float, float, float]:
    """Return a finite positive x/y/z target spacing."""
    values = np.asarray(spacing, dtype=float).reshape(-1)
    if values.size == 1:
        values = np.repeat(values, 3)
    if values.shape != (3,) or np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("spacing must be a finite positive scalar or 3-vector.")
    return float(values[0]), float(values[1]), float(values[2])


def _axis_edges(
    length_m: float,
    target_spacing_m: float,
    *,
    extra_edges_m: Iterable[float] = (),
) -> NDArray[np.float64]:
    """Return exact interval edges, adding geometry boundaries when requested."""
    length = float(length_m)
    spacing = float(target_spacing_m)
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("Surface axis lengths must be finite and positive.")
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Target patch spacing must be finite and positive.")
    cell_count = max(1, int(np.ceil(length / spacing)))
    base = np.linspace(0.0, length, cell_count + 1, dtype=float)
    extras = np.asarray(tuple(extra_edges_m), dtype=float).reshape(-1)
    if extras.size:
        if np.any(~np.isfinite(extras)):
            raise ValueError("Extra geometry edges must be finite.")
        if np.any(extras < -_GEOMETRY_TOLERANCE_M) or np.any(
            extras > length + _GEOMETRY_TOLERANCE_M
        ):
            raise ValueError("Extra geometry edges must lie within the surface axis.")
        extras = np.clip(extras, 0.0, length)
        base = np.concatenate([base, extras])
    edges = np.unique(np.round(base, decimals=12))
    edges[0] = 0.0
    edges[-1] = length
    if np.any(np.diff(edges) <= _GEOMETRY_TOLERANCE_M):
        raise ValueError("Surface edge construction produced a degenerate interval.")
    return edges.astype(float, copy=False)


def _quadrature_from_vertices(
    vertices_xyz: NDArray[np.float64],
    point_count: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return normalized centroid or tensor Gauss quadrature for rectangles."""
    vertices = np.asarray(vertices_xyz, dtype=float)
    if vertices.ndim != 3 or vertices.shape[1:] != (4, 3):
        raise ValueError("vertices_xyz must have shape (P, 4, 3).")
    count = int(point_count)
    p00 = vertices[:, 0]
    u_vector = vertices[:, 1] - p00
    v_vector = vertices[:, 3] - p00
    if count == 1:
        points = p00[:, None, :] + 0.5 * (
            u_vector[:, None, :] + v_vector[:, None, :]
        )
        return points, np.ones((vertices.shape[0], 1), dtype=float)
    if count != 4:
        raise ValueError("quadrature_points_per_patch must be 1 or 4.")
    offset = 1.0 / (2.0 * np.sqrt(3.0))
    low = 0.5 - offset
    high = 0.5 + offset
    coordinates = np.asarray(
        [[low, low], [high, low], [high, high], [low, high]],
        dtype=float,
    )
    points = (
        p00[:, None, :]
        + coordinates[None, :, 0, None] * u_vector[:, None, :]
        + coordinates[None, :, 1, None] * v_vector[:, None, :]
    )
    weights = np.full((vertices.shape[0], 4), 0.25, dtype=float)
    return points, weights


def _face_patches(
    *,
    patch_id_start: int,
    origin_xyz: Sequence[float],
    u_axis_xyz: Sequence[float],
    v_axis_xyz: Sequence[float],
    u_edges_m: NDArray[np.float64],
    v_edges_m: NDArray[np.float64],
    normal_xyz: Sequence[float],
    surface_kind: SurfaceKind,
    object_id: str,
    quadrature_points_per_patch: int,
    keep_mask: NDArray[np.bool_] | None = None,
    parent_patch_id: int | None = None,
    refinement_level: int = 0,
) -> list[SurfacePatch]:
    """Build all exact rectangles on one face in vectorized arrays."""
    origin = np.asarray(origin_xyz, dtype=float).reshape(3)
    u_axis = np.asarray(u_axis_xyz, dtype=float).reshape(3)
    v_axis = np.asarray(v_axis_xyz, dtype=float).reshape(3)
    normal = np.asarray(normal_xyz, dtype=float).reshape(3)
    u_edges = np.asarray(u_edges_m, dtype=float).reshape(-1)
    v_edges = np.asarray(v_edges_m, dtype=float).reshape(-1)
    if u_edges.size < 2 or v_edges.size < 2:
        raise ValueError("Each face axis needs at least one interval.")
    if not np.isclose(np.linalg.norm(u_axis), 1.0, atol=1.0e-12) or not np.isclose(
        np.linalg.norm(v_axis), 1.0, atol=1.0e-12
    ):
        raise ValueError("Face axes must be unit vectors.")
    if not np.allclose(np.cross(u_axis, v_axis), normal, atol=1.0e-12):
        raise ValueError("Face axes and normal must form an oriented orthonormal frame.")

    u0, v0 = np.meshgrid(u_edges[:-1], v_edges[:-1], indexing="ij")
    u1, v1 = np.meshgrid(u_edges[1:], v_edges[1:], indexing="ij")
    u0 = u0.reshape(-1)
    v0 = v0.reshape(-1)
    u1 = u1.reshape(-1)
    v1 = v1.reshape(-1)
    p00 = origin + u0[:, None] * u_axis + v0[:, None] * v_axis
    p10 = origin + u1[:, None] * u_axis + v0[:, None] * v_axis
    p11 = origin + u1[:, None] * u_axis + v1[:, None] * v_axis
    p01 = origin + u0[:, None] * u_axis + v1[:, None] * v_axis
    vertices = np.stack([p00, p10, p11, p01], axis=1)
    if keep_mask is not None:
        keep = np.asarray(keep_mask, dtype=bool).reshape(-1)
        if keep.shape != (vertices.shape[0],):
            raise ValueError("keep_mask must contain one flag per face rectangle.")
        vertices = vertices[keep]
    if vertices.shape[0] == 0:
        return []
    centroids = np.mean(vertices, axis=1)
    areas = np.linalg.norm(
        np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 3] - vertices[:, 0]),
        axis=1,
    )
    quadrature_points, quadrature_weights = _quadrature_from_vertices(
        vertices,
        quadrature_points_per_patch,
    )
    return [
        SurfacePatch(
            patch_id=int(patch_id_start + index),
            centroid_xyz=centroids[index],
            normal_xyz=normal,
            area_m2=float(areas[index]),
            surface_kind=surface_kind,
            object_id=object_id,
            vertices_xyz=vertices[index],
            quadrature_points_xyz=quadrature_points[index],
            quadrature_weights=quadrature_weights[index],
            parent_patch_id=parent_patch_id,
            refinement_level=refinement_level,
        )
        for index in range(vertices.shape[0])
    ]


def _blocked_cells_and_bounds(
    environment: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Return blocked cell indices and exact x0/x1/y0/y1 bounds."""
    if obstacle_grid is None or not obstacle_grid.blocked_cells:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0, 4), dtype=float)
    cells = np.asarray(obstacle_grid.blocked_cells, dtype=np.int64).reshape(-1, 2)
    origin = np.asarray(obstacle_grid.origin, dtype=float)
    cell_size = float(obstacle_grid.cell_size)
    x0 = origin[0] + cells[:, 0].astype(float) * cell_size
    y0 = origin[1] + cells[:, 1].astype(float) * cell_size
    bounds = np.column_stack([x0, x0 + cell_size, y0, y0 + cell_size])
    if np.any(bounds[:, 0] < -_GEOMETRY_TOLERANCE_M) or np.any(
        bounds[:, 2] < -_GEOMETRY_TOLERANCE_M
    ):
        raise ValueError("Blocked obstacle cells must lie inside the room.")
    if np.any(bounds[:, 1] > float(environment.size_x) + _GEOMETRY_TOLERANCE_M) or np.any(
        bounds[:, 3] > float(environment.size_y) + _GEOMETRY_TOLERANCE_M
    ):
        raise ValueError("Blocked obstacle cells must lie inside the room.")
    return cells, bounds


def _blocked_mask_xy(
    points_xy: NDArray[np.float64],
    obstacle_grid: ObstacleGrid | None,
) -> NDArray[np.bool_]:
    """Return a vectorized blocked-cell mask for face centroids."""
    points = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    if obstacle_grid is None or not obstacle_grid.blocked_cells:
        return np.zeros(points.shape[0], dtype=bool)
    relative = (
        points - np.asarray(obstacle_grid.origin, dtype=float)[None, :]
    ) / float(obstacle_grid.cell_size)
    indices = np.floor(relative).astype(np.int64)
    inside = (
        (indices[:, 0] >= 0)
        & (indices[:, 1] >= 0)
        & (indices[:, 0] < int(obstacle_grid.grid_shape[0]))
        & (indices[:, 1] < int(obstacle_grid.grid_shape[1]))
    )
    blocked = np.asarray(obstacle_grid.blocked_cells, dtype=np.int64).reshape(-1, 2)
    width = int(obstacle_grid.grid_shape[1])
    point_codes = indices[:, 0] * width + indices[:, 1]
    blocked_codes = blocked[:, 0] * width + blocked[:, 1]
    return inside & np.isin(point_codes, blocked_codes)


def _exposed_obstacle_sides(
    obstacle_grid: ObstacleGrid,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return blocked cells and west/east/south/north IDs for exposed sides."""
    cells = np.asarray(obstacle_grid.blocked_cells, dtype=np.int64).reshape(-1, 2)
    if cells.size == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.int64)
    offsets = np.asarray([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int64)
    neighbors = cells[:, None, :] + offsets[None, :, :]
    inside = (
        (neighbors[:, :, 0] >= 0)
        & (neighbors[:, :, 1] >= 0)
        & (neighbors[:, :, 0] < int(obstacle_grid.grid_shape[0]))
        & (neighbors[:, :, 1] < int(obstacle_grid.grid_shape[1]))
    )
    width = int(obstacle_grid.grid_shape[1])
    blocked_codes = cells[:, 0] * width + cells[:, 1]
    neighbor_codes = neighbors[:, :, 0] * width + neighbors[:, :, 1]
    neighbor_is_blocked = inside & np.isin(neighbor_codes, blocked_codes)
    cell_rows, side_ids = np.nonzero(~neighbor_is_blocked)
    return cells[cell_rows], side_ids.astype(np.int64, copy=False)


def _shared_edge_graph(
    patches: Sequence[SurfacePatch],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Find all positive-length shared axis-aligned edge segments.

    Edges are grouped by their varying coordinate and the two fixed physical
    coordinates.  Interval overlap then handles both equal-resolution and
    coarse/fine neighbors without an all-patches quadratic search.
    """
    if not patches:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=float)
    vertices = np.stack([patch.vertices_xyz for patch in patches], axis=0)
    starts = vertices
    ends = np.roll(vertices, shift=-1, axis=1)
    delta = ends - starts
    varying = np.abs(delta) > _GEOMETRY_TOLERANCE_M
    if np.any(np.sum(varying, axis=2) != 1):
        raise ValueError("Surface adjacency currently requires axis-aligned rectangles.")
    axes = np.argmax(varying, axis=2).reshape(-1)
    flat_starts = starts.reshape(-1, 3)
    flat_ends = ends.reshape(-1, 3)
    patch_ids = np.repeat(
        np.asarray([patch.patch_id for patch in patches], dtype=np.int64),
        4,
    )
    groups: dict[
        tuple[int, float, float],
        list[tuple[int, float, float]],
    ] = {}
    for row, axis in enumerate(axes):
        fixed_axes = tuple(index for index in range(3) if index != int(axis))
        fixed_first = float(
            np.round(flat_starts[row, fixed_axes[0]], decimals=10)
        )
        fixed_second = float(
            np.round(flat_starts[row, fixed_axes[1]], decimals=10)
        )
        low = float(min(flat_starts[row, axis], flat_ends[row, axis]))
        high = float(max(flat_starts[row, axis], flat_ends[row, axis]))
        groups.setdefault((int(axis), fixed_first, fixed_second), []).append(
            (int(patch_ids[row]), low, high)
        )

    pair_parts: list[NDArray[np.int64]] = []
    length_parts: list[NDArray[np.float64]] = []
    for records in groups.values():
        if len(records) < 2:
            continue
        values = np.asarray(records, dtype=float)
        ids = values[:, 0].astype(np.int64)
        lows = values[:, 1]
        highs = values[:, 2]
        overlap = np.minimum(highs[:, None], highs[None, :]) - np.maximum(
            lows[:, None], lows[None, :]
        )
        rows, columns = np.nonzero(
            np.triu(
                (overlap > _GEOMETRY_TOLERANCE_M)
                & (ids[:, None] != ids[None, :]),
                k=1,
            )
        )
        if rows.size:
            pair_parts.append(
                np.sort(np.column_stack([ids[rows], ids[columns]]), axis=1)
            )
            length_parts.append(overlap[rows, columns])
    if not pair_parts:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=float)
    raw_pairs = np.vstack(pair_parts).astype(np.int64, copy=False)
    raw_lengths = np.concatenate(length_parts).astype(float, copy=False)
    unique_pairs, inverse = np.unique(raw_pairs, axis=0, return_inverse=True)
    lengths = np.bincount(
        inverse,
        weights=raw_lengths,
        minlength=unique_pairs.shape[0],
    ).astype(float, copy=False)
    return unique_pairs, lengths


def _attach_neighbors(patches: Sequence[SurfacePatch]) -> SurfacePatchSet:
    """Build graph metadata and return a self-consistent patch set."""
    patch_id_edges, lengths = _shared_edge_graph(patches)
    neighbors: dict[int, list[tuple[int, float]]] = {
        patch.patch_id: [] for patch in patches
    }
    for (first, second), length in zip(patch_id_edges, lengths):
        neighbors[int(first)].append((int(second), float(length)))
        neighbors[int(second)].append((int(first), float(length)))
    enriched = []
    for patch in patches:
        entries = sorted(neighbors[patch.patch_id], key=lambda item: item[0])
        enriched.append(
            replace(
                patch,
                neighbor_patch_ids=tuple(item[0] for item in entries),
                neighbor_shared_edge_lengths_m=tuple(item[1] for item in entries),
            )
        )
    id_to_index = {
        patch.patch_id: index for index, patch in enumerate(enriched)
    }
    index_edges = np.asarray(
        [
            (id_to_index[int(first)], id_to_index[int(second)])
            for first, second in patch_id_edges
        ],
        dtype=np.int64,
    ).reshape(-1, 2)
    return SurfacePatchSet(
        patches=tuple(enriched),
        adjacency_edges=index_edges,
        shared_edge_lengths_m=lengths,
    )


def build_surface_patches(
    environment: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    spacing: float | Sequence[float],
    *,
    obstacle_height_m: float = 2.0,
    quadrature_points_per_patch: int = 4,
) -> SurfacePatchSet:
    """Build exact room and exposed obstacle surface rectangles.

    Room normals point into the room.  Obstacle normals point out of the solid
    obstacle and into free space.  Floor portions covered by blocked obstacle
    cells are split on exact obstacle boundaries and removed rather than being
    approximated with a center-only test.
    """
    dimensions = np.asarray(
        [environment.size_x, environment.size_y, environment.size_z],
        dtype=float,
    )
    if np.any(~np.isfinite(dimensions)) or np.any(dimensions <= 0.0):
        raise ValueError("Environment dimensions must be finite and positive.")
    spacing_x, spacing_y, spacing_z = _normalize_spacing(spacing)
    quadrature_count = int(quadrature_points_per_patch)
    if quadrature_count not in {1, 4}:
        raise ValueError("quadrature_points_per_patch must be 1 or 4.")
    obstacle_height = float(obstacle_height_m)
    if not np.isfinite(obstacle_height) or obstacle_height <= 0.0:
        raise ValueError("obstacle_height_m must be finite and positive.")
    has_obstacles = bool(obstacle_grid is not None and obstacle_grid.blocked_cells)
    if (
        has_obstacles
        and obstacle_height > float(environment.size_z) + _GEOMETRY_TOLERANCE_M
    ):
        raise ValueError("obstacle_height_m cannot exceed the room height.")
    obstacle_height = min(obstacle_height, float(environment.size_z))

    cells, bounds = _blocked_cells_and_bounds(environment, obstacle_grid)
    floor_x_extras = np.concatenate([bounds[:, 0], bounds[:, 1]]) if bounds.size else ()
    floor_y_extras = np.concatenate([bounds[:, 2], bounds[:, 3]]) if bounds.size else ()
    floor_x_edges = _axis_edges(
        float(environment.size_x),
        spacing_x,
        extra_edges_m=floor_x_extras,
    )
    floor_y_edges = _axis_edges(
        float(environment.size_y),
        spacing_y,
        extra_edges_m=floor_y_extras,
    )
    room_x_edges = _axis_edges(float(environment.size_x), spacing_x)
    room_y_edges = _axis_edges(float(environment.size_y), spacing_y)
    room_z_edges = _axis_edges(float(environment.size_z), spacing_z)

    patches: list[SurfacePatch] = []

    def append_face(**kwargs: object) -> None:
        """Split and append one rectangular physical face into exact patches."""
        face = _face_patches(
            patch_id_start=(patches[-1].patch_id + 1 if patches else 0),
            quadrature_points_per_patch=quadrature_count,
            **kwargs,
        )
        patches.extend(face)

    floor_u0, floor_v0 = np.meshgrid(
        floor_x_edges[:-1],
        floor_y_edges[:-1],
        indexing="ij",
    )
    floor_u1, floor_v1 = np.meshgrid(
        floor_x_edges[1:],
        floor_y_edges[1:],
        indexing="ij",
    )
    floor_centers_xy = np.column_stack(
        [
            0.5 * (floor_u0.reshape(-1) + floor_u1.reshape(-1)),
            0.5 * (floor_v0.reshape(-1) + floor_v1.reshape(-1)),
        ]
    )
    floor_keep = ~_blocked_mask_xy(floor_centers_xy, obstacle_grid)
    append_face(
        origin_xyz=(0.0, 0.0, 0.0),
        u_axis_xyz=(1.0, 0.0, 0.0),
        v_axis_xyz=(0.0, 1.0, 0.0),
        u_edges_m=floor_x_edges,
        v_edges_m=floor_y_edges,
        normal_xyz=(0.0, 0.0, 1.0),
        surface_kind="floor",
        object_id="room:floor",
        keep_mask=floor_keep,
    )
    append_face(
        origin_xyz=(0.0, float(environment.size_y), float(environment.size_z)),
        u_axis_xyz=(1.0, 0.0, 0.0),
        v_axis_xyz=(0.0, -1.0, 0.0),
        u_edges_m=room_x_edges,
        v_edges_m=room_y_edges,
        normal_xyz=(0.0, 0.0, -1.0),
        surface_kind="ceiling",
        object_id="room:ceiling",
    )
    append_face(
        origin_xyz=(0.0, 0.0, 0.0),
        u_axis_xyz=(0.0, 1.0, 0.0),
        v_axis_xyz=(0.0, 0.0, 1.0),
        u_edges_m=room_y_edges,
        v_edges_m=room_z_edges,
        normal_xyz=(1.0, 0.0, 0.0),
        surface_kind="wall",
        object_id="room:wall:x_min",
    )
    append_face(
        origin_xyz=(float(environment.size_x), 0.0, float(environment.size_z)),
        u_axis_xyz=(0.0, 1.0, 0.0),
        v_axis_xyz=(0.0, 0.0, -1.0),
        u_edges_m=room_y_edges,
        v_edges_m=room_z_edges,
        normal_xyz=(-1.0, 0.0, 0.0),
        surface_kind="wall",
        object_id="room:wall:x_max",
    )
    append_face(
        origin_xyz=(0.0, 0.0, float(environment.size_z)),
        u_axis_xyz=(1.0, 0.0, 0.0),
        v_axis_xyz=(0.0, 0.0, -1.0),
        u_edges_m=room_x_edges,
        v_edges_m=room_z_edges,
        normal_xyz=(0.0, 1.0, 0.0),
        surface_kind="wall",
        object_id="room:wall:y_min",
    )
    append_face(
        origin_xyz=(0.0, float(environment.size_y), 0.0),
        u_axis_xyz=(1.0, 0.0, 0.0),
        v_axis_xyz=(0.0, 0.0, 1.0),
        u_edges_m=room_x_edges,
        v_edges_m=room_z_edges,
        normal_xyz=(0.0, -1.0, 0.0),
        surface_kind="wall",
        object_id="room:wall:y_max",
    )

    if obstacle_grid is not None and cells.size:
        cell_size = float(obstacle_grid.cell_size)
        local_x_edges = _axis_edges(cell_size, spacing_x)
        local_y_edges = _axis_edges(cell_size, spacing_y)
        local_z_edges = _axis_edges(obstacle_height, spacing_z)
        if obstacle_height < float(environment.size_z) - _GEOMETRY_TOLERANCE_M:
            for (ix, iy), (x0, _x1, y0, _y1) in zip(cells, bounds):
                append_face(
                    origin_xyz=(float(x0), float(y0), obstacle_height),
                    u_axis_xyz=(1.0, 0.0, 0.0),
                    v_axis_xyz=(0.0, 1.0, 0.0),
                    u_edges_m=local_x_edges,
                    v_edges_m=local_y_edges,
                    normal_xyz=(0.0, 0.0, 1.0),
                    surface_kind="obstacle_top",
                    object_id=f"obstacle:{int(ix)}:{int(iy)}:top",
                )
        side_cells, side_ids = _exposed_obstacle_sides(obstacle_grid)
        origin_xy = np.asarray(obstacle_grid.origin, dtype=float)
        side_names = ("west", "east", "south", "north")
        for (ix, iy), side_id in zip(side_cells, side_ids):
            x0 = float(origin_xy[0] + int(ix) * cell_size)
            y0 = float(origin_xy[1] + int(iy) * cell_size)
            # A cell face coincident with a room boundary is buried against
            # that wall, not exposed to the room interior.
            if side_id == 0 and x0 <= _GEOMETRY_TOLERANCE_M:
                continue
            if (
                side_id == 1
                and x0 + cell_size
                >= float(environment.size_x) - _GEOMETRY_TOLERANCE_M
            ):
                continue
            if side_id == 2 and y0 <= _GEOMETRY_TOLERANCE_M:
                continue
            if (
                side_id == 3
                and y0 + cell_size
                >= float(environment.size_y) - _GEOMETRY_TOLERANCE_M
            ):
                continue
            side_name = side_names[int(side_id)]
            if side_id == 0:
                origin = (x0, y0, obstacle_height)
                u_axis = (0.0, 1.0, 0.0)
                v_axis = (0.0, 0.0, -1.0)
                normal = (-1.0, 0.0, 0.0)
                horizontal_edges = local_y_edges
            elif side_id == 1:
                origin = (x0 + cell_size, y0, 0.0)
                u_axis = (0.0, 1.0, 0.0)
                v_axis = (0.0, 0.0, 1.0)
                normal = (1.0, 0.0, 0.0)
                horizontal_edges = local_y_edges
            elif side_id == 2:
                origin = (x0, y0, 0.0)
                u_axis = (1.0, 0.0, 0.0)
                v_axis = (0.0, 0.0, 1.0)
                normal = (0.0, -1.0, 0.0)
                horizontal_edges = local_x_edges
            else:
                origin = (x0, y0 + cell_size, obstacle_height)
                u_axis = (1.0, 0.0, 0.0)
                v_axis = (0.0, 0.0, -1.0)
                normal = (0.0, 1.0, 0.0)
                horizontal_edges = local_x_edges
            append_face(
                origin_xyz=origin,
                u_axis_xyz=u_axis,
                v_axis_xyz=v_axis,
                u_edges_m=horizontal_edges,
                v_edges_m=local_z_edges,
                normal_xyz=normal,
                surface_kind="obstacle_side",
                object_id=f"obstacle:{int(ix)}:{int(iy)}:{side_name}",
            )

    return _attach_neighbors(patches)


def _child_patch(
    parent: SurfacePatch,
    *,
    patch_id: int,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    quadrature_points_per_patch: int,
) -> SurfacePatch:
    """Create one exact child rectangle in parent-local coordinates."""
    p00 = parent.vertices_xyz[0]
    u_vector = parent.vertices_xyz[1] - p00
    v_vector = parent.vertices_xyz[3] - p00
    vertices = np.asarray(
        [
            p00 + u_min * u_vector + v_min * v_vector,
            p00 + u_max * u_vector + v_min * v_vector,
            p00 + u_max * u_vector + v_max * v_vector,
            p00 + u_min * u_vector + v_max * v_vector,
        ],
        dtype=float,
    )
    quadrature_points, quadrature_weights = _quadrature_from_vertices(
        vertices[None, :, :],
        quadrature_points_per_patch,
    )
    return SurfacePatch(
        patch_id=patch_id,
        centroid_xyz=np.mean(vertices, axis=0),
        normal_xyz=parent.normal_xyz,
        area_m2=float(
            np.linalg.norm(
                np.cross(vertices[1] - vertices[0], vertices[3] - vertices[0])
            )
        ),
        surface_kind=parent.surface_kind,
        object_id=parent.object_id,
        vertices_xyz=vertices,
        quadrature_points_xyz=quadrature_points[0],
        quadrature_weights=quadrature_weights[0],
        parent_patch_id=parent.patch_id,
        refinement_level=parent.refinement_level + 1,
    )


def refine_surface_patches(
    patch_set: SurfacePatchSet,
    patch_ids: Iterable[int],
    *,
    quadrature_points_per_patch: int | None = None,
) -> SurfacePatchSet:
    """Replace selected active patches with four area-preserving children.

    Stable child IDs are allocated above the current maximum.  Parent patches
    leave the active set but remain referenced by each child's
    ``parent_patch_id``.  The complete physical adjacency graph is rebuilt, so
    a coarse edge can correctly neighbor two refined child edges.
    """
    selected = {int(value) for value in patch_ids}
    if not selected:
        return patch_set
    active_ids = {patch.patch_id for patch in patch_set.patches}
    missing = selected - active_ids
    if missing:
        raise KeyError(f"Cannot refine inactive or unknown patch IDs: {sorted(missing)}")
    if quadrature_points_per_patch is not None and int(
        quadrature_points_per_patch
    ) not in {1, 4}:
        raise ValueError("quadrature_points_per_patch must be 1 or 4.")

    next_patch_id = max(active_ids) + 1
    refined: list[SurfacePatch] = []
    child_bounds = (
        (0.0, 0.5, 0.0, 0.5),
        (0.5, 1.0, 0.0, 0.5),
        (0.5, 1.0, 0.5, 1.0),
        (0.0, 0.5, 0.5, 1.0),
    )
    for patch in patch_set.patches:
        if patch.patch_id not in selected:
            refined.append(
                replace(
                    patch,
                    neighbor_patch_ids=(),
                    neighbor_shared_edge_lengths_m=(),
                )
            )
            continue
        child_quadrature = (
            patch.quadrature_count
            if quadrature_points_per_patch is None
            else int(quadrature_points_per_patch)
        )
        for u_min, u_max, v_min, v_max in child_bounds:
            refined.append(
                _child_patch(
                    patch,
                    patch_id=next_patch_id,
                    u_min=u_min,
                    u_max=u_max,
                    v_min=v_min,
                    v_max=v_max,
                    quadrature_points_per_patch=child_quadrature,
                )
            )
            next_patch_id += 1
    return _attach_neighbors(refined)


build_surface_patch_set = build_surface_patches
refine_surface_patch_set = refine_surface_patches


__all__ = [
    "build_surface_patch_set",
    "build_surface_patches",
    "refine_surface_patch_set",
    "refine_surface_patches",
]
