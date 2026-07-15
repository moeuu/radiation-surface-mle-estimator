"""Surface-constrained source placement utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from measurement.model import EnvironmentConfig, PointSource
from measurement.obstacles import ObstacleGrid

SourceSurfaceKind = Literal[
    "floor",
    "ceiling",
    "wall",
    "obstacle_side",
    "obstacle_top",
]


def _cell_bounds(
    grid: ObstacleGrid,
    cell: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Return x/y bounds for an obstacle grid cell."""
    ix, iy = int(cell[0]), int(cell[1])
    x0 = grid.origin[0] + ix * grid.cell_size
    y0 = grid.origin[1] + iy * grid.cell_size
    return x0, x0 + grid.cell_size, y0, y0 + grid.cell_size


def _is_xy_in_blocked_cell(
    point_xy: Sequence[float],
    grid: ObstacleGrid | None,
) -> bool:
    """Return whether an x/y point lies inside a blocked obstacle footprint."""
    if grid is None:
        return False
    idx = grid.cell_index(point_xy)
    return idx is not None and not grid.is_cell_free(idx)


def transport_interior_mask(
    positions: NDArray[np.float64],
    obstacle_grid: ObstacleGrid | None,
    *,
    tolerance_m: float = 1.0e-6,
) -> NDArray[np.bool_]:
    """Return True for positions strictly inside known obstacle transport boxes."""
    points = np.asarray(positions, dtype=float).reshape(-1, 3)
    if obstacle_grid is None or not getattr(obstacle_grid, "has_transport_model", False):
        return np.zeros(points.shape[0], dtype=bool)
    boxes = np.asarray(obstacle_grid.transport_boxes(), dtype=float).reshape(-1, 6)
    if boxes.size == 0:
        return np.zeros(points.shape[0], dtype=bool)
    tol = max(float(tolerance_m), 0.0)
    lower = boxes[:, :3] + tol
    upper = boxes[:, 3:] - tol
    valid_boxes = np.all(upper > lower, axis=1)
    if not np.any(valid_boxes):
        return np.zeros(points.shape[0], dtype=bool)
    inside_lower = points[:, None, :] > lower[None, valid_boxes, :]
    inside_upper = points[:, None, :] < upper[None, valid_boxes, :]
    return np.any(np.all(inside_lower & inside_upper, axis=2), axis=1)


def _is_exposed_obstacle_side(
    grid: ObstacleGrid,
    cell: tuple[int, int],
    side: str,
) -> bool:
    """Return whether an obstacle side borders free space or the room exterior."""
    ix, iy = int(cell[0]), int(cell[1])
    neighbor_by_side = {
        "west": (ix - 1, iy),
        "east": (ix + 1, iy),
        "south": (ix, iy - 1),
        "north": (ix, iy + 1),
    }
    neighbor = neighbor_by_side[side]
    is_inside = (
        0 <= neighbor[0] < grid.grid_shape[0]
        and 0 <= neighbor[1] < grid.grid_shape[1]
    )
    return not is_inside or grid.is_cell_free(neighbor)


def obstacle_surface_sides(
    grid: ObstacleGrid,
) -> list[tuple[tuple[int, int], str]]:
    """Return obstacle sides that are exposed to free space."""
    sides: list[tuple[tuple[int, int], str]] = []
    for cell in grid.blocked_cells:
        for side in ("west", "east", "south", "north"):
            if _is_exposed_obstacle_side(grid, cell, side):
                sides.append((cell, side))
    return sides


def _surface_spacing_tuple(spacing: float | Sequence[float]) -> tuple[float, float, float]:
    """Return a validated 3D spacing tuple."""
    arr = np.asarray(spacing, dtype=float)
    if arr.size == 1:
        arr = np.repeat(float(arr.reshape(-1)[0]), 3)
    if arr.shape != (3,):
        raise ValueError("spacing must be a scalar or a 3-element vector.")
    if np.any(arr <= 0.0):
        raise ValueError("spacing values must be positive.")
    return float(arr[0]), float(arr[1]), float(arr[2])


def _axis_points(start: float, stop: float, spacing: float) -> NDArray[np.float64]:
    """Return evenly spaced axis points including both endpoints."""
    start_f = float(start)
    stop_f = float(stop)
    step = float(spacing)
    if step <= 0.0:
        raise ValueError("spacing must be positive.")
    if stop_f < start_f:
        return np.zeros(0, dtype=float)
    if np.isclose(stop_f, start_f):
        return np.array([start_f], dtype=float)
    count = int(np.floor((stop_f - start_f) / step)) + 1
    points = start_f + step * np.arange(max(1, count), dtype=float)
    if points.size == 0 or points[-1] < stop_f - 1.0e-9:
        points = np.append(points, stop_f)
    else:
        points[-1] = stop_f if np.isclose(points[-1], stop_f) else points[-1]
    return np.unique(np.round(points, 9)).astype(float, copy=False)


def _blocked_cell_array(grid: ObstacleGrid | None) -> NDArray[np.int64]:
    """Return blocked obstacle cells as an ``(N, 2)`` integer array."""
    if grid is None or not grid.blocked_cells:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(grid.blocked_cells, dtype=np.int64).reshape(-1, 2)


def _blocked_cell_bounds_array(grid: ObstacleGrid | None) -> NDArray[np.float64]:
    """Return blocked-cell bounds as ``(x0, x1, y0, y1)`` rows."""
    cells = _blocked_cell_array(grid)
    if grid is None or cells.size == 0:
        return np.zeros((0, 4), dtype=float)
    x0 = float(grid.origin[0]) + cells[:, 0].astype(float) * float(grid.cell_size)
    y0 = float(grid.origin[1]) + cells[:, 1].astype(float) * float(grid.cell_size)
    return np.column_stack(
        [
            x0,
            x0 + float(grid.cell_size),
            y0,
            y0 + float(grid.cell_size),
        ]
    )


def _blocked_mask_xy(
    points_xy: NDArray[np.float64],
    grid: ObstacleGrid | None,
) -> NDArray[np.bool_]:
    """Return a vectorized mask for points inside blocked obstacle cells."""
    points = np.asarray(points_xy, dtype=float)
    original_shape = points.shape[:-1]
    if points.ndim == 1:
        points = points.reshape(1, 2)
        original_shape = (1,)
    if points.shape[-1] != 2:
        raise ValueError("points_xy must have a final dimension of 2.")
    flat = points.reshape(-1, 2)
    cells = _blocked_cell_array(grid)
    if grid is None or cells.size == 0 or flat.size == 0:
        return np.zeros(original_shape, dtype=bool)
    rel = (
        flat - np.asarray(grid.origin, dtype=float)[None, :]
    ) / float(grid.cell_size)
    idx = np.floor(rel).astype(np.int64)
    inside = (
        (idx[:, 0] >= 0)
        & (idx[:, 1] >= 0)
        & (idx[:, 0] < int(grid.grid_shape[0]))
        & (idx[:, 1] < int(grid.grid_shape[1]))
    )
    point_codes = idx[:, 0] * int(grid.grid_shape[1]) + idx[:, 1]
    blocked_codes = cells[:, 0] * int(grid.grid_shape[1]) + cells[:, 1]
    blocked = inside & np.isin(point_codes, blocked_codes)
    return blocked.reshape(original_shape)


def _dedupe_positions(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return unique candidate positions with stable floating-point rounding."""
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=float)
    arr = arr.reshape(-1, 3)
    return np.unique(np.round(arr, 9), axis=0).astype(float, copy=False)


def _filter_position_bounds(
    points: NDArray[np.float64],
    env: EnvironmentConfig,
    position_min: Sequence[float] | None,
    position_max: Sequence[float] | None,
) -> NDArray[np.float64]:
    """Return candidate points inside the configured PF position support."""
    arr = np.asarray(points, dtype=float).reshape(-1, 3)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=float)
    room_lo = np.array([0.0, 0.0, 0.0], dtype=float)
    room_hi = np.array([env.size_x, env.size_y, env.size_z], dtype=float)
    lo = room_lo if position_min is None else np.asarray(position_min, dtype=float)
    hi = room_hi if position_max is None else np.asarray(position_max, dtype=float)
    if lo.shape != (3,) or hi.shape != (3,):
        raise ValueError("position bounds must be 3-element vectors.")
    lo = np.maximum(lo, room_lo)
    hi = np.minimum(hi, room_hi)
    keep = np.all((arr >= lo[None, :] - 1.0e-9) & (arr <= hi[None, :] + 1.0e-9), axis=1)
    return arr[keep]


def _stack_plane(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    constant: float,
    axis: int,
) -> NDArray[np.float64]:
    """Return a mesh plane embedded in 3D with one constant axis."""
    aa, bb = np.meshgrid(first, second, indexing="ij")
    points = np.zeros((aa.size, 3), dtype=float)
    free_axes = [idx for idx in range(3) if idx != axis]
    points[:, free_axes[0]] = aa.reshape(-1)
    points[:, free_axes[1]] = bb.reshape(-1)
    points[:, axis] = float(constant)
    return points


def _room_surface_candidates(
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    spacing: tuple[float, float, float],
) -> NDArray[np.float64]:
    """Return gridded candidates on room walls, floor, and ceiling."""
    xs = _axis_points(0.0, float(env.size_x), spacing[0])
    ys = _axis_points(0.0, float(env.size_y), spacing[1])
    zs = _axis_points(0.0, float(env.size_z), spacing[2])
    wall_x0 = _stack_plane(ys, zs, 0.0, axis=0)
    wall_x1 = _stack_plane(ys, zs, float(env.size_x), axis=0)
    wall_y0 = _stack_plane(xs, zs, 0.0, axis=1)
    wall_y1 = _stack_plane(xs, zs, float(env.size_y), axis=1)
    floor = _stack_plane(xs, ys, 0.0, axis=2)
    if floor.size:
        floor = floor[~_blocked_mask_xy(floor[:, :2], obstacle_grid)]
    ceiling = _stack_plane(xs, ys, float(env.size_z), axis=2)
    return np.vstack([wall_x0, wall_x1, wall_y0, wall_y1, floor, ceiling])


def _obstacle_top_candidates(
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    spacing: tuple[float, float, float],
    obstacle_height_m: float,
) -> NDArray[np.float64]:
    """Return gridded candidates on blocked-cell obstacle top surfaces."""
    bounds = _blocked_cell_bounds_array(obstacle_grid)
    if obstacle_grid is None or bounds.size == 0:
        return np.zeros((0, 3), dtype=float)
    obstacle_height = min(max(0.0, float(obstacle_height_m)), float(env.size_z))
    local_x = _axis_points(0.0, float(obstacle_grid.cell_size), spacing[0])
    local_y = _axis_points(0.0, float(obstacle_grid.cell_size), spacing[1])
    lx, ly = np.meshgrid(local_x, local_y, indexing="ij")
    offsets = np.column_stack([lx.reshape(-1), ly.reshape(-1)])
    x = bounds[:, 0, None] + offsets[None, :, 0]
    y = bounds[:, 2, None] + offsets[None, :, 1]
    z = np.full_like(x, obstacle_height, dtype=float)
    return np.column_stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)])


def _exposed_side_arrays(
    grid: ObstacleGrid | None,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return exposed obstacle side cells and side IDs in vectorized arrays."""
    cells = _blocked_cell_array(grid)
    if grid is None or cells.size == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.int64)
    offsets = np.array(
        [
            [-1, 0],
            [1, 0],
            [0, -1],
            [0, 1],
        ],
        dtype=np.int64,
    )
    neighbors = cells[:, None, :] + offsets[None, :, :]
    inside = (
        (neighbors[:, :, 0] >= 0)
        & (neighbors[:, :, 1] >= 0)
        & (neighbors[:, :, 0] < int(grid.grid_shape[0]))
        & (neighbors[:, :, 1] < int(grid.grid_shape[1]))
    )
    blocked_codes = cells[:, 0] * int(grid.grid_shape[1]) + cells[:, 1]
    neighbor_codes = neighbors[:, :, 0] * int(grid.grid_shape[1]) + neighbors[:, :, 1]
    neighbor_blocked = inside & np.isin(neighbor_codes, blocked_codes)
    exposed = ~inside | ~neighbor_blocked
    cell_idx, side_ids = np.nonzero(exposed)
    return cells[cell_idx], side_ids.astype(np.int64, copy=False)


def _obstacle_side_candidates(
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    spacing: tuple[float, float, float],
    obstacle_height_m: float,
) -> NDArray[np.float64]:
    """Return gridded candidates on exposed blocked-cell obstacle side surfaces."""
    if obstacle_grid is None or not obstacle_grid.blocked_cells:
        return np.zeros((0, 3), dtype=float)
    cells, side_ids = _exposed_side_arrays(obstacle_grid)
    if cells.size == 0:
        return np.zeros((0, 3), dtype=float)
    obstacle_height = min(max(0.0, float(obstacle_height_m)), float(env.size_z))
    z_axis = _axis_points(0.0, obstacle_height, spacing[2])
    xy_axis_by_side = {
        0: _axis_points(0.0, float(obstacle_grid.cell_size), spacing[1]),
        1: _axis_points(0.0, float(obstacle_grid.cell_size), spacing[1]),
        2: _axis_points(0.0, float(obstacle_grid.cell_size), spacing[0]),
        3: _axis_points(0.0, float(obstacle_grid.cell_size), spacing[0]),
    }
    parts: list[NDArray[np.float64]] = []
    for side_id in range(4):
        selected = side_ids == side_id
        if not np.any(selected):
            continue
        side_cells = cells[selected]
        x0 = float(obstacle_grid.origin[0]) + side_cells[:, 0].astype(float) * float(
            obstacle_grid.cell_size
        )
        y0 = float(obstacle_grid.origin[1]) + side_cells[:, 1].astype(float) * float(
            obstacle_grid.cell_size
        )
        axis_points = xy_axis_by_side[side_id]
        aa, zz = np.meshgrid(axis_points, z_axis, indexing="ij")
        if side_id in (0, 1):
            x_const = x0 if side_id == 0 else x0 + float(obstacle_grid.cell_size)
            x = np.broadcast_to(x_const[:, None], (side_cells.shape[0], aa.size))
            y = y0[:, None] + aa.reshape(1, -1)
        else:
            x = x0[:, None] + aa.reshape(1, -1)
            y_const = y0 if side_id == 2 else y0 + float(obstacle_grid.cell_size)
            y = np.broadcast_to(y_const[:, None], (side_cells.shape[0], aa.size))
        z = np.broadcast_to(zz.reshape(1, -1), (side_cells.shape[0], zz.size))
        parts.append(np.column_stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)]))
    if not parts:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(parts)


def build_surface_candidate_sources(
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    spacing: float | Sequence[float],
    *,
    position_min: Sequence[float] | None = None,
    position_max: Sequence[float] | None = None,
    obstacle_height_m: float = 2.0,
) -> NDArray[np.float64]:
    """Create source candidates on known room and obstacle surfaces."""
    spacing_tuple = _surface_spacing_tuple(spacing)
    parts = [
        _room_surface_candidates(env, obstacle_grid, spacing_tuple),
        _obstacle_top_candidates(env, obstacle_grid, spacing_tuple, obstacle_height_m),
        _obstacle_side_candidates(env, obstacle_grid, spacing_tuple, obstacle_height_m),
    ]
    candidates = _dedupe_positions(np.vstack(parts))
    candidates = _filter_position_bounds(candidates, env, position_min, position_max)
    if candidates.size:
        candidates = candidates[~transport_interior_mask(candidates, obstacle_grid)]
    candidates = _dedupe_positions(candidates)
    if candidates.size == 0:
        raise ValueError("Surface candidate grid is empty; check bounds and spacing.")
    return candidates


def _consider_projection(
    points: NDArray[np.float64],
    candidate: NDArray[np.float64],
    best: NDArray[np.float64],
    best_dist2: NDArray[np.float64],
    valid: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Update nearest projections using one batched candidate projection."""
    diff = points - np.asarray(candidate, dtype=float)
    dist2 = np.sum(diff * diff, axis=1)
    if valid is not None:
        dist2 = np.where(np.asarray(valid, dtype=bool).reshape(-1), dist2, np.inf)
    keep = dist2 < best_dist2
    if np.any(keep):
        best[keep] = candidate[keep]
        best_dist2[keep] = dist2[keep]
    return best, best_dist2


def _consider_obstacle_top_projection(
    points: NDArray[np.float64],
    bounds: NDArray[np.float64],
    obstacle_height: float,
    best: NDArray[np.float64],
    best_dist2: NDArray[np.float64],
    *,
    chunk_size: int = 256,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Update nearest projections against obstacle top rectangles in chunks."""
    if bounds.size == 0:
        return best, best_dist2
    for start in range(0, bounds.shape[0], max(1, int(chunk_size))):
        chunk = bounds[start : start + max(1, int(chunk_size))]
        x = np.clip(points[:, 0, None], chunk[None, :, 0], chunk[None, :, 1])
        y = np.clip(points[:, 1, None], chunk[None, :, 2], chunk[None, :, 3])
        z = np.full_like(x, float(obstacle_height), dtype=float)
        dist2 = (
            (points[:, 0, None] - x) ** 2
            + (points[:, 1, None] - y) ** 2
            + (points[:, 2, None] - z) ** 2
        )
        local_idx = np.argmin(dist2, axis=1)
        row = np.arange(points.shape[0])
        candidate = np.column_stack(
            [x[row, local_idx], y[row, local_idx], z[row, local_idx]]
        )
        best, best_dist2 = _consider_projection(
            points,
            candidate,
            best,
            best_dist2,
            valid=np.isfinite(dist2[row, local_idx]),
        )
    return best, best_dist2


def _consider_obstacle_side_projection(
    points: NDArray[np.float64],
    grid: ObstacleGrid,
    cells: NDArray[np.int64],
    side_ids: NDArray[np.int64],
    obstacle_height: float,
    best: NDArray[np.float64],
    best_dist2: NDArray[np.float64],
    *,
    chunk_size: int = 256,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Update nearest projections against exposed obstacle side rectangles."""
    if cells.size == 0:
        return best, best_dist2
    cell_size = float(grid.cell_size)
    for start in range(0, cells.shape[0], max(1, int(chunk_size))):
        side_cells = cells[start : start + max(1, int(chunk_size))]
        side = side_ids[start : start + max(1, int(chunk_size))]
        x0 = float(grid.origin[0]) + side_cells[:, 0].astype(float) * cell_size
        x1 = x0 + cell_size
        y0 = float(grid.origin[1]) + side_cells[:, 1].astype(float) * cell_size
        y1 = y0 + cell_size
        x = np.clip(points[:, 0, None], x0[None, :], x1[None, :])
        y = np.clip(points[:, 1, None], y0[None, :], y1[None, :])
        x = np.where(side[None, :] == 0, x0[None, :], x)
        x = np.where(side[None, :] == 1, x1[None, :], x)
        y = np.where(side[None, :] == 2, y0[None, :], y)
        y = np.where(side[None, :] == 3, y1[None, :], y)
        z = np.broadcast_to(
            np.clip(points[:, 2, None], 0.0, float(obstacle_height)),
            x.shape,
        )
        dist2 = (
            (points[:, 0, None] - x) ** 2
            + (points[:, 1, None] - y) ** 2
            + (points[:, 2, None] - z) ** 2
        )
        local_idx = np.argmin(dist2, axis=1)
        row = np.arange(points.shape[0])
        candidate = np.column_stack(
            [x[row, local_idx], y[row, local_idx], z[row, local_idx]]
        )
        best, best_dist2 = _consider_projection(
            points,
            candidate,
            best,
            best_dist2,
            valid=np.isfinite(dist2[row, local_idx]),
        )
    return best, best_dist2


def project_positions_to_allowed_surfaces(
    positions: NDArray[np.float64],
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None = None,
    *,
    obstacle_height_m: float = 2.0,
) -> NDArray[np.float64]:
    """Project batched source positions to the nearest allowed source surface."""
    arr = np.asarray(positions, dtype=float)
    original_shape = arr.shape
    if arr.size == 0:
        return np.zeros(original_shape, dtype=float)
    if arr.shape[-1] != 3:
        raise ValueError("positions must have a final dimension of 3.")
    points = arr.reshape(-1, 3)
    room_lo = np.array([0.0, 0.0, 0.0], dtype=float)
    room_hi = np.array([env.size_x, env.size_y, env.size_z], dtype=float)
    points = np.clip(points, room_lo[None, :], room_hi[None, :])
    best = points.copy()
    best_dist2 = np.full(points.shape[0], np.inf, dtype=float)

    floor = points.copy()
    floor[:, 2] = 0.0
    best, best_dist2 = _consider_projection(
        points,
        floor,
        best,
        best_dist2,
        valid=~_blocked_mask_xy(floor[:, :2], obstacle_grid),
    )
    ceiling = points.copy()
    ceiling[:, 2] = float(env.size_z)
    best, best_dist2 = _consider_projection(points, ceiling, best, best_dist2)
    for axis, value in ((0, 0.0), (0, float(env.size_x)), (1, 0.0), (1, float(env.size_y))):
        wall = points.copy()
        wall[:, axis] = value
        best, best_dist2 = _consider_projection(points, wall, best, best_dist2)

    if obstacle_grid is not None and obstacle_grid.blocked_cells:
        obstacle_height = min(max(0.0, float(obstacle_height_m)), float(env.size_z))
        best, best_dist2 = _consider_obstacle_top_projection(
            points,
            _blocked_cell_bounds_array(obstacle_grid),
            obstacle_height,
            best,
            best_dist2,
        )
        side_cells, side_ids = _exposed_side_arrays(obstacle_grid)
        best, best_dist2 = _consider_obstacle_side_projection(
            points,
            obstacle_grid,
            side_cells,
            side_ids,
            obstacle_height,
            best,
            best_dist2,
        )
    return best.reshape(original_shape)


def source_surface_kind(
    position: Sequence[float],
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None = None,
    *,
    obstacle_height_m: float = 2.0,
    tolerance_m: float = 1.0e-6,
) -> SourceSurfaceKind | None:
    """Return the allowed source surface kind for a position, or None."""
    if len(position) != 3:
        raise ValueError("position must be a 3-element vector.")
    x, y, z = (float(position[0]), float(position[1]), float(position[2]))
    tol = max(0.0, float(tolerance_m))
    if x < -tol or y < -tol or z < -tol:
        return None
    if x > float(env.size_x) + tol:
        return None
    if y > float(env.size_y) + tol:
        return None
    if z > float(env.size_z) + tol:
        return None
    if abs(z) <= tol and not _is_xy_in_blocked_cell((x, y), obstacle_grid):
        return "floor"
    if abs(z - float(env.size_z)) <= tol:
        return "ceiling"
    if (
        abs(x) <= tol
        or abs(x - float(env.size_x)) <= tol
        or abs(y) <= tol
        or abs(y - float(env.size_y)) <= tol
    ):
        return "wall"
    if obstacle_grid is None:
        return None
    obstacle_height = float(obstacle_height_m)
    if abs(z - obstacle_height) <= tol and _is_xy_in_blocked_cell((x, y), obstacle_grid):
        return "obstacle_top"
    if z < -tol or z > obstacle_height + tol:
        return None
    for cell, side in obstacle_surface_sides(obstacle_grid):
        x0, x1, y0, y1 = _cell_bounds(obstacle_grid, cell)
        if side == "west" and abs(x - x0) <= tol and y0 - tol <= y <= y1 + tol:
            return "obstacle_side"
        if side == "east" and abs(x - x1) <= tol and y0 - tol <= y <= y1 + tol:
            return "obstacle_side"
        if side == "south" and abs(y - y0) <= tol and x0 - tol <= x <= x1 + tol:
            return "obstacle_side"
        if side == "north" and abs(y - y1) <= tol and x0 - tol <= x <= x1 + tol:
            return "obstacle_side"
    return None


def source_surface_kinds(
    positions: NDArray[np.float64],
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None = None,
    *,
    obstacle_height_m: float = 2.0,
    tolerance_m: float = 1.0e-6,
) -> NDArray[np.object_]:
    """Return vectorized allowed source surface kinds for batched positions."""
    arr = np.asarray(positions, dtype=float)
    if arr.shape[-1:] != (3,):
        raise ValueError("positions must have a final dimension of 3.")
    if arr.size == 0:
        return np.zeros(arr.reshape(-1, 3).shape[0], dtype=object)
    points = arr.reshape(-1, 3)
    tol = max(0.0, float(tolerance_m))
    kinds = np.full(points.shape[0], None, dtype=object)
    valid = (
        (points[:, 0] >= -tol)
        & (points[:, 1] >= -tol)
        & (points[:, 2] >= -tol)
        & (points[:, 0] <= float(env.size_x) + tol)
        & (points[:, 1] <= float(env.size_y) + tol)
        & (points[:, 2] <= float(env.size_z) + tol)
    )
    if not np.any(valid):
        return kinds

    blocked = _blocked_mask_xy(points[:, :2], obstacle_grid)
    floor = valid & (np.abs(points[:, 2]) <= tol) & ~blocked
    kinds[floor] = "floor"

    unset = valid & np.equal(kinds, None)
    ceiling = unset & (np.abs(points[:, 2] - float(env.size_z)) <= tol)
    kinds[ceiling] = "ceiling"

    unset = valid & np.equal(kinds, None)
    wall = unset & (
        (np.abs(points[:, 0]) <= tol)
        | (np.abs(points[:, 0] - float(env.size_x)) <= tol)
        | (np.abs(points[:, 1]) <= tol)
        | (np.abs(points[:, 1] - float(env.size_y)) <= tol)
    )
    kinds[wall] = "wall"

    if obstacle_grid is None or not obstacle_grid.blocked_cells:
        return kinds

    obstacle_height = float(obstacle_height_m)
    unset = valid & np.equal(kinds, None)
    top = unset & (np.abs(points[:, 2] - obstacle_height) <= tol) & blocked
    kinds[top] = "obstacle_top"

    unset = valid & np.equal(kinds, None)
    height_valid = (
        unset
        & (points[:, 2] >= -tol)
        & (points[:, 2] <= obstacle_height + tol)
    )
    if not np.any(height_valid):
        return kinds
    side_cells, side_ids = _exposed_side_arrays(obstacle_grid)
    if side_cells.size == 0:
        return kinds

    cell_size = float(obstacle_grid.cell_size)
    for start in range(0, side_cells.shape[0], 256):
        cells = side_cells[start : start + 256]
        sides = side_ids[start : start + 256]
        x0 = float(obstacle_grid.origin[0]) + cells[:, 0].astype(float) * cell_size
        x1 = x0 + cell_size
        y0 = float(obstacle_grid.origin[1]) + cells[:, 1].astype(float) * cell_size
        y1 = y0 + cell_size
        x = points[:, 0, None]
        y = points[:, 1, None]
        west = (
            (sides[None, :] == 0)
            & (np.abs(x - x0[None, :]) <= tol)
            & (y >= y0[None, :] - tol)
            & (y <= y1[None, :] + tol)
        )
        east = (
            (sides[None, :] == 1)
            & (np.abs(x - x1[None, :]) <= tol)
            & (y >= y0[None, :] - tol)
            & (y <= y1[None, :] + tol)
        )
        south = (
            (sides[None, :] == 2)
            & (np.abs(y - y0[None, :]) <= tol)
            & (x >= x0[None, :] - tol)
            & (x <= x1[None, :] + tol)
        )
        north = (
            (sides[None, :] == 3)
            & (np.abs(y - y1[None, :]) <= tol)
            & (x >= x0[None, :] - tol)
            & (x <= x1[None, :] + tol)
        )
        matches = west | east | south | north
        side_match = height_valid & np.any(matches, axis=1)
        kinds[side_match] = "obstacle_side"
        height_valid &= ~side_match
        if not np.any(height_valid):
            break
    return kinds


def source_surface_kind_counts(
    positions: NDArray[np.float64],
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None = None,
    *,
    obstacle_height_m: float = 2.0,
    tolerance_m: float = 1.0e-6,
) -> dict[str, int]:
    """Return surface-kind counts for batched source positions."""
    kinds = source_surface_kinds(
        positions,
        env,
        obstacle_grid,
        obstacle_height_m=obstacle_height_m,
        tolerance_m=tolerance_m,
    )
    labels = [
        "floor",
        "ceiling",
        "wall",
        "obstacle_side",
        "obstacle_top",
        "off_surface",
    ]
    counts = {label: 0 for label in labels}
    for label in labels[:-1]:
        counts[label] = int(np.count_nonzero(kinds == label))
    counts["off_surface"] = int(np.count_nonzero(np.equal(kinds, None)))
    return counts


def is_allowed_source_surface_position(
    position: Sequence[float],
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None = None,
    *,
    obstacle_height_m: float = 2.0,
    tolerance_m: float = 1.0e-6,
) -> bool:
    """Return True when a source position lies on an allowed physical surface."""
    point = np.asarray(position, dtype=float).reshape(1, 3)
    if bool(transport_interior_mask(point, obstacle_grid, tolerance_m=tolerance_m)[0]):
        return False
    return (
        source_surface_kind(
            position,
            env,
            obstacle_grid,
            obstacle_height_m=obstacle_height_m,
            tolerance_m=tolerance_m,
        )
        is not None
    )


def _visibility_obstacle_boxes_m(
    obstacle_grid: ObstacleGrid | None,
    obstacle_height_m: float,
) -> NDArray[np.float64]:
    """Return footprint boxes used for source-placement visibility screening."""
    if obstacle_grid is None or not obstacle_grid.blocked_cells:
        return np.zeros((0, 6), dtype=float)
    boxes = obstacle_grid.blocked_boxes(
        z_min=0.0,
        z_max=max(0.0, float(obstacle_height_m)),
    )
    if not boxes:
        return np.zeros((0, 6), dtype=float)
    return np.asarray(boxes, dtype=float).reshape(-1, 6)


def _coerce_visibility_measurement_points(
    measurement_points: NDArray[np.float64],
    *,
    detector_height_m: float,
) -> NDArray[np.float64]:
    """Return visibility reference points as an ``(N, 3)`` float array."""
    points = np.asarray(measurement_points, dtype=float)
    if points.size == 0:
        return np.zeros((0, 3), dtype=float)
    points = points.reshape(-1, points.shape[-1])
    if points.shape[1] == 2:
        z = np.full((points.shape[0], 1), float(detector_height_m), dtype=float)
        return np.column_stack([points, z])
    if points.shape[1] >= 3:
        return points[:, :3].astype(float, copy=False)
    raise ValueError("measurement_points must contain 2D or 3D points.")


def _segment_path_lengths_through_boxes_m(
    source_positions: NDArray[np.float64],
    detector_positions: NDArray[np.float64],
    boxes_m: NDArray[np.float64],
    *,
    box_chunk_size: int = 64,
) -> NDArray[np.float64]:
    """Return batched obstacle path lengths for source-detector segments."""
    sources = np.asarray(source_positions, dtype=float).reshape(-1, 3)
    detectors = np.asarray(detector_positions, dtype=float).reshape(-1, 3)
    boxes = np.asarray(boxes_m, dtype=float).reshape(-1, 6)
    if sources.size == 0 or detectors.size == 0 or boxes.size == 0:
        return np.zeros((sources.shape[0], detectors.shape[0]), dtype=float)
    total = np.zeros((sources.shape[0], detectors.shape[0]), dtype=float)
    delta = detectors[None, :, :] - sources[:, None, :]
    distance = np.linalg.norm(delta, axis=-1)
    p0 = sources[:, None, None, :]
    step = delta[:, :, None, :]
    chunk_size = max(1, int(box_chunk_size))
    for start in range(0, boxes.shape[0], chunk_size):
        chunk = boxes[start : start + chunk_size]
        lower = chunk[None, None, :, :3]
        upper = chunk[None, None, :, 3:]
        t_enter = np.zeros((sources.shape[0], detectors.shape[0], chunk.shape[0]))
        t_exit = np.ones_like(t_enter)
        valid = np.ones_like(t_enter, dtype=bool)
        for axis in range(3):
            value = p0[..., axis]
            axis_step = step[..., axis]
            lo = lower[..., axis]
            hi = upper[..., axis]
            parallel = np.abs(axis_step) <= 1.0e-12
            inside = (value >= lo) & (value <= hi)
            valid &= ~parallel | inside
            safe_step = np.where(parallel, 1.0, axis_step)
            t0 = (lo - value) / safe_step
            t1 = (hi - value) / safe_step
            axis_enter = np.minimum(t0, t1)
            axis_exit = np.maximum(t0, t1)
            axis_enter = np.where(parallel & inside, -np.inf, axis_enter)
            axis_exit = np.where(parallel & inside, np.inf, axis_exit)
            t_enter = np.maximum(t_enter, axis_enter)
            t_exit = np.minimum(t_exit, axis_exit)
        overlap = valid & (t_exit > t_enter)
        lengths = np.where(overlap, (t_exit - t_enter) * distance[:, :, None], 0.0)
        total += np.sum(lengths, axis=2)
    return total


def surface_observable_fractions(
    positions: NDArray[np.float64],
    obstacle_grid: ObstacleGrid | None,
    measurement_points: NDArray[np.float64],
    *,
    obstacle_height_m: float = 2.0,
    detector_height_m: float = 0.5,
    clear_path_max_m: float = 0.01,
    box_chunk_size: int = 64,
) -> NDArray[np.float64]:
    """Return the clear ground-view fraction for each source position."""
    sources = np.asarray(positions, dtype=float).reshape(-1, 3)
    if sources.size == 0:
        return np.zeros(0, dtype=float)
    detectors = _coerce_visibility_measurement_points(
        measurement_points,
        detector_height_m=detector_height_m,
    )
    if detectors.size == 0:
        return np.ones(sources.shape[0], dtype=float)
    boxes = _visibility_obstacle_boxes_m(obstacle_grid, obstacle_height_m)
    if boxes.size == 0:
        return np.ones(sources.shape[0], dtype=float)
    path_lengths = _segment_path_lengths_through_boxes_m(
        sources,
        detectors,
        boxes,
        box_chunk_size=box_chunk_size,
    )
    visible = path_lengths <= max(0.0, float(clear_path_max_m))
    return np.mean(visible, axis=1).astype(float, copy=False)


def surface_response_observability_diagnostics(
    positions: NDArray[np.float64],
    obstacle_grid: ObstacleGrid | None,
    measurement_points: NDArray[np.float64],
    *,
    isotopes: Sequence[str] | None = None,
    obstacle_height_m: float = 2.0,
    detector_height_m: float = 0.5,
    clear_path_max_m: float = 0.01,
    box_chunk_size: int = 64,
) -> dict[str, float | int]:
    """
    Return geometry-based response separability diagnostics for source placement.

    This is a random-scenario screening diagnostic, not a PF observation model.
    It uses clear ground-view rays and inverse-square response proxies to reject
    source sets whose columns are nearly indistinguishable from the reachable
    floor measurement manifold.
    """
    sources = np.asarray(positions, dtype=float).reshape(-1, 3)
    detectors = _coerce_visibility_measurement_points(
        measurement_points,
        detector_height_m=detector_height_m,
    )
    if sources.size == 0 or detectors.size == 0:
        return {
            "source_count": int(sources.shape[0]),
            "reference_point_count": int(detectors.shape[0]),
            "condition_number": 1.0,
            "max_pairwise_correlation": 0.0,
            "same_isotope_pair_count": 0,
            "same_isotope_max_pairwise_correlation": 0.0,
            "weak_source_count": 0,
        }
    diff = detectors[:, None, :] - sources[None, :, :]
    distance2 = np.maximum(np.sum(diff * diff, axis=2), 1.0e-6)
    response = 1.0 / distance2
    boxes = _visibility_obstacle_boxes_m(obstacle_grid, obstacle_height_m)
    if boxes.size:
        path_lengths = _segment_path_lengths_through_boxes_m(
            sources,
            detectors,
            boxes,
            box_chunk_size=box_chunk_size,
        )
        visible = path_lengths <= max(0.0, float(clear_path_max_m))
        response = response * visible.T.astype(float)
    column_norm = np.linalg.norm(response, axis=0)
    valid = column_norm > 1.0e-12
    weak_count = int(np.count_nonzero(~valid))
    if np.count_nonzero(valid) <= 1:
        return {
            "source_count": int(sources.shape[0]),
            "reference_point_count": int(detectors.shape[0]),
            "condition_number": 1.0 if np.count_nonzero(valid) == 1 else float("inf"),
            "max_pairwise_correlation": 0.0,
            "same_isotope_pair_count": 0,
            "same_isotope_max_pairwise_correlation": 0.0,
            "weak_source_count": weak_count,
        }
    normalized = response[:, valid] / np.maximum(column_norm[valid], 1.0e-12)
    try:
        singular_values = np.linalg.svd(normalized, compute_uv=False)
        positive = singular_values[singular_values > 1.0e-12]
        condition = (
            float(np.max(positive) / max(float(np.min(positive)), 1.0e-12))
            if positive.size
            else float("inf")
        )
    except np.linalg.LinAlgError:
        condition = float("inf")
    corr = np.abs(normalized.T @ normalized)
    upper = np.triu_indices(corr.shape[0], k=1)
    max_corr = float(np.max(corr[upper])) if upper[0].size else 0.0
    same_pair_count = 0
    same_max_corr = 0.0
    if isotopes is not None:
        labels = np.asarray([str(value) for value in isotopes], dtype=object).reshape(-1)
        if labels.size == sources.shape[0]:
            valid_indices = np.flatnonzero(valid)
            full_corr = np.zeros((sources.shape[0], sources.shape[0]), dtype=float)
            full_corr[np.ix_(valid_indices, valid_indices)] = corr
            for label in np.unique(labels):
                indices = np.flatnonzero(labels == label)
                if indices.size < 2:
                    continue
                same_upper = np.triu_indices(indices.size, k=1)
                same_pair_count += int(same_upper[0].size)
                values = full_corr[np.ix_(indices, indices)][same_upper]
                if values.size:
                    same_max_corr = max(same_max_corr, float(np.max(values)))
    return {
        "source_count": int(sources.shape[0]),
        "reference_point_count": int(detectors.shape[0]),
        "condition_number": condition,
        "max_pairwise_correlation": max_corr,
        "same_isotope_pair_count": int(same_pair_count),
        "same_isotope_max_pairwise_correlation": float(same_max_corr),
        "weak_source_count": weak_count,
    }


def is_ground_observable_source_position(
    position: Sequence[float],
    obstacle_grid: ObstacleGrid | None,
    measurement_points: NDArray[np.float64],
    *,
    min_visible_fraction: float,
    obstacle_height_m: float = 2.0,
    detector_height_m: float = 0.5,
    clear_path_max_m: float = 0.01,
) -> bool:
    """Return True when enough reachable ground views have a clear ray."""
    fraction = surface_observable_fractions(
        np.asarray(position, dtype=float).reshape(1, 3),
        obstacle_grid,
        measurement_points,
        obstacle_height_m=obstacle_height_m,
        detector_height_m=detector_height_m,
        clear_path_max_m=clear_path_max_m,
    )
    return bool(fraction.size and fraction[0] >= float(min_visible_fraction))


def _weighted_choice(
    rng: np.random.Generator,
    weights_by_kind: Sequence[tuple[SourceSurfaceKind, float]],
) -> SourceSurfaceKind:
    """Sample a surface kind from non-negative area weights."""
    filtered = [(kind, float(weight)) for kind, weight in weights_by_kind if weight > 0.0]
    if not filtered:
        raise ValueError("No source-placement surfaces are available.")
    kinds = [kind for kind, _ in filtered]
    weights = np.array([weight for _, weight in filtered], dtype=float)
    weights /= float(np.sum(weights))
    return kinds[int(rng.choice(len(kinds), p=weights))]


def sample_surface_position(
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    rng: np.random.Generator,
    *,
    obstacle_height_m: float = 2.0,
) -> tuple[float, float, float]:
    """Sample one source position from room and obstacle surfaces."""
    obstacle_height = min(max(0.0, float(obstacle_height_m)), float(env.size_z))
    blocked_count = 0 if obstacle_grid is None else len(obstacle_grid.blocked_cells)
    exposed_sides = [] if obstacle_grid is None else obstacle_surface_sides(obstacle_grid)
    floor_area = float(env.size_x) * float(env.size_y)
    if obstacle_grid is not None:
        floor_area = max(
            0.0,
            floor_area - blocked_count * obstacle_grid.cell_size * obstacle_grid.cell_size,
        )
    ceiling_area = float(env.size_x) * float(env.size_y)
    wall_area = 2.0 * (float(env.size_x) + float(env.size_y)) * float(env.size_z)
    obstacle_top_area = (
        0.0
        if obstacle_grid is None
        else blocked_count * obstacle_grid.cell_size * obstacle_grid.cell_size
    )
    obstacle_side_area = (
        0.0
        if obstacle_grid is None
        else len(exposed_sides) * obstacle_grid.cell_size * obstacle_height
    )
    kind = _weighted_choice(
        rng,
        (
            ("floor", floor_area),
            ("ceiling", ceiling_area),
            ("wall", wall_area),
            ("obstacle_side", obstacle_side_area),
            ("obstacle_top", obstacle_top_area),
        ),
    )
    if kind == "floor":
        return _sample_floor_position(env, obstacle_grid, rng)
    if kind == "ceiling":
        return (
            float(rng.uniform(0.0, float(env.size_x))),
            float(rng.uniform(0.0, float(env.size_y))),
            float(env.size_z),
        )
    if kind == "wall":
        return _sample_wall_position(env, rng)
    if kind == "obstacle_side":
        return _sample_obstacle_side_position(obstacle_grid, exposed_sides, rng, obstacle_height)
    return _sample_obstacle_top_position(obstacle_grid, rng, obstacle_height)


def _sample_floor_position(
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Sample a floor point outside obstacle footprints."""
    if obstacle_grid is None or len(obstacle_grid.blocked_cells) == 0:
        return (
            float(rng.uniform(0.0, float(env.size_x))),
            float(rng.uniform(0.0, float(env.size_y))),
            0.0,
        )
    free_cells = [
        (ix, iy)
        for ix in range(obstacle_grid.grid_shape[0])
        for iy in range(obstacle_grid.grid_shape[1])
        if obstacle_grid.is_cell_free((ix, iy))
    ]
    if not free_cells:
        raise ValueError("No free floor cells are available for source placement.")
    cell = free_cells[int(rng.integers(0, len(free_cells)))]
    x0, x1, y0, y1 = _cell_bounds(obstacle_grid, cell)
    return (float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)), 0.0)


def _sample_wall_position(
    env: EnvironmentConfig,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Sample a point from one of the four room walls."""
    weights = np.array(
        [float(env.size_y), float(env.size_y), float(env.size_x), float(env.size_x)],
        dtype=float,
    )
    weights /= float(np.sum(weights))
    wall = int(rng.choice(4, p=weights))
    z = float(rng.uniform(0.0, float(env.size_z)))
    if wall == 0:
        return (0.0, float(rng.uniform(0.0, float(env.size_y))), z)
    if wall == 1:
        return (float(env.size_x), float(rng.uniform(0.0, float(env.size_y))), z)
    if wall == 2:
        return (float(rng.uniform(0.0, float(env.size_x))), 0.0, z)
    return (float(rng.uniform(0.0, float(env.size_x))), float(env.size_y), z)


def _sample_obstacle_side_position(
    obstacle_grid: ObstacleGrid | None,
    exposed_sides: Sequence[tuple[tuple[int, int], str]],
    rng: np.random.Generator,
    obstacle_height_m: float,
) -> tuple[float, float, float]:
    """Sample a point from an exposed obstacle side surface."""
    if obstacle_grid is None or not exposed_sides:
        raise ValueError("No exposed obstacle side is available.")
    cell, side = exposed_sides[int(rng.integers(0, len(exposed_sides)))]
    x0, x1, y0, y1 = _cell_bounds(obstacle_grid, cell)
    z = float(rng.uniform(0.0, obstacle_height_m))
    if side == "west":
        return (x0, float(rng.uniform(y0, y1)), z)
    if side == "east":
        return (x1, float(rng.uniform(y0, y1)), z)
    if side == "south":
        return (float(rng.uniform(x0, x1)), y0, z)
    return (float(rng.uniform(x0, x1)), y1, z)


def _sample_obstacle_top_position(
    obstacle_grid: ObstacleGrid | None,
    rng: np.random.Generator,
    obstacle_height_m: float,
) -> tuple[float, float, float]:
    """Sample a point from an obstacle top surface."""
    if obstacle_grid is None or not obstacle_grid.blocked_cells:
        raise ValueError("No obstacle top is available.")
    cell = obstacle_grid.blocked_cells[int(rng.integers(0, len(obstacle_grid.blocked_cells)))]
    x0, x1, y0, y1 = _cell_bounds(obstacle_grid, cell)
    return (
        float(rng.uniform(x0, x1)),
        float(rng.uniform(y0, y1)),
        float(obstacle_height_m),
    )


def _sample_surface_position_batch(
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    rng: np.random.Generator,
    *,
    obstacle_height_m: float,
    batch_size: int,
) -> NDArray[np.float64]:
    """Draw a small batch of surface positions before batched visibility tests."""
    batch = max(1, int(batch_size))
    return np.asarray(
        [
            sample_surface_position(
                env,
                obstacle_grid,
                rng,
                obstacle_height_m=obstacle_height_m,
            )
            for _ in range(batch)
        ],
        dtype=float,
    ).reshape(-1, 3)


def _surface_candidate_constraint_mask(
    candidates: NDArray[np.float64],
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    *,
    obstacle_height_m: float,
    excluded_surface_kinds: Sequence[SourceSurfaceKind] | None,
    preferred_max_z_m: float | None,
    min_distance_points: NDArray[np.float64] | None = None,
    min_distance_m: float = 0.0,
) -> NDArray[np.bool_]:
    """Return candidates satisfying random-source placement constraints."""
    points = np.asarray(candidates, dtype=float).reshape(-1, 3)
    if points.size == 0:
        return np.zeros(0, dtype=bool)
    mask = np.ones(points.shape[0], dtype=bool)
    mask &= ~transport_interior_mask(points, obstacle_grid)
    if excluded_surface_kinds:
        kinds = source_surface_kinds(
            points,
            env,
            obstacle_grid,
            obstacle_height_m=obstacle_height_m,
        )
        excluded = [str(kind) for kind in excluded_surface_kinds]
        mask &= ~np.isin(kinds, excluded)
    if preferred_max_z_m is not None:
        mask &= points[:, 2] <= float(preferred_max_z_m) + 1.0e-9
    distance_floor = max(0.0, float(min_distance_m))
    if distance_floor > 0.0 and min_distance_points is not None:
        anchors = np.asarray(min_distance_points, dtype=float).reshape(-1, 3)
        if anchors.size:
            deltas = points[:, None, :] - anchors[None, :, :]
            distances = np.linalg.norm(deltas, axis=2)
            mask &= np.min(distances, axis=1) >= distance_floor - 1.0e-9
    return mask


def sample_observable_surface_position(
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    rng: np.random.Generator,
    *,
    measurement_points: NDArray[np.float64] | None,
    min_visible_fraction: float,
    obstacle_height_m: float = 2.0,
    detector_height_m: float = 0.5,
    clear_path_max_m: float = 0.01,
    batch_size: int = 256,
    max_attempts: int = 4096,
    excluded_surface_kinds: Sequence[SourceSurfaceKind] | None = None,
    preferred_max_z_m: float | None = 5.0,
    min_distance_points: NDArray[np.float64] | None = None,
    min_distance_m: float = 0.0,
) -> tuple[float, float, float]:
    """Sample a surface position with enough clear ground measurement views."""
    min_fraction = max(0.0, float(min_visible_fraction))
    sample_batch = max(1, int(batch_size))
    distance_floor = max(0.0, float(min_distance_m))
    distance_points = None
    if min_distance_points is not None:
        distance_points = np.asarray(min_distance_points, dtype=float).reshape(-1, 3)
    detectors = np.zeros((0, 3), dtype=float)
    if min_fraction > 0.0 and measurement_points is not None:
        detectors = _coerce_visibility_measurement_points(
            np.asarray(measurement_points, dtype=float),
            detector_height_m=detector_height_m,
        )

    def _draw_with_z_constraint(
        z_limit: float | None,
    ) -> tuple[float, float, float] | None:
        """Draw one candidate under an optional preferred-height constraint."""
        remaining = max(1, int(max_attempts))
        while remaining > 0:
            current_batch = min(sample_batch, remaining)
            candidates = _sample_surface_position_batch(
                env,
                obstacle_grid,
                rng,
                obstacle_height_m=obstacle_height_m,
                batch_size=current_batch,
            )
            valid_mask = _surface_candidate_constraint_mask(
                candidates,
                env,
                obstacle_grid,
                obstacle_height_m=obstacle_height_m,
                excluded_surface_kinds=excluded_surface_kinds,
                preferred_max_z_m=z_limit,
                min_distance_points=distance_points,
                min_distance_m=distance_floor,
            )
            valid = np.flatnonzero(valid_mask)
            if valid.size and detectors.size:
                fractions = surface_observable_fractions(
                    candidates[valid],
                    obstacle_grid,
                    detectors,
                    obstacle_height_m=obstacle_height_m,
                    detector_height_m=detector_height_m,
                    clear_path_max_m=clear_path_max_m,
                )
                valid = valid[fractions >= min_fraction]
            if valid.size:
                selected = int(valid[int(rng.integers(0, valid.size))])
                point = candidates[selected]
                return (float(point[0]), float(point[1]), float(point[2]))
            remaining -= current_batch
        return None

    preferred = None
    if preferred_max_z_m is not None:
        preferred = _draw_with_z_constraint(float(preferred_max_z_m))
    if preferred is not None:
        return preferred
    fallback = _draw_with_z_constraint(None)
    if fallback is not None:
        return fallback
    excluded = sorted(str(kind) for kind in excluded_surface_kinds or ())
    preference = (
        "none" if preferred_max_z_m is None else f"{float(preferred_max_z_m):.3f}m"
    )
    raise ValueError(
        "No surface-random source position satisfied the placement constraints "
        f"after {int(max_attempts)} attempts "
        f"(min_visible_fraction={min_fraction:.3f}, "
        f"excluded_surface_kinds={excluded}, preferred_max_z={preference})."
    )


def _normalise_max_ceiling_sources(value: int | None) -> int | None:
    """Return a validated maximum number of random ceiling sources."""
    if value is None:
        return None
    return max(0, int(value))


def _sample_constrained_source_position(
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    rng: np.random.Generator,
    *,
    obstacle_height_m: float,
    visibility_measurement_points: NDArray[np.float64] | None,
    visibility_min_fraction: float,
    visibility_detector_height_m: float,
    visibility_clear_path_max_m: float,
    visibility_batch_size: int,
    visibility_max_attempts_per_source: int,
    ceiling_count: int,
    max_ceiling_sources: int | None,
    preferred_max_z_m: float | None,
    min_distance_points: NDArray[np.float64] | None,
    min_distance_m: float,
) -> tuple[float, float, float]:
    """Sample one random source while enforcing scenario-level surface limits."""
    excluded: list[SourceSurfaceKind] = []
    if (
        max_ceiling_sources is not None
        and int(ceiling_count) >= int(max_ceiling_sources)
    ):
        excluded.append("ceiling")
    return sample_observable_surface_position(
        env,
        obstacle_grid,
        rng,
        measurement_points=visibility_measurement_points,
        min_visible_fraction=float(visibility_min_fraction),
        obstacle_height_m=obstacle_height_m,
        detector_height_m=visibility_detector_height_m,
        clear_path_max_m=visibility_clear_path_max_m,
        batch_size=visibility_batch_size,
        max_attempts=visibility_max_attempts_per_source,
        excluded_surface_kinds=excluded,
        preferred_max_z_m=preferred_max_z_m,
        min_distance_points=min_distance_points,
        min_distance_m=min_distance_m,
    )


def generate_surface_sources(
    *,
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    isotopes: Sequence[str],
    intensity_cps_1m: float | Sequence[float] | Mapping[str, float | Sequence[float]],
    rng: np.random.Generator,
    count: int | None = None,
    obstacle_height_m: float = 2.0,
    visibility_measurement_points: NDArray[np.float64] | None = None,
    visibility_min_fraction: float = 0.0,
    visibility_detector_height_m: float = 0.5,
    visibility_clear_path_max_m: float = 0.01,
    visibility_batch_size: int = 256,
    visibility_max_attempts_per_source: int = 4096,
    max_ceiling_sources: int | None = 1,
    preferred_max_z_m: float | None = 5.0,
    same_isotope_min_distance_m: float = 0.0,
) -> list[PointSource]:
    """Generate point sources constrained to observable physical surfaces."""
    if not isotopes:
        raise ValueError("At least one isotope is required.")
    source_count = len(isotopes) if count is None else max(1, int(count))
    ceiling_limit = _normalise_max_ceiling_sources(max_ceiling_sources)
    same_isotope_distance = max(0.0, float(same_isotope_min_distance_m))
    ceiling_count = 0
    sources: list[PointSource] = []
    for idx in range(source_count):
        isotope = str(isotopes[idx % len(isotopes)])
        intensity = _sample_intensity_cps_1m(
            intensity_cps_1m,
            isotope=isotope,
            rng=rng,
        )
        same_isotope_positions = np.asarray(
            [
                source.position
                for source in sources
                if str(source.isotope) == isotope
            ],
            dtype=float,
        ).reshape(-1, 3)
        position = _sample_constrained_source_position(
            env,
            obstacle_grid,
            rng,
            obstacle_height_m=obstacle_height_m,
            visibility_measurement_points=visibility_measurement_points,
            visibility_min_fraction=float(visibility_min_fraction),
            visibility_detector_height_m=visibility_detector_height_m,
            visibility_clear_path_max_m=visibility_clear_path_max_m,
            visibility_batch_size=visibility_batch_size,
            visibility_max_attempts_per_source=visibility_max_attempts_per_source,
            ceiling_count=ceiling_count,
            max_ceiling_sources=ceiling_limit,
            preferred_max_z_m=preferred_max_z_m,
            min_distance_points=same_isotope_positions,
            min_distance_m=same_isotope_distance,
        )
        surface_kind = source_surface_kind(
            position,
            env,
            obstacle_grid,
            obstacle_height_m=obstacle_height_m,
        )
        if surface_kind == "ceiling":
            ceiling_count += 1
        sources.append(
            PointSource(
                isotope=isotope,
                position=position,
                intensity_cps_1m=intensity,
            )
        )
    return sources


def _sample_intensity_cps_1m(
    intensity_cps_1m: float | Sequence[float] | Mapping[str, float | Sequence[float]],
    *,
    isotope: str,
    rng: np.random.Generator,
) -> float:
    """Return a fixed or uniformly sampled detector-cps@1m source strength."""
    raw_value: object
    if isinstance(intensity_cps_1m, Mapping):
        raw_value = intensity_cps_1m[isotope]
    else:
        raw_value = intensity_cps_1m
    if isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes)):
        if len(raw_value) != 2:
            raise ValueError("intensity range must contain exactly two values.")
        lo = float(raw_value[0])
        hi = float(raw_value[1])
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError("intensity range values must be finite.")
        if lo <= 0.0 or hi <= 0.0:
            raise ValueError("intensity range values must be positive.")
        if hi < lo:
            raise ValueError("intensity range maximum must be >= minimum.")
        if hi == lo:
            return lo
        return float(rng.uniform(lo, hi))
    intensity = float(raw_value)
    if not np.isfinite(intensity) or intensity <= 0.0:
        raise ValueError("intensity_cps_1m must be a positive finite value.")
    return intensity
