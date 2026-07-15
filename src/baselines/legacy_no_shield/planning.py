"""Measurement planning for the baseline particle filter."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid


def measurement_count(total_time_s: float, measurement_time_s: float = 30.0) -> int:
    """Return the number of measurements implied by the total time."""
    if total_time_s <= 0.0:
        raise ValueError("total_time_s must be positive.")
    if measurement_time_s <= 0.0:
        raise ValueError("measurement_time_s must be positive.")
    count = int(total_time_s // measurement_time_s)
    return max(1, count)


def _grid_centers(
    origin: tuple[float, float],
    cell_size: float,
    grid_shape: tuple[int, int],
    blocked_cells: Iterable[tuple[int, int]] | None = None,
) -> NDArray[np.float64]:
    """Return centers of grid cells, excluding any blocked cells."""
    nx, ny = grid_shape
    xs = origin[0] + (np.arange(nx, dtype=float) + 0.5) * cell_size
    ys = origin[1] + (np.arange(ny, dtype=float) + 0.5) * cell_size
    blocked = set(blocked_cells or [])
    centers: list[list[float]] = []
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            if (ix, iy) in blocked:
                continue
            centers.append([float(x), float(y)])
    if not centers:
        return np.zeros((0, 2), dtype=float)
    return np.asarray(centers, dtype=float)


def _farthest_point_sample(
    points: NDArray[np.float64],
    n_select: int,
) -> NDArray[np.float64]:
    """Select points that maximize the minimum spacing (greedy FPS)."""
    if points.size == 0:
        return points
    if n_select <= 0:
        return np.zeros((0, points.shape[1]), dtype=float)
    if n_select >= points.shape[0]:
        return points.copy()
    centroid = np.mean(points, axis=0)
    start_idx = int(np.argmin(np.linalg.norm(points - centroid, axis=1)))
    selected_indices = [start_idx]
    min_dist = np.linalg.norm(points - points[start_idx], axis=1)
    for _ in range(1, n_select):
        next_idx = int(np.argmax(min_dist))
        selected_indices.append(next_idx)
        dist = np.linalg.norm(points - points[next_idx], axis=1)
        min_dist = np.minimum(min_dist, dist)
    return points[np.array(selected_indices, dtype=int)]


def _path_length(points: NDArray[np.float64]) -> float:
    """Return total path length through a sequence of points."""
    if points.shape[0] < 2:
        return 0.0
    diffs = points[1:] - points[:-1]
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def _two_opt_route(points: NDArray[np.float64], max_iters: int = 50) -> NDArray[np.float64]:
    """Improve a route using 2-opt swaps."""
    route = points.copy()
    best_len = _path_length(route)
    n = route.shape[0]
    if n < 4:
        return route
    for _ in range(max_iters):
        improved = False
        for i in range(1, n - 2):
            for k in range(i + 1, n - 1):
                new_route = route.copy()
                new_route[i:k + 1] = route[i:k + 1][::-1]
                new_len = _path_length(new_route)
                if new_len + 1e-9 < best_len:
                    route = new_route
                    best_len = new_len
                    improved = True
        if not improved:
            break
    return route


def _shortest_route(
    points: NDArray[np.float64],
    start_xy: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return a short route visiting each point once (nearest neighbor + 2-opt)."""
    remaining = list(range(points.shape[0]))
    current = np.asarray(start_xy, dtype=float).reshape(1, 2)
    route_indices: list[int] = []
    while remaining:
        pts = points[remaining]
        dists = np.linalg.norm(pts - current, axis=1)
        next_idx = int(np.argmin(dists))
        chosen = remaining.pop(next_idx)
        route_indices.append(chosen)
        current = points[chosen:chosen + 1]
    ordered = points[np.array(route_indices, dtype=int)]
    return _two_opt_route(ordered)


def generate_measurement_positions(
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    total_time_s: float,
    *,
    measurement_time_s: float = 30.0,
) -> NDArray[np.float64]:
    """Return measurement positions and a short visiting route, repeating if needed."""
    num_measurements = measurement_count(
        total_time_s,
        measurement_time_s=measurement_time_s,
    )
    if obstacle_grid is None:
        nx = int(np.floor(env.size_x))
        ny = int(np.floor(env.size_y))
        centers = _grid_centers((0.0, 0.0), 1.0, (nx, ny), blocked_cells=None)
    else:
        centers = _grid_centers(
            obstacle_grid.origin,
            obstacle_grid.cell_size,
            obstacle_grid.grid_shape,
            blocked_cells=obstacle_grid.blocked_cells,
        )
    if centers.size == 0:
        raise ValueError("No free grid cells available for measurement.")
    if num_measurements > centers.shape[0]:
        selected = centers.copy()
    else:
        selected = _farthest_point_sample(centers, num_measurements)
    start_xy = np.asarray(env.detector()[:2], dtype=float)
    ordered = _shortest_route(selected, start_xy)
    if num_measurements > ordered.shape[0]:
        repeats = int(np.ceil(num_measurements / ordered.shape[0]))
        ordered = np.tile(ordered, (repeats, 1))[:num_measurements]
    z = float(env.detector()[2])
    positions = np.column_stack([ordered, np.full(ordered.shape[0], z, dtype=float)])
    return positions
