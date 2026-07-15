"""Obstacle layout utilities for blocking measurement positions."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np

EXPLORATION_BACKBONE_MAX_GAP_CELLS = 4


def _normalize_isotope_key(value: str) -> str:
    """Return a normalized isotope key for attenuation table lookup."""
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


@dataclass(frozen=True)
class ObstacleGrid:
    """Represent blocked 1 m grid cells on the z=0 plane."""

    origin: tuple[float, float]
    cell_size: float
    grid_shape: tuple[int, int]
    blocked_cells: tuple[tuple[int, int], ...]
    transport_boxes_m: tuple[tuple[float, float, float, float, float, float], ...] = ()
    transport_mu_by_isotope: dict[str, tuple[float, ...]] = field(default_factory=dict)
    transport_line_mu_by_isotope: dict[str, tuple[tuple[float, ...], ...]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        """Normalize inputs and validate bounds."""
        if len(self.origin) != 2:
            raise ValueError("origin must have length 2.")
        if len(self.grid_shape) != 2:
            raise ValueError("grid_shape must have length 2.")
        origin = (float(self.origin[0]), float(self.origin[1]))
        cell_size = float(self.cell_size)
        grid_shape = (int(self.grid_shape[0]), int(self.grid_shape[1]))
        if cell_size <= 0.0:
            raise ValueError("cell_size must be positive.")
        if grid_shape[0] < 0 or grid_shape[1] < 0:
            raise ValueError("grid_shape must be non-negative.")
        blocked = tuple(
            sorted((int(cell[0]), int(cell[1])) for cell in self.blocked_cells)
        )
        for cell in blocked:
            if cell[0] < 0 or cell[1] < 0:
                raise ValueError("blocked_cells indices must be non-negative.")
            if cell[0] >= grid_shape[0] or cell[1] >= grid_shape[1]:
                raise ValueError("blocked_cells entry out of grid bounds.")
        transport_boxes = tuple(
            tuple(float(value) for value in box)
            for box in self.transport_boxes_m
        )
        for box in transport_boxes:
            if len(box) != 6:
                raise ValueError("transport_boxes_m entries must have six values.")
            if box[3] < box[0] or box[4] < box[1] or box[5] < box[2]:
                raise ValueError("transport_boxes_m entries must be ordered lower-to-upper.")
        transport_mu: dict[str, tuple[float, ...]] = {}
        for isotope, values in self.transport_mu_by_isotope.items():
            mu_values = tuple(float(value) for value in values)
            if len(mu_values) != len(transport_boxes):
                raise ValueError(
                    "transport_mu_by_isotope entries must match transport_boxes_m length."
                )
            transport_mu[str(isotope)] = mu_values
        transport_line_mu: dict[str, tuple[tuple[float, ...], ...]] = {}
        for isotope, rows in self.transport_line_mu_by_isotope.items():
            parsed_rows: list[tuple[float, ...]] = []
            for row in rows:
                mu_values = tuple(float(value) for value in row)
                if len(mu_values) != len(transport_boxes):
                    raise ValueError(
                        "transport_line_mu_by_isotope rows must match "
                        "transport_boxes_m length."
                    )
                parsed_rows.append(mu_values)
            transport_line_mu[str(isotope)] = tuple(parsed_rows)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "cell_size", cell_size)
        object.__setattr__(self, "grid_shape", grid_shape)
        object.__setattr__(self, "blocked_cells", blocked)
        object.__setattr__(self, "transport_boxes_m", transport_boxes)
        object.__setattr__(self, "transport_mu_by_isotope", transport_mu)
        object.__setattr__(
            self,
            "transport_line_mu_by_isotope",
            transport_line_mu,
        )
        object.__setattr__(self, "_blocked_set", frozenset(blocked))

    @property
    def total_cells(self) -> int:
        """Return total number of grid cells."""
        return int(self.grid_shape[0] * self.grid_shape[1])

    @property
    def blocked_fraction(self) -> float:
        """Return the fraction of blocked cells."""
        total = self.total_cells
        if total == 0:
            return 0.0
        return float(len(self.blocked_cells)) / float(total)

    def cell_index(self, point: Sequence[float]) -> tuple[int, int] | None:
        """Return the (ix, iy) cell index for a point, or None if outside."""
        if len(point) < 2:
            raise ValueError("point must have at least two coordinates.")
        x = float(point[0])
        y = float(point[1])
        rel_x = x - self.origin[0]
        rel_y = y - self.origin[1]
        if rel_x < 0.0 or rel_y < 0.0:
            return None
        ix = int(np.floor(rel_x / self.cell_size))
        iy = int(np.floor(rel_y / self.cell_size))
        if ix < 0 or iy < 0:
            return None
        if ix >= self.grid_shape[0] or iy >= self.grid_shape[1]:
            return None
        return ix, iy

    def is_free(self, point: Sequence[float]) -> bool:
        """Return True if the point is not inside a blocked cell."""
        idx = self.cell_index(point)
        if idx is None:
            return True
        return idx not in self._blocked_set

    def is_cell_free(self, cell: tuple[int, int]) -> bool:
        """Return True if the grid cell is inside bounds and not blocked."""
        ix, iy = (int(cell[0]), int(cell[1]))
        if ix < 0 or iy < 0:
            return False
        if ix >= self.grid_shape[0] or iy >= self.grid_shape[1]:
            return False
        return (ix, iy) not in self._blocked_set

    def has_free_path(
        self,
        start_point: Sequence[float],
        goal_point: Sequence[float],
    ) -> bool:
        """Return True when two points are connected through free cells."""
        start = self.cell_index(start_point)
        goal = self.cell_index(goal_point)
        if start is None or goal is None:
            return False
        if not self.is_cell_free(start) or not self.is_cell_free(goal):
            return False
        if start == goal:
            return True
        visited = {start}
        frontier = [start]
        while frontier:
            ix, iy = frontier.pop(0)
            for neighbor in ((ix - 1, iy), (ix + 1, iy), (ix, iy - 1), (ix, iy + 1)):
                if neighbor in visited or not self.is_cell_free(neighbor):
                    continue
                if neighbor == goal:
                    return True
                visited.add(neighbor)
                frontier.append(neighbor)
        return False

    def blocked_bounds(self) -> list[tuple[float, float, float, float]]:
        """Return (x0, x1, y0, y1) bounds for blocked cells."""
        bounds: list[tuple[float, float, float, float]] = []
        for ix, iy in self.blocked_cells:
            x0 = self.origin[0] + ix * self.cell_size
            y0 = self.origin[1] + iy * self.cell_size
            bounds.append((x0, x0 + self.cell_size, y0, y0 + self.cell_size))
        return bounds

    def blocked_boxes(
        self,
        z_min: float = 0.0,
        z_max: float = 2.0,
    ) -> list[tuple[float, float, float, float, float, float]]:
        """Return blocked cells as 3D boxes (x0, y0, z0, x1, y1, z1)."""
        z_min = float(z_min)
        z_max = float(z_max)
        if z_max < z_min:
            z_min, z_max = z_max, z_min
        boxes: list[tuple[float, float, float, float, float, float]] = []
        for x0, x1, y0, y1 in self.blocked_bounds():
            boxes.append((x0, y0, z_min, x1, y1, z_max))
        return boxes

    @property
    def has_transport_model(self) -> bool:
        """Return True when known transport components are attached."""
        return bool(self.transport_boxes_m)

    def transport_boxes(self) -> list[tuple[float, float, float, float, float, float]]:
        """Return known obstacle transport boxes in meters."""
        return [tuple(box) for box in self.transport_boxes_m]

    def transport_mu_values(self, isotope: str) -> tuple[float, ...] | None:
        """Return per-transport-box attenuation coefficients for an isotope."""
        if not self.transport_mu_by_isotope:
            return None
        if isotope in self.transport_mu_by_isotope:
            return self.transport_mu_by_isotope[isotope]
        normalized = {
            _normalize_isotope_key(key): values
            for key, values in self.transport_mu_by_isotope.items()
        }
        return normalized.get(_normalize_isotope_key(isotope))

    def transport_line_mu_values(
        self,
        isotope: str,
    ) -> tuple[tuple[float, ...], ...] | None:
        """Return per-line, per-box attenuation coefficients for an isotope."""
        if not self.transport_line_mu_by_isotope:
            return None
        if isotope in self.transport_line_mu_by_isotope:
            return self.transport_line_mu_by_isotope[isotope]
        normalized = {
            _normalize_isotope_key(key): values
            for key, values in self.transport_line_mu_by_isotope.items()
        }
        return normalized.get(_normalize_isotope_key(isotope))

    def with_transport_model(
        self,
        *,
        boxes_m: Iterable[Sequence[float]],
        mu_by_isotope: dict[str, Sequence[float]],
        line_mu_by_isotope: dict[str, Sequence[Sequence[float]]] | None = None,
    ) -> "ObstacleGrid":
        """Return a copy with known obstacle transport components attached."""
        return ObstacleGrid(
            origin=self.origin,
            cell_size=self.cell_size,
            grid_shape=self.grid_shape,
            blocked_cells=self.blocked_cells,
            transport_boxes_m=tuple(tuple(float(value) for value in box) for box in boxes_m),
            transport_mu_by_isotope={
                str(isotope): tuple(float(value) for value in values)
                for isotope, values in mu_by_isotope.items()
            },
            transport_line_mu_by_isotope={
                str(isotope): tuple(
                    tuple(float(value) for value in row) for row in rows
                )
                for isotope, rows in (line_mu_by_isotope or {}).items()
            },
        )

    def blocked_polygons(
        self, z: float = 0.0
    ) -> list[list[tuple[float, float, float]]]:
        """Return polygons for blocked cells at the given z-plane."""
        polygons: list[list[tuple[float, float, float]]] = []
        for x0, x1, y0, y1 in self.blocked_bounds():
            polygons.append([(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)])
        return polygons

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation of the grid."""
        return {
            "version": 1,
            "origin": [self.origin[0], self.origin[1]],
            "cell_size": self.cell_size,
            "grid_shape": [self.grid_shape[0], self.grid_shape[1]],
            "blocked_cells": [list(cell) for cell in self.blocked_cells],
            "blocked_fraction": self.blocked_fraction,
            "transport_boxes_m": [list(box) for box in self.transport_boxes_m],
            "transport_mu_by_isotope": {
                isotope: [float(value) for value in values]
                for isotope, values in sorted(self.transport_mu_by_isotope.items())
            },
            "transport_line_mu_by_isotope": {
                isotope: [[float(value) for value in row] for row in rows]
                for isotope, rows in sorted(self.transport_line_mu_by_isotope.items())
            },
        }

    def save(self, path: Path) -> None:
        """Save the obstacle layout to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict) -> ObstacleGrid:
        """Construct an ObstacleGrid from a dictionary payload."""
        if not isinstance(data, dict):
            raise ValueError("Obstacle layout must be a dict.")
        origin = data.get("origin", (0.0, 0.0))
        cell_size = data.get("cell_size", 1.0)
        grid_shape = data.get("grid_shape")
        blocked_cells = data.get("blocked_cells", [])
        transport_boxes = data.get("transport_boxes_m", [])
        transport_mu_by_isotope = data.get("transport_mu_by_isotope", {})
        transport_line_mu_by_isotope = data.get("transport_line_mu_by_isotope", {})
        if grid_shape is None:
            raise ValueError("Obstacle layout missing 'grid_shape'.")
        if not isinstance(blocked_cells, list):
            raise ValueError("blocked_cells must be a list.")
        if not isinstance(transport_boxes, list):
            raise ValueError("transport_boxes_m must be a list.")
        if not isinstance(transport_mu_by_isotope, dict):
            raise ValueError("transport_mu_by_isotope must be a dict.")
        if not isinstance(transport_line_mu_by_isotope, dict):
            raise ValueError("transport_line_mu_by_isotope must be a dict.")
        return cls(
            origin=(float(origin[0]), float(origin[1])),
            cell_size=float(cell_size),
            grid_shape=(int(grid_shape[0]), int(grid_shape[1])),
            blocked_cells=tuple((int(cell[0]), int(cell[1])) for cell in blocked_cells),
            transport_boxes_m=tuple(
                tuple(float(value) for value in box)
                for box in transport_boxes
            ),
            transport_mu_by_isotope={
                str(isotope): tuple(float(value) for value in values)
                for isotope, values in transport_mu_by_isotope.items()
            },
            transport_line_mu_by_isotope={
                str(isotope): tuple(
                    tuple(float(value) for value in row) for row in rows
                )
                for isotope, rows in transport_line_mu_by_isotope.items()
            },
        )

    @classmethod
    def load(cls, path: Path) -> ObstacleGrid:
        """Load an obstacle layout from a JSON file."""
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)


def _point_to_cell_index(
    point: Sequence[float],
    origin: tuple[float, float],
    cell_size: float,
    grid_shape: tuple[int, int],
) -> tuple[int, int] | None:
    """Return the grid cell index for a point or None if outside."""
    if len(point) < 2:
        raise ValueError("point must have at least two coordinates.")
    x = float(point[0])
    y = float(point[1])
    rel_x = x - origin[0]
    rel_y = y - origin[1]
    if rel_x < 0.0 or rel_y < 0.0:
        return None
    ix = int(np.floor(rel_x / cell_size))
    iy = int(np.floor(rel_y / cell_size))
    if ix < 0 or iy < 0:
        return None
    if ix >= grid_shape[0] or iy >= grid_shape[1]:
        return None
    return ix, iy


def _cell_center(
    cell: tuple[int, int],
    origin: tuple[float, float],
    cell_size: float,
) -> tuple[float, float]:
    """Return the center point of a grid cell."""
    return (
        origin[0] + (float(cell[0]) + 0.5) * cell_size,
        origin[1] + (float(cell[1]) + 0.5) * cell_size,
    )


def _default_passage_points(
    *,
    keep_free_cells: set[tuple[int, int]],
    origin: tuple[float, float],
    cell_size: float,
    grid_shape: tuple[int, int],
) -> list[tuple[float, float]]:
    """Return default passage waypoints through the grid."""
    nx, ny = grid_shape
    if nx <= 0 or ny <= 0:
        return []
    corners = [(0, 0), (nx - 1, 0), (0, ny - 1), (nx - 1, ny - 1)]
    if keep_free_cells:
        start = sorted(keep_free_cells)[0]
        goal = max(
            corners,
            key=lambda cell: abs(cell[0] - start[0]) + abs(cell[1] - start[1]),
        )
    else:
        start = (0, 0)
        goal = (nx - 1, ny - 1)
    return [
        _cell_center(start, origin, cell_size),
        _cell_center(goal, origin, cell_size),
    ]


def _random_manhattan_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Return a randomized 4-connected path between two grid cells."""
    x, y = start
    gx, gy = goal
    path = [(x, y)]
    while (x, y) != (gx, gy):
        can_step_x = x != gx
        can_step_y = y != gy
        if can_step_x and can_step_y:
            step_x = bool(rng.integers(0, 2))
        else:
            step_x = can_step_x
        if step_x:
            x += 1 if gx > x else -1
        else:
            y += 1 if gy > y else -1
        path.append((x, y))
    return path


def _expand_cells(
    cells: Iterable[tuple[int, int]],
    *,
    width_cells: int,
    grid_shape: tuple[int, int],
) -> set[tuple[int, int]]:
    """Return cells expanded to reserve a corridor with the requested width."""
    nx, ny = grid_shape
    width = max(1, int(width_cells))
    lo = (width - 1) // 2
    hi = width // 2
    expanded: set[tuple[int, int]] = set()
    for ix, iy in cells:
        for dx in range(-lo, hi + 1):
            for dy in range(-lo, hi + 1):
                cx = int(ix) + dx
                cy = int(iy) + dy
                if 0 <= cx < nx and 0 <= cy < ny:
                    expanded.add((cx, cy))
    return expanded


def _coverage_line_indices(cell_count: int) -> list[int]:
    """Return grid-line indices that keep coverage gaps bounded."""
    count = int(cell_count)
    if count <= 0:
        return []
    if count <= 2:
        return list(range(count))
    indices = list(range(0, count, EXPLORATION_BACKBONE_MAX_GAP_CELLS))
    midpoint = count // 2
    if midpoint not in indices:
        indices.append(midpoint)
    if indices[-1] != count - 1:
        indices.append(count - 1)
    return sorted(set(indices))


def _manhattan_cells_between(
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    """Return a deterministic 4-connected path between two grid cells."""
    x, y = int(start[0]), int(start[1])
    gx, gy = int(goal[0]), int(goal[1])
    cells = [(x, y)]
    while x != gx:
        x += 1 if gx > x else -1
        cells.append((x, y))
    while y != gy:
        y += 1 if gy > y else -1
        cells.append((x, y))
    return cells


def _exploration_backbone_cells(
    *,
    grid_shape: tuple[int, int],
    keep_free_cells: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Return a connected sparse grid backbone for whole-environment traversal."""
    nx, ny = grid_shape
    if nx <= 0 or ny <= 0:
        return set()
    xs = _coverage_line_indices(nx)
    ys = _coverage_line_indices(ny)
    backbone: set[tuple[int, int]] = set()
    for ix in xs:
        for iy in range(ny):
            backbone.add((ix, iy))
    for iy in ys:
        for ix in range(nx):
            backbone.add((ix, iy))
    anchor = sorted(backbone)[0]
    for cell in keep_free_cells:
        if 0 <= cell[0] < nx and 0 <= cell[1] < ny:
            nearest = min(
                backbone,
                key=lambda candidate: abs(candidate[0] - cell[0])
                + abs(candidate[1] - cell[1]),
            )
            backbone.update(_manhattan_cells_between(cell, nearest))
            backbone.add(cell)
    backbone.update(_manhattan_cells_between((0, 0), anchor))
    return backbone


def _passage_cells_from_points(
    points: Iterable[Sequence[float]],
    *,
    origin: tuple[float, float],
    cell_size: float,
    grid_shape: tuple[int, int],
    width_cells: int,
    rng: np.random.Generator,
) -> set[tuple[int, int]]:
    """Return reserved cells for a passable corridor through waypoints."""
    waypoint_cells: list[tuple[int, int]] = []
    for point in points:
        idx = _point_to_cell_index(point, origin, cell_size, grid_shape)
        if idx is not None and (not waypoint_cells or waypoint_cells[-1] != idx):
            waypoint_cells.append(idx)
    if len(waypoint_cells) < 2:
        return set(waypoint_cells)
    path_cells: list[tuple[int, int]] = []
    for start, goal in zip(waypoint_cells[:-1], waypoint_cells[1:]):
        segment = _random_manhattan_path(start, goal, rng)
        path_cells.extend(segment if not path_cells else segment[1:])
    return _expand_cells(path_cells, width_cells=width_cells, grid_shape=grid_shape)


def generate_obstacle_grid(
    size_x: float,
    size_y: float,
    *,
    cell_size: float = 1.0,
    blocked_fraction: float = 0.4,
    origin: tuple[float, float] = (0.0, 0.0),
    rng: np.random.Generator | None = None,
    keep_free_points: Iterable[Sequence[float]] | None = None,
    passage_points: Iterable[Sequence[float]] | None = None,
    passage_width_m: float = 0.0,
) -> ObstacleGrid:
    """
    Generate a random obstacle layout by blocking a fraction of grid cells.

    The grid is defined on the z=0 plane with 1 m x 1 m cells by default.
    """
    if cell_size <= 0.0:
        raise ValueError("cell_size must be positive.")
    if blocked_fraction < 0.0 or blocked_fraction > 1.0:
        raise ValueError("blocked_fraction must be between 0 and 1.")
    extent_x = max(0.0, float(size_x) - float(origin[0]))
    extent_y = max(0.0, float(size_y) - float(origin[1]))
    nx = int(np.floor(extent_x / cell_size))
    ny = int(np.floor(extent_y / cell_size))
    if nx <= 0 or ny <= 0:
        raise ValueError("Environment size is too small for the requested grid.")
    total = int(nx * ny)
    target = int(np.round(total * blocked_fraction))
    target = max(0, min(target, total))
    rng = np.random.default_rng() if rng is None else rng
    keep_free_cells: set[tuple[int, int]] = set()
    if keep_free_points is not None:
        for pt in keep_free_points:
            idx = _point_to_cell_index(pt, origin, cell_size, (nx, ny))
            if idx is not None:
                keep_free_cells.add(idx)
    width_cells = max(
        1,
        int(np.ceil(max(float(passage_width_m), cell_size) / cell_size)),
    )
    reserved_cells = _expand_cells(
        _exploration_backbone_cells(
            grid_shape=(nx, ny),
            keep_free_cells=keep_free_cells,
        ),
        width_cells=width_cells,
        grid_shape=(nx, ny),
    )
    reserved_cells.update(keep_free_cells)
    if passage_points is not None:
        waypoints = (
            list(passage_points)
        )
        reserved_cells.update(
            _passage_cells_from_points(
                waypoints,
                origin=origin,
                cell_size=cell_size,
                grid_shape=(nx, ny),
                width_cells=width_cells,
                rng=rng,
            )
        )
    all_indices = np.arange(total)
    if reserved_cells:
        keep_flat = np.array(
            [cell[0] + cell[1] * nx for cell in reserved_cells], dtype=int
        )
        mask = np.ones(total, dtype=bool)
        mask[keep_flat] = False
        available = all_indices[mask]
    else:
        available = all_indices
    target = min(target, int(available.size))
    if target > 0:
        selected = rng.choice(available, size=target, replace=False)
    else:
        selected = np.array([], dtype=int)
    blocked_cells = [(int(idx % nx), int(idx // nx)) for idx in selected]
    blocked_cells.sort()
    return ObstacleGrid(
        origin=origin,
        cell_size=cell_size,
        grid_shape=(nx, ny),
        blocked_cells=tuple(blocked_cells),
    )


def load_or_generate_obstacle_grid(
    path: Path,
    *,
    size_x: float,
    size_y: float,
    cell_size: float = 1.0,
    blocked_fraction: float = 0.4,
    rng_seed: int | None = None,
    keep_free_points: Iterable[Sequence[float]] | None = None,
    passage_points: Iterable[Sequence[float]] | None = None,
    passage_width_m: float = 0.0,
) -> ObstacleGrid:
    """
    Load a layout from disk, or generate and save one when missing.
    """
    if path.exists():
        return ObstacleGrid.load(path)
    rng = np.random.default_rng(rng_seed)
    grid = generate_obstacle_grid(
        size_x=size_x,
        size_y=size_y,
        cell_size=cell_size,
        blocked_fraction=blocked_fraction,
        origin=(0.0, 0.0),
        rng=rng,
        keep_free_points=keep_free_points,
        passage_points=passage_points,
        passage_width_m=passage_width_m,
    )
    grid.save(path)
    return grid


def build_obstacle_grid(
    *,
    mode: Literal["fixed", "random"],
    path: Path | None,
    size_x: float,
    size_y: float,
    cell_size: float = 1.0,
    blocked_fraction: float = 0.4,
    rng_seed: int | None = None,
    keep_free_points: Iterable[Sequence[float]] | None = None,
    passage_points: Iterable[Sequence[float]] | None = None,
    passage_width_m: float = 0.0,
) -> ObstacleGrid:
    """
    Build an obstacle grid in fixed or random mode.

    Fixed mode keeps the current JSON-backed workflow by loading the layout from
    disk or generating it once when the file does not exist. Random mode always
    creates a fresh in-memory layout for the current run and never writes it to
    disk.
    """
    normalized_mode = mode.strip().lower()
    if normalized_mode == "fixed":
        if path is None:
            raise ValueError("path is required when mode is 'fixed'.")
        return load_or_generate_obstacle_grid(
            path,
            size_x=size_x,
            size_y=size_y,
            cell_size=cell_size,
            blocked_fraction=blocked_fraction,
            rng_seed=rng_seed,
            keep_free_points=keep_free_points,
            passage_points=passage_points,
            passage_width_m=passage_width_m,
        )
    if normalized_mode == "random":
        rng = np.random.default_rng(rng_seed)
        return generate_obstacle_grid(
            size_x=size_x,
            size_y=size_y,
            cell_size=cell_size,
            blocked_fraction=blocked_fraction,
            origin=(0.0, 0.0),
            rng=rng,
            keep_free_points=keep_free_points,
            passage_points=passage_points,
            passage_width_m=passage_width_m,
        )
    raise ValueError(f"Unknown obstacle grid mode: {mode}")
