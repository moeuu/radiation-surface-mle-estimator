"""Shared runtime helpers for obstacle environment setup."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from measurement.obstacle_assets import (
    KnownObstacleInstance,
    generate_manchester_obstacle_instances,
)
from measurement.obstacles import ObstacleGrid, build_obstacle_grid
from sim.blender_environment import attach_known_obstacle_transport_model

EnvironmentMode = Literal["fixed", "random"]


@dataclass(frozen=True)
class RuntimeObstacleEnvironment:
    """Obstacle grid, generated assets, and diagnostics for one runtime."""

    grid: ObstacleGrid | None
    mode: EnvironmentMode
    layout_path: Path | None
    known_obstacle_instances: tuple[KnownObstacleInstance, ...] | None
    message: str | None

    def template_counts(self) -> dict[str, int]:
        """Return counts by generated obstacle asset template."""
        counts: dict[str, int] = {}
        if self.known_obstacle_instances is None:
            return counts
        for instance in self.known_obstacle_instances:
            counts[instance.template] = counts.get(instance.template, 0) + 1
        return counts

    def asset_summary(self) -> str | None:
        """Return a human-readable summary of generated obstacle assets."""
        if self.grid is None or self.known_obstacle_instances is None:
            return None
        return (
            "Known Manchester-style obstacle assets: "
            f"instances={len(self.known_obstacle_instances)}, "
            f"transport_components={len(self.grid.transport_boxes_m)}, "
            f"templates={self.template_counts()}"
        )


def normalize_environment_mode(environment_mode: str) -> EnvironmentMode:
    """Normalize and validate an obstacle environment mode string."""
    normalized = str(environment_mode).strip().lower()
    if normalized == "fixed":
        return "fixed"
    if normalized == "random":
        return "random"
    raise ValueError(f"Unknown environment_mode: {environment_mode}")


def resolve_runtime_path(root: Path, path: str | Path | None) -> Path | None:
    """Resolve a possibly repo-relative runtime path."""
    if path is None or str(path) == "":
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (Path(root) / candidate).resolve()


def _obstacle_mode_message(
    *,
    mode: EnvironmentMode,
    path: Path | None,
    seed: int | None,
    passage_width_m: float,
    grid: ObstacleGrid,
) -> str:
    """Return a concise runtime obstacle environment log message."""
    message = f"Obstacle environment mode: {mode}"
    if mode == "fixed" and path is not None:
        message += f" ({path})"
    if seed is not None:
        message += f", seed={int(seed)}"
    if mode == "random" and passage_width_m > 0.0:
        message += f", passage_width_m={float(passage_width_m):.2f}"
    message += f", blocked_fraction={grid.blocked_fraction:.3f}"
    return message


def build_runtime_obstacle_environment(
    *,
    root: Path,
    environment_mode: str,
    obstacle_layout_path: str | Path | None,
    room_size_xyz: tuple[float, float, float],
    detector_position_xy: Sequence[float] | None,
    obstacle_seed: int | None = None,
    cell_size: float = 1.0,
    blocked_fraction: float = 0.4,
    passage_width_m: float = 0.0,
    attach_known_transport: bool = False,
    obstacle_height_m: float = 2.0,
    include_room_boundaries: bool = False,
    room_boundary_thickness_m: float = 0.1,
) -> RuntimeObstacleEnvironment:
    """Build the obstacle grid and optional known-object transport model."""
    mode = normalize_environment_mode(environment_mode)
    obstacle_path = resolve_runtime_path(root, obstacle_layout_path)
    if obstacle_path is None:
        return RuntimeObstacleEnvironment(
            grid=None,
            mode=mode,
            layout_path=None,
            known_obstacle_instances=None,
            message=None,
        )
    keep_free = None
    if detector_position_xy is not None:
        keep_free = [(float(detector_position_xy[0]), float(detector_position_xy[1]))]
    active_passage_width_m = float(passage_width_m) if mode == "random" else 0.0
    grid = build_obstacle_grid(
        mode=mode,
        path=obstacle_path,
        size_x=float(room_size_xyz[0]),
        size_y=float(room_size_xyz[1]),
        cell_size=float(cell_size),
        blocked_fraction=float(blocked_fraction),
        rng_seed=obstacle_seed,
        keep_free_points=keep_free,
        passage_width_m=active_passage_width_m,
    )
    known_obstacle_instances: tuple[KnownObstacleInstance, ...] | None = None
    if attach_known_transport and mode == "random":
        known_obstacle_instances = generate_manchester_obstacle_instances(
            grid,
            room_size_xyz=room_size_xyz,
            obstacle_height_m=float(obstacle_height_m),
            rng_seed=obstacle_seed,
        )
        grid = attach_known_obstacle_transport_model(
            grid,
            instances=known_obstacle_instances,
            room_size_xyz=room_size_xyz,
            include_room_boundaries=include_room_boundaries,
            room_boundary_thickness_m=room_boundary_thickness_m,
        )
    return RuntimeObstacleEnvironment(
        grid=grid,
        mode=mode,
        layout_path=obstacle_path,
        known_obstacle_instances=known_obstacle_instances,
        message=_obstacle_mode_message(
            mode=mode,
            path=obstacle_path,
            seed=obstacle_seed,
            passage_width_m=active_passage_width_m,
            grid=grid,
        ),
    )
