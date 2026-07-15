"""Tests for robot traversability map generation."""

from pathlib import Path

import numpy as np

from measurement.obstacles import ObstacleGrid
from planning.traversability import (
    TraversabilityMap,
    build_traversability_map_from_obstacle_grid,
    build_traversability_map_from_stage_solids,
    render_traversability_map,
)
from sim.isaacsim_app.stage_backend import PrimPose, StageSolidPrim


def test_traversability_map_projects_obstacle_footprints() -> None:
    """A robot footprint should reject cells too close to projected obstacles."""
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(3, 3),
        blocked_cells=((1, 1),),
    )

    narrow_robot = build_traversability_map_from_obstacle_grid(
        grid,
        robot_radius_m=0.35,
    )
    wide_robot = build_traversability_map_from_obstacle_grid(
        grid,
        robot_radius_m=0.6,
    )

    assert not narrow_robot.is_free((1.5, 1.5))
    assert narrow_robot.is_free((0.5, 1.5))
    assert not wide_robot.is_free((0.5, 1.5))


def test_traversability_shortest_path_avoids_blocked_cells() -> None:
    """Shortest paths should route around obstacle cells instead of drawing a chord."""
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(5, 3),
        blocked_cells=((2, 0), (2, 1)),
    )
    traversable = build_traversability_map_from_obstacle_grid(
        grid,
        robot_radius_m=0.0,
    )

    path = traversable.shortest_path_points(
        (0.5, 0.5, 0.0),
        (4.5, 0.5, 0.0),
    )

    assert path is not None
    assert np.max(path[:, 1]) > 2.0
    assert sum(np.linalg.norm(delta) for delta in np.diff(path, axis=0)) > 4.0
    for waypoint in path:
        assert traversable.is_free(waypoint)


def test_traversability_shortest_path_reports_disconnected_cells() -> None:
    """Disconnected free components should return no path."""
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(5, 3),
        blocked_cells=((2, 0), (2, 1), (2, 2)),
    )
    traversable = build_traversability_map_from_obstacle_grid(
        grid,
        robot_radius_m=0.0,
    )

    path = traversable.shortest_path_points(
        (0.5, 0.5, 0.0),
        (4.5, 0.5, 0.0),
    )

    assert path is None


def test_traversability_map_keeps_reachable_component_only() -> None:
    """Reachable filtering should remove free cells isolated from the robot start."""
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(5, 3),
        blocked_cells=((2, 0), (2, 1), (2, 2)),
    )

    traversable = build_traversability_map_from_obstacle_grid(
        grid,
        robot_radius_m=0.35,
        reachable_from=(0.5, 1.5),
    )

    assert traversable.is_free((0.5, 1.5))
    assert not traversable.is_free((4.5, 1.5))


def test_traversability_map_roundtrip_and_render(tmp_path: Path) -> None:
    """Traversability maps should round-trip and render a PNG for inspection."""
    traversable = TraversabilityMap(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(2, 2),
        traversable_cells=((0, 0), (1, 0)),
        robot_radius_m=0.35,
    )
    json_path = tmp_path / "map.json"
    png_path = tmp_path / "map.png"

    traversable.save(json_path)
    render_traversability_map(traversable, png_path)

    loaded = TraversabilityMap.load(json_path)
    assert loaded == traversable
    assert png_path.exists()
    assert loaded.is_free((0.5, 0.5))
    assert not loaded.is_free((0.5, 1.5))


def test_traversability_map_projects_stage_solids() -> None:
    """USD/Isaac solid geometry should project into the same map API."""
    solids = [
        StageSolidPrim(
            path="/World/Environment/Floor",
            shape="box",
            pose=PrimPose(translation_xyz=(1.5, 1.5, -0.05)),
            size_xyz=(3.0, 3.0, 0.05),
        ),
        StageSolidPrim(
            path="/World/Environment/Wall",
            shape="box",
            pose=PrimPose(translation_xyz=(1.5, 1.5, 0.75)),
            size_xyz=(1.0, 1.0, 1.5),
        ),
        StageSolidPrim(
            path="/World/Environment/MeshCrate",
            shape="mesh",
            pose=PrimPose(),
            triangles_xyz=(
                ((2.0, 0.0, 0.2), (2.8, 0.0, 0.2), (2.0, 0.8, 0.8)),
            ),
        ),
    ]

    traversable = build_traversability_map_from_stage_solids(
        solids,
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(3, 3),
        robot_radius_m=0.35,
        reachable_from=(0.5, 0.5),
        blocking_z_range_m=(0.05, 2.0),
    )

    assert traversable.source == "stage_projected_3d_environment"
    assert traversable.is_free((0.5, 0.5))
    assert not traversable.is_free((1.5, 1.5))
    assert not traversable.is_free((2.5, 0.5))
