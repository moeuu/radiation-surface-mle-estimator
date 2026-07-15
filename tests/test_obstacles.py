"""Tests for obstacle grid generation and serialization."""

from pathlib import Path

import numpy as np

from measurement.obstacles import (
    ObstacleGrid,
    build_obstacle_grid,
    generate_obstacle_grid,
    load_or_generate_obstacle_grid,
)


def test_obstacle_grid_roundtrip_and_is_free(tmp_path: Path) -> None:
    """Obstacle grids should round-trip and block expected cells."""
    rng = np.random.default_rng(0)
    grid = generate_obstacle_grid(
        size_x=4.0,
        size_y=4.0,
        cell_size=1.0,
        blocked_fraction=0.5,
        rng=rng,
    )
    path = tmp_path / "layout.json"
    grid.save(path)
    loaded = ObstacleGrid.load(path)
    assert loaded == grid
    assert loaded.blocked_cells
    ix, iy = loaded.blocked_cells[0]
    x = loaded.origin[0] + ix * loaded.cell_size + 0.1
    y = loaded.origin[1] + iy * loaded.cell_size + 0.1
    assert loaded.is_free((x, y, 0.0)) is False
    assert loaded.is_free((-1.0, -1.0, 0.0)) is True


def test_generate_obstacle_grid_respects_keep_free_points() -> None:
    """Keep-free points should never be blocked."""
    rng = np.random.default_rng(1)
    grid = generate_obstacle_grid(
        size_x=3.0,
        size_y=3.0,
        cell_size=1.0,
        blocked_fraction=0.6,
        rng=rng,
        keep_free_points=[(0.2, 0.2)],
    )
    assert (0, 0) not in grid.blocked_cells


def test_generate_obstacle_grid_reserves_passable_corridor() -> None:
    """Passage waypoints should remain connected even in a fully blocked layout."""
    rng = np.random.default_rng(2)
    grid = generate_obstacle_grid(
        size_x=6.0,
        size_y=6.0,
        cell_size=1.0,
        blocked_fraction=1.0,
        rng=rng,
        passage_points=[(0.5, 0.5), (5.5, 0.5)],
        passage_width_m=2.0,
    )

    assert grid.has_free_path((0.5, 0.5), (5.5, 0.5))
    for ix in range(6):
        assert (ix, 0) not in grid.blocked_cells
        assert (ix, 1) not in grid.blocked_cells


def test_generate_obstacle_grid_reserves_exploration_backbone_by_default() -> None:
    """Generated layouts should keep a sparse whole-room exploration backbone."""
    rng = np.random.default_rng(3)
    grid = generate_obstacle_grid(
        size_x=10.0,
        size_y=20.0,
        cell_size=1.0,
        blocked_fraction=1.0,
        rng=rng,
        keep_free_points=[(1.5, 1.5)],
    )

    anchors = [
        (0.5, 0.5),
        (9.5, 0.5),
        (0.5, 19.5),
        (9.5, 19.5),
        (5.5, 10.5),
    ]
    for anchor in anchors:
        assert grid.has_free_path((1.5, 1.5), anchor)


def test_load_or_generate_obstacle_grid_creates_file(tmp_path: Path) -> None:
    """Missing obstacle layouts should be generated and saved."""
    path = tmp_path / "generated.json"
    grid = load_or_generate_obstacle_grid(
        path,
        size_x=2.0,
        size_y=2.0,
        cell_size=1.0,
        blocked_fraction=0.5,
        rng_seed=0,
    )
    assert path.exists()
    loaded = ObstacleGrid.load(path)
    assert loaded == grid


def test_build_obstacle_grid_fixed_uses_json_layout(tmp_path: Path) -> None:
    """Fixed mode should load the JSON-backed obstacle layout."""
    path = tmp_path / "fixed_layout.json"
    original = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(3, 3),
        blocked_cells=((0, 1), (2, 2)),
    )
    original.save(path)

    loaded = build_obstacle_grid(
        mode="fixed",
        path=path,
        size_x=3.0,
        size_y=3.0,
        rng_seed=123,
    )

    assert loaded == original
    assert ObstacleGrid.load(path) == original


def test_build_obstacle_grid_random_is_ephemeral_and_seeded(tmp_path: Path) -> None:
    """Random mode should create an in-memory layout without writing a file."""
    path = tmp_path / "random_layout.json"

    grid_one = build_obstacle_grid(
        mode="random",
        path=path,
        size_x=6.0,
        size_y=6.0,
        blocked_fraction=0.35,
        rng_seed=7,
    )
    grid_two = build_obstacle_grid(
        mode="random",
        path=path,
        size_x=6.0,
        size_y=6.0,
        blocked_fraction=0.35,
        rng_seed=7,
    )

    assert not path.exists()
    assert grid_one == grid_two
    assert grid_one.blocked_cells


def test_build_obstacle_grid_random_has_default_passage() -> None:
    """Random mode should reserve a passage from the start to a far corner."""
    grid = build_obstacle_grid(
        mode="random",
        path=None,
        size_x=6.0,
        size_y=6.0,
        blocked_fraction=1.0,
        rng_seed=9,
        keep_free_points=[(1.5, 1.5)],
        passage_width_m=1.0,
    )

    assert grid.has_free_path((1.5, 1.5), (5.5, 5.5))
