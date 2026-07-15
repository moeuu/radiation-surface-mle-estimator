"""Tests for shared runtime obstacle environment setup."""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime_environment import (
    build_runtime_obstacle_environment,
    normalize_environment_mode,
)


def test_random_runtime_obstacle_environment_is_in_memory(tmp_path: Path) -> None:
    """Random mode should build obstacles without writing the fixed-layout path."""
    obstacle_path = tmp_path / "random_unused.json"

    environment = build_runtime_obstacle_environment(
        root=tmp_path,
        environment_mode="random",
        obstacle_layout_path=obstacle_path,
        room_size_xyz=(10.0, 20.0, 10.0),
        detector_position_xy=(1.0, 1.0),
        obstacle_seed=7,
        passage_width_m=1.0,
    )

    assert environment.mode == "random"
    assert environment.grid is not None
    assert environment.grid.blocked_cells
    assert not obstacle_path.exists()
    assert environment.message is not None
    assert "passage_width_m=1.00" in environment.message


def test_random_runtime_environment_can_attach_transport_model(tmp_path: Path) -> None:
    """Runtime random obstacles should optionally expose PF transport boxes."""
    environment = build_runtime_obstacle_environment(
        root=tmp_path,
        environment_mode="random",
        obstacle_layout_path=tmp_path / "random_unused.json",
        room_size_xyz=(10.0, 20.0, 10.0),
        detector_position_xy=(1.0, 1.0),
        obstacle_seed=9,
        attach_known_transport=True,
        obstacle_height_m=2.0,
    )

    assert environment.grid is not None
    assert environment.known_obstacle_instances is not None
    assert environment.grid.transport_boxes_m
    assert environment.asset_summary() is not None


def test_random_runtime_environment_can_attach_room_boundary_transport(
    tmp_path: Path,
) -> None:
    """Runtime transport model should include authored room boundaries when requested."""
    base = build_runtime_obstacle_environment(
        root=tmp_path,
        environment_mode="random",
        obstacle_layout_path=tmp_path / "random_unused.json",
        room_size_xyz=(10.0, 20.0, 10.0),
        detector_position_xy=(1.0, 1.0),
        obstacle_seed=10,
        attach_known_transport=True,
        obstacle_height_m=2.0,
        include_room_boundaries=False,
    )
    with_boundaries = build_runtime_obstacle_environment(
        root=tmp_path,
        environment_mode="random",
        obstacle_layout_path=tmp_path / "random_unused.json",
        room_size_xyz=(10.0, 20.0, 10.0),
        detector_position_xy=(1.0, 1.0),
        obstacle_seed=10,
        attach_known_transport=True,
        obstacle_height_m=2.0,
        include_room_boundaries=True,
        room_boundary_thickness_m=0.1,
    )

    assert base.grid is not None
    assert with_boundaries.grid is not None
    assert len(with_boundaries.grid.transport_boxes_m) == (
        len(base.grid.transport_boxes_m) + 6
    )
    assert any(box[2] < 0.0 for box in with_boundaries.grid.transport_boxes_m)
    assert any(box[5] > 10.0 for box in with_boundaries.grid.transport_boxes_m)


def test_fixed_runtime_obstacle_environment_uses_layout_file(tmp_path: Path) -> None:
    """Fixed mode should load or create the requested obstacle layout file."""
    obstacle_path = tmp_path / "fixed.json"

    environment = build_runtime_obstacle_environment(
        root=tmp_path,
        environment_mode="fixed",
        obstacle_layout_path=obstacle_path,
        room_size_xyz=(10.0, 20.0, 10.0),
        detector_position_xy=(1.0, 1.0),
        obstacle_seed=11,
    )

    assert environment.mode == "fixed"
    assert environment.grid is not None
    assert obstacle_path.exists()
    assert environment.layout_path == obstacle_path


def test_runtime_obstacle_environment_can_be_disabled(tmp_path: Path) -> None:
    """A None obstacle path should preserve the explicit no-obstacle behavior."""
    environment = build_runtime_obstacle_environment(
        root=tmp_path,
        environment_mode="random",
        obstacle_layout_path=None,
        room_size_xyz=(10.0, 20.0, 10.0),
        detector_position_xy=(1.0, 1.0),
    )

    assert environment.mode == "random"
    assert environment.grid is None
    assert environment.message is None


def test_invalid_runtime_environment_mode_is_rejected() -> None:
    """Unknown obstacle environment modes should fail early."""
    with pytest.raises(ValueError, match="Unknown environment_mode"):
        normalize_environment_mode("unsupported")
