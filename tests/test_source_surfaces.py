"""Tests for surface-constrained random source placement."""

from __future__ import annotations

import numpy as np

from measurement.model import EnvironmentConfig
from measurement.continuous_kernels import segment_box_intersection_length_m
from measurement.obstacles import ObstacleGrid
from measurement.source_surfaces import (
    _segment_path_lengths_through_boxes_m,
    build_surface_candidate_sources,
    generate_surface_sources,
    is_ground_observable_source_position,
    is_allowed_source_surface_position,
    project_positions_to_allowed_surfaces,
    source_surface_kind_counts,
    source_surface_kind,
    source_surface_kinds,
    surface_observable_fractions,
    surface_response_observability_diagnostics,
    transport_interior_mask,
)


def test_generate_surface_sources_never_places_sources_in_air_or_obstacles() -> None:
    """Random source generation should only place sources on allowed surfaces."""
    env = EnvironmentConfig(size_x=10.0, size_y=20.0, size_z=10.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(10, 20),
        blocked_cells=((3, 4), (4, 4), (3, 5)),
    )
    sources = generate_surface_sources(
        env=env,
        obstacle_grid=grid,
        isotopes=("Cs-137", "Co-60", "Eu-154"),
        intensity_cps_1m=30000.0,
        rng=np.random.default_rng(4),
        count=200,
        obstacle_height_m=2.0,
    )

    assert len(sources) == 200
    for source in sources:
        assert is_allowed_source_surface_position(
            source.position,
            env,
            grid,
            obstacle_height_m=2.0,
        )


def test_generate_surface_sources_samples_random_intensity_range() -> None:
    """Random source generation should support randomized source strengths."""
    env = EnvironmentConfig(size_x=4.0, size_y=4.0, size_z=3.0)
    sources = generate_surface_sources(
        env=env,
        obstacle_grid=None,
        isotopes=("Cs-137",),
        intensity_cps_1m=(100000.0, 200000.0),
        rng=np.random.default_rng(123),
        count=4,
    )
    strengths = [source.intensity_cps_1m for source in sources]
    assert all(100000.0 <= value <= 200000.0 for value in strengths)
    assert len({round(value, 6) for value in strengths}) > 1


def test_generate_surface_sources_separates_same_isotope_sources() -> None:
    """Random source generation should space repeated isotopes apart."""
    env = EnvironmentConfig(size_x=10.0, size_y=20.0, size_z=10.0)
    min_distance_m = 3.0
    sources = generate_surface_sources(
        env=env,
        obstacle_grid=None,
        isotopes=("Cs-137", "Cs-137", "Co-60", "Cs-137", "Co-60"),
        intensity_cps_1m=30000.0,
        rng=np.random.default_rng(20260603),
        count=5,
        preferred_max_z_m=None,
        same_isotope_min_distance_m=min_distance_m,
    )

    for isotope in {"Cs-137", "Co-60"}:
        positions = np.asarray(
            [source.position for source in sources if source.isotope == isotope],
            dtype=float,
        )
        for left in range(positions.shape[0]):
            for right in range(left + 1, positions.shape[0]):
                distance = float(np.linalg.norm(positions[left] - positions[right]))
                assert distance >= min_distance_m - 1.0e-9


def test_generate_surface_sources_caps_ceiling_and_prefers_low_z() -> None:
    """Random scenarios should avoid many ceiling or high-wall truth sources."""
    env = EnvironmentConfig(size_x=10.0, size_y=20.0, size_z=10.0)
    sources = generate_surface_sources(
        env=env,
        obstacle_grid=None,
        isotopes=("Cs-137", "Co-60", "Eu-154"),
        intensity_cps_1m=30000.0,
        rng=np.random.default_rng(20260530),
        count=120,
    )
    positions = np.asarray([source.position for source in sources], dtype=float)
    counts = source_surface_kind_counts(positions, env, None)

    assert counts["ceiling"] <= 1
    assert float(np.max(positions[:, 2])) <= 5.0


def test_generate_surface_sources_ceiling_cap_without_low_z_preference() -> None:
    """Ceiling caps should be independent from the preferred-height filter."""
    env = EnvironmentConfig(size_x=10.0, size_y=20.0, size_z=10.0)
    sources = generate_surface_sources(
        env=env,
        obstacle_grid=None,
        isotopes=("Cs-137",),
        intensity_cps_1m=30000.0,
        rng=np.random.default_rng(11),
        count=120,
        max_ceiling_sources=0,
        preferred_max_z_m=None,
    )
    positions = np.asarray([source.position for source in sources], dtype=float)
    counts = source_surface_kind_counts(positions, env, None)

    assert counts["ceiling"] == 0


def test_surface_observable_fraction_rejects_obstacle_top_sources() -> None:
    """Visibility screening should reject sources hidden by their support obstacle."""
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(4, 4),
        blocked_cells=((1, 1),),
    )
    measurement_points = np.array(
        [
            [ix + 0.5, iy + 0.5, 0.5]
            for ix in range(4)
            for iy in range(4)
            if grid.is_cell_free((ix, iy))
        ],
        dtype=float,
    )
    positions = np.array(
        [
            [1.5, 1.5, 1.0],
            [0.5, 0.5, 0.0],
            [1.0, 1.5, 0.5],
        ],
        dtype=float,
    )

    fractions = surface_observable_fractions(
        positions,
        grid,
        measurement_points,
        obstacle_height_m=1.0,
        detector_height_m=0.5,
    )

    assert fractions[0] < 0.1
    assert fractions[1] > 0.5
    assert fractions[2] > fractions[0]
    assert not is_ground_observable_source_position(
        positions[0],
        grid,
        measurement_points,
        min_visible_fraction=0.5,
        obstacle_height_m=1.0,
    )


def test_batched_visibility_path_lengths_match_scalar_box_intersections() -> None:
    """Batched source-visibility geometry should match the scalar box oracle."""
    sources = np.array(
        [
            [0.0, 0.5, 0.5],
            [1.5, 1.5, 1.0],
        ],
        dtype=float,
    )
    detectors = np.array(
        [
            [3.0, 0.5, 0.5],
            [0.5, 3.0, 0.5],
        ],
        dtype=float,
    )
    boxes = np.array(
        [
            [1.0, 0.0, 0.0, 2.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 2.0, 2.0, 1.0],
        ],
        dtype=float,
    )

    batched = _segment_path_lengths_through_boxes_m(
        sources,
        detectors,
        boxes,
        box_chunk_size=1,
    )
    scalar = np.zeros_like(batched)
    for source_index, source in enumerate(sources):
        for detector_index, detector in enumerate(detectors):
            scalar[source_index, detector_index] = sum(
                segment_box_intersection_length_m(source, detector, box)
                for box in boxes
            )

    assert np.allclose(batched, scalar)


def test_generate_surface_sources_respects_ground_visibility_filter() -> None:
    """Random source generation should avoid mostly occluded surface locations."""
    env = EnvironmentConfig(size_x=4.0, size_y=4.0, size_z=3.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(4, 4),
        blocked_cells=((1, 1),),
    )
    measurement_points = np.array(
        [
            [ix + 0.5, iy + 0.5, 0.5]
            for ix in range(4)
            for iy in range(4)
            if grid.is_cell_free((ix, iy))
        ],
        dtype=float,
    )

    sources = generate_surface_sources(
        env=env,
        obstacle_grid=grid,
        isotopes=("Cs-137",),
        intensity_cps_1m=30000.0,
        rng=np.random.default_rng(12),
        count=30,
        obstacle_height_m=1.0,
        visibility_measurement_points=measurement_points,
        visibility_min_fraction=0.5,
        visibility_detector_height_m=0.5,
        visibility_batch_size=64,
        visibility_max_attempts_per_source=1024,
    )
    fractions = surface_observable_fractions(
        np.asarray([source.position for source in sources], dtype=float),
        grid,
        measurement_points,
        obstacle_height_m=1.0,
        detector_height_m=0.5,
    )

    assert len(sources) == 30
    assert np.min(fractions) >= 0.5
    assert not any(
        source_surface_kind(source.position, env, grid, obstacle_height_m=1.0)
        == "obstacle_top"
        for source in sources
    )


def test_surface_response_observability_diagnostics_are_batched() -> None:
    """Response observability screening should expose source-set conditioning."""
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(5, 5),
        blocked_cells=(),
    )
    measurement_points = np.array(
        [
            [0.5, 0.5, 0.5],
            [4.5, 0.5, 0.5],
            [0.5, 4.5, 0.5],
            [4.5, 4.5, 0.5],
        ],
        dtype=float,
    )
    separated = np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 5.0, 3.0],
        ],
        dtype=float,
    )
    duplicated = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=float,
    )

    separated_stats = surface_response_observability_diagnostics(
        separated,
        grid,
        measurement_points,
        obstacle_height_m=1.0,
    )
    duplicated_stats = surface_response_observability_diagnostics(
        duplicated,
        grid,
        measurement_points,
        isotopes=("Cs-137", "Cs-137"),
        obstacle_height_m=1.0,
    )

    assert separated_stats["source_count"] == 2
    assert separated_stats["reference_point_count"] == 4
    assert float(separated_stats["condition_number"]) >= 1.0
    assert float(separated_stats["max_pairwise_correlation"]) < 0.999
    assert float(duplicated_stats["max_pairwise_correlation"]) > 0.999999
    assert int(duplicated_stats["same_isotope_pair_count"]) == 1
    assert (
        float(duplicated_stats["same_isotope_max_pairwise_correlation"])
        > 0.999999
    )


def test_source_surface_kind_rejects_air_and_obstacle_interior() -> None:
    """Surface classification should reject unsupported 3D positions."""
    env = EnvironmentConfig(size_x=10.0, size_y=20.0, size_z=10.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(10, 20),
        blocked_cells=((3, 4),),
    )

    assert source_surface_kind((5.0, 5.0, 5.0), env, grid) is None
    assert source_surface_kind((3.5, 4.5, 1.0), env, grid) is None
    assert source_surface_kind((3.5, 4.5, 2.0), env, grid) == "obstacle_top"
    assert source_surface_kind((3.0, 4.5, 1.0), env, grid) == "obstacle_side"
    assert source_surface_kind((1.5, 1.5, 0.0), env, grid) == "floor"
    assert source_surface_kind((5.0, 20.0, 4.0), env, grid) == "wall"


def test_transport_component_interior_is_not_allowed_source_support() -> None:
    """Known transport-box interiors should not become source support."""
    env = EnvironmentConfig(size_x=4.0, size_y=4.0, size_z=3.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(4, 4),
        blocked_cells=((1, 1),),
    ).with_transport_model(
        boxes_m=((1.2, 1.2, 0.2, 1.8, 1.8, 1.4),),
        mu_by_isotope={"Cs-137": (0.1,)},
    )
    points = np.asarray(
        [
            (1.5, 1.5, 0.8),
            (1.0, 1.5, 0.8),
            (0.5, 0.5, 0.0),
        ],
        dtype=float,
    )

    mask = transport_interior_mask(points, grid)

    assert mask.tolist() == [True, False, False]
    assert not is_allowed_source_surface_position((1.5, 1.5, 0.8), env, grid)
    assert is_allowed_source_surface_position((1.0, 1.5, 0.8), env, grid)


def test_source_surface_kinds_matches_scalar_classification() -> None:
    """Vectorized surface classification should match the scalar oracle."""
    env = EnvironmentConfig(size_x=4.0, size_y=4.0, size_z=3.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(4, 4),
        blocked_cells=((2, 2),),
    )
    points = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 4.0, 2.0],
            [2.5, 2.5, 1.0],
            [2.0, 2.5, 0.5],
            [2.5, 2.5, 0.5],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )

    vectorized = source_surface_kinds(
        points,
        env,
        grid,
        obstacle_height_m=1.0,
    )
    scalar = np.array(
        [
            source_surface_kind(point, env, grid, obstacle_height_m=1.0)
            for point in points
        ],
        dtype=object,
    )
    counts = source_surface_kind_counts(
        points,
        env,
        grid,
        obstacle_height_m=1.0,
    )

    assert vectorized.tolist() == scalar.tolist()
    assert counts["floor"] == 1
    assert counts["wall"] == 1
    assert counts["obstacle_top"] == 1
    assert counts["obstacle_side"] == 1
    assert counts["off_surface"] == 2


def test_build_surface_candidate_sources_only_returns_allowed_surfaces() -> None:
    """Surface candidate generation should cover known room and obstacle surfaces."""
    env = EnvironmentConfig(size_x=2.0, size_y=2.0, size_z=2.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(2, 2),
        blocked_cells=((1, 1),),
    )

    candidates = build_surface_candidate_sources(
        env,
        grid,
        spacing=(0.5, 0.5, 0.5),
        obstacle_height_m=1.0,
    )

    assert candidates.shape[1] == 3
    assert candidates.shape[0] > 0
    assert not np.any(np.all(np.isclose(candidates, (1.5, 1.5, 0.0)), axis=1))
    kinds = {
        source_surface_kind(point, env, grid, obstacle_height_m=1.0)
        for point in candidates
    }
    assert None not in kinds
    assert "wall" in kinds
    assert "floor" in kinds
    assert "ceiling" in kinds
    assert "obstacle_top" in kinds or "obstacle_side" in kinds


def test_project_positions_to_allowed_surfaces_uses_nearest_known_surface() -> None:
    """Position projection should snap off-surface positions to allowed surfaces."""
    env = EnvironmentConfig(size_x=4.0, size_y=4.0, size_z=3.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(4, 4),
        blocked_cells=((2, 2),),
    )
    positions = np.array(
        [
            [2.5, 2.5, 0.6],
            [3.9, 2.0, 1.0],
            [1.2, 1.4, 1.4],
        ],
        dtype=float,
    )

    projected = project_positions_to_allowed_surfaces(
        positions,
        env,
        grid,
        obstacle_height_m=1.0,
    )

    assert projected.shape == positions.shape
    assert np.allclose(projected[0], [2.5, 2.5, 1.0])
    for point in projected:
        assert is_allowed_source_surface_position(
            point,
            env,
            grid,
            obstacle_height_m=1.0,
        )
