from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid
from three_d_estimation.surface_patches import (
    build_surface_patches,
    refine_surface_patches,
)
from three_d_estimation.types import MLEEstimate, ObservationBatch, SurfacePatchSet


def test_room_faces_are_exact_oriented_rectangles_with_physical_adjacency() -> None:
    patches = build_surface_patches(
        EnvironmentConfig(size_x=2.0, size_y=3.0, size_z=4.0),
        None,
        spacing=10.0,
        quadrature_points_per_patch=4,
    )

    assert patches.patch_count == 6
    assert Counter(patch.surface_kind for patch in patches.patches) == {
        "floor": 1,
        "ceiling": 1,
        "wall": 4,
    }
    assert patches.total_area_m2 == pytest.approx(2.0 * (2.0 * 3.0 + 2.0 * 4.0 + 3.0 * 4.0))
    assert patches.adjacency_edges.shape == (12, 2)
    assert np.sum(patches.shared_edge_lengths_m) == pytest.approx(
        4.0 * (2.0 + 3.0 + 4.0)
    )

    expected_normals = {
        "room:floor": (0.0, 0.0, 1.0),
        "room:ceiling": (0.0, 0.0, -1.0),
        "room:wall:x_min": (1.0, 0.0, 0.0),
        "room:wall:x_max": (-1.0, 0.0, 0.0),
        "room:wall:y_min": (0.0, 1.0, 0.0),
        "room:wall:y_max": (0.0, -1.0, 0.0),
    }
    for patch in patches.patches:
        np.testing.assert_allclose(patch.normal_xyz, expected_normals[patch.object_id])
        np.testing.assert_allclose(
            patch.centroid_xyz,
            np.mean(patch.vertices_xyz, axis=0),
        )
        assert patch.area_m2 == pytest.approx(
            np.linalg.norm(
                np.cross(
                    patch.vertices_xyz[1] - patch.vertices_xyz[0],
                    patch.vertices_xyz[3] - patch.vertices_xyz[0],
                )
            )
        )
        assert patch.quadrature_points_xyz.shape == (4, 3)
        np.testing.assert_allclose(patch.quadrature_weights, 0.25)
        assert np.sum(patch.quadrature_weights) == pytest.approx(1.0)
        assert patch.integrated_strength_cps_1m(2.5) == pytest.approx(
            2.5 * patch.area_m2
        )

    with pytest.raises(ValueError):
        patches.patches[0].centroid_xyz[0] = 123.0


def test_obstacle_tops_and_only_exposed_sides_replace_covered_floor() -> None:
    environment = EnvironmentConfig(size_x=4.0, size_y=3.0, size_z=3.0)
    obstacles = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(4, 3),
        blocked_cells=((1, 1), (2, 1)),
    )

    patches = build_surface_patches(
        environment,
        obstacles,
        spacing=(1.0, 1.0, 1.0),
        obstacle_height_m=2.0,
        quadrature_points_per_patch=4,
    )

    kinds = Counter(patch.surface_kind for patch in patches.patches)
    assert kinds == {
        "floor": 10,
        "ceiling": 12,
        "wall": 42,
        "obstacle_top": 2,
        "obstacle_side": 12,
    }
    assert patches.patch_count == 78
    assert patches.total_area_m2 == pytest.approx(78.0)

    floor_centers = np.asarray(
        [
            patch.centroid_xyz
            for patch in patches.patches
            if patch.surface_kind == "floor"
        ]
    )
    assert not any(not obstacles.is_free(center) for center in floor_centers)

    object_ids = {patch.object_id for patch in patches.patches}
    assert "obstacle:1:1:east" not in object_ids
    assert "obstacle:2:1:west" not in object_ids
    assert "obstacle:1:1:west" in object_ids
    assert "obstacle:2:1:east" in object_ids
    top_centers = np.asarray(
        [
            patch.centroid_xyz
            for patch in patches.patches
            if patch.surface_kind == "obstacle_top"
        ]
    )
    np.testing.assert_allclose(top_centers[:, 2], 2.0)

    graph = {
        tuple(edge): length
        for edge, length in zip(
            patches.adjacency_edges,
            patches.shared_edge_lengths_m,
        )
    }
    for patch in patches.patches:
        assert len(patch.neighbor_patch_ids) == len(
            patch.neighbor_shared_edge_lengths_m
        )
        for neighbor_id, length in zip(
            patch.neighbor_patch_ids,
            patch.neighbor_shared_edge_lengths_m,
        ):
            key = tuple(sorted((patch.patch_id, neighbor_id)))
            assert graph[key] == pytest.approx(length)
            neighbor = patches.patch_by_id(neighbor_id)
            reverse_index = neighbor.neighbor_patch_ids.index(patch.patch_id)
            assert neighbor.neighbor_shared_edge_lengths_m[reverse_index] == pytest.approx(
                length
            )


def test_one_point_quadrature_is_exactly_the_centroid() -> None:
    patches = build_surface_patches(
        EnvironmentConfig(size_x=1.0, size_y=1.0, size_z=1.0),
        None,
        spacing=2.0,
        quadrature_points_per_patch=1,
    )

    for patch in patches.patches:
        assert patch.quadrature_count == 1
        np.testing.assert_allclose(
            patch.quadrature_points_xyz[0],
            patch.centroid_xyz,
        )
        np.testing.assert_array_equal(patch.quadrature_weights, np.ones(1))


def test_nonaligned_obstacle_boundaries_split_floor_without_area_approximation() -> None:
    obstacles = ObstacleGrid(
        origin=(0.25, 0.25),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=((0, 0),),
    )
    patches = build_surface_patches(
        EnvironmentConfig(size_x=3.0, size_y=3.0, size_z=3.0),
        obstacles,
        spacing=1.0,
        obstacle_height_m=1.5,
    )

    floor = [patch for patch in patches.patches if patch.surface_kind == "floor"]
    tops = [patch for patch in patches.patches if patch.surface_kind == "obstacle_top"]
    sides = [patch for patch in patches.patches if patch.surface_kind == "obstacle_side"]
    assert sum(patch.area_m2 for patch in floor) == pytest.approx(8.0)
    assert sum(patch.area_m2 for patch in tops) == pytest.approx(1.0)
    assert sum(patch.area_m2 for patch in sides) == pytest.approx(6.0)

    for patch in floor:
        x_min, y_min = np.min(patch.vertices_xyz[:, :2], axis=0)
        x_max, y_max = np.max(patch.vertices_xyz[:, :2], axis=0)
        overlap_x = max(0.0, min(x_max, 1.25) - max(x_min, 0.25))
        overlap_y = max(0.0, min(y_max, 1.25) - max(y_min, 0.25))
        assert overlap_x * overlap_y == pytest.approx(0.0)


def test_selective_four_way_refinement_preserves_area_lineage_and_graph() -> None:
    initial = build_surface_patches(
        EnvironmentConfig(size_x=2.0, size_y=3.0, size_z=4.0),
        None,
        spacing=10.0,
        quadrature_points_per_patch=1,
    )
    floor = next(patch for patch in initial.patches if patch.surface_kind == "floor")

    refined = refine_surface_patches(initial, [floor.patch_id])

    assert refined.patch_count == initial.patch_count + 3
    assert refined.total_area_m2 == pytest.approx(initial.total_area_m2)
    children = [
        patch for patch in refined.patches if patch.parent_patch_id == floor.patch_id
    ]
    assert len(children) == 4
    assert all(patch.refinement_level == 1 for patch in children)
    assert sum(patch.area_m2 for patch in children) == pytest.approx(floor.area_m2)
    assert all(patch.area_m2 == pytest.approx(floor.area_m2 / 4.0) for patch in children)
    assert np.sum(refined.shared_edge_lengths_m) == pytest.approx(
        np.sum(initial.shared_edge_lengths_m) + 2.0 + 3.0
    )
    assert refined.adjacency_index_edges.shape == refined.adjacency_edges.shape
    assert np.max(refined.adjacency_index_edges) < refined.patch_count

    second_generation = refine_surface_patches(refined, [children[0].patch_id])
    grandchildren = [
        patch
        for patch in second_generation.patches
        if patch.parent_patch_id == children[0].patch_id
    ]
    assert len(grandchildren) == 4
    assert all(patch.refinement_level == 2 for patch in grandchildren)
    assert sum(patch.area_m2 for patch in grandchildren) == pytest.approx(
        children[0].area_m2
    )


def test_patch_and_patch_set_validation_reject_inconsistent_physics() -> None:
    patch_set = build_surface_patches(
        EnvironmentConfig(size_x=1.0, size_y=1.0, size_z=1.0),
        None,
        spacing=2.0,
        quadrature_points_per_patch=4,
    )
    patch = patch_set.patches[0]

    with pytest.raises(ValueError, match="sum to one"):
        replace(patch, quadrature_weights=np.ones(4))
    with pytest.raises(ValueError, match="vertex area"):
        replace(patch, area_m2=patch.area_m2 * 2.0)
    with pytest.raises(ValueError, match="neighbor IDs"):
        SurfacePatchSet(
            patches=(
                replace(
                    patch_set.patches[0],
                    neighbor_patch_ids=(),
                    neighbor_shared_edge_lengths_m=(),
                ),
                *patch_set.patches[1:],
            ),
            adjacency_edges=patch_set.adjacency_edges,
            shared_edge_lengths_m=patch_set.shared_edge_lengths_m,
        )


def test_observation_batch_and_estimate_enforce_shapes_units_and_covariance() -> None:
    observations = ObservationBatch(
        detector_positions_xyz=np.asarray([[0.5, 0.5, 0.5], [0.5, 0.5, 1.5]]),
        detector_quaternions_wxyz=np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        ),
        fe_indices=np.asarray([0, 7]),
        pb_indices=np.asarray([7, 0]),
        live_times_s=np.asarray([2.0, 3.0]),
        spectrum_counts=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        spectrum_variances=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        energy_bin_edges_keV=np.asarray([0.0, 100.0, 200.0]),
        isotope_counts=np.asarray([[2.0], [4.0]]),
        isotope_covariances=np.asarray([[[0.5]], [[0.75]]]),
        station_ids=np.asarray([4, 4]),
        isotope_names=("Cs-137",),
    )
    assert observations.measurement_count == 2
    assert observations.energy_bin_count == 2
    assert observations.isotope_count == 1
    with pytest.raises(ValueError):
        observations.live_times_s[0] = 0.0

    patch_set = build_surface_patches(
        EnvironmentConfig(size_x=1.0, size_y=1.0, size_z=1.0),
        None,
        spacing=2.0,
        quadrature_points_per_patch=1,
    )
    density = np.arange(1, patch_set.patch_count + 1, dtype=float)[None, :]
    strength = patch_set.integrated_strengths_cps_1m(density)
    estimate = MLEEstimate(
        isotope_names=("Cs-137",),
        patches=patch_set.patches,
        density_by_isotope=density,
        patch_strength_by_isotope=strength,
        predicted_spectra=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        predicted_isotope_counts=np.asarray([[2.0], [4.0]]),
        background_parameters=np.asarray([0.2]),
        nuisance_parameters=np.asarray([]),
        objective_value=12.0,
        poisson_deviance=1.5,
        iterations=20,
        converged=True,
        diagnostics={"density_unit": patch_set.density_unit},
    )
    np.testing.assert_allclose(
        estimate.patch_strength_by_isotope,
        estimate.density_by_isotope * patch_set.areas_m2,
    )

    with pytest.raises(ValueError, match="patch area"):
        replace(estimate, patch_strength_by_isotope=strength + 1.0)
    with pytest.raises(ValueError, match="symmetric"):
        replace(
            observations,
            isotope_names=("Cs-137", "Co-60"),
            isotope_counts=np.ones((2, 2)),
            isotope_covariances=np.asarray(
                [
                    [[1.0, 0.5], [0.0, 1.0]],
                    [[1.0, 0.5], [0.0, 1.0]],
                ]
            ),
        )
