"""Tests for batched count-domain surface response construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray

from measurement.continuous_kernels import ContinuousKernel
from measurement.kernels import ShieldParams
from measurement.shielding import generate_octant_orientations
from three_d_estimation.response_builder import (
    build_count_response,
    build_count_responses,
)
from three_d_estimation.types import ObservationBatch, SurfacePatch, SurfacePatchSet


@dataclass(frozen=True)
class _Observations:
    """Minimal observation batch used by the response-builder contract."""

    detector_positions_xyz: NDArray[np.float64]
    fe_indices: NDArray[np.int64]
    pb_indices: NDArray[np.int64]
    live_times_s: NDArray[np.float64]


@dataclass(frozen=True)
class _Patches:
    """Minimal surface patch set used by the response-builder contract."""

    areas_m2: NDArray[np.float64]
    quadrature_points_xyz: NDArray[np.float64]
    quadrature_weights: NDArray[np.float64]


class _DeterministicKernel:
    """Batched kernel oracle with detector, source, pair, and isotope effects."""

    def __init__(self) -> None:
        self.orientations = generate_octant_orientations()
        self.calls: list[tuple[int, int, int]] = []

    def kernel_values_selected_pairs_for_detectors(
        self,
        isotope: str,
        detector_positions: NDArray[np.float64],
        sources: NDArray[np.float64],
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
        chunk_size: int,
    ) -> NDArray[np.float64]:
        """Return a deterministic positive matrix for batch-contract tests."""
        detectors = np.asarray(detector_positions, dtype=float)
        source_points = np.asarray(sources, dtype=float)
        fe = np.asarray(fe_indices, dtype=float)
        pb = np.asarray(pb_indices, dtype=float)
        self.calls.append((detectors.shape[0], source_points.shape[0], chunk_size))
        isotope_scale = {"Cs-137": 1.0, "Co-60": 2.0}[isotope]
        squared_distance = np.sum(
            (detectors[:, None, :] - source_points[None, :, :]) ** 2,
            axis=2,
        )
        pair_scale = 1.0 + 0.1 * fe + 0.01 * pb
        return isotope_scale * pair_scale[:, None] / (1.0 + squared_distance)


def _observations(
    detector_positions: NDArray[np.float64],
    *,
    live_times: NDArray[np.float64] | None = None,
    fe_indices: NDArray[np.int64] | None = None,
    pb_indices: NDArray[np.int64] | None = None,
) -> _Observations:
    """Build a valid minimal observation batch."""
    positions = np.asarray(detector_positions, dtype=float).reshape(-1, 3)
    count = int(positions.shape[0])
    return _Observations(
        detector_positions_xyz=positions,
        fe_indices=(
            np.zeros(count, dtype=np.int64)
            if fe_indices is None
            else np.asarray(fe_indices, dtype=np.int64)
        ),
        pb_indices=(
            np.zeros(count, dtype=np.int64)
            if pb_indices is None
            else np.asarray(pb_indices, dtype=np.int64)
        ),
        live_times_s=(
            np.ones(count, dtype=float)
            if live_times is None
            else np.asarray(live_times, dtype=float)
        ),
    )


def _patches(
    quadrature_points: NDArray[np.float64],
    *,
    areas: NDArray[np.float64] | None = None,
    weights: NDArray[np.float64] | None = None,
) -> _Patches:
    """Build a valid minimal patch set."""
    points = np.asarray(quadrature_points, dtype=float)
    patch_count, quadrature_count = points.shape[:2]
    return _Patches(
        areas_m2=(
            np.ones(patch_count, dtype=float)
            if areas is None
            else np.asarray(areas, dtype=float)
        ),
        quadrature_points_xyz=points,
        quadrature_weights=(
            np.full(
                (patch_count, quadrature_count),
                1.0 / quadrature_count,
                dtype=float,
            )
            if weights is None
            else np.asarray(weights, dtype=float)
        ),
    )


def _unshielded_kernel(*, use_gpu: bool, device: str = "cpu") -> ContinuousKernel:
    """Return a ContinuousKernel whose response is pure detector-cps@1m geometry."""
    return ContinuousKernel(
        mu_by_isotope={
            "Cs-137": {"fe": 0.0, "pb": 0.0},
            "Co-60": {"fe": 0.0, "pb": 0.0},
        },
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
        use_gpu=use_gpu,
        gpu_device=device,
        gpu_dtype="float64",
        detector_radius_m=0.0,
        detector_aperture_radius_m=0.0,
    )


def test_single_quadrature_point_matches_kernel_times_live_time() -> None:
    """A centroid-only patch must preserve the shared kernel exactly."""
    kernel = _unshielded_kernel(use_gpu=False)
    observations = _observations(
        np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        live_times=np.asarray([2.0, 3.0]),
        fe_indices=np.asarray([0, 7]),
        pb_indices=np.asarray([7, 0]),
    )
    patches = _patches(np.asarray([[[2.0, 0.0, 0.5]]]))

    response = build_count_response(observations, patches, ("Cs-137",), kernel)
    direct = kernel.kernel_values_selected_pairs_for_detectors(
        isotope="Cs-137",
        detector_positions=observations.detector_positions_xyz,
        sources=patches.quadrature_points_xyz.reshape(-1, 3),
        fe_indices=observations.fe_indices,
        pb_indices=observations.pb_indices,
    )

    assert response.shape == (2, 1, 1)
    np.testing.assert_allclose(
        response[:, 0, 0],
        direct[:, 0] * observations.live_times_s,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_canonical_observation_and_patch_types_are_supported() -> None:
    """The builder consumes the production data contracts without adapters."""
    observations = ObservationBatch(
        detector_positions_xyz=np.asarray([[0.0, 0.0, 0.0]]),
        detector_quaternions_wxyz=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
        fe_indices=np.asarray([0], dtype=np.int64),
        pb_indices=np.asarray([0], dtype=np.int64),
        live_times_s=np.asarray([2.0]),
        spectrum_counts=np.asarray([[0.0]]),
        spectrum_variances=None,
        energy_bin_edges_keV=np.asarray([0.0, 1.0]),
        isotope_counts=None,
        isotope_covariances=None,
        station_ids=np.asarray([0], dtype=np.int64),
        isotope_names=("Cs-137",),
    )
    patch = SurfacePatch(
        patch_id=0,
        centroid_xyz=np.asarray([1.0, 0.0, 0.0]),
        normal_xyz=np.asarray([1.0, 0.0, 0.0]),
        area_m2=1.0,
        surface_kind="wall",
        object_id="room:x_max",
        vertices_xyz=np.asarray(
            [
                [1.0, -0.5, -0.5],
                [1.0, 0.5, -0.5],
                [1.0, 0.5, 0.5],
                [1.0, -0.5, 0.5],
            ]
        ),
        quadrature_points_xyz=np.asarray([[1.0, 0.0, 0.0]]),
        quadrature_weights=np.asarray([1.0]),
    )
    patches = SurfacePatchSet(patches=(patch,))

    response = build_count_response(
        observations,
        patches,
        observations.isotope_names,
        _unshielded_kernel(use_gpu=False),
    )

    np.testing.assert_allclose(
        response,
        np.asarray([[[2.0]]]),
        rtol=0.0,
        atol=1.0e-13,
    )


def test_density_response_scales_with_patch_area_only() -> None:
    """Doubling area doubles density response but not unit-strength response."""
    kernel = _DeterministicKernel()
    observations = _observations(np.asarray([[0.0, 0.0, 0.0]]))
    patches = _patches(
        np.asarray([[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]]),
        areas=np.asarray([0.75, 1.5]),
    )

    responses = build_count_responses(
        observations,
        patches,
        ("Cs-137",),
        kernel,
    )

    np.testing.assert_allclose(
        responses.response_by_integrated_strength[:, 0, :],
        responses.response_by_integrated_strength[:, 1, :],
    )
    np.testing.assert_allclose(
        responses.response_by_density,
        responses.response_by_integrated_strength * np.asarray([0.75, 1.5])[None, :, None],
    )
    np.testing.assert_allclose(
        responses.response_by_density[:, 1, :],
        2.0 * responses.response_by_density[:, 0, :],
    )


def test_nonuniform_quadrature_is_applied_as_a_normalized_weighted_sum() -> None:
    """Patch quadrature uses declared weights rather than an unweighted mean."""
    observations = _observations(
        np.asarray([[0.0, 0.0, 0.0]]),
        live_times=np.asarray([2.0]),
    )
    patches = _patches(
        np.asarray([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
        weights=np.asarray([[0.25, 0.75]]),
    )

    response = build_count_response(
        observations,
        patches,
        ("Cs-137", "Co-60"),
        _DeterministicKernel(),
    )

    expected_cs137 = 2.0 * (0.25 * 1.0 + 0.75 * 0.5)
    np.testing.assert_allclose(
        response,
        np.asarray([[[expected_cs137, 2.0 * expected_cs137]]]),
    )


def test_one_metre_unit_response_for_all_64_fe_pb_pairs() -> None:
    """The cps@1m convention gives one count per second at 1 m for every pair."""
    kernel = _unshielded_kernel(use_gpu=False)
    fe_indices = np.repeat(np.arange(8, dtype=np.int64), 8)
    pb_indices = np.tile(np.arange(8, dtype=np.int64), 8)
    observations = _observations(
        np.zeros((64, 3), dtype=float),
        fe_indices=fe_indices,
        pb_indices=pb_indices,
    )
    patches = _patches(np.asarray([[[1.0, 0.0, 0.0]]]))

    response = build_count_response(observations, patches, ("Cs-137",), kernel)

    assert response.shape == (64, 1, 1)
    np.testing.assert_allclose(response[:, 0, 0], 1.0, rtol=1.0e-13, atol=1.0e-13)


def test_detector_heights_are_preserved_in_batched_geometry() -> None:
    """Responses at 0.5, 1.5, and 2.0 m use the recorded detector z values."""
    kernel = _unshielded_kernel(use_gpu=False)
    heights = np.asarray([0.5, 1.5, 2.0])
    observations = _observations(
        np.column_stack([np.zeros(3), np.zeros(3), heights]),
    )
    patches = _patches(np.asarray([[[1.0, 0.0, 1.5]]]))

    response = build_count_response(observations, patches, ("Cs-137",), kernel)
    expected = 1.0 / (1.0 + (heights - 1.5) ** 2)

    np.testing.assert_allclose(response[:, 0, 0], expected, rtol=1.0e-13, atol=1.0e-13)


def test_quadrature_and_chunk_sizes_are_deterministic() -> None:
    """Changing deterministic memory chunks must not change response values."""
    detector_positions = np.column_stack(
        [np.linspace(0.0, 1.0, 5), np.zeros(5), np.linspace(0.5, 2.0, 5)]
    )
    observations = _observations(
        detector_positions,
        live_times=np.linspace(1.0, 2.0, 5),
        fe_indices=np.arange(5, dtype=np.int64),
        pb_indices=np.arange(4, -1, -1, dtype=np.int64),
    )
    points = np.arange(7 * 3 * 3, dtype=float).reshape(7, 3, 3) / 10.0
    weights = np.tile(np.asarray([0.2, 0.3, 0.5]), (7, 1))
    patches = _patches(points, areas=np.linspace(0.5, 2.0, 7), weights=weights)
    chunked_kernel = _DeterministicKernel()
    unchunked_kernel = _DeterministicKernel()

    chunked = build_count_responses(
        observations,
        patches,
        ("Cs-137", "Co-60"),
        chunked_kernel,
        measurement_chunk_size=2,
        patch_chunk_size=3,
        kernel_chunk_size=17,
    )
    unchunked = build_count_responses(
        observations,
        patches,
        ("Cs-137", "Co-60"),
        unchunked_kernel,
        measurement_chunk_size=100,
        patch_chunk_size=100,
        kernel_chunk_size=1_000,
    )

    np.testing.assert_array_equal(
        chunked.response_by_integrated_strength,
        unchunked.response_by_integrated_strength,
    )
    np.testing.assert_array_equal(
        chunked.response_by_density,
        unchunked.response_by_density,
    )
    assert len(chunked_kernel.calls) == 18
    assert max(call[0] for call in chunked_kernel.calls) == 2
    assert max(call[1] for call in chunked_kernel.calls) == 9
    assert {call[2] for call in chunked_kernel.calls} == {17}


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("fe_indices", np.asarray([8], dtype=np.int64), "must lie in"),
        ("live_times_s", np.asarray([-1.0]), "must be non-negative"),
    ],
)
def test_observation_validation(
    field: str,
    replacement: NDArray[np.generic],
    error: str,
) -> None:
    """Invalid observation shapes and physical values fail before kernel work."""
    valid = _observations(np.asarray([[0.0, 0.0, 0.0]]))
    values = {
        "detector_positions_xyz": valid.detector_positions_xyz,
        "fe_indices": valid.fe_indices,
        "pb_indices": valid.pb_indices,
        "live_times_s": valid.live_times_s,
    }
    values[field] = replacement
    observations = _Observations(**values)

    with pytest.raises((TypeError, ValueError), match=error):
        build_count_response(
            observations,
            _patches(np.asarray([[[1.0, 0.0, 0.0]]])),
            ("Cs-137",),
            _DeterministicKernel(),
        )


def test_patch_shape_and_weight_validation() -> None:
    """Quadrature shape, normalization, and physical area are enforced."""
    observations = _observations(np.asarray([[0.0, 0.0, 0.0]]))
    kernel = _DeterministicKernel()
    invalid_weights = _Patches(
        areas_m2=np.asarray([1.0]),
        quadrature_points_xyz=np.asarray([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
        quadrature_weights=np.asarray([[0.2, 0.2]]),
    )
    invalid_areas = _Patches(
        areas_m2=np.asarray([0.0]),
        quadrature_points_xyz=np.asarray([[[1.0, 0.0, 0.0]]]),
        quadrature_weights=np.asarray([[1.0]]),
    )

    with pytest.raises(ValueError, match="sum to one"):
        build_count_response(observations, invalid_weights, ("Cs-137",), kernel)
    with pytest.raises(ValueError, match="positive"):
        build_count_response(observations, invalid_areas, ("Cs-137",), kernel)


def test_cpu_gpu_equivalence_when_cuda_is_available() -> None:
    """The shared ContinuousKernel yields equivalent CPU and CUDA responses."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    observations = _observations(
        np.asarray([[0.0, 0.0, 0.5], [0.25, 0.0, 1.5]]),
        live_times=np.asarray([1.0, 2.0]),
        fe_indices=np.asarray([0, 7]),
        pb_indices=np.asarray([7, 0]),
    )
    patches = _patches(
        np.asarray(
            [
                [[1.0, 0.0, 0.5], [1.25, 0.0, 0.5]],
                [[0.5, 1.0, 1.5], [0.75, 1.0, 1.5]],
            ]
        ),
        areas=np.asarray([0.5, 1.25]),
        weights=np.asarray([[0.25, 0.75], [0.5, 0.5]]),
    )

    cpu = build_count_responses(
        observations,
        patches,
        ("Cs-137", "Co-60"),
        _unshielded_kernel(use_gpu=False),
        measurement_chunk_size=1,
        patch_chunk_size=1,
    )
    gpu = build_count_responses(
        observations,
        patches,
        ("Cs-137", "Co-60"),
        _unshielded_kernel(use_gpu=True, device="cuda"),
        measurement_chunk_size=1,
        patch_chunk_size=1,
        kernel_chunk_size=2,
    )

    np.testing.assert_allclose(
        gpu.response_by_integrated_strength,
        cpu.response_by_integrated_strength,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        gpu.response_by_density,
        cpu.response_by_density,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
