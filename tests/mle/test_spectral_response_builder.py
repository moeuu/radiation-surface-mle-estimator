"""Physics and batching tests for line-resolved spectral surface responses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray

from measurement.continuous_kernels import ContinuousKernel
from measurement.kernels import ShieldParams
from measurement.obstacles import ObstacleGrid
from measurement.shielding import generate_octant_orientations
from spectrum.response_matrix import (
    BACKSCATTER_FRACTION,
    COMPTON_CONTINUUM_TO_PEAK,
    cebr3_efficiency,
    default_resolution,
    detector_response_kernel_for_incident_gamma,
)
from three_d_estimation.spectral_response_builder import (
    build_spectral_nuisance_response,
    build_spectral_response,
)


@dataclass(frozen=True)
class _Observations:
    """Minimal spectrum observation contract used by the builder."""

    detector_positions_xyz: NDArray[np.float64]
    fe_indices: NDArray[np.int64]
    pb_indices: NDArray[np.int64]
    live_times_s: NDArray[np.float64]
    energy_bin_edges_keV: NDArray[np.float64]
    spectrum_counts: NDArray[np.float64]


@dataclass(frozen=True)
class _Patches:
    """Minimal aggregate surface quadrature contract used by the builder."""

    quadrature_points_xyz: NDArray[np.float64]
    quadrature_weights: NDArray[np.float64]
    areas_m2: NDArray[np.float64]


_CS_LINE = (
    {
        "energy_keV": 661.7,
        "weight": 1.0,
        "fe": 0.0,
        "pb": 0.0,
    },
)
_CO_LINES_UNATTENUATED = (
    {
        "energy_keV": 1173.2,
        "weight": 0.5,
        "fe": 0.0,
        "pb": 0.0,
    },
    {
        "energy_keV": 1332.5,
        "weight": 0.5,
        "fe": 0.0,
        "pb": 0.0,
    },
)


def _observations(
    detector_positions_xyz: NDArray[np.float64],
    energy_bin_edges_keV: NDArray[np.float64],
    *,
    live_times_s: NDArray[np.float64] | None = None,
    fe_indices: NDArray[np.int64] | None = None,
    pb_indices: NDArray[np.int64] | None = None,
) -> _Observations:
    """Return a valid minimal observation batch."""
    positions = np.asarray(detector_positions_xyz, dtype=float).reshape(-1, 3)
    edges = np.asarray(energy_bin_edges_keV, dtype=float)
    measurement_count = int(positions.shape[0])
    return _Observations(
        detector_positions_xyz=positions,
        fe_indices=(
            np.zeros(measurement_count, dtype=np.int64)
            if fe_indices is None
            else np.asarray(fe_indices, dtype=np.int64)
        ),
        pb_indices=(
            np.zeros(measurement_count, dtype=np.int64)
            if pb_indices is None
            else np.asarray(pb_indices, dtype=np.int64)
        ),
        live_times_s=(
            np.ones(measurement_count, dtype=float)
            if live_times_s is None
            else np.asarray(live_times_s, dtype=float)
        ),
        energy_bin_edges_keV=edges,
        spectrum_counts=np.zeros((measurement_count, edges.size - 1), dtype=float),
    )


def _patches(
    quadrature_points_xyz: NDArray[np.float64],
    *,
    quadrature_weights: NDArray[np.float64] | None = None,
    areas_m2: NDArray[np.float64] | None = None,
) -> _Patches:
    """Return a valid aggregate patch quadrature batch."""
    points = np.asarray(quadrature_points_xyz, dtype=float)
    patch_count, quadrature_count = points.shape[:2]
    return _Patches(
        quadrature_points_xyz=points,
        quadrature_weights=(
            np.full(
                (patch_count, quadrature_count),
                1.0 / quadrature_count,
                dtype=float,
            )
            if quadrature_weights is None
            else np.asarray(quadrature_weights, dtype=float)
        ),
        areas_m2=(
            np.ones(patch_count, dtype=float)
            if areas_m2 is None
            else np.asarray(areas_m2, dtype=float)
        ),
    )


def _shield_params(*, thickness_cm: float = 0.0) -> ShieldParams:
    """Return simple nested spherical shields for exact attenuation tests."""
    return ShieldParams(
        mu_fe=0.0,
        mu_pb=0.0,
        thickness_fe_cm=thickness_cm,
        thickness_pb_cm=thickness_cm,
        inner_radius_fe_cm=1.0,
        inner_radius_pb_cm=1.0 + thickness_cm,
    )


def _kernel(
    line_mu_by_isotope: dict[str, tuple[dict[str, float], ...]],
    *,
    thickness_cm: float = 0.0,
    obstacle_grid: ObstacleGrid | None = None,
    use_gpu: bool = False,
) -> ContinuousKernel:
    """Return a local shared kernel with controlled line physics."""
    return ContinuousKernel(
        mu_by_isotope={
            isotope: {"fe": 0.0, "pb": 0.0}
            for isotope in line_mu_by_isotope
        },
        shield_params=_shield_params(thickness_cm=thickness_cm),
        line_mu_by_isotope=line_mu_by_isotope,
        obstacle_grid=obstacle_grid,
        detector_radius_m=0.0,
        detector_aperture_radius_m=0.0,
        use_gpu=use_gpu,
        gpu_device="cuda" if use_gpu else "cpu",
        gpu_dtype="float64",
    )


def _line_window_sum(
    spectrum: NDArray[np.float64],
    edges: NDArray[np.float64],
    energy_keV: float,
    *,
    half_width_keV: float = 40.0,
) -> float:
    """Return response mass in a narrow, non-overlapping photopeak window."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = np.abs(centers - float(energy_keV)) <= float(half_width_keV)
    return float(np.sum(spectrum[mask]))


def test_shape_and_total_unshielded_detector_cps_semantics() -> None:
    """The tensor is MxBxGxI and sums to live-time-scaled cps@1m geometry."""
    edges = np.arange(0.0, 1505.0, 5.0)
    observations = _observations(
        np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        edges,
        live_times_s=np.asarray([2.0, 3.0]),
        fe_indices=np.asarray([0, 7]),
        pb_indices=np.asarray([7, 0]),
    )
    patches = _patches(
        np.asarray([[[1.0, 0.0, 0.0]], [[0.0, 2.0, 1.0]]]),
    )
    line_table = {
        "Cs-137": _CS_LINE,
        "Co-60": _CO_LINES_UNATTENUATED,
    }
    kernel = _kernel(line_table)

    result = build_spectral_response(
        observations,
        patches,
        ("Cs-137", "Co-60"),
        kernel,
    )

    assert result.response_per_integrated_strength.shape == (2, 300, 2, 2)
    expected = np.empty((2, 2, 2), dtype=float)
    flat_sources = patches.quadrature_points_xyz[:, 0, :]
    for isotope_index, isotope in enumerate(("Cs-137", "Co-60")):
        values = kernel.kernel_values_selected_pairs_for_detectors(
            isotope=isotope,
            detector_positions=observations.detector_positions_xyz,
            sources=flat_sources,
            fe_indices=observations.fe_indices,
            pb_indices=observations.pb_indices,
        )
        expected[:, :, isotope_index] = observations.live_times_s[:, None] * values
    np.testing.assert_allclose(
        np.sum(result.response_per_integrated_strength, axis=1),
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_detector_response_matches_shared_pulse_height_kernel() -> None:
    """A one-line source is folded through the shared CeBr3 detector response."""
    edges = np.arange(0.0, 1005.0, 5.0)
    observations = _observations(np.asarray([[0.0, 0.0, 0.0]]), edges)
    patches = _patches(np.asarray([[[1.0, 0.0, 0.0]]]))

    result = build_spectral_response(
        observations,
        patches,
        ("Cs-137",),
        _kernel({"Cs-137": _CS_LINE}),
    )

    centers = 0.5 * (edges[:-1] + edges[1:])
    expected = detector_response_kernel_for_incident_gamma(
        centers,
        661.7,
        default_resolution(),
        cebr3_efficiency,
        5.0,
        continuum_to_peak=COMPTON_CONTINUUM_TO_PEAK,
        backscatter_fraction=BACKSCATTER_FRACTION,
    )
    actual = result.response_per_integrated_strength[0, :, 0, 0]
    np.testing.assert_allclose(actual, expected, rtol=1.0e-13, atol=1.0e-13)
    assert np.count_nonzero(actual) > 20
    assert np.sum(actual[(centers > 50.0) & (centers < 400.0)]) > 0.0


def test_co60_lines_receive_distinct_shield_attenuation() -> None:
    """Co-60 line peaks must not share one aggregate attenuation multiplier."""
    edges = np.arange(1000.0, 1452.0, 2.0)
    orientation = generate_octant_orientations()[0]
    observations = _observations(
        np.asarray([[0.0, 0.0, 0.0]]),
        edges,
        fe_indices=np.asarray([0]),
        pb_indices=np.asarray([0]),
    )
    patches = _patches(np.asarray([[[-orientation[0], -orientation[1], -orientation[2]]]]))
    differential_lines = (
        dict(_CO_LINES_UNATTENUATED[0]),
        {
            **_CO_LINES_UNATTENUATED[1],
            "fe": 2.0,
            "pb": 2.0,
        },
    )
    build_kwargs = {
        "observations": observations,
        "patches": patches,
        "isotopes": ("Co-60",),
        "continuum_to_peak": 0.0,
        "backscatter_fraction": 0.0,
    }
    unshielded = build_spectral_response(
        kernel=_kernel({"Co-60": _CO_LINES_UNATTENUATED}, thickness_cm=1.0),
        **build_kwargs,
    ).response_per_integrated_strength[0, :, 0, 0]
    shielded = build_spectral_response(
        kernel=_kernel({"Co-60": differential_lines}, thickness_cm=1.0),
        **build_kwargs,
    ).response_per_integrated_strength[0, :, 0, 0]

    ratios = []
    for energy in (1173.2, 1332.5):
        baseline = _line_window_sum(unshielded, edges, energy)
        ratios.append(_line_window_sum(shielded, edges, energy) / baseline)

    assert ratios[0] == pytest.approx(1.0, rel=1.0e-12, abs=1.0e-12)
    assert ratios[1] < 0.05
    assert ratios[1] < 0.1 * ratios[0]


def test_obstacle_uses_the_matching_line_mu_row() -> None:
    """Per-line obstacle tables attenuate only their corresponding gamma line."""
    edges = np.arange(1000.0, 1452.0, 2.0)
    observations = _observations(np.asarray([[0.0, 0.0, 0.0]]), edges)
    patches = _patches(np.asarray([[[1.0, 0.0, 0.0]]]))
    obstacle_grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=(),
        transport_boxes_m=((0.4, -0.1, -0.1, 0.6, 0.1, 0.1),),
        transport_mu_by_isotope={"Co-60": (0.0,)},
        transport_line_mu_by_isotope={"Co-60": ((0.0,), (1.0,))},
    )
    build_kwargs = {
        "observations": observations,
        "patches": patches,
        "isotopes": ("Co-60",),
        "continuum_to_peak": 0.0,
        "backscatter_fraction": 0.0,
    }
    baseline = build_spectral_response(
        kernel=_kernel({"Co-60": _CO_LINES_UNATTENUATED}),
        **build_kwargs,
    ).response_per_integrated_strength[0, :, 0, 0]
    attenuated = build_spectral_response(
        kernel=_kernel(
            {"Co-60": _CO_LINES_UNATTENUATED},
            obstacle_grid=obstacle_grid,
        ),
        **build_kwargs,
    ).response_per_integrated_strength[0, :, 0, 0]

    first_ratio = _line_window_sum(attenuated, edges, 1173.2) / _line_window_sum(
        baseline,
        edges,
        1173.2,
    )
    second_ratio = _line_window_sum(attenuated, edges, 1332.5) / _line_window_sum(
        baseline,
        edges,
        1332.5,
    )
    assert first_ratio == pytest.approx(1.0, rel=1.0e-12, abs=1.0e-12)
    assert second_ratio < 1.0e-6


def test_incomplete_obstacle_line_table_is_rejected() -> None:
    """A partial line table cannot silently fall back to aggregate attenuation."""
    edges = np.arange(1000.0, 1452.0, 2.0)
    observations = _observations(np.asarray([[0.0, 0.0, 0.0]]), edges)
    patches = _patches(np.asarray([[[1.0, 0.0, 0.0]]]))
    incomplete_grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=(),
        transport_boxes_m=((0.4, -0.1, -0.1, 0.6, 0.1, 0.1),),
        transport_mu_by_isotope={"Co-60": (0.2,)},
        transport_line_mu_by_isotope={"Co-60": ((0.1,),)},
    )

    with pytest.raises(ValueError, match="has 1 rows but line index 1"):
        build_spectral_response(
            observations,
            patches,
            ("Co-60",),
            _kernel(
                {"Co-60": _CO_LINES_UNATTENUATED},
                obstacle_grid=incomplete_grid,
            ),
        )


def test_missing_obstacle_line_table_is_rejected_for_production_fit() -> None:
    """Spectral fitting cannot reuse one aggregate obstacle attenuation scalar."""
    edges = np.arange(1000.0, 1452.0, 2.0)
    observations = _observations(np.asarray([[0.0, 0.0, 0.0]]), edges)
    patches = _patches(np.asarray([[[1.0, 0.0, 0.0]]]))
    aggregate_only_grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=(),
        transport_boxes_m=((0.4, -0.1, -0.1, 0.6, 0.1, 0.1),),
        transport_mu_by_isotope={"Co-60": (0.2,)},
    )

    with pytest.raises(ValueError, match="line-resolved obstacle attenuation"):
        build_spectral_response(
            observations,
            patches,
            ("Co-60",),
            _kernel(
                {"Co-60": _CO_LINES_UNATTENUATED},
                obstacle_grid=aggregate_only_grid,
            ),
        )


def test_density_response_applies_exact_patch_area_scaling() -> None:
    """Density columns equal integrated-strength columns times exact area."""
    edges = np.arange(0.0, 1005.0, 5.0)
    observations = _observations(np.asarray([[0.0, 0.0, 0.0]]), edges)
    patches = _patches(
        np.asarray([[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]]),
        areas_m2=np.asarray([0.5, 2.0]),
    )

    result = build_spectral_response(
        observations,
        patches,
        ("Cs-137",),
        _kernel({"Cs-137": _CS_LINE}),
    )

    np.testing.assert_allclose(
        result.response_per_integrated_strength[:, :, 0, :],
        result.response_per_integrated_strength[:, :, 1, :],
    )
    np.testing.assert_allclose(
        result.response_per_density,
        result.response_per_integrated_strength
        * np.asarray([0.5, 2.0])[None, None, :, None],
    )


def test_nuisance_rate_bases_are_nonnegative_and_live_time_scaled() -> None:
    """Background and scatter coefficients retain cps rate semantics."""
    edges = np.arange(0.0, 1005.0, 5.0)
    live_times = np.asarray([2.0, 5.0])

    response, names = build_spectral_nuisance_response(live_times, edges)
    empty, empty_names = build_spectral_nuisance_response(
        live_times,
        edges,
        include_background=False,
        include_scatter=False,
    )

    assert response.shape == (2, 200, 2)
    assert names == ("background_rate_cps", "scatter_rate_cps")
    assert np.all(response >= 0.0)
    np.testing.assert_allclose(
        np.sum(response, axis=1),
        np.repeat(live_times[:, None], 2, axis=1),
    )
    assert empty.shape == (2, 200, 0)
    assert empty_names == ()


def test_kernel_chunk_size_does_not_change_cpu_result() -> None:
    """Kernel chunk boundaries alter memory use but not spectral responses."""
    edges = np.arange(0.0, 1505.0, 5.0)
    observations = _observations(
        np.asarray(
            [
                [0.0, 0.0, 0.5],
                [0.25, 0.0, 1.5],
                [0.5, 0.25, 2.0],
            ]
        ),
        edges,
        live_times_s=np.asarray([1.0, 1.5, 2.0]),
        fe_indices=np.asarray([0, 3, 7]),
        pb_indices=np.asarray([7, 4, 0]),
    )
    patches = _patches(
        np.asarray(
            [
                [[1.0, 0.0, 0.5], [1.25, 0.0, 0.5]],
                [[0.0, 1.0, 1.5], [0.0, 1.25, 1.5]],
            ]
        ),
        quadrature_weights=np.asarray([[0.25, 0.75], [0.6, 0.4]]),
    )
    line_table = {
        "Cs-137": _CS_LINE,
        "Co-60": _CO_LINES_UNATTENUATED,
    }

    small = build_spectral_response(
        observations,
        patches,
        ("Cs-137", "Co-60"),
        _kernel(line_table),
        chunk_size=1,
    )
    large = build_spectral_response(
        observations,
        patches,
        ("Cs-137", "Co-60"),
        _kernel(line_table),
        chunk_size=1_000_000,
    )

    np.testing.assert_array_equal(
        small.response_per_integrated_strength,
        large.response_per_integrated_strength,
    )
    np.testing.assert_array_equal(
        small.response_per_density,
        large.response_per_density,
    )
    np.testing.assert_array_equal(small.nuisance_response, large.nuisance_response)


def test_invalid_pair_index_and_chunk_are_rejected_before_kernel_work() -> None:
    """Invalid pairs cannot wrap modulo the orientation count on GPU."""
    edges = np.arange(0.0, 1005.0, 5.0)
    patches = _patches(np.asarray([[[1.0, 0.0, 0.0]]]))
    invalid_pair = _observations(
        np.asarray([[0.0, 0.0, 0.0]]),
        edges,
        fe_indices=np.asarray([8]),
    )

    with pytest.raises(ValueError, match="must lie in"):
        build_spectral_response(
            invalid_pair,
            patches,
            ("Cs-137",),
            _kernel({"Cs-137": _CS_LINE}),
        )
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        build_spectral_response(
            _observations(np.asarray([[0.0, 0.0, 0.0]]), edges),
            patches,
            ("Cs-137",),
            _kernel({"Cs-137": _CS_LINE}),
            chunk_size=0,
        )


def test_cpu_gpu_equivalence_when_cuda_is_available() -> None:
    """Line-resolved spectral responses agree across CPU and CUDA kernels."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    edges = np.arange(1000.0, 1452.0, 2.0)
    observations = _observations(
        np.asarray([[0.0, 0.0, 0.5], [0.1, 0.0, 1.5]]),
        edges,
        live_times_s=np.asarray([1.0, 2.0]),
        fe_indices=np.asarray([0, 7]),
        pb_indices=np.asarray([0, 7]),
    )
    orientation = generate_octant_orientations()[0]
    patches = _patches(
        np.asarray(
            [
                [
                    [-orientation[0], -orientation[1], -orientation[2] + 0.5],
                    [
                        -1.1 * orientation[0],
                        -1.1 * orientation[1],
                        -1.1 * orientation[2] + 0.5,
                    ],
                ],
                [[1.0, 0.0, 1.5], [1.25, 0.0, 1.5]],
            ]
        ),
        quadrature_weights=np.asarray([[0.4, 0.6], [0.5, 0.5]]),
        areas_m2=np.asarray([0.75, 1.5]),
    )
    differential_lines = (
        dict(_CO_LINES_UNATTENUATED[0]),
        {
            **_CO_LINES_UNATTENUATED[1],
            "fe": 0.4,
            "pb": 0.6,
        },
    )
    obstacle_grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=(),
        transport_boxes_m=((0.4, -0.1, 0.0, 0.6, 0.1, 2.0),),
        transport_mu_by_isotope={"Co-60": (0.05,)},
        transport_line_mu_by_isotope={"Co-60": ((0.02,), (0.08,))},
    )

    cpu = build_spectral_response(
        observations,
        patches,
        ("Co-60",),
        _kernel(
            {"Co-60": differential_lines},
            thickness_cm=1.0,
            obstacle_grid=obstacle_grid,
        ),
        chunk_size=2,
    )
    gpu = build_spectral_response(
        observations,
        patches,
        ("Co-60",),
        _kernel(
            {"Co-60": differential_lines},
            thickness_cm=1.0,
            obstacle_grid=obstacle_grid,
            use_gpu=True,
        ),
        chunk_size=2,
    )

    np.testing.assert_allclose(
        gpu.response_per_integrated_strength,
        cpu.response_per_integrated_strength,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        gpu.response_per_density,
        cpu.response_per_density,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
