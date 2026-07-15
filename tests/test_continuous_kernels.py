"""Tests for continuous 3D kernel evaluation (Sec. 3.2–3.3)."""

import numpy as np
import pytest

from measurement.continuous_kernels import (
    ContinuousKernel,
    finite_sphere_geometric_term,
    geometric_term,
    segment_rotated_octant_shell_path_length_cm_torch,
)
from measurement.kernels import ShieldParams
from measurement.obstacles import ObstacleGrid
from measurement.shielding import (
    DEFAULT_FE_SHIELD_INNER_RADIUS_CM,
    DEFAULT_PB_SHIELD_INNER_RADIUS_CM,
)
from measurement.continuous_kernels import expected_counts_single_isotope
from pf.gpu_utils import expected_counts_all_pairs_torch, expected_counts_pair_torch


def test_geometric_term_inverse_square() -> None:
    """Geometric term should follow 1/d^2 scaling."""
    det = np.array([0.0, 0.0, 0.0])
    s1 = np.array([1.0, 0.0, 0.0])
    s2 = np.array([2.0, 0.0, 0.0])
    g1 = geometric_term(det, s1)
    g2 = geometric_term(det, s2)
    assert np.isclose(g1 / g2, 4.0, rtol=1e-6)


def test_finite_sphere_geometric_term_preserves_one_meter_definition() -> None:
    """Finite detector geometry should remove the near-field point singularity."""
    det = np.array([0.0, 0.0, 0.0], dtype=float)
    radius = 0.04

    at_one_meter = finite_sphere_geometric_term(
        det,
        np.array([1.0, 0.0, 0.0], dtype=float),
        radius,
    )
    at_two_meters = finite_sphere_geometric_term(
        det,
        np.array([2.0, 0.0, 0.0], dtype=float),
        radius,
    )
    at_center = finite_sphere_geometric_term(det, det, radius)

    assert at_one_meter == pytest.approx(1.0)
    assert at_two_meters == pytest.approx(0.25, rel=1.0e-3)
    assert np.isfinite(at_center)
    assert at_center < 1.0e4


def test_torch_rotated_octant_shell_cached_rotation_matches_direct() -> None:
    """Cached torch octant rotations should preserve exact path-length results."""
    torch = pytest.importorskip("torch")
    kernel = ContinuousKernel(use_gpu=True, gpu_device="cpu", gpu_dtype="float64")
    device = torch.device("cpu")
    dtype = torch.float64
    source_pos = torch.as_tensor(
        [[1.0, 1.0, 1.0], [-1.0, 1.5, 0.5]],
        device=device,
        dtype=dtype,
    )
    target_pos = torch.zeros_like(source_pos)
    center_pos = torch.zeros(1, 3, device=device, dtype=dtype)
    shield_normal = -np.asarray(kernel.orientations[0], dtype=float)
    rotation = kernel._rotated_octant_rotation_torch(
        shield_normal,
        device=device,
        dtype=dtype,
    )

    direct = segment_rotated_octant_shell_path_length_cm_torch(
        source_pos=source_pos,
        target_pos=target_pos,
        center_pos=center_pos,
        shield_normal=shield_normal,
        inner_radius_cm=1.0,
        outer_radius_cm=20.0,
    )
    cached = segment_rotated_octant_shell_path_length_cm_torch(
        source_pos=source_pos,
        target_pos=target_pos,
        center_pos=center_pos,
        shield_normal=None,
        inner_radius_cm=1.0,
        outer_radius_cm=20.0,
        rotation=rotation,
    )

    assert torch.allclose(cached, direct, rtol=1e-12, atol=1e-12)
    assert len(kernel._torch_octant_rotation_cache) == 1


def test_attenuation_applies_blocking_factor() -> None:
    """Blocked orientation should reduce expected counts by exp(-mu*L)."""
    shield_params = ShieldParams()
    kernel = ContinuousKernel(shield_params=shield_params, use_gpu=False)
    det = np.array([0.0, 0.0, 0.0])
    src = np.array([1.0, 1.0, 1.0])
    strengths = np.array([10.0])

    # Vector from src->det is (-,-,-), so orient 7 blocks, orient 0 unblocks
    blocked_counts = kernel.expected_counts(
        isotope="Cs-137",
        detector_pos=det,
        sources=np.array([src]),
        strengths=strengths,
        orient_idx=7,
        live_time_s=1.0,
        background=0.0,
    )
    free_counts = kernel.expected_counts(
        isotope="Cs-137",
        detector_pos=det,
        sources=np.array([src]),
        strengths=strengths,
        orient_idx=0,
        live_time_s=1.0,
        background=0.0,
    )
    expected_ratio = np.exp(
        -(shield_params.mu_fe * shield_params.thickness_fe_cm + shield_params.mu_pb * shield_params.thickness_pb_cm)
    )
    assert np.isclose(blocked_counts, expected_ratio * free_counts, rtol=1e-6)


def test_line_resolved_attenuation_uses_weighted_transmission() -> None:
    """ContinuousKernel should average shield transmission over gamma lines."""
    shield_params = ShieldParams(
        mu_fe=0.0,
        mu_pb=0.0,
        thickness_fe_cm=2.0,
        thickness_pb_cm=0.0,
    )
    line_mu = {
        "TestIso": (
            {"weight": 1.0, "fe": 0.10, "pb": 0.0},
            {"weight": 3.0, "fe": 0.30, "pb": 0.0},
        )
    }
    kernel = ContinuousKernel(
        mu_by_isotope={"TestIso": {"fe": 0.0, "pb": 0.0}},
        shield_params=shield_params,
        line_mu_by_isotope=line_mu,
        use_gpu=False,
    )
    detector = np.zeros(3, dtype=float)
    source = np.array([[1.0, 1.0, 1.0]], dtype=float)
    strength = np.array([100.0], dtype=float)

    blocked = kernel.expected_counts_pair(
        "TestIso",
        detector,
        source,
        strength,
        fe_index=7,
        pb_index=0,
        live_time_s=1.0,
    )
    free = kernel.expected_counts_pair(
        "TestIso",
        detector,
        source,
        strength,
        fe_index=0,
        pb_index=0,
        live_time_s=1.0,
    )
    expected_ratio = 0.25 * np.exp(-0.10 * 2.0) + 0.75 * np.exp(-0.30 * 2.0)

    assert blocked == pytest.approx(free * expected_ratio, rel=1e-12)


def test_line_resolved_obstacle_attenuation_uses_line_mu_values() -> None:
    """ContinuousKernel should average obstacle transmission over gamma lines."""
    grid = ObstacleGrid(
        origin=(0.0, -0.5),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=((0, 0),),
    ).with_transport_model(
        boxes_m=((0.0, -0.5, 0.0, 1.0, 0.5, 2.0),),
        mu_by_isotope={"TestIso": (0.0,)},
        line_mu_by_isotope={"TestIso": ((0.01,), (0.03,))},
    )
    line_mu = {
        "TestIso": (
            {"weight": 1.0, "fe": 0.0, "pb": 0.0},
            {"weight": 3.0, "fe": 0.0, "pb": 0.0},
        )
    }
    kernel = ContinuousKernel(
        mu_by_isotope={"TestIso": {"fe": 0.0, "pb": 0.0}},
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
        obstacle_grid=grid,
        line_mu_by_isotope=line_mu,
        use_gpu=False,
    )
    free_kernel = ContinuousKernel(
        mu_by_isotope={"TestIso": {"fe": 0.0, "pb": 0.0}},
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
        line_mu_by_isotope=line_mu,
        use_gpu=False,
    )
    source = np.array([-1.0, 0.0, 1.0], dtype=float)
    detector = np.array([2.0, 0.0, 1.0], dtype=float)

    blocked = kernel.kernel_value_pair("TestIso", detector, source, 0, 0)
    free = free_kernel.kernel_value_pair("TestIso", detector, source, 0, 0)
    expected_ratio = 0.25 * np.exp(-1.0) + 0.75 * np.exp(-3.0)

    assert kernel.obstacle_path_lengths_by_box_cm(source, detector)[0] == pytest.approx(100.0)
    assert blocked == pytest.approx(free * expected_ratio, rel=1e-12)


def test_transport_response_terms_reconstruct_kernel_with_aperture() -> None:
    """Calibration terms should match the aperture-averaged runtime kernel."""
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=((0, 0),),
    ).with_transport_model(
        boxes_m=((0.2, 0.2, 0.2, 0.8, 0.8, 0.8),),
        mu_by_isotope={"TestIso": (0.01,)},
        line_mu_by_isotope={"TestIso": ((0.01,), (0.025,))},
    )
    line_mu = {
        "TestIso": (
            {"weight": 0.4, "fe": 0.05, "pb": 0.08},
            {"weight": 0.6, "fe": 0.09, "pb": 0.12},
        )
    }
    transport_model = {
        "enabled": True,
        "by_isotope": {
            "TestIso": {
                "scale": 1.1,
                "tau_coefficients": {
                    "shield": 0.03,
                    "obstacle": -0.02,
                    "shield_squared": 0.0,
                    "obstacle_squared": 0.0,
                    "shield_obstacle": 0.01,
                },
                "min_log_scale": -2.0,
                "max_log_scale": 2.0,
            }
        },
    }
    kernel = ContinuousKernel(
        mu_by_isotope={"TestIso": {"fe": 0.0, "pb": 0.0}},
        shield_params=ShieldParams(
            mu_fe=0.0,
            mu_pb=0.0,
            thickness_fe_cm=2.0,
            thickness_pb_cm=1.0,
        ),
        obstacle_grid=grid,
        detector_radius_m=0.04,
        detector_aperture_radius_m=0.03,
        detector_aperture_samples=9,
        line_mu_by_isotope=line_mu,
        transport_response_model=transport_model,
        use_gpu=False,
    )
    detector = np.array([0.0, 0.0, 0.0], dtype=float)
    source = np.array([1.0, 1.0, 1.0], dtype=float)

    terms = kernel.transport_response_terms_pair("TestIso", detector, source, 7, 5)
    reconstructed = sum(float(term["kernel"]) for term in terms)
    direct = kernel.kernel_value_pair("TestIso", detector, source, 7, 5)

    assert len(terms) == 9
    assert reconstructed == pytest.approx(direct, rel=1.0e-12, abs=1.0e-12)
    assert any(float(term["obstacle_tau_feature"]) > 0.0 for term in terms)
    assert any(float(term["shield_tau_feature"]) > 0.0 for term in terms)


def test_transport_response_base_terms_reconstruct_capped_base_kernel() -> None:
    """Base calibration terms should match capped no-sidecar runtime counts."""
    isotope = "TestIso"
    shield_params = ShieldParams(
        mu_fe=0.0,
        mu_pb=0.0,
        thickness_fe_cm=2.0,
        thickness_pb_cm=0.0,
        buildup_fe_coeff=30.0,
    )
    line_mu = {isotope: ({"weight": 1.0, "fe": 0.05, "pb": 0.0},)}
    transport_model = {
        "enabled": True,
        "by_isotope": {
            isotope: {
                "scale": 2.0,
                "tau_coefficients": {},
                "min_log_scale": -5.0,
                "max_log_scale": 5.0,
            }
        },
    }
    base_kernel = ContinuousKernel(
        mu_by_isotope={isotope: {"fe": 0.0, "pb": 0.0}},
        shield_params=shield_params,
        line_mu_by_isotope=line_mu,
        use_gpu=False,
    )
    response_kernel = ContinuousKernel(
        mu_by_isotope={isotope: {"fe": 0.0, "pb": 0.0}},
        shield_params=shield_params,
        line_mu_by_isotope=line_mu,
        transport_response_model=transport_model,
        use_gpu=False,
    )
    detector = np.zeros(3, dtype=float)
    source = np.array([1.0, 1.0, 1.0], dtype=float)

    terms = response_kernel.transport_response_terms_pair(
        isotope,
        detector,
        source,
        7,
        0,
    )

    assert sum(float(term["uncapped_kernel"]) for term in terms) > sum(
        float(term["base_kernel"]) for term in terms
    )
    assert sum(float(term["base_kernel"]) for term in terms) == pytest.approx(
        base_kernel.kernel_value_pair(isotope, detector, source, 7, 0),
        rel=1.0e-12,
    )
    assert sum(float(term["kernel"]) for term in terms) == pytest.approx(
        response_kernel.kernel_value_pair(isotope, detector, source, 7, 0),
        rel=1.0e-12,
    )


def test_cpu_attenuation_uses_explicit_aperture_radius_without_count_radius() -> None:
    """CPU attenuation should sample aperture rays independently of count radius."""
    kernel = ContinuousKernel(
        mu_by_isotope={"Cs-137": {"fe": 0.2, "pb": 0.0}},
        shield_params=ShieldParams(
            mu_fe=0.2,
            mu_pb=0.0,
            thickness_fe_cm=4.0,
            thickness_pb_cm=0.0,
            inner_radius_fe_cm=3.0,
        ),
        detector_radius_m=0.0,
        detector_aperture_radius_m=0.08,
        detector_aperture_samples=17,
        use_gpu=False,
    )
    detector = np.zeros(3, dtype=float)
    source = np.array([0.2, 0.2, 0.2], dtype=float)
    targets = kernel._detector_aperture_targets(source, detector)
    center_attenuation = kernel._attenuation_factor_for_target(
        "Cs-137",
        source,
        detector,
        detector,
        fe_index=7,
        pb_index=0,
    )
    aperture_attenuation = float(
        np.mean(
            [
                kernel._attenuation_factor_for_target(
                    "Cs-137",
                    source,
                    target,
                    detector,
                    fe_index=7,
                    pb_index=0,
                )
                for target in targets
            ]
        )
    )

    assert aperture_attenuation != pytest.approx(center_attenuation)
    assert kernel.attenuation_factor_pair(
        "Cs-137",
        source,
        detector,
        fe_index=7,
        pb_index=0,
    ) == pytest.approx(aperture_attenuation, rel=1.0e-12)


def test_background_added_to_expected_counts() -> None:
    """Background should add directly to expected rate/counts."""
    kernel = ContinuousKernel()
    det = np.array([0.0, 0.0, 0.0])
    src = np.array([10.0, 0.0, 0.0])
    counts = kernel.expected_counts(
        isotope="Cs-137",
        detector_pos=det,
        sources=np.array([src]),
        strengths=np.array([0.0]),
        orient_idx=0,
        live_time_s=2.0,
        background=5.0,
    )
    assert np.isclose(counts, 10.0, rtol=1e-6)


def test_detector_cone_aperture_targets_match_outer_sphere() -> None:
    """Cone aperture targets should match the Geant4 detector-cone geometry."""
    detector = np.array([0.0, 0.0, 0.0], dtype=float)
    source = np.array([2.0, 0.3, -0.4], dtype=float)
    aperture_radius = 0.052
    kernel = ContinuousKernel(
        detector_radius_m=0.05,
        detector_aperture_radius_m=aperture_radius,
        detector_aperture_samples=17,
        detector_aperture_sampling="solid_angle_cone",
        use_gpu=False,
    )

    targets = kernel._detector_aperture_targets(source, detector)
    target_radii = np.linalg.norm(targets - detector, axis=1)
    ray_dirs = targets - source
    ray_dirs /= np.linalg.norm(ray_dirs, axis=1, keepdims=True)
    axis = detector - source
    axis /= np.linalg.norm(axis)
    ray_angles = np.arccos(np.clip(ray_dirs @ axis, -1.0, 1.0))
    max_angle = np.arcsin(aperture_radius / np.linalg.norm(detector - source))

    assert targets.shape == (17, 3)
    assert np.allclose(target_radii, aperture_radius, rtol=0.0, atol=1.0e-10)
    assert float(np.max(ray_angles)) <= float(max_angle) + 1.0e-10
    assert np.linalg.matrix_rank(targets - targets.mean(axis=0)) == 3


def test_expected_counts_single_isotope_attenuation_levels() -> None:
    """Fe/Pb blocking should scale expected counts via exp(-mu*L)."""
    det = np.array([0.0, 0.0, 0.0])
    src = np.array([[1.0, 1.0, 1.0]])
    strengths = np.array([10.0])
    # Orientation normal aligned with direction (-,-,-) from src to det
    orient_block = np.array([-1.0, -1.0, -1.0])
    orient_free = np.array([1.0, 1.0, 1.0])
    base = expected_counts_single_isotope(
        detector_position=det,
        RFe=orient_free,
        RPb=orient_free,
        sources=src,
        strengths=strengths,
        background=0.0,
        duration=1.0,
        isotope_id="Cs-137",
        use_gpu=False,
    )
    shield_params = ShieldParams()
    expected_fe_ratio = np.exp(-(shield_params.mu_fe * shield_params.thickness_fe_cm))
    expected_both_ratio = np.exp(
        -(shield_params.mu_fe * shield_params.thickness_fe_cm + shield_params.mu_pb * shield_params.thickness_pb_cm)
    )
    fe_only = expected_counts_single_isotope(
        detector_position=det,
        RFe=orient_block,
        RPb=orient_free,
        sources=src,
        strengths=strengths,
        background=0.0,
        duration=1.0,
        isotope_id="Cs-137",
        use_gpu=False,
    )
    both = expected_counts_single_isotope(
        detector_position=det,
        RFe=orient_block,
        RPb=orient_block,
        sources=src,
        strengths=strengths,
        background=0.0,
        duration=1.0,
        isotope_id="Cs-137",
        use_gpu=False,
    )
    assert np.isclose(fe_only, expected_fe_ratio * base, rtol=1e-6)
    assert np.isclose(both, expected_both_ratio * base, rtol=1e-6)


def test_expected_counts_pair_matches_single_helper() -> None:
    """expected_counts_pair should match expected_counts_single_isotope with corresponding RFe/RPb."""
    kernel = ContinuousKernel()
    det = np.array([1.0, 1.0, 1.0])
    sources = np.array([[0.0, 0.0, 0.0]])
    strengths = np.array([1.0])
    fe_idx = 0  # (+,+,+) blocks
    pb_idx = 7  # (-,-,-) free
    pair_counts = kernel.expected_counts_pair(
        isotope="Cs-137",
        detector_pos=det,
        sources=sources,
        strengths=strengths,
        fe_index=fe_idx,
        pb_index=pb_idx,
        live_time_s=1.0,
        background=0.0,
    )
    from measurement.shielding import generate_octant_rotation_matrices

    mats = generate_octant_rotation_matrices()
    single = expected_counts_single_isotope(
        detector_position=det,
        RFe=mats[fe_idx],
        RPb=mats[pb_idx],
        sources=sources,
        strengths=strengths,
        background=0.0,
        duration=1.0,
        isotope_id="Cs-137",
        kernel=kernel,
    )
    assert pair_counts == pytest.approx(single, rel=1e-12)


def test_concrete_obstacle_path_reduces_kernel_value() -> None:
    """A blocked concrete cell should attenuate the source-detector kernel by its path length."""
    grid = ObstacleGrid(
        origin=(0.0, -0.5),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=((0, 0),),
    )
    shield_params = ShieldParams(mu_fe=0.0, mu_pb=0.0)
    kernel = ContinuousKernel(
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
        shield_params=shield_params,
        obstacle_grid=grid,
        obstacle_height_m=2.0,
        obstacle_mu_by_isotope={"Cs-137": 0.01},
        use_gpu=False,
    )
    free_kernel = ContinuousKernel(
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
        shield_params=shield_params,
        use_gpu=False,
    )
    source = np.array([-1.0, 0.0, 1.0], dtype=float)
    detector = np.array([2.0, 0.0, 1.0], dtype=float)

    assert kernel.obstacle_path_length_cm(source, detector) == pytest.approx(100.0)
    blocked = kernel.kernel_value_pair("Cs-137", detector, source, 0, 0)
    unblocked = free_kernel.kernel_value_pair("Cs-137", detector, source, 0, 0)
    assert blocked == pytest.approx(unblocked * np.exp(-1.0), rel=1e-12)


def test_obstacle_only_optical_depth_diagnostics_match_kernel() -> None:
    """Public obstacle diagnostics should expose the same attenuation used by the kernel."""
    grid = ObstacleGrid(
        origin=(0.0, -0.5),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=((0, 0),),
    )
    kernel = ContinuousKernel(
        obstacle_grid=grid,
        obstacle_height_m=2.0,
        obstacle_mu_by_isotope={"Cs-137": 0.01},
        use_gpu=False,
    )
    source = np.array([-1.0, 0.0, 1.0], dtype=float)
    detector = np.array([2.0, 0.0, 1.0], dtype=float)

    tau = kernel.obstacle_optical_depth_pair("Cs-137", source, detector)

    assert tau == pytest.approx(1.0)
    assert kernel.obstacle_log_attenuation_pair(
        "Cs-137",
        source,
        detector,
    ) == pytest.approx(-1.0)
    assert kernel.obstacle_attenuation_factor_pair(
        "Cs-137",
        source,
        detector,
    ) == pytest.approx(np.exp(-1.0))


def test_source_extent_obstacle_area_average_reduces_grazing_overattenuation() -> None:
    """Source extent sampling should expose partial obstacle occlusion."""
    grid = ObstacleGrid(
        origin=(0.0, -0.05),
        cell_size=0.1,
        grid_shape=(1, 1),
        blocked_cells=((0, 0),),
    )
    center_kernel = ContinuousKernel(
        obstacle_grid=grid,
        obstacle_height_m=2.0,
        obstacle_mu_by_isotope={"Cs-137": 0.1},
        use_gpu=False,
    )
    area_kernel = ContinuousKernel(
        obstacle_grid=grid,
        obstacle_height_m=2.0,
        obstacle_mu_by_isotope={"Cs-137": 0.1},
        source_extent_radius_m=0.2,
        source_extent_samples=9,
        use_gpu=False,
    )
    source = np.array([-1.0, 0.0, 1.0], dtype=float)
    detector = np.array([2.0, 0.0, 1.0], dtype=float)

    center_tau = center_kernel.obstacle_optical_depth_pair(
        "Cs-137",
        source,
        detector,
    )
    area_tau = area_kernel.obstacle_area_averaged_optical_depth_pair(
        "Cs-137",
        source,
        detector,
    )

    assert center_tau > 0.0
    assert 0.0 <= area_tau < center_tau
    assert area_kernel.obstacle_area_averaged_attenuation_pair(
        "Cs-137",
        source,
        detector,
    ) > center_kernel.obstacle_attenuation_factor_pair(
        "Cs-137",
        source,
        detector,
    )


def test_obstacle_log_attenuation_matrix_matches_pair_diagnostic() -> None:
    """Batched obstacle diagnostics should match the shared single-ray kernel."""
    grid = ObstacleGrid(
        origin=(0.0, -0.5),
        cell_size=1.0,
        grid_shape=(3, 1),
        blocked_cells=((0, 0), (1, 0), (2, 0)),
    )
    kernel = ContinuousKernel(
        obstacle_grid=grid,
        obstacle_height_m=2.0,
        obstacle_mu_by_isotope={"Cs-137": 0.01},
        use_gpu=False,
    )
    sources = np.asarray(
        [
            [-1.0, 0.0, 1.0],
            [-1.0, 0.4, 1.0],
        ],
        dtype=float,
    )
    detectors = np.asarray(
        [
            [4.0, 0.0, 1.0],
            [4.0, 0.4, 1.0],
            [4.0, 1.5, 1.0],
        ],
        dtype=float,
    )

    matrix = kernel.obstacle_log_attenuation_matrix(
        "Cs-137",
        sources,
        detectors,
        element_budget=2,
    )
    expected = np.asarray(
        [
            [
                kernel.obstacle_log_attenuation_pair("Cs-137", source, detector)
                for source in sources
            ]
            for detector in detectors
        ],
        dtype=float,
    )
    wide_chunk = kernel.obstacle_log_attenuation_matrix(
        "Cs-137",
        sources,
        detectors,
        element_budget=10_000,
    )

    np.testing.assert_allclose(matrix, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(wide_chunk, expected, rtol=1.0e-12, atol=1.0e-12)


def test_broad_beam_buildup_increases_but_bounds_attenuated_counts() -> None:
    """Build-up should increase attenuated broad-beam counts without exceeding unattenuated counts."""
    detector = np.zeros(3, dtype=float)
    source = np.array([1.0, 1.0, 1.0], dtype=float)
    base_params = ShieldParams(mu_fe=0.1, mu_pb=0.0, thickness_fe_cm=5.0, thickness_pb_cm=0.0)
    narrow_kernel = ContinuousKernel(
        mu_by_isotope={"Cs-137": {"fe": 0.1, "pb": 0.0}},
        shield_params=base_params,
        use_gpu=False,
    )
    buildup_kernel = ContinuousKernel(
        mu_by_isotope={"Cs-137": {"fe": 0.1, "pb": 0.0}},
        shield_params=ShieldParams(
            mu_fe=0.1,
            mu_pb=0.0,
            thickness_fe_cm=5.0,
            thickness_pb_cm=0.0,
            buildup_fe_coeff=0.5,
        ),
        use_gpu=False,
    )
    free_kernel = ContinuousKernel(
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
        use_gpu=False,
    )

    narrow = narrow_kernel.kernel_value_pair("Cs-137", detector, source, 7, 0)
    buildup = buildup_kernel.kernel_value_pair("Cs-137", detector, source, 7, 0)
    free = free_kernel.kernel_value_pair("Cs-137", detector, source, 7, 0)

    assert buildup > narrow
    assert buildup <= free


def test_spherical_shell_path_uses_radial_overlap_near_detector() -> None:
    """Shield path length should use exact radial overlap with the spherical shell."""
    shield_params = ShieldParams(
        mu_fe=0.1,
        mu_pb=0.0,
        thickness_fe_cm=5.0,
        thickness_pb_cm=0.0,
        inner_radius_fe_cm=DEFAULT_FE_SHIELD_INNER_RADIUS_CM,
        inner_radius_pb_cm=DEFAULT_PB_SHIELD_INNER_RADIUS_CM,
    )
    kernel = ContinuousKernel(
        mu_by_isotope={"Cs-137": {"fe": 0.1, "pb": 0.0}},
        shield_params=shield_params,
        use_gpu=False,
    )
    detector = np.zeros(3, dtype=float)
    direction = np.array([1.0, 1.0, 1.0], dtype=float) / np.sqrt(3.0)
    source_distance_m = (DEFAULT_FE_SHIELD_INNER_RADIUS_CM + 0.55) / 100.0
    source = direction * source_distance_m

    attenuation = kernel.attenuation_factor_pair(
        "Cs-137",
        source,
        detector,
        fe_index=7,
        pb_index=0,
    )
    assert attenuation == pytest.approx(np.exp(-0.1 * 0.55), rel=1e-12)


def test_concrete_obstacle_misses_off_axis_ray() -> None:
    """Obstacle attenuation should be unity when the ray does not cross blocked cells."""
    grid = ObstacleGrid(
        origin=(0.0, -0.5),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=((0, 0),),
    )
    kernel = ContinuousKernel(
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
        obstacle_grid=grid,
        obstacle_height_m=2.0,
        obstacle_mu_by_isotope={"Cs-137": 0.01},
        use_gpu=False,
    )
    source = np.array([-1.0, 2.0, 1.0], dtype=float)
    detector = np.array([2.0, 2.0, 1.0], dtype=float)

    assert kernel.obstacle_path_length_cm(source, detector) == pytest.approx(0.0)
    assert kernel.attenuation_factor_pair("Cs-137", source, detector, 0, 0) == pytest.approx(1.0)


def test_gpu_expected_counts_include_obstacle_attenuation_on_cpu_device() -> None:
    """The torch expected-count path should apply the same obstacle attenuation."""
    torch = pytest.importorskip("torch")
    device = torch.device("cpu")
    dtype = torch.float64
    positions = torch.as_tensor([[[-1.0, 0.0, 1.0]]], device=device, dtype=dtype)
    strengths = torch.as_tensor([[9.0]], device=device, dtype=dtype)
    backgrounds = torch.zeros(1, device=device, dtype=dtype)
    mask = torch.ones(1, 1, device=device, dtype=dtype)
    detector = np.array([2.0, 0.0, 1.0], dtype=float)
    boxes = np.array([[0.0, -0.5, 0.0, 1.0, 0.5, 2.0]], dtype=float)

    counts = expected_counts_pair_torch(
        detector_pos=detector,
        positions=positions,
        strengths=strengths,
        backgrounds=backgrounds,
        mask=mask,
        fe_index=0,
        pb_index=0,
        mu_fe=0.0,
        mu_pb=0.0,
        thickness_fe_cm=0.0,
        thickness_pb_cm=0.0,
        live_time_s=1.0,
        device=device,
        dtype=dtype,
        obstacle_boxes_m=boxes,
        obstacle_mu_cm_inv=0.01,
    )
    expected = (9.0 / 9.0) * np.exp(-1.0)
    assert float(counts[0]) == pytest.approx(expected, rel=1e-12)


def test_gpu_expected_counts_all_pairs_matches_pair_loop() -> None:
    """The batched all-pair GPU helper should match scalar pair evaluation."""
    torch = pytest.importorskip("torch")
    device = torch.device("cpu")
    dtype = torch.float64
    positions = torch.as_tensor(
        [
            [[-1.0, 0.0, 1.0], [-0.5, 0.4, 0.9]],
            [[1.2, -0.8, 1.1], [0.2, 1.1, 0.7]],
        ],
        device=device,
        dtype=dtype,
    )
    strengths = torch.as_tensor(
        [[9.0, 4.0], [6.0, 3.0]],
        device=device,
        dtype=dtype,
    )
    backgrounds = torch.as_tensor([0.1, 0.2], device=device, dtype=dtype)
    mask = torch.ones(2, 2, device=device, dtype=dtype)
    detector = np.array([2.0, 0.25, 1.0], dtype=float)
    boxes = np.array(
        [
            [0.0, -0.5, 0.0, 1.0, 0.5, 2.0],
            [0.5, 0.7, 0.0, 1.0, 1.2, 1.5],
        ],
        dtype=float,
    )
    for aperture_samples in (1, 3):
        all_pairs = expected_counts_all_pairs_torch(
            detector_pos=detector,
            positions=positions,
            strengths=strengths,
            backgrounds=backgrounds,
            mask=mask,
            mu_fe=0.03,
            mu_pb=0.06,
            thickness_fe_cm=1.5,
            thickness_pb_cm=0.8,
            inner_radius_fe_cm=3.0,
            inner_radius_pb_cm=4.5,
            live_time_s=2.0,
            device=device,
            dtype=dtype,
            obstacle_boxes_m=boxes,
            obstacle_mu_cm_inv=0.01,
            detector_radius_m=0.04,
            detector_aperture_samples=aperture_samples,
            buildup_fe_coeff=0.02,
            buildup_pb_coeff=0.01,
            obstacle_buildup_coeff=0.03,
        )
        rows = []
        for fe_idx in range(8):
            for pb_idx in range(8):
                rows.append(
                    expected_counts_pair_torch(
                        detector_pos=detector,
                        positions=positions,
                        strengths=strengths,
                        backgrounds=backgrounds,
                        mask=mask,
                        fe_index=fe_idx,
                        pb_index=pb_idx,
                        mu_fe=0.03,
                        mu_pb=0.06,
                        thickness_fe_cm=1.5,
                        thickness_pb_cm=0.8,
                        inner_radius_fe_cm=3.0,
                        inner_radius_pb_cm=4.5,
                        live_time_s=2.0,
                        device=device,
                        dtype=dtype,
                        obstacle_boxes_m=boxes,
                        obstacle_mu_cm_inv=0.01,
                        detector_radius_m=0.04,
                        detector_aperture_samples=aperture_samples,
                        buildup_fe_coeff=0.02,
                        buildup_pb_coeff=0.01,
                        obstacle_buildup_coeff=0.03,
                    )
                )
        pair_loop = torch.stack(rows, dim=0)
        assert all_pairs.shape == pair_loop.shape
        assert torch.allclose(all_pairs, pair_loop, rtol=1e-10, atol=1e-10)


def test_gpu_expected_counts_use_exact_spherical_shell_overlap() -> None:
    """The torch expected-count path should match exact spherical-shell path length."""
    torch = pytest.importorskip("torch")
    device = torch.device("cpu")
    dtype = torch.float64
    direction = np.array([1.0, 1.0, 1.0], dtype=float) / np.sqrt(3.0)
    source_distance_m = (DEFAULT_FE_SHIELD_INNER_RADIUS_CM + 0.55) / 100.0
    source = direction * source_distance_m
    positions = torch.as_tensor(source.reshape(1, 1, 3), device=device, dtype=dtype)
    strengths = torch.ones(1, 1, device=device, dtype=dtype)
    backgrounds = torch.zeros(1, device=device, dtype=dtype)
    mask = torch.ones(1, 1, device=device, dtype=dtype)

    counts = expected_counts_pair_torch(
        detector_pos=np.zeros(3, dtype=float),
        positions=positions,
        strengths=strengths,
        backgrounds=backgrounds,
        mask=mask,
        fe_index=7,
        pb_index=0,
        mu_fe=0.1,
        mu_pb=0.0,
        thickness_fe_cm=5.0,
        thickness_pb_cm=0.0,
        inner_radius_fe_cm=DEFAULT_FE_SHIELD_INNER_RADIUS_CM,
        inner_radius_pb_cm=DEFAULT_PB_SHIELD_INNER_RADIUS_CM,
        live_time_s=1.0,
        device=device,
        dtype=dtype,
    )
    expected = (1.0 / (source_distance_m**2)) * np.exp(-0.055)
    assert float(counts[0]) == pytest.approx(expected, rel=1e-12)


def test_continuous_kernel_cuda_matches_cpu_with_detector_aperture() -> None:
    """ContinuousKernel CUDA path should match the CPU finite-aperture geometry."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    from measurement.shielding import HVL_TVL_TABLE_MM, mu_by_isotope_from_tvl_mm

    rng = np.random.default_rng(77)
    detector = np.array([0.2, -0.3, 1.0], dtype=float)
    directions = rng.normal(size=(16, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    distances = rng.uniform(0.8, 3.5, size=(16, 1))
    sources = detector + directions * distances
    strengths = rng.uniform(500.0, 30000.0, size=16)
    mu = mu_by_isotope_from_tvl_mm(HVL_TVL_TABLE_MM)
    cpu_kernel = ContinuousKernel(
        mu_by_isotope=mu,
        use_gpu=False,
        detector_radius_m=0.038,
        detector_aperture_samples=31,
    )
    gpu_kernel = ContinuousKernel(
        mu_by_isotope=mu,
        use_gpu=True,
        gpu_device="cuda",
        gpu_dtype="float64",
        detector_radius_m=0.038,
        detector_aperture_samples=31,
    )

    for fe_index in range(8):
        for pb_index in range(8):
            cpu_counts = cpu_kernel.expected_counts_pair(
                "Cs-137",
                detector,
                sources,
                strengths,
                fe_index,
                pb_index,
                live_time_s=1.0,
            )
            gpu_counts = gpu_kernel.expected_counts_pair(
                "Cs-137",
                detector,
                sources,
                strengths,
                fe_index,
                pb_index,
                live_time_s=1.0,
            )
            assert gpu_counts == pytest.approx(cpu_counts, rel=2e-6, abs=1e-6)


def test_kernel_values_all_pairs_matches_pair_values_cpu() -> None:
    """All-pair kernel evaluation should match per-pair CPU evaluations."""
    from measurement.shielding import HVL_TVL_TABLE_MM, mu_by_isotope_from_tvl_mm

    detector = np.array([0.1, -0.2, 0.8], dtype=float)
    sources = np.array(
        [
            [1.0, 0.2, 0.8],
            [-0.5, 1.4, 1.0],
            [2.0, -1.0, 0.7],
        ],
        dtype=float,
    )
    mu = mu_by_isotope_from_tvl_mm(HVL_TVL_TABLE_MM)
    kernel = ContinuousKernel(
        mu_by_isotope=mu,
        use_gpu=False,
        detector_radius_m=0.038,
        detector_aperture_samples=5,
    )

    all_pair_values = kernel.kernel_values_all_pairs("Cs-137", detector, sources)
    assert all_pair_values.shape == (64, sources.shape[0])

    for fe_index in range(8):
        for pb_index in range(8):
            pair_id = fe_index * 8 + pb_index
            pair_values = kernel.kernel_values_pair(
                "Cs-137",
                detector,
                sources,
                fe_index,
                pb_index,
            )
            assert all_pair_values[pair_id] == pytest.approx(
                pair_values,
                rel=1.0e-12,
                abs=1.0e-12,
            )


def test_kernel_values_all_pairs_for_detectors_matches_cpu_pairs() -> None:
    """Batched detector all-pair evaluation should match scalar CPU calls."""
    from measurement.shielding import HVL_TVL_TABLE_MM, mu_by_isotope_from_tvl_mm

    detectors = np.array(
        [
            [0.1, -0.2, 0.8],
            [1.2, 0.4, 1.1],
            [-0.6, 0.7, 0.9],
        ],
        dtype=float,
    )
    sources = np.array(
        [
            [1.0, 0.2, 0.8],
            [-0.5, 1.4, 1.0],
            [2.0, -1.0, 0.7],
        ],
        dtype=float,
    )
    mu = mu_by_isotope_from_tvl_mm(HVL_TVL_TABLE_MM)
    kernel = ContinuousKernel(
        mu_by_isotope=mu,
        use_gpu=False,
        detector_radius_m=0.038,
        detector_aperture_samples=5,
    )

    batched = kernel.kernel_values_all_pairs_for_detectors(
        "Cs-137",
        detectors,
        sources,
    )

    assert batched.shape == (detectors.shape[0], 64, sources.shape[0])
    for pose_idx, detector in enumerate(detectors):
        expected = kernel.kernel_values_all_pairs("Cs-137", detector, sources)
        assert batched[pose_idx] == pytest.approx(
            expected,
            rel=1.0e-12,
            abs=1.0e-12,
        )


def test_kernel_values_selected_pairs_for_detectors_matches_cpu_pairs() -> None:
    """Batched selected-pair detector evaluation should match scalar CPU calls."""
    from measurement.shielding import HVL_TVL_TABLE_MM, mu_by_isotope_from_tvl_mm

    detectors = np.array(
        [
            [0.1, -0.2, 0.8],
            [1.2, 0.4, 1.1],
            [-0.6, 0.7, 0.9],
        ],
        dtype=float,
    )
    sources = np.array(
        [
            [1.0, 0.2, 0.8],
            [-0.5, 1.4, 1.0],
            [2.0, -1.0, 0.7],
        ],
        dtype=float,
    )
    fe_indices = np.array([0, 3, 7], dtype=int)
    pb_indices = np.array([7, 2, 0], dtype=int)
    mu = mu_by_isotope_from_tvl_mm(HVL_TVL_TABLE_MM)
    kernel = ContinuousKernel(
        mu_by_isotope=mu,
        use_gpu=False,
        detector_radius_m=0.038,
        detector_aperture_samples=5,
    )

    batched = kernel.kernel_values_selected_pairs_for_detectors(
        "Cs-137",
        detectors,
        sources,
        fe_indices,
        pb_indices,
    )

    assert batched.shape == (detectors.shape[0], sources.shape[0])
    for pose_idx, detector in enumerate(detectors):
        expected = kernel.kernel_values_pair(
            "Cs-137",
            detector,
            sources,
            int(fe_indices[pose_idx]),
            int(pb_indices[pose_idx]),
        )
        assert batched[pose_idx] == pytest.approx(
            expected,
            rel=1.0e-12,
            abs=1.0e-12,
        )


def test_kernel_values_all_pairs_cuda_matches_cpu_with_obstacles_and_aperture() -> None:
    """All-pair CUDA kernel evaluation should match the CPU observation model."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    from measurement.shielding import HVL_TVL_TABLE_MM, mu_by_isotope_from_tvl_mm

    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(4, 4),
        blocked_cells=((1, 1), (2, 1), (1, 2)),
    )
    detector = np.array([2.2, 2.2, 1.0], dtype=float)
    sources = np.array(
        [
            [0.2, 0.2, 1.0],
            [0.5, 3.5, 1.0],
            [3.5, 0.5, 1.0],
            [4.5, 2.0, 1.0],
        ],
        dtype=float,
    )
    mu = mu_by_isotope_from_tvl_mm(HVL_TVL_TABLE_MM)
    cpu_kernel = ContinuousKernel(
        mu_by_isotope=mu,
        obstacle_grid=grid,
        obstacle_mu_by_isotope={"Cs-137": 0.17},
        use_gpu=False,
        detector_radius_m=0.038,
        detector_aperture_samples=7,
        source_extent_radius_m=0.08,
        source_extent_samples=5,
    )
    gpu_kernel = ContinuousKernel(
        mu_by_isotope=mu,
        obstacle_grid=grid,
        obstacle_mu_by_isotope={"Cs-137": 0.17},
        use_gpu=True,
        gpu_device="cuda",
        gpu_dtype="float64",
        detector_radius_m=0.038,
        detector_aperture_samples=7,
        source_extent_radius_m=0.08,
        source_extent_samples=5,
    )

    cpu_values = cpu_kernel.kernel_values_all_pairs("Cs-137", detector, sources)
    gpu_values = gpu_kernel.kernel_values_all_pairs("Cs-137", detector, sources)

    assert gpu_values == pytest.approx(cpu_values, rel=1.0e-10, abs=1.0e-10)


def test_kernel_values_all_pairs_for_detectors_cuda_matches_cpu() -> None:
    """Batched detector CUDA all-pair kernels should match CPU results."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    from measurement.shielding import HVL_TVL_TABLE_MM, mu_by_isotope_from_tvl_mm

    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(4, 4),
        blocked_cells=((1, 1), (2, 1), (1, 2)),
    )
    detectors = np.array(
        [
            [2.2, 2.2, 1.0],
            [3.2, 1.4, 1.1],
        ],
        dtype=float,
    )
    sources = np.array(
        [
            [0.2, 0.2, 1.0],
            [0.5, 3.5, 1.0],
            [3.5, 0.5, 1.0],
            [4.5, 2.0, 1.0],
        ],
        dtype=float,
    )
    mu = mu_by_isotope_from_tvl_mm(HVL_TVL_TABLE_MM)
    cpu_kernel = ContinuousKernel(
        mu_by_isotope=mu,
        obstacle_grid=grid,
        obstacle_mu_by_isotope={"Cs-137": 0.17},
        use_gpu=False,
        detector_radius_m=0.038,
        detector_aperture_samples=7,
    )
    gpu_kernel = ContinuousKernel(
        mu_by_isotope=mu,
        obstacle_grid=grid,
        obstacle_mu_by_isotope={"Cs-137": 0.17},
        use_gpu=True,
        gpu_device="cuda",
        gpu_dtype="float64",
        detector_radius_m=0.038,
        detector_aperture_samples=7,
    )

    cpu_values = cpu_kernel.kernel_values_all_pairs_for_detectors(
        "Cs-137",
        detectors,
        sources,
    )
    gpu_values = gpu_kernel.kernel_values_all_pairs_for_detectors(
        "Cs-137",
        detectors,
        sources,
    )

    assert gpu_values == pytest.approx(cpu_values, rel=1.0e-10, abs=1.0e-10)


def test_kernel_values_selected_pairs_for_detectors_cuda_matches_cpu() -> None:
    """Batched selected-pair CUDA detector kernels should match CPU results."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    from measurement.shielding import HVL_TVL_TABLE_MM, mu_by_isotope_from_tvl_mm

    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(4, 4),
        blocked_cells=((1, 1), (2, 1), (1, 2)),
    )
    detectors = np.array(
        [
            [2.2, 2.2, 1.0],
            [3.2, 1.4, 1.1],
            [1.2, 3.4, 0.9],
        ],
        dtype=float,
    )
    sources = np.array(
        [
            [0.2, 0.2, 1.0],
            [0.5, 3.5, 1.0],
            [3.5, 0.5, 1.0],
            [4.5, 2.0, 1.0],
        ],
        dtype=float,
    )
    fe_indices = np.array([0, 4, 7], dtype=int)
    pb_indices = np.array([7, 3, 1], dtype=int)
    mu = mu_by_isotope_from_tvl_mm(HVL_TVL_TABLE_MM)
    cpu_kernel = ContinuousKernel(
        mu_by_isotope=mu,
        obstacle_grid=grid,
        obstacle_mu_by_isotope={"Cs-137": 0.17},
        use_gpu=False,
        detector_radius_m=0.038,
        detector_aperture_samples=7,
        source_extent_radius_m=0.08,
        source_extent_samples=5,
    )
    gpu_kernel = ContinuousKernel(
        mu_by_isotope=mu,
        obstacle_grid=grid,
        obstacle_mu_by_isotope={"Cs-137": 0.17},
        use_gpu=True,
        gpu_device="cuda",
        gpu_dtype="float64",
        detector_radius_m=0.038,
        detector_aperture_samples=7,
        source_extent_radius_m=0.08,
        source_extent_samples=5,
    )

    cpu_values = cpu_kernel.kernel_values_selected_pairs_for_detectors(
        "Cs-137",
        detectors,
        sources,
        fe_indices,
        pb_indices,
    )
    gpu_values = gpu_kernel.kernel_values_selected_pairs_for_detectors(
        "Cs-137",
        detectors,
        sources,
        fe_indices,
        pb_indices,
    )

    assert gpu_values == pytest.approx(cpu_values, rel=1.0e-10, abs=1.0e-10)


def test_continuous_kernel_cuda_matches_cpu_with_obstacles_and_aperture() -> None:
    """ContinuousKernel CUDA path should match CPU obstacle attenuation over aperture rays."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    from measurement.shielding import HVL_TVL_TABLE_MM, mu_by_isotope_from_tvl_mm

    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(4, 4),
        blocked_cells=((1, 1), (2, 1), (1, 2)),
    )
    detector = np.array([2.2, 2.2, 1.0], dtype=float)
    sources = np.array(
        [
            [0.2, 0.2, 1.0],
            [0.5, 3.5, 1.0],
            [3.5, 0.5, 1.0],
            [4.5, 2.0, 1.0],
        ],
        dtype=float,
    )
    strengths = np.array([10000.0, 20000.0, 15000.0, 12000.0], dtype=float)
    mu = mu_by_isotope_from_tvl_mm(HVL_TVL_TABLE_MM)
    cpu_kernel = ContinuousKernel(
        mu_by_isotope=mu,
        obstacle_grid=grid,
        obstacle_mu_by_isotope={"Cs-137": 0.17},
        use_gpu=False,
        detector_radius_m=0.038,
        detector_aperture_samples=31,
    )
    gpu_kernel = ContinuousKernel(
        mu_by_isotope=mu,
        obstacle_grid=grid,
        obstacle_mu_by_isotope={"Cs-137": 0.17},
        use_gpu=True,
        gpu_device="cuda",
        gpu_dtype="float64",
        detector_radius_m=0.038,
        detector_aperture_samples=31,
    )

    for fe_index in range(8):
        for pb_index in range(8):
            cpu_counts = cpu_kernel.expected_counts_pair(
                "Cs-137",
                detector,
                sources,
                strengths,
                fe_index,
                pb_index,
            )
            gpu_counts = gpu_kernel.expected_counts_pair(
                "Cs-137",
                detector,
                sources,
                strengths,
                fe_index,
                pb_index,
            )
            assert gpu_counts == pytest.approx(cpu_counts, rel=1e-10, abs=1e-10)


def test_gpu_chunk_size_accounts_for_obstacle_aperture_expansion() -> None:
    """GPU batching should shrink when obstacle-aperture tensors get large."""
    blocked = tuple((idx, 0) for idx in range(494))
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(494, 1),
        blocked_cells=blocked,
    )
    kernel = ContinuousKernel(
        obstacle_grid=grid,
        detector_radius_m=0.038,
        detector_aperture_samples=121,
        gpu_dtype="float64",
    )

    chunk = kernel._adaptive_torch_chunk_size(8192)

    assert chunk == 66
    assert chunk < 8192
