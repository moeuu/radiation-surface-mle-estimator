"""Shield geometry and attenuation tests for 1/8 spherical shells."""

import numpy as np
import pytest

from measurement.shielding import (
    DEFAULT_FE_SHIELD_THICKNESS_CM,
    DEFAULT_PB_SHIELD_THICKNESS_CM,
    DEFAULT_SHIELD_TRANSMISSION_TARGET,
    HVL_TVL_TABLE_MM,
    OCTANT_NORMALS,
    OctantShield,
    generate_octant_orientations,
    generate_octant_rotation_matrices,
    generate_fe_pb_orientation_pairs,
    line_resolved_shield_mu_by_isotope,
    octant_index_from_normal,
    cartesian_to_spherical,
    iron_shield,
    mu_by_isotope_from_hvl_mm,
    lead_shield,
    mu_by_isotope_from_tvl_mm,
    mu_from_hvl_mm,
    mu_from_tvl_mm,
    shield_blocks_radiation,
    transmission_from_hvl,
)
from measurement.kernels import KernelPrecomputer, ShieldParams


def test_cartesian_to_spherical_roundtrip_axes() -> None:
    """Basic spherical conversion sanity on unit axes."""
    r, theta, phi = cartesian_to_spherical(np.array([0.0, 0.0, 1.0]))
    assert r == 1.0
    assert abs(theta - 0.0) < 1e-6
    assert abs(phi - 0.0) < 1e-6


def test_shield_blocks_matching_octant_only() -> None:
    """A direction is blocked only when signs match the octant."""
    direction = np.array([1.0, 1.0, 1.0])
    assert shield_blocks_radiation(direction, OCTANT_NORMALS[0])
    assert not shield_blocks_radiation(direction, OCTANT_NORMALS[7])


def test_attenuation_respects_thickness() -> None:
    """Attenuation reduces flux when heading into the shield, none otherwise."""
    direction = np.array([1.0, 1.0, 1.0])
    pb = lead_shield(thickness_cm=2.0)
    fe = iron_shield(thickness_cm=2.0)
    blocked_pb = pb.attenuation_factor(direction, OCTANT_NORMALS[0])
    blocked_fe = fe.attenuation_factor(direction, OCTANT_NORMALS[0])
    unblocked = pb.attenuation_factor(direction, OCTANT_NORMALS[7])
    assert blocked_pb < 1.0
    assert blocked_fe < 1.0
    assert unblocked == 1.0


def test_octant_shield_blocks_ray_by_angles() -> None:
    """blocks_ray should map directions to correct octant via (theta, phi)."""
    shield = OctantShield()
    src = np.array([0.0, 0.0, 0.0])
    det_ppp = np.array([1.0, 1.0, 1.0])
    det_mmp = np.array([-1.0, -1.0, 1.0])
    assert shield.blocks_ray(detector_position=det_ppp, source_position=src, octant_index=0)
    assert not shield.blocks_ray(detector_position=det_ppp, source_position=src, octant_index=7)
    assert shield.blocks_ray(detector_position=det_mmp, source_position=src, octant_index=6)
    assert not shield.blocks_ray(detector_position=det_mmp, source_position=src, octant_index=1)


def test_kernel_attenuation_factor_applies_exponential_when_blocked() -> None:
    """KernelPrecomputer should attenuate according to exp(-mu*L) when blocked."""
    candidate_sources = np.array([[0.0, 0.0, 0.0]], dtype=float)
    poses = np.array([[1.0, 1.0, 1.0]], dtype=float)
    orientations = OCTANT_NORMALS
    shield_params = ShieldParams()
    mu = {"Cs-137": {"fe": shield_params.mu_fe, "pb": shield_params.mu_pb}}
    kernel = KernelPrecomputer(
        candidate_sources=candidate_sources,
        poses=poses,
        orientations=orientations,
        shield_params=shield_params,
        mu_by_isotope=mu,
    )
    k_block = kernel.kernel("Cs-137", pose_idx=0, orient_idx=0)[0]
    k_free = kernel.kernel("Cs-137", pose_idx=0, orient_idx=7)[0]
    expected_ratio = np.exp(
        -(shield_params.mu_fe * shield_params.thickness_fe_cm + shield_params.mu_pb * shield_params.thickness_pb_cm)
    )
    assert np.isclose(k_block, expected_ratio * k_free, rtol=1e-6)


def test_generate_octant_orientations_and_index() -> None:
    """Orientation helper should return 8 normals and index mapping should be consistent."""
    normals = generate_octant_orientations()
    assert normals.shape == (8, 3)
    for i, n in enumerate(normals):
        idx = octant_index_from_normal(n)
        assert idx == i


def test_generate_rotation_matrices_and_pairs() -> None:
    """Rotation matrices should be orthonormal with third column aligned to the octant normal; pair count = 8x8."""
    mats = generate_octant_rotation_matrices()
    assert mats.shape == (8, 3, 3)
    for n, m in zip(generate_octant_orientations(), mats):
        # third column = normalized normal
        n_unit = n / np.linalg.norm(n)
        assert np.allclose(m[:, 2], n_unit)
        # columns should be orthonormal
        assert np.allclose(m.T @ m, np.eye(3), atol=1e-6)
    pairs = generate_fe_pb_orientation_pairs()
    assert len(pairs) == 64
    assert pairs[0]["id"] == 0
    assert pairs[-1]["id"] == 63


def test_rotation_changes_counts_by_exponential_when_blocked() -> None:
    """
    Rotating from an unblocked to a blocked octant should attenuate counts via exp(-mu*L).

    This reproduces the qualitative behavior in IAS-19 Fig. 1: orientation
    modulates count rate and provides directional information.
    """
    candidate_sources = np.array([[0.0, 0.0, 0.0]], dtype=float)
    poses = np.array([[1.0, 1.0, 1.0]], dtype=float)
    orientations = generate_octant_orientations()
    shield_params = ShieldParams()
    mu = {"Cs-137": {"fe": shield_params.mu_fe, "pb": shield_params.mu_pb}}
    kernel = KernelPrecomputer(
        candidate_sources=candidate_sources,
        poses=poses,
        orientations=orientations,
        shield_params=shield_params,
        mu_by_isotope=mu,
    )
    strength = np.array([10.0])
    unblocked = kernel.expected_counts("Cs-137", pose_idx=0, orient_idx=7, source_strengths=strength, background=0.0)
    blocked = kernel.expected_counts("Cs-137", pose_idx=0, orient_idx=0, source_strengths=strength, background=0.0)
    expected_ratio = np.exp(
        -(shield_params.mu_fe * shield_params.thickness_fe_cm + shield_params.mu_pb * shield_params.thickness_pb_cm)
    )
    assert blocked == pytest.approx(expected_ratio * unblocked, rel=1e-6)


def test_tvl_table_mu_and_attenuation_factors() -> None:
    """TVL-derived mu values should reproduce attenuation at Cs-137 TVL thickness."""
    mu_by_isotope = mu_by_isotope_from_tvl_mm(
        HVL_TVL_TABLE_MM, isotopes=["Cs-137", "Co-60", "Eu-154"]
    )
    cs_pb_tvl = HVL_TVL_TABLE_MM["Cs-137"]["pb"]["tvl"]
    cs_fe_tvl = HVL_TVL_TABLE_MM["Cs-137"]["fe"]["tvl"]
    co_pb_tvl = HVL_TVL_TABLE_MM["Co-60"]["pb"]["tvl"]
    co_fe_tvl = HVL_TVL_TABLE_MM["Co-60"]["fe"]["tvl"]
    eu_pb_tvl = HVL_TVL_TABLE_MM["Eu-154"]["pb"]["tvl"]
    eu_fe_tvl = HVL_TVL_TABLE_MM["Eu-154"]["fe"]["tvl"]

    assert mu_by_isotope["Cs-137"]["pb"] == pytest.approx(mu_from_tvl_mm(cs_pb_tvl))
    assert mu_by_isotope["Cs-137"]["fe"] == pytest.approx(mu_from_tvl_mm(cs_fe_tvl))

    co_pb_factor = np.exp(-mu_by_isotope["Co-60"]["pb"] * (cs_pb_tvl / 10.0))
    eu_pb_factor = np.exp(-mu_by_isotope["Eu-154"]["pb"] * (cs_pb_tvl / 10.0))
    co_fe_factor = np.exp(-mu_by_isotope["Co-60"]["fe"] * (cs_fe_tvl / 10.0))
    eu_fe_factor = np.exp(-mu_by_isotope["Eu-154"]["fe"] * (cs_fe_tvl / 10.0))

    assert co_pb_factor == pytest.approx(10 ** (-(cs_pb_tvl / co_pb_tvl)))
    assert eu_pb_factor == pytest.approx(10 ** (-(cs_pb_tvl / eu_pb_tvl)))
    assert co_fe_factor == pytest.approx(10 ** (-(cs_fe_tvl / co_fe_tvl)))
    assert eu_fe_factor == pytest.approx(10 ** (-(cs_fe_tvl / eu_fe_tvl)))


def test_hvl_table_mu_and_attenuation_factors() -> None:
    """HVL-derived mu values should reproduce half-value attenuation exactly."""
    mu_by_isotope = mu_by_isotope_from_hvl_mm(
        HVL_TVL_TABLE_MM, isotopes=["Cs-137", "Co-60", "Eu-154"]
    )
    for isotope in ["Cs-137", "Co-60", "Eu-154"]:
        for material in ["pb", "fe"]:
            hvl_mm = HVL_TVL_TABLE_MM[isotope][material]["hvl"]
            hvl_cm = hvl_mm / 10.0
            assert mu_by_isotope[isotope][material] == pytest.approx(mu_from_hvl_mm(hvl_mm))
            assert np.exp(-mu_by_isotope[isotope][material] * hvl_cm) == pytest.approx(0.5)
            assert transmission_from_hvl(hvl_cm, hvl_mm) == pytest.approx(0.5)
            assert transmission_from_hvl(2.0 * hvl_cm, hvl_mm) == pytest.approx(0.25)


def test_default_shield_thickness_targets_cs137_one_fifth() -> None:
    """Default shield thicknesses should target one-fifth transmission for Cs-137 only."""
    shield_params = ShieldParams()
    mu_by_isotope = mu_by_isotope_from_tvl_mm(
        HVL_TVL_TABLE_MM, isotopes=["Cs-137", "Co-60", "Eu-154"]
    )

    assert shield_params.thickness_fe_cm == pytest.approx(DEFAULT_FE_SHIELD_THICKNESS_CM)
    assert shield_params.thickness_pb_cm == pytest.approx(DEFAULT_PB_SHIELD_THICKNESS_CM)
    assert np.exp(-shield_params.mu_fe * shield_params.thickness_fe_cm) == pytest.approx(
        DEFAULT_SHIELD_TRANSMISSION_TARGET
    )
    assert np.exp(-shield_params.mu_pb * shield_params.thickness_pb_cm) == pytest.approx(
        DEFAULT_SHIELD_TRANSMISSION_TARGET
    )
    for isotope in ["Co-60", "Eu-154"]:
        fe_factor = np.exp(-mu_by_isotope[isotope]["fe"] * shield_params.thickness_fe_cm)
        pb_factor = np.exp(-mu_by_isotope[isotope]["pb"] * shield_params.thickness_pb_cm)

        assert fe_factor != pytest.approx(DEFAULT_SHIELD_TRANSMISSION_TARGET)
        assert pb_factor != pytest.approx(DEFAULT_SHIELD_TRANSMISSION_TARGET)


def test_default_shield_thickness_is_consistent_with_cs137_hvl_theory() -> None:
    """Default Cs-137 shield thicknesses should stay consistent with HVL theory."""
    cs_fe_hvl = HVL_TVL_TABLE_MM["Cs-137"]["fe"]["hvl"]
    cs_pb_hvl = HVL_TVL_TABLE_MM["Cs-137"]["pb"]["hvl"]
    fe_factor = transmission_from_hvl(DEFAULT_FE_SHIELD_THICKNESS_CM, cs_fe_hvl)
    pb_factor = transmission_from_hvl(DEFAULT_PB_SHIELD_THICKNESS_CM, cs_pb_hvl)

    assert fe_factor == pytest.approx(DEFAULT_SHIELD_TRANSMISSION_TARGET, rel=0.03)
    assert pb_factor == pytest.approx(DEFAULT_SHIELD_TRANSMISSION_TARGET, rel=0.12)


def test_eu154_hvl_matches_configured_line_effective_attenuation() -> None:
    """Eu-154 HVL values should match the configured line-set attenuation scale."""
    eu_fe_hvl = HVL_TVL_TABLE_MM["Eu-154"]["fe"]["hvl"]
    eu_pb_hvl = HVL_TVL_TABLE_MM["Eu-154"]["pb"]["hvl"]

    assert eu_fe_hvl == pytest.approx(17.36)
    assert eu_pb_hvl == pytest.approx(8.45)
    assert transmission_from_hvl(DEFAULT_FE_SHIELD_THICKNESS_CM, eu_fe_hvl) == pytest.approx(
        0.247,
        rel=0.03,
    )
    assert transmission_from_hvl(DEFAULT_PB_SHIELD_THICKNESS_CM, eu_pb_hvl) == pytest.approx(
        0.284,
        rel=0.03,
    )


def test_line_resolved_shield_mu_uses_normalized_gamma_lines() -> None:
    """Line-resolved shield coefficients should mirror the Geant4 line mixture."""
    table = line_resolved_shield_mu_by_isotope(
        isotopes=["Cs-137", "Co-60", "Eu-154"],
        normalize_line_intensities=True,
    )

    assert set(table) == {"Cs-137", "Co-60", "Eu-154"}
    for isotope, rows in table.items():
        assert rows
        total_weight = sum(float(row["weight"]) for row in rows)
        assert total_weight == pytest.approx(1.0)
        for row in rows:
            assert float(row["energy_keV"]) > 0.0
            assert float(row["fe"]) > 0.0
            assert float(row["pb"]) > 0.0

    co_pb_values = {round(float(row["pb"]), 8) for row in table["Co-60"]}
    eu_pb_values = {round(float(row["pb"]), 8) for row in table["Eu-154"]}
    assert len(co_pb_values) > 1
    assert len(eu_pb_values) > 1
