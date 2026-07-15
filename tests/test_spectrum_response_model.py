"""Tests for response-matrix spectrum unmixing."""

import numpy as np
import pytest

from measurement.continuous_kernels import ContinuousKernel, geometric_term
from measurement.kernels import ShieldParams
from measurement.model import EnvironmentConfig, PointSource
from measurement.shielding import (
    HVL_TVL_TABLE_MM,
    generate_octant_orientations,
    mu_by_isotope_from_tvl_mm,
)
from spectrum.library import get_analysis_lines_with_intensity
from spectrum.pipeline import PhotopeakRoiEstimate, SpectralDecomposer, SpectrumConfig
from spectrum.response_matrix import build_incident_gamma_response_matrix, gaussian_peak


def test_photopeak_nnls_counts_recover_calibrated_peak_counts() -> None:
    """Photopeak NNLS should recover source counts from peaks plus local background."""
    cfg = SpectrumConfig(dead_time_tau_s=0.0)
    decomposer = SpectralDecomposer(cfg)
    energy_axis = decomposer.energy_axis
    bin_width = float(np.median(np.diff(energy_axis)))
    true_counts = {"Cs-137": 8000.0, "Co-60": 5000.0, "Eu-154": 3000.0}
    spectrum = 5.0 + 0.003 * energy_axis

    for isotope, source_count in true_counts.items():
        for energy, intensity in get_analysis_lines_with_intensity(
            isotope,
            decomposer.library,
            max_energy_keV=float(np.max(energy_axis)),
        ):
            if intensity < cfg.photopeak_min_line_intensity:
                continue
            sigma = decomposer.resolution_fn(float(energy))
            efficiency = decomposer.efficiency(float(energy))
            spectrum += (
                source_count
                * float(intensity)
                * efficiency
                * gaussian_peak(energy_axis, float(energy), sigma)
                * bin_width
            )

    counts = decomposer.compute_photopeak_nnls_counts(
        spectrum,
        live_time_s=1.0,
        isotopes=list(true_counts),
    )

    for isotope, expected in true_counts.items():
        assert counts[isotope] == pytest.approx(expected, rel=1e-3)


def test_photopeak_nnls_estimates_report_variance() -> None:
    """Photopeak NNLS should expose count uncertainty for PF likelihoods."""
    cfg = SpectrumConfig(dead_time_tau_s=0.0)
    decomposer = SpectralDecomposer(cfg)
    energy_axis = decomposer.energy_axis
    bin_width = float(np.median(np.diff(energy_axis)))
    true_counts = {"Cs-137": 2500.0}
    spectrum = 2.0 + np.zeros_like(energy_axis)

    for energy, intensity in get_analysis_lines_with_intensity(
        "Cs-137",
        decomposer.library,
        max_energy_keV=float(np.max(energy_axis)),
    ):
        if intensity < cfg.photopeak_min_line_intensity:
            continue
        sigma = decomposer.resolution_fn(float(energy))
        efficiency = decomposer.efficiency(float(energy))
        spectrum += (
            true_counts["Cs-137"]
            * float(intensity)
            * efficiency
            * gaussian_peak(energy_axis, float(energy), sigma)
            * bin_width
        )

    estimates = decomposer.compute_photopeak_nnls_estimates(
        spectrum,
        live_time_s=1.0,
        isotopes=["Cs-137"],
    )

    assert estimates["Cs-137"].counts == pytest.approx(true_counts["Cs-137"], rel=1e-3)
    assert estimates["Cs-137"].variance > 0.0
    assert decomposer.last_count_variances["Cs-137"] == pytest.approx(
        estimates["Cs-137"].variance,
    )


def test_response_poisson_counts_recover_full_response_mixture() -> None:
    """Full-spectrum Poisson regression should return source-equivalent photopeak counts."""
    decomposer = SpectralDecomposer(SpectrumConfig(dead_time_tau_s=0.0))
    isotopes = ["Cs-137", "Co-60", "Eu-154"]
    truth = {"Cs-137": 1800.0, "Co-60": 1200.0, "Eu-154": 900.0}
    indices = [decomposer.isotope_names.index(isotope) for isotope in isotopes]
    spectrum = np.zeros_like(decomposer.energy_axis, dtype=float)
    for isotope, index in zip(isotopes, indices):
        spectrum += truth[isotope] * decomposer.response_matrix[:, index]

    estimates = decomposer.compute_response_poisson_estimates(
        spectrum,
        isotopes=isotopes,
        include_background=False,
    )

    for isotope, expected in truth.items():
        index = decomposer.isotope_names.index(isotope)
        photopeak_integral = float(np.sum(decomposer._get_photopeak_response_matrix()[:, index]))
        assert estimates[isotope].counts == pytest.approx(expected, rel=1e-4)
        assert np.sum(decomposer.last_response_poisson_components[isotope]) == pytest.approx(
            expected * photopeak_integral,
            rel=1e-4,
        )
        assert estimates[isotope].variance > 0.0
        assert estimates[isotope].method == "response_poisson_source_equivalent"
    assert float(np.sum(decomposer.last_response_poisson_fit)) == pytest.approx(
        float(np.sum(spectrum)),
        rel=1e-4,
    )
    diagnostics = decomposer.last_response_poisson_diagnostics
    assert diagnostics["status"] == "ok"
    assert diagnostics["fit_isotopes"] == isotopes
    assert diagnostics["observed_total_counts"] == pytest.approx(float(np.sum(spectrum)))
    assert diagnostics["fitted_total_counts"] == pytest.approx(
        float(np.sum(decomposer.last_response_poisson_fit)),
    )
    assert diagnostics["design_condition_number"] >= 1.0
    assert "Cs-137" in diagnostics["snr"]


def test_response_poisson_selects_line_basis_for_shield_changed_line_ratio() -> None:
    """Line-resolved regression should handle shield-induced line-ratio changes."""
    decomposer = SpectralDecomposer(
        SpectrumConfig(
            dead_time_tau_s=0.0,
            response_poisson_line_resolved_fit=True,
            response_poisson_photopeak_fusion=False,
        )
    )
    line_matrix, _, line_columns = decomposer._get_response_poisson_line_basis()
    co_indices = [
        idx
        for idx, column in enumerate(line_columns)
        if column.isotope == "Co-60"
    ]
    assert len(co_indices) == 2
    line_counts = np.array([1000.0, 500.0], dtype=float)
    weights = np.array(
        [line_columns[idx].line_weight for idx in co_indices],
        dtype=float,
    )
    weights /= float(np.sum(weights))
    spectrum = np.zeros_like(decomposer.energy_axis, dtype=float)
    for coefficient, column_index in zip(line_counts, co_indices):
        spectrum += coefficient * line_matrix[:, column_index]

    estimates = decomposer.compute_response_poisson_estimates(
        spectrum,
        isotopes=["Co-60"],
        include_background=False,
    )

    expected_count = float(np.dot(weights, line_counts))
    assert estimates["Co-60"].counts == pytest.approx(expected_count, rel=1e-4)
    diagnostics = decomposer.last_response_poisson_diagnostics
    assert diagnostics["line_resolved_fit"] is True
    assert diagnostics["line_model_selection"]["selected"] is True
    assert diagnostics["line_model_selection"]["reason"] == "line_bic_selected"


def test_response_poisson_incident_gamma_line_basis_preserves_count_units() -> None:
    """Incident-gamma line fits should return weighted source-equivalent counts."""
    decomposer = SpectralDecomposer(
        SpectrumConfig(
            dead_time_tau_s=0.0,
            response_efficiency_model="unit",
            use_incident_gamma_response_matrix=True,
            normalize_line_intensities=True,
            response_poisson_line_resolved_fit=True,
            response_poisson_photopeak_fusion=False,
        )
    )
    line_matrix, _, line_columns = decomposer._get_response_poisson_line_basis()
    co_indices = [
        idx
        for idx, column in enumerate(line_columns)
        if column.isotope == "Co-60"
    ]
    assert len(co_indices) == 2
    line_counts = np.array([1000.0, 500.0], dtype=float)
    weights = np.array(
        [line_columns[idx].line_weight for idx in co_indices],
        dtype=float,
    )
    weights /= float(np.sum(weights))
    spectrum = np.zeros_like(decomposer.energy_axis, dtype=float)
    for coefficient, column_index in zip(line_counts, co_indices):
        spectrum += coefficient * line_matrix[:, column_index]

    estimates = decomposer.compute_response_poisson_estimates(
        spectrum,
        isotopes=["Co-60"],
        include_background=False,
    )

    assert estimates["Co-60"].counts == pytest.approx(
        float(np.dot(weights, line_counts)),
        rel=1e-4,
    )
    diagnostics = decomposer.last_response_poisson_diagnostics
    assert diagnostics["line_resolved_fit"] is True
    assert diagnostics["line_model_selection"]["selected"] is True
    assert diagnostics["line_model_selection"]["reason"] == "line_bic_selected"


def test_response_poisson_anchors_known_geant4_background_rate() -> None:
    """Known Geant4 background cps should prevent free background overfitting."""
    config = SpectrumConfig(
        dead_time_tau_s=0.0,
        response_efficiency_model="unit",
        use_incident_gamma_response_matrix=True,
        normalize_line_intensities=True,
        response_poisson_background_rate_cps=12.0,
        response_poisson_photopeak_fusion=False,
    )
    decomposer = SpectralDecomposer(config)
    live_time_s = 30.0
    background_counts = 12.0 * live_time_s
    isotope = "Co-60"
    truth_count = 4200.0
    index = decomposer.isotope_names.index(isotope)
    spectrum = (
        truth_count * decomposer._count_response_matrix()[:, index]
        + background_counts * decomposer._background_shape
    )

    estimates = decomposer.compute_response_poisson_estimates(
        spectrum,
        isotopes=[isotope],
        include_background=True,
        live_time_s=live_time_s,
    )

    assert estimates[isotope].counts == pytest.approx(truth_count, rel=1e-3)
    diagnostics = decomposer.last_response_poisson_diagnostics
    assert diagnostics["background_anchor"]["target_counts"] == pytest.approx(
        background_counts
    )
    assert diagnostics["background_total_counts"] == pytest.approx(
        background_counts,
        rel=0.03,
    )


def test_response_poisson_recovers_weak_peaks_under_co_dominant_continuum() -> None:
    """Quadratic photopeak anchors should preserve weak nuclides under strong Co-60."""
    config = SpectrumConfig(
        dead_time_tau_s=0.0,
        response_efficiency_model="unit",
        use_incident_gamma_response_matrix=True,
        normalize_line_intensities=True,
    )
    decomposer = SpectralDecomposer(config)
    isotopes = ["Cs-137", "Co-60", "Eu-154"]
    truth = {"Cs-137": 10000.0, "Co-60": 17_800_000.0, "Eu-154": 67_000.0}
    response = decomposer._count_response_matrix()
    spectrum = np.zeros_like(decomposer.energy_axis, dtype=float)
    for isotope in isotopes:
        index = decomposer.isotope_names.index(isotope)
        spectrum += truth[isotope] * response[:, index]

    estimates = decomposer.compute_response_poisson_counts(
        spectrum,
        isotopes=isotopes,
        include_background=True,
        live_time_s=30.0,
    )

    assert estimates["Cs-137"] == pytest.approx(truth["Cs-137"], rel=0.08)
    assert estimates["Eu-154"] == pytest.approx(truth["Eu-154"], rel=0.08)
    assert estimates["Co-60"] == pytest.approx(truth["Co-60"], rel=0.02)


def test_incident_gamma_response_folding_adds_continuum_and_preserves_counts() -> None:
    """Incident-gamma folding should make detector-like spectra without changing total entries."""
    config = SpectrumConfig(
        dead_time_tau_s=0.0,
        response_efficiency_model="unit",
        response_continuum_to_peak=2.0,
        response_backscatter_fraction=0.03,
    )
    decomposer = SpectralDecomposer(config)
    spectrum = np.zeros_like(decomposer.energy_axis, dtype=float)
    cs_index = int(np.argmin(np.abs(decomposer.energy_axis - 662.0)))
    spectrum[cs_index] = 1000.0

    folded = decomposer.fold_incident_gamma_spectrum(spectrum)
    folded_variance = decomposer.fold_incident_gamma_spectrum_variance(spectrum)
    low_energy_mask = (decomposer.energy_axis > 50.0) & (decomposer.energy_axis < 450.0)

    assert float(np.sum(folded)) == pytest.approx(1000.0, rel=1e-6)
    assert float(np.sum(folded[low_energy_mask])) > 100.0
    assert float(np.max(folded)) < 1000.0
    assert float(np.sum(folded_variance)) > 0.0
    assert float(np.sum(folded_variance)) < 1000.0


def test_incident_gamma_response_folding_preserves_binned_spectrum_counts() -> None:
    """Incident-gamma response columns should conserve arbitrary weighted bin counts."""
    config = SpectrumConfig(
        dead_time_tau_s=0.0,
        response_efficiency_model="unit",
        response_continuum_to_peak=2.0,
        response_backscatter_fraction=0.03,
    )
    decomposer = SpectralDecomposer(config)
    operator = build_incident_gamma_response_matrix(
        decomposer.energy_axis,
        decomposer.resolution_fn,
        decomposer.efficiency_fn,
        decomposer.config.bin_width_keV,
        continuum_to_peak=decomposer.config.response_continuum_to_peak,
        backscatter_fraction=decomposer.config.response_backscatter_fraction,
    )
    active = decomposer.energy_axis > 0.0
    assert np.allclose(np.sum(operator[:, active], axis=0), 1.0, rtol=0.0, atol=1.0e-10)

    spectrum = np.zeros_like(decomposer.energy_axis, dtype=float)
    for energy_kev, count in (
        (88.0, 17.0),
        (212.0, 43.0),
        (662.0, 1000.0),
        (1174.0, 2500.0),
        (1332.0, 3500.0),
    ):
        index = int(np.argmin(np.abs(decomposer.energy_axis - energy_kev)))
        spectrum[index] += count

    folded = decomposer.fold_incident_gamma_spectrum(spectrum)

    assert float(np.sum(folded)) == pytest.approx(float(np.sum(spectrum)), rel=1.0e-10)
    assert float(np.sum(folded[decomposer.energy_axis < 500.0])) > 0.0


def test_photopeak_combiner_downweights_low_snr_zero_outlier() -> None:
    """Robust photopeak combination should not let weak zero ROIs dominate."""
    decomposer = SpectralDecomposer(SpectrumConfig(dead_time_tau_s=0.0))
    estimates = [
        PhotopeakRoiEstimate(
            isotope="Co-60",
            counts=1000.0,
            variance=25.0,
            roi_min_keV=1160.0,
            roi_max_keV=1185.0,
            reduced_chi2=1.0,
            signal_to_noise=20.0,
        ),
        PhotopeakRoiEstimate(
            isotope="Co-60",
            counts=0.0,
            variance=1.0,
            roi_min_keV=1320.0,
            roi_max_keV=1345.0,
            reduced_chi2=1.0,
            signal_to_noise=0.1,
        ),
    ]

    combined = decomposer._combine_photopeak_estimates(estimates)

    assert combined == pytest.approx(1000.0, rel=1e-6)


def test_photopeak_combiner_limits_high_roi_outlier() -> None:
    """Robust photopeak combination should reduce isolated high ROI leverage."""
    decomposer = SpectralDecomposer(SpectrumConfig(dead_time_tau_s=0.0))
    estimates = [
        PhotopeakRoiEstimate(
            isotope="Eu-154",
            counts=1000.0,
            variance=25.0,
            roi_min_keV=120.0,
            roi_max_keV=150.0,
            reduced_chi2=1.0,
            signal_to_noise=20.0,
        ),
        PhotopeakRoiEstimate(
            isotope="Eu-154",
            counts=1050.0,
            variance=25.0,
            roi_min_keV=990.0,
            roi_max_keV=1020.0,
            reduced_chi2=1.0,
            signal_to_noise=20.0,
        ),
        PhotopeakRoiEstimate(
            isotope="Eu-154",
            counts=8000.0,
            variance=25.0,
            roi_min_keV=1260.0,
            roi_max_keV=1290.0,
            reduced_chi2=1.0,
            signal_to_noise=20.0,
        ),
    ]

    combined = decomposer._combine_photopeak_estimates(estimates)

    assert combined < 1500.0


def test_photopeak_combiner_suppresses_unsupported_mixed_roi() -> None:
    """Mixed-isotope ROI evidence should need independent nuclide support."""
    decomposer = SpectralDecomposer(SpectrumConfig(dead_time_tau_s=0.0))
    estimates = [
        PhotopeakRoiEstimate(
            isotope="Eu-154",
            counts=0.0,
            variance=1000.0,
            roi_min_keV=820.0,
            roi_max_keV=926.0,
            reduced_chi2=1.0,
            signal_to_noise=0.5,
            mixed_isotope_roi=False,
        ),
        PhotopeakRoiEstimate(
            isotope="Eu-154",
            counts=1000.0,
            variance=100.0,
            roi_min_keV=1110.0,
            roi_max_keV=1399.0,
            reduced_chi2=1.0,
            signal_to_noise=10.0,
            mixed_isotope_roi=True,
        ),
    ]

    combined = decomposer._combine_photopeak_estimates(estimates)

    assert combined == pytest.approx(0.0)


def test_photopeak_combiner_keeps_supported_mixed_roi() -> None:
    """Mixed-isotope ROI evidence should be retained when independent lines agree."""
    decomposer = SpectralDecomposer(SpectrumConfig(dead_time_tau_s=0.0))
    estimates = [
        PhotopeakRoiEstimate(
            isotope="Eu-154",
            counts=800.0,
            variance=100.0,
            roi_min_keV=820.0,
            roi_max_keV=926.0,
            reduced_chi2=1.0,
            signal_to_noise=8.0,
            mixed_isotope_roi=False,
        ),
        PhotopeakRoiEstimate(
            isotope="Eu-154",
            counts=1000.0,
            variance=100.0,
            roi_min_keV=1110.0,
            roi_max_keV=1399.0,
            reduced_chi2=1.0,
            signal_to_noise=10.0,
            mixed_isotope_roi=True,
        ),
    ]

    combined = decomposer._combine_photopeak_estimates(estimates)

    assert combined > 800.0


def test_photopeak_combiner_downweights_inconsistent_mixed_roi() -> None:
    """Mixed ROI evidence should not dominate conflicting independent lines."""
    decomposer = SpectralDecomposer(SpectrumConfig(dead_time_tau_s=0.0))
    estimates = [
        PhotopeakRoiEstimate(
            isotope="Eu-154",
            counts=900.0,
            variance=100.0,
            roi_min_keV=820.0,
            roi_max_keV=926.0,
            reduced_chi2=1.0,
            signal_to_noise=20.0,
            mixed_isotope_roi=False,
        ),
        PhotopeakRoiEstimate(
            isotope="Eu-154",
            counts=1100.0,
            variance=100.0,
            roi_min_keV=939.0,
            roi_max_keV=1053.0,
            reduced_chi2=1.0,
            signal_to_noise=20.0,
            mixed_isotope_roi=False,
        ),
        PhotopeakRoiEstimate(
            isotope="Eu-154",
            counts=5000.0,
            variance=25.0,
            roi_min_keV=1110.0,
            roi_max_keV=1399.0,
            reduced_chi2=1.0,
            signal_to_noise=80.0,
            mixed_isotope_roi=True,
        ),
    ]

    combined = decomposer._combine_photopeak_estimates(estimates)

    assert combined < 1500.0


def test_response_model_counts_match_shield_theory_for_mixed_python_spectrum() -> None:
    """Full-response NNLS should recover theory for mixed shielded Python spectra."""
    import spectrum.pipeline as pipeline

    background_backup = (pipeline.BACKGROUND_RATE_CPS, pipeline.BACKGROUND_COUNTS_PER_SECOND)
    pipeline.BACKGROUND_RATE_CPS = 0.0
    pipeline.BACKGROUND_COUNTS_PER_SECOND = 0.0
    try:
        isotopes = ["Co-60", "Cs-137", "Eu-154"]
        sources = [
            PointSource("Cs-137", position=(5.0, 5.0, 4.5), intensity_cps_1m=50000.0),
            PointSource("Co-60", position=(6.0, 6.0, 5.5), intensity_cps_1m=30000.0),
            PointSource("Eu-154", position=(7.0, 7.0, 6.5), intensity_cps_1m=30000.0),
        ]
        env = EnvironmentConfig(detector_position=(1.0, 1.0, 0.5))
        dwell_time_s = 30.0
        fe_index = 7
        pb_index = 7
        orientations = generate_octant_orientations()
        mu_by_isotope = mu_by_isotope_from_tvl_mm(HVL_TVL_TABLE_MM, isotopes=isotopes)
        shield_params = ShieldParams()
        kernel = ContinuousKernel(
            mu_by_isotope=mu_by_isotope,
            shield_params=shield_params,
            use_gpu=False,
        )
        decomposer = SpectralDecomposer()

        spectrum, _ = decomposer.simulate_spectrum(
            sources=sources,
            environment=env,
            acquisition_time=dwell_time_s,
            rng=None,
            fe_shield_orientation=orientations[fe_index],
            pb_shield_orientation=orientations[pb_index],
            mu_by_isotope=mu_by_isotope,
            shield_params=shield_params,
        )
        counts = decomposer.compute_response_model_counts(spectrum, isotopes=isotopes)

        detector = env.detector()
        theory = {isotope: 0.0 for isotope in isotopes}
        for source in sources:
            attenuation = kernel.attenuation_factor_pair(
                source.isotope,
                source.position_array(),
                detector,
                fe_index,
                pb_index,
            )
            theory[source.isotope] += (
                dwell_time_s
                * source.intensity_cps_1m
                * geometric_term(detector, source.position_array())
                * attenuation
            )

        for isotope in isotopes:
            assert counts[isotope] == pytest.approx(theory[isotope], rel=1e-10, abs=1e-8)
    finally:
        pipeline.BACKGROUND_RATE_CPS, pipeline.BACKGROUND_COUNTS_PER_SECOND = background_backup


def test_photopeak_nnls_counts_match_shield_theory_for_mixed_python_spectrum() -> None:
    """Photopeak NNLS should recover shielded source-equivalent counts."""
    import spectrum.pipeline as pipeline

    background_backup = (pipeline.BACKGROUND_RATE_CPS, pipeline.BACKGROUND_COUNTS_PER_SECOND)
    pipeline.BACKGROUND_RATE_CPS = 0.0
    pipeline.BACKGROUND_COUNTS_PER_SECOND = 0.0
    try:
        isotopes = ["Co-60", "Cs-137", "Eu-154"]
        sources = [
            PointSource("Cs-137", position=(5.0, 5.0, 4.5), intensity_cps_1m=50000.0),
            PointSource("Co-60", position=(6.0, 6.0, 5.5), intensity_cps_1m=30000.0),
            PointSource("Eu-154", position=(7.0, 7.0, 6.5), intensity_cps_1m=30000.0),
        ]
        env = EnvironmentConfig(detector_position=(1.0, 1.0, 0.5))
        dwell_time_s = 30.0
        fe_index = 7
        pb_index = 7
        orientations = generate_octant_orientations()
        mu_by_isotope = mu_by_isotope_from_tvl_mm(HVL_TVL_TABLE_MM, isotopes=isotopes)
        shield_params = ShieldParams()
        kernel = ContinuousKernel(
            mu_by_isotope=mu_by_isotope,
            shield_params=shield_params,
            use_gpu=False,
        )
        decomposer = SpectralDecomposer(SpectrumConfig(dead_time_tau_s=0.0))

        spectrum, _ = decomposer.simulate_spectrum(
            sources=sources,
            environment=env,
            acquisition_time=dwell_time_s,
            rng=None,
            fe_shield_orientation=orientations[fe_index],
            pb_shield_orientation=orientations[pb_index],
            mu_by_isotope=mu_by_isotope,
            shield_params=shield_params,
        )
        counts = decomposer.compute_photopeak_nnls_counts(
            spectrum,
            live_time_s=dwell_time_s,
            isotopes=isotopes,
        )

        detector = env.detector()
        theory = {isotope: 0.0 for isotope in isotopes}
        for source in sources:
            attenuation = kernel.attenuation_factor_pair(
                source.isotope,
                source.position_array(),
                detector,
                fe_index,
                pb_index,
            )
            theory[source.isotope] += (
                dwell_time_s
                * source.intensity_cps_1m
                * geometric_term(detector, source.position_array())
                * attenuation
            )

        for isotope in isotopes:
            assert counts[isotope] == pytest.approx(theory[isotope], rel=0.02)
    finally:
        pipeline.BACKGROUND_RATE_CPS, pipeline.BACKGROUND_COUNTS_PER_SECOND = background_backup


def test_peak_window_counts_are_not_conservative_for_mixed_spectra() -> None:
    """Peak-window counts should expose cross-talk in mixed shielded spectra."""
    import spectrum.pipeline as pipeline

    background_backup = (pipeline.BACKGROUND_RATE_CPS, pipeline.BACKGROUND_COUNTS_PER_SECOND)
    pipeline.BACKGROUND_RATE_CPS = 0.0
    pipeline.BACKGROUND_COUNTS_PER_SECOND = 0.0
    try:
        isotopes = ["Co-60", "Cs-137", "Eu-154"]
        sources = [
            PointSource("Cs-137", position=(5.0, 5.0, 4.5), intensity_cps_1m=50000.0),
            PointSource("Co-60", position=(6.0, 6.0, 5.5), intensity_cps_1m=60000.0),
            PointSource("Eu-154", position=(7.0, 7.0, 6.5), intensity_cps_1m=30000.0),
        ]
        env = EnvironmentConfig(detector_position=(1.0, 1.0, 0.5))
        orientations = generate_octant_orientations()
        mu_by_isotope = mu_by_isotope_from_tvl_mm(HVL_TVL_TABLE_MM, isotopes=isotopes)
        shield_params = ShieldParams()
        decomposer = SpectralDecomposer()

        free_spectrum, _ = decomposer.simulate_spectrum(
            sources=sources,
            environment=env,
            acquisition_time=30.0,
            rng=None,
            fe_shield_orientation=orientations[0],
            pb_shield_orientation=orientations[0],
            mu_by_isotope=mu_by_isotope,
            shield_params=shield_params,
        )
        blocked_spectrum, _ = decomposer.simulate_spectrum(
            sources=sources,
            environment=env,
            acquisition_time=30.0,
            rng=None,
            fe_shield_orientation=orientations[7],
            pb_shield_orientation=orientations[7],
            mu_by_isotope=mu_by_isotope,
            shield_params=shield_params,
        )
        free_peak = decomposer.compute_isotope_counts_thesis(
            free_spectrum,
            live_time_s=30.0,
            isotopes=isotopes,
        )
        blocked_peak = decomposer.compute_isotope_counts_thesis(
            blocked_spectrum,
            live_time_s=30.0,
            isotopes=isotopes,
        )
        free_response = decomposer.compute_response_model_counts(free_spectrum, isotopes=isotopes)
        blocked_response = decomposer.compute_response_model_counts(blocked_spectrum, isotopes=isotopes)

        peak_ratio = blocked_peak["Eu-154"] / free_peak["Eu-154"]
        response_ratio = blocked_response["Eu-154"] / free_response["Eu-154"]

        assert peak_ratio > response_ratio * 1.2
    finally:
        pipeline.BACKGROUND_RATE_CPS, pipeline.BACKGROUND_COUNTS_PER_SECOND = background_backup
