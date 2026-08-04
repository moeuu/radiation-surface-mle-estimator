"""Tests for covariance-aware extracted-count diagnostic fitting."""

from __future__ import annotations

import numpy as np
import pytest

from three_d_estimation.count_covariance import fit_surface_map_count_covariance
from three_d_estimation.solver import SurfaceMapConfig


def test_multivariate_student_t_recovers_density_with_cross_channel_covariance() -> (
    None
):
    """Full covariance must enter the fitted count-domain diagnostic objective."""
    response = np.asarray(
        [
            [[1.0, 0.1], [0.2, 0.0]],
            [[0.7, 0.2], [0.1, 0.3]],
            [[0.2, 0.8], [0.4, 0.1]],
            [[0.1, 0.3], [0.8, 0.7]],
        ]
    )
    # Each isotope channel responds only to its matching isotope source.
    truth = np.asarray([[20.0, 4.0], [3.0, 12.0]])
    observed = np.einsum("mgi,gi->mi", response, truth)
    covariance = np.repeat(
        np.asarray([[[4.0, 1.0], [1.0, 3.0]]]),
        observed.shape[0],
        axis=0,
    )

    result, diagnostics = fit_surface_map_count_covariance(
        observed,
        response,
        covariance,
        np.ones(2),
        likelihood_family="multivariate_student_t",
        config=SurfaceMapConfig(max_iterations=2000, tolerance=1.0e-9),
    )

    np.testing.assert_allclose(
        result.densities_cps_1m_m2,
        truth,
        rtol=2.0e-4,
        atol=2.0e-4,
    )
    assert diagnostics.likelihood_family == "multivariate_student_t"
    assert max(diagnostics.condition_numbers) < 3.0


def test_singular_covariance_fails_closed_without_regularization() -> None:
    """A singular covariance cannot silently become an independent Poisson fit."""
    observed = np.asarray([[10.0, 5.0], [8.0, 6.0]])
    response = np.ones((2, 1, 2), dtype=float)
    singular = np.repeat(
        np.asarray([[[1.0, 1.0], [1.0, 1.0]]]),
        2,
        axis=0,
    )

    with pytest.raises(ValueError, match="singular"):
        fit_surface_map_count_covariance(
            observed,
            response,
            singular,
            np.ones(1),
            covariance_regularization=0.0,
        )

    regularized, diagnostics = fit_surface_map_count_covariance(
        observed,
        response,
        singular,
        np.ones(1),
        covariance_regularization=1.0e-3,
    )
    assert np.all(np.isfinite(regularized.densities_cps_1m_m2))
    assert diagnostics.covariance_regularization == pytest.approx(1.0e-3)
