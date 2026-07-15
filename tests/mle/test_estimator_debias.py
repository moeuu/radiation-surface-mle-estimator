"""Regression tests for support-restricted surface-MLE debiasing."""

from __future__ import annotations

import numpy as np

from measurement.model import EnvironmentConfig
from three_d_estimation.config import MLEConfig
from three_d_estimation.estimator import _FitState, _debias_state, _full_prediction
from three_d_estimation.solver import SurfaceMapResult
from three_d_estimation.surface_patches import build_surface_patches


def _seed_result(densities: np.ndarray, areas_m2: np.ndarray) -> SurfaceMapResult:
    """Return a minimal regularized-fit result used to seed debiasing."""
    integrated = densities * areas_m2[:, None]
    return SurfaceMapResult(
        densities_cps_1m_m2=densities,
        integrated_strengths_cps_1m=integrated,
        nuisance_coefficients=np.zeros(0, dtype=float),
        expected_counts=np.asarray([[float(np.sum(integrated))]], dtype=float),
        objective=0.0,
        poisson_nll=0.0,
        l1_penalty=0.0,
        tv_penalty=0.0,
        group_penalty=0.0,
        nuisance_penalty=0.0,
        deviance=0.0,
        converged=True,
        iterations=1,
        relative_change=0.0,
        relative_objective_change=0.0,
        kkt_residual=0.0,
        objective_history=(0.0,),
    )


def test_debias_cannot_preserve_initialized_density_outside_selected_support() -> None:
    """A rejected positive warm value must be zero in the debiased map."""
    patches = build_surface_patches(
        EnvironmentConfig(size_x=1.0, size_y=1.0, size_z=1.0),
        None,
        spacing=2.0,
        quadrature_points_per_patch=1,
    )
    densities = np.zeros((patches.patch_count, 1), dtype=float)
    densities[0, 0] = 10.0
    densities[1, 0] = 1.0
    response = np.ones((1, 1, patches.patch_count, 1), dtype=float)
    state = _FitState(
        patches=patches,
        response=response,
        nuisance_response=np.zeros((1, 1, 0), dtype=float),
        nuisance_names=(),
        result=_seed_result(densities, patches.areas_m2),
        fit_indices=np.asarray([0], dtype=np.int64),
        held_out_indices=np.zeros(0, dtype=np.int64),
        spectral_details=None,
    )
    config = MLEConfig(
        mode="count",
        isotope_names=("Cs-137",),
        support_threshold_fraction=0.5,
        max_iterations=100,
        check_interval=1,
        tolerance=1.0e-10,
        objective_tolerance=1.0e-10,
    )

    debiased = _debias_state(state, np.asarray([[10.0]], dtype=float), config)

    assert densities[1, 0] == 1.0
    np.testing.assert_array_equal(
        debiased.result.densities_cps_1m_m2[1:],
        np.zeros((patches.patch_count - 1, 1), dtype=float),
    )
    np.testing.assert_array_equal(
        debiased.result.integrated_strengths_cps_1m[1:],
        np.zeros((patches.patch_count - 1, 1), dtype=float),
    )
    np.testing.assert_array_equal(debiased.response[:, :, 1:, :], 0.0)
    prediction = _full_prediction(debiased)
    np.testing.assert_allclose(
        prediction,
        debiased.result.integrated_strengths_cps_1m[0, 0],
    )
