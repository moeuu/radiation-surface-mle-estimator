"""Tests for MLE-specific Fisher optimal experimental design."""

from __future__ import annotations

import numpy as np
import pytest

from three_d_estimation.cli import _estimate_history_indices, build_argument_parser
from three_d_estimation.information_planner import (
    MLEPlanningConfig,
    _ambiguity_metrics,
    _source_basis,
    select_fisher_action,
)
from three_d_estimation.types import MLEEstimate, ObservationBatch, SurfacePatch


def _orientations() -> np.ndarray:
    """Return two unit directions for four synthetic Fe/Pb pair IDs."""
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )


def _floor_patch(patch_id: int, x_offset: float) -> SurfacePatch:
    """Return one exact unit floor patch for basis tests."""
    return SurfacePatch(
        patch_id=patch_id,
        centroid_xyz=np.asarray([x_offset + 0.5, 0.5, 0.0]),
        normal_xyz=np.asarray([0.0, 0.0, 1.0]),
        area_m2=1.0,
        surface_kind="floor",
        object_id=f"floor:{patch_id}",
        vertices_xyz=np.asarray(
            [
                [x_offset, 0.0, 0.0],
                [x_offset + 1.0, 0.0, 0.0],
                [x_offset + 1.0, 1.0, 0.0],
                [x_offset, 1.0, 0.0],
            ]
        ),
        quadrature_points_xyz=np.asarray([[x_offset + 0.5, 0.5, 0.0]]),
        quadrature_weights=np.asarray([1.0]),
    )


def _ceiling_patch(patch_id: int, x_offset: float) -> SurfacePatch:
    """Return one exact unit ceiling patch for vertical ambiguity tests."""
    return SurfacePatch(
        patch_id=patch_id,
        centroid_xyz=np.asarray([x_offset + 0.5, 0.5, 2.0]),
        normal_xyz=np.asarray([0.0, 0.0, -1.0]),
        area_m2=1.0,
        surface_kind="ceiling",
        object_id=f"ceiling:{patch_id}",
        vertices_xyz=np.asarray(
            [
                [x_offset, 0.0, 2.0],
                [x_offset, 1.0, 2.0],
                [x_offset + 1.0, 1.0, 2.0],
                [x_offset + 1.0, 0.0, 2.0],
            ]
        ),
        quadrature_points_xyz=np.asarray([[x_offset + 0.5, 0.5, 2.0]]),
        quadrature_weights=np.asarray([1.0]),
    )


def test_d_optimal_pose_balances_source_parameter_information() -> None:
    """D-optimality should prefer a balanced reduction in source uncertainty."""
    information = np.zeros((2, 1, 2, 2), dtype=float)
    information[0, 0] = np.diag([9.0, 0.0])
    information[1, 0] = np.diag([3.0, 3.0])

    selected, ranked = select_fisher_action(
        np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
        (0,),
        information,
        np.ones((2, 1)),
        np.eye(2),
        _orientations(),
        nuisance_count=0,
        config=MLEPlanningConfig(shield_program_length=1),
    )

    assert selected.candidate_index == 1
    assert ranked[0] is selected
    assert selected.information_gain_nats > ranked[1].information_gain_nats


def test_joint_shield_program_chooses_complementary_orientations() -> None:
    """A jointly selected two-view station should span both sensitivities."""
    information = np.zeros((1, 3, 2, 2), dtype=float)
    information[0, 0] = np.diag([9.0, 0.0])
    information[0, 1] = np.diag([8.0, 0.0])
    information[0, 2] = np.diag([0.0, 9.0])

    selected, _ = select_fisher_action(
        np.asarray([[0.0, 0.0, 1.0]]),
        (0, 1, 2),
        information,
        np.ones((1, 3)),
        np.eye(2),
        _orientations(),
        nuisance_count=0,
        config=MLEPlanningConfig(shield_program_length=2),
    )

    assert selected.shield_pair_ids == (0, 2)
    assert selected.fe_orientation_indices == (0, 1)
    assert selected.pb_orientation_indices == (0, 0)
    assert selected.live_time_s_by_view == (10.0, 10.0)
    assert selected.to_dict()["measurement_program"][-1]["station_complete"] is True


def test_joint_program_marginalizes_one_shared_station_rate() -> None:
    """A station block should prefer rate-independent shield contrast."""
    information = np.zeros((1, 3, 1, 1), dtype=float)
    information[0, :, 0, 0] = [100.0, 90.25, 10.0]
    station_cross = np.asarray([[[10.0], [9.5], [0.0]]])
    station_information = np.ones((1, 3), dtype=float)

    selected, _ = select_fisher_action(
        np.asarray([[0.0, 0.0, 1.0]]),
        (0, 1, 2),
        information,
        np.ones((1, 3)),
        np.eye(1),
        _orientations(),
        nuisance_count=0,
        station_rate_cross_information=station_cross,
        station_rate_information=station_information,
        config=MLEPlanningConfig(
            shield_program_length=2,
            future_station_rate_prior_precision=0.01,
        ),
    )

    assert selected.shield_pair_ids == (0, 2)


def test_joint_program_uses_pair_specific_ambiguity_utility() -> None:
    """Shield choice should include ambiguity utility during optimization."""
    information = np.ones((1, 3, 1, 1), dtype=float)

    selected, _ = select_fisher_action(
        np.asarray([[0.0, 0.0, 1.0]]),
        (0, 1, 2),
        information,
        np.ones((1, 3)),
        np.eye(1),
        _orientations(),
        nuisance_count=0,
        pair_utility_bonus=np.asarray([[0.0, 0.0, 2.0]]),
        config=MLEPlanningConfig(shield_program_length=1),
    )

    assert selected.shield_pair_ids == (2,)


def test_zero_mle_regions_remain_in_the_exploration_basis() -> None:
    """A sparse MLE must not permanently remove zero-strength surface regions."""
    patches = (_floor_patch(0, 0.0), _floor_patch(1, 1.0))
    estimate = MLEEstimate(
        isotope_names=("Cs-137",),
        patches=patches,
        density_by_isotope=np.asarray([[5.0, 0.0]]),
        patch_strength_by_isotope=np.asarray([[5.0, 0.0]]),
        predicted_spectra=None,
        predicted_isotope_counts=None,
        background_parameters=np.zeros(0),
        nuisance_parameters=np.zeros(0),
        objective_value=1.0,
        poisson_deviance=0.0,
        iterations=1,
        converged=True,
        diagnostics={},
    )

    basis, labels = _source_basis(
        estimate,
        MLEPlanningConfig(
            max_active_source_parameters=1,
            max_total_source_parameters=2,
        ),
    )

    assert basis.shape == (2, 1, 2)
    assert np.all(np.sum(basis, axis=2) > 0.0)
    assert {label["kind"] for label in labels} == {
        "active_patch",
        "residual_exploration",
    }


def test_d_s_optimality_marginalizes_nuisance_confounding() -> None:
    """A source-specific action should beat a stronger but confounded action."""
    information = np.zeros((1, 2, 2, 2), dtype=float)
    confounded = np.asarray([10.0, 10.0])
    source_specific = np.asarray([5.0, 0.0])
    information[0, 0] = np.outer(confounded, confounded)
    information[0, 1] = np.outer(source_specific, source_specific)

    selected, _ = select_fisher_action(
        np.asarray([[0.0, 0.0, 1.0]]),
        (0, 1),
        information,
        np.ones((1, 2)),
        np.eye(2),
        _orientations(),
        nuisance_count=1,
        config=MLEPlanningConfig(shield_program_length=1),
    )

    assert selected.shield_pair_ids == (1,)


def test_external_motion_cost_can_change_the_selected_pose() -> None:
    """Runtime-supplied travel cost should enter only through its explicit weight."""
    information = np.zeros((2, 1, 1, 1), dtype=float)
    information[0, 0, 0, 0] = 4.0
    information[1, 0, 0, 0] = 5.0

    selected, _ = select_fisher_action(
        np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
        (0,),
        information,
        np.ones((2, 1)),
        np.eye(1),
        _orientations(),
        nuisance_count=0,
        travel_costs=np.asarray([0.0, 10.0]),
        config=MLEPlanningConfig(
            shield_program_length=1,
            motion_cost_weight=1.0,
        ),
    )

    assert selected.candidate_index == 0


def test_plan_next_cli_requires_runtime_candidates_and_output() -> None:
    """CLI should expose a separate runtime-candidate planning operation."""
    args = build_argument_parser().parse_args(
        [
            "plan-next",
            "--run-dir",
            "/tmp/runtime-log",
            "--estimate",
            "/tmp/mle-report",
            "--mle-config",
            "/tmp/mle.json",
            "--candidates",
            "/tmp/candidates.json",
            "--output",
            "/tmp/action.json",
        ]
    )

    assert args.command == "plan-next"
    assert args.cpu is False
    assert args.gpu is False
    assert args.planning_config is None


def test_planning_history_must_be_an_exact_causal_prefix() -> None:
    """An old station estimate cannot inspect later MeasurementLog records."""

    class _Estimate:
        """Expose only diagnostics required by the CLI lineage check."""

        diagnostics = {"online_lineage": {"covered_step_ids": [0, 1]}}

    indices = _estimate_history_indices(_Estimate(), np.asarray([0, 1, 2]))
    np.testing.assert_array_equal(indices, [0, 1])

    _Estimate.diagnostics = {"online_lineage": {"covered_step_ids": [0, 2]}}
    with pytest.raises(ValueError, match="exact causal prefix"):
        _estimate_history_indices(_Estimate(), np.asarray([0, 1, 2]))


def test_floor_ceiling_competition_rewards_height_discrimination() -> None:
    """A candidate with distinct vertical signatures must receive a larger score."""
    patches = (_floor_patch(0, 0.0), _ceiling_patch(1, 0.0))
    estimate = MLEEstimate(
        isotope_names=("Cs-137",),
        patches=patches,
        density_by_isotope=np.asarray([[1.0, 1.0]]),
        patch_strength_by_isotope=np.asarray([[1.0, 1.0]]),
        predicted_spectra=None,
        predicted_isotope_counts=None,
        background_parameters=np.zeros(0),
        nuisance_parameters=np.zeros(0),
        objective_value=1.0,
        poisson_deviance=1.0,
        iterations=1,
        converged=True,
        diagnostics={},
    )
    history = ObservationBatch(
        detector_positions_xyz=np.asarray([[0.5, 0.5, 0.5]]),
        detector_quaternions_wxyz=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
        fe_indices=np.asarray([0]),
        pb_indices=np.asarray([0]),
        live_times_s=np.asarray([1.0]),
        spectrum_counts=np.ones((1, 2)),
        spectrum_variances=None,
        energy_bin_edges_keV=np.asarray([0.0, 1.0, 2.0]),
        isotope_counts=np.ones((1, 1)),
        isotope_covariances=np.ones((1, 1, 1)),
        station_ids=np.asarray([0]),
        isotope_names=("Cs-137",),
    )
    response = np.zeros((2, 2, 2, 1), dtype=float)
    response[0, :, 0, 0] = [1.0, 0.0]
    response[0, :, 1, 0] = [0.9, 0.1]
    response[1, :, 0, 0] = [1.0, 0.0]
    response[1, :, 1, 0] = [0.0, 1.0]
    source_basis = np.zeros((2, 1, 2), dtype=float)
    source_basis[0, 0, 0] = 1.0
    source_basis[1, 0, 1] = 1.0
    information = np.tile(np.eye(2), (2, 1, 1))

    metrics = _ambiguity_metrics(
        response,
        information,
        np.asarray([[0.5, 0.5, 0.5], [0.5, 0.5, 1.5]]),
        estimate,
        history,
        source_basis,
        (),
    )

    assert metrics["floor_ceiling"][1] > metrics["floor_ceiling"][0]
    assert metrics["correlation"][1] > metrics["correlation"][0]
