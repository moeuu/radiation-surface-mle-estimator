"""Tests for durable pre-estimator logging in the normal local runtime."""

from __future__ import annotations

from pathlib import Path

import pytest

from measurement.model import EnvironmentConfig, PointSource
from measurement.obstacles import ObstacleGrid
from realtime_demo import (
    _build_measurement_run_context,
    _complete_count_covariance,
)
from runtime.measurement_log import MeasurementLogRecorder, load_measurement_log
from runtime.records import MeasurementRecord, RunContext
from sim.protocol import SimulationObservation


def _record() -> MeasurementRecord:
    """Return one complete finalized observation suitable for staging."""
    observation = SimulationObservation(
        step_id=7,
        detector_pose_xyz=(1.0, 2.0, 0.5),
        detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        fe_orientation_index=2,
        pb_orientation_index=6,
        spectrum_counts=[2.0, 3.0],
        energy_bin_edges_keV=[0.0, 100.0, 200.0],
        metadata={"backend": "analytic"},
    )
    return MeasurementRecord.from_simulation_observation(
        observation,
        station_id=1,
        live_time_s=4.0,
        travel_time_s=0.5,
        shield_actuation_time_s=0.25,
        spectrum_variance=[2.0, 3.0],
        counts_by_isotope={"Cs-137": 4.5, "Co-60": 0.5},
        count_covariance_by_isotope={
            "Cs-137": {"Cs-137": 2.0, "Co-60": -0.1},
            "Co-60": {"Cs-137": -0.1, "Co-60": 1.0},
        },
    )


def _context() -> RunContext:
    """Build a portable run context through the production helper."""
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(2, 2),
        blocked_cells=((1, 1),),
    )
    return _build_measurement_run_context(
        runtime_config={"source_rate_model": "detector_cps_1m"},
        environment=EnvironmentConfig(
            size_x=2.0,
            size_y=2.0,
            size_z=3.0,
            detector_position=(0.5, 0.5, 0.5),
        ),
        obstacle_grid=grid,
        isotopes=("Cs-137", "Co-60"),
        sim_backend="analytic",
        spectrum_count_method="response_poisson",
        environment_mode="fixed",
        obstacle_layout_path=None,
        output_tag="test",
        measurement_time_s=4.0,
        adaptive_dwell=False,
    )


def test_recorder_fsyncs_a_shard_before_final_publication(tmp_path: Path) -> None:
    """A record is durable in recovery staging before the estimator can run."""
    target = tmp_path / "run"
    recorder = MeasurementLogRecorder(target, _context())

    recorder.append(_record())

    assert not target.exists()
    shards = tuple((recorder.staging_dir / "records").iterdir())
    assert len(shards) == 1
    assert (shards[0] / "observation.npz").is_file()
    assert (shards[0] / "metadata.json").is_file()

    assert recorder.finalize() == target
    assert not recorder.staging_dir.exists()
    restored = load_measurement_log(target)
    assert restored.context.environment["obstacle_grid"]["blocked_cells"] == [[1, 1]]
    assert len(restored.records) == 1
    assert restored.records[0].step_id == 7
    assert restored.records[0].counts_by_isotope == {
        "Cs-137": 4.5,
        "Co-60": 0.5,
    }


def test_complete_covariance_fills_diagonal_and_mirrors_triangle() -> None:
    """Replay receives a complete matrix even when metadata stores one triangle."""
    completed = _complete_count_covariance(
        isotopes=("Cs-137", "Co-60"),
        variances={"Cs-137": 3.0, "Co-60": 2.0},
        covariance={"Cs-137": {"Co-60": -0.25}},
    )
    assert completed == {
        "Cs-137": {"Cs-137": 3.0, "Co-60": -0.25},
        "Co-60": {"Cs-137": -0.25, "Co-60": 2.0},
    }


def test_complete_covariance_rejects_conflicting_symmetric_entries() -> None:
    """Conflicting simulator covariance metadata fails before PF mutation."""
    with pytest.raises(ValueError, match="not symmetric"):
        _complete_count_covariance(
            isotopes=("Cs-137", "Co-60"),
            variances={"Cs-137": 3.0, "Co-60": 2.0},
            covariance={
                "Cs-137": {"Co-60": -0.25},
                "Co-60": {"Cs-137": 0.5},
            },
        )
