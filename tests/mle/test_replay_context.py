"""Tests for standalone replay context and path resolution."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from measurement.continuous_kernels import ContinuousKernel
from measurement.observation_model import RuntimeObservationModel
from runtime.measurement_log import save_measurement_log
from runtime.records import MeasurementRecord, RunContext
from three_d_estimation.config import MLEConfig
from three_d_estimation.replay import prepare_replay, run_replay


def _measurement_record() -> MeasurementRecord:
    """Return one complete finalized count-domain observation."""
    return MeasurementRecord(
        station_id=0,
        step_id=0,
        detector_pose_xyz=(0.5, 0.5, 0.5),
        detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        fe_orientation_index=0,
        pb_orientation_index=1,
        live_time_s=2.0,
        travel_time_s=0.0,
        shield_actuation_time_s=0.1,
        spectrum_counts=np.array([3.0, 2.0]),
        spectrum_variance=np.array([3.5, 2.5]),
        energy_bin_edges_keV=np.array([0.0, 400.0, 800.0]),
        counts_by_isotope={"Cs-137": 4.0},
        count_covariance_by_isotope={"Cs-137": {"Cs-137": 1.5}},
        metadata={"finalized": True},
    )


def _write_log(
    root: Path,
    *,
    environment: dict[str, object] | None = None,
    obstacle_layout_path: str | None = None,
    source_rate_model: str = "detector_cps_1m",
) -> Path:
    """Write a minimal versioned replay log below a temporary root."""
    context = RunContext(
        upstream_pf_commit="standalone-snapshot",
        runtime_config={
            "source_rate_model": source_rate_model,
            "pf_line_resolved_shield_attenuation": False,
        },
        environment=(
            {"size_x": 2.0, "size_y": 3.0, "size_z": 2.0}
            if environment is None
            else environment
        ),
        sim_backend="analytic",
        spectrum_count_method="response_poisson",
        isotopes=("Cs-137",),
        obstacle_layout_path=obstacle_layout_path,
        source_rate_model=source_rate_model,
    )
    return save_measurement_log(root / "run", context, [_measurement_record()])


def _config() -> MLEConfig:
    """Return a small count-domain configuration for replay construction."""
    return MLEConfig(
        mode="count",
        isotope_names=("Cs-137",),
        patch_spacing_m=(1.0, 1.0, 1.0),
        max_iterations=2,
        debias_refit=False,
    )


class ReplayContextTests(unittest.TestCase):
    """Exercise embedded and strictly local replay resource resolution."""

    def test_embedded_environment_builds_batch_model_and_kernel(self) -> None:
        """Embedded obstacle data should need no filesystem layout dependency."""
        environment = {
            "room_size_xyz": [2.0, 3.0, 2.0],
            "obstacle_grid": {
                "origin": [0.0, 0.0],
                "cell_size": 1.0,
                "grid_shape": [2, 3],
                "blocked_cells": [[1, 2]],
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = _write_log(Path(temporary), environment=environment)
            replay = prepare_replay(run_dir, config=_config())

        self.assertEqual(replay.batch.measurement_count, 1)
        self.assertEqual(replay.environment.size_x, 2.0)
        self.assertEqual(replay.environment.size_y, 3.0)
        self.assertEqual(replay.environment.size_z, 2.0)
        self.assertEqual(replay.obstacle_grid.blocked_cells, ((1, 2),))
        self.assertIsNone(replay.resolved_obstacle_path)
        self.assertIsInstance(replay.observation_model, RuntimeObservationModel)
        self.assertIsInstance(replay.kernel, ContinuousKernel)

    def test_run_directory_layout_is_resolved_before_repository_assets(self) -> None:
        """A relative layout persisted beside a run should be replayed in place."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = _write_log(root, obstacle_layout_path="assets/layout.json")
            layout = run_dir / "assets" / "layout.json"
            layout.parent.mkdir(parents=True)
            layout.write_text(
                """{
  "origin": [0.0, 0.0],
  "cell_size": 1.0,
  "grid_shape": [2, 3],
  "blocked_cells": [[0, 1]]
}\n""",
                encoding="utf-8",
            )

            replay = prepare_replay(run_dir, config=_config())

            self.assertEqual(replay.resolved_obstacle_path, layout.resolve())
            self.assertEqual(replay.obstacle_grid.blocked_cells, ((0, 1),))

    def test_repository_obstacle_layout_is_resolved_without_a_sibling(self) -> None:
        """Committed obstacle_layouts assets should be valid standalone inputs."""
        with tempfile.TemporaryDirectory() as temporary:
            environment = {"size_x": 10.0, "size_y": 20.0, "size_z": 10.0}
            run_dir = _write_log(
                Path(temporary),
                environment=environment,
                obstacle_layout_path="obstacle_layouts/no_obstacles.json",
            )
            replay = prepare_replay(run_dir, config=_config())

        self.assertEqual(replay.obstacle_grid.blocked_cells, ())
        self.assertEqual(replay.resolved_obstacle_path.name, "no_obstacles.json")
        self.assertEqual(replay.resolved_obstacle_path.parent.name, "obstacle_layouts")

    def test_absolute_and_parent_traversal_layout_paths_are_rejected(self) -> None:
        """Persisted layouts must never escape the run or local asset root."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            absolute_run = _write_log(
                root / "absolute",
                obstacle_layout_path=str(root / "outside.json"),
            )
            with self.assertRaisesRegex(ValueError, "absolute paths are forbidden"):
                prepare_replay(absolute_run, config=_config())

            traversal_run = _write_log(
                root / "traversal",
                obstacle_layout_path="../outside.json",
            )
            with self.assertRaisesRegex(ValueError, "parent-directory traversal"):
                prepare_replay(traversal_run, config=_config())

    def test_wrong_source_rate_semantics_are_rejected_before_fit(self) -> None:
        """Replay must never reinterpret gamma-rate input as detector cps at 1 m."""
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = _write_log(
                Path(temporary),
                source_rate_model="gamma_per_second",
            )
            with self.assertRaisesRegex(ValueError, "source_rate_model"):
                prepare_replay(run_dir, config=_config())

    def test_run_replay_invokes_estimator_and_optional_save_hook(self) -> None:
        """run_replay should fit resolved objects and pass results to a save hook."""
        estimate = object()
        saved: list[tuple[object, Path]] = []

        class FakeEstimator:
            """Capture the resolved fit inputs without running numerical optimization."""

            instances: list["FakeEstimator"] = []

            def __init__(self, config: MLEConfig) -> None:
                """Store the supplied replay configuration."""
                self.config = config
                self.fit_args: tuple[object, ...] | None = None
                self.fit_kwargs: dict[str, object] | None = None
                self.instances.append(self)

            def fit(self, *args: object, **kwargs: object) -> object:
                """Record fit inputs and return the sentinel estimate."""
                self.fit_args = args
                self.fit_kwargs = kwargs
                return estimate

        def save_hook(value: object, output_dir: Path) -> str:
            """Capture an optional reporting call."""
            saved.append((value, output_dir))
            return "saved"

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = _write_log(root)
            output_dir = root / "result"
            with patch("three_d_estimation.replay.SurfaceMLEEstimator", FakeEstimator):
                result = run_replay(
                    run_dir,
                    config=_config(),
                    output_dir=output_dir,
                    save_hook=save_hook,
                )

        instance = FakeEstimator.instances[-1]
        self.assertIs(result.estimate, estimate)
        self.assertIs(instance.fit_args[0], result.context.batch)
        self.assertIs(instance.fit_args[1], result.context.environment)
        self.assertIs(instance.fit_args[2], result.context.kernel)
        self.assertIs(
            instance.fit_kwargs["obstacle_grid"],
            result.context.obstacle_grid,
        )
        self.assertEqual(saved, [(estimate, output_dir)])
        self.assertEqual(result.saved_output, "saved")


if __name__ == "__main__":
    unittest.main()
