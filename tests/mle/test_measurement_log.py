from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import tempfile
import unittest

import numpy as np

from runtime.measurement_log import load_measurement_log, save_measurement_log
from runtime.records import MeasurementRecord, RunContext
from sim.protocol import SimulationObservation


class MeasurementLogTests(unittest.TestCase):
    def build_context(self) -> RunContext:
        return RunContext(
            upstream_pf_commit="ec414e6d828b5213ae94f7adfc2e249e380d601e",
            runtime_config={
                "detector": {"height_m": 1.5},
                "spectrum": {"count_method": "response_poisson"},
            },
            environment={"size_x": 10.0, "size_y": 20.0, "size_z": 10.0},
            sim_backend="analytic",
            spectrum_count_method="response_poisson",
            isotopes=("Cs-137", "Co-60"),
            obstacle_layout_path="obstacle_layouts/holdout.json",
            source_layout_path="source_layouts/two_sources.json",
            truth_sources=(
                {
                    "isotope": "Cs-137",
                    "position_xyz": [1.0, 2.0, 0.0],
                    "intensity_cps_1m": 12.5,
                },
            ),
            metadata={"run_label": "round-trip"},
        )

    def build_record(
        self,
        *,
        step_id: int = 11,
        energy_edges: list[float] | None = None,
        spectrum_variance: list[float] | None = None,
    ) -> MeasurementRecord:
        observation = SimulationObservation(
            step_id=step_id,
            detector_pose_xyz=(2.5, 3.5, 1.5),
            detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            fe_orientation_index=3,
            pb_orientation_index=7,
            spectrum_counts=[4.0, 9.0, 0.0],
            energy_bin_edges_keV=(
                energy_edges if energy_edges is not None else [0.0, 100.0, 200.0, 300.0]
            ),
            metadata={"transport": "analytic", "chunk_count": 3},
        )
        return MeasurementRecord.from_simulation_observation(
            observation,
            station_id=5,
            live_time_s=12.75,
            travel_time_s=2.25,
            shield_actuation_time_s=0.5,
            spectrum_variance=(
                [4.5, 9.5, 0.25]
                if spectrum_variance is None
                else spectrum_variance
            ),
            counts_by_isotope={"Cs-137": 8.25, "Co-60": 1.5},
            count_covariance_by_isotope={
                "Cs-137": {"Cs-137": 2.0, "Co-60": -0.25},
                "Co-60": {"Cs-137": -0.25, "Co-60": 1.0},
            },
            metadata={"processed": True},
        )

    def test_exact_round_trip_preserves_finalized_estimator_input(self) -> None:
        context = self.build_context()
        record = self.build_record()

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run-001"
            saved = save_measurement_log(run_dir, context, [record])

            self.assertEqual(saved, run_dir)
            self.assertEqual(
                {path.name for path in run_dir.iterdir()},
                {
                    "run_manifest.json",
                    "runtime_config.resolved.json",
                    "environment.json",
                    "observations.npz",
                    "observation_metadata.jsonl",
                    "truth_sources.json",
                    "upstream_pf_commit.txt",
                },
            )

            loaded = load_measurement_log(run_dir)

        self.assertEqual(loaded.context, context)
        self.assertEqual(loaded.context.runtime_config_sha256, context.runtime_config_sha256)
        self.assertEqual(len(loaded.records), 1)
        restored = loaded.records[0]
        self.assertEqual(restored.station_id, 5)
        self.assertEqual(restored.step_id, 11)
        self.assertEqual(restored.detector_pose_xyz[2], 1.5)
        self.assertEqual(restored.fe_orientation_index, 3)
        self.assertEqual(restored.pb_orientation_index, 7)
        self.assertEqual(restored.live_time_s, 12.75)
        self.assertEqual(restored.travel_time_s, 2.25)
        self.assertEqual(restored.shield_actuation_time_s, 0.5)
        np.testing.assert_array_equal(restored.spectrum_counts, record.spectrum_counts)
        np.testing.assert_array_equal(restored.spectrum_variance, record.spectrum_variance)
        np.testing.assert_array_equal(
            restored.energy_bin_edges_keV,
            record.energy_bin_edges_keV,
        )
        self.assertEqual(restored.counts_by_isotope, record.counts_by_isotope)
        self.assertEqual(
            restored.count_covariance_by_isotope,
            record.count_covariance_by_isotope,
        )
        self.assertEqual(
            restored.metadata,
            {"transport": "analytic", "chunk_count": 3, "processed": True},
        )
        self.assertFalse(restored.spectrum_counts.flags.writeable)
        with self.assertRaises(FrozenInstanceError):
            restored.station_id = 99  # type: ignore[misc]

    def test_invalid_spectrum_shapes_and_mixed_energy_bins_are_rejected(self) -> None:
        observation = SimulationObservation(
            step_id=0,
            detector_pose_xyz=(0.0, 0.0, 0.5),
            detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            fe_orientation_index=0,
            pb_orientation_index=0,
            spectrum_counts=[1.0, 2.0],
            energy_bin_edges_keV=[0.0, 100.0, 200.0],
        )
        with self.assertRaisesRegex(ValueError, "spectrum_variance must have shape"):
            MeasurementRecord.from_simulation_observation(
                observation,
                station_id=0,
                live_time_s=1.0,
                travel_time_s=0.0,
                shield_actuation_time_s=0.0,
                spectrum_variance=[1.0],
                counts_by_isotope=None,
                count_covariance_by_isotope=None,
            )

        first = self.build_record(step_id=1)
        second = self.build_record(
            step_id=2,
            energy_edges=[0.0, 90.0, 200.0, 300.0],
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "different energy_bin_edges"):
                save_measurement_log(
                    Path(temporary) / "mixed-bins",
                    self.build_context(),
                    [first, second],
                )

    def test_directory_representation_is_byte_deterministic(self) -> None:
        context = self.build_context()
        records = [self.build_record()]
        with tempfile.TemporaryDirectory() as temporary:
            first = save_measurement_log(Path(temporary) / "a", context, records)
            second = save_measurement_log(Path(temporary) / "b", context, records)
            first_files = sorted(path.name for path in first.iterdir())
            second_files = sorted(path.name for path in second.iterdir())
            self.assertEqual(first_files, second_files)
            for name in first_files:
                self.assertEqual(
                    (first / name).read_bytes(),
                    (second / name).read_bytes(),
                    name,
                )

    def test_loader_rejects_runtime_config_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = save_measurement_log(
                Path(temporary) / "corrupt",
                self.build_context(),
                [self.build_record()],
            )
            (run_dir / "runtime_config.resolved.json").write_text(
                '{"tampered":true}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "runtime_config_sha256"):
                load_measurement_log(run_dir)

    def test_loader_rejects_noncanonical_npz_dtypes(self) -> None:
        """Tampered integer and presence-mask arrays must not be coerced."""
        for array_name, replacement_dtype in (
            ("step_id", np.float64),
            ("spectrum_variance_present", np.uint8),
        ):
            with self.subTest(array_name=array_name), tempfile.TemporaryDirectory() as temporary:
                run_dir = save_measurement_log(
                    Path(temporary) / "tampered-dtype",
                    self.build_context(),
                    [self.build_record()],
                )
                archive_path = run_dir / "observations.npz"
                with np.load(archive_path, allow_pickle=False) as loaded:
                    arrays = {
                        name: np.array(loaded[name], copy=True)
                        for name in loaded.files
                    }
                arrays[array_name] = arrays[array_name].astype(replacement_dtype)
                with archive_path.open("wb") as stream:
                    np.savez(stream, **arrays)

                with self.assertRaisesRegex(
                    ValueError,
                    rf"array '{array_name}'.*invalid dtype",
                ):
                    load_measurement_log(run_dir)


if __name__ == "__main__":
    unittest.main()
