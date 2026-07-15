from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

import numpy as np

from runtime.estimator_backend import (
    EstimatorBackend,
    EstimatorResult,
    EstimatorSnapshot,
    PlannerBeliefProvider,
    SourceMode,
    StationCompleteEstimatorBackend,
    SurfaceMapSnapshot,
)
from runtime.records import MeasurementRecord, RunContext
from runtime.session import LiveEstimationSession, SessionState


def build_context() -> RunContext:
    return RunContext(
        upstream_pf_commit="standalone-snapshot",
        runtime_config={"mode": "analytic"},
        environment={"size_x": 4.0, "size_y": 5.0, "size_z": 3.0},
        sim_backend="analytic",
        spectrum_count_method="response_poisson",
        isotopes=("Cs-137",),
    )


def build_record(step_id: int, station_id: int = 5) -> MeasurementRecord:
    return MeasurementRecord(
        station_id=station_id,
        step_id=step_id,
        detector_pose_xyz=(1.0, 2.0, 1.5),
        detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        fe_orientation_index=2,
        pb_orientation_index=6,
        live_time_s=4.0,
        travel_time_s=0.5,
        shield_actuation_time_s=0.25,
        spectrum_counts=np.array([2.0, 3.0]),
        spectrum_variance=np.array([2.5, 3.5]),
        energy_bin_edges_keV=np.array([0.0, 100.0, 200.0]),
        counts_by_isotope={"Cs-137": 4.5},
        count_covariance_by_isotope={"Cs-137": {"Cs-137": 1.25}},
        metadata={"finalized": True},
    )


def build_snapshot(step_id: int) -> EstimatorSnapshot:
    mode = SourceMode(
        position_xyz=(1.0, 2.0, 0.0),
        strength_cps_1m=4.5,
        covariance_xyz_m2=np.diag([0.1, 0.2, 0.0]),
    )
    surface_map = SurfaceMapSnapshot(
        patch_ids=(10, 11),
        patch_centroids_xyz=np.array([[1.0, 2.0, 0.0], [2.0, 2.0, 0.0]]),
        density_cps_1m_per_m2=np.array([4.0, 0.5]),
    )
    return EstimatorSnapshot(
        step_id=step_id,
        source_modes_by_isotope={"Cs-137": (mode,)},
        surface_map_by_isotope={"Cs-137": surface_map},
        predicted_spectrum=np.array([2.1, 2.9]),
        diagnostics={"converged": True},
    )


class RecordingBackend:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.context: RunContext | None = None
        self.measurements: list[MeasurementRecord] = []
        self.warm_station_ids: list[int] = []

    def initialize(self, context: RunContext) -> None:
        self.context = context
        self.events.append("initialize")

    def update(self, measurement: MeasurementRecord) -> None:
        self.measurements.append(measurement)
        self.events.append(f"update:{measurement.step_id}")

    def on_station_complete(
        self,
        station_id: int,
        measurements: tuple[MeasurementRecord, ...],
    ) -> None:
        self.warm_station_ids.append(station_id)
        self.events.append(f"warm:{station_id}:{len(measurements)}")

    def snapshot(self) -> EstimatorSnapshot:
        step_id = self.measurements[-1].step_id if self.measurements else -1
        self.events.append(f"snapshot:{step_id}")
        return build_snapshot(step_id)

    def finalize(self) -> EstimatorResult:
        self.events.append("finalize")
        step_id = self.measurements[-1].step_id if self.measurements else -1
        return EstimatorResult(
            final_snapshot=build_snapshot(step_id),
            diagnostics={"measurement_count": len(self.measurements)},
        )


class ExamplePlannerBelief:
    def source_modes(self) -> dict[str, tuple[SourceMode, ...]]:
        return build_snapshot(0).source_modes_by_isotope

    def predict_candidate_counts(self, actions: list[object]) -> dict[str, int]:
        return {"action_count": len(actions)}

    def uncertainty_summary(self) -> dict[str, float]:
        return {"entropy": 1.0}

    def model_order_summary(self) -> dict[str, int]:
        return {"source_count": 1}


class EstimatorBackendTests(unittest.TestCase):
    def test_snapshot_value_objects_are_frozen_and_defensively_copy_arrays(self) -> None:
        snapshot = build_snapshot(3)
        result = EstimatorResult(final_snapshot=snapshot, artifacts={"npz": "estimate.npz"})

        self.assertFalse(snapshot.predicted_spectrum.flags.writeable)
        surface = snapshot.surface_map_by_isotope["Cs-137"]
        self.assertFalse(surface.patch_centroids_xyz.flags.writeable)
        self.assertFalse(surface.density_cps_1m_per_m2.flags.writeable)
        mode = snapshot.source_modes_by_isotope["Cs-137"][0]
        self.assertFalse(mode.covariance_xyz_m2.flags.writeable)
        self.assertIs(result.final_snapshot, snapshot)
        with self.assertRaises(FrozenInstanceError):
            snapshot.step_id = 4  # type: ignore[misc]

    def test_estimator_and_planner_protocols_are_independent(self) -> None:
        backend = RecordingBackend([])
        planner = ExamplePlannerBelief()

        self.assertIsInstance(backend, EstimatorBackend)
        self.assertIsInstance(backend, StationCompleteEstimatorBackend)
        self.assertNotIsInstance(backend, PlannerBeliefProvider)
        self.assertIsInstance(planner, PlannerBeliefProvider)
        self.assertNotIsInstance(planner, EstimatorBackend)

    def test_session_records_and_writes_before_update_then_warm_snapshots(self) -> None:
        events: list[str] = []
        backend = RecordingBackend(events)
        session: LiveEstimationSession

        def writer(measurement: MeasurementRecord) -> None:
            self.assertIs(session.records[-1], measurement)
            events.append(f"write:{measurement.step_id}")

        session = LiveEstimationSession(
            context=build_context(),
            backend=backend,
            record_writer=writer,
        )
        self.assertIsNone(session.receive(build_record(1)))
        station_snapshot = session.receive(build_record(2), station_complete=True)

        self.assertEqual(
            events,
            [
                "initialize",
                "write:1",
                "update:1",
                "write:2",
                "update:2",
                "warm:5:2",
                "snapshot:2",
            ],
        )
        self.assertEqual(len(session.records), 2)
        self.assertEqual(session.station_snapshots, (station_snapshot,))
        self.assertEqual(backend.warm_station_ids, [5])

        result = session.finalize()
        self.assertEqual(result.diagnostics["measurement_count"], 2)
        self.assertIs(session.finalize(), result)
        self.assertEqual(events.count("finalize"), 1)
        self.assertIs(session.state, SessionState.FINALIZED)
        with self.assertRaisesRegex(RuntimeError, "already finalized"):
            session.receive(build_record(3))

    def test_writer_failure_preserves_record_and_prevents_backend_update(self) -> None:
        events: list[str] = []
        backend = RecordingBackend(events)

        def failing_writer(measurement: MeasurementRecord) -> None:
            events.append(f"write-failed:{measurement.step_id}")
            raise OSError("disk full")

        session = LiveEstimationSession(
            context=build_context(),
            backend=backend,
            record_writer=failing_writer,
        )
        record = build_record(1)
        with self.assertRaisesRegex(OSError, "disk full"):
            session.receive(record)

        self.assertEqual(session.records, (record,))
        self.assertEqual(backend.measurements, [])
        self.assertIs(session.state, SessionState.FAILED)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            session.finalize()


if __name__ == "__main__":
    unittest.main()
