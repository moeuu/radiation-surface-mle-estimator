"""MLE-local orchestration over finalized shared observation records."""

from __future__ import annotations

from enum import Enum
from typing import Callable

from three_d_estimation.backend_contracts import (
    EstimatorBackend,
    EstimatorResult,
    EstimatorSnapshot,
    StationCompleteEstimatorBackend,
)
from runtime.records import MeasurementRecord, RunContext


MeasurementRecordWriter = Callable[[MeasurementRecord], None]


class SessionState(str, Enum):
    """Observable lifecycle state for :class:`LiveEstimationSession`."""

    RUNNING = "running"
    FAILED = "failed"
    FINALIZED = "finalized"


class LiveEstimationSession:
    """Feed one persisted measurement history to an estimator backend.

    Every accepted record is first added to the in-memory immutable history,
    then passed to the optional writer, and only then delivered to
    ``backend.update``.  Consequently, an estimator exception cannot erase the
    exact input that caused it.
    """

    def __init__(
        self,
        *,
        context: RunContext,
        backend: EstimatorBackend,
        record_writer: MeasurementRecordWriter | None = None,
    ) -> None:
        """Initialize the backend and an empty pre-update record history."""
        if not isinstance(context, RunContext):
            raise TypeError("context must be a RunContext.")
        if not isinstance(backend, EstimatorBackend):
            raise TypeError("backend must implement EstimatorBackend.")
        if record_writer is not None and not callable(record_writer):
            raise TypeError("record_writer must be callable or None.")

        self._context = context
        self._backend = backend
        self._record_writer = record_writer
        self._records: list[MeasurementRecord] = []
        self._step_ids: set[int] = set()
        self._station_start_index = 0
        self._station_snapshots: list[EstimatorSnapshot] = []
        self._result: EstimatorResult | None = None
        self._state = SessionState.RUNNING

        try:
            self._backend.initialize(self._context)
        except BaseException:
            self._state = SessionState.FAILED
            raise

    @property
    def context(self) -> RunContext:
        """Return the immutable run context supplied at initialization."""
        return self._context

    @property
    def backend(self) -> EstimatorBackend:
        """Return the estimator backend owned by this session."""
        return self._backend

    @property
    def state(self) -> SessionState:
        """Return the current lifecycle state."""
        return self._state

    @property
    def records(self) -> tuple[MeasurementRecord, ...]:
        """Return an immutable view of all accepted records in acquisition order."""
        return tuple(self._records)

    @property
    def station_snapshots(self) -> tuple[EstimatorSnapshot, ...]:
        """Return completed-station snapshots in acquisition order."""
        return tuple(self._station_snapshots)

    def _ensure_running(self) -> None:
        """Raise unless new work is valid in the current lifecycle state."""
        if self._state is SessionState.FAILED:
            raise RuntimeError("LiveEstimationSession is failed and cannot continue.")
        if self._state is SessionState.FINALIZED:
            raise RuntimeError("LiveEstimationSession is already finalized.")

    def receive(
        self,
        measurement: MeasurementRecord,
        *,
        station_complete: bool = False,
    ) -> EstimatorSnapshot | None:
        """Persist/collect, update, and optionally close the current station."""
        self._ensure_running()
        if not isinstance(measurement, MeasurementRecord):
            raise TypeError("measurement must be a finalized MeasurementRecord.")
        if measurement.step_id in self._step_ids:
            raise ValueError(f"Duplicate finalized step_id {measurement.step_id}.")

        # Collection and external persistence deliberately precede estimator
        # mutation.  Do not move backend.update above these operations.
        self._records.append(measurement)
        self._step_ids.add(measurement.step_id)
        try:
            if self._record_writer is not None:
                self._record_writer(measurement)
            self._backend.update(measurement)
        except BaseException:
            self._state = SessionState.FAILED
            raise

        if station_complete:
            return self.complete_station(station_id=measurement.station_id)
        return None

    def complete_station(self, *, station_id: int | None = None) -> EstimatorSnapshot:
        """Run an optional warm-update hook and capture a station snapshot."""
        self._ensure_running()
        station_records = tuple(self._records[self._station_start_index :])
        if not station_records:
            raise RuntimeError("No new measurements are available for station completion.")

        actual_station_id = station_records[0].station_id
        if any(record.station_id != actual_station_id for record in station_records):
            raise ValueError(
                "Pending measurements span multiple station_id values; complete each "
                "station before receiving the next one."
            )
        if station_id is not None and station_id != actual_station_id:
            raise ValueError(
                f"station_id {station_id} does not match pending station {actual_station_id}."
            )

        try:
            if isinstance(self._backend, StationCompleteEstimatorBackend):
                self._backend.on_station_complete(actual_station_id, station_records)
            snapshot = self._backend.snapshot()
            if not isinstance(snapshot, EstimatorSnapshot):
                raise TypeError("backend.snapshot() must return EstimatorSnapshot.")
        except BaseException:
            self._state = SessionState.FAILED
            raise

        self._station_start_index = len(self._records)
        self._station_snapshots.append(snapshot)
        return snapshot

    def snapshot(self) -> EstimatorSnapshot:
        """Capture current state without declaring a station complete."""
        self._ensure_running()
        try:
            snapshot = self._backend.snapshot()
            if not isinstance(snapshot, EstimatorSnapshot):
                raise TypeError("backend.snapshot() must return EstimatorSnapshot.")
            return snapshot
        except BaseException:
            self._state = SessionState.FAILED
            raise

    def finalize(self) -> EstimatorResult:
        """Finalize once; repeated calls return the same immutable result."""
        if self._state is SessionState.FINALIZED:
            assert self._result is not None
            return self._result
        self._ensure_running()
        try:
            result = self._backend.finalize()
            if not isinstance(result, EstimatorResult):
                raise TypeError("backend.finalize() must return EstimatorResult.")
        except BaseException:
            self._state = SessionState.FAILED
            raise
        self._result = result
        self._state = SessionState.FINALIZED
        return result

    # A descriptive alias for integrations that call the operation a process.
    process_measurement = receive
