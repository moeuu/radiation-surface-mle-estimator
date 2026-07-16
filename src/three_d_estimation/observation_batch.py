"""Conversion between versioned measurement logs and validated MLE batches."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from runtime.measurement_log import MeasurementLog
from runtime.records import MeasurementRecord

from .types import ObservationBatch


def _shield_program_block_id(record: MeasurementRecord) -> str:
    """Return an explicit shield-program block ID or the station fallback."""
    for key in (
        "shield_program_block_id",
        "shield_program_id",
        "shield_block_id",
    ):
        value = record.metadata.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (str, int, np.integer)):
            raise ValueError(
                f"MeasurementRecord metadata {key} must be a string or integer."
            )
        normalized = str(value).strip()
        if not normalized:
            raise ValueError(f"MeasurementRecord metadata {key} must be non-empty.")
        return normalized
    return f"station:{record.station_id}"


def _stack_optional_spectrum_variances(
    records: Sequence[MeasurementRecord],
) -> np.ndarray | None:
    """Stack spectrum variances without silently erasing mixed availability."""
    present = [record.spectrum_variance is not None for record in records]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(
            "All records must either contain spectrum variance or omit it."
        )
    return np.vstack([record.spectrum_variance for record in records])


def _stack_isotope_counts(
    records: Sequence[MeasurementRecord],
    isotope_names: tuple[str, ...],
) -> np.ndarray | None:
    """Stack complete isotope-count mappings in declared channel order."""
    present = [record.counts_by_isotope is not None for record in records]
    if not any(present):
        return None
    if not all(present):
        raise ValueError("All records must either contain isotope counts or omit them.")
    expected = set(isotope_names)
    rows: list[list[float]] = []
    for index, record in enumerate(records):
        counts = record.counts_by_isotope or {}
        if set(counts) != expected:
            raise ValueError(
                f"Record {index} isotope count channels do not match {isotope_names}."
            )
        rows.append([float(counts[name]) for name in isotope_names])
    return np.asarray(rows, dtype=float)


def _stack_isotope_covariances(
    records: Sequence[MeasurementRecord],
    isotope_names: tuple[str, ...],
) -> np.ndarray | None:
    """Stack complete covariance matrices without inventing missing entries."""
    present = [record.count_covariance_by_isotope is not None for record in records]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(
            "All records must either contain isotope covariance or omit it."
        )
    expected = set(isotope_names)
    matrices: list[np.ndarray] = []
    for index, record in enumerate(records):
        covariance = record.count_covariance_by_isotope or {}
        if set(covariance) != expected or any(
            set(covariance[row_name]) != expected for row_name in isotope_names
        ):
            raise ValueError(
                f"Record {index} isotope covariance must contain a complete square matrix."
            )
        matrices.append(
            np.asarray(
                [
                    [float(covariance[row][column]) for column in isotope_names]
                    for row in isotope_names
                ],
                dtype=float,
            )
        )
    return np.stack(matrices, axis=0)


def observation_batch_from_records(
    records: Sequence[MeasurementRecord],
    isotope_names: Sequence[str],
) -> ObservationBatch:
    """Build one all-history batch from finalized estimator-independent records."""
    rows = tuple(records)
    if not rows:
        raise ValueError("At least one MeasurementRecord is required.")
    if any(not isinstance(record, MeasurementRecord) for record in rows):
        raise TypeError("records must contain only MeasurementRecord objects.")
    names = tuple(str(value) for value in isotope_names)
    first_edges = rows[0].energy_bin_edges_keV
    if any(
        not np.array_equal(record.energy_bin_edges_keV, first_edges)
        for record in rows[1:]
    ):
        raise ValueError("All records must use identical energy-bin edges.")
    return ObservationBatch(
        detector_positions_xyz=np.asarray(
            [record.detector_pose_xyz for record in rows], dtype=float
        ),
        detector_quaternions_wxyz=np.asarray(
            [record.detector_quat_wxyz for record in rows], dtype=float
        ),
        fe_indices=np.asarray(
            [record.fe_orientation_index for record in rows], dtype=np.int64
        ),
        pb_indices=np.asarray(
            [record.pb_orientation_index for record in rows], dtype=np.int64
        ),
        live_times_s=np.asarray([record.live_time_s for record in rows], dtype=float),
        spectrum_counts=np.vstack([record.spectrum_counts for record in rows]),
        spectrum_variances=_stack_optional_spectrum_variances(rows),
        energy_bin_edges_keV=np.asarray(first_edges, dtype=float),
        isotope_counts=_stack_isotope_counts(rows, names),
        isotope_covariances=_stack_isotope_covariances(rows, names),
        station_ids=np.asarray([record.station_id for record in rows], dtype=np.int64),
        isotope_names=names,
        step_ids=np.asarray([record.step_id for record in rows], dtype=np.int64),
        action_ids=np.asarray([record.action_id for record in rows], dtype=np.int64),
        travel_times_s=np.asarray(
            [record.travel_time_s for record in rows],
            dtype=float,
        ),
        shield_actuation_times_s=np.asarray(
            [record.shield_actuation_time_s for record in rows],
            dtype=float,
        ),
        shield_program_block_ids=tuple(
            _shield_program_block_id(record) for record in rows
        ),
    )


def observation_batch_from_log(log: MeasurementLog) -> ObservationBatch:
    """Convert a loaded versioned measurement log into an MLE batch."""
    if not isinstance(log, MeasurementLog):
        raise TypeError("log must be a MeasurementLog.")
    return observation_batch_from_records(log.records, log.context.isotopes)


def subset_observation_batch(
    batch: ObservationBatch,
    indices: Sequence[int] | np.ndarray,
) -> ObservationBatch:
    """Return a row subset while preserving all estimator data contracts."""
    selected = np.asarray(indices, dtype=np.int64).reshape(-1)
    if selected.size == 0:
        raise ValueError("An ObservationBatch subset must contain at least one row.")
    if np.any(selected < 0) or np.any(selected >= batch.measurement_count):
        raise IndexError("ObservationBatch subset index is out of range.")
    return ObservationBatch(
        detector_positions_xyz=batch.detector_positions_xyz[selected],
        detector_quaternions_wxyz=batch.detector_quaternions_wxyz[selected],
        fe_indices=batch.fe_indices[selected],
        pb_indices=batch.pb_indices[selected],
        live_times_s=batch.live_times_s[selected],
        spectrum_counts=batch.spectrum_counts[selected],
        spectrum_variances=(
            None
            if batch.spectrum_variances is None
            else batch.spectrum_variances[selected]
        ),
        energy_bin_edges_keV=batch.energy_bin_edges_keV,
        isotope_counts=(
            None if batch.isotope_counts is None else batch.isotope_counts[selected]
        ),
        isotope_covariances=(
            None
            if batch.isotope_covariances is None
            else batch.isotope_covariances[selected]
        ),
        station_ids=batch.station_ids[selected],
        isotope_names=batch.isotope_names,
        step_ids=batch.step_ids[selected],
        action_ids=batch.action_ids[selected],
        travel_times_s=batch.travel_times_s[selected],
        shield_actuation_times_s=batch.shield_actuation_times_s[selected],
        shield_program_block_ids=tuple(
            batch.shield_program_block_ids[int(index)] for index in selected
        ),
    )
