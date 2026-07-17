"""Causal, station-complete MeasurementLog prefix materialization."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Sequence

from runtime.measurement_log import (
    load_measurement_log,
    measurement_log_sha256,
    save_measurement_log,
)
from runtime.forward_model_manifest import file_backed_model_asset_paths
from runtime.records import MeasurementRecord, canonical_json_bytes


@dataclass(frozen=True, slots=True)
class MeasurementLogPrefix:
    """Describe one persisted causal prefix without estimator state."""

    output_dir: Path
    record_count: int
    covered_step_ids: tuple[int, ...]
    data_cutoff_step: int
    data_cutoff_station: int
    covered_records_sha256: str
    measurement_log_sha256: str


def covered_station_boundaries_sha256(
    records: Sequence[MeasurementRecord],
    *,
    source_run_id: str,
) -> str:
    """Hash only explicit station-end markers contained in the supplied records."""
    selected = tuple(records)
    if not selected:
        raise ValueError("Station-boundary coverage requires at least one record.")
    entries: list[dict[str, int]] = []
    for index, record in enumerate(selected):
        if record.metadata.get("station_complete") is not True:
            continue
        if (
            index + 1 < len(selected)
            and selected[index + 1].station_id == record.station_id
        ):
            raise ValueError("A station_complete marker precedes another station row.")
        entries.append(
            {"station_id": record.station_id, "terminal_step_id": record.step_id}
        )
    if not entries or entries[-1]["terminal_step_id"] != selected[-1].step_id:
        raise ValueError("A causal prefix must end at station_complete=true.")
    if {entry["station_id"] for entry in entries} != {
        record.station_id for record in selected
    }:
        raise ValueError(
            "Every station in a causal prefix must declare its end marker."
        )
    payload = {
        "schema_version": 1,
        "source_run_id": source_run_id,
        "station_end_steps": entries,
    }
    return sha256(canonical_json_bytes(payload)).hexdigest()


def _record_payload(record: MeasurementRecord) -> dict[str, object]:
    """Return every estimator-input field in deterministic JSON form."""
    isotope_counts = (
        None
        if record.counts_by_isotope is None
        else {
            isotope: record.counts_by_isotope[isotope]
            for isotope in sorted(record.counts_by_isotope)
        }
    )
    isotope_covariance = (
        None
        if record.count_covariance_by_isotope is None
        else {
            row_name: {
                column_name: record.count_covariance_by_isotope[row_name][column_name]
                for column_name in sorted(record.count_covariance_by_isotope[row_name])
            }
            for row_name in sorted(record.count_covariance_by_isotope)
        }
    )
    return {
        "step_id": record.step_id,
        "action_id": record.action_id,
        "station_id": record.station_id,
        "detector_pose_xyz": list(record.detector_pose_xyz),
        "detector_quat_wxyz": list(record.detector_quat_wxyz),
        "fe_orientation_index": record.fe_orientation_index,
        "pb_orientation_index": record.pb_orientation_index,
        "live_time_s": record.live_time_s,
        "travel_time_s": record.travel_time_s,
        "shield_actuation_time_s": record.shield_actuation_time_s,
        "energy_bin_edges_keV": record.energy_bin_edges_keV.tolist(),
        "spectrum_counts": record.spectrum_counts.tolist(),
        "spectrum_variance": (
            None
            if record.spectrum_variance is None
            else record.spectrum_variance.tolist()
        ),
        "isotope_counts": isotope_counts,
        "isotope_count_covariance": isotope_covariance,
        "metadata": record.metadata,
    }


def measurement_records_sha256(records: Sequence[MeasurementRecord]) -> str:
    """Hash an ordered sequence of complete estimator-input records."""
    rows = tuple(records)
    if not rows:
        raise ValueError("At least one record is required for a lineage digest.")
    return sha256(
        canonical_json_bytes([_record_payload(record) for record in rows])
    ).hexdigest()


def _prefix_stop_index(
    records: tuple[MeasurementRecord, ...],
    *,
    cutoff_step: int,
    cutoff_station: int,
    assert_station_complete: bool,
) -> int:
    """Validate an exact, explicitly attested station boundary."""
    step = int(cutoff_step)
    matching = [index for index, record in enumerate(records) if record.step_id == step]
    if not matching:
        raise ValueError(f"cutoff_step {step} is absent from the log.")
    stop = matching[0]
    record = records[stop]
    station = int(cutoff_station)
    if record.station_id != station:
        raise ValueError(
            f"cutoff_step {step} belongs to station {record.station_id}, not "
            f"cutoff_station {station}."
        )
    if stop + 1 < len(records) and (records[stop + 1].station_id == record.station_id):
        raise ValueError(
            f"cutoff_step {step} is not station-complete for station "
            f"{record.station_id}."
        )
    writer_marked_complete = record.metadata.get("station_complete") is True
    if not writer_marked_complete and not assert_station_complete:
        raise ValueError(
            "The cutoff record lacks metadata.station_complete=true; pass "
            "assert_station_complete=True only when an external validated "
            "schedule attests this exact step/station boundary."
        )
    return stop


def materialize_measurement_log_prefix(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    cutoff_step: int,
    cutoff_station: int,
    assert_station_complete: bool = False,
) -> MeasurementLogPrefix:
    """Persist one validated truth-free prefix ending at a station boundary.

    The output contains only fields already accepted by ``MeasurementLog``.
    Its bytes depend on the selected records and immutable run context, never
    on observations after the selected station.
    """
    source = Path(run_dir).resolve()
    target = Path(output_dir).resolve()
    if not isinstance(assert_station_complete, bool):
        raise TypeError("assert_station_complete must be a boolean.")
    if source == target or source in target.parents:
        raise ValueError("output_dir must not be the source log or its descendant.")
    log = load_measurement_log(source)
    stop = _prefix_stop_index(
        log.records,
        cutoff_step=cutoff_step,
        cutoff_station=cutoff_station,
        assert_station_complete=bool(assert_station_complete),
    )
    records = log.records[: stop + 1]
    has_writer_marker = records[-1].metadata.get("station_complete") is True
    attestation = (
        "covered_prefix_markers_v1"
        if has_writer_marker
        else "external_validated_schedule"
    )
    metadata = dict(log.context.metadata)
    metadata.pop("station_boundary_attestation", None)
    metadata["measurement_log_prefix"] = {
        "schema_version": 1,
        "source_run_id": log.context.run_id,
        "data_cutoff_step": records[-1].step_id,
        "data_cutoff_station": records[-1].station_id,
        "station_boundary_attestation": attestation,
    }
    if has_writer_marker:
        metadata["measurement_log_prefix"]["covered_station_boundaries_sha256"] = (  # type: ignore[index]
            covered_station_boundaries_sha256(
                records,
                source_run_id=log.context.run_id,
            )
        )
    prefix_context = replace(log.context, metadata=metadata)
    model_assets: dict[str, bytes] = {}
    for relative_name in file_backed_model_asset_paths(
        log.context.runtime_config,
        obstacle_layout_path=log.context.obstacle_layout_path,
    ):
        source_asset = source / relative_name
        if source_asset.is_file():
            model_assets[relative_name] = source_asset.read_bytes()
    saved = save_measurement_log(
        target,
        prefix_context,
        records,
        extra_artifacts=model_assets,
        model_asset_root=source,
    )
    digest = measurement_log_sha256(saved)
    return MeasurementLogPrefix(
        output_dir=saved,
        record_count=len(records),
        covered_step_ids=tuple(record.step_id for record in records),
        data_cutoff_step=records[-1].step_id,
        data_cutoff_station=records[-1].station_id,
        covered_records_sha256=measurement_records_sha256(records),
        measurement_log_sha256=digest,
    )


__all__ = [
    "covered_station_boundaries_sha256",
    "MeasurementLogPrefix",
    "materialize_measurement_log_prefix",
    "measurement_records_sha256",
]
