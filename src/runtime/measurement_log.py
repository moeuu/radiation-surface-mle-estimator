"""Versioned, deterministic persistence for estimator-independent records."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Mapping
import zipfile

import numpy as np
from numpy.typing import NDArray

from runtime.records import (
    MEASUREMENT_LOG_SCHEMA_VERSION,
    MeasurementRecord,
    RunContext,
    canonical_json_bytes,
)


_REQUIRED_FILES = (
    "run_manifest.json",
    "runtime_config.resolved.json",
    "environment.json",
    "observations.npz",
    "observation_metadata.jsonl",
    "upstream_pf_commit.txt",
)


@dataclass(frozen=True)
class MeasurementLog:
    """A completely reconstructed run context and ordered record sequence."""

    context: RunContext
    records: tuple[MeasurementRecord, ...]


def _write_bytes(path: Path, payload: bytes) -> None:
    """Create, flush, and fsync one binary file without replacing it."""
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    """Best-effort fsync of directory entries on platforms that support it."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_deterministic_npz(
    path: Path,
    arrays: Mapping[str, NDArray[np.generic]],
) -> None:
    """Write an ``np.load`` compatible archive without wall-clock timestamps."""
    with path.open("xb") as raw_handle:
        with zipfile.ZipFile(raw_handle, mode="w", compression=zipfile.ZIP_STORED) as archive:
            for name, array in arrays.items():
                if not name or "/" in name or "\\" in name:
                    raise ValueError(f"Invalid NPZ member name: {name!r}.")
                buffer = BytesIO()
                np.lib.format.write_array(
                    buffer,
                    np.asanyarray(array),
                    allow_pickle=False,
                )
                entry = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_STORED
                entry.create_system = 3
                entry.external_attr = 0o600 << 16
                archive.writestr(entry, buffer.getvalue())
        raw_handle.flush()
        os.fsync(raw_handle.fileno())


def _validate_records(
    context: RunContext,
    records: tuple[MeasurementRecord, ...],
) -> tuple[int, int]:
    """Validate one homogeneous record sequence and return row/bin counts."""
    if not records:
        raise ValueError("At least one MeasurementRecord is required to save a log.")

    first_edges = records[0].energy_bin_edges_keV
    bin_count = records[0].spectrum_counts.size
    known_isotopes = set(context.isotopes)
    seen_steps: set[int] = set()

    for index, record in enumerate(records):
        if not isinstance(record, MeasurementRecord):
            raise TypeError(f"records[{index}] is not a MeasurementRecord.")
        if record.step_id in seen_steps:
            raise ValueError(f"Duplicate finalized step_id {record.step_id} in measurement log.")
        seen_steps.add(record.step_id)
        if record.spectrum_counts.shape != (bin_count,):
            raise ValueError(
                f"records[{index}].spectrum_counts has shape "
                f"{record.spectrum_counts.shape}; expected ({bin_count},)."
            )
        if not np.array_equal(record.energy_bin_edges_keV, first_edges):
            raise ValueError(
                f"records[{index}] uses different energy_bin_edges_keV; "
                "one log must use one exact energy-bin definition."
            )

        record_isotopes: set[str] = set()
        if record.counts_by_isotope is not None:
            record_isotopes.update(record.counts_by_isotope)
        if record.count_covariance_by_isotope is not None:
            record_isotopes.update(record.count_covariance_by_isotope)
            for row in record.count_covariance_by_isotope.values():
                record_isotopes.update(row)
        unknown = record_isotopes - known_isotopes
        if unknown:
            raise ValueError(
                f"records[{index}] contains isotopes absent from RunContext: "
                f"{sorted(unknown)}."
            )
    return len(records), bin_count


def _arrays_for_records(
    context: RunContext,
    records: tuple[MeasurementRecord, ...],
) -> dict[str, NDArray[np.generic]]:
    """Convert records into the dense, presence-masked NPZ array schema."""
    record_count = len(records)
    isotope_count = len(context.isotopes)
    bin_count = records[0].spectrum_counts.size
    isotope_index = {name: index for index, name in enumerate(context.isotopes)}

    spectrum_variance = np.full((record_count, bin_count), np.nan, dtype=np.float64)
    spectrum_variance_present = np.zeros(record_count, dtype=np.bool_)
    isotope_counts = np.full((record_count, isotope_count), np.nan, dtype=np.float64)
    isotope_counts_present = np.zeros((record_count, isotope_count), dtype=np.bool_)
    isotope_counts_record_present = np.zeros(record_count, dtype=np.bool_)
    isotope_covariance = np.full(
        (record_count, isotope_count, isotope_count),
        np.nan,
        dtype=np.float64,
    )
    isotope_covariance_present = np.zeros(
        (record_count, isotope_count, isotope_count),
        dtype=np.bool_,
    )
    isotope_covariance_record_present = np.zeros(record_count, dtype=np.bool_)

    for row_index, record in enumerate(records):
        if record.spectrum_variance is not None:
            spectrum_variance[row_index] = record.spectrum_variance
            spectrum_variance_present[row_index] = True
        if record.counts_by_isotope is not None:
            isotope_counts_record_present[row_index] = True
            for isotope, value in record.counts_by_isotope.items():
                column_index = isotope_index[isotope]
                isotope_counts[row_index, column_index] = value
                isotope_counts_present[row_index, column_index] = True
        if record.count_covariance_by_isotope is not None:
            isotope_covariance_record_present[row_index] = True
            for row_isotope, covariance_row in record.count_covariance_by_isotope.items():
                covariance_row_index = isotope_index[row_isotope]
                for column_isotope, value in covariance_row.items():
                    covariance_column_index = isotope_index[column_isotope]
                    isotope_covariance[
                        row_index,
                        covariance_row_index,
                        covariance_column_index,
                    ] = value
                    isotope_covariance_present[
                        row_index,
                        covariance_row_index,
                        covariance_column_index,
                    ] = True

    # Insertion order is part of the deterministic archive representation.
    return {
        "step_id": np.asarray([record.step_id for record in records], dtype=np.int64),
        "station_id": np.asarray(
            [record.station_id for record in records],
            dtype=np.int64,
        ),
        "detector_pose_xyz": np.asarray(
            [record.detector_pose_xyz for record in records],
            dtype=np.float64,
        ),
        "detector_quat_wxyz": np.asarray(
            [record.detector_quat_wxyz for record in records],
            dtype=np.float64,
        ),
        "fe_orientation_index": np.asarray(
            [record.fe_orientation_index for record in records],
            dtype=np.int64,
        ),
        "pb_orientation_index": np.asarray(
            [record.pb_orientation_index for record in records],
            dtype=np.int64,
        ),
        "live_time_s": np.asarray(
            [record.live_time_s for record in records],
            dtype=np.float64,
        ),
        "travel_time_s": np.asarray(
            [record.travel_time_s for record in records],
            dtype=np.float64,
        ),
        "shield_actuation_time_s": np.asarray(
            [record.shield_actuation_time_s for record in records],
            dtype=np.float64,
        ),
        "energy_bin_edges_keV": np.asarray(
            records[0].energy_bin_edges_keV,
            dtype=np.float64,
        ),
        "spectrum_counts": np.asarray(
            [record.spectrum_counts for record in records],
            dtype=np.float64,
        ),
        "spectrum_variance": spectrum_variance,
        "spectrum_variance_present": spectrum_variance_present,
        "isotope_counts": isotope_counts,
        "isotope_counts_present": isotope_counts_present,
        "isotope_counts_record_present": isotope_counts_record_present,
        "isotope_count_covariance": isotope_covariance,
        "isotope_count_covariance_present": isotope_covariance_present,
        "isotope_count_covariance_record_present": isotope_covariance_record_present,
    }


def _manifest(
    context: RunContext,
    *,
    record_count: int,
    energy_bin_count: int,
) -> dict[str, object]:
    """Build the public, versioned manifest for one finalized log."""
    return {
        "schema_version": context.schema_version,
        "upstream_pf_commit": context.upstream_pf_commit,
        "runtime_config_sha256": context.runtime_config_sha256,
        "source_rate_model": context.source_rate_model,
        "isotopes": list(context.isotopes),
        "environment": context.environment,
        "obstacle_layout_path": context.obstacle_layout_path,
        "source_layout_path": context.source_layout_path,
        "sim_backend": context.sim_backend,
        "spectrum_count_method": context.spectrum_count_method,
        "metadata": context.metadata,
        "record_count": record_count,
        "energy_bin_count": energy_bin_count,
    }


def save_measurement_log(
    run_dir: str | Path,
    context: RunContext,
    records: Iterable[MeasurementRecord],
) -> Path:
    """Atomically publish a complete measurement-log directory.

    ``run_dir`` must not already exist.  All files are first fsynced in a
    sibling temporary directory, then that directory is renamed into place.
    Refusing replacement avoids exposing mixed generations of a run.
    """
    if not isinstance(context, RunContext):
        raise TypeError("context must be a RunContext.")
    current_config_hash = sha256(canonical_json_bytes(context.runtime_config)).hexdigest()
    if current_config_hash != context.runtime_config_sha256:
        raise ValueError(
            "RunContext.runtime_config changed after its runtime_config_sha256 was computed."
        )
    record_tuple = tuple(records)
    record_count, bin_count = _validate_records(context, record_tuple)

    target = Path(run_dir)
    if target.exists():
        raise FileExistsError(f"Measurement log target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )

    try:
        _write_bytes(
            temporary / "run_manifest.json",
            canonical_json_bytes(
                _manifest(context, record_count=record_count, energy_bin_count=bin_count)
            ),
        )
        _write_bytes(
            temporary / "runtime_config.resolved.json",
            canonical_json_bytes(context.runtime_config),
        )
        _write_bytes(
            temporary / "environment.json",
            canonical_json_bytes(context.environment),
        )
        _write_deterministic_npz(
            temporary / "observations.npz",
            _arrays_for_records(context, record_tuple),
        )

        metadata_lines = b"".join(
            (
                json.dumps(
                    {
                        "station_id": record.station_id,
                        "step_id": record.step_id,
                        "metadata": record.metadata,
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            for record in record_tuple
        )
        _write_bytes(temporary / "observation_metadata.jsonl", metadata_lines)
        _write_bytes(
            temporary / "upstream_pf_commit.txt",
            (context.upstream_pf_commit + "\n").encode("utf-8"),
        )
        if context.truth_sources is not None:
            _write_bytes(
                temporary / "truth_sources.json",
                canonical_json_bytes(list(context.truth_sources)),
            )

        _fsync_directory(temporary)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return target


class MeasurementLogRecorder:
    """Durably stage records before estimator updates, then publish one log.

    Appending writes a complete, independently recoverable one-record shard
    into a private sibling staging directory and fsyncs it before returning.
    ``finalize`` converts the in-memory ordered history into the documented
    dense NPZ/JSON log with :func:`save_measurement_log`, then removes the
    staging directory.  If acquisition or estimation fails, the staging
    directory is deliberately retained so the input that caused the failure
    is not lost.
    """

    def __init__(self, run_dir: str | Path, context: RunContext) -> None:
        """Create an empty durable recorder for a target that does not exist."""
        if not isinstance(context, RunContext):
            raise TypeError("context must be a RunContext.")
        target = Path(run_dir)
        if target.exists():
            raise FileExistsError(f"Measurement log target already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{target.name}.inprogress-",
                dir=str(target.parent),
            )
        )
        self._target = target
        self._context = context
        self._staging = staging
        self._records: list[MeasurementRecord] = []
        self._step_ids: set[int] = set()
        self._finalized = False
        _write_bytes(
            staging / "run_context.json",
            canonical_json_bytes(
                {
                    "schema_version": context.schema_version,
                    "upstream_pf_commit": context.upstream_pf_commit,
                    "runtime_config_sha256": context.runtime_config_sha256,
                    "runtime_config": context.runtime_config,
                    "environment": context.environment,
                    "sim_backend": context.sim_backend,
                    "spectrum_count_method": context.spectrum_count_method,
                    "source_rate_model": context.source_rate_model,
                    "isotopes": list(context.isotopes),
                    "obstacle_layout_path": context.obstacle_layout_path,
                    "source_layout_path": context.source_layout_path,
                    "truth_sources": context.truth_sources,
                    "metadata": context.metadata,
                }
            ),
        )
        (staging / "records").mkdir()
        _fsync_directory(staging)

    @property
    def target(self) -> Path:
        """Return the final public log path."""
        return self._target

    @property
    def staging_dir(self) -> Path:
        """Return the private recovery directory used until finalization."""
        return self._staging

    @property
    def records(self) -> tuple[MeasurementRecord, ...]:
        """Return the validated records staged so far in acquisition order."""
        return tuple(self._records)

    @property
    def finalized(self) -> bool:
        """Return whether the standard log has already been published."""
        return self._finalized

    def append(self, record: MeasurementRecord) -> None:
        """Fsync one finalized record shard before allowing estimation to run."""
        if self._finalized:
            raise RuntimeError("MeasurementLogRecorder is already finalized.")
        if not isinstance(record, MeasurementRecord):
            raise TypeError("record must be a MeasurementRecord.")
        if record.step_id in self._step_ids:
            raise ValueError(f"Duplicate finalized step_id {record.step_id}.")
        candidate_records = (*self._records, record)
        _validate_records(self._context, candidate_records)

        shard_name = f"{len(self._records):08d}-step-{record.step_id:08d}"
        shard_root = self._staging / "records"
        temporary = Path(tempfile.mkdtemp(prefix=f".{shard_name}-", dir=shard_root))
        published = shard_root / shard_name
        try:
            _write_deterministic_npz(
                temporary / "observation.npz",
                _arrays_for_records(self._context, (record,)),
            )
            _write_bytes(
                temporary / "metadata.json",
                canonical_json_bytes(
                    {
                        "station_id": record.station_id,
                        "step_id": record.step_id,
                        "metadata": record.metadata,
                    }
                ),
            )
            _fsync_directory(temporary)
            os.replace(temporary, published)
            _fsync_directory(shard_root)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        self._records.append(record)
        self._step_ids.add(record.step_id)

    def finalize(self) -> Path:
        """Atomically publish the standard log and remove recovery staging."""
        if self._finalized:
            return self._target
        if not self._records:
            raise ValueError("Cannot finalize a measurement log without records.")
        save_measurement_log(self._target, self._context, self._records)
        self._finalized = True
        shutil.rmtree(self._staging)
        _fsync_directory(self._target.parent)
        return self._target

    def discard_empty(self) -> None:
        """Remove staging for a run that ended before producing any observation."""
        if self._finalized:
            return
        if self._records:
            raise RuntimeError("Cannot discard a recorder that contains measurements.")
        shutil.rmtree(self._staging)
        _fsync_directory(self._target.parent)
        self._finalized = True


def _read_json(path: Path) -> object:
    """Read one JSON file and normalize parse/read failures."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read valid JSON from {path}.") from exc


def _array(
    archive: Mapping[str, NDArray[np.generic]],
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[object] | type,
) -> NDArray[np.generic]:
    """Return one required archive array after exact shape/dtype validation."""
    if name not in archive:
        raise ValueError(f"observations.npz is missing required array {name!r}.")
    value = np.asarray(archive[name])
    if value.shape != shape:
        raise ValueError(
            f"observations.npz array {name!r} has shape {value.shape}; expected {shape}."
        )
    expected_dtype = np.dtype(dtype)
    if value.dtype != expected_dtype:
        raise ValueError(
            f"observations.npz array {name!r} has invalid dtype {value.dtype}; "
            f"expected {expected_dtype}."
        )
    return np.array(value, copy=True)


def load_measurement_log(run_dir: str | Path) -> MeasurementLog:
    """Validate and reconstruct a measurement log saved by this module."""
    root = Path(run_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Measurement log directory does not exist: {root}")
    missing = [name for name in _REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(f"Measurement log is missing required files: {missing}.")

    manifest = _read_json(root / "run_manifest.json")
    runtime_config = _read_json(root / "runtime_config.resolved.json")
    environment = _read_json(root / "environment.json")
    if not isinstance(manifest, dict):
        raise ValueError("run_manifest.json must contain a JSON object.")
    if not isinstance(runtime_config, dict):
        raise ValueError("runtime_config.resolved.json must contain a JSON object.")
    if not isinstance(environment, dict):
        raise ValueError("environment.json must contain a JSON object.")

    schema_version = manifest.get("schema_version")
    if schema_version != MEASUREMENT_LOG_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported measurement-log schema_version {schema_version!r}; "
            f"expected {MEASUREMENT_LOG_SCHEMA_VERSION}."
        )
    expected_hash = str(manifest.get("runtime_config_sha256", ""))
    actual_hash = sha256((root / "runtime_config.resolved.json").read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError("runtime_config.resolved.json does not match runtime_config_sha256.")
    if manifest.get("environment") != environment:
        raise ValueError("environment.json does not match the run manifest.")

    upstream_commit = (root / "upstream_pf_commit.txt").read_text(encoding="utf-8").strip()
    if upstream_commit != manifest.get("upstream_pf_commit"):
        raise ValueError("upstream_pf_commit.txt does not match the run manifest.")

    truth_sources = None
    truth_path = root / "truth_sources.json"
    if truth_path.exists():
        truth_payload = _read_json(truth_path)
        if not isinstance(truth_payload, list) or not all(
            isinstance(item, dict) for item in truth_payload
        ):
            raise ValueError("truth_sources.json must contain a list of objects.")
        truth_sources = tuple(dict(item) for item in truth_payload)

    try:
        context = RunContext(
            upstream_pf_commit=upstream_commit,
            runtime_config=runtime_config,
            environment=environment,
            sim_backend=str(manifest["sim_backend"]),
            spectrum_count_method=str(manifest["spectrum_count_method"]),
            isotopes=tuple(manifest["isotopes"]),
            obstacle_layout_path=manifest.get("obstacle_layout_path"),
            source_layout_path=manifest.get("source_layout_path"),
            source_rate_model=str(manifest["source_rate_model"]),
            truth_sources=truth_sources,
            metadata=dict(manifest.get("metadata", {})),
            runtime_config_sha256=expected_hash,
            schema_version=int(schema_version),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("run_manifest.json contains an invalid RunContext.") from exc

    try:
        record_count = int(manifest["record_count"])
        bin_count = int(manifest["energy_bin_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("run_manifest.json has invalid record/bin counts.") from exc
    if record_count <= 0 or bin_count <= 0:
        raise ValueError("run_manifest.json record/bin counts must be positive.")

    try:
        with np.load(root / "observations.npz", allow_pickle=False) as loaded:
            archive = {name: np.array(loaded[name], copy=True) for name in loaded.files}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ValueError("Could not read a valid observations.npz archive.") from exc

    isotope_count = len(context.isotopes)
    step_ids = _array(archive, "step_id", shape=(record_count,), dtype=np.int64)
    station_ids = _array(archive, "station_id", shape=(record_count,), dtype=np.int64)
    poses = _array(
        archive,
        "detector_pose_xyz",
        shape=(record_count, 3),
        dtype=np.float64,
    )
    quaternions = _array(
        archive,
        "detector_quat_wxyz",
        shape=(record_count, 4),
        dtype=np.float64,
    )
    fe_indices = _array(
        archive,
        "fe_orientation_index",
        shape=(record_count,),
        dtype=np.int64,
    )
    pb_indices = _array(
        archive,
        "pb_orientation_index",
        shape=(record_count,),
        dtype=np.int64,
    )
    live_times = _array(
        archive,
        "live_time_s",
        shape=(record_count,),
        dtype=np.float64,
    )
    travel_times = _array(
        archive,
        "travel_time_s",
        shape=(record_count,),
        dtype=np.float64,
    )
    actuation_times = _array(
        archive,
        "shield_actuation_time_s",
        shape=(record_count,),
        dtype=np.float64,
    )
    energy_edges = _array(
        archive,
        "energy_bin_edges_keV",
        shape=(bin_count + 1,),
        dtype=np.float64,
    )
    spectra = _array(
        archive,
        "spectrum_counts",
        shape=(record_count, bin_count),
        dtype=np.float64,
    )
    variances = _array(
        archive,
        "spectrum_variance",
        shape=(record_count, bin_count),
        dtype=np.float64,
    )
    variance_present = _array(
        archive,
        "spectrum_variance_present",
        shape=(record_count,),
        dtype=np.bool_,
    )
    isotope_counts = _array(
        archive,
        "isotope_counts",
        shape=(record_count, isotope_count),
        dtype=np.float64,
    )
    isotope_counts_present = _array(
        archive,
        "isotope_counts_present",
        shape=(record_count, isotope_count),
        dtype=np.bool_,
    )
    isotope_counts_record_present = _array(
        archive,
        "isotope_counts_record_present",
        shape=(record_count,),
        dtype=np.bool_,
    )
    isotope_covariances = _array(
        archive,
        "isotope_count_covariance",
        shape=(record_count, isotope_count, isotope_count),
        dtype=np.float64,
    )
    isotope_covariance_present = _array(
        archive,
        "isotope_count_covariance_present",
        shape=(record_count, isotope_count, isotope_count),
        dtype=np.bool_,
    )
    isotope_covariance_record_present = _array(
        archive,
        "isotope_count_covariance_record_present",
        shape=(record_count,),
        dtype=np.bool_,
    )

    metadata_lines = (root / "observation_metadata.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    if len(metadata_lines) != record_count:
        raise ValueError(
            "observation_metadata.jsonl line count does not match record_count."
        )
    metadata_by_record: list[dict[str, object]] = []
    for index, line in enumerate(metadata_lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"observation_metadata.jsonl line {index + 1} is invalid JSON."
            ) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
            raise ValueError(
                f"observation_metadata.jsonl line {index + 1} has an invalid payload."
            )
        if payload.get("step_id") != int(step_ids[index]) or payload.get(
            "station_id"
        ) != int(station_ids[index]):
            raise ValueError(
                "observation_metadata.jsonl identifiers do not match observations.npz."
            )
        metadata_by_record.append(dict(payload["metadata"]))

    records: list[MeasurementRecord] = []
    for record_index in range(record_count):
        counts = None
        if bool(isotope_counts_record_present[record_index]):
            counts = {
                isotope: float(isotope_counts[record_index, isotope_index])
                for isotope_index, isotope in enumerate(context.isotopes)
                if bool(isotope_counts_present[record_index, isotope_index])
            }
        covariance = None
        if bool(isotope_covariance_record_present[record_index]):
            covariance = {}
            for row_index, row_isotope in enumerate(context.isotopes):
                row = {
                    column_isotope: float(
                        isotope_covariances[record_index, row_index, column_index]
                    )
                    for column_index, column_isotope in enumerate(context.isotopes)
                    if bool(
                        isotope_covariance_present[
                            record_index,
                            row_index,
                            column_index,
                        ]
                    )
                }
                if row:
                    covariance[row_isotope] = row

        records.append(
            MeasurementRecord(
                station_id=int(station_ids[record_index]),
                step_id=int(step_ids[record_index]),
                detector_pose_xyz=tuple(float(v) for v in poses[record_index]),
                detector_quat_wxyz=tuple(
                    float(v) for v in quaternions[record_index]
                ),
                fe_orientation_index=int(fe_indices[record_index]),
                pb_orientation_index=int(pb_indices[record_index]),
                live_time_s=float(live_times[record_index]),
                travel_time_s=float(travel_times[record_index]),
                shield_actuation_time_s=float(actuation_times[record_index]),
                spectrum_counts=spectra[record_index],
                spectrum_variance=(
                    variances[record_index]
                    if bool(variance_present[record_index])
                    else None
                ),
                energy_bin_edges_keV=energy_edges,
                counts_by_isotope=counts,
                count_covariance_by_isotope=covariance,
                metadata=metadata_by_record[record_index],
            )
        )

    _validate_records(context, tuple(records))
    return MeasurementLog(context=context, records=tuple(records))
