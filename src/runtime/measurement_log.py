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
    SUPPORTED_MEASUREMENT_LOG_SCHEMA_VERSIONS,
    MeasurementRecord,
    RunContext,
    canonical_json_bytes,
    validate_truth_free_estimator_input,
)
from runtime.forward_model_manifest import (
    SOURCE_RATE_SEMANTICS,
    build_forward_model_manifest,
    file_backed_model_asset_paths,
    resolve_file_backed_model_asset,
    validate_forward_model_manifest,
)


_REQUIRED_FILES = (
    "run_manifest.json",
    "runtime_config.resolved.json",
    "environment.json",
    "forward_model_manifest.json",
    "observations.npz",
    "observation_metadata.jsonl",
    "repository_commit.txt",
)
_LEGACY_COMMIT_FILENAME = "upstream_pf_commit.txt"
_TRUTH_FILENAME = "truth.json"
_LEGACY_TRUTH_FILENAME = "truth_sources.json"


def _is_forbidden_estimator_artifact(relative_name: str) -> bool:
    """Return whether any path component names truth or a source layout."""
    for part in Path(relative_name).parts:
        normalized = "".join(
            character for character in part.casefold() if character.isalnum()
        )
        if "truth" in normalized or "sourcelayout" in normalized:
            return True
    return False


@dataclass(frozen=True)
class MeasurementLog:
    """A completely reconstructed run context and ordered record sequence."""

    context: RunContext
    records: tuple[MeasurementRecord, ...]
    content_sha256: str | None = None


def measurement_log_sha256(run_dir: str | Path) -> str:
    """Hash the complete truth-free regular-file inventory of a log root."""
    root = Path(run_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Measurement log directory does not exist: {root}")
    inventory: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"MeasurementLog must not contain symlinks: {relative}")
        if _is_forbidden_estimator_artifact(relative):
            raise ValueError(
                "Truth or source-layout artifacts are forbidden inside an "
                "estimator-input MeasurementLog."
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(
                f"MeasurementLog contains a non-regular artifact: {relative}"
            )
        inventory[relative] = sha256(path.read_bytes()).hexdigest()
    if not inventory:
        raise ValueError("MeasurementLog contains no regular artifacts.")
    return sha256(canonical_json_bytes(inventory)).hexdigest()


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
        with zipfile.ZipFile(
            raw_handle, mode="w", compression=zipfile.ZIP_STORED
        ) as archive:
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


def _normalize_extra_artifacts(
    artifacts: Mapping[str, bytes] | None,
) -> dict[str, bytes]:
    """Validate deterministic truth-free auxiliary model artifacts."""
    if artifacts is None:
        return {}
    if not isinstance(artifacts, Mapping):
        raise TypeError("extra_artifacts must be a mapping or None.")
    reserved = set(_REQUIRED_FILES) | {
        _LEGACY_COMMIT_FILENAME,
        _TRUTH_FILENAME,
        _LEGACY_TRUTH_FILENAME,
    }
    normalized: dict[str, bytes] = {}
    for raw_name, raw_payload in artifacts.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError("extra_artifacts keys must be non-empty relative paths.")
        if "\\" in raw_name:
            raise ValueError("extra_artifacts paths must use forward slashes.")
        relative = Path(raw_name)
        name = relative.as_posix()
        if (
            relative.is_absolute()
            or name != raw_name
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError(f"Invalid extra_artifacts path: {raw_name!r}.")
        if name in reserved or _is_forbidden_estimator_artifact(name):
            raise ValueError(f"Forbidden extra_artifacts path: {name!r}.")
        if not isinstance(raw_payload, bytes):
            raise TypeError(f"extra_artifacts[{name!r}] must contain bytes.")
        normalized[name] = raw_payload
    paths = {name: Path(name) for name in normalized}
    for left_name, left in paths.items():
        for right_name, right in paths.items():
            if left_name != right_name and left in right.parents:
                raise ValueError(
                    "extra_artifacts cannot use one artifact as another's directory."
                )
    return {name: normalized[name] for name in sorted(normalized)}


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
    seen_actions: set[int] = set()
    previous_step_id: int | None = None
    previous_station_id: int | None = None
    count_presence: list[bool] = []
    covariance_presence: list[bool] = []

    for index, record in enumerate(records):
        if not isinstance(record, MeasurementRecord):
            raise TypeError(f"records[{index}] is not a MeasurementRecord.")
        if record.step_id in seen_steps:
            raise ValueError(
                f"Duplicate finalized step_id {record.step_id} in measurement log."
            )
        if previous_step_id is not None and record.step_id <= previous_step_id:
            raise ValueError(
                "Measurement records must be stored in strictly increasing causal step_id order."
            )
        seen_steps.add(record.step_id)
        previous_step_id = record.step_id
        if record.action_id in seen_actions:
            raise ValueError(
                f"Duplicate action_id {record.action_id} in measurement log."
            )
        seen_actions.add(int(record.action_id))
        if previous_station_id is not None and record.station_id < previous_station_id:
            raise ValueError(
                "station_id values must be nondecreasing in causal file order."
            )
        previous_station_id = record.station_id
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
            count_presence.append(True)
            record_isotopes.update(record.counts_by_isotope)
            if set(record.counts_by_isotope) != known_isotopes:
                raise ValueError(
                    f"records[{index}] isotope counts must contain every manifest isotope."
                )
        else:
            count_presence.append(False)
        if record.count_covariance_by_isotope is not None:
            covariance_presence.append(True)
            record_isotopes.update(record.count_covariance_by_isotope)
            for row in record.count_covariance_by_isotope.values():
                record_isotopes.update(row)
            covariance = record.count_covariance_by_isotope
            if set(covariance) != known_isotopes or any(
                set(covariance.get(isotope, {})) != known_isotopes
                for isotope in context.isotopes
            ):
                raise ValueError(
                    f"records[{index}] isotope covariance must be a complete square matrix."
                )
            matrix = np.asarray(
                [
                    [covariance[row][column] for column in context.isotopes]
                    for row in context.isotopes
                ],
                dtype=float,
            )
            eigenvalues = np.linalg.eigvalsh(matrix)
            scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
            if np.any(eigenvalues < -1.0e-9 * scale):
                raise ValueError(
                    f"records[{index}] isotope covariance must be positive semidefinite."
                )
        else:
            covariance_presence.append(False)
        unknown = record_isotopes - known_isotopes
        if unknown:
            raise ValueError(
                f"records[{index}] contains isotopes absent from RunContext: "
                f"{sorted(unknown)}."
            )
    if any(count_presence) and not all(count_presence):
        raise ValueError("All records must either contain isotope counts or omit them.")
    if any(covariance_presence) and not all(covariance_presence):
        raise ValueError(
            "All records must either contain isotope covariance or omit it."
        )
    if any(covariance_presence) and not all(count_presence):
        raise ValueError("Isotope covariance requires corresponding isotope counts.")
    return len(records), bin_count


def _normalize_truth_sources(
    truth_sources: Iterable[Mapping[str, object]] | None,
) -> tuple[dict[str, object], ...] | None:
    """Validate a separate evaluation-only truth payload for persistence."""
    if truth_sources is None:
        return None
    normalized: list[dict[str, object]] = []
    for index, item in enumerate(truth_sources):
        if not isinstance(item, Mapping):
            raise TypeError(f"truth_sources[{index}] must be a mapping.")
        value = json.loads(canonical_json_bytes(dict(item)).decode("utf-8"))
        if not isinstance(value, dict):
            raise TypeError(f"truth_sources[{index}] must normalize to an object.")
        normalized.append(value)
    return tuple(normalized)


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
            for (
                row_isotope,
                covariance_row,
            ) in record.count_covariance_by_isotope.items():
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
        "action_id": np.asarray(
            [record.action_id for record in records],
            dtype=np.int64,
        ),
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
    forward_model_manifest: Mapping[str, object],
    forward_model_manifest_sha256: str,
    artifact_hashes: Mapping[str, str],
) -> dict[str, object]:
    """Build the public, versioned manifest for one finalized log."""
    return {
        "schema_version": context.schema_version,
        "run_id": context.run_id,
        "repository_commit": context.repository_commit,
        "resolved_config_sha256": context.runtime_config_sha256,
        "forward_model_manifest_sha256": forward_model_manifest_sha256,
        "source_rate_model": context.source_rate_model,
        "source_rate_semantics": context.source_rate_semantics,
        "model_identifiers": forward_model_manifest["model_identifiers"],
        "index_conventions": {
            "record_order": "causal_step_order",
            "step_id": "zero_based_strictly_increasing",
            "action_id": "zero_based_unique_measurement_action",
            "station_id": "zero_based_nondecreasing_station_group",
        },
        "artifact_hashes": dict(artifact_hashes),
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
    *,
    extra_artifacts: Mapping[str, bytes] | None = None,
    model_asset_root: str | Path | None = None,
) -> Path:
    """Atomically publish a complete measurement-log directory.

    ``run_dir`` must not already exist.  All files are first fsynced in a
    sibling temporary directory, then that directory is renamed into place.
    Refusing replacement avoids exposing mixed generations of a run.
    """
    if not isinstance(context, RunContext):
        raise TypeError("context must be a RunContext.")
    current_config_hash = sha256(
        canonical_json_bytes(context.runtime_config)
    ).hexdigest()
    if current_config_hash != context.runtime_config_sha256:
        raise ValueError(
            "RunContext.runtime_config changed after its runtime_config_sha256 was computed."
        )
    record_tuple = tuple(records)
    record_count, bin_count = _validate_records(context, record_tuple)
    auxiliary_artifacts = _normalize_extra_artifacts(extra_artifacts)
    declared_assets = set(
        file_backed_model_asset_paths(
            context.runtime_config,
            obstacle_layout_path=context.obstacle_layout_path,
        )
    )
    undeclared_assets = sorted(set(auxiliary_artifacts) - declared_assets)
    if undeclared_assets:
        raise ValueError(
            "extra_artifacts contains files not declared by the forward model: "
            f"{undeclared_assets}."
        )
    for name, payload in auxiliary_artifacts.items():
        resolved_asset = resolve_file_backed_model_asset(
            name,
            field_name=f"extra_artifacts[{name!r}]",
            run_root=model_asset_root,
        )
        if resolved_asset.read_bytes() != payload:
            raise ValueError(
                f"extra_artifacts[{name!r}] differs from its resolved model asset."
            )
    expected_forward_manifest = build_forward_model_manifest(
        runtime_config=context.runtime_config,
        environment=context.environment,
        obstacle_layout_path=context.obstacle_layout_path,
        isotopes=context.isotopes,
        repository_commit=str(context.repository_commit),
        resolved_config_sha256=str(context.runtime_config_sha256),
        source_rate_model=context.source_rate_model,
        run_root=model_asset_root,
    )
    if context.forward_model_manifest is None:
        forward_manifest = expected_forward_manifest
    else:
        forward_manifest = validate_forward_model_manifest(
            context.forward_model_manifest,
            runtime_config=context.runtime_config,
            environment=context.environment,
            obstacle_layout_path=context.obstacle_layout_path,
            isotopes=context.isotopes,
            repository_commit=str(context.repository_commit),
            resolved_config_sha256=str(context.runtime_config_sha256),
            source_rate_model=context.source_rate_model,
            run_root=model_asset_root,
        )

    target = Path(run_dir)
    if target.exists():
        raise FileExistsError(f"Measurement log target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )

    try:
        _write_bytes(
            temporary / "runtime_config.resolved.json",
            canonical_json_bytes(context.runtime_config),
        )
        _write_bytes(
            temporary / "environment.json",
            canonical_json_bytes(context.environment),
        )
        _write_bytes(
            temporary / "forward_model_manifest.json",
            canonical_json_bytes(forward_manifest),
        )
        _write_deterministic_npz(
            temporary / "observations.npz",
            _arrays_for_records(context, record_tuple),
        )

        metadata_lines = b"".join(
            (
                json.dumps(
                    {
                        "run_id": context.run_id,
                        "array_index": record_index,
                        "action_id": record.action_id,
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
            for record_index, record in enumerate(record_tuple)
        )
        _write_bytes(temporary / "observation_metadata.jsonl", metadata_lines)
        _write_bytes(
            temporary / "repository_commit.txt",
            (str(context.repository_commit) + "\n").encode("utf-8"),
        )
        for name, payload in auxiliary_artifacts.items():
            artifact_path = temporary / name
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            _write_bytes(artifact_path, payload)
            _fsync_directory(artifact_path.parent)
        artifact_names = (
            "runtime_config.resolved.json",
            "environment.json",
            "forward_model_manifest.json",
            "observations.npz",
            "observation_metadata.jsonl",
            "repository_commit.txt",
        ) + tuple(auxiliary_artifacts)
        artifact_hashes = {
            name: sha256((temporary / name).read_bytes()).hexdigest()
            for name in artifact_names
        }
        forward_manifest_sha256 = artifact_hashes["forward_model_manifest.json"]
        _write_bytes(
            temporary / "run_manifest.json",
            canonical_json_bytes(
                _manifest(
                    context,
                    record_count=record_count,
                    energy_bin_count=bin_count,
                    forward_model_manifest=forward_manifest,
                    forward_model_manifest_sha256=forward_manifest_sha256,
                    artifact_hashes=artifact_hashes,
                )
            ),
        )

        _fsync_directory(temporary)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return target


def save_evaluation_truth(
    evaluation_dir: str | Path,
    truth_sources: Iterable[Mapping[str, object]],
    *,
    overwrite: bool = False,
) -> Path:
    """Persist truth in an explicit evaluation directory outside a log root."""
    normalized = _normalize_truth_sources(truth_sources)
    if normalized is None:
        raise ValueError("truth_sources must not be None.")
    root = Path(evaluation_dir)
    target = root / _TRUTH_FILENAME
    root.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Evaluation truth already exists: {target}")
    temporary = root / f".{_TRUTH_FILENAME}.tmp-{os.getpid()}"
    if temporary.exists():
        temporary.unlink()
    try:
        _write_bytes(temporary, canonical_json_bytes(list(normalized)))
        os.replace(temporary, target)
        _fsync_directory(root)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def load_evaluation_truth(evaluation_dir: str | Path) -> tuple[dict[str, object], ...]:
    """Load evaluation-only truth without exposing it through MeasurementLog."""
    root = Path(evaluation_dir)
    canonical = root / _TRUTH_FILENAME
    legacy = root / _LEGACY_TRUTH_FILENAME
    target = canonical if canonical.is_file() else legacy
    if not target.is_file():
        raise FileNotFoundError(f"Evaluation truth does not exist below {root}.")
    payload = _read_json(target)
    if not isinstance(payload, list) or not all(
        isinstance(item, dict) for item in payload
    ):
        raise ValueError(f"{target.name} must contain a JSON array of objects.")
    return tuple(dict(item) for item in payload)


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

    def __init__(
        self,
        run_dir: str | Path,
        context: RunContext,
    ) -> None:
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
                    "run_id": context.run_id,
                    "repository_commit": context.repository_commit,
                    "resolved_config_sha256": context.runtime_config_sha256,
                    "runtime_config": context.runtime_config,
                    "environment": context.environment,
                    "sim_backend": context.sim_backend,
                    "spectrum_count_method": context.spectrum_count_method,
                    "source_rate_model": context.source_rate_model,
                    "isotopes": list(context.isotopes),
                    "obstacle_layout_path": context.obstacle_layout_path,
                    "source_layout_path": context.source_layout_path,
                    "source_rate_semantics": context.source_rate_semantics,
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
                        "schema_version": self._context.schema_version,
                        "run_id": self._context.run_id,
                        "record_index": len(self._records),
                        "array_index": len(self._records),
                        "action_id": record.action_id,
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


def _validate_masked_numeric_storage(
    values: NDArray[np.float64],
    entry_presence: NDArray[np.bool_],
    *,
    name: str,
    nonnegative: bool,
) -> None:
    """Require finite present entries and canonical NaN absent storage."""
    expanded_presence = np.broadcast_to(entry_presence, values.shape)
    present_values = values[expanded_presence]
    absent_values = values[~expanded_presence]
    if np.any(~np.isfinite(present_values)):
        raise ValueError(f"observations.npz {name} has non-finite present values.")
    if nonnegative and np.any(present_values < 0.0):
        raise ValueError(f"observations.npz {name} has negative present values.")
    if absent_values.size and np.any(~np.isnan(absent_values)):
        raise ValueError(
            f"observations.npz {name} must store NaN exactly where presence is false."
        )


def _validate_v2_forward_model_manifest(
    payload: Mapping[str, object],
    *,
    run_manifest: Mapping[str, object],
    repository_commit: str,
    resolved_config_sha256: str,
) -> dict[str, object]:
    """Validate the producer-owned full-spectrum forward identity for log v2.

    The standalone estimator deliberately does not rebuild PF simulation
    physics.  It verifies the immutable producer manifest and consumes that
    identity as estimator input; the PF runtime remains the sole authority for
    constructing and validating the physical model assets.
    """
    required = {
        "schema_version",
        "repository_commit",
        "resolved_config_sha256",
        "source_rate_model",
        "source_rate_semantics",
        "model_identifiers",
        "units",
        "response_semantics",
        "line_mu_by_isotope",
    }
    if set(payload) != required or payload.get("schema_version") != 2:
        raise ValueError(
            "MeasurementLog v2 forward_model_manifest.json has an invalid schema."
        )
    expected_fields = {
        "repository_commit": repository_commit,
        "resolved_config_sha256": resolved_config_sha256,
        "source_rate_model": run_manifest.get("source_rate_model"),
        "source_rate_semantics": run_manifest.get("source_rate_semantics"),
        "model_identifiers": run_manifest.get("model_identifiers"),
    }
    for field_name, expected in expected_fields.items():
        if payload.get(field_name) != expected:
            raise ValueError(
                "MeasurementLog v2 forward model and run manifest differ at "
                f"{field_name}."
            )
    units = payload.get("units")
    if not isinstance(units, Mapping) or units.get("source_strength") != (
        "detector_cps_1m"
    ):
        raise ValueError(
            "MeasurementLog v2 forward model must use detector_cps_1m strength."
        )
    if not isinstance(payload.get("response_semantics"), Mapping):
        raise ValueError(
            "MeasurementLog v2 forward model requires response_semantics."
        )
    isotopes = tuple(str(value) for value in run_manifest.get("isotopes", ()))
    line_table = payload.get("line_mu_by_isotope")
    if not isinstance(line_table, Mapping) or set(line_table) != set(isotopes):
        raise ValueError(
            "MeasurementLog v2 line_mu_by_isotope must match manifest isotopes."
        )
    for isotope in isotopes:
        entries = line_table[isotope]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Forward line table for {isotope} must be nonempty.")
        energies: list[float] = []
        weight_sum = 0.0
        for entry in entries:
            if not isinstance(entry, Mapping) or set(entry) != {
                "energy_keV",
                "weight",
                "fe",
                "pb",
            }:
                raise ValueError(f"Forward line entry for {isotope} is invalid.")
            values = [float(entry[name]) for name in ("energy_keV", "weight", "fe", "pb")]
            if not np.all(np.isfinite(values)) or any(value < 0.0 for value in values):
                raise ValueError(f"Forward line entry for {isotope} is nonphysical.")
            energies.append(values[0])
            weight_sum += values[1]
        if any(right <= left for left, right in zip(energies, energies[1:])):
            raise ValueError(f"Forward line energies for {isotope} must increase.")
        if not np.isclose(weight_sum, 1.0, rtol=0.0, atol=1.0e-9):
            raise ValueError(f"Forward line weights for {isotope} must sum to one.")
    return dict(payload)


def load_measurement_log(run_dir: str | Path) -> MeasurementLog:
    """Validate and reconstruct producer v1 or raw full-spectrum v2 logs."""
    root = Path(run_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Measurement log directory does not exist: {root}")
    content_digest = measurement_log_sha256(root)
    required_without_commit = tuple(
        name for name in _REQUIRED_FILES if name != "repository_commit.txt"
    )
    missing = [name for name in required_without_commit if not (root / name).is_file()]
    canonical_commit_path = root / "repository_commit.txt"
    legacy_commit_path = root / _LEGACY_COMMIT_FILENAME
    if not canonical_commit_path.is_file() and not legacy_commit_path.is_file():
        missing.append("repository_commit.txt")
    if missing:
        raise ValueError(f"Measurement log is missing required files: {missing}.")

    manifest = _read_json(root / "run_manifest.json")
    runtime_config = _read_json(root / "runtime_config.resolved.json")
    environment = _read_json(root / "environment.json")
    forward_model_manifest = _read_json(root / "forward_model_manifest.json")
    if not isinstance(manifest, dict):
        raise ValueError("run_manifest.json must contain a JSON object.")
    if not isinstance(runtime_config, dict):
        raise ValueError("runtime_config.resolved.json must contain a JSON object.")
    if not isinstance(environment, dict):
        raise ValueError("environment.json must contain a JSON object.")
    if not isinstance(forward_model_manifest, dict):
        raise ValueError("forward_model_manifest.json must contain a JSON object.")
    if manifest.get("source_layout_path") is not None:
        raise ValueError(
            "run_manifest.json source_layout_path must be null for estimator input."
        )
    truth_free_manifest = dict(manifest)
    truth_free_manifest.pop("source_layout_path", None)
    validate_truth_free_estimator_input(
        truth_free_manifest,
        path="run_manifest",
    )
    validate_truth_free_estimator_input(runtime_config, path="runtime_config")
    validate_truth_free_estimator_input(environment, path="environment")
    validate_truth_free_estimator_input(
        forward_model_manifest,
        path="forward_model_manifest",
    )

    schema_version = manifest.get("schema_version")
    if schema_version not in SUPPORTED_MEASUREMENT_LOG_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported measurement-log schema_version {schema_version!r}; "
            f"expected one of {sorted(SUPPORTED_MEASUREMENT_LOG_SCHEMA_VERSIONS)}."
        )
    expected_hash = str(
        manifest.get(
            "resolved_config_sha256",
            manifest.get("runtime_config_sha256", ""),
        )
    )
    actual_hash = sha256(
        (root / "runtime_config.resolved.json").read_bytes()
    ).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(
            "runtime_config.resolved.json does not match resolved_config_sha256."
        )
    if manifest.get("environment") != environment:
        raise ValueError("environment.json does not match the run manifest.")

    commit_path = (
        canonical_commit_path if canonical_commit_path.is_file() else legacy_commit_path
    )
    repository_commit = commit_path.read_text(encoding="utf-8").strip()
    manifest_commit = manifest.get(
        "repository_commit",
        manifest.get("upstream_pf_commit"),
    )
    if repository_commit != manifest_commit:
        raise ValueError(f"{commit_path.name} does not match the run manifest.")

    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_manifest.json run_id must be a non-empty string.")
    source_rate_semantics = manifest.get("source_rate_semantics")
    expected_source_rate_semantics = (
        {
            "quantity": "expected_pre_dead_time_detector_pulse_rate",
            "unit": "cps",
            "normalization_distance_m": 1.0,
        }
        if schema_version == 2
        else SOURCE_RATE_SEMANTICS
    )
    if source_rate_semantics != expected_source_rate_semantics:
        raise ValueError("run_manifest.json source_rate_semantics is incompatible.")
    forward_manifest_digest = sha256(
        (root / "forward_model_manifest.json").read_bytes()
    ).hexdigest()
    if forward_manifest_digest != manifest.get("forward_model_manifest_sha256"):
        raise ValueError(
            "forward_model_manifest.json does not match forward_model_manifest_sha256."
        )

    artifact_hashes = manifest.get("artifact_hashes")
    if not isinstance(artifact_hashes, dict):
        raise ValueError("run_manifest.json artifact_hashes must be an object.")
    required_artifact_names = {
        name for name in required_without_commit if name != "run_manifest.json"
    } | {commit_path.name}
    actual_artifact_names = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "run_manifest.json"
    }
    if not required_artifact_names.issubset(actual_artifact_names):
        raise ValueError("Measurement log is missing a required hashed artifact.")
    if set(artifact_hashes) != actual_artifact_names:
        raise ValueError(
            "run_manifest.json artifact_hashes must name every estimator input artifact."
        )
    for artifact_name in sorted(actual_artifact_names):
        digest = sha256((root / artifact_name).read_bytes()).hexdigest()
        if artifact_hashes.get(artifact_name) != digest:
            raise ValueError(
                f"Measurement-log artifact {artifact_name!r} does not match its SHA-256."
            )

    if schema_version == 2:
        validated_forward_manifest = _validate_v2_forward_model_manifest(
            forward_model_manifest,
            run_manifest=manifest,
            repository_commit=repository_commit,
            resolved_config_sha256=expected_hash,
        )
    else:
        validated_forward_manifest = validate_forward_model_manifest(
            forward_model_manifest,
            runtime_config=runtime_config,
            environment=environment,
            obstacle_layout_path=manifest.get("obstacle_layout_path"),
            isotopes=tuple(manifest.get("isotopes", ())),
            repository_commit=repository_commit,
            resolved_config_sha256=expected_hash,
            source_rate_model=str(manifest.get("source_rate_model", "")),
            run_root=root,
        )
    if manifest.get("model_identifiers") != validated_forward_manifest.get(
        "model_identifiers"
    ):
        raise ValueError(
            "run_manifest.json model_identifiers do not match the forward-model manifest."
        )
    index_conventions = manifest.get("index_conventions")
    expected_index_conventions = {
        "record_order": "causal_step_order",
        "step_id": "zero_based_strictly_increasing",
        "action_id": "zero_based_unique_measurement_action",
        "station_id": "zero_based_nondecreasing_station_group",
    }
    if index_conventions != expected_index_conventions:
        raise ValueError("run_manifest.json index_conventions are incompatible.")

    try:
        context = RunContext(
            repository_commit=repository_commit,
            runtime_config=runtime_config,
            environment=environment,
            sim_backend=str(manifest["sim_backend"]),
            spectrum_count_method=str(
                manifest.get(
                    "spectrum_count_method",
                    manifest.get("observation_model", ""),
                )
            ),
            isotopes=tuple(manifest["isotopes"]),
            obstacle_layout_path=manifest.get("obstacle_layout_path"),
            source_layout_path=manifest.get("source_layout_path"),
            source_rate_model=str(manifest["source_rate_model"]),
            metadata=dict(manifest.get("metadata", {})),
            run_id=run_id,
            source_rate_semantics=dict(source_rate_semantics),
            forward_model_manifest=validated_forward_manifest,
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

    base_array_names = {
        "step_id",
        "action_id",
        "station_id",
        "detector_pose_xyz",
        "detector_quat_wxyz",
        "fe_orientation_index",
        "pb_orientation_index",
        "live_time_s",
        "travel_time_s",
        "shield_actuation_time_s",
        "energy_bin_edges_keV",
        "spectrum_counts",
    }
    legacy_optional_array_names = {
        "spectrum_variance",
        "spectrum_variance_present",
        "isotope_counts",
        "isotope_counts_present",
        "isotope_counts_record_present",
        "isotope_count_covariance",
        "isotope_count_covariance_present",
        "isotope_count_covariance_record_present",
    }
    expected_array_names = (
        base_array_names
        if schema_version == 2
        else base_array_names | legacy_optional_array_names
    )
    if set(archive) != expected_array_names:
        missing_arrays = sorted(expected_array_names - set(archive))
        extra_arrays = sorted(set(archive) - expected_array_names)
        raise ValueError(
            "observations.npz schema mismatch; "
            f"missing={missing_arrays}, extra={extra_arrays}."
        )

    isotope_count = len(context.isotopes)
    step_ids = _array(archive, "step_id", shape=(record_count,), dtype=np.int64)
    action_ids = _array(archive, "action_id", shape=(record_count,), dtype=np.int64)
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
        dtype=np.int64 if schema_version == 2 else np.float64,
    )
    if schema_version == 2:
        if np.any(spectra < 0):
            raise ValueError(
                "MeasurementLog v2 spectrum_counts must be nonnegative integers."
            )
        variances = None
        variance_present = None
        isotope_counts = None
        isotope_counts_present = None
        isotope_counts_record_present = None
        isotope_covariances = None
        isotope_covariance_present = None
        isotope_covariance_record_present = None
    else:
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
        _validate_masked_numeric_storage(
            variances,
            variance_present[:, None],
            name="spectrum_variance",
            nonnegative=True,
        )
        _validate_masked_numeric_storage(
            isotope_counts,
            isotope_counts_present,
            name="isotope_counts",
            nonnegative=True,
        )
        _validate_masked_numeric_storage(
            isotope_covariances,
            isotope_covariance_present,
            name="isotope_count_covariance",
            nonnegative=False,
        )
        if np.any(
            np.any(isotope_counts_present, axis=1)
            != isotope_counts_record_present
        ):
            raise ValueError(
                "isotope_counts_record_present does not match per-entry "
                "presence masks."
            )
        covariance_row_presence = np.any(
            isotope_covariance_present,
            axis=(1, 2),
        )
        if np.any(covariance_row_presence != isotope_covariance_record_present):
            raise ValueError(
                "isotope_count_covariance_record_present does not match entry masks."
            )

    metadata_lines = (
        (root / "observation_metadata.jsonl").read_text(encoding="utf-8").splitlines()
    )
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
        if not isinstance(payload, dict) or not isinstance(
            payload.get("metadata"), dict
        ):
            raise ValueError(
                f"observation_metadata.jsonl line {index + 1} has an invalid payload."
            )
        expected_metadata_fields = {
            "action_id",
            "array_index",
            "metadata",
            "run_id",
            "station_id",
            "step_id",
        }
        if set(payload) != expected_metadata_fields:
            raise ValueError(
                "observation_metadata.jsonl lines must contain exactly "
                f"{sorted(expected_metadata_fields)}."
            )
        validate_truth_free_estimator_input(
            payload["metadata"],
            path=f"observation_metadata[{index}].metadata",
        )
        if payload.get("step_id") != int(step_ids[index]) or payload.get(
            "station_id"
        ) != int(station_ids[index]):
            raise ValueError(
                "observation_metadata.jsonl identifiers do not match observations.npz."
            )
        canonical_metadata = {
            "run_id": context.run_id,
            "array_index": index,
            "action_id": int(action_ids[index]),
        }
        for name, expected_value in canonical_metadata.items():
            if payload.get(name) != expected_value:
                raise ValueError(
                    f"observation_metadata.jsonl line {index + 1} {name} is incompatible."
                )
        metadata_by_record.append(dict(payload["metadata"]))

    records: list[MeasurementRecord] = []
    for record_index in range(record_count):
        counts = None
        if (
            isotope_counts_record_present is not None
            and isotope_counts_present is not None
            and isotope_counts is not None
            and bool(isotope_counts_record_present[record_index])
        ):
            counts = {
                isotope: float(isotope_counts[record_index, isotope_index])
                for isotope_index, isotope in enumerate(context.isotopes)
                if bool(isotope_counts_present[record_index, isotope_index])
            }
        covariance = None
        if (
            isotope_covariance_record_present is not None
            and isotope_covariance_present is not None
            and isotope_covariances is not None
            and bool(isotope_covariance_record_present[record_index])
        ):
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
                action_id=int(action_ids[record_index]),
                detector_pose_xyz=tuple(float(v) for v in poses[record_index]),
                detector_quat_wxyz=tuple(float(v) for v in quaternions[record_index]),
                fe_orientation_index=int(fe_indices[record_index]),
                pb_orientation_index=int(pb_indices[record_index]),
                live_time_s=float(live_times[record_index]),
                travel_time_s=float(travel_times[record_index]),
                shield_actuation_time_s=float(actuation_times[record_index]),
                spectrum_counts=spectra[record_index],
                spectrum_variance=(
                    None
                    if variances is None
                    or variance_present is None
                    or not bool(variance_present[record_index])
                    else variances[record_index]
                ),
                energy_bin_edges_keV=energy_edges,
                counts_by_isotope=counts,
                count_covariance_by_isotope=covariance,
                metadata=metadata_by_record[record_index],
            )
        )

    _validate_records(context, tuple(records))
    return MeasurementLog(
        context=context,
        records=tuple(records),
        content_sha256=content_digest,
    )
