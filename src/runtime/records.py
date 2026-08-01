"""Estimator-independent records shared by acquisition and replay code.

The objects in this module intentionally contain no particle-filter or MLE
state.  A :class:`MeasurementRecord` is created only after a simulator
observation has been finalized (including adaptive-dwell merging and spectrum
processing), so the exact same record can be consumed by any estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from sim.protocol import SimulationObservation


MEASUREMENT_LOG_SCHEMA_VERSION = 1
SUPPORTED_MEASUREMENT_LOG_SCHEMA_VERSIONS = frozenset({1, 2})

_FORBIDDEN_REALIZED_SOURCE_MARKERS = (
    "truth",
    "sourcelayout",
    "sourceposition",
    "sourcelist",
    "sourcelocation",
    "sourcecoordinate",
)


def _normalized_estimator_input_field(name: str) -> str:
    """Collapse field-name punctuation and case for truth-hygiene checks."""
    return "".join(character for character in name.casefold() if character.isalnum())


def _is_forbidden_estimator_input_field(name: str, value: object) -> bool:
    """Return whether a field can disclose realized source truth to an estimator."""
    normalized = _normalized_estimator_input_field(name)
    if any(marker in normalized for marker in _FORBIDDEN_REALIZED_SOURCE_MARKERS):
        return True
    if normalized == "sources":
        return True
    if normalized.endswith(("numsources", "maxsources", "minsources")):
        return False
    return (
        normalized.endswith("sources")
        and normalized != "resources"
        and isinstance(value, (Mapping, list, tuple, str))
    )


def _is_forbidden_estimator_input_pointer(value: str) -> bool:
    """Return whether a string points at a realized source/truth artifact."""
    normalized = _normalized_estimator_input_field(value)
    markers = ("truth", "sourcelayout", "sourceposition", "pointsource")
    if not any(marker in normalized for marker in markers):
        return False
    lowered = value.casefold().strip()
    path_like = "/" in value or "\\" in value or lowered.endswith(
        (".json", ".jsonl", ".yaml", ".yml", ".toml", ".csv", ".npz", ".txt")
    )
    return path_like


def validate_truth_free_estimator_input(
    value: object,
    *,
    path: str,
) -> None:
    """Recursively reject realized truth/source fields from estimator inputs.

    Source-rate, source-strength, and source-extent physics remain valid model
    configuration. Only realized layouts, positions, coordinates, locations,
    source lists, and fields explicitly named as truth are forbidden.
    """
    if isinstance(value, Mapping):
        aggregate_validation_metrics = path.endswith(
            ".full_spectrum_generative_model.validation.metrics"
        )
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key: {key!r}.")
            child_path = f"{path}.{key}"
            if (
                not aggregate_validation_metrics
                and _is_forbidden_estimator_input_field(key, child)
            ):
                raise ValueError(
                    f"{child_path} is a forbidden realized-truth/source field in "
                    "estimator input."
                )
            validate_truth_free_estimator_input(child, path=child_path)
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            validate_truth_free_estimator_input(child, path=f"{path}[{index}]")
        return
    if isinstance(value, str) and _is_forbidden_estimator_input_pointer(value):
        raise ValueError(
            f"{path} points to a forbidden realized-truth/source artifact."
        )


def _json_value(value: object, *, path: str = "value") -> object:
    """Return a deterministic JSON-compatible copy or raise a clear error."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity.")
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item(), path=path)
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist(), path=path)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key: {key!r}.")
            normalized[key] = _json_value(item, path=f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{path} contains unsupported JSON value of type {type(value).__name__}."
    )


def canonical_json_bytes(value: object) -> bytes:
    """Encode *value* exactly as runtime JSON files are persisted."""
    normalized = _json_value(value)
    return (
        json.dumps(
            normalized,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    """Return the SHA-256 of :func:`canonical_json_bytes`."""
    return sha256(canonical_json_bytes(value)).hexdigest()


def _nonnegative_int(value: object, *, name: str) -> int:
    """Return a nonnegative integer without accepting booleans or coercion."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _finite_float(value: object, *, name: str, nonnegative: bool = False) -> float:
    """Return a finite float, optionally constrained to nonnegative values."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _vector(
    value: object,
    *,
    name: str,
    length: int | None = None,
    nonnegative: bool = False,
) -> NDArray[np.float64]:
    """Return an owned immutable finite vector with optional constraints."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a one-dimensional numeric array.") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}.")
    if length is not None and array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},); got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must contain only nonnegative values.")
    result = np.array(array, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _isotope_values(
    values: Mapping[str, object] | None,
    *,
    name: str,
    nonnegative: bool,
) -> dict[str, float] | None:
    """Normalize an optional isotope-to-scalar mapping."""
    if values is None:
        return None
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping or None.")
    result: dict[str, float] = {}
    for isotope, value in sorted(values.items()):
        if not isinstance(isotope, str) or not isotope.strip():
            raise ValueError(f"{name} keys must be non-empty isotope names.")
        result[isotope] = _finite_float(
            value,
            name=f"{name}[{isotope!r}]",
            nonnegative=nonnegative,
        )
    return result


def _isotope_covariance(
    covariance: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, dict[str, float]] | None:
    """Normalize and validate an optional isotope covariance mapping."""
    if covariance is None:
        return None
    if not isinstance(covariance, Mapping):
        raise TypeError("count_covariance_by_isotope must be a mapping or None.")

    result: dict[str, dict[str, float]] = {}
    for row_name, row in sorted(covariance.items()):
        if not isinstance(row_name, str) or not row_name.strip():
            raise ValueError("Covariance row keys must be non-empty isotope names.")
        if not isinstance(row, Mapping):
            raise TypeError(f"Covariance row {row_name!r} must be a mapping.")
        normalized_row: dict[str, float] = {}
        for column_name, value in sorted(row.items()):
            if not isinstance(column_name, str) or not column_name.strip():
                raise ValueError(
                    "Covariance column keys must be non-empty isotope names."
                )
            normalized_row[column_name] = _finite_float(
                value,
                name=f"count_covariance_by_isotope[{row_name!r}][{column_name!r}]",
            )
        if row_name in normalized_row and normalized_row[row_name] < 0.0:
            raise ValueError(
                f"Covariance diagonal for {row_name!r} must be nonnegative."
            )
        result[row_name] = normalized_row

    for row_name, row in result.items():
        for column_name, value in row.items():
            reverse = result.get(column_name, {}).get(row_name)
            if reverse is not None and not math.isclose(
                value,
                reverse,
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "count_covariance_by_isotope must be symmetric when both "
                    f"entries are supplied ({row_name!r}, {column_name!r})."
                )
    return result


@dataclass(frozen=True)
class RunContext:
    """Immutable, estimator-independent description of one acquisition run."""

    repository_commit: str
    runtime_config: dict[str, object]
    environment: dict[str, object]
    sim_backend: str
    spectrum_count_method: str
    isotopes: tuple[str, ...]
    obstacle_layout_path: str | None = None
    source_layout_path: str | None = None
    source_rate_model: str = "detector_cps_1m"
    metadata: dict[str, object] = field(default_factory=dict)
    run_id: str | None = None
    source_rate_semantics: dict[str, object] = field(
        default_factory=lambda: {
            "quantity": "expected_net_detector_count_rate",
            "unit": "cps",
            "normalization_distance_m": 1.0,
        }
    )
    forward_model_manifest: dict[str, object] | None = field(
        default=None,
        compare=False,
    )
    runtime_config_sha256: str | None = None
    schema_version: int = MEASUREMENT_LOG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate and freeze a JSON-safe, content-addressed run context."""
        if self.schema_version not in SUPPORTED_MEASUREMENT_LOG_SCHEMA_VERSIONS:
            raise ValueError(
                "schema_version must be one of "
                f"{sorted(SUPPORTED_MEASUREMENT_LOG_SCHEMA_VERSIONS)}; "
                f"got {self.schema_version}."
            )
        if (
            not isinstance(self.repository_commit, str)
            or not self.repository_commit.strip()
        ):
            raise ValueError("repository_commit must be a non-empty provenance string.")
        for name in ("sim_backend", "spectrum_count_method", "source_rate_model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")

        runtime_config = _json_value(self.runtime_config, path="runtime_config")
        environment = _json_value(self.environment, path="environment")
        metadata = _json_value(self.metadata, path="metadata")
        source_rate_semantics = _json_value(
            self.source_rate_semantics,
            path="source_rate_semantics",
        )
        forward_model_manifest = _json_value(
            self.forward_model_manifest,
            path="forward_model_manifest",
        )
        if not isinstance(runtime_config, dict):
            raise TypeError("runtime_config must be a mapping.")
        if not isinstance(environment, dict):
            raise TypeError("environment must be a mapping.")
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a mapping.")
        if not isinstance(source_rate_semantics, dict):
            raise TypeError("source_rate_semantics must be a mapping.")
        expected_semantics = {
            1: {
                "quantity": "expected_net_detector_count_rate",
                "unit": "cps",
                "normalization_distance_m": 1.0,
            },
            2: {
                "quantity": "expected_pre_dead_time_detector_pulse_rate",
                "unit": "cps",
                "normalization_distance_m": 1.0,
            },
        }[int(self.schema_version)]
        if source_rate_semantics != expected_semantics:
            raise ValueError(
                "source_rate_semantics must describe expected net detector cps at 1 m."
            )
        if forward_model_manifest is not None and not isinstance(
            forward_model_manifest,
            dict,
        ):
            raise TypeError("forward_model_manifest must be a mapping or None.")
        validate_truth_free_estimator_input(runtime_config, path="runtime_config")
        validate_truth_free_estimator_input(environment, path="environment")
        validate_truth_free_estimator_input(metadata, path="metadata")
        if self.source_layout_path is not None:
            raise ValueError(
                "source_layout_path is evaluation truth and must be None in "
                "MeasurementLog estimator inputs."
            )

        isotopes = tuple(self.isotopes)
        if not isotopes:
            raise ValueError("isotopes must contain at least one isotope name.")
        if any(not isinstance(name, str) or not name.strip() for name in isotopes):
            raise ValueError("isotopes must contain only non-empty strings.")
        if len(set(isotopes)) != len(isotopes):
            raise ValueError("isotopes must not contain duplicates.")

        computed_hash = canonical_json_sha256(runtime_config)
        if self.runtime_config_sha256 is not None:
            supplied_hash = str(self.runtime_config_sha256).lower()
            if len(supplied_hash) != 64 or any(
                character not in "0123456789abcdef" for character in supplied_hash
            ):
                raise ValueError(
                    "runtime_config_sha256 must be a lowercase 64-character SHA-256."
                )
        else:
            supplied_hash = computed_hash

        repository_commit = str(self.repository_commit).strip()
        if not repository_commit:
            raise ValueError("repository_commit must be a non-empty provenance string.")
        run_id = (
            str(self.run_id).strip()
            if self.run_id is not None
            else f"run-{supplied_hash[:16]}"
        )
        if not run_id:
            raise ValueError("run_id must be a non-empty string.")

        object.__setattr__(self, "runtime_config", runtime_config)
        object.__setattr__(self, "environment", environment)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "source_rate_semantics", source_rate_semantics)
        object.__setattr__(self, "forward_model_manifest", forward_model_manifest)
        object.__setattr__(self, "isotopes", isotopes)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "repository_commit", repository_commit)
        object.__setattr__(self, "runtime_config_sha256", supplied_hash)
        if self.obstacle_layout_path is not None:
            object.__setattr__(
                self,
                "obstacle_layout_path",
                str(self.obstacle_layout_path),
            )


@dataclass(frozen=True, eq=False)
class MeasurementRecord:
    """A finalized estimator input produced from one simulator observation."""

    station_id: int
    step_id: int

    detector_pose_xyz: tuple[float, float, float]
    detector_quat_wxyz: tuple[float, float, float, float]

    fe_orientation_index: int
    pb_orientation_index: int

    live_time_s: float
    travel_time_s: float
    shield_actuation_time_s: float

    spectrum_counts: NDArray[np.float64]
    spectrum_variance: NDArray[np.float64] | None
    energy_bin_edges_keV: NDArray[np.float64]

    counts_by_isotope: dict[str, float] | None
    count_covariance_by_isotope: dict[str, dict[str, float]] | None

    metadata: dict[str, object]
    action_id: int | None = None

    def __post_init__(self) -> None:
        """Validate and freeze one complete finalized measurement record."""
        station_id = _nonnegative_int(self.station_id, name="station_id")
        step_id = _nonnegative_int(self.step_id, name="step_id")
        action_id = _nonnegative_int(
            self.step_id if self.action_id is None else self.action_id,
            name="action_id",
        )
        fe_index = _nonnegative_int(
            self.fe_orientation_index,
            name="fe_orientation_index",
        )
        pb_index = _nonnegative_int(
            self.pb_orientation_index,
            name="pb_orientation_index",
        )
        if fe_index > 7 or pb_index > 7:
            raise ValueError("Fe/Pb orientation indices must lie in [0, 7].")

        pose_array = _vector(self.detector_pose_xyz, name="detector_pose_xyz", length=3)
        quat_array = _vector(
            self.detector_quat_wxyz,
            name="detector_quat_wxyz",
            length=4,
        )
        quaternion_norm = float(np.linalg.norm(quat_array))
        if not np.isclose(quaternion_norm, 1.0, rtol=1.0e-9, atol=1.0e-12):
            raise ValueError("detector_quat_wxyz must be a normalized quaternion.")

        spectrum = _vector(
            self.spectrum_counts,
            name="spectrum_counts",
            nonnegative=True,
        )
        if spectrum.size == 0:
            raise ValueError("spectrum_counts must contain at least one energy bin.")
        edges = _vector(
            self.energy_bin_edges_keV,
            name="energy_bin_edges_keV",
            length=spectrum.size + 1,
        )
        if np.any(np.diff(edges) <= 0.0):
            raise ValueError("energy_bin_edges_keV must be strictly increasing.")

        variance = None
        if self.spectrum_variance is not None:
            variance = _vector(
                self.spectrum_variance,
                name="spectrum_variance",
                length=spectrum.size,
                nonnegative=True,
            )

        counts = _isotope_values(
            self.counts_by_isotope,
            name="counts_by_isotope",
            nonnegative=True,
        )
        covariance = _isotope_covariance(self.count_covariance_by_isotope)
        metadata = _json_value(self.metadata, path="metadata")
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a mapping.")
        validate_truth_free_estimator_input(metadata, path="metadata")

        object.__setattr__(self, "station_id", station_id)
        object.__setattr__(self, "step_id", step_id)
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(
            self, "detector_pose_xyz", tuple(float(v) for v in pose_array)
        )
        object.__setattr__(
            self,
            "detector_quat_wxyz",
            tuple(float(v) for v in quat_array),
        )
        object.__setattr__(self, "fe_orientation_index", fe_index)
        object.__setattr__(self, "pb_orientation_index", pb_index)
        live_time_s = _finite_float(
            self.live_time_s,
            name="live_time_s",
            nonnegative=True,
        )
        if live_time_s <= 0.0:
            raise ValueError("live_time_s must be strictly positive.")
        object.__setattr__(self, "live_time_s", live_time_s)
        object.__setattr__(
            self,
            "travel_time_s",
            _finite_float(self.travel_time_s, name="travel_time_s", nonnegative=True),
        )
        object.__setattr__(
            self,
            "shield_actuation_time_s",
            _finite_float(
                self.shield_actuation_time_s,
                name="shield_actuation_time_s",
                nonnegative=True,
            ),
        )
        object.__setattr__(self, "spectrum_counts", spectrum)
        object.__setattr__(self, "spectrum_variance", variance)
        object.__setattr__(self, "energy_bin_edges_keV", edges)
        object.__setattr__(self, "counts_by_isotope", counts)
        object.__setattr__(self, "count_covariance_by_isotope", covariance)
        object.__setattr__(self, "metadata", metadata)

    @classmethod
    def from_simulation_observation(
        cls,
        observation: SimulationObservation,
        *,
        station_id: int,
        live_time_s: float,
        travel_time_s: float,
        shield_actuation_time_s: float,
        spectrum_variance: Sequence[float] | NDArray[np.floating] | None,
        counts_by_isotope: Mapping[str, float] | None,
        count_covariance_by_isotope: Mapping[str, Mapping[str, float]] | None,
        metadata: Mapping[str, object] | None = None,
        action_id: int | None = None,
        include_observation_metadata: bool = True,
    ) -> "MeasurementRecord":
        """Build a record from finalized simulator and spectrum-processing output.

        Processed isotope counts, covariance, variance, and timing are explicit
        keyword-only inputs.  They are deliberately not reconstructed from PF
        state or inferred from simulator metadata. Acquisition callers should
        set ``include_observation_metadata=False`` and supply an explicit
        allowlist mapping when simulator metadata can contain scene truth.
        """
        # Import lazily so replay-only consumers can read persisted records
        # without initializing simulator backends.
        from sim.protocol import SimulationObservation as LocalSimulationObservation

        if not isinstance(observation, LocalSimulationObservation):
            raise TypeError("observation must be a sim.protocol.SimulationObservation.")
        if not isinstance(include_observation_metadata, bool):
            raise TypeError("include_observation_metadata must be a boolean.")
        merged_metadata = (
            dict(observation.metadata) if include_observation_metadata else {}
        )
        if metadata is not None:
            merged_metadata.update(dict(metadata))
        return cls(
            station_id=station_id,
            step_id=observation.step_id,
            detector_pose_xyz=observation.detector_pose_xyz,
            detector_quat_wxyz=observation.detector_quat_wxyz,
            fe_orientation_index=observation.fe_orientation_index,
            pb_orientation_index=observation.pb_orientation_index,
            live_time_s=live_time_s,
            travel_time_s=travel_time_s,
            shield_actuation_time_s=shield_actuation_time_s,
            spectrum_counts=np.asarray(observation.spectrum_counts, dtype=np.float64),
            spectrum_variance=(
                None
                if spectrum_variance is None
                else np.asarray(spectrum_variance, dtype=np.float64)
            ),
            energy_bin_edges_keV=np.asarray(
                observation.energy_bin_edges_keV,
                dtype=np.float64,
            ),
            counts_by_isotope=(
                None if counts_by_isotope is None else dict(counts_by_isotope)
            ),
            count_covariance_by_isotope=(
                None
                if count_covariance_by_isotope is None
                else {
                    isotope: dict(row)
                    for isotope, row in count_covariance_by_isotope.items()
                }
            ),
            metadata=merged_metadata,
            action_id=action_id,
        )


def measurement_record_from_observation(
    observation: SimulationObservation,
    **finalized_fields: Any,
) -> MeasurementRecord:
    """Functional alias for :meth:`MeasurementRecord.from_simulation_observation`."""
    return MeasurementRecord.from_simulation_observation(
        observation, **finalized_fields
    )
