"""Standalone orchestration for replaying versioned measurement logs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
from importlib import import_module
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from measurement.continuous_kernels import ContinuousKernel
from measurement.model import EnvironmentConfig
from measurement.observation_model import (
    RuntimeObservationModel,
    build_runtime_observation_model,
    continuous_kernel_from_observation_model,
)
from measurement.obstacle_assets import (
    ObstacleComponent,
    line_transport_model_from_components,
    transport_model_from_components,
)
from measurement.obstacles import ObstacleGrid
from runtime.forward_model_manifest import resolve_file_backed_model_asset
from runtime.measurement_log import MeasurementLog, load_measurement_log
from runtime.records import canonical_json_bytes, canonical_json_sha256

from .config import MLEConfig
from .estimator import SurfaceMLEEstimator
from .observation_batch import observation_batch_from_log
from .provenance import estimator_provenance
from .types import MLEEstimate, ObservationBatch


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_RATE_MODEL = "detector_cps_1m"


ReplaySaveHook = Callable[[MLEEstimate, Path], object]


@dataclass(frozen=True)
class ReplayContext:
    """All validated local objects required for one all-history replay fit."""

    run_dir: Path
    log: MeasurementLog
    batch: ObservationBatch
    config: MLEConfig
    environment: EnvironmentConfig
    obstacle_grid: ObstacleGrid | None
    resolved_obstacle_path: Path | None
    observation_model: RuntimeObservationModel
    kernel: ContinuousKernel
    config_sha256: str
    resolved_estimator_config_sha256: str


@dataclass(frozen=True)
class ReplayResult:
    """Return a fitted estimate together with its fully resolved replay context."""

    estimate: MLEEstimate
    context: ReplayContext
    saved_output: object | None = None


def _normalized_source_rate_model(value: object, *, origin: str) -> str:
    """Validate detector-count-rate source-strength semantics."""
    normalized = str(value).strip().lower()
    if normalized != _SOURCE_RATE_MODEL:
        raise ValueError(
            f"{origin} source_rate_model must be {_SOURCE_RATE_MODEL!r}; got {value!r}."
        )
    return normalized


def _sequence_of_floats(
    value: object,
    *,
    name: str,
    length: int,
    positive: bool,
) -> tuple[float, ...]:
    """Parse a finite fixed-length numeric sequence."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of {length} numbers.")
    if len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} numbers.")
    parsed = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in parsed):
        raise ValueError(f"{name} must contain only finite numbers.")
    if positive and any(item <= 0.0 for item in parsed):
        raise ValueError(f"{name} must contain only positive numbers.")
    return parsed


def _environment_dimensions(
    payload: Mapping[str, object],
) -> tuple[float, float, float] | None:
    """Extract room dimensions from one environment/runtime mapping."""
    if all(key in payload for key in ("size_x", "size_y", "size_z")):
        return _sequence_of_floats(
            [payload["size_x"], payload["size_y"], payload["size_z"]],
            name="environment size_x/size_y/size_z",
            length=3,
            positive=True,
        )
    if all(key in payload for key in ("size_x_m", "size_y_m", "size_z_m")):
        return _sequence_of_floats(
            [payload["size_x_m"], payload["size_y_m"], payload["size_z_m"]],
            name="environment size_x_m/size_y_m/size_z_m",
            length=3,
            positive=True,
        )
    for key in ("room_size_xyz", "size_xyz"):
        if key in payload:
            return _sequence_of_floats(
                payload[key],
                name=key,
                length=3,
                positive=True,
            )
    return None


def _environment_mappings(
    environment_payload: Mapping[str, object],
    runtime_config: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    """Return explicit environment mappings in decreasing precedence order."""
    candidates: list[Mapping[str, object]] = [environment_payload]
    nested_environment = environment_payload.get("environment")
    if isinstance(nested_environment, Mapping):
        candidates.append(nested_environment)
    runtime_environment = runtime_config.get("environment")
    if isinstance(runtime_environment, Mapping):
        candidates.append(runtime_environment)
    candidates.append(runtime_config)
    return tuple(candidates)


def _resolve_environment_config(log: MeasurementLog) -> EnvironmentConfig:
    """Construct EnvironmentConfig from persisted environment/runtime payloads."""
    mappings = _environment_mappings(
        log.context.environment, log.context.runtime_config
    )
    dimensions = None
    for payload in mappings:
        dimensions = _environment_dimensions(payload)
        if dimensions is not None:
            break
    if dimensions is None:
        raise ValueError(
            "Measurement log does not contain room dimensions. Expected size_x/size_y/"
            "size_z or room_size_xyz in environment/runtime config."
        )

    detector_position = None
    for payload in mappings:
        for key in ("detector_position", "detector_position_xyz"):
            if key in payload and payload[key] is not None:
                detector_position = _sequence_of_floats(
                    payload[key],
                    name=key,
                    length=3,
                    positive=False,
                )
                break
        if detector_position is not None:
            break
    return EnvironmentConfig(
        size_x=dimensions[0],
        size_y=dimensions[1],
        size_z=dimensions[2],
        detector_position=detector_position,
    )


def _axis_aligned_obstacle_grid(
    payloads: Sequence[object],
    *,
    environment_payload: Mapping[str, object],
    isotopes: tuple[str, ...],
) -> ObstacleGrid | None:
    """Convert neutral axis-aligned boxes into local surface/transport geometry."""
    if not payloads:
        return None
    components: list[ObstacleComponent] = []
    footprints: list[tuple[float, float, float, float]] = []
    for index, raw_payload in enumerate(payloads):
        if not isinstance(raw_payload, Mapping):
            raise ValueError(f"environment obstacles[{index}] must be an object.")
        if raw_payload.get("kind") != "axis_aligned_box":
            raise ValueError("Only axis_aligned_box neutral obstacles are supported.")
        lower = _sequence_of_floats(
            raw_payload.get("min_xyz_m"),
            name=f"obstacles[{index}].min_xyz_m",
            length=3,
            positive=False,
        )
        upper = _sequence_of_floats(
            raw_payload.get("max_xyz_m"),
            name=f"obstacles[{index}].max_xyz_m",
            length=3,
            positive=False,
        )
        size = tuple(upper[axis] - lower[axis] for axis in range(3))
        if any(value <= 0.0 for value in size):
            raise ValueError("Neutral obstacle max_xyz_m must exceed min_xyz_m.")
        material = str(raw_payload.get("material", "")).strip()
        if not material:
            raise ValueError("Neutral obstacle material must be non-empty.")
        components.append(
            ObstacleComponent(
                name=str(raw_payload.get("object_id", f"obstacle-{index}")),
                center_xyz=tuple(
                    0.5 * (lower[axis] + upper[axis]) for axis in range(3)
                ),
                size_xyz=size,
                material=material,
            )
        )
        footprints.append((lower[0], lower[1], size[0], size[1]))

    cell_size = float(footprints[0][2])
    if any(
        not np.isclose(width, cell_size, rtol=1.0e-9, atol=1.0e-12)
        or not np.isclose(height, cell_size, rtol=1.0e-9, atol=1.0e-12)
        for _, _, width, height in footprints
    ):
        raise ValueError(
            "Neutral obstacle footprints must be equal square cells for surface patches."
        )
    origin_xyz = _sequence_of_floats(
        environment_payload.get("surface_origin_xyz_m", (0.0, 0.0, 0.0)),
        name="surface_origin_xyz_m",
        length=3,
        positive=False,
    )
    dimensions = _environment_dimensions(environment_payload)
    if dimensions is None:
        raise ValueError("Neutral obstacle geometry requires room dimensions.")
    grid_shape_float = (
        dimensions[0] / cell_size,
        dimensions[1] / cell_size,
    )
    grid_shape = tuple(int(round(value)) for value in grid_shape_float)
    if any(
        not np.isclose(value, rounded, rtol=1.0e-9, atol=1.0e-12)
        for value, rounded in zip(grid_shape_float, grid_shape, strict=True)
    ):
        raise ValueError("Room dimensions must align to neutral obstacle cell size.")
    blocked_cells: list[tuple[int, int]] = []
    for x_min, y_min, _, _ in footprints:
        indices_float = (
            (x_min - origin_xyz[0]) / cell_size,
            (y_min - origin_xyz[1]) / cell_size,
        )
        indices = tuple(int(round(value)) for value in indices_float)
        if any(
            not np.isclose(value, rounded, rtol=1.0e-9, atol=1.0e-12)
            for value, rounded in zip(indices_float, indices, strict=True)
        ):
            raise ValueError(
                "Neutral obstacle footprints must align to the surface grid."
            )
        blocked_cells.append(indices)
    boxes, mu_by_isotope = transport_model_from_components(
        components,
        isotopes=isotopes,
    )
    line_mu_by_isotope = line_transport_model_from_components(
        components,
        isotopes=isotopes,
    )
    return ObstacleGrid(
        origin=(origin_xyz[0], origin_xyz[1]),
        cell_size=cell_size,
        grid_shape=(grid_shape[0], grid_shape[1]),
        blocked_cells=tuple(blocked_cells),
        transport_boxes_m=boxes,
        transport_mu_by_isotope=mu_by_isotope,
        transport_line_mu_by_isotope=line_mu_by_isotope,
    )


def _embedded_obstacle_grid(
    environment_payload: Mapping[str, object],
    *,
    isotopes: tuple[str, ...],
) -> ObstacleGrid | None:
    """Return local geometry from an embedded grid or neutral box list."""
    candidates: list[object] = []
    for key in ("obstacle_grid", "obstacle_layout", "obstacles"):
        if key in environment_payload and environment_payload[key] is not None:
            candidates.append(environment_payload[key])
    if "grid_shape" in environment_payload and "blocked_cells" in environment_payload:
        candidates.append(environment_payload)
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(
            "environment.json contains multiple embedded obstacle layouts."
        )
    payload = candidates[0]
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        return _axis_aligned_obstacle_grid(
            payload,
            environment_payload=environment_payload,
            isotopes=isotopes,
        )
    if not isinstance(payload, Mapping):
        raise ValueError("Embedded obstacle layout must be an object or box array.")
    return ObstacleGrid.from_dict(dict(payload))


def _resolve_local_obstacle_path(run_dir: Path, path_value: object) -> Path:
    """Resolve an obstacle layout only inside the run or local repository assets."""
    return resolve_file_backed_model_asset(
        path_value,
        field_name="obstacle_layout_path",
        run_root=run_dir,
        repository_root=_REPOSITORY_ROOT,
    )


def _resolve_obstacle_grid(
    log: MeasurementLog,
    run_dir: Path,
) -> tuple[ObstacleGrid | None, Path | None]:
    """Resolve embedded, run-local, or repository-local obstacle data."""
    embedded = _embedded_obstacle_grid(
        log.context.environment,
        isotopes=log.context.isotopes,
    )
    if embedded is not None:
        return embedded, None
    path_value = log.context.obstacle_layout_path
    if path_value is None or not str(path_value).strip():
        return None, None
    resolved = _resolve_local_obstacle_path(run_dir, path_value)
    try:
        return ObstacleGrid.load(resolved), resolved
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load obstacle layout {resolved}: {exc}") from exc


def _safe_runtime_model_payload(
    runtime_config: Mapping[str, object],
    run_dir: Path,
) -> dict[str, object]:
    """Inline the only file-backed observation-model payload using local paths."""
    payload = deepcopy(dict(runtime_config))
    configured_rate_model = payload.get("source_rate_model")
    if configured_rate_model is not None:
        _normalized_source_rate_model(
            configured_rate_model,
            origin="runtime config",
        )
    payload["source_rate_model"] = _SOURCE_RATE_MODEL

    model_path_value = payload.pop("pf_transport_response_model_path", None)
    if model_path_value is None:
        return payload
    selected = resolve_file_backed_model_asset(
        model_path_value,
        field_name="pf_transport_response_model_path",
        run_root=run_dir,
        repository_root=_REPOSITORY_ROOT,
    )
    try:
        loaded = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not load transport response model {selected}."
        ) from exc
    if isinstance(loaded, Mapping) and isinstance(
        loaded.get("pf_transport_response_model"),
        Mapping,
    ):
        loaded = loaded["pf_transport_response_model"]
    if not isinstance(loaded, Mapping):
        raise ValueError("Transport response model must be a JSON object.")
    payload["pf_transport_response_model"] = dict(loaded)
    return payload


def _resolve_mle_config(
    config: MLEConfig | Mapping[str, Any] | str | Path | None,
    batch: ObservationBatch,
) -> MLEConfig:
    """Resolve a public MLEConfig and enforce logged isotope ordering."""
    if config is None:
        mode = "count" if batch.isotope_counts is not None else "spectral"
        resolved = MLEConfig(mode=mode, isotope_names=batch.isotope_names)
    elif isinstance(config, MLEConfig):
        resolved = config
    elif isinstance(config, Mapping):
        resolved = MLEConfig.from_dict(config)
    elif isinstance(config, (str, Path)):
        path = Path(config)
        if not path.is_file():
            raise FileNotFoundError(f"MLE configuration file does not exist: {path}")
        resolved = MLEConfig.load(path)
    else:
        raise TypeError("config must be MLEConfig, a mapping, a file path, or None.")
    if tuple(resolved.isotope_names) != tuple(batch.isotope_names):
        raise ValueError(
            "MLEConfig isotope_names must exactly match the measurement-log isotope order."
        )
    return resolved


def prepare_replay(
    run_dir: str | Path,
    *,
    config: MLEConfig | Mapping[str, Any] | str | Path | None = None,
    config_source_sha256: str | None = None,
) -> ReplayContext:
    """Load a measurement log and construct its fully local forward model."""
    resolved_run_dir = Path(run_dir).resolve()
    log = load_measurement_log(resolved_run_dir)
    _normalized_source_rate_model(
        log.context.source_rate_model,
        origin="measurement log",
    )
    batch = observation_batch_from_log(log)
    mle_config = _resolve_mle_config(config, batch)
    if config_source_sha256 is not None:
        config_sha256 = str(config_source_sha256).lower()
        if len(config_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in config_sha256
        ):
            raise ValueError("config_source_sha256 must be a lowercase SHA-256 digest.")
    elif isinstance(config, (str, Path)):
        config_sha256 = sha256(Path(config).read_bytes()).hexdigest()
    else:
        config_sha256 = sha256(canonical_json_bytes(mle_config.to_dict())).hexdigest()
    resolved_estimator_config_sha256 = canonical_json_sha256(mle_config.to_dict())
    environment = _resolve_environment_config(log)
    obstacle_grid, obstacle_path = _resolve_obstacle_grid(log, resolved_run_dir)
    runtime_config = _safe_runtime_model_payload(
        log.context.runtime_config,
        resolved_run_dir,
    )
    observation_model = build_runtime_observation_model(
        runtime_config,
        isotopes=batch.isotope_names,
    )
    kernel = continuous_kernel_from_observation_model(
        observation_model,
        obstacle_grid=obstacle_grid,
        use_gpu=bool(mle_config.use_gpu),
    )
    return ReplayContext(
        run_dir=resolved_run_dir,
        log=log,
        batch=batch,
        config=mle_config,
        environment=environment,
        obstacle_grid=obstacle_grid,
        resolved_obstacle_path=obstacle_path,
        observation_model=observation_model,
        kernel=kernel,
        config_sha256=config_sha256,
        resolved_estimator_config_sha256=resolved_estimator_config_sha256,
    )


def _available_reporting_hook() -> ReplaySaveHook | None:
    """Return an optional reporting hook without requiring a reporting module."""
    try:
        module = import_module("three_d_estimation.reporting")
    except ModuleNotFoundError as exc:
        if exc.name != "three_d_estimation.reporting":
            raise
        return None
    for name in ("save_estimate", "save_mle_estimate"):
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate
    return None


def run_replay(
    run_dir: str | Path,
    *,
    config: MLEConfig | Mapping[str, Any] | str | Path | None = None,
    output_dir: str | Path | None = None,
    save_hook: ReplaySaveHook | None = None,
    config_source_sha256: str | None = None,
) -> ReplayResult:
    """Prepare, fit, and optionally persist one standalone replay result."""
    context = prepare_replay(
        run_dir,
        config=config,
        config_source_sha256=config_source_sha256,
    )
    estimate = SurfaceMLEEstimator(context.config).fit(
        context.batch,
        context.environment,
        context.kernel,
        obstacle_grid=context.obstacle_grid,
    )
    if isinstance(estimate, MLEEstimate):
        measurement_log_digest = context.log.content_sha256
        if measurement_log_digest is None:
            raise ValueError("Loaded MeasurementLog is missing its content SHA-256.")
        forward_model_manifest_sha256 = sha256(
            (context.run_dir / "forward_model_manifest.json").read_bytes()
        ).hexdigest()
        provenance = estimator_provenance(
            variant=context.config.mode,
            measurement_log_schema_version=context.log.context.schema_version,
            measurement_run_id=context.log.context.run_id,
            measurement_repository_commit=context.log.context.repository_commit,
            resolved_config_sha256=context.log.context.runtime_config_sha256,
            forward_model_manifest_sha256=forward_model_manifest_sha256,
            measurement_log_sha256=measurement_log_digest,
            config_sha256=context.config_sha256,
            resolved_estimator_config_sha256=(context.resolved_estimator_config_sha256),
        )
        estimate = replace(
            estimate,
            diagnostics={
                **estimate.diagnostics,
                "provenance": provenance,
                "estimator_family": provenance["estimator_family"],
                "estimator_variant": provenance["estimator_variant"],
                "candidate_domain": provenance["candidate_domain"],
                "uses_pf_state": provenance["uses_pf_state"],
                "uses_pf_candidates": provenance["uses_pf_candidates"],
                "measurement_run_id": context.log.context.run_id,
                "measurement_log_schema_version": context.log.context.schema_version,
            },
        )
    saved_output = None
    if output_dir is not None:
        hook = save_hook if save_hook is not None else _available_reporting_hook()
        if hook is not None:
            saved_output = hook(estimate, Path(output_dir))
    elif save_hook is not None:
        raise ValueError("output_dir is required when save_hook is provided.")
    return ReplayResult(
        estimate=estimate,
        context=context,
        saved_output=saved_output,
    )


__all__ = [
    "ReplayContext",
    "ReplayResult",
    "prepare_replay",
    "run_replay",
]
