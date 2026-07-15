"""Standalone orchestration for replaying versioned measurement logs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module
import json
import math
from pathlib import Path
from typing import Any

from measurement.continuous_kernels import ContinuousKernel
from measurement.model import EnvironmentConfig
from measurement.observation_model import (
    RuntimeObservationModel,
    build_runtime_observation_model,
    continuous_kernel_from_observation_model,
)
from measurement.obstacles import ObstacleGrid
from runtime.measurement_log import MeasurementLog, load_measurement_log

from .config import MLEConfig
from .estimator import SurfaceMLEEstimator
from .observation_batch import observation_batch_from_log
from .types import MLEEstimate, ObservationBatch


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_OBSTACLE_ROOT = (_REPOSITORY_ROOT / "obstacle_layouts").resolve()
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
            f"{origin} source_rate_model must be {_SOURCE_RATE_MODEL!r}; "
            f"got {value!r}."
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


def _environment_dimensions(payload: Mapping[str, object]) -> tuple[float, float, float] | None:
    """Extract room dimensions from one environment/runtime mapping."""
    if all(key in payload for key in ("size_x", "size_y", "size_z")):
        return _sequence_of_floats(
            [payload["size_x"], payload["size_y"], payload["size_z"]],
            name="environment size_x/size_y/size_z",
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
    mappings = _environment_mappings(log.context.environment, log.context.runtime_config)
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


def _embedded_obstacle_grid(environment_payload: Mapping[str, object]) -> ObstacleGrid | None:
    """Return an obstacle grid embedded directly in environment.json, if present."""
    candidates: list[object] = []
    for key in ("obstacle_grid", "obstacle_layout", "obstacles"):
        if key in environment_payload and environment_payload[key] is not None:
            candidates.append(environment_payload[key])
    if "grid_shape" in environment_payload and "blocked_cells" in environment_payload:
        candidates.append(environment_payload)
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError("environment.json contains multiple embedded obstacle layouts.")
    payload = candidates[0]
    if not isinstance(payload, Mapping):
        raise ValueError("Embedded obstacle layout must be a JSON object.")
    return ObstacleGrid.from_dict(dict(payload))


def _is_within(path: Path, root: Path) -> bool:
    """Return whether a resolved path is contained by a resolved root."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_relative_path(path_value: object, *, field_name: str) -> Path:
    """Reject absolute paths and parent traversal from persisted logs."""
    path = Path(str(path_value))
    if path.is_absolute():
        raise ValueError(
            f"{field_name} must be relative; absolute paths are forbidden in standalone replay."
        )
    if not path.parts or any(part == ".." for part in path.parts):
        raise ValueError(f"{field_name} must not contain parent-directory traversal.")
    return path


def _resolve_local_obstacle_path(run_dir: Path, path_value: object) -> Path:
    """Resolve an obstacle layout only inside the run or local repository assets."""
    relative = _safe_relative_path(path_value, field_name="obstacle_layout_path")
    resolved_run_dir = run_dir.resolve()
    run_candidate = (resolved_run_dir / relative).resolve()
    if not _is_within(run_candidate, resolved_run_dir):
        raise ValueError("obstacle_layout_path escapes the measurement run directory.")
    if run_candidate.is_file():
        return run_candidate

    relative_parts = relative.parts
    if relative_parts and relative_parts[0] == "obstacle_layouts":
        repository_relative = Path(*relative_parts[1:])
    else:
        repository_relative = relative
    repository_candidate = (_LOCAL_OBSTACLE_ROOT / repository_relative).resolve()
    if not _is_within(repository_candidate, _LOCAL_OBSTACLE_ROOT):
        raise ValueError("obstacle_layout_path escapes the local obstacle_layouts directory.")
    if repository_candidate.is_file():
        return repository_candidate
    raise FileNotFoundError(
        "Obstacle layout was not found in the measurement run or this repository: "
        f"{relative.as_posix()!r}. Checked {run_candidate} and {repository_candidate}."
    )


def _resolve_obstacle_grid(
    log: MeasurementLog,
    run_dir: Path,
) -> tuple[ObstacleGrid | None, Path | None]:
    """Resolve embedded, run-local, or repository-local obstacle data."""
    embedded = _embedded_obstacle_grid(log.context.environment)
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
    relative = _safe_relative_path(
        model_path_value,
        field_name="pf_transport_response_model_path",
    )
    resolved_run_dir = run_dir.resolve()
    candidates = (
        (resolved_run_dir / relative).resolve(),
        (_REPOSITORY_ROOT / relative).resolve(),
    )
    allowed_roots = (resolved_run_dir, _REPOSITORY_ROOT)
    selected = None
    for candidate, allowed_root in zip(candidates, allowed_roots, strict=True):
        if _is_within(candidate, allowed_root) and candidate.is_file():
            selected = candidate
            break
    if selected is None:
        raise FileNotFoundError(
            "pf_transport_response_model_path was not found in the run or local repository: "
            f"{relative.as_posix()!r}."
        )
    try:
        loaded = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load transport response model {selected}.") from exc
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
) -> ReplayResult:
    """Prepare, fit, and optionally persist one standalone replay result."""
    context = prepare_replay(run_dir, config=config)
    estimate = SurfaceMLEEstimator(context.config).fit(
        context.batch,
        context.environment,
        context.kernel,
        obstacle_grid=context.obstacle_grid,
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
