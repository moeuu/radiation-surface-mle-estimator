"""Provider-neutral forward-response conformance generation for pure MLE."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import zipfile

import numpy as np
from numpy.typing import NDArray

from measurement.observation_model import (
    build_runtime_observation_model,
    continuous_kernel_from_observation_model,
)
from measurement.obstacle_assets import material_mu_cm_inv
from measurement.obstacles import ObstacleGrid


FORWARD_CONFORMANCE_SCHEMA_VERSION = 1
EXPECTED_CASE_ORDER = (
    "isotope",
    "detector_pose",
    "fe_orientation",
    "pb_orientation",
    "source_point",
    "obstacle",
)
EXPECTED_UNITS = {
    "distance": "m",
    "live_time": "s",
    "source_strength": "detector_cps_1m",
}
_RUNTIME_CONFIG = {
    "source_rate_model": "detector_cps_1m",
    "pf_line_resolved_shield_attenuation": True,
}


@dataclass(frozen=True, slots=True)
class ForwardConformanceResult:
    """Store canonical case IDs and unit-strength expected responses."""

    case_ids: NDArray[np.str_]
    unit_response: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate aligned, finite, one-dimensional conformance arrays."""
        case_ids = np.asarray(self.case_ids, dtype=np.str_)
        responses = np.asarray(self.unit_response, dtype=np.float64)
        if case_ids.ndim != 1 or responses.shape != case_ids.shape:
            raise ValueError("case_ids and unit_response must be aligned vectors.")
        if case_ids.size == 0 or any(not str(value) for value in case_ids):
            raise ValueError("case_ids must contain non-empty IDs.")
        if len(set(str(value) for value in case_ids)) != case_ids.size:
            raise ValueError("case_ids must be unique.")
        if np.any(~np.isfinite(responses)) or np.any(responses < 0.0):
            raise ValueError("unit_response must contain finite non-negative values.")
        case_ids = np.array(case_ids, dtype=np.str_, copy=True)
        responses = np.array(responses, dtype=np.float64, copy=True)
        case_ids.setflags(write=False)
        responses.setflags(write=False)
        object.__setattr__(self, "case_ids", case_ids)
        object.__setattr__(self, "unit_response", responses)


def _mapping(value: object, *, name: str) -> dict[str, object]:
    """Return a plain mapping or raise a path-specific validation error."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object.")
    return dict(value)


def _sequence(value: object, *, name: str) -> tuple[object, ...]:
    """Return a non-empty non-string sequence."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array.")
    result = tuple(value)
    if not result:
        raise ValueError(f"{name} must not be empty.")
    return result


def _id(value: object, *, name: str) -> str:
    """Return a non-empty case-axis ID that cannot corrupt the case format."""
    result = str(value).strip()
    if not result or "|" in result:
        raise ValueError(f"{name} must be non-empty and must not contain '|'.")
    return result


def _xyz(value: object, *, name: str) -> NDArray[np.float64]:
    """Return one finite XYZ vector in metres."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain three finite coordinates.")
    return array


def _positive_float(value: object, *, name: str) -> float:
    """Return one finite, strictly positive floating-point value."""
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive.")
    return result


def _validate_unique_ids(values: Sequence[str], *, name: str) -> None:
    """Reject duplicate IDs along one conformance axis."""
    if len(set(values)) != len(values):
        raise ValueError(f"{name} IDs must be unique.")


def _validate_axes(payload: Mapping[str, object]) -> dict[str, object]:
    """Return a validated, JSON-compatible copy of canonical axes data."""
    axes = deepcopy(dict(payload))
    required_fields = {
        "schema_version",
        "fixture_id",
        "units",
        "isotopes",
        "detector_poses",
        "shield_program",
        "source_points",
        "obstacles",
        "required_case_order",
    }
    if set(axes) != required_fields:
        raise ValueError(
            f"Forward-response axes fields must be exactly {sorted(required_fields)}."
        )
    if axes.get("schema_version") != FORWARD_CONFORMANCE_SCHEMA_VERSION:
        raise ValueError("Unsupported forward-response conformance schema_version.")
    if axes.get("fixture_id") != "forward-response-conformance-v1":
        raise ValueError("fixture_id must be forward-response-conformance-v1.")
    if _mapping(axes.get("units"), name="units") != EXPECTED_UNITS:
        raise ValueError(f"units must be exactly {EXPECTED_UNITS}.")
    case_order = tuple(
        str(value)
        for value in _sequence(
            axes.get("required_case_order"),
            name="required_case_order",
        )
    )
    if case_order != EXPECTED_CASE_ORDER:
        raise ValueError(
            f"required_case_order must be exactly {list(EXPECTED_CASE_ORDER)}."
        )

    isotopes = tuple(
        _id(value, name="isotopes[]")
        for value in _sequence(axes.get("isotopes"), name="isotopes")
    )
    _validate_unique_ids(isotopes, name="isotope")

    pose_ids: list[str] = []
    for index, raw_pose in enumerate(
        _sequence(axes.get("detector_poses"), name="detector_poses")
    ):
        pose = _mapping(raw_pose, name=f"detector_poses[{index}]")
        if set(pose) != {"pose_id", "xyz", "live_time_s"}:
            raise ValueError(
                f"detector_poses[{index}] must contain pose_id, xyz, and live_time_s."
            )
        pose_ids.append(_id(pose["pose_id"], name=f"detector_poses[{index}].pose_id"))
        _xyz(pose["xyz"], name=f"detector_poses[{index}].xyz")
        _positive_float(
            pose["live_time_s"],
            name=f"detector_poses[{index}].live_time_s",
        )
    _validate_unique_ids(pose_ids, name="detector pose")

    shield = _mapping(axes.get("shield_program"), name="shield_program")
    if set(shield) != {
        "fe_orientation_indices",
        "pb_orientation_indices",
        "pairing",
    }:
        raise ValueError("shield_program contains incompatible fields.")
    expected_indices = tuple(range(8))
    if tuple(shield["fe_orientation_indices"]) != expected_indices:
        raise ValueError("Fe orientation indices must be exactly 0 through 7.")
    if tuple(shield["pb_orientation_indices"]) != expected_indices:
        raise ValueError("Pb orientation indices must be exactly 0 through 7.")
    if shield["pairing"] != "cartesian_product":
        raise ValueError("shield_program pairing must be cartesian_product.")

    source_ids: list[str] = []
    for index, raw_source in enumerate(
        _sequence(axes.get("source_points"), name="source_points")
    ):
        source = _mapping(raw_source, name=f"source_points[{index}]")
        if set(source) != {"source_id", "xyz", "surface_kind"}:
            raise ValueError(
                f"source_points[{index}] must contain source_id, xyz, and surface_kind."
            )
        source_ids.append(
            _id(source["source_id"], name=f"source_points[{index}].source_id")
        )
        _xyz(source["xyz"], name=f"source_points[{index}].xyz")
        _id(source["surface_kind"], name=f"source_points[{index}].surface_kind")
    _validate_unique_ids(source_ids, name="source point")

    obstacle_ids: list[str] = []
    for obstacle_index, raw_obstacle in enumerate(
        _sequence(axes.get("obstacles"), name="obstacles")
    ):
        obstacle = _mapping(raw_obstacle, name=f"obstacles[{obstacle_index}]")
        if set(obstacle) != {"obstacle_id", "boxes"}:
            raise ValueError(
                f"obstacles[{obstacle_index}] must contain obstacle_id and boxes."
            )
        obstacle_ids.append(
            _id(
                obstacle["obstacle_id"],
                name=f"obstacles[{obstacle_index}].obstacle_id",
            )
        )
        boxes = obstacle["boxes"]
        if isinstance(boxes, (str, bytes)) or not isinstance(boxes, Sequence):
            raise ValueError(f"obstacles[{obstacle_index}].boxes must be an array.")
        for box_index, raw_box in enumerate(boxes):
            box = _mapping(
                raw_box,
                name=f"obstacles[{obstacle_index}].boxes[{box_index}]",
            )
            if set(box) != {"min_xyz", "max_xyz", "material"}:
                raise ValueError(
                    "Obstacle boxes require min_xyz, max_xyz, and material."
                )
            lower = _xyz(box["min_xyz"], name="obstacle box min_xyz")
            upper = _xyz(box["max_xyz"], name="obstacle box max_xyz")
            if np.any(upper <= lower):
                raise ValueError(
                    "Obstacle box max_xyz must exceed min_xyz on every axis."
                )
            material = _id(box["material"], name="obstacle box material")
            for isotope in isotopes:
                material_mu_cm_inv(material, isotope)
    _validate_unique_ids(obstacle_ids, name="obstacle")
    return axes


def load_forward_conformance_axes(path: str | Path) -> dict[str, object]:
    """Load and strictly validate provider-neutral conformance axes JSON."""
    target = Path(path)
    try:
        payload: Any = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read valid conformance axes from {target}."
        ) from exc
    return _validate_axes(_mapping(payload, name="axes"))


def _obstacle_grid(
    obstacle: Mapping[str, object],
    isotopes: tuple[str, ...],
) -> ObstacleGrid | None:
    """Construct local transport boxes and attenuation from neutral box data."""
    boxes = tuple(obstacle["boxes"])
    if not boxes:
        return None
    transport_boxes: list[tuple[float, float, float, float, float, float]] = []
    materials: list[str] = []
    for raw_box in boxes:
        box = _mapping(raw_box, name="obstacle box")
        lower = _xyz(box["min_xyz"], name="obstacle box min_xyz")
        upper = _xyz(box["max_xyz"], name="obstacle box max_xyz")
        transport_boxes.append(
            tuple(float(value) for value in np.concatenate((lower, upper)))
        )
        materials.append(str(box["material"]))
    return ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(0, 0),
        blocked_cells=(),
        transport_boxes_m=tuple(transport_boxes),
        transport_mu_by_isotope={
            isotope: tuple(
                material_mu_cm_inv(material, isotope) for material in materials
            )
            for isotope in isotopes
        },
    )


def compute_forward_conformance(
    axes_or_path: Mapping[str, object] | str | Path,
) -> ForwardConformanceResult:
    """Compute every canonical unit-strength case with the local runtime model."""
    axes = (
        _validate_axes(axes_or_path)
        if isinstance(axes_or_path, Mapping)
        else load_forward_conformance_axes(axes_or_path)
    )
    isotopes = tuple(str(value) for value in axes["isotopes"])
    poses = tuple(
        _mapping(value, name="detector pose") for value in axes["detector_poses"]
    )
    sources = tuple(
        _mapping(value, name="source point") for value in axes["source_points"]
    )
    obstacles = tuple(_mapping(value, name="obstacle") for value in axes["obstacles"])
    observation_model = build_runtime_observation_model(
        _RUNTIME_CONFIG,
        isotopes=isotopes,
    )
    kernels = tuple(
        continuous_kernel_from_observation_model(
            observation_model,
            obstacle_grid=_obstacle_grid(obstacle, isotopes),
            use_gpu=False,
        )
        for obstacle in obstacles
    )

    case_ids: list[str] = []
    responses: list[float] = []
    for isotope in isotopes:
        for pose in poses:
            pose_id = _id(pose["pose_id"], name="detector pose ID")
            detector_position = _xyz(pose["xyz"], name="detector pose xyz")
            live_time_s = float(pose["live_time_s"])
            for fe_index in range(8):
                for pb_index in range(8):
                    for source in sources:
                        source_id = _id(source["source_id"], name="source point ID")
                        source_position = _xyz(source["xyz"], name="source point xyz")
                        for obstacle, kernel in zip(obstacles, kernels, strict=True):
                            obstacle_id = _id(
                                obstacle["obstacle_id"],
                                name="obstacle ID",
                            )
                            case_ids.append(
                                f"{isotope}|pose={pose_id}|fe={fe_index:02d}"
                                f"|pb={pb_index:02d}|source={source_id}"
                                f"|obstacle={obstacle_id}"
                            )
                            responses.append(
                                kernel.expected_counts_pair(
                                    isotope=isotope,
                                    detector_pos=detector_position,
                                    sources=source_position.reshape(1, 3),
                                    strengths=np.ones(1, dtype=np.float64),
                                    fe_index=fe_index,
                                    pb_index=pb_index,
                                    live_time_s=live_time_s,
                                    background=0.0,
                                )
                            )
    return ForwardConformanceResult(
        case_ids=np.asarray(case_ids, dtype=np.str_),
        unit_response=np.asarray(responses, dtype=np.float64),
    )


def _npy_bytes(array: NDArray[np.generic]) -> bytes:
    """Return deterministic, non-pickle NPY bytes."""
    buffer = BytesIO()
    np.lib.format.write_array(
        buffer,
        np.asarray(array),
        version=(2, 0),
        allow_pickle=False,
    )
    return buffer.getvalue()


def save_forward_conformance(
    output_path: str | Path,
    result: ForwardConformanceResult,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically save exact case_ids and unit_response NPZ members."""
    if not isinstance(result, ForwardConformanceResult):
        raise TypeError("result must be a ForwardConformanceResult.")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Forward conformance output exists: {target}")
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name, array in (
            ("case_ids", result.case_ids),
            ("unit_response", result.unit_response),
        ):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(array))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(buffer.getvalue())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


__all__ = [
    "EXPECTED_CASE_ORDER",
    "EXPECTED_UNITS",
    "FORWARD_CONFORMANCE_SCHEMA_VERSION",
    "ForwardConformanceResult",
    "compute_forward_conformance",
    "load_forward_conformance_axes",
    "save_forward_conformance",
]
