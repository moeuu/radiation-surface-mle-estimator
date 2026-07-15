"""Shared protocol objects for simulator backends."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any


@dataclass(frozen=True)
class SimulationCommand:
    """Describe a single simulator step request."""

    step_id: int
    target_pose_xyz: tuple[float, float, float]
    target_base_yaw_rad: float
    fe_orientation_index: int
    pb_orientation_index: int
    dwell_time_s: float
    travel_time_s: float = 0.0
    shield_actuation_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the command."""
        return {
            "step_id": int(self.step_id),
            "target_pose_xyz": [float(v) for v in self.target_pose_xyz],
            "target_base_yaw_rad": float(self.target_base_yaw_rad),
            "fe_orientation_index": int(self.fe_orientation_index),
            "pb_orientation_index": int(self.pb_orientation_index),
            "dwell_time_s": float(self.dwell_time_s),
            "travel_time_s": float(self.travel_time_s),
            "shield_actuation_time_s": float(self.shield_actuation_time_s),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulationCommand":
        """Build a command from a dictionary payload."""
        pose = tuple(float(v) for v in data["target_pose_xyz"])
        if len(pose) != 3:
            raise ValueError("target_pose_xyz must have exactly three coordinates.")
        return cls(
            step_id=int(data["step_id"]),
            target_pose_xyz=pose,
            target_base_yaw_rad=float(data["target_base_yaw_rad"]),
            fe_orientation_index=int(data["fe_orientation_index"]),
            pb_orientation_index=int(data["pb_orientation_index"]),
            dwell_time_s=float(data["dwell_time_s"]),
            travel_time_s=float(data.get("travel_time_s", 0.0)),
            shield_actuation_time_s=float(data.get("shield_actuation_time_s", 0.0)),
        )


@dataclass(frozen=True)
class SimulationObservation:
    """Carry simulator output back to the estimator process."""

    step_id: int
    detector_pose_xyz: tuple[float, float, float]
    detector_quat_wxyz: tuple[float, float, float, float]
    fe_orientation_index: int
    pb_orientation_index: int
    spectrum_counts: list[float]
    energy_bin_edges_keV: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the observation."""
        return {
            "step_id": int(self.step_id),
            "detector_pose_xyz": [float(v) for v in self.detector_pose_xyz],
            "detector_quat_wxyz": [float(v) for v in self.detector_quat_wxyz],
            "fe_orientation_index": int(self.fe_orientation_index),
            "pb_orientation_index": int(self.pb_orientation_index),
            "spectrum_counts": [float(v) for v in self.spectrum_counts],
            "energy_bin_edges_keV": [float(v) for v in self.energy_bin_edges_keV],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulationObservation":
        """Build an observation from a dictionary payload."""
        pose = tuple(float(v) for v in data["detector_pose_xyz"])
        quat = tuple(float(v) for v in data["detector_quat_wxyz"])
        if len(pose) != 3:
            raise ValueError("detector_pose_xyz must have exactly three coordinates.")
        if len(quat) != 4:
            raise ValueError("detector_quat_wxyz must have exactly four coordinates.")
        return cls(
            step_id=int(data["step_id"]),
            detector_pose_xyz=pose,
            detector_quat_wxyz=quat,
            fe_orientation_index=int(data["fe_orientation_index"]),
            pb_orientation_index=int(data["pb_orientation_index"]),
            spectrum_counts=[float(v) for v in data["spectrum_counts"]],
            energy_bin_edges_keV=[float(v) for v in data["energy_bin_edges_keV"]],
            metadata=dict(data.get("metadata", {})),
        )


def encode_message(message_type: str, payload: dict[str, Any]) -> bytes:
    """Encode a message envelope as newline-delimited JSON bytes."""
    envelope = {"type": message_type, "payload": payload}
    return (json.dumps(envelope, sort_keys=True) + "\n").encode("utf-8")


def decode_message(data: bytes) -> tuple[str, dict[str, Any]]:
    """Decode a message envelope from newline-delimited JSON bytes."""
    envelope = json.loads(data.decode("utf-8"))
    return str(envelope["type"]), dict(envelope["payload"])
