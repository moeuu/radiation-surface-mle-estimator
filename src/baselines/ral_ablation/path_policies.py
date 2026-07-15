"""Path-planning ablation policies for RA-L comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BaselinePathSelection:
    """Represent a baseline path-policy selection from candidate poses."""

    name: str
    next_pose: NDArray[np.float64]
    candidate_index: int
    score: float


def _policy_name(policy_config: Mapping[str, Any] | str | None) -> str:
    """Return a normalized baseline path-policy name."""
    if policy_config is None:
        return ""
    if isinstance(policy_config, str):
        return policy_config.strip().lower()
    return str(policy_config.get("name", "")).strip().lower()


def _serpentine_target(
    *,
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]],
    visited_count: int,
    row_count: int,
) -> NDArray[np.float64]:
    """Return the next nominal waypoint of a floor-plane serpentine path."""
    lo, hi = bounds_xyz
    rows = max(1, int(row_count))
    row = min(rows - 1, max(0, int(visited_count)))
    y_values = np.linspace(float(lo[1]), float(hi[1]), rows)
    x = float(hi[0] if row % 2 else lo[0])
    return np.asarray([x, float(y_values[row]), float(lo[2])], dtype=float)


def select_baseline_next_pose(
    policy_config: Mapping[str, Any] | str | None,
    *,
    candidate_poses_xyz: NDArray[np.float64],
    current_pose_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> BaselinePathSelection | None:
    """Select the next pose with a baseline path policy."""
    policy = _policy_name(policy_config)
    if policy in {"", "none", "proposed", "dss_pp", "one_step"}:
        return None
    candidates = np.asarray(candidate_poses_xyz, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 3 or candidates.shape[0] == 0:
        raise ValueError("candidate_poses_xyz must be a non-empty (N, 3) array.")
    current = np.asarray(current_pose_xyz, dtype=float).reshape(3)
    visited_count = 0 if visited_poses_xyz is None else int(len(visited_poses_xyz))
    if policy in {"serpentine", "passive_serpentine", "coverage_serpentine"}:
        row_count = 6
        if isinstance(policy_config, Mapping):
            row_count = int(policy_config.get("row_count", row_count))
        target = _serpentine_target(
            bounds_xyz=bounds_xyz,
            visited_count=visited_count,
            row_count=row_count,
        )
        distances = np.linalg.norm(candidates - target[None, :], axis=1)
        idx = int(np.argmin(distances))
        return BaselinePathSelection(
            name="passive_serpentine",
            next_pose=candidates[idx].astype(float, copy=True),
            candidate_index=idx,
            score=-float(distances[idx]),
        )
    if policy in {"nearest_frontier", "greedy_coverage"}:
        distances = np.linalg.norm(candidates - current[None, :], axis=1)
        idx = int(np.argmax(distances))
        return BaselinePathSelection(
            name="nearest_frontier",
            next_pose=candidates[idx].astype(float, copy=True),
            candidate_index=idx,
            score=float(distances[idx]),
        )
    raise ValueError(f"Unknown baseline_path_policy: {policy}")
