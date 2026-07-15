"""Shared helpers for PF report construction and model-order diagnostics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def measurement_vector(
    values: float | NDArray[np.float64],
    count: int,
    name: str,
    *,
    min_value: float | None = None,
    allow_scalar: bool = True,
) -> NDArray[np.float64]:
    """Return a validated one-value-per-measurement vector."""
    expected = max(int(count), 0)
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        if expected == 0:
            return np.zeros(0, dtype=float)
        raise ValueError(f"{name} must contain one value per measurement.")
    if arr.size == 1 and expected != 1 and allow_scalar:
        arr = np.full(expected, float(arr[0]), dtype=float)
    elif arr.size != expected:
        scalar_text = "scalar or " if allow_scalar else ""
        raise ValueError(f"{name} must be {scalar_text}one value per measurement.")
    if min_value is not None:
        arr = np.maximum(arr, float(min_value))
    return np.asarray(arr, dtype=float)


def dedupe_report_candidates(
    positions: NDArray[np.float64],
    strengths: NDArray[np.float64],
    *,
    radius_m: float,
    max_candidates: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return report candidates after deterministic radius de-duplication."""
    pos_arr = np.asarray(positions, dtype=float).reshape(-1, 3)
    q_arr = np.asarray(strengths, dtype=float).reshape(-1)
    if pos_arr.shape[0] == 0 or q_arr.size == 0:
        return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
    if q_arr.size != pos_arr.shape[0]:
        raise ValueError("strengths must have one value per report candidate.")
    finite = np.all(np.isfinite(pos_arr), axis=1) & np.isfinite(q_arr)
    pos_arr = pos_arr[finite]
    q_arr = np.maximum(q_arr[finite], 0.0)
    limit = max(1, int(max_candidates))
    radius = max(float(radius_m), 0.0)
    kept_pos: list[NDArray[np.float64]] = []
    kept_q: list[float] = []
    for pos, strength in zip(pos_arr, q_arr):
        if len(kept_pos) >= limit:
            break
        if radius > 0.0 and kept_pos:
            distances = np.linalg.norm(np.vstack(kept_pos) - pos[None, :], axis=1)
            if np.any(distances <= radius):
                continue
        kept_pos.append(np.asarray(pos, dtype=float))
        kept_q.append(max(float(strength), 0.0))
    if not kept_pos:
        return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
    return np.vstack(kept_pos), np.asarray(kept_q, dtype=float)
