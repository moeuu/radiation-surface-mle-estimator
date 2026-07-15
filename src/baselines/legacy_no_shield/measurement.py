"""Measurement record for the baseline particle filter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from numpy.typing import NDArray


@dataclass
class BaselineMeasurement:
    """Hold counts and metadata for a single baseline measurement."""

    counts_by_isotope: Dict[str, float]
    live_time_s: float
    detector_position: NDArray[np.float64]
    pose_idx: int
    RFe: NDArray[np.float64]
    RPb: NDArray[np.float64]
