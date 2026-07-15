"""Dead-time correction helpers using the non-paralyzable model."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def non_paralyzable_correction(measured_counts: NDArray[np.float64], dead_time_s: float) -> NDArray[np.float64]:
    """
    Apply non-paralyzable dead-time correction to per-bin counts.

    Args:
        measured_counts: Measured counts per bin.
        dead_time_s: Dead time in seconds.

    Returns:
        Dead-time-corrected counts per bin.
    """
    total_measured = measured_counts.sum()
    if total_measured == 0 or dead_time_s <= 0:
        return measured_counts
    m_rate = total_measured  # assuming 1 s acquisition unless rescaled externally
    scale = 1.0 / max(1.0 - m_rate * dead_time_s, 1e-9)
    return measured_counts * scale
