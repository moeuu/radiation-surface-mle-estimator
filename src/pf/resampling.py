"""Implement effective-sample-size checks and systematic resampling utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def systematic_resample(log_weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Perform systematic resampling given log-weights."""
    weights = np.exp(log_weights)
    N = len(weights)
    positions = (np.arange(N) + np.random.uniform()) / N
    cumulative_sum = np.cumsum(weights)
    indices = np.zeros(N, dtype=int)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1
    return indices


def systematic_resample_count(
    weights: NDArray[np.float64],
    *,
    count: int,
) -> NDArray[np.int64]:
    """Draw ``count`` systematic samples from normalized positive weights."""
    n_draws = max(0, int(count))
    if n_draws <= 0:
        return np.zeros(0, dtype=np.int64)
    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0:
        return np.zeros(0, dtype=np.int64)
    total = float(np.sum(w))
    if not np.isfinite(total) or total <= 0.0:
        w = np.full(w.size, 1.0 / float(w.size), dtype=np.float64)
    else:
        w = w / total
    positions = (np.arange(n_draws, dtype=np.float64) + np.random.uniform()) / float(
        n_draws
    )
    cumulative = np.cumsum(w)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="left").astype(np.int64)
