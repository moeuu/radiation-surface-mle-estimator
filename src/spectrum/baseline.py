"""Baseline estimation utilities inspired by the ALS approach in Chapter 2."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    sparse = None
    spsolve = None
    _SCIPY_AVAILABLE = False


def baseline_als(
    y: NDArray[np.float64],
    lam: float,
    p: float,
    niter: int,
) -> NDArray[np.float64]:
    """
    Estimate the ALS baseline for Eq. 2.22-2.23 using a second-difference penalty.

    Args:
        y: Input spectrum values.
        lam: Smoothness weight for the second-difference penalty.
        p: Asymmetry parameter (0 < p < 1).
        niter: Number of reweighting iterations.

    Returns:
        Baseline estimate with the same length as y.
    """
    y = np.asarray(y, dtype=float)
    length = int(y.size)
    if length == 0:
        return np.asarray(y, dtype=float)
    if length < 3:
        return y.copy()
    w = np.ones(length, dtype=float)
    if _SCIPY_AVAILABLE and sparse is not None and spsolve is not None:
        D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(length - 2, length), format="csc")
        DTD = D.T @ D
        for _ in range(int(niter)):
            W = sparse.diags(w, 0, shape=(length, length), format="csc")
            Z = W + lam * DTD
            b = spsolve(Z, w * y)
            w = np.where(y > b, p, 1.0 - p)
        return np.asarray(b, dtype=float)
    D = np.zeros((length - 2, length), dtype=float)
    idx = np.arange(length - 2)
    D[idx, idx] = 1.0
    D[idx, idx + 1] = -2.0
    D[idx, idx + 2] = 1.0
    DTD = D.T @ D
    for _ in range(int(niter)):
        Z = np.diag(w) + lam * DTD
        b = np.linalg.solve(Z, w * y)
        w = np.where(y > b, p, 1.0 - p)
    return np.asarray(b, dtype=float)


def als_baseline(
    y: NDArray[np.float64],
    lam: float,
    p: float,
    niter: int,
) -> NDArray[np.float64]:
    """Backward-compatible alias for baseline_als."""
    return baseline_als(y, lam=lam, p=p, niter=niter)


def asymmetric_least_squares(
    signal: NDArray[np.float64],
    lam: float = 1e5,
    p: float = 0.01,
    niter: int = 10,
) -> NDArray[np.float64]:
    """
    Estimate a smooth baseline using asymmetric least squares.

    Args:
        signal: Input spectrum.
        lam: Smoothness parameter (larger = smoother).
        p: Asymmetry parameter (small values force baseline below data).
        niter: Number of reweighting iterations.

    Returns:
        Estimated baseline array.
    """
    return baseline_als(signal, lam=lam, p=p, niter=niter)
