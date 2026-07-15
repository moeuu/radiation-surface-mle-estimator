"""Small NNLS solver with SciPy fallback for peak stripping."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    from scipy.optimize import nnls as _scipy_nnls

    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _scipy_nnls = None
    _SCIPY_AVAILABLE = False


def nnls_solve(
    design: NDArray[np.float64],
    observations: NDArray[np.float64],
    *,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> NDArray[np.float64]:
    """
    Solve min ||Ax - b||_2 with x >= 0 for small systems.
    """
    A = np.asarray(design, dtype=float)
    b = np.asarray(observations, dtype=float)
    if A.size == 0 or b.size == 0:
        return np.zeros(A.shape[1] if A.ndim == 2 else 0, dtype=float)
    if _SCIPY_AVAILABLE and _scipy_nnls is not None:
        x, _ = _scipy_nnls(A, b)
        return np.asarray(x, dtype=float)
    n_cols = A.shape[1]
    x = np.zeros(n_cols, dtype=float)
    if np.allclose(A, 0.0):
        return x
    norm = float(np.linalg.norm(A, ord=2))
    step = 1.0 / max(norm * norm, 1e-12)
    for _ in range(int(max_iter)):
        grad = A.T @ (A @ x - b)
        x_next = np.maximum(x - step * grad, 0.0)
        if np.linalg.norm(x_next - x) < float(tol):
            x = x_next
            break
        x = x_next
    return np.asarray(x, dtype=float)
