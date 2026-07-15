"""Activity estimation using non-negative least squares for peak areas or full spectra."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import nnls


def estimate_activities(
    design: NDArray[np.float64],
    observations: NDArray[np.float64],
    isotope_names: Sequence[str],
) -> Dict[str, float]:
    """
    Estimate non-negative activities from a design matrix and observations.

    Args:
        design: Matrix where columns correspond to unit responses of isotopes.
        observations: Observed counts vector.
        isotope_names: Names matching design columns.

    Returns:
        Mapping from isotope name to estimated activity.
    """
    activities, _ = nnls(design, observations)
    return {name: float(val) for name, val in zip(isotope_names, activities)}
