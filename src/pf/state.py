"""Define per-isotope particle state vectors (source count, positions, intensities, background)."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class IsotopeState:
    """
    Continuous PF state for a single isotope (Sec. 3.3.2):
        θ_h = (r_h, {s_{h,m}}, {q_{h,m}}, b_h)
    """

    num_sources: int
    positions: NDArray[np.float64]  # shape (r_h,3)
    strengths: NDArray[np.float64]  # shape (r_h,)
    background: float
    covariances: NDArray[np.float64] | None = None  # optional (r_h,4,4) across (x,y,z,q)
    ages: NDArray[np.int64] | None = None
    low_q_streaks: NDArray[np.int64] | None = None
    support_scores: NDArray[np.float64] | None = None
    tentative_sources: NDArray[np.bool_] | None = None
    verification_fail_streaks: NDArray[np.int64] | None = None

    def copy(self) -> "IsotopeState":
        """Return a deep copy of the isotope state and per-source metadata."""
        return IsotopeState(
            num_sources=int(self.num_sources),
            positions=self.positions.copy(),
            strengths=self.strengths.copy(),
            background=float(self.background),
            covariances=None if self.covariances is None else self.covariances.copy(),
            ages=None if self.ages is None else self.ages.copy(),
            low_q_streaks=None if self.low_q_streaks is None else self.low_q_streaks.copy(),
            support_scores=None if self.support_scores is None else self.support_scores.copy(),
            tentative_sources=(
                None if self.tentative_sources is None else self.tentative_sources.copy()
            ),
            verification_fail_streaks=(
                None
                if self.verification_fail_streaks is None
                else self.verification_fail_streaks.copy()
            ),
        )
