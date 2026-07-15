"""Baseline particle filter that estimates sources without shielding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from pf.state import IsotopeState


@dataclass
class BaselinePFConfig:
    """Configuration for the baseline particle filter."""

    num_particles: int = 500
    resample_threshold: float = 0.5
    position_sigma: float = 0.5
    strength_sigma: float = 0.4
    min_strength: float = 1e-3
    position_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    position_max: tuple[float, float, float] = (10.0, 10.0, 10.0)
    init_strength_log_mean: float = 9.0
    init_strength_log_sigma: float = 1.0


@dataclass
class BaselineParticle:
    """Continuous particle with a single source hypothesis."""

    state: IsotopeState
    log_weight: float


def _log_weight_update_poisson(
    log_w_prev: NDArray[np.float64],
    z_obs: float,
    lambda_exp: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return normalized log-weights for a Poisson observation."""
    ll = z_obs * np.log(lambda_exp + 1e-12) - lambda_exp
    log_w = log_w_prev + ll
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    total = float(np.sum(w))
    if total <= 0.0:
        w = np.ones_like(w) / max(len(w), 1)
    else:
        w = w / total
    return np.log(w + 1e-20)


def _effective_sample_size(log_w: NDArray[np.float64]) -> float:
    """Return the effective sample size for the given log-weights."""
    w = np.exp(log_w)
    denom = float(np.sum(w**2))
    if denom <= 0.0:
        return 0.0
    return float(1.0 / denom)


def _systematic_resample(
    weights: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Return resample indices using the systematic resampling scheme."""
    n = int(len(weights))
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    positions = (np.arange(n) + rng.random()) / n
    cumulative_sum = np.cumsum(weights)
    indices = np.zeros(n, dtype=np.int64)
    i, j = 0, 0
    while i < n:
        if positions[i] < cumulative_sum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1
    return indices


class BaselineIsotopeFilter:
    """Single-isotope particle filter with inverse-square observations."""

    def __init__(
        self,
        isotope: str,
        config: BaselinePFConfig,
        rng: np.random.Generator,
    ) -> None:
        """Initialize particles from broad priors."""
        self.isotope = isotope
        self.config = config
        self.rng = rng
        self.continuous_particles: list[BaselineParticle] = []
        self._init_particles()

    def _init_particles(self) -> None:
        """Draw initial particles for a single-source hypothesis."""
        lo = np.array(self.config.position_min, dtype=float)
        hi = np.array(self.config.position_max, dtype=float)
        n = int(self.config.num_particles)
        self.continuous_particles = []
        if n <= 0:
            return
        for _ in range(n):
            pos = lo + self.rng.random(3) * (hi - lo)
            strength = float(
                self.rng.lognormal(
                    mean=self.config.init_strength_log_mean,
                    sigma=self.config.init_strength_log_sigma,
                )
            )
            strength = max(strength, self.config.min_strength)
            state = IsotopeState(
                num_sources=1,
                positions=pos.reshape(1, 3),
                strengths=np.array([strength], dtype=float),
                background=0.0,
            )
            self.continuous_particles.append(
                BaselineParticle(state=state, log_weight=float(np.log(1.0 / n)))
            )

    @property
    def continuous_weights(self) -> NDArray[np.float64]:
        """Return normalized particle weights."""
        if not self.continuous_particles:
            return np.zeros(0, dtype=float)
        log_w = np.array([p.log_weight for p in self.continuous_particles], dtype=float)
        log_w -= np.max(log_w)
        w = np.exp(log_w)
        total = float(np.sum(w))
        if total <= 0.0:
            return np.ones_like(w) / len(w)
        return w / total

    def _jitter_particles(self) -> None:
        """Apply Gaussian jitter to positions and log-normal jitter to strengths."""
        lo = np.array(self.config.position_min, dtype=float)
        hi = np.array(self.config.position_max, dtype=float)
        for particle in self.continuous_particles:
            pos = particle.state.positions[0]
            pos = pos + self.rng.normal(scale=self.config.position_sigma, size=3)
            pos = np.clip(pos, lo, hi)
            strength = particle.state.strengths[0]
            strength = strength * float(
                np.exp(self.rng.normal(scale=self.config.strength_sigma))
            )
            strength = max(float(strength), self.config.min_strength)
            particle.state.positions[0] = pos
            particle.state.strengths[0] = strength

    def _expected_counts(
        self, detector_pos: NDArray[np.float64], live_time_s: float
    ) -> NDArray[np.float64]:
        """Compute expected counts for each particle."""
        lam = np.zeros(len(self.continuous_particles), dtype=float)
        for i, particle in enumerate(self.continuous_particles):
            pos = particle.state.positions[0]
            strength = float(particle.state.strengths[0])
            distance = float(np.linalg.norm(detector_pos - pos))
            if distance <= 1e-6:
                distance = 1e-6
            lam[i] = live_time_s * strength / (distance**2)
        return lam

    def update(
        self,
        detector_pos: NDArray[np.float64],
        counts: float,
        live_time_s: float,
    ) -> None:
        """Update particle weights for the given measurement."""
        if not self.continuous_particles:
            return
        log_w_prev = np.array(
            [p.log_weight for p in self.continuous_particles], dtype=float
        )
        lam = self._expected_counts(detector_pos, live_time_s)
        log_w = _log_weight_update_poisson(log_w_prev, counts, lam)
        for particle, lw in zip(self.continuous_particles, log_w):
            particle.log_weight = float(lw)

    def resample_if_needed(self) -> None:
        """Resample particles when the effective sample size drops."""
        if not self.continuous_particles:
            return
        log_w = np.array([p.log_weight for p in self.continuous_particles], dtype=float)
        ess = _effective_sample_size(log_w)
        n = int(len(self.continuous_particles))
        if n <= 0:
            return
        if ess >= self.config.resample_threshold * n:
            return
        w = np.exp(log_w - np.max(log_w))
        w /= np.sum(w)
        indices = _systematic_resample(w, self.rng)
        new_particles: list[BaselineParticle] = []
        for idx in indices:
            state_copy = self.continuous_particles[int(idx)].state.copy()
            new_particles.append(
                BaselineParticle(state=state_copy, log_weight=float(np.log(1.0 / n)))
            )
        self.continuous_particles = new_particles
        self._jitter_particles()

    def step(
        self,
        detector_pos: NDArray[np.float64],
        counts: float,
        live_time_s: float,
    ) -> None:
        """Apply prediction jitter, update, and resampling for one measurement."""
        self._jitter_particles()
        self.update(detector_pos, counts, live_time_s)
        self.resample_if_needed()

    def estimate(self) -> IsotopeState:
        """Return the weighted mean estimate of the source."""
        if not self.continuous_particles:
            return IsotopeState(0, np.zeros((0, 3)), np.zeros(0), 0.0)
        weights = self.continuous_weights
        positions = np.array(
            [p.state.positions[0] for p in self.continuous_particles], dtype=float
        )
        strengths = np.array(
            [p.state.strengths[0] for p in self.continuous_particles], dtype=float
        )
        mean_pos = np.sum(weights[:, None] * positions, axis=0)
        mean_strength = float(np.sum(weights * strengths))
        return IsotopeState(
            num_sources=1,
            positions=mean_pos.reshape(1, 3),
            strengths=np.array([mean_strength], dtype=float),
            background=0.0,
        )


class BaselinePF:
    """Parallel baseline PF, one filter per isotope."""

    def __init__(
        self,
        isotopes: Sequence[str],
        config: BaselinePFConfig,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialize per-isotope filters."""
        self.rng = np.random.default_rng() if rng is None else rng
        self.filters: Dict[str, BaselineIsotopeFilter] = {
            iso: BaselineIsotopeFilter(isotope=iso, config=config, rng=self.rng)
            for iso in isotopes
        }

    def update_all(
        self,
        detector_pos: NDArray[np.float64],
        counts_by_isotope: Dict[str, float],
        live_time_s: float,
    ) -> None:
        """Update all isotope filters with the latest measurement."""
        det_pos = np.asarray(detector_pos, dtype=float)
        for iso, filt in self.filters.items():
            z = float(counts_by_isotope.get(iso, 0.0))
            filt.step(det_pos, z, live_time_s)

    def estimate_all(self) -> Dict[str, IsotopeState]:
        """Return weighted-mean estimates for all isotopes."""
        return {iso: filt.estimate() for iso, filt in self.filters.items()}

    def isotopes(self) -> Iterable[str]:
        """Return isotope names tracked by the filter."""
        return self.filters.keys()
