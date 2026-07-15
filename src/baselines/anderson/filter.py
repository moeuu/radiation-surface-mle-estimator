"""Recursive Bayesian particle filter baseline after Anderson et al."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import poisson

from baselines.anderson.kernels import AndersonAttenuationKernel
from runtime_defaults import DEFAULT_RANDOM_SOURCE_INTENSITY_CPS_1M


@dataclass(frozen=True)
class AndersonMeasurement:
    """Store one isotope-specific count measurement."""

    detector_pos: tuple[float, float, float]
    live_time_s: float
    counts: float
    elapsed_s: float | None = None


@dataclass(frozen=True)
class AndersonFilterConfig:
    """Configure one Anderson-style isotope-specific RBE particle filter."""

    num_particles: int = 1000
    num_sources: int = 1
    position_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    position_max: tuple[float, float, float] = (10.0, 20.0, 10.0)
    init_activity_log_mean: float = float(
        np.log(DEFAULT_RANDOM_SOURCE_INTENSITY_CPS_1M)
    )
    init_activity_log_sigma: float = 1.5
    min_activity_cps_1m: float = 1.0
    max_activity_cps_1m: float = 5.0e6
    background_cps: float = 0.0
    confidence_alpha: float = 1.0
    resample_ess_fraction: float = 0.5
    regularization_position_sigma_m: float = 0.05
    regularization_activity_rel_sigma: float = 0.10
    half_life_s: float | None = None
    rng_seed: int = 20260502


def poisson_interval_log_likelihood(
    counts: float,
    means: NDArray[np.float64],
    *,
    alpha: float,
    eps: float = 1.0e-300,
) -> NDArray[np.float64]:
    """
    Return Anderson et al. Poisson confidence-interval log likelihoods.

    The paper evaluates the probability of obtaining a count inside
    ``counts +/- alpha * sqrt(counts)`` rather than only the exact count.
    """
    mean_arr = np.maximum(np.asarray(means, dtype=float), 1.0e-12)
    observed = max(float(counts), 0.0)
    if float(alpha) <= 0.0:
        k = int(round(observed))
        return np.log(np.maximum(poisson.pmf(k, mean_arr), eps))
    sigma = float(np.sqrt(observed))
    lower = int(np.floor(max(0.0, observed - float(alpha) * sigma)))
    upper = int(np.ceil(max(0.0, observed + float(alpha) * sigma)))
    if upper < lower:
        upper = lower
    cdf_upper = poisson.cdf(upper, mean_arr)
    if lower <= 0:
        probability = cdf_upper
    else:
        probability = cdf_upper - poisson.cdf(lower - 1, mean_arr)
    return np.log(np.maximum(probability, eps))


def _normalize_log_weights(log_weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return normalized log weights."""
    values = np.asarray(log_weights, dtype=float)
    normalizer = float(logsumexp(values))
    if not np.isfinite(normalizer):
        n = max(int(values.size), 1)
        return np.full(values.shape, -np.log(float(n)), dtype=float)
    return values - normalizer


def _effective_sample_size_from_log_weights(log_weights: NDArray[np.float64]) -> float:
    """Return the effective sample size for normalized log weights."""
    weights = np.exp(_normalize_log_weights(log_weights))
    denom = float(np.sum(weights * weights))
    if denom <= 0.0:
        return 0.0
    return float(1.0 / denom)


class AndersonRBEParticleFilter:
    """Implement the Anderson et al. attenuation-aware RBE particle filter."""

    def __init__(
        self,
        *,
        isotope: str,
        kernel: AndersonAttenuationKernel,
        config: AndersonFilterConfig,
    ) -> None:
        """Initialize particles from uniform position and lognormal activity priors."""
        self.isotope = str(isotope)
        self.kernel = kernel
        self.config = config
        self.rng = np.random.default_rng(int(config.rng_seed))
        self.positions = np.zeros(
            (int(config.num_particles), int(config.num_sources), 3),
            dtype=float,
        )
        self.activities = np.zeros(
            (int(config.num_particles), int(config.num_sources)),
            dtype=float,
        )
        self.log_weights = np.full(
            int(config.num_particles),
            -np.log(float(max(int(config.num_particles), 1))),
            dtype=float,
        )
        self.measurements: list[AndersonMeasurement] = []
        self.resample_count = 0
        self._initialize_particles()

    @property
    def weights(self) -> NDArray[np.float64]:
        """Return normalized particle weights."""
        return np.exp(_normalize_log_weights(self.log_weights))

    def _initialize_particles(self) -> None:
        """Draw initial particles over source position and activity."""
        lo = np.asarray(self.config.position_min, dtype=float)
        hi = np.asarray(self.config.position_max, dtype=float)
        if lo.shape != (3,) or hi.shape != (3,):
            raise ValueError("position_min and position_max must be 3D vectors.")
        if np.any(hi <= lo):
            raise ValueError("position_max must be larger than position_min.")
        self.positions = lo + self.rng.random(self.positions.shape) * (hi - lo)
        activities = self.rng.lognormal(
            mean=float(self.config.init_activity_log_mean),
            sigma=max(float(self.config.init_activity_log_sigma), 1.0e-9),
            size=self.activities.shape,
        )
        self.activities = np.clip(
            activities,
            float(self.config.min_activity_cps_1m),
            float(self.config.max_activity_cps_1m),
        )

    def effective_sample_size(self) -> float:
        """Return the current effective sample size."""
        return _effective_sample_size_from_log_weights(self.log_weights)

    def predict_decay(self, elapsed_s: float) -> None:
        """Apply the Anderson short-lived-nuclide activity transition."""
        half_life = self.config.half_life_s
        if half_life is None or float(half_life) <= 0.0:
            return
        elapsed = max(float(elapsed_s), 0.0)
        decay = float(np.exp(-np.log(2.0) * elapsed / float(half_life)))
        self.activities *= decay

    def expected_counts(self, measurement: AndersonMeasurement) -> NDArray[np.float64]:
        """Return expected counts for all particles for one measurement."""
        expected = self.kernel.expected_counts_batch(
            isotope=self.isotope,
            detector_pos=measurement.detector_pos,
            sources=self.positions,
            activities=self.activities,
            live_time_s=float(measurement.live_time_s),
            background_cps=float(self.config.background_cps),
        )
        return np.maximum(expected, 1.0e-12)

    def update(self, measurement: AndersonMeasurement) -> dict[str, float | bool]:
        """Apply one Anderson RBE measurement update."""
        if measurement.elapsed_s is not None:
            self.predict_decay(float(measurement.elapsed_s))
        self.measurements.append(measurement)
        means = self.expected_counts(measurement)
        self.log_weights = _normalize_log_weights(
            self.log_weights
            + poisson_interval_log_likelihood(
                float(measurement.counts),
                means,
                alpha=float(self.config.confidence_alpha),
            )
        )
        ess_pre = self.effective_sample_size()
        resampled = False
        threshold = float(self.config.resample_ess_fraction) * float(
            self.config.num_particles
        )
        if ess_pre < threshold:
            self.resample()
            resampled = True
        return {
            "ess_pre": float(ess_pre),
            "ess_post": float(self.effective_sample_size()),
            "resampled": bool(resampled),
            "count": float(measurement.counts),
            "compute_backend": (
                "gpu"
                if bool(self.kernel.config.use_gpu)
                else (
                    "cpu-parallel"
                    if int(self.kernel.config.cpu_workers) > 1
                    else "cpu"
                )
            ),
        }

    def resample(self) -> None:
        """Perform multinomial resampling followed by Anderson regularization."""
        weights = self.weights
        n = int(self.config.num_particles)
        indices = self.rng.choice(n, size=n, replace=True, p=weights)
        self.positions = self.positions[indices].copy()
        self.activities = self.activities[indices].copy()
        self.log_weights[:] = -np.log(float(max(n, 1)))
        self.resample_count += 1
        self.regularize()

    def regularize(self) -> None:
        """Apply Gaussian position and relative activity regularization."""
        lo = np.asarray(self.config.position_min, dtype=float)
        hi = np.asarray(self.config.position_max, dtype=float)
        sigma_pos = max(float(self.config.regularization_position_sigma_m), 0.0)
        if sigma_pos > 0.0:
            self.positions += self.rng.normal(0.0, sigma_pos, size=self.positions.shape)
            self.positions = np.clip(self.positions, lo, hi)
        rel_sigma = max(float(self.config.regularization_activity_rel_sigma), 0.0)
        if rel_sigma > 0.0:
            self.activities *= np.exp(
                self.rng.normal(0.0, rel_sigma, size=self.activities.shape)
            )
            self.activities = np.clip(
                self.activities,
                float(self.config.min_activity_cps_1m),
                float(self.config.max_activity_cps_1m),
            )

    def map_state(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return source positions and activities for the MAP particle."""
        best = int(np.argmax(self.log_weights))
        return self.positions[best].copy(), self.activities[best].copy()

    def estimate(
        self,
        *,
        method: str = "weighted_mean",
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the current source estimate."""
        method_norm = str(method).strip().lower()
        if method_norm == "map":
            return self.map_state()
        weights = self.weights
        positions = np.sum(weights[:, None, None] * self.positions, axis=0)
        activities = np.sum(weights[:, None] * self.activities, axis=0)
        return positions, activities

    def estimates_for_metrics(
        self,
        *,
        method: str = "weighted_mean",
    ) -> list[dict[str, object]]:
        """Return estimates in the repository metric format."""
        positions, activities = self.estimate(method=method)
        return [
            {
                "position": [float(value) for value in position],
                "strength": float(activity),
            }
            for position, activity in zip(positions, activities)
        ]

    def set_particles_for_tests(
        self,
        *,
        positions: NDArray[np.float64],
        activities: NDArray[np.float64],
        log_weights: Sequence[float] | None = None,
    ) -> None:
        """Set particles directly for deterministic unit tests."""
        pos = np.asarray(positions, dtype=float)
        act = np.asarray(activities, dtype=float)
        if pos.shape != self.positions.shape:
            raise ValueError("positions shape does not match filter particles.")
        if act.shape != self.activities.shape:
            raise ValueError("activities shape does not match filter particles.")
        self.positions = pos.copy()
        self.activities = act.copy()
        if log_weights is None:
            n = int(self.config.num_particles)
            self.log_weights = np.full(n, -np.log(float(n)), dtype=float)
        else:
            lw = np.asarray(tuple(float(value) for value in log_weights), dtype=float)
            if lw.shape != self.log_weights.shape:
                raise ValueError("log_weights shape does not match filter particles.")
            self.log_weights = _normalize_log_weights(lw)
