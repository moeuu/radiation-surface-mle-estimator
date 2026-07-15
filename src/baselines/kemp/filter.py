"""Parallel log-domain DDPF components for the Kemp comparison baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import nnls
from scipy.special import logsumexp
from scipy.spatial import cKDTree

from baselines.kemp.kernels import DiscreteAttenuationKernel
from runtime_defaults import (
    DEFAULT_MAX_SOURCES_PER_ISOTOPE,
    DEFAULT_RANDOM_SOURCE_INTENSITY_CPS_1M,
)


@dataclass(frozen=True)
class KempMeasurement:
    """Store one isotope-specific count observation."""

    detector_pos: tuple[float, float, float]
    live_time_s: float
    counts: float
    variance: float
    fe_index: int = 0
    pb_index: int = 0


@dataclass(frozen=True)
class KempFilterConfig:
    """Configure one isotope-specific Kemp log-domain DDPF."""

    num_particles: int = 2000
    max_sources: int = DEFAULT_MAX_SOURCES_PER_ISOTOPE
    init_source_count_min: int = 1
    init_source_count_max: int = DEFAULT_MAX_SOURCES_PER_ISOTOPE
    init_strength_log_mean: float = float(
        np.log(DEFAULT_RANDOM_SOURCE_INTENSITY_CPS_1M)
    )
    init_strength_log_sigma: float = 1.5
    min_strength_cps_1m: float = 5.0
    max_strength_cps_1m: float = 5.0e6
    background_cps: float = 0.0
    resample_ess_fraction: float = 0.5
    position_jitter_m: float = 0.5
    strength_jitter_log_sigma: float = 0.25
    p_birth: float = 0.04
    p_death: float = 0.03
    p_move: float = 0.25
    refit_final_strengths: bool = True
    estimate_min_grid_probability: float = 0.02
    estimate_merge_radius_m: float = 0.75
    use_gpu: bool = False
    gpu_device: str = "cuda"
    gpu_dtype: str = "float64"
    rng_seed: int = 123


def _systematic_resample(
    weights: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Return systematic-resampling indices for normalized weights."""
    n = int(weights.size)
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    positions = (float(rng.random()) + np.arange(n, dtype=float)) / float(n)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="left").astype(np.int64)


def _merge_duplicate_sources(
    indices: NDArray[np.int64],
    strengths: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Merge duplicate discrete source indices by summing strengths."""
    merged: dict[int, float] = {}
    for index, strength in zip(indices, strengths):
        idx = int(index)
        val = float(strength)
        if idx < 0 or val <= 0.0:
            continue
        merged[idx] = merged.get(idx, 0.0) + val
    if not merged:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=float)
    keys = np.asarray(sorted(merged), dtype=np.int64)
    vals = np.asarray([merged[int(key)] for key in keys], dtype=float)
    return keys, vals


def _torch_config(device_name: str, dtype_name: str) -> tuple[object, object, object]:
    """Return torch, device, and dtype for an explicit GPU update."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Kemp GPU update requires torch.") from exc
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Kemp GPU update requested CUDA, but CUDA is unavailable.")
    device = torch.device(str(device_name))
    if str(dtype_name) == "float32":
        dtype = torch.float32
    elif str(dtype_name) == "float64":
        dtype = torch.float64
    else:
        raise ValueError("gpu_dtype must be 'float32' or 'float64'.")
    return torch, device, dtype


class KempLogDDPF:
    """Implement one isotope-specific log-domain dynamic discrete PF."""

    def __init__(
        self,
        *,
        isotope: str,
        kernel: DiscreteAttenuationKernel,
        config: KempFilterConfig,
    ) -> None:
        """Initialize particles over source cardinality, grid index, and strength."""
        self.isotope = str(isotope)
        self.kernel = kernel
        self.config = config
        self.rng = np.random.default_rng(int(config.rng_seed))
        self.grid_tree = cKDTree(np.asarray(self.kernel.source_grid, dtype=float))
        self.source_indices = np.full(
            (int(config.num_particles), int(config.max_sources)),
            -1,
            dtype=np.int64,
        )
        self.strengths = np.zeros_like(self.source_indices, dtype=float)
        self.log_weights = np.full(
            int(config.num_particles),
            -np.log(config.num_particles),
            dtype=float,
        )
        self.measurements: list[KempMeasurement] = []
        self.resample_count = 0
        self._initialize_particles()

    @property
    def weights(self) -> NDArray[np.float64]:
        """Return normalized particle weights."""
        return np.exp(self.log_weights)

    def _initialize_particles(self) -> None:
        """Draw the initial particle population."""
        cfg = self.config
        n = int(cfg.num_particles)
        max_sources = int(cfg.max_sources)
        r_min = max(1, int(cfg.init_source_count_min))
        r_max = min(max_sources, max(r_min, int(cfg.init_source_count_max)))
        for particle in range(n):
            count = int(self.rng.integers(r_min, r_max + 1))
            indices = self.rng.choice(self.kernel.num_sources, size=count, replace=False)
            strengths = self.rng.lognormal(
                mean=float(cfg.init_strength_log_mean),
                sigma=max(float(cfg.init_strength_log_sigma), 1.0e-6),
                size=count,
            )
            strengths = np.clip(
                strengths,
                float(cfg.min_strength_cps_1m),
                float(cfg.max_strength_cps_1m),
            )
            self.source_indices[particle, :count] = indices
            self.strengths[particle, :count] = strengths

    def effective_sample_size(self) -> float:
        """Return the current effective sample size."""
        w = self.weights
        return float(1.0 / max(float(np.sum(w * w)), 1.0e-300))

    def _particle_rates_cpu(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return expected count rates for all particles on the CPU path."""
        clipped = np.clip(self.source_indices, 0, self.kernel.num_sources - 1)
        active = self.source_indices >= 0
        theta_particles = theta[clipped] * active
        rate = np.sum(theta_particles * self.strengths, axis=1)
        rate += max(float(self.config.background_cps), 0.0)
        return np.asarray(rate, dtype=float)

    def _particle_rates_gpu(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return expected count rates for all particles on the GPU path."""
        torch, device, dtype = _torch_config(
            str(self.config.gpu_device),
            str(self.config.gpu_dtype),
        )
        theta_t = torch.as_tensor(theta, device=device, dtype=dtype)
        clipped = np.clip(self.source_indices, 0, self.kernel.num_sources - 1)
        clipped_t = torch.as_tensor(clipped, device=device, dtype=torch.long)
        active_t = torch.as_tensor(
            self.source_indices >= 0,
            device=device,
            dtype=dtype,
        )
        strength_t = torch.as_tensor(self.strengths, device=device, dtype=dtype)
        rate = torch.sum(theta_t[clipped_t] * active_t * strength_t, dim=1)
        rate = rate + max(float(self.config.background_cps), 0.0)
        return rate.detach().cpu().numpy()

    def _particle_rates(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return particle count rates using the configured compute backend."""
        if bool(self.config.use_gpu):
            return self._particle_rates_gpu(theta)
        return self._particle_rates_cpu(theta)

    def update(self, measurement: KempMeasurement) -> dict[str, float | bool]:
        """Apply a Poisson log-domain count update for one measurement."""
        self.measurements.append(measurement)
        theta = self.kernel.kernel_vector(
            self.isotope,
            measurement.detector_pos,
            measurement.fe_index,
            measurement.pb_index,
        )
        rate = self._particle_rates(theta)
        lam = np.maximum(float(measurement.live_time_s) * rate, 1.0e-12)
        z = max(float(measurement.counts), 0.0)
        self.log_weights += z * np.log(lam) - lam
        self._normalize_log_weights()
        ess_pre = self.effective_sample_size()
        resampled = False
        threshold = float(self.config.resample_ess_fraction) * float(self.config.num_particles)
        if ess_pre < threshold:
            self.resample()
            resampled = True
        return {
            "ess_pre": float(ess_pre),
            "ess_post": float(self.effective_sample_size()),
            "resampled": bool(resampled),
            "count": float(z),
            "compute_backend": "gpu" if bool(self.config.use_gpu) else "cpu",
        }

    def _normalize_log_weights(self) -> None:
        """Normalize log weights in a numerically stable way."""
        normalizer = float(logsumexp(self.log_weights))
        if not np.isfinite(normalizer):
            n = int(self.config.num_particles)
            self.log_weights[:] = -np.log(float(n))
            return
        self.log_weights -= normalizer

    def resample(self) -> None:
        """Resample particles and apply Kemp-style dynamic regularization."""
        indices = _systematic_resample(self.weights, self.rng)
        self.source_indices = self.source_indices[indices].copy()
        self.strengths = self.strengths[indices].copy()
        self.log_weights[:] = -np.log(float(self.config.num_particles))
        self.resample_count += 1
        self.regularize()

    def regularize(self) -> None:
        """Jitter positions, strengths, and source cardinality after resampling."""
        self._jitter_positions()
        self._jitter_strengths()
        self._death_proposals()
        self._birth_proposals()
        self._enforce_particle_validity()

    def _jitter_positions(self) -> None:
        """Move active discrete sources by Gaussian position jitter."""
        sigma = max(float(self.config.position_jitter_m), 0.0)
        if sigma <= 0.0 or float(self.config.p_move) <= 0.0:
            return
        active_locations = np.argwhere(self.source_indices >= 0)
        for particle, slot in active_locations:
            if float(self.rng.random()) > float(self.config.p_move):
                continue
            old_index = int(self.source_indices[particle, slot])
            old_pos = self.kernel.source_grid[old_index]
            proposed = old_pos + self.rng.normal(0.0, sigma, size=3)
            _, new_index = self.grid_tree.query(proposed)
            self.source_indices[particle, slot] = int(new_index)

    def _jitter_strengths(self) -> None:
        """Apply lognormal strength regularization to active sources."""
        sigma = max(float(self.config.strength_jitter_log_sigma), 0.0)
        if sigma <= 0.0:
            return
        active = self.source_indices >= 0
        noise = self.rng.lognormal(mean=0.0, sigma=sigma, size=self.strengths.shape)
        self.strengths = np.where(active, self.strengths * noise, self.strengths)
        self.strengths = np.where(
            active,
            np.clip(
                self.strengths,
                float(self.config.min_strength_cps_1m),
                float(self.config.max_strength_cps_1m),
            ),
            0.0,
        )

    def _death_proposals(self) -> None:
        """Randomly remove one source from some particles while keeping one source."""
        p_death = max(float(self.config.p_death), 0.0)
        if p_death <= 0.0:
            return
        for particle in range(int(self.config.num_particles)):
            active_slots = np.flatnonzero(self.source_indices[particle] >= 0)
            if active_slots.size <= 1 or float(self.rng.random()) > p_death:
                continue
            slot = int(self.rng.choice(active_slots))
            self.source_indices[particle, slot] = -1
            self.strengths[particle, slot] = 0.0

    def _birth_proposals(self) -> None:
        """Randomly add a new source to some non-full particles."""
        p_birth = max(float(self.config.p_birth), 0.0)
        if p_birth <= 0.0:
            return
        for particle in range(int(self.config.num_particles)):
            free_slots = np.flatnonzero(self.source_indices[particle] < 0)
            if free_slots.size == 0 or float(self.rng.random()) > p_birth:
                continue
            slot = int(free_slots[0])
            self.source_indices[particle, slot] = int(self.rng.integers(0, self.kernel.num_sources))
            strength = self.rng.lognormal(
                mean=float(self.config.init_strength_log_mean),
                sigma=max(float(self.config.init_strength_log_sigma), 1.0e-6),
            )
            self.strengths[particle, slot] = float(
                np.clip(
                    strength,
                    float(self.config.min_strength_cps_1m),
                    float(self.config.max_strength_cps_1m),
                )
            )

    def _enforce_particle_validity(self) -> None:
        """Keep each particle in the baseline's one-or-more-source state space."""
        for particle in range(int(self.config.num_particles)):
            active = self.source_indices[particle] >= 0
            if not np.any(active):
                self.source_indices[particle, 0] = int(
                    self.rng.integers(0, self.kernel.num_sources)
                )
                self.strengths[particle, 0] = float(self.config.min_strength_cps_1m)
                active = self.source_indices[particle] >= 0
            self.strengths[particle] = np.where(
                active,
                np.clip(
                    self.strengths[particle],
                    float(self.config.min_strength_cps_1m),
                    float(self.config.max_strength_cps_1m),
                ),
                0.0,
            )

    def map_state(self) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Return merged source indices and strengths for the MAP particle."""
        best = int(np.argmax(self.log_weights))
        active = self.source_indices[best] >= 0
        return _merge_duplicate_sources(
            self.source_indices[best, active],
            self.strengths[best, active],
        )

    def refit_strengths(
        self,
        source_indices: Sequence[int],
    ) -> NDArray[np.float64]:
        """Refit source strengths for fixed positions using NNLS counts."""
        indices = np.asarray(tuple(int(index) for index in source_indices), dtype=np.int64)
        if indices.size == 0 or not self.measurements:
            return np.zeros(0, dtype=float)
        design = np.zeros((len(self.measurements), indices.size), dtype=float)
        target = np.zeros(len(self.measurements), dtype=float)
        for row, measurement in enumerate(self.measurements):
            theta = self.kernel.kernel_vector(
                self.isotope,
                measurement.detector_pos,
                measurement.fe_index,
                measurement.pb_index,
            )
            design[row, :] = float(measurement.live_time_s) * theta[indices]
            target[row] = max(
                float(measurement.counts)
                - float(measurement.live_time_s) * max(float(self.config.background_cps), 0.0),
                0.0,
            )
        strengths, _ = nnls(design, target)
        return np.clip(
            strengths,
            float(self.config.min_strength_cps_1m),
            float(self.config.max_strength_cps_1m),
        )

    def posterior_grid_summary(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return posterior existence probability and strength mass per grid cell."""
        existence = np.zeros(self.kernel.num_sources, dtype=float)
        strength_mass = np.zeros(self.kernel.num_sources, dtype=float)
        weights = self.weights
        for particle, weight in enumerate(weights):
            active = self.source_indices[particle] >= 0
            if not np.any(active):
                continue
            indices, strengths = _merge_duplicate_sources(
                self.source_indices[particle, active],
                self.strengths[particle, active],
            )
            existence[indices] += float(weight)
            strength_mass[indices] += float(weight) * strengths
        return existence, strength_mass

    def _posterior_source_indices(self) -> NDArray[np.int64]:
        """Select source indices from posterior grid mass instead of one MAP particle."""
        existence, strength_mass = self.posterior_grid_summary()
        score = np.asarray(strength_mass, dtype=float)
        order = np.argsort(score)[::-1]
        selected: list[int] = []
        min_probability = max(float(self.config.estimate_min_grid_probability), 0.0)
        merge_radius = max(float(self.config.estimate_merge_radius_m), 0.0)
        for index in order:
            idx = int(index)
            if score[idx] <= 0.0:
                break
            if existence[idx] < min_probability:
                continue
            if selected:
                distances = np.linalg.norm(
                    self.kernel.source_grid[np.asarray(selected, dtype=int)]
                    - self.kernel.source_grid[idx],
                    axis=1,
                )
                if np.any(distances <= merge_radius):
                    continue
            selected.append(idx)
            if len(selected) >= int(self.config.max_sources):
                break
        if not selected:
            map_indices, _ = self.map_state()
            return map_indices
        return np.asarray(selected, dtype=np.int64)

    def estimate_sources(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
        """Return final source positions, strengths, and grid indices."""
        indices = self._posterior_source_indices()
        _, strength_mass = self.posterior_grid_summary()
        strengths = np.asarray(strength_mass[indices], dtype=float)
        if bool(self.config.refit_final_strengths) and indices.size > 0:
            strengths = self.refit_strengths(indices)
        positions = self.kernel.positions_for_indices(indices)
        return positions, strengths, indices
