"""Attenuation-aware response kernel for the Anderson et al. RBE baseline."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from measurement.continuous_kernels import ContinuousKernel
from measurement.kernels import ShieldParams
from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid


@dataclass(frozen=True)
class AndersonKernelConfig:
    """Configure the Anderson baseline attenuation kernel."""

    obstacle_height_m: float = 2.0
    obstacle_buildup_coeff: float = 0.0
    detector_radius_m: float = 0.0
    detector_aperture_radius_m: float | None = None
    detector_aperture_samples: int = 1
    transport_response_model: dict[str, object] | None = None
    use_gpu: bool = False
    gpu_device: str = "cuda"
    gpu_dtype: str = "float64"
    cpu_workers: int = 1
    kernel_chunk_size: int = 8192


def unshielded_shield_params() -> ShieldParams:
    """Return zero-thickness shield parameters for the Anderson baseline."""
    return ShieldParams(thickness_fe_cm=0.0, thickness_pb_cm=0.0)


class AndersonAttenuationKernel:
    """
    Evaluate Anderson-style inverse-square responses with obstacle attenuation.

    The Anderson et al. baseline uses a nondirectional spectrometer and models
    attenuation by objects in the environment.  It does not use the proposed
    rotating Fe/Pb shields, so the shared continuous kernel is configured with
    zero shield thickness by default while retaining obstacle attenuation.
    """

    def __init__(
        self,
        *,
        isotopes: Sequence[str],
        mu_by_isotope: dict[str, object] | None = None,
        obstacle_mu_by_isotope: dict[str, float] | None = None,
        obstacle_grid: ObstacleGrid | None = None,
        shield_params: ShieldParams | None = None,
        line_mu_by_isotope: dict[str, object] | None = None,
        config: AndersonKernelConfig | None = None,
    ) -> None:
        """Initialize the shared physical response model."""
        self.isotopes = tuple(str(isotope) for isotope in isotopes)
        self.config = AndersonKernelConfig() if config is None else config
        self.shield_params = (
            unshielded_shield_params() if shield_params is None else shield_params
        )
        self._kernel = ContinuousKernel(
            mu_by_isotope=mu_by_isotope,
            shield_params=self.shield_params,
            use_gpu=bool(self.config.use_gpu),
            gpu_device=str(self.config.gpu_device),
            gpu_dtype=str(self.config.gpu_dtype),
            obstacle_grid=obstacle_grid,
            obstacle_height_m=float(self.config.obstacle_height_m),
            obstacle_mu_by_isotope=obstacle_mu_by_isotope,
            obstacle_buildup_coeff=float(self.config.obstacle_buildup_coeff),
            detector_radius_m=float(self.config.detector_radius_m),
            detector_aperture_radius_m=self.config.detector_aperture_radius_m,
            detector_aperture_samples=int(self.config.detector_aperture_samples),
            line_mu_by_isotope=line_mu_by_isotope,
            transport_response_model=self.config.transport_response_model,
        )

    @classmethod
    def from_environment(
        cls,
        *,
        env: EnvironmentConfig,
        isotopes: Sequence[str],
        mu_by_isotope: dict[str, object] | None = None,
        obstacle_mu_by_isotope: dict[str, float] | None = None,
        obstacle_grid: ObstacleGrid | None = None,
        line_mu_by_isotope: dict[str, object] | None = None,
        config: AndersonKernelConfig | None = None,
    ) -> "AndersonAttenuationKernel":
        """Build a kernel from a repository environment definition."""
        _ = env
        return cls(
            isotopes=isotopes,
            mu_by_isotope=mu_by_isotope,
            obstacle_mu_by_isotope=obstacle_mu_by_isotope,
            obstacle_grid=obstacle_grid,
            line_mu_by_isotope=line_mu_by_isotope,
            config=config,
        )

    def response_vector(
        self,
        *,
        isotope: str,
        detector_pos: Sequence[float],
        sources: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return cps-per-cps@1m responses for candidate source positions."""
        detector = np.asarray(tuple(float(value) for value in detector_pos), dtype=float)
        source_arr = np.asarray(sources, dtype=float)
        if detector.shape != (3,):
            raise ValueError("detector_pos must contain three coordinates.")
        if source_arr.size == 0:
            return np.zeros(0, dtype=float)
        if source_arr.ndim != 2 or source_arr.shape[1] != 3:
            raise ValueError("sources must be shaped (N, 3).")
        if bool(self.config.use_gpu) or int(self.config.cpu_workers) <= 1:
            return self._kernel.kernel_values_pair(
                isotope=str(isotope),
                detector_pos=detector,
                sources=source_arr,
                fe_index=0,
                pb_index=0,
                chunk_size=max(1, int(self.config.kernel_chunk_size)),
            )
        chunk_size = max(1, int(self.config.kernel_chunk_size))
        chunks = [
            source_arr[start : start + chunk_size]
            for start in range(0, source_arr.shape[0], chunk_size)
        ]
        workers = min(max(1, int(self.config.cpu_workers)), len(chunks))

        def _evaluate(chunk: NDArray[np.float64]) -> NDArray[np.float64]:
            """Evaluate one source-position chunk on the CPU path."""
            return self._kernel.kernel_values_pair(
                isotope=str(isotope),
                detector_pos=detector,
                sources=chunk,
                fe_index=0,
                pb_index=0,
            )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            parts = list(executor.map(_evaluate, chunks))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=float)

    def expected_counts_batch(
        self,
        *,
        isotope: str,
        detector_pos: Sequence[float],
        sources: NDArray[np.float64],
        activities: NDArray[np.float64],
        live_time_s: float,
        background_cps: float = 0.0,
    ) -> NDArray[np.float64]:
        """Return expected counts for a batch of Anderson particle states."""
        source_arr = np.asarray(sources, dtype=float)
        activity_arr = np.asarray(activities, dtype=float)
        if source_arr.ndim != 3 or source_arr.shape[2] != 3:
            raise ValueError("sources must be shaped (N, S, 3).")
        if activity_arr.shape != source_arr.shape[:2]:
            raise ValueError("activities must be shaped (N, S).")
        particle_count, source_count = source_arr.shape[:2]
        if particle_count == 0 or source_count == 0:
            return np.zeros(particle_count, dtype=float)
        response = self.response_vector(
            isotope=str(isotope),
            detector_pos=detector_pos,
            sources=source_arr.reshape(-1, 3),
        )
        response = response.reshape(particle_count, source_count)
        rates = np.sum(response * np.maximum(activity_arr, 0.0), axis=1)
        rates += max(float(background_cps), 0.0)
        return max(float(live_time_s), 0.0) * np.maximum(rates, 0.0)

    def expected_counts(
        self,
        *,
        isotope: str,
        detector_pos: Sequence[float],
        sources: NDArray[np.float64],
        activities: NDArray[np.float64],
        live_time_s: float,
        background_cps: float = 0.0,
    ) -> float:
        """Return expected counts for one Anderson particle state."""
        source_arr = np.asarray(sources, dtype=float)
        activity_arr = np.asarray(activities, dtype=float)
        if source_arr.size == 0:
            return float(max(live_time_s, 0.0) * max(background_cps, 0.0))
        if source_arr.ndim != 2 or source_arr.shape[1] != 3:
            raise ValueError("sources must be shaped (N, 3).")
        if activity_arr.shape != (source_arr.shape[0],):
            raise ValueError("activities must match the number of sources.")
        expected = self.expected_counts_batch(
            isotope=isotope,
            detector_pos=detector_pos,
            sources=source_arr[np.newaxis, :, :],
            activities=activity_arr[np.newaxis, :],
            live_time_s=live_time_s,
            background_cps=background_cps,
        )
        return float(expected[0])
