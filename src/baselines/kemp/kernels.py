"""Discrete attenuation kernels for the Kemp et al. comparison baseline.

Kemp et al. precompute isotope-specific attenuation kernels between a
discretized source grid and measurement locations.  This module implements that
comparison model while using the repository's shared geometric, obstacle, and
shield attenuation utilities so the baseline is evaluated against the same
scene description as the proposed method.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from measurement.continuous_kernels import ContinuousKernel
from measurement.kernels import ShieldParams
from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid


@dataclass(frozen=True)
class KempKernelConfig:
    """Configure the discrete Kemp attenuation kernel."""

    grid_spacing_m: tuple[float, float, float] = (0.5, 0.5, 0.5)
    grid_margin_m: float = 0.5
    z_levels_m: tuple[float, ...] | None = None
    obstacle_height_m: float = 2.0
    obstacle_buildup_coeff: float = 0.0
    detector_radius_m: float = 0.0
    detector_aperture_radius_m: float | None = None
    detector_aperture_samples: int = 1
    source_extent_radius_m: float = 0.0
    source_extent_samples: int = 1
    transport_response_model: dict[str, object] | None = None
    use_gpu: bool = True
    gpu_device: str = "cuda"
    gpu_dtype: str = "float64"
    cpu_workers: int = 1
    kernel_chunk_size: int = 8192


def _axis_points(start: float, stop: float, step: float) -> NDArray[np.float64]:
    """Return evenly spaced candidate coordinates inside a closed interval."""
    if step <= 0.0:
        raise ValueError("Grid spacing must be positive.")
    if stop < start:
        return np.zeros(0, dtype=float)
    count = int(np.floor((stop - start) / step)) + 1
    return start + step * np.arange(max(count, 0), dtype=float)


def build_source_grid(
    env: EnvironmentConfig,
    config: KempKernelConfig,
) -> NDArray[np.float64]:
    """Build the discrete source grid used by the Kemp baseline."""
    margin = max(float(config.grid_margin_m), 0.0)
    spacing = tuple(float(value) for value in config.grid_spacing_m)
    xs = _axis_points(margin, float(env.size_x) - margin, spacing[0])
    ys = _axis_points(margin, float(env.size_y) - margin, spacing[1])
    if config.z_levels_m is None:
        zs = _axis_points(margin, float(env.size_z) - margin, spacing[2])
    else:
        levels = np.asarray(tuple(float(value) for value in config.z_levels_m), dtype=float)
        zs = levels[(levels >= 0.0) & (levels <= float(env.size_z))]
    if xs.size == 0 or ys.size == 0 or zs.size == 0:
        raise ValueError("Kemp source grid is empty.")
    return np.asarray([[x, y, z] for x in xs for y in ys for z in zs], dtype=float)


def _free_grid_mask(
    source_grid: NDArray[np.float64],
    obstacle_grid: ObstacleGrid | None,
) -> NDArray[np.bool_]:
    """Return a mask for source candidates outside blocked obstacle cells."""
    if obstacle_grid is None:
        return np.ones(source_grid.shape[0], dtype=bool)
    return np.asarray([obstacle_grid.is_free(point) for point in source_grid], dtype=bool)


class DiscreteAttenuationKernel:
    """Precompute and cache isotope-specific discrete attenuation kernels."""

    def __init__(
        self,
        *,
        source_grid: NDArray[np.float64],
        isotopes: Sequence[str],
        mu_by_isotope: dict[str, object] | None,
        shield_params: ShieldParams,
        obstacle_grid: ObstacleGrid | None,
        config: KempKernelConfig,
        obstacle_mu_by_isotope: dict[str, float] | None = None,
        line_mu_by_isotope: dict[str, object] | None = None,
    ) -> None:
        """Store the discrete grid and shared physical attenuation model."""
        sources = np.asarray(source_grid, dtype=float)
        if sources.ndim != 2 or sources.shape[1] != 3:
            raise ValueError("source_grid must be shaped (N, 3).")
        mask = _free_grid_mask(sources, obstacle_grid)
        self.source_grid = sources[mask]
        if self.source_grid.size == 0:
            raise ValueError("No free candidate sources remain for the Kemp kernel.")
        self.isotopes = tuple(str(isotope) for isotope in isotopes)
        self.config = config
        self._kernel = ContinuousKernel(
            mu_by_isotope=mu_by_isotope,
            shield_params=shield_params,
            use_gpu=bool(config.use_gpu),
            gpu_device=str(config.gpu_device),
            gpu_dtype=str(config.gpu_dtype),
            obstacle_grid=obstacle_grid,
            obstacle_height_m=float(config.obstacle_height_m),
            obstacle_mu_by_isotope=obstacle_mu_by_isotope,
            obstacle_buildup_coeff=float(config.obstacle_buildup_coeff),
            detector_radius_m=float(config.detector_radius_m),
            detector_aperture_radius_m=config.detector_aperture_radius_m,
            detector_aperture_samples=int(config.detector_aperture_samples),
            source_extent_radius_m=float(config.source_extent_radius_m),
            source_extent_samples=int(config.source_extent_samples),
            line_mu_by_isotope=line_mu_by_isotope,
            transport_response_model=config.transport_response_model,
        )

    def _kernel_values_pair(
        self,
        *,
        isotope: str,
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
    ) -> NDArray[np.float64]:
        """Return a source-grid response vector using GPU or CPU workers."""
        if bool(self.config.use_gpu) or int(self.config.cpu_workers) <= 1:
            return self._kernel.kernel_values_pair(
                isotope=str(isotope),
                detector_pos=detector_pos,
                sources=self.source_grid,
                fe_index=int(fe_index),
                pb_index=int(pb_index),
                chunk_size=max(1, int(self.config.kernel_chunk_size)),
            )
        chunk_size = max(1, int(self.config.kernel_chunk_size))
        chunks = [
            self.source_grid[start : start + chunk_size]
            for start in range(0, self.source_grid.shape[0], chunk_size)
        ]
        workers = min(max(1, int(self.config.cpu_workers)), len(chunks))
        if workers <= 1:
            return self._kernel.kernel_values_pair(
                isotope=str(isotope),
                detector_pos=detector_pos,
                sources=self.source_grid,
                fe_index=int(fe_index),
                pb_index=int(pb_index),
            )

        def _evaluate(chunk: NDArray[np.float64]) -> NDArray[np.float64]:
            """Evaluate one source-grid chunk on the CPU path."""
            return self._kernel.kernel_values_pair(
                isotope=str(isotope),
                detector_pos=detector_pos,
                sources=chunk,
                fe_index=int(fe_index),
                pb_index=int(pb_index),
            )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            parts = list(executor.map(_evaluate, chunks))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=float)

    @classmethod
    def from_environment(
        cls,
        *,
        env: EnvironmentConfig,
        isotopes: Sequence[str],
        mu_by_isotope: dict[str, object] | None,
        shield_params: ShieldParams,
        obstacle_grid: ObstacleGrid | None,
        config: KempKernelConfig,
        obstacle_mu_by_isotope: dict[str, float] | None = None,
        line_mu_by_isotope: dict[str, object] | None = None,
    ) -> "DiscreteAttenuationKernel":
        """Build a discrete kernel from an environment definition."""
        source_grid = build_source_grid(env, config)
        return cls(
            source_grid=source_grid,
            isotopes=isotopes,
            mu_by_isotope=mu_by_isotope,
            shield_params=shield_params,
            obstacle_grid=obstacle_grid,
            obstacle_mu_by_isotope=obstacle_mu_by_isotope,
            line_mu_by_isotope=line_mu_by_isotope,
            config=config,
        )

    @property
    def num_sources(self) -> int:
        """Return the number of discrete source candidates."""
        return int(self.source_grid.shape[0])

    @lru_cache(maxsize=4096)
    def values_for_measurement(
        self,
        isotope: str,
        detector_x: float,
        detector_y: float,
        detector_z: float,
        fe_index: int,
        pb_index: int,
    ) -> tuple[float, ...]:
        """Return expected cps per cps@1m for all source grid cells."""
        detector = np.asarray([detector_x, detector_y, detector_z], dtype=float)
        values = self._kernel_values_pair(
            isotope=str(isotope),
            detector_pos=detector,
            fe_index=int(fe_index),
            pb_index=int(pb_index),
        )
        return tuple(float(value) for value in values)

    def kernel_vector(
        self,
        isotope: str,
        detector_pos: Sequence[float],
        fe_index: int,
        pb_index: int,
    ) -> NDArray[np.float64]:
        """Return the cached kernel vector for one detector and shield state."""
        if len(detector_pos) != 3:
            raise ValueError("detector_pos must contain three coordinates.")
        values = self.values_for_measurement(
            str(isotope),
            float(detector_pos[0]),
            float(detector_pos[1]),
            float(detector_pos[2]),
            int(fe_index),
            int(pb_index),
        )
        return np.asarray(values, dtype=float)

    def expected_counts(
        self,
        *,
        isotope: str,
        detector_pos: Sequence[float],
        source_indices: Sequence[int],
        strengths: Sequence[float],
        live_time_s: float,
        fe_index: int,
        pb_index: int,
        background_cps: float = 0.0,
    ) -> float:
        """Compute expected counts for a discrete Kemp particle state."""
        indices = np.asarray(tuple(int(index) for index in source_indices), dtype=int)
        strength_arr = np.asarray(tuple(float(value) for value in strengths), dtype=float)
        if indices.size != strength_arr.size:
            raise ValueError("source_indices and strengths must have matching length.")
        if indices.size == 0:
            return float(live_time_s) * max(float(background_cps), 0.0)
        valid = (indices >= 0) & (indices < self.num_sources) & (strength_arr > 0.0)
        if not np.any(valid):
            return float(live_time_s) * max(float(background_cps), 0.0)
        theta = self.kernel_vector(isotope, detector_pos, fe_index, pb_index)
        rate = float(np.dot(theta[indices[valid]], strength_arr[valid]))
        rate += max(float(background_cps), 0.0)
        return float(max(float(live_time_s), 0.0) * max(rate, 0.0))

    def nearest_index(self, position: Sequence[float]) -> int:
        """Return the nearest discrete source-grid index for a position."""
        pos = np.asarray(tuple(float(value) for value in position), dtype=float)
        if pos.shape != (3,):
            raise ValueError("position must contain three coordinates.")
        dist2 = np.sum((self.source_grid - pos[np.newaxis, :]) ** 2, axis=1)
        return int(np.argmin(dist2))

    def positions_for_indices(self, indices: Iterable[int]) -> NDArray[np.float64]:
        """Return source-grid positions for valid indices."""
        index_arr = np.asarray(tuple(int(index) for index in indices), dtype=int)
        if index_arr.size == 0:
            return np.zeros((0, 3), dtype=float)
        valid = (index_arr >= 0) & (index_arr < self.num_sources)
        return np.asarray(self.source_grid[index_arr[valid]], dtype=float)
