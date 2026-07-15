"""Parallel isotope wrapper for the Anderson et al. RBE baseline."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from baselines.anderson.filter import (
    AndersonFilterConfig,
    AndersonMeasurement,
    AndersonRBEParticleFilter,
)
from baselines.anderson.kernels import AndersonAttenuationKernel


@dataclass(frozen=True)
class AndersonParallelConfig:
    """Configure a bank of isotope-specific Anderson RBE filters."""

    filter_config: AndersonFilterConfig = field(default_factory=AndersonFilterConfig)
    initialize_all_isotopes: bool = False
    detection_count_threshold: float = 1.0
    estimate_method: str = "weighted_mean"
    update_workers: int = 1


class AndersonParallelRBE:
    """Maintain independent Anderson RBE filters for detected isotopes."""

    def __init__(
        self,
        *,
        isotopes: Sequence[str],
        kernel: AndersonAttenuationKernel,
        config: AndersonParallelConfig,
    ) -> None:
        """Create an isotope-filter bank with optional lazy initialization."""
        self.isotopes = tuple(str(isotope) for isotope in isotopes)
        self.kernel = kernel
        self.config = config
        self.filters: dict[str, AndersonRBEParticleFilter] = {}
        self.step_diagnostics: list[dict[str, object]] = []
        if bool(config.initialize_all_isotopes):
            for isotope in self.isotopes:
                self.filters[isotope] = self._new_filter(isotope, offset=len(self.filters))

    def _new_filter(self, isotope: str, *, offset: int) -> AndersonRBEParticleFilter:
        """Create one isotope-specific filter with a deterministic seed offset."""
        base = self.config.filter_config
        cfg = AndersonFilterConfig(
            **{
                **base.__dict__,
                "rng_seed": int(base.rng_seed) + int(offset),
            }
        )
        return AndersonRBEParticleFilter(
            isotope=str(isotope),
            kernel=self.kernel,
            config=cfg,
        )

    def _ensure_filter(self, isotope: str) -> AndersonRBEParticleFilter:
        """Return an existing filter or lazily create one for a detected isotope."""
        key = str(isotope)
        filt = self.filters.get(key)
        if filt is None:
            filt = self._new_filter(key, offset=len(self.filters))
            self.filters[key] = filt
        return filt

    def update(
        self,
        *,
        detector_pos: Sequence[float],
        live_time_s: float,
        counts_by_isotope: Mapping[str, float],
        elapsed_s: float | None = None,
    ) -> dict[str, dict[str, float | bool]]:
        """Update all initialized or newly detected isotope filters."""
        jobs: list[tuple[str, AndersonRBEParticleFilter, AndersonMeasurement]] = []
        threshold = max(float(self.config.detection_count_threshold), 0.0)
        pose = tuple(float(value) for value in detector_pos)
        for isotope in self.isotopes:
            count = max(float(counts_by_isotope.get(isotope, 0.0)), 0.0)
            if isotope not in self.filters and count < threshold:
                continue
            filt = self._ensure_filter(isotope)
            jobs.append(
                (
                    isotope,
                    filt,
                    AndersonMeasurement(
                        detector_pos=pose,
                        live_time_s=float(live_time_s),
                        counts=count,
                        elapsed_s=elapsed_s,
                    ),
                )
            )
        workers = min(max(1, int(self.config.update_workers)), max(len(jobs), 1))
        if workers > 1 and len(jobs) > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(
                    executor.map(
                        lambda item: (item[0], item[1].update(item[2])),
                        jobs,
                    )
                )
            diagnostics = {isotope: diag for isotope, diag in results}
        else:
            diagnostics = {
                isotope: filt.update(measurement)
                for isotope, filt, measurement in jobs
            }
        self.step_diagnostics.append({"diagnostics": diagnostics})
        return diagnostics

    def estimates_for_metrics(self) -> dict[str, list[dict[str, object]]]:
        """Return all initialized isotope estimates in metric format."""
        estimates: dict[str, list[dict[str, object]]] = {}
        for isotope in self.isotopes:
            filt = self.filters.get(isotope)
            if filt is None:
                estimates[isotope] = []
                continue
            estimates[isotope] = filt.estimates_for_metrics(
                method=str(self.config.estimate_method)
            )
        return estimates

    def estimate_arrays(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Return raw position and activity arrays for initialized filters."""
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for isotope, filt in self.filters.items():
            out[isotope] = filt.estimate(method=str(self.config.estimate_method))
        return out
