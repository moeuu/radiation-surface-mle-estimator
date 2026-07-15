"""Parallel isotope wrapper and mixing logic for the Kemp baseline."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from baselines.kemp.filter import KempFilterConfig, KempLogDDPF, KempMeasurement
from baselines.kemp.kernels import DiscreteAttenuationKernel


@dataclass(frozen=True)
class KempParallelConfig:
    """Configure the parallel isotope-specific Kemp DDPF bank."""

    filter_config: KempFilterConfig = field(default_factory=KempFilterConfig)
    mixing_keep_fraction: float = 0.9
    mixing_total_support_fraction: float = 0.35
    mixing_min_expected_counts: float = 3.0
    output_min_strength_cps_1m: float = 100.0
    source_delta_ll_min: float = 2.0
    update_workers: int = 1


@dataclass(frozen=True)
class KempEstimate:
    """Store final estimates for one isotope."""

    positions: NDArray[np.float64]
    strengths: NDArray[np.float64]
    source_indices: NDArray[np.int64]


class KempParallelLogDDPF:
    """Run independent Kemp log-domain DDPFs and mix their outputs."""

    def __init__(
        self,
        *,
        isotopes: Sequence[str],
        kernel: DiscreteAttenuationKernel,
        config: KempParallelConfig,
    ) -> None:
        """Create one isotope-specific filter per candidate isotope."""
        self.isotopes = tuple(str(isotope) for isotope in isotopes)
        self.kernel = kernel
        self.config = config
        self.filters = {
            isotope: KempLogDDPF(
                isotope=isotope,
                kernel=kernel,
                config=KempFilterConfig(
                    **{
                        **config.filter_config.__dict__,
                        "rng_seed": int(config.filter_config.rng_seed) + offset,
                    }
                ),
            )
            for offset, isotope in enumerate(self.isotopes)
        }
        self.observations = {isotope: [] for isotope in self.isotopes}
        self.step_diagnostics: list[dict[str, object]] = []

    def update(
        self,
        *,
        detector_pos: Sequence[float],
        live_time_s: float,
        counts_by_isotope: Mapping[str, float],
        variances_by_isotope: Mapping[str, float] | None = None,
        fe_index: int = 0,
        pb_index: int = 0,
    ) -> dict[str, dict[str, float | bool]]:
        """Update every isotope-specific filter with one count vector."""
        jobs: list[tuple[str, KempLogDDPF, KempMeasurement]] = []
        variances = {} if variances_by_isotope is None else dict(variances_by_isotope)
        pose = tuple(float(value) for value in detector_pos)
        for isotope, filt in self.filters.items():
            count = max(float(counts_by_isotope.get(isotope, 0.0)), 0.0)
            variance = max(float(variances.get(isotope, max(count, 1.0))), 1.0)
            measurement = KempMeasurement(
                detector_pos=pose,
                live_time_s=float(live_time_s),
                counts=count,
                variance=variance,
                fe_index=int(fe_index),
                pb_index=int(pb_index),
            )
            self.observations[isotope].append(measurement)
            jobs.append((isotope, filt, measurement))
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

    def raw_estimates(self) -> dict[str, KempEstimate]:
        """Return unmixed MAP estimates from every isotope filter."""
        estimates: dict[str, KempEstimate] = {}
        for isotope, filt in self.filters.items():
            positions, strengths, source_indices = filt.estimate_sources()
            estimates[isotope] = KempEstimate(
                positions=positions,
                strengths=strengths,
                source_indices=source_indices,
            )
        return estimates

    def mixed_estimates(self) -> dict[str, KempEstimate]:
        """Apply Kemp-style isotope/source mixing to remove unsupported outputs."""
        raw = self.raw_estimates()
        mixed: dict[str, KempEstimate] = {}
        for isotope, estimate in raw.items():
            support_keep = self._mixing_keep_mask(isotope, estimate)
            ll_keep = self._delta_ll_keep_mask(isotope, estimate)
            keep = support_keep & ll_keep
            mixed[isotope] = KempEstimate(
                positions=estimate.positions[keep],
                strengths=estimate.strengths[keep],
                source_indices=estimate.source_indices[keep],
            )
        return mixed

    def _mixing_keep_mask(
        self,
        isotope: str,
        estimate: KempEstimate,
    ) -> NDArray[np.bool_]:
        """Return which source estimates are supported by the observation record."""
        strengths = np.asarray(estimate.strengths, dtype=float)
        indices = np.asarray(estimate.source_indices, dtype=np.int64)
        if strengths.size == 0:
            return np.zeros(0, dtype=bool)
        keep = np.zeros(strengths.size, dtype=bool)
        observations = self.observations.get(str(isotope), [])
        if not observations:
            return keep
        for item_idx, (source_index, strength) in enumerate(zip(indices, strengths)):
            if float(strength) < float(self.config.output_min_strength_cps_1m):
                continue
            expected = []
            measured = []
            for measurement in observations:
                theta = self.kernel.kernel_vector(
                    str(isotope),
                    measurement.detector_pos,
                    measurement.fe_index,
                    measurement.pb_index,
                )
                expected_count = (
                    float(measurement.live_time_s)
                    * float(theta[int(source_index)])
                    * float(strength)
                )
                expected.append(expected_count)
                measured.append(max(float(measurement.counts), 0.0))
            expected_arr = np.asarray(expected, dtype=float)
            measured_arr = np.asarray(measured, dtype=float)
            best = int(np.argmax(expected_arr))
            max_expected = float(expected_arr[best])
            if max_expected < float(self.config.mixing_min_expected_counts):
                continue
            best_pass = (
                float(measured_arr[best])
                >= float(self.config.mixing_keep_fraction) * max_expected
            )
            total_expected = float(np.sum(expected_arr))
            total_measured = float(np.sum(measured_arr))
            total_pass = (
                total_expected >= float(self.config.mixing_min_expected_counts)
                and total_measured / max(total_expected, 1.0e-12)
                >= float(self.config.mixing_total_support_fraction)
            )
            keep[item_idx] = bool(best_pass or total_pass)
        return keep

    def _delta_ll_keep_mask(
        self,
        isotope: str,
        estimate: KempEstimate,
    ) -> NDArray[np.bool_]:
        """Return sources whose removal decreases the isotope Poisson likelihood."""
        strengths = np.asarray(estimate.strengths, dtype=float)
        indices = np.asarray(estimate.source_indices, dtype=np.int64)
        if strengths.size == 0:
            return np.zeros(0, dtype=bool)
        observations = self.observations.get(str(isotope), [])
        if not observations:
            return np.zeros(strengths.size, dtype=bool)
        lambda_m = np.zeros((len(observations), strengths.size), dtype=float)
        z = np.zeros(len(observations), dtype=float)
        background = float(self.config.filter_config.background_cps)
        background_counts = np.zeros(len(observations), dtype=float)
        for row, measurement in enumerate(observations):
            theta = self.kernel.kernel_vector(
                str(isotope),
                measurement.detector_pos,
                measurement.fe_index,
                measurement.pb_index,
            )
            lambda_m[row, :] = (
                float(measurement.live_time_s)
                * theta[indices]
                * np.maximum(strengths, 0.0)
            )
            background_counts[row] = max(background, 0.0) * float(measurement.live_time_s)
            z[row] = max(float(measurement.counts), 0.0)
        total = np.maximum(np.sum(lambda_m, axis=1) + background_counts, 1.0e-12)
        keep = np.zeros(strengths.size, dtype=bool)
        threshold = float(self.config.source_delta_ll_min)
        for source_idx in range(strengths.size):
            removed = np.maximum(total - lambda_m[:, source_idx], 1.0e-12)
            delta_ll = float(np.sum(z * (np.log(total) - np.log(removed)) - (total - removed)))
            keep[source_idx] = bool(delta_ll >= threshold)
        return keep

    def raw_estimates_for_metrics(self) -> dict[str, list[dict[str, object]]]:
        """Return unmixed estimates in the format expected by evaluation metrics."""
        estimates = self.raw_estimates()
        return self._format_estimates_for_metrics(estimates)

    def estimates_for_metrics(self) -> dict[str, list[dict[str, object]]]:
        """Return mixed estimates in the format expected by evaluation metrics."""
        estimates = self.mixed_estimates()
        return self._format_estimates_for_metrics(estimates)

    def _format_estimates_for_metrics(
        self,
        estimates: dict[str, KempEstimate],
    ) -> dict[str, list[dict[str, object]]]:
        """Format Kemp estimates for metric computation and JSON output."""
        out: dict[str, list[dict[str, object]]] = {}
        for isotope, estimate in estimates.items():
            out[isotope] = [
                {
                    "position": position.tolist(),
                    "strength": float(strength),
                }
                for position, strength in zip(estimate.positions, estimate.strengths)
            ]
        return out
