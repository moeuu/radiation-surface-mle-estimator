"""Baseline PF using proposal-style updates with inverse-square observations (no shielding)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Sequence

import numpy as np
from numpy.typing import NDArray

from pf.particle_filter import IsotopeParticleFilter, PFConfig

if TYPE_CHECKING:
    import torch


@dataclass
class NoShieldPFConfig:
    """Configuration wrapper for the no-shield PF."""

    pf_config: PFConfig


class NoShieldIsotopeFilter(IsotopeParticleFilter):
    """Proposal-style filter with inverse-square expected counts."""

    def _continuous_expected_counts_pair_torch(
        self,
        pose_idx: int,
        fe_index: int,
        pb_index: int,
        live_time_s: float,
    ) -> "torch.Tensor":
        """Compute expected counts using inverse-square scaling (pose index unused)."""
        detector_pos = np.asarray(self.kernel.poses[pose_idx], dtype=float) if self.kernel else None
        if detector_pos is None:
            raise RuntimeError("No detector position available for inverse-square update.")
        return self._continuous_expected_counts_pair_at_pose_torch(
            detector_pos=detector_pos,
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=live_time_s,
        )

    def _continuous_expected_counts_pair_at_pose_torch(
        self,
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        live_time_s: float,
    ) -> "torch.Tensor":
        """Compute expected counts using inverse-square scaling at an explicit pose."""
        from pf import gpu_utils
        import torch

        device = gpu_utils.resolve_device(self.config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.config.gpu_dtype)
        if not self.continuous_particles:
            return torch.zeros(0, device=device, dtype=dtype)
        states = [p.state for p in self.continuous_particles]
        positions, strengths, backgrounds, mask = gpu_utils.pack_states(states, device=device, dtype=dtype)
        detector = torch.as_tensor(detector_pos, device=device, dtype=dtype).view(1, 1, 3)
        direction = detector - positions
        dist = torch.linalg.norm(direction, dim=-1)
        tol = torch.as_tensor(1e-6, device=device, dtype=dtype)
        dist = torch.where(dist <= tol, tol, dist)
        geom = 1.0 / (dist**2)
        rate = torch.sum(strengths * geom * mask, dim=1) + backgrounds
        return rate * float(live_time_s)

    def _ll_proxy_pair(
        self,
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        z_obs: float,
    ) -> float:
        """Return a Poisson log-likelihood proxy using inverse-square scaling."""
        if not self.continuous_particles:
            return 0.0
        state = self.best_particle().state
        lam_rate = float(state.background)
        if state.num_sources > 0:
            for pos, strength in zip(state.positions[:state.num_sources], state.strengths[:state.num_sources]):
                distance = float(np.linalg.norm(detector_pos - pos))
                if distance <= 1e-6:
                    distance = 1e-6
                lam_rate += float(strength) / (distance**2)
        lam = float(live_time_s) * lam_rate
        eps = 1e-12
        return float(z_obs * np.log(lam + eps) - lam)


class NoShieldPF:
    """Parallel PF with proposal-style updates and inverse-square observations."""

    def __init__(
        self,
        isotopes: Sequence[str],
        config: PFConfig,
    ) -> None:
        """Initialize per-isotope filters."""
        self.filters: Dict[str, NoShieldIsotopeFilter] = {
            iso: NoShieldIsotopeFilter(isotope=iso, kernel=None, config=config)
            for iso in isotopes
        }

    def update_all(
        self,
        detector_pos: NDArray[np.float64],
        counts_by_isotope: Dict[str, float],
        live_time_s: float,
        step_idx: int | None = None,
    ) -> None:
        """Update all isotope filters for a single measurement."""
        det_pos = np.asarray(detector_pos, dtype=float)
        for iso, filt in self.filters.items():
            z = float(counts_by_isotope.get(iso, 0.0))
            filt.update_continuous_pair_at_pose(
                z_obs=z,
                detector_pos=det_pos,
                fe_index=0,
                pb_index=0,
                live_time_s=live_time_s,
                step_idx=step_idx,
            )

    def estimate_all(self) -> Dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Return per-isotope estimates as (positions, strengths)."""
        return {iso: filt.estimate() for iso, filt in self.filters.items()}
