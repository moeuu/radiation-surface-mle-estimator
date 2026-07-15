"""Parallel PF wrapper with one PF per isotope (Chapter 3.3.5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from pf.particle_filter import IsotopeParticleFilter, PFConfig
from pf.state import IsotopeState


@dataclass
class Measurement:
    """Simple container for isotope-wise counts at one time step."""

    counts_by_isotope: Dict[str, float]
    pose_idx: int
    orient_idx: int
    live_time_s: float = 1.0
    fe_index: int | None = None
    pb_index: int | None = None
    RFe: np.ndarray | None = None
    RPb: np.ndarray | None = None
    detector_position: np.ndarray | None = None
    count_variance_by_isotope: Dict[str, float] | None = None
    # Optional IG bookkeeping can be added by callers


@dataclass
class IsotopePFStateSnapshot:
    """
    Lightweight view of the PF state for inspection/visualization.

    - particle_positions: list of (r,3) arrays (one per particle)
    - particle_strengths: list of (r,) arrays
    - particle_backgrounds: list of floats
    - weights: normalized weights (np.ndarray)
    - estimate: current IsotopeState (positions/strengths/background/covariances)
    """

    particle_positions: List[np.ndarray]
    particle_strengths: List[np.ndarray]
    particle_backgrounds: List[float]
    weights: np.ndarray
    estimate: IsotopeState


@dataclass
class EstimatedSource:
    """Simple container for a final estimated source."""

    position: np.ndarray  # shape (3,)
    strength: float


class ParallelIsotopePF:
    """
    Hold one PF per isotope and provide update/estimate helpers (Sec. 3.3.5).

    Uses the continuous PF path via IsotopeParticleFilter.update_continuous_pair.
    Use `estimate_all()` after convergence to retrieve per-isotope means and 4x4 covariances
    over (x, y, z, q) for downstream visualization/reporting.
    """

    def __init__(self, isotope_names: List[str], config: PFConfig) -> None:
        """Create one isotope filter per isotope name."""
        self.measurements: List[Measurement] = []
        self.filters: Dict[str, IsotopeParticleFilter] = {
            iso: IsotopeParticleFilter(isotope=iso, kernel=None, config=config)  # kernel to be injected later
            for iso in isotope_names
        }
        self.history: List[Dict[str, IsotopeState]] = []

    def attach_kernel(self, isotope: str, kernel) -> None:
        """Assign a measurement kernel to a specific isotope PF."""
        if isotope in self.filters:
            self.filters[isotope].set_kernel(kernel)

    def update_all(self, measurement: Measurement) -> None:
        """Update all isotope PFs with isotope-wise counts z_{k,h}."""
        self.measurements.append(measurement)
        for iso, pf in self.filters.items():
            z_kh = measurement.counts_by_isotope.get(iso, 0.0)
            # continuous update assumes kernel/poses/orientations are set on pf.kernel
            if measurement.fe_index is None or measurement.pb_index is None:
                raise ValueError(
                    "Measurement must include fe_index and pb_index. "
                    "Single-orientation updates are disabled."
                )
            pf.update_continuous_pair(
                z_obs=z_kh,
                pose_idx=measurement.pose_idx,
                fe_index=measurement.fe_index,
                pb_index=measurement.pb_index,
                live_time_s=measurement.live_time_s,
                observation_count_variance=(
                    0.0
                    if measurement.count_variance_by_isotope is None
                    else float(measurement.count_variance_by_isotope.get(iso, 0.0))
                ),
            )

    def estimate_all(self) -> Dict[str, IsotopeState]:
        """
        Return current weighted-average estimates for each isotope.

        For now, returns the weighted mean positions/strengths/background from continuous particles,
        and includes 4x4 covariance over (x, y, z, q) per source if continuous particles are present.
        """
        estimates: Dict[str, IsotopeState] = {}
        for iso, pf in self.filters.items():
            w = pf.continuous_weights
            if not pf.continuous_particles:
                continue
            # Expected num_sources = weighted average
            num_sources = int(np.round(np.sum([p.state.num_sources * wi for p, wi in zip(pf.continuous_particles, w)])))
            max_r = max((p.state.num_sources for p in pf.continuous_particles), default=0)
            positions = np.zeros((num_sources, 3)) if num_sources > 0 else np.zeros((0, 3))
            strengths = np.zeros(num_sources) if num_sources > 0 else np.zeros(0)
            covs = np.zeros((num_sources, 4, 4)) if num_sources > 0 else None
            if max_r > 0 and num_sources > 0:
                use_gpu_estimate = False
                if bool(pf.config.use_gpu):
                    from pf import gpu_utils

                    use_gpu_estimate = bool(gpu_utils.torch_available())
                if use_gpu_estimate:
                    from pf import gpu_utils
                    import torch

                    device = gpu_utils.resolve_device(pf.config.gpu_device)
                    dtype = gpu_utils.resolve_dtype(pf.config.gpu_dtype)
                    states = [p.state for p in pf.continuous_particles]
                    pos_t, str_t, _, mask_t = gpu_utils.pack_states(
                        states,
                        device=device,
                        dtype=dtype,
                    )
                    weights_t = torch.as_tensor(w, device=device, dtype=dtype)
                    weight_sum = torch.sum(weights_t)
                    if float(weight_sum) <= 0.0:
                        weights_t = torch.full_like(
                            weights_t,
                            1.0 / max(weights_t.numel(), 1),
                        )
                    else:
                        weights_t = weights_t / weight_sum
                    w_mask = weights_t[:, None] * mask_t
                    w_sum = torch.sum(w_mask, dim=0)
                    w_sum_safe = torch.where(w_sum > 0, w_sum, torch.ones_like(w_sum))
                    pos_mean = torch.sum(w_mask[:, :, None] * pos_t, dim=0) / w_sum_safe[:, None]
                    str_mean = torch.sum(w_mask * str_t, dim=0) / w_sum_safe
                    pos_cpu = pos_mean.detach().cpu().numpy()
                    str_cpu = str_mean.detach().cpu().numpy()
                    for j in range(min(num_sources, max_r)):
                        positions[j] = pos_cpu[j]
                        strengths[j] = float(str_cpu[j])
                else:
                    for j in range(num_sources):
                        pos_stack = []
                        str_stack = []
                        weights_stack = []
                        for wi, p in zip(w, pf.continuous_particles):
                            if p.state.num_sources > j:
                                pos_stack.append(p.state.positions[j])
                                str_stack.append(p.state.strengths[j])
                                weights_stack.append(wi)
                        if not weights_stack:
                            continue
                        wj = np.array(weights_stack, dtype=float)
                        wj = wj / max(np.sum(wj), 1e-12)
                        pos_arr = np.vstack(pos_stack)
                        str_arr = np.array(str_stack, dtype=float)
                        positions[j] = np.sum(wj[:, None] * pos_arr, axis=0)
                        strengths[j] = float(np.sum(wj * str_arr))
                for j in range(num_sources):
                    pos_stack = []
                    str_stack = []
                    weights_stack = []
                    for wi, p in zip(w, pf.continuous_particles):
                        if p.state.num_sources > j:
                            pos_stack.append(p.state.positions[j])
                            str_stack.append(p.state.strengths[j])
                            weights_stack.append(wi)
                    if not weights_stack:
                        continue
                    wj = np.array(weights_stack, dtype=float)
                    wj = wj / max(np.sum(wj), 1e-12)
                    pos_arr = np.vstack(pos_stack)
                    str_arr = np.array(str_stack, dtype=float)
                    pos_mean = np.sum(wj[:, None] * pos_arr, axis=0)
                    str_mean = float(np.sum(wj * str_arr))
                    centered_pos = pos_arr - pos_mean
                    centered_q = str_arr - str_mean
                    data = np.column_stack([centered_pos, centered_q])
                    cov = data.T @ (data * wj[:, None])
                    covs[j] = cov
                bg = float(np.sum([wi * p.state.background for p, wi in zip(pf.continuous_particles, w)]))
                estimates[iso] = IsotopeState(
                    num_sources=num_sources,
                    positions=positions,
                    strengths=strengths,
                    background=bg,
                    covariances=covs,
                )
                continue
            bg = float(np.sum([wi * p.state.background for p, wi in zip(pf.continuous_particles, w)]))
            estimates[iso] = IsotopeState(
                num_sources=num_sources,
                positions=positions,
                strengths=strengths,
                background=bg,
                covariances=covs,
            )
        self.history.append(estimates)
        return estimates

    def has_converged(self, tau_conv: float = 1e-2, window: int = 5) -> bool:
        """
        Simple convergence check: if recent estimate changes are below tau_conv for all isotopes.

        Uses L2 norm over concatenated positions + strengths + background across last `window` estimates.
        """
        if len(self.history) < window + 1:
            return False
        recent = self.history[-(window + 1) :]
        for i in range(1, len(recent)):
            prev = recent[i - 1]
            curr = recent[i]
            for iso in self.filters:
                if iso not in prev or iso not in curr:
                    continue
                p_state = prev[iso]
                c_state = curr[iso]
                vec_prev = np.concatenate(
                    [p_state.positions.ravel(), p_state.strengths.ravel(), np.array([p_state.background])]
                )
                vec_curr = np.concatenate(
                    [c_state.positions.ravel(), c_state.strengths.ravel(), np.array([c_state.background])]
                )
                # Pad to same length
                L = max(len(vec_prev), len(vec_curr))
                vec_prev = np.pad(vec_prev, (0, L - len(vec_prev)), mode="constant")
                vec_curr = np.pad(vec_curr, (0, L - len(vec_curr)), mode="constant")
                if np.linalg.norm(vec_curr - vec_prev) > tau_conv:
                    return False
        return True

    def export_state(self) -> Dict[str, IsotopePFStateSnapshot]:
        """
        Export per-isotope particle clouds and current estimates for inspection/visualization.

        Does not modify PF internals or semantics.
        """
        snapshots: Dict[str, IsotopePFStateSnapshot] = {}
        estimates = self.estimate_all()
        for iso, filt in self.filters.items():
            positions = [p.state.positions.copy() for p in filt.continuous_particles]
            strengths = [p.state.strengths.copy() for p in filt.continuous_particles]
            backgrounds = [p.state.background for p in filt.continuous_particles]
            weights = filt.continuous_weights
            snapshots[iso] = IsotopePFStateSnapshot(
                particle_positions=positions,
                particle_strengths=strengths,
                particle_backgrounds=backgrounds,
                weights=weights,
                estimate=estimates.get(iso, IsotopeState(num_sources=0, positions=np.zeros((0, 3)), strengths=np.zeros(0), background=0.0)),
            )
        return snapshots

    def get_estimated_sources(self) -> Dict[str, List[EstimatedSource]]:
        """
        Return final estimated sources per isotope as a list of EstimatedSource dataclasses.

        Uses estimate_all() (weighted MMSE) so it does not alter PF semantics.
        """
        estimates = self.estimate_all()
        output: Dict[str, List[EstimatedSource]] = {}
        for iso, est in estimates.items():
            sources: List[EstimatedSource] = []
            for pos, strength in zip(est.positions, est.strengths):
                sources.append(EstimatedSource(position=pos, strength=float(strength)))
            output[iso] = sources
        return output
