"""Fisher-information waypoint selection for the Anderson baseline."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from baselines.anderson.filter import AndersonRBEParticleFilter


@dataclass(frozen=True)
class AndersonFisherConfig:
    """Configure Anderson-style greedy Fisher waypoint selection."""

    live_time_s: float = 30.0
    information_weight: float = 2.0
    navigation_weight: float = 1.0
    ridge: float = 1.0e-6
    max_particles: int | None = 256
    cpu_workers: int = 1
    use_gpu: bool = False
    gpu_device: str = "cuda"
    gpu_dtype: str = "float64"


def _torch_config(device_name: str, dtype_name: str) -> tuple[object, object, object]:
    """Return torch, device, and dtype for an explicit GPU planning path."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Anderson Fisher GPU planning requires torch.") from exc
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Anderson Fisher GPU requested CUDA, but CUDA is unavailable.")
    device = torch.device(str(device_name))
    if str(dtype_name) == "float32":
        dtype = torch.float32
    elif str(dtype_name) == "float64":
        dtype = torch.float64
    else:
        raise ValueError("gpu_dtype must be 'float32' or 'float64'.")
    return torch, device, dtype


def fisher_information_matrix(
    *,
    detector_pos: Sequence[float],
    source_positions: NDArray[np.float64],
    activities: NDArray[np.float64],
    live_time_s: float,
    background_cps: float = 0.0,
    ridge: float = 0.0,
) -> NDArray[np.float64]:
    """
    Return the Poisson FIM used by Anderson et al. for one particle.

    This planning model intentionally follows the paper and omits directionality
    and attenuation from the Fisher metric, even when the update likelihood uses
    obstacle attenuation.
    """
    detector = np.asarray(tuple(float(value) for value in detector_pos), dtype=float)
    positions = np.asarray(source_positions, dtype=float)
    strengths = np.asarray(activities, dtype=float)
    if detector.shape != (3,):
        raise ValueError("detector_pos must contain three coordinates.")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("source_positions must be shaped (N, 3).")
    if strengths.shape != (positions.shape[0],):
        raise ValueError("activities must match source_positions.")
    source_count = int(positions.shape[0])
    grad = np.zeros(4 * source_count, dtype=float)
    live = max(float(live_time_s), 0.0)
    mean_rate = max(float(background_cps), 0.0)
    for idx in range(source_count):
        delta = detector - positions[idx]
        r2 = max(float(np.dot(delta, delta)), 1.0e-12)
        inv_r2 = 1.0 / r2
        activity = max(float(strengths[idx]), 0.0)
        mean_rate += activity * inv_r2
        base = 4 * idx
        grad[base] = live * inv_r2
        grad[base + 1 : base + 4] = live * activity * (2.0 * delta / (r2 * r2))
    mean_counts = max(live * mean_rate, 1.0e-12)
    fim = np.outer(grad, grad) / mean_counts
    if float(ridge) > 0.0:
        fim += float(ridge) * np.eye(fim.shape[0], dtype=float)
    return fim


def average_fisher_information(
    *,
    filt: AndersonRBEParticleFilter,
    detector_pos: Sequence[float],
    config: AndersonFisherConfig,
) -> NDArray[np.float64]:
    """Return the particle-weighted FIM at a candidate detector position."""
    weights = filt.weights
    indices = np.arange(weights.size)
    max_particles = config.max_particles
    if max_particles is not None and int(max_particles) < indices.size:
        order = np.argsort(weights)[::-1][: max(1, int(max_particles))]
        indices = np.asarray(order, dtype=int)
        local_weights = weights[indices]
        total = float(np.sum(local_weights))
        local_weights = (
            local_weights / total
            if total > 0.0
            else np.ones(indices.size, dtype=float) / float(indices.size)
        )
    else:
        local_weights = weights
    size = 4 * int(filt.config.num_sources)
    if bool(config.use_gpu):
        return _average_fisher_information_gpu(
            detector_pos=detector_pos,
            positions=filt.positions[indices],
            activities=filt.activities[indices],
            weights=local_weights,
            background_cps=float(filt.config.background_cps),
            config=config,
            size=size,
        )
    detector = np.asarray(tuple(float(value) for value in detector_pos), dtype=float)
    positions = np.asarray(filt.positions[indices], dtype=float)
    activities = np.asarray(filt.activities[indices], dtype=float)
    delta = detector[None, None, :] - positions
    r2 = np.maximum(np.sum(delta * delta, axis=2), 1.0e-12)
    inv_r2 = 1.0 / r2
    live = max(float(config.live_time_s), 0.0)
    mean_rate = (
        max(float(filt.config.background_cps), 0.0)
        + np.sum(np.maximum(activities, 0.0) * inv_r2, axis=1)
    )
    mean_counts = np.maximum(live * mean_rate, 1.0e-12)
    grad = np.zeros((indices.size, size), dtype=float)
    for source_idx in range(int(filt.config.num_sources)):
        base = 4 * source_idx
        grad[:, base] = live * inv_r2[:, source_idx]
        scale = (
            live
            * np.maximum(activities[:, source_idx], 0.0)
            * 2.0
            / (r2[:, source_idx] ** 2)
        )
        grad[:, base + 1 : base + 4] = scale[:, None] * delta[:, source_idx, :]
    factors = np.asarray(local_weights, dtype=float) / mean_counts
    total_fim = np.einsum("p,pi,pj->ij", factors, grad, grad)
    return total_fim + float(config.ridge) * np.eye(size, dtype=float)


def _average_fisher_information_gpu(
    *,
    detector_pos: Sequence[float],
    positions: NDArray[np.float64],
    activities: NDArray[np.float64],
    weights: NDArray[np.float64],
    background_cps: float,
    config: AndersonFisherConfig,
    size: int,
) -> NDArray[np.float64]:
    """Return the particle-weighted FIM using torch on the configured device."""
    torch, device, dtype = _torch_config(str(config.gpu_device), str(config.gpu_dtype))
    detector = torch.as_tensor(tuple(float(v) for v in detector_pos), device=device, dtype=dtype)
    pos_t = torch.as_tensor(positions, device=device, dtype=dtype)
    act_t = torch.clamp(torch.as_tensor(activities, device=device, dtype=dtype), min=0.0)
    weights_t = torch.as_tensor(weights, device=device, dtype=dtype)
    delta = detector.view(1, 1, 3) - pos_t
    r2 = torch.clamp(torch.sum(delta * delta, dim=2), min=1.0e-12)
    inv_r2 = 1.0 / r2
    live = max(float(config.live_time_s), 0.0)
    mean_rate = max(float(background_cps), 0.0) + torch.sum(act_t * inv_r2, dim=1)
    mean_counts = torch.clamp(live * mean_rate, min=1.0e-12)
    grad = torch.zeros((pos_t.shape[0], int(size)), device=device, dtype=dtype)
    for source_idx in range(pos_t.shape[1]):
        base = 4 * source_idx
        grad[:, base] = live * inv_r2[:, source_idx]
        scale = live * act_t[:, source_idx] * 2.0 / (r2[:, source_idx] ** 2)
        grad[:, base + 1 : base + 4] = scale[:, None] * delta[:, source_idx, :]
    factors = weights_t / mean_counts
    fim = torch.einsum("p,pi,pj->ij", factors, grad, grad)
    fim = fim + float(config.ridge) * torch.eye(int(size), device=device, dtype=dtype)
    return fim.detach().cpu().numpy()


def a_optimal_cost(fim: NDArray[np.float64]) -> float:
    """Return the A-optimal trace-inverse cost for a FIM."""
    matrix = np.asarray(fim, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("fim must be a square matrix.")
    inv = np.linalg.pinv(matrix)
    return float(np.trace(inv))


def score_candidate(
    *,
    filt: AndersonRBEParticleFilter,
    current_pos: Sequence[float],
    candidate_pos: Sequence[float],
    config: AndersonFisherConfig,
) -> float:
    """Return the Anderson objective value for one candidate waypoint."""
    fim = average_fisher_information(
        filt=filt,
        detector_pos=candidate_pos,
        config=config,
    )
    info_cost = a_optimal_cost(fim)
    current = np.asarray(tuple(float(value) for value in current_pos), dtype=float)
    candidate = np.asarray(tuple(float(value) for value in candidate_pos), dtype=float)
    nav_cost = float(np.linalg.norm(candidate - current))
    return (
        float(config.information_weight) * info_cost
        + float(config.navigation_weight) * nav_cost
    )


def select_fisher_waypoint(
    *,
    filt: AndersonRBEParticleFilter,
    current_pos: Sequence[float],
    candidate_positions: NDArray[np.float64],
    config: AndersonFisherConfig | None = None,
) -> tuple[NDArray[np.float64], dict[str, float]]:
    """Select the candidate waypoint with minimum Anderson Fisher objective."""
    cfg = AndersonFisherConfig() if config is None else config
    candidates = np.asarray(candidate_positions, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 3 or candidates.shape[0] == 0:
        raise ValueError("candidate_positions must be shaped (N, 3).")
    workers = min(max(1, int(cfg.cpu_workers)), candidates.shape[0])
    if workers > 1 and not bool(cfg.use_gpu):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            scores_list = list(
                executor.map(
                    lambda candidate: score_candidate(
                        filt=filt,
                        current_pos=current_pos,
                        candidate_pos=candidate,
                        config=cfg,
                    ),
                    list(candidates),
                )
            )
        scores = np.asarray(scores_list, dtype=float)
    else:
        scores = np.asarray(
            [
            score_candidate(
                filt=filt,
                current_pos=current_pos,
                candidate_pos=candidate,
                config=cfg,
            )
            for candidate in candidates
            ],
            dtype=float,
        )
    best = int(np.argmin(scores))
    return candidates[best].copy(), {
        "score": float(scores[best]),
        "candidate_index": float(best),
        "compute_backend": "gpu" if bool(cfg.use_gpu) else "cpu",
    }
