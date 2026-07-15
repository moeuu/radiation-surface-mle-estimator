"""Represent the measurement environment for a non-directional detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:  # optional dependency
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    _TORCH_AVAILABLE = False


@dataclass(frozen=True)
class EnvironmentConfig:
    """Hold environment dimensions and detector position."""

    size_x: float = 10.0
    size_y: float = 20.0
    size_z: float = 10.0
    detector_position: Tuple[float, float, float] | None = None

    def detector(self) -> np.ndarray:
        """Return the detector position (defaults to the environment center)."""
        if self.detector_position is None:
            return np.array([self.size_x / 2.0, self.size_y / 2.0, self.size_z / 2.0])
        return np.array(self.detector_position, dtype=float)


@dataclass(frozen=True)
class PointSource:
    """
    Represent a point radiation source on the detector-count-rate scale.

    ``intensity_cps_1m`` is the expected net detector count rate at a 1 m
    source-detector distance for the configured detector and spectrum
    processing pipeline. It is not total isotropic gamma/s activity.
    """

    isotope: str
    position: Tuple[float, float, float]
    intensity_cps_1m: float

    def position_array(self) -> np.ndarray:
        """Return the position as a NumPy array."""
        return np.array(self.position, dtype=float)

    @property
    def strength(self) -> float:
        """Backward-compatible strength accessor (detector cps at 1 m)."""
        return self.intensity_cps_1m


def inverse_square_scale(detector: np.ndarray, source: PointSource) -> float:
    """
    Return the inverse-square geometric scale for a point source.

    Computes 1/d^2 based on detector distance for detector cps@1m scaling.
    """
    distance = np.linalg.norm(detector - source.position_array())
    if distance == 0:
        # Clip zero distance to avoid unrealistic singularity.
        distance = 1e-6
    return 1.0 / (distance**2)


def inverse_square_scale_batch(
    detectors: np.ndarray,
    sources: np.ndarray,
    use_gpu: bool | None = None,
    gpu_device: str = "cuda",
    gpu_dtype: str = "float32",
) -> np.ndarray:
    """
    Return inverse-square scaling for paired detector/source arrays.

    Args:
        detectors: (N, 3) detector positions.
        sources: (N, 3) source positions.
        use_gpu: If True, compute on CUDA when available.
        gpu_device: Torch device string.
        gpu_dtype: Torch dtype string.
    """
    detectors = np.asarray(detectors, dtype=float)
    sources = np.asarray(sources, dtype=float)
    if detectors.shape != sources.shape or detectors.ndim != 2 or detectors.shape[1] != 3:
        raise ValueError("detectors and sources must be shape (N, 3)")
    if use_gpu is None:
        use_gpu = bool(_TORCH_AVAILABLE and torch is not None and torch.cuda.is_available())
    if use_gpu and torch is not None:
        device = torch.device(gpu_device) if torch.cuda.is_available() else torch.device("cpu")
        dtype = torch.float32 if gpu_dtype == "float32" else torch.float64
        det_t = torch.as_tensor(detectors, device=device, dtype=dtype)
        src_t = torch.as_tensor(sources, device=device, dtype=dtype)
        dist = torch.linalg.norm(det_t - src_t, dim=1)
        tol = torch.as_tensor(1e-6, device=device, dtype=dtype)
        dist = torch.where(dist <= tol, tol, dist)
        scale = 1.0 / (dist**2)
        return scale.detach().cpu().numpy()
    dist = np.linalg.norm(detectors - sources, axis=1)
    dist = np.where(dist <= 1e-6, 1e-6, dist)
    return 1.0 / (dist**2)
