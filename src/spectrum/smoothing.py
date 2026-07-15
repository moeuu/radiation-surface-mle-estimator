"""Smoothing utilities such as Gaussian convolution to suppress statistical noise while preserving peaks."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    import torch
    import torch.nn.functional as torch_f

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    torch_f = None
    _TORCH_AVAILABLE = False


def _resolve_torch_context(
    use_gpu: bool | None,
    gpu_device: str,
    gpu_dtype: str,
) -> tuple["torch.device", "torch.dtype"] | None:
    """Return torch device/dtype when GPU smoothing is requested and available."""
    if not use_gpu or not _TORCH_AVAILABLE or torch is None:
        return None
    if gpu_device.startswith("cuda") and not torch.cuda.is_available():
        return None
    if gpu_dtype == "float32":
        dtype = torch.float32
    elif gpu_dtype == "float64":
        dtype = torch.float64
    else:
        raise ValueError(f"Unsupported torch dtype: {gpu_dtype}")
    return torch.device(gpu_device), dtype


def _gaussian_kernel_torch(
    sigma_bins: float,
    device: "torch.device",
    dtype: "torch.dtype",
) -> "torch.Tensor":
    """Return a 1D Gaussian kernel for torch convolution."""
    if sigma_bins <= 0.0:
        return torch.ones(1, device=device, dtype=dtype)
    radius = max(1, int(np.ceil(4.0 * sigma_bins)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel = kernel / torch.sum(kernel)
    return kernel


def _gaussian_kernel_np(sigma_bins: float) -> NDArray[np.float64]:
    """Return a 1D Gaussian kernel for discrete convolution."""
    if sigma_bins <= 0.0:
        return np.ones(1, dtype=float)
    radius = max(1, int(np.ceil(4.0 * sigma_bins)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / float(sigma_bins)) ** 2)
    kernel_sum = float(np.sum(kernel))
    if kernel_sum > 0.0:
        kernel = kernel / kernel_sum
    return kernel.astype(float, copy=False)


def gaussian_smooth(
    signal: NDArray[np.float64],
    sigma_bins: float = 1.0,
    *,
    use_gpu: bool | None = None,
    gpu_device: str = "cuda",
    gpu_dtype: str = "float32",
) -> NDArray[np.float64]:
    """
    Apply 1D Gaussian smoothing to a spectrum.

    Args:
        signal: Input spectrum (counts per bin).
        sigma_bins: Standard deviation of the Gaussian kernel in bins.
        use_gpu: Enable CUDA acceleration when available.
        gpu_device: Torch device string (e.g., "cuda", "cuda:0", "cpu").
        gpu_dtype: Torch dtype string ("float32" or "float64").

    Returns:
        Smoothed spectrum.
    """
    if sigma_bins <= 0.0:
        return np.asarray(signal, dtype=float)
    ctx = _resolve_torch_context(use_gpu, gpu_device, gpu_dtype)
    if ctx is None:
        data = np.asarray(signal, dtype=float)
        if data.size == 0:
            return data
        kernel = _gaussian_kernel_np(sigma_bins)
        smoothed = np.convolve(data, kernel, mode="same")
        return np.asarray(smoothed, dtype=float)
    device, dtype = ctx
    if torch is None or torch_f is None:
        data = np.asarray(signal, dtype=float)
        if data.size == 0:
            return data
        kernel = _gaussian_kernel_np(sigma_bins)
        smoothed = np.convolve(data, kernel, mode="same")
        return np.asarray(smoothed, dtype=float)
    signal_t = torch.as_tensor(np.asarray(signal, dtype=float), device=device, dtype=dtype).view(1, 1, -1)
    kernel = _gaussian_kernel_torch(sigma_bins, device=device, dtype=dtype).view(1, 1, -1)
    pad = kernel.shape[-1] // 2
    padded = torch_f.pad(signal_t, (pad, pad), mode="replicate")
    smoothed = torch_f.conv1d(padded, kernel)
    return smoothed.view(-1).detach().cpu().numpy()
