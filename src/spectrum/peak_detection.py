"""Simple peak detection utilities for smoothed spectra."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from spectrum.smoothing import gaussian_smooth as _smoothing_gaussian_smooth

@dataclass(frozen=True)
class LineEvidence:
    """Summary statistics for a line window and its local sidebands."""

    gross: float
    background: float
    net: float
    snr: float


MIN_RESOLUTION_KEV = 0.5


def gaussian_smooth(
    spectrum: NDArray[np.float64],
    sigma_bins: float,
    *,
    use_gpu: bool | None = None,
    gpu_device: str = "cuda",
    gpu_dtype: str = "float32",
) -> NDArray[np.float64]:
    """
    Smooth a spectrum via discrete Gaussian convolution (Eq. 2.18-2.19).

    Args:
        spectrum: Input spectrum (counts per bin).
        sigma_bins: Gaussian sigma in bins.
        use_gpu: Unused, kept for API compatibility.
        gpu_device: Unused, kept for API compatibility.
        gpu_dtype: Unused, kept for API compatibility.

    Returns:
        Smoothed spectrum with the same length as the input.
    """
    return _smoothing_gaussian_smooth(
        spectrum,
        sigma_bins=sigma_bins,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
        gpu_dtype=gpu_dtype,
    )


def sigma_E_keV(E_keV: float, a: float, b: float) -> float:
    """
    Return detector resolution sigma(E) = a * sqrt(E) + b (keV), with a floor.

    Args:
        E_keV: Energy in keV.
        a: Resolution coefficient for sqrt(E).
        b: Resolution offset in keV.

    Returns:
        Sigma in keV, clamped to a small positive minimum.
    """
    energy = max(float(E_keV), 0.0)
    sigma = float(a) * np.sqrt(energy) + float(b)
    return float(max(sigma, MIN_RESOLUTION_KEV))


def detect_peaks(signal: NDArray[np.float64], prominence: float = 0.01, distance: int = 5) -> NDArray[np.int_]:
    """
    Detect peaks using scipy's find_peaks with basic parameters.

    Args:
        signal: Smoothed spectrum.
        prominence: Minimum prominence relative to max to keep a peak.
        distance: Minimum separation between peaks in bins.

    Returns:
        Indices of detected peaks.
    """
    if signal.size == 0:
        return np.array([], dtype=int)
    prom = prominence * float(signal.max() if signal.max() > 0 else 1.0)
    peaks, _ = find_peaks(signal, prominence=prom, distance=distance)
    return peaks.astype(int)


def line_window_evidence(
    energy_keV: NDArray[np.float64],
    counts: NDArray[np.float64],
    line_keV: float,
    half_window_keV: float,
    sideband_keV: float,
) -> LineEvidence:
    """
    Compute line evidence using a peak window and adjacent sidebands.

    Args:
        energy_keV: Energy axis in keV.
        counts: Spectrum counts (optionally smoothed).
        line_keV: Line center energy.
        half_window_keV: Half-width of the peak window.
        sideband_keV: Width of each sideband window.

    Returns:
        LineEvidence with gross, background, net, and SNR.
    """
    if counts.size == 0 or energy_keV.size == 0:
        return LineEvidence(gross=0.0, background=0.0, net=0.0, snr=0.0)
    peak_mask = (energy_keV >= line_keV - half_window_keV) & (energy_keV <= line_keV + half_window_keV)
    left_mask = (energy_keV >= line_keV - half_window_keV - sideband_keV) & (
        energy_keV < line_keV - half_window_keV
    )
    right_mask = (energy_keV > line_keV + half_window_keV) & (
        energy_keV <= line_keV + half_window_keV + sideband_keV
    )
    gross = float(np.sum(counts[peak_mask]))
    side_sum = float(np.sum(counts[left_mask]) + np.sum(counts[right_mask]))
    n_peak = int(np.sum(peak_mask))
    n_side = int(np.sum(left_mask) + np.sum(right_mask))
    background = side_sum * (n_peak / max(n_side, 1))
    net = gross - background
    snr = net / np.sqrt(max(gross + background, 1.0))
    return LineEvidence(gross=gross, background=background, net=net, snr=snr)


def has_peak_near(
    peak_energies_keV: NDArray[np.float64],
    line_keV: float,
    tolerance_keV: float,
) -> bool:
    """
    Return True when a detected peak is within tolerance of a line energy.

    Args:
        peak_energies_keV: Peak energies in keV.
        line_keV: Line center energy.
        tolerance_keV: Allowed mismatch.
    """
    if peak_energies_keV.size == 0:
        return False
    return bool(np.any(np.abs(peak_energies_keV - line_keV) <= tolerance_keV))
