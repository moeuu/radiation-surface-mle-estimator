"""Generate isotope-wise count sequences from unfolded spectra and apply weighted aggregation."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from spectrum.baseline import asymmetric_least_squares
from spectrum.library import Nuclide
from spectrum.smoothing import gaussian_smooth


def _cuda_available() -> bool:
    """Return True if torch with CUDA is available."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _compute_weight_matrix(
    energy_axis_keV: NDArray[np.float64],
    library: Dict[str, Nuclide],
    window_keV: float,
) -> Tuple[List[str], NDArray[np.float64]]:
    """
    Precompute per-isotope weight vectors for line windows.

    Returns:
        (isotope_names, weight_matrix) where weight_matrix shape = (H, B)
    """
    energy_axis_keV = np.asarray(energy_axis_keV, dtype=float)
    iso_names = list(library.keys())
    num_bins = energy_axis_keV.size
    weight_matrix = np.zeros((len(iso_names), num_bins), dtype=float)
    total_intensity: Dict[str, float] = {
        name: sum(line.intensity for line in nuclide.lines) for name, nuclide in library.items()
    }
    for j, iso in enumerate(iso_names):
        nuclide = library[iso]
        total_int = total_intensity.get(iso, 0.0)
        if total_int <= 0.0:
            continue
        for line in nuclide.lines:
            weight = line.intensity / total_int
            mask = np.abs(energy_axis_keV - line.energy_keV) <= window_keV
            if np.any(mask):
                weight_matrix[j, mask] += weight
    return iso_names, weight_matrix


def _dead_time_scale(total_counts: float, live_time_s: float, dead_time_s: float) -> float:
    """Return a global scale factor using a non-paralyzable dead-time model."""
    if dead_time_s <= 0.0 or live_time_s <= 0.0:
        return 1.0
    m_rate = total_counts / live_time_s
    return 1.0 / max(1.0 - m_rate * dead_time_s, 1e-9)


def _apply_preprocess(
    spectrum: NDArray[np.float64],
    live_time_s: float,
    dead_time_s: float,
    smooth_sigma_bins: float | None,
    subtract_baseline: bool,
) -> NDArray[np.float64]:
    """Apply dead-time correction, smoothing, and baseline subtraction."""
    corrected = spectrum.astype(float)
    scale = _dead_time_scale(corrected.sum(), live_time_s, dead_time_s)
    corrected *= scale
    if smooth_sigma_bins is not None and smooth_sigma_bins > 0.0:
        corrected = gaussian_smooth(corrected, sigma_bins=smooth_sigma_bins)
    if subtract_baseline:
        base = asymmetric_least_squares(corrected, lam=1e4, p=0.01, niter=10)
        corrected = np.clip(corrected - base, a_min=0.0, a_max=None)
    return corrected


def build_isotope_count_sequence(
    spectra: Iterable[NDArray[np.float64]],
    energy_axis_keV: NDArray[np.float64],
    library: Dict[str, Nuclide],
    live_time_s: float | Sequence[float],
    dead_time_s: float = 0.0,
    window_keV: float = 5.0,
    smooth_sigma_bins: float | None = None,
    subtract_baseline: bool = True,
    use_gpu: bool | None = None,
) -> Tuple[List[str], NDArray[np.float64]]:
    """
    Build isotope-wise count sequences z_k from short-time spectra.

    Args:
        spectra: Time-series spectra (iterable of channel-count arrays).
        energy_axis_keV: Energy axis per channel (keV).
        library: Nuclide library.
        live_time_s: Live time (single value or per-spectrum list).
        dead_time_s: Dead time in seconds.
        window_keV: Integration window (±window_keV) around each line.
        smooth_sigma_bins: Smoothing sigma in bins (None or 0 disables).
        subtract_baseline: Whether to subtract the baseline.
        use_gpu: If True, use CUDA for the final matrix multiply when available.

    Returns:
        (isotope_names, counts_matrix) where counts_matrix shape = (T, H)
    """
    energy_axis_keV = np.asarray(energy_axis_keV, dtype=float)
    spectra_list = list(spectra)
    iso_names, weight_matrix = _compute_weight_matrix(energy_axis_keV, library, window_keV)
    live_times = (
        [float(live_time_s)] * len(spectra_list)
        if isinstance(live_time_s, (int, float))
        else [float(v) for v in live_time_s]
    )
    if len(spectra_list) != len(live_times):
        raise ValueError("Number of spectra and live_time_s entries must match")

    processed_stack = np.zeros((len(spectra_list), energy_axis_keV.size), dtype=float)
    for idx, (spec, lt) in enumerate(zip(spectra_list, live_times)):
        processed_stack[idx] = _apply_preprocess(
            np.asarray(spec, dtype=float),
            live_time_s=lt,
            dead_time_s=dead_time_s,
            smooth_sigma_bins=smooth_sigma_bins,
            subtract_baseline=subtract_baseline,
        )

    if use_gpu is None:
        use_gpu = _cuda_available()
    if use_gpu:
        try:
            import torch
        except ImportError:
            use_gpu = False
        else:
            if not torch.cuda.is_available():
                use_gpu = False
    if use_gpu:
        device = torch.device("cuda")
        spectra_t = torch.as_tensor(processed_stack, device=device, dtype=torch.float64)
        weights_t = torch.as_tensor(weight_matrix, device=device, dtype=torch.float64)
        counts_t = spectra_t @ weights_t.T
        counts_matrix = counts_t.detach().cpu().numpy()
    else:
        counts_matrix = processed_stack @ weight_matrix.T
    return iso_names, np.asarray(counts_matrix, dtype=float)
