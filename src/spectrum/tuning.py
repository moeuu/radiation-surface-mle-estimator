"""Helpers for tuning spectrum parameters via reference spectra."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict
from numpy.typing import NDArray

from measurement.model import EnvironmentConfig, PointSource
from spectrum.pipeline import SpectralDecomposer
import spectrum.response_matrix as response_matrix
import spectrum.pipeline as pipeline
import matplotlib.pyplot as plt


def _standard_sources() -> list[PointSource]:
    """Return the standard point-source set used in main.py."""
    return [
        PointSource("Cs-137", position=(5.3, 10.0, 5.0), intensity_cps_1m=20000.0),
        PointSource("Co-60", position=(4.7, 10.6, 5.0), intensity_cps_1m=20000.0),
        PointSource("Eu-154", position=(5.0, 9.4, 4.6), intensity_cps_1m=20000.0),
    ]


def simulate_reference_spectrum(
    continuum_to_peak: float,
    backscatter_fraction: float,
    background_rate_cps: float,
    rng_seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Simulate a spectrum in the standard scenario and return (energy_axis, spectrum).

    - Cs-137 / Co-60 / Eu-154 (20 kcps at 1 m each)
    - Environment: 10 x 20 x 10 m, detector near the center
    - 120 s acquisition time
    """
    # Save current values.
    prev_cont = response_matrix.COMPTON_CONTINUUM_TO_PEAK
    prev_back = response_matrix.BACKSCATTER_FRACTION
    prev_bg_rate = pipeline.BACKGROUND_RATE_CPS
    prev_bg_rate_alias = pipeline.BACKGROUND_COUNTS_PER_SECOND
    try:
        # Override parameters for the simulation run.
        response_matrix.COMPTON_CONTINUUM_TO_PEAK = continuum_to_peak
        response_matrix.BACKSCATTER_FRACTION = backscatter_fraction
        pipeline.BACKGROUND_RATE_CPS = background_rate_cps
        pipeline.BACKGROUND_COUNTS_PER_SECOND = background_rate_cps

        decomposer = SpectralDecomposer()
        env = EnvironmentConfig(size_x=10.0, size_y=20.0, size_z=10.0)
        rng = np.random.default_rng(rng_seed)
        spectrum, _ = decomposer.simulate_spectrum(
            _standard_sources(),
            environment=env,
            acquisition_time=120.0,
            rng=rng,
            dead_time_s=0.0,
        )
        return decomposer.energy_axis, spectrum
    finally:
        # Restore parameters.
        response_matrix.COMPTON_CONTINUUM_TO_PEAK = prev_cont
        response_matrix.BACKSCATTER_FRACTION = prev_back
        pipeline.BACKGROUND_RATE_CPS = prev_bg_rate
        pipeline.BACKGROUND_COUNTS_PER_SECOND = prev_bg_rate_alias


@dataclass
class SpectrumQuality:
    """Simple spectrum quality metrics for tuning."""

    passes: bool
    mean_L: float
    mean_M: float
    mean_H: float
    global_max_energy_keV: float
    peak_prominence: Dict[str, float]


def peak_contrast(
    energy_keV: NDArray[np.float64],
    counts: NDArray[np.float64],
    E0: float,
    peak_half_width_keV: float = 10.0,
    sideband_inner_keV: float = 30.0,
    sideband_outer_keV: float = 60.0,
) -> float:
    """
    Compute peak contrast around a target energy.
    contrast = peak_max / sideband_mean
    """
    E = energy_keV
    C = counts
    peak_mask = (E >= E0 - peak_half_width_keV) & (E <= E0 + peak_half_width_keV)
    side_mask = ((E >= E0 - sideband_outer_keV) & (E <= E0 - sideband_inner_keV)) | (
        (E >= E0 + sideband_inner_keV) & (E <= E0 + sideband_outer_keV)
    )
    if not np.any(peak_mask):
        return 0.0
    peak_max = float(C[peak_mask].max())
    side_vals = C[side_mask]
    side_mean = float(side_vals.mean()) if side_vals.size > 0 else 0.0
    if side_mean <= 0.0:
        return 0.0
    return peak_max / side_mean


def evaluate_spectrum_quality(
    energy_keV: NDArray[np.float64],
    spectrum: NDArray[np.float64],
) -> SpectrumQuality:
    """
    Evaluate spectrum quality using a CeBr3-like shape and peak prominence.
    """
    E = energy_keV
    C = spectrum

    mask_L = (E >= 80.0) & (E <= 200.0)
    mask_M = (E >= 400.0) & (E <= 800.0)
    mask_H = (E >= 1200.0) & (E <= 1600.0)

    mean_L = float(C[mask_L].mean())
    mean_M = float(C[mask_M].mean())
    mean_H = float(C[mask_H].mean())

    cond_shape_1 = mean_L >= 0.6 * mean_M
    cond_shape_2 = mean_M >= 1.2 * mean_H

    idx_max = int(np.argmax(C))
    global_max_energy_keV = float(E[idx_max])
    cond_max = (80.0 <= global_max_energy_keV <= 200.0) or (mean_L >= 0.4 * mean_M)

    prom: Dict[str, float] = {}
    prom["Eu-154_1275"] = peak_contrast(E, C, 1274.5, peak_half_width_keV=15.0, sideband_inner_keV=40.0, sideband_outer_keV=90.0)
    prom["Cs-137_662"] = peak_contrast(E, C, 662.0, peak_half_width_keV=12.0, sideband_inner_keV=40.0, sideband_outer_keV=90.0)
    prom["Co-60_1173"] = peak_contrast(E, C, 1173.0, peak_half_width_keV=15.0, sideband_inner_keV=50.0, sideband_outer_keV=100.0)
    prom["Co-60_1332"] = peak_contrast(E, C, 1332.0, peak_half_width_keV=15.0, sideband_inner_keV=50.0, sideband_outer_keV=100.0)

    cond_peak_eu = prom["Eu-154_1275"] >= 1.2
    cond_peak_cs137 = prom["Cs-137_662"] >= 1.5
    cond_peak_co60_1 = prom["Co-60_1173"] >= 1.3
    cond_peak_co60_2 = prom["Co-60_1332"] >= 1.3

    passes = (
        cond_shape_1
        and cond_shape_2
        and cond_max
        and cond_peak_eu
        and cond_peak_cs137
        and cond_peak_co60_1
        and cond_peak_co60_2
    )

    return SpectrumQuality(
        passes=passes,
        mean_L=mean_L,
        mean_M=mean_M,
        mean_H=mean_H,
        global_max_energy_keV=global_max_energy_keV,
        peak_prominence=prom,
    )


def grid_search_parameters(rng_seed: int = 0) -> tuple[float, float, float, SpectrumQuality]:
    """
    Run a coarse grid search and return the best parameter set.
    """
    continuum_grid = [0.4, 0.7, 1.0, 1.5]
    backscatter_grid = [0.03, 0.05, 0.08, 0.10]
    background_grid = [0.0, 5.0, 10.0, 15.0]

    best_quality: SpectrumQuality | None = None
    best_params: tuple[float, float, float] | None = None

    for c in continuum_grid:
        for b in backscatter_grid:
            for bg in background_grid:
                E, spectrum = simulate_reference_spectrum(
                    continuum_to_peak=c,
                    backscatter_fraction=b,
                    background_rate_cps=bg,
                    rng_seed=rng_seed,
                )
                quality = evaluate_spectrum_quality(E, spectrum)
                if quality.passes:
                    return c, b, bg, quality
                # Keep the best-scoring parameters as a fallback.
                shape_score = (quality.mean_L / max(quality.mean_M, 1e-6)) + (
                    quality.mean_M / max(quality.mean_H, 1e-6)
                )
                min_prom = min(quality.peak_prominence.values()) if quality.peak_prominence else 0.0
                score = shape_score + 0.5 * min_prom
                if best_quality is None:
                    best_quality = quality
                    best_params = (c, b, bg, score)
                else:
                    _, _, _, best_score = best_params  # type: ignore
                    if score > best_score:
                        best_quality = quality
                        best_params = (c, b, bg, score)

    if best_quality is None or best_params is None:
        # Fallback (should not happen, but keep a safe default).
        return continuum_grid[0], backscatter_grid[0], background_grid[0], evaluate_spectrum_quality(
            *simulate_reference_spectrum(
                continuum_to_peak=continuum_grid[0],
                backscatter_fraction=backscatter_grid[0],
                background_rate_cps=background_grid[0],
                rng_seed=0,
            )
        )
    c_best, b_best, bg_best, _ = best_params  # type: ignore
    return c_best, b_best, bg_best, best_quality


def print_grid_search_result() -> None:
    """Print grid-search results in a readable format."""
    c, b, bg, q = grid_search_parameters()
    print("Best parameters:")
    print(f"  continuum_to_peak:   {c}")
    print(f"  backscatter_fraction:{b}")
    print(f"  background_rate_cps: {bg}")
    print("Quality metrics:")
    print(f"  passes: {q.passes}")
    print(f"  mean_L: {q.mean_L:.3f}, mean_M: {q.mean_M:.3f}, mean_H: {q.mean_H:.3f}")
    print(f"  global_max_energy_keV: {q.global_max_energy_keV:.1f}")
    for key, val in q.peak_prominence.items():
        print(f"  peak {key}: {val:.3f}")


def get_best_parameters() -> tuple[float, float, float, SpectrumQuality]:
    """Run the grid search with defaults and return the best parameters."""
    return grid_search_parameters(rng_seed=0)


def plot_cebr3_reference_spectrum() -> None:
    """
    Plot and save a tuned CeBr3-like reference spectrum.

    - Label peaks near 662 and 1332 keV
    - Save to results/spectrum/cebr3_reference.png
    """
    c, b, bg, _ = get_best_parameters()
    E, spec = simulate_reference_spectrum(
        continuum_to_peak=c, backscatter_fraction=b, background_rate_cps=bg, rng_seed=0
    )
    from pathlib import Path
    results_dir = Path(__file__).resolve().parents[2] / "results" / "spectrum"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(E, spec, label="CeBr3-like reference")
    for energy, label in [(662.0, "Cs-137"), (1332.0, "Co-60"), (1173.0, "Co-60")]:
        ax.axvline(energy, color="r", linestyle="--", alpha=0.5)
        ax.text(energy + 5, max(spec) * 0.05, label, rotation=90, va="bottom", fontsize=8)
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")
    ax.set_title("CeBr3-like reference spectrum")
    ax.legend()
    fig.tight_layout()
    out_path = results_dir / "cebr3_reference.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved CeBr3 reference spectrum to {out_path}")


if __name__ == "__main__":
    print_grid_search_result()
