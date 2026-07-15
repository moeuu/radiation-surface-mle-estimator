"""Visualization helpers for MLE surface maps and spectral residuals."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from .types import MLEEstimate


def plot_surface_map(
    estimate: MLEEstimate,
    isotope: str,
    *,
    output_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Render one isotope's patch density on the complete 3-D surface model."""
    if isotope not in estimate.isotope_names:
        raise KeyError(f"Unknown isotope: {isotope}")
    isotope_index = estimate.isotope_names.index(isotope)
    values = estimate.density_by_isotope[isotope_index]
    vertices = [patch.vertices_xyz for patch in estimate.patches]
    maximum = max(float(np.max(values, initial=0.0)), 1.0e-12)
    colors = plt.get_cmap("inferno")(np.clip(values / maximum, 0.0, 1.0))
    figure = plt.figure(figsize=(9.0, 7.0))
    axis = figure.add_subplot(111, projection="3d")
    collection = Poly3DCollection(
        vertices,
        facecolors=colors,
        edgecolors=(0.15, 0.15, 0.15, 0.25),
        linewidths=0.2,
    )
    axis.add_collection3d(collection)
    all_vertices = np.concatenate(vertices, axis=0)
    lower = np.min(all_vertices, axis=0)
    upper = np.max(all_vertices, axis=0)
    axis.set_xlim(float(lower[0]), float(upper[0]))
    axis.set_ylim(float(lower[1]), float(upper[1]))
    axis.set_zlim(float(lower[2]), float(upper[2]))
    axis.set_box_aspect(np.maximum(upper - lower, 1.0e-6))
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.set_title(f"{isotope} surface density [detector cps@1m/m²]")
    scalar = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=0.0, vmax=maximum), cmap="inferno"
    )
    figure.colorbar(scalar, ax=axis, shrink=0.65, label="detector cps@1m/m²")
    figure.tight_layout()
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return figure


def plot_spectral_residuals(
    observed_spectra: np.ndarray,
    estimate: MLEEstimate,
    energy_bin_edges_keV: np.ndarray,
    *,
    output_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot total observed, predicted, and residual spectra across measurements."""
    if estimate.predicted_spectra is None:
        raise ValueError("The estimate does not contain predicted spectra.")
    observed = np.asarray(observed_spectra, dtype=float)
    predicted = np.asarray(estimate.predicted_spectra, dtype=float)
    edges = np.asarray(energy_bin_edges_keV, dtype=float)
    if observed.shape != predicted.shape or edges.shape != (observed.shape[1] + 1,):
        raise ValueError("Observed, predicted, and energy-bin shapes do not align.")
    centers = 0.5 * (edges[:-1] + edges[1:])
    observed_total = np.sum(observed, axis=0)
    predicted_total = np.sum(predicted, axis=0)
    figure, (spectrum_axis, residual_axis) = plt.subplots(
        2, 1, figsize=(9.0, 6.5), sharex=True, height_ratios=(2.0, 1.0)
    )
    spectrum_axis.step(centers, observed_total, where="mid", label="Observed")
    spectrum_axis.step(centers, predicted_total, where="mid", label="Predicted")
    spectrum_axis.set_ylabel("Counts")
    spectrum_axis.legend()
    residual_axis.axhline(0.0, color="0.3", linewidth=0.8)
    residual_axis.step(
        centers, observed_total - predicted_total, where="mid", color="tab:red"
    )
    residual_axis.set_xlabel("Energy [keV]")
    residual_axis.set_ylabel("Residual")
    figure.tight_layout()
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return figure


__all__ = ["plot_spectral_residuals", "plot_surface_map"]
