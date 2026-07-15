"""Render a simple 3D schematic of a detector with two octant shield shells."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

from measurement.shielding import generate_octant_orientations, octant_index_from_normal
def _resolve_font_family(preferred: str, fallbacks: list[str]) -> str:
    """Return the first available font family name from preferred + fallbacks."""
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in [preferred] + fallbacks:
        if name in available:
            return name
    return preferred


mpl.rcParams["font.family"] = _resolve_font_family(
    "Times New Roman",
    ["Times", "DejaVu Serif"],
)

def _sphere_surface(
    radius: float,
    theta_range: Tuple[float, float],
    phi_range: Tuple[float, float],
    resolution: int,
    signs: Tuple[int, int, int] = (1, 1, 1),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a spherical surface patch.

    Args:
        radius: Sphere radius.
        theta_range: Polar angle range in radians.
        phi_range: Azimuthal angle range in radians.
        resolution: Number of samples per angular dimension.
        signs: (sx, sy, sz) applied to x, y, z to select an octant.

    Returns:
        (x, y, z) arrays for plotting.
    """
    theta = np.linspace(theta_range[0], theta_range[1], resolution)
    phi = np.linspace(phi_range[0], phi_range[1], resolution)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    x = radius * np.sin(theta_grid) * np.cos(phi_grid)
    y = radius * np.sin(theta_grid) * np.sin(phi_grid)
    z = radius * np.cos(theta_grid)
    sx, sy, sz = signs
    return sx * x, sy * y, sz * z


def _plot_detector_with_shells(
    ax: "plt.Axes",
    *,
    detector_radius: float,
    fe_radius: float,
    pb_radius: float,
    shell_thickness: float,
    fe_signs: Tuple[int, int, int],
    pb_signs: Tuple[int, int, int],
    resolution: int,
    top_text: str | None,
    bottom_text: str | None,
    top_fontsize: int,
    bottom_fontsize: int,
) -> None:
    """Draw a detector sphere with two 1/8 shell layers on an axis."""
    det_x, det_y, det_z = _sphere_surface(
        detector_radius,
        theta_range=(0.0, np.pi),
        phi_range=(0.0, 2.0 * np.pi),
        resolution=resolution,
    )
    ax.plot_surface(det_x, det_y, det_z, color="lightgray", alpha=0.35, linewidth=0.0)

    theta_oct = (0.0, np.pi / 2.0)
    phi_oct = (0.0, np.pi / 2.0)
    fe_inner = float(fe_radius)
    fe_outer = fe_inner + float(shell_thickness)
    pb_inner = float(pb_radius)
    pb_outer = pb_inner + float(shell_thickness)

    fe_x, fe_y, fe_z = _sphere_surface(
        fe_inner,
        theta_range=theta_oct,
        phi_range=phi_oct,
        resolution=resolution,
        signs=fe_signs,
    )
    fe_x_o, fe_y_o, fe_z_o = _sphere_surface(
        fe_outer,
        theta_range=theta_oct,
        phi_range=phi_oct,
        resolution=resolution,
        signs=fe_signs,
    )
    pb_x, pb_y, pb_z = _sphere_surface(
        pb_inner,
        theta_range=theta_oct,
        phi_range=phi_oct,
        resolution=resolution,
        signs=pb_signs,
    )
    pb_x_o, pb_y_o, pb_z_o = _sphere_surface(
        pb_outer,
        theta_range=theta_oct,
        phi_range=phi_oct,
        resolution=resolution,
        signs=pb_signs,
    )
    ax.plot_surface(fe_x, fe_y, fe_z, color="#d95f02", alpha=0.25, linewidth=0.0)
    ax.plot_surface(fe_x_o, fe_y_o, fe_z_o, color="#d95f02", alpha=0.6, linewidth=0.0)
    ax.plot_surface(pb_x, pb_y, pb_z, color="#7570b3", alpha=0.2, linewidth=0.0)
    ax.plot_surface(pb_x_o, pb_y_o, pb_z_o, color="#7570b3", alpha=0.4, linewidth=0.0)

    lim = max(pb_outer, detector_radius) * 1.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    plane_lin = np.linspace(-lim, lim, 2)
    plane_x, plane_y = np.meshgrid(plane_lin, plane_lin)
    plane_z = np.zeros_like(plane_x)
    ax.plot_surface(plane_x, plane_y, plane_z, color="black", alpha=0.08, linewidth=0.0)
    ax.plot_surface(plane_x, plane_z, plane_y, color="black", alpha=0.08, linewidth=0.0)
    ax.plot_surface(plane_z, plane_x, plane_y, color="black", alpha=0.08, linewidth=0.0)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_axis_off()
    ax.view_init(elev=20, azim=35)
    if top_text is not None:
        ax.text2D(0.5, 0.94, top_text, transform=ax.transAxes, ha="center", va="center", fontsize=top_fontsize)
    if bottom_text is not None:
        ax.text2D(
            0.5,
            0.06,
            bottom_text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=bottom_fontsize,
        )


def render_detector_with_octant_shells(
    output_path: Path,
    *,
    detector_radius: float = 1.0,
    fe_radius: float = 1.0,
    pb_radius: float = 1.2,
    shell_thickness: float = 0.2,
    fe_signs: Tuple[int, int, int] = (1, 1, 1),
    pb_signs: Tuple[int, int, int] = (1, 1, 1),
    resolution: int = 60,
    expected_ig: float = 0.0,
) -> None:
    """
    Render a detector sphere with two 1/8 spherical shells and save the image.

    Args:
        output_path: File path to write the figure.
        detector_radius: Radius of the detector sphere.
        fe_radius: Inner radius for the Fe octant shell.
        pb_radius: Inner radius for the Pb octant shell.
        shell_thickness: Radial thickness of each shell.
        fe_signs: Octant signs for the Fe shell.
        pb_signs: Octant signs for the Pb shell.
        resolution: Surface resolution for mesh generation.
        expected_ig: Expected IG value to display.
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    fe_idx = octant_index_from_normal(np.asarray(fe_signs, dtype=float))
    pb_idx = octant_index_from_normal(np.asarray(pb_signs, dtype=float))
    top_text = f"Fe_idex={fe_idx}, Pb_idx={pb_idx}"
    bottom_text = f"Expected IG={float(expected_ig):.4g}"
    _plot_detector_with_shells(
        ax,
        detector_radius=detector_radius,
        fe_radius=fe_radius,
        pb_radius=pb_radius,
        shell_thickness=shell_thickness,
        fe_signs=fe_signs,
        pb_signs=pb_signs,
        resolution=resolution,
        top_text=top_text,
        bottom_text=bottom_text,
        top_fontsize=10,
        bottom_fontsize=10,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def render_octant_grid(
    output_path: Path,
    *,
    detector_radius: float = 1.0,
    fe_radius: float = 1.0,
    pb_radius: float = 1.2,
    shell_thickness: float = 0.2,
    resolution: int = 30,
    grid_size: int = 8,
    ig_scores: np.ndarray | None = None,
    highlight_max: bool = True,
    highlight_idx: tuple[int, int] | None = None,
    font_size: int = 12,
) -> None:
    """
    Render an 8x8 grid of Fe/Pb octant combinations.

    Args:
        output_path: File path to write the figure.
        detector_radius: Radius of the detector sphere.
        fe_radius: Inner radius for the Fe octant shell.
        pb_radius: Inner radius for the Pb octant shell.
        shell_thickness: Radial thickness of each shell.
        resolution: Surface resolution for mesh generation.
        grid_size: Number of octants per axis (expected 8).
        ig_scores: Optional (grid_size, grid_size) array of expected IG values.
        highlight_max: Whether to outline the maximum IG cell in red.
        highlight_idx: Explicit (fe_idx, pb_idx) cell to outline in red.
        font_size: Font size for top/bottom captions.
    """
    normals = generate_octant_orientations()
    if ig_scores is None:
        ig_scores = np.zeros((grid_size, grid_size), dtype=float)
    else:
        ig_scores = np.asarray(ig_scores, dtype=float)
    max_idx = None
    if highlight_idx is not None:
        max_idx = (int(highlight_idx[0]), int(highlight_idx[1]))
    elif highlight_max and ig_scores.size:
        flat_idx = int(np.argmax(ig_scores))
        max_idx = (flat_idx // grid_size, flat_idx % grid_size)
    fig = plt.figure(figsize=(18, 18))
    axes: list[list[plt.Axes]] = [[None for _ in range(grid_size)] for _ in range(grid_size)]
    for fe_idx in range(grid_size):
        fe_signs = tuple(int(np.sign(val)) for val in normals[fe_idx])
        for pb_idx in range(grid_size):
            pb_signs = tuple(int(np.sign(val)) for val in normals[pb_idx])
            ax = fig.add_subplot(grid_size, grid_size, fe_idx * grid_size + pb_idx + 1, projection="3d")
            top_text = f"Fe_idex={fe_idx}, Pb_idx={pb_idx}"
            ig_val = float(ig_scores[fe_idx, pb_idx]) if ig_scores.size else 0.0
            bottom_text = f"Expected IG={ig_val:.4g}"
            _plot_detector_with_shells(
                ax,
                detector_radius=detector_radius,
                fe_radius=fe_radius,
                pb_radius=pb_radius,
                shell_thickness=shell_thickness,
                fe_signs=fe_signs,
                pb_signs=pb_signs,
                resolution=resolution,
                top_text=top_text,
                bottom_text=bottom_text,
                top_fontsize=font_size,
                bottom_fontsize=font_size,
            )
            axes[fe_idx][pb_idx] = ax
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995, wspace=0.0, hspace=0.0)
    for row in range(grid_size):
        for col in range(grid_size):
            ax = axes[row][col]
            if ax is None:
                continue
            pos = ax.get_position()
            is_max = max_idx is not None and (row, col) == max_idx
            rect = mpatches.Rectangle(
                (pos.x0, pos.y0),
                pos.width,
                pos.height,
                fill=False,
                edgecolor="red" if is_max else "black",
                linewidth=1.6 if is_max else 0.4,
                transform=fig.transFigure,
                zorder=50,
            )
            fig.add_artist(rect)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Entry point for rendering the detector and shield shells."""
    output_path = Path("results") / "IG" / "ig_shield_grid.png"
    rng = np.random.default_rng(7)
    ig_scores = rng.random((8, 8))
    best_idx = np.unravel_index(int(np.argmax(ig_scores)), ig_scores.shape)
    render_octant_grid(
        output_path,
        ig_scores=ig_scores,
        highlight_idx=(int(best_idx[0]), int(best_idx[1])),
        highlight_max=False,
        font_size=12,
    )


if __name__ == "__main__":
    main()
