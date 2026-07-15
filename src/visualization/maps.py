"""Render estimated source locations and strengths as heatmaps overlaid on trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from numpy.typing import NDArray

from measurement.continuous_kernels import ContinuousKernel, geometric_term
from measurement.kernels import ShieldParams
from measurement.shielding import octant_index_from_normal
from measurement.obstacles import ObstacleGrid

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    _TORCH_AVAILABLE = False


def _make_grid(
    env_bounds: Tuple[float, float, float, float], resolution: float
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create 2D grid (X,Y) covering the environment bounds."""
    xmin, xmax, ymin, ymax = env_bounds
    xs = np.arange(xmin, xmax + resolution, resolution)
    ys = np.arange(ymin, ymax + resolution, resolution)
    X, Y = np.meshgrid(xs, ys)
    return X, Y


def _expected_intensity_at_points(
    grid_points: NDArray[np.float64],
    estimates: Dict[str, Dict[str, NDArray[np.float64]]],
    shield_normal: Optional[NDArray[np.float64]] = None,
    mu_by_isotope: Optional[Dict[str, object]] = None,
    shield_params: Optional[ShieldParams] = None,
    use_gpu: bool | None = None,
    gpu_device: str = "cuda",
    gpu_dtype: str = "float32",
) -> NDArray[np.float64]:
    """
    Compute expected intensity (sum over isotopes) at given grid_points (N,2).

    estimates: dict[iso] -> {"positions": (M,3), "strengths": (M,)}
    GPU acceleration is enabled when use_gpu=True and torch supports the device.
    """
    kernel = ContinuousKernel(
        mu_by_isotope=mu_by_isotope,
        shield_params=shield_params or ShieldParams(),
        use_gpu=bool(use_gpu) if use_gpu is not None else False,
        gpu_device=gpu_device,
        gpu_dtype=gpu_dtype,
    )
    orient_idx = octant_index_from_normal(shield_normal) if shield_normal is not None else None
    if (
        not use_gpu
        or not _TORCH_AVAILABLE
        or torch is None
        or (gpu_device.startswith("cuda") and not torch.cuda.is_available())
    ):
        intensities = np.zeros(grid_points.shape[0], dtype=float)
        detector_z = 0.0  # assume ground plane rendering
        for iso, est in estimates.items():
            pos3d = est.get("positions", np.zeros((0, 3)))
            strengths = est.get("strengths", np.zeros(0))
            for src, q in zip(pos3d, strengths):
                src = np.asarray(src, dtype=float)
                for i, (x, y) in enumerate(grid_points):
                    detector = np.array([x, y, detector_z], dtype=float)
                    geom = geometric_term(detector, src)
                    att = 1.0
                    if orient_idx is not None:
                        att = kernel.attenuation_factor(
                            isotope=iso,
                            source_pos=src,
                            detector_pos=detector,
                            orient_idx=orient_idx,
                        )
                    intensities[i] += geom * att * float(q)
        return intensities
    if gpu_dtype == "float32":
        dtype = torch.float32
    elif gpu_dtype == "float64":
        dtype = torch.float64
    else:
        raise ValueError(f"Unsupported torch dtype: {gpu_dtype}")
    device = torch.device(gpu_device)
    detector_z = torch.zeros((grid_points.shape[0], 1), device=device, dtype=dtype)
    grid_t = torch.as_tensor(grid_points, device=device, dtype=dtype)
    detectors = torch.cat([grid_t, detector_z], dim=1)
    intensities_t = torch.zeros(grid_t.shape[0], device=device, dtype=dtype)
    if orient_idx is not None:
        normal = torch.as_tensor(kernel.orientations[orient_idx], device=device, dtype=dtype)
    tol = 1e-6
    tol_t = torch.as_tensor(tol, device=device, dtype=dtype)
    for iso, est in estimates.items():
        pos3d = est.get("positions", np.zeros((0, 3)))
        strengths = est.get("strengths", np.zeros(0))
        if pos3d.size == 0 or strengths.size == 0:
            continue
        sources_t = torch.as_tensor(np.asarray(pos3d, dtype=float), device=device, dtype=dtype)
        strengths_t = torch.as_tensor(np.asarray(strengths, dtype=float), device=device, dtype=dtype)
        mu_fe, mu_pb = kernel._mu_values(isotope=iso)
        for src, strength in zip(sources_t, strengths_t):
            direction = detectors - src
            dist = torch.linalg.norm(direction, dim=1)
            dist = torch.where(dist <= tol, torch.full_like(dist, tol), dist)
            dir_unit = direction / dist.unsqueeze(1)
            geom = 1.0 / (dist**2)
            if orient_idx is None:
                att = torch.ones_like(dist)
            else:
                detector_to_source_unit = -dir_unit
                blocked = kernel._rotated_octant_blocked_mask_torch(detector_to_source_unit, orient_idx, tol)
                cos_val = torch.clamp(torch.sum(dir_unit * normal, dim=1), 0.0, 1.0)
                L_fe, L_pb = kernel._shield_path_lengths_torch(
                    direction=direction,
                    blocked_fe=blocked,
                    blocked_pb=blocked,
                    cos_fe=cos_val,
                    cos_pb=cos_val,
                    tol_t=tol_t,
                    device=device,
                    dtype=dtype,
                )
                tau_fe = float(mu_fe) * L_fe
                tau_pb = float(mu_pb) * L_pb
                tau_obstacle = torch.zeros_like(tau_fe)
                buildup = kernel._buildup_factor_torch(tau_fe, tau_pb, tau_obstacle)
                att = torch.clamp(torch.exp(-(tau_fe + tau_pb)) * buildup, max=1.0)
            intensities_t = intensities_t + geom * att * strength
    return intensities_t.detach().cpu().numpy()


def _draw_obstacle_cells(ax: plt.Axes, obstacles: ObstacleGrid) -> None:
    """Draw obstacle grid cells as black rectangles."""
    patches: list[Rectangle] = []
    for x0, x1, y0, y1 in obstacles.blocked_bounds():
        patches.append(Rectangle((x0, y0), x1 - x0, y1 - y0))
    if not patches:
        return
    collection = PatchCollection(
        patches,
        facecolor="black",
        edgecolor="none",
        alpha=0.85,
        label="Obstacles",
    )
    ax.add_collection(collection)


def render_heatmap(
    env_bounds: Tuple[float, float, float, float],
    trajectory: NDArray[np.float64],
    estimates: Dict[str, Dict[str, NDArray[np.float64]]],
    resolution: float = 0.5,
    shield_normal: Optional[NDArray[np.float64]] = None,
    obstacles: Optional[Iterable[Tuple[float, float]] | ObstacleGrid] = None,
    cmap: str = "inferno",
    output_path: Optional[Path] = None,
    mu_by_isotope: Optional[Dict[str, object]] = None,
    shield_params: Optional[ShieldParams] = None,
    use_gpu: bool | None = None,
    gpu_device: str = "cuda",
    gpu_dtype: str = "float32",
) -> plt.Figure:
    """
    Render a 2D heatmap of expected radiation intensity over the environment.

    Args:
        env_bounds: (xmin, xmax, ymin, ymax)
        trajectory: (T,2) robot path
        estimates: dict[iso] -> {"positions": (M,3), "strengths": (M,)}
        resolution: grid spacing (m)
        shield_normal: optional shielding orientation to model attenuation
        obstacles: iterable of (x,y) points or ObstacleGrid to mark obstacles
        cmap: matplotlib colormap name
        output_path: if provided, save the figure to this path
        mu_by_isotope: optional per-isotope attenuation coefficients for Fe/Pb
        shield_params: optional shield thickness/attenuation parameters
        use_gpu: enable CUDA acceleration for grid intensity evaluation
        gpu_device: torch device string (e.g., "cuda", "cuda:0", "cpu")
        gpu_dtype: torch dtype string ("float32" or "float64")
    """
    X, Y = _make_grid(env_bounds, resolution)
    points = np.column_stack([X.ravel(), Y.ravel()])
    intensities = _expected_intensity_at_points(
        points,
        estimates,
        shield_normal=shield_normal,
        mu_by_isotope=mu_by_isotope,
        shield_params=shield_params,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
        gpu_dtype=gpu_dtype,
    )
    Z = intensities.reshape(X.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        Z,
        extent=env_bounds,
        origin="lower",
        cmap=cmap,
        aspect="auto",
    )
    plt.colorbar(im, ax=ax, label="Expected intensity (a.u.)")
    if trajectory is not None and trajectory.size:
        ax.plot(trajectory[:, 0], trajectory[:, 1], "-o", color="cyan", label="Trajectory")
    if isinstance(obstacles, ObstacleGrid):
        _draw_obstacle_cells(ax, obstacles)
    elif obstacles:
        obs = np.array(list(obstacles))
        ax.scatter(obs[:, 0], obs[:, 1], color="black", marker="s", label="Obstacles")
    # Plot estimated sources
    for iso, est in estimates.items():
        pos3d = est.get("positions", np.zeros((0, 3)))
        if pos3d.size == 0:
            continue
        ax.scatter(pos3d[:, 0], pos3d[:, 1], s=40, label=f"{iso} est.")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Estimated radiation heatmap")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
    return fig
