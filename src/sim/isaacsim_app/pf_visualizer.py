"""Author PF particle and estimate markers into an Isaac Sim stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sim.isaacsim_app.stage_backend import StageBackend


ISOTOPE_COLORS: dict[str, tuple[float, float, float]] = {
    "Cs-137": (1.0, 0.05, 0.05),
    "Co-60": (0.05, 0.45, 1.0),
    "Eu-154": (0.0, 0.75, 0.25),
    "Eu-155": (0.0, 0.75, 0.25),
}


@dataclass(frozen=True)
class PFSceneVisualizationConfig:
    """Collect visual-only PF marker settings for Isaac Sim."""

    enabled: bool = True
    max_particles_per_isotope: int = 800
    particle_radius_m: float = 0.025
    estimate_radius_m: float = 0.13
    estimate_cross_size_m: float = 0.35
    estimate_cross_width_m: float = 0.035
    min_weight_fraction: float = 0.0

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "PFSceneVisualizationConfig":
        """Build a config from an application config mapping."""
        payload = {} if data is None else dict(data)
        return cls(
            enabled=bool(payload.get("show_pf_particles", payload.get("pf_visualization_enabled", True))),
            max_particles_per_isotope=max(
                0,
                int(payload.get("pf_visual_max_particles_per_isotope", 800)),
            ),
            particle_radius_m=max(
                1.0e-4,
                float(payload.get("pf_visual_particle_radius_m", 0.025)),
            ),
            estimate_radius_m=max(
                1.0e-4,
                float(payload.get("pf_visual_estimate_radius_m", 0.13)),
            ),
            estimate_cross_size_m=max(
                1.0e-4,
                float(payload.get("pf_visual_estimate_cross_size_m", 0.35)),
            ),
            estimate_cross_width_m=max(
                1.0e-4,
                float(payload.get("pf_visual_estimate_cross_width_m", 0.035)),
            ),
            min_weight_fraction=max(
                0.0,
                float(payload.get("pf_visual_min_weight_fraction", 0.0)),
            ),
        )


class PFSceneVisualizer:
    """Render PF particles and estimates as visual-only Isaac Sim prims."""

    def __init__(
        self,
        stage_backend: StageBackend,
        *,
        config: PFSceneVisualizationConfig | None = None,
        root_path: str = "/World/SimBridge/PFVisualization",
    ) -> None:
        """Store the backend and marker roots."""
        self.stage_backend = stage_backend
        self.config = config or PFSceneVisualizationConfig()
        self.root_path = str(root_path)
        self.particles_root = f"{self.root_path}/Particles"
        self.estimates_root = f"{self.root_path}/Estimates"

    def update_from_payload(self, payload: dict[str, Any]) -> None:
        """Replace PF marker prims from a serialized PFFrame-like payload."""
        if not self.config.enabled:
            return
        self.stage_backend.remove_prim(self.root_path)
        self.stage_backend.ensure_xform(self.root_path)
        self.stage_backend.ensure_xform(self.particles_root)
        self.stage_backend.ensure_xform(self.estimates_root)
        particles = _coerce_point_mapping(payload.get("particle_positions", {}))
        weights = _coerce_raw_mapping(payload.get("particle_weights", {}))
        estimates = _coerce_point_mapping(payload.get("estimated_sources", {}))
        strengths = _coerce_raw_mapping(payload.get("estimated_strengths", {}))
        for isotope, positions in sorted(particles.items()):
            weight_arr = _coerce_vector(weights.get(isotope), positions.shape[0])
            selected_positions, selected_weights = self._select_particles(
                positions,
                weight_arr,
            )
            self._author_particles(isotope, selected_positions, selected_weights)
        for isotope, positions in sorted(estimates.items()):
            strength_arr = _coerce_vector(strengths.get(isotope), positions.shape[0])
            self._author_estimates(isotope, positions, strength_arr)
        self.stage_backend.step()

    def _select_particles(
        self,
        positions: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return a top-weight bounded particle subset for GUI rendering."""
        if positions.size == 0 or weights.size == 0:
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        valid = np.isfinite(positions).all(axis=1) & np.isfinite(weights)
        if self.config.min_weight_fraction > 0.0 and np.any(valid):
            max_weight = float(np.max(np.abs(weights[valid])))
            valid &= weights >= max_weight * self.config.min_weight_fraction
        positions = positions[valid]
        weights = weights[valid]
        if positions.size == 0:
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        max_particles = int(self.config.max_particles_per_isotope)
        if max_particles > 0 and positions.shape[0] > max_particles:
            order = np.argsort(weights)[::-1][:max_particles]
            positions = positions[order]
            weights = weights[order]
        return positions, weights

    def _author_particles(
        self,
        isotope: str,
        positions: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> None:
        """Author small PF particle spheres for one isotope."""
        isotope_token = _sanitize_token(isotope)
        root = f"{self.particles_root}/{isotope_token}"
        self.stage_backend.ensure_xform(root)
        color = _isotope_color(isotope)
        radii = _particle_radii(
            weights,
            base_radius=float(self.config.particle_radius_m),
        )
        for index, (position, radius_m) in enumerate(zip(positions, radii)):
            self.stage_backend.ensure_sphere(
                f"{root}/Particle_{index:04d}",
                radius_m=float(radius_m),
                translation_xyz=_tuple3(position),
                color_rgb=color,
                material="air",
            )

    def _author_estimates(
        self,
        isotope: str,
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
    ) -> None:
        """Author estimate markers and cross-hairs for one isotope."""
        isotope_token = _sanitize_token(isotope)
        root = f"{self.estimates_root}/{isotope_token}"
        self.stage_backend.ensure_xform(root)
        color = _isotope_color(isotope)
        for index, position in enumerate(positions):
            if not np.isfinite(position).all():
                continue
            marker_root = f"{root}/Estimate_{index:02d}"
            strength_scale = _estimate_strength_scale(strengths, index)
            radius = float(self.config.estimate_radius_m) * strength_scale
            self.stage_backend.ensure_sphere(
                f"{marker_root}/Center",
                radius_m=radius,
                translation_xyz=_tuple3(position),
                color_rgb=color,
                material="air",
            )
            self._author_cross(marker_root, position, color)

    def _author_cross(
        self,
        root: str,
        center: NDArray[np.float64],
        color: tuple[float, float, float],
    ) -> None:
        """Author three short cross-hair curves centered on an estimate."""
        half = 0.5 * float(self.config.estimate_cross_size_m)
        width = float(self.config.estimate_cross_width_m)
        axes = (
            ((-half, 0.0, 0.0), (half, 0.0, 0.0), "X"),
            ((0.0, -half, 0.0), (0.0, half, 0.0), "Y"),
            ((0.0, 0.0, -half), (0.0, 0.0, half), "Z"),
        )
        center_arr = np.asarray(center, dtype=float)
        for start_offset, end_offset, axis_name in axes:
            start = center_arr + np.asarray(start_offset, dtype=float)
            end = center_arr + np.asarray(end_offset, dtype=float)
            self.stage_backend.ensure_polyline(
                f"{root}/Cross_{axis_name}",
                points_xyz=(_tuple3(start), _tuple3(end)),
                color_rgb=color,
                width_m=width,
            )


def _coerce_point_mapping(value: Any) -> dict[str, NDArray[np.float64]]:
    """Return isotope keyed point arrays from a JSON-like payload."""
    if not isinstance(value, dict):
        return {}
    output: dict[str, NDArray[np.float64]] = {}
    for isotope, raw in value.items():
        arr = np.asarray(raw, dtype=float)
        if arr.size == 0 or arr.ndim != 2 or arr.shape[1] < 3:
            output[str(isotope)] = np.zeros((0, 3), dtype=float)
            continue
        output[str(isotope)] = arr[:, :3]
    return output


def _coerce_raw_mapping(value: Any) -> dict[str, Any]:
    """Return a shallow isotope keyed mapping for vector payloads."""
    return dict(value) if isinstance(value, dict) else {}


def _coerce_vector(value: Any, size: int) -> NDArray[np.float64]:
    """Return a one-dimensional numeric vector with a fallback length."""
    if value is None:
        return np.ones(max(size, 0), dtype=float)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == size:
        return arr
    if size <= 0:
        return np.zeros(0, dtype=float)
    if arr.size == 0:
        return np.ones(size, dtype=float)
    if arr.size > size:
        return arr[:size]
    padded = np.ones(size, dtype=float) * float(np.mean(arr))
    padded[: arr.size] = arr
    return padded


def _particle_radii(
    weights: NDArray[np.float64],
    *,
    base_radius: float,
) -> NDArray[np.float64]:
    """Scale particle marker radii by relative posterior weight."""
    if weights.size == 0:
        return np.zeros(0, dtype=float)
    finite = np.asarray(weights, dtype=float)
    finite = np.where(np.isfinite(finite), np.maximum(finite, 0.0), 0.0)
    max_weight = float(np.max(finite)) if finite.size else 0.0
    if max_weight <= 0.0:
        return np.full(finite.shape, base_radius, dtype=float)
    relative = np.sqrt(np.clip(finite / max_weight, 0.0, 1.0))
    return base_radius * (0.7 + 1.2 * relative)


def _estimate_strength_scale(strengths: NDArray[np.float64], index: int) -> float:
    """Return a mild display scale based on relative estimated strength."""
    if strengths.size == 0 or index >= strengths.size:
        return 1.0
    finite = np.where(np.isfinite(strengths), np.maximum(strengths, 0.0), 0.0)
    max_strength = float(np.max(finite)) if finite.size else 0.0
    if max_strength <= 0.0:
        return 1.0
    relative = float(np.sqrt(np.clip(finite[index] / max_strength, 0.0, 1.0)))
    return float(0.85 + 0.55 * relative)


def _isotope_color(isotope: str) -> tuple[float, float, float]:
    """Return the configured visualization color for an isotope."""
    return ISOTOPE_COLORS.get(str(isotope), (1.0, 0.8, 0.05))


def _sanitize_token(value: str) -> str:
    """Return a USD path token safe enough for generated marker names."""
    chars = [char if char.isalnum() else "_" for char in str(value)]
    token = "".join(chars).strip("_")
    if not token:
        return "Isotope"
    if token[0].isdigit():
        return f"Isotope_{token}"
    return token


def _tuple3(value: NDArray[np.float64]) -> tuple[float, float, float]:
    """Convert a length-three vector to a float tuple."""
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size < 3:
        raise ValueError("Expected a three-dimensional position.")
    return (float(arr[0]), float(arr[1]), float(arr[2]))
