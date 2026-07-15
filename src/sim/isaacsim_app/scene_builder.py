"""Scene description helpers and stage population for the Isaac Sim sidecar."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

import numpy as np

from measurement.model import PointSource
from measurement.obstacle_assets import KnownObstacleInstance, obstacle_instances_from_dicts

from sim.isaacsim_app.pf_visualizer import ISOTOPE_COLORS
from sim.isaacsim_app.stage_backend import StageBackend
from sim.shield_geometry import (
    SHIELD_CONTACT_RADIUS_M,
    ShieldThicknessConfig,
    nested_shield_inner_radii_cm,
    resolve_shield_thickness_config,
)


@dataclass(frozen=True)
class SourceDescription:
    """Describe a point source marker authored into the USD stage."""

    isotope: str
    position_xyz: tuple[float, float, float]
    intensity_cps_1m: float

    def to_point_source(self) -> PointSource:
        """Convert the source description into the estimator model type."""
        return PointSource(
            isotope=self.isotope,
            position=self.position_xyz,
            intensity_cps_1m=self.intensity_cps_1m,
        )


@dataclass(frozen=True)
class StagePrimPaths:
    """Collect prim paths used by the generated sidecar content."""

    world_root: str = "/World"
    generated_root: str = "/World/SimBridge"
    obstacles_root: str = "/World/SimBridge/Obstacles"
    sources_root: str = "/World/SimBridge/Sources"
    robot_root: str = "/World/SimBridge/Robot"
    robot_body_path: str = "/World/SimBridge/Robot/Body"
    robot_mast_path: str = "/World/SimBridge/Robot/Mast"
    robot_front_left_wheel_path: str = "/World/SimBridge/Robot/WheelFrontLeft"
    robot_front_right_wheel_path: str = "/World/SimBridge/Robot/WheelFrontRight"
    robot_rear_left_wheel_path: str = "/World/SimBridge/Robot/WheelRearLeft"
    robot_rear_right_wheel_path: str = "/World/SimBridge/Robot/WheelRearRight"
    detector_path: str = "/World/SimBridge/Robot/Detector"
    fe_shield_path: str = "/World/SimBridge/Robot/FeShield"
    pb_shield_path: str = "/World/SimBridge/Robot/PbShield"


@dataclass
class SceneDescription:
    """Describe world content and optional USD stage metadata."""

    room_size_xyz: tuple[float, float, float] = (10.0, 20.0, 10.0)
    obstacle_origin_xy: tuple[float, float] = (0.0, 0.0)
    obstacle_cell_size_m: float = 1.0
    obstacle_grid_shape: tuple[int, int] = (0, 0)
    obstacle_material: str = "concrete"
    obstacle_cells: list[tuple[int, int]] = field(default_factory=list)
    obstacle_instances: tuple[KnownObstacleInstance, ...] = ()
    author_obstacle_prims: bool = True
    author_room_boundary_prims: bool = False
    sources: list[SourceDescription] = field(default_factory=list)
    usd_path: str | None = None
    use_config_usd_fallback: bool = True
    prim_paths: StagePrimPaths = field(default_factory=StagePrimPaths)

    @property
    def source_count(self) -> int:
        """Return the number of configured source markers."""
        return len(self.sources)

    def to_point_sources(self) -> list[PointSource]:
        """Convert source descriptions into estimator point sources."""
        return [source.to_point_source() for source in self.sources]


def _as_float_tuple(values: Any, expected_len: int, field_name: str) -> tuple[float, ...]:
    """Validate and normalize a numeric tuple-like payload."""
    if not isinstance(values, (list, tuple)) or len(values) != expected_len:
        raise ValueError(f"{field_name} must be a {expected_len}-element list.")
    return tuple(float(v) for v in values)


def _sanitize_prim_token(value: str) -> str:
    """Convert an arbitrary label into a USD-safe prim token."""
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", str(value).strip())
    sanitized = sanitized.strip("_")
    if not sanitized:
        return "Prim"
    if sanitized[0].isdigit():
        return f"Prim_{sanitized}"
    return sanitized


def build_scene_description(payload: dict[str, Any]) -> SceneDescription:
    """Build a rich scene description from a reset payload."""
    room_size = _as_float_tuple(payload.get("room_size_xyz", (10.0, 20.0, 10.0)), 3, "room_size_xyz")
    obstacle_origin = _as_float_tuple(
        payload.get("obstacle_origin_xy", (0.0, 0.0)),
        2,
        "obstacle_origin_xy",
    )
    obstacle_grid_shape = tuple(
        int(v)
        for v in _as_float_tuple(
            payload.get("obstacle_grid_shape", (0, 0)),
            2,
            "obstacle_grid_shape",
        )
    )
    sources_payload = payload.get("sources", [])
    if not isinstance(sources_payload, list):
        raise ValueError("sources must be a list.")
    sources: list[SourceDescription] = []
    for idx, entry in enumerate(sources_payload):
        if not isinstance(entry, dict):
            raise ValueError(f"Source entry {idx} must be an object.")
        position = _as_float_tuple(entry.get("position", (0.0, 0.0, 0.0)), 3, "source position")
        sources.append(
            SourceDescription(
                isotope=str(entry.get("isotope", f"source_{idx}")),
                position_xyz=(position[0], position[1], position[2]),
                intensity_cps_1m=float(entry.get("intensity_cps_1m", 0.0)),
            )
        )
    prim_paths_payload = payload.get("prim_paths", {})
    if prim_paths_payload and not isinstance(prim_paths_payload, dict):
        raise ValueError("prim_paths must be a JSON object.")
    prim_paths = StagePrimPaths(**{key: str(value) for key, value in prim_paths_payload.items()})
    obstacle_cells = [tuple(int(v) for v in cell) for cell in payload.get("obstacle_cells", [])]
    obstacle_instances = obstacle_instances_from_dicts(
        payload.get("obstacle_instances", [])
    )
    return SceneDescription(
        room_size_xyz=(room_size[0], room_size[1], room_size[2]),
        obstacle_origin_xy=(obstacle_origin[0], obstacle_origin[1]),
        obstacle_cell_size_m=float(payload.get("obstacle_cell_size_m", 1.0)),
        obstacle_grid_shape=obstacle_grid_shape,
        obstacle_material=str(payload.get("obstacle_material", "concrete")),
        obstacle_cells=obstacle_cells,
        obstacle_instances=obstacle_instances,
        author_obstacle_prims=bool(payload.get("author_obstacle_prims", True)),
        author_room_boundary_prims=bool(payload.get("author_room_boundary_prims", False)),
        sources=sources,
        usd_path=None if payload.get("usd_path") in (None, "") else str(payload["usd_path"]),
        use_config_usd_fallback=bool(payload.get("use_config_usd_fallback", True)),
        prim_paths=prim_paths,
    )


class SceneBuilder:
    """Populate a stage with sidecar-generated helper prims."""

    def __init__(
        self,
        stage_backend: StageBackend,
        *,
        detector_height_m: float = 0.5,
        obstacle_height_m: float = 2.0,
        fe_shield_size_xyz: tuple[float, float, float] = (0.25, 0.08, 0.25),
        pb_shield_size_xyz: tuple[float, float, float] = (0.25, 0.08, 0.25),
        shield_thickness: ShieldThicknessConfig | None = None,
    ) -> None:
        """Store scene authoring defaults."""
        self.stage_backend = stage_backend
        self.detector_height_m = float(detector_height_m)
        self.obstacle_height_m = float(obstacle_height_m)
        self.fe_shield_size_xyz = tuple(float(v) for v in fe_shield_size_xyz)
        self.pb_shield_size_xyz = tuple(float(v) for v in pb_shield_size_xyz)
        self.shield_thickness = shield_thickness or resolve_shield_thickness_config()

    def load_scene(
        self,
        scene: SceneDescription,
        *,
        usd_path_override: str | None = None,
        reopen_stage: bool = True,
    ) -> None:
        """Open the requested stage and author bridge helper prims."""
        if reopen_stage:
            self.stage_backend.open_stage(usd_path_override or scene.usd_path)
        else:
            self._clear_scene_content(scene.prim_paths)
        self._ensure_base_hierarchy(scene.prim_paths)
        self._author_room_boundaries(scene)
        self._author_obstacles(scene)
        self._author_sources(scene)
        self._author_robot(scene.prim_paths)
        self.stage_backend.step()

    def _clear_scene_content(self, prim_paths: StagePrimPaths) -> None:
        """Remove generated scene content while preserving view helper prims."""
        self.stage_backend.remove_prim(prim_paths.obstacles_root)
        self.stage_backend.remove_prim(prim_paths.sources_root)
        self.stage_backend.remove_prim(prim_paths.robot_root)
        self.stage_backend.remove_prim(f"{prim_paths.generated_root}/Radiation")

    def _ensure_base_hierarchy(self, prim_paths: StagePrimPaths) -> None:
        """Ensure the generated content hierarchy exists."""
        self.stage_backend.ensure_xform(prim_paths.world_root)
        self.stage_backend.ensure_xform(prim_paths.generated_root)
        self.stage_backend.ensure_xform(prim_paths.obstacles_root)
        self.stage_backend.ensure_xform(prim_paths.sources_root)
        self.stage_backend.ensure_xform(prim_paths.robot_root)

    def _author_obstacles(self, scene: SceneDescription) -> None:
        """Create known obstacle assets or simple box markers for blocked cells."""
        if not scene.author_obstacle_prims:
            return
        if scene.obstacle_instances:
            for instance in scene.obstacle_instances:
                self.stage_backend.ensure_xform(
                    f"{scene.prim_paths.obstacles_root}/{instance.name}"
                )
                for component in instance.components:
                    self.stage_backend.ensure_box(
                        f"{scene.prim_paths.obstacles_root}/{instance.name}/{component.name}",
                        size_xyz=component.size_xyz,
                        translation_xyz=component.center_xyz,
                        color_rgb=(0.3, 0.3, 0.3),
                        material=component.material,
                        transport_group="obstacle",
                    )
            return
        z_center = 0.5 * self.obstacle_height_m
        cell_size = scene.obstacle_cell_size_m
        for index, (ix, iy) in enumerate(scene.obstacle_cells):
            x0 = scene.obstacle_origin_xy[0] + float(ix) * cell_size
            y0 = scene.obstacle_origin_xy[1] + float(iy) * cell_size
            center = (x0 + 0.5 * cell_size, y0 + 0.5 * cell_size, z_center)
            self.stage_backend.ensure_box(
                f"{scene.prim_paths.obstacles_root}/Obstacle_{index:04d}",
                size_xyz=(cell_size, cell_size, self.obstacle_height_m),
                translation_xyz=center,
                color_rgb=(0.2, 0.2, 0.2),
                material=scene.obstacle_material,
                transport_group="obstacle",
            )

    def _author_room_boundaries(self, scene: SceneDescription) -> None:
        """Create optional room boundary solids for CUI Geant4 scenes."""
        if not scene.author_room_boundary_prims:
            return
        size_x, size_y, size_z = (float(value) for value in scene.room_size_xyz)
        wall_height = max(0.1, size_z)
        wall_thickness = 0.1
        environment_root = "/World/Environment"
        wall_root = f"{environment_root}/Wall"
        self.stage_backend.ensure_xform(environment_root)
        self.stage_backend.ensure_xform(wall_root)
        for name, size_xyz, center_xyz in (
            (
                "Floor",
                (size_x, size_y, wall_thickness),
                (0.5 * size_x, 0.5 * size_y, -0.5 * wall_thickness),
            ),
            (
                "NorthWall",
                (size_x, wall_thickness, wall_height),
                (0.5 * size_x, size_y + 0.5 * wall_thickness, 0.5 * wall_height),
            ),
            (
                "SouthWall",
                (size_x, wall_thickness, wall_height),
                (0.5 * size_x, -0.5 * wall_thickness, 0.5 * wall_height),
            ),
            (
                "EastWall",
                (wall_thickness, size_y, wall_height),
                (size_x + 0.5 * wall_thickness, 0.5 * size_y, 0.5 * wall_height),
            ),
            (
                "WestWall",
                (wall_thickness, size_y, wall_height),
                (-0.5 * wall_thickness, 0.5 * size_y, 0.5 * wall_height),
            ),
            (
                "Ceiling",
                (size_x, size_y, wall_thickness),
                (0.5 * size_x, 0.5 * size_y, size_z + 0.5 * wall_thickness),
            ),
        ):
            color_rgb = (0.88, 0.93, 1.0) if name == "Ceiling" else (0.75, 0.78, 0.82)
            self.stage_backend.ensure_box(
                f"{wall_root}/{name}",
                size_xyz=size_xyz,
                translation_xyz=center_xyz,
                color_rgb=color_rgb,
                material="concrete",
                transport_group="wall",
            )

    def _author_sources(self, scene: SceneDescription) -> None:
        """Create simple sphere markers for radiation sources."""
        for index, source in enumerate(scene.sources):
            prim_name = _sanitize_prim_token(source.isotope)
            self.stage_backend.ensure_sphere(
                f"{scene.prim_paths.sources_root}/{prim_name}_{index:02d}",
                radius_m=0.08,
                translation_xyz=source.position_xyz,
                color_rgb=ISOTOPE_COLORS.get(source.isotope, (1.0, 0.8, 0.05)),
            )

    def _author_robot(self, prim_paths: StagePrimPaths) -> None:
        """Create a compact mobile robot model with detector and shield prims."""
        self.stage_backend.ensure_xform(prim_paths.robot_root)
        self.stage_backend.ensure_box(
            prim_paths.robot_body_path,
            size_xyz=(0.7, 0.45, 0.22),
            translation_xyz=(0.0, 0.0, 0.12),
            color_rgb=(0.16, 0.22, 0.28),
            material="steel",
        )
        self.stage_backend.ensure_box(
            prim_paths.robot_mast_path,
            size_xyz=(0.08, 0.08, max(self.detector_height_m, 0.1)),
            translation_xyz=(0.0, 0.0, 0.5 * self.detector_height_m),
            color_rgb=(0.18, 0.18, 0.18),
            material="steel",
        )
        for path, x_offset, y_offset in (
            (prim_paths.robot_front_left_wheel_path, 0.22, 0.28),
            (prim_paths.robot_front_right_wheel_path, 0.22, -0.28),
            (prim_paths.robot_rear_left_wheel_path, -0.22, 0.28),
            (prim_paths.robot_rear_right_wheel_path, -0.22, -0.28),
        ):
            self.stage_backend.ensure_box(
                path,
                size_xyz=(0.18, 0.08, 0.18),
                translation_xyz=(x_offset, y_offset, 0.08),
                color_rgb=(0.02, 0.02, 0.02),
                material="rubber",
            )
        detector_visual_radius_m = SHIELD_CONTACT_RADIUS_M
        self.stage_backend.ensure_sphere(
            prim_paths.detector_path,
            radius_m=detector_visual_radius_m,
            translation_xyz=(0.0, 0.0, self.detector_height_m),
            color_rgb=(0.0, 0.85, 1.0),
            material="air",
        )
        fe_inner_cm, pb_inner_cm = nested_shield_inner_radii_cm(
            thickness_fe_cm=float(self.shield_thickness.thickness_fe_cm),
            detector_outer_radius_cm=100.0 * detector_visual_radius_m,
        )
        fe_points, fe_counts, fe_indices = _octant_shell_mesh(
            inner_radius_m=fe_inner_cm / 100.0,
            outer_radius_m=(fe_inner_cm + float(self.shield_thickness.thickness_fe_cm)) / 100.0,
            theta_steps=8,
            phi_steps=8,
        )
        self.stage_backend.ensure_mesh(
            prim_paths.fe_shield_path,
            points_xyz=fe_points,
            face_vertex_counts=fe_counts,
            face_vertex_indices=fe_indices,
            translation_xyz=(0.0, 0.0, self.detector_height_m),
            color_rgb=(0.9, 0.45, 0.05),
            material="fe",
        )
        pb_points, pb_counts, pb_indices = _octant_shell_mesh(
            inner_radius_m=pb_inner_cm / 100.0,
            outer_radius_m=(pb_inner_cm + float(self.shield_thickness.thickness_pb_cm)) / 100.0,
            theta_steps=8,
            phi_steps=8,
        )
        self.stage_backend.ensure_mesh(
            prim_paths.pb_shield_path,
            points_xyz=pb_points,
            face_vertex_counts=pb_counts,
            face_vertex_indices=pb_indices,
            translation_xyz=(0.0, 0.0, self.detector_height_m),
            color_rgb=(0.35, 0.35, 0.65),
            material="pb",
        )


def _octant_shell_mesh(
    *,
    inner_radius_m: float,
    outer_radius_m: float,
    theta_steps: int,
    phi_steps: int,
) -> tuple[tuple[tuple[float, float, float], ...], tuple[int, ...], tuple[int, ...]]:
    """Build a local +X/+Y/+Z one-eighth spherical shell mesh."""
    theta_count = max(int(theta_steps), 2)
    phi_count = max(int(phi_steps), 2)
    inner_radius = max(0.0, float(inner_radius_m))
    outer_radius = max(inner_radius + 1.0e-6, float(outer_radius_m))
    theta_values = np.linspace(0.0, 0.5 * np.pi, theta_count + 1)
    phi_values = np.linspace(0.0, 0.5 * np.pi, phi_count + 1)
    points: list[tuple[float, float, float]] = []
    for radius in (outer_radius, inner_radius):
        for theta in theta_values:
            sin_theta = float(np.sin(theta))
            cos_theta = float(np.cos(theta))
            for phi in phi_values:
                points.append(
                    (
                        float(radius * sin_theta * np.cos(phi)),
                        float(radius * sin_theta * np.sin(phi)),
                        float(radius * cos_theta),
                    )
                )
    row = phi_count + 1
    layer = (theta_count + 1) * row
    outer_offset = 0
    inner_offset = layer
    counts: list[int] = []
    indices: list[int] = []

    def _append_quad(a: int, b: int, c: int, d: int) -> None:
        """Append one quad face by point indices."""
        counts.append(4)
        indices.extend((a, b, c, d))

    for theta_idx in range(theta_count):
        for phi_idx in range(phi_count):
            a = outer_offset + theta_idx * row + phi_idx
            b = outer_offset + (theta_idx + 1) * row + phi_idx
            c = outer_offset + (theta_idx + 1) * row + phi_idx + 1
            d = outer_offset + theta_idx * row + phi_idx + 1
            _append_quad(a, b, c, d)
            ai = inner_offset + theta_idx * row + phi_idx
            bi = inner_offset + (theta_idx + 1) * row + phi_idx
            ci = inner_offset + (theta_idx + 1) * row + phi_idx + 1
            di = inner_offset + theta_idx * row + phi_idx + 1
            _append_quad(di, ci, bi, ai)
    for theta_idx in range(theta_count):
        for phi_idx in (0, phi_count):
            a = outer_offset + theta_idx * row + phi_idx
            b = outer_offset + (theta_idx + 1) * row + phi_idx
            c = inner_offset + (theta_idx + 1) * row + phi_idx
            d = inner_offset + theta_idx * row + phi_idx
            _append_quad(a, b, c, d)
    for phi_idx in range(phi_count):
        for theta_idx in (0, theta_count):
            a = outer_offset + theta_idx * row + phi_idx
            b = outer_offset + theta_idx * row + phi_idx + 1
            c = inner_offset + theta_idx * row + phi_idx + 1
            d = inner_offset + theta_idx * row + phi_idx
            _append_quad(a, b, c, d)
    return tuple(points), tuple(counts), tuple(indices)
