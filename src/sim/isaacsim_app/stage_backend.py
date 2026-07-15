"""Stage backends for Isaac Sim and test doubles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
import importlib
import importlib.abc
import importlib.machinery
from math import cos, sin
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import numpy as np

from sim.isaacsim_app.materials import normalize_composition_by_mass, normalize_material_name


@dataclass(frozen=True)
class PrimPose:
    """Represent a prim pose using translation and quaternion orientation."""

    translation_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class StageMaterialInfo:
    """Describe a resolved material and optional attenuation overrides."""

    name: str
    mu_by_isotope: dict[str, float] = field(default_factory=dict)
    density_g_cm3: float | None = None
    mass_att_by_isotope_cm2_g: dict[str, float] = field(default_factory=dict)
    preset_name: str | None = None
    composition_by_mass: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class StageSolidPrim:
    """Describe a simple solid prim with world-space geometry."""

    path: str
    shape: str
    pose: PrimPose
    size_xyz: tuple[float, float, float] | None = None
    radius_m: float | None = None
    triangles_xyz: tuple[
        tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
        ...,
    ] | None = None
    material_info: StageMaterialInfo | None = None
    transport_group: str | None = None


def _normalize_transport_group(value: str) -> str:
    """Normalize semantic transport group labels for stable downstream use."""
    return str(value).strip().replace("-", "_").replace(" ", "_").lower()


def _infer_transport_group_from_path(path: str) -> str | None:
    """Infer broad transport grouping from generic USD path tokens."""
    tokens = [
        _normalize_transport_group(token)
        for token in str(path).strip("/").split("/")
        if token
    ]
    for token in tokens:
        if token in {"floor", "ceiling", "roof", "wall", "walls"}:
            return "wall"
        if token.endswith("wall") or token.endswith("walls"):
            return "wall"
        if "_wall" in token or "_walls" in token:
            return "wall"
    if any(token in {"obstacle", "obstacles"} or token.startswith("obstacle_") for token in tokens):
        return "obstacle"
    return None


def _axis_aligned_box_from_triangles(
    triangles_xyz: tuple[
        tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
        ...,
    ],
    *,
    tolerance: float = 1.0e-6,
) -> tuple[PrimPose, tuple[float, float, float]] | None:
    """Return an equivalent axis-aligned box for cuboid triangle meshes."""
    if not triangles_xyz:
        return None
    points = np.asarray(
        [point for triangle in triangles_xyz for point in triangle],
        dtype=float,
    )
    if points.size == 0 or points.shape[1] != 3:
        return None
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    size = upper - lower
    if np.any(size <= tolerance):
        return None
    rounded_unique = np.unique(np.round(points / tolerance).astype(np.int64), axis=0)
    unique_points = rounded_unique.astype(float) * float(tolerance)
    if unique_points.shape[0] != 8:
        return None
    expected_corners = np.asarray(
        [
            (x, y, z)
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ],
        dtype=float,
    )
    for point in unique_points:
        if not np.any(np.all(np.isclose(expected_corners, point, atol=tolerance), axis=1)):
            return None
    center = 0.5 * (lower + upper)
    return (
        PrimPose(
            translation_xyz=(float(center[0]), float(center[1]), float(center[2])),
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
        (float(size[0]), float(size[1]), float(size[2])),
    )


class StageBackend(ABC):
    """Abstract the minimal stage operations used by the sidecar."""

    @abstractmethod
    def open_stage(self, usd_path: str | None = None) -> None:
        """Open a USD stage or create a new one when no path is provided."""

    @abstractmethod
    def ensure_xform(self, path: str) -> None:
        """Ensure an Xform prim exists at the requested path."""

    @abstractmethod
    def ensure_box(
        self,
        path: str,
        *,
        size_xyz: tuple[float, float, float],
        translation_xyz: tuple[float, float, float],
        color_rgb: tuple[float, float, float] | None = None,
        material: str | None = None,
        transport_group: str | None = None,
    ) -> None:
        """Ensure a box prim exists with the requested geometry and pose."""

    @abstractmethod
    def ensure_sphere(
        self,
        path: str,
        *,
        radius_m: float,
        translation_xyz: tuple[float, float, float],
        color_rgb: tuple[float, float, float] | None = None,
        material: str | None = None,
        transport_group: str | None = None,
    ) -> None:
        """Ensure a sphere prim exists with the requested geometry and pose."""

    @abstractmethod
    def ensure_mesh(
        self,
        path: str,
        *,
        points_xyz: tuple[tuple[float, float, float], ...],
        face_vertex_counts: tuple[int, ...],
        face_vertex_indices: tuple[int, ...],
        translation_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
        color_rgb: tuple[float, float, float] | None = None,
        material: str | None = None,
        transport_group: str | None = None,
    ) -> None:
        """Ensure a mesh prim exists with the requested geometry and pose."""

    @abstractmethod
    def ensure_polyline(
        self,
        path: str,
        *,
        points_xyz: tuple[tuple[float, float, float], ...],
        color_rgb: tuple[float, float, float] | None = None,
        width_m: float = 0.02,
    ) -> None:
        """Ensure a polyline curve prim exists with the requested points."""

    @abstractmethod
    def ensure_dome_light(
        self,
        path: str,
        *,
        intensity: float,
        color_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        """Ensure a dome light exists for broad stage illumination."""

    @abstractmethod
    def ensure_sphere_light(
        self,
        path: str,
        *,
        intensity: float,
        radius_m: float,
        translation_xyz: tuple[float, float, float],
        color_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        """Ensure a local sphere light exists at the requested position."""

    @abstractmethod
    def set_camera_view(
        self,
        path: str,
        *,
        eye_xyz: tuple[float, float, float],
        target_xyz: tuple[float, float, float],
        focal_length_mm: float = 24.0,
    ) -> None:
        """Create or update a camera looking from eye to target."""

    @abstractmethod
    def apply_visual_material(
        self,
        path_prefix: str,
        *,
        color_rgb: tuple[float, float, float],
        opacity: float = 1.0,
        roughness: float = 0.8,
        emissive_scale: float = 0.0,
    ) -> None:
        """Apply a visual-only material override to matching stage prims."""

    @abstractmethod
    def remove_prim(self, path: str) -> None:
        """Remove a prim subtree when it exists."""

    @abstractmethod
    def set_local_pose(
        self,
        path: str,
        *,
        translation_xyz: tuple[float, float, float] | None = None,
        orientation_wxyz: tuple[float, float, float, float] | None = None,
        scale_xyz: tuple[float, float, float] | None = None,
    ) -> None:
        """Set the local pose of a prim."""

    @abstractmethod
    def get_world_pose(self, path: str) -> PrimPose:
        """Return the world pose of a prim."""

    @abstractmethod
    def list_solid_prims(self, path_prefixes: tuple[str, ...] | None = None) -> list[StageSolidPrim]:
        """Return authored simple solids, optionally filtered by path prefix."""

    @abstractmethod
    def step(self) -> None:
        """Flush stage edits into the simulator."""

    @abstractmethod
    def close(self) -> None:
        """Release simulator resources."""


@dataclass
class FakeStagePrim:
    """Store the authored state of a fake prim."""

    prim_type: str
    pose: PrimPose = field(default_factory=PrimPose)
    scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
    metadata: dict[str, Any] = field(default_factory=dict)


class FakeStageBackend(StageBackend):
    """Provide an in-memory stage backend for tests."""

    def __init__(self) -> None:
        """Initialize the fake stage state."""
        self.opened_usd_path: str | None = None
        self.open_stage_calls: list[str | None] = []
        self.prims: dict[str, FakeStagePrim] = {}

    def open_stage(self, usd_path: str | None = None) -> None:
        """Record the opened USD path and reset prim state."""
        self.opened_usd_path = usd_path
        self.open_stage_calls.append(usd_path)
        self.prims = {"/World": FakeStagePrim(prim_type="Xform")}
        if usd_path is not None and Path(usd_path).name == "demo_room.usda":
            self._seed_demo_room()

    def ensure_xform(self, path: str) -> None:
        """Create an Xform prim if it is missing."""
        self.prims.setdefault(path, FakeStagePrim(prim_type="Xform"))

    def ensure_box(
        self,
        path: str,
        *,
        size_xyz: tuple[float, float, float],
        translation_xyz: tuple[float, float, float],
        color_rgb: tuple[float, float, float] | None = None,
        material: str | None = None,
        transport_group: str | None = None,
    ) -> None:
        """Create or update a box prim."""
        self.prims[path] = FakeStagePrim(
            prim_type="Cube",
            pose=PrimPose(translation_xyz=translation_xyz),
            scale_xyz=size_xyz,
            metadata={
                "color_rgb": color_rgb,
                "material": material,
                "transport_group": transport_group,
            },
        )

    def ensure_sphere(
        self,
        path: str,
        *,
        radius_m: float,
        translation_xyz: tuple[float, float, float],
        color_rgb: tuple[float, float, float] | None = None,
        material: str | None = None,
        transport_group: str | None = None,
    ) -> None:
        """Create or update a sphere prim."""
        diameter = float(radius_m) * 2.0
        self.prims[path] = FakeStagePrim(
            prim_type="Sphere",
            pose=PrimPose(translation_xyz=translation_xyz),
            scale_xyz=(diameter, diameter, diameter),
            metadata={
                "color_rgb": color_rgb,
                "material": material,
                "transport_group": transport_group,
            },
        )

    def ensure_mesh(
        self,
        path: str,
        *,
        points_xyz: tuple[tuple[float, float, float], ...],
        face_vertex_counts: tuple[int, ...],
        face_vertex_indices: tuple[int, ...],
        translation_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
        color_rgb: tuple[float, float, float] | None = None,
        material: str | None = None,
        transport_group: str | None = None,
    ) -> None:
        """Create or update a mesh prim."""
        self.prims[path] = FakeStagePrim(
            prim_type="Mesh",
            pose=PrimPose(translation_xyz=translation_xyz),
            metadata={
                "color_rgb": color_rgb,
                "material": material,
                "transport_group": transport_group,
                "points_xyz": tuple(tuple(float(v) for v in point) for point in points_xyz),
                "face_vertex_counts": tuple(int(v) for v in face_vertex_counts),
                "face_vertex_indices": tuple(int(v) for v in face_vertex_indices),
            },
        )

    def ensure_polyline(
        self,
        path: str,
        *,
        points_xyz: tuple[tuple[float, float, float], ...],
        color_rgb: tuple[float, float, float] | None = None,
        width_m: float = 0.02,
    ) -> None:
        """Create or update a fake polyline prim."""
        self.prims[path] = FakeStagePrim(
            prim_type="BasisCurves",
            metadata={
                "color_rgb": color_rgb,
                "points_xyz": tuple(tuple(float(v) for v in point) for point in points_xyz),
                "width_m": float(width_m),
            },
        )

    def ensure_dome_light(
        self,
        path: str,
        *,
        intensity: float,
        color_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        """Create or update a fake dome light prim."""
        self.prims[path] = FakeStagePrim(
            prim_type="DomeLight",
            metadata={
                "intensity": float(intensity),
                "color_rgb": tuple(float(v) for v in color_rgb),
            },
        )

    def ensure_sphere_light(
        self,
        path: str,
        *,
        intensity: float,
        radius_m: float,
        translation_xyz: tuple[float, float, float],
        color_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        """Create or update a fake sphere light prim."""
        self.prims[path] = FakeStagePrim(
            prim_type="SphereLight",
            pose=PrimPose(translation_xyz=tuple(float(v) for v in translation_xyz)),
            metadata={
                "intensity": float(intensity),
                "radius_m": float(radius_m),
                "color_rgb": tuple(float(v) for v in color_rgb),
            },
        )

    def set_camera_view(
        self,
        path: str,
        *,
        eye_xyz: tuple[float, float, float],
        target_xyz: tuple[float, float, float],
        focal_length_mm: float = 24.0,
    ) -> None:
        """Create or update a fake camera prim."""
        self.prims[path] = FakeStagePrim(
            prim_type="Camera",
            pose=PrimPose(
                translation_xyz=tuple(float(v) for v in eye_xyz),
                orientation_wxyz=_look_at_quaternion_wxyz(eye_xyz, target_xyz),
            ),
            metadata={
                "target_xyz": tuple(float(v) for v in target_xyz),
                "focal_length_mm": float(focal_length_mm),
            },
        )

    def apply_visual_material(
        self,
        path_prefix: str,
        *,
        color_rgb: tuple[float, float, float],
        opacity: float = 1.0,
        roughness: float = 0.8,
        emissive_scale: float = 0.0,
    ) -> None:
        """Record a visual-only material override on fake matching prims."""
        prefix = str(path_prefix).rstrip("/")
        for prim_path, prim in self.prims.items():
            if not prim_path.startswith(prefix):
                continue
            if prim.prim_type not in {"Cube", "Sphere", "Mesh"}:
                continue
            prim.metadata["visual_color_rgb"] = tuple(float(v) for v in color_rgb)
            prim.metadata["visual_opacity"] = float(opacity)
            prim.metadata["visual_roughness"] = float(roughness)
            prim.metadata["visual_emissive_scale"] = float(emissive_scale)
            prim.metadata["visual_visible"] = bool(float(opacity) > 0.0)

    def remove_prim(self, path: str) -> None:
        """Remove a fake prim and any children under it."""
        prefix = f"{path.rstrip('/')}/"
        for prim_path in list(self.prims):
            if prim_path == path or prim_path.startswith(prefix):
                self.prims.pop(prim_path, None)

    def set_local_pose(
        self,
        path: str,
        *,
        translation_xyz: tuple[float, float, float] | None = None,
        orientation_wxyz: tuple[float, float, float, float] | None = None,
        scale_xyz: tuple[float, float, float] | None = None,
    ) -> None:
        """Update the stored fake pose."""
        prim = self.prims.setdefault(path, FakeStagePrim(prim_type="Xform"))
        translation = prim.pose.translation_xyz if translation_xyz is None else translation_xyz
        orientation = prim.pose.orientation_wxyz if orientation_wxyz is None else orientation_wxyz
        scale = prim.scale_xyz if scale_xyz is None else scale_xyz
        prim.pose = PrimPose(
            translation_xyz=tuple(float(v) for v in translation),
            orientation_wxyz=tuple(float(v) for v in orientation),
        )
        prim.scale_xyz = tuple(float(v) for v in scale)

    def get_world_pose(self, path: str) -> PrimPose:
        """Return the stored prim pose with parent translations accumulated."""
        prim = self.prims.get(path)
        if prim is None:
            raise KeyError(f"Prim not found: {path}")
        translation = np.asarray(prim.pose.translation_xyz, dtype=float)
        parent_path = path.rsplit("/", 1)[0]
        while parent_path and parent_path != path:
            parent = self.prims.get(parent_path)
            if parent is not None:
                translation += np.asarray(parent.pose.translation_xyz, dtype=float)
            if "/" not in parent_path.strip("/"):
                break
            parent_path = parent_path.rsplit("/", 1)[0]
        return PrimPose(
            translation_xyz=(float(translation[0]), float(translation[1]), float(translation[2])),
            orientation_wxyz=prim.pose.orientation_wxyz,
        )

    def list_solid_prims(self, path_prefixes: tuple[str, ...] | None = None) -> list[StageSolidPrim]:
        """Return stored solid prims with world-space geometry."""
        prefixes = tuple(path_prefixes or ())
        result: list[StageSolidPrim] = []
        for path, prim in self.prims.items():
            if prim.prim_type not in {"Cube", "Sphere", "Mesh"}:
                continue
            if prefixes and not any(path.startswith(prefix) for prefix in prefixes):
                continue
            if prim.prim_type == "Cube":
                result.append(
                    StageSolidPrim(
                        path=path,
                        shape="box",
                        pose=self.get_world_pose(path),
                        size_xyz=tuple(float(v) for v in prim.scale_xyz),
                        material_info=self._fake_material_info_from_prim(path, prim),
                        transport_group=self._fake_transport_group_from_prim(path, prim),
                    )
                )
                continue
            if prim.prim_type == "Mesh":
                triangles = self._fake_mesh_triangles(path)
                box_geometry = _axis_aligned_box_from_triangles(triangles)
                if box_geometry is not None:
                    box_pose, box_size = box_geometry
                    result.append(
                        StageSolidPrim(
                            path=path,
                            shape="box",
                            pose=box_pose,
                            size_xyz=box_size,
                            material_info=self._fake_material_info_from_prim(path, prim),
                            transport_group=self._fake_transport_group_from_prim(path, prim),
                        )
                    )
                    continue
                result.append(
                    StageSolidPrim(
                        path=path,
                        shape="mesh",
                        pose=self.get_world_pose(path),
                        triangles_xyz=triangles,
                        material_info=self._fake_material_info_from_prim(path, prim),
                        transport_group=self._fake_transport_group_from_prim(path, prim),
                    )
                )
                continue
            result.append(
                StageSolidPrim(
                    path=path,
                    shape="sphere",
                    pose=self.get_world_pose(path),
                    radius_m=0.5 * float(prim.scale_xyz[0]),
                    material_info=self._fake_material_info_from_prim(path, prim),
                    transport_group=self._fake_transport_group_from_prim(path, prim),
                )
            )
        return result

    def step(self) -> None:
        """The fake backend does not need stepping."""
        return None

    def close(self) -> None:
        """The fake backend does not own external resources."""
        return None

    def _fake_mesh_triangles(
        self,
        path: str,
    ) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]], ...]:
        """Triangulate a fake mesh prim into world-space triangles."""
        prim = self.prims[path]
        points = prim.metadata.get("points_xyz", ())
        counts = prim.metadata.get("face_vertex_counts", ())
        indices = prim.metadata.get("face_vertex_indices", ())
        translation = np.asarray(self.get_world_pose(path).translation_xyz, dtype=float)
        world_points = [
            (
                float(point[0] + translation[0]),
                float(point[1] + translation[1]),
                float(point[2] + translation[2]),
            )
            for point in points
        ]
        triangles: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]] = []
        cursor = 0
        for count in counts:
            face_count = int(count)
            face_indices = [int(indices[cursor + idx]) for idx in range(face_count)]
            cursor += face_count
            if face_count < 3:
                continue
            anchor = world_points[face_indices[0]]
            for tri_index in range(1, face_count - 1):
                triangles.append(
                    (
                        anchor,
                        world_points[face_indices[tri_index]],
                        world_points[face_indices[tri_index + 1]],
                    )
                )
        return tuple(triangles)

    def _seed_demo_room(self) -> None:
        """Populate a small static room matching the demo USD stage."""
        self.ensure_xform("/World/Environment")
        self.prims["/World/Looks"] = FakeStagePrim(prim_type="Scope")
        self.prims["/World/Looks/ConcreteMaterial"] = FakeStagePrim(
            prim_type="Material",
            metadata={
                "material": "concrete",
                "shader_inputs": {
                    "simbridge_density_g_cm3": 2.3,
                    "simbridge_mass_att_cs_137_cm2_g": 0.07391304347826087,
                    "simbridge_mass_att_co_60_cm2_g": 0.04782608695652174,
                    "simbridge_mass_att_eu_154_cm2_g": 0.0782608695652174,
                },
            },
        )
        self.ensure_box(
            "/World/Environment/Floor",
            size_xyz=(10.0, 20.0, 0.1),
            translation_xyz=(5.0, 10.0, -0.05),
            color_rgb=(0.85, 0.85, 0.85),
        )
        self.prims["/World/Environment/Floor"].metadata["material_binding"] = "/World/Looks/ConcreteMaterial"
        self.ensure_mesh(
            "/World/Environment/PillarMesh",
            points_xyz=(
                (0.0, 0.0, 0.0),
                (0.4, 0.0, 0.0),
                (0.4, 0.4, 0.0),
                (0.0, 0.4, 0.0),
                (0.0, 0.0, 2.0),
                (0.4, 0.0, 2.0),
                (0.4, 0.4, 2.0),
                (0.0, 0.4, 2.0),
            ),
            face_vertex_counts=(4, 4, 4, 4, 4, 4),
            face_vertex_indices=(
                0, 1, 2, 3,
                4, 5, 6, 7,
                0, 1, 5, 4,
                1, 2, 6, 5,
                2, 3, 7, 6,
                3, 0, 4, 7,
            ),
            translation_xyz=(6.8, 10.8, 0.0),
            color_rgb=(0.55, 0.6, 0.62),
        )
        self.prims["/World/Environment/PillarMesh"].metadata["material_binding"] = "/World/Looks/ConcreteMaterial"
        self.ensure_box(
            "/World/Environment/NorthWall",
            size_xyz=(10.0, 0.1, 3.0),
            translation_xyz=(5.0, 20.05, 1.5),
            color_rgb=(0.75, 0.78, 0.82),
        )
        self.prims["/World/Environment/NorthWall"].metadata["material_binding"] = "/World/Looks/ConcreteMaterial"
        self.ensure_box(
            "/World/Environment/SouthWall",
            size_xyz=(10.0, 0.1, 3.0),
            translation_xyz=(5.0, -0.05, 1.5),
            color_rgb=(0.75, 0.78, 0.82),
        )
        self.prims["/World/Environment/SouthWall"].metadata["material_binding"] = "/World/Looks/ConcreteMaterial"
        self.ensure_box(
            "/World/Environment/EastWall",
            size_xyz=(0.1, 20.0, 3.0),
            translation_xyz=(10.05, 10.0, 1.5),
            color_rgb=(0.7, 0.74, 0.78),
        )
        self.prims["/World/Environment/EastWall"].metadata["material_binding"] = "/World/Looks/ConcreteMaterial"
        self.ensure_box(
            "/World/Environment/WestWall",
            size_xyz=(0.1, 20.0, 3.0),
            translation_xyz=(-0.05, 10.0, 1.5),
            color_rgb=(0.7, 0.74, 0.78),
        )
        self.prims["/World/Environment/WestWall"].metadata["material_binding"] = "/World/Looks/ConcreteMaterial"

    def _fake_material_info_from_prim(
        self,
        path: str,
        prim: FakeStagePrim,
    ) -> StageMaterialInfo | None:
        """Resolve material metadata or a fake binding target."""
        material = prim.metadata.get("material")
        if material not in (None, ""):
            return StageMaterialInfo(
                name=str(material),
                mu_by_isotope=self._fake_mu_by_isotope_from_prim(prim),
                density_g_cm3=self._fake_density_from_prim(prim),
                mass_att_by_isotope_cm2_g=self._fake_mass_att_by_isotope_from_prim(prim),
                preset_name=self._fake_preset_name_from_prim(prim),
                composition_by_mass=self._fake_composition_by_mass_from_prim(prim),
            )
        binding_path = prim.metadata.get("material_binding")
        if binding_path in (None, ""):
            return None
        target = self.prims.get(str(binding_path))
        if target is None:
            return StageMaterialInfo(name=self._normalize_material_name(Path(str(binding_path)).name))
        bound_material = target.metadata.get("material")
        material_name = (
            str(bound_material)
            if bound_material not in (None, "")
            else self._normalize_material_name(Path(str(binding_path)).name)
        )
        return StageMaterialInfo(
            name=material_name,
            mu_by_isotope=self._fake_mu_by_isotope_from_prim(target),
            density_g_cm3=self._fake_density_from_prim(target),
            mass_att_by_isotope_cm2_g=self._fake_mass_att_by_isotope_from_prim(target),
            preset_name=self._fake_preset_name_from_prim(target),
            composition_by_mass=self._fake_composition_by_mass_from_prim(target),
        )

    def _fake_transport_group_from_prim(self, path: str, prim: FakeStagePrim) -> str | None:
        """Resolve semantic transport grouping from fake metadata or path tokens."""
        for key in (
            "transport_group",
            "simbridge_transport_group",
            "simbridge:transport_group",
            "simbridge_category",
            "simbridge:category",
        ):
            value = prim.metadata.get(key)
            if value not in (None, ""):
                return _normalize_transport_group(str(value))
        return _infer_transport_group_from_path(path)

    def _fake_mu_by_isotope_from_prim(self, prim: FakeStagePrim) -> dict[str, float]:
        """Extract fake attenuation overrides from prim metadata."""
        result: dict[str, float] = {}
        explicit = prim.metadata.get("mu_by_isotope", {})
        if isinstance(explicit, dict):
            for isotope, value in explicit.items():
                result[self._normalize_isotope_name(str(isotope))] = float(value)
        shader_inputs = prim.metadata.get("shader_inputs", {})
        if isinstance(shader_inputs, dict):
            for key, value in shader_inputs.items():
                isotope = self._isotope_from_mu_parameter_name(str(key))
                if isotope is not None:
                    result[isotope] = float(value)
        return result

    def _fake_density_from_prim(self, prim: FakeStagePrim) -> float | None:
        """Extract fake density overrides from prim metadata."""
        density = prim.metadata.get("density_g_cm3")
        if density not in (None, ""):
            return float(density)
        shader_inputs = prim.metadata.get("shader_inputs", {})
        if not isinstance(shader_inputs, dict):
            return None
        for key, value in shader_inputs.items():
            if self._is_density_parameter_name(str(key)):
                return float(value)
        return None

    def _fake_mass_att_by_isotope_from_prim(self, prim: FakeStagePrim) -> dict[str, float]:
        """Extract fake mass attenuation overrides from prim metadata."""
        result: dict[str, float] = {}
        explicit = prim.metadata.get("mass_att_by_isotope_cm2_g", {})
        if isinstance(explicit, dict):
            for isotope, value in explicit.items():
                result[self._normalize_isotope_name(str(isotope))] = float(value)
        shader_inputs = prim.metadata.get("shader_inputs", {})
        if isinstance(shader_inputs, dict):
            for key, value in shader_inputs.items():
                isotope = self._isotope_from_mass_att_parameter_name(str(key))
                if isotope is not None:
                    result[isotope] = float(value)
        return result

    def _fake_preset_name_from_prim(self, prim: FakeStagePrim) -> str | None:
        """Extract fake preset overrides from prim metadata."""
        preset = prim.metadata.get("material_preset")
        if preset not in (None, ""):
            return self._normalize_material_name(str(preset))
        shader_inputs = prim.metadata.get("shader_inputs", {})
        if not isinstance(shader_inputs, dict):
            return None
        for key, value in shader_inputs.items():
            if self._is_preset_parameter_name(str(key)) and value not in (None, ""):
                return self._normalize_material_name(str(value))
        return None

    def _fake_composition_by_mass_from_prim(self, prim: FakeStagePrim) -> dict[str, float]:
        """Extract fake composition overrides from prim metadata."""
        composition = prim.metadata.get("composition_by_mass")
        if isinstance(composition, (dict, str)):
            return normalize_composition_by_mass(composition)
        shader_inputs = prim.metadata.get("shader_inputs", {})
        if not isinstance(shader_inputs, dict):
            return {}
        for key, value in shader_inputs.items():
            if self._is_composition_parameter_name(str(key)) and value not in (None, ""):
                return normalize_composition_by_mass(str(value))
        return {}

    def _isotope_from_mu_parameter_name(self, parameter_name: str) -> str | None:
        """Map fake shader linear attenuation parameter names to isotope labels."""
        normalized = str(parameter_name).strip().lower()
        prefixes = ("simbridge_mu_", "simbridge:mu:")
        isotope_token = None
        for prefix in prefixes:
            if normalized.startswith(prefix):
                isotope_token = normalized[len(prefix) :]
                break
        if isotope_token is None:
            return None
        return self._normalize_isotope_name(isotope_token)

    def _isotope_from_mass_att_parameter_name(self, parameter_name: str) -> str | None:
        """Map fake shader mass attenuation parameter names to isotope labels."""
        normalized = str(parameter_name).strip().lower()
        prefixes = ("simbridge_mass_att_", "simbridge:mass_att:")
        isotope_token = None
        for prefix in prefixes:
            if normalized.startswith(prefix):
                isotope_token = normalized[len(prefix) :]
                break
        if isotope_token is None:
            return None
        for suffix in ("_cm2_g", ":cm2:g"):
            if isotope_token.endswith(suffix):
                isotope_token = isotope_token[: -len(suffix)]
                break
        return self._normalize_isotope_name(isotope_token)

    def _is_density_parameter_name(self, parameter_name: str) -> bool:
        """Return True when a parameter name encodes a density."""
        normalized = str(parameter_name).strip().lower()
        return normalized in {"simbridge_density_g_cm3", "simbridge:density:g_cm3"}

    def _is_preset_parameter_name(self, parameter_name: str) -> bool:
        """Return True when a parameter name encodes a preset override."""
        normalized = str(parameter_name).strip().lower()
        return normalized in {"simbridge_material_preset", "simbridge:preset"}

    def _is_composition_parameter_name(self, parameter_name: str) -> bool:
        """Return True when a parameter name encodes a composition override."""
        normalized = str(parameter_name).strip().lower()
        return normalized in {"simbridge_composition", "simbridge:composition"}

    def _normalize_isotope_name(self, token: str) -> str:
        """Normalize an isotope token into the repository naming convention."""
        normalized = str(token).strip().replace("-", "_").replace(" ", "_").lower()
        normalized = normalized.replace("__", "_")
        mapping = {
            "cs_137": "Cs-137",
            "co_60": "Co-60",
            "eu_154": "Eu-154",
        }
        return mapping.get(normalized, token)

    def _normalize_material_name(self, name: str) -> str:
        """Normalize a material prim name into a lookup key."""
        normalized = normalize_material_name(name)
        for suffix in ("_material", "material", "_mat", "mat"):
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                break
        return normalized


def _try_import_simulation_app() -> type[Any]:
    """Import the Isaac Sim application helper from supported package layouts."""
    try:
        from isaacsim.simulation_app import SimulationApp  # type: ignore

        return SimulationApp
    except ImportError:
        from isaacsim import SimulationApp  # type: ignore

        return SimulationApp


def merge_camera_gesture_bindings(
    default_bindings: Mapping[str, str],
    override_bindings: Mapping[str, str] | None,
) -> dict[str, str]:
    """Return Isaac Sim camera gesture bindings with user overrides applied."""
    merged = {str(gesture): str(binding) for gesture, binding in default_bindings.items()}
    if override_bindings:
        merged.update({str(gesture): str(binding) for gesture, binding in override_bindings.items()})
    return merged


def apply_camera_gesture_bindings_to_module(
    gestures_module: ModuleType,
    bindings: Mapping[str, str] | None,
) -> None:
    """Apply camera gesture overrides to an imported Isaac Sim gestures module."""
    if not bindings:
        return
    default_bindings = getattr(gestures_module, "kDefaultKeyBindings")
    merged_bindings = merge_camera_gesture_bindings(default_bindings, bindings)
    default_bindings.clear()
    default_bindings.update(merged_bindings)


def install_camera_gesture_binding_import_hook(bindings: Mapping[str, str] | None) -> None:
    """Install a one-shot hook that patches Isaac Sim camera bindings during import."""
    if not bindings:
        return
    target_module = "omni.kit.manipulator.camera.gestures"
    existing_module = sys.modules.get(target_module)
    if existing_module is not None:
        apply_camera_gesture_bindings_to_module(existing_module, bindings)
        return
    for finder in sys.meta_path:
        if getattr(finder, "_simbridge_camera_binding_hook", False):
            finder_bindings = getattr(finder, "_simbridge_camera_bindings")
            finder_bindings.update({str(gesture): str(binding) for gesture, binding in bindings.items()})
            return

    class _CameraGestureBindingLoader(importlib.abc.Loader):
        """Patch the target module immediately after the real loader executes it."""

        def __init__(
            self,
            wrapped_loader: importlib.abc.Loader,
            override_bindings: Mapping[str, str],
        ) -> None:
            """Store the loader and binding overrides."""
            self._wrapped_loader = wrapped_loader
            self._override_bindings = override_bindings

        def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
            """Delegate module creation to the wrapped loader when supported."""
            create_module = getattr(self._wrapped_loader, "create_module", None)
            if create_module is None:
                return None
            return create_module(spec)

        def exec_module(self, module: ModuleType) -> None:
            """Execute the real module and then apply the gesture override."""
            exec_module = getattr(self._wrapped_loader, "exec_module")
            exec_module(module)
            apply_camera_gesture_bindings_to_module(module, self._override_bindings)

    class _CameraGestureBindingFinder(importlib.abc.MetaPathFinder):
        """Find the Isaac Sim camera gestures module and wrap its loader."""

        _simbridge_camera_binding_hook = True

        def __init__(self, override_bindings: Mapping[str, str]) -> None:
            """Store binding overrides for the eventual gestures import."""
            self._simbridge_camera_bindings = {
                str(gesture): str(binding) for gesture, binding in override_bindings.items()
            }

        def find_spec(
            self,
            fullname: str,
            path: list[str] | None,
            target: ModuleType | None = None,
        ) -> importlib.machinery.ModuleSpec | None:
            """Return a wrapped module spec for the target gestures module."""
            if fullname != target_module:
                return None
            spec = None
            for finder in sys.meta_path:
                if finder is self or getattr(finder, "_simbridge_camera_binding_hook", False):
                    continue
                find_spec = getattr(finder, "find_spec", None)
                if find_spec is None:
                    continue
                spec = find_spec(fullname, path, target)
                if spec is not None:
                    break
            if spec is None or spec.loader is None:
                return spec
            spec.loader = _CameraGestureBindingLoader(spec.loader, self._simbridge_camera_bindings)
            return spec

    sys.meta_path.insert(0, _CameraGestureBindingFinder(bindings))


def _look_at_quaternion_wxyz(
    eye_xyz: tuple[float, float, float],
    target_xyz: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Return a USD camera orientation that looks from eye toward target."""
    eye = np.asarray(eye_xyz, dtype=float)
    target = np.asarray(target_xyz, dtype=float)
    forward = target - eye
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= 1e-12:
        raise ValueError("Camera eye and target must be different points.")
    forward = forward / forward_norm
    world_up = np.asarray((0.0, 0.0, 1.0), dtype=float)
    right = np.cross(forward, world_up)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1e-12:
        world_up = np.asarray((0.0, 1.0, 0.0), dtype=float)
        right = np.cross(forward, world_up)
        right_norm = float(np.linalg.norm(right))
    right = right / right_norm
    camera_up = np.cross(right, forward)
    rotation_matrix = np.column_stack((right, camera_up, -forward))
    return _rotation_matrix_to_quaternion_wxyz(rotation_matrix)


def _rotation_matrix_to_quaternion_wxyz(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a normalized wxyz quaternion."""
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = (trace + 1.0) ** 0.5 * 2.0
        qw = 0.25 * scale
        qx = (float(matrix[2, 1]) - float(matrix[1, 2])) / scale
        qy = (float(matrix[0, 2]) - float(matrix[2, 0])) / scale
        qz = (float(matrix[1, 0]) - float(matrix[0, 1])) / scale
    else:
        diagonal = [float(matrix[0, 0]), float(matrix[1, 1]), float(matrix[2, 2])]
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = (1.0 + float(matrix[0, 0]) - float(matrix[1, 1]) - float(matrix[2, 2])) ** 0.5 * 2.0
            qw = (float(matrix[2, 1]) - float(matrix[1, 2])) / scale
            qx = 0.25 * scale
            qy = (float(matrix[0, 1]) + float(matrix[1, 0])) / scale
            qz = (float(matrix[0, 2]) + float(matrix[2, 0])) / scale
        elif axis == 1:
            scale = (1.0 + float(matrix[1, 1]) - float(matrix[0, 0]) - float(matrix[2, 2])) ** 0.5 * 2.0
            qw = (float(matrix[0, 2]) - float(matrix[2, 0])) / scale
            qx = (float(matrix[0, 1]) + float(matrix[1, 0])) / scale
            qy = 0.25 * scale
            qz = (float(matrix[1, 2]) + float(matrix[2, 1])) / scale
        else:
            scale = (1.0 + float(matrix[2, 2]) - float(matrix[0, 0]) - float(matrix[1, 1])) ** 0.5 * 2.0
            qw = (float(matrix[1, 0]) - float(matrix[0, 1])) / scale
            qx = (float(matrix[0, 2]) + float(matrix[2, 0])) / scale
            qy = (float(matrix[1, 2]) + float(matrix[2, 1])) / scale
            qz = 0.25 * scale
    quat = np.asarray((qw, qx, qy, qz), dtype=float)
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


class IsaacSimStageBackend(StageBackend):
    """Use a live Isaac Sim Kit application and USD stage."""

    def __init__(
        self,
        *,
        headless: bool = True,
        renderer: str = "RayTracedLighting",
        camera_gesture_bindings: Mapping[str, str] | None = None,
    ) -> None:
        """Start Isaac Sim and defer Omniverse imports until Kit is running."""
        simulation_app_cls = _try_import_simulation_app()
        install_camera_gesture_binding_import_hook(camera_gesture_bindings)
        app_config = {
            "headless": bool(headless),
            "renderer": str(renderer),
        }
        self._simulation_app = simulation_app_cls(app_config)
        self._apply_camera_gesture_bindings(camera_gesture_bindings)
        from isaacsim.core.utils import stage as stage_utils  # type: ignore
        import omni.timeline  # type: ignore
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade  # type: ignore

        self._stage_utils = stage_utils
        self._timeline = omni.timeline.get_timeline_interface()
        self._Gf = Gf
        self._Sdf = Sdf
        self._Usd = Usd
        self._UsdGeom = UsdGeom
        self._UsdLux = UsdLux
        self._UsdShade = UsdShade
        self._stage = None

    def _apply_camera_gesture_bindings(self, bindings: Mapping[str, str] | None) -> None:
        """Apply optional viewport camera gesture bindings to Kit defaults."""
        if not bindings:
            return
        gestures_module = importlib.import_module("omni.kit.manipulator.camera.gestures")
        apply_camera_gesture_bindings_to_module(gestures_module, bindings)

    def open_stage(self, usd_path: str | None = None) -> None:
        """Open an existing stage or create an empty one."""
        if usd_path:
            resolved = Path(usd_path).expanduser().resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"USD scene not found: {resolved}")
            if not self._stage_utils.open_stage(resolved.as_posix()):
                raise RuntimeError(f"Failed to open USD stage: {resolved}")
        else:
            self._stage_utils.create_new_stage()
        self._simulation_app.update()
        self._stage = self._stage_utils.get_current_stage()
        self.ensure_xform("/World")
        self._timeline.play()
        self._simulation_app.update()

    def ensure_xform(self, path: str) -> None:
        """Ensure an Xform prim exists at the given path."""
        self._require_stage()
        self._UsdGeom.Xform.Define(self._stage, path)

    def ensure_box(
        self,
        path: str,
        *,
        size_xyz: tuple[float, float, float],
        translation_xyz: tuple[float, float, float],
        color_rgb: tuple[float, float, float] | None = None,
        material: str | None = None,
        transport_group: str | None = None,
    ) -> None:
        """Ensure a cube prim exists and scale it into a box."""
        self._require_stage()
        cube = self._UsdGeom.Cube.Define(self._stage, path)
        cube.CreateSizeAttr(1.0)
        self._set_display_color(cube, color_rgb)
        self._set_material_attr(cube.GetPrim(), material)
        self._set_transport_group_attr(cube.GetPrim(), transport_group)
        self.set_local_pose(
            path,
            translation_xyz=translation_xyz,
            scale_xyz=size_xyz,
        )

    def ensure_sphere(
        self,
        path: str,
        *,
        radius_m: float,
        translation_xyz: tuple[float, float, float],
        color_rgb: tuple[float, float, float] | None = None,
        material: str | None = None,
        transport_group: str | None = None,
    ) -> None:
        """Ensure a sphere prim exists with the requested radius."""
        self._require_stage()
        sphere = self._UsdGeom.Sphere.Define(self._stage, path)
        sphere.CreateRadiusAttr(float(radius_m))
        self._set_display_color(sphere, color_rgb)
        self._set_material_attr(sphere.GetPrim(), material)
        self._set_transport_group_attr(sphere.GetPrim(), transport_group)
        self.set_local_pose(path, translation_xyz=translation_xyz)

    def ensure_mesh(
        self,
        path: str,
        *,
        points_xyz: tuple[tuple[float, float, float], ...],
        face_vertex_counts: tuple[int, ...],
        face_vertex_indices: tuple[int, ...],
        translation_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
        color_rgb: tuple[float, float, float] | None = None,
        material: str | None = None,
        transport_group: str | None = None,
    ) -> None:
        """Ensure a mesh prim exists with the requested topology."""
        self._require_stage()
        mesh = self._UsdGeom.Mesh.Define(self._stage, path)
        mesh.GetPointsAttr().Set([self._Gf.Vec3f(*[float(v) for v in point]) for point in points_xyz])
        mesh.GetFaceVertexCountsAttr().Set([int(v) for v in face_vertex_counts])
        mesh.GetFaceVertexIndicesAttr().Set([int(v) for v in face_vertex_indices])
        self._set_display_color(mesh, color_rgb)
        self._set_material_attr(mesh.GetPrim(), material)
        self._set_transport_group_attr(mesh.GetPrim(), transport_group)
        self.set_local_pose(path, translation_xyz=translation_xyz)

    def ensure_polyline(
        self,
        path: str,
        *,
        points_xyz: tuple[tuple[float, float, float], ...],
        color_rgb: tuple[float, float, float] | None = None,
        width_m: float = 0.02,
    ) -> None:
        """Ensure a USD BasisCurves prim exists with linear segments."""
        self._require_stage()
        curves = self._UsdGeom.BasisCurves.Define(self._stage, path)
        curves.CreateTypeAttr("linear")
        curves.CreateCurveVertexCountsAttr([len(points_xyz)])
        curves.CreatePointsAttr([self._Gf.Vec3f(*[float(v) for v in point]) for point in points_xyz])
        curves.CreateWidthsAttr([float(width_m)])
        curves.SetWidthsInterpolation(self._UsdGeom.Tokens.constant)
        self._set_display_color(curves, color_rgb)

    def ensure_dome_light(
        self,
        path: str,
        *,
        intensity: float,
        color_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        """Ensure a USD DomeLight prim exists with the requested intensity."""
        self._require_stage()
        light = self._UsdLux.DomeLight.Define(self._stage, path)
        light.CreateIntensityAttr(float(intensity))
        light.CreateColorAttr(self._Gf.Vec3f(*[float(v) for v in color_rgb]))

    def ensure_sphere_light(
        self,
        path: str,
        *,
        intensity: float,
        radius_m: float,
        translation_xyz: tuple[float, float, float],
        color_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        """Ensure a USD SphereLight prim exists at the requested position."""
        self._require_stage()
        light = self._UsdLux.SphereLight.Define(self._stage, path)
        light.CreateIntensityAttr(float(intensity))
        light.CreateRadiusAttr(float(radius_m))
        light.CreateColorAttr(self._Gf.Vec3f(*[float(v) for v in color_rgb]))
        self.set_local_pose(path, translation_xyz=translation_xyz)

    def set_camera_view(
        self,
        path: str,
        *,
        eye_xyz: tuple[float, float, float],
        target_xyz: tuple[float, float, float],
        focal_length_mm: float = 24.0,
    ) -> None:
        """Create or update a USD camera and make it the active viewport camera."""
        self._require_stage()
        camera = self._UsdGeom.Camera.Define(self._stage, path)
        camera.CreateFocalLengthAttr(float(focal_length_mm))
        self.set_local_pose(
            path,
            translation_xyz=tuple(float(v) for v in eye_xyz),
            orientation_wxyz=_look_at_quaternion_wxyz(eye_xyz, target_xyz),
        )
        self._set_active_viewport_camera(path, eye_xyz=eye_xyz, target_xyz=target_xyz)

    def apply_visual_material(
        self,
        path_prefix: str,
        *,
        color_rgb: tuple[float, float, float],
        opacity: float = 1.0,
        roughness: float = 0.8,
        emissive_scale: float = 0.0,
    ) -> None:
        """Bind a simple visual material to matching USD geometry prims."""
        self._require_stage()
        material = self._define_visual_material(
            path_prefix,
            color_rgb=color_rgb,
            opacity=opacity,
            roughness=roughness,
            emissive_scale=emissive_scale,
        )
        prefix = str(path_prefix).rstrip("/")
        for prim in self._stage.Traverse():
            prim_path = str(prim.GetPath())
            if not prim_path.startswith(prefix):
                continue
            if prim.GetTypeName() not in {"Cube", "Sphere", "Mesh"}:
                continue
            geom = self._UsdGeom.Gprim(prim)
            self._set_display_color(geom, color_rgb)
            self._set_display_opacity(geom, opacity)
            self._set_visibility(prim, visible=float(opacity) > 0.0)
            self._UsdShade.MaterialBindingAPI(prim).Bind(material)

    def remove_prim(self, path: str) -> None:
        """Remove a USD prim subtree when it exists."""
        self._require_stage()
        self._stage.RemovePrim(self._Sdf.Path(str(path)))

    def set_local_pose(
        self,
        path: str,
        *,
        translation_xyz: tuple[float, float, float] | None = None,
        orientation_wxyz: tuple[float, float, float, float] | None = None,
        scale_xyz: tuple[float, float, float] | None = None,
    ) -> None:
        """Author translate/orient/scale ops on an Xformable prim."""
        self._require_stage()
        prim = self._stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise KeyError(f"Prim not found: {path}")
        xformable = self._UsdGeom.Xformable(prim)
        translate_op = self._get_or_add_xform_op(xformable, self._UsdGeom.XformOp.TypeTranslate)
        orient_op = self._get_or_add_xform_op(xformable, self._UsdGeom.XformOp.TypeOrient)
        scale_op = self._get_or_add_xform_op(xformable, self._UsdGeom.XformOp.TypeScale)
        if translation_xyz is not None:
            translate_op.Set(self._Gf.Vec3d(*[float(v) for v in translation_xyz]))
        if orientation_wxyz is not None:
            orient_op.Set(
                self._Gf.Quatf(
                    float(orientation_wxyz[0]),
                    self._Gf.Vec3f(
                        float(orientation_wxyz[1]),
                        float(orientation_wxyz[2]),
                        float(orientation_wxyz[3]),
                    ),
                )
            )
        if scale_xyz is not None:
            scale_op.Set(self._Gf.Vec3f(*[float(v) for v in scale_xyz]))

    def get_world_pose(self, path: str) -> PrimPose:
        """Read a prim world pose from its local-to-world transform."""
        self._require_stage()
        prim = self._stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise KeyError(f"Prim not found: {path}")
        xformable = self._UsdGeom.Xformable(prim)
        matrix = xformable.ComputeLocalToWorldTransform(self._Usd.TimeCode.Default())
        translation = matrix.ExtractTranslation()
        rotation = matrix.ExtractRotationQuat()
        real = float(rotation.GetReal())
        imag = rotation.GetImaginary()
        return PrimPose(
            translation_xyz=(float(translation[0]), float(translation[1]), float(translation[2])),
            orientation_wxyz=(real, float(imag[0]), float(imag[1]), float(imag[2])),
        )

    def list_solid_prims(self, path_prefixes: tuple[str, ...] | None = None) -> list[StageSolidPrim]:
        """Traverse the USD stage and return simple world-space solid geometry."""
        self._require_stage()
        prefixes = tuple(path_prefixes or ())
        result: list[StageSolidPrim] = []
        for prim in self._stage.Traverse():
            path = str(prim.GetPath())
            if prefixes and not any(path.startswith(prefix) for prefix in prefixes):
                continue
            prim_type = prim.GetTypeName()
            if prim_type not in {"Cube", "Sphere", "Mesh"}:
                continue
            pose = self.get_world_pose(path)
            material = self._material_from_prim(prim)
            if prim_type == "Cube":
                cube = self._UsdGeom.Cube(prim)
                size_attr = cube.GetSizeAttr()
                size = 1.0 if not size_attr or not size_attr.HasAuthoredValue() else float(size_attr.Get())
                scale_xyz = self._extract_local_scale_xyz(prim)
                result.append(
                    StageSolidPrim(
                        path=path,
                        shape="box",
                        pose=pose,
                        size_xyz=(
                            size * float(scale_xyz[0]),
                            size * float(scale_xyz[1]),
                            size * float(scale_xyz[2]),
                        ),
                        material_info=material,
                        transport_group=self._transport_group_from_prim(prim),
                    )
                )
                continue
            if prim_type == "Mesh":
                triangles = self._mesh_triangles_world(prim)
                box_geometry = _axis_aligned_box_from_triangles(triangles)
                if box_geometry is not None:
                    box_pose, box_size = box_geometry
                    result.append(
                        StageSolidPrim(
                            path=path,
                            shape="box",
                            pose=box_pose,
                            size_xyz=box_size,
                            material_info=material,
                            transport_group=self._transport_group_from_prim(prim),
                        )
                    )
                    continue
                result.append(
                    StageSolidPrim(
                        path=path,
                        shape="mesh",
                        pose=pose,
                        triangles_xyz=triangles,
                        material_info=material,
                        transport_group=self._transport_group_from_prim(prim),
                    )
                )
                continue
            sphere = self._UsdGeom.Sphere(prim)
            radius_attr = sphere.GetRadiusAttr()
            radius = 0.5 if not radius_attr or not radius_attr.HasAuthoredValue() else float(radius_attr.Get())
            scale_xyz = self._extract_local_scale_xyz(prim)
            result.append(
                StageSolidPrim(
                    path=path,
                    shape="sphere",
                    pose=pose,
                    radius_m=radius * float(max(scale_xyz)),
                    material_info=material,
                    transport_group=self._transport_group_from_prim(prim),
                )
            )
        return result

    def step(self) -> None:
        """Advance the Kit app by one update."""
        self._simulation_app.update()

    def close(self) -> None:
        """Stop the timeline and close the Kit application."""
        try:
            self._timeline.stop()
        except Exception:
            pass
        self._simulation_app.close()

    def _set_active_viewport_camera(
        self,
        path: str,
        *,
        eye_xyz: tuple[float, float, float],
        target_xyz: tuple[float, float, float],
    ) -> None:
        """Best-effort bind the active Isaac Sim viewport to the configured camera."""
        try:
            import omni.kit.viewport.utility as viewport_utility  # type: ignore
        except Exception:
            return
        try:
            viewport = viewport_utility.get_active_viewport()
        except Exception:
            return
        if viewport is None:
            return
        self._set_perspective_camera_view(viewport, eye_xyz=eye_xyz, target_xyz=target_xyz)
        try:
            if hasattr(viewport, "camera_path"):
                viewport.camera_path = self._Sdf.Path(str(path))
                return
            set_active_camera = getattr(viewport, "set_active_camera", None)
            if set_active_camera is not None:
                set_active_camera(str(path))
        except Exception:
            return

    def _define_visual_material(
        self,
        path_prefix: str,
        *,
        color_rgb: tuple[float, float, float],
        opacity: float,
        roughness: float,
        emissive_scale: float,
    ) -> Any:
        """Define a reusable UsdPreviewSurface material for visual overrides."""
        material_path = f"/World/SimBridge/Looks/{_visual_material_token(path_prefix)}"
        self._UsdGeom.Scope.Define(self._stage, "/World/SimBridge/Looks")
        material = self._UsdShade.Material.Define(self._stage, material_path)
        shader = self._UsdShade.Shader.Define(self._stage, f"{material_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", self._Sdf.ValueTypeNames.Color3f).Set(
            self._Gf.Vec3f(*[float(v) for v in color_rgb])
        )
        shader.CreateInput("roughness", self._Sdf.ValueTypeNames.Float).Set(float(roughness))
        shader.CreateInput("opacity", self._Sdf.ValueTypeNames.Float).Set(float(opacity))
        emissive_rgb = tuple(float(emissive_scale) * float(value) for value in color_rgb)
        shader.CreateInput("emissiveColor", self._Sdf.ValueTypeNames.Color3f).Set(
            self._Gf.Vec3f(*emissive_rgb)
        )
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material

    def _set_perspective_camera_view(
        self,
        viewport: Any,
        *,
        eye_xyz: tuple[float, float, float],
        target_xyz: tuple[float, float, float],
    ) -> None:
        """Best-effort set the standard Perspective viewport camera pose."""
        try:
            from isaacsim.core.utils.viewports import set_camera_view as set_isaac_camera_view  # type: ignore
        except Exception:
            return
        try:
            set_isaac_camera_view(
                eye=np.asarray(eye_xyz, dtype=float),
                target=np.asarray(target_xyz, dtype=float),
                camera_prim_path="/OmniverseKit_Persp",
                viewport_api=viewport,
            )
        except Exception:
            return

    def _require_stage(self) -> None:
        """Ensure the USD stage has been opened."""
        if self._stage is None:
            raise RuntimeError("USD stage is not initialized. Call open_stage() first.")

    def _get_or_add_xform_op(self, xformable: Any, op_type: Any) -> Any:
        """Return the first matching xform op, creating one when needed."""
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == op_type:
                return op
        if op_type == self._UsdGeom.XformOp.TypeTranslate:
            return xformable.AddTranslateOp()
        if op_type == self._UsdGeom.XformOp.TypeOrient:
            return xformable.AddOrientOp()
        if op_type == self._UsdGeom.XformOp.TypeScale:
            return xformable.AddScaleOp()
        raise ValueError(f"Unsupported xform op type: {op_type}")

    def _set_display_color(self, geom_prim: Any, color_rgb: tuple[float, float, float] | None) -> None:
        """Set a simple display color on a gprim."""
        if color_rgb is None:
            return
        color_attr = geom_prim.CreateDisplayColorAttr()
        vec = self._Gf.Vec3f(*[float(v) for v in color_rgb])
        color_attr.Set([vec])

    def _set_display_opacity(self, geom_prim: Any, opacity: float) -> None:
        """Set simple display opacity on a gprim."""
        opacity_attr = geom_prim.CreateDisplayOpacityAttr()
        opacity_attr.Set([float(opacity)])

    def _set_visibility(self, prim: Any, *, visible: bool) -> None:
        """Set USD visibility for visual-only hidden geometry."""
        imageable = self._UsdGeom.Imageable(prim)
        visibility_attr = imageable.CreateVisibilityAttr()
        visibility_attr.Set("inherited" if visible else "invisible")

    def _set_material_attr(self, prim: Any, material: str | None) -> None:
        """Author a lightweight custom material attribute on a prim."""
        if material is None:
            return
        attr = prim.CreateAttribute("simbridge:material", self._Sdf.ValueTypeNames.String, custom=True)
        attr.Set(str(material))

    def _set_transport_group_attr(self, prim: Any, transport_group: str | None) -> None:
        """Author a lightweight semantic transport group attribute on a prim."""
        if transport_group in (None, ""):
            return
        attr = prim.CreateAttribute("simbridge:transport_group", self._Sdf.ValueTypeNames.String, custom=True)
        attr.Set(_normalize_transport_group(str(transport_group)))

    def _transport_group_from_prim(self, prim: Any) -> str | None:
        """Resolve semantic transport grouping from USD attributes or path tokens."""
        for name in (
            "simbridge:transport_group",
            "simbridge_transport_group",
            "simbridge:category",
            "simbridge_category",
        ):
            attr = prim.GetAttribute(name)
            if attr and attr.HasAuthoredValue():
                value = attr.Get()
                if value not in (None, ""):
                    return _normalize_transport_group(str(value))
            try:
                custom_value = prim.GetCustomDataByKey(name)
            except Exception:
                custom_value = None
            if custom_value not in (None, ""):
                return _normalize_transport_group(str(custom_value))
        return _infer_transport_group_from_path(str(prim.GetPath()))

    def _material_from_prim(self, prim: Any) -> StageMaterialInfo | None:
        """Resolve a material name and attenuation overrides from a prim."""
        attr = prim.GetAttribute("simbridge:material")
        direct_name = None
        if attr and attr.HasAuthoredValue():
            value = attr.Get()
            if value not in (None, ""):
                direct_name = str(value)
        bound_material = self._bound_material_from_prim(prim)
        if direct_name is None and bound_material is None:
            return None
        if direct_name is not None and bound_material is None:
            return StageMaterialInfo(
                name=direct_name,
                mu_by_isotope=self._prim_mu_by_isotope(prim),
                density_g_cm3=self._prim_density_g_cm3(prim),
                mass_att_by_isotope_cm2_g=self._prim_mass_att_by_isotope_cm2_g(prim),
                preset_name=self._prim_preset_name(prim),
                composition_by_mass=self._prim_composition_by_mass(prim),
            )
        if direct_name is not None and bound_material is not None:
            direct_density_g_cm3 = self._prim_density_g_cm3(prim)
            return StageMaterialInfo(
                name=direct_name,
                mu_by_isotope={
                    **dict(bound_material.mu_by_isotope),
                    **self._prim_mu_by_isotope(prim),
                },
                density_g_cm3=(
                    direct_density_g_cm3
                    if direct_density_g_cm3 is not None
                    else bound_material.density_g_cm3
                ),
                mass_att_by_isotope_cm2_g={
                    **dict(bound_material.mass_att_by_isotope_cm2_g),
                    **self._prim_mass_att_by_isotope_cm2_g(prim),
                },
                preset_name=self._prim_preset_name(prim) or bound_material.preset_name,
                composition_by_mass=self._prim_composition_by_mass(prim) or dict(bound_material.composition_by_mass),
            )
        return bound_material

    def _extract_local_scale_xyz(self, prim: Any) -> tuple[float, float, float]:
        """Read the authored local scale op from an xformable prim."""
        xformable = self._UsdGeom.Xformable(prim)
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == self._UsdGeom.XformOp.TypeScale:
                value = op.Get()
                return (float(value[0]), float(value[1]), float(value[2]))
        return (1.0, 1.0, 1.0)

    def _bound_material_from_prim(self, prim: Any) -> StageMaterialInfo | None:
        """Resolve a material from standard USD material bindings."""
        try:
            material, _ = self._UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        except Exception:
            return None
        if material is None:
            return None
        material_prim = material.GetPrim()
        if material_prim is None or not material_prim.IsValid():
            return None
        attr = material_prim.GetAttribute("simbridge:material")
        name = None
        if attr and attr.HasAuthoredValue():
            value = attr.Get()
            if value not in (None, ""):
                name = str(value)
        if name is None:
            name = self._normalize_material_name(material_prim.GetName())
        return StageMaterialInfo(
            name=name,
            mu_by_isotope=self._material_mu_by_isotope(material),
            density_g_cm3=self._material_density_g_cm3(material),
            mass_att_by_isotope_cm2_g=self._material_mass_att_by_isotope_cm2_g(material),
            preset_name=self._material_preset_name(material),
            composition_by_mass=self._material_composition_by_mass(material),
        )

    def _normalize_material_name(self, name: str) -> str:
        """Normalize a material prim name into a lookup key."""
        normalized = normalize_material_name(name)
        for suffix in ("_material", "material", "_mat", "mat"):
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                break
        return normalized

    def _material_mu_by_isotope(self, material: Any) -> dict[str, float]:
        """Read attenuation overrides from a material or its surface shader."""
        result: dict[str, float] = {}
        for input_obj in material.GetInputs():
            isotope = self._isotope_from_mu_parameter_name(str(input_obj.GetBaseName()))
            if isotope is None:
                continue
            value = input_obj.Get()
            if value is None:
                continue
            result[isotope] = float(value)
        try:
            surface_source = material.ComputeSurfaceSource()
        except Exception:
            surface_source = None
        shader = None
        if isinstance(surface_source, tuple):
            shader = surface_source[0] if surface_source else None
        else:
            shader = surface_source
        if shader:
            for input_obj in shader.GetInputs():
                isotope = self._isotope_from_mu_parameter_name(str(input_obj.GetBaseName()))
                if isotope is None:
                    continue
                value = input_obj.Get()
                if value is None:
                    continue
                result[isotope] = float(value)
        return result

    def _material_density_g_cm3(self, material: Any) -> float | None:
        """Read density overrides from a material or its surface shader."""
        for input_obj in material.GetInputs():
            if self._is_density_parameter_name(str(input_obj.GetBaseName())):
                value = input_obj.Get()
                if value is not None:
                    return float(value)
        try:
            surface_source = material.ComputeSurfaceSource()
        except Exception:
            surface_source = None
        shader = surface_source[0] if isinstance(surface_source, tuple) and surface_source else surface_source
        if shader:
            for input_obj in shader.GetInputs():
                if self._is_density_parameter_name(str(input_obj.GetBaseName())):
                    value = input_obj.Get()
                    if value is not None:
                        return float(value)
        return None

    def _material_preset_name(self, material: Any) -> str | None:
        """Read preset overrides from a material or its surface shader."""
        for input_obj in material.GetInputs():
            if not self._is_preset_parameter_name(str(input_obj.GetBaseName())):
                continue
            value = input_obj.Get()
            if value not in (None, ""):
                return self._normalize_material_name(str(value))
        try:
            surface_source = material.ComputeSurfaceSource()
        except Exception:
            surface_source = None
        shader = surface_source[0] if isinstance(surface_source, tuple) and surface_source else surface_source
        if shader:
            for input_obj in shader.GetInputs():
                if not self._is_preset_parameter_name(str(input_obj.GetBaseName())):
                    continue
                value = input_obj.Get()
                if value not in (None, ""):
                    return self._normalize_material_name(str(value))
        return None

    def _material_composition_by_mass(self, material: Any) -> dict[str, float]:
        """Read composition overrides from a material or its surface shader."""
        for input_obj in material.GetInputs():
            if not self._is_composition_parameter_name(str(input_obj.GetBaseName())):
                continue
            value = input_obj.Get()
            if value not in (None, ""):
                return normalize_composition_by_mass(str(value))
        try:
            surface_source = material.ComputeSurfaceSource()
        except Exception:
            surface_source = None
        shader = surface_source[0] if isinstance(surface_source, tuple) and surface_source else surface_source
        if shader:
            for input_obj in shader.GetInputs():
                if not self._is_composition_parameter_name(str(input_obj.GetBaseName())):
                    continue
                value = input_obj.Get()
                if value not in (None, ""):
                    return normalize_composition_by_mass(str(value))
        return {}

    def _material_mass_att_by_isotope_cm2_g(self, material: Any) -> dict[str, float]:
        """Read mass attenuation overrides from a material or its surface shader."""
        result: dict[str, float] = {}
        for input_obj in material.GetInputs():
            isotope = self._isotope_from_mass_att_parameter_name(str(input_obj.GetBaseName()))
            if isotope is None:
                continue
            value = input_obj.Get()
            if value is None:
                continue
            result[isotope] = float(value)
        try:
            surface_source = material.ComputeSurfaceSource()
        except Exception:
            surface_source = None
        shader = surface_source[0] if isinstance(surface_source, tuple) and surface_source else surface_source
        if shader:
            for input_obj in shader.GetInputs():
                isotope = self._isotope_from_mass_att_parameter_name(str(input_obj.GetBaseName()))
                if isotope is None:
                    continue
                value = input_obj.Get()
                if value is None:
                    continue
                result[isotope] = float(value)
        return result

    def _isotope_from_mu_parameter_name(self, parameter_name: str) -> str | None:
        """Map shader or material linear attenuation parameter names to isotope labels."""
        normalized = str(parameter_name).strip().lower()
        prefixes = ("simbridge_mu_", "simbridge:mu:")
        isotope_token = None
        for prefix in prefixes:
            if normalized.startswith(prefix):
                isotope_token = normalized[len(prefix) :]
                break
        if isotope_token is None:
            return None
        return self._normalize_isotope_name(isotope_token)

    def _isotope_from_mass_att_parameter_name(self, parameter_name: str) -> str | None:
        """Map shader or material mass attenuation parameter names to isotope labels."""
        normalized = str(parameter_name).strip().lower()
        prefixes = ("simbridge_mass_att_", "simbridge:mass_att:")
        isotope_token = None
        for prefix in prefixes:
            if normalized.startswith(prefix):
                isotope_token = normalized[len(prefix) :]
                break
        if isotope_token is None:
            return None
        for suffix in ("_cm2_g", ":cm2:g"):
            if isotope_token.endswith(suffix):
                isotope_token = isotope_token[: -len(suffix)]
                break
        return self._normalize_isotope_name(isotope_token)

    def _is_density_parameter_name(self, parameter_name: str) -> bool:
        """Return True when a parameter name encodes a density."""
        normalized = str(parameter_name).strip().lower()
        return normalized in {"simbridge_density_g_cm3", "simbridge:density:g_cm3"}

    def _is_preset_parameter_name(self, parameter_name: str) -> bool:
        """Return True when a parameter name encodes a preset override."""
        normalized = str(parameter_name).strip().lower()
        return normalized in {"simbridge_material_preset", "simbridge:preset"}

    def _is_composition_parameter_name(self, parameter_name: str) -> bool:
        """Return True when a parameter name encodes a composition override."""
        normalized = str(parameter_name).strip().lower()
        return normalized in {"simbridge_composition", "simbridge:composition"}

    def _normalize_isotope_name(self, token: str) -> str:
        """Normalize an isotope token into the repository naming convention."""
        normalized = str(token).strip().replace("-", "_").replace(" ", "_").lower()
        normalized = normalized.replace("__", "_")
        mapping = {
            "cs_137": "Cs-137",
            "co_60": "Co-60",
            "eu_154": "Eu-154",
        }
        return mapping.get(normalized, token)

    def _prim_mu_by_isotope(self, prim: Any) -> dict[str, float]:
        """Read linear attenuation overrides from prim attributes."""
        result: dict[str, float] = {}
        for attr in prim.GetAttributes():
            isotope = self._isotope_from_mu_parameter_name(str(attr.GetBaseName()))
            if isotope is None or not attr.HasAuthoredValue():
                continue
            value = attr.Get()
            if value is None:
                continue
            result[isotope] = float(value)
        return result

    def _prim_density_g_cm3(self, prim: Any) -> float | None:
        """Read density overrides from prim attributes."""
        for attr in prim.GetAttributes():
            if not self._is_density_parameter_name(str(attr.GetBaseName())):
                continue
            if not attr.HasAuthoredValue():
                continue
            value = attr.Get()
            if value is not None:
                return float(value)
        return None

    def _prim_mass_att_by_isotope_cm2_g(self, prim: Any) -> dict[str, float]:
        """Read mass attenuation overrides from prim attributes."""
        result: dict[str, float] = {}
        for attr in prim.GetAttributes():
            isotope = self._isotope_from_mass_att_parameter_name(str(attr.GetBaseName()))
            if isotope is None or not attr.HasAuthoredValue():
                continue
            value = attr.Get()
            if value is None:
                continue
            result[isotope] = float(value)
        return result

    def _prim_preset_name(self, prim: Any) -> str | None:
        """Read preset overrides from prim attributes."""
        for attr in prim.GetAttributes():
            if not self._is_preset_parameter_name(str(attr.GetBaseName())):
                continue
            if not attr.HasAuthoredValue():
                continue
            value = attr.Get()
            if value not in (None, ""):
                return self._normalize_material_name(str(value))
        return None

    def _prim_composition_by_mass(self, prim: Any) -> dict[str, float]:
        """Read composition overrides from prim attributes."""
        for attr in prim.GetAttributes():
            if not self._is_composition_parameter_name(str(attr.GetBaseName())):
                continue
            if not attr.HasAuthoredValue():
                continue
            value = attr.Get()
            if value not in (None, ""):
                return normalize_composition_by_mass(str(value))
        return {}

    def _mesh_triangles_world(
        self,
        prim: Any,
    ) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]], ...]:
        """Triangulate a mesh prim into world-space triangles."""
        mesh = self._UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get() or []
        counts = mesh.GetFaceVertexCountsAttr().Get() or []
        indices = mesh.GetFaceVertexIndicesAttr().Get() or []
        transform = self._UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(self._Usd.TimeCode.Default())
        world_points: list[tuple[float, float, float]] = []
        for point in points:
            world_point = transform.Transform(self._Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
            world_points.append((float(world_point[0]), float(world_point[1]), float(world_point[2])))
        triangles: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]] = []
        cursor = 0
        for count in counts:
            face_count = int(count)
            face_indices = [int(indices[cursor + idx]) for idx in range(face_count)]
            cursor += face_count
            if face_count < 3:
                continue
            anchor = world_points[face_indices[0]]
            for tri_index in range(1, face_count - 1):
                triangles.append(
                    (
                        anchor,
                        world_points[face_indices[tri_index]],
                        world_points[face_indices[tri_index + 1]],
                    )
                )
        return tuple(triangles)



def yaw_to_quaternion_wxyz(yaw_rad: float) -> tuple[float, float, float, float]:
    """Convert a Z-axis yaw angle into a quaternion."""
    half = 0.5 * float(yaw_rad)
    return (cos(half), 0.0, 0.0, sin(half))


def _visual_material_token(path_prefix: str) -> str:
    """Return a stable USD-safe material token for a path prefix."""
    token = str(path_prefix).strip("/").replace("/", "_").replace(":", "_")
    token = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in token)
    token = token.strip("_") or "VisualMaterial"
    if token[0].isdigit():
        token = f"Visual_{token}"
    return f"{token}_Visual"
