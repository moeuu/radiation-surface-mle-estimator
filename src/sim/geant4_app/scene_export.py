"""Scene export helpers for the Geant4 sidecar."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

from measurement.detector_geometry import (
    detector_outer_radius_cm as detector_outer_radius_cm_from_model,
)
from sim.isaacsim_app.observation_model import IsaacAssetGeometry
from sim.isaacsim_app.scene_builder import SceneDescription, StagePrimPaths
from sim.isaacsim_app.stage_backend import StageBackend, StageMaterialInfo
from sim.shield_geometry import (
    SHIELD_SHAPE_SPHERICAL_OCTANT,
    ShieldThicknessConfig,
    nested_shield_inner_radii_cm,
    resolve_shield_thickness_config,
)

DEFAULT_DETECTOR_CRYSTAL_RADIUS_M = 0.038
DEFAULT_DETECTOR_CRYSTAL_LENGTH_M = 0.076
DEFAULT_DETECTOR_HOUSING_THICKNESS_M = 0.0015


@dataclass(frozen=True)
class ExportedGeant4Material:
    """Describe a Geant4-ready material definition exported from the stage."""

    name: str
    density_g_cm3: float | None = None
    mu_by_isotope: dict[str, float] = field(default_factory=dict)
    mass_att_by_isotope_cm2_g: dict[str, float] = field(default_factory=dict)
    preset_name: str | None = None
    composition_by_mass: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable material payload."""
        return {
            "name": self.name,
            "density_g_cm3": self.density_g_cm3,
            "mu_by_isotope": dict(sorted(self.mu_by_isotope.items())),
            "mass_att_by_isotope_cm2_g": dict(sorted(self.mass_att_by_isotope_cm2_g.items())),
            "preset_name": self.preset_name,
            "composition_by_mass": dict(sorted(self.composition_by_mass.items())),
        }

    @classmethod
    def from_stage_material(cls, material_info: StageMaterialInfo) -> "ExportedGeant4Material":
        """Build an exported material payload from a stage material record."""
        return cls(
            name=str(material_info.name),
            density_g_cm3=material_info.density_g_cm3,
            mu_by_isotope={str(key): float(value) for key, value in material_info.mu_by_isotope.items()},
            mass_att_by_isotope_cm2_g={
                str(key): float(value) for key, value in material_info.mass_att_by_isotope_cm2_g.items()
            },
            preset_name=None if material_info.preset_name in (None, "") else str(material_info.preset_name),
            composition_by_mass={
                str(key): float(value) for key, value in material_info.composition_by_mass.items()
            },
        )


@dataclass(frozen=True)
class ExportedGeant4Volume:
    """Describe a static volume exported from the stage."""

    path: str
    shape: str
    translation_xyz: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    size_xyz: tuple[float, float, float] | None = None
    radius_m: float | None = None
    triangles_xyz: tuple[
        tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
        ...,
    ] | None = None
    material: ExportedGeant4Material | None = None
    transport_group: str | None = None
    transport_mode: str = "geant4"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable volume payload."""
        return {
            "path": self.path,
            "shape": self.shape,
            "translation_xyz": list(self.translation_xyz),
            "orientation_wxyz": list(self.orientation_wxyz),
            "size_xyz": None if self.size_xyz is None else list(self.size_xyz),
            "radius_m": self.radius_m,
            "triangles_xyz": None if self.triangles_xyz is None else self.triangles_xyz,
            "material": None if self.material is None else self.material.to_dict(),
            "transport_group": self.transport_group,
            "transport_mode": self.transport_mode,
        }


@dataclass(frozen=True)
class ExportedGeant4Source:
    """Describe a source term exported for Geant4 transport."""

    isotope: str
    position_xyz: tuple[float, float, float]
    intensity_cps_1m: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable source payload."""
        return {
            "isotope": self.isotope,
            "position_xyz": list(self.position_xyz),
            "intensity_cps_1m": float(self.intensity_cps_1m),
        }


@dataclass(frozen=True)
class ExportedDetectorModel:
    """Describe the detector assembly used by the Geant4 engine."""

    crystal_radius_m: float = DEFAULT_DETECTOR_CRYSTAL_RADIUS_M
    crystal_length_m: float = DEFAULT_DETECTOR_CRYSTAL_LENGTH_M
    housing_thickness_m: float = DEFAULT_DETECTOR_HOUSING_THICKNESS_M
    crystal_shape: str = "sphere"
    crystal_material: str = "cebr3"
    housing_material: str = "aluminum"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable detector payload."""
        return {
            "crystal_radius_m": float(self.crystal_radius_m),
            "crystal_length_m": float(self.crystal_length_m),
            "housing_thickness_m": float(self.housing_thickness_m),
            "crystal_shape": self.crystal_shape,
            "crystal_material": self.crystal_material,
            "housing_material": self.housing_material,
        }

    @property
    def active_volume_m3(self) -> float:
        """Return the active crystal volume in cubic meters."""
        radius_m = max(0.0, float(self.crystal_radius_m))
        length_m = max(0.0, float(self.crystal_length_m))
        if self.crystal_shape.strip().lower() == "sphere":
            return float((4.0 / 3.0) * 3.141592653589793 * radius_m**3)
        return float(3.141592653589793 * radius_m * radius_m * length_m)


@dataclass(frozen=True)
class ExportedShieldModel:
    """Describe a movable shield volume."""

    path: str
    shape: str
    inner_radius_m: float
    outer_radius_m: float
    thickness_cm: float
    size_xyz: tuple[float, float, float] | None
    material: ExportedGeant4Material
    use_angle_attenuation: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable shield payload."""
        return {
            "path": self.path,
            "shape": self.shape,
            "inner_radius_m": float(self.inner_radius_m),
            "outer_radius_m": float(self.outer_radius_m),
            "thickness_cm": float(self.thickness_cm),
            "thickness_m": float(self.thickness_cm) / 100.0,
            "size_xyz": None if self.size_xyz is None else list(self.size_xyz),
            "material": self.material.to_dict(),
            "use_angle_attenuation": bool(self.use_angle_attenuation),
        }


@dataclass(frozen=True)
class ExportedGeant4Scene:
    """Describe the Geant4-ready scene built from the USD stage."""

    scene_hash: str
    usd_path: str | None
    room_size_xyz: tuple[float, float, float]
    static_volumes: tuple[ExportedGeant4Volume, ...]
    sources: tuple[ExportedGeant4Source, ...]
    detector_model: ExportedDetectorModel
    fe_shield: ExportedShieldModel
    pb_shield: ExportedShieldModel
    prim_paths: StagePrimPaths

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable scene payload."""
        return {
            "scene_hash": self.scene_hash,
            "usd_path": self.usd_path,
            "room_size_xyz": list(self.room_size_xyz),
            "static_volumes": [volume.to_dict() for volume in self.static_volumes],
            "sources": [source.to_dict() for source in self.sources],
            "detector_model": self.detector_model.to_dict(),
            "fe_shield": self.fe_shield.to_dict(),
            "pb_shield": self.pb_shield.to_dict(),
            "prim_paths": {
                "world_root": self.prim_paths.world_root,
                "generated_root": self.prim_paths.generated_root,
                "obstacles_root": self.prim_paths.obstacles_root,
                "sources_root": self.prim_paths.sources_root,
                "robot_root": self.prim_paths.robot_root,
                "detector_path": self.prim_paths.detector_path,
                "fe_shield_path": self.prim_paths.fe_shield_path,
                "pb_shield_path": self.prim_paths.pb_shield_path,
            },
        }


def export_scene_for_geant4(
    scene: SceneDescription,
    *,
    stage_backend: StageBackend,
    asset_geometry: IsaacAssetGeometry,
    detector_model: ExportedDetectorModel,
    shield_thickness: ShieldThicknessConfig | None = None,
    stage_material_rules: tuple[object, ...] = (),
    absorbing_transport_groups: tuple[str, ...] = (),
    absorbing_path_prefixes: tuple[str, ...] = (),
) -> ExportedGeant4Scene:
    """Export the loaded stage and dynamic assets into a Geant4-ready scene."""
    shield_thickness = shield_thickness or resolve_shield_thickness_config()
    prefixes = tuple(
        sorted(
            {
                *{str(getattr(rule, "path_prefix")) for rule in stage_material_rules},
                scene.prim_paths.obstacles_root,
                "/World/Environment",
            }
        )
    )
    static_volumes: list[ExportedGeant4Volume] = []
    for solid_prim in stage_backend.list_solid_prims(path_prefixes=prefixes):
        if solid_prim.path in {
            scene.prim_paths.detector_path,
            scene.prim_paths.fe_shield_path,
            scene.prim_paths.pb_shield_path,
        }:
            continue
        if solid_prim.path.startswith(scene.prim_paths.sources_root):
            continue
        material = _resolve_material_for_export(
            solid_prim.path,
            solid_prim.material_info,
            scene=scene,
            stage_material_rules=stage_material_rules,
        )
        transport_group = _normalize_transport_group(solid_prim.transport_group)
        transport_mode = _resolve_transport_mode(
            solid_prim.path,
            transport_group=transport_group,
            absorbing_transport_groups=absorbing_transport_groups,
            absorbing_path_prefixes=absorbing_path_prefixes,
        )
        static_volumes.append(
            ExportedGeant4Volume(
                path=solid_prim.path,
                shape=str(solid_prim.shape),
                translation_xyz=tuple(float(v) for v in solid_prim.pose.translation_xyz),
                orientation_wxyz=tuple(float(v) for v in solid_prim.pose.orientation_wxyz),
                size_xyz=None if solid_prim.size_xyz is None else tuple(float(v) for v in solid_prim.size_xyz),
                radius_m=None if solid_prim.radius_m is None else float(solid_prim.radius_m),
                triangles_xyz=solid_prim.triangles_xyz,
                material=None if material is None else ExportedGeant4Material.from_stage_material(material),
                transport_group=transport_group,
                transport_mode=transport_mode,
            )
        )
    static_volumes.sort(key=lambda volume: volume.path)
    sources = tuple(
        ExportedGeant4Source(
            isotope=source.isotope,
            position_xyz=tuple(float(v) for v in source.position_xyz),
            intensity_cps_1m=float(source.intensity_cps_1m),
        )
        for source in scene.sources
    )
    detector_outer_radius_cm = detector_outer_radius_cm_from_model(
        detector_model.to_dict()
    )
    fe_inner_cm, pb_inner_cm = nested_shield_inner_radii_cm(
        thickness_fe_cm=float(shield_thickness.thickness_fe_cm),
        detector_outer_radius_cm=detector_outer_radius_cm,
    )
    fe_shield = ExportedShieldModel(
        path=scene.prim_paths.fe_shield_path,
        shape=SHIELD_SHAPE_SPHERICAL_OCTANT,
        inner_radius_m=fe_inner_cm / 100.0,
        outer_radius_m=(fe_inner_cm + float(shield_thickness.thickness_fe_cm)) / 100.0,
        thickness_cm=float(shield_thickness.thickness_fe_cm),
        size_xyz=None,
        material=ExportedGeant4Material(name="fe", preset_name="iron"),
    )
    pb_shield = ExportedShieldModel(
        path=scene.prim_paths.pb_shield_path,
        shape=SHIELD_SHAPE_SPHERICAL_OCTANT,
        inner_radius_m=pb_inner_cm / 100.0,
        outer_radius_m=(pb_inner_cm + float(shield_thickness.thickness_pb_cm)) / 100.0,
        thickness_cm=float(shield_thickness.thickness_pb_cm),
        size_xyz=None,
        material=ExportedGeant4Material(name="pb", preset_name="lead"),
    )
    scene_payload = ExportedGeant4Scene(
        scene_hash="",
        usd_path=scene.usd_path,
        room_size_xyz=tuple(float(v) for v in scene.room_size_xyz),
        static_volumes=tuple(static_volumes),
        sources=sources,
        detector_model=detector_model,
        fe_shield=fe_shield,
        pb_shield=pb_shield,
        prim_paths=scene.prim_paths,
    )
    stable_payload = scene_payload.to_dict()
    stable_payload.pop("scene_hash", None)
    scene_hash = hashlib.sha256(json.dumps(stable_payload, sort_keys=True).encode("utf-8")).hexdigest()
    return ExportedGeant4Scene(
        scene_hash=scene_hash,
        usd_path=scene_payload.usd_path,
        room_size_xyz=scene_payload.room_size_xyz,
        static_volumes=scene_payload.static_volumes,
        sources=scene_payload.sources,
        detector_model=scene_payload.detector_model,
        fe_shield=scene_payload.fe_shield,
        pb_shield=scene_payload.pb_shield,
        prim_paths=scene_payload.prim_paths,
    )


def _resolve_material_for_export(
    path: str,
    prim_material: StageMaterialInfo | None,
    *,
    scene: SceneDescription,
    stage_material_rules: tuple[object, ...],
) -> StageMaterialInfo | None:
    """Resolve the exported material for a stage solid."""
    if prim_material is not None:
        return prim_material
    matched_material: str | None = None
    matched_len = -1
    for rule in stage_material_rules:
        path_prefix = str(getattr(rule, "path_prefix"))
        if path.startswith(path_prefix) and len(path_prefix) > matched_len:
            matched_material = str(getattr(rule, "material"))
            matched_len = len(path_prefix)
    if matched_material is None and path.startswith(scene.prim_paths.obstacles_root):
        return StageMaterialInfo(name=scene.obstacle_material)
    if matched_material is None and path.startswith("/World/Environment"):
        return StageMaterialInfo(name="concrete")
    if matched_material is None:
        return None
    return StageMaterialInfo(name=matched_material)


def _normalize_transport_group(value: str | None) -> str | None:
    """Normalize an optional semantic transport group."""
    if value in (None, ""):
        return None
    return str(value).strip().replace("-", "_").replace(" ", "_").lower()


def _resolve_transport_mode(
    path: str,
    *,
    transport_group: str | None,
    absorbing_transport_groups: tuple[str, ...],
    absorbing_path_prefixes: tuple[str, ...],
) -> str:
    """Return the Geant4 transport mode for an exported static volume."""
    absorbing_groups = {
        normalized
        for normalized in (_normalize_transport_group(group) for group in absorbing_transport_groups)
        if normalized not in (None, "")
    }
    if transport_group is not None and transport_group in absorbing_groups:
        return "absorber"
    prefixes = tuple(str(prefix).rstrip("/") for prefix in absorbing_path_prefixes if str(prefix).strip())
    if any(path == prefix or path.startswith(f"{prefix}/") for prefix in prefixes):
        return "absorber"
    return "geant4"
