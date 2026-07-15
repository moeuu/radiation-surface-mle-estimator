"""Approximate Python observation generation for Isaac/mock debug sidecars."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from measurement.detector_geometry import (
    detector_outer_radius_cm as detector_outer_radius_cm_from_model,
)
from measurement.kernels import ShieldParams
from measurement.obstacles import ObstacleGrid
from measurement.shielding import HVL_TVL_TABLE_MM, mu_by_isotope_from_tvl_mm
from sim.isaacsim_app.geometry import (
    OrientedBox,
    Sphere,
    TriangleMesh,
    quaternion_wxyz_to_matrix,
    segment_path_length_through_box,
    segment_path_length_through_mesh,
    segment_path_length_through_sphere,
)
from sim.isaacsim_app.robot_controller import RobotController
from sim.isaacsim_app.scene_builder import SceneDescription
from sim.isaacsim_app.stage_backend import StageMaterialInfo
from sim.approx.python_transport import PythonTransportSpectrumModel
from sim.protocol import SimulationCommand, SimulationObservation
from sim.transport import (
    TransportSegment,
    make_transport_segment,
)
from sim.shield_geometry import (
    ShieldThicknessConfig,
    nested_shield_inner_radii_cm,
    resolve_shield_thickness_config,
    spherical_octant_path_length_cm,
)
from spectrum.pipeline import SpectralDecomposer


@dataclass(frozen=True)
class IsaacAssetGeometry:
    """Collect simple geometric parameters for the bridge-authored assets."""

    detector_height_m: float = 0.5
    obstacle_height_m: float = 2.0
    fe_shield_size_xyz: tuple[float, float, float] = (0.25, 0.08, 0.25)
    pb_shield_size_xyz: tuple[float, float, float] = (0.25, 0.08, 0.25)


@dataclass(frozen=True)
class StageMaterialRule:
    """Map a stage path prefix to a named attenuation material."""

    path_prefix: str
    material: str


class ObservationModel(ABC):
    """Define the observation interface used by the bridge app."""

    @abstractmethod
    def reset(self, scene: SceneDescription) -> None:
        """Update any internal state after a new scene is loaded."""

    @abstractmethod
    def observe(self, command: SimulationCommand) -> SimulationObservation:
        """Return an observation for the requested command."""


def _obstacle_grid_from_scene(scene: SceneDescription) -> ObstacleGrid | None:
    """Build an obstacle grid from the scene description when cells exist."""
    if scene.obstacle_grid_shape[0] <= 0 or scene.obstacle_grid_shape[1] <= 0:
        return None
    return ObstacleGrid(
        origin=scene.obstacle_origin_xy,
        cell_size=scene.obstacle_cell_size_m,
        grid_shape=scene.obstacle_grid_shape,
        blocked_cells=tuple(scene.obstacle_cells),
    )


class MockObservationModel(ObservationModel):
    """Generate bridge observations without requiring Isaac Sim installation."""

    def __init__(
        self,
        *,
        asset_geometry: IsaacAssetGeometry | None = None,
        rng_seed: int = 123,
        scatter_gain: float = 0.03,
        detector_model: dict[str, object] | None = None,
        shield_thickness: ShieldThicknessConfig | None = None,
    ) -> None:
        """Create the shared Python transport simulator used in mock mode."""
        geometry = asset_geometry or IsaacAssetGeometry()
        decomposer = SpectralDecomposer()
        self.scene = SceneDescription()
        shield_thickness = shield_thickness or resolve_shield_thickness_config()
        self._mu_by_isotope = mu_by_isotope_from_tvl_mm(
            HVL_TVL_TABLE_MM,
            isotopes=list(decomposer.isotope_names),
        )
        detector_outer_radius_cm = detector_outer_radius_cm_from_model(detector_model)
        fe_inner_cm, pb_inner_cm = nested_shield_inner_radii_cm(
            thickness_fe_cm=float(shield_thickness.thickness_fe_cm),
            detector_outer_radius_cm=detector_outer_radius_cm,
        )
        self.transport_model = PythonTransportSpectrumModel(
            sources=(),
            decomposer=decomposer,
            mu_by_isotope=self._mu_by_isotope,
            shield_params=ShieldParams(
                thickness_fe_cm=float(shield_thickness.thickness_fe_cm),
                thickness_pb_cm=float(shield_thickness.thickness_pb_cm),
                inner_radius_fe_cm=fe_inner_cm,
                inner_radius_pb_cm=pb_inner_cm,
            ),
            obstacle_height_m=geometry.obstacle_height_m,
            scatter_gain=scatter_gain,
            rng_seed=rng_seed,
            detector_model=detector_model,
        )

    def reset(self, scene: SceneDescription) -> None:
        """Store the active scene for the mock backend."""
        self.scene = scene
        self.transport_model.reset_scene(
            sources=scene.to_point_sources(),
            obstacle_grid=_obstacle_grid_from_scene(scene),
            obstacle_material=scene.obstacle_material,
        )

    def observe(self, command: SimulationCommand) -> SimulationObservation:
        """Return an analytic observation anchored to the commanded detector pose."""
        detector_pose = tuple(float(v) for v in command.target_pose_xyz)
        return self.transport_model.observe(
            command,
            detector_pose_xyz=detector_pose,
            detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            backend_label="isaacsim-mock",
        )


class IsaacSimObservationModel(ObservationModel):
    """Generate spectra using stage geometry for obstacle and shield attenuation."""

    def __init__(
        self,
        robot_controller: RobotController,
        *,
        usd_path: str | None,
        asset_geometry: IsaacAssetGeometry,
        stage_material_rules: tuple[StageMaterialRule, ...] = (),
        rng_seed: int = 123,
        scatter_gain: float = 0.03,
        detector_model: dict[str, object] | None = None,
        shield_thickness: ShieldThicknessConfig | None = None,
    ) -> None:
        """Store stage-backed handles and spectrum simulation helpers."""
        self.robot_controller = robot_controller
        self.usd_path = usd_path
        self.asset_geometry = asset_geometry
        self.stage_material_rules = tuple(stage_material_rules)
        self.rng_seed = int(rng_seed)
        self.scatter_gain = float(scatter_gain)
        self.decomposer = SpectralDecomposer()
        self.scene = SceneDescription()
        self.shield_thickness = shield_thickness or resolve_shield_thickness_config()
        self._mu_by_isotope = mu_by_isotope_from_tvl_mm(
            HVL_TVL_TABLE_MM,
            isotopes=list(self.decomposer.isotope_names),
        )
        detector_outer_radius_cm = detector_outer_radius_cm_from_model(detector_model)
        fe_inner_cm, pb_inner_cm = nested_shield_inner_radii_cm(
            thickness_fe_cm=float(self.shield_thickness.thickness_fe_cm),
            detector_outer_radius_cm=detector_outer_radius_cm,
        )
        self.transport_model = PythonTransportSpectrumModel(
            sources=(),
            decomposer=self.decomposer,
            mu_by_isotope=self._mu_by_isotope,
            shield_params=ShieldParams(
                thickness_fe_cm=float(self.shield_thickness.thickness_fe_cm),
                thickness_pb_cm=float(self.shield_thickness.thickness_pb_cm),
                inner_radius_fe_cm=fe_inner_cm,
                inner_radius_pb_cm=pb_inner_cm,
            ),
            obstacle_height_m=self.asset_geometry.obstacle_height_m,
            scatter_gain=self.scatter_gain,
            rng_seed=self.rng_seed,
            detector_model=detector_model,
        )

    def reset(self, scene: SceneDescription) -> None:
        """Store the loaded scene description for subsequent observations."""
        self.scene = scene
        self.transport_model.reset_scene(
            sources=scene.to_point_sources(),
            obstacle_grid=_obstacle_grid_from_scene(scene),
            obstacle_material=scene.obstacle_material,
        )

    def observe(self, command: SimulationCommand) -> SimulationObservation:
        """Return a spectrum attenuated by stage-authored obstacles and shields."""
        detector_pose = self.robot_controller.detector_world_pose()
        metadata: dict[str, object] = {}
        if self.usd_path:
            metadata["usd_path"] = self.usd_path
        return self.transport_model.observe(
            command,
            detector_pose_xyz=detector_pose.translation_xyz,
            detector_quat_wxyz=detector_pose.orientation_wxyz,
            backend_label="isaacsim",
            stage_segments_provider=self._stage_geometry_segments_for_source,
            shield_path_provider=self._shield_path_lengths_for_command,
            extra_metadata=metadata,
        )

    def _stage_geometry_segments_for_source(
        self,
        source: object,
        detector_xyz: tuple[float, float, float],
    ) -> tuple[TransportSegment, ...]:
        """Return static stage material crossings for one point source."""
        source_position = tuple(float(value) for value in getattr(source, "position"))
        return self._stage_geometry_segments(source_position, detector_xyz)

    def _shield_path_lengths_for_command(
        self,
        source: object,
        detector_xyz: tuple[float, float, float],
        command: SimulationCommand,
    ) -> tuple[float, float]:
        """Return shield path lengths from the current stage-authored shield poses."""
        del detector_xyz, command
        source_position = tuple(float(value) for value in getattr(source, "position"))
        return self._shield_path_lengths_cm(source_position)

    def _stage_geometry_segments(
        self,
        source_xyz: tuple[float, float, float],
        detector_xyz: tuple[float, float, float],
    ) -> tuple[TransportSegment, ...]:
        """Return shared transport segments for crossed static materials."""
        segments: list[TransportSegment] = []
        prefixes = tuple(
            sorted(
                {
                    *{rule.path_prefix for rule in self.stage_material_rules},
                    self.robot_controller.prim_paths.obstacles_root,
                    "/World/Environment",
                }
            )
        )
        for solid_prim in self.robot_controller.stage_backend.list_solid_prims(path_prefixes=prefixes):
            material_info = self._material_for_prim(solid_prim.path, solid_prim.material_info)
            if material_info is None:
                continue
            if solid_prim.path in {
                self.robot_controller.prim_paths.fe_shield_path,
                self.robot_controller.prim_paths.pb_shield_path,
                self.robot_controller.prim_paths.detector_path,
            }:
                continue
            if solid_prim.path.startswith(self.robot_controller.prim_paths.sources_root):
                continue
            path_length_cm = 100.0 * self._solid_path_length_m(source_xyz, detector_xyz, solid_prim)
            if path_length_cm <= 0.0:
                continue
            segments.append(
                make_transport_segment(
                    material_info,
                    float(path_length_cm),
                    is_obstacle=material_info.name.lower() == self.scene.obstacle_material.lower(),
                )
            )
        return tuple(segments)

    def _shield_path_lengths_cm(
        self,
        source_xyz: tuple[float, float, float],
    ) -> tuple[float, float]:
        """Return path lengths through the Fe and Pb spherical octant shells."""
        fe_pose = self.robot_controller.stage_backend.get_world_pose(
            self.robot_controller.prim_paths.fe_shield_path
        )
        pb_pose = self.robot_controller.stage_backend.get_world_pose(
            self.robot_controller.prim_paths.pb_shield_path
        )
        detector_pose = self.robot_controller.detector_world_pose()
        fe_length_cm = spherical_octant_path_length_cm(
            source_xyz,
            detector_pose.translation_xyz,
            fe_pose.orientation_wxyz,
            thickness_cm=float(self.shield_thickness.thickness_fe_cm),
            inner_radius_cm=self.transport_model.shield_params.inner_radius_fe_cm,
        )
        pb_length_cm = spherical_octant_path_length_cm(
            source_xyz,
            detector_pose.translation_xyz,
            pb_pose.orientation_wxyz,
            thickness_cm=float(self.shield_thickness.thickness_pb_cm),
            inner_radius_cm=self.transport_model.shield_params.inner_radius_pb_cm,
        )
        return float(fe_length_cm), float(pb_length_cm)

    def _material_for_prim(
        self,
        path: str,
        prim_material: StageMaterialInfo | None,
    ) -> StageMaterialInfo | None:
        """Resolve a material for a prim using authored metadata first."""
        if prim_material is not None:
            return prim_material
        matched_material: str | None = None
        matched_len = -1
        for rule in self.stage_material_rules:
            if path.startswith(rule.path_prefix) and len(rule.path_prefix) > matched_len:
                matched_material = rule.material
                matched_len = len(rule.path_prefix)
        if matched_material is None and path.startswith(self.robot_controller.prim_paths.obstacles_root):
            return StageMaterialInfo(name=self.scene.obstacle_material)
        if matched_material is None and path.startswith("/World/Environment"):
            return StageMaterialInfo(name="concrete")
        if matched_material is None:
            return None
        return StageMaterialInfo(name=matched_material)

    def _solid_path_length_m(
        self,
        source_xyz: tuple[float, float, float],
        detector_xyz: tuple[float, float, float],
        solid_prim: object,
    ) -> float:
        """Return the path length through a supported solid prim."""
        shape = getattr(solid_prim, "shape")
        pose = getattr(solid_prim, "pose")
        if shape == "box":
            size_xyz = getattr(solid_prim, "size_xyz")
            if size_xyz is None:
                return 0.0
            box = OrientedBox(
                center_xyz=pose.translation_xyz,
                size_xyz=size_xyz,
                rotation_matrix=quaternion_wxyz_to_matrix(pose.orientation_wxyz),
            )
            return float(segment_path_length_through_box(source_xyz, detector_xyz, box))
        if shape == "sphere":
            radius_m = getattr(solid_prim, "radius_m")
            if radius_m is None:
                return 0.0
            sphere = Sphere(center_xyz=pose.translation_xyz, radius_m=float(radius_m))
            return float(segment_path_length_through_sphere(source_xyz, detector_xyz, sphere))
        if shape == "mesh":
            triangles_xyz = getattr(solid_prim, "triangles_xyz")
            if not triangles_xyz:
                return 0.0
            mesh = TriangleMesh(triangles_xyz=triangles_xyz)
            return float(segment_path_length_through_mesh(source_xyz, detector_xyz, mesh))
        return 0.0
