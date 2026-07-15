"""Geant4 sidecar application entry points."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sim.geant4_app.engine import Geant4EngineConfig, Geant4StepRequest, build_geant4_engine
from sim.geant4_app.scene_export import (
    DEFAULT_DETECTOR_CRYSTAL_LENGTH_M,
    DEFAULT_DETECTOR_CRYSTAL_RADIUS_M,
    DEFAULT_DETECTOR_HOUSING_THICKNESS_M,
    ExportedDetectorModel,
    export_scene_for_geant4,
)
from sim.isaacsim_app.app import IsaacAssetGeometry, StageMaterialRule
from sim.isaacsim_app.robot_controller import RobotController
from sim.isaacsim_app.scene_builder import SceneBuilder, SceneDescription
from sim.isaacsim_app.stage_backend import FakeStageBackend, IsaacSimStageBackend, StageBackend
from sim.protocol import SimulationCommand, SimulationObservation
from sim.radiation_visualization import RadiationVisualizationConfig
from sim.shield_geometry import ShieldThicknessConfig, resolve_shield_thickness_config
from spectrum.pipeline import SpectralDecomposer


@dataclass(frozen=True)
class Geant4AppConfig:
    """Collect sidecar configuration relevant to the Geant4 app."""

    use_mock_stage: bool = True
    headless: bool = True
    renderer: str = "RayTracedLighting"
    usd_path: str | None = None
    detector_height_m: float = 0.5
    robot_ground_z_m: float = 0.0
    obstacle_height_m: float = 2.0
    author_obstacle_prims: bool | None = None
    author_room_boundary_prims: bool | None = None
    fe_shield_size_xyz: tuple[float, float, float] = (0.25, 0.08, 0.25)
    pb_shield_size_xyz: tuple[float, float, float] = (0.25, 0.08, 0.25)
    stage_material_rules: tuple[StageMaterialRule, ...] = field(default_factory=tuple)
    engine_mode: str = "external"
    physics_profile: str = "balanced"
    thread_count: int = 1
    random_seed_base: int = 123
    dead_time_tau_s: float = 5.813e-9
    scatter_gain: float = 0.0
    executable_path: str | None = "build/geant4_sidecar"
    executable_args: tuple[str, ...] = field(default_factory=tuple)
    timeout_s: float = 120.0
    persistent_process: bool = False
    source_rate_model: str = "detector_cps_1m"
    source_bias_mode: str = "detector_cone"
    source_bias_cone_half_angle_deg: float = 0.0
    source_bias_isotropic_fraction: float = 0.1
    detector_scoring_mode: str = "full_transport"
    secondary_transport_mode: str = "full_transport"
    primary_sampling_fraction: float = 1.0
    detector_model: ExportedDetectorModel = field(default_factory=ExportedDetectorModel)
    shield_thickness: ShieldThicknessConfig = field(default_factory=resolve_shield_thickness_config)
    absorbing_transport_groups: tuple[str, ...] = field(default_factory=tuple)
    absorbing_path_prefixes: tuple[str, ...] = field(default_factory=tuple)
    radiation_visualization: RadiationVisualizationConfig = field(default_factory=RadiationVisualizationConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Geant4AppConfig":
        """Normalize a JSON config payload into a strongly typed object."""
        payload = {} if data is None else dict(data)
        stage_material_rules_payload = payload.get("stage_material_rules", ())
        if not isinstance(stage_material_rules_payload, (list, tuple)):
            raise ValueError("stage_material_rules must be a list of objects.")
        detector_payload = payload.get("detector_model", {})
        if not isinstance(detector_payload, dict):
            raise ValueError("detector_model must be a JSON object.")
        visualization_payload = payload.get("radiation_visualization", {})
        if not isinstance(visualization_payload, dict):
            raise ValueError("radiation_visualization must be a JSON object.")
        executable_args = payload.get("executable_args", ())
        if not isinstance(executable_args, (list, tuple)):
            raise ValueError("executable_args must be a list of strings.")
        absorbing_transport_groups = payload.get("absorbing_transport_groups", ())
        if not isinstance(absorbing_transport_groups, (list, tuple)):
            raise ValueError("absorbing_transport_groups must be a list of strings.")
        absorbing_path_prefixes = payload.get("absorbing_path_prefixes", ())
        if not isinstance(absorbing_path_prefixes, (list, tuple)):
            raise ValueError("absorbing_path_prefixes must be a list of strings.")
        return cls(
            use_mock_stage=bool(payload.get("use_mock_stage", True)),
            headless=bool(payload.get("headless", True)),
            renderer=str(payload.get("renderer", "RayTracedLighting")),
            usd_path=None if payload.get("usd_path") in (None, "") else str(payload["usd_path"]),
            detector_height_m=float(payload.get("detector_height_m", 0.5)),
            robot_ground_z_m=float(payload.get("robot_ground_z_m", 0.0)),
            obstacle_height_m=float(payload.get("obstacle_height_m", 2.0)),
            author_obstacle_prims=(
                None
                if payload.get("author_obstacle_prims") is None
                else bool(payload.get("author_obstacle_prims"))
            ),
            author_room_boundary_prims=(
                None
                if payload.get("author_room_boundary_prims") is None
                else bool(payload.get("author_room_boundary_prims"))
            ),
            fe_shield_size_xyz=tuple(float(v) for v in payload.get("fe_shield_size_xyz", (0.25, 0.08, 0.25))),
            pb_shield_size_xyz=tuple(float(v) for v in payload.get("pb_shield_size_xyz", (0.25, 0.08, 0.25))),
            stage_material_rules=tuple(
                StageMaterialRule(
                    path_prefix=str(entry["path_prefix"]),
                    material=str(entry["material"]),
                )
                for entry in stage_material_rules_payload
            ),
            engine_mode=str(payload.get("engine_mode", "external")),
            physics_profile=str(payload.get("physics_profile", "balanced")),
            thread_count=int(payload.get("thread_count", 1)),
            random_seed_base=int(payload.get("random_seed_base", 123)),
            dead_time_tau_s=float(payload.get("dead_time_tau_s", 5.813e-9)),
            scatter_gain=float(payload.get("scatter_gain", 0.0)),
            executable_path=(
                "build/geant4_sidecar"
                if payload.get("executable_path") in (None, "")
                else str(payload.get("executable_path"))
            ),
            executable_args=tuple(str(v) for v in executable_args),
            timeout_s=float(payload.get("timeout_s", 120.0)),
            persistent_process=bool(payload.get("persistent_process", False)),
            source_rate_model=str(payload.get("source_rate_model", "detector_cps_1m")),
            source_bias_mode=str(payload.get("source_bias_mode", "detector_cone")),
            source_bias_cone_half_angle_deg=float(
                payload.get("source_bias_cone_half_angle_deg", 0.0)
            ),
            source_bias_isotropic_fraction=float(
                payload.get("source_bias_isotropic_fraction", 0.1)
            ),
            detector_scoring_mode=str(payload.get("detector_scoring_mode", "full_transport")),
            secondary_transport_mode=str(payload.get("secondary_transport_mode", "full_transport")),
            primary_sampling_fraction=float(payload.get("primary_sampling_fraction", 1.0)),
            detector_model=ExportedDetectorModel(
                crystal_radius_m=float(
                    detector_payload.get("crystal_radius_m", DEFAULT_DETECTOR_CRYSTAL_RADIUS_M)
                ),
                crystal_length_m=float(
                    detector_payload.get("crystal_length_m", DEFAULT_DETECTOR_CRYSTAL_LENGTH_M)
                ),
                housing_thickness_m=float(
                    detector_payload.get(
                        "housing_thickness_m",
                        DEFAULT_DETECTOR_HOUSING_THICKNESS_M,
                    )
                ),
                crystal_shape=str(detector_payload.get("crystal_shape", "sphere")),
                crystal_material=str(detector_payload.get("crystal_material", "cebr3")),
                housing_material=str(detector_payload.get("housing_material", "aluminum")),
            ),
            shield_thickness=resolve_shield_thickness_config(payload),
            absorbing_transport_groups=tuple(str(v) for v in absorbing_transport_groups),
            absorbing_path_prefixes=tuple(str(v) for v in absorbing_path_prefixes),
            radiation_visualization=RadiationVisualizationConfig.from_dict(visualization_payload),
        )


class Geant4Application:
    """Wrap Geant4 sidecar scene handling and spectrum generation."""

    def __init__(
        self,
        *,
        app_config: dict[str, Any] | None = None,
        stage_backend: StageBackend | None = None,
    ) -> None:
        """Create the application and initialize the requested stage backend."""
        self.config = Geant4AppConfig.from_dict(app_config)
        self.scene = SceneDescription()
        self.asset_geometry = IsaacAssetGeometry(
            detector_height_m=self.config.detector_height_m,
            obstacle_height_m=self.config.obstacle_height_m,
            fe_shield_size_xyz=self.config.fe_shield_size_xyz,
            pb_shield_size_xyz=self.config.pb_shield_size_xyz,
        )
        backend = stage_backend
        if backend is None:
            if self.config.use_mock_stage:
                backend = FakeStageBackend()
            else:
                try:
                    backend = IsaacSimStageBackend(
                        headless=self.config.headless,
                        renderer=self.config.renderer,
                    )
                except ModuleNotFoundError as exc:
                    raise RuntimeError(
                        "Geant4 use_mock_stage=false requires Isaac Sim Python modules. "
                        "Run the bridge with Isaac Sim's python.sh or set "
                        "ISAACSIM_PYTHON=/path/to/isaacsim/python.sh for auto-start."
                    ) from exc
        self._stage_backend = backend
        self.scene_builder = SceneBuilder(
            backend,
            detector_height_m=self.config.detector_height_m,
            obstacle_height_m=self.config.obstacle_height_m,
            fe_shield_size_xyz=self.config.fe_shield_size_xyz,
            pb_shield_size_xyz=self.config.pb_shield_size_xyz,
        )
        self.robot_controller = RobotController(
            backend,
            self.scene.prim_paths,
            detector_height_m=self.config.detector_height_m,
            fe_offset_xyz=(0.0, 0.0, self.config.detector_height_m),
            pb_offset_xyz=(0.0, 0.0, self.config.detector_height_m),
            ground_z_m=self.config.robot_ground_z_m,
        )
        self.engine = build_geant4_engine(
            Geant4EngineConfig(
                physics_profile=self.config.physics_profile,
                thread_count=self.config.thread_count,
                random_seed_base=self.config.random_seed_base,
                dead_time_tau_s=self.config.dead_time_tau_s,
                scatter_gain=self.config.scatter_gain,
                executable_path=self.config.executable_path,
                executable_args=self.config.executable_args,
                timeout_s=self.config.timeout_s,
                persistent_process=self.config.persistent_process,
                source_rate_model=self.config.source_rate_model,
                source_bias_mode=self.config.source_bias_mode,
                source_bias_cone_half_angle_deg=self.config.source_bias_cone_half_angle_deg,
                source_bias_isotropic_fraction=self.config.source_bias_isotropic_fraction,
                detector_scoring_mode=self.config.detector_scoring_mode,
                secondary_transport_mode=self.config.secondary_transport_mode,
                primary_sampling_fraction=self.config.primary_sampling_fraction,
                radiation_visualization=self.config.radiation_visualization,
            ),
            engine_mode=self.config.engine_mode,
        )
        self._last_cache_hit = False
        self._decomposer = SpectralDecomposer()

    def reset(self, scene: SceneDescription) -> None:
        """Load a new scene description and rebuild or reuse the Geant4 world."""
        if (
            scene.usd_path is None
            and scene.use_config_usd_fallback
            and self.config.usd_path is not None
        ):
            scene.usd_path = self.config.usd_path
        if self.config.author_obstacle_prims is not None:
            scene.author_obstacle_prims = self.config.author_obstacle_prims
        if self.config.author_room_boundary_prims is not None:
            scene.author_room_boundary_prims = self.config.author_room_boundary_prims
        self.scene = scene
        self.scene_builder.load_scene(scene, usd_path_override=None)
        self.robot_controller = RobotController(
            self._stage_backend,
            scene.prim_paths,
            detector_height_m=self.config.detector_height_m,
            fe_offset_xyz=(0.0, 0.0, self.config.detector_height_m),
            pb_offset_xyz=(0.0, 0.0, self.config.detector_height_m),
            ground_z_m=self.config.robot_ground_z_m,
        )
        self.robot_controller.reset()
        exported_scene = export_scene_for_geant4(
            scene,
            stage_backend=self._stage_backend,
            asset_geometry=self.asset_geometry,
            detector_model=self.config.detector_model,
            shield_thickness=self.config.shield_thickness,
            stage_material_rules=self.config.stage_material_rules,
            absorbing_transport_groups=self.config.absorbing_transport_groups,
            absorbing_path_prefixes=self.config.absorbing_path_prefixes,
        )
        self._last_cache_hit = bool(self.engine.load_scene(exported_scene))

    def step(self, command: SimulationCommand) -> SimulationObservation:
        """Apply a command and return the resulting Geant4-backed observation."""
        self.robot_controller.apply_command(command)
        detector_pose = self.robot_controller.detector_world_pose()
        fe_pose = self._stage_backend.get_world_pose(self.scene.prim_paths.fe_shield_path)
        pb_pose = self._stage_backend.get_world_pose(self.scene.prim_paths.pb_shield_path)
        spectrum, metadata = self.engine.simulate(
            Geant4StepRequest(
                step_id=command.step_id,
                dwell_time_s=float(command.dwell_time_s),
                seed=int(self.config.random_seed_base + int(command.step_id)),
                detector_pose_xyz=detector_pose.translation_xyz,
                detector_quat_wxyz=detector_pose.orientation_wxyz,
                fe_shield_pose_xyz=fe_pose.translation_xyz,
                fe_shield_quat_wxyz=fe_pose.orientation_wxyz,
                pb_shield_pose_xyz=pb_pose.translation_xyz,
                pb_shield_quat_wxyz=pb_pose.orientation_wxyz,
            )
        )
        metadata = dict(metadata)
        metadata.setdefault("cache_hit", self._last_cache_hit)
        metadata.setdefault("fe_orientation_index", int(command.fe_orientation_index))
        metadata.setdefault("pb_orientation_index", int(command.pb_orientation_index))
        metadata.setdefault("shield_num_orientations", 8)
        metadata.setdefault(
            "shield_pair_id",
            int(command.fe_orientation_index) * 8 + int(command.pb_orientation_index),
        )
        metadata.setdefault(
            "shield_thickness_scale",
            float(self.config.shield_thickness.thickness_scale),
        )
        metadata.setdefault(
            "shield_thickness_fe_cm",
            float(self.config.shield_thickness.thickness_fe_cm),
        )
        metadata.setdefault(
            "shield_thickness_pb_cm",
            float(self.config.shield_thickness.thickness_pb_cm),
        )
        energy = self._decomposer.energy_axis
        bin_width_keV = float(self._decomposer.config.bin_width_keV)
        edges = list(energy) + [float(energy[-1] + bin_width_keV)]
        return SimulationObservation(
            step_id=command.step_id,
            detector_pose_xyz=detector_pose.translation_xyz,
            detector_quat_wxyz=detector_pose.orientation_wxyz,
            fe_orientation_index=command.fe_orientation_index,
            pb_orientation_index=command.pb_orientation_index,
            spectrum_counts=np.asarray(spectrum, dtype=float).tolist(),
            energy_bin_edges_keV=[float(v) for v in edges],
            metadata=metadata,
        )

    def close(self) -> None:
        """Close the underlying engine and stage backend."""
        self.engine.close()
        self._stage_backend.close()
