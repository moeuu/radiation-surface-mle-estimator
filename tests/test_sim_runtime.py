"""Tests for simulator protocol and runtime integration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from measurement.model import PointSource
from measurement.obstacles import ObstacleGrid
from pf.estimator import MeasurementRecord, RotatingShieldPFEstimator
from realtime_demo import _has_environment_obstacles, run_live_pf
from sim.geant4_app.app import Geant4Application
from sim.geant4_app.bridge_server import Geant4BridgeServerConfig, serve_forever as serve_geant4_forever
from sim.geant4_app.io_format import read_response_file
from sim.geant4_app.scene_export import ExportedDetectorModel, export_scene_for_geant4
from sim.isaacsim_app.app import IsaacSimAppConfig, IsaacSimApplication
from sim.isaacsim_app.geometry import (
    OrientedBox,
    Sphere,
    TriangleMesh,
    segment_path_length_through_box,
    segment_path_length_through_mesh,
    segment_path_length_through_sphere,
)
from sim.isaacsim_app.materials import composition_mass_attenuation_at_energy
from sim.isaacsim_app.bridge_server import BridgeServerConfig, serve_forever
from sim.isaacsim_app.scene_builder import SceneBuilder, SceneDescription, SourceDescription, StagePrimPaths
from sim.isaacsim_app.stage_backend import (
    FakeStageBackend,
    apply_camera_gesture_bindings_to_module,
    merge_camera_gesture_bindings,
)
from sim.protocol import (
    SimulationCommand,
    SimulationObservation,
    decode_message,
    encode_message,
)
from sim.runtime import (
    AnalyticSimulationRuntime,
    Geant4WithIsaacSimRuntime,
    Geant4TCPClientRuntime,
    IsaacSimTCPClientRuntime,
    ManagedIsaacSimTCPClientRuntime,
    ManagedGeant4TCPClientRuntime,
    create_simulation_runtime,
)
from sim.radiation_visualization import (
    RadiationVisualizationConfig,
    build_visualization_metadata_from_scene,
)
from sim.shield_geometry import (
    FE_SHIELD_INNER_RADIUS_M,
    FE_SHIELD_THICKNESS_CM,
    FE_SHIELD_TVL_THICKNESS_CM,
    PB_SHIELD_INNER_RADIUS_M,
    PB_SHIELD_TVL_THICKNESS_CM,
    SHIELD_SHAPE_SPHERICAL_OCTANT,
    resolve_shield_thickness_config,
    shield_thickness_scale_for_transmission,
)
from spectrum.library import ANALYSIS_ISOTOPES
from spectrum.pipeline import SpectralDecomposer


def _free_port() -> int:
    """Return an available localhost TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_fake_external_geant4(path: Path) -> Path:
    """Write a fake external Geant4 executable for protocol-level tests."""
    path.write_text(
        "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                "RESPONSE=''",
                "while [[ $# -gt 0 ]]; do",
                "  case \"$1\" in",
                "    --response) RESPONSE=\"$2\"; shift 2 ;;",
                "    *) shift ;;",
                "  esac",
                "done",
                "printf 'META backend=geant4\\nMETA engine_mode=external\\nMETA num_primaries=42\\nSPECTRUM 1.0,2.0,3.0\\n' > \"$RESPONSE\"",
            ]
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _write_fake_persistent_geant4(path: Path) -> Path:
    """Write a fake persistent Geant4 executable for engine lifecycle tests."""
    path.write_text(
        "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                "PERSISTENT=0",
                "for arg in \"$@\"; do",
                "  if [[ \"$arg\" == \"--persistent\" ]]; then PERSISTENT=1; fi",
                "done",
                "if [[ \"$PERSISTENT\" != \"1\" ]]; then",
                "  echo 'expected --persistent' >&2",
                "  exit 2",
                "fi",
                "RUN_INDEX=0",
                "while IFS= read -r LINE; do",
                "  if [[ \"$LINE\" == \"SHUTDOWN\"* ]]; then",
                "    echo 'SIMBRIDGE_OK shutdown'",
                "    exit 0",
                "  fi",
                "  RESPONSE=''",
                "  for TOKEN in $LINE; do",
                "    case \"$TOKEN\" in",
                "      response=*) RESPONSE=\"${TOKEN#response=}\" ;;",
                "    esac",
                "  done",
                "  RESPONSE=\"${RESPONSE//%20/ }\"",
                "  RUN_INDEX=$((RUN_INDEX + 1))",
                "  printf 'META backend=geant4\\nMETA engine_mode=external\\nMETA persistent_process=true\\nMETA run_index=%s\\nMETA num_primaries=42\\nSPECTRUM 1.0,2.0,3.0\\n' \"$RUN_INDEX\" > \"$RESPONSE\"",
                "  echo \"SIMBRIDGE_OK response=$RESPONSE\"",
                "done",
            ]
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def test_geant4_response_reader_parses_spectrum_variance(tmp_path: Path) -> None:
    """Response files should preserve weighted spectrum variance for PF likelihoods."""
    response_path = tmp_path / "response.txt"
    response_path.write_text(
        "\n".join(
            [
                "META backend=geant4",
                "META weighted_transport=true",
                "SPECTRUM 1.5,0.0,2.0",
                "SPECTRUM_VARIANCE 0.25,0.0,4.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    spectrum, metadata = read_response_file(response_path)

    assert spectrum.tolist() == pytest.approx([1.5, 0.0, 2.0])
    assert metadata["weighted_transport"] is True
    assert metadata["spectrum_count_variance"] == pytest.approx([0.25, 0.0, 4.0])
    assert metadata["spectrum_count_variance_total"] == pytest.approx(4.25)


def test_protocol_round_trip() -> None:
    """Protocol envelopes and dataclasses should round-trip cleanly."""
    command = SimulationCommand(
        step_id=3,
        target_pose_xyz=(1.0, 2.0, 0.5),
        target_base_yaw_rad=0.25,
        fe_orientation_index=4,
        pb_orientation_index=7,
        dwell_time_s=30.0,
        travel_time_s=2.5,
        shield_actuation_time_s=0.75,
    )
    encoded = encode_message("step", command.to_dict())
    msg_type, payload = decode_message(encoded.strip())
    restored = SimulationCommand.from_dict(payload)
    assert msg_type == "step"
    assert restored == command

    observation = SimulationObservation(
        step_id=3,
        detector_pose_xyz=(1.0, 2.0, 0.5),
        detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        fe_orientation_index=4,
        pb_orientation_index=7,
        spectrum_counts=[1.0, 2.0, 3.0],
        energy_bin_edges_keV=[0.0, 10.0, 20.0, 30.0],
        metadata={"backend": "mock"},
    )
    assert SimulationObservation.from_dict(observation.to_dict()) == observation


def test_mock_bridge_server_round_trip() -> None:
    """The TCP sidecar should accept reset and step commands in mock mode."""
    port = _free_port()
    thread = threading.Thread(
        target=serve_forever,
        args=(BridgeServerConfig(host="127.0.0.1", port=port, use_mock=True),),
        daemon=True,
    )
    thread.start()
    time.sleep(0.2)

    runtime = IsaacSimTCPClientRuntime(host="127.0.0.1", port=port, timeout_s=5.0)
    runtime.reset({"source_count": 0})
    observation = runtime.step(
        SimulationCommand(
            step_id=0,
            target_pose_xyz=(1.0, 1.0, 0.5),
            target_base_yaw_rad=0.0,
            fe_orientation_index=0,
            pb_orientation_index=0,
            dwell_time_s=1.0,
        )
    )
    runtime.close()
    thread.join(timeout=5.0)

    assert observation.step_id == 0
    assert observation.detector_pose_xyz == (1.0, 1.0, 0.5)
    assert observation.metadata["backend"] == "isaacsim-mock"
    assert len(observation.spectrum_counts) > 0
    assert not thread.is_alive()


def test_analytic_runtime_uses_python_transport_obstacle_attenuation() -> None:
    """Python CUI mode should attenuate spectra through reset-payload obstacles."""
    source = PointSource(
        isotope="Cs-137",
        position=(0.5, 3.5, 0.5),
        intensity_cps_1m=5.0e6,
    )
    command = SimulationCommand(
        step_id=7,
        target_pose_xyz=(0.5, 0.5, 0.5),
        target_base_yaw_rad=0.0,
        fe_orientation_index=0,
        pb_orientation_index=0,
        dwell_time_s=20.0,
    )
    common_payload = {
        "sources": [
            {
                "isotope": source.isotope,
                "position": list(source.position),
                "intensity_cps_1m": source.intensity_cps_1m,
            }
        ],
        "obstacle_origin_xy": [0.0, 0.0],
        "obstacle_cell_size_m": 1.0,
        "obstacle_grid_shape": [2, 5],
        "obstacle_material": "concrete",
    }
    clear_runtime = AnalyticSimulationRuntime(
        sources=[source],
        decomposer=SpectralDecomposer(),
        mu_by_isotope={},
        shield_params=None,
        rng_seed=99,
    )
    clear_runtime.reset({**common_payload, "obstacle_cells": []})
    clear_observation = clear_runtime.step(command)

    blocked_runtime = AnalyticSimulationRuntime(
        sources=[source],
        decomposer=SpectralDecomposer(),
        mu_by_isotope={},
        shield_params=None,
        rng_seed=99,
    )
    blocked_runtime.reset({**common_payload, "obstacle_cells": [[0, 1], [0, 2]]})
    blocked_observation = blocked_runtime.step(command)

    assert blocked_observation.metadata["backend"] == "analytic"
    assert blocked_observation.metadata["transport_backend"] == "python"
    assert float(blocked_observation.metadata["total_obstacle_path_cm"]) > 0.0
    assert sum(blocked_observation.spectrum_counts) < sum(clear_observation.spectrum_counts)


def test_analytic_runtime_uses_python_transport_spherical_shield() -> None:
    """Python CUI mode should apply Fe/Pb spherical-octant shield paths."""
    source = PointSource(
        isotope="Cs-137",
        position=(0.0, 0.0, 0.0),
        intensity_cps_1m=5.0e6,
    )
    clear_command = SimulationCommand(
        step_id=9,
        target_pose_xyz=(1.0, 1.0, 1.0),
        target_base_yaw_rad=0.0,
        fe_orientation_index=7,
        pb_orientation_index=7,
        dwell_time_s=20.0,
    )
    blocked_command = SimulationCommand(
        step_id=9,
        target_pose_xyz=(1.0, 1.0, 1.0),
        target_base_yaw_rad=0.0,
        fe_orientation_index=0,
        pb_orientation_index=0,
        dwell_time_s=20.0,
    )
    runtime = AnalyticSimulationRuntime(
        sources=[source],
        decomposer=SpectralDecomposer(),
        mu_by_isotope={},
        shield_params=None,
        rng_seed=101,
    )

    clear_observation = runtime.step(clear_command)
    blocked_observation = runtime.step(blocked_command)

    assert float(blocked_observation.metadata["total_fe_path_cm"]) > 0.0
    assert float(blocked_observation.metadata["total_pb_path_cm"]) > 0.0
    assert sum(blocked_observation.spectrum_counts) < sum(clear_observation.spectrum_counts)


def test_mock_gui_uses_shared_python_transport_obstacles() -> None:
    """Python GUI mock mode should use the same obstacle attenuation path as CUI."""
    scene_kwargs = dict(
        room_size_xyz=(3.0, 5.0, 3.0),
        obstacle_origin_xy=(0.0, 0.0),
        obstacle_cell_size_m=1.0,
        obstacle_grid_shape=(2, 5),
        sources=[
            SourceDescription(
                isotope="Cs-137",
                position_xyz=(0.5, 3.5, 0.5),
                intensity_cps_1m=5.0e6,
            )
        ],
    )
    command = SimulationCommand(
        step_id=11,
        target_pose_xyz=(0.5, 0.5, 0.5),
        target_base_yaw_rad=0.0,
        fe_orientation_index=0,
        pb_orientation_index=0,
        dwell_time_s=20.0,
    )
    clear_app = IsaacSimApplication(
        use_mock=True,
        app_config={"detector_height_m": 0.5, "obstacle_height_m": 2.0},
    )
    clear_app.reset(SceneDescription(obstacle_cells=[], **scene_kwargs))
    clear_observation = clear_app.step(command)
    clear_app.close()

    blocked_app = IsaacSimApplication(
        use_mock=True,
        app_config={"detector_height_m": 0.5, "obstacle_height_m": 2.0},
    )
    blocked_app.reset(SceneDescription(obstacle_cells=[(0, 1), (0, 2)], **scene_kwargs))
    blocked_observation = blocked_app.step(command)
    blocked_app.close()

    assert blocked_observation.metadata["backend"] == "isaacsim-mock"
    assert blocked_observation.metadata["transport_backend"] == "python"
    assert float(blocked_observation.metadata["total_obstacle_path_cm"]) > 0.0
    assert sum(blocked_observation.spectrum_counts) < sum(clear_observation.spectrum_counts)


def test_isaacsim_config_parses_camera_gesture_bindings() -> None:
    """Isaac Sim GUI config should preserve optional camera gesture bindings."""
    config = IsaacSimAppConfig.from_dict(
        {"camera_gesture_bindings": {"TumbleGesture": "LeftButton"}}
    )

    assert dict(config.camera_gesture_bindings) == {"TumbleGesture": "LeftButton"}


def test_isaacsim_config_parses_view_helpers() -> None:
    """Isaac Sim GUI config should preserve initial camera and lighting helpers."""
    config = IsaacSimAppConfig.from_dict(
        {
            "author_obstacle_prims": False,
            "initial_camera": {
                "eye_xyz": [2.8, -2.6, 1.6],
                "target_xyz": [1.0, 1.0, 0.65],
                "focal_length_mm": 22.0,
            },
            "lighting": {
                "dome_intensity": 150.0,
                "interior_light_position_xyz": [2.5, 2.5, 3.0],
                "interior_light_intensity": 4500.0,
                "interior_light_radius_m": 3.0,
                "interior_lights": [
                    {
                        "position_xyz": [0.0, 0.0, 3.0],
                        "intensity": 5000.0,
                        "radius_m": 0.05,
                    }
                ],
            },
            "preserve_viewport_on_reset": True,
            "stage_visual_rules": [
                {
                    "path_prefix": "/World/Environment",
                    "color_rgb": [0.42, 0.46, 0.5],
                    "opacity": 1.0,
                    "roughness": 0.85,
                    "emissive_scale": 0.25,
                }
            ],
        }
    )

    assert config.author_obstacle_prims is False
    assert config.initial_camera is not None
    assert config.initial_camera.eye_xyz == pytest.approx((2.8, -2.6, 1.6))
    assert config.initial_camera.target_xyz == pytest.approx((1.0, 1.0, 0.65))
    assert config.initial_camera.focal_length_mm == pytest.approx(22.0)
    assert config.lighting is not None
    assert config.lighting.dome_intensity == pytest.approx(150.0)
    assert config.lighting.interior_light_position_xyz == pytest.approx((2.5, 2.5, 3.0))
    assert len(config.lighting.interior_lights) == 1
    assert config.lighting.interior_lights[0].position_xyz == pytest.approx((0.0, 0.0, 3.0))
    assert config.lighting.interior_lights[0].intensity == pytest.approx(5000.0)
    assert config.lighting.interior_lights[0].radius_m == pytest.approx(0.05)
    assert config.preserve_viewport_on_reset is True
    assert len(config.stage_visual_rules) == 1
    assert config.stage_visual_rules[0].path_prefix == "/World/Environment"
    assert config.stage_visual_rules[0].color_rgb == pytest.approx((0.42, 0.46, 0.5))
    assert config.stage_visual_rules[0].emissive_scale == pytest.approx(0.25)


def test_merge_camera_gesture_bindings_preserves_defaults() -> None:
    """Camera gesture overrides should not drop unspecified Isaac Sim defaults."""
    defaults = {
        "PanGesture": "Any MiddleButton",
        "TumbleGesture": "Alt LeftButton",
        "ZoomGesture": "Alt RightButton",
        "LookGesture": "RightButton",
    }

    merged = merge_camera_gesture_bindings(defaults, {"TumbleGesture": "LeftButton"})

    assert merged["TumbleGesture"] == "LeftButton"
    assert merged["PanGesture"] == "Any MiddleButton"
    assert merged["ZoomGesture"] == "Alt RightButton"
    assert merged["LookGesture"] == "RightButton"


def test_apply_camera_gesture_bindings_to_module_updates_defaults() -> None:
    """Gesture module patching should update the live Isaac Sim default dict."""
    module = ModuleType("fake_gestures")
    module.kDefaultKeyBindings = {
        "PanGesture": "Any MiddleButton",
        "TumbleGesture": "Alt LeftButton",
    }

    apply_camera_gesture_bindings_to_module(module, {"TumbleGesture": "LeftButton"})

    assert module.kDefaultKeyBindings == {
        "PanGesture": "Any MiddleButton",
        "TumbleGesture": "LeftButton",
    }


def test_geant4_bridge_server_round_trip(tmp_path: Path) -> None:
    """The Geant4 sidecar should accept reset and step commands."""
    port = _free_port()
    executable = _write_fake_external_geant4(tmp_path / "fake_geant4.py")
    thread = threading.Thread(
        target=serve_geant4_forever,
        args=(
            Geant4BridgeServerConfig(
                host="127.0.0.1",
                port=port,
                app_config={
                    "use_mock_stage": True,
                    "engine_mode": "external",
                    "executable_path": executable.as_posix(),
                },
            ),
        ),
        daemon=True,
    )
    thread.start()
    time.sleep(0.2)

    runtime = Geant4TCPClientRuntime(host="127.0.0.1", port=port, timeout_s=10.0)
    runtime.reset({"source_count": 0})
    observation = runtime.step(
        SimulationCommand(
            step_id=0,
            target_pose_xyz=(1.0, 1.0, 0.5),
            target_base_yaw_rad=0.0,
            fe_orientation_index=0,
            pb_orientation_index=0,
            dwell_time_s=1.0,
        )
    )
    runtime.close()
    thread.join(timeout=5.0)

    assert observation.step_id == 0
    assert observation.detector_pose_xyz == pytest.approx((1.0, 1.0, 0.5))
    assert observation.metadata["backend"] == "geant4"
    assert "scene_hash" in observation.metadata
    assert len(observation.spectrum_counts) > 0
    assert not thread.is_alive()


def test_geant4_real_stage_does_not_fall_back_to_fake_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """use_mock_stage=false should fail loudly when Isaac Sim modules are unavailable."""
    import sim.geant4_app.app as geant4_app_module

    class _MissingIsaacBackend:
        """Stage backend stub that simulates missing Isaac Sim modules."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Raise the same import error as the real backend outside Isaac Python."""
            raise ModuleNotFoundError("isaacsim")

    monkeypatch.setattr(geant4_app_module, "IsaacSimStageBackend", _MissingIsaacBackend)

    with pytest.raises(RuntimeError, match="requires Isaac Sim Python modules"):
        geant4_app_module.Geant4Application(app_config={"use_mock_stage": False})


def test_real_application_loads_scene_into_stage_backend() -> None:
    """Real mode should load a USD scene and publish detector poses from the stage."""
    backend = FakeStageBackend()
    app = IsaacSimApplication(
        use_mock=False,
        app_config={"usd_path": "demo_room.usda", "detector_height_m": 0.5},
        stage_backend=backend,
    )
    scene = SceneDescription(
        room_size_xyz=(10.0, 20.0, 3.0),
        obstacle_origin_xy=(0.0, 0.0),
        obstacle_cell_size_m=1.0,
        obstacle_grid_shape=(10, 20),
        obstacle_cells=[(1, 2)],
        sources=[
            SourceDescription(
                isotope="Cs-137",
                position_xyz=(5.0, 6.0, 1.0),
                intensity_cps_1m=1000.0,
            )
        ],
        usd_path="demo_room.usda",
    )
    app.reset(scene)
    observation = app.step(
        SimulationCommand(
            step_id=1,
            target_pose_xyz=(2.0, 3.0, 0.5),
            target_base_yaw_rad=0.3,
            fe_orientation_index=1,
            pb_orientation_index=2,
            dwell_time_s=1.0,
        )
    )
    app.close()

    assert backend.opened_usd_path == "demo_room.usda"
    assert "/World/SimBridge/Obstacles/Obstacle_0000" in backend.prims
    assert "/World/SimBridge/Sources/Cs_137_00" in backend.prims
    assert backend.prims["/World/SimBridge/Sources/Cs_137_00"].scale_xyz == pytest.approx((0.16, 0.16, 0.16))
    assert backend.prims["/World/SimBridge/Sources/Cs_137_00"].metadata["color_rgb"] == pytest.approx(
        (1.0, 0.05, 0.05)
    )
    assert "/World/SimBridge/Robot/Body" in backend.prims
    assert "/World/SimBridge/Robot/WheelFrontLeft" in backend.prims
    assert observation.metadata["backend"] == "isaacsim"
    assert observation.metadata["usd_path"] == "demo_room.usda"
    assert observation.detector_pose_xyz == pytest.approx((2.0, 3.0, 0.5))
    assert len(observation.spectrum_counts) > 0
    assert float(observation.metadata["total_fe_path_cm"]) >= 0.0
    assert float(observation.metadata["total_pb_path_cm"]) >= 0.0


def test_real_application_authors_view_helpers() -> None:
    """Real mode should author configured camera and lighting helper prims."""
    backend = FakeStageBackend()
    app = IsaacSimApplication(
        use_mock=False,
        app_config={
            "usd_path": "demo_room.usda",
            "author_obstacle_prims": False,
            "initial_camera": {
                "eye_xyz": [2.8, -2.6, 1.6],
                "target_xyz": [1.0, 1.0, 0.65],
                "focal_length_mm": 22.0,
            },
            "lighting": {
                "dome_intensity": 150.0,
                "interior_light_position_xyz": [2.5, 2.5, 3.0],
                "interior_light_intensity": 4500.0,
                "interior_light_radius_m": 3.0,
                "interior_lights": [
                    {
                        "position_xyz": [0.0, 0.0, 3.0],
                        "intensity": 5000.0,
                        "radius_m": 0.05,
                    }
                ],
            },
            "stage_visual_rules": [
                {
                    "path_prefix": "/World/Environment",
                    "color_rgb": [0.42, 0.46, 0.5],
                    "opacity": 1.0,
                    "roughness": 0.85,
                    "emissive_scale": 0.25,
                }
            ],
        },
        stage_backend=backend,
    )
    scene = SceneDescription(
        obstacle_cells=[(1, 2)],
        usd_path="demo_room.usda",
    )

    app.reset(scene)
    app.close()

    assert "/World/SimBridge/Obstacles/Obstacle_0000" not in backend.prims
    assert backend.prims["/World/SimBridge/View/DomeLight"].prim_type == "DomeLight"
    assert backend.prims["/World/SimBridge/View/InteriorLight"].prim_type == "SphereLight"
    assert backend.prims["/World/SimBridge/View/InteriorLight_00"].prim_type == "SphereLight"
    camera = backend.prims["/World/SimBridge/View/InitialCamera"]
    assert camera.prim_type == "Camera"
    assert camera.pose.translation_xyz == pytest.approx((2.8, -2.6, 1.6))
    assert camera.metadata["target_xyz"] == pytest.approx((1.0, 1.0, 0.65))
    assert backend.prims["/World/Environment/Floor"].metadata["visual_color_rgb"] == pytest.approx(
        (0.42, 0.46, 0.5)
    )
    assert backend.prims["/World/Environment/Floor"].metadata["visual_emissive_scale"] == pytest.approx(0.25)


def test_real_application_preserves_viewport_helpers_on_reused_stage() -> None:
    """Preserving the viewport should not reopen the same USD or reset the camera."""
    backend = FakeStageBackend()
    app = IsaacSimApplication(
        use_mock=False,
        app_config={
            "usd_path": "demo_room.usda",
            "author_obstacle_prims": False,
            "preserve_viewport_on_reset": True,
            "initial_camera": {
                "eye_xyz": [7.8, -2.8, 2.7],
                "target_xyz": [5.2, 8.8, 0.75],
                "focal_length_mm": 20.0,
            },
        },
        stage_backend=backend,
    )
    scene = SceneDescription(
        sources=[
            SourceDescription(
                isotope="Cs-137",
                position_xyz=(5.0, 5.0, 1.0),
                intensity_cps_1m=1000.0,
            )
        ],
        usd_path="demo_room.usda",
    )

    app.reset(scene)
    camera = backend.prims["/World/SimBridge/View/InitialCamera"]
    camera.metadata["user_view_marker"] = "kept"
    app.reset(scene)
    app.close()

    assert backend.open_stage_calls == ["demo_room.usda"]
    assert backend.prims["/World/SimBridge/View/InitialCamera"].metadata["user_view_marker"] == "kept"
    assert "/World/SimBridge/Robot/Body" in backend.prims
    assert "/World/SimBridge/Sources/Cs_137_00" in backend.prims


def test_isaacsim_application_visualizes_radiation_metadata() -> None:
    """Real mode should author sampled radiation tracks and hits from metadata."""
    backend = FakeStageBackend()
    app = IsaacSimApplication(
        use_mock=False,
        app_config={"usd_path": "demo_room.usda"},
        stage_backend=backend,
    )
    app.reset(SceneDescription(usd_path="demo_room.usda"))
    app.visualize_observation(
        SimulationObservation(
            step_id=1,
            detector_pose_xyz=(2.0, 3.0, 1.0),
            detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            fe_orientation_index=0,
            pb_orientation_index=0,
            spectrum_counts=[10.0, 5.0],
            energy_bin_edges_keV=[0.0, 1.0, 2.0],
            metadata={
                "radiation_tracks": [
                    {
                        "points_xyz": [[5.0, 5.0, 5.0], [2.0, 3.0, 1.0]],
                        "isotope": "Cs-137",
                        "detected": True,
                    }
                ],
                "radiation_hits": [
                    {
                        "position_xyz": [2.0, 3.0, 1.0],
                        "isotope": "Cs-137",
                    }
                ],
            },
        )
    )
    app.close()

    assert "/World/SimBridge/Radiation/Tracks/Track_0000" in backend.prims
    assert "/World/SimBridge/Radiation/Hits/Hit_0000" in backend.prims
    assert "/World/SimBridge/Radiation/DetectorPulse" in backend.prims
    track = backend.prims["/World/SimBridge/Radiation/Tracks/Track_0000"]
    assert track.metadata["points_xyz"][0] == pytest.approx((5.0, 5.0, 5.0))
    assert track.metadata["color_rgb"] == pytest.approx((1.0, 0.95, 0.0))
    hit = backend.prims["/World/SimBridge/Radiation/Hits/Hit_0000"]
    assert hit.metadata["color_rgb"] == pytest.approx((1.0, 0.95, 0.0))


def test_isaacsim_application_visualizes_pf_particles_and_estimates() -> None:
    """Real mode should author bounded PF particle and estimate markers."""
    backend = FakeStageBackend()
    app = IsaacSimApplication(
        use_mock=False,
        app_config={
            "usd_path": "demo_room.usda",
            "pf_visual_max_particles_per_isotope": 2,
            "pf_visual_particle_radius_m": 0.02,
        },
        stage_backend=backend,
    )
    app.reset(SceneDescription(usd_path="demo_room.usda"))
    app.visualize_pf_state(
        {
            "step_index": 1,
            "time_s": 2.0,
            "particle_positions": {
                "Cs-137": [
                    [1.0, 2.0, 3.0],
                    [2.0, 3.0, 4.0],
                    [3.0, 4.0, 5.0],
                ]
            },
            "particle_weights": {"Cs-137": [0.1, 0.9, 0.2]},
            "estimated_sources": {"Cs-137": [[4.0, 5.0, 6.0]]},
            "estimated_strengths": {"Cs-137": [30000.0]},
        }
    )
    app.close()

    particle_prefix = "/World/SimBridge/PFVisualization/Particles/Cs_137/"
    particle_paths = [
        path for path in backend.prims if path.startswith(particle_prefix)
    ]
    assert len(particle_paths) == 2
    assert (
        "/World/SimBridge/PFVisualization/Estimates/Cs_137/Estimate_00/Center"
        in backend.prims
    )
    assert (
        "/World/SimBridge/PFVisualization/Estimates/Cs_137/Estimate_00/Cross_X"
        in backend.prims
    )


def test_isaacsim_application_replays_radiation_tracks() -> None:
    """Real mode should animate sampled tracks when playback metadata is present."""

    class _RecordingStageBackend(FakeStageBackend):
        """Fake backend that records live radiation particle authoring."""

        def __init__(self) -> None:
            """Initialize recorded live particle paths."""
            super().__init__()
            self.live_particle_paths: list[str] = []

        def ensure_sphere(
            self,
            path: str,
            *,
            radius_m: float,
            translation_xyz: tuple[float, float, float],
            color_rgb: tuple[float, float, float] | None = None,
            material: str | None = None,
        ) -> None:
            """Record live particle spheres while preserving fake behavior."""
            super().ensure_sphere(
                path,
                radius_m=radius_m,
                translation_xyz=translation_xyz,
                color_rgb=color_rgb,
                material=material,
            )
            if "/World/SimBridge/Radiation/Live/Particles/" in path:
                self.live_particle_paths.append(path)

    backend = _RecordingStageBackend()
    app = IsaacSimApplication(
        use_mock=False,
        app_config={"usd_path": "demo_room.usda"},
        stage_backend=backend,
    )
    app.reset(SceneDescription(usd_path="demo_room.usda"))
    app.visualize_observation(
        SimulationObservation(
            step_id=1,
            detector_pose_xyz=(2.0, 3.0, 1.0),
            detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            fe_orientation_index=0,
            pb_orientation_index=0,
            spectrum_counts=[1.0],
            energy_bin_edges_keV=[0.0, 1.0],
            metadata={
                "radiation_tracks": [
                    {
                        "points_xyz": [[5.0, 5.0, 5.0], [2.0, 3.0, 1.0]],
                        "isotope": "Cs-137",
                        "detected": True,
                        "emission_time_s": 0.0,
                        "flight_time_s": 0.05,
                        "persistence_s": 0.05,
                    }
                ],
                "radiation_visualization": {
                    "playback_enabled": True,
                    "playback_duration_s": 0.1,
                    "playback_time_scale": 0.0,
                    "playback_fps": 2.0,
                    "max_live_tracks": 4,
                },
            },
        )
    )
    app.close()

    assert backend.live_particle_paths
    assert "/World/SimBridge/Radiation/Tracks/Track_0000" in backend.prims
    assert not any(path.startswith("/World/SimBridge/Radiation/Live") for path in backend.prims)


def test_radiation_visualization_samples_isotropic_non_detected_tracks() -> None:
    """Non-detected representative tracks should radiate away from the source."""
    scene = SceneDescription(
        sources=[
            SourceDescription(
                isotope="Cs-137",
                position_xyz=(0.0, 0.0, 0.0),
                intensity_cps_1m=1.0e6,
            )
        ],
    )
    metadata = build_visualization_metadata_from_scene(
        scene,
        SimpleNamespace(detector_pose_xyz=(1.0, 0.0, 0.0), dwell_time_s=30.0),
        seed=42,
        config=RadiationVisualizationConfig(
            max_tracks=24,
            max_hits=0,
            source_jitter_m=0.0,
            playback_enabled=False,
        ),
        library=SpectralDecomposer().library,
        mode="test",
    )
    tracks = metadata["radiation_tracks"]
    endpoints = np.asarray([track["points_xyz"][-1] for track in tracks], dtype=float)
    lateral = np.linalg.norm(endpoints[:, 1:], axis=1)

    assert tracks
    assert all(not bool(track["detected"]) for track in tracks)
    assert max(lateral) > 0.5
    assert {track["visual_kind"] for track in tracks} <= {"primary", "scatter"}


def test_radiation_visualization_anchors_scatter_inside_obstacle() -> None:
    """Obstacle crossings should draw attenuated tracks ending inside the obstacle."""
    concrete = SimpleNamespace(
        name="concrete",
        density_g_cm3=2.3,
        mu_by_isotope={},
        mass_att_by_isotope_cm2_g={},
        preset_name=None,
        composition_by_mass={"O": 0.525, "Si": 0.325, "Ca": 0.090, "Al": 0.060},
    )
    scene = SimpleNamespace(
        sources=[
            SimpleNamespace(
                isotope="Cs-137",
                position_xyz=(5.0, 5.0, 1.0),
                intensity_cps_1m=3.0e4,
            )
        ],
        static_volumes=[
            SimpleNamespace(
                path="/World/SimBridge/Obstacles/Column",
                shape="box",
                translation_xyz=(3.5, 3.5, 1.0),
                orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
                size_xyz=(1.0, 1.0, 2.0),
                material=concrete,
            )
        ],
        prim_paths=StagePrimPaths(),
    )
    metadata = build_visualization_metadata_from_scene(
        scene,
        SimpleNamespace(detector_pose_xyz=(1.0, 1.0, 0.5), dwell_time_s=30.0),
        seed=7,
        config=RadiationVisualizationConfig(
            max_tracks=64,
            max_hits=16,
            source_jitter_m=0.0,
            playback_enabled=False,
        ),
        library=SpectralDecomposer().library,
        mode="test",
    )
    tracks = metadata["radiation_tracks"]
    attenuated_tracks = [track for track in tracks if track["visual_kind"] == "attenuated"]
    primary_tracks = [track for track in tracks if track["visual_kind"] == "primary"]

    assert tracks
    assert attenuated_tracks
    assert len(primary_tracks) > len(attenuated_tracks)
    assert all(not bool(track["detected"]) for track in tracks)
    assert all("scatter_anchor_xyz" in track for track in attenuated_tracks)
    attenuated_endpoints = np.asarray([track["points_xyz"][-1] for track in attenuated_tracks], dtype=float)
    assert float(np.min(np.linalg.norm(attenuated_endpoints - np.asarray((3.5, 3.5, 0.8125)), axis=1))) < 0.3
    assert max(track["attenuated_fraction"] for track in attenuated_tracks) > 0.99
    source = np.asarray((5.0, 5.0, 1.0), dtype=float)
    detector_direction = np.asarray((1.0, 1.0, 0.5), dtype=float) - source
    detector_direction /= np.linalg.norm(detector_direction)
    end_vectors = np.asarray([track["points_xyz"][-1] for track in tracks], dtype=float) - source
    end_vectors /= np.linalg.norm(end_vectors, axis=1)[:, None]
    direction_dots = end_vectors @ detector_direction
    assert float(np.min(direction_dots)) < -0.5
    assert float(np.max(direction_dots)) > 0.5
    assert abs(float(np.mean(direction_dots))) < 0.25


def test_radiation_visualization_can_limit_environment_obstacle_paths() -> None:
    """Configured obstacle prefixes should keep wall volumes out of visual scatter."""
    concrete = SimpleNamespace(
        name="concrete",
        density_g_cm3=2.3,
        mu_by_isotope={},
        mass_att_by_isotope_cm2_g={},
        preset_name=None,
        composition_by_mass={"O": 0.525, "Si": 0.325, "Ca": 0.090, "Al": 0.060},
    )
    slab_center = (3.15, 3.15, 1.5)
    slab_size = (0.4, 1.8, 1.8)
    scene = SimpleNamespace(
        sources=[
            SimpleNamespace(
                isotope="Cs-137",
                position_xyz=(5.0, 5.0, 2.4),
                intensity_cps_1m=3.0e4,
            )
        ],
        static_volumes=[
            SimpleNamespace(
                path="/World/Environment/Floor",
                shape="box",
                translation_xyz=(3.0, 3.0, -0.05),
                orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
                size_xyz=(12.0, 12.0, 0.1),
                material=concrete,
            ),
            SimpleNamespace(
                path="/World/Environment/ScatterAttenuationSlab",
                shape="box",
                translation_xyz=slab_center,
                orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
                size_xyz=slab_size,
                material=concrete,
            ),
        ],
        prim_paths=StagePrimPaths(),
    )

    metadata = build_visualization_metadata_from_scene(
        scene,
        SimpleNamespace(detector_pose_xyz=(1.0, 1.0, 0.5), dwell_time_s=30.0),
        seed=12,
        config=RadiationVisualizationConfig(
            max_tracks=120,
            max_hits=0,
            source_jitter_m=0.0,
            playback_enabled=False,
            scatter_visual_gain_scale=20.0,
            obstacle_path_prefixes=("/World/Environment/ScatterAttenuationSlab",),
            obstacle_focus_fraction=1.0,
        ),
        library=SpectralDecomposer().library,
        mode="test",
        scatter_gain=0.35,
    )

    tracks = metadata["radiation_tracks"]
    anchored_tracks = [track for track in tracks if "scatter_anchor_xyz" in track]
    attenuated_tracks = [track for track in tracks if track["visual_kind"] == "attenuated"]
    scatter_tracks = [track for track in tracks if track["visual_kind"] == "scatter"]
    anchors = np.asarray([track["scatter_anchor_xyz"] for track in anchored_tracks], dtype=float)
    lower = np.asarray(slab_center, dtype=float) - 0.5 * np.asarray(slab_size, dtype=float)
    upper = np.asarray(slab_center, dtype=float) + 0.5 * np.asarray(slab_size, dtype=float)

    assert anchored_tracks
    assert attenuated_tracks
    assert scatter_tracks
    assert np.all(anchors >= lower - 1.0e-6)
    assert np.all(anchors <= upper + 1.0e-6)


def test_robot_controller_interpolates_travel_between_measurement_points() -> None:
    """Real-mode robot commands should author intermediate poses before measuring."""

    class _RecordingStageBackend(FakeStageBackend):
        """Fake stage backend that records robot root translations."""

        def __init__(self) -> None:
            """Initialize recorded robot root poses."""
            super().__init__()
            self.robot_poses: list[tuple[float, float, float]] = []

        def set_local_pose(
            self,
            path: str,
            *,
            translation_xyz: tuple[float, float, float] | None = None,
            orientation_wxyz: tuple[float, float, float, float] | None = None,
            scale_xyz: tuple[float, float, float] | None = None,
        ) -> None:
            """Record robot root translation updates while preserving fake behavior."""
            super().set_local_pose(
                path,
                translation_xyz=translation_xyz,
                orientation_wxyz=orientation_wxyz,
                scale_xyz=scale_xyz,
            )
            if path == "/World/SimBridge/Robot" and translation_xyz is not None:
                self.robot_poses.append(tuple(float(v) for v in translation_xyz))

    backend = _RecordingStageBackend()
    app = IsaacSimApplication(
        use_mock=False,
        app_config={
            "usd_path": "demo_room.usda",
            "robot_motion_speed_m_s": 1.0,
            "robot_animation_dt_s": 0.25,
            "robot_animation_time_scale": 0.0,
        },
        stage_backend=backend,
    )
    app.reset(SceneDescription(usd_path="demo_room.usda"))
    assert backend.robot_poses[-1] == pytest.approx((1.0, 1.0, 0.0))
    app.step(
        SimulationCommand(
            step_id=0,
            target_pose_xyz=(3.0, 1.0, 0.5),
            target_base_yaw_rad=0.0,
            fe_orientation_index=0,
            pb_orientation_index=0,
            dwell_time_s=1.0,
            travel_time_s=2.0,
        )
    )
    app.close()

    assert len(backend.robot_poses) >= 2
    assert backend.robot_poses[-1] == pytest.approx((3.0, 1.0, 0.0))
    assert backend.robot_poses[0][0] < backend.robot_poses[-1][0]


def test_scene_builder_can_skip_python_obstacle_authoring() -> None:
    """Blender-authored environments should be able to own obstacle geometry."""
    backend = FakeStageBackend()
    builder = SceneBuilder(backend)
    scene = SceneDescription(
        obstacle_origin_xy=(0.0, 0.0),
        obstacle_cell_size_m=1.0,
        obstacle_grid_shape=(2, 2),
        obstacle_cells=[(0, 0)],
        author_obstacle_prims=False,
    )

    builder.load_scene(scene)

    assert "/World/SimBridge/Obstacles/Obstacle_0000" not in backend.prims


def test_scene_builder_can_author_room_boundaries_as_wall_group() -> None:
    """CUI scenes should be able to author grouped room boundary geometry."""
    backend = FakeStageBackend()
    builder = SceneBuilder(backend)
    scene = SceneDescription(
        room_size_xyz=(4.0, 5.0, 3.0),
        author_obstacle_prims=False,
        author_room_boundary_prims=True,
    )

    builder.load_scene(scene)

    assert "/World/Environment/Wall/NorthWall" in backend.prims
    wall = backend.prims["/World/Environment/Wall/NorthWall"]
    assert wall.metadata["transport_group"] == "wall"
    assert wall.metadata["material"] == "concrete"
    assert "/World/Environment/Wall/Ceiling" in backend.prims
    ceiling = backend.prims["/World/Environment/Wall/Ceiling"]
    assert ceiling.metadata["transport_group"] == "wall"
    assert ceiling.metadata["material"] == "concrete"


def test_fake_visual_material_zero_opacity_marks_prim_hidden() -> None:
    """Zero-opacity visual overrides should mark matching fake prims hidden."""
    backend = FakeStageBackend()
    backend.ensure_box(
        "/World/Environment/Wall/NorthWall",
        size_xyz=(1.0, 0.1, 1.0),
        translation_xyz=(0.0, 0.0, 0.0),
    )

    backend.apply_visual_material(
        "/World/Environment/Wall",
        color_rgb=(0.8, 0.8, 0.8),
        opacity=0.0,
    )

    metadata = backend.prims["/World/Environment/Wall/NorthWall"].metadata
    assert metadata["visual_opacity"] == 0.0
    assert metadata["visual_visible"] is False


def test_application_prefers_scene_usd_over_config_default() -> None:
    """A generated scene USD in the reset payload should override config defaults."""
    backend = FakeStageBackend()
    app = IsaacSimApplication(
        use_mock=False,
        app_config={"usd_path": "demo_room.usda"},
        stage_backend=backend,
    )
    scene = SceneDescription(usd_path="generated_random.usda")

    app.reset(scene)
    app.close()

    assert backend.opened_usd_path == "generated_random.usda"


def test_geant4_empty_scene_suppresses_config_demo_room() -> None:
    """An explicit no-fallback scene should prevent no-obstacle CUI runs from loading demo obstacles."""
    backend = FakeStageBackend()
    app = Geant4Application(
        app_config={"usd_path": "demo_room.usda", "use_mock_stage": True},
        stage_backend=backend,
    )
    scene = SceneDescription(
        usd_path=None,
        use_config_usd_fallback=False,
        room_size_xyz=(10.0, 20.0, 3.0),
        author_obstacle_prims=True,
        author_room_boundary_prims=True,
    )

    app.reset(scene)
    app.close()

    assert backend.opened_usd_path is None
    assert "/World/Environment/PillarMesh" not in backend.prims
    assert "/World/Environment/Wall/NorthWall" in backend.prims


def test_geant4_scene_export_is_stable() -> None:
    """Exporting the same stage twice should produce the same scene hash."""
    backend = FakeStageBackend()
    app = Geant4Application(
        app_config={"usd_path": "demo_room.usda", "use_mock_stage": True},
        stage_backend=backend,
    )
    scene = SceneDescription(
        room_size_xyz=(10.0, 20.0, 3.0),
        obstacle_origin_xy=(0.0, 0.0),
        obstacle_cell_size_m=1.0,
        obstacle_grid_shape=(10, 20),
        obstacle_cells=[(1, 2)],
        sources=[
            SourceDescription(
                isotope="Cs-137",
                position_xyz=(5.0, 6.0, 1.0),
                intensity_cps_1m=1000.0,
            )
        ],
        usd_path="demo_room.usda",
    )
    app.reset(scene)
    export_one = export_scene_for_geant4(
        scene,
        stage_backend=backend,
        asset_geometry=app.asset_geometry,
        detector_model=ExportedDetectorModel(),
        stage_material_rules=app.config.stage_material_rules,
    )
    export_two = export_scene_for_geant4(
        scene,
        stage_backend=backend,
        asset_geometry=app.asset_geometry,
        detector_model=ExportedDetectorModel(),
        stage_material_rules=app.config.stage_material_rules,
    )
    app.close()

    assert export_one.scene_hash == export_two.scene_hash
    assert export_one.to_dict() == export_two.to_dict()
    assert export_one.fe_shield.shape == SHIELD_SHAPE_SPHERICAL_OCTANT
    assert export_one.fe_shield.inner_radius_m == pytest.approx(FE_SHIELD_INNER_RADIUS_M)
    assert export_one.pb_shield.inner_radius_m == pytest.approx(PB_SHIELD_INNER_RADIUS_M)
    assert export_one.fe_shield.thickness_cm == pytest.approx(FE_SHIELD_THICKNESS_CM)


def test_shield_transmission_target_scales_geant4_export() -> None:
    """Shared shield transmission targets should scale exported Geant4 geometry."""
    backend = FakeStageBackend()
    app = Geant4Application(
        app_config={
            "usd_path": "demo_room.usda",
            "use_mock_stage": True,
            "shield_transmission_target": 0.5,
        },
        stage_backend=backend,
    )
    scene = SceneDescription(usd_path="demo_room.usda")
    shield_thickness = resolve_shield_thickness_config({"shield_transmission_target": 0.5})

    app.reset(scene)
    exported = export_scene_for_geant4(
        scene,
        stage_backend=backend,
        asset_geometry=app.asset_geometry,
        detector_model=ExportedDetectorModel(),
        shield_thickness=app.config.shield_thickness,
        stage_material_rules=app.config.stage_material_rules,
    )
    app.close()

    assert shield_thickness.thickness_scale == pytest.approx(
        shield_thickness_scale_for_transmission(0.5)
    )
    assert exported.fe_shield.thickness_cm == pytest.approx(
        FE_SHIELD_TVL_THICKNESS_CM * shield_thickness.thickness_scale
    )
    assert exported.pb_shield.thickness_cm == pytest.approx(
        PB_SHIELD_TVL_THICKNESS_CM * shield_thickness.thickness_scale
    )
    assert exported.fe_shield.inner_radius_m == pytest.approx(FE_SHIELD_INNER_RADIUS_M)
    assert exported.pb_shield.inner_radius_m == pytest.approx(
        FE_SHIELD_INNER_RADIUS_M + exported.fe_shield.thickness_cm / 100.0
    )


def test_geant4_scene_export_groups_walls_without_name_list() -> None:
    """Wall-like environment prims should carry one semantic transport group."""
    backend = FakeStageBackend()
    app = Geant4Application(
        app_config={"usd_path": "demo_room.usda", "use_mock_stage": True},
        stage_backend=backend,
    )
    scene = SceneDescription(usd_path="demo_room.usda")

    app.reset(scene)
    exported = export_scene_for_geant4(
        scene,
        stage_backend=backend,
        asset_geometry=app.asset_geometry,
        detector_model=ExportedDetectorModel(),
        stage_material_rules=app.config.stage_material_rules,
    )
    app.close()

    groups_by_path = {volume.path: volume.transport_group for volume in exported.static_volumes}
    assert groups_by_path["/World/Environment/NorthWall"] == "wall"
    assert groups_by_path["/World/Environment/SouthWall"] == "wall"
    assert groups_by_path["/World/Environment/EastWall"] == "wall"
    assert groups_by_path["/World/Environment/WestWall"] == "wall"
    assert groups_by_path["/World/Environment/Floor"] == "wall"
    assert groups_by_path["/World/Environment/PillarMesh"] is None


def test_geant4_scene_export_can_mark_wall_group_as_absorber(tmp_path: Path) -> None:
    """Explicit absorber config should only change wall-group transport mode."""
    from sim.geant4_app.io_format import write_scene_file

    backend = FakeStageBackend()
    app = Geant4Application(
        app_config={
            "usd_path": "demo_room.usda",
            "use_mock_stage": True,
            "absorbing_transport_groups": ["wall"],
        },
        stage_backend=backend,
    )
    scene = SceneDescription(
        usd_path="demo_room.usda",
        obstacle_cells=[(0, 0)],
        obstacle_grid_shape=(1, 1),
    )

    app.reset(scene)
    exported = export_scene_for_geant4(
        scene,
        stage_backend=backend,
        asset_geometry=app.asset_geometry,
        detector_model=ExportedDetectorModel(),
        stage_material_rules=app.config.stage_material_rules,
        absorbing_transport_groups=app.config.absorbing_transport_groups,
    )
    app.close()

    modes_by_path = {volume.path: volume.transport_mode for volume in exported.static_volumes}
    assert modes_by_path["/World/Environment/NorthWall"] == "absorber"
    assert modes_by_path["/World/Environment/PillarMesh"] == "geant4"
    assert modes_by_path["/World/SimBridge/Obstacles/Obstacle_0000"] == "geant4"

    scene_path = tmp_path / "scene.txt"
    write_scene_file(exported, scene_path)
    scene_text = scene_path.read_text(encoding="utf-8")
    assert "path=/World/Environment/NorthWall" in scene_text
    assert "transport_group=wall transport_mode=absorber" in scene_text


def test_stage_backend_exports_cuboid_mesh_as_box_for_geant4() -> None:
    """Axis-aligned cuboid meshes should avoid tessellated Geant4 solids."""
    backend = FakeStageBackend()
    backend.open_stage("demo_room.usda")

    solids = {
        solid.path: solid
        for solid in backend.list_solid_prims(path_prefixes=("/World/Environment/PillarMesh",))
    }

    pillar = solids["/World/Environment/PillarMesh"]
    assert pillar.shape == "box"
    assert pillar.size_xyz == pytest.approx((0.4, 0.4, 2.0))
    assert pillar.pose.translation_xyz == pytest.approx((7.0, 11.0, 1.0))
    assert pillar.triangles_xyz is None


def test_geant4_application_reuses_geometry_cache_on_same_scene(tmp_path: Path) -> None:
    """Resetting the same scene twice should report a geometry cache hit."""
    backend = FakeStageBackend()
    executable = _write_fake_external_geant4(tmp_path / "fake_geant4.py")
    app = Geant4Application(
        app_config={
            "usd_path": "demo_room.usda",
            "use_mock_stage": True,
            "engine_mode": "external",
            "executable_path": executable.as_posix(),
        },
        stage_backend=backend,
    )
    scene = SceneDescription(
        room_size_xyz=(10.0, 20.0, 3.0),
        obstacle_origin_xy=(0.0, 0.0),
        obstacle_cell_size_m=1.0,
        obstacle_grid_shape=(10, 20),
        obstacle_cells=[(1, 2)],
        sources=[
            SourceDescription(
                isotope="Cs-137",
                position_xyz=(5.0, 6.0, 1.0),
                intensity_cps_1m=1000.0,
            )
        ],
        usd_path="demo_room.usda",
    )
    command = SimulationCommand(
        step_id=2,
        target_pose_xyz=(2.0, 3.0, 0.5),
        target_base_yaw_rad=0.3,
        fe_orientation_index=1,
        pb_orientation_index=2,
        dwell_time_s=1.0,
    )

    app.reset(scene)
    first_observation = app.step(command)
    app.reset(scene)
    second_observation = app.step(command)
    app.close()

    assert first_observation.metadata["cache_hit"] is False
    assert second_observation.metadata["cache_hit"] is True
    assert second_observation.metadata["backend"] == "geant4"
    assert int(second_observation.metadata["num_primaries"]) >= 0
    assert len(second_observation.metadata["radiation_tracks"]) > 0
    assert second_observation.metadata["radiation_visualization"]["sampled_track_count"] > 0
    visualization = second_observation.metadata["radiation_visualization"]
    assert visualization["playback_duration_s"] == pytest.approx(1.0)
    assert "emission_time_s" in second_observation.metadata["radiation_tracks"][0]


def test_geant4_application_can_keep_native_process_persistent(tmp_path: Path) -> None:
    """Persistent native mode should reuse one executable across step requests."""
    backend = FakeStageBackend()
    executable = _write_fake_persistent_geant4(tmp_path / "fake_persistent_geant4.sh")
    app = Geant4Application(
        app_config={
            "usd_path": "demo_room.usda",
            "use_mock_stage": True,
            "engine_mode": "external",
            "persistent_process": True,
            "executable_path": executable.as_posix(),
        },
        stage_backend=backend,
    )
    scene = SceneDescription(usd_path="demo_room.usda")
    command = SimulationCommand(
        step_id=2,
        target_pose_xyz=(2.0, 3.0, 0.5),
        target_base_yaw_rad=0.3,
        fe_orientation_index=1,
        pb_orientation_index=2,
        dwell_time_s=1.0,
    )

    app.reset(scene)
    first_observation = app.step(command)
    second_observation = app.step(
        SimulationCommand(
            step_id=3,
            target_pose_xyz=(2.0, 3.0, 0.5),
            target_base_yaw_rad=0.3,
            fe_orientation_index=1,
            pb_orientation_index=2,
            dwell_time_s=1.0,
        )
    )
    app.close()

    assert first_observation.metadata["persistent_process"] is True
    assert second_observation.metadata["persistent_process"] is True
    assert int(first_observation.metadata["run_index"]) == 1
    assert int(second_observation.metadata["run_index"]) == 2


def test_create_simulation_runtime_supports_geant4() -> None:
    """The runtime factory should accept the Geant4 backend."""
    runtime = create_simulation_runtime(
        "geant4",
        sources=[],
        decomposer=SpectralDecomposer(),
        mu_by_isotope={},
        shield_params=None,
        runtime_config={
            "host": "127.0.0.1",
            "port": 5556,
            "timeout_s": 12.0,
            "auto_start_sidecar": False,
            "start_isaacsim_sidecar_with_geant4": False,
        },
    )

    assert isinstance(runtime, Geant4TCPClientRuntime)


def test_sidecar_python_resolves_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Isaac-dependent sidecars should use an environment-configured Python."""
    import sim.runtime as runtime_module

    monkeypatch.setenv("ISAACSIM_PYTHON", "/opt/isaacsim/python.sh")
    resolved = runtime_module._resolve_sidecar_python(
        {"use_mock_stage": False},
        "Geant4",
    )

    assert resolved == "/opt/isaacsim/python.sh"


def test_sidecar_python_does_not_use_isaac_env_for_mock_configs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mock sidecars should stay on the current Python unless explicitly configured."""
    import sim.runtime as runtime_module

    monkeypatch.setenv("ISAACSIM_PYTHON", "/opt/isaacsim/python.sh")
    monkeypatch.delenv("SIMBRIDGE_SIDECAR_PYTHON", raising=False)
    resolved = runtime_module._resolve_sidecar_python(
        {"use_mock_stage": True},
        "Geant4",
    )

    assert resolved == sys.executable


def test_sidecar_python_requires_env_for_real_isaac_configs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Real Isaac sidecars should fail when no configured or local Python exists."""
    import sim.runtime as runtime_module

    monkeypatch.delenv("ISAACSIM_PYTHON", raising=False)
    monkeypatch.delenv("SIMBRIDGE_SIDECAR_PYTHON", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="ISAACSIM_PYTHON"):
        runtime_module._resolve_sidecar_python({"mode": "real"}, "Isaac Sim")


def test_sidecar_python_auto_detects_local_isaacsim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Real Isaac sidecars should use a local Isaac Sim install when available."""
    import sim.runtime as runtime_module

    python_sh = tmp_path / ".local" / "isaacsim" / "5.1.0" / "python.sh"
    python_sh.parent.mkdir(parents=True)
    python_sh.write_text("#!/bin/sh\n", encoding="utf-8")
    python_sh.chmod(0o755)
    monkeypatch.delenv("ISAACSIM_PYTHON", raising=False)
    monkeypatch.delenv("SIMBRIDGE_SIDECAR_PYTHON", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    resolved = runtime_module._resolve_sidecar_python(
        {"mode": "real"},
        "Isaac Sim",
    )

    assert resolved == python_sh.as_posix()


def test_create_simulation_runtime_auto_starts_geant4_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runtime factory should auto-start Geant4 when no server is running."""
    import sim.runtime as runtime_module

    started: dict[str, object] = {}

    def _fake_tcp_available(host: str, port: int, timeout_s: float = 0.25) -> bool:
        """Pretend no sidecar is already listening."""
        return False

    def _fake_start_geant4_sidecar(
        config: dict[str, object],
        *,
        host: str,
        port: int,
        runtime_config_path: str | Path | None,
    ) -> ManagedGeant4TCPClientRuntime:
        """Capture auto-start inputs and return a managed runtime shell."""
        started.update(
            {
                "config": dict(config),
                "host": host,
                "port": port,
                "runtime_config_path": runtime_config_path,
            }
        )
        process = subprocess.Popen(["true"])
        return ManagedGeant4TCPClientRuntime(
            host=host,
            port=port,
            timeout_s=12.0,
            process=process,
        )

    monkeypatch.setattr(runtime_module, "_tcp_server_available", _fake_tcp_available)
    monkeypatch.setattr(runtime_module, "_start_geant4_sidecar", _fake_start_geant4_sidecar)

    runtime = create_simulation_runtime(
        "geant4",
        sources=[],
        decomposer=SpectralDecomposer(),
        mu_by_isotope={},
        shield_params=None,
        runtime_config={
            "host": "127.0.0.1",
            "port": 5556,
            "timeout_s": 12.0,
            "start_isaacsim_sidecar_with_geant4": False,
        },
        runtime_config_path=(
            "configs/geant4/variance_reduction_external_no_isaac_32threads.json"
        ),
    )

    assert isinstance(runtime, ManagedGeant4TCPClientRuntime)
    assert started["host"] == "127.0.0.1"
    assert started["port"] == 5556
    assert started["runtime_config_path"] == (
        "configs/geant4/variance_reduction_external_no_isaac_32threads.json"
    )
    runtime.process.wait(timeout=5.0)


def test_managed_geant4_sidecar_restart_replays_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crashed managed Geant4 bridge should restart and replay the scene reset."""
    import sim.runtime as runtime_module

    calls: list[tuple[str, dict[str, object]]] = []
    starts: list[dict[str, object]] = []
    failed_once = False

    def _fake_round_trip(
        runtime: object,
        message_type: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        """Fail the first step request and record subsequent retry ordering."""
        nonlocal failed_once
        calls.append((message_type, dict(payload)))
        if message_type == "step" and not failed_once:
            failed_once = True
            raise ConnectionRefusedError("sidecar stopped")
        return {"status": message_type}

    def _fake_start_sidecar_process(**kwargs: object) -> tuple[subprocess.Popen[str], None]:
        """Record restart inputs and return a completed process handle."""
        starts.append(dict(kwargs))
        return subprocess.Popen(["true"]), None

    monkeypatch.setattr(
        runtime_module.TCPSidecarClientRuntime,
        "_round_trip",
        _fake_round_trip,
    )
    monkeypatch.setattr(
        runtime_module,
        "_start_sidecar_process",
        _fake_start_sidecar_process,
    )
    runtime = ManagedGeant4TCPClientRuntime(
        host="127.0.0.1",
        port=5556,
        timeout_s=1.0,
        process=subprocess.Popen(["true"]),
        restart_config={
            "enabled": True,
            "max_restarts": 1,
            "script_path": "scripts/run_geant4_bridge.py",
            "config_path": "config.json",
            "config": {},
            "log_path": "sidecar.log",
            "startup_timeout_s": 1.0,
            "extra_args": [],
        },
    )

    runtime.reset({"sources": ["scene"]})
    response = runtime._round_trip("step", {"step_id": 12})

    assert response == {"status": "step"}
    assert starts
    assert calls == [
        ("reset", {"sources": ["scene"]}),
        ("step", {"step_id": 12}),
        ("reset", {"sources": ["scene"]}),
        ("step", {"step_id": 12}),
    ]


def test_create_simulation_runtime_auto_starts_isaacsim_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runtime factory should auto-start Isaac Sim when requested."""
    import sim.runtime as runtime_module

    started: dict[str, object] = {}

    def _fake_tcp_available(host: str, port: int, timeout_s: float = 0.25) -> bool:
        """Pretend no sidecar is already listening."""
        return False

    def _fake_start_isaacsim_sidecar(
        config: dict[str, object],
        runtime_config_path: str | Path | None = None,
        *,
        direct_config: bool = False,
    ) -> ManagedIsaacSimTCPClientRuntime:
        """Capture auto-start inputs and return a managed runtime shell."""
        started.update(
            {
                "config": dict(config),
                "runtime_config_path": runtime_config_path,
                "direct_config": direct_config,
            }
        )
        return ManagedIsaacSimTCPClientRuntime(
            host="127.0.0.1",
            port=5555,
            timeout_s=12.0,
            process=subprocess.Popen(["true"]),
        )

    monkeypatch.setattr(runtime_module, "_tcp_server_available", _fake_tcp_available)
    monkeypatch.setattr(runtime_module, "_start_isaacsim_sidecar", _fake_start_isaacsim_sidecar)

    runtime = create_simulation_runtime(
        "isaacsim",
        sources=[],
        decomposer=SpectralDecomposer(),
        mu_by_isotope={},
        shield_params=None,
        runtime_config={"host": "127.0.0.1", "port": 5555, "timeout_s": 12.0},
        runtime_config_path="configs/isaacsim/demo_room_gui.json",
    )

    assert isinstance(runtime, ManagedIsaacSimTCPClientRuntime)
    assert started["runtime_config_path"] == "configs/isaacsim/demo_room_gui.json"
    assert started["direct_config"] is True
    runtime.process.wait(timeout=5.0)


def test_create_simulation_runtime_reuses_running_isaacsim_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An already-running Isaac Sim sidecar should be reused instead of spawned."""
    import sim.runtime as runtime_module

    def _fake_tcp_available(host: str, port: int, timeout_s: float = 0.25) -> bool:
        """Pretend the sidecar is already listening."""
        return True

    def _fail_start(*args: object, **kwargs: object) -> None:
        """Fail if auto-start is attempted."""
        raise AssertionError("sidecar should not be auto-started")

    monkeypatch.setattr(runtime_module, "_tcp_server_available", _fake_tcp_available)
    monkeypatch.setattr(runtime_module, "_start_isaacsim_sidecar", _fail_start)

    runtime = create_simulation_runtime(
        "isaacsim",
        sources=[],
        decomposer=SpectralDecomposer(),
        mu_by_isotope={},
        shield_params=None,
        runtime_config={"host": "127.0.0.1", "port": 5555, "timeout_s": 12.0},
    )

    assert isinstance(runtime, IsaacSimTCPClientRuntime)
    assert not isinstance(runtime, ManagedIsaacSimTCPClientRuntime)


def test_create_simulation_runtime_reuses_running_geant4_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An already-running sidecar should be reused instead of spawned again."""
    import sim.runtime as runtime_module

    def _fake_tcp_available(host: str, port: int, timeout_s: float = 0.25) -> bool:
        """Pretend the sidecar is already listening."""
        return True

    def _fail_start(*args: object, **kwargs: object) -> None:
        """Fail if auto-start is attempted."""
        raise AssertionError("sidecar should not be auto-started")

    monkeypatch.setattr(runtime_module, "_tcp_server_available", _fake_tcp_available)
    monkeypatch.setattr(runtime_module, "_start_geant4_sidecar", _fail_start)

    runtime = create_simulation_runtime(
        "geant4",
        sources=[],
        decomposer=SpectralDecomposer(),
        mu_by_isotope={},
        shield_params=None,
        runtime_config={
            "host": "127.0.0.1",
            "port": 5556,
            "timeout_s": 12.0,
            "start_isaacsim_sidecar_with_geant4": False,
        },
    )

    assert isinstance(runtime, Geant4TCPClientRuntime)
    assert not isinstance(runtime, ManagedGeant4TCPClientRuntime)


def test_create_simulation_runtime_pairs_geant4_with_isaacsim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Geant4 runtime should start an Isaac Sim companion by default."""
    import sim.runtime as runtime_module

    started: dict[str, object] = {}

    def _fake_tcp_available(host: str, port: int, timeout_s: float = 0.25) -> bool:
        """Pretend no sidecar is already listening."""
        return False

    def _fake_start_geant4_sidecar(
        config: dict[str, object],
        *,
        host: str,
        port: int,
        runtime_config_path: str | Path | None,
    ) -> ManagedGeant4TCPClientRuntime:
        """Return a fake managed Geant4 runtime."""
        started["geant4"] = {"host": host, "port": port, "path": runtime_config_path}
        return ManagedGeant4TCPClientRuntime(
            host=host,
            port=port,
            timeout_s=12.0,
            process=subprocess.Popen(["true"]),
        )

    def _fake_start_isaacsim_sidecar(
        config: dict[str, object],
    ) -> ManagedIsaacSimTCPClientRuntime:
        """Return a fake managed Isaac Sim runtime."""
        started["isaacsim"] = dict(config)
        return ManagedIsaacSimTCPClientRuntime(
            host="127.0.0.1",
            port=5555,
            timeout_s=10.0,
            process=subprocess.Popen(["true"]),
        )

    monkeypatch.setattr(runtime_module, "_tcp_server_available", _fake_tcp_available)
    monkeypatch.setattr(runtime_module, "_start_geant4_sidecar", _fake_start_geant4_sidecar)
    monkeypatch.setattr(runtime_module, "_start_isaacsim_sidecar", _fake_start_isaacsim_sidecar)

    runtime = create_simulation_runtime(
        "geant4",
        sources=[],
        decomposer=SpectralDecomposer(),
        mu_by_isotope={},
        shield_params=None,
        runtime_config={"host": "127.0.0.1", "port": 5556, "timeout_s": 12.0},
        runtime_config_path=(
            "configs/geant4/variance_reduction_external_no_isaac_32threads.json"
        ),
    )

    assert isinstance(runtime, Geant4WithIsaacSimRuntime)
    assert "geant4" in started
    assert "isaacsim" in started
    assert isinstance(runtime.geant4_runtime, ManagedGeant4TCPClientRuntime)
    assert isinstance(runtime.isaacsim_runtime, ManagedIsaacSimTCPClientRuntime)
    runtime.geant4_runtime.process.wait(timeout=5.0)  # type: ignore[attr-defined]
    runtime.isaacsim_runtime.process.wait(timeout=5.0)  # type: ignore[attr-defined]


def test_geant4_with_isaacsim_runtime_moves_robot_before_observation() -> None:
    """Composite runtime should step Isaac Sim first and return Geant4 results."""

    class _FakeRuntime:
        """Runtime stub for ordering checks."""

        def __init__(self, name: str, calls: list[str]) -> None:
            """Store the runtime name and shared call log."""
            self.name = name
            self.calls = calls

        def reset(self, payload: dict | None = None) -> None:
            """Record reset calls."""
            self.calls.append(f"{self.name}:reset")

        def step(self, command: SimulationCommand) -> SimulationObservation:
            """Record step calls and return a small observation."""
            self.calls.append(f"{self.name}:step")
            return SimulationObservation(
                step_id=command.step_id,
                detector_pose_xyz=command.target_pose_xyz,
                detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                fe_orientation_index=command.fe_orientation_index,
                pb_orientation_index=command.pb_orientation_index,
                spectrum_counts=[1.0],
                energy_bin_edges_keV=[0.0, 1.0],
                metadata={"backend": self.name},
            )

        def close(self) -> None:
            """Record close calls."""
            self.calls.append(f"{self.name}:close")

    calls: list[str] = []
    runtime = Geant4WithIsaacSimRuntime(
        geant4_runtime=_FakeRuntime("geant4", calls),  # type: ignore[arg-type]
        isaacsim_runtime=_FakeRuntime("isaacsim", calls),  # type: ignore[arg-type]
    )
    command = SimulationCommand(
        step_id=0,
        target_pose_xyz=(1.0, 2.0, 0.5),
        target_base_yaw_rad=0.0,
        fe_orientation_index=1,
        pb_orientation_index=2,
        dwell_time_s=1.0,
    )

    runtime.reset({"scene": "demo"})
    observation = runtime.step(command)
    runtime.close()

    assert calls == [
        "isaacsim:reset",
        "geant4:reset",
        "isaacsim:step",
        "geant4:step",
        "geant4:close",
        "isaacsim:close",
    ]
    assert observation.metadata["backend"] == "geant4"


def test_geant4_with_isaacsim_runtime_forwards_radiation_visualization() -> None:
    """Composite runtime should forward Geant4 radiation metadata to Isaac Sim."""

    class _Geant4Runtime:
        """Runtime stub returning radiation visualization metadata."""

        def __init__(self, calls: list[str]) -> None:
            """Store the shared call log."""
            self.calls = calls

        def reset(self, payload: dict | None = None) -> None:
            """Record reset calls."""
            self.calls.append("geant4:reset")

        def step(self, command: SimulationCommand) -> SimulationObservation:
            """Return a Geant4 observation with one sampled track."""
            self.calls.append("geant4:step")
            return SimulationObservation(
                step_id=command.step_id,
                detector_pose_xyz=command.target_pose_xyz,
                detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                fe_orientation_index=command.fe_orientation_index,
                pb_orientation_index=command.pb_orientation_index,
                spectrum_counts=[3.0],
                energy_bin_edges_keV=[0.0, 1.0],
                metadata={
                    "backend": "geant4",
                    "radiation_tracks": [
                        {
                            "points_xyz": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                            "isotope": "Cs-137",
                            "detected": True,
                        }
                    ],
                    "radiation_hits": [
                        {
                            "position_xyz": [1.0, 0.0, 0.0],
                            "isotope": "Cs-137",
                        }
                    ],
                },
            )

        def close(self) -> None:
            """Record close calls."""
            self.calls.append("geant4:close")

    class _IsaacRuntime:
        """Runtime stub recording motion and visualization calls."""

        def __init__(self, calls: list[str]) -> None:
            """Store the shared call log."""
            self.calls = calls

        def reset(self, payload: dict | None = None) -> None:
            """Record reset calls."""
            self.calls.append("isaacsim:reset")

        def step(self, command: SimulationCommand) -> SimulationObservation:
            """Record robot motion calls."""
            self.calls.append("isaacsim:step")
            return SimulationObservation(
                step_id=command.step_id,
                detector_pose_xyz=command.target_pose_xyz,
                detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                fe_orientation_index=command.fe_orientation_index,
                pb_orientation_index=command.pb_orientation_index,
                spectrum_counts=[0.0],
                energy_bin_edges_keV=[0.0, 1.0],
                metadata={"backend": "isaacsim"},
            )

        def visualize_observation(self, observation: SimulationObservation) -> None:
            """Record visualization calls."""
            self.calls.append(f"isaacsim:visualize:{observation.metadata['backend']}")

        def close(self) -> None:
            """Record close calls."""
            self.calls.append("isaacsim:close")

    calls: list[str] = []
    runtime = Geant4WithIsaacSimRuntime(
        geant4_runtime=_Geant4Runtime(calls),  # type: ignore[arg-type]
        isaacsim_runtime=_IsaacRuntime(calls),  # type: ignore[arg-type]
    )
    command = SimulationCommand(
        step_id=1,
        target_pose_xyz=(1.0, 2.0, 0.5),
        target_base_yaw_rad=0.0,
        fe_orientation_index=0,
        pb_orientation_index=0,
        dwell_time_s=1.0,
    )

    runtime.reset({"scene": "demo"})
    observation = runtime.step(command)
    runtime.close()

    assert calls == [
        "isaacsim:reset",
        "geant4:reset",
        "isaacsim:step",
        "geant4:step",
        "isaacsim:visualize:geant4",
        "geant4:close",
        "isaacsim:close",
    ]
    assert len(observation.metadata["radiation_tracks"]) == 1


def test_geant4_with_isaacsim_runtime_forwards_pf_visualization() -> None:
    """Composite runtime should forward PF marker payloads to Isaac Sim."""

    class _Runtime:
        """Runtime stub recording PF visualization calls."""

        def __init__(self, name: str, calls: list[str]) -> None:
            """Store the runtime name and shared call log."""
            self.name = name
            self.calls = calls

        def reset(self, payload: dict | None = None) -> None:
            """Record reset calls."""
            self.calls.append(f"{self.name}:reset")

        def step(self, command: SimulationCommand) -> SimulationObservation:
            """Return a minimal observation."""
            self.calls.append(f"{self.name}:step")
            return SimulationObservation(
                step_id=command.step_id,
                detector_pose_xyz=command.target_pose_xyz,
                detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                fe_orientation_index=command.fe_orientation_index,
                pb_orientation_index=command.pb_orientation_index,
                spectrum_counts=[1.0],
                energy_bin_edges_keV=[0.0, 1.0],
                metadata={"backend": self.name},
            )

        def visualize_pf_state(self, payload: dict) -> None:
            """Record PF visualization calls."""
            self.calls.append(f"{self.name}:pf:{payload['step_index']}")

        def close(self) -> None:
            """Record close calls."""
            self.calls.append(f"{self.name}:close")

    calls: list[str] = []
    runtime = Geant4WithIsaacSimRuntime(
        geant4_runtime=_Runtime("geant4", calls),  # type: ignore[arg-type]
        isaacsim_runtime=_Runtime("isaacsim", calls),  # type: ignore[arg-type]
    )

    runtime.visualize_pf_state({"step_index": 7})

    assert calls == ["isaacsim:pf:7"]


def test_external_geant4_engine_reads_file_protocol(tmp_path: Path) -> None:
    """The external Geant4 engine should round-trip through the file protocol."""
    executable = tmp_path / "fake_geant4.py"
    executable.write_text(
        "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                "SCENE=''",
                "REQUEST=''",
                "RESPONSE=''",
                "while [[ $# -gt 0 ]]; do",
                "  case \"$1\" in",
                "    --scene) SCENE=\"$2\"; shift 2 ;;",
                "    --request) REQUEST=\"$2\"; shift 2 ;;",
                "    --response) RESPONSE=\"$2\"; shift 2 ;;",
                "    *) shift ;;",
                "  esac",
                "done",
                "grep -q '^SCENE ' \"$SCENE\"",
                "grep -q 'shape=spherical_octant_shell' \"$SCENE\"",
                "grep -q 'inner_radius_m=' \"$SCENE\"",
                "grep -q '^STEP ' \"$REQUEST\"",
                "printf 'META backend=geant4\\nMETA engine_mode=external\\nMETA num_primaries=42\\nSPECTRUM 1.0,2.0,3.0\\n' > \"$RESPONSE\"",
            ]
        ),
        encoding="utf-8",
    )
    executable.chmod(0o755)

    app = Geant4Application(
        app_config={
            "usd_path": "demo_room.usda",
            "use_mock_stage": True,
            "engine_mode": "external",
            "executable_path": executable.as_posix(),
            "executable_args": (),
        },
        stage_backend=FakeStageBackend(),
    )
    scene = SceneDescription(
        room_size_xyz=(10.0, 20.0, 3.0),
        obstacle_origin_xy=(0.0, 0.0),
        obstacle_cell_size_m=1.0,
        obstacle_grid_shape=(10, 20),
        obstacle_cells=[(1, 2)],
        sources=[
            SourceDescription(
                isotope="Cs-137",
                position_xyz=(5.0, 6.0, 1.0),
                intensity_cps_1m=1000.0,
            )
        ],
        usd_path="demo_room.usda",
    )
    command = SimulationCommand(
        step_id=3,
        target_pose_xyz=(2.0, 3.0, 0.5),
        target_base_yaw_rad=0.3,
        fe_orientation_index=1,
        pb_orientation_index=2,
        dwell_time_s=1.0,
    )
    app.reset(scene)
    observation = app.step(command)
    app.close()

    assert observation.metadata["backend"] == "geant4"
    assert observation.metadata["engine_mode"] == "external"
    assert observation.spectrum_counts == [1.0, 2.0, 3.0]


def test_segment_path_length_through_box_returns_expected_overlap() -> None:
    """The oriented-box intersection helper should report the correct chord length."""
    box = OrientedBox(
        center_xyz=(0.0, 0.0, 0.0),
        size_xyz=(2.0, 2.0, 2.0),
        rotation_matrix=np.eye(3, dtype=float),
    )
    path_length = segment_path_length_through_box(
        start_xyz=(-2.0, 0.0, 0.0),
        end_xyz=(2.0, 0.0, 0.0),
        box=box,
    )
    assert path_length == pytest.approx(2.0)


def test_segment_path_length_through_sphere_returns_expected_overlap() -> None:
    """The sphere intersection helper should report the correct chord length."""
    sphere = Sphere(center_xyz=(0.0, 0.0, 0.0), radius_m=1.0)
    path_length = segment_path_length_through_sphere(
        start_xyz=(-2.0, 0.0, 0.0),
        end_xyz=(2.0, 0.0, 0.0),
        sphere=sphere,
    )
    assert path_length == pytest.approx(2.0)


def test_segment_path_length_through_mesh_returns_expected_overlap() -> None:
    """The mesh intersection helper should report the correct chord length."""
    mesh = TriangleMesh(
        triangles_xyz=(
            ((-1.0, -1.0, -1.0), (1.0, -1.0, -1.0), (1.0, 1.0, -1.0)),
            ((-1.0, -1.0, -1.0), (1.0, 1.0, -1.0), (-1.0, 1.0, -1.0)),
            ((-1.0, -1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, 1.0)),
            ((-1.0, -1.0, 1.0), (1.0, 1.0, 1.0), (-1.0, 1.0, 1.0)),
            ((-1.0, -1.0, -1.0), (1.0, -1.0, -1.0), (1.0, -1.0, 1.0)),
            ((-1.0, -1.0, -1.0), (1.0, -1.0, 1.0), (-1.0, -1.0, 1.0)),
            ((1.0, -1.0, -1.0), (1.0, 1.0, -1.0), (1.0, 1.0, 1.0)),
            ((1.0, -1.0, -1.0), (1.0, 1.0, 1.0), (1.0, -1.0, 1.0)),
            ((1.0, 1.0, -1.0), (-1.0, 1.0, -1.0), (-1.0, 1.0, 1.0)),
            ((1.0, 1.0, -1.0), (-1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
            ((-1.0, 1.0, -1.0), (-1.0, -1.0, -1.0), (-1.0, -1.0, 1.0)),
            ((-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0), (-1.0, 1.0, 1.0)),
        )
    )
    path_length = segment_path_length_through_mesh(
        start_xyz=(-2.0, 0.0, 0.0),
        end_xyz=(2.0, 0.0, 0.0),
        mesh=mesh,
    )
    assert path_length == pytest.approx(2.0)


def test_real_application_obstacle_geometry_reduces_counts() -> None:
    """Obstacle intersections should reduce the measured spectrum in real mode."""
    scene_kwargs = dict(
        room_size_xyz=(10.0, 20.0, 3.0),
        obstacle_origin_xy=(0.0, 0.0),
        obstacle_cell_size_m=1.0,
        obstacle_grid_shape=(10, 20),
        sources=[
            SourceDescription(
                isotope="Cs-137",
                position_xyz=(4.5, 4.5, 1.0),
                intensity_cps_1m=5.0e6,
            )
        ],
        usd_path="demo_room.usda",
    )
    command = SimulationCommand(
        step_id=1,
        target_pose_xyz=(4.5, 1.5, 0.5),
        target_base_yaw_rad=0.0,
        fe_orientation_index=0,
        pb_orientation_index=0,
        dwell_time_s=20.0,
    )

    app_clear = IsaacSimApplication(
        use_mock=False,
        app_config={"usd_path": "demo_room.usda", "detector_height_m": 0.5},
        stage_backend=FakeStageBackend(),
    )
    app_clear.reset(SceneDescription(obstacle_cells=[], **scene_kwargs))
    clear_observation = app_clear.step(command)
    app_clear.close()

    app_blocked = IsaacSimApplication(
        use_mock=False,
        app_config={"usd_path": "demo_room.usda", "detector_height_m": 0.5},
        stage_backend=FakeStageBackend(),
    )
    app_blocked.reset(SceneDescription(obstacle_cells=[(4, 2), (4, 3)], **scene_kwargs))
    blocked_observation = app_blocked.step(command)
    app_blocked.close()

    assert float(blocked_observation.metadata["total_obstacle_path_cm"]) > 0.0
    assert sum(blocked_observation.spectrum_counts) < sum(clear_observation.spectrum_counts)


def test_real_application_stage_wall_geometry_reduces_counts() -> None:
    """Static wall boxes loaded from USD should contribute attenuation."""
    command = SimulationCommand(
        step_id=2,
        target_pose_xyz=(5.0, 19.5, 0.5),
        target_base_yaw_rad=0.0,
        fe_orientation_index=0,
        pb_orientation_index=0,
        dwell_time_s=20.0,
    )
    base_scene_kwargs = dict(
        room_size_xyz=(10.0, 20.0, 3.0),
        obstacle_origin_xy=(0.0, 0.0),
        obstacle_cell_size_m=1.0,
        obstacle_grid_shape=(10, 20),
        obstacle_cells=[],
        usd_path="demo_room.usda",
    )

    app_inside = IsaacSimApplication(
        use_mock=False,
        app_config={"usd_path": "demo_room.usda", "detector_height_m": 0.5},
        stage_backend=FakeStageBackend(),
    )
    app_inside.reset(
        SceneDescription(
            sources=[
                SourceDescription(
                    isotope="Cs-137",
                    position_xyz=(5.0, 18.5, 1.0),
                    intensity_cps_1m=5.0e6,
                )
            ],
            **base_scene_kwargs,
        )
    )
    inside_observation = app_inside.step(command)
    app_inside.close()

    app_outside = IsaacSimApplication(
        use_mock=False,
        app_config={"usd_path": "demo_room.usda", "detector_height_m": 0.5},
        stage_backend=FakeStageBackend(),
    )
    app_outside.reset(
        SceneDescription(
            sources=[
                SourceDescription(
                    isotope="Cs-137",
                    position_xyz=(5.0, 20.5, 1.0),
                    intensity_cps_1m=5.0e6,
                )
            ],
            **base_scene_kwargs,
        )
    )
    outside_observation = app_outside.step(command)
    app_outside.close()

    assert float(outside_observation.metadata["total_stage_path_cm"]) > float(
        inside_observation.metadata["total_stage_path_cm"]
    )
    assert sum(outside_observation.spectrum_counts) < sum(inside_observation.spectrum_counts)


def test_real_application_prim_material_overrides_path_rules() -> None:
    """Authored prim material metadata should override prefix-based defaults."""
    backend = FakeStageBackend()
    app = IsaacSimApplication(
        use_mock=False,
        app_config={
            "usd_path": "demo_room.usda",
            "detector_height_m": 0.5,
            "stage_material_rules": [{"path_prefix": "/World/Environment", "material": "concrete"}],
        },
        stage_backend=backend,
    )
    scene = SceneDescription(
        room_size_xyz=(10.0, 20.0, 3.0),
        obstacle_origin_xy=(0.0, 0.0),
        obstacle_cell_size_m=1.0,
        obstacle_grid_shape=(10, 20),
        obstacle_cells=[],
        sources=[
            SourceDescription(
                isotope="Cs-137",
                position_xyz=(5.0, 20.5, 1.0),
                intensity_cps_1m=5.0e6,
            )
        ],
        usd_path="demo_room.usda",
    )
    command = SimulationCommand(
        step_id=3,
        target_pose_xyz=(5.0, 19.5, 0.5),
        target_base_yaw_rad=0.0,
        fe_orientation_index=0,
        pb_orientation_index=0,
        dwell_time_s=20.0,
    )
    app.reset(scene)
    baseline_observation = app.step(command)
    backend.prims["/World/Environment/NorthWall"].metadata["material"] = "air"
    override_observation = app.step(command)
    app.close()

    assert float(baseline_observation.metadata["total_stage_path_cm"]) > 0.0
    assert sum(override_observation.spectrum_counts) > sum(baseline_observation.spectrum_counts)


def test_real_application_material_shader_inputs_override_default_mu() -> None:
    """Material density and mass attenuation inputs should override the default table."""
    backend = FakeStageBackend()
    app = IsaacSimApplication(
        use_mock=False,
        app_config={"usd_path": "demo_room.usda", "detector_height_m": 1.0},
        stage_backend=backend,
    )
    scene = SceneDescription(
        room_size_xyz=(10.0, 20.0, 3.0),
        obstacle_origin_xy=(0.0, 0.0),
        obstacle_cell_size_m=1.0,
        obstacle_grid_shape=(10, 20),
        obstacle_cells=[],
        sources=[
            SourceDescription(
                isotope="Cs-137",
                position_xyz=(5.0, 20.5, 1.0),
                intensity_cps_1m=5.0e6,
            )
        ],
        usd_path="demo_room.usda",
    )
    command = SimulationCommand(
        step_id=5,
        target_pose_xyz=(5.0, 19.5, 0.5),
        target_base_yaw_rad=0.0,
        fe_orientation_index=0,
        pb_orientation_index=0,
        dwell_time_s=20.0,
    )
    app.reset(scene)
    baseline_observation = app.step(command)
    backend.prims["/World/Looks/ConcreteMaterial"].metadata["shader_inputs"]["simbridge_mass_att_cs_137_cm2_g"] = 0.0
    override_observation = app.step(command)
    app.close()

    assert float(baseline_observation.metadata["total_stage_path_cm"]) > 0.0
    assert sum(override_observation.spectrum_counts) > sum(baseline_observation.spectrum_counts)


def test_real_application_material_preset_fills_missing_attenuation() -> None:
    """Preset fallback should attenuate even without explicit mu inputs."""
    backend = FakeStageBackend()
    app = IsaacSimApplication(
        use_mock=False,
        app_config={"usd_path": "demo_room.usda", "detector_height_m": 0.5},
        stage_backend=backend,
    )
    scene = SceneDescription(
        room_size_xyz=(10.0, 20.0, 3.0),
        obstacle_origin_xy=(0.0, 0.0),
        obstacle_cell_size_m=1.0,
        obstacle_grid_shape=(10, 20),
        obstacle_cells=[],
        sources=[
            SourceDescription(
                isotope="Cs-137",
                position_xyz=(5.0, 20.5, 1.0),
                intensity_cps_1m=5.0e6,
            )
        ],
        usd_path="demo_room.usda",
    )
    command = SimulationCommand(
        step_id=6,
        target_pose_xyz=(5.0, 19.5, 0.5),
        target_base_yaw_rad=0.0,
        fe_orientation_index=0,
        pb_orientation_index=0,
        dwell_time_s=20.0,
    )
    app.reset(scene)
    backend.prims["/World/Environment/NorthWall"].metadata["material"] = "generic_wall"
    backend.prims["/World/Environment/NorthWall"].metadata["material_preset"] = "steel"
    preset_observation = app.step(command)
    backend.prims["/World/Environment/NorthWall"].metadata["material"] = "air"
    backend.prims["/World/Environment/NorthWall"].metadata.pop("material_preset", None)
    air_observation = app.step(command)
    app.close()

    assert float(preset_observation.metadata["total_stage_path_cm"]) > 0.0
    assert sum(preset_observation.spectrum_counts) < sum(air_observation.spectrum_counts)


def test_real_application_material_composition_fills_missing_attenuation() -> None:
    """Composition fallback should derive attenuation from density and mass fractions."""
    backend = FakeStageBackend()
    app = IsaacSimApplication(
        use_mock=False,
        app_config={"usd_path": "demo_room.usda", "detector_height_m": 0.5},
        stage_backend=backend,
    )
    scene = SceneDescription(
        room_size_xyz=(10.0, 20.0, 3.0),
        obstacle_origin_xy=(0.0, 0.0),
        obstacle_cell_size_m=1.0,
        obstacle_grid_shape=(10, 20),
        obstacle_cells=[],
        sources=[
            SourceDescription(
                isotope="Cs-137",
                position_xyz=(5.0, 20.5, 1.0),
                intensity_cps_1m=5.0e6,
            )
        ],
        usd_path="demo_room.usda",
    )
    command = SimulationCommand(
        step_id=7,
        target_pose_xyz=(5.0, 19.5, 0.5),
        target_base_yaw_rad=0.0,
        fe_orientation_index=0,
        pb_orientation_index=0,
        dwell_time_s=20.0,
    )
    app.reset(scene)
    backend.prims["/World/Environment/NorthWall"].metadata["material"] = "custom_alloy"
    backend.prims["/World/Environment/NorthWall"].metadata["density_g_cm3"] = 7.5
    backend.prims["/World/Environment/NorthWall"].metadata["composition_by_mass"] = "Fe=0.9,C=0.1"
    composition_observation = app.step(command)
    backend.prims["/World/Environment/NorthWall"].metadata["material"] = "air"
    backend.prims["/World/Environment/NorthWall"].metadata.pop("density_g_cm3", None)
    backend.prims["/World/Environment/NorthWall"].metadata.pop("composition_by_mass", None)
    air_observation = app.step(command)
    app.close()

    assert float(composition_observation.metadata["total_stage_path_cm"]) > 0.0
    assert sum(composition_observation.spectrum_counts) < sum(air_observation.spectrum_counts)


def test_composition_mass_attenuation_at_energy_decreases_with_energy() -> None:
    """Interpolated composition attenuation should decrease at higher photon energies."""
    composition = {"Fe": 0.9, "C": 0.1}
    low_energy_mass_att = composition_mass_attenuation_at_energy(composition, 723.3)
    high_energy_mass_att = composition_mass_attenuation_at_energy(composition, 1332.0)

    assert low_energy_mass_att is not None
    assert high_energy_mass_att is not None
    assert float(high_energy_mass_att) < float(low_energy_mass_att)


def test_fe_pb_attenuation_uses_xcom_high_energy_values() -> None:
    """Fe/Pb mass attenuation values should match NIST XCOM anchor points."""
    fe_at_1250 = composition_mass_attenuation_at_energy({"Fe": 1.0}, 1250.0)
    pb_at_1000 = composition_mass_attenuation_at_energy({"Pb": 1.0}, 1000.0)

    assert fe_at_1250 == pytest.approx(0.05350)
    assert pb_at_1000 == pytest.approx(0.07102)


def test_real_application_energy_dependent_wall_attenuation_hardens_spectrum() -> None:
    """Energy-dependent wall attenuation should increase the high-to-low line ratio."""
    backend = FakeStageBackend()
    app = IsaacSimApplication(
        use_mock=False,
        app_config={"usd_path": "demo_room.usda", "detector_height_m": 1.0},
        stage_backend=backend,
    )
    scene = SceneDescription(
        room_size_xyz=(10.0, 20.0, 3.0),
        obstacle_origin_xy=(0.0, 0.0),
        obstacle_cell_size_m=1.0,
        obstacle_grid_shape=(10, 20),
        obstacle_cells=[],
        sources=[
                SourceDescription(
                    isotope="Eu-154",
                    position_xyz=(5.0, 20.5, 1.0),
                    intensity_cps_1m=5.0e10,
                )
        ],
        usd_path="demo_room.usda",
    )
    command = SimulationCommand(
        step_id=8,
        target_pose_xyz=(5.0, 19.5, 0.5),
        target_base_yaw_rad=0.0,
        fe_orientation_index=0,
        pb_orientation_index=0,
        dwell_time_s=30.0,
    )
    app.reset(scene)
    backend.prims["/World/Environment/NorthWall"].metadata["material"] = "lead"
    blocked_observation = app.step(command)
    backend.prims["/World/Environment/NorthWall"].metadata["material"] = "air"
    clear_observation = app.step(command)
    app.close()

    energy_axis = np.asarray(SpectralDecomposer().energy_axis, dtype=float)
    blocked_counts = np.asarray(blocked_observation.spectrum_counts, dtype=float)
    clear_counts = np.asarray(clear_observation.spectrum_counts, dtype=float)
    low_mask = (energy_axis >= 680.0) & (energy_axis <= 940.0)
    high_mask = (energy_axis >= 1180.0) & (energy_axis <= 1400.0)
    blocked_ratio = float(np.sum(blocked_counts[high_mask]) / max(np.sum(blocked_counts[low_mask]), 1.0))
    clear_ratio = float(np.sum(clear_counts[high_mask]) / max(np.sum(clear_counts[low_mask]), 1.0))

    assert float(blocked_observation.metadata["total_stage_path_cm"]) > 0.0
    assert blocked_ratio > clear_ratio


def test_real_application_mesh_geometry_reduces_counts() -> None:
    """Mesh solids loaded from the stage should contribute attenuation."""
    app = IsaacSimApplication(
        use_mock=False,
        app_config={"usd_path": "demo_room.usda", "detector_height_m": 0.5},
        stage_backend=FakeStageBackend(),
    )
    scene = SceneDescription(
        room_size_xyz=(10.0, 20.0, 3.0),
        obstacle_origin_xy=(0.0, 0.0),
        obstacle_cell_size_m=1.0,
        obstacle_grid_shape=(10, 20),
        obstacle_cells=[],
        sources=[
            SourceDescription(
                isotope="Cs-137",
                position_xyz=(7.8, 11.8, 1.0),
                intensity_cps_1m=5.0e6,
            )
        ],
        usd_path="demo_room.usda",
    )
    command = SimulationCommand(
        step_id=4,
        target_pose_xyz=(5.8, 9.8, 0.5),
        target_base_yaw_rad=0.0,
        fe_orientation_index=0,
        pb_orientation_index=0,
        dwell_time_s=20.0,
    )
    app.reset(scene)
    with_mesh_observation = app.step(command)
    backend = app._stage_backend
    assert isinstance(backend, FakeStageBackend)
    backend.prims["/World/Environment/PillarMesh"].metadata["material"] = "air"
    without_mesh_observation = app.step(command)
    app.close()

    assert float(with_mesh_observation.metadata["total_stage_path_cm"]) > 0.0
    assert sum(without_mesh_observation.spectrum_counts) > sum(with_mesh_observation.spectrum_counts)


def test_run_live_pf_uses_simulation_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real-time loop should route measurements through the runtime interface."""
    import realtime_demo

    @dataclass
    class _FakeRuntime:
        """Small runtime stub that tracks lifecycle calls."""

        reset_called: bool = False
        close_called: bool = False
        step_calls: int = 0
        reset_payload: dict | None = None

        def reset(self, payload: dict | None = None) -> None:
            """Record reset calls."""
            self.reset_called = True
            self.reset_payload = payload

        def step(self, command: SimulationCommand) -> SimulationObservation:
            """Return a zero spectrum with the requested pose."""
            self.step_calls += 1
            decomposer = SpectralDecomposer()
            energy = np.asarray(decomposer.energy_axis, dtype=float)
            step = float(np.median(np.diff(energy)))
            return SimulationObservation(
                step_id=command.step_id,
                detector_pose_xyz=command.target_pose_xyz,
                detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                fe_orientation_index=command.fe_orientation_index,
                pb_orientation_index=command.pb_orientation_index,
                spectrum_counts=np.zeros_like(energy, dtype=float).tolist(),
                energy_bin_edges_keV=np.concatenate([energy, [energy[-1] + step]]).tolist(),
                metadata={"backend": "fake"},
            )

        def close(self) -> None:
            """Record close calls."""
            self.close_called = True

    runtime = _FakeRuntime()

    class _DummyViz:
        """Minimal visualizer stub for fast regression testing."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Initialize the stub visualizer."""
            return None

        def update(self, frame: object) -> None:
            """Ignore frame updates in tests."""
            return None

        def save_final(self, path: str) -> None:
            """Skip saving final snapshots in tests."""
            return None

        def save_estimates_only(self, path: str) -> None:
            """Skip saving estimate snapshots in tests."""
            return None

    def _fake_update_pair(
        self: RotatingShieldPFEstimator,
        z_k: dict[str, float],
        pose_idx: int,
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        z_variance_k: dict[str, float] | None = None,
    ) -> None:
        """Append a lightweight measurement record without GPU updates."""
        self.measurements.append(
            MeasurementRecord(
                z_k={iso: float(v) for iso, v in z_k.items()},
                pose_idx=pose_idx,
                orient_idx=fe_index,
                live_time_s=live_time_s,
                fe_index=fe_index,
                pb_index=pb_index,
                z_variance_k=z_variance_k,
            )
        )

    def _fake_estimates(self: RotatingShieldPFEstimator) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Return a non-empty estimate for each isotope."""
        positions = np.array([[0.5, 0.5, 0.5]], dtype=float)
        strengths = np.array([1.0], dtype=float)
        return {iso: (positions.copy(), strengths.copy()) for iso in ANALYSIS_ISOTOPES}

    def _fake_counts(
        self: SpectralDecomposer,
        spectrum: np.ndarray,
        *,
        live_time_s: float = 1.0,
        **kwargs: object,
    ) -> tuple[dict[str, float], set[str]]:
        """Return deterministic counts and a stable detection set."""
        counts = {iso: 10.0 for iso in ANALYSIS_ISOTOPES}
        return counts, {"Cs-137"}

    def _fake_ig_grid(*args: object, **kwargs: object) -> np.ndarray:
        """Return a zero IG grid to bypass heavy IG evaluation."""
        return np.zeros((8, 8), dtype=float)

    def _fake_shield_selection_grid(
        *args: object,
        **kwargs: object,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Return a lightweight shield selection grid for reset-payload tests."""
        ig_scores = np.asarray(kwargs.get("ig_scores", np.zeros((8, 8))), dtype=float)
        zeros = np.zeros_like(ig_scores, dtype=float)
        return ig_scores, {
            "signature": zeros,
            "signature_utility": zeros,
            "low_count_penalty": zeros,
            "count_balance_penalty": zeros,
            "rotation_cost": zeros,
        }

    def _fake_shield_selection_grid(
        *args: object,
        **kwargs: object,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Return a lightweight shield selection grid for runtime payload tests."""
        ig_scores = np.asarray(kwargs.get("ig_scores", np.zeros((8, 8))), dtype=float)
        zeros = np.zeros_like(ig_scores, dtype=float)
        return ig_scores, {
            "signature": zeros,
            "signature_utility": zeros,
            "low_count_penalty": zeros,
            "count_balance_penalty": zeros,
            "rotation_cost": zeros,
        }

    def _fake_frame(*args: object, **kwargs: object) -> dict[str, object]:
        """Return an empty frame placeholder."""
        return {}

    def _fake_candidate_poses(*args: object, **kwargs: object) -> np.ndarray:
        """Return two deterministic candidate poses."""
        return np.array([[1.0, 1.0, 0.5], [2.0, 2.0, 0.5]], dtype=float)

    def _fake_next_pose(*args: object, **kwargs: object) -> int:
        """Select the first candidate pose."""
        return 0

    def _fake_gpu_enabled(self: RotatingShieldPFEstimator) -> bool:
        """Pretend GPU is disabled to avoid CUDA checks in tests."""
        return False

    monkeypatch.setattr(realtime_demo, "create_simulation_runtime", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(realtime_demo, "RealTimePFVisualizer", _DummyViz)
    monkeypatch.setattr(realtime_demo, "build_frame_from_pf", _fake_frame)
    monkeypatch.setattr(realtime_demo, "_compute_ig_grid", _fake_ig_grid)
    monkeypatch.setattr(
        realtime_demo,
        "_compute_shield_selection_grid",
        _fake_shield_selection_grid,
    )
    monkeypatch.setattr(realtime_demo, "generate_candidate_poses", _fake_candidate_poses)
    monkeypatch.setattr(realtime_demo, "select_next_pose_from_candidates", _fake_next_pose)
    monkeypatch.setattr(SpectralDecomposer, "isotope_counts_with_detection", _fake_counts)
    monkeypatch.setattr(RotatingShieldPFEstimator, "update_pair", _fake_update_pair)
    monkeypatch.setattr(RotatingShieldPFEstimator, "estimates", _fake_estimates)
    monkeypatch.setattr(RotatingShieldPFEstimator, "_gpu_enabled", _fake_gpu_enabled)

    estimator = run_live_pf(
        live=False,
        max_steps=1,
        max_poses=1,
        detect_threshold_abs=0.0,
        detect_threshold_rel=0.0,
        detect_consecutive=1,
        detect_min_steps=1,
        min_peaks_by_isotope={"Cs-137": 1, "Co-60": 1, "Eu-154": 1},
        ig_threshold_mode="absolute",
        ig_threshold_min=0.0,
        obstacle_layout_path=None,
        pf_config_overrides={"orientation_k": 1},
        save_outputs=False,
        return_state=True,
        sim_backend="isaacsim",
    )

    assert estimator is not None
    assert runtime.reset_called
    assert runtime.reset_payload is not None
    assert runtime.reset_payload["usd_path"] == ""
    assert runtime.reset_payload["use_config_usd_fallback"] is False
    assert runtime.reset_payload["obstacle_cells"] == []
    assert runtime.close_called
    assert runtime.step_calls == 1
    assert estimator.mission_metrics["total_measurements"] == 1
    assert (
        estimator.mission_metrics["estimated_end_to_end_time_s"]
        >= estimator.mission_metrics["total_live_time_s"]
    )


def test_empty_obstacle_grid_disables_config_usd_fallback() -> None:
    """An empty obstacle grid must be treated like no environment obstacles."""
    empty_grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(10, 20),
        blocked_cells=(),
    )
    blocked_grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(10, 20),
        blocked_cells=((1, 2),),
    )

    assert not _has_environment_obstacles(None)
    assert not _has_environment_obstacles(empty_grid)
    assert _has_environment_obstacles(blocked_grid)


def test_run_live_pf_random_environment_uses_blender_usd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Random environment mode should generate and reset with a Blender USD scene."""
    import realtime_demo

    @dataclass
    class _FakeRuntime:
        """Small runtime stub that captures reset payloads."""

        reset_payload: dict | None = None
        close_called: bool = False

        def reset(self, payload: dict | None = None) -> None:
            """Store the reset payload."""
            self.reset_payload = payload

        def step(self, command: SimulationCommand) -> SimulationObservation:
            """Return a zero spectrum with the requested pose."""
            decomposer = SpectralDecomposer()
            energy = np.asarray(decomposer.energy_axis, dtype=float)
            step = float(np.median(np.diff(energy)))
            return SimulationObservation(
                step_id=command.step_id,
                detector_pose_xyz=command.target_pose_xyz,
                detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                fe_orientation_index=command.fe_orientation_index,
                pb_orientation_index=command.pb_orientation_index,
                spectrum_counts=np.zeros_like(energy, dtype=float).tolist(),
                energy_bin_edges_keV=np.concatenate([energy, [energy[-1] + step]]).tolist(),
                metadata={"backend": "fake"},
            )

        def close(self) -> None:
            """Record close calls."""
            self.close_called = True

    class _DummyViz:
        """Minimal visualizer stub for fast regression testing."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Initialize the stub visualizer."""
            return None

        def update(self, frame: object) -> None:
            """Ignore frame updates in tests."""
            return None

        def save_final(self, path: str) -> None:
            """Skip saving final snapshots in tests."""
            return None

        def save_estimates_only(self, path: str) -> None:
            """Skip saving estimate snapshots in tests."""
            return None

    def _fake_update_pair(
        self: RotatingShieldPFEstimator,
        z_k: dict[str, float],
        pose_idx: int,
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        z_variance_k: dict[str, float] | None = None,
    ) -> None:
        """Append a lightweight measurement record without GPU updates."""
        self.measurements.append(
            MeasurementRecord(
                z_k={iso: float(v) for iso, v in z_k.items()},
                pose_idx=pose_idx,
                orient_idx=fe_index,
                live_time_s=live_time_s,
                fe_index=fe_index,
                pb_index=pb_index,
                z_variance_k=z_variance_k,
            )
        )

    def _fake_estimates(
        self: RotatingShieldPFEstimator,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Return a non-empty estimate for each isotope."""
        positions = np.array([[0.5, 0.5, 0.5]], dtype=float)
        strengths = np.array([1.0], dtype=float)
        return {iso: (positions.copy(), strengths.copy()) for iso in ANALYSIS_ISOTOPES}

    def _fake_counts(
        self: SpectralDecomposer,
        spectrum: np.ndarray,
        *,
        live_time_s: float = 1.0,
        **kwargs: object,
    ) -> tuple[dict[str, float], set[str]]:
        """Return deterministic counts and a stable detection set."""
        counts = {iso: 10.0 for iso in ANALYSIS_ISOTOPES}
        return counts, {"Cs-137"}

    def _fake_ig_grid(*args: object, **kwargs: object) -> np.ndarray:
        """Return a zero IG grid to bypass heavy IG evaluation."""
        return np.zeros((8, 8), dtype=float)

    def _fake_shield_selection_grid(
        *args: object,
        **kwargs: object,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Return a lightweight shield selection grid for reset-payload tests."""
        ig_scores = np.asarray(kwargs.get("ig_scores", np.zeros((8, 8))), dtype=float)
        zeros = np.zeros_like(ig_scores, dtype=float)
        return ig_scores, {
            "signature": zeros,
            "signature_utility": zeros,
            "low_count_penalty": zeros,
            "count_balance_penalty": zeros,
            "rotation_cost": zeros,
        }

    def _fake_frame(*args: object, **kwargs: object) -> dict[str, object]:
        """Return an empty frame placeholder."""
        return {}

    def _fake_candidate_poses(*args: object, **kwargs: object) -> np.ndarray:
        """Return two deterministic candidate poses."""
        return np.array([[1.0, 1.0, 0.5], [2.0, 2.0, 0.5]], dtype=float)

    def _fake_next_pose(*args: object, **kwargs: object) -> int:
        """Select the first candidate pose."""
        return 0

    def _fake_gpu_enabled(self: RotatingShieldPFEstimator) -> bool:
        """Pretend GPU is disabled to avoid CUDA checks in tests."""
        return False

    blender_calls: list[dict[str, object]] = []
    generated_usd = tmp_path / "generated.usda"
    config_path = tmp_path / "configs" / "isaacsim" / "manchester_random.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        '{"usd_path": "../../base_manchester.usda", "obstacle_material": "air"}\n',
        encoding="utf-8",
    )
    expected_base_usd = (tmp_path / "base_manchester.usda").resolve()

    def _fake_generate_blender_environment_usd(**kwargs: object) -> Path:
        """Capture the Blender generation call and return a fake USD path."""
        blender_calls.append(dict(kwargs))
        map_path = Path(str(kwargs["traversability_output_path"]))
        map_path.parent.mkdir(parents=True, exist_ok=True)
        map_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "source": "blender_projected_3d_environment",
                    "origin": [0.0, 0.0],
                    "cell_size": 1.0,
                    "grid_shape": [10, 20],
                    "robot_radius_m": 0.35,
                    "traversable_cells": [[1, 1], [2, 2]],
                }
            ),
            encoding="utf-8",
        )
        generated_usd.write_text("#usda 1.0\n", encoding="utf-8")
        return generated_usd

    runtime = _FakeRuntime()
    monkeypatch.setattr(realtime_demo, "create_simulation_runtime", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(realtime_demo, "generate_blender_environment_usd", _fake_generate_blender_environment_usd)
    monkeypatch.setattr(realtime_demo, "RealTimePFVisualizer", _DummyViz)
    monkeypatch.setattr(realtime_demo, "build_frame_from_pf", _fake_frame)
    monkeypatch.setattr(realtime_demo, "_compute_ig_grid", _fake_ig_grid)
    monkeypatch.setattr(
        realtime_demo,
        "_compute_shield_selection_grid",
        _fake_shield_selection_grid,
    )
    monkeypatch.setattr(realtime_demo, "generate_candidate_poses", _fake_candidate_poses)
    monkeypatch.setattr(realtime_demo, "select_next_pose_from_candidates", _fake_next_pose)
    monkeypatch.setattr(SpectralDecomposer, "isotope_counts_with_detection", _fake_counts)
    monkeypatch.setattr(RotatingShieldPFEstimator, "update_pair", _fake_update_pair)
    monkeypatch.setattr(RotatingShieldPFEstimator, "estimates", _fake_estimates)
    monkeypatch.setattr(RotatingShieldPFEstimator, "_gpu_enabled", _fake_gpu_enabled)

    estimator = run_live_pf(
        live=False,
        max_steps=1,
        max_poses=1,
        detect_threshold_abs=0.0,
        detect_threshold_rel=0.0,
        detect_consecutive=1,
        detect_min_steps=1,
        min_peaks_by_isotope={"Cs-137": 1, "Co-60": 1, "Eu-154": 1},
        ig_threshold_mode="absolute",
        ig_threshold_min=0.0,
        environment_mode="random",
        obstacle_layout_path="obstacle_layouts/random_test_unused.json",
        obstacle_seed=11,
        pf_config_overrides={"orientation_k": 1},
        save_outputs=False,
        return_state=True,
        sim_backend="isaacsim",
        sim_config_path=config_path.as_posix(),
        blender_output_path=generated_usd.as_posix(),
    )

    assert estimator is not None
    assert blender_calls
    assert blender_calls[0]["base_usd_path"] == expected_base_usd
    assert blender_calls[0]["obstacle_material"] == "air"
    assert blender_calls[0]["traversability_output_path"] == generated_usd.with_suffix(
        ".traversability.json"
    )
    assert runtime.reset_payload is not None
    assert runtime.reset_payload["usd_path"] == generated_usd.as_posix()
    assert runtime.reset_payload["author_obstacle_prims"] is True
    assert runtime.reset_payload["obstacle_material"] == "air"
    assert runtime.reset_payload["obstacle_cells"]
    map_path = Path(str(runtime.reset_payload["traversability_map_path"]))
    map_png_path = Path(str(runtime.reset_payload["traversability_map_png_path"]))
    assert map_path.exists()
    assert map_png_path.exists()
    assert json.loads(map_path.read_text(encoding="utf-8"))["source"] == (
        "blender_projected_3d_environment"
    )
    assert runtime.reset_payload["robot_radius_m"] == pytest.approx(0.35)
    assert runtime.close_called
