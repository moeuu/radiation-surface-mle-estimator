"""Geant4-side observation engine backed by an external Geant4 executable."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import selectors
import subprocess
import tempfile
import time
from typing import Any

import numpy as np

from sim.geant4_app.io_format import read_response_file, write_request_file, write_scene_file
from sim.geant4_app.scene_export import ExportedGeant4Scene
from sim.radiation_visualization import (
    RadiationVisualizationConfig,
    build_visualization_metadata_from_scene,
)
from spectrum.pipeline import SpectralDecomposer


@dataclass(frozen=True)
class Geant4StepRequest:
    """Describe a single Geant4 step request."""

    step_id: int
    dwell_time_s: float
    seed: int
    detector_pose_xyz: tuple[float, float, float]
    detector_quat_wxyz: tuple[float, float, float, float]
    fe_shield_pose_xyz: tuple[float, float, float]
    fe_shield_quat_wxyz: tuple[float, float, float, float]
    pb_shield_pose_xyz: tuple[float, float, float]
    pb_shield_quat_wxyz: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable request payload."""
        return {
            "step_id": int(self.step_id),
            "dwell_time_s": float(self.dwell_time_s),
            "seed": int(self.seed),
            "detector_pose_xyz": list(self.detector_pose_xyz),
            "detector_quat_wxyz": list(self.detector_quat_wxyz),
            "fe_shield_pose_xyz": list(self.fe_shield_pose_xyz),
            "fe_shield_quat_wxyz": list(self.fe_shield_quat_wxyz),
            "pb_shield_pose_xyz": list(self.pb_shield_pose_xyz),
            "pb_shield_quat_wxyz": list(self.pb_shield_quat_wxyz),
        }


@dataclass(frozen=True)
class Geant4EngineConfig:
    """Collect external Geant4 engine settings."""

    physics_profile: str = "balanced"
    thread_count: int = 1
    random_seed_base: int = 123
    dead_time_tau_s: float = 5.813e-9
    scatter_gain: float = 0.0
    executable_path: str | None = None
    executable_args: tuple[str, ...] = ()
    timeout_s: float = 120.0
    persistent_process: bool = False
    source_rate_model: str = "detector_cps_1m"
    source_bias_mode: str = "detector_cone"
    source_bias_cone_half_angle_deg: float = 0.0
    source_bias_isotropic_fraction: float = 0.1
    detector_scoring_mode: str = "full_transport"
    secondary_transport_mode: str = "full_transport"
    primary_sampling_fraction: float = 1.0
    radiation_visualization: RadiationVisualizationConfig = field(default_factory=RadiationVisualizationConfig)


class Geant4Engine(ABC):
    """Define the Geant4 engine interface used by the sidecar app."""

    @abstractmethod
    def load_scene(self, scene: ExportedGeant4Scene) -> bool:
        """Load a scene and return whether a cached world was reused."""

    @abstractmethod
    def simulate(self, request: Geant4StepRequest) -> tuple[np.ndarray, dict[str, Any]]:
        """Run one transport step and return a spectrum plus metadata."""

    @abstractmethod
    def close(self) -> None:
        """Release engine-owned resources."""


class ExternalCommandGeant4Engine(Geant4Engine):
    """Delegate transport to an external executable or persistent native process."""

    def __init__(self, config: Geant4EngineConfig) -> None:
        """Store external-engine launch configuration."""
        if config.executable_path in (None, ""):
            raise ValueError("executable_path is required for the external Geant4 engine.")
        self.config = config
        self.scene: ExportedGeant4Scene | None = None
        self._last_cache_hit = False
        self.decomposer = SpectralDecomposer()
        self._persistent_process: subprocess.Popen[str] | None = None
        self._persistent_tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._persistent_scene_path: Path | None = None
        self._persistent_scene_hash: str | None = None

    def load_scene(self, scene: ExportedGeant4Scene) -> bool:
        """Store the latest scene for the next external simulation call."""
        cache_hit = self.scene is not None and self.scene.scene_hash == scene.scene_hash
        self.scene = scene
        self._last_cache_hit = cache_hit
        if not cache_hit:
            self._persistent_scene_hash = None
            self._persistent_scene_path = None
        return cache_hit

    def simulate(self, request: Geant4StepRequest) -> tuple[np.ndarray, dict[str, Any]]:
        """Call the configured external executable and parse its response."""
        if self.scene is None:
            raise RuntimeError("Geant4 scene was not loaded before simulate().")
        if self.config.persistent_process:
            spectrum, metadata = self._simulate_persistent(request)
        else:
            spectrum, metadata = self._simulate_one_shot(request)
        metadata.setdefault("backend", "geant4")
        metadata.setdefault("engine_mode", "external")
        metadata.setdefault("scene_hash", self.scene.scene_hash)
        metadata.setdefault("cache_hit", self._last_cache_hit)
        metadata.setdefault("seed", int(request.seed))
        metadata.update(
            build_visualization_metadata_from_scene(
                self.scene,
                request,
                seed=int(request.seed),
                config=self.config.radiation_visualization,
                library=self.decomposer.library,
                mode="geant4-external-representative",
                scatter_gain=self.config.scatter_gain,
            )
        )
        return spectrum, metadata

    def _simulate_one_shot(self, request: Geant4StepRequest) -> tuple[np.ndarray, dict[str, Any]]:
        """Run one request by launching a fresh native executable process."""
        if self.scene is None:
            raise RuntimeError("Geant4 scene was not loaded before simulate().")
        with tempfile.TemporaryDirectory(prefix="geant4_sidecar_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            scene_path = tmp_path / "scene.txt"
            request_path = tmp_path / "request.txt"
            response_path = tmp_path / "response.txt"
            write_scene_file(self.scene, scene_path)
            write_request_file(request, request_path)
            command = [
                str(self.config.executable_path),
                "--scene",
                scene_path.as_posix(),
                "--request",
                request_path.as_posix(),
                "--response",
                response_path.as_posix(),
                "--physics-profile",
                self.config.physics_profile,
                "--threads",
                str(self.config.thread_count),
                "--dead-time-tau-s",
                str(self.config.dead_time_tau_s),
                *self._source_bias_args(),
                *self.config.executable_args,
            ]
            result = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=self.config.timeout_s,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "External Geant4 executable failed: "
                    f"returncode={result.returncode} stderr={result.stderr.strip()}"
                )
            spectrum, metadata = read_response_file(response_path)
        return spectrum, metadata

    def _simulate_persistent(self, request: Geant4StepRequest) -> tuple[np.ndarray, dict[str, Any]]:
        """Run one request through a persistent native executable process."""
        if self.scene is None:
            raise RuntimeError("Geant4 scene was not loaded before simulate().")
        restart_count = 0
        for attempt in range(2):
            try:
                spectrum, metadata = self._simulate_persistent_once(request)
                if restart_count > 0:
                    metadata["persistent_restart_count"] = int(restart_count)
                return spectrum, metadata
            except RuntimeError as exc:
                if (
                    attempt > 0
                    or "Persistent Geant4 executable exited unexpectedly" not in str(exc)
                ):
                    raise
                restart_count += 1
                self._close_persistent_process()
        raise RuntimeError("Persistent Geant4 retry loop terminated unexpectedly.")

    def _simulate_persistent_once(
        self,
        request: Geant4StepRequest,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Run one request through the current persistent native process."""
        if self.scene is None:
            raise RuntimeError("Geant4 scene was not loaded before simulate().")
        process = self._ensure_persistent_process()
        tmp_path = self._persistent_tmp_path()
        scene_path = self._ensure_persistent_scene_file(tmp_path)
        request_path = tmp_path / f"request_{int(request.step_id)}.txt"
        response_path = tmp_path / f"response_{int(request.step_id)}.txt"
        write_request_file(request, request_path)
        response_path.unlink(missing_ok=True)
        command = (
            "RUN "
            f"scene={_encode_token(scene_path.as_posix())} "
            f"request={_encode_token(request_path.as_posix())} "
            f"response={_encode_token(response_path.as_posix())}\n"
        )
        if process.stdin is None:
            raise RuntimeError("Persistent Geant4 process does not expose stdin.")
        process.stdin.write(command)
        process.stdin.flush()
        self._wait_for_persistent_ok(process, response_path)
        return read_response_file(response_path)

    def _ensure_persistent_process(self) -> subprocess.Popen[str]:
        """Start the persistent native process if it is not already running."""
        if self._persistent_process is not None and self._persistent_process.poll() is None:
            return self._persistent_process
        if self._persistent_process is not None:
            self._persistent_process = None
        self._persistent_tmpdir = tempfile.TemporaryDirectory(prefix="geant4_persistent_")
        command = [
            str(self.config.executable_path),
            "--persistent",
            "--physics-profile",
            self.config.physics_profile,
            "--threads",
            str(self.config.thread_count),
            "--dead-time-tau-s",
            str(self.config.dead_time_tau_s),
            *self._source_bias_args(),
            *self.config.executable_args,
        ]
        self._persistent_process = subprocess.Popen(
            command,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        return self._persistent_process

    def _persistent_tmp_path(self) -> Path:
        """Return the persistent process temporary directory path."""
        if self._persistent_tmpdir is None:
            raise RuntimeError("Persistent Geant4 temporary directory is not initialized.")
        return Path(self._persistent_tmpdir.name)

    def _ensure_persistent_scene_file(self, tmp_path: Path) -> Path:
        """Write the scene file once per scene hash for the persistent process."""
        if self.scene is None:
            raise RuntimeError("Geant4 scene was not loaded before simulate().")
        if (
            self._persistent_scene_path is not None
            and self._persistent_scene_hash == self.scene.scene_hash
            and self._persistent_scene_path.exists()
        ):
            return self._persistent_scene_path
        scene_path = tmp_path / f"scene_{self.scene.scene_hash[:16]}.txt"
        write_scene_file(self.scene, scene_path)
        self._persistent_scene_hash = self.scene.scene_hash
        self._persistent_scene_path = scene_path
        return scene_path

    def _wait_for_persistent_ok(
        self,
        process: subprocess.Popen[str],
        response_path: Path,
    ) -> None:
        """Wait until the persistent process reports request completion."""
        if process.stdout is None:
            raise RuntimeError("Persistent Geant4 process does not expose stdout.")
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + float(self.config.timeout_s)
        output_lines: list[str] = []
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    remaining = process.stdout.read() or ""
                    raise RuntimeError(
                        "Persistent Geant4 executable exited unexpectedly: "
                        f"returncode={process.returncode} output={(remaining or '').strip()}"
                    )
                timeout = max(0.0, min(0.25, deadline - time.monotonic()))
                events = selector.select(timeout)
                for key, _ in events:
                    line = key.fileobj.readline()
                    if not line:
                        continue
                    stripped = line.strip()
                    output_lines.append(stripped)
                    if stripped.startswith("SIMBRIDGE_OK"):
                        if not response_path.exists():
                            raise RuntimeError(
                                "Persistent Geant4 completed without writing response file."
                            )
                        return
                    if stripped.startswith("SIMBRIDGE_ERR"):
                        raise RuntimeError(f"Persistent Geant4 executable failed: {stripped}")
        finally:
            selector.close()
        tail = "\n".join(output_lines[-20:])
        raise TimeoutError(
            "Timed out waiting for persistent Geant4 response. "
            f"Recent native output:\n{tail}"
        )

    def close(self) -> None:
        """Release the cached scene reference."""
        self._close_persistent_process()
        self.scene = None

    def _close_persistent_process(self) -> None:
        """Terminate the persistent native process and remove temp files."""
        process = self._persistent_process
        self._persistent_process = None
        if process is not None and process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.write("SHUTDOWN\n")
                    process.stdin.flush()
            except OSError:
                pass
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
        if self._persistent_tmpdir is not None:
            self._persistent_tmpdir.cleanup()
        self._persistent_tmpdir = None
        self._persistent_scene_path = None
        self._persistent_scene_hash = None

    def _source_bias_args(self) -> list[str]:
        """Return native executable arguments for the configured source bias mode."""
        return [
            "--source-rate-model",
            str(self.config.source_rate_model),
            "--source-bias-mode",
            str(self.config.source_bias_mode),
            "--source-bias-cone-half-angle-deg",
            str(float(self.config.source_bias_cone_half_angle_deg)),
            "--source-bias-isotropic-fraction",
            str(float(self.config.source_bias_isotropic_fraction)),
            "--detector-scoring-mode",
            str(self.config.detector_scoring_mode),
            "--secondary-transport-mode",
            str(self.config.secondary_transport_mode),
            "--primary-sampling-fraction",
            str(float(self.config.primary_sampling_fraction)),
        ]


def build_geant4_engine(config: Geant4EngineConfig, *, engine_mode: str) -> Geant4Engine:
    """Instantiate the requested Geant4 engine implementation."""
    normalized = engine_mode.strip().lower()
    if normalized == "external":
        return ExternalCommandGeant4Engine(config)
    raise ValueError(
        f"Unsupported Geant4 engine mode: {engine_mode}. "
        "Only 'external' native Geant4 transport is supported."
    )


def _encode_token(value: str) -> str:
    """Encode a whitespace-free line-protocol token."""
    return str(value).replace(" ", "%20")
