"""Simulation runtime abstractions for analytic and sidecar backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any

from measurement.model import PointSource
from sim.protocol import (
    SimulationCommand,
    SimulationObservation,
    decode_message,
    encode_message,
)
from sim.approx.python_transport import PythonTransportSpectrumModel
from spectrum.pipeline import SpectralDecomposer


class SimulationRuntime(ABC):
    """Define the runtime interface used by the real-time PF loop."""

    @abstractmethod
    def reset(self, payload: dict[str, Any] | None = None) -> None:
        """Reset simulator state for a new episode."""

    @abstractmethod
    def step(self, command: SimulationCommand) -> SimulationObservation:
        """Execute one step and return the resulting observation."""

    @abstractmethod
    def close(self) -> None:
        """Release runtime resources."""


class AnalyticSimulationRuntime(SimulationRuntime):
    """Provide approximate observations using the Python transport debug model."""

    def __init__(
        self,
        *,
        sources: list[PointSource],
        decomposer: SpectralDecomposer,
        mu_by_isotope: dict[str, object],
        shield_params: Any,
        rng_seed: int = 123,
        obstacle_height_m: float = 2.0,
        obstacle_material: str = "concrete",
        scatter_gain: float = 0.03,
        dead_time_s: float = 0.0,
        detector_model: dict[str, Any] | None = None,
    ) -> None:
        """Store simulator inputs for analytic observation generation."""
        self.transport_model = PythonTransportSpectrumModel(
            sources=sources,
            decomposer=decomposer,
            mu_by_isotope=mu_by_isotope,
            shield_params=shield_params,
            obstacle_height_m=float(obstacle_height_m),
            obstacle_material=str(obstacle_material),
            scatter_gain=float(scatter_gain),
            rng_seed=int(rng_seed),
            dead_time_s=float(dead_time_s),
            detector_model=detector_model,
        )

    def reset(self, payload: dict[str, Any] | None = None) -> None:
        """Reset sources and static scene geometry from the episode payload."""
        self.transport_model.reset_from_payload(payload)

    def step(self, command: SimulationCommand) -> SimulationObservation:
        """Generate a spectrum at the requested pose and shield orientation."""
        return self.transport_model.observe(
            command,
            detector_pose_xyz=command.target_pose_xyz,
            detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            backend_label="analytic",
        )

    def close(self) -> None:
        """Closing the analytic runtime is a no-op."""
        return None


class TCPSidecarClientRuntime(SimulationRuntime):
    """Send simulator commands to a remote sidecar over TCP."""

    def __init__(
        self,
        host: str,
        port: int,
        timeout_s: float = 10.0,
        *,
        close_on_close: bool = True,
    ) -> None:
        """Store sidecar connection parameters."""
        self.host = str(host)
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.close_on_close = bool(close_on_close)

    def _round_trip(self, message_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a single request and return the response payload."""
        with socket.create_connection((self.host, self.port), timeout=self.timeout_s) as conn:
            conn.sendall(encode_message(message_type, payload))
            conn.shutdown(socket.SHUT_WR)
            chunks: list[bytes] = []
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
        if not chunks:
            raise RuntimeError("Simulator sidecar returned an empty response.")
        response_type, response_payload = decode_message(b"".join(chunks).strip())
        if response_type == "error":
            raise RuntimeError(str(response_payload.get("message", "Unknown sidecar error.")))
        if response_type != "ok":
            raise RuntimeError(f"Unexpected sidecar response type: {response_type}")
        return response_payload

    def reset(self, payload: dict[str, Any] | None = None) -> None:
        """Reset the remote sidecar episode state."""
        self._round_trip("reset", payload or {})

    def step(self, command: SimulationCommand) -> SimulationObservation:
        """Execute one remote step and parse the resulting observation."""
        payload = self._round_trip("step", command.to_dict())
        return SimulationObservation.from_dict(payload["observation"])

    def close(self) -> None:
        """Request clean sidecar shutdown and ignore transport errors."""
        if not self.close_on_close:
            return None
        try:
            self._round_trip("shutdown", {})
        except OSError:
            return None
        except RuntimeError:
            return None


class IsaacSimTCPClientRuntime(TCPSidecarClientRuntime):
    """Backward-compatible TCP client for the Isaac Sim sidecar."""

    def visualize_observation(self, observation: SimulationObservation) -> None:
        """Send observation metadata to Isaac Sim for stage visualization."""
        self._round_trip("visualize", {"observation": observation.to_dict()})

    def visualize_pf_state(self, payload: dict[str, Any]) -> None:
        """Send PF particle and estimate markers to Isaac Sim for visualization."""
        self._round_trip("visualize_pf", payload)


class Geant4TCPClientRuntime(TCPSidecarClientRuntime):
    """TCP client for the Geant4 sidecar."""


class Geant4WithIsaacSimRuntime(SimulationRuntime):
    """Route robot motion to Isaac Sim while using Geant4 observations."""

    def __init__(
        self,
        *,
        geant4_runtime: SimulationRuntime,
        isaacsim_runtime: SimulationRuntime,
    ) -> None:
        """Store the paired runtimes."""
        self.geant4_runtime = geant4_runtime
        self.isaacsim_runtime = isaacsim_runtime

    def reset(self, payload: dict[str, Any] | None = None) -> None:
        """Reset both simulators with the same scene payload."""
        scene_payload = payload or {}
        self.isaacsim_runtime.reset(scene_payload)
        self.geant4_runtime.reset(scene_payload)

    def step(self, command: SimulationCommand) -> SimulationObservation:
        """Move the Isaac Sim robot and return the Geant4 observation."""
        self.isaacsim_runtime.step(command)
        observation = self.geant4_runtime.step(command)
        visualizer = getattr(self.isaacsim_runtime, "visualize_observation", None)
        if visualizer is not None:
            visualizer(observation)
        return observation

    def visualize_pf_state(self, payload: dict[str, Any]) -> None:
        """Forward PF particle visualization to the Isaac Sim companion runtime."""
        visualizer = getattr(self.isaacsim_runtime, "visualize_pf_state", None)
        if visualizer is not None:
            visualizer(payload)

    def close(self) -> None:
        """Close both runtimes, preserving the first close error if any."""
        first_error: Exception | None = None
        for runtime in (self.geant4_runtime, self.isaacsim_runtime):
            try:
                runtime.close()
            except Exception as exc:  # pragma: no cover - defensive cleanup path
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


class ManagedIsaacSimTCPClientRuntime(IsaacSimTCPClientRuntime):
    """Isaac Sim TCP client that owns an auto-started sidecar process."""

    def __init__(
        self,
        host: str,
        port: int,
        timeout_s: float,
        *,
        process: subprocess.Popen[str],
        log_handle: object | None = None,
        temp_config_path: Path | None = None,
        close_on_close: bool = True,
    ) -> None:
        """Store the client parameters and owned process handles."""
        super().__init__(host=host, port=port, timeout_s=timeout_s, close_on_close=close_on_close)
        self.process = process
        self.log_handle = log_handle
        self.temp_config_path = temp_config_path

    def close(self) -> None:
        """Shutdown the sidecar and clean up process resources."""
        super().close()
        if not self.close_on_close:
            if self.log_handle is not None:
                close = getattr(self.log_handle, "close", None)
                if close is not None:
                    close()
            if self.temp_config_path is not None:
                self.temp_config_path.unlink(missing_ok=True)
            return
        try:
            self.process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5.0)
        if self.log_handle is not None:
            close = getattr(self.log_handle, "close", None)
            if close is not None:
                close()
        if self.temp_config_path is not None:
            self.temp_config_path.unlink(missing_ok=True)


class ManagedGeant4TCPClientRuntime(Geant4TCPClientRuntime):
    """Geant4 TCP client that owns an auto-started sidecar process."""

    def __init__(
        self,
        host: str,
        port: int,
        timeout_s: float,
        *,
        process: subprocess.Popen[str],
        log_handle: object | None = None,
        temp_config_path: Path | None = None,
        restart_config: dict[str, Any] | None = None,
    ) -> None:
        """Store the client parameters and owned process handles."""
        super().__init__(host=host, port=port, timeout_s=timeout_s)
        self.process = process
        self.log_handle = log_handle
        self.temp_config_path = temp_config_path
        self.restart_config = {} if restart_config is None else dict(restart_config)
        self._last_reset_payload: dict[str, Any] | None = None
        self._restart_count = 0

    def reset(self, payload: dict[str, Any] | None = None) -> None:
        """Reset the sidecar and retain the payload for crash recovery."""
        reset_payload = {} if payload is None else dict(payload)
        self._last_reset_payload = reset_payload
        self._round_trip("reset", reset_payload)

    def _round_trip(self, message_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a request, restarting a crashed managed sidecar when safe."""
        try:
            return super()._round_trip(message_type, payload)
        except OSError:
            if not self._can_restart_for(message_type):
                raise
            self._restart_sidecar()
            if message_type != "reset":
                if self._last_reset_payload is None:
                    raise RuntimeError(
                        "Managed Geant4 sidecar crashed before any reset payload was recorded."
                    )
                super()._round_trip("reset", self._last_reset_payload)
            return super()._round_trip(message_type, payload)

    def _can_restart_for(self, message_type: str) -> bool:
        """Return whether a transport failure can be retried without changing semantics."""
        if message_type == "shutdown":
            return False
        if not bool(self.restart_config.get("enabled", True)):
            return False
        max_restarts = int(self.restart_config.get("max_restarts", 2))
        return self._restart_count < max_restarts

    def _restart_sidecar(self) -> None:
        """Restart the owned Geant4 bridge process using the original config."""
        script_path = Path(str(self.restart_config["script_path"]))
        config_path = Path(str(self.restart_config["config_path"]))
        config = dict(self.restart_config.get("config", {}))
        log_path = Path(str(self.restart_config["log_path"]))
        extra_args = list(self.restart_config.get("extra_args", []))
        startup_timeout_s = float(self.restart_config.get("startup_timeout_s", 30.0))
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5.0)
        if self.log_handle is not None:
            close = getattr(self.log_handle, "close", None)
            if close is not None:
                close()
            self.log_handle = None
        self._restart_count += 1
        process, log_handle = _start_sidecar_process(
            script_path=script_path,
            config_path=config_path,
            config=config,
            host=self.host,
            port=self.port,
            timeout_s=startup_timeout_s,
            log_path=log_path,
            sidecar_name="Geant4",
            extra_args=extra_args,
        )
        self.process = process
        self.log_handle = log_handle

    def close(self) -> None:
        """Shutdown the sidecar and clean up process resources."""
        super().close()
        try:
            self.process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5.0)
        if self.log_handle is not None:
            close = getattr(self.log_handle, "close", None)
            if close is not None:
                close()
        if self.temp_config_path is not None:
            self.temp_config_path.unlink(missing_ok=True)


def load_runtime_config(path: str | Path | None) -> dict[str, Any]:
    """Load a JSON runtime configuration file with optional inheritance."""
    return _load_runtime_config(path, seen=set())


def _load_runtime_config(
    path: str | Path | None,
    *,
    seen: set[Path],
) -> dict[str, Any]:
    """Load a runtime config and recursively merge an ``extends`` parent."""
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Simulation config not found: {config_path}")
    config_path = config_path.resolve()
    if config_path in seen:
        raise ValueError(f"Cyclic runtime config inheritance at {config_path}")
    seen.add(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Simulation config must be a JSON object.")
    parent_ref = data.pop("extends", None)
    if parent_ref in (None, ""):
        return data
    parent_path = Path(str(parent_ref)).expanduser()
    if not parent_path.is_absolute():
        parent_path = (config_path.parent / parent_path).resolve()
    parent = _load_runtime_config(parent_path, seen=seen)
    merged = dict(parent)
    merged.update(data)
    return merged


def _repo_root() -> Path:
    """Return the repository root path."""
    return Path(__file__).resolve().parents[2]


def _tcp_server_available(host: str, port: int, timeout_s: float = 0.25) -> bool:
    """Return True when a TCP server is already accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _write_temp_sidecar_config(config: dict[str, Any]) -> Path:
    """Write an ephemeral sidecar config file and return its path."""
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="geant4_sidecar_",
        delete=False,
        encoding="utf-8",
    )
    with handle:
        json.dump(config, handle, indent=2, sort_keys=True)
    return Path(handle.name)


def _merged_config_from_path(
    config_path: Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a config file and apply optional overrides."""
    loaded = load_runtime_config(config_path)
    merged = dict(loaded)
    if overrides:
        merged.update(overrides)
    return merged


def _resolve_executable_path(path_value: str) -> str:
    """Expand shell variables and user home markers in an executable path."""
    return Path(os.path.expandvars(path_value)).expanduser().as_posix()


def _local_isaacsim_python_candidates() -> list[Path]:
    """Return likely Isaac Sim Python launchers installed on this machine."""
    candidates: list[Path] = []
    home = Path.home()
    local_root = home / ".local" / "isaacsim"
    if local_root.exists():
        candidates.extend(sorted(local_root.glob("*/python.sh"), reverse=True))
    candidates.extend(
        [
            home / ".local" / "isaacsim" / "python.sh",
            home / "isaacsim" / "python.sh",
            Path("/opt/isaacsim/python.sh"),
        ]
    )
    return candidates


def _resolve_local_isaacsim_python() -> str | None:
    """Return a local Isaac Sim Python launcher if one exists."""
    for candidate in _local_isaacsim_python_candidates():
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate.as_posix()
    return None


def _config_requires_isaacsim_python(config: dict[str, Any]) -> bool:
    """Return True when a sidecar must be launched with Isaac Sim's Python."""
    if bool(config.get("requires_isaacsim_python", False)):
        return True
    if str(config.get("mode", "")).strip().lower() == "real":
        return True
    return config.get("use_mock_stage") is False


def _resolve_sidecar_python(config: dict[str, Any], sidecar_name: str) -> str:
    """Resolve the Python executable used to launch a sidecar process."""
    configured = config.get("sidecar_python")
    if configured not in (None, ""):
        return _resolve_executable_path(str(configured))

    env_names: list[str] = ["SIMBRIDGE_SIDECAR_PYTHON"]
    configured_env = config.get("sidecar_python_env")
    if configured_env not in (None, ""):
        env_names.insert(0, str(configured_env))
    if (
        _config_requires_isaacsim_python(config)
        and "ISAACSIM_PYTHON" not in env_names
    ):
        env_names.append("ISAACSIM_PYTHON")
    for env_name in env_names:
        env_value = os.environ.get(env_name)
        if env_value:
            return _resolve_executable_path(env_value)

    if _config_requires_isaacsim_python(config):
        local_python = _resolve_local_isaacsim_python()
        if local_python is not None:
            return local_python
        raise RuntimeError(
            f"{sidecar_name} sidecar requires Isaac Sim Python. Set "
            "ISAACSIM_PYTHON=/path/to/isaacsim/python.sh or set "
            "sidecar_python in the config."
        )
    return sys.executable


def _start_sidecar_process(
    *,
    script_path: Path,
    config_path: Path,
    config: dict[str, Any],
    host: str,
    port: int,
    timeout_s: float,
    log_path: Path,
    sidecar_name: str,
    extra_args: list[str] | None = None,
) -> tuple[subprocess.Popen[str], object]:
    """Start a sidecar subprocess and wait until it accepts TCP connections."""
    root = _repo_root()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    python_executable = _resolve_sidecar_python(config, sidecar_name)
    log_handle = log_path.open("a", encoding="utf-8")
    command = [
        python_executable,
        script_path.as_posix(),
        "--config",
        config_path.as_posix(),
    ]
    if extra_args:
        command.extend(extra_args)
    process = subprocess.Popen(
        command,
        cwd=root.as_posix(),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log_handle.close()
            raise RuntimeError(
                f"Auto-started {sidecar_name} sidecar exited before accepting connections. "
                f"See log: {log_path}"
            )
        if _tcp_server_available(host, port):
            print(f"Auto-started {sidecar_name} sidecar on {host}:{port} (log: {log_path})")
            return process, log_handle
        time.sleep(0.1)
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)
    log_handle.close()
    raise TimeoutError(
        f"Timed out waiting for auto-started {sidecar_name} sidecar on {host}:{port}. "
        f"See log: {log_path}"
    )


def _resolve_geant4_sidecar_config_path(
    config: dict[str, Any],
    runtime_config_path: str | Path | None,
) -> tuple[Path, Path | None]:
    """Return the config path passed to the sidecar and an optional temp file."""
    if runtime_config_path is not None:
        return Path(runtime_config_path).expanduser().resolve(), None
    configured = config.get("sidecar_config_path")
    if configured not in (None, ""):
        return Path(str(configured)).expanduser().resolve(), None
    default_path = (
        _repo_root()
        / "configs"
        / "geant4"
        / "variance_reduction_external_no_isaac_32threads.json"
    )
    if not config and default_path.exists():
        return default_path, None
    temp_path = _write_temp_sidecar_config(config)
    return temp_path, temp_path


def _start_geant4_sidecar(
    config: dict[str, Any],
    *,
    host: str,
    port: int,
    runtime_config_path: str | Path | None,
) -> ManagedGeant4TCPClientRuntime:
    """Start a Geant4 bridge sidecar subprocess and return its managed client."""
    root = _repo_root()
    script_path = root / "scripts" / "run_geant4_bridge.py"
    config_path, temp_config_path = _resolve_geant4_sidecar_config_path(
        config,
        runtime_config_path,
    )
    log_path = Path(
        str(config.get("sidecar_log_path", root / "results" / "sidecars" / f"geant4_bridge_{port}.log"))
    ).expanduser()
    if not log_path.is_absolute():
        log_path = (root / log_path).resolve()
    startup_timeout_s = float(config.get("sidecar_startup_timeout_s", 30.0))
    try:
        process, log_handle = _start_sidecar_process(
            script_path=script_path,
            config_path=config_path,
            config=config,
            host=host,
            port=port,
            timeout_s=startup_timeout_s,
            log_path=log_path,
            sidecar_name="Geant4",
            extra_args=["--mock-stage"] if bool(config.get("sidecar_mock_stage", False)) else None,
        )
    except Exception:
        if temp_config_path is not None:
            temp_config_path.unlink(missing_ok=True)
        raise
    return ManagedGeant4TCPClientRuntime(
        host=host,
        port=port,
        timeout_s=float(config.get("timeout_s", 120.0)),
        process=process,
        log_handle=log_handle,
        temp_config_path=temp_config_path,
        restart_config={
            "enabled": bool(config.get("sidecar_restart_on_disconnect", True)),
            "max_restarts": int(config.get("sidecar_max_restarts", 2)),
            "script_path": script_path.as_posix(),
            "config_path": config_path.as_posix(),
            "config": dict(config),
            "log_path": log_path.as_posix(),
            "startup_timeout_s": startup_timeout_s,
            "extra_args": (
                ["--mock-stage"] if bool(config.get("sidecar_mock_stage", False)) else []
            ),
        },
    )


def _resolve_isaacsim_sidecar_config_path(
    config: dict[str, Any],
    runtime_config_path: str | Path | None = None,
    *,
    direct_config: bool = False,
) -> tuple[Path, Path | None, dict[str, Any]]:
    """Return an Isaac Sim sidecar config path and loaded config."""
    root = _repo_root()
    if direct_config:
        loaded = dict(config)
        if runtime_config_path is not None:
            return Path(runtime_config_path).expanduser().resolve(), None, loaded
        temp_path = _write_temp_sidecar_config(loaded)
        return temp_path, temp_path, loaded
    configured = config.get("isaacsim_sidecar_config_path")
    config_path = (
        Path(str(configured)).expanduser().resolve()
        if configured not in (None, "")
        else root / "configs" / "isaacsim" / "real_scene.json"
    )
    overrides = config.get("isaacsim_sidecar_config", {})
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError("isaacsim_sidecar_config must be a JSON object when provided.")
    loaded = _merged_config_from_path(config_path, overrides)
    for src_key, dst_key in (
        ("isaacsim_host", "host"),
        ("isaacsim_port", "port"),
        ("isaacsim_timeout_s", "timeout_s"),
        ("isaacsim_mode", "mode"),
    ):
        if src_key in config:
            loaded[dst_key] = config[src_key]
    override_keys = (
        "isaacsim_host",
        "isaacsim_port",
        "isaacsim_timeout_s",
        "isaacsim_mode",
    )
    if overrides or any(key in config for key in override_keys):
        temp_path = _write_temp_sidecar_config(loaded)
        return temp_path, temp_path, loaded
    return config_path, None, loaded


def _start_isaacsim_sidecar(
    config: dict[str, Any],
    runtime_config_path: str | Path | None = None,
    *,
    direct_config: bool = False,
) -> IsaacSimTCPClientRuntime:
    """Start or reuse an Isaac Sim bridge sidecar for Geant4 companion motion."""
    config_path, temp_config_path, isaac_config = _resolve_isaacsim_sidecar_config_path(
        config,
        runtime_config_path=runtime_config_path,
        direct_config=direct_config,
    )
    host = str(isaac_config.get("host", "127.0.0.1"))
    port = int(isaac_config.get("port", 5555))
    timeout_s = float(isaac_config.get("timeout_s", 10.0))
    keep_alive = bool(
        isaac_config.get("keep_sidecar_alive", False)
        or config.get("isaacsim_keep_sidecar_alive", False)
    )
    if _tcp_server_available(host, port):
        if temp_config_path is not None:
            temp_config_path.unlink(missing_ok=True)
        return IsaacSimTCPClientRuntime(
            host=host,
            port=port,
            timeout_s=timeout_s,
            close_on_close=not keep_alive,
        )
    root = _repo_root()
    script_path = root / "scripts" / "run_isaacsim_bridge.py"
    default_log_path = root / "results" / "sidecars" / f"isaacsim_bridge_{port}.log"
    log_path = Path(
        str(config.get("isaacsim_sidecar_log_path", default_log_path))
    ).expanduser()
    if not log_path.is_absolute():
        log_path = (root / log_path).resolve()
    startup_timeout_s = float(config.get("isaacsim_sidecar_startup_timeout_s", 60.0))
    process_config = dict(isaac_config)
    isaac_python = config.get(
        "isaacsim_sidecar_python",
        isaac_config.get("sidecar_python"),
    )
    if isaac_python not in (None, ""):
        process_config["sidecar_python"] = str(isaac_python)
    isaac_python_env = config.get(
        "isaacsim_sidecar_python_env",
        isaac_config.get("sidecar_python_env"),
    )
    if isaac_python_env not in (None, ""):
        process_config["sidecar_python_env"] = str(isaac_python_env)
    try:
        process, log_handle = _start_sidecar_process(
            script_path=script_path,
            config_path=config_path,
            config=process_config,
            host=host,
            port=port,
            timeout_s=startup_timeout_s,
            log_path=log_path,
            sidecar_name="Isaac Sim",
            extra_args=(
                ["--mock"] if bool(config.get("isaacsim_sidecar_mock", False)) else None
            ),
        )
    except Exception:
        if temp_config_path is not None:
            temp_config_path.unlink(missing_ok=True)
        raise
    return ManagedIsaacSimTCPClientRuntime(
        host=host,
        port=port,
        timeout_s=timeout_s,
        process=process,
        log_handle=log_handle,
        temp_config_path=temp_config_path,
        close_on_close=not keep_alive,
    )


def _maybe_pair_geant4_with_isaacsim(
    config: dict[str, Any],
    geant4_runtime: SimulationRuntime,
) -> SimulationRuntime:
    """Wrap Geant4 with an Isaac Sim companion runtime when configured."""
    if not bool(config.get("start_isaacsim_sidecar_with_geant4", True)):
        return geant4_runtime
    try:
        isaacsim_runtime = _start_isaacsim_sidecar(config)
    except Exception:
        geant4_runtime.close()
        raise
    return Geant4WithIsaacSimRuntime(
        geant4_runtime=geant4_runtime,
        isaacsim_runtime=isaacsim_runtime,
    )


def create_simulation_runtime(
    backend: str,
    *,
    sources: list[PointSource],
    decomposer: SpectralDecomposer,
    mu_by_isotope: dict[str, object],
    shield_params: Any,
    runtime_config: dict[str, Any] | None = None,
    runtime_config_path: str | Path | None = None,
) -> SimulationRuntime:
    """Instantiate the requested simulation runtime."""
    config = {} if runtime_config is None else dict(runtime_config)
    normalized = backend.strip().lower()
    if normalized == "analytic":
        rng_seed = int(config.get("rng_seed", 123))
        return AnalyticSimulationRuntime(
            sources=sources,
            decomposer=decomposer,
            mu_by_isotope=mu_by_isotope,
            shield_params=shield_params,
            rng_seed=rng_seed,
            obstacle_height_m=float(config.get("obstacle_height_m", 2.0)),
            obstacle_material=str(config.get("obstacle_material", "concrete")),
            scatter_gain=float(config.get("scatter_gain", 0.03)),
            dead_time_s=float(config.get("dead_time_tau_s", config.get("dead_time_s", 0.0))),
            detector_model=(
                dict(config["detector_model"])
                if isinstance(config.get("detector_model"), dict)
                else None
            ),
        )
    if normalized == "isaacsim":
        host = str(config.get("host", "127.0.0.1"))
        port = int(config.get("port", 5555))
        timeout_s = float(config.get("timeout_s", 10.0))
        keep_alive = bool(config.get("keep_sidecar_alive", False))
        auto_start = bool(config.get("auto_start_sidecar", True))
        if auto_start and not _tcp_server_available(host, port):
            return _start_isaacsim_sidecar(
                config,
                runtime_config_path=runtime_config_path,
                direct_config=True,
            )
        return IsaacSimTCPClientRuntime(
            host=host,
            port=port,
            timeout_s=timeout_s,
            close_on_close=not keep_alive,
        )
    if normalized == "geant4":
        host = str(config.get("host", "127.0.0.1"))
        port = int(config.get("port", 5556))
        timeout_s = float(config.get("timeout_s", 120.0))
        auto_start = bool(config.get("auto_start_sidecar", True))
        if auto_start and not _tcp_server_available(host, port):
            geant4_runtime = _start_geant4_sidecar(
                config,
                host=host,
                port=port,
                runtime_config_path=runtime_config_path,
            )
        else:
            geant4_runtime = Geant4TCPClientRuntime(host=host, port=port, timeout_s=timeout_s)
        return _maybe_pair_geant4_with_isaacsim(config, geant4_runtime)
    raise ValueError(f"Unknown simulation backend: {backend}")
