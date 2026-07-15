"""TCP bridge server for the Geant4 sidecar process."""

from __future__ import annotations

from dataclasses import dataclass, field
import socketserver
import threading
from typing import Any

from sim.geant4_app.app import Geant4Application
from sim.isaacsim_app.scene_builder import build_scene_description
from sim.protocol import SimulationCommand, decode_message, encode_message


@dataclass
class Geant4BridgeServerConfig:
    """Define host and port for the Geant4 bridge server."""

    host: str = "127.0.0.1"
    port: int = 5556
    app_config: dict[str, Any] = field(default_factory=dict)


class _BridgeTCPServer(socketserver.TCPServer):
    """Carry shared application state for serialized bridge requests."""

    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[socketserver.BaseRequestHandler],
        app: Geant4Application,
    ) -> None:
        """Bind the server and retain the application handle."""
        super().__init__(server_address, handler_class)
        self.app = app


class _BridgeRequestHandler(socketserver.BaseRequestHandler):
    """Handle a single bridge request."""

    def handle(self) -> None:
        """Decode a request, dispatch it, and write the response."""
        raw = b""
        while True:
            chunk = self.request.recv(65536)
            if not chunk:
                break
            raw += chunk
        try:
            msg_type, payload = decode_message(raw.strip())
            response = self._dispatch(msg_type, payload)
            self.request.sendall(encode_message("ok", response))
        except Exception as exc:  # pragma: no cover - network failure path
            self.request.sendall(encode_message("error", {"message": str(exc)}))

    def _dispatch(self, msg_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a decoded request to the application."""
        server = self.server
        if not isinstance(server, _BridgeTCPServer):
            raise RuntimeError("Unexpected server type.")
        if msg_type == "reset":
            server.app.reset(build_scene_description(payload))
            return {"status": "reset"}
        if msg_type == "step":
            command = SimulationCommand.from_dict(payload)
            observation = server.app.step(command)
            return {"observation": observation.to_dict()}
        if msg_type == "shutdown":
            threading.Thread(target=server.shutdown, daemon=True).start()
            return {"status": "shutdown"}
        raise ValueError(f"Unsupported message type: {msg_type}")


def serve_forever(config: Geant4BridgeServerConfig) -> None:
    """Run the Geant4 TCP bridge until shutdown is requested."""
    app = Geant4Application(app_config=config.app_config)
    with _BridgeTCPServer((config.host, config.port), _BridgeRequestHandler, app) as server:
        try:
            server.serve_forever()
        finally:
            app.close()
