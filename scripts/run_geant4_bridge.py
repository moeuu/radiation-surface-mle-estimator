"""Run the Geant4 bridge sidecar."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.geant4_app.bridge_server import Geant4BridgeServerConfig, serve_forever
from sim.runtime import load_runtime_config


def main() -> None:
    """Parse CLI arguments and start the Geant4 bridge server."""
    parser = argparse.ArgumentParser(description="Run the Geant4 bridge sidecar.")
    parser.add_argument(
        "--config",
        type=str,
        default=(
            ROOT
            / "configs"
            / "geant4"
            / "variance_reduction_external_no_isaac_32threads.json"
        ).as_posix(),
        help="Path to the bridge configuration JSON file.",
    )
    parser.add_argument(
        "--mock-stage",
        action="store_true",
        help="Force the sidecar to use the fake stage backend.",
    )
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_runtime_config(config_path)
    usd_path = config.get("usd_path")
    if usd_path:
        resolved_usd_path = Path(str(usd_path)).expanduser()
        if not resolved_usd_path.is_absolute():
            resolved_usd_path = (config_path.parent / resolved_usd_path).resolve()
        config["usd_path"] = resolved_usd_path.as_posix()
    if args.mock_stage:
        config["use_mock_stage"] = True
    server_config = Geant4BridgeServerConfig(
        host=str(config.get("host", "127.0.0.1")),
        port=int(config.get("port", 5556)),
        app_config=config,
    )
    serve_forever(server_config)


if __name__ == "__main__":
    main()
