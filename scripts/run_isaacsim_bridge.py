"""Run the Isaac Sim bridge sidecar."""
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

from sim.runtime import load_runtime_config
from sim.isaacsim_app.bridge_server import BridgeServerConfig, serve_forever


def main() -> None:
    """Parse CLI arguments and start the bridge server."""
    parser = argparse.ArgumentParser(description="Run the Isaac Sim bridge sidecar.")
    parser.add_argument(
        "--config",
        type=str,
        default=(ROOT / "configs" / "isaacsim" / "default_scene.json").as_posix(),
        help="Path to the bridge configuration JSON file.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force mock mode even when Isaac Sim is installed.",
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
    server_config = BridgeServerConfig(
        host=str(config.get("host", "127.0.0.1")),
        port=int(config.get("port", 5555)),
        use_mock=bool(args.mock or config.get("mode", "mock") == "mock"),
        app_config=config,
    )
    serve_forever(server_config)


if __name__ == "__main__":
    main()
