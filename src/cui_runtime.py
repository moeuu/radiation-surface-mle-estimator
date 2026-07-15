"""CUI split-view runtime helpers for long-running simulations."""

from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Mapping

from runtime_defaults import DEFAULT_CUI_SPLIT_VIEW_DIR

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CUI_VIEW_DIR = ROOT / DEFAULT_CUI_SPLIT_VIEW_DIR

_CUI_HTTP_SERVER: ThreadingHTTPServer | None = None
_CUI_HTTP_THREAD: threading.Thread | None = None


def resolve_cui_split_view_enabled(
    runtime_config: Mapping[str, object],
    *,
    save_outputs: bool,
) -> bool:
    """Return whether the URL-served CUI progress view should run."""
    if "cui_split_view" in runtime_config:
        return bool(runtime_config["cui_split_view"])
    return bool(save_outputs)


def _cui_view_relative_url(output_dir: Path, static_root: Path) -> str:
    """Return the CUI split-view URL path relative to the static root."""
    try:
        relative = output_dir.resolve().relative_to(static_root.resolve())
    except ValueError:
        return "index.html"
    if str(relative) == ".":
        return "index.html"
    return f"{relative.as_posix()}/index.html"


def _tcp_port_is_open(host: str, port: int) -> bool:
    """Return whether a TCP port already accepts local connections."""
    connect_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    try:
        with socket.create_connection((connect_host, int(port)), timeout=0.2):
            return True
    except OSError:
        return False


def _default_cui_public_host() -> str:
    """Return a likely browser-reachable host for local CUI visualization."""
    env_host = os.environ.get("CUI_SPLIT_VIEW_PUBLIC_HOST")
    if env_host:
        return env_host
    try:
        raw_hosts = subprocess.check_output(
            ["hostname", "-I"],
            text=True,
            timeout=0.2,
        )
        hosts = [host.strip() for host in raw_hosts.split() if host.strip()]
        for candidate in hosts:
            if candidate.startswith("100."):
                return candidate
        for candidate in hosts:
            if candidate and not candidate.startswith(("127.", "172.")):
                return candidate
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            candidate = str(info[4][0])
            if candidate.startswith("100."):
                return candidate
    except OSError:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(0.2)
            sock.connect(("8.8.8.8", 80))
            host = str(sock.getsockname()[0])
            if host and not host.startswith("127."):
                return host
    except OSError:
        pass
    return "127.0.0.1"


def ensure_cui_view_server(
    output_dir: Path,
    *,
    host: str = "0.0.0.0",
    port: int = 8877,
    public_host: str | None = None,
    static_root: Path = DEFAULT_CUI_VIEW_DIR,
) -> str:
    """Start or reuse an HTTP server for CUI split visualization."""
    global _CUI_HTTP_SERVER, _CUI_HTTP_THREAD
    static_root.mkdir(parents=True, exist_ok=True)
    port = int(port)
    relative_url = _cui_view_relative_url(output_dir, static_root)
    display_host = (
        _default_cui_public_host()
        if public_host is None and host in {"0.0.0.0", "::"}
        else str(public_host or host)
    )
    url = f"http://{display_host}:{port}/{relative_url}"
    if _CUI_HTTP_SERVER is not None:
        return url
    if _tcp_port_is_open(host, port):
        return url
    log_path = static_root / f"http_server_{port}.log"
    pid_path = static_root / f"http_server_{port}.pid"
    try:
        log_handle = log_path.open("ab")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "http.server",
                str(port),
                "--bind",
                host,
                "--directory",
                static_root.as_posix(),
            ],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_handle.close()
        pid_path.write_text(f"{int(process.pid)}\n", encoding="utf-8")
        for _ in range(20):
            if _tcp_port_is_open(host, port):
                return url
            if process.poll() is not None:
                break
            time.sleep(0.05)
    except OSError as exc:
        print(f"Persistent CUI split visualization server failed: {exc}")
    handler = partial(SimpleHTTPRequestHandler, directory=static_root.as_posix())
    try:
        server = ThreadingHTTPServer((host, port), handler)
    except OSError as exc:
        print(f"CUI split visualization URL unavailable on {host}:{port}: {exc}")
        return output_dir.as_posix()
    thread = threading.Thread(
        target=server.serve_forever,
        name="cui-view-http-server",
        daemon=True,
    )
    thread.start()
    _CUI_HTTP_SERVER = server
    _CUI_HTTP_THREAD = thread
    return url
