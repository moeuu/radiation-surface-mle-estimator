"""Optional piplup-notify event client for simulation runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import getpass
import json
import os
import socket
import uuid
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PIPLUP_DEFAULT_BASE_URL = "https://piplup-notify-production.up.railway.app"
PIPLUP_EVENT_SOURCE = "rotating-shield-pf"
_FALSE_VALUES = {"0", "false", "no", "off", "disable", "disabled"}
_TRUE_VALUES = {"1", "true", "yes", "on", "enable", "enabled"}


@dataclass(frozen=True)
class PiplupNotificationConfig:
    """Configuration for optional piplup-notify delivery."""

    enabled: bool = False
    base_url: str = PIPLUP_DEFAULT_BASE_URL
    token: str | None = None
    account: str | None = None
    run_id: str = field(default_factory=lambda: new_run_id())
    timeout_s: float = 3.0
    source: str = PIPLUP_EVENT_SOURCE
    fail_silently: bool = True

    @classmethod
    def from_env(
        cls,
        *,
        enabled: bool | None = None,
        base_url: str | None = None,
        token: str | None = None,
        account: str | None = None,
        run_id: str | None = None,
        timeout_s: float | None = None,
    ) -> "PiplupNotificationConfig":
        """Build configuration from explicit values and environment variables."""
        env_enabled = _parse_env_bool(os.environ.get("PIPLUP_NOTIFY_ENABLED"))
        notify_enabled = env_enabled if enabled is None else bool(enabled)
        resolved_token = _first_non_empty(
            token,
            os.environ.get("PIPLUP_NOTIFY_TOKEN"),
            os.environ.get("EVENT_API_TOKEN"),
        )
        resolved_url = _first_non_empty(
            base_url,
            os.environ.get("PIPLUP_NOTIFY_URL"),
            os.environ.get("PIPLUP_NOTIFY_BASE_URL"),
            PIPLUP_DEFAULT_BASE_URL,
        )
        resolved_account = _first_non_empty(
            account,
            os.environ.get("PIPLUP_NOTIFY_ACCOUNT"),
            default_account(),
        )
        resolved_run_id = _first_non_empty(
            run_id,
            os.environ.get("PIPLUP_NOTIFY_RUN_ID"),
            new_run_id(),
        )
        resolved_timeout = timeout_s
        if resolved_timeout is None:
            resolved_timeout = _float_env("PIPLUP_NOTIFY_TIMEOUT_S", 3.0)
        return cls(
            enabled=notify_enabled,
            base_url=resolved_url,
            token=resolved_token,
            account=resolved_account,
            run_id=resolved_run_id,
            timeout_s=float(resolved_timeout),
        )


class PiplupNotifier:
    """Small no-op-safe client for normalized piplup event ingest."""

    def __init__(self, config: PiplupNotificationConfig | None) -> None:
        """Initialize the notifier with an optional resolved configuration."""
        self.config = config or PiplupNotificationConfig()
        self._warned_disabled = False

    @property
    def active(self) -> bool:
        """Return True when notifications should be sent."""
        return bool(self.config.enabled and self.config.token)

    def notify_started(self, payload: dict[str, Any]) -> bool:
        """Send a simulation started event."""
        return self.send_event(
            "simulation.started",
            {
                "status": "running",
                "message": "Rotating-shield PF simulation started.",
                **payload,
            },
            severity="info",
            dedupe_suffix="started",
        )

    def notify_finished(self, payload: dict[str, Any]) -> bool:
        """Send a simulation finished event."""
        return self.send_event(
            "simulation.finished",
            {
                "status": "succeeded",
                "message": "Rotating-shield PF simulation finished.",
                **payload,
            },
            severity="info",
            dedupe_suffix="finished",
        )

    def notify_failed(self, payload: dict[str, Any]) -> bool:
        """Send a simulation failed event."""
        return self.send_event(
            "simulation.failed",
            {
                "status": "failed",
                "message": "Rotating-shield PF simulation failed.",
                **payload,
            },
            severity="error",
            dedupe_suffix="failed",
        )

    def notify_spectrum(self, step_index: int, payload: dict[str, Any]) -> bool:
        """Send a per-measurement spectrum event."""
        return self.send_event(
            "simulation.spectrum",
            {
                "status": "running",
                "message": "Rotating-shield PF spectrum measurement.",
                "step_index": int(step_index),
                **payload,
            },
            severity="info",
            dedupe_suffix=f"spectrum-{int(step_index):04d}",
        )

    def send_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        severity: str = "info",
        dedupe_suffix: str | None = None,
    ) -> bool:
        """POST one normalized event. Returns False when disabled or delivery fails."""
        if not self.config.enabled:
            return False
        if not self.config.token:
            self._warn_once("Piplup notifications disabled: token is not configured.")
            return False

        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        safe_type = event_type.replace(".", "-")
        dedupe_part = dedupe_suffix or safe_type
        event = {
            "id": f"{self.config.source}-{self.config.run_id}-{dedupe_part}",
            "source": self.config.source,
            "type": event_type,
            "occurred_at": now,
            "severity": severity,
            "dedupe_key": f"{self.config.source}:{self.config.run_id}:{dedupe_part}",
            "payload": {
                "run_id": self.config.run_id,
                "host": socket.gethostname(),
                "user": getpass.getuser(),
                **_json_safe(payload),
            },
        }
        if self.config.account:
            event["account"] = self.config.account

        body = json.dumps(event, separators=(",", ":"), sort_keys=True).encode("utf-8")
        endpoint = self.config.base_url.rstrip("/") + "/api/events"
        request = Request(
            endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.config.token}",
                "Content-Type": "application/json",
                "User-Agent": "rotating-shield-particle-filter/notify",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=max(float(self.config.timeout_s), 0.1)) as response:
                return 200 <= int(response.status) < 300
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            if self.config.fail_silently:
                self._warn_once(f"Piplup notification failed: {exc}")
                return False
            raise

    def _warn_once(self, message: str) -> None:
        """Print at most one warning for optional notification failures."""
        if self._warned_disabled:
            return
        print(message)
        self._warned_disabled = True


def new_run_id() -> str:
    """Return a compact run id suitable for event ids and dedupe keys."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def default_account() -> str:
    """Return a stable default account label for piplup events."""
    return f"{getpass.getuser()}@{socket.gethostname()}"


def _parse_env_bool(value: str | None) -> bool:
    """Parse an opt-in boolean environment variable."""
    if value is None:
        return False
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return False


def _float_env(name: str, default: float) -> float:
    """Parse a float from the environment with a safe fallback."""
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _first_non_empty(*values: str | None) -> str | None:
    """Return the first non-empty string from values."""
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _json_safe(value: Any) -> Any:
    """Convert common numeric/container objects into JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)
