"""Tests for browser-reachable CUI dashboard URL publication."""

from __future__ import annotations

from pathlib import Path

import pytest

from three_d_estimation import dashboard
from three_d_estimation.cli import _print_cui_dashboard_url


def test_dashboard_url_has_explicit_index_and_ipv6_brackets() -> None:
    """CUI URLs must be directly clickable for IPv4, names, and IPv6 hosts."""
    assert (
        dashboard._dashboard_browser_url("100.127.159.83", 8878)
        == "http://100.127.159.83:8878/index.html"
    )
    assert (
        dashboard._dashboard_browser_url("fd7a:115c:a1e0::1", 8878)
        == "http://[fd7a:115c:a1e0::1]:8878/index.html"
    )
    with pytest.raises(ValueError, match="host name or IP"):
        dashboard._dashboard_browser_url("https://example.test", 8878)


def test_dashboard_selects_next_port_instead_of_reusing_stale_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An occupied prior-run port must not silently serve stale dashboard files."""
    monkeypatch.setattr(
        dashboard,
        "_tcp_port_is_open",
        lambda _host, port: int(port) == 8878,
    )

    assert dashboard._available_dashboard_port("0.0.0.0", 8878) == 8879


def test_cui_url_is_visible_without_corrupting_json_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Human output uses stdout while JSON mode relays the immediate URL on stderr."""
    url = "http://100.127.159.83:8878/index.html"
    _print_cui_dashboard_url(url, json_output=False)
    human = capsys.readouterr()
    assert human.out == f"CUI dashboard URL: {url}\n"
    assert human.err == ""

    _print_cui_dashboard_url(url, json_output=True)
    structured = capsys.readouterr()
    assert structured.out == ""
    assert structured.err == f"CUI dashboard URL: {url}\n"


def test_dashboard_publishes_pf_style_scientific_images(tmp_path: Path) -> None:
    """The browser work surface must be the same PNG-first form as the PF CUI."""
    publisher = dashboard.OnlineMLEDashboard(
        tmp_path,
        environment={"size_x": 10.0, "size_y": 20.0, "size_z": 10.0},
    )
    publisher.publish(
        None,
        {
            "status": "starting",
            "run_id": "test-run",
            "mode": "spectral",
            "isotopes": ["Co-60", "Cs-137", "Eu-154"],
            "record_count": 0,
            "latest_step_id": None,
            "latest_station_id": None,
        },
    )

    expected = (
        dashboard.OVERVIEW_IMAGE_FILENAME,
        dashboard.ROBOT_IMAGE_FILENAME,
        dashboard.MLE_IMAGE_FILENAME,
        dashboard.SPECTRUM_IMAGE_FILENAME,
    )
    html = (tmp_path / dashboard.DASHBOARD_INDEX_FILENAME).read_text(
        encoding="utf-8"
    )
    for filename in expected:
        payload = (tmp_path / filename).read_bytes()
        assert payload.startswith(b"\x89PNG\r\n\x1a\n")
        assert filename in html
