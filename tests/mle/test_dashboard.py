"""Tests for browser-reachable CUI dashboard URL publication."""

from __future__ import annotations

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
