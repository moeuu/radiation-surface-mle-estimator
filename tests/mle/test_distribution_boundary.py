"""Distribution-boundary checks for the standalone MLE wheel configuration."""

from __future__ import annotations

from pathlib import Path
import tomllib

from setuptools.discovery import PackageFinder


ROOT = Path(__file__).resolve().parents[2]


def test_package_discovery_excludes_pf_and_realtime_demo() -> None:
    """The configured distribution contains MLE physics/runtime, never PF code."""
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    setuptools_config = configuration["tool"]["setuptools"]
    find_config = setuptools_config["packages"]["find"]
    packages = set(
        PackageFinder.find(
            str(ROOT / find_config["where"][0]),
            include=tuple(find_config["include"]),
            exclude=tuple(find_config["exclude"]),
        )
    )

    assert not any(name == "pf" or name.startswith("pf.") for name in packages)
    assert not any(
        name == "planning" or name.startswith("planning.") for name in packages
    )
    assert "realtime_demo" not in setuptools_config.get("py-modules", ())
    assert {
        "counts",
        "measurement",
        "runtime",
        "sim",
        "spectrum",
        "three_d_estimation",
    }.issubset(packages)
