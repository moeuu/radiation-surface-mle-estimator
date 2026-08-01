"""Distribution-boundary checks for the estimator-only MLE wheel."""

from __future__ import annotations

from pathlib import Path
import tomllib

from setuptools.discovery import PackageFinder


ROOT = Path(__file__).resolve().parents[2]


def test_package_discovery_contains_only_mle_code() -> None:
    """The wheel must not vendor shared simulation or another estimator."""
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    setuptools_config = configuration["tool"]["setuptools"]
    find_config = setuptools_config["packages"]["find"]
    packages = set(
        PackageFinder.find(
            str(ROOT / find_config["where"][0]),
            include=tuple(find_config["include"]),
            exclude=tuple(find_config.get("exclude", ())),
        )
    )

    assert not any(name == "pf" or name.startswith("pf.") for name in packages)
    assert not any(
        name == "planning" or name.startswith("planning.") for name in packages
    )
    assert "realtime_demo" not in setuptools_config.get("py-modules", ())
    assert packages
    assert all(
        name == "three_d_estimation" or name.startswith("three_d_estimation.")
        for name in packages
    )


def test_source_distribution_contains_only_mle_assets() -> None:
    """The sdist must include only estimator configuration and fixtures."""
    directives = {
        line.strip()
        for line in (ROOT / "MANIFEST.in").read_text("utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "recursive-include configs/mle *.json" in directives
    assert "prune tests" in directives
    assert not (ROOT / "src/measurement").exists()
    assert not (ROOT / "src/sim").exists()
    assert not (ROOT / "src/spectrum").exists()
