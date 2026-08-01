"""Verify that the MLE repository contains estimator-specific code only."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PATHS = (
    "native",
    "obstacle_layouts",
    "source_layouts",
    "src/measurement",
    "src/pf",
    "src/planning",
    "src/runtime",
    "src/sim",
    "src/spectrum",
    "src/realtime_demo.py",
)


def _check_forbidden_paths() -> list[str]:
    """Report copied simulator, PF, planner, or log-owner paths."""
    return [
        f"forbidden duplicated implementation: {relative}"
        for relative in FORBIDDEN_PATHS
        if (ROOT / relative).exists()
    ]


def _check_package_boundary() -> list[str]:
    """Require only the MLE package in wheel discovery."""
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    included = payload["tool"]["setuptools"]["packages"]["find"]["include"]
    if included != ["three_d_estimation*"]:
        return [f"unexpected package discovery include: {included!r}"]
    dependencies = payload["project"]["dependencies"]
    if "rotating-shield-simulation-runtime" not in dependencies:
        return ["shared simulation runtime dependency is missing"]
    return []


def _check_python_syntax() -> list[str]:
    """Parse all retained Python files without importing optional GPU code."""
    errors: list[str] = []
    for root_name in ("src/three_d_estimation", "scripts"):
        root = ROOT / root_name
        for path in root.rglob("*.py"):
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError as exc:
                errors.append(f"syntax error in {path.relative_to(ROOT)}: {exc}")
    return errors


def run_checks() -> list[str]:
    """Return every repository-boundary error."""
    return [
        *_check_forbidden_paths(),
        *_check_package_boundary(),
        *_check_python_syntax(),
    ]


def main(argv: list[str] | None = None) -> int:
    """Run the MLE repository-boundary audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args(argv)
    errors = run_checks()
    payload = {"boundary_ok": not errors, "errors": errors}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif errors:
        for error in errors:
            print(f"- {error}", file=sys.stderr)
    else:
        print("MLE repository boundary passed.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
