"""Verify that this repository has no build or runtime sibling dependency."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
IGNORED_TREE_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "results",
}


def _forbidden_reference() -> str:
    """Return the sibling-repository token without embedding it in source."""
    return "../" + "Rotating-shield-" + "particle-filter"


def _source_files() -> list[Path]:
    """Return project files whose behavior may create an external dependency."""
    roots = [ROOT / "src", ROOT / "scripts", ROOT / "tests"]
    files = [ROOT / "pyproject.toml"]
    for root in roots:
        if root.exists():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file()
                and path.suffix in {".py", ".toml", ".yaml", ".yml"}
                and path.name != Path(__file__).name
            )
    return sorted(set(files))


def _is_generated_or_environment_path(path: Path) -> bool:
    """Return whether a path belongs to local tooling/output rather than the repo."""
    relative = path.relative_to(ROOT)
    return any(
        part in IGNORED_TREE_NAMES or part.endswith(".egg-info")
        for part in relative.parts
    )


def _check_text_references() -> list[str]:
    """Find direct sibling, path-dependency, and runtime-sync references."""
    errors: list[str] = []
    forbidden = _forbidden_reference()
    for path in _source_files():
        text = path.read_text(encoding="utf-8")
        if forbidden in text:
            errors.append(f"forbidden sibling reference: {path.relative_to(ROOT)}")
        if "sync_pf_upstream" in text or "check_pf_upstream_sync" in text:
            errors.append(f"forbidden runtime sync hook: {path.relative_to(ROOT)}")
    return errors


def _check_dependencies() -> list[str]:
    """Reject Git, URL, and local-path project dependencies."""
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    errors: list[str] = []

    def _dependency_strings(value: object, path: str) -> list[tuple[str, str]]:
        """Collect dependency-like strings from supported pyproject sections."""
        found: list[tuple[str, str]] = []
        if isinstance(value, str):
            found.append((path, value))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                found.extend(_dependency_strings(item, f"{path}[{index}]"))
        elif isinstance(value, dict):
            for key, item in value.items():
                found.extend(_dependency_strings(item, f"{path}.{key}"))
        return found

    sections = {
        "build-system.requires": payload.get("build-system", {}).get("requires", []),
        "project.dependencies": payload.get("project", {}).get("dependencies", []),
        "project.optional-dependencies": payload.get("project", {}).get(
            "optional-dependencies",
            {},
        ),
        "dependency-groups": payload.get("dependency-groups", {}),
        "tool.uv.sources": payload.get("tool", {}).get("uv", {}).get("sources", {}),
    }
    for section, section_value in sections.items():
        for location, dependency in _dependency_strings(section_value, section):
            value = dependency.lower()
            if (
                " @ " in value
                or "git+" in value
                or "../" in value
                or "file:" in value
                or value.startswith(("http://", "https://"))
            ):
                errors.append(
                    f"non-standalone dependency at {location}: {dependency}"
                )
    return errors


def _check_symlinks() -> list[str]:
    """Reject symbolic links whose resolved target leaves the repository."""
    errors: list[str] = []
    root_resolved = ROOT.resolve()
    for path in ROOT.rglob("*"):
        if _is_generated_or_environment_path(path):
            continue
        if not path.is_symlink():
            continue
        try:
            path.resolve(strict=True).relative_to(root_resolved)
        except (FileNotFoundError, ValueError):
            errors.append(f"external or broken symlink: {path.relative_to(ROOT)}")
    return errors


def _check_submodules() -> list[str]:
    """Reject Git submodules because they are external runtime/build inputs."""
    gitmodules = ROOT / ".gitmodules"
    if gitmodules.exists() and gitmodules.read_text(encoding="utf-8").strip():
        return ["Git submodules are forbidden in the standalone repository"]
    return []


def _check_required_snapshot() -> list[str]:
    """Verify locally vendored runtime code and assets are present."""
    required = (
        "COMMON_RUNTIME_SNAPSHOT.json",
        "UPSTREAM_PF_COMMIT",
        "main.py",
        "native/geant4_sidecar/geant4_sidecar.cpp",
        "configs/geant4/variance_reduction_external_no_isaac_32threads.json",
        "configs/python/high_fidelity_no_isaac.json",
        "src/measurement/continuous_kernels.py",
        "src/measurement/observation_model.py",
        "src/sim/protocol.py",
        "src/spectrum/pipeline.py",
        "src/three_d_estimation/cli.py",
        "src/runtime/measurement_log.py",
    )
    return [f"missing standalone artifact: {name}" for name in required if not (ROOT / name).is_file()]


def _check_python_import_syntax() -> list[str]:
    """Parse every local Python source without importing optional simulators."""
    errors: list[str] = []
    for path in _source_files():
        if path.suffix != ".py":
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            errors.append(f"syntax error in {path.relative_to(ROOT)}: {exc}")
    return errors


def run_checks() -> list[str]:
    """Run all standalone-repository checks and return error messages."""
    errors: list[str] = []
    errors.extend(_check_text_references())
    errors.extend(_check_dependencies())
    errors.extend(_check_symlinks())
    errors.extend(_check_submodules())
    errors.extend(_check_required_snapshot())
    errors.extend(_check_python_import_syntax())
    return errors


def main(argv: list[str] | None = None) -> int:
    """Run the standalone audit and print a machine-readable summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args(argv)
    errors = run_checks()
    payload = {"standalone": not errors, "errors": errors}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif errors:
        print("Standalone audit failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
    else:
        print("Standalone audit passed.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
