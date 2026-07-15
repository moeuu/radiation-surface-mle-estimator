"""Build the native Geant4 sidecar executable."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _split_flags(raw_flags: str) -> list[str]:
    """Split shell-style compiler flags into a list."""
    return shlex.split(raw_flags.strip()) if raw_flags.strip() else []


def main() -> None:
    """Resolve Geant4 flags and compile the native sidecar executable."""
    parser = argparse.ArgumentParser(description="Build the native Geant4 sidecar executable.")
    parser.add_argument(
        "--source",
        type=str,
        default=(ROOT / "native" / "geant4_sidecar" / "geant4_sidecar.cpp").as_posix(),
        help="Path to the C++ source file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=(ROOT / "build" / "geant4_sidecar").as_posix(),
        help="Output executable path.",
    )
    parser.add_argument(
        "--std",
        type=str,
        default="c++17",
        help="C++ language standard to use for the build.",
    )
    args = parser.parse_args()

    geant4_config = subprocess.run(
        ["geant4-config", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if geant4_config.returncode != 0:
        raise SystemExit(
            "geant4-config was not found. Install Geant4 and ensure geant4-config is on PATH before building."
        )
    cflags = subprocess.run(
        ["geant4-config", "--cflags"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    libs = subprocess.run(
        ["geant4-config", "--libs"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    source_path = Path(args.source).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "g++",
        f"-std={args.std}",
        "-O2",
        "-o",
        output_path.as_posix(),
        source_path.as_posix(),
        *_split_flags(cflags),
        *_split_flags(libs),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    if result.stdout:
        sys.stdout.write(result.stdout)
    print(f"Built Geant4 sidecar: {output_path}")


if __name__ == "__main__":
    main()
