"""Configure import paths for local source and script modules during tests."""

import sys
from pathlib import Path


def pytest_configure() -> None:
    """Add the repository root and src directory to the import path."""
    root = Path(__file__).resolve().parents[1]
    src_path = root / "src"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
