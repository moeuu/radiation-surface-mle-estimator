"""Generate the standalone MLE unit-strength forward-response conformance NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from three_d_estimation.conformance import (
    compute_forward_conformance,
    save_forward_conformance,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AXES = ROOT / "fixtures" / "forward_response_conformance.json"


def main(argv: Sequence[str] | None = None) -> int:
    """Parse axes/output paths, compute local physics, and save exact ordering."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axes", type=Path, default=DEFAULT_AXES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(None if argv is None else list(argv))
    result = compute_forward_conformance(args.axes)
    output = save_forward_conformance(
        args.output,
        result,
        overwrite=bool(args.overwrite),
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
