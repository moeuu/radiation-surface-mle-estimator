"""Forward-response conformance tests for the MLE provider."""

from pathlib import Path

import numpy as np

from three_d_estimation.conformance import (
    compute_forward_conformance,
    load_forward_conformance_axes,
    save_forward_conformance,
)


ROOT = Path(__file__).resolve().parents[2]
AXES = ROOT / "fixtures/forward_response_conformance.json"


def test_mle_forward_conformance_is_complete_and_deterministic(tmp_path) -> None:
    """Exercise every declared MLE forward-response conformance case."""
    axes = load_forward_conformance_axes(AXES)
    result = compute_forward_conformance(axes)

    assert result.case_ids.shape == (3 * 3 * 8 * 8 * 4 * 2,)
    assert result.unit_response.shape == result.case_ids.shape
    assert np.all(np.isfinite(result.unit_response))
    assert np.all(result.unit_response >= 0.0)
    first = save_forward_conformance(tmp_path / "first.npz", result)
    second = save_forward_conformance(tmp_path / "second.npz", result)
    assert first.read_bytes() == second.read_bytes()
