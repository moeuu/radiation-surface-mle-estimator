"""Tests for unseen-environment RA-L holdout separation."""

from __future__ import annotations

from pathlib import Path

import pytest
from runtime.measurement_log import MeasurementLog

from three_d_estimation.config import MLEConfig
from three_d_estimation.holdout import validate_holdout_separation


def _log(run_id: str, environment_id: str, size_x: float) -> MeasurementLog:
    """Return an in-memory log identity sufficient for separation validation."""
    return MeasurementLog(
        run_manifest={
            "run_id": run_id,
            "metadata": {"environment_realization_id": environment_id},
        },
        runtime_config={},
        environment={"size_x": size_x, "size_y": 2.0, "size_z": 2.0},
        forward_model_manifest={},
        records=(),
        path=Path(f"/{run_id}"),
    )


def test_holdout_requires_distinct_run_and_environment_manifests() -> None:
    """A final holdout must be disjoint from regularization tuning."""
    config = MLEConfig(
        mode="spectral",
        tuning_environment_id="tune-env",
        final_holdout_environment_id="test-env",
    )

    identities = validate_holdout_separation(
        _log("tune", "tune-env", 2.0),
        _log("holdout", "test-env", 3.0),
        config,
    )

    assert identities == ("tune-env", "test-env")


def test_holdout_rejects_cv_on_final_environment() -> None:
    """Regularization cannot be selected using the final evaluation rows."""
    config = MLEConfig(
        mode="spectral",
        regularization_selection="grouped_cv",
    )

    with pytest.raises(ValueError, match="cannot tune"):
        validate_holdout_separation(
            _log("tune", "tune-env", 2.0),
            _log("holdout", "test-env", 3.0),
            config,
        )
