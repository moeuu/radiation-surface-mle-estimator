"""Content-binding tests for file-backed replay model assets."""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime.forward_model_manifest import (
    build_forward_model_manifest,
    validate_forward_model_manifest,
)
from runtime.records import canonical_json_sha256


@pytest.mark.parametrize(
    ("component", "runtime_path_field", "obstacle_path"),
    (
        ("obstacle", None, "assets/model.json"),
        ("transport", "pf_transport_response_model_path", None),
        ("detector", "detector_response_model_path", None),
        ("spectrum", "spectrum_calibration_path", None),
    ),
)
def test_same_asset_path_with_different_bytes_fails_component_validation(
    tmp_path: Path,
    component: str,
    runtime_path_field: str | None,
    obstacle_path: str | None,
) -> None:
    """A stable relative filename cannot disguise changed replay physics bytes."""
    relative_path = "assets/model.json"
    asset = tmp_path / relative_path
    asset.parent.mkdir()
    asset.write_bytes(b'{"generation":1}\n')
    runtime_config: dict[str, object] = {
        "source_rate_model": "detector_cps_1m",
    }
    if runtime_path_field is not None:
        runtime_config[runtime_path_field] = relative_path
    resolved_config_sha256 = canonical_json_sha256(runtime_config)
    manifest = build_forward_model_manifest(
        runtime_config=runtime_config,
        environment={"size_x": 2.0, "size_y": 2.0, "size_z": 2.0},
        obstacle_layout_path=obstacle_path,
        isotopes=("Cs-137",),
        repository_commit="asset-binding-test",
        resolved_config_sha256=resolved_config_sha256,
        source_rate_model="detector_cps_1m",
        repository_root=tmp_path,
    )

    asset.write_bytes(b'{"generation":2}\n')

    with pytest.raises(ValueError, match=f"compatibility error for {component}"):
        validate_forward_model_manifest(
            manifest,
            runtime_config=runtime_config,
            environment={"size_x": 2.0, "size_y": 2.0, "size_z": 2.0},
            obstacle_layout_path=obstacle_path,
            isotopes=("Cs-137",),
            repository_commit="asset-binding-test",
            resolved_config_sha256=resolved_config_sha256,
            source_rate_model="detector_cps_1m",
            repository_root=tmp_path,
        )
