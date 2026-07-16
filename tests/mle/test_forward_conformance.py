"""Forward-response conformance tests for the standalone MLE provider."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from runtime.forward_model_manifest import (
    CANONICAL_UNITS,
    CONFORMANCE_FORWARD_MODEL_ID,
    CONFORMANCE_MODEL_IDENTIFIERS,
    RESPONSE_SEMANTICS,
    SOURCE_RATE_SEMANTICS,
    registered_conformance_line_mu_by_isotope,
    validate_forward_model_manifest,
)
from runtime.measurement_log import load_measurement_log
from runtime.records import canonical_json_sha256
from measurement.observation_model import build_runtime_observation_model
from three_d_estimation.conformance import (
    compute_forward_conformance,
    load_forward_conformance_axes,
    save_forward_conformance,
)


ROOT = Path(__file__).resolve().parents[2]
AXES = ROOT / "fixtures" / "forward_response_conformance.json"
MEASUREMENT_FIXTURE = (
    ROOT / "fixtures" / "shared_measurement_log_v1" / "measurement_log"
)


def test_canonical_forward_conformance_is_complete_ordered_and_deterministic(
    tmp_path: Path,
) -> None:
    """Exercise all isotope, pose, shield, source, and obstacle combinations."""
    axes = load_forward_conformance_axes(AXES)
    result = compute_forward_conformance(axes)
    assert result.case_ids.shape == (3 * 3 * 8 * 8 * 4 * 2,)
    assert result.unit_response.shape == result.case_ids.shape
    assert result.case_ids[0] == (
        "Cs-137|pose=low-corner|fe=00|pb=00|source=floor|obstacle=none"
    )
    assert result.case_ids[-1] == (
        "Eu-154|pose=high-far|fe=07|pb=07|source=obstacle-top|obstacle=one-box"
    )
    assert np.all(np.isfinite(result.unit_response))
    assert np.all(result.unit_response >= 0.0)

    first = save_forward_conformance(tmp_path / "first.npz", result)
    second = save_forward_conformance(tmp_path / "second.npz", result)
    assert first.read_bytes() == second.read_bytes()
    with np.load(first, allow_pickle=False) as archive:
        assert archive.files == ["case_ids", "unit_response"]
        np.testing.assert_array_equal(archive["case_ids"], result.case_ids)
        np.testing.assert_array_equal(archive["unit_response"], result.unit_response)
        assert archive["unit_response"].dtype == np.dtype(np.float64)


def test_shared_manifest_spectral_lines_match_local_production_contract() -> None:
    """Bind shared-fixture line IDs and hashes to the local production table."""
    log = load_measurement_log(MEASUREMENT_FIXTURE)
    manifest = log.context.forward_model_manifest
    assert manifest is not None
    expected = registered_conformance_line_mu_by_isotope()
    model = build_runtime_observation_model(
        {
            "source_rate_model": "detector_cps_1m",
            "pf_line_resolved_shield_attenuation": True,
        },
        isotopes=log.context.isotopes,
    )
    assert model.line_mu_by_isotope is not None
    production = {
        isotope: [dict(entry) for entry in model.line_mu_by_isotope[isotope]]
        for isotope in log.context.isotopes
    }
    assert expected == production
    assert manifest["line_mu_by_isotope"] == production
    spectrum_identity = {
        isotope: [
            {
                "energy_keV": entry["energy_keV"],
                "weight": entry["weight"],
            }
            for entry in entries
        ]
        for isotope, entries in production.items()
    }
    assert (
        canonical_json_sha256(production)
        == (CONFORMANCE_MODEL_IDENTIFIERS["shield"]["sha256"])
    )
    assert (
        canonical_json_sha256(spectrum_identity)
        == (CONFORMANCE_MODEL_IDENTIFIERS["spectrum"]["sha256"])
    )
    validated = validate_forward_model_manifest(
        manifest,
        runtime_config=log.context.runtime_config,
        environment=log.context.environment,
        obstacle_layout_path=log.context.obstacle_layout_path,
        isotopes=log.context.isotopes,
        repository_commit=log.context.repository_commit,
        resolved_config_sha256=str(log.context.runtime_config_sha256),
        source_rate_model=log.context.source_rate_model,
    )
    assert validated["line_mu_by_isotope"] == production


def test_versioned_registry_is_exact_and_rejects_unknown_model_ids() -> None:
    """Keep the conformance-only registry narrow and fail closed on its ID."""
    line_table = registered_conformance_line_mu_by_isotope()
    manifest = {
        "schema_version": 1,
        "forward_model_id": CONFORMANCE_FORWARD_MODEL_ID,
        "repository_commit": "fixture-provider-commit",
        "resolved_config_sha256": "1" * 64,
        "source_rate_model": "detector_cps_1m",
        "source_rate_semantics": SOURCE_RATE_SEMANTICS,
        "model_identifiers": CONFORMANCE_MODEL_IDENTIFIERS,
        "units": CANONICAL_UNITS,
        "response_semantics": RESPONSE_SEMANTICS,
        "shield_program": {
            "fe_orientation_count": 8,
            "pb_orientation_count": 8,
            "pair_count": 64,
        },
        "line_mu_by_isotope": line_table,
    }
    validated = validate_forward_model_manifest(
        manifest,
        runtime_config={"source_rate_model": "detector_cps_1m"},
        environment={},
        obstacle_layout_path=None,
        isotopes=("Cs-137", "Co-60", "Eu-154"),
        repository_commit="fixture-provider-commit",
        resolved_config_sha256="1" * 64,
        source_rate_model="detector_cps_1m",
    )
    assert validated["forward_model_id"] == CONFORMANCE_FORWARD_MODEL_ID

    unknown = dict(manifest)
    unknown["forward_model_id"] = "unregistered-model-v1"
    with np.testing.assert_raises_regex(ValueError, "Unknown registered"):
        validate_forward_model_manifest(
            unknown,
            runtime_config={"source_rate_model": "detector_cps_1m"},
            environment={},
            obstacle_layout_path=None,
            isotopes=("Cs-137", "Co-60", "Eu-154"),
            repository_commit="fixture-provider-commit",
            resolved_config_sha256="1" * 64,
            source_rate_model="detector_cps_1m",
        )
