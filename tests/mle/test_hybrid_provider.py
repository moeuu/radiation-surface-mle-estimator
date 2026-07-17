"""Tests for causal MeasurementLog prefixes and artifact warm starts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from runtime.measurement_log import load_measurement_log, save_measurement_log
from three_d_estimation.cli import main
from three_d_estimation.config import MLEConfig
from three_d_estimation.measurement_prefix import (
    materialize_measurement_log_prefix,
)
from three_d_estimation.replay import run_replay
from three_d_estimation.reporting import save_mle_estimate


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "fixtures" / "shared_measurement_log_v1" / "measurement_log"


def _config(mode: str = "count") -> MLEConfig:
    """Return a small deterministic configuration for provider tests."""
    return MLEConfig(
        mode=mode,
        isotope_names=("Cs-137", "Co-60", "Eu-154"),
        patch_spacing_m=(6.0, 6.0, 3.0),
        max_iterations=4,
        check_interval=2,
        debias_refit=False,
        fit_background_nuisance=False,
        fit_scatter_nuisance=False,
        use_gpu=False,
        random_seed=23,
    )


def _prefix(
    source: Path,
    target: Path,
    *,
    step: int,
    station: int,
) -> Path:
    """Materialize one fixture boundary with explicit schedule attestation."""
    return materialize_measurement_log_prefix(
        source,
        target,
        cutoff_step=step,
        cutoff_station=station,
        assert_station_complete=True,
    ).output_dir


def _save_prior(prefix: Path, output: Path, config: MLEConfig) -> Path:
    """Fit and save one prefix estimate for later warm initialization."""
    estimate = run_replay(prefix, config=config).estimate
    save_mle_estimate(estimate, output, config=config)
    return output


def _inventory(root: Path) -> dict[str, bytes]:
    """Return every relative regular-file payload below a test log."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_prefix_is_deterministic_suffix_invariant_and_truth_free(
    tmp_path: Path,
) -> None:
    """The same attested records produce identical logs despite later data."""
    loaded = load_measurement_log(FIXTURE)
    changed_suffix = list(loaded.records)
    changed_suffix[3] = replace(
        changed_suffix[3],
        spectrum_counts=changed_suffix[3].spectrum_counts + 17.0,
        counts_by_isotope={
            name: value + 3.0
            for name, value in (changed_suffix[3].counts_by_isotope or {}).items()
        },
    )
    alternative = save_measurement_log(
        tmp_path / "alternative-full",
        loaded.context,
        changed_suffix,
    )
    first = _prefix(FIXTURE, tmp_path / "prefix-a", step=2, station=0)
    second = _prefix(alternative, tmp_path / "prefix-b", step=2, station=0)

    assert _inventory(first) == _inventory(second)
    prefix_log = load_measurement_log(first)
    assert [record.step_id for record in prefix_log.records] == [0, 1, 2]
    assert prefix_log.context.metadata["measurement_log_prefix"] == {
        "schema_version": 1,
        "source_run_id": loaded.context.run_id,
        "data_cutoff_step": 2,
        "data_cutoff_station": 0,
        "station_boundary_attestation": "external_validated_schedule",
    }
    assert (
        materialize_measurement_log_prefix(
            FIXTURE,
            tmp_path / "prefix-contract-digest",
            cutoff_step=2,
            cutoff_station=0,
            assert_station_complete=True,
        ).covered_records_sha256
        == "f57c5e5cc83689dfed4b12310e3b63d27e3e95d0c5d53e0904763879f7430efb"
    )
    assert not any("truth" in name.casefold() for name in _inventory(first))


def test_prefix_preserves_declared_run_local_forward_model_assets(
    tmp_path: Path,
) -> None:
    """A standalone prefix retains every referenced run-local model asset."""
    loaded = load_measurement_log(FIXTURE)
    relative_asset = "assets/obstacle.json"
    asset_payload = (
        b'{"blocked_cells":[],"cell_size":1.0,"grid_shape":[6,6],"origin":[0.0,0.0]}\n'
    )
    asset_root = tmp_path / "asset-root"
    asset_path = asset_root / relative_asset
    asset_path.parent.mkdir(parents=True)
    asset_path.write_bytes(asset_payload)
    environment = dict(loaded.context.environment)
    environment.pop("obstacle_grid", None)
    context = replace(
        loaded.context,
        environment=environment,
        obstacle_layout_path=relative_asset,
        forward_model_manifest=None,
    )
    source = save_measurement_log(
        tmp_path / "source-with-asset",
        context,
        loaded.records,
        extra_artifacts={relative_asset: asset_payload},
        model_asset_root=asset_root,
    )

    prefix = _prefix(source, tmp_path / "asset-prefix", step=2, station=0)

    assert (prefix / relative_asset).read_bytes() == asset_payload
    assert load_measurement_log(prefix).records[-1].step_id == 2


def test_prefix_rejects_unattested_and_incomplete_station_boundaries(
    tmp_path: Path,
) -> None:
    """Prefix construction never infers station completion from later rows."""
    with pytest.raises(ValueError, match="station_complete=true"):
        materialize_measurement_log_prefix(
            FIXTURE,
            tmp_path / "unattested",
            cutoff_step=2,
            cutoff_station=0,
        )
    with pytest.raises(ValueError, match="not station-complete"):
        materialize_measurement_log_prefix(
            FIXTURE,
            tmp_path / "incomplete",
            cutoff_step=1,
            cutoff_station=0,
            assert_station_complete=True,
        )
    with pytest.raises(ValueError, match="belongs to station"):
        materialize_measurement_log_prefix(
            FIXTURE,
            tmp_path / "wrong-station",
            cutoff_step=2,
            cutoff_station=1,
            assert_station_complete=True,
        )


def test_materialize_prefix_cli_reports_exact_lineage(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The public CLI exposes exact step/station boundary assertions."""
    output = tmp_path / "prefix"
    assert (
        main(
            [
                "materialize-prefix",
                "--run-dir",
                str(FIXTURE),
                "--output-dir",
                str(output),
                "--cutoff-step",
                "2",
                "--cutoff-station",
                "0",
                "--assert-station-complete",
                "--json",
            ]
        )
        == 0
    )
    stdout = capsys.readouterr().out
    assert '"data_cutoff_step": 2' in stdout
    assert '"data_cutoff_station": 0' in stdout
    assert load_measurement_log(output).records[-1].step_id == 2


@pytest.mark.parametrize("mode", ["count", "spectral"])
def test_count_and_spectral_warm_start_are_causal_and_deterministic(
    tmp_path: Path,
    mode: str,
) -> None:
    """Both variants use prior artifacts only as validated initialization."""
    first_prefix = _prefix(
        FIXTURE,
        tmp_path / f"{mode}-station-0",
        step=2,
        station=0,
    )
    second_prefix = _prefix(
        FIXTURE,
        tmp_path / f"{mode}-station-1",
        step=5,
        station=1,
    )
    config = _config(mode)
    prior = _save_prior(first_prefix, tmp_path / f"{mode}-prior", config)

    first = run_replay(
        second_prefix,
        config=config,
        initial_estimate_path=prior,
    ).estimate
    second = run_replay(
        second_prefix,
        config=config,
        initial_estimate_path=prior,
    ).estimate

    np.testing.assert_array_equal(first.density_by_isotope, second.density_by_isotope)
    assert first.diagnostics == second.diagnostics
    assert first.diagnostics["warm_started"] is True
    lineage = first.diagnostics["causal_lineage"]
    assert lineage["covered_step_ids"] == [0, 1, 2, 3, 4, 5]
    assert lineage["data_cutoff_step"] == 5
    assert lineage["data_cutoff_station"] == 1
    assert lineage["record_count"] == 6
    assert lineage["fit_kind"] == "warm_start_all_history"
    warm = lineage["warm_start"]
    assert isinstance(warm, dict)
    assert warm["data_cutoff_step"] == 2
    assert warm["data_cutoff_station"] == 0
    assert warm["record_count"] == 3
    for name in ("report_sha256", "estimate_sha256", "diagnostics_sha256"):
        assert len(warm[name]) == 64
    assert first.diagnostics["provenance"]["causal_lineage"] == lineage


def test_warm_start_fails_closed_on_every_compatibility_boundary(
    tmp_path: Path,
) -> None:
    """Mode, isotope, config, forward identity, and prefix content are exact."""
    first_prefix = _prefix(
        FIXTURE,
        tmp_path / "station-0",
        step=2,
        station=0,
    )
    second_prefix = _prefix(
        FIXTURE,
        tmp_path / "station-1",
        step=5,
        station=1,
    )
    config = _config("count")
    prior = _save_prior(first_prefix, tmp_path / "prior", config)

    with pytest.raises(ValueError, match="mode is incompatible"):
        run_replay(
            second_prefix,
            config=_config("spectral"),
            initial_estimate_path=prior,
        )
    with pytest.raises(ValueError, match="configuration is incompatible"):
        run_replay(
            second_prefix,
            config=replace(config, max_iterations=config.max_iterations + 1),
            initial_estimate_path=prior,
        )

    loaded = load_measurement_log(FIXTURE)
    reordered = tuple(reversed(loaded.context.isotopes))
    isotope_context = replace(
        loaded.context,
        isotopes=reordered,
        forward_model_manifest=None,
    )
    isotope_log = save_measurement_log(
        tmp_path / "different-isotopes",
        isotope_context,
        loaded.records[:6],
    )
    with pytest.raises(ValueError, match="isotope ordering is incompatible"):
        run_replay(
            isotope_log,
            config=replace(config, isotope_names=reordered),
            initial_estimate_path=prior,
        )

    changed_environment = dict(loaded.context.environment)
    changed_environment["size_z"] = float(changed_environment["size_z"]) + 0.25
    forward_context = replace(
        loaded.context,
        environment=changed_environment,
        forward_model_manifest=None,
    )
    forward_log = save_measurement_log(
        tmp_path / "different-forward-model",
        forward_context,
        loaded.records[:6],
    )
    with pytest.raises(ValueError, match="forward_model_manifest_sha256"):
        run_replay(
            forward_log,
            config=config,
            initial_estimate_path=prior,
        )

    changed_records = list(loaded.records[:6])
    changed_records[0] = replace(
        changed_records[0],
        spectrum_counts=changed_records[0].spectrum_counts + 1.0,
        counts_by_isotope={
            name: value + 1.0
            for name, value in (changed_records[0].counts_by_isotope or {}).items()
        },
    )
    changed_log = save_measurement_log(
        tmp_path / "different-prefix-content",
        loaded.context,
        changed_records,
    )
    with pytest.raises(ValueError, match="record content is not a prefix"):
        run_replay(
            changed_log,
            config=config,
            initial_estimate_path=prior,
        )


def test_cold_full_replay_remains_the_default(tmp_path: Path) -> None:
    """Omitting warm-start flags retains an independent full-history fit."""
    config = _config("count")
    first = run_replay(FIXTURE, config=config).estimate
    second = run_replay(FIXTURE, config=config).estimate

    np.testing.assert_array_equal(first.density_by_isotope, second.density_by_isotope)
    assert first.diagnostics == second.diagnostics
    assert first.diagnostics["warm_started"] is False
    lineage = first.diagnostics["causal_lineage"]
    assert lineage["fit_kind"] == "cold_start_all_history"
    assert lineage["warm_start"] is None
    assert lineage["record_count"] == 12
