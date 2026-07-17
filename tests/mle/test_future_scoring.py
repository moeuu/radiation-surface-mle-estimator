"""Tests for frozen-snapshot future-only count candidate scoring."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from runtime.measurement_log import load_measurement_log, save_measurement_log
from runtime.records import canonical_json_bytes
from three_d_estimation.cli import main
from three_d_estimation.config import MLEConfig
from three_d_estimation.future_scoring import (
    covered_station_boundaries_sha256,
    save_future_candidate_scores,
    score_future_count_candidates,
)
from three_d_estimation.measurement_prefix import (
    materialize_measurement_log_prefix,
)
from three_d_estimation.replay import run_replay
from three_d_estimation.reporting import mle_report_sha256, save_mle_estimate


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "fixtures" / "shared_measurement_log_v1" / "measurement_log"
PROVENANCE_FIELDS = (
    "estimator_family",
    "estimator_variant",
    "candidate_domain",
    "uses_pf_state",
    "uses_pf_candidates",
    "estimator_commit",
    "measurement_run_id",
    "measurement_log_schema_version",
    "measurement_log_sha256",
    "forward_model_manifest_sha256",
    "config_sha256",
    "resolved_config_sha256",
    "resolved_estimator_config_sha256",
)


def _config() -> MLEConfig:
    """Return a fast deterministic count configuration."""
    return MLEConfig(
        mode="count",
        isotope_names=("Cs-137", "Co-60", "Eu-154"),
        patch_spacing_m=(6.0, 6.0, 3.0),
        max_iterations=4,
        check_interval=2,
        debias_refit=False,
        fit_background_nuisance=False,
        fit_scatter_nuisance=False,
        use_gpu=False,
        random_seed=31,
    )


def _marked_source(tmp_path: Path) -> Path:
    """Persist a fixture copy with cumulative explicit station-end markers."""
    loaded = load_measurement_log(FIXTURE)
    terminals = {2, 5, 8, 11}
    records = [
        replace(
            record,
            metadata={
                **record.metadata,
                **({"station_complete": True} if record.step_id in terminals else {}),
            },
        )
        for record in loaded.records
    ]
    return save_measurement_log(tmp_path / "marked-source", loaded.context, records)


def _prefix(source: Path, target: Path, *, step: int, station: int) -> Path:
    """Create a writer-marked deterministic prefix."""
    return materialize_measurement_log_prefix(
        source,
        target,
        cutoff_step=step,
        cutoff_station=station,
    ).output_dir


def _snapshot_payload(
    *,
    prefix: Path,
    estimate: object,
    report: Path,
) -> dict[str, object]:
    """Build the exact MLESnapshot v2 contract from a saved count report."""
    diagnostics = dict(getattr(estimate, "diagnostics"))
    provenance = diagnostics["provenance"]
    lineage = diagnostics["causal_lineage"]
    predictions = np.asarray(getattr(estimate, "predicted_isotope_counts"), dtype=float)
    isotopes = tuple(getattr(estimate, "isotope_names"))
    clusters = [
        {
            "snapshot_candidate_id": f"snapshot-0:cluster:{cluster['cluster_id']}",
            "cluster_id": cluster["cluster_id"],
            "isotope": cluster["isotope"],
            "centroid_xyz": list(cluster["centroid_xyz"]),
            "integrated_strength_cps_1m": cluster["integrated_strength_cps_1m"],
            "surface_kinds": list(cluster["surface_kinds"]),
            "patch_ids": list(cluster["patch_ids"]),
        }
        for cluster in diagnostics["hotspot_clusters"]
    ]
    return {
        "schema_version": 2,
        "snapshot_id": "snapshot-0",
        "trigger_id": "trigger-0",
        "estimator_family": "surface_mle",
        "estimator_variant": "count",
        "data_cutoff_step": lineage["data_cutoff_step"],
        "data_cutoff_station": lineage["data_cutoff_station"],
        "cutoff_station_complete": True,
        "covered_step_ids": lineage["covered_step_ids"],
        "source_run_id": provenance["measurement_run_id"],
        "prefix_measurement_log_sha256": provenance["measurement_log_sha256"],
        "covered_records_sha256": lineage["covered_records_sha256"],
        "covered_station_boundaries_sha256": covered_station_boundaries_sha256(
            load_measurement_log(prefix),
            cutoff_step=int(lineage["data_cutoff_step"]),
        ),
        "mle_result_sha256": mle_report_sha256(report),
        "warm_start": {
            "used": False,
            "snapshot_id": None,
            "mle_result_sha256": None,
        },
        "clusters": clusters,
        "predicted_observations": [
            {
                "step_id": int(step_id),
                "isotope_counts": {
                    isotope: float(predictions[row, column])
                    for column, isotope in enumerate(isotopes)
                },
            }
            for row, step_id in enumerate(lineage["covered_step_ids"])
        ],
        "fit_diagnostics": {
            "objective": float(getattr(estimate, "objective_value")),
            "converged": bool(getattr(estimate, "converged")),
        },
        "safety": {
            "direct_mle_objective_reweight": False,
            "hard_prune_authorized": False,
        },
        "provenance": {name: provenance[name] for name in PROVENANCE_FIELDS},
    }


def _scenario(tmp_path: Path) -> dict[str, object]:
    """Build one earlier snapshot report and a later marked prefix."""
    source = _marked_source(tmp_path)
    earlier = _prefix(source, tmp_path / "station-0", step=2, station=0)
    current = _prefix(source, tmp_path / "station-1", step=5, station=1)
    config = _config()
    config_path = tmp_path / "mle-count.json"
    config.save(config_path)
    estimate = run_replay(earlier, config=config_path).estimate
    report = tmp_path / "mle-report"
    save_mle_estimate(estimate, report, config=config)
    snapshot_payload = _snapshot_payload(
        prefix=earlier,
        estimate=estimate,
        report=report,
    )
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_bytes(canonical_json_bytes(snapshot_payload))
    return {
        "source": source,
        "earlier": earlier,
        "current": current,
        "config": config,
        "config_path": config_path,
        "estimate": estimate,
        "report": report,
        "snapshot": snapshot,
        "snapshot_payload": snapshot_payload,
    }


def test_future_scoring_is_frozen_future_only_and_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scoring uses post-cutoff counts without invoking any MLE refit."""
    scenario = _scenario(tmp_path)

    def forbidden_fit(*args: object, **kwargs: object) -> object:
        """Fail if future verification calls the optimizer."""
        raise AssertionError("Future-only candidate scoring must not refit MLE.")

    monkeypatch.setattr(
        "three_d_estimation.estimator.SurfaceMLEEstimator.fit",
        forbidden_fit,
    )
    first = score_future_count_candidates(
        scenario["current"],
        config=scenario["config_path"],
        snapshot_estimate=scenario["report"],
        snapshot=scenario["snapshot"],
    )
    second = score_future_count_candidates(
        scenario["current"],
        config=scenario["config_path"],
        snapshot_estimate=scenario["report"],
        snapshot=scenario["snapshot"],
    )

    assert first == second
    assert first["future_step_ids"] == [3, 4, 5]
    assert first["future_station_ids"] == [1, 1, 1]
    assert first["safety"] == {
        "future_only": True,
        "snapshot_parameters_frozen": True,
        "no_refit": True,
        "truth_used": False,
    }
    assert len(first["candidates"]) > 0
    for candidate in first["candidates"]:
        per_step = candidate["future_step_scores"]
        assert [row["step_id"] for row in per_step] == [3, 4, 5]
        values = [row["log_predictive_likelihood_ratio"] for row in per_step]
        assert all(np.isfinite(values))
        assert candidate["cumulative_log_predictive_likelihood_ratio"] == pytest.approx(
            sum(values), rel=0.0, abs=1.0e-12
        )
    hashes = first["hashes"]
    assert (
        hashes["snapshot_file_sha256"]
        == sha256(Path(scenario["snapshot"]).read_bytes()).hexdigest()
    )
    assert hashes["snapshot_mle_report_sha256"] == mle_report_sha256(scenario["report"])


def test_future_scoring_cli_writes_canonical_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The public CLI persists the same deterministic score contract."""
    scenario = _scenario(tmp_path)
    output = tmp_path / "future-score.json"

    assert (
        main(
            [
                "score-future",
                "--run-dir",
                str(scenario["current"]),
                "--mle-config",
                str(scenario["config_path"]),
                "--snapshot-estimate",
                str(scenario["report"]),
                "--snapshot",
                str(scenario["snapshot"]),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert "future_steps: 3" in capsys.readouterr().out
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["safety"]["truth_used"] is False
    second = tmp_path / "future-score-copy.json"
    save_future_candidate_scores(second, payload)
    assert second.read_bytes() == output.read_bytes()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload.update(estimator_variant="spectral"), "count"),
        (
            lambda payload: payload.update(mle_result_sha256="0" * 64),
            "bind the supplied MLE report",
        ),
        (
            lambda payload: payload["clusters"][0].update(
                integrated_strength_cps_1m=(
                    payload["clusters"][0]["integrated_strength_cps_1m"] + 1.0
                )
            ),
            "differs from the MLE report",
        ),
        (
            lambda payload: payload.update(covered_station_boundaries_sha256="0" * 64),
            "station-boundary lineage",
        ),
        (lambda payload: payload.update(unexpected=True), "top-level fields"),
    ),
)
def test_future_scoring_rejects_snapshot_contract_mismatch(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    """Snapshot mode, report binding, candidates, lineage, and schema are exact."""
    scenario = _scenario(tmp_path)
    payload = json.loads(Path(scenario["snapshot"]).read_text(encoding="utf-8"))
    mutation(payload)
    broken = tmp_path / "broken-snapshot.json"
    broken.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(ValueError, match=message):
        score_future_count_candidates(
            scenario["current"],
            config=scenario["config_path"],
            snapshot_estimate=scenario["report"],
            snapshot=broken,
        )


def test_future_scoring_rejects_missing_counts_and_nonfuture_cutoff(
    tmp_path: Path,
) -> None:
    """Count observations and at least one strict post-cutoff row are mandatory."""
    scenario = _scenario(tmp_path)
    current_log = load_measurement_log(scenario["current"])
    no_counts = [
        replace(
            record,
            counts_by_isotope=None,
            count_covariance_by_isotope=None,
        )
        for record in current_log.records
    ]
    missing = save_measurement_log(
        tmp_path / "missing-counts",
        current_log.context,
        no_counts,
    )
    with pytest.raises(ValueError, match="requires isotope count observations"):
        score_future_count_candidates(
            missing,
            config=scenario["config_path"],
            snapshot_estimate=scenario["report"],
            snapshot=scenario["snapshot"],
        )

    payload = json.loads(Path(scenario["snapshot"]).read_text(encoding="utf-8"))
    payload["data_cutoff_step"] = 5
    payload["data_cutoff_station"] = 1
    payload["covered_step_ids"] = [0, 1, 2, 3, 4, 5]
    no_future = tmp_path / "no-future.json"
    no_future.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ValueError, match="strict prefix"):
        score_future_count_candidates(
            scenario["current"],
            config=scenario["config_path"],
            snapshot_estimate=scenario["report"],
            snapshot=no_future,
        )
