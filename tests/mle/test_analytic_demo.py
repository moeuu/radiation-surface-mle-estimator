"""End-to-end tests for the sampled analytic surface-MLE demo log."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from measurement.continuous_kernels import ContinuousKernel
from runtime.measurement_log import load_measurement_log
from three_d_estimation.demo import (
    AnalyticMLEDemoScenario,
    build_analytic_mle_demo,
    create_analytic_mle_demo_log,
)
from three_d_estimation.replay import prepare_replay, run_replay
from three_d_estimation.reporting import load_mle_estimate, save_mle_estimate
from three_d_estimation.response_builder import build_count_response


ROOT = Path(__file__).resolve().parents[2]


def _directory_bytes(root: Path) -> dict[str, bytes]:
    """Return every regular file below a directory keyed by relative path."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture(scope="module")
def demo_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[AnalyticMLEDemoScenario, Path]:
    """Create one module-scoped sampled demo log for replay assertions."""
    scenario = build_analytic_mle_demo()
    run_dir = create_analytic_mle_demo_log(
        tmp_path_factory.mktemp("analytic-demo") / "run",
        scenario=scenario,
    )
    return scenario, run_dir


def test_sampled_demo_log_is_byte_deterministic_and_self_contained(
    demo_artifacts: tuple[AnalyticMLEDemoScenario, Path],
    tmp_path: Path,
) -> None:
    """Repeated genuine simulator acquisition should persist identical bytes."""
    first_scenario, first_run = demo_artifacts
    second_scenario = build_analytic_mle_demo()
    second_run = create_analytic_mle_demo_log(
        tmp_path / "run",
        scenario=second_scenario,
    )

    assert _directory_bytes(first_run) == _directory_bytes(second_run)
    loaded = load_measurement_log(first_run)
    assert len(loaded.records) == 32
    assert len(
        {
            (record.fe_orientation_index, record.pb_orientation_index)
            for record in loaded.records
        }
    ) == 16
    assert loaded.context.obstacle_layout_path is None
    assert loaded.context.source_layout_path is None
    assert loaded.context.environment["obstacle_grid"]["blocked_cells"] == []
    assert all(record.spectrum_variance is None for record in loaded.records)
    assert all(
        record.metadata["backend"] == "analytic"
        and record.metadata["transport_backend"] == "python"
        and record.metadata["observation_generation"]
        == "sampled_analytic_runtime_spectrum"
        for record in loaded.records
    )

    replay = prepare_replay(first_run, config=first_scenario.mle_config)
    assert replay.resolved_obstacle_path is None
    assert isinstance(replay.kernel, ContinuousKernel)
    response = build_count_response(
        replay.batch,
        first_scenario.patches,
        first_scenario.mle_config.isotope_names,
        replay.kernel,
    )
    model_counts = (
        response[:, first_scenario.source_patch_index, 0]
        * first_scenario.source_strength_cps_1m
    )
    sampled_counts = replay.batch.isotope_counts[:, 0]
    assert not np.array_equal(sampled_counts, model_counts)
    assert np.linalg.norm(sampled_counts - model_counts) / np.linalg.norm(
        model_counts
    ) < 0.08
    assert first_scenario.route_response_rank == first_scenario.patches.patch_count


def test_replay_recovers_patch_and_report_round_trips_deterministically(
    demo_artifacts: tuple[AnalyticMLEDemoScenario, Path],
    tmp_path: Path,
) -> None:
    """All-history replay should recover the known patch and stable report."""
    scenario, run_dir = demo_artifacts
    first = run_replay(run_dir, config=scenario.mle_config).estimate
    second = run_replay(run_dir, config=scenario.mle_config).estimate

    assert first.converged
    np.testing.assert_array_equal(
        first.patch_strength_by_isotope,
        second.patch_strength_by_isotope,
    )
    np.testing.assert_array_equal(
        first.predicted_isotope_counts,
        second.predicted_isotope_counts,
    )
    strengths = first.patch_strength_by_isotope[0]
    strongest_index = int(np.argmax(strengths))
    assert first.patches[strongest_index].patch_id == scenario.source_patch_id
    assert strengths[strongest_index] == pytest.approx(
        scenario.source_strength_cps_1m,
        rel=0.02,
    )

    clusters = [
        cluster
        for cluster in first.diagnostics["hotspot_clusters"]
        if cluster["isotope"] == "Cs-137"
    ]
    assert clusters
    source_centroid = scenario.patches.patches[
        scenario.source_patch_index
    ].centroid_xyz
    nearest = min(
        clusters,
        key=lambda cluster: np.linalg.norm(
            np.asarray(cluster["centroid_xyz"], dtype=float) - source_centroid
        ),
    )
    assert tuple(nearest["patch_ids"]) == (scenario.source_patch_id,)
    np.testing.assert_allclose(nearest["centroid_xyz"], source_centroid, atol=1.0e-12)

    first_report = tmp_path / "first-report"
    second_report = tmp_path / "second-report"
    save_mle_estimate(first, first_report, scenario.mle_config)
    save_mle_estimate(second_report, second, scenario.mle_config)
    assert _directory_bytes(first_report) == _directory_bytes(second_report)
    restored = load_mle_estimate(first_report)
    np.testing.assert_array_equal(
        restored.patch_strength_by_isotope,
        first.patch_strength_by_isotope,
    )
    np.testing.assert_array_equal(
        restored.predicted_isotope_counts,
        first.predicted_isotope_counts,
    )


def test_generator_script_and_replay_cli_run_end_to_end(tmp_path: Path) -> None:
    """The public scripts should create, replay, and save the sampled demo."""
    run_dir = tmp_path / "run"
    config_path = tmp_path / "mle-config.json"
    create_process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "create_analytic_mle_demo_log.py"),
            "--output-dir",
            str(run_dir),
            "--mle-config-output",
            str(config_path),
            "--json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert create_process.returncode == 0, create_process.stderr
    create_summary = json.loads(create_process.stdout)
    assert create_summary["record_count"] == 32
    assert create_summary["observation_generation"] == (
        "sampled_analytic_runtime_spectrum"
    )
    assert config_path.is_file()

    report_dir = tmp_path / "mle-report"
    replay_process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_mle_replay.py"),
            "replay",
            "--run-dir",
            str(run_dir),
            "--mle-config",
            str(config_path),
            "--output-dir",
            str(report_dir),
            "--cpu",
            "--json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert replay_process.returncode == 0, replay_process.stderr
    replay_summary = json.loads(replay_process.stdout)
    assert replay_summary["mode"] == "count"
    assert replay_summary["converged"] is True
    assert replay_summary["cluster_count"] >= 1

    estimate = load_mle_estimate(report_dir)
    strongest = int(np.argmax(estimate.patch_strength_by_isotope[0]))
    assert estimate.patches[strongest].patch_id == create_summary["source_patch_id"]
    assert estimate.patch_strength_by_isotope[0, strongest] == pytest.approx(
        create_summary["source_strength_cps_1m"],
        rel=0.02,
    )
