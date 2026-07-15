"""Tests for surface clustering and JSON-safe response diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from three_d_estimation.postprocess import (
    cluster_surface_hotspots,
    response_identifiability_diagnostics,
)


def test_clusters_report_stable_patch_ids_after_refinement() -> None:
    """Cluster payloads use stable IDs rather than transient dense indices."""
    patches = SimpleNamespace(
        areas_m2=np.asarray([1.0, 1.0]),
        centroids_xyz=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        kinds=("floor", "floor"),
        patch_ids=np.asarray([10, 42]),
        adjacency_edges=np.asarray([[0, 1]], dtype=np.int64),
    )

    clusters = cluster_surface_hotspots(
        patches,
        np.asarray([[2.0], [1.0]]),
        ("Cs-137",),
        threshold_fraction=0.4,
    )

    assert len(clusters) == 1
    assert clusters[0].patch_ids == (10, 42)


def test_all_zero_response_has_json_safe_condition_marker() -> None:
    """Unidentifiable all-zero designs report null instead of infinity."""
    diagnostics = response_identifiability_diagnostics(
        np.zeros((2, 3, 1), dtype=float)
    )

    assert diagnostics["rank"] == 0
    assert diagnostics["condition_number_nonzero"] is None
    assert diagnostics["zero_response_columns"] == [0, 1, 2]
