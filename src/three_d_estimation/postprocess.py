"""Post-processing and identifiability diagnostics for fitted surface maps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class HotspotCluster:
    """Describe one connected high-intensity region on the surface graph."""

    isotope: str
    cluster_id: int
    patch_ids: tuple[int, ...]
    centroid_xyz: tuple[float, float, float]
    integrated_strength_cps_1m: float
    peak_density_cps_1m_m2: float
    surface_kinds: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable cluster payload."""
        return asdict(self)


def cluster_surface_hotspots(
    patches: object,
    densities_cps_1m_m2: NDArray[np.float64],
    isotope_names: Sequence[str],
    *,
    threshold_fraction: float = 0.1,
    min_strength_cps_1m: float = 0.0,
) -> tuple[HotspotCluster, ...]:
    """Cluster above-threshold patches using the physical adjacency graph."""
    densities = np.asarray(densities_cps_1m_m2, dtype=float)
    areas = np.asarray(getattr(patches, "areas_m2"), dtype=float).reshape(-1)
    centers = np.asarray(getattr(patches, "centroids_xyz"), dtype=float)
    kinds = np.asarray(getattr(patches, "kinds"), dtype=str).reshape(-1)
    stable_patch_ids = np.asarray(
        getattr(patches, "patch_ids", np.arange(areas.size)),
        dtype=np.int64,
    ).reshape(-1)
    edges = np.asarray(getattr(patches, "adjacency_edges"), dtype=np.int64)
    names = tuple(str(value) for value in isotope_names)
    if densities.shape != (areas.size, len(names)):
        raise ValueError("densities must have shape (patches, isotopes).")
    if stable_patch_ids.shape != (areas.size,):
        raise ValueError("patch_ids must contain one stable ID per patch.")
    if not 0.0 <= float(threshold_fraction) <= 1.0:
        raise ValueError("threshold_fraction must lie between zero and one.")
    if edges.size:
        rows = np.concatenate([edges[:, 0], edges[:, 1]])
        cols = np.concatenate([edges[:, 1], edges[:, 0]])
        graph = csr_matrix(
            (np.ones(rows.size, dtype=float), (rows, cols)),
            shape=(areas.size, areas.size),
        )
    else:
        graph = csr_matrix((areas.size, areas.size), dtype=float)
    clusters: list[HotspotCluster] = []
    next_id = 0
    for isotope_index, isotope in enumerate(names):
        values = np.maximum(densities[:, isotope_index], 0.0)
        peak = float(np.max(values, initial=0.0))
        if peak <= 0.0:
            continue
        active = values >= peak * float(threshold_fraction)
        indices = np.flatnonzero(active)
        if indices.size == 0:
            continue
        component_count, labels = connected_components(
            graph[indices][:, indices], directed=False, return_labels=True
        )
        strengths = values * areas
        for component_id in range(int(component_count)):
            patch_indices = indices[labels == component_id]
            integrated = float(np.sum(strengths[patch_indices]))
            if integrated < float(min_strength_cps_1m):
                continue
            weights = strengths[patch_indices]
            if float(np.sum(weights)) <= 0.0:
                weights = areas[patch_indices]
            centroid = np.average(centers[patch_indices], axis=0, weights=weights)
            clusters.append(
                HotspotCluster(
                    isotope=isotope,
                    cluster_id=next_id,
                    patch_ids=tuple(
                        int(value) for value in stable_patch_ids[patch_indices]
                    ),
                    centroid_xyz=tuple(float(value) for value in centroid),
                    integrated_strength_cps_1m=integrated,
                    peak_density_cps_1m_m2=float(np.max(values[patch_indices])),
                    surface_kinds=tuple(
                        sorted(set(kinds[patch_indices].tolist()))
                    ),
                )
            )
            next_id += 1
    return tuple(clusters)


def response_identifiability_diagnostics(
    response: NDArray[np.float64],
    *,
    correlation_threshold: float = 0.995,
    maximum_pairs: int = 1000,
) -> dict[str, object]:
    """Return rank, condition, and high cosine-correlation diagnostics."""
    array = np.asarray(response, dtype=float)
    if array.ndim < 2:
        raise ValueError("response must contain observation and source dimensions.")
    if array.ndim >= 4:
        patch_count = int(array.shape[-2])
        isotope_count = int(array.shape[-1])
        matrix = array.reshape(-1, patch_count * isotope_count)
    elif array.ndim == 3:
        patch_count = int(array.shape[-2])
        isotope_count = int(array.shape[-1])
        matrix = array.reshape(-1, patch_count * isotope_count)
    else:
        patch_count = int(array.shape[-1])
        isotope_count = 1
        matrix = array.reshape(-1, patch_count)
    norms = np.linalg.norm(matrix, axis=0)
    active = norms > 1.0e-15
    normalized = np.zeros_like(matrix)
    normalized[:, active] = matrix[:, active] / norms[active][None, :]
    gram = np.clip(normalized.T @ normalized, -1.0, 1.0)
    upper = np.triu(gram, k=1)
    rows, cols = np.nonzero(upper >= float(correlation_threshold))
    values = upper[rows, cols]
    if values.size:
        order = np.argsort(values)[::-1][: max(int(maximum_pairs), 0)]
        correlated = [
            {
                "column_a": int(rows[index]),
                "column_b": int(cols[index]),
                "patch_a": int(rows[index] // isotope_count),
                "patch_b": int(cols[index] // isotope_count),
                "isotope_a": int(rows[index] % isotope_count),
                "isotope_b": int(cols[index] % isotope_count),
                "cosine_correlation": float(values[index]),
            }
            for index in order
        ]
    else:
        correlated = []
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    positive = singular_values[singular_values > 1.0e-12]
    condition: float | None = (
        float(positive[0] / positive[-1])
        if positive.size > 1
        else (1.0 if positive.size == 1 else None)
    )
    return {
        "matrix_shape": [int(value) for value in matrix.shape],
        "rank": int(np.linalg.matrix_rank(matrix)),
        "condition_number_nonzero": condition,
        "zero_response_columns": np.flatnonzero(~active).astype(int).tolist(),
        "high_correlation_threshold": float(correlation_threshold),
        "high_correlation_pair_count": int(rows.size),
        "high_correlation_pairs": correlated,
    }


def poisson_deviance(
    observed: NDArray[np.float64],
    expected: NDArray[np.float64],
    *,
    min_mean: float = 1.0e-12,
) -> float:
    """Return Poisson deviance for evaluation or held-out reporting."""
    counts = np.asarray(observed, dtype=float)
    mean = np.maximum(np.asarray(expected, dtype=float), float(min_mean))
    if counts.shape != mean.shape or np.any(counts < 0.0):
        raise ValueError("observed and expected must be matching non-negative arrays.")
    positive = counts > 0.0
    terms = np.array(mean - counts, copy=True)
    terms[positive] += counts[positive] * np.log(counts[positive] / mean[positive])
    return float(2.0 * np.sum(terms))
