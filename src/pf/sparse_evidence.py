"""Sparse all-history Poisson evidence for isotope source cardinality."""

from __future__ import annotations

from dataclasses import dataclass, replace
import itertools
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SparsePoissonEvidenceConfig:
    """Configure sparse Poisson evidence scoring over a fixed response dictionary."""

    max_sources: int
    candidate_limit: int = 2048
    parameter_count_per_source: int = 4
    refit_iters: int = 64
    holdout_stride: int = 0
    correlation_prune_threshold: float = 0.995
    eps: float = 1.0e-12
    q_max: float = 0.0
    nuisance_parameter_count: int = 0


@dataclass(frozen=True)
class SparsePoissonEvidence:
    """Store source-count evidence and the selected sparse dictionary model."""

    available: bool
    reason: str
    selected_count: int
    selected_indices: tuple[int, ...]
    selected_strengths: tuple[float, ...]
    bic_by_count: tuple[float, ...]
    aicc_by_count: tuple[float, ...]
    log_likelihood_by_count: tuple[float, ...]
    heldout_deviance_by_count: tuple[float, ...]
    best_bic_count: int
    best_aicc_count: int
    best_heldout_count: int | None
    bic_gap_to_simpler: float
    bic_gap_to_more_complex: float
    bic_gap_to_previous_count: float
    bic_gap_to_next_count: float
    bic_margin_to_runner_up: float
    condition_number: float
    selected_max_response_correlation: float
    n_observations: int
    n_candidates: int
    evaluated_candidate_count: int
    candidate_indices: tuple[int, ...]
    method: str = "all_history_sparse_poisson"
    selected_nuisance_strengths: tuple[float, ...] = ()
    selected_column_metadata: tuple[dict[str, Any], ...] = ()
    ambiguity_clusters: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class SpectralResponseTensor:
    """Store a flattened spectrum-bin response tensor for sparse evidence."""

    counts: NDArray[np.float64]
    response_matrix: NDArray[np.float64]
    background: NDArray[np.float64]
    column_metadata: tuple[dict[str, Any], ...]
    measurement_count: int
    bin_count: int
    candidate_count: int
    isotope_count: int


@dataclass(frozen=True)
class JointSparsePoissonEvidence:
    """Store joint multi-isotope source-cardinality evidence."""

    available: bool
    reason: str
    selected_counts_by_isotope: dict[str, int]
    selected_indices_by_isotope: dict[str, tuple[int, ...]]
    selected_strengths_by_isotope: dict[str, tuple[float, ...]]
    bic_by_cardinality: dict[str, float]
    aicc_by_cardinality: dict[str, float]
    log_likelihood_by_cardinality: dict[str, float]
    selected_cardinality_key: str
    selected_bic: float
    bic_margin_to_runner_up: float
    n_observations: int
    n_candidates_by_isotope: dict[str, int]


@dataclass(frozen=True)
class OffGridSparsePoissonRefinement:
    """Store continuous off-grid refinement results for selected candidates."""

    available: bool
    reason: str
    positions: tuple[tuple[float, ...], ...]
    strengths: tuple[float, ...]
    nuisance_strengths: tuple[float, ...]
    log_likelihood: float
    bic: float
    improvement_log_likelihood: float
    success: bool
    iterations: int


def _empty_evidence(reason: str, *, max_sources: int = 0) -> SparsePoissonEvidence:
    """Return an unavailable evidence payload with finite empty fields."""
    length = max(1, int(max_sources) + 1)
    return SparsePoissonEvidence(
        available=False,
        reason=str(reason),
        selected_count=0,
        selected_indices=(),
        selected_strengths=(),
        bic_by_count=tuple(float("inf") for _ in range(length)),
        aicc_by_count=tuple(float("inf") for _ in range(length)),
        log_likelihood_by_count=tuple(float("-inf") for _ in range(length)),
        heldout_deviance_by_count=tuple(float("inf") for _ in range(length)),
        best_bic_count=0,
        best_aicc_count=0,
        best_heldout_count=None,
        bic_gap_to_simpler=float("inf"),
        bic_gap_to_more_complex=float("inf"),
        bic_gap_to_previous_count=float("inf"),
        bic_gap_to_next_count=float("inf"),
        bic_margin_to_runner_up=float("inf"),
        condition_number=1.0,
        selected_max_response_correlation=0.0,
        n_observations=0,
        n_candidates=0,
        evaluated_candidate_count=0,
        candidate_indices=(),
    )


def _as_count_vector(values: NDArray[np.float64] | Sequence[float]) -> NDArray[np.float64]:
    """Return a non-negative one-dimensional count vector."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.zeros(0, dtype=float)
    return np.maximum(np.where(np.isfinite(arr), arr, 0.0), 0.0)


def _as_background_vector(
    background: float | NDArray[np.float64] | Sequence[float],
    count: int,
) -> NDArray[np.float64]:
    """Return a non-negative background vector with one value per observation."""
    arr = np.asarray(background, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.zeros(max(int(count), 0), dtype=float)
    if arr.size == 1 and int(count) != 1:
        arr = np.full(max(int(count), 0), float(arr[0]), dtype=float)
    if arr.size != int(count):
        raise ValueError("background must be scalar or one value per observation.")
    return np.maximum(np.where(np.isfinite(arr), arr, 0.0), 0.0)


def _validate_response_matrix(
    response_matrix: NDArray[np.float64] | Sequence[Sequence[float]],
    observation_count: int,
) -> NDArray[np.float64]:
    """Return a non-negative response matrix with observations by candidates."""
    matrix = np.asarray(response_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("response_matrix must be two-dimensional.")
    if matrix.shape[0] != int(observation_count):
        raise ValueError("response_matrix row count must match observations.")
    return np.maximum(np.where(np.isfinite(matrix), matrix, 0.0), 0.0)


def _validate_optional_response_matrix(
    response_matrix: NDArray[np.float64] | Sequence[Sequence[float]] | None,
    observation_count: int,
    *,
    name: str,
) -> NDArray[np.float64]:
    """Return an optional non-negative response matrix with matching rows."""
    if response_matrix is None:
        return np.zeros((int(observation_count), 0), dtype=float)
    matrix = np.asarray(response_matrix, dtype=float)
    if matrix.size == 0:
        return np.zeros((int(observation_count), 0), dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")
    if matrix.shape[0] != int(observation_count):
        raise ValueError(f"{name} row count must match observations.")
    return np.maximum(np.where(np.isfinite(matrix), matrix, 0.0), 0.0)


def _as_spectral_background(
    background: float | NDArray[np.float64] | Sequence[float],
    measurement_count: int,
    bin_count: int,
) -> NDArray[np.float64]:
    """Return a flattened non-negative background for spectrum-bin evidence."""
    rows = int(measurement_count)
    bins = int(bin_count)
    arr = np.asarray(background, dtype=float)
    if arr.size == 0:
        return np.zeros(rows * bins, dtype=float)
    if arr.ndim == 0 or arr.size == 1:
        return np.full(rows * bins, float(arr.reshape(-1)[0]), dtype=float)
    if arr.ndim == 1 and arr.size == bins:
        return np.tile(arr.astype(float, copy=False), rows)
    if arr.ndim == 1 and arr.size == rows * bins:
        return arr.astype(float, copy=False).reshape(-1)
    if arr.shape == (rows, bins):
        return arr.astype(float, copy=False).reshape(-1)
    raise ValueError(
        "background_spectrum must be scalar, B, MxB, or flattened M*B values."
    )


def flatten_spectral_response_tensor(
    spectrum_counts: NDArray[np.float64] | Sequence[Sequence[float]],
    response_tensor: NDArray[np.float64],
    *,
    background_spectrum: float | NDArray[np.float64] | Sequence[float] = 0.0,
    isotope_names: Sequence[str] | None = None,
) -> SpectralResponseTensor:
    """
    Flatten station x shield x energy-bin responses into sparse-evidence rows.

    ``response_tensor`` may have shape ``M x B x C`` for one isotope or
    ``M x B x C x I`` for a joint isotope dictionary.  The returned response
    matrix has one row per measurement/bin and one column per candidate or
    candidate/isotope pair, preserving all spectral-bin Poisson information.
    """
    counts_mb = np.asarray(spectrum_counts, dtype=float)
    if counts_mb.ndim != 2:
        raise ValueError("spectrum_counts must have shape measurements x bins.")
    measurement_count, bin_count = counts_mb.shape
    response = np.asarray(response_tensor, dtype=float)
    if response.ndim not in {3, 4}:
        raise ValueError("response_tensor must be MxBxC or MxBxCxI.")
    if response.shape[0] != measurement_count or response.shape[1] != bin_count:
        raise ValueError("response_tensor measurement/bin dimensions must match counts.")
    candidate_count = int(response.shape[2])
    isotope_count = int(response.shape[3]) if response.ndim == 4 else 1
    if response.ndim == 4 and isotope_names is not None:
        if len(tuple(isotope_names)) != isotope_count:
            raise ValueError("isotope_names length must match response tensor isotopes.")
    if response.ndim == 3:
        response_matrix = response.reshape(measurement_count * bin_count, candidate_count)
        metadata = tuple(
            {"candidate_index": int(candidate_idx)}
            for candidate_idx in range(candidate_count)
        )
    else:
        response_matrix = response.reshape(
            measurement_count * bin_count,
            candidate_count * isotope_count,
        )
        names = (
            tuple(str(name) for name in isotope_names)
            if isotope_names is not None
            else tuple(str(idx) for idx in range(isotope_count))
        )
        metadata = tuple(
            {
                "candidate_index": int(candidate_idx),
                "isotope_index": int(isotope_idx),
                "isotope": str(names[isotope_idx]),
            }
            for candidate_idx in range(candidate_count)
            for isotope_idx in range(isotope_count)
        )
    counts = _as_count_vector(counts_mb.reshape(-1))
    background = _as_spectral_background(
        background_spectrum,
        measurement_count,
        bin_count,
    )
    background = np.maximum(np.where(np.isfinite(background), background, 0.0), 0.0)
    return SpectralResponseTensor(
        counts=counts,
        response_matrix=np.maximum(
            np.where(np.isfinite(response_matrix), response_matrix, 0.0),
            0.0,
        ),
        background=background,
        column_metadata=metadata,
        measurement_count=int(measurement_count),
        bin_count=int(bin_count),
        candidate_count=int(candidate_count),
        isotope_count=int(isotope_count),
    )


def _poisson_log_likelihood(
    counts: NDArray[np.float64],
    mean: NDArray[np.float64],
    *,
    eps: float,
) -> float:
    """Return Poisson log likelihood with constants independent of the model omitted."""
    mu = np.maximum(np.asarray(mean, dtype=float).reshape(-1), float(eps))
    z = np.maximum(np.asarray(counts, dtype=float).reshape(-1), 0.0)
    if mu.size != z.size:
        raise ValueError("mean and counts must have the same length.")
    return float(np.sum(z * np.log(mu) - mu))


def _poisson_deviance(
    counts: NDArray[np.float64],
    mean: NDArray[np.float64],
    *,
    eps: float,
) -> float:
    """Return the Poisson deviance for held-out predictive checks."""
    z = np.maximum(np.asarray(counts, dtype=float).reshape(-1), 0.0)
    mu = np.maximum(np.asarray(mean, dtype=float).reshape(-1), float(eps))
    if z.size != mu.size:
        raise ValueError("mean and counts must have the same length.")
    positive = z > 0.0
    terms = np.zeros_like(z, dtype=float)
    terms[positive] = z[positive] * np.log(
        z[positive] / np.maximum(mu[positive], float(eps))
    )
    return float(2.0 * np.sum(terms - z + mu))


def _bic(log_likelihood: float, observation_count: int, parameter_count: int) -> float:
    """Return the Bayesian information criterion for a fitted sparse model."""
    n_obs = max(int(observation_count), 1)
    params = max(int(parameter_count), 0)
    return float(-2.0 * float(log_likelihood) + params * np.log(float(n_obs)))


def _aicc(
    log_likelihood: float,
    observation_count: int,
    parameter_count: int,
) -> float:
    """Return the finite-sample corrected Akaike information criterion."""
    n_obs = max(int(observation_count), 1)
    params = max(int(parameter_count), 0)
    aic = float(-2.0 * float(log_likelihood) + 2.0 * params)
    denom = n_obs - params - 1
    if denom <= 0:
        return float("inf")
    return float(aic + (2.0 * params * (params + 1.0)) / float(denom))


def _fit_poisson_strengths(
    design: NDArray[np.float64],
    counts: NDArray[np.float64],
    background: NDArray[np.float64],
    *,
    iters: int,
    eps: float,
    q_max: float,
) -> NDArray[np.float64]:
    """Fit non-negative source strengths for fixed dictionary columns."""
    matrix = np.maximum(np.asarray(design, dtype=float), 0.0)
    z = np.maximum(np.asarray(counts, dtype=float).reshape(-1), 0.0)
    bg = np.maximum(np.asarray(background, dtype=float).reshape(-1), 0.0)
    if matrix.ndim != 2 or matrix.shape[0] != z.size or bg.size != z.size:
        raise ValueError("design, counts, and background shapes are inconsistent.")
    source_count = int(matrix.shape[1])
    if source_count <= 0:
        return np.zeros(0, dtype=float)
    column_sum = np.sum(matrix, axis=0)
    observable = column_sum > float(eps)
    signal = np.maximum(z - bg, 0.0)
    numerator = signal @ matrix
    denominator = np.sum(matrix * matrix, axis=0)
    q = np.divide(
        numerator,
        np.maximum(denominator, float(eps)),
        out=np.zeros(source_count, dtype=float),
        where=denominator > float(eps),
    )
    if not np.any(q > 0.0):
        signal_total = float(np.sum(signal))
        denom = max(float(np.sum(column_sum[observable])), float(eps))
        q[observable] = signal_total / denom
    q = np.where(observable, np.maximum(q, float(eps)), 0.0)
    if q_max > 0.0:
        q = np.minimum(q, float(q_max))
    for _ in range(max(0, int(iters))):
        mean = np.maximum(bg + matrix @ q, float(eps))
        ratio = np.divide(z, mean, out=np.zeros_like(z, dtype=float), where=mean > 0.0)
        step_num = matrix.T @ ratio
        step_den = np.maximum(column_sum, float(eps))
        q = q * np.clip(step_num / step_den, 0.0, np.inf)
        q = np.where(observable & np.isfinite(q), np.maximum(q, 0.0), 0.0)
        if q_max > 0.0:
            q = np.minimum(q, float(q_max))
    return np.maximum(np.where(np.isfinite(q), q, 0.0), 0.0)


def _fit_source_and_nuisance_strengths(
    source_design: NDArray[np.float64],
    nuisance_design: NDArray[np.float64],
    counts: NDArray[np.float64],
    background: NDArray[np.float64],
    *,
    iters: int,
    eps: float,
    q_max: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Fit source and low-dimensional nuisance strengths for one sparse model."""
    source = np.maximum(np.asarray(source_design, dtype=float), 0.0)
    nuisance = np.maximum(np.asarray(nuisance_design, dtype=float), 0.0)
    z = np.maximum(np.asarray(counts, dtype=float).reshape(-1), 0.0)
    bg = np.maximum(np.asarray(background, dtype=float).reshape(-1), 0.0)
    if source.ndim != 2 or source.shape[0] != z.size:
        raise ValueError("source_design rows must match counts.")
    if nuisance.ndim != 2 or nuisance.shape[0] != z.size:
        raise ValueError("nuisance_design rows must match counts.")
    if nuisance.shape[1] == 0 and source.shape[1] == 0:
        return (
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=float),
            np.maximum(bg, float(eps)),
        )
    if nuisance.shape[1] == 0:
        source_q = _fit_poisson_strengths(
            source,
            z,
            bg,
            iters=iters,
            eps=eps,
            q_max=q_max,
        )
        return source_q, np.zeros(0, dtype=float), np.maximum(bg + source @ source_q, eps)
    if source.shape[1] == 0:
        nuisance_q = _fit_poisson_strengths(
            nuisance,
            z,
            bg,
            iters=iters,
            eps=eps,
            q_max=0.0,
        )
        return (
            np.zeros(0, dtype=float),
            nuisance_q,
            np.maximum(bg + nuisance @ nuisance_q, eps),
        )
    combined = np.column_stack([nuisance, source])
    q = _fit_poisson_strengths(
        combined,
        z,
        bg,
        iters=iters,
        eps=eps,
        q_max=0.0,
    )
    nuisance_count = int(nuisance.shape[1])
    nuisance_q = q[:nuisance_count]
    source_q = q[nuisance_count:]
    if q_max > 0.0:
        source_q = np.minimum(source_q, float(q_max))
    mean = np.maximum(bg + nuisance @ nuisance_q + source @ source_q, eps)
    return source_q, nuisance_q, mean


def _candidate_prefilter_indices(
    response: NDArray[np.float64],
    counts: NDArray[np.float64],
    background: NDArray[np.float64],
    *,
    candidate_limit: int,
    eps: float,
) -> NDArray[np.int64]:
    """Return dictionary columns retained for sparse evidence scoring."""
    candidate_count = int(response.shape[1])
    limit = int(candidate_limit)
    if limit <= 0 or candidate_count <= limit:
        return np.arange(candidate_count, dtype=np.int64)
    signal = np.maximum(counts - background, 0.0)
    numerator = signal @ response
    denominator = np.sum(response * response, axis=0)
    q_hat = np.divide(
        numerator,
        np.maximum(denominator, float(eps)),
        out=np.zeros(candidate_count, dtype=float),
        where=denominator > float(eps),
    )
    scores = np.where(np.isfinite(numerator * q_hat), numerator * q_hat, 0.0)
    if limit >= candidate_count:
        return np.arange(candidate_count, dtype=np.int64)
    selected = np.argpartition(scores, -limit)[-limit:]
    selected = selected[np.argsort(scores[selected])[::-1]]
    return np.asarray(selected, dtype=np.int64)


def _normalized_columns(
    response: NDArray[np.float64],
    *,
    eps: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Return L2-normalized response columns and their active mask."""
    matrix = np.maximum(np.asarray(response, dtype=float), 0.0)
    norms = np.linalg.norm(matrix, axis=0)
    active = norms > max(float(eps), 1.0e-12)
    normalized = np.divide(
        matrix,
        np.maximum(norms[None, :], max(float(eps), 1.0e-12)),
        out=np.zeros_like(matrix, dtype=float),
        where=active[None, :],
    )
    return normalized, active


def _correlation_allowed_mask(
    normalized_response: NDArray[np.float64],
    active: NDArray[np.bool_],
    selected_local: Sequence[int],
    *,
    threshold: float,
) -> NDArray[np.bool_]:
    """Return candidate mask after pruning columns collinear with selected columns."""
    allowed = np.asarray(active, dtype=bool).copy()
    if not selected_local or not 0.0 < float(threshold) < 1.0:
        return allowed
    selected = np.asarray(selected_local, dtype=int)
    selected_norm = normalized_response[:, selected]
    correlations = np.abs(normalized_response.T @ selected_norm)
    max_corr = np.max(correlations, axis=1) if correlations.size else np.zeros(allowed.size)
    allowed &= max_corr < float(threshold)
    allowed[selected] = False
    return allowed


def _select_next_candidate(
    response: NDArray[np.float64],
    counts: NDArray[np.float64],
    current_mean: NDArray[np.float64],
    allowed: NDArray[np.bool_],
    *,
    eps: float,
) -> int | None:
    """Return the best next dictionary column under one-step Poisson improvement."""
    valid = np.asarray(allowed, dtype=bool).reshape(-1)
    if valid.size != response.shape[1] or not np.any(valid):
        return None
    residual = np.maximum(counts - current_mean, 0.0)
    valid_indices = np.flatnonzero(valid)
    target_elements = 2_000_000
    chunk_cols = max(1, min(valid_indices.size, target_elements // max(counts.size, 1)))
    best_idx: int | None = None
    best_ll = float("-inf")
    any_positive = False
    for start in range(0, valid_indices.size, chunk_cols):
        chunk_indices = valid_indices[start : start + chunk_cols]
        candidate_response = response[:, chunk_indices]
        numerator = residual @ candidate_response
        denominator = np.einsum("ij,ij->j", candidate_response, candidate_response)
        q_hat = np.divide(
            numerator,
            np.maximum(denominator, float(eps)),
            out=np.zeros(candidate_response.shape[1], dtype=float),
            where=denominator > float(eps),
        )
        q_hat = np.maximum(np.where(np.isfinite(q_hat), q_hat, 0.0), 0.0)
        positive = q_hat > 0.0
        if not np.any(positive):
            continue
        any_positive = True
        active_response = candidate_response[:, positive]
        active_q = q_hat[positive]
        trial_mean = np.maximum(
            current_mean[:, None] + active_response * active_q[None, :],
            float(eps),
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            ll_values = np.sum(counts[:, None] * np.log(trial_mean) - trial_mean, axis=0)
        ll_values = np.where(np.isfinite(ll_values), ll_values, -np.inf)
        best_local = int(np.argmax(ll_values))
        candidate_ll = float(ll_values[best_local])
        if candidate_ll > best_ll:
            best_ll = candidate_ll
            best_idx = int(chunk_indices[positive][best_local])
    if not any_positive or best_idx is None or not np.isfinite(best_ll):
        return None
    return best_idx


def _design_condition_number(design: NDArray[np.float64], *, eps: float) -> float:
    """Return a scale-normalized condition number for selected response columns."""
    matrix = np.maximum(np.asarray(design, dtype=float), 0.0)
    if matrix.ndim != 2 or matrix.shape[1] <= 1:
        return 1.0
    normalized, active = _normalized_columns(matrix, eps=eps)
    if np.count_nonzero(active) <= 1:
        return float("inf")
    try:
        singular_values = np.linalg.svd(normalized[:, active], compute_uv=False)
    except np.linalg.LinAlgError:
        return float("inf")
    positive = singular_values[singular_values > max(float(eps), 1.0e-12)]
    if positive.size == 0:
        return float("inf")
    return float(np.max(positive) / max(float(np.min(positive)), float(eps)))


def _design_max_abs_correlation(design: NDArray[np.float64], *, eps: float) -> float:
    """Return the maximum absolute normalized response-column correlation."""
    matrix = np.maximum(np.asarray(design, dtype=float), 0.0)
    if matrix.ndim != 2 or matrix.shape[1] <= 1:
        return 0.0
    normalized, active = _normalized_columns(matrix, eps=eps)
    if np.count_nonzero(active) <= 1:
        return 0.0
    active_norm = normalized[:, active]
    corr = np.abs(active_norm.T @ active_norm)
    upper = np.triu_indices(corr.shape[0], k=1)
    if upper[0].size == 0:
        return 0.0
    return float(np.max(corr[upper]))


def _best_finite_count(values: Sequence[float]) -> int:
    """Return the index of the smallest finite criterion value."""
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return 0
    finite_indices = np.flatnonzero(finite)
    return int(finite_indices[int(np.argmin(arr[finite]))])


def _runner_up_margin(values: Sequence[float], best_count: int) -> float:
    """Return the criterion margin between the best and second-best counts."""
    arr = np.asarray(values, dtype=float)
    if best_count < 0 or best_count >= arr.size or not np.isfinite(arr[best_count]):
        return float("-inf")
    mask = np.ones(arr.size, dtype=bool)
    mask[int(best_count)] = False
    finite = arr[mask & np.isfinite(arr)]
    if finite.size == 0:
        return float("inf")
    return float(np.min(finite) - arr[int(best_count)])


def _json_float(value: float) -> float:
    """Return a finite float suitable for strict JSON serialization."""
    number = float(value)
    if np.isposinf(number):
        return 1.0e300
    if np.isneginf(number):
        return -1.0e300
    if not np.isfinite(number):
        return 0.0
    return number


def _criterion_gap(
    values: Sequence[float],
    best_count: int,
    *,
    direction: str,
) -> float:
    """Return lower-is-better criterion gap to simpler or more complex counts."""
    arr = np.asarray(values, dtype=float)
    if best_count < 0 or best_count >= arr.size or not np.isfinite(arr[best_count]):
        return float("-inf")
    if direction == "simpler":
        other = arr[: int(best_count)]
    elif direction == "more_complex":
        other = arr[int(best_count) + 1 :]
    elif direction == "previous":
        other = arr[int(best_count) - 1 : int(best_count)] if best_count > 0 else np.asarray([], dtype=float)
    elif direction == "next":
        other = arr[int(best_count) + 1 : int(best_count) + 2]
    else:
        raise ValueError("direction must be simpler, more_complex, previous, or next.")
    finite = other[np.isfinite(other)]
    if finite.size == 0:
        return float("inf")
    return float(np.min(finite) - arr[int(best_count)])


def _heldout_mask(count: int, stride: int) -> NDArray[np.bool_]:
    """Return a deterministic holdout mask for station/view evidence diagnostics."""
    if int(stride) <= 1 or int(count) <= 1:
        return np.zeros(max(int(count), 0), dtype=bool)
    indices = np.arange(max(int(count), 0), dtype=np.int64)
    return (indices + 1) % int(stride) == 0


def _heldout_deviance_by_count(
    response: NDArray[np.float64],
    counts: NDArray[np.float64],
    background: NDArray[np.float64],
    selected_by_count: Sequence[tuple[int, ...]],
    *,
    config: SparsePoissonEvidenceConfig,
    nuisance_response: NDArray[np.float64] | None = None,
) -> tuple[float, ...]:
    """Return held-out Poisson deviance for each selected source count."""
    holdout = _heldout_mask(counts.size, int(config.holdout_stride))
    if not np.any(holdout) or np.all(holdout):
        return tuple(float("inf") for _ in selected_by_count)
    train = ~holdout
    nuisance = _validate_optional_response_matrix(
        nuisance_response,
        counts.size,
        name="nuisance_response_matrix",
    )
    deviance: list[float] = []
    for indices in selected_by_count:
        idx = np.asarray(indices, dtype=int)
        source_train = (
            response[np.ix_(train, idx)]
            if idx.size
            else np.zeros((int(np.count_nonzero(train)), 0), dtype=float)
        )
        nuisance_train = (
            nuisance[train]
            if nuisance.shape[1]
            else np.zeros((int(np.count_nonzero(train)), 0), dtype=float)
        )
        source_q, nuisance_q, _mean_train = _fit_source_and_nuisance_strengths(
            source_train,
            nuisance_train,
            counts[train],
            background[train],
            iters=int(config.refit_iters),
            eps=float(config.eps),
            q_max=float(config.q_max),
        )
        source_holdout = (
            response[np.ix_(holdout, idx)]
            if idx.size
            else np.zeros((int(np.count_nonzero(holdout)), 0), dtype=float)
        )
        nuisance_holdout = (
            nuisance[holdout]
            if nuisance.shape[1]
            else np.zeros((int(np.count_nonzero(holdout)), 0), dtype=float)
        )
        mean = (
            background[holdout]
            + source_holdout @ source_q
            + nuisance_holdout @ nuisance_q
        )
        deviance.append(
            _poisson_deviance(counts[holdout], mean, eps=float(config.eps))
        )
    return tuple(float(value) for value in deviance)


def fit_sparse_poisson_evidence(
    counts: NDArray[np.float64] | Sequence[float],
    response_matrix: NDArray[np.float64] | Sequence[Sequence[float]],
    *,
    background: float | NDArray[np.float64] | Sequence[float] = 0.0,
    nuisance_response_matrix: NDArray[np.float64] | Sequence[Sequence[float]] | None = None,
    column_metadata: Sequence[Mapping[str, Any]] | None = None,
    config: SparsePoissonEvidenceConfig,
) -> SparsePoissonEvidence:
    """Fit all-history sparse Poisson evidence over K=0..Kmax source counts."""
    max_sources = max(0, int(config.max_sources))
    z = _as_count_vector(counts)
    if z.size == 0:
        return _empty_evidence("no_observations", max_sources=max_sources)
    response_all = _validate_response_matrix(response_matrix, z.size)
    if response_all.shape[1] == 0:
        return _empty_evidence("no_candidates", max_sources=max_sources)
    bg = _as_background_vector(background, z.size)
    nuisance = _validate_optional_response_matrix(
        nuisance_response_matrix,
        z.size,
        name="nuisance_response_matrix",
    )
    eps = max(float(config.eps), 1.0e-12)
    candidate_indices = _candidate_prefilter_indices(
        response_all,
        z,
        bg,
        candidate_limit=int(config.candidate_limit),
        eps=eps,
    )
    response = response_all[:, candidate_indices]
    evaluated_count = int(response.shape[1])
    if evaluated_count == 0:
        return _empty_evidence("no_evaluated_candidates", max_sources=max_sources)
    normalized_response, active = _normalized_columns(response, eps=eps)
    bic_values: list[float] = []
    aicc_values: list[float] = []
    ll_values: list[float] = []
    selected_by_count: list[tuple[int, ...]] = []
    strengths_by_count: list[tuple[float, ...]] = []
    nuisance_by_count: list[tuple[float, ...]] = []
    selected_local: list[int] = []
    nuisance_param_count = max(
        int(config.nuisance_parameter_count),
        int(nuisance.shape[1]) if nuisance.shape[1] else 0,
    )

    _source_q0, nuisance_q0, mean0 = _fit_source_and_nuisance_strengths(
        np.zeros((z.size, 0), dtype=float),
        nuisance,
        z,
        bg,
        iters=int(config.refit_iters),
        eps=eps,
        q_max=float(config.q_max),
    )
    ll0 = _poisson_log_likelihood(z, mean0, eps=eps)
    ll_values.append(ll0)
    bic_values.append(_bic(ll0, z.size, nuisance_param_count))
    aicc_values.append(_aicc(ll0, z.size, nuisance_param_count))
    selected_by_count.append(())
    strengths_by_count.append(())
    nuisance_by_count.append(tuple(float(value) for value in nuisance_q0))
    current_mean = mean0.copy()

    for k in range(1, max_sources + 1):
        allowed = _correlation_allowed_mask(
            normalized_response,
            active,
            selected_local,
            threshold=float(config.correlation_prune_threshold),
        )
        if selected_local:
            allowed[np.asarray(selected_local, dtype=int)] = False
        next_idx = _select_next_candidate(
            response,
            z,
            current_mean,
            allowed,
            eps=eps,
        )
        if next_idx is None:
            ll_values.append(float("-inf"))
            bic_values.append(float("inf"))
            aicc_values.append(float("inf"))
            selected_by_count.append(tuple(candidate_indices[selected_local].tolist()))
            strengths_by_count.append(())
            nuisance_by_count.append(())
            continue
        selected_local.append(int(next_idx))
        selected_design = response[:, np.asarray(selected_local, dtype=int)]
        q, nuisance_q, current_mean = _fit_source_and_nuisance_strengths(
            selected_design,
            nuisance,
            z,
            bg,
            iters=int(config.refit_iters),
            eps=eps,
            q_max=float(config.q_max),
        )
        ll_value = _poisson_log_likelihood(z, current_mean, eps=eps)
        params = nuisance_param_count + max(0, int(config.parameter_count_per_source)) * k
        ll_values.append(float(ll_value))
        bic_values.append(_bic(ll_value, z.size, params))
        aicc_values.append(_aicc(ll_value, z.size, params))
        selected_by_count.append(tuple(int(candidate_indices[idx]) for idx in selected_local))
        strengths_by_count.append(tuple(float(value) for value in q))
        nuisance_by_count.append(tuple(float(value) for value in nuisance_q))

    best_bic_count = _best_finite_count(bic_values)
    best_aicc_count = _best_finite_count(aicc_values)
    heldout_deviance = _heldout_deviance_by_count(
        response_all,
        z,
        bg,
        selected_by_count,
        config=config,
        nuisance_response=nuisance,
    )
    finite_holdout = np.isfinite(np.asarray(heldout_deviance, dtype=float))
    best_heldout_count = (
        int(np.argmin(np.asarray(heldout_deviance, dtype=float)))
        if np.any(finite_holdout)
        else None
    )
    best_indices = selected_by_count[best_bic_count]
    best_strengths = strengths_by_count[best_bic_count]
    best_nuisance_strengths = nuisance_by_count[best_bic_count]
    metadata_by_column: tuple[dict[str, Any], ...] = ()
    if column_metadata is not None:
        metadata_source = tuple(dict(item) for item in column_metadata)
        metadata_by_column = tuple(
            metadata_source[idx]
            for idx in best_indices
            if 0 <= int(idx) < len(metadata_source)
        )
    selected_design_full = (
        response_all[:, np.asarray(best_indices, dtype=int)]
        if best_indices
        else np.zeros((z.size, 0), dtype=float)
    )
    return SparsePoissonEvidence(
        available=True,
        reason="ok",
        selected_count=int(best_bic_count),
        selected_indices=tuple(int(idx) for idx in best_indices),
        selected_strengths=tuple(float(value) for value in best_strengths),
        bic_by_count=tuple(float(value) for value in bic_values),
        aicc_by_count=tuple(float(value) for value in aicc_values),
        log_likelihood_by_count=tuple(float(value) for value in ll_values),
        heldout_deviance_by_count=heldout_deviance,
        best_bic_count=int(best_bic_count),
        best_aicc_count=int(best_aicc_count),
        best_heldout_count=best_heldout_count,
        bic_gap_to_simpler=_criterion_gap(
            bic_values,
            best_bic_count,
            direction="simpler",
        ),
        bic_gap_to_more_complex=_criterion_gap(
            bic_values,
            best_bic_count,
            direction="more_complex",
        ),
        bic_gap_to_previous_count=_criterion_gap(
            bic_values,
            best_bic_count,
            direction="previous",
        ),
        bic_gap_to_next_count=_criterion_gap(
            bic_values,
            best_bic_count,
            direction="next",
        ),
        bic_margin_to_runner_up=_runner_up_margin(bic_values, best_bic_count),
        condition_number=_design_condition_number(selected_design_full, eps=eps),
        selected_max_response_correlation=_design_max_abs_correlation(
            selected_design_full,
            eps=eps,
        ),
        n_observations=int(z.size),
        n_candidates=int(response_all.shape[1]),
        evaluated_candidate_count=int(evaluated_count),
        candidate_indices=tuple(int(idx) for idx in candidate_indices),
        selected_nuisance_strengths=tuple(
            float(value) for value in best_nuisance_strengths
        ),
        selected_column_metadata=metadata_by_column,
    )


def fit_sparse_poisson_spectral_evidence(
    spectrum_counts: NDArray[np.float64] | Sequence[Sequence[float]],
    response_tensor: NDArray[np.float64],
    *,
    background_spectrum: float | NDArray[np.float64] | Sequence[float] = 0.0,
    isotope_names: Sequence[str] | None = None,
    nuisance_response_tensor: NDArray[np.float64] | None = None,
    config: SparsePoissonEvidenceConfig,
) -> SparsePoissonEvidence:
    """Fit sparse Poisson evidence directly on measured spectrum bins."""
    flattened = flatten_spectral_response_tensor(
        spectrum_counts,
        response_tensor,
        background_spectrum=background_spectrum,
        isotope_names=isotope_names,
    )
    nuisance_matrix = None
    if nuisance_response_tensor is not None:
        nuisance_flat = np.asarray(nuisance_response_tensor, dtype=float)
        if nuisance_flat.ndim == 3:
            if nuisance_flat.shape[:2] != (
                flattened.measurement_count,
                flattened.bin_count,
            ):
                raise ValueError("nuisance_response_tensor dimensions must match spectra.")
            nuisance_matrix = nuisance_flat.reshape(flattened.counts.size, -1)
        elif nuisance_flat.ndim == 2:
            if nuisance_flat.shape[0] != flattened.counts.size:
                raise ValueError("nuisance_response_tensor rows must match flattened bins.")
            nuisance_matrix = nuisance_flat
        else:
            raise ValueError("nuisance_response_tensor must be MxBxR or flattened rows.")
    evidence = fit_sparse_poisson_evidence(
        flattened.counts,
        flattened.response_matrix,
        background=flattened.background,
        nuisance_response_matrix=nuisance_matrix,
        column_metadata=flattened.column_metadata,
        config=config,
    )
    return replace(
        evidence,
        method="spectral_bin_sparse_poisson",
    )


def _cardinality_key(isotopes: Sequence[str], counts: Sequence[int]) -> str:
    """Return a stable string key for a joint cardinality vector."""
    return "|".join(
        f"{str(isotope)}:{int(count)}"
        for isotope, count in zip(isotopes, counts)
    )


def _joint_column_blocks(
    response_by_isotope: Mapping[str, NDArray[np.float64] | Sequence[Sequence[float]]],
    observation_count: int,
) -> tuple[list[str], NDArray[np.float64], dict[str, slice], dict[str, int]]:
    """Return concatenated isotope response columns and per-isotope blocks."""
    isotopes = [str(isotope) for isotope in response_by_isotope]
    blocks: list[NDArray[np.float64]] = []
    slices: dict[str, slice] = {}
    counts: dict[str, int] = {}
    start = 0
    for isotope in isotopes:
        matrix = _validate_response_matrix(
            response_by_isotope[isotope],
            observation_count,
        )
        stop = start + int(matrix.shape[1])
        blocks.append(matrix)
        slices[isotope] = slice(start, stop)
        counts[isotope] = int(matrix.shape[1])
        start = stop
    if not blocks:
        return [], np.zeros((observation_count, 0), dtype=float), {}, {}
    return isotopes, np.column_stack(blocks), slices, counts


def _prefilter_joint_response_by_isotope(
    response_by_isotope: Mapping[str, NDArray[np.float64] | Sequence[Sequence[float]]],
    counts: NDArray[np.float64],
    background: NDArray[np.float64],
    *,
    candidate_limit: int,
    eps: float,
) -> tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.int64]], dict[str, int]]:
    """Return per-isotope candidate-prefiltered response matrices and index maps."""
    isotope_names = [str(isotope) for isotope in response_by_isotope]
    isotope_count = max(len(isotope_names), 1)
    total_limit = max(0, int(candidate_limit))
    per_isotope_limit = (
        max(1, (total_limit + isotope_count - 1) // isotope_count)
        if total_limit > 0
        else 0
    )
    filtered: dict[str, NDArray[np.float64]] = {}
    index_maps: dict[str, NDArray[np.int64]] = {}
    original_counts: dict[str, int] = {}
    for isotope in isotope_names:
        matrix = _validate_response_matrix(
            response_by_isotope[isotope],
            counts.size,
        )
        original_counts[isotope] = int(matrix.shape[1])
        if matrix.shape[1] == 0:
            filtered[isotope] = matrix
            index_maps[isotope] = np.zeros(0, dtype=np.int64)
            continue
        selected = _candidate_prefilter_indices(
            matrix,
            counts,
            background,
            candidate_limit=per_isotope_limit,
            eps=max(float(eps), 1.0e-12),
        )
        filtered[isotope] = matrix[:, selected]
        index_maps[isotope] = np.asarray(selected, dtype=np.int64)
    return filtered, index_maps, original_counts


def _select_constrained_joint_columns(
    response: NDArray[np.float64],
    counts: NDArray[np.float64],
    background: NDArray[np.float64],
    nuisance: NDArray[np.float64],
    isotope_slices: Mapping[str, slice],
    target_counts: Mapping[str, int],
    *,
    config: SparsePoissonEvidenceConfig,
    fit_cache: dict[
        tuple[int, ...],
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    ]
    | None = None,
    selection_cache: dict[tuple[tuple[int, ...], tuple[str, ...]], int | None] | None = None,
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...], float]:
    """Greedily select columns while enforcing a joint isotope cardinality vector."""
    eps = max(float(config.eps), 1.0e-12)
    normalized, active = _normalized_columns(response, eps=eps)
    selected: list[int] = []
    remaining = {str(key): max(0, int(value)) for key, value in target_counts.items()}
    fit_states = fit_cache if fit_cache is not None else {}
    selection_states = selection_cache if selection_cache is not None else {}

    def _fit_for_selected(
        selected_indices: Sequence[int],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return profiled strengths and mean for a selected column tuple."""
        key = tuple(int(idx) for idx in selected_indices)
        cached = fit_states.get(key)
        if cached is not None:
            return cached
        selected_design = (
            response[:, np.asarray(key, dtype=int)]
            if key
            else np.zeros((counts.size, 0), dtype=float)
        )
        state = _fit_source_and_nuisance_strengths(
            selected_design,
            nuisance,
            counts,
            background,
            iters=int(config.refit_iters),
            eps=eps,
            q_max=float(config.q_max),
        )
        fit_states[key] = state
        return state

    source_q, nuisance_q, current_mean = _fit_for_selected(())
    while sum(remaining.values()) > 0:
        isotope_allowed = np.zeros(response.shape[1], dtype=bool)
        allowed_isotopes: list[str] = []
        for isotope, remain in remaining.items():
            if remain <= 0:
                continue
            allowed_isotopes.append(str(isotope))
            block = isotope_slices[str(isotope)]
            isotope_allowed[block] = True
        selection_key = (
            tuple(int(idx) for idx in selected),
            tuple(sorted(allowed_isotopes)),
        )
        if selection_key in selection_states:
            next_idx = selection_states[selection_key]
        else:
            allowed = _correlation_allowed_mask(
                normalized,
                active,
                selected,
                threshold=float(config.correlation_prune_threshold),
            )
            if selected:
                allowed[np.asarray(selected, dtype=int)] = False
            allowed &= isotope_allowed
            next_idx = _select_next_candidate(
                response,
                counts,
                current_mean,
                allowed,
                eps=eps,
            )
            selection_states[selection_key] = next_idx
        if next_idx is None:
            return (), (), tuple(float(value) for value in nuisance_q), float("-inf")
        selected.append(int(next_idx))
        for isotope, block in isotope_slices.items():
            if block.start <= int(next_idx) < block.stop:
                remaining[str(isotope)] -= 1
                break
        source_q, nuisance_q, current_mean = _fit_for_selected(selected)
    ll_value = _poisson_log_likelihood(counts, current_mean, eps=eps)
    return (
        tuple(int(idx) for idx in selected),
        tuple(float(value) for value in source_q),
        tuple(float(value) for value in nuisance_q),
        float(ll_value),
    )


def fit_joint_sparse_poisson_evidence(
    counts: NDArray[np.float64] | Sequence[float],
    response_by_isotope: Mapping[str, NDArray[np.float64] | Sequence[Sequence[float]]],
    *,
    max_sources_by_isotope: Mapping[str, int],
    background: float | NDArray[np.float64] | Sequence[float] = 0.0,
    nuisance_response_matrix: NDArray[np.float64] | Sequence[Sequence[float]] | None = None,
    config: SparsePoissonEvidenceConfig,
) -> JointSparsePoissonEvidence:
    """Fit joint sparse evidence over all isotope cardinality vectors."""
    z = _as_count_vector(counts)
    if z.size == 0:
        return JointSparsePoissonEvidence(
            available=False,
            reason="no_observations",
            selected_counts_by_isotope={},
            selected_indices_by_isotope={},
            selected_strengths_by_isotope={},
            bic_by_cardinality={},
            aicc_by_cardinality={},
            log_likelihood_by_cardinality={},
            selected_cardinality_key="",
            selected_bic=float("inf"),
            bic_margin_to_runner_up=float("-inf"),
            n_observations=0,
            n_candidates_by_isotope={},
        )
    bg = _as_background_vector(background, z.size)
    eps = max(float(config.eps), 1.0e-12)
    filtered_response, original_index_maps, original_candidate_counts = (
        _prefilter_joint_response_by_isotope(
            response_by_isotope,
            z,
            bg,
            candidate_limit=int(config.candidate_limit),
            eps=eps,
        )
    )
    isotopes, response, isotope_slices, _filtered_candidate_counts = _joint_column_blocks(
        filtered_response,
        z.size,
    )
    if response.shape[1] == 0:
        return JointSparsePoissonEvidence(
            available=False,
            reason="no_candidates",
            selected_counts_by_isotope={isotope: 0 for isotope in isotopes},
            selected_indices_by_isotope={isotope: () for isotope in isotopes},
            selected_strengths_by_isotope={isotope: () for isotope in isotopes},
            bic_by_cardinality={},
            aicc_by_cardinality={},
            log_likelihood_by_cardinality={},
            selected_cardinality_key="",
            selected_bic=float("inf"),
            bic_margin_to_runner_up=float("-inf"),
            n_observations=int(z.size),
            n_candidates_by_isotope=original_candidate_counts,
        )
    nuisance = _validate_optional_response_matrix(
        nuisance_response_matrix,
        z.size,
        name="nuisance_response_matrix",
    )
    nuisance_param_count = max(
        int(config.nuisance_parameter_count),
        int(nuisance.shape[1]) if nuisance.shape[1] else 0,
    )
    max_counts = [
        max(0, int(max_sources_by_isotope.get(isotope, int(config.max_sources))))
        for isotope in isotopes
    ]
    bic_by_key: dict[str, float] = {}
    aicc_by_key: dict[str, float] = {}
    ll_by_key: dict[str, float] = {}
    selected_by_key: dict[str, tuple[int, ...]] = {}
    strengths_by_key: dict[str, tuple[float, ...]] = {}
    fit_cache: dict[
        tuple[int, ...],
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    ] = {}
    selection_cache: dict[tuple[tuple[int, ...], tuple[str, ...]], int | None] = {}
    for count_vector in itertools.product(
        *[range(max_count + 1) for max_count in max_counts]
    ):
        target = {
            isotope: int(count)
            for isotope, count in zip(isotopes, count_vector)
        }
        key = _cardinality_key(isotopes, count_vector)
        selected, strengths, _nuisance_strengths, ll_value = (
            _select_constrained_joint_columns(
                response,
                z,
                bg,
                nuisance,
                isotope_slices,
                target,
                config=config,
                fit_cache=fit_cache,
                selection_cache=selection_cache,
            )
        )
        parameter_count = nuisance_param_count + (
            max(0, int(config.parameter_count_per_source)) * sum(count_vector)
        )
        if sum(count_vector) > 0 and not selected:
            bic_by_key[key] = float("inf")
            aicc_by_key[key] = float("inf")
            ll_by_key[key] = float("-inf")
            selected_by_key[key] = ()
            strengths_by_key[key] = ()
            continue
        bic_by_key[key] = _bic(ll_value, z.size, parameter_count)
        aicc_by_key[key] = _aicc(ll_value, z.size, parameter_count)
        ll_by_key[key] = float(ll_value)
        selected_by_key[key] = selected
        strengths_by_key[key] = strengths
    finite_items = [
        (key, value)
        for key, value in bic_by_key.items()
        if np.isfinite(float(value))
    ]
    if not finite_items:
        best_key = ""
        best_bic = float("inf")
        margin = float("-inf")
    else:
        best_key, best_bic = min(finite_items, key=lambda item: item[1])
        runner_up = [value for key, value in finite_items if key != best_key]
        margin = (
            float(min(runner_up) - best_bic)
            if runner_up
            else float("inf")
        )
    selected_indices_by_isotope: dict[str, tuple[int, ...]] = {
        isotope: () for isotope in isotopes
    }
    selected_strengths_by_isotope: dict[str, tuple[float, ...]] = {
        isotope: () for isotope in isotopes
    }
    selected_counts_by_isotope = {isotope: 0 for isotope in isotopes}
    selected_global = selected_by_key.get(best_key, ())
    selected_strengths = strengths_by_key.get(best_key, ())
    for global_idx, strength in zip(selected_global, selected_strengths):
        for isotope, block in isotope_slices.items():
            if block.start <= int(global_idx) < block.stop:
                local_idx = int(global_idx) - int(block.start)
                original_indices = original_index_maps.get(
                    isotope,
                    np.arange(block.stop - block.start, dtype=np.int64),
                )
                original_idx = (
                    int(original_indices[local_idx])
                    if 0 <= local_idx < int(original_indices.size)
                    else int(local_idx)
                )
                selected_indices_by_isotope[isotope] = (
                    *selected_indices_by_isotope[isotope],
                    original_idx,
                )
                selected_strengths_by_isotope[isotope] = (
                    *selected_strengths_by_isotope[isotope],
                    float(strength),
                )
                selected_counts_by_isotope[isotope] += 1
                break
    return JointSparsePoissonEvidence(
        available=bool(best_key),
        reason="ok" if best_key else "no_finite_models",
        selected_counts_by_isotope=selected_counts_by_isotope,
        selected_indices_by_isotope=selected_indices_by_isotope,
        selected_strengths_by_isotope=selected_strengths_by_isotope,
        bic_by_cardinality={key: float(value) for key, value in bic_by_key.items()},
        aicc_by_cardinality={key: float(value) for key, value in aicc_by_key.items()},
        log_likelihood_by_cardinality={
            key: float(value) for key, value in ll_by_key.items()
        },
        selected_cardinality_key=str(best_key),
        selected_bic=float(best_bic),
        bic_margin_to_runner_up=float(margin),
        n_observations=int(z.size),
        n_candidates_by_isotope=original_candidate_counts,
    )


def joint_sparse_poisson_evidence_to_diagnostics(
    evidence: JointSparsePoissonEvidence,
) -> dict[str, Any]:
    """Return JSON-safe diagnostics for joint multi-isotope sparse evidence."""
    return {
        "available": bool(evidence.available),
        "reason": str(evidence.reason),
        "method": "joint_multi_isotope_sparse_poisson",
        "selected_counts_by_isotope": {
            str(key): int(value)
            for key, value in evidence.selected_counts_by_isotope.items()
        },
        "selected_indices_by_isotope": {
            str(key): [int(idx) for idx in value]
            for key, value in evidence.selected_indices_by_isotope.items()
        },
        "selected_strengths_by_isotope": {
            str(key): [float(strength) for strength in value]
            for key, value in evidence.selected_strengths_by_isotope.items()
        },
        "selected_cardinality_key": str(evidence.selected_cardinality_key),
        "selected_bic": _json_float(evidence.selected_bic),
        "bic_margin_to_runner_up": _json_float(evidence.bic_margin_to_runner_up),
        "bic_by_cardinality": {
            str(key): _json_float(value)
            for key, value in evidence.bic_by_cardinality.items()
        },
        "aicc_by_cardinality": {
            str(key): _json_float(value)
            for key, value in evidence.aicc_by_cardinality.items()
        },
        "log_likelihood_by_cardinality": {
            str(key): _json_float(value)
            for key, value in evidence.log_likelihood_by_cardinality.items()
        },
        "measurement_count": int(evidence.n_observations),
        "candidate_count_by_isotope": {
            str(key): int(value)
            for key, value in evidence.n_candidates_by_isotope.items()
        },
    }


def refine_sparse_poisson_evidence_offgrid(
    counts: NDArray[np.float64] | Sequence[float],
    initial_positions: NDArray[np.float64] | Sequence[Sequence[float]],
    response_at_positions: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    *,
    background: float | NDArray[np.float64] | Sequence[float] = 0.0,
    nuisance_response_matrix: NDArray[np.float64] | Sequence[Sequence[float]] | None = None,
    bounds: Sequence[tuple[float, float]] | None = None,
    config: SparsePoissonEvidenceConfig,
    max_iter: int = 64,
) -> OffGridSparsePoissonRefinement:
    """
    Refine selected sparse-evidence source positions in continuous coordinates.

    The supplied ``response_at_positions`` callable must evaluate all selected
    positions as a batch and return an observation x source response matrix.
    This keeps the runtime path vectorized over selected sources; the optimizer
    only iterates over the small post-evidence parameter vector.
    """
    z = _as_count_vector(counts)
    positions0 = np.asarray(initial_positions, dtype=float)
    if positions0.ndim != 2 or positions0.shape[0] == 0:
        return OffGridSparsePoissonRefinement(
            available=False,
            reason="no_initial_positions",
            positions=(),
            strengths=(),
            nuisance_strengths=(),
            log_likelihood=float("-inf"),
            bic=float("inf"),
            improvement_log_likelihood=0.0,
            success=False,
            iterations=0,
        )
    bg = _as_background_vector(background, z.size)
    nuisance = _validate_optional_response_matrix(
        nuisance_response_matrix,
        z.size,
        name="nuisance_response_matrix",
    )
    eps = max(float(config.eps), 1.0e-12)
    initial_response = _validate_response_matrix(response_at_positions(positions0), z.size)
    initial_q, initial_nuisance_q, initial_mean = _fit_source_and_nuisance_strengths(
        initial_response,
        nuisance,
        z,
        bg,
        iters=int(config.refit_iters),
        eps=eps,
        q_max=float(config.q_max),
    )
    initial_ll = _poisson_log_likelihood(z, initial_mean, eps=eps)
    flat0 = positions0.reshape(-1)

    def _objective(flat_positions: NDArray[np.float64]) -> float:
        """Return the negative profiled Poisson log likelihood."""
        positions = np.asarray(flat_positions, dtype=float).reshape(positions0.shape)
        response = _validate_response_matrix(response_at_positions(positions), z.size)
        _q, _nuisance_q, mean = _fit_source_and_nuisance_strengths(
            response,
            nuisance,
            z,
            bg,
            iters=int(config.refit_iters),
            eps=eps,
            q_max=float(config.q_max),
        )
        return -_poisson_log_likelihood(z, mean, eps=eps)

    try:
        from scipy.optimize import minimize
    except ImportError:
        return OffGridSparsePoissonRefinement(
            available=False,
            reason="scipy_optimize_unavailable",
            positions=tuple(tuple(float(v) for v in row) for row in positions0),
            strengths=tuple(float(value) for value in initial_q),
            nuisance_strengths=tuple(float(value) for value in initial_nuisance_q),
            log_likelihood=float(initial_ll),
            bic=_bic(
                initial_ll,
                z.size,
                int(config.parameter_count_per_source) * positions0.shape[0],
            ),
            improvement_log_likelihood=0.0,
            success=False,
            iterations=0,
        )
    method = "Powell" if bounds is not None else "Nelder-Mead"
    options = {"maxiter": max(1, int(max_iter))}
    if method == "Powell":
        options.update({"xtol": 1.0e-3, "ftol": max(eps, 1.0e-6)})
    else:
        options.update({"xatol": 1.0e-3, "fatol": max(eps, 1.0e-6)})
    result = minimize(
        _objective,
        flat0,
        method=method,
        bounds=bounds,
        options=options,
    )
    refined_positions = np.asarray(result.x, dtype=float).reshape(positions0.shape)
    refined_response = _validate_response_matrix(
        response_at_positions(refined_positions),
        z.size,
    )
    refined_q, refined_nuisance_q, refined_mean = _fit_source_and_nuisance_strengths(
        refined_response,
        nuisance,
        z,
        bg,
        iters=int(config.refit_iters),
        eps=eps,
        q_max=float(config.q_max),
    )
    refined_ll = _poisson_log_likelihood(z, refined_mean, eps=eps)
    parameter_count = (
        int(config.parameter_count_per_source) * positions0.shape[0]
        + max(int(config.nuisance_parameter_count), int(nuisance.shape[1]))
    )
    return OffGridSparsePoissonRefinement(
        available=True,
        reason="ok",
        positions=tuple(
            tuple(float(value) for value in row)
            for row in np.asarray(refined_positions, dtype=float)
        ),
        strengths=tuple(float(value) for value in refined_q),
        nuisance_strengths=tuple(float(value) for value in refined_nuisance_q),
        log_likelihood=float(refined_ll),
        bic=_bic(refined_ll, z.size, parameter_count),
        improvement_log_likelihood=float(refined_ll - initial_ll),
        success=bool(result.success),
        iterations=int(getattr(result, "nit", 0)),
    )


def sparse_poisson_ambiguity_diagnostics(
    evidence: SparsePoissonEvidence,
    response_matrix: NDArray[np.float64] | Sequence[Sequence[float]],
    *,
    candidate_positions: NDArray[np.float64] | Sequence[Sequence[float]] | None = None,
    correlation_threshold: float = 0.98,
    bic_gap_threshold: float = 2.0,
    condition_number_threshold: float = 100.0,
) -> tuple[dict[str, Any], ...]:
    """Return source clusters that are not identifiable as separate points."""
    if not bool(evidence.available) or not evidence.selected_indices:
        return ()
    response = _validate_response_matrix(response_matrix, int(evidence.n_observations))
    if response.shape[1] == 0:
        return ()
    positions = None
    if candidate_positions is not None:
        pos_arr = np.asarray(candidate_positions, dtype=float)
        if pos_arr.ndim == 2 and pos_arr.shape[0] == response.shape[1]:
            positions = pos_arr
    normalized, active = _normalized_columns(response, eps=1.0e-12)
    selected = np.asarray(evidence.selected_indices, dtype=int)
    selected = selected[(selected >= 0) & (selected < response.shape[1])]
    clusters: list[dict[str, Any]] = []
    threshold = float(np.clip(float(correlation_threshold), 0.0, 1.0))
    for selected_idx in selected:
        if not bool(active[selected_idx]):
            continue
        correlations = np.abs(normalized.T @ normalized[:, selected_idx])
        members = np.flatnonzero(correlations >= threshold)
        if members.size <= 1:
            continue
        payload: dict[str, Any] = {
            "reason": "high_response_correlation",
            "identifiable": False,
            "selected_indices": [int(selected_idx)],
            "candidate_indices": [int(idx) for idx in members],
            "max_response_correlation": float(np.max(correlations[members])),
        }
        if positions is not None:
            member_pos = positions[members]
            payload["position_min"] = [
                float(value) for value in np.min(member_pos, axis=0)
            ]
            payload["position_max"] = [
                float(value) for value in np.max(member_pos, axis=0)
            ]
            payload["position_center"] = [
                float(value) for value in np.mean(member_pos, axis=0)
            ]
        clusters.append(payload)
    if (
        float(condition_number_threshold) > 0.0
        and np.isfinite(float(evidence.condition_number))
        and float(evidence.condition_number) > float(condition_number_threshold)
    ):
        clusters.append(
            {
                "reason": "ill_conditioned_selected_response",
                "identifiable": False,
                "selected_indices": [int(idx) for idx in selected],
                "condition_number": float(evidence.condition_number),
            }
        )
    if (
        float(bic_gap_threshold) > 0.0
        and np.isfinite(float(evidence.bic_margin_to_runner_up))
        and float(evidence.bic_margin_to_runner_up) < float(bic_gap_threshold)
    ):
        clusters.append(
            {
                "reason": "small_model_order_bic_gap",
                "identifiable": False,
                "selected_indices": [int(idx) for idx in selected],
                "bic_margin_to_runner_up": float(evidence.bic_margin_to_runner_up),
            }
        )
    return tuple(clusters)


def sparse_poisson_evidence_to_diagnostics(
    evidence: SparsePoissonEvidence,
) -> dict[str, Any]:
    """Return a JSON-safe diagnostic mapping for sparse Poisson evidence."""
    return {
        "available": bool(evidence.available),
        "reason": str(evidence.reason),
        "method": str(evidence.method),
        "candidate_count": int(evidence.n_candidates),
        "evaluated_candidate_count": int(evidence.evaluated_candidate_count),
        "measurement_count": int(evidence.n_observations),
        "selected_count": int(evidence.selected_count),
        "selected_indices": [int(value) for value in evidence.selected_indices],
        "selected_strengths": [float(value) for value in evidence.selected_strengths],
        "selected_nuisance_strengths": [
            float(value) for value in evidence.selected_nuisance_strengths
        ],
        "selected_column_metadata": [
            dict(value) for value in evidence.selected_column_metadata
        ],
        "best_bic_count": int(evidence.best_bic_count),
        "best_aicc_count": int(evidence.best_aicc_count),
        "best_heldout_count": (
            None
            if evidence.best_heldout_count is None
            else int(evidence.best_heldout_count)
        ),
        "bic_by_count": [_json_float(value) for value in evidence.bic_by_count],
        "aicc_by_count": [_json_float(value) for value in evidence.aicc_by_count],
        "log_likelihood_by_count": [
            _json_float(value) for value in evidence.log_likelihood_by_count
        ],
        "heldout_deviance_by_count": [
            _json_float(value) for value in evidence.heldout_deviance_by_count
        ],
        "bic_gap_to_simpler": _json_float(evidence.bic_gap_to_simpler),
        "bic_gap_to_more_complex": _json_float(evidence.bic_gap_to_more_complex),
        "bic_gap_to_previous_count": _json_float(evidence.bic_gap_to_previous_count),
        "bic_gap_to_next_count": _json_float(evidence.bic_gap_to_next_count),
        "bic_margin_to_runner_up": _json_float(evidence.bic_margin_to_runner_up),
        "criterion_margin_to_simpler": _json_float(evidence.bic_gap_to_simpler),
        "criterion_margin_to_runner_up": _json_float(
            evidence.bic_margin_to_runner_up
        ),
        "condition_number": _json_float(evidence.condition_number),
        "selected_max_response_correlation": _json_float(
            evidence.selected_max_response_correlation
        ),
        "ambiguity_clusters": [dict(value) for value in evidence.ambiguity_clusters],
        "candidate_indices": [int(value) for value in evidence.candidate_indices],
    }
