"""PF-independent Poisson reconstruction on an area-aware surface patch graph."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix

from .response_operator import ResponseOperator


@dataclass(frozen=True)
class SurfaceMapConfig:
    """Configure non-negative Poisson reconstruction with spatial regularization."""

    l1_weight: float = 0.0
    tv_weight: float = 0.0
    isotope_group_weight: float = 0.0
    nuisance_l1_weight: float = 0.0
    nuisance_l2_weight: float = 0.0
    nuisance_l2_weights: tuple[float, ...] = ()
    likelihood_family: str = "poisson"
    overdispersion_alpha: tuple[float, ...] = ()
    max_iterations: int = 4000
    tolerance: float = 1.0e-6
    objective_tolerance: float = 1.0e-7
    check_interval: int = 20
    step_safety: float = 0.95
    over_relaxation: float = 1.0
    min_mean: float = 1.0e-12

    def __post_init__(self) -> None:
        """Validate solver parameters without changing their physical meaning."""
        non_negative = {
            "l1_weight": self.l1_weight,
            "tv_weight": self.tv_weight,
            "isotope_group_weight": self.isotope_group_weight,
            "nuisance_l1_weight": self.nuisance_l1_weight,
            "nuisance_l2_weight": self.nuisance_l2_weight,
            "tolerance": self.tolerance,
            "objective_tolerance": self.objective_tolerance,
        }
        if any(
            not np.isfinite(value) or value < 0.0 for value in non_negative.values()
        ):
            raise ValueError(
                "Regularization weights and tolerances must be finite and non-negative."
            )
        if any(
            not np.isfinite(value) or float(value) < 0.0
            for value in self.nuisance_l2_weights
        ):
            raise ValueError("nuisance_l2_weights must be finite and non-negative.")
        if self.likelihood_family not in {"poisson", "negative_binomial"}:
            raise ValueError("likelihood_family must be poisson or negative_binomial.")
        if any(
            not np.isfinite(value) or float(value) < 0.0
            for value in self.overdispersion_alpha
        ):
            raise ValueError("overdispersion_alpha must be finite and non-negative.")
        if self.likelihood_family == "poisson" and any(
            float(value) != 0.0 for value in self.overdispersion_alpha
        ):
            raise ValueError("Poisson likelihood cannot have positive overdispersion.")
        if int(self.max_iterations) < 1:
            raise ValueError("max_iterations must be at least one.")
        if int(self.check_interval) < 1:
            raise ValueError("check_interval must be at least one.")
        if not np.isfinite(self.step_safety) or not 0.0 < self.step_safety < 1.0:
            raise ValueError("step_safety must lie strictly between zero and one.")
        if (
            not np.isfinite(self.over_relaxation)
            or not 0.0 <= self.over_relaxation <= 1.0
        ):
            raise ValueError("over_relaxation must lie between zero and one.")
        if not np.isfinite(self.min_mean) or self.min_mean <= 0.0:
            raise ValueError("min_mean must be finite and positive.")


@dataclass(frozen=True)
class SurfaceMapObjective:
    """Store the terms of the regularized Poisson surface-map objective."""

    total: float
    poisson_nll: float
    l1_penalty: float
    tv_penalty: float
    group_penalty: float
    nuisance_penalty: float
    deviance: float


@dataclass(frozen=True)
class SurfaceMapResult:
    """Store a reconstructed surface intensity map and convergence diagnostics."""

    densities_cps_1m_m2: NDArray[np.float64]
    integrated_strengths_cps_1m: NDArray[np.float64]
    nuisance_coefficients: NDArray[np.float64]
    expected_counts: NDArray[np.float64]
    objective: float
    poisson_nll: float
    l1_penalty: float
    tv_penalty: float
    group_penalty: float
    nuisance_penalty: float
    deviance: float
    converged: bool
    iterations: int
    relative_change: float
    relative_objective_change: float
    kkt_residual: float
    objective_history: tuple[float, ...]


@dataclass(frozen=True)
class _PreparedSurfaceMapProblem:
    """Store validated, flattened arrays used by the batched solver."""

    observed: NDArray[np.float64]
    response_by_density: NDArray[np.float64]
    response_by_integrated_strength: NDArray[np.float64]
    background: NDArray[np.float64]
    nuisance_response: NDArray[np.float64]
    patch_areas: NDArray[np.float64]
    column_areas: NDArray[np.float64]
    incidence: csr_matrix
    edge_weights: NDArray[np.float64]
    observation_shape: tuple[int, ...]
    patch_count: int
    isotope_count: int


def _nuisance_l2_vector(
    config: SurfaceMapConfig,
    count: int,
) -> NDArray[np.float64]:
    """Return one L2 shrinkage weight per nuisance coefficient."""
    if not config.nuisance_l2_weights:
        return np.full(count, float(config.nuisance_l2_weight), dtype=np.float64)
    weights = np.asarray(config.nuisance_l2_weights, dtype=np.float64)
    if weights.shape != (count,):
        raise ValueError("nuisance_l2_weights must match nuisance_response columns.")
    return weights


def _as_non_negative_vector(
    values: ArrayLike,
    *,
    name: str,
) -> NDArray[np.float64]:
    """Return a finite non-negative vector or raise for invalid physical inputs."""
    array = np.asarray(values, dtype=float).reshape(-1)
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(array < -1.0e-12):
        raise ValueError(f"{name} must be non-negative.")
    return np.maximum(array, 0.0)


def _broadcast_observation_vector(
    values: float | ArrayLike,
    observation_shape: tuple[int, ...],
    *,
    name: str,
) -> NDArray[np.float64]:
    """Return one finite non-negative value per flattened observation."""
    observation_count = int(np.prod(observation_shape, dtype=np.int64))
    array = np.asarray(values, dtype=float)
    if array.size == 1:
        vector = np.full(observation_count, float(array.reshape(-1)[0]), dtype=float)
    elif array.shape == observation_shape or array.size == observation_count:
        vector = array.reshape(-1).astype(float, copy=False)
    else:
        raise ValueError(
            f"{name} must be scalar, match observed_counts, or contain one value per observation."
        )
    return _as_non_negative_vector(vector, name=name)


def _flatten_response(
    observed_shape: tuple[int, ...],
    response: ArrayLike,
    patch_count: int,
) -> tuple[NDArray[np.float64], int]:
    """Flatten a shared-physics matrix or spectrum tensor to observations by columns."""
    response_array = np.asarray(response, dtype=float)
    prefix_rank = len(observed_shape)
    if response_array.ndim not in {prefix_rank + 1, prefix_rank + 2}:
        raise ValueError(
            "response must have shape observed_shape + (patches,) or "
            "observed_shape + (patches, isotopes)."
        )
    if tuple(response_array.shape[:prefix_rank]) != observed_shape:
        raise ValueError("response observation dimensions must match observed_counts.")
    if int(response_array.shape[prefix_rank]) != int(patch_count):
        raise ValueError("response patch dimension must match patch_areas_m2.")
    isotope_count = (
        int(response_array.shape[prefix_rank + 1])
        if response_array.ndim == prefix_rank + 2
        else 1
    )
    if isotope_count < 1:
        raise ValueError("response must contain at least one isotope channel.")
    if np.any(~np.isfinite(response_array)):
        raise ValueError("response must contain only finite values.")
    if np.any(response_array < -1.0e-12):
        raise ValueError("response must be non-negative.")
    observation_count = int(np.prod(observed_shape, dtype=np.int64))
    matrix = np.maximum(response_array, 0.0).reshape(
        observation_count,
        patch_count * isotope_count,
    )
    return matrix, isotope_count


def _flatten_nuisance_response(
    nuisance_response: ArrayLike | None,
    observation_shape: tuple[int, ...],
) -> NDArray[np.float64]:
    """Return an observations-by-nuisance non-negative design matrix."""
    observation_count = int(np.prod(observation_shape, dtype=np.int64))
    if nuisance_response is None:
        return np.zeros((observation_count, 0), dtype=float)
    array = np.asarray(nuisance_response, dtype=float)
    if array.size == 0:
        return np.zeros((observation_count, 0), dtype=float)
    if array.ndim == 2 and array.shape[0] == observation_count:
        matrix = array
    elif (
        array.ndim == len(observation_shape) + 1
        and tuple(array.shape[: len(observation_shape)]) == observation_shape
    ):
        matrix = array.reshape(observation_count, -1)
    else:
        raise ValueError(
            "nuisance_response must be observations x nuisance or "
            "observed_shape + (nuisance,)."
        )
    if np.any(~np.isfinite(matrix)):
        raise ValueError("nuisance_response must contain only finite values.")
    if np.any(matrix < -1.0e-12):
        raise ValueError("nuisance_response must be non-negative.")
    return np.maximum(matrix, 0.0)


def _canonical_graph(
    adjacency_edges: ArrayLike | None,
    adjacency_weights: ArrayLike | None,
    patch_count: int,
) -> tuple[csr_matrix, NDArray[np.float64]]:
    """Return a deduplicated oriented incidence matrix and summed edge weights."""
    if adjacency_edges is None:
        return csr_matrix((0, patch_count), dtype=float), np.zeros(0, dtype=float)
    edges = np.asarray(adjacency_edges, dtype=np.int64)
    if edges.size == 0:
        return csr_matrix((0, patch_count), dtype=float), np.zeros(0, dtype=float)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("adjacency_edges must have shape edges x 2.")
    if np.any(edges < 0) or np.any(edges >= int(patch_count)):
        raise ValueError("adjacency edge indices must refer to existing patches.")
    if np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("adjacency edges must connect distinct patches.")
    canonical = np.sort(edges, axis=1)
    unique_edges, inverse = np.unique(canonical, axis=0, return_inverse=True)
    if adjacency_weights is None:
        raw_weights = np.ones(edges.shape[0], dtype=float)
    else:
        raw_weights = _as_non_negative_vector(
            adjacency_weights,
            name="adjacency_weights",
        )
        if raw_weights.size != edges.shape[0]:
            raise ValueError("adjacency_weights must contain one value per edge.")
    weights = np.bincount(
        inverse,
        weights=raw_weights,
        minlength=unique_edges.shape[0],
    ).astype(float, copy=False)
    row_indices = np.repeat(np.arange(unique_edges.shape[0], dtype=np.int64), 2)
    column_indices = unique_edges.reshape(-1)
    values = np.tile(np.asarray([-1.0, 1.0], dtype=float), unique_edges.shape[0])
    incidence = csr_matrix(
        (values, (row_indices, column_indices)),
        shape=(unique_edges.shape[0], patch_count),
        dtype=float,
    )
    return incidence, weights


def _prepare_surface_map_problem(
    observed_counts: ArrayLike,
    response: ArrayLike,
    patch_areas_m2: ArrayLike,
    adjacency_edges: ArrayLike | None,
    adjacency_weights: ArrayLike | None,
    *,
    background: float | ArrayLike,
    nuisance_response: ArrayLike | None,
) -> _PreparedSurfaceMapProblem:
    """Validate and flatten all inputs while preserving candidate-isotope ordering."""
    observed_array = np.asarray(observed_counts, dtype=float)
    if observed_array.ndim < 1 or observed_array.size == 0:
        raise ValueError("observed_counts must contain at least one observation.")
    observed_shape = tuple(int(value) for value in observed_array.shape)
    observed = _as_non_negative_vector(observed_array, name="observed_counts")
    patch_areas = _as_non_negative_vector(patch_areas_m2, name="patch_areas_m2")
    if patch_areas.size == 0 or np.any(patch_areas <= 0.0):
        raise ValueError("patch_areas_m2 must contain positive patch areas.")
    response_integrated, isotope_count = _flatten_response(
        observed_shape,
        response,
        int(patch_areas.size),
    )
    column_areas = np.repeat(patch_areas, isotope_count)
    response_density = response_integrated * column_areas[None, :]
    background_vector = _broadcast_observation_vector(
        background,
        observed_shape,
        name="background",
    )
    nuisance_matrix = _flatten_nuisance_response(
        nuisance_response,
        observed_shape,
    )
    incidence, edge_weights = _canonical_graph(
        adjacency_edges,
        adjacency_weights,
        int(patch_areas.size),
    )
    return _PreparedSurfaceMapProblem(
        observed=observed,
        response_by_density=response_density,
        response_by_integrated_strength=response_integrated,
        background=background_vector,
        nuisance_response=nuisance_matrix,
        patch_areas=patch_areas,
        column_areas=column_areas,
        incidence=incidence,
        edge_weights=edge_weights,
        observation_shape=observed_shape,
        patch_count=int(patch_areas.size),
        isotope_count=int(isotope_count),
    )


def _poisson_nll(
    observed: NDArray[np.float64],
    expected: NDArray[np.float64],
    *,
    min_mean: float,
) -> float:
    """Return Poisson negative log likelihood with model-independent constants omitted."""
    mean = np.maximum(np.asarray(expected, dtype=float).reshape(-1), float(min_mean))
    counts = np.asarray(observed, dtype=float).reshape(-1)
    return float(np.sum(mean - counts * np.log(mean)))


def _poisson_deviance(
    observed: NDArray[np.float64],
    expected: NDArray[np.float64],
    *,
    min_mean: float,
) -> float:
    """Return the Poisson deviance from the saturated count model."""
    counts = np.asarray(observed, dtype=float).reshape(-1)
    mean = np.maximum(np.asarray(expected, dtype=float).reshape(-1), float(min_mean))
    positive = counts > 0.0
    log_terms = np.zeros_like(counts, dtype=float)
    log_terms[positive] = counts[positive] * np.log(counts[positive] / mean[positive])
    return float(2.0 * np.sum(log_terms - counts + mean))


def _objective_from_prepared(
    problem: _PreparedSurfaceMapProblem,
    densities: NDArray[np.float64],
    nuisance_coefficients: NDArray[np.float64],
    config: SurfaceMapConfig,
) -> tuple[SurfaceMapObjective, NDArray[np.float64]]:
    """Evaluate all objective terms for validated density and nuisance arrays."""
    density_matrix = np.asarray(densities, dtype=float).reshape(
        problem.patch_count,
        problem.isotope_count,
    )
    nuisance = np.asarray(nuisance_coefficients, dtype=float).reshape(-1)
    signal = problem.response_by_density @ density_matrix.reshape(-1)
    if nuisance.size:
        signal = signal + problem.nuisance_response @ nuisance
    expected = np.maximum(problem.background + signal, float(config.min_mean))
    poisson_nll = _poisson_nll(
        problem.observed,
        expected,
        min_mean=float(config.min_mean),
    )
    l1_penalty = float(config.l1_weight) * float(
        np.sum(problem.patch_areas[:, None] * density_matrix)
    )
    if problem.incidence.shape[0] and float(config.tv_weight) > 0.0:
        differences = problem.incidence @ density_matrix
        tv_penalty = float(config.tv_weight) * float(
            np.sum(problem.edge_weights[:, None] * np.abs(differences))
        )
    else:
        tv_penalty = 0.0
    group_penalty = float(config.isotope_group_weight) * float(
        np.sum(problem.patch_areas * np.linalg.norm(density_matrix, axis=1))
    )
    nuisance_l2 = _nuisance_l2_vector(config, nuisance.size)
    nuisance_penalty = float(config.nuisance_l1_weight) * float(
        np.sum(nuisance)
    ) + 0.5 * float(np.dot(nuisance_l2, nuisance * nuisance))
    objective = SurfaceMapObjective(
        total=float(
            poisson_nll + l1_penalty + tv_penalty + group_penalty + nuisance_penalty
        ),
        poisson_nll=float(poisson_nll),
        l1_penalty=float(l1_penalty),
        tv_penalty=float(tv_penalty),
        group_penalty=float(group_penalty),
        nuisance_penalty=float(nuisance_penalty),
        deviance=_poisson_deviance(
            problem.observed,
            expected,
            min_mean=float(config.min_mean),
        ),
    )
    return objective, expected


def evaluate_surface_map_objective(
    observed_counts: ArrayLike,
    response: ArrayLike,
    patch_areas_m2: ArrayLike,
    densities_cps_1m_m2: ArrayLike,
    adjacency_edges: ArrayLike | None = None,
    adjacency_weights: ArrayLike | None = None,
    *,
    background: float | ArrayLike = 0.0,
    nuisance_response: ArrayLike | None = None,
    nuisance_coefficients: ArrayLike | None = None,
    config: SurfaceMapConfig | None = None,
) -> SurfaceMapObjective:
    """Evaluate the public regularized objective for a supplied surface map."""
    solver_config = SurfaceMapConfig() if config is None else config
    problem = _prepare_surface_map_problem(
        observed_counts,
        response,
        patch_areas_m2,
        adjacency_edges,
        adjacency_weights,
        background=background,
        nuisance_response=nuisance_response,
    )
    densities = _as_non_negative_vector(
        densities_cps_1m_m2,
        name="densities_cps_1m_m2",
    )
    expected_density_count = problem.patch_count * problem.isotope_count
    if densities.size != expected_density_count:
        raise ValueError(
            "densities_cps_1m_m2 must contain one value per patch and isotope."
        )
    nuisance_count = int(problem.nuisance_response.shape[1])
    nuisance = (
        np.zeros(nuisance_count, dtype=float)
        if nuisance_coefficients is None
        else _as_non_negative_vector(
            nuisance_coefficients,
            name="nuisance_coefficients",
        )
    )
    if nuisance.size != nuisance_count:
        raise ValueError("nuisance_coefficients must match nuisance_response columns.")
    objective, _expected = _objective_from_prepared(
        problem,
        densities.reshape(problem.patch_count, problem.isotope_count),
        nuisance,
        solver_config,
    )
    return objective


def _dual_poisson_prox(
    dual_trial: NDArray[np.float64],
    dual_steps: NDArray[np.float64],
    observed: NDArray[np.float64],
    background: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply the closed-form proximal operator of the shifted Poisson conjugate."""
    sigma = np.asarray(dual_steps, dtype=float)
    trial = np.asarray(dual_trial, dtype=float)
    gamma = 1.0 / sigma
    proximal_center = trial / sigma
    shifted = background + proximal_center - gamma
    mean = 0.5 * (
        shifted + np.sqrt(np.maximum(shifted * shifted + 4.0 * gamma * observed, 0.0))
    )
    primal_prox = mean - background
    return trial - sigma * primal_prox


def _preconditioned_steps(
    problem: _PreparedSurfaceMapProblem,
    *,
    tv_active: bool,
    safety: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return diagonal Chambolle-Pock steps from absolute operator row/column sums."""
    response = problem.response_by_density
    nuisance = problem.nuisance_response
    observation_row_sums = np.sum(response, axis=1)
    if nuisance.shape[1]:
        observation_row_sums = observation_row_sums + np.sum(nuisance, axis=1)
    observation_steps = float(safety) / np.maximum(observation_row_sums, 1.0e-12)
    density_column_sums = np.sum(response, axis=0).reshape(
        problem.patch_count,
        problem.isotope_count,
    )
    edge_steps = np.zeros(0, dtype=float)
    if tv_active:
        degrees = np.asarray(
            np.abs(problem.incidence).sum(axis=0), dtype=float
        ).reshape(-1)
        density_column_sums = density_column_sums + degrees[:, None]
        edge_steps = np.full(
            problem.incidence.shape[0], float(safety) / 2.0, dtype=float
        )
    # A shared step within each isotope group makes the non-negative group
    # shrinkage below an exact proximal map instead of a diagonal approximation.
    patch_column_bound = np.max(density_column_sums, axis=1, keepdims=True)
    density_steps = np.broadcast_to(
        float(safety) / np.maximum(patch_column_bound, 1.0e-12),
        density_column_sums.shape,
    ).copy()
    if nuisance.shape[1]:
        nuisance_column_sums = np.sum(nuisance, axis=0)
        nuisance_steps = float(safety) / np.maximum(nuisance_column_sums, 1.0e-12)
    else:
        nuisance_steps = np.zeros(0, dtype=float)
    return density_steps, nuisance_steps, observation_steps, edge_steps


def _kkt_residual(
    problem: _PreparedSurfaceMapProblem,
    densities: NDArray[np.float64],
    nuisance: NDArray[np.float64],
    tv_dual: NDArray[np.float64],
    expected: NDArray[np.float64],
    config: SurfaceMapConfig,
) -> float:
    """Return a scale-normalized first-order residual for non-negative variables."""
    likelihood_gradient = 1.0 - problem.observed / np.maximum(
        expected,
        float(config.min_mean),
    )
    density_gradient = (problem.response_by_density.T @ likelihood_gradient).reshape(
        problem.patch_count, problem.isotope_count
    )
    density_gradient = (
        density_gradient + float(config.l1_weight) * problem.patch_areas[:, None]
    )
    if float(config.isotope_group_weight) > 0.0:
        density_norm = np.linalg.norm(densities, axis=1, keepdims=True)
        active = density_norm[:, 0] > 1.0e-12
        density_gradient[active] += (
            float(config.isotope_group_weight)
            * problem.patch_areas[active, None]
            * densities[active]
            / density_norm[active]
        )
    if tv_dual.size:
        density_gradient = density_gradient + problem.incidence.T @ tv_dual
    density_stationarity = np.where(
        densities > 1.0e-9,
        density_gradient,
        np.minimum(density_gradient, 0.0),
    )
    residual_parts = [density_stationarity.reshape(-1)]
    if nuisance.size:
        nuisance_gradient = problem.nuisance_response.T @ likelihood_gradient
        nuisance_gradient = (
            nuisance_gradient
            + float(config.nuisance_l1_weight)
            + _nuisance_l2_vector(config, nuisance.size) * nuisance
        )
        nuisance_stationarity = np.where(
            nuisance > 1.0e-9,
            nuisance_gradient,
            np.minimum(nuisance_gradient, 0.0),
        )
        residual_parts.append(nuisance_stationarity)
    residual = np.concatenate(residual_parts)
    scale = max(1.0, float(np.linalg.norm(likelihood_gradient)))
    return float(np.linalg.norm(residual) / scale)


def fit_surface_map_poisson(
    observed_counts: ArrayLike,
    response: ArrayLike,
    patch_areas_m2: ArrayLike,
    adjacency_edges: ArrayLike | None = None,
    adjacency_weights: ArrayLike | None = None,
    *,
    background: float | ArrayLike = 0.0,
    nuisance_response: ArrayLike | None = None,
    initial_densities_cps_1m_m2: ArrayLike | None = None,
    initial_nuisance_coefficients: ArrayLike | None = None,
    config: SurfaceMapConfig | None = None,
) -> SurfaceMapResult:
    """
    Fit a non-negative all-history Poisson surface intensity map.

    Response columns have unit integrated-strength semantics in cps at 1 m.
    The solver multiplies each column by its patch area, so optimized source
    variables are densities in cps at 1 m per square meter.  The L1 term is
    therefore total integrated strength, while graph TV compares neighboring
    densities and weights each difference by the supplied shared-edge measure.
    Matrix responses use shape ``observations x patches``.  Spectrum tensors
    use ``observed_shape + (patches,)`` or
    ``observed_shape + (patches, isotopes)`` and are flattened in one batch.
    """
    solver_config = SurfaceMapConfig() if config is None else config
    if solver_config.likelihood_family != "poisson":
        raise ValueError(
            "Calibrated overdispersion requires the matrix-free solver path."
        )
    problem = _prepare_surface_map_problem(
        observed_counts,
        response,
        patch_areas_m2,
        adjacency_edges,
        adjacency_weights,
        background=background,
        nuisance_response=nuisance_response,
    )
    density_shape = (problem.patch_count, problem.isotope_count)
    if initial_densities_cps_1m_m2 is None:
        densities = np.zeros(density_shape, dtype=float)
    else:
        density_vector = _as_non_negative_vector(
            initial_densities_cps_1m_m2,
            name="initial_densities_cps_1m_m2",
        )
        if density_vector.size != int(np.prod(density_shape, dtype=np.int64)):
            raise ValueError(
                "initial_densities_cps_1m_m2 must match patches by isotopes."
            )
        densities = density_vector.reshape(density_shape).copy()
    nuisance_count = int(problem.nuisance_response.shape[1])
    nuisance_l2 = _nuisance_l2_vector(solver_config, nuisance_count)
    if initial_nuisance_coefficients is None:
        nuisance = np.zeros(nuisance_count, dtype=float)
    else:
        nuisance = _as_non_negative_vector(
            initial_nuisance_coefficients,
            name="initial_nuisance_coefficients",
        ).copy()
        if nuisance.size != nuisance_count:
            raise ValueError(
                "initial_nuisance_coefficients must match nuisance_response columns."
            )
    tv_active = bool(
        float(solver_config.tv_weight) > 0.0 and problem.incidence.shape[0] > 0
    )
    (
        density_steps,
        nuisance_steps,
        observation_dual_steps,
        edge_dual_steps,
    ) = _preconditioned_steps(
        problem,
        tv_active=tv_active,
        safety=float(solver_config.step_safety),
    )
    observation_dual = np.zeros(problem.observed.size, dtype=float)
    tv_dual = (
        np.zeros((problem.incidence.shape[0], problem.isotope_count), dtype=float)
        if tv_active
        else np.zeros((0, problem.isotope_count), dtype=float)
    )
    densities_bar = densities.copy()
    nuisance_bar = nuisance.copy()
    previous_check_density = densities.copy()
    previous_check_nuisance = nuisance.copy()
    previous_objective = float("inf")
    relative_change = float("inf")
    relative_objective_change = float("inf")
    converged = False
    iterations = 0
    objective_history: list[float] = []

    for iteration in range(1, int(solver_config.max_iterations) + 1):
        signal_bar = problem.response_by_density @ densities_bar.reshape(-1)
        if nuisance_bar.size:
            signal_bar = signal_bar + problem.nuisance_response @ nuisance_bar
        observation_trial = observation_dual + observation_dual_steps * signal_bar
        observation_dual = _dual_poisson_prox(
            observation_trial,
            observation_dual_steps,
            problem.observed,
            problem.background,
        )
        if tv_active:
            difference_bar = problem.incidence @ densities_bar
            tv_trial = tv_dual + edge_dual_steps[:, None] * difference_bar
            tv_bound = float(solver_config.tv_weight) * problem.edge_weights[:, None]
            tv_dual = np.clip(tv_trial, -tv_bound, tv_bound)

        density_previous = densities
        nuisance_previous = nuisance
        density_gradient = (problem.response_by_density.T @ observation_dual).reshape(
            density_shape
        )
        if tv_active:
            density_gradient = density_gradient + problem.incidence.T @ tv_dual
        densities = np.maximum(
            densities
            - density_steps
            * (
                density_gradient
                + float(solver_config.l1_weight) * problem.patch_areas[:, None]
            ),
            0.0,
        )
        if float(solver_config.isotope_group_weight) > 0.0:
            group_norm = np.linalg.norm(densities, axis=1, keepdims=True)
            group_threshold = (
                density_steps[:, :1]
                * float(solver_config.isotope_group_weight)
                * problem.patch_areas[:, None]
            )
            group_scale = np.maximum(
                1.0 - group_threshold / np.maximum(group_norm, 1.0e-30),
                0.0,
            )
            densities = densities * group_scale
        if nuisance.size:
            nuisance_gradient = problem.nuisance_response.T @ observation_dual
            nuisance = np.maximum(
                nuisance
                - nuisance_steps
                * (nuisance_gradient + float(solver_config.nuisance_l1_weight)),
                0.0,
            ) / (1.0 + nuisance_steps * nuisance_l2)
        relaxation = float(solver_config.over_relaxation)
        densities_bar = densities + relaxation * (densities - density_previous)
        nuisance_bar = nuisance + relaxation * (nuisance - nuisance_previous)
        iterations = int(iteration)

        should_check = iteration % int(
            solver_config.check_interval
        ) == 0 or iteration == int(solver_config.max_iterations)
        if not should_check:
            continue
        density_delta = np.linalg.norm(densities - previous_check_density)
        nuisance_delta = np.linalg.norm(nuisance - previous_check_nuisance)
        state_delta = float(np.hypot(density_delta, nuisance_delta))
        state_norm = float(
            np.hypot(np.linalg.norm(densities), np.linalg.norm(nuisance))
        )
        relative_change = state_delta / max(state_norm, 1.0)
        objective_terms, _expected = _objective_from_prepared(
            problem,
            densities,
            nuisance,
            solver_config,
        )
        objective_history.append(float(objective_terms.total))
        if np.isfinite(previous_objective):
            relative_objective_change = abs(
                float(objective_terms.total) - previous_objective
            ) / max(abs(previous_objective), 1.0)
        previous_check_density = densities.copy()
        previous_check_nuisance = nuisance.copy()
        previous_objective = float(objective_terms.total)
        if relative_change <= float(
            solver_config.tolerance
        ) and relative_objective_change <= float(solver_config.objective_tolerance):
            converged = True
            break

    objective_terms, expected = _objective_from_prepared(
        problem,
        densities,
        nuisance,
        solver_config,
    )
    kkt_residual = _kkt_residual(
        problem,
        densities,
        nuisance,
        tv_dual,
        expected,
        solver_config,
    )
    integrated = densities * problem.patch_areas[:, None]
    return SurfaceMapResult(
        densities_cps_1m_m2=np.asarray(densities, dtype=float),
        integrated_strengths_cps_1m=np.asarray(integrated, dtype=float),
        nuisance_coefficients=np.asarray(nuisance, dtype=float),
        expected_counts=np.asarray(expected, dtype=float).reshape(
            problem.observation_shape
        ),
        objective=float(objective_terms.total),
        poisson_nll=float(objective_terms.poisson_nll),
        l1_penalty=float(objective_terms.l1_penalty),
        tv_penalty=float(objective_terms.tv_penalty),
        group_penalty=float(objective_terms.group_penalty),
        nuisance_penalty=float(objective_terms.nuisance_penalty),
        deviance=float(objective_terms.deviance),
        converged=bool(converged),
        iterations=int(iterations),
        relative_change=float(relative_change),
        relative_objective_change=float(relative_objective_change),
        kkt_residual=float(kkt_residual),
        objective_history=tuple(objective_history),
    )


def _torch_response_product(
    operator: ResponseOperator,
    values: object,
    *,
    transpose: bool,
    torch_module: object,
) -> object:
    """Apply a streamed response operator while retaining state on one device."""
    torch = torch_module
    vector = values
    if transpose:
        result = torch.zeros(
            operator.source_count,
            dtype=vector.dtype,
            device=vector.device,
        )
        for block in operator.iter_blocks():
            rows = torch.as_tensor(
                np.array(block.observation_indices, dtype=np.int64, copy=True),
                dtype=torch.long,
                device=vector.device,
            )
            columns = torch.as_tensor(
                np.array(block.source_indices, dtype=np.int64, copy=True),
                dtype=torch.long,
                device=vector.device,
            )
            matrix = torch.as_tensor(
                np.array(block.values, dtype=np.float64, copy=True),
                dtype=vector.dtype,
                device=vector.device,
            )
            result.index_add_(0, columns, matrix.T @ vector[rows])
        return result
    result = torch.zeros(
        operator.observation_count,
        dtype=vector.dtype,
        device=vector.device,
    )
    for block in operator.iter_blocks():
        rows = torch.as_tensor(
            np.array(block.observation_indices, dtype=np.int64, copy=True),
            dtype=torch.long,
            device=vector.device,
        )
        columns = torch.as_tensor(
            np.array(block.source_indices, dtype=np.int64, copy=True),
            dtype=torch.long,
            device=vector.device,
        )
        matrix = torch.as_tensor(
            np.array(block.values, dtype=np.float64, copy=True),
            dtype=vector.dtype,
            device=vector.device,
        )
        result.index_add_(0, rows, matrix @ vector[columns])
    return result


def fit_surface_map_poisson_operator(
    observed_counts: ArrayLike,
    response_operator: ResponseOperator,
    patch_areas_m2: ArrayLike,
    adjacency_edges: ArrayLike | None = None,
    adjacency_weights: ArrayLike | None = None,
    *,
    background: float | ArrayLike = 0.0,
    nuisance_response: ArrayLike | None = None,
    initial_densities_cps_1m_m2: ArrayLike | None = None,
    initial_nuisance_coefficients: ArrayLike | None = None,
    config: SurfaceMapConfig | None = None,
    use_gpu: bool = False,
    gpu_device: str = "cuda",
    gpu_dtype: str = "float64",
) -> SurfaceMapResult:
    """Fit Poisson surface density using streamed response products.

    The supplied operator maps density variables directly to expected counts;
    unlike :func:`fit_surface_map_poisson`, patch areas are therefore already
    included in each response block.  Areas remain explicit for L1, group
    regularization, and integrated-strength reporting.  CPU and CUDA execute
    the same Torch primal-dual updates while response blocks stay bounded.
    """
    if not isinstance(response_operator, ResponseOperator):
        raise TypeError("response_operator must implement ResponseOperator.")
    import torch

    solver_config = SurfaceMapConfig() if config is None else config
    observed_array = np.asarray(observed_counts, dtype=np.float64)
    if observed_array.shape != response_operator.observation_shape:
        raise ValueError("observed_counts must match response_operator shape.")
    observed_vector = _as_non_negative_vector(
        observed_array,
        name="observed_counts",
    )
    if solver_config.overdispersion_alpha:
        raw_alpha = np.asarray(
            solver_config.overdispersion_alpha,
            dtype=np.float64,
        )
        if raw_alpha.shape == (response_operator.observation_count,):
            alpha_vector = raw_alpha
        elif len(response_operator.observation_shape) == 2 and raw_alpha.shape == (
            response_operator.observation_shape[1],
        ):
            alpha_vector = np.tile(raw_alpha, response_operator.observation_shape[0])
        else:
            raise ValueError(
                "overdispersion_alpha must match all observations or energy bins."
            )
    else:
        alpha_vector = np.zeros(response_operator.observation_count, dtype=np.float64)
    if solver_config.likelihood_family == "negative_binomial" and not np.any(
        alpha_vector > 0.0
    ):
        raise ValueError("Negative-binomial likelihood requires positive calibration.")
    areas = _as_non_negative_vector(patch_areas_m2, name="patch_areas_m2")
    if areas.shape != (response_operator.patch_count,) or np.any(areas <= 0.0):
        raise ValueError("patch_areas_m2 must contain one positive area per patch.")
    nuisance_matrix = _flatten_nuisance_response(
        nuisance_response,
        response_operator.observation_shape,
    )
    background_vector = _broadcast_observation_vector(
        background,
        response_operator.observation_shape,
        name="background",
    )
    incidence, edge_weights = _canonical_graph(
        adjacency_edges,
        adjacency_weights,
        response_operator.patch_count,
    )
    density_shape = (
        response_operator.patch_count,
        response_operator.isotope_count,
    )
    if initial_densities_cps_1m_m2 is None:
        initial_density = np.zeros(density_shape, dtype=np.float64)
    else:
        initial_density = _as_non_negative_vector(
            initial_densities_cps_1m_m2,
            name="initial_densities_cps_1m_m2",
        )
        if initial_density.size != response_operator.source_count:
            raise ValueError("Initial densities must match patches by isotopes.")
        initial_density = initial_density.reshape(density_shape)
    if solver_config.likelihood_family == "negative_binomial" and not np.any(
        initial_density > 0.0
    ):
        unit_total = float(np.sum(response_operator.row_sums()))
        initial_scale = float(np.sum(observed_vector)) / max(unit_total, 1.0e-12)
        initial_density = np.full(density_shape, initial_scale, dtype=np.float64)
    nuisance_count = nuisance_matrix.shape[1]
    nuisance_l2_values = _nuisance_l2_vector(solver_config, nuisance_count)
    if initial_nuisance_coefficients is None:
        initial_nuisance = np.zeros(nuisance_count, dtype=np.float64)
    else:
        initial_nuisance = _as_non_negative_vector(
            initial_nuisance_coefficients,
            name="initial_nuisance_coefficients",
        )
        if initial_nuisance.size != nuisance_count:
            raise ValueError("Initial nuisance values must match nuisance columns.")
    if gpu_dtype not in {"float32", "float64"}:
        raise ValueError("gpu_dtype must be float32 or float64.")
    device = torch.device(gpu_device if use_gpu else "cpu")
    if use_gpu and device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA matrix-free solve requested but CUDA is unavailable.")
    dtype = torch.float32 if gpu_dtype == "float32" else torch.float64

    def tensor(values: ArrayLike, *, integer: bool = False) -> object:
        """Copy one bounded array to the selected solver device."""
        return torch.as_tensor(
            np.asarray(values),
            dtype=torch.long if integer else dtype,
            device=device,
        )

    observed = tensor(observed_vector)
    alpha_t = tensor(alpha_vector)
    background_t = tensor(background_vector)
    nuisance_design = tensor(nuisance_matrix)
    areas_t = tensor(areas)
    edge_weights_t = tensor(edge_weights)
    densities = tensor(initial_density).clone()
    nuisance = tensor(initial_nuisance).clone()
    tv_active = bool(float(solver_config.tv_weight) > 0.0 and incidence.shape[0] > 0)
    coo = incidence.tocoo()
    incidence_t = torch.sparse_coo_tensor(
        tensor(np.vstack((coo.row, coo.col)), integer=True),
        tensor(coo.data),
        size=incidence.shape,
        dtype=dtype,
        device=device,
        check_invariants=False,
    ).coalesce()

    row_sums = response_operator.row_sums()
    if nuisance_count:
        row_sums = row_sums + np.sum(nuisance_matrix, axis=1)
    observation_steps = tensor(
        float(solver_config.step_safety) / np.maximum(row_sums, 1.0e-12)
    )
    column_sums = response_operator.column_sums().reshape(density_shape)
    if tv_active:
        degrees = np.asarray(np.abs(incidence).sum(axis=0)).reshape(-1)
        column_sums = column_sums + degrees[:, None]
    group_bound = np.max(column_sums, axis=1, keepdims=True)
    density_steps = tensor(
        np.broadcast_to(
            float(solver_config.step_safety) / np.maximum(group_bound, 1.0e-12),
            density_shape,
        ).copy()
    )
    nuisance_steps = tensor(
        (
            float(solver_config.step_safety)
            / np.maximum(np.sum(nuisance_matrix, axis=0), 1.0e-12)
        )
        if nuisance_count
        else np.zeros(0, dtype=np.float64)
    )
    nuisance_l2_t = tensor(nuisance_l2_values)
    edge_steps = (
        torch.full(
            (incidence.shape[0],),
            float(solver_config.step_safety) / 2.0,
            dtype=dtype,
            device=device,
        )
        if tv_active
        else torch.zeros(0, dtype=dtype, device=device)
    )
    observation_dual = torch.zeros(
        response_operator.observation_count,
        dtype=dtype,
        device=device,
    )
    tv_dual = torch.zeros(
        (incidence.shape[0] if tv_active else 0, response_operator.isotope_count),
        dtype=dtype,
        device=device,
    )
    densities_bar = densities.clone()
    nuisance_bar = nuisance.clone()
    previous_density = densities.clone()
    previous_nuisance = nuisance.clone()
    previous_objective = float("inf")
    relative_change = float("inf")
    relative_objective_change = float("inf")
    converged = False
    iterations = 0
    objective_history: list[float] = []

    def forward(density_values: object) -> object:
        """Apply the streamed density operator on the solver device."""
        return _torch_response_product(
            response_operator,
            density_values.reshape(-1),
            transpose=False,
            torch_module=torch,
        )

    def transpose(observation_values: object) -> object:
        """Apply the streamed transpose on the solver device."""
        return _torch_response_product(
            response_operator,
            observation_values,
            transpose=True,
            torch_module=torch,
        ).reshape(density_shape)

    def expected_counts(density_values: object, nuisance_values: object) -> object:
        """Return the current positive expected-count vector."""
        signal = forward(density_values)
        if nuisance_count:
            signal = signal + nuisance_design @ nuisance_values
        return torch.clamp(
            background_t + signal,
            min=float(solver_config.min_mean),
        )

    def objective_values(
        density_values: object,
        nuisance_values: object,
    ) -> tuple[tuple[float, ...], object]:
        """Return objective components and expected counts on the device."""
        expected = expected_counts(density_values, nuisance_values)
        if solver_config.likelihood_family == "negative_binomial":
            positive_alpha = alpha_t > 0.0
            concentration = torch.where(
                positive_alpha,
                1.0 / torch.clamp(alpha_t, min=1.0e-30),
                torch.ones_like(alpha_t),
            )
            nb_nll = (
                (observed + concentration) * torch.log(concentration + expected)
                - observed * torch.log(expected)
                - concentration * torch.log(concentration)
            )
            poisson_terms = expected - observed * torch.log(expected)
            likelihood = torch.sum(torch.where(positive_alpha, nb_nll, poisson_terms))
        else:
            likelihood = torch.sum(expected - observed * torch.log(expected))
        l1 = float(solver_config.l1_weight) * torch.sum(
            areas_t[:, None] * density_values
        )
        if tv_active:
            differences = torch.sparse.mm(incidence_t, density_values)
            tv = float(solver_config.tv_weight) * torch.sum(
                edge_weights_t[:, None] * torch.abs(differences)
            )
        else:
            tv = torch.zeros((), dtype=dtype, device=device)
        group = float(solver_config.isotope_group_weight) * torch.sum(
            areas_t * torch.linalg.vector_norm(density_values, dim=1)
        )
        nuisance_penalty = float(solver_config.nuisance_l1_weight) * torch.sum(
            nuisance_values
        ) + 0.5 * torch.dot(nuisance_l2_t * nuisance_values, nuisance_values)
        if solver_config.likelihood_family == "negative_binomial":
            saturated_mean = torch.clamp(observed, min=float(solver_config.min_mean))
            saturated_nb = (
                (observed + concentration) * torch.log(concentration + saturated_mean)
                - observed * torch.log(saturated_mean)
                - concentration * torch.log(concentration)
            )
            saturated_poisson = saturated_mean - observed * torch.log(saturated_mean)
            saturated = torch.where(
                positive_alpha,
                saturated_nb,
                saturated_poisson,
            )
            deviance = 2.0 * (likelihood - torch.sum(saturated))
        else:
            positive = observed > 0.0
            log_term = torch.where(
                positive,
                observed * torch.log(torch.clamp(observed, min=1.0e-30) / expected),
                torch.zeros_like(observed),
            )
            deviance = 2.0 * torch.sum(log_term - observed + expected)
        total = likelihood + l1 + tv + group + nuisance_penalty
        return (
            float(total.item()),
            float(likelihood.item()),
            float(l1.item()),
            float(tv.item()),
            float(group.item()),
            float(nuisance_penalty.item()),
            float(deviance.item()),
        ), expected

    for iteration in range(1, int(solver_config.max_iterations) + 1):
        if solver_config.likelihood_family == "negative_binomial":
            expected_current = expected_counts(densities, nuisance)
            likelihood_gradient = (expected_current - observed) / torch.clamp(
                expected_current + alpha_t * expected_current * expected_current,
                min=float(solver_config.min_mean),
            )
            density_old = densities
            nuisance_old = nuisance
            gradient = transpose(likelihood_gradient)
            gradient = gradient + float(solver_config.l1_weight) * areas_t[:, None]
            if float(solver_config.isotope_group_weight) > 0.0:
                norms = torch.linalg.vector_norm(densities, dim=1, keepdim=True)
                active = norms[:, 0] > 1.0e-12
                gradient[active] += (
                    float(solver_config.isotope_group_weight)
                    * areas_t[active][:, None]
                    * densities[active]
                    / norms[active]
                )
            if tv_active:
                differences = torch.sparse.mm(incidence_t, densities)
                tv_dual = (
                    float(solver_config.tv_weight)
                    * edge_weights_t[:, None]
                    * torch.sign(differences)
                )
                gradient = gradient + torch.sparse.mm(
                    incidence_t.transpose(0, 1),
                    tv_dual,
                )
            # Diagonal response-sum preconditioning is damped for the
            # non-quadratic calibrated likelihood.  The diminishing factor is
            # deterministic and prevents large early nuisance corrections.
            damping = min(0.25, 2.5 / math.sqrt(float(iteration)))
            densities = torch.clamp(
                densities - damping * density_steps * gradient,
                min=0.0,
            )
            if nuisance_count:
                nuisance_gradient = (
                    nuisance_design.T @ likelihood_gradient
                    + float(solver_config.nuisance_l1_weight)
                    + nuisance_l2_t * nuisance
                )
                nuisance = torch.clamp(
                    nuisance - damping * nuisance_steps * nuisance_gradient,
                    min=0.0,
                )
            densities_bar = densities
            nuisance_bar = nuisance
            iterations = iteration
            if iteration % int(solver_config.check_interval) != 0 and iteration != int(
                solver_config.max_iterations
            ):
                continue
            state_delta = torch.sqrt(
                torch.sum((densities - previous_density) ** 2)
                + torch.sum((nuisance - previous_nuisance) ** 2)
            )
            state_norm = torch.sqrt(torch.sum(densities**2) + torch.sum(nuisance**2))
            relative_change = float(
                (state_delta / torch.clamp(state_norm, min=1.0)).item()
            )
            terms, _ = objective_values(densities, nuisance)
            objective_history.append(terms[0])
            if np.isfinite(previous_objective):
                relative_objective_change = abs(terms[0] - previous_objective) / max(
                    abs(previous_objective), 1.0
                )
            previous_density = densities.clone()
            previous_nuisance = nuisance.clone()
            previous_objective = terms[0]
            if relative_change <= float(
                solver_config.tolerance
            ) and relative_objective_change <= float(solver_config.objective_tolerance):
                converged = True
                break
            continue
        signal_bar = forward(densities_bar)
        if nuisance_count:
            signal_bar = signal_bar + nuisance_design @ nuisance_bar
        trial = observation_dual + observation_steps * signal_bar
        gamma = 1.0 / observation_steps
        center = trial / observation_steps
        shifted = background_t + center - gamma
        mean = 0.5 * (
            shifted
            + torch.sqrt(
                torch.clamp(shifted * shifted + 4.0 * gamma * observed, min=0.0)
            )
        )
        observation_dual = trial - observation_steps * (mean - background_t)
        if tv_active:
            difference_bar = torch.sparse.mm(incidence_t, densities_bar)
            tv_trial = tv_dual + edge_steps[:, None] * difference_bar
            tv_bound = float(solver_config.tv_weight) * edge_weights_t[:, None]
            tv_dual = torch.maximum(torch.minimum(tv_trial, tv_bound), -tv_bound)

        density_old = densities
        nuisance_old = nuisance
        gradient = transpose(observation_dual)
        if tv_active:
            gradient = gradient + torch.sparse.mm(incidence_t.transpose(0, 1), tv_dual)
        densities = torch.clamp(
            densities
            - density_steps
            * (gradient + float(solver_config.l1_weight) * areas_t[:, None]),
            min=0.0,
        )
        if float(solver_config.isotope_group_weight) > 0.0:
            group_norm = torch.linalg.vector_norm(densities, dim=1, keepdim=True)
            threshold = (
                density_steps[:, :1]
                * float(solver_config.isotope_group_weight)
                * areas_t[:, None]
            )
            densities = densities * torch.clamp(
                1.0 - threshold / torch.clamp(group_norm, min=1.0e-30),
                min=0.0,
            )
        if nuisance_count:
            nuisance_gradient = nuisance_design.T @ observation_dual
            nuisance = torch.clamp(
                nuisance
                - nuisance_steps
                * (nuisance_gradient + float(solver_config.nuisance_l1_weight)),
                min=0.0,
            ) / (1.0 + nuisance_steps * nuisance_l2_t)
        relaxation = float(solver_config.over_relaxation)
        densities_bar = densities + relaxation * (densities - density_old)
        nuisance_bar = nuisance + relaxation * (nuisance - nuisance_old)
        iterations = iteration
        if iteration % int(solver_config.check_interval) != 0 and iteration != int(
            solver_config.max_iterations
        ):
            continue
        state_delta = torch.sqrt(
            torch.sum((densities - previous_density) ** 2)
            + torch.sum((nuisance - previous_nuisance) ** 2)
        )
        state_norm = torch.sqrt(torch.sum(densities**2) + torch.sum(nuisance**2))
        relative_change = float((state_delta / torch.clamp(state_norm, min=1.0)).item())
        terms, _ = objective_values(densities, nuisance)
        objective_history.append(terms[0])
        if np.isfinite(previous_objective):
            relative_objective_change = abs(terms[0] - previous_objective) / max(
                abs(previous_objective), 1.0
            )
        previous_density = densities.clone()
        previous_nuisance = nuisance.clone()
        previous_objective = terms[0]
        if relative_change <= float(
            solver_config.tolerance
        ) and relative_objective_change <= float(solver_config.objective_tolerance):
            converged = True
            break

    terms, expected = objective_values(densities, nuisance)
    if solver_config.likelihood_family == "negative_binomial":
        likelihood_gradient = (expected - observed) / torch.clamp(
            expected + alpha_t * expected * expected,
            min=float(solver_config.min_mean),
        )
    else:
        likelihood_gradient = 1.0 - observed / expected
    density_gradient = transpose(likelihood_gradient)
    density_gradient = (
        density_gradient + float(solver_config.l1_weight) * areas_t[:, None]
    )
    if float(solver_config.isotope_group_weight) > 0.0:
        norms = torch.linalg.vector_norm(densities, dim=1, keepdim=True)
        active = norms[:, 0] > 1.0e-12
        density_gradient[active] += (
            float(solver_config.isotope_group_weight)
            * areas_t[active][:, None]
            * densities[active]
            / norms[active]
        )
    if tv_active:
        density_gradient = density_gradient + torch.sparse.mm(
            incidence_t.transpose(0, 1), tv_dual
        )
    stationarity = torch.where(
        densities > 1.0e-9,
        density_gradient,
        torch.minimum(density_gradient, torch.zeros_like(density_gradient)),
    ).reshape(-1)
    residual_parts = [stationarity]
    if nuisance_count:
        nuisance_gradient = (
            nuisance_design.T @ likelihood_gradient
            + float(solver_config.nuisance_l1_weight)
            + nuisance_l2_t * nuisance
        )
        residual_parts.append(
            torch.where(
                nuisance > 1.0e-9,
                nuisance_gradient,
                torch.minimum(nuisance_gradient, torch.zeros_like(nuisance_gradient)),
            )
        )
    kkt = float(
        (
            torch.linalg.vector_norm(torch.cat(residual_parts))
            / torch.clamp(torch.linalg.vector_norm(likelihood_gradient), min=1.0)
        ).item()
    )
    densities_numpy = densities.detach().cpu().numpy().astype(np.float64, copy=False)
    nuisance_numpy = nuisance.detach().cpu().numpy().astype(np.float64, copy=False)
    expected_numpy = expected.detach().cpu().numpy().astype(np.float64, copy=False)
    return SurfaceMapResult(
        densities_cps_1m_m2=densities_numpy,
        integrated_strengths_cps_1m=densities_numpy * areas[:, None],
        nuisance_coefficients=nuisance_numpy,
        expected_counts=expected_numpy.reshape(response_operator.observation_shape),
        objective=terms[0],
        poisson_nll=terms[1],
        l1_penalty=terms[2],
        tv_penalty=terms[3],
        group_penalty=terms[4],
        nuisance_penalty=terms[5],
        deviance=terms[6],
        converged=converged,
        iterations=iterations,
        relative_change=relative_change,
        relative_objective_change=relative_objective_change,
        kkt_residual=kkt,
        objective_history=tuple(objective_history),
    )
