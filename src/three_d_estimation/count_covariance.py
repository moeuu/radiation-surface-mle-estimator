"""Covariance-aware diagnostic fitting for extracted isotope-count observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from .solver import SurfaceMapConfig, SurfaceMapResult


@dataclass(frozen=True, slots=True)
class CountCovarianceDiagnostics:
    """Describe covariance regularization and numerical conditioning."""

    likelihood_family: str
    covariance_regularization: float
    maximum_condition_number: float
    condition_numbers: tuple[float, ...]


def _regularized_precisions(
    covariance: ArrayLike,
    *,
    regularization: float,
    maximum_condition_number: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[float, ...]]:
    """Return regularized precision matrices while failing on singular inputs."""
    matrices = np.asarray(covariance, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("isotope_covariances must have shape (M, I, I).")
    if np.any(~np.isfinite(matrices)) or not np.allclose(
        matrices,
        np.swapaxes(matrices, 1, 2),
        rtol=1.0e-8,
        atol=1.0e-10,
    ):
        raise ValueError("isotope_covariances must be finite and symmetric.")
    ridge = float(regularization)
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("covariance_regularization must be finite and non-negative.")
    condition_limit = float(maximum_condition_number)
    if not np.isfinite(condition_limit) or condition_limit <= 1.0:
        raise ValueError("maximum_condition_number must exceed one.")
    regularized = np.array(matrices, dtype=np.float64, copy=True)
    scale = np.maximum(np.trace(regularized, axis1=1, axis2=2) / matrices.shape[1], 1.0)
    regularized += ridge * scale[:, None, None] * np.eye(matrices.shape[1])[None, :, :]
    eigenvalues = np.linalg.eigvalsh(regularized)
    if np.any(eigenvalues[:, 0] <= 0.0):
        raise ValueError(
            "Isotope covariance is singular; configure positive regularization."
        )
    conditions = eigenvalues[:, -1] / eigenvalues[:, 0]
    if np.any(conditions > condition_limit):
        raise ValueError(
            "Regularized isotope covariance exceeds maximum_condition_number."
        )
    precision = np.linalg.inv(regularized)
    log_determinants = np.linalg.slogdet(regularized)[1]
    return precision, log_determinants, tuple(float(value) for value in conditions)


def fit_surface_map_count_covariance(
    observed_counts: ArrayLike,
    response_per_integrated_strength: ArrayLike,
    isotope_covariances: ArrayLike,
    patch_areas_m2: ArrayLike,
    adjacency_edges: ArrayLike | None = None,
    adjacency_weights: ArrayLike | None = None,
    *,
    nuisance_response: ArrayLike | None = None,
    initial_densities_cps_1m_m2: ArrayLike | None = None,
    initial_nuisance_coefficients: ArrayLike | None = None,
    likelihood_family: str = "multivariate_student_t",
    student_t_degrees_of_freedom: float = 4.0,
    covariance_regularization: float = 1.0e-6,
    maximum_condition_number: float = 1.0e10,
    config: SurfaceMapConfig | None = None,
) -> tuple[SurfaceMapResult, CountCovarianceDiagnostics]:
    """Fit a non-negative count surface map with full isotope covariance.

    This is a diagnostic count-domain estimator.  Raw-spectrum fitting remains
    authoritative because covariance weighting cannot correct systematic count
    extraction bias.
    """
    if likelihood_family not in {"covariance_gaussian", "multivariate_student_t"}:
        raise ValueError("Unsupported covariance-aware count likelihood.")
    degrees = float(student_t_degrees_of_freedom)
    if not np.isfinite(degrees) or degrees <= 2.0:
        raise ValueError("student_t_degrees_of_freedom must exceed two.")
    observed = np.asarray(observed_counts, dtype=np.float64)
    response = np.asarray(response_per_integrated_strength, dtype=np.float64)
    if observed.ndim != 2 or response.ndim != 3:
        raise ValueError("Count observations and response must be MxI and MxGxI.")
    measurement_count, isotope_count = observed.shape
    if response.shape[0] != measurement_count or response.shape[2] != isotope_count:
        raise ValueError("Count response dimensions must match observations.")
    if np.any(~np.isfinite(observed)) or np.any(observed < 0.0):
        raise ValueError("observed_counts must be finite and non-negative.")
    if np.any(~np.isfinite(response)) or np.any(response < 0.0):
        raise ValueError("Count response must be finite and non-negative.")
    areas = np.asarray(patch_areas_m2, dtype=np.float64).reshape(-1)
    patch_count = response.shape[1]
    if (
        areas.shape != (patch_count,)
        or np.any(~np.isfinite(areas))
        or np.any(areas <= 0.0)
    ):
        raise ValueError("patch_areas_m2 must contain one positive patch area.")
    precision, logdet, conditions = _regularized_precisions(
        isotope_covariances,
        regularization=covariance_regularization,
        maximum_condition_number=maximum_condition_number,
    )
    nuisance = (
        np.zeros((measurement_count, isotope_count, 0), dtype=np.float64)
        if nuisance_response is None
        else np.asarray(nuisance_response, dtype=np.float64)
    )
    if nuisance.ndim != 3 or nuisance.shape[:2] != observed.shape:
        raise ValueError("nuisance_response must have shape (M, I, N).")
    if np.any(~np.isfinite(nuisance)) or np.any(nuisance < 0.0):
        raise ValueError("nuisance_response must be finite and non-negative.")
    nuisance_count = nuisance.shape[2]
    solver = SurfaceMapConfig() if config is None else config
    if solver.likelihood_family != "poisson":
        raise ValueError("Count covariance fitting owns its likelihood selection.")
    density_response = response * areas[None, :, None]
    edges = (
        np.zeros((0, 2), dtype=np.int64)
        if adjacency_edges is None
        else np.asarray(
            adjacency_edges,
            dtype=np.int64,
        ).reshape(-1, 2)
    )
    edge_weights = (
        np.ones(edges.shape[0], dtype=np.float64)
        if adjacency_weights is None
        else np.asarray(adjacency_weights, dtype=np.float64).reshape(-1)
    )
    if edge_weights.shape != (edges.shape[0],) or np.any(edge_weights < 0.0):
        raise ValueError("adjacency_weights must align with graph edges.")
    density_size = patch_count * isotope_count
    density_initial = (
        np.zeros(density_size, dtype=np.float64)
        if initial_densities_cps_1m_m2 is None
        else np.asarray(initial_densities_cps_1m_m2, dtype=np.float64).reshape(-1)
    )
    nuisance_initial = (
        np.zeros(nuisance_count, dtype=np.float64)
        if initial_nuisance_coefficients is None
        else np.asarray(initial_nuisance_coefficients, dtype=np.float64).reshape(-1)
    )
    if density_initial.shape != (density_size,) or nuisance_initial.shape != (
        nuisance_count,
    ):
        raise ValueError("Initial values do not match count covariance problem.")
    initial = np.concatenate(
        (np.maximum(density_initial, 0.0), np.maximum(nuisance_initial, 0.0))
    )
    nuisance_l2 = (
        np.full(nuisance_count, float(solver.nuisance_l2_weight))
        if not solver.nuisance_l2_weights
        else np.asarray(solver.nuisance_l2_weights, dtype=np.float64)
    )
    if nuisance_l2.shape != (nuisance_count,):
        raise ValueError("nuisance_l2_weights must match nuisance columns.")

    def evaluate(values: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        """Return covariance likelihood plus regularization and its gradient."""
        density = values[:density_size].reshape(patch_count, isotope_count)
        nuisance_values = values[density_size:]
        expected = np.einsum("mgi,gi->mi", density_response, density, optimize=True)
        if nuisance_count:
            expected += np.einsum("min,n->mi", nuisance, nuisance_values, optimize=True)
        residual = observed - expected
        whitened = np.einsum("mij,mj->mi", precision, residual, optimize=True)
        mahalanobis = np.einsum("mi,mi->m", residual, whitened, optimize=True)
        if likelihood_family == "multivariate_student_t":
            factors = (degrees + isotope_count) / (degrees + mahalanobis)
            nll = 0.5 * np.sum(
                logdet + (degrees + isotope_count) * np.log1p(mahalanobis / degrees)
            )
        else:
            factors = np.ones(measurement_count, dtype=np.float64)
            nll = 0.5 * np.sum(logdet + mahalanobis)
        mean_gradient = -factors[:, None] * whitened
        density_gradient = np.einsum(
            "mi,mgi->gi",
            mean_gradient,
            density_response,
            optimize=True,
        )
        density_gradient += float(solver.l1_weight) * areas[:, None]
        l1 = float(solver.l1_weight) * float(np.sum(areas[:, None] * density))
        tv = 0.0
        if edges.size and float(solver.tv_weight) > 0.0:
            differences = density[edges[:, 1]] - density[edges[:, 0]]
            weighted_sign = edge_weights[:, None] * np.sign(differences)
            np.add.at(
                density_gradient,
                edges[:, 1],
                float(solver.tv_weight) * weighted_sign,
            )
            np.add.at(
                density_gradient,
                edges[:, 0],
                -float(solver.tv_weight) * weighted_sign,
            )
            tv = float(solver.tv_weight) * float(
                np.sum(edge_weights[:, None] * np.abs(differences))
            )
        group = 0.0
        if float(solver.isotope_group_weight) > 0.0:
            norms = np.linalg.norm(density, axis=1)
            active = norms > 1.0e-12
            density_gradient[active] += (
                float(solver.isotope_group_weight)
                * areas[active, None]
                * density[active]
                / norms[active, None]
            )
            group = float(solver.isotope_group_weight) * float(np.sum(areas * norms))
        nuisance_gradient = (
            np.einsum("mi,min->n", mean_gradient, nuisance, optimize=True)
            if nuisance_count
            else np.zeros(0, dtype=np.float64)
        )
        nuisance_gradient += (
            float(solver.nuisance_l1_weight) + nuisance_l2 * nuisance_values
        )
        nuisance_penalty = float(solver.nuisance_l1_weight) * float(
            np.sum(nuisance_values)
        ) + 0.5 * float(np.dot(nuisance_l2, nuisance_values * nuisance_values))
        objective = float(nll + l1 + tv + group + nuisance_penalty)
        gradient = np.concatenate((density_gradient.reshape(-1), nuisance_gradient))
        return objective, gradient

    history: list[float] = []

    def callback(values: NDArray[np.float64]) -> None:
        """Record deterministic optimizer objective history."""
        history.append(evaluate(values)[0])

    optimized = minimize(
        evaluate,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, None)] * initial.size,
        callback=callback,
        options={
            "maxiter": int(solver.max_iterations),
            "ftol": float(solver.objective_tolerance),
            "gtol": float(solver.tolerance),
            "maxls": 40,
        },
    )
    values = np.maximum(np.asarray(optimized.x, dtype=np.float64), 0.0)
    density = values[:density_size].reshape(patch_count, isotope_count)
    nuisance_values = values[density_size:]
    expected = np.einsum("mgi,gi->mi", density_response, density, optimize=True)
    if nuisance_count:
        expected += np.einsum("min,n->mi", nuisance, nuisance_values, optimize=True)
    residual = observed - expected
    whitened = np.einsum("mij,mj->mi", precision, residual, optimize=True)
    mahalanobis = np.einsum("mi,mi->m", residual, whitened, optimize=True)
    deviance = (
        float(np.sum((degrees + isotope_count) * np.log1p(mahalanobis / degrees)))
        if likelihood_family == "multivariate_student_t"
        else float(np.sum(mahalanobis))
    )
    objective, gradient = evaluate(values)
    projected = np.where(values > 1.0e-9, gradient, np.minimum(gradient, 0.0))
    scale = max(1.0, float(np.linalg.norm(gradient)))
    diagnostics = CountCovarianceDiagnostics(
        likelihood_family=likelihood_family,
        covariance_regularization=float(covariance_regularization),
        maximum_condition_number=float(maximum_condition_number),
        condition_numbers=conditions,
    )
    return (
        SurfaceMapResult(
            densities_cps_1m_m2=density,
            integrated_strengths_cps_1m=density * areas[:, None],
            nuisance_coefficients=nuisance_values,
            expected_counts=np.maximum(expected, 0.0),
            objective=objective,
            poisson_nll=objective,
            l1_penalty=float(solver.l1_weight)
            * float(np.sum(areas[:, None] * density)),
            tv_penalty=0.0,
            group_penalty=0.0,
            nuisance_penalty=float(solver.nuisance_l1_weight)
            * float(np.sum(nuisance_values))
            + 0.5 * float(np.dot(nuisance_l2, nuisance_values * nuisance_values)),
            deviance=deviance,
            converged=bool(optimized.success),
            iterations=int(optimized.nit),
            relative_change=0.0,
            relative_objective_change=0.0,
            kkt_residual=float(np.linalg.norm(projected) / scale),
            objective_history=tuple(history),
        ),
        diagnostics,
    )


__all__ = [
    "CountCovarianceDiagnostics",
    "fit_surface_map_count_covariance",
]
