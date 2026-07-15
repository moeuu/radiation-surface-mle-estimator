"""Likelihood helpers and per-source expected-count utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from measurement.continuous_kernels import ContinuousKernel

if TYPE_CHECKING:
    import torch


DEFAULT_GEANT4_COUNT_LIKELIHOOD_MODEL = "student_t"
DEFAULT_GEANT4_TRANSPORT_MODEL_REL_SIGMA = 0.10
DEFAULT_GEANT4_TRANSPORT_MODEL_ABS_SIGMA = 5.0
DEFAULT_GEANT4_SPECTRUM_COUNT_REL_SIGMA = 0.05
DEFAULT_GEANT4_SPECTRUM_COUNT_ABS_SIGMA = 5.0
DEFAULT_GEANT4_LOW_COUNT_ABS_SIGMA = 20.0
DEFAULT_GEANT4_LOW_COUNT_TRANSITION_COUNTS = 100.0
DEFAULT_GEANT4_COUNT_LIKELIHOOD_DF = 5.0


def normalize_count_likelihood_model(model: str) -> str:
    """Return a canonical count likelihood model name."""
    normalized = str(model).strip().lower()
    aliases = {
        "": "poisson",
        "normal": "gaussian",
        "robust": "student_t",
        "robust_gaussian": "student_t",
        "t": "student_t",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"poisson", "gaussian", "student_t"}:
        raise ValueError(f"Unknown count likelihood model: {model}")
    return normalized


def _normalize_count_likelihood_model(model: str) -> str:
    """Return a canonical count likelihood model name for legacy callers."""
    return normalize_count_likelihood_model(model)


def poisson_log_likelihood(z_k: NDArray[np.float64], lambda_k: NDArray[np.float64], epsilon: float = 1e-12) -> float:
    """
    Return the Poisson log-likelihood sum_k [z_k * log(lambda_k) - lambda_k] (constants omitted).
    """
    lambda_safe = np.maximum(lambda_k, float(epsilon))
    return float(np.sum(z_k * np.log(lambda_safe) - lambda_safe))


def count_likelihood_variance(
    z_k: NDArray[np.float64],
    lambda_k: NDArray[np.float64],
    *,
    transport_model_rel_sigma: float = 0.0,
    transport_model_abs_sigma: float = 0.0,
    spectrum_count_rel_sigma: float = 0.0,
    spectrum_count_abs_sigma: float = 0.0,
    low_count_abs_sigma: float = 0.0,
    low_count_transition_counts: float = 0.0,
    observation_count_variance: float | NDArray[np.float64] = 0.0,
    epsilon: float = 1e-12,
) -> NDArray[np.float64]:
    """
    Return observation variance for count likelihoods with model discrepancy.

    The variance includes Poisson counting noise plus multiplicative uncertainty
    from transport-model mismatch and spectrum-count extraction error. This is
    the likelihood-side place to absorb Geant4 effects that are intentionally
    not represented in the fast PF kernel, such as scatter, build-up, and
    photopeak-decomposition residuals.
    """
    z_arr = np.maximum(np.asarray(z_k, dtype=float), 0.0)
    lam_arr = np.maximum(np.asarray(lambda_k, dtype=float), float(epsilon))
    transport_rel = max(float(transport_model_rel_sigma), 0.0)
    transport_abs = max(float(transport_model_abs_sigma), 0.0)
    spectrum_rel = max(float(spectrum_count_rel_sigma), 0.0)
    spectrum_abs = max(float(spectrum_count_abs_sigma), 0.0)
    low_count_abs = max(float(low_count_abs_sigma), 0.0)
    low_count_transition = max(float(low_count_transition_counts), 0.0)
    obs_var = np.maximum(np.asarray(observation_count_variance, dtype=float), 0.0)
    scale_ref = np.maximum(z_arr, lam_arr)
    low_count_weight = 0.0
    if low_count_abs > 0.0 and low_count_transition > 0.0:
        low_count_weight = low_count_transition / (scale_ref + low_count_transition)
    variance = (
        lam_arr
        + (transport_rel * lam_arr) ** 2
        + transport_abs**2
        + (spectrum_rel * scale_ref) ** 2
        + spectrum_abs**2
        + (low_count_abs * low_count_weight) ** 2
        + obs_var
    )
    return np.maximum(variance, float(epsilon))


def count_likelihood_variance_torch(
    z_k: "torch.Tensor",
    lambda_k: "torch.Tensor",
    *,
    transport_model_rel_sigma: float = 0.0,
    transport_model_abs_sigma: float = 0.0,
    spectrum_count_rel_sigma: float = 0.0,
    spectrum_count_abs_sigma: float = 0.0,
    low_count_abs_sigma: float = 0.0,
    low_count_transition_counts: float = 0.0,
    observation_count_variance: float | "torch.Tensor" = 0.0,
    epsilon: float = 1e-12,
) -> "torch.Tensor":
    """Return torch observation variance equivalent to count_likelihood_variance."""
    import torch

    if torch.is_tensor(lambda_k):
        lam_arr = lambda_k.to(dtype=torch.float64)
    else:
        lam_arr = torch.as_tensor(lambda_k, dtype=torch.float64)
    lam_arr = torch.clamp(lam_arr, min=float(epsilon))
    z_arr = torch.clamp(
        torch.as_tensor(z_k, device=lam_arr.device, dtype=torch.float64),
        min=0.0,
    )
    obs_var = torch.clamp(
        torch.as_tensor(
            observation_count_variance,
            device=lam_arr.device,
            dtype=torch.float64,
        ),
        min=0.0,
    )
    transport_rel = max(float(transport_model_rel_sigma), 0.0)
    transport_abs = max(float(transport_model_abs_sigma), 0.0)
    spectrum_rel = max(float(spectrum_count_rel_sigma), 0.0)
    spectrum_abs = max(float(spectrum_count_abs_sigma), 0.0)
    low_count_abs = max(float(low_count_abs_sigma), 0.0)
    low_count_transition = max(float(low_count_transition_counts), 0.0)
    scale_ref = torch.maximum(z_arr, lam_arr)
    low_count_weight = torch.zeros_like(scale_ref)
    if low_count_abs > 0.0 and low_count_transition > 0.0:
        low_count_weight = low_count_transition / (scale_ref + low_count_transition)
    variance = (
        lam_arr
        + (transport_rel * lam_arr) ** 2
        + transport_abs**2
        + (spectrum_rel * scale_ref) ** 2
        + spectrum_abs**2
        + (low_count_abs * low_count_weight) ** 2
        + obs_var
    )
    return torch.clamp(variance, min=float(epsilon))


def count_log_likelihood(
    z_k: NDArray[np.float64],
    lambda_k: NDArray[np.float64],
    *,
    model: str = "poisson",
    transport_model_rel_sigma: float = 0.0,
    transport_model_abs_sigma: float = 0.0,
    spectrum_count_rel_sigma: float = 0.0,
    spectrum_count_abs_sigma: float = 0.0,
    low_count_abs_sigma: float = 0.0,
    low_count_transition_counts: float = 0.0,
    observation_count_variance: float | NDArray[np.float64] = 0.0,
    student_t_df: float = 5.0,
    epsilon: float = 1e-12,
) -> float:
    """
    Return a count log-likelihood with optional model-discrepancy variance.

    ``poisson`` preserves the original PF likelihood. ``gaussian`` and
    ``student_t`` allow non-integer spectrum-decomposition counts and inflate
    the variance for unmodelled Geant4 transport effects.
    """
    normalized_model = _normalize_count_likelihood_model(model)
    z_arr = np.asarray(z_k, dtype=float)
    lam_arr = np.maximum(np.asarray(lambda_k, dtype=float), float(epsilon))
    if normalized_model == "poisson":
        return poisson_log_likelihood(z_arr, lam_arr, epsilon=epsilon)

    variance = count_likelihood_variance(
        z_arr,
        lam_arr,
        transport_model_rel_sigma=transport_model_rel_sigma,
        transport_model_abs_sigma=transport_model_abs_sigma,
        spectrum_count_rel_sigma=spectrum_count_rel_sigma,
        spectrum_count_abs_sigma=spectrum_count_abs_sigma,
        low_count_abs_sigma=low_count_abs_sigma,
        low_count_transition_counts=low_count_transition_counts,
        observation_count_variance=observation_count_variance,
        epsilon=epsilon,
    )
    residual = z_arr - lam_arr
    if normalized_model == "gaussian":
        terms = -0.5 * ((residual**2) / variance + np.log(variance))
        return float(np.sum(terms))

    df = max(float(student_t_df), 1.0 + float(epsilon))
    terms = -0.5 * (df + 1.0) * np.log1p((residual**2) / (df * variance))
    terms -= 0.5 * np.log(variance)
    return float(np.sum(terms))


def delta_log_likelihood_remove(
    z_k: NDArray[np.float64],
    lambda_total: NDArray[np.float64],
    lambda_m: NDArray[np.float64],
    epsilon: float = 1e-12,
    model: str = "poisson",
    transport_model_rel_sigma: float = 0.0,
    transport_model_abs_sigma: float = 0.0,
    spectrum_count_rel_sigma: float = 0.0,
    spectrum_count_abs_sigma: float = 0.0,
    low_count_abs_sigma: float = 0.0,
    low_count_transition_counts: float = 0.0,
    observation_count_variance: float | NDArray[np.float64] = 0.0,
    student_t_df: float = 5.0,
) -> NDArray[np.float64]:
    """
    Compute ΔLL when removing each source m using a stable log1p formulation.

    lambda_m must be shaped (K, M). Returns a vector (M,).
    """
    if lambda_m.ndim != 2:
        raise ValueError("lambda_m must be a (K, M) array.")
    normalized_model = _normalize_count_likelihood_model(model)
    if normalized_model != "poisson":
        base_ll = count_log_likelihood(
            z_k,
            lambda_total,
            model=normalized_model,
            transport_model_rel_sigma=transport_model_rel_sigma,
            transport_model_abs_sigma=transport_model_abs_sigma,
            spectrum_count_rel_sigma=spectrum_count_rel_sigma,
            spectrum_count_abs_sigma=spectrum_count_abs_sigma,
            low_count_abs_sigma=low_count_abs_sigma,
            low_count_transition_counts=low_count_transition_counts,
            observation_count_variance=observation_count_variance,
            student_t_df=student_t_df,
            epsilon=epsilon,
        )
        delta_values = np.zeros(lambda_m.shape[1], dtype=float)
        for source_idx in range(lambda_m.shape[1]):
            reduced_lambda = np.maximum(
                lambda_total - lambda_m[:, source_idx],
                float(epsilon),
            )
            reduced_ll = count_log_likelihood(
                z_k,
                reduced_lambda,
                model=normalized_model,
                transport_model_rel_sigma=transport_model_rel_sigma,
                transport_model_abs_sigma=transport_model_abs_sigma,
                spectrum_count_rel_sigma=spectrum_count_rel_sigma,
                spectrum_count_abs_sigma=spectrum_count_abs_sigma,
                low_count_abs_sigma=low_count_abs_sigma,
                low_count_transition_counts=low_count_transition_counts,
                observation_count_variance=observation_count_variance,
                student_t_df=student_t_df,
                epsilon=epsilon,
            )
            delta_values[source_idx] = base_ll - reduced_ll
        return delta_values
    lambda_total_safe = np.maximum(lambda_total, float(epsilon))
    ratio = lambda_m / lambda_total_safe[:, None]
    ratio = np.clip(ratio, 0.0, 1.0 - float(epsilon))
    log_term = -np.log1p(-ratio)
    delta_ll = np.sum(z_k[:, None] * log_term - lambda_m, axis=0)
    return delta_ll


def delta_log_likelihood_update(
    z_k: NDArray[np.float64],
    lambda_old: NDArray[np.float64],
    lambda_new: NDArray[np.float64],
    epsilon: float = 1e-12,
    model: str = "poisson",
    transport_model_rel_sigma: float = 0.0,
    transport_model_abs_sigma: float = 0.0,
    spectrum_count_rel_sigma: float = 0.0,
    spectrum_count_abs_sigma: float = 0.0,
    low_count_abs_sigma: float = 0.0,
    low_count_transition_counts: float = 0.0,
    observation_count_variance: float | NDArray[np.float64] = 0.0,
    student_t_df: float = 5.0,
) -> float:
    """
    Compute ΔLL for replacing lambda_old with lambda_new across measurements.
    """
    normalized_model = _normalize_count_likelihood_model(model)
    if normalized_model != "poisson":
        ll_old = count_log_likelihood(
            z_k,
            lambda_old,
            model=normalized_model,
            transport_model_rel_sigma=transport_model_rel_sigma,
            transport_model_abs_sigma=transport_model_abs_sigma,
            spectrum_count_rel_sigma=spectrum_count_rel_sigma,
            spectrum_count_abs_sigma=spectrum_count_abs_sigma,
            low_count_abs_sigma=low_count_abs_sigma,
            low_count_transition_counts=low_count_transition_counts,
            observation_count_variance=observation_count_variance,
            student_t_df=student_t_df,
            epsilon=epsilon,
        )
        ll_new = count_log_likelihood(
            z_k,
            lambda_new,
            model=normalized_model,
            transport_model_rel_sigma=transport_model_rel_sigma,
            transport_model_abs_sigma=transport_model_abs_sigma,
            spectrum_count_rel_sigma=spectrum_count_rel_sigma,
            spectrum_count_abs_sigma=spectrum_count_abs_sigma,
            low_count_abs_sigma=low_count_abs_sigma,
            low_count_transition_counts=low_count_transition_counts,
            observation_count_variance=observation_count_variance,
            student_t_df=student_t_df,
            epsilon=epsilon,
        )
        return float(ll_new - ll_old)
    lambda_old_safe = np.maximum(lambda_old, float(epsilon))
    lambda_new_safe = np.maximum(lambda_new, float(epsilon))
    return float(np.sum(z_k * (np.log(lambda_new_safe) - np.log(lambda_old_safe)) - (lambda_new - lambda_old)))


def expected_counts_per_source(
    kernel: ContinuousKernel,
    isotope: str,
    detector_positions: NDArray[np.float64],
    sources: NDArray[np.float64],
    strengths: NDArray[np.float64],
    live_times: NDArray[np.float64],
    fe_indices: NDArray[np.int64] | None = None,
    pb_indices: NDArray[np.int64] | None = None,
    orient_indices: NDArray[np.int64] | None = None,
    source_scale: float | NDArray[np.float64] = 1.0,
) -> NDArray[np.float64]:
    """
    Return per-source expected counts Λ_{k,m} for each measurement k.

    Supports either paired Fe/Pb indices or single orientation indices.
    source_scale maps ideal source counts into the measurement domain while
    leaving the additive background term to the caller. It may be a scalar or a
    length-``num_meas`` vector for shield-pair-conditioned calibration.
    """
    if sources.size == 0:
        return np.zeros((len(live_times), 0), dtype=float)
    if sources.shape[0] != strengths.shape[0]:
        raise ValueError("sources and strengths must have matching length.")
    num_meas = int(len(live_times))
    num_sources = int(sources.shape[0])
    lambda_m = np.zeros((num_meas, num_sources), dtype=float)
    scale_arr = _source_scale_vector(source_scale, num_meas)
    if fe_indices is None or pb_indices is None:
        if orient_indices is None:
            raise ValueError("Either fe_indices/pb_indices or orient_indices must be provided.")
        fe_indices_use = np.asarray(orient_indices, dtype=int)
        pb_indices_use = fe_indices_use
    else:
        fe_indices_use = np.asarray(fe_indices, dtype=int)
        pb_indices_use = np.asarray(pb_indices, dtype=int)
    if hasattr(kernel, "kernel_values_selected_pairs_for_detectors"):
        values = kernel.kernel_values_selected_pairs_for_detectors(
            isotope=isotope,
            detector_positions=np.asarray(detector_positions, dtype=float),
            sources=np.asarray(sources, dtype=float),
            fe_indices=fe_indices_use,
            pb_indices=pb_indices_use,
        )
        values_arr = np.asarray(values, dtype=float)
        if values_arr.shape != (num_meas, num_sources):
            raise ValueError(
                "Batched selected-pair kernel returned shape "
                f"{values_arr.shape}, expected {(num_meas, num_sources)}."
            )
        strengths_arr = np.asarray(strengths, dtype=float)
        scale = np.asarray(live_times, dtype=float) * scale_arr
        return scale[:, None] * values_arr * strengths_arr[None, :]
    if getattr(kernel, "use_gpu", False) and hasattr(kernel, "kernel_values_pair"):
        scale = np.asarray(live_times, dtype=float) * scale_arr
        strengths_arr = np.asarray(strengths, dtype=float)
        for k in range(num_meas):
            values = kernel.kernel_values_pair(
                isotope=isotope,
                detector_pos=detector_positions[k],
                sources=sources,
                fe_index=int(fe_indices_use[k]),
                pb_index=int(pb_indices_use[k]),
            )
            lambda_m[k, :] = scale[k] * values * strengths_arr
        return lambda_m
    for k in range(num_meas):
        det = detector_positions[k]
        live_time = float(live_times[k])
        for m in range(num_sources):
            if fe_indices is not None and pb_indices is not None:
                kernel_val = kernel.kernel_value_pair(
                    isotope=isotope,
                    detector_pos=det,
                    source_pos=sources[m],
                    fe_index=int(fe_indices[k]),
                    pb_index=int(pb_indices[k]),
                )
            elif orient_indices is not None:
                kernel_val = kernel.kernel_value(
                    isotope=isotope,
                    detector_pos=det,
                    source_pos=sources[m],
                    orient_idx=int(orient_indices[k]),
                )
            else:
                raise ValueError("Either fe_indices/pb_indices or orient_indices must be provided.")
            lambda_m[k, m] = live_time * scale_arr[k] * kernel_val * float(strengths[m])
    return lambda_m


def _source_scale_vector(
    source_scale: float | NDArray[np.float64],
    num_meas: int,
) -> NDArray[np.float64]:
    """Return a non-negative source-response scale per measurement."""
    arr = np.asarray(source_scale, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.ones(max(int(num_meas), 0), dtype=float)
    if arr.size == 1:
        return np.full(max(int(num_meas), 0), max(float(arr[0]), 0.0), dtype=float)
    if arr.size != int(num_meas):
        raise ValueError("source_scale vector must match the number of measurements.")
    return np.maximum(arr.astype(float, copy=False), 0.0)
