"""Merge PF outputs, reject spurious sources, and evaluate best-case measurement checks."""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
from numpy.typing import NDArray

from pf.estimator import RotatingShieldPFEstimator
from pf.likelihood import delta_log_likelihood_remove, expected_counts_per_source


def prune_spurious_sources(
    z_k: NDArray[np.float64],
    live_times: NDArray[np.float64],
    positions: NDArray[np.float64],
    strengths: NDArray[np.float64],
    background: float | NDArray[np.float64],
    forward_model: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    method: str = "legacy",
    params: Dict[str, float] | None = None,
) -> NDArray[np.bool_]:
    """
    Prune spurious sources using delta-LL, best-case residual gating, or legacy dominance.

    forward_model must return a (K, M) array of per-source expected counts.
    """
    if positions.size == 0:
        return np.ones(0, dtype=bool)
    if params is None:
        params = {}
    epsilon = float(params.get("epsilon", 1e-12))
    tau_mix = float(params.get("tau_mix", 0.9))
    delta_ll_min = float(params.get("deltaLL_min", 0.0))
    penalty_d = float(params.get("penalty_d", 0.0))
    alpha = float(params.get("alpha", 0.7))
    lambda_min = float(params.get("lambda_min", 0.0))
    lrt_threshold = float(params.get("lrt_threshold", 0.0))
    min_strength_abs = params.get("min_strength_abs")
    min_strength_ratio = params.get("min_strength_ratio")
    min_obs_count = float(params.get("min_obs_count", 0.0))
    likelihood_kwargs = {
        "model": str(params.get("count_likelihood_model", params.get("model", "poisson"))),
        "transport_model_rel_sigma": float(params.get("transport_model_rel_sigma", 0.0)),
        "transport_model_abs_sigma": float(params.get("transport_model_abs_sigma", 0.0)),
        "spectrum_count_rel_sigma": float(params.get("spectrum_count_rel_sigma", 0.0)),
        "spectrum_count_abs_sigma": float(params.get("spectrum_count_abs_sigma", 0.0)),
        "low_count_abs_sigma": float(params.get("low_count_abs_sigma", 0.0)),
        "low_count_transition_counts": float(params.get("low_count_transition_counts", 0.0)),
        "student_t_df": float(params.get("count_likelihood_df", 5.0)),
    }

    lambda_m = forward_model(positions, strengths)
    if lambda_m.ndim != 2:
        raise ValueError("forward_model must return a (K, M) array.")
    if lambda_m.shape[1] != strengths.shape[0]:
        raise ValueError("forward_model output must have one column per source.")
    if np.isscalar(background):
        background_k = np.full(lambda_m.shape[0], float(background), dtype=float)
    else:
        background_k = np.asarray(background, dtype=float)
    lambda_total = background_k + np.sum(lambda_m, axis=1)

    max_strength = float(np.max(strengths)) if strengths.size else 0.0
    min_strength = 0.0
    if min_strength_abs is not None:
        min_strength = max(min_strength, float(min_strength_abs))
    if min_strength_ratio is not None:
        min_strength = max(min_strength, float(min_strength_ratio) * max_strength)
    strength_ok = strengths >= min_strength if min_strength > 0.0 else np.ones_like(strengths, dtype=bool)

    method_key = method.lower()
    if method_key not in {"deltall", "bestcase", "legacy"}:
        raise ValueError(f"Unsupported pruning method: {method}")

    if min_obs_count > 0.0:
        obs_mask = z_k > min_obs_count
        if not np.any(obs_mask):
            return strength_ok
        z_use = z_k[obs_mask]
        lambda_m_use = lambda_m[obs_mask]
        lambda_total_use = lambda_total[obs_mask]
    else:
        z_use = z_k
        lambda_m_use = lambda_m
        lambda_total_use = lambda_total

    if method_key == "deltall":
        if z_use.size == 0:
            return strength_ok
        delta_ll = delta_log_likelihood_remove(
            z_use,
            lambda_total_use,
            lambda_m_use,
            epsilon=epsilon,
            **likelihood_kwargs,
        )
        score = delta_ll
        if penalty_d > 0.0:
            penalty = 0.5 * penalty_d * np.log(max(int(z_use.size), 1))
            score = delta_ll - penalty
        keep_mask = (delta_ll >= delta_ll_min) & (score > 0.0)
        return keep_mask & strength_ok

    if method_key == "legacy":
        denom = z_use + epsilon
        ratios = np.max(lambda_m_use / denom[:, None], axis=0)
        keep_mask = ratios >= tau_mix
        return keep_mask & strength_ok

    if z_use.size == 0:
        return strength_ok
    keep_mask = np.ones(lambda_m_use.shape[1], dtype=bool)
    for m in range(lambda_m_use.shape[1]):
        lam_m = lambda_m_use[:, m]
        k_star = int(np.argmax(lam_m))
        lam_star = float(lam_m[k_star])
        if lam_star < lambda_min:
            keep_mask[m] = False
            continue
        total_star = float(lambda_total_use[k_star])
        residual = float(z_use[k_star] - (total_star - lam_star))
        if residual >= alpha * lam_star:
            continue
        delta_ll = float(
            delta_log_likelihood_remove(
                np.array([z_use[k_star]], dtype=float),
                np.array([total_star], dtype=float),
                np.array([[lam_star]], dtype=float),
                epsilon=epsilon,
                **likelihood_kwargs,
            )[0]
        )
        if delta_ll < lrt_threshold:
            keep_mask[m] = False
    return keep_mask & strength_ok


def prune_spurious_sources_continuous(
    estimator: RotatingShieldPFEstimator,
    method: str = "legacy",
    params: Dict[str, float] | None = None,
    tau_mix: float = 0.9,
    epsilon: float = 1e-6,
    min_support: int = 1,
    min_obs_count: float = 0.0,
    min_strength_abs: float | None = None,
    min_strength_ratio: float | None = None,
    estimate_snapshot: Dict[
        str, tuple[NDArray[np.float64], NDArray[np.float64]]
    ]
    | None = None,
) -> Dict[str, NDArray[np.bool_]]:
    """
    Apply spurious-source pruning to continuous PF estimates.

    Uses MMSE estimates as candidate sources. Returns a keep mask per isotope that
    can be applied to continuous particle source indices by order.  When supplied,
    estimate_snapshot pins pruning to the caller's report estimate generation so
    report-side particle pruning cannot mix masks from a later estimate revision.
    """
    if not estimator.measurements:
        return {iso: np.ones(0, dtype=bool) for iso in estimator.filters}

    use_gpu_kernel = False
    pf_config = getattr(estimator, "pf_config", None)
    if pf_config is not None and bool(getattr(pf_config, "use_gpu", False)):
        from pf import gpu_utils

        use_gpu_kernel = bool(gpu_utils.torch_available())
    kernel = estimator.continuous_kernel(use_gpu=use_gpu_kernel)
    keep_masks: Dict[str, NDArray[np.bool_]] = {}
    estimates = estimate_snapshot if estimate_snapshot is not None else estimator.estimates()
    method_params = dict(params or {})
    method_params.setdefault("epsilon", float(epsilon))
    method_params.setdefault("tau_mix", float(tau_mix))
    if min_obs_count > 0.0:
        method_params.setdefault("min_obs_count", float(min_obs_count))
    if min_strength_abs is not None:
        method_params.setdefault("min_strength_abs", float(min_strength_abs))
    if min_strength_ratio is not None:
        method_params.setdefault("min_strength_ratio", float(min_strength_ratio))

    for iso, (positions, strengths) in estimates.items():
        if positions.size == 0:
            keep_masks[iso] = np.ones(0, dtype=bool)
            continue
        z_list: list[float] = []
        live_times: list[float] = []
        poses: list[NDArray[np.float64]] = []
        fe_indices: list[int] = []
        pb_indices: list[int] = []
        for rec in estimator.measurements:
            if iso not in rec.z_k:
                continue
            z_list.append(float(rec.z_k[iso]))
            live_times.append(float(rec.live_time_s))
            poses.append(estimator.poses[rec.pose_idx])
            if rec.fe_index is not None and rec.pb_index is not None:
                fe_indices.append(int(rec.fe_index))
                pb_indices.append(int(rec.pb_index))
            else:
                fe_indices.append(int(rec.orient_idx))
                pb_indices.append(int(rec.orient_idx))
        assert len(z_list) == len(live_times) == len(poses) == len(fe_indices) == len(pb_indices)
        if not z_list:
            keep_masks[iso] = np.ones(positions.shape[0], dtype=bool)
            continue
        z_k = np.asarray(z_list, dtype=float)
        live_times_arr = np.asarray(live_times, dtype=float)
        poses_arr = np.asarray(poses, dtype=float)
        fe_arr = np.asarray(fe_indices, dtype=int)
        pb_arr = np.asarray(pb_indices, dtype=int)

        if min_support > 0 and int(z_k.size) < int(min_support):
            keep_masks[iso] = np.ones(positions.shape[0], dtype=bool)
            continue
        params_for_iso = dict(method_params)
        if iso in estimator.filters:
            likelihood = estimator.filters[iso]._count_likelihood_kwargs()
            params_for_iso.setdefault("count_likelihood_model", str(likelihood["model"]))
            params_for_iso.setdefault(
                "transport_model_rel_sigma",
                float(likelihood["transport_model_rel_sigma"]),
            )
            params_for_iso.setdefault(
                "transport_model_abs_sigma",
                float(likelihood["transport_model_abs_sigma"]),
            )
            params_for_iso.setdefault(
                "spectrum_count_rel_sigma",
                float(likelihood["spectrum_count_rel_sigma"]),
            )
            params_for_iso.setdefault(
                "spectrum_count_abs_sigma",
                float(likelihood["spectrum_count_abs_sigma"]),
            )
            params_for_iso.setdefault(
                "low_count_abs_sigma",
                float(likelihood["low_count_abs_sigma"]),
            )
            params_for_iso.setdefault(
                "low_count_transition_counts",
                float(likelihood["low_count_transition_counts"]),
            )
            params_for_iso.setdefault(
                "count_likelihood_df",
                float(likelihood["student_t_df"]),
            )

        def _forward_model(pos: NDArray[np.float64], strg: NDArray[np.float64]) -> NDArray[np.float64]:
            """Return per-source expected counts for one isotope estimate."""
            return expected_counts_per_source(
                kernel=kernel,
                isotope=iso,
                detector_positions=poses_arr,
                sources=pos,
                strengths=strg,
                live_times=live_times_arr,
                fe_indices=fe_arr,
                pb_indices=pb_arr,
                source_scale=estimator.response_scales_for_measurements(
                    iso,
                    fe_arr,
                    pb_arr,
                ),
            )

        background_rate = 0.0
        if iso in estimator.filters and estimator.filters[iso].continuous_particles:
            background_rate = float(estimator.filters[iso].best_particle().state.background)
        elif iso in estimator.filters:
            background_rate = estimator.filters[iso].config.background_level
            if isinstance(background_rate, dict):
                background_rate = float(background_rate.get(iso, 0.0))
        background_counts = float(background_rate) * live_times_arr
        keep_masks[iso] = prune_spurious_sources(
            z_k=z_k,
            live_times=live_times_arr,
            positions=positions,
            strengths=strengths,
            background=background_counts,
            forward_model=_forward_model,
            method=method,
            params=params_for_iso,
        )
    return keep_masks
