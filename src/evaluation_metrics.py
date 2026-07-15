"""Compute and report evaluation metrics for PF source estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

POSITION_ERROR_TARGET_M = 0.5


@dataclass(frozen=True)
class Source:
    """Lightweight source representation for evaluation."""

    pos: NDArray[np.float64]
    strength: float


def _as_array(value: Sequence[float]) -> NDArray[np.float64]:
    """Return a NumPy array for a position sequence."""
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError("Position must be a 3-element sequence.")
    return arr


def _extract_strength(obj: Any) -> float | None:
    """Extract strength from a dict/object if present."""
    for key in ("strength", "intensity_cps_1m", "intensity"):
        if isinstance(obj, dict) and key in obj:
            return float(obj[key])
        if hasattr(obj, key):
            return float(getattr(obj, key))
    return None


def _extract_position(obj: Any) -> NDArray[np.float64] | None:
    """Extract a position array from a dict/object if present."""
    if isinstance(obj, dict):
        if "pos" in obj:
            return _as_array(obj["pos"])
        if "position" in obj:
            return _as_array(obj["position"])
    if hasattr(obj, "pos"):
        return _as_array(getattr(obj, "pos"))
    if hasattr(obj, "position"):
        return _as_array(getattr(obj, "position"))
    return None


def _normalize_source(entry: Any) -> Source:
    """Convert various source-like entries to Source."""
    if isinstance(entry, Source):
        return entry
    if isinstance(entry, (tuple, list, np.ndarray)) and len(entry) == 4:
        pos = _as_array(entry[:3])
        strength = float(entry[3])
        return Source(pos=pos, strength=strength)
    pos = _extract_position(entry)
    strength = _extract_strength(entry)
    if pos is None or strength is None:
        raise ValueError("Unsupported source entry format.")
    return Source(pos=pos, strength=float(strength))


def _normalize_sources(entries: Iterable[Any] | None) -> List[Source]:
    """Normalize a list of source-like entries."""
    if entries is None:
        return []
    return [_normalize_source(entry) for entry in entries]


def _pairwise_distances(
    gt: List[Source],
    est: List[Source],
) -> NDArray[np.float64]:
    """Return pairwise distance matrix between GT and EST sources."""
    gt_pos = np.vstack([s.pos for s in gt]) if gt else np.zeros((0, 3), dtype=float)
    est_pos = np.vstack([s.pos for s in est]) if est else np.zeros((0, 3), dtype=float)
    if gt_pos.size == 0 or est_pos.size == 0:
        return np.zeros((len(gt), len(est)), dtype=float)
    diff = gt_pos[:, None, :] - est_pos[None, :, :]
    return np.linalg.norm(diff, axis=2)


def _hungarian_assignment(cost: NDArray[np.float64]) -> List[Tuple[int, int]]:
    """Return minimal-cost assignment pairs using Hungarian or greedy fallback."""
    if cost.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return _greedy_assignment(cost)
    row_idx, col_idx = linear_sum_assignment(cost)
    return list(zip(row_idx.tolist(), col_idx.tolist()))


def _greedy_assignment(cost: NDArray[np.float64]) -> List[Tuple[int, int]]:
    """Greedy assignment by selecting smallest distances iteratively."""
    if cost.size == 0:
        return []
    remaining_rows = set(range(cost.shape[0]))
    remaining_cols = set(range(cost.shape[1]))
    pairs: List[Tuple[int, int]] = []
    while remaining_rows and remaining_cols:
        best_pair = None
        best_val = float("inf")
        for i in remaining_rows:
            for j in remaining_cols:
                val = float(cost[i, j])
                if val < best_val:
                    best_val = val
                    best_pair = (i, j)
        if best_pair is None:
            break
        pairs.append(best_pair)
        remaining_rows.remove(best_pair[0])
        remaining_cols.remove(best_pair[1])
    return pairs


def _summary_stats(values: Sequence[float]) -> Dict[str, float | None]:
    """Return summary statistics for a list of values."""
    if not values:
        return {
            "mean": None,
            "median": None,
            "rmse": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=float)
    rmse = float(np.sqrt(np.mean(arr**2)))
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "rmse": rmse,
        "max": float(np.max(arr)),
    }


def _summary_abs(values: Sequence[float]) -> Dict[str, float | None]:
    """Return summary statistics for absolute errors."""
    if not values:
        return {
            "mean": None,
            "median": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


def _position_error_summary(values: Sequence[float]) -> Dict[str, float | bool | None]:
    """Return position-error summary stats augmented with the fixed target check."""
    summary = _summary_stats(values)
    mean_error = summary["mean"]
    summary["target_m"] = float(POSITION_ERROR_TARGET_M)
    summary["within_target"] = None if mean_error is None else bool(mean_error <= POSITION_ERROR_TARGET_M)
    return summary


def _threshold_count_summary(
    *,
    gt_count: int,
    est_count: int,
    assigned_distances: Sequence[float],
    thresholds_m: Sequence[float],
) -> Dict[str, Dict[str, float | int]]:
    """Return recall/precision summaries at localization distance thresholds."""
    distances = np.asarray(assigned_distances, dtype=float)
    payload: Dict[str, Dict[str, float | int]] = {}
    for threshold in thresholds_m:
        radius = max(float(threshold), 0.0)
        true_positive = int(np.count_nonzero(distances <= radius))
        false_positive = max(0, int(est_count) - true_positive)
        false_negative = max(0, int(gt_count) - true_positive)
        precision = (
            true_positive / float(true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )
        recall = (
            true_positive / float(true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0.0
        )
        payload[f"{radius:g}m"] = {
            "radius_m": radius,
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
            "precision": float(precision),
            "recall": float(recall),
        }
    return payload


def compute_metrics(
    gt_by_iso: Dict[str, List[Any]],
    est_by_iso: Dict[str, List[Any]],
    *,
    match_radius_m: float,
    distance_thresholds_m: Sequence[float] = (0.5, 1.0, 2.0, 3.0),
    match_strength_weight: float = 2.0,
    match_distance_weight: float = 1.0,
    outside_radius_penalty: float = 1e3,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute per-isotope evaluation metrics for estimated sources.

    Matching uses a weighted cost based on position and strength differences.
    Pairs outside match_radius_m receive an additional penalty to prefer
    within-radius matches when available.
    Position and strength errors are summarized over assigned matches.
    """
    eps = 1e-12
    isotopes = sorted(set(gt_by_iso.keys()) | set(est_by_iso.keys()))
    metrics: Dict[str, Dict[str, Any]] = {}
    for iso in isotopes:
        gt = _normalize_sources(gt_by_iso.get(iso, []))
        est = _normalize_sources(est_by_iso.get(iso, []))
        dist = _pairwise_distances(gt, est)
        if dist.size == 0:
            assignments = []
        else:
            q_true = np.asarray([src.strength for src in gt], dtype=float)
            q_hat = np.asarray([src.strength for src in est], dtype=float)
            strength_diff = np.abs(q_true[:, None] - q_hat[None, :]) / np.maximum(q_true[:, None], eps)
            within = dist <= float(match_radius_m)
            cost = float(match_distance_weight) * dist + float(match_strength_weight) * strength_diff
            if float(outside_radius_penalty) > 0.0:
                cost = np.where(within, cost, cost + float(outside_radius_penalty))
            assignments = _hungarian_assignment(cost)
        matched: List[Tuple[int, int, float]] = []
        for i, j in assignments:
            d = float(dist[i, j])
            matched.append((i, j, d))
        fp = max(0, len(est) - len(matched))
        fn = max(0, len(gt) - len(matched))
        pos_errors = [d for _, _, d in matched]
        abs_errors = []
        rel_errors = []
        match_details = []
        for i, j, d in matched:
            q_true = float(gt[i].strength)
            q_hat = float(est[j].strength)
            abs_err = abs(q_hat - q_true)
            rel_err = abs_err / max(q_true, eps) * 100.0
            abs_errors.append(abs_err)
            rel_errors.append(rel_err)
            match_details.append(
                {
                    "gt_index": i,
                    "est_index": j,
                    "distance": d,
                    "within_radius": bool(d <= float(match_radius_m)),
                    "q_true": q_true,
                    "q_hat": q_hat,
                    "abs_err": abs_err,
                    "rel_err_pct": rel_err,
                }
            )
        metrics[iso] = {
            "counts": {
                "gt": len(gt),
                "est": len(est),
                "assigned": len(matched),
                "matched": len(matched),
                "fp": fp,
                "fn": fn,
                "source_count_error": int(len(est) - len(gt)),
                "source_count_abs_error": int(abs(len(est) - len(gt))),
            },
            "threshold_counts": _threshold_count_summary(
                gt_count=len(gt),
                est_count=len(est),
                assigned_distances=pos_errors,
                thresholds_m=distance_thresholds_m,
            ),
            "position_error": _position_error_summary(pos_errors),
            "intensity_abs_error": _summary_abs(abs_errors),
            "intensity_rel_error_pct": _summary_abs(rel_errors),
            "matches": match_details,
        }
    return {
        "match_radius_m": float(match_radius_m),
        "isotopes": metrics,
    }


def _format_value(value: float | None) -> str:
    """Format a numeric value or return 'n/a'."""
    if value is None:
        return "n/a"
    return f"{value:.4g}"


def print_metrics_report(metrics: Dict[str, Any]) -> None:
    """Print a human-readable metrics report."""
    isotopes = metrics.get("isotopes", {})
    print("=== Evaluation Metrics (PF Final) ===")
    for iso in sorted(isotopes.keys()):
        data = isotopes[iso]
        counts = data["counts"]
        assigned = counts.get("assigned", None)
        pos_err = data["position_error"]
        abs_err = data["intensity_abs_error"]
        rel_err = data["intensity_rel_error_pct"]
        print(f"\n[Isotope: {iso}]")
        assigned_msg = ""
        if assigned is not None:
            assigned_msg = f", Assigned={assigned}"
        print(
            f"  GT={counts['gt']}, EST={counts['est']}{assigned_msg}, "
            f"TP={counts['matched']}, FP={counts['fp']}, FN={counts['fn']}"
        )
        print(
            "  Position error [m]: "
            f"mean={_format_value(pos_err['mean'])}, "
            f"median={_format_value(pos_err['median'])}, "
            f"rmse={_format_value(pos_err['rmse'])}, "
            f"max={_format_value(pos_err['max'])}"
        )
        print(
            "  Position target [m]: "
            f"<={_format_value(pos_err.get('target_m'))}, "
            f"within_target={pos_err.get('within_target')}"
        )
        threshold_counts = data.get("threshold_counts", {})
        if threshold_counts:
            threshold_msg = ", ".join(
                f"{key}:P={value['precision']:.2f}/R={value['recall']:.2f}"
                for key, value in sorted(
                    threshold_counts.items(),
                    key=lambda item: float(item[1]["radius_m"]),
                )
            )
            print(f"  Threshold precision/recall: {threshold_msg}")
        print(
            "  Intensity abs error [cps@1m]: "
            f"mean={_format_value(abs_err['mean'])}, "
            f"median={_format_value(abs_err['median'])}, "
            f"max={_format_value(abs_err['max'])}"
        )
        print(
            "  Intensity rel error [%]: "
            f"mean={_format_value(rel_err['mean'])}, "
            f"median={_format_value(rel_err['median'])}, "
            f"max={_format_value(rel_err['max'])}"
        )
        matches = data.get("matches", [])
        if not matches:
            print("  Matches: none")
            continue
        print("  Matches:")
        for m in matches:
            print(
                "    "
                f"GT#{m['gt_index']} -> EST#{m['est_index']} : "
                f"d={m['distance']:.3f} m, "
                f"q_true={m['q_true']:.3f}, q_hat={m['q_hat']:.3f}, "
                f"abs={m['abs_err']:.3f}, rel={m['rel_err_pct']:.2f}%"
            )


def save_metrics_json(metrics: Dict[str, Any], out_path: str) -> None:
    """Save metrics to a JSON file."""
    import json
    from pathlib import Path

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
