"""Grouped cross-validation and one-standard-error regularization selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class RegularizationCandidate:
    """Identify one nonnegative L1/graph-TV candidate."""

    l1_weight: float
    tv_weight: float

    def __post_init__(self) -> None:
        """Validate finite nonnegative regularization strengths."""
        if any(
            not np.isfinite(float(value)) or float(value) < 0.0
            for value in (self.l1_weight, self.tv_weight)
        ):
            raise ValueError("Regularization weights must be finite and nonnegative.")


@dataclass(frozen=True, slots=True)
class RegularizationCVResult:
    """Store grouped-fold scores and the one-standard-error selection."""

    selected: RegularizationCandidate
    candidates: tuple[RegularizationCandidate, ...]
    fold_scores: NDArray[np.float64]
    means: NDArray[np.float64]
    standard_errors: NDArray[np.float64]
    best_mean_index: int
    one_standard_error_threshold: float
    fold_group_ids: tuple[tuple[int, ...], ...]

    def to_dict(self) -> dict[str, object]:
        """Return strict JSON-safe model-selection diagnostics."""
        return {
            "method": "grouped_kfold_one_standard_error",
            "selected_l1_weight": float(self.selected.l1_weight),
            "selected_tv_weight": float(self.selected.tv_weight),
            "best_mean_index": int(self.best_mean_index),
            "one_standard_error_threshold": float(self.one_standard_error_threshold),
            "candidates": [
                {
                    "l1_weight": float(candidate.l1_weight),
                    "tv_weight": float(candidate.tv_weight),
                    "fold_scores": self.fold_scores[index].tolist(),
                    "mean_score": float(self.means[index]),
                    "standard_error": float(self.standard_errors[index]),
                }
                for index, candidate in enumerate(self.candidates)
            ],
            "fold_group_ids": [list(values) for values in self.fold_group_ids],
        }


def grouped_kfold_indices(
    group_labels: Sequence[int],
    fold_count: int,
    *,
    random_seed: int,
) -> tuple[tuple[NDArray[np.int64], NDArray[np.int64]], ...]:
    """Split complete related groups into deterministic balanced folds."""
    labels = np.asarray(group_labels, dtype=np.int64)
    if labels.ndim != 1 or labels.size < 2:
        raise ValueError("group_labels must contain at least two rows.")
    unique, counts = np.unique(labels, return_counts=True)
    if isinstance(fold_count, bool) or not 2 <= int(fold_count) <= unique.size:
        raise ValueError("fold_count must lie between two and the group count.")
    rng = np.random.default_rng(int(random_seed))
    tie_rank = {int(value): rank for rank, value in enumerate(rng.permutation(unique))}
    ordered = sorted(
        zip(unique.tolist(), counts.tolist(), strict=True),
        key=lambda item: (-item[1], tie_rank[int(item[0])], int(item[0])),
    )
    fold_groups: list[list[int]] = [[] for _ in range(int(fold_count))]
    fold_sizes = np.zeros(int(fold_count), dtype=np.int64)
    for label, count in ordered:
        fold = int(np.argmin(fold_sizes))
        fold_groups[fold].append(int(label))
        fold_sizes[fold] += int(count)
    all_rows = np.arange(labels.size, dtype=np.int64)
    result = []
    for groups in fold_groups:
        validation = np.flatnonzero(np.isin(labels, groups)).astype(np.int64)
        fit = np.setdiff1d(all_rows, validation, assume_unique=True)
        if fit.size == 0 or validation.size == 0:
            raise RuntimeError("Grouped CV produced an empty fit or validation fold.")
        result.append((fit, validation))
    return tuple(result)


def select_regularization_one_standard_error(
    candidates: Sequence[RegularizationCandidate],
    folds: Sequence[tuple[NDArray[np.int64], NDArray[np.int64]]],
    score: Callable[
        [RegularizationCandidate, NDArray[np.int64], NDArray[np.int64]],
        float,
    ],
    *,
    use_one_standard_error: bool = True,
    group_labels: Sequence[int] | None = None,
) -> RegularizationCVResult:
    """Evaluate grouped folds and choose the strongest statistically tied fit."""
    candidate_values = tuple(candidates)
    fold_values = tuple(folds)
    if not candidate_values or not fold_values:
        raise ValueError("Regularization selection requires candidates and folds.")
    scores = np.empty((len(candidate_values), len(fold_values)), dtype=np.float64)
    for candidate_index, candidate in enumerate(candidate_values):
        for fold_index, (fit_indices, validation_indices) in enumerate(fold_values):
            value = float(score(candidate, fit_indices, validation_indices))
            if not np.isfinite(value):
                raise ValueError("Cross-validation scores must be finite.")
            scores[candidate_index, fold_index] = value
    means = np.mean(scores, axis=1)
    standard_errors = (
        np.std(scores, axis=1, ddof=1) / np.sqrt(scores.shape[1])
        if scores.shape[1] > 1
        else np.zeros(scores.shape[0], dtype=np.float64)
    )
    best_index = int(np.argmin(means))
    threshold = float(
        means[best_index]
        + (standard_errors[best_index] if use_one_standard_error else 0.0)
    )
    eligible = np.flatnonzero(means <= threshold + 1.0e-15)
    selected_index = max(
        eligible.tolist(),
        key=lambda index: (
            candidate_values[index].l1_weight + candidate_values[index].tv_weight,
            candidate_values[index].tv_weight,
            candidate_values[index].l1_weight,
            -index,
        ),
    )
    fold_group_ids: tuple[tuple[int, ...], ...] = ()
    if group_labels is not None:
        labels = np.asarray(group_labels, dtype=np.int64)
        fold_group_ids = tuple(
            tuple(sorted({int(labels[index]) for index in validation}))
            for _fit, validation in fold_values
        )
    return RegularizationCVResult(
        selected=candidate_values[selected_index],
        candidates=candidate_values,
        fold_scores=scores,
        means=means,
        standard_errors=standard_errors,
        best_mean_index=best_index,
        one_standard_error_threshold=threshold,
        fold_group_ids=fold_group_ids,
    )


__all__ = [
    "RegularizationCVResult",
    "RegularizationCandidate",
    "grouped_kfold_indices",
    "select_regularization_one_standard_error",
]
