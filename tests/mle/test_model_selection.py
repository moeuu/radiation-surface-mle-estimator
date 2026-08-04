"""Tests for grouped one-standard-error regularization selection."""

from __future__ import annotations

import numpy as np

from three_d_estimation.model_selection import (
    RegularizationCandidate,
    grouped_kfold_indices,
    select_regularization_one_standard_error,
)


def test_grouped_folds_never_split_related_station_rows() -> None:
    """Rows from one station must remain entirely within one validation fold."""
    labels = (0, 0, 1, 1, 1, 2, 3, 3)
    folds = grouped_kfold_indices(labels, 3, random_seed=7)

    validation_sets = [set(indices.tolist()) for _fit, indices in folds]
    for group in ({0, 1}, {2, 3, 4}, {5}, {6, 7}):
        assert sum(group.issubset(rows) for rows in validation_sets) == 1


def test_one_standard_error_selects_stronger_tied_regularization() -> None:
    """The one-SE rule should prefer the simpler fit within the best error bar."""
    candidates = (
        RegularizationCandidate(0.0, 0.0),
        RegularizationCandidate(0.1, 0.1),
        RegularizationCandidate(1.0, 1.0),
    )
    folds = grouped_kfold_indices((0, 1, 2, 3), 2, random_seed=0)
    scores = {
        candidates[0]: (1.0, 1.2),
        candidates[1]: (1.02, 1.08),
        candidates[2]: (2.0, 2.0),
    }

    def score(
        candidate: RegularizationCandidate, fit: np.ndarray, valid: np.ndarray
    ) -> float:
        """Return one deterministic score indexed by the validation fold."""
        del fit
        fold = 0 if 0 in valid else 1
        return scores[candidate][fold]

    selected = select_regularization_one_standard_error(
        candidates,
        folds,
        score,
        group_labels=(0, 1, 2, 3),
    )

    assert selected.selected == candidates[1]
    assert selected.best_mean_index == 1
    assert selected.fold_group_ids
