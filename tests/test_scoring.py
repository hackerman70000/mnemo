from __future__ import annotations

import pytest

from mnemo.core.scoring import dataset_level_auc


def test_perfect_separation_auc():
    assert dataset_level_auc([0.95, 0.9, 0.85], [0.1, 0.05, 0.0]) == 1.0


def test_inverted_separation_auc():
    assert dataset_level_auc([0.0, 0.05, 0.1], [0.85, 0.9, 0.95]) == 0.0


def test_tied_scores_yield_half_auc():
    assert dataset_level_auc([0.5, 0.5], [0.5, 0.5]) == 0.5


def test_empty_inputs_raise():
    with pytest.raises(ValueError, match="non-empty"):
        dataset_level_auc([], [0.1])
    with pytest.raises(ValueError, match="non-empty"):
        dataset_level_auc([0.9], [])
