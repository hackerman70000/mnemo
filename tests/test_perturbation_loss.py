from __future__ import annotations

import random

import numpy as np
import pytest

from mnemo.detectors.perturbation_loss import (
    PerturbationLoss,
    adjacent_word_swap,
    random_word_drop,
)
from tests.conftest import FakeModel


def test_adjacent_word_swap_changes_order():
    rng = random.Random(0)
    text = "alpha beta gamma delta"
    swapped = adjacent_word_swap(text, rng, n_swaps=1)
    assert sorted(swapped.split()) == sorted(text.split())
    assert swapped != text or text.count(" ") == 0


def test_adjacent_word_swap_handles_short_text():
    assert adjacent_word_swap("solo", random.Random(0)) == "solo"


def test_random_word_drop_drops_words():
    rng = random.Random(0)
    text = "one two three four five six seven eight"
    dropped = random_word_drop(text, rng, drop_fraction=0.5)
    assert len(dropped.split()) <= len(text.split())


def test_random_word_drop_rejects_invalid_fraction():
    with pytest.raises(ValueError, match="drop_fraction"):
        random_word_drop("a b c", random.Random(0), drop_fraction=1.0)


def test_perturbation_loss_high_when_perturbations_drop_logprob():
    """Memorised mock: original logprob -1, perturbed logprob -3."""

    def logprobs(text: str) -> np.ndarray:
        n = max(1, len(text.split()))
        return np.full(n, -1.0 if text == "alpha beta gamma delta epsilon" else -3.0)

    model = FakeModel("memorised", logprobs)

    detector = PerturbationLoss(perturbation_fn=adjacent_word_swap, n_perturbations=10)
    score = detector.score_sample("alpha beta gamma delta epsilon", model)
    assert score == pytest.approx(2.0, abs=0.01)


def test_perturbation_loss_low_when_perturbations_dont_change_logprob():
    """Unseen mock: logprob constant regardless of perturbation."""
    model = FakeModel("unseen", lambda text: np.full(max(1, len(text.split())), -2.0))

    detector = PerturbationLoss(perturbation_fn=adjacent_word_swap, n_perturbations=10)
    score = detector.score_sample("alpha beta gamma delta", model)
    assert score == pytest.approx(0.0)


def test_perturbation_loss_rejects_invalid_n_perturbations():
    with pytest.raises(ValueError, match="n_perturbations"):
        PerturbationLoss(n_perturbations=0)


def test_perturbation_loss_handles_empty_logprobs():
    model = FakeModel("empty", lambda _t: np.array([]))
    detector = PerturbationLoss(n_perturbations=5)
    assert detector.score_sample("anything", model) == 0.0


def test_perturbation_loss_handles_no_change_perturbation():
    """Identity perturbation produces no usable perturbed losses → score 0."""
    model = FakeModel("any", lambda text: np.full(max(1, len(text.split())), -1.0))
    detector = PerturbationLoss(
        perturbation_fn=lambda text, _rng: text,
        n_perturbations=10,
    )
    assert detector.score_sample("a b c d", model) == 0.0
