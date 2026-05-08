from __future__ import annotations

import random

import numpy as np
import pytest

from mnemo.detectors.perturbation_loss import (
    PerturbationLoss,
    adjacent_word_swap,
    butter_fingers,
    change_char_case,
    random_word_drop,
    synonym_substitution,
    underscore_trick,
    whitespace_perturbation,
)
from tests.conftest import FakeModel

# ---------------------------------------------------------------------------
# Perturbation function unit tests
# ---------------------------------------------------------------------------


def test_adjacent_word_swap_changes_order():
    rng = random.Random(0)
    text = "the quick brown fox jumps over the lazy dog"
    perturbed = adjacent_word_swap(text, rng, n_swaps=5)
    assert perturbed != text
    assert set(perturbed.split()) == set(text.split())


def test_adjacent_word_swap_handles_short_text():
    rng = random.Random(0)
    assert adjacent_word_swap("x", rng) == "x"
    assert adjacent_word_swap("x y", rng) in ("x y", "y x")


def test_random_word_drop_drops_words():
    rng = random.Random(0)
    text = "one two three four five"
    dropped = random_word_drop(text, rng, drop_fraction=0.4)
    assert len(dropped.split()) < len(text.split())


def test_random_word_drop_rejects_invalid_fraction():
    with pytest.raises(ValueError, match="drop_fraction"):
        random_word_drop("x", random.Random(0), drop_fraction=1.5)


def test_butter_fingers_changes_text():
    rng = random.Random(0)
    text = "hello world"
    perturbed = butter_fingers(text, rng, prob=1.0)
    assert perturbed != text
    assert len(perturbed) == len(text)


def test_synonym_substitution_changes_text():
    rng = random.Random(0)
    text = "this is a good test"
    perturbed = synonym_substitution(text, rng, prob=1.0)
    assert perturbed != text


def test_change_char_case_changes_text():
    rng = random.Random(0)
    text = "Hello World"
    perturbed = change_char_case(text, rng, prob=1.0)
    assert perturbed != text
    assert perturbed.lower() == text.lower()


def test_whitespace_perturbation_changes_text():
    rng = random.Random(0)
    text = "hello world test"
    perturbed = whitespace_perturbation(text, rng, prob=1.0)
    assert perturbed != text


def test_underscore_trick_changes_text():
    rng = random.Random(0)
    text = "hello world"
    perturbed = underscore_trick(text, rng, prob=1.0)
    assert "_" in perturbed


# ---------------------------------------------------------------------------
# PerturbationLoss detector tests
# ---------------------------------------------------------------------------


def test_perturbation_loss_high_when_perturbations_drop_logprob():
    """Model gives high logprob on original, low on everything else."""

    def scorer(text: str) -> np.ndarray:
        return np.array([-0.1, -0.1, -0.1]) if "ORIGINAL" in text else np.array([-3.0, -3.0, -3.0])

    model = FakeModel("high", scorer)
    det = PerturbationLoss(
        perturbation_fn=butter_fingers, n_perturbations=5, seed=0
    )
    score = det.score_sample("ORIGINAL sentence here", model)
    assert score > 1.0


def test_perturbation_loss_low_when_perturbations_dont_change_logprob():
    def scorer(_text: str) -> np.ndarray:
        return np.array([-2.0, -2.0])

    model = FakeModel("flat", scorer)
    det = PerturbationLoss(n_perturbations=5, seed=0)
    score = det.score_sample("anything", model)
    assert abs(score) < 0.01


def test_perturbation_loss_rejects_invalid_n_perturbations():
    with pytest.raises(ValueError, match="n_perturbations"):
        PerturbationLoss(n_perturbations=0)


def test_perturbation_loss_rejects_invalid_mode():
    with pytest.raises(ValueError, match="mode"):
        PerturbationLoss(mode="invalid")


def test_perturbation_loss_handles_empty_logprobs():
    model = FakeModel("empty", lambda _t: np.array([]))
    det = PerturbationLoss(n_perturbations=3, seed=0)
    assert det.score_sample("anything", model) == 0.0


def test_perturbation_loss_handles_no_change_perturbation():
    """If perturbation_fn is identity, all perturbed texts equal original."""

    def identity(text: str, _rng: random.Random) -> str:
        return text

    def scorer(_text: str) -> np.ndarray:
        return np.array([-1.0, -1.0])

    model = FakeModel("identity", scorer)
    det = PerturbationLoss(perturbation_fn=identity, n_perturbations=5, seed=0)
    assert det.score_sample("anything", model) == 0.0


def test_perturbation_loss_ratio_mode():
    """Ratio mode returns baseline / perturbed_mean."""

    def scorer(text: str) -> np.ndarray:
        # Exact match: original text gets high logprob, any change gets low
        if text == "ORIGINAL sentence here":
            return np.array([-0.5, -0.5])
        return np.array([-2.0, -2.0])

    model = FakeModel("ratio", scorer)
    det = PerturbationLoss(
        perturbation_fn=random_word_drop, n_perturbations=5, seed=0, mode="ratio"
    )
    score = det.score_sample("ORIGINAL sentence here", model)
    # (-0.5) / (-2.0) = 0.25
    assert 0.2 < score < 0.3


def test_perturbation_loss_ratio_mode_guard_division_by_zero():
    """If perturbed_mean is ~0, ratio mode should not crash."""

    def scorer(text: str) -> np.ndarray:
        return np.array([0.0, 0.0]) if "pert" in text else np.array([-1.0, -1.0])

    model = FakeModel("zero", scorer)

    def always_pert(text: str, _rng: random.Random) -> str:
        return text + " pert"

    det = PerturbationLoss(perturbation_fn=always_pert, n_perturbations=1, seed=0, mode="ratio")
    score = det.score_sample("original", model)
    assert score == 0.0


def test_perturbation_loss_seed_determinism():
    def scorer(_text: str) -> np.ndarray:
        return np.array([-1.0, -2.0])

    model = FakeModel("det", scorer)
    det = PerturbationLoss(n_perturbations=3, seed=42)
    s1 = det.score_sample("hello world", model)
    s2 = det.score_sample("hello world", model)
    assert s1 == s2


def test_perturbation_loss_different_seeds_different_scores():
    det1 = PerturbationLoss(n_perturbations=3, seed=1)
    det2 = PerturbationLoss(n_perturbations=3, seed=2)

    def hash_scorer(text: str) -> np.ndarray:
        return np.full(4, float(hash(text) % 100) / 100 - 0.5)

    model2 = FakeModel("hash", hash_scorer)
    s1 = det1.score_sample("the quick brown fox", model2)
    s2 = det2.score_sample("the quick brown fox", model2)
    assert s1 != s2


def test_perturbation_loss_butter_fingers_integration():
    def scorer(_text: str) -> np.ndarray:
        return np.array([-1.0, -2.0])

    model = FakeModel("butter", scorer)
    det = PerturbationLoss(perturbation_fn=butter_fingers, n_perturbations=3, seed=0)
    score = det.score_sample("hello world test sentence", model)
    assert isinstance(score, float)
