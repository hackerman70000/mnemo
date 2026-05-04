from __future__ import annotations

import numpy as np
import pytest

from mnemo.detectors.reference_loss import ReferenceLoss
from tests.conftest import FakeModel


def test_reference_loss_positive_when_suspect_more_confident():
    suspect = FakeModel("suspect", lambda text: np.full(max(1, len(text.split())), -1.0))
    reference = FakeModel("reference", lambda text: np.full(max(1, len(text.split())), -3.0))

    detector = ReferenceLoss(reference_model=reference)
    score = detector.score_sample("alpha beta gamma", suspect)
    assert score == pytest.approx(2.0)


def test_reference_loss_negative_when_reference_more_confident():
    suspect = FakeModel("suspect", lambda text: np.full(max(1, len(text.split())), -3.0))
    reference = FakeModel("reference", lambda text: np.full(max(1, len(text.split())), -1.0))

    detector = ReferenceLoss(reference_model=reference)
    score = detector.score_sample("alpha beta gamma", suspect)
    assert score == pytest.approx(-2.0)


def test_reference_loss_zero_when_both_models_agree(constant_model):
    detector = ReferenceLoss(reference_model=constant_model)
    score = detector.score_sample("alpha beta gamma", constant_model)
    assert score == pytest.approx(0.0)


def test_reference_loss_handles_empty_logprobs():
    suspect = FakeModel("suspect", lambda _t: np.array([]))
    reference = FakeModel("reference", lambda _t: np.array([-1.0, -1.0]))
    detector = ReferenceLoss(reference_model=reference)
    assert detector.score_sample("anything", suspect) == 0.0
