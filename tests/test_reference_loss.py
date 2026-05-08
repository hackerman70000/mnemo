from __future__ import annotations

import numpy as np
import pytest

from mnemo.detectors.reference_loss import ReferenceLoss
from tests.conftest import FakeModel


def test_reference_loss_positive_when_suspect_more_confident():
    suspect = FakeModel("suspect", lambda _t: np.array([-0.5, -0.5]))
    reference = FakeModel("ref", lambda _t: np.array([-3.0, -3.0]))
    det = ReferenceLoss(reference_model=reference)
    score = det.score_sample("foo", suspect)
    assert score == pytest.approx(2.5)


def test_reference_loss_negative_when_reference_more_confident():
    suspect = FakeModel("suspect", lambda _t: np.array([-3.0, -3.0]))
    reference = FakeModel("ref", lambda _t: np.array([-0.5, -0.5]))
    det = ReferenceLoss(reference_model=reference)
    score = det.score_sample("foo", suspect)
    assert score == pytest.approx(-2.5)


def test_reference_loss_zero_when_both_models_agree():
    model = FakeModel("same", lambda _t: np.array([-1.0, -2.0]))
    det = ReferenceLoss(reference_model=model)
    score = det.score_sample("foo", model)
    assert score == pytest.approx(0.0)


def test_reference_loss_handles_empty_logprobs():
    empty = FakeModel("empty", lambda _t: np.array([]))
    det = ReferenceLoss(reference_model=empty)
    score = det.score_sample("foo", empty)
    assert score == 0.0


def test_reference_loss_ratio_mode():
    suspect = FakeModel("suspect", lambda _t: np.array([-0.5, -0.5]))
    reference = FakeModel("ref", lambda _t: np.array([-2.0, -2.0]))
    det = ReferenceLoss(reference_model=reference, mode="ratio")
    score = det.score_sample("foo", suspect)
    # (-0.5) / (-2.0) = 0.25
    assert score == pytest.approx(0.25)


def test_reference_loss_ratio_mode_guard_division_by_zero():
    suspect = FakeModel("suspect", lambda _t: np.array([-1.0, -1.0]))
    zero = FakeModel("zero", lambda _t: np.array([0.0, 0.0]))
    det = ReferenceLoss(reference_model=zero, mode="ratio")
    score = det.score_sample("foo", suspect)
    assert score == 0.0


def test_reference_loss_rejects_invalid_mode():
    with pytest.raises(ValueError, match="mode"):
        ReferenceLoss(reference_model=FakeModel("x", lambda _t: np.array([-1.0])), mode="invalid")
