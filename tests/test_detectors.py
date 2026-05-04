from __future__ import annotations

import numpy as np
import pytest

from mnemo.detectors import CoDeC, MinKProb, VanillaLoss, ZlibRatio
from tests.conftest import FakeModel


def test_vanilla_loss_returns_mean_logprob(constant_model):
    score = VanillaLoss().score_sample("alpha beta gamma delta", constant_model)
    assert score == pytest.approx(-2.0)


def test_min_k_picks_lowest_tokens():
    model = FakeModel("var", lambda _t: np.array([-0.1, -0.5, -2.0, -3.0, -5.0]))
    score = MinKProb(k_percent=40).score_sample("a b c d e", model)
    assert score == pytest.approx(np.mean([-5.0, -3.0]))


def test_min_k_rejects_invalid_percent():
    with pytest.raises(ValueError, match="k_percent"):
        MinKProb(k_percent=0)
    with pytest.raises(ValueError, match="k_percent"):
        MinKProb(k_percent=150)


def test_zlib_returns_negative_signal_for_real_text(constant_model):
    score = ZlibRatio().score_sample("repeat repeat repeat repeat", constant_model)
    assert score < 0


def test_zlib_handles_empty_logprobs():
    model = FakeModel("empty", lambda _t: np.array([]))
    assert ZlibRatio().score_sample("anything", model) == 0.0


def test_codec_no_context_is_zero(constant_model):
    det = CoDeC(skip_first_tokens=0)
    assert det.score_sample("a b c d e f g", constant_model, context=None) == 0.0
    assert det.score_sample("a b c d e f g", constant_model, context=[]) == 0.0


def test_codec_flags_when_context_lowers_confidence(memorized_model):
    det = CoDeC(skip_first_tokens=0)
    score = det.score_sample("a b c d e f g h", memorized_model, context=["[CTX]"])
    assert score == 1.0


def test_codec_clears_when_context_raises_confidence(unseen_model):
    det = CoDeC(skip_first_tokens=0)
    score = det.score_sample("a b c d e f g h", unseen_model, context=["[CTX]"])
    assert score == 0.0


def test_codec_skip_tokens_does_not_explode_on_short_input(constant_model):
    det = CoDeC(skip_first_tokens=10)
    assert det.score_sample("short", constant_model, context=["one"]) == 0.0
