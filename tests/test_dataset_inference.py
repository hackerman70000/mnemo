from __future__ import annotations

import numpy as np

from mnemo.detectors import MaxKProb, MinKProb, Perplexity, VanillaLoss
from mnemo.pipelines.dataset_inference import dataset_inference
from tests.conftest import FakeModel


def _make_split_model() -> FakeModel:
    """Return higher logprobs for samples containing 'TRAIN' (memorised),
    lower for samples containing 'VAL'. Tiny per-text jitter avoids the
    catastrophic-cancellation warning from scipy when the t-test sees
    identical inputs.
    """

    def logprobs(text: str) -> np.ndarray:
        n = max(1, len(text.split()))
        base = -0.5 if "TRAIN" in text else -3.0
        jitter = ((hash(text) & 0xFFFF) / 0xFFFF - 0.5) * 0.01
        return np.full(n, base + jitter)

    return FakeModel("split", logprobs)


def test_dataset_inference_detects_membership():
    model = _make_split_model()
    suspect = [f"TRAIN sample with content number {i}" for i in range(120)]
    validation = [f"VAL sample with content number {i}" for i in range(120)]

    result = dataset_inference(
        model,
        suspect,
        validation,
        [VanillaLoss(), Perplexity(), MinKProb(), MaxKProb()],
        n_seeds=3,
        holdout_size=80,
        progress=False,
    )

    assert result.p_combined < 0.05
    assert result.trained is True
    assert set(result.detectors) == {"vanilla_loss", "perplexity", "min_k_prob", "max_k_prob"}
    assert len(result.p_values) == 3


def test_dataset_inference_no_false_positive_when_iid():
    rng = np.random.default_rng(0)

    def logprobs(text: str) -> np.ndarray:
        del text
        return rng.normal(-2.0, 0.1, size=8)

    model = FakeModel("iid", logprobs)
    suspect = [f"sample {i}" for i in range(120)]
    validation = [f"different {i}" for i in range(120)]

    result = dataset_inference(
        model,
        suspect,
        validation,
        [VanillaLoss(), Perplexity(), MinKProb()],
        n_seeds=3,
        holdout_size=80,
        progress=False,
    )

    assert result.p_combined > 0.05
    assert result.trained is False


def test_dataset_inference_returns_per_detector_weights():
    model = _make_split_model()
    suspect = [f"TRAIN {i}" for i in range(60)]
    validation = [f"VAL {i}" for i in range(60)]

    result = dataset_inference(
        model,
        suspect,
        validation,
        [VanillaLoss(), Perplexity()],
        n_seeds=2,
        holdout_size=40,
        progress=False,
    )

    assert set(result.feature_weights.keys()) == {"vanilla_loss", "perplexity"}
    for w in result.feature_weights.values():
        assert isinstance(w, float)
