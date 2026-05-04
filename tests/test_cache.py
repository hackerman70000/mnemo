from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from mnemo.detectors import (
    CoDeC,
    MaxKProb,
    MinKProb,
    Perplexity,
    VanillaLoss,
    ZlibRatio,
)
from mnemo.utils.cache import compute_logprobs_batch, score_corpus, score_with_cache
from tests.conftest import FakeModel


def _counting_model() -> tuple[FakeModel, Counter[str]]:
    """Returns model whose call counts let tests verify cache hits."""
    counts: Counter[str] = Counter()

    def logprobs(text: str) -> np.ndarray:
        counts[text] += 1
        return np.full(max(1, len(text.split())), -1.5)

    return FakeModel("count", logprobs), counts


def test_compute_logprobs_batch_calls_each_sample_once():
    model, counts = _counting_model()
    samples = [f"sample {i}" for i in range(5)]
    cache = compute_logprobs_batch(model, samples, progress=False)
    assert len(cache) == 5
    assert all(counts[s] == 1 for s in samples)


def test_score_with_cache_matches_score_sample(constant_model):
    samples = [f"sample {i} body text" for i in range(8)]
    cache = [constant_model.token_logprobs(s) for s in samples]

    detector = VanillaLoss()
    cached_scores = score_with_cache(detector, samples, cache)
    direct_scores = [detector.score_sample(s, constant_model) for s in samples]
    assert cached_scores == direct_scores


def test_score_with_cache_rejects_length_mismatch():
    samples = ["one", "two"]
    cache = [np.array([-1.0])]
    with pytest.raises(ValueError, match="same length"):
        score_with_cache(VanillaLoss(), samples, cache)


def test_score_corpus_runs_one_forward_pass_per_sample():
    model, counts = _counting_model()
    samples = [f"sample {i} text" for i in range(10)]
    detectors = [VanillaLoss(), Perplexity(), MinKProb(), MaxKProb(), ZlibRatio()]

    score_corpus(detectors, model, samples, progress=False)

    # 5 cacheable detectors but only one forward pass per sample.
    for s in samples:
        assert counts[s] == 1


def test_score_corpus_handles_context_detector_fallback(memorized_model):
    """CoDeC has requires_context=True; should fall back to score_sample."""
    samples = [f"sample {i} body" for i in range(8)]
    detectors = [VanillaLoss(), CoDeC(skip_first_tokens=0)]

    results = score_corpus(detectors, memorized_model, samples, progress=False)
    assert set(results.keys()) == {"vanilla_loss", "codec"}
    assert len(results["codec"]) == 8
    assert all(s in {0.0, 1.0} for s in results["codec"])


def test_score_corpus_returns_empty_dicts_for_empty_input():
    model, _ = _counting_model()
    detectors = [VanillaLoss(), Perplexity()]
    results = score_corpus(detectors, model, [], progress=False)
    assert results == {"vanilla_loss": [], "perplexity": []}


def test_from_logprobs_consistency_across_cacheable_detectors(constant_model):
    """Each cacheable detector's from_logprobs and score_sample must match."""
    sample = "alpha beta gamma delta epsilon"
    lp = constant_model.token_logprobs(sample)

    for detector in (VanillaLoss(), Perplexity(), MinKProb(), MaxKProb(), ZlibRatio()):
        direct = detector.score_sample(sample, constant_model)
        cached = detector.from_logprobs(lp, sample)
        assert direct == pytest.approx(cached)


def test_codec_from_logprobs_raises():
    """Context detectors must signal incompatibility with caching."""
    detector = CoDeC()
    with pytest.raises(NotImplementedError, match="codec"):
        detector.from_logprobs(np.array([-1.0]), "sample")
