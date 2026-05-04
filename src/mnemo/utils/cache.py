"""Score caching: forward pass once per sample, reuse logprobs across detectors.

The non-context detectors (`VanillaLoss`, `Perplexity`, `MinKProb`,
`MaxKProb`, `ZlibRatio`) all derive from per-token log-probabilities.
Running them naively means N detectors x N forward passes per sample.
This module computes the logprobs once and dispatches them to every
cacheable detector via `Detector.from_logprobs`.

CoDeC and other context-dependent detectors fall back to per-sample
scoring inside `score_corpus` so callers can pass a heterogeneous
detector list without special-casing.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
from loguru import logger
from tqdm.auto import tqdm

from mnemo.core.detector import Detector
from mnemo.models.base import ModelBackend


def compute_logprobs_batch(
    model: ModelBackend,
    samples: Sequence[str],
    *,
    progress: bool = True,
) -> list[np.ndarray]:
    """Forward pass each sample once, return per-sample token log-probabilities."""
    iterator = tqdm(samples, desc="logprobs", disable=not progress)
    return [model.token_logprobs(s) for s in iterator]


def score_with_cache(
    detector: Detector,
    samples: Sequence[str],
    cached_logprobs: Sequence[np.ndarray],
) -> list[float]:
    """Score every sample with `detector` using precomputed logprobs.

    Raises `NotImplementedError` if the detector has no `from_logprobs`
    (e.g. CoDeC). Use `score_corpus` for mixed detector lists.
    """
    if len(samples) != len(cached_logprobs):
        raise ValueError(
            f"samples ({len(samples)}) and cached_logprobs ({len(cached_logprobs)}) "
            "must be the same length"
        )
    return [detector.from_logprobs(lp, s) for lp, s in zip(cached_logprobs, samples, strict=True)]


def score_corpus(
    detectors: Sequence[Detector],
    model: ModelBackend,
    samples: Sequence[str],
    *,
    num_context_examples: int = 1,
    seed: int = 42,
    progress: bool = True,
) -> dict[str, list[float]]:
    """Score one corpus with many detectors as cheaply as possible.

    Strategy:
    - Run a single forward pass per sample → cache.
    - For every detector that supports `from_logprobs`, score from cache.
    - For context-dependent detectors (CoDeC), fall back to `score_sample`
      with random in-context neighbours from the same corpus.
    """
    if len(samples) == 0:
        return {d.name: [] for d in detectors}

    cached = compute_logprobs_batch(model, samples, progress=progress)
    rng = random.Random(seed)
    samples_list = list(samples)

    results: dict[str, list[float]] = {}
    for det in detectors:
        if det.requires_context:
            scores: list[float] = []
            for i, s in enumerate(samples_list):
                others = samples_list[:i] + samples_list[i + 1 :]
                k = min(num_context_examples, len(others))
                ctx = rng.sample(others, k) if k > 0 else []
                scores.append(det.score_sample(s, model, context=ctx))
            results[det.name] = scores
            continue

        try:
            results[det.name] = score_with_cache(det, samples_list, cached)
        except NotImplementedError:
            logger.info(f"{det.name} has no from_logprobs — falling back to score_sample")
            results[det.name] = [det.score_sample(s, model) for s in samples_list]

    return results
