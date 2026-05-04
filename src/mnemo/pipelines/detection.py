from __future__ import annotations

import random
from collections.abc import Sequence

from loguru import logger
from tqdm.auto import tqdm

from mnemo.core.detector import Detector
from mnemo.core.result import DatasetResult
from mnemo.models.base import ModelBackend


def detect_contamination(
    detector: Detector,
    model: ModelBackend,
    dataset: Sequence[str],
    *,
    dataset_name: str = "unknown",
    num_context_examples: int = 1,
    max_samples: int | None = 1000,
    seed: int = 42,
    progress: bool = True,
) -> DatasetResult:
    rng = random.Random(seed)
    samples = list(dataset)
    if max_samples is not None and len(samples) > max_samples:
        samples = rng.sample(samples, max_samples)

    sample_scores: list[float] = []
    use_context = detector.requires_context

    logger.info(
        f"Running {detector.name} on {dataset_name} "
        f"(model={model.name}, n={len(samples)}, ctx={num_context_examples if use_context else 0})"
    )

    iterator = tqdm(samples, desc=detector.name, disable=not progress)
    for i, target in enumerate(iterator):
        ctx: list[str] | None = None
        if use_context:
            others = samples[:i] + samples[i + 1 :]
            k = min(num_context_examples, len(others))
            ctx = rng.sample(others, k) if k > 0 else []
        score = detector.score_sample(target, model, context=ctx)
        sample_scores.append(score)

    return DatasetResult(
        detector=detector.name,
        model=model.name,
        dataset_name=dataset_name,
        sample_scores=sample_scores,
        metadata={
            "num_context_examples": num_context_examples if use_context else 0,
            "seed": seed,
        },
    )
