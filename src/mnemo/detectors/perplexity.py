from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from mnemo.core.detector import Detector
from mnemo.models.base import ModelBackend


@dataclass
class Perplexity(Detector):
    """Negative perplexity of the sample — `-exp(-mean(logprob))`.

    Perplexity is a monotonic transform of vanilla loss but presents a
    differently scaled feature for the Dataset Inference linear regressor
    (Maini et al. 2024). Returned as `-PPL` so that higher = lower
    perplexity = more likely contaminated, matching package conventions.
    """

    name: ClassVar[str] = "perplexity"
    requires_context: ClassVar[bool] = False

    def score_sample(
        self,
        sample: str,
        model: ModelBackend,
        context: list[str] | None = None,
    ) -> float:
        logprobs = model.token_logprobs(sample)
        if logprobs.size == 0:
            return 0.0
        return float(-np.exp(-np.mean(logprobs)))
