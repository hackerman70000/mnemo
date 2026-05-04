from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from mnemo.core.detector import Detector
from mnemo.models.base import ModelBackend


@dataclass
class MinKProb(Detector):
    """Min-K% Prob (Zhang et al., 2021b / Shi et al., 2024).

    Mean of the bottom K% token log-probabilities. Higher ⇒ even the worst
    tokens are confidently predicted ⇒ stronger memorization signal.
    """

    k_percent: float = 20.0

    name: ClassVar[str] = "min_k_prob"
    requires_context: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if not 0 < self.k_percent <= 100:
            raise ValueError(f"k_percent must be in (0, 100], got {self.k_percent}")

    def score_sample(
        self,
        sample: str,
        model: ModelBackend,
        context: list[str] | None = None,
    ) -> float:
        del context
        return self.from_logprobs(model.token_logprobs(sample), sample)

    def from_logprobs(self, logprobs: np.ndarray, sample: str = "") -> float:
        del sample
        if logprobs.size == 0:
            return 0.0
        k = max(1, int(logprobs.size * self.k_percent / 100))
        bottom_k = np.partition(logprobs, k - 1)[:k]
        return float(np.mean(bottom_k))
