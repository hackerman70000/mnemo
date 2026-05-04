from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from mnemo.core.detector import Detector
from mnemo.models.base import ModelBackend


@dataclass
class VanillaLoss(Detector):
    """Average per-token log-probability of the sample (Fu et al., 2024).

    Higher value (less negative) ⇒ lower loss ⇒ more likely memorized.
    """

    name: ClassVar[str] = "vanilla_loss"
    requires_context: ClassVar[bool] = False

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
        return float(np.mean(logprobs))
