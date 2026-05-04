from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from mnemo.models.base import ModelBackend


class Detector(ABC):
    name: ClassVar[str]
    requires_context: ClassVar[bool] = False

    @abstractmethod
    def score_sample(
        self,
        sample: str,
        model: ModelBackend,
        context: list[str] | None = None,
    ) -> float:
        """Per-sample contamination score. Higher = more likely contaminated."""

    def from_logprobs(self, logprobs: np.ndarray, sample: str = "") -> float:
        """Score from precomputed token log-probabilities (skips forward pass).

        Override in detectors that don't need context. Lets `mnemo.utils.cache`
        run a single forward pass per sample and reuse the logprobs across many
        detectors. Detectors that need context (e.g. CoDeC) leave the default
        which raises.
        """
        del logprobs, sample
        raise NotImplementedError(f"{self.name} does not support cached logprob scoring")
