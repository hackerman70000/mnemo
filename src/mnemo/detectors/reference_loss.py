"""Reference-model MIA — paper Maini et al. 2024 §2.1.

A model trained on a sample produces a lower loss on it than a model
that wasn't. We score each candidate by the suspect's edge over a small
reference LM (Phi-1.5, Tinystories, SILO) which was trained on disjoint
data.

The reference is supplied as any `ModelBackend` so we can swap real HF
models for mocks during testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from mnemo.core.detector import Detector
from mnemo.models.base import ModelBackend


@dataclass
class ReferenceLoss(Detector):
    """Score = mean_logprob(suspect) - mean_logprob(reference)  (diff mode)
    or  mean_logprob(suspect) / mean_logprob(reference)  (ratio mode).

    Higher diff ⇒ suspect is much more confident than the reference ⇒ sample
    is more likely a member of the suspect's training set.
    """

    reference_model: ModelBackend
    mode: str = "diff"

    name: ClassVar[str] = "reference_loss"
    requires_context: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if self.mode not in {"diff", "ratio"}:
            raise ValueError(f"mode must be 'diff' or 'ratio', got {self.mode}")

    def score_sample(
        self,
        sample: str,
        model: ModelBackend,
        context: list[str] | None = None,
    ) -> float:
        del context

        suspect_lp = model.token_logprobs(sample)
        ref_lp = self.reference_model.token_logprobs(sample)
        if suspect_lp.size == 0 or ref_lp.size == 0:
            return 0.0

        suspect_mean = float(np.mean(suspect_lp))
        ref_mean = float(np.mean(ref_lp))

        if self.mode == "diff":
            return suspect_mean - ref_mean

        if abs(ref_mean) < 1e-12:
            return 0.0
        return suspect_mean / ref_mean
