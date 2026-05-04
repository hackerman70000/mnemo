from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from mnemo.core.detector import Detector
from mnemo.models.base import ModelBackend


@dataclass
class ZlibRatio(Detector):
    """Loss-to-zlib ratio (Carlini et al., 2022), normalized as contamination signal.

    Memorized text has both low loss and low zlib entropy. We return
    `-loss / zlib_size` so higher = more likely contaminated, matching the
    convention used by other detectors in this package.
    """

    name: ClassVar[str] = "zlib_ratio"
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
        if logprobs.size == 0:
            return 0.0
        loss = -float(np.mean(logprobs))
        encoded = sample.encode("utf-8")
        if not encoded:
            return 0.0
        zlib_size = len(zlib.compress(encoded))
        if zlib_size == 0:
            return 0.0
        return -loss / zlib_size
