from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from mnemo.core.detector import Detector
from mnemo.models.base import ModelBackend


@dataclass
class CoDeC(Detector):
    """Contamination Detection via Context (Zawalski et al., 2025).

    Per-sample score: 1.0 if adding in-context examples *reduces* model
    confidence on the target (memorization disrupted), else 0.0.
    Aggregated as the fraction of samples flagged.
    """

    skip_first_tokens: int = 10
    separator: str = "\n\n"

    name: ClassVar[str] = "codec"
    requires_context: ClassVar[bool] = True

    def score_sample(
        self,
        sample: str,
        model: ModelBackend,
        context: list[str] | None = None,
    ) -> float:
        baseline = model.token_logprobs(sample)
        if not context or baseline.size == 0:
            return 0.0

        prefix = self.separator.join(context) + self.separator
        full = model.token_logprobs(prefix + sample)

        n_target = baseline.size
        if full.size < n_target:
            return 0.0
        target_with_ctx = full[-n_target:]

        start = min(self.skip_first_tokens, max(0, n_target - 1))
        if start >= n_target:
            return 0.0

        baseline_conf = float(np.mean(baseline[start:]))
        context_conf = float(np.mean(target_with_ctx[start:]))
        return 1.0 if (baseline_conf - context_conf) > 0 else 0.0
