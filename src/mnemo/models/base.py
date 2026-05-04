from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ModelBackend(Protocol):
    """Minimal interface a contamination detector requires from a model."""

    name: str

    def token_logprobs(self, text: str) -> np.ndarray:
        """Return log-probabilities for each token in `text`.

        Shape: (n_tokens - 1,) — the log-prob of token i+1 given tokens [0..i].
        """
        ...
