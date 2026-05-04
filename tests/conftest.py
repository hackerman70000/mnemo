from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest


class FakeModel:
    """Deterministic model stub driven by a logprob function. No torch dependency."""

    def __init__(self, name: str, logprob_fn: Callable[[str], np.ndarray]) -> None:
        self.name = name
        self._logprob_fn = logprob_fn

    def token_logprobs(self, text: str) -> np.ndarray:
        return self._logprob_fn(text)


@pytest.fixture
def constant_model() -> FakeModel:
    return FakeModel(
        "const",
        lambda text: np.full(max(1, len(text.split())), -2.0),
    )


@pytest.fixture
def memorized_model() -> FakeModel:
    """Confidence drops when the in-context marker is present (mimics paper's contaminated case)."""

    def logprobs(text: str) -> np.ndarray:
        n = max(1, len(text.split()))
        value = -3.0 if "[CTX]" in text else -1.0
        return np.full(n, value)

    return FakeModel("memorized", logprobs)


@pytest.fixture
def unseen_model() -> FakeModel:
    """Confidence improves with context (mimics paper's unseen case)."""

    def logprobs(text: str) -> np.ndarray:
        n = max(1, len(text.split()))
        value = -1.0 if "[CTX]" in text else -3.0
        return np.full(n, value)

    return FakeModel("unseen", logprobs)
