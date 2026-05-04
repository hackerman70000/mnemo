"""DetectGPT-style perturbation detector (Mitchell et al. 2023).

Memorised text sits near a local maximum of the model's log-probability.
Perturbing the sample (e.g. mask + infill with T5, synonym substitution,
adjacent word swaps) should drop the average log-probability for
memorised inputs but barely change it for unseen text.

The detector is decoupled from any specific perturbation strategy: it
takes a `perturbation_fn(text) -> text` callable. Use the helpers in
`mnemo.detectors.perturbation_loss` for cheap stochastic perturbations,
or wire your own T5 mask-fill via a closure for the real paper setup.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from mnemo.core.detector import Detector
from mnemo.models.base import ModelBackend

PerturbationFn = Callable[[str, random.Random], str]


def adjacent_word_swap(text: str, rng: random.Random, *, n_swaps: int = 5) -> str:
    """Swap pairs of adjacent words `n_swaps` times — cheap CPU-only perturbation.

    Weaker than the T5 mask-fill from the original DetectGPT paper, but
    sufficient as a sanity-check perturbation when tooling is constrained.
    """
    words = text.split()
    if len(words) < 2:
        return text
    for _ in range(n_swaps):
        i = rng.randrange(len(words) - 1)
        words[i], words[i + 1] = words[i + 1], words[i]
    return " ".join(words)


def random_word_drop(text: str, rng: random.Random, *, drop_fraction: float = 0.15) -> str:
    """Drop a fraction of words at random — another CPU-friendly perturbation."""
    if not 0 <= drop_fraction < 1:
        raise ValueError(f"drop_fraction must be in [0, 1), got {drop_fraction}")
    words = text.split()
    if not words:
        return text
    n_drop = max(1, int(len(words) * drop_fraction))
    keep_idx = sorted(rng.sample(range(len(words)), max(1, len(words) - n_drop)))
    return " ".join(words[i] for i in keep_idx)


@dataclass
class PerturbationLoss(Detector):
    """Average log-prob gap between original and perturbed versions of `sample`.

    Score = mean(logprob(sample)) - mean(mean(logprob(perturbed_i))).
    Higher ⇒ perturbations reduced confidence ⇒ original sat near a
    local maximum ⇒ memorised.
    """

    perturbation_fn: PerturbationFn = field(default=adjacent_word_swap)
    n_perturbations: int = 20
    seed: int = 0

    name: ClassVar[str] = "perturbation_loss"
    requires_context: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if self.n_perturbations <= 0:
            raise ValueError(f"n_perturbations must be positive, got {self.n_perturbations}")

    def score_sample(
        self,
        sample: str,
        model: ModelBackend,
        context: list[str] | None = None,
    ) -> float:
        del context

        baseline_logprobs = model.token_logprobs(sample)
        if baseline_logprobs.size == 0:
            return 0.0
        baseline = float(np.mean(baseline_logprobs))

        rng = random.Random(self.seed + hash(sample))
        perturbed_means: list[float] = []
        for _ in range(self.n_perturbations):
            perturbed = self.perturbation_fn(sample, rng)
            if perturbed == sample:
                continue
            lp = model.token_logprobs(perturbed)
            if lp.size > 0:
                perturbed_means.append(float(np.mean(lp)))

        if not perturbed_means:
            return 0.0
        return baseline - float(np.mean(perturbed_means))
