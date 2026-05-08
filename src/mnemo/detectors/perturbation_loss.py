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
import string
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from mnemo.core.detector import Detector
from mnemo.models.base import ModelBackend

PerturbationFn = Callable[[str, random.Random], str]


# ---------------------------------------------------------------------------
# Lightweight perturbation functions (CPU-only, no heavy deps)
# ---------------------------------------------------------------------------


def adjacent_word_swap(text: str, rng: random.Random, *, n_swaps: int = 5) -> str:
    """Swap pairs of adjacent words `n_swaps` times."""
    words = text.split()
    if len(words) < 2:
        return text
    for _ in range(n_swaps):
        i = rng.randrange(len(words) - 1)
        words[i], words[i + 1] = words[i + 1], words[i]
    return " ".join(words)


def random_word_drop(text: str, rng: random.Random, *, drop_fraction: float = 0.15) -> str:
    """Drop a fraction of words at random."""
    if not 0 <= drop_fraction < 1:
        raise ValueError(f"drop_fraction must be in [0, 1), got {drop_fraction}")
    words = text.split()
    if not words:
        return text
    n_drop = max(1, int(len(words) * drop_fraction))
    keep_idx = sorted(rng.sample(range(len(words)), max(1, len(words) - n_drop)))
    return " ".join(words[i] for i in keep_idx)


# Keyboard-neighbour map for butter-fingers style typos
_BUTTER_FINGERS = {
    "a": "asqwzx",
    "b": "bvghnj",
    "c": "cxdfgv",
    "d": "dsfercxz",
    "e": "erdsfwq",
    "f": "fdgrtvcx",
    "g": "gfhtybvc",
    "h": "hgjuytgb",
    "i": "ikjuhylo",
    "j": "jhkuiyhn",
    "k": "kjilmjuhb",
    "l": "lkopji",
    "m": "mnjkl",
    "n": "nbhjm",
    "o": "oplikujp",
    "p": "po;loki",
    "q": "qwaesz",
    "r": "retdfg",
    "s": "swadxz",
    "t": "tyrfgh",
    "u": "uyjhki",
    "v": "vcfgb",
    "w": "wqasde",
    "x": "xzcsd",
    "y": "yuhjgt",
    "z": "zasx",
}


def butter_fingers(text: str, rng: random.Random, *, prob: float = 0.1) -> str:
    """Inject keyboard-neighbour typos (Maini §2.1 perturbation 3)."""
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch.lower() in _BUTTER_FINGERS and rng.random() < prob:
            neighbours = _BUTTER_FINGERS[ch.lower()]
            replacement = rng.choice(neighbours)
            chars[i] = replacement if ch.islower() else replacement.upper()
    return "".join(chars)


_SYNONYMS = {
    "good": ["nice", "fine", "great"],
    "bad": ["poor", "awful", "terrible"],
    "big": ["large", "huge", "vast"],
    "small": ["tiny", "little", "mini"],
    "happy": ["glad", "joyful", "cheerful"],
    "sad": ["unhappy", "sorrowful", "gloomy"],
    "fast": ["quick", "rapid", "swift"],
    "slow": ["sluggish", "gradual", "leisurely"],
    "important": ["significant", "crucial", "vital"],
    "use": ["utilize", "employ", "apply"],
    "make": ["create", "produce", "build"],
    "get": ["obtain", "acquire", "receive"],
    "say": ["state", "mention", "declare"],
    "think": ["believe", "consider", "suppose"],
    "know": ["understand", "recognize", "realize"],
    "see": ["observe", "notice", "spot"],
    "want": ["desire", "wish", "need"],
    "come": ["arrive", "appear", "reach"],
    "go": ["leave", "depart", "travel"],
    "find": ["discover", "locate", "detect"],
    "give": ["provide", "offer", "supply"],
    "take": ["seize", "grab", "capture"],
    "have": ["possess", "own", "hold"],
    "do": ["perform", "execute", "accomplish"],
    "work": ["function", "operate", "perform"],
    "new": ["novel", "fresh", "recent"],
    "old": ["ancient", "aged", "vintage"],
    "first": ["initial", "primary", "opening"],
    "last": ["final", "ultimate", "closing"],
    "high": ["tall", "elevated", "lofty"],
    "low": ["short", "reduced", "minimal"],
    "long": ["lengthy", "extended", "prolonged"],
    "right": ["correct", "accurate", "proper"],
    "wrong": ["incorrect", "erroneous", "false"],
    "same": ["identical", "equal", "equivalent"],
    "different": ["distinct", "unlike", "diverse"],
    "early": ["premature", "initial", "advanced"],
    "late": ["tardy", "delayed", "overdue"],
    "young": ["youthful", "juvenile", "immature"],
    "easy": ["simple", "effortless", "straightforward"],
    "hard": ["difficult", "challenging", "tough"],
    "best": ["finest", "greatest", "optimal"],
    "worst": ["poorest", "weakest", "gravest"],
    "better": ["superior", "improved", "enhanced"],
    "worse": ["inferior", "worsened", "deteriorated"],
    "more": ["additional", "extra", "further"],
    "most": ["majority", "bulk", "maximum"],
    "many": ["numerous","countless","abundant"],
    "few": ["scarce", "sparse", "limited"],
    "much": ["plenty", "abundance", "profusion"],
    "little": ["slight", "minimal", "negligible"],
    "very": ["extremely", "exceedingly", "intensely"],
    "really": ["truly", "genuinely", "actually"],
    "just": ["merely", "only", "simply"],
    "still": ["yet", "nevertheless", "nonetheless"],
    "also": ["too", "likewise", "furthermore"],
    "well": ["fine", "satisfactorily", "adequately"],
    "back": ["return", "revert", "retreat"],
    "only": ["solely", "exclusively", "merely"],
    "other": ["alternative", "different", "another"],
    "another": ["additional", "extra", "further"],
    "each": ["every", "all", " respective"],
    "both": ["two", "pair", "couple"],
    "neither": ["none", "not one", "no one"],
    "either": ["any", "one", "whichever"],
    "all": ["every", "entire", "whole"],
    "none": ["nothing", "zero", "nil"],
    "one": ["single", "individual", "sole"],
    "two": ["pair", "couple", "duo"],
    "three": ["trio", "triad", "triple"],
    "four": ["quartet", "quadruple", "foursome"],
    "five": ["quintet", "quintuple", "fivesome"],
    "six": ["sextet", "sextuple", "sixsome"],
    "seven": ["septet", "septuple", "sevensome"],
    "eight": ["octet", "octuple", "eightsome"],
    "nine": ["nonet", "nonuple", "ninesome"],
    "ten": ["decade", "decet", "tensome"],
}


def synonym_substitution(text: str, rng: random.Random, *, prob: float = 0.2) -> str:
    """Replace common words with synonyms (Maini §2.1 perturbation 2)."""
    words = text.split()
    new_words = []
    for w in words:
        lower = w.lower().strip(string.punctuation)
        if lower in _SYNONYMS and rng.random() < prob:
            syn = rng.choice(_SYNONYMS[lower])
            # Preserve basic casing / punctuation
            if w[0].isupper():
                syn = syn.capitalize()
            if w[-1] in string.punctuation:
                syn += w[-1]
            new_words.append(syn)
        else:
            new_words.append(w)
    return " ".join(new_words)


def change_char_case(text: str, rng: random.Random, *, prob: float = 0.25) -> str:
    """Randomly flip the case of characters (Maini §2.1 perturbation 5)."""
    chars = []
    for ch in text:
        if ch.isalpha() and rng.random() < prob:
            chars.append(ch.lower() if ch.isupper() else ch.upper())
        else:
            chars.append(ch)
    return "".join(chars)


def whitespace_perturbation(text: str, rng: random.Random, *, prob: float = 0.25) -> str:
    """Add or remove whitespace around words (Maini §2.1 perturbation 1)."""
    words = text.split()
    if not words:
        return text
    out = []
    for w in words:
        out.append(w)
        # Randomly add extra spaces after some words
        if rng.random() < prob:
            out.append(" " * rng.randint(2, 4))
        else:
            out.append(" ")
    result = "".join(out).strip()
    # Occasionally remove spaces between two words
    if " " in result and rng.random() < prob * 0.5:
        parts = result.split()
        if len(parts) > 1:
            idx = rng.randrange(len(parts) - 1)
            parts[idx] = parts[idx] + parts[idx + 1]
            del parts[idx + 1]
            result = " ".join(parts)
    return result


def underscore_trick(text: str, rng: random.Random, *, prob: float = 0.25) -> str:
    """Replace spaces with underscores (Maini §2.1 perturbation 6)."""
    if rng.random() < prob:
        return text.replace(" ", "_")
    return text


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


@dataclass
class PerturbationLoss(Detector):
    """Average log-prob gap or ratio between original and perturbed versions.

    * ``mode="diff"``  →  ``mean(logprob(sample)) - mean(logprob(perturbed))``
    * ``mode="ratio"`` →  ``mean(logprob(sample)) / mean(logprob(perturbed))``

    Higher diff ⇒ perturbations reduced confidence ⇒ original sat near a
    local maximum ⇒ memorised.  For ``ratio`` the sign is learned by the
    downstream linear classifier (Maini et al. 2024).
    """

    perturbation_fn: PerturbationFn = field(default=adjacent_word_swap)
    n_perturbations: int = 20
    seed: int = 0
    mode: str = "diff"

    name: ClassVar[str] = "perturbation_loss"
    requires_context: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if self.n_perturbations <= 0:
            raise ValueError(f"n_perturbations must be positive, got {self.n_perturbations}")
        if self.mode not in {"diff", "ratio"}:
            raise ValueError(f"mode must be 'diff' or 'ratio', got {self.mode}")

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

        perturbed_mean = float(np.mean(perturbed_means))

        if self.mode == "diff":
            return baseline - perturbed_mean

        # ratio mode — guard against division by zero
        if abs(perturbed_mean) < 1e-12:
            return 0.0
        return baseline / perturbed_mean

    def from_logprobs(self, logprobs: np.ndarray, sample: str = "") -> float:
        del logprobs, sample
        raise NotImplementedError(f"{self.name} does not support cached logprob scoring")
