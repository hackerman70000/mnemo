from mnemo.core.detector import Detector
from mnemo.detectors.codec import CoDeC
from mnemo.detectors.max_k import MaxKProb
from mnemo.detectors.min_k import MinKProb
from mnemo.detectors.perplexity import Perplexity
from mnemo.detectors.perturbation_loss import (
    PerturbationLoss,
    adjacent_word_swap,
    butter_fingers,
    change_char_case,
    random_word_drop,
    synonym_substitution,
    underscore_trick,
    whitespace_perturbation,
)
from mnemo.detectors.reference_loss import ReferenceLoss
from mnemo.detectors.vanilla_loss import VanillaLoss
from mnemo.detectors.zlib_ratio import ZlibRatio

DETECTOR_REGISTRY: dict[str, type[Detector]] = {
    CoDeC.name: CoDeC,
    VanillaLoss.name: VanillaLoss,
    Perplexity.name: Perplexity,
    MinKProb.name: MinKProb,
    MaxKProb.name: MaxKProb,
    ZlibRatio.name: ZlibRatio,
}
"""Default-constructible detectors exposed to the CLI.

`PerturbationLoss` and `ReferenceLoss` need constructor arguments
(a perturbation callable and a reference `ModelBackend` respectively)
and are intentionally absent from this registry — instantiate them via
the Python API."""

__all__ = [
    "DETECTOR_REGISTRY",
    "CoDeC",
    "MaxKProb",
    "MinKProb",
    "Perplexity",
    "PerturbationLoss",
    "ReferenceLoss",
    "VanillaLoss",
    "ZlibRatio",
    "adjacent_word_swap",
    "butter_fingers",
    "change_char_case",
    "random_word_drop",
    "synonym_substitution",
    "underscore_trick",
    "whitespace_perturbation",
]
