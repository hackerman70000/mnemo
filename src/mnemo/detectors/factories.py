"""Factory helpers for building the full Maini-style detector suite.

The paper aggregates ~28-52 MIA features (§5.3). Rather than bloating
`DETECTOR_REGISTRY` with dozens of near-identical entries, these helpers
let users compose the full feature vector programmatically.
"""

from __future__ import annotations

import random

from mnemo.detectors.max_k import MaxKProb
from mnemo.detectors.min_k import MinKProb
from mnemo.detectors.perplexity import Perplexity
from mnemo.detectors.perturbation_loss import (
    PerturbationLoss,
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
from mnemo.models.base import ModelBackend

# K-values used in the paper's selected_features.py
_K_VALUES = [5, 10, 20, 30, 40, 50, 60]


def _random_deletion(text: str, rng: random.Random) -> str:
    return random_word_drop(text, rng, drop_fraction=0.25)


# Perturbation functions used in the paper
_PERTURBATION_FNS: dict[str, object] = {
    "synonym_substitution": synonym_substitution,
    "butter_fingers": butter_fingers,
    "random_deletion": _random_deletion,
    "change_char_case": change_char_case,
    "whitespace_perturbation": whitespace_perturbation,
    "underscore_trick": underscore_trick,
}


def maini_selected_features(
    reference_models: dict[str, ModelBackend] | None = None,
) -> list[object]:
    """Return the 28-feature detector list from the paper's ``selected_features.py``.

    The list contains:
    * perplexity, vanilla_loss, zlib_ratio
    * Min-k% Prob  (k ∈ {5,10,20,30,40,50,60})
    * Max-k% Prob  (same k values)
    * Perturbation ratio  (6 perturbation types)
    * Reference-model ratio (4 reference models)

    Args:
        reference_models: mapping ``{"silo": model, "tinystories-33M": model, ...}``.
            If omitted, reference-model detectors are skipped.
    """
    detectors: list[object] = [
        Perplexity(),
        VanillaLoss(),
        ZlibRatio(),
    ]

    for k in _K_VALUES:
        detectors.append(MinKProb(k_percent=float(k)))
        detectors.append(MaxKProb(k_percent=float(k)))

    for _name, fn in _PERTURBATION_FNS.items():
        detectors.append(
            PerturbationLoss(
                perturbation_fn=fn,  # type: ignore[arg-type]
                n_perturbations=20,
                mode="ratio",
                seed=0,
            )
        )

    if reference_models:
        for _name, ref_model in reference_models.items():
            detectors.append(
                ReferenceLoss(reference_model=ref_model, mode="ratio")
            )

    return detectors


def maini_full_features(
    reference_models: dict[str, ModelBackend] | None = None,
) -> list[object]:
    """Return the expanded ~36-feature suite (ratio + diff for perturbations & refs).

    This is a superset of `maini_selected_features` that includes both
    ``ppl_ratio`` and ``ppl_diff`` variants for perturbations and
    reference models, matching the full feature matrix in the reference
    implementation's ``metrics.py``.
    """
    detectors: list[object] = [
        Perplexity(),
        VanillaLoss(),
        ZlibRatio(),
    ]

    for k in _K_VALUES:
        detectors.append(MinKProb(k_percent=float(k)))
        detectors.append(MaxKProb(k_percent=float(k)))

    for _name, fn in _PERTURBATION_FNS.items():
        detectors.append(
            PerturbationLoss(
                perturbation_fn=fn,  # type: ignore[arg-type]
                n_perturbations=20,
                mode="diff",
                seed=0,
            )
        )
        detectors.append(
            PerturbationLoss(
                perturbation_fn=fn,  # type: ignore[arg-type]
                n_perturbations=20,
                mode="ratio",
                seed=0,
            )
        )

    if reference_models:
        for _name, ref_model in reference_models.items():
            detectors.append(ReferenceLoss(reference_model=ref_model, mode="diff"))
            detectors.append(ReferenceLoss(reference_model=ref_model, mode="ratio"))

    return detectors
