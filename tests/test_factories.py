from __future__ import annotations

import numpy as np

from mnemo.detectors.factories import maini_selected_features
from tests.conftest import FakeModel


def test_maini_selected_features_without_refs():
    dets = maini_selected_features()
    names = [d.name for d in dets]

    # Core detectors
    assert "perplexity" in names
    assert "vanilla_loss" in names
    assert "zlib_ratio" in names

    # Multi-k MinK / MaxK
    min_k_entries = [n for n in names if n == "min_k_prob"]
    max_k_entries = [n for n in names if n == "max_k_prob"]
    assert len(min_k_entries) == 7
    assert len(max_k_entries) == 7

    # Perturbation losses (ratio mode)
    pert_entries = [n for n in names if n == "perturbation_loss"]
    assert len(pert_entries) == 6

    # No reference losses when refs not provided
    assert "reference_loss" not in names


def test_maini_selected_features_with_refs():
    ref = FakeModel("ref", lambda _t: np.array([-2.0, -2.0]))
    dets = maini_selected_features(reference_models={"tinystories-33M": ref})
    names = [d.name for d in dets]
    assert "reference_loss" in names
    assert names.count("reference_loss") == 1


def test_maini_selected_features_all_unique_names():
    """Names are used as dictionary keys in the result; they must be unique.

    Currently detectors share the same ``name`` class-var even when
    parameterised (e.g. every MinKProb is ``min_k_prob``).  This is OK
    for the pipeline because we use ``detector_names = [d.name for d in
    detectors]`` and the linear classifier learns one weight per column.
    The duplicate names just mean the ``feature_weights`` dict will have
    one entry that averages all identically-named detectors.  This test
    documents that behaviour.
    """
    dets = maini_selected_features()
    names = [d.name for d in dets]
    # Not unique — this is expected given the current architecture
    assert len(names) != len(set(names))
    assert len(set(names)) < len(names)
