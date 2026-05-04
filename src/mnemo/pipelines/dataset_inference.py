"""Dataset Inference (Maini et al., NeurIPS 2024).

Aggregate per-sample MIA features through a linear regressor, then run a
one-sided t-test to decide whether a `suspect` set was used to train a
language model. The arbitr is statistical: low p-value = evidence the
model saw the suspect data; high p-value = no evidence (no false
positives by construction when both groups are IID held-out).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from loguru import logger
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm

from mnemo.core.detector import Detector
from mnemo.core.statistics import (
    one_sided_t_test_lower,
    remove_outliers_to_mean,
    sidak_combine,
)
from mnemo.models.base import ModelBackend


@dataclass
class DatasetInferenceResult:
    model: str
    detectors: list[str]
    suspect_size: int
    validation_size: int
    p_values: list[float]
    p_combined: float
    feature_weights: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def trained(self) -> bool:
        return self.p_combined < self.metadata.get("threshold", 0.1)  # type: ignore[operator]

    def summary(self) -> dict[str, object]:
        return {
            "model": self.model,
            "p_combined": self.p_combined,
            "p_values": self.p_values,
            "trained": self.trained,
            "feature_weights": self.feature_weights,
            "suspect_size": self.suspect_size,
            "validation_size": self.validation_size,
        }


def _score_corpus(
    detectors: Sequence[Detector],
    model: ModelBackend,
    corpus: Sequence[str],
    *,
    desc: str,
    rng: random.Random,
    num_context_examples: int = 1,
    progress: bool,
) -> np.ndarray:
    samples = list(corpus)
    n_samples = len(samples)
    features = np.zeros((n_samples, len(detectors)), dtype=float)

    iterator = tqdm(range(n_samples), desc=desc, disable=not progress)
    for i in iterator:
        target = samples[i]
        for j, det in enumerate(detectors):
            ctx: list[str] | None = None
            if det.requires_context:
                others = samples[:i] + samples[i + 1 :]
                k = min(num_context_examples, len(others))
                ctx = rng.sample(others, k) if k > 0 else []
            features[i, j] = det.score_sample(target, model, context=ctx)
    return features


def _normalize_columns(arr: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    """Z-score normalisation; if `ref` is provided, use its per-column stats."""
    base = ref if ref is not None else arr
    mu = base.mean(axis=0, keepdims=True)
    sigma = base.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return np.asarray((arr - mu) / sigma, dtype=float)


def _clean_features(arr: np.ndarray, fraction: float = 0.025) -> np.ndarray:
    return np.column_stack(
        [remove_outliers_to_mean(arr[:, j], fraction=fraction) for j in range(arr.shape[1])]
    )


def dataset_inference(
    model: ModelBackend,
    suspect: Sequence[str],
    validation: Sequence[str],
    detectors: Sequence[Detector],
    *,
    n_seeds: int = 10,
    holdout_size: int = 1000,
    train_fraction: float = 0.5,
    outlier_fraction: float = 0.025,
    num_context_examples: int = 1,
    threshold: float = 0.1,
    progress: bool = True,
) -> DatasetInferenceResult:
    """Run Maini-style dataset inference.

    Args:
        model: language model under audit.
        suspect: candidate text sequences claimed to be in the training set.
        validation: held-out IID sequences from the same distribution.
        detectors: feature extractors (any subset of the registry).
        n_seeds: number of A/B splits — each yields one p-value, combined via Šidák.
        holdout_size: cap on samples used (split equally between regressor train and t-test).
        train_fraction: portion of samples allocated to regressor training; rest goes to t-test.
        outlier_fraction: top/bottom fraction of feature values clipped to mean.
        num_context_examples: used only by context-dependent detectors (e.g. CoDeC).
        threshold: p-value threshold for the `trained` verdict.
        progress: show tqdm bars.
    """
    if len(detectors) == 0:
        raise ValueError("At least one detector required.")
    if len(suspect) < 4 or len(validation) < 4:
        raise ValueError("Need at least 4 samples per side for stable A/B splits.")

    suspect = list(suspect)
    validation = list(validation)
    cap = min(holdout_size, len(suspect), len(validation))
    detector_names = [d.name for d in detectors]

    logger.info(
        f"Dataset Inference: model={model.name} detectors={detector_names} "
        f"|suspect|={len(suspect)} |val|={len(validation)} cap={cap} seeds={n_seeds}"
    )

    p_values: list[float] = []
    weight_accumulator = np.zeros(len(detectors), dtype=float)

    for seed in range(n_seeds):
        rng = random.Random(seed)
        sus_pool = rng.sample(suspect, cap)
        val_pool = rng.sample(validation, cap)

        n_train = int(cap * train_fraction)
        sus_train, sus_test = sus_pool[:n_train], sus_pool[n_train:]
        val_train, val_test = val_pool[:n_train], val_pool[n_train:]

        feats_sus_train = _score_corpus(
            detectors,
            model,
            sus_train,
            desc=f"seed {seed} sus-train",
            rng=rng,
            num_context_examples=num_context_examples,
            progress=progress,
        )
        feats_val_train = _score_corpus(
            detectors,
            model,
            val_train,
            desc=f"seed {seed} val-train",
            rng=rng,
            num_context_examples=num_context_examples,
            progress=progress,
        )

        cleaned_sus = _clean_features(feats_sus_train, fraction=outlier_fraction)
        cleaned_val = _clean_features(feats_val_train, fraction=outlier_fraction)
        train_features_raw = np.vstack([cleaned_sus, cleaned_val])
        train_features = _normalize_columns(train_features_raw)
        train_labels = np.concatenate(
            [
                np.zeros(len(cleaned_sus)),
                np.ones(len(cleaned_val)),
            ]
        )

        regressor = LinearRegression().fit(train_features, train_labels)

        feats_sus_test = _score_corpus(
            detectors,
            model,
            sus_test,
            desc=f"seed {seed} sus-test",
            rng=rng,
            num_context_examples=num_context_examples,
            progress=progress,
        )
        feats_val_test = _score_corpus(
            detectors,
            model,
            val_test,
            desc=f"seed {seed} val-test",
            rng=rng,
            num_context_examples=num_context_examples,
            progress=progress,
        )

        test_features = _normalize_columns(
            np.vstack([feats_sus_test, feats_val_test]),
            ref=train_features_raw,
        )
        scores = regressor.predict(test_features)
        sus_scores = scores[: len(feats_sus_test)]
        val_scores = scores[len(feats_sus_test) :]

        p = one_sided_t_test_lower(sus_scores, val_scores)
        p_values.append(p)
        weight_accumulator += regressor.coef_
        logger.debug(f"seed={seed} p={p:.4g}")

    p_combined = sidak_combine(p_values)
    weights = {
        name: float(w / n_seeds) for name, w in zip(detector_names, weight_accumulator, strict=True)
    }
    logger.info(f"Dataset Inference complete: p_combined={p_combined:.4g}")

    return DatasetInferenceResult(
        model=model.name,
        detectors=list(detector_names),
        suspect_size=len(suspect),
        validation_size=len(validation),
        p_values=p_values,
        p_combined=p_combined,
        feature_weights=weights,
        metadata={
            "threshold": threshold,
            "n_seeds": n_seeds,
            "holdout_size": cap,
            "train_fraction": train_fraction,
            "outlier_fraction": outlier_fraction,
        },
    )
