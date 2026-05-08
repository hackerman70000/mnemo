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
import torch
from loguru import logger
from torch import nn
from tqdm.auto import tqdm

from mnemo.core.detector import Detector
from mnemo.core.statistics import (
    one_sided_t_test_lower,
    remove_outliers_to_mean,
    sidak_combine,
)
from mnemo.models.base import ModelBackend

_P_SAMPLE_LIST = [2, 5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


@dataclass
class DatasetInferenceResult:
    model: str
    detectors: list[str]
    suspect_size: int
    validation_size: int
    p_values: list[float]
    p_combined: float
    p_value_curves: list[list[float]] = field(default_factory=list)
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
            "p_value_curves": self.p_value_curves,
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


def _normalize_columns(
    arr: np.ndarray,
    ref: np.ndarray | None = None,
    *,
    suspect_only: bool = False,
    n_suspect: int | None = None,
) -> np.ndarray:
    """Z-score normalisation.

    If `suspect_only` is True and `n_suspect` is provided, use only the
    first *n_suspect* rows of `ref` (or `arr`) to compute stats. This
    matches the paper's "train" normalization mode.
    """
    base = ref if ref is not None else arr
    if suspect_only and n_suspect is not None and n_suspect > 0:
        base = base[:n_suspect]
    mu = base.mean(axis=0, keepdims=True)
    sigma = base.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return np.asarray((arr - mu) / sigma, dtype=float)


def _clean_features(arr: np.ndarray, fraction: float = 0.025) -> np.ndarray:
    return np.column_stack(
        [remove_outliers_to_mean(arr[:, j], fraction=fraction) for j in range(arr.shape[1])]
    )


def _train_logistic_regressor(
    x: np.ndarray,
    y: np.ndarray,
    num_epochs: int = 1000,
    lr: float = 0.01,
) -> tuple[np.ndarray, float]:
    """Train a single-layer logistic regressor with BCE loss (Maini §5.1).

    Returns the learnt coefficient vector and intercept so that predictions
    are raw logits: ``scores = x @ coef + intercept``.
    """
    n_features = x.shape[1]
    model = nn.Linear(n_features, 1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    for _ in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(x_t).squeeze()
        loss = criterion(outputs, y_t)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        coef = np.atleast_1d(model.weight.data.squeeze().numpy())
        intercept = float(model.bias.data.item())

    return coef, intercept


def _p_value_curve(
    suspect_scores: np.ndarray,
    val_scores: np.ndarray,
    sample_sizes: list[int] | None = None,
) -> list[float]:
    """Compute one-sided t-test p-values at multiple sample sizes.

    Mirrors ``get_p_value_list`` in the reference implementation.
    """
    if sample_sizes is None:
        sample_sizes = _P_SAMPLE_LIST
    curve: list[float] = []
    for n in sample_sizes:
        n_sus = min(n, len(suspect_scores))
        n_val = min(n, len(val_scores))
        if n_sus < 2 or n_val < 2:
            curve.append(1.0)
            continue
        p = one_sided_t_test_lower(
            suspect_scores[:n_sus],  # type: ignore[arg-type]
            val_scores[:n_val],  # type: ignore[arg-type]
        )
        curve.append(p)
    return curve


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
    normalize_mode: str = "train",
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
        normalize_mode: ``"train"`` uses suspect-only stats for z-scoring
            (paper default); ``"combined"`` uses suspect+validation stats.
        num_context_examples: used only by context-dependent detectors (e.g. CoDeC).
        threshold: p-value threshold for the `trained` verdict.
        progress: show tqdm bars.
    """
    if len(detectors) == 0:
        raise ValueError("At least one detector required.")
    if len(suspect) < 4 or len(validation) < 4:
        raise ValueError("Need at least 4 samples per side for stable A/B splits.")
    if normalize_mode not in {"train", "combined"}:
        raise ValueError(f"normalize_mode must be 'train' or 'combined', got {normalize_mode}")

    suspect = list(suspect)
    validation = list(validation)
    cap = min(holdout_size, len(suspect), len(validation))
    detector_names = [d.name for d in detectors]

    logger.info(
        f"Dataset Inference: model={model.name} detectors={detector_names} "
        f"|suspect|={len(suspect)} |val|={len(validation)} cap={cap} seeds={n_seeds}"
    )

    p_values: list[float] = []
    p_value_curves: list[list[float]] = []
    weight_accumulator = np.zeros(len(detectors), dtype=float)

    for seed in range(n_seeds):
        rng = random.Random(seed)
        sus_pool = rng.sample(suspect, cap)
        val_pool = rng.sample(validation, cap)

        n_train = int(cap * train_fraction)

        # Stage 1: score the FULL pool (train + test) so that outlier cleaning
        # and normalization see the same distribution for both splits.
        feats_sus = _score_corpus(
            detectors,
            model,
            sus_pool,
            desc=f"seed {seed} suspect",
            rng=rng,
            num_context_examples=num_context_examples,
            progress=progress,
        )
        feats_val = _score_corpus(
            detectors,
            model,
            val_pool,
            desc=f"seed {seed} validation",
            rng=rng,
            num_context_examples=num_context_examples,
            progress=progress,
        )

        # Stage 2: outlier removal on the FULL scored pool (paper §5.1).
        cleaned_sus = _clean_features(feats_sus, fraction=outlier_fraction)
        cleaned_val = _clean_features(feats_val, fraction=outlier_fraction)

        # Split the cleaned features back into train / test.
        sus_train_clean = cleaned_sus[:n_train]
        sus_test_clean = cleaned_sus[n_train:]
        val_train_clean = cleaned_val[:n_train]
        val_test_clean = cleaned_val[n_train:]

        train_features_raw = np.vstack([sus_train_clean, val_train_clean])
        train_features = _normalize_columns(
            train_features_raw,
            suspect_only=(normalize_mode == "train"),
            n_suspect=len(sus_train_clean),
        )
        train_labels = np.concatenate(
            [
                np.zeros(len(sus_train_clean)),
                np.ones(len(val_train_clean)),
            ]
        )

        coef, intercept = _train_logistic_regressor(train_features, train_labels)

        test_features = _normalize_columns(
            np.vstack([sus_test_clean, val_test_clean]),
            ref=train_features_raw,
            suspect_only=(normalize_mode == "train"),
            n_suspect=len(sus_train_clean),
        )
        scores = test_features @ coef + intercept
        sus_scores = scores[: len(sus_test_clean)]
        val_scores = scores[len(sus_test_clean) :]

        # Stage 3: t-test at multiple sample sizes (paper §5.1).
        curve = _p_value_curve(sus_scores, val_scores)
        p_value_curves.append(curve)
        p_values.append(curve[-1])
        weight_accumulator += coef
        logger.debug(f"seed={seed} p={curve[-1]:.4g}")

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
        p_value_curves=p_value_curves,
        feature_weights=weights,
        metadata={
            "threshold": threshold,
            "n_seeds": n_seeds,
            "holdout_size": cap,
            "train_fraction": train_fraction,
            "outlier_fraction": outlier_fraction,
            "normalize_mode": normalize_mode,
        },
    )
