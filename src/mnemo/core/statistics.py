"""Statistical primitives for dataset-level inference (Maini et al. 2024)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy import stats


def remove_outliers_to_mean(values: np.ndarray, fraction: float = 0.025) -> np.ndarray:
    """Replace top and bottom `fraction` of values with the mean of the rest.

    Mirrors the pre-regressor cleanup in Maini et al. §5.1 — keeps the
    feature distribution stable and prevents a few extreme samples from
    dominating the linear classifier.
    """
    if not 0 <= fraction < 0.5:
        raise ValueError(f"fraction must be in [0, 0.5), got {fraction}")
    if values.size == 0:
        return values

    arr = values.copy()
    if fraction == 0:
        return arr

    n_clip = max(1, int(arr.size * fraction))
    sorted_idx = np.argsort(arr)
    low_mask = sorted_idx[:n_clip]
    high_mask = sorted_idx[-n_clip:]
    middle = np.delete(arr, np.concatenate([low_mask, high_mask]))
    if middle.size == 0:
        return arr
    fill = float(np.mean(middle))
    arr[low_mask] = fill
    arr[high_mask] = fill
    return arr


def one_sided_t_test_lower(
    suspect_scores: Sequence[float],
    val_scores: Sequence[float],
) -> float:
    """One-sided Welch's t-test: H1 says suspect mean < validation mean.

    Maini et al.: members of the training set produce systematically lower
    aggregated scores than non-members. Returns the p-value for the
    alternative `μ(suspect) < μ(val)`.
    """
    sus = np.asarray(suspect_scores, dtype=float)
    val = np.asarray(val_scores, dtype=float)
    if sus.size < 2 or val.size < 2:
        raise ValueError("Need at least 2 samples per group for the t-test.")

    result = stats.ttest_ind(sus, val, equal_var=False, alternative="less")
    return float(result.pvalue)


def sidak_combine(p_values: Sequence[float]) -> float:
    """Combine dependent p-values using the Šidák method (Maini eq. 2).

    `p_combined = 1 - exp(sum(log(1 - p_i)))` — conservative under
    dependence between the individual tests, which is what we get from
    running the same pipeline with different random seeds.
    """
    if not p_values:
        raise ValueError("Need at least one p-value to combine.")
    clipped = np.clip(np.asarray(p_values, dtype=float), 0.0, 1.0 - 1e-12)
    log_terms = np.log1p(-clipped)
    return float(1.0 - math.exp(float(np.sum(log_terms))))
