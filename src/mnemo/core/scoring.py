from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


def dataset_level_auc(
    seen_scores: Sequence[float],
    unseen_scores: Sequence[float],
) -> float:
    """AUC for separating seen (label=1) from unseen (label=0) at dataset level.

    Mirrors Table 1 of CoDeC paper (Zawalski et al., 2025).
    """
    if not seen_scores or not unseen_scores:
        raise ValueError("Both seen_scores and unseen_scores must be non-empty.")

    y_true = np.concatenate([np.ones(len(seen_scores)), np.zeros(len(unseen_scores))])
    y_score = np.concatenate([np.asarray(seen_scores), np.asarray(unseen_scores)])
    return float(roc_auc_score(y_true, y_score))
