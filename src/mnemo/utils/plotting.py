"""Quick visual sanity checks for detector outputs.

Lazy-imports matplotlib so the package stays light when plots aren't
needed. Install the optional extra with `uv sync --extra plotting`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _lazy_pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError("matplotlib not installed. Run `uv sync --extra plotting`.") from exc
    return plt


def plot_score_histogram(
    scores: Sequence[float],
    *,
    title: str = "Detector scores",
    bins: int = 50,
    save_path: Path | str | None = None,
) -> None:
    """Histogram of per-sample scores. First sanity check on any new run."""
    plt = _lazy_pyplot()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(list(scores), bins=bins, color="steelblue", edgecolor="white")
    ax.set_xlabel("score")
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close(fig)


def plot_roc(
    member_scores: Sequence[float],
    nonmember_scores: Sequence[float],
    *,
    title: str = "ROC curve",
    save_path: Path | str | None = None,
) -> float:
    """Plot ROC and return AUC. Reveals miscalibration / wrong direction."""
    import numpy as np
    from sklearn.metrics import auc, roc_curve

    plt = _lazy_pyplot()
    y_true = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
    y_score = np.concatenate([np.asarray(member_scores), np.asarray(nonmember_scores)])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = float(auc(fpr, tpr))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close(fig)
    return roc_auc
