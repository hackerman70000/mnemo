"""Mini reproduction of paper Tab. 1: CoDeC vs three baselines on Pythia-410M.

Expected pattern: CoDeC achieves AUC ≈ 1.0 separating Pile (seen) from
benchmarks (unseen); baselines have noticeable overlap.
"""

from __future__ import annotations

from loguru import logger

from mnemo.datasets.benchmarks import (
    gpqa_diamond,
    gsm8k_test,
    pile_github,
    pile_wikipedia,
)
from mnemo.detectors import CoDeC, MinKProb, VanillaLoss, ZlibRatio
from mnemo.models.hf import HFModel
from mnemo.pipelines.auc_eval import evaluate_auc


def main() -> None:
    model = HFModel("EleutherAI/pythia-410m")

    seen = {
        "pile_wikipedia": pile_wikipedia(),
        "pile_github": pile_github(),
    }
    unseen = {
        "gsm8k": gsm8k_test(),
        "gpqa": gpqa_diamond(),
    }

    for det_cls in (CoDeC, VanillaLoss, MinKProb, ZlibRatio):
        report = evaluate_auc(det_cls(), model, seen, unseen, max_samples=300)
        logger.info(report.summary())


if __name__ == "__main__":
    main()
