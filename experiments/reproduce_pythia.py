"""Reproduce the notebook's CoDeC numbers on Pythia-410M (Pile vs benchmarks).

Expected (from CoDeC paper supplementary notebook):
    pile_wikipedia ≈ 0.95   pile_github ≈ 0.90
    gsm8k          ≈ 0.06   gpqa        ≈ 0.42
"""

from __future__ import annotations

from loguru import logger

from mnemo.datasets.benchmarks import (
    gpqa_diamond,
    gsm8k_test,
    pile_github,
    pile_wikipedia,
)
from mnemo.detectors import CoDeC
from mnemo.models.hf import HFModel
from mnemo.pipelines.detection import detect_contamination


def main() -> None:
    model = HFModel("EleutherAI/pythia-410m")
    detector = CoDeC()

    splits = {
        "Seen (Pile)": {
            "pile_wikipedia": pile_wikipedia,
            "pile_github": pile_github,
        },
        "Unseen": {
            "gsm8k": gsm8k_test,
            "gpqa": gpqa_diamond,
        },
    }

    for label, datasets in splits.items():
        logger.info(f"=== {label} ===")
        for name, loader in datasets.items():
            data = loader()
            r = detect_contamination(detector, model, data, dataset_name=name, max_samples=1000)
            logger.info(f"  {name}: {r.score:.4f} (n={r.n_samples})")


if __name__ == "__main__":
    main()
