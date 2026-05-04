"""Finetune-probe demo (paper Sec. 3.3): watch CoDeC climb on a candidate dataset.

Useful for models with undisclosed training corpora — finetune briefly and
check whether CoDeC saturates near 100% (signalling the model already
internalises the dataset's distribution).
"""

from __future__ import annotations

from loguru import logger

from mnemo.datasets.benchmarks import gsm8k_test
from mnemo.detectors import CoDeC
from mnemo.models.hf import HFModel
from mnemo.pipelines.finetune_probe import finetune_probe


def main() -> None:
    model = HFModel("EleutherAI/pythia-410m")
    detector = CoDeC()

    dataset = gsm8k_test()[:200]

    result = finetune_probe(
        detector,
        model,
        dataset,
        dataset_name="gsm8k",
        num_steps=60,
        eval_every=10,
        eval_max_samples=100,
        learning_rate=3e-5,
    )

    logger.info("Score trajectory:")
    for step, score in result.scores_per_step:
        logger.info(f"  step={step:>3d}  score={score:.4f}")


if __name__ == "__main__":
    main()
