"""Maini-style Dataset Inference (NeurIPS 2024) on Pythia x Pile.

Setup mirrors paper section 5.4: Pythia models trained on the Pile, with
the official train/validation splits from `iamgroot42/mimir`. Suspect =
train samples; validation = held-out IID samples.

Expected: p_combined < 0.1 across all Pile subsets, with no false
positives when comparing val-vs-val (Maini Fig. 4).
"""

from __future__ import annotations

from datasets import load_dataset
from loguru import logger

from mnemo.detectors import MaxKProb, MinKProb, Perplexity, VanillaLoss, ZlibRatio
from mnemo.models.hf import HFModel
from mnemo.pipelines.dataset_inference import dataset_inference


def _load_member_and_nonmember(subset: str) -> tuple[list[str], list[str]]:
    ds = load_dataset("iamgroot42/mimir", subset, split="ngram_13_0.8", trust_remote_code=True)
    return list(ds["member"]), list(ds["nonmember"])


def main() -> None:
    model = HFModel("EleutherAI/pythia-410m")
    detectors = [VanillaLoss(), Perplexity(), MinKProb(), MaxKProb(), ZlibRatio()]

    for subset in ("wikipedia_(en)", "github"):
        logger.info(f"=== {subset} ===")
        suspect, validation = _load_member_and_nonmember(subset)

        result = dataset_inference(
            model,
            suspect,
            validation,
            detectors,
            n_seeds=5,
            holdout_size=500,
        )
        verdict = "TRAINED" if result.trained else "no evidence"
        logger.info(f"{subset}: {verdict} (p={result.p_combined:.4g})")
        logger.info(f"  weights: {result.feature_weights}")


if __name__ == "__main__":
    main()
