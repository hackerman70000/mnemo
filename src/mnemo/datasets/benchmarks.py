"""Convenience loaders for benchmarks used in the CoDeC paper."""

from __future__ import annotations

from collections.abc import Callable

from datasets import load_dataset


def gpqa_diamond() -> list[str]:
    df = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"].to_pandas()
    return [str(q) for q in df["Question"].tolist()]


def gsm8k_test() -> list[str]:
    df = load_dataset("openai/gsm8k", "main")["test"].to_pandas()
    return [str(q) for q in df["question"].tolist()]


def mmlu_test() -> list[str]:
    df = load_dataset("cais/mmlu", "all")["test"].to_pandas()
    return [str(q) for q in df["question"].tolist()]


def pile_subset(subset: str = "wikipedia_(en)") -> list[str]:
    ds = load_dataset("iamgroot42/mimir", subset, split="ngram_13_0.8", trust_remote_code=True)
    return list(ds["member"])


def pile_wikipedia() -> list[str]:
    return pile_subset("wikipedia_(en)")


def pile_github() -> list[str]:
    return pile_subset("github")


BENCHMARKS: dict[str, Callable[[], list[str]]] = {
    "gpqa": gpqa_diamond,
    "gsm8k": gsm8k_test,
    "mmlu": mmlu_test,
    "pile_wikipedia": pile_wikipedia,
    "pile_github": pile_github,
}
