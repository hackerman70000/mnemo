"""Finetuning probe — paper Section 3.3.

Take a model with unknown training corpus, finetune on a candidate dataset,
and watch the contamination score climb. If it stays low after several
batches, the model probably had not seen the data; if it shoots toward 100%
within a handful of steps, the dataset (or close kin) was already in train.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from mnemo.core.detector import Detector
from mnemo.models.hf import HFModel
from mnemo.pipelines.detection import detect_contamination


@dataclass
class FinetuneProbeResult:
    detector: str
    model: str
    dataset_name: str
    scores_per_step: list[tuple[int, float]] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


class _CausalLMDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        texts: Sequence[str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        ids = enc["input_ids"][0]
        return {"input_ids": ids}


_CollateFn = Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]


def _make_collate_fn(pad_id: int, device: str) -> _CollateFn:
    def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        ids = [b["input_ids"] for b in batch]
        max_len = max(x.size(0) for x in ids)
        padded = torch.stack(
            [torch.cat([x, torch.full((max_len - x.size(0),), pad_id, dtype=x.dtype)]) for x in ids]
        )
        labels = padded.clone()
        labels[labels == pad_id] = -100
        return {
            "input_ids": padded.to(device),
            "labels": labels.to(device),
        }

    return collate


def finetune_probe(
    detector: Detector,
    model: HFModel,
    dataset: Sequence[str],
    *,
    dataset_name: str = "probe",
    num_steps: int = 60,
    batch_size: int = 1,
    learning_rate: float = 3e-5,
    eval_every: int = 10,
    eval_max_samples: int = 100,
    max_seq_length: int = 512,
    detection_kwargs: dict[str, Any] | None = None,
) -> FinetuneProbeResult:
    detection_kwargs = detection_kwargs or {}
    detection_kwargs.setdefault("max_samples", eval_max_samples)
    detection_kwargs.setdefault("num_context_examples", 1)
    detection_kwargs.setdefault("progress", False)

    result = FinetuneProbeResult(
        detector=detector.name,
        model=model.name,
        dataset_name=dataset_name,
        metadata={
            "num_steps": num_steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_seq_length": max_seq_length,
        },
    )

    pre = detect_contamination(
        detector, model, dataset, dataset_name=dataset_name, **detection_kwargs
    )
    result.scores_per_step.append((0, pre.score))
    logger.info(f"[probe step 0] {detector.name}={pre.score:.3f}")

    optim = torch.optim.AdamW(model.model.parameters(), lr=learning_rate)
    pad_id = model.tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer must have a pad_token_id for finetune probe.")

    ds = _CausalLMDataset(dataset, model.tokenizer, max_length=max_seq_length)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_make_collate_fn(pad_id, model.device),
    )

    model.model.train()
    step = 0
    while step < num_steps:
        for batch in loader:
            optim.zero_grad()
            out = model.model(**batch)
            out.loss.backward()
            optim.step()
            step += 1

            if step % eval_every == 0 or step == num_steps:
                model.model.eval()  # type: ignore[no-untyped-call]
                snapshot = detect_contamination(
                    detector, model, dataset, dataset_name=dataset_name, **detection_kwargs
                )
                result.scores_per_step.append((step, snapshot.score))
                logger.info(f"[probe step {step}] {detector.name}={snapshot.score:.3f}")
                model.model.train()

            if step >= num_steps:
                break

    model.model.eval()  # type: ignore[no-untyped-call]
    return result
