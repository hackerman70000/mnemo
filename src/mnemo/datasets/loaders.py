from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from datasets import load_dataset
from loguru import logger


def load_pickle(path: Path | str) -> list[str]:
    path = Path(path)
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data).__name__}")
    return [str(x) for x in data]


def load_text_chunks(
    path: Path | str,
    chunk_size: int = 500,
    min_chunk_size: int = 100,
) -> list[str]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    chunks: list[str] = []
    while len(text) > chunk_size:
        sample = text[:chunk_size]
        sample = " ".join(sample.split(" ")[1:-1])
        if len(sample) > min_chunk_size:
            chunks.append(sample.strip())
        text = text[chunk_size:]

    if chunks:
        avg = float(np.mean([len(c) for c in chunks]))
        logger.info(f"Loaded {len(chunks)} chunks from {path} (avg {avg:.0f} chars)")
    return chunks


def load_local(path: Path | str, **kwargs: object) -> list[str]:
    path = Path(path)
    if path.suffix == ".pkl":
        return load_pickle(path)
    if path.suffix == ".txt":
        return load_text_chunks(path, **kwargs)  # type: ignore[arg-type]
    raise ValueError(f"Unsupported file format: {path.suffix} (use .pkl or .txt)")


def load_hf(
    name: str,
    config: str | None = None,
    split: str = "train",
    text_field: str = "text",
    limit: int | None = None,
) -> list[str]:
    ds = load_dataset(name, config) if config else load_dataset(name)
    rows = ds[split]
    if text_field not in rows.column_names:
        raise KeyError(f"Field '{text_field}' not in dataset columns: {rows.column_names}")
    samples = list(rows[text_field])
    if limit is not None:
        samples = samples[:limit]
    return [str(s) for s in samples]
