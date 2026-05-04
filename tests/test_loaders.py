from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from mnemo.datasets.loaders import load_local, load_pickle, load_text_chunks


def test_load_pickle_roundtrip(tmp_path: Path):
    path = tmp_path / "data.pkl"
    samples = ["one", "two", "three"]
    with path.open("wb") as f:
        pickle.dump(samples, f)
    assert load_pickle(path) == samples


def test_load_pickle_coerces_non_strings(tmp_path: Path):
    path = tmp_path / "data.pkl"
    with path.open("wb") as f:
        pickle.dump([1, 2.5, "three"], f)
    assert load_pickle(path) == ["1", "2.5", "three"]


def test_load_pickle_rejects_non_list(tmp_path: Path):
    path = tmp_path / "bad.pkl"
    with path.open("wb") as f:
        pickle.dump({"oops": 1}, f)
    with pytest.raises(ValueError, match="Expected list"):
        load_pickle(path)


def test_load_text_chunks_returns_chunks(tmp_path: Path):
    path = tmp_path / "doc.txt"
    path.write_text("word " * 1000)
    chunks = load_text_chunks(path, chunk_size=200, min_chunk_size=50)
    assert chunks
    assert all(len(c) >= 50 for c in chunks)


def test_load_local_dispatches_by_suffix(tmp_path: Path):
    pkl = tmp_path / "x.pkl"
    with pkl.open("wb") as f:
        pickle.dump(["a"], f)
    assert load_local(pkl) == ["a"]

    with pytest.raises(ValueError, match="Unsupported file format"):
        load_local(tmp_path / "x.json")
