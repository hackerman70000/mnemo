from __future__ import annotations

from mnemo.core.result import DatasetResult
from mnemo.detectors import CoDeC, VanillaLoss
from mnemo.pipelines.detection import detect_contamination


def test_pipeline_runs_with_codec(memorized_model):
    samples = [f"sample number {i} text body" for i in range(10)]
    result = detect_contamination(
        CoDeC(skip_first_tokens=0),
        memorized_model,
        samples,
        dataset_name="test",
        num_context_examples=1,
        progress=False,
    )
    assert result.detector == "codec"
    assert result.n_samples == 10
    assert 0.0 <= result.score <= 1.0


def test_pipeline_skips_context_for_non_context_detector(constant_model):
    samples = [f"sample {i}" for i in range(5)]
    result = detect_contamination(
        VanillaLoss(),
        constant_model,
        samples,
        dataset_name="test",
        progress=False,
    )
    assert result.metadata["num_context_examples"] == 0
    assert result.n_samples == 5


def test_pipeline_respects_max_samples(constant_model):
    samples = [f"sample {i}" for i in range(50)]
    result = detect_contamination(
        VanillaLoss(),
        constant_model,
        samples,
        max_samples=10,
        progress=False,
    )
    assert result.n_samples == 10


def test_result_serialization_roundtrip(tmp_path, constant_model):
    samples = [f"sample {i}" for i in range(5)]
    result = detect_contamination(VanillaLoss(), constant_model, samples, progress=False)
    out = tmp_path / "result.json"
    result.save_json(out)
    loaded = DatasetResult.load_json(out)
    assert loaded.detector == result.detector
    assert loaded.sample_scores == result.sample_scores
