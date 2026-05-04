from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from loguru import logger

from mnemo.core.detector import Detector
from mnemo.core.result import DatasetResult
from mnemo.core.scoring import dataset_level_auc
from mnemo.models.base import ModelBackend
from mnemo.pipelines.detection import detect_contamination


@dataclass
class AUCReport:
    detector: str
    model: str
    auc: float
    seen: dict[str, DatasetResult] = field(default_factory=dict)
    unseen: dict[str, DatasetResult] = field(default_factory=dict)

    def summary(self) -> dict[str, object]:
        return {
            "detector": self.detector,
            "model": self.model,
            "auc": self.auc,
            "seen": {n: r.score for n, r in self.seen.items()},
            "unseen": {n: r.score for n, r in self.unseen.items()},
        }


def evaluate_auc(
    detector: Detector,
    model: ModelBackend,
    seen_datasets: Mapping[str, Sequence[str]],
    unseen_datasets: Mapping[str, Sequence[str]],
    **detection_kwargs: object,
) -> AUCReport:
    """Reproduce paper Tab. 1 — dataset-level AUC for seen vs. unseen sets."""
    if not seen_datasets or not unseen_datasets:
        raise ValueError("Need at least one seen and one unseen dataset.")

    seen_results = {
        name: detect_contamination(detector, model, ds, dataset_name=name, **detection_kwargs)  # type: ignore[arg-type]
        for name, ds in seen_datasets.items()
    }
    unseen_results = {
        name: detect_contamination(detector, model, ds, dataset_name=name, **detection_kwargs)  # type: ignore[arg-type]
        for name, ds in unseen_datasets.items()
    }

    auc = dataset_level_auc(
        [r.score for r in seen_results.values()],
        [r.score for r in unseen_results.values()],
    )

    logger.info(f"{detector.name} on {model.name}: AUC = {auc:.3f}")

    return AUCReport(
        detector=detector.name,
        model=model.name,
        auc=auc,
        seen=seen_results,
        unseen=unseen_results,
    )
