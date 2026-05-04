from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DatasetResult:
    detector: str
    model: str
    dataset_name: str
    sample_scores: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        if not self.sample_scores:
            return 0.0
        return float(np.mean(self.sample_scores))

    @property
    def n_samples(self) -> int:
        return len(self.sample_scores)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["score"] = self.score
        d["n_samples"] = self.n_samples
        return d

    def save_json(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load_json(cls, path: Path | str) -> DatasetResult:
        data = json.loads(Path(path).read_text())
        return cls(
            detector=data["detector"],
            model=data["model"],
            dataset_name=data["dataset_name"],
            sample_scores=data["sample_scores"],
            metadata=data.get("metadata", {}),
        )
