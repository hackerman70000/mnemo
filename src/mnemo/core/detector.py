from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from mnemo.models.base import ModelBackend


class Detector(ABC):
    name: ClassVar[str]
    requires_context: ClassVar[bool] = False

    @abstractmethod
    def score_sample(
        self,
        sample: str,
        model: ModelBackend,
        context: list[str] | None = None,
    ) -> float:
        """Per-sample contamination score. Higher = more likely contaminated."""
