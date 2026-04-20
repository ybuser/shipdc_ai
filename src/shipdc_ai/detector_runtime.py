"""Detector runtime interface placeholder."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DetectorResult:
    camera_id: str
    timestamp: datetime
    label: str
    confidence: float
    box_xyxy: tuple[float, float, float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DetectorRuntime:
    """Future CPU-only ONNX detector wrapper."""

    def __init__(self, model_path: Path, providers: tuple[str, ...] = ("CPUExecutionProvider",)) -> None:
        self.model_path = Path(model_path)
        self.providers = providers

    def infer(self, _frame: object, *, camera_id: str, timestamp: datetime) -> list[DetectorResult]:
        raise NotImplementedError(
            "Detector inference is not implemented in the scaffold phase. "
            "Future runtime should use CPU-only ONNX Runtime, not Torch or Ultralytics."
        )
