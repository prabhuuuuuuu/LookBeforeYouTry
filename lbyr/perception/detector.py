"""Object detection interface (bible 11.3). The VLM never does raw detection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Detection:
    id: int  # stable ID the planner refers to (target_id)
    label: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]  # pixels

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


class Detector(Protocol):
    def detect(self, color_bgr: np.ndarray) -> list[Detection]: ...


class FakeDetector:
    """Returns a fixed list of detections (mock mode / tests)."""

    def __init__(self, detections: Sequence[Detection]) -> None:
        self._detections = list(detections)

    def detect(self, color_bgr: np.ndarray) -> list[Detection]:
        return list(self._detections)


class YoloTRTDetector:
    """YOLOv11n TensorRT INT8 backend (bible 9.2: ~0.4 GB VRAM, ~8 ms, 30 Hz).

    TODO: implement once the engine is built on the Jetson. Not hardware-blocked.
    """

    def __init__(self, engine_path: Path) -> None:
        self.engine_path = engine_path
        raise NotImplementedError("YOLOv11n TensorRT backend not implemented yet (TODO)")

    def detect(self, color_bgr: np.ndarray) -> list[Detection]:
        raise NotImplementedError
