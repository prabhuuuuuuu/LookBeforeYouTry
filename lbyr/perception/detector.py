"""Object detection interface (bible 11.3). The VLM never does raw detection.

A detector takes the latest overhead frame (color + aligned depth + intrinsics, or ``None``
when no overhead camera runs, e.g. ``--zone``) and returns :class:`Detection` objects in
the one format the planners, the target providers and the logs use.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from lbyr.perception.overhead import RGBDFrame


@dataclass(frozen=True)
class Detection:
    id: int  # stable ID the planner refers to (target_id)
    label: str  # class, e.g. "wood_block"
    score: float
    bbox_xyxy: tuple[float, float, float, float]  # pixels
    centroid_px: tuple[float, float] | None = None  # contour centroid; None = bbox centre
    yaw_deg: float | None = None  # image-plane yaw of the long side, [-90, 90)
    xyz_cam_mm: tuple[float, float, float] | None = None  # D435 camera frame, mm
    zone: str | None = None  # taught zone containing the centroid
    area_px: float | None = None

    @property
    def centroid(self) -> tuple[float, float]:
        if self.centroid_px is not None:
            return self.centroid_px
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


class Detector(Protocol):
    def detect(self, frame: RGBDFrame | None) -> list[Detection]: ...


class FakeDetector:
    """Returns a fixed list of detections (mock mode / tests)."""

    def __init__(self, detections: Sequence[Detection]) -> None:
        self._detections = list(detections)

    def detect(self, frame: RGBDFrame | None) -> list[Detection]:
        return list(self._detections)


class FixedDetector(FakeDetector):
    """Always returns the same detections. Used by ``--zone X``: the block is known to be
    in taught zone X, so no overhead camera / detection runs."""


class YoloTRTDetector:
    """YOLOv11n TensorRT INT8 backend (bible 9.2: ~0.4 GB VRAM, ~8 ms, 30 Hz).

    TODO: implement once the engine is built on the Jetson. Not hardware-blocked. The live
    demo (one wooden block on a white table) uses :class:`lbyr.perception.block.BlockDetector`.
    """

    def __init__(self, engine_path: Path) -> None:
        self.engine_path = engine_path
        raise NotImplementedError("YOLOv11n TensorRT backend not implemented yet (TODO)")

    def detect(self, frame: RGBDFrame | None) -> list[Detection]:
        raise NotImplementedError
