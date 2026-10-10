"""Trinet wrist camera capture (bible 5.2) and a fake for mock mode.

SEMANTIC-ONLY: frames feed the Critic (after dewarp) and logs. Never metric control.
Raw frames are published; dewarping happens in the Critic worker, off the FSM thread.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from lbyr.config import WristCameraConfig, require
from lbyr.perception.buffers import CaptureThread, LatestFrameBuffer

log = logging.getLogger(__name__)


class TrinetWristCamera:
    """UVC capture via cv2.VideoCapture. Vendor-SDK path is TODO pending OQ-7."""

    def __init__(self, cfg: WristCameraConfig) -> None:
        self.index = require(cfg.device_index, "Trinet device index / UVC mode", "OQ-7")
        self.width = require(cfg.width, "Trinet capture width", "OQ-8")
        self.height = require(cfg.height, "Trinet capture height", "OQ-8")
        self.fps = require(cfg.fps, "Trinet capture FPS", "OQ-8")
        self.buffer: LatestFrameBuffer[np.ndarray] = LatestFrameBuffer()
        self._cap: Any = None
        self._thread: CaptureThread[np.ndarray] | None = None

    def start(self) -> None:
        import cv2  # lazy

        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open Trinet at index {self.index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap = cap
        self._thread = CaptureThread("wrist-cam", self._grab, self.buffer)
        self._thread.start()
        log.info("Trinet started on index %d", self.index)

    def _grab(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return frame if ok else None

    def stop(self) -> None:
        if self._thread is not None:
            self._thread.stop()
        if self._cap is not None:
            self._cap.release()


class FakeWristCamera:
    """Publishes a synthetic BGR frame (gripper jaws + object patch) at ~30 Hz."""

    def __init__(self, width: int = 640, height: int = 480, fps: float = 30.0) -> None:
        self.width, self.height = width, height
        self.buffer: LatestFrameBuffer[np.ndarray] = LatestFrameBuffer()
        self._frame = self._render()
        self._thread = CaptureThread("wrist-fake", lambda: self._frame, self.buffer, 1.0 / fps)

    def _render(self) -> np.ndarray:
        h, w = self.height, self.width
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :] = np.linspace(40, 120, w, dtype=np.uint8)[None, :, None]
        img[h // 3 : h, w // 4 : w // 4 + 30] = 200  # left jaw
        img[h // 3 : h, 3 * w // 4 - 30 : 3 * w // 4] = 200  # right jaw
        img[h // 2 : h // 2 + 80, w // 2 - 50 : w // 2 + 50] = (40, 160, 220)  # object
        return img

    def start(self) -> None:
        self.buffer.put(self._frame)
        self._thread.start()

    def stop(self) -> None:
        self._thread.stop()
