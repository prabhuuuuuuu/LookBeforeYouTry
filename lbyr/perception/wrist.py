"""Trinet wrist camera capture (bible 5.2) and a fake for mock mode.

SEMANTIC-ONLY: frames feed the Critic (after dewarp) and logs. Never metric control.
Raw frames are published to a latest-frame buffer; the dewarp (:func:`build_wrist_preprocess`)
and the Critic crop run in the Critic worker, off the FSM thread.

Pipeline: Trinet (UVC, ``cv2.VideoCapture`` by index or V4L2 path) -> raw frame ->
dewarp if ``data/calib/trinet_fisheye.yaml`` exists (else a warning and the raw frame)
-> gripper box (``CriticConfig.crop_box_norm``) resized to 378x378 -> Moondream2.
Cost: capture ~1-5 ms, dewarp remap ~2-5 ms per Critic call (bible 5.2), 0 VRAM.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from lbyr.config import WristCameraConfig, require
from lbyr.errors import CalibrationError, HardwareNotConfirmedError
from lbyr.perception.buffers import CaptureThread, LatestFrameBuffer

log = logging.getLogger(__name__)


def wrist_device(cfg: WristCameraConfig) -> int | str:
    """The configured Trinet device: V4L2 path if set, else the index (OQ-7)."""
    if cfg.device_path:
        return cfg.device_path
    if cfg.device_index is None:
        raise HardwareNotConfirmedError("Trinet device index / path (UVC mode)", "OQ-7")
    return cfg.device_index


class TrinetWristCamera:
    """UVC capture via cv2.VideoCapture (index or path). Vendor-SDK path: TODO (OQ-7)."""

    def __init__(self, cfg: WristCameraConfig) -> None:
        self.device = wrist_device(cfg)
        self.width = require(cfg.width, "Trinet capture width", "OQ-8")
        self.height = require(cfg.height, "Trinet capture height", "OQ-8")
        self.fps = require(cfg.fps, "Trinet capture FPS", "OQ-8")
        self.fourcc = cfg.fourcc
        self.buffer: LatestFrameBuffer[np.ndarray] = LatestFrameBuffer()
        self._cap: Any = None
        self._thread: CaptureThread[np.ndarray] | None = None
        self.actual_size: tuple[int, int] | None = None

    @property
    def measured_fps(self) -> float:
        return self._thread.fps if self._thread is not None else 0.0

    def start(self) -> None:
        import cv2  # lazy

        if isinstance(self.device, str):
            cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(self.device)
        if not cap.isOpened():
            raise RuntimeError(
                f"cannot open the Trinet at {self.device!r} (tools/fps_test.py --list; OQ-7)"
            )
        if self.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"the Trinet at {self.device!r} opened but gave no frame")
        self.actual_size = (int(frame.shape[1]), int(frame.shape[0]))
        if self.actual_size != (self.width, self.height):
            log.warning(
                "Trinet delivers %dx%d, config asks %dx%d (OQ-8): fix WristCameraConfig",
                *self.actual_size,
                self.width,
                self.height,
            )
        self._cap = cap
        self.buffer.put(frame)
        self._thread = CaptureThread("wrist-cam", self._grab, self.buffer)
        self._thread.start()
        log.info("Trinet started on %r at %dx%d", self.device, *self.actual_size)

    def _grab(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return frame if ok else None

    def stop(self) -> None:
        if self._thread is not None:
            self._thread.stop()
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None


_JPEG_HEADER_MARKERS = {0xC0, 0xC1, 0xC2, 0xC4, 0xDB, 0xDD, 0xDA, 0xFE, *range(0xE0, 0xF0)}


def _walks_to_sos(b: bytes, k: int, limit: int = 8) -> bool:
    """True if a chain of valid header segments starting at ``k`` reaches SOS (FF DA)."""
    for _ in range(limit):
        if k + 4 > len(b) or b[k] != 0xFF or b[k + 1] not in _JPEG_HEADER_MARKERS:
            return False
        if b[k + 1] == 0xDA:
            return True
        k += 2 + int.from_bytes(b[k + 2 : k + 4], "big")
    return False


def strip_trinet_sei(b: bytes) -> bytes:
    """Repair a Trinet MJPEG frame (measured 2026-10-11).

    The camera inserts an H.264-style SEI NAL (``00 00 00 01 06 05 ... "TRINETIMUSEI"``,
    IMU data, ~1.1-1.4 KB) INSIDE the first DHT segment, right after its length field, so
    every standard decoder fails (OpenCV's V4L2 path silently returns all-zero frames).
    Walk the header; where a segment's declared length does not land on a marker, cut from
    after its length field to ``(L - 2)`` bytes before the next marker from which the
    segment chain validly reaches SOS. Clean JPEGs are returned unchanged. Cost ~0.1 ms.
    """
    i = 2
    while i + 4 <= len(b) and b[i] == 0xFF:
        if b[i + 1] == 0xDA:
            return b
        seg_len = int.from_bytes(b[i + 2 : i + 4], "big")
        nxt = i + 2 + seg_len
        if nxt < len(b) and b[nxt] != 0xFF:  # body displaced by the SEI blob
            k = b.find(b"\xff", nxt)
            while k >= 0 and not _walks_to_sos(b, k):
                k = b.find(b"\xff", k + 1)
            if k < 0:
                return b
            return b[: i + 4] + b[k - (seg_len - 2) :]
        i = nxt
    return b


def decode_wrist_jpeg(raw: np.ndarray | bytes) -> np.ndarray | None:
    """Raw Trinet MJPEG buffer (``CAP_PROP_CONVERT_RGB = 0``) -> BGR frame, or None.
    ~15 ms for 1920x1080 on the Orin CPU."""
    import cv2  # lazy

    data = raw if isinstance(raw, bytes) else raw.tobytes()
    fixed = strip_trinet_sei(data)
    img = cv2.imdecode(np.frombuffer(fixed, np.uint8), cv2.IMREAD_COLOR)
    return img if img is not None else None


def build_wrist_preprocess(
    cfg: WristCameraConfig, frame_size: tuple[int, int] | None = None
) -> Callable[[np.ndarray], np.ndarray] | None:
    """Dewarp for the Critic worker, or None (raw frames).

    - ``dewarp_enabled = False``: None.
    - ``cfg.calib_path`` missing (OQ-9 not done yet): WARNING, None (the Critic sees raw
      fisheye frames; accuracy may drop).
    - calibration size != the capture size: :class:`CalibrationError` (never dewarp with
      the wrong maps).
    """
    if not cfg.dewarp_enabled:
        log.info("Trinet dewarp disabled (WristCameraConfig.dewarp_enabled = False)")
        return None
    if not cfg.calib_path.is_file():
        log.warning(
            "No Trinet fisheye calibration at %s (OQ-9: tools/capture_checkerboard.py + "
            "tools/calibrate_fisheye.py): the Critic gets RAW (not dewarped) frames",
            cfg.calib_path,
        )
        return None
    from lbyr.perception.dewarp import Dewarper, FisheyeCalibration

    calib = FisheyeCalibration.load(cfg.calib_path)
    size = frame_size or (
        (cfg.width, cfg.height) if cfg.width is not None and cfg.height is not None else None
    )
    if size is not None and tuple(calib.image_size) != tuple(size):
        raise CalibrationError(
            f"{cfg.calib_path} is for {calib.image_size[0]}x{calib.image_size[1]} but the Trinet "
            f"runs at {size[0]}x{size[1]}: recalibrate or change WristCameraConfig"
        )
    return Dewarper(calib, cfg.dewarp_balance, cfg.crop_xywh)


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
