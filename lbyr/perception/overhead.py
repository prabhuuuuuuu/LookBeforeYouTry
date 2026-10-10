"""Overhead Intel RealSense D435 capture (bible 5.1) and a fake for mock mode.

The D435 is the ONLY camera used for metric control. Depth is converted to mm at
this boundary. ``pyrealsense2`` is imported lazily (JetPack/source build, not pip).

Real camera: 640x480@30 color (BGR) + depth aligned to color, published by a capture
thread into a latest-frame buffer (readers never wait). Exposure and white balance are
locked after an auto-settle (:func:`lock_color_controls`). Frame metadata (the auto values)
is not readable on this Jetson kernel, so the lock MATCHES the settled image instead:
exposure by bisection to the auto mean brightness, white balance by bisection to a neutral
(B = R) bright region (the white table). Cost: ~4 s once at start; per frame ~1 ms copy
(+ the librealsense align); 0 VRAM; ~2 MB per buffered frame.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lbyr.config import OverheadCameraConfig
from lbyr.perception.buffers import CaptureThread, LatestFrameBuffer
from lbyr.perception.detector import Detection, FakeDetector
from lbyr.perception.geometry import Intrinsics, apply_transform, make_transform, project

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RGBDFrame:
    color: np.ndarray  # HxWx3 uint8 BGR
    depth_mm: np.ndarray  # HxW float32, 0 = invalid, aligned to color
    intrinsics: Intrinsics


@dataclass(frozen=True)
class ColorLock:
    exposure: float
    gain: float
    white_balance_k: float
    auto_mean: float  # mean brightness under auto exposure (the target)
    locked_mean: float  # mean brightness after the lock
    source: str  # "config" | "matched" (per control, e.g. "exposure=matched, wb=config")


def _luma(img: np.ndarray) -> float:
    return float(img.mean())


def _neutral_error(img: np.ndarray) -> float:
    """mean(B) - mean(R) over the brightest half of the image (the white table)."""
    y = img.mean(axis=2)
    bright = y >= np.median(y)
    return float(img[..., 0][bright].mean() - img[..., 2][bright].mean())


def _bisect(
    f: Callable[[float], float], lo: float, hi: float, step: float, max_iter: int = 12
) -> float:
    """Root of a monotonic ``f`` on [lo, hi] (either direction); the better end if none."""
    f_lo, f_hi = f(lo), f(hi)
    if f_lo == 0.0 or f_hi == 0.0 or (f_lo > 0) == (f_hi > 0):
        return lo if abs(f_lo) <= abs(f_hi) else hi
    best = (abs(f_lo), lo) if abs(f_lo) <= abs(f_hi) else (abs(f_hi), hi)
    for _ in range(max_iter):
        if hi - lo <= step:
            break
        mid = round((lo + hi) / 2.0 / step) * step
        f_mid = f(mid)
        best = min(best, (abs(f_mid), mid))
        if (f_mid > 0) == (f_lo > 0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return best[1]


def lock_color_controls(
    sensor: Any, option: Any, grab: Callable[[], np.ndarray], cfg: OverheadCameraConfig
) -> ColorLock:
    """Auto-settle, then lock exposure and white balance (bible 5.1).

    ``sensor``: librealsense color sensor (``get_option`` / ``set_option``); ``option``:
    ``rs.option``; ``grab``: returns the next color frame (BGR). Config values win; the
    others are matched to the settled auto image (see module docstring). Blocking: startup
    only, before the capture thread runs.
    """

    def settled(n: int = 3) -> np.ndarray:
        img = grab()
        for _ in range(n - 1):  # a new option value needs a couple of frames to apply
            img = grab()
        return img

    sensor.set_option(option.enable_auto_exposure, 1)
    sensor.set_option(option.enable_auto_white_balance, 1)
    t_end = time.perf_counter() + cfg.settle_s
    img = grab()
    while time.perf_counter() < t_end:
        img = grab()
    auto_mean = float(np.mean([_luma(img), _luma(grab()), _luma(grab())]))
    sensor.set_option(option.enable_auto_exposure, 0)
    if cfg.gain is not None:
        sensor.set_option(option.gain, float(cfg.gain))
    gain = float(sensor.get_option(option.gain))
    srcs = []
    if cfg.exposure is not None:
        exposure = float(cfg.exposure)
        srcs.append("exposure=config")
    else:

        def bright_err(e: float) -> float:
            sensor.set_option(option.exposure, float(e))
            return _luma(settled()) - auto_mean

        exposure = _bisect(bright_err, 1.0, float(cfg.max_exposure), 1.0)
        srcs.append("exposure=matched")
    sensor.set_option(option.exposure, exposure)
    sensor.set_option(option.enable_auto_white_balance, 0)
    if cfg.white_balance_k is not None:
        wb = float(cfg.white_balance_k)
        srcs.append("wb=config")
    else:

        def wb_err(k: float) -> float:
            sensor.set_option(option.white_balance, float(k))
            return _neutral_error(settled())

        wb = _bisect(wb_err, 2800.0, 6500.0, 10.0)
        srcs.append("wb=matched")
    sensor.set_option(option.white_balance, wb)
    locked_mean = _luma(settled())
    return ColorLock(exposure, gain, wb, auto_mean, locked_mean, ", ".join(srcs))


class RealSenseOverheadCamera:
    """D435 aligned color+depth stream on a capture thread. Lock exposure/WB (bible 5.1)."""

    def __init__(self, cfg: OverheadCameraConfig) -> None:
        self._cfg = cfg
        self.buffer: LatestFrameBuffer[RGBDFrame] = LatestFrameBuffer()
        self._pipeline: Any = None
        self._align: Any = None
        self._depth_scale_mm = 1.0
        self._intr: Intrinsics | None = None
        self._thread: CaptureThread[RGBDFrame] | None = None
        self.color_lock: ColorLock | None = None

    @property
    def fps(self) -> float:
        return self._thread.fps if self._thread is not None else 0.0

    @property
    def intrinsics(self) -> Intrinsics | None:
        return self._intr

    def start(self) -> None:
        import pyrealsense2 as rs  # lazy: librealsense source build on Jetson

        cfg = rs.config()
        if self._cfg.device_serial:
            cfg.enable_device(self._cfg.device_serial)
        w, h, fps = self._cfg.width, self._cfg.height, self._cfg.fps
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(cfg)
        try:
            dev = profile.get_device()
            self._depth_scale_mm = dev.first_depth_sensor().get_depth_scale() * 1000
            self._align = rs.align(rs.stream.color)
            vs = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self._intr = Intrinsics(vs.fx, vs.fy, vs.ppx, vs.ppy, vs.width, vs.height)
            if self._cfg.lock_color:
                self.color_lock = lock_color_controls(
                    dev.first_color_sensor(), rs.option, self._grab_color, self._cfg
                )
                lk = self.color_lock
                log.info(
                    "D435 color locked (%s): exposure %.0f, gain %.0f, white balance %.0f K; "
                    "mean brightness auto %.0f -> locked %.0f. To reuse: "
                    "OverheadCameraConfig(exposure=%.0f, gain=%.0f, white_balance_k=%.0f)",
                    lk.source,
                    lk.exposure,
                    lk.gain,
                    lk.white_balance_k,
                    lk.auto_mean,
                    lk.locked_mean,
                    lk.exposure,
                    lk.gain,
                    lk.white_balance_k,
                )
            else:
                log.warning("D435 exposure / white balance NOT locked (lock_color = False)")
        except Exception:
            self._pipeline.stop()
            self._pipeline = None
            raise
        self._thread = CaptureThread("overhead-cam", self._grab, self.buffer)
        self._thread.start()
        log.info("D435 started %dx%d@%d, depth scale %.4f mm", w, h, fps, self._depth_scale_mm)

    def _grab_color(self) -> np.ndarray:
        c = self._pipeline.wait_for_frames(1000).get_color_frame()
        return np.asanyarray(c.get_data())

    def _grab(self) -> RGBDFrame | None:
        assert self._intr is not None
        frames = self._align.process(self._pipeline.wait_for_frames(1000))
        c, d = frames.get_color_frame(), frames.get_depth_frame()
        if not c or not d:
            return None
        depth = np.asanyarray(d.get_data()).astype(np.float32) * self._depth_scale_mm
        return RGBDFrame(np.asanyarray(c.get_data()).copy(), depth, self._intr)

    def stop(self) -> None:
        """Join the capture thread (it can sit in wait_for_frames for up to 1 s), THEN stop
        the pipeline. Exiting with a live pipeline aborts the process in librealsense's
        thread destructors ("terminate called without an active exception")."""
        if self._thread is not None:
            self._thread.stop(timeout=2.5)
            if self._thread.is_alive():
                log.warning("overhead-cam thread did not stop within 2.5 s")
            self._thread = None
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            finally:
                self._pipeline = None
                self._align = None
            log.info("D435 pipeline stopped")


# --------------------------------------------------------------------------- fakes
@dataclass(frozen=True)
class FakeObject:
    id: int
    label: str
    base_xyz_mm: tuple[float, float, float]  # centre of the object's base on the mat
    size_mm: float  # cube edge (or patch size for flat zones)
    height_mm: float


def _default_objects() -> list[FakeObject]:
    return [
        FakeObject(2, "acrylic_block", (220.0, 60.0, 0.0), 30.0, 30.0),
        FakeObject(5, "place_zone", (220.0, -100.0, 0.0), 60.0, 2.0),
    ]


def _default_base_T_cam() -> np.ndarray:
    # FAKE: camera 700 mm above the mat looking straight down (real rig is tilted ~50 deg).
    R = np.diag([1.0, -1.0, -1.0])
    return make_transform(R, np.array([200.0, 0.0, 700.0]))


@dataclass
class FakeScene:
    """A synthetic, self-consistent scene: objects, camera pose and intrinsics."""

    objects: list[FakeObject] = field(default_factory=_default_objects)
    base_T_cam: np.ndarray = field(default_factory=_default_base_T_cam)
    intrinsics: Intrinsics = Intrinsics(600.0, 600.0, 320.0, 240.0, 640, 480)

    def _cam_T_base(self) -> np.ndarray:
        return np.linalg.inv(self.base_T_cam)

    def bbox(self, obj: FakeObject) -> tuple[float, float, float, float]:
        half = obj.size_mm / 2.0
        x, y, _ = obj.base_xyz_mm
        corners = [(x - half, y - half), (x + half, y + half)]
        pix = [
            project(
                apply_transform(self._cam_T_base(), np.array([cx, cy, obj.height_mm])),
                self.intrinsics,
            )
            for cx, cy in corners
        ]
        us, vs = [p[0] for p in pix], [p[1] for p in pix]
        return min(us), min(vs), max(us), max(vs)

    def render(self) -> RGBDFrame:
        intr = self.intrinsics
        color = np.full((intr.height, intr.width, 3), 90, dtype=np.uint8)
        table_depth = float(self._cam_T_base()[2, 3])  # straight-down camera: constant
        depth = np.full((intr.height, intr.width), table_depth, dtype=np.float32)
        for obj in self.objects:
            x1, y1, x2, y2 = (int(round(c)) for c in self.bbox(obj))
            color[y1:y2, x1:x2] = (40 + 30 * obj.id, 160, 200)
            depth[y1:y2, x1:x2] = table_depth - obj.height_mm
        return RGBDFrame(color, depth, intr)

    def detections(self) -> list[Detection]:
        return [Detection(o.id, o.label, 0.99, self.bbox(o)) for o in self.objects]

    def detector(self) -> FakeDetector:
        return FakeDetector(self.detections())


class FakeOverheadCamera:
    """Publishes a rendered FakeScene (or a fixed ``frame``) at ~30 Hz on a capture thread."""

    def __init__(
        self, scene: FakeScene | None = None, fps: float = 30.0, frame: RGBDFrame | None = None
    ) -> None:
        self.scene = scene or FakeScene()
        self.buffer: LatestFrameBuffer[RGBDFrame] = LatestFrameBuffer()
        self._frame = frame if frame is not None else self.scene.render()
        self._thread = CaptureThread("overhead-fake", lambda: self._frame, self.buffer, 1.0 / fps)

    def start(self) -> None:
        self.buffer.put(self._frame)
        self._thread.start()

    def stop(self) -> None:
        self._thread.stop()
