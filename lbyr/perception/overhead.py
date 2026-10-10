"""Overhead Intel RealSense D435 capture (bible 5.1) and a fake for mock mode.

The D435 is the ONLY camera used for metric control. Depth is converted to mm at
this boundary. ``pyrealsense2`` is imported lazily (JetPack/source build, not pip).
"""

from __future__ import annotations

import logging
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
        self._depth_scale_mm = profile.get_device().first_depth_sensor().get_depth_scale() * 1000
        self._align = rs.align(rs.stream.color)
        vs = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self._intr = Intrinsics(vs.fx, vs.fy, vs.ppx, vs.ppy, vs.width, vs.height)
        # TODO: lock exposure / white balance via color sensor options (bible 5.1).
        self._thread = CaptureThread("overhead-cam", self._grab, self.buffer)
        self._thread.start()
        log.info("D435 started %dx%d@%d, depth scale %.4f mm", w, h, fps, self._depth_scale_mm)

    def _grab(self) -> RGBDFrame | None:
        assert self._intr is not None
        frames = self._align.process(self._pipeline.wait_for_frames(1000))
        c, d = frames.get_color_frame(), frames.get_depth_frame()
        if not c or not d:
            return None
        depth = np.asanyarray(d.get_data()).astype(np.float32) * self._depth_scale_mm
        return RGBDFrame(np.asanyarray(c.get_data()).copy(), depth, self._intr)

    def stop(self) -> None:
        if self._thread is not None:
            self._thread.stop()
        if self._pipeline is not None:
            self._pipeline.stop()


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
    """Publishes a rendered FakeScene at ~30 Hz on a capture thread."""

    def __init__(self, scene: FakeScene | None = None, fps: float = 30.0) -> None:
        self.scene = scene or FakeScene()
        self.buffer: LatestFrameBuffer[RGBDFrame] = LatestFrameBuffer()
        self._frame = self.scene.render()
        self._thread = CaptureThread("overhead-fake", lambda: self._frame, self.buffer, 1.0 / fps)

    def start(self) -> None:
        self.buffer.put(self._frame)
        self._thread.start()

    def stop(self) -> None:
        self._thread.stop()
