"""Block detector for the live demo: ONE object on a white table, no YOLO.

Pipeline (D435 color + aligned depth, 640x480):

1. "object" mask = not white: Lab chroma / darkness (or HSV saturation / value), thresholds
   from :class:`~lbyr.config.BlockDetectorConfig`;
2. AND the workspace mask (polygon in overhead pixels; default = frame minus a border);
3. morphological open + close, external contours, area filter, keep the largest;
4. centroid (contour moments), bbox, min-area-rect yaw (long side, deg, [-90, 90));
5. median depth around the centroid -> camera-frame 3D point via the D435 intrinsics.

Output: the shared :class:`~lbyr.perception.detector.Detection` format (id, class
``wood_block``, bbox, centroid, yaw, xyz_cam, zone). Cost: CPU only, 0 VRAM; measured on
the Orin NX (640x480, jetson_clocks, 2026-10-11): ~4-6 ms (Lab) / ~2.5 ms (HSV) per frame
(``last_ms``). It runs once per PERCEIVE on the FSM thread, never on the motion thread.
``cv2`` is imported lazily.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from lbyr.config import BlockDetectorConfig
from lbyr.perception.detector import Detection
from lbyr.perception.geometry import median_depth, unproject
from lbyr.perception.overhead import RGBDFrame

log = logging.getLogger(__name__)

ZoneLookup = Callable[[float, float], str | None]


def long_side_yaw_deg(box_pts: np.ndarray) -> float:
    """Angle of the longer rectangle side from the image +u axis (toward +v), [-90, 90)."""
    p0, p1, p2 = (np.asarray(box_pts[i], dtype=float) for i in range(3))
    e1, e2 = p1 - p0, p2 - p1
    e = e1 if np.hypot(*e1) >= np.hypot(*e2) else e2
    a = math.degrees(math.atan2(e[1], e[0]))
    return (a + 90.0) % 180.0 - 90.0


class BlockDetector:
    """Largest non-white blob inside the workspace -> one :class:`Detection`."""

    def __init__(
        self,
        cfg: BlockDetectorConfig,
        workspace_px: Sequence[tuple[float, float]] | None = None,
        zone_of: ZoneLookup | None = None,
    ) -> None:
        import cv2  # lazy: JetPack build on the Jetson, opencv-python on a laptop

        cfg.validate()
        self._cv2: Any = cv2
        self.cfg = cfg
        poly = cfg.workspace_polygon_px or workspace_px
        self.workspace_px = tuple(poly) if poly else None
        if self.workspace_px is not None and len(self.workspace_px) < 3:
            raise ValueError("workspace polygon needs >= 3 points")
        self.zone_of = zone_of
        self._ws_cache: tuple[tuple[int, int], np.ndarray] | None = None
        k = max(1, cfg.morph_kernel_px)
        self._kernel = np.ones((k, k), np.uint8)
        self.last_ms = 0.0
        self.last_areas: list[float] = []  # every contour area of the last frame (tuning)
        self._warm_up()
        if self.workspace_px is None:
            log.warning(
                "BlockDetector: no workspace polygon (tools/define_zones.py --workspace): "
                "using the whole frame minus a %d px border",
                cfg.border_px,
            )

    def _warm_up(self, shape: tuple[int, int] = (480, 640)) -> None:
        """The first OpenCV call is ~200 ms (lazy init); pay it at startup, not in PERCEIVE."""
        blank = np.full((*shape, 3), 255, np.uint8)
        self._cv2.findContours(
            self.mask(blank), self._cv2.RETR_EXTERNAL, self._cv2.CHAIN_APPROX_SIMPLE
        )
        self._ws_cache = None

    # ------------------------------------------------------------------ masks
    def object_mask(self, color_bgr: np.ndarray) -> np.ndarray:
        """uint8 mask, 255 where the pixel is NOT white table (before the workspace mask)."""
        cv2, c = self._cv2, self.cfg
        if c.color_space == "lab":
            lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)
            L = lab[..., 0]
            a = lab[..., 1].astype(np.int16) - 128
            b = lab[..., 2].astype(np.int16) - 128
            chroma2 = a.astype(np.int32) ** 2 + b.astype(np.int32) ** 2
            m = (chroma2 >= c.lab_chroma_min**2) | (L <= c.lab_l_max)
        else:
            hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
            m = (hsv[..., 1] >= c.hsv_s_min) | (hsv[..., 2] <= c.hsv_v_max)
        return m.astype(np.uint8) * 255

    def workspace_mask(self, shape: tuple[int, int]) -> np.ndarray:
        if self._ws_cache is not None and self._ws_cache[0] == shape:
            return self._ws_cache[1]
        h, w = shape
        ws = np.zeros((h, w), np.uint8)
        if self.workspace_px is not None:
            pts = np.round(np.asarray(self.workspace_px, dtype=float)).astype(np.int32)
            self._cv2.fillPoly(ws, [pts.reshape(-1, 1, 2)], 255)
        else:
            b = self.cfg.border_px
            ws[b : h - b, b : w - b] = 255
        self._ws_cache = ((h, w), ws)
        return ws

    def mask(self, color_bgr: np.ndarray) -> np.ndarray:
        """Final cleaned mask (object AND workspace, opened + closed)."""
        cv2 = self._cv2
        h, w = color_bgr.shape[:2]
        m = cv2.bitwise_and(self.object_mask(color_bgr), self.workspace_mask((h, w)))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self._kernel)
        return np.asarray(cv2.morphologyEx(m, cv2.MORPH_CLOSE, self._kernel))

    # ------------------------------------------------------------------ detect
    def detect(self, frame: RGBDFrame | None) -> list[Detection]:
        if frame is None:
            return []
        t0 = time.perf_counter()
        try:
            return self._detect(frame)
        finally:
            self.last_ms = (time.perf_counter() - t0) * 1000.0

    def _detect(self, frame: RGBDFrame) -> list[Detection]:
        cv2, c = self._cv2, self.cfg
        contours, _ = cv2.findContours(
            self.mask(frame.color), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        areas = [float(cv2.contourArea(k)) for k in contours]
        self.last_areas = sorted(areas, reverse=True)
        ok = [(a, k) for a, k in zip(areas, contours, strict=True) if c.min_area_px <= a]
        ok = [(a, k) for a, k in ok if a <= c.max_area_px]
        if not ok:
            log.debug("BlockDetector: no blob in [%g, %g] px (areas %s)", *self._range(), areas[:5])
            return []
        area, cnt = max(ok, key=lambda t: t[0])
        m = cv2.moments(cnt)
        if m["m00"] <= 0:
            return []
        u, v = m["m10"] / m["m00"], m["m01"] / m["m00"]
        x, y, w, h = cv2.boundingRect(cnt)
        rect = cv2.minAreaRect(cnt)
        yaw = long_side_yaw_deg(cv2.boxPoints(rect))
        rw, rh = rect[1]
        fill = area / (rw * rh) if rw * rh > 0 else 0.0  # 1.0 = perfect rectangle
        xyz: tuple[float, float, float] | None = None
        d = median_depth(frame.depth_mm, u, v, c.depth_window_px)
        if d is not None:
            p = unproject(u, v, d, frame.intrinsics)
            xyz = (float(p[0]), float(p[1]), float(p[2]))
        zone = self.zone_of(u, v) if self.zone_of is not None else None
        return [
            Detection(
                id=c.object_id,
                label=c.label,
                score=round(min(1.0, fill), 3),
                bbox_xyxy=(float(x), float(y), float(x + w), float(y + h)),
                centroid_px=(float(u), float(v)),
                yaw_deg=round(yaw, 1),
                xyz_cam_mm=xyz,
                zone=zone,
                area_px=area,
            )
        ]

    def _range(self) -> tuple[float, float]:
        return self.cfg.min_area_px, self.cfg.max_area_px
