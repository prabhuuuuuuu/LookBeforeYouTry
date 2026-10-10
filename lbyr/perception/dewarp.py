"""Trinet fisheye dewarping (bible 5.2): calibrate once, precompute maps, remap per frame.

Cost: map precompute once at startup; per-frame ``cv2.remap`` ~2-5 ms (bible 5.2).
``cv2`` is imported lazily (JetPack build on the Jetson, opencv-python on a laptop).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from lbyr.errors import CalibrationError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FisheyeCalibration:
    K: np.ndarray  # 3x3
    D: np.ndarray  # (4,) fisheye distortion k1..k4
    image_size: tuple[int, int]  # (width, height)
    rms: float | None = None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model": "opencv_fisheye",
            "K": self.K.tolist(),
            "D": self.D.reshape(-1).tolist(),
            "image_size": list(self.image_size),
            "rms": self.rms,
        }
        path.write_text(yaml.safe_dump(data), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> FisheyeCalibration:
        if not path.exists():
            raise CalibrationError(
                f"fisheye calibration not found: {path} "
                "(OQ-9: run tools/capture_checkerboard.py then tools/calibrate_fisheye.py)"
            )
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        try:
            K = np.asarray(data["K"], dtype=np.float64)
            D = np.asarray(data["D"], dtype=np.float64).reshape(-1)
            w, h = (int(v) for v in data["image_size"])
        except (KeyError, TypeError, ValueError) as e:
            raise CalibrationError(f"malformed calibration file {path}: {e}") from e
        if K.shape != (3, 3) or D.shape != (4,):
            raise CalibrationError(f"calibration {path}: K must be 3x3 and D length 4")
        return cls(K, D, (w, h), data.get("rms"))


def synthetic_calibration(width: int, height: int) -> FisheyeCalibration:
    """Plausible fake fisheye intrinsics for tests and mock mode (NOT the Trinet's)."""
    f = width / 3.0
    K = np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]], dtype=np.float64)
    D = np.array([0.05, -0.01, 0.002, -0.0005], dtype=np.float64)
    return FisheyeCalibration(K, D, (width, height))


class Dewarper:
    """Precomputes undistort maps once; ``__call__`` remaps (and optionally crops) a frame."""

    def __init__(
        self,
        calib: FisheyeCalibration,
        balance: float = 0.0,
        crop_xywh: tuple[int, int, int, int] | None = None,
    ) -> None:
        import cv2  # lazy

        self._cv2: Any = cv2
        self.calib = calib
        self.crop_xywh = crop_xywh
        size = calib.image_size
        D = calib.D.reshape(4, 1)
        self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            calib.K, D, size, np.eye(3), balance=balance
        )
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            calib.K, D, np.eye(3), self.new_K, size, cv2.CV_16SC2
        )
        log.info("Dewarp maps precomputed for %dx%d (balance=%.2f)", *size, balance)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        w, h = self.calib.image_size
        if frame.shape[1] != w or frame.shape[0] != h:
            raise CalibrationError(
                f"frame {frame.shape[1]}x{frame.shape[0]} does not match calibration {w}x{h}"
            )
        out = self._cv2.remap(frame, self.map1, self.map2, interpolation=self._cv2.INTER_LINEAR)
        if self.crop_xywh is not None:
            x, y, cw, ch = self.crop_xywh
            out = out[y : y + ch, x : x + cw]
        return out


def calibrate_fisheye(
    object_points: Sequence[np.ndarray],
    image_points: Sequence[np.ndarray],
    image_size: tuple[int, int],
) -> FisheyeCalibration:
    """Run cv2.fisheye.calibrate on detected checkerboard corners (used by tools/)."""
    import cv2  # lazy

    obj = [np.asarray(o, dtype=np.float64).reshape(-1, 1, 3) for o in object_points]
    img = [np.asarray(i, dtype=np.float64).reshape(-1, 1, 2) for i in image_points]
    if len(obj) < 3:
        raise CalibrationError("need at least 3 checkerboard views (10-20 recommended)")
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_CHECK_COND
        | cv2.fisheye.CALIB_FIX_SKEW
    )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    rms, K_out, D_out, _, _ = cv2.fisheye.calibrate(
        obj, img, image_size, K, D, None, None, flags, criteria
    )
    return FisheyeCalibration(
        np.asarray(K_out, np.float64), np.asarray(D_out, np.float64).reshape(-1), image_size, rms
    )
