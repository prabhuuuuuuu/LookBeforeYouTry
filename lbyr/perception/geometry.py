"""Metric geometry for the overhead D435 (bible 5.1, 12). Pure numpy, units = mm.

base_T_obj = base_T_cam . cam_T_obj
cam_T_obj  <- detector centroid + median-filtered depth + pinhole unprojection
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from lbyr.errors import CalibrationError


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def unproject(u: float, v: float, depth_mm: float, intr: Intrinsics) -> np.ndarray:
    """Pixel (u, v) + depth along the optical axis -> 3D point in the camera frame (mm)."""
    x = (u - intr.cx) * depth_mm / intr.fx
    y = (v - intr.cy) * depth_mm / intr.fy
    return np.array([x, y, depth_mm], dtype=float)


def project(p_cam: np.ndarray, intr: Intrinsics) -> tuple[float, float]:
    """Camera-frame point (mm) -> pixel (u, v)."""
    x, y, z = (float(c) for c in p_cam)
    return intr.fx * x / z + intr.cx, intr.fy * y / z + intr.cy


def median_depth(depth_mm: np.ndarray, u: float, v: float, window: int = 5) -> float | None:
    """Median of valid (>0) depth values in a window centred at (u, v). None if no valid px."""
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    h, w = depth_mm.shape[:2]
    ui, vi, r = int(round(u)), int(round(v)), window // 2
    if not (0 <= ui < w and 0 <= vi < h):
        return None
    patch = depth_mm[max(0, vi - r) : vi + r + 1, max(0, ui - r) : ui + r + 1]
    valid = patch[patch > 0]
    return float(np.median(valid)) if valid.size else None


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def apply_transform(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to a (3,) or (N,3) array of points."""
    p = np.asarray(p, dtype=float)
    return p @ T[:3, :3].T + T[:3, 3]


def umeyama(
    src: np.ndarray, dst: np.ndarray, with_scale: bool = False
) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares (R, t, s) with dst ~= s * R @ src + t (Umeyama 1991).

    For hand-eye: src = points in camera frame, dst = same points in base frame
    (12 tip-touch points, bible 5.1) -> base_T_cam.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or src.shape[0] < 3:
        raise ValueError("src and dst must both be (N>=3, 3)")
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / src.shape[0]
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / xs.var(axis=0).sum()) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return R, t, s


def rigid_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """4x4 rigid transform mapping src -> dst (no scale)."""
    R, t, _ = umeyama(src, dst, with_scale=False)
    return make_transform(R, t)


def rms_error_mm(T: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    err = apply_transform(T, src) - np.asarray(dst, dtype=float)
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


def locate_in_base(
    u: float,
    v: float,
    depth_mm: np.ndarray,
    intr: Intrinsics,
    base_T_cam: np.ndarray,
    window: int = 5,
) -> np.ndarray | None:
    """Full chain: centroid -> median depth -> unproject -> base frame (mm)."""
    d = median_depth(depth_mm, u, v, window)
    if d is None:
        return None
    return apply_transform(base_T_cam, unproject(u, v, d, intr))


def save_transform(path: Path, T: np.ndarray, rms_mm: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"base_T_cam": np.asarray(T).tolist(), "rms_mm": rms_mm, "units": "mm"}
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def load_transform(path: Path) -> np.ndarray:
    if not path.exists():
        raise CalibrationError(f"hand-eye calibration not found: {path} (run calibration first)")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    T = np.asarray(data.get("base_T_cam"), dtype=float)
    if T.shape != (4, 4) or not np.allclose(T[3], [0, 0, 0, 1]):
        raise CalibrationError(f"invalid base_T_cam in {path}")
    return T
