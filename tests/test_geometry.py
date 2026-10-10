from pathlib import Path

import numpy as np
import pytest

from lbyr.errors import CalibrationError
from lbyr.perception import geometry as g
from lbyr.perception.overhead import FakeScene

INTR = g.Intrinsics(fx=600.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)


def test_unproject_hand_computed() -> None:
    # u=380 -> x = (380-320)*1000/600 = 100 ; v=140 -> y = (140-240)*1000/500 = -200
    p = g.unproject(380.0, 140.0, 1000.0, INTR)
    np.testing.assert_allclose(p, [100.0, -200.0, 1000.0])
    u, v = g.project(p, INTR)
    assert (u, v) == pytest.approx((380.0, 140.0))


def test_median_depth_ignores_invalid() -> None:
    depth = np.full((10, 10), 500.0, dtype=np.float32)
    depth[5, 5] = 0.0  # invalid
    depth[4, 4] = 9000.0  # outlier
    assert g.median_depth(depth, 5, 5, window=3) == 500.0
    assert g.median_depth(np.zeros((10, 10)), 5, 5) is None
    assert g.median_depth(depth, 50, 5) is None


def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def test_umeyama_recovers_known_transform() -> None:
    rng = np.random.default_rng(0)
    R = _rot(np.array([0.3, -0.5, 0.8]), 0.9)
    t = np.array([120.0, -45.0, 700.0])
    src = rng.uniform(-200, 200, size=(12, 3))  # 12 tip-touch points (bible 5.1)
    dst = src @ R.T + t
    R_est, t_est, s = g.umeyama(src, dst)
    np.testing.assert_allclose(R_est, R, atol=1e-9)
    np.testing.assert_allclose(t_est, t, atol=1e-7)
    assert s == 1.0
    T = g.rigid_transform(src, dst + rng.normal(0, 0.5, size=dst.shape))
    assert g.rms_error_mm(T, src, dst) < 1.0


def test_umeyama_with_scale() -> None:
    rng = np.random.default_rng(1)
    src = rng.uniform(-1, 1, size=(20, 3))
    R = _rot(np.array([1.0, 0.0, 0.0]), 0.4)
    dst = 2.5 * src @ R.T + np.array([1.0, 2.0, 3.0])
    _, _, s = g.umeyama(src, dst, with_scale=True)
    assert s == pytest.approx(2.5)


def test_transform_yaml_round_trip(tmp_path: Path) -> None:
    T = g.make_transform(_rot(np.array([0, 0, 1.0]), 0.3), np.array([1.0, 2.0, 3.0]))
    path = tmp_path / "base_T_cam.yaml"
    g.save_transform(path, T, rms_mm=2.1)
    np.testing.assert_allclose(g.load_transform(path), T)
    with pytest.raises(CalibrationError):
        g.load_transform(tmp_path / "missing.yaml")


def test_fake_scene_locates_objects() -> None:
    scene = FakeScene()
    frame = scene.render()
    for det, obj in zip(scene.detections(), scene.objects, strict=True):
        u, v = det.centroid
        p = g.locate_in_base(u, v, frame.depth_mm, frame.intrinsics, scene.base_T_cam)
        assert p is not None
        np.testing.assert_allclose(p[:2], obj.base_xyz_mm[:2], atol=2.0)
        assert p[2] == pytest.approx(obj.height_mm, abs=1.0)
