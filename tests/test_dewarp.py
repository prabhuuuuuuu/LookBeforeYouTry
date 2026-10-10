from pathlib import Path

import numpy as np
import pytest

from lbyr.errors import CalibrationError
from lbyr.perception.dewarp import Dewarper, FisheyeCalibration, synthetic_calibration

pytest.importorskip("cv2")


def test_precompute_and_remap_synthetic() -> None:
    calib = synthetic_calibration(320, 240)
    dw = Dewarper(calib, balance=0.0)
    assert dw.map1.shape[:2] == (240, 320)
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
    out = dw(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    # Optical centre should map (nearly) to itself.
    centre = dw(np.pad(np.full((1, 1, 3), 255, np.uint8), ((119, 120), (159, 160), (0, 0))))
    ys, xs = np.nonzero(centre[..., 0] > 0)
    assert abs(xs.mean() - 159.5) < 3 and abs(ys.mean() - 119.5) < 3


def test_crop() -> None:
    dw = Dewarper(synthetic_calibration(320, 240), crop_xywh=(10, 20, 100, 50))
    assert dw(np.zeros((240, 320, 3), np.uint8)).shape == (50, 100, 3)


def test_size_mismatch_raises() -> None:
    dw = Dewarper(synthetic_calibration(320, 240))
    with pytest.raises(CalibrationError):
        dw(np.zeros((480, 640, 3), np.uint8))


def test_calibrate_fisheye_recovers_synthetic_intrinsics() -> None:
    import cv2

    from lbyr.perception.dewarp import calibrate_fisheye

    truth = synthetic_calibration(640, 480)
    cols, rows, sq = 9, 6, 25.0
    board = np.zeros((cols * rows, 3))
    board[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq
    board -= board.mean(axis=0)
    obj, img = [], []
    rng = np.random.default_rng(3)
    for _ in range(15):
        rvec = rng.uniform(-0.5, 0.5, size=3)
        tvec = np.array([rng.uniform(-60, 60), rng.uniform(-40, 40), rng.uniform(250, 400)])
        pts, _ = cv2.fisheye.projectPoints(
            board.reshape(-1, 1, 3), rvec, tvec, truth.K, truth.D.reshape(4, 1)
        )
        obj.append(board)
        img.append(pts.reshape(-1, 2))
    calib = calibrate_fisheye(obj, img, (640, 480))
    assert calib.rms is not None and calib.rms < 0.1
    np.testing.assert_allclose(calib.K, truth.K, rtol=0.02, atol=2.0)


def test_calibration_yaml_round_trip(tmp_path: Path) -> None:
    calib = synthetic_calibration(640, 480)
    path = tmp_path / "trinet.yaml"
    calib.save(path)
    loaded = FisheyeCalibration.load(path)
    np.testing.assert_allclose(loaded.K, calib.K)
    np.testing.assert_allclose(loaded.D, calib.D)
    assert loaded.image_size == (640, 480)
    with pytest.raises(CalibrationError, match="OQ-9"):
        FisheyeCalibration.load(tmp_path / "nope.yaml")
