"""Trinet wrist camera config + dewarp-if-calibrated preprocess (no camera needed)."""

import logging
from pathlib import Path

import numpy as np
import pytest

from lbyr.config import WristCameraConfig
from lbyr.errors import CalibrationError, HardwareNotConfirmedError
from lbyr.perception.wrist import TrinetWristCamera, build_wrist_preprocess, wrist_device


def test_device_path_wins_over_index_and_missing_device_is_unconfirmed() -> None:
    assert wrist_device(WristCameraConfig(device_index=2)) == 2
    assert wrist_device(WristCameraConfig(device_index=2, device_path="/dev/v4l/x")) == "/dev/v4l/x"
    with pytest.raises(HardwareNotConfirmedError, match="OQ-7"):
        TrinetWristCamera(WristCameraConfig(width=640, height=480, fps=30))
    with pytest.raises(HardwareNotConfirmedError, match="OQ-8"):
        TrinetWristCamera(WristCameraConfig(device_index=0, width=None))


def test_no_calibration_warns_and_uses_raw(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = WristCameraConfig(width=640, height=480, calib_path=tmp_path / "none.yaml")
    with caplog.at_level(logging.WARNING):
        assert build_wrist_preprocess(cfg) is None
    assert "RAW" in caplog.text
    assert build_wrist_preprocess(WristCameraConfig(dewarp_enabled=False)) is None


def test_calibration_is_used_and_size_mismatch_refused(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    from lbyr.perception.dewarp import synthetic_calibration

    path = tmp_path / "trinet_fisheye.yaml"
    synthetic_calibration(640, 480).save(path)
    pre = build_wrist_preprocess(WristCameraConfig(width=640, height=480, calib_path=path))
    assert pre is not None
    assert pre(np.zeros((480, 640, 3), np.uint8)).shape == (480, 640, 3)
    with pytest.raises(CalibrationError, match="1280x720"):
        build_wrist_preprocess(WristCameraConfig(width=1280, height=720, calib_path=path))
