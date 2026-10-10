"""D435 exposure / white-balance lock logic with a simulated color sensor (no camera)."""

from types import SimpleNamespace

import numpy as np
import pytest

from lbyr.config import OverheadCameraConfig
from lbyr.perception.overhead import FakeOverheadCamera, RGBDFrame, lock_color_controls

OPT = SimpleNamespace(
    enable_auto_exposure="ae",
    exposure="exp",
    gain="gain",
    enable_auto_white_balance="awb",
    white_balance="wb",
)


class SimSensor:
    """Brightness = exposure / 2 (auto: 100); B - R = (4600 - K) / 20 (auto: neutral)."""

    def __init__(self) -> None:
        self.o = {"ae": 1.0, "exp": 166.0, "gain": 64.0, "awb": 1.0, "wb": 3000.0}
        self.sets: list[tuple[str, float]] = []

    def get_option(self, k: str) -> float:
        return self.o[k]

    def set_option(self, k: str, v: float) -> None:
        self.sets.append((k, v))
        self.o[k] = float(v)

    def frame(self) -> np.ndarray:
        y = 100.0 if self.o["ae"] else self.o["exp"] / 2.0
        d = 0.0 if self.o["awb"] else (4600.0 - self.o["wb"]) / 20.0
        img = np.empty((48, 64, 3), np.float64)
        img[..., 0], img[..., 1], img[..., 2] = y + d / 2, y, y - d / 2
        return img


def test_lock_matches_the_settled_auto_image() -> None:
    s = SimSensor()
    lock = lock_color_controls(s, OPT, s.frame, OverheadCameraConfig(settle_s=0.0))
    assert lock.source == "exposure=matched, wb=matched"
    assert lock.exposure == pytest.approx(200.0, abs=2.0)
    assert lock.white_balance_k == pytest.approx(4600.0, abs=20.0)
    assert lock.locked_mean == pytest.approx(lock.auto_mean, abs=1.5)
    assert s.o["ae"] == 0.0 and s.o["awb"] == 0.0  # really locked
    assert s.o["exp"] == lock.exposure and s.o["wb"] == lock.white_balance_k


def test_config_values_win_and_exposure_search_is_capped() -> None:
    s = SimSensor()
    cfg = OverheadCameraConfig(settle_s=0.0, exposure=120, gain=32, white_balance_k=5000)
    lock = lock_color_controls(s, OPT, s.frame, cfg)
    assert (lock.exposure, lock.gain, lock.white_balance_k) == (120.0, 32.0, 5000.0)
    assert lock.source == "exposure=config, wb=config"
    s2 = SimSensor()
    lock2 = lock_color_controls(
        s2, OPT, s2.frame, OverheadCameraConfig(settle_s=0.0, max_exposure=50)
    )
    assert lock2.exposure == 50.0  # too dark at the cap: take the cap (keeps 30 FPS)


def test_fake_overhead_camera_can_publish_a_fixed_frame() -> None:
    from lbyr.perception.geometry import Intrinsics

    f = RGBDFrame(
        np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4), np.float32), Intrinsics(1, 1, 2, 2, 4, 4)
    )
    cam = FakeOverheadCamera(frame=f)
    cam.start()
    try:
        item = cam.buffer.get()
        assert item is not None and item.value is f
    finally:
        cam.stop()
