import time

import pytest

from lbyr.config import WristCameraConfig
from lbyr.errors import HardwareNotConfirmedError
from lbyr.perception.buffers import LatestValueBuffer
from lbyr.perception.wrist import FakeWristCamera, TrinetWristCamera


def test_latest_value_overwrites_and_never_blocks() -> None:
    buf: LatestValueBuffer[int] = LatestValueBuffer()
    assert buf.get() is None
    buf.put(1)
    seq = buf.put(2)
    item = buf.get()
    assert item is not None and item.value == 2 and item.seq == seq
    assert buf.get_if_newer(seq) is None
    buf.put(3)
    newer = buf.get_if_newer(seq)
    assert newer is not None and newer.value == 3


def test_fake_wrist_camera_publishes() -> None:
    cam = FakeWristCamera(64, 48, fps=100)
    cam.start()
    time.sleep(0.05)
    item = cam.buffer.get()
    cam.stop()
    assert item is not None and item.value.shape == (48, 64, 3)


def test_trinet_requires_confirmed_index() -> None:
    with pytest.raises(HardwareNotConfirmedError, match="OQ-7"):
        TrinetWristCamera(WristCameraConfig())
