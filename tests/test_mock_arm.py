import time

import pytest

from lbyr.arm.mock_arm import MockArm
from lbyr.arm.serial_arm import SerialArm
from lbyr.config import ArmConfig, mock_profile
from lbyr.errors import ConfigError, HardwareNotConfirmedError


def make_arm() -> MockArm:
    cfg = mock_profile().arm
    arm = MockArm(cfg, initial_deg=(0.0,) * 6)
    arm.connect()
    arm.torque(True)
    return arm


def test_clamps_to_joint_limits() -> None:
    arm = make_arm()
    cfg = mock_profile().arm
    assert cfg.joint_max_deg is not None and cfg.joint_min_deg is not None
    for _ in range(100):
        arm.stream_point((1000.0, -1000.0, 1000.0, -1000.0, 1000.0, 1000.0))
    s = arm.read_state()
    assert s.joints[0] == cfg.joint_max_deg[0]
    assert s.joints[1] == cfg.joint_min_deg[1]
    assert s.joints[5] == cfg.joint_max_deg[5]


def test_max_delta_per_tick() -> None:
    arm = make_arm()
    max_d = mock_profile().arm.max_delta_per_tick_deg
    assert max_d is not None
    arm.stream_point((100.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert arm.read_state().joints[0] == pytest.approx(max_d)
    arm.stream_point((-100.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert arm.read_state().joints[0] == pytest.approx(0.0)


def test_rejects_wrong_length() -> None:
    arm = make_arm()
    with pytest.raises(ValueError):
        arm.stream_point((0.0, 1.0))


def test_torque_off_ignores_commands() -> None:
    arm = make_arm()
    arm.torque(False)
    arm.stream_point((5.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert arm.read_state().joints[0] == 0.0


def test_jam_freezes_joint_and_raises_load() -> None:
    arm = make_arm()
    arm.inject_jam(1)
    arm.stream_point((5.0, 5.0, 0.0, 0.0, 0.0, 0.0))
    s = arm.read_state()
    assert s.joints[0] == 5.0
    assert s.joints[1] == 0.0
    assert s.load is not None and s.load[1] > s.load[0]


def test_move_joints_interpolates() -> None:
    arm = make_arm()
    arm.move_joints((20.0, 0.0, 0.0, 0.0, 0.0, 0.0), time_ms=100)
    time.sleep(0.15)
    assert arm.read_state().joints[0] == pytest.approx(20.0)


def test_mock_arm_requires_limits() -> None:
    with pytest.raises(HardwareNotConfirmedError):
        MockArm(ArmConfig())


def test_serial_arm_refuses_until_confirmed() -> None:
    with pytest.raises(ConfigError):
        SerialArm(mock_profile().arm)
    arm = SerialArm(ArmConfig())
    with pytest.raises(HardwareNotConfirmedError, match="OQ-3"):
        arm.connect()
