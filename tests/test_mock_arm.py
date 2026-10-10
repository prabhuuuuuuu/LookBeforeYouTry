import time

import numpy as np
import pytest

from lbyr.arm.interface import DONE, IDLE, MOVING
from lbyr.arm.mock_arm import MockArm
from lbyr.config import ArmConfig, mock_profile
from lbyr.errors import HardwareNotConfirmedError


def make_arm() -> MockArm:
    cfg = mock_profile().arm
    cfg.max_joint_speed_deg_s, cfg.max_joint_accel_deg_s2 = 600.0, 3000.0  # FAKE, fast
    arm = MockArm(cfg, initial_deg=(0.0,) * 6)
    arm.connect()
    arm.torque(True)
    return arm


def test_one_segment_done_matches_prediction() -> None:
    arm = make_arm()
    assert arm.motion_status() == (IDLE, None)
    goal = (40.0, -20.0, 10.0, 0.0, 15.0, 30.0)
    predicted = arm.move_joints(goal)
    t0 = time.perf_counter()
    assert arm.motion_status()[0] == MOVING
    assert arm.wait_done(predicted * 1.5 + 0.5) == DONE
    assert time.perf_counter() - t0 == pytest.approx(predicted, abs=0.1)
    s = arm.read_state()
    np.testing.assert_allclose(s.joints, goal, atol=0.1)
    assert s.feedback == "measured" and s.commanded is not None and s.moving is False


def test_clamps_to_joint_limits() -> None:
    arm = make_arm()
    cfg = mock_profile().arm
    assert cfg.joint_max_deg is not None and cfg.joint_min_deg is not None
    p = arm.move_joints((1000.0, -1000.0, 0.0, 0.0, 0.0, 1000.0))
    assert arm.wait_done(p + 1) == DONE
    s = arm.read_state()
    assert s.joints[0] == pytest.approx(cfg.joint_max_deg[0], abs=0.1)
    assert s.joints[1] == pytest.approx(cfg.joint_min_deg[1], abs=0.1)
    assert s.joints[5] == pytest.approx(cfg.joint_max_deg[5], abs=0.1)


def test_rejects_wrong_length() -> None:
    with pytest.raises(ValueError):
        make_arm().move_joints((0.0, 1.0))


def test_jam_freezes_measured_joint_but_commanded_continues() -> None:
    arm = make_arm()
    arm.inject_jam(1)
    p = arm.move_joints((5.0, 5.0, 0.0, 0.0, 0.0, 0.0))
    assert arm.wait_done(p + 1) == DONE  # the board does not know (no feedback)
    s = arm.read_state()
    assert s.joints[0] == pytest.approx(5.0, abs=0.1) and s.joints[1] == pytest.approx(0.0)
    assert s.commanded is not None and s.commanded[1] == pytest.approx(5.0, abs=0.1)
    assert s.load is not None and s.load[1] > s.load[0]


def test_torque_off_releases_and_m_reattaches() -> None:
    arm = make_arm()
    arm.torque(False)
    assert not arm.torque_on
    p = arm.move_joints((5.0, 0.0, 0.0, 0.0, 0.0, 0.0))  # like the firmware: `m` attaches
    assert arm.torque_on and arm.wait_done(p + 1) == DONE


def test_link_dead_never_reports_done() -> None:
    arm = make_arm()
    arm.set_link_dead(True)
    arm.move_joints((10.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert arm.wait_done(0.3) == MOVING
    assert arm.read_state().joints[0] == 0.0


def test_mock_arm_requires_limits() -> None:
    with pytest.raises(HardwareNotConfirmedError):
        MockArm(ArmConfig())
