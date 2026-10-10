"""MotionController: never blocks on the Critic, one segment per command, LINK faults."""

import os
import time

import numpy as np
import pytest

from lbyr.ai.critic import CriticWorker, ScriptedCritic, Verdict
from lbyr.arm.mock_arm import MockArm
from lbyr.config import CriticConfig, mock_profile
from lbyr.motion.controller import FAULT, MOVING, SEG_DONE, MotionController

JITTER_P99_MS = float(os.environ.get("LBYR_JITTER_P99_MS", "10.0"))


def _arm() -> MockArm:
    cfg = mock_profile()
    cfg.arm.max_joint_speed_deg_s, cfg.arm.max_joint_accel_deg_s2 = 600.0, 3000.0  # FAKE
    arm = MockArm(cfg.arm, initial_deg=(0,) * 6)
    arm.connect()
    arm.torque(True)
    return arm


def _wait(mc: MotionController, seg: int, timeout_s: float = 5.0) -> str:
    t_end = time.perf_counter() + timeout_s
    while time.perf_counter() < t_end:
        st = mc.current()
        if st.seg_id == seg and st.phase != MOVING:
            return st.phase
        time.sleep(0.01)
    return mc.current().phase


def test_controller_never_blocks_on_critic_and_sends_one_m_per_segment() -> None:
    cfg = mock_profile()
    arm = _arm()
    mc = MotionController(arm, cfg.motion, (0,) * 6)
    critic = CriticWorker(
        ScriptedCritic([Verdict.GRASPED], delay_s=2.0), CriticConfig(timeout_s=3.0)
    )
    mc.start()
    critic.start()
    try:
        time.sleep(0.2)
        seg = mc.command((60, 20, 30, 10, 0, 40))
        critic.submit(np.zeros((480, 640, 3), np.uint8), "VERIFY_GRASP")
        result = None
        t_end = time.perf_counter() + 3.0
        while result is None and time.perf_counter() < t_end:
            result = critic.poll()
            time.sleep(0.1)
        assert _wait(mc, seg) == SEG_DONE
        stats = mc.stats()
    finally:
        critic.stop()
        mc.stop()
    assert result is not None and result.verdict is Verdict.GRASPED
    assert stats.ticks >= int(2.0 * cfg.motion.poll_hz)
    assert stats.missed == 0, stats
    assert stats.p99_ms < JITTER_P99_MS, stats
    assert arm.moves == [(60, 20, 30, 10, 0, 40)]  # ONE segment, no streaming
    assert arm.status_requests >= stats.ticks - 1  # q polled every tick
    np.testing.assert_allclose(arm.read_state().joints, (60, 20, 30, 10, 0, 40), atol=0.1)


def test_new_segment_preempts() -> None:
    arm = _arm()
    mc = MotionController(arm, mock_profile().motion, (0,) * 6)
    mc.start()
    try:
        mc.command((90, 0, 0, 0, 0, 0))
        time.sleep(0.1)
        seg = mc.command((10, 0, 0, 0, 0, 0))
        assert _wait(mc, seg) == SEG_DONE
        assert mc.current().goal == (10, 0, 0, 0, 0, 0)
        assert (
            arm.read_state().joints[0] == np.float64(10.0)
            or abs(arm.read_state().joints[0] - 10.0) < 0.1
        )
    finally:
        mc.stop()


def test_dead_link_is_a_fault_after_predicted_x_1_5_plus_margin() -> None:
    arm = _arm()
    cfg = mock_profile().motion
    mc = MotionController(arm, cfg, (0,) * 6)
    arm.set_link_dead(True)
    mc.start()
    try:
        t0 = time.perf_counter()
        seg = mc.command((20, 0, 0, 0, 0, 0))
        assert _wait(mc, seg) == FAULT
        st = mc.current()
        assert "LINK: no DONE" in (st.fault or "")
        assert time.perf_counter() - t0 == pytest.approx(
            cfg.done_timeout_s(st.predicted_s), abs=0.15
        )
    finally:
        mc.stop()


def test_output_disabled_faults_instead_of_moving() -> None:
    arm = _arm()
    mc = MotionController(arm, mock_profile().motion, (0,) * 6)
    mc.set_output_enabled(False)
    seg = mc.command((5, 0, 0, 0, 0, 0))
    mc.tick()
    st = mc.current()
    assert st.seg_id == seg and st.phase == FAULT and arm.moves == []
