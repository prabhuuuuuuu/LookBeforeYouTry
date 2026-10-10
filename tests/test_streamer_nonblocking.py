"""Non-blocking guarantee: the 50 Hz streamer keeps time while the Critic is busy.

p99 jitter threshold is configurable via LBYR_JITTER_P99_MS (generous default for CI).
"""

import os
import time

import numpy as np

from lbyr.ai.critic import CriticWorker, ScriptedCritic, Verdict
from lbyr.arm.mock_arm import MockArm
from lbyr.config import CriticConfig, mock_profile
from lbyr.motion.streamer import MotionStreamer
from lbyr.motion.trajectory import plan_capped

JITTER_P99_MS = float(os.environ.get("LBYR_JITTER_P99_MS", "10.0"))


def test_streamer_never_blocks_on_critic() -> None:
    cfg = mock_profile()
    arm = MockArm(cfg.arm, initial_deg=(0,) * 6)
    arm.connect()
    arm.torque(True)
    streamer = MotionStreamer(arm, cfg.motion, (0,) * 6)
    backend = ScriptedCritic([Verdict.GRASPED], delay_s=2.0)
    critic = CriticWorker(backend, CriticConfig(timeout_s=3.0))
    streamer.start()
    critic.start()
    try:
        time.sleep(0.2)  # warm-up
        streamer.command(plan_capped((0,) * 6, (60, 20, 30, 10, 0, 40), cfg.arm, cfg.motion))
        critic.submit(np.zeros((480, 640, 3), np.uint8), "VERIFY_GRASP")
        result = None
        t_end = time.perf_counter() + 3.0
        while result is None and time.perf_counter() < t_end:
            result = critic.poll()  # FSM-style non-blocking poll
            time.sleep(0.1)
        stats = streamer.stats()
    finally:
        critic.stop()
        streamer.stop()

    assert result is not None and result.verdict is Verdict.GRASPED
    assert result.latency_ms >= 2000
    assert stats.ticks >= int(2.0 * cfg.motion.rate_hz)
    assert stats.missed == 0, stats
    assert stats.p99_ms < JITTER_P99_MS, stats
    status = streamer.current()
    assert status.done and status.errors == 0
    np.testing.assert_allclose(arm.read_state().joints, (60, 20, 30, 10, 0, 40), atol=1e-6)


def test_streamer_holds_when_idle_and_preempts() -> None:
    cfg = mock_profile()
    arm = MockArm(cfg.arm, initial_deg=(0,) * 6)
    arm.connect()
    arm.torque(True)
    streamer = MotionStreamer(arm, cfg.motion, (0,) * 6)
    streamer.start()
    try:
        time.sleep(0.1)
        n0 = arm.stream_count
        time.sleep(0.1)
        assert arm.stream_count > n0  # keeps sending hold setpoints
        streamer.command(plan_capped((0,) * 6, (90, 0, 0, 0, 0, 0), cfg.arm, cfg.motion))
        time.sleep(0.1)
        tid = streamer.hold(arm.read_state().joints)  # pre-empt mid-motion
        time.sleep(0.1)
        status = streamer.current()
        assert status.traj_id == tid and status.done
        assert arm.read_state().joints[0] < 90
    finally:
        streamer.stop()
