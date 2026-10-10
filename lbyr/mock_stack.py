"""Wire up a complete SIMULATION-ONLY stack (MockArm, fake cameras, FakePlanner,
ScriptedCritic) for ``python -m lbyr.main --mock`` and the FSM tests."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lbyr.ai.critic import CriticWorker, ScriptedCritic, Verdict
from lbyr.ai.planner import FakePlanner
from lbyr.arm.mock_arm import MockArm
from lbyr.config import AppConfig, mock_profile
from lbyr.fsm.supervisor import Supervisor, SupervisorDeps
from lbyr.logging_.episodes import EpisodeLogger, VramMonitor
from lbyr.motion.ik import MockIK
from lbyr.motion.streamer import MotionStreamer
from lbyr.perception.overhead import FakeOverheadCamera, FakeScene
from lbyr.perception.wrist import FakeWristCamera

log = logging.getLogger(__name__)

# Demo script (definition of done): SLIPPING, then GRASPED; then hold OK; place EMPTY.
DEMO_SCRIPT: tuple[Verdict, ...] = (
    Verdict.SLIPPING,  # VERIFY_GRASP  -> RECOVER(1): +15% closure
    Verdict.GRASPED,  # VERIFY_GRASP (after re-grasp)
    Verdict.GRASPED,  # VERIFY_HOLD
    Verdict.EMPTY,  # VERIFY_PLACE (object released)
)


def _make_dewarp(width: int, height: int) -> Callable[[np.ndarray], np.ndarray] | None:
    try:
        from lbyr.perception.dewarp import Dewarper, synthetic_calibration

        return Dewarper(synthetic_calibration(width, height))
    except ImportError:
        log.warning("cv2 not installed: mock Critic runs without dewarp")
        return None


@dataclass
class MockStack:
    cfg: AppConfig
    arm: MockArm
    streamer: MotionStreamer
    overhead: FakeOverheadCamera
    wrist: FakeWristCamera
    critic_backend: ScriptedCritic
    critic: CriticWorker | None
    logger: EpisodeLogger | None
    vram: VramMonitor
    supervisor: Supervisor

    def start(self) -> None:
        self.arm.connect()
        if self.logger is not None:
            self.logger.start()
        self.vram.start()
        self.overhead.start()
        self.wrist.start()
        self.streamer.start()
        if self.critic is not None:
            self.critic.start()

    def stop(self) -> None:
        if self.critic is not None:
            self.critic.stop()
        self.streamer.stop()
        self.wrist.stop()
        self.overhead.stop()
        self.vram.stop()
        if self.logger is not None:
            self.logger.stop()
        self.arm.close()

    def __enter__(self) -> MockStack:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def build_mock_stack(
    cfg: AppConfig | None = None,
    critic_script: Sequence[Verdict | str] = DEMO_SCRIPT,
    critic_delay_s: float = 0.3,
    critic_enabled: bool = True,
    log_dir: Path | None = None,
    scene: FakeScene | None = None,
    dewarp: bool = True,
) -> MockStack:
    cfg = cfg or mock_profile()
    cfg.critic.enabled = critic_enabled
    cfg.validate()
    arm = MockArm(cfg.arm)
    assert cfg.arm.safe_pose_deg is not None
    streamer = MotionStreamer(arm, cfg.motion, cfg.arm.safe_pose_deg)
    scene = scene or FakeScene()
    overhead = FakeOverheadCamera(scene)
    w, h = cfg.cameras.wrist.width or 640, cfg.cameras.wrist.height or 480
    wrist = FakeWristCamera(w, h)
    backend = ScriptedCritic(critic_script, delay_s=critic_delay_s)
    critic = None
    if critic_enabled:
        pre = _make_dewarp(w, h) if dewarp and cfg.cameras.wrist.dewarp_enabled else None
        critic = CriticWorker(backend, cfg.critic, preprocess=pre)
    logger = EpisodeLogger.for_new_session(log_dir) if log_dir is not None else None
    vram = VramMonitor(cfg.logging.tegrastats_interval_ms)
    deps = SupervisorDeps(
        arm=arm,
        streamer=streamer,
        ik=MockIK(cfg.arm),
        planner=FakePlanner(cfg.skills.default_closure_pct, cfg.skills.default_standoff_mm),
        detector=scene.detector(),
        overhead=overhead.buffer,
        wrist=wrist.buffer,
        base_T_cam=scene.base_T_cam,
        critic=critic,
        logger=logger,
        vram=vram,
    )
    sup = Supervisor(cfg, deps)
    return MockStack(cfg, arm, streamer, overhead, wrist, backend, critic, logger, vram, sup)
