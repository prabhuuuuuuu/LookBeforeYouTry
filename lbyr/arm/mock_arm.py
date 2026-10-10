"""Simulated arm for mock mode and tests.

Runs the Python port of the bridge firmware planner (:mod:`lbyr.arm.fw_planner`) in real
time, so mock motion timing matches the real board: one ``move_joints`` = one ``m``, and
``motion_status`` reports DONE when every joint has settled. Joint limits (FAKE in
``mock_profile``) act as the firmware's pulse clamp.

Unlike the real arm, the mock has "measured" positions: an injected jam freezes a joint
while its commanded position keeps going, which the FSM detects as JAMMED. A dead link
(``set_link_dead``) never reports DONE, which the motion controller turns into LINK.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence

import numpy as np

from lbyr.arm.calib import JointsCalib, mock_joints
from lbyr.arm.fw_planner import TICK_S, US_PER_DEG, FirmwarePlanner
from lbyr.arm.interface import DONE, IDLE, MOVING, STATUS_OK, ArmInterface, ArmState
from lbyr.config import NUM_JOINTS, ArmConfig, require

log = logging.getLogger(__name__)

JAM_LOAD = 1.0  # synthetic load units
NOMINAL_LOAD = 0.1


class MockArm(ArmInterface):
    def __init__(
        self,
        config: ArmConfig,
        latency_s: float = 0.0,
        initial_deg: Sequence[float] | None = None,
        joints: JointsCalib | None = None,
    ) -> None:
        self._cfg = config
        lo, hi = config.limits()
        self._lo = np.asarray(lo, dtype=float)
        self._hi = np.asarray(hi, dtype=float)
        self._joints = joints or mock_joints()
        vmax = require(config.max_joint_speed_deg_s, "max joint speed", "A-1")
        amax = require(config.max_joint_accel_deg_s2, "max joint acceleration", "A-1")
        zero = self._joints.zero_us
        fw_lo = self._joints.host_to_fw_deg(lo)
        fw_hi = self._joints.host_to_fw_deg(hi)
        min_us = [z + min(a, b) * US_PER_DEG for z, a, b in zip(zero, fw_lo, fw_hi, strict=True)]
        max_us = [z + max(a, b) * US_PER_DEG for z, a, b in zip(zero, fw_lo, fw_hi, strict=True)]
        self._planner = FirmwarePlanner(zero, vmax, amax, min_us, max_us)
        start = initial_deg if initial_deg is not None else config.safe_pose_deg
        start = start if start is not None else (0.0,) * NUM_JOINTS
        self._planner.set_positions_us(self._host_to_us(self.clamp(start).tolist()))
        self._actual = self._commanded_host()
        self.peak_deg = self._actual.copy()  # test hook: per-joint maximum ever reached
        self._lock = threading.Lock()
        self._latency_s = latency_s
        self._jammed: set[int] = set()
        self._torque_on = False
        self._connected = False
        self._t_sim = time.perf_counter()
        self._segments = 0
        self._link_dead = False
        self.status_requests = 0
        self.moves: list[tuple[float, ...]] = []  # test hook: every commanded segment

    # ------------------------------------------------------------- test hooks
    def set_latency(self, latency_s: float) -> None:
        self._latency_s = latency_s

    def inject_jam(self, joint_index: int) -> None:
        with self._lock:
            self._advance_locked(time.perf_counter())
            self._jammed.add(joint_index)

    def clear_jams(self) -> None:
        with self._lock:
            self._jammed.clear()

    def set_link_dead(self, dead: bool) -> None:
        """Simulate a lost link: commands are dropped and DONE never arrives."""
        self._link_dead = dead

    @property
    def torque_on(self) -> bool:
        return self._torque_on

    # ------------------------------------------------------------- interface
    def connect(self) -> None:
        self._connected = True
        log.info("MockArm connected (SIMULATION ONLY)")

    def close(self) -> None:
        self._connected = False

    def clamp(self, joints_deg: Sequence[float]) -> np.ndarray:
        return np.clip(self._as_vec(joints_deg), self._lo, self._hi)

    def move_joints(self, joints_deg: Sequence[float]) -> float:
        target = self.clamp(joints_deg)
        self._sleep_latency()
        with self._lock:
            self._advance_locked(time.perf_counter())
            self.moves.append(tuple(float(x) for x in target))
            self._segments += 1
            if self._link_dead:
                return self._predict_locked(target)
            # Like the firmware's `m`: (re-)attaches every joint, then a synced move.
            self._torque_on = True
            for q in self._planner.j:
                q.attached = True
            self._planner.move_deg(self._joints.host_to_fw_deg(target.tolist()))
            return self._planner.predict_settle_s()

    def motion_status(self) -> tuple[str, str | None]:
        with self._lock:
            self._advance_locked(time.perf_counter())
            if self._segments == 0:
                return IDLE, None
            if self._link_dead or self._planner.any_moving():
                return MOVING, None
            return DONE, None

    def request_state(self) -> None:
        self.status_requests += 1

    def read_state(self) -> ArmState:
        self._sleep_latency()
        with self._lock:
            now = time.perf_counter()
            self._advance_locked(now)
            load = tuple(JAM_LOAD if i in self._jammed else NOMINAL_LOAD for i in range(NUM_JOINTS))
            return ArmState(
                joints=tuple(float(x) for x in self._actual),
                load=load,
                temp=(30.0,) * NUM_JOINTS,
                status=STATUS_OK,
                t=now,
                feedback="measured",
                commanded=tuple(float(x) for x in self._commanded_host()),
                moving=self._planner.any_moving(),
            )

    def torque(self, on: bool) -> None:
        with self._lock:
            self._advance_locked(time.perf_counter())
            if not on:
                self._planner.stop_all()
                for q in self._planner.j:  # `off`: released, positions frozen
                    q.vel, q.tgt, q.has_fin, q.attached = 0.0, q.pos, False, False
            else:
                for q in self._planner.j:
                    q.attached = True
            self._torque_on = on
        log.info("MockArm torque %s", "ON" if on else "OFF")

    def stop(self) -> None:
        with self._lock:
            self._advance_locked(time.perf_counter())
            self._planner.stop_all()
        log.info("MockArm stop: braking, then holding")

    # ------------------------------------------------------------- internals
    def predict(self, joints_deg: Sequence[float]) -> float:
        """Predicted duration of a segment from the current state (no side effects)."""
        with self._lock:
            self._advance_locked(time.perf_counter())
            return self._predict_locked(self.clamp(joints_deg))

    def _predict_locked(self, target: np.ndarray) -> float:
        import copy

        sim = copy.deepcopy(self._planner)
        sim.move_deg(self._joints.host_to_fw_deg(target.tolist()))
        return sim.predict_settle_s()

    @staticmethod
    def _as_vec(joints_deg: Sequence[float]) -> np.ndarray:
        v = np.asarray(joints_deg, dtype=float)
        if v.shape != (NUM_JOINTS,) or not np.all(np.isfinite(v)):
            raise ValueError(f"expected {NUM_JOINTS} finite joint values, got {joints_deg!r}")
        return v

    def _host_to_us(self, host: Sequence[float]) -> list[float]:
        fw = self._joints.host_to_fw_deg(host)
        return [z + d * US_PER_DEG for z, d in zip(self._joints.zero_us, fw, strict=True)]

    def _commanded_host(self) -> np.ndarray:
        return np.asarray(self._joints.us_to_host_deg(self._planner.positions_us()), dtype=float)

    def _sleep_latency(self) -> None:
        if self._latency_s > 0:
            time.sleep(self._latency_s)

    def _advance_locked(self, now: float) -> None:
        """Step the firmware planner in 10 ms ticks up to ``now``."""
        if not self._planner.any_moving():
            self._t_sim = now
        else:
            n = int((now - self._t_sim) / TICK_S)
            for _ in range(n):
                self._planner.step(TICK_S)
                if not self._planner.any_moving():
                    break
            self._t_sim += n * TICK_S
        cmd = self._commanded_host()
        for i in range(NUM_JOINTS):
            if i not in self._jammed:
                self._actual[i] = cmd[i]
        self.peak_deg = np.maximum(self.peak_deg, self._actual)
