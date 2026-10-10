"""Simulated arm for mock mode and tests.

Mirrors the bridge-side safety layer (bible Section 6): joint clamps and a maximum
delta per streamed tick. Supports injected latency and jams for testing.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence

import numpy as np

from lbyr.arm.bridge_proto import STATUS_OK
from lbyr.arm.interface import ArmInterface, ArmState
from lbyr.config import NUM_JOINTS, ArmConfig, require

log = logging.getLogger(__name__)

STREAM_PERIOD_S = 0.02  # used to convert move durations into per-tick deltas
JAM_LOAD = 1.0  # synthetic load units
NOMINAL_LOAD = 0.1


class MockArm(ArmInterface):
    def __init__(
        self,
        config: ArmConfig,
        latency_s: float = 0.0,
        initial_deg: Sequence[float] | None = None,
    ) -> None:
        self._cfg = config
        lo, hi = config.limits()
        self._lo = np.asarray(lo, dtype=float)
        self._hi = np.asarray(hi, dtype=float)
        self._max_delta = require(config.max_delta_per_tick_deg, "max delta per tick", "A-2")
        start = initial_deg if initial_deg is not None else config.safe_pose_deg
        start = start if start is not None else (0.0,) * NUM_JOINTS
        self._pos = np.clip(np.asarray(start, dtype=float), self._lo, self._hi)
        self._lock = threading.Lock()
        self._latency_s = latency_s
        self._jammed: set[int] = set()
        self._torque_on = False
        self._connected = False
        self._move: tuple[np.ndarray, np.ndarray, float, float] | None = None
        self.stream_count = 0

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

    def stream_point(self, joints_deg: Sequence[float]) -> None:
        target = self.clamp(joints_deg)
        self._sleep_latency()
        with self._lock:
            self.stream_count += 1
            if not self._torque_on:
                return
            self._advance_locked(time.perf_counter())
            self._move = None
            delta = np.clip(target - self._pos, -self._max_delta, self._max_delta)
            self._apply_locked(self._pos + delta)

    def move_joints(self, joints_deg: Sequence[float], time_ms: int) -> None:
        if time_ms <= 0:
            raise ValueError("time_ms must be > 0")
        target = self.clamp(joints_deg)
        self._sleep_latency()
        with self._lock:
            if not self._torque_on:
                return
            now = time.perf_counter()
            self._advance_locked(now)
            # Stretch the duration so no 20 ms slice exceeds the per-tick clamp.
            biggest = float(np.max(np.abs(target - self._pos)))
            min_s = biggest / self._max_delta * STREAM_PERIOD_S
            self._move = (self._pos.copy(), target, now, max(time_ms / 1000.0, min_s))

    def read_state(self) -> ArmState:
        self._sleep_latency()
        with self._lock:
            now = time.perf_counter()
            self._advance_locked(now)
            load = tuple(JAM_LOAD if i in self._jammed else NOMINAL_LOAD for i in range(NUM_JOINTS))
            return ArmState(
                joints=tuple(float(x) for x in self._pos),
                load=load,
                temp=(30.0,) * NUM_JOINTS,
                status=STATUS_OK,
                t=now,
            )

    def torque(self, on: bool) -> None:
        with self._lock:
            self._advance_locked(time.perf_counter())
            self._move = None
            self._torque_on = on
        log.info("MockArm torque %s", "ON" if on else "OFF")

    def stop(self) -> None:
        with self._lock:
            self._advance_locked(time.perf_counter())
            self._move = None
        log.info("MockArm stop: holding position")

    # ------------------------------------------------------------- internals
    @staticmethod
    def _as_vec(joints_deg: Sequence[float]) -> np.ndarray:
        v = np.asarray(joints_deg, dtype=float)
        if v.shape != (NUM_JOINTS,) or not np.all(np.isfinite(v)):
            raise ValueError(f"expected {NUM_JOINTS} finite joint values, got {joints_deg!r}")
        return v

    def _sleep_latency(self) -> None:
        if self._latency_s > 0:
            time.sleep(self._latency_s)

    def _apply_locked(self, new_pos: np.ndarray) -> None:
        for i in self._jammed:
            new_pos[i] = self._pos[i]
        self._pos = np.clip(new_pos, self._lo, self._hi)

    def _advance_locked(self, now: float) -> None:
        if self._move is None:
            return
        start, target, t0, dur = self._move
        frac = min(max((now - t0) / dur, 0.0), 1.0)
        self._apply_locked(start + frac * (target - start))
        if frac >= 1.0:
            self._move = None
