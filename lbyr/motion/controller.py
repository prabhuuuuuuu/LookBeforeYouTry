"""Motion thread: segment dispatcher + status poller (PP-1, replaces the 50 Hz streamer).

The bridge firmware generates the trajectory. This thread, at ``MotionConfig.poll_hz``
(20 Hz):

1. picks up the latest commanded segment (a latest-value buffer; a new segment pre-empts
   the old one) and sends it as ONE ``m`` (:meth:`ArmInterface.move_joints`), which returns
   the duration predicted by the firmware planner port;
2. sends a status request (``q``) and checks :meth:`ArmInterface.motion_status`;
3. marks the segment DONE, or FAULT when the board reports ERR, a write fails, or no DONE
   arrives within ``predicted x done_timeout_factor + done_timeout_margin_s`` (LINK fault).

It never blocks on AI, detection, logging or replies: every arm call is at most one serial
write or a read of state the serial reader thread already parsed. Deadline scheduling
with ``time.perf_counter()``; jitter statistics as before.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from lbyr.arm.interface import DONE, ERROR, ArmInterface
from lbyr.config import MotionConfig
from lbyr.perception.buffers import LatestValueBuffer

log = logging.getLogger(__name__)

# MotionStatus.phase
IDLE = "idle"
MOVING = "moving"
SEG_DONE = "done"
FAULT = "fault"


@dataclass(frozen=True)
class MotionStatus:
    seg_id: int  # id of the latest dispatched segment (0 = none yet)
    goal: tuple[float, ...]  # its goal, host deg (the "setpoint" the FSM builds on)
    phase: str  # IDLE | MOVING | SEG_DONE | FAULT
    fault: str | None
    predicted_s: float
    timeout_s: float
    ticks: int
    missed: int
    errors: int

    @property
    def done(self) -> bool:
        return self.phase == SEG_DONE


@dataclass(frozen=True)
class JitterStats:
    ticks: int
    missed: int
    p50_ms: float
    p99_ms: float
    max_ms: float


class MotionController(threading.Thread):
    def __init__(self, arm: ArmInterface, cfg: MotionConfig, initial_goal: Sequence[float]) -> None:
        super().__init__(name="motion", daemon=True)
        cfg.validate()
        self._arm = arm
        self._cfg = cfg
        self._period = 1.0 / cfg.poll_hz
        self._commands: LatestValueBuffer[np.ndarray] = LatestValueBuffer()
        self.status: LatestValueBuffer[MotionStatus] = LatestValueBuffer()
        self._goal = tuple(float(x) for x in initial_goal)
        self._stop_event = threading.Event()
        self._jitter: collections.deque[float] = collections.deque(maxlen=cfg.jitter_window)
        self._jitter_lock = threading.Lock()
        self._ticks = self._missed = self._errors = 0
        self._output_enabled = True
        self._seg_id = 0
        self._phase = IDLE
        self._fault: str | None = None
        self._predicted = self._timeout = 0.0
        self._t_start = 0.0
        self._publish()

    # ------------------------------------------------------------ producer API (FSM)
    def command(self, goal_deg: Sequence[float] | np.ndarray) -> int:
        """Queue one motion segment (non-blocking). Returns its id."""
        return self._commands.put(np.asarray(goal_deg, dtype=float).copy())

    def current(self) -> MotionStatus:
        item = self.status.get()
        assert item is not None  # set in __init__
        return item.value

    def set_output_enabled(self, enabled: bool) -> None:
        """Disable to send nothing more (e.g. after a JAMMED torque-off)."""
        self._output_enabled = enabled

    def stats(self) -> JitterStats:
        with self._jitter_lock:
            j = np.array(self._jitter, dtype=float)
        if j.size == 0:
            return JitterStats(self._ticks, self._missed, 0.0, 0.0, 0.0)
        return JitterStats(
            self._ticks,
            self._missed,
            float(np.percentile(j, 50) * 1000),
            float(np.percentile(j, 99) * 1000),
            float(j.max() * 1000),
        )

    def stop(self, timeout: float = 1.0) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout)

    # ------------------------------------------------------------ thread
    def run(self) -> None:
        deadline = time.perf_counter()
        while not self._stop_event.is_set():
            now = time.perf_counter()
            late = now - deadline
            if late >= self._period:  # a whole period slipped: count and re-anchor
                skipped = int(late // self._period)
                self._missed += skipped
                deadline += skipped * self._period
                late = now - deadline
            with self._jitter_lock:
                self._jitter.append(abs(late))
            self.tick()
            deadline += self._period
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                self._stop_event.wait(remaining)

    def tick(self) -> None:
        """One poll cycle (public for deterministic tests)."""
        new = self._commands.get_if_newer(self._seg_id)
        if new is not None:
            self._dispatch(new.seq, new.value)
        try:
            if self._output_enabled:
                self._arm.request_state()
        except Exception:
            self._errors += 1
            if self._errors == 1 or self._errors % 20 == 0:
                log.exception("status request failed (%d errors)", self._errors)
        if self._phase == MOVING:
            self._check_segment()
        self._ticks += 1
        self._publish()

    def _dispatch(self, seg_id: int, goal: np.ndarray) -> None:
        self._seg_id = seg_id
        self._goal = tuple(float(x) for x in goal)
        self._fault = None
        if not self._output_enabled:
            self._phase, self._fault = FAULT, "output disabled (after torque off)"
            return
        try:
            self._predicted = float(self._arm.move_joints(self._goal))
        except Exception as e:
            self._phase, self._fault = FAULT, f"LINK: could not send segment: {e}"
            log.error("segment %d: %s", seg_id, self._fault)
            return
        self._timeout = self._cfg.done_timeout_s(self._predicted)
        self._t_start = time.perf_counter()
        self._phase = MOVING
        log.debug(
            "segment %d sent: predicted %.2f s, timeout %.2f s",
            seg_id,
            self._predicted,
            self._timeout,
        )

    def _check_segment(self) -> None:
        try:
            phase, detail = self._arm.motion_status()
        except Exception as e:
            phase, detail = ERROR, f"LINK: status failed: {e}"
        if phase == DONE:
            self._phase = SEG_DONE
        elif phase == ERROR:
            self._phase, self._fault = FAULT, detail or "bridge error"
        elif time.perf_counter() - self._t_start > self._timeout:
            self._phase = FAULT
            self._fault = (
                f"LINK: no DONE within {self._timeout:.1f} s "
                f"(predicted {self._predicted:.1f} s); check the USB cable and the bridge"
            )
        if self._phase == FAULT:
            log.error("segment %d fault: %s", self._seg_id, self._fault)

    def _publish(self) -> None:
        self.status.put(
            MotionStatus(
                self._seg_id,
                self._goal,
                self._phase,
                self._fault,
                self._predicted,
                self._timeout,
                self._ticks,
                self._missed,
                self._errors,
            )
        )
