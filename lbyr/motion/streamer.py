"""50 Hz motion streamer (bible 9.3). NEVER blocks on AI, detection, logging or I/O.

- Deadline scheduling with ``time.perf_counter()`` (absolute deadlines, no drift).
- Input: a latest-value slot holding a whole :class:`Trajectory`. A new trajectory
  pre-empts the old one; the streamer advances one point per tick and holds the last
  point when the trajectory is exhausted (or none was given). Holding still sends the
  setpoint every tick so the bridge watchdog stays fed.
- Output: a latest-value :class:`StreamerStatus` and jitter statistics.
- Logging goes through the QueueHandler (see log_setup), so it only enqueues.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from lbyr.arm.interface import ArmInterface
from lbyr.config import MotionConfig
from lbyr.motion.trajectory import Trajectory
from lbyr.perception.buffers import LatestValueBuffer

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StreamerStatus:
    traj_id: int  # id of the trajectory currently being followed (0 = initial hold)
    setpoint: tuple[float, ...]  # last streamed setpoint, deg
    done: bool  # trajectory exhausted, now holding
    ticks: int
    missed: int
    errors: int


@dataclass(frozen=True)
class JitterStats:
    ticks: int
    missed: int
    p50_ms: float
    p99_ms: float
    max_ms: float


class MotionStreamer(threading.Thread):
    def __init__(
        self, arm: ArmInterface, cfg: MotionConfig, initial_setpoint: Sequence[float]
    ) -> None:
        super().__init__(name="motion-50Hz", daemon=True)
        self._arm = arm
        self._cfg = cfg
        self._period = cfg.period_s
        self._commands: LatestValueBuffer[Trajectory] = LatestValueBuffer()
        self.status: LatestValueBuffer[StreamerStatus] = LatestValueBuffer()
        self._hold = np.asarray(initial_setpoint, dtype=float)
        self._stop_event = threading.Event()
        self._jitter: collections.deque[float] = collections.deque(maxlen=cfg.jitter_window)
        self._jitter_lock = threading.Lock()
        self._ticks = 0
        self._missed = 0
        self._errors = 0
        self.status.put(StreamerStatus(0, tuple(self._hold), True, 0, 0, 0))

    # ------------------------------------------------------------ producer API
    def command(self, traj: Trajectory) -> int:
        """Hand a trajectory to the streamer (non-blocking). Returns its id."""
        return self._commands.put(traj)

    def hold(self, joints_deg: Sequence[float]) -> int:
        pts = np.asarray(joints_deg, dtype=float).reshape(1, -1)
        return self.command(Trajectory(pts, self._period))

    def current(self) -> StreamerStatus:
        item = self.status.get()
        assert item is not None  # set in __init__
        return item.value

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
        traj: Trajectory | None = None
        traj_id, idx = 0, 0
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

            new = self._commands.get_if_newer(traj_id)
            if new is not None:
                traj, traj_id, idx = new.value, new.seq, 0
            if traj is not None and idx < len(traj.points):
                self._hold = traj.points[idx]
                idx += 1
            done = traj is None or idx >= len(traj.points)

            try:
                self._arm.stream_point(self._hold.tolist())
            except Exception:
                self._errors += 1
                if self._errors == 1 or self._errors % 50 == 0:
                    log.exception("stream_point failed (%d errors)", self._errors)

            self._ticks += 1
            self.status.put(
                StreamerStatus(
                    traj_id,
                    tuple(float(x) for x in self._hold),
                    done,
                    self._ticks,
                    self._missed,
                    self._errors,
                )
            )
            deadline += self._period
            self._sleep_until(deadline)

    def _sleep_until(self, deadline: float) -> None:
        remaining = deadline - time.perf_counter()
        if remaining > self._cfg.spin_margin_s:
            time.sleep(remaining - self._cfg.spin_margin_s)
        while time.perf_counter() < deadline and not self._stop_event.is_set():
            time.sleep(0)
