"""Thread-safe latest-value buffers and a generic capture thread.

A latest-value buffer holds exactly one item. Writers overwrite; readers never wait.
This is how data is handed to time-critical consumers (e.g. the motion controller) and
how camera frames are shared without unbounded queues.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Stamped(Generic[T]):
    value: T
    t: float  # time.perf_counter() at put()
    seq: int  # monotonically increasing per buffer


class LatestValueBuffer(Generic[T]):
    """Single-slot, overwrite-on-write buffer. All operations are O(1) and non-blocking
    (the internal lock is only held for a pointer swap)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: Stamped[T] | None = None
        self._seq = 0

    def put(self, value: T) -> int:
        with self._lock:
            self._seq += 1
            self._item = Stamped(value, time.perf_counter(), self._seq)
            return self._seq

    def get(self) -> Stamped[T] | None:
        with self._lock:
            return self._item

    def get_if_newer(self, seq: int) -> Stamped[T] | None:
        """Return the item only if it is newer than ``seq`` (else None)."""
        with self._lock:
            item = self._item
        return item if item is not None and item.seq > seq else None

    def clear(self) -> None:
        with self._lock:
            self._item = None


# Camera frames use the same mechanism.
LatestFrameBuffer = LatestValueBuffer


class CaptureThread(threading.Thread, Generic[T]):
    """Calls ``grab()`` in a loop and publishes results to a latest-value buffer.

    ``grab`` returns a frame or ``None`` (dropped frame). Measured FPS is exposed for
    the latency HUD and tools/fps_test.py.
    """

    def __init__(
        self,
        name: str,
        grab: Callable[[], T | None],
        buffer: LatestValueBuffer[T],
        min_period_s: float = 0.0,
    ) -> None:
        super().__init__(name=name, daemon=True)
        self._grab = grab
        self.buffer = buffer
        self._min_period_s = min_period_s
        self._stop_event = threading.Event()
        self.frames = 0
        self.dropped = 0
        self._t_start = 0.0

    def run(self) -> None:
        self._t_start = time.perf_counter()
        next_t = self._t_start
        while not self._stop_event.is_set():
            try:
                frame = self._grab()
            except Exception:
                log.exception("%s: grab failed", self.name)
                self.dropped += 1
                self._stop_event.wait(0.1)
                continue
            if frame is None:
                self.dropped += 1
            else:
                self.buffer.put(frame)
                self.frames += 1
            if self._min_period_s > 0:
                next_t += self._min_period_s
                self._stop_event.wait(max(0.0, next_t - time.perf_counter()))

    def stop(self, timeout: float = 1.0) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout)

    @property
    def fps(self) -> float:
        dt = time.perf_counter() - self._t_start
        return self.frames / dt if self._t_start and dt > 0 else 0.0
