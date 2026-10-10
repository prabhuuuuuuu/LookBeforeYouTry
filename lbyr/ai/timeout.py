"""Timeout wrapper for every VLM call (bible 21.9: all VLM calls have explicit timeouts).

Python cannot kill a thread, so a timed-out call keeps running in the background until
it returns. :class:`TimeoutRunner` therefore refuses to start a new call while an
abandoned one is still in flight, so GPU work never stacks up.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

R = TypeVar("R")


@dataclass(frozen=True)
class TimeoutResult:
    timeout_s: float
    elapsed_s: float
    busy: bool = False  # True if rejected because a previous call is still running


class _Call(Generic[R]):
    def __init__(self, fn: Callable[[], R]) -> None:
        self.fn = fn
        self.done = threading.Event()
        self.result: Any = None
        self.exc: BaseException | None = None

    def __call__(self) -> None:
        try:
            self.result = self.fn()
        except BaseException as e:  # re-raised in the caller thread
            self.exc = e
        finally:
            self.done.set()


def run_with_timeout(
    fn: Callable[[], R], timeout_s: float, name: str = "vlm-call"
) -> R | TimeoutResult:
    """Run ``fn`` on a daemon worker thread; return its result or a TimeoutResult.

    Exceptions raised by ``fn`` are re-raised in the calling thread.
    """
    if timeout_s <= 0:
        raise ValueError("timeout_s must be > 0")
    call: _Call[R] = _Call(fn)
    t0 = time.perf_counter()
    threading.Thread(target=call, name=name, daemon=True).start()
    if not call.done.wait(timeout_s):
        return TimeoutResult(timeout_s, time.perf_counter() - t0)
    if call.exc is not None:
        raise call.exc
    return call.result


class TimeoutRunner:
    """Single-flight wrapper around :func:`run_with_timeout`."""

    def __init__(self, timeout_s: float, name: str = "vlm-call") -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        self.timeout_s = timeout_s
        self._name = name
        self._inflight: _Call[Any] | None = None

    @property
    def busy(self) -> bool:
        return self._inflight is not None and not self._inflight.done.is_set()

    def run(self, fn: Callable[[], R]) -> R | TimeoutResult:
        if self.busy:
            return TimeoutResult(self.timeout_s, 0.0, busy=True)
        call: _Call[R] = _Call(fn)
        self._inflight = call
        t0 = time.perf_counter()
        threading.Thread(target=call, name=self._name, daemon=True).start()
        if not call.done.wait(self.timeout_s):
            return TimeoutResult(self.timeout_s, time.perf_counter() - t0)
        if call.exc is not None:
            raise call.exc
        return call.result
