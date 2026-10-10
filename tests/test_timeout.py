import time

import pytest

from lbyr.ai.timeout import TimeoutResult, TimeoutRunner, run_with_timeout


def test_fast_function_returns_result() -> None:
    assert run_with_timeout(lambda: 42, 1.0) == 42


def test_slow_function_returns_timeout_result() -> None:
    t0 = time.perf_counter()
    out = run_with_timeout(lambda: time.sleep(2.0), 0.1)
    elapsed = time.perf_counter() - t0
    assert isinstance(out, TimeoutResult)
    assert out.timeout_s == 0.1
    assert elapsed < 0.5


def test_exception_propagates() -> None:
    def boom() -> int:
        raise ValueError("bad")

    with pytest.raises(ValueError, match="bad"):
        run_with_timeout(boom, 1.0)


def test_runner_single_flight() -> None:
    runner = TimeoutRunner(0.05)
    first = runner.run(lambda: time.sleep(0.3))
    assert isinstance(first, TimeoutResult) and not first.busy
    second = runner.run(lambda: 1)
    assert isinstance(second, TimeoutResult) and second.busy
    time.sleep(0.35)
    assert runner.run(lambda: 7) == 7
