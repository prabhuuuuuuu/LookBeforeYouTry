"""A fake XIAO bridge for SerialArm tests: speaks the text protocol and moves with the
firmware planner port in real time. Implements the pyserial subset SerialArm uses."""

from __future__ import annotations

import queue
import threading
import time

from lbyr.arm.fw_planner import DEFAULT_ZERO_US, TICK_S, US_PER_DEG, FirmwarePlanner


class FakeBoard:
    def __init__(
        self,
        zero_us: tuple[int, ...] = DEFAULT_ZERO_US,
        vmax: float = 60.0,
        amax: float = 200.0,
        send_done: bool = True,
        answer: bool = True,
    ) -> None:
        self.zero_us = zero_us
        self.p = FirmwarePlanner(zero_us, vmax, amax)
        self.send_done = send_done  # False: never prints DONE (dead link simulation)
        self.answer = answer  # False: silent board
        self.out: queue.Queue[str] = queue.Queue()
        self.written: list[str] = []
        self.pending = False
        self.t = time.perf_counter()
        self.lock = threading.Lock()
        self.closed = False
        self.read_error: Exception | None = None

    # pyserial subset
    def write(self, data: bytes) -> int:
        for line in data.decode().splitlines():
            with self.lock:
                self.written.append(line)
                self._advance()
                self._handle(line.strip())
        return len(data)

    def readline(self) -> bytes:
        if self.read_error is not None:
            raise self.read_error
        with self.lock:
            self._advance()
        try:
            return (self.out.get(timeout=0.01) + "\n").encode()
        except queue.Empty:
            return b""

    def close(self) -> None:
        self.closed = True

    # firmware behaviour
    def say(self, line: str) -> None:
        if self.answer:
            self.out.put(line)

    def _advance(self) -> None:
        now = time.perf_counter()
        n = int((now - self.t) / TICK_S)
        for _ in range(n):
            self.p.step(TICK_S)
        self.t += n * TICK_S
        if self.pending and not self.p.any_moving():
            self.pending = False
            if self.send_done:
                self.say("DONE")

    def _handle(self, cmd: str) -> None:
        if cmd == "list":
            for i, q in enumerate(self.p.j):
                moving = int(q.attached and not self.p.settled(i))
                self.say(
                    f"S,{i + 1},{self.zero_us[i]},{q.pos:.0f},{int(q.attached)},"
                    f"{self.p.goal_of(i):.0f},{moving},Joint {i + 1}"
                )
            self.say(f"V,{self.p.vmax_deg:.0f},{self.p.amax_deg:.0f}")
        elif cmd == "q":
            cur = ",".join(f"{q.pos:.0f}" for q in self.p.j)
            self.say(f"Q,{int(self.p.any_moving())},{cur}")
        elif cmd.startswith("m "):
            self.p.move_deg([float(v) for v in cmd.split()[1:]])
            self.pending = True
            self.say("OK m")
        elif cmd.startswith("speed "):
            v, a = (float(x) for x in cmd.split()[1:3])
            self.p.vmax_deg, self.p.amax_deg = min(v, 60.0), min(a, 200.0)
            self.say(f"OK speed {self.p.vmax_deg:.0f} deg/s, {self.p.amax_deg:.0f} deg/s^2")
        elif cmd == "x":
            self.p.stop_all()
            self.say("OK stopped; holding")
        elif cmd == "off":
            for q in self.p.j:
                q.attached = False
            self.say("OK off")
        else:
            self.say("ERR unknown command; enter help")

    def fw_deg(self) -> tuple[float, ...]:
        return tuple((q.pos - z) / US_PER_DEG for q, z in zip(self.p.j, self.zero_us, strict=True))
