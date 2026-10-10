"""Python port of the XIAO bridge firmware motion planner (firmware/xiao_c6_bridge/Code.ino).

The firmware is the trajectory generator (PP-1): the host sends ONE ``m`` per motion
segment and waits for ``DONE``. This port reproduces the firmware's per-joint planner
tick for tick (100 Hz, speed + acceleration limited, JOINT_SPEED scaling, synced arrival,
backlash overshoot leg, settle rule) so the host can:

- predict how long a segment takes (:meth:`FirmwarePlanner.predict_settle_s`), which sets
  the DONE timeout in :class:`lbyr.arm.serial_arm.SerialArm`;
- simulate the board in :class:`lbyr.arm.mock_arm.MockArm`, so mock and real timing match.

The constants mirror Code.ino; ``tests/test_fw_planner.py`` parses the sketch and fails if
they drift. Units: microseconds of PWM pulse; degrees are zero-relative (firmware frame).
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass

COUNT = 6
US_PER_DEG = 1000.0 / 90.0
MIN_US: tuple[float, ...] = (500.0,) * COUNT
MAX_US: tuple[float, ...] = (2500.0,) * COUNT
DEFAULT_ZERO_US: tuple[int, ...] = (1477, 1989, 1844, 1223, 1467, 1433)
VMAX_CAP_DEG = 60.0
AMAX_CAP_DEG = 200.0
DEFAULT_VMAX_DEG = 30.0
DEFAULT_AMAX_DEG = 60.0
JOINT_SPEED: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 0.35, 0.35)
BACKLASH_US = 20.0
TICK_S = 0.01  # TICK_US = 10000
SETTLE_US = 0.5
MAX_DT_S = 0.05


@dataclass
class _Joint:
    pos: float
    vel: float = 0.0
    tgt: float = 0.0
    fin: float = 0.0
    has_fin: bool = False
    scale: float = 1.0
    attached: bool = True


class FirmwarePlanner:
    """State of the six firmware joints. Not thread-safe (callers lock)."""

    def __init__(
        self,
        zero_us: Sequence[float],
        vmax_deg: float = DEFAULT_VMAX_DEG,
        amax_deg: float = DEFAULT_AMAX_DEG,
        min_us: Sequence[float] = MIN_US,
        max_us: Sequence[float] = MAX_US,
        joint_speed: Sequence[float] = JOINT_SPEED,
    ) -> None:
        if len(zero_us) != COUNT:
            raise ValueError(f"expected {COUNT} zero pulses")
        if vmax_deg <= 0 or amax_deg <= 0:
            raise ValueError("vmax and amax must be > 0")
        self.zero_us = tuple(float(z) for z in zero_us)
        self.vmax_deg = vmax_deg
        self.amax_deg = amax_deg
        self.min_us = tuple(float(v) for v in min_us)
        self.max_us = tuple(float(v) for v in max_us)
        self.joint_speed = tuple(float(v) for v in joint_speed)
        self.j = [_Joint(pos=self.clamp(i, z), tgt=self.clamp(i, z)) for i, z in enumerate(zero_us)]

    # ------------------------------------------------------------ firmware functions
    def clamp(self, j: int, us: float) -> float:
        return min(max(us, self.min_us[j]), self.max_us[j])

    def goal_of(self, j: int) -> float:
        q = self.j[j]
        return q.fin if q.has_fin else q.tgt

    def set_target(self, j: int, us: float, scale: float) -> None:
        """Approach from below to cancel backlash (firmware ``setTarget``)."""
        q = self.j[j]
        us = self.clamp(j, us)
        q.scale = min(max(scale, 0.02), 1.0)
        if us < q.pos:
            q.tgt = self.clamp(j, us - BACKLASH_US)
            q.fin = us
            q.has_fin = True
        else:
            q.tgt = us
            q.has_fin = False

    def step_joint(self, j: int, dt: float) -> None:
        q = self.j[j]
        k = q.scale * self.joint_speed[j]
        amax = self.amax_deg * US_PER_DEG * k
        vmax = self.vmax_deg * US_PER_DEG * k
        err = q.tgt - q.pos
        if abs(err) <= SETTLE_US and abs(q.vel) <= amax * dt:
            q.pos, q.vel = q.tgt, 0.0
            if q.has_fin:  # overshoot leg done: come back up to the real target
                q.tgt, q.has_fin = q.fin, False
            return
        h = amax * dt * dt
        n = (-1.0 + math.sqrt(1.0 + 8.0 * abs(err) / h)) * 0.5
        vdes = min(n * amax * dt, vmax)
        if err < 0:
            vdes = -vdes
        maxdv = amax * dt
        q.vel += min(max(vdes - q.vel, -maxdv), maxdv)
        nxt = q.pos + q.vel * dt
        if (err > 0 and nxt > q.tgt) or (err < 0 and nxt < q.tgt):
            nxt, q.vel = q.tgt, 0.0
        q.pos = self.clamp(j, nxt)

    def step(self, dt: float = TICK_S) -> None:
        dt = min(dt, MAX_DT_S)
        for i, q in enumerate(self.j):
            if q.attached:
                self.step_joint(i, dt)

    def settled(self, j: int) -> bool:
        q = self.j[j]
        return q.pos == q.tgt and q.vel == 0.0 and not q.has_fin

    def any_moving(self) -> bool:
        return any(q.attached and not self.settled(i) for i, q in enumerate(self.j))

    def move_synced(self, goal_us: Sequence[float], use: Sequence[bool] | None = None) -> None:
        """Start and arrive together, from rest (firmware ``moveSynced``)."""
        use = use if use is not None else [True] * COUNT
        d = [
            abs(self.clamp(i, goal_us[i]) - self.j[i].pos) if use[i] else 0.0 for i in range(COUNT)
        ]
        dmax = max(d)
        for i in range(COUNT):
            if use[i]:
                self.set_target(i, goal_us[i], d[i] / dmax if dmax > 0 else 1.0)

    def move_deg(self, deg: Sequence[float]) -> None:
        """The ``m D1 .. D6`` command (zero-relative degrees). Attaches every joint."""
        if len(deg) != COUNT:
            raise ValueError(f"expected {COUNT} joint values")
        for q in self.j:
            q.attached = True
        self.move_synced([self.zero_us[i] + deg[i] * US_PER_DEG for i in range(COUNT)])

    def stop_all(self) -> None:
        """``x``: brake at the acceleration limit, then hold."""
        for i, q in enumerate(self.j):
            if not q.attached:
                continue
            amax = self.amax_deg * US_PER_DEG * q.scale * self.joint_speed[i]
            q.tgt = self.clamp(i, q.pos + q.vel * abs(q.vel) / (2.0 * amax))
            q.has_fin = False

    # ------------------------------------------------------------ host helpers
    def positions_us(self) -> tuple[float, ...]:
        return tuple(q.pos for q in self.j)

    def positions_deg(self) -> tuple[float, ...]:
        """Commanded positions, zero-relative degrees (firmware frame)."""
        return tuple((q.pos - z) / US_PER_DEG for q, z in zip(self.j, self.zero_us, strict=True))

    def set_positions_us(self, pos_us: Sequence[float]) -> None:
        """Resynchronise from the board (``Q``/``S`` lines): at rest at these pulses."""
        for i, q in enumerate(self.j):
            q.pos = q.tgt = self.clamp(i, float(pos_us[i]))
            q.vel, q.has_fin = 0.0, False

    def predict_settle_s(self, max_s: float = 300.0) -> float:
        """Seconds until every attached joint settles, from the current state (no side
        effects). Counts 10 ms planner ticks exactly like the firmware."""
        sim = copy.deepcopy(self)
        n = 0
        limit = int(max_s / TICK_S)
        while sim.any_moving() and n < limit:
            sim.step(TICK_S)
            n += 1
        return n * TICK_S


def predict_move_s(
    start_us: Sequence[float],
    goal_deg: Sequence[float],
    zero_us: Sequence[float],
    vmax_deg: float = DEFAULT_VMAX_DEG,
    amax_deg: float = DEFAULT_AMAX_DEG,
) -> float:
    """Predicted duration of ``m goal_deg`` from rest at ``start_us``."""
    p = FirmwarePlanner(zero_us, vmax_deg, amax_deg)
    p.set_positions_us(start_us)
    p.move_deg(goal_deg)
    return p.predict_settle_s()
