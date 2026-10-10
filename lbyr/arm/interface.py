"""Hardware-agnostic arm interface (bible Section 18: ArmInterface abstraction).

PP-1 (2026-10-10): the bridge firmware is the trajectory generator. The host sends ONE
motion segment (``move_joints``) and then watches for completion (``motion_status``,
the firmware's ``DONE`` line). There is no setpoint streaming.

All joint vectors are 6 values in HOST degrees, ordered as ``config.JOINT_NAMES`` (the last
entry is the gripper). ``lbyr.arm.calib`` converts host <-> firmware (zero-relative) degrees.
Every method except :meth:`ArmInterface.wait_done` returns quickly (at most one serial
write); none of them waits for a reply.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

STATUS_OK = "ok"

# motion_status() phases
IDLE = "idle"  # nothing commanded yet
MOVING = "moving"
DONE = "done"  # the board reported DONE for the latest segment
ERROR = "error"  # the board rejected it (ERR) or the link failed


@dataclass(frozen=True)
class ArmState:
    joints: tuple[float, ...]  # host deg: measured if feedback == "measured", else commanded
    load: tuple[float, ...] | None  # units depend on servo (OQ-5: none on the real arm)
    temp: tuple[float, ...] | None
    status: str
    t: float  # time.perf_counter() when produced
    feedback: str = "commanded"  # "commanded" (real arm, no sensors) | "measured" (mock)
    commanded: tuple[float, ...] | None = None  # host deg, the board's commanded position
    moving: bool | None = None

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK


class ArmInterface(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def move_joints(self, joints_deg: Sequence[float]) -> float:
        """Send ONE motion segment (non-blocking). Returns the predicted duration in s
        (firmware planner port), used for the DONE timeout."""

    @abstractmethod
    def motion_status(self) -> tuple[str, str | None]:
        """(IDLE | MOVING | DONE | ERROR, detail) for the latest segment. Non-blocking."""

    @abstractmethod
    def request_state(self) -> None:
        """Ask the board for a status update (``q``). Non-blocking."""

    @abstractmethod
    def read_state(self) -> ArmState:
        """Latest known state. Non-blocking (may be up to one poll period old)."""

    @abstractmethod
    def torque(self, on: bool) -> None: ...

    @abstractmethod
    def stop(self) -> None:
        """Safe stop: brake and hold (``x``)."""

    def set_speed(self, vmax_deg_s: float, amax_deg_s2: float) -> tuple[float, float] | None:
        """Set the trajectory speed / acceleration limits (firmware ``speed V A``). Returns the
        values now in force, or None if this arm has no such command (mock: FAKE limits
        from mock_profile). Blocking: call at connect time only."""
        return None

    def wait_done(self, timeout_s: float, poll_s: float = 0.01) -> str:
        """BLOCKING helper for tools/tests (never call from the FSM or motion thread).
        Returns the final phase (DONE / ERROR), or MOVING on timeout."""
        t_end = time.perf_counter() + timeout_s
        while True:
            phase, _ = self.motion_status()
            if phase in (DONE, ERROR, IDLE) or time.perf_counter() >= t_end:
                return phase
            time.sleep(poll_s)
