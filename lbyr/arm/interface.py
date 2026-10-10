"""Hardware-agnostic arm interface (bible Section 18: ArmInterface abstraction).

All joint vectors are 6 values in degrees, ordered as ``config.JOINT_NAMES``
(the last entry is the gripper). The host never depends on whether the bridge
drives serial-bus or PWM servos (bible Section 6).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from lbyr.arm.bridge_proto import STATUS_OK


@dataclass(frozen=True)
class ArmState:
    joints: tuple[float, ...]  # deg
    load: tuple[float, ...] | None  # units depend on servo (OQ-5)
    temp: tuple[float, ...] | None  # deg C if readable (OQ-5)
    status: str
    t: float  # time.perf_counter() when read

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK


class ArmInterface(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def move_joints(self, joints_deg: Sequence[float], time_ms: int) -> None:
        """Move to a joint target over ``time_ms`` (bridge-side interpolation)."""

    @abstractmethod
    def stream_point(self, joints_deg: Sequence[float]) -> None:
        """Send one 50 Hz streaming setpoint. Must return quickly (called by the streamer)."""

    @abstractmethod
    def read_state(self) -> ArmState: ...

    @abstractmethod
    def torque(self, on: bool) -> None: ...

    @abstractmethod
    def stop(self) -> None:
        """Safe stop: hold current position."""
