"""Joint calibration: ``data/calib/joints.yaml`` (zeros, signs, host-frame offsets).

Frames:
- firmware degrees ("zero-relative"): what ``m`` / ``g`` and tools/arm_gui.py sliders use;
  0 = the joint's saved zero pulse, +1 deg = +1000/90 us.
- host degrees: what the FSM, IK and :class:`lbyr.arm.interface.ArmInterface` use::

      host_deg = zero_pose_deg + sign * firmware_deg

  With the seeded values (sign +1, zero_pose_deg 0, A-7 unconfirmed) both frames are equal.
  Taught poses are stored in firmware degrees and converted with this file.

``SerialArm.connect`` compares ``zero_us`` with the board's ``list`` output and refuses to
move on any mismatch. ``verified: true`` must be set by a person after checking.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lbyr.arm.fw_planner import US_PER_DEG
from lbyr.config import JOINT_NAMES, NUM_JOINTS
from lbyr.errors import CalibrationError


@dataclass(frozen=True)
class JointCalib:
    name: str
    zero_us: int
    sign: int = 1
    zero_pose_deg: float = 0.0


@dataclass(frozen=True)
class JointsCalib:
    joints: tuple[JointCalib, ...]
    verified: bool
    us_per_deg: float = US_PER_DEG
    path: Path | None = None

    @property
    def zero_us(self) -> tuple[int, ...]:
        return tuple(j.zero_us for j in self.joints)

    def host_to_fw_deg(self, host_deg: Sequence[float]) -> tuple[float, ...]:
        _check_len(host_deg)
        return tuple(
            j.sign * (float(h) - j.zero_pose_deg)
            for j, h in zip(self.joints, host_deg, strict=True)
        )

    def fw_to_host_deg(self, fw_deg: Sequence[float]) -> tuple[float, ...]:
        _check_len(fw_deg)
        return tuple(
            j.zero_pose_deg + j.sign * float(f) for j, f in zip(self.joints, fw_deg, strict=True)
        )

    def us_to_host_deg(self, us: Sequence[float]) -> tuple[float, ...]:
        _check_len(us)
        fw = [
            (float(u) - j.zero_us) / self.us_per_deg for j, u in zip(self.joints, us, strict=True)
        ]
        return self.fw_to_host_deg(fw)

    def zero_mismatches(self, board_zero_us: Sequence[int], tol_us: int = 0) -> list[str]:
        """Human-readable lines for every joint whose board zero differs from this file."""
        out = []
        for i, (j, b) in enumerate(zip(self.joints, board_zero_us, strict=True)):
            if abs(int(b) - j.zero_us) > tol_us:
                out.append(f"joint {i + 1} ({j.name}): board {b} us, joints.yaml {j.zero_us} us")
        return out


def _check_len(v: Sequence[float]) -> None:
    if len(v) != NUM_JOINTS:
        raise ValueError(f"expected {NUM_JOINTS} joint values, got {len(v)}")


def joints_from_dict(data: Any, path: Path | None = None) -> JointsCalib:
    if not isinstance(data, dict) or not isinstance(data.get("joints"), list):
        raise CalibrationError(f"{path}: expected a mapping with a 'joints' list")
    rows = data["joints"]
    if len(rows) != NUM_JOINTS:
        raise CalibrationError(f"{path}: expected {NUM_JOINTS} joints, got {len(rows)}")
    joints = []
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise CalibrationError(f"{path}: joint {i + 1} must be a mapping")
        name = str(r.get("name", ""))
        if name != JOINT_NAMES[i]:
            raise CalibrationError(f"{path}: joint {i + 1} name {name!r} != {JOINT_NAMES[i]!r}")
        try:
            zero = int(r["zero_us"])
            sign = int(r.get("sign", 1))
            zpose = float(r.get("zero_pose_deg", 0.0))
        except (KeyError, TypeError, ValueError) as e:
            raise CalibrationError(f"{path}: joint {i + 1}: bad zero_us/sign/zero_pose_deg") from e
        if sign not in (1, -1):
            raise CalibrationError(f"{path}: joint {i + 1}: sign must be +1 or -1")
        if not 500 <= zero <= 2500:
            raise CalibrationError(f"{path}: joint {i + 1}: zero_us {zero} outside 500..2500")
        joints.append(JointCalib(name, zero, sign, zpose))
    verified = data.get("verified", False) is True
    upd = float(data.get("us_per_deg", US_PER_DEG))
    return JointsCalib(tuple(joints), verified, upd, path)


def load_joints(path: Path) -> JointsCalib:
    import yaml

    if not path.is_file():
        raise CalibrationError(f"{path} not found (seeded copy is in the repo: data/calib/)")
    return joints_from_dict(yaml.safe_load(path.read_text(encoding="utf-8")), path)


def mock_joints() -> JointsCalib:
    """SIMULATION-ONLY: centred zeros (1500 us) so +/-90 deg fits the firmware clamp."""
    return JointsCalib(tuple(JointCalib(n, 1500) for n in JOINT_NAMES), verified=True)
