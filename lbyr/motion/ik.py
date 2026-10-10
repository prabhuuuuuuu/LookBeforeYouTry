"""Inverse kinematics (bible 11.4: ikpy numerical IK, <2 ms).

Cartesian targets are in the base frame, in mm. Output joints are degrees for the 5 arm
joints (base yaw, shoulder pitch, elbow pitch, wrist pitch, wrist roll); the gripper is
commanded separately via ``GripperConfig.closure_to_deg``.

- :class:`IkpyIK` is the real solver. It refuses to build until OQ-4 (URDF / link
  lengths) is answered. ``ikpy`` is imported lazily.
- :class:`MockIK` is an analytic solver for the FAKE geometry in ``mock_profile()``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from lbyr.config import NUM_JOINTS, ArmConfig, require
from lbyr.errors import ConfigError, HardwareNotConfirmedError, IKError

NUM_ARM_JOINTS = NUM_JOINTS - 1


@dataclass(frozen=True)
class CartesianTarget:
    xyz_mm: tuple[float, float, float]  # tool-centre point, base frame
    wrist_roll_deg: float = 0.0  # top-down grasp; roll about the approach axis


class IKSolver(Protocol):
    def solve(self, target: CartesianTarget, seed_deg: Sequence[float]) -> tuple[float, ...]:
        """Return 5 arm joint angles in degrees, or raise IKError."""
        ...


def _check_limits(joints: Sequence[float], cfg: ArmConfig) -> None:
    lo, hi = cfg.limits()
    for i, q in enumerate(joints):
        if not lo[i] <= q <= hi[i]:
            raise IKError(f"joint {i} = {q:.1f} deg outside [{lo[i]}, {hi[i]}]")


class IkpyIK:
    """ikpy chain built from the official URDF (preferred) once OQ-4 is answered."""

    def __init__(self, cfg: ArmConfig, urdf_path: Path | None = None) -> None:
        if cfg.simulated:
            raise ConfigError("IkpyIK refuses a SIMULATION-ONLY config (mock_profile)")
        cfg.limits()  # OQ-4 joint limits
        if urdf_path is None:
            require(cfg.link_lengths_mm, "link lengths", "OQ-4")
            # Link lengths alone do not fix joint axes, zero pose and sign conventions.
            raise HardwareNotConfirmedError(
                "joint axes / zero-pose convention for a link-length-only chain (prefer URDF)",
                "OQ-4",
            )
        from ikpy.chain import Chain  # lazy

        self._cfg = cfg
        self._chain: Any = Chain.from_urdf_file(str(urdf_path))
        # TODO(OQ-4): verify the URDF active-joint order matches config.JOINT_NAMES[:5].

    def solve(self, target: CartesianTarget, seed_deg: Sequence[float]) -> tuple[float, ...]:
        active = [i for i, a in enumerate(self._chain.active_links_mask) if a]
        if len(active) < NUM_ARM_JOINTS:
            raise IKError("URDF chain has fewer than 5 active joints")
        init = np.zeros(len(self._chain.links))
        for k, i in enumerate(active[:NUM_ARM_JOINTS]):
            init[i] = math.radians(seed_deg[k])
        sol = self._chain.inverse_kinematics(
            target_position=np.asarray(target.xyz_mm) / 1000.0,
            target_orientation=[0.0, 0.0, -1.0],
            orientation_mode="Z",
            initial_position=init,
        )
        q = [math.degrees(sol[i]) for i in active[:NUM_ARM_JOINTS]]
        q[4] = target.wrist_roll_deg
        _check_limits(q, self._cfg)
        return tuple(q)


class MockIK:
    """Analytic top-down IK for the FAKE mock geometry (SIMULATION ONLY).

    Fake convention: shoulder/elbow/wrist pitch measured from vertical; tool points down
    when shoulder + elbow + wrist = 180 deg. link_lengths_mm = (base_h, upper, fore, tool).
    """

    def __init__(self, cfg: ArmConfig) -> None:
        if not cfg.simulated:
            raise ConfigError("MockIK is only valid with the SIMULATION-ONLY mock profile")
        self._cfg = cfg
        base_h, l1, l2, tool = require(cfg.link_lengths_mm, "link lengths", "OQ-4")
        self._base_h, self._l1, self._l2, self._tool = base_h, l1, l2, tool

    def solve(self, target: CartesianTarget, seed_deg: Sequence[float]) -> tuple[float, ...]:
        x, y, z = target.xyz_mm
        yaw = math.degrees(math.atan2(y, x))
        dx = math.hypot(x, y)
        dz = z + self._tool - self._base_h
        l1, l2 = self._l1, self._l2
        c = (dx * dx + dz * dz - l1 * l1 - l2 * l2) / (2 * l1 * l2)
        if not -1.0 <= c <= 1.0:
            raise IKError(f"target {target.xyz_mm} out of reach")
        elbow = math.acos(c)
        shoulder = math.atan2(dx, dz) - math.atan2(l2 * math.sin(elbow), l1 + l2 * math.cos(elbow))
        sh, el = math.degrees(shoulder), math.degrees(elbow)
        q = (yaw, sh, el, 180.0 - sh - el, target.wrist_roll_deg)
        _check_limits(q, self._cfg)
        return q

    def forward(self, joints_deg: Sequence[float]) -> tuple[float, float, float]:
        yaw, sh, el, wr = (math.radians(j) for j in joints_deg[:4])
        r = (
            self._l1 * math.sin(sh)
            + self._l2 * math.sin(sh + el)
            + self._tool * math.sin(sh + el + wr)
        )
        z = (
            self._base_h
            + self._l1 * math.cos(sh)
            + self._l2 * math.cos(sh + el)
            + self._tool * math.cos(sh + el + wr)
        )
        return r * math.cos(yaw), r * math.sin(yaw), z
