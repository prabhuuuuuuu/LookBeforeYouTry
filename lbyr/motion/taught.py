"""Taught-pose mode (demo path): skills -> taught joint poses, no link lengths / hand-eye.

``data/calib/poses.yaml`` holds named poses in ZERO-RELATIVE degrees (what the
tools/arm_gui.py sliders show; base -> gripper) and the pick zones::

    gripper: {open_deg: .., closed_deg: ..}       # joint 6, zero-relative
    poses:
      home: [...]                 # ABORT safe pose
      place_pre: [...]            # above the place location
      place: [...]                # release height
      zone_A_pre: [...]           # above zone A
      zone_A_grasp: [...]         # grasp height in zone A
    zones:
      A: {polygon_px: [[u, v], ...], radius_mm: 220, standoff_mm: 40}
    place_polygon_px: [[u, v], ...]   # optional

Skill mapping (one motion segment = one ``m``):

- APPROACH: open the gripper in place -> zone_X_pre -> zone_X_grasp (gripper open)
- GRASP: close to the planned closure % (joint 6 between open_deg and closed_deg)
- LIFT: back to zone_X_pre (closed);  TRANSPORT: place_pre;  PLACE: place, open, place_pre
- ABORT: home, keeping the current gripper value (a held object is not dropped)

Joint 6 of every arm pose is ignored: the gripper follows the skill. The object's zone is
the zone polygon (overhead image pixels) that contains the detection centroid. The
planner's ``standoff_mm`` is not used: the taught poses define the standoff.

Recovery mapping (bible 13 ladders, taught-mode interpretation):

- "+15% closure"         -> joint 6: closure % + 15 (capped at 100), via open/closed degrees
- "wrist re-orient 30"   -> joint 5 (wrist roll) + 30 deg on the zone poses
- "lateral offset 10 mm" -> joint 1 (base yaw) + atan(10 / radius_mm) on the zone poses
- "-5 mm standoff"       -> the pre-grasp pose moves along the JOINT-SPACE segment
  zone_X_pre -> zone_X_grasp by f = 5 / standoff_mm (standoff_mm per zone, default
  SkillConfig.default_standoff_mm). Joint-space interpolation is not a straight Cartesian
  line, and the grasp pose itself is unchanged: this only shortens the final approach.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from lbyr.arm.calib import JointsCalib
from lbyr.config import GRIPPER_INDEX, NUM_JOINTS, WRIST_ROLL_INDEX
from lbyr.errors import CalibrationError, IKError
from lbyr.perception.detector import Detection

BASE_YAW_INDEX = 0
REQUIRED_POSES: tuple[str, ...] = ("home", "place_pre", "place")
PLACE_ZONE_ID = 900  # synthetic detection id for the taught place location
PLACE_ZONE_LABEL = "place_zone"

POSES_HEADER = """\
# Taught poses for motion.mode = taught (lbyr/motion/taught.py). Written by
# tools/arm_gui.py "Save pose as..." (comments below this header are not preserved).
# Angles: ZERO-RELATIVE degrees per joint, base -> gripper (the arm_gui slider values).
# Joint 6 of arm poses is ignored; the gripper uses gripper.open_deg / closed_deg.
# Required: poses home, place_pre, place; per zone X: zone_X_pre, zone_X_grasp and
# zones.X.polygon_px (overhead image pixels, >= 3 points). radius_mm (base axis -> zone
# centre, measured) is needed only for the lateral-offset recovery.
"""


class TaughtPoseError(IKError):
    """A taught pose / zone is missing or the target is in no zone (-> IK_FAILED)."""


def point_in_polygon(u: float, v: float, poly: Sequence[tuple[float, float]]) -> bool:
    """Even-odd ray casting; points on an edge count as inside."""
    inside = False
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (x1, y1) == (u, v):
            return True
        if (y1 > v) != (y2 > v):
            x = x1 + (v - y1) * (x2 - x1) / (y2 - y1)
            if abs(x - u) < 1e-9:
                return True
            if x > u:
                inside = not inside
    return inside


@dataclass(frozen=True)
class Zone:
    name: str
    polygon_px: tuple[tuple[float, float], ...]
    radius_mm: float | None = None
    standoff_mm: float | None = None

    def contains(self, u: float, v: float) -> bool:
        return len(self.polygon_px) >= 3 and point_in_polygon(u, v, self.polygon_px)


@dataclass(frozen=True)
class TaughtPoses:
    poses: dict[str, tuple[float, ...]]  # HOST degrees (converted with joints.yaml)
    zones: dict[str, Zone]
    gripper_open_deg: float | None  # host degrees of joint 6
    gripper_closed_deg: float | None
    place_polygon_px: tuple[tuple[float, float], ...] | None = None
    path: Path | None = field(default=None, compare=False)

    def missing(self) -> list[str]:
        """Everything taught mode still needs (empty = ready)."""
        out = [f"pose '{p}'" for p in REQUIRED_POSES if p not in self.poses]
        if self.gripper_open_deg is None or self.gripper_closed_deg is None:
            out.append("gripper.open_deg / gripper.closed_deg")
        if not self.zones:
            out.append("at least one zone (zones.<X>.polygon_px)")
        for z in self.zones.values():
            out += [
                f"pose 'zone_{z.name}_{w}'"
                for w in ("pre", "grasp")
                if f"zone_{z.name}_{w}" not in self.poses
            ]
            if len(z.polygon_px) < 3:
                out.append(f"zones.{z.name}.polygon_px (>= 3 points)")
        return out

    def pose(self, name: str) -> np.ndarray:
        if name not in self.poses:
            raise TaughtPoseError(f"taught pose '{name}' missing in poses.yaml")
        return np.asarray(self.poses[name], dtype=float)

    def zone_at(self, u: float, v: float) -> Zone | None:
        for z in self.zones.values():
            if z.contains(u, v):
                return z
        return None

    def gripper_deg(self, closure_pct: float) -> float:
        if self.gripper_open_deg is None or self.gripper_closed_deg is None:
            raise TaughtPoseError("gripper.open_deg / closed_deg missing in poses.yaml")
        c = min(max(closure_pct, 0.0), 100.0) / 100.0
        return self.gripper_open_deg + c * (self.gripper_closed_deg - self.gripper_open_deg)


# ------------------------------------------------------------------ file I/O
def _polygon(v: Any, what: str) -> tuple[tuple[float, float], ...]:
    if v is None:
        return ()
    try:
        pts = tuple((float(p[0]), float(p[1])) for p in v)
    except (TypeError, ValueError, IndexError) as e:
        raise CalibrationError(f"{what}: expected [[u, v], ...]") from e
    return pts


def _opt_float(v: Any, what: str) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError) as e:
        raise CalibrationError(f"{what}: expected a number") from e


def poses_from_dict(data: Any, joints: JointsCalib, path: Path | None = None) -> TaughtPoses:
    """Parse poses.yaml content; converts zero-relative degrees to host degrees."""
    data = data or {}
    if not isinstance(data, dict):
        raise CalibrationError(f"{path}: expected a mapping")
    poses: dict[str, tuple[float, ...]] = {}
    for name, vals in (data.get("poses") or {}).items():
        if not isinstance(vals, list) or len(vals) != NUM_JOINTS:
            raise CalibrationError(f"{path}: pose '{name}' needs {NUM_JOINTS} values")
        try:
            fw = [float(x) for x in vals]
        except (TypeError, ValueError) as e:
            raise CalibrationError(f"{path}: pose '{name}' has a non-number") from e
        poses[str(name)] = joints.fw_to_host_deg(fw)
    zones: dict[str, Zone] = {}
    for name, z in (data.get("zones") or {}).items():
        if not isinstance(z, dict):
            raise CalibrationError(f"{path}: zone '{name}' must be a mapping")
        zones[str(name)] = Zone(
            str(name),
            _polygon(z.get("polygon_px"), f"{path}: zones.{name}.polygon_px"),
            _opt_float(z.get("radius_mm"), f"zones.{name}.radius_mm"),
            _opt_float(z.get("standoff_mm"), f"zones.{name}.standoff_mm"),
        )
    g = data.get("gripper") or {}
    g6 = joints.joints[GRIPPER_INDEX]

    def _g(key: str) -> float | None:
        v = _opt_float(g.get(key), f"gripper.{key}")
        return None if v is None else g6.zero_pose_deg + g6.sign * v

    return TaughtPoses(
        poses,
        zones,
        _g("open_deg"),
        _g("closed_deg"),
        _polygon(data.get("place_polygon_px"), "place_polygon_px") or None,
        path,
    )


def load_poses(path: Path, joints: JointsCalib) -> TaughtPoses:
    import yaml

    if not path.is_file():
        raise CalibrationError(f"{path} not found (teach poses with tools/arm_gui.py)")
    return poses_from_dict(yaml.safe_load(path.read_text(encoding="utf-8")), joints, path)


def save_pose(path: Path, name: str, fw_deg: Sequence[float]) -> None:
    """Add/replace pose ``name`` (zero-relative degrees) in poses.yaml, keeping the rest.
    Used by tools/arm_gui.py "Save pose as..."."""
    import yaml

    if not name or not all(c.isalnum() or c == "_" for c in name):
        raise ValueError("pose name: letters, digits and '_' only (e.g. zone_A_pre)")
    if len(fw_deg) != NUM_JOINTS:
        raise ValueError(f"expected {NUM_JOINTS} values")
    data: dict[str, Any] = {}
    if path.is_file():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data.setdefault("gripper", {"open_deg": None, "closed_deg": None})
    poses = data.get("poses") or {}
    poses[name] = [round(float(v), 1) for v in fw_deg]
    data["poses"] = poses
    data.setdefault("zones", {})
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(data, sort_keys=False, default_flow_style=None)
    path.write_text(POSES_HEADER + body, encoding="utf-8")


# ------------------------------------------------------------------ executor
class TaughtExecutor:
    """Maps skills + recovery adjustments to joint goals (host degrees). Pure functions of
    the taught poses; raises :class:`TaughtPoseError` (-> IK_FAILED) when something is
    missing."""

    def __init__(self, poses: TaughtPoses, default_standoff_mm: float = 40.0) -> None:
        self.poses = poses
        self.default_standoff_mm = default_standoff_mm

    # ---- perception -> zone
    def zone_for(self, det: Detection) -> Zone:
        u, v = det.centroid
        zone = self.poses.zone_at(u, v)
        if zone is None:
            raise TaughtPoseError(
                f"{det.label} at pixel ({u:.0f}, {v:.0f}) is in no taught zone "
                f"({', '.join(self.poses.zones) or 'none defined'})"
            )
        return zone

    def place_detection(self) -> Detection:
        """Synthetic detection for the taught place location (planner place target)."""
        poly = self.poses.place_polygon_px
        if poly:
            us, vs = [p[0] for p in poly], [p[1] for p in poly]
            bbox = (min(us), min(vs), max(us), max(vs))
        else:
            bbox = (0.0, 0.0, 0.0, 0.0)
        return Detection(PLACE_ZONE_ID, PLACE_ZONE_LABEL, 1.0, bbox)

    # ---- recovery mapping
    def zone_pose(
        self,
        zone: Zone,
        which: str,
        wrist_roll_deg: float = 0.0,
        lateral_offset_mm: float = 0.0,
    ) -> np.ndarray:
        q = self.poses.pose(f"zone_{zone.name}_{which}").copy()
        q[WRIST_ROLL_INDEX] += wrist_roll_deg  # "wrist re-orient" -> joint 5
        if lateral_offset_mm:
            if not zone.radius_mm:
                raise TaughtPoseError(
                    f"zones.{zone.name}.radius_mm not set (the lateral-offset recovery needs it)"
                )
            q[BASE_YAW_INDEX] += math.degrees(math.atan(lateral_offset_mm / zone.radius_mm))
        return q

    def pre_grasp(
        self,
        zone: Zone,
        wrist_roll_deg: float = 0.0,
        lateral_offset_mm: float = 0.0,
        standoff_delta_mm: float = 0.0,
    ) -> np.ndarray:
        """zone_X_pre, moved toward zone_X_grasp by f = -standoff_delta / standoff."""
        pre = self.zone_pose(zone, "pre", wrist_roll_deg, lateral_offset_mm)
        grasp = self.zone_pose(zone, "grasp", wrist_roll_deg, lateral_offset_mm)
        standoff = zone.standoff_mm or self.default_standoff_mm
        f = min(max(-standoff_delta_mm / standoff, 0.0), 1.0)
        return pre + f * (grasp - pre)

    def _with_gripper(self, q: np.ndarray, gripper_deg: float) -> np.ndarray:
        q = q.copy()
        q[GRIPPER_INDEX] = gripper_deg
        return q

    # ---- skills
    def approach_goals(
        self,
        zone: Zone,
        current: Sequence[float] | np.ndarray,
        wrist_roll_deg: float = 0.0,
        lateral_offset_mm: float = 0.0,
        standoff_delta_mm: float = 0.0,
    ) -> list[np.ndarray]:
        open_deg = self.poses.gripper_deg(0.0)
        here = self._with_gripper(np.asarray(current, dtype=float), open_deg)
        pre = self.pre_grasp(zone, wrist_roll_deg, lateral_offset_mm, standoff_delta_mm)
        grasp = self.zone_pose(zone, "grasp", wrist_roll_deg, lateral_offset_mm)
        return [here, self._with_gripper(pre, open_deg), self._with_gripper(grasp, open_deg)]

    def lift_goals(
        self,
        zone: Zone,
        closure_pct: float,
        wrist_roll_deg: float = 0.0,
        lateral_offset_mm: float = 0.0,
    ) -> list[np.ndarray]:
        pre = self.zone_pose(zone, "pre", wrist_roll_deg, lateral_offset_mm)
        return [self._with_gripper(pre, self.poses.gripper_deg(closure_pct))]

    def transport_goals(self, closure_pct: float) -> list[np.ndarray]:
        g = self.poses.gripper_deg(closure_pct)
        return [self._with_gripper(self.poses.pose("place_pre"), g)]

    def place_goals(self, closure_pct: float) -> list[np.ndarray]:
        closed, opened = self.poses.gripper_deg(closure_pct), self.poses.gripper_deg(0.0)
        place, pre = self.poses.pose("place"), self.poses.pose("place_pre")
        return [
            self._with_gripper(place, closed),
            self._with_gripper(place, opened),
            self._with_gripper(pre, opened),
        ]

    def safe_pose(self, current_gripper_deg: float) -> np.ndarray:
        return self._with_gripper(self.poses.pose("home"), current_gripper_deg)


def mock_taught_poses(dets: Sequence[Detection]) -> TaughtPoses:
    """SIMULATION-ONLY poses (FAKE, host degrees) for mock mode: zone A around every
    non-zone fake detection; gripper 0..90 deg like mock_profile's GripperConfig."""
    objs = [d for d in dets if "zone" not in d.label]
    us = [c for d in objs for c in (d.bbox_xyxy[0], d.bbox_xyxy[2])] or [0.0, 640.0]
    vs = [c for d in objs for c in (d.bbox_xyxy[1], d.bbox_xyxy[3])] or [0.0, 480.0]
    x1, y1, x2, y2 = min(us) - 20, min(vs) - 20, max(us) + 20, max(vs) + 20
    fake: dict[str, tuple[float, ...]] = {
        "home": (0.0, -30.0, 90.0, 30.0, 0.0, 0.0),
        "zone_A_pre": (15.0, 10.0, 60.0, 40.0, 0.0, 0.0),
        "zone_A_grasp": (15.0, 25.0, 55.0, 50.0, 0.0, 0.0),
        "place_pre": (-25.0, 10.0, 60.0, 40.0, 0.0, 0.0),
        "place": (-25.0, 25.0, 55.0, 50.0, 0.0, 0.0),
    }
    zone = Zone("A", ((x1, y1), (x2, y1), (x2, y2), (x1, y2)), radius_mm=220.0)
    return TaughtPoses(fake, {"A": zone}, 0.0, 90.0)
