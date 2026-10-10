"""Target providers: skill + recovery adjustments -> joint goals (host degrees).

The skill executor (the FSM supervisor) never computes joint targets itself. For every
skill it asks a :class:`TargetProvider`, so motion targets can come from:

- :class:`TaughtTargets` (``motion.mode = "taught"``, the demo path): taught joint poses
  from ``data/calib/poses.yaml``; the object's zone is the zone polygon that contains its
  overhead detection centroid. No depth, no hand-eye, no link lengths.
- :class:`IKTargets` (``motion.mode = "ik"``): D435 depth -> base frame (hand-eye
  ``base_T_cam``) -> top-down TCP pose -> IK. Needs OQ-4 (link lengths / limits) and the
  hand-eye calibration; on the real arm this is the next task.

A provider is stateless across episodes. Per episode the supervisor calls:
``scene`` (PERCEIVE) -> ``bind`` (PLAN) -> ``approach/lift/transport/place`` goals.
Each goal list is one motion segment per entry (one ``m`` each, PP-1). Errors:
:class:`~lbyr.errors.PerceptionError` (-> PERCEPTION_FAILED) and
:class:`~lbyr.errors.IKError` (-> IK_FAILED, also raised for a missing taught pose / zone).
Cost: pure NumPy, well under 1 ms per call (IK mode: one IK solve per pose, < 2 ms).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lbyr.config import GRIPPER_INDEX, AppConfig
from lbyr.errors import IKError, PerceptionError
from lbyr.motion.ik import CartesianTarget, IKSolver
from lbyr.motion.taught import TaughtExecutor, Zone
from lbyr.perception.detector import Detection
from lbyr.perception.geometry import locate_in_base
from lbyr.perception.overhead import RGBDFrame


@dataclass
class GraspAdjustments:
    """Accumulated recovery deltas, applied on top of the planner's skill parameters."""

    closure_delta_pct: float = 0.0
    wrist_roll_deg: float = 0.0
    lateral_offset_mm: float = 0.0
    # EMPTY recovery "-5 mm standoff": IK mode = grasp 5 mm lower (D-8); taught mode = the
    # pre-grasp pose moves toward the grasp pose (lbyr.motion.taught).
    grasp_z_offset_mm: float = 0.0


@dataclass(frozen=True)
class Scene:
    """What PERCEIVE found: the detections the planner may refer to (by id) and, in IK
    mode, their base-frame positions (mm)."""

    detections: dict[int, Detection]
    positions_mm: dict[int, np.ndarray] = field(default_factory=dict)
    summary: str = ""


@dataclass
class Targets:
    """The planner's pick / place choice resolved by the provider (PLAN)."""

    pick: Detection
    place: Detection
    standoff_mm: float
    zone: Zone | None = None  # taught: zone of the pick object
    pick_xyz_mm: np.ndarray | None = None  # ik: base frame
    place_xyz_mm: np.ndarray | None = None
    grasp_xyz_mm: np.ndarray | None = None  # ik: set by approach_goals, used by lift/place


@dataclass(frozen=True)
class GoalPlan:
    goals: list[np.ndarray]  # one motion segment each, host degrees (6 values)
    info: dict[str, Any]  # logged as the record's ``target``


def with_gripper(q: Sequence[float] | np.ndarray, gripper_deg: float) -> np.ndarray:
    out = np.asarray(q, dtype=float).copy()
    out[GRIPPER_INDEX] = gripper_deg
    return out


class TargetProvider(ABC):
    name: str = ""

    @abstractmethod
    def scene(self, detections: Sequence[Detection], frame: RGBDFrame | None) -> Scene:
        """PERCEIVE: detections -> Scene. Raises PerceptionError."""

    @abstractmethod
    def bind(self, scene: Scene, pick_id: int, place_id: int, standoff_mm: float) -> Targets:
        """PLAN: resolve the planner's ids. Raises IKError (no zone / pose / position)."""

    @abstractmethod
    def gripper_deg(self, closure_pct: float) -> float: ...

    @abstractmethod
    def approach_goals(
        self, t: Targets, current: np.ndarray, adj: GraspAdjustments
    ) -> GoalPlan: ...

    def grasp_goals(self, current: np.ndarray, closure_pct: float) -> GoalPlan:
        """Close the gripper in place (same for every provider)."""
        goal = with_gripper(current, self.gripper_deg(closure_pct))
        return GoalPlan([goal], {"closure_pct": closure_pct})

    @abstractmethod
    def lift_goals(
        self, t: Targets, current: np.ndarray, closure_pct: float, adj: GraspAdjustments
    ) -> GoalPlan: ...

    @abstractmethod
    def transport_goals(
        self, t: Targets, current: np.ndarray, closure_pct: float, adj: GraspAdjustments
    ) -> GoalPlan: ...

    @abstractmethod
    def place_goals(
        self, t: Targets, current: np.ndarray, closure_pct: float, adj: GraspAdjustments
    ) -> GoalPlan: ...

    @abstractmethod
    def safe_pose(self, current: np.ndarray) -> np.ndarray | None:
        """ABORT retreat pose keeping the current gripper value; None = unknown (hold)."""


# ---------------------------------------------------------------------------- taught
class TaughtTargets(TargetProvider):
    """Taught joint poses (demo path). See :mod:`lbyr.motion.taught` for the mapping."""

    name = "taught"

    def __init__(self, executor: TaughtExecutor) -> None:
        self.executor = executor

    def scene(self, detections: Sequence[Detection], frame: RGBDFrame | None) -> Scene:
        dets = {d.id: d for d in detections}
        if not any("zone" not in d.label for d in detections):
            raise PerceptionError("no object detected")
        if not any("zone" in d.label for d in detections):
            place = self.executor.place_detection()  # planner place target
            dets[place.id] = place
        return Scene(dets, summary=f"{len(detections)} objects detected")

    def bind(self, scene: Scene, pick_id: int, place_id: int, standoff_mm: float) -> Targets:
        pick = scene.detections[pick_id]
        zone = self.executor.zone_for(pick)  # TaughtPoseError (IKError): in no zone
        return Targets(pick, scene.detections[place_id], standoff_mm, zone=zone)

    def gripper_deg(self, closure_pct: float) -> float:
        return self.executor.poses.gripper_deg(closure_pct)

    def _plan(self, t: Targets, goals: list[np.ndarray]) -> GoalPlan:
        assert t.zone is not None
        return GoalPlan(goals, {"zone": t.zone.name, "joints_deg": goals[-1]})

    def approach_goals(self, t: Targets, current: np.ndarray, adj: GraspAdjustments) -> GoalPlan:
        assert t.zone is not None
        goals = self.executor.approach_goals(
            t.zone, current, adj.wrist_roll_deg, adj.lateral_offset_mm, adj.grasp_z_offset_mm
        )
        return self._plan(t, goals)

    def lift_goals(
        self, t: Targets, current: np.ndarray, closure_pct: float, adj: GraspAdjustments
    ) -> GoalPlan:
        assert t.zone is not None
        goals = self.executor.lift_goals(
            t.zone, closure_pct, adj.wrist_roll_deg, adj.lateral_offset_mm
        )
        return self._plan(t, goals)

    def transport_goals(
        self, t: Targets, current: np.ndarray, closure_pct: float, adj: GraspAdjustments
    ) -> GoalPlan:
        return self._plan(t, self.executor.transport_goals(closure_pct))

    def place_goals(
        self, t: Targets, current: np.ndarray, closure_pct: float, adj: GraspAdjustments
    ) -> GoalPlan:
        return self._plan(t, self.executor.place_goals(closure_pct))

    def safe_pose(self, current: np.ndarray) -> np.ndarray | None:
        try:  # taught 'home' pose (A-3 decided per poses.yaml)
            return self.executor.safe_pose(float(current[GRIPPER_INDEX]))
        except IKError:
            return None


# ---------------------------------------------------------------------------- IK
class IKTargets(TargetProvider):
    """Cartesian targets from the D435 (base frame via ``base_T_cam``) + IK.

    Grasp point = object centre at mid-height (+ lateral / z recovery offsets); pre-grasp =
    grasp + the planner's ``standoff_mm`` straight up; top-down TCP, wrist roll from the
    recovery adjustments; IK is seeded with the current setpoint.
    """

    name = "ik"

    def __init__(self, cfg: AppConfig, ik: IKSolver, base_T_cam: np.ndarray) -> None:
        self.cfg = cfg
        self.ik = ik
        self.base_T_cam = base_T_cam

    def scene(self, detections: Sequence[Detection], frame: RGBDFrame | None) -> Scene:
        if frame is None:
            raise PerceptionError("IK mode needs the overhead depth frame")
        dets: dict[int, Detection] = {}
        pos: dict[int, np.ndarray] = {}
        for det in detections:
            u, v = det.centroid
            p = locate_in_base(
                u,
                v,
                frame.depth_mm,
                frame.intrinsics,
                self.base_T_cam,
                self.cfg.cameras.overhead.depth_median_window,
            )
            if p is not None:
                dets[det.id], pos[det.id] = det, p
        if not pos:
            raise PerceptionError("no detections with valid depth")
        return Scene(dets, pos, summary=f"{len(pos)} objects localised")

    def bind(self, scene: Scene, pick_id: int, place_id: int, standoff_mm: float) -> Targets:
        return Targets(
            scene.detections[pick_id],
            scene.detections[place_id],
            standoff_mm,
            pick_xyz_mm=scene.positions_mm[pick_id],
            place_xyz_mm=scene.positions_mm[place_id],
        )

    def gripper_deg(self, closure_pct: float) -> float:
        return self.cfg.arm.gripper.closure_to_deg(closure_pct)

    def _pose(
        self, xyz: np.ndarray, gripper_deg: float, current: np.ndarray, adj: GraspAdjustments
    ) -> np.ndarray:
        """IK for a top-down TCP pose + gripper -> 6-joint goal (deg)."""
        target = CartesianTarget((float(xyz[0]), float(xyz[1]), float(xyz[2])), adj.wrist_roll_deg)
        q = self.ik.solve(target, np.asarray(current, dtype=float)[:GRIPPER_INDEX].tolist())
        return np.array([*q, gripper_deg], dtype=float)

    def _plan(
        self,
        poses: list[tuple[np.ndarray, float]],
        current: np.ndarray,
        adj: GraspAdjustments,
    ) -> GoalPlan:
        goals = [self._pose(xyz, g, current, adj) for xyz, g in poses]
        return GoalPlan(goals, {"xyz_mm": poses[-1][0]})

    def approach_goals(self, t: Targets, current: np.ndarray, adj: GraspAdjustments) -> GoalPlan:
        obj = t.pick_xyz_mm
        assert obj is not None
        # mid-height of an object resting on the mat
        grasp = np.array(
            [obj[0], obj[1] + adj.lateral_offset_mm, obj[2] / 2.0 + adj.grasp_z_offset_mm]
        )
        pre = grasp + [0.0, 0.0, t.standoff_mm]
        open_deg = self.gripper_deg(0.0)
        goals = [
            with_gripper(current, open_deg),
            self._pose(pre, open_deg, current, adj),
            self._pose(grasp, open_deg, current, adj),
        ]
        t.grasp_xyz_mm = grasp
        return GoalPlan(goals, {"xyz_mm": grasp})

    def lift_goals(
        self, t: Targets, current: np.ndarray, closure_pct: float, adj: GraspAdjustments
    ) -> GoalPlan:
        assert t.grasp_xyz_mm is not None
        up = t.grasp_xyz_mm + [0.0, 0.0, self.cfg.skills.lift_height_mm]
        return self._plan([(up, self.gripper_deg(closure_pct))], current, adj)

    def transport_goals(
        self, t: Targets, current: np.ndarray, closure_pct: float, adj: GraspAdjustments
    ) -> GoalPlan:
        p = t.place_xyz_mm
        assert p is not None
        xyz = np.array([p[0], p[1], self.cfg.skills.transport_height_mm])
        return self._plan([(xyz, self.gripper_deg(closure_pct))], current, adj)

    def place_goals(
        self, t: Targets, current: np.ndarray, closure_pct: float, adj: GraspAdjustments
    ) -> GoalPlan:
        p, g = t.place_xyz_mm, t.grasp_xyz_mm
        assert p is not None and g is not None
        release = np.array([p[0], p[1], p[2] + g[2] + self.cfg.skills.place_release_height_mm])
        retreat = release + [0.0, 0.0, self.cfg.skills.lift_height_mm]
        closed, opened = self.gripper_deg(closure_pct), self.gripper_deg(0.0)
        return self._plan([(release, closed), (release, opened), (retreat, opened)], current, adj)

    def safe_pose(self, current: np.ndarray) -> np.ndarray | None:
        safe = self.cfg.arm.safe_pose_deg
        if safe is None:
            return None
        return with_gripper(safe, float(current[GRIPPER_INDEX]))  # do not drop a held object
