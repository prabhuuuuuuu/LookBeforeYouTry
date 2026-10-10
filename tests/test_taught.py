"""Taught-pose mode: zone lookup, recovery mapping, poses.yaml I/O, FSM integration."""

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from lbyr.ai.critic import Verdict
from lbyr.arm.calib import JointCalib, JointsCalib, mock_joints
from lbyr.config import GRIPPER_INDEX, JOINT_NAMES, WRIST_ROLL_INDEX, mock_profile
from lbyr.errors import CalibrationError
from lbyr.fsm.recovery import FailureClass
from lbyr.fsm.states import State
from lbyr.fsm.supervisor import EpisodeSpec
from lbyr.mock_stack import build_mock_stack
from lbyr.motion.taught import (
    PLACE_ZONE_ID,
    TaughtExecutor,
    TaughtPoseError,
    TaughtPoses,
    Zone,
    load_poses,
    point_in_polygon,
    poses_from_dict,
    save_pose,
)
from lbyr.perception.detector import Detection
from lbyr.perception.overhead import FakeObject, FakeScene

ROOT = Path(__file__).parents[1]
SQUARE = ((100.0, 300.0), (220.0, 300.0), (220.0, 420.0), (100.0, 420.0))
POSES: dict[str, tuple[float, ...]] = {
    "home": (0, 0, 0, 0, 0, 0),
    "zone_A_pre": (10, -20, 30, 10, 0, 0),
    "zone_A_grasp": (10, -40, 40, 30, 0, 0),
    "zone_B_pre": (-10, -20, 30, 10, 0, 0),
    "zone_B_grasp": (-10, -40, 40, 30, 0, 0),
    "place_pre": (-40, -20, 30, 10, 0, 0),
    "place": (-40, -30, 35, 20, 0, 0),
}


def taught(radius: float | None = 200.0) -> TaughtExecutor:
    zones = {
        "A": Zone("A", SQUARE, radius_mm=radius, standoff_mm=40.0),
        "B": Zone("B", ((300, 300), (400, 300), (350, 400)), radius_mm=None),
    }
    return TaughtExecutor(TaughtPoses(POSES, zones, 0.0, 60.0), default_standoff_mm=40.0)


# ------------------------------------------------------------------ zone lookup
def test_point_in_polygon() -> None:
    assert point_in_polygon(150, 350, SQUARE)
    assert point_in_polygon(100, 350, SQUARE)  # on the edge
    assert not point_in_polygon(99, 350, SQUARE)
    tri = ((300, 300), (400, 300), (350, 400))
    assert point_in_polygon(350, 320, tri) and not point_in_polygon(310, 390, tri)


def test_zone_by_detection_centroid() -> None:
    t = taught()
    assert t.zone_for(Detection(1, "wooden_block", 0.9, (140, 330, 180, 370))).name == "A"
    assert t.zone_for(Detection(1, "wooden_block", 0.9, (340, 300, 360, 340))).name == "B"
    with pytest.raises(TaughtPoseError, match="in no taught zone"):
        t.zone_for(Detection(1, "wooden_block", 0.9, (0, 0, 10, 10)))
    place = t.place_detection()
    assert place.id == PLACE_ZONE_ID and "zone" in place.label


# ------------------------------------------------------------------ recovery mapping
def test_closure_maps_to_joint_6() -> None:
    p = taught().poses
    assert p.gripper_deg(0) == 0 and p.gripper_deg(100) == 60
    assert p.gripper_deg(80 + 15) == pytest.approx(57.0)  # "+15% closure"
    assert p.gripper_deg(150) == 60  # capped


def test_wrist_reorient_is_joint_5() -> None:
    t = taught()
    z = t.poses.zones["A"]
    q = t.zone_pose(z, "grasp", wrist_roll_deg=30.0)
    expected = np.asarray(POSES["zone_A_grasp"], dtype=float)
    expected[WRIST_ROLL_INDEX] += 30.0
    np.testing.assert_allclose(q, expected)


def test_lateral_offset_is_base_yaw_atan() -> None:
    t = taught(radius=200.0)
    q = t.zone_pose(t.poses.zones["A"], "pre", lateral_offset_mm=10.0)
    assert q[0] == pytest.approx(10.0 + math.degrees(math.atan(10.0 / 200.0)))  # ~2.86 deg
    np.testing.assert_allclose(q[1:], POSES["zone_A_pre"][1:])
    with pytest.raises(TaughtPoseError, match="radius_mm"):
        t.zone_pose(t.poses.zones["B"], "pre", lateral_offset_mm=10.0)


def test_minus_5mm_standoff_interpolates_pre_toward_grasp() -> None:
    t = taught()
    z = t.poses.zones["A"]
    pre = np.asarray(POSES["zone_A_pre"], dtype=float)
    grasp = np.asarray(POSES["zone_A_grasp"], dtype=float)
    np.testing.assert_allclose(t.pre_grasp(z), pre)
    np.testing.assert_allclose(t.pre_grasp(z, standoff_delta_mm=-5.0), pre + 5 / 40 * (grasp - pre))
    np.testing.assert_allclose(t.pre_grasp(z, standoff_delta_mm=-400.0), grasp)  # clamped


def test_skill_goals_and_gripper_override() -> None:
    t = taught()
    z = t.poses.zones["A"]
    here = np.array([1, 2, 3, 4, 5, 33.0])
    app = t.approach_goals(z, here)
    assert len(app) == 3 and all(g[GRIPPER_INDEX] == 0.0 for g in app)
    np.testing.assert_allclose(app[0][:5], here[:5])  # open in place first
    np.testing.assert_allclose(app[2][:5], POSES["zone_A_grasp"][:5])
    (lift,) = t.lift_goals(z, 80.0)
    assert lift[GRIPPER_INDEX] == pytest.approx(48.0)
    np.testing.assert_allclose(lift[:5], POSES["zone_A_pre"][:5])
    place = t.place_goals(80.0)
    assert [g[GRIPPER_INDEX] for g in place] == pytest.approx([48.0, 0.0, 0.0])
    assert t.safe_pose(48.0)[GRIPPER_INDEX] == 48.0


# ------------------------------------------------------------------ poses.yaml
def test_seeded_poses_yaml_is_empty_and_lists_what_is_missing() -> None:
    p = load_poses(ROOT / "data" / "calib" / "poses.yaml", mock_joints())
    missing = p.missing()
    assert "pose 'home'" in missing and "pose 'place'" in missing
    assert any("zone" in m for m in missing) and any("gripper" in m for m in missing)


def test_save_pose_roundtrip_and_frame_conversion(tmp_path: Path) -> None:
    path = tmp_path / "poses.yaml"
    for name, vals in POSES.items():
        save_pose(path, name, vals)
    assert "Save pose as" in path.read_text()  # header written by save_pose
    data = yaml.safe_load(path.read_text())
    data["gripper"] = {"open_deg": 0.0, "closed_deg": 60.0}
    data["zones"] = {"A": {"polygon_px": [list(p) for p in SQUARE], "radius_mm": 200}}
    path.write_text(yaml.safe_dump(data))
    joints = JointsCalib(
        tuple(
            JointCalib(n, 1500, -1 if n == "gripper" else 1, 5.0 if n == "base_yaw" else 0.0)
            for n in JOINT_NAMES
        ),
        verified=True,
    )
    p = load_poses(path, joints)
    assert p.missing() == []  # zone B poses exist but zone B is not defined: not required
    assert p.poses["zone_A_pre"][0] == pytest.approx(15.0)  # host = 5 + fw
    assert p.gripper_closed_deg == -60.0  # sign -1 on the gripper
    save_pose(path, "home", (1, 2, 3, 4, 5, 6))  # update keeps the zones and gripper
    again = load_poses(path, mock_joints())
    assert again.poses["home"] == (1, 2, 3, 4, 5, 6) and "A" in again.zones
    assert again.gripper_closed_deg == 60.0
    with pytest.raises(ValueError):
        save_pose(path, "bad name!", (0,) * 6)


def test_bad_poses_yaml() -> None:
    with pytest.raises(CalibrationError):
        poses_from_dict({"poses": {"home": [0, 0, 0]}}, mock_joints())
    with pytest.raises(CalibrationError):
        poses_from_dict({"zones": {"A": {"polygon_px": [[1]]}}}, mock_joints())


# ------------------------------------------------------------------ FSM (mock, taught)
def _fast() -> "object":
    cfg = mock_profile()
    cfg.arm.max_joint_speed_deg_s, cfg.arm.max_joint_accel_deg_s2 = 3000.0, 30000.0
    cfg.fsm_rate_hz = 50.0
    return cfg


G, S, E = Verdict.GRASPED, Verdict.SLIPPING, Verdict.EMPTY


def _run(script: tuple, scene: FakeScene | None = None):  # type: ignore[no-untyped-def]
    cfg = _fast()
    assert cfg.motion.mode == "taught"  # type: ignore[attr-defined]
    stack = build_mock_stack(cfg, script, critic_delay_s=0.01, scene=scene)  # type: ignore[arg-type]
    with stack:
        result = stack.supervisor.run_episode(EpisodeSpec("put the block on the place zone"), 30.0)
    return result, stack


def test_taught_episode_moves_through_the_taught_poses() -> None:
    result, stack = _run((G, G, E))
    assert result.outcome == "success"
    t = stack.supervisor._taught
    assert t is not None
    moves = [np.asarray(m) for m in stack.arm.moves]
    grasp = t.poses.pose("zone_A_grasp")
    assert any(np.allclose(m[:5], grasp[:5]) for m in moves)
    assert np.allclose(moves[-1][:5], t.poses.pose("place_pre")[:5])  # retreat after release
    assert moves[-1][GRIPPER_INDEX] == 0.0  # released


def test_taught_slipping_twice_reorients_joint_5() -> None:
    result, stack = _run((S, S, G, G, E))
    assert result.outcome == "success"
    assert [d for _, d in result.recovery_history] == ["+15% closure", "wrist re-orient 30 deg"]
    t = stack.supervisor._taught
    assert t is not None
    grasp = t.poses.pose("zone_A_grasp")
    assert any(
        m[WRIST_ROLL_INDEX] == pytest.approx(grasp[WRIST_ROLL_INDEX] + 30.0)
        and m[0] == pytest.approx(grasp[0])
        for m in stack.arm.moves
    )


def test_taught_empty_after_lift_uses_interpolated_pre_grasp() -> None:
    result, stack = _run((G, E, G, G, E))
    assert result.recovery_history == [(FailureClass.EMPTY, "re-approach, -5 mm standoff")]
    t = stack.supervisor._taught
    assert t is not None
    pre, grasp = t.poses.pose("zone_A_pre"), t.poses.pose("zone_A_grasp")
    f = 5.0 / t.default_standoff_mm
    target = pre + f * (grasp - pre)
    assert any(np.allclose(np.asarray(m)[:5], target[:5]) for m in stack.arm.moves)


def test_taught_object_outside_every_zone_aborts() -> None:
    from lbyr.mock_stack import build_mock_stack as build
    from lbyr.motion.taught import TaughtPoses as TP

    cfg = _fast()
    stack = build(cfg, (G, G, E), critic_delay_s=0.01)  # type: ignore[arg-type]
    t = stack.supervisor._taught
    assert t is not None
    far = Zone("A", ((0, 0), (5, 0), (5, 5)), 200.0)
    t.poses = TP(t.poses.poses, {"A": far}, 0.0, 90.0)
    with stack:
        result = stack.supervisor.run_episode(EpisodeSpec("put the block on the place zone"), 30.0)
    assert result.outcome == "abort" and result.failure is FailureClass.IK_FAILED
    assert "no taught zone" in result.report
    assert State.APPROACH not in [tr.dst for tr in result.transitions]


def test_taught_mode_without_a_place_detection_adds_the_taught_place() -> None:
    scene = FakeScene(objects=[FakeObject(2, "acrylic_block", (220.0, 60.0, 0.0), 30.0, 30.0)])
    result, _ = _run((G, G, E), scene=scene)
    assert result.outcome == "success"


def test_rules_and_language_planners_both_resolve_to_a_taught_zone() -> None:
    from typing import Any

    from lbyr.ai.planner import (
        PlanRequest,
        Qwen3TextPlanner,
        RuleBasedPlanner,
        plan_language,
        plan_with_timeout,
    )
    from lbyr.config import PlannerConfig

    t = taught()
    block = Detection(3, "wooden_block", 0.9, (140, 330, 180, 370))  # inside zone A
    dets = [block, t.place_detection()]
    req = PlanRequest("put the wooden block on the place zone", dets)
    rules = plan_with_timeout(RuleBasedPlanner(), req, PlannerConfig())
    assert rules.chain is not None and rules.chain.place.target_id == PLACE_ZONE_ID
    assert t.zone_for({d.id: d for d in dets}[rules.chain.approach.target_id]).name == "A"

    class Engine:
        def run(self, prompt: str, image: Any, n: int, max_time: float | None) -> tuple[str, int]:
            return f'{{"a":3,"s":40,"g":80,"p":{PLACE_ZONE_ID}}}', 19

    cfg = PlannerConfig()
    lang = plan_language(Qwen3TextPlanner(cfg, engine=Engine()), req, cfg)
    assert lang.chain is not None
    assert t.zone_for({d.id: d for d in dets}[lang.chain.approach.target_id]).name == "A"
