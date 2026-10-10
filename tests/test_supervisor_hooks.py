"""Supervisor hooks used by the hardware loop: live view, induced slip, preset (language) plan."""

import json

from lbyr.ai.critic import Verdict
from lbyr.ai.planner import PlanRequest, RuleBasedPlanner, plan_with_timeout
from lbyr.config import GRIPPER_INDEX, AppConfig, PlannerConfig, mock_profile
from lbyr.fsm.recovery import FailureClass
from lbyr.fsm.states import State
from lbyr.fsm.supervisor import EpisodeSpec, LiveEvent
from lbyr.mock_stack import build_mock_stack
from lbyr.perception.detector import Detection

G, S, E = Verdict.GRASPED, Verdict.SLIPPING, Verdict.EMPTY


def fast_cfg() -> AppConfig:
    cfg = mock_profile()
    cfg.arm.max_joint_speed_deg_s = 3000.0  # FAKE, test speed only
    cfg.arm.max_joint_accel_deg_s2 = 30000.0
    cfg.fsm_rate_hz = 50.0
    return cfg


def test_live_view_gets_verdict_and_recovery_step() -> None:
    events: list[LiveEvent] = []
    stack = build_mock_stack(fast_cfg(), (S, G, G, E), critic_delay_s=0.01)
    stack.supervisor.d.observer = events.append
    with stack:
        result = stack.supervisor.run_episode(EpisodeSpec("put the acrylic block on the zone"))
    assert result.outcome == "success"
    to_recover = next(e for e in events if e.dst is State.RECOVER)
    assert to_recover.verdict is not None and to_recover.verdict.verdict is S
    assert to_recover.verdict.raw_text == "SLIPPING" and to_recover.verdict.latency_ms > 0
    out_of_recover = next(e for e in events if e.src is State.RECOVER)
    assert out_of_recover.recovery == "k=1 +15% closure"
    assert [v.verdict for v in result.verdict_log] == [S, G, G, E]
    assert result.verdict_log[0].state is State.VERIFY_GRASP
    assert len(events) == len(result.transitions)


def test_induced_slip_only_loosens_the_first_grasp(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cfg = fast_cfg()
    stack = build_mock_stack(cfg, (S, G, G, E), critic_delay_s=0.01, log_dir=tmp_path)
    with stack:
        result = stack.supervisor.run_episode(
            EpisodeSpec("put the acrylic block on the zone", induce_slip=True)
        )
    assert result.outcome == "success" and result.induced_slip
    grip = [round(m[GRIPPER_INDEX], 1) for m in stack.arm.moves]
    induced = cfg.arm.gripper.closure_to_deg(cfg.skills.induced_slip_closure_pct)
    regrasp = cfg.arm.gripper.closure_to_deg(95.0)  # planned 80 % + 15 after SLIPPING
    assert round(induced, 1) in grip and round(regrasp, 1) in grip
    assert grip.index(round(induced, 1)) < grip.index(round(regrasp, 1))
    records = [
        json.loads(line)
        for line in next(tmp_path.glob("*/episodes.jsonl")).read_text().splitlines()
    ]
    assert [r["closure_pct"] for r in records if r["event"] == "induced_slip"] == [50.0]


def test_preset_plan_is_used_and_checked_against_detections() -> None:
    stack = build_mock_stack(fast_cfg(), (G, G, E), critic_delay_s=0.01)
    dets = stack.overhead.scene.detections()
    req = PlanRequest("put the acrylic block on the place zone", dets)
    preset = plan_with_timeout(RuleBasedPlanner(), req, PlannerConfig())
    assert preset.chain is not None
    with stack:
        ok = stack.supervisor.run_episode(EpisodeSpec("anything", preset=preset))
    assert ok.outcome == "success"

    bad_req = PlanRequest("put the block on the zone", [Detection(77, "block", 1.0, (0, 0, 1, 1))])
    bad = plan_with_timeout(
        RuleBasedPlanner(),
        PlanRequest(bad_req.instruction, [*bad_req.detections, *dets]),
        PlannerConfig(),
    )
    assert bad.chain is not None and bad.chain.approach.target_id == 77
    stack2 = build_mock_stack(fast_cfg(), (G, G, E), critic_delay_s=0.01)
    with stack2:
        res = stack2.supervisor.run_episode(EpisodeSpec("anything", preset=bad))
    assert res.outcome == "abort" and res.failure is FailureClass.PLANNER_FAILED
