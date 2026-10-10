import json
from pathlib import Path

import pytest

from lbyr.ai.critic import Verdict
from lbyr.config import GRIPPER_INDEX, AppConfig, mock_profile
from lbyr.fsm.recovery import RECOVERY_TABLE, FailureClass, RecoveryBudget, classify
from lbyr.fsm.states import PIPELINE, State
from lbyr.fsm.supervisor import EpisodeResult, EpisodeSpec
from lbyr.logging_.episodes import REQUIRED_FIELDS
from lbyr.mock_stack import DEMO_SCRIPT, build_mock_stack

G, S, E, OC = Verdict.GRASPED, Verdict.SLIPPING, Verdict.EMPTY, Verdict.OCCLUDED
HAPPY = (G, G, E)
SPEC = EpisodeSpec("put the acrylic block on the place zone")


def fast_cfg() -> AppConfig:
    cfg = mock_profile()
    cfg.arm.max_joint_speed_deg_s = 3000.0  # FAKE, test speed only
    cfg.arm.max_joint_accel_deg_s2 = 30000.0
    cfg.arm.max_delta_per_tick_deg = 40.0
    cfg.fsm_rate_hz = 50.0
    return cfg


def run(
    script: tuple = HAPPY,
    spec: EpisodeSpec = SPEC,
    log_dir: Path | None = None,
    jam_joint: int | None = None,
    critic_enabled: bool = True,
    critic_timeout_s: float | None = None,
    delay_s: float = 0.01,
) -> tuple[EpisodeResult, object]:
    cfg = fast_cfg()
    if critic_timeout_s is not None:
        cfg.critic.timeout_s = critic_timeout_s
    stack = build_mock_stack(
        cfg, script, critic_delay_s=delay_s, critic_enabled=critic_enabled, log_dir=log_dir
    )
    with stack:
        if jam_joint is not None:
            stack.arm.inject_jam(jam_joint)
        result = stack.supervisor.run_episode(spec, timeout_s=30.0)
    return result, stack


def path_of(result: EpisodeResult) -> list[State]:
    return [State.IDLE] + [t.dst for t in result.transitions]


def test_happy_path_reaches_done(tmp_path: Path) -> None:
    result, _ = run(HAPPY, log_dir=tmp_path)
    assert result.outcome == "success"
    assert result.final_state is State.DONE
    assert result.recoveries == 0
    assert path_of(result) == list(PIPELINE)
    lines = next(tmp_path.glob("*/episodes.jsonl")).read_text().splitlines()
    records = [json.loads(line) for line in lines]
    assert all(all(k in r for k in REQUIRED_FIELDS) for r in records)
    assert records[-1]["outcome"] == "success"
    assert any(r.get("verdict") == "GRASPED" for r in records)
    traces = [r for r in records if r["event"] == "reasoning_trace"]
    assert len(traces) == 1
    trace = traces[0]["trace"]
    assert trace["backend"] == "fake" and trace["mode"] == "loop"
    assert trace["raw_output"] and trace["chain"]["skill_chain"][0]["skill"] == "approach"
    assert {d["id"] for d in trace["detections"]} >= {trace["chain"]["skill_chain"][0]["target_id"]}


def test_slipping_then_grasped_triggers_exactly_one_recovery() -> None:
    result, stack = run(DEMO_SCRIPT)
    assert result.outcome == "success"
    assert result.recoveries == 1
    assert result.recovery_history == [(FailureClass.SLIPPING, "+15% closure")]
    assert path_of(result).count(State.RECOVER) == 1
    # recovery re-entered GRASP and the gripper really closed to 80 + 15 = 95 %
    i = path_of(result).index(State.RECOVER)
    assert path_of(result)[i + 1] is State.GRASP
    gripper = stack.cfg.arm.gripper  # type: ignore[attr-defined]
    peak = stack.arm.peak_deg[GRIPPER_INDEX]  # type: ignore[attr-defined]
    assert peak == pytest.approx(gripper.closure_to_deg(95.0), abs=0.5)


def test_happy_path_closes_to_planned_closure() -> None:
    _, stack = run(HAPPY)
    gripper = stack.cfg.arm.gripper  # type: ignore[attr-defined]
    peak = stack.arm.peak_deg[GRIPPER_INDEX]  # type: ignore[attr-defined]
    assert peak == pytest.approx(gripper.closure_to_deg(80.0), abs=0.5)


def test_three_slipping_ends_in_abort() -> None:
    result, stack = run((S, S, S))
    assert result.outcome == "abort"
    assert result.final_state is State.ABORT
    assert result.failure is FailureClass.SLIPPING
    assert result.recoveries == 2
    assert [d for _, d in result.recovery_history] == ["+15% closure", "wrist re-orient 30 deg"]
    assert result.verdicts == [S, S, S]
    assert "ABORT" in result.report and "SLIPPING" in result.report
    assert stack.arm.torque_on  # type: ignore[attr-defined]  # non-jam abort keeps holding


def test_jammed_goes_straight_to_abort() -> None:
    result, stack = run(HAPPY, jam_joint=1)
    assert result.outcome == "abort"
    assert result.failure is FailureClass.JAMMED
    assert result.recoveries == 0
    assert State.RECOVER not in path_of(result)
    assert not stack.arm.torque_on  # type: ignore[attr-defined]
    assert "torque OFF" in result.report


def test_critic_off_baseline_passes_through() -> None:
    result, _ = run(HAPPY, critic_enabled=False)
    assert result.outcome == "success" and result.verdicts == []


def test_critic_timeout_requeries_once_without_using_budget() -> None:
    # first call sleeps past the timeout; the worker answers TIMEOUT, FSM re-queries
    result, _ = run((G, G, E), critic_timeout_s=0.05, delay_s=0.2)
    assert result.outcome == "abort"  # every call times out -> re-query once -> ABORT
    assert result.failure is FailureClass.CRITIC_UNAVAILABLE
    assert result.recoveries == 0


def test_empty_after_close_is_miss_and_replans() -> None:
    result, _ = run((E, G, G, E))
    assert result.outcome == "success"
    assert result.recovery_history == [(FailureClass.MISS, "re-plan from PERCEIVE")]
    p = path_of(result)
    assert p[p.index(State.RECOVER) + 1] is State.PERCEIVE


def test_occluded_requeries_same_state() -> None:
    result, _ = run((OC, G, G, E))
    assert result.outcome == "success"
    p = path_of(result)
    i = p.index(State.RECOVER)
    assert p[i - 1] is State.VERIFY_GRASP and p[i + 1] is State.VERIFY_GRASP


def test_wrong_object_replans_then_aborts() -> None:
    result, _ = run(HAPPY, spec=EpisodeSpec("pick the sponge", expected_label="sponge"))
    assert result.outcome == "abort"
    assert result.failure is FailureClass.WRONG_OBJECT
    assert path_of(result).count(State.PLAN) == 2


# ---------------------------------------------------------------- pure recovery logic
def test_recovery_table_matches_bible() -> None:
    assert RECOVERY_TABLE[FailureClass.SLIPPING].max_attempts == 2
    assert RECOVERY_TABLE[FailureClass.EMPTY].max_attempts == 2
    assert RECOVERY_TABLE[FailureClass.MISS].max_attempts == 2
    assert RECOVERY_TABLE[FailureClass.JAMMED].max_attempts == 0
    assert RECOVERY_TABLE[FailureClass.OCCLUDED].max_attempts == 1
    assert RECOVERY_TABLE[FailureClass.WRONG_OBJECT].max_attempts == 1


@pytest.mark.parametrize(
    ("state", "verdict", "expected"),
    [
        (State.VERIFY_GRASP, G, None),
        (State.VERIFY_GRASP, E, FailureClass.MISS),
        (State.VERIFY_HOLD, E, FailureClass.EMPTY),
        (State.VERIFY_HOLD, S, FailureClass.SLIPPING),
        (State.VERIFY_PLACE, E, None),
        (State.VERIFY_PLACE, G, FailureClass.PLACE_FAILED),
        (State.VERIFY_HOLD, Verdict.TIMEOUT, FailureClass.CRITIC_UNAVAILABLE),
        (State.VERIFY_GRASP, OC, FailureClass.OCCLUDED),
    ],
)
def test_classify(state: State, verdict: Verdict, expected: FailureClass | None) -> None:
    assert classify(state, verdict) is expected


def test_global_budget_caps_total() -> None:
    b = RecoveryBudget(max_total=2)
    assert b.next_action(FailureClass.OCCLUDED) is not None
    assert b.next_action(FailureClass.SLIPPING) is not None
    assert b.next_action(FailureClass.SLIPPING) is None  # k would be 3
    assert b.next_action(FailureClass.CRITIC_UNAVAILABLE) is not None  # re-query not counted


def test_gripper_index_is_last() -> None:
    assert GRIPPER_INDEX == 5
