import sys
import time

import pytest

from lbyr.ai.planner import (
    FakePlanner,
    FallbackPlanner,
    PlanRequest,
    RuleBasedPlanner,
    build_planner,
    parse_skill_chain,
    plan_with_timeout,
)
from lbyr.config import PlannerConfig
from lbyr.errors import ConfigError
from lbyr.perception.detector import Detection

DETS = [
    Detection(1, "wooden_block", 0.95, (0, 0, 10, 10)),
    Detection(2, "acrylic_block", 0.80, (20, 0, 30, 10)),
    Detection(3, "kitchen_sponge", 0.70, (40, 0, 50, 10)),
    Detection(5, "place_zone", 0.90, (60, 60, 90, 90)),
]


def _plan(instruction: str, dets: list[Detection] = DETS):  # type: ignore[no-untyped-def]
    return plan_with_timeout(
        RuleBasedPlanner(), PlanRequest(instruction, dets), PlannerConfig(backend="rules")
    )


# ------------------------------------------------------------------ RuleBasedPlanner
def test_rules_picks_full_class_name_over_higher_score_partial() -> None:
    res = _plan("put the acrylic block on the place zone")
    assert res.chain is not None, res.error
    assert res.chain.approach.target_id == 2  # not the higher-scoring wooden_block
    assert res.chain.place.target_id == 5
    assert res.chain.grasp.closure == 80 and res.chain.approach.standoff_mm == 40


def test_rules_partial_word_match() -> None:
    res = _plan("pick up the sponge and put it in the zone")
    assert res.chain is not None and res.chain.approach.target_id == 3


def test_rules_output_goes_through_strict_parser_without_joints() -> None:
    text = RuleBasedPlanner(closure=70, standoff_mm=30).generate(
        PlanRequest("move the wooden block to the zone", DETS)
    )
    assert "joint" not in text
    chain = parse_skill_chain(text, {d.id for d in DETS})
    assert (chain.approach.target_id, chain.grasp.closure, chain.approach.standoff_mm) == (
        1,
        70,
        30,
    )


def test_rules_picks_named_zone() -> None:
    dets = [
        *DETS,
        Detection(6, "zone_b", 0.9, (0, 0, 1, 1)),
        Detection(4, "zone_a", 0.9, (0, 0, 1, 1)),
    ]
    assert _plan("put the wooden block on zone b", dets).chain.place.target_id == 6  # type: ignore[union-attr]
    # nothing names a zone -> lowest id
    assert _plan("put the wooden block somewhere", dets).chain.place.target_id == 4  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("instruction", "dets"),
    [
        ("stack the lego", DETS),  # no class matches
        ("put the acrylic block on the zone", [d for d in DETS if "zone" not in d.label]),
        ("put the acrylic block on the zone", [d for d in DETS if d.id == 5]),
    ],
)
def test_rules_refuses_instead_of_guessing(instruction: str, dets: list[Detection]) -> None:
    res = _plan(instruction, dets)
    assert res.chain is None and res.error is not None


def test_zone_is_never_the_pick_target() -> None:
    res = _plan("grab the place zone")  # only matches the zone
    assert res.chain is None


# ------------------------------------------------------------------ FallbackPlanner
class _Primary:
    """Fake VLM: sleeps ``delay_s`` then returns a valid plan; counts calls."""

    def __init__(self) -> None:
        self.delay_s = 0.0
        self.calls = 0
        self.fail = False
        self._inner = FakePlanner(delay_s=0.0)

    def generate(self, request: PlanRequest) -> str:
        self.calls += 1
        time.sleep(self.delay_s)
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        return self._inner.generate(request)


REQ = PlanRequest("put the acrylic block on the place zone", DETS)


def _fb(n: int = 2, timeout_s: float = 0.3) -> tuple[FallbackPlanner, _Primary, PlannerConfig]:
    cfg = PlannerConfig(timeout_s=timeout_s, fallback_after_timeouts=n)
    primary = _Primary()
    return FallbackPlanner(primary, RuleBasedPlanner(), cfg), primary, cfg


def test_primary_used_while_healthy() -> None:
    fb, primary, cfg = _fb()
    res = plan_with_timeout(fb, REQ, cfg)
    assert res.chain is not None and "fake planner" in (res.raw_text or "")
    assert fb.active_backend == "qwen" and fb.consecutive_timeouts == 0


def test_switches_to_rules_after_n_consecutive_timeouts() -> None:
    fb, primary, cfg = _fb(n=2)
    primary.delay_s = 0.6
    first = plan_with_timeout(fb, REQ, cfg)
    assert first.chain is None and first.error == "timeout"
    assert fb.active_backend == "qwen" and fb.consecutive_timeouts == 1
    # the abandoned call is still running: single-flight reports "busy", counted as a timeout
    second = plan_with_timeout(fb, REQ, cfg)
    assert second.chain is not None and "rules" in (second.raw_text or "")  # answered by rules
    assert fb.active_backend == "rules"
    calls = primary.calls
    third = plan_with_timeout(fb, REQ, cfg)
    assert third.chain is not None and primary.calls == calls  # VLM no longer called


def test_success_resets_the_timeout_streak() -> None:
    fb, primary, cfg = _fb(n=2)
    primary.delay_s = 0.6
    assert plan_with_timeout(fb, REQ, cfg).error == "timeout"
    time.sleep(0.6)  # let the abandoned call finish
    primary.delay_s = 0.0
    assert plan_with_timeout(fb, REQ, cfg).chain is not None
    assert fb.consecutive_timeouts == 0
    primary.delay_s = 0.6
    assert plan_with_timeout(fb, REQ, cfg).error == "timeout"
    assert fb.active_backend == "qwen"  # streak restarted, so not switched


def test_backend_errors_are_not_timeouts() -> None:
    fb, primary, cfg = _fb(n=1)
    primary.fail = True
    res = plan_with_timeout(fb, REQ, cfg)
    assert res.chain is None and "CUDA out of memory" in (res.error or "")
    assert fb.active_backend == "qwen" and fb.consecutive_timeouts == 0


def test_reset_returns_to_primary() -> None:
    fb, primary, cfg = _fb(n=1)
    primary.delay_s = 0.5
    assert plan_with_timeout(fb, REQ, cfg).chain is not None  # 1st timeout == N -> rules
    assert fb.active_backend == "rules"
    fb.reset()
    assert fb.active_backend == "qwen"


# ------------------------------------------------------------------ reasoning trace
def test_rules_trace_has_inputs_raw_and_chain_but_no_prompt() -> None:
    res = _plan("put the acrylic block on the place zone")
    t = res.trace
    assert t is not None and t.backend == "rules" and t.mode == "loop" and t.prompt is None
    assert t.instruction == "put the acrylic block on the place zone"
    assert [d["id"] for d in t.detections] == [1, 2, 3, 5]
    assert t.raw_output == res.raw_text and t.error is None
    assert t.chain is not None and t.chain["skill_chain"][0]["target_id"] == 2
    assert t.to_dict()["chain"] == t.chain


def test_trace_on_timeout_and_after_fallback_switch() -> None:
    fb, primary, cfg = _fb(n=2)
    primary.delay_s = 0.5
    first = plan_with_timeout(fb, REQ, cfg)
    assert first.trace is not None and first.trace.error == "timeout"
    assert first.trace.backend == "qwen" and first.trace.raw_output is None
    second = plan_with_timeout(fb, REQ, cfg)  # answered by rules after the switch
    assert second.trace is not None and second.trace.backend == "rules"
    assert second.trace.chain is not None


def test_default_backend_is_rules() -> None:
    assert PlannerConfig().backend == "rules"
    assert isinstance(build_planner(PlannerConfig()), RuleBasedPlanner)


# ------------------------------------------------------------------ config / factory
def test_build_planner_rules_needs_no_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    planner = build_planner(PlannerConfig(backend="rules"))
    assert isinstance(planner, RuleBasedPlanner)


def test_config_validation() -> None:
    with pytest.raises(ConfigError):
        PlannerConfig(backend="gpt").validate()
    with pytest.raises(ConfigError):
        PlannerConfig(fallback_after_timeouts=0).validate()
    with pytest.raises(ConfigError):
        PlannerConfig(timeout_s=5.0).validate()
