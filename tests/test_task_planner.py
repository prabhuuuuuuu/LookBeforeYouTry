"""Language prompting: strict move/none contract, name validation, timeout, rules fallback."""

import json
import time
from pathlib import Path

import pytest

from lbyr.ai.task_planner import (
    RulesTaskPlanner,
    build_task_prompt,
    clean_instruction,
    parse_task_action,
    plan_task,
)
from lbyr.demo import load_demo_poses
from lbyr.errors import CalibrationError, PlanValidationError

NAMES = ["A", "B", "place"]


def test_parse_move_and_none() -> None:
    a = parse_task_action('{"action":"move","from":"A","to":"place","reason":"asked"}', NAMES)
    assert (a.action, a.src, a.dst, a.reason) == ("move", "A", "place", "asked")
    n = parse_task_action('```json\n{"action": "none", "reason": "no such place"}\n```', NAMES)
    assert n.to_dict() == {"action": "none", "reason": "no such place"}


@pytest.mark.parametrize(
    "text",
    [
        "move A to place",  # not JSON
        '{"action":"move","from":"A","to":"C","reason":"x"}',  # unknown location
        '{"action":"move","from":"a","to":"place","reason":"x"}',  # names are exact
        '{"action":"move","from":"A","to":"A","reason":"x"}',  # same location
        '{"action":"move","from":"A","to":"place"}',  # missing reason
        '{"action":"move","from":"A","to":"place","reason":"x","joints":[1]}',  # extra key
        '{"action":"pick","reason":"x"}',
        '{"action":"none","reason":""}',
    ],
)
def test_parse_rejects(text: str) -> None:
    with pytest.raises(PlanValidationError):
        parse_task_action(text, NAMES)


def test_prompt_lists_locations_block_and_quotes_instruction() -> None:
    p = build_task_prompt('put it "left"', {"A": "left side", "place": ""}, "unknown", True)
    assert "- A: left side" in p and "- place: (no description)" in p
    assert "currently at: unknown" in p and json.dumps('put it "left"') in p


def test_plan_task_timeout_and_invalid() -> None:
    def slow() -> str:
        time.sleep(0.5)
        return "{}"

    res = plan_task(slow, NAMES, 0.05, "qwen", "p")
    assert res.action is None and res.error is not None and "timeout" in res.error
    res = plan_task(lambda: '{"action":"move","from":"Z","to":"A","reason":"r"}', NAMES, 1, "q", "")
    assert res.action is None and res.error is not None and "not a location" in res.error


def test_rules_task_planner() -> None:
    r = RulesTaskPlanner()
    a = parse_task_action(r.generate("move the block from A to the place zone", NAMES, "?"), NAMES)
    assert (a.src, a.dst) == ("A", "place")
    a = parse_task_action(r.generate("put it to B", NAMES, "A"), NAMES)
    assert (a.src, a.dst) == ("A", "B")
    assert parse_task_action(r.generate("dance", NAMES, "unknown"), NAMES).action == "none"


def test_clean_instruction() -> None:
    assert clean_instruction("  move \n it ") == "move it"
    with pytest.raises(PlanValidationError):
        clean_instruction("   ")
    with pytest.raises(PlanValidationError):
        clean_instruction("x" * 301)


def test_poses_yaml_locations(tmp_path: Path) -> None:
    p = load_demo_poses(Path("data/calib/poses.yaml"))
    assert {"A", "place"} <= set(p.locations)
    assert all(x.pre in p.poses and x.grasp in p.poses for x in p.locations.values())
    bad = tmp_path / "poses.yaml"
    bad.write_text(
        "gripper: {open_deg: -40, closed_deg: 0}\n"
        "poses: {home: [0,0,0,0,0,0], zone_A_pre: [0,0,0,0,0,0], zone_A_grasp: [0,0,0,0,0,0],"
        " place_pre: [0,0,0,0,0,0], place: [0,0,0,0,0,0]}\n"
        "locations: {B: {description: back, pre: zone_B_pre, grasp: zone_B_grasp}}\n"
    )
    with pytest.raises(CalibrationError, match="not a taught pose"):
        load_demo_poses(bad)
