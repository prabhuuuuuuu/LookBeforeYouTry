import json
import time

import pytest

from lbyr.ai.planner import FakePlanner, PlanRequest, parse_skill_chain, plan_with_timeout
from lbyr.config import PlannerConfig
from lbyr.errors import PlanValidationError
from lbyr.perception.detector import Detection

BIBLE_EXAMPLE = """{
  "skill_chain": [
    {"skill": "approach", "target_id": 2, "standoff_mm": 40},
    {"skill": "grasp", "closure": 80},
    {"skill": "place", "target_id": 5}
  ],
  "notes": "..."
}"""

DETS = [
    Detection(2, "acrylic_block", 0.9, (0, 0, 10, 10)),
    Detection(5, "place_zone", 0.9, (20, 20, 30, 30)),
]


def test_parses_bible_example() -> None:
    chain = parse_skill_chain(BIBLE_EXAMPLE, {2, 5})
    assert chain.approach.target_id == 2 and chain.approach.standoff_mm == 40
    assert chain.grasp.closure == 80
    assert chain.place.target_id == 5


def test_tolerates_code_fence_only() -> None:
    assert parse_skill_chain("```json\n" + BIBLE_EXAMPLE + "\n```").grasp.closure == 80
    with pytest.raises(PlanValidationError):
        parse_skill_chain("Sure! Here is the plan: " + BIBLE_EXAMPLE)


def _mutate(fn) -> str:  # type: ignore[no-untyped-def]
    obj = json.loads(BIBLE_EXAMPLE)
    fn(obj)
    return json.dumps(obj)


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        _mutate(lambda o: o["skill_chain"].reverse()),
        _mutate(lambda o: o["skill_chain"][1].update(closure=150)),
        _mutate(lambda o: o["skill_chain"][1].update(closure="80")),
        _mutate(lambda o: o["skill_chain"][0].update(joints=[0, 0, 0, 0, 0, 0])),  # no joints!
        _mutate(lambda o: o["skill_chain"][0].update(target_id=99)),  # not detected
        _mutate(lambda o: o["skill_chain"][2].update(target_id=True)),
        _mutate(lambda o: o.update(extra=1)),
        _mutate(lambda o: o["skill_chain"].pop()),
    ],
)
def test_rejects_invalid(text: str) -> None:
    with pytest.raises(PlanValidationError):
        parse_skill_chain(text, {2, 5})


def test_plan_with_timeout_ok_and_timeout() -> None:
    req = PlanRequest("put the block on the zone", DETS)
    ok = plan_with_timeout(FakePlanner(), req, PlannerConfig())
    assert ok.chain is not None and ok.error is None
    t0 = time.perf_counter()
    slow = plan_with_timeout(FakePlanner(delay_s=1.0), req, PlannerConfig(timeout_s=0.1))
    assert slow.chain is None and slow.error == "timeout"
    assert time.perf_counter() - t0 < 0.5
