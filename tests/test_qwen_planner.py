"""QwenPlanner with a fake engine (no torch, no weights)."""

import json
import time
from typing import Any

import numpy as np
import pytest

from lbyr.ai.planner import (
    FallbackPlanner,
    PlanRequest,
    QwenPlanner,
    RuleBasedPlanner,
    build_planner,
    build_planner_prompt,
    expand_compact_plan,
    language_planner_of,
    plan_language,
    plan_with_timeout,
)
from lbyr.config import PlannerConfig
from lbyr.errors import ConfigError
from lbyr.perception.detector import Detection

pytest.importorskip("PIL")

DETS = [
    Detection(2, "acrylic_block", 0.9, (200.0, 100.0, 300.0, 200.0)),
    Detection(5, "place_zone", 0.8, (400.0, 300.0, 600.0, 460.0)),
]
GOOD = json.dumps(
    {
        "skill_chain": [
            {"skill": "approach", "target_id": 2, "standoff_mm": 40},
            {"skill": "grasp", "closure": 80},
            {"skill": "place", "target_id": 5},
        ],
        "notes": "",
    }
)


class FakeEngine:
    def __init__(self, text: str = GOOD, n_tokens: int = 50, delay_s: float = 0.0) -> None:
        self.text, self.n_tokens, self.delay_s = text, n_tokens, delay_s
        self.calls: list[dict[str, Any]] = []

    def run(
        self, prompt: str, image: Any | None, max_new_tokens: int, max_time_s: float | None
    ) -> tuple[str, int]:
        self.calls.append(
            {
                "prompt": prompt,
                "image": image,
                "max_new_tokens": max_new_tokens,
                "max_time_s": max_time_s,
            }
        )
        if self.delay_s:
            time.sleep(self.delay_s)
        return self.text, self.n_tokens


def _req(image: bool = True, w: int = 640, h: int = 480) -> PlanRequest:
    img = np.zeros((h, w, 3), dtype=np.uint8) if image else None
    return PlanRequest("put the acrylic block on the place zone", DETS, img)


def test_prompt_has_task_objects_bboxes_and_no_joint_angles_rule() -> None:
    p = build_planner_prompt(_req(image=False))
    assert "Instruction: put the acrylic block on the place zone" in p
    assert "id=2 label=acrylic_block score=0.90 bbox_xyxy_px=[200,100,300,200]" in p
    assert "id=5 label=place_zone" in p
    assert "Never output joint angles" in p and "JSON" in p


def test_prompt_bbox_scale() -> None:
    p = build_planner_prompt(_req(image=False), bbox_scale=0.5)
    assert "bbox_xyxy_px=[100,50,150,100]" in p


def test_construction_warms_up_with_a_short_generation() -> None:
    eng = FakeEngine()
    QwenPlanner(PlannerConfig(), engine=eng)
    assert len(eng.calls) == 1
    assert eng.calls[0]["max_new_tokens"] == 8 and eng.calls[0]["image"] is not None


def test_generate_is_greedy_bounded_and_records_stats() -> None:
    eng = FakeEngine(n_tokens=60)
    cfg = PlannerConfig(max_new_tokens=128)
    planner = QwenPlanner(cfg, engine=eng)
    assert planner.generate(_req()) == GOOD
    call = eng.calls[-1]
    assert call["max_new_tokens"] == 128 and call["max_time_s"] == cfg.timeout_s == 3.0
    assert planner.last_stats is not None and planner.last_stats.new_tokens == 60
    assert planner.last_stats.tokens_per_s > 0


def test_generate_untimed_has_no_max_time() -> None:
    eng = FakeEngine()
    QwenPlanner(PlannerConfig(), engine=eng).generate_untimed(_req())
    assert eng.calls[-1]["max_time_s"] is None


def test_text_only_request_sends_no_image() -> None:
    eng = FakeEngine()
    QwenPlanner(PlannerConfig(), engine=eng).generate(_req(image=False))
    assert eng.calls[-1]["image"] is None


def test_large_images_are_downscaled_and_bboxes_follow() -> None:
    eng = FakeEngine()
    planner = QwenPlanner(PlannerConfig(max_image_pixels=320 * 240), engine=eng)
    planner.generate(_req(w=640, h=480))
    call = eng.calls[-1]
    assert call["image"].size == (320, 240)
    assert "bbox_xyxy_px=[100,50,150,100]" in call["prompt"]
    planner2 = QwenPlanner(PlannerConfig(), engine=eng)  # default budget keeps 640x480
    planner2.generate(_req(w=640, h=480))
    assert eng.calls[-1]["image"].size == (640, 480)


def test_valid_output_goes_through_strict_parser() -> None:
    cfg = PlannerConfig()
    res = plan_with_timeout(
        QwenPlanner(cfg, engine=FakeEngine("```json\n" + GOOD + "\n```")), _req(), cfg
    )
    assert res.chain is not None and res.chain.place.target_id == 5


def test_trace_has_the_exact_prompt_sent_raw_output_and_chain() -> None:
    eng = FakeEngine()
    cfg = PlannerConfig(max_image_pixels=320 * 240)
    res = plan_with_timeout(QwenPlanner(cfg, engine=eng), _req(), cfg)
    t = res.trace
    assert t is not None and t.backend == "qwen" and t.mode == "loop"
    assert t.prompt == eng.calls[-1]["prompt"]  # includes the bbox rescale
    assert t.raw_output == GOOD and t.chain == json.loads(GOOD)


def test_joint_angles_and_hallucinated_ids_are_rejected() -> None:
    cfg = PlannerConfig()
    obj = json.loads(GOOD)
    obj["skill_chain"][0]["joints"] = [0, 0, 0, 0, 0, 0]
    res = plan_with_timeout(QwenPlanner(cfg, engine=FakeEngine(json.dumps(obj))), _req(), cfg)
    assert res.chain is None and res.error is not None
    obj = json.loads(GOOD)
    obj["skill_chain"][2]["target_id"] = 99
    res = plan_with_timeout(QwenPlanner(cfg, engine=FakeEngine(json.dumps(obj))), _req(), cfg)
    assert res.chain is None and "not a detected object" in (res.error or "")
    res = plan_with_timeout(
        QwenPlanner(cfg, engine=FakeEngine("I think you should...")), _req(), cfg
    )
    assert res.chain is None and "not JSON" in (res.error or "")


def test_hard_timeout() -> None:
    cfg = PlannerConfig(timeout_s=0.2)
    eng = FakeEngine()
    planner = QwenPlanner(cfg, engine=eng)
    eng.delay_s = 1.0
    t0 = time.perf_counter()
    res = plan_with_timeout(planner, _req(), cfg)
    assert res.error == "timeout" and time.perf_counter() - t0 < 0.6


def test_qwen_then_rules_fallback_end_to_end() -> None:
    cfg = PlannerConfig(timeout_s=0.3, fallback_after_timeouts=2)
    eng = FakeEngine()
    fb = FallbackPlanner(QwenPlanner(cfg, engine=eng), RuleBasedPlanner(), cfg)
    eng.delay_s = 1.0
    assert plan_with_timeout(fb, _req(), cfg).error == "timeout"
    res = plan_with_timeout(fb, _req(), cfg)  # 2nd in a row -> rules answers
    assert res.chain is not None and "rules" in (res.raw_text or "")
    assert fb.active_backend == "rules"


# ------------------------------------------------------------------ language mode
COMPACT = '{"a":2,"s":40,"g":80,"p":5}'


def test_language_mode_is_text_only_compact_bounded_and_uses_its_own_timeout() -> None:
    eng = FakeEngine(COMPACT, n_tokens=17)
    cfg = PlannerConfig()
    planner = QwenPlanner(cfg, engine=eng)
    res = plan_language(planner, _req(image=True), cfg)  # the image is dropped
    assert res.chain is not None, res.error
    assert (res.chain.approach.target_id, res.chain.place.target_id) == (2, 5)
    assert res.chain.grasp.closure == 80 and res.chain.approach.standoff_mm == 40
    call = eng.calls[-1]
    assert call["image"] is None
    assert call["max_new_tokens"] == cfg.language_max_new_tokens == 64
    assert call["max_time_s"] == cfg.qwen_timeout_s == 8.0
    assert cfg.timeout_s == 3.0  # the in-loop default is unchanged
    p = call["prompt"]
    assert '{"id":2,"label":"acrylic_block","xy":[250,150]}' in p
    assert '"a":' in p and "notes" not in p and "Never output joint angles" in p
    assert "Task: put the acrylic block on the place zone" in p
    t = res.trace
    assert t is not None and t.mode == "language" and t.backend == "qwen"
    assert t.prompt == p and t.raw_output == COMPACT
    assert t.chain is not None and t.chain["skill_chain"][2]["target_id"] == 5


@pytest.mark.parametrize(
    "text, msg",
    [
        ('{"a":2,"s":40,"g":80,"p":5,"notes":""}', "keys"),
        ('{"a":2,"s":40,"g":80,"p":5,"joints":[0,0,0,0,0,0]}', "keys"),
        ('{"a":2,"s":40,"g":80}', "keys"),
        ('{"a":9,"s":40,"g":80,"p":5}', "not a detected object"),
        ('{"a":2,"s":40,"g":180,"p":5}', "outside"),
        ('{"a":"2","s":40,"g":80,"p":5}', "integer"),
        ("[2,40,80,5]", "keys"),
        ("pick the block", "not JSON"),
    ],
)
def test_language_mode_output_is_strictly_validated(text: str, msg: str) -> None:
    cfg = PlannerConfig()
    res = plan_language(QwenPlanner(cfg, engine=FakeEngine(text)), _req(), cfg)
    assert res.chain is None and msg in (res.error or ""), res.error


def test_expand_compact_plan_goes_to_the_canonical_contract() -> None:
    assert json.loads(expand_compact_plan("```json\n" + COMPACT + "\n```")) == json.loads(GOOD)


def test_language_mode_timeout() -> None:
    cfg = PlannerConfig(qwen_timeout_s=0.2)
    eng = FakeEngine(COMPACT)
    planner = QwenPlanner(cfg, engine=eng)
    eng.delay_s = 1.0
    t0 = time.perf_counter()
    res = plan_language(planner, _req(), cfg)
    assert res.error == "timeout" and time.perf_counter() - t0 < 0.6
    assert res.trace is not None and res.trace.raw_output is None


def test_language_planner_of() -> None:
    cfg = PlannerConfig()
    qwen = QwenPlanner(cfg, engine=FakeEngine())
    assert language_planner_of(qwen) is qwen
    assert language_planner_of(FallbackPlanner(qwen, RuleBasedPlanner(), cfg)) is qwen
    assert language_planner_of(RuleBasedPlanner()) is None


def test_language_config_validation() -> None:
    with pytest.raises(ConfigError):
        PlannerConfig(qwen_timeout_s=0.0).validate()
    with pytest.raises(ConfigError):
        PlannerConfig(language_max_new_tokens=0).validate()
    with pytest.raises(ConfigError):  # the 8 s budget is NOT allowed for in-loop calls
        PlannerConfig(timeout_s=8.0).validate()


def test_refuses_without_local_model_path_or_nonzero_temperature() -> None:
    with pytest.raises(ConfigError, match="model_path"):
        QwenPlanner(PlannerConfig())
    with pytest.raises(ConfigError, match="model_path"):
        build_planner(PlannerConfig(backend="qwen"))
    with pytest.raises(ConfigError):
        QwenPlanner(PlannerConfig(temperature=0.7), engine=FakeEngine())
