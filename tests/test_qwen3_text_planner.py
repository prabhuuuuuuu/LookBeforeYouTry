"""Qwen3 text planner (language mode) with a fake engine: no torch, no weights."""

import time
from pathlib import Path
from typing import Any

import pytest

from lbyr.ai.planner import (
    PlanRequest,
    Qwen3TextPlanner,
    build_language_planner,
    plan_language,
    strip_empty_think,
)
from lbyr.config import PlannerConfig
from lbyr.errors import ConfigError
from lbyr.perception.detector import Detection

DETS = [
    Detection(2, "red_block", 0.9, (200.0, 100.0, 300.0, 200.0)),
    Detection(5, "place_zone", 0.8, (400.0, 300.0, 600.0, 460.0)),
]
REQ = PlanRequest("put the red block on the place zone", DETS)
COMPACT = '{"a":2,"s":40,"g":80,"p":5}'


class FakeEngine:
    def __init__(self, text: str = COMPACT, delay_s: float = 0.0) -> None:
        self.text, self.delay_s = text, delay_s
        self.calls: list[dict[str, Any]] = []

    def run(
        self, prompt: str, image: Any | None, max_new_tokens: int, max_time_s: float | None
    ) -> tuple[str, int]:
        self.calls.append(
            {"prompt": prompt, "image": image, "max_new": max_new_tokens, "max_time": max_time_s}
        )
        if self.delay_s:
            time.sleep(self.delay_s)
        return self.text, 19


def test_defaults() -> None:
    cfg = PlannerConfig()
    assert cfg.backend == "rules" and cfg.language_backend == "qwen3_text"
    assert cfg.language_model_path == Path("~/models/qwen3-1.7b")
    assert cfg.language_dtype == "bf16" and cfg.language_mode is False
    assert cfg.language_max_new_tokens == 64 and cfg.qwen_timeout_s == 8.0


def test_language_mode_text_only_greedy_bounded_and_traced() -> None:
    eng = FakeEngine()
    cfg = PlannerConfig()
    planner = Qwen3TextPlanner(cfg, engine=eng)
    assert eng.calls[0]["max_new"] == 8  # warm-up
    res = plan_language(planner, REQ, cfg)
    assert res.chain is not None, res.error
    assert (res.chain.approach.target_id, res.chain.place.target_id) == (2, 5)
    call = eng.calls[-1]
    assert call["image"] is None and call["max_new"] == 64 and call["max_time"] == 8.0
    t = res.trace
    assert t is not None and t.backend == "qwen3_text" and t.mode == "language"
    assert t.prompt == call["prompt"] and t.raw_output == COMPACT
    assert planner.last_stats is not None and planner.last_stats.new_tokens == 19


def test_empty_think_block_is_stripped_but_reasoning_is_rejected() -> None:
    assert strip_empty_think("<think>\n\n</think>\n\n" + COMPACT) == COMPACT
    cfg = PlannerConfig()
    ok = plan_language(
        Qwen3TextPlanner(cfg, engine=FakeEngine("<think></think>" + COMPACT)), REQ, cfg
    )
    assert ok.chain is not None
    bad = plan_language(
        Qwen3TextPlanner(cfg, engine=FakeEngine("<think>the red one</think>" + COMPACT)), REQ, cfg
    )
    assert bad.chain is None and "not JSON" in (bad.error or "")


def test_strict_parser_rejects_hallucinated_ids_and_joints() -> None:
    cfg = PlannerConfig()
    for text in ('{"a":7,"s":40,"g":80,"p":5}', '{"a":2,"s":40,"g":80,"p":5,"j":[0,0]}'):
        res = plan_language(Qwen3TextPlanner(cfg, engine=FakeEngine(text)), REQ, cfg)
        assert res.chain is None and res.error


def test_timeout() -> None:
    cfg = PlannerConfig(qwen_timeout_s=0.2)
    eng = FakeEngine()
    planner = Qwen3TextPlanner(cfg, engine=eng)
    eng.delay_s = 1.0
    t0 = time.perf_counter()
    assert plan_language(planner, REQ, cfg).error == "timeout"
    assert time.perf_counter() - t0 < 0.6


def test_missing_weights_and_bad_config(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="does not exist"):
        build_language_planner(PlannerConfig(language_model_path=tmp_path / "nope"))
    with pytest.raises(ConfigError, match="model_path"):  # qwen_vl uses model_path (unset)
        build_language_planner(PlannerConfig(language_backend="qwen_vl"))
    with pytest.raises(ConfigError):
        PlannerConfig(language_backend="smolvlm").validate()
    with pytest.raises(ConfigError):
        PlannerConfig(language_dtype="int4").validate()
