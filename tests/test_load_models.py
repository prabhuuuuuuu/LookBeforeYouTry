import logging

import pytest

from lbyr.ai import models
from lbyr.ai.models import load_models
from lbyr.ai.planner import RuleBasedPlanner
from lbyr.config import AppConfig


def test_critic_loads_before_planner_and_memory_is_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    order: list[str] = []
    mem = iter([0.0, 3700.0, 10900.0])  # at start, after critic, after planner
    monkeypatch.setattr(models, "cuda_allocated_mb", lambda: next(mem))

    def critic_factory(cfg):  # type: ignore[no-untyped-def]
        order.append("critic")
        return object()

    def planner_factory(cfg, skills):  # type: ignore[no-untyped-def]
        order.append("planner")
        return RuleBasedPlanner()

    with caplog.at_level(logging.INFO, logger="lbyr.ai.models"):
        loaded = load_models(
            AppConfig(), critic_factory=critic_factory, planner_factory=planner_factory
        )
    assert order == ["critic", "planner"]
    assert (loaded.critic_mb, loaded.planner_mb) == (3700.0, 10900.0)
    text = caplog.text
    assert "after Critic load" in text and "3700 MB" in text
    assert "after Planner" in text and "10900 MB" in text
    assert text.index("after Critic") < text.index("after Planner")


def test_critic_off_baseline_does_not_load_moondream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "cuda_allocated_mb", lambda: None)
    cfg = AppConfig()
    cfg.critic.enabled = False
    cfg.planner.backend = "rules"

    def boom(_cfg):  # type: ignore[no-untyped-def]
        raise AssertionError("Critic must not be loaded")

    loaded = load_models(cfg, critic_factory=boom)
    assert loaded.critic is None and isinstance(loaded.planner, RuleBasedPlanner)
