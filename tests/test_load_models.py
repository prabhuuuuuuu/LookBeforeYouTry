import logging

import pytest

from lbyr.ai import models
from lbyr.ai.models import check_memory_headroom, load_models
from lbyr.ai.planner import RuleBasedPlanner
from lbyr.config import AppConfig, PlannerConfig
from lbyr.errors import InsufficientMemoryError
from lbyr.logging_.episodes import RamReading, parse_meminfo, parse_tegrastats_ram

TOTAL = 15656.0


def _ram(used: float) -> RamReading:
    return RamReading(used, TOTAL, "tegrastats")


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
            AppConfig(),
            critic_factory=critic_factory,
            planner_factory=planner_factory,
            ram_reader=lambda: None,
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

    loaded = load_models(cfg, critic_factory=boom, ram_reader=lambda: None)
    assert loaded.critic is None and isinstance(loaded.planner, RuleBasedPlanner)


# ------------------------------------------------------------------ memory guard
def _load(backend: str, readings: list[RamReading | None]) -> models.LoadedModels:
    cfg = AppConfig()
    cfg.planner.backend = backend
    it = iter(readings)
    return load_models(
        cfg,
        critic_factory=lambda c: object(),  # type: ignore[arg-type,return-value]
        planner_factory=lambda p, s: RuleBasedPlanner(),
        ram_reader=lambda: next(it),
    )


def test_ram_is_logged_at_startup_and_after_each_load(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(models, "cuda_allocated_mb", lambda: None)
    with caplog.at_level(logging.INFO, logger="lbyr.ai.models"):
        loaded = _load("rules", [_ram(1500.0), _ram(5600.0), _ram(14846.0)])
    text = caplog.text
    assert "startup (before model load): 1500/15656 MB used" in text
    assert "after Critic load: 5600/15656 MB used" in text
    assert "after Planner (rules) load: 14846/15656 MB used, 810 MB free" in text
    assert loaded.ram_after_load == _ram(14846.0)  # rules: no guard even when tight


def test_qwen_refused_when_free_ram_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "cuda_allocated_mb", lambda: None)
    with pytest.raises(InsufficientMemoryError, match="810 MB system RAM free.*Refusing"):
        _load("qwen", [_ram(1500.0), _ram(5600.0), _ram(14846.0)])  # measured 2026-10-10


def test_qwen_allowed_with_enough_headroom(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "cuda_allocated_mb", lambda: None)
    loaded = _load("qwen", [_ram(1500.0), _ram(5600.0), _ram(TOTAL - 2000.0)])
    assert loaded.ram_after_load is not None and loaded.ram_after_load.free_mb == 2000.0


def test_guard_threshold_and_fail_closed() -> None:
    check_memory_headroom(PlannerConfig(backend="qwen"), _ram(TOTAL - 1536.0))  # == limit: ok
    with pytest.raises(InsufficientMemoryError):
        check_memory_headroom(PlannerConfig(backend="qwen", qwen_min_free_ram_mb=3000), _ram(13000))
    with pytest.raises(InsufficientMemoryError, match="cannot read"):
        check_memory_headroom(PlannerConfig(backend="qwen"), None)
    check_memory_headroom(PlannerConfig(backend="rules"), None)  # rules: never refused


def test_ram_parsers() -> None:
    line = "10-10-2026 21:50:56 RAM 1517/15656MB (lfb 7x4MB) SWAP 0/7828MB (cached 0MB)"
    assert parse_tegrastats_ram(line) == RamReading(1517.0, 15656.0, "tegrastats")
    assert parse_tegrastats_ram("garbage") is None
    mi = parse_meminfo("MemTotal:  16000000 kB\nMemFree: 1 kB\nMemAvailable: 4000000 kB\n")
    assert mi is not None and mi.source == "meminfo"
    assert round(mi.total_mb) == 15625 and round(mi.free_mb) == 3906
    assert parse_meminfo("nothing") is None
