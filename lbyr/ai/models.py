"""Load the on-device models in a fixed order: Critic first, then Planner.

Both must stay co-resident on the Orin 16 GB (bible 9.2). The Critic goes first because it
is the smaller one and its load must never be the thing that fails after the planner already
took the memory; ``torch.cuda.memory_allocated`` and system RAM (tegrastats) are logged at
startup and after each load.

Memory guard: with ``planner.backend = "qwen"``, if free system RAM after loading both
models is below ``planner.qwen_min_free_ram_mb`` (default 1.5 GB), :func:`load_models` raises
:class:`InsufficientMemoryError` so the episode loop never starts.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from lbyr.ai.critic import CriticBackend, Moondream2Critic
from lbyr.ai.cuda_mem import cuda_allocated_mb
from lbyr.ai.planner import Planner, build_planner
from lbyr.config import AppConfig, CriticConfig, PlannerConfig, SkillConfig
from lbyr.errors import InsufficientMemoryError
from lbyr.logging_.episodes import RamReading, read_system_ram

log = logging.getLogger(__name__)


@dataclass
class LoadedModels:
    critic: CriticBackend | None  # None when the Critic is disabled (baseline condition)
    planner: Planner
    critic_mb: float | None  # torch.cuda.memory_allocated after loading the Critic
    planner_mb: float | None  # ... after loading the Planner (cumulative)
    ram_after_load: RamReading | None = None  # system RAM after both loads


def load_models(
    cfg: AppConfig,
    *,
    critic_factory: Callable[[CriticConfig], CriticBackend] = Moondream2Critic,
    planner_factory: Callable[[PlannerConfig, SkillConfig], Planner] = build_planner,
    ram_reader: Callable[[], RamReading | None] = read_system_ram,
) -> LoadedModels:
    """Load Critic (if enabled), then Planner, then apply the memory guard.

    Factories and the RAM reader are injectable for tests.
    """
    _log_ram("startup (before model load)", ram_reader())
    critic: CriticBackend | None = None
    critic_mb = cuda_allocated_mb()
    if cfg.critic.enabled:
        critic = critic_factory(cfg.critic)
        critic_mb = cuda_allocated_mb()
        log.info("after Critic load: torch.cuda.memory_allocated = %s", _fmt(critic_mb))
        _log_ram("after Critic load", ram_reader())
    else:
        log.info("Critic disabled: not loading Moondream2")
    planner = planner_factory(cfg.planner, cfg.skills)
    planner_mb = cuda_allocated_mb()
    log.info(
        "after Planner (%s) load: torch.cuda.memory_allocated = %s",
        cfg.planner.backend,
        _fmt(planner_mb),
    )
    ram = ram_reader()
    _log_ram(f"after Planner ({cfg.planner.backend}) load", ram)
    check_memory_headroom(cfg.planner, ram)
    return LoadedModels(critic, planner, critic_mb, planner_mb, ram)


def check_memory_headroom(cfg: PlannerConfig, ram: RamReading | None) -> None:
    """Raise :class:`InsufficientMemoryError` if ``backend = "qwen"`` and free system RAM is
    below ``cfg.qwen_min_free_ram_mb`` (or cannot be read: fail closed). No-op for rules."""
    if cfg.backend != "qwen":
        return
    fix = (
        'Use planner.backend = "rules" (the default), or free memory first '
        "(close other processes; Critic GGUF-Q4 per bible 9.2 fallback chain)."
    )
    if ram is None:
        raise InsufficientMemoryError(
            "Memory guard: cannot read system RAM (no tegrastats or /proc/meminfo) with "
            f"planner.backend = 'qwen'. Refusing to start the episode loop. {fix}"
        )
    if ram.free_mb < cfg.qwen_min_free_ram_mb:
        raise InsufficientMemoryError(
            f"Memory guard: only {ram.free_mb:.0f} MB system RAM free after loading the Critic "
            f"and Qwen ({ram.used_mb:.0f}/{ram.total_mb:.0f} MB used, {ram.source}); "
            f"planner.qwen_min_free_ram_mb = {cfg.qwen_min_free_ram_mb:.0f} MB. "
            f"Refusing to start the episode loop. {fix}"
        )
    log.info("Memory guard OK: %.0f MB free >= %.0f MB", ram.free_mb, cfg.qwen_min_free_ram_mb)


def _log_ram(when: str, ram: RamReading | None) -> None:
    if ram is None:
        log.info("system RAM %s: n/a (no tegrastats / meminfo)", when)
    else:
        log.info(
            "system RAM %s: %.0f/%.0f MB used, %.0f MB free (%s)",
            when,
            ram.used_mb,
            ram.total_mb,
            ram.free_mb,
            ram.source,
        )


def _fmt(mb: float | None) -> str:
    return "n/a" if mb is None else f"{mb:.0f} MB"
