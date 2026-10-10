"""Load the on-device models in a fixed order: Critic first, then Planner.

Both must stay co-resident on the Orin 16 GB (bible 9.2). The Critic goes first because it
is the smaller one and its load must never be the thing that fails after the planner already
took the memory; ``torch.cuda.memory_allocated`` and system RAM (tegrastats) are logged at
startup and after each load.

Order: Critic -> in-loop planner (rules by default: no model) -> language-mode planner (only
with ``planner.language_mode``; Qwen3 text by default). Free RAM is logged after each load.

Memory guard: with ``planner.backend = "qwen"`` or ``planner.language_mode``, if free system
RAM after loading all models is below ``planner.qwen_min_free_ram_mb`` (default 1.5 GB),
:func:`load_models` raises :class:`InsufficientMemoryError` so the episode loop never starts.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from lbyr.ai.critic import CriticBackend, Moondream2Critic
from lbyr.ai.cuda_mem import cuda_allocated_mb
from lbyr.ai.planner import LanguagePlanner, Planner, build_language_planner, build_planner
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
    ram_after_load: RamReading | None = None  # system RAM after all loads
    language_planner: LanguagePlanner | None = None  # only with planner.language_mode


def load_models(
    cfg: AppConfig,
    *,
    critic_factory: Callable[[CriticConfig], CriticBackend] = Moondream2Critic,
    planner_factory: Callable[[PlannerConfig, SkillConfig], Planner] = build_planner,
    language_factory: Callable[[PlannerConfig], LanguagePlanner] = build_language_planner,
    ram_reader: Callable[[], RamReading | None] = read_system_ram,
) -> LoadedModels:
    """Load Critic (if enabled), Planner, language planner (if enabled), then apply the
    memory guard.

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
    language: LanguagePlanner | None = None
    if cfg.planner.language_mode:
        language = language_factory(cfg.planner)
        log.info(
            "after language planner (%s) load: torch.cuda.memory_allocated = %s",
            cfg.planner.language_backend,
            _fmt(cuda_allocated_mb()),
        )
        ram = ram_reader()
        _log_ram(f"after language planner ({cfg.planner.language_backend}) load", ram)
    check_memory_headroom(cfg.planner, ram)
    return LoadedModels(critic, planner, critic_mb, planner_mb, ram, language)


def check_memory_headroom(cfg: PlannerConfig, ram: RamReading | None) -> None:
    """Raise :class:`InsufficientMemoryError` if a Qwen model is loaded (``backend = "qwen"``
    or ``language_mode``) and free system RAM is below ``cfg.qwen_min_free_ram_mb`` (or cannot
    be read: fail closed). No-op for rules-only."""
    if cfg.backend != "qwen" and not cfg.language_mode:
        return
    what = " + ".join(
        ([f"Qwen2.5-VL (backend={cfg.backend})"] if cfg.backend == "qwen" else [])
        + ([f"language planner ({cfg.language_backend})"] if cfg.language_mode else [])
    )
    fix = (
        'Use planner.backend = "rules" (the default), language_backend = "qwen3_text" with a '
        "smaller model (qwen3-0.6b), or free memory first (close other processes)."
    )
    if ram is None:
        raise InsufficientMemoryError(
            "Memory guard: cannot read system RAM (no tegrastats or /proc/meminfo) with "
            f"{what} loaded. Refusing to start the episode loop. {fix}"
        )
    if ram.free_mb < cfg.qwen_min_free_ram_mb:
        raise InsufficientMemoryError(
            f"Memory guard: only {ram.free_mb:.0f} MB system RAM free after loading the Critic "
            f"and {what} ({ram.used_mb:.0f}/{ram.total_mb:.0f} MB used, {ram.source}); "
            f"planner.qwen_min_free_ram_mb = {cfg.qwen_min_free_ram_mb:.0f} MB. "
            f"Refusing to start the episode loop. {fix}"
        )
    log.info(
        "Memory guard OK with %s: %.0f MB free >= %.0f MB",
        what,
        ram.free_mb,
        cfg.qwen_min_free_ram_mb,
    )


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
