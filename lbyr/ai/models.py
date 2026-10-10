"""Load the on-device models in a fixed order: Critic first, then Planner.

Both must stay co-resident on the Orin 16 GB (bible 9.2). The Critic goes first because it
is the smaller one and its load must never be the thing that fails after the planner already
took the memory; ``torch.cuda.memory_allocated`` is logged after each load.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from lbyr.ai.critic import CriticBackend, Moondream2Critic
from lbyr.ai.cuda_mem import cuda_allocated_mb
from lbyr.ai.planner import Planner, build_planner
from lbyr.config import AppConfig, CriticConfig, PlannerConfig, SkillConfig

log = logging.getLogger(__name__)


@dataclass
class LoadedModels:
    critic: CriticBackend | None  # None when the Critic is disabled (baseline condition)
    planner: Planner
    critic_mb: float | None  # torch.cuda.memory_allocated after loading the Critic
    planner_mb: float | None  # ... after loading the Planner (cumulative)


def load_models(
    cfg: AppConfig,
    *,
    critic_factory: Callable[[CriticConfig], CriticBackend] = Moondream2Critic,
    planner_factory: Callable[[PlannerConfig, SkillConfig], Planner] = build_planner,
) -> LoadedModels:
    """Load Critic (if enabled), then Planner. Factories are injectable for tests."""
    critic: CriticBackend | None = None
    critic_mb = cuda_allocated_mb()
    if cfg.critic.enabled:
        critic = critic_factory(cfg.critic)
        critic_mb = cuda_allocated_mb()
        log.info("after Critic load: torch.cuda.memory_allocated = %s", _fmt(critic_mb))
    else:
        log.info("Critic disabled: not loading Moondream2")
    planner = planner_factory(cfg.planner, cfg.skills)
    planner_mb = cuda_allocated_mb()
    log.info(
        "after Planner (%s) load: torch.cuda.memory_allocated = %s",
        cfg.planner.backend,
        _fmt(planner_mb),
    )
    return LoadedModels(critic, planner, critic_mb, planner_mb)


def _fmt(mb: float | None) -> str:
    return "n/a" if mb is None else f"{mb:.0f} MB"
