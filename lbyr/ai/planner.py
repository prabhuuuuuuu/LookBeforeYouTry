"""VLM planner: language + overhead image -> strict JSON skill chain (bible 11.1).

The planner NEVER outputs joint angles; IK produces joints. Contract::

    {"skill_chain": [
        {"skill": "approach", "target_id": 2, "standoff_mm": 40},
        {"skill": "grasp", "closure": 80},
        {"skill": "place", "target_id": 5}],
     "notes": "..."}

Temperature 0, 3 s hard timeout (enforced by :func:`plan_with_timeout`).
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from lbyr.ai.timeout import TimeoutResult, run_with_timeout
from lbyr.config import PlannerConfig
from lbyr.errors import PlanValidationError
from lbyr.perception.detector import Detection

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApproachSkill:
    target_id: int
    standoff_mm: float


@dataclass(frozen=True)
class GraspSkill:
    closure: float  # percent, 0 = open, 100 = fully closed


@dataclass(frozen=True)
class PlaceSkill:
    target_id: int


Skill = ApproachSkill | GraspSkill | PlaceSkill


@dataclass(frozen=True)
class SkillChain:
    approach: ApproachSkill
    grasp: GraspSkill
    place: PlaceSkill
    notes: str = ""


@dataclass(frozen=True)
class PlanRequest:
    instruction: str
    detections: Sequence[Detection]
    image_bgr: np.ndarray | None = None


@dataclass(frozen=True)
class PlanResult:
    chain: SkillChain | None
    latency_ms: float
    raw_text: str | None
    error: str | None = None  # "timeout" or a validation message


class Planner(Protocol):
    def generate(self, request: PlanRequest) -> str:
        """Return the raw model text (expected to be the JSON contract)."""
        ...


# ------------------------------------------------------------------ prompt
def build_planner_prompt(request: PlanRequest) -> str:
    objs = "\n".join(f"- id={d.id} label={d.label} score={d.score:.2f}" for d in request.detections)
    return (
        "You are a robot task planner. Output ONLY JSON, no prose.\n"
        'Schema: {"skill_chain":[{"skill":"approach","target_id":int,"standoff_mm":number},'
        '{"skill":"grasp","closure":0-100},{"skill":"place","target_id":int}],"notes":string}\n'
        "Never output joint angles. Use only the object ids listed.\n"
        f"Detected objects:\n{objs}\n"
        f"Instruction: {request.instruction}"
    )


# ------------------------------------------------------------------ parsing
def _strip_code_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```") and s.endswith("```"):
        s = s[3:-3].strip()
        if s.lower().startswith("json"):
            s = s[4:].strip()
    return s


def _num(obj: dict[str, Any], key: str, lo: float, hi: float) -> float:
    v = obj.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise PlanValidationError(f"{key}: expected a number, got {v!r}")
    if not lo <= v <= hi:
        raise PlanValidationError(f"{key}={v} outside [{lo}, {hi}]")
    return float(v)


def _int_id(obj: dict[str, Any], known_ids: set[int] | None) -> int:
    v = obj.get("target_id")
    if isinstance(v, bool) or not isinstance(v, int):
        raise PlanValidationError(f"target_id: expected integer, got {v!r}")
    if known_ids is not None and v not in known_ids:
        raise PlanValidationError(f"target_id {v} is not a detected object {sorted(known_ids)}")
    return v


_SKILL_KEYS = {
    "approach": {"skill", "target_id", "standoff_mm"},
    "grasp": {"skill", "closure"},
    "place": {"skill", "target_id"},
}
_ORDER = ("approach", "grasp", "place")


def parse_skill_chain(text: str, known_ids: set[int] | None = None) -> SkillChain:
    """Strict parse + validation. Only a surrounding ```json fence is tolerated."""
    try:
        obj = json.loads(_strip_code_fence(text))
    except json.JSONDecodeError as e:
        raise PlanValidationError(f"planner output is not JSON: {e.msg}") from e
    if not isinstance(obj, dict) or "skill_chain" not in obj:
        raise PlanValidationError("expected an object with 'skill_chain'")
    if set(obj) - {"skill_chain", "notes"}:
        raise PlanValidationError(
            f"unexpected top-level keys {sorted(set(obj) - {'skill_chain', 'notes'})}"
        )
    notes = obj.get("notes", "")
    if not isinstance(notes, str):
        raise PlanValidationError("notes must be a string")
    steps = obj["skill_chain"]
    if not isinstance(steps, list) or [
        s.get("skill") if isinstance(s, dict) else None for s in steps
    ] != list(_ORDER):
        raise PlanValidationError(f"skill_chain must be exactly {list(_ORDER)} in order")
    for step in steps:
        if set(step) != _SKILL_KEYS[step["skill"]]:
            raise PlanValidationError(
                f"{step['skill']}: keys must be {sorted(_SKILL_KEYS[step['skill']])}"
            )
    a, g, p = steps
    return SkillChain(
        approach=ApproachSkill(_int_id(a, known_ids), _num(a, "standoff_mm", 0.0, 200.0)),
        grasp=GraspSkill(_num(g, "closure", 0.0, 100.0)),
        place=PlaceSkill(_int_id(p, known_ids)),
        notes=notes,
    )


def plan_with_timeout(planner: Planner, request: PlanRequest, cfg: PlannerConfig) -> PlanResult:
    """Call the planner under the hard timeout, then strictly validate its JSON."""
    t0 = time.perf_counter()
    try:
        out = run_with_timeout(lambda: planner.generate(request), cfg.timeout_s, "planner")
    except Exception as e:  # backend crash -> treated like invalid output
        log.exception("planner backend raised")
        return PlanResult(None, (time.perf_counter() - t0) * 1000, None, f"backend error: {e}")
    latency_ms = (time.perf_counter() - t0) * 1000
    if isinstance(out, TimeoutResult):
        return PlanResult(None, latency_ms, None, "timeout")
    try:
        chain = parse_skill_chain(out, {d.id for d in request.detections})
    except PlanValidationError as e:
        return PlanResult(None, latency_ms, out, str(e))
    return PlanResult(chain, latency_ms, out)


# ------------------------------------------------------------------ backends
class FakePlanner:
    """Deterministic planner for mock mode: pick the first non-zone object, place on a zone."""

    def __init__(
        self, closure: float = 80.0, standoff_mm: float = 40.0, delay_s: float = 0.0
    ) -> None:
        self.closure = closure
        self.standoff_mm = standoff_mm
        self.delay_s = delay_s

    def generate(self, request: PlanRequest) -> str:
        if self.delay_s:
            time.sleep(self.delay_s)
        objs = [d for d in request.detections if "zone" not in d.label]
        zones = [d for d in request.detections if "zone" in d.label]
        if not objs or not zones:
            return '{"skill_chain": [], "notes": "nothing to do"}'
        chain = {
            "skill_chain": [
                {"skill": "approach", "target_id": objs[0].id, "standoff_mm": self.standoff_mm},
                {"skill": "grasp", "closure": self.closure},
                {"skill": "place", "target_id": zones[0].id},
            ],
            "notes": "fake planner",
        }
        return json.dumps(chain)


class QwenPlanner:
    """Qwen2.5-VL-3B-Instruct FP16 (bible 9.2: ~6.5 GB VRAM, 0.8-2 s).

    TODO: implement on the Jetson with local weights only (``cfg.model_path``), greedy
    decoding (``do_sample=False`` == temperature 0). transformers/torch imported lazily.
    """

    def __init__(self, cfg: PlannerConfig) -> None:
        cfg.validate()
        self.cfg = cfg
        raise NotImplementedError("QwenPlanner not implemented yet (TODO); use FakePlanner")

    def generate(self, request: PlanRequest) -> str:
        raise NotImplementedError
