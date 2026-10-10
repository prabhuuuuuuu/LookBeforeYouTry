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
import re
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from lbyr.ai.timeout import TimeoutResult, TimeoutRunner, run_with_timeout
from lbyr.config import PlannerConfig, SkillConfig
from lbyr.errors import PlannerTimeoutError, PlanValidationError
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
    except PlannerTimeoutError:  # the planner's own single-flight timeout (FallbackPlanner)
        return PlanResult(None, (time.perf_counter() - t0) * 1000, None, "timeout")
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


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _match_score(label: str, instruction_words: list[str]) -> tuple[int, int]:
    """(full-phrase match 0/1, number of label words found in the instruction)."""
    lw = _words(label)
    if not lw:
        return 0, 0
    n = len(lw)
    phrase = any(instruction_words[i : i + n] == lw for i in range(len(instruction_words) - n + 1))
    return int(phrase), sum(1 for w in lw if w in instruction_words)


def _best_match(dets: Sequence[Detection], instruction: str) -> Detection | None:
    """Detection whose class name best matches the instruction; None if nothing matches.

    Full class-name phrase beats partial word overlap; ties go to the higher detector score,
    then the lower id (deterministic).
    """
    iw = _words(instruction)
    best: tuple[tuple[int, int, float, int], Detection] | None = None
    for d in dets:
        phrase, overlap = _match_score(d.label, iw)
        if overlap == 0:
            continue
        key = (phrase, overlap, d.score, -d.id)
        if best is None or key > best[0]:
            best = (key, d)
    return None if best is None else best[1]


class RuleBasedPlanner:
    """No-model planner: pick the pick-target by class-name match against the instruction.

    Picks the non-zone detection whose label best matches the instruction, and places it on
    the zone detection (a label containing "zone") the instruction names, else the first zone
    by id. Emits the same JSON contract as the VLM: approach -> grasp -> place (LIFT is an FSM
    state, not a planner skill; the strict parser accepts exactly these three). If nothing
    matches it emits an empty chain, which the strict parser rejects (PLANNER_FAILED).
    Cost: pure Python, microseconds, no VRAM.
    """

    def __init__(self, closure: float = 80.0, standoff_mm: float = 40.0) -> None:
        self.closure = closure
        self.standoff_mm = standoff_mm

    def generate(self, request: PlanRequest) -> str:
        objs = [d for d in request.detections if "zone" not in d.label]
        zones = sorted((d for d in request.detections if "zone" in d.label), key=lambda d: d.id)
        target = _best_match(objs, request.instruction)
        if target is None or not zones:
            return json.dumps({"skill_chain": [], "notes": "rules: no matching target or zone"})
        zone = _best_match(zones, request.instruction) or zones[0]
        return json.dumps(
            {
                "skill_chain": [
                    {"skill": "approach", "target_id": target.id, "standoff_mm": self.standoff_mm},
                    {"skill": "grasp", "closure": self.closure},
                    {"skill": "place", "target_id": zone.id},
                ],
                "notes": f"rules: {target.label} -> {zone.label}",
            }
        )


# Margin so the single-flight timeout below fires before plan_with_timeout's outer one.
_OUTER_MARGIN_S = 0.15


class FallbackPlanner:
    """Primary (VLM) planner that degrades to a fallback after repeated timeouts.

    The primary runs under its own single-flight :class:`TimeoutRunner` (a bit shorter than
    the hard ``cfg.timeout_s`` enforced by :func:`plan_with_timeout`, so GPU work never
    stacks up behind an abandoned call). Timeouts (and a still-busy abandoned call) below
    ``cfg.fallback_after_timeouts`` in a row are reported as ``"timeout"``; the Nth one
    switches permanently to the fallback and that same request is answered by it. Any
    completed primary call resets the counter. Other errors are not counted: they propagate.
    """

    def __init__(
        self,
        primary: Planner,
        fallback: Planner,
        cfg: PlannerConfig,
        primary_name: str = "qwen",
        fallback_name: str = "rules",
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._limit = cfg.fallback_after_timeouts
        self._names = (primary_name, fallback_name)
        self._runner = TimeoutRunner(max(0.05, cfg.timeout_s - _OUTER_MARGIN_S), "planner-primary")
        self._lock = threading.Lock()
        self._timeouts = 0
        self._switched = False

    @property
    def active_backend(self) -> str:
        return self._names[1] if self._switched else self._names[0]

    @property
    def consecutive_timeouts(self) -> int:
        return self._timeouts

    def reset(self) -> None:
        """Go back to the primary backend (e.g. between trials)."""
        with self._lock:
            self._timeouts = 0
            self._switched = False

    def generate(self, request: PlanRequest) -> str:
        if self._switched:
            return self._fallback.generate(request)
        out = self._runner.run(lambda: self._primary.generate(request))
        if isinstance(out, TimeoutResult):
            with self._lock:
                self._timeouts += 1
                n = self._timeouts
                if n >= self._limit:
                    self._switched = True
            log.warning(
                "%s planner timed out (%d/%d in a row%s)",
                self._names[0],
                n,
                self._limit,
                ", busy" if out.busy else "",
            )
            if n < self._limit:
                raise PlannerTimeoutError(f"{self._names[0]} planner timed out")
            log.error("Switching planner backend %s -> %s", *self._names)
            return self._fallback.generate(request)
        with self._lock:
            self._timeouts = 0
        return out


def build_planner(cfg: PlannerConfig, skills: SkillConfig | None = None) -> Planner:
    """Planner for ``cfg.backend``: "rules", or "qwen" with automatic fallback to rules."""
    cfg.validate()
    skills = skills or SkillConfig()
    rules = RuleBasedPlanner(skills.default_closure_pct, skills.default_standoff_mm)
    if cfg.backend == "rules":
        return rules
    return FallbackPlanner(QwenPlanner(cfg), rules, cfg)


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
