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
import os
import re
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from lbyr.ai.cuda_mem import log_cuda_allocated
from lbyr.ai.imaging import bgr_to_pil
from lbyr.ai.timeout import TimeoutResult, TimeoutRunner, run_with_timeout
from lbyr.config import PlannerConfig, SkillConfig
from lbyr.errors import ConfigError, LBYRError, PlannerTimeoutError, PlanValidationError
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
def build_planner_prompt(request: PlanRequest, bbox_scale: float = 1.0) -> str:
    """Task text + detected objects (id, class, score, pixel bbox). The image is passed
    separately. ``bbox_scale`` converts bboxes if the image given to the model was resized."""

    def _bbox(d: Detection) -> str:
        return "[" + ",".join(str(round(v * bbox_scale)) for v in d.bbox_xyxy) + "]"

    objs = "\n".join(
        f"- id={d.id} label={d.label} score={d.score:.2f} bbox_xyxy_px={_bbox(d)}"
        for d in request.detections
    )
    return (
        "You are a robot task planner. Output ONLY minified single-line JSON, no prose.\n"
        'Schema: {"skill_chain":[{"skill":"approach","target_id":int,"standoff_mm":number},'
        '{"skill":"grasp","closure":0-100},{"skill":"place","target_id":int}],"notes":string}\n'
        'Use "" for notes. Never output joint angles. Use only the object ids listed.\n'
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


@dataclass(frozen=True)
class GenStats:
    new_tokens: int
    seconds: float  # wall time of the engine call: image preprocessing + prefill + decode

    @property
    def tokens_per_s(self) -> float:
        return self.new_tokens / self.seconds if self.seconds > 0 else 0.0


class QwenEngine(Protocol):
    def run(
        self, prompt: str, image: Any | None, max_new_tokens: int, max_time_s: float | None
    ) -> tuple[str, int]:
        """Greedy-decode. Returns (text, number of generated tokens)."""
        ...


class _HFQwenEngine:
    """transformers Qwen2.5-VL-3B-Instruct, FP16, greedy. torch is imported lazily."""

    def __init__(self, path: Path, device: str) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # edge-only: never touch the network
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self._torch = torch
        self._device = device
        self._processor: Any = AutoProcessor.from_pretrained(str(path), local_files_only=True)
        self._model: Any = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(path), dtype=torch.float16, device_map={"": device}, local_files_only=True
        ).eval()

    def run(
        self, prompt: str, image: Any | None, max_new_tokens: int, max_time_s: float | None
    ) -> tuple[str, int]:
        content: list[dict[str, str]] = [{"type": "text", "text": prompt}]
        if image is not None:
            content.insert(0, {"type": "image"})
        text = self._processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=[image] if image is not None else None, return_tensors="pt"
        ).to(self._device)
        with self._torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy == temperature 0
                temperature=None,
                top_p=None,
                top_k=None,
                max_time=max_time_s,  # an abandoned (timed-out) call stops itself
            )
        new = out[0, inputs["input_ids"].shape[1] :]
        return self._processor.batch_decode([new], skip_special_tokens=True)[0], int(new.shape[0])


class QwenPlanner:
    """Qwen2.5-VL-3B-Instruct FP16 (bible 9.2: ~6.5 GB VRAM, 0.8-2 s; measured numbers in
    the README / ``tools/bench_models.py``).

    Local weights only (``cfg.model_path``). Greedy decoding (temperature 0), at most
    ``cfg.max_new_tokens`` tokens, and generation also stops by itself after
    ``cfg.timeout_s``. Returns the RAW text; :func:`plan_with_timeout` owns the hard timeout
    and the strict :func:`parse_skill_chain` validation, so joint angles are never accepted.
    ``engine`` injects a fake (tests). ``last_stats`` has the token count / time of the
    latest call.
    """

    def __init__(self, cfg: PlannerConfig, *, engine: QwenEngine | None = None) -> None:
        cfg.validate()
        self.cfg = cfg
        self.last_stats: GenStats | None = None
        self._lock = threading.Lock()  # one generation at a time
        if engine is None:
            if cfg.model_path is None:
                raise ConfigError("PlannerConfig.model_path is not set (local weights only)")
            t0 = time.perf_counter()
            engine = _HFQwenEngine(cfg.model_path, cfg.device)
            log.info("Qwen2.5-VL loaded in %.1f s", time.perf_counter() - t0)
            log_cuda_allocated("Qwen2.5-VL planner")
        self._engine = engine
        self._warm_up()

    def _warm_up(self) -> None:
        dummy = PlanRequest(
            "pick up the block",
            [Detection(1, "block", 0.9, (200.0, 150.0, 300.0, 250.0))],
            np.zeros((480, 640, 3), dtype=np.uint8),
        )
        t0 = time.perf_counter()
        out = run_with_timeout(
            lambda: self._generate(dummy, 8, self.cfg.timeout_s),
            self.cfg.warmup_timeout_s,
            "qwen-warmup",
        )
        if isinstance(out, TimeoutResult):
            raise LBYRError(f"Planner warm-up exceeded {self.cfg.warmup_timeout_s:.0f} s")
        log.info("Planner warm-up in %.0f ms", (time.perf_counter() - t0) * 1000)

    def generate(self, request: PlanRequest) -> str:
        return self._generate(request, self.cfg.max_new_tokens, self.cfg.timeout_s)

    def generate_untimed(self, request: PlanRequest) -> str:
        """BENCHMARK/TOOLS ONLY: like :meth:`generate` but without the self-imposed
        ``max_time`` cut-off, to measure the true full-plan latency. Never use at runtime."""
        return self._generate(request, self.cfg.max_new_tokens, None)

    def _generate(self, request: PlanRequest, max_new_tokens: int, max_time_s: float | None) -> str:
        image: Any | None = None
        scale = 1.0
        if request.image_bgr is not None:
            image = bgr_to_pil(request.image_bgr)
            w, h = image.size
            if w * h > self.cfg.max_image_pixels:  # bound visual tokens (latency)
                scale = math.sqrt(self.cfg.max_image_pixels / (w * h))
                image = image.resize((max(28, round(w * scale)), max(28, round(h * scale))))
        prompt = build_planner_prompt(request, bbox_scale=scale)
        with self._lock:
            t0 = time.perf_counter()
            text, n_tokens = self._engine.run(prompt, image, max_new_tokens, max_time_s)
            self.last_stats = GenStats(n_tokens, time.perf_counter() - t0)
        return text
