"""Language prompting for the live demo: instruction -> ONE block move between named
locations (data/calib/poses.yaml ``locations``). The VLM never outputs joints: it picks
location NAMES; the taught poses of those locations drive the arm.

Strict output contract (anything else is rejected, nothing moves)::

    {"action": "move", "from": "<location>", "to": "<location>", "reason": "<one sentence>"}
    {"action": "none", "reason": "<why it can't/shouldn't do it>"}

Qwen2.5-VL-3B (the loaded planner, no extra VRAM): overhead image + instruction + the
location list + the detector's block location; greedy, ``timeout_s`` (DemoConfig, 30 s,
arm idle, never in the FSM loop). Latency ~ the demo intro plan (one image prefill +
<= ``max_new_tokens`` tokens). :class:`RulesTaskPlanner` is the no-model fallback
(--planner rules / rehearsal / dry-run): it only understands "from X to Y" / "to Y".
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Any

from lbyr.ai.planner import _strip_code_fence
from lbyr.ai.timeout import TimeoutResult, run_with_timeout
from lbyr.errors import PlanValidationError

MAX_INSTRUCTION_CHARS = 300
UNKNOWN = "unknown"


@dataclass(frozen=True)
class TaskAction:
    action: str  # "move" | "none"
    reason: str
    src: str | None = None  # "from"
    dst: str | None = None  # "to"

    def to_dict(self) -> dict[str, Any]:
        if self.action == "move":
            return {"action": "move", "from": self.src, "to": self.dst, "reason": self.reason}
        return {"action": "none", "reason": self.reason}


@dataclass(frozen=True)
class TaskPlanResult:
    backend: str  # "qwen" | "rules"
    prompt: str
    raw_text: str
    action: TaskAction | None  # None = invalid / timeout / failure (error says why)
    error: str | None
    latency_ms: float


def build_task_prompt(
    instruction: str,
    locations: Mapping[str, str],
    block_at: str,
    has_image: bool,
    labelled: bool = False,
) -> str:
    """``locations``: name -> description (poses.yaml order). Grounding rule: Qwen cannot
    tell which area is "A" from the image unless it is labelled, so it must NOT veto a move
    by guessing where locations are: "none" only for no block at all / an unlisted place."""
    names = ", ".join(f'"{n}"' for n in locations)
    lines = "\n".join(f"- {n}: {d or '(no description)'}" for n, d in locations.items())
    if not has_image:
        seen = "There is no camera image: assume the block is on the table.\n"
    elif labelled:
        seen = (
            "The image is the overhead camera view of the table. Each location is outlined "
            "and labelled with its name in large yellow letters.\n"
        )
    else:
        seen = (
            "The image is the overhead camera view of the table. Locations may be marked "
            "with printed letters on the table; their positions are known to the robot.\n"
        )
    return (
        "You control a robot arm that moves one wooden block between fixed table locations.\n"
        f"{seen}"
        f"Locations (use these exact names: {names}):\n{lines}\n"
        f"Block location reported by the detector: {block_at}\n"
        f"Instruction: {json.dumps(instruction)}\n"
        'Follow the instruction: set "from" and "to" to the locations it names (if it '
        'names only the target, "from" is where the block is). Do not refuse because you '
        "are unsure where a location is on the table: the robot knows its positions.\n"
        'Answer "none" ONLY if no block is visible on the table at all, or the instruction '
        "names a location that is not in the list.\n"
        "Answer with ONE JSON object and nothing else. Either\n"
        '{"action": "move", "from": "<location>", "to": "<location>", '
        '"reason": "<one sentence>"}\n'
        "or\n"
        '{"action": "none", "reason": "<why it can\'t do it>"}'
    )


def _reason(obj: dict[str, Any]) -> str:
    r = obj.get("reason")
    if not isinstance(r, str) or not r.strip():
        raise PlanValidationError('"reason" must be a non-empty string')
    return r.strip()


def parse_task_action(text: str, names: Collection[str]) -> TaskAction:
    """Strict: exactly the contract keys, known location names, from != to."""
    try:
        obj = json.loads(_strip_code_fence(text))
    except json.JSONDecodeError as e:
        raise PlanValidationError(f"not JSON: {e.msg}") from e
    if not isinstance(obj, dict):
        raise PlanValidationError(f"expected a JSON object, got {type(obj).__name__}")
    action = obj.get("action")
    if action == "none":
        if set(obj) != {"action", "reason"}:
            raise PlanValidationError(f'"none" keys must be [action, reason]: {sorted(obj)}')
        return TaskAction("none", _reason(obj))
    if action != "move":
        raise PlanValidationError(f'"action" must be "move" or "none", got {action!r}')
    if set(obj) != {"action", "from", "to", "reason"}:
        raise PlanValidationError(f'"move" keys must be [action, from, to, reason]: {sorted(obj)}')
    src, dst = obj["from"], obj["to"]
    known = list(names)
    for key, val in (("from", src), ("to", dst)):
        if not isinstance(val, str) or val not in known:
            raise PlanValidationError(f'"{key}": {val!r} is not a location in poses.yaml {known}')
    if src == dst:
        raise PlanValidationError(f'"from" and "to" are the same location ({src!r})')
    return TaskAction("move", _reason(obj), src, dst)


class RulesTaskPlanner:
    """No-model fallback: "from X to Y" or "to Y" (X = the block's known location).
    Location names match case-insensitively as whole words. Emits contract JSON."""

    backend_name = "rules"

    def generate(self, instruction: str, names: Collection[str], block_at: str) -> str:
        by_lower = {n.lower(): n for n in names}
        text = instruction.lower()

        def find(word: str) -> str | None:
            m = re.search(rf"\b{word}\s+(?:the\s+)?([\w-]+)", text)
            return by_lower.get(m.group(1)) if m else None

        src, dst = find("from"), find("to") or find("into") or find("at")
        if src is None and block_at in names:
            src = block_at
        if src is None or dst is None:
            why = "rules planner (no Qwen) only understands 'from <location> to <location>'"
            return json.dumps({"action": "none", "reason": why})
        why = f"rules planner: '{src}' -> '{dst}' named in the instruction"
        return json.dumps({"action": "move", "from": src, "to": dst, "reason": why})


def plan_task(
    generate: Callable[[], str],
    names: Collection[str],
    timeout_s: float,
    backend: str,
    prompt: str,
) -> TaskPlanResult:
    """Run ``generate`` under the hard timeout wrapper, then :func:`parse_task_action`."""
    t0 = time.perf_counter()
    raw, action, error = "", None, None
    try:
        out = run_with_timeout(generate, timeout_s, f"task-planner-{backend}")
    except Exception as e:  # model failure: report, never move
        out, error = "", f"planner failed: {e}"
    if isinstance(out, TimeoutResult):
        error = f"timeout ({timeout_s:.0f} s)"
    elif error is None:
        raw = str(out)
        try:
            action = parse_task_action(raw, names)
        except PlanValidationError as e:
            error = f"invalid output: {e}"
    ms = (time.perf_counter() - t0) * 1000
    return TaskPlanResult(backend, prompt, raw, action, error, ms)


def clean_instruction(text: str) -> str:
    s = " ".join(str(text).split())
    if not s:
        raise PlanValidationError("empty instruction")
    if len(s) > MAX_INSTRUCTION_CHARS:
        raise PlanValidationError(f"instruction longer than {MAX_INSTRUCTION_CHARS} characters")
    return s
