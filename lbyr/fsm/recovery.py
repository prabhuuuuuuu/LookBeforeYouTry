"""Failure taxonomy and recovery policy as DATA (bible Section 13).

- SLIPPING (Critic=SLIPPING after close/lift), max 2:
  +15% closure -> wrist re-orient 30 deg -> lateral offset 10 mm
- EMPTY (Critic=EMPTY after lift), max 2: re-approach with -5 mm standoff -> re-grasp
- MISS (Critic=EMPTY after close), max 2: re-plan from PERCEIVE
- JAMMED (load/position-error spike or motion timeout), max 0:
  stop -> torque off -> human report
- OCCLUDED (Critic=OCCLUDED), max 1: re-position wrist cam -> re-query once
- WRONG_OBJECT (planner re-check vs detected class), max 1: re-plan

Software-added classes (NOT in the bible table; see docs/OPEN_QUESTIONS.md):
CRITIC_UNAVAILABLE (TIMEOUT/INVALID: re-query once), PLACE_FAILED, PLANNER_FAILED,
PERCEPTION_FAILED, IK_FAILED (all 0 attempts -> ABORT).

Interpretations (approved): "+15%" = +15 percentage points of closure, capped at 100;
"wrist re-orient" = wrist roll; the 3rd SLIPPING step is unreachable while max = 2.
Besides per-class limits, a global budget caps RECOVER(k) at k <= 2 per episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from lbyr.ai.critic import Verdict
from lbyr.fsm.states import State


class FailureClass(str, Enum):
    SLIPPING = "SLIPPING"
    EMPTY = "EMPTY"
    MISS = "MISS"
    JAMMED = "JAMMED"
    OCCLUDED = "OCCLUDED"
    WRONG_OBJECT = "WRONG_OBJECT"
    # software-added
    CRITIC_UNAVAILABLE = "CRITIC_UNAVAILABLE"
    PLACE_FAILED = "PLACE_FAILED"
    PLANNER_FAILED = "PLANNER_FAILED"
    PERCEPTION_FAILED = "PERCEPTION_FAILED"
    IK_FAILED = "IK_FAILED"


@dataclass(frozen=True)
class RecoveryAction:
    description: str
    reentry: State | None  # None = re-enter the verify state that failed (re-query)
    closure_delta_pct: float = 0.0
    wrist_roll_delta_deg: float = 0.0
    lateral_offset_mm: float = 0.0
    standoff_delta_mm: float = 0.0
    reposition_wrist_cam: bool = False  # wrist-pitch step (RecoveryConfig) inside RECOVER


@dataclass(frozen=True)
class RecoveryPolicy:
    signal: str
    max_attempts: int
    ladder: tuple[RecoveryAction, ...] = ()
    counts_toward_total: bool = True  # False = a re-query, not a physical RECOVER(k)

    def action(self, attempt: int) -> RecoveryAction:
        """Action for the 1-based attempt number (last step repeats if the ladder is short)."""
        return self.ladder[min(attempt, len(self.ladder)) - 1]


def _abort_only(signal: str) -> RecoveryPolicy:
    return RecoveryPolicy(signal, 0)


RECOVERY_TABLE: dict[FailureClass, RecoveryPolicy] = {
    FailureClass.SLIPPING: RecoveryPolicy(
        "Critic=SLIPPING after close/lift",
        2,
        (
            RecoveryAction("+15% closure", State.GRASP, closure_delta_pct=15.0),
            RecoveryAction("wrist re-orient 30 deg", State.APPROACH, wrist_roll_delta_deg=30.0),
            RecoveryAction("lateral offset 10 mm", State.APPROACH, lateral_offset_mm=10.0),
        ),
    ),
    FailureClass.EMPTY: RecoveryPolicy(
        "Critic=EMPTY after lift",
        2,
        (RecoveryAction("re-approach, -5 mm standoff", State.APPROACH, standoff_delta_mm=-5.0),),
    ),
    FailureClass.MISS: RecoveryPolicy(
        "Critic=EMPTY after close",
        2,
        (RecoveryAction("re-plan from PERCEIVE", State.PERCEIVE),),
    ),
    FailureClass.JAMMED: _abort_only("load/position-error spike or motion timeout"),
    FailureClass.OCCLUDED: RecoveryPolicy(
        "Critic=OCCLUDED",
        1,
        (RecoveryAction("re-position wrist cam, re-query", None, reposition_wrist_cam=True),),
    ),
    FailureClass.WRONG_OBJECT: RecoveryPolicy(
        "planner target class != expected class",
        1,
        (RecoveryAction("re-plan", State.PLAN),),
    ),
    FailureClass.CRITIC_UNAVAILABLE: RecoveryPolicy(
        "Critic TIMEOUT/INVALID",
        1,
        (RecoveryAction("re-query Critic once", None),),
        counts_toward_total=False,
    ),
    FailureClass.PLACE_FAILED: _abort_only("object still in gripper after place"),
    FailureClass.PLANNER_FAILED: _abort_only("planner timeout / invalid JSON"),
    FailureClass.PERCEPTION_FAILED: _abort_only("no frame / no detections / no depth"),
    FailureClass.IK_FAILED: _abort_only("target unreachable"),
}

_VERDICT_MAP: dict[State, dict[Verdict, FailureClass | None]] = {
    State.VERIFY_GRASP: {
        Verdict.GRASPED: None,
        Verdict.SLIPPING: FailureClass.SLIPPING,
        Verdict.EMPTY: FailureClass.MISS,  # EMPTY after close = MISS
    },
    State.VERIFY_HOLD: {
        Verdict.GRASPED: None,
        Verdict.SLIPPING: FailureClass.SLIPPING,
        Verdict.EMPTY: FailureClass.EMPTY,  # EMPTY after lift = EMPTY
    },
    State.VERIFY_PLACE: {
        Verdict.EMPTY: None,  # released = success
        Verdict.GRASPED: FailureClass.PLACE_FAILED,
        Verdict.SLIPPING: FailureClass.PLACE_FAILED,
    },
}


def classify(state: State, verdict: Verdict) -> FailureClass | None:
    """Map (verify state, verdict) to a failure class, or None for success."""
    if verdict is Verdict.OCCLUDED:
        return FailureClass.OCCLUDED
    if verdict in (Verdict.TIMEOUT, Verdict.INVALID):
        return FailureClass.CRITIC_UNAVAILABLE
    return _VERDICT_MAP[state][verdict]


@dataclass
class RecoveryBudget:
    max_total: int = 2
    per_class: dict[FailureClass, int] = field(default_factory=dict)
    history: list[tuple[FailureClass, str]] = field(default_factory=list)
    total: int = 0  # RECOVER(k) count that is capped by max_total

    def next_action(self, failure: FailureClass) -> RecoveryAction | None:
        """Consume one attempt and return its action, or None if we must ABORT."""
        policy = RECOVERY_TABLE[failure]
        used = self.per_class.get(failure, 0)
        if used >= policy.max_attempts:
            return None
        if policy.counts_toward_total and self.total >= self.max_total:
            return None
        self.per_class[failure] = used + 1
        if policy.counts_toward_total:
            self.total += 1
        action = policy.action(used + 1)
        self.history.append((failure, action.description))
        return action
