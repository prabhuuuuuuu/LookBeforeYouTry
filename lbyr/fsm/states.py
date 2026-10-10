"""FSM states, exactly per bible Section 10, plus RECOVER and ABORT."""

from __future__ import annotations

from enum import Enum


class State(str, Enum):
    IDLE = "IDLE"
    PERCEIVE = "PERCEIVE"
    PLAN = "PLAN"
    APPROACH = "APPROACH"
    GRASP = "GRASP"
    VERIFY_GRASP = "VERIFY_GRASP"
    LIFT = "LIFT"
    VERIFY_HOLD = "VERIFY_HOLD"
    TRANSPORT = "TRANSPORT"
    PLACE = "PLACE"
    VERIFY_PLACE = "VERIFY_PLACE"
    DONE = "DONE"
    RECOVER = "RECOVER"  # RECOVER(k), k <= 2
    ABORT = "ABORT"


# Locked happy-path order (bible Section 10).
PIPELINE: tuple[State, ...] = (
    State.IDLE,
    State.PERCEIVE,
    State.PLAN,
    State.APPROACH,
    State.GRASP,
    State.VERIFY_GRASP,
    State.LIFT,
    State.VERIFY_HOLD,
    State.TRANSPORT,
    State.PLACE,
    State.VERIFY_PLACE,
    State.DONE,
)

VERIFY_STATES = frozenset({State.VERIFY_GRASP, State.VERIFY_HOLD, State.VERIFY_PLACE})
TERMINAL_STATES = frozenset({State.DONE, State.ABORT})


def next_in_pipeline(state: State) -> State:
    i = PIPELINE.index(state)
    if i + 1 >= len(PIPELINE):
        raise ValueError(f"{state} has no successor")
    return PIPELINE[i + 1]
