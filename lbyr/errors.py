"""Project-wide exception types."""

from __future__ import annotations


class LBYRError(Exception):
    """Base class for all LBYR errors."""


class HardwareNotConfirmedError(LBYRError):
    """Raised when code needs a hardware value that has not been physically confirmed.

    Every such value lives in config as ``None`` and is tied to an open question
    (bible Section 20, or the software-surfaced list in docs/OPEN_QUESTIONS.md).
    We refuse to guess instead of silently falling back to a default.
    """

    def __init__(self, what: str, question: str) -> None:
        self.what = what
        self.question = question
        super().__init__(
            f"{what} is not confirmed ({question}). "
            "See docs/OPEN_QUESTIONS.md; refusing to guess a hardware value."
        )


class ProtocolError(LBYRError):
    """Malformed or invalid bridge message."""


class PlanValidationError(LBYRError):
    """Planner output failed strict JSON / schema validation."""


class CalibrationError(LBYRError):
    """Calibration file missing, malformed, or numerically invalid."""


class IKError(LBYRError):
    """Target unreachable or outside joint limits."""


class ConfigError(LBYRError):
    """Configuration violates a locked project rule (e.g. temperature != 0)."""


class InsufficientMemoryError(LBYRError):
    """Startup memory guard: not enough free system RAM to run the episode loop safely."""


class PlannerTimeoutError(LBYRError):
    """A planner backend gave up within the hard timeout; reported as ``error="timeout"``."""
