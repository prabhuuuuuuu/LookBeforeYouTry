"""Text protocol of the XIAO ESP32-C6 bridge firmware (firmware/xiao_c6_bridge/Code.ino).

Host -> board (newline-terminated): ``m D1 .. D6`` (zero-relative degrees), ``q``,
``list``, ``x``, ``off``, ``home``, ... Board -> host, one line each::

    OK ... / ERR ...
    DONE                                   all attached joints settled after a motion
    Q,<moving 0|1>,<cur_us j1..j6>         reply to q (commanded pulses, no feedback)
    S,j,zero_us,cur_us,attached,goal_us,moving,name    per joint, from list
    S,j,zero_us,cur_us,attached,name                   (older firmware, still parsed)
    V,vmax_deg_s,amax_deg_s2               last line of list

Anything else (help text, test messages) is an :class:`InfoMsg`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from lbyr.errors import ProtocolError

COUNT = 6


@dataclass(frozen=True)
class DoneMsg:
    pass


@dataclass(frozen=True)
class StatusMsg:
    moving: bool
    cur_us: tuple[float, ...]


@dataclass(frozen=True)
class JointLine:
    joint: int  # 1-based
    zero_us: int
    cur_us: float
    attached: bool
    goal_us: float | None  # None on older firmware
    moving: bool | None
    name: str


@dataclass(frozen=True)
class SpeedLine:
    vmax_deg: float
    amax_deg: float


@dataclass(frozen=True)
class OkMsg:
    text: str


@dataclass(frozen=True)
class ErrMsg:
    text: str


@dataclass(frozen=True)
class InfoMsg:
    text: str


Message = DoneMsg | StatusMsg | JointLine | SpeedLine | OkMsg | ErrMsg | InfoMsg


def _flag(s: str, what: str) -> bool:
    if s not in ("0", "1"):
        raise ProtocolError(f"{what}: expected 0|1, got {s!r}")
    return s == "1"


def _num(s: str, what: str) -> float:
    try:
        v = float(s)
    except ValueError as e:
        raise ProtocolError(f"{what}: not a number: {s!r}") from e
    if not math.isfinite(v):
        raise ProtocolError(f"{what}: not finite")
    return v


def parse_line(line: str | bytes) -> Message:
    """Parse one board line. Raises :class:`ProtocolError` for malformed Q/S/V lines."""
    if isinstance(line, bytes):
        line = line.decode("ascii", errors="replace")
    text = line.strip()
    if text == "DONE":
        return DoneMsg()
    if text.startswith("OK"):
        return OkMsg(text)
    if text.startswith("ERR"):
        return ErrMsg(text)
    parts = text.split(",")
    tag = parts[0]
    if tag == "Q" and len(parts) > 1:
        if len(parts) != 2 + COUNT:
            raise ProtocolError(f"Q line needs {2 + COUNT} fields: {text!r}")
        return StatusMsg(
            _flag(parts[1], "Q moving"),
            tuple(_num(p, "Q cur_us") for p in parts[2:]),
        )
    if tag == "S" and len(parts) >= 6:
        try:
            joint, zero = int(parts[1]), int(parts[2])
        except ValueError as e:
            raise ProtocolError(f"S line joint/zero not integers: {text!r}") from e
        if not 1 <= joint <= COUNT:
            raise ProtocolError(f"S line joint out of range: {text!r}")
        cur = _num(parts[3], "S cur_us")
        attached = _flag(parts[4], "S attached")
        if len(parts) >= 8 and parts[6] in ("0", "1"):
            try:
                goal: float | None = float(parts[5])
            except ValueError:
                goal = None
            if goal is not None:
                return JointLine(
                    joint, zero, cur, attached, goal, parts[6] == "1", ",".join(parts[7:])
                )
        return JointLine(joint, zero, cur, attached, None, None, ",".join(parts[5:]))
    if tag == "V" and len(parts) == 3:
        return SpeedLine(_num(parts[1], "V vmax"), _num(parts[2], "V amax"))
    return InfoMsg(text)


def format_move(deg: Sequence[float]) -> str:
    """``m D1 .. D6`` in zero-relative degrees (two decimals; firmware parses %f)."""
    if len(deg) != COUNT or not all(math.isfinite(float(d)) for d in deg):
        raise ValueError(f"expected {COUNT} finite joint values, got {deg!r}")
    return "m " + " ".join(f"{float(d):.2f}" for d in deg)
