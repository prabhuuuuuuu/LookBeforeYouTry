"""Host <-> ESP32 bridge protocol (bible Section 6): newline-delimited JSON over USB serial.

Commands (host -> bridge)::

    {"type":"move","joints":[...6 deg...],"time_ms":800}
    {"type":"traj","pts":[[...6 deg...],...],"dt_ms":20}
    {"type":"state"}
    {"type":"torque","on":false}
    {"type":"stop"}

Response to ``state`` (bridge -> host)::

    {"joints":[...],"load":[...],"temp":[...],"status":"ok"}

``load``/``temp`` may be ``null`` until OQ-5 (feedback readability) is confirmed.
Units: degrees and milliseconds. Validation is strict: unknown keys are rejected.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from lbyr.config import NUM_JOINTS
from lbyr.errors import ProtocolError

MAX_LINE_BYTES = 8192
MAX_TRAJ_POINTS = 100
STATUS_OK = "ok"


@dataclass(frozen=True)
class MoveCmd:
    joints: tuple[float, ...]
    time_ms: int


@dataclass(frozen=True)
class TrajCmd:
    pts: tuple[tuple[float, ...], ...]
    dt_ms: int


@dataclass(frozen=True)
class StateRequest:
    pass


@dataclass(frozen=True)
class TorqueCmd:
    on: bool


@dataclass(frozen=True)
class StopCmd:
    pass


@dataclass(frozen=True)
class StateResponse:
    joints: tuple[float, ...]
    load: tuple[float, ...] | None
    temp: tuple[float, ...] | None
    status: str


Command = MoveCmd | TrajCmd | StateRequest | TorqueCmd | StopCmd
Message = Command | StateResponse


# ---------------------------------------------------------------- validation helpers
def _number(x: Any, what: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ProtocolError(f"{what}: expected number, got {x!r}")
    if not math.isfinite(x):
        raise ProtocolError(f"{what}: non-finite value")
    return float(x)


def _joint_vector(x: Any, what: str) -> tuple[float, ...]:
    if not isinstance(x, list) or len(x) != NUM_JOINTS:
        raise ProtocolError(f"{what}: expected list of {NUM_JOINTS} numbers")
    return tuple(_number(v, f"{what}[{i}]") for i, v in enumerate(x))


def _optional_vector(x: Any, what: str) -> tuple[float, ...] | None:
    return None if x is None else _joint_vector(x, what)


def _positive_int(x: Any, what: str) -> int:
    if isinstance(x, bool) or not isinstance(x, int) or x <= 0:
        raise ProtocolError(f"{what}: expected positive integer, got {x!r}")
    return x


def _check_keys(obj: dict[str, Any], allowed: set[str], what: str) -> None:
    missing = allowed - obj.keys()
    extra = obj.keys() - allowed
    if missing or extra:
        raise ProtocolError(f"{what}: missing={sorted(missing)} extra={sorted(extra)}")


def _load_obj(line: str | bytes) -> dict[str, Any]:
    if isinstance(line, bytes):
        if len(line) > MAX_LINE_BYTES:
            raise ProtocolError("line too long")
        try:
            line = line.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ProtocolError("invalid utf-8") from e
    if len(line) > MAX_LINE_BYTES:
        raise ProtocolError("line too long")
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as e:
        raise ProtocolError(f"invalid JSON: {e.msg}") from e
    if not isinstance(obj, dict):
        raise ProtocolError("message must be a JSON object")
    return obj


# ---------------------------------------------------------------- encode
def _vec(v: Sequence[float]) -> list[float]:
    return [round(float(x), 3) for x in v]


def to_dict(msg: Message) -> dict[str, Any]:
    if isinstance(msg, MoveCmd):
        return {"type": "move", "joints": _vec(msg.joints), "time_ms": msg.time_ms}
    if isinstance(msg, TrajCmd):
        return {"type": "traj", "pts": [_vec(p) for p in msg.pts], "dt_ms": msg.dt_ms}
    if isinstance(msg, StateRequest):
        return {"type": "state"}
    if isinstance(msg, TorqueCmd):
        return {"type": "torque", "on": msg.on}
    if isinstance(msg, StopCmd):
        return {"type": "stop"}
    if isinstance(msg, StateResponse):
        return {
            "joints": _vec(msg.joints),
            "load": None if msg.load is None else _vec(msg.load),
            "temp": None if msg.temp is None else _vec(msg.temp),
            "status": msg.status,
        }
    raise ProtocolError(f"unknown message type {type(msg).__name__}")


def encode(msg: Message) -> bytes:
    """Validate and encode a message as one compact JSON line (with trailing newline)."""
    d = to_dict(msg)
    # Round-trip through the decoder so we never send something the bridge would reject.
    if isinstance(msg, StateResponse):
        _state_response_from_dict(d)
    else:
        _command_from_dict(d)
    return (json.dumps(d, separators=(",", ":")) + "\n").encode("utf-8")


# ---------------------------------------------------------------- decode
def _command_from_dict(obj: dict[str, Any]) -> Command:
    kind = obj.get("type")
    if kind == "move":
        _check_keys(obj, {"type", "joints", "time_ms"}, "move")
        return MoveCmd(
            _joint_vector(obj["joints"], "joints"), _positive_int(obj["time_ms"], "time_ms")
        )
    if kind == "traj":
        _check_keys(obj, {"type", "pts", "dt_ms"}, "traj")
        pts = obj["pts"]
        if not isinstance(pts, list) or not 0 < len(pts) <= MAX_TRAJ_POINTS:
            raise ProtocolError(f"traj.pts: expected 1..{MAX_TRAJ_POINTS} points")
        return TrajCmd(
            tuple(_joint_vector(p, f"pts[{i}]") for i, p in enumerate(pts)),
            _positive_int(obj["dt_ms"], "dt_ms"),
        )
    if kind == "state":
        _check_keys(obj, {"type"}, "state")
        return StateRequest()
    if kind == "torque":
        _check_keys(obj, {"type", "on"}, "torque")
        if not isinstance(obj["on"], bool):
            raise ProtocolError("torque.on: expected boolean")
        return TorqueCmd(obj["on"])
    if kind == "stop":
        _check_keys(obj, {"type"}, "stop")
        return StopCmd()
    raise ProtocolError(f"unknown command type {kind!r}")


def _state_response_from_dict(obj: dict[str, Any]) -> StateResponse:
    _check_keys(obj, {"joints", "load", "temp", "status"}, "state response")
    status = obj["status"]
    if not isinstance(status, str) or not status:
        raise ProtocolError("status: expected non-empty string")
    return StateResponse(
        joints=_joint_vector(obj["joints"], "joints"),
        load=_optional_vector(obj["load"], "load"),
        temp=_optional_vector(obj["temp"], "temp"),
        status=status,
    )


def decode_command(line: str | bytes) -> Command:
    return _command_from_dict(_load_obj(line))


def decode_state_response(line: str | bytes) -> StateResponse:
    return _state_response_from_dict(_load_obj(line))
