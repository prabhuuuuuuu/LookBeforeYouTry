import json

import pytest

from lbyr.arm import bridge_proto as bp
from lbyr.errors import ProtocolError

J = (0.0, 45.0, 60.0, 30.0, 0.0, 50.0)

COMMANDS = [
    bp.MoveCmd(J, 800),
    bp.TrajCmd((J, J), 20),
    bp.StateRequest(),
    bp.TorqueCmd(False),
    bp.TorqueCmd(True),
    bp.StopCmd(),
]


@pytest.mark.parametrize("msg", COMMANDS)
def test_command_round_trip(msg: bp.Command) -> None:
    line = bp.encode(msg)
    assert line.endswith(b"\n")
    assert bp.decode_command(line) == msg


def test_bible_examples_decode() -> None:
    assert bp.decode_command('{"type":"move","joints":[0,45,60,30,0,50],"time_ms":800}') == (
        bp.MoveCmd(J, 800)
    )
    assert bp.decode_command('{"type":"torque","on":false}') == bp.TorqueCmd(False)
    assert bp.decode_command('{"type":"stop"}') == bp.StopCmd()


def test_state_response_round_trip() -> None:
    resp = bp.StateResponse(J, (0.1,) * 6, (30.0,) * 6, "ok")
    assert bp.decode_state_response(bp.encode(resp)) == resp
    no_feedback = bp.StateResponse(J, None, None, "ok")
    assert bp.decode_state_response(bp.encode(no_feedback)) == no_feedback


@pytest.mark.parametrize(
    "line",
    [
        "not json",
        "[1,2,3]",
        '{"type":"fly"}',
        '{"joints":[0,0,0,0,0,0],"time_ms":10}',  # missing type
        '{"type":"move","joints":[0,0,0],"time_ms":800}',  # wrong length
        '{"type":"move","joints":[0,0,0,0,0,"a"],"time_ms":800}',
        '{"type":"move","joints":[0,0,0,0,0,true],"time_ms":800}',  # bool is not a number
        '{"type":"move","joints":[0,0,0,0,0,NaN],"time_ms":800}',
        '{"type":"move","joints":[0,0,0,0,0,0],"time_ms":0}',
        '{"type":"move","joints":[0,0,0,0,0,0],"time_ms":1.5}',
        '{"type":"move","joints":[0,0,0,0,0,0]}',  # missing time_ms
        '{"type":"move","joints":[0,0,0,0,0,0],"time_ms":5,"x":1}',  # extra key
        '{"type":"traj","pts":[],"dt_ms":20}',
        '{"type":"torque","on":0}',
        '{"type":"stop","now":true}',
    ],
)
def test_malformed_commands_rejected(line: str) -> None:
    with pytest.raises(ProtocolError):
        bp.decode_command(line)


@pytest.mark.parametrize(
    "obj",
    [
        {"joints": list(J), "load": None, "temp": None, "status": ""},
        {"joints": list(J), "load": None, "temp": None},
        {"joints": list(J)[:5], "load": None, "temp": None, "status": "ok"},
        {"joints": list(J), "load": [1, 2], "temp": None, "status": "ok"},
    ],
)
def test_malformed_state_response_rejected(obj: dict) -> None:
    with pytest.raises(ProtocolError):
        bp.decode_state_response(json.dumps(obj))


def test_encode_refuses_invalid() -> None:
    with pytest.raises(ProtocolError):
        bp.encode(bp.MoveCmd((0.0,) * 3, 100))
    with pytest.raises(ProtocolError):
        bp.encode(bp.MoveCmd(J, -1))


def test_oversized_line_rejected() -> None:
    with pytest.raises(ProtocolError):
        bp.decode_command(b"{" + b" " * (bp.MAX_LINE_BYTES + 1) + b"}")
