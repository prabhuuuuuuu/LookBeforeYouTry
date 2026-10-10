from pathlib import Path

import pytest

from lbyr.arm.calib import joints_from_dict, load_joints, mock_joints
from lbyr.arm.text_proto import (
    DoneMsg,
    ErrMsg,
    InfoMsg,
    JointLine,
    OkMsg,
    SpeedLine,
    StatusMsg,
    format_move,
    parse_line,
)
from lbyr.errors import CalibrationError, ProtocolError

ROOT = Path(__file__).parents[1]


def test_done_ok_err_info() -> None:
    assert parse_line("DONE\r\n") == DoneMsg()
    assert parse_line(b"OK m\n") == OkMsg("OK m")
    assert parse_line("ERR attach 3") == ErrMsg("ERR attach 3")
    assert isinstance(parse_line("OK test done (holding zero)"), OkMsg)
    assert parse_line("home | x stop | speed V A") == InfoMsg("home | x stop | speed V A")
    assert isinstance(parse_line("Boot: joints go to saved zeros."), InfoMsg)


def test_q_line() -> None:
    m = parse_line("Q,1,1479,1988,1845,1223,1468,1432")
    assert m == StatusMsg(True, (1479.0, 1988.0, 1845.0, 1223.0, 1468.0, 1432.0))
    assert parse_line("Q,0,1,2,3,4,5,6") == StatusMsg(False, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))


@pytest.mark.parametrize(
    "bad", ["Q,1,1,2,3", "Q,2,1,2,3,4,5,6", "Q,1,a,2,3,4,5,6", "Q,1,1,2,3,4,5,nan"]
)
def test_bad_q_lines(bad: str) -> None:
    with pytest.raises(ProtocolError):
        parse_line(bad)


def test_s_lines_new_and_old_format() -> None:
    new = parse_line("S,5,1467,1800,1,1820,1,Joint 5 MG996R")
    assert new == JointLine(5, 1467, 1800.0, True, 1820.0, True, "Joint 5 MG996R")
    old = parse_line("S,2,1989,1989,0,Joint 2 Feetech")
    assert old == JointLine(2, 1989, 1989.0, False, None, None, "Joint 2 Feetech")
    assert parse_line("V,30,60") == SpeedLine(30.0, 60.0)
    with pytest.raises(ProtocolError):
        parse_line("S,9,1500,1500,1,1500,0,x")
    with pytest.raises(ProtocolError):
        parse_line("S,1,1500,1500,2,1500,0,x")


def test_format_move() -> None:
    assert format_move((1, -2.5, 0, 0, 0, 33.333)) == "m 1.00 -2.50 0.00 0.00 0.00 33.33"
    with pytest.raises(ValueError):
        format_move((1, 2))
    with pytest.raises(ValueError):
        format_move((float("nan"),) * 6)


# ------------------------------------------------------------------ joints.yaml
def test_seeded_joints_yaml_matches_firmware_defaults_and_is_unverified() -> None:
    from lbyr.arm.fw_planner import DEFAULT_ZERO_US

    j = load_joints(ROOT / "data" / "calib" / "joints.yaml")
    assert j.zero_us == DEFAULT_ZERO_US
    assert j.verified is False  # must be checked against the board's `list`


def test_frame_conversion_with_sign_and_offset() -> None:
    rows = [
        {"name": n, "zero_us": 1500, "sign": s, "zero_pose_deg": z}
        for n, s, z in [
            ("base_yaw", 1, 0.0),
            ("shoulder_pitch", -1, 90.0),
            ("elbow_pitch", 1, 0.0),
            ("wrist_pitch", 1, 0.0),
            ("wrist_roll", 1, 0.0),
            ("gripper", 1, 0.0),
        ]
    ]
    j = joints_from_dict({"verified": True, "joints": rows})
    fw = j.host_to_fw_deg((10, 80, 0, 0, 0, 0))
    assert fw == (10, 10, 0, 0, 0, 0)
    assert j.fw_to_host_deg(fw) == (10, 80, 0, 0, 0, 0)
    assert j.us_to_host_deg((1500 + 1000 / 9, 1500, 1500, 1500, 1500, 1500))[0] == pytest.approx(
        10.0
    )


def test_zero_mismatch_lines() -> None:
    j = mock_joints()
    assert j.zero_mismatches((1500,) * 6) == []
    bad = j.zero_mismatches((1500, 1500, 1510, 1500, 1500, 1500))
    assert bad == ["joint 3 (elbow_pitch): board 1510 us, joints.yaml 1500 us"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["joints"].pop(),
        lambda d: d["joints"][0].update(name="shoulder"),
        lambda d: d["joints"][1].update(sign=0),
        lambda d: d["joints"][2].update(zero_us=3000),
        lambda d: d["joints"][3].pop("zero_us"),
    ],
)
def test_bad_joints_yaml(mutate) -> None:  # type: ignore[no-untyped-def]
    import yaml

    d = yaml.safe_load((ROOT / "data" / "calib" / "joints.yaml").read_text())
    mutate(d)
    with pytest.raises(CalibrationError):
        joints_from_dict(d)
