"""Firmware planner port vs the real sketch.

1. The constants in lbyr/arm/fw_planner.py must match firmware/xiao_c6_bridge/Code.ino.
2. If g++ is available, the sketch is compiled against tests/fw_sim/ stubs (simulated clock)
   and run; DONE timing must match the port's prediction, and the DONE / Q / S lines must
   follow the host contract.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from lbyr.arm import fw_planner as fp
from lbyr.arm.text_proto import DoneMsg, JointLine, OkMsg, SpeedLine, StatusMsg, parse_line
from lbyr.motion.trajectory import trapezoid_duration

ROOT = Path(__file__).parents[1]
INO = ROOT / "firmware" / "xiao_c6_bridge" / "Code.ino"


def _const(name: str) -> str:
    m = re.search(rf"const\s+\w+\s+{name}(?:\[COUNT\])?\s*=\s*([^;]+);", INO.read_text())
    assert m, name
    return m.group(1).strip()


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(x.rstrip("f")) for x in re.findall(r"-?\d+(?:\.\d+)?f?", text))


def test_constants_match_the_sketch() -> None:
    assert _floats(_const("MIN_US")) == fp.MIN_US
    assert _floats(_const("MAX_US")) == fp.MAX_US
    assert _floats(_const("DEFAULT_ZERO_US")) == tuple(float(z) for z in fp.DEFAULT_ZERO_US)
    assert _floats(_const("JOINT_SPEED")) == fp.JOINT_SPEED
    assert _floats(_const("VMAX_CAP_DEG")) == (fp.VMAX_CAP_DEG,)
    assert _floats(_const("AMAX_CAP_DEG")) == (fp.AMAX_CAP_DEG,)
    assert _floats(_const("BACKLASH_US")) == (fp.BACKLASH_US,)
    assert _floats(_const("SETTLE_US")) == (fp.SETTLE_US,)
    assert _floats(_const("TICK_US")) == (fp.TICK_S * 1e6,)
    text = INO.read_text()
    assert re.search(r"float vmaxDeg = 30\.0f;", text) and fp.DEFAULT_VMAX_DEG == 30.0
    assert re.search(r"float amaxDeg = 60\.0f;", text) and fp.DEFAULT_AMAX_DEG == 60.0
    assert "1000.0f / 90.0f" in text and fp.US_PER_DEG == pytest.approx(1000 / 90)


def test_single_joint_duration_close_to_trapezoid() -> None:
    zero = (1500.0,) * 6
    t = fp.predict_move_s(zero, (40, 0, 0, 0, 0, 0), zero)  # upward: no backlash leg
    assert t == pytest.approx(trapezoid_duration(40, 30, 60), abs=0.05)


def test_downward_move_adds_backlash_leg_and_slow_joints_are_slower() -> None:
    zero = (1500.0,) * 6
    up = fp.predict_move_s(zero, (20, 0, 0, 0, 0, 0), zero)
    down = fp.predict_move_s(zero, (-20, 0, 0, 0, 0, 0), zero)
    assert down > up + 0.1  # overshoot 20 us, then come back up
    gripper = fp.predict_move_s(zero, (0, 0, 0, 0, 0, 20), zero)
    assert gripper > 2 * up  # JOINT_SPEED 0.35


def test_synced_move_arrives_together_and_final_positions() -> None:
    p = fp.FirmwarePlanner((1500.0,) * 6)
    p.move_deg((30, -10, 5, 0, 0, 0))
    t = 0.0
    while p.any_moving():
        p.step()
        t += fp.TICK_S
    assert p.positions_deg() == pytest.approx((30, -10, 5, 0, 0, 0), abs=1e-6)
    assert t == pytest.approx(p.predict_settle_s() + t, abs=1e-9)  # settled: 0 left


def test_clamp_and_stop() -> None:
    p = fp.FirmwarePlanner((1500.0,) * 6)
    p.move_deg((200, 0, 0, 0, 0, 0))  # 1500 + 2222 us -> clamped to 2500
    for _ in range(50):
        p.step()
    p.stop_all()
    t_stop = p.predict_settle_s()
    assert 0 < t_stop < 1.0
    p2 = fp.FirmwarePlanner((1500.0,) * 6)
    p2.move_deg((200, 0, 0, 0, 0, 0))
    assert p2.predict_settle_s() == pytest.approx(
        fp.predict_move_s((1500.0,) * 6, (90, 0, 0, 0, 0, 0), (1500.0,) * 6), abs=0.02
    )


# ------------------------------------------------------------------ real sketch
@pytest.fixture(scope="module")
def fw_bin(tmp_path_factory: pytest.TempPathFactory) -> Path:
    gxx = shutil.which("g++")
    if gxx is None:
        pytest.skip("g++ not available: firmware-in-the-loop tests skipped")
    out = tmp_path_factory.mktemp("fw") / "fw"
    subprocess.run(
        [gxx, "-std=c++17", "-O1", f"-I{ROOT / 'tests' / 'fw_sim'}", f"-I{INO.parent}"]
        + ["-o", str(out), str(ROOT / "tests" / "fw_sim" / "harness.cpp")],
        check=True,
        capture_output=True,
    )
    return out


def run_fw(fw_bin: Path, script: list[tuple[int, str]]) -> list[tuple[int, str]]:
    text = "".join(f"{t} {c}\n" for t, c in script)
    out = subprocess.run([str(fw_bin)], input=text, capture_output=True, text=True, timeout=60)
    lines = []
    for ln in out.stdout.splitlines():
        t, _, rest = ln.partition("|")
        lines.append((int(t), rest))
    return lines


T0 = 5000  # boot homing (staggered attach) is over by then


@pytest.mark.parametrize(
    "goal",
    [
        (20, -10, 5, 0, 30, -20),
        (45, 0, 0, 0, 0, 0),
        (-30, 15, -15, 10, -10, 10),
        (0, 0, 0, 0, 0, 60),
    ],
)
def test_done_timing_matches_port(fw_bin: Path, goal: tuple[float, ...]) -> None:
    cmd = "m " + " ".join(str(g) for g in goal)
    lines = run_fw(fw_bin, [(T0, cmd), (T0 + 30000, "END")])
    after = [(t, s) for t, s in lines if t >= T0]
    assert [s for _, s in after][:1] == ["OK m"]
    dones = [t for t, s in after if s == "DONE"]
    assert len(dones) == 1, after
    measured = (dones[0] - T0) / 1000.0
    zero = fp.DEFAULT_ZERO_US
    predicted = fp.predict_move_s(zero, goal, zero)
    assert measured == pytest.approx(predicted, abs=0.03), (measured, predicted)


def test_done_once_on_preempt_and_q_s_lines(fw_bin: Path) -> None:
    lines = run_fw(
        fw_bin,
        [
            (T0, "list"),
            (T0 + 100, "m 20 0 0 0 0 0"),
            (T0 + 300, "q"),
            (T0 + 400, "m 25 0 0 0 0 -10"),  # pre-empts the first move
            (T0 + 8000, "q"),
            (T0 + 8100, "list"),
            (T0 + 9000, "END"),
        ],
    )
    boot = [s for t, s in lines if t < T0]
    assert boot.count("DONE") == 1  # boot homing
    msgs = [(t, parse_line(s)) for t, s in lines if t >= T0]
    joints = [m for _, m in msgs if isinstance(m, JointLine)]
    assert len(joints) == 12 and all(j.goal_us is not None for j in joints)
    assert joints[0].zero_us == 1477 and joints[0].moving is False and joints[0].attached
    assert any(isinstance(m, SpeedLine) and m.vmax_deg == 30 for _, m in msgs)
    statuses = [m for _, m in msgs if isinstance(m, StatusMsg)]
    assert statuses[0].moving and not statuses[1].moving
    assert statuses[1].cur_us[0] == pytest.approx(1477 + 25 * fp.US_PER_DEG, abs=1)
    assert statuses[1].cur_us[5] == pytest.approx(1433 - 10 * fp.US_PER_DEG, abs=1)
    assert sum(isinstance(m, DoneMsg) for _, m in msgs) == 1
    oks = [t for t, m in msgs if isinstance(m, OkMsg) and m.text == "OK m"]
    done_t = next(t for t, m in msgs if isinstance(m, DoneMsg))
    assert done_t > oks[-1]
