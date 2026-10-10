"""SerialArm (text protocol) against tests/fake_board.py: no hardware."""

import time

import pytest
from fake_board import FakeBoard

from lbyr.arm.calib import JointCalib, JointsCalib
from lbyr.arm.fw_planner import DEFAULT_ZERO_US
from lbyr.arm.interface import DONE, ERROR, IDLE, MOVING
from lbyr.arm.serial_arm import FirmwareTooOldError, SerialArm, ZeroMismatchError
from lbyr.arm.text_proto import DoneMsg, ErrMsg, OkMsg
from lbyr.config import JOINT_NAMES, ArmConfig, MotionConfig, mock_profile
from lbyr.errors import ConfigError, HardwareNotConfirmedError, LBYRError
from lbyr.motion.controller import FAULT, SEG_DONE, MotionController


def calib(zero=DEFAULT_ZERO_US, sign=(1,) * 6, zpose=(0.0,) * 6) -> JointsCalib:  # type: ignore[no-untyped-def]
    return JointsCalib(
        tuple(
            JointCalib(n, z, s, p)
            for n, z, s, p in zip(JOINT_NAMES, zero, sign, zpose, strict=True)
        ),
        verified=True,
    )


def make(board: FakeBoard, joints: JointsCalib | None = None) -> SerialArm:
    cfg = ArmConfig(serial_port="/dev/fake")
    return SerialArm(cfg, joints or calib(), port_factory=lambda port, baud: board)


def test_refusals_without_port_or_with_mock_config() -> None:
    with pytest.raises(ConfigError):
        SerialArm(mock_profile().arm)
    with pytest.raises(HardwareNotConfirmedError, match="OQ-10"):
        SerialArm(ArmConfig()).connect()


def test_connect_lists_compares_zeros_and_waits_for_idle() -> None:
    board = FakeBoard()
    arm = make(board)
    arm.connect()
    try:
        assert board.written[:2] == ["q", "list"]  # firmware probe first
        s = arm.read_state()
        assert s.feedback == "commanded" and s.load is None
        assert s.joints == pytest.approx((0.0,) * 6, abs=0.1)
        assert arm.motion_status() == (IDLE, None)
    finally:
        arm.close()


def test_zero_mismatch_refuses_to_move() -> None:
    board = FakeBoard(zero_us=(1477, 1989, 1850, 1223, 1467, 1433))
    arm = make(board)
    with pytest.raises(ZeroMismatchError, match=r"joint 3 \(elbow_pitch\): board 1850 us"):
        arm.connect()
    assert board.closed and not any(w.startswith("m ") for w in board.written)


def test_silent_board_fails_connect() -> None:
    arm = make(FakeBoard(answer=False))
    with pytest.raises(LBYRError, match="did not answer 'q'"):
        arm.connect()


@pytest.mark.parametrize(
    "board",
    [
        FakeBoard(q_reply="ERR unknown command; enter help"),
        FakeBoard(q_reply="Q,0,1500,1500"),  # malformed Q
        FakeBoard(old_s_lines=True),  # 6-field S lines
    ],
    ids=["q-err", "q-malformed", "s-6-fields"],
)
def test_old_firmware_is_refused(board: FakeBoard) -> None:
    arm = make(board)
    with pytest.raises(FirmwareTooOldError, match="reflash firmware/xiao_c6_bridge/Code.ino"):
        arm.connect()
    assert board.closed and not any(w.startswith("m ") for w in board.written)


def test_move_sends_one_m_in_firmware_degrees_and_gets_done() -> None:
    board = FakeBoard()
    arm = make(board, calib(sign=(1, -1, 1, 1, 1, 1), zpose=(0.0, 90.0, 0, 0, 0, 0)))
    arm.connect()
    try:
        t0 = time.perf_counter()
        predicted = arm.move_joints((10.0, 80.0, 0.0, 0.0, 0.0, -5.0))
        moves = [w for w in board.written if w.startswith("m ")]
        assert moves == ["m 10.00 10.00 0.00 0.00 0.00 -5.00"]  # host 80 -> fw -(80-90)
        assert arm.wait_done(predicted * 1.5 + 0.5) == DONE
        assert time.perf_counter() - t0 == pytest.approx(predicted, abs=0.15)
        assert board.fw_deg() == pytest.approx((10, 10, 0, 0, 0, -5), abs=0.1)
        arm.request_state()
        time.sleep(0.1)
        assert arm.read_state().joints == pytest.approx((10, 80, 0, 0, 0, -5), abs=0.1)
    finally:
        arm.close()


def test_done_before_ack_belongs_to_the_previous_motion() -> None:
    arm = make(FakeBoard())
    arm.connect()
    try:
        arm._handle(OkMsg("OK m"))  # an earlier move the host did not send this session
        arm.move_joints((1.0, 0, 0, 0, 0, 0))
        time.sleep(0.05)
        # The fake board already answered 'OK m'; simulate the race explicitly instead:
        with arm._lock:
            arm._ack_seq, arm._done_seq = arm._sent_seq - 1, arm._sent_seq - 1
        arm._handle(DoneMsg())  # old motion's DONE, before our OK m
        assert arm.motion_status() == (MOVING, None)
        arm._handle(OkMsg("OK m"))
        assert arm.motion_status() == (MOVING, None)
        arm._handle(DoneMsg())
        assert arm.motion_status() == (DONE, None)
    finally:
        arm.close()


def test_err_reply_is_an_error_for_the_current_segment() -> None:
    arm = make(FakeBoard())
    arm.connect()
    try:
        with arm._lock:
            arm._sent_seq += 1  # a segment the board rejected
        arm._handle(ErrMsg("ERR attach"))
        assert arm.motion_status() == (ERROR, "bridge: ERR attach")
    finally:
        arm.close()


def test_stop_off_and_torque_on_after_release() -> None:
    board = FakeBoard()
    arm = make(board)
    arm.connect()
    try:
        arm.stop()
        arm.torque(False)
        assert board.written[-2:] == ["x", "off"]
        with pytest.raises(LBYRError, match="released"):
            arm.torque(True)
    finally:
        arm.close()


def test_read_error_is_a_link_error() -> None:
    board = FakeBoard()
    arm = make(board)
    arm.connect()
    try:
        board.read_error = OSError("device disconnected")
        time.sleep(0.1)
        phase, detail = arm.motion_status()
        assert phase == ERROR and "LINK" in (detail or "")
    finally:
        arm.close()


def test_no_done_becomes_link_fault_via_controller() -> None:
    board = FakeBoard()
    arm = make(board)
    arm.connect()
    board.send_done = False  # DONE lost from now on
    cfg = MotionConfig(poll_hz=50.0)
    mc = MotionController(arm, cfg, (0.0,) * 6)
    try:
        seg = mc.command((5.0, 0, 0, 0, 0, 0))
        t0 = time.perf_counter()
        st = mc.current()
        while time.perf_counter() - t0 < 5.0:
            mc.tick()
            st = mc.current()
            if st.seg_id == seg and st.phase != "moving":
                break
            time.sleep(0.02)
        elapsed = time.perf_counter() - t0
        assert st.phase == FAULT and "LINK: no DONE" in (st.fault or "")
        assert elapsed == pytest.approx(cfg.done_timeout_s(st.predicted_s), abs=0.15)
        board.send_done = True
        seg2 = mc.command((0.0,) * 6)
        for _ in range(200):
            mc.tick()
            if mc.current().seg_id == seg2 and mc.current().phase == SEG_DONE:
                break
            time.sleep(0.02)
        assert mc.current().phase == SEG_DONE
    finally:
        arm.close()


def test_set_speed_sends_speed_and_resyncs_planner() -> None:
    from lbyr.arm.serial_arm import speed_for_cap

    board = FakeBoard(vmax=30.0, amax=60.0)
    arm = make(board)
    arm.connect()
    try:
        v, a = speed_for_cap(0.30)
        assert (v, a) == (18.0, 60.0)  # 30 % of the firmware caps 60 deg/s, 200 deg/s^2
        assert arm.set_speed(v, a) == (18.0, 60.0)
        assert "speed 18 60" in board.written
        assert board.written[board.written.index("speed 18 60") + 1] == "list"
        assert arm._planner is not None and arm._planner.vmax_deg == 18.0
        predicted = arm.move_joints((30.0, 0, 0, 0, 0, 0))
        assert predicted > 30.0 / 18.0  # slower than the old 30 deg/s
        assert arm.wait_done(predicted * 1.5 + 1.0) == DONE
    finally:
        arm.close()


def test_speed_cap_bounds_and_port_detection(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from lbyr.arm.serial_arm import detect_bridge_port, speed_for_cap

    assert speed_for_cap(0.5) == (30.0, 100.0)
    with pytest.raises(ValueError):
        speed_for_cap(0.8)
    assert detect_bridge_port(str(tmp_path / "ttyACM*")) == (None, [])
    (tmp_path / "ttyACM0").touch()
    one = str(tmp_path / "ttyACM0")
    assert detect_bridge_port(str(tmp_path / "ttyACM*")) == (one, [one])
    (tmp_path / "ttyACM1").touch()
    port, found = detect_bridge_port(str(tmp_path / "ttyACM*"))
    assert port is None and len(found) == 2
