"""Real arm over the XIAO ESP32-C6 bridge, TEXT protocol (firmware/xiao_c6_bridge/Code.ino).

PP-1 (2026-10-10): the firmware is the trajectory generator. One ``m`` per motion segment;
completion = the firmware's ``DONE`` line. No setpoint streaming.

- A reader thread parses every board line (:mod:`lbyr.arm.text_proto`) into state shared
  under a lock: DONE / OK / ERR bookkeeping, the latest ``Q`` status, ``list`` rows.
  Public methods only write (one serial write) and read that state: they never wait for
  a reply, except :meth:`connect` (startup) and :meth:`ArmInterface.wait_done` (tools).
- DONE pairing: the firmware prints ``OK m`` while handling ``m`` and the next ``DONE``
  after it. A DONE that arrives before the ``OK m`` of the latest segment belongs to an
  older motion and is ignored.
- :meth:`connect` sends ``list``, compares the board's zero pulses with
  ``data/calib/joints.yaml`` and refuses to move on any mismatch.
- :meth:`move_joints` converts host degrees to zero-relative degrees (sign and zero_pose_deg
  from joints.yaml) and returns the duration predicted by the firmware planner port; the
  motion controller turns "no DONE within predicted x 1.5 + 0.5 s" into a LINK fault.
- Feedback: none (OQ-5). ``read_state`` reports the COMMANDED position from ``q``.

``pyserial`` is imported lazily so this module can be imported on any machine.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from lbyr.arm.calib import JointsCalib, load_joints
from lbyr.arm.fw_planner import TICK_S, FirmwarePlanner
from lbyr.arm.interface import DONE, ERROR, IDLE, MOVING, STATUS_OK, ArmInterface, ArmState
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
from lbyr.config import NUM_JOINTS, ArmConfig, require
from lbyr.errors import ConfigError, LBYRError, ProtocolError

log = logging.getLogger(__name__)

READ_TIMEOUT_S = 0.05
WRITE_TIMEOUT_S = 0.05
LIST_TIMEOUT_S = 2.0
BOOT_SETTLE_TIMEOUT_S = 20.0  # the board homes itself at boot (staggered attach)


class ZeroMismatchError(LBYRError):
    """The board's saved zeros differ from data/calib/joints.yaml: refuse to move."""


class PortLike(Protocol):
    def write(self, data: bytes) -> int | None: ...

    def readline(self) -> bytes: ...

    def close(self) -> None: ...


PortFactory = Callable[[str, int], PortLike]


def _pyserial_port(port: str, baud: int) -> PortLike:
    import serial  # lazy: pyserial

    p: Any = serial.Serial(
        port=port, baudrate=baud, timeout=READ_TIMEOUT_S, write_timeout=WRITE_TIMEOUT_S
    )
    return p


class SerialArm(ArmInterface):
    def __init__(
        self,
        config: ArmConfig,
        joints: JointsCalib | None = None,
        port_factory: PortFactory | None = None,
    ) -> None:
        if config.simulated:
            raise ConfigError("SerialArm refuses a SIMULATION-ONLY config (mock_profile)")
        self._cfg = config
        self._joints = joints
        self._port_factory = port_factory or _pyserial_port
        self._port: PortLike | None = None
        self._wlock = threading.Lock()  # serial writes
        self._lock = threading.Lock()  # shared reader state
        self._reader: threading.Thread | None = None
        self._running = threading.Event()
        # reader state (under _lock)
        self._sent_seq = 0  # 'm' commands sent
        self._ack_seq = 0  # 'OK m' replies seen
        self._done_seq = 0  # value of _ack_seq when the latest DONE arrived
        self._err: tuple[int, str] | None = None  # (sent_seq at the time, ERR text)
        self._link_error: str | None = None
        self._status: StatusMsg | None = None
        self._status_t = 0.0
        self._rows: dict[int, JointLine] = {}
        self._speed: SpeedLine | None = None
        self._list_done = threading.Event()
        self._attached = [True] * NUM_JOINTS
        self._planner: FirmwarePlanner | None = None
        self._planner_t = 0.0

    # ------------------------------------------------------------ setup
    def check_confirmed(self) -> None:
        """Raise HardwareNotConfirmedError for the first missing value needed to connect."""
        require(self._cfg.servo_voltage_v, "servo PSU voltage (safety interlock)", "OQ-3")
        require(self._cfg.serial_port, "bridge serial port (XIAO /dev/ttyACM*)", "OQ-10")
        require(self._cfg.host_baud, "host<->bridge baud", "A-5")

    @property
    def joints(self) -> JointsCalib:
        if self._joints is None:
            self._joints = load_joints(self._cfg.joints_path)
        return self._joints

    def connect(self) -> None:
        self.check_confirmed()
        joints = self.joints
        assert self._cfg.serial_port is not None and self._cfg.host_baud is not None
        self._port = self._port_factory(self._cfg.serial_port, self._cfg.host_baud)
        self._running.set()
        self._reader = threading.Thread(target=self._read_loop, name="bridge-rx", daemon=True)
        self._reader.start()
        try:
            rows, speed = self._read_list()
            board_zero = [rows[j].zero_us for j in range(1, NUM_JOINTS + 1)]
            bad = joints.zero_mismatches(board_zero)
            if bad:
                raise ZeroMismatchError(
                    "Board zeros differ from data/calib/joints.yaml; refusing to move:\n  "
                    + "\n  ".join(bad)
                    + "\nCheck that the right board is connected and the arm rests in its zero "
                    "pose. Then either fix zero_us in joints.yaml (and set verified: true) or "
                    "restore the board zeros with 'set J US' in tools/arm_gui.py."
                )
            self._planner = FirmwarePlanner(board_zero, speed.vmax_deg, speed.amax_deg)
            self._planner.set_positions_us([rows[j].cur_us for j in range(1, NUM_JOINTS + 1)])
            self._planner_t = time.perf_counter()
            self._attached = [rows[j].attached for j in range(1, NUM_JOINTS + 1)]
            self._wait_boot_settled()
        except Exception:
            self.close()
            raise
        log.info(
            "SerialArm connected on %s @ %s (zeros match joints.yaml; speed %.0f deg/s, "
            "%.0f deg/s^2)",
            self._cfg.serial_port,
            self._cfg.host_baud,
            speed.vmax_deg,
            speed.amax_deg,
        )

    def _read_list(self) -> tuple[dict[int, JointLine], SpeedLine]:
        with self._lock:
            self._rows.clear()
            self._speed = None
            self._list_done.clear()
        self._write("list")
        if not self._list_done.wait(LIST_TIMEOUT_S):
            raise LBYRError(
                f"bridge did not answer 'list' within {LIST_TIMEOUT_S:.0f} s "
                f"({self._cfg.serial_port}); check the USB cable and that the XIAO runs "
                "firmware/xiao_c6_bridge"
            )
        with self._lock:
            rows, speed = dict(self._rows), self._speed
        missing = [j for j in range(1, NUM_JOINTS + 1) if j not in rows]
        if missing or speed is None:
            raise LBYRError(f"incomplete 'list' reply (missing joints {missing})")
        return rows, speed

    def _wait_boot_settled(self) -> None:
        """The board homes itself at boot; never send motion before that is finished."""
        t_end = time.perf_counter() + BOOT_SETTLE_TIMEOUT_S
        while time.perf_counter() < t_end:
            t_req = time.perf_counter()
            self._write("q")
            time.sleep(0.1)
            with self._lock:
                st, st_t = self._status, self._status_t
            if st is not None and st_t >= t_req and not st.moving:
                return
        raise LBYRError("bridge still moving (boot homing?) after 20 s; check the arm")

    def close(self) -> None:
        self._running.clear()
        if self._reader is not None and self._reader is not threading.current_thread():
            self._reader.join(1.0)
        self._reader = None
        if self._port is not None:
            try:
                self._port.close()
            finally:
                self._port = None

    # ------------------------------------------------------------ reader thread
    def _read_loop(self) -> None:
        while self._running.is_set():
            port = self._port
            if port is None:
                return
            try:
                raw = port.readline()
            except Exception as e:  # serial.SerialException, OSError (cable pulled)
                with self._lock:
                    self._link_error = f"LINK: serial read failed: {e}"
                log.error("bridge read failed: %s", e)
                return
            if not raw:
                continue
            try:
                msg = parse_line(raw)
            except ProtocolError as e:
                log.warning("bridge: unparseable line %r (%s)", raw, e)
                continue
            self._handle(msg)

    def _handle(self, msg: Any) -> None:
        with self._lock:
            if isinstance(msg, DoneMsg):
                self._done_seq = self._ack_seq
            elif isinstance(msg, OkMsg):
                if msg.text == "OK m":
                    self._ack_seq += 1
                else:
                    log.debug("bridge: %s", msg.text)
            elif isinstance(msg, ErrMsg):
                self._err = (self._sent_seq, msg.text)
                log.warning("bridge: %s", msg.text)
            elif isinstance(msg, StatusMsg):
                self._status, self._status_t = msg, time.perf_counter()
            elif isinstance(msg, JointLine):
                self._rows[msg.joint] = msg
            elif isinstance(msg, SpeedLine):
                self._speed = msg
                self._list_done.set()
            elif isinstance(msg, InfoMsg) and msg.text:
                log.info("bridge: %s", msg.text)

    # ------------------------------------------------------------ writes
    def _write(self, text: str) -> None:
        port = self._port
        if port is None:
            raise LBYRError("SerialArm not connected")
        try:
            with self._wlock:
                port.write((text + "\n").encode("ascii"))
        except Exception as e:  # serial.SerialTimeoutException, OSError
            with self._lock:
                self._link_error = f"LINK: serial write failed: {e}"
            raise LBYRError(f"LINK: serial write failed: {e}") from e

    def move_joints(self, joints_deg: Sequence[float]) -> float:
        fw = self.joints.host_to_fw_deg(joints_deg)
        line = format_move(fw)
        predicted = self._predict(fw)
        with self._lock:
            self._sent_seq += 1
        self._write(line)
        self._attached = [True] * NUM_JOINTS  # `m` attaches every joint
        return predicted

    def _predict(self, fw_deg: Sequence[float]) -> float:
        """Advance the planner port to now (or resync from a fresh, idle Q), then predict."""
        with self._lock:
            planner, st, st_t = self._planner, self._status, self._status_t
        if planner is None:
            raise LBYRError("SerialArm not connected")
        now = time.perf_counter()
        if st is not None and not st.moving and st_t > self._planner_t:
            planner.set_positions_us(st.cur_us)
        else:
            for _ in range(min(int((now - self._planner_t) / TICK_S), 100_000)):
                if not planner.any_moving():
                    break
                planner.step(TICK_S)
        self._planner_t = now
        planner.move_deg(fw_deg)
        return planner.predict_settle_s()

    def motion_status(self) -> tuple[str, str | None]:
        with self._lock:
            if self._link_error is not None:
                return ERROR, self._link_error
            if self._sent_seq == 0:
                return IDLE, None
            if self._ack_seq >= self._sent_seq and self._done_seq >= self._sent_seq:
                return DONE, None
            if self._err is not None and self._err[0] == self._sent_seq:
                return ERROR, f"bridge: {self._err[1]}"
            return MOVING, None

    def request_state(self) -> None:
        self._write("q")

    def read_state(self) -> ArmState:
        with self._lock:
            st, st_t = self._status, self._status_t
            rows = dict(self._rows)
        if st is not None:
            cur, moving, t = st.cur_us, st.moving, st_t
        elif len(rows) == NUM_JOINTS:
            cur = tuple(rows[j].cur_us for j in range(1, NUM_JOINTS + 1))
            moving, t = False, time.perf_counter()
        else:
            raise LBYRError("no status from the bridge yet")
        joints = self.joints.us_to_host_deg(cur)
        return ArmState(
            joints=joints,
            load=None,
            temp=None,
            status=STATUS_OK,
            t=t,
            feedback="commanded",
            commanded=joints,
            moving=moving,
        )

    def torque(self, on: bool) -> None:
        if not on:
            self._write("off")
            self._attached = [False] * NUM_JOINTS
            return
        if not all(self._attached):
            # Re-attaching makes the joint jump to its zero pulse (no feedback, no smoothing).
            raise LBYRError(
                "some joints are released; support the arm and press Home in tools/arm_gui.py "
                "(re-attaching jumps to the zero pulse)"
            )

    def stop(self) -> None:
        self._write("x")
