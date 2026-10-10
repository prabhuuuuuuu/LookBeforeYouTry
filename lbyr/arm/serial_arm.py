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
- The port is opened with plain pyserial defaults; DTR/RTS are never set or toggled
  (setting them reset the ESP32-C6 on open: servo jitter).
- :meth:`connect` watches the line for 0.3 s, flushes the input, sends a bare newline, then
  probes with ``q`` (up to 5 tries over 3 s) and needs a well-formed ``Q``. ERR, a malformed
  Q or S lines with fewer than 8 fields mean old firmware: refuse
  (:class:`FirmwareTooOldError`, "reflash firmware/xiao_c6_bridge/Code.ino"). ROM boot
  lines (the board was reset anyway) -> WARNING, wait for the boot-homing DONE, then probe.
- :meth:`connect` then sends ``list``, compares the board's zero pulses with
  ``data/calib/joints.yaml`` and refuses to move on any mismatch.
- :meth:`move_joints` converts host degrees to zero-relative degrees (sign and zero_pose_deg
  from joints.yaml) and returns the duration predicted by the firmware planner port; the
  motion controller turns "no DONE within predicted x 1.5 + 0.5 s" into a LINK fault.
- Robustness: write timeout 1 s; a failed write raises but is not sticky (the controller
  retries the ``q`` poll); the reader retries reads and declares the link lost only after
  ``READ_FAILS_LINK`` (3) consecutive failures.
- Feedback: none (OQ-5). ``read_state`` reports the COMMANDED position from ``q``.

``pyserial`` is imported lazily so this module can be imported on any machine.
"""

from __future__ import annotations

import glob
import logging
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from lbyr.arm.calib import JointsCalib, load_joints
from lbyr.arm.fw_planner import AMAX_CAP_DEG, TICK_S, VMAX_CAP_DEG, FirmwarePlanner
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
# 0.05 s timed out under load (Qwen loading, 2026-10-11): "SerialTimeoutException: Write
# timeout" on a 2-byte 'q'. 1 s covers USB stalls; a failed write is retried by the caller.
WRITE_TIMEOUT_S = 1.0
READ_FAILS_LINK = 3  # consecutive serial read failures before the link is declared lost
LIST_TIMEOUT_S = 2.0
OPEN_SETTLE_S = 0.3  # connect: watch the line this long after open, then flush
PROBE_TRIES = 5  # connect: 'q' attempts ...
PROBE_WINDOW_S = 3.0  # ... spread over this window
BOOT_SETTLE_TIMEOUT_S = 20.0  # the board homes itself at boot (staggered attach)
ROM_BOOT_MARKERS = (b"entry 0x", b"mode:DIO", b"ESP-ROM")  # ESP32-C6 ROM bootloader lines
S_LINE_FIELDS = 8  # S,j,zero_us,cur_us,attached,goal_us,moving,name (current Code.ino)
FW_TOO_OLD = "bridge firmware too old: reflash firmware/xiao_c6_bridge/Code.ino"


class ZeroMismatchError(LBYRError):
    """The board's saved zeros differ from data/calib/joints.yaml: refuse to move."""


class FirmwareTooOldError(LBYRError):
    """The bridge does not speak the current text protocol (q -> Q, 8-field S lines)."""


def detect_bridge_port(pattern: str = "/dev/ttyACM*") -> tuple[str | None, list[str]]:
    """Auto-detect the XIAO bridge: (port, candidates). The port is set only when exactly
    one device matches; with zero or several the caller must ask (``--port``, OQ-10)."""
    found = sorted(glob.glob(pattern))
    return (found[0] if len(found) == 1 else None), found


def speed_for_cap(cap: float) -> tuple[float, float]:
    """Firmware ``speed V A`` for a speed cap fraction (bible 8: dev 0.30, stage 0.50) of the
    firmware maxima VMAX_CAP_DEG / AMAX_CAP_DEG (Code.ino; not measured servo maxima, A-1)."""
    if not 0.0 < cap <= 0.5:
        raise ValueError("speed cap must be in (0, 0.5] (bible 8: dev 30%, stage 50%)")
    return round(cap * VMAX_CAP_DEG, 1), round(cap * AMAX_CAP_DEG, 1)


class PortLike(Protocol):
    def write(self, data: bytes) -> int | None: ...

    def readline(self) -> bytes: ...

    def close(self) -> None: ...


PortFactory = Callable[[str, int], PortLike]


def _pyserial_port(port: str, baud: int) -> PortLike:
    """Plain pyserial defaults. NEVER set or toggle DTR/RTS (before or after open): setting
    them low before open() reset the ESP32-C6 on this board (ROM boot lines, servo jitter);
    the defaults do not (verified on the rig, 2026-10-11)."""
    import serial  # lazy: pyserial

    p: Any = serial.Serial(port, baud, timeout=READ_TIMEOUT_S, write_timeout=WRITE_TIMEOUT_S)
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
        # connect probe: first reply to 'q' (None = well-formed Q, else the bad line)
        self._probing = False
        self._probe_bad: str | None = None
        self._probe_done = threading.Event()
        self._reset_seen = threading.Event()  # ROM boot lines seen: the board was reset
        self._boot_done = threading.Event()  # DONE seen since the latest reset

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
        try:
            self._settle_after_open()
            self._running.set()
            self._reader = threading.Thread(target=self._read_loop, name="bridge-rx", daemon=True)
            self._reader.start()
            self._probe_firmware()
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

    def set_speed(self, vmax_deg_s: float, amax_deg_s2: float) -> tuple[float, float]:
        """Send ``speed V A`` (the firmware caps it), then re-read ``list`` to get the values
        in force (``V`` line) and re-sync the planner port so DONE timeouts stay right."""
        if vmax_deg_s <= 0 or amax_deg_s2 <= 0:
            raise ValueError("speed and acceleration must be > 0")
        self._write(f"speed {vmax_deg_s:g} {amax_deg_s2:g}")
        time.sleep(0.05)  # let the OK line arrive before the list rows
        rows, speed = self._read_list()
        if abs(speed.vmax_deg - vmax_deg_s) > 0.6 or abs(speed.amax_deg - amax_deg_s2) > 0.6:
            log.warning(
                "bridge speed is %.0f deg/s, %.0f deg/s^2 (asked %g, %g)",
                speed.vmax_deg,
                speed.amax_deg,
                vmax_deg_s,
                amax_deg_s2,
            )
        board_zero = [rows[j].zero_us for j in range(1, NUM_JOINTS + 1)]
        planner = FirmwarePlanner(board_zero, speed.vmax_deg, speed.amax_deg)
        planner.set_positions_us([rows[j].cur_us for j in range(1, NUM_JOINTS + 1)])
        with self._lock:
            self._planner = planner
        self._planner_t = time.perf_counter()
        log.info("bridge speed set: %.0f deg/s, %.0f deg/s^2", speed.vmax_deg, speed.amax_deg)
        return speed.vmax_deg, speed.amax_deg

    def _settle_after_open(self) -> None:
        """Before the reader starts: watch the line for OPEN_SETTLE_S (ROM boot lines = the
        board was reset by the open), flush the input, send a bare newline (the firmware
        ignores empty lines; it terminates any half-typed command left in its buffer)."""
        port = self._port
        assert port is not None
        t_end = time.perf_counter() + OPEN_SETTLE_S
        while time.perf_counter() < t_end:
            raw = port.readline()  # READ_TIMEOUT_S
            if raw:
                self._check_rom_line(raw)
        flush = getattr(port, "reset_input_buffer", None)
        if callable(flush):
            flush()
        self._write("")

    def _check_rom_line(self, raw: bytes) -> bool:
        if not any(m in raw for m in ROM_BOOT_MARKERS):
            return False
        with self._lock:
            first = not self._reset_seen.is_set()
            self._reset_seen.set()
            self._boot_done.clear()  # the next DONE ends the boot homing
        if first:
            log.warning(
                "bridge was reset on open (ROM boot line %r); waiting for the firmware's "
                "boot-homing DONE before probing",
                raw.decode("ascii", errors="replace").strip(),
            )
        return True

    def _probe_firmware(self) -> None:
        """Send ``q`` up to PROBE_TRIES times over PROBE_WINDOW_S; require a well-formed
        ``Q,...``. ERR or a malformed Q means an old bridge firmware: refuse before anything
        else is sent. If the board was reset (ROM lines), wait for its boot DONE first."""
        boot_waits = 0
        tries = 0
        while tries < PROBE_TRIES:
            if self._reset_seen.is_set():
                if boot_waits:
                    raise LBYRError(
                        "bridge reset again while connecting (brown-out? USB?); "
                        "check the servo supply and the USB cable"
                    )
                boot_waits += 1
                if not self._boot_done.wait(BOOT_SETTLE_TIMEOUT_S):
                    raise LBYRError(
                        f"bridge was reset on open and printed no boot DONE within "
                        f"{BOOT_SETTLE_TIMEOUT_S:.0f} s; check the arm"
                    )
                self._reset_seen.clear()
                tries = 0
            tries += 1
            with self._lock:
                self._probing, self._probe_bad = True, None
                self._probe_done.clear()
            try:
                self._write("q")
                answered = self._probe_done.wait(PROBE_WINDOW_S / PROBE_TRIES)
            finally:
                with self._lock:
                    self._probing = False
                    bad = self._probe_bad
            if answered and bad is not None:
                raise FirmwareTooOldError(f"{FW_TOO_OLD} ('q' answered {bad!r})")
            if answered and not self._reset_seen.is_set():
                if tries > 1:
                    log.info("bridge answered 'q' on try %d", tries)
                return
            log.debug("bridge: no Q reply to 'q' (try %d/%d)", tries, PROBE_TRIES)
        raise LBYRError(
            f"bridge did not answer 'q' ({PROBE_TRIES} tries over {PROBE_WINDOW_S:.0f} s, "
            f"{self._cfg.serial_port}); check the USB cable and that the XIAO runs "
            f"firmware/xiao_c6_bridge (if it does: {FW_TOO_OLD})"
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
        old = [j for j, r in sorted(rows.items()) if r.goal_us is None or r.moving is None]
        if old:
            raise FirmwareTooOldError(
                f"{FW_TOO_OLD} (S lines of joints {old} have fewer than {S_LINE_FIELDS} fields)"
            )
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
        fails = 0
        while self._running.is_set():
            port = self._port
            if port is None:
                return
            try:
                raw = port.readline()
            except Exception as e:  # serial.SerialException, OSError (cable pulled)
                fails += 1
                if fails >= READ_FAILS_LINK:
                    with self._lock:
                        self._link_error = f"LINK: serial read failed {fails}x: {e}"
                    log.error("bridge read failed %d times in a row: %s", fails, e)
                    return
                log.warning("bridge read failed (%d/%d), retrying: %s", fails, READ_FAILS_LINK, e)
                time.sleep(0.02)
                continue
            fails = 0
            if not raw:
                continue
            if self._check_rom_line(raw):
                continue
            try:
                msg = parse_line(raw)
            except ProtocolError as e:
                log.warning("bridge: unparseable line %r (%s)", raw, e)
                if raw.lstrip().startswith(b"Q"):
                    self._probe_reply(raw.decode("ascii", errors="replace").strip())
                continue
            self._handle(msg)

    def _probe_reply(self, bad: str | None) -> None:
        """First reply to the connect 'q': None = well-formed Q, else the offending line."""
        with self._lock:
            if self._probing and not self._probe_done.is_set():
                self._probe_bad = bad
                self._probe_done.set()

    def _handle(self, msg: Any) -> None:
        if isinstance(msg, StatusMsg):
            self._probe_reply(None)
        elif isinstance(msg, ErrMsg) and not msg.text.startswith("ERR attach"):
            self._probe_reply(msg.text)  # 'ERR attach N' is boot homing, not a reply to q
        with self._lock:
            if isinstance(msg, DoneMsg):
                self._done_seq = self._ack_seq
                self._boot_done.set()
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
        """One write (<= WRITE_TIMEOUT_S). A failure raises but is NOT sticky: the caller
        decides (the motion controller retries ``q`` and faults after 3 in a row)."""
        port = self._port
        if port is None:
            raise LBYRError("SerialArm not connected")
        try:
            with self._wlock:
                port.write((text + "\n").encode("ascii"))
        except Exception as e:  # serial.SerialTimeoutException, OSError
            cmd = (text.split() or [""])[0]
            raise LBYRError(f"LINK: serial write {cmd!r} failed: {e}") from e

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
