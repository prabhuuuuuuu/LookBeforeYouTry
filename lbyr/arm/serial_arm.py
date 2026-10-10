"""Real arm over the ESP32 USB-serial JSON bridge (bible Section 6).

SKELETON: refuses to connect until every required hardware value is confirmed.
Required: OQ-3 (servo PSU voltage, safety interlock), OQ-4 (joint limits),
OQ-10 (serial port), host baud (set with the bridge firmware). ``read_state`` also
requires OQ-5 (feedback readable).

``pyserial`` is imported lazily so this module can be imported on any machine.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from typing import Any

from lbyr.arm import bridge_proto as bp
from lbyr.arm.interface import ArmInterface, ArmState
from lbyr.config import ArmConfig, require
from lbyr.errors import ConfigError, HardwareNotConfirmedError, LBYRError

log = logging.getLogger(__name__)

STREAM_TIME_MS = 20  # one 50 Hz tick; each streamed point is sent as a short "move"
READ_TIMEOUT_S = 0.05
WRITE_TIMEOUT_S = 0.01  # keep the 50 Hz streamer from blocking on a stalled port


class SerialArm(ArmInterface):
    def __init__(self, config: ArmConfig) -> None:
        if config.simulated:
            raise ConfigError("SerialArm refuses a SIMULATION-ONLY config (mock_profile)")
        self._cfg = config
        self._port: Any = None
        self._lock = threading.Lock()

    def check_confirmed(self) -> None:
        """Raise HardwareNotConfirmedError for the first missing hardware value."""
        require(self._cfg.servo_voltage_v, "servo PSU voltage (safety interlock)", "OQ-3")
        self._cfg.limits()  # OQ-4
        require(self._cfg.serial_port, "bridge serial port", "OQ-10")
        require(self._cfg.host_baud, "host<->bridge baud (bridge firmware TBD)", "A-5")

    def connect(self) -> None:
        self.check_confirmed()
        import serial  # lazy: pyserial

        self._port = serial.Serial(
            port=self._cfg.serial_port,
            baudrate=self._cfg.host_baud,
            timeout=READ_TIMEOUT_S,
            write_timeout=WRITE_TIMEOUT_S,
        )
        log.info("SerialArm connected on %s @ %s", self._cfg.serial_port, self._cfg.host_baud)

    def close(self) -> None:
        if self._port is not None:
            self._port.close()
            self._port = None

    def _send(self, msg: bp.Command) -> None:
        if self._port is None:
            raise LBYRError("SerialArm not connected")
        data = bp.encode(msg)
        with self._lock:
            self._port.write(data)

    def move_joints(self, joints_deg: Sequence[float], time_ms: int) -> None:
        self._send(bp.MoveCmd(tuple(float(j) for j in joints_deg), int(time_ms)))

    def stream_point(self, joints_deg: Sequence[float]) -> None:
        self._send(bp.MoveCmd(tuple(float(j) for j in joints_deg), STREAM_TIME_MS))

    def read_state(self) -> ArmState:
        if self._cfg.feedback_readable is not True:
            raise HardwareNotConfirmedError("servo position/load feedback", "OQ-5")
        if self._port is None:
            raise LBYRError("SerialArm not connected")
        with self._lock:
            self._port.write(bp.encode(bp.StateRequest()))
            line = self._port.readline()
        if not line:
            raise LBYRError("bridge did not answer 'state' within timeout")
        resp = bp.decode_state_response(line)
        return ArmState(resp.joints, resp.load, resp.temp, resp.status, time.perf_counter())

    def torque(self, on: bool) -> None:
        self._send(bp.TorqueCmd(on))

    def stop(self) -> None:
        self._send(bp.StopCmd())
