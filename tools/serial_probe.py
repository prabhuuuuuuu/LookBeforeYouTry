"""Ask the ESP32 bridge for its state once and print the reply. READ-ONLY.

!!! RUN ONLY WITH THE SOFTWARE LEAD PRESENT !!!

WHY
    Checks that the USB serial link to the ESP32 bridge works (bible Section 6) and
    what the bridge reports (helps answer OQ-5: is position/load feedback readable?).

WHAT IT SENDS
    Exactly one line: {"type":"state"}
    It NEVER sends move / traj / torque commands. It cannot make the arm move.

BEFORE YOU RUN
    * Software lead is next to you.
    * Servo power is OFF unless the lead says otherwise. E-stop within reach.
    * Use a DATA-capable USB cable (charge-only cables show no serial port).
    * On the Jetson the port is usually /dev/ttyACM0 or /dev/ttyUSB0
      (ls /dev/ttyACM* /dev/ttyUSB*). On Windows it is COM3, COM4, ...
    * You may need to be in the "dialout" group on Linux:
      sudo usermod -aG dialout $USER  (then log out and in)

HOW TO USE
    python tools/serial_probe.py --port /dev/ttyACM0 --baud 115200 --confirm

WHAT TO REPORT
    Copy the raw reply and the "decoded" line (or the error) to the software lead.
"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--port", required=True, help="e.g. /dev/ttyACM0 or COM3")
    p.add_argument("--baud", type=int, default=115200, help="must match the bridge firmware")
    p.add_argument("--timeout", type=float, default=1.0, help="seconds to wait for a reply")
    p.add_argument(
        "--confirm", action="store_true", help="required: confirms the software lead is present"
    )
    args = p.parse_args()

    if not args.confirm:
        print("Refusing to run without --confirm (software lead must be present).")
        return 2

    import serial

    from lbyr.arm import bridge_proto as bp
    from lbyr.errors import ProtocolError

    request = bp.encode(bp.StateRequest())
    print(f"Opening {args.port} @ {args.baud} ...")
    try:
        port = serial.Serial(args.port, args.baud, timeout=args.timeout)
    except serial.SerialException as e:
        print(f"ERROR: cannot open port: {e}")
        return 1
    with port:
        time.sleep(0.5)  # some ESP32 boards reset when the port opens
        port.reset_input_buffer()
        print(f"sending: {request!r}")
        port.write(request)
        reply = port.readline()
    if not reply:
        print("No reply within timeout. Check: baud rate, cable, firmware running?")
        return 1
    print(f"raw reply: {reply!r}")
    try:
        state = bp.decode_state_response(reply)
    except ProtocolError as e:
        print(f"reply does not match the protocol: {e}")
        return 1
    print(
        f"decoded: joints={state.joints} load={state.load} temp={state.temp} status={state.status}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
