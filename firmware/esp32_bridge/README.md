# ESP32 bridge firmware (placeholder)

**Status: firmware not written yet.** Blocked on OQ-1 (serial-bus vs PWM servos), OQ-2
(baud, IDs), OQ-3 (voltage), OQ-4 (joint limits), OQ-5 (feedback), A-2, A-5, A-6.
See `docs/OPEN_QUESTIONS.md`.

## Link (bible Section 6)

Jetson USB-A host -> data-capable USB cable -> ESP32-S3 USB port (USB CDC serial,
`/dev/ttyACM0` or `/dev/ttyUSB0`). The USB link carries **data and optional ESP32 logic
power only, never servo power.**

## Protocol: newline-delimited JSON, >= 115200 baud

The host-side reference implementation and validator is `lbyr/arm/bridge_proto.py`.
Units: degrees, milliseconds. 6 joints: base yaw, shoulder pitch, elbow pitch, wrist
pitch, wrist roll, gripper.

```json
{"type":"move","joints":[0,45,60,30,0,50],"time_ms":800}
{"type":"traj","pts":[[...],[...]],"dt_ms":20}
{"type":"state"}
{"type":"torque","on":false}
{"type":"stop"}
```

Reply to `state`:

```json
{"joints":[...],"load":[...],"temp":[...],"status":"ok"}
```

`load` and `temp` may be `null` if not readable (OQ-5). The host streams one `move`
with `time_ms = 20` per 50 Hz tick.

## Responsibilities the firmware must enforce (bible Section 6)

1. Joint clamps (limits from OQ-4).
2. Maximum delta per tick (A-2).
3. Watchdog: if no command arrives within a timeout, hold.
4. Hold position on link loss.
5. Safe `stop` handling (hold current position immediately).

Actuation backend (host does not care which):

* Serial-bus servos: ESP32 -> SyncWrite packets -> Feetech-style bus.
* PWM servos: ESP32 -> PCA9685 -> servos.

Firmware clamps + host FSM limits + the human E-stop are three independent safety layers
(bible Section 8). The firmware layer must not rely on the host being correct.
