# Open Questions

Source: bible Section 20 (`docs/LookBeforeYouTry-context.md`), plus questions the software
surfaced while scaffolding. **Nothing below may be guessed in code.** Every item maps to a
config field that is `None` today; code that needs it raises `HardwareNotConfirmedError`.

How to close an item: write the answer, who measured it, and the date in the **Answer**
column, then the software lead updates `lbyr/config.py`. `python -m lbyr.main` (no `--mock`)
prints every item that still blocks real-hardware runs.

**Status (bible v1.3, 2026-10-10):** OQ-1, OQ-2, OQ-3, OQ-5 are closed (PWM servos, joint roles
confirmed, 6.0 V rail, no feedback). Bridge firmware exists (`firmware/xiao_c6_bridge/`).
The matching `lbyr/config.py` fields have **not** been updated yet. The zeros in
`data/calib/zero_positions.json` (git-ignored, not in the repo) are still to be verified
against the board's `list` output (`S,<joint>,<zero_us>,...` lines); the firmware's
`DEFAULT_ZERO_US` fallback was copied from that file.

**Status (bible v1.4, 2026-10-10):** model cost measured (bible 9.2). Planner default is now
`rules`; Qwen only with `backend = qwen` and only if the memory guard passes (D-10..D-12).

**Status (bible v1.5, 2026-10-10):** M-1 closed (Critic = Moondream2 BF16 on a 378x378 crop).
Language mode served by Qwen3-1.7B text (M-2 closed). Critic accuracy on real wrist
frames is still NOT measured (no labelled set in `data/datasets/critic/`). The crop box must
be re-centred on the gripper bite zone once the Trinet is mounted (OQ-12).

## Hardware (bible Section 20)

| # | Question | Unblocks (config field / code) | Answer (value, who, date) |
|---|---|---|---|
| [x] OQ-1 | ROT3U servo bus type: serial Feetech-style or PWM? | `ArmConfig.servo_bus`; bridge firmware design | **PWM.** Bible v1.3, 2026-10-10. |
| [x] OQ-2 | Servo baud rate and IDs (and which ID is which joint) | `ArmConfig.servo_baud`, `servo_ids`; joint order vs `JOINT_NAMES` | Baud/IDs: not applicable (PWM). Joints, base to gripper: 1 base yaw (Feetech 25 kg), 2 shoulder (Feetech), 3 elbow (Feetech), 4 wrist pitch (Feetech), 5 wrist roll (MG996R), 6 gripper (MG996R). Matches `JOINT_NAMES` order. Bible v1.3, 2026-10-10. |
| [x] OQ-3 | Exact servo PSU voltage and current. **Do not assume 5 V.** | `ArmConfig.servo_voltage_v` (safety interlock in `SerialArm.connect`) | **6.0 V** (MG996R max ~7.2 V). PSU current rating was not recorded. Bible v1.3, 2026-10-10. |
| [ ] OQ-4 | Official URDF/CAD, or measured link lengths; joint limits in degrees | `ArmConfig.link_lengths_mm`, `joint_min_deg`, `joint_max_deg`; `IkpyIK` | |
| [x] OQ-5 | Is servo position/load feedback readable? Load units? | `ArmConfig.feedback_readable`; `SerialArm.read_state`; jam detection | **No feedback.** "Position" is the last commanded pulse. JAMMED = operator stop only (bible 13) unless an INA226 current sensor is added. Bible v1.3, 2026-10-10. |
| [ ] OQ-6 | Gripper stroke and closure-to-angle mapping (open deg, closed deg) | `GripperConfig.open_deg`, `closed_deg` | |
| [ ] OQ-7 | Trinet live-stream mode: UVC webcam or vendor SDK? Device index? | `WristCameraConfig.device_index`; `TrinetWristCamera` | |
| [ ] OQ-8 | Trinet resolution/FPS that works reliably on the Jetson | `WristCameraConfig.width/height/fps` (`tools/fps_test.py`) | |
| [ ] OQ-9 | Trinet fisheye calibration (K, D, RMS) | `data/calib/trinet_fisheye.yaml` (`tools/calibrate_fisheye.py`) | |
| [ ] OQ-10 | USB port plan: RealSense on USB 3.0; Trinet and XIAO ESP32-C6 on stable separate ports; XIAO device name (`/dev/ttyACM*`?) | `ArmConfig.serial_port` | |
| [ ] OQ-11 | Final tripod height and tilt at venue | hand-eye calibration (`base_T_cam.yaml`) | |
| [ ] OQ-12 | Final wrist mount geometry for Trinet (gripper ROI crop) | `WristCameraConfig.crop_xywh` | |

## Surfaced by software (not in the bible)

| # | Question | Unblocks | Answer |
|---|---|---|---|
| [ ] A-1 | Servo max speed (deg/s) and a safe joint acceleration (deg/s^2). The 30% dev / 50% stage caps are fractions of this. Firmware defaults are 30 deg/s and 60 deg/s^2 (caps 60 / 200, joints 5-6 scaled x0.35); these are firmware limits, not measured servo maxima. | `ArmConfig.max_joint_speed_deg_s`, `max_joint_accel_deg_s2` | |
| [ ] A-2 | Max delta per 20 ms tick the bridge should allow (deg). The firmware has no per-tick clamp; it limits speed/accel instead. | `ArmConfig.max_delta_per_tick_deg`; firmware clamp | |
| [ ] A-3 | Safe / home pose (6 joint angles, deg) used on ABORT. The firmware's `home` goes to the saved zeros, which must be this pose or a decision is needed. | `ArmConfig.safe_pose_deg` | |
| [x] A-4 | Load units and a jam threshold | `ArmConfig.jam_load_threshold` | Moot: no feedback (OQ-5). Revisit only if an INA226 is added. |
| [x] A-5 | Host <-> bridge baud (bible: >= 115200), fixed when firmware is written | `ArmConfig.host_baud` | **115200** (firmware `Serial.begin(115200)`). |
| [ ] A-6 | Bridge `status` values other than `"ok"` (host treats any non-`ok` as JAMMED) | `bridge_proto`, firmware | Superseded: the firmware replies `OK ...` / `ERR ...` text lines and has no `status` field. Decide the host mapping when the text protocol is implemented (below). |
| [ ] A-7 | Joint zero pose and sign conventions (what does 0 deg mean per joint?) | `IkpyIK`, safe pose | Firmware: 0 deg = saved zero pulse, +deg = longer pulse, 1000/90 us per deg nominal. Zeros (us) in the firmware fallback: 1477, 1989, 1844, 1223, 1467, 1433. Still to verify against `list` and `data/calib/zero_positions.json`; sign per joint and the physical meaning of the zero pose are unconfirmed. |

## Host protocol plan: SerialArm / bridge_proto (recorded 2026-10-10, NOT implemented)

The firmware (`firmware/xiao_c6_bridge/Code.ino`, bible v1.3 Section 6) speaks a **text**
protocol at 115200, not the JSON protocol that `lbyr/arm/bridge_proto.py` and
`SerialArm` implement today. Until this is done, `SerialArm` cannot drive the real board.
Plan (for a later increment; keep `bridge_proto` JSON + `MockArm` working for tests):

| Host call (`ArmInterface`) | Firmware text command | Notes |
|---|---|---|
| `move_joints(deg, time_ms)` | `m D1 .. D6` | Degrees relative to saved zero. The firmware has **no `time_ms`**: it moves at its own speed/accel limits and all joints arrive together. Duration is set with `speed V A`, not per move. |
| `stream_point(deg)` (50 Hz) | `m D1 .. D6` (see PP-1) | Open problem PP-1. |
| `stop()` | `x` | Brakes smoothly, then holds. |
| `torque(False)` | `off` | Releases PWM; the arm may fall. Only with the arm supported. |
| `torque(True)` | `home` | Re-attaches released joints at their zeros and moves there. There is no way to re-enable torque without moving. |
| `read_state()` | `list` | Returns **commanded** pulses (`S,<joint>,<zero_us>,<pos_us>,<attached>,<name>` x6, then `V,<vmax>,<amax>`), not measured angles. Convert with `(pos_us - zero_us) / (1000/90)`. `load` and `temp` are `None`. `ArmState.status` must not be treated as measured feedback (OQ-5). |
| `connect()` | open port, then expect the boot `help` text | The board homes itself at boot (arm must already rest in the zero pose). Host must not send motion until homing is finished. |

Replies are `OK ...` / `ERR ...` lines; `list` is multi-line. No reply is a timeout.

Open points for that increment:

| # | Question |
|---|---|
| PP-1 | The firmware re-plans from the current position on every command (accel-limited, backlash overshoot, `m` scales speeds by remaining distance). A 50 Hz stream of `m` commands will likely not behave like the trapezoid trajectories. Options: (a) send sparse waypoints (about 5-10 Hz) and let the firmware profile them; (b) add a firmware streaming/"direct set" mode (needs a firmware change, not made yet); (c) accept it after bench testing with `tools/arm_gui.py`. Decide before wiring `MotionStreamer` to the real arm. |
| PP-2 | Degrees in the host are relative to each joint's saved zero. IK and `ArmConfig.safe_pose_deg` need that convention (A-3, A-7). Joint limits in degrees (OQ-4) must be enforced on the host because the firmware only clamps pulses to 500-2500 us. |
| PP-3 | The firmware has no link-loss watchdog and no per-tick delta clamp (A-2). PWM servos hold the last pulse on link loss. Decide whether to add a watchdog in firmware. |
| PP-4 | `ArmConfig.servo_bus` / `servo_ids` / `servo_baud` need to become "pwm" / not applicable; `SerialArm.check_confirmed` should drop the IDs/baud checks and use A-5 = 115200. Preflight in `lbyr/main.py` lists open questions from config, so closing OQ-1/2/3/5 in `config.py` is a separate step. |

## Software policy decisions (approved by software lead, scaffold session)

| # | Decision |
|---|---|
| D-1 | "+15% closure" = +15 percentage points (80 -> 95), capped at 100. |
| D-2 | "Wrist re-orient 30 deg" = wrist roll. SLIPPING's 3rd step (lateral 10 mm) is kept as data but unreachable while max attempts = 2. |
| D-3 | Recovery budget: per-class max attempts AND a global RECOVER(k), k <= 2 per episode. |
| D-4 | VERIFY_PLACE: `EMPTY` = success; `GRASPED`/`SLIPPING` -> PLACE_FAILED -> ABORT. The locked Critic prompt is reused unchanged. |
| D-5 | Critic `TIMEOUT`: re-query once (not counted in k), then ABORT. Never treated as success. `INVALID`: superseded by D-13. |
| D-6 | Default Critic timeout 2.0 s (`CriticConfig.timeout_s`). |
| D-7 | Critic OFF baseline: verify states pass through; verdict logged as `null`. |
| D-8 | **Proposed, please confirm:** EMPTY recovery "-5 mm standoff" is implemented as "grasp 5 mm lower" (the pre-grasp standoff from the planner is unchanged). |
| D-9 | JAMMED abort: stop, stop streaming, torque OFF (bible 13). Other aborts: move to safe pose and hold with torque ON; the gripper is not opened. |
| D-10 | (2026-10-10) `planner.backend` defaults to `rules`; Qwen is not loaded unless `backend = qwen`. In-loop calls keep the 3 s timeout. |
| D-11 | (2026-10-10) Qwen "language mode" for the demo intro only (`plan_language`): text-only prompt, compact `{"a","s","g","p"}` schema, 64 new tokens max, `planner.qwen_timeout_s` = 8 s; strict parser unchanged. Every plan is logged as a `reasoning_trace` event. |
| D-12 | (2026-10-10) Memory guard: with `backend = qwen` or `language_mode`, refuse to start the episode loop if free system RAM after loading all models is < `planner.qwen_min_free_ram_mb` (1536 MB), or if RAM cannot be read. |
| D-13 | (2026-10-10) Critic `INVALID` (unparseable answer): re-query once with a fresh frame (not counted in k), then treat as `OCCLUDED` (its recovery: re-position wrist cam, re-query; then ABORT). Logged as `critic_invalid_retry` (`action` = `requery` / `treat_as_occluded`). Every verdict record has the raw answer, verdict, total latency and inference time. |

## Model performance decisions

| # | Question | Unblocks | Answer |
|---|---|---|---|
| [x] M-1 | Which Critic speed variant becomes the default? | `CriticConfig.dtype`, `crop_enabled`, `crop_xywh` | **Closed 2026-10-10 (software lead):** Moondream2 kept as the Critic. `dtype = bf16` (FP16 selectable), runtime input = dewarped frame → 378x378 crop (`CriticConfig.crop_xywh`, default centred; re-centre on the gripper bite zone after mounting, OQ-12). `compile()` not adopted (no Triton). Measured runtime p50 0.85 s. Accuracy still NOT measured. |
| [x] M-2 | Which language-mode planner? Candidates: Qwen3-1.7B text, Qwen3-0.6B text, Qwen2.5-VL-3B text-only. | `PlannerConfig.language_backend`, `language_model_path` | **Closed 2026-10-10 (software lead): Qwen3-1.7B** (`language_backend = qwen3_text`, `~/models/qwen3-1.7b`, already the default). With jetson_clocks: p50 2.14 s, 9.4 tok/s, 9/10 correct; Critic + Qwen3-1.7B leave 4.49 GB free (guard PASS). Qwen3-0.6B rejected (2/10 correct). |

M-1 measurements (2026-10-10, Orin NX 40 W, synthetic 640x480 frames, 10 timed calls per
variant, `tools/bench_models.py --critic-only ...`):

| Variant | Input | p50 | p95 | p50 vs full | torch peak | RAM peak (tegrastats) |
|---|---|---:|---:|---:|---:|---:|
| bf16, full | 640x480 | 1.315 s | 1.322 s | 1.00 | 4.14 GB | 6.70 GB |
| bf16, full (2nd run) | 640x480 | 1.333 s | 1.337 s | 1.00 | 4.14 GB | 6.72 GB |
| bf16, crop | 378x378 | 0.851 s | 0.855 s | 0.64 | 4.13 GB | 6.70 GB |
| fp16, full | 640x480 | 1.409 s | 1.413 s | 1.00 | 4.14 GB | 6.85 GB |
| fp16, crop | 378x378 | 0.857 s | 0.863 s | 0.61 | 4.13 GB | 6.83 GB |

`--critic-compile`: not measured. Triton is not installed in the Jetson venv, so
`torch.compile` (inductor) is expected to fail on the first call; the bench reports that
as "skipped".

M-2 measurements, before jetson_clocks (2026-10-10, Orin NX 40 W, `tools/bench_models.py --planner-models
qwen3-1.7b,qwen3-0.6b --combined`; 10 fixed commands, 6-object detection list, compact
schema, greedy, thinking off, full generation without the 8 s cut-off). Each model loaded
alone:

| Model | Load + warm-up | p50 | p95 | tok/s | tokens | valid parse | correct targets | within 8 s | torch alloc | RAM peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 10.7 s | 2.37 s | 2.40 s | 8.4 | 20 | 9/10 | 9/10 | 10/10 | 3.21 GB | 6.79 GB |
| Qwen3-0.6B | 2.9 s | 2.82 s | 3.21 s | 8.4 | 24 | 5/10 | 2/10 | 10/10 | 1.13 GB | 4.64 GB |

- Qwen3-1.7B miss: "Place the acrylic block into zone B." -> `"p":5` (no such id), rejected by
  the strict parser.
- Qwen3-0.6B: 5 answers used string ids (`"a": "4"`), rejected; 3 picked the wrong zone.
- Qwen2.5-VL-3B text-only was not re-run in this session (its full-schema in-loop number is
  11.2 s; it does not fit next to the Critic under the memory guard).
- The 0.6B "RAM +load" from tegrastats is not reliable (the 1.7B was still being freed);
  use torch alloc.

Combined (Critic loaded first, then the planner, interleaved calls):

| Critic | Planner | torch alloc | RAM used after load | RAM free | RAM peak | Critic p50 | Planner p50 | Memory guard |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Moondream2 bf16, 378x378 crop | Qwen3-1.7B | 7.20 GB | 10.75 / 15.29 GB | 4.54 GB | 10.78 GB | 0.91 s | 2.32 s | PASS (>= 1536 MB) |

M-2 re-run with jetson_clocks (2026-10-10, 40 W, CPU 1.5 GHz, GPU 1173 MHz, EMC 3199 MHz;
`--planner-models qwen3-1.7b --combined`, `data/bench/bench-20261010-224019.json`):
Qwen3-1.7B p50 2.14 s / p95 2.18 s, 9.4 tok/s, 9/10 valid and correct (same zone-id miss),
load+warm-up 7.7 s. Combined: torch 7.20 GB, RAM 10.80 / 15.29 GB, 4.49 GB free, guard PASS,
Critic p50 0.78 s, planner p50 2.13 s.
