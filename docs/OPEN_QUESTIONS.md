# Open Questions

Source: bible Section 20 (`docs/LookBeforeYouTry-context.md`), plus questions the software
surfaced while scaffolding. **Nothing below may be guessed in code.** Every item maps to a
config field that is `None` today; code that needs it raises `HardwareNotConfirmedError`.

How to close an item: write the answer, who measured it, and the date in the **Answer**
column, then the software lead updates `lbyr/config.py`. `python -m lbyr.main` (no `--mock`)
prints every item that still blocks real-hardware runs.

**Status (bible v1.3, 2026-10-10):** OQ-1, OQ-2, OQ-3, OQ-5 are closed (PWM servos, joint roles
confirmed, 6.0 V rail, no feedback). Bridge firmware exists (`firmware/xiao_c6_bridge/`).
The matching `lbyr/config.py` fields were updated in v1.6 (below). The zeros in
`data/calib/zero_positions.json` (git-ignored, not in the repo) are still to be verified
against the board's `list` output (`S,<joint>,<zero_us>,...` lines); the firmware's
`DEFAULT_ZERO_US` fallback was copied from that file.

**Status (bible v1.4, 2026-10-10):** model cost measured (bible 9.2). Planner default is now
`rules`; Qwen only with `backend = qwen` and only if the memory guard passes (D-10..D-12).

**Status (bible v1.5, 2026-10-10):** M-1 closed (Critic = Moondream2 BF16 on a 378x378 crop).
Language mode served by Qwen3-1.7B text (M-2 closed). Critic accuracy on real wrist
frames is still NOT measured (no labelled set in `data/datasets/critic/`). The crop box must
be re-centred on the gripper bite zone once the Trinet is mounted (OQ-12).

**Status (bible v1.6, 2026-10-10):** PP-1 resolved: the firmware is the trajectory generator;
one `m` per segment, `DONE` on completion (firmware updated, NOT yet flashed / tested on the
board). `SerialArm` speaks the text protocol (PP-4 done; config now has pwm / 6.0 V / 115200
/ no feedback). `data/calib/` is tracked: `joints.yaml` is seeded from the firmware's
`DEFAULT_ZERO_US` and is **unverified**; `poses.yaml` is empty (teach with tools/arm_gui.py).
Demo path = taught-pose mode (`motion.mode: taught`), so OQ-4 / OQ-6 / A-1 / A-3 no longer
block it (`python -m lbyr.main` preflight).

## Hardware (bible Section 20)

| # | Question | Unblocks (config field / code) | Answer (value, who, date) |
|---|---|---|---|
| [x] OQ-1 | ROT3U servo bus type: serial Feetech-style or PWM? | `ArmConfig.servo_bus`; bridge firmware design | **PWM.** Bible v1.3, 2026-10-10. |
| [x] OQ-2 | Servo baud rate and IDs (and which ID is which joint) | `ArmConfig.servo_baud`, `servo_ids`; joint order vs `JOINT_NAMES` | Baud/IDs: not applicable (PWM). Joints, base to gripper: 1 base yaw (Feetech 25 kg), 2 shoulder (Feetech), 3 elbow (Feetech), 4 wrist pitch (Feetech), 5 wrist roll (MG996R), 6 gripper (MG996R). Matches `JOINT_NAMES` order. Bible v1.3, 2026-10-10. |
| [x] OQ-3 | Exact servo PSU voltage and current. **Do not assume 5 V.** | `ArmConfig.servo_voltage_v` (safety interlock in `SerialArm.connect`) | **6.0 V** (MG996R max ~7.2 V). PSU current rating was not recorded. Bible v1.3, 2026-10-10. |
| [ ] OQ-4 | Official URDF/CAD, or measured link lengths; joint limits in degrees | `ArmConfig.link_lengths_mm`, `joint_min_deg`, `joint_max_deg`; `IkpyIK` | |
| [x] OQ-5 | Is servo position/load feedback readable? Load units? | `ArmConfig.feedback_readable`; `SerialArm.read_state`; jam detection | **No feedback.** "Position" is the last commanded pulse. JAMMED = operator stop only (bible 13) unless an INA226 current sensor is added. Bible v1.3, 2026-10-10. |
| [ ] OQ-6 | Gripper stroke and closure-to-angle mapping (open deg, closed deg) | `GripperConfig.open_deg`, `closed_deg` (ik mode); taught mode: `gripper.open_deg/closed_deg` in `data/calib/poses.yaml` | |
| [ ] OQ-7 | Trinet live-stream mode: UVC webcam or vendor SDK? Device index? | `WristCameraConfig.device_index`; `TrinetWristCamera` | |
| [ ] OQ-8 | Trinet resolution/FPS that works reliably on the Jetson | `WristCameraConfig.width/height/fps` (`tools/fps_test.py`) | |
| [ ] OQ-9 | Trinet fisheye calibration (K, D, RMS) | `data/calib/trinet_fisheye.yaml` (`tools/calibrate_fisheye.py`) | |
| [ ] OQ-10 | USB port plan: RealSense on USB 3.0; Trinet and XIAO ESP32-C6 on stable separate ports; XIAO device name (`/dev/ttyACM*`?) | `ArmConfig.serial_port` | |
| [ ] OQ-11 | Final tripod height and tilt at venue | hand-eye calibration (`base_T_cam.yaml`) | |
| [ ] OQ-12 | Final wrist mount geometry for Trinet (gripper ROI crop) | `CriticConfig.crop_xywh` (378x378 Critic crop; default centred) | |

## Surfaced by software (not in the bible)

| # | Question | Unblocks | Answer |
|---|---|---|---|
| [ ] A-1 | Servo max speed (deg/s) and a safe joint acceleration (deg/s^2). The 30% dev / 50% stage caps are fractions of this. Firmware defaults are 30 deg/s and 60 deg/s^2 (caps 60 / 200, joints 5-6 scaled x0.35); these are firmware limits, not measured servo maxima. | Since PP-1 the firmware's `speed V A` sets real motion speed (SerialArm reads it from the `V` line); `ArmConfig` values are used by MockArm only. Not blocking the demo. | |
| [x] A-2 | Max delta per 20 ms tick the bridge should allow (deg). | — | **Superseded by PP-1** (2026-10-10): no setpoint streaming; the firmware limits speed/accel itself. |
| [ ] A-3 | Safe / home pose (6 joint angles, deg) used on ABORT. The firmware's `home` goes to the saved zeros, which must be this pose or a decision is needed. | Taught mode: the taught `home` pose in `data/calib/poses.yaml` (operator decides by teaching it). ik mode: `ArmConfig.safe_pose_deg` | |
| [x] A-4 | Load units and a jam threshold | `ArmConfig.jam_load_threshold` | Moot: no feedback (OQ-5). Revisit only if an INA226 is added. |
| [x] A-5 | Host <-> bridge baud (bible: >= 115200), fixed when firmware is written | `ArmConfig.host_baud` | **115200** (firmware `Serial.begin(115200)`). |
| [x] A-6 | Bridge `status` values other than `"ok"` | `SerialArm` | **Decided 2026-10-10:** `ERR ...` after a motion command, a serial read/write failure, or no `DONE` within predicted x 1.5 + 0.5 s = **LINK** fault -> ABORT (stop sent, torque kept ON so the arm does not fall). Not JAMMED. |
| [ ] A-7 | Joint zero pose and sign conventions (what does 0 deg mean per joint?) | `data/calib/joints.yaml` (`zero_us`, `sign`, `zero_pose_deg`); `IkpyIK` | Firmware: 0 deg = saved zero pulse, +deg = longer pulse, 1000/90 us per deg nominal. `joints.yaml` is seeded with the firmware fallback zeros (1477, 1989, 1844, 1223, 1467, 1433), `sign: 1`, `zero_pose_deg: 0`, `verified: false`. **To do:** compare with the board's `list` and set `verified: true` (blocks real runs). Signs / zero-pose meaning are only needed for ik mode. |

## Host protocol: SerialArm <-> firmware text protocol (implemented 2026-10-10)

PP-1 decision: the firmware (`firmware/xiao_c6_bridge/Code.ino`) is the trajectory
generator. The host sends ONE `m` per motion segment and waits for `DONE`; no streaming.

| Host call (`ArmInterface`) | Firmware | Notes |
|---|---|---|
| `connect()` | `list`, then `q` until not moving | Zeros from the `S` lines must equal `data/calib/joints.yaml`, else `ZeroMismatchError` (no motion). Speed from the `V` line. Waits for boot homing to finish. |
| `move_joints(deg)` | `m D1 .. D6` | Host degrees -> zero-relative degrees (`sign`, `zero_pose_deg`). Returns the duration predicted by `lbyr/arm/fw_planner.py` (port of the firmware planner, checked against the compiled sketch in `tests/test_fw_planner.py`). |
| `motion_status()` | `OK m` then `DONE` | DONE counts only after the `OK m` of the latest segment. `ERR` / I/O error -> ERROR. The MotionController adds the timeout: predicted x 1.5 + 0.5 s -> LINK. |
| `request_state()` / `read_state()` | `q` -> `Q,<moving>,<cur_us x6>` | COMMANDED pulses (no feedback, OQ-5); `feedback = "commanded"`. |
| `stop()` | `x` | Brakes smoothly, then holds. |
| `torque(False)` | `off` | Releases PWM; the arm may fall. Only with the arm supported (JAMMED abort). |
| `torque(True)` | — | No-op while attached. Refuses after `off` (re-attaching jumps to the zero pulse; use Home in arm_gui). |

| # | Point | Status |
|---|---|---|
| [x] PP-1 | Streaming vs firmware planner | **Resolved 2026-10-10:** one `m` per segment + `DONE`. |
| [ ] PP-2 | Joint limits in degrees (OQ-4) must be enforced on the host; the firmware only clamps pulses to 500-2500 us. | Open (taught poses are taught inside the safe range; ik mode needs OQ-4). |
| [ ] PP-3 | No link-loss watchdog in the firmware; PWM servos hold the last pulse on link loss. | Open. The host detects LINK (no DONE) and aborts; a firmware watchdog is not added. |
| [x] PP-4 | Config / check_confirmed for PWM | **Done 2026-10-10:** `servo_bus = pwm`, 6.0 V, 115200, `feedback_readable = False`; `check_confirmed` needs only voltage, port (OQ-10) and baud. |

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
| D-14 | (2026-10-10) Motion PP-1: one `m` per segment + `DONE`, MotionController at 20 Hz (`q` poll). LINK fault (no DONE in predicted x 1.5 + 0.5 s, `ERR`, I/O error) -> ABORT: stop sent, torque kept ON, no further motion. JAMMED (mock / future sensors only) -> stop + torque OFF. |
| D-15 | (2026-10-10) `motion.mode = taught` is the default (demo path). Recovery mapping: +15% closure -> joint 6; wrist re-orient 30 deg -> joint 5; lateral 10 mm -> base yaw + atan(10 / radius_mm); -5 mm standoff -> pre-grasp moves along the joint-space segment zone_X_pre -> zone_X_grasp by 5 / standoff_mm (joint-space, not Cartesian; grasp pose unchanged). Object -> zone: detection centroid inside the zone polygon (overhead pixels); outside every zone -> IK_FAILED -> ABORT. The planner's standoff_mm is not used in taught mode. |

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
