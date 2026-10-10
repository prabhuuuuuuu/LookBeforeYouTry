# Open Questions

Source: bible Section 20 (`docs/LookBeforeYouTry-context.md`), plus questions the software
surfaced while scaffolding. **Nothing below may be guessed in code.** Every item maps to a
config field that is `None` today; code that needs it raises `HardwareNotConfirmedError`.

How to close an item: write the answer, who measured it, and the date in the **Answer**
column, then the software lead updates `lbyr/config.py`. `python -m lbyr.main` (no `--mock`)
prints every item that still blocks real-hardware runs.

## Hardware (bible Section 20)

| # | Question | Unblocks (config field / code) | Answer (value, who, date) |
|---|---|---|---|
| [ ] OQ-1 | ROT3U servo bus type: serial Feetech-style or PWM? | `ArmConfig.servo_bus`; bridge firmware design | |
| [ ] OQ-2 | Servo baud rate and IDs (and which ID is which joint) | `ArmConfig.servo_baud`, `servo_ids`; joint order vs `JOINT_NAMES` | |
| [ ] OQ-3 | Exact servo PSU voltage and current. **Do not assume 5 V.** | `ArmConfig.servo_voltage_v` (safety interlock in `SerialArm.connect`) | |
| [ ] OQ-4 | Official URDF/CAD, or measured link lengths; joint limits in degrees | `ArmConfig.link_lengths_mm`, `joint_min_deg`, `joint_max_deg`; `IkpyIK` | |
| [ ] OQ-5 | Is servo position/load feedback readable? Load units? | `ArmConfig.feedback_readable`; `SerialArm.read_state`; jam detection | |
| [ ] OQ-6 | Gripper stroke and closure-to-angle mapping (open deg, closed deg) | `GripperConfig.open_deg`, `closed_deg` | |
| [ ] OQ-7 | Trinet live-stream mode: UVC webcam or vendor SDK? Device index? | `WristCameraConfig.device_index`; `TrinetWristCamera` | |
| [ ] OQ-8 | Trinet resolution/FPS that works reliably on the Jetson | `WristCameraConfig.width/height/fps` (`tools/fps_test.py`) | |
| [ ] OQ-9 | Trinet fisheye calibration (K, D, RMS) | `data/calib/trinet_fisheye.yaml` (`tools/calibrate_fisheye.py`) | |
| [ ] OQ-10 | USB port plan: RealSense on USB 3.0; Trinet and ESP32 on stable separate ports; ESP32 device name | `ArmConfig.serial_port` | |
| [ ] OQ-11 | Final tripod height and tilt at venue | hand-eye calibration (`base_T_cam.yaml`) | |
| [ ] OQ-12 | Final wrist mount geometry for Trinet (gripper ROI crop) | `WristCameraConfig.crop_xywh` | |

## Surfaced by software (not in the bible)

| # | Question | Unblocks | Answer |
|---|---|---|---|
| [ ] A-1 | Servo max speed (deg/s) and a safe joint acceleration (deg/s^2). The 30% dev / 50% stage caps are fractions of this. | `ArmConfig.max_joint_speed_deg_s`, `max_joint_accel_deg_s2` | |
| [ ] A-2 | Max delta per 20 ms tick the bridge should allow (deg) | `ArmConfig.max_delta_per_tick_deg`; firmware clamp | |
| [ ] A-3 | Safe / home pose (6 joint angles, deg) used on ABORT | `ArmConfig.safe_pose_deg` | |
| [ ] A-4 | Load units and a jam threshold | `ArmConfig.jam_load_threshold` | |
| [ ] A-5 | Host <-> ESP32 baud (bible: >= 115200), fixed when firmware is written | `ArmConfig.host_baud` | |
| [ ] A-6 | Bridge `status` values other than `"ok"` (host treats any non-`ok` as JAMMED) | `bridge_proto`, firmware | |
| [ ] A-7 | Joint zero pose and sign conventions (what does 0 deg mean per joint?) | `IkpyIK`, safe pose | |

## Software policy decisions (approved by software lead, scaffold session)

| # | Decision |
|---|---|
| D-1 | "+15% closure" = +15 percentage points (80 -> 95), capped at 100. |
| D-2 | "Wrist re-orient 30 deg" = wrist roll. SLIPPING's 3rd step (lateral 10 mm) is kept as data but unreachable while max attempts = 2. |
| D-3 | Recovery budget: per-class max attempts AND a global RECOVER(k), k <= 2 per episode. |
| D-4 | VERIFY_PLACE: `EMPTY` = success; `GRASPED`/`SLIPPING` -> PLACE_FAILED -> ABORT. The locked Critic prompt is reused unchanged. |
| D-5 | Critic `TIMEOUT`/`INVALID`: re-query once (not counted in k), then ABORT. Never treated as success. |
| D-6 | Default Critic timeout 2.0 s (`CriticConfig.timeout_s`). |
| D-7 | Critic OFF baseline: verify states pass through; verdict logged as `null`. |
| D-8 | **Proposed, please confirm:** EMPTY recovery "-5 mm standoff" is implemented as "grasp 5 mm lower" (the pre-grasp standoff from the planner is unchanged). |
| D-9 | JAMMED abort: stop, stop streaming, torque OFF (bible 13). Other aborts: move to safe pose and hold with torque ON; the gripper is not opened. |
