Save as:

```markdown
# PROJECT CONTEXT / BIBLE — “Look Before You Retry” (LBYR)

**Version:** 1.2  
**Status:** Locked hackathon context  
**Hardware revision:** ROT3U 6-DOF arm + Jetson Orin 16GB + RealSense D435 + Panoculonlabs Trinet wrist camera  
**Supersedes:** Version 1.1 where conflicting  

> This document is the updated single source of truth for the hackathon build.  
> Where this document conflicts with older notes, this document wins.  
> The core research and architecture remain unchanged: edge-native, modular VLA, async visual Critic, no cloud at runtime, no ROS 2 during hackathon.

---

## 0. Locked Decisions Summary

| # | Decision | Locked Choice | Notes |
|---:|---|---|---|
| 1 | Robot arm | **ROT3U 6-DOF desktop arm** | Replaces any earlier MK3 / 3D-printed arm plan. |
| 2 | Main compute | **NVIDIA Jetson Orin 16GB** | All inference local. JetPack 6, MAXN, headless. |
| 3 | Overhead camera | **Intel RealSense D435** | Overhead RGB-D for detection + metric 3D. |
| 4 | Overhead camera mount height | **~70 cm above 40×40 cm mat** | Tilt ~50° downward. Lock exposure/white balance. |
| 5 | Wrist / gripper camera | **Panoculonlabs Trinet camera** | Replaces Raspberry Pi Camera v2 / IMX219 as primary wrist Critic camera. |
| 6 | Wrist camera interface | **USB live stream to Jetson Orin** | Do not use Jetson CSI ribbon cable for primary wrist camera. |
| 7 | Wrist camera image handling | **Dewarp before VLM** | Trinet ultra-wide/fisheye image must be rectified before Moondream2. |
| 8 | Wrist camera role | **Semantic-only Critic** | Not used for metric control or hand-eye calibration. |
| 9 | ESP32-to-Jetson link | **USB serial / USB CDC** | Use a good USB-A to USB-C / Micro-USB data cable. |
| 10 | Judge-facing display | **Laptop screen via USB HDMI capture card** | Preferred over external HDMI monitor for portability. |
| 11 | Workspace | **40×40 cm rigid mat** | Clamped to table. Contains ArUco / calibration grid and object start zones. |
| 12 | Object set | **Hard + soft failure-inducing objects** | See Section 7. All objects should be under ~150 g. |
| 13 | Architecture | **Modular VLA + deterministic motion stack** | No end-to-end policy during hackathon. |
| 14 | Runtime inference | **Edge-only** | No cloud APIs at runtime. |
| 15 | Software style | **Python 3.10+, threads + queues, no ROS 2** | 50 Hz motion thread must never block on AI. |

---

## 1. Project Identity

| Field | Value |
|---|---|
| Title | **Look Before You Retry: Edge-Native Visual Grounding for VLA Failure Recovery** |
| Event | Awign Hackathon 2026, Oct 10–11 |
| Track | Physical AI |
| Post-event goal | arXiv preprint + open-source repo + dataset |
| Team role split | Software/AI lead owns this document; hardware support assists assembly/wiring |
| Success criteria | Working live demo, trial-matrix numbers, public repo, arXiv draft |

---

## 2. Core Idea

A Vision-Language-Action system usually assumes that if the planner says “grasp,” the world complies. In reality, objects slip, grasps are empty, cameras become occluded, and joints jam.

This project adds a lightweight, edge-native visual **Critic** that verifies grasp state at critical FSM checkpoints.

The system does **not** blindly retry the same action. Instead:

```text
Planner proposes action
→ arm executes action
→ wrist camera verifies result
→ Critic outputs one-word verdict
→ FSM selects parameterized recovery policy
→ arm retries with corrected parameters or aborts safely
```

The core research contribution is not “a VLA controls an arm.” The contribution is:

1. **Event-driven visual self-verification** as a first-class edge component.
2. **Recovery policies conditioned on visually grounded failure classes.**
3. **Ablation evidence** that a small fine-tuned Critic beats zero-shot verification on cheap edge hardware.
4. **A released wrist-image grasp-verification dataset and trial protocol.**

---

## 3. Hardware Platform

### 3.1 Robot Arm

| Attribute | Locked / Assumed Value | Status |
|---|---|---|
| Arm | ROT3U 6-DOF desktop robotic arm | Locked |
| DOF | 6: base yaw, shoulder pitch, elbow pitch, wrist pitch, wrist roll, gripper | Confirm joint mapping |
| Class | Desktop arm | Locked |
| Reach | Approximately 300–450 mm | Confirm |
| Payload | Approximately 250–500 g | Confirm |
| Servo type | Likely Feetech STS/SCS-class serial-bus smart servos | Must confirm |
| Servo voltage | Likely 6.0–7.4 V, **not assumed 5 V** | Must confirm before powering |
| Feedback | Position/load/temperature may be readable | Confirm |
| Gripper | Parallel jaw, position-controlled | Confirm stroke and closure mapping |

Before any software work, the following must be confirmed:

1. Servo bus type: serial TTL or PWM.
2. Servo protocol, baud rate, and servo IDs.
3. Correct PSU voltage and current.
4. Joint limits in degrees.
5. Link lengths or official URDF.
6. Payload at full extension.
7. Whether position/load feedback is readable.
8. Gripper stroke and closure-to-angle mapping.
9. Trinet mounting point and cable routing path.

---

## 4. Compute Stack

| Item | Role | Interface | Notes |
|---|---|---|---|
| Jetson Orin 16GB | All AI inference, perception, FSM, logging | Headless JetPack 6 | Use MAXN: `sudo nvpmodel -m 0`, `sudo jetson_clocks` |
| Laptop | SSH, logs, judge-facing UI display | GbE / USB capture | Primary operator machine |
| USB HDMI capture card | Displays Jetson HDMI output on laptop | USB-A / USB-C | Required if using laptop as judge-facing screen |
| External HDMI monitor | Optional backup | HDMI | Not primary if capture card is available |

### Locked display decision

Primary judge-facing display:

```text
Jetson Orin HDMI out
→ USB HDMI capture card
→ laptop screen
```

The laptop shows the overlay UI, latency HUD, wrist camera verdict, FSM state, and detection boxes.

Do **not** assume a laptop HDMI port can accept HDMI input directly. Most laptop HDMI ports are output-only.

Fallback if capture card is unavailable:

```text
Jetson Orin
→ GStreamer / WebSocket / lightweight web stream
→ laptop browser or viewer
```

However, the capture card is preferred because it avoids network jitter and makes the demo look stable.

---

## 5. Sensor Setup

### 5.1 Overhead Camera: Intel RealSense D435

| Attribute | Locked Setup |
|---|---|
| Role | Overhead RGB-D perception |
| Use | YOLO detection + metric depth + 3D object pose estimation |
| Mount | Tripod or gantry |
| Height | Approximately **70 cm above the mat** |
| Tilt | Approximately **50° downward** |
| Workspace | 40×40 cm mat |
| Interface | USB 3.0 |
| Port | Use blue USB 3.0 port on Jetson |
| Camera settings | Lock exposure and white balance |

The D435 is the only camera used for metric 3D control.

Core perception equation:

```text
base_T_obj = base_T_cam · cam_T_obj
```

Where:

```text
cam_T_obj
```

is estimated from:

```text
YOLO centroid + median-filtered depth + pinhole unprojection
```

Hand-eye calibration:

- 12 tip-touch points on grid/ArUco.
- Solve with Umeyama or `cv2.calibrateHandEye`.
- Acceptance target: TCP reaches 10 test marks within `<8 mm`.

---

### 5.2 Wrist / Gripper Camera: Panoculonlabs Trinet

This decision replaces the Raspberry Pi Camera v2 / IMX219 as the primary wrist camera.

| Attribute | Locked Setup |
|---|---|
| Camera | Panoculonlabs Trinet |
| Role | Wrist-mounted semantic Critic camera |
| Use | Verify grasp state: `GRASPED`, `SLIPPING`, `EMPTY`, `OCCLUDED` |
| Interface | USB live stream to Jetson Orin |
| Mount location | Gripper / wrist area, looking at gripper bite zone |
| Metric calibration | Not required for control |
| Depth usage | Not required for primary control |
| Image preprocessing | Fisheye / ultra-wide dewarping required before VLM |
| Fallback | Raspberry Pi Camera v2 via Jetson CSI only if Trinet fails |

### Why Trinet is locked

1. It is designed for egocentric robotics data collection.
2. It is lightweight and suitable for wrist mounting.
3. It avoids fragile CSI ribbon cable routing.
4. It connects over USB, which is easier to debug at a hackathon.
5. Its wide field of view helps keep the gripper bite zone visible.

### Trinet integration rules

1. Use the Trinet in **USB live-stream mode** if available.
2. If it appears as a UVC webcam, read it with OpenCV:

```python
cap = cv2.VideoCapture(index)
```

3. If UVC is not available, use the vendor Python toolkit / SDK.
4. Capture at the lowest reliable resolution sufficient for the Critic.
5. Recommended starting point:

```text
720p @ 30 FPS
```

or

```text
1080p @ 15–30 FPS if USB bandwidth permits
```

6. Dewarp the image before sending it to Moondream2.

### Dewarping requirement

The Trinet uses an ultra-wide / fisheye lens. Moondream2 performs better on rectilinear images.

Required pipeline:

```text
Trinet raw frame
→ fisheye dewarp / undistort
→ crop to gripper region if useful
→ Moondream2 Critic query
```

Implementation:

```python
cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
```

Calibration:

- Estimate fisheye intrinsics using checkerboard / calibration target.
- Precompute remap matrices once at startup.
- Use remap per frame.
- Do not use Trinet depth or stereo output for closed-loop control unless explicitly validated after hackathon.

### Latency and VRAM cost

| Addition | Estimated Cost |
|---|---|
| USB camera capture | ~1–5 ms |
| Dewarp / undistort | ~2–5 ms |
| Extra buffers | Small, usually <0.2 GB |
| Critic inference | Unchanged: ~0.3–0.8 s |

The Trinet decision does not break the edge compute budget.

---

## 6. ESP32 Bridge Connection

### Locked connection

```text
Jetson Orin USB-A host
→ USB-A to USB-C / Micro-USB data cable
→ ESP32-S3 USB port
```

The ESP32 appears as a USB serial device.

Typical device names:

```text
/dev/ttyACM0
```

or

```text
/dev/ttyUSB0
```

### Cable requirement

Use a **data-capable USB cable**, not a charge-only cable.

Depending on the ESP32-S3 board:

| ESP32 Connector | Required Cable |
|---|---|
| USB-C | USB-A to USB-C data cable |
| Micro-USB | USB-A to Micro-USB data cable |

Bring spare cables.

### Communication protocol

JSON over USB serial.

Baud rate:

```text
115200 or higher
```

Example messages:

```json
{"type":"move","joints":[0,45,60,30,0,50],"time_ms":800}
```

```json
{"type":"traj","pts":[[...],[...]],"dt_ms":20}
```

```json
{"type":"state"}
```

Response:

```json
{"joints":[...],"load":[...],"temp":[...],"status":"ok"}
```

```json
{"type":"torque","on":false}
```

```json
{"type":"stop"}
```

### Bridge responsibilities

The ESP32 / arm MCU bridge must enforce:

1. Joint clamps.
2. Maximum delta per tick.
3. Watchdog behavior.
4. Hold position on link loss.
5. Safe stop command handling.

If serial-bus servos are used:

```text
ESP32 → SyncWrite packets → Feetech-style serial bus
```

If PWM servos are used:

```text
ESP32 → PCA9685 → PWM servos
```

The host software must not depend on which low-level actuation method is used.

### Important power rule

The USB link between Jetson and ESP32 is for:

```text
data + optional ESP32 logic power
```

It is **not** for powering the arm servos.

Never power servos from:

- ESP32 GPIO pins.
- Jetson USB 5V rail.
- MCU signal pins.
- PCA9685 signal pins.

Servos must be powered from a separate confirmed servo PSU.

---

## 7. Workspace and Physical Layout

### 7.1 Mat

| Attribute | Locked Setup |
|---|---|
| Mat size | 40×40 cm |
| Type | Rigid board or high-contrast cutting mat |
| Mounting | Clamp to table |
| Contents | ArUco markers / calibration grid / object zones |
| Purpose | Defines stable workspace and calibration frame |

The mat must not move during the demo. If the mat shifts, hand-eye calibration becomes invalid.

### 7.2 Overhead camera placement

```text
RealSense D435
Height: ~70 cm above mat
Tilt: ~50° downward
Mount: tripod or gantry
```

The tripod/gantry must be rigid. If the camera shakes, perception and calibration quality degrade.

### 7.3 Arm placement

```text
ROT3U arm bolted or clamped to board at edge of mat
```

The arm base must not shift during motion.

### 7.4 Wrist camera placement

```text
Trinet mounted near gripper
Looking at gripper bite zone
Cable strain-relieved
```

The camera should see:

- gripper fingers,
- object during close,
- object during lift,
- object during place.

Avoid mounting the camera so close that the gripper occludes the entire field of view.

### 7.5 Operator layout

```text
Arm on table
RealSense overhead
Jetson beside table
Laptop beside table
Capture card between Jetson HDMI and laptop
PSU + E-stop within operator reach
```

---

## 8. Power and Safety

### Locked power rules

1. Separate servo PSU.
2. Confirm servo voltage before powering.
3. Do not assume 5 V.
4. Use inline fuse on positive servo rail.
5. Use rocker switch / E-stop on positive servo rail.
6. Use bulk capacitor across servo rail.
7. Use single star ground at PSU negative terminal.
8. Do not add an extra ground wire from servo PSU to Orin.
9. USB already ties Jetson and ESP32 grounds.

### Grounding concept

```text
Servo PSU −V
Servo GND
ESP32 / logic GND
        ↓
single star ground
```

Jetson ground is connected through the USB link to ESP32. Do not create a second ground path to the Jetson.

### Safety rules

These are non-negotiable:

1. E-stop reachable by operator at all times.
2. Test E-stop before every session.
3. Torque off whenever hands are near the arm.
4. Development speed cap: 30%.
5. Stage speed cap: 50%.
6. Never run the arm unattended.
7. Support the arm by hand before cutting servo power.
8. Firmware clamps + FSM limits + human E-stop are three independent safety layers.

---

## 9. System Architecture

### 9.1 Architecture style

Locked:

```text
Modular VLA
= VLM planner
+ parameterized skill library
+ deterministic motion stack
+ async visual Critic
```

Not locked / explicitly out of scope for hackathon:

```text
End-to-end OpenVLA / SmolVLA / ACT policy training
```

That is post-hackathon work.

---

### 9.2 Component Budget on Orin 16GB

| Component | Model / Precision | Approx VRAM | Approx Latency | Rate |
|---|---|---:|---:|---:|
| Detection | YOLOv11n TensorRT INT8 | 0.4 GB | ~8 ms | 30 Hz |
| Depth | D435 aligned frames | 0.3 GB buffers | ~5 ms | 30 Hz |
| Planner | Qwen2.5-VL-3B-Instruct FP16 | ~6.5 GB | 0.8–2 s | On demand, ≤0.5 Hz |
| Critic | Moondream2 FP16 or GGUF-Q4 | 4 GB / 1.2 GB | 0.3–0.8 s | Event-driven |
| Motion / IK | `ikpy` + NumPy | 0.2 GB | <2 ms | 50 Hz stream |
| Wrist capture + dewarp | OpenCV / USB | Small | ~2–5 ms | 30 Hz |
| System / CUDA | — | ~3 GB | — | — |

Expected peak:

```text
~11–12 GB / 16 GB
```

Fallback chain if memory pressure appears:

```text
1. Critic → GGUF-Q4 via llama.cpp
2. Planner → INT8 TensorRT-LLM
3. Reduce wrist stream resolution / FPS
4. Reduce YOLO input resolution if necessary
```

---

### 9.3 Thread / Process Graph

Single Python process, headless.

```text
FSM supervisor (10 Hz)
 ├─ overhead-cam thread (30 Hz)
 │    → frame queue
 │    → YOLO + depth on demand
 │
 ├─ wrist-cam thread (30 Hz)
 │    → latest-frame buffer
 │    → optional dewarp
 │    → Critic input
 │
 ├─ planner call
 │    → on demand
 │    → hard timeout 3 s
 │    → temperature 0
 │
 ├─ motion streamer (50 Hz)
 │    → serial
 │    → bridge MCU
 │    → servos
 │    → NEVER blocks on AI
 │
 ├─ critic worker
 │    → event-driven
 │    → verdict queue
 │
 └─ logger
      → episodes.jsonl
      → frames/
      → latency HUD
```

Hard rule:

```text
The 50 Hz motion thread must never block on VLM inference, detection, logging, or network operations.
```

---

## 10. FSM States

Locked FSM pipeline:

```text
IDLE
→ PERCEIVE
→ PLAN
→ APPROACH
→ GRASP
→ VERIFY_GRASP
→ LIFT
→ VERIFY_HOLD
→ TRANSPORT
→ PLACE
→ VERIFY_PLACE
→ DONE
```

Any verification state may branch to:

```text
RECOVER(k)
```

where:

```text
k ≤ 2
```

After maximum recovery attempts:

```text
ABORT
```

Abort behavior:

- Move to safe pose if safe.
- Torque off if required.
- Save logs.
- Produce human-readable failure report.

---

## 11. AI Model Stack

### 11.1 Primary Planner

| Attribute | Value |
|---|---|
| Model | Qwen2.5-VL-3B-Instruct |
| Precision | FP16 primary; INT8 TensorRT-LLM optional |
| Role | Convert language + overhead image into structured JSON skill chain |
| Temperature | 0 |
| Timeout | 3 s hard timeout |
| Output | Strict JSON |

Planner output contract:

```json
{
  "skill_chain": [
    {"skill": "approach", "target_id": 2, "standoff_mm": 40},
    {"skill": "grasp", "closure": 80},
    {"skill": "place", "target_id": 5}
  ],
  "notes": "..."
}
```

### 11.2 Critic

| Attribute | Value |
|---|---|
| Model | Moondream2 |
| Optional | LoRA fine-tune |
| Role | Visual grasp verification |
| Input | Dewarped Trinet wrist frame |
| Output | One word |
| Classes | `GRASPED`, `SLIPPING`, `EMPTY`, `OCCLUDED` |
| Timeout | Explicit timeout required |
| Execution | Event-driven, asynchronous |

Critic prompt contract:

```text
You are a robot grasp inspector.
Image = wrist camera after gripper closed.
Answer ONE word: GRASPED | SLIPPING | EMPTY | OCCLUDED
```

### 11.3 Detection

| Attribute | Value |
|---|---|
| Model | YOLOv11n |
| Precision | TensorRT INT8 |
| Rate | 30 Hz |
| Role | Object proposals for planner and perception |

The VLM should not do raw detection.

### 11.4 Motion Stack

| Attribute | Value |
|---|---|
| IK | `ikpy` numerical IK |
| Trajectories | Trapezoidal joint-space profiles |
| Stream rate | 50 Hz |
| Safety | Joint limits, speed caps, bridge clamps |

The VLM must not output raw joint angles directly to the arm.

---

## 12. Frames and Calibration

### Frames

```text
base
cam_over
tcp
wrist_cam
```

### Overhead camera

```text
cam_over is eye-to-hand.
It is metric-calibrated.
It is used for control.
```

### Wrist camera

```text
wrist_cam is semantic-only.
It is not used for metric control.
It is used only for Critic verification and logging.
```

### Hand-eye calibration

```text
12 tip-touch points on grid/ArUco
→ rigid solve
→ base_T_cam
```

Acceptance:

```text
TCP reaches 10 test marks within <8 mm
```

### Trinet calibration

Required only for:

- dewarping,
- overlay graphics,
- dataset quality.

Not required for closed-loop Cartesian control.

---

## 13. Failure Taxonomy and Recovery Policy

| Failure Class | Detection Signal | Recovery Policy | Max Attempts |
|---|---|---|---:|
| `SLIPPING` | Critic = SLIPPING after close/lift | +15% closure → wrist re-orient 30° → lateral offset 10 mm | 2 |
| `EMPTY` | Critic = EMPTY after lift | Re-approach with −5 mm standoff → re-grasp | 2 |
| `MISS` | Critic = EMPTY after close | Re-plan from PERCEIVE | 2 |
| `JAMMED` | Load/position-error spike or motion timeout | Stop → torque off → human report | 0 |
| `OCCLUDED` | Critic = OCCLUDED | Re-position wrist cam pose → re-query once | 1 |
| `WRONG_OBJECT` | Planner re-check vs detected class | Re-plan | 1 |

Optional paper bonus:

```text
Fuse Critic visual verdict with servo load / position-error proprioception.
Report agreement statistics.
```

---

## 14. Locked Trial Object Set

Objects should be lightweight, safe, and chosen to induce visible failures.

Recommended maximum object mass:

```text
<150 g
```

### Hard objects

| Object | Purpose | Expected Failure |
|---|---|---|
| Clear acrylic block | Slippery, reflective, difficult for depth | `SLIPPING`, noisy depth |
| Smooth wooden / PLA block | Low-friction rigid object | `SLIPPING` |
| Distractor object | Wrong-object test | `WRONG_OBJECT` |

### Soft / deformable objects

| Object | Purpose | Expected Failure |
|---|---|---|
| Kitchen sponge or foam block | Deforms under gripper | `EMPTY`, `SLIPPING` |
| Empty paper cup | Crushes if closure too high | `JAMMED`, crush failure |
| Coiled cable | Occludes wrist camera | `OCCLUDED` |

### Demo narrative

Recommended stage sequence:

1. **Baseline success**  
   Pick and place a standard textured block.

2. **Induced slip**  
   Use clear acrylic block.  
   Arm grasps, object slips, Critic detects `SLIPPING`, arm recovers.

3. **Occlusion / edge case**  
   Use coiled cable or sponge.  
   Critic detects `OCCLUDED` or `EMPTY`, FSM responds.

---

## 15. Data, Logging, and Evaluation

### Log format

```text
episodes.jsonl
```

Per tick / decision, log:

```text
t
fsm_state
joints
target
verdict
verdict_latency_ms
planner_latency_ms
vram_mb
outcome
```

Save frames at decision points:

```text
frames/
```

### Trial matrix

```text
N = 30 trials per condition
```

Conditions:

```text
Critic OFF = blind retry baseline
Critic ON  = recovery enabled
```

Variability:

```text
3 object types × 2 placements
```

Ground truth:

```text
Human-labeled per trial
```

### Metrics

- Task success rate.
- First-attempt grasp success.
- Mean recovery attempts.
- Mean added latency per verification.
- Critic precision / recall / F1.
- Peak VRAM.
- Tokens per second.
- Edge latency vs simulated cloud RTT.

### Ablations

```text
Zero-shot Moondream2
vs
LoRA-fine-tuned Critic
```

Held-out set:

```text
100 images
```

---

## 16. Critic Dataset Plan

Target dataset size:

```text
300–600 wrist images
```

Classes:

```text
GRASPED
SLIPPING
EMPTY
OCCLUDED
```

Sources:

- Robot prep runs.
- Hand-held Trinet captures.
- Phone-photo augmentation if needed.
- Different lighting conditions.
- Different object poses.

Fine-tuning:

```text
LoRA in Colab
```

Deployment:

```text
Adapter weights deployed to Jetson Orin
```

---

## 17. Milestones

| Phase | Content | Exit Test |
|---|---|---|
| Prep D1–2 | Environment, cameras streaming, ROT3U checklist | `tegrastats` MAXN, 30 FPS both cameras, checklist complete |
| Prep D3 | Bridge firmware + protocol | Sine sweep smooth, link-loss hold works |
| Prep D4 | Calibration + metric perception | `<8 mm` point error |
| Prep D5 | Planner + skills + open-loop picks | ≥3/5 picks |
| Prep D6 | Critic zero-shot + FSM, scope freeze | 4/5 induced slips recovered |
| Prep D7 | Critic dataset + LoRA + logging | Fine-tuned > zero-shot |
| Prep D8 | Buffer, stress test, demo polish | 10 stable full-pipeline trials |
| Hackathon | Smoke tests, trial matrix, video, numbers | Paper tables populated |

---

## 18. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| ROT3U servo protocol unknown | ArmInterface abstraction, candidate drivers, checklist first |
| Orin OOM | `tegrastats`, fallback to GGUF-Q4 Critic, INT8 planner |
| Trinet USB bandwidth issue | Lower resolution/FPS, use separate USB port, avoid shared hub with RealSense if unstable |
| Fisheye image confuses VLM | Dewarp before Critic, validate on held-out frames |
| Wrist camera cable snag | Short flexible USB cable, strain relief, center-channel routing |
| Venue lighting shift | Lock exposure/WB, recalibrate YOLO threshold on-site |
| Demo curse | Scope freeze D6, recorded backup video, boot smoke tests |
| 2-day hackathon window | Modular VLA only, no end-to-end training, only Critic LoRA trained |

---

## 19. Updated Bill of Materials

| Category | Item | Notes |
|---|---|---|
| Robot | ROT3U 6-DOF arm | Primary arm |
| Compute | Jetson Orin 16GB | Edge inference |
| Overhead camera | Intel RealSense D435 | USB 3.0 |
| Wrist camera | Panoculonlabs Trinet | USB live stream |
| Wrist camera cable | Short flexible USB data cable | Strain relief required |
| Bridge MCU | ESP32-S3 or arm native MCU | USB serial to Jetson |
| ESP32 cable | USB-A to USB-C / Micro-USB data cable | Not charge-only |
| RealSense cable | USB 3.0 cable | Use blue port |
| Workspace | 40×40 cm rigid mat | Clamp to table |
| Calibration | ArUco / checkerboard / grid | Fixed to mat |
| Overhead mount | Tripod or gantry | ~70 cm height |
| Judge display | Laptop | Via capture card |
| Display capture | USB HDMI capture card | Preferred |
| Servo PSU | Confirmed voltage, ≥6–10 A | Separate from Jetson supply |
| E-stop | Rocker switch / button | On positive servo rail |
| Fuse | Inline fuse | Servo rail |
| Capacitor | Bulk capacitor | Across servo rail |
| Grounding | Star ground block | Single point |
| Tools | Multimeter, calipers, hex keys, clamps, tape | Hackathon kit |
| Objects | Hard + soft trial objects | Under 150 g each |

---

## 20. Open Questions to Confirm Before Coding

These do not change the locked architecture, but they must be physically confirmed.

1. ROT3U servo bus type: serial Feetech-style or PWM?
2. Servo baud rate and IDs.
3. Exact servo PSU voltage and current.
4. Official URDF/CAD availability, or link lengths must be measured.
5. Whether servo position/load feedback is readable.
6. Gripper stroke and closure-to-angle mapping.
7. Trinet live-stream mode: UVC webcam or vendor SDK.
8. Trinet resolution/FPS that works reliably on Jetson.
9. Trinet fisheye calibration method and dewarp parameters.
10. USB port plan: RealSense on USB 3.0, Trinet and ESP32 on stable separate ports.
11. Final tripod height and tilt at venue.
12. Final wrist mount geometry for Trinet.

---

## 21. Instructions for Assisting LLMs

When assisting with this project, conform to the following:

1. Assume **edge-only inference**.
2. Assume **modular VLA**, not end-to-end policy.
3. Assume **async Critic**.
4. Assume **temperature 0** for all VLM calls.
5. Use **mm/deg at protocol boundaries**.
6. Use **Python 3.10+**, type hints, and dataclass configs.
7. Do **not** introduce ROS 2 during the hackathon.
8. Use **threads + queues**.
9. All VLM calls must have explicit timeouts.
10. The 50 Hz motion thread must never block on AI inference.
11. Never suggest powering servos from MCU pins.
12. Never assume a 5 V servo rail without confirmation.
13. Respect all safety rules.
14. If hardware details are unknown, request the specific open question instead of guessing.
15. Prefer minimal diffs and testable increments.
16. Every feature proposal must state its latency and VRAM cost.
17. Deliverable priority:

```text
1. Live demo
2. Paper-quality logs
3. Code elegance
```

---

## 22. Final Locked One-Line Architecture

```text
Qwen2.5-VL-3B planner + deterministic IK motion + RealSense D435 overhead perception + Trinet wrist Critic + Moondream2 verification + parameterized recovery FSM, all running locally on Jetson Orin 16GB.
```
```