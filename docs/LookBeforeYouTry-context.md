Save as:

```markdown
# PROJECT CONTEXT / BIBLE — “Look Before You Retry” (LBYR)

**Version:** 1.6  
**Status:** Locked hackathon context  
**Hardware revision:** ROT3U 6-DOF PWM arm (no feedback) + XIAO ESP32-C6 bridge + Jetson Orin NX 16GB (reComputer J401) + RealSense D435 + Panoculonlabs Trinet wrist camera  
**Supersedes:** Version 1.2 where conflicting (v1.3 changes: compute platform, browser dashboard instead of HDMI capture, PWM arm + XIAO C6 text-protocol bridge, 6.0 V servo rail, JAMMED policy, OQ-1/2/3/5 closed). v1.4 (2026-10-10): measured model cost in 9.2; planner policy (rule-based planner by default, Qwen "language mode" for the demo intro only, memory guard), see 9.2 and 11.1. v1.5 (2026-10-10): Critic = Moondream2 BF16 on a 378×378 crop of the dewarped frame (M-1 closed); language mode served by Qwen3-1.7B text (thinking off) instead of Qwen2.5-VL; see 9.2, 11.1, 11.2. v1.6 (2026-10-10): motion architecture PP-1 — the bridge firmware is the trajectory generator (one `m` per segment, `DONE` on completion, no 50 Hz streaming); taught-pose mode is the demo path; see 9.3, 11.4.  

> This document is the updated single source of truth for the hackathon build.  
> Where this document conflicts with older notes, this document wins.  
> The core research and architecture remain unchanged: edge-native, modular VLA, async visual Critic, no cloud at runtime, no ROS 2 during hackathon.

---

## 0. Locked Decisions Summary

| # | Decision | Locked Choice | Notes |
|---:|---|---|---|
| 1 | Robot arm | **ROT3U 6-DOF desktop arm** | Replaces any earlier MK3 / 3D-printed arm plan. |
| 2 | Main compute | **Jetson Orin NX 16GB on Seeed reComputer J401** | All inference local. JetPack 6.2 / L4T 36.4.3, `nvpmodel -m 4` (40 W) + `jetson_clocks`, headless. |
| 3 | Overhead camera | **Intel RealSense D435** | Overhead RGB-D for detection + metric 3D. |
| 4 | Overhead camera mount height | **~70 cm above 40×40 cm mat** | Tilt ~50° downward. Lock exposure/white balance. |
| 5 | Wrist / gripper camera | **Panoculonlabs Trinet camera** | Replaces Raspberry Pi Camera v2 / IMX219 as primary wrist Critic camera. |
| 6 | Wrist camera interface | **USB live stream to Jetson Orin** | Do not use Jetson CSI ribbon cable for primary wrist camera. |
| 7 | Wrist camera image handling | **Dewarp before VLM** | Trinet ultra-wide/fisheye image must be rectified before Moondream2. |
| 8 | Wrist camera role | **Semantic-only Critic** | Not used for metric control or hand-eye calibration. |
| 9 | Bridge-to-Jetson link | **USB serial / USB CDC, 115200, text protocol** | XIAO ESP32-C6 bridge. Use a good USB-A to USB-C data cable. |
| 10 | Judge-facing display | **Browser dashboard (FastAPI/WebSocket, separate process)** | Bound to 0.0.0.0; reached over Tailscale or LAN. Replaces the HDMI capture card. |
| 11 | Workspace | **40×40 cm rigid mat** | Clamped to table. Contains ArUco / calibration grid and object start zones. |
| 12 | Object set | **Hard + soft failure-inducing objects** | See Section 7. All objects should be under ~150 g. |
| 13 | Architecture | **Modular VLA + deterministic motion stack** | No end-to-end policy during hackathon. |
| 14 | Runtime inference | **Edge-only** | No cloud APIs at runtime. |
| 15 | Software style | **Python 3.10+, threads + queues, no ROS 2** | The motion thread must never block on AI (v1.6: 20 Hz segment dispatcher + status poller). |

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
| DOF | 6, base to gripper: 1 base yaw, 2 shoulder, 3 elbow, 4 wrist pitch, 5 wrist roll, 6 gripper | Confirmed (v1.3) |
| Class | Desktop arm | Locked |
| Reach | Approximately 300–450 mm | Confirm |
| Payload | Approximately 250–500 g | Confirm |
| Servo type | **PWM servos**: joints 1-4 Feetech 25 kg, joints 5-6 MG996R | Confirmed (v1.3) |
| Servo voltage | **6.0 V** rail (MG996R max ~7.2 V). **Never assume 5 V.** | Confirmed (v1.3) |
| Feedback | **None.** No position, load or temperature readback. "Position" = last commanded pulse | Confirmed (v1.3) |
| Gripper | Parallel jaw, position-controlled | Confirm stroke and closure mapping |

Confirmed in v1.3: servo bus type (PWM), joint order and roles, servo voltage (6.0 V),
no feedback. Servo IDs and baud rate do not apply to PWM servos.

Still to be confirmed before relying on them:

1. Joint limits in degrees.
2. Link lengths or official URDF.
3. Payload at full extension.
4. Gripper stroke and closure-to-angle mapping.
5. Servo PSU current rating.
6. Trinet mounting point and cable routing path.

---

## 4. Compute Stack

| Item | Role | Interface | Notes |
|---|---|---|---|
| Jetson Orin NX 16GB on Seeed reComputer J401 | All AI inference, perception, FSM, logging | Headless, JetPack 6.2 / L4T 36.4.3 | Power mode `sudo nvpmodel -m 4` (40 W, the maximum on this image) + `jetson_clocks` applied at boot via systemd |
| Laptop | SSH, logs, judge-facing dashboard in a browser | GbE / Wi-Fi / Tailscale | Primary operator machine |
| Browser dashboard | Overlay UI, latency HUD, verdict, FSM state, detection boxes | HTTP + WebSocket (FastAPI) | Separate process, bound to `0.0.0.0` |
| External HDMI monitor | Optional backup | HDMI | Not required |

### Locked display decision

Primary judge-facing display (v1.3, replaces the HDMI capture card):

```text
Jetson Orin NX
→ dashboard process (FastAPI + WebSocket, 0.0.0.0)
→ Tailscale or LAN
→ laptop browser
```

The browser shows the overlay UI, latency HUD, wrist camera verdict, FSM state, and detection boxes.

The dashboard is a **separate process** from the control loop. The 50 Hz motion thread, the
FSM and the Critic never wait on it: it reads logged or published state only, and a dead or
slow browser must not affect the arm. Prefer a wired LAN at the venue if Wi-Fi is unreliable.

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

## 6. Bridge MCU Connection (XIAO ESP32-C6)

### Locked connection

```text
Jetson Orin USB-A host
→ USB-A to USB-C data cable
→ XIAO ESP32-C6 USB port
```

The XIAO ESP32-C6 appears as a USB serial device.

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

The XIAO ESP32-C6 has a USB-C port: use a USB-A to USB-C data cable. Bring spare cables.

### Communication protocol

Newline-terminated **text** over USB serial, **115200 baud** (v1.3; replaces the JSON
protocol of v1.2). Authoritative reference: `firmware/xiao_c6_bridge/Code.ino`.
Angles are degrees relative to each joint's saved zero; joints are numbered 1-6.

| Command | Meaning |
|---|---|
| `J` | Test joint J: swing ±30°, back to zero |
| `g J DEG` | Move joint J to DEG |
| `m D1 .. D6` | Move all joints together, arriving at the same time |
| `t J D` | Nudge joint J by D µs from its target; `z J` then stores it |
| `z [J]` | Store the current target as zero (saved to flash) |
| `set J US` | Set joint J's zero to an exact pulse, move there, save |
| `save` | Write all zeros to flash |
| `home` | Move all joints to their zeros (re-attaches released joints) |
| `x` / `stop` | Brake smoothly and hold |
| `speed V A` | Max speed (deg/s) and accel (deg/s²), capped in firmware |
| `off [J]` | Release PWM (arm may fall; support it first) |
| `list` | Print state: `S,<joint>,<zero_us>,<pos_us>,<attached>,<name>` and `V,<vmax>,<amax>` |
| `help` | Print commands |

Replies are `OK ...` or `ERR ...` lines. There is no `state` query with load or temperature.

The host-side `lbyr/arm/bridge_proto.py` JSON protocol is **not** what the firmware speaks.
`SerialArm` must speak this text protocol for now (not yet implemented; see
`docs/OPEN_QUESTIONS.md`).

### Bridge responsibilities

The firmware implements: PWM generation (direct LEDC, one pin per joint), pulse clamps
(500-2500 µs, to be tightened per joint), a speed- and acceleration-limited planner,
backlash approach from below, per-joint speed scaling (joints 5-6 slower), per-joint zero
stored in flash, staggered home-on-boot, smooth `stop`.

Not implemented in the firmware yet: joint limits in degrees, a link-loss watchdog, a
per-tick delta clamp. PWM servos hold their last pulse, so a link loss means "hold".

### Actuation

```text
XIAO ESP32-C6 → direct LEDC PWM (D0, D1, D3, D4, D5, D8) → PWM servos
```

No PCA9685 and no serial-bus servos. The host software must still not depend on this
low-level detail beyond the command set above.

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

Servos must be powered from a separate confirmed servo PSU (6.0 V, see Section 8).

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
Laptop beside table (browser dashboard over LAN / Tailscale)
PSU + E-stop within operator reach
```

---

## 8. Power and Safety

### Locked power rules

1. Separate servo PSU.
2. Servo rail is **6.0 V** (MG996R max ~7.2 V). Confirm with a multimeter before powering.
3. Do not assume 5 V.
4. Use inline fuse on positive servo rail.
5. Use rocker switch / E-stop on positive servo rail.
6. Use bulk capacitor across servo rail.
7. Use single star ground at PSU negative terminal.
8. Do not add an extra ground wire from servo PSU to Orin.
9. USB already ties Jetson and ESP32 grounds.
10. The servo supply negative must be tied to XIAO GND (single star ground).
11. **Boot rule:** the first PWM pulse at boot goes to each joint's saved zero, and there is
    no feedback to smooth it. The arm must rest in its zero pose **before** the servo rail is
    powered.

### Grounding concept

```text
Servo PSU −V
Servo GND
XIAO / logic GND
        ↓
single star ground
```

Jetson ground is connected through the USB link to the XIAO. Do not create a second ground path to the Jetson.

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
9. With no feedback, a stalled or blocked servo cannot be detected in software: the
   human E-stop is the only jam protection (see Section 13).

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

Original estimates (v1.2/v1.3). The measured numbers below win where they differ.

| Component | Model / Precision | Approx VRAM | Approx Latency | Rate |
|---|---|---:|---:|---:|
| Detection | YOLOv11n TensorRT INT8 | 0.4 GB | ~8 ms | 30 Hz |
| Depth | D435 aligned frames | 0.3 GB buffers | ~5 ms | 30 Hz |
| Planner | Qwen2.5-VL-3B-Instruct FP16 | ~6.5 GB | 0.8–2 s | On demand, ≤0.5 Hz |
| Critic | Moondream2 FP16 or GGUF-Q4 | 4 GB / 1.2 GB | 0.3–0.8 s | Event-driven |
| Motion / IK | `ikpy` + NumPy (taught poses in v1.6) | 0.2 GB | <2 ms | one `m` per segment, 20 Hz status poll |
| Wrist capture + dewarp | OpenCV / USB | Small | ~2–5 ms | 30 Hz |
| System / CUDA | — | ~3 GB | — | — |

Expected peak:

```text
~11–12 GB / 16 GB
```

**Measured 2026-10-10, 40 W, synthetic images.** Orin NX 16 GB, `nvpmodel` 40 W (mode 4),
JetPack 6.2, torch 2.11, transformers 5.19, HF FP16 (no TensorRT). Source:
`tools/bench_models.py` (`data/bench/`, 5 warm calls). Memory = `torch.cuda.memory_allocated`;
the Orin has unified memory, so the real limit is system RAM (tegrastats).

| Component | Model / Precision | Memory (torch) | Latency (p50) | vs estimate |
|---|---|---:|---:|---|
| Critic | Moondream2 FP16, 640×480 frame | 4.1 GB | 1.35 s | ~4 GB OK; latency ~2× the 0.3–0.8 s estimate (fits the 2 s Critic timeout) |
| Planner | Qwen2.5-VL-3B-Instruct FP16, full JSON contract (98 tokens) | 7.2 GB | 11.2 s (8.7 tok/s) | +0.7 GB; ~6–14× the 0.8–2 s estimate; **never meets the 3 s hard timeout** |
| Both loaded | Critic + Qwen | 11.3 GB | — | system RAM peak **14.8 / 16 GB** (~0.8 GB free) |

**Measured 2026-10-10, 40 W + jetson_clocks, synthetic images (current, v1.5).** Same board
and software; `jetson-clocks.service` enabled (CPU pinned 1.5 GHz, GPU 1173 MHz, EMC
3199 MHz). Source: `tools/bench_models.py --planner-models qwen3-1.7b --combined`
(`data/bench/bench-20261010-224019.json`, 10 calls each). Planner latency = full generation
of the compact schema (~20 tokens) for 10 fixed language commands against a fixed 6-object
detection list.

| Component | Model / Precision | Memory (torch) | Latency p50 / p95 | Notes |
|---|---|---:|---:|---|
| Critic (runtime) | Moondream2 **BF16**, 378×378 crop of the dewarped frame | 4.1 GB peak (3.7 GB weights) | **0.785 s** / 0.789 s | accuracy NOT measured yet (no labelled set) |
| Language planner (default) | **Qwen3-1.7B** BF16, text-only, thinking off | 3.2 GB | **2.14 s** / 2.18 s | 9.4 tok/s, 20 tokens; valid 9/10, correct 9/10 (the miss: hallucinated zone id 5, rejected by the strict parser); 10/10 within 8 s; load+warm-up 7.7 s |
| Critic + Qwen3-1.7B loaded together | — | 7.2 GB (8.1 GB peak) | Critic 0.78 s, planner 2.13 s | system RAM 10.80 / 15.29 GB used after load, **4.49 GB free** → memory guard **PASS** |

Before jetson_clocks (same day, 40 W, clocks not pinned): Critic BF16 crop p50 0.849 s
(full 640×480 FP16 frame: 1.35 s); Qwen3-1.7B p50 2.37 s / 8.4 tok/s, load+warm-up 10.7 s;
combined Critic 0.91 s, planner 2.32 s, 4.5 GB free. Qwen3-0.6B (not re-run): 2.82 s,
1.1 GB, valid 5/10, correct 2/10 (string ids, wrong zones): not usable.

Decode speed note: pinning the clocks (GPU to 1173 MHz, about 3× its unpinned clock per the
operator; the unpinned GPU frequency was not logged by the bench) gave only ~10% (Qwen3-1.7B
2.37 → 2.14 s, 8.4 → 9.4 tok/s; Critic 0.849 → 0.785 s). Qwen3-0.6B, Qwen3-1.7B and
Qwen2.5-VL-3B also all decode at ~8.5 tok/s. So decoding is limited by per-token framework
overhead (HF `generate`, Python + kernel launches), not compute. Post-hackathon option: a
llama.cpp or TensorRT-LLM backend for the language planner. Not for now.

Planner policy (v1.4, follows from the numbers above):

```text
1. Default planner.backend = rules (RuleBasedPlanner: class-name match against the
   instruction; microseconds, no VRAM). Qwen is NOT loaded unless backend = qwen.
2. In-loop planner calls keep the 3 s hard timeout (default for every in-loop call).
   With backend = qwen, 3 timeouts in a row switch to rules for the rest of the run.
3. "Language mode" (demo intro only, never inside the FSM loop): text-only prompt
   (task + detections as compact JSON, no image), compact output schema (11.1),
   max_new_tokens 64 (~20 tokens measured), own timeout planner.qwen_timeout_s = 8 s.
   v1.5: served by Qwen3-1.7B (planner.language_backend = qwen3_text, default; weights
   ~/models/qwen3-1.7b; enable_thinking=False; greedy). Qwen2.5-VL text-only stays
   selectable (language_backend = qwen_vl) but is not loaded by default anywhere.
   The language planner is only loaded with planner.language_mode = true.
4. Every planner output (rules, Qwen in-loop, Qwen language mode) goes through the same
   strict skill-chain parser; it is the only path to skills.
5. Every plan is logged as a "reasoning_trace" episode event (prompt, raw output, parsed
   chain) for the dashboard.
6. Memory guard: system RAM is logged at startup and after each model load. With
   backend = qwen or language_mode, if free RAM after loading all models is below
   planner.qwen_min_free_ram_mb (1.5 GB) the episode loop does not start.
   Critic + Qwen2.5-VL (~0.8 GB free): refused. Critic + Qwen3-1.7B (4.5 GB free): passes.
```

Critic defaults (M-1 closed, v1.5): BF16 (FP16 still selectable, `critic.dtype`), input =
dewarped wrist frame → 378×378 crop (`critic.crop_xywh`, default centred; must be re-centred
on the gripper bite zone once the Trinet is mounted, OQ-12). Language-mode planner: Qwen3-1.7B
(M-2 closed 2026-10-10).

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
 ├─ motion controller (20 Hz, v1.6; was the 50 Hz streamer)
 │    → ONE `m` per motion segment (latest-value buffer from the FSM)
 │    → `q` status poll, waits for `DONE` (no reply ever blocks it)
 │    → bridge MCU = trajectory generator → servos
 │    → no DONE within predicted × 1.5 + 0.5 s → LINK → ABORT
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
The motion thread must never block on VLM inference, detection, logging, or network operations.
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
| Timeout | 3 s hard timeout (every in-loop call) |
| Output | Strict JSON |
| Default backend (v1.4) | Rule-based planner; Qwen only with `planner.backend = qwen` (see 9.2) |
| Language mode (v1.4) | Demo intro only: text-only, compact schema, ≤64 tokens, `planner.qwen_timeout_s` = 8 s |
| Language-mode model (v1.5) | Qwen3-1.7B text (BF16, `enable_thinking=False`, greedy), `planner.language_backend = qwen3_text`; Qwen2.5-VL text-only selectable (`qwen_vl`). Measured 2.14 s p50 with jetson_clocks, 9/10 correct (9.2) |

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

Language-mode compact contract (v1.4; no `notes`, ~19 tokens). It is expanded to the
contract above and validated by the same strict parser:

```json
{"a": 2, "s": 40, "g": 80, "p": 5}
```

`a` = approach `target_id`, `s` = `standoff_mm`, `g` = grasp `closure` (%), `p` = place `target_id`.

### 11.2 Critic

| Attribute | Value |
|---|---|
| Model | Moondream2 |
| Optional | LoRA fine-tune |
| Role | Visual grasp verification |
| Input | Dewarped Trinet wrist frame → 378×378 crop (v1.5; `critic.crop_xywh`, default centred, re-centre on the gripper bite zone after mounting) |
| Precision (v1.5) | BF16 default; FP16 selectable (`critic.dtype`) |
| Output | One word |
| Classes | `GRASPED`, `SLIPPING`, `EMPTY`, `OCCLUDED` |
| Timeout | Explicit timeout required (`critic.timeout_s`, 2 s; measured p50 0.785 s with jetson_clocks) |
| Unparseable answer (v1.5) | `INVALID` → re-query once with a fresh frame, then treated as `OCCLUDED`; logged as `critic_invalid_retry` |
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
| IK | `ikpy` numerical IK (`motion.mode = ik`; needs OQ-4 + hand-eye) |
| Taught poses (v1.6, demo path) | `motion.mode = taught` (default): skills map to joint poses taught with `tools/arm_gui.py` (`data/calib/poses.yaml`); object → zone by detection centroid in a zone polygon |
| Trajectories (v1.6) | Generated by the bridge firmware (speed/accel limited, synced arrival, backlash approach from below). Host sends ONE `m` per segment and waits for `DONE`; a Python port of the firmware planner predicts each duration (DONE timeout = predicted × 1.5 + 0.5 s → LINK) |
| Stream rate | None (v1.6). Status poll `q` at 20 Hz |
| Safety | Joint limits, speed caps (firmware `speed`), firmware pulse clamps; zeros checked against `data/calib/joints.yaml` before any motion |

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
| `JAMMED` | **Operator stop only** (no load/position feedback on PWM servos) | Stop → torque off → human report | 0 |
| `OCCLUDED` | Critic = OCCLUDED | Re-position wrist cam pose → re-query once | 1 |
| `WRONG_OBJECT` | Planner re-check vs detected class | Re-plan | 1 |

`JAMMED` (v1.3): the PWM servos give no load or position feedback, so software cannot
detect a jam by itself. It is triggered by the operator stop only, unless an INA226 current
sensor on the servo rail is added later (then a current spike can raise `JAMMED`).

Optional paper bonus (only if an INA226 current sensor is added):

```text
Fuse Critic visual verdict with servo-rail current proprioception.
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
| Prep D1–2 | Environment, cameras streaming, ROT3U checklist | `tegrastats` at `nvpmodel -m 4` + `jetson_clocks`, 30 FPS both cameras, checklist complete |
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
| Servos have no feedback | Conservative speed/accel caps, E-stop in reach, rest in zero pose before power-on, optional INA226 later |
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
| Compute | Jetson Orin NX 16GB on Seeed reComputer J401 | Edge inference, JetPack 6.2 |
| Overhead camera | Intel RealSense D435 | USB 3.0 |
| Wrist camera | Panoculonlabs Trinet | USB live stream |
| Wrist camera cable | Short flexible USB data cable | Strain relief required |
| Bridge MCU | Seeed XIAO ESP32-C6 | USB serial to Jetson, direct PWM |
| Bridge cable | USB-A to USB-C data cable | Not charge-only |
| RealSense cable | USB 3.0 cable | Use blue port |
| Workspace | 40×40 cm rigid mat | Clamp to table |
| Calibration | ArUco / checkerboard / grid | Fixed to mat |
| Overhead mount | Tripod or gantry | ~70 cm height |
| Judge display | Laptop | Browser dashboard over Tailscale / LAN |
| Servo PSU | 6.0 V, ≥6–10 A | Separate from Jetson supply |
| E-stop | Rocker switch / button | On positive servo rail |
| Fuse | Inline fuse | Servo rail |
| Capacitor | Bulk capacitor | Across servo rail |
| Grounding | Star ground block | Single point |
| Tools | Multimeter, calipers, hex keys, clamps, tape | Hackathon kit |
| Objects | Hard + soft trial objects | Under 150 g each |

---

## 20. Open Questions to Confirm Before Coding

These do not change the locked architecture, but they must be physically confirmed.

1. ~~ROT3U servo bus type~~ **CLOSED (v1.3): PWM.**
2. ~~Servo baud rate and IDs~~ **CLOSED (v1.3): not applicable (PWM).** Joint order is 1 base yaw, 2 shoulder, 3 elbow, 4 wrist pitch, 5 wrist roll, 6 gripper.
3. ~~Exact servo PSU voltage~~ **CLOSED (v1.3): 6.0 V** (MG996R max ~7.2 V). PSU current rating not recorded.
4. Official URDF/CAD availability, or link lengths must be measured.
5. ~~Whether servo position/load feedback is readable~~ **CLOSED (v1.3): no feedback.**
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
10. The motion thread must never block on AI inference.
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