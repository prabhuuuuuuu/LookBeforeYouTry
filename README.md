# Look Before You Retry (LBYR)

**Edge-native visual grounding for VLA failure recovery.** A modular VLA (Qwen2.5-VL-3B
planner + deterministic IK motion stack) with an asynchronous visual Critic (Moondream2 on a
wrist-mounted Trinet camera) that verifies grasps at FSM checkpoints and picks a
parameterized recovery instead of retrying blindly. Everything runs on a Jetson Orin 16GB.
There are no cloud calls.

Awign Hackathon 2026, Physical AI track. Project bible: `docs/LookBeforeYouTry-context.md`.
Repository: https://github.com/prabhuuuuuuu/LookBeforeYouTry

```
Operator instruction ("move the block from A to the place zone")
-> Qwen2.5-VL-3B picks move(from, to) among taught locations (strict JSON, never joints)
-> arm executes taught poses -> wrist camera verifies
-> Critic: GRASPED | SLIPPING | EMPTY | OCCLUDED -> re-grasp +15% (k <= 2) or abort safely
```

## Quick start (laptop, no hardware)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[laptop,dev]"
pytest
python -m lbyr.main --mock               # one episode: SLIPPING -> +15% closure -> DONE
python -m lbyr.main --mock --critic-off  # baseline condition
python -m lbyr.main --mock --script SLIPPING,SLIPPING,SLIPPING   # ends in ABORT + report
python -m lbyr.demo --dry-run --induce-slip                       # live-demo path, simulated
python -m lbyr.demo --dry-run --task "move the block from B to A" # language prompting, simulated
```

`pytest`: 3 tests (`test_config`, `test_taught`, `test_text_proto` "seeded" checks) assert
the original unverified calibration templates and fail now that `data/calib/` holds the
real, verified rig values. Everything else passes without hardware.

Logs go to `data/episodes/<session>/episodes.jsonl`, and decision-point frames go to `frames/`.

Running without `--mock` performs a hardware preflight and lists every open question that
still blocks real-hardware runs (see `docs/OPEN_QUESTIONS.md`).

## Status (2026-10-11)

| Area | State |
|---|---|
| Config, FSM + recovery, logging, MockArm (firmware planner port), MotionController (one `m` per segment + DONE, PP-1), taught-pose mode | Implemented and tested |
| Bridge firmware (XIAO ESP32-C6, text protocol: `DONE`, `q`, extended `S`) + `tools/arm_gui.py` | Running on the board. `data/calib/joints.yaml` verified against the board zeros; `poses.yaml` taught (`home`, `zone_A_*`, `place_*`, gripper open -40° / closed 0°) |
| SerialArm | Runs on the real board: no DTR/RTS toggling (it reset the ESP32-C6), `q` probe on connect, zeros check, old-firmware refusal, 1 s write timeout, `q` polled only while a move is active, LINK fault only after 3 failed polls/reads in a row or the DONE timeout |
| Live demo `python -m lbyr.demo` | Brought up on the rig (arm + D435 + Trinet + Qwen2.5-VL + Moondream2), under active testing. Language prompting (web text box / `--task`), web view, episode logs, wrist snapshots |
| Cameras | D435 overhead (exposure / WB lock, block detector) and Trinet wrist (MJPG 1080p30, raw-JPEG decode, auto-reconnect after USB drops) both open before the arm link; Trinet dewarp needs `trinet_fisheye.yaml` (OQ-9, else raw) |
| Moondream2Critic (BF16, 378x378 crop), QwenPlanner (Qwen2.5-VL-3B), Qwen3-1.7B language mode, RuleBasedPlanner, `tools/bench_models.py` | Verified on the Orin NX (`pytest -m jetson`). Critic accuracy on real wrist frames is NOT measured yet |
| Fisheye dewarp, metric geometry, Umeyama hand-eye solve | Implemented and tested |
| IkpyIK (IK motion mode) | Gated on open hardware questions (OQ-4); the demo uses taught poses |
| YOLOv11n TensorRT | Stub (TODO); the demo uses the classical BlockDetector |

Open items: location polygons (`polygon_px`) and table-relative descriptions in
`poses.yaml`; Critic accuracy on real wrist frames; Trinet fisheye calibration.

## Layout

```
lbyr/       arm/ perception/ ai/ motion/ fsm/ logging_/ config.py main.py mock_stack.py
tools/      beginner-friendly camera / calibration / labeling / serial scripts
firmware/   XIAO ESP32-C6 bridge (xiao_c6_bridge/)
tests/      pytest, runs without hardware
docs/       bible, open questions, onboarding
data/       calib/ is tracked (joints.yaml, poses.yaml); frames/, datasets/, episodes/,
            bench/, live/ are git-ignored
```

## Motion and taught poses (demo path)

The bridge firmware is the trajectory generator (PP-1). The host sends ONE `m` per motion
segment and waits for the firmware's `DONE`. While a segment is active, the
`MotionController` thread polls `q` at 20 Hz (idle: no serial traffic). A failed poll is
logged and retried; 3 failures in a row, or no DONE within predicted × 1.5 + 0.5 s, is a
LINK fault and the episode aborts. The predicted duration comes from a Python port of the
firmware planner (`lbyr/arm/fw_planner.py`); `MockArm` runs the same port, so mock timing
matches the board.

Bring-up on the real arm (done on our rig; repeat after re-flashing or re-assembly):

1. Flash `firmware/xiao_c6_bridge/Code.ino` (adds `DONE`, `q` and the extended `S` line).
2. `python tools/arm_gui.py` → Connect → "Sync sliders". Compare every `S,<j>,<zero_us>,...`
   line with `data/calib/joints.yaml`; fix if needed, then set `verified: true`.
   `SerialArm` refuses to move if the board's zeros differ from the file.
3. Teach the poses: move with the sliders, then "Save pose as..." with the names `home`,
   `zone_A_pre`, `zone_A_grasp`, `place_pre`, `place` (and `zone_B_pre` / `zone_B_grasp`
   for a location B). Use "Go to pose" to check each one.
4. Edit `data/calib/poses.yaml`: `gripper.open_deg` / `closed_deg` (slider 6 values) and the
   `locations` section (below). For `lbyr.main` also per zone `polygon_px` and `radius_mm`.
5. `python -m lbyr.main` lists anything still missing (preflight).

Taught-mode recovery: +15% closure → joint 6; wrist re-orient 30° → joint 5; lateral 10 mm
→ base yaw + atan(10 / radius_mm); −5 mm standoff → the pre-grasp pose moves along the
joint-space segment zone_X_pre → zone_X_grasp by 5 / standoff_mm (not a straight line in
Cartesian space; the grasp pose is unchanged).

## Live demo (`python -m lbyr.demo`)

```bash
python -m lbyr.demo                           # real arm + D435 + Trinet + Qwen + Moondream
python -m lbyr.demo --task "move the block from A to the place zone"   # Enter runs it
python -m lbyr.demo --induce-slip             # first grasp loose -> SLIPPING -> re-grasp +15%
python -m lbyr.demo --rehearsal               # real arm + cameras, rules planner, fake Critic
python -m lbyr.demo --planner rules           # real Critic, no Qwen
python -m lbyr.demo --critic-off              # baseline: no verification
python -m lbyr.demo --dry-run --induce-slip   # MockArm, fake cameras, fake Critic
```

Startup order: D435 and Trinet open and deliver a first frame, then the arm link, then the
operator confirms the E-stop (`y`), then Qwen2.5-VL and Moondream2 load (both stay
resident). Nothing moves until an instruction or Enter. A setup failure stops the camera
threads and the RealSense pipeline cleanly (exit code 2).

**Web view** `http://<jetson-ip>:8000`: overhead + wrist video, an instruction box with a
Run button (disabled while an episode runs; new submissions get "busy"), a large outcome
line (SUCCESS / NONE + reason / ERROR / ABORT), the planner actually used (qwen or rules
fallback) with its raw and parsed output, every Critic verdict with its latency, the
recovery steps, RAM and the event log. `GET /snapshot` saves the wrist frame + Critic crop.
Anyone who can reach the page can start motion: keep it on the lab network.

**Terminal** (between episodes, same queue as the web page): Enter = run (`--task` if
given, else A -> place), `s` = run with induced slip, `w` = wrist snapshot, `q` = quit, any
other text = an instruction.

**Language prompting.** Qwen2.5-VL-3B gets the overhead image, the instruction, the
`locations` list (name + description) and the block location from the detector, and must
answer strict JSON: `{"action":"move","from":"A","to":"place","reason":"..."}` or
`{"action":"none","reason":"..."}` (greedy, 30 s timeout). It answers "none" only when no
block is visible or the instruction names an unlisted place. Location-like names that are
not taught ("zone C") are refused before Qwen is called, listing the valid names. Planning
happens before any motion; invalid output or "none" moves nothing. A move runs from.pre →
from.grasp → close → VERIFY_GRASP → from.pre (LIFT, monitored) → VERIFY_HOLD → to.pre
(TRANSPORT, monitored) → to.grasp → open → VERIFY_PLACE → home with the usual re-grasp
recovery.

**Critic backend and grasp monitoring.** `--critic probe` is the default when
`~/models/critic_probe.pt` exists (`tools/train_critic_probe.py`; `--probe-path`). It loads
Moondream2 once and runs only the vision encoder and a logistic head on the 378x378 crop:
verdict, all 4 probabilities, ~90 ms. If the max probability is below 0.55 the verdict is
LOW_CONF, which goes through the usual re-query path. Without the file the demo falls back
to `moondream` (zero-shot text, unchanged) and shows a warning on the page.
During LIFT and TRANSPORT the Critic worker re-runs back-to-back on the latest wrist frame
while the FSM waits for DONE; the motion thread never waits on it. Two consecutive
SLIPPING/EMPTY verdicts with p >= 0.6 count as a **DROP** (`DemoConfig.drop_consecutive`,
`drop_min_prob`). The bridge's `m` pre-empts a running move (Code.ino `moveSynced` replaces
the targets), so the demo halts by sending an `m` to the current position from `q`, opens
the gripper there, and returns home. A dropped block is not re-grasped. After the release,
VERIFY_PLACE expects EMPTY; GRASPED/SLIPPING logs `FAILURE: object stuck in gripper`.
Each verdict (state, verdict, 4 probabilities, latency, crop JPEG path) is written to
`episodes.jsonl` and shown on the page. Failure lines start with `FAILURE:` (red, also in
the FAILURES panel), and the episode ends with SUCCESS / RECOVERED (n re-grasps) /
FAILED (<class> at <state>). The dry run exercises all of this with scripted per-state
verdicts (`VERDICT:p`):

```bash
python -m lbyr.demo --dry-run --script "VERIFY_GRASP=EMPTY:0.91,GRASPED:0.95"         # RECOVERED
python -m lbyr.demo --dry-run --script "TRANSPORT=GRASPED:0.9,SLIPPING:0.78,SLIPPING:0.81"  # DROP
python -m lbyr.demo --dry-run --script "VERIFY_PLACE=GRASPED:0.93"                    # stuck
```

Locations (`data/calib/poses.yaml`):

```yaml
locations:
  A:     {description: "left side of the table", pre: zone_A_pre, grasp: zone_A_grasp}
  place: {description: "right side of the table", pre: place_pre, grasp: place}
  # optional per location: polygon_px: [[u, v], ...]  (overhead pixels, >= 3 points):
  # outlined and labelled on Qwen's image, and used to report where the block is
```

Write descriptions as the operator sees the table; printed letters on the table also
help Qwen. Without polygons the block location is reported as "unknown".

**Robustness.** The Trinet can drop off USB and re-enumerate (up to ~20 s): startup waits
up to 45 s; mid-run the Critic waits 5 s for a fresh frame, else the verdict is OCCLUDED
(re-queried once, then a safe retreat home). Requirements: `joints.yaml` with
`verified: true`; taught poses with `gripper.open_deg/closed_deg` (closed = on the block;
re-grasps squeeze 15% further, never past 0° = fully shut). The bridge is auto-detected
(`--port`); the wrist camera is the first non-D435 `/dev/video*` (`--wrist-dev`, prefer a
`/dev/v4l/by-id/...` path). Logs (`data/episodes/demo-<stamp>/episodes.jsonl`) record the
instruction, Qwen's prompt and raw output, the parsed action, every verdict and the outcome.

## Jetson install

JetPack provides CUDA-enabled OpenCV, torch and pyrealsense2 (the last one built from
source). Do **not** pip-install them on the Jetson.

```bash
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install -U pip setuptools wheel "packaging>=24.2"
export PIP_CONSTRAINT=constraints-jetson.txt   # numpy<2: NumPy 2 breaks the JetPack cv2
pip install -e ".[jetson,dev]"
```

### Enable the jetson_clocks systemd service

All benchmark numbers below assume 40 W mode (`sudo nvpmodel -m 4`) plus pinned clocks
(CPU 1.5 GHz, GPU 1173 MHz, EMC 3199 MHz). Make `jetson_clocks` run at every boot:

```ini
# /etc/systemd/system/jetson-clocks.service
[Unit]
Description=Max Jetson clocks
After=nvpmodel.service
[Service]
Type=oneshot
ExecStart=/usr/bin/jetson_clocks
[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now jetson-clocks.service
sudo jetson_clocks --show          # check CPU / GPU / EMC frequencies
```

### Known Jetson pitfalls

- **numpy<2**: the system OpenCV is built against NumPy 1.x (`_ARRAY_API not found`).
  Covered by the `[jetson]` extra and `constraints-jetson.txt`.
- **pillow>=10**: with `--system-site-packages`, Ubuntu's Pillow 9.0.1 is too old for
  `transformers` (needs `PIL.Image.Resampling`, Pillow >= 9.1). Covered by the `[jetson]` extra.
- **torch**: install the Jetson wheel from the jetson-ai-lab `jp6/cu126` index, never the
  PyPI wheel.
- **cuDSS**: install via the tegra deb (`cudss-local-tegra-repo-ubuntu2204`, package
  `cudss-cuda-12`), then run `sudo ldconfig`.
- **System SciPy**: Ubuntu's SciPy (visible through `--system-site-packages`) expects
  `numpy<1.25` and warns ("A NumPy version >=1.17.3 and <1.25.0 is required") with the
  NumPy 1.26 we use. Avoid scikit-learn / SciPy in the Jetson runtime path; use plain
  torch / NumPy (e.g. the Critic metrics in `tools/bench_models.py` are plain Python).

## Real-model tests and benchmark (Jetson)

Weights are local only (no downloads; `HF_HUB_OFFLINE=1` is forced). Defaults:
`/home/sentrix/models/moondream2` and `/home/sentrix/models/qwen2.5-vl-3b`; override with
`LBYR_MOONDREAM_PATH` / `LBYR_QWEN_PATH`.

```bash
pytest                      # default: laptop-safe, real-model tests are deselected
pytest -m jetson -v         # Moondream2 + Qwen3-1.7B on CUDA (~8 GB); VL: LBYR_TEST_QWEN_VL=1
python tools/bench_models.py --critic-only                  # Critic at its runtime input
python tools/bench_models.py --critic-only --critic-crop    # full frame vs 378x378 crop
python tools/bench_models.py --planner-models qwen3-1.7b --combined
```

Critic linear probe (offline: no arm I/O, refuses to run while `lbyr.demo` is running):
`tools/train_critic_probe.py` reads `~/critic_raw/{GRASPED,SLIPPING,EMPTY,OCCLUDED}/*.{jpg,png}`
(sorted per class: first 30 train, last 15 test), applies the runtime dewarp + `critic_crop`
to 1920x1080 frames, mean-pools the Moondream2 vision-encoder patch features (global 378 tile),
trains a 4-class logistic regression (plain torch) to `~/models/critic_probe.pt`, and compares
it with the zero-shot `CRITIC_PROMPT` on the same test split. Output:
`runs/critic_probe_<ts>/{summary.md,predictions.csv,emb.pt,misclassified/}`. Used by
`lbyr.demo --critic probe` (the embedding function lives in `lbyr/ai/critic_probe.py`). Cost: same ~4.1 GB as the Critic; encoder + head ~86 ms vs ~0.8 s per
zero-shot query (Orin NX, smoke test).

```bash
python tools/train_critic_probe.py --limit 2   # smoke test (probe saved in the run dir)
python tools/train_critic_probe.py             # full run
```

Planner config (bible 9.2 / 11.1, v1.4):

- `planner.backend: rules | qwen`, default **rules**. Qwen is only loaded with `qwen`.
  In-loop calls keep the 3 s timeout; after `planner.fallback_after_timeouts` (default 3)
  Qwen timeouts in a row the planner switches to rules for the rest of the run.
- Language mode (demo intro only): `plan_language(...)`, loaded only with
  `planner.language_mode: true`. Default model: Qwen3-1.7B text
  (`planner.language_backend: qwen3_text`, weights `~/models/qwen3-1.7b`, thinking disabled,
  greedy); `qwen_vl` selects Qwen2.5-VL text-only. It uses a text-only prompt, the compact
  schema `{"a":2,"s":40,"g":80,"p":5}`, at most 64 new tokens, and `planner.qwen_timeout_s`
  (8 s). The strict parser is the only path to skills.
- Critic: `critic.dtype: bf16` (FP16 selectable); input = dewarped frame -> 378x378 crop
  (`critic.crop_xywh`, default centred; re-centre on the gripper bite zone once the Trinet is
  mounted). Unparseable answers are re-queried once, then treated as OCCLUDED
  (`critic_invalid_retry` event).
- Every plan is logged as a `reasoning_trace` event in `episodes.jsonl` (prompt, raw output,
  parsed chain).
- Memory guard (`lbyr.main`; the live demo only warns, see below): `load_models` logs
  system RAM (tegrastats) at startup and after each load.
  With `backend: qwen` or `language_mode`, it refuses to start if less than
  `planner.qwen_min_free_ram_mb` (1.5 GB) is free.

### Measured model cost (measured 2026-10-10, 40 W + jetson_clocks; Orin NX 16 GB, JetPack 6.2, torch 2.11, synthetic images)

| Piece | Memory (torch alloc) | Latency | Bible 9.2 estimate |
|---|---|---|---|
| **Moondream2 Critic, BF16, 378x378 crop (runtime)** | ~3.7 GB (4.1 GB peak) | p50 0.785 s / p95 0.789 s | ~4 GB, 0.3-0.8 s |
| **Qwen3-1.7B language mode (default), ~20 tokens** | 3.2 GB | p50 2.14 s / p95 2.18 s, 9.4 tok/s, 9/10 correct, load+warm-up 7.7 s | |
| **Critic + Qwen3-1.7B resident** | 7.2 GB; system RAM 10.80 of 15.29 GB, 4.49 GB free (guard PASS) | Critic 0.78 s, planner 2.13 s | |

Before jetson_clocks (same day, 40 W, clocks not pinned): Critic BF16 crop 0.849 s (FP16
full 640x480 frame, the old default: 1.35 s); Qwen3-1.7B 2.37 s, 8.4 tok/s; combined Critic
0.91 s, planner 2.32 s, 4.5 GB free. Qwen3-0.6B: 2.82 s, 1.1 GB, 2/10 correct (rejected).
Qwen2.5-VL-3B in-loop full plan (~98 tokens, opt-in): ~7.2 GB, 11.2 s, 8.7 tok/s; Critic +
Qwen2.5-VL leaves ~0.8 GB free and is refused by the memory guard.

Pinning the clocks (GPU to 1173 MHz, about 3x) gave only ~10%. All models decode at
~8.5-9.5 tok/s whatever their size, so decoding is limited by per-token framework overhead
(HF `generate`), not compute. Post-hackathon option: a llama.cpp / TensorRT-LLM backend for
the language planner. Not for now.

`lbyr.main` trials use the rules planner (Qwen2.5-VL never meets the 3 s in-loop timeout
here). The live demo uses Qwen2.5-VL once per instruction, arm idle, 30 s timeout.

Live demo, Qwen2.5-VL-3B + Moondream2 both resident (bf16, observed on the rig
2026-10-11): ~14.5 GB of system RAM used, under 1 GB free. The demo warns below 0.8 GB free
instead of refusing. Unloading Qwen did not return memory on the Jetson (RAM stayed ~14.5
GB after "freed"), and a reload costs ~11 s, so Qwen stays loaded for the whole session.
Critic accuracy on real wrist frames is NOT measured yet.
