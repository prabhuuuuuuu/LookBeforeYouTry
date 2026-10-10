# Look Before You Retry (LBYR)

**Edge-native visual grounding for VLA failure recovery.** A modular VLA (Qwen2.5-VL-3B
planner + deterministic IK motion stack) with an asynchronous visual Critic (Moondream2 on a
wrist-mounted Trinet camera) that verifies grasps at FSM checkpoints and picks a
parameterized recovery instead of retrying blindly. Everything runs on a Jetson Orin 16GB.
There are no cloud calls.

Awign Hackathon 2026, Physical AI track. Project bible: `docs/LookBeforeYouTry-context.md`.

```
Planner proposes skill chain -> arm executes -> wrist camera verifies
-> Critic: GRASPED | SLIPPING | EMPTY | OCCLUDED -> FSM recovers (k <= 2) or aborts safely
```

## Quick start (laptop, no hardware)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[laptop,dev]"
pytest
python -m lbyr.main --mock               # one episode: SLIPPING -> +15% closure -> DONE
python -m lbyr.main --mock --critic-off  # baseline condition
python -m lbyr.main --mock --script SLIPPING,SLIPPING,SLIPPING   # ends in ABORT + report
```

Logs go to `data/episodes/<session>/episodes.jsonl`, and decision-point frames go to `frames/`.

Running without `--mock` performs a hardware preflight and lists every open question that
still blocks real-hardware runs (see `docs/OPEN_QUESTIONS.md`).

## Status

| Area | State |
|---|---|
| Config, FSM + recovery, logging, MockArm (firmware planner port), MotionController (one `m` per segment + DONE, PP-1), taught-pose mode | Implemented and tested |
| Fisheye dewarp, metric geometry, Umeyama hand-eye solve | Implemented and tested |
| SerialArm (firmware text protocol, zeros checked against `data/calib/joints.yaml`) | Implemented; tested against a fake board and the real sketch compiled on the host (`tests/fw_sim/`). Not yet run on the real board |
| IkpyIK, Trinet capture | Gated on open hardware questions (OQ-4, OQ-6..12, A-7) |
| Moondream2Critic (BF16, 378x378 crop at runtime), RuleBasedPlanner (default), Qwen3-1.7B text planner for language mode, QwenPlanner (Qwen2.5-VL, opt-in), memory guard, `reasoning_trace` log event, `tools/bench_models.py` | Implemented; verified on the Orin NX (`pytest -m jetson`). Trials use the rules planner. Critic accuracy on real wrist frames is NOT measured yet. Language-mode planner: Qwen3-1.7B (M-2 closed) |
| YOLOv11n TensorRT | Stub (TODO, lazy imports) |
| Bridge firmware (XIAO ESP32-C6, PWM, text protocol: `DONE`, `q`, extended `S`) + `tools/arm_gui.py` (sliders, Save pose as / Go to pose) | Firmware changes NOT yet flashed / tested on the board. `data/calib/joints.yaml` is seeded from the firmware defaults and must be verified against the board's `list` (`verified: true`). `data/calib/poses.yaml` is empty: teach the poses |

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
segment and waits for the firmware's `DONE`. The `MotionController` thread polls `q` at
20 Hz. If no DONE arrives within predicted × 1.5 + 0.5 s, that is a LINK fault and the
episode aborts. The predicted duration comes from a Python port of the firmware planner
(`lbyr/arm/fw_planner.py`); `MockArm` runs the same port, so mock timing matches the board.

Bring-up on the real arm:

1. Flash `firmware/xiao_c6_bridge/Code.ino` (adds `DONE`, `q` and the extended `S` line).
2. `python tools/arm_gui.py` → Connect → "Sync sliders". Compare every `S,<j>,<zero_us>,...`
   line with `data/calib/joints.yaml`; fix if needed, then set `verified: true`.
   `SerialArm` refuses to move if the board's zeros differ from the file.
3. Teach the poses: move with the sliders, then "Save pose as..." with the names `home`,
   `zone_A_pre`, `zone_A_grasp`, `place_pre`, `place`. Use "Go to pose" to check each one.
4. Edit `data/calib/poses.yaml`: `gripper.open_deg` / `closed_deg` (slider 6 values), and per
   zone `polygon_px` (overhead image pixels) and `radius_mm` (base axis to zone centre).
5. `python -m lbyr.main` lists anything still missing (preflight).

Taught-mode recovery: +15% closure → joint 6; wrist re-orient 30° → joint 5; lateral 10 mm
→ base yaw + atan(10 / radius_mm); −5 mm standoff → the pre-grasp pose moves along the
joint-space segment zone_X_pre → zone_X_grasp by 5 / standoff_mm (not a straight line in
Cartesian space; the grasp pose is unchanged).

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
- Memory guard: `load_models` logs system RAM (tegrastats) at startup and after each load.
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

Trials use the rules planner (Qwen2.5-VL never meets the 3 s in-loop timeout here). Critic
accuracy on real wrist frames is NOT measured yet.
