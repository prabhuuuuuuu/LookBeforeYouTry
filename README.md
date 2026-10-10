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
| Config, bridge protocol, MockArm, trajectories, 50 Hz streamer, FSM + recovery, logging | Implemented and tested |
| Fisheye dewarp, metric geometry, Umeyama hand-eye solve | Implemented and tested |
| SerialArm, IkpyIK, Trinet capture | Gated on open hardware questions (OQ-4, 6-12, A-1..3, A-6, A-7). `SerialArm` also needs the firmware's text protocol (not implemented) |
| Moondream2Critic (BF16, 378x378 crop at runtime), RuleBasedPlanner (default), Qwen3-1.7B text planner for language mode, QwenPlanner (Qwen2.5-VL, opt-in), memory guard, `reasoning_trace` log event, `tools/bench_models.py` | Implemented; verified on the Orin NX (`pytest -m jetson`). Trials use the rules planner. Critic accuracy on real wrist frames is NOT measured yet. Language-planner pick: OPEN_QUESTIONS M-2 |
| YOLOv11n TensorRT | Stub (TODO, lazy imports) |
| Bridge firmware (XIAO ESP32-C6, PWM, text protocol) + `tools/arm_gui.py` bench GUI | Exists (`firmware/xiao_c6_bridge/`). Joint roles, PWM servos, 6.0 V rail and no-feedback are confirmed (OQ-1/2/3/5 closed). Zeros in `data/calib/zero_positions.json` still to be verified against the board's `list` output |

## Layout

```
lbyr/       arm/ perception/ ai/ motion/ fsm/ logging_/ config.py main.py mock_stack.py
tools/      beginner-friendly camera / calibration / labeling / serial scripts
firmware/   XIAO ESP32-C6 bridge (xiao_c6_bridge/)
tests/      pytest, runs without hardware
docs/       bible, open questions, onboarding
data/       git-ignored: frames/, datasets/, calib/, episodes/
```

## Jetson install

JetPack provides CUDA-enabled OpenCV, torch and pyrealsense2 (the last one built from
source). Do **not** pip-install them on the Jetson.

```bash
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install -U pip setuptools wheel "packaging>=24.2"
export PIP_CONSTRAINT=constraints-jetson.txt   # numpy<2: NumPy 2 breaks the JetPack cv2
pip install -e ".[jetson,dev]"
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

## Real-model tests and benchmark (Jetson)

Weights are local only (no downloads; `HF_HUB_OFFLINE=1` is forced). Defaults:
`/home/sentrix/models/moondream2` and `/home/sentrix/models/qwen2.5-vl-3b`; override with
`LBYR_MOONDREAM_PATH` / `LBYR_QWEN_PATH`.

```bash
pytest                      # default: laptop-safe, real-model tests are deselected
pytest -m jetson -v         # Moondream2 + Qwen3-1.7B on CUDA (~8 GB); VL: LBYR_TEST_QWEN_VL=1
python tools/bench_models.py --critic-only                  # Critic at its runtime input
python tools/bench_models.py --critic-only --critic-crop    # full frame vs 378x378 crop
python tools/bench_models.py --planner-models qwen3-1.7b,qwen3-0.6b --combined
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

### Measured model cost (2026-10-10, Orin NX 16 GB, nvpmodel 40 W, JetPack 6.2, torch 2.11, synthetic images)

| Piece | Memory (torch alloc) | Latency | Bible 9.2 estimate |
|---|---|---|---|
| **Moondream2 Critic, BF16, 378x378 crop (runtime)** | ~3.7 GB (4.1 GB peak) | p50 0.85 s | ~4 GB, 0.3-0.8 s |
| Moondream2 Critic, FP16, full 640x480 frame (old default) | ~4.1 GB | p50 1.35 s | |
| **Qwen3-1.7B language mode (default), ~20 tokens** | 3.2 GB | p50 2.37 s, 8.4 tok/s, 9/10 correct | |
| Qwen3-0.6B language mode | 1.1 GB | p50 2.82 s, 2/10 correct | |
| Qwen2.5-VL-3B in-loop full plan (~98 tokens, opt-in) | ~7.2 GB | p50 11.2 s, 8.7 tok/s | ~6.5 GB, 0.8-2 s |
| **Critic + Qwen3-1.7B resident** | 7.2 GB; system RAM 10.75 of 15.3 GB, 4.5 GB free | Critic 0.91 s, planner 2.32 s | |
| Critic + Qwen2.5-VL resident | 11.3 GB; system RAM peak ~14.8 GB, ~0.8 GB free | | peak 11-12 GB |

Trials use the rules planner (Qwen2.5-VL never meets the 3 s in-loop timeout here). With
Qwen3-1.7B, Critic + language planner leave 4.5 GB free and pass the 1.5 GB memory guard;
Critic + Qwen2.5-VL (~0.8 GB free) is refused. Critic accuracy is NOT measured yet.
