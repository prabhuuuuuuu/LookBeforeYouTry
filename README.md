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
| Moondream2Critic (FP16), QwenPlanner (FP16) + RuleBasedPlanner fallback, `tools/bench_models.py` | Implemented; verified on the Orin NX (`pytest -m jetson`). Qwen cannot meet the 3 s timeout on this board (see "Measured model cost") |
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
pytest -m jetson -v         # real models on CUDA (loads both, about 11 GB, about 1 min)
python tools/bench_models.py --n 10    # p50/p95, tokens/s, peak VRAM -> data/bench/*.json
```

Config: `planner.backend: qwen | rules`; after `planner.fallback_after_timeouts` (default 3)
Qwen timeouts in a row the planner switches to the rule-based one for the rest of the run.

### Measured model cost (Orin NX 16 GB, nvpmodel 40 W, JetPack 6.2, torch 2.11, FP16)

| Piece | VRAM (torch alloc) | Latency | Bible 9.2 |
|---|---|---|---|
| Moondream2 Critic, 640x480 frame | ~4.0 GB | p50 1.35 s (0.8 s at 378x378) | ~4 GB, 0.3-0.8 s |
| Qwen2.5-VL-3B planner, full plan (~98 tokens) | ~7.0 GB | p50 11.2 s, 8.7 tok/s | ~6.5 GB, 0.8-2 s |
| Both resident | 11.3 GB torch; system RAM peak ~14.8 GB of 15.6 GB | | peak 11-12 GB |

The planner's 3 s hard timeout is NOT met by free-form JSON generation on this board, so
with `backend: qwen` every plan times out and the fallback to rules takes over. System RAM
headroom with both models loaded is under 1 GB.
