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
| SerialArm, IkpyIK, Trinet capture | Gated on open hardware questions (OQ-1..12, A-1..7) |
| QwenPlanner, Moondream2Critic, YOLOv11n TensorRT | Stubs (TODO, lazy imports) |
| ESP32 firmware | Not started (`firmware/esp32_bridge/README.md`) |

## Layout

```
lbyr/       arm/ perception/ ai/ motion/ fsm/ logging_/ config.py main.py mock_stack.py
tools/      beginner-friendly camera / calibration / labeling / serial scripts
firmware/   ESP32 bridge (placeholder)
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
