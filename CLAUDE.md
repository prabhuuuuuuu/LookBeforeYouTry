# CLAUDE.md: rules for AI-assisted sessions on LBYR

Source of truth: `docs/LookBeforeYouTry-context.md` (the bible). It wins over everything,
including this file. Open hardware questions: `docs/OPEN_QUESTIONS.md`.

## Project in one line

Qwen2.5-VL-3B planner + deterministic IK motion + RealSense D435 overhead perception +
Trinet wrist Critic (Moondream2) + parameterized recovery FSM, all local on a Jetson Orin 16GB.

## Hard rules

1. Edge-only inference. No cloud APIs anywhere, ever.
2. Modular VLA, not end-to-end. The VLM outputs a JSON skill chain; **never joint angles**. IK makes joints.
3. Python 3.10+ (no `StrEnum`, no `type` statements), full type hints, `@dataclass` configs,
   `pathlib`, `logging`. No `print` in library code (`tools/` may print).
4. Single process, threads + `queue.Queue`. No ROS 2. No asyncio in the core loop.
5. Units at all protocol boundaries: mm and degrees.
6. Every VLM call goes through the timeout wrapper (`lbyr/ai/timeout.py`), temperature 0.
   Planner: 3 s hard timeout. Critic: explicit `CriticConfig.timeout_s`.
7. The 50 Hz `MotionStreamer` never blocks on AI, detection, logging or I/O. Deadline
   scheduling with `time.perf_counter()`. Feed it only via latest-value buffers.
8. The Critic is async: the FSM submits and polls, never waits.
9. The wrist camera (Trinet) is semantic-only: dewarp -> Critic. Never metric control.
   The D435 is the only metric camera.
10. **Never guess hardware values.** Unconfirmed values are `None` in `lbyr/config.py`
    with an `OQ-n`/`A-n` comment; code raises `HardwareNotConfirmedError`. If you need one,
    ask for the specific open question.
11. `mock_profile()` values are FAKE and SIMULATION-ONLY. Real drivers reject them.
12. Safety: never suggest powering servos from MCU pins / Jetson USB; never assume a 5 V
    servo rail; dev speed cap 30%, stage 50%; torque off when hands are near; E-stop always.
13. torch / opencv / pyrealsense2 come from JetPack or source builds: import them lazily
    inside the modules that need them; never add them as pip deps.
14. Every feature proposal states its latency and VRAM cost (budget: bible 9.2).
15. Minimal diffs, testable increments. Priority: live demo > paper-quality logs > elegance.

## Workflow

```bash
pip install -e ".[laptop,dev]"
pytest                       # must pass on a laptop with no hardware
ruff check . && ruff format --check . && mypy lbyr
python -m lbyr.main --mock   # one simulated episode with one recovery
```

Commit in small increments with conventional commit messages (`feat(fsm): ...`).

## Map

- `lbyr/config.py`: all configs, `mock_profile()`, `require()`
- `lbyr/arm/`: `bridge_proto` (JSON protocol), `MockArm`, `SerialArm` (gated)
- `lbyr/perception/`: latest-value buffers, D435/Trinet capture + fakes, dewarp, geometry, detector
- `lbyr/ai/`: timeout wrapper, planner (strict JSON), Critic (verdict parser + worker)
- `lbyr/motion/`: IK (ikpy gated on OQ-4; `MockIK`), trapezoid trajectories, 50 Hz streamer
- `lbyr/fsm/`: states (bible 10), recovery table as data (bible 13), supervisor (10 Hz)
- `lbyr/logging_/episodes.py`: episodes.jsonl + frames/, tegrastats VRAM
- `lbyr/mock_stack.py`: wires the simulated stack for `--mock` and tests
