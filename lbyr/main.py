"""LBYR entry point.

    python -m lbyr.main --mock               # full simulated episode (no hardware)
    python -m lbyr.main --mock --critic-off  # "Critic OFF" baseline condition
    python -m lbyr.main                      # real hardware: preflight lists open questions

Exit codes: 0 = DONE, 1 = ABORT, 2 = hardware not confirmed / not implemented / memory guard.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path

from lbyr.ai.critic import Verdict
from lbyr.config import AppConfig, mock_profile, require
from lbyr.errors import HardwareNotConfirmedError, InsufficientMemoryError
from lbyr.log_setup import setup_logging
from lbyr.logging_.episodes import REQUIRED_FIELDS

log = logging.getLogger("lbyr.main")


def preflight(cfg: AppConfig) -> list[str]:
    """Run every hardware-readiness check and collect what is still unconfirmed."""
    from lbyr.arm.serial_arm import SerialArm
    from lbyr.motion.ik import IkpyIK
    from lbyr.motion.trajectory import capped_limits
    from lbyr.perception.wrist import TrinetWristCamera

    checks: list[tuple[str, Callable[[], object]]] = [
        ("bridge / servos", lambda: SerialArm(cfg.arm).check_confirmed()),
        (
            "feedback",
            lambda: require(cfg.arm.feedback_readable or None, "feedback readable", "OQ-5"),
        ),
        ("speed limits", lambda: capped_limits(cfg.arm, cfg.motion)),
        ("per-tick clamp", lambda: require(cfg.arm.max_delta_per_tick_deg, "max delta", "A-2")),
        ("safe pose", lambda: require(cfg.arm.safe_pose_deg, "safe pose", "A-3")),
        ("gripper mapping", lambda: cfg.arm.gripper.closure_to_deg(0.0)),
        ("IK", lambda: IkpyIK(cfg.arm)),
        ("wrist camera", lambda: TrinetWristCamera(cfg.cameras.wrist)),
    ]
    missing: list[str] = []
    for name, check in checks:
        try:
            check()
        except HardwareNotConfirmedError as e:
            missing.append(f"{name}: {e.what} ({e.question})")
    return missing


def validate_jsonl(path: Path) -> int:
    """Check every line parses and has the bible Section 15 fields. Returns record count."""
    n = 0
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        rec = json.loads(line)
        missing = [k for k in REQUIRED_FIELDS if k not in rec]
        if missing:
            raise ValueError(f"{path}:{i} missing fields {missing}")
        n += 1
    return n


def run_mock(args: argparse.Namespace) -> int:
    from lbyr.fsm.supervisor import EpisodeSpec
    from lbyr.mock_stack import DEMO_SCRIPT, build_mock_stack

    cfg = mock_profile()
    cfg.motion.stage_mode = args.stage
    script = [Verdict(v.upper()) for v in args.script.split(",")] if args.script else DEMO_SCRIPT
    stack = build_mock_stack(
        cfg,
        critic_script=script,
        critic_delay_s=args.critic_delay,
        critic_enabled=not args.critic_off,
        log_dir=args.log_dir,
    )
    log.warning("MOCK MODE: simulated arm and cameras, FAKE geometry. No hardware is driven.")
    with stack:
        result = stack.supervisor.run_episode(EpisodeSpec(args.instruction))
        stats = stack.streamer.stats()
    log.info(
        "Motion streamer: %d ticks, %d missed, jitter p50 %.2f ms / p99 %.2f ms / max %.2f ms",
        stats.ticks,
        stats.missed,
        stats.p50_ms,
        stats.p99_ms,
        stats.max_ms,
    )
    assert stack.logger is not None
    n = validate_jsonl(stack.logger.jsonl_path)
    log.info("Wrote %d valid records to %s", n, stack.logger.jsonl_path)
    return 0 if result.outcome == "success" else 1


def run_hardware(cfg: AppConfig) -> int:
    missing = preflight(cfg)
    if missing:
        log.error(
            "Hardware not confirmed; refusing to run. Answer these in docs/OPEN_QUESTIONS.md:"
        )
        for m in missing:
            log.error("  - %s", m)
        return 2
    log.error("Real-hardware pipeline not wired yet (QwenPlanner / Moondream2Critic / YOLO TODO)")
    return 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m lbyr.main",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mock", action="store_true", help="run fully simulated (no hardware)")
    p.add_argument("--critic-off", action="store_true", help="baseline: verify states pass")
    p.add_argument("--stage", action="store_true", help="use stage speed cap (50%%) not dev (30%%)")
    p.add_argument("--instruction", default="put the acrylic block on the place zone")
    p.add_argument(
        "--script",
        default="",
        help="mock Critic answers, comma-separated (default SLIPPING,GRASPED,GRASPED,EMPTY)",
    )
    p.add_argument("--critic-delay", type=float, default=0.5, help="mock Critic latency, s")
    p.add_argument("--log-dir", type=Path, default=AppConfig().logging.episodes_dir)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    listener = setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    try:
        return run_mock(args) if args.mock else run_hardware(AppConfig())
    except InsufficientMemoryError as e:  # raised by lbyr.ai.models.load_models
        log.error("%s", e)
        return 2
    finally:
        listener.stop()


if __name__ == "__main__":
    sys.exit(main())
