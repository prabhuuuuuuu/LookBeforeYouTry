"""Smoke test: `python -m lbyr.demo --dry-run --induce-slip` (MockArm + fake cameras)."""

import json
from pathlib import Path

from lbyr.demo import main


def test_demo_dry_run_induced_slip(tmp_path: Path) -> None:
    assert main(["--dry-run", "--induce-slip", "--log-dir", str(tmp_path), "--http-port", "0"]) == 0
    recs = [json.loads(x) for x in next(tmp_path.glob("*/episodes.jsonl")).read_text().splitlines()]
    assert [r["verdict"] for r in recs if r["event"] == "verdict"] == [
        "SLIPPING",
        "GRASPED",
        "GRASPED",
    ]
    assert sum(r["event"] == "recover" for r in recs) == 1
    assert recs[-1]["outcome"] == "success"


def test_demo_wrist_drop_is_occluded_not_a_crash(tmp_path: Path) -> None:
    """No fresh wrist frame -> OCCLUDED (re-queried once), then the normal retreat home."""
    from lbyr.config import DemoConfig
    from lbyr.demo import Demo, DemoStatus, build_dry_rig, parse_args, run_stoppers

    args = parse_args(["--dry-run", "--log-dir", str(tmp_path), "--http-port", "0"])
    rig = build_dry_rig(args, DemoConfig(http_port=0), DemoStatus())
    rig.wrist_fresh = lambda: None  # camera re-enumerating
    try:
        assert Demo(rig).run(induce_slip=False) is False
    finally:
        run_stoppers(rig.stoppers)
    recs = [json.loads(x) for x in next(tmp_path.glob("*/episodes.jsonl")).read_text().splitlines()]
    verdicts = [r for r in recs if r["event"] == "verdict"]
    assert [r["verdict"] for r in verdicts] == ["OCCLUDED", "OCCLUDED"]
    assert all(r["no_frame"] for r in verdicts)
    assert recs[-1]["outcome"] == "abort"
    assert any(r.get("target", {}) and r["target"]["label"] == "home (abort)" for r in recs)
