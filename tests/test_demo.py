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
