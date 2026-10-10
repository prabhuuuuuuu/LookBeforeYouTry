"""Definition of done: `python -m lbyr.main --mock` completes one episode with one recovery."""

import json
from pathlib import Path

from lbyr.main import main


def test_mock_demo_end_to_end(tmp_path: Path) -> None:
    assert main(["--mock", "--log-dir", str(tmp_path), "--critic-delay", "0.05"]) == 0
    records = [
        json.loads(line)
        for line in next(tmp_path.glob("*/episodes.jsonl")).read_text().splitlines()
    ]
    assert sum(r["event"] == "recover" for r in records) == 1
    assert records[-1]["outcome"] == "success"
    assert [r["verdict"] for r in records if r["event"] == "verdict"] == [
        "SLIPPING",
        "GRASPED",
        "GRASPED",
        "EMPTY",
    ]
    assert any(next(tmp_path.glob("*/frames")).rglob("*.jpg"))


def test_hardware_mode_refuses_without_confirmation() -> None:
    assert main([]) == 2
