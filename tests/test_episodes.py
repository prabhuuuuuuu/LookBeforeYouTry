import json
from enum import Enum
from pathlib import Path

import numpy as np

from lbyr.logging_.episodes import (
    REQUIRED_FIELDS,
    EpisodeLogger,
    make_record,
    parse_tegrastats_ram_mb,
)


class Color(str, Enum):
    RED = "RED"


def test_make_record_has_required_fields() -> None:
    rec = make_record(fsm_state=Color.RED, joints=np.array([1.23456, 2.0]), extra=(1, 2))
    for k in REQUIRED_FIELDS:
        assert k in rec
    assert rec["fsm_state"] == "RED"
    assert rec["joints"] == [1.235, 2.0]
    json.dumps(rec)


def test_logger_writes_jsonl_and_frames(tmp_path: Path) -> None:
    logger = EpisodeLogger(tmp_path / "session")
    logger.start()
    logger.log(make_record(fsm_state="IDLE", outcome=None))
    rel = logger.save_frame("ep1/000_VERIFY_GRASP.jpg", np.zeros((8, 8, 3), np.uint8))
    logger.log(make_record(fsm_state="DONE", outcome="success", frame=rel))
    logger.stop()
    lines = (tmp_path / "session" / "episodes.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["outcome"] == "success"
    assert logger.frames_written == 1
    assert any((tmp_path / "session" / "frames" / "ep1").iterdir())


def test_tegrastats_parse() -> None:
    line = "RAM 7123/15388MB (lfb 3x4MB) SWAP 0/7694MB CPU [3%@729,...] GR3D_FREQ 0%"
    assert parse_tegrastats_ram_mb(line) == 7123.0
    assert parse_tegrastats_ram_mb("garbage") is None
