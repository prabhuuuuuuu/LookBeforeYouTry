"""tools/autolabel.py offline: fake episodes + faked urlopen (no network, no API key use)."""

from __future__ import annotations

import csv
import importlib.util
import sys
import urllib.request
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from test_critic_cloud import FakeUrlopen, _http_error, _text  # tests/ is on sys.path


def _load() -> ModuleType:
    path = Path(__file__).parents[1] / "tools" / "autolabel.py"
    spec = importlib.util.spec_from_file_location("autolabel", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["autolabel"] = mod
    spec.loader.exec_module(mod)
    return mod


al = _load()


def _jpeg(w: int, h: int) -> bytes:
    sof = b"\x08" + h.to_bytes(2, "big") + w.to_bytes(2, "big") + b"\x03" + b"\x01\x22\x00" * 3
    return b"\xff\xd8\xff\xe0\x00\x04ab" + b"\xff\xc0" + (len(sof) + 2).to_bytes(2, "big") + sof


def _episodes(root: Path) -> None:
    real = root / "demo-1" / "frames" / "20261011-100000"
    fake = root / "demo-2" / "frames" / "20261011-110000"
    for d, size in ((real, (1920, 1080)), (fake, (640, 480))):
        d.mkdir(parents=True)
        (d / "001_VERIFY_GRASP_raw.jpg").write_bytes(_jpeg(*size))
        (d / "002_VERIFY_GRASP_crop_GRASPED.jpg").write_bytes(_jpeg(378, 378))
        (d / "003_LIFT_crop_LOW_CONF.jpg").write_bytes(_jpeg(378, 378))
        (d / "004_VERIFY_PLACE_crop_EMPTY.jpg").write_bytes(_jpeg(378, 378))


def test_jpeg_size_and_parse() -> None:
    assert al.parse_crop(Path("x/20261011-1/003_LIFT_crop_LOW_CONF.jpg")) == (
        "20261011-1", "LIFT", "LOW_CONF",
    )  # fmt: skip
    assert al.parse_crop(Path("e/012_VERIFY_GRASP_crop_GRASPED.jpg"))[1:] == (
        "VERIFY_GRASP", "GRASPED",
    )  # fmt: skip


def test_autolabel_labels_copies_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    eps, out, runs = tmp_path / "episodes", tmp_path / "auto", tmp_path / "runs"
    _episodes(eps)
    assert al.jpeg_size(eps / "demo-2/frames/20261011-110000/001_VERIFY_GRASP_raw.jpg") == (
        640, 480,
    )  # fmt: skip
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k-test")
    fake = FakeUrlopen([_text("GRASPED"), _http_error(400), _text("EMPTY")])
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    base = ["--episodes", str(eps), "--out", str(out), "--runs-dir", str(runs), "--rate", "6000"]
    assert al.main(base) == 0
    assert len(fake.requests) == 3  # the fake (640x480) episode is skipped
    rows = list(csv.DictReader(next(runs.glob("autolabel_*/labels.csv")).open()))
    assert tuple(rows[0]) == al.FIELDS
    assert [(r["state"], r["runtime_verdict"], r["cloud_verdict"]) for r in rows] == [
        ("VERIFY_GRASP", "GRASPED", "GRASPED"),
        ("LIFT", "LOW_CONF", "TIMEOUT"),
        ("VERIFY_PLACE", "EMPTY", "EMPTY"),
    ]
    assert {p.relative_to(out).as_posix() for p in out.rglob("*.jpg")} == {
        "GRASPED/20261011-100000_002_VERIFY_GRASP_crop_GRASPED.jpg",
        "EMPTY/20261011-100000_004_VERIFY_PLACE_crop_EMPTY.jpg",
    }
    text = capsys.readouterr().out
    assert "agreement (both a class): 2/2" in text and "k-test" not in text

    # resume: only the TIMEOUT row is retried
    fake2 = FakeUrlopen([_text("SLIPPING")])
    monkeypatch.setattr(urllib.request, "urlopen", fake2)
    assert al.main([*base, "--resume"]) == 0
    assert len(fake2.requests) == 1
    final = al.read_rows(next(runs.glob("autolabel_*/labels.csv")))
    assert sorted(r["cloud_verdict"] for r in final.values()) == ["EMPTY", "GRASPED", "SLIPPING"]


def test_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eps = tmp_path / "episodes"
    _episodes(eps)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k-test")
    fake: Any = FakeUrlopen([_text("GRASPED")])
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    argv = ["--episodes", str(eps), "--out", str(tmp_path / "a"), "--runs-dir",
            str(tmp_path / "r"), "--limit", "1", "--include-fake", "--rate", "6000"]  # fmt: skip
    assert al.main(argv) == 0 and len(fake.requests) == 1
