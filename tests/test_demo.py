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


def test_demo_dry_run_task_moves_from_to(tmp_path: Path) -> None:
    """--task: plan first (rules planner in dry-run), then B -> A with the taught sequence."""
    task = "move the block from B to A"
    assert main(["--dry-run", "--task", task, "--log-dir", str(tmp_path), "--http-port", "0"]) == 0
    recs = [json.loads(x) for x in next(tmp_path.glob("*/episodes.jsonl")).read_text().splitlines()]
    plan = next(r for r in recs if r["event"] == "task_plan")
    assert plan["instruction"] == task and plan["raw_text"]
    assert plan["parsed_action"]["from"] == "B" and plan["parsed_action"]["to"] == "A"
    labels = [r["target"]["label"] for r in recs if r["event"] == "segment_done"]
    assert labels[1].startswith("zone_B_pre") and labels[2].startswith("zone_B_grasp")
    assert any(x.startswith("zone_A_grasp") for x in labels[5:])
    end = recs[-1]
    assert end["outcome"] == "success" and end["instruction"] == task
    assert end["parsed_action"]["to"] == "A"


def test_demo_task_none_moves_nothing(tmp_path: Path) -> None:
    assert main(["--dry-run", "--task", "dance", "--log-dir", str(tmp_path), "--http-port", "0"])
    recs = [json.loads(x) for x in next(tmp_path.glob("*/episodes.jsonl")).read_text().splitlines()]
    assert not any(r["event"] == "segment_done" for r in recs)
    assert recs[-1]["outcome"] == "none"


def test_demo_task_qwen_path_resident(tmp_path: Path) -> None:
    """Qwen path with a fake engine (resident, never unloaded): the prompt carries image +
    locations + the grounding rule; an unknown location is refused before Qwen."""
    from typing import Any

    from lbyr.ai.planner import QwenPlanner
    from lbyr.config import DemoConfig, PlannerConfig
    from lbyr.demo import Demo, DemoStatus, build_dry_rig, parse_args, run_stoppers

    seen: list[tuple[str, Any]] = []

    class Engine:
        def run(self, prompt: str, image: Any, n: int, t: float | None) -> tuple[str, int]:
            seen.append((prompt, image))
            return '{"action":"move","from":"A","to":"B","reason":"user asked"}', 20

    args = parse_args(["--dry-run", "--log-dir", str(tmp_path), "--http-port", "0"])
    rig = build_dry_rig(args, DemoConfig(http_port=0), DemoStatus())
    rig.qwen = QwenPlanner(PlannerConfig(backend="qwen"), engine=Engine())
    try:
        demo = Demo(rig)
        assert demo.run(False, "take the block from A over to B") is True
        n = len(seen)
        assert demo.run(False, "move the block from A to zone C") is False
        assert len(seen) == n  # refused before Qwen
    finally:
        run_stoppers(rig.stoppers)
    assert rig.qwen is not None
    prompt, image = seen[-1]
    assert image is not None and "- B: FAKE: back of the table" in prompt
    assert '"none" ONLY if no block is visible' in prompt
    assert rig.status.result_kind == "rejected" and "valid names are A, B, place" in (
        rig.status.result
    )


def test_web_task_post_and_busy() -> None:
    import urllib.request

    from lbyr.demo import Commands, DemoStatus, DemoWeb

    status = DemoStatus()
    cmds = Commands(status)

    def task(body: dict) -> dict:
        return {"accepted": True} if cmds.submit(str(body["instruction"])) else {"busy": True}

    web = DemoWeb("127.0.0.1", 0, status, {}, {}, {"task": task})
    web.start()
    try:

        def post(text: str) -> dict:
            req = urllib.request.Request(
                f"http://127.0.0.1:{web.port}/task",
                json.dumps({"instruction": text}).encode(),
                {"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                return json.loads(r.read())

        assert post("move from A to B") == {"accepted": True}
        assert post("again") == {"busy": True} and status.busy
        assert cmds.q.get_nowait() == ("move from A to B", False, "web")
        cmds.done()
        assert post("again") == {"accepted": True}
    finally:
        web.stop()


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
