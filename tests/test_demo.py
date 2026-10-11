"""Smoke test: `python -m lbyr.demo --dry-run --induce-slip` (MockArm + fake cameras)."""

import json
from pathlib import Path

from lbyr.demo import main


def test_demo_dry_run_induced_slip(tmp_path: Path) -> None:
    assert main(["--dry-run", "--induce-slip", "--log-dir", str(tmp_path), "--http-port", "0"]) == 0
    recs = [json.loads(x) for x in next(tmp_path.glob("*/episodes.jsonl")).read_text().splitlines()]
    assert [r["verdict"] for r in recs if r["event"] == "verdict" and not r.get("monitor")] == [
        "SLIPPING",
        "GRASPED",
        "GRASPED",
        "EMPTY",  # VERIFY_PLACE
    ]
    assert sum(r["event"] == "recover" for r in recs) == 1
    assert recs[-1]["outcome"] == "success"
    assert recs[-1]["outcome_label"] == "RECOVERED" and recs[-1]["regrasps"] == 1


def _dry(tmp_path: Path, script: str) -> tuple[int, list[dict]]:
    rc = main(["--dry-run", "--script", script, "--log-dir", str(tmp_path), "--http-port", "0"])
    recs = [json.loads(x) for x in next(tmp_path.glob("*/episodes.jsonl")).read_text().splitlines()]
    return rc, recs


def test_demo_empty_at_verify_grasp_then_grasped_is_recovered(tmp_path: Path) -> None:
    rc, recs = _dry(tmp_path, "VERIFY_GRASP=EMPTY:0.91,GRASPED:0.95;VERIFY_HOLD=GRASPED:0.9")
    assert rc == 0
    fails = [r for r in recs if r["event"] == "failure"]
    assert [r["report"] for r in fails] == ["EMPTY at VERIFY_GRASP (EMPTY 0.91) -> re-grasp 1/2"]
    verdicts = [r for r in recs if r["event"] == "verdict"]
    assert all(set(r["probs"]) == {"GRASPED", "SLIPPING", "EMPTY", "OCCLUDED"} for r in verdicts)
    assert all(r["frame"].endswith(".jpg") for r in verdicts)
    mon = {r["critic_state"] for r in verdicts if r["monitor"] and not r.get("late")}
    assert mon == {"LIFT", "TRANSPORT"}
    end = recs[-1]
    assert end["outcome_label"] == "RECOVERED"
    assert end["result"] == "RECOVERED (1 re-grasp): A -> place"


def test_demo_drop_during_transport_halts_and_retreats_open(tmp_path: Path) -> None:
    rc, recs = _dry(tmp_path, "TRANSPORT=GRASPED:0.9,SLIPPING:0.78,SLIPPING:0.81")
    assert rc == 1
    assert any(r["event"] == "drop_halt" and r["halt"].startswith("m to the current") for r in recs)
    fails = [r for r in recs if r["event"] == "failure"]
    assert [r["report"] for r in fails] == ["DROP during TRANSPORT (SLIPPING 0.78, 0.81) -> abort"]
    labels = [r["target"]["label"] for r in recs if r["event"] == "segment_done"]
    assert not any(x.startswith(("place_pre", "place @")) for x in labels)  # never arrived
    assert labels[-3:] == ["halt (drop)", "open gripper (drop abort)", "home (abort, gripper open)"]
    end = recs[-1]
    assert end["outcome_label"] == "FAILED"
    assert end["result"].startswith("FAILED (DROP at TRANSPORT)")
    assert end["closure_pct"] == 0.0


def test_demo_object_stuck_at_verify_place(tmp_path: Path) -> None:
    rc, recs = _dry(tmp_path, "VERIFY_PLACE=GRASPED:0.93")
    assert rc == 1
    fails = [r for r in recs if r["event"] == "failure"]
    stuck = "object stuck in gripper (GRASPED 0.93 at VERIFY_PLACE)"
    assert [r["report"] for r in fails] == [stuck]
    end = recs[-1]
    assert end["outcome"] == "failed"
    assert end["result"].startswith("FAILED (GRASPED at VERIFY_PLACE)")
    assert [r["target"]["label"] for r in recs if r["event"] == "segment_done"][-1] == "home @ 0%"


def test_demo_low_conf_requeries_then_aborts(tmp_path: Path) -> None:
    rc, recs = _dry(tmp_path, "VERIFY_GRASP=GRASPED:0.50")
    assert rc == 1
    v = [r["verdict"] for r in recs if r["event"] == "verdict" and not r.get("monitor")]
    assert v == ["LOW_CONF", "LOW_CONF"]
    assert recs[-1]["result"].startswith("FAILED (LOW_CONF at VERIFY_GRASP)")


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
