"""Probe backend, LOW_CONF, scripted per-state sequences and the continuous monitor."""

import time
from pathlib import Path

import numpy as np
import pytest

from lbyr.ai.critic import (
    CriticOutput,
    CriticResult,
    CriticWorker,
    ScriptedCritic,
    Verdict,
    apply_confidence,
    parse_script,
    scripted_output,
)
from lbyr.config import CriticConfig


def test_parse_script_default_and_per_state() -> None:
    default, per = parse_script("SLIPPING,grasped")
    assert default == ["SLIPPING", "GRASPED"] and per == {}
    default, per = parse_script("VERIFY_GRASP=EMPTY:0.91,GRASPED;TRANSPORT=SLIPPING:0.78; EMPTY")
    assert default == ["EMPTY"]
    assert per == {"VERIFY_GRASP": ["EMPTY:0.91", "GRASPED"], "TRANSPORT": ["SLIPPING:0.78"]}
    with pytest.raises(ValueError):
        parse_script("BANANA")
    with pytest.raises(ValueError):
        parse_script("EMPTY:1.5")


def test_scripted_output_probs_and_low_conf() -> None:
    out = scripted_output("EMPTY:0.91")
    assert out.verdict is Verdict.EMPTY and out.probs is not None
    assert out.probs["EMPTY"] == pytest.approx(0.91) and sum(out.probs.values()) == pytest.approx(1)
    assert scripted_output("GRASPED").probs is None
    low = scripted_output("GRASPED:0.5")
    assert apply_confidence(low.verdict, low.probs, 0.55) is Verdict.LOW_CONF
    assert apply_confidence(Verdict.GRASPED, None, 0.55) is Verdict.GRASPED  # zero-shot
    assert apply_confidence(Verdict.TIMEOUT, low.probs, 0.55) is Verdict.TIMEOUT


def test_scripted_per_state_sequences_are_independent() -> None:
    c = ScriptedCritic(["GRASPED"], per_state={"TRANSPORT": ["GRASPED", "SLIPPING:0.8"]})
    img = np.zeros((8, 8, 3), np.uint8)
    assert c.classify(img, "TRANSPORT").verdict is Verdict.GRASPED
    assert c.classify(img, "VERIFY_GRASP").verdict is Verdict.GRASPED
    assert c.classify(img, "TRANSPORT").verdict is Verdict.SLIPPING
    assert c.classify(img, "TRANSPORT").verdict is Verdict.SLIPPING  # last repeats
    assert c.infer(img, "p") == "GRASPED"


def _results(worker: CriticWorker, n: int, timeout: float = 3.0) -> list[CriticResult]:
    out: list[CriticResult] = []
    t_end = time.perf_counter() + timeout
    while len(out) < n and time.perf_counter() < t_end:
        r = worker.poll()
        if r is None:
            time.sleep(0.005)
        else:
            out.append(r)
    return out


def test_worker_classify_path_low_conf_and_probs() -> None:
    w = CriticWorker(ScriptedCritic(["GRASPED:0.5", "EMPTY:0.9"]), CriticConfig(timeout_s=1.0))
    w.start()
    try:
        img = np.zeros((480, 640, 3), np.uint8)
        w.submit(img, "VERIFY_GRASP")
        w.submit(img, "VERIFY_GRASP")
        a, b = _results(w, 2)
    finally:
        w.stop()
    assert a.verdict is Verdict.LOW_CONF and a.probs is not None and not a.monitor
    assert b.verdict is Verdict.EMPTY and b.probs is not None and b.probs["EMPTY"] == 0.9


def test_worker_monitor_runs_back_to_back_and_stops() -> None:
    backend = ScriptedCritic(["GRASPED"], delay_s=0.01, per_state={"LIFT": ["GRASPED:0.9"]})
    w = CriticWorker(backend, CriticConfig(timeout_s=1.0))
    frames = {"n": 0}

    def source() -> np.ndarray:
        frames["n"] += 1
        return np.zeros((480, 640, 3), np.uint8)

    w.start()
    try:
        w.start_monitor("LIFT", source)
        assert w.monitoring == "LIFT"
        got = _results(w, 5)
        assert len(got) == 5 and all(r.monitor and r.state == "LIFT" for r in got)
        assert all(r.image_bgr is not None and r.image_bgr.shape == (378, 378, 3) for r in got)
        w.submit(np.zeros((480, 640, 3), np.uint8), "VERIFY_HOLD")  # FSM requests still served
        assert any(not r.monitor and r.state == "VERIFY_HOLD" for r in _results(w, 10))
        w.stop_monitor()
        time.sleep(0.05)
        while w.poll() is not None:  # at most one in-flight result
            pass
        n = frames["n"]
        time.sleep(0.15)
        assert w.poll() is None and frames["n"] == n and w.monitoring is None
    finally:
        w.stop()


def test_worker_monitor_no_frame_is_retried_not_posted() -> None:
    w = CriticWorker(ScriptedCritic(["GRASPED"]), CriticConfig(timeout_s=1.0))
    w.start()
    try:
        w.start_monitor("TRANSPORT", lambda: None)
        time.sleep(0.2)
        assert w.poll() is None
    finally:
        w.stop()


def test_probe_critic_with_injected_head() -> None:
    torch = pytest.importorskip("torch")
    from lbyr.ai.critic_probe import CLASS_NAMES, EMBED_FN_NAME, ProbeCritic, ProbeHead

    w = torch.zeros(4, 3)
    w[2, 0] = 5.0  # feature 0 -> EMPTY
    head = ProbeHead(w, torch.zeros(4), torch.zeros(3), torch.ones(3), CLASS_NAMES, None,
                     EMBED_FN_NAME)  # fmt: skip
    feats = {"x": torch.tensor([0.0, 0.0, 0.0])}
    critic = ProbeCritic(CriticConfig(), head=head, embed=lambda img: feats["x"])
    out = critic.classify(np.zeros((378, 378, 3), np.uint8), "VERIFY_GRASP")
    assert isinstance(out, CriticOutput) and out.probs is not None
    assert max(out.probs.values()) == pytest.approx(0.25)  # uniform -> LOW_CONF in the worker
    feats["x"] = torch.tensor([1.0, 0.0, 0.0])
    out = critic.classify(np.zeros((378, 378, 3), np.uint8), "VERIFY_GRASP")
    assert out.verdict is Verdict.EMPTY and out.probs is not None and out.probs["EMPTY"] > 0.9
    assert set(out.probs) == set(CLASS_NAMES)


def test_probe_head_load_checks_embedding_name(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from lbyr.ai.critic_probe import CLASS_NAMES, EMBED_FN_NAME, ProbeHead
    from lbyr.errors import LBYRError

    d = {
        "weights": torch.zeros(4, 5),
        "bias": torch.zeros(4),
        "mean": torch.zeros(5),
        "std": torch.ones(5),
        "classes": list(CLASS_NAMES),
        "crop": (0.1, 0.2, 0.5, 0.9),
        "embed_fn_name": EMBED_FN_NAME,
    }
    torch.save(d, tmp_path / "p.pt")
    head = ProbeHead.load(tmp_path / "p.pt")
    assert head.crop == (0.1, 0.2, 0.5, 0.9) and head.classes == CLASS_NAMES
    torch.save({**d, "embed_fn_name": "other"}, tmp_path / "q.pt")
    with pytest.raises(LBYRError):
        ProbeHead.load(tmp_path / "q.pt")
