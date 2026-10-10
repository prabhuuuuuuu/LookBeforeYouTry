import time

import numpy as np
import pytest

from lbyr.ai.critic import CRITIC_PROMPT, CriticWorker, ScriptedCritic, Verdict, parse_verdict
from lbyr.config import CriticConfig


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Grasped.", Verdict.GRASPED),
        (" slipping\n", Verdict.SLIPPING),
        ("EMPTY", Verdict.EMPTY),
        ("**Occluded**", Verdict.OCCLUDED),
        ("'grasped'", Verdict.GRASPED),
        ("", Verdict.INVALID),
        (None, Verdict.INVALID),
        ("asdf qwerty", Verdict.INVALID),
        ("banana", Verdict.INVALID),
        ("not grasped", Verdict.INVALID),
        ("GRASPED | SLIPPING", Verdict.INVALID),
        ("The object is grasped.", Verdict.INVALID),
        ("TIMEOUT", Verdict.INVALID),  # the model may not emit internal verdicts
    ],
)
def test_parse_verdict(text: str | None, expected: Verdict) -> None:
    assert parse_verdict(text) is expected


def test_prompt_is_locked_bible_text() -> None:
    assert CRITIC_PROMPT.splitlines() == [
        "You are a robot grasp inspector.",
        "Image = wrist camera after gripper closed.",
        "Answer ONE word: GRASPED | SLIPPING | EMPTY | OCCLUDED",
    ]


def _wait_result(worker: CriticWorker, timeout: float = 3.0):  # type: ignore[no-untyped-def]
    t_end = time.perf_counter() + timeout
    while time.perf_counter() < t_end:
        r = worker.poll()
        if r is not None:
            return r
        time.sleep(0.005)
    raise AssertionError("no critic result")


def test_worker_returns_verdicts_in_order() -> None:
    worker = CriticWorker(ScriptedCritic(["Slipping.", "grasped"]), CriticConfig(timeout_s=1.0))
    worker.start()
    img = np.zeros((4, 4, 3), np.uint8)
    id1 = worker.submit(img, "VERIFY_GRASP")
    r1 = _wait_result(worker)
    id2 = worker.submit(img, "VERIFY_GRASP")
    r2 = _wait_result(worker)
    worker.stop()
    assert (r1.request_id, r1.verdict) == (id1, Verdict.SLIPPING)
    assert (r2.request_id, r2.verdict) == (id2, Verdict.GRASPED)


def test_worker_timeout_and_busy() -> None:
    worker = CriticWorker(ScriptedCritic(["GRASPED"], delay_s=0.5), CriticConfig(timeout_s=0.1))
    worker.start()
    img = np.zeros((4, 4, 3), np.uint8)
    worker.submit(img, "VERIFY_GRASP")
    assert _wait_result(worker).verdict is Verdict.TIMEOUT
    worker.submit(img, "VERIFY_GRASP")  # abandoned call still running -> immediate TIMEOUT
    r = _wait_result(worker)
    worker.stop()
    assert r.verdict is Verdict.TIMEOUT
    assert r.latency_ms < 100


def test_worker_backend_exception_is_invalid() -> None:
    class Boom:
        def infer(self, image_bgr: np.ndarray, prompt: str) -> str:
            raise RuntimeError("cuda oom")

    worker = CriticWorker(Boom(), CriticConfig(timeout_s=1.0))
    worker.start()
    worker.submit(np.zeros((2, 2, 3), np.uint8), "VERIFY_HOLD")
    r = _wait_result(worker)
    worker.stop()
    assert r.verdict is Verdict.INVALID


# ------------------------------------------------------------------ runtime crop (M-1)
class _Recorder:
    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []

    def infer(self, image_bgr: np.ndarray, prompt: str) -> str:
        self.shapes.append(image_bgr.shape)
        return "GRASPED"


def _one(cfg: CriticConfig, img: np.ndarray, pre=None):  # type: ignore[no-untyped-def]
    rec = _Recorder()
    worker = CriticWorker(rec, cfg, preprocess=pre)
    worker.start()
    try:
        worker.submit(img, "VERIFY_GRASP")
        return _wait_result(worker), rec
    finally:
        worker.stop()


def test_defaults_are_bf16_and_378_crop() -> None:
    cfg = CriticConfig()
    assert (cfg.dtype, cfg.crop_enabled, cfg.crop_size_px, cfg.crop_box_norm) == (
        "bf16",
        True,
        378,
        None,
    )


def test_worker_crops_after_preprocess_and_reports_infer_time() -> None:
    img = np.zeros((480, 640, 3), np.uint8)
    seen: list[tuple[int, ...]] = []

    def dewarp(frame: np.ndarray) -> np.ndarray:
        seen.append(frame.shape)
        return frame

    r, rec = _one(CriticConfig(timeout_s=1.0), img, dewarp)
    assert seen == [(480, 640, 3)] and rec.shapes == [(378, 378, 3)]
    assert r.image_bgr.shape == (378, 378, 3) and r.verdict is Verdict.GRASPED
    assert r.infer_ms is not None and r.infer_ms <= r.latency_ms


def test_worker_crop_box_and_crop_off() -> None:
    img = np.zeros((480, 640, 3), np.uint8)
    img[100, 200] = (9, 9, 9)
    r, _ = _one(
        CriticConfig(timeout_s=1.0, crop_box_norm=(200 / 640, 100 / 480, 578 / 640, 478 / 480)), img
    )
    assert tuple(r.image_bgr[0, 0]) == (9, 9, 9)
    r, rec = _one(CriticConfig(timeout_s=1.0, crop_enabled=False), img)
    assert rec.shapes == [(480, 640, 3)]
