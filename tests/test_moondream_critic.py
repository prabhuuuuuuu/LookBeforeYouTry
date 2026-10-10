"""Moondream2Critic with a fake model object (no torch, no weights)."""

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lbyr.ai.critic import CRITIC_PROMPT, CriticWorker, Moondream2Critic, Verdict
from lbyr.ai.imaging import bgr_to_pil
from lbyr.config import CriticConfig
from lbyr.errors import ConfigError, LBYRError

pytest.importorskip("PIL")


class FakeMoondream:
    """Mimics ``model.query(image, question, settings=...) -> {"answer": str}``."""

    def __init__(self, answers: list[str], delay_s: float = 0.0) -> None:
        self.answers = answers
        self.delay_s = delay_s
        self.calls: list[tuple[Any, str, dict[str, Any]]] = []

    def query(self, image: Any, question: str, settings: dict[str, Any] | None = None) -> dict:
        self.calls.append((image, question, settings or {}))
        if self.delay_s:
            time.sleep(self.delay_s)
        return {"answer": self.answers[min(len(self.calls) - 1, len(self.answers) - 1)]}


def _wait(worker: CriticWorker, timeout_s: float = 3.0) -> Any:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        res = worker.poll()
        if res is not None:
            return res
        time.sleep(0.01)
    raise AssertionError("no verdict")


def test_warmup_runs_once_with_locked_prompt_and_greedy_settings() -> None:
    fake = FakeMoondream(["GRASPED"])
    Moondream2Critic(CriticConfig(), model=fake)
    assert len(fake.calls) == 1
    _, question, settings = fake.calls[0]
    assert question == CRITIC_PROMPT
    assert settings == {"temperature": 0.0, "max_tokens": 8}


def test_infer_converts_bgr_to_rgb_pil_and_strips_answer() -> None:
    fake = FakeMoondream(["GRASPED", "  Slipping.\n"])
    critic = Moondream2Critic(CriticConfig(), model=fake)
    bgr = np.zeros((4, 6, 3), dtype=np.uint8)
    bgr[0, 0] = (10, 20, 30)  # B, G, R
    assert critic.infer(bgr, CRITIC_PROMPT) == "Slipping."
    image = fake.calls[-1][0]
    assert image.mode == "RGB" and image.size == (6, 4)
    assert image.getpixel((0, 0)) == (30, 20, 10)


def test_worker_pipeline_parses_one_word_verdicts() -> None:
    fake = FakeMoondream(["GRASPED", "slipping.", "The object is grasped."])
    worker = CriticWorker(Moondream2Critic(CriticConfig(), model=fake), CriticConfig())
    worker.start()
    try:
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        verdicts = []
        for _ in range(2):
            worker.submit(img, "VERIFY_GRASP")
            verdicts.append(_wait(worker).verdict)
        assert verdicts == [Verdict.SLIPPING, Verdict.INVALID]
    finally:
        worker.stop()


def test_worker_timeout_is_explicit() -> None:
    fake = FakeMoondream(["GRASPED"])
    cfg = CriticConfig(timeout_s=0.1)
    critic = Moondream2Critic(cfg, model=fake)  # warm-up is fast
    fake.delay_s = 0.6
    worker = CriticWorker(critic, cfg)
    worker.start()
    try:
        worker.submit(np.zeros((8, 8, 3), dtype=np.uint8), "VERIFY_GRASP")
        res = _wait(worker)
        assert res.verdict is Verdict.TIMEOUT
        assert res.latency_ms < 500
    finally:
        worker.stop()


def test_warmup_timeout_raises() -> None:
    with pytest.raises(LBYRError, match="warm-up"):
        Moondream2Critic(
            CriticConfig(warmup_timeout_s=0.05), model=FakeMoondream(["GRASPED"], delay_s=0.5)
        )


def test_refuses_to_load_without_local_model_path() -> None:
    with pytest.raises(ConfigError, match="model_path"):
        Moondream2Critic(CriticConfig())


def test_lora_not_silently_ignored() -> None:
    with pytest.raises(NotImplementedError):
        Moondream2Critic(CriticConfig(lora_adapter_path=Path("x")), model=FakeMoondream(["A"]))


def test_temperature_must_be_zero() -> None:
    with pytest.raises(ConfigError):
        Moondream2Critic(CriticConfig(temperature=0.5), model=FakeMoondream(["A"]))


def test_bgr_to_pil_rejects_non_color() -> None:
    with pytest.raises(ValueError):
        bgr_to_pil(np.zeros((4, 4), dtype=np.uint8))
