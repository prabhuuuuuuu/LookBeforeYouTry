"""Real-model tests. Skipped by default; run on the Jetson with:  pytest -m jetson -v

Weights: /home/sentrix/models/{moondream2,qwen2.5-vl-3b}, override with the
LBYR_MOONDREAM_PATH / LBYR_QWEN_PATH environment variables. Loads both models (about 11 GB).
"""

import time
from pathlib import Path

import numpy as np
import pytest

from lbyr.ai.critic import CRITIC_PROMPT, MODEL_CLASSES, Moondream2Critic, parse_verdict
from lbyr.ai.cuda_mem import cuda_allocated_mb
from lbyr.ai.planner import PlanRequest, QwenPlanner, parse_skill_chain, plan_with_timeout
from lbyr.config import CriticConfig, PlannerConfig
from lbyr.perception.detector import Detection

pytestmark = [pytest.mark.jetson, pytest.mark.usefixtures("cuda")]

DETS = [
    Detection(2, "red_block", 0.92, (150.0, 150.0, 250.0, 250.0)),
    Detection(5, "place_zone", 0.88, (400.0, 300.0, 560.0, 440.0)),
]


def _frame() -> np.ndarray:
    img = np.full((480, 640, 3), 235, dtype=np.uint8)
    img[150:250, 150:250] = (30, 30, 210)
    img[300:440, 400:560] = (70, 70, 70)
    return img


@pytest.fixture(scope="module")
def critic(moondream_path: Path) -> Moondream2Critic:
    if not moondream_path.is_dir():
        pytest.skip(f"no Moondream2 weights at {moondream_path}")
    return Moondream2Critic(CriticConfig(model_path=moondream_path))


@pytest.fixture(scope="module")
def planner(critic: Moondream2Critic, qwen_path: Path) -> QwenPlanner:
    # Depends on `critic`: Critic first, then Planner (co-residency, as in production).
    if not qwen_path.is_dir():
        pytest.skip(f"no Qwen weights at {qwen_path}")
    return QwenPlanner(PlannerConfig(model_path=qwen_path))


def test_critic_answers_one_class_within_its_timeout(critic: Moondream2Critic) -> None:
    t0 = time.perf_counter()
    text = critic.infer(_frame(), CRITIC_PROMPT)
    assert time.perf_counter() - t0 < CriticConfig().timeout_s
    assert parse_verdict(text) in MODEL_CLASSES, text


def test_critic_is_deterministic(critic: Moondream2Critic) -> None:
    assert critic.infer(_frame(), CRITIC_PROMPT) == critic.infer(_frame(), CRITIC_PROMPT)


def test_planner_full_generation_is_valid_strict_json(planner: QwenPlanner) -> None:
    req = PlanRequest("put the red block on the place zone", DETS, _frame())
    chain = parse_skill_chain(planner.generate_untimed(req), {2, 5})
    assert chain.approach.target_id == 2 and chain.place.target_id == 5
    assert planner.last_stats is not None and planner.last_stats.new_tokens > 0


def test_planner_never_runs_past_the_hard_timeout(planner: QwenPlanner) -> None:
    """Latency is hardware dependent (~11 s for a full plan on the Orin NX), so a timeout is
    an acceptable outcome; hanging past the 3 s budget is not."""
    req = PlanRequest("put the red block on the place zone", DETS, _frame())
    t0 = time.perf_counter()
    res = plan_with_timeout(planner, req, planner.cfg)
    assert time.perf_counter() - t0 < planner.cfg.timeout_s + 0.5
    assert res.chain is not None or res.error is not None
    time.sleep(1.5)  # abandoned generation stops itself about 0.6 s after the timeout


def test_both_models_are_resident_together(critic: Moondream2Critic, planner: QwenPlanner) -> None:
    mb = cuda_allocated_mb()
    assert mb is not None and mb > 9000  # ~4 GB Critic + ~7 GB Planner
