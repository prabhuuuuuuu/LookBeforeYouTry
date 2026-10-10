"""tools/bench_models.py Critic variants and the crop helper, with fakes (no torch)."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from lbyr.ai.critic import Moondream2Critic
from lbyr.ai.imaging import MOONDREAM_TILE_PX, critic_crop
from lbyr.config import CriticConfig
from lbyr.errors import ConfigError

pytest.importorskip("PIL")


def _load_bench() -> ModuleType:
    path = Path(__file__).parents[1] / "tools" / "bench_models.py"
    spec = importlib.util.spec_from_file_location("bench_models", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench_models"] = mod
    spec.loader.exec_module(mod)
    return mod


bench = _load_bench()


# ------------------------------------------------------------------ crop helper
def test_default_crop_is_centred_square_resized_to_one_tile() -> None:
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    img[:, 420:1500] = 200  # exactly the width-centred 1080x1080 square
    out = critic_crop(img)
    assert out.shape == (MOONDREAM_TILE_PX, MOONDREAM_TILE_PX, 3) == (378, 378, 3)
    assert out.min() == 200  # the whole square, resized (not a raw 378-px centre crop)


def test_normalized_box_is_clipped_and_resized() -> None:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    out = critic_crop(img, (0.9, 0.8, 1.3, 1.4))  # runs off the bottom-right corner
    assert out.shape == (378, 378, 3)
    assert critic_crop(np.zeros((240, 320, 3), dtype=np.uint8)).shape == (378, 378, 3)
    with pytest.raises(ValueError):
        critic_crop(img, (1.1, 0.0, 1.2, 0.1))


# ------------------------------------------------------------------ dtype / compile
def test_critic_dtype_default_bf16_fp16_selectable() -> None:
    assert CriticConfig().dtype == "bf16"  # M-1
    CriticConfig(dtype="fp16").validate()
    with pytest.raises(ConfigError):
        CriticConfig(dtype="int8").validate()


class _Inner:
    def __init__(self, fail: bool = False) -> None:
        self.compiled = False
        self.fail = fail

    def compile(self) -> None:
        if self.fail:
            raise RuntimeError("no triton")
        self.compiled = True


class _Model:
    def __init__(self, inner: Any = None) -> None:
        if inner is not None:
            self.model = inner

    def query(self, image: Any, question: str, settings: Any = None) -> dict[str, str]:
        return {"answer": "GRASPED"}


def test_try_compile_skips_or_compiles() -> None:
    assert "no compile()" in (Moondream2Critic(CriticConfig(), model=_Model()).try_compile() or "")
    inner = _Inner()
    assert Moondream2Critic(CriticConfig(), model=_Model(inner)).try_compile() is None
    assert inner.compiled
    bad = Moondream2Critic(CriticConfig(), model=_Model(_Inner(fail=True)))
    assert "no triton" in (bad.try_compile() or "")


# ------------------------------------------------------------------ bench variants
class _FakeCritic:
    def __init__(self, compile_reason: str | None) -> None:
        self.shapes: list[tuple[int, ...]] = []
        self.compile_reason = compile_reason

    def infer(self, image_bgr: np.ndarray, prompt: str) -> str:
        self.shapes.append(image_bgr.shape)
        return "GRASPED"

    def try_compile(self) -> str | None:
        return self.compile_reason


class _Mon:
    peak_mb: float | None = 5000.0

    def reset_peak(self) -> None:
        self.peak_mb = 5000.0


def test_critic_only_flag_and_dtype_choices() -> None:
    args = bench.parse_args(["--critic-only", "--critic-dtype", "fp16", "--critic-crop"])
    assert args.planner_list == [] and args.critic_dtype == "fp16" and args.critic_crop
    assert bench.parse_args([]).critic_dtype == "bf16"
    with pytest.raises(SystemExit):
        bench.parse_args(["--critic-dtype", "int8"])


def test_variants_full_crop_and_compile_skip(capsys: pytest.CaptureFixture[str]) -> None:
    frames = [bench.synthetic_frame(i) for i in range(3)]
    args = bench.parse_args(["--critic-only", "--critic-crop", "--critic-compile", "--n", "3"])
    critic = _FakeCritic("this Moondream2 revision exposes no compile() method")
    variants, skipped = bench.run_critic_variants(critic, frames, args, _Mon())
    assert [v["variant"] for v in variants] == ["bf16-full", "bf16-crop378"]
    assert [v["input"] for v in variants] == ["640x480", "378x378"]
    assert all(v["latency_ms"]["n"] == 3 and v["verdicts"] == {"GRASPED": 3} for v in variants)
    assert skipped and "no compile()" in skipped[0]
    result = {
        "critic_variants": variants,
        "critic_skipped": skipped,
        "n_warm_calls": 3,
        "images": "synthetic",
    }
    bench.print_critic_variants(result)
    out = capsys.readouterr().out
    assert "bf16-crop378" in out and "p50 vs first" in out and "no compile()" in out


def test_compiled_variants_run_after_compile() -> None:
    frames = [bench.synthetic_frame(i) for i in range(2)]
    args = bench.parse_args(["--critic-only", "--critic-compile", "--n", "2"])
    variants, skipped = bench.run_critic_variants(_FakeCritic(None), frames, args, _Mon())
    assert [v["variant"] for v in variants] == ["bf16-crop378", "bf16-crop378-compiled"]
    assert skipped == []


def test_default_critic_variant_is_the_runtime_crop() -> None:
    frames = [bench.synthetic_frame(i) for i in range(2)]
    args = bench.parse_args(["--critic-only", "--n", "2"])
    critic = _FakeCritic(None)
    variants, _ = bench.run_critic_variants(critic, frames, args, _Mon())
    assert [v["variant"] for v in variants] == ["bf16-crop378"]
    assert set(critic.shapes) == {(378, 378, 3)}


# ------------------------------------------------------------------ planner models (M-2)
def test_planner_models_args() -> None:
    args = bench.parse_args(["--planner-models", "qwen3-1.7b,qwen3-0.6b", "--combined"])
    assert args.planner_list == ["qwen3-1.7b", "qwen3-0.6b"] and args.combined
    with pytest.raises(SystemExit):
        bench.parse_args(["--planner-models", "smolvlm2"])
    with pytest.raises(SystemExit):
        bench.parse_args(["--combined", "--skip-critic"])


class _FakeLangPlanner:
    """Answers the compact schema; gets the 'blue' commands wrong / invalid on purpose."""

    def __init__(self) -> None:
        from lbyr.ai.planner import GenStats

        self._stats = GenStats
        self.last_stats: Any = None

    def generate_language_untimed(self, request: Any) -> str:
        self.last_stats = self._stats(19, 0.5)
        text = request.instruction.lower()
        if "blue" in text:
            return "the blue cube" if "zone b" in text else '{"a":1,"s":40,"g":80,"p":7}'
        pick = next(i for kw, i in (("red", 1), ("sponge", 3), ("acrylic", 4)) if kw in text)
        place = 7 if "zone a" in text else 8
        return f'{{"a":{pick},"s":40,"g":80,"p":{place}}}'


def test_bench_language_planner_rates() -> None:
    out = bench.bench_language_planner(_FakeLangPlanner())
    assert len(bench.LANGUAGE_CASES) == 10
    blue = sum("blue" in c.instruction.lower() for c in bench.LANGUAGE_CASES)
    blue_b = sum(
        "blue" in c.instruction.lower() and "zone b" in c.instruction.lower()
        for c in bench.LANGUAGE_CASES
    )
    assert out["valid_parse_fraction"] == pytest.approx((10 - blue_b) / 10)
    assert out["correct_fraction"] == pytest.approx((10 - blue) / 10)
    assert out["tokens_per_s"]["p50"] == pytest.approx(38.0)
    assert out["within_timeout_fraction"] == 1.0
    assert len(out["samples"]) == 10


def test_language_cases_resolve_against_the_detection_list() -> None:
    ids = {d.id: d.label for d in bench.language_detections()}
    for c in bench.LANGUAGE_CASES:
        assert c.pick_id in ids and "zone" not in ids[c.pick_id]
        assert "zone" in ids[c.place_id]


def test_missing_planner_weights_are_skipped(tmp_path: Path) -> None:
    out = bench.bench_planner_model("qwen3-0.6b", tmp_path, _Mon())
    assert "skipped" in out
    bench.print_planner_table([out])


# ------------------------------------------------------------------ Critic accuracy
def test_classification_report() -> None:
    y_true = ["GRASPED", "GRASPED", "EMPTY", "SLIPPING", "OCCLUDED"]
    y_pred = ["GRASPED", "SLIPPING", "EMPTY", "SLIPPING", "INVALID"]
    r = bench.classification_report(y_true, y_pred)
    assert r["accuracy"] == pytest.approx(3 / 5)
    assert r["labels"][-1] == "INVALID"
    assert r["confusion"]["GRASPED"]["SLIPPING"] == 1
    assert r["confusion"]["OCCLUDED"]["INVALID"] == 1
    g, s = r["per_class"]["GRASPED"], r["per_class"]["SLIPPING"]
    assert (g["precision"], g["recall"]) == (1.0, 0.5)
    assert (s["precision"], s["recall"]) == (0.5, 1.0)
    assert r["per_class"]["OCCLUDED"]["f1"] == 0.0


def test_accuracy_on_dataset_and_warning_without(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from PIL import Image

    assert bench.critic_accuracy(_FakeCritic(None), tmp_path) is None
    bench.print_accuracy(None, tmp_path)
    assert "NOT measured" in capsys.readouterr().out
    for cls in ("GRASPED", "EMPTY"):
        (tmp_path / cls).mkdir()
        Image.new("RGB", (640, 480)).save(tmp_path / cls / "a.jpg")
    critic = _FakeCritic(None)  # always answers GRASPED
    acc = bench.critic_accuracy(critic, tmp_path)
    assert acc is not None and acc["n"] == 2 and acc["accuracy"] == 0.5
    assert set(critic.shapes) == {(378, 378, 3)}  # runtime crop applied
    bench.print_accuracy(acc, tmp_path)
    assert "Confusion matrix" in capsys.readouterr().out
