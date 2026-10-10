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
def test_centred_crop_is_one_moondream_tile() -> None:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[240, 320] = (1, 2, 3)  # frame centre
    out = critic_crop(img)
    assert out.shape == (MOONDREAM_TILE_PX, MOONDREAM_TILE_PX, 3) == (378, 378, 3)
    assert tuple(out[240 - 51, 320 - 131]) == (1, 2, 3)  # offsets (480-378)/2, (640-378)/2


def test_box_crop_is_clipped_and_resized() -> None:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    out = critic_crop(img, (600, 400, 200, 200))  # runs off the bottom-right corner
    assert out.shape == (378, 378, 3)
    assert critic_crop(np.zeros((240, 320, 3), dtype=np.uint8)).shape == (378, 378, 3)
    with pytest.raises(ValueError):
        critic_crop(img, (700, 10, 50, 50))


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
    args = bench.parse_args(["--critic-only", "--critic-dtype", "bf16", "--critic-crop"])
    assert args.skip_planner and args.critic_dtype == "bf16" and args.critic_crop
    assert bench.parse_args([]).critic_dtype == "fp16"
    with pytest.raises(SystemExit):
        bench.parse_args(["--critic-dtype", "int8"])


def test_variants_full_crop_and_compile_skip(capsys: pytest.CaptureFixture[str]) -> None:
    frames = [bench.synthetic_frame(i) for i in range(3)]
    args = bench.parse_args(["--critic-only", "--critic-crop", "--critic-compile", "--n", "3"])
    critic = _FakeCritic("this Moondream2 revision exposes no compile() method")
    variants, skipped = bench.run_critic_variants(critic, frames, args, _Mon())
    assert [v["variant"] for v in variants] == ["fp16-full", "fp16-crop378"]
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
    assert "fp16-crop378" in out and "p50 vs full" in out and "no compile()" in out


def test_compiled_variants_run_after_compile() -> None:
    frames = [bench.synthetic_frame(i) for i in range(2)]
    args = bench.parse_args(["--critic-only", "--critic-compile", "--n", "2"])
    variants, skipped = bench.run_critic_variants(_FakeCritic(None), frames, args, _Mon())
    assert [v["variant"] for v in variants] == ["fp16-full", "fp16-full-compiled"]
    assert skipped == []
