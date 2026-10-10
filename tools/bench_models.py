"""Benchmark the on-device models: Moondream2 Critic and Qwen2.5-VL planner.

WHY
    Bible 9.2 budgets VRAM and latency for both models. This measures the real numbers on
    the Jetson so the budget (and the planner's 3 s hard timeout) can be checked.

HOW TO USE (on the Jetson, weights already downloaded)
        python tools/bench_models.py
        python tools/bench_models.py --images data/frames/wrist --n 20
        python tools/bench_models.py --critic-only         # Critic only (= --skip-planner)
    Without --images, synthetic frames are used. With --images, the same folder feeds both
    models (the planner also gets a fixed synthetic detections list).

CRITIC SPEED EXPERIMENTS (each variant is reported separately; defaults stay unchanged)
        --critic-dtype fp16|bf16   fp16 = weights cast + crop-dtype patch (current default);
                                   bf16 = Moondream's native dtype, no patch
        --critic-crop              also run a 378x378 crop (gripper ROI from
                                   WristCameraConfig.crop_xywh, default centred) vs full frame
        --critic-compile           then also run after Moondream's own compile(), if this
                                   revision has one (skipped with a message otherwise)
    Per variant: p50/p95 latency, torch peak (max_memory_allocated, includes the weights)
    and tegrastats system-RAM peak during that variant.

OUTPUT
    A table matching bible 9.2 (measured next to the bible estimate) and a JSON file in
    data/bench/. Planner latency is the FULL generation (no 3 s cut-off), so you can see
    whether it can meet the 3 s hard timeout; the share of calls within 3 s is reported.

SAFETY
    Never talks to the robot arm. Loads Critic first, then Planner (co-residency check).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_CRITIC = Path(os.environ.get("LBYR_MOONDREAM_PATH", "/home/sentrix/models/moondream2"))
DEFAULT_PLANNER = Path(os.environ.get("LBYR_QWEN_PATH", "/home/sentrix/models/qwen2.5-vl-3b"))
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
PLANNER_HARD_TIMEOUT_S = 3.0
CRITIC_DTYPES = ("fp16", "bf16")  # mirrors lbyr.config.CRITIC_DTYPES (keeps --help light)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--critic-path", type=Path, default=DEFAULT_CRITIC)
    p.add_argument("--planner-path", type=Path, default=DEFAULT_PLANNER)
    p.add_argument(
        "--images", type=Path, default=None, help="folder of images (default: synthetic)"
    )
    p.add_argument("--n", type=int, default=10, help="warm calls per model (default 10)")
    p.add_argument("--out", type=Path, default=Path("data/bench"), help="result JSON folder")
    p.add_argument("--skip-critic", action="store_true")
    p.add_argument("--skip-planner", action="store_true")
    p.add_argument("--critic-only", action="store_true", help="same as --skip-planner")
    p.add_argument("--max-new-tokens", type=int, default=256, help="planner generation cap")
    p.add_argument(
        "--critic-dtype",
        choices=CRITIC_DTYPES,
        default="fp16",
        help="Critic weights dtype (default fp16 = current config default)",
    )
    p.add_argument(
        "--critic-crop",
        action="store_true",
        help="also benchmark a 378x378 crop (default centred) against the full frame",
    )
    p.add_argument(
        "--critic-compile",
        action="store_true",
        help="also benchmark after Moondream's compile(), if the revision exposes one",
    )
    args = p.parse_args(argv)
    if args.n < 1:
        p.error("--n must be >= 1")
    if args.critic_only:
        if args.skip_critic:
            p.error("--critic-only and --skip-critic contradict each other")
        args.skip_planner = True
    return args


# ------------------------------------------------------------------ helpers
def percentile_stats(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {
        "n": float(a.size),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
    }


def synthetic_frame(i: int, w: int = 640, h: int = 480) -> np.ndarray:
    """BGR frame with a red block and a dark zone; slightly different every call."""
    rng = np.random.default_rng(i)
    img = np.full((h, w, 3), 235, dtype=np.uint8)
    img += rng.integers(0, 12, img.shape, dtype=np.uint8) // 2
    x = 150 + (i * 17) % 100
    img[150:250, x : x + 100] = (30, 30, 210)
    img[300:440, 400:560] = (70, 70, 70)
    return img


def load_images(folder: Path | None, n: int) -> tuple[list[np.ndarray], str]:
    if folder is None:
        return [synthetic_frame(i) for i in range(n)], "synthetic"
    from PIL import Image

    files = sorted(f for f in folder.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise SystemExit(f"no images ({', '.join(sorted(IMAGE_SUFFIXES))}) in {folder}")
    imgs = [
        np.asarray(Image.open(files[i % len(files)]).convert("RGB"))[..., ::-1] for i in range(n)
    ]
    return [np.ascontiguousarray(im) for im in imgs], f"{folder} ({len(files)} files)"


def nvpmodel_query() -> str | None:
    exe = shutil.which("nvpmodel")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "-q"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return " ".join(out.stdout.split()) or None


def environment() -> dict[str, Any]:
    env: dict[str, Any] = {"python": sys.version.split()[0], "machine": platform.machine()}
    try:
        import torch
        import transformers

        env.update(torch=torch.__version__, transformers=transformers.__version__)
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["device"] = torch.cuda.get_device_name(0)
    except ImportError:
        env["torch"] = None
    env["nvpmodel"] = nvpmodel_query()
    return env


# ------------------------------------------------------------------ benchmarks
def bench_critic(
    critic: Any,
    frames: list[np.ndarray],
    crop: bool = False,
    crop_box: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    """Time ``critic.infer`` (including the crop, if any) on every frame."""
    from lbyr.ai.critic import CRITIC_PROMPT, parse_verdict
    from lbyr.ai.imaging import critic_crop

    lat_ms: list[float] = []
    answers: list[str] = []
    shape = frames[0].shape
    for f in frames:
        t0 = time.perf_counter()
        img = critic_crop(f, crop_box) if crop else f
        text = critic.infer(img, CRITIC_PROMPT)
        lat_ms.append((time.perf_counter() - t0) * 1000)
        answers.append(parse_verdict(text).value)
        shape = img.shape
    return {
        "input": f"{shape[1]}x{shape[0]}",
        "latency_ms": percentile_stats(lat_ms),
        "verdicts": {v: answers.count(v) for v in sorted(set(answers))},
    }


def _critic_variant(
    name: str,
    critic: Any,
    frames: list[np.ndarray],
    crop: bool,
    crop_box: tuple[int, int, int, int] | None,
    mon: Any,
) -> dict[str, Any]:
    from lbyr.ai.critic import CRITIC_PROMPT
    from lbyr.ai.cuda_mem import cuda_peak_mb, reset_cuda_peak
    from lbyr.ai.imaging import critic_crop

    print(f"Critic [{name}]: 1 warm-up + {len(frames)} timed calls ...")
    critic.infer(critic_crop(frames[0], crop_box) if crop else frames[0], CRITIC_PROMPT)
    reset_cuda_peak()
    mon.reset_peak()
    out = bench_critic(critic, frames, crop, crop_box)
    time.sleep(0.3)  # let tegrastats take one more sample
    out["variant"] = name
    out["torch_peak_mb"] = cuda_peak_mb()
    out["tegrastats_peak_mb"] = mon.peak_mb
    return out


def run_critic_variants(
    critic: Any, frames: list[np.ndarray], args: argparse.Namespace, mon: Any
) -> tuple[list[dict[str, Any]], list[str]]:
    """Full frame, then (--critic-crop) 378x378 crop, then (--critic-compile) the same
    after Moondream's compile(). Returns (variants, skip messages)."""
    from lbyr.ai.critic import CRITIC_PROMPT
    from lbyr.ai.imaging import critic_crop
    from lbyr.config import WristCameraConfig

    crop_box = WristCameraConfig().crop_xywh  # OQ-12 gripper ROI; None = centred
    inputs = [("full", False)] + ([("crop378", True)] if args.critic_crop else [])
    variants = [
        _critic_variant(f"{args.critic_dtype}-{tag}", critic, frames, crop, crop_box, mon)
        for tag, crop in inputs
    ]
    skipped: list[str] = []
    if args.critic_compile:
        reason = critic.try_compile()
        if reason is not None:
            skipped.append(f"--critic-compile skipped: {reason}")
        else:
            for tag, crop in inputs:
                name = f"{args.critic_dtype}-{tag}-compiled"
                try:
                    for f in frames[:2]:  # compilation happens lazily on the first calls
                        critic.infer(critic_crop(f, crop_box) if crop else f, CRITIC_PROMPT)
                    variants.append(_critic_variant(name, critic, frames, crop, crop_box, mon))
                except Exception as e:  # torch.compile can fail lazily (e.g. no Triton)
                    skipped.append(f"{name} skipped: {type(e).__name__}: {e}")
                    break
    for msg in skipped:
        print(msg)
    return variants, skipped


def bench_planner(planner: Any, frames: list[np.ndarray]) -> dict[str, Any]:
    from lbyr.ai.planner import PlanRequest, parse_skill_chain
    from lbyr.perception.detector import Detection

    dets = [
        Detection(2, "red_block", 0.92, (150.0, 150.0, 250.0, 250.0)),
        Detection(3, "wooden_block", 0.81, (40.0, 300.0, 120.0, 380.0)),
        Detection(5, "place_zone", 0.88, (400.0, 300.0, 560.0, 440.0)),
    ]
    lat_s: list[float] = []
    tok_s: list[float] = []
    toks: list[int] = []
    valid = 0
    for f in frames:
        req = PlanRequest("put the red block on the place zone", dets, f)
        t0 = time.perf_counter()
        text = planner.generate_untimed(req)  # full generation, no cut-off
        lat_s.append(time.perf_counter() - t0)
        stats = planner.last_stats
        if stats is not None:
            tok_s.append(stats.tokens_per_s)
            toks.append(stats.new_tokens)
        try:
            parse_skill_chain(text, {d.id for d in dets})
            valid += 1
        except Exception:
            pass
    return {
        "latency_s": percentile_stats(lat_s),
        "tokens_per_s": percentile_stats(tok_s) if tok_s else None,
        "new_tokens": percentile_stats([float(t) for t in toks]) if toks else None,
        "valid_json_fraction": valid / len(frames),
        "within_hard_timeout_fraction": sum(t <= PLANNER_HARD_TIMEOUT_S for t in lat_s)
        / len(lat_s),
        "hard_timeout_s": PLANNER_HARD_TIMEOUT_S,
    }


# ------------------------------------------------------------------ report
def print_table(result: dict[str, Any]) -> None:
    mem = result["memory"]
    rows: list[tuple[str, str, str, str, str]] = []
    c = result.get("critic")
    if c:
        lat = c["latency_ms"]
        rows.append(
            (
                "Critic",
                f"Moondream2 {result.get('critic_dtype', 'fp16').upper()}, {c.get('input', '?')}",
                f"{mem['critic_alloc_mb'] / 1024:.1f} GB (bible ~4)"
                if mem.get("critic_alloc_mb") is not None
                else "n/a",
                f"p50 {lat['p50'] / 1000:.2f} s / p95 {lat['p95'] / 1000:.2f} s (bible 0.3-0.8 s)",
                "event-driven",
            )
        )
    pl = result.get("planner")
    if pl:
        lat = pl["latency_s"]
        tps = pl["tokens_per_s"]
        planner_gb = None
        if mem.get("planner_alloc_mb") is not None:
            planner_gb = (mem["planner_alloc_mb"] - (mem.get("critic_alloc_mb") or 0.0)) / 1024
        rows.append(
            (
                "Planner",
                "Qwen2.5-VL-3B FP16",
                f"{planner_gb:.1f} GB (bible ~6.5)" if planner_gb is not None else "n/a",
                f"p50 {lat['p50']:.2f} s / p95 {lat['p95']:.2f} s (bible 0.8-2 s), "
                f"{tps['p50']:.1f} tok/s"
                if tps
                else f"p50 {lat['p50']:.2f} s",
                f"on demand; {pl['within_hard_timeout_fraction'] * 100:.0f}% within 3 s",
            )
        )
    head = ("Component", "Model / precision", "VRAM (torch alloc)", "Latency", "Rate")
    widths = [max(len(head[i]), *(len(r[i]) for r in rows)) for i in range(5)] if rows else []
    for line in (head, *rows):
        print("| " + " | ".join(line[i].ljust(widths[i]) for i in range(5)) + " |")
        if line is head:
            print("|-" + "-|-".join("-" * w for w in widths) + "-|")
    print()
    peak = mem.get("torch_peak_mb")
    print(f"torch.cuda.max_memory_allocated: {f'{peak:.0f} MB' if peak is not None else 'n/a'}")
    ram = mem.get("tegrastats_ram_mb")
    if ram:
        print(
            f"tegrastats system RAM (unified memory): {ram['baseline']:.0f} MB before load, "
            f"peak {ram['peak']:.0f} MB (+{ram['peak'] - ram['baseline']:.0f} MB)"
        )
    else:
        print("tegrastats: not available")


def _gb(mb: float | None) -> str:
    return f"{mb / 1024:.2f} GB" if mb is not None else "n/a"


def print_critic_variants(result: dict[str, Any]) -> None:
    """One row per Critic variant; p50 is also compared with the same dtype's full frame."""
    variants = result.get("critic_variants") or []
    if not variants:
        return
    full = variants[0]["latency_ms"]["p50"]
    head = ("Variant", "Input", "p50", "p95", "p50 vs full", "torch peak", "RAM peak", "Verdicts")
    rows = [
        (
            v["variant"],
            v["input"],
            f"{v['latency_ms']['p50'] / 1000:.3f} s",
            f"{v['latency_ms']['p95'] / 1000:.3f} s",
            f"x{v['latency_ms']['p50'] / full:.2f}",
            _gb(v.get("torch_peak_mb")),
            _gb(v.get("tegrastats_peak_mb")),
            ",".join(f"{k}:{n}" for k, n in v["verdicts"].items()),
        )
        for v in variants
    ]
    widths = [max(len(head[i]), *(len(r[i]) for r in rows)) for i in range(len(head))]
    print(f"Critic variants ({result['n_warm_calls']} timed calls each, {result['images']})")
    for line in (head, *rows):
        print("| " + " | ".join(line[i].ljust(widths[i]) for i in range(len(head))) + " |")
        if line is head:
            print("|-" + "-|-".join("-" * w for w in widths) + "-|")
    for msg in result.get("critic_skipped") or []:
        print(f"note: {msg}")
    print("torch peak = max_memory_allocated (weights included); RAM peak = tegrastats used.")
    print()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if args.skip_critic and args.skip_planner:
        print("nothing to do: both models skipped")
        return 2

    from lbyr.ai.offline import force_offline

    force_offline()  # before torch / transformers are imported anywhere below
    from lbyr.ai.cuda_mem import cuda_allocated_mb, cuda_peak_mb
    from lbyr.config import CriticConfig, PlannerConfig
    from lbyr.logging_.episodes import VramMonitor

    mon = VramMonitor(200)
    mon.start()
    time.sleep(0.5)
    ram_before = mon.latest_mb
    result: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "environment": environment(),
        "n_warm_calls": args.n,
    }
    frames, source = load_images(args.images, args.n)
    result["images"] = source
    memory: dict[str, Any] = {}
    result["memory"] = memory

    critic = planner = None
    # Load order matters: Critic first, then Planner (co-residency, lbyr.ai.models).
    if not args.skip_critic:
        from lbyr.ai.critic import Moondream2Critic

        critic = Moondream2Critic(
            CriticConfig(model_path=args.critic_path, dtype=args.critic_dtype)
        )
        memory["critic_alloc_mb"] = cuda_allocated_mb()
        result["critic_dtype"] = args.critic_dtype
    if not args.skip_planner:
        from lbyr.ai.planner import QwenPlanner

        planner = QwenPlanner(
            PlannerConfig(model_path=args.planner_path, max_new_tokens=args.max_new_tokens)
        )
        memory["planner_alloc_mb"] = cuda_allocated_mb()

    # Per-variant windows reset the peaks, so keep every window's peak for the overall one.
    torch_peaks: list[float | None] = [cuda_peak_mb()]
    ram_peaks: list[float | None] = [mon.peak_mb]
    if critic is not None:
        variants, skipped = run_critic_variants(critic, frames, args, mon)
        result["critic_variants"] = variants
        result["critic_skipped"] = skipped
        result["critic"] = variants[0]  # full frame: the bible 9.2 row
        torch_peaks += [v["torch_peak_mb"] for v in variants]
        ram_peaks += [v["tegrastats_peak_mb"] for v in variants]
    if planner is not None:
        print(f"Planner: {args.n} warm calls (full generation, can take ~10 s each) ...")
        result["planner"] = bench_planner(planner, frames)

    torch_peaks.append(cuda_peak_mb())
    time.sleep(0.5)
    mon.stop()
    ram_peaks.append(mon.peak_mb)
    known_torch = [p for p in torch_peaks if p is not None]
    known_ram = [p for p in ram_peaks if p is not None]
    memory["torch_peak_mb"] = max(known_torch) if known_torch else None
    if known_ram and ram_before is not None:
        memory["tegrastats_ram_mb"] = {"baseline": ram_before, "peak": max(known_ram)}

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"bench-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(result, indent=2))
    print()
    print_critic_variants(result)
    print_table(result)
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
