"""Benchmark the on-device models: Moondream2 Critic and the language-mode planners.

WHY
    Bible 9.2 budgets memory and latency. This measures the real numbers on the Jetson:
    the Critic at its runtime input (bf16, 378x378 crop), the candidate language-mode
    planners (M-2), and both loaded together (combined RAM peak, memory guard).

HOW TO USE (on the Jetson, weights already downloaded, nothing else running)
        python tools/bench_models.py --critic-only                  # Critic (runtime input)
        python tools/bench_models.py --planner-models qwen3-1.7b,qwen3-0.6b --combined
        python tools/bench_models.py --critic-only --critic-crop    # full frame vs crop
    Planner models (folders under --models-dir, default /home/sentrix/models):
        qwen3-1.7b, qwen3-0.6b    Qwen3 text LLM, thinking disabled (language_backend qwen3_text)
        qwen2.5-vl-3b             Qwen2.5-VL used text-only (language_backend qwen_vl)
    Models without weights are reported as skipped.

WHAT IS MEASURED
    Planners (each loaded ALONE, then unloaded): load + warm-up time, p50/p95 latency of the
    full generation (no 8 s cut-off), tokens/s, valid-parse rate and correct-target rate on
    10 fixed language commands against a fixed detection list, torch memory and system RAM.
    --combined: Critic (config defaults: bf16, 378x378 crop) + the chosen planner loaded
    together (Critic first, as in production): combined RAM peak, latency of both, and the
    memory guard verdict (planner.qwen_min_free_ram_mb).
    Critic: p50/p95 and peak memory per variant (--critic-dtype, --critic-crop, --critic-compile).
    Critic accuracy: if data/datasets/critic/<CLASS>/*.jpg exists, accuracy, per-class
    precision/recall/F1 and a confusion matrix; otherwise a warning that accuracy is NOT
    measured. Dataset images are treated as dewarped wrist frames (the crop is applied).
    --vl-full-plan: the old in-loop Qwen2.5-VL full-JSON benchmark (opt-in; 11 s per plan).

OUTPUT
    Markdown tables on stdout and a JSON file in data/bench/.

SAFETY
    Never talks to the robot arm. Edge-only: HF_HUB_OFFLINE is forced, local weights only.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MODELS_DIR = Path(os.environ.get("LBYR_MODELS_DIR", "/home/sentrix/models"))
DEFAULT_CRITIC = Path(os.environ.get("LBYR_MOONDREAM_PATH", str(MODELS_DIR / "moondream2")))
DEFAULT_PLANNER = Path(os.environ.get("LBYR_QWEN_PATH", str(MODELS_DIR / "qwen2.5-vl-3b")))
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
PLANNER_HARD_TIMEOUT_S = 3.0
CRITIC_DTYPES = ("bf16", "fp16")  # mirrors lbyr.config.CRITIC_DTYPES (keeps --help light)
CRITIC_CLASSES = ("GRASPED", "SLIPPING", "EMPTY", "OCCLUDED")  # lbyr.ai.critic.MODEL_CLASSES
# name -> language_backend (folder = --models-dir / name)
PLANNER_MODELS: dict[str, str] = {
    "qwen3-1.7b": "qwen3_text",
    "qwen3-0.6b": "qwen3_text",
    "qwen2.5-vl-3b": "qwen_vl",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--critic-path", type=Path, default=DEFAULT_CRITIC)
    p.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="planner model folders")
    p.add_argument("--planner-path", type=Path, default=DEFAULT_PLANNER, help="for --vl-full-plan")
    p.add_argument(
        "--images", type=Path, default=None, help="folder of wrist images (default: synthetic)"
    )
    p.add_argument("--n", type=int, default=10, help="timed Critic calls per variant")
    p.add_argument("--out", type=Path, default=Path("data/bench"), help="result JSON folder")
    p.add_argument("--dataset", type=Path, default=Path("data/datasets/critic"))
    p.add_argument("--skip-critic", action="store_true")
    p.add_argument("--critic-only", action="store_true", help="no planners at all")
    p.add_argument(
        "--planner-models",
        default="",
        help=f"comma-separated, from {', '.join(PLANNER_MODELS)}",
    )
    p.add_argument(
        "--combined",
        action="store_true",
        help="Critic + chosen planner loaded together (combined RAM peak, memory guard)",
    )
    p.add_argument(
        "--combined-planner",
        default=None,
        help="planner for --combined (default: first of --planner-models with weights)",
    )
    p.add_argument("--vl-full-plan", action="store_true", help="old Qwen2.5-VL in-loop bench")
    p.add_argument("--max-new-tokens", type=int, default=256, help="--vl-full-plan cap")
    p.add_argument(
        "--critic-dtype",
        choices=CRITIC_DTYPES,
        default="bf16",
        help="Critic weights dtype (default bf16 = config default)",
    )
    p.add_argument(
        "--critic-crop",
        action="store_true",
        help="also benchmark the full frame next to the runtime 378x378 crop",
    )
    p.add_argument(
        "--critic-compile",
        action="store_true",
        help="also benchmark after Moondream's compile(), if the revision exposes one",
    )
    args = p.parse_args(argv)
    if args.n < 1:
        p.error("--n must be >= 1")
    names = [m.strip() for m in args.planner_models.split(",") if m.strip()]
    unknown = [m for m in names if m not in PLANNER_MODELS]
    if unknown:
        p.error(f"unknown --planner-models {unknown}; choose from {list(PLANNER_MODELS)}")
    args.planner_list = names
    if args.critic_only:
        if args.skip_critic:
            p.error("--critic-only and --skip-critic contradict each other")
        args.planner_list = []
        args.combined = False
        args.vl_full_plan = False
    if args.combined and args.skip_critic:
        p.error("--combined needs the Critic")
    if args.combined_planner is not None and args.combined_planner not in PLANNER_MODELS:
        p.error(f"--combined-planner must be one of {list(PLANNER_MODELS)}")
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


def read_bgr(path: Path) -> np.ndarray:
    from PIL import Image

    return np.ascontiguousarray(np.asarray(Image.open(path).convert("RGB"))[..., ::-1])


def load_images(folder: Path | None, n: int) -> tuple[list[np.ndarray], str]:
    if folder is None:
        return [synthetic_frame(i) for i in range(n)], "synthetic"
    files = sorted(f for f in folder.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise SystemExit(f"no images ({', '.join(sorted(IMAGE_SUFFIXES))}) in {folder}")
    return [read_bgr(files[i % len(files)]) for i in range(n)], f"{folder} ({len(files)} files)"


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


def free_cuda() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _gb(mb: float | None) -> str:
    return f"{mb / 1024:.2f} GB" if mb is not None else "n/a"


def print_md_table(head: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    widths = [max(len(head[i]), *(len(r[i]) for r in rows)) for i in range(len(head))]
    for line in (head, *rows):
        print("| " + " | ".join(line[i].ljust(widths[i]) for i in range(len(head))) + " |")
        if line is head:
            print("|-" + "-|-".join("-" * w for w in widths) + "-|")


# ------------------------------------------------------------------ language commands (M-2)
@dataclass(frozen=True)
class LanguageCase:
    instruction: str
    pick_id: int
    place_id: int


def language_detections() -> list[Any]:
    """Fixed detection list for the 10 language commands (pixel boxes, overhead frame)."""
    from lbyr.perception.detector import Detection

    return [
        Detection(1, "red_block", 0.93, (120.0, 140.0, 180.0, 200.0)),
        Detection(2, "blue_cube", 0.90, (260.0, 150.0, 320.0, 210.0)),
        Detection(3, "kitchen_sponge", 0.88, (380.0, 120.0, 470.0, 190.0)),
        Detection(4, "acrylic_block", 0.85, (200.0, 260.0, 260.0, 320.0)),
        Detection(7, "zone_A", 0.95, (60.0, 330.0, 200.0, 460.0)),
        Detection(8, "zone_B", 0.95, (440.0, 330.0, 580.0, 460.0)),
    ]


LANGUAGE_CASES: tuple[LanguageCase, ...] = (
    LanguageCase("Put the red block in zone A.", 1, 7),
    LanguageCase("Move the blue cube to zone B.", 2, 8),
    LanguageCase("Pick up the sponge and place it on zone A.", 3, 7),
    LanguageCase("Place the acrylic block into zone B.", 4, 8),
    LanguageCase("Grab the kitchen sponge, drop it in zone B.", 3, 8),
    LanguageCase("Take the blue cube and put it in zone A.", 2, 7),
    LanguageCase("Could you move the red block over to zone B?", 1, 8),
    LanguageCase("Transfer the acrylic block to zone A, please.", 4, 7),
    LanguageCase("Put the cube that is blue on the B zone.", 2, 8),
    LanguageCase("The sponge goes to zone A.", 3, 7),
)


def bench_language_planner(planner: Any, timeout_s: float = 8.0) -> dict[str, Any]:
    """Run the 10 fixed commands through ``planner.generate_language_untimed`` (full
    generation, no cut-off) and the strict parser. Pure measurement; no memory handling."""
    from lbyr.ai.planner import PlanRequest, expand_compact_plan, parse_skill_chain
    from lbyr.errors import PlanValidationError

    dets = language_detections()
    ids = {d.id for d in dets}
    lat_s: list[float] = []
    tok_s: list[float] = []
    toks: list[float] = []
    valid = correct = 0
    samples: list[dict[str, Any]] = []
    for case in LANGUAGE_CASES:
        req = PlanRequest(case.instruction, dets)
        t0 = time.perf_counter()
        text = planner.generate_language_untimed(req)
        lat_s.append(time.perf_counter() - t0)
        stats = getattr(planner, "last_stats", None)
        if stats is not None:
            tok_s.append(stats.tokens_per_s)
            toks.append(float(stats.new_tokens))
        error = None
        try:
            chain = parse_skill_chain(expand_compact_plan(text), ids)
            valid += 1
            if (chain.approach.target_id, chain.place.target_id) == (case.pick_id, case.place_id):
                correct += 1
            else:
                error = "wrong target"
        except PlanValidationError as e:
            error = str(e)
        samples.append({"instruction": case.instruction, "raw": text, "error": error})
    n = len(LANGUAGE_CASES)
    return {
        "latency_s": percentile_stats(lat_s),
        "tokens_per_s": percentile_stats(tok_s) if tok_s else None,
        "new_tokens": percentile_stats(toks) if toks else None,
        "valid_parse_fraction": valid / n,
        "correct_fraction": correct / n,
        "within_timeout_fraction": sum(t <= timeout_s for t in lat_s) / n,
        "timeout_s": timeout_s,
        "samples": samples,
    }


def _build_language_planner(name: str, models_dir: Path) -> Any:
    from lbyr.ai.planner import build_language_planner
    from lbyr.config import PlannerConfig

    path = models_dir / name
    backend = PLANNER_MODELS[name]
    if backend == "qwen_vl":
        cfg = PlannerConfig(language_backend="qwen_vl", model_path=path)
    else:
        cfg = PlannerConfig(language_backend="qwen3_text", language_model_path=path)
    return build_language_planner(cfg)


def has_weights(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.safetensors"))


def bench_planner_model(name: str, models_dir: Path, mon: Any) -> dict[str, Any]:
    """Load ``name`` alone, run the 10 commands, measure memory, unload."""
    from lbyr.ai.cuda_mem import cuda_allocated_mb, cuda_peak_mb, reset_cuda_peak

    path = models_dir / name
    if not has_weights(path):
        msg = f"no weights in {path} (download it locally first)"
        print(f"Planner [{name}]: SKIPPED, {msg}")
        return {"model": name, "skipped": msg}
    free_cuda()
    time.sleep(0.5)
    reset_cuda_peak()
    ram_before = mon.latest_mb
    print(f"Planner [{name}]: loading ...")
    t0 = time.perf_counter()
    planner = _build_language_planner(name, models_dir)
    load_s = time.perf_counter() - t0
    time.sleep(0.5)
    out: dict[str, Any] = {
        "model": name,
        "language_backend": PLANNER_MODELS[name],
        "load_s": load_s,  # load + 8-token warm-up
        "torch_alloc_mb": cuda_allocated_mb(),
        "ram_before_mb": ram_before,
        "ram_after_load_mb": mon.latest_mb,
    }
    print(f"Planner [{name}]: {len(LANGUAGE_CASES)} language commands ...")
    mon.reset_peak()
    out.update(bench_language_planner(planner))
    time.sleep(0.3)
    out["torch_peak_mb"] = cuda_peak_mb()
    out["ram_peak_mb"] = mon.peak_mb
    del planner
    free_cuda()
    time.sleep(1.0)
    return out


def print_planner_table(planners: list[dict[str, Any]]) -> None:
    if not planners:
        return
    head = (
        "Model",
        "Load+warm",
        "p50",
        "p95",
        "tok/s",
        "tokens",
        "valid",
        "correct",
        "<=8 s",
        "torch alloc",
        "RAM +load",
        "RAM peak",
    )
    rows: list[tuple[str, ...]] = []
    for p in planners:
        if "skipped" in p:
            rows.append((p["model"], "skipped", "", "", "", "", "", "", "", "", "", ""))
            continue
        lat, tps, nt = p["latency_s"], p["tokens_per_s"], p["new_tokens"]
        ram_delta = (
            p["ram_after_load_mb"] - p["ram_before_mb"]
            if p.get("ram_after_load_mb") is not None and p.get("ram_before_mb") is not None
            else None
        )
        rows.append(
            (
                p["model"],
                f"{p['load_s']:.1f} s",
                f"{lat['p50']:.2f} s",
                f"{lat['p95']:.2f} s",
                f"{tps['p50']:.1f}" if tps else "n/a",
                f"{nt['p50']:.0f}" if nt else "n/a",
                f"{p['valid_parse_fraction'] * 100:.0f}%",
                f"{p['correct_fraction'] * 100:.0f}%",
                f"{p['within_timeout_fraction'] * 100:.0f}%",
                _gb(p.get("torch_alloc_mb")),
                f"+{_gb(ram_delta)}" if ram_delta is not None else "n/a",
                _gb(p.get("ram_peak_mb")),
            )
        )
    print(f"Language-mode planners (each loaded alone; {len(LANGUAGE_CASES)} fixed commands)")
    print_md_table(head, rows)
    for p in planners:
        if "skipped" in p:
            print(f"note: {p['model']} skipped: {p['skipped']}")
    print()


# ------------------------------------------------------------------ Critic
def bench_critic(
    critic: Any,
    frames: list[np.ndarray],
    crop: bool = False,
    crop_box: tuple[float, float, float, float] | None = None,
    crop_size: int = 378,
) -> dict[str, Any]:
    """Time ``critic.infer`` (including the crop, if any) on every frame."""
    from lbyr.ai.critic import CRITIC_PROMPT, parse_verdict
    from lbyr.ai.imaging import critic_crop

    lat_ms: list[float] = []
    answers: list[str] = []
    shape = frames[0].shape
    for f in frames:
        t0 = time.perf_counter()
        img = critic_crop(f, crop_box, crop_size) if crop else f
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
    crop_box: tuple[float, float, float, float] | None,
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
    """Runtime input (378x378 crop, CriticConfig.crop_box_norm); with --critic-crop the full
    frame first for comparison; with --critic-compile the same after Moondream's compile().
    Returns (variants, skip messages)."""
    from lbyr.ai.critic import CRITIC_PROMPT
    from lbyr.ai.imaging import critic_crop
    from lbyr.config import CriticConfig

    crop_box = CriticConfig().crop_box_norm  # OQ-12 gripper ROI; None = centred square
    inputs = ([("full", False)] if args.critic_crop else []) + [("crop378", True)]
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


def print_critic_variants(result: dict[str, Any]) -> None:
    """One row per Critic variant; p50 is also compared with the first variant."""
    variants = result.get("critic_variants") or []
    if not variants:
        return
    first = variants[0]["latency_ms"]["p50"]
    head = ("Variant", "Input", "p50", "p95", "p50 vs first", "torch peak", "RAM peak", "Verdicts")
    rows = [
        (
            v["variant"],
            v["input"],
            f"{v['latency_ms']['p50'] / 1000:.3f} s",
            f"{v['latency_ms']['p95'] / 1000:.3f} s",
            f"x{v['latency_ms']['p50'] / first:.2f}",
            _gb(v.get("torch_peak_mb")),
            _gb(v.get("tegrastats_peak_mb")),
            ",".join(f"{k}:{n}" for k, n in v["verdicts"].items()),
        )
        for v in variants
    ]
    print(f"Critic variants ({result['n_warm_calls']} timed calls each, {result['images']})")
    print_md_table(head, rows)
    for msg in result.get("critic_skipped") or []:
        print(f"note: {msg}")
    print("torch peak = max_memory_allocated (weights included); RAM peak = tegrastats used.")
    print()


# ------------------------------------------------------------------ Critic accuracy
def find_dataset(root: Path) -> dict[str, list[Path]]:
    """``root/<CLASS>/*.jpg`` (also png/jpeg/bmp) for the four Critic classes; {} if none."""
    out: dict[str, list[Path]] = {}
    for cls in CRITIC_CLASSES:
        d = root / cls
        files = (
            sorted(f for f in d.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
            if d.is_dir()
            else []
        )
        if files:
            out[cls] = files
    return out


def classification_report(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    """Accuracy, per-class precision/recall/F1 and a confusion matrix (rows = true class,
    columns = predicted: the four classes + INVALID / TIMEOUT if they occur)."""
    labels = list(CRITIC_CLASSES) + sorted(set(y_pred) - set(CRITIC_CLASSES))
    matrix = {t: {p: 0 for p in labels} for t in CRITIC_CLASSES}
    for t, p in zip(y_true, y_pred, strict=True):
        matrix[t][p] += 1
    per_class: dict[str, dict[str, float]] = {}
    for c in CRITIC_CLASSES:
        tp = matrix[c][c]
        fp = sum(matrix[t][c] for t in CRITIC_CLASSES if t != c)
        fn = sum(matrix[c][p] for p in labels if p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": float(tp + fn),
        }
    n = len(y_true)
    return {
        "n": n,
        "accuracy": sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / n if n else 0.0,
        "per_class": per_class,
        "labels": labels,
        "confusion": matrix,
    }


def critic_accuracy(critic: Any, root: Path) -> dict[str, Any] | None:
    """Run the Critic (runtime input: crop per CriticConfig) on the labelled dataset."""
    from lbyr.ai.critic import CRITIC_PROMPT, parse_verdict
    from lbyr.ai.imaging import critic_crop
    from lbyr.config import CriticConfig

    data = find_dataset(root)
    if not data:
        return None
    cfg = CriticConfig()
    y_true: list[str] = []
    y_pred: list[str] = []
    total = sum(len(v) for v in data.values())
    print(f"Critic accuracy: {total} labelled images in {root} ...")
    for cls, files in data.items():
        for f in files:
            img = read_bgr(f)
            if cfg.crop_enabled:
                img = critic_crop(img, cfg.crop_box_norm, cfg.crop_size_px)
            y_true.append(cls)
            y_pred.append(parse_verdict(critic.infer(img, CRITIC_PROMPT)).value)
    report = classification_report(y_true, y_pred)
    report["dataset"] = str(root)
    return report


def print_accuracy(acc: dict[str, Any] | None, root: Path) -> None:
    if acc is None:
        print(
            f"WARNING: Critic accuracy is NOT measured: no labelled images in {root}/<CLASS>/ "
            f"({', '.join(CRITIC_CLASSES)}). Latency only."
        )
        print()
        return
    print(f"Critic accuracy on {acc['dataset']}: {acc['accuracy'] * 100:.1f}% (n={acc['n']})")
    rows = [
        (
            c,
            f"{m['precision']:.2f}",
            f"{m['recall']:.2f}",
            f"{m['f1']:.2f}",
            f"{m['support']:.0f}",
        )
        for c, m in acc["per_class"].items()
    ]
    print_md_table(("Class", "Precision", "Recall", "F1", "Support"), rows)
    print()
    labels = acc["labels"]
    rows = [(t, *(str(acc["confusion"][t][p]) for p in labels)) for t in CRITIC_CLASSES]
    print("Confusion matrix (rows = true, columns = predicted)")
    print_md_table(("true \\ pred", *labels), rows)
    print()


# ------------------------------------------------------------------ combined
def bench_combined(
    critic: Any, name: str, models_dir: Path, frames: list[np.ndarray], mon: Any
) -> dict[str, Any]:
    """Critic already loaded; load planner ``name`` on top and interleave calls."""
    from lbyr.ai.critic import CRITIC_PROMPT
    from lbyr.ai.cuda_mem import cuda_allocated_mb, cuda_peak_mb, reset_cuda_peak
    from lbyr.ai.imaging import critic_crop
    from lbyr.ai.models import check_memory_headroom
    from lbyr.ai.planner import PlanRequest
    from lbyr.config import CriticConfig, PlannerConfig
    from lbyr.errors import InsufficientMemoryError
    from lbyr.logging_.episodes import read_system_ram

    path = models_dir / name
    if not has_weights(path):
        return {"planner": name, "skipped": f"no weights in {path}"}
    ccfg = CriticConfig()
    print(f"Combined: loading {name} next to the Critic ...")
    mon.reset_peak()
    reset_cuda_peak()
    t0 = time.perf_counter()
    planner = _build_language_planner(name, models_dir)
    load_s = time.perf_counter() - t0
    time.sleep(0.5)
    ram = read_system_ram()
    guard_cfg = PlannerConfig(language_mode=True, language_backend=PLANNER_MODELS[name])
    try:
        check_memory_headroom(guard_cfg, ram)
        guard = f"PASS (>= {guard_cfg.qwen_min_free_ram_mb:.0f} MB free)"
    except InsufficientMemoryError:
        guard = f"FAIL (< {guard_cfg.qwen_min_free_ram_mb:.0f} MB free)"
    dets = language_detections()
    critic_ms: list[float] = []
    planner_s: list[float] = []
    print(f"Combined: {len(frames)} x (Critic query + planner command) ...")
    for i, f in enumerate(frames):
        t = time.perf_counter()
        critic.infer(critic_crop(f, ccfg.crop_box_norm, ccfg.crop_size_px), CRITIC_PROMPT)
        critic_ms.append((time.perf_counter() - t) * 1000)
        case = LANGUAGE_CASES[i % len(LANGUAGE_CASES)]
        t = time.perf_counter()
        planner.generate_language_untimed(PlanRequest(case.instruction, dets))
        planner_s.append(time.perf_counter() - t)
    time.sleep(0.3)
    out = {
        "planner": name,
        "critic": f"Moondream2 {ccfg.dtype}, {ccfg.crop_size_px}x{ccfg.crop_size_px} crop",
        "planner_load_s": load_s,
        "torch_alloc_mb": cuda_allocated_mb(),
        "torch_peak_mb": cuda_peak_mb(),
        "ram_after_load": None
        if ram is None
        else {"used_mb": ram.used_mb, "total_mb": ram.total_mb, "free_mb": ram.free_mb},
        "ram_peak_mb": mon.peak_mb,
        "critic_latency_ms": percentile_stats(critic_ms),
        "planner_latency_s": percentile_stats(planner_s),
        "memory_guard": guard,
    }
    del planner
    free_cuda()
    return out


def print_combined(c: dict[str, Any] | None) -> None:
    if not c:
        return
    if "skipped" in c:
        print(f"Combined run skipped: {c['skipped']}")
        print()
        return
    ram = c["ram_after_load"]
    total = ram["total_mb"] if ram else None
    rows = [
        (
            c["critic"],
            c["planner"],
            _gb(c["torch_alloc_mb"]),
            f"{_gb(ram['used_mb'])} / {_gb(total)}" if ram else "n/a",
            _gb(ram["free_mb"]) if ram else "n/a",
            _gb(c["ram_peak_mb"]),
            f"{c['critic_latency_ms']['p50'] / 1000:.2f} s",
            f"{c['planner_latency_s']['p50']:.2f} s",
            c["memory_guard"],
        )
    ]
    print("Combined: Critic + language planner co-resident (Critic loaded first)")
    print_md_table(
        (
            "Critic",
            "Planner",
            "torch alloc",
            "RAM used after load",
            "RAM free",
            "RAM peak",
            "Critic p50",
            "Planner p50",
            "Memory guard",
        ),
        rows,
    )
    print()


# ------------------------------------------------------------------ legacy VL full plan
def bench_planner(planner: Any, frames: list[np.ndarray]) -> dict[str, Any]:
    """--vl-full-plan: Qwen2.5-VL in-loop full JSON contract with the overhead image."""
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


def print_vl_full_plan(pl: dict[str, Any] | None) -> None:
    if not pl:
        return
    lat, tps = pl["latency_s"], pl["tokens_per_s"]
    print(
        f"Qwen2.5-VL in-loop full plan: p50 {lat['p50']:.2f} s / p95 {lat['p95']:.2f} s, "
        f"{tps['p50']:.1f} tok/s, valid {pl['valid_json_fraction'] * 100:.0f}%, "
        f"{pl['within_hard_timeout_fraction'] * 100:.0f}% within 3 s"
        if tps
        else f"Qwen2.5-VL in-loop full plan: p50 {lat['p50']:.2f} s"
    )
    print()


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if args.skip_critic and not args.planner_list and not args.vl_full_plan:
        print("nothing to do: Critic skipped and no --planner-models / --vl-full-plan")
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
    memory: dict[str, Any] = {"tegrastats_baseline_mb": ram_before}
    result["memory"] = memory

    # 1. Planners, each ALONE (nothing else loaded).
    if args.planner_list:
        result["planners"] = [
            bench_planner_model(name, args.models_dir, mon) for name in args.planner_list
        ]

    # 2. Critic (runtime input), accuracy, then 3. combined with the chosen planner.
    critic = None
    if not args.skip_critic:
        from lbyr.ai.critic import Moondream2Critic

        free_cuda()
        critic = Moondream2Critic(
            CriticConfig(model_path=args.critic_path, dtype=args.critic_dtype)
        )
        memory["critic_alloc_mb"] = cuda_allocated_mb()
        result["critic_dtype"] = args.critic_dtype
        variants, skipped = run_critic_variants(critic, frames, args, mon)
        result["critic_variants"] = variants
        result["critic_skipped"] = skipped
        result["critic"] = next(v for v in variants if v["variant"].endswith("-crop378"))
        result["critic_accuracy"] = critic_accuracy(critic, args.dataset)
        if args.combined:
            chosen = args.combined_planner or next(
                (m for m in args.planner_list if has_weights(args.models_dir / m)), "qwen3-1.7b"
            )
            result["combined"] = bench_combined(critic, chosen, args.models_dir, frames, mon)

    # 4. Legacy: Qwen2.5-VL in-loop full plan (opt-in).
    if args.vl_full_plan:
        from lbyr.ai.planner import QwenPlanner

        planner = QwenPlanner(
            PlannerConfig(model_path=args.planner_path, max_new_tokens=args.max_new_tokens)
        )
        print(f"Qwen2.5-VL full plan: {args.n} calls (~11 s each) ...")
        result["vl_full_plan"] = bench_planner(planner, frames)

    memory["torch_peak_mb_end"] = cuda_peak_mb()
    time.sleep(0.5)
    mon.stop()

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"bench-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(result, indent=2))
    print()
    print_planner_table(result.get("planners") or [])
    print_critic_variants(result)
    if critic is not None:
        print_accuracy(result.get("critic_accuracy"), args.dataset)
    print_combined(result.get("combined"))
    print_vl_full_plan(result.get("vl_full_plan"))
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
