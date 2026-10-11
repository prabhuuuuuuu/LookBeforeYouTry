"""Train a linear probe on Moondream2 vision embeddings and compare it with the zero-shot Critic.

WHY
    The runtime Critic asks Moondream2 a one-word question (~0.85 s per call). A logistic
    regression on the vision encoder's pooled features needs only the encoder (no text
    prefill / decode). This measures, on the SAME held-out split, whether the probe is more
    accurate and faster than the zero-shot prompt before anything is wired into the runtime.

HOW TO USE (on the Jetson; OFFLINE: no arm, and NOT while lbyr.demo is running)
        python tools/train_critic_probe.py                       # full run
        python tools/train_critic_probe.py --limit 2             # smoke test (2 per class/split)
        python tools/train_critic_probe.py --crop 0.2,0.1,0.6,0.8
        python tools/train_critic_probe.py --reuse-emb runs/critic_probe_<ts>/emb.pt
    Data: <data>/{GRASPED,SLIPPING,EMPTY,OCCLUDED}/*.{jpg,jpeg,png} (default ~/critic_raw).
    Per class the filenames are SORTED: the first 30 = train, the last 15 = test.

WHAT IT DOES
    1. Loads each image (cv2.imread; None or mean < 5 -> lbyr decode_wrist_jpeg). 1920x1080
       frames get the runtime preprocessing: build_wrist_preprocess (dewarp, only if the
       fisheye calibration exists) then critic_crop(box, 378). Other sizes are used as-is.
    2. Moondream2 bf16 via the runtime loader (lbyr.ai.critic._load_moondream2). Embedding
       = vision encoder on the global 378x378 crop, mean-pooled over the 729 patch tokens
       (1152-d, EMBED_FN_NAME). Cached to runs/critic_probe_<ts>/emb.pt.
    3. 4-class logistic regression (plain torch, standardized features, L2, Adam,
       full batch). Saved to ~/models/critic_probe.pt (--limit runs: the run dir).
    4. Zero-shot baseline: the runtime Moondream2Critic + CRITIC_PROMPT + parse_verdict under
       the runtime TimeoutRunner (CriticConfig.timeout_s), on the same test images.
    5. summary.md (confusion matrices, P/R/F1, 4-class and binary accuracy, latency),
       predictions.csv, misclassified/<method>_<label>_as_<pred>_<stem>.jpg (post-crop).

COST
    VRAM: Moondream2 bf16 ~4.1 GB (same as the runtime Critic). Probe head: 4x1152 floats,
    ~0.05 ms on CPU. Encoder-only latency is measured here (bible 9.2).

SAFETY
    Never talks to the robot arm or any serial port. Edge-only: local weights, HF offline.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

CLASSES = ("GRASPED", "SLIPPING", "EMPTY", "OCCLUDED")  # == lbyr.ai.critic.MODEL_CLASSES
OTHER = "OTHER"  # zero-shot INVALID / TIMEOUT
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
RUNTIME_SIZE = (1920, 1080)  # Trinet capture size the runtime crop is defined for
EMBED_FN_NAME = "moondream2_vis_enc_global378_meanpool729_v1"
DEFAULT_DATA = Path("~/critic_raw")
DEFAULT_MODEL = Path("~/models/moondream2")
DEFAULT_OUT = Path("~/models/critic_probe.pt")
REPO = Path(__file__).resolve().parents[1]

log = logging.getLogger("train_critic_probe")


@dataclass
class Sample:
    path: Path
    label: str
    split: str  # "train" | "test"
    condition: str = ""  # e.g. bright/dim/side, parsed from s<n>_<LABEL>_<obj>_<cond>_...
    image: np.ndarray | None = field(default=None, repr=False)  # post-crop BGR


# ------------------------------------------------------------------ data
def list_samples(root: Path, n_train: int, n_test: int, limit: int | None) -> list[Sample]:
    """Sorted filenames per class: first ``n_train`` = train, last ``n_test`` = test."""
    out: list[Sample] = []
    for label in CLASSES:
        files = sorted(p for p in (root / label).glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise SystemExit(f"no images in {root / label}")
        if len(files) < n_train + n_test:
            print(
                f"WARNING: {label} has {len(files)} images < {n_train}+{n_test}: "
                f"test = last {min(n_test, len(files))}, train = the rest (max {n_train})"
            )
        test = files[-n_test:] if len(files) > n_test else files[len(files) // 2 :]
        train = files[: min(n_train, len(files) - len(test))]
        if limit is not None:
            train, test = train[:limit], test[:limit]
        for split, group in (("train", train), ("test", test)):
            out += [Sample(p, label, split, _condition(p, label)) for p in group]
    return out


def _condition(path: Path, label: str) -> str:
    parts = path.stem.split("_")
    return parts[3] if len(parts) > 4 and parts[1] == label else ""


def load_image(path: Path) -> np.ndarray | None:
    """cv2.imread; if None or mean < 5, retry with lbyr's Trinet decoder (SEI-blob repair)."""
    import cv2

    from lbyr.perception.wrist import decode_wrist_jpeg

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None or float(img.mean()) < 5.0:
        alt = decode_wrist_jpeg(path.read_bytes())
        if alt is not None and (img is None or float(alt.mean()) > float(img.mean())):
            img = alt
    return img


def preprocess(
    samples: list[Sample], box: tuple[float, float, float, float] | None, size_px: int
) -> dict[str, Any]:
    """Fill ``Sample.image`` with the Critic input. Returns facts for the summary."""
    from lbyr.ai.imaging import critic_crop
    from lbyr.config import WristCameraConfig
    from lbyr.perception.wrist import build_wrist_preprocess

    raw: dict[int, np.ndarray] = {}
    sizes: Counter[tuple[int, int]] = Counter()
    unreadable: list[str] = []
    for i, s in enumerate(samples):
        img = load_image(s.path)
        if img is None:
            unreadable.append(str(s.path))
            continue
        raw[i] = img
        sizes[(img.shape[1], img.shape[0])] += 1
    print("Image size histogram (W x H: count):")
    for (w, h), n in sizes.most_common():
        print(f"  {w} x {h}: {n}")
    for p in unreadable:
        print(f"WARNING: unreadable, skipped: {p}")

    # Same order as the runtime: dewarp (CriticWorker preprocess) -> critic_crop.
    dewarp = build_wrist_preprocess(WristCameraConfig(), RUNTIME_SIZE)
    warned = False
    for i, img in raw.items():
        if (img.shape[1], img.shape[0]) == RUNTIME_SIZE:
            if dewarp is not None:
                img = dewarp(img)
            img = critic_crop(img, box, size_px)
        elif not warned:
            print(
                f"WARNING: images not {RUNTIME_SIZE[0]}x{RUNTIME_SIZE[1]} are used AS-IS "
                "(no dewarp, no runtime crop)"
            )
            warned = True
        samples[i].image = img
    kept = [s for s in samples if s.image is not None]
    samples[:] = kept
    return {
        "sizes": dict(sizes),
        "unreadable": unreadable,
        "dewarp": dewarp is not None,
        "as_is": warned,
    }


# ------------------------------------------------------------------ embeddings
def make_embed_fn(model: Any) -> Callable[[np.ndarray], Any]:
    """BGR image -> float32 (1152,) CPU tensor. EMBED_FN_NAME; keep both in sync.

    Moondream's own ``prepare_crops`` (same normalization as ``query``), then ``_vis_enc`` on
    the GLOBAL crop only (index 0: the image resized to 378x378; for a 378x378 input this is
    the input itself), then the mean over the 729 patch tokens of the post-LN output.
    """
    import torch

    from lbyr.ai.imaging import bgr_to_pil

    inner = model.model
    prepare_crops = sys.modules[type(inner).__module__].prepare_crops

    def embed(image_bgr: np.ndarray) -> torch.Tensor:
        with torch.inference_mode():
            crops, _ = prepare_crops(bgr_to_pil(image_bgr), inner.config.vision, inner.device)
            feats = inner._vis_enc(crops[:1])  # (1, 729, 1152)
            return feats[0].float().mean(dim=0).cpu()

    return embed


def extract_embeddings(
    embed: Callable[[np.ndarray], Any], samples: list[Sample], timeout_s: float
) -> tuple[Any, list[float]]:
    import torch

    from lbyr.ai.timeout import TimeoutResult, run_with_timeout

    for _ in range(2):  # warm-up (first CUDA calls are slow); not timed
        embed(np.zeros((378, 378, 3), np.uint8))
    rows: list[torch.Tensor] = []
    lat: list[float] = []
    for k, s in enumerate(samples):
        assert s.image is not None
        img = s.image

        def call(img: np.ndarray = img) -> tuple[torch.Tensor, float]:
            t0 = time.perf_counter()
            e = embed(img)  # .cpu() synchronizes
            return e, (time.perf_counter() - t0) * 1000

        out = run_with_timeout(call, timeout_s, "probe-embed")
        if isinstance(out, TimeoutResult):
            raise SystemExit(f"vision encoder exceeded {timeout_s:.0f} s on {s.path}")
        rows.append(out[0])
        lat.append(out[1])
        if (k + 1) % 20 == 0 or k + 1 == len(samples):
            print(f"  embedded {k + 1}/{len(samples)} (last {out[1]:.0f} ms)")
    return torch.stack(rows), lat


# ------------------------------------------------------------------ probe
def train_probe(
    x: Any, y: Any, steps: int, lr: float, weight_decay: float, seed: int
) -> dict[str, Any]:
    """Multinomial logistic regression: logits = ((x - mean) / std) @ W.T + b."""
    import torch

    torch.manual_seed(seed)
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False).clamp_min(1e-6)
    xs = (x - mean) / std
    w = torch.zeros(len(CLASSES), x.shape[1], requires_grad=True)
    b = torch.zeros(len(CLASSES), requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    loss = torch.tensor(0.0)
    for _ in range(steps):
        opt.zero_grad()
        ce = torch.nn.functional.cross_entropy(xs @ w.T + b, y)
        loss = ce + weight_decay * (w * w).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        train_acc = float(((xs @ w.T + b).argmax(1) == y).float().mean())
    return {
        "weights": w.detach().clone(),
        "bias": b.detach().clone(),
        "mean": mean,
        "std": std,
        "final_loss": float(loss.detach()),
        "train_acc": train_acc,
    }


def probe_predict(probe: dict[str, Any], e: Any) -> int:
    z = ((e - probe["mean"]) / probe["std"]) @ probe["weights"].T + probe["bias"]
    return int(z.argmax())


# ------------------------------------------------------------------ zero-shot
def zero_shot(critic: Any, samples: list[Sample], timeout_s: float) -> list[tuple[str, str, float]]:
    """(prediction, raw text, latency ms) per sample, the CriticWorker way: TimeoutRunner +
    CRITIC_PROMPT + parse_verdict. INVALID / TIMEOUT -> OTHER. After a timeout, waits for the
    abandoned call to finish (offline) so the next image is not rejected as busy."""
    from lbyr.ai.critic import CRITIC_PROMPT, parse_verdict
    from lbyr.ai.timeout import TimeoutResult, TimeoutRunner

    runner = TimeoutRunner(timeout_s, name="critic-infer")
    out: list[tuple[str, str, float]] = []
    for k, s in enumerate(samples):
        img = s.image
        box: dict[str, float] = {}

        def call(img: np.ndarray | None = img, box: dict[str, float] = box) -> str:
            t0 = time.perf_counter()
            try:
                return str(critic.infer(img, CRITIC_PROMPT))
            finally:
                box["ms"] = (time.perf_counter() - t0) * 1000

        res = runner.run(call)
        if isinstance(res, TimeoutResult):
            while runner.busy:
                time.sleep(0.02)
            pred, raw = "TIMEOUT", ""
        else:
            raw = res
            pred = parse_verdict(res).value
        out.append((pred, raw, box.get("ms", float("nan"))))
        if (k + 1) % 10 == 0 or k + 1 == len(samples):
            print(f"  zero-shot {k + 1}/{len(samples)}: {raw!r} -> {pred}")
    return out


# ------------------------------------------------------------------ metrics
def confusion(labels: list[str], preds: list[str]) -> np.ndarray:
    """Rows = true class (CLASSES), columns = CLASSES + OTHER."""
    cols = [*CLASSES, OTHER]
    m = np.zeros((len(CLASSES), len(cols)), dtype=int)
    for t, p in zip(labels, preds, strict=True):
        m[CLASSES.index(t), cols.index(p if p in CLASSES else OTHER)] += 1
    return m


def per_class_prf(m: np.ndarray) -> list[tuple[str, float, float, float, int]]:
    rows = []
    for i, c in enumerate(CLASSES):
        tp = m[i, i]
        npred, ntrue = m[:, i].sum(), m[i, :].sum()
        p = tp / npred if npred else 0.0
        r = tp / ntrue if ntrue else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        rows.append((c, float(p), float(r), float(f1), int(ntrue)))
    return rows


def binary_accuracy(labels: list[str], preds: list[str]) -> tuple[float, int]:
    """GRASPED vs (SLIPPING + EMPTY): OCCLUDED-labelled images are excluded; any prediction
    other than GRASPED (incl. OCCLUDED / INVALID / TIMEOUT) counts as "not grasped"."""
    pairs = [(t, p) for t, p in zip(labels, preds, strict=True) if t != "OCCLUDED"]
    if not pairs:
        return float("nan"), 0
    ok = sum((t == "GRASPED") == (p == "GRASPED") for t, p in pairs)
    return ok / len(pairs), len(pairs)


def lat_stats(ms: list[float]) -> tuple[float, float]:
    a = np.asarray([v for v in ms if np.isfinite(v)], dtype=float)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(np.percentile(a, 95))


def method_report(
    name: str, latency_label: str, labels: list[str], preds: list[str], lat: list[float]
) -> tuple[list[str], dict[str, Any]]:
    m = confusion(labels, preds)
    acc = float(np.trace(m[:, : len(CLASSES)]) / max(1, len(labels)))
    bacc, bn = binary_accuracy(labels, preds)
    mean_ms, p95_ms = lat_stats(lat)
    md = [f"## {name}", ""]
    md += [
        f"- 4-class accuracy: **{acc:.3f}** ({int(np.trace(m[:, :4]))}/{len(labels)})",
        f"- Binary accuracy (GRASPED vs SLIPPING+EMPTY, n={bn}): **{bacc:.3f}**",
        f"- Latency per image ({latency_label}): mean {mean_ms:.1f} ms, p95 {p95_ms:.1f} ms",
        "",
        "Confusion matrix (rows = true, columns = predicted; OTHER = INVALID/TIMEOUT):",
        "",
        "| true \\ pred | " + " | ".join([*CLASSES, OTHER]) + " |",
        "|---" * (len(CLASSES) + 2) + "|",
    ]
    for i, c in enumerate(CLASSES):
        md.append(f"| {c} | " + " | ".join(str(v) for v in m[i]) + " |")
    md += ["", "| class | precision | recall | F1 | n |", "|---|---|---|---|---|"]
    for c, p, r, f1, n in per_class_prf(m):
        md.append(f"| {c} | {p:.3f} | {r:.3f} | {f1:.3f} | {n} |")
    md.append("")
    return md, {"acc": acc, "bin_acc": bacc, "mean_ms": mean_ms, "p95_ms": p95_ms}


# ------------------------------------------------------------------ main
def demo_running() -> list[str]:
    """Command lines of running ``lbyr.demo`` processes (Linux /proc)."""
    hits = []
    for d in Path("/proc").glob("[0-9]*"):
        if d.name == str(os.getpid()):
            continue
        try:
            argv = (d / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        # argv-exact: "python -m lbyr.demo ..." or ".../lbyr/demo.py" (not a shell mentioning it)
        as_module = any(
            a == "-m" and b == "lbyr.demo" for a, b in zip(argv, argv[1:], strict=False)
        )
        if as_module or any(a.endswith("lbyr/demo.py") for a in argv[:3]):
            hits.append(f"{d.name}: {' '.join(argv).strip()}")
    return hits


def parse_box(text: str | None) -> tuple[float, float, float, float] | None:
    if text is None or text.lower() in ("", "none", "centred", "centered"):
        return None
    vals = [float(v) for v in text.split(",")]
    if len(vals) != 4:
        raise SystemExit("--crop needs x0,y0,x1,y1 (normalized 0..1) or 'none'")
    return (vals[0], vals[1], vals[2], vals[3])


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Offline: Moondream2 embedding probe vs zero-shot Critic (no arm I/O).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="class folders root")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="local Moondream2 dir")
    ap.add_argument(
        "--crop",
        default=None,
        help="Critic box x0,y0,x1,y1 normalized (default: CriticConfig.crop_box_norm; "
        "'none' = centred square)",
    )
    ap.add_argument("--train", type=int, default=30, help="first N per class (sorted) = train")
    ap.add_argument("--test", type=int, default=15, help="last N per class (sorted) = test")
    ap.add_argument("--limit", type=int, default=None, help="max images per class per split")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"probe file (default {DEFAULT_OUT}; with --limit: <run dir>/critic_probe.pt)",
    )
    ap.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    ap.add_argument("--reuse-emb", type=Path, default=None, help="emb.pt from an earlier run")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=1e-2, help="L2 on W (sum of squares)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-zero-shot", action="store_true")
    ap.add_argument("--force", action="store_true", help="run even if lbyr.demo is running")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)  # interleave with log lines
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    running = demo_running()
    if running and not args.force:
        print("REFUSING: lbyr.demo is running (GPU/RAM contention). Stop it first:")
        print("\n".join(f"  {r}" for r in running))
        return 2

    import torch

    from lbyr.ai.critic import MODEL_CLASSES, Moondream2Critic, _load_moondream2
    from lbyr.config import CriticConfig

    assert tuple(v.value for v in MODEL_CLASSES) == CLASSES
    ccfg = CriticConfig()
    box = parse_box(args.crop) if args.crop is not None else ccfg.crop_box_norm
    data = args.data.expanduser()
    model_path = args.model.expanduser()
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.runs_dir / f"critic_probe_{ts}"
    (run_dir / "misclassified").mkdir(parents=True, exist_ok=True)
    out_path = (
        args.out or (run_dir / "critic_probe.pt" if args.limit else DEFAULT_OUT)
    ).expanduser()
    print(f"run dir: {run_dir}\ncrop box: {box or 'centred square'} -> {ccfg.crop_size_px} px")

    samples = list_samples(data, args.train, args.test, args.limit)
    facts = preprocess(samples, box, ccfg.crop_size_px)
    train = [s for s in samples if s.split == "train"]
    test = [s for s in samples if s.split == "test"]
    print(f"train {len(train)}, test {len(test)}; dewarp: {facts['dewarp']}")

    # -- model (runtime loader, bf16) + embeddings
    model: Any = None
    emb_lat: list[float]
    cached = torch.load(args.reuse_emb, weights_only=False) if args.reuse_emb else None
    paths = [str(s.path) for s in samples]
    if cached is not None and (
        cached["paths"] != paths
        or cached["crop"] != box
        or cached["embed_fn_name"] != EMBED_FN_NAME
    ):
        print("WARNING: --reuse-emb does not match (paths/crop/embed fn): recomputing")
        cached = None
    if cached is not None:
        emb, emb_lat = cached["emb"], list(cached["latency_ms"])
        print(f"reusing {len(paths)} embeddings from {args.reuse_emb}")
    else:
        t0 = time.perf_counter()
        model = _load_moondream2(model_path, args.device, "bf16")
        print(f"Moondream2 bf16 loaded in {time.perf_counter() - t0:.1f} s")
        print(f"embedding {len(samples)} images ({EMBED_FN_NAME}) ...")
        emb, emb_lat = extract_embeddings(make_embed_fn(model), samples, ccfg.warmup_timeout_s)
    torch.save(
        {
            "paths": paths,
            "labels": [s.label for s in samples],
            "splits": [s.split for s in samples],
            "emb": emb,
            "latency_ms": emb_lat,
            "crop": box,
            "embed_fn_name": EMBED_FN_NAME,
        },
        run_dir / "emb.pt",
    )

    # -- probe
    idx_tr = [i for i, s in enumerate(samples) if s.split == "train"]
    idx_te = [i for i, s in enumerate(samples) if s.split == "test"]
    y_tr = torch.tensor([CLASSES.index(samples[i].label) for i in idx_tr])
    probe = train_probe(emb[idx_tr], y_tr, args.steps, args.lr, args.weight_decay, args.seed)
    print(f"probe: train acc {probe['train_acc']:.3f}, final loss {probe['final_loss']:.4f}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "weights": probe["weights"],  # (4, 1152); logits = ((e - mean) / std) @ W.T + b
            "bias": probe["bias"],
            "mean": probe["mean"],
            "std": probe["std"],
            "classes": list(CLASSES),
            "crop": box,
            "crop_size_px": ccfg.crop_size_px,
            "embed_fn_name": EMBED_FN_NAME,
            "model_path": str(model_path),
            "dtype": "bf16",
            "n_train": len(idx_tr),
            "created": ts,
        },
        out_path,
    )
    print(f"probe saved: {out_path}")
    probe_preds, probe_lat = [], []
    for i in idx_te:
        t0 = time.perf_counter()
        probe_preds.append(CLASSES[probe_predict(probe, emb[i])])
        probe_lat.append(emb_lat[i] + (time.perf_counter() - t0) * 1000)

    # -- zero-shot (same model object, runtime Critic wrapper + warm-up)
    zs: list[tuple[str, str, float]] = []
    if not args.skip_zero_shot:
        if model is None:
            model = _load_moondream2(model_path, args.device, "bf16")
        critic = Moondream2Critic(ccfg, model=model)
        print(f"zero-shot on {len(test)} test images (timeout {ccfg.timeout_s:.1f} s) ...")
        zs = zero_shot(critic, test, ccfg.timeout_s)

    # -- outputs
    labels = [s.label for s in test]
    results = [("probe", probe_preds, probe_lat)]
    if zs:
        results.append(("zeroshot", [p for p, _, _ in zs], [ms for _, _, ms in zs]))
    import cv2

    for method, preds, _ in results:
        for s, p in zip(test, preds, strict=True):
            if p != s.label and s.image is not None:
                name = f"{method}_{s.label}_as_{p}_{s.path.stem}.jpg"
                cv2.imwrite(str(run_dir / "misclassified" / name), s.image)
    with (run_dir / "predictions.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(
            ["path", "label", "condition", "probe", "probe_ms", "zeroshot", "zs_raw", "zs_ms"]
        )
        for k, s in enumerate(test):
            z = zs[k] if zs else ("", "", float("nan"))
            wr.writerow(
                [s.path, s.label, s.condition, probe_preds[k], f"{probe_lat[k]:.1f}", z[0], z[1],
                 f"{z[2]:.1f}"]
            )  # fmt: skip
    md = summary_md(args, facts, box, samples, probe, out_path, results, labels, ccfg.timeout_s)
    (run_dir / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {run_dir / 'summary.md'}")
    return 0


def summary_md(
    args: argparse.Namespace,
    facts: dict[str, Any],
    box: tuple[float, float, float, float] | None,
    samples: list[Sample],
    probe: dict[str, Any],
    out_path: Path,
    results: list[tuple[str, list[str], list[float]]],
    labels: list[str],
    zs_timeout_s: float,
) -> list[str]:
    md = [
        "# Critic probe vs zero-shot",
        "",
        f"- Data: `{args.data}` (sorted filenames per class: first {args.train} train, "
        f"last {args.test} test{f', limit {args.limit}' if args.limit else ''})",
        "- Image sizes: " + ", ".join(f"{w}x{h}: {n}" for (w, h), n in facts["sizes"].items()),
        f"- Preprocessing: dewarp {'ON' if facts['dewarp'] else 'OFF (no fisheye calibration)'}"
        f", critic_crop box {box or 'centred square'} -> 378x378"
        + ("; some images used AS-IS (not 1920x1080)" if facts["as_is"] else ""),
        f"- Unreadable (skipped): {len(facts['unreadable'])}",
        f"- Embedding: `{EMBED_FN_NAME}` (Moondream2 bf16, `{args.model}`)",
        f"- Probe: logistic regression, {args.steps} Adam steps, lr {args.lr}, "
        f"L2 {args.weight_decay}; train acc {probe['train_acc']:.3f}; saved `{out_path}`",
        f"- Zero-shot: runtime `CRITIC_PROMPT` + `parse_verdict`, timeout {zs_timeout_s:.1f} s",
        "",
        "## Split composition (per class: train / test, test conditions)",
        "",
        "| class | train | test | test conditions |",
        "|---|---|---|---|",
    ]
    for c in CLASSES:
        tr = [s for s in samples if s.label == c and s.split == "train"]
        te = [s for s in samples if s.label == c and s.split == "test"]
        cond = Counter(s.condition or "?" for s in te)
        md.append(
            f"| {c} | {len(tr)} | {len(te)} | "
            + ", ".join(f"{k}: {v}" for k, v in sorted(cond.items()))
            + " |"
        )
    md.append("")
    lat_label = {"probe": "vision encoder + head", "zeroshot": "full Moondream query"}
    rows = []
    for method, preds, lat in results:
        sec, r = method_report(method, lat_label[method], labels, preds, lat)
        md += sec
        rows.append((method, r))
    md += ["## Comparison", "", "| method | 4-class acc | binary acc | mean ms | p95 ms |"]
    md.append("|---|---|---|---|---|")
    for method, r in rows:
        md.append(
            f"| {method} | {r['acc']:.3f} | {r['bin_acc']:.3f} | {r['mean_ms']:.1f} | "
            f"{r['p95_ms']:.1f} |"
        )
    md += [
        "",
        "Latency excludes image load, dewarp and crop (identical for both, ~1-3 ms). "
        "Misclassified test images (post-crop): `misclassified/`. Per-image: `predictions.csv`.",
    ]
    return md


if __name__ == "__main__":
    sys.exit(main())
