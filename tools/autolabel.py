"""Offline auto-labelling of demo Critic crops with the cloud Critic (Anthropic Messages API).

WHY
    Every demo verdict saves the exact 378x378 crop the runtime Critic judged
    (data/episodes/demo-*/frames/<episode>/<n>_<STATE>_crop_<VERDICT>.jpg). Labelling those
    crops gives in-distribution training data for the probe (tools/train_critic_probe.py
    --extra ~/critic_auto) and a runtime-vs-cloud agreement matrix.

    OPT-IN cloud use: owner-approved exception to the edge-only rule (CLAUDE.md rule 1,
    2026-10-11). Offline only; the runtime default stays local.

HOW TO USE (any machine with network; never touches the arm or a serial port)
        export ANTHROPIC_API_KEY=...                       # read from the env, never stored
        python tools/autolabel.py --limit 2                # smoke test (2 requests)
        python tools/autolabel.py                          # everything, 30 req/min
        python tools/autolabel.py --resume                 # continue the latest run
        python tools/autolabel.py --resume runs/autolabel_<ts> --rate 50

WHAT IT DOES
    1. Walks data/episodes/demo-*/frames/**/*_crop_*.jpg (sorted). Episodes whose raw wrist
       frames are not the Trinet size (WristCameraConfig, 1920x1080) are --dry-run fakes and
       are skipped unless --include-fake.
    2. Sends each crop's JPEG bytes as-is with lbyr.ai.critic_cloud.CLOUD_PROMPT (same client
       as `lbyr.demo --critic cloud`: temperature 0, max_tokens 10, 15 s, one retry on
       429/5xx), parsed with the runtime parse_verdict. At most --rate requests per minute.
    3. Appends (flushes) one row per image to runs/autolabel_<ts>/labels.csv:
       path, episode, state, runtime_verdict, cloud_verdict, latency_ms.
       --resume skips images that already have a class verdict there (TIMEOUT / INVALID
       rows are retried; the last row per path wins).
    4. Copies each image with a class verdict to ~/critic_auto/<cloud_verdict>/<episode>_<name>.jpg.
    5. Prints per-class counts and the runtime-vs-cloud agreement matrix (whole CSV).

COST
    No GPU. One HTTPS request per image (~1-3 s, network-bound); --rate 30 = 2 s spacing.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

from lbyr.ai.critic_cloud import CLOUD_PROMPT, AnthropicCriticClient
from lbyr.config import CloudCriticConfig, WristCameraConfig

REPO = Path(__file__).resolve().parents[1]
CLASSES = ("GRASPED", "SLIPPING", "EMPTY", "OCCLUDED")
FIELDS = ("path", "episode", "state", "runtime_verdict", "cloud_verdict", "latency_ms")
CROP_RE = re.compile(r"^\d+_(?P<state>.+)_crop_(?P<verdict>[A-Z_]+)$")
SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def jpeg_size(path: Path) -> tuple[int, int] | None:
    """(width, height) from the JPEG SOF header (stdlib only), None if unreadable."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xFF, 0x01) or 0xD0 <= marker <= 0xD9:
            i += 2 if marker != 0xFF else 1
            continue
        seg_len = int.from_bytes(data[i + 2 : i + 4], "big")
        if marker in SOF:
            h = int.from_bytes(data[i + 5 : i + 7], "big")
            w = int.from_bytes(data[i + 7 : i + 9], "big")
            return w, h
        i += 2 + seg_len
    return None


def episode_is_real(ep_dir: Path, size: tuple[int, int]) -> bool | None:
    """True/False from the first raw wrist frame's size; None = no raw frame to tell."""
    for raw in sorted(ep_dir.glob("*_raw.jpg")):
        got = jpeg_size(raw)
        if got is not None:
            return got == size
    return None


def find_crops(episodes: Path, include_fake: bool) -> tuple[list[Path], list[str]]:
    wc = WristCameraConfig()
    size = (wc.width or 1920, wc.height or 1080)
    crops: list[Path] = []
    skipped: list[str] = []
    for session in sorted(episodes.glob("demo-*")):
        for ep_dir in sorted(p for p in (session / "frames").glob("*") if p.is_dir()):
            files = sorted(ep_dir.glob("**/*_crop_*.jpg"))
            if not files:
                continue
            real = episode_is_real(ep_dir, size)
            if real is False and not include_fake:
                skipped.append(f"{session.name}/{ep_dir.name} ({len(files)} crops)")
                continue
            if real is None:
                print(f"WARNING: {session.name}/{ep_dir.name}: no raw frame, size unknown: kept")
            crops += files
    return crops, skipped


def parse_crop(path: Path) -> tuple[str, str, str]:
    """(episode, state, runtime verdict) from frames/<episode>/<n>_<STATE>_crop_<VERDICT>.jpg."""
    m = CROP_RE.match(path.stem)
    state, verdict = (m["state"], m["verdict"]) if m else ("?", "?")
    return path.parent.name, state, verdict


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path.resolve())


def read_rows(csv_path: Path) -> dict[str, dict[str, str]]:
    """Last row per path (a retried TIMEOUT row is superseded by its later answer)."""
    rows: dict[str, dict[str, str]] = {}
    if csv_path.is_file():
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                rows[row["path"]] = row
    return rows


def report(rows: dict[str, dict[str, str]]) -> None:
    final = list(rows.values())
    cloud = Counter(r["cloud_verdict"] for r in final)
    print(f"\nper-class counts (cloud, {len(final)} images):")
    for c in (*CLASSES, *sorted(set(cloud) - set(CLASSES))):
        print(f"  {c:<9} {cloud.get(c, 0)}")
    run_cols = [*CLASSES, *sorted({r["runtime_verdict"] for r in final} - set(CLASSES))]
    cloud_cols = [*CLASSES, *sorted(set(cloud) - set(CLASSES))]
    m = Counter((r["runtime_verdict"], r["cloud_verdict"]) for r in final)
    w = max(9, *(len(c) for c in run_cols))
    print("\nagreement matrix (rows = runtime verdict, columns = cloud verdict):")
    print(f"  {'runtime':<{w}} " + " ".join(f"{c[:8]:>8}" for c in cloud_cols))
    for rv in run_cols:
        if any(m[(rv, cv)] for cv in cloud_cols):
            print(f"  {rv:<{w}} " + " ".join(f"{m[(rv, cv)]:>8}" for cv in cloud_cols))
    both = [r for r in final if r["runtime_verdict"] in CLASSES and r["cloud_verdict"] in CLASSES]
    agree = sum(r["runtime_verdict"] == r["cloud_verdict"] for r in both)
    if both:
        print(f"\nagreement (both a class): {agree}/{len(both)} = {agree / len(both):.1%}")


def build_parser() -> argparse.ArgumentParser:
    d = CloudCriticConfig()
    ap = argparse.ArgumentParser(
        description="Offline cloud auto-labelling of demo Critic crops (no arm I/O).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--episodes", type=Path, default=REPO / "data" / "episodes")
    ap.add_argument("--out", type=Path, default=Path("~/critic_auto"), help="class folders root")
    ap.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    ap.add_argument("--resume", nargs="?", const="latest", default=None,
                    help="continue runs/autolabel_<ts> (no value: the latest run)")  # fmt: skip
    ap.add_argument("--rate", type=float, default=30.0, help="max requests per minute")
    ap.add_argument("--limit", type=int, default=None, help="max requests in this invocation")
    ap.add_argument("--cloud-model", default=d.model)
    ap.add_argument("--include-fake", action="store_true", help="also label --dry-run episodes")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.rate <= 0:
        raise SystemExit("--rate must be > 0")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if args.resume == "latest":
        runs = sorted(args.runs_dir.glob("autolabel_*"))
        if not runs:
            raise SystemExit(f"--resume: no {args.runs_dir}/autolabel_* run")
        run_dir = runs[-1]
    elif args.resume:
        run_dir = Path(args.resume)
    else:
        run_dir = args.runs_dir / f"autolabel_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "labels.csv"
    out_root = args.out.expanduser()

    crops, skipped = find_crops(args.episodes, args.include_fake)
    for s in skipped:
        print(f"skipped (dry-run / fake wrist frames): {s}")
    rows = read_rows(csv_path)
    todo = [p for p in crops if rows.get(rel(p), {}).get("cloud_verdict") not in CLASSES]
    done = len(crops) - len(todo)
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"run dir: {run_dir}  (resume: --resume {run_dir})")
    print(f"crops: {len(crops)}, already labelled: {done}, to label now: {len(todo)}; "
          f"model {args.cloud_model}, {args.rate:g} req/min")  # fmt: skip

    if todo:
        client = AnthropicCriticClient(CloudCriticConfig(model=args.cloud_model))  # env key
        interval = 60.0 / args.rate
        new_file = not csv_path.is_file() or csv_path.stat().st_size == 0
        with csv_path.open("a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                wr.writeheader()
            t_next = time.monotonic()
            for k, path in enumerate(todo, 1):
                time.sleep(max(0.0, t_next - time.monotonic()))
                t_next = time.monotonic() + interval
                episode, state, runtime = parse_crop(path)
                reply = client.classify_jpeg(path.read_bytes(), CLOUD_PROMPT)
                row = {
                    "path": rel(path),
                    "episode": episode,
                    "state": state,
                    "runtime_verdict": runtime,
                    "cloud_verdict": reply.verdict.value,
                    "latency_ms": f"{reply.latency_ms:.0f}",
                }
                wr.writerow(row)
                f.flush()
                rows[row["path"]] = row
                note = ""
                if reply.verdict.value in CLASSES:
                    dst = out_root / reply.verdict.value / f"{episode}_{path.stem}.jpg"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dst)
                else:
                    note = f"  raw={reply.text!r} err={reply.error}"
                print(f"[{k}/{len(todo)}] {row['path']}: runtime {runtime} -> cloud "
                      f"{reply.verdict.value} ({row['latency_ms']} ms){note}")  # fmt: skip
    report(rows)
    print(f"\nlabels: {csv_path}\nimages: {out_root}/<class>/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
