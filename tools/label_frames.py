"""Keyboard labeler for the Critic dataset (bible Section 16).

WHY
    The Critic learns from wrist-camera images sorted into four classes. This tool
    shows each unlabeled image and you press one key to file it.

CLASSES (look at the gripper bite zone)
    1  GRASPED   object firmly between the fingers
    2  SLIPPING  object in the fingers but tilted / sliding / partly out
    3  EMPTY     fingers closed (or open) with no object
    4  OCCLUDED  you cannot tell: view blocked (cable, hand, glare, too dark)

KEYS
    1-4  move the image into that class folder
    s    skip (leave it where it is)
    u    undo the last move
    q    quit (progress is kept: labeled files are already moved)

HOW TO USE
    python tools/label_frames.py --src data/frames/unlabeled --dst data/datasets/critic

    Images are MOVED (not copied) to data/datasets/critic/<CLASS>/.
    If unsure between two classes, prefer OCCLUDED only when the view is truly
    blocked; otherwise pick your best guess and tell the software lead.

SAFETY
    Files only. Never talks to the robot arm.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

CLASSES = {ord("1"): "GRASPED", ord("2"): "SLIPPING", ord("3"): "EMPTY", ord("4"): "OCCLUDED"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def counts(dst: Path) -> str:
    return "  ".join(
        f"{c}={len(list((dst / c).glob('*')))}" for c in CLASSES.values() if (dst / c).exists()
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--src", type=Path, required=True, help="folder with unlabeled images")
    p.add_argument("--dst", type=Path, default=Path("data/datasets/critic"))
    args = p.parse_args()

    import cv2

    files = sorted(f for f in args.src.rglob("*") if f.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        print(f"No images found in {args.src}")
        return 1
    for c in CLASSES.values():
        (args.dst / c).mkdir(parents=True, exist_ok=True)

    history: list[tuple[Path, Path]] = []  # (new_path, old_path) for undo
    i = 0
    while i < len(files):
        f = files[i]
        img = cv2.imread(str(f))
        if img is None:
            print(f"cannot read {f}, skipping")
            i += 1
            continue
        view = img.copy()
        cv2.putText(
            view,
            f"[{i + 1}/{len(files)}] 1 GRASPED 2 SLIPPING 3 EMPTY 4 OCCLUDED",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            view,
            "s skip  u undo  q quit",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.imshow("label_frames", view)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            i += 1
        elif key == ord("u"):
            if history:
                new, old = history.pop()
                shutil.move(str(new), str(old))
                i -= 1
                print(f"undo: {new.name} back to {old.parent}")
        elif key in CLASSES:
            target = args.dst / CLASSES[key] / f.name
            if target.exists():
                target = target.with_name(f"{f.stem}_{i}{f.suffix}")
            shutil.move(str(f), str(target))
            history.append((target, f))
            print(f"{CLASSES[key]:9s} <- {f.name}")
            i += 1
    cv2.destroyAllWindows()
    print(f"Counts: {counts(args.dst)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
