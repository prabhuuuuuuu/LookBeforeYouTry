"""Compute the Trinet fisheye calibration (K, D) from checkerboard photos (OQ-9).

WHY
    The Critic must see "dewarped" (rectilinear) images (bible 5.2). This script turns
    the photos from capture_checkerboard.py into a small YAML file that the robot
    software loads once at startup to build its dewarp maps.

HOW TO USE
    python tools/calibrate_fisheye.py --frames data/calib/trinet_frames --board 9x6 \\
        --square-mm 25 --out data/calib/trinet_fisheye.yaml --preview

    --board      INNER corners, columns x rows (same as when capturing)
    --square-mm  side length of ONE square, measured with calipers
    --preview    also writes <out>_preview.jpg: original left, dewarped right

HOW TO READ THE RESULT
    "RMS reprojection error" is in pixels. Below ~0.5 px is great, below 1.0 px is OK.
    Above 1 px: retake photos (blurry? board not flat? too few edge/corner shots?).
    Look at the preview: straight edges in the world should look straight on the right.

OUTPUT
    data/calib/trinet_fisheye.yaml (K, D, image_size, rms). Record the RMS in
    docs/OPEN_QUESTIONS.md under OQ-9.

SAFETY
    Offline computation only. Never talks to the robot arm.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--frames", type=Path, default=Path("data/calib/trinet_frames"))
    p.add_argument("--board", default="9x6", help="INNER corners, columns x rows")
    p.add_argument("--square-mm", type=float, required=True, help="square side length in mm")
    p.add_argument("--out", type=Path, default=Path("data/calib/trinet_fisheye.yaml"))
    p.add_argument("--preview", action="store_true", help="write a before/after image")
    args = p.parse_args()

    import cv2

    from lbyr.errors import CalibrationError
    from lbyr.perception.dewarp import Dewarper, calibrate_fisheye

    cols, rows = (int(v) for v in args.board.lower().split("x"))
    objp = np.zeros((cols * rows, 3), np.float64)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * args.square_mm

    files = sorted(args.frames.glob("*.png")) + sorted(args.frames.glob("*.jpg"))
    if not files:
        print(f"ERROR: no images in {args.frames}. Run capture_checkerboard.py first.")
        return 1

    obj_pts, img_pts, size = [], [], None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    for f in files:
        img = cv2.imread(str(f))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        this_size = (gray.shape[1], gray.shape[0])
        if size is None:
            size = this_size
        elif this_size != size:
            print(f"skip {f.name}: size {this_size} differs from {size}")
            continue
        found, corners = cv2.findChessboardCorners(gray, (cols, rows))
        if not found:
            print(f"skip {f.name}: board not found")
            continue
        corners = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)
        obj_pts.append(objp)
        img_pts.append(corners)
        print(f"ok   {f.name}")

    assert size is not None
    print(f"Using {len(obj_pts)} of {len(files)} images at {size[0]}x{size[1]}")
    try:
        calib = calibrate_fisheye(obj_pts, img_pts, size)
    except (CalibrationError, cv2.error) as e:
        print(f"ERROR: calibration failed: {e}")
        print("Usually: a bad/blurry image. Remove suspicious images and try again.")
        return 1

    calib.save(args.out)
    print(f"RMS reprojection error: {calib.rms:.3f} px")
    print(f"K =\n{calib.K}\nD = {calib.D}")
    print(f"Saved {args.out}")

    if args.preview:
        img = cv2.imread(str(files[0]))
        out = Dewarper(calib, balance=0.0)(img)
        side = np.hstack([img, out])
        prev = args.out.with_name(args.out.stem + "_preview.jpg")
        cv2.imwrite(str(prev), side)
        print(f"Preview (left original, right dewarped): {prev}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
