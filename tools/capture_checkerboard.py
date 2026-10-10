"""Capture checkerboard photos from the Trinet for fisheye calibration (OQ-9).

WHY
    The Trinet has a very wide (fisheye) lens. Before its images go to the Critic we
    "dewarp" them so straight lines look straight. To do that we first need photos of a
    checkerboard from many angles.

WHAT YOU NEED
    A printed checkerboard glued flat on stiff card. Count the INNER corners (where four
    squares meet), not the squares. A board with 10x7 squares has 9x6 inner corners.
    Measure one square's side with calipers in mm (needed by calibrate_fisheye.py).

HOW TO USE
    python tools/capture_checkerboard.py --index 0 --width 1280 --height 720 --board 9x6

    A window opens. When the board is detected, coloured corner dots are drawn.
        SPACE  save the current frame (only if the board is detected, unless --force)
        q      quit

TIPS FOR A GOOD CALIBRATION
    * Save 15-25 photos.
    * Cover the WHOLE picture: put the board near every edge and corner, where the
      fisheye distortion is strongest - not only in the centre.
    * Tilt the board (up to ~45 degrees) in different directions. Fill a good part of
      the frame in some shots.
    * Keep it still and sharp (no motion blur), with even lighting.
    * Use the SAME resolution you will use on the robot (see fps_test.py results).

OUTPUT
    PNG files in data/calib/trinet_frames/ (default). Next: tools/calibrate_fisheye.py

SAFETY
    Camera only. Never talks to the robot arm.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def parse_board(text: str) -> tuple[int, int]:
    cols, rows = (int(v) for v in text.lower().split("x"))
    return cols, rows


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--index", type=int, default=0, help="camera index (see fps_test.py --list)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--board", default="9x6", help="INNER corners, columns x rows")
    p.add_argument("--out", type=Path, default=Path("data/calib/trinet_frames"))
    p.add_argument("--force", action="store_true", help="save even if the board is not found")
    args = p.parse_args()

    import cv2

    board = parse_board(args.board)
    args.out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.index)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {args.index}. Try tools/fps_test.py --list")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    saved = len(list(args.out.glob("*.png")))
    print(f"Saving to {args.out} ({saved} already there). SPACE = save, q = quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("WARNING: failed to read a frame")
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, board, flags)
        preview = frame.copy()
        if found:
            cv2.drawChessboardCorners(preview, board, corners, found)
        status = "BOARD FOUND" if found else "board not found"
        cv2.putText(
            preview,
            f"{status} | saved {saved} | SPACE save, q quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if found else (0, 0, 255),
            2,
        )
        cv2.imshow("capture_checkerboard", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            if not found and not args.force:
                print("Not saved: board not detected (use --force to override).")
                continue
            path = args.out / f"cb_{time.strftime('%Y%m%d_%H%M%S')}_{saved:03d}.png"
            cv2.imwrite(str(path), frame)  # save the clean frame, not the preview
            saved += 1
            print(f"saved {path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. {saved} images in {args.out}. Next: python tools/calibrate_fisheye.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
