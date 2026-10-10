"""Find cameras and measure their REAL frame rate and resolution.

WHY
    We need to know which camera index the Trinet wrist camera uses (open question
    OQ-7) and which resolution/FPS it can really deliver (OQ-8). Cameras often ignore
    the resolution/FPS you ask for, so we measure what actually comes out.

HOW TO USE
    1. List cameras that OpenCV can open:
           python tools/fps_test.py --list
    2. Measure one camera for 10 seconds at 1280x720 @ 30 FPS:
           python tools/fps_test.py --index 0 --width 1280 --height 720 --fps 30 --seconds 10
    3. Add --show to see the live picture (press q to stop early).

WHAT TO REPORT
    Copy the final "RESULT" line into docs/OPEN_QUESTIONS.md under OQ-7 / OQ-8.
    Try 1280x720@30 first, then 1920x1080@15 (bible 5.2). Run it once with the
    RealSense plugged in too, to check USB bandwidth (OQ-10).

SAFETY
    This script only reads cameras. It never talks to the robot arm.
"""

from __future__ import annotations

import argparse
import sys
import time


def list_cameras(max_index: int) -> None:
    import cv2

    print(f"Probing camera indices 0..{max_index - 1} (this can take a few seconds)...")
    found = 0
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, frame = cap.read()
            shape = f"{frame.shape[1]}x{frame.shape[0]}" if ok else "no frame"
            backend = cap.getBackendName()
            print(f"  index {i}: OPEN  default {shape}  backend={backend}")
            found += 1
        cap.release()
    if not found:
        print("  No cameras found. Is the camera plugged in? Try another USB port.")


def measure(index: int, width: int, height: int, fps: int, seconds: float, show: bool) -> int:
    import cv2

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera index {index}. Run with --list first.")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    print(f"Requested {width}x{height} @ {fps} FPS; measuring for {seconds:.0f} s ...")

    frames, failures = 0, 0
    real_w = real_h = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            failures += 1
            continue
        frames += 1
        real_h, real_w = frame.shape[:2]
        if show:
            cv2.imshow(f"camera {index} (q = quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    elapsed = time.perf_counter() - t0
    cap.release()
    if show:
        cv2.destroyAllWindows()

    measured = frames / elapsed if elapsed > 0 else 0.0
    print(
        f"RESULT index={index} requested={width}x{height}@{fps} "
        f"got={real_w}x{real_h} measured_fps={measured:.1f} failed_reads={failures}"
    )
    if real_w != width or real_h != height:
        print("NOTE: the camera did not give the resolution you asked for.")
    if measured < 0.9 * fps:
        print("NOTE: FPS is below 90% of the request. Try a lower resolution or another USB port.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--list", action="store_true", help="list camera indices that open")
    p.add_argument("--max-index", type=int, default=10, help="how many indices to probe")
    p.add_argument("--index", type=int, default=0, help="camera index to measure")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--show", action="store_true", help="show the live picture")
    args = p.parse_args()
    if args.list:
        list_cameras(args.max_index)
        return 0
    return measure(args.index, args.width, args.height, args.fps, args.seconds, args.show)


if __name__ == "__main__":
    sys.exit(main())
