"""Live demo: taught poses + Qwen2.5-VL plan + Moondream2 Critic + web view.

    python -m lbyr.demo --rehearsal          # real arm + cameras + web, rules plan, fake Critic,
                                             # NO model loading (motion-only bring-up)
    python -m lbyr.demo                      # full: Qwen plan + Moondream Critic
    python -m lbyr.demo --planner rules      # real Critic, no Qwen (fast trials)
    python -m lbyr.demo --induce-slip        # loose first grasp (induce_slip_gripper_deg)
    python -m lbyr.demo --critic-off         # baseline: verify states pass through
    python -m lbyr.demo --dry-run            # MockArm + fake cameras + scripted Critic

    python -m lbyr.demo --task "move the block from A to the place zone"

Language prompting: type an instruction in the web page (Run) or at the console, or pass
--task (Enter runs it). Qwen2.5-VL-3B sees the overhead image, the instruction, the
poses.yaml ``locations`` (name + description) and the detector's block location, and answers
strict JSON move(from, to) | none (lbyr/ai/task_planner.py, 30 s, greedy). Planning happens
BEFORE any motion; invalid / "none" -> nothing moves. Location-like names that are not
taught ("zone C") are refused before Qwen is called; location polygons (if defined) are
drawn and labelled on Qwen's image. Qwen is loaded ONCE and stays resident (unloading
does not return memory on the Jetson; a reload costs ~11 s). While an episode runs, new
instructions are ignored ("busy"). NOTE: anyone who can reach the web page can start motion.

Keys (console, between episodes): Enter = run, s = run with induced slip, w = wrist snapshot
(runs/<ts>/snap_<n>_{full,crop}.jpg, also GET /snapshot?crop=x0,y0,x1,y1), q = quit, any
other text = an instruction.
Web view: http://<jetson>:8000 (overhead + wrist MJPEG, instruction box, state, plan, verdicts).

Sequence (one ``m`` per segment, DONE awaited via the MotionController, PP-1):
home -> zone_A_pre -> zone_A_grasp -> close -> VERIFY_GRASP -> LIFT (zone_A_pre, monitored)
-> VERIFY_HOLD -> TRANSPORT (place_pre, monitored) -> place -> open -> VERIFY_PLACE -> home.
SLIPPING / EMPTY at a verify: re-open, re-grasp at +15% closure (beyond closed_deg, never
past 0 = fully shut), at most 2 times, then abort to home holding the gripper.
Monitored segments: the Critic worker judges the latest wrist frame back-to-back while the
FSM waits for DONE; 2 consecutive SLIPPING / EMPTY (p >= 0.6) = DROP -> halt (an ``m`` to the
current position from ``q``: the firmware's ``m`` pre-empts a running move), then abort home
with the gripper OPEN (a dropped block is not re-grasped). VERIFY_PLACE expects EMPTY; still
GRASPED / SLIPPING = "FAILURE: object stuck in gripper". Every verdict (state, verdict, the 4
probabilities, latency, crop path) goes to episodes.jsonl and the web page; failure lines
start with "FAILURE:" (red); the episode end line is SUCCESS / RECOVERED (n re-grasps) /
FAILED (<class> at <state>). Poses: data/calib/poses.yaml (zero-relative degrees).

Critic backends: probe (default when ~/models/critic_probe.pt exists: Moondream2 vision
encoder + logistic head, verdict + probabilities, max p < 0.55 = LOW_CONF -> re-query) |
moondream (zero-shot text) | scripted (no model; --script, e.g.
"VERIFY_GRASP=EMPTY:0.91,GRASPED:0.95;TRANSPORT=GRASPED,SLIPPING:0.78,SLIPPING:0.81") | off.

Cost: Qwen2.5-VL-3B bf16 ~6.5 GB, one plan (~11 s, 30 s cap) before the motion; Moondream2
bf16 ~4.1 GB either way; probe ~90 ms per verdict (monitor ~7 Hz), zero-shot ~0.85 s
(bible 9.2). Monitor: one 378x378 JPEG per verdict (logger thread). MJPEG ~3 ms/frame/viewer.
Exit codes: 0 = DONE, 1 = ABORT / FAILED, 2 = setup refused (hardware / calibration / models).
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import queue
import re
import select
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from lbyr.ai.critic import (
    CriticBackend,
    CriticResult,
    CriticWorker,
    ScriptedCritic,
    Verdict,
    parse_script,
)
from lbyr.ai.imaging import critic_crop
from lbyr.ai.planner import (
    PlanRequest,
    PlanResult,
    QwenPlanner,
    RuleBasedPlanner,
    plan_demo_intro,
    plan_with_timeout,
)
from lbyr.ai.task_planner import (
    UNKNOWN,
    RulesTaskPlanner,
    TaskPlanResult,
    build_task_prompt,
    clean_instruction,
    plan_task,
)
from lbyr.arm.calib import JointsCalib, load_joints, mock_joints
from lbyr.arm.interface import ArmInterface
from lbyr.config import (
    GRIPPER_INDEX,
    AppConfig,
    DemoConfig,
    WristCameraConfig,
    mock_profile,
    require,
)
from lbyr.errors import CalibrationError, LBYRError, PlanValidationError
from lbyr.log_setup import setup_logging
from lbyr.logging_.episodes import EpisodeLogger, VramMonitor, make_record, read_system_ram
from lbyr.motion.controller import FAULT, MOVING, MotionController
from lbyr.motion.taught import point_in_polygon
from lbyr.perception.buffers import LatestFrameBuffer
from lbyr.perception.detector import Detection
from lbyr.perception.overhead import FakeObject, FakeOverheadCamera, FakeScene, RGBDFrame
from lbyr.perception.wrist import FakeWristCamera, decode_wrist_jpeg

log = logging.getLogger("lbyr.demo")

REQUIRED = ("home", "zone_A_pre", "zone_A_grasp", "place_pre", "place")
BLOCK_ID, PLACE_ID = 1, 900
GOOD = (Verdict.GRASPED, Verdict.SLIPPING, Verdict.EMPTY)  # decisive Critic answers
NOT_HELD = (Verdict.SLIPPING, Verdict.EMPTY)
VERDICTS_SHOWN = 60  # web CRITIC panel keeps the newest N (jsonl keeps all)


class DemoAbort(LBYRError):
    """Stop the sequence and retreat to home.

    ``failure_class`` / ``failure_state`` name the failure for the end line ("FAILED (EMPTY
    at VERIFY_GRASP)"); None = derived from the message (LINK) and the current state.
    ``open_gripper``: retreat with the gripper open (DROP: nothing left to hold)."""

    def __init__(
        self,
        msg: str,
        failure_class: str | None = None,
        failure_state: str | None = None,
        open_gripper: bool = False,
    ) -> None:
        super().__init__(msg)
        self.failure_class = failure_class
        self.failure_state = failure_state
        self.open_gripper = open_gripper


def fmt_vote(verdict: str, probs: Mapping[str, float] | None) -> str:
    """Short vote text: ``EMPTY 0.91``; LOW_CONF shows its top class; no probs: the verdict."""
    if not probs:
        return verdict
    top = verdict if verdict in probs else max(probs, key=lambda k: probs[k])
    return f"{top} {probs[top]:.2f}"


def _probs_str(probs: Mapping[str, float] | None) -> str:
    """All 4 probabilities: ``[GRASPED 0.03 SLIPPING 0.78 ...]``; ``[no probs]`` (zero-shot)."""
    if not probs:
        return "[no probs]"
    return "[" + " ".join(f"{k} {v:.2f}" for k, v in probs.items()) + "]"


def fmt_votes(votes: list[CriticResult]) -> str:
    """DROP votes: "SLIPPING 0.78, 0.81" (same class) or "SLIPPING 0.78, EMPTY 0.81"."""
    names = {v.verdict.value for v in votes}
    if len(names) == 1 and all(v.probs for v in votes):
        name = names.pop()
        return f"{name} " + ", ".join(f"{(v.probs or {})[name]:.2f}" for v in votes)
    return ", ".join(fmt_vote(v.verdict.value, v.probs) for v in votes)


# ------------------------------------------------------------------ poses
@dataclass(frozen=True)
class Location:
    """Named table location (poses.yaml ``locations.<name>``): pick source or place target.
    ``pre`` / ``grasp`` name taught poses; ``polygon_px`` (optional, overhead pixels) lets
    the block detector say where the block is."""

    name: str
    description: str
    pre: str
    grasp: str
    polygon_px: tuple[tuple[float, float], ...] | None = None


# The default episode (Enter, no instruction): zone A -> place, the original pose names.
DEFAULT_SRC = Location("A", "", "zone_A_pre", "zone_A_grasp")
DEFAULT_DST = Location("place", "", "place_pre", "place")


@dataclass(frozen=True)
class DemoPoses:
    """Taught poses in ZERO-RELATIVE degrees (poses.yaml / arm_gui slider values)."""

    poses: dict[str, tuple[float, ...]]
    open_deg: float
    closed_deg: float
    place_polygon_px: tuple[tuple[float, float], ...] | None = None
    locations: dict[str, Location] = field(default_factory=dict)

    def gripper_deg(self, closure_pct: float) -> float:
        """0 % = open_deg, 100 % = closed_deg (closed on the block); > 100 % squeezes further
        toward 0 deg = fully shut and is clamped there. If 0 is not beyond closed_deg, no
        over-closure is possible: clamped at closed_deg."""
        o, c = self.open_deg, self.closed_deg
        v = o + max(closure_pct, 0.0) / 100.0 * (c - o)
        if (c - o) * (0.0 - c) >= 0:  # 0 lies at/beyond closed_deg in the closing direction
            return max(v, 0.0) if c < o else min(v, 0.0)
        return max(v, c) if c < o else min(v, c)

    def pct_for_deg(self, gripper_deg: float) -> float:
        """Closure % that :meth:`gripper_deg` maps to ``gripper_deg`` (inverse map)."""
        return 100.0 * (gripper_deg - self.open_deg) / (self.closed_deg - self.open_deg)

    def pose(self, name: str, closure_pct: float) -> np.ndarray:
        q = np.asarray(self.poses[name], dtype=float).copy()
        q[GRIPPER_INDEX] = self.gripper_deg(closure_pct)
        return q


def load_demo_poses(path: Path) -> DemoPoses:
    import yaml

    if not path.is_file():
        raise CalibrationError(f"{path} not found (teach poses with tools/arm_gui.py)")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    poses = {str(k): tuple(float(x) for x in v) for k, v in (data.get("poses") or {}).items()}
    g = data.get("gripper") or {}
    missing = [p for p in REQUIRED if p not in poses]
    missing += [f"gripper.{k}" for k in ("open_deg", "closed_deg") if g.get(k) is None]
    if missing:
        raise CalibrationError(f"{path}: not taught yet: {', '.join(missing)}")
    poly = data.get("place_polygon_px")
    p = DemoPoses(
        poses,
        float(g["open_deg"]),
        float(g["closed_deg"]),
        tuple((float(a), float(b)) for a, b in poly) if poly else None,
        parse_locations(data.get("locations"), poses, path),
    )
    if p.gripper_deg(130.0) == p.closed_deg:
        log.info(
            "gripper: closed_deg %.0f is the limit (0 = fully shut): re-grasps close to %.0f deg",
            p.closed_deg,
            p.closed_deg,
        )
    return p


def parse_locations(
    raw: Any, poses: dict[str, tuple[float, ...]], path: Path | None = None
) -> dict[str, Location]:
    """poses.yaml ``locations``: ``<name>: {description, pre, grasp, polygon_px?}``; pre /
    grasp must name taught poses. Missing section: A and place from the original poses."""
    if not raw:
        return {loc.name: loc for loc in (DEFAULT_SRC, DEFAULT_DST)}
    if not isinstance(raw, dict):
        raise CalibrationError(f"{path}: locations must be a mapping name -> {{pre, grasp}}")
    out: dict[str, Location] = {}
    for name, v in raw.items():
        v = v or {}
        bad = [k for k in ("pre", "grasp") if v.get(k) not in poses]
        if bad:
            what = ", ".join(f"{k}={v.get(k)!r}" for k in bad)
            raise CalibrationError(f"{path}: locations.{name}: {what} is not a taught pose")
        poly = v.get("polygon_px")
        if poly is not None and len(poly) < 3:
            raise CalibrationError(f"{path}: locations.{name}.polygon_px needs >= 3 points")
        desc = str(v.get("description") or "").strip()
        if not desc:
            log.warning("poses.yaml locations.%s has no description (the planner sees none)", name)
        out[str(name)] = Location(
            str(name),
            desc,
            str(v["pre"]),
            str(v["grasp"]),
            tuple((float(a), float(b)) for a, b in poly) if poly else None,
        )
    return out


_NAMED = re.compile(r"\b(?:zone|location|area|spot|position|point)\s+([A-Za-z]|\d+)\b", re.I)
_LETTER = re.compile(r"\b([A-HJ-Z])\b")  # a lone capital letter (not "I") names a location


def unknown_locations(instruction: str, locations: Mapping[str, Location]) -> list[str]:
    """Location-like names in the instruction ("zone C", "C", "area 3") that are NOT taught
    locations (case-insensitive). Checked BEFORE Qwen: an unknown place never reaches it."""
    known = {n.lower() for n in locations}
    found = _NAMED.findall(instruction) + _LETTER.findall(instruction)
    return list(dict.fromkeys(x for x in found if x.lower() not in known))


def label_locations(image: np.ndarray, locations: Mapping[str, Location]) -> np.ndarray:
    """Overhead image for Qwen with every location polygon outlined and its NAME in large
    letters (Qwen cannot know which area is "A" otherwise). No polygons: ``image`` itself.
    Cost: a few cv2 draw calls on a copy (< 2 ms)."""
    polys = [(n, x.polygon_px) for n, x in locations.items() if x.polygon_px]
    if not polys:
        return image
    import cv2  # lazy

    out = image.copy()
    scale = max(1.0, out.shape[1] / 640)
    for name, poly in polys:
        pts = np.round(np.asarray(poly)).astype(np.int32)
        cv2.polylines(out, [pts], True, (0, 255, 255), max(2, round(2 * scale)))
        cx, cy = pts.mean(axis=0)
        font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 1.6 * scale, max(3, round(4 * scale))
        (w, h), _ = cv2.getTextSize(name, font, fs, th)
        org = (int(cx - w / 2), int(cy + h / 2))
        cv2.putText(out, name, org, font, fs, (0, 0, 0), th + 4, cv2.LINE_AA)
        cv2.putText(out, name, org, font, fs, (0, 255, 255), th, cv2.LINE_AA)
    return out


def mock_demo_poses() -> DemoPoses:
    """SIMULATION-ONLY (FAKE) poses for --dry-run."""
    fake: dict[str, tuple[float, ...]] = {
        "home": (0.0, -30.0, 90.0, 30.0, 0.0, 0.0),
        "zone_A_pre": (15.0, 10.0, 60.0, 40.0, 0.0, 0.0),
        "zone_A_grasp": (15.0, 25.0, 55.0, 50.0, 0.0, 0.0),
        "place_pre": (-25.0, 10.0, 60.0, 40.0, 0.0, 0.0),
        "place": (-25.0, 25.0, 55.0, 50.0, 0.0, 0.0),
        "zone_B_pre": (40.0, 10.0, 60.0, 40.0, 0.0, 0.0),
        "zone_B_grasp": (40.0, 25.0, 55.0, 50.0, 0.0, 0.0),
    }
    locs = {
        "A": replace(DEFAULT_SRC, description="FAKE: left side of the table"),
        "B": Location("B", "FAKE: back of the table", "zone_B_pre", "zone_B_grasp"),
        "place": replace(DEFAULT_DST, description="FAKE: right side of the table"),
    }
    return DemoPoses(fake, open_deg=-40.0, closed_deg=0.0, locations=locs)  # arm's gripper map


# ------------------------------------------------------------------ live status (web panel)
@dataclass
class DemoStatus:
    state: str = "STARTUP"
    plan: dict[str, Any] = field(default_factory=dict)
    verdicts: list[dict[str, Any]] = field(default_factory=list)
    recovery: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)  # this episode's "FAILURE: ..." lines
    events: list[str] = field(default_factory=list)
    ram: str = "n/a"
    closure_pct: float | None = None
    outcome: str | None = None
    episode: int = 0
    mode: str = ""
    busy: bool = False  # an episode is queued / running: new instructions are ignored
    task: str = ""  # the instruction being run ("" = default episode)
    result: str = ""  # big line on the PLAN panel: SUCCESS ... / NONE: reason / ERROR: ...
    result_kind: str = ""  # running | success | abort | none | rejected (CSS class)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set(self, **kw: Any) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def event(self, msg: str, level: int = logging.INFO) -> None:
        log.log(level, "%s", msg)
        with self._lock:
            self.events.append(f"{time.strftime('%H:%M:%S')} {msg}")
            del self.events[:-40]

    def add(self, key: str, item: Any, keep: int | None = None) -> None:
        with self._lock:
            items = getattr(self, key)
            items.append(item)
            if keep is not None:
                del items[:-keep]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                k: (list(v) if isinstance(v, list) else v)
                for k, v in vars(self).items()
                if not k.startswith("_")
            }


# ------------------------------------------------------------------ cameras
def open_wrist(device: int | str, wc: WristCameraConfig) -> tuple[Any, np.ndarray]:
    """Open the Trinet in its ONE mode (MJPG 1920x1080@30, OQ-8) and read a first frame.
    No fallback mode: anything else raises."""
    import cv2  # lazy

    w, h, fps = (
        require(wc.width, "Trinet width", "OQ-8"),
        require(wc.height, "Trinet height", "OQ-8"),
        require(wc.fps, "Trinet FPS", "OQ-8"),
    )
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise LBYRError(f"cannot open the wrist camera {device!r} (use --wrist-dev)")
    if wc.fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*wc.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # OpenCV's own MJPEG decode returns all-zero frames for the Trinet (SEI blob inside the
    # DHT segment): take the raw JPEG and decode it with decode_wrist_jpeg.
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    ok, raw = cap.read()
    frame = decode_wrist_jpeg(raw) if ok and raw is not None and raw.size else None
    if frame is None:
        cap.release()
        raise LBYRError(f"wrist camera {device!r} opened but gave no decodable frame")
    if (frame.shape[1], frame.shape[0]) != (w, h):
        cap.release()
        raise LBYRError(
            f"wrist camera {device!r} gives {frame.shape[1]}x{frame.shape[0]}, need {w}x{h} "
            f"{wc.fourcc} (the Trinet's only mode, OQ-8)"
        )
    return cap, frame


def _wrist_candidates() -> list[str]:
    """Non-RealSense V4L2 nodes: "Trinet" capture nodes (sysfs index 0) first, so the first
    open normally hits the right node (every extra open/close cycle costs a USB drop)."""
    found: list[tuple[tuple[bool, bool, int], str]] = []
    for d in glob.glob("/sys/class/video4linux/video*"):
        try:
            name = Path(d, "name").read_text().strip()
            index = Path(d, "index").read_text().strip()
        except OSError:
            continue
        if "realsense" in name.lower():
            continue
        key = ("trinet" not in name.lower(), index != "0", int(d.rsplit("video", 1)[1]))
        found.append((key, f"/dev/{Path(d).name}"))
    return [dev for _, dev in sorted(found)]


def open_wrist_wait(
    devices: Callable[[], list[str]], wc: WristCameraConfig, wait_s: float | None = None
) -> tuple[str, Any, np.ndarray]:
    """Open the first device that delivers a frame; keep it OPEN (returned).

    Trinet quirk (measured 2026-10-11): after a stream is closed, the NEXT open gets no frame
    and the camera drops off USB and re-enumerates (~6-20 s); the open after that works. So
    never probe-and-close, and retry for ``wait_s`` (default ``wc.open_wait_s``). Nodes
    without frames (metadata, e.g. /dev/video7) are skipped."""
    wait_s = wc.open_wait_s if wait_s is None else wait_s
    t_end = time.perf_counter() + wait_s
    errors: list[str] = []
    announced = False
    while True:
        for dev in devices():
            try:
                cap, frame = open_wrist(dev, wc)
            except Exception as e:
                errors.append(f"{dev}: {e}")
                log.info("wrist: skip %s (%s)", dev, e)
                continue
            log.info("wrist: using %s", dev)
            return dev, cap, frame
        if time.perf_counter() >= t_end:
            raise LBYRError(
                f"no wrist camera frame within {wait_s:.0f} s: {errors[-3:] or 'no candidates'}"
                f"\nvideo devices seen:\n{video_devices_report()}\n(use --wrist-dev /dev/videoN)"
            )
        if not announced:
            log.warning("wrist camera not ready: retrying for up to %.0f s", wait_s)
            announced = True
        time.sleep(1.0)


def video_devices_report() -> str:
    """``v4l2-ctl --list-devices`` output, else the /sys/class/video4linux names (for the
    wrist-camera error). Cost: one subprocess (< 2 s cap), only on the failure path."""
    import subprocess

    try:
        r = subprocess.run(
            ["v4l2-ctl", "--list-devices"], capture_output=True, text=True, timeout=2.0
        )
        out = (r.stdout + r.stderr).strip()
        if out:
            return "  " + out.replace("\n", "\n  ")
    except (OSError, subprocess.SubprocessError) as e:
        log.debug("v4l2-ctl unavailable: %s", e)
    lines = []
    for d in sorted(glob.glob("/sys/class/video4linux/video*")):
        try:
            name = Path(d, "name").read_text().strip()
            index = Path(d, "index").read_text().strip()
        except OSError:
            name, index = "?", "?"
        lines.append(f"  /dev/{Path(d).name}: {name} (index {index})")
    return "\n".join(lines) or "  none (/sys/class/video4linux is empty)"


class WristCapture:
    """Trinet capture thread: ``grab()`` every frame (30 FPS, keeps the 1-frame buffer fresh,
    no decode); ``retrieve()`` of the RAW JPEG + :func:`decode_wrist_jpeg` (1080p, ~15 ms
    CPU) only at ``retrieve_hz`` (12 Hz, ~18% of one core) and on demand (:meth:`fresh`,
    for the Critic)."""

    def __init__(self, device: str, wc: WristCameraConfig, opened: Any = None) -> None:
        """``opened`` = (cap, first frame) from :func:`open_wrist_wait` (kept open)."""
        self.device = device
        self.wc = wc
        self._opened = opened
        self.buffer: LatestFrameBuffer[np.ndarray] = LatestFrameBuffer()
        self.grabs = self.retrieves = self.reconnects = 0
        self._cap: Any = None
        self._want = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="wrist-cam", daemon=True)
        self._t0 = 0.0

    def start(self) -> None:
        if self._opened is not None:
            self._cap, frame = self._opened
            self._opened = None
        else:
            _, self._cap, frame = open_wrist_wait(lambda: [self.device], self.wc)
        self.buffer.put(frame)
        self._t0 = time.perf_counter()
        self._thread.start()
        log.info("wrist camera %r: %dx%d MJPG", self.device, frame.shape[1], frame.shape[0])

    def _reconnect(self) -> None:
        """The Trinet can drop off USB and re-enumerate (~3-20 s): re-open every 1 s, for as
        long as it takes (the Critic sees OCCLUDED meanwhile, see :meth:`fresh`)."""
        log.warning("wrist camera %r lost: reconnecting", self.device)
        try:
            self._cap.release()
        except Exception:
            log.exception("wrist release failed")
        while not self._stop.is_set():
            try:
                self._cap, frame = open_wrist(self.device, self.wc)
            except Exception as e:
                log.debug("wrist reopen failed: %s", e)
                self._stop.wait(1.0)
                continue
            self.buffer.put(frame)
            self.reconnects += 1
            log.warning("wrist camera %r reconnected (%d)", self.device, self.reconnects)
            return

    def _run(self) -> None:
        period = 1.0 / self.wc.retrieve_hz
        due = 0.0
        fails = 0
        while not self._stop.is_set():
            try:  # a cv2 error must never kill the thread (no frames ever again)
                grabbed = self._cap.grab()
            except Exception:
                log.exception("wrist grab failed")
                grabbed = False
            if not grabbed:
                fails += 1
                if fails >= 20:  # ~0.2 s without a frame: device gone
                    self._reconnect()
                    fails = 0
                self._stop.wait(0.01)
                continue
            fails = 0
            self.grabs += 1
            now = time.perf_counter()
            if self._want.is_set() or now >= due:
                try:
                    ok, raw = self._cap.retrieve()
                    good = ok and raw is not None and raw.size
                    frame = decode_wrist_jpeg(raw) if good else None
                except Exception:
                    log.exception("wrist retrieve/decode failed")
                    frame = None
                if frame is not None:
                    self.buffer.put(frame)
                    self.retrieves += 1
                    due = max(due + period, now)  # deadline schedule: averages retrieve_hz
                    self._want.clear()

    def fresh(self, timeout_s: float | None = None) -> np.ndarray | None:
        """A frame grabbed AFTER this call (Critic input), or None after ``timeout_s``
        (default ``wc.fresh_frame_timeout_s``, 5 s): normally <= 1 frame period; a USB
        re-enumeration can take ~20 s, and the caller treats None as OCCLUDED."""
        item = self.buffer.get()
        seq = item.seq if item is not None else 0
        self._want.set()
        timeout_s = self.wc.fresh_frame_timeout_s if timeout_s is None else timeout_s
        t_end = time.perf_counter() + timeout_s
        while time.perf_counter() < t_end:
            new = self.buffer.get_if_newer(seq)
            if new is not None:
                return new.value
            time.sleep(0.005)
        return None

    def rates(self) -> tuple[float, float]:
        dt = max(time.perf_counter() - self._t0, 1e-6)
        return self.grabs / dt, self.retrieves / dt

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(2.0)
            if self._thread.is_alive():
                log.warning("wrist-cam thread did not stop within 2 s")
        if self._opened is not None:  # opened but never started
            self._opened[0].release()
            self._opened = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None


# ------------------------------------------------------------------ web
PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>LBYR live</title><style>
body{background:#111;color:#ddd;font:14px monospace;margin:10px}
img{width:49%;border:1px solid #444}#st{font-size:28px;color:#4f4}
.box{border:1px solid #444;padding:6px;margin:6px 0;white-space:pre-wrap}
.SLIPPING,.EMPTY,.ABORT,.FAILED,.fail{color:#f55}.GRASPED,.DONE,.ok{color:#4f4}
.OCCLUDED,.TIMEOUT,.LOW_CONF,.INVALID{color:#fa0}.fail{font-weight:bold}.recovered{color:#cf4}
input{background:#222;color:#eee;border:1px solid #555;font:16px monospace;width:60%}
button{font:16px monospace}#tmsg{color:#fa0}button:disabled{opacity:.4}
.big{font-size:26px;font-weight:bold;white-space:pre-wrap}.running{color:#8cf}
.success{color:#4f4}.none{color:#fa0}.abort,.rejected{color:#f55}
</style></head><body><img src="/overhead.mjpg"><img src="/wrist.mjpg">
<div><input id="task" placeholder="e.g. move the block from A to the place zone"
maxlength="300"> <button id="go" onclick="runTask()">Run</button> <span id="tmsg"></span></div>
<div><a href="/snapshot" target="_blank" style="color:#8cf">snapshot</a> &nbsp;
<span id="mode"></span></div><div>State: <span id="st">?</span> &nbsp; closure <span
id="cl"></span> &nbsp; RAM <span id="ram"></span></div><div class="box" id="plan"></div>
<div class="box fail" id="fl"></div><div class="box" id="ver"></div>
<div class="box" id="rec"></div><div class="box" id="ev"></div><script>
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function pr(p){return p?' ['+Object.entries(p).map(([k,v])=>k+' '+v.toFixed(2)).join(' ')+']':''}
function cls(l){return l.includes('FAILURE:')||l.includes('episode FAILED')?'fail':
(l.includes('episode SUCCESS')||l.includes('episode RECOVERED')?'ok':'')}
function line(l){const c=cls(l);return c?'<span class="'+c+'">'+esc(l)+'</span>':esc(l)}
async function runTask(){if(go.disabled)return;go.disabled=true;
try{const r=await fetch('/task',{method:'POST',
headers:{'Content-Type':'application/json'},body:JSON.stringify({instruction:task.value})});
const j=await r.json();tmsg.textContent=j.busy?'busy: episode running, ignored':
(j.error?'error: '+j.error:'accepted')}catch(e){tmsg.textContent='error: '+e}}
task.addEventListener('keydown',e=>{if(e.key==='Enter')runTask()});
async function tick(){try{const s=await (await fetch('/state.json')).json();
go.disabled=s.busy;if(s.busy)tmsg.textContent='busy: '+(s.task||'default episode');
else if(tmsg.textContent.startsWith('busy'))tmsg.textContent='ready';
st.textContent=s.state+(s.outcome?' ('+s.outcome+')':'');st.className=s.state;
mode.textContent=s.mode;cl.textContent=s.closure_pct==null?'-':s.closure_pct.toFixed(0)+'%';ram.textContent=s.ram;
const p=s.plan||{};plan.innerHTML=(s.result?'<div class="big '+esc(s.result_kind)+'">'+
esc(s.result)+'</div>':'')+'<b>PLAN</b> planner: '+esc(p.backend||'-')+' &nbsp; '+
(p.instruction?'instruction: '+esc(p.instruction)+'\\n':'')+
esc(p.note||'')+'\\nraw: '+esc(p.raw||'-')+'\\nparsed: '+esc(p.parsed||'-');
fl.innerHTML='<b>FAILURES</b>\\n'+(s.failures.map(esc).join('\\n')||'-');
ver.innerHTML='<b>CRITIC</b>\\n'+s.verdicts.slice(-14).map(v=>esc(v.state)+(v.monitor?
' (monitor)':'')+': <span class="'+esc(v.verdict)+'">'+esc(v.verdict)+'</span>'+esc(pr(v.probs))+
' ('+v.latency_ms.toFixed(0)+' ms) '+esc(v.frame||('raw='+v.raw))).join('\\n');
rec.innerHTML='<b>RECOVERY</b>\\n'+esc(s.recovery.join('\\n')||'-');
ev.innerHTML='<b>LOG</b>\\n'+s.events.slice(-16).map(line).join('\\n')}catch(e){}}
setInterval(tick,500);tick();</script></body></html>"""


class DemoWeb:
    """stdlib HTTP server: page, /state.json, /overhead.mjpg, /wrist.mjpg (daemon threads)."""

    def __init__(
        self,
        host: str,
        port: int,
        status: DemoStatus,
        frames: dict[str, Callable[[], tuple[int, np.ndarray] | None]],
        actions: dict[str, Callable[[dict[str, str]], dict[str, Any]]] | None = None,
        posts: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] | None = None,
    ) -> None:
        web = self
        acts: dict[str, Callable[[dict[str, str]], dict[str, Any]]] = actions or {}
        post_acts: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = posts or {}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                log.debug("http: " + fmt, *args)

            def do_GET(self) -> None:  # noqa: N802 (stdlib name)
                path = self.path.split("?")[0]
                if path == "/":
                    self._send(PAGE.encode(), "text/html; charset=utf-8")
                elif path == "/state.json":
                    self._send(json.dumps(status.snapshot()).encode(), "application/json")
                elif path.endswith(".mjpg") and path[1:-5] in frames:
                    web.stream(self, frames[path[1:-5]])
                elif path[1:] in acts:
                    query = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
                    try:
                        out = acts[path[1:]](query)
                    except Exception as e:
                        out = {"error": str(e)}
                    self._send(json.dumps(out).encode(), "application/json")
                else:
                    self.send_error(404)

            def do_POST(self) -> None:  # noqa: N802 (stdlib name)
                """JSON body (or form-encoded) -> posts[path] -> JSON reply. Body <= 4 KB."""
                path = self.path.split("?")[0]
                if path[1:] not in post_acts:
                    self.send_error(404)
                    return
                n = int(self.headers.get("Content-Length") or 0)
                if n > 4096:
                    self.send_error(413)
                    return
                raw = self.rfile.read(n).decode("utf-8", "replace")
                try:
                    if "json" in (self.headers.get("Content-Type") or ""):
                        body = json.loads(raw or "{}")
                    else:
                        body = {k: v[0] for k, v in parse_qs(raw).items()}
                    if not isinstance(body, dict):
                        raise ValueError("body must be a JSON object")
                    out = post_acts[path[1:]](body)
                except Exception as e:
                    out = {"error": str(e)}
                self._send(json.dumps(out).encode(), "application/json")

            def _send(self, body: bytes, ctype: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self._running = True
        self._thread = threading.Thread(target=self.server.serve_forever, name="web", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stream(
        self, h: BaseHTTPRequestHandler, get: Callable[[], tuple[int, np.ndarray] | None]
    ) -> None:
        try:
            import cv2  # lazy
        except ImportError:
            h.send_error(503, "cv2 not available")
            return
        h.send_response(200)
        h.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        h.send_header("Cache-Control", "no-store")
        h.end_headers()
        last = -1
        try:
            while self._running:
                item = get()
                if item is not None and item[0] != last:
                    last, img = item
                    if img.shape[1] > 640:  # 1080p wrist -> 640x360 before the JPEG encode
                        size = (640, round(640 * img.shape[0] / img.shape[1]))
                        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok:
                        data = jpg.tobytes()
                        h.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                            + str(len(data)).encode()
                            + b"\r\n\r\n"
                            + data
                            + b"\r\n"
                        )
                time.sleep(0.066)  # ~15 FPS per viewer
        except (BrokenPipeError, ConnectionResetError):
            pass

    def stop(self) -> None:
        self._running = False
        self.server.shutdown()
        self.server.server_close()


# ------------------------------------------------------------------ the demo
@dataclass
class Rig:
    cfg: AppConfig
    dcfg: DemoConfig
    arm: ArmInterface
    joints: JointsCalib
    poses: DemoPoses
    overhead: LatestFrameBuffer[RGBDFrame] | None
    wrist: LatestFrameBuffer[np.ndarray]
    detect: Callable[[RGBDFrame | None], list[Detection]]
    critic: CriticWorker | None  # None = Critic OFF baseline (verify passes through)
    qwen: QwenPlanner | None
    logger: EpisodeLogger
    vram: VramMonitor
    status: DemoStatus
    stoppers: list[Callable[[], None]] = field(default_factory=list)
    wrist_fresh: Callable[[], np.ndarray | None] | None = None  # on-demand decode (real)
    preprocess: Callable[[np.ndarray], np.ndarray] | None = None  # Critic dewarp (if any)
    no_qwen_why: str = "--planner rules"  # shown when the rules planner is used

    def wrist_frame(self) -> np.ndarray | None:
        """Newest full wrist frame: decoded on demand on the real Trinet."""
        if self.wrist_fresh is not None:
            return self.wrist_fresh()
        item = self.wrist.get()
        return item.value if item is not None else None


class Demo:
    def __init__(self, rig: Rig) -> None:
        self.r = rig
        self.s = rig.status
        st = rig.arm.read_state()
        self.q = np.asarray(rig.joints.host_to_fw_deg(st.joints), dtype=float)  # zero-relative
        self.motion = MotionController(rig.arm, rig.cfg.motion, st.joints)
        self.motion.start()
        rig.stoppers.insert(0, self.motion.stop)
        self.episode_id = ""
        self.planner_ms: float | None = None
        self.frame_n = 0
        self.end_fields: dict[str, Any] = {}  # instruction / parsed action for episode_end
        self.move_desc = ""
        self.regrasps = 0
        self.failure: tuple[str, str] | None = None  # (class, state) of the episode's failure
        self.place_note = ""  # e.g. "place unverified (LOW_CONF)"
        self.last_vote = ""  # the latest verify answer for FAILURE lines ("EMPTY 0.91")

    # ---- logging
    def rec(self, event: str, **kw: Any) -> None:
        fields: dict[str, Any] = {
            "episode_id": self.episode_id,
            "event": event,
            "fsm_state": self.s.state,
            "joints": self.q.tolist(),
            "planner_latency_ms": self.planner_ms,
            "vram_mb": self.r.vram.latest_mb,
            "closure_pct": self.s.closure_pct,
        }
        self.r.logger.log(make_record(**{**fields, **kw}))

    def frame(self, image: np.ndarray, tag: str, state: str | None = None) -> str:
        self.frame_n += 1
        return self.r.logger.save_frame(
            f"{self.episode_id}/{self.frame_n:03d}_{state or self.s.state}_{tag}.jpg", image
        )

    def state(self, name: str, note: str = "") -> None:
        self.s.set(state=name)
        self.s.event(f"{name} {note}".strip())
        self.rec("transition", reason=note)

    # ---- motion: one segment, wait for DONE (the motion thread dispatches + polls)
    def move(self, q_fw: np.ndarray, what: str, monitor: str | None = None) -> None:
        """One segment; returns at DONE. ``monitor`` (LIFT / TRANSPORT): the Critic worker
        judges the latest wrist frame back-to-back meanwhile and this loop polls its
        verdicts (never waits on them); a DROP halts the arm and raises DemoAbort."""
        seg = self.motion.command(self.r.joints.fw_to_host_deg(q_fw.tolist()))
        self.q = q_fw.copy()
        critic = self.r.critic if monitor is not None else None
        if critic is None or monitor is None:
            self._wait_segment(seg, what)
            return
        streak: list[CriticResult] = []
        critic.start_monitor(monitor, self.r.wrist_frame)
        self.rec("monitor_start", monitor_state=monitor, target={"label": what})
        try:
            self._wait_segment(seg, what, lambda: self._watch(critic, seg, monitor, streak))
        finally:
            critic.stop_monitor()
            self.rec("monitor_stop", monitor_state=monitor)

    def _wait_segment(
        self, seg: int, what: str, tick: Callable[[], None] | None = None, limit_s: float = 120.0
    ) -> None:
        t_end = time.perf_counter() + limit_s
        while time.perf_counter() < t_end:
            if tick is not None:
                tick()
            ms = self.motion.current()
            if ms.seg_id == seg and ms.phase == FAULT:
                raise DemoAbort(f"LINK: {what}: {ms.fault}", "LINK")
            if ms.seg_id == seg and ms.done:
                self.rec("segment_done", target={"label": what})
                return
            time.sleep(0.02)
        raise DemoAbort(f"{what}: no DONE within {limit_s:.0f} s", "LINK")

    def _watch(
        self, critic: CriticWorker, seg: int, state: str, streak: list[CriticResult]
    ) -> None:
        """Drain the verdict queue (non-blocking); log every verdict; DROP check."""
        d = self.r.dcfg
        while (res := critic.poll()) is not None:
            if not (res.monitor and res.state == state):  # stale (earlier request/monitor)
                self.log_verdict(res, res.state, late=True)
                continue
            self.log_verdict(res, state)
            p = (res.probs or {}).get(res.verdict.value)
            if res.verdict in NOT_HELD and (p is None or p >= d.drop_min_prob):
                streak.append(res)
            else:
                streak.clear()
            if len(streak) >= d.drop_consecutive:
                critic.stop_monitor()
                self._halt_for_drop(seg)
                raise DemoAbort(
                    f"DROP during {state} ({fmt_votes(streak)}) -> abort",
                    "DROP",
                    state,
                    open_gripper=True,
                )

    def _halt_for_drop(self, seg: int) -> None:
        """Stop the running segment. firmware/xiao_c6_bridge/Code.ino: ``m`` -> moveSynced()
        -> setTarget() REPLACES every joint's target while it moves (one DONE, for the newest
        command), i.e. a new ``m`` pre-empts. So: command the current (commanded) position
        from the latest ``q`` (<= 1 poll period old; the joints brake at the planner's
        acceleration limit and settle there). A segment that already finished is left
        alone (a segment not yet dispatched is superseded: latest-value buffer)."""
        ms = self.motion.current()
        if ms.seg_id == seg and ms.phase != MOVING:
            self.s.event("DROP: segment already finished, no halt needed", logging.WARNING)
            self.rec("drop_halt", halt="none: segment already finished")
            return
        st = self.r.arm.read_state()
        cur = list(st.commanded or st.joints)
        age_ms = (time.perf_counter() - st.t) * 1000
        self.s.event(f"DROP: halting at the current position (q {age_ms:.0f} ms old)",
                     logging.WARNING)  # fmt: skip
        halt = self.motion.command(cur)
        self.q = np.asarray(self.r.joints.host_to_fw_deg(cur), dtype=float)
        self.rec("drop_halt", halt="m to the current position from q", q_age_ms=age_ms)
        self._wait_segment(halt, "halt (drop)", limit_s=30.0)

    def pose(self, name: str, closure: float, monitor: str | None = None) -> None:
        self.move(self.r.poses.pose(name, closure), f"{name} @ {closure:.0f}%", monitor)

    def grip(self, closure: float, what: str) -> None:
        q = self.q.copy()
        q[GRIPPER_INDEX] = self.r.poses.gripper_deg(closure)
        self.s.set(closure_pct=closure)
        self.move(q, f"{what} gripper {closure:.0f}% ({q[GRIPPER_INDEX]:.1f} deg)")

    # ---- Critic logging: every verdict -> jsonl + web (state, verdict, probs, latency, crop)
    def log_verdict(self, res: CriticResult, state: str, **kw: Any) -> str | None:
        crop = (
            self.frame(res.image_bgr, f"crop_{res.verdict.value}", state)
            if res.image_bgr is not None
            else None
        )
        self.s.add(
            "verdicts",
            {"state": state, "verdict": res.verdict.value, "latency_ms": res.latency_ms,
             "raw": res.raw_text, "probs": res.probs, "frame": crop, "monitor": res.monitor},
            VERDICTS_SHOWN,
        )  # fmt: skip
        text = (
            f"Critic {state}{' [monitor]' if res.monitor else ''}"
            f"{' [late]' if kw.get('late') else ''}: {res.verdict.value}"
            f" {_probs_str(res.probs)} ({res.latency_ms:.0f} ms) crop={crop}"
        )
        if res.monitor or kw.get("late"):  # ~7 Hz while monitoring: CRITIC panel + log only
            log.info("%s", text)
        else:
            self.s.event(text)
        self.rec(
            "verdict",
            verdict=res.verdict,
            verdict_latency_ms=res.latency_ms,
            critic_state=state,
            probs=res.probs,
            raw_text=res.raw_text,
            critic_infer_ms=res.infer_ms,
            monitor=res.monitor,
            frame=crop,
            **kw,
        )
        return crop

    def failure_line(self, cls: str, state: str, text: str, final: bool) -> None:
        """Log ``FAILURE: <text>`` (red on the page, FAILURES panel) + a jsonl ``failure``."""
        if final:
            self.failure = (cls, state)
        self.s.add("failures", f"FAILURE: {text}")
        self.s.event(f"FAILURE: {text}", logging.ERROR if final else logging.WARNING)
        self.rec("failure", failure_class=cls, failure_state=state, report=text, final=final)

    # ---- Critic: submit + poll (never a blocking wait on the model)
    def verify(self, state: str, passthrough: Verdict = Verdict.GRASPED) -> Verdict:
        """Judge one fresh wrist frame. OCCLUDED / INVALID / TIMEOUT / LOW_CONF: re-query
        once. Critic OFF: returns ``passthrough``. Sets ``last_vote`` for FAILURE lines."""
        self.state(state)
        time.sleep(self.r.dcfg.settle_s)  # frame after the gripper settled
        verdict = Verdict.INVALID
        for attempt in range(2):
            img = self.r.wrist_frame()  # real Trinet: waits <= fresh_frame_timeout_s
            if img is None and self.r.critic is not None:  # wrist dropped / re-enumerating
                verdict = Verdict.OCCLUDED
                why = "no fresh wrist frame (camera reconnecting?)"
                self.last_vote = "no wrist frame"
                self.s.add(
                    "verdicts", {"state": state, "verdict": verdict.value, "latency_ms": 0.0,
                                 "raw": why, "probs": None, "frame": None, "monitor": False},
                    VERDICTS_SHOWN,
                )  # fmt: skip
                self.s.event(f"Critic {state}: {why} -> OCCLUDED", logging.WARNING)
                self.rec("verdict", verdict=verdict, raw_text=None, no_frame=True, attempt=attempt)
                continue
            raw_path = self.frame(img, "raw") if img is not None else None
            if self.r.overhead is not None and (o := self.r.overhead.get()) is not None:
                self.frame(o.value.color, "overhead")
            if self.r.critic is None:  # Critic OFF baseline (D-7): pass through, verdict null
                self.s.add(
                    "verdicts",
                    {"state": state, "verdict": "OFF", "latency_ms": 0.0, "raw": "pass-through",
                     "probs": None, "frame": None, "monitor": False},
                    VERDICTS_SHOWN,
                )  # fmt: skip
                self.s.event(f"Critic OFF: {state} passes through")
                self.rec("verdict", verdict=None, critic_skipped=True, frame_raw=raw_path)
                return passthrough
            assert img is not None  # None was handled above (Critic on)
            rid = self.r.critic.submit(img, state)
            if rid is None:
                raise DemoAbort("Critic queue full", "CRITIC", state)
            deadline = time.perf_counter() + self.r.cfg.critic.timeout_s + 1.0
            res = None
            while res is None and time.perf_counter() < deadline:
                r = self.r.critic.poll()
                if r is None:
                    time.sleep(0.02)
                elif r.request_id == rid:
                    res = r
                else:  # a late monitor result / an earlier timed-out request
                    self.log_verdict(r, r.state, late=True)
            if res is None:
                verdict = Verdict.TIMEOUT
                tmo = self.r.cfg.critic.timeout_s + 1.0
                self.last_vote = f"no answer within {tmo:.0f} s"
                self.s.event(f"Critic {state}: TIMEOUT (no answer within {tmo:.0f} s)",
                             logging.WARNING)  # fmt: skip
                self.rec("verdict", verdict=verdict, critic_state=state, frame_raw=raw_path,
                         attempt=attempt)  # fmt: skip
            else:
                verdict = res.verdict
                self.last_vote = fmt_vote(verdict.value, res.probs)
                self.log_verdict(res, state, frame_raw=raw_path, attempt=attempt)
            if verdict in GOOD:
                return verdict
        return verdict

    def block_location(self, frame: RGBDFrame | None) -> str:
        """The detector's block location: the location whose polygon_px contains the block
        centre, else "unknown" (no frame, not detected, or outside every polygon)."""
        if frame is None:
            return UNKNOWN
        try:
            det = next((x for x in self.r.detect(frame) if x.id == BLOCK_ID), None)
        except Exception as e:
            self.s.event(f"block detector failed ({e})", logging.WARNING)
            return UNKNOWN
        if det is None:
            return UNKNOWN
        u, v = det.centroid
        for loc in self.r.poses.locations.values():
            if loc.polygon_px and point_in_polygon(u, v, loc.polygon_px):
                return loc.name
        return UNKNOWN

    # ---- language prompting: instruction -> move(from, to) | none (arm idle, BEFORE motion)
    def plan_task(self, instruction: str) -> TaskPlanResult:
        r, d = self.r, self.r.dcfg
        self.state("PLAN", f"instruction: {instruction}")
        item = r.overhead.get() if r.overhead is not None else None
        frame = item.value if item is not None else None
        image = frame.color if frame is not None else None
        if image is not None:
            self.frame(image, "task_overhead")
        block_at = self.block_location(frame)
        locs = r.poses.locations
        unknown = unknown_locations(instruction, locs)
        qwen = r.qwen
        if unknown:  # never ask Qwen about a place the arm has no taught poses for
            err = (
                f"unknown location {', '.join(repr(u) for u in unknown)}: valid names are "
                f"{', '.join(locs)} (poses.yaml locations)"
            )
            res = TaskPlanResult("check", "", "", None, err, 0.0)
        elif qwen is not None:
            labelled = label_locations(image, locs) if image is not None else None
            if labelled is not None and labelled is not image:
                self.frame(labelled, "task_overhead_labelled")
            prompt = build_task_prompt(
                instruction,
                {n: x.description for n, x in locs.items()},
                block_at,
                image is not None,
                labelled is not None and labelled is not image,
            )
            self.s.set(plan={"backend": "qwen", "instruction": instruction,
                             "note": "Qwen2.5-VL-3B thinking…"})  # fmt: skip
            q, n, t = qwen, d.task_max_new_tokens, max(1.0, d.qwen_timeout_s - 1.0)
            res = plan_task(
                lambda: q.generate_task(prompt, labelled, n, t),
                locs,
                d.qwen_timeout_s,
                "qwen",
                prompt,
            )
        else:  # --planner rules / rehearsal / dry-run / Qwen load failed: no model
            rules = RulesTaskPlanner()
            res = plan_task(
                lambda: rules.generate(instruction, locs, block_at),
                locs,
                d.qwen_timeout_s,
                f"rules ({r.no_qwen_why})",
                "(rules planner, no model)",
            )
        self.planner_ms = res.latency_ms
        parsed = res.action.to_dict() if res.action is not None else None
        why = res.error or (res.action.reason if res.action is not None else "")
        self.s.set(
            plan={
                "backend": res.backend,
                "instruction": instruction,
                "note": f"{why} ({res.latency_ms:.0f} ms; block at: {block_at})",
                "raw": res.raw_text or "-",
                "parsed": json.dumps(parsed) if parsed is not None else f"REJECTED: {res.error}",
            }
        )
        level = logging.ERROR if res.error else logging.INFO
        self.s.event(f"task plan [{res.backend}]: {json.dumps(parsed) if parsed else why}", level)
        self.rec(
            "task_plan",
            instruction=instruction,
            planner=res.backend,
            prompt=res.prompt,
            raw_text=res.raw_text,
            parsed_action=parsed,
            plan_error=res.error,
            block_at=block_at,
        )
        return res

    # ---- plan (once per episode, arm idle at home)
    def plan(self) -> PlanResult:
        r, d = self.r, self.r.dcfg
        self.state("PLAN")
        item = r.overhead.get() if r.overhead is not None else None
        frame = item.value if item is not None else None
        dets = r.detect(frame)
        note = ""
        if not any(x.id == BLOCK_ID for x in dets):
            note = "block NOT detected: assumed in zone A (taught poses)"
            dets.append(Detection(BLOCK_ID, r.cfg.detector.label, 0.0, (0.0, 0.0, 0.0, 0.0)))
        if not any(x.id == PLACE_ID for x in dets):
            poly = r.poses.place_polygon_px
            bbox = (
                (
                    min(p[0] for p in poly),
                    min(p[1] for p in poly),
                    max(p[0] for p in poly),
                    max(p[1] for p in poly),
                )
                if poly
                else (0.0, 0.0, 0.0, 0.0)
            )
            dets.append(Detection(PLACE_ID, "place_zone", 1.0, bbox))
        if frame is not None:
            self.frame(frame.color, "plan_overhead")
        image = frame.color if frame is not None else None
        req = PlanRequest(d.instruction, dets, image)
        qwen_raw = ""
        why = f"rules planner ({r.no_qwen_why})"
        if r.qwen is not None:
            self.s.set(plan={"backend": "qwen", "note": "Qwen2.5-VL-3B thinking..."})
            res = plan_demo_intro(r.qwen, req, d.qwen_timeout_s)
            self.rec("reasoning_trace", plan_error=res.error, trace=_trace(res), planner="qwen")
            if res.chain is not None:
                return self.show_plan(res, "qwen", "Qwen2.5-VL-3B OK", note)
            why = f"Qwen FAILED ({res.error}, {res.latency_ms:.0f} ms) -> rules planner"
            qwen_raw = f"[qwen] {res.raw_text!r}\n[rules] "
            self.s.event(why, logging.WARNING)
        rules = RuleBasedPlanner(100.0, r.cfg.skills.default_standoff_mm)
        res = plan_with_timeout(rules, req, r.cfg.planner)
        self.rec("reasoning_trace", plan_error=res.error, trace=_trace(res), planner="rules")
        return self.show_plan(res, "rules fallback" if qwen_raw else "rules", why, note, qwen_raw)

    def show_plan(
        self, res: PlanResult, backend: str, why: str, note: str, raw_prefix: str = ""
    ) -> PlanResult:
        parsed = json.dumps(_trace(res).get("chain")) if res.chain is not None else res.error
        self.planner_ms = res.latency_ms
        self.s.set(
            plan={
                "backend": backend,
                "note": f"{why} ({res.latency_ms:.0f} ms) {note}".strip(),
                "raw": f"{raw_prefix}{res.raw_text}",
                "parsed": parsed,
            }
        )
        self.s.event(f"plan [{backend}] {why}: {parsed}")
        if note:
            self.s.event(note, logging.WARNING)
        return res

    # ---- the episode
    def run(self, induce_slip: bool, instruction: str | None = None) -> bool:
        """One episode. ``instruction`` (language prompting): plan FIRST with the arm idle;
        invalid output or "none" -> nothing moves. None: the default A -> place episode."""
        r, d = self.r, self.r.dcfg
        self.episode_id = time.strftime("%Y%m%d-%H%M%S")
        self.regrasps, self.failure, self.place_note, self.last_vote = 0, None, "", ""
        self.s.set(
            verdicts=[],
            recovery=[],
            failures=[],
            outcome=None,
            closure_pct=0.0,
            episode=self.s.episode + 1,
            result=f"RUNNING: {instruction or 'default episode A -> place'}",
            result_kind="running",
        )
        self.end_fields = {"instruction": instruction or d.instruction}
        self.move_desc = f"{DEFAULT_SRC.name} -> {DEFAULT_DST.name}"
        self.rec("episode_start", induce_slip=induce_slip, instruction=instruction or d.instruction)
        src, dst = DEFAULT_SRC, DEFAULT_DST
        if instruction is not None:
            task = self.plan_task(instruction)
            act = task.action
            self.end_fields.update(
                parsed_action=act.to_dict() if act is not None else None, raw_text=task.raw_text
            )
            if act is None or act.action != "move" or act.src is None or act.dst is None:
                outcome = "rejected" if act is None else "none"
                self.finish(outcome, task.error or (act.reason if act is not None else ""))
                return False
            src, dst = r.poses.locations[act.src], r.poses.locations[act.dst]
            self.move_desc = f"{src.name} -> {dst.name}"
            self.s.event(f"MOVE {src.name} -> {dst.name}: {act.reason}")
        closure = 0.0
        try:
            self.state("HOME")
            self.pose("home", 0.0)
            if instruction is None:
                res = self.plan()
                if res.chain is None:
                    raise DemoAbort(f"no valid plan ({res.error})")
            self.state("APPROACH", f"{src.name} -> {dst.name}")
            self.pose(src.pre, 0.0)
            self.pose(src.grasp, 0.0)
            base = 100.0  # closed_deg = "closed on the block"
            closure = base
            if induce_slip:
                g = d.induce_slip_gripper_deg
                closure = r.poses.pct_for_deg(g)
                self.rec("induced_slip", closure_pct=closure, gripper_deg=g)
                self.s.event(
                    f"INDUCED SLIP: first grasp at {g:.0f} deg ({closure:.0f}%) only",
                    logging.WARNING,
                )
            mx = d.max_regrasps
            while True:
                self.state("GRASP", f"closure {closure:.0f}%")
                self.grip(closure, "close")
                at = "VERIFY_GRASP"
                v = self.verify(at)
                if v is Verdict.GRASPED:
                    self.state("LIFT")
                    self.pose(src.pre, closure, monitor="LIFT")
                    at = "VERIFY_HOLD"
                    v = self.verify(at)
                    if v is Verdict.GRASPED:
                        break
                what = f"{v.value} at {at} ({self.last_vote})"
                if v not in NOT_HELD:
                    raise DemoAbort(f"{what} after a re-query -> abort", v.value, at)
                if self.regrasps >= mx:
                    raise DemoAbort(
                        f"{what} -> abort (re-grasp budget {mx}/{mx} used)", v.value, at
                    )
                self.regrasps += 1
                self.failure_line(v.value, at, f"{what} -> re-grasp {self.regrasps}/{mx}", False)
                closure = base + d.regrasp_step_pct * self.regrasps
                step = f"k={self.regrasps}: {v.value} -> re-open, re-grasp at {closure:.0f}%"
                self.s.add("recovery", step)
                self.state("RECOVER", step)
                self.rec("recover", action=step, attempt=self.regrasps)
                at_grasp = r.poses.poses[src.grasp][:GRIPPER_INDEX]
                if not np.allclose(self.q[:GRIPPER_INDEX], at_grasp):
                    self.pose(src.grasp, self.s.closure_pct or 0.0)  # back down, holding
                self.grip(0.0, "re-open")
            self.state("TRANSPORT")
            self.pose(dst.pre, closure, monitor="TRANSPORT")
            self.state("PLACE")
            self.pose(dst.grasp, closure)
            self.grip(0.0, "release")
            v = self.verify("VERIFY_PLACE", passthrough=Verdict.EMPTY)
            if v in (Verdict.GRASPED, Verdict.SLIPPING):
                self.failure_line(
                    v.value, "VERIFY_PLACE",
                    f"object stuck in gripper ({self.last_vote} at VERIFY_PLACE)", True,
                )  # fmt: skip
            elif v is not Verdict.EMPTY:
                self.place_note = f"place unverified ({v.value}: {self.last_vote})"
                self.s.event(f"VERIFY_PLACE inconclusive: {self.place_note}", logging.WARNING)
            self.state("HOME")
            self.pose("home", 0.0)
            if self.failure is not None:
                self.finish("failed", "object stuck in gripper after release")
                return False
            self.finish("success")
            return True
        except DemoAbort as e:
            self.abort(e)
            return False

    def abort(self, e: DemoAbort) -> None:
        why = str(e)
        link = e.failure_class == "LINK" or "LINK" in why
        cls = e.failure_class or ("LINK" if link else "ABORT")
        at = e.failure_state or self.s.state
        text = why if e.failure_class not in (None, "LINK") else f"{cls} at {at}: {why} -> abort"
        self.failure_line(cls, at, text, True)
        self.state("ABORT", why)
        try:
            if link:
                self.r.arm.stop()  # unreliable link: brake and hold, no more motion
                self.s.event("LINK fault: stop sent, servos hold. Keep the E-stop in reach.")
            elif e.open_gripper:  # DROP: nothing to hold; open where we stopped, then home
                q = self.q.copy()
                q[GRIPPER_INDEX] = self.r.poses.gripper_deg(0.0)
                self.s.set(closure_pct=0.0)
                self.move(q, "open gripper (drop abort)")
                q = np.asarray(self.r.poses.poses["home"], dtype=float).copy()
                q[GRIPPER_INDEX] = self.r.poses.gripper_deg(0.0)
                self.move(q, "home (abort, gripper open)")
            else:
                g = self.q[GRIPPER_INDEX]  # keep the gripper: a held block is not dropped
                q = np.asarray(self.r.poses.poses["home"], dtype=float).copy()
                q[GRIPPER_INDEX] = g
                self.move(q, "home (abort)")
        except Exception as ex:
            self.s.event(f"retreat failed: {ex}; stop", logging.ERROR)
            try:
                self.r.arm.stop()
            except Exception:
                log.exception("stop failed")
        self.finish("abort", why)

    def finish(self, outcome: str, why: str = "") -> None:
        """outcome: success | abort | failed (arm home, e.g. object stuck) | none (planner
        declined) | rejected (invalid plan). End line: SUCCESS / RECOVERED (n re-grasps) /
        FAILED (<class> at <state>)."""
        state = {"success": "DONE", "abort": "ABORT", "failed": "FAILED"}.get(outcome, "NO ACTION")
        kind = outcome
        if outcome == "success":
            n = self.regrasps
            label = "RECOVERED" if n else "SUCCESS"
            head = f"RECOVERED ({n} re-grasp{'s' if n != 1 else ''})" if n else "SUCCESS"
            text = f"{head}: {self.move_desc}" + (f"; {self.place_note}" if self.place_note else "")
            kind = "recovered" if n else "success"
        elif outcome in ("abort", "failed"):
            cls, at = self.failure or ("ABORT", self.s.state)
            label = "FAILED"
            text = f"FAILED ({cls} at {at}): {why.removesuffix(' -> abort')}"
            kind = "abort"
        else:
            label = {"none": "NONE"}.get(outcome, "ERROR")
            text = f"{label}: {why}"
        self.s.set(outcome=outcome, state=state, result=text, result_kind=kind)
        self.s.event(f"episode {text}", logging.ERROR if label == "FAILED" else logging.INFO)
        self.rec(
            "episode_end",
            outcome=outcome,
            outcome_label=label,
            result=text,
            report=why or None,
            regrasps=self.regrasps,
            failure_class=self.failure[0] if self.failure else None,
            failure_state=self.failure[1] if self.failure else None,
            **self.end_fields,
        )


def _trace(res: PlanResult) -> dict[str, Any]:
    return res.trace.to_dict() if res.trace is not None else {}


# ------------------------------------------------------------------ setup
def ram(status: DemoStatus, when: str) -> float | None:
    r = read_system_ram()
    if r is None:
        status.event(f"RAM {when}: n/a")
        return None
    status.set(ram=f"{r.used_mb:.0f}/{r.total_mb:.0f} MB ({r.free_mb:.0f} free)")
    status.event(f"RAM {when}: {r.used_mb:.0f}/{r.total_mb:.0f} MB used, {r.free_mb:.0f} MB free")
    return r.free_mb


def build_dry_rig(args: argparse.Namespace, dcfg: DemoConfig, status: DemoStatus) -> Rig:
    from lbyr.arm.mock_arm import MockArm

    cfg = mock_profile()
    log.warning("DRY RUN: MockArm, fake cameras, FAKE poses. No hardware is driven.")
    scene = FakeScene(
        objects=[
            FakeObject(BLOCK_ID, cfg.detector.label, (220.0, 60.0, 0.0), 30.0, 30.0),
            FakeObject(2, "place_zone", (220.0, -100.0, 0.0), 60.0, 2.0),
        ]
    )
    overhead = FakeOverheadCamera(scene)
    wrist = FakeWristCamera(640, 480)
    overhead.start()
    wrist.start()
    lo = list(cfg.arm.joint_min_deg or ())
    lo[GRIPPER_INDEX] = -90.0  # FAKE: room for the -40..0 gripper map
    cfg.arm = replace(cfg.arm, joint_min_deg=tuple(lo))
    arm = MockArm(cfg.arm, initial_deg=mock_demo_poses().poses["home"])
    arm.connect()
    backend = None if args.critic == "off" else scripted_critic(args, dcfg)
    return _finish_rig(
        args, cfg, dcfg, status, arm, mock_joints(), mock_demo_poses(),
        overhead.buffer, wrist.buffer, lambda f: [
            replace(x, id=PLACE_ID) if x.id == 2 else x for x in scene.detections()
        ], backend, None,
        [overhead.stop, wrist.stop, arm.close],
    )  # fmt: skip


def build_real_rig(args: argparse.Namespace, dcfg: DemoConfig, status: DemoStatus) -> Rig:
    cfg = AppConfig()
    joints = load_joints(cfg.arm.joints_path)
    if not joints.verified:
        raise CalibrationError(f"{cfg.arm.joints_path}: verified is false (A-7, tools/arm_gui.py)")
    poses = load_demo_poses(cfg.motion.poses_path)
    stoppers: list[Callable[[], None]] = []  # cameras first (stop before the arm link)
    try:
        return _build_real_rig(args, dcfg, status, cfg, joints, poses, stoppers)
    except BaseException:  # incl. Ctrl-C at a prompt: never exit with camera threads live
        status.event("setup failed: stopping cameras / arm link", logging.WARNING)
        run_stoppers(stoppers)
        raise


def run_stoppers(stoppers: list[Callable[[], None]]) -> None:
    """Run every shutdown step in order; one failing step never skips the rest."""
    for stop in stoppers:
        try:
            stop()
        except Exception:
            log.exception("shutdown step failed")
    stoppers.clear()


def open_overhead(
    cfg: AppConfig, status: DemoStatus, stoppers: list[Callable[[], None]], first_s: float = 5.0
) -> LatestFrameBuffer[RGBDFrame] | None:
    """Start the D435 and wait for its first frame. Failure -> None (no overhead view, rules
    plan), with the pipeline stopped and its thread joined."""
    try:
        from lbyr.perception.overhead import RealSenseOverheadCamera

        cam = RealSenseOverheadCamera(cfg.cameras.overhead)
        cam.start()
        stoppers.insert(0, cam.stop)
        t_end = time.perf_counter() + first_s
        while cam.buffer.get() is None:
            if time.perf_counter() >= t_end:
                stoppers.remove(cam.stop)
                cam.stop()
                raise LBYRError(f"no D435 frame within {first_s:.0f} s")
            time.sleep(0.02)
        return cam.buffer
    except Exception as e:
        status.event(f"D435 unavailable ({e}): no overhead view, rules plan", logging.ERROR)
        return None


def _build_real_rig(
    args: argparse.Namespace,
    dcfg: DemoConfig,
    status: DemoStatus,
    cfg: AppConfig,
    joints: JointsCalib,
    poses: DemoPoses,
    stoppers: list[Callable[[], None]],
) -> Rig:
    from lbyr.arm.serial_arm import SerialArm, detect_bridge_port, speed_for_cap

    # cameras BEFORE the arm link and the E-stop prompt: a camera failure surfaces while the
    # servos are still unpowered. Both must deliver a first frame.
    overhead_buf = open_overhead(cfg, status, stoppers)
    wc = cfg.cameras.wrist
    wdev = args.wrist_dev or dcfg.wrist_device
    if wdev is not None and wdev.isdigit():
        wdev = f"/dev/video{wdev}"
    dev, cap, frame = open_wrist_wait(lambda: [wdev] if wdev else _wrist_candidates(), wc)
    wrist = WristCapture(dev, wc, (cap, frame))
    stoppers.insert(0, wrist.stop)  # before start(): releases the open capture either way
    wrist.start()
    d435 = "first frame OK" if overhead_buf is not None else "UNAVAILABLE"
    status.event(f"cameras: D435 {d435}; wrist {dev} first frame {frame.shape[1]}x{frame.shape[0]}")

    port, found = detect_bridge_port()
    port = args.port or port
    if port is None:
        raise LBYRError(f"bridge port: found {found or 'no /dev/ttyACM*'}; use --port")
    cfg.arm.serial_port = port
    arm = SerialArm(cfg.arm, joints)
    arm.connect()  # closes itself on failure
    stoppers.append(arm.close)
    v, a = arm.set_speed(*speed_for_cap(cfg.motion.dev_speed_cap))
    status.event(f"arm on {port}: speed {v:.0f} deg/s, {a:.0f} deg/s^2 (dev cap)")
    if input("E-STOP within reach and tested? Type y to continue: ").strip().lower() != "y":
        raise LBYRError("E-stop not confirmed")

    detect: Callable[[RGBDFrame | None], list[Detection]] = lambda f: []  # noqa: E731
    try:
        from lbyr.perception.block import BlockDetector

        det = BlockDetector(replace(cfg.detector, object_id=BLOCK_ID))
        detect = det.detect
    except Exception as e:
        status.event(f"block detector unavailable ({e})", logging.WARNING)

    # models: Qwen once (RESIDENT for the whole session), then Moondream
    ram(status, "before model load")
    qwen: QwenPlanner | None = None
    no_qwen_why = "--planner rules"
    if args.planner == "qwen":
        pcfg = replace(
            cfg.planner,
            backend="qwen",
            model_path=dcfg.qwen_path.expanduser(),
            vl_dtype=dcfg.qwen_dtype,
        )
        try:
            status.event("loading Qwen2.5-VL-3B (bf16)...")
            qwen = QwenPlanner(pcfg)
        except Exception as e:
            log.exception("Qwen load failed")
            no_qwen_why = f"Qwen load failed: {e}"
            status.event(f"Qwen load failed ({e}): rules planner", logging.ERROR)
        ram(status, "after Qwen load")
    backend: CriticBackend | None = None
    if args.critic == "probe":
        status.event(f"loading Moondream2 (bf16) + Critic probe {args.probe_path}...")
        from lbyr.ai.critic_probe import ProbeCritic

        mpath = dcfg.moondream_path.expanduser()
        backend = ProbeCritic(replace(cfg.critic, model_path=mpath), args.probe_path.expanduser())
    elif args.critic == "moondream":
        status.event("loading Moondream2 (bf16)...")
        from lbyr.ai.critic import Moondream2Critic

        mpath = dcfg.moondream_path.expanduser()
        backend = Moondream2Critic(replace(cfg.critic, model_path=mpath))
    elif args.critic == "scripted":
        backend = scripted_critic(args, dcfg)
    free = ram(status, f"after Critic ({args.critic}) setup")
    if free is not None and free < dcfg.warn_free_ram_mb:  # warn only: Qwen stays resident
        status.event(
            f"LOW RAM: {free:.0f} MB free < {dcfg.warn_free_ram_mb:.0f} MB with the models "
            "loaded (continuing; Qwen stays resident)",
            logging.WARNING,
        )
    from lbyr.perception.wrist import build_wrist_preprocess

    item = wrist.buffer.get()
    size = (item.value.shape[1], item.value.shape[0]) if item is not None else None
    pre = build_wrist_preprocess(wc, size)
    rig = _finish_rig(
        args, cfg, dcfg, status, arm, joints, poses, overhead_buf, wrist.buffer, detect,
        backend, qwen, stoppers, pre,
    )  # fmt: skip
    rig.wrist_fresh = wrist.fresh
    rig.no_qwen_why = no_qwen_why
    return rig


def scripted_critic(args: argparse.Namespace, dcfg: DemoConfig) -> ScriptedCritic:
    """Fake Critic (no model). Default sequence (verify states): --script, else
    SLIPPING,GRASPED with --induce-slip, else DemoConfig.scripted_verdicts. Per-state
    sequences: DemoConfig.scripted_state_verdicts (LIFT / TRANSPORT GRASPED, VERIFY_PLACE
    EMPTY), overridden by --script "STATE=..." parts (see parse_script)."""
    per_state = {k: list(v) for k, v in dcfg.scripted_state_verdicts}
    default: list[str] = []
    if args.script:
        try:
            default, given = parse_script(args.script)
        except ValueError as e:
            raise LBYRError(f"--script: {e}") from e
        per_state.update(given)
    elif args.induce_slip:
        default = ["SLIPPING", "GRASPED"]
    return ScriptedCritic(
        default or list(dcfg.scripted_verdicts), delay_s=dcfg.scripted_delay_s, per_state=per_state
    )


def _finish_rig(
    args: argparse.Namespace,
    cfg: AppConfig,
    dcfg: DemoConfig,
    status: DemoStatus,
    arm: ArmInterface,
    joints: JointsCalib,
    poses: DemoPoses,
    overhead: LatestFrameBuffer[RGBDFrame] | None,
    wrist: LatestFrameBuffer[np.ndarray],
    detect: Callable[[RGBDFrame | None], list[Detection]],
    backend: CriticBackend | None,
    qwen: QwenPlanner | None,
    stoppers: list[Callable[[], None]],
    preprocess: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Rig:
    if args.crop:
        box = tuple(float(v) for v in args.crop.split(","))
        if len(box) != 4:
            raise LBYRError("--crop needs x0,y0,x1,y1 (normalized 0..1)")
        cfg.critic = replace(cfg.critic, crop_box_norm=(box[0], box[1], box[2], box[3]))
    status.event(f"Critic crop box (normalized): {cfg.critic.crop_box_norm or 'centred square'}")
    trained = getattr(backend, "trained_crop", cfg.critic.crop_box_norm)
    if trained != cfg.critic.crop_box_norm:
        status.event(
            f"Critic probe was trained on crop {trained or 'centred square'}, runtime crop is "
            f"{cfg.critic.crop_box_norm or 'centred square'}: accuracy may drop (retrain or "
            "pass the training box with --crop)",
            logging.WARNING,
        )
    critic = None
    if backend is not None:
        critic = CriticWorker(backend, cfg.critic, preprocess=preprocess)
        critic.start()
    logger = EpisodeLogger(args.log_dir / time.strftime("demo-%Y%m%d-%H%M%S"))
    logger.start()
    vram = VramMonitor(cfg.logging.tegrastats_interval_ms)
    vram.start()
    stoppers = [*([critic.stop] if critic else []), vram.stop, logger.stop, *stoppers]
    status.event(f"episode log: {logger.jsonl_path}")
    planner = "qwen2.5-vl (resident)" if qwen is not None else "rules"
    status.set(mode=f"planner={planner} critic={args.critic} {'DRY RUN' if args.dry_run else ''}")
    return Rig(
        cfg, dcfg, arm, joints, poses, overhead, wrist, detect, critic, qwen, logger, vram,
        status, stoppers, preprocess=preprocess,
    )  # fmt: skip


class Snapshots:
    """'w' key / GET /snapshot[?crop=x0,y0,x1,y1]: newest FULL wrist frame + the Critic crop
    (same preprocess + critic_crop as the CriticWorker) -> <dir>/snap_<n>_{full,crop}.jpg.
    ``crop`` previews another box without changing the Critic's. Cost: one on-demand decode
    (~15 ms) + two JPEG writes (~30 ms), off the motion path."""

    def __init__(self, root: Path, rig_box: list[Rig]) -> None:
        self.root = root
        self._rig_box = rig_box
        self._n = 0
        self._lock = threading.Lock()

    def take(self, crop: str | None = None) -> dict[str, Any]:
        import cv2  # lazy

        if not self._rig_box:
            return {"error": "rig not ready"}
        rig = self._rig_box[0]
        box = rig.cfg.critic.crop_box_norm
        if crop:
            vals = tuple(float(v) for v in crop.split(","))
            if len(vals) != 4:
                raise ValueError("crop needs x0,y0,x1,y1 (normalized 0..1)")
            box = (vals[0], vals[1], vals[2], vals[3])
        frame = rig.wrist_frame()
        if frame is None:
            return {"error": "no wrist frame"}
        img = rig.preprocess(frame) if rig.preprocess is not None else frame
        cropped = critic_crop(img, box, rig.cfg.critic.crop_size_px)
        with self._lock:
            self._n += 1
            n = self._n
        self.root.mkdir(parents=True, exist_ok=True)
        full, small = self.root / f"snap_{n}_full.jpg", self.root / f"snap_{n}_crop.jpg"
        cv2.imwrite(str(full), frame)
        cv2.imwrite(str(small), cropped)
        out = {"full": str(full), "crop": str(small), "box": box or "centred square"}
        rig.status.event(f"snapshot {n}: {small} (box {out['box']})")
        return out


QUIT = "__quit__"


class Commands:
    """Episode requests (web Run, keyboard, --task) -> the main thread, one at a time.
    While one is queued or running, new submissions are refused ("busy")."""

    def __init__(self, status: DemoStatus) -> None:
        self.status = status
        self.q: queue.Queue[tuple[str | None, bool, str] | str] = queue.Queue()
        self._lock = threading.Lock()
        self._busy = False

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def submit(
        self, instruction: str | None, induce_slip: bool = False, source: str = "web"
    ) -> bool:
        """``instruction`` None = the default A -> place episode; ``source`` web | terminal |
        cli. False = busy (ignored)."""
        with self._lock:
            if self._busy:
                return False
            self._busy = True
        self.status.set(busy=True, task=instruction or "")
        self.q.put((instruction, induce_slip, source))
        return True

    def done(self) -> None:
        with self._lock:
            self._busy = False
        self.status.set(busy=False)

    def quit(self) -> None:
        self.q.put(QUIT)


class KeyboardInput:
    """Operator keys, read in the MAIN thread between episodes with select() on stdin (no
    input() thread: one blocked at exit can abort the interpreter). Lines typed during an
    episode are discarded (busy). Enter = run (--task if given), s = default run with
    induced slip, w = wrist snapshot, q = quit, anything else = an instruction."""

    def __init__(self, cmds: Commands, snaps: Snapshots, task: str | None) -> None:
        self.cmds, self.snaps, self.task = cmds, snaps, task
        self.eof = False

    def prompt(self) -> None:
        run = f"run {self.task!r}" if self.task else "run A -> place"
        log.info(
            "[Enter] %s, [s] run with slip, [w] wrist snapshot, [q] quit, "
            "or type an instruction (or use the web page)",
            run,
        )

    def _line(self, timeout_s: float) -> str | None:
        if self.eof:
            time.sleep(timeout_s)
            return None
        ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
        if not ready:
            return None
        line = sys.stdin.readline()
        if line == "":
            self.eof = True
            log.warning("stdin closed: instructions from the web page only")
            return None
        return line.strip()

    def drain(self) -> None:
        while (line := self._line(0.0)) is not None:
            log.warning("busy: ignored keyboard input %r typed during the episode", line)

    def poll(self, timeout_s: float = 0.2) -> None:
        ans = self._line(timeout_s)
        if ans is None:
            return
        low = ans.lower()
        if low == "q":
            self.cmds.quit()
        elif low == "w":
            try:
                log.info("snapshot: %s", self.snaps.take())
            except Exception as e:
                log.error("snapshot failed: %s", e)
        elif low in ("", "s"):
            if not self.cmds.submit(self.task if low == "" else None, low == "s", "terminal"):
                log.warning("busy: episode running, ignored")
        else:
            try:
                ok = self.cmds.submit(clean_instruction(ans), False, "terminal")
            except PlanValidationError as e:
                log.error("instruction refused: %s", e)
                return
            if not ok:
                log.warning("busy: episode running, ignored")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    d = DemoConfig()
    p = argparse.ArgumentParser(
        prog="python -m lbyr.demo",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dry-run", action="store_true", help="MockArm + fake cameras + fake Critic")
    p.add_argument("--induce-slip", action="store_true", help="first grasp at reduced closure")
    p.add_argument(
        "--planner", choices=("qwen", "rules"), default="qwen", help="rules = no Qwen load"
    )
    p.add_argument("--no-planner", action="store_true", help="same as --planner rules")
    p.add_argument(
        "--critic",
        choices=("probe", "moondream", "scripted", "off"),
        default=None,
        help="probe (default if --probe-path exists, else moondream; dry-run: scripted) | "
        "moondream (zero-shot text) | scripted (fake, no model) | off (baseline)",
    )
    p.add_argument(
        "--probe-path",
        type=Path,
        default=d.probe_path,
        help=f"Critic probe file (tools/train_critic_probe.py; default {d.probe_path})",
    )
    p.add_argument("--critic-off", action="store_true", help="same as --critic off (baseline)")
    p.add_argument(
        "--rehearsal",
        action="store_true",
        help="motion-only: real arm + cameras + web, --planner rules --critic scripted, no models",
    )
    p.add_argument("--script", help="scripted Critic verdicts, e.g. SLIPPING,GRASPED,GRASPED")
    p.add_argument("--port", help="bridge serial port (default: the only /dev/ttyACM*)")
    p.add_argument("--wrist-dev", help="wrist camera: /dev/videoN or index (default: auto)")
    p.add_argument(
        "--crop",
        help="Critic gripper box x0,y0,x1,y1 normalized 0..1 (default CriticConfig.crop_box_norm)",
    )
    p.add_argument("--http-port", type=int, default=d.http_port)
    p.add_argument("--log-dir", type=Path, default=AppConfig().logging.episodes_dir)
    p.add_argument("--once", action="store_true", help="one episode, then exit")
    p.add_argument(
        "--task",
        help='language instruction, e.g. "move the block from A to the place zone" (Enter runs '
        "it; --dry-run runs it once)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    if args.task is not None:
        try:
            args.task = clean_instruction(args.task)
        except PlanValidationError as e:
            p.error(f"--task: {e}")
    if args.no_planner or args.rehearsal or args.dry_run:
        args.planner = "rules"
    args.critic_note = None  # shown on the page + log at startup (fallback warning)
    if args.critic is None:
        if args.rehearsal or args.dry_run:
            args.critic = "scripted"
        elif args.probe_path.expanduser().is_file():
            args.critic = "probe"
        else:
            args.critic = "moondream"
            args.critic_note = (
                f"WARNING: no Critic probe at {args.probe_path} -> zero-shot Moondream2 "
                "(slower, less accurate). Train one: python tools/train_critic_probe.py"
            )
    if args.critic_off:
        args.critic = "off"
    if args.dry_run and args.critic in ("moondream", "probe"):
        p.error("--dry-run never loads models: use --critic scripted or off")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    listener = setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    dcfg = DemoConfig(http_port=args.http_port)
    status = DemoStatus()
    rig: Rig | None = None
    web: DemoWeb | None = None
    try:
        rig_box: list[Rig] = []

        def frame_of(which: str) -> Callable[[], tuple[int, np.ndarray] | None]:
            def get() -> tuple[int, np.ndarray] | None:
                if not rig_box:
                    return None
                if which == "overhead":
                    buf = rig_box[0].overhead
                    o = buf.get() if buf is not None else None
                    return (o.seq, o.value.color) if o is not None else None
                w = rig_box[0].wrist.get()
                return (w.seq, w.value) if w is not None else None

            return get

        snaps = Snapshots(dcfg.snapshot_dir / time.strftime("%Y%m%d-%H%M%S"), rig_box)
        cmds = Commands(status)

        def web_task(body: dict[str, Any]) -> dict[str, Any]:
            text = clean_instruction(str(body.get("instruction", "")))  # raises -> {"error"}
            if not rig_box:
                return {"error": "rig not ready"}
            if not cmds.submit(text):
                return {"busy": True}
            return {"accepted": True, "instruction": text}

        web = DemoWeb(
            dcfg.http_host,
            dcfg.http_port,
            status,
            {"overhead": frame_of("overhead"), "wrist": frame_of("wrist")},
            {"snapshot": lambda q: snaps.take(q.get("crop"))},
            {"task": web_task},
        )
        web.start()
        log.info("web view on http://%s:%d", dcfg.http_host, web.port)
        if args.critic_note:
            status.event(args.critic_note, logging.WARNING)
        status.event(f"Critic backend: {args.critic}")
        try:
            build = build_dry_rig if args.dry_run else build_real_rig
            rig = build(args, dcfg, status)
        except Exception as e:  # hardware / calibration / model load: refuse, exit 2
            log.exception("setup failed")
            status.set(state="SETUP FAILED")
            status.event(f"setup refused: {e}", logging.ERROR)
            return 2
        demo = Demo(rig)
        rig_box.append(rig)
        once = args.once or args.dry_run
        keys: KeyboardInput | None = None
        if args.dry_run:  # no operator: run --task (or the default episode) once
            cmds.submit(args.task, args.induce_slip, "cli")
        else:  # nothing moves until Enter / an instruction (keyboard or web)
            keys = KeyboardInput(cmds, snaps, args.task)
            log.info("Ready (web view on port %d).", web.port)
            keys.prompt()
        first_slip, rc = args.induce_slip, 0
        while True:
            if keys is not None:
                keys.poll(0.2)
            try:
                item = cmds.q.get(timeout=0.05 if keys is not None else 0.5)
            except queue.Empty:
                continue
            if isinstance(item, str):  # QUIT
                return rc
            instruction, slip, source = item
            try:
                ok = demo.run(slip or first_slip, instruction)
            finally:
                first_slip = False
                cmds.done()
            rc = 0 if ok else 1
            if once:
                return rc
            if keys is not None:
                keys.drain()
                if source == "terminal":  # web runs never re-print the terminal menu
                    keys.prompt()
    except KeyboardInterrupt:
        log.warning("Ctrl-C: stopping the arm")
        if rig is not None:
            try:
                rig.arm.stop()
            except Exception:
                log.exception("stop failed")
        return 1
    finally:
        if rig is not None:
            run_stoppers(rig.stoppers)
        if web is not None:
            web.stop()
        listener.stop()


if __name__ == "__main__":
    sys.exit(main())
