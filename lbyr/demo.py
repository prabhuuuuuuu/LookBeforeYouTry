"""Live demo: taught poses + Qwen2.5-VL plan + Moondream2 Critic + web view.

    python -m lbyr.demo --rehearsal          # real arm + cameras + web, rules plan, fake Critic,
                                             # NO model loading (motion-only bring-up)
    python -m lbyr.demo                      # full: Qwen plan + Moondream Critic
    python -m lbyr.demo --planner rules      # real Critic, no Qwen (fast trials)
    python -m lbyr.demo --induce-slip        # loose first grasp (induce_slip_gripper_deg)
    python -m lbyr.demo --critic-off         # baseline: verify states pass through
    python -m lbyr.demo --dry-run            # MockArm + fake cameras + scripted Critic

Keys at the prompts: Enter = run, s = run with induced slip, w = wrist snapshot
(runs/<ts>/snap_<n>_{full,crop}.jpg, also GET /snapshot?crop=x0,y0,x1,y1), q = quit.
Web view: http://<jetson>:8000 (overhead + wrist MJPEG, FSM state, plan, verdicts).

Sequence (one ``m`` per segment, DONE awaited via the MotionController, PP-1):
home -> zone_A_pre -> zone_A_grasp -> close -> VERIFY_GRASP -> lift (zone_A_pre) ->
VERIFY_HOLD -> place_pre -> place -> open -> home. SLIPPING / EMPTY: re-open, re-grasp
at +15% closure (beyond closed_deg, never past 0 = fully shut), at most 2 times, then
abort to home holding the gripper. Poses: data/calib/poses.yaml (zero-relative degrees).

Cost: Qwen2.5-VL-3B bf16 ~6.5 GB, one plan (~11 s, 30 s cap) before the motion; Moondream2
bf16 ~4.1 GB, ~0.85 s per verify (bible 9.2). MJPEG encode ~3 ms per frame per viewer (CPU).
Exit codes: 0 = DONE, 1 = ABORT, 2 = setup refused (hardware / calibration / models).
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import logging
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from lbyr.ai.critic import CriticBackend, CriticWorker, ScriptedCritic, Verdict
from lbyr.ai.imaging import critic_crop
from lbyr.ai.planner import (
    PlanRequest,
    PlanResult,
    QwenPlanner,
    RuleBasedPlanner,
    plan_demo_intro,
    plan_with_timeout,
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
from lbyr.errors import CalibrationError, LBYRError
from lbyr.log_setup import setup_logging
from lbyr.logging_.episodes import EpisodeLogger, VramMonitor, make_record, read_system_ram
from lbyr.motion.controller import FAULT, MotionController
from lbyr.perception.buffers import LatestFrameBuffer
from lbyr.perception.detector import Detection
from lbyr.perception.overhead import FakeObject, FakeOverheadCamera, FakeScene, RGBDFrame
from lbyr.perception.wrist import FakeWristCamera, decode_wrist_jpeg

log = logging.getLogger("lbyr.demo")

REQUIRED = ("home", "zone_A_pre", "zone_A_grasp", "place_pre", "place")
BLOCK_ID, PLACE_ID = 1, 900
GOOD = (Verdict.GRASPED, Verdict.SLIPPING, Verdict.EMPTY)  # decisive Critic answers


class DemoAbort(LBYRError):
    """Stop the sequence and retreat to home."""


# ------------------------------------------------------------------ poses
@dataclass(frozen=True)
class DemoPoses:
    """Taught poses in ZERO-RELATIVE degrees (poses.yaml / arm_gui slider values)."""

    poses: dict[str, tuple[float, ...]]
    open_deg: float
    closed_deg: float
    place_polygon_px: tuple[tuple[float, float], ...] | None = None

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
    )
    if p.gripper_deg(130.0) == p.closed_deg:
        log.info(
            "gripper: closed_deg %.0f is the limit (0 = fully shut): re-grasps close to %.0f deg",
            p.closed_deg,
            p.closed_deg,
        )
    return p


def mock_demo_poses() -> DemoPoses:
    """SIMULATION-ONLY (FAKE) poses for --dry-run."""
    fake: dict[str, tuple[float, ...]] = {
        "home": (0.0, -30.0, 90.0, 30.0, 0.0, 0.0),
        "zone_A_pre": (15.0, 10.0, 60.0, 40.0, 0.0, 0.0),
        "zone_A_grasp": (15.0, 25.0, 55.0, 50.0, 0.0, 0.0),
        "place_pre": (-25.0, 10.0, 60.0, 40.0, 0.0, 0.0),
        "place": (-25.0, 25.0, 55.0, 50.0, 0.0, 0.0),
    }
    return DemoPoses(fake, open_deg=-40.0, closed_deg=0.0)  # same gripper map as the arm


# ------------------------------------------------------------------ live status (web panel)
@dataclass
class DemoStatus:
    state: str = "STARTUP"
    plan: dict[str, Any] = field(default_factory=dict)
    verdicts: list[dict[str, Any]] = field(default_factory=list)
    recovery: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    ram: str = "n/a"
    closure_pct: float | None = None
    outcome: str | None = None
    episode: int = 0
    mode: str = ""
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

    def add(self, key: str, item: Any) -> None:
        with self._lock:
            getattr(self, key).append(item)

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
.SLIPPING,.EMPTY,.ABORT{color:#f55}.GRASPED,.DONE{color:#4f4}.OCCLUDED,.TIMEOUT{color:#fa0}
</style></head><body><img src="/overhead.mjpg"><img src="/wrist.mjpg">
<div><a href="/snapshot" target="_blank" style="color:#8cf">snapshot</a> &nbsp;
<span id="mode"></span></div><div>State: <span id="st">?</span> &nbsp; closure <span
id="cl"></span> &nbsp; RAM <span id="ram"></span></div><div class="box" id="plan"></div>
<div class="box" id="ver"></div>
<div class="box" id="rec"></div><div class="box" id="ev"></div><script>
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
async function tick(){try{const s=await (await fetch('/state.json')).json();
st.textContent=s.state+(s.outcome?' ('+s.outcome+')':'');st.className=s.state;
mode.textContent=s.mode;cl.textContent=s.closure_pct==null?'-':s.closure_pct.toFixed(0)+'%';ram.textContent=s.ram;
const p=s.plan||{};plan.innerHTML='<b>PLAN</b> ['+esc(p.backend||'-')+'] '+
esc(p.note||'')+'\\nraw: '+esc(p.raw||'-')+'\\nparsed: '+esc(p.parsed||'-');
ver.innerHTML='<b>CRITIC</b>\\n'+s.verdicts.map(v=>esc(v.state)+': <span class="'+
v.verdict+'">'+v.verdict+'</span> ('+v.latency_ms.toFixed(0)+' ms) raw='+esc(v.raw)).join('\\n');
rec.innerHTML='<b>RECOVERY</b>\\n'+esc(s.recovery.join('\\n')||'-');
ev.innerHTML='<b>LOG</b>\\n'+esc(s.events.slice(-12).join('\\n'))}catch(e){}}
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
    ) -> None:
        web = self
        acts: dict[str, Callable[[dict[str, str]], dict[str, Any]]] = actions or {}

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

    def frame(self, image: np.ndarray, tag: str) -> str:
        self.frame_n += 1
        return self.r.logger.save_frame(
            f"{self.episode_id}/{self.frame_n:03d}_{self.s.state}_{tag}.jpg", image
        )

    def state(self, name: str, note: str = "") -> None:
        self.s.set(state=name)
        self.s.event(f"{name} {note}".strip())
        self.rec("transition", reason=note)

    # ---- motion: one segment, wait for DONE (the motion thread dispatches + polls)
    def move(self, q_fw: np.ndarray, what: str) -> None:
        seg = self.motion.command(self.r.joints.fw_to_host_deg(q_fw.tolist()))
        self.q = q_fw.copy()
        t_end = time.perf_counter() + 120.0
        while time.perf_counter() < t_end:
            ms = self.motion.current()
            if ms.seg_id == seg and ms.phase == FAULT:
                raise DemoAbort(f"LINK: {what}: {ms.fault}")
            if ms.seg_id == seg and ms.done:
                self.rec("segment_done", target={"label": what})
                return
            time.sleep(0.02)
        raise DemoAbort(f"{what}: no DONE within 120 s")

    def pose(self, name: str, closure: float) -> None:
        self.move(self.r.poses.pose(name, closure), f"{name} @ {closure:.0f}%")

    def grip(self, closure: float, what: str) -> None:
        q = self.q.copy()
        q[GRIPPER_INDEX] = self.r.poses.gripper_deg(closure)
        self.s.set(closure_pct=closure)
        self.move(q, f"{what} gripper {closure:.0f}% ({q[GRIPPER_INDEX]:.1f} deg)")

    # ---- Critic: submit + poll (never a blocking wait on the model)
    def verify(self, state: str) -> Verdict:
        self.state(state)
        time.sleep(self.r.dcfg.settle_s)  # frame after the gripper settled
        verdict = Verdict.INVALID
        for attempt in range(2):  # OCCLUDED / INVALID / TIMEOUT: re-query once
            img = self.r.wrist_frame()  # real Trinet: waits <= fresh_frame_timeout_s
            if img is None and self.r.critic is not None:  # wrist dropped / re-enumerating
                verdict = Verdict.OCCLUDED
                why = "no fresh wrist frame (camera reconnecting?)"
                self.s.add(
                    "verdicts", {"state": state, "verdict": verdict.value, "latency_ms": 0.0,
                                 "raw": why},
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
                    {"state": state, "verdict": "OFF", "latency_ms": 0.0, "raw": "pass-through"},
                )
                self.s.event(f"Critic OFF: {state} passes through")
                self.rec("verdict", verdict=None, critic_skipped=True, frame_raw=raw_path)
                return Verdict.GRASPED
            assert img is not None  # None was handled above (Critic on)
            rid = self.r.critic.submit(img, state)
            if rid is None:
                raise DemoAbort("Critic queue full")
            deadline = time.perf_counter() + self.r.cfg.critic.timeout_s + 1.0
            res = None
            while res is None and time.perf_counter() < deadline:
                r = self.r.critic.poll()
                if r is not None and r.request_id == rid:
                    res = r
                else:
                    time.sleep(0.02)
            verdict = res.verdict if res is not None else Verdict.TIMEOUT
            lat = res.latency_ms if res is not None else 0.0
            raw = res.raw_text if res is not None else None
            judged = (
                self.frame(res.image_bgr, "judged")
                if res is not None and res.image_bgr is not None
                else None
            )
            self.s.add(
                "verdicts",
                {"state": state, "verdict": verdict.value, "latency_ms": lat, "raw": raw},
            )
            self.s.event(f"Critic {state}: {verdict.value} ({lat:.0f} ms, raw={raw!r})")
            self.rec(
                "verdict",
                verdict=verdict,
                verdict_latency_ms=lat,
                raw_text=raw,
                frame=judged,
                frame_raw=raw_path,
                attempt=attempt,
            )
            if verdict in GOOD:
                return verdict
        return verdict

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
        why = "rules planner (--no-planner or Qwen not loaded)"
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
        return self.show_plan(res, "rules", why, note, qwen_raw)

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
    def run(self, induce_slip: bool) -> bool:
        r, d = self.r, self.r.dcfg
        self.episode_id = time.strftime("%Y%m%d-%H%M%S")
        self.s.set(
            verdicts=[], recovery=[], outcome=None, closure_pct=0.0, episode=self.s.episode + 1
        )
        self.rec("episode_start", induce_slip=induce_slip, instruction=d.instruction)
        closure = 0.0
        try:
            self.state("HOME")
            self.pose("home", 0.0)
            res = self.plan()
            if res.chain is None:
                raise DemoAbort(f"no valid plan ({res.error})")
            self.state("APPROACH")
            self.pose("zone_A_pre", 0.0)
            self.pose("zone_A_grasp", 0.0)
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
            regrasps = 0
            while True:
                self.state("GRASP", f"closure {closure:.0f}%")
                self.grip(closure, "close")
                v = self.verify("VERIFY_GRASP")
                if v is Verdict.GRASPED:
                    self.state("LIFT")
                    self.pose("zone_A_pre", closure)
                    v = self.verify("VERIFY_HOLD")
                    if v is Verdict.GRASPED:
                        break
                if v not in (Verdict.SLIPPING, Verdict.EMPTY):
                    raise DemoAbort(f"Critic {v.value} after a re-query")
                if regrasps >= d.max_regrasps:
                    raise DemoAbort(f"Critic {v.value}: re-grasp budget exhausted ({regrasps})")
                regrasps += 1
                closure = base + d.regrasp_step_pct * regrasps
                step = f"k={regrasps}: {v.value} -> re-open, re-grasp at {closure:.0f}%"
                self.s.add("recovery", step)
                self.state("RECOVER", step)
                self.rec("recover", action=step, attempt=regrasps)
                at_grasp = r.poses.poses["zone_A_grasp"][:GRIPPER_INDEX]
                if not np.allclose(self.q[:GRIPPER_INDEX], at_grasp):
                    self.pose("zone_A_grasp", self.s.closure_pct or 0.0)  # back down, holding
                self.grip(0.0, "re-open")
            self.state("TRANSPORT")
            self.pose("place_pre", closure)
            self.state("PLACE")
            self.pose("place", closure)
            self.grip(0.0, "release")
            self.state("HOME")
            self.pose("home", 0.0)
            self.finish("success")
            return True
        except DemoAbort as e:
            self.abort(str(e))
            return False

    def abort(self, why: str) -> None:
        self.state("ABORT", why)
        try:
            if "LINK" in why:
                self.r.arm.stop()  # unreliable link: brake and hold, no more motion
                self.s.event("LINK fault: stop sent, servos hold. Keep the E-stop in reach.")
            else:
                g = self.q[GRIPPER_INDEX]  # keep the gripper: a held block is not dropped
                q = np.asarray(self.r.poses.poses["home"], dtype=float).copy()
                q[GRIPPER_INDEX] = g
                self.move(q, "home (abort)")
        except Exception as e:
            self.s.event(f"retreat failed: {e}; stop", logging.ERROR)
            try:
                self.r.arm.stop()
            except Exception:
                log.exception("stop failed")
        self.finish("abort", why)

    def finish(self, outcome: str, why: str = "") -> None:
        self.s.set(outcome=outcome, state="DONE" if outcome == "success" else "ABORT")
        self.s.event(f"episode {outcome.upper()} {why}".strip())
        self.rec("episode_end", outcome=outcome, report=why or None)


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


def free_torch(what: str) -> None:
    gc.collect()
    try:
        import torch  # lazy

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    log.info("freed %s", what)


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

    # models: Qwen first, plan later; Moondream next; free Qwen if RAM is short
    ram(status, "before model load")
    qwen: QwenPlanner | None = None
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
            status.event(f"Qwen load failed ({e}): rules planner", logging.ERROR)
        ram(status, "after Qwen load")
    backend: CriticBackend | None = None
    if args.critic == "moondream":
        status.event("loading Moondream2 (bf16)...")
        from lbyr.ai.critic import Moondream2Critic

        mpath = dcfg.moondream_path.expanduser()
        backend = Moondream2Critic(replace(cfg.critic, model_path=mpath))
    elif args.critic == "scripted":
        backend = scripted_critic(args, dcfg)
    free = ram(status, f"after Critic ({args.critic}) setup")
    if qwen is not None and free is not None and free < dcfg.min_free_ram_mb:
        status.event(
            f"free RAM {free:.0f} MB < {dcfg.min_free_ram_mb:.0f} MB: freeing Qwen (rules plan)",
            logging.WARNING,
        )
        qwen = None
        free_torch("Qwen2.5-VL")
        ram(status, "after freeing Qwen")
    from lbyr.perception.wrist import build_wrist_preprocess

    item = wrist.buffer.get()
    size = (item.value.shape[1], item.value.shape[0]) if item is not None else None
    pre = build_wrist_preprocess(wc, size)
    rig = _finish_rig(
        args, cfg, dcfg, status, arm, joints, poses, overhead_buf, wrist.buffer, detect,
        backend, qwen, stoppers, pre,
    )  # fmt: skip
    rig.wrist_fresh = wrist.fresh
    return rig


def scripted_critic(args: argparse.Namespace, dcfg: DemoConfig) -> ScriptedCritic:
    """Fake Critic (no model): --script, else SLIPPING,GRASPED with --induce-slip, else
    DemoConfig.scripted_verdicts."""
    if args.script:
        script = [v.strip().upper() for v in args.script.split(",") if v.strip()]
    elif args.induce_slip:
        script = ["SLIPPING", "GRASPED"]
    else:
        script = list(dcfg.scripted_verdicts)
    bad = [v for v in script if v not in Verdict.__members__]
    if bad:
        raise LBYRError(f"--script: unknown verdicts {bad}")
    return ScriptedCritic(script, delay_s=0.2)


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
    planner = "qwen" if qwen is not None else "rules"
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


def ask(prompt: str, snaps: Snapshots) -> str:
    """input() that handles 'w' (wrist snapshot) itself and asks again."""
    while True:
        ans = input(prompt).strip().lower()
        if ans != "w":
            return ans
        try:
            log.info("snapshot: %s", snaps.take())
        except Exception as e:
            log.error("snapshot failed: %s", e)


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
        choices=("moondream", "scripted", "off"),
        default=None,
        help="moondream (default; dry-run: scripted) | scripted (fake, no model) | off (baseline)",
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
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    if args.no_planner or args.rehearsal or args.dry_run:
        args.planner = "rules"
    if args.critic is None:
        args.critic = "scripted" if (args.rehearsal or args.dry_run) else "moondream"
    if args.critic_off:
        args.critic = "off"
    if args.dry_run and args.critic == "moondream":
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
        web = DemoWeb(
            dcfg.http_host,
            dcfg.http_port,
            status,
            {"overhead": frame_of("overhead"), "wrist": frame_of("wrist")},
            {"snapshot": lambda q: snaps.take(q.get("crop"))},
        )
        web.start()
        log.info("web view on http://%s:%d", dcfg.http_host, web.port)
        try:
            build = build_dry_rig if args.dry_run else build_real_rig
            rig = build(args, dcfg, status)
        except Exception as e:  # hardware / calibration / model load: refuse, exit 2
            log.exception("setup failed")
            status.set(state="SETUP FAILED")
            status.event(f"setup refused: {e}", logging.ERROR)
            return 2
        rig_box.append(rig)
        demo = Demo(rig)
        once = args.once or args.dry_run
        if not args.dry_run:
            ask(f"Ready (web view on port {web.port}). [Enter] start, [w] wrist snapshot: ", snaps)
        while True:
            ok = demo.run(args.induce_slip)
            if once:
                return 0 if ok else 1
            ans = ask("[Enter] run again, [s] again with slip, [w] snapshot, [q] quit: ", snaps)
            if ans == "q":
                return 0 if ok else 1
            args.induce_slip = ans == "s"
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
