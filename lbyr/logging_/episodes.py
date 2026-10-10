"""Episode logging (bible Section 15): ``episodes.jsonl`` + decision-point ``frames/``.

All disk I/O happens on the logger thread; callers only enqueue (never block).
Each session gets its own directory::

    data/episodes/<session>/episodes.jsonl
    data/episodes/<session>/frames/<episode_id>/<n>_<state>_<tag>.jpg

VRAM: the Orin has unified memory, so ``vram_mb`` is tegrastats' "RAM used" figure
(CPU+GPU). It is ``None`` when tegrastats is unavailable (e.g. on a laptop).
"""

from __future__ import annotations

import json
import logging
import queue
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Bible Section 15 per-tick/decision fields. Every record contains all of them.
REQUIRED_FIELDS: tuple[str, ...] = (
    "t",
    "fsm_state",
    "joints",
    "target",
    "verdict",
    "verdict_latency_ms",
    "planner_latency_ms",
    "vram_mb",
    "outcome",
)

_TEGRA_RAM = re.compile(r"RAM (\d+)/(\d+)MB")


def parse_tegrastats_ram_mb(line: str) -> float | None:
    m = _TEGRA_RAM.search(line)
    return float(m.group(1)) if m else None


@dataclass(frozen=True)
class RamReading:
    used_mb: float
    total_mb: float
    source: str  # "tegrastats" | "meminfo"

    @property
    def free_mb(self) -> float:
        return self.total_mb - self.used_mb


def parse_tegrastats_ram(line: str) -> RamReading | None:
    m = _TEGRA_RAM.search(line)
    return RamReading(float(m.group(1)), float(m.group(2)), "tegrastats") if m else None


def parse_meminfo(text: str) -> RamReading | None:
    """``/proc/meminfo``: used = MemTotal - MemAvailable (kB -> MB)."""
    kb: dict[str, float] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts:
            kb[key.strip()] = float(parts[0])
    if "MemTotal" not in kb or "MemAvailable" not in kb:
        return None
    return RamReading(
        (kb["MemTotal"] - kb["MemAvailable"]) / 1024.0, kb["MemTotal"] / 1024.0, "meminfo"
    )


def read_system_ram(sample_s: float = 0.6) -> RamReading | None:
    """One system-RAM sample (the Orin's unified memory = CPU + GPU).

    tegrastats first (runs ~``sample_s``, interval 100 ms), else ``/proc/meminfo``, else None.
    Startup-only helper: it blocks for up to ``sample_s``.
    """
    exe = shutil.which("tegrastats")
    if exe is not None:
        out: str | bytes | None = None
        try:
            subprocess.run(
                [exe, "--interval", "100"], capture_output=True, timeout=sample_s, check=False
            )
        except subprocess.TimeoutExpired as e:  # tegrastats never exits on its own
            out = e.stdout
        except OSError:
            log.warning("could not run tegrastats", exc_info=True)
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        for line in reversed((out or "").splitlines()):
            reading = parse_tegrastats_ram(line)
            if reading is not None:
                return reading
    try:
        return parse_meminfo(Path("/proc/meminfo").read_text())
    except OSError:
        return None


class VramMonitor:
    """Background reader of ``tegrastats``; ``latest_mb`` is None if unavailable."""

    def __init__(self, interval_ms: int = 1000) -> None:
        self.latest_mb: float | None = None
        self.peak_mb: float | None = None
        self._interval_ms = interval_ms
        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        exe = shutil.which("tegrastats")
        if exe is None:
            log.info("tegrastats not found: vram_mb will be null")
            return
        self._proc = subprocess.Popen(
            [exe, "--interval", str(self._interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._thread = threading.Thread(target=self._read, name="tegrastats", daemon=True)
        self._thread.start()

    def _read(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            mb = parse_tegrastats_ram_mb(line)
            if mb is not None:
                self.latest_mb = mb
                self.peak_mb = mb if self.peak_mb is None else max(self.peak_mb, mb)

    def reset_peak(self) -> None:
        """Start a new peak window (e.g. per benchmark variant) from the latest sample."""
        self.peak_mb = self.latest_mb

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            self._proc = None


@dataclass(frozen=True)
class _Record:
    data: dict[str, Any]


@dataclass(frozen=True)
class _Frame:
    path: Path
    image: np.ndarray


_STOP = object()


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return [round(float(v), 3) for v in x.reshape(-1)]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (tuple, list)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, float):
        return round(x, 3)
    if hasattr(x, "value") and isinstance(x.value, str):  # str Enums
        return x.value
    return x


def make_record(**fields: Any) -> dict[str, Any]:
    """Build a record with every REQUIRED_FIELD present (None if not provided)."""
    rec: dict[str, Any] = {k: None for k in REQUIRED_FIELDS}
    rec["t"] = time.time()
    rec.update(fields)
    return {k: _jsonable(v) for k, v in rec.items()}


class EpisodeLogger(threading.Thread):
    def __init__(self, session_dir: Path) -> None:
        super().__init__(name="logger", daemon=True)
        self.session_dir = session_dir
        self.jsonl_path = session_dir / "episodes.jsonl"
        self.frames_dir = session_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._q: queue.Queue[Any] = queue.Queue()
        self.records_written = 0
        self.frames_written = 0

    @classmethod
    def for_new_session(cls, episodes_dir: Path) -> EpisodeLogger:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return cls(episodes_dir / stamp)

    # ---- non-blocking producer API
    def log(self, record: dict[str, Any]) -> None:
        self._q.put_nowait(_Record(record))

    def save_frame(self, rel_path: str, image: np.ndarray) -> str:
        """Queue a frame for saving; returns its path relative to the session dir."""
        path = self.frames_dir / rel_path
        self._q.put_nowait(_Frame(path, image.copy()))
        return str(path.relative_to(self.session_dir).as_posix())

    # ---- thread
    def run(self) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            while True:
                item = self._q.get()
                if item is _STOP:
                    break
                try:
                    if isinstance(item, _Record):
                        f.write(json.dumps(item.data, separators=(",", ":")) + "\n")
                        f.flush()
                        self.records_written += 1
                    elif isinstance(item, _Frame):
                        _write_image(item.path, item.image)
                        self.frames_written += 1
                except Exception:
                    log.exception("logger failed to write %s", type(item).__name__)

    def stop(self, timeout: float = 5.0) -> None:
        """Flush everything queued so far, then stop."""
        self._q.put(_STOP)
        if self.is_alive():
            self.join(timeout)


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2  # lazy
    except ImportError:
        np.save(path.with_suffix(".npy"), image)
        return
    cv2.imwrite(str(path), image)
