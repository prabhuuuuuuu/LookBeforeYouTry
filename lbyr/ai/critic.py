"""Async visual grasp-verification Critic (bible 11.2).

Event-driven: the FSM submits a request (non-blocking) and polls the verdict queue
(non-blocking). Inference runs on the CriticWorker thread under an explicit timeout.
Moondream2 cost (bible 9.2): ~4 GB FP16 / ~1.2 GB GGUF-Q4 VRAM, 0.3-0.8 s per query.
"""

from __future__ import annotations

import itertools
import logging
import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from lbyr.ai.timeout import TimeoutResult, TimeoutRunner
from lbyr.config import CriticConfig

log = logging.getLogger(__name__)


class Verdict(str, Enum):
    GRASPED = "GRASPED"
    SLIPPING = "SLIPPING"
    EMPTY = "EMPTY"
    OCCLUDED = "OCCLUDED"
    TIMEOUT = "TIMEOUT"  # no answer within CriticConfig.timeout_s
    INVALID = "INVALID"  # unparseable answer or backend error


MODEL_CLASSES = (Verdict.GRASPED, Verdict.SLIPPING, Verdict.EMPTY, Verdict.OCCLUDED)

# LOCKED prompt contract (bible 11.2). Do not edit without updating the bible.
CRITIC_PROMPT = (
    "You are a robot grasp inspector.\n"
    "Image = wrist camera after gripper closed.\n"
    "Answer ONE word: GRASPED | SLIPPING | EMPTY | OCCLUDED"
)

_STRIP_CHARS = " \t\r\n.,;:!?\"'`*_()[]{}<>"


def parse_verdict(text: str | None) -> Verdict:
    """Map raw model text to a Verdict. Exactly one class word is accepted, case-insensitive,
    ignoring surrounding whitespace/punctuation/markdown. Anything else -> INVALID."""
    if not text:
        return Verdict.INVALID
    words = [w.strip(_STRIP_CHARS) for w in text.split()]
    words = [w for w in words if w]
    if len(words) != 1:
        return Verdict.INVALID
    word = words[0].upper()
    for v in MODEL_CLASSES:
        if word == v.value:
            return v
    return Verdict.INVALID


# ------------------------------------------------------------------ backends
class CriticBackend(Protocol):
    def infer(self, image_bgr: np.ndarray, prompt: str) -> str: ...


class ScriptedCritic:
    """Test/mock backend: returns scripted answers in order (last one repeats).

    ``delay_s`` simulates inference latency (e.g. 2 s to stress the non-blocking design).
    """

    def __init__(self, script: Sequence[Verdict | str], delay_s: float = 0.0) -> None:
        if not script:
            raise ValueError("script must not be empty")
        self._script = [s.value if isinstance(s, Verdict) else s for s in script]
        self._i = 0
        self._lock = threading.Lock()
        self.delay_s = delay_s
        self.calls = 0

    def infer(self, image_bgr: np.ndarray, prompt: str) -> str:
        with self._lock:
            answer = self._script[min(self._i, len(self._script) - 1)]
            self._i += 1
            self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        return answer


class Moondream2Critic:
    """Moondream2 (FP16 or GGUF-Q4 via llama.cpp), optional LoRA adapter.

    TODO: implement on the Jetson with local weights only (``cfg.model_path``), greedy
    decoding (temperature 0). Heavy imports must stay lazy, inside this class.
    """

    def __init__(self, cfg: CriticConfig) -> None:
        cfg.validate()
        self.cfg = cfg
        raise NotImplementedError("Moondream2Critic not implemented yet (TODO); use ScriptedCritic")

    def infer(self, image_bgr: np.ndarray, prompt: str) -> str:
        raise NotImplementedError


# ------------------------------------------------------------------ worker
@dataclass(frozen=True)
class CriticRequest:
    request_id: int
    image_bgr: np.ndarray  # raw wrist frame (dewarped by the worker)
    state: str  # FSM state that asked
    t_submit: float


@dataclass(frozen=True)
class CriticResult:
    request_id: int
    state: str
    verdict: Verdict
    raw_text: str | None
    latency_ms: float  # submit -> verdict (includes queueing + dewarp + inference)
    image_bgr: np.ndarray | None = None  # the (dewarped) image actually judged


class CriticWorker(threading.Thread):
    """Request queue -> (dewarp) -> backend under timeout -> verdict queue."""

    def __init__(
        self,
        backend: CriticBackend,
        cfg: CriticConfig,
        preprocess: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        super().__init__(name="critic", daemon=True)
        cfg.validate()
        self._backend = backend
        self._preprocess = preprocess
        self._runner = TimeoutRunner(cfg.timeout_s, name="critic-infer")
        self._requests: queue.Queue[CriticRequest] = queue.Queue(maxsize=cfg.request_queue_size)
        self._results: queue.Queue[CriticResult] = queue.Queue()
        self._ids = itertools.count(1)
        self._stop_event = threading.Event()

    # ---- FSM-facing, non-blocking API
    def submit(self, image_bgr: np.ndarray, state: str) -> int | None:
        """Queue a verification request. Returns its id, or None if the queue is full."""
        req = CriticRequest(next(self._ids), image_bgr, state, time.perf_counter())
        try:
            self._requests.put_nowait(req)
        except queue.Full:
            log.warning("Critic request queue full; dropping request from %s", state)
            return None
        return req.request_id

    def poll(self) -> CriticResult | None:
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    # ---- worker thread
    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                req = self._requests.get(timeout=0.1)
            except queue.Empty:
                continue
            self._results.put(self._process(req))

    def _process(self, req: CriticRequest) -> CriticResult:
        image = req.image_bgr
        raw: str | None = None
        try:
            if self._preprocess is not None:
                image = self._preprocess(image)
            out = self._runner.run(lambda: self._backend.infer(image, CRITIC_PROMPT))
            if isinstance(out, TimeoutResult):
                verdict = Verdict.TIMEOUT
                if out.busy:
                    log.warning("Critic backend still busy with an abandoned call")
            else:
                raw = out
                verdict = parse_verdict(out)
        except Exception:
            log.exception("Critic backend/preprocess failed")
            verdict = Verdict.INVALID
        latency_ms = (time.perf_counter() - req.t_submit) * 1000
        return CriticResult(req.request_id, req.state, verdict, raw, latency_ms, image)

    def stop(self, timeout: float = 1.0) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout)
