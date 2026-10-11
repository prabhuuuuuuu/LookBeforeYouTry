"""Async visual grasp-verification Critic (bible 11.2).

Event-driven: the FSM submits a request (non-blocking) and polls the verdict queue
(non-blocking). Inference runs on the CriticWorker thread under an explicit timeout.

Runtime input (M-1, 2026-10-10): wrist frame -> dewarp -> 378x378 crop
(``CriticConfig.crop_box_norm``, resized) -> Moondream2 BF16. Measured on the Orin NX
40 W: ~4.1 GB, p50 0.85 s (1.35 s on the full 640x480 frame). See bible 9.2.
"""

from __future__ import annotations

import itertools
import logging
import queue
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from lbyr.ai.cuda_mem import log_cuda_allocated
from lbyr.ai.imaging import bgr_to_pil, critic_crop
from lbyr.ai.offline import force_offline
from lbyr.ai.timeout import TimeoutResult, TimeoutRunner, run_with_timeout
from lbyr.config import CriticConfig
from lbyr.errors import ConfigError, LBYRError

log = logging.getLogger(__name__)


class Verdict(str, Enum):
    GRASPED = "GRASPED"
    SLIPPING = "SLIPPING"
    EMPTY = "EMPTY"
    OCCLUDED = "OCCLUDED"
    TIMEOUT = "TIMEOUT"  # no answer within CriticConfig.timeout_s
    INVALID = "INVALID"  # unparseable answer or backend error
    LOW_CONF = "LOW_CONF"  # class probabilities given, max < CriticConfig.low_conf_threshold


MODEL_CLASSES = (Verdict.GRASPED, Verdict.SLIPPING, Verdict.EMPTY, Verdict.OCCLUDED)


@dataclass(frozen=True)
class CriticOutput:
    """Structured backend answer (backends with ``classify``: probe, scripted).

    ``probs`` = softmax probability per MODEL_CLASSES value (None = the backend has none,
    e.g. zero-shot text). The worker turns a max probability below
    ``CriticConfig.low_conf_threshold`` into LOW_CONF (:func:`apply_confidence`)."""

    verdict: Verdict
    raw_text: str
    probs: dict[str, float] | None = None


def apply_confidence(verdict: Verdict, probs: dict[str, float] | None, threshold: float) -> Verdict:
    """LOW_CONF when probabilities exist and the largest is below ``threshold``."""
    if probs and verdict in MODEL_CLASSES and max(probs.values()) < threshold:
        return Verdict.LOW_CONF
    return verdict


def parse_script(text: str) -> tuple[list[str], dict[str, list[str]]]:
    """Scripted-Critic spec -> (default sequence, per-state sequences).

    ``"SLIPPING,GRASPED"`` = one sequence for every state (the old form).
    ``"VERIFY_GRASP=EMPTY:0.91,GRASPED;TRANSPORT=GRASPED,SLIPPING:0.78,SLIPPING:0.81"`` =
    per-state sequences (``;``-separated; a part without ``STATE=`` is the default).
    An entry is ``VERDICT`` or ``VERDICT:p`` (p = probability of that class; the other three
    share 1 - p). Each sequence repeats its last entry."""
    default: list[str] = []
    per_state: dict[str, list[str]] = {}
    for part in (p.strip() for p in text.split(";")):
        if not part:
            continue
        state, sep, body = part.partition("=")
        items = [x.strip().upper() for x in (body if sep else part).split(",") if x.strip()]
        for it in items:
            word, _, p = it.partition(":")
            if word not in Verdict.__members__:
                raise ValueError(f"unknown verdict {word!r} in script entry {it!r}")
            if p and not 0.0 <= float(p) <= 1.0:
                raise ValueError(f"probability out of [0, 1] in script entry {it!r}")
        if not items:
            raise ValueError(f"empty script part {part!r}")
        if sep:
            per_state[state.strip().upper()] = items
        else:
            default = items
    return default, per_state


def scripted_output(entry: str) -> CriticOutput:
    """``"EMPTY:0.91"`` -> EMPTY with probs {EMPTY: 0.91, others 0.03}; ``"EMPTY"`` -> no probs."""
    word, _, p = entry.partition(":")
    verdict = parse_verdict(word)
    probs = None
    if p:
        pf = float(p)
        probs = {c.value: (pf if c.value == word else (1.0 - pf) / 3.0) for c in MODEL_CLASSES}
    return CriticOutput(verdict, entry, probs)


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
    ``per_state`` (``classify`` only): own sequence per requesting state, e.g. the monitor
    states LIFT / TRANSPORT, whose call count depends on timing; other states use ``script``.
    Entries may carry a probability (``"SLIPPING:0.78"``, see :func:`scripted_output`).
    """

    def __init__(
        self,
        script: Sequence[Verdict | str],
        delay_s: float = 0.0,
        per_state: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        if not script:
            raise ValueError("script must not be empty")
        self._script = [s.value if isinstance(s, Verdict) else s for s in script]
        self._per_state = {k: list(v) for k, v in (per_state or {}).items() if v}
        self._i = 0
        self._state_i: dict[str, int] = {}
        self._lock = threading.Lock()
        self.delay_s = delay_s
        self.calls = 0

    def _next(self, state: str | None) -> str:
        with self._lock:
            self.calls += 1
            if state is not None and state in self._per_state:
                seq, i = self._per_state[state], self._state_i.get(state, 0)
                self._state_i[state] = i + 1
            else:
                seq, i = self._script, self._i
                self._i += 1
            return seq[min(i, len(seq) - 1)]

    def infer(self, image_bgr: np.ndarray, prompt: str) -> str:
        answer = self._next(None).partition(":")[0]
        if self.delay_s:
            time.sleep(self.delay_s)
        return answer

    def classify(self, image_bgr: np.ndarray, state: str) -> CriticOutput:
        entry = self._next(state)
        if self.delay_s:
            time.sleep(self.delay_s)
        return scripted_output(entry)


def _load_moondream2(path: Path, device: str, dtype: str = "fp16") -> Any:
    """Load Moondream2 from LOCAL files only, as ``dtype`` ("fp16" | "bf16"), on ``device``.

    The model's remote code builds its modules in bfloat16 and hard-codes a bfloat16 cast
    of the image crops, so FP16 needs two things: casting the weights, and a forward
    pre-hook that casts the crops to the weight dtype. Outputs were checked to match bf16.
    BF16 is the model's native dtype (the Orin supports it): no cast, no patch.
    """
    force_offline()  # edge-only: never touch the network
    import torch
    from transformers import AutoModelForCausalLM

    tdtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    model: Any = AutoModelForCausalLM.from_pretrained(
        str(path), trust_remote_code=True, dtype=tdtype, local_files_only=True
    )
    model = model.to(tdtype).to(device).eval()
    try:
        patch_emb = model.model.vision["patch_emb"]
    except (AttributeError, KeyError, TypeError) as e:
        raise LBYRError(
            f"{path}: unexpected Moondream2 remote-code layout (need the 2025-06-21 revision)"
        ) from e
    if tdtype is not torch.bfloat16:
        patch_emb.register_forward_pre_hook(lambda mod, args: (args[0].to(mod.weight.dtype),))
    return model


class Moondream2Critic:
    """Moondream2 on CUDA, BF16 by default (``cfg.dtype``; bible 11.2, ~4.1 GB).

    Weights come from ``cfg.model_path`` only (no downloads). The model is loaded once here
    and warmed up with one dummy query (under ``cfg.warmup_timeout_s``). Decoding is greedy
    (temperature 0, ``max_tokens = cfg.max_new_tokens``). ``infer`` returns the raw answer
    text; the verdict is produced by :func:`parse_verdict` in :class:`CriticWorker`, which
    also enforces ``cfg.timeout_s`` around every call.

    ``model`` injects an already-built object with Moondream's ``query`` API (tests).
    """

    def __init__(self, cfg: CriticConfig, *, model: Any | None = None) -> None:
        cfg.validate()
        if cfg.lora_adapter_path is not None:
            raise NotImplementedError("Moondream2 LoRA adapter loading is not implemented yet")
        self.cfg = cfg
        self._lock = threading.Lock()  # one query at a time: the KV cache is shared state
        if model is None:
            if cfg.model_path is None:
                raise ConfigError("CriticConfig.model_path is not set (local weights only)")
            t0 = time.perf_counter()
            model = _load_moondream2(cfg.model_path, cfg.device, cfg.dtype)
            log.info("Moondream2 (%s) loaded in %.1f s", cfg.dtype, time.perf_counter() - t0)
            log_cuda_allocated("Moondream2 Critic")
        self._model = model
        self._warm_up()

    def _warm_up(self) -> None:
        side = self.cfg.crop_size_px
        shape = (side, side, 3) if self.cfg.crop_enabled else (480, 640, 3)
        dummy = np.zeros(shape, dtype=np.uint8)  # same size as the runtime input
        t0 = time.perf_counter()
        out = run_with_timeout(
            lambda: self.infer(dummy, CRITIC_PROMPT), self.cfg.warmup_timeout_s, "critic-warmup"
        )
        if isinstance(out, TimeoutResult):
            raise LBYRError(f"Critic warm-up exceeded {self.cfg.warmup_timeout_s:.0f} s")
        log.info("Critic warm-up: %r in %.0f ms", out, (time.perf_counter() - t0) * 1000)

    def infer(self, image_bgr: np.ndarray, prompt: str) -> str:
        image = bgr_to_pil(image_bgr)
        settings = {"temperature": self.cfg.temperature, "max_tokens": self.cfg.max_new_tokens}
        with self._lock:
            result = self._model.query(image, prompt, settings=settings)
        return str(result["answer"]).strip()

    def try_compile(self) -> str | None:
        """BENCHMARK ONLY: call the Moondream revision's own ``compile()`` (torch.compile of
        the vision encoder / prefill / decode). Returns None on success, else why it was
        skipped. Compilation is lazy: the next calls are slow and can still fail."""
        inner = getattr(self._model, "model", None)
        owner = next((c for c in type(inner).__mro__ if "compile" in vars(c)), None)
        if inner is None or owner is None or owner.__module__.startswith("torch"):
            return "this Moondream2 revision exposes no compile() method"
        with self._lock:
            try:
                inner.compile()
            except Exception as e:
                return f"compile() failed: {type(e).__name__}: {e}"
        return None


# ------------------------------------------------------------------ worker
@dataclass(frozen=True)
class CriticRequest:
    request_id: int
    image_bgr: np.ndarray  # raw wrist frame (dewarped by the worker)
    state: str  # FSM state that asked
    t_submit: float
    monitor: bool = False  # produced by the continuous monitor, not submitted by the FSM


@dataclass(frozen=True)
class CriticResult:
    request_id: int
    state: str
    verdict: Verdict
    raw_text: str | None
    latency_ms: float  # submit -> verdict (includes queueing + dewarp + inference)
    image_bgr: np.ndarray | None = None  # the (dewarped, cropped) image actually judged
    infer_ms: float | None = None  # backend call only (None if it never ran)
    probs: dict[str, float] | None = None  # per MODEL_CLASSES value (classify backends)
    monitor: bool = False  # continuous-monitor result (see CriticWorker.start_monitor)


@dataclass(frozen=True)
class _Monitor:
    state: str
    source: Callable[[], np.ndarray | None]


class CriticWorker(threading.Thread):
    """Request queue -> (dewarp) -> crop -> backend under timeout -> verdict queue.

    ``preprocess`` is the dewarp (None = raw frame). The crop is applied when
    ``cfg.crop_enabled`` (box ``cfg.crop_box_norm``, default centred square), resized to
    ``cfg.crop_size_px``.

    Backends with ``classify(image, state) -> CriticOutput`` (probe, scripted) are called
    through it (verdict + probs; max prob < ``cfg.low_conf_threshold`` -> LOW_CONF); others
    through ``infer(image, CRITIC_PROMPT)`` + :func:`parse_verdict`.

    Continuous monitor (:meth:`start_monitor`): while no FSM request is queued, the worker
    re-runs back-to-back on ``source()`` (latest wrist frame) and posts results with
    ``monitor=True``. Everything stays on this thread: the FSM only polls, the motion
    thread is never involved.
    """

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
        self._cfg = cfg
        self._runner = TimeoutRunner(cfg.timeout_s, name="critic-infer")
        self._requests: queue.Queue[CriticRequest] = queue.Queue(maxsize=cfg.request_queue_size)
        self._results: queue.Queue[CriticResult] = queue.Queue()
        self._ids = itertools.count(1)
        self._stop_event = threading.Event()
        self._monitor: _Monitor | None = None
        self._monitor_lock = threading.Lock()

    # ---- FSM-facing, non-blocking API
    def start_monitor(self, state: str, source: Callable[[], np.ndarray | None]) -> None:
        """Judge ``source()`` back-to-back (results tagged ``monitor=True``, ``state``) until
        :meth:`stop_monitor`. ``source`` runs on the worker thread (it may wait for a fresh
        frame); None = no frame, retried after 50 ms. Non-blocking."""
        with self._monitor_lock:
            self._monitor = _Monitor(state, source)

    def stop_monitor(self) -> None:
        """Stop monitoring (non-blocking). A call already in flight still posts its result."""
        with self._monitor_lock:
            self._monitor = None

    @property
    def monitoring(self) -> str | None:
        with self._monitor_lock:
            return self._monitor.state if self._monitor is not None else None

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
            with self._monitor_lock:
                mon = self._monitor
            try:
                if mon is None:
                    req = self._requests.get(timeout=0.1)
                else:  # FSM requests first, then one monitor round
                    req = self._requests.get_nowait()
            except queue.Empty:
                if mon is None:
                    continue
                monitor_req = self._monitor_request(mon)
                if monitor_req is None:
                    self._stop_event.wait(0.05)
                    continue
                req = monitor_req
            self._results.put(self._process(req))

    def _monitor_request(self, mon: _Monitor) -> CriticRequest | None:
        t0 = time.perf_counter()
        try:
            frame = mon.source()
        except Exception:
            log.exception("Critic monitor frame source failed")
            return None
        with self._monitor_lock:
            if self._monitor is not mon:  # stopped / restarted while waiting for the frame
                return None
        if frame is None:
            log.debug("Critic monitor %s: no frame", mon.state)
            return None
        return CriticRequest(next(self._ids), frame, mon.state, t0, monitor=True)

    def _process(self, req: CriticRequest) -> CriticResult:
        image = req.image_bgr
        raw: str | None = None
        infer_ms: float | None = None
        probs: dict[str, float] | None = None
        classify = getattr(self._backend, "classify", None)
        try:
            if self._preprocess is not None:
                image = self._preprocess(image)
            if self._cfg.crop_enabled:
                image = critic_crop(image, self._cfg.crop_box_norm, self._cfg.crop_size_px)
            judged = image
            t0 = time.perf_counter()
            if classify is not None:
                out: Any = self._runner.run(lambda: classify(judged, req.state))
            else:
                out = self._runner.run(lambda: self._backend.infer(judged, CRITIC_PROMPT))
            infer_ms = (time.perf_counter() - t0) * 1000
            if isinstance(out, TimeoutResult):
                verdict = Verdict.TIMEOUT
                if out.busy:
                    log.warning("Critic backend still busy with an abandoned call")
            elif isinstance(out, CriticOutput):
                raw, probs = out.raw_text, out.probs
                verdict = apply_confidence(out.verdict, probs, self._cfg.low_conf_threshold)
            else:
                raw = out
                verdict = parse_verdict(out)
        except Exception:
            log.exception("Critic backend/preprocess failed")
            verdict = Verdict.INVALID
        latency_ms = (time.perf_counter() - req.t_submit) * 1000
        log.log(
            logging.DEBUG if req.monitor else logging.INFO,
            "Critic %s%s: raw=%r -> %s (infer %s ms, total %.0f ms, input %dx%d)",
            req.state,
            " [monitor]" if req.monitor else "",
            raw,
            verdict.value,
            "n/a" if infer_ms is None else f"{infer_ms:.0f}",
            latency_ms,
            image.shape[1],
            image.shape[0],
        )
        return CriticResult(
            req.request_id, req.state, verdict, raw, latency_ms, image, infer_ms, probs,
            req.monitor,
        )  # fmt: skip

    def stop(self, timeout: float = 1.0) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout)
