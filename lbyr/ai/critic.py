"""Async visual grasp-verification Critic (bible 11.2).

Event-driven: the FSM submits a request (non-blocking) and polls the verdict queue
(non-blocking). Inference runs on the CriticWorker thread under an explicit timeout.
Moondream2 cost (bible 9.2): ~4 GB FP16 / ~1.2 GB GGUF-Q4 VRAM, 0.3-0.8 s per query.
Measured on the Orin NX: see ``tools/bench_models.py`` and the README.
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
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from lbyr.ai.cuda_mem import log_cuda_allocated
from lbyr.ai.imaging import bgr_to_pil
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
    """Moondream2 FP16 on CUDA (bible 11.2; ~4 GB VRAM, see ``tools/bench_models.py``).

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
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
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
