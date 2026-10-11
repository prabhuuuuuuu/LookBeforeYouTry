"""Linear-probe Critic: Moondream2 vision encoder + 4-class logistic head (no text decode).

Trained offline by ``tools/train_critic_probe.py`` (file ``DemoConfig.probe_path``,
default ``~/models/critic_probe.pt``). Runtime input is the same as the zero-shot Critic:
dewarped wrist frame -> ``critic_crop`` 378x378 (done by the CriticWorker). The embedding
function below is shared with the training tool; the probe file records its name and is
refused if it does not match.

Cost (Orin NX, bf16): ~4.1 GB (Moondream2 weights, same as the zero-shot Critic; the text
model is loaded but unused), ~86 ms per call (vision encoder on one 378 tile + a
4x1152 head on the CPU) vs ~0.85 s for a zero-shot query.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lbyr.ai.critic import MODEL_CLASSES, CriticOutput, Verdict, _load_moondream2
from lbyr.ai.cuda_mem import log_cuda_allocated
from lbyr.ai.imaging import bgr_to_pil
from lbyr.ai.timeout import TimeoutResult, run_with_timeout
from lbyr.config import CriticConfig
from lbyr.errors import ConfigError, LBYRError

log = logging.getLogger(__name__)

EMBED_FN_NAME = "moondream2_vis_enc_global378_meanpool729_v1"
CLASS_NAMES = tuple(v.value for v in MODEL_CLASSES)


def make_embed_fn(model: Any) -> Callable[[np.ndarray], Any]:
    """BGR image -> float32 (1152,) CPU tensor (``EMBED_FN_NAME``; change both together).

    Moondream's own ``prepare_crops`` (same normalization as ``query``), then ``_vis_enc``
    on the GLOBAL crop only (index 0: the image resized to 378x378; for a 378x378 input
    the input itself), then the mean over the 729 patch tokens of the post-LN output.
    """
    import torch

    inner = model.model
    prepare_crops = sys.modules[type(inner).__module__].prepare_crops

    def embed(image_bgr: np.ndarray) -> Any:
        with torch.inference_mode():
            crops, _ = prepare_crops(bgr_to_pil(image_bgr), inner.config.vision, inner.device)
            feats = inner._vis_enc(crops[:1])  # (1, 729, 1152)
            return feats[0].float().mean(dim=0).cpu()

    return embed


@dataclass
class ProbeHead:
    """logits = ((e - mean) / std) @ weights.T + bias; softmax over ``classes``."""

    weights: Any  # torch (C, D)
    bias: Any  # (C,)
    mean: Any  # (D,)
    std: Any  # (D,)
    classes: tuple[str, ...]
    crop: tuple[float, float, float, float] | None
    embed_fn_name: str

    @classmethod
    def load(cls, path: Path) -> ProbeHead:
        import torch

        if not path.is_file():
            raise ConfigError(f"Critic probe {path} not found (tools/train_critic_probe.py)")
        d = torch.load(path, map_location="cpu", weights_only=True)
        missing = [k for k in ("weights", "bias", "mean", "std", "classes") if k not in d]
        if missing:
            raise LBYRError(f"{path}: not a Critic probe file (missing {missing})")
        head = cls(
            d["weights"].float(),
            d["bias"].float(),
            d["mean"].float(),
            d["std"].float(),
            tuple(d["classes"]),
            tuple(d["crop"]) if d.get("crop") is not None else None,
            str(d.get("embed_fn_name", "")),
        )
        head.check()
        return head

    def check(self) -> None:
        if self.embed_fn_name != EMBED_FN_NAME:
            raise LBYRError(
                f"probe embedding {self.embed_fn_name!r} != runtime {EMBED_FN_NAME!r}: retrain"
            )
        if self.classes != CLASS_NAMES:
            raise LBYRError(f"probe classes {self.classes} != {CLASS_NAMES}")
        c, dim = self.weights.shape
        if c != len(CLASS_NAMES) or self.bias.shape != (c,) or self.mean.shape != (dim,):
            raise LBYRError(f"probe shapes inconsistent: W {tuple(self.weights.shape)}")

    def probs(self, emb: Any) -> dict[str, float]:
        import torch

        z = ((emb - self.mean) / self.std) @ self.weights.T + self.bias
        p = torch.softmax(z, dim=-1)
        return {c: float(p[i]) for i, c in enumerate(self.classes)}


class ProbeCritic:
    """Critic backend: ``classify`` -> verdict (argmax) + all 4 probabilities.

    Moondream2 is loaded ONCE (``cfg.model_path``, ``cfg.dtype``) and warmed up under
    ``cfg.warmup_timeout_s``. ``model`` / ``head`` inject prebuilt objects (tests).
    LOW_CONF (max prob < ``cfg.low_conf_threshold``) is applied by the CriticWorker.
    """

    def __init__(
        self,
        cfg: CriticConfig,
        probe_path: Path | None = None,
        *,
        model: Any | None = None,
        head: ProbeHead | None = None,
        embed: Callable[[np.ndarray], Any] | None = None,
    ) -> None:
        cfg.validate()
        self.cfg = cfg
        if head is None:
            if probe_path is None:
                raise ConfigError("ProbeCritic needs probe_path (DemoConfig.probe_path)")
            head = ProbeHead.load(probe_path.expanduser())
        self.head = head
        self.trained_crop = head.crop
        self._lock = threading.Lock()
        if embed is None:
            if model is None:
                if cfg.model_path is None:
                    raise ConfigError("CriticConfig.model_path is not set (local weights only)")
                t0 = time.perf_counter()
                model = _load_moondream2(cfg.model_path, cfg.device, cfg.dtype)
                log.info("Moondream2 (%s) for the probe loaded in %.1f s", cfg.dtype,
                         time.perf_counter() - t0)  # fmt: skip
                log_cuda_allocated("Moondream2 probe Critic")
            embed = make_embed_fn(model)
        self._embed = embed
        self._warm_up()

    def _warm_up(self) -> None:
        side = self.cfg.crop_size_px
        dummy = np.zeros((side, side, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        out = run_with_timeout(
            lambda: self.classify(dummy, "warmup"), self.cfg.warmup_timeout_s, "probe-warmup"
        )
        if isinstance(out, TimeoutResult):
            raise LBYRError(f"probe Critic warm-up exceeded {self.cfg.warmup_timeout_s:.0f} s")
        log.info("probe Critic warm-up: %s in %.0f ms", out.raw_text,
                 (time.perf_counter() - t0) * 1000)  # fmt: skip

    def classify(self, image_bgr: np.ndarray, state: str) -> CriticOutput:
        with self._lock:
            emb = self._embed(image_bgr)
        probs = self.head.probs(emb)
        top = max(probs, key=lambda k: probs[k])
        return CriticOutput(Verdict(top), f"probe {top} {probs[top]:.2f}", probs)

    def infer(self, image_bgr: np.ndarray, prompt: str) -> str:
        """CriticBackend compatibility: the argmax class word (``prompt`` is unused)."""
        return self.classify(image_bgr, "infer").verdict.value
