"""OPT-IN cloud Critic: Anthropic Messages API over stdlib urllib (no SDK, no pip deps).

Owner-approved exception to the edge-only rule (CLAUDE.md rule 1, 2026-10-11). Used by
``lbyr.demo --critic cloud`` (checkpoints only, never the continuous monitor) and by
``tools/autolabel.py`` (offline labelling). Never a default.

One request = the 378x378 Critic crop as a base64 JPEG image block + :data:`CLOUD_PROMPT`,
temperature 0, ``max_tokens`` 10, thinking disabled, ``CloudCriticConfig.timeout_s`` (15 s)
per HTTP attempt. Thinking MUST be off: claude-haiku-5-5 thinks by default and spends all
10 tokens on an empty thinking block (no text). Optional parameters a model rejects with a
400 naming them (claude-haiku-5-5: "`temperature` is deprecated for this model") are dropped:
the request is re-sent once at once without it and the client stops sending it (WARNING once).
429 / 5xx (incl. 529 overloaded) -> one retry after ``retry_delay_s`` (2 s). Any other
failure (network error, socket timeout, 4xx, second failure) -> Verdict.TIMEOUT, which the
demo treats like a local timeout (re-query once, then abort safely). The answer text goes
through the runtime :func:`parse_verdict`.

The API key is read from the environment (``CloudCriticConfig.api_key_env``) only. It is
never logged, never put in exception messages and never written to disk.

Cost: no VRAM, no local compute beyond one JPEG encode (~2 ms). Latency is network-bound
(typically ~1-3 s per call, worst case 2 x 15 s + 2 s), so the CriticConfig.timeout_s used
with this backend is raised to cover ``CloudCriticConfig.worst_case_s``.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from lbyr.ai.critic import CriticOutput, Verdict, parse_verdict
from lbyr.config import CloudCriticConfig
from lbyr.errors import ConfigError

log = logging.getLogger(__name__)

CLOUD_PROMPT = (
    "This is a robot wrist camera after the gripper closed. Look between the two black "
    "gripper fingers. Answer ONE word: GRASPED (object clearly held), SLIPPING (object partly "
    "held or tilting out), EMPTY (nothing between the fingers), OCCLUDED (fingers not visible)."
)

RETRY_STATUS = frozenset({429, 529})  # plus every 5xx


def is_retryable(status: int | None) -> bool:
    return status is not None and (status in RETRY_STATUS or 500 <= status <= 599)


@dataclass(frozen=True)
class CloudReply:
    """One labelled image. ``text`` is None when every attempt failed (``error`` says why)."""

    text: str | None
    verdict: Verdict  # parse_verdict(text); TIMEOUT when no answer
    latency_ms: float  # all attempts + the retry pause
    attempts: int
    status: int | None = None  # last HTTP status (None: no HTTP response)
    error: str | None = None


def api_key_from_env(cfg: CloudCriticConfig) -> str:
    key = os.environ.get(cfg.api_key_env, "").strip()
    if not key:
        raise ConfigError(f"cloud Critic: environment variable {cfg.api_key_env} is not set")
    return key


def encode_jpeg(image_bgr: np.ndarray, quality: int) -> bytes:
    """BGR uint8 -> JPEG bytes (cv2, lazily; Pillow fallback)."""
    try:
        import cv2  # lazy (JetPack build)

        ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise ValueError("cv2.imencode failed")
        return bytes(buf.tobytes())
    except ImportError:
        import io

        from lbyr.ai.imaging import bgr_to_pil

        out = io.BytesIO()
        bgr_to_pil(image_bgr).save(out, "JPEG", quality=quality)
        return out.getvalue()


def optional_params(cfg: CloudCriticConfig) -> dict[str, Any]:
    """Request fields a model may reject (then dropped, see :func:`rejects_param`)."""
    params: dict[str, Any] = {"temperature": cfg.temperature}
    if cfg.disable_thinking:
        params["thinking"] = {"type": "disabled"}
    return params


def build_request_body(
    cfg: CloudCriticConfig, jpeg: bytes, prompt: str, dropped: frozenset[str] = frozenset()
) -> dict[str, Any]:
    image = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.b64encode(jpeg).decode("ascii"),
        },
    }
    body: dict[str, Any] = {"model": cfg.model, "max_tokens": cfg.max_tokens}
    body.update({k: v for k, v in optional_params(cfg).items() if k not in dropped})
    body["messages"] = [{"role": "user", "content": [image, {"type": "text", "text": prompt}]}]
    return body


def rejects_param(status: int | None, error: str | None, name: str) -> bool:
    """A 400 that names an optional request field, e.g. "`temperature` is deprecated for
    this model" (claude-haiku-5-5, 2026-10-11)."""
    return status == 400 and error is not None and name in error


def response_text(payload: dict[str, Any]) -> str:
    """Concatenated text blocks of a Messages API response."""
    blocks = payload.get("content") or []
    return "".join(str(b.get("text", "")) for b in blocks if b.get("type") == "text").strip()


Urlopen = Callable[..., Any]  # urllib.request.urlopen-compatible (tests inject a fake)


class AnthropicCriticClient:
    """Thread-safe for sequential use (one request at a time per caller); holds no session."""

    def __init__(
        self,
        cfg: CloudCriticConfig,
        api_key: str | None = None,
        *,
        urlopen: Urlopen | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        cfg.validate()
        self.cfg = cfg
        self._key = api_key if api_key is not None else api_key_from_env(cfg)
        self._urlopen = urlopen or urllib.request.urlopen
        self._sleep = sleep
        # temperature 0 (rule 6) and thinking-off are sent unless the model rejects the field
        # (400 naming it); then it is dropped for the rest of the process (rejects_param)
        self.dropped: set[str] = set()

    def __repr__(self) -> str:  # never expose the key
        return f"AnthropicCriticClient(model={self.cfg.model!r})"

    def _redact(self, text: str) -> str:
        return text.replace(self._key, "***") if self._key else text

    def _post(self, body: bytes) -> tuple[int | None, str | None, str | None]:
        """One attempt -> (status, answer text, error)."""
        req = urllib.request.Request(
            self.cfg.url,
            data=body,
            method="POST",
            headers={
                "x-api-key": self._key,
                "anthropic-version": self.cfg.api_version,
                "content-type": "application/json",
            },
        )
        try:
            with self._urlopen(req, timeout=self.cfg.timeout_s) as resp:
                status = int(getattr(resp, "status", 200))
                payload = json.loads(resp.read().decode("utf-8"))
            return status, response_text(payload), None
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            return e.code, None, self._redact(f"HTTP {e.code}: {detail}".strip())
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            reason = getattr(e, "reason", e)
            return None, None, self._redact(f"{type(e).__name__}: {reason}")
        except (ValueError, KeyError, AttributeError) as e:  # bad JSON / unexpected shape
            return None, None, f"bad response: {type(e).__name__}"

    def _body(self, jpeg: bytes, prompt: str) -> bytes:
        payload = build_request_body(self.cfg, jpeg, prompt, frozenset(self.dropped))
        return json.dumps(payload).encode("utf-8")

    def _send(self, jpeg: bytes, prompt: str) -> tuple[int | None, str | None, str | None]:
        """One attempt; a 400 naming an optional field drops it and re-sends at once
        (not the retry). Each field is dropped at most once per process."""
        status, text, error = self._post(self._body(jpeg, prompt))
        while text is None:
            name = next(
                (n for n in optional_params(self.cfg)
                 if n not in self.dropped and rejects_param(status, error, n)),
                None,
            )  # fmt: skip
            if name is None:
                break
            self.dropped.add(name)
            log.warning("cloud Critic: %s rejects `%s`; sending without it from now on",
                        self.cfg.model, name)  # fmt: skip
            status, text, error = self._post(self._body(jpeg, prompt))
        return status, text, error

    def classify_jpeg(self, jpeg: bytes, prompt: str = CLOUD_PROMPT) -> CloudReply:
        t0 = time.perf_counter()
        status: int | None = None
        error: str | None = None
        attempts = 0
        for attempt in range(2):
            attempts = attempt + 1
            status, text, error = self._send(jpeg, prompt)
            if text is not None:
                ms = (time.perf_counter() - t0) * 1000
                return CloudReply(text, parse_verdict(text), ms, attempts, status)
            if attempt == 0 and is_retryable(status):
                log.warning("cloud Critic: %s -> retry in %.0f s", error, self.cfg.retry_delay_s)
                self._sleep(self.cfg.retry_delay_s)
                continue
            break
        ms = (time.perf_counter() - t0) * 1000
        log.warning("cloud Critic: no answer after %d attempt(s): %s -> TIMEOUT", attempts, error)
        return CloudReply(None, Verdict.TIMEOUT, ms, attempts, status, error)


class CloudCritic:
    """Critic backend (``classify``): crop -> JPEG -> Messages API -> parse_verdict.

    No probabilities (``probs`` None), so LOW_CONF never applies; failures come back as
    TIMEOUT (the demo's re-query path). ``model_label`` is shown on the web page.
    ``checkpoints_only``: the demo never runs the continuous LIFT/TRANSPORT monitor with it.
    """

    checkpoints_only = True

    def __init__(self, cfg: CloudCriticConfig, client: AnthropicCriticClient | None = None) -> None:
        cfg.validate()
        self.cfg = cfg
        self.client = client or AnthropicCriticClient(cfg)
        self.model_label = cfg.model

    def classify(self, image_bgr: np.ndarray, state: str) -> CriticOutput:
        reply = self.client.classify_jpeg(encode_jpeg(image_bgr, self.cfg.jpeg_quality))
        raw = reply.text if reply.text is not None else f"[cloud: {reply.error}]"
        return CriticOutput(reply.verdict, raw, None)

    def infer(self, image_bgr: np.ndarray, prompt: str) -> str:
        """CriticBackend compatibility (``prompt`` is unused: the cloud prompt is fixed)."""
        out = self.classify(image_bgr, "infer")
        return out.raw_text

    def check(self, side_px: int = 378) -> CloudReply:
        """Startup check: one request on a grey square. A 4xx other than 429 (bad key,
        unknown model, rejected request) -> ConfigError (setup refused); any other failure
        (network, 429 / 5xx twice) is returned for a warning."""
        grey = np.full((side_px, side_px, 3), 128, dtype=np.uint8)
        reply = self.client.classify_jpeg(encode_jpeg(grey, self.cfg.jpeg_quality))
        if reply.text is None and reply.status is not None and 400 <= reply.status < 500:
            if reply.status != 429:
                raise ConfigError(f"cloud Critic refused by the API ({reply.error})")
        return reply
