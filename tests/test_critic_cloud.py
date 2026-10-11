"""Cloud Critic (Anthropic Messages API over urllib): request shape, retry, TIMEOUT mapping,
key hygiene, and the demo wiring (checkpoints only). No network: urlopen is faked."""

from __future__ import annotations

import base64
import io
import json
import logging
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lbyr.ai.critic import CriticWorker, Verdict
from lbyr.ai.critic_cloud import (
    CLOUD_PROMPT,
    AnthropicCriticClient,
    CloudCritic,
)
from lbyr.config import CloudCriticConfig, CriticConfig
from lbyr.errors import ConfigError

KEY = "sk-ant-test-SECRET-123"


class _Resp:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self._b = json.dumps(payload).encode()
        self.status = status

    def read(self) -> bytes:
        return self._b

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *a: object) -> None:
        return None


def _http_error(code: int, body: str = "") -> urllib.error.HTTPError:
    return urllib.error.HTTPError("u", code, "err", {}, io.BytesIO(body.encode()))  # type: ignore[arg-type]


def _text(t: str) -> _Resp:
    return _Resp({"content": [{"type": "text", "text": t}], "stop_reason": "end_turn"})


class FakeUrlopen:
    """Each call pops the next item: a _Resp is returned, an exception is raised."""

    def __init__(self, items: list[Any]) -> None:
        self.items = list(items)
        self.requests: list[tuple[urllib.request.Request, float]] = []

    def __call__(self, req: urllib.request.Request, timeout: float) -> Any:
        self.requests.append((req, timeout))
        item = self.items.pop(0) if len(self.items) > 1 else self.items[0]
        if isinstance(item, BaseException):
            raise item
        return item


def _client(items: list[Any]) -> tuple[AnthropicCriticClient, FakeUrlopen, list[float]]:
    fake, sleeps = FakeUrlopen(items), []
    c = AnthropicCriticClient(CloudCriticConfig(), KEY, urlopen=fake, sleep=sleeps.append)
    return c, fake, sleeps


def test_request_shape_and_parse() -> None:
    c, fake, sleeps = _client([_text("GRASPED.")])
    r = c.classify_jpeg(b"\xff\xd8jpeg")
    assert r.verdict is Verdict.GRASPED and r.attempts == 1 and not sleeps
    req, timeout = fake.requests[0]
    assert timeout == 15.0 and req.full_url == "https://api.anthropic.com/v1/messages"
    assert req.get_header("X-api-key") == KEY and req.get_header("Anthropic-version")
    assert req.data is not None
    body = json.loads(req.data)  # type: ignore[arg-type]
    assert body["model"] == "claude-haiku-5-5" and body["temperature"] == 0.0
    assert body["max_tokens"] == 10 and body["thinking"] == {"type": "disabled"}
    img, txt = body["messages"][0]["content"]
    assert img["source"]["media_type"] == "image/jpeg"
    assert base64.b64decode(img["source"]["data"]) == b"\xff\xd8jpeg"
    assert txt == {"type": "text", "text": CLOUD_PROMPT}


def test_temperature_rejected_resends_without_it_once() -> None:
    msg = '{"error":{"message":"`temperature` is deprecated for this model."}}'
    c, fake, sleeps = _client([_http_error(400, msg), _text("GRASPED"), _text("EMPTY")])
    assert c.classify_jpeg(b"x").verdict is Verdict.GRASPED and not sleeps
    assert c.classify_jpeg(b"x").verdict is Verdict.EMPTY
    sent = [json.loads(r.data) for r, _ in fake.requests]  # type: ignore[arg-type]
    assert ["temperature" in b for b in sent] == [True, False, False]
    assert all(b["thinking"] == {"type": "disabled"} for b in sent)  # still sent
    assert c.dropped == {"temperature"}


def test_other_400_is_not_resent() -> None:
    c, fake, _ = _client([_http_error(400, "max_tokens too small"), _text("GRASPED")])
    assert c.classify_jpeg(b"x").verdict is Verdict.TIMEOUT and len(fake.requests) == 1
    assert not c.dropped


def test_unparseable_answer_is_invalid() -> None:
    c, _, _ = _client([_text("The object is grasped")])
    assert c.classify_jpeg(b"x").verdict is Verdict.INVALID


@pytest.mark.parametrize("code", [429, 500, 503, 529])
def test_retry_once_on_retryable_then_success(code: int) -> None:
    c, fake, sleeps = _client([_http_error(code), _text("EMPTY")])
    r = c.classify_jpeg(b"x")
    assert r.verdict is Verdict.EMPTY and r.attempts == 2 and sleeps == [2.0]
    assert len(fake.requests) == 2


def test_retryable_twice_is_timeout() -> None:
    c, fake, sleeps = _client([_http_error(529), _http_error(529)])
    r = c.classify_jpeg(b"x")
    assert r.verdict is Verdict.TIMEOUT and r.text is None and r.attempts == 2
    assert r.status == 529 and sleeps == [2.0] and len(fake.requests) == 2


@pytest.mark.parametrize(
    "exc", [_http_error(400), _http_error(401), urllib.error.URLError("down"), TimeoutError()]
)
def test_non_retryable_failure_is_timeout_without_retry(exc: BaseException) -> None:
    c, fake, sleeps = _client([exc, _text("GRASPED")])
    r = c.classify_jpeg(b"x")
    assert r.verdict is Verdict.TIMEOUT and r.attempts == 1 and not sleeps
    assert len(fake.requests) == 1


def test_key_never_leaks(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    c, _, _ = _client([_http_error(401, f'{{"error": "bad key {KEY}"}}')])
    r = c.classify_jpeg(b"x")
    assert r.error is not None and KEY not in r.error and "***" in r.error
    assert KEY not in repr(c) and KEY not in caplog.text


def test_missing_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        AnthropicCriticClient(CloudCriticConfig())


def test_config_validate() -> None:
    with pytest.raises(ConfigError):
        CloudCriticConfig(temperature=0.5).validate()
    assert CloudCriticConfig().worst_case_s == 32.0


def test_worker_maps_failure_to_timeout_and_passes_text() -> None:
    c, _, _ = _client([_text("SLIPPING"), urllib.error.URLError("down")])
    backend = CloudCritic(CloudCriticConfig(), c)
    w = CriticWorker(backend, replace(CriticConfig(), timeout_s=5.0))
    w.start()
    try:
        img = np.zeros((480, 640, 3), np.uint8)
        got = []
        for state in ("VERIFY_GRASP", "VERIFY_HOLD"):
            w.submit(img, state)
            while (r := w.poll()) is None:
                pass
            got.append(r)
    finally:
        w.stop()
    assert got[0].verdict is Verdict.SLIPPING and got[0].raw_text == "SLIPPING"
    assert got[0].probs is None
    assert got[1].verdict is Verdict.TIMEOUT and "URLError" in (got[1].raw_text or "")


def test_check_refuses_bad_key_or_model() -> None:
    c, _, _ = _client([_http_error(404, "model not found")])
    with pytest.raises(ConfigError, match="refused"):
        CloudCritic(CloudCriticConfig(), c).check()
    c, _, _ = _client([urllib.error.URLError("offline")])
    assert CloudCritic(CloudCriticConfig(), c).check().text is None  # warning only


# ---------------------------------------------------------------- demo wiring (dry run)
def _dry_cloud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, items: list[Any]
) -> tuple[int, list[dict[str, Any]], FakeUrlopen, list[str]]:
    from lbyr import demo

    fake = FakeUrlopen(items)
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    fast: Callable[..., CloudCriticConfig] = lambda **kw: CloudCriticConfig(  # noqa: E731
        retry_delay_s=0.01, **kw
    )
    monkeypatch.setattr(demo, "CloudCriticConfig", fast)
    modes: list[str] = []
    orig_set = demo.DemoStatus.set

    def spy(self: Any, **kw: Any) -> None:
        if "mode" in kw:
            modes.append(kw["mode"])
        orig_set(self, **kw)

    monkeypatch.setattr(demo.DemoStatus, "set", spy)
    argv = ["--dry-run", "--critic", "cloud", "--log-dir", str(tmp_path), "--http-port", "0"]
    rc = demo.main(argv)
    jsonl = next(tmp_path.glob("*/episodes.jsonl")).read_text()
    assert KEY not in jsonl
    return rc, [json.loads(x) for x in jsonl.splitlines()], fake, modes


def test_demo_cloud_checkpoints_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # startup check, VERIFY_GRASP (529 -> retry -> GRASPED), VERIFY_HOLD, VERIFY_PLACE
    items = [_text("EMPTY"), _http_error(529), _text("GRASPED"), _text("GRASPED"), _text("EMPTY")]
    rc, recs, fake, modes = _dry_cloud(tmp_path, monkeypatch, items)
    assert rc == 0 and recs[-1]["outcome"] == "success"
    v = [r for r in recs if r["event"] == "verdict"]
    assert [r["verdict"] for r in v] == ["GRASPED", "GRASPED", "EMPTY"]
    assert not any(r["monitor"] for r in v)
    assert not any(r["event"] == "monitor_start" for r in recs)  # no continuous monitor
    assert len(fake.requests) == 5
    assert any("critic=cloud (claude-haiku-5-5)" in m for m in modes)


def test_demo_cloud_failure_requeries_then_aborts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rc, recs, _, _ = _dry_cloud(tmp_path, monkeypatch, [_text("EMPTY"), _http_error(400)])
    assert rc == 1
    v = [r["verdict"] for r in recs if r["event"] == "verdict"]
    assert v == ["TIMEOUT", "TIMEOUT"]  # re-queried once
    assert recs[-1]["result"].startswith("FAILED (TIMEOUT at VERIFY_GRASP)")
