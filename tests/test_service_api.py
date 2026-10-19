"""End-to-end tests of the HTTP API and service with the fake worker (no GPU needed)."""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from edgeserve import settings as limits
from edgeserve.api.app import _rejection_reason, _wait_final, create_app
from edgeserve.scheduler import AdmissionError
from edgeserve.service import Service
from edgeserve.settings import Settings

MESSAGES = [{"role": "system", "content": "Answer briefly."}, {"role": "user", "content": "say hello there"}]
CHAT = "/v1/chat/completions"
COMPLETIONS = "/v1/completions"
JSON = {"content-type": "application/json"}


def make_settings(tmp_path: Path, model: Path, native: Path, **overrides) -> Settings:
    values = dict(model=model, native_dir=native, cache_dir=tmp_path / "cache", output_dir=tmp_path / "out",
                  max_input_len=64, max_new_tokens=6, max_batch_size=1)
    values.update(overrides)
    return Settings(**values)


@contextlib.asynccontextmanager
async def serving(settings: Settings, expect_ready: bool = True):
    service = Service(settings)
    app = create_app(service)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as client:
            for _ in range(400):
                if service.state in ("ready", "failed"):
                    break
                await asyncio.sleep(0.05)
            if expect_ready:
                assert service.state == "ready", service.detail
            yield app, client, service


@contextlib.asynccontextmanager
async def running(settings: Settings, expect_ready: bool = True):
    async with serving(settings, expect_ready) as (_, client, service):
        yield client, service


async def wait_for_file(path: Path, timeout: float = 5.0) -> None:
    for _ in range(int(timeout / 0.05)):
        if path.exists():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"{path} was not written")


async def read_record(tmp_path: Path, custom_id: str) -> dict:
    path = tmp_path / "out" / "records" / f"{custom_id}.json"
    await wait_for_file(path)
    return json.loads(path.read_text())


async def eventually(condition, timeout: float = 5.0) -> None:
    for _ in range(int(timeout / 0.02)):
        if condition():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition not reached in time")


class SendFailed(Exception):
    """What uvicorn raises when h11 refuses to write a response head (e.g. an illegal header value)."""


class ASGIRequest:
    """One request driven straight through the app the way uvicorn's h11 server does it (ASGI spec 2.3):
    receive() hands over the body, then blocks until the client goes away (``gone``), after which send() is a
    no-op. httpx's ASGITransport cannot disconnect mid-request, hence this."""

    def __init__(self, app, path: str, payload: dict, headers: dict[str, str] | None = None,
                 fail_start: bool = False) -> None:
        self.app = app
        self.body = json.dumps(payload).encode()
        fields = {**JSON, "content-length": str(len(self.body)), **(headers or {})}
        self.scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
                      "method": "POST", "scheme": "http", "path": path, "raw_path": path.encode(),
                      "query_string": b"", "root_path": "", "client": ("127.0.0.1", 50000),
                      "server": ("test", 80), "headers": [(k.lower().encode(), v.encode()) for k, v in fields.items()]}
        self.fail_start = fail_start
        self.body_sent = False
        self.gone = asyncio.Event()
        self.status: int | None = None
        self.received = b""

    async def receive(self) -> dict:
        if not self.body_sent:
            self.body_sent = True
            return {"type": "http.request", "body": self.body, "more_body": False}
        await self.gone.wait()
        return {"type": "http.disconnect"}

    async def send(self, message: dict) -> None:
        if self.gone.is_set():
            return
        if message["type"] == "http.response.start":
            if self.fail_start:
                raise SendFailed("Illegal header value")
            self.status = message["status"]
        else:
            self.received += message.get("body", b"")

    async def run(self) -> None:
        await self.app(self.scope, self.receive, self.send)


def tamper_done(service: Service, monkeypatch, **changes) -> None:
    """Edit every result of the worker's done events before the scheduler sees them."""
    dispatch = service.worker._dispatch

    def tampered(event: dict) -> None:
        if event.get("ev") == "done":
            for result in event.get("results", []):
                result.update(changes)
        dispatch(event)

    monkeypatch.setattr(service.worker, "_dispatch", tampered)


async def test_chat_completion_keeps_client_id_and_writes_record(tmp_path, tiny_model, fake_native):
    async with running(make_settings(tmp_path, tiny_model, fake_native)) as (client, service):
        r = await client.post("/v1/chat/completions", json={"messages": MESSAGES, "metadata": {"task": "demo"}},
                              headers={"X-Request-Id": "seg-001"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert r.headers["x-request-id"] == "seg-001"
        assert body["id"] == "chatcmpl-seg-001" and body["request_id"] == "seg-001"
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"].startswith("tok0")
        assert body["usage"]["completion_tokens"] == 6
        assert body["metrics"]["ttft_ms"] > 0 and body["metrics"]["tpot_ms"] > 0
        assert body["system_fingerprint"].startswith("fp_")

        record_file = tmp_path / "out" / "records" / "seg-001.json"
        await wait_for_file(record_file)
        record = json.loads(record_file.read_text())
        assert record["custom_id"] == "seg-001" and record["status"] == "completed"
        assert record["metadata"] == {"task": "demo"}
        steps = record["steps"]
        assert steps["count"] == 6 and len(steps["t_ms"]) == 6 and steps["itl_ms"][0] is None
        assert steps["flops"][0] > steps["flops"][1] > 0  # prefill costs more than one decode step
        assert record["timestamps_ns"]["arrival"] <= record["timestamps_ns"]["first_token"]

        r = await client.get("/v1/records/seg-001")
        assert r.status_code == 200 and r.json()["attempt_id"] == record["attempt_id"]
        assert (await client.get("/v1/records/missing")).status_code == 404


async def test_streaming_sends_chunks_usage_and_done(tmp_path, tiny_model, fake_native):
    async with running(make_settings(tmp_path, tiny_model, fake_native)) as (client, _):
        payload = {"messages": MESSAGES, "stream": True, "stream_options": {"include_usage": True},
                   "request_id": "stream-1"}
        async with client.stream("POST", "/v1/chat/completions", json=payload) as r:
            assert r.status_code == 200
            assert r.headers["x-request-id"] == "stream-1"
            lines = [line async for line in r.aiter_lines() if line.startswith("data: ")]
        assert lines[-1] == "data: [DONE]"
        chunks = [json.loads(line[6:]) for line in lines[:-1]]
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
        text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
        assert text.startswith("tok0")
        assert chunks[-2]["choices"][0]["finish_reason"] in ("stop", "length")
        assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] == 6
        assert "metrics" in chunks[-1]


async def test_completions_endpoint_and_return_record(tmp_path, tiny_model, fake_native):
    async with running(make_settings(tmp_path, tiny_model, fake_native)) as (client, _):
        r = await client.post("/v1/completions", json={"prompt": "<s>hello world", "max_tokens": 3,
                                                        "return_record": True})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["object"] == "text_completion" and body["id"].startswith("cmpl-req_")
        assert body["usage"]["completion_tokens"] == 3
        assert body["record"]["steps"]["count"] == 3
        assert body["record"]["request"]["params"]["max_tokens"] == 3


async def test_admission_errors(tmp_path, tiny_model, fake_native):
    async with running(make_settings(tmp_path, tiny_model, fake_native)) as (client, _):
        long_text = " ".join(["word"] * 100)  # fake count = words + 10 > max_input_len 64
        r = await client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": long_text}]})
        assert r.status_code == 400 and r.json()["error"]["code"] == "context_length_exceeded"

        r = await client.post("/v1/chat/completions", json={"messages": MESSAGES, "max_tokens": 10_000})
        assert r.status_code == 400 and r.json()["error"]["param"] == "max_tokens"

        r = await client.post("/v1/chat/completions", json={"messages": MESSAGES, "tools": []})
        assert r.status_code == 400 and r.json()["error"]["code"] == "unsupported_parameter"

        r = await client.post("/v1/chat/completions", json={"messages": MESSAGES}, headers={"X-Request-Id": "bad\x01"})
        assert r.status_code in (400, 422)

        r = await client.post("/v1/chat/completions", json={"messages": MESSAGES, "model": "gpt-4"})
        assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"

        r = await client.post("/v1/chat/completions", content=b"{not json")
        assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_json"


async def test_api_key_required_when_configured(tmp_path, tiny_model, fake_native):
    settings = make_settings(tmp_path, tiny_model, fake_native, api_key="secret-key")
    async with running(settings) as (client, _):
        assert (await client.get("/v1/models")).status_code == 401
        r = await client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401
        r = await client.get("/v1/models", headers={"Authorization": "Bearer secret-key"})
        assert r.status_code == 200
        model = r.json()["data"][0]
        assert model["id"] == "tiny_model" and model["max_batch_size"] == 1 and model["ready"] is True
        assert (await client.get("/health")).status_code == 200  # probes stay open
        assert (await client.get("/metrics")).status_code == 200


async def test_duplicate_inflight_id_and_queue_full(tmp_path, tiny_model, fake_native, monkeypatch):
    monkeypatch.setenv("FAKE_STEP_MS", "150")
    monkeypatch.setattr(limits, "QUEUE_SIZE", 1)
    async with running(make_settings(tmp_path, tiny_model, fake_native)) as (client, _):
        first = asyncio.create_task(client.post("/v1/chat/completions", json={"messages": MESSAGES},
                                                headers={"X-Request-Id": "dup"}))
        await asyncio.sleep(0.2)
        r = await client.post("/v1/chat/completions", json={"messages": MESSAGES}, headers={"X-Request-Id": "dup"})
        assert r.status_code == 409 and r.json()["error"]["code"] == "duplicate_request_id"
        assert r.headers["x-should-retry"] == "false"  # else the OpenAI SDK silently runs the request again
        second = asyncio.create_task(client.post("/v1/chat/completions", json={"messages": MESSAGES},
                                                 headers={"X-Request-Id": "queued"}))
        await asyncio.sleep(0.2)
        r = await client.post("/v1/chat/completions", json={"messages": MESSAGES}, headers={"X-Request-Id": "third"})
        assert r.status_code == 503 and "retry-after" in r.headers
        assert (await first).status_code == 200 and (await second).status_code == 200


async def test_worker_crash_fails_request_then_recovers(tmp_path, tiny_model, fake_native, monkeypatch):
    monkeypatch.setenv("FAKE_CRASH_FILE", str(tmp_path / "crashed-once"))
    async with running(make_settings(tmp_path, tiny_model, fake_native)) as (client, service):
        r = await client.post("/v1/chat/completions", json={"messages": MESSAGES}, headers={"X-Request-Id": "c1"})
        assert r.status_code == 503 and r.json()["error"]["code"] == "worker_restarted"
        for _ in range(200):
            if service.state == "ready":
                break
            await asyncio.sleep(0.05)
        assert service.state == "ready"
        r = await client.post("/v1/chat/completions", json={"messages": MESSAGES}, headers={"X-Request-Id": "c2"})
        assert r.status_code == 200
        record = tmp_path / "out" / "records" / "c1.json"
        await wait_for_file(record)
        assert json.loads(record.read_text())["status"] == "failed"


async def test_batching_groups_compatible_requests(tmp_path, tiny_model, fake_native, monkeypatch):
    monkeypatch.setenv("FAKE_STEP_MS", "60")
    settings = make_settings(tmp_path, tiny_model, fake_native, max_batch_size=2)
    async with running(settings) as (client, _):
        blocker = asyncio.create_task(client.post("/v1/chat/completions", json={"messages": MESSAGES},
                                                  headers={"X-Request-Id": "b0"}))
        await asyncio.sleep(0.1)  # b0 runs alone; b1 and b2 queue up and then share a batch
        rest = [asyncio.create_task(client.post("/v1/chat/completions", json={"messages": MESSAGES,
                                                                              "return_record": True},
                                                headers={"X-Request-Id": f"b{i}"})) for i in (1, 2)]
        responses = [await t for t in rest]
        assert (await blocker).status_code == 200
        sizes = [r.json()["record"]["metrics"]["batch_size"] for r in responses]
        assert sizes == [2, 2]


async def test_metrics_endpoint_exposes_vllm_style_names(tmp_path, tiny_model, fake_native):
    async with running(make_settings(tmp_path, tiny_model, fake_native)) as (client, _):
        await client.post("/v1/chat/completions", json={"messages": MESSAGES})
        text = (await client.get("/metrics")).text
        for name in ("edgeserve_time_to_first_token_seconds_bucket", "edgeserve_inter_token_latency_seconds_count",
                     "edgeserve_e2e_request_latency_seconds_sum", "edgeserve_generation_tokens_total",
                     "edgeserve_num_requests_running", 'edgeserve_server_state{model_name="tiny_model",state="ready"} 1.0'):
            assert name in text, name


async def test_engine_is_cached_between_starts(tmp_path, tiny_model, fake_native):
    settings = make_settings(tmp_path, tiny_model, fake_native)
    async with running(settings) as (_, service):
        assert service.engine.built_now is True
    async with running(settings) as (_, service):
        assert service.engine.built_now is False
    async with running(make_settings(tmp_path, tiny_model, fake_native, max_input_len=48)) as (_, service):
        assert service.engine.built_now is True  # different limits -> different engine


async def test_failed_engine_build_reports_failed_state(tmp_path, tiny_model, fake_native, monkeypatch):
    monkeypatch.setenv("FAKE_LLM_BUILD_FAIL", "1")
    async with running(make_settings(tmp_path, tiny_model, fake_native), expect_ready=False) as (client, service):
        assert service.state == "failed" and "out of memory" in service.detail
        r = await client.get("/health")
        assert r.status_code == 503 and r.json()["state"] == "failed"
        assert (await client.get("/health/ready")).status_code == 503
        for payload in ({"messages": MESSAGES}, {"messages": MESSAGES, "stream": True}):
            r = await client.post("/v1/chat/completions", json=payload)
            assert r.status_code == 503 and r.headers["content-type"] == "application/json"  # not an event stream
            assert r.json()["error"]["code"] == "server_unavailable" and int(r.headers["retry-after"]) > 0


async def test_non_loopback_without_key_is_refused(tmp_path, tiny_model, fake_native):
    with pytest.raises(ValueError, match="API key"):
        make_settings(tmp_path, tiny_model, fake_native, host="0.0.0.0").validate()


# ---------------------------------------------------------------------------- request bodies

async def test_oversized_bodies_are_refused_without_buffering_them(tmp_path, tiny_model, fake_native, monkeypatch):
    monkeypatch.setattr(limits, "MAX_BODY_BYTES", 4096)
    async with running(make_settings(tmp_path, tiny_model, fake_native)) as (client, _):
        r = await client.post(CHAT, content=b" " * 5000, headers=JSON)
        assert r.status_code == 413 and r.json()["error"]["code"] == "body_too_large"
        r = await client.post(CHAT, content=b" " * 5000, headers=JSON | {"content-length": "10"})
        assert r.status_code == 413
        pulled = 0

        async def chunked():  # no Content-Length at all
            nonlocal pulled
            for _ in range(1000):
                pulled += 1
                yield b" " * 1024

        r = await client.post(CHAT, content=chunked(), headers=JSON)
        assert r.status_code == 413
        assert pulled <= 6  # refused right after the limit, not after buffering the whole upload


async def test_malformed_json_gets_400_not_500(tmp_path, tiny_model, fake_native):
    async with running(make_settings(tmp_path, tiny_model, fake_native)) as (client, _):
        user = b'[{"role": "user", "content": "hi"}]'
        for raw in (b'{"messages": %s, "temperature": NaN}' % user, b'{"messages": %s, "top_p": -Infinity}' % user,
                    b'{"messages": ' + b"[" * 100_000 + b"]" * 100_000 + b"}", b"[1, 2]", b"null", b"", b"\xff\xfe"):
            r = await client.post(CHAT, content=raw, headers=JSON)
            assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_json", raw[:40]
        # a number too long to print in the context-length message used to crash the request
        r = await client.post(CHAT, content=b'{"messages": %s, "max_tokens": %s}' % (user, b"9" * 4300), headers=JSON)
        assert r.status_code == 400 and r.json()["error"]["param"] == "max_tokens"
        r = await client.post(CHAT, json={"messages": MESSAGES, "n": None, "stream": None, "stop": None})
        assert r.status_code == 200, r.text  # OpenAI allows null for these


async def test_upload_time_is_not_counted_as_server_latency(tmp_path, tiny_model, fake_native):
    payload = json.dumps({"messages": MESSAGES}).encode()

    async def slow_upload():
        yield payload[:10]
        await asyncio.sleep(0.8)
        yield payload[10:]

    async with running(make_settings(tmp_path, tiny_model, fake_native)) as (client, _):
        r = await client.post(CHAT, content=slow_upload(), headers=JSON)
        assert r.status_code == 200, r.text
        assert r.json()["metrics"]["ttft_ms"] < 500  # TTFT starts once the body has arrived (docs/METRICS.md)


async def test_metadata_without_a_utf8_form_is_refused_before_any_work(tmp_path, tiny_model, fake_native):
    # A lone surrogate (a JSON "\ud83d" escape, e.g. a JavaScript string cut in the middle of an emoji) used to pass:
    # the generation ran, then return_record and every later GET /v1/records for that ID failed with a 500.
    note = "café \U0001f600"
    async with running(make_settings(tmp_path, tiny_model, fake_native)) as (client, _):
        raw = b'{"messages": [{"role": "user", "content": "hi"}], "metadata": {"note": "\\ud83d"}, "return_record": true}'
        r = await client.post(CHAT, content=raw, headers=JSON | {"X-Request-Id": "lone"})
        assert r.status_code == 400 and r.json()["error"]["param"] == "metadata"
        r = await client.post(CHAT, json={"messages": MESSAGES, "metadata": {"note": note}, "return_record": True},
                              headers={"X-Request-Id": "emoji"})
        assert r.status_code == 200 and r.json()["record"]["metadata"] == {"note": note}
        await wait_for_file(tmp_path / "out" / "records" / "emoji.json")
        r = await client.get("/v1/records/emoji")
        assert r.status_code == 200 and r.json()["metadata"] == {"note": note}
    assert not (tmp_path / "out" / "records" / "lone.json").exists()


# ---------------------------------------------------------------------------- IDs and access

async def test_request_ids_that_http_cannot_echo_are_refused(tmp_path, tiny_model, fake_native):
    async with running(make_settings(tmp_path, tiny_model, fake_native)) as (client, _):
        for value in (" lead", "trail ", " ", "tab\tid", "café"):
            r = await client.post(CHAT, json={"messages": MESSAGES, "request_id": value})
            assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request_id", value
            assert "x-request-id" not in r.headers
        r = await client.post(CHAT, json={"messages": MESSAGES, "request_id": "inner space ok"})
        assert r.status_code == 200 and r.headers["x-request-id"] == "inner space ok"
        assert (await client.get("/v1/records/%20lead")).status_code == 400


async def test_every_v1_route_needs_the_key_and_records_stay_in_their_folder(tmp_path, tiny_model, fake_native):
    settings = make_settings(tmp_path, tiny_model, fake_native, api_key="secret-key")
    auth = {"Authorization": "Bearer secret-key"}
    async with running(settings) as (client, _):
        routes = [("GET", "/v1/models", None), ("GET", "/v1/records/x", None),
                  ("POST", CHAT, {"messages": MESSAGES}), ("POST", COMPLETIONS, {"prompt": "hi"})]
        for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "secret-key"},
                        {"Authorization": "Basic secret-key"}, {"Authorization": "Bearer secret-key-and-more"}):
            for method, path, payload in routes:
                r = await client.request(method, path, json=payload, headers=headers)
                assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key", (path, headers)
        r = await client.post(CHAT, content=b"x" * (limits.MAX_BODY_BYTES + 1), headers=JSON)
        assert r.status_code == 401  # nothing is read or parsed before authentication

        r = await client.post(CHAT, json={"messages": MESSAGES}, headers=auth | {"X-Request-Id": "a/../b"})
        assert r.status_code == 200
        await wait_for_file(tmp_path / "out" / "records" / "a%2F..%2Fb.json")
        r = await client.get("/v1/records/a%2F..%2Fb", headers=auth)
        assert r.status_code == 200 and r.json()["custom_id"] == "a/../b"
        # files outside records/ are never served, whatever the ID says
        (tmp_path / "out" / "secret.json").write_text(json.dumps({"custom_id": "../secret"}))
        (tmp_path / "out" / "records" / "plain.json").write_text(json.dumps({"custom_id": "other"}))
        for path in ("/v1/records/..%2Fsecret", "/v1/records/%2E%2E%2Fsecret", "/v1/records/plain"):
            assert (await client.get(path, headers=auth)).status_code == 404, path


# ---------------------------------------------------------------------------- streaming

async def test_streams_have_openai_shapes_and_keep_alive_while_queued(tmp_path, tiny_model, fake_native, monkeypatch):
    monkeypatch.setenv("FAKE_STEP_MS", "40")
    monkeypatch.setattr(limits, "SSE_KEEPALIVE_S", 0.02)
    async with running(make_settings(tmp_path, tiny_model, fake_native)) as (client, service):
        blocker = asyncio.create_task(client.post(CHAT, json={"messages": MESSAGES}))
        await eventually(lambda: service.scheduler.running == 1)
        payload = {"prompt": "<s>hello world", "stream": True, "stream_options": {"include_usage": True},
                   "request_id": "cs-1"}
        r = await client.post(COMPLETIONS, json=payload)
        assert (await blocker).status_code == 200
        plain = await client.post(CHAT, json={"messages": MESSAGES, "stream": True})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = [e for e in r.text.split("\n\n") if e]
    assert events[0] == ": keep-alive"  # queued behind the blocker: the connection is kept busy
    data = [e.removeprefix("data: ") for e in events if e.startswith("data: ")]
    assert data[-1] == "[DONE]"
    chunks = [json.loads(d) for d in data[:-1]]
    assert {(c["id"], c["object"], c["created"]) for c in chunks} == {("cmpl-cs-1", "text_completion",
                                                                      chunks[0]["created"])}
    assert "".join(c["choices"][0]["text"] for c in chunks if c["choices"]).startswith("tok0")
    assert chunks[-2]["choices"][0]["finish_reason"] in ("stop", "length")
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] == 6
    assert "metrics" in chunks[-1]

    chat = [json.loads(e.removeprefix("data: ")) for e in plain.text.split("\n\n") if e.startswith("data: {")]
    assert chat[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert chat[-1]["choices"][0]["delta"] == {} and chat[-1]["choices"][0]["finish_reason"] in ("stop", "length")
    assert all(c["usage"] is None for c in chat)  # no usage chunk without include_usage
    assert plain.text.endswith("data: [DONE]\n\n")


async def test_stream_that_cannot_start_is_cancelled_recorded_and_released(tmp_path, tiny_model, fake_native,
                                                                           monkeypatch):
    monkeypatch.setenv("FAKE_TOKENS", "40")
    monkeypatch.setenv("FAKE_STEP_MS", "50")
    async with serving(make_settings(tmp_path, tiny_model, fake_native)) as (app, client, service):
        call = ASGIRequest(app, CHAT, {"messages": MESSAGES, "stream": True, "max_tokens": 40},
                           headers={"X-Request-Id": "nostart"}, fail_start=True)
        with pytest.raises(SendFailed):
            await call.run()
        record = await read_record(tmp_path, "nostart")
        assert record["status"] == "cancelled" and record["http_status"] == 499
        assert record["usage"]["completion_tokens"] < 40
        await eventually(lambda: not service.scheduler.is_inflight("nostart"))
        r = await client.post(CHAT, json={"messages": MESSAGES}, headers={"X-Request-Id": "nostart"})
        assert r.status_code == 200


async def test_client_disconnect_cancels_and_records_once(tmp_path, tiny_model, fake_native, monkeypatch):
    monkeypatch.setenv("FAKE_TOKENS", "40")
    monkeypatch.setenv("FAKE_STEP_MS", "50")
    async with serving(make_settings(tmp_path, tiny_model, fake_native)) as (app, _, service):
        streamed = ASGIRequest(app, CHAT, {"messages": MESSAGES, "stream": True, "max_tokens": 40},
                               headers={"X-Request-Id": "gone-stream"})
        task = asyncio.create_task(streamed.run())
        await eventually(lambda: streamed.received.count(b'"content":"tok') >= 2)
        streamed.gone.set()
        await asyncio.wait_for(task, 5)

        plain = ASGIRequest(app, CHAT, {"messages": MESSAGES, "max_tokens": 40}, headers={"X-Request-Id": "gone-plain"})
        task = asyncio.create_task(plain.run())
        await eventually(lambda: service.scheduler.running == 1)
        await asyncio.sleep(0.2)
        plain.gone.set()
        await asyncio.wait_for(task, 5)

        for custom_id in ("gone-stream", "gone-plain"):
            record = await read_record(tmp_path, custom_id)
            assert record["status"] == "cancelled" and record["http_status"] == 499
            assert record["error"]["code"] == "client_disconnected"  # results.jsonl says why, too
            assert 1 <= record["usage"]["completion_tokens"] < 40  # stopped at the next step, not run to the end
            await eventually(lambda: not service.scheduler.is_inflight(custom_id))
        results = tmp_path / "out" / "results.jsonl"
        await eventually(lambda: results.exists() and len(results.read_text().splitlines()) >= 2)
        await asyncio.sleep(0.3)  # a second commit of either attempt would show up by now
        lines = [json.loads(line) for line in results.read_text().splitlines()]
        assert sorted(line["custom_id"] for line in lines) == ["gone-plain", "gone-stream"]  # one per attempt


async def test_cancelled_requests_stop_their_jobs_and_are_recorded(tmp_path, tiny_model, fake_native, monkeypatch):
    # uvicorn cancels the requests still running when its graceful-shutdown timeout ends
    monkeypatch.setenv("FAKE_TOKENS", "40")
    monkeypatch.setenv("FAKE_STEP_MS", "50")
    async with serving(make_settings(tmp_path, tiny_model, fake_native)) as (app, _, service):
        calls = {cid: ASGIRequest(app, CHAT, {"messages": MESSAGES, "max_tokens": 40}, headers={"X-Request-Id": cid})
                 for cid in ("cut-running", "cut-queued")}
        running_task = asyncio.create_task(calls["cut-running"].run())
        await eventually(lambda: service.scheduler.running == 1)
        queued_task = asyncio.create_task(calls["cut-queued"].run())
        await eventually(lambda: service.scheduler.waiting == 1)
        for task in (running_task, queued_task):
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        await eventually(lambda: service.scheduler.running == 0 and service.scheduler.waiting == 0, timeout=1.0)
        stopped = await read_record(tmp_path, "cut-running")
        assert stopped["status"] == "cancelled" and stopped["usage"]["completion_tokens"] < 40
        never_ran = await read_record(tmp_path, "cut-queued")
        assert never_ran["status"] == "cancelled" and never_ran["usage"]["completion_tokens"] == 0
        await eventually(lambda: not any(service.scheduler.is_inflight(cid) for cid in calls))


async def test_a_final_event_taken_just_before_the_request_is_cancelled_is_not_lost():
    # The request task can be cancelled after its reader took the final event but before the task resumed; the
    # job is then recorded by whoever takes it over (abandon -> _record_cancelled), which needs that event.
    class Connected:
        async def receive(self) -> dict:
            await asyncio.Event().wait()  # the client never goes away

    job = SimpleNamespace(events=asyncio.Queue())
    final = ("done", {"id": 1}, {"finish_reason": "stop"})
    job.events.put_nowait(("text", {"text": "a"}))
    job.events.put_nowait(final)
    task = asyncio.ensure_future(_wait_final(Connected(), job, SimpleNamespace(cancel=lambda job: None)))
    while not job.events.empty():  # the reader has taken the final event...
        await asyncio.sleep(0)
    assert not task.done()  # ...and the request task has not resumed yet
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert job.events.get_nowait() == final


# ---------------------------------------------------------------------------- responses and records agree

async def test_response_usage_is_the_recorded_usage(tmp_path, tiny_model, fake_native, monkeypatch):
    async with running(make_settings(tmp_path, tiny_model, fake_native)) as (client, service):
        tamper_done(service, monkeypatch, prompt_tokens=-1)  # the worker reports -1 when it has no count
        r = await client.post(CHAT, json={"messages": MESSAGES, "return_record": True})
        body = r.json()
        assert body["usage"] == body["record"]["usage"] and body["usage"]["prompt_tokens"] > 0
        r = await client.post(CHAT, json={"messages": MESSAGES, "stream": True,
                                          "stream_options": {"include_usage": True}})
        usage = json.loads(r.text.split("\n\n")[-3].removeprefix("data: "))["usage"]
        assert usage == body["usage"]


async def test_engine_errors_are_recorded_with_their_error(tmp_path, tiny_model, fake_native, monkeypatch):
    async with running(make_settings(tmp_path, tiny_model, fake_native)) as (client, service):
        tamper_done(service, monkeypatch, finish_reason="error")
        r = await client.post(CHAT, json={"messages": MESSAGES}, headers={"X-Request-Id": "boom"})
        assert r.status_code == 500 and r.json()["error"]["code"] == "engine_error"
        assert r.headers["x-request-id"] == "boom"
        r = await client.post(CHAT, json={"messages": MESSAGES, "stream": True}, headers={"X-Request-Id": "boom-sse"})
        assert r.status_code == 200 and '"code":"engine_error"' in r.text and r.text.endswith("data: [DONE]\n\n")
        for custom_id in ("boom", "boom-sse"):
            record = await read_record(tmp_path, custom_id)
            assert record["status"] == "failed" and record["http_status"] == 500
            assert record["error"]["code"] == "engine_error"


async def test_requests_stopped_by_a_shutdown_get_a_retryable_503(tmp_path, tiny_model, fake_native, monkeypatch):
    # the shutdown drains the service while clients are still connected: they must be told to retry, not 499
    monkeypatch.setenv("FAKE_TOKENS", "40")
    monkeypatch.setenv("FAKE_STEP_MS", "50")
    async with running(make_settings(tmp_path, tiny_model, fake_native)) as (client, service):
        plain = asyncio.create_task(client.post(CHAT, json={"messages": MESSAGES, "max_tokens": 40},
                                                headers={"X-Request-Id": "sd-running"}))
        await eventually(lambda: service.scheduler.running == 1)
        streamed = asyncio.create_task(client.post(CHAT, json={"messages": MESSAGES, "max_tokens": 40, "stream": True},
                                                   headers={"X-Request-Id": "sd-queued"}))
        await eventually(lambda: service.scheduler.waiting == 1)
        await service.scheduler.drain(0.1)
        r = await plain
        assert r.status_code == 503 and r.json()["error"]["code"] == "server_shutting_down"
        assert int(r.headers["retry-after"]) > 0
        s = await streamed
        assert s.status_code == 200 and '"code":"server_shutting_down"' in s.text
        assert s.text.endswith("data: [DONE]\n\n")
        for custom_id in ("sd-running", "sd-queued"):
            record = await read_record(tmp_path, custom_id)
            assert record["status"] == "cancelled" and record["http_status"] == 503
            assert record["error"]["code"] == "server_shutting_down"


async def test_client_gone_during_the_upload_is_not_an_application_error(tmp_path, tiny_model, fake_native):
    async with serving(make_settings(tmp_path, tiny_model, fake_native)) as (app, _, service):
        call = ASGIRequest(app, CHAT, {"messages": MESSAGES})
        messages = iter([{"type": "http.request", "body": call.body[:10], "more_body": True},
                         {"type": "http.disconnect"}])

        async def receive() -> dict:
            return next(messages)

        await app(call.scope, receive, call.send)  # used to raise ClientDisconnect (a traceback in the log)
        assert call.status == 400 and service.scheduler.waiting == service.scheduler.running == 0


def test_refusals_are_counted_under_their_own_reason():
    assert _rejection_reason(AdmissionError(503, "server_is_overloaded", "full")) == "queue_full"
    assert _rejection_reason(AdmissionError(503, "server_shutting_down", "bye")) == "shutting_down"
    assert _rejection_reason(AdmissionError(409, "duplicate_request_id", "again")) == "duplicate_request_id"


# ---------------------------------------------------------------------------- HTTP conventions

async def test_routing_errors_use_the_openai_error_format(tmp_path, tiny_model, fake_native):
    async with running(make_settings(tmp_path, tiny_model, fake_native)) as (client, _):
        r = await client.get(CHAT, headers={"X-Request-Id": "route-1"})
        assert r.status_code == 405 and r.json()["error"]["code"] == "method_not_allowed"
        assert "POST" in r.headers["allow"] and r.headers["x-request-id"] == "route-1"
        r = await client.post("/health")
        assert r.status_code == 405 and set(r.headers["allow"].replace(" ", "").split(",")) == {"GET", "HEAD"}
        r = await client.post("/v1/embeddings", json={})
        assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
        assert set(r.json()["error"]) == {"message", "type", "param", "code"}


async def test_models_are_listed_and_retrieved_and_probes_answer_head(tmp_path, tiny_model, fake_native):
    async with running(make_settings(tmp_path, tiny_model, fake_native)) as (client, service):
        listed = (await client.get("/v1/models")).json()["data"][0]
        assert (await client.get(f"/v1/models/{listed['id']}")).json() == listed  # what models.retrieve() calls
        r = await client.get("/v1/models/gpt-4o")
        assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"
        # the largest prompt + max_tokens accepted (one KV slot is the first prompt position)
        assert listed["max_model_len"] == service.engine.profile.kv_capacity - 1
        for path in ("/health", "/health/ready", "/metrics", "/v1/models"):
            r = await client.head(path)
            assert r.status_code == 200 and r.content == b"", path


async def test_errors_echo_the_client_id_and_a_401_names_its_scheme(tmp_path, tiny_model, fake_native):
    settings = make_settings(tmp_path, tiny_model, fake_native, api_key="secret-key")
    async with running(settings) as (client, _):
        r = await client.post(CHAT, json={"messages": MESSAGES}, headers={"X-Request-Id": "early-1"})
        assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
        assert r.headers["x-request-id"] == "early-1"
        auth = {"Authorization": "Bearer secret-key", "X-Request-Id": "early-2"}
        r = await client.post(CHAT, content=b"{not json", headers=auth | JSON)
        assert r.status_code == 400 and r.headers["x-request-id"] == "early-2"
        r = await client.post(CHAT, json={"messages": MESSAGES, "tools": [{}]}, headers=auth)
        assert r.status_code == 400 and r.headers["x-request-id"] == "early-2"
        r = await client.post(CHAT, json={"messages": MESSAGES},
                              headers=[("Authorization", "Bearer secret-key"), ("X-Request-Id", "a"),
                                       ("X-Request-Id", "b")])
        assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request_id"  # which one is meant?


async def test_sampling_requests_get_a_fresh_seed_that_the_record_keeps(tmp_path, tiny_model, fake_native):
    async with running(make_settings(tmp_path, tiny_model, fake_native)) as (client, _):
        for custom_id, extra in (("seed-a", {"temperature": 0.8}), ("seed-b", {"temperature": 0.8}),
                                 ("seed-c", {"temperature": 0.8, "seed": 7}), ("seed-d", {})):
            r = await client.post(CHAT, json={"messages": MESSAGES, **extra}, headers={"X-Request-Id": custom_id})
            assert r.status_code == 200
        seeds = {cid: (await read_record(tmp_path, cid))["request"]["params"]["seed"]
                 for cid in ("seed-a", "seed-b", "seed-c", "seed-d")}
        # Without one, Edge-LLM would draw the same random number at every step (greedy-like output).
        assert isinstance(seeds["seed-a"], int) and isinstance(seeds["seed-b"], int)
        assert seeds["seed-a"] != seeds["seed-b"]
        assert seeds["seed-c"] == 7 and seeds["seed-d"] is None  # the client's seed; greedy needs none


async def test_streams_deliver_the_text_the_engine_held_back(tmp_path, tiny_model, fake_native, monkeypatch):
    monkeypatch.setenv("FAKE_STREAM_HOLD", "4")  # as Edge-LLM's stop-string look-ahead never flushes its tail
    async with running(make_settings(tmp_path, tiny_model, fake_native)) as (client, _):
        for endpoint, payload in ((CHAT, {"messages": MESSAGES}), (COMPLETIONS, {"prompt": "<s>hi"})):
            plain = (await client.post(endpoint, json=payload)).json()["choices"][0]
            plain_text = plain["message"]["content"] if endpoint == CHAT else plain["text"]
            r = await client.post(endpoint, json=payload | {"stream": True})
            chunks = [json.loads(line[6:]) for line in r.text.splitlines()
                      if line.startswith("data: ") and line != "data: [DONE]"]
            streamed = "".join((c["choices"][0]["delta"].get("content") or "") if endpoint == CHAT
                               else c["choices"][0]["text"] for c in chunks if c["choices"])
            assert streamed == plain_text and plain_text.endswith("tok5 "), endpoint


async def test_the_role_chunk_waits_for_the_first_token(tmp_path, tiny_model, fake_native, monkeypatch):
    # Clients (and benchmarks) time the first chunk as the first token: it must not go out before the token does.
    monkeypatch.setenv("FAKE_STEP_MS", "400")
    monkeypatch.setenv("FAKE_TOKENS", "2")
    async with serving(make_settings(tmp_path, tiny_model, fake_native)) as (app, _, service):
        call = ASGIRequest(app, CHAT, {"messages": MESSAGES, "stream": True}, headers={"X-Request-Id": "role-1"})
        task = asyncio.create_task(call.run())
        await eventually(lambda: service.scheduler.running == 1)
        await asyncio.sleep(0.1)  # generating, first token ~0.4 s away
        assert b"data:" not in call.received
        await asyncio.wait_for(task, 10)
        assert call.received.index(b'"role":"assistant"') < call.received.index(b'"content":"tok0')


async def test_requests_during_a_drain_are_told_the_server_is_shutting_down(tmp_path, tiny_model, fake_native):
    async with running(make_settings(tmp_path, tiny_model, fake_native)) as (client, service):
        service.set_state("draining", "finishing in-flight requests")
        r = await client.post(CHAT, json={"messages": MESSAGES})
        assert r.status_code == 503 and r.json()["error"]["code"] == "server_shutting_down"
        assert int(r.headers["retry-after"]) > 0
        assert service.prometheus.rejections.labels(service.prometheus.model_name, "shutting_down")._value.get() == 1
        service.set_state("ready", "serving")


async def test_the_openapi_schema_describes_the_bodies_and_the_key(tmp_path, tiny_model, fake_native):
    async with running(make_settings(tmp_path, tiny_model, fake_native)) as (client, _):
        schema = (await client.get("/openapi.json")).json()
        chat = schema["paths"][CHAT]["post"]
        ref = chat["requestBody"]["content"]["application/json"]["schema"]["$ref"]
        assert ref == "#/components/schemas/ChatCompletionRequest"
        assert "messages" in schema["components"]["schemas"]["ChatCompletionRequest"]["properties"]
        assert "ChatMessage" in schema["components"]["schemas"]  # nested models resolve
        assert chat["security"] == [{"apiKey": []}] and "422" not in chat["responses"]
        assert "security" not in schema["paths"]["/health"]["get"]
        assert schema["components"]["securitySchemes"]["apiKey"]["scheme"] == "bearer"


async def test_top_k_is_bounded_and_the_whole_vocabulary_means_no_limit(tmp_path, tiny_model, fake_native, monkeypatch):
    import edgeserve.api.app as app_module

    sent: list[dict] = []
    async with running(make_settings(tmp_path, tiny_model, fake_native)) as (client, service):
        generate = service.worker.generate

        async def spy(items, sampling, stream):
            sent.append(dict(sampling))
            return await generate(items, sampling, stream)

        monkeypatch.setattr(service.worker, "generate", spy)
        vocab = service.model.output_vocab_size  # 128 for the tiny model
        r = await client.post(CHAT, json={"messages": MESSAGES, "temperature": 0.8, "top_k": vocab})
        assert r.status_code == 200 and sent[-1]["top_k"] == 0  # every token: no limit
        r = await client.post(CHAT, json={"messages": MESSAGES, "temperature": 0.8, "top_k": 40})
        assert r.status_code == 200 and sent[-1]["top_k"] == 40
        monkeypatch.setattr(app_module, "MAX_TOP_K", 32)  # (the real limit, 1024, is above the tiny vocabulary)
        r = await client.post(CHAT, json={"messages": MESSAGES, "temperature": 0.8, "top_k": 40})
        assert r.status_code == 400 and r.json()["error"]["param"] == "top_k"
        r = await client.post(CHAT, json={"messages": MESSAGES, "top_k": 40})  # greedy: top_k does not matter
        assert r.status_code == 200
