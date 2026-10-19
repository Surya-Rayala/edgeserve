from __future__ import annotations

import asyncio
import email.utils
import functools
import json
import re
import signal
import ssl
import stat
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import certifi
import httpcore
import httpx
import pytest

from edgeserve import batch, config
from edgeserve.batch import BatchFileError, BatchItem, BatchServerError, main, parse_batch_file, run_batch
from edgeserve.records import OutputDir

SERVER = "http://edge.test:8000"
CHAT = "/v1/chat/completions"
COMPLETIONS = "/v1/completions"
MESSAGES = [{"role": "user", "content": "hi"}]
RESULT_ID = re.compile(r"batch_req_[0-9a-f]{24}")
ITEM_OK = re.compile(r"\[\d+/\d+\] \S+ ok 8\.4 s \(ttft 2\.0 s, 24\.6 tok/s\)")
CONTEXT_ERROR = {
    "message": "This model's maximum context length is 3072 tokens",
    "type": "invalid_request_error",
    "param": "messages",
    "code": "context_length_exceeded",
}
_ORIGINAL_RUN_BATCH = batch.run_batch


@pytest.fixture(autouse=True)
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Backoff waits are recorded instead of slept."""
    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(batch, "_sleep", fake_sleep)
    return delays


def _server_record(custom_id: str) -> dict:
    return {
        "object": "edgeserve.record",
        "schema_version": "1.0",
        "custom_id": custom_id,
        "attempt_id": f"att-{custom_id}",
        "status": "completed",
        "timestamps_ns": {"arrival": 1_000_000_000, "finished": 9_400_000_000},
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        "metrics": {"ttft_ms": 2000.0, "tpot_ms": 40.0, "e2e_ms": 8400.0, "queue_ms": 1.0, "decode_tokens_per_s": 24.6},
    }


def _openai_body(custom_id: str) -> dict:
    return {
        "id": f"chatcmpl-{custom_id}",
        "object": "chat.completion",
        "model": "summarizer",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "a summary"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


def _ok(custom_id: str) -> httpx.Response:
    return httpx.Response(200, json={**_openai_body(custom_id), "record": _server_record(custom_id)})


def respond(
    status: int, json_body: object = None, text: str | None = None, headers: dict | None = None
) -> Callable[[], httpx.Response]:
    """A scripted answer; a factory, so every attempt gets a fresh response."""

    def make() -> httpx.Response:
        if text is not None:
            return httpx.Response(status, text=text, headers=headers)
        return httpx.Response(status, json=json_body, headers=headers)

    return make


class FakeServer:
    """MockTransport handler: GET /v1/models plus scripted answers per X-Request-Id ("ok" once a script runs out).

    A script step is "ok", a response factory, or an httpx exception class (raised as "connection refused").
    """

    def __init__(
        self,
        script: dict[str, list] | None = None,
        max_batch_size: int | None = 1,
        models: Callable[[], httpx.Response] | None = None,
    ) -> None:
        self.script = {custom_id: list(steps) for custom_id, steps in (script or {}).items()}
        self.max_batch_size = max_batch_size
        self.models = models
        self.posts: list[httpx.Request] = []
        self.model_requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/models":
            self.model_requests.append(request)
            if self.models is not None:
                return self.models()
            entry: dict = {"id": "summarizer", "object": "model"}
            if self.max_batch_size is not None:
                entry["max_batch_size"] = self.max_batch_size
            return httpx.Response(200, json={"object": "list", "data": [entry]})
        self.posts.append(request)
        custom_id = request.headers["x-request-id"]
        steps = self.script.get(custom_id)
        step = steps.pop(0) if steps else "ok"
        if isinstance(step, type) and issubclass(step, Exception):
            raise step("connection refused", request=request)
        if callable(step):
            return step()
        return _ok(custom_id)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    @property
    def model_gets(self) -> int:
        return len(self.model_requests)

    def attempts(self, custom_id: str) -> int:
        return sum(request.headers["x-request-id"] == custom_id for request in self.posts)


def _item(custom_id: str, url: str = CHAT, line_no: int = 1, **body: Any) -> BatchItem:
    if not body:
        body = {"messages": MESSAGES} if url == CHAT else {"prompt": "<|begin_of_text|>hi"}
    return BatchItem(custom_id=custom_id, url=url, body=body, line_no=line_no)


async def _run(items: list[BatchItem], out: OutputDir, server: FakeServer, **kwargs: Any) -> tuple[dict, list[str]]:
    lines: list[str] = []
    summary = await run_batch(items, out, SERVER, progress=lines.append, _transport=server.transport, **kwargs)
    return summary, lines


async def _until(predicate: Callable[[], bool], steps: int = 10_000) -> None:
    for _ in range(steps):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition never became true")


def _results(root: Path) -> list[dict]:
    path = root / "results.jsonl"
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()] if path.exists() else []


def _run_json(root: Path) -> dict:
    return json.loads((root / "run.json").read_text("utf-8"))


def _failed(custom_id: str, http_status: int | None, error: dict, attempts: int) -> dict:
    return {
        "object": "edgeserve.record",
        "schema_version": "1.0",
        "custom_id": custom_id,
        "status": "failed",
        "http_status": http_status,
        "error": error,
        "attempts": attempts,
    }


def _write_input(tmp_path: Path, *lines: str | bytes) -> Path:
    path = tmp_path / "input.jsonl"
    path.write_bytes(b"\n".join(line if isinstance(line, bytes) else line.encode() for line in lines) + b"\n")
    return path


def _input(tmp_path: Path, *objects: dict) -> Path:
    return _write_input(tmp_path, *(json.dumps(obj) for obj in objects))


def _patch_network(monkeypatch: pytest.MonkeyPatch, handler: Callable) -> None:
    """main() builds its own client: hand run_batch a mock transport."""
    patched = functools.partial(_ORIGINAL_RUN_BATCH, _transport=httpx.MockTransport(handler))
    monkeypatch.setattr(batch, "run_batch", patched)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


# --- parse_batch_file --------------------------------------------------------------------------


def test_parse_accepts_batch_and_short_forms(tmp_path: Path) -> None:
    body = {"messages": MESSAGES, "max_tokens": 8}
    batch_line = {"custom_id": "seg-001", "method": "POST", "url": CHAT, "body": body}
    path = _write_input(
        tmp_path,
        "# segments for class 3",
        json.dumps(batch_line),
        "",
        json.dumps({"id": "seg-002", "messages": MESSAGES, "max_tokens": 512}),
        json.dumps({"id": "seg-003", "prompt": "<|begin_of_text|>hello"}),
        "   " + json.dumps({"custom_id": "seg-004", "body": {"prompt": "x", "id": "dropped"}}),
        json.dumps({"custom_id": "seg-005", "id": "seg-005", "messages": [], "stream": False}) + "\r",
        "  # an indented comment",
        json.dumps({"custom_id": "seg 006", "method": "post", "url": COMPLETIONS, "body": {"prompt": ["a", "b"]}}),
    )
    items, invalid = parse_batch_file(path)
    assert invalid == []
    assert items == [
        BatchItem("seg-001", CHAT, {"messages": MESSAGES, "max_tokens": 8}, 2),
        BatchItem("seg-002", CHAT, {"messages": MESSAGES, "max_tokens": 512}, 4),
        BatchItem("seg-003", COMPLETIONS, {"prompt": "<|begin_of_text|>hello"}, 5),
        BatchItem("seg-004", COMPLETIONS, {"prompt": "x"}, 6),
        BatchItem("seg-005", CHAT, {"messages": [], "stream": False}, 7),
        BatchItem("seg 006", COMPLETIONS, {"prompt": ["a", "b"]}, 9),
    ]


def test_parse_handles_bom_crlf_and_512_char_ids(tmp_path: Path) -> None:
    longest = "x" * 512
    path = tmp_path / "input.jsonl"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"id": longest, "prompt": "p"}).encode() + b"\r\n")
    items, invalid = parse_batch_file(path)
    assert invalid == []
    assert [item.custom_id for item in items] == [longest]


@pytest.mark.parametrize(
    ("line", "custom_id", "fragment"),
    [
        ("not json", None, "invalid JSON"),
        ("[1, 2]", None, "expected a JSON object, got an array"),
        ('{"messages": []}', None, 'missing "custom_id"'),
        ('{"id": 7, "messages": []}', None, "custom_id must be a string, got a number"),
        ('{"id": "", "messages": []}', None, "custom_id is empty"),
        (json.dumps({"id": "x" * 513, "messages": []}), None, "longer than 512"),
        ('{"id": "caf\\u00e9", "messages": []}', None, "printable ASCII"),
        ('{"id": "tab\\there", "messages": []}', None, "printable ASCII"),
        ('{"id": " lead", "messages": []}', None, "start or end with a space"),
        ('{"id": "trail ", "messages": []}', None, "start or end with a space"),
        ('{"id": "a", "custom_id": "b", "messages": []}', None, "differ"),
        ('{"id": "nan", "messages": [], "temperature": NaN}', None, "invalid JSON"),
        ('{"id": "big", "messages": [], "temperature": 1e999}', None, "invalid JSON"),
        (json.dumps({"custom_id": "m1", "method": "GET", "url": CHAT, "body": {}}), "m1", "unsupported method"),
        (json.dumps({"custom_id": "u1", "url": "/v1/embeddings", "body": {"input": "x"}}), "u1", "unsupported url"),
        ('{"custom_id": "u2", "url": "/v1/completions", "body": {"messages": []}}', "u2", 'needs "prompt"'),
        ('{"custom_id": "u3", "url": "/v1/chat/completions", "body": {"prompt": "x"}}', "u3", 'needs "messages"'),
        ('{"id": "b1", "messages": [], "prompt": "x"}', "b1", "not both"),
        ('{"id": "n1", "max_tokens": 5}', "n1", 'missing "messages"'),
        ('{"id": "s1", "messages": [], "stream": true}', "s1", '"stream": true is not supported'),
        ('{"custom_id": "k1", "body": {"messages": []}, "max_tokens": 5}', "k1", "unexpected top-level key(s)"),
        ('{"custom_id": "k2", "body": [1]}', "k2", '"body" must be a JSON object'),
        ('{"id": "t1", "messages": "hello"}', "t1", '"messages" must be an array, got a string'),
        ('{"id": "t2", "prompt": 5}', "t2", '"prompt" must be a string'),
    ],
)
def test_parse_reports_invalid_lines(tmp_path: Path, line: str, custom_id: str | None, fragment: str) -> None:
    path = _write_input(tmp_path, json.dumps({"id": "good", "prompt": "x"}), line)
    items, invalid = parse_batch_file(path)
    assert [item.custom_id for item in items] == ["good"]
    assert len(invalid) == 1
    assert (invalid[0]["line_no"], invalid[0]["custom_id"]) == (2, custom_id)
    assert fragment in invalid[0]["message"]


def test_parse_ignores_byte_order_marks_of_concatenated_files(tmp_path: Path) -> None:
    bom = b"\xef\xbb\xbf"  # e.g. `cat a.jsonl b.jsonl` of two files saved by a Windows editor
    first = bom + json.dumps({"id": "a", "prompt": "x"}).encode()
    second = bom + json.dumps({"id": "b", "prompt": "y"}).encode()
    items, invalid = parse_batch_file(_write_input(tmp_path, first, second, bom))
    assert invalid == []
    assert [(item.custom_id, item.line_no) for item in items] == [("a", 1), ("b", 2)]


def test_parse_reports_lines_that_are_not_utf8(tmp_path: Path) -> None:
    path = _write_input(tmp_path, b'{"id": "a\xff", "prompt": "x"}', json.dumps({"id": "ok", "prompt": "x"}))
    items, invalid = parse_batch_file(path)
    assert [item.custom_id for item in items] == ["ok"]
    assert invalid == [{"line_no": 1, "custom_id": None, "message": "line is not valid UTF-8"}]


def test_parse_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = _write_input(
        tmp_path,
        json.dumps({"id": "seg-1", "prompt": "a"}),
        json.dumps({"id": "seg-2", "prompt": "a"}),
        json.dumps({"id": "seg-1", "prompt": "b"}),
        json.dumps({"id": "seg-2", "prompt": "b", "messages": []}),  # invalid, but its failure record would collide
        json.dumps({"id": "seg-3", "prompt": "a"}),
    )
    with pytest.raises(BatchFileError) as info:
        parse_batch_file(path)
    message = str(info.value)
    assert "'seg-1' (lines 1, 3)" in message
    assert "'seg-2' (lines 2, 4)" in message
    assert "seg-3" not in message


def test_parse_lists_at_most_20_duplicates(tmp_path: Path) -> None:
    lines = [json.dumps({"id": f"id-{i}", "prompt": "x"}) for i in range(25)] * 2
    with pytest.raises(BatchFileError, match="and 5 more$"):
        parse_batch_file(_write_input(tmp_path, *lines))


def test_parse_unreadable_files(tmp_path: Path) -> None:
    assert issubclass(BatchFileError, ValueError)
    with pytest.raises(BatchFileError, match="cannot read"):
        parse_batch_file(tmp_path / "missing.jsonl")
    with pytest.raises(BatchFileError, match="cannot read"):
        parse_batch_file(tmp_path)  # a directory


def test_parse_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    assert parse_batch_file(path) == ([], [])


# --- run_batch: success, protocol and concurrency ----------------------------------------------


async def test_success_saves_records_results_and_run(tmp_path: Path) -> None:
    server = FakeServer(max_batch_size=1)
    out = OutputDir(tmp_path / "out")
    items = [_item("seg-001"), _item("seg-002", COMPLETIONS, line_no=2, prompt="<|begin_of_text|>hi", max_tokens=5)]
    lines: list[str] = []
    summary = await run_batch(
        items, out, SERVER + "/", api_key="sekret", progress=lines.append, _transport=server.transport
    )

    assert server.model_gets == 1
    assert server.model_requests[0].headers["authorization"] == "Bearer sekret"
    posts = {request.headers["x-request-id"]: request for request in server.posts}
    assert set(posts) == {"seg-001", "seg-002"}
    assert posts["seg-001"].method == "POST" and posts["seg-001"].url == SERVER + CHAT
    assert posts["seg-002"].url == SERVER + COMPLETIONS
    assert json.loads(posts["seg-001"].content) == {"messages": MESSAGES, "return_record": True}
    assert json.loads(posts["seg-002"].content) == {
        "prompt": "<|begin_of_text|>hi",
        "max_tokens": 5,
        "return_record": True,
    }
    assert {request.headers["authorization"] for request in server.posts} == {"Bearer sekret"}

    for custom_id in posts:
        assert out.read_record(custom_id) == _server_record(custom_id)
        assert _mode(out.record_path(custom_id)) == 0o600

    results = _results(out.root)
    assert sorted(line["custom_id"] for line in results) == ["seg-001", "seg-002"]
    for line in results:
        assert RESULT_ID.fullmatch(line["id"])
        assert line["error"] is None
        custom_id = line["custom_id"]
        expected = {"status_code": 200, "request_id": f"att-{custom_id}", "body": _openai_body(custom_id)}
        assert line["response"] == expected
    assert results[0]["id"] != results[1]["id"]

    run = _run_json(out.root)
    assert run == summary
    assert run["object"] == "batch_run" and run["server"] == SERVER
    assert run["started_at"] <= run["finished_at"]
    assert (run["input_items"], run["skipped_completed"], run["invalid_lines"]) == (2, 0, [])
    assert run["concurrency"] == 2 and run["interrupted"] is False
    assert run["request_counts"] == {"total": 2, "completed": 2, "failed": 0, "cancelled": 0}
    assert run["usage"] == {"prompt_tokens": 20, "completion_tokens": 40, "total_tokens": 60}
    assert run["e2e_ms"]["p50"] == 8400.0
    assert _mode(out.root / "results.jsonl") == 0o600 and _mode(out.root / "run.json") == 0o600

    assert lines[0] == f"sending 2 requests to {SERVER} (concurrency 2)"
    assert len(lines) == 3 and all(ITEM_OK.fullmatch(line) for line in lines[1:])
    assert sorted(line.split()[0] for line in lines[1:]) == ["[1/2]", "[2/2]"]


@pytest.mark.parametrize("api_key", [None, ""])
async def test_no_authorization_header_without_api_key(tmp_path: Path, api_key: str | None) -> None:
    server = FakeServer()
    await _run([_item("seg-1")], OutputDir(tmp_path), server, api_key=api_key)
    assert server.posts and server.model_requests
    assert all("authorization" not in request.headers for request in server.posts + server.model_requests)


@pytest.mark.parametrize(
    ("max_batch_size", "concurrency", "expected"),
    [(3, None, 4), (None, None, 2), (3, 1, 1), (1, 5, 5)],
)
async def test_concurrency(tmp_path: Path, max_batch_size: int | None, concurrency: int | None, expected: int) -> None:
    in_flight = peak = 0
    gate = asyncio.Event()
    probes: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        if request.url.path == "/v1/models":
            probes.append(request)
            entry: dict = {"id": "summarizer"}
            if max_batch_size is not None:
                entry["max_batch_size"] = max_batch_size
            return httpx.Response(200, json={"object": "list", "data": [entry]})
        in_flight += 1
        peak = max(peak, in_flight)
        if in_flight >= expected:
            gate.set()
        try:
            await asyncio.wait_for(gate.wait(), timeout=2)  # only times out when the expected level never builds up
        except asyncio.TimeoutError:
            pass
        finally:
            in_flight -= 1
        return _ok(request.headers["x-request-id"])

    items = [_item(f"seg-{i}") for i in range(8)]
    transport = httpx.MockTransport(handler)
    summary = await run_batch(
        items, OutputDir(tmp_path), SERVER, concurrency=concurrency, progress=lambda line: None, _transport=transport
    )
    assert peak == expected
    assert summary["concurrency"] == expected
    assert len(probes) == 1  # always probed: a refused key or untrusted certificate stops the run before it starts
    assert summary["request_counts"]["completed"] == 8


def _refuse() -> httpx.Response:
    raise httpx.ConnectError("connection refused")


@pytest.mark.parametrize(
    ("models", "notice"),
    [
        (respond(500, text="boom"), True),
        (respond(200, text="not json"), True),
        (respond(200, json_body={"object": "list"}), True),
        (respond(200, json_body={"data": "x"}), True),
        (respond(200, text="[" * 100_000 + "]" * 100_000), True),  # json raises RecursionError
        (_refuse, True),
        (respond(200, json_body={"data": []}), False),
        (respond(200, json_body={"data": [{"id": "m", "max_batch_size": 0}]}), False),
        (respond(200, json_body={"data": [{"id": "m", "max_batch_size": "4"}]}), False),
        (respond(200, json_body={"data": [{"id": "m", "max_batch_size": True}]}), False),
    ],
)
async def test_concurrency_falls_back_to_2(tmp_path: Path, models: Callable[[], httpx.Response], notice: bool) -> None:
    server = FakeServer(models=models)
    summary, lines = await _run([_item("seg-1")], OutputDir(tmp_path), server)
    assert summary["concurrency"] == 2
    notices = [line for line in lines if line.startswith("cannot read max_batch_size")]
    assert bool(notices) == notice
    assert all(line.endswith("; using concurrency 2") for line in notices)
    assert summary["request_counts"]["completed"] == 1


@pytest.mark.parametrize("status", [401, 403])
async def test_a_refused_key_stops_the_run_before_anything_is_sent(tmp_path: Path, status: int) -> None:
    server = FakeServer(models=respond(status, json_body={"error": {"message": "bad key"}}))
    with pytest.raises(BatchServerError, match=r"refused the API key.*EDGESERVE_API_KEY"):
        await _run([_item("seg-1"), _item("seg-2")], OutputDir(tmp_path), server, concurrency=1)
    assert server.posts == [] and _results(tmp_path) == []  # every item would have failed the same way


async def test_plain_http_to_an_https_server_stops_the_run_with_a_hint(tmp_path: Path) -> None:
    def closes(request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("Server disconnected without sending a response.", request=request)

    with pytest.raises(BatchServerError, match="use https://"):
        await run_batch([_item("seg-1")], OutputDir(tmp_path), SERVER, progress=lambda line: None,
                        _transport=httpx.MockTransport(closes))
    assert _results(tmp_path) == []


async def test_the_first_retry_of_an_item_is_reported(tmp_path: Path, sleeps: list[float]) -> None:
    server = FakeServer({"seg-1": [respond(503, json_body={"error": {"message": "busy"}}), "ok"]})
    _, lines = await _run([_item("seg-1")], OutputDir(tmp_path), server, concurrency=1)
    assert any(line.startswith("seg-1: HTTP 503; retrying in") for line in lines)
    assert lines[-1].startswith("[1/1] seg-1 ok")


async def test_an_invalid_line_is_recorded_once_however_often_the_file_is_run(tmp_path: Path) -> None:
    invalid = [{"line_no": 2, "custom_id": "bad-1", "message": "broken"}]
    for _ in range(3):
        await _run([_item("seg-1")], OutputDir(tmp_path), FakeServer(), invalid=invalid)
    assert [line["custom_id"] for line in _results(tmp_path)].count("bad-1") == 1


async def test_concurrency_must_be_positive(tmp_path: Path) -> None:
    server = FakeServer()
    with pytest.raises(ValueError, match="concurrency"):
        await _run([_item("seg-1")], OutputDir(tmp_path), server, concurrency=0)
    assert server.posts == []


_INVALID_WITH_ID = [{"line_no": 2, "custom_id": "bad-1", "message": "broken"}]


def _nothing_written(root: Path) -> bool:
    return sorted(path.name for path in root.iterdir()) == ["records"] and not any((root / "records").iterdir())


@pytest.mark.parametrize("api_key", ["s3cr3t\nX-Injected: 1", "s3cr3té", " s3cr3t", "s3cr3t ", "s3c\tr3t"])
async def test_unusable_api_key_is_rejected_before_anything_is_written(tmp_path: Path, api_key: str) -> None:
    # A real transport (h11) rejects such a header with an error that quotes it, i.e. the key would be saved
    # in every failed record; a non-ASCII key made httpx raise before run.json was written.
    server = FakeServer()
    with pytest.raises(ValueError, match="api_key") as info:
        await _run([_item("seg-1")], OutputDir(tmp_path), server, api_key=api_key, invalid=_INVALID_WITH_ID)
    assert "s3c" not in str(info.value)  # never echoed
    assert server.posts == [] and server.model_gets == 0
    assert _nothing_written(tmp_path)


async def test_api_key_with_inner_spaces_is_sent(tmp_path: Path) -> None:
    server = FakeServer()
    await _run([_item("seg-1")], OutputDir(tmp_path), server, api_key="two words")
    assert {request.headers["authorization"] for request in server.posts} == {"Bearer two words"}


@pytest.mark.parametrize(
    "server_url",
    [
        "ftp://edge.test",
        "edge.test:8000",
        "http://",
        "http://edge.test:8000?x=1",
        "http://edge\x01test:8000",  # httpx: non-printable character
        "http://☃️.example",  # httpx: invalid IDNA host name (escaped the probe before)
    ],
)
async def test_unusable_server_url_is_rejected_before_anything_is_written(tmp_path: Path, server_url: str) -> None:
    server = FakeServer()
    with pytest.raises(ValueError):
        await run_batch(
            [_item("seg-1")],
            OutputDir(tmp_path),
            server_url,
            progress=lambda line: None,
            invalid=_INVALID_WITH_ID,
            _transport=server.transport,
        )
    assert server.posts == [] and server.model_gets == 0
    assert _nothing_written(tmp_path)


async def test_duplicate_item_ids_are_rejected(tmp_path: Path) -> None:
    server = FakeServer()
    items = [_item("seg-1"), _item("seg-2"), _item("seg-1", line_no=3)]
    with pytest.raises(BatchFileError, match=r"duplicate IDs: 'seg-1'$"):
        await _run(items, OutputDir(tmp_path), server)
    assert server.posts == [] and server.model_gets == 0
    assert _nothing_written(tmp_path)


async def test_200_without_record_is_saved_as_completed(tmp_path: Path) -> None:
    server = FakeServer({"seg-1": [respond(200, json_body=_openai_body("seg-1"))]})
    out = OutputDir(tmp_path)
    _, lines = await _run([_item("seg-1")], out, server, concurrency=1)
    assert out.read_record("seg-1") == {
        "object": "edgeserve.record",
        "schema_version": "1.0",
        "custom_id": "seg-1",
        "status": "completed",
        "http_status": 200,
        "usage": _openai_body("seg-1")["usage"],
        "response": _openai_body("seg-1"),
    }
    assert out.completed_ids() == {"seg-1"}
    [line] = _results(tmp_path)
    assert line["response"] == {"status_code": 200, "request_id": None, "body": _openai_body("seg-1")}
    assert re.fullmatch(r"\[1/1\] seg-1 ok \d+\.\d s", lines[-1])


async def test_record_is_filed_under_the_input_id(tmp_path: Path) -> None:
    record = {**_server_record("seg-1"), "custom_id": "renamed-by-server"}
    server = FakeServer({"seg-1": [respond(200, json_body={**_openai_body("seg-1"), "record": record})]})
    out = OutputDir(tmp_path)
    await _run([_item("seg-1")], out, server, concurrency=1)
    assert out.read_record("seg-1") == {**record, "custom_id": "seg-1"}
    assert out.read_record("renamed-by-server") is None


async def test_progress_shows_a_non_completed_server_status(tmp_path: Path) -> None:
    record = {**_server_record("seg-1"), "status": "cancelled"}
    server = FakeServer({"seg-1": [respond(200, json_body={**_openai_body("seg-1"), "record": record})]})
    _, lines = await _run([_item("seg-1")], OutputDir(tmp_path), server, concurrency=1)
    assert lines[-1] == "[1/1] seg-1 cancelled 8.4 s (ttft 2.0 s, 24.6 tok/s)"


@pytest.mark.parametrize(
    ("answer", "body"), [(respond(200, text="<html>proxy</html>"), None), (respond(200, json_body=[1, 2]), [1, 2])]
)
async def test_200_that_is_not_a_json_object_fails(
    tmp_path: Path, sleeps: list[float], answer: Callable[[], httpx.Response], body: object
) -> None:
    server = FakeServer({"seg-1": [answer]})
    out = OutputDir(tmp_path)
    await _run([_item("seg-1")], out, server, concurrency=1)
    assert server.attempts("seg-1") == 1 and sleeps == []
    record = out.read_record("seg-1")
    assert (record["status"], record["http_status"], record["error"]["code"]) == ("failed", 200, "invalid_response")
    [line] = _results(tmp_path)
    assert line["response"] == {"status_code": 200, "request_id": None, "body": body}
    assert line["error"]["code"] == "invalid_response"


# --- run_batch: errors and retries -------------------------------------------------------------


async def test_503_with_retry_after_then_success(tmp_path: Path, sleeps: list[float]) -> None:
    busy = {"error": {"message": "queue full", "type": "server_busy", "param": None, "code": None}}
    server = FakeServer({"seg-1": [respond(503, json_body=busy, headers={"Retry-After": "7"}), "ok"]})
    out = OutputDir(tmp_path)
    _, lines = await _run([_item("seg-1")], out, server, concurrency=1)
    assert sleeps == [7.0]
    assert server.attempts("seg-1") == 2
    assert out.read_record("seg-1") == _server_record("seg-1")
    assert [line["response"]["status_code"] for line in _results(tmp_path)] == [200]  # one line per item
    assert ITEM_OK.fullmatch(lines[-1])


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
async def test_retryable_statuses(tmp_path: Path, sleeps: list[float], status: int) -> None:
    server = FakeServer({"seg-1": [respond(status, text="busy")]})
    out = OutputDir(tmp_path)
    await _run([_item("seg-1")], out, server, concurrency=1)
    assert sleeps == [1.0]
    assert server.attempts("seg-1") == 2
    assert out.read_record("seg-1")["status"] == "completed"


async def test_retries_back_off_exponentially_then_fail(tmp_path: Path, sleeps: list[float]) -> None:
    error = {"message": "worker crashed", "type": "server_error", "param": None, "code": "worker_error"}
    server = FakeServer({"seg-1": [respond(500, json_body={"error": error})] * 6})
    out = OutputDir(tmp_path)
    summary, lines = await _run([_item("seg-1")], out, server, concurrency=1)
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0]
    assert server.attempts("seg-1") == 6
    assert out.read_record("seg-1") == _failed("seg-1", 500, error, 6)
    [line] = _results(tmp_path)
    assert line["response"] == {"status_code": 500, "request_id": None, "body": {"error": error}}
    assert line["error"] is None
    assert lines[-1] == "[1/1] seg-1 failed: HTTP 500 worker_error: worker crashed after 6 attempts"
    assert summary["request_counts"] == {"total": 1, "completed": 0, "failed": 1, "cancelled": 0}


def _http_date(seconds_from_now: float) -> str:
    return email.utils.format_datetime(datetime.now(timezone.utc) + timedelta(seconds=seconds_from_now), usegmt=True)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("600", 60.0),
        ("0", 0.0),
        ("2.5", 2.5),
        ("-5", 0.0),
        ("Wed, 21 Oct 2015 07:28:00 GMT", 0.0),
        ("soon", 1.0),
        ("nan", 1.0),
        ("", 1.0),
    ],
)
async def test_retry_after_values(tmp_path: Path, sleeps: list[float], value: str, expected: float) -> None:
    server = FakeServer({"seg-1": [respond(429, text="slow down", headers={"Retry-After": value})]})
    await _run([_item("seg-1")], OutputDir(tmp_path), server, concurrency=1)
    assert sleeps == [expected]


async def test_retry_after_http_date_in_the_future(tmp_path: Path, sleeps: list[float]) -> None:
    server = FakeServer({"seg-1": [respond(503, text="later", headers={"Retry-After": _http_date(30)})]})
    await _run([_item("seg-1")], OutputDir(tmp_path), server, concurrency=1)
    assert len(sleeps) == 1 and 27.0 <= sleeps[0] <= 30.0


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 413, 422, 501])
async def test_client_errors_are_not_retried(tmp_path: Path, sleeps: list[float], status: int) -> None:
    server = FakeServer({"seg-1": [respond(status, json_body={"error": CONTEXT_ERROR})]})
    out = OutputDir(tmp_path)
    _, lines = await _run([_item("seg-1")], out, server, concurrency=1)
    assert sleeps == [] and server.attempts("seg-1") == 1
    assert out.read_record("seg-1") == _failed("seg-1", status, CONTEXT_ERROR, 1)
    assert out.completed_ids() == set()
    [line] = _results(tmp_path)
    assert line["response"] == {"status_code": status, "request_id": None, "body": {"error": CONTEXT_ERROR}}
    assert line["error"] is None
    assert lines[-1] == f"[1/1] seg-1 failed: HTTP {status} context_length_exceeded: {CONTEXT_ERROR['message']}"


@pytest.mark.parametrize(("text", "message"), [("upstream says no", "upstream says no"), ("", "Not Found")])
async def test_error_without_a_json_body(tmp_path: Path, text: str, message: str) -> None:
    server = FakeServer({"seg-1": [respond(404, text=text)]})
    out = OutputDir(tmp_path)
    _, lines = await _run([_item("seg-1")], out, server, concurrency=1)
    assert out.read_record("seg-1") == _failed("seg-1", 404, {"message": message}, 1)
    [line] = _results(tmp_path)
    assert line["response"] == {"status_code": 404, "request_id": None, "body": {"error": {"message": message}}}
    assert lines[-1] == f"[1/1] seg-1 failed: HTTP 404: {message}"


@pytest.mark.parametrize(
    ("status", "message"), [(499, "cancelled by the server"), (599, "empty response body")]
)
async def test_error_without_body_or_reason_phrase(
    tmp_path: Path, sleeps: list[float], status: int, message: str
) -> None:
    # The server answers 499 with an empty body when it cancels a request while shutting down.
    server = FakeServer({"seg-1": [respond(status, text="")]})
    out = OutputDir(tmp_path)
    _, lines = await _run([_item("seg-1")], out, server, concurrency=1)
    assert out.read_record("seg-1") == _failed("seg-1", status, {"message": message}, 1)
    assert sleeps == [] and server.attempts("seg-1") == 1
    assert lines[-1] == f"[1/1] seg-1 failed: HTTP {status}: {message}"


async def test_long_error_text_is_truncated(tmp_path: Path) -> None:
    server = FakeServer({"seg-1": [respond(400, text="x" * 10_000)]})
    out = OutputDir(tmp_path)
    _, lines = await _run([_item("seg-1")], out, server, concurrency=1)
    assert len(out.read_record("seg-1")["error"]["message"]) == 4096
    assert len(lines[-1]) < 250


@pytest.mark.parametrize(
    ("error", "attempts"),
    [
        (httpx.ConnectError, 6),
        (httpx.ConnectTimeout, 6),
        (httpx.ReadTimeout, 6),
        (httpx.ReadError, 6),
        (httpx.RemoteProtocolError, 6),
        (httpx.UnsupportedProtocol, 1),
        (httpx.LocalProtocolError, 1),
    ],
)
async def test_connection_errors(tmp_path: Path, sleeps: list[float], error: type[Exception], attempts: int) -> None:
    server = FakeServer({"seg-1": [error] * 6})
    out = OutputDir(tmp_path)
    summary, lines = await _run([_item("seg-1")], out, server, concurrency=1)
    assert server.attempts("seg-1") == attempts
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0][: attempts - 1]
    message = f"{error.__name__}: connection refused"
    expected_error = {"message": message, "type": "connection_error", "param": None, "code": "connection_error"}
    assert out.read_record("seg-1") == _failed("seg-1", None, expected_error, attempts)
    [line] = _results(tmp_path)
    assert line["response"] is None
    assert line["error"] == {"code": "connection_error", "message": message}
    assert lines[-1] == f"[1/1] seg-1 failed: {message}" + (f" after {attempts} attempts" if attempts > 1 else "")
    assert summary["request_counts"]["failed"] == 1


async def test_connection_errors_then_success(tmp_path: Path, sleeps: list[float]) -> None:
    server = FakeServer({"seg-1": [httpx.ConnectError, httpx.RemoteProtocolError, "ok"]})
    out = OutputDir(tmp_path)
    await _run([_item("seg-1")], out, server, concurrency=1)
    assert sleeps == [1.0, 2.0]
    assert out.read_record("seg-1") == _server_record("seg-1")


_CERT_FAILURE = "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate (_ssl.c:1007)"


def _untrusted_certificate(request: httpx.Request) -> httpx.Response:
    """What httpx raises for a server certificate the client does not trust: httpx <- httpcore <- ssl."""
    error = httpx.ConnectError(_CERT_FAILURE, request=request)
    error.__cause__ = httpcore.ConnectError(_CERT_FAILURE)
    error.__cause__.__cause__ = ssl.SSLCertVerificationError(1, _CERT_FAILURE)
    raise error


async def test_https_to_a_plain_http_server_stops_the_run_with_a_hint(tmp_path: Path) -> None:
    def plain(request: httpx.Request) -> httpx.Response:
        error = httpx.ConnectError("[SSL: WRONG_VERSION_NUMBER] wrong version number", request=request)
        cause = ssl.SSLError(1, "[SSL: WRONG_VERSION_NUMBER] wrong version number")
        cause.reason = "WRONG_VERSION_NUMBER"
        error.__cause__ = cause
        raise error

    with pytest.raises(BatchServerError, match="use http://"):
        await run_batch([_item("seg-1")], OutputDir(tmp_path), "https://edge.test:8000", progress=lambda line: None,
                        _transport=httpx.MockTransport(plain))


def test_a_certificate_name_mismatch_says_so():
    from edgeserve.batch import _certificate_problem

    error = ssl.SSLCertVerificationError(1, "certificate verify failed")
    error.verify_code, error.verify_message = 64, "IP address mismatch, certificate is not valid for '10.0.0.9'."
    wrapped = httpx.ConnectError("failed")
    wrapped.__cause__ = error
    problem = _certificate_problem(wrapped, "https://10.0.0.9:8000")
    assert "IP address mismatch" in problem and "host name or address the certificate lists" in problem
    assert _certificate_problem(httpx.ConnectError("refused"), "https://x") is None


@pytest.mark.parametrize("server", ["http://h:8000?", "http://h:8000#", "http://h:8000/?", "http://h:8000/#x"])
def test_a_server_url_with_a_query_or_fragment_is_refused(server: str) -> None:
    from edgeserve.batch import _normalize_server

    assert _normalize_server(server) is None
    assert _normalize_server("HTTPS://h:8000/v1") == "https://h:8000"


async def test_untrusted_certificate_stops_the_run_with_a_hint(tmp_path: Path, sleeps: list[float]) -> None:
    out = OutputDir(tmp_path)
    transport = httpx.MockTransport(_untrusted_certificate)
    with pytest.raises(BatchServerError, match="--ca-cert"):
        await run_batch([_item("seg-1"), _item("seg-2")], out, SERVER, progress=lambda line: None,
                        _transport=transport)
    assert sleeps == []  # a wrong --ca-cert used to cost 31 s of retries per item
    assert out.read_record("seg-1") is None and _results(tmp_path) == []


async def test_untrusted_certificate_on_a_request_is_not_retried(tmp_path: Path, sleeps: list[float]) -> None:
    # The probe got through (e.g. another proxy path), a request did not: recorded once, never retried.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": [{"id": "m", "max_batch_size": 1}]})
        return _untrusted_certificate(request)

    out = OutputDir(tmp_path)
    lines: list[str] = []
    await run_batch([_item("seg-1")], out, SERVER, progress=lines.append, _transport=httpx.MockTransport(handler))
    assert sleeps == []
    message = f"ConnectError: {_CERT_FAILURE}"
    error = {"message": message, "type": "connection_error", "param": None, "code": "connection_error"}
    assert out.read_record("seg-1") == _failed("seg-1", None, error, 1)
    assert any(line.startswith("[1/1] ") and line.endswith(f"failed: {message}") for line in lines)


async def test_undecodable_response_is_not_retried(tmp_path: Path, sleeps: list[float]) -> None:
    server = FakeServer({"seg-1": [httpx.DecodingError]})
    out = OutputDir(tmp_path)
    _, lines = await _run([_item("seg-1")], out, server, concurrency=1)
    assert server.attempts("seg-1") == 1 and sleeps == []
    record = out.read_record("seg-1")
    assert (record["status"], record["http_status"], record["error"]["code"]) == ("failed", None, "invalid_response")
    [line] = _results(tmp_path)
    assert line["response"] is None
    assert line["error"] == {"code": "invalid_response", "message": "DecodingError: connection refused"}
    assert lines[-1] == "[1/1] seg-1 failed: DecodingError: connection refused"


async def test_unsendable_request_is_recorded_without_retry(tmp_path: Path, sleeps: list[float]) -> None:
    item = BatchItem("seg-1", CHAT, {"messages": MESSAGES, "temperature": float("nan")}, 1)  # not valid JSON
    server = FakeServer()
    out = OutputDir(tmp_path)
    await _run([item], out, server, concurrency=1)
    assert server.posts == [] and sleeps == []
    record = out.read_record("seg-1")
    assert (record["status"], record["http_status"], record["error"]["code"]) == ("failed", None, "invalid_request")
    assert _results(tmp_path)[0]["error"]["code"] == "invalid_request"


async def test_broken_progress_output_does_not_stop_the_batch(tmp_path: Path) -> None:
    def broken(line: str) -> None:
        raise BrokenPipeError(32, "Broken pipe")

    out = OutputDir(tmp_path)
    transport = FakeServer().transport
    await run_batch([_item("a"), _item("b")], out, SERVER, concurrency=1, progress=broken, _transport=transport)
    assert out.completed_ids() == {"a", "b"}


async def test_unencodable_progress_text_does_not_stop_the_batch(tmp_path: Path) -> None:
    lines: list[str] = []

    def ascii_console(line: str) -> None:
        line.encode("ascii")  # what print() raises for a stdout with an ASCII encoding
        lines.append(line)

    error = {**CONTEXT_ERROR, "message": "le prompt dépasse la limite"}
    server = FakeServer({"seg-1": [respond(400, json_body={"error": error})]})
    out = OutputDir(tmp_path)
    items = [_item("seg-1"), _item("seg-2")]
    await run_batch(items, out, SERVER, concurrency=1, progress=ascii_console, _transport=server.transport)
    assert out.completed_ids() == {"seg-2"}
    assert "[1/2] seg-1 failed: HTTP 400 context_length_exceeded: le prompt d\\xe9passe la limite" in lines
    assert lines[-1].startswith("[2/2] seg-2 ok")


# --- run_batch: resuming and invalid lines -----------------------------------------------------


async def test_resume_skips_completed_ids(tmp_path: Path) -> None:
    out = OutputDir(tmp_path)
    out.write_record(_server_record("seg-1"))
    out.write_record({"object": "edgeserve.record", "custom_id": "seg-2", "status": "failed", "http_status": 503})
    server = FakeServer()
    summary, lines = await _run([_item("seg-1"), _item("seg-2"), _item("seg-3")], out, server, concurrency=1)
    assert [request.headers["x-request-id"] for request in server.posts] == ["seg-2", "seg-3"]
    assert (summary["input_items"], summary["skipped_completed"]) == (3, 1)
    assert summary["request_counts"] == {"total": 3, "completed": 3, "failed": 0, "cancelled": 0}
    assert lines[0] == f"sending 2 requests to {SERVER} (concurrency 1, 1 already completed)"
    assert lines[1].startswith("[1/2] seg-2 ok") and lines[2].startswith("[2/2] seg-3 ok")


async def test_nothing_to_send_makes_no_requests(tmp_path: Path) -> None:
    out = OutputDir(tmp_path)
    out.write_record(_server_record("seg-1"))
    server = FakeServer()
    summary, lines = await _run([_item("seg-1")], out, server)
    assert server.posts == [] and server.model_gets == 0
    assert lines == [f"nothing to send: 1 of 1 request already completed in {tmp_path}"]
    assert summary["concurrency"] is None
    assert summary["request_counts"]["completed"] == 1
    assert _run_json(tmp_path) == summary
    assert _results(tmp_path) == []


async def test_invalid_lines_are_recorded_as_failed(tmp_path: Path) -> None:
    out = OutputDir(tmp_path)
    out.write_record(_server_record("old-done"))
    invalid = [
        {"line_no": 3, "custom_id": "bad-1", "message": '"stream": true is not supported in batch mode'},
        {"line_no": 4, "custom_id": None, "message": "invalid JSON: Expecting value"},
        {"line_no": 5, "custom_id": "old-done", "message": "edited after it completed"},
    ]
    server = FakeServer()
    summary, lines = await _run([_item("seg-1")], out, server, concurrency=1, invalid=invalid)

    message = invalid[0]["message"]
    error = {"message": message, "type": "invalid_request_error", "param": None, "code": "invalid_batch_line"}
    assert out.read_record("bad-1") == {**_failed("bad-1", None, error, 0), "line_no": 3}
    assert out.read_record("old-done") == _server_record("old-done")  # a completed result is never overwritten
    failures = [(line["custom_id"], line["response"], line["error"]) for line in _results(tmp_path) if line["error"]]
    assert failures == [("bad-1", None, {"code": "invalid_batch_line", "message": message})]
    assert summary["invalid_lines"] == invalid
    assert summary["input_items"] == 1
    assert summary["request_counts"] == {"total": 3, "completed": 2, "failed": 1, "cancelled": 0}
    assert [request.headers["x-request-id"] for request in server.posts] == ["seg-1"]
    assert not any("bad-1" in line for line in lines)  # main() reports invalid lines, not progress()


# --- run_batch: interruption and local failures ------------------------------------------------


async def test_cancellation_lets_in_flight_requests_finish(tmp_path: Path) -> None:
    release = asyncio.Event()
    started: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        custom_id = request.headers["x-request-id"]
        started.append(custom_id)
        await release.wait()
        return _ok(custom_id)

    out = OutputDir(tmp_path)
    lines: list[str] = []
    items = [_item(f"seg-{i}") for i in range(5)]
    task = asyncio.create_task(
        run_batch(items, out, SERVER, concurrency=2, progress=lines.append, _transport=httpx.MockTransport(handler))
    )
    await _until(lambda: len(started) == 2)
    task.cancel()
    await _until(lambda: any(line.startswith("interrupted") for line in lines))
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert sorted(started) == ["seg-0", "seg-1"]
    assert out.completed_ids() == {"seg-0", "seg-1"}
    run = _run_json(tmp_path)
    assert run["interrupted"] is True and run["input_items"] == 5
    assert run["request_counts"] == {"total": 2, "completed": 2, "failed": 0, "cancelled": 0}
    assert "interrupted: letting 2 in-flight request(s) finish (interrupt again to abort them)" in lines


async def test_second_cancellation_aborts_in_flight_requests(tmp_path: Path) -> None:
    started: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        started.append(request.headers["x-request-id"])
        await asyncio.Event().wait()  # never answers
        raise AssertionError("unreachable")

    out = OutputDir(tmp_path)
    lines: list[str] = []
    items = [_item(f"seg-{i}") for i in range(4)]
    task = asyncio.create_task(
        run_batch(items, out, SERVER, concurrency=2, progress=lines.append, _transport=httpx.MockTransport(handler))
    )
    await _until(lambda: len(started) == 2)
    task.cancel()
    await _until(lambda: any(line.startswith("interrupted") for line in lines))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)
    assert len(started) == 2
    assert list((tmp_path / "records").iterdir()) == []  # aborted requests are left unrecorded
    run = _run_json(tmp_path)
    assert run["interrupted"] is True and run["request_counts"]["total"] == 0


async def test_cancellation_ends_backoff_waits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def never(seconds: float) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(batch, "_sleep", never)
    busy = {"error": {"message": "queue full", "code": "server_busy"}}
    server = FakeServer({"seg-0": [respond(503, json_body=busy, headers={"Retry-After": "30"})] * 6})
    out = OutputDir(tmp_path)
    lines: list[str] = []
    items = [_item("seg-0"), _item("seg-1")]
    task = asyncio.create_task(
        run_batch(items, out, SERVER, concurrency=1, progress=lines.append, _transport=server.transport)
    )
    await _until(lambda: server.attempts("seg-0") == 1)
    for _ in range(20):  # let it reach the backoff wait
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)
    assert out.read_record("seg-0") == _failed("seg-0", 503, busy["error"], 1)
    assert server.attempts("seg-1") == 0
    assert lines[-1] == "[1/2] seg-0 failed: HTTP 503 server_busy: queue full"


async def test_record_write_failure_stops_the_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = OutputDir(tmp_path)

    def disk_full(record: dict) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(out, "write_record", disk_full)
    server = FakeServer()
    with pytest.raises(OSError, match="No space"):
        await _run([_item(f"seg-{i}") for i in range(3)], out, server, concurrency=1)
    assert len(server.posts) == 1
    assert _run_json(tmp_path)["interrupted"] is False


async def test_invalid_line_record_failure_sends_nothing_but_writes_run_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = OutputDir(tmp_path)

    def disk_full(record: dict) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(out, "write_record", disk_full)
    server = FakeServer()
    with pytest.raises(OSError, match="No space"):
        await _run([_item("seg-1")], out, server, invalid=_INVALID_WITH_ID)
    assert server.posts == [] and server.model_gets == 0
    run = _run_json(tmp_path)
    assert run["interrupted"] is False and run["invalid_lines"] == _INVALID_WITH_ID


async def test_run_summary_reads_one_record_at_a_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Records carry per-token arrays: loading all of them at once could exhaust a Jetson's memory."""
    out = OutputDir(tmp_path)
    for i in range(3):
        out.write_record(_server_record(f"seg-{i}"))  # all completed: nothing to send
    reads: list[str] = []
    real_read = out.read_record

    def counting_read(custom_id: str) -> dict | None:
        reads.append(custom_id)
        return real_read(custom_id)

    real_summarize = batch.summarize

    def lazy_summarize(records: Any) -> dict:
        iterator = iter(records)
        first = next(iterator)
        assert reads == ["seg-0"]  # the others are read only when summarize() gets to them
        return real_summarize([first, *iterator])

    monkeypatch.setattr(out, "read_record", counting_read)
    monkeypatch.setattr(batch, "summarize", lazy_summarize)
    summary, _ = await _run([_item(f"seg-{i}") for i in range(3)], out, FakeServer())
    assert summary["request_counts"]["completed"] == 3
    assert reads == ["seg-0", "seg-1", "seg-2"]


async def test_cancellation_while_probing_still_writes_run_json(tmp_path: Path) -> None:
    probing = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            probing.set()
            await asyncio.Event().wait()  # never answers
        raise AssertionError("nothing may be sent after the interruption")

    out = OutputDir(tmp_path)
    task = asyncio.create_task(
        run_batch([_item("seg-1")], out, SERVER, progress=lambda line: None, _transport=httpx.MockTransport(handler))
    )
    await asyncio.wait_for(probing.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)
    run = _run_json(tmp_path)
    assert run["interrupted"] is True and run["concurrency"] is None
    assert run["request_counts"]["total"] == 0 and _results(tmp_path) == []


# --- main --------------------------------------------------------------------------------------


def test_main_all_completed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    server = FakeServer(max_batch_size=1)
    _patch_network(monkeypatch, server)
    monkeypatch.setenv("EDGESERVE_API_KEY", " key-123 \n")
    path = _input(tmp_path, {"id": "seg-1", "messages": MESSAGES}, {"id": "seg-2", "prompt": "x"})
    out_dir = tmp_path / "out"
    assert main([str(path), "--output-dir", str(out_dir), "--server", SERVER]) == 0
    assert {request.headers["authorization"] for request in server.posts} == {"Bearer key-123"}
    assert {str(request.url) for request in server.posts} == {SERVER + CHAT, SERVER + COMPLETIONS}
    assert OutputDir(out_dir).completed_ids() == {"seg-1", "seg-2"}
    captured = capsys.readouterr()
    assert captured.out.count(" ok 8.4 s") == 2
    assert f"2/2 completed, 0 failed; results in {out_dir}" in captured.out
    assert captured.err == ""


def test_main_some_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    _patch_network(monkeypatch, FakeServer({"seg-2": [respond(400, json_body={"error": CONTEXT_ERROR})]}))
    path = _input(tmp_path, {"id": "seg-1", "prompt": "x"}, {"id": "seg-2", "prompt": "y"})
    assert main([str(path), "--output-dir", str(tmp_path / "out"), "--server", SERVER]) == 1
    assert "1/2 completed, 1 failed" in capsys.readouterr().out


def test_main_reports_and_records_invalid_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    server = FakeServer()
    _patch_network(monkeypatch, server)
    good = json.dumps({"id": "seg-1", "prompt": "x"})
    path = _write_input(tmp_path, good, json.dumps({"id": "bad-1", "prompt": "x", "stream": True}), "{broken")
    out_dir = tmp_path / "out"
    assert main([str(path), "--output-dir", str(out_dir), "--server", SERVER]) == 1
    err = capsys.readouterr().err
    assert f"{path}:2 (bad-1): invalid line: " in err
    assert f"{path}:3: invalid line: invalid JSON" in err
    assert OutputDir(out_dir).read_record("bad-1")["error"]["code"] == "invalid_batch_line"
    assert [request.headers["x-request-id"] for request in server.posts] == ["seg-1"]
    run = _run_json(out_dir)
    assert [entry["line_no"] for entry in run["invalid_lines"]] == [2, 3]
    assert run["request_counts"] == {"total": 2, "completed": 1, "failed": 1, "cancelled": 0}


def test_main_duplicate_ids_send_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    server = FakeServer()
    _patch_network(monkeypatch, server)
    path = _input(tmp_path, {"id": "seg-1", "prompt": "x"}, {"id": "seg-1", "prompt": "y"})
    out_dir = tmp_path / "out"
    assert main([str(path), "--output-dir", str(out_dir), "--server", SERVER]) == 2
    assert server.posts == [] and server.model_gets == 0
    assert not out_dir.exists()
    assert "duplicate IDs" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra",
    [
        ["--concurrency", "0"],
        ["--concurrency", "many"],
        ["--server", "ftp://edge.test"],
        ["--server", "edge.test:8000"],
        ["--server", "http://edge.test:port"],
        ["--server", "http://edge.test:8000?x=1"],
        ["--server", "http://edge\x01test:8000"],  # urlsplit accepts these; httpx raised InvalidURL in the probe
        ["--server", "http://☃️.example"],
        ["--ca-cert", "MISSING"],
        ["--unknown-flag"],
    ],
)
def test_main_bad_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: list[str]) -> None:
    server = FakeServer()
    _patch_network(monkeypatch, server)
    path = _input(tmp_path, {"id": "seg-1", "prompt": "x"})
    extra = [str(tmp_path / "missing.pem") if arg == "MISSING" else arg for arg in extra]
    assert main([str(path), "--output-dir", str(tmp_path / "out"), *extra]) == 2
    assert server.posts == [] and server.model_gets == 0


def test_main_bad_input_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    server = FakeServer()
    _patch_network(monkeypatch, server)
    path = _input(tmp_path, {"id": "seg-1", "prompt": "x"})
    assert main([str(path)]) == 2  # --output-dir is required
    assert main([str(tmp_path / "missing.jsonl"), "--output-dir", str(tmp_path / "out")]) == 2
    assert main([str(path), "--output-dir", str(path)]) == 2  # the output folder is a file
    empty = tmp_path / "empty.jsonl"
    empty.write_text("# nothing yet\n\n")
    assert main([str(empty), "--output-dir", str(tmp_path / "out")]) == 2
    assert "contains no requests" in capsys.readouterr().err
    assert server.posts == [] and not (tmp_path / "out").exists()


def test_main_rejects_an_unusable_api_key_without_echoing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    server = FakeServer()
    _patch_network(monkeypatch, server)
    monkeypatch.setenv("EDGESERVE_API_KEY", "s3cr3t\nsecond-line")  # e.g. "$(cat key-file)" with two lines
    path = _input(tmp_path, {"id": "seg-1", "prompt": "x"})
    assert main([str(path), "--output-dir", str(tmp_path / "out"), "--server", SERVER]) == 2
    captured = capsys.readouterr()
    assert "EDGESERVE_API_KEY" in captured.err
    assert "s3cr3t" not in captured.err + captured.out
    assert server.posts == [] and server.model_gets == 0 and not (tmp_path / "out").exists()


def test_main_help(capsys: pytest.CaptureFixture) -> None:
    assert main(["--help"]) == 0
    assert "edgeserve batch" in capsys.readouterr().out


def test_main_accepts_an_openai_style_base_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeServer()
    _patch_network(monkeypatch, server)
    path = _input(tmp_path, {"id": "seg-1", "messages": MESSAGES})
    assert main([str(path), "--output-dir", str(tmp_path / "out"), "--server", SERVER + "/v1/"]) == 0
    assert [str(request.url) for request in server.posts] == [SERVER + CHAT]
    assert _run_json(tmp_path / "out")["server"] == SERVER


def test_main_ca_cert_option(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    _patch_network(monkeypatch, FakeServer())
    path = _input(tmp_path, {"id": "seg-1", "prompt": "x"})
    https = "https://edge.test:8443"
    assert main([str(path), "--output-dir", str(tmp_path / "out"), "--server", https, "--ca-cert", certifi.where()]) == 0
    missing = tmp_path / "missing.pem"
    assert main([str(path), "--output-dir", str(tmp_path / "out2"), "--server", https, "--ca-cert", str(missing)]) == 2
    assert f"cannot use CA certificate {missing}" in capsys.readouterr().err


def _network_settings(tmp_path: Path, port: int = 8443) -> Path:
    """A settings file like the one edgeserve setup --network writes (the CA is a real PEM bundle)."""
    folder = config.config_dir()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "edgeserve-ca.crt").write_bytes(Path(certifi.where()).read_bytes())
    config.write({"EDGESERVE_HOST": "0.0.0.0", "EDGESERVE_PORT": str(port), "EDGESERVE_API_KEY": "file-key",
                  "EDGESERVE_TLS_CERT": str(folder / "server.crt"), "EDGESERVE_TLS_KEY": str(folder / "server.key")})
    return folder / "edgeserve-ca.crt"


def test_main_defaults_to_this_machines_server_key_and_ca(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeServer()
    _patch_network(monkeypatch, server)
    ca = _network_settings(tmp_path)
    used: list[object] = []
    original = batch._ssl_verify
    monkeypatch.setattr(batch, "_ssl_verify", lambda verify: used.append(verify) or original(verify))
    path = _input(tmp_path, {"id": "seg-1", "prompt": "x"})
    assert main([str(path), "--output-dir", str(tmp_path / "out")]) == 0
    assert [str(request.url) for request in server.posts] == ["https://localhost:8443" + COMPLETIONS]
    assert [request.headers["authorization"] for request in server.posts] == ["Bearer file-key"]
    assert str(ca) in used


def test_main_never_sends_this_machines_key_to_another_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeServer()
    _patch_network(monkeypatch, server)
    _network_settings(tmp_path)
    path = _input(tmp_path, {"id": "seg-1", "prompt": "x"})
    assert main([str(path), "--output-dir", str(tmp_path / "out"), "--server", SERVER]) == 0
    assert [str(request.url) for request in server.posts] == [SERVER + COMPLETIONS]
    assert "authorization" not in server.posts[0].headers
    monkeypatch.setenv("EDGESERVE_API_KEY", "env-key")  # the caller's own choice: always sent
    assert main([str(path), "--output-dir", str(tmp_path / "out2"), "--server", SERVER]) == 0
    assert server.posts[-1].headers["authorization"] == "Bearer env-key"


def test_main_ctrl_c_finishes_in_flight_and_exits_130(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    posts: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        custom_id = request.headers["x-request-id"]
        posts.append(custom_id)
        if custom_id == "seg-0":
            signal.raise_signal(signal.SIGINT)  # Ctrl-C while this request is in flight
            for _ in range(50):
                await asyncio.sleep(0)
        return _ok(custom_id)

    _patch_network(monkeypatch, handler)
    path = _input(tmp_path, *({"id": f"seg-{i}", "prompt": "x"} for i in range(3)))
    out_dir = tmp_path / "out"
    before = signal.getsignal(signal.SIGINT)
    assert main([str(path), "--output-dir", str(out_dir), "--server", SERVER, "--concurrency", "1"]) == 130
    assert signal.getsignal(signal.SIGINT) is before
    assert posts == ["seg-0"]
    assert OutputDir(out_dir).completed_ids() == {"seg-0"}
    run = _run_json(out_dir)
    assert run["interrupted"] is True and run["request_counts"]["completed"] == 1
    assert "interrupted" in capsys.readouterr().err


def test_main_reports_os_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    _patch_network(monkeypatch, FakeServer())

    def disk_full(self: OutputDir, record: dict) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(OutputDir, "write_record", disk_full)
    path = _input(tmp_path, {"id": "seg-1", "prompt": "x"})
    assert main([str(path), "--output-dir", str(tmp_path / "out"), "--server", SERVER]) == 1
    assert "No space left on device" in capsys.readouterr().err
    assert (tmp_path / "out" / "run.json").exists()
