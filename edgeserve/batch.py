"""Batch client: run a JSONL file of ID'd requests against an edgeserve server and save every result.

Usable from any machine that can reach the server (it needs only this package, not a GPU). Without
``--server`` it runs against this machine's server with the address, API key and CA certificate from
this machine's settings file (see edgeserve.config.client_target).

Input lines (one JSON object per line; blank lines and lines starting with "#" are ignored):

* OpenAI Batch format::

    {"custom_id": "seg-001", "method": "POST", "url": "/v1/chat/completions", "body": {"messages": [...], ...}}

* Short forms (``id`` is accepted as an alias of ``custom_id``; all other keys are the request body)::

    {"id": "seg-002", "messages": [...], "max_tokens": 512}      -> /v1/chat/completions
    {"id": "seg-003", "prompt": "<|begin_of_text|>..."}          -> /v1/completions

Output (via edgeserve.records.OutputDir): records/<id>.json (the server's full record),
results.jsonl (OpenAI BatchRequestOutput lines) and run.json (batch metadata + summarize()).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import email.utils
import functools
import json
import math
import os
import secrets
import signal
import ssl
import sys
import time
import urllib.parse
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

from edgeserve import config
from edgeserve.records import OutputDir, summarize

_CHAT_URL = "/v1/chat/completions"
_COMPLETIONS_URL = "/v1/completions"
_ID_KEYS = ("custom_id", "id")
_BATCH_FORMAT_KEYS = frozenset({"custom_id", "id", "method", "url", "body"})
_MAX_ID_CHARS = 512
_MAX_ATTEMPTS = 6
_MAX_DELAY_S = 60.0
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_TIMEOUT = httpx.Timeout(1800.0, connect=10.0)
_DEFAULT_CONCURRENCY = 2
_MAX_ERROR_CHARS = 4096
_MAX_PROGRESS_CHARS = 200
_MAX_LISTED_DUPLICATES = 20
_STATUS_TEXT = {499: "cancelled by the server"}  # for error responses without a body (current servers send 503)
# Anything a broken or unexpected /v1/models answer can raise (deeply nested JSON raises RecursionError).
_PROBE_ERRORS = (httpx.HTTPError, httpx.InvalidURL, ValueError, LookupError, TypeError, AttributeError, RecursionError)
_RECORD_HEADER = {"object": "edgeserve.record", "schema_version": "1.0"}
_JSON_TYPES = {
    bool: "a boolean",
    int: "a number",
    float: "a number",
    str: "a string",
    list: "an array",
    dict: "an object",
    type(None): "null",
}
_EXIT_OK, _EXIT_FAILED, _EXIT_USAGE, _EXIT_INTERRUPTED = 0, 1, 2, 130

_sleep = asyncio.sleep  # module-level so tests can skip the backoff waits


class BatchFileError(ValueError):
    """The input file cannot be run as a whole (duplicate IDs, unreadable file)."""


class BatchServerError(ValueError):
    """The server cannot be used at all with these settings (key refused, certificate not trusted, wrong scheme):
    nothing is sent, since every request would fail the same way."""


@dataclass(frozen=True)
class BatchItem:
    custom_id: str
    url: str  # "/v1/chat/completions" | "/v1/completions"
    body: dict  # request body without any ID field
    line_no: int  # 1-based line number in the input file


def parse_batch_file(path: Path) -> tuple[list[BatchItem], list[dict]]:
    """Parse and validate an input file.

    Returns (items, invalid) where ``invalid`` holds {"line_no", "custom_id" (or None), "message"} for lines that
    are not valid JSON objects, lack an ID, use an unsupported method/url, have both or neither of
    messages/prompt, or set "stream": true. IDs must be 1-512 printable ASCII characters.
    Raises BatchFileError when the file is unreadable or an ID appears more than once (lists the duplicates).

    Also invalid: IDs with a leading or trailing space (HTTP drops it from the X-Request-Id header), NaN or
    out-of-range numbers, unknown top-level keys in the Batch format, a url that does not match the body,
    "messages" that is not an array and "prompt" that is neither a string nor an array. In the Batch format a
    missing "method" means POST and a missing "url" is inferred from the body. An invalid line whose ID is
    well-formed keeps that ID and takes part in the duplicate check (its failure record would collide). A UTF-8
    byte order mark at the start of a line is ignored (editors write one; concatenated files carry it inside).
    """
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise BatchFileError(f"cannot read {path}: {exc.strerror or exc}") from exc
    items: list[BatchItem] = []
    invalid: list[dict] = []
    lines_by_id: dict[str, list[int]] = {}
    # Split on "\n" only: str.splitlines() would also split on characters JSON allows inside strings.
    for line_no, raw in enumerate(data.split(b"\n"), start=1):
        try:
            text = raw.decode("utf-8-sig").strip()  # -sig: drop a leading byte order mark
        except UnicodeDecodeError:
            invalid.append({"line_no": line_no, "custom_id": None, "message": "line is not valid UTF-8"})
            continue
        if not text or text.startswith("#"):
            continue
        custom_id, parsed = _parse_line(text, line_no)
        if custom_id is not None:
            lines_by_id.setdefault(custom_id, []).append(line_no)
        if isinstance(parsed, BatchItem):
            items.append(parsed)
        else:
            invalid.append({"line_no": line_no, "custom_id": custom_id, "message": parsed})
    duplicates = [(cid, numbers) for cid, numbers in lines_by_id.items() if len(numbers) > 1]
    if duplicates:
        shown = [
            f"{cid!r} (lines {', '.join(map(str, numbers))})" for cid, numbers in duplicates[:_MAX_LISTED_DUPLICATES]
        ]
        more = f"; and {len(duplicates) - len(shown)} more" if len(duplicates) > len(shown) else ""
        raise BatchFileError(f"{path}: duplicate IDs: {'; '.join(shown)}{more}")
    return items, invalid


async def run_batch(
    items: list[BatchItem],
    out: OutputDir,
    server: str,
    api_key: str | None = None,
    verify: str | bool = True,
    concurrency: int | None = None,
    progress: Callable[[str], None] = print,
    *,
    invalid: Sequence[dict] = (),
    _transport: httpx.AsyncBaseTransport | None = None,
) -> dict:
    """Send every item that is not already completed in ``out`` and record the outcome. Returns the run summary.

    Protocol with the server:
    - concurrency defaults to max_batch_size + 1 from GET {server}/v1/models (data[0]["max_batch_size"]; 2 if absent).
    - each item: POST {server}{item.url}, JSON body = item.body + {"return_record": true}, headers
      {"X-Request-Id": custom_id, "Authorization": "Bearer <api_key>"} (no Authorization header when api_key is None);
      timeout: 10 s connect, 1800 s read.
    - 200: the body has a "record" key -> out.write_record(record); results line uses the body without "record".
    - 400/401/404/409/413/422: not retried; writes a record {"object": "edgeserve.record", "schema_version": "1.0",
      "custom_id", "status": "failed", "http_status", "error": body["error"] (or {"message": text})} and a results line.
    - 429/500/502/503/504 and connection errors: retried up to 6 attempts with exponential backoff (1, 2, 4, 8, 16 s,
      capped at 60 s), honouring Retry-After (seconds) when present; afterwards recorded as failed like above.
    - results.jsonl lines: {"id": "batch_req_<24 hex>", "custom_id", "response": {"status_code", "request_id":
      <record attempt_id or None>, "body": <response JSON>} or null, "error": null or {"code", "message"}}; HTTP
      responses go in "response", network failures in "error" with code "connection_error".
    - progress() gets one line per finished item, e.g. "[12/300] seg-012 ok 8.4 s (ttft 2.0 s, 24.6 tok/s)".
    - on asyncio cancellation (Ctrl-C) stop starting new items, let in-flight ones finish, still write run.json.
    - run.json = summarize(all records for the input's IDs found in out) + {"object": "batch_run", "server",
      "started_at", "finished_at" (unix seconds), "input_items", "skipped_completed", "invalid_lines"}.

    Details: other non-200 statuses are not retried either, nor is a server certificate the client does not trust
    (a connection error retrying cannot fix); Retry-After may also be an HTTP date and is capped at 60 s; a 200
    without "record" (a server without return_record) is saved as a minimal completed record holding the response.
    Failed records also carry "attempts". A first cancellation also ends pending backoff waits (the item is recorded
    with its last outcome); a second one cancels the in-flight requests (left unrecorded). After run.json is written
    the cancellation is re-raised. If writing a record fails, no new items start and the error is raised after
    run.json. progress() additionally gets a start line and an interruption notice. Before anything is written or
    sent: ValueError for a ``server`` that is not an http(s) URL or an ``api_key`` that is not printable ASCII
    without leading/trailing spaces (the key is never echoed), BatchFileError for duplicate IDs in ``items``.

    ``invalid`` takes parse_batch_file()'s invalid lines: those with an ID are recorded as failed (unless that ID
    is already completed) and count in the summary; run.json lists all of them under "invalid_lines" (a list),
    while "input_items" counts ``items``. run.json also has "concurrency" and "interrupted". ``_transport``
    replaces the network (tests only).
    """
    if concurrency is not None and concurrency < 1:
        raise ValueError(f"concurrency must be at least 1, got {concurrency}")
    server = _check_server(server)
    if api_key and not _header_safe(api_key):
        # h11 would reject the header with an error quoting it, and that text would end up in every record.
        raise ValueError("api_key must be printable ASCII without leading or trailing spaces")
    _check_unique(items)
    ssl_verify = _ssl_verify(verify)
    started_at = time.time()
    completed = out.completed_ids()
    pending = [item for item in items if item.custom_id not in completed]
    skipped = len(items) - len(pending)
    errors: list[BaseException] = []
    try:
        _record_invalid(out, invalid, completed)
    except Exception as exc:  # e.g. disk full: send nothing, but still write run.json
        errors.append(exc)
    if items and not pending and not errors:
        noun = "request" if len(items) == 1 else "requests"
        _say(progress, f"nothing to send: {len(items)} of {len(items)} {noun} already completed in {out.root}")

    interrupted = False
    try:
        if pending and not errors:
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
            async with httpx.AsyncClient(
                verify=ssl_verify, timeout=_TIMEOUT, headers=headers, transport=_transport
            ) as client:
                probed = await _probe_concurrency(client, server, progress, requested=concurrency)
                if concurrency is None:
                    concurrency = probed
                note = f", {skipped} already completed" if skipped else ""
                noun = "request" if len(pending) == 1 else "requests"
                _say(progress, f"sending {len(pending)} {noun} to {server} (concurrency {concurrency}{note})")
                runner = _Runner(client, out, server, progress, len(pending))
                interrupted, errors = await runner.run(pending, concurrency)
    except asyncio.CancelledError:  # while probing or closing the client: nothing is in flight
        interrupted = True

    input_ids = [item.custom_id for item in items]
    input_ids += [entry["custom_id"] for entry in invalid if isinstance(entry.get("custom_id"), str)]
    # A generator, not a list: records hold per-token arrays, and thousands of them would not fit in memory.
    records = (record for cid in dict.fromkeys(input_ids) if (record := out.read_record(cid)) is not None)
    run = {
        "object": "batch_run",
        "server": server,
        "started_at": started_at,
        "finished_at": time.time(),
        "input_items": len(items),
        "skipped_completed": skipped,
        "invalid_lines": [dict(entry) for entry in invalid],
        "concurrency": concurrency,
        "interrupted": interrupted,
        **summarize(records),
    }
    try:
        out.write_run(run)
    except BaseException:
        with contextlib.suppress(OSError):
            out.close()
        raise
    out.close()
    if errors:
        raise errors[0]
    if interrupted:
        raise asyncio.CancelledError
    return run


def main(argv: list[str] | None = None) -> int:
    """``edgeserve batch INPUT --output-dir DIR [--server URL] [--ca-cert FILE] [--concurrency N]``.

    Without --server it uses this machine's server, API key and CA certificate (from the settings file); with
    --server the key comes from $EDGESERVE_API_KEY (this machine's key is sent only to this machine), and
    --ca-cert is the CA certificate for HTTPS with a private CA. Invalid lines are reported and recorded as
    failed; duplicate IDs abort before anything is sent. Exit codes: 0 all completed, 1 some failed, 2 bad
    input/arguments, 130 interrupted.

    ``argv`` excludes the program and the "batch" subcommand. A trailing "/v1" on --server is dropped (OpenAI
    base-URL style). Any invalid line makes the exit code at least 1; an empty input, an output folder that
    cannot be created or an API key that is not printable ASCII is bad input (2); an OS error during the run (e.g.
    disk full) exits 1.
    """
    try:
        return _main(argv)
    except KeyboardInterrupt:  # before the event loop runs (or where signal handlers cannot be installed)
        return _EXIT_INTERRUPTED


@dataclass
class _Outcome:
    """The last attempt for one item: an HTTP response or the exception that prevented one."""

    attempts: int
    response: httpx.Response | None = None
    error: BaseException | None = None


class _Runner:
    """Sends items over one client with up to ``concurrency`` in flight and records every outcome."""

    def __init__(
        self, client: httpx.AsyncClient, out: OutputDir, server: str, progress: Callable[[str], None], total: int
    ) -> None:
        self.client = client
        self.out = out
        self.server = server
        self.progress = progress
        self.total = total
        self.stop = asyncio.Event()
        self.finished = 0
        self.in_flight = 0

    async def run(self, pending: list[BatchItem], concurrency: int) -> tuple[bool, list[BaseException]]:
        """Process ``pending``; returns (interrupted, worker exceptions).

        A first cancellation stops new items and lets in-flight ones finish; a second one cancels those too.
        """
        queue = iter(pending)

        async def worker() -> None:
            while not self.stop.is_set():
                item = next(queue, None)
                if item is None:
                    return
                try:
                    await self.run_item(item)
                except Exception:
                    self.stop.set()  # e.g. the output folder became unwritable: start nothing new
                    raise

        tasks = [asyncio.create_task(worker()) for _ in range(min(concurrency, len(pending)))]
        interrupted = False
        while True:
            try:
                # asyncio.wait (unlike gather) leaves the tasks running when this coroutine is cancelled.
                await asyncio.wait(tasks)
                break
            except asyncio.CancelledError:
                if interrupted:
                    for task in tasks:
                        task.cancel()
                    continue
                interrupted = True
                self.stop.set()
                if self.in_flight:
                    _say(
                        self.progress,
                        f"interrupted: letting {self.in_flight} in-flight request(s) finish "
                        "(interrupt again to abort them)",
                    )
        errors = [error for task in tasks if not task.cancelled() and (error := task.exception()) is not None]
        return interrupted, errors

    async def run_item(self, item: BatchItem) -> None:
        self.in_flight += 1
        started = time.monotonic()
        try:
            outcome = await self._send(item)
        finally:
            self.in_flight -= 1
        record, line, text = _describe(item, outcome, time.monotonic() - started)
        self.out.write_record(record)
        self.out.append_result(line)
        self.finished += 1
        _say(self.progress, f"[{self.finished}/{self.total}] {item.custom_id} {text}")

    async def _send(self, item: BatchItem) -> _Outcome:
        url = self.server + item.url
        body = {**item.body, "return_record": True}
        headers = {"X-Request-Id": item.custom_id}
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await self.client.post(url, json=body, headers=headers)
            except (httpx.UnsupportedProtocol, httpx.LocalProtocolError) as exc:  # not transient
                return _Outcome(attempt, error=exc)
            except httpx.TransportError as exc:  # refused, reset, timed out, closed mid-response
                if _caused_by(exc, ssl.SSLCertVerificationError):  # untrusted or wrong certificate: not transient
                    return _Outcome(attempt, error=exc)
                outcome, delay = _Outcome(attempt, error=exc), _backoff_s(attempt)
            except (httpx.RequestError, httpx.InvalidURL, ValueError, TypeError) as exc:
                # An undecodable response (e.g. broken gzip from a proxy) or a request that could not be built.
                return _Outcome(attempt, error=exc)
            else:
                outcome = _Outcome(attempt, response=response)
                if response.status_code not in _RETRY_STATUSES:
                    return outcome
                delay = _retry_delay_s(response, attempt)
            if attempt >= _MAX_ATTEMPTS or self.stop.is_set():
                return outcome
            if attempt == 1:
                reason = (f"HTTP {outcome.response.status_code}" if outcome.response is not None
                          else _one_line(_exception_text(outcome.error)))
                _say(self.progress, f"{item.custom_id}: {reason}; retrying in {delay:.0f} s "
                                    f"(up to {_MAX_ATTEMPTS} attempts)")
            await _pause(delay, self.stop)
            if self.stop.is_set():  # interrupted during the wait: give up instead of retrying
                return outcome


def _describe(item: BatchItem, outcome: _Outcome, elapsed_s: float) -> tuple[dict, dict, str]:
    """The record, the results.jsonl line and the progress text for one finished item."""
    tries = f" after {outcome.attempts} attempts" if outcome.attempts > 1 else ""
    response = outcome.response
    if response is None:
        code = _exception_code(outcome.error)
        message = _exception_text(outcome.error)
        error = {"message": message, "type": code, "param": None, "code": code}
        record = _failed_record(item.custom_id, None, error, outcome.attempts)
        line = _result_line(item.custom_id, None, {"code": code, "message": message})
        return record, line, f"failed: {_one_line(message)}{tries}"

    status = response.status_code
    body = _json_body(response)
    if status == 200 and isinstance(body, dict):
        rest = {key: value for key, value in body.items() if key != "record"}
        record = body.get("record")
        if isinstance(record, dict):
            record = {**record, "custom_id": item.custom_id}  # the file name and resume key must match the input
        else:  # a server without return_record: keep what the response says
            record = {
                **_RECORD_HEADER,
                "custom_id": item.custom_id,
                "status": "completed",
                "http_status": 200,
                "usage": rest.get("usage"),
                "response": rest,
            }
        response_part = {"status_code": 200, "request_id": record.get("attempt_id"), "body": rest}
        return record, _result_line(item.custom_id, response_part, None), _finished_text(record, elapsed_s)

    if status == 200:
        message = "the server's 200 response is not a JSON object"
        error = {"message": message, "type": "invalid_response", "param": None, "code": "invalid_response"}
        response_part = {"status_code": 200, "request_id": None, "body": body}
        line = _result_line(item.custom_id, response_part, {"code": "invalid_response", "message": message})
        return _failed_record(item.custom_id, 200, error, outcome.attempts), line, f"failed: {message}"

    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        error = body["error"]
    else:
        text = response.text[:_MAX_ERROR_CHARS] or response.reason_phrase
        error = {"message": text or _STATUS_TEXT.get(status, "empty response body")}
    if not isinstance(body, dict):
        body = {"error": error}
    record = _failed_record(item.custom_id, status, error, outcome.attempts)
    line = _result_line(item.custom_id, {"status_code": status, "request_id": None, "body": body}, None)
    code = error.get("code") or error.get("type")
    label = f"HTTP {status} {code}" if code else f"HTTP {status}"
    return record, line, f"failed: {label}: {_one_line(str(error.get('message') or ''))}{tries}"


def _caused_by(exc: BaseException, kind: type[BaseException]) -> bool:
    """Whether ``exc`` or an exception in its cause/context chain is a ``kind`` (httpx wraps httpcore's errors)."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, kind):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def _exception_code(exc: BaseException | None) -> str:
    if isinstance(exc, httpx.TransportError):
        return "connection_error"
    if isinstance(exc, httpx.RequestError):  # the server answered, but not with a usable response
        return "invalid_response"
    return "invalid_request"


def _finished_text(record: dict, elapsed_s: float) -> str:
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    e2e_ms, ttft_ms, rate = (_finite(metrics.get(key)) for key in ("e2e_ms", "ttft_ms", "decode_tokens_per_s"))
    seconds = e2e_ms / 1000.0 if e2e_ms is not None else elapsed_s
    status = record.get("status")
    text = f"{'ok' if status == 'completed' else status or 'unknown'} {seconds:.1f} s"
    details = [f"ttft {ttft_ms / 1000.0:.1f} s"] if ttft_ms is not None else []
    if rate is not None:
        details.append(f"{rate:.1f} tok/s")
    return f"{text} ({', '.join(details)})" if details else text


def _failed_record(custom_id: str, http_status: int | None, error: dict, attempts: int, **extra: object) -> dict:
    return {
        **_RECORD_HEADER,
        "custom_id": custom_id,
        "status": "failed",
        "http_status": http_status,
        "error": error,
        "attempts": attempts,
        **extra,
    }


def _result_line(custom_id: str, response: dict | None, error: dict | None) -> dict:
    return {"id": f"batch_req_{secrets.token_hex(12)}", "custom_id": custom_id, "response": response, "error": error}


def _record_invalid(out: OutputDir, invalid: Sequence[dict], completed: set[str]) -> None:
    for entry in invalid:
        custom_id = entry.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id or custom_id in completed:
            continue  # without an ID there is nothing to file it under; never overwrite a completed result
        message = str(entry.get("message") or "invalid input line")
        error = {"message": message, "type": "invalid_request_error", "param": None, "code": "invalid_batch_line"}
        previous = out.read_record(custom_id)
        if previous and previous.get("error") == error and previous.get("line_no") == entry.get("line_no"):
            continue  # recorded by an earlier run of this file
        out.write_record(_failed_record(custom_id, None, error, 0, line_no=entry.get("line_no")))
        out.append_result(_result_line(custom_id, None, {"code": "invalid_batch_line", "message": message}))


async def _probe_concurrency(client: httpx.AsyncClient, server: str, progress: Callable[[str], None],
                             requested: int | None = None) -> int:
    """Concurrency from the server's max_batch_size (unless ``requested``); raises BatchServerError when no
    request could succeed."""
    fallback = f"; using concurrency {_DEFAULT_CONCURRENCY}" if requested is None else ""
    try:
        response = await client.get(f"{server}/v1/models")
        if response.status_code in (401, 403):
            raise BatchServerError(f"{server} refused the API key (HTTP {response.status_code}): set $EDGESERVE_API_KEY "
                                   "to the server's key (EDGESERVE_API_KEY in its ~/.config/edgeserve/edgeserve.env)")
        response.raise_for_status()
        models = response.json()["data"]
        size = models[0].get("max_batch_size") if models else None
    except BatchServerError:
        raise
    except httpx.TransportError as exc:
        if (problem := _certificate_problem(exc, server)) is not None:
            raise BatchServerError(problem) from exc
        if isinstance(exc, httpx.RemoteProtocolError) and server.startswith("http://"):
            raise BatchServerError(f"{server} closed the connection without an HTTP response; if it serves HTTPS, "
                                   "use https://") from exc
        if server.startswith("https://") and _ssl_reason(exc) == "WRONG_VERSION_NUMBER":
            raise BatchServerError(f"{server} does not answer TLS; if it serves plain HTTP, use http://") from exc
        reason = _one_line(_exception_text(exc))
        _say(progress, f"cannot read max_batch_size from {server}/v1/models ({reason}){fallback}")
        return _DEFAULT_CONCURRENCY
    except _PROBE_ERRORS as exc:
        reason = _one_line(_exception_text(exc))
        _say(progress, f"cannot read max_batch_size from {server}/v1/models ({reason}){fallback}")
        return _DEFAULT_CONCURRENCY
    if isinstance(size, int) and not isinstance(size, bool) and size >= 1:
        return size + 1
    return _DEFAULT_CONCURRENCY


def _certificate_problem(exc: BaseException, server: str) -> str | None:
    """What is wrong with the server's certificate, with the fix, if that is why ``exc`` happened."""
    error = exc
    while error is not None and not isinstance(error, ssl.SSLCertVerificationError):
        error = error.__cause__ or error.__context__
    if error is None:
        return None
    detail = getattr(error, "verify_message", None) or str(error)
    code = getattr(error, "verify_code", None)
    if code in (62, 64):  # X509_V_ERR_HOSTNAME_MISMATCH, X509_V_ERR_IP_ADDRESS_MISMATCH
        hint = ("connect with a host name or address the certificate lists (if the server's address changed, "
                "see README, Troubleshooting)")
    elif code == 10:  # X509_V_ERR_CERT_HAS_EXPIRED
        hint = "the server's certificate has expired: run edgeserve setup --network on the server and restart it"
    else:
        hint = ("pass --ca-cert with the server's edgeserve-ca.crt (from its ~/.config/edgeserve); see README, "
                "'Access from other machines'")
    return f"the certificate of {server} was not accepted ({detail}): {hint}"


def _ssl_reason(exc: BaseException) -> str | None:
    """The OpenSSL reason (e.g. WRONG_VERSION_NUMBER) of the SSLError behind ``exc``, if any."""
    error = exc
    while error is not None and not isinstance(error, ssl.SSLError):
        error = error.__cause__ or error.__context__
    return getattr(error, "reason", None) if error is not None else None


async def _pause(seconds: float, stop: asyncio.Event) -> None:
    """Sleep ``seconds`` or until ``stop`` is set, whichever comes first."""
    sleeper = asyncio.ensure_future(_sleep(seconds))
    stopper = asyncio.ensure_future(stop.wait())
    try:
        await asyncio.wait((sleeper, stopper), return_when=asyncio.FIRST_COMPLETED)
    finally:
        sleeper.cancel()
        stopper.cancel()


def _backoff_s(attempt: int) -> float:
    return min(2.0 ** (attempt - 1), _MAX_DELAY_S)


def _retry_delay_s(response: httpx.Response, attempt: int) -> float:
    retry_after = _parse_retry_after(response.headers.get("retry-after"))
    return min(_backoff_s(attempt) if retry_after is None else retry_after, _MAX_DELAY_S)


def _parse_retry_after(value: str | None) -> float | None:
    """Seconds to wait from a Retry-After header (delay in seconds or an HTTP date); None if absent or invalid."""
    if value is None:
        return None
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        seconds = (when - datetime.now(timezone.utc)).total_seconds()
    return max(0.0, seconds) if math.isfinite(seconds) else None


def _check_server(server: str) -> str:
    """``server`` without trailing slashes; ValueError unless httpx can send requests to it."""
    base = server.rstrip("/")
    if "?" in base or "#" in base:
        raise ValueError(f"the server must be an http:// or https:// base URL, got {server!r}")
    try:
        url = httpx.URL(base)
    except httpx.InvalidURL as exc:  # e.g. control characters or an invalid international host name
        raise ValueError(f"invalid server URL {server!r}: {exc}") from None
    if url.scheme not in ("http", "https") or not url.host or url.query or url.fragment:
        raise ValueError(f"the server must be an http:// or https:// base URL, got {server!r}")
    return base


def _header_safe(value: str) -> bool:
    """Printable ASCII without leading or trailing spaces: what an HTTP header value carries unchanged."""
    return bool(value) and all(" " <= char <= "~" for char in value) and value[0] != " " and value[-1] != " "


def _check_unique(items: Sequence[BatchItem]) -> None:
    """BatchFileError for IDs that occur twice (their requests would race for one record)."""
    duplicates = [custom_id for custom_id, count in Counter(item.custom_id for item in items).items() if count > 1]
    if duplicates:
        shown = duplicates[:_MAX_LISTED_DUPLICATES]
        more = f" and {len(duplicates) - len(shown)} more" if len(duplicates) > len(shown) else ""
        raise BatchFileError(f"duplicate IDs: {', '.join(map(repr, shown))}{more}")


def _ssl_verify(verify: str | bool | ssl.SSLContext) -> ssl.SSLContext | bool:
    """httpx 0.28 deprecates verify=<path>: build the context here (a directory is a c_rehash'ed CA folder)."""
    if isinstance(verify, (str, os.PathLike)):
        path = os.fspath(verify)
        if os.path.isdir(path):
            return ssl.create_default_context(capath=path)
        return ssl.create_default_context(cafile=path)
    return verify


def _parse_line(text: str, line_no: int) -> tuple[str | None, BatchItem | str]:
    """(the ID if well-formed, the item or why the line is invalid)."""
    try:
        obj = json.loads(text, parse_constant=_reject_constant, parse_float=_finite_float)
    except RecursionError:
        return None, "invalid JSON: nested too deeply"
    except ValueError as exc:
        return None, f"invalid JSON: {exc}"
    if not isinstance(obj, dict):
        return None, f"expected a JSON object, got {_json_type(obj)}"
    custom_id, problem = _line_id(obj)
    if custom_id is None:
        return None, problem

    url = None
    if obj.keys() & {"body", "method", "url"}:  # OpenAI Batch format
        unexpected = sorted(obj.keys() - _BATCH_FORMAT_KEYS)
        if unexpected:
            keys = ", ".join(map(repr, unexpected))
            return custom_id, f'unexpected top-level key(s) {keys}: request parameters go in "body"'
        method = obj.get("method", "POST")
        if not isinstance(method, str) or method.upper() != "POST":
            return custom_id, f"unsupported method {method!r} (only POST)"
        url = obj.get("url")
        if url is not None and url not in (_CHAT_URL, _COMPLETIONS_URL):
            return custom_id, f"unsupported url {url!r} (use {_CHAT_URL} or {_COMPLETIONS_URL})"
        body = obj.get("body")
        if not isinstance(body, dict):
            return custom_id, f'"body" must be a JSON object, got {_json_type(body)}'
    else:
        body = obj
    body = {key: value for key, value in body.items() if key not in _ID_KEYS}

    has_messages, has_prompt = "messages" in body, "prompt" in body
    if has_messages and has_prompt:
        return custom_id, 'give either "messages" (chat) or "prompt" (completions), not both'
    if not has_messages and not has_prompt:
        return custom_id, 'missing "messages" (chat) or "prompt" (completions)'
    needed = _CHAT_URL if has_messages else _COMPLETIONS_URL
    if url is not None and url != needed:
        return custom_id, f'{url} needs "{"messages" if url == _CHAT_URL else "prompt"}"'
    if has_messages and not isinstance(body["messages"], list):
        return custom_id, f'"messages" must be an array, got {_json_type(body["messages"])}'
    if has_prompt and not isinstance(body["prompt"], (str, list)):
        return custom_id, f'"prompt" must be a string or an array, got {_json_type(body["prompt"])}'
    if body.get("stream"):
        return custom_id, '"stream": true is not supported in batch mode'
    return custom_id, BatchItem(custom_id=custom_id, url=needed, body=body, line_no=line_no)


def _line_id(obj: dict) -> tuple[str | None, str]:
    """(custom_id, "") or (None, the problem)."""
    values = [obj[key] for key in _ID_KEYS if key in obj]
    if not values:
        return None, 'missing "custom_id" (or "id")'
    if len(values) == 2 and values[0] != values[1]:
        return None, '"custom_id" and "id" differ'
    custom_id = values[0]
    if not isinstance(custom_id, str):
        return None, f"custom_id must be a string, got {_json_type(custom_id)}"
    if not custom_id:
        return None, "custom_id is empty"
    if len(custom_id) > _MAX_ID_CHARS:
        return None, f"custom_id is longer than {_MAX_ID_CHARS} characters"
    if not all(" " <= char <= "~" for char in custom_id):
        return None, "custom_id must be printable ASCII (no control or non-ASCII characters)"
    if custom_id[0] == " " or custom_id[-1] == " ":
        return None, "custom_id must not start or end with a space"
    return custom_id, ""


def _reject_constant(name: str) -> float:
    raise ValueError(f"{name} is not valid JSON")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"number {text} is out of range")
    return value


def _json_type(value: object) -> str:
    return _JSON_TYPES.get(type(value), type(value).__name__)


def _json_body(response: httpx.Response) -> object:
    try:
        return response.json()
    except (ValueError, RecursionError):
        return None


def _finite(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _exception_text(exc: BaseException | None) -> str:
    if exc is None:
        return "unknown error"
    text = str(exc).strip()
    name = type(exc).__name__
    return f"{name}: {text.splitlines()[0]}" if text else name


def _one_line(text: str, limit: int = _MAX_PROGRESS_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _say(progress: Callable[[str], None], text: str) -> None:
    """Hand ``text`` to ``progress``; output trouble never stops the batch (the results are saved already)."""
    try:
        progress(text)
    except UnicodeEncodeError:  # e.g. stdout redirected with an ASCII or legacy code-page encoding
        with contextlib.suppress(OSError, UnicodeError):
            progress(text.encode("ascii", "backslashreplace").decode("ascii"))
    except OSError:  # e.g. stdout piped into a pager that quit
        pass


def _main(argv: list[str] | None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:  # argparse: 2 for bad arguments, 0 for --help
        return exc.code if isinstance(exc.code, int) else (_EXIT_OK if exc.code is None else _EXIT_USAGE)

    try:
        target = config.client_target(args.server, args.ca_cert)
    except config.ConfigError as exc:
        return _usage_error(str(exc))
    server = _normalize_server(target.url)
    if server is None:
        return _usage_error(f"--server must be an http:// or https:// URL, got {target.url!r}")
    api_key = target.api_key
    if api_key is not None and not _header_safe(api_key):  # never echo the key
        return _usage_error("the API key (EDGESERVE_API_KEY) must be printable ASCII (no line breaks, tabs or "
                            "non-ASCII characters)")
    verify: str | bool = True
    if target.ca_cert is not None:
        ca_cert = str(target.ca_cert)
        try:
            _ssl_verify(ca_cert)
        except (OSError, ssl.SSLError) as exc:
            return _usage_error(f"cannot use CA certificate {ca_cert}: {exc}")
        verify = ca_cert
    input_path: Path = args.input.expanduser()
    try:
        items, invalid = parse_batch_file(input_path)
    except BatchFileError as exc:
        return _usage_error(str(exc))
    for entry in invalid:
        label = f" ({entry['custom_id']})" if entry["custom_id"] else ""
        print(f"{input_path}:{entry['line_no']}{label}: invalid line: {entry['message']}", file=sys.stderr)
    if not items and not invalid:
        return _usage_error(f"{input_path} contains no requests")
    output_dir: Path = args.output_dir.expanduser()
    try:
        out = OutputDir(output_dir)
    except OSError as exc:
        return _usage_error(f"cannot create output folder {output_dir}: {exc}")

    progress = functools.partial(print, flush=True)
    try:
        run = asyncio.run(_run_interruptible(items, out, server, api_key, verify, args.concurrency, progress, invalid))
    except KeyboardInterrupt:
        run = None
    except BatchServerError as exc:
        print(f"edgeserve batch: {exc}", file=sys.stderr)
        return _EXIT_USAGE
    except OSError as exc:
        print(f"edgeserve batch: {exc}", file=sys.stderr)
        return _EXIT_FAILED
    finally:
        with contextlib.suppress(OSError):
            out.close()
    if run is None:
        print("edgeserve batch: interrupted; run the same command again to resume", file=sys.stderr)
        return _EXIT_INTERRUPTED

    completed = out.completed_ids()
    unfiled = sum(1 for entry in invalid if not entry.get("custom_id"))  # nothing to file them under
    total = len(items) + len(invalid) - unfiled
    # an invalid line whose ID completed earlier keeps that result (it is never overwritten)
    failed = (sum(1 for entry in invalid if entry.get("custom_id") and entry["custom_id"] not in completed)
              + sum(item.custom_id not in completed for item in items))
    note = f"; {unfiled} invalid line{'s' if unfiled != 1 else ''} without an ID (listed above)" if unfiled else ""
    _say(progress, f"{total - failed}/{total} completed, {failed} failed{note}; results in {out.root}")
    return _EXIT_OK if failed == 0 and not invalid else _EXIT_FAILED  # any invalid line needs a look


async def _run_interruptible(
    items: list[BatchItem],
    out: OutputDir,
    server: str,
    api_key: str | None,
    verify: str | bool,
    concurrency: int | None,
    progress: Callable[[str], None],
    invalid: list[dict],
) -> dict | None:
    """run_batch with Ctrl-C turned into a cancellation it handles gracefully; None when interrupted."""
    loop = asyncio.get_running_loop()
    coro = run_batch(
        items, out, server, api_key=api_key, verify=verify, concurrency=concurrency, progress=progress, invalid=invalid
    )
    task = asyncio.ensure_future(coro)
    try:
        loop.add_signal_handler(signal.SIGINT, task.cancel)
        loop.add_signal_handler(signal.SIGTERM, task.cancel)  # kill, timeout, systemd: the same as Ctrl-C
        installed = True
    except (NotImplementedError, RuntimeError, ValueError):  # no signal support here: KeyboardInterrupt instead
        installed = False
    try:
        return await task
    except asyncio.CancelledError:
        if task.cancelled():
            return None
        raise
    finally:
        if installed:
            loop.remove_signal_handler(signal.SIGINT)
            loop.remove_signal_handler(signal.SIGTERM)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edgeserve batch",
        allow_abbrev=False,
        description="Run a JSONL file of ID'd requests against an edgeserve server and save every result. "
        "Running again with the same output folder skips the requests that already completed.",
        epilog="API key: $EDGESERVE_API_KEY, else (for this machine's own server only) the settings file's.",
    )
    parser.add_argument("input", metavar="INPUT", type=Path, help="JSONL file with one request per line")
    parser.add_argument(
        "--output-dir", metavar="DIR", type=Path, required=True, help="folder for records/, results.jsonl and run.json"
    )
    parser.add_argument(
        "--server",
        metavar="URL",
        default=None,
        help="server base URL, e.g. https://192.168.1.20:8000 (default: this machine's server, from its settings)",
    )
    parser.add_argument(
        "--ca-cert",
        metavar="FILE",
        default=None,
        help="CA certificate to trust for HTTPS (the server's edgeserve-ca.crt; default: this machine's, for its "
        "own server)",
    )
    parser.add_argument(
        "--concurrency",
        metavar="N",
        type=_positive_int,
        default=None,
        help="requests in flight at once (default: the server's max_batch_size + 1)",
    )
    return parser


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}")
    return value


def _normalize_server(url: str) -> str | None:
    url = url.strip().rstrip("/")
    if "?" in url or "#" in url:  # a base URL has neither (an empty query would send every request to /)
        return None
    scheme, sep, rest = url.partition("://")
    url = scheme.lower() + sep + rest  # HTTPS://host is https://host
    try:
        parts = urllib.parse.urlsplit(url)
        _ = parts.port  # raises ValueError for a malformed port
        httpx.URL(url)  # what urlsplit accepts but httpx cannot request (control characters, invalid IDNA hosts)
    except (ValueError, httpx.InvalidURL):
        return None
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.query or parts.fragment:
        return None
    if url.endswith("/v1"):  # an OpenAI-style base URL; the request paths already start with /v1
        url = url[: -len("/v1")]
    return url


def _usage_error(message: str) -> int:
    print(f"edgeserve batch: {message}", file=sys.stderr)
    return _EXIT_USAGE


__all__ = ["BatchFileError", "BatchItem", "BatchServerError", "main", "parse_batch_file", "run_batch"]
