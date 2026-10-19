"""WorkerClient against tests/fake_worker.py and small scripted workers in a fake native build (no GPU needed).

Covers spawning (OOM-victim score, own session), the protocol round trips, and the failure paths: load errors
and timeouts, a cancelled start, crashes mid-generation and mid-count, events written just before an exit,
oversized and malformed lines, values the native JSON parser rejects, fatal errors and stubborn workers.
"""

from __future__ import annotations

import asyncio
import gc
import os
import sys
import textwrap
from pathlib import Path

import pytest

from edgeserve import worker as worker_module
from edgeserve.engine import NativeBuild
from edgeserve.worker import WorkerClient, WorkerError

CHAT = {"messages": [{"role": "user", "content": "hello there"}]}
SAMPLING = {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "max_new_tokens": 6}

PRELUDE = '''
import json, os, sys, time

def emit(event):
    sys.stdout.write(json.dumps(event) + "\\n")
    sys.stdout.flush()

def commands():  # ends at "shutdown", like the real worker
    for line in sys.stdin:
        if line.strip():
            cmd = json.loads(line)
            if cmd["op"] == "shutdown":
                return
            yield cmd

def done(cmd):
    t = time.monotonic_ns()
    results = [{"req": r, "finish_reason": "stop", "prompt_tokens": 3, "text": "ok", "token_ids": [5, 2],
                "pieces": ["ok", ""]} for r in range(len(cmd["requests"]))]
    return {"ev": "done", "id": cmd["id"], "t_received_ns": t, "t_start_ns": t, "t_end_ns": t,
            "steps": {"t_ns": [t, t], "gpu_ms": [1.0, 1.0], "free_bytes": [1, 1], "active": [1, 1]},
            "results": results}

READY = {"ev": "ready", "max_batch_size": 4, "cuda_graph": False, "load_ms": 1.0, "warmup_ms": 1.0}
'''


def use_script(native: Path, body: str) -> None:
    """Replace the fake native build's worker with a scripted one."""
    script = native / "scripted_worker.py"
    script.write_text(PRELUDE + textwrap.dedent(body))
    wrapper = native / "edgeserve-worker"
    wrapper.write_text(f'#!/bin/sh\nexec {sys.executable} {script} "$@"\n')
    wrapper.chmod(0o755)


@pytest.fixture
def engine_dir(tmp_path: Path) -> Path:
    path = tmp_path / "engine"
    path.mkdir()
    (path / "llm.engine").write_bytes(b"")
    return path


@pytest.fixture
async def spawn(fake_native: Path, engine_dir: Path):
    """Make WorkerClients for the fake native build; every worker is stopped at the end of the test."""
    clients: list[WorkerClient] = []

    def make(**kwargs) -> WorkerClient:
        client = WorkerClient(NativeBuild.locate(fake_native), engine_dir, **kwargs)
        clients.append(client)
        return client

    yield make
    for client in clients:
        await client.stop(timeout=2)


class _SlowStdout:
    """Hands the worker's events over late, like a busy event loop: its exit is noticed before its last lines."""

    def __init__(self, stream: asyncio.StreamReader, delay: float) -> None:
        self._stream = stream
        self._delay = delay

    def __getattr__(self, name: str):
        return getattr(self._stream, name)

    async def readuntil(self, separator: bytes = b"\n") -> bytes:
        await asyncio.sleep(self._delay)
        return await self._stream.readuntil(separator)

    async def readline(self) -> bytes:
        await asyncio.sleep(self._delay)
        return await self._stream.readline()

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        line = await self.readline()
        if not line:
            raise StopAsyncIteration
        return line


@pytest.fixture
def slow_stdout(monkeypatch):
    spawn_process = asyncio.create_subprocess_exec

    async def spawn_with_slow_stdout(*args, **kwargs):
        proc = await spawn_process(*args, **kwargs)
        proc.stdout = _SlowStdout(proc.stdout, 0.3)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn_with_slow_stdout)


async def collect(generation, timeout: float = 10.0) -> list[dict]:
    async def events() -> list[dict]:
        return [event async for event in generation.events()]

    return await asyncio.wait_for(events(), timeout)


async def exit_code(client: WorkerClient, timeout: float = 5.0) -> int:
    return await asyncio.wait_for(client._proc.wait(), timeout)


async def until(predicate, timeout: float = 5.0) -> None:
    for _ in range(int(timeout / 0.01)):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached in time")


# --------------------------------------------------------------------------- normal operation

async def test_count_generate_stream_and_stop(spawn):
    exits: list[int | None] = []
    client = spawn(on_exit=exits.append)
    ready = await client.start(timeout=20)
    assert ready["max_batch_size"] == 4 and client.ready_info is ready and client.alive
    assert await client.count([CHAT, {"prompt": "a b"}]) == [12, 3]
    assert await client.count([{"messages": [{"role": "user", "content": "grüße 你好 ✓"}]}]) == [13]  # UTF-8 intact

    generation = await client.generate([CHAT, CHAT], SAMPLING, stream=True)
    events = await collect(generation)
    assert events[-1]["ev"] == "done" and [e["ev"] for e in events[:-1]] == ["text"] * 12  # 6 tokens x 2
    assert [len(r["token_ids"]) for r in events[-1]["results"]] == [6, 6]

    await client.stop(timeout=5)
    assert client._proc.returncode == 0 and not client.alive
    with pytest.raises(WorkerError) as excinfo:
        await client.count([CHAT])
    assert excinfo.value.code == "worker_unavailable"
    assert exits == []  # a requested stop is not reported as a crash


async def test_worker_volunteers_for_the_oom_killer_in_a_session_of_its_own(spawn):
    own_score = Path("/proc/self/oom_score_adj").read_text().strip()
    client = spawn()
    await client.start(timeout=20)
    pid = client.pid
    assert Path(f"/proc/{pid}/oom_score_adj").read_text().strip() == "900"
    assert os.getsid(pid) == pid and os.getpgid(pid) == pid  # Ctrl-C to the terminal's process group misses it
    assert Path("/proc/self/oom_score_adj").read_text().strip() == own_score


async def test_cancel_stops_one_sequence_or_the_whole_batch(spawn, monkeypatch):
    monkeypatch.setenv("FAKE_TOKENS", "100")
    monkeypatch.setenv("FAKE_STEP_MS", "10")
    client = spawn()
    await client.start(timeout=20)
    sampling = {**SAMPLING, "max_new_tokens": 100}

    generation = await client.generate([CHAT, CHAT], sampling, stream=True)
    events = generation.events()
    assert (await asyncio.wait_for(events.__anext__(), 5))["ev"] == "text"
    await generation.cancel(1)
    rest = [event async for event in events]
    results = rest[-1]["results"]
    assert results[0]["finish_reason"] == "length" and len(results[0]["token_ids"]) == 100
    assert results[1]["finish_reason"] == "cancelled" and len(results[1]["token_ids"]) < 100

    generation = await client.generate([CHAT, CHAT], sampling, stream=False)
    await asyncio.sleep(0.1)
    await generation.cancel()
    results = (await collect(generation))[-1]["results"]
    assert [r["finish_reason"] for r in results] == ["cancelled", "cancelled"]


# --------------------------------------------------------------------------- start-up failures

async def test_load_failure_is_reported_with_the_workers_message(spawn, monkeypatch):
    monkeypatch.setenv("FAKE_LOAD_FAIL", "1")
    client = spawn()
    with pytest.raises(WorkerError) as excinfo:
        await client.start(timeout=20)
    assert excinfo.value.code == "load_failed" and excinfo.value.fatal
    assert "fake load failure" in str(excinfo.value)
    await exit_code(client)
    assert not client.alive


async def test_load_error_written_just_before_the_exit_is_not_lost(spawn, monkeypatch, slow_stdout):
    monkeypatch.setenv("FAKE_LOAD_FAIL", "1")
    client = spawn()
    with pytest.raises(WorkerError) as excinfo:
        await client.start(timeout=20)
    assert excinfo.value.code == "load_failed" and "fake load failure" in str(excinfo.value)


async def test_load_timeout_stops_the_worker_and_leaves_no_unretrieved_errors(fake_native, engine_dir, caplog):
    use_script(fake_native, '''
        for cmd in commands():  # "loading" forever, but leaves when asked to
            pass
    ''')
    client = WorkerClient(NativeBuild.locate(fake_native), engine_dir)
    with pytest.raises(WorkerError) as excinfo:
        await client.start(timeout=0.3)
    assert excinfo.value.code == "load_timeout" and excinfo.value.fatal
    assert client._proc.returncode == 0
    await asyncio.wait(client._tasks, timeout=5)
    del excinfo, client
    gc.collect()
    assert "exception was never retrieved" not in caplog.text


async def test_cancelled_start_kills_the_loading_worker(spawn, fake_native):
    use_script(fake_native, '''
        time.sleep(60)  # loading a large engine
    ''')
    exits: list[int | None] = []
    client = spawn(on_exit=exits.append)
    start = asyncio.create_task(client.start(timeout=60))
    await until(lambda: client.pid is not None)
    await asyncio.sleep(0.2)
    start.cancel()  # e.g. the server is shut down during loading
    with pytest.raises(asyncio.CancelledError):
        await start
    assert await exit_code(client) == -9
    await asyncio.sleep(0.05)
    assert exits == []


# --------------------------------------------------------------------------- the worker dies

async def test_crash_mid_generation_fails_it_and_everything_queued_behind_it(spawn, monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_CRASH_FILE", str(tmp_path / "crashed-once"))
    exits: list[int | None] = []
    client = spawn(on_exit=exits.append)
    await client.start(timeout=20)
    first = await client.generate([CHAT], SAMPLING, stream=True)
    second = await client.generate([CHAT], SAMPLING, stream=False)
    for generation in (first, second):
        event = (await collect(generation))[-1]
        assert event["ev"] == "error" and event["code"] == "worker_exited" and event["fatal"]
        assert "code 3" in event["message"]
    await until(lambda: exits == [3])
    assert not client.alive
    with pytest.raises(WorkerError) as excinfo:
        await client.generate([CHAT], SAMPLING, stream=False)
    assert excinfo.value.code == "worker_unavailable" and client._generations == {}


async def test_pending_count_fails_when_the_worker_dies(spawn, fake_native):
    use_script(fake_native, '''
        emit(READY)
        for cmd in commands():
            if cmd["op"] == "count":
                os._exit(3)
    ''')
    client = spawn()
    await client.start(timeout=20)
    with pytest.raises(WorkerError) as excinfo:
        await asyncio.wait_for(client.count([CHAT], timeout=20), 10)
    assert excinfo.value.code == "worker_exited" and client._pending_counts == {}


async def test_done_written_just_before_a_crash_is_delivered(spawn, fake_native, slow_stdout):
    use_script(fake_native, '''
        emit(READY)
        for cmd in commands():
            if cmd["op"] == "generate":
                emit(done(cmd))
                os._exit(3)
    ''')
    exits: list[int | None] = []
    client = spawn(on_exit=exits.append)
    await client.start(timeout=20)
    generation = await client.generate([CHAT], SAMPLING, stream=False)
    assert [event["ev"] for event in await collect(generation)] == ["done"]
    await until(lambda: exits == [3])


async def test_fatal_error_stops_new_work_before_the_worker_has_exited(spawn, fake_native):
    use_script(fake_native, '''
        emit(READY)
        for cmd in commands():
            if cmd["op"] == "generate":
                emit({"ev": "error", "id": cmd["id"], "code": "cuda_error", "message": "sticky", "fatal": True})
                time.sleep(5)  # cleaning up before the exit
                os._exit(3)
    ''')
    client = spawn()
    await client.start(timeout=20)
    event = (await collect(await client.generate([CHAT], SAMPLING, stream=False)))[-1]
    assert event["ev"] == "error" and event["code"] == "cuda_error" and event["fatal"]
    assert client._proc.returncode is None  # still on its way out...
    assert not client.alive  # ...but nothing more is sent to it
    with pytest.raises(WorkerError) as excinfo:
        await client.generate([CHAT], SAMPLING, stream=False)
    assert excinfo.value.code == "worker_unavailable"
    client._proc.kill()
    await exit_code(client)


# --------------------------------------------------------------------------- what goes over the pipes

async def test_count_that_is_never_answered_times_out_as_a_worker_error(spawn, fake_native):
    use_script(fake_native, '''
        emit(READY)
        for cmd in commands():
            pass
    ''')
    client = spawn()
    await client.start(timeout=20)
    with pytest.raises(WorkerError) as excinfo:
        await client.count([CHAT], timeout=0.2)
    assert excinfo.value.code == "worker_unavailable" and client._pending_counts == {}


async def test_values_the_worker_cannot_parse_are_refused_before_sending(spawn, fake_native):
    use_script(fake_native, '''
        emit(READY)
        for line in sys.stdin:
            try:  # as strict as the native worker's JSON parser: lone surrogates are invalid
                cmd = json.loads(line)
                json.dumps(cmd, ensure_ascii=False).encode()
            except ValueError:
                emit({"ev": "error", "id": None, "code": "bad_request", "message": "invalid JSON", "fatal": False})
                continue
            if cmd["op"] == "shutdown":
                break
            if cmd["op"] == "count":
                emit({"ev": "count", "id": cmd["id"], "prompt_tokens": [len(cmd["requests"])]})
    ''')
    client = spawn()
    await client.start(timeout=20)
    lone_surrogate = {"messages": [{"role": "user", "content": "hi"}], "stop": ["\ud800"]}
    with pytest.raises(WorkerError) as excinfo:
        await client.count([lone_surrogate], timeout=2)
    assert excinfo.value.code == "bad_request"
    with pytest.raises(WorkerError) as excinfo:
        await client.generate([CHAT], {**SAMPLING, "temperature": float("nan")}, stream=False)
    assert excinfo.value.code == "bad_request" and client._generations == {}
    assert await client.count([CHAT, CHAT], timeout=5) == [2]  # the worker never saw them and still answers


async def test_events_longer_than_the_stream_buffer_are_read_whole(spawn, monkeypatch):
    monkeypatch.setattr(worker_module, "_LINE_LIMIT", 256)  # the ready and done events are both longer
    client = spawn()
    await client.start(timeout=10)
    generation = await client.generate([CHAT, CHAT, CHAT], SAMPLING, stream=False)
    event = (await collect(generation))[-1]
    assert event["ev"] == "done" and [len(r["token_ids"]) for r in event["results"]] == [6, 6, 6]
    assert await client.count([CHAT], timeout=5) == [12]  # still in step afterwards


async def test_long_log_lines_do_not_block_the_worker(spawn, fake_native, monkeypatch):
    monkeypatch.setattr(worker_module, "_LINE_LIMIT", 1024)
    use_script(fake_native, '''
        sys.stderr.write("[edgeserve-worker] warning: " + "x" * 200000 + "\\n")
        sys.stderr.flush()
        emit(READY)
        for cmd in commands():
            if cmd["op"] == "count":
                sys.stderr.write("y" * 300000 + "\\n")
                sys.stderr.flush()
                emit({"ev": "count", "id": cmd["id"], "prompt_tokens": [7]})
    ''')
    client = spawn()
    await client.start(timeout=10)
    for _ in range(3):
        assert await client.count([CHAT], timeout=5) == [7]


async def test_malformed_lines_are_skipped(spawn, fake_native):
    use_script(fake_native, '''
        for line in ("not json", "[1, 2, 3]", '"text"', "42", ""):
            print(line, flush=True)
        emit(READY)
        for cmd in commands():
            if cmd["op"] == "count":
                emit({"ev": "mystery", "id": cmd["id"]})
                emit({"ev": "count", "id": cmd["id"] + 1000, "prompt_tokens": [1]})
                emit({"ev": "count", "id": cmd["id"], "prompt_tokens": [7]})
    ''')
    client = spawn()
    await client.start(timeout=10)
    assert await client.count([CHAT], timeout=5) == [7]


async def test_bad_event_lines_are_logged_without_their_text(spawn, fake_native, caplog):
    use_script(fake_native, '''
        emit(READY)
        for cmd in commands():
            if cmd["op"] == "generate":
                print('["the secret reply"]', flush=True)
                # sorted keys, like the native worker: the results (with the output text) come first
                line = json.dumps(done(cmd), sort_keys=True).replace('"ok"', '"the secret reply"')
                sys.stdout.write(line[:line.index("secret") + 20])  # killed while writing it (no newline)
                sys.stdout.flush()
                os._exit(9)
    ''')
    client = spawn()
    await client.start(timeout=10)
    event = (await collect(await client.generate([CHAT], SAMPLING, stream=False)))[-1]
    assert event["ev"] == "error" and event["code"] == "worker_exited"
    assert "unexpected worker event" in caplog.text and "unparseable worker event" in caplog.text
    assert "secret" not in caplog.text  # logs never contain prompts or outputs


async def test_an_event_that_breaks_the_protocol_replaces_the_worker(spawn, fake_native):
    use_script(fake_native, '''
        emit(READY)
        for cmd in commands():
            if cmd["op"] == "generate":
                emit({"ev": "done", "id": [cmd["id"]]})  # not an id this client can look up
                time.sleep(60)
    ''')
    exits: list[int | None] = []
    client = spawn(on_exit=exits.append)
    await client.start(timeout=10)
    event = (await collect(await client.generate([CHAT], SAMPLING, stream=False)))[-1]
    assert event["ev"] == "error" and event["code"] == "worker_exited"
    await until(lambda: exits == [-9])


async def test_stop_kills_a_worker_that_ignores_shutdown(spawn, fake_native):
    use_script(fake_native, '''
        emit(READY)
        for line in sys.stdin:  # not even "shutdown"
            pass
    ''')
    client = spawn()
    await client.start(timeout=10)
    await client.stop(timeout=0.5)
    assert client._proc.returncode == -9
