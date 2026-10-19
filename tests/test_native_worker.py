"""edgeserve/native/worker.cpp itself, without a GPU: it is compiled against stand-in Edge-LLM, CUDA and TensorRT
headers (tests/worker_mock/include) and driven over its JSON-lines protocol, so the reader, the job
registry, cancellation, the streaming pump and the terminal events are the real code. Words such as
MOCK:tokens=N in a prompt steer the stand-in runtime (worker_mock/include/runtime/llmInferenceRuntime.h).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

from edgeserve.nativebuild import sources_dir
from edgeserve.settings import MAX_BODY_BYTES, Settings

SOURCE = sources_dir() / "worker.cpp"
MOCK_INCLUDE = Path(__file__).resolve().parent / "worker_mock" / "include"
FLAGS = ["-std=gnu++17", "-O0", "-pthread", "-Wall", "-Werror", "-Wno-deprecated-declarations",
         "-Wno-error=unused-parameter"]  # the warning flags of the real build


def _json_include() -> list[str] | None:
    """nlohmann/json: from the Edge-LLM checkout edgeserve setup made, else the system's (nlohmann-json3-dev)."""
    bundled = Path.home() / ".cache" / "edgeserve" / "native-build" / "edgellm-src" / "3rdParty" / "nlohmannJson"
    if (bundled / "include" / "nlohmann" / "json.hpp").is_file():
        return ["-I", str(bundled / "include")]
    if Path("/usr/include/nlohmann/json.hpp").is_file():
        return []
    return None


JSON_INCLUDE = _json_include()  # resolved at import, before the tests' isolated HOME is in place


@pytest.fixture(scope="session")
def worker_binary(request, tmp_path_factory) -> Path:
    compiler = shutil.which("g++")
    json_include = JSON_INCLUDE
    if compiler is None or json_include is None:
        pytest.skip("needs g++ and the nlohmann/json headers (edgeserve setup fetches them; or install the "
                    "nlohmann-json3-dev package)")
    digest = hashlib.sha256(subprocess.run([compiler, "--version"], capture_output=True).stdout)
    digest.update(" ".join(FLAGS + json_include).encode())
    for path in [SOURCE, *sorted(MOCK_INCLUDE.rglob("*.h"))]:
        digest.update(path.read_bytes())
    try:
        cache = Path(request.config.cache.mkdir("edgeserve-worker-mock"))
    except AttributeError:  # pytest -p no:cacheprovider
        cache = tmp_path_factory.mktemp("edgeserve-worker-mock")
    binary = cache / f"edgeserve-worker-{digest.hexdigest()[:16]}"
    if not binary.exists():
        partial = binary.with_name(f"{binary.name}.{os.getpid()}.tmp")
        build = subprocess.run([compiler, *FLAGS, "-I", str(MOCK_INCLUDE), *json_include, str(SOURCE), "-o",
                                str(partial)], capture_output=True, text=True)
        if build.returncode != 0:
            pytest.fail(f"edgeserve/native/worker.cpp does not build against tests/worker_mock:\n"
                        f"{build.stderr[-4000:]}")
        os.replace(partial, binary)
        for stale in cache.glob("edgeserve-worker-*"):
            if stale != binary and not stale.name.endswith(".tmp"):
                with contextlib.suppress(OSError):
                    stale.unlink()
    return binary


class WorkerProcess:
    """The worker on pipes, its events collected by a reader thread."""

    def __init__(self, binary: Path, args: list[str], env: dict[str, str]) -> None:
        self.proc = subprocess.Popen([str(binary), *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, env={**os.environ, **env})
        self.stderr: list[bytes] = []
        self._events: queue.Queue = queue.Queue()
        self._threads = [threading.Thread(target=self._read_events, daemon=True),
                         threading.Thread(target=self._read_stderr, daemon=True)]
        for thread in self._threads:
            thread.start()

    def _read_events(self) -> None:
        for line in self.proc.stdout:
            self._events.put(json.loads(line))
        self._events.put(None)

    def _read_stderr(self) -> None:
        for line in self.proc.stderr:
            self.stderr.append(line)

    def send(self, *commands: dict | bytes) -> None:
        """All commands in one write, so the worker usually reads them together."""
        self.proc.stdin.write(b"".join((c if isinstance(c, bytes) else json.dumps(c).encode()) + b"\n"
                                       for c in commands))
        self.proc.stdin.flush()

    def until(self, predicate, timeout: float = 20.0) -> list[dict]:
        """Events up to and including the first one that matches."""
        events: list[dict] = []
        deadline = time.monotonic() + timeout
        while True:
            try:
                event = self._events.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                pytest.fail(f"no matching event within {timeout:.0f} s after {events}")
            if event is None:
                pytest.fail(f"the worker exited with {self.proc.wait()} after {events}; stderr: "
                            f"{b''.join(self.stderr)[-2000:].decode(errors='replace')}")
            events.append(event)
            if predicate(event):
                return events

    def until_all(self, *predicates, timeout: float = 20.0) -> list[dict]:
        """Events until each predicate has matched one of them."""
        events: list[dict] = []
        pending = list(predicates)
        while pending:
            events += self.until(lambda e: any(p(e) for p in pending), timeout)
            pending = [p for p in pending if not any(p(e) for e in events)]
        return events

    def remaining(self, timeout: float = 20.0) -> list[dict]:
        """Every event until the worker closes its output."""
        events: list[dict] = []
        deadline = time.monotonic() + timeout
        while (event := self._events.get(timeout=max(0.0, deadline - time.monotonic()))) is not None:
            events.append(event)
        return events

    def close_input(self) -> None:
        with contextlib.suppress(BrokenPipeError):
            self.proc.stdin.close()

    def exit_code(self, timeout: float = 10.0) -> int:
        self.close_input()
        return self.proc.wait(timeout)

    def cleanup(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        for thread in self._threads:
            thread.join(5)
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            with contextlib.suppress(OSError):
                stream.close()


@pytest.fixture
def engine_dir(tmp_path: Path) -> Path:
    (tmp_path / "llm.engine").write_bytes(b"")
    (tmp_path / "tokenizer.json").write_text("{}")
    return tmp_path


@pytest.fixture
def start_worker(worker_binary, engine_dir):
    started: list[WorkerProcess] = []

    def start(**env: str) -> WorkerProcess:
        worker = WorkerProcess(worker_binary, ["--engine-dir", str(engine_dir)], env)
        started.append(worker)
        ready = worker.until(lambda e: True)[0]
        assert ready["ev"] == "ready" and ready["max_batch_size"] == 4, ready
        return worker

    yield start
    for worker in started:
        worker.cleanup()


def chat(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


def generate(job_id, text: str = "hello", *, max_new=512, stream=False, items=None, **sampling) -> dict:
    return {"op": "generate", "id": job_id, "stream": stream, "requests": items or [chat(text)],
            "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "max_new_tokens": max_new, **sampling}}


def terminal(job_id):
    return lambda e: e.get("id") == job_id and e["ev"] in ("done", "error")


def only_result(event: dict) -> dict:
    assert event["ev"] == "done", event
    (result,) = event["results"]
    return result


# ------------------------------------------------------------------ protocol

def test_done_event_layout(start_worker):
    worker = start_worker()
    worker.send(generate(1, "MOCK:tokens=4"))
    done = worker.until(terminal(1))[-1]
    result = only_result(done)
    assert result["finish_reason"] == "stop" and len(result["token_ids"]) == 4 and result["text"] == "w0 w1 w2 "
    assert result["pieces"][-1] == "<|eot_id|>" and result["prompt_tokens"] == 5
    steps = done["steps"]
    assert len(steps["t_ns"]) == len(steps["gpu_ms"]) == len(steps["free_bytes"]) == 4
    assert steps["active"] == [1, 1, 1, 1] and None not in steps["gpu_ms"]
    assert done["t_received_ns"] <= done["t_start_ns"] <= steps["t_ns"][0] <= steps["t_ns"][-1] <= done["t_end_ns"]
    assert worker.exit_code() == 0


def test_batch_steps_follow_the_sequences_still_running(start_worker):
    worker = start_worker()
    worker.send(generate(1, items=[chat("MOCK:tokens=2"), chat("MOCK:tokens=5"), chat("MOCK:tokens=3")]))
    done = worker.until(terminal(1))[-1]
    assert [len(r["token_ids"]) for r in done["results"]] == [2, 5, 3]
    assert done["steps"]["active"] == [3, 3, 2, 1, 1]
    assert len(done["steps"]["gpu_ms"]) == 5 and None not in done["steps"]["gpu_ms"]


def test_count(start_worker):
    worker = start_worker()
    worker.send({"op": "count", "id": 1, "requests": [{"prompt": "a b c"}, {"prompt": "<|begin_of_text|> a"}]},
                {"op": "count", "id": 2, "requests": [chat("a b")]},
                {"op": "count", "id": 3, "requests": [{"prompt": "a"}, chat("b")]})
    events = worker.until(lambda e: e.get("id") == 3)
    by_id = {e["id"]: e for e in events}
    assert by_id[1]["prompt_tokens"] == [4, 2]  # BOS added only where missing
    assert by_id[2]["prompt_tokens"] == [6]
    assert by_id[3]["ev"] == "error" and by_id[3]["code"] == "bad_request"  # prompts and messages mixed


def test_streamed_text_matches_the_result(start_worker):
    worker = start_worker()
    worker.send(generate(1, "MOCK:tokens=5", stream=True))
    events = worker.until(terminal(1))
    assert "".join(e["text"] for e in events if e["ev"] == "text") == only_result(events[-1])["text"] == "w0 w1 w2 w3 "


def test_a_character_cut_between_stream_chunks_arrives_whole(start_worker):
    worker = start_worker()
    worker.send(generate(1, "MOCK:utf8", max_new=5, stream=True))
    events = worker.until(terminal(1))
    texts = [e["text"] for e in events if e["ev"] == "text"]
    assert texts and all("\ufffd" not in text for text in texts)  # whole characters only, never U+FFFD
    # The runtime never flushed its held-back byte; the front-end sends the rest from the final text.
    assert "".join(texts) == "\u00e9" * 4 and only_result(events[-1])["text"] == "\u00e9" * 5


def test_a_character_cut_by_max_tokens_is_never_streamed_in_pieces(start_worker):
    worker = start_worker()
    worker.send(generate(1, "MOCK:tokens=10 MOCK:utf8_cut", max_new=3, stream=True))
    events = worker.until(terminal(1))
    result = only_result(events[-1])
    assert result["finish_reason"] == "length" and len(result["token_ids"]) == 3
    # The text shows the incomplete character as U+FFFD (as Edge-LLM's and Hugging Face's decoding do); the stream
    # holds its bytes back, and the front-end sends the rest of the final text when the request ends.
    assert result["text"].startswith("w0 w1 ") and set(result["text"][6:]) == {"\ufffd"}
    assert "".join(e["text"] for e in events if e["ev"] == "text") == "w0 w1 "


@pytest.mark.parametrize(("sampling", "received"), [
    ({"temperature": 0.8, "top_k": 0, "top_p": 1.0}, "temperature=0.800 top_k=0 top_p=0.999998927"),
    # at temperature 1 a top_p of 1 - 1e-6 or more would make Edge-LLM decode greedily
    ({"temperature": 1.0, "top_k": 0, "top_p": 1.0}, "temperature=1.000 top_k=0 top_p=0.999998927"),
    ({"temperature": 1.0, "top_k": 0, "top_p": 0.9999999}, "temperature=1.000 top_k=0 top_p=0.999998927"),
    ({"temperature": 0.0, "top_k": 0, "top_p": 1.0}, "temperature=0.000 top_k=1 top_p=1.000000000"),
    ({"temperature": 0.8, "top_k": 40, "top_p": 1.0}, "temperature=0.800 top_k=40 top_p=1.000000000"),
    ({"temperature": 0.8, "top_k": 0, "top_p": 0.9}, "temperature=0.800 top_k=0 top_p=0.899999976"),
])
def test_unlimited_sampling_reaches_the_runtime_in_a_form_it_accepts(start_worker, sampling, received):
    # Edge-LLM refuses top_k 0 with top_p 1 ("Either topK or topP must be set").
    worker = start_worker()
    worker.send(generate(1, "MOCK:echo_sampling MOCK:tokens=2", **sampling))
    assert only_result(worker.until(terminal(1))[-1])["text"] == received


def test_failures_are_reported_and_the_worker_keeps_serving(start_worker):
    worker = start_worker()
    worker.send(generate(1, "MOCK:too_long"), generate(2, "MOCK:fail"), generate(3, "MOCK:throw"),
                generate(4, "MOCK:throw_unknown"), {"op": "frobnicate", "id": 5}, generate(6, "MOCK:tokens=2"))
    events = worker.until(terminal(6))
    by_id = {e["id"]: e for e in events if e["ev"] in ("done", "error")}
    assert by_id[1]["code"] == "input_too_long"
    assert by_id[2]["code"] == by_id[3]["code"] == by_id[4]["code"] == "generation_failed"
    assert "mock runtime failure" in by_id[3]["message"]
    assert by_id[5]["code"] == "bad_request"
    assert by_id[6]["ev"] == "done"
    assert not any(e["fatal"] for e in by_id.values() if e["ev"] == "error")
    assert worker.exit_code() == 0


def test_sticky_cuda_error_ends_the_worker(start_worker):
    worker = start_worker()
    worker.send(generate(1, "MOCK:sticky"))
    error = worker.until(terminal(1))[-1]
    assert error["code"] == "cuda_error" and error["fatal"] is True
    assert worker.proc.wait(10) == 3


def test_cancel_one_sequence_of_a_running_batch(start_worker):
    worker = start_worker()
    worker.send(generate(1, items=[chat("MOCK:tokens=300 MOCK:step_ms=5")] * 2))
    time.sleep(0.2)
    worker.send({"op": "cancel", "id": 1, "req": 1})
    done = worker.until(terminal(1))[-1]
    assert [r["finish_reason"] for r in done["results"]] == ["stop", "cancelled"]


def test_probe_arguments_and_load_failure(worker_binary, engine_dir):
    def run(*args: str, **env: str) -> subprocess.CompletedProcess:
        return subprocess.run([str(worker_binary), *args], capture_output=True, env={**os.environ, **env},
                              timeout=30)

    probe = run("--probe")
    assert probe.returncode == 0 and json.loads(probe.stdout)["ev"] == "probe"
    no_device = run("--probe", MOCK_NO_DEVICE="1")
    assert no_device.returncode == 2 and json.loads(no_device.stdout)["code"] == "no_device"
    assert run("--frobnicate").returncode == 64
    missing = run("--engine-dir", str(engine_dir / "missing"))
    assert missing.returncode == 2 and json.loads(missing.stdout)["code"] == "load_failed"


# ------------------------------------------------------------------ regressions

def test_cancel_sent_with_the_generate_stops_it(start_worker):
    """The scheduler cancels right after dispatching when the client left meanwhile: the cancel then
    reaches the worker before the job runs and must still stop it (it once ran all 512 tokens)."""
    worker = start_worker()
    worker.send(generate(1, "MOCK:tokens=1000 MOCK:step_ms=5"), {"op": "cancel", "id": 1})
    result = only_result(worker.until(terminal(1))[-1])
    assert result["finish_reason"] == "cancelled" and len(result["token_ids"]) == 1  # the prefill always runs


def test_cancel_reaches_a_queued_job(start_worker):
    worker = start_worker()
    worker.send(generate(1, "MOCK:tokens=40 MOCK:step_ms=5"), generate(2, "MOCK:tokens=1000 MOCK:step_ms=5"),
                {"op": "cancel", "id": 2})
    events = worker.until(terminal(2))
    assert only_result(next(e for e in events if terminal(1)(e)))["finish_reason"] == "stop"
    assert only_result(events[-1])["finish_reason"] == "cancelled"


@pytest.mark.parametrize("how", ["shutdown", "end of input"])
def test_shutdown_stops_a_job_that_has_not_started(start_worker, how):
    """A job that already left the queue when the worker is told to stop is cancelled (it used to run to
    the end); one still queued is dropped, and the client fails it when the worker exits."""
    worker = start_worker()
    worker.send(generate(1, "MOCK:tokens=1000 MOCK:step_ms=5"), *([{"op": "shutdown"}] if how == "shutdown" else []))
    if how == "end of input":
        worker.close_input()
    reasons = [only_result(e)["finish_reason"] for e in worker.remaining() if terminal(1)(e)]
    assert reasons in ([], ["cancelled"])
    assert worker.proc.wait(10) == 0


def test_queued_jobs_counts_and_cancels_interleaved(start_worker):
    worker = start_worker()
    commands: list[dict] = []
    for i in range(1, 9):
        commands += [generate(i, f"MOCK:tokens={3 + i} MOCK:step_ms=1", stream=i % 2 == 0),
                     {"op": "count", "id": 100 + i, "requests": [chat("a b c")]}]
    worker.send(*commands, {"op": "cancel", "id": 5}, {"op": "cancel", "id": 6, "req": 0})
    counted = [lambda e, i=i: e["ev"] == "count" and e["id"] == i for i in range(101, 109)]
    events = worker.until_all(terminal(8), *counted, timeout=30)
    finished = [e for e in events if e["ev"] in ("done", "error") and e["id"] < 100]
    assert [e["id"] for e in finished] == list(range(1, 9))
    assert sorted(e["id"] for e in events if e["ev"] == "count") == list(range(101, 109))
    reasons = {e["id"]: only_result(e)["finish_reason"] for e in finished}
    assert reasons == {1: "stop", 2: "stop", 3: "stop", 4: "stop", 5: "cancelled", 6: "cancelled", 7: "stop", 8: "stop"}
    for e in finished:
        if e["id"] % 2 == 0:
            streamed = "".join(t["text"] for t in events if t["ev"] == "text" and t["id"] == e["id"])
            assert streamed == only_result(e)["text"]


def test_last_stream_delta_is_forwarded(start_worker):
    """The final chunk can be published right after the pump found the channel empty and just before
    the generation ends; the pump must still forward it (the API streams only text events)."""
    worker = start_worker()
    worker.send(generate(1, "MOCK:tokens=100 MOCK:pump_race", max_new=3, stream=True))
    events = worker.until(terminal(1))
    result = only_result(events[-1])
    assert result["finish_reason"] == "length"
    assert "".join(e["text"] for e in events if e["ev"] == "text") == result["text"] == "w0 w1 w2 "


def test_malformed_commands_are_answered_without_an_id(start_worker):
    """Malformed lines used to abort the worker (an exception escaping its reader thread). They get an
    error without an id, so a bad cancel never ends the job it names, and serving goes on."""
    worker = start_worker()
    worker.send(generate(1, "MOCK:tokens=200 MOCK:step_ms=5"))
    time.sleep(0.1)  # the old reader crashed on a bad cancel only while a job was running
    malformed = [b"[1]", b"5", b"null", b'"generate"', b'{"op": 5}', b'{"op": "cancel", "id": 1, "req": "a"}',
                 b'{"op": "cancel", "id": 1, "req": 1.5}', b"{not json"]
    ignored = [b'{"op": "cancel", "id": "x"}', b'{"op": "cancel"}']  # valid, they match no job
    worker.send(*malformed, *ignored, {"op": "count", "id": 2, "requests": [chat("a")]})
    events = worker.until_all(terminal(1), lambda e: e["ev"] == "count" and e["id"] == 2)
    errors = [e for e in events if e["ev"] == "error"]
    assert len(errors) == len(malformed) and all(e["id"] is None and e["code"] == "bad_request" for e in errors)
    assert only_result(next(e for e in events if terminal(1)(e)))["finish_reason"] == "stop"
    assert worker.exit_code() == 0


def test_invalid_generate_fields_get_bad_request(start_worker):
    """Out-of-range sampling values used to crash the worker (reserve(SIZE_MAX) for -2) or be narrowed
    to int32 by the runtime (2**32 + 5 became 5); now each job gets a bad_request of its own."""
    worker = start_worker()
    invalid = [generate(1, max_new=-2), generate(2, max_new=0), generate(3, max_new=2**32 + 5),
               generate(4, max_new=2**64 - 1), generate(5, max_new=1e300), generate(6, top_k=-1),
               generate(7, top_k=1025), {**generate(8), "stream": "yes"}, {**generate(9), "sampling": [1]},
               generate(10, items=[{**chat("a"), "seed": 1.5}]), generate(11, items=[chat("a")] * 5),
               {"op": "generate", "id": 12, "requests": []}, {"op": "generate", "id": 13},
               generate(14, items=[{"prompt": "a"}, chat("b")])]
    worker.send(*invalid, generate(100, items=[{**chat("MOCK:tokens=2"), "seed": 2**64 - 1}], top_k=1024))
    events = worker.until(terminal(100))
    errors = {e["id"]: e for e in events if e["ev"] == "error"}
    assert sorted(errors) == list(range(1, 15)) and all(e["code"] == "bad_request" for e in errors.values())
    assert only_result(events[-1])["finish_reason"] == "stop"


def test_non_sticky_cuda_error_does_not_fail_the_next_request(start_worker):
    """A failed request can leave a non-sticky error as the thread's last CUDA error; Edge-LLM's launch
    checks (cudaGetLastError) would report it again in the next request unless the worker clears it."""
    worker = start_worker()
    worker.send(generate(1, "MOCK:fail_nonsticky"))
    assert worker.until(terminal(1))[-1]["code"] == "generation_failed"
    worker.send(generate(2, "MOCK:tokens=2"))
    assert only_result(worker.until(terminal(2))[-1])["finish_reason"] == "stop"


def test_first_step_time_excludes_the_memory_query(start_worker):
    """T_0 is when the first token became available, before the per-step free-memory query."""
    worker = start_worker(MOCK_MEMINFO_MS="300")
    worker.send(generate(1, "MOCK:tokens=2 MOCK:step_ms=5"))
    done = worker.until(terminal(1))[-1]
    assert (done["steps"]["t_ns"][0] - done["t_start_ns"]) / 1e6 < 200


def test_command_limit_holds_the_largest_generate():
    """The worker drops a longer command line with an error that has no id, so a generate over the limit
    would never end and its batch would wait forever: the limit must hold the largest batch the
    front-end can send (its JSON is about as long as the request bodies together)."""
    match = re.search(r"kMaxCommandBytes = size_t\{(\d+)\} << (\d+);", SOURCE.read_text())
    assert match, "kMaxCommandBytes not found in native/worker.cpp"
    limit = int(match[1]) << int(match[2])
    max_batch = 16
    Settings(model=Path("model"), max_batch_size=max_batch).validate()
    with pytest.raises(ValueError):
        Settings(model=Path("model"), max_batch_size=max_batch + 1).validate()
    assert limit >= 1.5 * max_batch * MAX_BODY_BYTES
