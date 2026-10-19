"""Scheduler: admission, batching, dispatch, cancellation, expiry and drain against a stub worker (no subprocess).

The stub hands out real Generation objects; each test plays the worker's events into them. Every submitted job
must end with exactly one terminal event (done / error / cancelled), whatever happens to the worker.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from edgeserve import scheduler as scheduler_module
from edgeserve.scheduler import AdmissionError, Job, Scheduler
from edgeserve.worker import Generation, WorkerError


class StubWorker:
    """Stands in for WorkerClient: records generate commands and the cancels sent through its generations."""

    def __init__(self) -> None:
        self.alive = True
        self.generations: list[Generation] = []
        self.batches: list[list[dict]] = []
        self.samplings: list[dict] = []
        self.streams: list[bool] = []
        self.sent: list[dict] = []
        self.generate_error: WorkerError | None = None
        self.gate: asyncio.Event | None = None  # when given, generate() waits for it (a slow pipe write)

    async def generate(self, items: list[dict], sampling: dict, stream: bool) -> Generation:
        if self.gate is not None:
            await self.gate.wait()
        if self.generate_error is not None:
            raise self.generate_error
        generation = Generation(len(self.generations) + 1, self)  # type: ignore[arg-type]
        self.generations.append(generation)
        self.batches.append(items)
        self.samplings.append(sampling)
        self.streams.append(stream)
        return generation

    async def send(self, command: dict) -> None:
        if not self.alive:
            raise WorkerError("worker_unavailable", "the inference worker is not running")
        self.sent.append(command)


def make_job(custom_id: str, *, temperature: float = 0.0, max_new_tokens: int = 8, prompt: bool = False,
             stream: bool = False) -> Job:
    item = {"prompt": f"<s>{custom_id}"} if prompt else {"messages": [{"role": "user", "content": custom_id}]}
    sampling = {"temperature": temperature, "top_p": 1.0, "top_k": 1, "max_new_tokens": max_new_tokens}
    return Job(custom_id=custom_id, attempt_id=f"att_{custom_id}", endpoint="/v1/chat/completions", item=item,
               sampling=sampling, stream=stream, prompt_tokens=5, t_arrival_ns=time.monotonic_ns(),
               t_arrival_unix_ns=time.time_ns(), request_echo={}, params_echo={})


def done_event(generation: Generation, reqs: list[int], finish: dict[int, str] | None = None) -> dict:
    results = [{"req": r, "finish_reason": (finish or {}).get(r, "stop"), "prompt_tokens": 5, "text": "a b c",
                "token_ids": [7, 8, 9], "pieces": ["a", " b", " c"]} for r in reqs]
    return {"ev": "done", "id": generation.job_id, "t_received_ns": 1, "t_start_ns": 2, "t_end_ns": 3,
            "steps": {"t_ns": [4, 5, 6], "gpu_ms": [1.0, 1.0, 1.0], "free_bytes": [0, 0, 0], "active": [1, 1, 1]},
            "results": results}


def error_event(generation: Generation, code: str = "worker_exited", fatal: bool = True) -> dict:
    return {"ev": "error", "id": generation.job_id, "code": code, "message": f"{code} happened", "fatal": fatal}


async def terminal(job: Job, timeout: float = 2.0) -> tuple:
    """The job's next event that is not streamed text."""
    while True:
        event = await asyncio.wait_for(job.events.get(), timeout)
        if event[0] != "text":
            return event


async def assert_no_second_terminal(job: Job) -> None:
    await asyncio.sleep(0.02)
    leftovers = []
    while not job.events.empty():
        leftovers.append(job.events.get_nowait())
    assert [e for e in leftovers if e[0] != "text"] == []


async def until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.005)


@pytest.fixture
async def make():
    """Start schedulers; their dispatchers are stopped at the end of the test."""
    started: list[Scheduler] = []

    def factory(get_worker, *, max_batch_size: int = 1, queue_size: int = 16, queue_timeout_s: float = 60.0,
                on_change=None) -> Scheduler:
        getter = get_worker if callable(get_worker) else (lambda: get_worker)
        scheduler = Scheduler(getter, max_batch_size=max_batch_size, kv_capacity=128, queue_size=queue_size,
                              queue_timeout_s=queue_timeout_s, on_change=on_change)
        scheduler.start()
        started.append(scheduler)
        return scheduler

    yield factory
    for scheduler in started:
        if scheduler._task is not None:
            scheduler._task.cancel()
            await asyncio.gather(scheduler._task, return_exceptions=True)
        if scheduler._expiry is not None:
            scheduler._expiry.cancel()


# --------------------------------------------------------------------------- happy path and admission

async def test_job_runs_and_receives_text_then_its_result(make):
    worker = StubWorker()
    scheduler = make(worker)
    job = make_job("a", stream=True)
    scheduler.submit(job)
    assert job.state == "queued" and scheduler.is_inflight("a")
    await until(lambda: worker.generations)
    assert worker.batches == [[job.item]] and worker.samplings == [job.sampling] and worker.streams == [True]
    assert job.state == "running" and scheduler.running == 1 and scheduler.waiting == 0
    assert job.t_dispatch_ns is not None and job.batch_size == 1

    generation = worker.generations[0]
    generation._put({"ev": "text", "id": generation.job_id, "req": 0, "t_ns": 4, "text": "a"})
    generation._put(done_event(generation, [0]))
    kind, text = await asyncio.wait_for(job.events.get(), 2)
    assert kind == "text" and text["text"] == "a"
    kind, event, result = await terminal(job)
    assert kind == "done" and result["token_ids"] == [7, 8, 9] and job.state == "finished"
    assert job.batch_contexts == [[5], [6], [7]]
    await until(lambda: scheduler.running == 0)
    assert scheduler.is_inflight("a")  # until the response and record are complete
    scheduler.release(job)
    assert not scheduler.is_inflight("a")
    await assert_no_second_terminal(job)


async def test_admission_refuses_duplicates_a_full_queue_and_shutdown(make):
    scheduler = make(None, queue_size=2)  # no worker: everything stays queued
    first = make_job("a")
    scheduler.submit(first)
    with pytest.raises(AdmissionError) as excinfo:
        scheduler.submit(make_job("a"))
    assert excinfo.value.status == 409 and excinfo.value.code == "duplicate_request_id"
    scheduler.submit(make_job("b"))
    with pytest.raises(AdmissionError) as excinfo:
        scheduler.submit(make_job("c"))
    assert excinfo.value.status == 503 and excinfo.value.code == "server_is_overloaded"
    assert excinfo.value.retry_after >= 1 and not scheduler.is_inflight("c")

    await asyncio.wait_for(scheduler.drain(0.05), 2)
    assert await terminal(first) == ("cancelled", None)
    with pytest.raises(AdmissionError) as excinfo:
        scheduler.check_capacity()
    assert excinfo.value.status == 503 and excinfo.value.code == "server_shutting_down"


# --------------------------------------------------------------------------- batching

async def test_batch_contexts_use_the_admission_count_when_the_worker_has_none(make):
    # The worker reports prompt_tokens -1 when Edge-LLM gives no count; MFU/MBU then use the length counted at
    # admission, never -1 as a context length.
    worker = StubWorker()
    scheduler = make(worker, max_batch_size=2)
    first, second = make_job("a"), make_job("b")
    second.prompt_tokens = 9
    scheduler.submit(first)
    scheduler.submit(second)
    await until(lambda: worker.generations)
    generation = worker.generations[0]
    event = done_event(generation, [0, 1])
    for result in event["results"]:
        result["prompt_tokens"] = -1
    generation._put(event)
    await terminal(first)
    await terminal(second)
    assert first.batch_contexts == second.batch_contexts == [[5, 9], [6, 10], [7, 11]]


async def test_batches_take_the_oldest_job_plus_compatible_later_ones(make):
    worker = StubWorker()
    scheduler = make(worker, max_batch_size=3)
    first = make_job("first")
    scheduler.submit(first)
    await until(lambda: len(worker.generations) == 1)  # runs alone; the rest queue up behind it

    a, b, c = make_job("a"), make_job("b", temperature=0.7), make_job("c", stream=True)
    d, e, f, g = make_job("d", prompt=True), make_job("e"), make_job("f"), make_job("g", max_new_tokens=4)
    for job in (a, b, c, d, e, f, g):
        scheduler.submit(job)

    expected = [[first], [a, c, e], [b], [d], [f], [g]]  # compatible = same kind, sampling and output limit
    for n, batch in enumerate(expected):
        await until(lambda: len(worker.generations) == n + 1)
        assert worker.batches[n] == [job.item for job in batch]
        assert [job.batch_size for job in batch] == [len(batch)] * len(batch)
        assert [job._req_index for job in batch] == list(range(len(batch)))
        assert worker.samplings[n] == batch[0].sampling
        assert worker.streams[n] == any(job.stream for job in batch)
        generation = worker.generations[n]
        if n == 1:  # streamed text goes to the job of its sequence
            generation._put({"ev": "text", "id": generation.job_id, "req": 1, "t_ns": 4, "text": "for c"})
            kind, text = await asyncio.wait_for(c.events.get(), 2)
            assert kind == "text" and text["text"] == "for c" and a.events.empty() and e.events.empty()
        generation._put(done_event(generation, list(range(len(batch)))))
        for job in batch:
            assert (await terminal(job))[0] == "done"
    assert scheduler.waiting == 0


async def test_done_event_with_missing_or_repeated_results_gives_one_terminal_event_each(make):
    worker = StubWorker()
    scheduler = make(worker, max_batch_size=3)
    jobs = [make_job("a"), make_job("b"), make_job("c")]
    for job in jobs:
        scheduler.submit(job)
    await until(lambda: worker.generations)
    generation = worker.generations[0]
    event = done_event(generation, [0, 1, 0, 7])  # c missing, a twice, an index outside the batch
    generation._put(event)
    assert (await terminal(jobs[0]))[0] == "done"
    assert (await terminal(jobs[1]))[0] == "done"
    kind, error = await terminal(jobs[2])
    assert kind == "error" and error.code == "internal_error"
    for job in jobs:
        await assert_no_second_terminal(job)


# --------------------------------------------------------------------------- cancellation

async def test_cancel_of_a_queued_job_drops_it_and_updates_the_queue_gauge(make):
    gauge: list[int] = []
    holder: dict[str, Scheduler] = {}
    scheduler = make(None, on_change=lambda: gauge.append(holder["s"].waiting))
    holder["s"] = scheduler
    a, b = make_job("a"), make_job("b")
    scheduler.submit(a)
    scheduler.submit(b)
    assert gauge[-1] == 2
    scheduler.cancel(a)
    assert await terminal(a) == ("cancelled", None) and a.state == "finished"
    assert scheduler.waiting == 1 and gauge[-1] == 1
    scheduler.cancel(a)  # a second cancel changes nothing
    await assert_no_second_terminal(a)
    assert b.state == "queued"


async def test_cancel_of_a_running_job_stops_only_its_sequence(make):
    worker = StubWorker()
    scheduler = make(worker, max_batch_size=2)
    a, b = make_job("a"), make_job("b")
    scheduler.submit(a)
    scheduler.submit(b)
    await until(lambda: worker.generations)
    generation = worker.generations[0]
    scheduler.cancel(b)
    await until(lambda: worker.sent)
    assert worker.sent == [{"op": "cancel", "id": generation.job_id, "req": 1}]
    scheduler.cancel(b)
    await asyncio.sleep(0.02)
    assert len(worker.sent) == 1
    generation._put(done_event(generation, [0, 1], finish={1: "cancelled"}))
    assert (await terminal(a))[2]["finish_reason"] == "stop"
    assert (await terminal(b))[2]["finish_reason"] == "cancelled"


async def test_cancel_while_dispatching_reaches_the_worker_once_the_generation_exists(make):
    worker = StubWorker()
    worker.gate = asyncio.Event()
    scheduler = make(worker)
    job = make_job("a")
    scheduler.submit(job)
    await until(lambda: job.state == "running")  # generate() has not returned yet
    scheduler.cancel(job)
    await asyncio.sleep(0.02)
    assert worker.sent == []
    worker.gate.set()
    await until(lambda: worker.sent)
    assert worker.sent == [{"op": "cancel", "id": 1, "req": 0}]
    generation = worker.generations[0]
    generation._put(done_event(generation, [0], finish={0: "cancelled"}))
    assert (await terminal(job))[2]["finish_reason"] == "cancelled"


async def test_release_of_an_unfinished_job_cancels_it(make):
    """A handler that goes away without its result (e.g. cancelled when the HTTP server's graceful shutdown runs
    out) must not leave its job holding the queue or the GPU, nor make drain() wait for it."""
    worker = StubWorker()
    scheduler = make(worker)
    running, queued = make_job("running"), make_job("queued")
    scheduler.submit(running)
    await until(lambda: worker.generations)
    scheduler.submit(queued)

    scheduler.release(queued)
    assert await terminal(queued) == ("cancelled", None)
    assert scheduler.waiting == 0 and not scheduler.is_inflight("queued")

    scheduler.release(running)
    assert not scheduler.is_inflight("running")
    await until(lambda: worker.sent)
    assert worker.sent == [{"op": "cancel", "id": 1, "req": 0}]
    generation = worker.generations[0]
    generation._put(done_event(generation, [0], finish={0: "cancelled"}))
    assert (await terminal(running))[0] == "done"
    await asyncio.wait_for(scheduler.drain(5), 1)


# --------------------------------------------------------------------------- worker failures

async def test_generate_failure_fails_every_job_of_the_batch(make):
    worker = StubWorker()
    worker.generate_error = WorkerError("worker_unavailable", "the inference worker is not running")
    scheduler = make(worker, max_batch_size=2)
    jobs = [make_job("a"), make_job("b")]
    for job in jobs:
        scheduler.submit(job)
    for job in jobs:
        kind, error = await terminal(job)
        assert kind == "error" and error.code == "worker_unavailable"
        await assert_no_second_terminal(job)
    await until(lambda: scheduler.running == 0)


async def test_worker_death_mid_batch_fails_the_batch_and_queued_jobs_wait_for_a_new_worker(make):
    worker = StubWorker()
    holder = {"worker": worker}
    scheduler = make(lambda: holder["worker"])
    a = make_job("a")
    scheduler.submit(a)
    await until(lambda: worker.generations)
    b = make_job("b")
    scheduler.submit(b)

    worker.alive = False
    worker.generations[0]._put(error_event(worker.generations[0]))
    kind, error = await terminal(a)
    assert kind == "error" and error.code == "worker_exited" and error.fatal
    await asyncio.sleep(0.05)
    assert b.state == "queued" and scheduler.waiting == 1 and len(worker.generations) == 1

    holder["worker"] = None  # restarting
    scheduler.wake()
    await asyncio.sleep(0.02)
    assert b.state == "queued"
    replacement = StubWorker()
    holder["worker"] = replacement
    scheduler.wake()
    await until(lambda: replacement.generations)
    assert replacement.batches == [[b.item]]
    replacement.generations[0]._put(done_event(replacement.generations[0], [0]))
    assert (await terminal(b))[0] == "done"


async def test_fail_queued_fails_every_waiting_job(make):
    gauge: list[int] = []
    holder: dict[str, Scheduler] = {}
    scheduler = make(None, on_change=lambda: gauge.append(holder["s"].waiting))
    holder["s"] = scheduler
    jobs = [make_job("a"), make_job("b")]
    for job in jobs:
        scheduler.submit(job)
    error = WorkerError("worker_exited", "the worker failed too often", fatal=True)
    scheduler.fail_queued(error)
    for job in jobs:
        assert await terminal(job) == ("error", error)
    assert scheduler.waiting == 0 and gauge[-1] == 0
    await asyncio.wait_for(scheduler.drain(5), 1)


async def test_dispatcher_survives_an_unexpected_error_in_a_batch(make):
    worker = StubWorker()
    scheduler = make(worker)
    a, b = make_job("a"), make_job("b")
    scheduler.submit(a)
    scheduler.submit(b)
    await until(lambda: worker.generations)
    worker.generations[0]._put({"ev": "done", "id": 1, "results": [{"req": "zero"}]})  # malformed
    kind, error = await terminal(a)
    assert kind == "error" and error.code == "internal_error"
    await until(lambda: len(worker.generations) == 2)
    worker.generations[1]._put(done_event(worker.generations[1], [0]))
    assert (await terminal(b))[0] == "done"


# --------------------------------------------------------------------------- queue expiry

async def test_queued_job_expires_while_no_worker_is_available(make):
    gauge: list[int] = []
    holder: dict[str, Scheduler] = {}
    scheduler = make(None, queue_timeout_s=0.1, on_change=lambda: gauge.append(holder["s"].waiting))
    holder["s"] = scheduler
    job = make_job("a")
    scheduler.submit(job)
    kind, error = await terminal(job, timeout=2)
    assert kind == "error" and error.code == "queue_timeout"
    assert scheduler.waiting == 0 and gauge[-1] == 0
    await asyncio.wait_for(scheduler.drain(5), 1)  # nothing left to wait for


async def test_queued_jobs_expire_on_time_while_a_batch_runs(make):
    worker = StubWorker()
    scheduler = make(worker, queue_timeout_s=0.3)
    running = make_job("running")
    scheduler.submit(running)
    await until(lambda: worker.generations)  # the worker never finishes this one
    early = make_job("early")
    scheduler.submit(early)
    await asyncio.sleep(0.2)
    late = make_job("late")
    scheduler.submit(late)

    kind, error = await terminal(early, timeout=2)
    assert kind == "error" and error.code == "queue_timeout"
    assert late.state == "queued"  # its own deadline is later
    kind, error = await terminal(late, timeout=2)
    assert kind == "error" and error.code == "queue_timeout"
    assert running.state == "running" and scheduler.waiting == 0


# --------------------------------------------------------------------------- drain

async def test_drain_waits_for_running_work_and_refuses_new_work(make):
    worker = StubWorker()
    scheduler = make(worker)
    job = make_job("a")
    scheduler.submit(job)
    await until(lambda: worker.generations)
    drain = asyncio.create_task(scheduler.drain(5))
    await asyncio.sleep(0.05)
    assert not drain.done()
    with pytest.raises(AdmissionError) as excinfo:
        scheduler.submit(make_job("late"))
    assert excinfo.value.code == "server_shutting_down"
    worker.generations[0]._put(done_event(worker.generations[0], [0]))
    await asyncio.wait_for(drain, 2)
    assert (await terminal(job))[0] == "done"
    assert worker.sent == []  # it finished in time: nothing was cancelled


async def test_drain_cancels_what_is_left_and_gives_up_on_a_batch_the_worker_does_not_stop(make, monkeypatch):
    monkeypatch.setattr(scheduler_module, "_CANCEL_GRACE_S", 0.1)
    worker = StubWorker()
    scheduler = make(worker)
    running, queued = make_job("running"), make_job("queued")
    scheduler.submit(running)
    await until(lambda: worker.generations)
    scheduler.submit(queued)

    await asyncio.wait_for(scheduler.drain(0.1), 2)
    assert await terminal(queued) == ("cancelled", None)
    assert worker.sent == [{"op": "cancel", "id": 1, "req": 0}]  # asked to stop at its next step...
    assert await terminal(running) == ("cancelled", None)  # ...and given up on when it did not
    assert scheduler.running == 0 and scheduler.waiting == 0
    for job in (running, queued):
        await assert_no_second_terminal(job)
