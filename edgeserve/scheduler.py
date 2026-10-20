"""Request queue in front of the single GPU worker.

Requests are admitted only after every check that can fail (context length, queue space,
memory, the client's own limit) has passed, so overload is reported as an HTTP status rather
than mid-stream. The dispatcher runs one generate command at a time on the worker; with
--max-batch-size > 1 it takes the oldest queued request plus later ones that can share its batch
(Edge-LLM batches need identical sampling settings and output limits), so a batch never waits for
latecomers. A request still waiting after --queue-timeout fails; with --request-timeout, a request
not finished that long after it arrived is stopped (queued: dropped; running: cancelled at the next
token). RateLimiter is the per-client requests-per-minute limit.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from edgeserve.worker import Generation, WorkerClient, WorkerError

log = logging.getLogger("edgeserve.scheduler")

_CANCEL_GRACE_S = 10.0  # drain(): how long cancelled requests get to stop before they are given up on


class AdmissionError(Exception):
    """A request refused before it was queued; maps to an OpenAI-style HTTP error."""

    def __init__(self, status: int, code: str, message: str, retry_after: int | None = None,
                 err_type: str = "invalid_request_error", param: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.retry_after = retry_after
        self.err_type = err_type
        self.param = param


@dataclass(eq=False)
class Job:
    custom_id: str
    attempt_id: str
    endpoint: str  # "/v1/chat/completions" | "/v1/completions"
    item: dict  # worker request: {"messages": [...]} or {"prompt": "..."}, plus "stop" / "seed"
    sampling: dict  # temperature, top_p, top_k, max_new_tokens
    stream: bool
    prompt_tokens: int
    t_arrival_ns: int
    t_arrival_unix_ns: int
    request_echo: dict  # what the client sent (messages or prompt), stored in the record
    params_echo: dict  # effective generation parameters, stored in the record
    metadata: dict | None = None
    client: str = "local"  # the API key's client name, or the client's address on a server without keys
    deadline_ns: int | None = None  # --request-timeout: monotonic time by which the request must have finished
    timed_out: bool = False  # stopped because of --request-timeout
    t_queued_ns: int = 0
    t_dispatch_ns: int | None = None
    batch_size: int = 1
    batch_contexts: list[list[int]] | None = None  # per step: context length of every sequence in the batch
    state: str = "new"  # new | queued | running | finished
    events: asyncio.Queue = field(default_factory=asyncio.Queue)
    cancel_requested: bool = False
    _generation: Generation | None = None
    _req_index: int = 0
    _counted: bool = False  # counted in its client's requests (until released)

    def batch_key(self) -> tuple:
        kind = "prompt" if "prompt" in self.item else "messages"
        s = self.sampling
        return kind, s["temperature"], s["top_p"], s["top_k"], s["max_new_tokens"]


class Scheduler:
    def __init__(
        self,
        get_worker: Callable[[], WorkerClient | None],
        max_batch_size: int,
        kv_capacity: int,
        queue_size: int,
        queue_timeout_s: float,
        on_change: Callable[[], None] | None = None,
        request_timeout_s: float = 0,
        max_client_requests: int = 0,
    ) -> None:
        self._get_worker = get_worker
        self._max_batch = max_batch_size
        self._kv_capacity = kv_capacity
        self._queue_size = queue_size
        self._queue_timeout_ns = int(queue_timeout_s * 1e9)
        self._request_timeout_ns = int(request_timeout_s * 1e9)
        self._max_client = max_client_requests
        self._on_change = on_change or (lambda: None)
        self._queue: deque[Job] = deque()
        self._running: list[Job] = []
        self._inflight: set[str] = set()
        self._per_client: dict[str, int] = {}  # client -> its requests waiting or running
        self._wake = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._accepting = True
        self._task: asyncio.Task | None = None
        self._expiry: asyncio.TimerHandle | None = None
        self._deadlines: asyncio.TimerHandle | None = None
        self._batch_s: float | None = None  # moving average of a batch's run time (for Retry-After hints)
        self._step_s: float | None = None  # moving average of one decode step (bounds a batch that runs long)
        self._batch_started: float | None = None  # when the running batch started (monotonic s)

    # ------------------------------------------------------------------ admission

    @property
    def waiting(self) -> int:
        return len(self._queue)

    @property
    def running(self) -> int:
        return len(self._running)

    def kv_usage(self) -> float:
        if not self._running or not self._kv_capacity:
            return 0.0
        used = sum(j.prompt_tokens + j.sampling["max_new_tokens"] for j in self._running)
        return min(1.0, used / (self._kv_capacity * self._max_batch))

    def retry_after_s(self, client: str | None = None) -> int:
        """Seconds until a refused request can get in: when the running batch ends, the next one leaves the
        queue. Its expected remaining time from the measured batch times, bounded by its token budget when it runs
        longer than usual; for one client (``client``), also the batches ahead of its oldest waiting request.
        1 to 60 s; 5 s before any batch ran."""
        if self._batch_s is None:
            return 5
        remaining = 0.0
        if self._batch_started is not None:
            elapsed = time.monotonic() - self._batch_started
            remaining = self._batch_s - elapsed
            if remaining < 1 and self._step_s and self._running:  # longer than usual: at most its token budget
                remaining = self._running[0].sampling["max_new_tokens"] * self._step_s - elapsed
        if client is not None and not any(job.client == client for job in self._running):
            ahead = next((i for i, job in enumerate(self._queue) if job.client == client), 0)
            remaining += (ahead // self._max_batch + 1) * self._batch_s  # its oldest request has yet to run
        return int(min(60, max(1, math.ceil(remaining))))

    def is_inflight(self, custom_id: str) -> bool:
        return custom_id in self._inflight

    def check_client(self, client: str) -> None:
        """--max-client-requests: refuse a client that already has that many requests waiting or running."""
        if self._max_client and self._per_client.get(client, 0) >= self._max_client:
            raise AdmissionError(429, "too_many_requests",
                                 f"client {client!r} already has {self._max_client} requests waiting or running "
                                 "(the server's --max-client-requests); retry when one has finished",
                                 retry_after=self.retry_after_s(client), err_type="rate_limit_error")

    def check_capacity(self) -> None:
        if not self._accepting:
            raise AdmissionError(503, "server_shutting_down", "the server is shutting down",
                                 retry_after=self.retry_after_s(), err_type="server_error")
        if 0 <= self._queue_size <= len(self._queue):  # -1: no limit
            raise AdmissionError(503, "server_is_overloaded",
                                 f"the request queue is full ({self._queue_size} waiting); retry later",
                                 retry_after=self.retry_after_s(), err_type="server_error")

    def submit(self, job: Job) -> None:
        """Queue an admitted job (callers ran check_capacity and the context checks first)."""
        self.check_capacity()
        if job.custom_id in self._inflight:
            raise AdmissionError(409, "duplicate_request_id",
                                 f"a request with id {job.custom_id!r} is already in progress")
        self.check_client(job.client)
        self._inflight.add(job.custom_id)
        self._per_client[job.client] = self._per_client.get(job.client, 0) + 1
        job._counted = True
        job.state = "queued"
        job.t_queued_ns = time.monotonic_ns()
        if self._request_timeout_ns:
            job.deadline_ns = job.t_arrival_ns + self._request_timeout_ns
        self._queue.append(job)
        self._idle.clear()
        self._wake.set()
        self._arm_expiry()
        self._arm_deadlines()
        self._on_change()

    def release(self, job: Job) -> None:
        """Forget a job once its response and record are complete (a job nobody waits for any more is cancelled)."""
        if job.state != "finished":
            self.cancel(job)
        self._inflight.discard(job.custom_id)
        self._uncount(job)

    def _uncount(self, job: Job) -> None:
        """Give the job's place back to its client (once)."""
        if job._counted:
            job._counted = False
            left = self._per_client.get(job.client, 1) - 1
            if left > 0:
                self._per_client[job.client] = left
            else:
                self._per_client.pop(job.client, None)

    def cancel(self, job: Job) -> None:
        """Client went away: drop a queued job, or ask the worker to stop a running one at its next step."""
        if job.state == "queued":
            with contextlib.suppress(ValueError):
                self._queue.remove(job)
            job.state = "finished"
            job.events.put_nowait(("cancelled", None))
            self._maybe_idle()
            self._on_change()
        elif job.state == "running" and not job.cancel_requested:
            job.cancel_requested = True
            self._uncount(job)  # it stops at its next token: its client need not wait for the rest of the batch
            if job._generation is not None:
                asyncio.ensure_future(_quiet(job._generation.cancel(job._req_index)))

    def wake(self) -> None:
        """Called when a worker becomes available again."""
        self._wake.set()

    # ------------------------------------------------------------------ dispatching

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="scheduler")

    async def drain(self, timeout: float) -> None:
        """Stop accepting work, wait for queued and running requests, cancel what is left."""
        self._accepting = False
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._idle.wait(), timeout)
        for job in list(self._queue):
            self.cancel(job)
        for job in list(self._running):
            self.cancel(job)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._idle.wait(), _CANCEL_GRACE_S)
        if self._task:
            self._task.cancel()  # a batch still running now ends as cancelled
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        for timer in (self._expiry, self._deadlines):
            if timer is not None:
                timer.cancel()
        self._expiry = self._deadlines = None

    def fail_queued(self, error: WorkerError) -> None:
        """Fail every queued job (the worker cannot be brought back)."""
        while self._queue:
            job = self._queue.popleft()
            job.state = "finished"
            job.events.put_nowait(("error", error))
        self._maybe_idle()
        self._on_change()

    async def _run(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            while self._queue:
                self._expire_old()
                self._drop_overdue()
                worker = self._get_worker()
                if not self._queue or worker is None or not worker.alive:
                    break  # the service wakes us when a worker is ready again
                batch = self._take_batch()
                try:
                    await self._run_batch(worker, batch)
                except asyncio.CancelledError:
                    _finish(batch, ("cancelled", None))
                    raise
                except Exception:  # never let one batch kill the dispatcher
                    log.exception("unexpected error while running a batch")
                    _finish(batch, ("error", WorkerError("internal_error", "internal server error")))
                finally:
                    self._running = []
                    self._batch_started = None
                    self._maybe_idle()
                    self._on_change()
            self._maybe_idle()

    def _arm_expiry(self) -> None:
        """Expire queued jobs on time even while no worker is available or a batch is running."""
        if self._expiry is None and self._queue and self._queue_timeout_ns:  # 0: no limit
            due_ns = self._queue[0].t_queued_ns + self._queue_timeout_ns - time.monotonic_ns()
            self._expiry = asyncio.get_running_loop().call_later(max(0.0, due_ns / 1e9), self._expiry_due)

    def _expiry_due(self) -> None:
        self._expiry = None
        self._expire_old()
        self._arm_expiry()

    def _arm_deadlines(self) -> None:
        """--request-timeout: check the waiting and running requests every second while there are any."""
        if self._deadlines is None and self._request_timeout_ns and (self._queue or self._running):
            self._deadlines = asyncio.get_running_loop().call_later(1.0, self._deadlines_due)

    def _deadlines_due(self) -> None:
        self._deadlines = None
        self._drop_overdue()
        now = time.monotonic_ns()
        for job in self._running:
            if job.deadline_ns is not None and now >= job.deadline_ns and not job.timed_out:
                job.timed_out = True
                self.cancel(job)  # alone in its batch, its "done" event follows, with finish_reason "cancelled"
                if len(self._running) > 1 and job.state != "finished":
                    # Edge-LLM reports a batch once, when all its sequences have ended: answer this one now
                    job.state = "finished"
                    job.events.put_nowait(("error", WorkerError("time_limit_exceeded", self._time_limit_message())))
        self._arm_deadlines()

    def _drop_overdue(self) -> None:
        """Fail the waiting requests whose --request-timeout has passed (also checked before each dispatch)."""
        if not self._request_timeout_ns:
            return
        now = time.monotonic_ns()
        overdue = [job for job in self._queue if job.deadline_ns is not None and now >= job.deadline_ns]
        for job in overdue:
            self._queue.remove(job)
            job.state = "finished"
            job.timed_out = True
            job.events.put_nowait(("error", WorkerError("time_limit_exceeded", self._time_limit_message())))
        if overdue:
            self._maybe_idle()
            self._on_change()

    def _time_limit_message(self) -> str:
        return (f"the request did not finish within {self._request_timeout_ns / 1e9:g} s of its arrival (the "
                "server's --request-timeout)")

    def _expire_old(self) -> None:
        now = time.monotonic_ns()
        expired = False
        while self._queue_timeout_ns and self._queue and now - self._queue[0].t_queued_ns > self._queue_timeout_ns:
            job = self._queue.popleft()
            job.state = "finished"
            job.events.put_nowait(("error", WorkerError("queue_timeout",
                                                        "the request waited too long for the GPU; retry later")))
            expired = True
        if expired:
            self._maybe_idle()
            self._on_change()

    def _take_batch(self) -> list[Job]:
        head = self._queue.popleft()
        batch = [head]
        if self._max_batch > 1:
            key = head.batch_key()
            for job in list(self._queue):
                if len(batch) >= self._max_batch:
                    break
                if job.batch_key() == key:
                    self._queue.remove(job)
                    batch.append(job)
        return batch

    async def _run_batch(self, worker: WorkerClient, batch: list[Job]) -> None:
        now = time.monotonic_ns()
        for index, job in enumerate(batch):
            job.state = "running"
            job._req_index = index
            job.t_dispatch_ns = now
            job.batch_size = len(batch)
        self._running = batch
        self._on_change()
        started = self._batch_started = time.monotonic()
        try:
            generation = await worker.generate([j.item for j in batch], batch[0].sampling,
                                               stream=any(j.stream for j in batch))
        except WorkerError as e:
            _finish(batch, ("error", e))
            return
        for job in batch:
            job._generation = generation
            if job.cancel_requested:  # disconnected while being dispatched
                await _quiet(generation.cancel(job._req_index))

        steps = 0  # decode steps of the batch, from its "done" event
        async for event in generation.events():
            kind = event["ev"]
            if kind == "text":
                req = event.get("req", 0)
                if 0 <= req < len(batch):
                    batch[req].events.put_nowait(("text", event))
            elif kind == "done":
                steps = len((event.get("steps") or {}).get("t_ns", []))
                contexts = _batch_contexts(event.get("results", []), batch)
                for result in event.get("results", []):
                    req = result.get("req", 0)
                    if 0 <= req < len(batch) and batch[req].state != "finished":
                        job = batch[req]
                        job.batch_contexts = contexts
                        job.state = "finished"
                        job.events.put_nowait(("done", event, result))
            elif kind == "error":
                _finish(batch, ("error", WorkerError(event.get("code", "worker_error"), event.get("message", ""),
                                                     bool(event.get("fatal")))))
        # a done event that omitted a sequence must not leave its client hanging
        _finish(batch, ("error", WorkerError("internal_error", "the worker returned no result")))
        elapsed = time.monotonic() - started
        self._batch_s = elapsed if self._batch_s is None else 0.7 * self._batch_s + 0.3 * elapsed
        if steps:
            per_step = elapsed / steps
            self._step_s = per_step if self._step_s is None else 0.7 * self._step_s + 0.3 * per_step

    def _maybe_idle(self) -> None:
        if not self._queue and not self._running:
            self._idle.set()


class RateLimiter:
    """--client-rate-limit: at most N requests per minute per client, as a token bucket (a client may send N at
    once, then one more every 60/N seconds)."""

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic) -> None:
        self._rate = per_minute / 60.0
        self._capacity = float(per_minute)
        self._clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}  # client -> (tokens, at)
        self._prune_at = 4096

    def take(self, client: str) -> None:
        """Count one request; raise AdmissionError(429) when the client is over its limit."""
        if not self._rate:
            return
        now = self._clock()
        tokens, at = self._buckets.get(client, (self._capacity, now))
        tokens = min(self._capacity, tokens + (now - at) * self._rate)
        if tokens < 1.0:
            self._buckets[client] = (tokens, now)
            wait = math.ceil((1.0 - tokens) / self._rate)
            raise AdmissionError(429, "rate_limit_exceeded",
                                 f"client {client!r} is over the server's --client-rate-limit ({self._capacity:g} "
                                 f"requests a minute: that many at once, then one every {60 / self._capacity:g} s); "
                                 f"retry in {wait} s", retry_after=wait, err_type="rate_limit_error")
        self._buckets[client] = (tokens - 1.0, now)
        if len(self._buckets) > self._prune_at:  # forget clients whose allowance is full again
            self._buckets = {c: (t, a) for c, (t, a) in self._buckets.items()
                             if t + (now - a) * self._rate < self._capacity}
            self._prune_at = max(4096, 2 * len(self._buckets))  # so that pruning costs O(1) per request


def _finish(jobs: list[Job], event: tuple) -> None:
    """Give every job that has not finished yet its one terminal event."""
    for job in jobs:
        if job.state != "finished":
            job.state = "finished"
            job.events.put_nowait(event)


def _batch_contexts(results: list[dict], batch: list[Job]) -> list[list[int]]:
    """Per step, the context length of every sequence still generating (for batch-level MFU/MBU)."""
    def prompt_tokens(r: dict) -> int:
        reported, req = r.get("prompt_tokens", -1), r.get("req", -1)
        if isinstance(reported, int) and reported >= 0:
            return reported
        return batch[req].prompt_tokens if isinstance(req, int) and 0 <= req < len(batch) else 0  # -1: unknown

    lengths = [(prompt_tokens(r), len(r.get("token_ids", []))) for r in results]
    steps = max((n for _, n in lengths), default=0)
    return [[p + k for p, n in lengths if k < n] for k in range(steps)]


async def _quiet(coroutine) -> None:
    with contextlib.suppress(WorkerError):
        await coroutine


__all__ = ["AdmissionError", "Job", "RateLimiter", "Scheduler"]
