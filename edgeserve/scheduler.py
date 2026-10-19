"""Request queue in front of the single GPU worker.

Requests are admitted only after every check that can fail (context length, queue space,
memory) has passed, so overload is reported as an HTTP status rather than mid-stream. The
dispatcher runs one generate command at a time on the worker; with --max-batch-size > 1 it
takes the oldest queued request plus later ones that can share its batch (Edge-LLM batches
need identical sampling settings and output limits), so a batch never waits for latecomers.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
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
    t_queued_ns: int = 0
    t_dispatch_ns: int | None = None
    batch_size: int = 1
    batch_contexts: list[list[int]] | None = None  # per step: context length of every sequence in the batch
    state: str = "new"  # new | queued | running | finished
    events: asyncio.Queue = field(default_factory=asyncio.Queue)
    cancel_requested: bool = False
    _generation: Generation | None = None
    _req_index: int = 0

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
    ) -> None:
        self._get_worker = get_worker
        self._max_batch = max_batch_size
        self._kv_capacity = kv_capacity
        self._queue_size = queue_size
        self._queue_timeout_ns = int(queue_timeout_s * 1e9)
        self._on_change = on_change or (lambda: None)
        self._queue: deque[Job] = deque()
        self._running: list[Job] = []
        self._inflight: set[str] = set()
        self._wake = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._accepting = True
        self._task: asyncio.Task | None = None
        self._expiry: asyncio.TimerHandle | None = None
        self._e2e_ewma_s = 30.0  # refined from observed batches; only used for Retry-After hints

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

    def retry_after_s(self) -> int:
        batches_ahead = (len(self._queue) + len(self._running)) / self._max_batch + 1
        return int(min(120, max(1, batches_ahead * self._e2e_ewma_s)))

    def is_inflight(self, custom_id: str) -> bool:
        return custom_id in self._inflight

    def check_capacity(self) -> None:
        if not self._accepting:
            raise AdmissionError(503, "server_shutting_down", "the server is shutting down",
                                 retry_after=self.retry_after_s(), err_type="server_error")
        if len(self._queue) >= self._queue_size:
            raise AdmissionError(503, "server_is_overloaded",
                                 f"the request queue is full ({self._queue_size} waiting); retry later",
                                 retry_after=self.retry_after_s(), err_type="server_error")

    def submit(self, job: Job) -> None:
        """Queue an admitted job (callers ran check_capacity and the context checks first)."""
        self.check_capacity()
        if job.custom_id in self._inflight:
            raise AdmissionError(409, "duplicate_request_id",
                                 f"a request with id {job.custom_id!r} is already in progress")
        self._inflight.add(job.custom_id)
        job.state = "queued"
        job.t_queued_ns = time.monotonic_ns()
        self._queue.append(job)
        self._idle.clear()
        self._wake.set()
        self._arm_expiry()
        self._on_change()

    def release(self, job: Job) -> None:
        """Forget a job once its response and record are complete (a job nobody waits for any more is cancelled)."""
        if job.state != "finished":
            self.cancel(job)
        self._inflight.discard(job.custom_id)

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
        if self._expiry is not None:
            self._expiry.cancel()
            self._expiry = None

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
                    self._maybe_idle()
                    self._on_change()
            self._maybe_idle()

    def _arm_expiry(self) -> None:
        """Expire queued jobs on time even while no worker is available or a batch is running."""
        if self._expiry is None and self._queue:
            due_ns = self._queue[0].t_queued_ns + self._queue_timeout_ns - time.monotonic_ns()
            self._expiry = asyncio.get_running_loop().call_later(max(0.0, due_ns / 1e9), self._expiry_due)

    def _expiry_due(self) -> None:
        self._expiry = None
        self._expire_old()
        self._arm_expiry()

    def _expire_old(self) -> None:
        now = time.monotonic_ns()
        expired = False
        while self._queue and now - self._queue[0].t_queued_ns > self._queue_timeout_ns:
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
        started = time.monotonic()
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

        async for event in generation.events():
            kind = event["ev"]
            if kind == "text":
                req = event.get("req", 0)
                if 0 <= req < len(batch):
                    batch[req].events.put_nowait(("text", event))
            elif kind == "done":
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
        self._e2e_ewma_s = 0.8 * self._e2e_ewma_s + 0.2 * elapsed

    def _maybe_idle(self) -> None:
        if not self._queue and not self._running:
            self._idle.set()


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


__all__ = ["AdmissionError", "Job", "Scheduler"]
