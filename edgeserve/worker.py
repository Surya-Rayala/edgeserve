"""Supervised connection to the edgeserve-worker process (the only process that touches the GPU).

The worker speaks line-delimited JSON (see docs/ARCHITECTURE.md, "Worker protocol"): commands
on its stdin, events on its stdout, logs on its stderr. Keeping the engine in its own process
means a CUDA fault or an out-of-memory kill takes down only the worker; this client then fails
the affected requests with an error and the service starts a fresh worker.

On Jetson the kernel OOM killer cannot see GPU memory (it is not counted in RSS), so the worker
volunteers as the preferred victim (oom_score_adj 900): freeing its multi-GB engine is what
actually relieves memory pressure, and the front-end survives to report what happened.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
from collections.abc import AsyncIterator, Callable
from pathlib import Path

from edgeserve.engine import NativeBuild

log = logging.getLogger("edgeserve.worker")

_LINE_LIMIT = 64 * 1024**2  # a "done" event carries per-step arrays for the whole batch
_EXIT_DRAIN_S = 5.0  # after the worker exits, how long to wait for the events it wrote before exiting


class WorkerError(RuntimeError):
    def __init__(self, code: str, message: str, fatal: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.fatal = fatal


class Generation:
    """Events of one generate command: "text" events (when streaming), then exactly one "done" or "error"."""

    def __init__(self, job_id: int, client: WorkerClient) -> None:
        self.job_id = job_id
        self._client = client
        self._queue: asyncio.Queue[dict] = asyncio.Queue()

    def _put(self, event: dict) -> None:
        self._queue.put_nowait(event)

    async def events(self) -> AsyncIterator[dict]:
        while True:
            event = await self._queue.get()
            yield event
            if event["ev"] in ("done", "error"):
                return

    async def cancel(self, req: int | None = None) -> None:
        await self._client.send({"op": "cancel", "id": self.job_id, **({"req": req} if req is not None else {})})


class WorkerClient:
    def __init__(
        self,
        native: NativeBuild,
        engine_dir: Path,
        on_exit: Callable[[int | None], None] | None = None,
        cuda_graph: bool = True,
    ) -> None:
        self._native = native
        self._engine_dir = engine_dir
        self._on_exit = on_exit
        self._cuda_graph = cuda_graph
        self._proc: asyncio.subprocess.Process | None = None
        self._ids = itertools.count(1)
        self._pending_counts: dict[int, asyncio.Future[list[int]]] = {}
        self._generations: dict[int, Generation] = {}
        self._write_lock = asyncio.Lock()
        self._ready: asyncio.Future[dict] | None = None
        self._tasks: list[asyncio.Task] = []
        self._events_task: asyncio.Task | None = None
        self._stopping = False
        self._fatal = False
        self.ready_info: dict = {}

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    @property
    def alive(self) -> bool:
        # A fatal error event means the worker exits next: nothing more may be sent to it.
        return self._proc is not None and self._proc.returncode is None and not self._fatal

    async def start(self, timeout: float = 600.0) -> dict:
        """Spawn the worker and wait until the engine is loaded and warmed up; returns the ready event."""
        args = [str(self._native.worker), "--engine-dir", str(self._engine_dir)]
        if not self._cuda_graph:
            args.append("--no-cuda-graph")
        loop = asyncio.get_running_loop()
        self._ready = loop.create_future()
        self._proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._native.env(),
            limit=_LINE_LIMIT,
            preexec_fn=_volunteer_for_oom_killer,
            start_new_session=True,  # Ctrl-C in a terminal must not kill it before the server drains
        )
        self._events_task = asyncio.create_task(self._read_events(), name="worker-events")
        self._tasks = [
            self._events_task,
            asyncio.create_task(self._read_logs(), name="worker-logs"),
            asyncio.create_task(self._wait_exit(), name="worker-exit"),
        ]
        try:
            self.ready_info = await asyncio.wait_for(asyncio.shield(self._ready), timeout)
        except asyncio.TimeoutError:
            self._ready.cancel()
            await self.stop(timeout=5)
            raise WorkerError("load_timeout", f"the worker did not become ready within {timeout:.0f} s", fatal=True)
        except BaseException:
            # A load error, or start() cancelled (shutdown while loading): never leave a worker behind that
            # goes on loading a multi-GB engine.
            self._ready.cancel()
            self._stopping = True
            self._kill()
            raise
        return self.ready_info

    async def send(self, command: dict) -> None:
        if not self.alive or self._proc is None or self._proc.stdin is None:
            raise WorkerError("worker_unavailable", "the inference worker is not running")
        try:
            # The worker answers a line it cannot parse without an id, so such values must never reach it:
            # lone surrogates have no UTF-8 form and NaN / Infinity are not JSON.
            data = (json.dumps(command, ensure_ascii=False, allow_nan=False) + "\n").encode()
        except ValueError as e:
            raise WorkerError("bad_request", f"the request contains a value that cannot be processed ({e})") from e
        async with self._write_lock:
            try:
                self._proc.stdin.write(data)
                await self._proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as e:
                raise WorkerError("worker_unavailable", f"the inference worker went away ({e})") from e

    async def count(self, items: list[dict], timeout: float = 30.0) -> list[int]:
        """Prompt tokens per request after the chat template (answered even while a generation runs)."""
        job_id = next(self._ids)
        future: asyncio.Future[list[int]] = asyncio.get_running_loop().create_future()
        self._pending_counts[job_id] = future
        try:
            await self.send({"op": "count", "id": job_id, "requests": items})
            try:
                return await asyncio.wait_for(future, timeout)
            except asyncio.TimeoutError:
                raise WorkerError("worker_unavailable",
                                  f"the inference worker did not answer within {timeout:.0f} s") from None
        finally:
            self._pending_counts.pop(job_id, None)

    async def generate(self, items: list[dict], sampling: dict, stream: bool) -> Generation:
        job_id = next(self._ids)
        generation = Generation(job_id, self)
        self._generations[job_id] = generation
        try:
            await self.send({"op": "generate", "id": job_id, "stream": stream, "requests": items, "sampling": sampling})
        except WorkerError:
            self._generations.pop(job_id, None)
            raise
        return generation

    async def stop(self, timeout: float = 10.0) -> None:
        """Ask the worker to exit (cancelling anything running), then kill it if it does not."""
        self._stopping = True
        proc = self._proc
        if proc is None or proc.returncode is not None:
            return
        with contextlib.suppress(WorkerError):
            await self.send({"op": "shutdown"})
        try:
            await asyncio.wait_for(proc.wait(), timeout)
        except asyncio.TimeoutError:
            log.warning("worker did not exit within %.0f s; killing it", timeout)
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()

    def _kill(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                self._proc.kill()

    # ------------------------------------------------------------------ event plumbing

    async def _read_events(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            async for raw in _read_lines(self._proc.stdout):
                # Never log the line itself: events carry output text (and a truncated "done" starts with it).
                try:
                    event = json.loads(raw)
                except ValueError:
                    log.error("unparseable worker event (%d bytes)", len(raw))
                    continue
                if not isinstance(event, dict):
                    log.error("unexpected worker event (a JSON %s, %d bytes)", type(event).__name__, len(raw))
                    continue
                self._dispatch(event)
        except Exception:
            # Nothing the worker does can complete without its events: replace it (the exit fails what is pending).
            log.exception("lost the event stream of the inference worker; stopping it")
            self._kill()

    def _dispatch(self, event: dict) -> None:
        kind = event.get("ev")
        job_id = event.get("id")
        if kind == "ready":
            if self._ready and not self._ready.done():
                self._ready.set_result(event)
        elif kind == "count":
            future = self._pending_counts.get(job_id)
            if future and not future.done():
                future.set_result(list(event.get("prompt_tokens", [])))
        elif kind in ("text", "done"):
            generation = self._generations.get(job_id)
            if generation:
                generation._put(event)
                if kind == "done":
                    self._generations.pop(job_id, None)
        elif kind == "error":
            error = WorkerError(event.get("code", "worker_error"), event.get("message", ""), bool(event.get("fatal")))
            if error.fatal:
                self._fatal = True
            if job_id is None:
                if self._ready and not self._ready.done():
                    self._ready.set_exception(error)
                else:
                    log.error("worker error: %s: %s", error.code, error)
            elif job_id in self._pending_counts:
                future = self._pending_counts[job_id]
                if not future.done():
                    future.set_exception(error)
            elif job_id in self._generations:
                self._generations.pop(job_id)._put(event)
        else:
            log.debug("ignoring worker event %s", kind)

    async def _read_logs(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        try:
            async for raw in _read_lines(self._proc.stderr):
                line = raw.decode(errors="replace").rstrip()
                if not line:
                    continue
                if "[ERROR]" in line or " error:" in line:
                    log.error("%s", line)
                elif "[WARNING]" in line or " warning:" in line:
                    log.warning("%s", line)
                else:
                    log.debug("%s", line)
        except Exception:
            # An undrained stderr pipe blocks the worker at its next log line.
            log.exception("lost the log stream of the inference worker; stopping it")
            self._kill()

    async def _wait_exit(self) -> None:
        assert self._proc is not None
        code = await self._proc.wait()
        if self._events_task is not None:
            # What the worker wrote before exiting (its load error, a last "done") is dispatched first.
            await asyncio.wait({self._events_task}, timeout=_EXIT_DRAIN_S)
        message = f"the inference worker exited with code {code}"
        if code in (-9, 137):
            message += " (killed; most likely out of memory)"
        error = WorkerError("worker_exited", message, fatal=True)
        if self._ready and not self._ready.done():
            self._ready.set_exception(error)
        for future in self._pending_counts.values():
            if not future.done():
                future.set_exception(error)
        for job_id, generation in list(self._generations.items()):
            generation._put({"ev": "error", "id": job_id, "code": error.code, "message": message, "fatal": True})
        self._generations.clear()
        if not self._stopping:
            log.error("%s", message)
            if self._on_exit:
                self._on_exit(code)


async def _read_lines(stream: asyncio.StreamReader) -> AsyncIterator[bytes]:
    """Lines of any length: one longer than the stream's buffer limit is collected in pieces instead of failing."""
    pieces: list[bytes] = []
    while True:
        try:
            line = await stream.readuntil(b"\n")
        except asyncio.LimitOverrunError as e:
            pieces.append(await stream.readexactly(e.consumed))
            continue
        except asyncio.IncompleteReadError as e:  # end of the stream
            if pieces or e.partial:
                yield b"".join(pieces) + e.partial
            return
        if pieces:
            line = b"".join(pieces) + line
            pieces = []
        yield line


def _volunteer_for_oom_killer() -> None:
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("900")
    except OSError:
        pass


__all__ = ["Generation", "WorkerClient", "WorkerError"]
