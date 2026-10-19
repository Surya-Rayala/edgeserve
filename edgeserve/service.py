"""The running server: startup pipeline, worker supervision, admission and result bookkeeping.

Startup runs in the background so the HTTP port is up immediately (``/health`` answers, and
``/health/ready`` reports the current step: probing, building_engine, loading, ...). Steps:

1. probe the GPU (through the worker binary) and the host;
2. validate the model folder and its compatibility with this GPU and Edge-LLM build;
3. check that the engine fits in memory;
4. reuse the cached TensorRT engine for (model, limits, toolchain, GPU) or build it;
5. start telemetry and the worker (load + CUDA graphs + warm-up), then accept requests.

If the worker dies while serving (CUDA fault, OOM kill), in-flight requests fail with 503, a
fresh worker is started with back-off, and the server reports itself failed after
MAX_WORKER_RESTARTS restarts within RESTART_WINDOW_S.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import logging
import secrets
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from edgeserve import settings as limits
from edgeserve.engine import BuildProfile, EngineCache, EngineError, EngineInfo, NativeBuild
from edgeserve.hardware import Platform, PlatformError, probe, probe_gpu, refresh_clocks
from edgeserve.metrics import PrometheusMetrics, RecordContext, build_record
from edgeserve.model import ModelError, ModelInfo, estimate_memory, fingerprint, load_model, model_name
from edgeserve.records import OutputDir
from edgeserve.scheduler import Job, Scheduler
from edgeserve.settings import Settings
from edgeserve.telemetry import Telemetry
from edgeserve.worker import WorkerClient, WorkerError

log = logging.getLogger("edgeserve")
request_log = logging.getLogger("edgeserve.requests")

# GPUs with kernels for Edge-LLM's INT4 GEMM plugin (CuTe DSL); other GPUs need a different export.
INT4_PLUGIN_SMS = frozenset({80, 86, 87, 89, 100, 101, 110, 120, 121})


class StartupError(RuntimeError):
    pass


class Service:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.state = "starting"
        self.detail = "starting"
        self.native: NativeBuild | None = None
        self.gpu: dict | None = None
        self.platform: Platform | None = None
        self.model: ModelInfo | None = None
        self.engine: EngineInfo | None = None
        self.worker: WorkerClient | None = None
        self.telemetry: Telemetry | None = None
        self.scheduler: Scheduler | None = None
        self.output: OutputDir | None = None
        self.ctx: RecordContext | None = None
        self.started_at = time.time()
        self.model_name = model_name(settings.model)  # what clients call it (before the model is loaded)
        self.prometheus = PrometheusMetrics(model_name=self.model_name)
        self._restarts: deque[float] = deque()
        self._tasks: list[asyncio.Task] = []
        self._stopping = False
        self._cache: EngineCache | None = None
        self._build_cancel = threading.Event()
        # One writer thread: records of the same ID land in commit order, and stop() can flush them all.
        self._records = ThreadPoolExecutor(max_workers=1, thread_name_prefix="edgeserve-records")

    # ------------------------------------------------------------------ lifecycle

    def set_state(self, state: str, detail: str | None = None) -> None:
        self.state = state
        self.detail = detail or state
        self.prometheus.set_state(state)
        log.info("state: %s (%s)", state, self.detail)

    @property
    def ready(self) -> bool:
        return self.state == "ready" and self.worker is not None and self.worker.alive

    async def start(self) -> None:
        self._tasks.append(asyncio.create_task(self._startup(), name="startup"))

    async def _startup(self) -> None:
        try:
            await self._prepare()
            await self._start_worker()
            assert self.scheduler is not None
            self.scheduler.start()
            self.set_state("draining" if self._stopping else "ready",
                           f"serving {self.ctx.served_name if self.ctx else ''}")
            self._tasks.append(asyncio.create_task(self._refresh_clocks(), name="clocks"))
        except (StartupError, EngineError, ModelError, PlatformError, WorkerError, ValueError, OSError) as e:
            if self._stopping:  # cut short by a shutdown (a cancelled engine build), not a failure
                log.info("startup stopped: %s", e)
                return
            self.set_state("failed", str(e))
            log.error("startup failed: %s", e)
        except Exception as e:  # report anything unexpected instead of hanging in "starting"
            self.set_state("failed", f"unexpected error: {e}")
            log.exception("startup failed")

    async def prepare_engine(self) -> EngineInfo:
        """Steps 1-4 only (used by ``edgeserve build``)."""
        await self._prepare()
        assert self.engine is not None
        return self.engine

    async def _prepare(self) -> None:
        s = self.settings
        self.set_state("starting", "probing the GPU")
        self.native = NativeBuild.locate(s.native_dir)
        self.gpu = await asyncio.to_thread(probe_gpu, self.native.worker, self.native.env())
        check_runtime(self.native, self.gpu)
        self.platform = await asyncio.to_thread(probe, self.gpu)
        log.info("platform: %s", json.dumps(self.platform.as_dict()))

        self.set_state("starting", "checking the model")
        self.model = await asyncio.to_thread(load_model, s.model)
        check_compatibility(self.model, self.gpu)
        for note in generation_notes(self.model):
            log.warning("%s", note)
        profile = BuildProfile(max_batch_size=s.max_batch_size, max_input_len=s.max_input_len,
                               max_new_tokens=default_max_new_tokens(s, self.model))
        check_memory(self.model, profile, self.platform, self.gpu)

        self.set_state("starting", "checking the engine cache")
        digest = await asyncio.to_thread(fingerprint, self.model.path)
        self._cache = cache = EngineCache(s.cache_dir)
        loop = asyncio.get_running_loop()

        def progress(message: str) -> None:
            loop.call_soon_threadsafe(self._building, message)

        building = loop.run_in_executor(None, functools.partial(cache.ensure, self.model, digest, profile,
                                                                self.native, self.gpu, progress,
                                                                cancel=self._build_cancel))
        try:
            self.engine = await asyncio.shield(building)
        except asyncio.CancelledError:
            # llm_build runs in its own session and would outlive this process: stop it and let it clean up.
            self._build_cancel.set()
            with contextlib.suppress(BaseException):
                await building
            raise
        if self.engine.built_now:
            build = self.engine.manifest.get("build", {})
            log.info("engine built in %s s, peak memory %.2f GiB", build.get("seconds"),
                     (build.get("peak_rss_bytes") or 0) / 2**30)

        self.telemetry = Telemetry(self.platform)
        await asyncio.to_thread(self.telemetry.start)
        self.prometheus.set_telemetry(self.telemetry)
        self.ctx = RecordContext(model=self.model, engine=self.engine, platform=self.platform,
                                 telemetry=self.telemetry, served_name=self.model.name)
        self.prometheus.set_model_info(self.ctx)
        if s.output_dir:
            self.output = OutputDir(s.output_dir)
        self.scheduler = Scheduler(
            get_worker=lambda: self.worker if self.state in ("ready", "draining") else None,
            max_batch_size=self.engine.profile.max_batch_size,
            kv_capacity=self.engine.profile.kv_capacity,
            queue_size=limits.QUEUE_SIZE,
            queue_timeout_s=limits.QUEUE_TIMEOUT_S,
            on_change=self._queue_changed,
        )

    async def _start_worker(self) -> None:
        assert self.native is not None and self.engine is not None
        self.set_state("loading", "loading the engine and warming up")
        worker = WorkerClient(self.native, self.engine.path, on_exit=self._worker_exited)
        ready = await worker.start(timeout=900)
        if not worker.alive:  # died before it was serving, so its exit did not trigger a restart
            await worker.stop(timeout=5)
            raise WorkerError("worker_exited", "the inference worker exited right after loading the engine",
                              fatal=True)
        self.worker = worker
        if self.telemetry:
            self.telemetry.set_worker_pid(worker.pid)
        used = ready.get("device_free_before_load_bytes", 0) - ready.get("device_free_after_load_bytes", 0)
        log.info("worker ready: load %.1f s, warm-up %.0f ms, CUDA graphs %s, ~%.2f GiB in use after load",
                 ready.get("load_ms", 0) / 1e3, ready.get("warmup_ms", 0), ready.get("cuda_graph"), used / 2**30)

    def _building(self, message: str) -> None:
        if not self._stopping:  # a late progress report must not overwrite "draining" / "stopped"
            self.set_state("building_engine", message)

    def _worker_exited(self, code: int | None) -> None:
        # Exits during startup or a restart are reported by WorkerClient.start(); only a worker that was
        # serving triggers the restart loop here.
        if self._stopping or self.state != "ready":
            return
        self._tasks.append(asyncio.create_task(self._restart_worker(code), name="restart"))

    async def _restart_worker(self, code: int | None) -> None:
        while True:
            now = time.monotonic()
            while self._restarts and now - self._restarts[0] > limits.RESTART_WINDOW_S:
                self._restarts.popleft()
            if len(self._restarts) >= limits.MAX_WORKER_RESTARTS:
                message = (f"the inference worker failed {len(self._restarts) + 1} times within "
                           f"{limits.RESTART_WINDOW_S / 60:.0f} minutes (last exit code {code}); not restarting")
                self.set_state("failed", message)
                if self.scheduler:
                    self.scheduler.fail_queued(WorkerError("worker_exited", message, fatal=True))
                return
            self._restarts.append(now)
            delay = min(30.0, 2.0 ** (len(self._restarts) - 1))
            self.set_state("restarting", f"worker exited with code {code}; restarting in {delay:.0f} s")
            await asyncio.sleep(delay)
            try:
                await self._start_worker()
            except Exception as e:  # noqa: BLE001 - e.g. OSError for a missing binary: counts as a failed restart
                log.error("worker restart failed: %s", e)
                code = None
                continue
            self.set_state("draining" if self._stopping else "ready", "worker restarted")
            if self.scheduler:
                self.scheduler.wake()
            return

    async def _refresh_clocks(self) -> None:
        """Power modes can change at runtime; keep MFU/MBU denominators current."""
        while True:
            await asyncio.sleep(30)
            if self.platform is None or self.ctx is None:
                continue
            with contextlib.suppress(Exception):
                self.platform = await asyncio.to_thread(refresh_clocks, self.platform)
                self.ctx.platform = self.platform
                self.prometheus.set_peaks(self.platform)

    def _queue_changed(self) -> None:
        if self.scheduler:
            self.prometheus.set_queue(self.scheduler.running, self.scheduler.waiting, self.scheduler.kv_usage())

    async def drain(self) -> None:
        """Stop admitting requests and building, let in-flight requests finish (bounded), cancel what is left."""
        self._stopping = True
        self._build_cancel.set()
        if self.state == "ready":
            self.set_state("draining", "finishing in-flight requests")
        if self.scheduler:
            await self.scheduler.drain(limits.DRAIN_TIMEOUT_S)

    async def stop(self) -> None:
        """Drain, then stop the background tasks, the worker and telemetry, and flush the records to disk."""
        try:
            await self.drain()
        except Exception:  # e.g. a dispatcher that died: the worker must still be stopped and the records flushed
            log.exception("draining failed")
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)  # their cleanup (a cancelled build) first
        if self.worker:
            await self.worker.stop()
        if self._cache:
            self._cache.release()
        if self.telemetry:
            await asyncio.to_thread(self.telemetry.stop)
        if self.output:
            try:  # queued behind every record committed so far
                await asyncio.get_running_loop().run_in_executor(self._records, self.output.close)
            except OSError as e:
                log.error("could not flush %s: %s", self.output.root / "results.jsonl", e)
        if self.state != "failed":  # keep the reason (and the exit status) of a failed service
            self.set_state("stopped")

    # ------------------------------------------------------------------ requests

    def new_attempt_id(self) -> str:
        return "att_" + secrets.token_hex(12)

    async def count_tokens(self, item: dict) -> int:
        if not self.ready or self.worker is None:
            raise WorkerError("worker_unavailable", f"the model is not ready ({self.detail})")
        return (await self.worker.count([item]))[0]

    def make_record(self, job: Job, *, status: str, http_status: int, done: dict | None = None,
                    result: dict | None = None, error: dict | None = None) -> dict:
        assert self.ctx is not None
        if error is None and http_status == 499:  # say why, in the record and its results.jsonl line
            error = {"message": "the client disconnected; generation was cancelled", "type": "cancelled",
                     "param": None, "code": "client_disconnected"}
        return build_record(job, self.ctx, status=status, http_status=http_status,
                            t_finished_ns=time.monotonic_ns(), done=done, result=result, error=error)

    def commit(self, job: Job, record: dict, response_body: dict | None = None) -> None:
        """Update metrics, log one line per request (no prompt or output text), and write the record."""
        status, http_status, error = record["status"], record["http_status"], record.get("error")
        self.prometheus.observe(record)
        m = record["metrics"]
        request_log.info(json.dumps({
            "event": "request", "custom_id": job.custom_id, "attempt_id": job.attempt_id, "status": status,
            "http_status": http_status, "finish_reason": record["output"]["finish_reason"],
            "prompt_tokens": record["usage"]["prompt_tokens"],
            "completion_tokens": record["usage"]["completion_tokens"], "ttft_ms": m.get("ttft_ms"),
            "tpot_ms": m.get("tpot_ms"), "e2e_ms": m.get("e2e_ms"), "energy_j": m.get("energy_j"),
            "error": (error or {}).get("code") if error else None,
        }))
        if self.output is not None:
            body = response_body if response_body is not None else ({"error": error} if error else None)
            line = {"id": job.attempt_id, "custom_id": job.custom_id,
                    "response": {"status_code": http_status, "request_id": job.attempt_id, "body": body},
                    "error": None}
            asyncio.get_running_loop().run_in_executor(self._records, self._persist, record, line)

    def _persist(self, record: dict, line: dict) -> None:
        try:
            assert self.output is not None
            self.output.write_record(record)
            self.output.append_result(line)
        except Exception:
            log.exception("could not write the record for %s", record.get("custom_id"))

    def read_record(self, custom_id: str) -> dict | None:
        return self.output.read_record(custom_id) if self.output else None

    def models_entry(self) -> dict:
        entry = {"id": self.model_name, "object": "model", "created": int(self.started_at),
                 "owned_by": "edgeserve", "ready": self.ready}
        if self.engine and self.model:
            p = self.engine.profile
            # max_model_len: the largest prompt + max_tokens accepted (the last generated token needs no KV slot,
            # the first prompt position does: prompt + max_tokens + 1 <= KV capacity)
            entry.update({"max_model_len": p.kv_capacity - 1, "max_input_tokens": p.max_input_len,
                          "max_new_tokens": p.max_new_tokens, "max_batch_size": p.max_batch_size,
                          "quantization": self.model.quantization, "engine_fingerprint": self.engine.fingerprint})
        return entry


def default_max_new_tokens(settings: Settings, model: ModelInfo) -> int:
    """--max-new-tokens, else max_new_tokens from the model's generation_config.json, else DEFAULT_MAX_NEW_TOKENS."""
    if settings.max_new_tokens:
        return settings.max_new_tokens
    value = model.generation_defaults.get("max_new_tokens")
    if value is None or value == 0:
        return limits.DEFAULT_MAX_NEW_TOKENS
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StartupError(f"{model.path / 'generation_config.json'}: max_new_tokens must be a positive whole number, "
                           f"not {value!r}; fix it or pass --max-new-tokens")
    return value


def check_runtime(native: NativeBuild, gpu: dict) -> None:
    """The TensorRT library the worker loaded must be the one the GPU runtime was built against: after a TensorRT
    upgrade, engines built by the new library would never match the runtime again."""
    built = str(native.info.get("tensorrt") or "")
    loaded = str(gpu.get("tensorrt_runtime") or "")
    if built and loaded and built.split(".")[:3] != loaded.split(".")[:3]:
        raise StartupError(f"the GPU runtime was built for TensorRT {built}, but this machine now has TensorRT "
                           f"{loaded}: run edgeserve setup")


def check_compatibility(model: ModelInfo, gpu: dict) -> None:
    runtime = str(gpu.get("edgellm", ""))
    exported = model.edgellm_version or ""
    if exported and runtime and exported.split(".")[:2] != runtime.split(".")[:2]:
        raise StartupError(
            f"{model.name} was exported with TensorRT Edge-LLM {exported}, but this server runs Edge-LLM {runtime}. "
            "Exports only work with the Edge-LLM version that produced them: re-export the model with "
            f"Edge-LLM {runtime} (see README, 'Export a model')."
        )
    if not model.has_chat_template:
        raise StartupError(
            f"{model.name} has no chat_template.jinja. TensorRT Edge-LLM needs the model's chat template to build "
            "and run an engine (its exporter writes it from the checkpoint's tokenizer_config.json): re-export "
            "the model, or copy the template into the folder as chat_template.jinja."
        )
    sm = int(gpu.get("sm", 0))
    if "int4" in model.quantization.lower() and sm not in INT4_PLUGIN_SMS:
        raise StartupError(
            f"{model.name} uses Edge-LLM's INT4 GEMM kernels, which do not exist for this GPU (sm_{sm}); "
            "supported: " + ", ".join(f"sm_{s}" for s in sorted(INT4_PLUGIN_SMS)) + ". Use another GPU or a "
            "different export (FP16/FP8)."
        )


def check_memory(model: ModelInfo, profile: BuildProfile, platform: Platform, gpu: dict) -> None:
    """Refuse configurations that clearly cannot fit; warn when it is tight."""
    estimate = estimate_memory(model, profile.max_batch_size, profile.kv_capacity, profile.max_input_len)
    available = available_memory(platform, gpu)
    need = estimate["total"]
    gib = 2**30
    if not available:
        return
    hints = memory_hints(platform.integrated)
    if need > available + 512 * 1024**2:
        raise StartupError(
            f"not enough memory for {model.name}: it needs about {need / gib:.1f} GiB "
            f"(weights {estimate['weights'] / gib:.1f}, KV cache {estimate['kv_cache'] / gib:.2f}, "
            f"activations ~{estimate['activations'] / gib:.2f}) but only {available / gib:.1f} GiB is available. "
            f"To fix: {hints}."
        )
    if need + limits.LOAD_HEADROOM_BYTES > available:
        log.warning("memory is tight: the model needs about %.1f GiB and %.1f GiB is available; if loading fails, %s",
                    need / gib, available / gib, hints)


def memory_hints(integrated: bool) -> str:
    """How to make a model fit: on Jetson the GPU shares the system memory with everything else."""
    free = ("stop the desktop (sudo systemctl isolate multi-user.target) and other programs" if integrated
            else "close other programs that use the GPU (nvidia-smi lists them)")
    return (f"{free}, lower --max-input-len / --max-new-tokens, use --max-batch-size 1, or serve a smaller or more "
            "strongly quantized model")


# generation_config.json options that change outputs but that TensorRT Edge-LLM's sampler does not have, with the
# value at which they do nothing (edgeserve applies do_sample, temperature, top_p, top_k, max_new_tokens and the
# EOS token IDs).
_UNSUPPORTED_GENERATION = {
    "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "encoder_repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0, "num_beams": 1, "num_beam_groups": 1, "diversity_penalty": 0.0,
    "min_new_tokens": 0, "min_length": 0, "min_p": 0.0, "typical_p": 1.0, "epsilon_cutoff": 0.0, "eta_cutoff": 0.0,
    "bad_words_ids": None, "suppress_tokens": None, "begin_suppress_tokens": None, "forced_bos_token_id": None,
    "forced_eos_token_id": None, "sequence_bias": None,
}


def generation_notes(model: ModelInfo) -> list[str]:
    """Startup warnings about the model's generation_config.json: missing, or setting options edgeserve ignores."""
    if not (model.path / "generation_config.json").is_file():
        return [f"{model.path} has no generation_config.json: requests that set no sampling parameters use greedy "
                f"decoding and up to {limits.DEFAULT_MAX_NEW_TOKENS} new tokens (copy the checkpoint's "
                "generation_config.json into the folder to use its defaults)"]
    ignored = [f"{key}={value!r}" for key, neutral in _UNSUPPORTED_GENERATION.items()
               if (value := model.generation_defaults.get(key)) is not None and value != neutral
               and value not in ([], {})]
    if not ignored:
        return []
    return [f"{model.path / 'generation_config.json'} sets {', '.join(ignored)}: TensorRT Edge-LLM does not support "
            f"{'this option' if len(ignored) == 1 else 'these options'}, so edgeserve ignores "
            f"{'it' if len(ignored) == 1 else 'them'} and outputs can differ from Hugging Face transformers"]


def available_memory(platform: Platform, gpu: dict) -> int:
    """Bytes a model can use now: the system's available memory on Jetson (shared with the GPU), else the GPU's
    free memory."""
    return _mem_available() if platform.integrated else int(gpu.get("free_memory_bytes", 0))


def _mem_available() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 0


__all__ = ["Service", "StartupError", "available_memory", "check_compatibility", "check_memory", "check_runtime",
           "default_max_new_tokens", "generation_notes", "memory_hints"]
