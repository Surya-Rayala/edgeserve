"""Per-request records (every token step + whole-request summary) and Prometheus metrics.

Definitions (also in docs/METRICS.md). Timestamps share one monotonic clock across the front-end
and the worker; T_i is when output token i became available (the per-token callback fires right
after each step's stream synchronize), t_sch when the worker started the request.

    queue_ms     = t_sch - t_queued
    prefill_ms   = T_0 - t_sch                 (template + tokenization + prefill + first sample)
    ttft_ms      = T_0 - t_arrival
    decode_ms    = T_{n-1} - T_0
    e2e_ms       = T_{n-1} - t_arrival         (server-side; network time excluded)
    tpot_ms      = decode_ms / (n - 1)         (one value per request; MLPerf / vLLM TPOT)
    itl_ms[i]    = T_i - T_{i-1}               (distribution of gaps; vLLM ITL)
    MFU = FLOPs / (time x peak FLOP/s), MBU = bytes / (time x peak bandwidth), per step on GPU time

n counts every sampled token including a terminating end-of-sequence token. FLOPs and bytes
come from the model's own tensor shapes (edgeserve.model); with batching, a step's weight reads
are shared by the sequences in it, so each sequence is charged weights / active + its own KV.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.core import CounterMetricFamily

if TYPE_CHECKING:
    from edgeserve.engine import EngineInfo
    from edgeserve.hardware import Platform
    from edgeserve.model import ModelInfo
    from edgeserve.scheduler import Job
    from edgeserve.telemetry import Telemetry

RECORD_SCHEMA = "1.1"  # 1.1: "client", and the status "rejected" (a request refused before it was queued)


@dataclass
class RecordContext:
    """What every record needs to know about the service that produced it."""

    model: ModelInfo
    engine: EngineInfo
    platform: Platform
    telemetry: Telemetry | None
    served_name: str

    @property
    def system_fingerprint(self) -> str:
        return "fp_" + self.engine.key[:12]


def percentile(values: list[float], q: float) -> float:
    """Linear interpolation between closest ranks (numpy's default method)."""
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def distribution(values: list[float]) -> dict | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    stdev = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    return {
        "mean": round(mean, 3), "p50": round(percentile(values, 0.5), 3), "p90": round(percentile(values, 0.9), 3),
        "p99": round(percentile(values, 0.99), 3), "min": round(min(values), 3), "max": round(max(values), 3),
        "stdev": round(stdev, 3),
    }


def _ms(ns: int | float | None) -> float | None:
    return None if ns is None else round(ns / 1e6, 3)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or not denominator:
        return None
    return numerator / denominator


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def _mapping(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _amount(value: object) -> float | None:
    """``value`` as a finite number >= 0, else None (a counter cannot go down; NaN would poison a histogram sum)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) and number >= 0 else None


def build_record(
    job: Job,
    ctx: RecordContext,
    *,
    status: str,
    http_status: int,
    t_finished_ns: int,
    done: dict | None = None,
    result: dict | None = None,
    error: dict | None = None,
) -> dict:
    """The full structured record of one request attempt (schema "edgeserve.record" v1)."""
    model, platform = ctx.model, ctx.platform
    wall = lambda mono_ns: None if mono_ns is None else job.t_arrival_unix_ns + (mono_ns - job.t_arrival_ns)  # noqa: E731

    token_ids: list[int] = list(result.get("token_ids", [])) if result else []
    pieces: list[str] = list(result.get("pieces", [])) if result else []
    n = len(token_ids)
    prompt_tokens = int(result["prompt_tokens"]) if result and result.get("prompt_tokens", -1) >= 0 else job.prompt_tokens
    step_log = (done or {}).get("steps", {})
    step_t = list(step_log.get("t_ns", []))[:n]
    t_sch = done.get("t_start_ns") if done else None

    steps = _steps(job, ctx, step_log, step_t, token_ids, pieces, prompt_tokens, t_sch) if n and step_t else None
    metrics = _request_metrics(job, ctx, steps, step_t, prompt_tokens, n, t_sch, t_finished_ns)

    record = {
        "object": "edgeserve.record",
        "schema_version": RECORD_SCHEMA,
        "custom_id": job.custom_id,
        "attempt_id": job.attempt_id,
        "status": status,
        "http_status": http_status,
        "error": error,
        "created": job.t_arrival_unix_ns // 1_000_000_000,
        "endpoint": job.endpoint,
        "metadata": job.metadata,
        "client": job.client,
        "system_fingerprint": ctx.system_fingerprint,
        "model": {
            "name": ctx.served_name,
            "path": str(model.path),
            "engine_fingerprint": ctx.engine.fingerprint,
            "quantization": model.quantization,
            "kv_cache_dtype": model.kv_cache_dtype,
            "vocab_size": model.vocab_size,
            "output_vocab_size": model.output_vocab_size,
            "linear_params": model.linear_params,
            "streamed_weight_bytes": model.streamed_weight_bytes,
            "cost_model": model.cost_model,
            "max_input_tokens": ctx.engine.profile.max_input_len,
            "kv_capacity_tokens": ctx.engine.profile.kv_capacity,
            "max_batch_size": ctx.engine.profile.max_batch_size,
        },
        "system": platform.as_dict(),
        "request": {**job.request_echo, "params": job.params_echo},
        "output": {
            "text": result.get("text", "") if result else "",
            "finish_reason": result.get("finish_reason") if result else None,
            "token_ids": token_ids,
        },
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n, "total_tokens": prompt_tokens + n},
        "timestamps_ns": {
            "arrival": job.t_arrival_unix_ns,
            "queued": wall(job.t_queued_ns or None),
            "scheduled": wall(t_sch),
            "first_token": wall(step_t[0]) if step_t else None,
            "last_token": wall(step_t[-1]) if step_t else None,
            "finished": wall(t_finished_ns),
        },
        "metrics": metrics,
    }
    if steps is not None:
        record["steps"] = steps
    return record


def _steps(
    job: Job,
    ctx: RecordContext,
    step_log: dict,
    step_t: list[int],
    token_ids: list[int],
    pieces: list[str],
    prompt_tokens: int,
    t_sch: int | None,
) -> dict:
    """Columnar per-step arrays; entry i is the forward pass that produced token i (i = 0 is prefill)."""
    model, platform, telemetry = ctx.model, ctx.platform, ctx.telemetry
    n = len(step_t)
    gpu_ms = list(step_log.get("gpu_ms", []))[:n] + [None] * max(0, n - len(step_log.get("gpu_ms", [])))
    free_bytes = list(step_log.get("free_bytes", []))[:n]
    active = list(step_log.get("active", []))[:n] or [1] * n
    batch_ctx = job.batch_contexts  # per step: prompt+i of every sequence active in the batch (for MBU/MFU)
    capacity = ctx.engine.profile.kv_capacity

    t_ms, itl_ms, context_len, flops, bytes_, mfu, mbu, kv_usage = [], [], [], [], [], [], [], []
    for i in range(n):
        t_ms.append(_ms(step_t[i] - job.t_arrival_ns))
        itl_ms.append(None if i == 0 else _ms(step_t[i] - step_t[i - 1]))
        context = prompt_tokens + i
        context_len.append(context)
        share = max(1, active[i] if i < len(active) else 1)
        f = model.prefill_flops(prompt_tokens) if i == 0 else model.decode_flops(context)
        b = model.prefill_bytes(prompt_tokens) if i == 0 else model.decode_bytes(context)
        # This sequence's share: its own FLOPs; weight reads split across the sequences in the step.
        weights = model.streamed_weight_bytes
        flops.append(f)
        bytes_.append(None if b is None else int(b - weights + weights / share))
        # Utilisation of the whole step (all sequences), on GPU time when the runtime reported it.
        if gpu_ms[i]:
            step_s = gpu_ms[i] / 1e3
        elif i > 0:
            step_s = (step_t[i] - step_t[i - 1]) / 1e9
        elif t_sch is not None:
            step_s = (step_t[0] - t_sch) / 1e9
        else:
            step_s = None
        batch_f, batch_b = _batch_step_cost(model, i, batch_ctx, f, b)
        mfu.append(_round(_ratio(batch_f, (step_s or 0) * (platform.peak_flops_per_s or 0)), 5))
        mbu.append(_round(_ratio(batch_b, (step_s or 0) * (platform.peak_mem_bw_bytes_per_s or 0)), 5))
        kv_usage.append(round(context / capacity, 5) if capacity else None)

    power_w: list[float | None] = [None] * n
    energy_mj: list[float | None] = [None] * n
    if telemetry is not None and telemetry.energy_source and t_sch is not None:
        boundaries = [t_sch] + step_t
        energies = telemetry.energy_series_j(boundaries)
        for i, energy in enumerate(energies):
            if energy is None:
                continue
            share = max(1, active[i] if i < len(active) else 1)
            duration = (boundaries[i + 1] - boundaries[i]) / 1e9
            energy_mj[i] = round(energy * 1000 / share, 3)
            power_w[i] = round(energy / duration, 3) if duration > 0 else None

    return {
        "count": n,
        "token_id": token_ids[:n],
        "token_text": pieces[:n],
        "t_ms": t_ms,
        "itl_ms": itl_ms,
        "gpu_ms": [None if g is None else round(g, 3) for g in gpu_ms],
        "context_len": context_len,
        "flops": flops,
        "bytes": bytes_,
        "mfu": mfu,
        "mbu": mbu,
        "power_w": power_w,
        "energy_mj": energy_mj,
        "kv_usage": kv_usage,
        "device_free_bytes": free_bytes,
        "active_sequences": active,
    }


def _batch_step_cost(model: ModelInfo, step: int, batch_ctx: list[list[int]] | None, own_flops, own_bytes):
    """FLOPs and bytes of a whole batched step (weights read once, every sequence's KV)."""
    if not batch_ctx or step >= len(batch_ctx) or len(batch_ctx[step]) <= 1:
        return own_flops, own_bytes
    if any(c < 0 for c in batch_ctx[0]):  # the worker reports -1 for a prompt length it does not know
        return None, None
    contexts = batch_ctx[step]
    if step == 0:
        per_seq = [(model.prefill_flops(c), model.prefill_bytes(c)) for c in contexts]
    else:
        per_seq = [(model.decode_flops(c), model.decode_bytes(c)) for c in contexts]
    if any(f is None or b is None for f, b in per_seq):
        return None, None
    weights = model.streamed_weight_bytes
    return sum(f for f, _ in per_seq), sum(b - weights for _, b in per_seq) + weights


def _request_metrics(
    job: Job,
    ctx: RecordContext,
    steps: dict | None,
    step_t: list[int],
    prompt_tokens: int,
    n: int,
    t_sch: int | None,
    t_finished_ns: int,
) -> dict:
    platform, telemetry = ctx.platform, ctx.telemetry
    t0 = step_t[0] if step_t else None
    t_last = step_t[-1] if step_t else None
    queue_ns = (t_sch - job.t_queued_ns) if t_sch is not None and job.t_queued_ns else None
    prefill_ns = (t0 - t_sch) if t0 is not None and t_sch is not None else None
    decode_ns = (t_last - t0) if t0 is not None else None  # 0 for a single token: the decode phase is empty
    gaps = [(b - a) / 1e6 for a, b in zip(step_t, step_t[1:])]  # unrounded, so the mean ITL equals the TPOT
    gpu = (steps or {}).get("gpu_ms", [])

    def phase_sum(key: str, decode: bool) -> float | None:
        values = (steps or {}).get(key, [])
        chosen = values[1:] if decode else values[:1]
        if not values or any(v is None for v in values[:1] + chosen):
            return None
        return sum(chosen)

    prefill_flops, decode_flops = phase_sum("flops", False), phase_sum("flops", True)
    prefill_bytes, decode_bytes = phase_sum("bytes", False), phase_sum("bytes", True)
    prefill_s = (gpu[0] / 1e3) if gpu and gpu[0] else (prefill_ns / 1e9 if prefill_ns else None)
    decode_gpu = gpu[1:]
    decode_gpu_ms = sum(decode_gpu, 0.0) if gpu and all(g is not None for g in decode_gpu) else None
    decode_s = (sum(decode_gpu) / 1e3) if decode_gpu and all(decode_gpu) else (
        decode_ns / 1e9 if decode_ns else None)
    mfu_values = [v for v in (steps or {}).get("mfu", [])[1:] if v is not None]
    mbu_values = [v for v in (steps or {}).get("mbu", [])[1:] if v is not None]

    energy = energy_prefill = None
    if telemetry is not None and telemetry.energy_source and t_sch is not None and t_last is not None:
        energy_steps = (steps or {}).get("energy_mj", [])
        if energy_steps and all(e is not None for e in energy_steps):
            energy = sum(energy_steps) / 1000
            energy_prefill = energy_steps[0] / 1000
    inference_s = (t_last - t_sch) / 1e9 if t_last is not None and t_sch is not None else None

    window_start = t_sch if t_sch is not None else job.t_arrival_ns
    memory_min = telemetry.min_mem_available(window_start, t_finished_ns) if telemetry else None
    rss_peak = telemetry.peak_worker_rss(window_start, t_finished_ns) if telemetry else None

    return {
        "preprocess_ms": _ms(job.t_queued_ns - job.t_arrival_ns) if job.t_queued_ns else None,
        "queue_ms": _ms(queue_ns),
        "prefill_ms": _ms(prefill_ns),
        "ttft_ms": _ms(t0 - job.t_arrival_ns) if t0 is not None else None,
        "decode_ms": _ms(decode_ns),
        "inference_ms": _ms(t_last - t_sch) if t_last is not None and t_sch is not None else None,
        "e2e_ms": _ms(t_last - job.t_arrival_ns) if t_last is not None else None,
        "tpot_ms": _round(decode_ns / 1e6 / (n - 1), 3) if decode_ns is not None and n >= 2 else None,
        "itl_ms": distribution(gaps),
        "prefill_tokens_per_s": _round(_ratio(prompt_tokens, prefill_ns / 1e9 if prefill_ns else None), 2),
        "decode_tokens_per_s": _round(_ratio(n - 1, decode_ns / 1e9 if decode_ns else None), 2),
        "e2e_tokens_per_s": _round(_ratio(n, (t_last - job.t_arrival_ns) / 1e9 if t_last else None), 2),
        "prefill_gpu_ms": _round(gpu[0], 3) if gpu and gpu[0] is not None else None,
        "decode_gpu_ms": _round(decode_gpu_ms, 3),
        "prefill_flops": prefill_flops,
        "decode_flops": decode_flops,
        "prefill_flops_per_s": _round(_ratio(prefill_flops, prefill_s), 0),
        "decode_flops_per_s": _round(_ratio(decode_flops, decode_s), 0),
        "prefill_mfu": (steps or {}).get("mfu", [None])[0] if steps else None,
        "decode_mfu": _round(sum(mfu_values) / len(mfu_values), 5) if mfu_values else None,
        "prefill_bytes": prefill_bytes,
        "decode_bytes": decode_bytes,
        "decode_bytes_per_s": _round(_ratio(decode_bytes, decode_s), 0),
        "decode_mbu": _round(sum(mbu_values) / len(mbu_values), 5) if mbu_values else None,
        "energy_j": _round(energy, 4),
        "prefill_energy_j": _round(energy_prefill, 4),
        "decode_energy_j": _round(energy - energy_prefill, 4) if energy is not None and energy_prefill is not None else None,
        "avg_power_w": _round(_ratio(energy, inference_s), 3),
        "energy_per_output_token_mj": _round(_ratio(energy * 1000 if energy is not None else None, n), 3),
        "output_tokens_per_j": _round(_ratio(n, energy), 4),
        "energy_source": telemetry.energy_source if telemetry else None,
        "min_mem_available_bytes": memory_min,
        "peak_worker_rss_bytes": rss_peak,
        "kv_capacity_tokens": ctx.engine.profile.kv_capacity,
        "kv_peak_used_tokens": prompt_tokens + n if n else prompt_tokens,
        "kv_peak_usage": _round(_ratio(prompt_tokens + n, ctx.engine.profile.kv_capacity), 5),
        "batch_size": job.batch_size,
        "peak_flops_per_s": platform.peak_flops_per_s,
        "peak_mem_bw_bytes_per_s": platform.peak_mem_bw_bytes_per_s,
        "timing_source": "engine_callback" if step_t else None,
    }


def response_metrics(record: dict) -> dict:
    """The compact metrics object returned in API responses (the full set is in the record)."""
    keys = ("queue_ms", "prefill_ms", "ttft_ms", "decode_ms", "e2e_ms", "tpot_ms", "prefill_tokens_per_s",
            "decode_tokens_per_s", "prefill_mfu", "decode_mbu", "energy_j", "energy_per_output_token_mj",
            "avg_power_w", "batch_size")
    metrics = record.get("metrics", {})
    summary = {k: metrics.get(k) for k in keys}
    itl = metrics.get("itl_ms") or {}
    summary["itl_p50_ms"], summary["itl_p99_ms"] = itl.get("p50"), itl.get("p99")
    return summary


# --------------------------------------------------------------------------- Prometheus

_LATENCY_BUCKETS = (0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0,
                    240.0, 480.0, 960.0, 1920.0, 7680.0)
_TTFT_BUCKETS = (0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
                 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0)
# vLLM's ITL buckets plus finer resolution in the 30-60 ms range typical of edge decoding.
_ITL_BUCKETS = (0.01, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75,
                1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0)
_TOKEN_BUCKETS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)
STATES = ("starting", "building_engine", "loading", "ready", "restarting", "draining", "failed", "stopped")
REJECTION_REASONS = ("not_ready", "low_memory", "duplicate_request_id", "queue_full", "shutting_down",
                     "context_length", "client_limit", "rate_limited")


class PrometheusMetrics:
    """vLLM-compatible metric names with an ``edgeserve_`` prefix (underscore: Prometheus reserves colons)."""

    def __init__(self, model_name: str, telemetry: Telemetry | None = None) -> None:
        self.model_name = model_name
        self.registry = CollectorRegistry()
        r, m = self.registry, ["model_name"]

        def hist(name: str, doc: str, buckets: tuple) -> Histogram:
            return Histogram(f"edgeserve_{name}", doc, m, buckets=buckets, registry=r)

        self.ttft = hist("time_to_first_token_seconds", "Time to first token (arrival to first token)", _TTFT_BUCKETS)
        self.itl = hist("inter_token_latency_seconds", "Gap between consecutive output tokens", _ITL_BUCKETS)
        self.tpot = hist("request_time_per_output_token_seconds", "Per-request mean time per output token",
                         _ITL_BUCKETS)
        self.e2e = hist("e2e_request_latency_seconds", "Arrival to last token", _LATENCY_BUCKETS)
        self.queue = hist("request_queue_time_seconds", "Time waiting for the GPU", _LATENCY_BUCKETS)
        self.prefill = hist("request_prefill_time_seconds", "Start to first token", _LATENCY_BUCKETS)
        self.decode = hist("request_decode_time_seconds", "First to last token", _LATENCY_BUCKETS)
        self.inference = hist("request_inference_time_seconds", "Start to last token", _LATENCY_BUCKETS)
        self.prompt_len = hist("request_prompt_tokens", "Prompt tokens per request", _TOKEN_BUCKETS)
        self.gen_len = hist("request_generation_tokens", "Generated tokens per request", _TOKEN_BUCKETS)
        self.energy = hist("request_energy_joules", "Energy per request", (0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200,
                                                                          500, 1000, 2000))
        self.prompt_tokens = Counter("edgeserve_prompt_tokens", "Prefilled prompt tokens", m, registry=r)
        self.generation_tokens = Counter("edgeserve_generation_tokens", "Generated tokens", m, registry=r)
        self.success = Counter("edgeserve_request_success", "Finished requests", m + ["finished_reason"], registry=r)
        self.errors = Counter("edgeserve_request_errors", "Failed requests", m + ["error_type"], registry=r)
        self.rejections = Counter("edgeserve_admission_rejections", "Requests refused before generation",
                                  m + ["reason"], registry=r)
        self.flops = Counter("edgeserve_estimated_flops", "Estimated floating-point operations", m, registry=r)
        self.read_bytes = Counter("edgeserve_estimated_read_bytes", "Estimated DRAM bytes moved", m, registry=r)
        self.running = Gauge("edgeserve_num_requests_running", "Requests generating", m, registry=r)
        self.waiting = Gauge("edgeserve_num_requests_waiting", "Requests queued", m, registry=r)
        self.kv_usage = Gauge("edgeserve_kv_cache_usage_perc",
                              "KV cache fraction reserved by running requests (prompt + max_tokens, 0-1)",
                              m, registry=r)
        self.mem_available = Gauge("edgeserve_system_memory_available_bytes", "MemAvailable", m, registry=r)
        self.peak_flops = Gauge("edgeserve_peak_flops_per_second", "MFU denominator (current power mode)", m,
                                registry=r)
        self.peak_bw = Gauge("edgeserve_peak_memory_bandwidth_bytes_per_second", "MBU denominator", m, registry=r)
        self.state = Gauge("edgeserve_server_state", "1 for the current server state", m + ["state"], registry=r)
        self.info = Gauge("edgeserve_model_info", "Served model and engine",
                          m + ["engine_fingerprint", "quantization", "tensorrt_version", "edgellm_version",
                               "max_input_tokens", "kv_capacity_tokens", "max_batch_size"], registry=r)
        # Every documented series exists from the start (at 0), so rate() and increase() also see its first event.
        for counter in (self.prompt_tokens, self.generation_tokens, self.flops, self.read_bytes):
            counter.labels(model_name)
        for finish in ("stop", "length"):
            self.success.labels(model_name, finish)
        for status in ("failed", "cancelled"):
            self.errors.labels(model_name, status)
        for reason in REJECTION_REASONS:
            self.rejections.labels(model_name, reason)
        self._telemetry = telemetry
        if telemetry is not None:
            r.register(_EnergyCollector(self))
        self.set_state("starting")

    def set_telemetry(self, telemetry: Telemetry) -> None:
        if self._telemetry is None:
            self._telemetry = telemetry
            self.registry.register(_EnergyCollector(self))

    def set_state(self, state: str) -> None:
        for name in STATES:
            self.state.labels(self.model_name, name).set(1 if name == state else 0)

    def set_model_info(self, ctx: RecordContext) -> None:
        p = ctx.engine.profile
        self.info.labels(self.model_name, ctx.engine.fingerprint, ctx.model.quantization, ctx.platform.tensorrt,
                         ctx.platform.edgellm, str(p.max_input_len), str(p.kv_capacity), str(p.max_batch_size)).set(1)
        self.set_peaks(ctx.platform)

    def set_peaks(self, platform: Platform) -> None:
        if platform.peak_flops_per_s:
            self.peak_flops.labels(self.model_name).set(platform.peak_flops_per_s)
        if platform.peak_mem_bw_bytes_per_s:
            self.peak_bw.labels(self.model_name).set(platform.peak_mem_bw_bytes_per_s)

    def set_queue(self, running: int, waiting: int, kv_usage: float) -> None:
        self.running.labels(self.model_name).set(running)
        self.waiting.labels(self.model_name).set(waiting)
        self.kv_usage.labels(self.model_name).set(kv_usage)

    def set_memory(self, available: int | None) -> None:
        if available is not None:
            self.mem_available.labels(self.model_name).set(available)

    def rejected(self, reason: str) -> None:
        self.rejections.labels(self.model_name, reason).inc()

    def observe(self, record: dict) -> None:
        """Export one finished attempt. Never raises: values that are missing or not usable numbers are skipped."""
        name = self.model_name
        record = _mapping(record)
        m, usage, steps = _mapping(record.get("metrics")), _mapping(record.get("usage")), _mapping(record.get("steps"))
        completed = record.get("status") == "completed"
        if completed:
            self.success.labels(name, str(_mapping(record.get("output")).get("finish_reason") or "stop")).inc()
        else:
            self.errors.labels(name, str(record.get("status") or "failed")).inc()
        if not (completed or steps):
            return
        # GPU work counts whatever the outcome, as in vLLM: these counters feed throughput, power and MBU queries.
        prompt, generated = _amount(usage.get("prompt_tokens")), _amount(usage.get("completion_tokens"))
        flops = sum(v for v in (_amount(m.get("prefill_flops")), _amount(m.get("decode_flops"))) if v is not None)
        moved = sum(v for v in (_amount(m.get("prefill_bytes")), _amount(m.get("decode_bytes"))) if v is not None)
        for counter, value in ((self.prompt_tokens, prompt), (self.generation_tokens, generated),
                               (self.flops, flops), (self.read_bytes, moved)):
            if value:
                counter.labels(name).inc(value)
        if not completed:
            return
        for hist, value in ((self.prompt_len, prompt), (self.gen_len, generated),
                            (self.energy, _amount(m.get("energy_j")))):
            if value is not None:
                hist.labels(name).observe(value)
        for hist, key in ((self.ttft, "ttft_ms"), (self.e2e, "e2e_ms"), (self.queue, "queue_ms"),
                          (self.prefill, "prefill_ms"), (self.decode, "decode_ms"),
                          (self.inference, "inference_ms"), (self.tpot, "tpot_ms")):
            value = _amount(m.get(key))
            if value is not None:
                hist.labels(name).observe(value / 1e3)
        gaps = steps.get("itl_ms")
        itl = self.itl.labels(name)
        for gap in gaps if isinstance(gaps, list) else ():
            value = _amount(gap)
            if value is not None:
                itl.observe(value / 1e3)

    def render(self) -> bytes:
        if self._telemetry is not None:
            sample = self._telemetry.memory()
            if sample is not None:
                self.set_memory(sample.mem_available_bytes)
        return generate_latest(self.registry)


class _EnergyCollector:
    """Exports the telemetry's cumulative energy as a counter (rate() of it is power in watts)."""

    def __init__(self, owner: PrometheusMetrics) -> None:
        self._owner = owner

    def collect(self):
        telemetry = self._owner._telemetry
        family = CounterMetricFamily("edgeserve_energy_joules", "Energy measured since start", labels=["model_name", "rail"])
        if telemetry is not None and telemetry.energy_source:
            family.add_metric([self._owner.model_name, telemetry.energy_source], telemetry.energy_total_j())
        yield family

    def describe(self):
        return []


__all__ = ["PrometheusMetrics", "RecordContext", "build_record", "distribution", "percentile", "response_metrics"]
