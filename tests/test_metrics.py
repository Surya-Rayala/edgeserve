"""Records and Prometheus export (edgeserve.metrics) checked number by number against docs/METRICS.md."""

from __future__ import annotations

import dataclasses
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from prometheus_client.parser import text_string_to_metric_families

from conftest import make_tiny_model
from edgeserve.metrics import (
    PrometheusMetrics,
    RecordContext,
    build_record,
    distribution,
    percentile,
    response_metrics,
)
from edgeserve.model import ModelInfo, load_model
from edgeserve.scheduler import Job
from edgeserve.telemetry import MemorySample

# One monotonic timeline (ns): arrival, queued, worker start (t_sch), tokens T_0..T_3, record built.
T_ARRIVAL = 1_000_000_000
T_QUEUED = 1_002_000_000
T_SCH = 1_005_000_000
T_STEPS = [1_025_000_000, 1_035_000_000, 1_046_000_000, 1_058_000_000]  # gaps 10, 11, 12 ms
T_FINISHED = 1_060_000_000
UNIX_ARRIVAL = 1_700_000_000_123_456_789
GPU_MS = [18.0, 9.0, 10.0, 11.0]
PROMPT = 10
KV_CAPACITY = 256
PEAK_FLOPS = 1e9
PEAK_BW = 1e8
ENGINE_KEY = "0123456789abcdef" * 4


@dataclass
class StubProfile:
    max_input_len: int = 64
    kv_capacity: int = KV_CAPACITY
    max_batch_size: int = 2


@dataclass
class StubEngine:
    key: str = ENGINE_KEY
    profile: StubProfile = field(default_factory=StubProfile)

    @property
    def fingerprint(self) -> str:
        return self.key[:16]


@dataclass
class StubPlatform:
    peak_flops_per_s: float | None = PEAK_FLOPS
    peak_mem_bw_bytes_per_s: float | None = PEAK_BW
    tensorrt: str = "10.3.0.30"
    edgellm: str = "0.11.0"

    def as_dict(self) -> dict:
        return {"kind": "stub", "peak_flops_per_s": self.peak_flops_per_s}


class StubTelemetry:
    """Constant board power (or fixed per-window energies); records the windows it is asked about."""

    def __init__(self, watts: float = 10.0, source: str | None = "VDD_IN", energies: list | None = None,
                 mem_available: int | None = 3_000_000_000, rss: int | None = 500_000_000,
                 total_j: float = 12.5, memory: MemorySample | None = None) -> None:
        self.watts, self.energy_source, self.energies = watts, source, energies
        self.mem_available, self.rss, self.total_j, self._memory = mem_available, rss, total_j, memory
        self.series_calls: list[list[int]] = []
        self.memory_windows: list[tuple[int, int]] = []

    def energy_series_j(self, boundaries: list[int]) -> list[float | None]:
        self.series_calls.append(list(boundaries))
        if self.energies is not None:
            return list(self.energies)
        return [self.watts * (b - a) / 1e9 for a, b in zip(boundaries, boundaries[1:])]

    def min_mem_available(self, t0: int, t1: int) -> int | None:
        self.memory_windows.append((t0, t1))
        return self.mem_available

    def peak_worker_rss(self, t0: int, t1: int) -> int | None:
        return self.rss

    def memory(self) -> MemorySample | None:
        return self._memory

    def energy_total_j(self) -> float:
        return self.total_j


@pytest.fixture(scope="module")
def model(tmp_path_factory) -> ModelInfo:
    return load_model(make_tiny_model(Path(tmp_path_factory.mktemp("models"))))


def make_ctx(model: ModelInfo, telemetry=None, platform=None, engine=None) -> RecordContext:
    return RecordContext(model=model, engine=engine or StubEngine(), platform=platform or StubPlatform(),
                         telemetry=telemetry, served_name="tiny")


def make_job(**overrides) -> Job:
    values = dict(custom_id="seg-1", attempt_id="att_1", endpoint="/v1/chat/completions",
                  item={"messages": [{"role": "user", "content": "hi"}]},
                  sampling={"temperature": 0.0, "top_p": 1.0, "top_k": 1, "max_new_tokens": 8}, stream=False,
                  prompt_tokens=PROMPT, t_arrival_ns=T_ARRIVAL, t_arrival_unix_ns=UNIX_ARRIVAL,
                  request_echo={"messages": [{"role": "user", "content": "hi"}]}, params_echo={"max_tokens": 8},
                  metadata={"task": "demo"}, t_queued_ns=T_QUEUED)
    values.update(overrides)
    return Job(**values)


def make_done(n: int = 4, gpu_ms=GPU_MS, active=None, t_start: int | None = T_SCH, **steps) -> dict:
    log = {"t_ns": T_STEPS[:n], "gpu_ms": list(gpu_ms)[:n], "free_bytes": [7_000 + i for i in range(n)],
           "active": list(active or [1] * n)[:n], **steps}
    done = {"ev": "done", "id": 1, "t_received_ns": T_SCH - 100_000, "t_end_ns": T_FINISHED - 1_000_000, "steps": log}
    if t_start is not None:
        done["t_start_ns"] = t_start
    return done


def make_result(n: int = 4, prompt_tokens: int = PROMPT, finish: str = "stop") -> dict:
    ids = [100 + i for i in range(n)]
    return {"req": 0, "finish_reason": finish, "prompt_tokens": prompt_tokens, "text": "x" * n,
            "token_ids": ids, "pieces": [f"p{i}" for i in range(n)]}


def record_for(model: ModelInfo, n: int = 4, *, job: Job | None = None, telemetry=None, platform=None,
               status: str = "completed", done=None, result=None, **kwargs) -> dict:
    return build_record(job or make_job(), make_ctx(model, telemetry, platform), status=status, http_status=200,
                        t_finished_ns=T_FINISHED, done=make_done(n) if done is None else done,
                        result=make_result(n) if result is None else result, **kwargs)


# Cost model of docs/METRICS.md, written out independently of edgeserve.model.
def attention_per_pair(m: ModelInfo) -> int:
    return 4 * m.num_layers * m.num_heads * m.head_dim


def decode_flops(m: ModelInfo, context: int) -> int:
    return 2 * m.linear_params + attention_per_pair(m) * context


def prefill_flops(m: ModelInfo, prompt: int) -> int:
    return (2 * (m.linear_params - m.lm_head_params) * prompt + 2 * m.lm_head_params
            + attention_per_pair(m) * prompt * (prompt + 1) // 2)


def pass_bytes(m: ModelInfo, context: int) -> int:
    return m.streamed_weight_bytes + m.kv_bytes_per_token * context + 4 * m.output_vocab_size


def test_tiny_model_cost_model_matches_the_documented_formulas(model) -> None:
    assert (model.linear_params, model.lm_head_params, attention_per_pair(model)) == (38912, 8192, 256)
    assert decode_flops(model, 11) == model.decode_flops(11) == 80640
    assert prefill_flops(model, 10) == model.prefill_flops(10) == 644864
    assert pass_bytes(model, 11) == model.decode_bytes(11) == 36054
    assert pass_bytes(model, 10) == model.prefill_bytes(10) == 35926


# --- one sequence, four tokens -------------------------------------------------------------------


def test_single_sequence_record_has_every_documented_metric(model) -> None:
    telemetry = StubTelemetry(energies=[0.3, 0.12, 0.11, 0.18])  # J per window: 15, 12, 10, 15 W
    record = record_for(model, telemetry=telemetry)
    m = record["metrics"]
    flops = [prefill_flops(model, 10), decode_flops(model, 11), decode_flops(model, 12), decode_flops(model, 13)]
    moved = [pass_bytes(model, 10), pass_bytes(model, 11), pass_bytes(model, 12), pass_bytes(model, 13)]

    assert m["preprocess_ms"] == 2.0  # t_queued - t_arrival
    assert m["queue_ms"] == 3.0  # t_sch - t_queued
    assert m["prefill_ms"] == 20.0  # T_0 - t_sch
    assert m["ttft_ms"] == 25.0  # T_0 - t_arrival
    assert m["decode_ms"] == 33.0  # T_3 - T_0
    assert m["inference_ms"] == 53.0  # T_3 - t_sch
    assert m["e2e_ms"] == 58.0  # T_3 - t_arrival
    assert m["tpot_ms"] == 11.0  # 33 / (4 - 1)
    assert m["itl_ms"] == {"mean": 11.0, "p50": 11.0, "p90": 11.8, "p99": 11.98, "min": 10.0, "max": 12.0,
                           "stdev": round(math.sqrt(2 / 3), 3)}
    assert m["prefill_tokens_per_s"] == 500.0  # 10 / 0.020 s
    assert m["decode_tokens_per_s"] == round(3 / 0.033, 2)
    assert m["e2e_tokens_per_s"] == round(4 / 0.058, 2)
    assert m["prefill_gpu_ms"] == 18.0
    assert m["decode_gpu_ms"] == 30.0
    assert m["prefill_flops"] == flops[0]
    assert m["decode_flops"] == sum(flops[1:]) == 242688
    assert m["prefill_flops_per_s"] == round(flops[0] / 0.018)
    assert m["decode_flops_per_s"] == round(sum(flops[1:]) / 0.030)
    assert m["prefill_mfu"] == round(flops[0] / (0.018 * PEAK_FLOPS), 5) == 0.03583
    step_mfu = [round(f / (g / 1e3 * PEAK_FLOPS), 5) for f, g in zip(flops, GPU_MS)]
    assert m["decode_mfu"] == pytest.approx(sum(step_mfu[1:]) / 3, abs=1e-5)
    assert m["prefill_bytes"] == moved[0]
    assert m["decode_bytes"] == sum(moved[1:])
    assert m["decode_bytes_per_s"] == round(sum(moved[1:]) / 0.030)
    step_mbu = [round(b / (g / 1e3 * PEAK_BW), 5) for b, g in zip(moved, GPU_MS)]
    assert m["decode_mbu"] == pytest.approx(sum(step_mbu[1:]) / 3, abs=1e-5)
    assert m["energy_j"] == 0.71
    assert m["prefill_energy_j"] == 0.3
    assert m["decode_energy_j"] == 0.41
    assert m["avg_power_w"] == round(0.71 / 0.053, 3)  # energy / inference time
    assert m["energy_per_output_token_mj"] == 177.5
    assert m["output_tokens_per_j"] == round(4 / 0.71, 4)
    assert m["energy_source"] == "VDD_IN"
    assert m["min_mem_available_bytes"] == 3_000_000_000
    assert m["peak_worker_rss_bytes"] == 500_000_000
    assert (m["kv_capacity_tokens"], m["kv_peak_used_tokens"]) == (KV_CAPACITY, 14)
    assert m["kv_peak_usage"] == round(14 / KV_CAPACITY, 5)
    assert m["batch_size"] == 1
    assert (m["peak_flops_per_s"], m["peak_mem_bw_bytes_per_s"]) == (PEAK_FLOPS, PEAK_BW)

    # Energy windows are [t_sch, T_0], [T_0, T_1], ...; memory is the window the request ran in.
    assert telemetry.series_calls == [[T_SCH, *T_STEPS]]
    assert telemetry.memory_windows == [(T_SCH, T_FINISHED)]

    steps = record["steps"]
    assert steps["count"] == 4
    assert steps["token_id"] == [100, 101, 102, 103]
    assert steps["token_text"] == ["p0", "p1", "p2", "p3"]
    assert steps["t_ms"] == [25.0, 35.0, 46.0, 58.0]
    assert steps["itl_ms"] == [None, 10.0, 11.0, 12.0]
    assert steps["gpu_ms"] == GPU_MS
    assert steps["context_len"] == [10, 11, 12, 13]
    assert steps["flops"] == flops
    assert steps["bytes"] == moved
    assert steps["mfu"] == step_mfu
    assert steps["mbu"] == step_mbu
    assert steps["power_w"] == [15.0, 12.0, 10.0, 15.0]
    assert steps["energy_mj"] == [300.0, 120.0, 110.0, 180.0]
    assert steps["kv_usage"] == [round((10 + i) / KV_CAPACITY, 5) for i in range(4)]
    assert steps["device_free_bytes"] == [7000, 7001, 7002, 7003]
    assert steps["active_sequences"] == [1, 1, 1, 1]
    assert all(len(steps[key]) == 4 for key in steps if key != "count")


def test_record_envelope_and_unix_timestamps(model) -> None:
    record = record_for(model)
    shift = UNIX_ARRIVAL - T_ARRIVAL  # monotonic -> unix: the offset taken at arrival
    assert record["timestamps_ns"] == {
        "arrival": UNIX_ARRIVAL, "queued": T_QUEUED + shift, "scheduled": T_SCH + shift,
        "first_token": T_STEPS[0] + shift, "last_token": T_STEPS[-1] + shift, "finished": T_FINISHED + shift,
    }
    assert record["created"] == 1_700_000_000
    assert (record["object"], record["schema_version"]) == ("edgeserve.record", "1.1")
    assert (record["custom_id"], record["attempt_id"], record["status"], record["http_status"]) == (
        "seg-1", "att_1", "completed", 200)
    assert record["error"] is None and record["metadata"] == {"task": "demo"}
    assert record["system_fingerprint"] == "fp_" + ENGINE_KEY[:12]
    assert record["model"]["engine_fingerprint"] == ENGINE_KEY[:16]
    assert record["model"]["linear_params"] == model.linear_params
    assert (record["model"]["max_input_tokens"], record["model"]["kv_capacity_tokens"],
            record["model"]["max_batch_size"]) == (64, KV_CAPACITY, 2)
    assert record["system"] == {"kind": "stub", "peak_flops_per_s": PEAK_FLOPS}
    assert record["request"] == {"messages": [{"role": "user", "content": "hi"}], "params": {"max_tokens": 8}}
    assert record["output"] == {"text": "xxxx", "finish_reason": "stop", "token_ids": [100, 101, 102, 103]}
    assert record["usage"] == {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}


# --- n = 0, 1, 2 -------------------------------------------------------------------------------


def test_one_token_has_an_empty_decode_phase(model) -> None:
    m = record_for(model, 1, telemetry=StubTelemetry())["metrics"]
    # decode_ms = T_{n-1} - T_0 = 0; sums over no decode passes are 0, ratios over them undefined.
    assert m["decode_ms"] == 0.0
    assert m["decode_gpu_ms"] == 0.0
    assert (m["decode_flops"], m["decode_bytes"]) == (0, 0)
    assert m["decode_energy_j"] == 0.0 and m["energy_j"] == m["prefill_energy_j"] == 0.2
    for key in ("tpot_ms", "itl_ms", "decode_tokens_per_s", "decode_flops_per_s", "decode_bytes_per_s",
                "decode_mfu", "decode_mbu"):
        assert m[key] is None, key
    assert m["inference_ms"] == m["prefill_ms"] == 20.0
    assert m["e2e_ms"] == m["ttft_ms"] == 25.0
    assert m["e2e_tokens_per_s"] == 40.0
    assert m["prefill_flops"] == prefill_flops(model, 10)
    assert m["energy_per_output_token_mj"] == 200.0 and m["output_tokens_per_j"] == 5.0
    assert m["kv_peak_used_tokens"] == 11


def test_two_tokens_have_one_gap(model) -> None:
    m = record_for(model, 2, telemetry=StubTelemetry())["metrics"]
    assert m["decode_ms"] == m["tpot_ms"] == 10.0
    assert m["itl_ms"] == {"mean": 10.0, "p50": 10.0, "p90": 10.0, "p99": 10.0, "min": 10.0, "max": 10.0,
                           "stdev": 0.0}
    assert m["decode_tokens_per_s"] == 100.0
    assert m["decode_gpu_ms"] == 9.0
    assert m["decode_flops"] == decode_flops(model, 11)
    assert m["decode_flops_per_s"] == round(decode_flops(model, 11) / 0.009)
    assert m["decode_mfu"] == round(decode_flops(model, 11) / (0.009 * PEAK_FLOPS), 5)
    assert m["decode_mbu"] == round(pass_bytes(model, 11) / (0.009 * PEAK_BW), 5)
    assert m["decode_energy_j"] == 0.1 and m["energy_j"] == 0.3


def test_zero_tokens_after_scheduling(model) -> None:
    telemetry = StubTelemetry()
    record = record_for(model, 0, telemetry=telemetry, status="cancelled", result=make_result(0, finish="cancelled"))
    m = record["metrics"]
    assert "steps" not in record
    assert (m["preprocess_ms"], m["queue_ms"]) == (2.0, 3.0)
    for key in ("prefill_ms", "ttft_ms", "decode_ms", "inference_ms", "e2e_ms", "tpot_ms", "itl_ms",
                "prefill_tokens_per_s", "decode_tokens_per_s", "e2e_tokens_per_s", "prefill_gpu_ms", "decode_gpu_ms",
                "prefill_flops", "decode_flops", "prefill_mfu", "decode_mfu", "prefill_bytes", "decode_bytes",
                "decode_mbu", "energy_j", "prefill_energy_j", "decode_energy_j", "avg_power_w",
                "energy_per_output_token_mj", "output_tokens_per_j", "timing_source"):
        assert m[key] is None, key
    assert (m["kv_peak_used_tokens"], m["kv_peak_usage"]) == (10, round(10 / KV_CAPACITY, 5))
    assert record["timestamps_ns"]["scheduled"] == T_SCH + UNIX_ARRIVAL - T_ARRIVAL
    assert record["timestamps_ns"]["first_token"] is None and record["timestamps_ns"]["last_token"] is None
    assert telemetry.series_calls == []
    assert telemetry.memory_windows == [(T_SCH, T_FINISHED)]


def test_failed_record_without_a_done_event(model) -> None:
    telemetry = StubTelemetry()
    job = make_job(prompt_tokens=37)
    error = {"message": "worker restarted", "type": "server_error", "param": None, "code": "worker_restarted"}
    record = build_record(job, make_ctx(model, telemetry), status="failed", http_status=503,
                          t_finished_ns=T_FINISHED, error=error)
    m = record["metrics"]
    assert "steps" not in record
    assert record["error"] == error and record["status"] == "failed" and record["http_status"] == 503
    assert record["usage"] == {"prompt_tokens": 37, "completion_tokens": 0, "total_tokens": 37}
    assert record["output"] == {"text": "", "finish_reason": None, "token_ids": []}
    assert m["preprocess_ms"] == 2.0
    assert all(m[key] is None for key in ("queue_ms", "prefill_ms", "ttft_ms", "decode_ms", "e2e_ms", "energy_j"))
    assert record["timestamps_ns"]["scheduled"] is None
    assert record["timestamps_ns"]["finished"] == T_FINISHED + UNIX_ARRIVAL - T_ARRIVAL
    assert telemetry.memory_windows == [(T_ARRIVAL, T_FINISHED)]  # never scheduled: from arrival
    assert m["kv_peak_used_tokens"] == 37


def test_cancelled_while_queued_has_no_queue_time(model) -> None:
    record = build_record(make_job(t_queued_ns=0), make_ctx(model), status="cancelled", http_status=499,
                          t_finished_ns=T_FINISHED)
    assert record["metrics"]["preprocess_ms"] is None and record["metrics"]["queue_ms"] is None
    assert record["timestamps_ns"]["queued"] is None
    assert record["metrics"]["energy_source"] is None and record["metrics"]["min_mem_available_bytes"] is None


def test_unknown_prompt_count_from_the_worker_falls_back_to_the_admission_count(model) -> None:
    record = record_for(model, 2, result=make_result(2, prompt_tokens=-1), job=make_job(prompt_tokens=12))
    assert record["usage"]["prompt_tokens"] == 12
    assert record["steps"]["context_len"] == [12, 13]


# --- GPU time missing ------------------------------------------------------------------------------


def test_missing_gpu_times_fall_back_to_wall_time(model) -> None:
    record = record_for(model, done=make_done(4, gpu_ms=[]))
    m, steps = record["metrics"], record["steps"]
    flops = prefill_flops(model, 10)
    assert steps["gpu_ms"] == [None] * 4
    assert m["prefill_gpu_ms"] is None and m["decode_gpu_ms"] is None
    assert m["prefill_flops_per_s"] == round(flops / 0.020)  # prefill_ms instead of the GPU time
    assert m["decode_flops_per_s"] == round(242688 / 0.033)  # decode_ms
    assert m["prefill_mfu"] == round(flops / (0.020 * PEAK_FLOPS), 5)
    assert steps["mfu"][1:] == [round(decode_flops(model, 11 + i) / (gap * PEAK_FLOPS / 1e3), 5)
                                for i, gap in enumerate([10, 11, 12])]


def test_partial_gpu_times_do_not_make_a_partial_decode_sum(model) -> None:
    m = record_for(model, done=make_done(4, gpu_ms=[18.0, None, 10.0, 11.0]))["metrics"]
    assert m["prefill_gpu_ms"] == 18.0
    assert m["decode_gpu_ms"] is None  # not 21.0: one of the three decode passes has no GPU time
    assert m["decode_flops_per_s"] == round(242688 / 0.033)  # the whole decode phase on wall time
    assert m["decode_bytes_per_s"] == round(sum(pass_bytes(model, c) for c in (11, 12, 13)) / 0.033)


def test_itl_distribution_uses_unrounded_gaps(model) -> None:
    # Gaps of 1.0006, 1.0006 and 1.0001 ms: averaging them rounded to the microsecond gives 1.001.
    t_ns = [T_STEPS[0], T_STEPS[0] + 1_000_600, T_STEPS[0] + 2_001_200, T_STEPS[0] + 3_001_300]
    m = record_for(model, done=make_done(4, t_ns=t_ns))["metrics"]
    assert m["tpot_ms"] == 1.0
    assert m["itl_ms"]["mean"] == m["tpot_ms"]
    assert m["itl_ms"]["min"] == 1.0 and m["itl_ms"]["max"] == 1.001


# --- batches ---------------------------------------------------------------------------------------


def batch_records(model: ModelInfo, results: list[dict], contexts: list[list[int]], telemetry=None) -> list[dict]:
    """Two sequences batched together: 3 and 2 tokens, steps shared. ``contexts`` is Job.batch_contexts: per step,
    the context length (prompt + step) of every sequence still generating, as the scheduler sets it."""
    done = make_done(3, gpu_ms=GPU_MS, active=[2, 2, 1])
    done["results"] = results
    records = []
    for result in results:
        own_prompt = result["prompt_tokens"] if result["prompt_tokens"] >= 0 else 20
        job = make_job(prompt_tokens=own_prompt, batch_size=2, batch_contexts=contexts)
        records.append(build_record(job, make_ctx(model, telemetry), status="completed", http_status=200,
                                    t_finished_ns=T_FINISHED, done=done, result=result))
    return records


def test_batch_attribution_of_bytes_energy_and_utilisation(model) -> None:
    a, b = make_result(3, prompt_tokens=10), make_result(2, prompt_tokens=20)
    b["req"] = 1
    rec_a, rec_b = batch_records(model, [a, b], [[10, 20], [11, 21], [12]], StubTelemetry(watts=10.0))
    w = model.streamed_weight_bytes
    kv = lambda context: pass_bytes(model, context) - w  # noqa: E731 - a sequence's own KV + logits bytes

    # bytes: the weights are split between the sequences in the step, each pays its own KV.
    assert rec_a["steps"]["bytes"] == [int(kv(10) + w / 2), int(kv(11) + w / 2), pass_bytes(model, 12)]
    assert rec_b["steps"]["bytes"] == [int(kv(20) + w / 2), int(kv(21) + w / 2)]
    assert rec_a["steps"]["bytes"][0] + rec_b["steps"]["bytes"][0] == kv(10) + kv(20) + w
    # flops: each sequence's own.
    assert rec_a["steps"]["flops"] == [prefill_flops(model, 10), decode_flops(model, 11), decode_flops(model, 12)]
    assert rec_b["steps"]["flops"] == [prefill_flops(model, 20), decode_flops(model, 21)]
    # mfu / mbu: the whole step (both sequences) on its GPU time.
    mfu = [round((prefill_flops(model, 10) + prefill_flops(model, 20)) / (0.018 * PEAK_FLOPS), 5),
           round((decode_flops(model, 11) + decode_flops(model, 21)) / (0.009 * PEAK_FLOPS), 5),
           round(decode_flops(model, 12) / (0.010 * PEAK_FLOPS), 5)]
    mbu = [round((kv(10) + kv(20) + w) / (0.018 * PEAK_BW), 5), round((kv(11) + kv(21) + w) / (0.009 * PEAK_BW), 5),
           round(pass_bytes(model, 12) / (0.010 * PEAK_BW), 5)]
    assert rec_a["steps"]["mfu"] == mfu and rec_b["steps"]["mfu"] == mfu[:2]
    assert rec_a["steps"]["mbu"] == mbu and rec_b["steps"]["mbu"] == mbu[:2]
    assert rec_a["metrics"]["prefill_mfu"] == rec_b["metrics"]["prefill_mfu"] == mfu[0]
    assert rec_b["metrics"]["decode_mbu"] == mbu[1]
    # energy: shared equally by the sequences in a step; power is the board's.
    assert rec_a["steps"]["energy_mj"] == [100.0, 50.0, 110.0]
    assert rec_b["steps"]["energy_mj"] == [100.0, 50.0]
    assert rec_a["steps"]["power_w"] == [10.0, 10.0, 10.0]
    assert rec_a["metrics"]["energy_j"] == 0.26 and rec_b["metrics"]["energy_j"] == 0.15
    assert rec_a["steps"]["active_sequences"] == [2, 2, 1] and rec_b["steps"]["active_sequences"] == [2, 2]
    assert rec_a["metrics"]["batch_size"] == rec_b["metrics"]["batch_size"] == 2
    # a sequence's timing is its own tokens' steps
    assert rec_b["metrics"]["e2e_ms"] == 35.0 and rec_a["metrics"]["e2e_ms"] == 46.0


def test_batch_with_an_unreported_prompt_length_still_builds_records(model) -> None:
    a, b = make_result(3, prompt_tokens=10), make_result(2, prompt_tokens=-1)
    b["req"] = 1
    rec_a, rec_b = batch_records(model, [a, b], [[10, -1], [11, 0], [12]])  # -1: the worker's "unknown"
    # The batch's cost is unknown while the other sequence is in it; alone, the step is this sequence's.
    assert rec_a["steps"]["mfu"][:2] == [None, None] and rec_a["steps"]["mbu"][:2] == [None, None]
    assert rec_a["steps"]["mfu"][2] == round(decode_flops(model, 12) / (0.010 * PEAK_FLOPS), 5)
    assert rec_a["metrics"]["prefill_mfu"] is None
    assert rec_b["usage"]["prompt_tokens"] == 20 and rec_b["steps"]["flops"][0] == prefill_flops(model, 20)


# --- what can be missing -----------------------------------------------------------------------------


def test_no_energy_without_a_power_source(model) -> None:
    for telemetry in (None, StubTelemetry(source=None)):
        record = record_for(model, telemetry=telemetry)
        m = record["metrics"]
        assert record["steps"]["power_w"] == [None] * 4 and record["steps"]["energy_mj"] == [None] * 4
        assert all(m[key] is None for key in ("energy_j", "prefill_energy_j", "decode_energy_j", "avg_power_w",
                                              "energy_per_output_token_mj", "output_tokens_per_j"))
    assert record_for(model, telemetry=None)["metrics"]["energy_source"] is None


def test_one_window_without_power_samples_leaves_the_request_energy_unknown(model) -> None:
    record = record_for(model, telemetry=StubTelemetry(energies=[0.3, None, 0.11, 0.18]))
    assert record["steps"]["energy_mj"] == [300.0, None, 110.0, 180.0]
    assert record["steps"]["power_w"] == [15.0, None, 10.0, 15.0]
    assert record["metrics"]["energy_j"] is None and record["metrics"]["avg_power_w"] is None


def test_unknown_peaks_leave_utilisation_unknown(model) -> None:
    record = record_for(model, platform=StubPlatform(peak_flops_per_s=None, peak_mem_bw_bytes_per_s=0))
    assert record["steps"]["mfu"] == [None] * 4 and record["steps"]["mbu"] == [None] * 4
    m = record["metrics"]
    assert m["prefill_mfu"] is None and m["decode_mfu"] is None and m["decode_mbu"] is None
    assert m["decode_flops_per_s"] == round(242688 / 0.030)  # throughput does not need the peaks


def test_unknown_kv_capacity_leaves_kv_usage_unknown(model) -> None:
    engine = StubEngine(profile=StubProfile(kv_capacity=0))
    record = build_record(make_job(), make_ctx(model, engine=engine), status="completed", http_status=200,
                          t_finished_ns=T_FINISHED, done=make_done(2), result=make_result(2))
    assert record["steps"]["kv_usage"] == [None, None]
    assert record["metrics"]["kv_peak_usage"] is None and record["metrics"]["kv_peak_used_tokens"] == 12


def test_unsupported_architecture_reports_null_costs(model) -> None:
    other = dataclasses.replace(model, cost_model="unavailable", cost_model_reason="MoE")
    for n in (1, 4):
        record = build_record(make_job(), make_ctx(other), status="completed", http_status=200,
                              t_finished_ns=T_FINISHED, done=make_done(n), result=make_result(n))
        m = record["metrics"]
        assert record["steps"]["flops"] == [None] * n and record["steps"]["bytes"] == [None] * n
        assert all(m[key] is None for key in ("prefill_flops", "decode_flops", "prefill_bytes", "decode_bytes",
                                              "prefill_flops_per_s", "decode_flops_per_s", "decode_bytes_per_s",
                                              "prefill_mfu", "decode_mfu", "decode_mbu")), n
        assert m["tpot_ms"] == (11.0 if n == 4 else None)


def test_response_metrics_is_the_summary_subset() -> None:
    record = {"metrics": {"ttft_ms": 25.0, "tpot_ms": 11.0, "itl_ms": {"p50": 11.0, "p99": 11.98}, "batch_size": 1,
                          "energy_j": 0.71, "decode_flops": 5}}
    summary = response_metrics(record)
    assert summary["ttft_ms"] == 25.0 and summary["tpot_ms"] == 11.0 and summary["energy_j"] == 0.71
    assert (summary["itl_p50_ms"], summary["itl_p99_ms"]) == (11.0, 11.98)
    assert "decode_flops" not in summary and summary["queue_ms"] is None
    assert response_metrics({"metrics": {"itl_ms": None}})["itl_p50_ms"] is None


# --- percentiles ------------------------------------------------------------------------------------


@pytest.mark.parametrize("size", [1, 2, 3, 10, 101, 512])
def test_percentile_matches_numpy_linear(size: int) -> None:
    rng = random.Random(size)
    values = [rng.uniform(5, 80) for _ in range(size)]
    for q in (0.0, 0.5, 0.9, 0.99, 1.0):
        assert percentile(values, q) == pytest.approx(float(np.percentile(values, q * 100)), rel=1e-12, abs=1e-12)
    assert math.isnan(percentile([], 0.5))


def test_distribution_numbers() -> None:
    assert distribution([]) is None
    values = [40.0, 38.0, 39.0, 41.0, 60.0]
    d = distribution(values)
    assert d == {"mean": 43.6, "p50": 40.0, "p90": round(41.0 + 0.6 * 19.0, 3), "p99": round(41.0 + 0.96 * 19.0, 3),
                 "min": 38.0, "max": 60.0, "stdev": round(float(np.std(values)), 3)}


# --- Prometheus ---------------------------------------------------------------------------------------


def samples(metrics: PrometheusMetrics) -> dict[tuple[str, tuple], float]:
    out = {}
    for family in text_string_to_metric_families(metrics.render().decode()):
        for s in family.samples:
            out[(s.name, tuple(sorted(s.labels.items())))] = s.value
    return out


def value(metrics: PrometheusMetrics, name: str, **labels) -> float | None:
    return samples(metrics).get((name, tuple(sorted({"model_name": "tiny", **labels}.items()))))


def test_exposition_uses_the_documented_names_types_and_labels(model) -> None:
    pm = PrometheusMetrics("tiny", telemetry=StubTelemetry())
    pm.set_model_info(make_ctx(model))
    pm.set_queue(1, 2, 0.25)
    pm.set_memory(123)
    pm.rejected("queue_full")
    pm.observe(record_for(model, telemetry=StubTelemetry()))
    pm.observe({"status": "failed"})
    families = {f.name: f for f in text_string_to_metric_families(pm.render().decode())}
    histograms = ["time_to_first_token_seconds", "inter_token_latency_seconds", "request_time_per_output_token_seconds",
                  "e2e_request_latency_seconds", "request_queue_time_seconds", "request_prefill_time_seconds",
                  "request_decode_time_seconds", "request_inference_time_seconds", "request_prompt_tokens",
                  "request_generation_tokens", "request_energy_joules"]
    counters = ["prompt_tokens", "generation_tokens", "request_success", "request_errors", "admission_rejections",
                "estimated_flops", "estimated_read_bytes", "energy_joules"]
    gauges = ["num_requests_running", "num_requests_waiting", "kv_cache_usage_perc", "system_memory_available_bytes",
              "peak_flops_per_second", "peak_memory_bandwidth_bytes_per_second", "server_state", "model_info"]
    for kind, names in (("histogram", histograms), ("counter", counters), ("gauge", gauges)):
        for short in names:
            family = families[f"edgeserve_{short}"]
            assert family.type == kind, short
            assert family.samples, short
            for s in family.samples:
                assert s.labels["model_name"] == "tiny", s
                if kind == "counter":
                    assert s.name == f"edgeserve_{short}_total", s.name
    # every documented label value is exported from the start (at 0), so rate() sees the first event
    assert {s.labels["finished_reason"] for s in families["edgeserve_request_success"].samples} == {"stop", "length"}
    assert {s.labels["error_type"] for s in families["edgeserve_request_errors"].samples} == {"failed", "cancelled"}
    assert {s.labels["reason"] for s in families["edgeserve_admission_rejections"].samples} == {
        "not_ready", "low_memory", "duplicate_request_id", "queue_full", "shutting_down", "context_length",
        "client_limit", "rate_limited"}
    assert value(pm, "edgeserve_admission_rejections_total", reason="queue_full") == 1
    assert value(pm, "edgeserve_admission_rejections_total", reason="low_memory") == 0
    assert value(pm, "edgeserve_request_errors_total", error_type="cancelled") == 0
    assert {s.labels["rail"] for s in families["edgeserve_energy_joules"].samples} == {"VDD_IN"}
    info = families["edgeserve_model_info"].samples[0].labels
    assert info == {"model_name": "tiny", "engine_fingerprint": ENGINE_KEY[:16], "quantization": model.quantization,
                    "tensorrt_version": "10.3.0.30", "edgellm_version": "0.11.0", "max_input_tokens": "64",
                    "kv_capacity_tokens": str(KV_CAPACITY), "max_batch_size": "2"}


def test_observe_completed_record_in_seconds(model) -> None:
    pm = PrometheusMetrics("tiny")
    record = record_for(model, telemetry=StubTelemetry(energies=[0.3, 0.12, 0.11, 0.18]))
    pm.observe(record)
    m = record["metrics"]
    for name, key in (("time_to_first_token_seconds", "ttft_ms"), ("e2e_request_latency_seconds", "e2e_ms"),
                      ("request_queue_time_seconds", "queue_ms"), ("request_prefill_time_seconds", "prefill_ms"),
                      ("request_decode_time_seconds", "decode_ms"), ("request_inference_time_seconds", "inference_ms"),
                      ("request_time_per_output_token_seconds", "tpot_ms")):
        assert value(pm, f"edgeserve_{name}_count") == 1.0, name
        assert value(pm, f"edgeserve_{name}_sum") == pytest.approx(m[key] / 1e3), name
    assert value(pm, "edgeserve_inter_token_latency_seconds_count") == 3.0
    assert value(pm, "edgeserve_inter_token_latency_seconds_sum") == pytest.approx(0.033)
    assert value(pm, "edgeserve_inter_token_latency_seconds_bucket", le="0.01") == 1.0  # the 10 ms gap
    assert value(pm, "edgeserve_request_energy_joules_sum") == pytest.approx(0.71)
    assert value(pm, "edgeserve_request_prompt_tokens_sum") == 10.0
    assert value(pm, "edgeserve_request_generation_tokens_sum") == 4.0
    assert value(pm, "edgeserve_prompt_tokens_total") == 10.0
    assert value(pm, "edgeserve_generation_tokens_total") == 4.0
    assert value(pm, "edgeserve_estimated_flops_total") == m["prefill_flops"] + m["decode_flops"]
    assert value(pm, "edgeserve_estimated_read_bytes_total") == m["prefill_bytes"] + m["decode_bytes"]
    assert value(pm, "edgeserve_request_success_total", finished_reason="stop") == 1.0


def test_one_token_request_observes_a_zero_decode_time(model) -> None:
    pm = PrometheusMetrics("tiny")
    pm.observe(record_for(model, 1))
    assert value(pm, "edgeserve_request_decode_time_seconds_count") == 1.0
    assert value(pm, "edgeserve_request_decode_time_seconds_sum") == 0.0
    assert value(pm, "edgeserve_request_time_per_output_token_seconds_count") in (None, 0.0)


def test_cancelled_request_counts_its_gpu_work_but_no_latencies(model) -> None:
    pm = PrometheusMetrics("tiny")
    record = record_for(model, 2, status="cancelled", result=make_result(2, finish="cancelled"))
    pm.observe(record)
    assert value(pm, "edgeserve_request_errors_total", error_type="cancelled") == 1.0
    assert value(pm, "edgeserve_prompt_tokens_total") == 10.0
    assert value(pm, "edgeserve_generation_tokens_total") == 2.0
    assert value(pm, "edgeserve_estimated_flops_total") == prefill_flops(model, 10) + decode_flops(model, 11)
    assert value(pm, "edgeserve_estimated_read_bytes_total") == pass_bytes(model, 10) + pass_bytes(model, 11)
    assert value(pm, "edgeserve_time_to_first_token_seconds_count") in (None, 0.0)
    assert value(pm, "edgeserve_request_generation_tokens_count") in (None, 0.0)
    assert value(pm, "edgeserve_request_success_total", finished_reason="cancelled") is None


def test_request_that_never_ran_counts_only_as_an_error(model) -> None:
    pm = PrometheusMetrics("tiny")
    pm.observe(build_record(make_job(), make_ctx(model), status="failed", http_status=503, t_finished_ns=T_FINISHED,
                            error={"code": "queue_timeout"}))
    assert value(pm, "edgeserve_request_errors_total", error_type="failed") == 1.0
    for name in ("edgeserve_prompt_tokens_total", "edgeserve_generation_tokens_total",
                 "edgeserve_estimated_flops_total", "edgeserve_estimated_read_bytes_total"):
        assert value(pm, name) in (None, 0.0), name


ODD_RECORDS = [
    None, [], "record", {}, {"status": None}, {"status": "completed"},
    {"status": "completed", "metrics": None, "usage": None, "output": None, "steps": None},
    {"status": "completed", "metrics": [], "usage": "x", "output": 3, "steps": []},
    {"status": "completed", "usage": {"prompt_tokens": None, "completion_tokens": "7"}},
    {"status": "completed", "usage": {"prompt_tokens": -3, "completion_tokens": True}},
    {"status": "completed", "usage": {"prompt_tokens": 10 ** 400}},
    {"status": "completed", "metrics": {"ttft_ms": "x", "e2e_ms": float("nan"), "queue_ms": float("inf"),
                                        "tpot_ms": -1.0, "energy_j": float("nan")}},
    {"status": "completed", "metrics": {"prefill_flops": "1", "decode_flops": -5, "prefill_bytes": float("inf")}},
    {"status": "completed", "steps": {"itl_ms": None}},
    {"status": "completed", "steps": {"itl_ms": ["a", None, float("nan"), -2.0, {}]}},
    {"status": "completed", "steps": {"itl_ms": 12.0}},
    {"status": "completed", "output": {"finish_reason": None}},
    {"status": "completed", "output": {"finish_reason": {"odd": 1}}},
    {"status": "cancelled", "steps": {"itl_ms": [None, 1.0]}, "usage": {"prompt_tokens": -1}},
    {"status": ["odd"]},
]


@pytest.mark.parametrize("record", ODD_RECORDS)
def test_observe_never_raises(record) -> None:
    pm = PrometheusMetrics("tiny")
    pm.observe(record)
    for (name, _), number in samples(pm).items():
        assert math.isfinite(number), name  # NaN or inf never reaches a sum or a counter


def test_observe_keeps_the_usable_values_of_an_odd_record() -> None:
    pm = PrometheusMetrics("tiny")
    pm.observe({"status": "completed", "usage": {"prompt_tokens": 9, "completion_tokens": None},
                "metrics": {"ttft_ms": 250.0, "e2e_ms": float("nan")}, "steps": {"itl_ms": [None, 30.0, "x"]}})
    assert value(pm, "edgeserve_time_to_first_token_seconds_sum") == 0.25
    assert value(pm, "edgeserve_e2e_request_latency_seconds_count") in (None, 0.0)
    assert value(pm, "edgeserve_inter_token_latency_seconds_count") == 1.0
    assert value(pm, "edgeserve_prompt_tokens_total") == 9.0
    assert value(pm, "edgeserve_request_generation_tokens_count") in (None, 0.0)


def test_energy_counter_follows_the_telemetry() -> None:
    telemetry = StubTelemetry(total_j=12.5)
    pm = PrometheusMetrics("tiny")
    assert ("edgeserve_energy_joules_total", (("model_name", "tiny"), ("rail", "VDD_IN"))) not in samples(pm)
    pm.set_telemetry(telemetry)
    pm.set_telemetry(StubTelemetry(total_j=99.0))  # the first telemetry stays; no second collector
    assert value(pm, "edgeserve_energy_joules_total", rail="VDD_IN") == 12.5
    telemetry.total_j = 20.0
    assert value(pm, "edgeserve_energy_joules_total", rail="VDD_IN") == 20.0
    text = pm.render().decode()
    assert text.count("# TYPE edgeserve_energy_joules_total counter") == 1

    quiet = PrometheusMetrics("tiny", telemetry=StubTelemetry(source=None))
    assert not [key for key in samples(quiet) if key[0] == "edgeserve_energy_joules_total"]


def test_gauges_state_peaks_queue_and_memory(model) -> None:
    sample = MemorySample(t_ns=1, mem_available_bytes=4_000_000, swap_free_bytes=0, worker_rss_bytes=None,
                          frontend_rss_bytes=1)
    pm = PrometheusMetrics("tiny", telemetry=StubTelemetry(memory=sample))
    assert value(pm, "edgeserve_server_state", state="starting") == 1.0
    pm.set_state("ready")
    states = {key[1]: v for key, v in samples(pm).items() if key[0] == "edgeserve_server_state"}
    assert sorted(v for v in states.values()) == [0.0] * 7 + [1.0]
    assert value(pm, "edgeserve_server_state", state="ready") == 1.0
    assert value(pm, "edgeserve_server_state", state="starting") == 0.0

    pm.set_peaks(StubPlatform(peak_flops_per_s=None, peak_mem_bw_bytes_per_s=None))
    assert value(pm, "edgeserve_peak_flops_per_second") is None
    pm.set_model_info(make_ctx(model))
    assert value(pm, "edgeserve_peak_flops_per_second") == PEAK_FLOPS
    assert value(pm, "edgeserve_peak_memory_bandwidth_bytes_per_second") == PEAK_BW

    pm.set_queue(running=2, waiting=5, kv_usage=0.75)
    assert (value(pm, "edgeserve_num_requests_running"), value(pm, "edgeserve_num_requests_waiting"),
            value(pm, "edgeserve_kv_cache_usage_perc")) == (2.0, 5.0, 0.75)
    assert value(pm, "edgeserve_system_memory_available_bytes") == 4_000_000.0  # render() reads the telemetry
    pm.set_memory(None)
    assert value(pm, "edgeserve_system_memory_available_bytes") == 4_000_000.0
