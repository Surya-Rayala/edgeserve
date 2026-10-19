# Metrics

Every admitted request produces a **record** (in API responses with `"return_record": true`, in
`<output-dir>/records/<id>.json`, and from `GET /v1/records/<id>`) with a whole-request
summary (`metrics`) and per-token arrays (`steps`). A request refused before it is queued gets no
record, so it cannot overwrite the record of an earlier attempt with the same ID. Refusals at
admission (not ready, low memory, a duplicate ID, a full queue, shutting down, a prompt that is too
long) are counted in `edgeserve_admission_rejections_total{reason}`; requests rejected earlier (401,
invalid JSON or parameters, an unknown model) appear only in the access log. A request cut short by a worker crash is recorded as failed (`worker_restarted`)
without per-token data: the worker died before reporting it, although a streaming client may
already have received part of the text. Aggregates are exported on `/metrics` for
Prometheus. Tools name these metrics inconsistently (NVIDIA AIPerf's "ITL" is what vLLM and
MLPerf call TPOT), so the formulas below are the definition.

## Record layout

| Key | Content |
|---|---|
| `object`, `schema_version` | `"edgeserve.record"`, `"1.0"` |
| `custom_id`, `attempt_id` | the request ID; this attempt's own ID (`att_…`) |
| `status`, `http_status`, `error` | `completed`, `failed` or `cancelled`; the HTTP status of the outcome (499: the client went away; for a stream that had already started, the response itself was 200 and the error went out as an SSE event; `results.jsonl` `status_code` is the same); the OpenAI-style error object (code `client_disconnected` for a 499), or null |
| `created`, `endpoint`, `metadata`, `system_fingerprint` | arrival time (Unix seconds); `/v1/chat/completions` or `/v1/completions`; the request's `metadata`; `fp_` + 12 hex digits of the engine key |
| `model` | `name`, `path`, `engine_fingerprint`, `quantization`, `kv_cache_dtype`, `vocab_size`, `output_vocab_size`, `linear_params`, `streamed_weight_bytes`, `cost_model`, `max_input_tokens`, `kv_capacity_tokens`, `max_batch_size` |
| `system` | the machine: board, L4T/JetPack, GPU, CUDA, TensorRT and Edge-LLM versions, power mode, clock caps, peaks, power rails |
| `request` | the `messages` as the chat template received them (a `developer` role becomes `system`, text parts are joined) or `prompt`, and `params`: the effective `max_tokens`, `temperature`, `top_p`, `top_k`, `seed` (the server's random seed when the request sampled without one), `stop`, `stream` |
| `output` | `text` (without the end-of-sequence token and stop string), `finish_reason`, `token_ids` |
| `usage` | `prompt_tokens`, `completion_tokens`, `total_tokens` |
| `timestamps_ns` | Unix time in ns of `arrival`, `queued`, `scheduled`, `first_token`, `last_token`, `finished` (the events below) |
| `metrics`, `steps` | below; `steps` is absent when no token was generated |

Records the batch client writes for a request that never got a server record (an invalid input
line, an HTTP error) have only `object`, `schema_version`, `custom_id`, `status` (`failed`),
`http_status`, `error` and `attempts` (and `line_no` for an invalid line).

## Clock and events

All durations come from one monotonic clock shared by the front-end and the worker.

| Symbol | Event |
|---|---|
| t_arrival | request body fully received, before it is parsed (the upload itself is network time and not counted) |
| t_queued | validated, prompt tokens counted, admitted to the queue |
| t_sch | the worker started the request (`handleRequest` called) |
| T_i | output token *i* became available: Edge-LLM's per-token callback, right after the step's stream synchronize. T_0 ends the prefill. |
| t_finished | the worker's result received and the record built (just before the response is sent) |

*n* = completion tokens = every sampled token, including a terminating end-of-sequence token
(the returned text excludes it). *P* = prompt tokens after the chat template.

## Per-request (`record.metrics`)

| Key | Definition | Unit |
|---|---|---|
| `preprocess_ms` | t_queued − t_arrival (validation + token counting) | ms |
| `queue_ms` | t_sch − t_queued | ms |
| `prefill_ms` | T_0 − t_sch (template + tokenization + prefill + first sample) | ms |
| `ttft_ms` | T_0 − t_arrival — time to first token, server side | ms |
| `decode_ms` | T_{n−1} − T_0 | ms |
| `inference_ms` | T_{n−1} − t_sch | ms |
| `e2e_ms` | T_{n−1} − t_arrival — time to last token, server side | ms |
| `tpot_ms` | decode_ms / (n − 1): time per output token, one value per request (MLPerf / vLLM TPOT, AIPerf "ITL") | ms |
| `itl_ms` | mean, p50, p90, p99, min, max, stdev of the gaps T_i − T_{i−1} (vLLM ITL, AIPerf "inter-chunk latency") | ms |
| `prefill_tokens_per_s` | P / prefill time | tok/s |
| `decode_tokens_per_s` | (n − 1) / decode time | tok/s |
| `e2e_tokens_per_s` | n / e2e | tok/s |
| `prefill_gpu_ms`, `decode_gpu_ms` | GPU time of the prefill pass / sum over decode passes (CUDA events inside Edge-LLM) | ms |
| `prefill_flops`, `decode_flops` | sum of `steps.flops` over the phase | FLOP |
| `prefill_flops_per_s`, `decode_flops_per_s` | FLOPs / GPU time of the phase | FLOP/s |
| `prefill_mfu`, `decode_mfu` | model FLOPs utilisation: FLOPs / (GPU time × peak FLOP/s); decode = mean over steps | 0–1 |
| `prefill_bytes`, `decode_bytes` | estimated DRAM bytes moved in the phase | B |
| `decode_bytes_per_s` | decode bytes / decode GPU time | B/s |
| `decode_mbu` | memory-bandwidth utilisation: bytes / (GPU time × peak bandwidth), mean over decode steps | 0–1 |
| `energy_j` | energy from t_sch to T_{n−1} (`prefill_energy_j` + `decode_energy_j`) | J |
| `avg_power_w` | energy_j / inference time | W |
| `energy_per_output_token_mj`, `output_tokens_per_j` | energy_j / n, n / energy_j | mJ, tok/J |
| `energy_source` | what was measured: Jetson rail (e.g. `VDD_IN`, module input) or `nvml` (board power) | — |
| `min_mem_available_bytes` | lowest system MemAvailable while the request ran | B |
| `peak_worker_rss_bytes` | worker process RSS peak (CPU-side memory only) | B |
| `kv_capacity_tokens`, `kv_peak_used_tokens`, `kv_peak_usage` | KV cache capacity per sequence, P + n, their ratio | tokens, 0–1 |
| `batch_size` | sequences generated together with this one | — |
| `peak_flops_per_s`, `peak_mem_bw_bytes_per_s` | the MFU / MBU denominators used (current power mode) | FLOP/s, B/s |
| `timing_source` | `engine_callback` (T_i from Edge-LLM's per-token callback); null when no token was produced | — |

When the runtime reports no GPU time for a pass, wall time stands in for it in the values above that use GPU
time: T_0 − t_sch for the prefill, decode_ms for the decode phase, T_i − T_{i−1} for one decode pass (in the
MFU / MBU means).

TTFT and e2e are measured at the server: a client additionally sees network and TLS time.

## Per-step (`record.steps`)

Columnar arrays of length `count` = n. Entry *i* is the forward pass that produced token *i*:
*i* = 0 is the prefill, *i* ≥ 1 are decode passes.

| Array | Definition |
|---|---|
| `token_id`, `token_text` | sampled token and its raw piece (pieces of a multi-byte character may show as "�") |
| `t_ms` | T_i − t_arrival |
| `itl_ms` | T_i − T_{i−1} (null for i = 0) |
| `gpu_ms` | GPU time of the pass (CUDA events) |
| `context_len` | keys attended: P + i |
| `flops` | i = 0: prefill FLOPs of P tokens; i ≥ 1: decode FLOPs at context P + i |
| `bytes` | this sequence's share of the bytes moved (weights split across the sequences in the step + its own KV cache) |
| `mfu`, `mbu` | utilisation of the whole step (all sequences in it) on its GPU time (on the step's wall time when the runtime reported no GPU time) |
| `power_w`, `energy_mj` | mean board power over the pass window; energy of the window (shared equally by the sequences in it) |
| `kv_usage` | (P + i) / KV capacity |
| `device_free_bytes` | `cudaMemGetInfo` free memory after the pass (on Jetson: the whole system's available memory) |
| `active_sequences` | sequences that produced a token in this step |

## FLOPs and bytes

Computed from the model's own tensors (`edgeserve/model.py`), never from `config.json`, so pruned
models whose layers differ in width are handled too. Every linear layer's shape is read from the
graph: INT4 plugin attributes `gemm_n × gemm_k`, and the weight initializers of Gemm/MatMul nodes
(e.g. an fp16 lm_head). Counting 2 FLOPs per multiply-accumulate:

* decode at context *c*: `2 · linear_params + 4 · layers · heads · head_dim · c`
* prefill of *P* new tokens after *c₀* cached: `2 · (linear − lm_head) · P + 2 · lm_head + 4 · layers · heads · head_dim · (P·c₀ + P(P+1)/2)` (causal; logits only for the last token)
* bytes per decode pass: every weight byte except the embedding table (e.g. INT4 codes + scales +
  an fp16 lm_head; read once per pass, far more than the GPU's L2 cache holds) + KV cache for *c*
  positions + fp32 logits.

Example, a pruned 6B INT4 Llama-3.1 model with a 16,384-token output vocabulary: 5.03 B linear
parameters, about 10.1 GFLOP per decode token plus 0.52 MFLOP per token of context, 2.70 GB read
per decode pass. Decode is memory-bound (≈ 4 FLOP per byte against a hardware balance of
≈ 147 FLOP/B on an Orin Nano), so **MBU** is the meaningful decode efficiency metric; MFU matters
for prefill. Element-wise work (norms, RoPE,
softmax) is under 0.1 % and not counted. For architectures the cost model does not cover (MoE,
hybrid state-space / linear attention) FLOPs and bytes are `null` rather than wrong.

## Peaks (MFU / MBU denominators)

| Platform | Peak FLOP/s | Peak bandwidth |
|---|---|---|
| Jetson | SMs × FLOPs/clock/SM (2048 on Orin, sm_87) × GPU `devfreq` max_freq | EMC clock cap (`/sys/kernel/nvpmodel_clk_cap/emc`; Thor: `/sys/class/devfreq/bwmgr/max_freq`) × 2 × bus width / 8 |
| dGPU | SMs × FLOPs/clock/SM for its architecture × max SM clock (NVML) | max memory clock (NVML) × 2 × bus width / 8 |

Re-read every 30 s because the Jetson power mode can change. On an Orin Nano Super: 25W mode =
15.0 TFLOP/s and 102.4 GB/s; MAXN_SUPER = 16.7 TFLOP/s and 102.4 GB/s (same memory clock, so
decode is about as fast in 25W); 15W/7W = 68.3 GB/s. CUDA's own clock attributes are wrong on
Jetson (they ignore power-mode caps) and are not used.

## Energy

Jetson: the module input rail (`VDD_IN` on Orin Nano/NX; the most inclusive rail elsewhere) is
sampled at 20 Hz on a background thread and integrated with the trapezoid rule over each step and
request. It measures the module, not the carrier board, NVMe or fan. The INA3221 sensor averages
512 conversions (`samples` in its hwmon folder), so its readings lag the real power by about
0.3 s and settle within ~0.6-1 s: after idle, a request's first ~0.5 s is under-counted and its
last ~0.5 s of energy shows up after it ended. **Per-request energy is reliable for requests of
several seconds (e.g. summarizing a long document); requests under ~1.5 s that follow idle read 25-50% low;
per-step energy is an estimate.** Back-to-back requests are not affected (the lag moves energy
between neighbours), and `edgeserve_energy_joules_total` is exact over time. Measured idle power on
the Orin Nano Super: ~5 W.
Discrete GPUs: NVML's total-energy counter (board power).

## Memory

On Jetson the GPU has no memory of its own and Linux cannot attribute GPU allocations to a
process without root, so records report what can be measured honestly: system MemAvailable
(lowest during the request), free memory per step, and the worker's CPU-side RSS. The memory the
loaded engine takes is logged at startup (`worker ready: … GiB in use after load`, measured as the
drop in free memory) and the build manifest records the engine build's peak memory.

## Prometheus (`GET /metrics`)

vLLM's metric names with an `edgeserve_` prefix (an underscore: Prometheus reserves colons), all
labelled `model_name`, so vLLM-style dashboards and NVIDIA AIPerf's server-metrics collection work.

| Metric | Type |
|---|---|
| `edgeserve_time_to_first_token_seconds`, `edgeserve_inter_token_latency_seconds`, `edgeserve_request_time_per_output_token_seconds`, `edgeserve_e2e_request_latency_seconds`, `edgeserve_request_queue_time_seconds`, `edgeserve_request_prefill_time_seconds`, `edgeserve_request_decode_time_seconds`, `edgeserve_request_inference_time_seconds` | histogram (s) |
| `edgeserve_request_prompt_tokens`, `edgeserve_request_generation_tokens`, `edgeserve_request_energy_joules` | histogram |
| `edgeserve_prompt_tokens_total`, `edgeserve_generation_tokens_total`, `edgeserve_request_success_total{finished_reason}`, `edgeserve_request_errors_total{error_type}`, `edgeserve_admission_rejections_total{reason}`, `edgeserve_estimated_flops_total`, `edgeserve_estimated_read_bytes_total`, `edgeserve_energy_joules_total{rail}` | counter |
| `edgeserve_num_requests_running`, `edgeserve_num_requests_waiting`, `edgeserve_kv_cache_usage_perc`, `edgeserve_system_memory_available_bytes`, `edgeserve_peak_flops_per_second`, `edgeserve_peak_memory_bandwidth_bytes_per_second`, `edgeserve_server_state{state}`, `edgeserve_model_info{...}` | gauge |

Label values: `error_type` is the record status (`failed`, `cancelled`); `reason` is `not_ready`,
`low_memory`, `duplicate_request_id`, `queue_full`, `shutting_down` or `context_length`; `state` is one of
`starting`, `building_engine`, `loading`, `ready`, `restarting`, `draining`, `failed`, `stopped`. Every
one of these series is exported from the start (at 0), so `rate()` and `increase()` also count the
first event. `edgeserve_kv_cache_usage_perc` is the KV cache *reserved* by the running requests
(prompt + `max_tokens` each, over capacity × batch size; Edge-LLM allocates a request's KV pages up
front), so it stays constant while a request runs.

Useful queries: power `rate(edgeserve_energy_joules_total[1m])`; average memory-bandwidth utilisation
(all phases and idle time included; per-request decode MBU is in the records)
`rate(edgeserve_estimated_read_bytes_total[1m]) / edgeserve_peak_memory_bandwidth_bytes_per_second`;
p99 TTFT `histogram_quantile(0.99, sum by (le) (rate(edgeserve_time_to_first_token_seconds_bucket[5m])))`.
