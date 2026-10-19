# Metrics

Every admitted request produces a **record**: a whole-request summary (`metrics`) and per-token
arrays (`steps`). A non-streaming request with `"return_record": true` gets it in the response.
With `--output-dir`, every record is also saved as `<output-dir>/records/<id>.json` and served at
`GET /v1/records/<id>`. Aggregates are exported on `/metrics` for Prometheus.

## Terms

| Term | Meaning |
|---|---|
| prefill | the first pass over the whole prompt; it produces the first output token |
| decode | each later pass; it produces one more token |
| TTFT | time to first token |
| ITL | inter-token latency: the gap between two consecutive output tokens |
| TPOT | time per output token: the mean ITL of one request |
| KV cache | memory holding the attention keys and values of the prompt and the output so far; its capacity limits prompt + output |
| FLOP, FLOPs | floating-point operation(s); a multiply-accumulate counts as 2 |
| MFU | model FLOPs utilisation: FLOPs done per second ÷ the GPU's peak FLOP/s (0–1) |
| MBU | memory-bandwidth utilisation: bytes moved per second ÷ the peak memory bandwidth (0–1) |

Tools name some of these differently (NVIDIA AIPerf's "ITL" is vLLM's and MLPerf's TPOT), so the
formulas below are the definitions.

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

A request refused before it is queued gets no record, so it cannot overwrite the record of an
earlier attempt with the same ID. Admission refusals (not ready, low memory, a duplicate ID, a full
queue, shutting down, a prompt or `max_tokens` too long for the context) are counted in
`edgeserve_admission_rejections_total{reason}`; other refusals (401, invalid JSON or parameters, an
unknown model) appear only in the access log. A request cut short by a worker crash is recorded as
failed (`worker_restarted`) without `steps`, although a streaming client may already have received
part of the text.

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

Responses carry a subset of these in `metrics` (a stream only in its `include_usage` chunk), plus
`itl_p50_ms` and `itl_p99_ms` (p50 and p99 of `itl_ms`).

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
| `prefill_tokens_per_s` | P / prefill_ms | tok/s |
| `decode_tokens_per_s` | (n − 1) / decode_ms | tok/s |
| `e2e_tokens_per_s` | n / e2e_ms | tok/s |
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
| `batch_size` | sequences in this request's batch, itself included (1 = alone) | — |
| `peak_flops_per_s`, `peak_mem_bw_bytes_per_s` | the MFU / MBU denominators used (current power mode) | FLOP/s, B/s |
| `timing_source` | `engine_callback` (T_i from Edge-LLM's per-token callback); null when no token was produced | — |

When the runtime reports no GPU time for a pass, wall time stands in for it in the values above that use GPU
time: T_0 − t_sch for the prefill, decode_ms for the decode phase, T_i − T_{i−1} for one decode pass (in the
MFU / MBU means).

TTFT and e2e are measured at the server: a client additionally sees network and TLS time.

With `batch_size` > 1, FLOPs, bytes and energy (and the rates and `avg_power_w` made from them) are
this request's share of each step, while `prefill_mfu`, `decode_mfu` and `decode_mbu` are of the
whole step (all its sequences).

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
| `power_w`, `energy_mj` | mean power over the pass window (module input power on Jetson, not split across sequences); energy of the window, shared equally by the sequences in it |
| `kv_usage` | (P + i) / KV capacity |
| `device_free_bytes` | `cudaMemGetInfo` free memory after the pass (on Jetson: the whole system's available memory) |
| `active_sequences` | sequences that produced a token in this step |

## FLOPs and bytes

Every linear layer's size is read from the model's own graph (`edgeserve/model.py`), never from
`config.json`, so pruned models whose layers differ in width are handled too: the `gemm_n × gemm_k`
attributes of GEMM plugins (e.g. INT4) and the weight shapes of Gemm/MatMul nodes (e.g. FP16, FP8 or
INT8 weights, an fp16 lm_head). Layer count, head count and head_dim come from the model folder's
`config.json`. Counting 2 FLOPs per multiply-accumulate:

* decode at context *c*: `2 · linear_params + 4 · layers · heads · head_dim · c` (`linear_params`: the weights of every linear layer, lm_head included, the embedding table not)
* prefill of *P* prompt tokens: `2 · (linear_params − lm_head) · P + 2 · lm_head + 4 · layers · heads · head_dim · P(P+1)/2` (the output layer, lm_head, runs for the last token only; each prompt token attends to itself and the ones before it)
* bytes per decode pass: `streamed_weight_bytes` (every weight byte except the embedding table, read
  once per pass: far more than the GPU's L2 cache holds) + KV cache for *c* positions + fp32 logits.

Example, a pruned 6B INT4 Llama-3.1 model with a 16,384-token output vocabulary: 5.03 B linear
parameters, about 10.1 GFLOP per decode token plus 0.52 MFLOP per token of context, 2.70 GB read
per decode pass. Decode does ≈ 4 FLOP per byte it reads, while an Orin Nano can do ≈ 147 FLOP in
the time it reads one byte: decode speed is set by memory bandwidth, so **MBU** is the meaningful
decode efficiency metric; MFU matters for prefill. Element-wise work (norms, RoPE, softmax) is
under 0.1 % and not counted.

For models the cost model does not cover (mixture-of-experts, hybrid state-space / linear
attention, sliding-window or sparse attention) FLOPs, bytes, MFU and MBU are `null` rather than
wrong, and `model.cost_model` is `unavailable`; `edgeserve info --model <model-dir>` shows why
(`cost_model_reason`).

## Peaks (MFU / MBU denominators)

| Platform | Peak FLOP/s | Peak bandwidth |
|---|---|---|
| Jetson | SMs × FLOPs/clock/SM (2048 on Orin, sm_87) × GPU `devfreq` max_freq | EMC clock cap (`/sys/kernel/nvpmodel_clk_cap/emc`; Thor: `/sys/class/devfreq/bwmgr/max_freq`) × 2 × bus width / 8 |
| dGPU | SMs × FLOPs/clock/SM for its architecture × max SM clock (NVML) | max memory clock (NVML) × 2 × bus width / 8 |

Peak FLOP/s is the dense FP16 tensor-core peak, used whatever the model's precision. Both peaks are
re-read every 30 s because the Jetson power mode can change. On an Orin Nano Super: 25W mode =
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

## Batch summary (`run.json`)

`edgeserve batch` also writes `run.json`: request counts; `usage` and `energy_j` summed over all
records (failed and cancelled ones too); request and token throughputs per `busy_s`, the time at
least one request was in progress; and mean, p50, p90, p99, min and max of `ttft_ms`, `tpot_ms`,
`e2e_ms`, `queue_ms` and `decode_tokens_per_s` over the completed requests (`itl_ms`: over all
their token gaps).

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
first event. `edgeserve_kv_cache_usage_perc` is the fraction (0–1, not a percentage) of the KV cache
*reserved* by the running requests (prompt + `max_tokens` each, over capacity × `--max-batch-size`;
Edge-LLM allocates a request's KV pages up front), so it stays constant while a request runs.
`edgeserve_server_state` is 1 for the current state and 0 for the others.

Useful queries: power `rate(edgeserve_energy_joules_total[1m])`; average memory-bandwidth utilisation
(all phases and idle time included; per-request decode MBU is in the records)
`rate(edgeserve_estimated_read_bytes_total[1m]) / edgeserve_peak_memory_bandwidth_bytes_per_second`;
p99 TTFT `histogram_quantile(0.99, sum by (le) (rate(edgeserve_time_to_first_token_seconds_bucket[5m])))`.
