# Architecture

```
   clients: OpenAI SDK / curl / `edgeserve batch`            (any machine on the network)
                 │  HTTPS + API key (or http on 127.0.0.1)
┌────────────────▼──────────────────────────────────────────────┐
│ edgeserve  (Python · FastAPI · uvicorn)   front-end/supervisor│
│  api/app.py      OpenAI endpoints, IDs, auth, SSE, errors     │
│  scheduler.py    admission, bounded queue, batching, cancel   │
│  metrics.py      per-step + per-request records, Prometheus  │
│  telemetry.py    power (20 Hz) and memory (2 Hz) sampling     │
│  records.py      output folder (records/, results.jsonl)      │
│  engine.py       engine cache + llm_build orchestration       │
│  service.py      startup pipeline, worker restarts, state     │
└───────────┬──────────────────────────────▲────────────────────┘
   commands │ stdin (JSON lines)    events │ stdout (JSON lines)        logs: stderr
┌───────────▼──────────────────────────────┴────────────────────┐
│ edgeserve-worker  (C++ · edgeserve/native/worker.cpp)         │
│  TensorRT Edge-LLM LLMInferenceRuntime: engine, KV cache,     │
│  CUDA graphs, per-token callback, CUDA-event step timers      │
└───────────────────────────────────────────────────────────────┘
   engines: ~/.cache/edgeserve/engines/<model>-<key>/   built by Edge-LLM's llm_build (own process)
```

## The GPU runtime

The package ships the worker's C++ source, the pinned Edge-LLM commit, edgeserve's patches to it
and `build.sh`, all in `edgeserve/native/`. Nothing in the package is compiled for a particular
GPU, so the same wheel works on every platform. `edgeserve setup` (`edgeserve/nativebuild.py`):

1. Copies the packaged sources into the build tree, `~/.cache/edgeserve/native-build/sources`, when
   they changed (so CMake never refers into a virtual environment that may be deleted).
2. Runs `build.sh` there, under a file lock. It clones Edge-LLM at the pinned commit and applies
   the patches (again only when they changed), generates the CuTe DSL kernel pack if no prebuilt
   one matches the GPU, configures CMake for the platform (Jetson Orin, Jetson Thor or a discrete
   GPU) and builds the TensorRT plugin, `llm_build`, `edgeserve-worker` and `attention-check`.
   Its output goes to the terminal and to `build.log`.
3. Installs those four files and `build_info.json` (commit, patch hashes, plugin hash, CUDA,
   TensorRT, GPU, kernel pack, the hash of the packaged sources) into a new version folder,
   `~/.local/share/edgeserve/native/<version>/`, and points the `current` link at it with an atomic
   rename. A server that is already running keeps the binaries it started with; the previous
   version is kept.
4. Runs `attention-check` and records the result.

At startup, `edgeserve serve` resolves `current` once. It refuses to start when the installed
runtime was built from other sources than the installed package ("run edgeserve setup"). Engines
depend on the plugin's hash, so they are rebuilt once after a runtime rebuild.

## Exporting

How to use it: [EXPORT.md](EXPORT.md). `edgeserve export` (`edgeserve/exporter.py`) runs TensorRT Edge-LLM's own exporter, installed on
first use into `~/.local/share/edgeserve/exporter/`: a clone at the pinned commit with edgeserve's
exporter patches applied (`edgeserve/export_patches/`), and its own virtual environment with the
`[export]` extra. The installation is recorded with a key over the commit and the patches, so an
edgeserve upgrade that changes them reinstalls it on the next export. The patches:

| Patch | Why |
|---|---|
| 0001 per-layer MLP widths | pruned checkpoints whose layers have different MLP widths (`intermediate_sizes` in `config.json`) |
| 0002 exact INT4 zero points | the INT4 kernel has no zero-point input; instead of folding asymmetric zero points into the codes (which clips them), the codes stay exact and a small FP16 GEMM over per-group activation means adds `(8 − zero) × scale` |

Around the exporter, `edgeserve export` chooses `--externalize-weights` (INT4 feed-forward weights
and an FP16 `lm_head` stay outside the engine, which bounds the engine build's memory), keeps only
the `llm` part of the output, copies `generation_config.json`, rewrites `model.onnx.data` without
the bytes the exporter left unused, and checks the result with the same model analysis `serve`
uses. The C++ runtime needs no change for any of this.

## Settings

Every command reads `~/.config/edgeserve/edgeserve.env` (`edgeserve/config.py`). A setting comes
from its command-line flag, else the environment variable of the same name, else the file, else
its default. File values are never copied into the environment, so the API key does not reach the
Edge-LLM processes (`NativeBuild.env()` also removes it from their environment when it was set
there). The clients `edgeserve check` and `edgeserve batch` take the address, API key and CA
certificate from the same file for this machine's own server, and never send that key to another
host.

## Why two processes

On Jetson the GPU uses the same RAM as the CPU, and GPU allocations are invisible to the usual
Linux controls: they are not counted in a process's RSS, not charged to cgroups (systemd
`MemoryMax` and `docker --memory` do not limit them), and not swappable. When memory runs out
the kernel's OOM killer, which only sees RSS, may pick an unrelated process. edgeserve therefore
keeps the engine in its own process:

* The worker volunteers as the preferred OOM victim (`oom_score_adj` 900; NVIDIA's own advice for
  Jetson). Killing it frees the multi-GB engine, which is what actually relieves memory. (The
  systemd unit sets `OOMPolicy=continue`: systemd would otherwise stop the whole service.)
* A CUDA fault, a sticky CUDA error or an OOM kill ends only the worker. The front-end fails the
  affected requests with HTTP 503 (`worker_restarted`), writes their records, and starts a fresh
  worker with back-off (1, 2, 4, 8, 16 s). After 5 restarts in 10 minutes it reports `failed`.
* The front-end never creates a CUDA context of its own (a context costs ~65 MB on Jetson); GPU
  facts come from `edgeserve-worker --probe`.

Ollama (runner processes) and vLLM (engine-core process) separate their engines the same way.

## Startup

The HTTP port opens immediately. `/health` answers at once; `/health/ready` returns 503 with the
current step until the server can take requests:

1. **probe** — GPU facts from the worker probe; host facts (Jetson model, L4T/JetPack, power mode,
   clock caps, power sensors) from sysfs and `nvpmodel`.
2. **model** — validate the folder, parse the ONNX graph for the cost model, compare the export's
   Edge-LLM version with the runtime's, and check the GPU has kernels for the model's quantization.
3. **memory** — estimate weights + KV cache + activations; refuse when it clearly cannot fit,
   warn when it is tight.
4. **engine** — look up the engine cache (see below) or build the engine (`building_engine`).
5. **worker** — load the engine, capture decode CUDA graphs, run a short warm-up (excluded from
   metrics), then `ready`.

## Engine cache

TensorRT engines only work with the exact TensorRT build, GPU architecture and plugin build that
produced them. The cache key is a SHA-256 over:

| Component | Source |
|---|---|
| model content digest | every file except those the build never reads (`generation_config.json`, `*.md`, `LICENSE*`, `NOTICE*`, hidden files and folders); files ≤ 1 MiB are hashed in full, larger ones contribute their size + three 64 KiB samples |
| build limits | max batch size, max input tokens, KV capacity (= input + output + 1, rounded up to 128) |
| Edge-LLM build | pinned commit, patch hashes, plugin library SHA-256, CuTe DSL kernel pack metadata, CUDA version |
| GPU | compute capability, GPU name, TensorRT version |

Builds run `llm_build` in a separate, low-priority process (first OOM victim) under a file lock,
into `.building-*`, and are renamed into place only when complete, with a `manifest.json`
recording the key, profile, TensorRT version and the build's peak memory. A cached engine is
reused only if its manifest is complete, every file is present and the TensorRT version in the
engine's own header matches. The model's weight files (`*.safetensors`: embedding, vocabulary map,
external INT4 weights) are hard-linked from the model folder instead of copied (symlinked when the
cache is on another file system). The three most recently used engines per model are kept.

## Request lifecycle

1. **Parse and identify.** Headers within 30 s of connecting (or of the previous response), body
   ≤ 4 MiB within 60 s; a client that stops reading for 60 s or keeps a closing connection open for
   60 s is dropped; 32 connections per client address, 128 in all (beyond that uvicorn answers every
   request with a plain 503). JSON, validated against the OpenAI schema subset
   (unknown fields → 400). Client ID from `X-Request-Id` or `request_id` (else `req_<hex>`).
2. **Admit.** Ready? Memory above the floor (256 MiB)? Same ID already in flight (409)? Queue space
   (16 waiting, else 503 + `Retry-After`)? Then the worker counts prompt tokens with the model's own
   chat template (answered even while a generation runs) and the request is checked against the
   engine: prompt ≤ max input, prompt + max_tokens + 1 ≤ KV capacity (else 400
   `context_length_exceeded`). Nothing is ever truncated silently.
3. **Queue and dispatch.** FIFO. With `--max-batch-size N > 1` the oldest request is batched with
   up to N−1 later ones of the same kind (chat or formatted prompt) that have identical sampling
   settings and output limits. A sampling request without a seed gets a random one (Edge-LLM would
   otherwise draw the same random number at every step), recorded in its record.
4. **Generate.** Streaming text is forwarded as Edge-LLM publishes it, in whole UTF-8 characters
   (Edge-LLM's stop-string look-ahead holds back bytes and can cut a character in two); the final
   chunk adds whatever the stream did not show, so the streamed text always equals the final text.
   The chat role chunk goes out with the first streamed text. A disconnecting client (streaming or not)
   cancels its sequence at the next decode step.
5. **Account.** The record is built from the worker's step timestamps, GPU step times and memory
   samples, the model's cost model, the hardware peaks and the energy samples (docs/METRICS.md),
   written to the output folder, exported to Prometheus, and summarised in one JSON log line
   (no prompt or output text in logs).

Shutdown (SIGTERM / Ctrl-C): the listening socket closes, so new connections are refused; requests
on connections that are already open get 503 `server_shutting_down`; in-flight ones get up to 60 s
to finish; anything still running after that is cancelled (record status `cancelled`; the
worker gets 10 s to stop it), the last responses get 5 s to go out, then the worker is stopped.
systemd allows 90 s (`TimeoutStopSec`); give Docker the same with `--stop-timeout 90`.

## Worker protocol

One JSON object per line. Timestamps are `CLOCK_MONOTONIC` nanoseconds (the front-end uses the
same clock).

Commands (stdin):

| op | fields | notes |
|---|---|---|
| `generate` | `id`, `requests: [item]`, `sampling: {temperature, top_p, top_k, max_new_tokens}`, `stream` | item = `{"messages": [{role, content}], "stop"?, "seed"?}` or `{"prompt": str, "stop"?, "seed"?}`; up to the engine's max batch size; one generation runs at a time; `top_k` 0 with `top_p` 1 (no limit) reaches Edge-LLM, which needs one of them, as the largest `top_p` below 1 |
| `count` | `id`, `requests: [item]` | prompt tokens after the chat template; runs concurrently with a generation |
| `cancel` | `id`, `req`? | cancel one sequence or the whole batch at the next step |
| `shutdown` | — | cancel what is running and exit; closing stdin does the same |

Events (stdout):

| ev | fields |
|---|---|
| `probe` | (`--probe` mode) `gpu_name, sm, sm_count, integrated, total_memory_bytes, free_memory_bytes, memory_bus_width_bits, l2_cache_bytes, clock_rate_khz, memory_clock_rate_khz, cuda_runtime, cuda_driver, tensorrt, tensorrt_runtime, edgellm` |
| `ready` | `engine_dir, max_batch_size, cuda_graph, load_ms, warmup_ms, device_free_before_load_bytes, device_free_after_load_bytes, device{...probe fields}` |
| `count` | `id, prompt_tokens: [int]` |
| `text` | `id, req, t_ns, text` (stream deltas) |
| `done` | `id, t_received_ns, t_start_ns, t_end_ns, steps{t_ns[], gpu_ms[], free_bytes[], active[]}, results[{req, finish_reason, prompt_tokens, text, token_ids[], pieces[]}]` |
| `error` | `id` (null when the error belongs to no command: load errors, unparsable or oversized lines, bad `cancel` fields), `code, message, fatal` — fatal errors are followed by exit |

`steps` has one entry per batch step: step 0 is the prefill (it produces the first token), step i
produces token i of every sequence still running. `t_ns` is taken in Edge-LLM's per-token callback,
which runs right after each step's stream synchronize; `gpu_ms` comes from Edge-LLM's CUDA-event
stage timers; `free_bytes` is `cudaMemGetInfo` free memory (system-wide on Jetson). `token_ids`
include a terminating end-of-sequence token; `text` excludes it and any stop string (a character cut by
`max_new_tokens` shows as U+FFFD, as in Hugging Face decoding). `text` events carry whole UTF-8
characters.
Error codes: `bad_request`, `input_too_long`, `generation_failed`, `cuda_error` (fatal),
`load_failed` (fatal), `no_device` (probe).

## Edge-LLM patches

edgeserve pins one Edge-LLM commit (`edgeserve/native/edgellm/COMMIT`) and applies five small
patches to it, mainly so that it builds and runs on JetPack 6 (CUDA 12.6, TensorRT 10.3). They are
listed, with the reason for each, in [PLATFORMS.md](PLATFORMS.md#edge-llm-patches).
