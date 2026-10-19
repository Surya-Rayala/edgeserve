# API reference

edgeserve implements the OpenAI chat completions, completions and models API. The official OpenAI
SDKs, curl and tools such as NVIDIA AIPerf work unchanged. Interactive documentation is served at
`/docs` (its *Authorize* button takes the API key; the page loads its scripts from a CDN, so the
browser needs internet access).

## Endpoints

| Method | Path | Purpose | API key |
|---|---|---|---|
| POST | `/v1/chat/completions` | `messages` → the model's chat template → generation (streaming or not) | yes |
| POST | `/v1/completions` | a prompt that already has the chat template applied | yes |
| GET | `/v1/models`, `/v1/models/{model}` | the served model, its limits and quantization | yes |
| GET | `/v1/records/{id}` | the stored record of a request ID (needs `--output-dir`) | yes |
| GET | `/health` | liveness: 200 while running, 503 once the server has failed (startup failed, or the worker kept crashing) | no |
| GET | `/health/ready` | readiness: 200 when requests are accepted, else 503 with the current step | no |
| GET | `/metrics` | Prometheus metrics | no |

"API key: yes" applies only when the server has one (`EDGESERVE_API_KEY`). It is sent as
`Authorization: Bearer <key>`. The GET endpoints also answer HEAD.

`GET /v1/models` returns the served model with its limits (they appear once the engine is built;
until then the entry has only the name and `ready: false`):

```json
{"object": "list", "data": [{"id": "qwen2.5-0.5b-instruct-awq", "object": "model", "created": 1792432212,
  "owned_by": "edgeserve", "ready": true, "max_model_len": 3711, "max_input_tokens": 3072, "max_new_tokens": 512,
  "max_batch_size": 1, "quantization": "INT4 groupwise W4A16 (group 128), fp16 lm_head",
  "engine_fingerprint": "ebfa954e92999d79"}]}
```

## Requests

**Chat** (`/v1/chat/completions`):

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is a GPU?"}
  ],
  "max_tokens": 256
}
```

**Formatted prompt** (`/v1/completions`): `{"prompt": "<the prompt, with the chat template applied>",
"max_tokens": 256}`. Special tokens in the text are recognised, and the tokenizer's beginning-of-text
token is added if the model has one and the text does not start with it.

| Field | Values | Default |
|---|---|---|
| `messages` | roles `system`, `developer` (treated as system), `user`, `assistant`. `content` is a string or a list of `{"type": "text", "text": ...}` parts. Up to 64 messages of up to 128 KB each. A message `name` is accepted and ignored | — |
| `prompt` | one string | — |
| `max_tokens` / `max_completion_tokens` | ≥ 1; prompt + `max_tokens` + 1 must fit the engine's context | `--max-new-tokens` (the model's `generation_config.json`, else 512) |
| `temperature`, `top_p`, `top_k` | `temperature` 0–2; 0 is greedy decoding. `top_p` above 0, up to 1. `top_k` ≤ 1024; 0, or the vocabulary size or more, means no limit. When sampling, `top_p` and `top_k` come from the request, else from the model's `generation_config.json` (unless it is greedy, `do_sample: false`), else no limit | the model's `generation_config.json`, else greedy |
| `seed` | integer, for reproducible sampling | a new random seed per sampled request, saved in its record (`request.params.seed`) so the output can be reproduced |
| `stop` | a string or up to 4 strings | none |
| `stream`, `stream_options.include_usage` | Server-Sent Events, with an optional final chunk carrying usage and metrics (`stream_options` only with `stream: true`) | false |
| `metadata` | up to 16 string key/value pairs (keys up to 64 characters, values up to 512), saved in the record | none |
| `request_id` | your ID (the same as the `X-Request-Id` header) | generated |
| `return_record` | true: add the full record (with per-token data) to the response as `record`; not with `stream` | false |
| `model` | optional; if given, it must be the served model's name | — |

**Limits:**

- The prompt, counted after the chat template (as in `usage.prompt_tokens`): at most
  `--max-input-len` tokens (default 3072).
- Prompt + output + 1: at most the engine's context, `--max-input-len` + `--max-new-tokens` + 1
  rounded up to 128 (3712 by default). So a short prompt can ask for more than `--max-new-tokens`.
- The body: at most 4 MiB.

**Not supported**, so rejected with HTTP 400 (never silently ignored): images and audio,
tools and function calling, `response_format`, `logprobs`, `n` > 1, `echo`, penalties other
than 0, and any other field not in the table above (`unsupported_parameter`). A `null` for an
unsupported OpenAI field means "not used" and is accepted (some clients send every field). `user`,
`safety_identifier` and `prompt_cache_key` are accepted and ignored.

**Sampling defaults** come from the model's `generation_config.json`: the server uses its
`do_sample`, `temperature`, `top_p`, `top_k` and `max_new_tokens`, and a request's own values win
(its end-of-sequence tokens are taken in at export). The file's `max_new_tokens` also sizes the
engine's context, and with it the KV cache memory, unless the server is started with
`--max-new-tokens`. Edge-LLM has no repetition, presence or frequency penalties, beam search or
minimum length: when the file sets one, the server log says so at startup, as answers can then
differ from Hugging Face transformers.

## Responses

Responses are the standard OpenAI objects, with two extra fields:

- `request_id`: your ID.
- `metrics`: a summary of the record: `queue_ms`, `prefill_ms`, `ttft_ms`, `decode_ms`, `e2e_ms`,
  `tpot_ms`, `itl_p50_ms`, `itl_p99_ms`, `prefill_tokens_per_s`, `decode_tokens_per_s`,
  `prefill_mfu`, `decode_mbu`, `energy_j`, `energy_per_output_token_mj`, `avg_power_w` and
  `batch_size` (definitions: [METRICS.md](METRICS.md)).

```json
{"id": "chatcmpl-demo-001", "object": "chat.completion", "model": "qwen2.5-0.5b-instruct-awq",
 "choices": [{"index": 0, "message": {"role": "assistant", "content": "…"}, "finish_reason": "stop"}],
 "usage": {"prompt_tokens": 30, "completion_tokens": 25, "total_tokens": 55},
 "request_id": "demo-001", "metrics": {"ttft_ms": 41.2, "tpot_ms": 9.8, "decode_tokens_per_s": 101.6, …}}
```

**Streaming** sends `chat.completion.chunk` events (`text_completion` for `/v1/completions`):

- The first chat chunk carries the assistant role and goes out with the first text, so a client
  that times the first chunk measures the time to first token.
- With `stop` strings, Edge-LLM holds back up to the longest stop string's length minus one byte,
  so the first chunk can come a token or two after `ttft_ms`.
- With `include_usage`, a final chunk carries `usage` and `metrics`. Then comes `data: [DONE]`.
- The streamed text always adds up to the final text, in whole UTF-8 characters.
- An error after the stream has started (the engine failed, the worker crashed, 10 minutes in the
  queue, shutdown) arrives as a `data: {"error": {…}}` event, then `data: [DONE]`. The HTTP status
  is already 200 and there is no `Retry-After`. The OpenAI SDK raises it as an `APIError`.

A client that disconnects cancels its generation at the next token. The request is recorded as
`cancelled` (HTTP 499, code `client_disconnected`).

## Errors

Errors use OpenAI's format, `{"error": {"message", "type", "param", "code"}}`:

| HTTP | code | When |
|---|---|---|
| 400 | `context_length_exceeded` | the prompt is longer than the engine allows, or prompt + `max_tokens` is too long (nothing is truncated) |
| 400 | `unsupported_parameter`, `invalid_value`, `invalid_json`, `invalid_request_id` | bad input |
| 401 | `invalid_api_key` | a missing or wrong API key (with `WWW-Authenticate: Bearer`) |
| 404 | `model_not_found`, `not_found` | another model's name; an unknown record or endpoint |
| 405 | `method_not_allowed` | e.g. GET on a POST endpoint |
| 408 | `request_timeout` | the body did not arrive within 60 s |
| 409 | `duplicate_request_id` | the same ID is still being processed (with `x-should-retry: false`, so the OpenAI SDK does not resend it) |
| 413 | `body_too_large` | a body over 4 MiB |
| 503 | `server_unavailable`, `server_is_overloaded`, `low_memory`, `worker_restarted`, `queue_timeout`, `server_shutting_down` | not ready yet; queue full (16 waiting); memory low; the GPU worker crashed and is restarting; waited 10 minutes in the queue; shutting down. All with `Retry-After`; safe to retry |
| 500 | `engine_error` | generation failed in the engine |

One client address can hold at most 32 connections; more are dropped at once, without a response.
While all 128 connections are taken, every request (`/health` included) gets a plain-text 503.

## Request IDs

Send your ID in the **`X-Request-Id` header** (preferred) or the **`request_id` body field**:

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -H "X-Request-Id: doc-000123" \
  -d '{"messages": [{"role": "user", "content": "Summarize: TensorRT builds optimized GPU engines."}],
       "metadata": {"source": "example"}}'
```

```python
reply = client.chat.completions.create(
    model="qwen2.5-0.5b-instruct-awq",
    messages=[{"role": "user", "content": "Summarize: TensorRT builds optimized GPU engines."}],
    extra_headers={"X-Request-Id": "doc-000123"},        # or extra_body={"request_id": "doc-000123"}
)
reply.id                          # "chatcmpl-doc-000123"
reply.model_extra["request_id"]   # "doc-000123"
reply.model_extra["metrics"]      # {"ttft_ms": ..., "tpot_ms": ..., "decode_tokens_per_s": ..., "energy_j": ...}
```

The ID comes back in three places:

- the `X-Request-Id` response header (error responses repeat a valid one too);
- the response `id`: `chatcmpl-doc-000123`, or `cmpl-…` for completions;
- `"request_id": "doc-000123"`.

Streams carry it in every chunk's `id` and in the header. With `--output-dir`, the full record is
saved as `records/doc-000123.json`. It can be fetched from any machine with
`GET /v1/records/doc-000123`.

| Rule | Behaviour |
|---|---|
| format | 1–512 printable ASCII characters, no leading or trailing space. With up to 200 letters, digits, `.`, `_` and `-` (not a leading `.`), the file name is the ID itself; other IDs are percent-encoded and, if long, shortened with a hash. `GET /v1/records/<id>` finds it either way |
| no ID given | the server creates `req_<hex>` and returns it |
| the header and the body disagree, or there are two `X-Request-Id` headers | 400 |
| the same ID while it is still running | 409 |
| the same ID later | runs again: `records/<id>.json` holds the latest attempt; `results.jsonl` keeps every attempt, each line with its own `id` (the record's `attempt_id`) |

## Records

With `--output-dir DIR`:

```
DIR/records/<id>.json   one file per ID: the request, the output text and token IDs, usage, timestamps,
                        per-request metrics, per-token arrays, the model, engine and system
DIR/results.jsonl       one line per attempt (OpenAI Batch output format)
```

A request refused before it is queued (for example a 400, a 409 or a full queue) gets no record
and no `results.jsonl` line.

A record, abridged:

```json
{"custom_id": "doc-000123", "attempt_id": "att_…", "status": "completed", "http_status": 200,
 "model": {"name": "qwen2.5-0.5b-instruct-awq", "engine_fingerprint": "…", "max_input_tokens": 3072, "kv_capacity_tokens": 3712},
 "system": {"board": "…", "jetpack": "6.2.1", "power_mode": "25W", "tensorrt": "10.3.0.30", …},
 "request": {"messages": […], "params": {"max_tokens": 512, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "seed": …}},
 "output": {"text": "…", "finish_reason": "stop", "token_ids": […]},
 "usage": {"prompt_tokens": 30, "completion_tokens": 25, "total_tokens": 55},
 "timestamps_ns": {"arrival": …, "queued": …, "scheduled": …, "first_token": …, "last_token": …, "finished": …},
 "metrics": {"ttft_ms": …, "tpot_ms": …, "itl_ms": {"p50": …, "p99": …}, "decode_tokens_per_s": …,
             "prefill_mfu": …, "decode_mbu": …, "energy_j": …, "avg_power_w": …, "batch_size": 1, …},
 "steps": {"count": 25, "token_id": […], "token_text": […], "t_ms": […], "itl_ms": […], "gpu_ms": […],
           "context_len": […], "flops": […], "bytes": […], "mfu": […], "mbu": […], "power_w": […],
           "energy_mj": […], "kv_usage": […], "device_free_bytes": […]}}
```

`steps` has one entry per generated token; entry 0 is the prefill (the pass over the prompt). The
record folder is readable only by the user running the server (`records/` 0700, files 0600). Every
field, how it is measured, and the Prometheus metrics: [METRICS.md](METRICS.md).
