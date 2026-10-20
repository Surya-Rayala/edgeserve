# edgeserve

Run LLMs on NVIDIA Jetson and NVIDIA GPUs behind an OpenAI-compatible API, using
[NVIDIA TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM).

- OpenAI API (chat, completions, streaming): works with the OpenAI SDKs and curl.
- Serves Hugging Face models and your own fine-tuned, pruned or quantized checkpoints, in FP16/BF16,
  INT4 (AWQ, GPTQ), INT8, FP8 or NVFP4, whichever your GPU supports.
- Converts the model to a TensorRT engine for your GPU on first start, and caches it.
- Keeps your request IDs and records metrics per request and per token (latency, tokens/s, energy).
- Guards small devices against running out of memory.
- `edgeserve setup --network` adds an API key and HTTPS for other machines.

**Steps:** [1. Install](#1-install) · [2. Build the GPU runtime](#2-build-the-gpu-runtime) ·
[3. Export a model](#3-export-a-model) · [4. Serve](#4-serve) ([free memory](#free-memory-on-a-jetson)) ·
[5. Send requests](#5-send-requests) ·
[6. Access from other machines](#6-access-from-other-machines). Also: [request format](#request-format),
[your own models](#your-own-models), [batch jobs](#batch-jobs), [service](#run-as-a-service),
[uninstall](#uninstall), [commands](#command-reference), [options and settings](#options-and-settings),
[troubleshooting](#troubleshooting), [documentation](#documentation).

In the commands below, replace each `<placeholder>` with your own value.

## Requirements

| Device | Software | Status |
|---|---|---|
| Jetson Orin Nano / Orin NX / AGX Orin | JetPack 6.2.x or 7.2.x | tested (Orin Nano 8 GB, JetPack 6.2.1) |
| Jetson Thor | JetPack 7.x | expected to work |
| x86-64 PC with an NVIDIA RTX 30/40/50, A100, A10, L4, L40S or B200 | Ubuntu 22.04/24.04, CUDA 12 or 13, TensorRT 10 | expected to work |

Python 3.10 or newer and build tools are needed too. On a Jetson:

```bash
sudo apt install nvidia-jetpack git cmake build-essential python3-venv
```

On a PC, install the NVIDIA driver, CUDA and TensorRT 10 with its development packages, then
`sudo apt install git cmake build-essential python3-venv`. `edgeserve setup` checks all of this
and prints what is missing. edgeserve 0.1 is built on TensorRT Edge-LLM 0.11 (commit
`0f52705723ab`). Details per platform:
[docs/PLATFORMS.md](https://github.com/Surya-Rayala/edgeserve/blob/main/docs/PLATFORMS.md).

**Disk space.** edgeserve's tools take about 7 GB, the same whichever models you use: 1.7 GB for
the GPU runtime (step 2, which needs 3 GB free while it builds) and 5.5 GB for the exporter
(step 3). Each model needs its own space on top, which grows with its size: its download, its
exported folder and its engines. A 0.5B model takes up to 2 GB in INT4 and 3.5 GB in FP16
([details](https://github.com/Surya-Rayala/edgeserve/blob/main/docs/MODELS.md#disk-space)).

## 1. Install

With pip:

```bash
python3 -m venv <venv-dir>
source <venv-dir>/bin/activate
pip install --upgrade pip
pip install edgeserve
```

Or from the repository:

```bash
git clone https://github.com/Surya-Rayala/edgeserve.git
cd edgeserve
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

`<venv-dir>` is any new folder for the Python environment, for example `~/edgeserve-venv`. Both ways
give the same `edgeserve` command, and both keep their files in the same places: settings in
`~/.config/edgeserve/`, the GPU runtime and the model exporter in `~/.local/share/edgeserve/`, engines
and build files in `~/.cache/edgeserve/`. In a new terminal, run the `source …/bin/activate` line again.

## 2. Build the GPU runtime

```bash
edgeserve setup
```

Run it once per machine, as your normal user (not with sudo). It downloads TensorRT Edge-LLM,
builds it for your GPU and tests it. This takes about 30 minutes on a Jetson Orin Nano (less on a
PC) and needs internet access and 3 GB of free disk. Run it again after upgrading edgeserve, JetPack,
CUDA or TensorRT: it rebuilds only what changed.

## 3. Export a model

edgeserve serves models in TensorRT Edge-LLM's format: a folder with the model converted to ONNX,
its weights and its tokenizer. `edgeserve export` makes this folder from a Hugging Face checkpoint:

```bash
edgeserve export <checkpoint> <model-dir>
```

| Argument | What it is |
|---|---|
| `<checkpoint>` | a Hugging Face model ID, such as `Qwen/Qwen2.5-0.5B-Instruct-AWQ`, or the folder of your own checkpoint ([your own models](#your-own-models)) |
| `<model-dir>` | a new folder for the exported model, such as `~/models/qwen2.5-0.5b-instruct-awq`; its name becomes the model's name |

The first export installs TensorRT Edge-LLM's exporter (a few minutes, about 5.5 GB of disk, once
for all models). An export takes a few minutes; it runs on the CPU, so it works on the Jetson itself
or on any Linux PC with edgeserve and git installed (no `edgeserve setup` needed there; copy
`<model-dir>` to the Jetson afterwards). It also copies the model's default sampling settings
(`generation_config.json`). For a gated model such as Llama, accept its license on huggingface.co
and pass your access token: `HF_TOKEN=<your token> edgeserve export <checkpoint> <model-dir>`. A
folder that TensorRT Edge-LLM 0.11 already exported needs no export: serve it as it is. GGUF files,
ONNX models from other tools and ready-made engines cannot be served ([what you can bring](#your-own-models)).

**Which models.** Text models of the Llama 3.x, Qwen2/Qwen2.5, Qwen3 and HunYuan (dense) families;
Qwen3 MoE, Qwen3.5 and Nemotron-H should work but are untested
([families and tested models](https://github.com/Surya-Rayala/edgeserve/blob/main/docs/MODELS.md#supported-families)).
Choose a precision your GPU supports:

| Weights | Hugging Face name looks like | Jetson Orin | RTX 30/40, A100, A10, L4, L40S | Thor, RTX 50, B200 |
|---|---|---|---|---|
| FP16 / BF16 (the original weights) | `Qwen/Qwen2.5-0.5B-Instruct` | ✅ | ✅ | ✅ |
| INT4 (AWQ or GPTQ) | `…-Instruct-AWQ`, `…-GPTQ-Int4` | ✅ | ✅ | ✅ |
| INT8 (SmoothQuant) | quantize it yourself ([how](https://github.com/Surya-Rayala/edgeserve/blob/main/docs/EXPORT.md#quantizing-and-other-tools-by-hand)) | ✅ | ✅ | ✅ |
| FP8 | `nvidia/…-FP8` | ❌ | RTX 40, L4, L40S only | ✅ |
| NVFP4 | `nvidia/…-NVFP4` | ❌ | ❌ | ✅ |

Memory decides the model size. With `edgeserve export`, an INT4 model builds its engine in about the
memory it serves in; an FP16 model needs two to three times that. On an 8 GB Orin Nano, FP16 models
up to about 0.6B parameters and INT4 models of 0.5B to 4B were tested, and a pruned 6B INT4 model
with a reduced output vocabulary fits too. Boards and GPUs with more memory run correspondingly larger
models, in any precision they support. More: every argument, what the export does, exporter options
and errors in [docs/EXPORT.md](https://github.com/Surya-Rayala/edgeserve/blob/main/docs/EXPORT.md);
supported models and memory in [docs/MODELS.md](https://github.com/Surya-Rayala/edgeserve/blob/main/docs/MODELS.md).

## 4. Serve

On a Jetson, [free memory](#free-memory-on-a-jetson) first: the model shares the board's memory with
everything else running.

```bash
edgeserve serve --model <model-dir>
```

The first start converts the model to a TensorRT engine for your GPU (a minute to a few minutes)
and caches it; later starts take seconds. The server listens on `http://127.0.0.1:8000`, which
only this machine can reach. Stop it with Ctrl-C.

To test it, run `edgeserve check` in a second terminal. To serve the model without typing its path
each time, run `edgeserve setup --model <model-dir>` once; after that, `edgeserve serve` is enough.

## Free memory on a Jetson

On Jetson the GPU has no memory of its own: the model, its engine build and everything else
running share the board's RAM. Before serving (and especially before the first start, which
builds the engine), free what you do not need:

```bash
free -h                                           # "available" is what a model can use
ps -eo rss,comm --sort=-rss | head                # the programs that use the most memory (in KB)
sudo systemctl isolate multi-user.target          # stop the desktop until the next reboot (about 0.8 GB)
sudo systemctl set-default multi-user.target      # keep it stopped after reboots (graphical.target undoes it)
docker ps                                         # stop containers you do not need: docker stop <name>
edgeserve info --model <model-dir>                # compare memory_estimate_gib.total with memory_available_gib
```

Also close IDE remote sessions (VS Code or Cursor servers), notebooks and other GPU programs on
the board: each can hold 0.5–1 GB. Serve one model at a time. If an engine build or load still
fails while `free -h` shows enough available memory, release the file caches first:
`sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`. Sizes per model, and how to make a
model fit: [docs/MODELS.md](https://github.com/Surya-Rayala/edgeserve/blob/main/docs/MODELS.md#memory);
power modes and swap: [docs/PLATFORMS.md](https://github.com/Surya-Rayala/edgeserve/blob/main/docs/PLATFORMS.md#jetson).

## 5. Send requests

With curl:

```bash
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -H "X-Request-Id: demo-001" \
  -d '{"messages": [{"role": "user", "content": "What is a GPU? Answer in one sentence."}], "max_tokens": 100}'
```

With Python (`pip install openai`):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")  # no key needed until step 6
model = client.models.list().data[0].id                                  # the served model's name
reply = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "What is a GPU? Answer in one sentence."}],
    max_tokens=100,
    extra_headers={"X-Request-Id": "demo-002"},
)
print(reply.choices[0].message.content)
print(reply.model_extra["metrics"]["decode_tokens_per_s"], "tokens/s")
```

### Request format

A chat request (`POST /v1/chat/completions`) is a JSON object; only `messages` is required:

```json
{
  "messages": [
    {"role": "system", "content": "Summarize the text in two sentences."},
    {"role": "user", "content": "<the text>"}
  ],
  "max_tokens": 160,
  "temperature": 0,
  "request_id": "meeting-0001",
  "metadata": {"source": "room-12"}
}
```

| Field | Meaning | Default |
|---|---|---|
| `messages` | the conversation, oldest first. `role`: `system` (instructions), `user`, `assistant` (an earlier answer) or `developer` (same as `system`); `content`: the text. Up to 64 messages of up to 128 KB each | required |
| `max_tokens` | the longest answer, in tokens (`max_completion_tokens` means the same) | `--max-new-tokens` (the model's `generation_config.json`, else 512) |
| `temperature` | randomness, 0 to 2. 0 always picks the most likely next token: the same answer every time | the model's `generation_config.json`, else 0 |
| `top_p`, `top_k` | when sampling, choose only among the most likely tokens: those that make up `top_p` (above 0, up to 1) of the probability, or the `top_k` (up to 1024) most likely | the model's `generation_config.json`, else no limit |
| `seed` | a number that makes sampling repeatable | random, saved in the record |
| `stop` | a string, or up to 4, that ends the answer | none |
| `stream` | `true`: send the answer in pieces as it is generated; add `"stream_options": {"include_usage": true}` for a last piece with the token counts and metrics | `false` |
| `request_id` | your ID for this request (the same as the `X-Request-Id` header) | `req_<random hex>` |
| `metadata` | up to 16 string pairs of your own (keys up to 64 characters, values up to 512), saved in the record | none |
| `return_record` | `true`: the response also holds the full record, with per-token timings (not with `stream`) | `false` |
| `model` | the served model's name, as `GET /v1/models` lists it; optional | the served model |
| `n` (only 1), `frequency_penalty` and `presence_penalty` (only 0), `logprobs` (only `false`), `user`, `safety_identifier`, `prompt_cache_key`, a message's `name` | accepted so that every OpenAI client works; they change nothing | — |

A completion request (`POST /v1/completions`) takes `prompt`, one string with the model's chat
template already applied, instead of `messages`. Any other field is refused with HTTP 400, never
ignored.

**Request IDs.** Send your ID in the `X-Request-Id` header or the `request_id` field: 1 to 512
printable ASCII characters (with letters, digits, `.`, `_` and `-`, the record's file name is the ID
itself). Without one, the server makes one (`req_…`). The ID comes back in the response's
`request_id` and `id` (`chatcmpl-<id>`) and in its `X-Request-Id` header. With
`edgeserve serve --output-dir <output-dir>`, every request is saved as
`<output-dir>/records/<id>.json`, which `GET /v1/records/<id>` also returns. A request with the ID
of one that is still running gets HTTP 409.

The response is OpenAI's, plus `request_id` and `metrics`:

```json
{"id": "chatcmpl-meeting-0001", "object": "chat.completion", "model": "<model name>",
 "choices": [{"index": 0, "message": {"role": "assistant", "content": "…"}, "finish_reason": "stop"}],
 "usage": {"prompt_tokens": 162, "completion_tokens": 72, "total_tokens": 234},
 "request_id": "meeting-0001", "metrics": {"ttft_ms": 303, "decode_tokens_per_s": 25.9, …}}
```

For streaming, send `"stream": true` (`stream=True` in the SDK). Every field, error code and
metric: [docs/API.md](https://github.com/Surya-Rayala/edgeserve/blob/main/docs/API.md).

## 6. Access from other machines

```bash
edgeserve setup --network
```

This creates three things in `~/.config/edgeserve/`:

- **An API key.** It is printed once. To see it again:
  `grep EDGESERVE_API_KEY ~/.config/edgeserve/edgeserve.env`.
- **An HTTPS certificate**, and `edgeserve-ca.crt`, the file clients need to trust it.
- **The setting to listen on the network** (all interfaces, port 8000).

Restart `edgeserve serve`. From then on the server speaks only HTTPS and every `/v1/` request needs
the key, on this machine too (`edgeserve check` finds both by itself), so the `http://` examples of
step 5 stop working.

**On each client machine** you need the server's `edgeserve-ca.crt` and the key. Below,
`<server-ip>` is the server's IP address (`edgeserve setup --network` prints it) and `<user>` is your
user name on the server.

Linux and macOS (Terminal):

```bash
scp <user>@<server-ip>:.config/edgeserve/edgeserve-ca.crt .
export EDGESERVE_API_KEY=<the key>
curl --cacert edgeserve-ca.crt -H "Authorization: Bearer $EDGESERVE_API_KEY" https://<server-ip>:8000/v1/models
curl --cacert edgeserve-ca.crt https://<server-ip>:8000/v1/chat/completions \
  -H "Authorization: Bearer $EDGESERVE_API_KEY" -H "Content-Type: application/json" -H "X-Request-Id: client-001" \
  -d '{"messages": [{"role": "user", "content": "What is a GPU? Answer in one sentence."}], "max_tokens": 100}'
```

Windows (PowerShell; `scp` and `curl.exe` come with Windows 10 and 11):

```powershell
scp <user>@<server-ip>:.config/edgeserve/edgeserve-ca.crt .
$env:EDGESERVE_API_KEY = "<the key>"
curl.exe --ssl-no-revoke --cacert edgeserve-ca.crt -H "Authorization: Bearer $env:EDGESERVE_API_KEY" https://<server-ip>:8000/v1/models
Set-Content -Encoding ascii request.json '{"messages": [{"role": "user", "content": "What is a GPU? Answer in one sentence."}], "max_tokens": 100}'
curl.exe --ssl-no-revoke --cacert edgeserve-ca.crt https://<server-ip>:8000/v1/chat/completions `
  -H "Authorization: Bearer $env:EDGESERVE_API_KEY" -H "Content-Type: application/json" -H "X-Request-Id: client-002" `
  -d "@request.json"
```

Type `curl.exe`, not `curl` (in Windows PowerShell, `curl` is a different command).
`--ssl-no-revoke` is needed because Windows otherwise asks whether the certificate was revoked,
which a private CA cannot answer. The request body goes in a file because PowerShell changes the
quotes inside inline JSON.

Python, on any of them: make a virtual environment with the OpenAI SDK
(`python3 -m venv <client-venv>` and `<client-venv>/bin/pip install openai httpx`; on Windows
`py -m venv <client-venv>` and `<client-venv>\Scripts\pip install openai httpx`), then run this
with its Python:

```python
import os, ssl
import httpx
from openai import OpenAI

client = OpenAI(base_url="https://<server-ip>:8000/v1", api_key=os.environ["EDGESERVE_API_KEY"],
                http_client=httpx.Client(verify=ssl.create_default_context(cafile="edgeserve-ca.crt")))
reply = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[{"role": "user", "content": "What is a GPU? Answer in one sentence."}],
    max_tokens=100,
    extra_headers={"X-Request-Id": "client-003"},
)
print(reply.model_extra["request_id"], reply.choices[0].message.content)
```

Every other request works as in [step 5](#5-send-requests), with `https://<server-ip>:8000`, the
key and the CA file.

- **New key:** delete the `EDGESERVE_API_KEY=` line in `~/.config/edgeserve/edgeserve.env`, run
  `edgeserve setup --network` (it prints the new key) and restart the server. You can also write
  your own key on that line.
- **New IP address or host name:** run `edgeserve setup --network` again and restart the server.
  Clients keep their `edgeserve-ca.crt`.
- **Firewall:** allow only your own network to reach port 8000. `edgeserve setup --network` prints
  the `ufw` commands for your network.
- **This machine only again:** set `EDGESERVE_HOST=127.0.0.1` in the settings file.

## Your own models

Find what you have:

| You have | What to do |
|---|---|
| A Hugging Face checkpoint folder (`config.json` and `*.safetensors`): the original model, or your fine-tuned, pruned or quantized version of it | export it: `edgeserve export <checkpoint-dir> <model-dir>`, then serve `<model-dir>` |
| A LoRA adapter | load its base model in FP16/BF16 (not 4-bit), merge the adapter (PEFT's `merge_and_unload()`), save the model and the tokenizer into one folder with `save_pretrained()`, then export that folder |
| A folder already exported (converted to ONNX) by TensorRT Edge-LLM 0.11: its `config.json` has `"edgellm_version": "0.11.…"` and sits next to `model.onnx` | nothing: serve it as it is, `edgeserve serve --model <model-dir>` (if the export made an `llm` folder, pass that) |
| A folder exported by another TensorRT Edge-LLM version | export the checkpoint again with `edgeserve export`; edgeserve refuses the folder and says so |
| An ONNX model made by another tool (such as Optimum), a GGUF file (llama.cpp, Ollama), an engine built by TensorRT-LLM or another tool, or an INT4 model in another format (bitsandbytes, compressed-tensors, EXL2, HQQ) | cannot be served: export the Hugging Face checkpoint it was made from, or an AWQ or GPTQ version of it |
| A model of another family (such as Mistral, Gemma or Phi), or one that needs images or audio | not supported, except Qwen3.5, whose image part the export leaves out ([supported families](https://github.com/Surya-Rayala/edgeserve/blob/main/docs/MODELS.md#supported-families)) |

**A checkpoint folder** (`<checkpoint-dir>`) holds these files, as `save_pretrained()` writes them:

| File | Needed | What it holds |
|---|---|---|
| `config.json` | yes | the architecture, with `model_type` such as `llama`, `qwen2` or `qwen3` ([supported families](https://github.com/Surya-Rayala/edgeserve/blob/main/docs/MODELS.md#supported-families)) |
| `model.safetensors` (or shards with `model.safetensors.index.json`; `pytorch_model.bin` also works) | yes | the weights |
| `tokenizer.json`, `tokenizer_config.json` | yes | the tokenizer, and the chat template (`chat_template` in `tokenizer_config.json`, or a `chat_template.jinja` file) |
| `generation_config.json` | no | default sampling settings and end-of-sequence tokens; copied into `<model-dir>` |

- **Fine-tuned:** no changes needed; save the model and its tokenizer into one folder.
- **Pruned:** fewer layers, or fewer heads or a smaller hidden size in every layer, need only the
  matching numbers in `config.json`. A different MLP width in each layer needs
  `"intermediate_sizes": [<width of layer 0>, <width of layer 1>, …]` in the checkpoint's `config.json`,
  one entry per layer (multiples of 128 for INT4). An exported `<model-dir>` needs nothing more: the
  widths are then part of its `model.onnx`, and `edgeserve info --model <model-dir>` lists them (`mlp_widths`).
- **Quantized:** INT4 AWQ or GPTQ (group size 128), or FP8, NVFP4, INT8 SmoothQuant or INT4 AWQ made
  with NVIDIA ModelOpt or Edge-LLM's quantizer, for GPUs that run them
  ([INT4 quality](https://github.com/Surya-Rayala/edgeserve/blob/main/docs/EXPORT.md#quality-of-int4-models)).
- **Smaller output vocabulary:** add `-- --reduced-vocab-dir <vocab-dir>` to the export command
  (`<vocab-dir>` is the folder with `vocab_map.safetensors`).

**An exported folder** is served as a whole: copy all of it
([every file](https://github.com/Surya-Rayala/edgeserve/blob/main/docs/EXPORT.md#the-model-folder)). `edgeserve info --model <model-dir>` checks it
without starting a server. A folder made with Edge-LLM's own `tensorrt-edgellm-export` (without
edgeserve) differs in three ways: it has no `generation_config.json` (copy the checkpoint's in, or
requests decode greedily), its INT4 AWQ and asymmetric GPTQ weights are slightly clipped, and unless
it was exported with `--externalize-weights`, its engine build needs more memory. If you have the
checkpoint, export it again with `edgeserve export`
([INT4 quality](https://github.com/Surya-Rayala/edgeserve/blob/main/docs/EXPORT.md#quality-of-int4-models)).

To host your model for your network, starting at every boot:

```bash
edgeserve export <checkpoint-dir> <model-dir>        # once (skip it for a folder that is already exported)
edgeserve setup --network --model <model-dir>        # make it the default model; API key and HTTPS
edgeserve service install                            # serve it now and at every boot
```

Details, quantizing a model yourself and checking a pruned export: [docs/EXPORT.md](https://github.com/Surya-Rayala/edgeserve/blob/main/docs/EXPORT.md#your-own-checkpoints).

## Batch jobs

```bash
edgeserve batch <requests.jsonl> --output-dir <results-dir>
```

`<requests.jsonl>` has one request per line, for example:

```jsonl
{"id": "q1", "messages": [{"role": "user", "content": "What is CUDA?"}], "max_tokens": 200}
{"id": "q2", "prompt": "<a prompt with the chat template already applied>"}
```

The OpenAI Batch format (`custom_id`, `method`, `url`, `body`) works too. Results go into
`<results-dir>/records/<id>.json`, `<results-dir>/results.jsonl` and `<results-dir>/run.json`
(counts, throughput, latency, energy). Running the same command again skips the finished IDs, so
an interrupted batch resumes. For another machine's server, set `EDGESERVE_API_KEY` as in
[step 6](#6-access-from-other-machines) and add `--server https://<server-ip>:8000 --ca-cert edgeserve-ca.crt`.

## Run as a service

```bash
edgeserve setup --model <model-dir>    # the model the service serves (add --network for other machines)
edgeserve service install              # starts the server now and at every boot; asks for your sudo password
journalctl -u edgeserve -f             # its log
edgeserve service remove               # stop and remove it
```

Run these commands as your normal user, not with sudo. The service uses the settings file: after
changing it, restart the service with `sudo systemctl restart edgeserve`.
`edgeserve service install --print` only shows the unit file.

## Uninstall

First [remove the Hugging Face downloads](https://github.com/Surya-Rayala/edgeserve/blob/main/docs/MODELS.md#disk-space)
you no longer need (they are in `~/.cache/huggingface`), then:

```bash
edgeserve service remove     # if you installed the service
rm -rf ~/.config/edgeserve ~/.local/share/edgeserve ~/.cache/edgeserve
```

Finally, delete `<venv-dir>` (or the cloned `edgeserve` folder) and the model folders you no longer
need. After a new install, `edgeserve setup --network` makes a new key and `edgeserve-ca.crt`: give
both to the clients again.

## Command reference

| Command | What it does |
|---|---|
| `edgeserve setup [--network] [--model DIR]` | build the GPU runtime; `--model` sets the default model; `--network` adds an API key, HTTPS and listening on the network |
| `edgeserve export CHECKPOINT MODEL_DIR [-- exporter options]` | export a Hugging Face checkpoint (an ID or a folder) into a model folder |
| `edgeserve serve` | run the server |
| `edgeserve build` | only convert the model to an engine (or check the cached one), then exit |
| `edgeserve info` | settings and where each comes from, the GPU runtime, the hardware; with `--model`, the model, its memory estimate and its cached engine |
| `edgeserve check [--server URL] [--ca-cert FILE]` | test a running server end to end |
| `edgeserve batch INPUT --output-dir DIR [--server URL] [--ca-cert FILE] [--concurrency N]` | run a JSONL file of requests |
| `edgeserve service install [--print]`, `edgeserve service remove` | run the server at boot (systemd) |
| `edgeserve --version` | the edgeserve and TensorRT Edge-LLM versions |

## Options and settings

Each option can also be set as a line `NAME=value` in `~/.config/edgeserve/edgeserve.env` (the file
`edgeserve setup` writes; restart the server after editing it) or as an environment variable. An
option on the command line wins over the environment, which wins over the file. `edgeserve info`
shows where each value came from.

| Option (`serve`; `build` and `info` take the first five) | Setting | Default | Meaning |
|---|---|---|---|
| `--model DIR` | `EDGESERVE_MODEL` | none | the model folder (an absolute path in the file) |
| `--max-input-len N` | `EDGESERVE_MAX_INPUT_LEN` | 3072 | longest prompt in tokens; longer prompts are refused, never cut |
| `--max-new-tokens N` | `EDGESERVE_MAX_NEW_TOKENS` | the model's `generation_config.json`, else 512 | output length when a request sets no `max_tokens` |
| `--max-batch-size N` | `EDGESERVE_MAX_BATCH_SIZE` | 1 | requests generated together; each one needs its own memory |
| `--cache-dir DIR` | `EDGESERVE_CACHE_DIR` | `~/.cache/edgeserve` | where engines are kept |
| `--output-dir DIR` | `EDGESERVE_OUTPUT_DIR` | none | save a record of every request here |
| `--host ADDRESS` | `EDGESERVE_HOST` | 127.0.0.1 | `127.0.0.1`: this machine only; `0.0.0.0`: the network (needs an API key) |
| `--port N` | `EDGESERVE_PORT` | 8000 | port |
| none | `EDGESERVE_API_KEY` | none | the key clients must send as `Authorization: Bearer <key>` (no option, so it never shows in process lists) |
| none | `EDGESERVE_TLS_CERT`, `EDGESERVE_TLS_KEY` | none | HTTPS certificate and private key (PEM files) |
| none | `EDGESERVE_NATIVE_DIR` | none | use another GPU runtime build (development) |

Changing `--max-input-len`, `--max-new-tokens` or `--max-batch-size` builds a new engine (cached
too). `check` and `batch` use this machine's server unless given `--server URL`, and trust a
server's certificate with `--ca-cert FILE`; `batch --concurrency N` sets how many requests are in
flight (default: the server's batch size + 1). Environment only: `EDGESERVE_CONFIG_DIR` (another
settings folder), `EDGESERVE_API_KEY` on a client (the key `check` and `batch` send), and for
`edgeserve setup`, `TRT_PACKAGE_DIR` (a TensorRT tarball folder) and `JOBS` (parallel compile jobs).

## Troubleshooting

| Problem | Fix |
|---|---|
| `edgeserve setup` lists missing packages | install them with the `apt` line it prints, then run it again |
| "the GPU runtime is not built yet" or "… from other sources" | run `edgeserve setup` |
| `/health/ready` says `building_engine` | normal on the first start of a model or new limits |
| "not enough memory", or the engine build "ran out of memory" | [free memory](#free-memory-on-a-jetson) on a Jetson; use an INT4 or a smaller model, lower `--max-input-len`, keep `--max-batch-size 1` |
| "exported with TensorRT Edge-LLM x.y" | export the model again with `edgeserve export` |
| "is a Hugging Face checkpoint, not an exported model" | export it first: `edgeserve export <checkpoint-dir> <model-dir>`, then serve `<model-dir>` |
| "is a GGUF file", "holds GGUF files" or "holds an ONNX model made by another tool" | export the Hugging Face checkpoint the model was made from |
| "has no config.json" | `--model` must be the folder `edgeserve export` made; for `export`, the checkpoint folder must hold `config.json` |
| HTTP 401 | send the key: `Authorization: Bearer <key>` |
| certificate errors | give the client `edgeserve-ca.crt`; use a host name or address the certificate lists, or run `edgeserve setup --network` again after an address change |
| `<host>.local` does not connect | use the IP address; mDNS can announce a Docker bridge address (fix: `deny-interfaces=docker0` in `/etc/avahi/avahi-daemon.conf`) |
| other machines cannot connect | run `edgeserve setup --network` and restart; check the firewall (`sudo ufw status`) |
| answers differ from Hugging Face | the server log lists `generation_config.json` options Edge-LLM does not support (e.g. `repetition_penalty`); for INT4 models see [INT4 quality](https://github.com/Surya-Rayala/edgeserve/blob/main/docs/EXPORT.md#quality-of-int4-models) |
| `edgeserve export` fails | the error table in [docs/EXPORT.md](https://github.com/Surya-Rayala/edgeserve/blob/main/docs/EXPORT.md#errors) |

## Documentation

| Topic | Document |
|---|---|
| Exporting: every argument, what the export does, your own fine-tuned, pruned and quantized checkpoints, INT4 quality, exporter options, errors | [docs/EXPORT.md](https://github.com/Surya-Rayala/edgeserve/blob/main/docs/EXPORT.md) |
| Models: supported families, tested models, precision per GPU, memory, disk space | [docs/MODELS.md](https://github.com/Surya-Rayala/edgeserve/blob/main/docs/MODELS.md) |
| The API: endpoints, request fields, sampling, streaming, errors, request IDs, records | [docs/API.md](https://github.com/Surya-Rayala/edgeserve/blob/main/docs/API.md) |
| Metrics: every per-request and per-token metric and how it is measured | [docs/METRICS.md](https://github.com/Surya-Rayala/edgeserve/blob/main/docs/METRICS.md) |
| Platforms: requirements per device, Jetson settings, measured performance, Edge-LLM patches | [docs/PLATFORMS.md](https://github.com/Surya-Rayala/edgeserve/blob/main/docs/PLATFORMS.md) |
| How it works: processes, GPU runtime, engine cache, request lifecycle, worker protocol | [docs/ARCHITECTURE.md](https://github.com/Surya-Rayala/edgeserve/blob/main/docs/ARCHITECTURE.md) |

## Development

```bash
pip install -e ".[test]"
python -m pytest            # no GPU needed; needs g++ and nlohmann-json3-dev
```

## License

Apache License 2.0 ([LICENSE](https://github.com/Surya-Rayala/edgeserve/blob/main/LICENSE)). edgeserve
downloads and builds NVIDIA TensorRT Edge-LLM (Apache 2.0), which is not part of this package. Models
have their own licenses.
