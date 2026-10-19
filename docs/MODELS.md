# Models

edgeserve serves **text-in, text-out** language models in the format of TensorRT Edge-LLM
**0.11** (commit `0f52705723ab`; `edgeserve --version` prints it): folders made by
`edgeserve export` from a Hugging Face checkpoint, or the `llm/` folder that Edge-LLM 0.11's own
exporter writes. GGUF files, ONNX models from other tools and ready-made engines cannot be served,
and INT4 checkpoints in formats other than AWQ and GPTQ (bitsandbytes, compressed-tensors, EXL2,
HQQ) cannot be exported. This page covers which models work, which precision each GPU runs, and
what fits in memory and on disk. How to export a model, including your own fine-tuned, pruned or
quantized checkpoints: [EXPORT.md](EXPORT.md).

## Supported families

From Edge-LLM 0.11's support list, restricted to text generation:

| Family | `model_type` | Checkpoints (Hugging Face) | edgeserve |
|---|---|---|---|
| Llama 3.x | `llama` | meta-llama/Llama-3.2-1B-Instruct, -3B-Instruct, Llama-3.1-8B-Instruct | supported ([tested](#tested-models)) |
| Qwen2, Qwen2.5 | `qwen2` | Qwen/Qwen2.5-0.5B … 14B (-Instruct), Coder, Math, DeepSeek-R1-Distill-Qwen; every `-Instruct-AWQ` and `-Instruct-GPTQ-Int4` version | supported ([tested](#tested-models)) |
| Qwen3 (dense) | `qwen3` | Qwen/Qwen3-0.6B … 14B, Qwen3-4B-Instruct-2507, Qwen/Qwen3-4B/8B/14B-AWQ | supported ([tested](#tested-models)) |
| HunYuan (dense) | `hunyuan_v1_dense` | tencent/Hy-MT2-1.8B, Hy-MT2-7B, HY-MT1.5-7B | supported by Edge-LLM; not tested here (no INT4 version exists, and FP16 1.8B does not build on 8 GB) |
| Qwen3 MoE | `qwen3_moe` | Qwen/Qwen3-30B-A3B-GPTQ-Int4 | should run, untested (too large for 8 GB); FLOPs and bandwidth metrics are `null` |
| Qwen3.5 / 3.6 (hybrid attention) | `qwen3_5`, `qwen3_5_moe` | Qwen/Qwen3.5-0.8B … 27B (text-only use) | should run, untested; export with `edgeserve export` (export tested with Qwen3.5-0.8B; the vision part is left out); FLOPs and bandwidth metrics are `null` |
| Nemotron-H (hybrid Mamba) | `nemotron_h` | NVIDIA Nemotron-3 Nano | should run, untested; FLOPs and bandwidth metrics are `null` |

Fine-tuned, pruned and quantized versions of Llama, Qwen2/2.5, Qwen3 and HunYuan checkpoints
export the same way ([your own checkpoints](EXPORT.md#your-own-checkpoints)).

Not supported: vision-language models (Qwen2.5-VL, Qwen3-VL, InternVL), Phi-4 multimodal,
Gemma 4, speech recognition and speech generation (Qwen3-ASR, Qwen3-TTS), omni and robot-action
models, and speculative-decoding drafts (EAGLE3, DFlash). Their inputs are images or audio, or
they need a second engine. Families Edge-LLM cannot export (e.g. Mistral, Gemma 2/3, Phi-4 text)
are not supported either.

## Tested models

On a Jetson Orin Nano 8 GB Super (JetPack 6.2.1, 25W mode, desktop off, an IDE session running),
exported with `edgeserve export` and served at the default limits (3072 input + 512 output
tokens), then asked questions with known answers ("What is the capital of France?", "Name three
primary colors.") and decoded greedily.
"Same tokens as the reference" means that the served model's greedy output equals that of PyTorch
computing the same (pruned, dequantized) weights in float32, until the two hit a near-tie.

| Model | Precision | Result |
|---|---|---|
| Llama-3.2-1B-Instruct pruned to per-layer MLP widths 5120–8192 (`intermediate_sizes`) | INT4 GPTQ, asymmetric | ✅ same tokens as the reference; 47–65 tokens/s |
| Qwen2.5-0.5B-Instruct pruned to per-layer MLP widths 3328–4864 | INT4 GPTQ, asymmetric | ✅ same tokens as the reference; 50–54 tokens/s |
| Qwen2.5-0.5B-Instruct pruned to per-layer MLP widths 3328–4864 | FP16 | ✅ same tokens as the reference; 34–42 tokens/s |
| A pruned Llama-3.1 6B fine-tune (MLP widths 4352–14336, 16,384-token output vocabulary; exported before edgeserve export existed) | INT4 GPTQ, asymmetric | ✅ good summaries; 25 tokens/s, MBU 0.67 |
| The same model with its full 128,256-token vocabulary | INT4 GPTQ, asymmetric | ⚠️ needs 5.6 GiB to serve: refused while an IDE session left 4.8 GiB free |
| Qwen/Qwen3-4B-AWQ | INT4 AWQ | ✅ correct answers; 18–25 tokens/s |
| casperhansen/llama-3.2-3b-instruct-awq | INT4 AWQ | ✅ correct answers; 23–30 tokens/s |
| casperhansen/llama-3.2-1b-instruct-awq | INT4 AWQ | ✅ correct answers; 49–62 tokens/s |
| Qwen/Qwen2.5-0.5B-Instruct-AWQ | INT4 AWQ | ✅ correct answers; 54 tokens/s |
| Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4 | INT4 GPTQ, symmetric | ✅ correct answers; 41–53 tokens/s |
| JunHowie/Qwen3-0.6B-GPTQ-Int4 | INT4 GPTQ, symmetric | ✅ 44 tokens/s |
| Qwen/Qwen3-0.6B | FP16 | ✅ correct answers; 44–46 tokens/s |
| Qwen/Qwen2.5-0.5B-Instruct | FP16 | ✅ correct answers; 37 tokens/s |

The pruned test checkpoints were made from the public models by removing the least important MLP
channels of every layer (a different number per layer) and, for INT4, quantizing with
round-to-nearest, without any retraining: their answers are poor, which does not matter for the
comparison.

## Precision by platform

| Precision | How you get it | Jetson Orin | Jetson Thor | Ampere (RTX 30, A100, A10) | Ada (RTX 40, L4, L40S) | Blackwell (RTX 50, B200) |
|---|---|---|---|---|---|---|
| FP16 / BF16 | export the original checkpoint | ✅ | ✅ | ✅ | ✅ | ✅ |
| INT4 weights (AWQ, GPTQ; W4A16) | a pre-quantized checkpoint, or `tensorrt-edgellm-quantize llm … --quantization int4_awq` | ✅ | ✅ | ✅ | ✅ | ✅ |
| INT8 SmoothQuant (W8A8) | `--quantization int8_sq` | ✅ | ✅ | ✅ | ✅ | ✅ |
| FP8 | `--quantization fp8`, or an `nvidia/…-FP8` checkpoint | ❌ | ✅ | ❌ | ✅ | ✅ |
| NVFP4, MXFP8 | `--quantization nvfp4` / `mxfp8` | ❌ | ✅ | ❌ | ❌ | ✅ |

Quantizing a model yourself: [EXPORT.md](EXPORT.md#quantizing-and-other-tools-by-hand). INT4 needs
Edge-LLM's INT4 GEMM kernels. They exist for sm_80, 86, 87, 89, 100, 101, 110, 120 and
121, so not for Hopper (sm_90), Turing or Volta; edgeserve refuses such a model there and says so.

## Memory

On Jetson the GPU and the CPU share one memory. The model has to fit next to the operating system
and everything else running; on a discrete GPU it has to fit in the GPU's memory. edgeserve
estimates the need before it builds or loads anything:

```
weights + KV cache (max batch size × context × bytes per token) + activations
        + runtime (0.3 GiB: CUDA, TensorRT, tokenizer)
```

`edgeserve info --model <model-dir>` prints the estimate. `edgeserve serve` refuses a model that
clearly cannot fit, with the numbers, and warns when it is tight. Converting a model to an engine
(the first start) can need more memory than serving it, because TensorRT holds the weights it
builds with several times. `edgeserve export` keeps the biggest weights outside the engine, so
INT4 models build in about the memory they serve in; FP16 models need two to three times as much
([EXPORT.md](EXPORT.md#what-it-does)). A failed build says that it ran out of memory, and the
server log shows its peak.

Measured on an 8 GB Jetson Orin Nano (all but the 6B model exported with `edgeserve export`):

| Model | Serving needs, estimated (GiB) | Engine build peak (GiB) |
|---|---|---|
| Qwen2.5-0.5B-Instruct, INT4 AWQ | 1.1 | 1.6 |
| Llama-3.2-1B-Instruct, INT4 AWQ | 2.1 | 2.4 |
| Llama-3.2-3B-Instruct, INT4 AWQ | 3.9 | 3.1 |
| Qwen3-4B, INT4 AWQ | 4.3 | 3.4 |
| Qwen3-0.6B, FP16 | 2.2 | 4.5 |
| A pruned Llama-3.1 6B, INT4 GPTQ, 16k-token output vocabulary | 4.7 | 4.0 |

On an 8 GB board, use INT4 models beyond about 0.6B parameters. Large vocabularies cost memory
twice, in the embedding table and the FP16 output layer. A reduced output vocabulary
([EXPORT.md](EXPORT.md#quantizing-and-other-tools-by-hand)) shrinks only the output layer; the
embedding table keeps every token. The KV cache grows with `--max-input-len`, `--max-new-tokens`
and `--max-batch-size`, so lower these to fit a larger model.

## Disk space

edgeserve's tools take the same space whichever models you use:

| What | Where | Size |
|---|---|---|
| GPU runtime build files | `~/.cache/edgeserve/native-build` | 1.5 GB (`edgeserve setup` needs 3 GB free while it builds). Kept so that a rebuild after an upgrade takes minutes; safe to delete |
| GPU runtime | `~/.local/share/edgeserve/native` | 0.1 GB per version; the 2 newest are kept |
| Exporter | `~/.local/share/edgeserve/exporter` | 5.5 GB, mostly PyTorch and the CUDA libraries it brings |

Each model needs its own space on top:

| What | Where | Size |
|---|---|---|
| Its download | `~/.cache/huggingface` (or `HF_HOME`); none for a local checkpoint folder | the checkpoint's size; not needed after the export |
| The exported model | `<model-dir>` | about the checkpoint's size, more when the checkpoint shares one table for its input and output embeddings (the export stores both); up to about twice the final size while exporting |
| Its engines | `~/.cache/edgeserve/engines` (`<cache-dir>/engines` with `--cache-dir`) | one engine per set of limits (`--max-input-len`, `--max-new-tokens`, `--max-batch-size`), the 3 most recently used per model are kept. Weight files kept outside the engine are linked from `<model-dir>`, not copied, so INT4 engines are small; an FP16 engine holds most of the weights |

Measured on a Jetson Orin Nano:

| Model | Download | While exporting | Model folder | Engine |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct, INT4 AWQ | 0.73 GB | 1.2 GB | 0.74 GB | 0.06 GB |
| Qwen2.5-0.5B-Instruct, FP16 | 0.99 GB | 2.1 GB | 1.3 GB | 1.2 GB |
| A pruned Llama-3.1 6B, INT4 GPTQ, 16k-token output vocabulary | – | – | 3.8 GB | 0.86 GB |

To free space, delete:

- a download, with `~/.local/share/edgeserve/exporter/venv/bin/hf cache rm model/<model ID>`;
- an engine folder you no longer need (it is rebuilt when needed);
- the runtime build files (a later rebuild then starts from scratch, about 30 minutes on a Jetson
  Orin Nano);
- once your models are exported, the exporter folder (the next `edgeserve export` installs it
  again; it holds the `hf` command, so remove downloads first).

Engines hard-link the model folder's weight files: when you delete a model folder, delete its
engine folders too to free that space.

## Switching models

One server serves one model. To switch, run `edgeserve setup --model <model-dir>` (or edit
`EDGESERVE_MODEL` in the settings file) and restart the server, or pass `--model`. The first start
with a model builds its engine. Later starts reuse the cached engine and take seconds. To serve
two models at once, run two servers with different `--model` and `--port` (memory permitting).

## Long inputs

The engine accepts prompts up to `--max-input-len` tokens. Whether the model makes good use of a
long prompt depends on the model: one fine-tuned only on short examples may answer worse beyond
those lengths, even though its base model supports longer contexts. Test your model at the lengths
you will use. The engine's attention kernels are checked at long lengths: `edgeserve setup` runs
`attention-check`, which compares them with a CPU reference at lengths from 256 tokens up to the
default limits, for the attention shape of Llama 3.1 8B
([PLATFORMS.md](PLATFORMS.md#checking-a-platform)).
