# Exporting models

edgeserve serves models in TensorRT Edge-LLM's format. `edgeserve export` makes that format from a
Hugging Face checkpoint: an original model, a fine-tuned or pruned one, or one quantized to INT4.
Which models and precisions work, and what fits in memory: [MODELS.md](MODELS.md).

## The command

```bash
edgeserve export <checkpoint> <model-dir>
edgeserve export <checkpoint> <model-dir> -- <exporter options>
```

| Argument | What it is |
|---|---|
| `<checkpoint>` | a Hugging Face model ID (e.g. `Qwen/Qwen2.5-0.5B-Instruct-AWQ`), downloaded automatically, or a local checkpoint folder ([your own checkpoints](#your-own-checkpoints)) |
| `<model-dir>` | the model folder to create. It must not exist yet, or be empty. Its name becomes the model's name in the API |
| `-- <exporter options>` | everything after `--` goes to Edge-LLM's `tensorrt-edgellm-export` unchanged ([exporter options](#exporter-options)) |

Then serve it with `edgeserve serve --model <model-dir>`. The export needs no GPU and no
`edgeserve setup`: it runs on the CPU, on the Jetson itself or on any Linux PC with edgeserve
installed. A model exported on a PC is copied to the Jetson as a whole folder, for example
`rsync -a <model-dir> <user>@<jetson-ip>:<folder>/`.

| Environment variable | Meaning |
|---|---|
| `HF_TOKEN` | your Hugging Face access token, for gated models such as Llama (accept the license on huggingface.co first) |
| `HF_HOME` | where Hugging Face downloads are kept (default `~/.cache/huggingface`) |
| `EDGELLM_FOLD_ZERO_POINTS=1` | fold INT4 zero points the way stock Edge-LLM does, for comparisons ([INT4 quality](#quality-of-int4-models)) |

## What it does

1. **Installs the exporter** on the first run, into `~/.local/share/edgeserve/exporter/` (a few
   minutes, about 5 GB): TensorRT Edge-LLM at edgeserve's commit (`0f52705723ab`) with edgeserve's
   two exporter patches, in its own Python environment with PyTorch, Transformers and ONNX. Later
   exports reuse it; an edgeserve upgrade that changes the commit or the patches reinstalls it.
   The installation log is `~/.local/share/edgeserve/exporter/install.log`.
2. **Reads the checkpoint's `config.json`** (from the folder, or from Hugging Face) and chooses the
   exporter options:

   | Checkpoint | Options edgeserve passes | Why |
   |---|---|---|
   | INT4 AWQ or GPTQ | `--externalize-weights int4_ffn lm_head` | keeps the INT4 feed-forward weights and the FP16 output layer out of the TensorRT engine |
   | INT4 MoE (AWQ or GPTQ) | `--externalize-weights int4_ffn int4_moe lm_head` | the same for the expert weights |
   | FP16 / BF16 | `--externalize-weights lm_head` | keeps the output layer out of the engine |
   | INT4 GPTQ with a quantized `lm_head`, FP8, NVFP4 | as above without `lm_head`, or nothing | Edge-LLM can keep only an unquantized `lm_head` outside |

   Weights outside the engine are loaded by the runtime as they are, so decoding is as fast as
   before. But the engine build needs much less memory: TensorRT holds the weights it builds with
   several times. On an 8 GB Jetson this decides whether a model builds at all (Llama-3.2-1B INT4:
   2.4 GB with these options, more than 5.5 GB without). Giving `--externalize-weights` yourself
   replaces this choice.
3. **Runs the exporter** (a few minutes; its output is shown and kept in
   `~/.cache/edgeserve/exports/<model-dir name>.log`). edgeserve's patches make it read per-layer
   MLP widths of pruned checkpoints and keep INT4 zero points exact (both described below).
4. **Writes the model folder**: the language model only (the visual and audio parts of a
   multimodal checkpoint are left out), without the bytes the exporter leaves unused in
   `model.onnx.data` (often most of that file), plus the checkpoint's `generation_config.json`.
5. **Checks the folder** the way `edgeserve serve` will, and prints the architecture, the
   precision and the memory needed to serve it with the default limits.

## The model folder

| File | Content |
|---|---|
| `config.json` | Edge-LLM's model config: architecture, layer sizes, `edgellm_version`, end-of-sequence tokens |
| `model.onnx`, `model.onnx.data` | the network and the weights that stay in the engine |
| `external_int4_ffn_weights.safetensors`, `external_lm_head_weight.safetensors` | the weights kept outside the engine |
| `embedding.safetensors` | the input embedding table |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | the tokenizer and the chat template that turns `messages` into a prompt |
| `vocab_map.safetensors` | with a reduced output vocabulary: which tokens the model can output |
| `generation_config.json` | the checkpoint's default sampling settings |

`edgeserve info --model <model-dir>` checks a folder without starting a server: architecture,
precision, each layer's MLP width (`mlp_widths`, INT4 models), the memory estimate, and whether
this GPU can run it. Exports record the Edge-LLM version that made them; edgeserve accepts 0.11.x
exports and refuses others.

## Your own checkpoints

Fine-tuned, pruned and quantized checkpoints of the supported families (Llama 3.x, Qwen2/2.5,
Qwen3, HunYuan dense) export like any other. `<checkpoint>` is then a folder in the layout
`save_pretrained()` writes:

| File | Needed | What it holds |
|---|---|---|
| `config.json` | yes | `model_type` (`llama`, `qwen2`, `qwen3` or `hunyuan_v1_dense`), the layer sizes, and for quantized models `quantization_config` |
| `model.safetensors`, or shards with `model.safetensors.index.json` | yes | the weights, with the standard names (`model.layers.<i>.mlp.gate_proj.weight`, …); `pytorch_model.bin` also works |
| `tokenizer.json`, `tokenizer_config.json` | yes | the tokenizer; the chat template as `chat_template` in `tokenizer_config.json` or as `chat_template.jinja` |
| `generation_config.json` | no | default sampling settings and end-of-sequence tokens |

**Fine-tuned.** Nothing special. Merge LoRA adapters into the weights first, e.g. with PEFT's
`merge_and_unload()` followed by `save_pretrained()`.

**Pruned.** Removing whole layers, or the same number of heads or hidden channels from every
layer, only needs the matching numbers in `config.json` (`num_hidden_layers`,
`num_attention_heads`, `num_key_value_heads`, `hidden_size`, `head_dim`). A different MLP width in
each layer needs a list of widths, one per layer, and the largest as `intermediate_size`:

```json
{
  "model_type": "llama",
  "num_hidden_layers": 32,
  "intermediate_size": 14336,
  "intermediate_sizes": [7168, 9472, 8960, 7936, 9472, 9984, 10752, 12288, "…one width per layer…"]
}
```

Each layer's `gate_proj` and `up_proj` weights then have `intermediate_sizes[i]` output rows, and
its `down_proj` weight that many input columns. Stock Edge-LLM sizes every layer from
`intermediate_size` and cannot load such a checkpoint; edgeserve's patch
`0001-per-layer-mlp-widths.patch` reads `intermediate_sizes`. edgeserve reads every layer's size
from the exported graph, so serving, the memory estimate and the FLOPs and bandwidth metrics need
nothing more. For INT4 checkpoints the widths must be multiples of the group size (128). After
exporting, `edgeserve info --model <model-dir>` shows the widths it found (`mlp_widths`): they
must equal `intermediate_sizes`. Pruned test checkpoints served this way give the same greedy
tokens as PyTorch running the same weights ([tested models](MODELS.md#tested-models)).

**Quantized.** INT4 checkpoints in the AutoAWQ format (`"quant_method": "awq"`) or the GPTQ format
(`"quant_method": "gptq"`, `"bits": 4`, with `qweight`, `qzeros`, `scales` and `g_idx` tensors),
group size 128, symmetric or asymmetric. Embeddings and the output layer stay FP16.

**Reduced vocabulary.** Pass the folder that holds `vocab_map.safetensors` (and
`reduced_vocab.json`) to the exporter:
`edgeserve export <checkpoint> <model-dir> -- --reduced-vocab-dir <vocab-dir>`.

## Quality of INT4 models

Edge-LLM's INT4 kernels compute `(code − 8) × scale` and have no zero-point input. Stock
Edge-LLM 0.11 therefore folds each group's zero point into the 4-bit codes, as
`clamp(code − zero + 8, 0, 15)`, which clips the codes of groups whose zero point is far from 8.
AWQ checkpoints and asymmetric GPTQ checkpoints are affected; symmetric GPTQ (zero point 8) is
not. In Qwen/Qwen2.5-0.5B-Instruct-AWQ, 0.6% of the codes are clipped, by 1.35 quantization steps
on average, and that changes answers: asked "What is the capital of France? Answer with one
word.", the folded export answers "one Word." (PyTorch computing the same folded weights answers
the same), while the original model answers "Paris".

edgeserve's patch `0002-exact-int4-zero-points.patch` keeps the codes exact. It uses

```
(code − zero) × scale = (code − 8) × scale + (8 − zero) × scale
```

The INT4 kernel computes the first part from the unchanged codes. A small FP16 matrix multiplies
the activations' mean over each 128-channel group by `(8 − zero) × scale × 128` and adds the second
part. The extra weights are 1/32 of the INT4 weights. Decoding speed was unchanged in our tests
(54 tokens/s for the 0.5B model either way), and the export answers "Paris". Symmetric
checkpoints need no correction and get none. Models exported with stock Edge-LLM keep the
clipping: export them again with `edgeserve export` from their checkpoint.

Quantization itself still changes outputs, so compare a quantized model with its original on your
own prompts before relying on it (for example with `edgeserve batch` against both).

## Exporter options

Options after `--` go to `tensorrt-edgellm-export` unchanged. The useful ones:

| Option | What it does |
|---|---|
| `--reduced-vocab-dir <vocab-dir>` | export with a reduced output vocabulary from `<vocab-dir>/vocab_map.safetensors` |
| `--externalize-weights <kind> …` | keep these weights outside the engine instead of edgeserve's choice; kinds `int4_ffn`, `int4_moe`, `nvfp4_moe`, `lm_head`, `all` |

All of them: `~/.local/share/edgeserve/exporter/venv/bin/tensorrt-edgellm-export --help`.

## Quantizing and other tools by hand

To quantize a model yourself, or to make a vocabulary map, install Edge-LLM's tools at
edgeserve's commit, with edgeserve's exporter patches:

```bash
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git <edgellm-dir>
cd <edgellm-dir>
git checkout 0f52705723ab
git submodule update --init 3rdParty/nlohmannJson
git apply "$(python3 -c 'import edgeserve, pathlib; print(pathlib.Path(edgeserve.__file__).parent)')"/export_patches/*.patch
python3 -m venv <tools-venv>
<tools-venv>/bin/pip install --upgrade pip
<tools-venv>/bin/pip install ".[tools]"
```

Run the `python3 -c` part with the Python that has edgeserve installed. This is Python only (no
CUDA or compiler needed), but quantizing itself needs an x86-64 PC with an NVIDIA GPU:

```bash
<tools-venv>/bin/tensorrt-edgellm-quantize llm --model_dir <checkpoint> --output_dir <quantized-dir> \
    --quantization int4_awq
edgeserve export <quantized-dir> <model-dir>
```

`--quantization` takes `int4_awq`, `int8_sq`, `fp8`, `nvfp4` or `mxfp8` (which GPUs run which:
[MODELS.md](MODELS.md#precision-by-platform)).

**A smaller output vocabulary.** A model whose outputs use few distinct tokens (one language, one
domain) can be exported with a smaller output vocabulary. The final layer then computes logits
only for those tokens: faster decoding and less memory, but tokens outside the vocabulary can never
be generated.

```bash
<tools-venv>/bin/tensorrt-edgellm-reduce-vocab --model_dir <checkpoint> --output_dir <vocab-dir> \
    --reduced_vocab_size 16384 --method input_aware
edgeserve export <checkpoint> <model-dir> -- --reduced-vocab-dir <vocab-dir>
```

The reference script picks tokens from a public news dataset (CNN/DailyMail). For your own task,
build the token list from your own data.

The PyPI package `tensorrt-edgellm==0.11.0` contains the same exporter without edgeserve's patches
(for x86-64, and aarch64 with glibc 2.39 or newer, so not JetPack 6).

## Sampling defaults

edgeserve uses `generation_config.json`'s `do_sample`, `temperature`, `top_p`, `top_k`,
`max_new_tokens` and `eos_token_id`. A request's own values always win. Without the file, requests
that set no sampling parameters decode greedily, with up to 512 new tokens. TensorRT Edge-LLM has
no repetition, presence or frequency penalties, no beam search and no minimum length: when the
file sets one of these, the server log says so at startup, as outputs can then differ from Hugging
Face transformers.

## Errors

| Message | Fix |
|---|---|
| "… already exists and is not empty" | choose a new `<model-dir>`, or delete the old one |
| "… has no config.json, so it is not a Hugging Face checkpoint folder" | pass the folder that holds `config.json` and the weights |
| "cannot read config.json of … from Hugging Face" | check the model name; for a gated model accept its license and set `HF_TOKEN` |
| "installing the exporter failed at …" | check the internet connection, then run the command again; details in `~/.local/share/edgeserve/exporter/install.log` |
| "the export failed (exit …)" | the exporter's error is shown above it and kept in `~/.cache/edgeserve/exports/`; check that the model's family is supported |
| `ModuleNotFoundError: No module named 'modelopt'` | Qwen3.5 exports need Edge-LLM's `[tools]` extra: use the [by hand](#quantizing-and-other-tools-by-hand) install |
| the export is killed (out of memory) | exporting holds the checkpoint in memory: export on a machine with more memory and copy the folder |
