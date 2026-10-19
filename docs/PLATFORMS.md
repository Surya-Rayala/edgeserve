# Platforms

edgeserve runs on the TensorRT Edge-LLM platforms in the table below. `edgeserve setup` detects
the platform: the CPU architecture, the GPU's compute capability, and the CUDA and TensorRT
versions. It then builds the pinned Edge-LLM commit and edgeserve's GPU worker for that platform.
Each model's TensorRT engine is then built on first start, for that exact GPU and TensorRT version.

| Platform | Status | Notes |
|---|---|---|
| Jetson Orin Nano / Orin NX / AGX Orin, **JetPack 6.2.x** (CUDA 12.6, TensorRT 10.3) | ✅ tested on the reference device: Orin Nano 8 GB Super, JetPack 6.2.1 | Edge-LLM 0.11 no longer supports JetPack 6; edgeserve's patches (below) make it work. The CuTe DSL kernel pack is generated on the device during setup (about 5 minutes, needs internet once). |
| Jetson Orin, **JetPack 7.2.x** (CUDA 13.2, TensorRT 10.16) | expected to work: Edge-LLM's official Orin platform | A prebuilt kernel pack is used; the JetPack 6 fixes change nothing there. Moving from JetPack 6 is a reflash, see below. |
| Jetson Thor, JetPack 7.x | expected to work: an official Edge-LLM platform | Selected automatically (`EMBEDDED_TARGET=jetson-thor`). |
| x86-64 Linux with an NVIDIA GPU: Ampere (sm_80/86: A100, A10, RTX 30), Ada (sm_89: L4, L40S, RTX 40), Blackwell (sm_100/120: B200, RTX 50) | expected to work: Edge-LLM's developer platform | Ubuntu 22.04/24.04, CUDA 12.x or 13.x, TensorRT 10.x. Ada GPUs and CUDA 12 hosts generate their kernel pack during setup. |
| Hopper (H100/H200, sm_90) | untested | Edge-LLM has no INT4 GEMM kernels for Hopper, so INT4 models cannot run; edgeserve refuses them with that explanation. FP16 and FP8 exports may work. |
| Turing (T4), Volta (V100) and older GPUs | ❌ | Edge-LLM's attention kernels need Ampere (sm_80) or newer |
| Jetson Xavier, Jetson Nano (JetPack 5 and 4) | ❌ | not supported by TensorRT Edge-LLM |

"Expected to work" means not yet tested with edgeserve, on a platform Edge-LLM supports.
`edgeserve setup` checks every new platform ([Checking a platform](#checking-a-platform)).

## Jetson

Install JetPack 6.2.x or 7.2.x with its CUDA and TensorRT (`sudo apt install nvidia-jetpack`),
plus the build tools: `sudo apt install git cmake build-essential python3-venv`. Then follow the
[README](../README.md): install edgeserve, run `edgeserve setup`, export a model, and serve it.

Recommended settings, most important on an 8 GB Orin Nano:

- **Run without the desktop.** `sudo systemctl isolate multi-user.target` stops it until the next
  reboot, and frees about 0.8 GB. `sudo systemctl set-default multi-user.target` makes it
  permanent (`graphical.target` restores it). Do not run IDE servers on the device in production.
- **Power mode.** On the Orin Nano Super, the default 25W mode already runs the memory at full
  speed (102 GB/s), so decoding is as fast as in MAXN_SUPER. `sudo nvpmodel -m 2` (MAXN_SUPER)
  speeds up only the prefill, by about 10%. Avoid the 15W and 7W modes: they cap the memory at
  68 GB/s. Every record states the power mode it ran in.
- **Swap.** GPU memory cannot be swapped. A swap file on the NVMe still gives the engine build's
  CPU-side peak somewhere to go:
  `sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile`.
  It lasts until the next reboot; to keep it, add the line `/swapfile none swap sw 0 0` to `/etc/fstab`.
- **Security updates (JetPack 6).** L4T R36.4.4 is affected by CVE-2025-33177 (NvMap). JetPack
  6.2.2 and 6.2.3 (an `apt` upgrade to R36.5) fix it and keep CUDA 12.6 and TensorRT 10.3. After
  the upgrade, run `edgeserve setup` again. It rebuilds the runtime if CUDA or TensorRT changed,
  and engines are rebuilt automatically when anything they depend on changed.

### Moving to JetPack 7.2.1

JetPack 7.2.1 is the official Edge-LLM platform for Orin. Moving there is a reflash with
NVIDIA's Jetson ISO installer, and it erases the target disk: back up first, or install onto a
spare NVMe or microSD card. Afterwards, install edgeserve and run `edgeserve setup` again. The
same code builds against CUDA 13.2 and TensorRT 10.16, and each model's engine is built again on
first start. Exported model folders work as they are: keep them in your backup.

## x86-64 Linux with an NVIDIA GPU

Requirements:

- Ubuntu 22.04 or 24.04 and the NVIDIA driver.
- The CUDA toolkit 12.x or 13.x at `/usr/local/cuda-<version>`, with `nvcc` on the PATH (or at
  `/usr/local/cuda/bin/nvcc`).
- TensorRT 10.x with its development files. Either NVIDIA's apt packages (`libnvinfer-dev`,
  `libnvinfer-plugin-dev`, `libnvonnxparsers-dev`), or a TensorRT tarball unpacked anywhere and
  named with `TRT_PACKAGE_DIR`:

  ```bash
  TRT_PACKAGE_DIR=<tensorrt-dir> edgeserve setup
  ```

  The runtime remembers the tarball's library folder, so `edgeserve serve` (and the systemd
  service) find TensorRT without `LD_LIBRARY_PATH`.
- Build tools and Python 3.10 or newer: `sudo apt install git cmake build-essential python3-venv`
  (CMake 3.20 or newer).

On these machines, memory checks use the GPU's free memory, and energy comes from NVML (the
`nvidia-ml-py` package, installed with edgeserve on x86-64). A larger GPU can take a larger
`--max-batch-size`. Each extra sequence costs its own KV cache: the context size × the model's KV
bytes per token; `edgeserve info --model <model-dir>` prints the estimate.

## Measured on the reference device

Jetson Orin Nano 8 GB Super, JetPack 6.2.1, 25W mode, desktop stopped:

| Step | Result |
|---|---|
| `edgeserve setup` (GPU runtime build) | about 30 minutes, once (3 parallel jobs) |
| Attention-kernel check | about a minute |

With a pruned 6B INT4 Llama-3.1 model (16,384-token output vocabulary, 3.8 GB of weights) at the
default limits (3072 input + 512 output tokens):

| Step | Result |
|---|---|
| Engine build | 163–171 s, peak about 4.0 GiB of host memory, 845 MB engine file |
| Load + CUDA graphs + warm-up | about 10 s |
| Memory in use while serving | 3.9–4.2 GiB (weights 3.5 GiB, KV cache 464 MiB, TensorRT activations 348 MiB) |
| Prefill | 660–770 tokens/s for 200–3,000-token prompts (205 tokens: 266 ms; 2,914 tokens: 4.28 s) |
| Decode | 25–26 tokens/s (38–40 ms per token), about 24 tokens/s at a 2.9K-token context |
| Utilisation ([MFU, MBU](METRICS.md#terms)) | prefill MFU 0.44–0.51, decode MBU 0.69–0.73 |
| Power | about 20 W under load, about 5 W idle (module input, `VDD_IN`) |

These are warm-device figures. After idle, the GPU and memory clocks take about 0.8 s to ramp up,
so the first steps of a request run about 25% slower; records show it as a slower first token and
slower tokens, and a lower MBU. For benchmarking, `sudo jetson_clocks` pins the clocks at their
maximum, at the cost of higher idle power. Run `sudo jetson_clocks --store` once before it;
`sudo jetson_clocks --restore` (or a reboot) undoes it.

## Checking a platform

`edgeserve setup` runs `attention-check` after every build. It runs the two attention kernels the
engines use on this GPU, FMHA for the prompt (prefill) and XQA for each new token (decode), and
compares them with a CPU reference at lengths from 256 tokens up to the default limits: prompts up
to 3,072 tokens, decoding with up to 3,711 tokens in the cache, including a test that keys far back
are still attended. It takes about a minute and must end with `all ok`. On the reference device
every error is about 1e-4. The result is recorded in the runtime's `build_info.json` (`self_test`),
which `edgeserve info` shows. To run it again by hand:

```bash
EDGELLM_PLUGIN_PATH=~/.local/share/edgeserve/native/current/libNvInfer_edgellm_plugin.so \
    ~/.local/share/edgeserve/native/current/attention-check
```

By default it checks the attention shape of Llama 3.1 8B (32 query heads, 8 KV heads, head size
128). To check your own model, add its four numbers to the command:
`<q-heads> <kv-heads> <head-dim> <kv-capacity>`; `edgeserve info --model <model-dir>` prints them
as `num_heads`, `num_kv_heads`, `head_dim` and `kv_capacity`. If it fails, please
[report it](https://github.com/Surya-Rayala/edgeserve/issues) with its output.

## Edge-LLM patches

`edgeserve/native/edgellm/COMMIT` pins the Edge-LLM commit. `edgeserve setup` applies
`edgeserve/native/edgellm/patches/*.patch` on top of it (the exporter has two patches of its own:
[ARCHITECTURE.md](ARCHITECTURE.md#exporting)):

| Patch | Why |
|---|---|
| 0001 NVRTC include path | CUDA < 12.9: `cuda_fp16.h` pulls in `vector_types.h` under NVRTC, so the decode-attention kernel failed to compile during engine builds (JetPack 6) |
| 0002 `cudaLibrary_t` compatibility | CUDA < 12.8 lacks `cudaLibrary_t`; lets the RMSNorm, LayerNorm and FP16-MoE CuTe DSL runners build (used by other model families) |
| 0003 propagate `--wrap` | CUDA < 12.8: the cudart shim's link wrap now reaches executables that link Edge-LLM's static libraries |
| 0004 external subdirectory | builds edgeserve's worker and attention check inside the Edge-LLM tree, against its targets |
| 0005 missing `opt` profile | TensorRT 10.3 passes an empty optimum shape to V3 plugins' `configurePlugin`; the plugins now fall back to the maximum shape, which makes engine builds work on JetPack 6 |

0004 adds edgeserve's own build targets on every platform. The others are guarded, so they change
nothing on newer CUDA and TensorRT.
