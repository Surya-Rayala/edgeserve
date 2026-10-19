#!/usr/bin/env bash
# Build edgeserve's GPU runtime for this machine: the pinned TensorRT Edge-LLM (runtime libraries,
# TensorRT plugin, llm_build), patched for this platform, plus the edgeserve worker and the
# attention-check diagnostic. Safe to re-run: finished steps are skipped.
#
# Normally run by `edgeserve setup`, which copies these sources into the build tree, runs this
# script and installs the result. For development: bash edgeserve/native/build.sh
#
# Supported hosts: Jetson Orin (JetPack 6.2.x or 7.2.x), Jetson Thor (JetPack 7.x),
# x86_64 Linux with an NVIDIA GPU (CUDA 12.x/13.x + TensorRT 10.x).
#
# Environment (all optional):
#   EDGESERVE_SOURCES=dir     these sources (default: the folder of this script)
#   EDGESERVE_BUILD_ROOT=dir  build tree (default: ~/.cache/edgeserve/native-build)
#   JOBS=N                    parallel compile jobs (default: min(nproc/2, GiB available * 2/3))
#   TRT_PACKAGE_DIR=dir       TensorRT install prefix (default: /usr)
set -euo pipefail

SOURCES="$(cd "${EDGESERVE_SOURCES:-$(dirname "${BASH_SOURCE[0]}")}" && pwd)"
BUILD_ROOT="${EDGESERVE_BUILD_ROOT:-$HOME/.cache/edgeserve/native-build}"
COMMIT="$(tr -d '[:space:]' < "$SOURCES/edgellm/COMMIT")"
PATCH_DIR="$SOURCES/edgellm/patches"
SRC="$BUILD_ROOT/edgellm-src"
BLD="$BUILD_ROOT/build"
VENV="$BUILD_ROOT/cutedsl-venv"
TRT_PACKAGE_DIR="${TRT_PACKAGE_DIR:-/usr}"

log() { printf '[edgeserve-build] %s\n' "$*"; }
die() { printf '[edgeserve-build] ERROR: %s\n' "$*" >&2; exit 1; }
for dir in "$SOURCES" "$BUILD_ROOT"; do
  [[ "$dir" != *[[:space:]]* ]] || die "CMake/nvcc builds do not support paths with spaces: $dir"
done

# A failed build must never take the rest of the machine down with it: be the first
# process the kernel OOM killer picks, and yield the CPU to interactive work.
echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true
renice -n 10 $$ > /dev/null 2>&1 || true

# ---------------------------------------------------------------- platform probe
ARCH="$(uname -m)"
[[ "$ARCH" == "x86_64" || "$ARCH" == "aarch64" ]] || die "unsupported CPU architecture: $ARCH"

NVCC="$(command -v nvcc || true)"
[[ -z "$NVCC" && -x /usr/local/cuda/bin/nvcc ]] && NVCC=/usr/local/cuda/bin/nvcc
[[ -n "$NVCC" ]] || die "nvcc not found (install the CUDA toolkit)"
CUDA_VER="$("$NVCC" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
CUDA_MAJOR="${CUDA_VER%%.*}"
[[ -d "/usr/local/cuda-$CUDA_VER" ]] || die "expected CUDA toolkit at /usr/local/cuda-$CUDA_VER"

SM="$(python3 - <<'EOF'
import ctypes, ctypes.util, glob
paths = [ctypes.util.find_library("cudart")] + sorted(glob.glob("/usr/local/cuda/lib64/libcudart.so*"))
for p in filter(None, paths):
    try:
        rt = ctypes.CDLL(p)
        break
    except OSError:
        continue
else:
    raise SystemExit("libcudart not found")
major, minor = ctypes.c_int(), ctypes.c_int()
# cudaDevAttrComputeCapabilityMajor = 75, ...Minor = 76
if rt.cudaDeviceGetAttribute(ctypes.byref(major), 75, 0) or rt.cudaDeviceGetAttribute(ctypes.byref(minor), 76, 0):
    raise SystemExit("no CUDA device visible")
print(f"{major.value}{minor.value}")
EOF
)" || die "could not query the GPU compute capability"

EMBEDDED_TARGET=""
if [[ -f /etc/nv_tegra_release ]]; then
  if [[ "$SM" == "87" ]]; then EMBEDDED_TARGET="jetson-orin"
  elif [[ "$SM" == "110" ]]; then EMBEDDED_TARGET="jetson-thor"
  else die "unsupported Jetson GPU (sm_$SM); Edge-LLM supports Orin (sm_87) and Thor (sm_110)"
  fi
fi

TRT_HDR="$(ls "$TRT_PACKAGE_DIR"/include/NvInferVersion.h "$TRT_PACKAGE_DIR"/include/"$ARCH"-linux-gnu/NvInferVersion.h 2>/dev/null | head -1 || true)"
[[ -n "$TRT_HDR" ]] || die "TensorRT headers not found under $TRT_PACKAGE_DIR (set TRT_PACKAGE_DIR)"
# NV_TENSORRT_MAJOR etc. are numbers up to TensorRT 10.7 and name TRT_*_ENTERPRISE macros from 10.8 on.
TRT_VER="$(awk '$1 == "#define" { value[$2] = $3 }
  END { split("MAJOR MINOR PATCH BUILD", part, " ")
        for (i = 1; i <= 4; i++) { v = value["NV_TENSORRT_" part[i]]; if (v in value) v = value[v]; printf "%s%s", (i > 1 ? "." : ""), v } }' "$TRT_HDR")"
TRT_LIB_DIR=""
for dir in "$TRT_PACKAGE_DIR/lib" "$TRT_PACKAGE_DIR/lib/$ARCH-linux-gnu" "$TRT_PACKAGE_DIR/lib64"; do
  if compgen -G "$dir/libnvinfer.so*" > /dev/null; then TRT_LIB_DIR="$dir"; break; fi
done
[[ -n "$TRT_LIB_DIR" ]] || die "TensorRT libraries (libnvinfer.so) not found under $TRT_PACKAGE_DIR (set TRT_PACKAGE_DIR)"

if [[ -z "${JOBS:-}" ]]; then
  avail_gib=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1048576 ))
  JOBS=$(( $(nproc) / 2 )); mem_jobs=$(( avail_gib * 2 / 3 ))
  (( mem_jobs < JOBS )) && JOBS=$mem_jobs
  (( JOBS < 1 )) && JOBS=1
fi
log "host: $ARCH  gpu: sm_$SM  cuda: $CUDA_VER  tensorrt: $TRT_VER  target: ${EMBEDDED_TARGET:-x86-dgpu}  jobs: $JOBS"

# ---------------------------------------------------------------- source
mkdir -p "$BUILD_ROOT"
if [[ ! -d "$SRC/.git" ]]; then
  rm -rf "$SRC" "$SRC.partial"  # an interrupted clone must not block the next run
  log "cloning TensorRT-Edge-LLM"
  git clone --quiet --filter=blob:none https://github.com/NVIDIA/TensorRT-Edge-LLM.git "$SRC.partial"
  mv "$SRC.partial" "$SRC"
fi
if [[ "$(git -C "$SRC" rev-parse HEAD)" != "$COMMIT" ]]; then
  git -C "$SRC" fetch --quiet origin
  git -C "$SRC" checkout --quiet --force "$COMMIT"  # drops the patches applied to the previous commit
fi
git -C "$SRC" submodule update --quiet --init --recursive

# The patches go onto a clean checkout: unless all of them are applied already, start again from the pinned
# commit (an updated patch cannot be applied on top of its older version).
for patch in "$PATCH_DIR"/*.patch; do
  git -C "$SRC" apply --reverse --check "$patch" 2>/dev/null && continue
  git -C "$SRC" checkout --quiet --force "$COMMIT"
  for each in "$PATCH_DIR"/*.patch; do
    name="$(basename "$each")"
    git -C "$SRC" apply --check "$each" || die "patch does not apply: $name"
    git -C "$SRC" apply "$each"
    log "applied $name"
  done
  break
done

# ---------------------------------------------------------------- CuTe DSL kernel pack
# One pack per (CPU arch, GPU SM, CUDA major). CMake extracts a shipped prebuilt
# tarball when one matches; otherwise generate it here (offline AOT compile, ~5 min).
ARTIFACT_DIR="$SRC/cpp/kernels/cuteDSLArtifact/$ARCH/sm_$SM"
PREBUILT="$SRC/kernelSrcs/cuteDSLPrebuilt/cutedsl_${ARCH}_sm_${SM}_cuda${CUDA_MAJOR}.tar.gz"
pack_matches() {
  [[ -f "$ARTIFACT_DIR/metadata.json" ]] && python3 - "$ARTIFACT_DIR/metadata.json" "$CUDA_MAJOR" <<'EOF'
import json, sys
meta = json.load(open(sys.argv[1]))
raise SystemExit(0 if str(meta.get("cuda_version", "")).split(".")[0] == sys.argv[2] else 1)
EOF
}
if pack_matches; then
  log "CuTe DSL kernel pack present for sm_$SM / CUDA $CUDA_MAJOR"
elif [[ -f "$PREBUILT" ]]; then
  log "using shipped CuTe DSL kernel pack $(basename "$PREBUILT")"
  rm -rf "$ARTIFACT_DIR"
else
  log "generating CuTe DSL kernel pack for sm_$SM / CUDA $CUDA_MAJOR (needs internet once, ~5 min)"
  if [[ ! -x "$VENV/bin/pip" ]]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet --upgrade pip wheel
  fi
  # Every time (a no-op once installed): an install that failed half-way, e.g. offline, is completed.
  # Pinned to the versions the generator was validated with.
  if [[ "$CUDA_MAJOR" == "12" ]]; then
    "$VENV/bin/pip" install --quiet "nvidia-cutlass-dsl==4.7.0" "cupy-cuda12x==12.3.0" "cuda-python==12.9.10" \
      "numpy==1.26.4"
  else
    "$VENV/bin/pip" install --quiet "nvidia-cutlass-dsl[cu13]==4.7.0" "cupy-cuda13x==13.6.0" "cuda-python>=13,<14"
  fi
  (cd "$SRC" && "$VENV/bin/python" kernelSrcs/build_cutedsl.py --gpu_arch "sm_$SM" --kernels ALL \
      --cuda-version "$CUDA_VER" -j "$JOBS" --clean)
fi

# ---------------------------------------------------------------- configure + build
CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release -DTRT_PACKAGE_DIR="$TRT_PACKAGE_DIR"
            -DCUDA_CTK_VERSION="$CUDA_VER" -DENABLE_CUTE_DSL=ALL
            -DEDGELLM_EXTERNAL_SUBDIR="$SOURCES")
if [[ -n "$EMBEDDED_TARGET" ]]; then
  CMAKE_ARGS+=(-DCMAKE_TOOLCHAIN_FILE="$SRC/cmake/aarch64_linux_toolchain.cmake" -DEMBEDDED_TARGET="$EMBEDDED_TARGET")
else
  CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="$SM" -DCUTE_DSL_ARTIFACT_TAG="sm_$SM")
fi
log "configuring"
cmake -S "$SRC" -B "$BLD" "${CMAKE_ARGS[@]}" > "$BUILD_ROOT/configure.log" 2>&1 \
  || { tail -30 "$BUILD_ROOT/configure.log"; die "CMake configure failed (full log: $BUILD_ROOT/configure.log)"; }
log "compiling (this takes a while)"
cmake --build "$BLD" -j "$JOBS" --target NvInfer_edgellm_plugin llm_build llm_inference llm_bench edgeserve-worker attention-check

for f in "$BLD/libNvInfer_edgellm_plugin.so" "$BLD/examples/llm/llm_build" "$BLD/examples/llm/llm_inference" \
         "$BLD/edgeserve-worker" "$BLD/attention-check"; do
  [[ -e "$f" ]] || die "expected build output missing: $f"
done

# Everything an engine depends on; the service folds this into its engine-cache key.
python3 - "$BLD/build_info.json" "$COMMIT" "$PATCH_DIR" "$CUDA_VER" "$TRT_VER" "$SM" "$ARCH" \
    "$EMBEDDED_TARGET" "$BLD/libNvInfer_edgellm_plugin.so" "$ARTIFACT_DIR/metadata.json" "$TRT_LIB_DIR" <<'EOF'
import hashlib, json, pathlib, sys
out, commit, patch_dir, cuda, trt, sm, arch, target, plugin, pack_meta, trt_lib_dir = sys.argv[1:]
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
meta = json.loads(pathlib.Path(pack_meta).read_text())
json.dump({
    "edgellm_commit": commit,
    "patches": {p.name: sha(p) for p in sorted(pathlib.Path(patch_dir).glob("*.patch"))},
    "cuda": cuda, "tensorrt": trt, "sm": int(sm), "arch": arch, "embedded_target": target or None,
    "plugin_sha256": sha(plugin),
    "cute_dsl": {k: meta.get(k) for k in ("groups", "cuda_version", "cutlass_dsl_version")},
    "trt_lib_dir": trt_lib_dir,
}, open(out, "w"), indent=2)
EOF
log "built in $BLD"
