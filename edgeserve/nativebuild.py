"""Build and install edgeserve's GPU runtime (the C++ worker, Edge-LLM's llm_build and TensorRT plugin).

The sources ship inside the package (``edgeserve/native``: the worker, the pinned Edge-LLM commit and
its patches, and ``build.sh``). ``edgeserve setup`` copies them into the build tree
(``~/.cache/edgeserve/native-build``), runs ``build.sh`` there, and installs the few files the
server needs into a version folder under ``~/.local/share/edgeserve/native/``; the ``current`` link
is switched atomically, so a rebuild never replaces binaries a running server is using. The
previous version is kept.

A build is reused while the packaged sources, the CUDA toolkit and TensorRT are the ones it was
built from; otherwise ``setup`` builds again and ``serve`` refuses with "run edgeserve setup".
"""

from __future__ import annotations

import contextlib
import datetime
import fcntl
import hashlib
import importlib.resources
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from edgeserve import __version__, config

# What the server needs from a build (relative paths, kept as in Edge-LLM's build tree).
EDGELLM_VERSION = "0.11.0"  # the release of the pinned commit (native/edgellm/COMMIT)
_INSTALL = ("edgeserve-worker", "attention-check", "examples/llm/llm_build", "libNvInfer_edgellm_plugin.so")
KEEP_VERSIONS = 2


class NativeBuildError(RuntimeError):
    """Building or installing the GPU runtime failed; the message says what to do."""


def sources_dir() -> Path:
    """The packaged native sources (a real folder in every install)."""
    return Path(str(importlib.resources.files("edgeserve") / "native"))


def edgellm_commit() -> str:
    return (sources_dir() / "edgellm" / "COMMIT").read_text().strip()


def source_files(root: Path | None = None) -> list[Path]:
    root = sources_dir() if root is None else root
    return sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts)


def source_sha256(root: Path | None = None) -> str:
    """One hash over every packaged native source file (names and contents)."""
    root = sources_dir() if root is None else root
    digest = hashlib.sha256()
    for path in source_files(root):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


# ----------------------------------------------------------------------------------------------- status

@dataclass(frozen=True)
class Status:
    ok: bool
    reason: str  # why it is not usable (empty when ok)
    root: Path | None  # the installed version folder
    info: dict | None


def status(root: Path | None = None) -> Status:
    """Whether the installed GPU runtime exists and matches the packaged sources."""
    link = config.native_current() if root is None else Path(root)
    target = link.resolve() if link.exists() else None
    if target is None or not (target / "build_info.json").is_file():
        return Status(False, "the GPU runtime is not built yet", None, None)
    try:
        info = json.loads((target / "build_info.json").read_text())
    except (OSError, ValueError) as e:
        return Status(False, f"cannot read {target / 'build_info.json'} ({e})", target, None)
    missing = [name for name in _INSTALL if not (target / name).exists()]
    if missing:
        return Status(False, f"the GPU runtime in {target} is incomplete (missing {', '.join(missing)})", target, info)
    if info.get("source_sha256") != source_sha256():
        built = info.get("edgeserve_version", "an older edgeserve")
        return Status(False, f"the GPU runtime was built by edgeserve {built} from other sources", target, info)
    return Status(True, "", target, info)


def toolchain_changes(info: dict) -> list[str]:
    """How this machine's CUDA toolkit / TensorRT differ from the ones ``info`` was built with."""
    changes = []
    cuda = _cuda_version()
    if cuda and info.get("cuda") and cuda != info["cuda"]:
        changes.append(f"CUDA {info['cuda']} -> {cuda}")
    trt = _tensorrt_version()
    if trt and info.get("tensorrt") and trt != info["tensorrt"]:
        changes.append(f"TensorRT {info['tensorrt']} -> {trt}")
    return changes


def _cuda_version() -> str | None:
    nvcc = shutil.which("nvcc") or ("/usr/local/cuda/bin/nvcc" if os.access("/usr/local/cuda/bin/nvcc", os.X_OK)
                                    else None)
    if not nvcc:
        return None
    try:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"release (\d+\.\d+)", out)
    return match.group(1) if match else None


def tensorrt_header(prefix: str | None = None) -> Path | None:
    prefix = prefix or os.environ.get("TRT_PACKAGE_DIR") or "/usr"
    arch = os.uname().machine
    for candidate in (Path(prefix) / "include" / "NvInferVersion.h",
                      Path(prefix) / "include" / f"{arch}-linux-gnu" / "NvInferVersion.h"):
        if candidate.is_file():
            return candidate
    return None


def _tensorrt_version() -> str | None:
    """The version in NvInferVersion.h (numbers up to TensorRT 10.7, TRT_*_ENTERPRISE macros from 10.8 on)."""
    header = tensorrt_header()
    if header is None:
        return None
    defines = dict(re.findall(r"^#define[ \t]+(\w+)[ \t]+(\S+)", header.read_text(errors="replace"), re.MULTILINE))
    parts = []
    for part in ("MAJOR", "MINOR", "PATCH", "BUILD"):
        value = defines.get(f"NV_TENSORRT_{part}")
        value = defines.get(value, value) if value else None
        if value is None:
            return None
        parts.append(value)
    return ".".join(parts)


# ----------------------------------------------------------------------------------------------- build

def build(progress: Callable[[str], None] = print) -> Path:
    """Build the GPU runtime from the packaged sources and install it; returns the installed folder."""
    root = config.build_root()
    root.mkdir(parents=True, exist_ok=True)
    with file_lock(root / ".lock", progress, "another edgeserve setup is building the GPU runtime"):
        sources = root / "sources"
        digest = source_sha256()
        if not sources.is_dir() or source_sha256(sources) != digest:
            # CMake keeps this path; it must not point into a venv that may be deleted or contain spaces
            if sources.exists():
                shutil.rmtree(sources)
            shutil.copytree(sources_dir(), sources, ignore=shutil.ignore_patterns("__pycache__"))
        log_path = root / "build.log"
        if (root / "build" / "build_info.json").is_file():  # an earlier build: only what changed is compiled
            progress(f"updating the GPU runtime build in {root} (log: {log_path}); this takes a few minutes")
        else:
            progress(f"building the GPU runtime in {root} (log: {log_path}); this takes about 30 minutes on a Jetson "
                     "Orin Nano, less on a PC")
        env = dict(os.environ, EDGESERVE_SOURCES=str(sources), EDGESERVE_BUILD_ROOT=str(root))
        env.pop("EDGESERVE_API_KEY", None)
        code = run_logged(["bash", str(sources / "build.sh")], env, log_path)
        if code != 0:
            raise NativeBuildError(f"the GPU runtime build failed (exit {code}); the last lines are above and the "
                                   f"whole log is {log_path}")
        return install(root / "build", digest)


def install(build_dir: Path, digest: str) -> Path:
    """Copy a finished build into a new version folder and make it ``current``."""
    info = json.loads((build_dir / "build_info.json").read_text())
    info.update(source_sha256=digest, edgeserve_version=__version__,
                built_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
    version = hashlib.sha256(f"{digest}:{info.get('plugin_sha256')}".encode()).hexdigest()[:16]
    native = config.native_root()
    native.mkdir(parents=True, exist_ok=True)
    final = native / version
    if not final.exists():
        staging = Path(tempfile.mkdtemp(dir=native, prefix=f".installing-{version}-"))
        try:
            for name in _INSTALL:
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2((build_dir / name).resolve(), target)  # the plugin .so is a symlink chain
            (staging / "build_info.json").write_text(json.dumps(info, indent=2) + "\n")
            os.replace(staging, final)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    _switch_current(native, version)
    _prune(native, version)
    return final


def _switch_current(native: Path, version: str) -> None:
    link = native / "current"
    tmp = native / f".current-{os.getpid()}"
    with contextlib.suppress(FileNotFoundError):
        tmp.unlink()
    tmp.symlink_to(version)
    os.replace(tmp, link)  # atomic: a server starting now sees either the old or the new version


def _prune(native: Path, current: str) -> None:
    """Keep the current version and the newest others up to KEEP_VERSIONS; remove the rest and leftovers."""
    versions = sorted((p for p in native.iterdir() if p.is_dir() and re.fullmatch(r"[0-9a-f]{16}", p.name)),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    keep = {current}
    for path in versions:
        if len(keep) >= KEEP_VERSIONS:
            break
        keep.add(path.name)
    for path in versions:
        if path.name not in keep:
            shutil.rmtree(path, ignore_errors=True)
    for leftover in native.glob(".installing-*"):
        shutil.rmtree(leftover, ignore_errors=True)


@contextlib.contextmanager
def file_lock(path: Path, progress: Callable[[str], None], holder: str) -> Iterator[None]:
    """Hold an exclusive lock on ``path``; says "<holder>; waiting for it to finish" while another process has it."""
    with open(path, "w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            progress(f"{holder}; waiting for it to finish")
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_logged(command: list[str], env: dict[str, str], log_path: Path, show: bool = True) -> int:
    """Run ``command``, showing its output (unless ``show`` is false) and appending it to ``log_path``; Ctrl-C stops
    it and everything it started."""
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n===== {datetime.datetime.now().isoformat(timespec='seconds')} {' '.join(command)}\n")
        proc = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                errors="replace", start_new_session=True, bufsize=1)
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                if show:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                log_file.write(line)
            return proc.wait()
        except BaseException:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=30)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            raise


# ----------------------------------------------------------------------------------------------- self-test

def self_test(root: Path, progress: Callable[[str], None] = print) -> bool:
    """Run attention-check (the engine's attention kernels against a CPU reference) and record the result."""
    checker = root / "attention-check"
    progress("checking the attention kernels against a CPU reference (about a minute)")
    started = time.monotonic()
    try:
        result = subprocess.run([str(checker)], capture_output=True, text=True, timeout=1800,
                                env=dict(os.environ, EDGELLM_PLUGIN_PATH=str(root / "libNvInfer_edgellm_plugin.so")))
        output, ok = result.stdout + result.stderr, result.returncode == 0 and "all ok" in result.stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        output, ok = str(e), False
    info_path = root / "build_info.json"
    with contextlib.suppress(OSError, ValueError):
        info = json.loads(info_path.read_text())
        info["self_test"] = {"ok": ok, "seconds": round(time.monotonic() - started, 1),
                             "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
        info_path.write_text(json.dumps(info, indent=2) + "\n")
    if not ok:
        progress(output.strip()[-3000:])
    return ok


__all__ = ["EDGELLM_VERSION", "KEEP_VERSIONS", "NativeBuildError", "Status", "build", "edgellm_commit", "file_lock",
           "install", "run_logged", "self_test", "source_sha256", "sources_dir", "status", "tensorrt_header",
           "toolchain_changes"]
