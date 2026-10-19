"""TensorRT engine cache: build an engine for (model, limits, toolchain, GPU) once, reuse it afterwards.

A TensorRT engine only works with the exact TensorRT build, GPU architecture and Edge-LLM
plugin/kernel build that produced it, so the cache key covers all of them plus the model's
content digest and the build limits. Builds run Edge-LLM's ``llm_build`` in a separate process
(all builder memory is released before the server loads the engine), under a per-model file lock,
into a staging folder that is flushed to disk and renamed into place only after it is complete —
an interrupted build never leaves a half-written engine behind, and what a killed build left in
its staging folder is removed by the next build of that model.
"""

from __future__ import annotations

import contextlib
import ctypes
import errno
import fcntl
import functools
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from edgeserve.model import ModelInfo

log = logging.getLogger("edgeserve.engine")

CACHE_SCHEMA = 1
KV_PAGE_TOKENS = 128  # Edge-LLM allocates the KV cache in pages of 128 tokens
KEEP_ENGINES_PER_MODEL = 3
_REQUIRED_ENGINE_FILES = ("llm.engine", "config.json", "tokenizer.json", "tokenizer_config.json")
# Folder names in the cache are "<model>-<key16>" and ".building-<model>-<key16>-<pid>"; model names may be
# prefixes of one another ("llama", "llama-3b"), so names are matched exactly, never by glob.
_ENGINE_SUFFIX = re.compile(r"[0-9a-f]{16}")
_STAGING_SUFFIX = re.compile(r"[0-9a-f]{16}-[0-9]+")
_PR_SET_PDEATHSIG = 1
try:
    _prctl = ctypes.CDLL(None, use_errno=True).prctl
except (OSError, AttributeError):  # not Linux
    _prctl = None


class EngineError(RuntimeError):
    """Building or validating an engine failed; the message says what to do."""


@dataclass(frozen=True)
class NativeBuild:
    """The GPU runtime for this machine: the edgeserve worker, Edge-LLM's llm_build and TensorRT plugin.

    Built and installed by ``edgeserve setup`` (see edgeserve.nativebuild); ``root`` is the installed
    version folder (the ``current`` link is resolved once, so a rebuild never changes binaries under a
    running server) or an Edge-LLM build folder given with EDGESERVE_NATIVE_DIR.
    """

    root: Path
    worker: Path
    llm_build: Path
    plugin: Path
    info: dict  # build_info.json: Edge-LLM commit, patches, plugin hash, CUDA, TensorRT, SM, CuTe DSL kernels

    @classmethod
    def locate(cls, root: Path) -> NativeBuild:
        root = Path(root).expanduser().resolve()
        paths = {
            "worker": root / "edgeserve-worker",
            "llm_build": root / "examples" / "llm" / "llm_build",
            "plugin": root / "libNvInfer_edgellm_plugin.so",
            "info": root / "build_info.json",
        }
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise EngineError(
                "the GPU runtime is not built yet (missing " + ", ".join(missing) + "): run edgeserve setup "
                "(or set EDGESERVE_NATIVE_DIR to a runtime build folder)"
            )
        try:
            info = json.loads(paths["info"].read_text())
        except (OSError, ValueError) as e:
            raise EngineError(f"cannot read {paths['info']} ({e}): run edgeserve setup") from e
        if not isinstance(info, dict):
            raise EngineError(f"{paths['info']} is not a JSON object: run edgeserve setup")
        return cls(root=root, worker=paths["worker"], llm_build=paths["llm_build"], plugin=paths["plugin"], info=info)

    def env(self) -> dict[str, str]:
        """Environment for Edge-LLM processes: the plugin by absolute path, TensorRT's libraries found, no secrets."""
        env = {k: v for k, v in os.environ.items() if k not in ("EDGESERVE_API_KEY", "EDGESERVE_TLS_KEY")}
        env["EDGELLM_PLUGIN_PATH"] = str(self.plugin)
        lib_dir = self.info.get("trt_lib_dir")
        if isinstance(lib_dir, str) and lib_dir and lib_dir not in _SYSTEM_LIB_DIRS:
            # a TensorRT tarball: its libraries are not on the loader's default path (under systemd)
            env["LD_LIBRARY_PATH"] = os.pathsep.join(p for p in (lib_dir, env.get("LD_LIBRARY_PATH", "")) if p)
        return env


_SYSTEM_LIB_DIRS = frozenset({"/usr/lib", "/usr/lib64", "/usr/lib/aarch64-linux-gnu", "/usr/lib/x86_64-linux-gnu"})


@dataclass(frozen=True)
class BuildProfile:
    max_batch_size: int
    max_input_len: int
    max_new_tokens: int

    @property
    def kv_capacity(self) -> int:
        """Tokens of KV cache per sequence: prompt + output + 1 (Edge-LLM keeps one token of headroom), page-aligned."""
        tokens = self.max_input_len + self.max_new_tokens + 1
        return -(-tokens // KV_PAGE_TOKENS) * KV_PAGE_TOKENS


@dataclass(frozen=True)
class EngineInfo:
    path: Path
    key: str
    profile: BuildProfile
    manifest: dict
    built_now: bool

    @property
    def fingerprint(self) -> str:
        return self.key[:16]

    @property
    def max_new_tokens(self) -> int:
        """Largest output a request can ask for (KV capacity minus the shortest prompt and headroom)."""
        return self.profile.kv_capacity - 2


def engine_key(model_digest: str, profile: BuildProfile, native: NativeBuild, gpu: dict) -> tuple[str, dict]:
    """Cache key over everything an engine depends on; returns (sha256 hex, the hashed components)."""
    components = {
        "schema": CACHE_SCHEMA,
        "model": model_digest,
        "profile": {"max_batch_size": profile.max_batch_size, "max_input_len": profile.max_input_len,
                    "kv_capacity": profile.kv_capacity},
        "edgellm_commit": native.info.get("edgellm_commit"),
        "patches": native.info.get("patches"),
        "plugin_sha256": native.info.get("plugin_sha256"),
        "cute_dsl": native.info.get("cute_dsl"),
        "cuda": native.info.get("cuda"),
        "tensorrt": gpu.get("tensorrt"),
        "sm": gpu.get("sm"),
        "gpu_name": gpu.get("gpu_name"),
    }
    digest = hashlib.sha256(json.dumps(components, sort_keys=True).encode()).hexdigest()
    return digest, components


def plan_tensorrt_version(engine_file: Path) -> str | None:
    """TensorRT version recorded in a serialized plan header ("ftrt" magic, version bytes 24-27)."""
    try:
        with open(engine_file, "rb") as f:
            header = f.read(28)
    except OSError:
        return None
    if len(header) < 28 or header[:4] != b"ftrt":
        return None
    return ".".join(str(b) for b in header[24:28])


class EngineCache:
    def __init__(self, cache_dir: Path) -> None:
        self.root = cache_dir / "engines"
        self._held: dict[Path, int] = {}  # engine folder -> descriptor holding a shared "in use" lock on it

    def path_for(self, model_name: str, key: str) -> Path:
        return self.root / f"{model_name}-{key[:16]}"

    def lookup(self, model: ModelInfo, key: str, profile: BuildProfile, tensorrt: str) -> EngineInfo | None:
        """A valid cached engine for this key, or None (stale or broken entries are reported and ignored)."""
        path = self.path_for(model.name, key)
        manifest_file = path / "manifest.json"
        if not manifest_file.exists():
            return None
        manifest, problem = self._validate(path, manifest_file, key, tensorrt)
        if problem:
            log.warning("ignoring cached engine %s: %s", path, problem)
            return None
        with contextlib.suppress(OSError):  # a cache this user may read but not modify is still usable
            os.utime(path)  # most-recently-used marker for pruning
        return EngineInfo(path=path, key=key, profile=profile, manifest=manifest, built_now=False)

    @staticmethod
    def _validate(path: Path, manifest_file: Path, key: str, tensorrt: str) -> tuple[dict, str | None]:
        try:
            manifest = json.loads(manifest_file.read_text())
        except (OSError, ValueError) as e:
            return {}, f"unreadable manifest ({e})"
        if not isinstance(manifest, dict) or not manifest.get("complete") or manifest.get("key") != key:
            return {}, "incomplete build or key mismatch"
        for name in _REQUIRED_ENGINE_FILES:
            if not (path / name).exists():
                return {}, f"missing {name}"
        for name in manifest.get("linked_files", []):
            if not (path / name).exists():  # a symlinked weight file whose model folder moved
                return {}, f"dangling {name}"
        plan_version = plan_tensorrt_version(path / "llm.engine")
        if plan_version != tensorrt:
            return {}, f"engine was built by TensorRT {plan_version}, this machine has {tensorrt}"
        return manifest, None

    def ensure(
        self,
        model: ModelInfo,
        model_digest: str,
        profile: BuildProfile,
        native: NativeBuild,
        gpu: dict,
        on_progress: Callable[[str], None] = lambda message: None,
        cancel: threading.Event | None = None,
    ) -> EngineInfo:
        """Return the cached engine, building it first when needed (blocking; run it in a thread).

        Other processes' pruning leaves the returned engine alone until release(). Setting ``cancel`` stops a
        build: llm_build is killed, its staging folder removed and EngineError raised.
        """
        key, components = engine_key(model_digest, profile, native, gpu)
        tensorrt = str(gpu.get("tensorrt"))
        cached = self._lookup_held(model, key, profile, tensorrt)
        if cached:
            return cached
        self.root.mkdir(parents=True, exist_ok=True)
        # One build per model at a time across processes (and never unlinked: a waiter would otherwise lock a
        # deleted file while a newcomer locks a new one). Holding it also makes every staging folder of the
        # model left in the cache a leftover of a killed build, and keeps prunes from racing builds.
        with open(self.root / f".lock-{model.name}", "a") as lock_file:
            _lock(lock_file, on_progress, cancel)
            cached = self._lookup_held(model, key, profile, tensorrt)
            if cached:
                return cached
            for leftover in self._folders(f".building-{model.name}-", _STAGING_SUFFIX):
                log.info("removing an interrupted engine build: %s", leftover)
                shutil.rmtree(leftover, ignore_errors=True)
            info = self._build(model, key, components, profile, native, tensorrt, on_progress, cancel,
                               integrated=bool(gpu.get("integrated")))
            self._hold(info.path)
            self.prune(model.name, keep=info.path)
        return info

    def release(self) -> None:
        """Let other processes prune the engines this cache returned (call when no longer using them)."""
        for fd in self._held.values():
            with contextlib.suppress(OSError):
                os.close(fd)
        self._held.clear()

    def _lookup_held(self, model: ModelInfo, key: str, profile: BuildProfile, tensorrt: str) -> EngineInfo | None:
        info = self.lookup(model, key, profile, tensorrt)
        if info is not None and not self._hold(info.path):
            return None  # removed by another process's prune after it was validated
        return info

    def _hold(self, path: Path) -> bool:
        """Mark an engine folder as in use (a shared flock that prune() respects); False if it is gone or going."""
        if path in self._held:
            return True
        try:
            fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        except OSError:
            return False
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:  # a prune holds it exclusively: it is being removed
            os.close(fd)
            return False
        except OSError:
            pass  # a file system without flock: usable, just not protected
        try:
            present = os.path.samestat(os.fstat(fd), os.stat(path)) and (path / "manifest.json").exists()
        except OSError:
            present = False
        if not present:  # removed (and possibly rebuilt) between the lookup and the lock
            os.close(fd)
            return False
        self._held[path] = fd
        return True

    def _build(
        self,
        model: ModelInfo,
        key: str,
        components: dict,
        profile: BuildProfile,
        native: NativeBuild,
        tensorrt: str,
        on_progress: Callable[[str], None],
        cancel: threading.Event | None = None,
        integrated: bool = False,
    ) -> EngineInfo:
        final = self.path_for(model.name, key)
        staging = self.root / f".building-{model.name}-{key[:16]}-{os.getpid()}"
        shutil.rmtree(staging, ignore_errors=True)
        try:
            staging.mkdir(parents=True)
            linked = self._link_sidecars(model, staging)

            cmd = [
                str(native.llm_build),
                "--onnxDir", str(model.path),
                "--engineDir", str(staging),
                "--maxBatchSize", str(profile.max_batch_size),
                "--maxInputLen", str(profile.max_input_len),
                "--maxKVCacheCapacity", str(profile.kv_capacity),
            ]
            log_file = staging / "build.log"
            log.info("building TensorRT engine for %s (%s)", model.name, " ".join(cmd[3:]))
            on_progress("building the TensorRT engine (a few minutes)")
            started = time.monotonic()
            with open(log_file, "w") as out:
                proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=native.env(),
                                        preexec_fn=functools.partial(_lowest_priority, os.getpid()),
                                        start_new_session=True)
                try:
                    peak_rss, min_available = _watch_memory(proc, on_progress, started, cancel)
                finally:
                    _kill_group(proc)  # still running only if the build was cancelled or watching it failed
            seconds = time.monotonic() - started

            if proc.returncode != 0:
                log.warning("llm_build exited with %s after %.0f s: its peak memory was %.2f GiB, the lowest free "
                            "system memory %.2f GiB", proc.returncode, seconds, peak_rss / 2**30,
                            min_available / 2**30 if min_available < 1 << 62 else -1)
                raise EngineError(_explain_build_failure(proc.returncode, _tail(log_file), model, profile,
                                                         integrated))
            missing = [name for name in _REQUIRED_ENGINE_FILES if not (staging / name).exists()]
            if missing:
                raise EngineError(f"llm_build finished but did not produce {', '.join(missing)}")

            manifest = {
                "complete": True,
                "key": key,
                "components": components,
                "model": model.name,
                "model_path": str(model.path),
                "profile": asdict(profile) | {"kv_capacity": profile.kv_capacity},
                "tensorrt": plan_tensorrt_version(staging / "llm.engine") or tensorrt,
                "linked_files": linked,
                "build": {"seconds": round(seconds, 1), "peak_rss_bytes": peak_rss,
                          "min_mem_available_bytes": min_available},
                "created_at": int(time.time()),
            }
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=1))
            _fsync_tree(staging)  # after a power cut a published engine must not turn out truncated
            if final.exists():
                shutil.rmtree(final)
            os.replace(staging, final)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        _fsync_dir(self.root)
        log.info("engine ready in %.0f s (peak build RSS %.2f GiB): %s", seconds, peak_rss / 2**30, final)
        return EngineInfo(path=final, key=key, profile=profile, manifest=manifest, built_now=True)

    @staticmethod
    def _link_sidecars(model: ModelInfo, staging: Path) -> list[str]:
        """Hard-link (or, across file systems, symlink) the weight files llm_build would copy
        (embedding, vocab map, externalized weights: every *.safetensors) so it skips copying them."""
        linked = []
        for source in sorted(model.path.glob("*.safetensors")):
            name = source.name
            target = staging / name
            try:
                os.link(source, target)
            except OSError as e:
                if e.errno not in (errno.EXDEV, errno.EPERM, errno.EMLINK):
                    raise
                os.symlink(source, target)
                linked.append(name)  # symlinks break if the model folder moves: validated on lookup
        return linked

    def prune(self, model_name: str, keep: Path) -> None:
        """Keep the most recently used KEEP_ENGINES_PER_MODEL engines of a model; delete older ones that no
        process is using."""
        candidates = []
        for path in self._folders(f"{model_name}-", _ENGINE_SUFFIX):
            if path == keep:
                continue
            try:
                candidates.append((path.stat().st_mtime, path))
            except OSError:
                continue  # removed meanwhile
        candidates.sort(reverse=True)
        for _, stale in candidates[KEEP_ENGINES_PER_MODEL - 1:]:
            _remove_unused(stale)

    def _folders(self, prefix: str, suffix: re.Pattern[str]) -> list[Path]:
        """Folders of the cache named exactly ``prefix`` followed by a ``suffix`` match."""
        try:
            entries = list(os.scandir(self.root))
        except OSError:
            return []
        found = []
        for entry in entries:
            if not entry.name.startswith(prefix) or not suffix.fullmatch(entry.name[len(prefix):]):
                continue
            with contextlib.suppress(OSError):
                if entry.is_dir(follow_symlinks=False):
                    found.append(Path(entry.path))
        return found


def _lock(lock_file, on_progress: Callable[[str], None], cancel: threading.Event | None) -> None:
    """Take the model's build lock, waiting (cancellably) while another process holds it."""
    waiting = False
    while True:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            pass
        if not waiting:
            on_progress("waiting for the engine build lock")
            waiting = True
        if cancel is None:
            time.sleep(1.0)
        elif cancel.wait(1.0):
            raise EngineError("the engine build was cancelled")


def _remove_unused(path: Path) -> None:
    """Delete an engine folder unless a process holds it (EngineCache._hold)."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log.info("keeping old engine %s: it is in use", path)
            return
        except OSError:
            pass  # a file system without flock: nothing can be in use as far as we can tell
        log.info("removing old engine %s", path)
        shutil.rmtree(path, ignore_errors=True)
    finally:
        os.close(fd)


def _lowest_priority(parent: int) -> None:
    """Child setup for the builder: first in line for the OOM killer, lower CPU priority, and killed when the server
    dies (it leads its own session, so a SIGKILL or a closed terminal ending the server would leave it running)."""
    if _prctl is not None:
        _prctl(ctypes.c_int(_PR_SET_PDEATHSIG), ctypes.c_ulong(signal.SIGKILL))
        if os.getppid() != parent:  # the server was gone before that took effect
            os.kill(os.getpid(), signal.SIGKILL)
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass
    os.nice(10)


def _kill_group(proc: subprocess.Popen) -> None:
    """Kill a builder that is still running (with anything it started: it leads its own session) and reap it."""
    if proc.poll() is None:
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def _watch_memory(
    proc: subprocess.Popen, on_progress: Callable[[str], None], started: float, cancel: threading.Event | None = None
) -> tuple[int, int]:
    """Wait for the builder while sampling its peak RSS and the system's lowest MemAvailable."""
    peak_rss, min_available = 0, 1 << 62
    last_report = started
    while True:
        try:
            proc.wait(timeout=1.0)
            break
        except subprocess.TimeoutExpired:
            pass
        if cancel is not None and cancel.is_set():
            raise EngineError("the engine build was cancelled")
        peak_rss = max(peak_rss, _vm_hwm(proc.pid))
        min_available = min(min_available, _mem_available())
        now = time.monotonic()
        if now - last_report >= 30:
            on_progress(f"building the TensorRT engine ({now - started:.0f} s so far)")
            last_report = now
    return peak_rss, (min_available if min_available < 1 << 62 else 0)


def _vm_hwm(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _mem_available() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 1 << 62


def _fsync_tree(root: Path) -> None:
    """Flush every file written into ``root`` (symlinks point at model files) and the folders themselves."""
    for directory, _, files in os.walk(root):
        for name in files:
            path = os.path.join(directory, name)
            if os.path.islink(path):
                continue
            fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        _fsync_dir(Path(directory))


def _fsync_dir(path: Path) -> None:
    """Make renames and new entries in a folder durable (best effort: not every file system supports it)."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _tail(path: Path, lines: int = 25) -> str:
    try:
        text = path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    keep = [line for line in text if "[ERROR]" in line] or text
    return "\n".join(keep[-lines:])


def _explain_build_failure(returncode: int, tail: str, model: ModelInfo, profile: BuildProfile,
                           integrated: bool = False) -> str:
    # SIGKILL is what the OOM killer sends (137 when a shell reports it); other signals are crashes.
    killed = returncode in (-signal.SIGKILL, 128 + signal.SIGKILL)
    oom = killed or any(s in tail.lower() for s in ("out of memory", "bad_alloc", "cudaerrormemoryallocation"))
    advice = ""
    if oom:
        free = ("stop the desktop: sudo systemctl isolate multi-user.target; close IDE servers" if integrated
                else "close other programs")
        advice = (f"\nThe build ran out of memory. Free memory ({free}), add swap for the build, or lower "
                  f"--max-input-len (now {profile.max_input_len}).")
    return f"TensorRT engine build for {model.name} failed (exit code {returncode}).{advice}\nllm_build log:\n{tail}"


__all__ = [
    "BuildProfile",
    "EngineCache",
    "EngineError",
    "EngineInfo",
    "NativeBuild",
    "engine_key",
    "plan_tensorrt_version",
]
