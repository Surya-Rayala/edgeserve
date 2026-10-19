"""Hardware and software facts about this machine (Jetson module or discrete NVIDIA GPU).

GPU facts come from the worker binary (``edgeserve-worker --probe``) so the front-end
process never creates a CUDA context of its own (on Jetson a context costs ~65 MB of the
shared memory). Host facts come from procfs/sysfs, ``nvpmodel`` and NVML (dGPU only).

All filesystem reads go through ``sysroot`` so tests can point the module at a fake tree.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import re
import shutil
import signal
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Dense FP16 tensor-core FLOPs per clock per SM (FP16 accumulate), by compute capability.
# Peak = SMs x this x max GPU clock. Matches NVIDIA's published dense figures, e.g.
# Orin Nano Super 8 SM x 2048 x 1.02 GHz = 16.7 TFLOPS, RTX 4090 128 x 1024 x 2.52 GHz = 330 TFLOPS.
FP16_FLOPS_PER_CLOCK_PER_SM: dict[int, int] = {
    80: 2048,  # A100, A30
    86: 1024,  # RTX 30, A10, A40
    87: 2048,  # Jetson Orin
    89: 1024,  # RTX 40, L4, L40S
    90: 4096,  # H100, H200
    100: 8192,  # B200
    110: 8192,  # Jetson Thor
    120: 1024,  # RTX 50, RTX PRO Blackwell
    121: 1024,  # DGX Spark (GB10)
}

# Rails preferred for energy accounting, most inclusive first (module/board input power).
ENERGY_RAIL_PREFERENCE: tuple[str, ...] = ("VDD_IN", "VIN", "VIN_SYS_5V0")

# JetPack release of each L4T release, used when the nvidia-jetpack meta package is absent.
_JETPACK_BY_L4T: dict[str, str] = {
    "R36.4.3": "6.2",
    "R36.4.4": "6.2.1",
    "R36.5.0": "6.2.2",
    "R39.2.0": "7.2",
    "R39.2.1": "7.2.1",
}

_L4T_RE = re.compile(r"#\s*R(\d+)\s*\(release\),\s*REVISION:\s*(\d+(?:\.\d+)*)", re.IGNORECASE)
_DPKG_VERSION_RE = re.compile(r"(?:\d+:)?(\d+(?:\.\d+)*)")  # "6.2.1+b38" -> "6.2.1"
_POWER_MODE_RE = re.compile(r"^[ \t]*NV Power Mode:[ \t]*(\S.*?)[ \t]*$", re.MULTILINE)
_INA3221_LABEL_RE = re.compile(r"in(\d+)_label")
_TRAILING_NUMBER_RE = re.compile(r"(\d+)$")

_GPU_DEVFREQ_PATTERNS: tuple[str, ...] = ("*.gpu", "*gpu*", "*ga10b*")
_EMC_FREQ_FILES: tuple[str, ...] = ("sys/kernel/nvpmodel_clk_cap/emc", "sys/class/devfreq/bwmgr/max_freq")

_HOST_TOOL_TIMEOUT_S = 5.0
# nvpmodel lives in /usr/sbin, which minimal service environments can leave off PATH.
_HOST_TOOL_DIRS: tuple[str, ...] = ("/usr/sbin", "/usr/bin", "/sbin", "/bin")

_STDERR_TAIL_LINES = 20
_STDERR_TAIL_CHARS = 4000

# Worker probe keys copied verbatim into Platform, with their required JSON types.
_GPU_FIELDS: dict[str, type] = {
    "gpu_name": str,
    "sm": int,
    "sm_count": int,
    "integrated": bool,
    "total_memory_bytes": int,
    "memory_bus_width_bits": int,
    "cuda_runtime": str,
    "cuda_driver": str,
    "tensorrt": str,
    "edgellm": str,
}


class PlatformError(RuntimeError):
    """The GPU could not be probed (no device, broken driver, missing worker binary)."""


@dataclass(frozen=True)
class PowerRail:
    """One power sensor channel exposed through hwmon.

    ``kind`` is ``"ina3221"`` (paths = (millivolts file, milliamps file)) or ``"ina238"``
    (paths = (microwatts file,)).
    """

    name: str
    kind: str
    paths: tuple[str, ...]

    def read_watts(self) -> float | None:
        """Instantaneous power in watts, or None when the sensor files cannot be read or parsed."""
        try:
            if self.kind == "ina3221":
                millivolts = float(Path(self.paths[0]).read_text())
                milliamps = float(Path(self.paths[1]).read_text())
                return millivolts * milliamps / 1e6
            if self.kind == "ina238":
                return float(Path(self.paths[0]).read_text()) / 1e6
        except (OSError, ValueError, IndexError):
            return None
        return None


@dataclass(frozen=True)
class Platform:
    """Machine facts; build with :func:`probe`, update clock caps with :func:`refresh_clocks`."""

    kind: str  # "jetson" | "dgpu"
    hostname: str
    board: str | None  # Jetson: /proc/device-tree/model; dGPU: None
    l4t: str | None  # e.g. "R36.4.4"
    jetpack: str | None  # e.g. "6.2.1"
    gpu_name: str
    sm: int  # compute capability as an int, e.g. 87
    sm_count: int
    integrated: bool
    total_memory_bytes: int
    memory_bus_width_bits: int
    cuda_runtime: str
    cuda_driver: str
    tensorrt: str  # full version the worker was compiled against, e.g. "10.3.0.30"
    edgellm: str  # Edge-LLM runtime version, e.g. "0.11.0"
    power_mode: str | None  # nvpmodel mode name, e.g. "25W" (Jetson only)
    gpu_max_freq_hz: int | None  # current GPU clock cap (Jetson devfreq max_freq / dGPU max SM clock)
    emc_max_freq_hz: int | None  # Jetson memory-controller clock cap
    peak_flops_per_s: float | None  # dense FP16 tensor peak at gpu_max_freq_hz
    peak_mem_bw_bytes_per_s: float | None  # DRAM peak at the current memory clock cap
    power_rails: tuple[PowerRail, ...] = ()
    energy_rail: PowerRail | None = None  # rail integrated for energy on Jetson
    nvml_index: int | None = None  # dGPU: NVML device index used for power/energy
    gpu_freq_path: str | None = field(default=None, repr=False)  # sysfs file re-read by refresh_clocks
    emc_freq_path: str | None = field(default=None, repr=False)
    # probe() read the live host (sysroot "/"), so refresh_clocks may re-run nvpmodel.
    _live_host: bool = field(default=False, repr=False, compare=False)

    def as_dict(self) -> dict:
        """JSON-serialisable snapshot for records and run summaries (no sysfs paths)."""
        snapshot: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            if f.name.startswith("_") or f.name.endswith("_path"):
                continue
            value = getattr(self, f.name)
            if f.name == "power_rails":
                value = [rail.name for rail in value]
            elif f.name == "energy_rail":
                value = None if value is None else value.name
            snapshot[f.name] = value
        return snapshot


def probe_gpu(worker_binary: Path, env: dict[str, str] | None = None, timeout: float = 60.0) -> dict:
    """Run ``<worker_binary> --probe`` and return its JSON event (``"ev": "probe"``).

    ``env`` adds variables to the current environment rather than replacing it.
    Raises PlatformError with the worker's stderr tail when it fails or reports an error event.
    """
    executable = os.fspath(worker_binary)
    if os.sep not in executable:  # a bare file name means the file in the working directory, not a PATH search
        executable = os.path.join(os.curdir, executable)
    try:
        proc = subprocess.run(
            [executable, "--probe"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            env=None if env is None else {**os.environ, **env},
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise PlatformError(_with_stderr(f"GPU probe timed out after {timeout:g} s", exc.stderr)) from None
    except OSError as exc:
        raise PlatformError(f"cannot run the worker binary {worker_binary}: {exc.strerror or exc}") from exc

    events = _json_objects(proc.stdout)
    error = next((event for event in events if event.get("ev") == "error"), None)
    if error is not None:
        detail = error.get("message") or error.get("code") or "unknown error"
        raise PlatformError(_with_stderr(f"GPU probe failed: {detail}", proc.stderr))
    if proc.returncode != 0:
        summary = f"GPU probe failed: the worker {_describe_exit(proc.returncode)}"
        raise PlatformError(_with_stderr(summary, proc.stderr))
    for event in events:
        if event.get("ev") == "probe":
            return event
    raise PlatformError(_with_stderr("GPU probe failed: the worker printed no probe event", proc.stderr))


def probe(gpu: dict, sysroot: Path = Path("/")) -> Platform:
    """Combine GPU facts from :func:`probe_gpu` with host facts.

    Rules:
    - kind is "jetson" if ``<sysroot>/etc/nv_tegra_release`` exists, else "dgpu".
    - l4t from the first line of nv_tegra_release ("# R36 (release), REVISION: 4.4, ..." -> "R36.4.4").
    - jetpack from ``dpkg-query -W -f='${Version}' nvidia-jetpack`` ("6.2.1+b38" -> "6.2.1") when
      sysroot is "/", else from a table keyed by l4t (R36.4.3 -> 6.2, R36.4.4 -> 6.2.1, R36.5.0 -> 6.2.2,
      R39.2.0 -> 7.2, R39.2.1 -> 7.2.1); None when unknown.
    - board from ``<sysroot>/proc/device-tree/model`` (strip trailing NUL), Jetson only.
    - power_mode from ``nvpmodel -q`` ("NV Power Mode: 25W" -> "25W") when sysroot is "/" (5 s timeout),
      None otherwise or on failure.
    - Jetson clocks: gpu_max_freq_hz from the first existing ``<sysroot>/sys/class/devfreq/<node>/max_freq``
      among nodes matching ``*.gpu``, ``*gpu*`` or ``*ga10b*``; emc_max_freq_hz from
      ``<sysroot>/sys/kernel/nvpmodel_clk_cap/emc`` (Orin) or ``<sysroot>/sys/class/devfreq/bwmgr/max_freq`` (Thor).
      Never use the worker's clock_rate_khz / memory_clock_rate_khz on Jetson (they ignore power-mode caps).
    - dGPU clocks: NVML (``pynvml``) max SM clock and max memory clock when importable, else the worker's
      clock_rate_khz / memory_clock_rate_khz. nvml_index = 0 when NVML works, else None.
    - peak_mem_bw = memory clock (Hz) x 2 (DDR) x memory_bus_width_bits / 8; None if the clock is unknown.
    - peak_flops = sm_count x FP16_FLOPS_PER_CLOCK_PER_SM[sm] x gpu_max_freq_hz; None if either is unknown.
    - power_rails: every ``<sysroot>/sys/class/hwmon/hwmon*`` whose ``name`` is "ina3221" contributes one rail
      per ``in<N>_label`` (skip labels starting with "sum of"), paths (in<N>_input, curr<N>_input); "ina238"
      contributes one rail named by its ``label`` file if present else "VIN", path power1_input.
    - energy_rail: first rail whose name is in ENERGY_RAIL_PREFERENCE order; None when there is none.
      Jetson only (see the field comment): a dGPU's energy comes from NVML device ``nvml_index``.
    - hostname from socket.gethostname().

    Raises PlatformError when ``gpu`` lacks a field Platform needs or has one of the wrong type.
    """
    facts = _gpu_facts(gpu)
    root = Path(sysroot)
    live_host = _is_live_host(root)
    tegra_release = root / "etc/nv_tegra_release"

    board: str | None = None
    l4t: str | None = None
    jetpack: str | None = None
    power_mode: str | None = None
    gpu_freq_file: Path | None = None
    emc_freq_file: Path | None = None
    emc_hz: int | None = None
    nvml_index: int | None = None
    if _exists(tegra_release):
        kind = "jetson"
        l4t = _parse_l4t(_read_text(tegra_release))
        jetpack = (_query_jetpack() if live_host else None) or (_JETPACK_BY_L4T.get(l4t) if l4t else None)
        board = _read_board(root / "proc/device-tree/model")
        power_mode = _query_power_mode() if live_host else None
        gpu_freq_file = _find_gpu_freq_file(root)
        emc_freq_file = next((root / rel for rel in _EMC_FREQ_FILES if _exists(root / rel)), None)
        gpu_hz = _read_hz(gpu_freq_file)
        emc_hz = mem_hz = _read_hz(emc_freq_file)
    else:
        kind = "dgpu"
        nvml_clocks = _nvml_max_clocks()
        nvml_sm_hz, nvml_mem_hz = nvml_clocks if nvml_clocks is not None else (None, None)
        nvml_index = None if nvml_clocks is None else 0
        gpu_hz = nvml_sm_hz or _positive_int(gpu.get("clock_rate_khz"), 1000)
        mem_hz = nvml_mem_hz or _positive_int(gpu.get("memory_clock_rate_khz"), 1000)

    rails = _power_rails(root)
    return Platform(
        kind=kind,
        hostname=socket.gethostname(),
        board=board,
        l4t=l4t,
        jetpack=jetpack,
        **facts,
        power_mode=power_mode,
        gpu_max_freq_hz=gpu_hz,
        emc_max_freq_hz=emc_hz,
        peak_flops_per_s=_peak_flops(facts["sm"], facts["sm_count"], gpu_hz),
        peak_mem_bw_bytes_per_s=_peak_mem_bw(mem_hz, facts["memory_bus_width_bits"]),
        power_rails=rails,
        # Telemetry prefers energy_rail over NVML, so on a dGPU a board sensor would replace NVML's GPU power.
        energy_rail=_pick_energy_rail(rails) if kind == "jetson" else None,
        nvml_index=nvml_index,
        gpu_freq_path=None if gpu_freq_file is None else str(gpu_freq_file),
        emc_freq_path=None if emc_freq_file is None else str(emc_freq_file),
        _live_host=live_host,
    )


def refresh_clocks(platform: Platform) -> Platform:
    """Re-read power mode and clock caps (cheap; nvpmodel can change at runtime) and recompute peaks.

    Jetson only: clock caps are re-read from ``gpu_freq_path`` / ``emc_freq_path`` and the power mode
    from ``nvpmodel`` (only when :func:`probe` read the live host). A value that cannot be re-read keeps
    its previous value. dGPU platforms are returned unchanged (NVML max clocks are fixed).
    """
    if platform.kind != "jetson":
        return platform
    power_mode = platform.power_mode
    if platform._live_host:
        power_mode = _query_power_mode() or power_mode
    gpu_hz = _read_hz(platform.gpu_freq_path) or platform.gpu_max_freq_hz
    emc_hz = _read_hz(platform.emc_freq_path) or platform.emc_max_freq_hz
    return dataclasses.replace(
        platform,
        power_mode=power_mode,
        gpu_max_freq_hz=gpu_hz,
        emc_max_freq_hz=emc_hz,
        peak_flops_per_s=_peak_flops(platform.sm, platform.sm_count, gpu_hz),
        peak_mem_bw_bytes_per_s=_peak_mem_bw(emc_hz, platform.memory_bus_width_bits),
    )


def _gpu_facts(gpu: Any) -> dict[str, Any]:
    """The Platform fields taken from the worker's probe event, type-checked."""
    if not isinstance(gpu, dict):
        raise PlatformError(f"GPU probe result is not a JSON object (got {type(gpu).__name__})")
    bad = [key for key, expected in _GPU_FIELDS.items() if not _has_json_type(gpu.get(key), expected)]
    if bad:
        raise PlatformError("GPU probe result is missing or has invalid fields: " + ", ".join(bad))
    return {key: gpu[key] for key in _GPU_FIELDS}


def _has_json_type(value: Any, expected: type) -> bool:
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, expected)


def _positive_int(value: Any, scale: int = 1) -> int | None:
    """``value * scale`` when value is a positive int (bools, floats, strings and None give None)."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value * scale


def _peak_flops(sm: int, sm_count: int, gpu_hz: int | None) -> float | None:
    per_clock = FP16_FLOPS_PER_CLOCK_PER_SM.get(sm)
    if per_clock is None or gpu_hz is None or sm_count <= 0:
        return None
    return float(sm_count * per_clock * gpu_hz)


def _peak_mem_bw(mem_hz: int | None, bus_width_bits: int) -> float | None:
    if mem_hz is None or bus_width_bits <= 0:
        return None
    return mem_hz * 2 * bus_width_bits / 8


def _is_live_host(root: Path) -> bool:
    try:
        return root.resolve() == Path("/")
    except (OSError, RuntimeError):  # RuntimeError: symlink loop (Python < 3.13)
        return False


def _exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:  # e.g. EACCES on a parent directory (Python < 3.12 raises it)
        return False


def _glob(directory: Path, pattern: str) -> list[Path]:
    try:
        return sorted(directory.glob(pattern))
    except OSError:
        return []


def _read_text(path: Path) -> str | None:
    try:
        return path.read_bytes().decode("utf-8", "replace")
    except OSError:
        return None


def _read_hz(path: str | Path | None) -> int | None:
    """A positive integer clock (Hz) from a sysfs file; None when absent, unreadable or not positive."""
    if path is None:
        return None
    text = _read_text(Path(path))
    if text is None:
        return None
    try:
        return _positive_int(int(text.strip()))
    except ValueError:
        return None


def _parse_l4t(release: str | None) -> str | None:
    if not release:
        return None
    match = _L4T_RE.match(release.partition("\n")[0].strip())
    return f"R{match.group(1)}.{match.group(2)}" if match else None


def _read_board(path: Path) -> str | None:
    text = _read_text(path)
    if text is None:
        return None
    return text.strip("\x00 \t\r\n") or None


def _find_gpu_freq_file(root: Path) -> Path | None:
    devfreq = root / "sys/class/devfreq"
    for pattern in _GPU_DEVFREQ_PATTERNS:
        for node in _glob(devfreq, pattern):
            if _exists(node / "max_freq"):
                return node / "max_freq"
    return None


def _power_rails(root: Path) -> tuple[PowerRail, ...]:
    rails: list[PowerRail] = []
    for hwmon in sorted(_glob(root / "sys/class/hwmon", "hwmon*"), key=_natural_key):
        chip = (_read_text(hwmon / "name") or "").strip()
        if chip == "ina3221":
            rails.extend(_ina3221_rails(hwmon))
        elif chip == "ina238":
            label = (_read_text(hwmon / "label") or "").strip() or "VIN"
            rails.append(PowerRail(label, "ina238", (str(hwmon / "power1_input"),)))
    return tuple(rails)


def _ina3221_rails(hwmon: Path) -> list[PowerRail]:
    channels: list[tuple[int, str]] = []
    for label_file in _glob(hwmon, "in*_label"):
        match = _INA3221_LABEL_RE.fullmatch(label_file.name)
        label = (_read_text(label_file) or "").strip()
        if match and label and not label.lower().startswith("sum of"):
            channels.append((int(match.group(1)), label))
    return [
        PowerRail(label, "ina3221", (str(hwmon / f"in{n}_input"), str(hwmon / f"curr{n}_input")))
        for n, label in sorted(channels)
    ]


def _natural_key(path: Path) -> tuple[int, str]:
    """Sort hwmon2 before hwmon10."""
    match = _TRAILING_NUMBER_RE.search(path.name)
    return (int(match.group(1)) if match else -1, path.name)


def _pick_energy_rail(rails: tuple[PowerRail, ...]) -> PowerRail | None:
    for preferred in ENERGY_RAIL_PREFERENCE:
        for rail in rails:
            if rail.name == preferred:
                return rail
    return None


def _host_command(argv: list[str]) -> str | None:
    """Stdout of a host tool, or None when it is missing, fails or times out."""
    search_path = os.pathsep.join([os.environ.get("PATH", os.defpath), *_HOST_TOOL_DIRS])
    executable = shutil.which(argv[0], path=search_path) or argv[0]
    try:
        proc = subprocess.run(
            [executable, *argv[1:]],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=_HOST_TOOL_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def _query_power_mode() -> str | None:
    output = _host_command(["nvpmodel", "-q"])
    match = _POWER_MODE_RE.search(output) if output else None
    return match.group(1).strip() if match else None


def _query_jetpack() -> str | None:
    output = _host_command(["dpkg-query", "-W", "-f=${Version}", "nvidia-jetpack"])
    match = _DPKG_VERSION_RE.match(output.strip()) if output else None
    return match.group(1) if match else None


def _nvml_max_clocks() -> tuple[int | None, int | None] | None:
    """(max SM clock, max memory clock) in Hz of NVML device 0; None when NVML is unusable.

    A clock NVML cannot report is None on its own, so the caller can fall back per clock.
    """
    try:
        import pynvml  # nvidia-ml-py: optional, dGPU hosts only
    except Exception:  # absent (ImportError) or broken at import time: not importable, so no NVML
        return None
    try:
        pynvml.nvmlInit()
    except Exception:  # NVMLError (no driver/library) or a broken binding: treat as no NVML
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return (
            _nvml_max_clock_hz(pynvml, handle, "NVML_CLOCK_SM"),
            _nvml_max_clock_hz(pynvml, handle, "NVML_CLOCK_MEM"),
        )
    except Exception:
        return None
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()


def _nvml_max_clock_hz(nvml: Any, handle: Any, clock_type: str) -> int | None:
    try:
        mhz = nvml.nvmlDeviceGetMaxClockInfo(handle, getattr(nvml, clock_type))
    except Exception:  # e.g. NVML_ERROR_NOT_SUPPORTED on some boards
        return None
    return _positive_int(mhz, 1_000_000)


def _json_objects(stdout: str | None) -> list[dict]:
    """Every JSON object line in the worker's stdout (other lines are ignored)."""
    objects: list[dict] = []
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            objects.append(value)
    return objects


def _with_stderr(summary: str, stderr: str | bytes | None) -> str:
    """``summary`` followed by the last non-empty lines of the worker's stderr."""
    if isinstance(stderr, bytes):  # TimeoutExpired carries bytes even in text mode
        stderr = stderr.decode("utf-8", "replace")
    lines = [line for line in (stderr or "").splitlines() if line.strip()]
    if not lines:
        return summary
    tail = "\n".join(lines[-_STDERR_TAIL_LINES:])[-_STDERR_TAIL_CHARS:]
    return f"{summary}\nworker stderr (last lines):\n{tail}"


def _describe_exit(returncode: int) -> str:
    if returncode < 0:
        try:
            name = signal.Signals(-returncode).name
        except ValueError:
            name = str(-returncode)
        return f"was killed by signal {name}"
    return f"exited with status {returncode}"


__all__ = [
    "ENERGY_RAIL_PREFERENCE",
    "FP16_FLOPS_PER_CLOCK_PER_SM",
    "Platform",
    "PlatformError",
    "PowerRail",
    "probe",
    "probe_gpu",
    "refresh_clocks",
]
