"""Background sampling of board power and memory, and energy integration over time windows.

All timestamps are ``time.monotonic_ns()`` (CLOCK_MONOTONIC), the same clock the worker uses
for token timestamps (std::chrono::steady_clock), so request/step windows from the worker can
be integrated directly.

Power: Jetson rails are read from INA3221/INA238 hwmon files (one read costs ~0.6 ms of I2C
time; the INA3221 averages 512 conversions, so readings lag the real power by ~0.3 s, see
docs/METRICS.md), so sampling runs on its own thread at ``power_hz``;
nothing here is ever called on the decode path. On discrete GPUs the NVML total-energy counter
(mJ) is sampled instead.

Memory: MemAvailable / SwapFree from /proc/meminfo and the worker's RSS from /proc/<pid>/statm.
On Jetson, GPU allocations are system RAM that RSS does not include, so these are system-level
figures; callers label them as such.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from array import array
from bisect import bisect_left, bisect_right
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from edgeserve.hardware import Platform

_log = logging.getLogger(__name__)

# How far a query may reach past the first/last power sample; power is held at the edge value.
_EDGE_TOLERANCE_NS = 500_000_000
# Expired power samples are dropped logically at once and physically in chunks of this many.
_COMPACT_AT = 4096
_WARN_INTERVAL_NS = 60_000_000_000
_JOIN_TIMEOUT_S = 5.0
# Periods and windows are capped (~146 years) so tiny rates / huge windows stay valid integers.
_MAX_NS = 1 << 62
# Longest single sleep of the sampler: Event.wait raises OverflowError past threading.TIMEOUT_MAX
# (~292 years on Linux, which _MAX_NS stays under, but ~49 days on Windows).
_MAX_WAIT_S = 60.0
_SELF_STATM = Path("/proc/self/statm")


@dataclass(frozen=True)
class MemorySample:
    t_ns: int
    mem_available_bytes: int
    swap_free_bytes: int
    worker_rss_bytes: int | None  # None until a worker pid is set or when unreadable
    frontend_rss_bytes: int


class Telemetry:
    """Samples power at ``power_hz`` and memory at ``memory_hz`` on a daemon thread.

    Samples older than ``window_s`` are discarded. All query methods are thread-safe and
    return None when no data covers the requested window (e.g. no power sensor).

    A rate of 0 disables that sampler. The energy source is ``platform.energy_rail`` when set
    (whole-module input power), else NVML device ``platform.nvml_index``.
    """

    def __init__(
        self,
        platform: Platform,
        power_hz: float = 20.0,
        memory_hz: float = 2.0,
        window_s: float = 1800.0,
        sysroot: Path = Path("/"),
    ) -> None:
        self._power_period_ns = _period_ns("power_hz", power_hz)
        self._memory_period_ns = _period_ns("memory_hz", memory_hz)
        if not (math.isfinite(window_s) and window_s > 0):
            raise ValueError(f"window_s must be a positive finite number, got {window_s!r}")
        self._window_ns = _whole_ns(window_s * 1e9)
        self._sysroot = Path(sysroot)
        self._page_size = _page_size()

        self._rail = platform.energy_rail
        self._nvml_index = platform.nvml_index if self._rail is None else None
        self._nvml: Any = None  # the pynvml module while initialised
        self._nvml_handle: Any = None
        self._nvml_failed = False
        self._nvml_mode: str | None = None  # "counter" | "power", decided once on first init
        # True when _e is a sampled energy counter (linear between samples) rather than integrated power.
        self._counter_mode = False

        self._lock = threading.Lock()
        # Power series, live from index _head: timestamps, watts (counter mode: mean power of the
        # interval ending at the sample) and cumulative joules since the first sample.
        self._t = array("q")
        self._w = array("d")
        self._e = array("d")
        self._head = 0
        self._energy_total = 0.0
        self._last_counter_j: float | None = None
        self._mem_t: list[int] = []
        self._mem: list[MemorySample] = []
        self._worker_pid: int | None = None

        self._state_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._lingering: list[threading.Thread] = []  # stopped threads that outlived their join
        self._stop_event = threading.Event()
        self._last_warning_ns: dict[str, int] = {}

    @property
    def energy_source(self) -> str | None:
        """Name of what energy is measured from: the Jetson rail name (e.g. "VDD_IN"), "nvml", or None."""
        if self._power_period_ns is None:  # power_hz=0: nothing is measured
            return None
        if self._rail is not None:
            return self._rail.name
        if self._nvml_index is not None and not self._nvml_failed:
            return "nvml"
        return None

    def start(self) -> None:
        """Start the sampler thread (idempotent)."""
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            schedule: list[tuple[str, int, Callable[[], object]]] = []
            if self._power_period_ns is not None:
                if self._nvml_index is not None:
                    self._ensure_nvml()
                if self.energy_source is not None:
                    schedule.append(("power", self._power_period_ns, self._sample_power))
            if self._memory_period_ns is not None:
                schedule.append(("memory", self._memory_period_ns, self._sample_memory))
            if not schedule:
                return
            # A fresh Event per thread: a thread that outlived a timed-out join still sees its own stop.
            self._stop_event = threading.Event()
            thread = threading.Thread(
                target=self._run, args=(self._stop_event, schedule), name="edgeserve-telemetry", daemon=True
            )
            thread.start()  # kept only once running: stop() cannot join a thread that never started
            self._thread = thread

    def stop(self) -> None:
        """Stop and join the sampler thread (idempotent)."""
        with self._state_lock:
            thread, self._thread = self._thread, None
            self._stop_event.set()
            if thread is not None:
                thread.join(_JOIN_TIMEOUT_S)
                if thread.is_alive():
                    _log.warning("telemetry sampler did not stop within %g s (blocked sensor read?)", _JOIN_TIMEOUT_S)
                    self._lingering.append(thread)
            self._lingering = [t for t in self._lingering if t.is_alive()]
            # Shutting NVML down under a call still in flight on a lingering thread is unsafe.
            if not self._lingering:
                self._shutdown_nvml()

    def set_worker_pid(self, pid: int | None) -> None:
        """Process whose RSS is reported as worker_rss_bytes from the next memory sample on (None: no worker)."""
        self._worker_pid = None if pid is None else int(pid)

    def energy_j(self, t0_ns: int, t1_ns: int) -> float | None:
        """Energy over [t0, t1] in joules.

        Power source: trapezoid integration of power samples, linearly interpolating power at
        t0 and t1 (clamp to the nearest sample when the window extends past the data by less
        than 0.5 s; return None when it extends further or there are fewer than 2 samples).
        NVML source: difference of the interpolated cumulative energy counter.

        Clamping holds the edge sample's power over the uncovered part, so the window keeps its
        full duration. Up to 0.5 s inclusive is covered. A reversed window (t1 < t0) gives None.
        """
        if t1_ns < t0_ns:
            return None
        with self._lock:
            start = self._locate(t0_ns)
            end = self._locate(t1_ns) if start is not None else None
        if start is None or end is None:
            return None
        return _energy_between(start, end)

    def energy_series_j(self, boundaries_ns: list[int]) -> list[float | None]:
        """Energy of each consecutive window [b0,b1], [b1,b2], ... in one pass (len = len(boundaries) - 1).

        Each element equals ``energy_j(b[k], b[k+1])``; O(k log n) for k boundaries and n samples.
        """
        if len(boundaries_ns) < 2:
            return []
        with self._lock:
            points = [self._locate(t) for t in boundaries_ns]
        out: list[float | None] = []
        for k in range(len(points) - 1):
            start, end = points[k], points[k + 1]
            if start is None or end is None or boundaries_ns[k + 1] < boundaries_ns[k]:
                out.append(None)
            else:
                out.append(_energy_between(start, end))
        return out

    def mean_power_w(self, t0_ns: int, t1_ns: int) -> float | None:
        """energy_j(t0, t1) / duration; None when energy is None or the window is empty."""
        if t1_ns <= t0_ns:
            return None
        energy = self.energy_j(t0_ns, t1_ns)
        return None if energy is None else energy * 1e9 / (t1_ns - t0_ns)

    def energy_total_j(self) -> float:
        """Cumulative energy since start() (monotonic counter for Prometheus); 0.0 without a source."""
        with self._lock:
            return self._energy_total

    def memory(self) -> MemorySample | None:
        """Latest memory sample (None before the first sample)."""
        with self._lock:
            return self._mem[-1] if self._mem else None

    def min_mem_available(self, t0_ns: int, t1_ns: int) -> int | None:
        """Lowest MemAvailable among samples in [t0, t1] (or the latest sample before t0 if none inside)."""
        samples = self._memory_window(t0_ns, t1_ns)
        return min(s.mem_available_bytes for s in samples) if samples else None

    def peak_worker_rss(self, t0_ns: int, t1_ns: int) -> int | None:
        """Highest worker RSS among samples in [t0, t1] (same fallback as min_mem_available).

        Samples without a worker RSS are skipped; None when none of the chosen samples has one.
        """
        values = [s.worker_rss_bytes for s in self._memory_window(t0_ns, t1_ns) if s.worker_rss_bytes is not None]
        return max(values) if values else None

    # -- sampling ---------------------------------------------------------------------------

    def _run(self, stop: threading.Event, schedule: list[tuple[str, int, Callable[[], object]]]) -> None:
        deadlines = [time.monotonic_ns()] * len(schedule)
        while not stop.is_set():
            timeout = 1.0
            try:
                for k, (label, period, sample) in enumerate(schedule):
                    if time.monotonic_ns() >= deadlines[k]:
                        try:
                            sample()
                        except Exception as exc:
                            self._warn(label, "telemetry %s sampling failed: %r", label, exc)
                        deadlines[k] = _next_deadline(deadlines[k], period, time.monotonic_ns())
                timeout = min(_MAX_WAIT_S, max(0, min(deadlines) - time.monotonic_ns()) / 1e9)
            except Exception as exc:
                self._warn("sampler", "telemetry sampler error: %r", exc)
            stop.wait(timeout)

    def _sample_power(self, now_ns: int | None = None) -> float | None:
        """Read the energy source once and store the reading at ``now_ns`` (default: the middle of the read).

        Returns the stored reading (watts; joules for the NVML energy counter) or None when nothing was stored.
        """
        if self._rail is None and not self._ensure_nvml():
            return None
        counter = self._rail is None and self._nvml_mode == "counter"
        begin = time.monotonic_ns()
        error: object = "no value"
        value: float | None = None
        try:
            raw = self._rail.read_watts() if self._rail is not None else self._read_nvml(counter)
            if raw is not None:
                value = float(raw)
                if not math.isfinite(value):
                    value, error = None, f"non-finite value {raw!r}"
        except Exception as exc:  # pynvml raises NVMLError subclasses; float() a TypeError/ValueError
            value, error = None, exc
        t_ns = int(now_ns) if now_ns is not None else (begin + time.monotonic_ns()) // 2
        if value is None:
            source = self._rail.name if self._rail is not None else "nvml"
            self._warn("power-read", "could not read power from %s: %s", source, error)
            return None
        with self._lock:
            if counter:
                stored = self._record_counter(t_ns, value)
            else:
                # Input power cannot be negative; small negatives are shunt offset noise.
                value = max(0.0, value)
                stored = self._record_power(t_ns, value)
        return value if stored else None

    def _sample_memory(self, now_ns: int | None = None) -> MemorySample | None:
        """Take one memory sample stamped ``now_ns`` (default: now) and store it; None when meminfo is unreadable."""
        t_ns = int(now_ns) if now_ns is not None else time.monotonic_ns()
        meminfo_path = self._sysroot / "proc" / "meminfo"
        meminfo = _read_meminfo(meminfo_path)
        if meminfo is None:
            self._warn("meminfo", "could not read MemAvailable from %s", meminfo_path)
            return None
        pid = self._worker_pid
        worker_rss = None if pid is None else _read_rss(self._sysroot / "proc" / str(pid) / "statm", self._page_size)
        frontend_rss = _read_rss(_SELF_STATM, self._page_size)
        sample = MemorySample(
            t_ns=t_ns,
            mem_available_bytes=meminfo[0],
            swap_free_bytes=meminfo[1],
            worker_rss_bytes=worker_rss,
            frontend_rss_bytes=frontend_rss if frontend_rss is not None else 0,
        )
        with self._lock:
            if self._mem_t and t_ns <= self._mem_t[-1]:
                return None
            self._mem_t.append(t_ns)
            self._mem.append(sample)
            cutoff = t_ns - self._window_ns
            if self._mem_t[0] < cutoff:
                drop = bisect_left(self._mem_t, cutoff)
                del self._mem_t[:drop]
                del self._mem[:drop]
        return sample

    def _record_power(self, t_ns: int, watts: float) -> bool:
        """Append a power sample (lock held); False when its timestamp does not advance."""
        if len(self._t) > self._head:
            prev_t = self._t[-1]
            if t_ns <= prev_t:
                return False
            self._energy_total += (t_ns - prev_t) * (self._w[-1] + watts) / 2e9
        self._append(t_ns, watts)
        return True

    def _record_counter(self, t_ns: int, joules: float) -> bool:
        """Append an energy-counter sample (lock held); False when its timestamp does not advance."""
        watts = 0.0
        if len(self._t) > self._head:
            prev_t = self._t[-1]
            if t_ns <= prev_t:
                return False
            last = self._last_counter_j
            # A counter that went backwards was reset (driver reload): that interval's energy is unknown.
            increment = joules - last if last is not None and joules >= last else 0.0
            self._energy_total += increment
            watts = increment * 1e9 / (t_ns - prev_t)
        self._last_counter_j = joules
        self._append(t_ns, watts)
        return True

    def _append(self, t_ns: int, watts: float) -> None:
        self._t.append(t_ns)
        self._w.append(watts)
        self._e.append(self._energy_total)
        cutoff = t_ns - self._window_ns
        if self._t[self._head] < cutoff:
            self._head = bisect_left(self._t, cutoff, self._head)
            if self._head >= _COMPACT_AT:
                del self._t[: self._head]
                del self._w[: self._head]
                del self._e[: self._head]
                self._head = 0

    def _locate(self, t_ns: int) -> tuple[float, float] | None:
        """Cumulative energy at ``t_ns`` as (stored cumulative J of a sample, J from that sample to t_ns).

        Kept as two parts so differences of nearby times stay exact when the running total is large.
        None when the time is not covered. Caller holds the lock.
        """
        ts, head = self._t, self._head
        end = len(ts) - 1
        if end - head < 1:
            return None
        first, last = ts[head], ts[end]
        if not first - _EDGE_TOLERANCE_NS <= t_ns <= last + _EDGE_TOLERANCE_NS:  # also rejects NaN
            return None
        if t_ns <= first:
            # Counter mode: _w[head] describes an interval before the data; use the first interval's power.
            edge_w = self._w[head + 1] if self._counter_mode else self._w[head]
            return self._e[head], -(first - t_ns) * edge_w / 1e9
        if t_ns >= last:
            return self._e[end], (t_ns - last) * self._w[end] / 1e9
        i = bisect_right(ts, t_ns, head) - 1  # ts[i] <= t_ns < ts[i + 1]
        t_i, e_i = ts[i], self._e[i]
        dt, span = t_ns - t_i, ts[i + 1] - t_i
        if self._counter_mode:
            return e_i, (self._e[i + 1] - e_i) * dt / span
        w_i = self._w[i]
        w_t = w_i + (self._w[i + 1] - w_i) * dt / span
        return e_i, dt * (w_i + w_t) / 2e9

    def _memory_window(self, t0_ns: int, t1_ns: int) -> list[MemorySample]:
        """Samples in [t0, t1], else the latest one before t0, else []."""
        if not t0_ns <= t1_ns:  # reversed, or NaN
            return []
        with self._lock:
            lo = bisect_left(self._mem_t, t0_ns)
            hi = bisect_right(self._mem_t, t1_ns)
            if lo < hi:
                return self._mem[lo:hi]
            return [self._mem[lo - 1]] if lo > 0 else []

    # -- NVML -------------------------------------------------------------------------------

    def _ensure_nvml(self) -> bool:
        """Initialise NVML on first use; False (and energy_source None) when it is unavailable."""
        if self._nvml is not None:
            return True
        if self._nvml_failed or self._nvml_index is None:
            return False
        try:
            import pynvml

            pynvml.nvmlInit()
        except Exception as exc:
            self._nvml_failed = True
            _log.warning("NVML is unavailable (%r); energy will not be measured", exc)
            return False
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(self._nvml_index)
            if self._nvml_mode is None:
                self._nvml_mode = _probe_nvml_mode(pynvml, handle)
                self._counter_mode = self._nvml_mode == "counter"
        except Exception as exc:
            self._nvml_failed = True
            _log.warning(
                "NVML device %s has no power readings (%r); energy will not be measured", self._nvml_index, exc
            )
            _shutdown_quietly(pynvml)
            return False
        self._nvml, self._nvml_handle = pynvml, handle
        return True

    def _read_nvml(self, counter: bool) -> float:
        if counter:
            return self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._nvml_handle) / 1000.0  # mJ -> J
        return self._nvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0  # mW -> W

    def _shutdown_nvml(self) -> None:
        nvml, self._nvml, self._nvml_handle = self._nvml, None, None
        if nvml is not None:
            _shutdown_quietly(nvml)

    def _warn(self, key: str, msg: str, *args: object) -> None:
        """Log a warning at most once per minute per key (the sampler can fail 20 times a second)."""
        now = time.monotonic_ns()
        last = self._last_warning_ns.get(key)
        if last is not None and now - last < _WARN_INTERVAL_NS:
            return
        self._last_warning_ns[key] = now
        _log.warning(msg, *args)


def _energy_between(start: tuple[float, float], end: tuple[float, float]) -> float:
    # Clamped at 0: rounding can leave -1e-17 for a zero-power window.
    return max(0.0, (end[0] - start[0]) + (end[1] - start[1]))


def _period_ns(name: str, hz: float) -> int | None:
    if not (math.isfinite(hz) and hz >= 0):
        raise ValueError(f"{name} must be a finite number >= 0 (0 disables sampling), got {hz!r}")
    return None if hz == 0 else _whole_ns(1e9 / hz)


def _whole_ns(ns: float) -> int:
    """A positive duration rounded to whole nanoseconds and clamped to [1, _MAX_NS] (1e9 / 5e-324 is inf)."""
    return _MAX_NS if not ns < _MAX_NS else max(1, round(ns))


def _next_deadline(deadline: int, period: int, now: int) -> int:
    """Next tick of a fixed-rate schedule that is after ``now`` (missed ticks are skipped, not bunched)."""
    deadline += period
    if deadline <= now:
        deadline += ((now - deadline) // period + 1) * period
    return deadline


def _page_size() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return 4096


def _read_meminfo(path: Path) -> tuple[int, int] | None:
    """(MemAvailable, SwapFree) in bytes; None when MemAvailable is missing or malformed.

    A missing or malformed SwapFree counts as 0 (no swap) so it cannot cost the MemAvailable reading.
    """
    try:
        text = path.read_text(encoding="ascii", errors="replace")
    except OSError:
        return None
    values: dict[str, int | None] = {}
    for line in text.splitlines():
        key, sep, rest = line.partition(":")
        if sep and key in ("MemAvailable", "SwapFree"):
            values[key] = _meminfo_bytes(rest)
    available = values.get("MemAvailable")
    if available is None:
        return None
    return available, values.get("SwapFree") or 0


def _meminfo_bytes(field: str) -> int | None:
    """Bytes of a meminfo value such as " 5150980 kB" (no unit: bytes); None when malformed or negative."""
    parts = field.split()
    try:
        value = int(parts[0])
    except (IndexError, ValueError):
        return None
    if value < 0:
        return None
    return value * 1024 if parts[1:2] == ["kB"] else value


def _read_rss(statm: Path, page_size: int) -> int | None:
    """Resident set size in bytes from a /proc/<pid>/statm file (2nd field, in pages)."""
    try:
        pages = int(statm.read_text(encoding="ascii", errors="replace").split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return pages * page_size if pages >= 0 else None


def _probe_nvml_mode(pynvml: Any, handle: Any) -> str:
    """"counter" when the total-energy counter works (Volta+), else "power" (raises if neither does)."""
    try:
        pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        return "counter"
    except Exception as exc:
        _log.info("NVML energy counter unavailable (%r); integrating nvmlDeviceGetPowerUsage instead", exc)
    pynvml.nvmlDeviceGetPowerUsage(handle)
    return "power"


def _shutdown_quietly(pynvml: Any) -> None:
    try:
        pynvml.nvmlShutdown()
    except Exception as exc:
        _log.debug("nvmlShutdown failed: %r", exc)


__all__ = ["MemorySample", "Telemetry"]
