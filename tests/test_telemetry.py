"""Tests for edgeserve.telemetry: integration math on injected samples, memory parsing, the sampler thread, NVML."""

from __future__ import annotations

import logging
import math
import os
import random
import sys
import threading
import time
import types
from bisect import bisect_right
from collections.abc import Callable, Iterator
from fractions import Fraction
from pathlib import Path

import pytest

import edgeserve.telemetry as telemetry_module
from edgeserve.hardware import Platform, PowerRail
from edgeserve.telemetry import MemorySample, Telemetry

S = 1_000_000_000
MS = 1_000_000
T0 = 1_000 * S  # arbitrary monotonic origin for injected samples
MB = 2**20
PAGE = os.sysconf("SC_PAGE_SIZE")


def make_platform(energy_rail: object = None, nvml_index: int | None = None) -> Platform:
    return Platform(
        kind="jetson" if nvml_index is None else "dgpu",
        hostname="test-host",
        board=None,
        l4t=None,
        jetpack=None,
        gpu_name="Orin",
        sm=87,
        sm_count=8,
        integrated=nvml_index is None,
        total_memory_bytes=8 * 2**30,
        memory_bus_width_bits=128,
        cuda_runtime="12.6",
        cuda_driver="12.6",
        tensorrt="10.3.0.30",
        edgellm="0.11.0",
        power_mode=None,
        gpu_max_freq_hz=None,
        emc_max_freq_hz=None,
        peak_flops_per_s=None,
        peak_mem_bw_bytes_per_s=None,
        power_rails=() if energy_rail is None else (energy_rail,),  # type: ignore[arg-type]
        energy_rail=energy_rail,  # type: ignore[arg-type]
        nvml_index=nvml_index,
        gpu_freq_path=None,
        emc_freq_path=None,
    )


class Ina3221:
    """An INA3221 hwmon channel in tmp_path; set() writes a 5 V bus and the current that gives ``watts``."""

    def __init__(self, root: Path, name: str = "VDD_IN") -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.millivolts = root / "in1_input"
        self.milliamps = root / "curr1_input"
        self.rail = PowerRail(name, "ina3221", (str(self.millivolts), str(self.milliamps)))
        self.set(0.0)

    def set(self, watts: float) -> None:
        self.millivolts.write_text("5000\n")
        self.milliamps.write_text(f"{watts * 200:.6f}\n")  # 5000 mV x 200*W mA / 1e6 = W


class FastRail:
    """Duck-typed rail without file I/O, for injecting tens of thousands of samples quickly."""

    name = "VDD_IN"

    def __init__(self, watts: float = 0.0) -> None:
        self.watts = watts

    def read_watts(self) -> float | None:
        return self.watts


class CountingRail(FastRail):
    """A FastRail that counts its reads."""

    def __init__(self, watts: float = 0.0) -> None:
        super().__init__(watts)
        self.reads = 0

    def read_watts(self) -> float | None:
        self.reads += 1
        return self.watts


@pytest.fixture
def sensor(tmp_path: Path) -> Ina3221:
    return Ina3221(tmp_path / "hwmon1")


def feed(tel: Telemetry, sensor: Ina3221, samples: list[tuple[int, float]]) -> None:
    """Store (t_ns, watts) samples through the real rail-reading path."""
    for t_ns, watts in samples:
        sensor.set(watts)
        assert tel._sample_power(t_ns) == pytest.approx(watts)


def write_meminfo(sysroot: Path, available_kb: int | None = 4_000_000, swap_free_kb: int | None = 1_000_000) -> None:
    lines = ["MemTotal:        7802776 kB", "MemFree:         4381520 kB"]
    if available_kb is not None:
        lines.append(f"MemAvailable:   {available_kb} kB")
    lines += ["Buffers:           42952 kB", "SwapTotal:       3901368 kB"]
    if swap_free_kb is not None:
        lines.append(f"SwapFree:       {swap_free_kb} kB")
    lines.append("HugePages_Total:       0")
    path = sysroot / "proc" / "meminfo"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_statm(sysroot: Path, pid: int, resident_pages: int) -> Path:
    path = sysroot / "proc" / str(pid) / "statm"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{resident_pages * 4} {resident_pages} 3000 10 0 50000 0\n")
    return path


def wait_for(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not reached in time"
        time.sleep(0.005)


# -- construction ---------------------------------------------------------------------------


def test_energy_source_names(sensor: Ina3221) -> None:
    assert Telemetry(make_platform(sensor.rail)).energy_source == "VDD_IN"
    assert Telemetry(make_platform(nvml_index=0)).energy_source == "nvml"
    assert Telemetry(make_platform()).energy_source is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"power_hz": -1.0},
        {"power_hz": float("inf")},
        {"memory_hz": float("nan")},
        {"window_s": 0.0},
        {"window_s": -5.0},
        {"window_s": float("nan")},
        {"window_s": float("inf")},
    ],
)
def test_invalid_settings_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        Telemetry(make_platform(), **kwargs)


def test_tiny_rates_and_huge_windows_are_valid(
    sensor: Ina3221, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: these raised OverflowError (1e9 / 5e-324 and 1e300 * 1e9 are inf), and a period longer than
    # threading.TIMEOUT_MAX killed the sampler thread in Event.wait().
    thread_errors: list[BaseException | None] = []
    monkeypatch.setattr(threading, "excepthook", lambda args: thread_errors.append(args.exc_value))
    write_meminfo(tmp_path)
    tel = Telemetry(make_platform(sensor.rail), power_hz=5e-324, memory_hz=1e-12, window_s=1e300, sysroot=tmp_path)
    assert Telemetry(make_platform(), power_hz=1e12)._power_period_ns == 1
    tel.start()
    try:
        wait_for(lambda: tel.memory() is not None and len(tel._t) == 1)  # each sampler's first tick is immediate
        time.sleep(0.05)
        assert tel._thread is not None and tel._thread.is_alive()
    finally:
        began = time.monotonic()
        tel.stop()
    assert time.monotonic() - began < 1.0
    assert thread_errors == []

    rail = FastRail(2.0)
    unbounded = Telemetry(make_platform(rail), window_s=1e300)
    for hours in range(3):
        unbounded._sample_power(T0 + hours * 3600 * S)
    assert unbounded.energy_j(T0, T0 + 7200 * S) == pytest.approx(2.0 * 7200)  # nothing was trimmed


def test_sampler_sleeps_are_capped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A period past threading.TIMEOUT_MAX (reachable on platforms with a smaller limit) must not reach Event.wait().
    monkeypatch.setattr(telemetry_module, "_MAX_NS", 1 << 80)
    thread_errors: list[BaseException | None] = []
    monkeypatch.setattr(threading, "excepthook", lambda args: thread_errors.append(args.exc_value))
    write_meminfo(tmp_path)
    tel = Telemetry(make_platform(), memory_hz=1e-12, sysroot=tmp_path)  # a 1e12 s period
    tel.start()
    try:
        wait_for(lambda: tel.memory() is not None)
        time.sleep(0.05)
        assert tel._thread is not None and tel._thread.is_alive()
    finally:
        tel.stop()
    assert thread_errors == []


def test_no_energy_source() -> None:
    tel = Telemetry(make_platform())
    assert tel._sample_power(T0) is None
    assert tel.energy_j(T0, T0 + S) is None
    assert tel.energy_series_j([T0, T0 + S]) == [None]
    assert tel.mean_power_w(T0, T0 + S) is None
    assert tel.energy_total_j() == 0.0


# -- power integration ----------------------------------------------------------------------


def test_constant_power(sensor: Ina3221) -> None:
    tel = Telemetry(make_platform(sensor.rail))
    feed(tel, sensor, [(T0 + i * 50 * MS, 5.0) for i in range(21)])  # 1 s at 20 Hz
    assert tel.energy_j(T0, T0 + S) == pytest.approx(5.0)
    assert tel.energy_j(T0 + 10 * MS, T0 + 30 * MS) == pytest.approx(0.1)  # between two samples
    assert tel.energy_j(T0 + 30 * MS, T0 + 730 * MS) == pytest.approx(3.5)
    assert tel.mean_power_w(T0 + 123 * MS, T0 + 987 * MS) == pytest.approx(5.0)
    assert tel.energy_total_j() == pytest.approx(5.0)


def ramp_energy(a_ns: int, b_ns: int) -> float:
    """Exact energy of P(t) = 1 + 2t W (t in s after T0): integral = t + t^2."""
    a, b = (a_ns - T0) / S, (b_ns - T0) / S
    return (b + b * b) - (a + a * a)


@pytest.mark.parametrize(
    "a_ms, b_ms",
    [(0, 2000), (30, 70), (50, 1234), (100, 200), (1990, 2000), (333, 1667), (700, 700)],
)
def test_linear_ramp_is_integrated_exactly(sensor: Ina3221, a_ms: int, b_ms: int) -> None:
    tel = Telemetry(make_platform(sensor.rail))
    feed(tel, sensor, [(T0 + i * 100 * MS, 1.0 + 2.0 * (i / 10)) for i in range(21)])
    a, b = T0 + a_ms * MS, T0 + b_ms * MS
    assert tel.energy_j(a, b) == pytest.approx(ramp_energy(a, b), rel=1e-9, abs=1e-12)


def held_linear(ts: list[int], values: list[Fraction], t: int) -> Fraction:
    """values linearly interpolated at t, held at the first/last value outside the samples."""
    if t <= ts[0]:
        return values[0]
    if t >= ts[-1]:
        return values[-1]
    k = bisect_right(ts, t) - 1
    return values[k] + (values[k + 1] - values[k]) * Fraction(t - ts[k], ts[k + 1] - ts[k])


def piecewise_energy(ts: list[int], watts: list[float], a: int, b: int) -> float:
    """Exact energy (J) over [a, b] of power that is linear between samples and held at the edge samples."""
    power = [Fraction(w) for w in watts]
    points = sorted({a, b, *(t for t in ts if a < t < b)})
    total = sum(
        (held_linear(ts, power, x) + held_linear(ts, power, y)) * (y - x) / 2 for x, y in zip(points, points[1:])
    )
    return float(Fraction(total) / S)


@pytest.mark.parametrize("window_s", [1800.0, 1.5])
def test_energy_matches_an_independent_integrator(window_s: float) -> None:
    rng = random.Random(11)
    for _ in range(30):
        rail = FastRail()
        tel = Telemetry(make_platform(rail), window_s=window_s)
        ts, watts, t = [], [], T0
        for _ in range(rng.randint(2, 30)):
            t += rng.randint(1, 300 * MS)
            rail.watts = rng.choice([0.0, -0.2, rng.uniform(0.0, 25.0)])
            assert tel._sample_power(t) is not None
            ts.append(t)
            watts.append(max(0.0, rail.watts))
        live = [k for k, t_k in enumerate(ts) if t_k >= ts[-1] - int(window_s * S)]  # trimming keeps these
        live_t, live_w = [ts[k] for k in live], [watts[k] for k in live]
        for _ in range(30):
            a = rng.randint(live_t[0] - 700 * MS, live_t[-1] + 700 * MS)
            b = rng.choice([a, rng.randint(a, live_t[-1] + 700 * MS)])
            got = tel.energy_j(a, b)
            if len(live_t) < 2 or a < live_t[0] - 500 * MS or b > live_t[-1] + 500 * MS:
                assert got is None
            else:
                assert got == pytest.approx(piecewise_energy(live_t, live_w, a, b), rel=1e-9, abs=1e-12)


def test_windows_past_the_data_hold_the_edge_power(sensor: Ina3221) -> None:
    tel = Telemetry(make_platform(sensor.rail))
    feed(tel, sensor, [(T0 + S, 1.0), (T0 + 2 * S, 3.0)])
    assert tel.energy_j(T0 + S, T0 + 2 * S) == pytest.approx(2.0)
    assert tel.energy_j(T0 + 1250 * MS, T0 + 1750 * MS) == pytest.approx(1.0)  # interpolated 1.5 W .. 2.5 W
    assert tel.energy_j(T0 + 600 * MS, T0 + S) == pytest.approx(0.4)  # 0.4 s before the data at 1 W
    assert tel.energy_j(T0 + 600 * MS, T0 + 2 * S) == pytest.approx(2.4)
    assert tel.energy_j(T0 + 500 * MS, T0 + S) == pytest.approx(0.5)  # exactly 0.5 s is still covered
    assert tel.energy_j(T0 + 2 * S, T0 + 2500 * MS) == pytest.approx(1.5)  # 0.5 s after at 3 W
    assert tel.energy_j(T0 + 2100 * MS, T0 + 2300 * MS) == pytest.approx(0.6)  # wholly after the data
    assert tel.mean_power_w(T0 + 2 * S, T0 + 2500 * MS) == pytest.approx(3.0)
    assert tel.energy_j(T0 + 2 * S, T0 + 2500 * MS + 1) is None
    assert tel.energy_j(T0 + 500 * MS - 1, T0 + 1500 * MS) is None
    assert tel.mean_power_w(T0, T0 + 1500 * MS) is None


def test_fewer_than_two_samples_give_none(sensor: Ina3221) -> None:
    tel = Telemetry(make_platform(sensor.rail))
    assert tel.energy_j(T0, T0 + S) is None
    assert tel.energy_series_j([T0, T0 + S, T0 + 2 * S]) == [None, None]
    feed(tel, sensor, [(T0, 5.0)])
    assert tel.energy_j(T0, T0) is None
    assert tel.energy_j(T0 - MS, T0 + MS) is None
    assert tel.mean_power_w(T0 - MS, T0 + MS) is None
    assert tel.energy_total_j() == 0.0
    feed(tel, sensor, [(T0 + S, 5.0)])
    assert tel.energy_j(T0, T0) == 0.0
    assert tel.energy_j(T0, T0 + S) == pytest.approx(5.0)


def test_empty_and_reversed_windows(sensor: Ina3221) -> None:
    tel = Telemetry(make_platform(sensor.rail))
    feed(tel, sensor, [(T0, 2.0), (T0 + S, 2.0)])
    mid = T0 + S // 2
    assert tel.energy_j(mid, mid) == 0.0
    assert tel.mean_power_w(mid, mid) is None
    assert tel.energy_j(T0 + S, T0) is None
    assert tel.mean_power_w(T0 + S, T0) is None


def test_nan_and_infinite_bounds_are_not_covered(sensor: Ina3221) -> None:
    # Regression: NaN fails every comparison, so bisect ran off the end of the data (IndexError).
    tel = Telemetry(make_platform(sensor.rail))
    feed(tel, sensor, [(T0, 2.0), (T0 + S, 2.0)])
    nan, inf = float("nan"), float("inf")
    for t0, t1 in [(nan, nan), (T0, nan), (nan, T0 + S), (-inf, T0 + S), (T0, inf), (-inf, inf)]:
        assert tel.energy_j(t0, t1) is None  # type: ignore[arg-type]
        assert tel.mean_power_w(t0, t1) is None  # type: ignore[arg-type]
    assert tel.energy_series_j([T0, nan, T0 + S, T0 + S // 2]) == [None, None, None]  # type: ignore[list-item]
    assert tel.energy_series_j([T0, T0 + S // 2, nan]) == [pytest.approx(1.0), None]  # type: ignore[list-item]


def test_energy_series_matches_energy_j_per_window(sensor: Ina3221) -> None:
    rng = random.Random(7)
    tel = Telemetry(make_platform(sensor.rail))
    samples, t = [], T0
    for _ in range(120):
        samples.append((t, round(rng.uniform(3.0, 12.0), 3)))
        t += rng.randint(20 * MS, 80 * MS)
    feed(tel, sensor, samples)
    first, last = samples[0][0], samples[-1][0]
    boundaries = sorted(rng.randint(first - 800 * MS, last + 800 * MS) for _ in range(400))
    boundaries[50:50] = [boundaries[50]] * 2  # repeated boundaries make zero-length windows
    boundaries = [first - 2 * S, *boundaries, last + 2 * S]

    series = tel.energy_series_j(boundaries)

    assert len(series) == len(boundaries) - 1
    assert series[0] is None and series[-1] is None
    for k, energy in enumerate(series):
        expected = tel.energy_j(boundaries[k], boundaries[k + 1])
        if expected is None:
            assert energy is None
        else:
            assert energy == pytest.approx(expected, rel=1e-12, abs=1e-12)
    covered = [k for k, energy in enumerate(series) if energy is not None]
    assert covered == list(range(covered[0], covered[-1] + 1))
    whole = tel.energy_j(boundaries[covered[0]], boundaries[covered[-1] + 1])
    assert sum(series[k] for k in covered) == pytest.approx(whole, rel=1e-9)  # type: ignore[misc]


def test_energy_series_short_and_unsorted_input(sensor: Ina3221) -> None:
    tel = Telemetry(make_platform(sensor.rail))
    feed(tel, sensor, [(T0 + i * 100 * MS, 4.0) for i in range(11)])
    assert tel.energy_series_j([]) == []
    assert tel.energy_series_j([T0]) == []
    series = tel.energy_series_j([T0 + 500 * MS, T0 + 200 * MS, T0 + 700 * MS])
    assert series[0] is None  # reversed window
    assert series[1] == pytest.approx(2.0)


def test_energy_series_is_fast_on_a_full_window() -> None:
    rail = FastRail()
    tel = Telemetry(make_platform(rail), window_s=1800.0)
    for i in range(36_000):  # 30 min at 20 Hz
        rail.watts = 5.0 + (i % 40) * 0.25
        tel._sample_power(T0 + i * 50 * MS)
    start = T0 + 1000 * S
    boundaries = [start + k * 37 * MS for k in range(513)]

    began = time.perf_counter()
    series = tel.energy_series_j(boundaries)
    elapsed = time.perf_counter() - began

    assert len(series) == 512 and all(e is not None and e > 0 for e in series)
    assert sum(series) == pytest.approx(tel.energy_j(boundaries[0], boundaries[-1]), rel=1e-9)  # type: ignore[arg-type]
    assert elapsed < 0.5  # O(k log n): ~1 ms; a per-window scan would take tens of seconds


def test_total_energy_is_not_limited_by_the_window() -> None:
    rail = FastRail(3.0)
    tel = Telemetry(make_platform(rail), window_s=10.0)
    for i in range(601):  # 30 s at 20 Hz
        tel._sample_power(T0 + i * 50 * MS)
    last = T0 + 30 * S
    assert tel.energy_total_j() == pytest.approx(90.0)
    assert tel.energy_j(last - 10 * S, last) == pytest.approx(30.0)
    assert tel.energy_j(last - 10_500 * MS, last) == pytest.approx(31.5)
    assert tel.energy_j(last - 10_600 * MS, last) is None
    assert tel.energy_j(T0, T0 + S) is None


def test_the_power_sample_exactly_window_s_old_is_kept() -> None:
    # Irregular gaps: on a regular grid the live head is always one sample ahead, which hides an off-by-one.
    tel = Telemetry(make_platform(FastRail(2.0)), window_s=10.0)
    for s in (0, 5, 10, 20):
        tel._sample_power(T0 + s * S)
    assert tel.energy_j(T0 + 10 * S, T0 + 20 * S) == pytest.approx(20.0)  # the 10 s sample is exactly 10 s old
    assert tel.energy_j(T0 + 9 * S, T0 + 20 * S) is None  # the 5 s sample is gone: 1 s before the data


def test_trimmed_storage_stays_bounded() -> None:
    rail = FastRail(2.0)
    tel = Telemetry(make_platform(rail), window_s=1.0)
    for i in range(10_000):  # crosses several compactions
        tel._sample_power(T0 + i * 50 * MS)
    last = T0 + 9_999 * 50 * MS
    assert len(tel._t) - tel._head == 21  # samples no older than window_s
    assert len(tel._t) <= 4096 + 21
    assert tel.energy_j(last - S, last) == pytest.approx(2.0)
    assert tel.energy_j(last - 1600 * MS, last) is None
    assert tel.energy_total_j() == pytest.approx(2.0 * 9_999 * 0.05)


def test_non_increasing_timestamps_are_dropped(sensor: Ina3221) -> None:
    tel = Telemetry(make_platform(sensor.rail))
    feed(tel, sensor, [(T0, 5.0), (T0 + S, 5.0)])
    sensor.set(100.0)
    assert tel._sample_power(T0 + S) is None
    assert tel._sample_power(T0 + S // 2) is None
    feed(tel, sensor, [(T0 + 2 * S, 5.0)])
    assert tel.energy_j(T0, T0 + 2 * S) == pytest.approx(10.0)
    assert tel.energy_total_j() == pytest.approx(10.0)


def test_bad_rail_readings_are_skipped_and_logged_once(sensor: Ina3221, caplog: pytest.LogCaptureFixture) -> None:
    tel = Telemetry(make_platform(sensor.rail))
    sensor.milliamps.write_text("-20\n")  # -0.1 W of shunt offset: input power cannot be negative
    assert tel._sample_power(T0) == 0.0
    with caplog.at_level(logging.WARNING, logger="edgeserve.telemetry"):
        for bad in ("nan\n", "inf\n", "garbage\n", None):
            if bad is None:
                sensor.milliamps.unlink()
            else:
                sensor.milliamps.write_text(bad)
            assert tel._sample_power(T0 + S) is None
    assert len([r for r in caplog.records if "could not read power" in r.getMessage()]) == 1
    sensor.set(4.0)
    assert tel._sample_power(T0 + S) == pytest.approx(4.0)
    assert tel.energy_j(T0, T0 + S) == pytest.approx(2.0)  # trapezoid 0 W -> 4 W


@pytest.mark.parametrize("reading", ["n/a", object(), [1.0], float("-inf")])
def test_non_numeric_readings_are_skipped(reading: object) -> None:
    # Regression: math.isfinite() raised TypeError out of _sample_power for anything but a number.
    rail = FastRail()
    tel = Telemetry(make_platform(rail))
    rail.watts = reading  # type: ignore[assignment]
    assert tel._sample_power(T0) is None
    assert len(tel._t) == 0 and tel.energy_total_j() == 0.0


def test_ina238_rail(tmp_path: Path) -> None:
    power = tmp_path / "power1_input"
    tel = Telemetry(make_platform(PowerRail("VIN", "ina238", (str(power),))))
    assert tel.energy_source == "VIN"
    for i, microwatts in enumerate((7_500_000, 7_500_000, 12_500_000)):
        power.write_text(f"{microwatts}\n")
        tel._sample_power(T0 + i * S)
    assert tel.energy_j(T0, T0 + 2 * S) == pytest.approx(7.5 + 10.0)


def test_default_power_timestamp_is_taken_during_the_read(sensor: Ina3221) -> None:
    tel = Telemetry(make_platform(sensor.rail))
    before = time.monotonic_ns()
    tel._sample_power()
    after = time.monotonic_ns()
    assert before <= tel._t[-1] <= after


# -- memory ---------------------------------------------------------------------------------


def test_memory_sample_reads_meminfo_and_statm(tmp_path: Path) -> None:
    write_meminfo(tmp_path, available_kb=5_150_980, swap_free_kb=3_440_608)
    write_statm(tmp_path, 4242, 25_000)
    tel = Telemetry(make_platform(), sysroot=tmp_path)
    assert tel.memory() is None
    tel.set_worker_pid(4242)

    sample = tel._sample_memory(T0)

    assert isinstance(sample, MemorySample) and tel.memory() == sample
    assert sample.t_ns == T0
    assert sample.mem_available_bytes == 5_150_980 * 1024
    assert sample.swap_free_bytes == 3_440_608 * 1024
    assert sample.worker_rss_bytes == 25_000 * PAGE
    own_rss = int(Path("/proc/self/statm").read_text().split()[1]) * PAGE  # front-end RSS is this process
    assert sample.frontend_rss_bytes > 0
    assert abs(sample.frontend_rss_bytes - own_rss) < 64 * MB


def test_worker_rss_none_without_pid_or_readable_statm(tmp_path: Path) -> None:
    write_meminfo(tmp_path)
    tel = Telemetry(make_platform(), sysroot=tmp_path)
    t = iter(range(T0, T0 + 100))
    assert tel._sample_memory(next(t)).worker_rss_bytes is None  # type: ignore[union-attr]  # no pid yet
    tel.set_worker_pid(999_999)
    assert tel._sample_memory(next(t)).worker_rss_bytes is None  # type: ignore[union-attr]  # process gone
    statm = write_statm(tmp_path, 999_999, 10)
    for bad in ("", "123", "123 abc 5", "1 -5 2"):
        statm.write_text(bad)
        assert tel._sample_memory(next(t)).worker_rss_bytes is None  # type: ignore[union-attr]
    write_statm(tmp_path, 999_999, 10)
    assert tel._sample_memory(next(t)).worker_rss_bytes == 10 * PAGE  # type: ignore[union-attr]
    tel.set_worker_pid(None)
    assert tel._sample_memory(next(t)).worker_rss_bytes is None  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "content",
    [
        None,
        "MemTotal: 100 kB\n",
        "MemAvailable: lots kB\n",
        "MemAvailable:\n",
        "MemAvailable: -5 kB\n",
        "MemAvailable: \uff15\uff10 kB\n",  # fullwidth digits: int() would accept them, meminfo never has them
        b"MemAvailable: \xff\xfe kB\n\x00\x01",
        "directory",
    ],
)
def test_unusable_meminfo_skips_the_sample(tmp_path: Path, content: str | bytes | None) -> None:
    meminfo = tmp_path / "proc" / "meminfo"
    if content == "directory":
        meminfo.mkdir(parents=True)
    elif isinstance(content, bytes):
        meminfo.parent.mkdir()
        meminfo.write_bytes(content)
    elif content is not None:
        meminfo.parent.mkdir()
        meminfo.write_text(content, encoding="utf-8")
    tel = Telemetry(make_platform(), sysroot=tmp_path)
    assert tel._sample_memory(T0) is None
    assert tel.memory() is None
    assert tel.min_mem_available(0, 2**62) is None
    assert tel.peak_worker_rss(0, 2**62) is None


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads files regardless of their mode")
def test_unreadable_proc_files(tmp_path: Path) -> None:
    write_meminfo(tmp_path)
    statm = write_statm(tmp_path, 4242, 10)
    tel = Telemetry(make_platform(), sysroot=tmp_path)
    tel.set_worker_pid(4242)
    statm.chmod(0)
    try:
        sample = tel._sample_memory(T0)
        assert sample is not None and sample.worker_rss_bytes is None  # an unreadable statm is no RSS
        (tmp_path / "proc" / "meminfo").chmod(0)
        assert tel._sample_memory(T0 + S) is None
    finally:
        statm.chmod(0o644)
        (tmp_path / "proc" / "meminfo").chmod(0o644)


@pytest.mark.parametrize("swap_line", [None, "SwapFree:       lots kB", "SwapFree:", "SwapFree:      -4 kB"])
def test_missing_or_malformed_swapfree_counts_as_no_swap(tmp_path: Path, swap_line: str | None) -> None:
    # Regression: a malformed SwapFree line used to drop the whole sample, MemAvailable included.
    write_meminfo(tmp_path, available_kb=1000, swap_free_kb=None)
    meminfo = tmp_path / "proc" / "meminfo"
    if swap_line is not None:
        meminfo.write_text(meminfo.read_text() + swap_line + "\n")
    sample = Telemetry(make_platform(), sysroot=tmp_path)._sample_memory(T0)
    assert sample is not None
    assert (sample.mem_available_bytes, sample.swap_free_bytes) == (1000 * 1024, 0)


def test_meminfo_values_without_a_unit_are_bytes(tmp_path: Path) -> None:
    (tmp_path / "proc").mkdir()
    (tmp_path / "proc" / "meminfo").write_text("MemAvailable: 123456\nSwapFree: 7 kB\n")
    sample = Telemetry(make_platform(), sysroot=tmp_path)._sample_memory(T0)
    assert sample is not None and (sample.mem_available_bytes, sample.swap_free_bytes) == (123456, 7 * 1024)


def test_memory_window_queries(tmp_path: Path) -> None:
    tel = Telemetry(make_platform(), sysroot=tmp_path)
    tel.set_worker_pid(77)
    statm = tmp_path / "proc" / "77" / "statm"
    for s, available_mb, rss_pages in [(1, 500, 10), (2, 300, None), (3, 400, 30), (4, 200, 20), (5, 600, None)]:
        write_meminfo(tmp_path, available_kb=available_mb * 1024)
        if rss_pages is None:
            statm.unlink(missing_ok=True)
        else:
            write_statm(tmp_path, 77, rss_pages)
        assert tel._sample_memory(T0 + s * S) is not None

    assert tel.min_mem_available(T0 + S, T0 + 3 * S) == 300 * MB  # bounds are inclusive
    assert tel.min_mem_available(T0 + 4 * S, T0 + 4 * S) == 200 * MB
    assert tel.min_mem_available(T0 + 2500 * MS, T0 + 3500 * MS) == 400 * MB
    assert tel.min_mem_available(T0 + 3200 * MS, T0 + 3800 * MS) == 400 * MB  # none inside: latest before t0
    assert tel.min_mem_available(T0 + 9 * S, T0 + 10 * S) == 600 * MB
    assert tel.min_mem_available(T0, T0 + 500 * MS) is None  # nothing at or before the window
    assert tel.min_mem_available(T0 + 3 * S, T0 + S) is None  # reversed

    assert tel.peak_worker_rss(T0, T0 + 10 * S) == 30 * PAGE
    assert tel.peak_worker_rss(T0 + 4 * S, T0 + 5 * S) == 20 * PAGE  # samples without RSS are skipped
    assert tel.peak_worker_rss(T0 + 2 * S, T0 + 2 * S) is None  # the only sample inside has no RSS
    assert tel.peak_worker_rss(T0 + 3500 * MS, T0 + 3900 * MS) == 30 * PAGE  # fallback: the 3 s sample
    assert tel.peak_worker_rss(T0 + 6 * S, T0 + 7 * S) is None  # fallback sample (5 s) has no RSS
    assert tel.peak_worker_rss(T0 + 3 * S, T0 + S) is None
    nan = float("nan")  # regression: a NaN bound used to select every sample
    for t0, t1 in [(nan, nan), (T0 + S, nan), (nan, T0 + 5 * S)]:
        assert tel.min_mem_available(t0, t1) is None  # type: ignore[arg-type]
        assert tel.peak_worker_rss(t0, t1) is None  # type: ignore[arg-type]

    latest = tel.memory()
    assert latest is not None and latest.t_ns == T0 + 5 * S and latest.mem_available_bytes == 600 * MB


def test_memory_samples_older_than_the_window_are_dropped(tmp_path: Path) -> None:
    write_meminfo(tmp_path)
    tel = Telemetry(make_platform(), window_s=10.0, sysroot=tmp_path)
    for s in range(31):
        tel._sample_memory(T0 + s * S)
    assert tel._sample_memory(T0 + 30 * S) is None  # same timestamp again
    assert tel._sample_memory(T0 + 29 * S) is None  # older
    assert tel.min_mem_available(T0, T0 + 5 * S) is None
    assert tel.min_mem_available(T0 + 19 * S, T0 + 19 * S + 1) is None  # 19 s sample is gone, no fallback
    assert tel.min_mem_available(T0 + 20 * S, T0 + 20 * S) == 4_000_000 * 1024  # exactly window_s old: kept
    assert tel.memory().t_ns == T0 + 30 * S  # type: ignore[union-attr]


def test_default_memory_timestamp_is_now(tmp_path: Path) -> None:
    write_meminfo(tmp_path)
    tel = Telemetry(make_platform(), sysroot=tmp_path)
    before = time.monotonic_ns()
    sample = tel._sample_memory()
    assert sample is not None and before <= sample.t_ns <= time.monotonic_ns()


# -- sampler thread -------------------------------------------------------------------------


def test_sampler_thread_samples_power_and_memory(sensor: Ina3221, tmp_path: Path) -> None:
    sensor.set(4.0)
    root = tmp_path / "root"
    write_meminfo(root)
    tel = Telemetry(make_platform(sensor.rail), power_hz=200.0, memory_hz=100.0, sysroot=root)
    tel.start()
    thread = tel._thread
    tel.start()  # idempotent
    assert thread is not None and tel._thread is thread
    assert thread.daemon and thread.is_alive()
    try:
        wait_for(lambda: tel.memory() is not None and tel.energy_total_j() > 4.0 * 0.1)
        newest = tel._t[-1]  # relative to the data, not the clock: a starved sampler must not fail this
        assert tel.mean_power_w(newest - 50 * MS, newest) == pytest.approx(4.0)
        assert tel.memory().mem_available_bytes == 4_000_000 * 1024  # type: ignore[union-attr]
    finally:
        tel.stop()
    assert not thread.is_alive()
    tel.stop()  # idempotent

    tel.start()  # restartable
    try:
        assert tel._thread is not None and tel._thread is not thread and tel._thread.is_alive()
    finally:
        tel.stop()


def test_stop_returns_promptly_between_samples(sensor: Ina3221, tmp_path: Path) -> None:
    write_meminfo(tmp_path)
    tel = Telemetry(make_platform(sensor.rail), power_hz=0.05, memory_hz=0.05, sysroot=tmp_path)  # 20 s periods
    tel.start()
    wait_for(lambda: tel.memory() is not None)  # first samples are taken right away
    began = time.monotonic()
    tel.stop()
    assert time.monotonic() - began < 1.0


class BlockingRail:
    """A rail whose reads hang until ``release`` is set (a stuck I2C transaction)."""

    name = "VDD_IN"

    def __init__(self) -> None:
        self.in_read = threading.Event()
        self.release = threading.Event()

    def read_watts(self) -> float | None:
        self.in_read.set()
        self.release.wait(5.0)
        return 1.0


def test_stop_gives_up_on_a_hung_read_and_can_restart(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(telemetry_module, "_JOIN_TIMEOUT_S", 0.05)
    rail = BlockingRail()
    tel = Telemetry(make_platform(rail), memory_hz=0.0)
    tel.start()
    stuck = tel._thread
    assert stuck is not None and rail.in_read.wait(2.0)
    try:
        with caplog.at_level(logging.WARNING, logger="edgeserve.telemetry"):
            tel.stop()
        assert "did not stop within 0.05 s" in caplog.text
        assert stuck.is_alive() and tel._thread is None
        tel.start()  # a new sampler starts while the stuck one lingers
        assert tel._thread is not None and tel._thread is not stuck
    finally:
        rail.release.set()
        tel.stop()
    stuck.join(2.0)
    assert not stuck.is_alive()


def test_nvml_is_not_shut_down_under_a_call_in_flight(
    fake_nvml: Callable[..., FakeNvml], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(telemetry_module, "_JOIN_TIMEOUT_S", 0.05)
    nvml = fake_nvml()
    tel = Telemetry(make_platform(nvml_index=0), power_hz=100.0, memory_hz=0.0)
    tel.start()  # probes the counter, unblocked
    stuck = tel._thread
    gate = nvml.gate = threading.Event()
    try:
        assert stuck is not None and nvml.in_read.wait(2.0)
        tel.stop()
        tel.stop()  # a second stop (service and CLI both stop) must not shut down under the call either
        assert stuck.is_alive() and nvml.shutdowns == 0
    finally:
        gate.set()
    stuck.join(2.0)
    assert not stuck.is_alive()
    tel.stop()
    assert (nvml.inits, nvml.shutdowns) == (1, 1)


def test_sampler_survives_exceptions(
    sensor: Ina3221, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    write_meminfo(tmp_path)
    tel = Telemetry(make_platform(sensor.rail), power_hz=500.0, memory_hz=500.0, sysroot=tmp_path)
    calls: list[int | None] = []

    def broken(now_ns: int | None = None) -> float | None:
        calls.append(now_ns)
        raise RuntimeError("sensor exploded")

    monkeypatch.setattr(tel, "_sample_power", broken)
    with caplog.at_level(logging.WARNING, logger="edgeserve.telemetry"):
        tel.start()
        try:
            wait_for(lambda: len(calls) >= 5 and tel.memory() is not None)
            assert tel._thread is not None and tel._thread.is_alive()
        finally:
            tel.stop()
    assert len([r for r in caplog.records if "sensor exploded" in r.getMessage()]) == 1  # rate-limited


def test_failed_thread_start_leaves_a_stoppable_object(
    fake_nvml: Callable[..., FakeNvml], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: the unstarted Thread was kept, so stop() raised "cannot join thread before it is started"
    # and skipped the NVML shutdown.
    nvml = fake_nvml()
    tel = Telemetry(make_platform(nvml_index=0), memory_hz=0.0)

    def refuse(self: threading.Thread) -> None:
        raise RuntimeError("can't start new thread")

    with monkeypatch.context() as patch:
        patch.setattr(threading.Thread, "start", refuse)
        with pytest.raises(RuntimeError, match="can't start new thread"):
            tel.start()
    assert tel._thread is None
    tel.stop()
    assert (nvml.inits, nvml.shutdowns) == (1, 1)
    tel.start()  # works once threads can be started again
    try:
        assert tel._thread is not None and tel._thread.is_alive()
    finally:
        tel.stop()


def test_nothing_to_sample_starts_no_thread(tmp_path: Path) -> None:
    tel = Telemetry(make_platform(), memory_hz=0.0, sysroot=tmp_path)
    tel.start()
    assert tel._thread is None
    tel.stop()


def test_zero_power_rate_samples_memory_only(sensor: Ina3221, tmp_path: Path) -> None:
    sensor.set(5.0)
    write_meminfo(tmp_path)
    tel = Telemetry(make_platform(sensor.rail), power_hz=0.0, memory_hz=200.0, sysroot=tmp_path)
    # Regression: the rail was still named, so records and the Prometheus counter claimed a source that was
    # never sampled (a flat 0 J counter labelled VDD_IN).
    assert tel.energy_source is None
    tel.start()
    try:
        wait_for(lambda: tel.memory() is not None)
        time.sleep(0.05)
    finally:
        tel.stop()
    assert len(tel._t) == 0 and tel.energy_total_j() == 0.0


def test_zero_power_rate_never_initialises_nvml(fake_nvml: Callable[..., FakeNvml]) -> None:
    nvml = fake_nvml()
    tel = Telemetry(make_platform(nvml_index=0), power_hz=0.0, memory_hz=0.0)
    assert tel.energy_source is None
    tel.start()
    assert tel._thread is None
    tel.stop()
    assert (nvml.inits, nvml.shutdowns) == (0, 0)


def test_queries_are_safe_while_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(telemetry_module, "_COMPACT_AT", 8)  # compact every few samples, under the queries
    rail = CountingRail(6.0)
    tel = Telemetry(make_platform(rail), power_hz=1000.0, memory_hz=0.0, window_s=0.05)
    tel.start()
    try:
        wait_for(lambda: tel.energy_total_j() > 0)
        end = time.monotonic() + 0.3
        while time.monotonic() < end:
            now = time.monotonic_ns()
            for energy in tel.energy_series_j([now - 40 * MS + k * MS // 4 for k in range(161)]):
                assert energy is None or (math.isfinite(energy) and energy >= 0)
            power = tel.mean_power_w(now - 30 * MS, now)
            assert power is None or power == pytest.approx(6.0)
    finally:
        tel.stop()
    assert len(tel._t) < rail.reads  # every read was stored, so the missing ones were compacted away


# -- NVML -----------------------------------------------------------------------------------


class FakeNvml(types.ModuleType):
    """Stand-in for pynvml: an energy counter (mJ) and a power reading (mW) set by the test."""

    class NVMLError(Exception):
        pass

    def __init__(
        self, counter_supported: bool = True, power_supported: bool = True, init_error: bool = False
    ) -> None:
        super().__init__("pynvml")
        self.counter_supported = counter_supported
        self.power_supported = power_supported
        self.init_error = init_error
        self.energy_mj = 0
        self.energy_step_mj = 0  # added after every counter read (for the threaded test)
        self.power_mw = 0
        self.inits = 0
        self.shutdowns = 0
        self.handles: list[int] = []
        self.gate: threading.Event | None = None  # when set, counter reads block until the gate opens
        self.in_read = threading.Event()

    def nvmlInit(self) -> None:
        if self.init_error:
            raise self.NVMLError("Driver Not Loaded")
        self.inits += 1

    def nvmlShutdown(self) -> None:
        self.shutdowns += 1

    def nvmlDeviceGetHandleByIndex(self, index: int) -> str:
        self.handles.append(index)
        return f"handle-{index}"

    def nvmlDeviceGetTotalEnergyConsumption(self, handle: str) -> int:
        if not self.counter_supported:
            raise self.NVMLError("Not Supported")
        if self.gate is not None:
            self.in_read.set()
            self.gate.wait(5.0)
        value = self.energy_mj
        self.energy_mj += self.energy_step_mj
        return value

    def nvmlDeviceGetPowerUsage(self, handle: str) -> int:
        if not self.power_supported:
            raise self.NVMLError("Not Supported")
        return self.power_mw


@pytest.fixture
def fake_nvml(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., FakeNvml]]:
    def install(**kwargs: bool) -> FakeNvml:
        module = FakeNvml(**kwargs)
        monkeypatch.setitem(sys.modules, "pynvml", module)
        return module

    yield install


def test_nvml_energy_counter_is_interpolated(fake_nvml: Callable[..., FakeNvml]) -> None:
    nvml = fake_nvml()
    tel = Telemetry(make_platform(nvml_index=1))
    assert tel.energy_source == "nvml"
    for t_ns, millijoules in ((T0, 1_000_000), (T0 + S, 1_010_000), (T0 + 2 * S, 1_030_000)):
        nvml.energy_mj = millijoules
        assert tel._sample_power(t_ns) == pytest.approx(millijoules / 1000)
    assert nvml.handles == [1] and nvml.inits == 1
    assert tel.energy_j(T0, T0 + 2 * S) == pytest.approx(30.0)
    assert tel.energy_j(T0 + 500 * MS, T0 + 1500 * MS) == pytest.approx(15.0)  # 5 J + 10 J, linear counter
    assert tel.energy_j(T0 + 2 * S, T0 + 2400 * MS) == pytest.approx(8.0)  # last interval's 20 W
    assert tel.energy_j(T0 - 200 * MS, T0) == pytest.approx(2.0)  # first interval's 10 W
    assert tel.energy_j(T0 + 2 * S, T0 + 2600 * MS) is None
    assert tel.mean_power_w(T0 + S, T0 + 2 * S) == pytest.approx(20.0)
    assert tel.energy_series_j([T0, T0 + S, T0 + 2 * S]) == [pytest.approx(10.0), pytest.approx(20.0)]
    assert tel.energy_total_j() == pytest.approx(30.0)


def test_nvml_counter_matches_an_independent_integrator(fake_nvml: Callable[..., FakeNvml]) -> None:
    rng = random.Random(5)
    nvml = fake_nvml()
    for _ in range(30):
        tel = Telemetry(make_platform(nvml_index=0))
        ts: list[int] = []
        readings: list[int] = []
        t, nvml.energy_mj = T0, rng.randint(0, 10**9)
        for _ in range(rng.randint(2, 25)):
            t += rng.randint(1, 300 * MS)
            reset = rng.random() < 0.1
            nvml.energy_mj = rng.randint(0, 500) if reset else nvml.energy_mj + rng.randint(0, 30_000)
            assert tel._sample_power(t) == pytest.approx(nvml.energy_mj / 1000)
            ts.append(t)
            readings.append(nvml.energy_mj)
        # Expected counter in J: increments add up; an interval where the counter went backwards adds nothing.
        joules = [Fraction(0)]
        for prev, cur in zip(readings, readings[1:]):
            joules.append(joules[-1] + Fraction(max(0, cur - prev), 1000))
        first_w = (joules[1] - joules[0]) / (ts[1] - ts[0])
        last_w = (joules[-1] - joules[-2]) / (ts[-1] - ts[-2])

        def counter(t_ns: int) -> Fraction:
            if t_ns < ts[0]:
                return joules[0] - first_w * (ts[0] - t_ns)  # extended at the first interval's power
            if t_ns > ts[-1]:
                return joules[-1] + last_w * (t_ns - ts[-1])
            return held_linear(ts, joules, t_ns)

        assert tel.energy_total_j() == pytest.approx(float(joules[-1]))
        for _ in range(30):
            a = rng.randint(ts[0] - 700 * MS, ts[-1] + 700 * MS)
            b = rng.choice([a, rng.randint(a, ts[-1] + 700 * MS)])
            got = tel.energy_j(a, b)
            if a < ts[0] - 500 * MS or b > ts[-1] + 500 * MS:
                assert got is None
            else:
                assert got == pytest.approx(float(counter(b) - counter(a)), rel=1e-9, abs=1e-9)


def test_nvml_falls_back_to_power_usage(fake_nvml: Callable[..., FakeNvml]) -> None:
    nvml = fake_nvml(counter_supported=False)
    tel = Telemetry(make_platform(nvml_index=0))
    for t_ns, milliwatts in ((T0, 100_000), (T0 + S, 300_000)):
        nvml.power_mw = milliwatts
        assert tel._sample_power(t_ns) == pytest.approx(milliwatts / 1000)
    assert tel.energy_source == "nvml"
    assert tel.energy_j(T0, T0 + S) == pytest.approx(200.0)  # trapezoid 100 W -> 300 W
    assert tel.energy_j(T0 + S, T0 + 1500 * MS) == pytest.approx(150.0)  # held at 300 W
    assert tel.energy_total_j() == pytest.approx(200.0)


def test_nvml_counter_reset_keeps_the_total_monotonic(fake_nvml: Callable[..., FakeNvml]) -> None:
    nvml = fake_nvml()
    tel = Telemetry(make_platform(nvml_index=0))
    totals = []
    for i, millijoules in enumerate((5_000, 6_000, 100, 1_100)):
        nvml.energy_mj = millijoules
        tel._sample_power(T0 + i * S)
        totals.append(tel.energy_total_j())
    assert totals == pytest.approx([0.0, 1.0, 1.0, 2.0])
    assert tel.energy_j(T0 + S, T0 + 2 * S) == 0.0  # unknown across the reset
    assert tel.energy_j(T0, T0 + 3 * S) == pytest.approx(2.0)


@pytest.mark.parametrize("failure", ["not-installed", "init", "no-readings"])
def test_unavailable_nvml_means_no_energy_source(
    fake_nvml: Callable[..., FakeNvml], monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    nvml = None
    if failure == "not-installed":
        monkeypatch.setitem(sys.modules, "pynvml", None)  # makes `import pynvml` raise ImportError
    elif failure == "init":
        nvml = fake_nvml(init_error=True)
    else:
        nvml = fake_nvml(counter_supported=False, power_supported=False)
    tel = Telemetry(make_platform(nvml_index=0), memory_hz=0.0, sysroot=tmp_path)
    assert tel.energy_source == "nvml"  # until NVML has been tried
    tel.start()
    assert tel.energy_source is None
    assert tel._thread is None  # nothing left to sample
    assert tel._sample_power(T0) is None
    assert tel.energy_j(T0, T0) is None and tel.energy_total_j() == 0.0
    tel.stop()
    if failure == "no-readings":
        assert nvml is not None and nvml.inits == 1 and nvml.shutdowns == 1


def test_nvml_start_stop_lifecycle(fake_nvml: Callable[..., FakeNvml]) -> None:
    nvml = fake_nvml()
    nvml.energy_step_mj = 10
    tel = Telemetry(make_platform(nvml_index=0), power_hz=200.0, memory_hz=0.0)
    tel.start()
    try:
        assert nvml.inits == 1
        wait_for(lambda: tel.energy_total_j() > 0)
    finally:
        tel.stop()
    assert nvml.shutdowns == 1
    tel.stop()
    assert nvml.shutdowns == 1
    tel.start()
    tel.stop()
    assert (nvml.inits, nvml.shutdowns) == (2, 2)


def test_rail_takes_precedence_over_nvml(sensor: Ina3221, fake_nvml: Callable[..., FakeNvml]) -> None:
    nvml = fake_nvml()
    tel = Telemetry(make_platform(sensor.rail, nvml_index=0), memory_hz=0.0)
    sensor.set(2.0)
    assert tel.energy_source == "VDD_IN"
    assert tel._sample_power(T0) == pytest.approx(2.0)
    tel.start()
    tel.stop()
    assert nvml.inits == 0
