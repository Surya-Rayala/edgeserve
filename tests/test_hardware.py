"""Tests for edgeserve.hardware: fake sysroots, probe results, host tools, NVML and workers."""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from edgeserve import hardware
from edgeserve.hardware import (
    FP16_FLOPS_PER_CLOCK_PER_SM,
    Platform,
    PlatformError,
    PowerRail,
    probe,
    probe_gpu,
    refresh_clocks,
)

# What `edgeserve-worker --probe` prints on a Jetson Orin Nano Super (JetPack 6.2.1).
ORIN_PROBE: dict[str, Any] = {
    "clock_rate_khz": 1020000,
    "cuda_driver": "12.6",
    "cuda_runtime": "12.6",
    "edgellm": "0.11.0",
    "ev": "probe",
    "free_memory_bytes": 5301952512,
    "gpu_name": "Orin",
    "integrated": True,
    "l2_cache_bytes": 2097152,
    "memory_bus_width_bits": 128,
    "memory_clock_rate_khz": 918000,
    "sm": 87,
    "sm_count": 8,
    "tensorrt": "10.3.0.30",
    "tensorrt_runtime": "10.3.0",
    "total_memory_bytes": 7990042624,
}

RTX4090_PROBE: dict[str, Any] = {
    "clock_rate_khz": 2520000,
    "cuda_driver": "12.8",
    "cuda_runtime": "12.8",
    "edgellm": "0.11.0",
    "ev": "probe",
    "free_memory_bytes": 24_000_000_000,
    "gpu_name": "NVIDIA GeForce RTX 4090",
    "integrated": False,
    "l2_cache_bytes": 75497472,
    "memory_bus_width_bits": 384,
    "memory_clock_rate_khz": 10501000,
    "sm": 89,
    "sm_count": 128,
    "tensorrt": "10.8.0.43",
    "tensorrt_runtime": "10.8.0",
    "total_memory_bytes": 25_390_809_088,
}

ORIN_RELEASE = (
    "# R36 (release), REVISION: 4.4, GCID: 41062509, BOARD: generic, EABI: aarch64, "
    "DATE: Mon Jun 16 16:07:13 UTC 2025\n# KERNEL_VARIANT: oot\n"
)
ORIN_BOARD = "NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super"

AS_DICT_KEYS = [
    "kind",
    "hostname",
    "board",
    "l4t",
    "jetpack",
    "gpu_name",
    "sm",
    "sm_count",
    "integrated",
    "total_memory_bytes",
    "memory_bus_width_bits",
    "cuda_runtime",
    "cuda_driver",
    "tensorrt",
    "edgellm",
    "power_mode",
    "gpu_max_freq_hz",
    "emc_max_freq_hz",
    "peak_flops_per_s",
    "peak_mem_bw_bytes_per_s",
    "power_rails",
    "energy_rail",
    "nvml_index",
]


# --- helpers ---------------------------------------------------------------------------------


def write(path: Path, content: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)
    return path


def add_ina3221(root: Path, index: int, labels: dict[int, str]) -> Path:
    """An INA3221 hwmon node with one labelled channel per entry, each reading 5 V x 1 A."""
    hwmon = root / f"sys/class/hwmon/hwmon{index}"
    write(hwmon / "name", "ina3221\n")
    for channel, label in labels.items():
        write(hwmon / f"in{channel}_label", f"{label}\n")
        write(hwmon / f"in{channel}_input", "5000\n")
        write(hwmon / f"curr{channel}_input", "1000\n")
    return hwmon


def add_ina238(root: Path, index: int, label: str | None = None, microwatts: int = 12_345_678) -> Path:
    hwmon = root / f"sys/class/hwmon/hwmon{index}"
    write(hwmon / "name", "ina238\n")
    write(hwmon / "power1_input", f"{microwatts}\n")
    if label is not None:
        write(hwmon / "label", f"{label}\n")
    return hwmon


def orin_root(tmp_path: Path) -> Path:
    """A fake Orin Nano Super sysroot in 25W mode, laid out like the real device."""
    root = tmp_path / "orin"
    write(root / "etc/nv_tegra_release", ORIN_RELEASE)
    write(root / "proc/device-tree/model", ORIN_BOARD.encode() + b"\x00")
    for node, hz in (("15340000.vic", 435200000), ("15480000.nvdec", 524800000), ("17000000.gpu", 918000000)):
        write(root / f"sys/class/devfreq/{node}/max_freq", f"{hz}\n")
    write(root / "sys/kernel/nvpmodel_clk_cap/emc", "3199000000\n")
    write(root / "sys/class/hwmon/hwmon0/name", "pwmfan\n")
    hwmon = add_ina3221(root, 1, {1: "VDD_IN", 2: "VDD_CPU_GPU_CV", 3: "VDD_SOC", 7: "sum of shunt voltages"})
    (hwmon / "curr7_input").unlink()  # the summed channel has no current file on the real chip
    write(hwmon / "in1_input", "5056\n")
    write(hwmon / "curr1_input", "1504\n")
    return root


def make_platform(**overrides: Any) -> Platform:
    """A Platform built by hand, as other modules' tests do (no probe(), so no private state)."""
    values: dict[str, Any] = {
        "kind": "jetson",
        "hostname": "h",
        "board": None,
        "l4t": None,
        "jetpack": None,
        "gpu_name": "Orin",
        "sm": 87,
        "sm_count": 8,
        "integrated": True,
        "total_memory_bytes": 7990042624,
        "memory_bus_width_bits": 128,
        "cuda_runtime": "12.6",
        "cuda_driver": "12.6",
        "tensorrt": "10.3.0.30",
        "edgellm": "0.11.0",
        "power_mode": "25W",
        "gpu_max_freq_hz": 918_000_000,
        "emc_max_freq_hz": 3_199_000_000,
        "peak_flops_per_s": None,
        "peak_mem_bw_bytes_per_s": None,
    }
    values.update(overrides)
    return Platform(**values)


class FakeHost:
    """Stands in for subprocess.run: tool name -> (returncode, stdout), or an exception to raise.

    Tools without an entry behave as if they were not installed.
    """

    def __init__(self) -> None:
        self.results: dict[str, tuple[int, str] | BaseException] = {}
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append((list(argv), kwargs))
        result = self.results.get(Path(argv[0]).name)
        if result is None:
            raise FileNotFoundError(2, "No such file or directory", argv[0])
        if isinstance(result, BaseException):
            raise result
        returncode, stdout = result
        return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr="")

    @property
    def tools(self) -> list[str]:
        return [Path(argv[0]).name for argv, _ in self.calls]


def fake_nvml(
    monkeypatch: pytest.MonkeyPatch,
    sm_mhz: Any = 2595,
    mem_mhz: Any = 10251,
    init_error: Exception | None = None,
    handle_error: Exception | None = None,
    unsupported: tuple[str, ...] = (),
) -> types.ModuleType:
    """Install a stand-in ``pynvml`` module; its ``calls`` list records the NVML calls made."""
    module = types.ModuleType("pynvml")
    calls: list[str] = []
    clocks = {1: ("sm", sm_mhz), 2: ("mem", mem_mhz)}

    class NVMLError(Exception):
        pass

    def nvml_init() -> None:
        calls.append("init")
        if init_error is not None:
            raise init_error

    def get_handle(index: int) -> str:
        calls.append(f"handle:{index}")
        if handle_error is not None:
            raise handle_error
        return f"device{index}"

    def get_max_clock(handle: str, clock_type: int) -> Any:
        name, value = clocks[clock_type]
        calls.append(f"max_clock:{name}")
        if name in unsupported:
            raise NVMLError("Not Supported")
        return value

    def nvml_shutdown() -> None:
        calls.append("shutdown")

    module.__dict__.update(
        NVML_CLOCK_SM=1,
        NVML_CLOCK_MEM=2,
        NVMLError=NVMLError,
        calls=calls,
        nvmlInit=nvml_init,
        nvmlShutdown=nvml_shutdown,
        nvmlDeviceGetHandleByIndex=get_handle,
        nvmlDeviceGetMaxClockInfo=get_max_clock,
    )
    monkeypatch.setitem(sys.modules, "pynvml", module)
    return module


def fake_worker(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "edgeserve-worker"
    script.write_text(f"#!/bin/sh\n{body}\n")
    script.chmod(0o755)
    return script


@pytest.fixture(autouse=True)
def _isolate_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware.socket, "gethostname", lambda: "testhost")
    monkeypatch.setitem(sys.modules, "pynvml", None)  # NVML absent unless a test installs a fake


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch) -> FakeHost:
    """Intercept host tools (nvpmodel, dpkg-query) so a test can never run the real ones."""
    fake = FakeHost()
    monkeypatch.setattr(hardware.subprocess, "run", fake)
    return fake


@pytest.fixture
def live_host(monkeypatch: pytest.MonkeyPatch, host: FakeHost) -> FakeHost:
    """Make probe() treat the fake sysroot as the live host, so it runs the (fake) host tools."""
    monkeypatch.setattr(hardware, "_is_live_host", lambda root: True)
    return host


# --- probe(): Jetson -------------------------------------------------------------------------


def test_probe_jetson_orin(tmp_path: Path, host: FakeHost) -> None:
    root = orin_root(tmp_path)
    platform = probe(ORIN_PROBE, root)

    assert platform.kind == "jetson"
    assert platform.hostname == "testhost"
    assert platform.board == ORIN_BOARD
    assert platform.l4t == "R36.4.4"
    assert platform.jetpack == "6.2.1"  # from the L4T table: host tools never run for a fake sysroot
    assert platform.power_mode is None
    assert (platform.gpu_name, platform.sm, platform.sm_count, platform.integrated) == ("Orin", 87, 8, True)
    assert (platform.total_memory_bytes, platform.memory_bus_width_bits) == (7990042624, 128)
    assert (platform.cuda_runtime, platform.cuda_driver) == ("12.6", "12.6")
    assert (platform.tensorrt, platform.edgellm) == ("10.3.0.30", "0.11.0")
    # Power-mode caps from sysfs, not the worker's uncapped clock_rate_khz (1.02 GHz).
    assert platform.gpu_max_freq_hz == 918_000_000
    assert platform.emc_max_freq_hz == 3_199_000_000
    assert platform.peak_flops_per_s == pytest.approx(8 * 2048 * 918e6)  # 15.04 TFLOPS
    assert platform.peak_mem_bw_bytes_per_s == pytest.approx(3199e6 * 2 * 16)  # 102.37 GB/s
    assert platform.gpu_freq_path == str(root / "sys/class/devfreq/17000000.gpu/max_freq")
    assert platform.emc_freq_path == str(root / "sys/kernel/nvpmodel_clk_cap/emc")
    assert platform.nvml_index is None
    hwmon = root / "sys/class/hwmon/hwmon1"
    assert platform.power_rails == tuple(
        PowerRail(name, "ina3221", (str(hwmon / f"in{n}_input"), str(hwmon / f"curr{n}_input")))
        for n, name in ((1, "VDD_IN"), (2, "VDD_CPU_GPU_CV"), (3, "VDD_SOC"))
    )
    assert platform.energy_rail is platform.power_rails[0]
    assert host.calls == []
    assert probe(ORIN_PROBE, str(root)) == platform  # type: ignore[arg-type]  # str sysroots work too
    assert "_path" not in repr(platform) and "_live_host" not in repr(platform)


def test_probed_rails_read_watts(tmp_path: Path, host: FakeHost) -> None:
    root = orin_root(tmp_path)
    add_ina238(root, 2, microwatts=12_345_678)
    rails = {rail.name: rail for rail in probe(ORIN_PROBE, root).power_rails}

    assert rails["VDD_IN"].read_watts() == pytest.approx(5.056 * 1.504)
    assert rails["VDD_SOC"].read_watts() == pytest.approx(5.0)
    assert rails["VIN"].read_watts() == pytest.approx(12.345678)
    write(root / "sys/class/hwmon/hwmon1/curr3_input", "garbage\n")
    assert rails["VDD_SOC"].read_watts() is None
    (root / "sys/class/hwmon/hwmon2/power1_input").unlink()
    assert rails["VIN"].read_watts() is None
    assert PowerRail("X", "ina3221", rails["VDD_IN"].paths[:1]).read_watts() is None
    assert PowerRail("X", "unknown", rails["VDD_IN"].paths).read_watts() is None


@pytest.mark.parametrize("release", ["", "garbage\n", None], ids=["empty", "garbage", "directory"])
def test_probe_jetson_with_only_the_release_file(tmp_path: Path, host: FakeHost, release: str | None) -> None:
    root = tmp_path / "bare"
    if release is None:
        (root / "etc/nv_tegra_release").mkdir(parents=True)  # exists, but cannot be read as a file
    else:
        write(root / "etc/nv_tegra_release", release)

    platform = probe(ORIN_PROBE, root)

    assert platform.kind == "jetson"
    assert (platform.l4t, platform.jetpack, platform.board, platform.power_mode) == (None, None, None, None)
    # No devfreq/EMC files: unknown, never the worker's clock_rate_khz / memory_clock_rate_khz.
    assert (platform.gpu_max_freq_hz, platform.emc_max_freq_hz) == (None, None)
    assert (platform.peak_flops_per_s, platform.peak_mem_bw_bytes_per_s) == (None, None)
    assert (platform.gpu_freq_path, platform.emc_freq_path) == (None, None)
    assert platform.power_rails == ()
    assert platform.energy_rail is None
    assert platform.nvml_index is None
    assert host.calls == []


@pytest.mark.parametrize(
    ("first_line", "l4t", "jetpack"),
    [
        ("# R36 (release), REVISION: 4.3, GCID: 38968081, BOARD: generic, EABI: aarch64", "R36.4.3", "6.2"),
        ("# R36 (release), REVISION: 4.4, GCID: 41062509, BOARD: generic, EABI: aarch64", "R36.4.4", "6.2.1"),
        ("# R36 (release), REVISION: 5.0, GCID: 1, BOARD: generic, EABI: aarch64", "R36.5.0", "6.2.2"),
        ("# R39 (release), REVISION: 2.0, GCID: 1, BOARD: generic, EABI: aarch64", "R39.2.0", "7.2"),
        ("# R39 (release), REVISION: 2.1, GCID: 1, BOARD: generic, EABI: aarch64", "R39.2.1", "7.2.1"),
        ("# R35 (release), REVISION: 4.1, GCID: 33958178, BOARD: t186ref, EABI: aarch64", "R35.4.1", None),
        ("R36 REVISION 4.4", None, None),
        ("", None, None),  # only the first line counts
    ],
)
def test_l4t_release_and_jetpack_table(
    tmp_path: Path, host: FakeHost, first_line: str, l4t: str | None, jetpack: str | None
) -> None:
    root = tmp_path / "root"
    write(root / "etc/nv_tegra_release", f"{first_line}\n# R36 (release), REVISION: 4.4, GCID: 1\n")
    platform = probe(ORIN_PROBE, root)
    assert (platform.l4t, platform.jetpack) == (l4t, jetpack)


@pytest.mark.parametrize(
    ("nodes", "expected"),
    [
        ({"17000000.ga10b": 1300500000}, "17000000.ga10b"),  # JetPack 5 naming
        ({"a-gpu-x": 1, "z.gpu": 918000000}, "z.gpu"),  # "*.gpu" wins although it sorts later
        ({"gpu-gpc-0": 1575000000, "15340000.vic": 435200000}, "gpu-gpc-0"),
        ({"17000000.gpu": None, "17000000.ga10b": 1300500000}, "17000000.ga10b"),  # node without max_freq
        ({"15340000.vic": 435200000}, None),
    ],
)
def test_gpu_devfreq_node_selection(
    tmp_path: Path, host: FakeHost, nodes: dict[str, int | None], expected: str | None
) -> None:
    root = tmp_path / "root"
    write(root / "etc/nv_tegra_release", ORIN_RELEASE)
    for node, hz in nodes.items():
        (root / "sys/class/devfreq" / node).mkdir(parents=True)
        if hz is not None:
            write(root / "sys/class/devfreq" / node / "max_freq", f"{hz}\n")

    platform = probe(ORIN_PROBE, root)

    if expected is None:
        assert (platform.gpu_freq_path, platform.gpu_max_freq_hz, platform.peak_flops_per_s) == (None, None, None)
    else:
        assert platform.gpu_freq_path == str(root / "sys/class/devfreq" / expected / "max_freq")
        assert platform.gpu_max_freq_hz == nodes[expected]


@pytest.mark.parametrize("content", ["garbage\n", "0\n", "-5\n", ""])
def test_unparsable_clock_caps_are_unknown(tmp_path: Path, host: FakeHost, content: str) -> None:
    root = orin_root(tmp_path)
    write(root / "sys/class/devfreq/17000000.gpu/max_freq", content)
    write(root / "sys/kernel/nvpmodel_clk_cap/emc", content)

    platform = probe(ORIN_PROBE, root)

    assert platform.gpu_freq_path is not None  # still the file refresh_clocks re-reads
    assert (platform.gpu_max_freq_hz, platform.emc_max_freq_hz) == (None, None)
    assert (platform.peak_flops_per_s, platform.peak_mem_bw_bytes_per_s) == (None, None)


def test_probe_jetson_thor_layout(tmp_path: Path, host: FakeHost) -> None:
    root = tmp_path / "thor"
    write(root / "etc/nv_tegra_release", "# R39 (release), REVISION: 2.0, GCID: 1, BOARD: generic, EABI: aarch64\n")
    write(root / "proc/device-tree/model", b"NVIDIA Jetson AGX Thor Developer Kit\x00")
    write(root / "sys/class/devfreq/gpu-gpc-0/max_freq", "1575000000\n")
    write(root / "sys/class/devfreq/bwmgr/max_freq", "4266000000\n")
    add_ina238(root, 3)  # no label file: "VIN"
    add_ina238(root, 4, label="VDD_GPU")
    thor = {**ORIN_PROBE, "gpu_name": "NVIDIA Thor", "sm": 110, "sm_count": 20, "memory_bus_width_bits": 256}

    platform = probe(thor, root)

    assert (platform.l4t, platform.jetpack) == ("R39.2.0", "7.2")
    assert platform.board == "NVIDIA Jetson AGX Thor Developer Kit"
    assert platform.gpu_max_freq_hz == 1_575_000_000
    assert platform.emc_freq_path == str(root / "sys/class/devfreq/bwmgr/max_freq")
    assert platform.emc_max_freq_hz == 4_266_000_000
    assert platform.peak_flops_per_s == pytest.approx(20 * 8192 * 1575e6)
    assert platform.peak_mem_bw_bytes_per_s == pytest.approx(4266e6 * 2 * 256 / 8)
    assert [(rail.name, rail.kind) for rail in platform.power_rails] == [("VIN", "ina238"), ("VDD_GPU", "ina238")]
    assert platform.power_rails[0].paths == (str(root / "sys/class/hwmon/hwmon3/power1_input"),)
    assert platform.energy_rail is platform.power_rails[0]


def test_emc_prefers_the_nvpmodel_cap(tmp_path: Path, host: FakeHost) -> None:
    root = orin_root(tmp_path)
    write(root / "sys/class/devfreq/bwmgr/max_freq", "4266000000\n")
    platform = probe(ORIN_PROBE, root)
    assert platform.emc_max_freq_hz == 3_199_000_000


def test_energy_rail_follows_the_preference_order(tmp_path: Path, host: FakeHost) -> None:
    root = tmp_path / "root"
    write(root / "etc/nv_tegra_release", ORIN_RELEASE)
    add_ina3221(root, 0, {1: "VDD_GPU_SOC", 2: "VDD_CPU_CV", 3: "VIN_SYS_5V0"})
    add_ina238(root, 1, label="VDD_IN")

    platform = probe(ORIN_PROBE, root)

    assert [rail.name for rail in platform.power_rails] == ["VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0", "VDD_IN"]
    assert platform.energy_rail is platform.power_rails[3]  # VDD_IN beats VIN_SYS_5V0 though listed later


def test_no_energy_rail_without_a_preferred_name(tmp_path: Path, host: FakeHost) -> None:
    root = tmp_path / "root"
    write(root / "etc/nv_tegra_release", ORIN_RELEASE)
    add_ina3221(root, 0, {1: "VDD_GPU_SOC", 2: "VDD_CPU_CV"})
    platform = probe(ORIN_PROBE, root)
    assert len(platform.power_rails) == 2
    assert platform.energy_rail is None


def test_hwmon_scan_order_and_skipped_entries(tmp_path: Path, host: FakeHost) -> None:
    root = tmp_path / "root"
    write(root / "etc/nv_tegra_release", ORIN_RELEASE)
    add_ina3221(root, 10, {1: "LATE"})
    hwmon2 = add_ina3221(root, 2, {3: "THIRD", 1: "FIRST", 2: "SECOND", 7: "Sum Of shunt voltages"})
    write(hwmon2 / "in4_label", "\n")  # empty label
    write(hwmon2 / "inX_label", "BOGUS\n")  # not a channel label
    write(root / "sys/class/hwmon/hwmon3/name", "lm75\n")  # other sensors are ignored
    (root / "sys/class/hwmon/hwmon4").mkdir()  # no name file

    platform = probe(ORIN_PROBE, root)

    assert [rail.name for rail in platform.power_rails] == ["FIRST", "SECOND", "THIRD", "LATE"]


def test_unknown_sm_has_no_flops_peak(tmp_path: Path, host: FakeHost) -> None:
    assert 75 not in FP16_FLOPS_PER_CLOCK_PER_SM
    platform = probe({**ORIN_PROBE, "sm": 75}, orin_root(tmp_path))
    assert platform.sm == 75
    assert platform.peak_flops_per_s is None
    assert platform.peak_mem_bw_bytes_per_s == pytest.approx(102.368e9)


@pytest.mark.parametrize(
    ("override", "flops_known", "bandwidth_known"),
    [({"sm_count": 0}, False, True), ({"memory_bus_width_bits": 0}, True, False)],
)
def test_zero_sizes_give_no_peak(
    tmp_path: Path, host: FakeHost, override: dict[str, int], flops_known: bool, bandwidth_known: bool
) -> None:
    platform = probe({**ORIN_PROBE, **override}, orin_root(tmp_path))
    assert (platform.peak_flops_per_s is not None) is flops_known
    assert (platform.peak_mem_bw_bytes_per_s is not None) is bandwidth_known


@pytest.mark.parametrize(
    ("gpu", "field_name"),
    [
        ({k: v for k, v in ORIN_PROBE.items() if k != "sm"}, "sm"),
        ({**ORIN_PROBE, "sm": "87"}, "sm"),
        ({**ORIN_PROBE, "sm_count": True}, "sm_count"),
        ({**ORIN_PROBE, "integrated": 1}, "integrated"),
        ({**ORIN_PROBE, "gpu_name": None}, "gpu_name"),
        ({**ORIN_PROBE, "tensorrt": 10}, "tensorrt"),
    ],
    ids=["missing-sm", "string-sm", "bool-sm_count", "int-integrated", "null-gpu_name", "int-tensorrt"],
)
def test_malformed_probe_result_raises(tmp_path: Path, host: FakeHost, gpu: dict[str, Any], field_name: str) -> None:
    with pytest.raises(PlatformError, match=f"invalid fields: {field_name}$"):
        probe(gpu, orin_root(tmp_path))


def test_non_object_probe_result_raises(tmp_path: Path, host: FakeHost) -> None:
    with pytest.raises(PlatformError, match="not a JSON object"):
        probe([ORIN_PROBE], tmp_path)  # type: ignore[arg-type]


# --- probe(): discrete GPU -------------------------------------------------------------------


def test_probe_dgpu_without_nvml_uses_worker_clocks(tmp_path: Path, host: FakeHost) -> None:
    platform = probe(RTX4090_PROBE, tmp_path)

    assert platform.kind == "dgpu"
    assert (platform.board, platform.l4t, platform.jetpack, platform.power_mode) == (None, None, None, None)
    assert (platform.gpu_name, platform.sm, platform.integrated) == ("NVIDIA GeForce RTX 4090", 89, False)
    assert platform.gpu_max_freq_hz == 2_520_000_000
    assert platform.emc_max_freq_hz is None
    assert platform.peak_flops_per_s == pytest.approx(128 * 1024 * 2.52e9)  # ~330 TFLOPS
    assert platform.peak_mem_bw_bytes_per_s == pytest.approx(1008.096e9)
    assert platform.nvml_index is None
    assert (platform.power_rails, platform.energy_rail) == ((), None)
    assert (platform.gpu_freq_path, platform.emc_freq_path) == (None, None)
    assert host.calls == []


def test_probe_dgpu_prefers_nvml_max_clocks(tmp_path: Path, host: FakeHost, monkeypatch: pytest.MonkeyPatch) -> None:
    nvml = fake_nvml(monkeypatch, sm_mhz=2595, mem_mhz=10251)

    platform = probe(RTX4090_PROBE, tmp_path)

    assert platform.gpu_max_freq_hz == 2_595_000_000
    assert platform.peak_flops_per_s == pytest.approx(128 * 1024 * 2.595e9)
    assert platform.peak_mem_bw_bytes_per_s == pytest.approx(10251e6 * 2 * 384 / 8)
    assert platform.emc_max_freq_hz is None
    assert platform.nvml_index == 0
    assert nvml.calls == ["init", "handle:0", "max_clock:sm", "max_clock:mem", "shutdown"]


def test_probe_dgpu_falls_back_when_nvml_init_fails(
    tmp_path: Path, host: FakeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    nvml = fake_nvml(monkeypatch, init_error=RuntimeError("NVML Shared Library Not Found"))
    platform = probe(RTX4090_PROBE, tmp_path)
    assert platform.nvml_index is None
    assert platform.gpu_max_freq_hz == 2_520_000_000
    assert platform.peak_mem_bw_bytes_per_s == pytest.approx(1008.096e9)
    assert nvml.calls == ["init"]


def test_probe_dgpu_falls_back_when_nvml_has_no_device(
    tmp_path: Path, host: FakeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    nvml = fake_nvml(monkeypatch, handle_error=RuntimeError("Invalid Argument"))
    platform = probe(RTX4090_PROBE, tmp_path)
    assert platform.nvml_index is None
    assert platform.gpu_max_freq_hz == 2_520_000_000
    assert nvml.calls[-1] == "shutdown"


def test_probe_dgpu_falls_back_per_clock(tmp_path: Path, host: FakeHost, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_nvml(monkeypatch, sm_mhz=0, unsupported=("mem",))
    platform = probe(RTX4090_PROBE, tmp_path)
    assert platform.nvml_index == 0  # NVML works (power/energy), it just cannot report these clocks
    assert platform.gpu_max_freq_hz == 2_520_000_000
    assert platform.peak_mem_bw_bytes_per_s == pytest.approx(1008.096e9)


def test_probe_dgpu_with_unknown_clocks(tmp_path: Path, host: FakeHost) -> None:
    gpu = {k: v for k, v in RTX4090_PROBE.items() if k != "clock_rate_khz"} | {"memory_clock_rate_khz": 0}
    platform = probe(gpu, tmp_path)
    assert platform.gpu_max_freq_hz is None
    assert (platform.peak_flops_per_s, platform.peak_mem_bw_bytes_per_s) == (None, None)


@pytest.mark.parametrize("nvml_works", [True, False])
def test_dgpu_board_sensors_never_become_the_energy_rail(
    tmp_path: Path, host: FakeHost, monkeypatch: pytest.MonkeyPatch, nvml_works: bool
) -> None:
    # Telemetry integrates energy_rail in preference to NVML, so on a dGPU it must stay None.
    add_ina238(tmp_path, 0)  # "VIN"
    add_ina3221(tmp_path, 1, {1: "VDD_IN"})
    if nvml_works:
        fake_nvml(monkeypatch)

    platform = probe(RTX4090_PROBE, tmp_path)

    assert platform.kind == "dgpu"
    assert [rail.name for rail in platform.power_rails] == ["VIN", "VDD_IN"]  # still reported
    assert platform.energy_rail is None
    assert platform.nvml_index == (0 if nvml_works else None)
    assert platform.as_dict()["energy_rail"] is None


def test_probe_dgpu_with_a_pynvml_that_fails_to_import(
    tmp_path: Path, host: FakeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    modules = tmp_path / "site"
    write(modules / "pynvml.py", "raise OSError('libnvidia-ml.so.1: cannot open shared object file')\n")
    monkeypatch.syspath_prepend(str(modules))
    monkeypatch.delitem(sys.modules, "pynvml")  # undo the autouse "absent" marker so the import is attempted

    platform = probe(RTX4090_PROBE, tmp_path / "sysroot")

    assert platform.nvml_index is None
    assert platform.gpu_max_freq_hz == 2_520_000_000  # the worker's clocks
    assert platform.peak_mem_bw_bytes_per_s == pytest.approx(1008.096e9)


# --- probe(): live host tools ----------------------------------------------------------------


def test_is_live_host(tmp_path: Path) -> None:
    (tmp_path / "root-link").symlink_to("/")
    assert hardware._is_live_host(Path("/"))
    assert hardware._is_live_host(tmp_path / "root-link")
    assert not hardware._is_live_host(tmp_path)


def test_live_jetson_runs_nvpmodel_and_dpkg_query(tmp_path: Path, live_host: FakeHost) -> None:
    live_host.results["nvpmodel"] = (0, "NV Power Mode: 25W\n1\n")
    live_host.results["dpkg-query"] = (0, "6.2.1+b38")

    platform = probe(ORIN_PROBE, orin_root(tmp_path))

    assert platform.power_mode == "25W"
    assert platform.jetpack == "6.2.1"
    argv_by_tool = {Path(argv[0]).name: argv[1:] for argv, _ in live_host.calls}
    assert argv_by_tool == {"nvpmodel": ["-q"], "dpkg-query": ["-W", "-f=${Version}", "nvidia-jetpack"]}
    assert all(kwargs["timeout"] == 5.0 for _, kwargs in live_host.calls)


def test_host_tools_are_found_outside_path(
    tmp_path: Path, live_host: FakeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    sbin = tmp_path / "sbin"
    for tool in ("nvpmodel", "dpkg-query"):
        write(sbin / tool, "#!/bin/sh\n").chmod(0o755)
    (tmp_path / "empty").mkdir()
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))  # a minimal service PATH without /usr/sbin
    monkeypatch.setattr(hardware, "_HOST_TOOL_DIRS", (str(sbin),))
    live_host.results["nvpmodel"] = (0, "NV Power Mode: 25W\n1\n")

    assert probe(ORIN_PROBE, orin_root(tmp_path)).power_mode == "25W"
    assert sorted(argv[0] for argv, _ in live_host.calls) == [str(sbin / "dpkg-query"), str(sbin / "nvpmodel")]


def test_live_dgpu_runs_no_jetson_tools(tmp_path: Path, live_host: FakeHost) -> None:
    platform = probe(RTX4090_PROBE, tmp_path)
    assert (platform.power_mode, platform.jetpack) == (None, None)
    assert live_host.calls == []


@pytest.mark.parametrize(
    ("dpkg", "expected"),
    [
        ((0, "6.2.1+b38"), "6.2.1"),
        ((0, "6.2+b77\n"), "6.2"),  # the package wins over the L4T table (R36.4.4 -> 6.2.1)
        ((0, "5.1.2-b104"), "5.1.2"),
        ((0, "1:6.2.2+b10"), "6.2.2"),
        ((1, ""), "6.2.1"),  # meta package absent: L4T table
        ((0, ""), "6.2.1"),
        ((0, "unknown"), "6.2.1"),
        (FileNotFoundError(2, "No such file or directory"), "6.2.1"),
        (subprocess.TimeoutExpired(["dpkg-query"], 5.0), "6.2.1"),
    ],
)
def test_live_jetson_jetpack_version(
    tmp_path: Path, live_host: FakeHost, dpkg: tuple[int, str] | BaseException, expected: str
) -> None:
    live_host.results["dpkg-query"] = dpkg
    assert probe(ORIN_PROBE, orin_root(tmp_path)).jetpack == expected


def test_live_jetson_jetpack_unknown_everywhere(tmp_path: Path, live_host: FakeHost) -> None:
    root = tmp_path / "root"
    write(root / "etc/nv_tegra_release", "# R35 (release), REVISION: 4.1, GCID: 1, BOARD: t186ref\n")
    assert probe(ORIN_PROBE, root).jetpack is None  # dpkg-query missing and R35.4.1 not in the table


@pytest.mark.parametrize(
    ("nvpmodel", "expected"),
    [
        ((0, "NV Power Mode: 25W\n1\n"), "25W"),
        ((0, "NVPM WARN: Golden image context is already created\nNV Power Mode: MAXN_SUPER\n2\n"), "MAXN_SUPER"),
        ((0, "NV Power Mode: \n1\n"), None),
        ((0, "something else\n"), None),
        ((1, "NV Power Mode: 25W\n"), None),
        (subprocess.TimeoutExpired(["nvpmodel", "-q"], 5.0), None),
        (PermissionError(13, "Permission denied"), None),
        (FileNotFoundError(2, "No such file or directory"), None),
    ],
)
def test_live_jetson_power_mode(
    tmp_path: Path, live_host: FakeHost, nvpmodel: tuple[int, str] | BaseException, expected: str | None
) -> None:
    live_host.results["nvpmodel"] = nvpmodel
    assert probe(ORIN_PROBE, orin_root(tmp_path)).power_mode == expected


# --- refresh_clocks() ------------------------------------------------------------------------


def test_refresh_clocks_rereads_caps_and_recomputes_peaks(tmp_path: Path, host: FakeHost) -> None:
    root = orin_root(tmp_path)
    platform = probe(ORIN_PROBE, root)
    write(root / "sys/class/devfreq/17000000.gpu/max_freq", "624750000\n")
    write(root / "sys/kernel/nvpmodel_clk_cap/emc", "2133000000\n")

    refreshed = refresh_clocks(platform)

    assert refreshed.gpu_max_freq_hz == 624_750_000
    assert refreshed.emc_max_freq_hz == 2_133_000_000
    assert refreshed.peak_flops_per_s == pytest.approx(8 * 2048 * 624.75e6)
    assert refreshed.peak_mem_bw_bytes_per_s == pytest.approx(2133e6 * 2 * 16)
    assert refreshed.power_mode is None  # fake sysroot: nvpmodel is not run
    assert (refreshed.gpu_freq_path, refreshed.emc_freq_path) == (platform.gpu_freq_path, platform.emc_freq_path)
    assert refreshed.power_rails == platform.power_rails
    assert platform.gpu_max_freq_hz == 918_000_000  # the input is not modified
    assert host.calls == []


def test_refresh_clocks_keeps_last_known_caps_when_unreadable(tmp_path: Path, host: FakeHost) -> None:
    root = orin_root(tmp_path)
    platform = probe(ORIN_PROBE, root)
    (root / "sys/class/devfreq/17000000.gpu/max_freq").unlink()
    write(root / "sys/class/devfreq/57000000.gpu/max_freq", "1\n")  # nodes are not searched for again
    write(root / "sys/kernel/nvpmodel_clk_cap/emc", "garbage\n")

    assert refresh_clocks(platform) == platform


def test_refresh_clocks_reruns_nvpmodel_on_the_live_host(tmp_path: Path, live_host: FakeHost) -> None:
    live_host.results["nvpmodel"] = (0, "NV Power Mode: 25W\n1\n")
    platform = probe(ORIN_PROBE, orin_root(tmp_path))
    assert platform.power_mode == "25W"
    live_host.calls.clear()

    live_host.results["nvpmodel"] = (0, "NV Power Mode: 15W\n0\n")
    refreshed = refresh_clocks(platform)
    assert refreshed.power_mode == "15W"
    assert live_host.tools == ["nvpmodel"]  # dpkg-query is not re-run

    live_host.results["nvpmodel"] = subprocess.TimeoutExpired(["nvpmodel", "-q"], 5.0)
    assert refresh_clocks(refreshed).power_mode == "15W"  # a failed query keeps the last known mode


def test_refresh_clocks_leaves_dgpu_unchanged(
    tmp_path: Path, live_host: FakeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_nvml(monkeypatch)
    platform = probe(RTX4090_PROBE, tmp_path)
    assert refresh_clocks(platform) is platform
    assert live_host.calls == []


def test_refresh_clocks_on_a_hand_built_platform(host: FakeHost) -> None:
    refreshed = refresh_clocks(make_platform())
    assert refreshed.power_mode == "25W"  # not probed from the live host: kept, nvpmodel not run
    assert refreshed.gpu_max_freq_hz == 918_000_000  # no sysfs path: kept
    assert refreshed.peak_flops_per_s == pytest.approx(15.04e12, rel=1e-3)
    assert refreshed.peak_mem_bw_bytes_per_s == pytest.approx(102.368e9)
    assert host.calls == []


# --- Platform.as_dict() ----------------------------------------------------------------------


def test_as_dict_is_a_json_snapshot_without_paths(tmp_path: Path, live_host: FakeHost) -> None:
    live_host.results["nvpmodel"] = (0, "NV Power Mode: 25W\n1\n")
    snapshot = probe(ORIN_PROBE, orin_root(tmp_path)).as_dict()

    assert list(snapshot) == AS_DICT_KEYS
    assert json.loads(json.dumps(snapshot)) == snapshot
    assert snapshot["power_rails"] == ["VDD_IN", "VDD_CPU_GPU_CV", "VDD_SOC"]
    assert snapshot["energy_rail"] == "VDD_IN"
    assert (snapshot["kind"], snapshot["power_mode"], snapshot["jetpack"]) == ("jetson", "25W", "6.2.1")
    assert snapshot["gpu_max_freq_hz"] == 918_000_000
    assert snapshot["peak_mem_bw_bytes_per_s"] == pytest.approx(102.368e9)
    assert str(tmp_path) not in json.dumps(snapshot)  # no sysfs paths


def test_as_dict_without_rails(tmp_path: Path, host: FakeHost) -> None:
    snapshot = probe(RTX4090_PROBE, tmp_path).as_dict()
    assert list(snapshot) == AS_DICT_KEYS
    assert (snapshot["power_rails"], snapshot["energy_rail"], snapshot["nvml_index"]) == ([], None, None)
    assert make_platform().as_dict()["power_rails"] == []


# --- probe_gpu() -----------------------------------------------------------------------------


def test_probe_gpu_returns_the_probe_event(tmp_path: Path) -> None:
    worker = fake_worker(
        tmp_path,
        '[ "$#" -eq 1 ] && [ "$1" = "--probe" ] || exit 64\n'
        'echo "[edgeserve-worker] probing" >&2\n'
        'echo "stray log line"\n'
        f"printf '%s\\n' '{json.dumps(ORIN_PROBE)}'",
    )
    assert probe_gpu(worker) == ORIN_PROBE


def test_probe_gpu_never_passes_the_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    worker = fake_worker(tmp_path, """printf '{"ev":"probe","key":"%s","tls":"%s"}\\n' "${EDGESERVE_API_KEY:-none}" \
"${EDGESERVE_TLS_KEY:-none}" """)
    monkeypatch.setenv("EDGESERVE_API_KEY", "es-secret")
    monkeypatch.setenv("EDGESERVE_TLS_KEY", "/keys/server.key")
    for env in (None, {"EDGELLM_PLUGIN_PATH": "x.so"}, {"EDGESERVE_API_KEY": "es-secret"}):
        assert probe_gpu(worker, env=env) == {"ev": "probe", "key": "none", "tls": "none"}


def test_probe_gpu_env_extends_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    worker = fake_worker(
        tmp_path, """printf '{"ev":"probe","var":"%s","path":"%s"}\\n' "$EDGESERVE_TEST_VAR" "${PATH:+set}" """
    )
    event = probe_gpu(worker, env={"EDGESERVE_TEST_VAR": "libNvInfer_edgellm_plugin.so"})
    assert event == {"ev": "probe", "var": "libNvInfer_edgellm_plugin.so", "path": "set"}

    monkeypatch.setenv("EDGESERVE_TEST_VAR", "inherited")
    assert probe_gpu(worker)["var"] == "inherited"  # env=None: the current environment unchanged
    assert probe_gpu(worker, env={"EDGESERVE_TEST_VAR": "override"})["var"] == "override"


def test_probe_gpu_error_event_raises_with_the_stderr_tail(tmp_path: Path) -> None:
    worker = fake_worker(
        tmp_path,
        'i=1\nwhile [ "$i" -le 30 ]; do echo "stderr line $i" >&2; i=$((i + 1)); done\n'
        """echo '{"ev":"error","id":null,"code":"no_device","message":"no usable CUDA device: """
        """CUDA driver version is insufficient for CUDA runtime version","fatal":true}'\n"""
        "exit 2",
    )
    with pytest.raises(PlatformError) as info:
        probe_gpu(worker)
    message = str(info.value)
    assert message.startswith("GPU probe failed: no usable CUDA device: CUDA driver version is insufficient")
    assert "stderr line 11" in message and "stderr line 30" in message
    assert "stderr line 10" not in message  # only the last 20 lines


def test_probe_gpu_nonzero_exit_includes_stderr(tmp_path: Path) -> None:
    worker = fake_worker(tmp_path, 'echo "error while loading shared libraries: libnvinfer.so.10" >&2\nexit 127')
    with pytest.raises(PlatformError, match="exited with status 127") as info:
        probe_gpu(worker)
    assert "libnvinfer.so.10" in str(info.value)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("kill -KILL $$", "killed by signal SIGKILL"),
        ('echo "all good" >&2', "printed no probe event"),
        ("echo '{not json'\necho '[1, 2]'", "printed no probe event"),
        ("""echo '{"ev":"probe","sm":87}'\nexit 3""", "exited with status 3"),  # a failed exit wins
        ("""echo '{"ev":"error","code":"no_device"}'""", "GPU probe failed: no_device"),
    ],
    ids=["signal", "no-output", "not-json", "probe-then-exit-3", "error-without-message"],
)
def test_probe_gpu_failures(tmp_path: Path, body: str, expected: str) -> None:
    with pytest.raises(PlatformError, match=expected):
        probe_gpu(fake_worker(tmp_path, body))


def test_probe_gpu_runs_a_bare_file_name_from_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Path(".") / "edgeserve-worker" is Path("edgeserve-worker"): it names a file, not a command on PATH.
    fake_worker(tmp_path, """echo '{"ev":"probe","from":"cwd"}'""")
    decoy = write(tmp_path / "bin" / "edgeserve-worker", """#!/bin/sh\necho '{"ev":"probe","from":"PATH"}'\n""")
    decoy.chmod(0o755)
    monkeypatch.setenv("PATH", f"{decoy.parent}:/usr/bin:/bin")
    monkeypatch.chdir(tmp_path)

    assert probe_gpu(Path(".") / "edgeserve-worker") == {"ev": "probe", "from": "cwd"}
    assert probe_gpu("edgeserve-worker") == {"ev": "probe", "from": "cwd"}  # type: ignore[arg-type]


def test_probe_gpu_missing_binary(tmp_path: Path) -> None:
    missing = tmp_path / "build" / "edgeserve-worker"
    with pytest.raises(PlatformError, match="cannot run the worker binary") as info:
        probe_gpu(missing)
    assert str(missing) in str(info.value)


def test_probe_gpu_binary_not_executable(tmp_path: Path) -> None:
    worker = fake_worker(tmp_path, "exit 0")
    worker.chmod(0o644)
    with pytest.raises(PlatformError, match="Permission denied"):
        probe_gpu(worker)


def test_probe_gpu_timeout(tmp_path: Path) -> None:
    worker = fake_worker(tmp_path, 'echo "initialising CUDA" >&2\nexec sleep 10')
    # 1 s leaves the shell time to write its stderr line even on a loaded machine.
    with pytest.raises(PlatformError, match=r"timed out after 1 s") as info:
        probe_gpu(worker, timeout=1.0)
    assert "initialising CUDA" in str(info.value)
