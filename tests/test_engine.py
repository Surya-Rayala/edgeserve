"""Engine cache (build, reuse, locking, staging, pruning, durability) and the service lifecycle around it."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import edgeserve
from conftest import TESTS, make_tiny_model
from edgeserve import settings as limits
from edgeserve.engine import BuildProfile, EngineCache, EngineError, NativeBuild, _explain_build_failure
from edgeserve.model import fingerprint, load_model
from edgeserve.records import OutputDir
from edgeserve.service import Service
from edgeserve.settings import Settings
from edgeserve.worker import WorkerClient

GPU = {"tensorrt": "10.3.0.30", "sm": 87, "gpu_name": "FakeGPU"}
PROFILE = BuildProfile(max_batch_size=1, max_input_len=64, max_new_tokens=6)
SAMPLING = {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "max_new_tokens": 6}
PACKAGE_PARENT = Path(edgeserve.__file__).resolve().parents[1]


def _install(path: Path, source: str) -> None:
    """Replace a fake native binary with a Python script."""
    script = path.with_name(path.name + "_impl.py")
    script.write_text(source)
    path.write_text(f"#!/bin/sh\nexec {sys.executable} {script} \"$@\"\n")
    path.chmod(0o755)


def slow_builder(native: Path, pid_file: Path, seconds: float = 120.0) -> None:
    """llm_build that records its pid, sleeps, then builds like tests/fake_llm_build.py."""
    _install(native / "examples" / "llm" / "llm_build", f"""
import os, sys, time
with open({str(pid_file)!r}, "a") as f:
    f.write(f"{{os.getpid()}}\\n")
time.sleep({seconds})
os.execv(sys.executable, [sys.executable, {str(TESTS / "fake_llm_build.py")!r}, *sys.argv[1:]])
""")


def builder_pids(pid_file: Path) -> list[int]:
    return [int(line) for line in pid_file.read_text().split()] if pid_file.exists() else []


def wait_until(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def kill(pids: list[int]) -> None:
    for pid in pids:
        if alive(pid):
            os.kill(pid, signal.SIGKILL)


def running(pid: int) -> bool:
    """Alive and not a zombie (an orphan's zombie lingers until whoever adopted it reaps it)."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def staging_dirs(cache: EngineCache) -> list[str]:
    return [p.name for p in cache.root.iterdir() if p.name.startswith(".building-")] if cache.root.exists() else []


@pytest.fixture
def setup(tmp_path, tiny_model, fake_native):
    native = NativeBuild.locate(fake_native)
    model = load_model(tiny_model)
    return SimpleNamespace(native=native, model=model, digest=fingerprint(tiny_model),
                           cache=EngineCache(tmp_path / "cache"), root=tmp_path)


def ensure(s, profile: BuildProfile = PROFILE, cache: EngineCache | None = None, **kwargs):
    return (cache or s.cache).ensure(s.model, s.digest, profile, s.native, GPU, **kwargs)


# --------------------------------------------------------------------------- engine cache

def test_build_then_reuse(setup):
    built = ensure(setup)
    assert built.built_now and built.manifest["complete"] and built.manifest["key"] == built.key
    # the large sidecars are hard links of the model's files, not copies
    assert os.path.samestat(os.stat(built.path / "embedding.safetensors"),
                            os.stat(setup.model.path / "embedding.safetensors"))
    again = ensure(setup, cache=EngineCache(setup.root / "cache"))
    assert not again.built_now and again.path == built.path
    assert (built.path / "invocations.log").read_text().count("\n") == 1  # llm_build ran once


def test_failed_build_is_explained_and_leaves_nothing(setup, monkeypatch):
    monkeypatch.setenv("FAKE_LLM_BUILD_FAIL", "1")
    with pytest.raises(EngineError, match="ran out of memory"):
        ensure(setup)
    assert [p.name for p in setup.cache.root.iterdir() if not p.name.startswith(".lock-")] == []


def test_exception_while_starting_the_build_removes_the_staging_folder(setup):
    (setup.native.llm_build).chmod(0o644)  # Popen raises PermissionError
    with pytest.raises(PermissionError):
        ensure(setup)
    assert staging_dirs(setup.cache) == []


def test_cancel_kills_the_builder_and_removes_its_staging_folder(setup):
    pid_file = setup.root / "builder.pid"
    slow_builder(setup.root / "native", pid_file)
    cancel = threading.Event()
    errors: list[BaseException] = []

    def run() -> None:
        try:
            ensure(setup, cancel=cancel)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        wait_until(lambda: builder_pids(pid_file))
        cancel.set()
        thread.join(10)
        assert not thread.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], EngineError) and "cancelled" in str(errors[0])
        assert not alive(builder_pids(pid_file)[0])
        assert staging_dirs(setup.cache) == []
    finally:
        kill(builder_pids(pid_file))


BUILD_IN_A_SERVER = """
from pathlib import Path
from edgeserve.engine import BuildProfile, EngineCache, NativeBuild
from edgeserve.model import fingerprint, load_model
model = load_model(Path({model!r}))
EngineCache(Path({cache!r})).ensure(model, fingerprint(model.path), BuildProfile(1, 64, 6),
                                    NativeBuild.locate(Path({native!r})), {gpu!r})
"""


def test_builder_dies_with_a_server_that_is_killed(setup):
    # llm_build leads its own session, so neither a SIGKILL nor the SIGHUP of a closed terminal (or a dropped SSH
    # connection) aimed at the server reaches it, and a server that dies that way cannot stop it either.
    pid_file = setup.root / "builder.pid"
    slow_builder(setup.root / "native", pid_file)
    code = BUILD_IN_A_SERVER.format(model=str(setup.model.path), cache=str(setup.root / "cache"),
                                    native=str(setup.root / "native"), gpu=GPU)
    server = subprocess.Popen([sys.executable, "-c", code], cwd=PACKAGE_PARENT)
    try:
        wait_until(lambda: builder_pids(pid_file) or server.poll() is not None, timeout=60)
        assert server.poll() is None, "the build did not start"
        server.kill()
        server.wait(10)
        wait_until(lambda: not running(builder_pids(pid_file)[0]), timeout=10)
    finally:
        if server.poll() is None:
            server.kill()
            server.wait()
        kill(builder_pids(pid_file))


def test_waiting_for_another_build_can_be_cancelled(setup):
    setup.cache.root.mkdir(parents=True)
    with open(setup.cache.root / ".lock-tiny_model", "a") as other:
        fcntl.flock(other, fcntl.LOCK_EX)  # another process is building an engine of this model
        cancel = threading.Event()
        threading.Timer(0.3, cancel.set).start()
        started = time.monotonic()
        with pytest.raises(EngineError, match="cancelled"):
            ensure(setup, cancel=cancel)
        assert time.monotonic() - started < 5


def test_concurrent_builds_of_one_engine_run_llm_build_once(setup):
    pid_file = setup.root / "builder.pid"
    slow_builder(setup.root / "native", pid_file, seconds=0.5)
    results = []
    threads = [threading.Thread(target=lambda: results.append(ensure(setup, cache=EngineCache(setup.root / "cache"))))
               for _ in range(2)]
    for thread in threads:
        thread.start()
        time.sleep(0.1)
    for thread in threads:
        thread.join(20)
    assert len(builder_pids(pid_file)) == 1
    assert sorted(r.built_now for r in results) == [False, True] and results[0].path == results[1].path


def test_staging_folders_of_killed_builds_are_removed(setup):
    setup.cache.root.mkdir(parents=True)
    leftover = setup.cache.root / ".building-tiny_model-0123456789abcdef-424242"  # e.g. SIGKILLed by systemd
    leftover.mkdir()
    (leftover / "llm.engine").write_bytes(b"x" * 1024)
    other_model = setup.cache.root / ".building-tiny_model-v2-0123456789abcdef-424242"
    other_model.mkdir()
    ensure(setup)
    assert not leftover.exists()
    assert other_model.exists()  # another model's build is not ours to judge


def test_prune_only_counts_engines_of_this_model(setup):
    setup.cache.root.mkdir(parents=True)
    others = [setup.cache.root / f"tiny_model-v2-{i:016x}" for i in range(3)]  # model "tiny_model-v2"
    for i, path in enumerate(others):
        path.mkdir()
        os.utime(path, (1_000_000 + i, 1_000_000 + i))
    ensure(setup)
    assert all(path.exists() for path in others)


def test_prune_keeps_the_newest_engines(setup):
    paths = []
    for input_len in (16, 24, 32, 40, 48):
        info = ensure(setup, BuildProfile(1, input_len, 6))
        setup.cache.release()  # not in use: only the age counts
        paths.append(info.path)
        os.utime(info.path, (2_000_000 + input_len, 2_000_000 + input_len))
    assert [p.exists() for p in paths] == [False, False, True, True, True]


def test_prune_leaves_an_engine_that_another_process_uses(setup):
    serving = EngineCache(setup.root / "cache")
    in_use = ensure(setup, BuildProfile(1, 16, 6), cache=serving)  # held until released
    os.utime(in_use.path, (1_000_000, 1_000_000))
    builder = EngineCache(setup.root / "cache")
    for input_len in (24, 32, 40):
        newest = ensure(setup, BuildProfile(1, input_len, 6), cache=builder)
    assert in_use.path.exists()  # the oldest, but in use
    serving.release()
    builder.prune(setup.model.name, keep=newest.path)
    assert not in_use.path.exists()


def test_lookup_works_on_a_cache_it_cannot_modify(setup, monkeypatch):
    built = ensure(setup)

    def refuse(*args, **kwargs):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "utime", refuse)
    found = setup.cache.lookup(setup.model, built.key, PROFILE, GPU["tensorrt"])
    assert found is not None and found.path == built.path


def test_engine_is_flushed_to_disk_before_it_is_published(setup, monkeypatch):
    events: list[tuple[str, str]] = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        events.append(("fsync", os.path.realpath(f"/proc/self/fd/{fd}")))
        return real_fsync(fd)

    def replace(src, dst, *args, **kwargs):
        events.append(("replace", os.path.realpath(src)))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    ensure(setup)
    publish = next(i for i, (kind, path) in enumerate(events) if kind == "replace" and ".building-" in path)
    staging = events[publish][1]
    synced = {path for kind, path in events[:publish] if kind == "fsync"}
    assert {f"{staging}/llm.engine", f"{staging}/manifest.json", f"{staging}/config.json", staging} <= synced


def test_unreadable_build_info_is_explained(fake_native):
    (fake_native / "build_info.json").write_text("{")
    with pytest.raises(EngineError, match="build_info.json"):
        NativeBuild.locate(fake_native)


def test_only_sigkill_or_oom_messages_are_blamed_on_memory(tiny_model):
    model = load_model(tiny_model)
    assert "ran out of memory" in _explain_build_failure(-9, "", model, PROFILE)
    assert "ran out of memory" in _explain_build_failure(137, "", model, PROFILE)
    assert "ran out of memory" in _explain_build_failure(1, "[ERROR] std::bad_alloc", model, PROFILE)
    assert "ran out of memory" not in _explain_build_failure(-11, "Segmentation fault", model, PROFILE)
    assert "ran out of memory" not in _explain_build_failure(-6, "", model, PROFILE)


# --------------------------------------------------------------------------- service lifecycle

def make_settings(tmp_path: Path, model: Path, native: Path, **overrides) -> Settings:
    values = dict(model=model, native_dir=native, cache_dir=tmp_path / "cache", output_dir=tmp_path / "out",
                  max_input_len=64, max_new_tokens=6, max_batch_size=1)
    values.update(overrides)
    return Settings(**values)


async def wait_for_state(service: Service, *states: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while service.state not in states:
        assert time.monotonic() < deadline, f"state is {service.state} ({service.detail}), wanted {states}"
        await asyncio.sleep(0.02)


def fake_record(custom_id: str, attempt_id: str) -> dict:
    return {"custom_id": custom_id, "attempt_id": attempt_id, "status": "completed", "http_status": 200,
            "error": None, "metrics": {}, "output": {"finish_reason": "stop"},
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


async def test_records_of_one_id_are_written_in_commit_order(tmp_path, tiny_model, fake_native, monkeypatch):
    service = Service(make_settings(tmp_path, tiny_model, fake_native))
    service.output = OutputDir(tmp_path / "out")
    write_record = OutputDir.write_record
    calls = []

    def slow_first(self, record):
        calls.append(record["attempt_id"])
        if len(calls) == 1:
            time.sleep(0.3)
        return write_record(self, record)

    monkeypatch.setattr(OutputDir, "write_record", slow_first)
    for attempt in ("att_1", "att_2"):  # a failed attempt, then its immediate retry
        service.commit(SimpleNamespace(custom_id="seg-1", attempt_id=attempt), fake_record("seg-1", attempt))
    await service.stop()
    results = tmp_path / "out" / "results.jsonl"
    for _ in range(100):  # both writes done (stop() waits for them; this test is about their order)
        if results.exists() and results.read_text().count("\n") == 2:
            break
        await asyncio.sleep(0.02)
    stored = json.loads((tmp_path / "out" / "records" / "seg-1.json").read_text())
    assert stored["attempt_id"] == "att_2"  # records/<id>.json holds the latest attempt
    lines = [json.loads(line) for line in (tmp_path / "out" / "results.jsonl").read_text().splitlines()]
    assert [line["id"] for line in lines] == ["att_1", "att_2"]


async def test_stop_writes_pending_records_before_returning(tmp_path, tiny_model, fake_native, monkeypatch):
    service = Service(make_settings(tmp_path, tiny_model, fake_native))
    service.output = OutputDir(tmp_path / "out")
    write_record = OutputDir.write_record

    def slow(self, record):
        time.sleep(0.3)
        return write_record(self, record)

    monkeypatch.setattr(OutputDir, "write_record", slow)
    service.commit(SimpleNamespace(custom_id="last", attempt_id="att_9"), fake_record("last", "att_9"))
    await service.stop()  # under systemd the process is gone right after this (uvicorn re-raises SIGTERM)
    assert (tmp_path / "out" / "records" / "last.json").exists()
    assert "att_9" in (tmp_path / "out" / "results.jsonl").read_text()


async def test_stop_during_an_engine_build_stops_the_builder(tmp_path, tiny_model, fake_native):
    pid_file = tmp_path / "builder.pid"
    slow_builder(fake_native, pid_file)
    service = Service(make_settings(tmp_path, tiny_model, fake_native))
    await service.start()
    try:
        for _ in range(1500):
            if builder_pids(pid_file):
                break
            await asyncio.sleep(0.02)
        await wait_for_state(service, "building_engine")
        await asyncio.wait_for(service.stop(), 15)
        assert not alive(builder_pids(pid_file)[0])
        assert staging_dirs(EngineCache(tmp_path / "cache")) == []
        assert service.state == "stopped"  # a build cut short by a shutdown is not a failure
    finally:
        kill(builder_pids(pid_file))


async def test_failed_state_survives_stop(tmp_path, tiny_model, fake_native, monkeypatch):
    monkeypatch.setenv("FAKE_LLM_BUILD_FAIL", "1")
    service = Service(make_settings(tmp_path, tiny_model, fake_native))
    await service.start()
    await wait_for_state(service, "failed")
    await service.stop()
    assert service.state == "failed" and "out of memory" in service.detail  # `edgeserve serve` exits 1


async def test_stop_still_stops_the_worker_when_draining_fails(tmp_path, tiny_model, fake_native, monkeypatch):
    service = Service(make_settings(tmp_path, tiny_model, fake_native))
    await service.start()
    await wait_for_state(service, "ready")
    worker_pid = service.worker.pid

    async def broken_drain(timeout: float) -> None:
        raise RuntimeError("the dispatcher died")

    monkeypatch.setattr(service.scheduler, "drain", broken_drain)
    service.commit(SimpleNamespace(custom_id="last", attempt_id="att_1"), fake_record("last", "att_1"))
    await service.stop()
    assert not alive(worker_pid) and service.state == "stopped"
    assert "att_1" in (tmp_path / "out" / "results.jsonl").read_text()


async def test_restart_that_cannot_launch_the_worker_ends_failed(tmp_path, tiny_model, fake_native, monkeypatch):
    monkeypatch.setenv("FAKE_CRASH_FILE", str(tmp_path / "crashed-once"))
    monkeypatch.setattr(limits, "MAX_WORKER_RESTARTS", 1)
    service = Service(make_settings(tmp_path, tiny_model, fake_native))
    await service.start()
    try:
        await wait_for_state(service, "ready")
        (fake_native / "edgeserve-worker").chmod(0o644)  # e.g. replaced mid-upgrade: launching raises OSError
        await service.worker.generate([{"prompt": "a b"}], SAMPLING, stream=False)  # the fake worker crashes
        await wait_for_state(service, "failed")
        assert "not restarting" in service.detail
    finally:
        await service.stop()


READY_THEN_EXIT = """
import json, sys
sys.path.insert(0, {tests!r})
from fake_worker import PROBE
if sys.argv[1:] == ["--probe"]:
    print(json.dumps(PROBE), flush=True)
    raise SystemExit(0)
print(json.dumps({{"ev": "ready", "max_batch_size": 1, "cuda_graph": True, "load_ms": 1.0, "warmup_ms": 1.0,
                  "device_free_before_load_bytes": 0, "device_free_after_load_bytes": 0, "device": PROBE}}),
      flush=True)
raise SystemExit(3)
"""


async def test_worker_that_dies_right_after_loading_is_not_served(tmp_path, tiny_model, fake_native, monkeypatch):
    _install(fake_native / "edgeserve-worker", READY_THEN_EXIT.format(tests=str(TESTS)))
    start = WorkerClient.start

    async def start_and_exit(self, timeout: float = 600.0) -> dict:
        ready = await start(self, timeout)
        while self.alive:  # its exit is handled while the service is still "loading"
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.2)
        return ready

    monkeypatch.setattr(WorkerClient, "start", start_and_exit)
    service = Service(make_settings(tmp_path, tiny_model, fake_native))
    await service.start()
    try:
        await wait_for_state(service, "ready", "failed")
        assert service.state == "failed" and "exited" in service.detail
    finally:
        await service.stop()


def test_ensure_without_a_model_lock_holder_does_not_wait(setup):
    other_model = make_tiny_model(setup.root / "models", name="tiny_model-v2")
    cache = EngineCache(setup.root / "cache")
    cache.root.mkdir(parents=True)
    with open(cache.root / ".lock-tiny_model", "a") as busy:
        fcntl.flock(busy, fcntl.LOCK_EX)  # building "tiny_model" does not block "tiny_model-v2"
        info = cache.ensure(load_model(other_model), fingerprint(other_model), PROFILE, setup.native, GPU)
    assert info.built_now


@pytest.mark.parametrize(("flag", "configured", "expected"), [(None, None, 512), (None, 300, 300), (100, 300, 100),
                                                              (None, 0, 512)])
def test_output_limit_comes_from_the_flag_then_generation_config(tmp_path, tiny_model, flag, configured, expected):
    from edgeserve.service import default_max_new_tokens

    (tiny_model / "generation_config.json").write_text(json.dumps({"max_new_tokens": configured}))
    settings = Settings(model=tiny_model, max_new_tokens=flag)
    assert default_max_new_tokens(settings, load_model(tiny_model)) == expected


@pytest.mark.parametrize("configured", [-1, "512", 512.5, True])
def test_unusable_generation_config_output_limit_is_explained(tmp_path, tiny_model, configured):
    from edgeserve.service import StartupError, default_max_new_tokens

    (tiny_model / "generation_config.json").write_text(json.dumps({"max_new_tokens": configured}))
    with pytest.raises(StartupError, match="generation_config.json: max_new_tokens"):
        default_max_new_tokens(Settings(model=tiny_model), load_model(tiny_model))


def test_a_model_without_a_chat_template_is_refused_before_any_build(tmp_path):
    from edgeserve.service import StartupError, check_compatibility

    model = load_model(make_tiny_model(tmp_path, chat_template=False))
    with pytest.raises(StartupError, match="chat_template.jinja"):  # Edge-LLM's builder and runtime both need it
        check_compatibility(model, GPU)


@pytest.mark.parametrize("label", ["INT4 (group 128)", "Int4GroupwiseGemmPlugin", "int4_awq"])
def test_int4_models_are_refused_on_gpus_without_int4_kernels_however_the_format_is_spelled(tmp_path, tiny_model,
                                                                                            label):
    import dataclasses

    from edgeserve.service import StartupError, check_compatibility

    model = dataclasses.replace(load_model(tiny_model), quantization=label)
    check_compatibility(model, GPU)  # sm_87 (Orin) has the kernels
    with pytest.raises(StartupError, match="INT4 GEMM kernels"):
        check_compatibility(model, GPU | {"sm": 90})  # Hopper has none
