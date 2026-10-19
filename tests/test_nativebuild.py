"""The GPU runtime's build and install (edgeserve.nativebuild), run with a stand-in build.sh: nothing is compiled."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import threading
from pathlib import Path

import pytest

from edgeserve import __version__, cli, config, nativebuild
from edgeserve.engine import NativeBuild

FAKE_BUILD_SH = """#!/usr/bin/env bash
set -euo pipefail
echo "fake build from $EDGESERVE_SOURCES into $EDGESERVE_BUILD_ROOT"
printf '%s' "${EDGESERVE_API_KEY-unset}" > "$EDGESERVE_BUILD_ROOT/key-seen"
[[ -f "$EDGESERVE_SOURCES/fail" ]] && { echo "compiler exploded"; exit 3; }
B="$EDGESERVE_BUILD_ROOT/build"
mkdir -p "$B/examples/llm"
for f in edgeserve-worker attention-check examples/llm/llm_build; do
  printf '#!/bin/sh\\necho all ok\\n' > "$B/$f"; chmod +x "$B/$f"
done
echo plugin > "$B/libNvInfer_edgellm_plugin.so.0.11.0"
ln -sf libNvInfer_edgellm_plugin.so.0.11.0 "$B/libNvInfer_edgellm_plugin.so"
plugin="$(cat "$EDGESERVE_SOURCES/plugin-version" 2>/dev/null || echo p1)"
printf '{"edgellm_commit": "c", "cuda": "12.6", "tensorrt": "10.3.0.30", "plugin_sha256": "%s"}' "$plugin" \\
  > "$B/build_info.json"
"""


@pytest.fixture
def sources(tmp_path, monkeypatch) -> Path:
    """Packaged sources with a stand-in build.sh."""
    root = tmp_path / "packaged-native"
    (root / "edgellm" / "patches").mkdir(parents=True)
    (root / "edgellm" / "COMMIT").write_text("0" * 40 + "\n")
    (root / "edgellm" / "patches" / "0001-x.patch").write_text("--- a\n+++ b\n")
    (root / "worker.cpp").write_text("int main() {}\n")
    (root / "build.sh").write_text(FAKE_BUILD_SH)
    monkeypatch.setattr(nativebuild, "sources_dir", lambda: root)
    return root


def _fake_build_dir(root: Path, plugin: str) -> Path:
    build = root / f"build-{plugin}"
    (build / "examples" / "llm").mkdir(parents=True)
    for name in ("edgeserve-worker", "attention-check", "examples/llm/llm_build", "libNvInfer_edgellm_plugin.so"):
        (build / name).write_text(name)
    (build / "build_info.json").write_text(json.dumps({"plugin_sha256": plugin, "cuda": "12.6"}))
    return build


def _versions() -> list[str]:
    return sorted(p.name for p in config.native_root().iterdir() if p.is_dir() and not p.is_symlink())


# --------------------------------------------------------------------------- packaged sources

def test_the_real_packaged_sources_are_complete():
    names = {p.relative_to(nativebuild.sources_dir()).as_posix() for p in nativebuild.source_files()}
    assert {"worker.cpp", "attention_check.cpp", "CMakeLists.txt", "build.sh", "edgellm/COMMIT"} <= names
    assert len([n for n in names if n.startswith("edgellm/patches/") and n.endswith(".patch")]) == 5
    assert len(nativebuild.edgellm_commit()) == 40 and int(nativebuild.edgellm_commit(), 16) >= 0


def test_the_source_hash_covers_names_and_contents_but_not_caches(sources):
    first = nativebuild.source_sha256()
    (sources / "__pycache__").mkdir()
    (sources / "__pycache__" / "x.pyc").write_bytes(b"x")
    assert nativebuild.source_sha256() == first
    (sources / "worker.cpp").write_text("int main() { return 1; }\n")
    second = nativebuild.source_sha256()
    assert second != first
    (sources / "worker.cpp").rename(sources / "worker2.cpp")
    assert nativebuild.source_sha256() not in (first, second)


# --------------------------------------------------------------------------- status

def test_status_says_what_is_wrong_with_the_installed_runtime(sources, tmp_path):
    state = nativebuild.status()
    assert not state.ok and state.reason == "the GPU runtime is not built yet"
    root = nativebuild.install(_fake_build_dir(tmp_path, "p1"), nativebuild.source_sha256())
    state = nativebuild.status()
    assert state.ok and state.root == root and state.info["edgeserve_version"] == __version__
    (sources / "worker.cpp").write_text("changed\n")  # an upgrade of edgeserve changed the sources
    state = nativebuild.status()
    assert not state.ok and state.reason == f"the GPU runtime was built by edgeserve {__version__} from other sources"
    (root / "attention-check").unlink()
    assert "is incomplete (missing attention-check)" in nativebuild.status().reason
    (root / "build_info.json").write_text("{")
    assert nativebuild.status().reason.startswith("cannot read")


def test_serve_and_build_refuse_a_missing_or_stale_runtime(tiny_model, capsys):
    for command in ("serve", "build"):
        assert cli.main([command, "--model", str(tiny_model)]) == 2
        assert "the GPU runtime is not built yet: run edgeserve setup" in capsys.readouterr().err


def test_info_reports_a_missing_runtime_and_the_fix(capsys):
    assert cli.main(["info"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["gpu_runtime"]["ok"] is False and report["gpu_runtime"]["fix"] == "run edgeserve setup"
    assert report["settings"]["port"] == {"value": "8000", "from": "default"}


def test_info_still_checks_the_model_without_a_runtime(tiny_model, capsys):
    # e.g. on a PC that only exports: the model folder is checked, the GPU parts are left out
    assert cli.main(["info", "--model", str(tiny_model)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["gpu_runtime"]["fix"] == "run edgeserve setup"
    assert report["model"]["num_layers"] > 0 and report["memory_estimate_gib"]["total"] > 0
    assert "engine_cache" not in report and "error" not in report


def test_toolchain_changes_compare_cuda_and_tensorrt(monkeypatch):
    monkeypatch.setattr(nativebuild, "_cuda_version", lambda: "12.8")
    monkeypatch.setattr(nativebuild, "_tensorrt_version", lambda: "10.3.0.30")
    assert nativebuild.toolchain_changes({"cuda": "12.6", "tensorrt": "10.3.0.30"}) == ["CUDA 12.6 -> 12.8"]
    monkeypatch.setattr(nativebuild, "_cuda_version", lambda: None)  # nvcc not found: cannot tell, no rebuild
    assert nativebuild.toolchain_changes({"cuda": "12.6", "tensorrt": "10.2.0.19"}) == ["TensorRT 10.2.0.19 -> 10.3.0.30"]


@pytest.mark.parametrize(("header", "version"), [
    ("#define NV_TENSORRT_MAJOR 10 //!< major\n#define NV_TENSORRT_MINOR 3\n#define NV_TENSORRT_PATCH 0\n"
     "#define NV_TENSORRT_BUILD 30\n", "10.3.0.30"),
    ("#ifndef NV_INFER_VERSION_H\n#define NV_INFER_VERSION_H\n#define TRT_MAJOR_ENTERPRISE 10\n"
     "#define TRT_MINOR_ENTERPRISE 16\n#define TRT_PATCH_ENTERPRISE 0\n#define TRT_BUILD_ENTERPRISE 12\n"
     "#define NV_TENSORRT_MAJOR TRT_MAJOR_ENTERPRISE\n#define NV_TENSORRT_MINOR TRT_MINOR_ENTERPRISE\n"
     "#define NV_TENSORRT_PATCH TRT_PATCH_ENTERPRISE\n#define NV_TENSORRT_BUILD TRT_BUILD_ENTERPRISE\n", "10.16.0.12"),
])
def test_the_tensorrt_version_is_read_from_old_and_new_headers(tmp_path, monkeypatch, header, version):
    include = tmp_path / "trt" / "include" / f"{os.uname().machine}-linux-gnu"
    include.mkdir(parents=True)
    (include / "NvInferVersion.h").write_text(header)
    monkeypatch.setenv("TRT_PACKAGE_DIR", str(tmp_path / "trt"))
    assert nativebuild.tensorrt_header() == include / "NvInferVersion.h"
    assert nativebuild._tensorrt_version() == version


# --------------------------------------------------------------------------- install

def test_install_makes_a_version_folder_and_switches_current_atomically(sources, tmp_path):
    digest = nativebuild.source_sha256()
    first = nativebuild.install(_fake_build_dir(tmp_path, "p1"), digest)
    assert first.name == hashlib.sha256(f"{digest}:p1".encode()).hexdigest()[:16]
    assert config.native_current().resolve() == first
    assert os.readlink(config.native_current()) == first.name  # relative: the folder can be moved as a whole
    info = json.loads((first / "build_info.json").read_text())
    assert info["source_sha256"] == digest and info["edgeserve_version"] == __version__ and info["built_at"]
    located = NativeBuild.locate(config.native_current())
    assert located.root == first and located.worker == first / "edgeserve-worker"

    second = nativebuild.install(_fake_build_dir(tmp_path, "p2"), digest)
    assert config.native_current().resolve() == second and _versions() == sorted([first.name, second.name])
    assert located.root == first and located.worker.exists()  # a running server keeps its binaries
    os.utime(first, (1, 1))
    os.utime(second, (2, 2))
    third = nativebuild.install(_fake_build_dir(tmp_path, "p3"), digest)
    assert _versions() == sorted([second.name, third.name])  # the current one and the newest other one
    assert not [p for p in config.native_root().iterdir() if p.name.startswith(".")]  # no staging leftovers

    again = nativebuild.install(tmp_path / "build-p2", digest)
    assert again == second and config.native_current().resolve() == second  # an existing version is reused


def test_the_plugin_symlink_chain_is_installed_as_a_file(sources, tmp_path):
    build = _fake_build_dir(tmp_path, "p1")
    (build / "libNvInfer_edgellm_plugin.so").unlink()
    (build / "libNvInfer_edgellm_plugin.so.0.11.0").write_text("real plugin")
    (build / "libNvInfer_edgellm_plugin.so").symlink_to("libNvInfer_edgellm_plugin.so.0.11.0")
    root = nativebuild.install(build, nativebuild.source_sha256())
    plugin = root / "libNvInfer_edgellm_plugin.so"
    assert not plugin.is_symlink() and plugin.read_text() == "real plugin"


def test_the_trt_library_folder_of_a_tarball_install_is_put_on_the_loader_path(tmp_path, fake_native, monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/other")
    info = json.loads((fake_native / "build_info.json").read_text())
    for lib_dir, expected in (("/opt/TensorRT-10.9/lib", "/opt/TensorRT-10.9/lib:/opt/other"),
                              ("/usr/lib/aarch64-linux-gnu", "/opt/other")):
        (fake_native / "build_info.json").write_text(json.dumps(info | {"trt_lib_dir": lib_dir}))
        assert NativeBuild.locate(fake_native).env()["LD_LIBRARY_PATH"] == expected


def test_serving_refuses_a_tensorrt_library_the_runtime_was_not_built_for(fake_native):
    from edgeserve.service import StartupError, check_runtime

    native = NativeBuild.locate(fake_native)  # built for TensorRT 10.3.0.30
    check_runtime(native, {"tensorrt_runtime": "10.3.0"})
    check_runtime(native, {})  # an old worker that does not report it: nothing to compare
    with pytest.raises(StartupError, match=r"built for TensorRT 10\.3\.0\.30, but this machine now has TensorRT "
                                           r"10\.4\.1: run edgeserve setup"):
        check_runtime(native, {"tensorrt_runtime": "10.4.1"})


# --------------------------------------------------------------------------- build

def test_build_copies_the_sources_runs_build_sh_and_installs(sources, monkeypatch, capsys):
    monkeypatch.setenv("EDGESERVE_API_KEY", "es-secret")
    messages: list[str] = []
    root = nativebuild.build(progress=messages.append)
    build_root = config.build_root()
    assert root == config.native_current().resolve() and nativebuild.status().ok
    assert nativebuild.source_sha256(build_root / "sources") == nativebuild.source_sha256()
    assert (build_root / "key-seen").read_text() == "unset"  # the API key never reaches the build
    assert "fake build from " + str(build_root / "sources") in capsys.readouterr().out  # shown while it runs
    assert "fake build from" in (build_root / "build.log").read_text()  # and kept
    assert any("building the GPU runtime" in m for m in messages)

    inode = (build_root / "sources" / "worker.cpp").stat().st_ino
    nativebuild.build(progress=messages.append)
    assert (build_root / "sources" / "worker.cpp").stat().st_ino == inode  # unchanged sources are not copied again
    (sources / "plugin-version").write_text("p2")
    second = nativebuild.build(progress=messages.append)
    assert second != root and (build_root / "sources" / "plugin-version").read_text() == "p2"


def test_a_failed_build_says_where_the_log_is(sources):
    (sources / "fail").write_text("")
    with pytest.raises(nativebuild.NativeBuildError, match=r"failed \(exit 3\).*build\.log"):
        nativebuild.build(progress=lambda message: None)
    assert "compiler exploded" in (config.build_root() / "build.log").read_text()
    assert not nativebuild.status().ok


def test_a_second_build_waits_for_the_first(sources):
    root = config.build_root()
    root.mkdir(parents=True)
    waiting, done = threading.Event(), threading.Event()
    messages: list[str] = []

    def progress(message: str) -> None:
        messages.append(message)
        if message.startswith("another edgeserve setup is building"):
            waiting.set()

    with open(root / ".lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        thread = threading.Thread(target=lambda: (nativebuild.build(progress=progress), done.set()))
        thread.start()
        assert waiting.wait(30)
        assert not done.is_set()
    thread.join(60)
    assert done.is_set() and nativebuild.status().ok


# --------------------------------------------------------------------------- self-test

@pytest.mark.parametrize(("script", "ok"), [("echo 'fmha ok'; echo 'all ok'", True),
                                            ("echo 'xqa MISMATCH'; exit 1", False)])
def test_the_self_test_runs_attention_check_and_records_the_result(tmp_path, script, ok, monkeypatch):
    monkeypatch.setenv("EDGESERVE_API_KEY", "es-secret")  # never passed on
    root = tmp_path / "runtime"
    root.mkdir()
    checker = root / "attention-check"
    checker.write_text(f'#!/bin/sh\n[ -n "$EDGELLM_PLUGIN_PATH" ] || exit 9\n[ -z "$EDGESERVE_API_KEY" ] || exit 8\n'
                       f'{script}\n')
    checker.chmod(checker.stat().st_mode | stat.S_IXUSR)
    (root / "build_info.json").write_text("{}")
    messages: list[str] = []
    assert nativebuild.self_test(root, progress=messages.append) is ok
    assert json.loads((root / "build_info.json").read_text())["self_test"]["ok"] is ok
    assert ("xqa MISMATCH" in "\n".join(messages)) is not ok
