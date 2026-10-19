"""``edgeserve setup``: settings, API key and certificates (real openssl), and when it builds the GPU runtime (the
build itself is replaced: see tests/test_nativebuild.py)."""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import stat
import subprocess
from pathlib import Path

import pytest

from edgeserve import config, nativebuild, setupcmd, tls
from edgeserve import settings as cfg

FILES = ["edgeserve-ca.crt", "edgeserve-ca.key", "edgeserve.env", "server.crt", "server.key"]

pytestmark = pytest.mark.skipif(shutil.which("openssl") is None, reason="needs the openssl command")


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    """The GPU runtime: up to date unless a test says otherwise; builds and self-tests are recorded, not run."""
    state = {"ok": True, "changes": [], "built": 0, "self_test": True}
    root = tmp_path / "runtime"

    def status(root_=None):
        if state["ok"]:
            return nativebuild.Status(True, "", root, {"cuda": "12.6", "tensorrt": "10.3.0.30"})
        return nativebuild.Status(False, "the GPU runtime is not built yet", None, None)

    def build(progress):
        state["built"] += 1
        if state.get("fail"):
            raise nativebuild.NativeBuildError("the GPU runtime build failed (exit 2)")
        return root

    monkeypatch.setattr(nativebuild, "status", status)
    monkeypatch.setattr(nativebuild, "toolchain_changes", lambda info: state["changes"])
    monkeypatch.setattr(nativebuild, "build", build)
    monkeypatch.setattr(nativebuild, "self_test", lambda root_, progress: state["self_test"])
    monkeypatch.setattr(setupcmd, "prerequisites", lambda: state.get("missing", []))
    return state


def _folder() -> Path:
    return config.config_dir()


def _snapshot() -> dict[str, bytes]:
    return {name: (_folder() / name).read_bytes() for name in FILES if (_folder() / name).exists()}


def _signed_by_the_ca() -> str:
    folder = _folder()
    verify = subprocess.run(["openssl", "verify", "-CAfile", str(folder / tls.CA_CERT), str(folder / tls.SERVER_CERT)],
                            capture_output=True, text=True, timeout=60)
    assert verify.returncode == 0, verify.stdout + verify.stderr
    return subprocess.run(["openssl", "x509", "-in", str(folder / tls.SERVER_CERT), "-noout", "-ext", "subjectAltName"],
                          capture_output=True, text=True, timeout=60).stdout


def _settings(model: Path) -> cfg.Settings:
    parser = argparse.ArgumentParser()
    cfg.add_engine_arguments(parser)
    cfg.add_server_arguments(parser)
    return cfg.from_args(parser.parse_args(["--model", str(model)]))


# --------------------------------------------------------------------------- --network

def test_network_setup_writes_a_private_config_key_and_certificates(runtime, tiny_model, capsys):
    assert setupcmd.run(network=True) == 0
    out = capsys.readouterr().out
    folder = _folder()
    assert sorted(p.name for p in folder.iterdir()) == FILES  # no CSR, extension or serial file left behind
    assert _mode(folder) == 0o700
    assert [_mode(folder / n) for n in ("edgeserve-ca.key", "server.key", "edgeserve.env")] == [0o600] * 3
    assert [_mode(folder / n) for n in ("edgeserve-ca.crt", "server.crt")] == [0o644] * 2
    names = _signed_by_the_ca()
    assert "DNS:localhost" in names and "IP Address:127.0.0.1" in names and f"DNS:{socket.gethostname()}" in names
    assert "+++" not in out  # openssl's key-generation progress is not shown

    values = config.read()
    key = values["EDGESERVE_API_KEY"]
    assert key.startswith("es-") and len(key) > 40
    assert out.count(key) == 1  # shown once, when it is created
    settings = _settings(tiny_model)  # a configuration the server accepts
    assert (settings.host, settings.api_key) == ("0.0.0.0", key)
    assert (settings.tls_cert, settings.tls_key) == (folder / "server.crt", folder / "server.key")
    assert config.client_target().url == f"https://localhost:{settings.port}"

    before = _snapshot()
    assert setupcmd.run(network=True) == 0  # again: nothing changes, and the key is not shown again
    again = capsys.readouterr().out
    assert _snapshot() == before and key not in again and "API key: EDGESERVE_API_KEY in the settings file" in again


def test_a_new_server_certificate_comes_from_the_ca_the_clients_already_trust(runtime):
    assert setupcmd.run(network=True) == 0
    before = _snapshot()
    for name in ("server.crt", "server.key"):  # e.g. deleted by hand after the machine's address changed
        (_folder() / name).unlink()
    assert setupcmd.run(network=True) == 0
    after = _snapshot()
    assert after["edgeserve-ca.crt"] == before["edgeserve-ca.crt"] and after["edgeserve.env"] == before["edgeserve.env"]
    assert after["server.crt"] != before["server.crt"]
    _signed_by_the_ca()


def test_the_certificates_are_kept_when_the_ca_key_was_moved_off_the_machine(runtime):
    assert setupcmd.run(network=True) == 0
    folder = _folder()
    before = _snapshot()
    (folder / "edgeserve-ca.key").unlink()  # only needed to sign server certificates, so it may be kept elsewhere
    assert setupcmd.run(network=True) == 0
    assert _snapshot() == {k: v for k, v in before.items() if k != "edgeserve-ca.key"}

    # A new server certificate cannot be signed without that key: only then is a new CA made.
    for name in ("server.crt", "server.key"):
        (folder / name).unlink()
    assert setupcmd.run(network=True) == 0
    assert (folder / "edgeserve-ca.crt").read_bytes() != before["edgeserve-ca.crt"]
    _signed_by_the_ca()


def test_the_server_certificate_is_renewed_for_new_addresses_and_before_it_expires(tmp_path, monkeypatch):
    folder = tmp_path / "tls"
    first = tls.ensure(folder, tls.Names(dns=("box", "localhost"), ips=("127.0.0.1",)))
    assert first.ca_created and first.server_issued
    same = tls.ensure(folder, tls.Names(dns=("box", "localhost"), ips=("127.0.0.1",)))
    assert not same.ca_created and not same.server_issued
    moved = tls.ensure(folder, tls.Names(dns=("box", "localhost"), ips=("127.0.0.1", "192.0.2.9")))
    assert moved.server_issued and not moved.ca_created and moved.reason == "it does not cover 192.0.2.9"

    monkeypatch.setattr(tls, "SERVER_DAYS", 10)
    tls.ensure(folder, tls.Names(dns=("other",), ips=()))
    monkeypatch.setattr(tls, "SERVER_DAYS", 825)
    soon = tls.ensure(folder, tls.Names(dns=("other",), ips=()))
    assert soon.server_issued and soon.reason.startswith("it expires on ")


def test_openssl_errors_are_shown(runtime, tmp_path, monkeypatch, capsys):
    fake = tmp_path / "bin" / "openssl"
    fake.parent.mkdir()
    fake.write_text("#!/bin/sh\necho 'req: Unrecognized flag addext' >&2\nexit 1\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake.parent}{os.pathsep}{os.environ['PATH']}")
    assert setupcmd.run(network=True) == 1
    err = capsys.readouterr().err
    assert "openssl req failed" in err and "req: Unrecognized flag addext" in err
    assert not config.config_file().exists()


@pytest.mark.parametrize(("host", "kept"), [("192.0.2.5", "192.0.2.5"), ("127.0.0.1", "0.0.0.0"), ("", "0.0.0.0")])
def test_network_setup_keeps_a_chosen_listen_address(runtime, host, kept):
    config.write({"EDGESERVE_HOST": host or None, "EDGESERVE_PORT": "8443"})
    assert setupcmd.run(network=True) == 0
    assert config.read()["EDGESERVE_HOST"] == kept and config.read()["EDGESERVE_PORT"] == "8443"


# --------------------------------------------------------------------------- model, local use

def test_setup_records_the_model_as_an_absolute_path(runtime, tiny_model, monkeypatch, capsys):
    monkeypatch.chdir(tiny_model.parent)
    assert setupcmd.run(model=Path(tiny_model.name)) == 0
    assert config.read() == {"EDGESERVE_MODEL": str(tiny_model.resolve())}
    out = capsys.readouterr().out
    assert "serving on this machine only" in out and "Next: edgeserve serve" in out
    assert not (_folder() / "server.crt").exists()


def test_setup_refuses_a_folder_that_is_not_a_model(runtime, tmp_path, capsys):
    assert setupcmd.run(model=tmp_path / "missing") == 2
    assert "no such folder" in capsys.readouterr().err
    (tmp_path / "empty").mkdir()
    assert setupcmd.run(model=tmp_path / "empty") == 2
    assert "config.json" in capsys.readouterr().err
    assert not config.config_file().exists()


def test_setup_refuses_to_run_as_root(runtime, monkeypatch, capsys):
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    assert setupcmd.run() == 2
    assert "not as root" in capsys.readouterr().err and runtime["built"] == 0


# --------------------------------------------------------------------------- building the runtime

def test_an_up_to_date_runtime_is_not_rebuilt(runtime, capsys):
    assert setupcmd.run() == 0
    assert runtime["built"] == 0 and "GPU runtime: up to date" in capsys.readouterr().out


def test_a_missing_runtime_or_a_new_toolchain_is_built_and_checked(runtime, capsys):
    runtime["ok"] = False
    assert setupcmd.run() == 0
    out = capsys.readouterr().out
    assert runtime["built"] == 1 and "attention kernels match the reference" in out
    runtime.update(ok=True, changes=["CUDA 12.6 -> 12.8"])
    assert setupcmd.run() == 0
    assert runtime["built"] == 2 and "GPU runtime: CUDA 12.6 -> 12.8; building it" in capsys.readouterr().out


def test_missing_prerequisites_are_listed_together_with_the_apt_command(runtime, capsys):
    runtime.update(ok=False, missing=["git is not installed (package git)",
                                      "CMake 3.20 or newer is not installed (package cmake)", "nvcc missing"])
    assert setupcmd.run() == 2
    err = capsys.readouterr().err
    assert "git is not installed" in err and "nvcc missing" in err and "sudo apt install git cmake" in err
    assert runtime["built"] == 0


def test_a_failed_build_or_self_test_fails_setup(runtime, capsys):
    runtime.update(ok=False, fail=True)
    assert setupcmd.run() == 1
    assert "build failed" in capsys.readouterr().err
    runtime.update(fail=False, self_test=False)
    assert setupcmd.run() == 1
    assert "attention-kernel check FAILED" in capsys.readouterr().err


def test_prerequisites_reports_every_missing_tool(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(os, "access", lambda path, mode: False)
    monkeypatch.setattr(nativebuild, "tensorrt_header", lambda prefix=None: None)
    problems = setupcmd.prerequisites()
    for word in ("git", "CMake", "C++ compiler", "virtual environments", "nvcc", "NvInferVersion.h"):
        assert any(word in item for item in problems), word
    assert "sudo apt install git cmake build-essential python3-venv" in setupcmd._install_hint(problems)
