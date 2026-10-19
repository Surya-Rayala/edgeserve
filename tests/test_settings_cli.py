"""Settings (flags, environment, validation) and the command line (serve / build dispatch, TLS, shutdown)."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

import edgeserve
from edgeserve import cli
from edgeserve import settings as cfg
from edgeserve.api.app import create_app
from edgeserve.service import Service
from edgeserve.settings import Settings, is_loopback

MESSAGES = [{"role": "system", "content": "Answer briefly."}, {"role": "user", "content": "say hello there"}]
PACKAGE_PARENT = Path(edgeserve.__file__).resolve().parents[1]


# --------------------------------------------------------------------------- settings

@pytest.mark.parametrize(("host", "loopback"), [
    ("127.0.0.1", True), ("127.0.0.2", True), ("::1", True), ("0:0:0:0:0:0:0:1", True), ("::ffff:127.0.0.1", True),
    ("localhost", True), ("LocalHost", True),
    ("", False),  # every interface, for uvicorn / asyncio
    ("0.0.0.0", False), ("::", False), ("0", False), ("192.168.1.10", False), ("fe80::1%eth0", False),
    ("jetson.local", False),
])
def test_loopback_detection(host, loopback):
    assert is_loopback(host) is loopback


def test_empty_host_needs_an_api_key(tmp_path):
    with pytest.raises(ValueError, match="every interface without an API key"):
        Settings(model=tmp_path, host="").validate()
    Settings(model=tmp_path, host="", api_key="secret").validate()
    Settings(model=tmp_path, host="LOCALHOST").validate()


@pytest.mark.parametrize("port", [0, -1, 65536, 70000])
def test_port_out_of_range_is_refused(tmp_path, port):
    with pytest.raises(ValueError, match="--port"):
        Settings(model=tmp_path, port=port).validate()


def test_environment_numbers_are_defaults_and_flags_win(tmp_path, monkeypatch):
    for name, value in (("EDGESERVE_MAX_INPUT_LEN", "2048"), ("EDGESERVE_MAX_NEW_TOKENS", "256"),
                        ("EDGESERVE_MAX_BATCH_SIZE", "2"), ("EDGESERVE_PORT", "9000")):
        monkeypatch.setenv(name, value)
    settings = cfg.from_args(cli._parser().parse_args(["serve", "--model", str(tmp_path)]))
    assert (settings.max_input_len, settings.max_new_tokens, settings.max_batch_size, settings.port) == (2048, 256, 2,
                                                                                                         9000)
    settings = cfg.from_args(cli._parser().parse_args(["serve", "--model", str(tmp_path), "--port", "9100"]))
    assert settings.port == 9100


@pytest.mark.parametrize(("command", "variable"), [
    ("serve", "EDGESERVE_PORT"), ("info", "EDGESERVE_MAX_INPUT_LEN"), ("build", "EDGESERVE_MAX_BATCH_SIZE"),
    ("serve", "EDGESERVE_MAX_NEW_TOKENS"),
])
def test_non_numeric_environment_values_are_a_message(command, variable, monkeypatch, capsys):
    monkeypatch.setenv(variable, "12x")
    assert cli.main([command, "--model", "/nonexistent"]) == 2
    err = capsys.readouterr().err
    assert variable in err and "12x" in err and "Traceback" not in err


# --------------------------------------------------------------------------- serve: TLS and Ctrl-C

def serve_args(tmp_path: Path, model: Path) -> list[str]:
    return ["serve", "--model", str(model), "--cache-dir", str(tmp_path / "cache")]


class _Captured:
    def __init__(self, config, service) -> None:
        self.config, self.service = config, service

    def run(self) -> None:
        pass


def test_missing_tls_files_are_a_message(tmp_path, tiny_model, fake_native, monkeypatch, capsys):
    monkeypatch.setenv("EDGESERVE_NATIVE_DIR", str(fake_native))
    monkeypatch.setenv("EDGESERVE_TLS_CERT", str(tmp_path / "missing.crt"))
    monkeypatch.setenv("EDGESERVE_TLS_KEY", str(tmp_path / "missing.key"))
    monkeypatch.setattr(cli, "_server", lambda config, service: pytest.fail("the server must not start"))
    assert cli.main(serve_args(tmp_path, tiny_model)) == 2
    assert "EDGESERVE_TLS_CERT" in capsys.readouterr().err


def test_tls_uses_strong_ciphers_whatever_uvicorn_defaults_to(tmp_path, tiny_model, fake_native, monkeypatch):
    cert, key = tmp_path / "server.crt", tmp_path / "server.key"
    try:
        subprocess.run(["openssl", "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1",
                        "-nodes", "-keyout", str(key), "-out", str(cert), "-days", "1", "-subj", "/CN=localhost"],
                       check=True, capture_output=True, timeout=60)
    except (OSError, subprocess.CalledProcessError) as e:
        pytest.skip(f"openssl cannot make a test certificate: {e}")
    monkeypatch.setenv("EDGESERVE_NATIVE_DIR", str(fake_native))
    monkeypatch.setenv("EDGESERVE_TLS_CERT", str(cert))
    monkeypatch.setenv("EDGESERVE_TLS_KEY", str(key))
    captured = []
    monkeypatch.setattr(cli, "_server", lambda config, service: captured.append(_Captured(config, service))
                        or captured[-1])
    assert cli.main(serve_args(tmp_path, tiny_model)) == 0
    config = captured[0].config
    assert config.ssl_ciphers == cli._TLS_CIPHERS
    names = [c["name"] for c in config.ssl.get_ciphers()]
    assert names and not any("NULL" in name for name in names)
    assert all(name.startswith(("TLS_", "ECDHE-", "DHE-")) for name in names)  # TLS 1.3 or forward secret


def test_ctrl_c_after_the_graceful_shutdown_is_not_a_traceback(tmp_path, tiny_model, fake_native, monkeypatch):
    class Interrupted(_Captured):
        def run(self) -> None:
            raise KeyboardInterrupt  # what uvicorn does after shutting down on SIGINT

    monkeypatch.setenv("EDGESERVE_NATIVE_DIR", str(fake_native))
    monkeypatch.setattr(cli, "_server", Interrupted)
    try:
        assert cli.main(serve_args(tmp_path, tiny_model)) == 0
    except KeyboardInterrupt:
        pytest.fail("KeyboardInterrupt escaped from edgeserve serve")


# --------------------------------------------------------------------------- build: Ctrl-C

def test_ctrl_c_during_edgeserve_build_stops_llm_build(tmp_path, tiny_model, fake_native):
    pid_file = tmp_path / "builder.pid"
    builder = fake_native / "examples" / "llm" / "llm_build"
    script = tmp_path / "slow_llm_build.py"
    script.write_text(f"import os, time\nopen({str(pid_file)!r}, 'w').write(str(os.getpid()))\ntime.sleep(120)\n")
    builder.write_text(f"#!/bin/sh\nexec {sys.executable} {script} \"$@\"\n")
    env = {name: value for name, value in os.environ.items() if not name.startswith("EDGESERVE_")}
    env["EDGESERVE_NATIVE_DIR"] = str(fake_native)
    proc = subprocess.Popen([sys.executable, "-m", "edgeserve", "build", "--model", str(tiny_model), "--cache-dir",
                             str(tmp_path / "cache"), "--max-input-len", "64", "--max-new-tokens", "6"],
                            cwd=PACKAGE_PARENT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    pid = None
    try:
        deadline = time.monotonic() + 60
        while pid is None:
            assert proc.poll() is None, proc.communicate()[1]
            assert time.monotonic() < deadline, "llm_build did not start"
            if pid_file.exists() and pid_file.read_text():
                pid = int(pid_file.read_text())
            time.sleep(0.05)
        proc.send_signal(signal.SIGINT)
        _, err = proc.communicate(timeout=30)
        assert proc.returncode == 130, err
        assert "interrupted" in err and "Traceback" not in err
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)  # killed, not left running in its own session
        engines = tmp_path / "cache" / "engines"
        assert not [p for p in engines.iterdir() if p.name.startswith(".building-")]
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
        if pid is not None:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


# --------------------------------------------------------------------------- serve: drain before uvicorn's shutdown

def make_settings(tmp_path: Path, model: Path, native: Path) -> Settings:
    return Settings(model=model, native_dir=native, cache_dir=tmp_path / "cache", output_dir=tmp_path / "out",
                    max_input_len=64, max_new_tokens=6, max_batch_size=1)


async def wait_for(predicate, what: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        await asyncio.sleep(0.02)


async def shutdown_during_a_request(tmp_path, tiny_model, fake_native) -> tuple[httpx.Response, Service, list[str]]:
    """Serve over real HTTP, start a request, then shut uvicorn down (as SIGTERM does) while it runs."""
    import uvicorn

    service = Service(make_settings(tmp_path, tiny_model, fake_native))
    config = uvicorn.Config(create_app(service), host="127.0.0.1", port=0, log_config=None,
                            timeout_graceful_shutdown=cli._RESPONSE_GRACE_S)
    server = cli._server(config, service)
    serving = asyncio.create_task(server.serve())
    states: list[str] = []
    try:
        await wait_for(lambda: server.started or serving.done(), "the server to listen")
        port = server.servers[0].sockets[0].getsockname()[1]
        await wait_for(lambda: service.state in ("ready", "failed"), "the service")
        assert service.state == "ready", service.detail
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=30) as client:
            request = asyncio.create_task(client.post("/v1/chat/completions", json={"messages": MESSAGES},
                                                      headers={"X-Request-Id": "inflight-1"}))
            await wait_for(lambda: service.scheduler.running == 1, "the request to run")
            server.should_exit = True
            while not request.done():
                states.append(service.state)
                await asyncio.sleep(0.02)
            response = await request
        await asyncio.wait_for(serving, 30)
    finally:
        if not serving.done():
            server.should_exit = server.force_exit = True
            await asyncio.wait({serving}, timeout=30)
    return response, service, states


async def test_shutdown_lets_running_requests_finish_while_draining(tmp_path, tiny_model, fake_native, monkeypatch):
    monkeypatch.setenv("FAKE_STEP_MS", "150")
    response, service, states = await shutdown_during_a_request(tmp_path, tiny_model, fake_native)
    assert response.status_code == 200
    assert "draining" in states  # /health/ready reports it while requests finish
    record = json.loads((tmp_path / "out" / "records" / "inflight-1.json").read_text())
    assert record["status"] == "completed"
    assert service.state == "stopped"


async def test_shutdown_records_requests_the_drain_had_to_cancel(tmp_path, tiny_model, fake_native, monkeypatch):
    monkeypatch.setenv("FAKE_STEP_MS", "400")  # 6 tokens: longer than the drain allows
    monkeypatch.setattr(cfg, "DRAIN_TIMEOUT_S", 0.3)
    response, service, _ = await shutdown_during_a_request(tmp_path, tiny_model, fake_native)
    record = json.loads((tmp_path / "out" / "records" / "inflight-1.json").read_text())
    # answered (not a dropped connection) and recorded with what the client was told
    assert record["status"] == "cancelled" and record["http_status"] == response.status_code != 200
    assert service.state == "stopped"
