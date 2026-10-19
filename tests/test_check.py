"""``edgeserve check`` against a real ``edgeserve serve`` (with the fake worker), and its connection diagnostics."""

from __future__ import annotations

import ssl
from types import SimpleNamespace

import certifi
import httpx
from conftest import edgeserve_server
from edgeserve import check


def test_check_passes_against_this_machines_server_with_key_and_tls(tiny_model, fake_native, tmp_path, capsys):
    with edgeserve_server(tiny_model, fake_native, network=True, EDGESERVE_OUTPUT_DIR=str(tmp_path / "out")):
        assert check.run() == 0  # address, API key and CA all come from the settings file
    out = capsys.readouterr().out
    assert "ok   model tiny_model" in out and "ok   streaming" in out and "all checks passed" in out
    assert "ok   record check-" in out


def test_check_without_the_key_says_how_to_set_it(tiny_model, fake_native, monkeypatch, capsys):
    with edgeserve_server(tiny_model, fake_native, network=True) as url:
        monkeypatch.setenv("EDGESERVE_API_KEY", "es-wrong")
        assert check.run(url) == 1
    out = capsys.readouterr().out
    assert "FAIL /v1/models: HTTP 401" in out and "set EDGESERVE_API_KEY to the server's key" in out


def test_check_without_records_still_passes(tiny_model, fake_native, capsys):
    with edgeserve_server(tiny_model, fake_native) as url:
        assert check.run(url) == 0
    assert "the server keeps records only with --output-dir" in capsys.readouterr().out


def test_check_reports_a_server_that_failed_to_start(monkeypatch, capsys):
    made: list[dict] = []

    class Client:
        def __init__(self, **kwargs):
            made.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def get(self, path):
            return SimpleNamespace(status_code=503, json=lambda: {"status": "failed", "detail": "no engine"})

    monkeypatch.setattr(check.httpx, "Client", Client)
    assert check.run("https://edge.test:8000", certifi.where()) == 1
    assert "FAIL the server failed to start: no engine" in capsys.readouterr().out
    verify = made[0]["verify"]
    assert isinstance(verify, ssl.SSLContext) and verify.get_ca_certs()  # httpx 0.28 deprecates verify=<path>


def _wrapped(error: BaseException) -> httpx.ConnectError:
    try:
        raise httpx.ConnectError("failed") from error
    except httpx.ConnectError as wrapped:
        return wrapped


def test_connection_problems_that_waiting_cannot_fix_are_reported_at_once():
    untrusted = ssl.SSLCertVerificationError(1, "certificate verify failed")
    untrusted.verify_message = "self-signed certificate in certificate chain"
    assert "pass --ca-cert" in check._permanent(_wrapped(untrusted), "https://edge.test:8000")
    plain = ssl.SSLError(1, "wrong version number")
    plain.reason = "WRONG_VERSION_NUMBER"
    assert check._permanent(_wrapped(plain), "https://edge.test:8000") == "https://edge.test:8000 does not answer TLS: use http://"
    assert "use https://" in check._permanent(httpx.RemoteProtocolError("closed"), "http://edge.test:8000")
    assert check._permanent(httpx.ConnectError("refused"), "http://edge.test:8000") is None  # not up yet: wait
