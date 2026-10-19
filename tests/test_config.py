"""The settings file (parsing, writing), where each setting comes from, and which server gets this machine's key."""

from __future__ import annotations

import argparse
import logging
import os
import re
import socket
import stat
from pathlib import Path

import pytest

from edgeserve import config
from edgeserve import settings as cfg
from edgeserve.engine import NativeBuild


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _settings_file(text: str) -> Path:
    path = config.config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _parser(server: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    cfg.add_engine_arguments(parser)
    if server:
        cfg.add_server_arguments(parser)
    return parser


# --------------------------------------------------------------------------- paths

def test_everything_lives_under_the_home_folder(isolated_home):
    assert config.config_file() == isolated_home / ".config" / "edgeserve" / "edgeserve.env"
    assert config.native_current() == isolated_home / ".local" / "share" / "edgeserve" / "native" / "current"
    assert config.default_cache_dir() == isolated_home / ".cache" / "edgeserve"
    assert config.build_root() == isolated_home / ".cache" / "edgeserve" / "native-build"


def test_edgeserve_config_dir_moves_the_settings_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("EDGESERVE_CONFIG_DIR", str(tmp_path / "elsewhere"))
    assert config.config_file() == tmp_path / "elsewhere" / "edgeserve.env"


def test_settings_defaults_are_computed_when_used_not_at_import(isolated_home):
    settings = cfg.Settings(model=Path("/m"))
    assert settings.cache_dir == isolated_home / ".cache" / "edgeserve"
    assert settings.native_dir == config.native_current()


# --------------------------------------------------------------------------- reading

def test_the_file_is_read_as_name_value_lines(caplog):
    _settings_file("# a comment\n\n  EDGESERVE_PORT=9000  \nexport EDGESERVE_HOST=0.0.0.0\nEDGESERVE_API_KEY=\n"
                   "EDGESERVE_MODEL=/models/a b\nEDGESERVE_TYPO=1\n")
    with caplog.at_level(logging.WARNING, logger="edgeserve"):
        values = config.read()
    assert values == {"EDGESERVE_PORT": "9000", "EDGESERVE_HOST": "0.0.0.0", "EDGESERVE_API_KEY": "",
                      "EDGESERVE_MODEL": "/models/a b"}
    assert "line 7: unknown setting EDGESERVE_TYPO" in caplog.text
    assert config.value("EDGESERVE_API_KEY") is None  # empty counts as unset
    assert config.value("EDGESERVE_MODEL") == "/models/a b"


@pytest.mark.parametrize("line", ["EDGESERVE_PORT = 9000", "edgeserve_port=9000", "just words", "=9000"])
def test_a_line_that_is_not_name_value_is_an_error_with_its_line_number(line):
    path = _settings_file(f"EDGESERVE_HOST=127.0.0.1\n{line}\n")
    with pytest.raises(config.ConfigError, match=re.escape(f"{path}, line 2: expected NAME=value")):
        config.read()


def test_a_missing_file_is_no_settings():
    assert config.read() == {} and config.value("EDGESERVE_PORT") is None and config.source("EDGESERVE_PORT") is None


def test_the_file_is_read_again_when_it_changes():
    path = _settings_file("EDGESERVE_PORT=9000\n")
    assert config.value("EDGESERVE_PORT") == "9000"
    path.write_text("EDGESERVE_PORT=9001\n")
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1_000_000))
    assert config.value("EDGESERVE_PORT") == "9001"


def test_a_broken_settings_file_stops_every_command_with_a_message(capsys):
    from edgeserve import cli

    _settings_file("EDGESERVE_PORT 9000\n")
    assert cli.main(["serve", "--model", "/m"]) == 2
    assert "line 1: expected NAME=value" in capsys.readouterr().err


# --------------------------------------------------------------------------- writing

def test_write_creates_a_private_file_from_the_template():
    path = config.write({"EDGESERVE_HOST": "0.0.0.0", "EDGESERVE_API_KEY": "es-x"})
    assert _mode(path) == 0o600 and _mode(path.parent) == 0o700
    text = path.read_text()
    assert "EDGESERVE_HOST=0.0.0.0" in text and "EDGESERVE_API_KEY=es-x" in text
    for name in config.SETTINGS:  # every setting is listed, set or as a commented example
        assert name + "=" in text
    assert config.read() == {"EDGESERVE_HOST": "0.0.0.0", "EDGESERVE_API_KEY": "es-x"}
    assert [p.name for p in path.parent.iterdir()] == ["edgeserve.env"]  # no temporary file left behind


def test_write_keeps_the_users_lines_and_changes_only_the_named_settings():
    path = _settings_file("# my notes\nEDGESERVE_PORT=9000\nEDGESERVE_HOST=127.0.0.1\n# EDGESERVE_MODEL=    (x)\n")
    config.write({"EDGESERVE_HOST": "0.0.0.0", "EDGESERVE_MODEL": "/m", "EDGESERVE_OUTPUT_DIR": "/out",
                  "EDGESERVE_PORT": None})
    assert path.read_text() == "# my notes\nEDGESERVE_HOST=0.0.0.0\nEDGESERVE_MODEL=/m\n\nEDGESERVE_OUTPUT_DIR=/out\n"


# --------------------------------------------------------------------------- precedence

def test_a_flag_beats_the_environment_which_beats_the_file_which_beats_the_default(monkeypatch, tmp_path):
    _settings_file(f"EDGESERVE_MODEL={tmp_path / 'from-file'}\nEDGESERVE_PORT=9000\nEDGESERVE_MAX_INPUT_LEN=2048\n")
    monkeypatch.setenv("EDGESERVE_PORT", "9100")
    resolved = cfg.resolve(_parser().parse_args(["--max-input-len", "1024"]))
    assert resolved["max_input_len"] == (1024, "flag")
    assert resolved["port"] == (9100, "environment")
    assert resolved["model"] == (tmp_path / "from-file", "config file")
    assert resolved["max_batch_size"] == (1, "default")
    settings = cfg.from_args(_parser().parse_args(["--max-input-len", "1024"]))
    assert (settings.max_input_len, settings.port, settings.model) == (1024, 9100, tmp_path / "from-file")


def test_a_bad_number_in_the_file_names_the_file(monkeypatch):
    path = _settings_file("EDGESERVE_MAX_BATCH_SIZE=two\n")
    with pytest.raises(ValueError, match=re.escape(f"EDGESERVE_MAX_BATCH_SIZE (in {path}) must be a whole number")):
        cfg.resolve(_parser().parse_args([]))


def test_secrets_from_the_file_are_never_put_into_the_environment(tmp_path, monkeypatch, fake_native):
    _settings_file(f"EDGESERVE_MODEL={tmp_path}\nEDGESERVE_HOST=0.0.0.0\nEDGESERVE_API_KEY=es-file-secret\n")
    settings = cfg.from_args(_parser().parse_args([]))
    assert settings.api_key == "es-file-secret" and settings.host == "0.0.0.0"
    assert "EDGESERVE_API_KEY" not in os.environ
    # ... and one set in the environment does not reach the Edge-LLM processes either
    monkeypatch.setenv("EDGESERVE_API_KEY", "es-env-secret")
    monkeypatch.setenv("EDGESERVE_TLS_KEY", "/private/server.key")
    env = NativeBuild.locate(fake_native).env()
    assert "EDGESERVE_API_KEY" not in env and "EDGESERVE_TLS_KEY" not in env
    assert env["EDGELLM_PLUGIN_PATH"] == str(fake_native.resolve() / "libNvInfer_edgellm_plugin.so")


def test_the_native_dir_setting_overrides_the_installed_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("EDGESERVE_NATIVE_DIR", str(tmp_path / "my-build"))
    settings = cfg.from_args(_parser().parse_args(["--model", str(tmp_path)]))
    assert settings.native_dir == tmp_path / "my-build"


# --------------------------------------------------------------------------- client target

def _network_file(port: int = 8443, host: str = "0.0.0.0", tls: bool = True) -> Path:
    folder = config.config_dir()
    lines = [f"EDGESERVE_HOST={host}", f"EDGESERVE_PORT={port}", "EDGESERVE_API_KEY=es-local-key"]
    if tls:
        lines += [f"EDGESERVE_TLS_CERT={folder / 'server.crt'}", f"EDGESERVE_TLS_KEY={folder / 'server.key'}"]
    _settings_file("\n".join(lines) + "\n")
    (folder / "edgeserve-ca.crt").write_text("CA")
    return folder / "edgeserve-ca.crt"


def test_without_a_server_clients_use_this_machines_server_key_and_ca():
    ca = _network_file()
    target = config.client_target()
    assert target == config.ClientTarget(url="https://localhost:8443", api_key="es-local-key", ca_cert=ca, local=True)


@pytest.mark.parametrize(("host", "tls", "url"), [
    ("127.0.0.1", False, "http://127.0.0.1:8000"),
    ("0.0.0.0", False, "http://localhost:8000"),
    ("::", True, "https://localhost:8000"),
    ("::1", False, "http://[::1]:8000"),
    ("192.0.2.7", True, "https://192.0.2.7:8000"),
])
def test_the_default_url_follows_the_servers_settings(host, tls, url):
    _network_file(port=8000, host=host, tls=tls)
    assert config.client_target().url == url


def test_with_no_settings_the_default_is_the_local_default_server():
    assert config.client_target() == config.ClientTarget("http://127.0.0.1:8000", None, None, True)


@pytest.mark.parametrize("server", [
    "https://localhost:8443", "HTTPS://LOCALHOST:8443", "https://localhost.:8443", "https://127.0.0.1:8443",
    "https://127.8.9.10", "http://[::1]:8000", "http://[::ffff:127.0.0.1]:8000", "https://user@localhost:8443",
    "https://192.0.2.10:8000",
])
def test_this_machines_key_and_ca_go_to_this_machine(server, monkeypatch):
    monkeypatch.setattr(config, "local_addresses", lambda: {"192.0.2.10"})
    ca = _network_file()
    target = config.client_target(server)
    assert (target.url, target.api_key, target.ca_cert, target.local) == (server, "es-local-key", ca, True)


@pytest.mark.parametrize("server", [
    "https://127.0.0.1.evil.example:8443", "https://localhost.evil.example", "https://localhost@evil.example:8443",
    "https://evil.example/localhost", "https://192.0.2.11:8000", "https://10.0.0.1", "https://[2001:db8::1]:8000",
    "not a url", "https://",
])
def test_this_machines_key_never_goes_to_another_host(server, monkeypatch):
    monkeypatch.setattr(config, "local_addresses", lambda: {"192.0.2.10"})
    _network_file()
    target = config.client_target(server)
    assert (target.api_key, target.ca_cert, target.local) == (None, None, False)


def test_the_host_name_counts_as_this_machine(monkeypatch):
    monkeypatch.setattr(socket, "gethostname", lambda: "Jetson-Lab")
    assert config.this_machine("https://jetson-lab:8000") and config.this_machine("https://JETSON-LAB.local.:8000")
    assert not config.this_machine("https://jetson-lab.example:8000")


def test_a_key_in_the_environment_is_the_callers_choice_and_always_sent(monkeypatch):
    _network_file()
    monkeypatch.setenv("EDGESERVE_API_KEY", "es-mine")
    assert config.client_target("https://other.example:8000").api_key == "es-mine"
    assert config.client_target().api_key == "es-mine"


def test_a_given_ca_certificate_is_always_used(tmp_path):
    _network_file()
    target = config.client_target("https://other.example:8000", tmp_path / "their-ca.crt")
    assert target.ca_cert == tmp_path / "their-ca.crt" and target.api_key is None
