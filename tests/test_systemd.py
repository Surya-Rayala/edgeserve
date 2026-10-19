"""``edgeserve service``: the unit it renders and the commands it runs (sudo and systemctl are replaced)."""

from __future__ import annotations

import getpass
import grp
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from edgeserve import config, nativebuild, systemd
from edgeserve import settings as cfg


def _unit(text: str) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "[")) and "=" in line:
            key, _, value = line.partition("=")
            values.setdefault(key, []).append(value)
    return values


def test_the_service_runs_this_python_as_this_user_with_this_settings_folder(isolated_home):
    unit = _unit(systemd.render())
    assert unit["User"] == [getpass.getuser()]
    assert unit["ExecStart"] == [f"{sys.executable} -m edgeserve serve"]
    assert unit["Environment"] == [f"EDGESERVE_CONFIG_DIR={isolated_home / '.config' / 'edgeserve'}"]
    assert "EnvironmentFile" not in unit  # the settings file is read by edgeserve, never put into the environment
    assert unit["WorkingDirectory"] == ["~"]  # the user's home folder


def test_an_oom_kill_of_the_worker_or_a_build_does_not_stop_the_service():
    unit = _unit(systemd.render())
    assert unit["OOMPolicy"] == ["continue"]  # systemd's default (stop) takes the front-end down with the worker
    assert unit["KillMode"] == ["mixed"]  # SIGTERM reaches only the front-end, which drains and stops the worker
    assert "PrivateDevices" not in unit  # CUDA needs /dev/nvmap, /dev/nvhost-* and /dev/dri


def test_the_unit_gives_the_shutdown_drain_its_time_and_the_first_build_no_limit():
    unit = _unit(systemd.render())
    # drain + cancelling what is left (10 s) + the last responses (5 s) + the worker's exit (10 s)
    assert int(unit["TimeoutStopSec"][0]) >= cfg.DRAIN_TIMEOUT_S + 25
    assert unit["TimeoutStartSec"] == ["infinity"]


def test_configuration_errors_are_not_restarted_over_and_over(tmp_path):
    assert "2" in _unit(systemd.render())["RestartPreventExitStatus"][0].split()
    # ... because 2 is how the server exits for a configuration it cannot use
    env = {k: v for k, v in os.environ.items() if not k.startswith("EDGESERVE_")}
    result = subprocess.run([sys.executable, "-m", "edgeserve", "serve"], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 2 and "no model given" in result.stderr


def test_paths_with_spaces_are_quoted(tmp_path, monkeypatch):
    monkeypatch.setenv("EDGESERVE_CONFIG_DIR", str(tmp_path / "my settings"))
    unit = _unit(systemd.render(python=str(tmp_path / "my venv" / "bin" / "python")))
    assert unit["ExecStart"] == [f'"{tmp_path / "my venv" / "bin" / "python"}" -m edgeserve serve']
    assert unit["Environment"] == [f'"EDGESERVE_CONFIG_DIR={tmp_path / "my settings"}"']  # the whole assignment


@pytest.mark.parametrize(("groups", "expected"), [
    ({"primary": [], "video": [], "render": []}, ["video", "render"]),
    ({"primary": [], "video": ["me"], "render": []}, ["render"]),  # already a member: nothing to add
    ({"primary": [], "video": []}, ["video"]),  # no render group on this system
    ({"video": [], "render": []}, ["render"]),  # video is the user's primary group
    ({"primary": []}, None),
])
def test_gpu_groups_are_added_only_when_they_exist_and_are_missing(monkeypatch, groups, expected):
    import pwd

    entries = [SimpleNamespace(gr_name=name, gr_mem=members, gr_gid=gid)
               for gid, (name, members) in enumerate(groups.items(), start=100)]
    by_name = {g.gr_name: g for g in entries}
    me = SimpleNamespace(pw_name="me", pw_gid=by_name.get("primary", entries[0]).gr_gid, pw_dir="/home/me")
    monkeypatch.setattr(pwd, "getpwuid", lambda uid: me)
    monkeypatch.setattr(grp, "getgrall", lambda: entries)
    monkeypatch.setattr(grp, "getgrgid", lambda gid: next(g for g in entries if g.gr_gid == gid))
    monkeypatch.setattr(grp, "getgrnam", lambda name: by_name[name])
    unit = _unit(systemd.render())
    assert unit["User"] == ["me"]
    assert (unit["SupplementaryGroups"][0].split() if "SupplementaryGroups" in unit else None) == expected


def test_the_rendered_unit_is_valid_for_systemd(tmp_path, monkeypatch):
    analyze = shutil.which("systemd-analyze")
    if analyze is None:
        pytest.skip("systemd-analyze is not installed")
    python = tmp_path / "check out" / "bin" / "python"  # a space in the path: ExecStart must quote it
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    monkeypatch.setenv("EDGESERVE_CONFIG_DIR", str(tmp_path / "settings dir"))
    unit = tmp_path / "edgeserve-verify.service"  # not the installed unit's name: its drop-ins must not apply
    unit.write_text(systemd.render(python=str(python)))
    result = subprocess.run([analyze, "verify", str(unit)], capture_output=True, text=True, timeout=120)
    complaints = [line for line in (result.stdout + result.stderr).splitlines() if unit.name in line]
    assert not complaints
    if result.returncode != 0:
        pytest.skip(f"systemd-analyze cannot verify units here: {result.stderr.strip()[:200]}")


# --------------------------------------------------------------------------- install / remove

@pytest.fixture
def commands(monkeypatch) -> list[list[str]]:
    ran: list[list[str]] = []

    def run(command, **kwargs):
        if command[:2] == ["sudo", "install"]:
            ran.append([*command[:-2], Path(command[-2]).read_text(), command[-1]])
        else:
            ran.append(list(command))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(systemd.subprocess, "run", run)
    return ran


@pytest.fixture
def ready(tiny_model, monkeypatch):
    """Settings a service can start with."""
    config.write({"EDGESERVE_MODEL": str(tiny_model)})
    monkeypatch.setattr(nativebuild, "status", lambda root=None: nativebuild.Status(True, "", Path("/rt"), {}))


def test_print_shows_the_unit_and_the_commands_and_changes_nothing(commands, capsys):
    assert systemd.install(print_only=True) == 0
    out = capsys.readouterr().out
    assert "[Service]" in out and "# installed with:" in out and "sudo systemctl enable edgeserve.service" in out
    assert commands == []


def test_install_copies_the_unit_and_starts_the_service(ready, commands, capsys):
    assert systemd.install() == 0
    assert [c[:3] for c in commands] == [["sudo", "install", "-m"], ["sudo", "systemctl", "daemon-reload"],
                                         ["sudo", "systemctl", "enable"], ["sudo", "systemctl", "restart"]]
    assert commands[0][-1] == str(systemd.UNIT_PATH) and commands[0][-2] == systemd.render()
    assert "journalctl -u edgeserve -f" in capsys.readouterr().out


def test_install_refuses_settings_the_service_could_only_fail_with(commands, tiny_model, monkeypatch, capsys):
    assert systemd.install() == 2
    assert "no model is set" in capsys.readouterr().err
    config.write({"EDGESERVE_MODEL": str(tiny_model)})
    assert systemd.install() == 2  # no GPU runtime built yet
    assert "run edgeserve setup first" in capsys.readouterr().err
    monkeypatch.setattr(nativebuild, "status", lambda root=None: nativebuild.Status(True, "", Path("/rt"), {}))
    config.write({"EDGESERVE_TLS_CERT": "/nonexistent/server.crt", "EDGESERVE_TLS_KEY": "/nonexistent/server.key"})
    assert systemd.install() == 2
    assert "EDGESERVE_TLS_CERT=/nonexistent/server.crt cannot be read" in capsys.readouterr().err
    assert commands == []


def test_service_commands_run_as_the_user_not_with_sudo(commands, monkeypatch, capsys):
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    assert systemd.install() == 2 and systemd.remove() == 2
    assert "not with sudo" in capsys.readouterr().err and commands == []


def test_remove_disables_the_service_and_keeps_the_settings(commands, capsys):
    assert systemd.remove() == 0
    assert commands == [["sudo", "systemctl", "disable", "--now", "edgeserve.service"],
                        ["sudo", "rm", "-f", str(systemd.UNIT_PATH)], ["sudo", "systemctl", "daemon-reload"]]
    assert "settings, engines and records are kept" in capsys.readouterr().out
