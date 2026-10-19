"""``edgeserve service install|remove``: run the server at boot as a systemd service.

Run as the normal user (never with sudo): edgeserve renders the unit for that user and only the
steps that need root (``install`` into /etc/systemd/system and ``systemctl``) run through sudo,
which asks for the password. The service starts the same Python and edgeserve this command was
run with, reads the same settings file, and runs as this user.
"""

from __future__ import annotations

import grp
import os
import pwd
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from edgeserve import config

UNIT_PATH = Path("/etc/systemd/system/edgeserve.service")

_TEMPLATE = """\
# Written by `edgeserve service install`; remove with `edgeserve service remove`.
[Unit]
Description=edgeserve LLM inference server (TensorRT Edge-LLM)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User={user}
{groups}WorkingDirectory=~
Environment={environment}
ExecStart={exec_start}
# The first start may build the TensorRT engine (minutes); readiness is reported on /health/ready.
TimeoutStartSec=infinity
# Shutdown drains in-flight requests for up to 60 s before stopping the worker.
TimeoutStopSec=90
KillMode=mixed
Restart=on-failure
RestartSec=10
# Exit status 2 is a configuration error (the journal says what to fix): restarting cannot help.
RestartPreventExitStatus=2
# The GPU worker raises its own OOM score so it, not this front-end or the rest of the system,
# is chosen if memory runs out; the front-end then reports the failure and restarts it.
OOMScoreAdjust=200
# systemd's default (stop) would stop the whole service when the worker or an engine build is
# OOM-killed, instead of letting the front-end recover and report it.
OOMPolicy=continue
LimitCORE=0
# CUDA needs /dev/nvmap, /dev/nvhost-* and /dev/dri: do not enable PrivateDevices.
NoNewPrivileges=yes

[Install]
WantedBy=multi-user.target
"""


def render(user: str | None = None, python: str | None = None) -> str:
    """The unit for ``user`` (default: the current user) running ``python -m edgeserve serve``."""
    entry = pwd.getpwnam(user) if user else pwd.getpwuid(os.getuid())
    python = python or sys.executable
    member_of = {g.gr_name for g in grp.getgrall() if entry.pw_name in g.gr_mem}
    try:
        member_of.add(grp.getgrgid(entry.pw_gid).gr_name)
    except KeyError:
        pass
    groups = [g for g in ("video", "render") if _group_exists(g) and g not in member_of]
    return _TEMPLATE.format(
        user=entry.pw_name,
        groups=f"SupplementaryGroups={' '.join(groups)}\n" if groups else "",
        environment=_quote(f"EDGESERVE_CONFIG_DIR={config.config_dir()}"),
        exec_start=f"{_quote(python)} -m edgeserve serve",
    )


def _group_exists(name: str) -> bool:
    try:
        grp.getgrnam(name)
    except KeyError:
        return False
    return True


def _quote(value: str) -> str:
    """A unit-file value: quoted (and backslash-escaped) when it holds spaces or quotes."""
    if any(c in value for c in " \t\"'\\"):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return value


def install(print_only: bool = False) -> int:
    if os.geteuid() == 0 and not print_only:
        print("edgeserve service install: run it as the user the server should run as (not with sudo); it asks "
              "for the sudo password itself", file=sys.stderr)
        return 2
    problem = _check_settings()
    if problem and not print_only:
        print(f"edgeserve service install: {problem}", file=sys.stderr)
        return 2
    unit = render()
    commands = [
        ["sudo", "install", "-m", "0644", "<unit file>", str(UNIT_PATH)],
        ["sudo", "systemctl", "daemon-reload"],
        ["sudo", "systemctl", "enable", "edgeserve.service"],
        ["sudo", "systemctl", "restart", "edgeserve.service"],
    ]
    if print_only:
        print(unit, end="")
        print("\n# installed with:")
        for command in commands:
            print("#   " + shlex.join(command))
        return 0
    with tempfile.NamedTemporaryFile("w", suffix=".service", delete=False) as handle:
        handle.write(unit)
    try:
        for command in commands:
            command = [handle.name if part == "<unit file>" else part for part in command]
            if subprocess.run(command).returncode != 0:
                print(f"edgeserve service install: `{shlex.join(command)}` failed", file=sys.stderr)
                return 1
    finally:
        os.unlink(handle.name)
    print(f"installed {UNIT_PATH}: the server starts now and at every boot")
    print("  logs:   journalctl -u edgeserve -f")
    print("  check:  edgeserve check")
    print("  stop:   sudo systemctl stop edgeserve   (start again: sudo systemctl start edgeserve)")
    return 0


def remove() -> int:
    if os.geteuid() == 0:
        print("edgeserve service remove: run it as your user (not with sudo)", file=sys.stderr)
        return 2
    for command in (["sudo", "systemctl", "disable", "--now", "edgeserve.service"],
                    ["sudo", "rm", "-f", str(UNIT_PATH)], ["sudo", "systemctl", "daemon-reload"]):
        subprocess.run(command)
    print("edgeserve service removed (settings, engines and records are kept)")
    return 0


def _check_settings() -> str | None:
    """Refuse to install a service that could only fail (exit status 2 is never restarted)."""
    from edgeserve import nativebuild

    if not config.value("EDGESERVE_MODEL"):
        return "no model is set: run edgeserve setup --model <folder> first (or add EDGESERVE_MODEL to " \
               f"{config.config_file()})"
    model = Path(config.value("EDGESERVE_MODEL") or "").expanduser()
    if not model.is_dir():
        return f"EDGESERVE_MODEL={model} is not a folder"
    if not config.value("EDGESERVE_NATIVE_DIR"):
        state = nativebuild.status()
        if not state.ok:
            return f"{state.reason}: run edgeserve setup first"
    for name in ("EDGESERVE_TLS_CERT", "EDGESERVE_TLS_KEY"):
        value = config.value(name)
        if value and not os.access(Path(value).expanduser(), os.R_OK):
            return f"{name}={value} cannot be read"
    return None


__all__ = ["UNIT_PATH", "install", "remove", "render"]
