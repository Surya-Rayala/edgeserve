"""Where edgeserve keeps its files, and its configuration file.

Everything lives under the user's home folder, so a shell and the systemd service see the same files:

    ~/.config/edgeserve/edgeserve.env     settings (EDGESERVE_CONFIG_DIR moves the whole folder)
    ~/.config/edgeserve/*.crt, *.key      TLS certificates made by ``edgeserve setup --network``
    ~/.config/edgeserve/client-keys.json  the clients' API keys (``edgeserve keys``; SHA-256 hashes only)
    ~/.local/share/edgeserve/native/      the GPU runtime built by ``edgeserve setup`` (versions + ``current``)
    ~/.cache/edgeserve/engines/           TensorRT engines built from models (EDGESERVE_CACHE_DIR moves it)
    ~/.cache/edgeserve/native-build/      the runtime's build tree (safe to delete)

The settings file holds ``EDGESERVE_NAME=value`` lines (no quotes) and ``#`` comment lines. Every
command resolves a setting as: command-line flag, else environment variable, else this file, else
the built-in default. File values are never copied into the environment, so secrets such as the
API key do not reach child processes.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import socket
import subprocess
import tempfile
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("edgeserve")

CONFIG_FILE_NAME = "edgeserve.env"
CA_CERT_NAME = "edgeserve-ca.crt"

# Every setting the file (or the environment) may hold, with its meaning (also used to write the file).
SETTINGS = {
    "EDGESERVE_MODEL": "model folder to serve (an Edge-LLM export); the --model flag",
    "EDGESERVE_MAX_INPUT_LEN": "longest prompt in tokens (default 3072); sizes the engine",
    "EDGESERVE_MAX_NEW_TOKENS": "default output length in tokens (default: the model's generation_config.json, else 512)",
    "EDGESERVE_MAX_BATCH_SIZE": "sequences generated together (default 1); each one costs KV-cache memory",
    "EDGESERVE_OUTPUT_DIR": "folder for a record of every request (default: none)",
    "EDGESERVE_HOST": "address to listen on: 127.0.0.1 (default, this machine only) or 0.0.0.0 (the network)",
    "EDGESERVE_PORT": "port (default 8000)",
    "EDGESERVE_API_KEY": "the key of client default (Authorization: Bearer <key>); 0.0.0.0 needs it or a key from edgeserve keys add",
    "EDGESERVE_TLS_CERT": "server certificate (PEM) for HTTPS",
    "EDGESERVE_TLS_KEY": "its private key (PEM)",
    "EDGESERVE_QUEUE_SIZE": "requests that may wait beyond the running batch (default 16; -1: no limit at all)",
    "EDGESERVE_QUEUE_TIMEOUT": "seconds a request may wait before it fails (default 600; 0: no limit)",
    "EDGESERVE_REQUEST_TIMEOUT": "seconds from arrival after which a request is stopped (default 0: no limit)",
    "EDGESERVE_MAX_CLIENT_REQUESTS": "requests one client may have waiting or running (default 0: no limit)",
    "EDGESERVE_CLIENT_RATE_LIMIT": "requests a minute per client: that many at once, then one every 60/N s (default 0: no limit)",
    "EDGESERVE_CACHE_DIR": "where built engines are kept (default ~/.cache/edgeserve)",
    "EDGESERVE_NATIVE_DIR": "a GPU runtime build to use instead of the one edgeserve setup made",
}
_LINE = re.compile(r"^([A-Z][A-Z0-9_]*)=(.*)$")


class ConfigError(ValueError):
    """The settings file cannot be read as KEY=VALUE lines."""


# ----------------------------------------------------------------------------------------------- paths

def config_dir() -> Path:
    override = os.environ.get("EDGESERVE_CONFIG_DIR", "").strip()
    return Path(override).expanduser() if override else Path.home() / ".config" / "edgeserve"


def config_file() -> Path:
    return config_dir() / CONFIG_FILE_NAME


def data_dir() -> Path:
    return Path.home() / ".local" / "share" / "edgeserve"


def native_root() -> Path:
    """Installed GPU runtimes: one folder per build, and ``current`` pointing at the one in use."""
    return data_dir() / "native"


def native_current() -> Path:
    return native_root() / "current"


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "edgeserve"


def build_root() -> Path:
    """The GPU runtime's build tree (Edge-LLM checkout, CMake build, kernel-pack generator)."""
    return Path.home() / ".cache" / "edgeserve" / "native-build"


# ----------------------------------------------------------------------------------------------- the file

_cache: dict[Path, tuple[tuple[int, int], dict[str, str]]] = {}


def read(path: Path | None = None) -> dict[str, str]:
    """The file's settings ({} when it does not exist). Raises ConfigError for a line it cannot read."""
    path = config_file() if path is None else path
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {}
    except OSError as e:
        raise ConfigError(f"cannot read {path}: {e.strerror or e}") from e
    stamp = (stat.st_mtime_ns, stat.st_size)
    cached = _cache.get(path)
    if cached and cached[0] == stamp:
        return dict(cached[1])
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise ConfigError(f"cannot read {path}: {e}") from e
    values: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE.match(line.removeprefix("export ").strip())
        if not match:
            raise ConfigError(f"{path}, line {number}: expected NAME=value (no spaces around '='), got {raw!r}")
        name, value = match.group(1), match.group(2).strip()
        if name not in SETTINGS:
            log.warning("%s, line %d: unknown setting %s (ignored)", path, number, name)
            continue
        values[name] = value
    _cache[path] = (stamp, values)
    return dict(values)


# Settings that are secrets (the API key, the TLS key's path): never passed on to a process edgeserve starts.
SECRET_VARIABLES = ("EDGESERVE_API_KEY", "EDGESERVE_TLS_KEY")


def child_environment(base: dict[str, str] | None = None, **extra: str) -> dict[str, str]:
    """The environment for a process edgeserve starts: ``base`` (default: this process's environment) without
    edgeserve's secrets, plus ``extra``."""
    env = {k: v for k, v in (os.environ if base is None else base).items() if k not in SECRET_VARIABLES}
    env.update(extra)
    return env


def value(name: str) -> str | None:
    """A setting from the environment, else from the settings file (empty counts as unset)."""
    env = os.environ.get(name, "").strip()
    if env:
        return env
    return read().get(name) or None


def source(name: str) -> str | None:
    """Where ``value(name)`` comes from: "environment", "config file" or None (unset)."""
    if os.environ.get(name, "").strip():
        return "environment"
    return "config file" if read().get(name) else None


def write(updates: dict[str, str | None], path: Path | None = None) -> Path:
    """Set (or, with None, remove) settings in the file, keeping its other lines and comments.

    Creates the file (from a commented template) and its folder if needed. The folder is private
    (0700) and the file 0600: it can hold the API key. The file is replaced atomically.
    """
    path = config_file() if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else _template()
    pending = dict(updates)
    out: list[str] = []
    for raw in lines:
        match = _LINE.match(raw.strip().removeprefix("export ").strip())
        commented = re.match(r"^#\s*([A-Z][A-Z0-9_]*)=", raw.strip())
        name = match.group(1) if match else (commented.group(1) if commented else None)
        if name in pending:
            new = pending.pop(name)
            if new is not None:
                out.append(f"{name}={new}")
            elif not match:  # an already commented-out line stays as it is
                out.append(raw)
            continue
        out.append(raw)
    added = [f"{name}={new}" for name, new in pending.items() if new is not None]
    if added:
        out.extend([""] if out and out[-1].strip() else [])
        out.extend(added)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".edgeserve.env.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write("\n".join(out) + "\n")
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    _cache.pop(path, None)
    return path


def _template() -> list[str]:
    lines = [
        "# edgeserve settings (edgeserve setup writes this file; edit it and restart the server).",
        "# One NAME=value per line, no quotes. Command-line flags and environment variables override these.",
        "# Keep this file private: it can hold the API key.",
        "",
    ]
    for name, meaning in SETTINGS.items():
        lines.append(f"# {name}=    ({meaning})")
    return lines


# ----------------------------------------------------------------------------------------------- clients

@dataclass(frozen=True)
class ClientTarget:
    """Where a client command (batch, check) sends requests, and with which key and CA certificate."""

    url: str
    api_key: str | None
    ca_cert: Path | None
    local: bool  # the server runs on this machine


def client_target(server: str | None = None, ca_cert: str | Path | None = None) -> ClientTarget:
    """The server to talk to: ``server`` if given, else the one this machine's settings describe.

    This machine's API key and CA certificate (from the settings file) are only ever used for this
    machine's server, never sent to another host; a key in EDGESERVE_API_KEY is the caller's choice
    and is always used.
    """
    file = read()
    env_key = os.environ.get("EDGESERVE_API_KEY", "").strip() or None
    if server:
        url = server.strip()
        local = this_machine(url)
    else:
        local = True
        scheme = "https" if (os.environ.get("EDGESERVE_TLS_CERT", "").strip() or file.get("EDGESERVE_TLS_CERT")) \
            else "http"
        host = (value("EDGESERVE_HOST") or "127.0.0.1").strip()
        if host in ("", "0.0.0.0", "::", "[::]"):
            host = "localhost"  # listening everywhere: reach it locally (the certificate names localhost)
        elif ":" in host and not host.startswith("["):
            host = f"[{host}]"
        url = f"{scheme}://{host}:{value('EDGESERVE_PORT') or 8000}"
    api_key = env_key or (file.get("EDGESERVE_API_KEY") if local else None)
    ca: Path | None = Path(ca_cert).expanduser() if ca_cert else None
    if ca is None and local:
        tls_cert = value("EDGESERVE_TLS_CERT")
        candidate = Path(tls_cert).expanduser().parent / CA_CERT_NAME if tls_cert else None
        if candidate is not None and candidate.is_file():
            ca = candidate
    return ClientTarget(url=url, api_key=api_key, ca_cert=ca, local=local)


def this_machine(url: str) -> bool:
    """Whether the host of ``url`` is this machine (exact names and addresses only)."""
    try:
        host = urllib.parse.urlsplit(url if "://" in url else f"//{url}").hostname
    except ValueError:
        return False
    if not host:
        return False
    host = host.rstrip(".").lower()
    if host == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        name = socket.gethostname().rstrip(".").lower()
        return host in (name, f"{name}.local")
    mapped = getattr(address, "ipv4_mapped", None)
    if (mapped or address).is_loopback:
        return True
    return str(mapped or address) in local_addresses()


_addresses: set[str] | None = None


def local_addresses() -> set[str]:
    """The IP addresses of this machine's network interfaces."""
    global _addresses
    if _addresses is None:
        found: set[str] = set()
        try:
            out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=5).stdout
            found.update(out.split())
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None):
                found.add(info[4][0])
        except OSError:
            pass
        _addresses = {str(ipaddress.ip_address(a.split("%")[0])) for a in found if _is_address(a.split("%")[0])}
    return set(_addresses)


def _is_address(text: str) -> bool:
    try:
        ipaddress.ip_address(text)
    except ValueError:
        return False
    return True


__all__ = [
    "ClientTarget", "ConfigError", "SETTINGS", "build_root", "client_target", "config_dir", "config_file",
    "data_dir", "default_cache_dir", "local_addresses", "native_current", "native_root", "read", "source",
    "this_machine", "value", "write",
]
