"""API keys per client: ``edgeserve keys add | list | remove``.

Each client (a person, a machine, a program) gets its own key, so one can be revoked without changing the
others, and every record says which client sent the request. The keys live in
``~/.config/edgeserve/client-keys.json`` (mode 0600) as SHA-256 hashes only: a key is shown once, when it is made,
and cannot be read back. The server rereads the file when it changes, so an added or removed key works, or stops
working, at once, without a restart. The key in ``EDGESERVE_API_KEY`` keeps working too, as client "default".
"""

from __future__ import annotations

import contextlib
import datetime
import fcntl
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import tempfile
import time
from pathlib import Path

from edgeserve import config

FILE_NAME = "client-keys.json"
DEFAULT_CLIENT = "default"  # the client of EDGESERVE_API_KEY
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
log = logging.getLogger("edgeserve")


class KeysError(ValueError):
    """A client name or the keys file that cannot be used; the message says why."""


def path() -> Path:
    return config.config_dir() / FILE_NAME


def new_key() -> str:
    return "es-" + secrets.token_urlsafe(32)  # 256 random bits


def digest(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def check_name(name: str) -> None:
    if not _NAME.fullmatch(name):
        raise KeysError(f"client names are 1-64 letters, digits, '.', '_' or '-', starting with a letter or digit, "
                        f"not {name!r}")
    if name == DEFAULT_CLIENT:
        raise KeysError(f"{DEFAULT_CLIENT!r} is the client of EDGESERVE_API_KEY (edgeserve setup --network); choose "
                        "another name")


def load(file: Path | None = None) -> dict[str, dict]:
    """``{name: {"sha256": ..., "created": ...}}``, or {} when there is no keys file."""
    file = path() if file is None else file
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        raise KeysError(f"cannot read {file}: {e}") from e
    keys = data.get("keys") if isinstance(data, dict) else None
    if not isinstance(keys, dict) or not all(isinstance(entry, dict) and isinstance(entry.get("sha256"), str)
                                             for entry in keys.values()):
        raise KeysError(f"{file} is not a keys file written by edgeserve keys")
    return keys


def add(name: str, file: Path | None = None) -> str:
    """Make a key for a new client and store its hash; returns the key (the only time it can be seen)."""
    check_name(name)
    file = path() if file is None else file
    with _locked(file):
        keys = load(file)
        if name in keys:
            raise KeysError(f"client {name!r} already has a key; to replace it, run edgeserve keys remove {name} "
                            "first")
        key = new_key()
        keys[name] = {"sha256": digest(key),
                      "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
        _save(keys, file)
    return key


def remove(name: str, file: Path | None = None) -> bool:
    """Delete a client's key (False: there was none)."""
    file = path() if file is None else file
    with _locked(file):
        keys = load(file)
        if keys.pop(name, None) is None:
            return False
        _save(keys, file)
    return True


@contextlib.contextmanager
def _locked(file: Path):
    """One change of the keys file at a time (two commands at once would otherwise lose one of the changes)."""
    file.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(file.parent, 0o700)
    fd = os.open(file.parent / f".{FILE_NAME}.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _save(keys: dict, file: Path) -> None:
    """Write the keys file atomically and durably, readable by this user only."""
    fd, tmp = tempfile.mkstemp(dir=file.parent, prefix=f".{FILE_NAME}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"keys": keys}, f, indent=1, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, file)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    directory = os.open(file.parent, os.O_RDONLY)
    try:  # so that a removal survives a power cut
        os.fsync(directory)
    finally:
        os.close(directory)


class KeyStore:
    """The keys a server accepts: ``EDGESERVE_API_KEY`` (client "default") and the keys file, reread whenever it
    changes. A key is required on a server on the network (``always_required``), and on any server once it has
    seen a client key, until it restarts: removing the last key, or breaking the file, never lets everyone in."""

    def __init__(self, default_key: str | None, always_required: bool, file: Path | None = None) -> None:
        self._default = default_key or None
        self._always = always_required
        self._file = path() if file is None else file
        self._stamp: tuple | None = None
        self._recent = False  # the file was written just before it was read: the next write may keep its stamp
        self._hashes: dict[str, str] = {}  # sha256 -> client name
        self.refresh()

    def refresh(self) -> None:
        try:
            st = self._file.stat()
            stamp: tuple | None = (st.st_mtime_ns, st.st_size, st.st_ino)
        except FileNotFoundError:
            stamp = None
        except OSError as e:
            if self._stamp != "unreadable":
                log.error("cannot read %s (%s); keeping the client keys read before", self._file, e)
            self._stamp, self._always = "unreadable", True
            return
        if stamp == self._stamp and not self._recent:
            return
        self._stamp = stamp
        # A key replaced within the file system's time resolution gives the same size and often the same inode and
        # time: read the file again while it is that fresh.
        self._recent = stamp is not None and time.time_ns() - stamp[0] < 2_000_000_000
        try:
            keys = load(self._file) if stamp is not None else {}
        except KeysError as e:
            # Read once per change. Only the keys still written in the broken file keep working, so a hand edit
            # that deleted a client (and broke the file) still shuts that client out.
            try:
                text = self._file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            self._hashes = {h: name for h, name in self._hashes.items() if h in text}
            self._always = True
            log.error("%s; until it is fixed, only the keys still written in it work", e)
            return
        self._hashes = {entry["sha256"]: name for name, entry in keys.items()}
        if self._hashes:
            self._always = True

    @property
    def required(self) -> bool:
        self.refresh()
        return self._always or self._default is not None or bool(self._hashes)

    def client(self, key: str) -> str | None:
        """The client a presented key belongs to, or None (every stored key is compared, in constant time)."""
        self.refresh()
        found = None
        if self._default is not None and hmac.compare_digest(key.encode(), self._default.encode()):
            found = DEFAULT_CLIENT
        hashed = digest(key)
        for stored, name in self._hashes.items():
            if hmac.compare_digest(hashed, stored) and found is None:
                found = name
        return found


def run(action: str, name: str | None) -> int:
    """``edgeserve keys add NAME | list | remove NAME``."""
    import sys

    try:
        if action == "list":
            keys = load()
            default = config.value("EDGESERVE_API_KEY")
            if not keys and not default:
                print("no keys: edgeserve setup --network makes the default one; edgeserve keys add <name> one per "
                      "client")
                return 0
            if default:
                print(f"{DEFAULT_CLIENT:24} EDGESERVE_API_KEY in {config.config_file()}")
            for client, entry in sorted(keys.items()):
                print(f"{client:24} added {entry.get('created', '?')}")
            return 0
        assert name is not None
        if action == "add":
            key = add(name)
            print(f"API key for client {name} (shown only now: edgeserve keeps only its SHA-256 hash):\n\n  {key}\n")
            print("The client sends it as 'Authorization: Bearer <key>'; it works at once, also on a running server.")
            print(f"Revoke it with: edgeserve keys remove {name}")
            return 0
        if name == DEFAULT_CLIENT:
            print(f"edgeserve keys: the {DEFAULT_CLIENT} key is EDGESERVE_API_KEY in {config.config_file()}; delete "
                  "that line (and restart the server) to remove it", file=sys.stderr)
            return 2
        if not remove(name):
            print(f"edgeserve keys: no client named {name!r} (edgeserve keys list shows them)", file=sys.stderr)
            return 1
        print(f"removed the key of client {name}: it stops working at once, also on a running server")
        if not load() and not config.value("EDGESERVE_API_KEY"):
            print("no keys are left: a running server now refuses every request; restart it to serve this machine "
                  "without keys")
        return 0
    except KeysError as e:
        print(f"edgeserve keys: {e}", file=sys.stderr)
        return 2


__all__ = ["DEFAULT_CLIENT", "KeyStore", "KeysError", "add", "load", "path", "remove", "run"]
