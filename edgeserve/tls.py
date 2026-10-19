"""The API key and the HTTPS certificates for serving on the network (``edgeserve setup --network``).

A small private certificate authority (CA) is created once per machine; it signs a server
certificate valid for the machine's host name, ``<name>.local``, ``localhost`` and its IP
addresses. Clients trust the CA file (``edgeserve-ca.crt``), so the server certificate can be
renewed without touching them. Uses the ``openssl`` command (installed on every Linux system).
"""

from __future__ import annotations

import datetime
import ipaddress
import os
import re
import secrets
import socket
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

CA_CERT, CA_KEY = "edgeserve-ca.crt", "edgeserve-ca.key"
SERVER_CERT, SERVER_KEY = "server.crt", "server.key"
SERVER_DAYS = 825  # the longest validity Apple platforms accept for a TLS server certificate
RENEW_DAYS = 30


class TLSError(RuntimeError):
    """A certificate could not be made; the message carries openssl's own error."""


def new_api_key() -> str:
    return "es-" + secrets.token_urlsafe(32)


@dataclass(frozen=True)
class Names:
    """What the server certificate must cover."""

    dns: tuple[str, ...]
    ips: tuple[str, ...]


def machine_names() -> Names:
    host = socket.gethostname()
    dns = tuple(dict.fromkeys(n for n in (host, f"{host}.local", "localhost") if n))
    ips = tuple(dict.fromkeys(["127.0.0.1", *(str(i.ip) for i in global_interfaces())]))
    return Names(dns=dns, ips=ips)


def global_interfaces() -> list[ipaddress.IPv4Interface]:
    """This machine's global IPv4 addresses with their subnets (from ``ip -4 addr``)."""
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show", "scope", "global"], capture_output=True, text=True,
                             timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    found = []
    for line in out.splitlines():
        match = re.search(r"\binet (\d+\.\d+\.\d+\.\d+/\d+)", line)
        if match:
            found.append((line.split()[1], ipaddress.ip_interface(match.group(1))))
    default = default_route_device()
    found.sort(key=lambda item: item[0] != default)  # the interface the default route uses first
    return [interface for _, interface in found]


def default_route_device() -> str | None:
    try:
        out = subprocess.run(["ip", "-4", "route", "show", "default"], capture_output=True, text=True,
                             timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"\bdev (\S+)", out)
    return match.group(1) if match else None


def client_subnet() -> str | None:
    """The subnet of the interface the default route uses (where the clients most likely are)."""
    interfaces = global_interfaces()
    return str(interfaces[0].network) if interfaces else None


@dataclass(frozen=True)
class Outcome:
    ca_created: bool
    server_issued: bool
    reason: str  # why the server certificate was (re)issued


def ensure(folder: Path, names: Names | None = None) -> Outcome:
    """Make sure ``folder`` holds a CA and a current server certificate for ``names``; never replaces a CA
    whose key is still here."""
    names = machine_names() if names is None else names
    folder.mkdir(parents=True, exist_ok=True)
    os.chmod(folder, 0o700)
    ca_crt, ca_key = folder / CA_CERT, folder / CA_KEY
    srv_crt, srv_key = folder / SERVER_CERT, folder / SERVER_KEY
    ca_created = False
    reason = _renewal_reason(srv_crt, srv_key, ca_crt, names)
    if not ca_crt.exists() or (reason and not ca_key.exists()):
        # No CA yet, or the server certificate must be signed again but the CA key is gone: a new CA (clients
        # must then be given the new edgeserve-ca.crt).
        _make_ca(folder, names)
        ca_created = True
        reason = reason or "no server certificate yet"
    if reason or ca_created:
        _make_server(folder, names)
    return Outcome(ca_created=ca_created, server_issued=bool(reason or ca_created), reason=reason)


def _renewal_reason(crt: Path, key: Path, ca: Path, names: Names) -> str:
    if not crt.exists() or not key.exists():
        return "no server certificate yet"
    try:
        text = _openssl(["x509", "-in", str(crt), "-noout", "-enddate", "-ext", "subjectAltName"])
    except TLSError:
        return "the server certificate cannot be read"
    covered = set(re.findall(r"DNS:([^,\s]+)", text)) | set(re.findall(r"IP Address:([^,\s]+)", text))
    missing = [n for n in (*names.dns, *names.ips) if n not in covered]
    if missing:
        return "it does not cover " + ", ".join(missing)
    match = re.search(r"notAfter=(.+)", text)
    if match:
        try:
            expires = datetime.datetime.strptime(match.group(1).strip(), "%b %d %H:%M:%S %Y %Z")
        except ValueError:
            expires = None
        now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)  # openssl prints GMT
        if expires and expires - now < datetime.timedelta(days=RENEW_DAYS):
            return f"it expires on {expires:%Y-%m-%d}"
    if ca.exists():
        try:
            _openssl(["verify", "-CAfile", str(ca), str(crt)])
        except TLSError:
            return "it is not signed by the current CA"
    return ""


def _make_ca(folder: Path, names: Names) -> None:
    for stale in (SERVER_CERT, SERVER_KEY):
        (folder / stale).unlink(missing_ok=True)
    subject = f"/CN=edgeserve CA ({names.dns[0] if names.dns else 'edgeserve'})"
    _openssl(["req", "-x509", "-newkey", "rsa:3072", "-sha256", "-days", "3650", "-nodes",
              "-keyout", str(folder / CA_KEY), "-out", str(folder / CA_CERT), "-subj", subject,
              "-addext", "basicConstraints=critical,CA:TRUE", "-addext", "keyUsage=critical,keyCertSign,cRLSign"])
    os.chmod(folder / CA_KEY, 0o600)
    os.chmod(folder / CA_CERT, 0o644)


def _make_server(folder: Path, names: Names) -> None:
    san = ",".join([*(f"DNS:{n}" for n in names.dns), *(f"IP:{i}" for i in names.ips)])
    with tempfile.TemporaryDirectory(dir=folder, prefix=".issue-") as tmp:
        work = Path(tmp)
        (work / "server.ext").write_text(
            f"subjectAltName={san}\nbasicConstraints=CA:FALSE\nkeyUsage=digitalSignature,keyEncipherment\n"
            "extendedKeyUsage=serverAuth\nsubjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid,issuer\n")
        _openssl(["req", "-new", "-newkey", "rsa:3072", "-sha256", "-nodes", "-keyout", str(work / SERVER_KEY),
                  "-out", str(work / "server.csr"), "-subj", f"/CN={names.dns[0] if names.dns else 'localhost'}"])
        _openssl(["x509", "-req", "-in", str(work / "server.csr"), "-CA", str(folder / CA_CERT),
                  "-CAkey", str(folder / CA_KEY), "-CAcreateserial", "-CAserial", str(work / "ca.srl"),
                  "-out", str(work / SERVER_CERT), "-days", str(SERVER_DAYS), "-sha256",
                  "-extfile", str(work / "server.ext")])
        os.chmod(work / SERVER_KEY, 0o600)
        os.chmod(work / SERVER_CERT, 0o644)
        os.replace(work / SERVER_KEY, folder / SERVER_KEY)
        os.replace(work / SERVER_CERT, folder / SERVER_CERT)


def _openssl(args: list[str]) -> str:
    """Run openssl; its progress chatter is shown only when it fails."""
    old = os.umask(0o077)
    try:
        result = subprocess.run(["openssl", *args], capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        raise TLSError("openssl is not installed (sudo apt install openssl)") from None
    finally:
        os.umask(old)
    if result.returncode != 0:
        raise TLSError(f"openssl {args[0]} failed: {(result.stderr or result.stdout).strip()[-800:]}")
    return result.stdout


__all__ = ["CA_CERT", "CA_KEY", "Names", "Outcome", "SERVER_CERT", "SERVER_KEY", "TLSError", "client_subnet",
           "default_route_device", "ensure", "global_interfaces", "machine_names", "new_api_key"]
