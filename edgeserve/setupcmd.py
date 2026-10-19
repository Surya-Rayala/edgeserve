"""``edgeserve setup``: prepare this machine to serve models.

1. Checks what building the GPU runtime needs (every missing piece is reported at once).
2. Builds and installs the GPU runtime when it is missing or out of date (see edgeserve.nativebuild),
   and checks its attention kernels against a CPU reference.
3. Writes the settings file (``~/.config/edgeserve/edgeserve.env``); ``--model`` records the model
   to serve; ``--network`` makes the server reachable from other machines: listen on every
   interface, require an API key, serve HTTPS with a certificate from a private CA.

Running it again is safe and quick: nothing that is up to date is redone, an existing API key and
CA are never replaced, and the server certificate is renewed only when the machine's names or
addresses changed or it is about to expire.
"""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

from edgeserve import config, nativebuild, tls

_MIN_FREE_BYTES = 3 * 1024**3  # build tree ~1.5 GB plus compiler temporaries


def run(network: bool = False, model: Path | None = None) -> int:
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        print("edgeserve setup: run this as the user who will run the server, not as root (it builds into that "
              "user's home folder)", file=sys.stderr)
        return 2
    if model is not None:
        problem = _check_model(model)
        if problem:
            print(f"edgeserve setup: {problem}", file=sys.stderr)
            return 2

    # 1-2: the GPU runtime
    state = nativebuild.status()
    changes = nativebuild.toolchain_changes(state.info) if state.ok and state.info else []
    if state.ok and not changes:
        _say(f"GPU runtime: up to date ({state.root})")
    else:
        _say(f"GPU runtime: {', '.join(changes) or state.reason}; building it")
        problems = prerequisites()
        if problems:
            print("edgeserve setup: cannot build the GPU runtime yet:", file=sys.stderr)
            for item in problems:
                print(f"  - {item}", file=sys.stderr)
            print(_install_hint(problems), file=sys.stderr)
            return 2
        try:
            root = nativebuild.build(progress=_say)
        except nativebuild.NativeBuildError as e:
            print(f"edgeserve setup: {e}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("edgeserve setup: interrupted; run it again to continue the build", file=sys.stderr)
            return 130
        _say(f"GPU runtime: installed in {root}")
        if not nativebuild.self_test(root, progress=_say):
            print("edgeserve setup: the attention-kernel check FAILED on this machine; outputs may be wrong. Please "
                  "report it with the output above (see docs/PLATFORMS.md, 'Checking a platform').", file=sys.stderr)
            return 1
        _say("GPU runtime: attention kernels match the reference")

    # 3: settings
    current = config.read()
    updates: dict[str, str | None] = {}
    if model is not None:
        updates["EDGESERVE_MODEL"] = str(model.expanduser().resolve())
    key_created, outcome = False, None
    if network:
        if shutil.which("openssl") is None:
            print("edgeserve setup: --network needs openssl (sudo apt install openssl)", file=sys.stderr)
            return 2
        folder = config.config_dir()
        try:
            outcome = tls.ensure(folder)
        except tls.TLSError as e:
            print(f"edgeserve setup: {e}", file=sys.stderr)
            return 1
        key = current.get("EDGESERVE_API_KEY")
        if not key:
            key, key_created = tls.new_api_key(), True
        host = current.get("EDGESERVE_HOST") or ""
        updates.update({
            "EDGESERVE_HOST": host if host and host not in ("127.0.0.1", "localhost", "::1") else "0.0.0.0",
            "EDGESERVE_API_KEY": key,
            "EDGESERVE_TLS_CERT": str(folder / tls.SERVER_CERT),
            "EDGESERVE_TLS_KEY": str(folder / tls.SERVER_KEY),
        })
    path = config.write(updates)
    _report(path, network, key_created, outcome)
    return 0


# ----------------------------------------------------------------------------------------------- checks

_PACKAGES = ("git", "cmake", "build-essential", "python3-venv", "openssl")


def prerequisites() -> list[str]:
    """Everything missing for building the GPU runtime (empty: ready)."""
    problems = []
    if shutil.which("git") is None:
        problems.append("git is not installed (package git)")
    cmake = shutil.which("cmake")
    if cmake is None:
        problems.append("CMake 3.20 or newer is not installed (package cmake)")
    else:
        out = subprocess.run([cmake, "--version"], capture_output=True, text=True).stdout
        match = re.search(r"(\d+)\.(\d+)", out)
        if match and (int(match.group(1)), int(match.group(2))) < (3, 20):
            problems.append(f"CMake {match.group(0)} is too old, 3.20 or newer is needed (package cmake)")
    if shutil.which("make") is None or (shutil.which("c++") is None and shutil.which("g++") is None):
        problems.append("a C++ compiler and make are not installed (package build-essential)")
    if shutil.which("python3") is None or subprocess.run(["python3", "-c", "import venv, ensurepip"],
                                                         capture_output=True).returncode != 0:
        problems.append("python3 cannot create virtual environments (package python3-venv)")
    nvcc = shutil.which("nvcc") or ("/usr/local/cuda/bin/nvcc" if os.access("/usr/local/cuda/bin/nvcc", os.X_OK)
                                    else None)
    if nvcc is None:
        problems.append("the CUDA toolkit (nvcc) is not installed; Jetson: sudo apt install nvidia-jetpack; x86: "
                        "install CUDA 12.x or 13.x from NVIDIA")
    if nativebuild.tensorrt_header() is None:
        problems.append("TensorRT's development headers (NvInferVersion.h) are not installed; Jetson: part of "
                        "nvidia-jetpack; x86: install TensorRT 10.x (or set TRT_PACKAGE_DIR to its folder)")
    probe = config.build_root().parent
    probe.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(probe).free
    if free < _MIN_FREE_BYTES:
        problems.append(f"only {free / 1024**3:.1f} GB free in {probe}; the build needs about 3 GB")
    return problems


def _install_hint(problems: list[str]) -> str:
    packages = [p for p in _PACKAGES if any(f"package {p}" in item for item in problems)]
    lines = []
    if packages:
        lines.append(f"  install with: sudo apt install {' '.join(packages)}")
    lines.append("  then run edgeserve setup again (docs/PLATFORMS.md lists what each platform needs)")
    return "\n".join(lines)


def _check_model(model: Path) -> str | None:
    from edgeserve.model import ModelError, load_model

    path = model.expanduser()
    if not path.is_dir():
        return f"--model {model}: no such folder"
    try:
        load_model(path)
    except ModelError as e:
        return f"--model {model}: {e}"
    return None


# ----------------------------------------------------------------------------------------------- report

def _report(path: Path, network: bool, key_created: bool, outcome: tls.Outcome | None) -> None:
    settings = config.read(path)
    model = settings.get("EDGESERVE_MODEL")
    print()
    print(f"Settings: {path}")
    print(f"  model: {model or '(none yet: pass --model to edgeserve serve, or edgeserve setup --model <folder>)'}")
    if not network:
        print("  serving on this machine only (http://127.0.0.1:8000); edgeserve setup --network opens it to the "
              "network")
        print("\nNext: edgeserve serve" + ("" if model else " --model <model folder>"))
        return
    folder = config.config_dir()
    port = settings.get("EDGESERVE_PORT") or "8000"
    if outcome is not None:
        if outcome.ca_created:
            print(f"  created a private CA: {folder / tls.CA_CERT} (give this file to every client)")
        if outcome.server_issued and not outcome.ca_created:
            print(f"  renewed the server certificate: {outcome.reason}")
    print(f"  listening on every interface, port {port}, HTTPS, API key required")
    if key_created:
        print(f"  API key (also in the settings file): {settings.get('EDGESERVE_API_KEY')}")
    else:
        print("  API key: EDGESERVE_API_KEY in the settings file (unchanged)")
    print(f"  CA certificate for clients: {folder / tls.CA_CERT}")
    subnet = tls.client_subnet() or "<your-network>/<prefix>"
    print("\nFirewall: allow only machines on your network to reach the port, for example with ufw:")
    if not Path("/etc/ufw/ufw.conf").exists() or "ENABLED=yes" not in Path("/etc/ufw/ufw.conf").read_text():
        print("  sudo ufw allow OpenSSH        # first, so SSH (and remote IDE sessions) keep working")
        print(f"  sudo ufw allow from {subnet} to any port {port} proto tcp")
        print("  sudo ufw enable               # ufw is not enabled on this machine yet")
    else:
        print(f"  sudo ufw allow from {subnet} to any port {port} proto tcp")
    interfaces = tls.global_interfaces()
    address = str(interfaces[0].ip) if interfaces else f"{socket.gethostname()}.local"
    print(f"\nNext: restart edgeserve serve; then, on another machine, with {tls.CA_CERT} and the key:")
    print(f"  curl --cacert {tls.CA_CERT} -H \"Authorization: Bearer $EDGESERVE_API_KEY\" "
          f"https://{address}:{port}/v1/models")


def _say(text: str) -> None:
    print(f"[edgeserve setup] {text}", flush=True)


__all__ = ["prerequisites", "run"]
