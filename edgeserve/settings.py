"""Server settings: a handful of command-line flags, each with an EDGESERVE_* fallback.

A setting comes from its flag, else the EDGESERVE_* environment variable, else the settings file
(``~/.config/edgeserve/edgeserve.env``, see edgeserve.config), else its default. Secrets and TLS paths
have no flags (EDGESERVE_API_KEY, EDGESERVE_TLS_CERT, EDGESERVE_TLS_KEY), so they never appear in
process listings or shell history; ``edgeserve setup --network`` writes them to the settings file.
"""

from __future__ import annotations

import argparse
import ipaddress
from dataclasses import dataclass, field
from pathlib import Path

from edgeserve import config

# Operating limits. Deliberately not flags: these defaults suit a single-GPU edge server.
QUEUE_SIZE = 16  # requests waiting for the GPU beyond the running batch
QUEUE_TIMEOUT_S = 600.0  # a queued request that has not started by then fails with 503
MEMORY_FLOOR_BYTES = 256 * 1024**2  # refuse new requests while system memory is below this
LOAD_HEADROOM_BYTES = 512 * 1024**2  # keep this much memory free after loading an engine
DRAIN_TIMEOUT_S = 60.0  # how long shutdown waits for in-flight requests
SSE_KEEPALIVE_S = 15.0
MAX_BODY_BYTES = 4 * 1024**2
BODY_TIMEOUT_S = 60.0  # a request body must arrive within this (slow uploads hold a connection)
HEAD_TIMEOUT_S = 30.0  # a connection must send a complete request head within this (also between requests)
WRITE_STALL_S = 60.0  # a client that stops reading a response for this long is dropped
CLOSE_GRACE_S = 60.0  # a closing connection is aborted if it has not closed within this (TLS: peer never answers)
MAX_CONNECTIONS = 128  # server-wide: with more open connections or running requests uvicorn answers 503
MAX_CONNECTIONS_PER_CLIENT = 32  # per client address: further connections are dropped at once
MAX_WORKER_RESTARTS = 5  # within RESTART_WINDOW_S, then the server reports itself failed
RESTART_WINDOW_S = 600.0
DEFAULT_MAX_NEW_TOKENS = 512

@dataclass(frozen=True)
class Settings:
    model: Path
    max_input_len: int = 3072
    max_new_tokens: int | None = None  # None: model's generation_config.json, else DEFAULT_MAX_NEW_TOKENS
    max_batch_size: int = 1
    output_dir: Path | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    cache_dir: Path = field(default_factory=config.default_cache_dir)
    api_key: str | None = None
    tls_cert: Path | None = None
    tls_key: Path | None = None
    native_dir: Path = field(default_factory=config.native_current)

    @property
    def tls(self) -> bool:
        return self.tls_cert is not None and self.tls_key is not None

    def validate(self) -> None:
        """Reject unsafe or inconsistent settings with a message that says how to fix them."""
        if self.max_input_len < 1 or self.max_batch_size < 1:
            raise ValueError("--max-input-len and --max-batch-size must be at least 1")
        if self.max_new_tokens is not None and self.max_new_tokens < 1:
            raise ValueError("--max-new-tokens must be at least 1")
        if self.max_batch_size > 16:
            raise ValueError("--max-batch-size above 16 is not supported by Edge-LLM")
        if not 0 < self.port < 65536:
            raise ValueError(f"--port must be between 1 and 65535, not {self.port}")
        if (self.tls_cert is None) != (self.tls_key is None):
            raise ValueError("set both EDGESERVE_TLS_CERT and EDGESERVE_TLS_KEY, or neither")
        if not is_loopback(self.host) and not self.api_key:
            raise ValueError(
                f"refusing to listen on {self.host or 'every interface'} without an API key: set EDGESERVE_API_KEY "
                "(edgeserve setup --network creates one) or use --host 127.0.0.1"
            )


def is_loopback(host: str) -> bool:
    """Whether listening on ``host`` is reachable from this machine only ("" means every interface)."""
    if host.lower() == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)  # ::ffff:127.0.0.1
    return (mapped or address).is_loopback


# flag -> (setting, type, default); the defaults the help texts state.
_ENGINE_FLAGS = {
    "model": ("EDGESERVE_MODEL", Path, None),
    "max_input_len": ("EDGESERVE_MAX_INPUT_LEN", int, 3072),
    "max_new_tokens": ("EDGESERVE_MAX_NEW_TOKENS", int, None),
    "max_batch_size": ("EDGESERVE_MAX_BATCH_SIZE", int, 1),
    "cache_dir": ("EDGESERVE_CACHE_DIR", Path, None),  # None: config.default_cache_dir()
}
_SERVER_FLAGS = {
    "output_dir": ("EDGESERVE_OUTPUT_DIR", Path, None),
    "host": ("EDGESERVE_HOST", str, "127.0.0.1"),
    "port": ("EDGESERVE_PORT", int, 8000),
}


def add_engine_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags shared by serve / build / info (model and engine limits)."""
    parser.add_argument("--model", type=Path, metavar="DIR",
                        help="model folder: a TensorRT Edge-LLM export (the llm/ folder tensorrt-edgellm-export "
                             "writes) [EDGESERVE_MODEL]")
    parser.add_argument("--max-input-len", type=int, metavar="N",
                        help="longest prompt in tokens the engine accepts (default 3072) [EDGESERVE_MAX_INPUT_LEN]")
    parser.add_argument("--max-new-tokens", type=int, metavar="N",
                        help="default output length in tokens; with --max-input-len it sizes the engine (default: "
                             "the model's generation_config.json, else 512) [EDGESERVE_MAX_NEW_TOKENS]")
    parser.add_argument("--max-batch-size", type=int, metavar="N",
                        help="sequences generated together; raise only with spare memory (default 1) "
                             "[EDGESERVE_MAX_BATCH_SIZE]")
    parser.add_argument("--cache-dir", type=Path, metavar="DIR",
                        help="where built TensorRT engines are kept (default ~/.cache/edgeserve) [EDGESERVE_CACHE_DIR]")


def add_server_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, metavar="DIR",
                        help="save a structured record of every request here (default: none) [EDGESERVE_OUTPUT_DIR]")
    parser.add_argument("--host", metavar="ADDRESS",
                        help="address to listen on; 0.0.0.0 for other machines (needs EDGESERVE_API_KEY) "
                             "(default 127.0.0.1) [EDGESERVE_HOST]")
    parser.add_argument("--port", type=int, metavar="N", help="port (default 8000) [EDGESERVE_PORT]")


def resolve(args: argparse.Namespace) -> dict[str, tuple[object, str]]:
    """Each setting's value and where it came from: "flag", "environment", "config file" or "default"."""
    out: dict[str, tuple[object, str]] = {}
    flags = dict(_ENGINE_FLAGS)
    if hasattr(args, "host"):
        flags.update(_SERVER_FLAGS)
    for attr, (name, kind, default) in flags.items():
        given = getattr(args, attr, None)
        if given is not None:
            out[attr] = (given, "flag")
            continue
        text = config.value(name)
        if text is None:
            out[attr] = (default, "default")
            continue
        origin = config.source(name) or "environment"
        if kind is int:
            try:
                out[attr] = (int(text), origin)
            except ValueError:
                where = config.config_file() if origin == "config file" else "the environment"
                raise ValueError(f"{name} (in {where}) must be a whole number, not {text!r}") from None
        else:
            out[attr] = (kind(text), origin)
    return out


def from_args(args: argparse.Namespace) -> Settings:
    values = {attr: v for attr, (v, _) in resolve(args).items()}
    if values["model"] is None:
        raise ValueError("no model given: pass --model <folder> or set EDGESERVE_MODEL (edgeserve setup --model)")
    tls_cert, tls_key = config.value("EDGESERVE_TLS_CERT"), config.value("EDGESERVE_TLS_KEY")
    output_dir = values.get("output_dir")
    native = config.value("EDGESERVE_NATIVE_DIR")
    settings = Settings(
        model=Path(values["model"]).expanduser().resolve(),
        max_input_len=values["max_input_len"],
        max_new_tokens=values["max_new_tokens"],
        max_batch_size=values["max_batch_size"],
        output_dir=Path(output_dir).expanduser().resolve() if output_dir else None,
        host=values.get("host", "127.0.0.1"),
        port=values.get("port", 8000),
        cache_dir=Path(values["cache_dir"] or config.default_cache_dir()).expanduser().resolve(),
        api_key=config.value("EDGESERVE_API_KEY"),
        tls_cert=Path(tls_cert).expanduser() if tls_cert else None,
        tls_key=Path(tls_key).expanduser() if tls_key else None,
        native_dir=Path(native).expanduser().resolve() if native else config.native_current(),
    )
    settings.validate()
    return settings
